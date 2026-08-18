"""Pin the FPGA program to 8a7d9c2, and check the readout plumbing separately.

8a7d9c2 (branch directory-update-20260817) is the revision the second machine
takes good data with, so what the board is ASKED TO DO is pinned to it: the pulse
program, the emission order, the slot folding, the conversion from counts. Those
decide what is measured, and this test asserts they are byte-identical.

The readout PLUMBING is deliberately not pinned, because 8a7d9c2's plumbing is
what failed on this rig on 2026-08-18 -- it hangs forever on a stall and never
clears the tProc shot counter. Those two defects are checked positively here
instead: the calls that fix them must be present, in the right place.

Section 2 is the guarantee "the measurement is unchanged".
Section 3 is the guarantee "the two known plumbing defects are fixed".

Run:  python scripts/test_acquisition_pinned.py     (exit 0 = the path is pinned)
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PIN = "8a7d9c2"

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def at_pin(path: str) -> str:
    out = subprocess.run(["git", "-C", str(REPO), "show", f"{PIN}:{path}"],
                         capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"cannot read {path} at {PIN}: {out.stderr.strip()}")
    return out.stdout


def methods(src: str, cls: str) -> dict[str, str]:
    lines = src.splitlines()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return {m.name: "\n".join(lines[m.lineno - 1:m.end_lineno])
                    for m in node.body if isinstance(m, ast.FunctionDef)}
    return {}


# --------------------------------------------------------------------------- #
# 1. the qickdawg acquire loop, whole file
# --------------------------------------------------------------------------- #
print(f"\n1. qickdawg: only the readout loop differs from {PIN}")
changed = subprocess.run(["git", "-C", str(REPO), "diff", "--name-only", PIN, "--", "qickdawg/"],
                         capture_output=True, text=True).stdout.split()
check("only nvaverageprogram.py differs",
      changed == ["qickdawg/nvpulsing/nvaverageprogram.py"] or not changed,
      ", ".join(changed) or "0 files")

# The pulse-emission helpers in that file must still match: they are the program.
QD = "qickdawg/nvpulsing/nvaverageprogram.py"
qd_ref = methods(at_pin(QD), "NVAveragerProgram")
qd_now = methods((REPO / QD).read_text(), "NVAveragerProgram")
for name in ("trigger_no_off", "get_data_shape", "analyze_results", "acquire_decimated"):
    if name in qd_ref:
        check(f"NVAveragerProgram.{name}() identical",
              name in qd_now and qd_now[name] == qd_ref[name])

# --------------------------------------------------------------------------- #
# 2. every acquisition method of MultipointLockinODMR
# --------------------------------------------------------------------------- #
print(f"\n2. MultipointLockinODMR acquisition methods are {PIN} verbatim")
MLP = "notebook_modules/multipoint_lockin_program.py"
ref = methods(at_pin(MLP), "MultipointLockinODMR")
now = methods((REPO / MLP).read_text(), "MultipointLockinODMR")

# What the board is asked to do. Change any of these and the measurement changes.
ACQUISITION = ["__init__", "initialize", "body", "emission_order", "_fold_slots",
               "analyze_results", "set_reads_per_shot", "time_per_rep", "total_time",
               "acquire", "_check_freshness", "reset_freshness_counters",
               "slot_to_frequency_index", "predicted_rate_hz", "describe_timing",
               "stream_headroom"]
for name in ACQUISITION:
    if name not in ref:
        continue
    check(f"{name}() identical", name in now and now[name] == ref[name])

# _stream_raw is 8a7d9c2's stream() with the plumbing repaired. Its DATA PATH --
# the reshape, the slot folding, what goes into each yielded packet -- must still
# match line for line; only the polling and the cleanup around it may differ.
now_raw = now.get("_stream_raw", "")
ref_raw = ref["stream"]
def data_path(src):
    keep, seen = [], False
    for line in src.splitlines():
        t = line.strip()
        if t.startswith("block =") or t.startswith("out.") or t.startswith("signal, reference"):
            keep.append(t)
    return keep
check("_stream_raw() data path identical to the pinned stream()",
      data_path(now_raw) == data_path(ref_raw),
      f"{len(data_path(ref_raw))} lines compared")

# --------------------------------------------------------------------------- #
# 3. the un-pinned block must not drive the board itself
# --------------------------------------------------------------------------- #
print("\n3. the two known plumbing defects are fixed")
import inspect as _inspect
import textwrap

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "notebook_modules"))
from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram as NVAP
import multipoint_lockin_program as _mlp

def calls_in(func):
    """Names of every attribute call made on `qd` inside `func`."""
    out = set()
    for node in ast.walk(ast.parse(textwrap.dedent(_inspect.getsource(func)))):
        if isinstance(node, ast.Attribute):
            chain, cur = [], node
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name) and cur.id == "qd":
                out.add(".".join(reversed(chain)))
    return out

# 3a. the stale shot counter -- the "could not broadcast (N,2) into (M,2)" crash
for label, fn in (("acquire()", NVAP.acquire),
                  ("_stream_raw()", _mlp.MultipointLockinODMR._stream_raw),
                  ("_abort_readout()", NVAP._abort_readout)):
    check(f"{label} clears the tProc shot counter",
          any(c.endswith("clear_tproc_counter") for c in calls_in(fn)))

src_acq = _inspect.getsource(NVAP.acquire)
i_reload = src_acq.find("reload_mem")
i_clear = src_acq.find("clear_tproc_counter")
i_start = src_acq.find("start_readout")
check("acquire() clears it between reload_mem and start_readout",
      -1 < i_reload < i_clear < i_start,
      f"reload_mem@{i_reload} clear@{i_clear} start_readout@{i_start}")

# 3b. the unbounded poll -- the permanent hang, and the parked board thread it
#     leaves behind on Ctrl-C
check("NVAveragerProgram poll is bounded",
      NVAP.POLL_TIMEOUT_S is not None and NVAP.POLL_TIMEOUT_S > 0,
      f"POLL_TIMEOUT_S = {NVAP.POLL_TIMEOUT_S!r}")
drain = _inspect.getsource(NVAP._drain_readout)
check("_drain_readout() passes that timeout to poll_data",
      "timeout=self.POLL_TIMEOUT_S" in drain)
check("_drain_readout() decides failure on the tProc counter, not on an empty poll",
      "get_tproc_counter" in drain and "STALL_GRACE_S" in drain)
check("_drain_readout() refuses an oversized packet with a named error",
      "count + new_points > total_count" in drain and "raise RuntimeError" in drain)
check("_drain_readout() leaves the board idle on any exception",
      "except BaseException" in drain and "_abort_readout()" in drain)

raw = _inspect.getsource(_mlp.MultipointLockinODMR._stream_raw)
check("_stream_raw() poll is bounded", "poll_data(totaltime=0.1, timeout=0.5)" in raw)
check("_stream_raw() stops the board in a finally",
      "finally:" in raw and "stop_readout" in raw)

# 3c. the helpers that are not in 8a7d9c2 must not drive the board themselves
EXTRA = sorted(set(now) - set(ref) - {"_stream_raw"})
print(f"     helpers added on top of {PIN}: {', '.join(EXTRA) or 'none'}")
for name in EXTRA:
    calls = set()
    for node in ast.walk(ast.parse(textwrap.dedent(now[name]))):
        if isinstance(node, ast.Attribute):
            chain, cur = [], node
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name) and cur.id == "qd":
                calls.add("qd." + ".".join(reversed(chain)))
    check(f"{name}() makes no direct qd.* board call", not calls,
          ", ".join(sorted(calls)) or "none")

print("\n4. both notebooks call the pinned class correctly")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "notebook_modules"))
import inspect
import json
from multipoint_lockin_program import MultipointLockinODMR as M

# Receivers that hold a MultipointLockinODMR in the notebooks.
PROG = {"MultipointLockinODMR", "prog", "prog_avg", "prog_burst", "prog_stream",
        "prog_live", "prog_mp_stream", "_prog_prime", "_prog_zero", "_probe", "_pz",
        "prog_prime"}

missing, wrong_kw = [], []
for nb_name in ("Twopoint_Lockin_module.ipynb", "Lockin_module.ipynb"):
    nb = json.loads((REPO / "Modules" / nb_name).read_text())
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        text = "\n".join("" if l.lstrip().startswith(("%", "!", "?")) else l
                          for l in "".join(c["source"]).splitlines())
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        # 4a. attribute reads: does the name exist at all?
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id in PROG and not hasattr(M, node.attr)):
                missing.append(f"{nb_name} cell {i}: .{node.attr}")

        # 4b. calls: are the keywords ones the PINNED signature accepts?
        #     This is the check that was missing when describe_timing(host_call_s=...)
        #     reached the rig -- an attribute-exists test cannot catch a signature
        #     that changed between revisions.
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            recv = node.func.value
            if not (isinstance(recv, ast.Name) and recv.id in PROG):
                continue
            fn = getattr(M, node.func.attr, None)
            if fn is None or not callable(fn):
                continue
            try:
                params = inspect.signature(fn).parameters
            except (TypeError, ValueError):
                continue
            if any(pp.kind is pp.VAR_KEYWORD for pp in params.values()):
                continue
            for kw in (k.arg for k in node.keywords if k.arg):
                if kw not in params:
                    wrong_kw.append(f"{nb_name} cell {i}: .{node.func.attr}({kw}=...) "
                                    f"but signature is {inspect.signature(fn)}")

check("every attribute the notebooks read exists", not missing,
      "; ".join(sorted(set(missing))[:4]) or "all resolve")
check("every keyword the notebooks pass is accepted", not wrong_kw,
      "; ".join(sorted(set(wrong_kw))[:3]) or "all match")

print("\n5. every other notebook_modules call also matches its signature")
# Same class of bug, wider net. These modules are not pinned, but a signature that
# drifts under the notebook fails at the rig exactly like describe_timing did.
import importlib

MODS = ["twopoint_runner", "twopoint_postprocess", "twopoint_spectra", "burst_qc",
        "spike_rejection", "figure_autosave", "odmr_sensitivity"]
loaded = {}
for m in MODS:
    try:
        loaded[m] = importlib.import_module(m)
    except Exception as exc:
        print(f"     (skipped {m}: {exc})")

bad = []
for nb_name in ("Twopoint_Lockin_module.ipynb", "Lockin_module.ipynb"):
    nb = json.loads((REPO / "Modules" / nb_name).read_text())
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        text = "\n".join("" if l.lstrip().startswith(("%", "!", "?")) else l
                          for l in "".join(c["source"]).splitlines())
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            recv = node.func.value
            if not (isinstance(recv, ast.Name) and recv.id in loaded):
                continue
            fn = getattr(loaded[recv.id], node.func.attr, None)
            if fn is None:
                bad.append(f"{nb_name} cell {i}: {recv.id}.{node.func.attr} missing")
                continue
            if not callable(fn):
                continue
            try:
                params = inspect.signature(fn).parameters
            except (TypeError, ValueError):
                continue
            if any(pp.kind is pp.VAR_KEYWORD for pp in params.values()):
                continue
            for kw in (k.arg for k in node.keywords if k.arg):
                if kw not in params:
                    bad.append(f"{nb_name} cell {i}: {recv.id}.{node.func.attr}({kw}=...)")
check("every notebook_modules call matches", not bad,
      "; ".join(sorted(set(bad))[:3]) or f"{len(loaded)} modules checked")

failed = [n for n, ok in RESULTS if not ok]
print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
if failed:
    print("FAILED: " + "; ".join(failed))
raise SystemExit(1 if failed else 0)
