"""Run the notebooks' own post-processing cells and check the PNGs land on disk.

Drives the real cell source out of the .ipynb files inside a real IPython shell
with the inline backend, against a scratch copy of a recorded run -- so it tests
the wiring as it will actually execute, not a paraphrase of it.

Run:  python scripts/test_notebook_figure_saving.py     (exit 0 = all passed)
"""
import contextlib, io, json, shutil, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebook_modules"))

from IPython.core.interactiveshell import InteractiveShell

fails = []
def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        fails.append(name)

InteractiveShell.clear_instance()
ip = InteractiveShell.instance()
# A bare shell has no GUI loop, so %matplotlib raises after configuring the
# inline backend -- which is the part under test. Swallow the traceback.
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    ip.run_cell("%matplotlib inline")

RUNS = REPO / "data/results/081726 (Test after Increased Sensitivity)/Two-point lockin"
# (cell marker, label, recorded run, name to copy it under). Cells are found by the
# first line of their source rather than by index, so inserting a cell above them
# does not silently point this test at the wrong one. The 2026-08-17 burst runs
# predate the _avg_/_burst_/_stream_ rename and are still called _live_, so the copy
# takes the name the cell's newest-run fallback globs for.
CASES = [
    ("# Step 5C", "Step 5C  stream", "twopoint_lockin_stream_20260817_121002.csv",
         "twopoint_lockin_stream_20260817_121002.csv"),
    ("# Step 5B", "Step 5B  burst",  "twopoint_lockin_live_20260817_114413.csv",
         "twopoint_lockin_burst_20260817_114413.csv"),
]

nb = json.loads((REPO / "Modules/Twopoint_Lockin_module.ipynb").read_text())


def cell_source(marker):
    hits = [c for c in nb["cells"] if c["cell_type"] == "code"
            and "".join(c["source"]).lstrip().startswith(marker)]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one cell starting with {marker!r}, found {len(hits)}")
    return "".join(hits[0]["source"])

for marker, label, source_name, csv_name in CASES:
    tmp = Path(tempfile.mkdtemp(prefix="nbfig_"))
    shutil.copy(RUNS / source_name, tmp / csv_name)
    stem = Path(csv_name).stem
    ip.run_cell(f"""
import sys
sys.path.insert(0, {str(REPO / 'notebook_modules')!r})
from pathlib import Path
TWOPOINT_DIR = Path({str(tmp)!r})
for _v in ("AVG_LAST_CSV", "BURST_LAST_CSV", "STREAM_LAST_CSV"):
    globals().pop(_v, None)
import figure_autosave
figure_autosave.enable(TWOPOINT_DIR, verbose=False)
""")
    with contextlib.redirect_stdout(io.StringIO()):
        res = ip.run_cell(cell_source(marker))
    if res.error_in_exec:
        check(f"{label}: cell runs", False, repr(res.error_in_exec))
        shutil.rmtree(tmp, ignore_errors=True)
        continue
    # Since 2026-08-20 each analysis cell also draws the 3 Hz low-pass view, so a
    # run writes three PNGs, not two.
    want = sorted([f"{stem}_fft.png", f"{stem}_lowpass.png", f"{stem}_result.png"])
    got = sorted(p.name for p in tmp.glob("*.png"))
    check(f"{label}: writes _result.png, _lowpass.png and _fft.png beside the CSV",
          got == want, str(got))
    check(f"{label}: all PNGs are non-trivial",
          all((tmp / n).stat().st_size > 20_000 for n in got))

    with contextlib.redirect_stdout(io.StringIO()):
        ip.run_cell(cell_source(marker))
    got2 = sorted(p.name for p in tmp.glob("*.png"))
    check(f"{label}: re-running overwrites rather than accumulates", got2 == want, str(got2))
    shutil.rmtree(tmp, ignore_errors=True)

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
