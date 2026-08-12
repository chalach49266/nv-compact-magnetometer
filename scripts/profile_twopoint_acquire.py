#!/usr/bin/env python3
"""Measure where a two-point `acquire()` call actually spends its time.

Run this ON THE RIG (the machine that can reach the RFSoC). Everything else in
the analysis is derived offline from saved CSVs, but the split between FPGA pulse
work and host round-trips can only be measured live, one RPC at a time. This
script fills section 2.2 of
`docs/2026-08-06_twopoint_timing/TIMING_AND_NOISE_ANALYSIS.md`.

    python scripts/profile_twopoint_acquire.py --check-only        # no hardware
    python scripts/profile_twopoint_acquire.py --ip 192.168.0.103
    python scripts/profile_twopoint_acquire.py --reps 1 4 23 100 --tau 120 213

What it does
------------
1. **Static check (no hardware).** Confirms whether `qickdawg`'s call into
   `config_all` matches the installed `qick` signature. On qick 0.2.386 it does
   not -- qickdawg passes `load_pulses=`, qick expects `load_envelopes=` -- so
   every acquire raises TypeError and silently falls back to `config_all(soc)`
   with `reset=False`, which is a no-op stop on tProc v1. That fallback is the
   leading suspect for the burst-mode stale reads, so the check runs first and
   runs anywhere.

2. **Per-RPC timing.** Wraps the soc proxy so every remote call is timed, then
   runs a batch of acquires. Reports median/p95 for each RPC and the residual
   (which is FPGA wait time inside the poll loop).

3. **Rate model.** Sweeps rep count and readout window, and compares the measured
   batch time against `n_reps * time_per_rep()` so the model can be trusted (or
   corrected) before it is used to pick operating points.
"""
from __future__ import annotations

import argparse
import inspect
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "notebook_modules"))

DEFAULT_OUTDIR = REPO_ROOT / "docs" / "2026-08-06_twopoint_timing" / "tables"


# ------------------------------------------------------------ static check


def check_config_all_signature() -> dict:
    """Does qickdawg's `config_all` call match the installed qick signature?

    Returns a dict describing what was found; safe to call without hardware.
    """
    result = {"qick_version": None, "accepts_load_pulses": None,
              "accepts_load_envelopes": None, "accepts_reset": None,
              "qickdawg_passes": None, "falls_back": None}
    try:
        import qick
        from qick.asm_v1 import AcquireProgram
        result["qick_version"] = getattr(qick, "__version__", "unknown")
        params = inspect.signature(AcquireProgram.config_all).parameters
        result["accepts_load_pulses"] = "load_pulses" in params
        result["accepts_load_envelopes"] = "load_envelopes" in params
        result["accepts_reset"] = "reset" in params
    except ImportError as exc:
        result["qick_version"] = f"not importable ({exc})"
        return result

    src_path = REPO_ROOT / "qickdawg" / "nvpulsing" / "nvaverageprogram.py"
    if src_path.exists():
        src = src_path.read_text()
        result["qickdawg_passes"] = ("load_pulses=" if "config_all(qd.soc, load_pulses="
                                     in src else "load_envelopes=" if
                                     "config_all(qd.soc, load_envelopes=" in src else "?")
        result["falls_back"] = (result["qickdawg_passes"] == "load_pulses="
                                and not result["accepts_load_pulses"])
    return result


def report_signature_check(info: dict) -> bool:
    """Print the static check. Returns True if the mismatch is present."""
    print("=" * 78)
    print("1. STATIC CHECK -- config_all signature")
    print("=" * 78)
    print(f"  qick version              : {info['qick_version']}")
    print(f"  config_all accepts ...    : load_pulses={info['accepts_load_pulses']}, "
          f"load_envelopes={info['accepts_load_envelopes']}, reset={info['accepts_reset']}")
    print(f"  qickdawg passes           : {info['qickdawg_passes']}")
    if info["falls_back"]:
        print("\n  MISMATCH. Every acquire() raises TypeError and falls back to")
        print("  config_all(soc) with reset=False. On tProc v1 that means")
        print("  stop_tproc(lazy=True), which does nothing -- the tProc is never")
        print("  stopped between acquisitions. Expect stale accumulated-buffer")
        print("  reads whenever one program run is long (burst mode).")
    else:
        print("\n  OK -- qickdawg's call matches the installed qick signature.")
    print()
    return bool(info["falls_back"])


# ------------------------------------------------------------- rpc timing


class TimedSoc:
    """Transparent wrapper around the soc proxy that times every remote call.

    `NVAveragerProgram.acquire` reaches the board exclusively through the module
    global `qd.soc`, so swapping that global for this wrapper times the whole RPC
    chain -- including the calls qick makes internally from `config_all` -- with
    no changes to the driver.
    """

    def __init__(self, soc, sink: dict):
        object.__setattr__(self, "_soc", soc)
        object.__setattr__(self, "_sink", sink)

    def __getattr__(self, name):
        attr = getattr(object.__getattribute__(self, "_soc"), name)
        if not callable(attr):
            return attr
        sink = object.__getattribute__(self, "_sink")

        def timed(*args, **kwargs):
            t0 = perf_counter()
            try:
                return attr(*args, **kwargs)
            finally:
                sink[name].append(perf_counter() - t0)

        return timed

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_soc"), name, value)

    def __getitem__(self, key):
        return object.__getattribute__(self, "_soc")[key]


def profile_acquires(prog, n_batches: int, per_rep: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """Time `n_batches` acquires, breaking each one down by remote call."""
    import qickdawg as qd

    real_soc = qd.soc
    sink: dict[str, list[float]] = defaultdict(list)
    totals = []

    qd.soc = TimedSoc(real_soc, sink)
    try:
        prog.acquire(progress=False, per_rep=per_rep)   # warm-up, not recorded
        sink.clear()
        for _ in range(n_batches):
            t0 = perf_counter()
            prog.acquire(progress=False, per_rep=per_rep)
            totals.append(perf_counter() - t0)
    finally:
        qd.soc = real_soc

    rows = []
    for name, samples in sorted(sink.items()):
        arr = np.asarray(samples) * 1e3
        rows.append({
            "rpc": name,
            "calls_per_batch": len(samples) / n_batches,
            "ms_median": float(np.median(arr)),
            "ms_p95": float(np.percentile(arr, 95)),
            "ms_total_per_batch": float(arr.sum() / n_batches),
        })
    return pd.DataFrame(rows).sort_values("ms_total_per_batch", ascending=False), np.asarray(totals)


# ------------------------------------------------------------- rate model


def build_program(freqs_mhz, tau_us: float, reps: int, skip_reference: bool = True,
                  relax_treg: int = 1000):
    """A live-mode two-point program: pre_init off, matching the notebook's Step 4."""
    from copy import copy
    import qickdawg as qd
    from multipoint_lockin_program import MultipointLockinODMR

    cfg = copy(qd.NVConfiguration())
    cfg.adc_channel = 0
    cfg.mw_channel = 1
    cfg.mw_nqz = 1
    cfg.mw_gain = 10500
    cfg.laser_gate_pmod = 0
    cfg.readout_integration_tus = float(tau_us)
    cfg.relax_delay_treg = int(relax_treg)
    cfg.multipoint_freqs_mhz = list(freqs_mhz)
    cfg.odmr_reference_offres_mhz = 2700.0
    cfg.multipoint_skip_reference = bool(skip_reference)
    cfg.mw_start_fMHz = float(freqs_mhz[0])
    cfg.mw_end_fMHz = float(freqs_mhz[0])
    cfg.nsweep_points = 1
    cfg.reps = int(reps)
    cfg.pre_init = False
    return MultipointLockinODMR(cfg)


def sweep_rate_model(freqs_mhz, reps_list, tau_list, n_batches: int) -> pd.DataFrame:
    """Measured batch time vs the program's own prediction, over reps and tau."""
    rows = []
    for tau in tau_list:
        for reps in reps_list:
            prog = build_program(freqs_mhz, tau, reps)
            predicted_ms = prog.total_time() * 1e3
            times = []
            prog.acquire(progress=False)                 # warm-up
            for _ in range(n_batches):
                t0 = perf_counter()
                prog.acquire(progress=False)
                times.append(perf_counter() - t0)
            arr = np.asarray(times) * 1e3
            rows.append({
                "tau_us": tau, "reps": reps,
                "predicted_ms": predicted_ms,
                "measured_ms_median": float(np.median(arr)),
                "measured_ms_p05": float(np.percentile(arr, 5)),
                "measured_ms_p95": float(np.percentile(arr, 95)),
                "host_overhead_ms": float(np.median(arr)) - predicted_ms,
                "rate_hz": 1e3 / float(np.median(arr)),
            })
            print(f"  tau={tau:6.1f} us  reps={reps:5d}  predicted {predicted_ms:8.2f} ms  "
                  f"measured {np.median(arr):8.2f} ms  -> {1e3 / np.median(arr):7.1f} Hz")
    return pd.DataFrame(rows)


def default_frequencies() -> list[float]:
    """Parked pair from the most recent calibration JSON in the data folder."""
    import json

    cals = sorted((REPO_ROOT / "data" / "twopoint_lockin").rglob(
        "twopoint_calibration_odmr_sweep_*.json"))
    if not cals:
        raise SystemExit("no calibration JSON found; pass --freqs explicitly")
    with open(cals[-1]) as fh:
        cal = json.load(fh)
    print(f"  parked pair from {cals[-1].name}: "
          f"{cal['f_minus_mhz']:.4f} / {cal['f_plus_mhz']:.4f} MHz")
    return [float(cal["f_minus_mhz"]), float(cal["f_plus_mhz"])]


# ------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="192.168.0.103", help="RFSoC address")
    ap.add_argument("--check-only", action="store_true",
                    help="run the static signature check and exit (no hardware)")
    ap.add_argument("--batches", type=int, default=40, help="acquires per operating point")
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 4, 23, 100, 500],
                    help="rep counts to sweep")
    ap.add_argument("--tau", type=float, nargs="+", default=[120.0, 213.0],
                    help="readout windows (us) to sweep")
    ap.add_argument("--freqs", type=float, nargs=2, default=None,
                    help="parked pair in MHz (default: newest calibration)")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    mismatch = report_signature_check(check_config_all_signature())
    if args.check_only:
        return 0

    import qickdawg as qd
    print(f"connecting to RFSoC at {args.ip} ...")
    qd.start_client(args.ip)
    print("connected.\n")

    freqs = args.freqs if args.freqs else default_frequencies()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("2. PER-RPC TIMING  (tau=213 us, 23 reps -- today's operating point)")
    print("=" * 78)
    prog = build_program(freqs, 213.0, 23)

    # The saved CSVs say the real cadence is ~424.8 us/rep while time_per_rep()
    # predicts 435.3, i.e. the two relax windows the model adds are absorbed
    # rather than serialised. Print the resolved constants so the discrepancy is
    # attributable to a specific number rather than left as a 2.5% mystery.
    cfg = prog.cfg
    n_freqs = len(cfg.multipoint_freqs_mhz)
    print(f"  resolved readout window : {cfg.readout_integration_tus:.4f} us "
          f"({cfg.readout_integration_treg} treg)")
    print(f"  resolved relax delay    : {cfg.relax_delay_tus:.4f} us "
          f"({cfg.relax_delay_treg} treg)")
    print(f"  time_per_rep() model    : {prog.time_per_rep() * 1e6:.2f} us")
    print(f"  readout-only model      : {n_freqs * cfg.readout_integration_tus:.2f} us "
          f"({n_freqs} freqs x one window)")
    print()
    fpga_ms = prog.total_time() * 1e3
    rpc, totals = profile_acquires(prog, args.batches)
    total_ms = float(np.median(totals)) * 1e3
    rpc_ms = float(rpc["ms_total_per_batch"].sum())

    print(rpc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  FPGA pulse work (model)   {fpga_ms:7.3f} ms  ({100 * fpga_ms / total_ms:5.1f}%)")
    print(f"  sum of all RPC calls      {rpc_ms:7.3f} ms   (includes the poll loop's FPGA wait)")
    print(f"  total acquire()           {total_ms:7.3f} ms  -> {1e3 / total_ms:.1f} Hz")
    print(f"  host-only (total - FPGA)  {total_ms - fpga_ms:7.3f} ms  "
          f"({100 * (total_ms - fpga_ms) / total_ms:5.1f}%)")
    rpc.to_csv(args.outdir / "measured_rpc_breakdown.csv", index=False)

    print()
    print("=" * 78)
    print("3. RATE MODEL  (measured vs predicted)")
    print("=" * 78)
    sweep = sweep_rate_model(freqs, args.reps, args.tau, max(8, args.batches // 4))
    sweep.to_csv(args.outdir / "measured_rate_model.csv", index=False)

    print()
    ratio = sweep["measured_ms_median"] / sweep["predicted_ms"]
    big = sweep[sweep["reps"] >= 100]
    print(f"  measured/predicted overall : {ratio.min():.3f} - {ratio.max():.3f}")
    if not big.empty:
        print(f"  at reps >= 100 (FPGA-dominated): "
              f"{(big['measured_ms_median'] / big['predicted_ms']).mean():.3f}")
    print(f"  host overhead (median across points): "
          f"{sweep['host_overhead_ms'].median():.2f} ms per acquire")
    if mismatch:
        print("\n  NOTE: the config_all mismatch above is still present, so these")
        print("  numbers include the TypeError fallback on every call.")
    print(f"\ntables -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
