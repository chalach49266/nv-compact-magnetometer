#!/usr/bin/env python3
"""Measure where a two-point `acquire()` call actually spends its time.

Run this ON THE RIG (the machine that can reach the RFSoC). Everything else in
the analysis is derived offline from saved CSVs, but the split between FPGA pulse
work and host round-trips can only be measured live, one RPC at a time. This
script fills section 3.2 of
`docs/2026-08-14_twopoint_methods/TWOPOINT_METHODS_AND_TIMING.md`.

    python scripts/profile_twopoint_acquire.py --check-only        # no hardware
    python scripts/profile_twopoint_acquire.py --ip 192.168.0.103
    python scripts/profile_twopoint_acquire.py --reps 1 4 23 100 --tau 120 213

Three targeted tests, each of which decides one open question and exits:

    --floor               S3: is the per-call cost really a ~10 ms floor, with the
                          rate flat against reps below it? (~30 s)
    --compare-read-paths  S1: interleaved burst vs stream on the SAME program.
                          Decides whether streaming's ~6x noise excess is the read
                          path or the environment. Run this one first. (~1 min)
    --stream-scan         S2: does the streaming excess grow with run length, i.e.
                          is the board worker losing the race as the buffer fills?

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

DEFAULT_OUTDIR = REPO_ROOT / "docs" / "2026-08-14_twopoint_methods" / "tables"


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


# ------------------------------------------------- read-path comparison (S1)


def _per_rep_matrix(data) -> np.ndarray:
    """(n_reps, n_freqs) of normalised counts from an acquire/stream result."""
    return np.atleast_2d(np.asarray(data.signal, dtype=float))


def _floor_and_sigma(block: np.ndarray, cadence_s: float, band=(10.0, None)) -> dict:
    """Noise of the differential and of the common mode, for one contiguous run.

    Deliberately works in raw normalised counts, not in kHz: the ratio between two
    read paths is what matters and it does not depend on the calibration, so this
    stays valid even if the ODMR fit has moved since.
    """
    import twopoint_spectra as spec

    diff = block[:, 1] - block[:, 0]
    common = block[:, 0]
    fs = 1.0 / cadence_s
    lo, hi = band[0], (band[1] or 0.4 * fs)
    f, psd = spec.welch_psd(diff, fs, nseg=8, detrend="linear")
    m = (f > lo) & (f < hi)
    return {
        "n": int(block.shape[0]),
        "sigma_diff": float(np.std(diff, ddof=1)),
        "sigma_common": float(np.std(common, ddof=1)),
        "mean_common": float(np.mean(common)),
        "asd_floor": float(np.sqrt(np.median(psd[m]))) if m.any() else float("nan"),
    }


def compare_read_paths(freqs_mhz, tau_us: float, reps: int, n_pairs: int) -> pd.DataFrame:
    """S1 -- the decisive test for the streaming noise excess.

    On 2026-08-14 the streamed data measured 5.3-6.3x the burst amplitude spectral
    density at the same tau, flat across 10-500 Hz, ten minutes apart with nothing
    changed on the bench. Two explanations survive: the read path differs, or the
    room got quieter in between. Ten minutes is long enough for the second to be
    plausible, so the runs are not comparable.

    This makes them comparable. The SAME program object, the same reps, alternating
    burst / stream / burst / stream within a few seconds. Identical FPGA work; only
    the way the host takes the data out differs.

        ratio ~ 1.0   the read paths agree -- the ~6x was the environment
        ratio >> 1    streaming really is noisier, and it is a software defect
    """
    prog = build_program(freqs_mhz, tau_us, reps)
    cadence = prog.time_per_rep()
    print(f"  program: tau={tau_us:.1f} us, {reps} reps, cadence {cadence * 1e6:.1f} us/rep "
          f"({1 / cadence:.0f} Hz), {reps * cadence * 1e3:.1f} ms of FPGA work per run")
    print(f"  {n_pairs} interleaved pairs, burst first\n")

    prog.acquire(progress=False)                       # warm-up, discarded
    rows = []
    for k in range(n_pairs):
        prog.reset_freshness_counters()
        t0 = perf_counter()
        d = prog.acquire(progress=False, per_rep=True, reset_tproc=True)
        t_burst = perf_counter() - t0
        if d is None:
            print(f"  pair {k}: burst came back stale on every retry, skipping")
            continue
        rec = _floor_and_sigma(_per_rep_matrix(d), cadence)
        rec.update({"pair": k, "path": "burst", "wall_s": t_burst,
                    "stale": int(prog.n_stale_acquires)})
        rows.append(rec)

        t0 = perf_counter()
        blocks = [np.atleast_2d(np.asarray(pkt.signal, dtype=float))
                  for pkt in prog.stream(total_reps=reps)]
        t_stream = perf_counter() - t0
        rec = _floor_and_sigma(np.concatenate(blocks, axis=0), cadence)
        rec.update({"pair": k, "path": "stream", "wall_s": t_stream, "stale": 0})
        rows.append(rec)

        print(f"  pair {k}: burst sigma_diff={rows[-2]['sigma_diff']:.4g} "
              f"floor={rows[-2]['asd_floor']:.4g}   |   "
              f"stream sigma_diff={rows[-1]['sigma_diff']:.4g} "
              f"floor={rows[-1]['asd_floor']:.4g}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("\n  no usable pairs -- every burst came back stale.")
        return df

    print("\n  " + "-" * 66)
    summary = df.groupby("path")[["sigma_diff", "sigma_common", "mean_common",
                                  "asd_floor"]].median()
    print(summary.to_string())
    if {"burst", "stream"} <= set(summary.index):
        r_floor = summary.loc["stream", "asd_floor"] / summary.loc["burst", "asd_floor"]
        r_sigma = summary.loc["stream", "sigma_diff"] / summary.loc["burst", "sigma_diff"]
        r_level = summary.loc["stream", "mean_common"] / summary.loc["burst", "mean_common"]
        print(f"\n  stream / burst   ASD floor  {r_floor:6.2f}x")
        print(f"                   sigma      {r_sigma:6.2f}x")
        print(f"                   PL level   {r_level:6.2f}x")
        print()
        if r_floor < 1.25:
            print("  VERDICT: the read paths agree. The ~6x seen on 2026-08-14 was the")
            print("  environment changing between the two runs, not a streaming defect.")
            print("  Streaming is then the preferred high-rate path with no caveat.")
        else:
            print(f"  VERDICT: streaming really is {r_floor:.1f}x noisier on identical FPGA")
            print("  work. This is a software defect in the read path, not physics.")
            print("  Next: does it scale with run length? Run --stream-scan.")
    return df


def stream_length_scan(freqs_mhz, tau_us: float, lengths) -> pd.DataFrame:
    """S2 -- does the streaming excess grow with run length?

    If the noise rises with `total_reps`, the board-side worker is losing the race
    against the tProc as the accumulated buffer fills, and the fix is to bound the
    poll interval. If it is flat, the excess is per-sample and the buffer is fine.
    """
    rows = []
    for total in lengths:
        prog = build_program(freqs_mhz, tau_us, int(total))
        cadence = prog.time_per_rep()
        try:
            head = prog.stream_headroom()
            headroom = f"{head['shots_that_fit']} shots ({head['seconds_that_fit']:.2f} s)"
        except Exception as exc:
            headroom = f"unknown ({exc})"
        t0 = perf_counter()
        blocks = [np.atleast_2d(np.asarray(pkt.signal, dtype=float))
                  for pkt in prog.stream(total_reps=int(total))]
        wall = perf_counter() - t0
        rec = _floor_and_sigma(np.concatenate(blocks, axis=0), cadence)
        rec.update({"total_reps": int(total), "run_s": total * cadence, "wall_s": wall,
                    "duty_cycle": total * cadence / wall, "headroom": headroom})
        rows.append(rec)
        print(f"  {total:7d} reps ({total * cadence:6.2f} s): "
              f"sigma_diff={rec['sigma_diff']:.4g}  floor={rec['asd_floor']:.4g}  "
              f"duty={100 * rec['duty_cycle']:.0f}%  buffer fits {headroom}")

    df = pd.DataFrame(rows)
    if len(df) > 1:
        trend = df["asd_floor"].iloc[-1] / df["asd_floor"].iloc[0]
        print(f"\n  floor at the longest run / shortest run: {trend:.2f}x")
        print("  -> buffer racing" if trend > 1.3 else
              "  -> flat: the excess is per-sample, not a buffer-fill effect")
    return df


def measure_floor_and_flatness(freqs_mhz, tau_us: float, reps_list, n_batches: int
                               ) -> pd.DataFrame:
    """S3 -- confirm the per-call floor, and that the rate is flat against reps.

    The offline claim is that `period = max(host_call, reps * time_per_rep)` rather
    than a serial sum. This tests it directly: sweep reps at fixed tau and watch the
    period stay put until the FPGA work outgrows the floor, then track it one for one.
    """
    from multipoint_lockin_program import MultipointLockinODMR

    probe = build_program(freqs_mhz, tau_us, 1)
    cal = MultipointLockinODMR.measure_host_floor(probe, n=max(20, n_batches), reps=1)
    print(f"  reps=1 probe: FPGA work {cal['fpga_s'] * 1e3:.3f} ms, "
          f"fastest call {cal['floor_s'] * 1e3:.2f} ms, median {cal['median_s'] * 1e3:.2f} ms, "
          f"jitter {cal['jitter_s'] * 1e3:.2f} ms")
    print(f"  -> per-call floor is {cal['floor_s'] * 1e3:.2f} ms against "
          f"{cal['fpga_s'] * 1e3:.3f} ms of pulse work "
          f"({cal['fpga_s'] / cal['floor_s'] * 100:.1f}% FPGA)\n")

    rows = []
    for reps in reps_list:
        prog = build_program(freqs_mhz, tau_us, int(reps))
        fpga_ms = prog.total_time() * 1e3
        prog.acquire(progress=False)
        times = []
        for _ in range(n_batches):
            t0 = perf_counter()
            prog.acquire(progress=False)
            times.append(perf_counter() - t0)
        arr = np.asarray(times) * 1e3
        rows.append({
            "tau_us": tau_us, "reps": int(reps), "fpga_ms": fpga_ms,
            "period_ms_p05": float(np.percentile(arr, 5)),
            "period_ms_median": float(np.median(arr)),
            "rate_hz": 1e3 / float(np.median(arr)),
            "predicted_ms": max(cal["median_s"] * 1e3, fpga_ms),
        })
        print(f"  reps={reps:5d}  FPGA {fpga_ms:7.2f} ms  period {np.median(arr):7.2f} ms  "
              f"-> {1e3 / np.median(arr):7.1f} Hz   "
              f"(max-model predicts {max(cal['median_s'] * 1e3, fpga_ms):.2f} ms)")

    df = pd.DataFrame(rows)
    below = df[df["fpga_ms"] < cal["floor_s"] * 1e3]
    if len(below) > 1:
        spread = below["period_ms_median"].max() / below["period_ms_median"].min()
        print(f"\n  across the reps whose FPGA work fits under the floor, the period varies "
              f"by {spread:.2f}x while the FPGA work varies by "
              f"{below['fpga_ms'].max() / below['fpga_ms'].min():.1f}x")
        print("  -> host-bound, as the offline analysis concluded" if spread < 1.15 else
              "  -> NOT flat; the offline conclusion does not reproduce on this rig")
    return df


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
    ap.add_argument("--compare-read-paths", action="store_true",
                    help="S1: interleaved burst vs stream on the same program. This is "
                         "the test that decides whether the streaming noise excess is "
                         "the read path or the room. ~1 minute.")
    ap.add_argument("--stream-scan", action="store_true",
                    help="S2: stream at several run lengths; does the noise grow with "
                         "buffer fill?")
    ap.add_argument("--floor", action="store_true",
                    help="S3: measure the per-call host floor and confirm the rate is "
                         "flat against reps below it")
    ap.add_argument("--pairs", type=int, default=5,
                    help="interleaved burst/stream pairs for --compare-read-paths")
    ap.add_argument("--stream-lengths", type=int, nargs="+",
                    default=[2000, 20000, 125000],
                    help="run lengths in reps for --stream-scan")
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
    tau0 = args.tau[0]

    # Targeted tests run on their own and exit -- they are the ones you reach for
    # when chasing a specific defect, and none of them needs the full profile.
    if args.compare_read_paths or args.stream_scan or args.floor:
        if args.floor:
            print("=" * 78)
            print("S3. PER-CALL FLOOR AND RATE FLATNESS")
            print("=" * 78)
            measure_floor_and_flatness(freqs, tau0, args.reps, args.batches).to_csv(
                args.outdir / "measured_floor_flatness.csv", index=False)
            print()
        if args.compare_read_paths:
            print("=" * 78)
            print("S1. BURST vs STREAM, INTERLEAVED, SAME PROGRAM")
            print("=" * 78)
            compare_read_paths(freqs, tau0, max(args.reps), args.pairs).to_csv(
                args.outdir / "measured_read_path_compare.csv", index=False)
            print()
        if args.stream_scan:
            print("=" * 78)
            print("S2. STREAM NOISE vs RUN LENGTH")
            print("=" * 78)
            stream_length_scan(freqs, tau0, args.stream_lengths).to_csv(
                args.outdir / "measured_stream_scan.csv", index=False)
            print()
        print(f"tables -> {args.outdir}")
        return 0

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
