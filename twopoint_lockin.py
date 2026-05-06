"""Two-point lock-in modulation for NV magnetometry.

The FPGA hardware loop order (from NVAveragerProgram.make_program) is:

    outer: reps
    inner: sweep (f1 → f2)

So for every rep the FPGA naturally alternates  f1 → f2  at the hardware
rate (~1.17 kHz at max integration window, faster with shorter windows).
Calling acquire(raw_data=True) returns the per-rep, per-frequency raw buffer
instead of the averaged result — exactly the same trick used by fast PL readout
with d_buf.

Lock-in signal per rep:
    contrast_f1[i] = (MW_on[i,f1] - MW_off[i,f1]) / readout_integration_treg
    contrast_f2[i] = (MW_on[i,f2] - MW_off[i,f2]) / readout_integration_treg
    lockin[i]      = contrast_f1[i] - contrast_f2[i]

Hardware switching rate = 1 / (2 × time_per_body):
    time_per_body ≈ 2 × readout_integration_tus + 2 × relax_delay_tus
    Default (max int window): ~427 µs/body → ~1.17 kHz switching

Tuning the switching rate:
    Reduce readout_integration_tus (e.g. 50 µs → ~5 kHz switching).
    Fewer averages per batch → noisier per point but faster effective rate.
"""

from __future__ import annotations

from copy import copy
from time import perf_counter

import numpy as np
import pandas as pd

import qickdawg as qd


def run_twopoint_lockin(
    config,
    *,
    center_fMHz: float = 2870.91,
    fwhm_mhz: float = 17.47,
    slope_fraction: float = 0.5,
    duration_sec: float | None = 60.0,
    reps_per_batch: int = 1000,
    csv_filename: str | None = None,
) -> pd.DataFrame:
    """
    Acquire the two-point lock-in signal as a high-rate time series.

    The FPGA alternates f1 ↔ f2 every rep at the hardware rate (~1 kHz).
    Each batch call returns reps_per_batch individual per-rep measurements.
    Python loops over batches like run_fast_pl_no_display does with d_buf.

    Parameters
    ----------
    config : NVConfiguration
        Base hardware configuration. Not modified — a copy is used internally.
    center_fMHz : float
        ODMR resonance center in MHz.
    fwhm_mhz : float
        ODMR full-width at half-maximum in MHz.
    slope_fraction : float
        Offset from center as fraction of FWHM.
        0.5  → ±FWHM/2 (half-max points, default).
        0.289 → ±FWHM/(2√3) (Lorentzian inflection / max-slope points).
    duration_sec : float | None
        Total run time. None → run until keyboard interrupt.
    reps_per_batch : int
        FPGA reps per acquire() call — controls batch size.
        Each rep = one f1+f2 hardware cycle (~854 µs at max int. window).
        1000 reps → ~854 ms per batch, returning 1000 time-stamped points.
        Increase for longer batches; decrease to get data back to Python faster.
    csv_filename : str | None
        Save full DataFrame to this path after the run (optional).

    Returns
    -------
    pd.DataFrame with columns:
        time_s         hardware timestamp (seconds from start)
        signal_f1      MW-on  PL at f1 (ADC units, normalised)
        reference_f1   MW-off PL at f1
        signal_f2      MW-on  PL at f2
        reference_f2   MW-off PL at f2
        contrast_f1    signal_f1 − reference_f1
        contrast_f2    signal_f2 − reference_f2
        lockin_signal  contrast_f1 − contrast_f2
    """
    f1 = center_fMHz - slope_fraction * fwhm_mhz
    f2 = center_fMHz + slope_fraction * fwhm_mhz

    cfg = copy(config)
    cfg.mw_start_fMHz = f1
    cfg.mw_end_fMHz = f2
    cfg.nsweep_points = 2
    cfg.reps = reps_per_batch
    cfg.pre_init = getattr(config, "pre_init", True)
    cfg.odmr_reference_offres_mhz = 2000.0  # reference shot fires at 2000 MHz (off-resonance)

    prog = qd.LockinODMR(cfg)

    # Each rep: 2 bodies (f1+f2), each body = 2×readout + 2×relax
    body_tus = 2 * cfg.readout_integration_tus + 2 * cfg.relax_delay_tus
    cycle_tus = 2 * body_tus          # one f1+f2 pair
    batch_sec = reps_per_batch * cycle_tus / 1e6
    switch_hz = 1e6 / cycle_tus

    print("Two-point lock-in — hardware rate mode")
    print(f"  Center:           {center_fMHz:.2f} MHz")
    print(f"  FWHM:             {fwhm_mhz:.2f} MHz  (slope_fraction={slope_fraction})")
    print(f"  f1 (left  slope): {f1:.2f} MHz")
    print(f"  f2 (right slope): {f2:.2f} MHz")
    print(f"  Integration time: {cfg.readout_integration_tus:.1f} µs per readout window")
    print(f"  Hardware f1↔f2 switching rate: {switch_hz:.0f} Hz  (~{cycle_tus:.0f} µs/cycle)")
    print(f"  Batch size:       {reps_per_batch} reps  (~{batch_sec*1e3:.0f} ms per batch)")
    print(f"  Duration:         {duration_sec} s")
    print("Press Interrupt (■) to stop early.\n")

    scale = float(cfg.readout_integration_treg)

    all_rows: list[dict] = []
    t0 = perf_counter()
    n_total = 0

    try:
        while True:
            t_batch_start = perf_counter() - t0

            # raw_data=True returns the full (reps, nsweep_pts, readouts_per_exp) buffer
            # shape: (reps_per_batch, 2, 2)
            #   axis 0: rep index
            #   axis 1: sweep point  [0=f1, 1=f2]
            #   axis 2: readout      [0=MW on at f1/f2 → signal, 1=MW at 2000 MHz (off-resonance) → reference]
            raw = prog.acquire(raw_data=True)  # shape (reps, 2, 2), int64 ADC counts

            t_batch_end = perf_counter() - t0
            actual_batch_sec = t_batch_end - t_batch_start

            # Assign timestamps linearly across the batch
            times = t_batch_start + np.linspace(0, actual_batch_sec, reps_per_batch, endpoint=False)

            # Respect LockinODMR swap flag (analyze_data swaps axis 2 when True;
            # raw_data=True bypasses analyze_data so we replicate the swap here).
            if getattr(cfg, "odmr_lockin_swap_signal_reference", False):
                sig_idx, ref_idx = 1, 0
            else:
                sig_idx, ref_idx = 0, 1

            # Normalise by integration register value (same as LockinODMR.analyze_data)
            sig_f1  = raw[:, 0, sig_idx] / scale   # MW on at f1
            sig_f2  = raw[:, 1, sig_idx] / scale   # MW on at f2

            lockin = sig_f1 - sig_f2

            for i in range(reps_per_batch):
                all_rows.append({
                    "time_s":        float(times[i]),
                    "signal_f1":     float(sig_f1[i]),
                    "signal_f2":     float(sig_f2[i]),
                    "lockin_signal": float(lockin[i]),
                })

            n_total += reps_per_batch

            if duration_sec is not None and t_batch_end >= duration_sec:
                break

    except KeyboardInterrupt:
        print(f"\nLock-in stopped — {n_total} points collected.")

    df = pd.DataFrame(all_rows)

    if not df.empty:
        li = df["lockin_signal"]
        elapsed = float(df["time_s"].iloc[-1])
        eff_rate = n_total / elapsed if elapsed > 0 else 0.0
        print(f"\nSummary ({n_total} points in {elapsed:.1f} s):")
        print(f"  Effective sample rate:  {eff_rate:.0f} pts/s  (hardware switch: {switch_hz:.0f} Hz)")
        print(f"  f1={f1:.2f} MHz  signal mean: {df['signal_f1'].mean():.5f}")
        print(f"  f2={f2:.2f} MHz  signal mean: {df['signal_f2'].mean():.5f}")
        print(f"  Lock-in signal:  mean={li.mean():.5f}  std={li.std():.5f} ADC")

    if csv_filename is not None and not df.empty:
        df.to_csv(csv_filename, index=False)
        print(f"Saved {len(df)} rows to {csv_filename}")

    return df


def single_shot_twopoint(
    config,
    *,
    center_fMHz: float = 2874.0,
    fwhm_mhz: float = 5.71,
    slope_fraction: float = 0.5,
    reps: int = 1000,
) -> dict:
    """
    Single-shot two-point measurement — one averaged value at f1 and f2, then their difference.

    Like an ODMR sweep but with only 2 points. No time series, no streaming.
    Reference shot fires MW at 2000 MHz (off-resonance) so contrast = signal - off-res baseline.

    Returns
    -------
    dict with keys:
        f1_mhz, f2_mhz                  — the two frequencies used
        signal_f1, signal_f2            — MW-on PL averaged over `reps` reps
        reference_f1, reference_f2      — MW-at-2000-MHz PL averaged over `reps` reps
        contrast_f1, contrast_f2        — signal − reference at each frequency
        lockin_signal                   — signal_f1 − signal_f2  (raw, no reference subtraction)
        lockin_contrast                 — contrast_f1 − contrast_f2  (with off-res baseline removed)
    """
    f1 = center_fMHz - slope_fraction * fwhm_mhz
    f2 = center_fMHz + slope_fraction * fwhm_mhz

    cfg = copy(config)
    cfg.mw_start_fMHz = f1
    cfg.mw_end_fMHz = f2
    cfg.nsweep_points = 2
    cfg.reps = reps
    cfg.pre_init = getattr(config, "pre_init", True)
    cfg.odmr_reference_offres_mhz = 2000.0

    prog = qd.LockinODMR(cfg)
    d = prog.acquire(progress=False)

    sig = np.asarray(d.signal,    dtype=float)   # length 2: [f1, f2]
    ref = np.asarray(d.reference, dtype=float)
    con = np.asarray(d.contrast,  dtype=float)

    result = {
        "f1_mhz":          float(f1),
        "f2_mhz":          float(f2),
        "signal_f1":       float(sig[0]),
        "signal_f2":       float(sig[1]),
        "reference_f1":    float(ref[0]),
        "reference_f2":    float(ref[1]),
        "contrast_f1":     float(con[0]),
        "contrast_f2":     float(con[1]),
        "lockin_signal":   float(sig[0] - sig[1]),
        "lockin_contrast": float(con[0] - con[1]),
    }
    return result


def twopoint_lockin_from_odmr(odmr_result, config, **kwargs) -> pd.DataFrame:
    """
    Run two-point lock-in using center and FWHM extracted from a prior ODMR sweep.

    Parameters
    ----------
    odmr_result : ItemAttribute
        Result from LockinODMR.acquire() (has .frequencies, .reference, etc.).
    config : NVConfiguration
        Hardware configuration.
    **kwargs
        Forwarded to run_twopoint_lockin:
        slope_fraction, duration_sec, reps_per_batch, csv_filename.

    Returns
    -------
    pd.DataFrame — same as run_twopoint_lockin.
    """
    from odmr_sensitivity import estimate_odmr_sensitivity

    sens = estimate_odmr_sensitivity(odmr_result, trace="reference")
    ref_info = sens["reference"] if isinstance(sens, dict) and "reference" in sens else sens

    center = float(ref_info["center_mhz"])
    fwhm = float(ref_info["fwhm_mhz"])
    print(f"Lorentzian fit → center = {center:.2f} MHz,  FWHM = {fwhm:.2f} MHz")

    return run_twopoint_lockin(config, center_fMHz=center, fwhm_mhz=fwhm, **kwargs)
