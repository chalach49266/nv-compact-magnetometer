"""Mechanical plumbing shared by the three two-point acquisition cells.

Only the parts that are *not* tuning knobs live here -- the incremental CSV
writer, the per-batch row construction, the end-of-run report and the standard
three-panel figure. The acquisition loops themselves stay in the notebook, where
the parameters that get changed run to run are visible next to the code that
uses them.

Without this module Step 4A and Step 4B would be ~150 duplicated lines each, and
they would drift apart the first time one of them was edited.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


class IncrementalCsvWriter:
    """Append-only CSV writer: header once, then only the new rows.

    The previous live cell called `pd.DataFrame(rows).to_csv(...)` every 5 s,
    rewriting every row collected so far -- O(N^2) over a run. It stalled the
    loop for 25-220 ms at each 5 s boundary (46 of one session's 94 outlier
    batches sat on one) and produced the visible break in the burst-mode plots.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_written = 0

    def flush(self, rows: list) -> int:
        """Write rows[n_written:]; returns how many were appended."""
        if len(rows) == self.n_written:
            return 0
        chunk = pd.DataFrame(rows[self.n_written:])
        chunk.to_csv(self.path, mode="w" if self.n_written == 0 else "a",
                     header=(self.n_written == 0), index=False)
        n = len(chunk)
        self.n_written = len(rows)
        return n


def build_rows(batch: int, times_s: np.ndarray, acq_seconds: float, wall_epoch_s: float,
               signal: np.ndarray, reference: Optional[np.ndarray],
               signal_raw: np.ndarray, reference_raw: Optional[np.ndarray],
               spike_sig: np.ndarray, spike_ref: np.ndarray,
               frequencies_mhz: np.ndarray, z_minus: np.ndarray, z_plus: np.ndarray,
               delta_f_mhz: np.ndarray, f0_mhz: float, gamma_mhz_per_ut: float) -> list:
    """One dict per sample, in the schema every downstream tool expects.

    All arrays are (n_samples, n_freqs); `n_samples` is 1 in averaged mode and
    the rep count in burst mode, so one implementation covers both.
    """
    n_samples, n_freqs = signal.shape
    has_ref = reference is not None
    rows = []
    for k in range(n_samples):
        row = {
            "batch": batch,
            "sample": k,
            "time_s": float(times_s[k]),
            "timestamp_epoch_s": wall_epoch_s,
            "acq_seconds": float(acq_seconds),
        }
        for j in range(n_freqs):
            row[f"peak_{j+1:02d}"] = float(signal[k, j])
            row[f"peak_{j+1:02d}_freq_mhz"] = float(frequencies_mhz[j])
            row[f"peak_{j+1:02d}_raw"] = float(signal_raw[k, j])
            row[f"peak_{j+1:02d}_spike_sig"] = bool(spike_sig[k, j])
            if has_ref:
                row[f"peak_{j+1:02d}_ref"] = float(reference[k, j])
                row[f"peak_{j+1:02d}_ref_raw"] = float(reference_raw[k, j])
                row[f"peak_{j+1:02d}_spike_ref"] = bool(spike_ref[k, j])
        row.update({
            "z_minus": float(z_minus[k]),
            "z_plus": float(z_plus[k]),
            "lockin_signal": float(z_plus[k] - z_minus[k]),
            "delta_f_mhz": float(delta_f_mhz[k]),
            "peak_shift_kHz": float(delta_f_mhz[k]) * 1e3,
            "f_new_mhz": float(f0_mhz) + float(delta_f_mhz[k]),
            "B_shift_uT": float(delta_f_mhz[k]) / gamma_mhz_per_ut,
        })
        rows.append(row)
    return rows


def report_timing(acq_seconds_history, program, reps, n_watchdog=0) -> None:
    """Print the per-call budget and say how much of it the FPGA actually used.

    The point of this block: the call cost is a floor, not a sum. If the FPGA
    share is small, the rate cannot be improved by shortening the readout window
    -- but the averaging can be increased for free.
    """
    acq_ms = np.asarray(acq_seconds_history, dtype=float) * 1e3
    if acq_ms.size == 0:
        return
    fpga_ms = program.time_per_rep() * reps * 1e3
    floor_ms = float(np.percentile(acq_ms, 5))
    median_ms = float(np.median(acq_ms))

    print(f"\nPer-call timing (ms): fastest={floor_ms:.2f}  median={median_ms:.2f}  "
          f"p95={np.percentile(acq_ms, 95):.2f}  max={acq_ms.max():.2f}")
    print(f"  FPGA readout work   : {fpga_ms:.2f} ms "
          f"({100 * fpga_ms / median_ms:.0f}% of the median call)")
    if fpga_ms < floor_ms:
        fits = max(1, int(floor_ms / (program.time_per_rep() * 1e3)))
        print(f"  -> the call is host-bound. Shortening tau will NOT raise the rate.")
        if fits > reps:
            print(f"  -> FREE AVERAGING: {fits} reps fit in the same {floor_ms:.1f} ms. "
                  f"Raising {reps} -> {fits} costs no rate and buys up to "
                  f"{np.sqrt(fits / reps):.1f}x in sigma if the noise is white.")
    else:
        print(f"  -> the call is FPGA-bound; reps and tau now trade against rate directly.")
    if n_watchdog:
        print(f"  watchdog fired {n_watchdog} time(s) out of {acq_ms.size} calls "
              f"(host/network jitter; the FPGA work is constant).")


def report_freshness(program, n_calls: int, mode: str) -> None:
    """Say plainly how much of the run was a replay of the previous batch."""
    stale = int(getattr(program, "n_stale_acquires", 0))
    if not stale:
        print(f"\nData freshness: all {n_calls} acquires returned new data.")
        return
    dropped = int(getattr(program, "n_stale_dropped", 0))
    print(f"\nSTALE DATA: {stale} acquires returned a buffer identical to the previous "
          f"call ({100 * stale / max(1, stale + n_calls):.0f}% of attempts).")
    if dropped:
        print(f"  {dropped} were retried or discarded by the 'drop' policy, so they are "
              f"NOT in the CSV. The recorded rate is honest.")
    else:
        print(f"  Those samples are duplicates, not measurements. The recorded rate is "
              f"roughly {stale / max(1, n_calls) + 1:.1f}x the real one. Set "
              f"cfg.multipoint_on_stale = 'drop', clean the file with "
              f"scripts/clean_burst_lockin.py, or use the streaming mode.")
    if mode == "burst":
        print("  Note: forcing the tProc stop (reset_tproc=True) does not fix this. It "
              "was tried on 2026-08-14 and 50.0% of batches still replayed.")


def report_calibration_validity(df: pd.DataFrame, calib: dict) -> bool:
    """Check the parked points are still on the flank the calibration described.

    Both conversions assume the Step 1 calibration still describes the line. If
    the resonance moved, or the laser/MW level drifted, the measured dip depth at
    the parked points collapses and the shift estimate gets noisier in exact
    proportion, because sigma(delta_f) = sigma(z+ - z-) / |m- - m+| and the
    slopes come from the calibration.

    Returns True if the flank is well conditioned.
    """
    baseline = float(calib["baseline_fit"])
    live = {"f-": baseline - df["z_minus"].to_numpy(float),
            "f+": baseline - df["z_plus"].to_numpy(float)}
    cal = {"f-": baseline - float(calib["baseline_minus"]),
           "f+": baseline - float(calib["baseline_plus"])}

    print("\nCalibration validity - dip depth at the parked points:")
    ok = True
    for label in ("f-", "f+"):
        d, dc = live[label], cal[label]
        frac = d.mean() / dc if dc else float("nan")
        cond = d.mean() / d.std() if d.std() > 0 else float("inf")
        print(f"  {label}: calibrated {dc:.5f}   measured {d.mean():.5f} +/- {d.std():.5f}   "
              f"({100 * frac:.0f}% of calibrated, depth/noise = {cond:.1f})")
        if frac < 0.5:
            ok = False
            print(f"      WARNING: depth is {100 * frac:.0f}% of what the calibration "
                  f"expects. The peak has moved off the parked pair, or the laser/MW "
                  f"level drifted since the ODMR sweep. Re-run Steps 1a/1b.")
        if cond < 3:
            ok = False
            print(f"      WARNING: depth/noise = {cond:.1f}. The flank carries little "
                  f"signal, so the peak shift is correspondingly noisy.")
    if not ok:
        print("  -> Shift noise scales as sigma(z) / |m- - m+|, so a collapsed depth "
              "costs you directly. Fix the contrast before trusting the numbers.")
    return ok


def plot_run(df: pd.DataFrame, gamma_mhz_per_ut: float, title: str, calib=None):
    """The standard three-panel live figure: parked z, lock-in, peak shift."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(df["time_s"], df["z_minus"], lw=0.9, color="tab:orange", label="z at f-")
    axes[0].plot(df["time_s"], df["z_plus"], lw=0.9, color="tab:blue", label="z at f+")
    if calib is not None:
        axes[0].axhline(calib["baseline_minus"], color="tab:orange", ls=":", lw=0.8)
        axes[0].axhline(calib["baseline_plus"], color="tab:blue", ls=":", lw=0.8)
    axes[0].set_ylabel("normalised PL (z)")
    axes[0].legend(fontsize=8)

    axes[1].plot(df["time_s"], df["lockin_signal"], lw=0.9, color="tab:green")
    axes[1].set_ylabel("lock-in (z+ - z-)")

    axes[2].plot(df["time_s"], df["peak_shift_kHz"], lw=0.9, color="crimson")
    axes[2].axhline(0, color="0.4", lw=0.5)
    axes[2].set_ylabel("peak shift (kHz)")
    axes[2].set_xlabel("time (s)")
    ax_b = axes[2].secondary_yaxis(
        "right", functions=(lambda k: k * 1e-3 / gamma_mhz_per_ut,
                            lambda u: u * gamma_mhz_per_ut * 1e3))
    ax_b.set_ylabel("equivalent dB (uT)")

    for a in axes:
        a.grid(True, alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    return fig


def report_shift(df: pd.DataFrame, gamma_mhz_per_ut: float, dt_s: float) -> None:
    """Width of the shift trace and the sensitivity it implies."""
    import twopoint_spectra as spec

    x = df["peak_shift_kHz"].to_numpy(float)
    x = x[np.isfinite(x)]
    if not x.size:
        return
    print(f"\nPeak shift vs calibration f0: mean {x.mean():+.1f} kHz, std {x.std():.1f} kHz, "
          f"pk-pk {np.ptp(x):.1f} kHz")
    print(f"Equivalent field along this NV axis: std "
          f"{x.std() * 1e-3 / gamma_mhz_per_ut:.3f} uT, "
          f"pk-pk {np.ptp(x) * 1e-3 / gamma_mhz_per_ut:.3f} uT")
    print(f"White-noise sensitivity: eta = sigma*sqrt(2dt) = "
          f"{spec.sensitivity_nt_rthz(x.std(), dt_s, gamma_mhz_per_ut):.0f} nT/sqrt(Hz) "
          f"(only meaningful if the trace is white -- check the ASD in the analysis cell)")
    for warning in spec.alias_report(1.0 / dt_s):
        print(f"  ALIAS WARNING: {warning}")
