"""Row building and per-peak splitting for the four-point (two-peak) lock-in.

Four parked points are two peaks with two flanks each. Each peak is an ordinary
two-point estimator and is analysed by the existing `twopoint_postprocess`
pipeline unchanged -- `split_pair` is what makes that possible.

Why two peaks is worth four points
----------------------------------
The two dips of one NV axis move in opposite directions with field and in the
same direction with temperature and strain:

    f_low(t)  = D(t) - gamma * B_par(t)
    f_high(t) = D(t) + gamma * B_par(t)

so tracking both separates the two, which a single peak cannot do at all:

    B_par = (df_high - df_low) / (2 * gamma)    magnetic; D drift cancels
    dD    = (df_high + df_low) / 2              thermal/strain; field cancels

`combine()` computes both from a frame written by `build_rows()`. On a
single-peak run a slow thermal drift is indistinguishable from a slow field
change; here it is a separate column.

Cost
----
Twice the parked points is twice the rep time: at tau = 120 us,
t_rep = 4 x 120 = 480 us, so the streaming cadence is 2083 Hz rather than
4167 Hz, and the burst in-burst rate halves the same way. The averaged mode is
unaffected, because it is host-bound (see docs/twopoint_master_reference).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Structural columns that describe the sample rather than any one peak. They are
# copied verbatim into every per-peak view so the mode detection in
# `twopoint_postprocess.detect_mode` still works on the split frame.
SHARED_COLUMNS = ("batch", "sample", "packet", "rep_index", "reps_averaged",
                  "time_s", "timestamp_epoch_s", "acq_seconds")


def build_rows(batch: int, times_s: np.ndarray, acq_seconds: float, wall_epoch_s: float,
               signal: np.ndarray, reference: Optional[np.ndarray],
               signal_raw: np.ndarray, reference_raw: Optional[np.ndarray],
               spike_sig: np.ndarray, spike_ref: np.ndarray,
               frequencies_mhz: np.ndarray, pairs: Sequence[dict]) -> list:
    """One dict per sample, four parked points plus one block per peak.

    `pairs` is one dict per tracked peak, each carrying that peak's `z_minus`,
    `z_plus`, `delta_f_mhz` (all length n_samples) and its scalar `f0_mhz` and
    `gamma_nv_mhz_per_uT`. Peak k writes the columns `pk{k}_*`.

    The first peak's estimate is ALSO written to the bare `peak_shift_kHz` /
    `delta_f_mhz` / `z_minus` / `z_plus` names, so that any tool which expects a
    two-point file -- the Step 6 comparison, `scripts/analyze_twopoint_session.py`
    -- reads a four-point run without modification instead of failing on a
    missing column.
    """
    signal = np.atleast_2d(np.asarray(signal, dtype=float))
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

        for idx, pair in enumerate(pairs, start=1):
            p = f"pk{idx}"
            zm = float(np.asarray(pair["z_minus"], dtype=float).ravel()[k])
            zp = float(np.asarray(pair["z_plus"], dtype=float).ravel()[k])
            df = float(np.asarray(pair["delta_f_mhz"], dtype=float).ravel()[k])
            f0 = float(pair["f0_mhz"])
            gam = float(pair["gamma_nv_mhz_per_uT"])
            row.update({
                f"{p}_z_minus": zm,
                f"{p}_z_plus": zp,
                f"{p}_lockin_signal": zp - zm,
                f"{p}_delta_f_mhz": df,
                f"{p}_peak_shift_kHz": df * 1e3,
                f"{p}_f0_mhz": f0,
                f"{p}_f_new_mhz": f0 + df,
                f"{p}_B_shift_uT": df / gam,
            })

        first = f"pk1"
        row.update({
            "z_minus": row[f"{first}_z_minus"],
            "z_plus": row[f"{first}_z_plus"],
            "lockin_signal": row[f"{first}_lockin_signal"],
            "delta_f_mhz": row[f"{first}_delta_f_mhz"],
            "peak_shift_kHz": row[f"{first}_peak_shift_kHz"],
            "f_new_mhz": row[f"{first}_f_new_mhz"],
            "B_shift_uT": row[f"{first}_B_shift_uT"],
        })
        rows.append(row)
    return rows


def convert_pairs(signal, reference, calibs, z_from_counts, delta_f_linear,
                  zero_lockin=None) -> list:
    """Turn a (n_samples, 2*n_peaks) count array into one estimate per peak.

    `z_from_counts` and `delta_f_linear` are the notebook's own conversion
    helpers, passed in rather than imported: they close over the calibration
    scale that Step 1b built, and they already take a `calib` argument, so the
    two-point maths runs per peak with nothing changed.

    Parked point 2k-1 is peak k's low flank and 2k its high flank -- the order
    `emission_order()` puts them on the FPGA, and the order `split_pair` undoes.

    `zero_lockin` is a per-peak sequence (or None). Each peak carries its own
    offset because each pair is balanced on its own dip.
    """
    sig = np.atleast_2d(np.asarray(signal, dtype=float))
    ref = None if reference is None else np.atleast_2d(np.asarray(reference, dtype=float))
    n_peaks = len(calibs)
    if sig.shape[1] < 2 * n_peaks:
        raise ValueError(f"expected at least {2 * n_peaks} parked points for "
                         f"{n_peaks} peaks, got {sig.shape[1]}")

    out = []
    for k, calib in enumerate(calibs):
        lo, hi = 2 * k, 2 * k + 1
        zm = np.asarray(z_from_counts(sig[:, lo], None if ref is None else ref[:, lo],
                                      calib=calib), dtype=float)
        zp = np.asarray(z_from_counts(sig[:, hi], None if ref is None else ref[:, hi],
                                      calib=calib), dtype=float)
        z0 = None if zero_lockin is None else zero_lockin[k]
        df = np.asarray(delta_f_linear(zm, zp, calib=calib, zero_lockin=z0), dtype=float)
        out.append({"z_minus": zm, "z_plus": zp, "delta_f_mhz": df,
                    "f0_mhz": float(calib["f0_mhz"]),
                    "gamma_nv_mhz_per_uT": float(calib["gamma_nv_mhz_per_uT"])})
    return out


def measure_zero(acquire_fn, calibs, z_from_counts, n_batches: int = 20) -> list:
    """Per-peak auto-zero: the (z_plus - z_minus) each pair reads right now.

    Every mode re-zeroes on its own run rather than on the ODMR sweep, because
    dropping the per-shot reference leaves a constant offset that the sweep's
    balance point does not predict. With two peaks there are two offsets, and
    they are not the same number -- the pairs sit on different flanks of
    different lines.
    """
    acc = [[] for _ in calibs]
    for _ in range(int(n_batches)):
        d = acquire_fn()
        ref = d.reference if d.reference is not None else None
        sig = np.atleast_2d(np.asarray(d.signal, dtype=float))
        ref = None if ref is None else np.atleast_2d(np.asarray(ref, dtype=float))
        for k, calib in enumerate(calibs):
            lo, hi = 2 * k, 2 * k + 1
            zm = float(np.ravel(z_from_counts(sig[:, lo],
                                              None if ref is None else ref[:, lo],
                                              calib=calib))[0])
            zp = float(np.ravel(z_from_counts(sig[:, hi],
                                              None if ref is None else ref[:, hi],
                                              calib=calib))[0])
            acc[k].append(zp - zm)
    return [float(np.median(v)) for v in acc]


def n_pairs(df: pd.DataFrame) -> int:
    """How many peaks this frame tracks."""
    return sum(1 for k in range(1, 9) if f"pk{k}_delta_f_mhz" in df.columns)


def split_pair(df: pd.DataFrame, index: int) -> pd.DataFrame:
    """A two-point-shaped view of peak `index` (1-based).

    The point of this function is that nothing downstream has to learn about four
    points: `twopoint_postprocess.process()`, the despiker, the spectra and the
    figures all run on the result exactly as they do on a real two-point file.

    The two parked points of peak k are columns 2k-1 and 2k, which is the order
    `emission_order()` puts them on the FPGA.
    """
    p = f"pk{index}"
    if f"{p}_delta_f_mhz" not in df.columns:
        raise KeyError(f"frame has no peak {index}: found {n_pairs(df)} peak block(s). "
                       f"Columns like '{p}_delta_f_mhz' are written by "
                       f"fourpoint_runner.build_rows().")

    lo, hi = 2 * index - 1, 2 * index          # 1-based parked-point numbers
    out = pd.DataFrame(index=df.index)
    for col in SHARED_COLUMNS:
        if col in df.columns:
            out[col] = df[col]

    for new, old in ((1, lo), (2, hi)):
        for suffix in ("", "_freq_mhz", "_raw", "_ref", "_ref_raw",
                       "_spike_sig", "_spike_ref"):
            src = f"peak_{old:02d}{suffix}"
            if src in df.columns:
                out[f"peak_{new:02d}{suffix}"] = df[src]

    for name in ("z_minus", "z_plus", "lockin_signal", "delta_f_mhz",
                 "peak_shift_kHz", "f_new_mhz", "B_shift_uT"):
        out[name] = df[f"{p}_{name}"]
    return out


def split_difference(df: pd.DataFrame, low: int = 1, high: int = 2) -> pd.DataFrame:
    """A two-point-shaped view of the SPLITTING, `df_high - df_low`.

    The point is the same as `split_pair`: hand it to `twopoint_postprocess.process()`
    and the difference channel gets the whole pipeline -- despiking, Welch ASD,
    plain FFT, white floor -- with nothing taught about four points.

    What the columns mean here:

        peak_shift_kHz   the change in SPLITTING, f_high - f_low, in kHz
        z_minus/z_plus   the two peaks' own lock-in signals, so the middle panel
                         of `figure()` shows what each peak did while the
                         difference did whatever it did
        peak_01/peak_02  the raw counts of the two INNER parked points, kept so the
                         readout-quantum and PL-droop diagnostics still work

    Scale, and this matters when reading the spectra
    ------------------------------------------------
    The splitting moves at 2*gamma per unit field, not gamma:

        f_high - f_low = 2 * gamma * B_par

    so the nT axis that `process()` computes for this channel -- which divides by
    gamma -- reads **twice** the true field. Halve it, or use `combine()`, whose
    `B_par_uT` already has the 2 in it. The kHz axis is exact either way.
    """
    lo_p, hi_p = f"pk{low}", f"pk{high}"
    for pref in (lo_p, hi_p):
        if f"{pref}_delta_f_mhz" not in df.columns:
            raise KeyError(f"frame has no {pref}: found {n_pairs(df)} peak block(s)")

    out = pd.DataFrame(index=df.index)
    for col in SHARED_COLUMNS:
        if col in df.columns:
            out[col] = df[col]

    # The inner parked points of each peak: 2*low and 2*high-1.
    for new, old in ((1, 2 * low), (2, 2 * high - 1)):
        for suffix in ("", "_freq_mhz", "_raw", "_ref", "_ref_raw"):
            src = f"peak_{old:02d}{suffix}"
            if src in df.columns:
                out[f"peak_{new:02d}{suffix}"] = df[src]

    d = (np.asarray(df[f"{hi_p}_delta_f_mhz"], dtype=float)
         - np.asarray(df[f"{lo_p}_delta_f_mhz"], dtype=float))
    out["z_minus"] = df[f"{lo_p}_lockin_signal"]
    out["z_plus"] = df[f"{hi_p}_lockin_signal"]
    out["lockin_signal"] = out["z_plus"] - out["z_minus"]
    out["delta_f_mhz"] = d
    out["peak_shift_kHz"] = d * 1e3
    # Absolute splitting, so the trace can be read against the ODMR sweep.
    out["splitting_mhz"] = (np.asarray(df[f"{hi_p}_f_new_mhz"], dtype=float)
                            - np.asarray(df[f"{lo_p}_f_new_mhz"], dtype=float))
    out["f_new_mhz"] = out["splitting_mhz"]
    return out


def plot_run(df: pd.DataFrame, gamma_nv_mhz_per_ut: float, title: str,
             low: int = 1, high: int = 2, show: bool = True):
    """The three channels a four-point run produces, one panel each.

    peak 1, peak 2, and the difference -- not just whichever peak happens to sit
    in the bare `peak_shift_kHz` column.
    """
    import matplotlib.pyplot as plt

    t = df["time_s"].to_numpy(float)
    g = float(gamma_nv_mhz_per_ut)
    lo = np.asarray(df[f"pk{low}_peak_shift_kHz"], dtype=float)
    hi = np.asarray(df[f"pk{high}_peak_shift_kHz"], dtype=float)
    diff = hi - lo
    f0_lo = float(df[f"pk{low}_f0_mhz"].iloc[0])
    f0_hi = float(df[f"pk{high}_f0_mhz"].iloc[0])

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, y, colour, name, conv in (
            (axes[0], lo, "tab:orange", f"peak {low}  ({f0_lo:.1f} MHz)", g),
            (axes[1], hi, "tab:blue", f"peak {high}  ({f0_hi:.1f} MHz)", g),
            (axes[2], diff, "crimson",
             f"difference  (peak {high} - peak {low})", 2.0 * g)):
        ax.plot(t, y, lw=0.6, color=colour)
        ax.axhline(0.0, color="0.4", lw=0.5)
        ax.set_ylabel("shift (kHz)")
        ax.grid(True, alpha=0.3)
        sec = ax.secondary_yaxis(
            "right", functions=(lambda k, c=conv: k * 1e-3 / c,
                                lambda u, c=conv: u * c * 1e3))
        sec.set_ylabel("dB (uT)")
        ax.set_title(f"{name}   --   mean {y.mean():+.2f} kHz, "
                     f"std {y.std():.2f} kHz "
                     f"({y.std() * 1e-3 / conv * 1e3:.0f} nT)", fontsize=10)
    axes[2].set_xlabel("time (s)")
    axes[2].text(0.995, 0.03,
                 "the splitting moves at 2*gamma, so its uT axis is scaled by 2",
                 transform=axes[2].transAxes, ha="right", va="bottom",
                 fontsize=7.5, color="0.4")
    fig.suptitle(title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def combine(df: pd.DataFrame, gamma_nv_mhz_per_ut: float,
            low: int = 1, high: int = 2) -> pd.DataFrame:
    """Add the field and zero-field-splitting channels the second peak buys.

    `low`/`high` are the peak indices of the lower- and higher-frequency dip. The
    columns added are

        B_par_uT     (df_high - df_low) / (2 * gamma)   field, D drift cancelled
        dD_mhz       (df_high + df_low) / 2             D shift, field cancelled

    Both are differences of two simultaneously measured quantities, so anything
    common to the pair -- laser power, a slow thermal ramp, the host jitter --
    subtracts out of one of them by construction.
    """
    out = df.copy()
    lo = np.asarray(df[f"pk{low}_delta_f_mhz"], dtype=float)
    hi = np.asarray(df[f"pk{high}_delta_f_mhz"], dtype=float)
    out["B_par_uT"] = (hi - lo) / (2.0 * float(gamma_nv_mhz_per_ut))
    out["dD_mhz"] = 0.5 * (hi + lo)
    out["dD_khz"] = out["dD_mhz"] * 1e3
    return out
