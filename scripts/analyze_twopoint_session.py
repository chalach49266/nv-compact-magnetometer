#!/usr/bin/env python3
"""Regenerate every number and figure in the two-point timing/noise analysis.

The companion document
`docs/2026-08-06_twopoint_timing/TIMING_AND_NOISE_ANALYSIS.md` quotes a lot of
measured values. None of them are typed by hand: this script derives all of them
from the raw session CSVs and writes both the figures and the machine-readable
tables the document cites. Re-run it after any new session to refresh the
document, or against a different session folder to compare.

    python scripts/analyze_twopoint_session.py
    python scripts/analyze_twopoint_session.py --session data/twopoint_lockin/08062026
    python scripts/analyze_twopoint_session.py --outdir /tmp/check --no-figures

Five analyses, matching the document's sections:

  A  batch timing      -- where the per-acquire budget goes, and why the rate wobbles
  B  burst staleness   -- the duplicate-batch defect and its periodic artefact
  C  integration time  -- contrast, noise and sensitivity vs readout window
  D  noise structure   -- PSD, block averaging, autocorrelation
  E  common mode       -- how much of a PL excursion survives the two-point difference

Everything runs offline against committed data; no hardware is needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "notebook_modules"))

import burst_qc as qc  # noqa: E402

# NV gyromagnetic ratio along one axis, MHz per microtesla. Matches the value the
# live notebook writes into every calibration JSON as "gamma_nv_mhz_per_uT".
GAMMA_NV_MHZ_PER_UT = 0.028024

DEFAULT_SESSION = REPO_ROOT / "data" / "twopoint_lockin" / "08062026"
DEFAULT_OUTDIR = REPO_ROOT / "docs" / "2026-08-06_twopoint_timing"


# --------------------------------------------------------------------------- io


def find_live_csvs(session: Path, stamp_prefix: str) -> list[Path]:
    """Live CSVs recorded during the session itself, newest name last.

    The session folder also carries copies of earlier days' runs; those are not
    part of this session's timing story, so they are filtered out by timestamp.
    """
    # Three prefixes because Step 4A and Step 4B both used to write
    # `twopoint_lockin_live_*`, which made an averaged run and a burst run
    # indistinguishable by name. New runs write `_avg_` and `_burst_`; `_live_`
    # is kept so every already-recorded session still loads.
    hits = sorted(q for prefix in ("live", "avg", "burst")
                  for q in session.rglob(f"twopoint_lockin_{prefix}_{stamp_prefix}*.csv"))
    return [p for p in hits if not p.name.endswith(("_summary.csv",
                                                    "_peakshift_calibration.csv",
                                                    "_spectrum.csv"))]


def load_calibration(session: Path) -> dict:
    """The most recent two-point calibration in the session, as a plain dict.

    Supplies the slopes that convert a lock-in difference into a frequency shift.
    """
    import json

    cals = sorted(session.rglob("twopoint_calibration_odmr_sweep_*.json"))
    if not cals:
        raise FileNotFoundError(f"no calibration JSON under {session}")
    with open(cals[-1]) as fh:
        cal = json.load(fh)
    cal["_source"] = str(cals[-1].relative_to(session))
    return cal


def shift_denominator(cal: dict) -> float:
    """|m- - m+|, the z-per-MHz gain that turns lock-in noise into shift noise."""
    return abs(float(cal["slope_minus_per_mhz"]) - float(cal["slope_plus_per_mhz"]))


def khz_to_nt(khz: np.ndarray | float) -> np.ndarray | float:
    """Peak shift in kHz -> equivalent field along this NV axis, in nanotesla."""
    return np.asarray(khz) / GAMMA_NV_MHZ_PER_UT  # kHz / (MHz/uT) = nT


# ----------------------------------------------------------------- A: timing


def analyse_timing(paths: list[Path]) -> pd.DataFrame:
    """Per-run timing budget: period, acquire time, and the Python gap between them.

    `acq_seconds` is the wall-clock time inside `prog.acquire()`. The period is
    the start-to-start spacing of consecutive acquire calls, so
    `period - acq` isolates everything the live loop does outside the driver.
    """
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        table = qc.batch_table(df)
        if len(table.batch) < 3:
            continue

        period = np.diff(table.t_start) * 1e3               # ms
        acq = table.acq_seconds * 1e3                       # ms
        gap = period - acq[:-1]                             # ms outside acquire()

        rows.append({
            "file": path.name,
            "mode": "burst" if qc.is_burst(df) else "averaged",
            "n_rows": len(df),
            "n_batches": len(table.batch),
            "samples_per_batch": float(np.median(table.n_samples)),
            "duration_s": float(df["time_s"].iloc[-1]),
            "period_ms_p05": float(np.percentile(period, 5)),
            "period_ms_median": float(np.median(period)),
            "period_ms_p95": float(np.percentile(period, 95)),
            "period_ms_max": float(period.max()),
            "acq_ms_median": float(np.median(acq)),
            "acq_ms_p95": float(np.percentile(acq, 95)),
            "python_gap_ms_median": float(np.median(gap)),
            "python_gap_ms_p95": float(np.percentile(gap, 95)),
            "python_gap_ms_max": float(gap.max()),
            "rate_hz_slow": 1e3 / float(np.percentile(period, 95)),
            "rate_hz_median": 1e3 / float(np.median(period)),
            "rate_hz_fast": 1e3 / float(np.percentile(period, 5)),
            "stale_fraction": float(table.stale.mean()),
            # Robust width, so one glitch does not set the run's noise figure.
            "sigma_shift_khz": float(np.nanstd(df["peak_shift_kHz"])),
            "mad_shift_khz": float(1.4826 * np.nanmedian(np.abs(
                df["peak_shift_kHz"] - np.nanmedian(df["peak_shift_kHz"])))),
        })
    return pd.DataFrame(rows)


def analyse_flush_stalls(paths: list[Path], save_every_s: float = 5.0,
                         tol_s: float = 0.15) -> pd.DataFrame:
    """Test whether the slow batches line up with the periodic CSV rewrite.

    The live loop rewrites the whole DataFrame every `save_every_s` seconds, which
    costs more as the run grows. If that is what stalls the loop, the outlier
    batches should sit at multiples of `save_every_s` and nowhere else.
    """
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        table = qc.batch_table(df)
        if len(table.batch) < 10:
            continue
        period = np.diff(table.t_start)
        slow = np.where(period > 1.3 * np.median(period))[0]
        if slow.size == 0:
            rows.append({"file": path.name, "n_slow": 0, "n_at_flush": 0,
                         "fraction_at_flush": float("nan"),
                         "worst_stall_ms": float(period.max() * 1e3)})
            continue
        t_slow = table.t_start[slow]
        near = np.abs(t_slow - np.round(t_slow / save_every_s) * save_every_s) < tol_s
        near &= t_slow > save_every_s / 2          # t=0 is not a flush
        rows.append({
            "file": path.name,
            "n_slow": int(slow.size),
            "n_at_flush": int(near.sum()),
            "fraction_at_flush": float(near.mean()),
            "worst_stall_ms": float(period.max() * 1e3),
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------- B: staleness


def analyse_staleness(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify the duplicate-batch defect and the first-sample transient.

    Returns
    -------
    per_run : one row per burst file (stale fraction, cadence, duty cycle)
    profile : mean lock-in and shift at each sample index, split by batch health
    """
    per_run, profiles = [], []

    for path in paths:
        df = pd.read_csv(path)
        if not qc.is_burst(df):
            continue
        table = qc.batch_table(df)
        cadence = qc.fpga_cadence_seconds(df, table)
        stale_rows = qc.stale_sample_mask(df, table)
        first_rows = qc.first_sample_mask(df)

        shift = df["peak_shift_kHz"].to_numpy(float)
        real = ~stale_rows & ~first_rows

        per_run.append({
            "file": path.name,
            "n_batches": len(table.batch),
            "stale_batches": int(table.stale.sum()),
            "stale_fraction": float(table.stale.mean()),
            "dup_fraction_stale": float(np.mean(table.dup_fraction[table.stale])),
            "dup_fraction_real": float(np.mean(table.dup_fraction[~table.stale][1:]))
                                  if (~table.stale).sum() > 1 else float("nan"),
            "cadence_us": cadence * 1e6,
            "duty_cycle": qc.duty_cycle(table, cadence),
            "recorded_rate_hz": len(df) / float(df["time_s"].iloc[-1]),
            "true_rate_hz": int((~stale_rows).sum()) / float(df["time_s"].iloc[-1]),
            "sigma_shift_khz_as_recorded": float(np.nanstd(shift)),
            "sigma_shift_khz_real_only": float(np.nanstd(shift[real])),
            "row0_shift_khz_stale": float(np.nanmean(shift[first_rows & stale_rows])),
            "row0_shift_khz_real": float(np.nanmean(shift[first_rows & ~stale_rows])),
        })

        # Sample-index profile: isolates transients tied to position in the burst
        # rather than to time, which is how the row-0 artefact shows itself.
        sample = df["sample"].to_numpy()
        lock = df["lockin_signal"].to_numpy(float)
        for idx in range(0, min(int(sample.max()) + 1, 16)):
            for label, mask in (("stale", stale_rows), ("real", ~stale_rows)):
                sel = (sample == idx) & mask
                if sel.sum():
                    profiles.append({
                        "file": path.name, "sample": idx, "batch_kind": label,
                        "n": int(sel.sum()),
                        "lockin_mean": float(np.nanmean(lock[sel])),
                        "shift_khz_mean": float(np.nanmean(shift[sel])),
                        "shift_nt_mean": float(khz_to_nt(np.nanmean(shift[sel]))),
                    })

    return pd.DataFrame(per_run), pd.DataFrame(profiles)


# -------------------------------------------------- C: integration-time sweep


def _lorentzian_dip(f, amplitude, centre, fwhm, baseline):
    return baseline - amplitude / (1.0 + ((f - centre) / (fwhm / 2.0)) ** 2)


def calibrate_sweep_reps(session: Path, integ: pd.DataFrame,
                         tau_ref: float = 213.0) -> dict:
    """Recover how many reps each ODMR sweep point averaged, from the burst runs.

    The sweep CSVs do not record `reps`, and the file timestamps cannot supply it.
    But the burst runs measure a genuine single-rep sigma(delta_f) at tau = 213 us,
    which is one of the scanned windows. Averaging N reps reduces sigma by sqrt(N)
    for white noise, so

        sweep_reps = (sigma_1rep / sigma_sweep)^2   at the same tau

    Returns the inferred rep count and the two sigmas it came from, so the caller
    can show the inference rather than assert it.
    """
    row = integ[integ["tau_us"] == tau_ref]
    if row.empty:
        return {}

    sigmas = []
    for path in qc_find_burst_paths(session):
        df = pd.read_csv(path)
        table = qc.batch_table(df)
        keep = ~qc.stale_sample_mask(df, table) & ~qc.first_sample_mask(df)
        vals = df["peak_shift_kHz"].to_numpy(float)[keep]
        vals = vals[np.isfinite(vals)]
        if vals.size > 100:
            sigmas.append(float(vals.std()))
    if not sigmas:
        return {}

    sigma_1rep = float(np.median(sigmas))
    sigma_sweep = float(row.iloc[0]["sigma_df_khz"])
    ratio = sigma_1rep / sigma_sweep
    return {"tau_ref": tau_ref, "sigma_1rep_khz": sigma_1rep,
            "sigma_sweep_khz": sigma_sweep, "ratio": ratio,
            "sweep_reps": ratio ** 2, "n_burst_runs": len(sigmas)}


def analyse_integration_time(session: Path, sweep_reps: float = 1.0,
                             fit_lo: float = 2840.0, fit_hi: float = 2900.0,
                             off_lo: float = 2820.0, off_hi: float = 2980.0
                             ) -> pd.DataFrame:
    """Contrast, noise and sensitivity as a function of the ADC readout window.

    For each sweep: fit a Lorentzian dip over [fit_lo, fit_hi], and take the noise
    from the point-to-point scatter of the off-resonance wings (outside
    [off_lo, off_hi]), where successive differences are pure measurement noise.

    sigma(delta_f) = sqrt(2) * sigma_z / (2 * max_slope): two parked points on
    opposite flanks, each carrying independent noise sigma_z, differenced and
    divided by the combined slope.

    Turning that into a sensitivity needs BOTH time dependencies, not one:

      1. sigma_z falls with tau, because a longer window integrates more photons.
         That is already in the measured data.
      2. A sample costs time proportional to tau, so a longer window buys fewer
         samples per second. That enters as sqrt(time per sample).

    The trap is which *time per sample* pairs with the measured sigma_z. Each ODMR
    sweep point is an average over `sweep_reps` reps, so its sigma_z corresponds to
    `sweep_reps * t_rep` of integration, not to one rep. Pairing a reps-averaged
    sigma with a single-rep time understates eta by sqrt(sweep_reps) -- a factor of
    ~10 here. So:

        eta_per_rep = sigma(delta_f) * sqrt(sweep_reps) * sqrt(t_rep)

    `sweep_reps` is not recorded in the sweep CSVs and cannot be recovered from file
    timestamps (those gaps are dominated by how fast the operator clicked, not by
    acquisition time: the median gap is 4 s while t_point varies 4x). It is instead
    cross-calibrated against the burst runs, which measure a true single-rep sigma at
    tau = 213 us -- see `calibrate_sweep_reps`.

    Note that `eta_relative`, and the shape of the eta curve, are unaffected by
    sweep_reps ONLY IF it was held constant across the tau scan. That is the natural
    reading of a controlled one-variable scan, but it is an assumption, not a
    measurement, and the flat-band conclusion rests on it.
    """
    from scipy.optimize import curve_fit

    root = session / "Integration time" / "Integration Time Sweep"
    if not root.is_dir():
        return pd.DataFrame()

    rows = []
    for folder in sorted(root.iterdir(), key=lambda p: int(p.name.rstrip("us"))
                         if p.name.rstrip("us").isdigit() else 0):
        if not folder.is_dir() or not folder.name.endswith("us"):
            continue
        tau = int(folder.name[:-2])

        for csv in sorted(folder.glob("odmr_sweep_*.csv")):
            d = pd.read_csv(csv)
            f = d["frequency_MHz"].to_numpy(float)
            y = d["photoluminescence_mw_on_ADC"].to_numpy(float)

            wings = (f < off_lo) | (f > off_hi)
            if wings.sum() < 10:
                continue
            baseline = np.median(y[wings])
            # sigma from successive differences: immune to slow drift across the sweep
            sigma_counts = np.std(np.diff(y[wings])) / np.sqrt(2.0)
            z = y / abs(baseline)
            sigma_z = sigma_counts / abs(baseline)

            band = (f > fit_lo) & (f < fit_hi)
            if band.sum() < 10:
                continue
            try:
                guess = [z[band].max() - z[band].min(), float(f[band][np.argmin(z[band])]),
                         8.0, float(np.median(z[wings]))]
                popt, _ = curve_fit(_lorentzian_dip, f[band], z[band], p0=guess, maxfev=40000)
            except (RuntimeError, ValueError):
                continue
            amplitude, centre, fwhm, _ = popt
            amplitude, fwhm = abs(amplitude), abs(fwhm)
            if not (1.0 < fwhm < 100.0):
                continue

            # Steepest slope of a Lorentzian of depth A and width G.
            max_slope = amplitude * 3 * np.sqrt(3) / (4 * fwhm)
            sigma_df_khz = np.sqrt(2) * sigma_z / (2 * max_slope) * 1e3

            rows.append({
                "tau_us": tau, "file": csv.name,
                "contrast": amplitude, "fwhm_mhz": fwhm, "centre_mhz": centre,
                "sigma_z": sigma_z, "max_slope_per_mhz": max_slope,
                "sigma_df_khz": sigma_df_khz,
            })

    if not rows:
        return pd.DataFrame()

    per_sweep = pd.DataFrame(rows)

    # Per-rep cost: two parked frequencies, one readout each, two relax windows
    # per frequency. relax_delay_treg = 1000 -> 2.33 us on this tProc clock.
    relax_us = 2.33
    per_sweep["t_rep_us"] = 2 * (per_sweep["tau_us"] + 2 * relax_us)

    # Both time dependencies, per the docstring. sqrt(sweep_reps) converts the
    # reps-averaged sweep sigma into the single-rep sigma that pairs with t_rep;
    # sqrt(t_rep) is the time cost of one sample. Leaving sweep_reps at 1 gives a
    # RELATIVE figure of merit whose shape is right but whose scale is low by
    # sqrt(sweep_reps).
    per_sweep["eta_relative"] = (khz_to_nt(per_sweep["sigma_df_khz"])
                                 * np.sqrt(per_sweep["t_rep_us"] * 1e-6))
    per_sweep["eta_nt_rthz"] = per_sweep["eta_relative"] * np.sqrt(sweep_reps)

    out = per_sweep.groupby("tau_us").agg(
        n_sweeps=("file", "size"),
        contrast=("contrast", "median"),
        fwhm_mhz=("fwhm_mhz", "median"),
        sigma_z=("sigma_z", "median"),
        max_slope_per_mhz=("max_slope_per_mhz", "median"),
        sigma_df_khz=("sigma_df_khz", "median"),
        t_rep_us=("t_rep_us", "first"),
        eta_relative=("eta_relative", "median"),
        eta_nt_rthz=("eta_nt_rthz", "median"),
        # Sweep-to-sweep scatter at fixed tau. Without it the eta curve looks
        # like it has structure that repeat measurements do not support.
        eta_nt_rthz_std=("eta_nt_rthz", "std"),
    ).reset_index()

    out["rate_hz_23reps"] = 1e6 / (out["t_rep_us"] * 23)
    # Constant if the readout is averaging down as sqrt(time); rising means the
    # extra window length is buying drift instead of statistics.
    out["sigma_z_x_sqrt_tau"] = out["sigma_z"] * np.sqrt(out["tau_us"])
    return out


# ------------------------------------------------------------------ D: noise


def analyse_noise(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Noise spectrum, block-averaging behaviour and short-lag autocorrelation.

    The spectrum comes from the burst runs (real batches only), which are the only
    data with enough bandwidth to see above a few tens of Hz. The block-averaging
    curve answers the practical question: does raising the rep count actually buy
    sensitivity, or has correlated drift already taken over?
    """
    psd_rows, block_rows, acf_rows = [], [], []

    for path in paths:
        df = pd.read_csv(path)
        table = qc.batch_table(df)
        shift = df["peak_shift_kHz"].to_numpy(float)

        if qc.is_burst(df):
            cadence = qc.fpga_cadence_seconds(df, table)
            keep = ~qc.stale_sample_mask(df, table) & ~qc.first_sample_mask(df)
            _, segment = qc.retime(df, table, cadence)
            values, segs = shift[keep], segment[keep]
            fs = 1.0 / cadence

            try:
                freq, psd = qc.segment_psd(values, segs, fs)
            except ValueError:
                continue
            amp_nt = khz_to_nt(np.sqrt(psd))
            for lo, hi in [(1, 3), (3, 10), (10, 20), (20, 40), (40, 55),
                           (55, 65), (65, 100), (100, 200), (200, 400),
                           (400, 700), (700, 1100)]:
                band = (freq >= lo) & (freq < hi)
                if band.any():
                    psd_rows.append({"file": path.name, "f_lo": lo, "f_hi": hi,
                                     "amp_nt_rthz": float(np.median(amp_nt[band]))})

            blocks = qc.block_average_sigma(values, segs, [1, 2, 4, 8, 16, 23, 32, 64, 128])
            for _, r in blocks.iterrows():
                block_rows.append({"file": path.name, **r.to_dict()})
        else:
            values = shift[np.isfinite(shift)]

        x = values - np.mean(values)
        for lag in (1, 2, 3):
            if len(x) > lag + 2:
                acf_rows.append({
                    "file": path.name,
                    "mode": "burst" if qc.is_burst(df) else "averaged",
                    "lag": lag,
                    "autocorr": float(np.corrcoef(x[:-lag], x[lag:])[0, 1]),
                })

    return pd.DataFrame(psd_rows), pd.DataFrame(block_rows), pd.DataFrame(acf_rows)


# ------------------------------------------------------------ E: common mode


def analyse_common_mode(paths: list[Path]) -> pd.DataFrame:
    """How much of a common PL excursion survives into the two-point difference.

    Both parked points sit on the same line, so anything that scales the whole PL
    level -- laser drift, thermal motion, vibration -- moves z_minus and z_plus
    together and should cancel in their difference. The ratio of the two
    peak-to-peak swings is the achieved common-mode rejection.
    """
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        if not {"z_minus", "z_plus"}.issubset(df.columns):
            continue
        common = 0.5 * (df["z_minus"].to_numpy(float) + df["z_plus"].to_numpy(float))
        diff = df["lockin_signal"].to_numpy(float)
        # Smooth before taking the excursion so single-sample spikes do not set it.
        win = max(3, len(df) // 200)
        smooth = pd.Series(common).rolling(win, center=True, min_periods=1).mean().to_numpy()
        cm_ptp = float(np.ptp(smooth))
        rows.append({
            "file": path.name,
            "common_mode_ptp": cm_ptp,
            "common_mode_ptp_percent": 100 * cm_ptp / abs(np.median(common)),
            "difference_ptp": float(np.ptp(pd.Series(diff).rolling(
                win, center=True, min_periods=1).mean().to_numpy())),
            "sigma_difference": float(np.std(diff)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["rejection_ratio"] = out["common_mode_ptp"] / out["difference_ptp"]
    return out


# ---------------------------------------------------------------- figures


def make_figures(outdir: Path, timing: pd.DataFrame, stale: pd.DataFrame,
                 profile: pd.DataFrame, integ: pd.DataFrame,
                 psd: pd.DataFrame, blocks: pd.DataFrame,
                 session: Path, cal: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: where the acquire budget goes, and how it jitters ---
    averaged = timing[timing["mode"] == "averaged"]
    if not averaged.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax = axes[0]
        idx = np.arange(len(averaged))
        ax.errorbar(idx, averaged["period_ms_median"],
                    yerr=[averaged["period_ms_median"] - averaged["period_ms_p05"],
                          averaged["period_ms_p95"] - averaged["period_ms_median"]],
                    fmt="o", capsize=3, color="tab:blue", label="loop period (p05-p95)")
        ax.plot(idx, averaged["acq_ms_median"], "s", color="tab:orange",
                label="inside acquire()")
        ax.set_xticks(idx)
        ax.set_xticklabels([f[-10:-4] for f in averaged["file"]], rotation=60, fontsize=7)
        ax.set_ylabel("ms per sample")
        ax.set_title("Averaged mode: the loop is acquire()")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.bar(idx - 0.2, averaged["rate_hz_slow"], 0.2, label="p95 period (slowest)")
        ax.bar(idx, averaged["rate_hz_median"], 0.2, label="median")
        ax.bar(idx + 0.2, averaged["rate_hz_fast"], 0.2, label="p05 period (fastest)")
        ax.set_xticks(idx)
        ax.set_xticklabels([f[-10:-4] for f in averaged["file"]], rotation=60, fontsize=7)
        ax.set_ylabel("instantaneous rate (Hz)")
        ax.set_title("The 82-98 Hz wobble")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(figdir / "timing_budget.png", dpi=140)
        plt.close(fig)

    # --- Figure 2: the stale-batch defect ---
    if not stale.empty:
        example = pick_illustrative_burst(qc_find_burst_paths(session))
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))

        if example is not None:
            df = pd.read_csv(example)
            table = qc.batch_table(df)
            fig.suptitle(f"Burst-mode stale reads -- {example.name}", fontsize=11)
            ax = axes[0, 0]
            ax.plot(table.batch, table.acq_seconds * 1e3, "o-", ms=3, lw=0.8)
            ax.set_yscale("log")
            ax.set_xlabel("batch"); ax.set_ylabel("acquire (ms, log)")
            ax.set_title("Alternating fast/slow acquires")
            ax.grid(alpha=0.3, which="both")

            ax = axes[0, 1]
            ax.plot(table.batch, table.dup_fraction, "o-", ms=3, lw=0.8, color="crimson")
            ax.axhline(qc.STALE_DUPLICATE_FRACTION, ls="--", color="0.4",
                       label="stale threshold")
            ax.set_xlabel("batch"); ax.set_ylabel("rows identical to previous batch")
            ax.set_title("Every second batch is a replay")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)

            ax = axes[1, 0]
            stale_rows = qc.stale_sample_mask(df, table)
            first_rows = qc.first_sample_mask(df)
            t = df["time_s"].to_numpy()
            sh = df["peak_shift_kHz"].to_numpy()
            # Scatter, not lines: at ~4 kHz a line plot fills the axes solid and
            # hides which samples are the artefact.
            ax.scatter(t[~stale_rows], sh[~stale_rows], s=0.6, color="tab:blue",
                       alpha=0.35, label="real", rasterized=True)
            ax.scatter(t[stale_rows], sh[stale_rows], s=0.6, color="tab:red",
                       alpha=0.35, label="stale replay", rasterized=True)
            spikes = first_rows & stale_rows
            ax.scatter(t[spikes], sh[spikes], s=26, facecolors="none",
                       edgecolors="k", lw=0.8, label="row-0 transient")
            ax.set_xlabel("time (s)"); ax.set_ylabel("peak shift (kHz)")
            ax.set_title("As recorded, coloured by batch health")
            ax.legend(fontsize=8, markerscale=3); ax.grid(alpha=0.3)

        ax = axes[1, 1]
        for kind, colour in (("stale", "tab:red"), ("real", "tab:blue")):
            sel = profile[(profile["batch_kind"] == kind)]
            if sel.empty:
                continue
            agg = sel.groupby("sample")["shift_khz_mean"].mean()
            ax.plot(agg.index, agg.to_numpy(), "o-", ms=4, color=colour, label=kind)
        ax.axhline(0, color="0.4", lw=0.5)
        ax.set_xlabel("sample index within burst"); ax.set_ylabel("mean peak shift (kHz)")
        ax.set_title("The transient lives at row 0 of stale batches")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(figdir / "burst_staleness.png", dpi=140)
        plt.close(fig)

    # --- Figure 3: sensitivity vs readout window ---
    if not integ.empty:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        ax = axes[0]
        ax.plot(integ["tau_us"], integ["contrast"], "o-")
        ax.set_xlabel("readout window (us)"); ax.set_ylabel("fitted dip depth")
        ax.set_title("Contrast is flat: a longer\nwindow buys no signal")
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(integ["tau_us"], integ["sigma_z_x_sqrt_tau"], "o-", color="tab:orange")
        ax.set_xlabel("readout window (us)")
        ax.set_ylabel(r"$\sigma_z\,\sqrt{\tau}$")
        ax.set_title("Flat = averaging down.\nRising = collecting drift")
        ax.grid(alpha=0.3)

        ax = axes[2]
        ax.errorbar(integ["tau_us"], integ["eta_nt_rthz"],
                    yerr=integ["eta_nt_rthz_std"], fmt="o-", color="crimson",
                    capsize=3, ms=4, lw=1.0)
        flat = integ[integ["tau_us"] <= 140]["eta_nt_rthz"]
        ax.axhspan(flat.mean() - flat.std(), flat.mean() + flat.std(),
                   color="tab:green", alpha=0.15, label="flat band (tau <= 140 us)")
        ax.axvline(213, ls=":", color="tab:blue", label="current 213 us")
        ax.axvline(120, ls="--", color="0.4", label="new default 120 us")
        ax.set_xlabel("readout window (us)")
        ax.set_ylabel(r"$\eta$ (nT/$\sqrt{Hz}$)")
        ax.set_title("Sensitivity per unit time")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / "integration_time.png", dpi=140)
        plt.close(fig)

    # --- Figure 4: noise structure ---
    if not psd.empty or not blocks.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax = axes[0]
        for name, grp in psd.groupby("file"):
            centre = np.sqrt(grp["f_lo"] * np.maximum(grp["f_hi"], 1))
            ax.semilogx(centre, grp["amp_nt_rthz"], "o-", ms=4, label=name[-10:-4])
        ax.set_xlabel("frequency (Hz)"); ax.set_ylabel(r"nT/$\sqrt{Hz}$")
        ax.set_title("Noise is white above ~10 Hz")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

        ax = axes[1]
        for name, grp in blocks.groupby("file"):
            ax.loglog(grp["block"], grp["sigma"], "o-", ms=4, label=name[-10:-4])
            ax.loglog(grp["block"], grp["sigma_white_model"], ":", color="0.5")
        ax.set_xlabel("reps averaged"); ax.set_ylabel(r"$\sigma(\Delta f)$ (kHz)")
        ax.set_title("Dotted = ideal $1/\\sqrt{N}$")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(figdir / "noise_structure.png", dpi=140)
        plt.close(fig)


def qc_find_burst_paths(session: Path) -> list[Path]:
    """Burst-mode CSVs in the session, fewest batches first.

    The batch-by-batch figures need a run short enough to plot one marker per
    batch; a 200-batch run of strict alternation just fills in as a solid block.
    Ordering by batch count lets the caller take the most legible example.
    """
    scored = []
    for path in sorted(q for prefix in ("live", "avg", "burst")
                       for q in session.rglob(f"twopoint_lockin_{prefix}_*.csv")):
        if path.name.endswith(("_summary.csv", "_peakshift_calibration.csv", "_spectrum.csv")):
            continue
        try:
            head = pd.read_csv(path, nrows=4, usecols=["batch", "sample"])
        except (ValueError, KeyError):
            continue
        if not (head["sample"] > 0).any():
            continue
        n_batches = pd.read_csv(path, usecols=["batch"])["batch"].nunique()
        scored.append((n_batches, path))
    return [p for _, p in sorted(scored)]


def pick_illustrative_burst(paths: list[Path], want: int = 20) -> Path | None:
    """The shortest burst run that still has enough batches to show the pattern."""
    if not paths:
        return None
    for path in paths:
        if pd.read_csv(path, usecols=["batch"])["batch"].nunique() >= want:
            return path
    return paths[-1]


# -------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path, default=DEFAULT_SESSION,
                    help="session data folder (default: 2026-08-06)")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                    help="where tables and figures are written")
    ap.add_argument("--stamp", default="20260806",
                    help="only analyse runs whose timestamp starts with this")
    ap.add_argument("--no-figures", action="store_true", help="tables only")
    args = ap.parse_args()

    session, outdir = args.session, args.outdir
    if not session.is_dir():
        print(f"error: session folder not found: {session}", file=sys.stderr)
        return 1
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    paths = find_live_csvs(session, args.stamp)
    if not paths:
        print(f"error: no live CSVs matching {args.stamp} under {session}", file=sys.stderr)
        return 1
    cal = load_calibration(session)

    print(f"session : {session}")
    print(f"runs    : {len(paths)} live CSVs matching {args.stamp}")
    print(f"calib   : {cal['_source']}  f0={cal['f0_mhz']:.4f} MHz  "
          f"FWHM={cal['fwhm_mhz']:.3f} MHz  |m- - m+|={shift_denominator(cal):.6f} /MHz")
    print()

    timing = analyse_timing(paths)
    flush = analyse_flush_stalls(paths)
    stale, profile = analyse_staleness(paths)
    integ = analyse_integration_time(session)
    reps_cal = calibrate_sweep_reps(session, integ)
    if reps_cal:
        integ = analyse_integration_time(session, sweep_reps=reps_cal["sweep_reps"])
    psd, blocks, acf = analyse_noise(paths)
    common = analyse_common_mode(paths)

    tables = {
        "timing": timing, "flush_stalls": flush, "staleness": stale,
        "burst_sample_profile": profile, "integration_time": integ,
        "noise_psd": psd, "noise_block_average": blocks, "autocorrelation": acf,
        "common_mode": common,
    }
    for name, frame in tables.items():
        if frame.empty:
            print(f"[skip] {name}: no rows")
            continue
        frame.to_csv(outdir / "tables" / f"{name}.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 40)

    print("=" * 78)
    print("A. TIMING -- averaged mode")
    print("=" * 78)
    av = timing[timing["mode"] == "averaged"]
    if not av.empty:
        print(av[["file", "n_batches", "period_ms_median", "acq_ms_median",
                  "python_gap_ms_median", "rate_hz_slow", "rate_hz_median",
                  "rate_hz_fast"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\n  median period      {av['period_ms_median'].median():.3f} ms "
              f"-> {1e3 / av['period_ms_median'].median():.1f} Hz")
        print(f"  median acquire     {av['acq_ms_median'].median():.3f} ms")
        print(f"  median Python gap  {av['python_gap_ms_median'].median():.3f} ms "
              f"({100 * av['python_gap_ms_median'].median() / av['period_ms_median'].median():.1f}% of the period)")
        print(f"  rate spans         {av['rate_hz_slow'].min():.1f} - {av['rate_hz_fast'].max():.1f} Hz")
        print(f"  batches inspected  {int(av['n_batches'].sum())}, "
              f"stale detected {int((av['stale_fraction'] * av['n_batches']).sum())}")
        print(f"  sigma(shift)       {av['sigma_shift_khz'].min():.0f} - "
              f"{av['sigma_shift_khz'].max():.0f} kHz "
              f"(MAD-based {av['mad_shift_khz'].min():.0f} - {av['mad_shift_khz'].max():.0f} kHz)")
    if not flush.empty:
        f_av = flush[flush["file"].isin(av["file"])]
        if not f_av.empty and f_av["n_slow"].sum():
            print(f"\n  slow batches at a 5 s CSV-flush boundary: "
                  f"{int(f_av['n_at_flush'].sum())}/{int(f_av['n_slow'].sum())}")
            print(f"  worst stall observed: {f_av['worst_stall_ms'].max():.0f} ms")

    print()
    print("=" * 78)
    print("B. BURST STALENESS")
    print("=" * 78)
    if not stale.empty:
        print(stale[["file", "n_batches", "stale_batches", "stale_fraction",
                     "cadence_us", "duty_cycle", "recorded_rate_hz", "true_rate_hz",
                     "sigma_shift_khz_as_recorded", "sigma_shift_khz_real_only",
                     "row0_shift_khz_stale", "row0_shift_khz_real"]
                    ].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\n  duplicate fraction: {stale['dup_fraction_stale'].mean():.3f} in stale "
              f"batches vs {stale['dup_fraction_real'].mean():.3f} in real ones")
        print(f"  row-0 transient: {stale['row0_shift_khz_stale'].mean():+.0f} kHz "
              f"({khz_to_nt(stale['row0_shift_khz_stale'].mean()) / 1e3:+.1f} uT) in stale batches, "
              f"{stale['row0_shift_khz_real'].mean():+.0f} kHz in real ones")

    print()
    print("=" * 78)
    print("C. INTEGRATION TIME")
    print("=" * 78)
    if not integ.empty:
        if reps_cal:
            print(f"  Sweep points are reps-averaged, and `reps` is not recorded in the CSVs.")
            print(f"  Cross-calibrated against {reps_cal['n_burst_runs']} burst runs at "
                  f"tau = {reps_cal['tau_ref']:.0f} us:")
            print(f"    single-rep sigma (burst)  {reps_cal['sigma_1rep_khz']:7.1f} kHz")
            print(f"    sweep-point sigma         {reps_cal['sigma_sweep_khz']:7.1f} kHz")
            print(f"    ratio {reps_cal['ratio']:.2f} -> sweep averaged "
                  f"~{reps_cal['sweep_reps']:.0f} reps per point")
            print(f"  eta below is per-rep and includes that sqrt({reps_cal['sweep_reps']:.0f}) "
                  f"= {np.sqrt(reps_cal['sweep_reps']):.1f}x factor.\n")
        print(integ[["tau_us", "n_sweeps", "contrast", "fwhm_mhz", "sigma_z",
                     "sigma_z_x_sqrt_tau", "sigma_df_khz", "t_rep_us",
                     "eta_relative", "eta_nt_rthz", "eta_nt_rthz_std", "rate_hz_23reps"]
                    ].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        # The eta curve has real sweep-to-sweep scatter, so the useful statement
        # is which band is flat within that scatter -- not which single tau won.
        lo = integ[integ["tau_us"] <= 140]
        hi = integ[(integ["tau_us"] > 140) & (integ["tau_us"] < 213)]
        print(f"\n  eta, tau <= 140 us : {lo['eta_nt_rthz'].mean():.1f} +/- "
              f"{lo['eta_nt_rthz'].std():.1f} nT/rtHz over {len(lo)} windows")
        print(f"  eta, 150-200 us    : {hi['eta_nt_rthz'].mean():.1f} +/- "
              f"{hi['eta_nt_rthz'].std():.1f} nT/rtHz over {len(hi)} windows")
        print(f"  typical scatter at fixed tau: +/- {integ['eta_nt_rthz_std'].median():.1f} nT/rtHz")

        for tau in (120, 213):
            row = integ[integ["tau_us"] == tau]
            if not row.empty:
                r = row.iloc[0]
                print(f"  tau = {tau:3d} us -> {r['eta_nt_rthz']:.1f} nT/rtHz, "
                      f"{r['t_rep_us']:.0f} us/rep, {r['rate_hz_23reps']:.0f} Hz at 23 reps")
        pair = integ[integ["tau_us"].isin([120, 213])]
        if len(pair) == 2:
            a, b = pair.iloc[0], pair.iloc[1]
            print(f"  -> 213 -> 120 us is {b['t_rep_us'] / a['t_rep_us']:.2f}x faster, "
                  f"and the eta difference ({b['eta_nt_rthz']:.1f} vs {a['eta_nt_rthz']:.1f}) "
                  f"is inside the {integ['eta_nt_rthz_std'].median():.1f} nT/rtHz scatter")

        flat = integ[integ["tau_us"] <= 140]["sigma_z_x_sqrt_tau"]
        rise = integ[integ["tau_us"] > 140]["sigma_z_x_sqrt_tau"]
        if len(flat) and len(rise):
            print(f"  sigma_z*sqrt(tau): {flat.mean():.4f} below 140 us, "
                  f"{rise.mean():.4f} above -> readout stops averaging down past ~140 us")
        print("  CAVEAT: the shape of this curve assumes `reps` was held constant across")
        print("  the tau scan. That is the natural reading of a one-variable scan, but it")
        print("  is not recorded and cannot be recovered from the files. A reps change")
        print("  would distort the curve; the rig check at tau = 120 us tests the")
        print("  conclusion directly.")

    print()
    print("=" * 78)
    print("D. NOISE STRUCTURE")
    print("=" * 78)
    if not psd.empty:
        agg = psd.groupby(["f_lo", "f_hi"])["amp_nt_rthz"].median().reset_index()
        print(agg.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
        high = agg[agg["f_lo"] >= 10]["amp_nt_rthz"]
        print(f"\n  white floor above 10 Hz: {high.median():.0f} nT/rtHz "
              f"(spread {high.min():.0f}-{high.max():.0f})")
    if not blocks.empty:
        agg = blocks.groupby("block")[["sigma", "sigma_white_model", "excess"]].mean().reset_index()
        print("\n  block averaging within bursts:")
        print(agg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if not acf.empty:
        print("\n  lag-1 autocorrelation by mode:")
        print(acf[acf["lag"] == 1].groupby("mode")["autocorr"].describe()[
            ["count", "mean", "min", "max"]].to_string(float_format=lambda v: f"{v:.3f}"))

    print()
    print("=" * 78)
    print("E. COMMON-MODE REJECTION")
    print("=" * 78)
    if not common.empty:
        top = common.sort_values("common_mode_ptp_percent", ascending=False).head(5)
        print(top[["file", "common_mode_ptp_percent", "common_mode_ptp",
                   "difference_ptp", "rejection_ratio"]
                  ].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if not args.no_figures:
        make_figures(outdir, timing, stale, profile, integ, psd, blocks, session, cal)
        print(f"\nfigures -> {outdir / 'figures'}")
    print(f"tables  -> {outdir / 'tables'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
