#!/usr/bin/env python3
"""Recover a usable time series from a burst-mode two-point lock-in CSV.

Burst runs recorded before the 2026-08-12 fixes contain three artefacts that are
not measurements, all traced in `docs/2026-08-06_twopoint_timing/`:

  1. **Stale batches.** The accumulated buffer was read before the new run refilled
     it, so every second acquire returned a bit-identical replay of the previous
     one. Half the samples are duplicates, which doubles the apparent rate and
     makes every real glitch appear twice about one burst period apart -- the
     "smaller, less periodic" spikes.
  2. **A row-0 transient.** Row 0 of a stale batch is not a copy; it reads several
     percent high and lands at roughly +1900 kHz (+68 uT). This is the large,
     strictly periodic spike.
  3. **A broken time axis.** Samples were stamped across the *measured* acquire
     window, which alternates ~13 ms and ~425 ms, so half the data is compressed
     33x in time. The dead time between bursts also appears as a gap in the trace
     -- the visible "break" -- widened by the periodic CSV rewrite.

This script removes all three and then applies the existing causal Hampel filter
to whatever impulsive noise is left. What it cannot do is invent the samples lost
to the dead time between bursts: it recovers a correct ~2.1 kHz record from a
file that claimed 4.4 kHz, it does not recover 4.4 kHz.

    python scripts/clean_burst_lockin.py <live_csv>
    python scripts/clean_burst_lockin.py <live_csv> --no-despike --outdir analysis/
    python scripts/clean_burst_lockin.py <live_csv> --k-sigma 5 --window 21

Outputs `<name>_clean.csv`, `<name>_clean.png` and a before/after noise report.
The cleaned file is self-describing: `segment` marks contiguous acquisitions and
must never be filtered or transformed across, and `time_s` is rebuilt on the FPGA
cadence rather than on host arrival times.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "notebook_modules"))

import burst_qc as qc                      # noqa: E402
from spike_rejection import HampelDespiker  # noqa: E402

GAMMA_NV_MHZ_PER_UT = 0.028024


# ------------------------------------------------------- conversion recovery


def recover_conversion(df: pd.DataFrame) -> dict:
    """Recover the counts->z scale and the z->frequency transform from the file.

    Both are exact linear maps applied by the live cell, so they can be read back
    off the recorded columns without needing the calibration JSON:

        z            = counts / ref_norm_counts
        lockin       = z_plus - z_minus
        delta_f_mhz  = (lockin - zero) / denom

    Fitting `lockin` against `delta_f_mhz` returns `denom` and `zero` to machine
    precision (residuals ~1e-16 on the 2026-08-06 runs). Doing it this way means the
    cleaner works on any live CSV, including ones whose calibration file has moved.
    """
    scale = float(np.median(df["peak_01"].to_numpy() / df["z_minus"].to_numpy()))

    x = df["delta_f_mhz"].to_numpy(float)
    y = df["lockin_signal"].to_numpy(float)
    good = np.isfinite(x) & np.isfinite(y)
    denom, zero = np.polyfit(x[good], y[good], 1)
    residual = float(np.max(np.abs(y[good] - (denom * x[good] + zero))))

    return {"counts_per_z": scale, "denom": float(denom), "zero": float(zero),
            "fit_residual": residual}


def recompute_from_counts(df: pd.DataFrame, conv: dict) -> pd.DataFrame:
    """Rebuild z, lock-in and shift columns from the (possibly cleaned) raw counts."""
    out = df.copy()
    out["z_minus"] = out["peak_01"] / conv["counts_per_z"]
    out["z_plus"] = out["peak_02"] / conv["counts_per_z"]
    out["lockin_signal"] = out["z_plus"] - out["z_minus"]
    out["delta_f_mhz"] = (out["lockin_signal"] - conv["zero"]) / conv["denom"]
    out["peak_shift_kHz"] = out["delta_f_mhz"] * 1e3
    out["B_shift_uT"] = out["delta_f_mhz"] / GAMMA_NV_MHZ_PER_UT
    if "f_new_mhz" in out.columns:
        f0 = float((df["f_new_mhz"] - df["delta_f_mhz"]).median())
        out["f_new_mhz"] = f0 + out["delta_f_mhz"]
    return out


# ----------------------------------------------------------------- despiking


def robust_sigma(values: np.ndarray) -> float:
    """Noise scale from successive differences -- unaffected by drift or offsets."""
    d = np.diff(values[np.isfinite(values)])
    if d.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def despike_by_segment(df: pd.DataFrame, columns: list[str], *, window: int,
                       k_sigma: float, sigma_floor: float | None,
                       sigma_cap: float | None) -> tuple[pd.DataFrame, np.ndarray]:
    """Run the causal Hampel filter within each segment, never across a gap.

    `HampelDespiker` is stateful over a trailing window, so feeding it across the
    dead time between bursts would compare samples separated by tens of ms as if
    they were adjacent. Each segment gets its own filter instance.

    The filter's default sigma_floor/sigma_cap (1.5 / 2.5 ADC) were tuned for a
    setup whose per-channel noise was ~1 ADC. On the 2026-08-06 burst runs it is
    ~11 ADC, and leaving the cap at 2.5 would clamp the threshold far below the
    real noise and reject almost every sample. So unless the caller overrides them,
    both are derived from this file's own robust noise estimate.
    """
    sigmas = [robust_sigma(df[c].to_numpy(float)) for c in columns]
    typical = float(np.median(sigmas)) if sigmas else 1.0
    floor = 0.5 * typical if sigma_floor is None else sigma_floor
    cap = 3.0 * typical if sigma_cap is None else sigma_cap

    out = df.copy()
    flags = np.zeros((len(df), len(columns)), dtype=bool)
    values = df[columns].to_numpy(float)

    for seg in np.unique(df["segment"].to_numpy()):
        rows = np.where(df["segment"].to_numpy() == seg)[0]
        despiker = HampelDespiker(n_channels=len(columns), window=window,
                                  k_sigma=k_sigma, sigma_floor=floor, sigma_cap=cap)
        for r in rows:
            clean, flag = despiker.update(values[r])
            values[r] = clean
            flags[r] = flag

    for i, c in enumerate(columns):
        out[c] = values[:, i]
        out[f"{c}_despiked"] = flags[:, i]
    return out, flags, {"sigma_typical": typical, "sigma_floor": floor, "sigma_cap": cap}


# --------------------------------------------------------------------- report


def noise_report(shift_khz: np.ndarray, segment: np.ndarray, cadence_s: float,
                 label: str) -> dict:
    """Width and per-root-Hz sensitivity of a shift series."""
    finite = np.isfinite(shift_khz)
    x = shift_khz[finite]
    if x.size < 2:
        return {"label": label, "n": int(x.size)}
    mad = 1.4826 * np.median(np.abs(x - np.median(x)))
    return {
        "label": label,
        "n": int(x.size),
        "std_khz": float(x.std()),
        "mad_khz": float(mad),
        "ptp_khz": float(np.ptp(x)),
        "std_uT": float(x.std() * 1e-3 / GAMMA_NV_MHZ_PER_UT),
        "eta_nT_rtHz": float(x.std() / GAMMA_NV_MHZ_PER_UT * np.sqrt(cadence_s)),
    }


def make_plot(raw: pd.DataFrame, clean: pd.DataFrame, path: Path,
              stale_rows: np.ndarray, cadence_s: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(13, 9))

    ax = axes[0]
    ax.scatter(raw["time_s"][~stale_rows], raw["peak_shift_kHz"][~stale_rows],
               s=0.5, color="tab:blue", alpha=0.35, label="real", rasterized=True)
    ax.scatter(raw["time_s"][stale_rows], raw["peak_shift_kHz"][stale_rows],
               s=0.5, color="tab:red", alpha=0.35, label="stale replay", rasterized=True)
    ax.set_ylabel("peak shift (kHz)")
    ax.set_title(f"BEFORE -- as recorded ({len(raw)} rows, "
                 f"{100 * stale_rows.mean():.0f}% stale)")
    ax.legend(fontsize=8, markerscale=6); ax.grid(alpha=0.3)

    ax = axes[1]
    for seg in np.unique(clean["segment"]):
        sub = clean[clean["segment"] == seg]
        ax.plot(sub["time_s"], sub["peak_shift_kHz"], lw=0.5, color="tab:green")
    ax.set_ylabel("peak shift (kHz)")
    ax.set_title(f"AFTER -- stale batches and row-0 transients removed, "
                 f"re-timed on the {cadence_s * 1e6:.0f} us cadence "
                 f"({len(clean)} rows, gaps left as gaps)")
    ax.grid(alpha=0.3)
    ax.set_xlim(axes[0].get_xlim())

    ax = axes[2]
    lo = min(raw["peak_shift_kHz"].quantile(0.001), clean["peak_shift_kHz"].quantile(0.001))
    hi = max(raw["peak_shift_kHz"].quantile(0.999), clean["peak_shift_kHz"].quantile(0.999))
    bins = np.linspace(lo, hi, 200)
    # Density, not counts: cleaning removes about half the rows, so raw counts would
    # show the "after" curve below the "before" one everywhere and say nothing about
    # the tail, which is the part that matters.
    ax.hist(raw["peak_shift_kHz"], bins=bins, histtype="step", color="tab:red",
            density=True,
            label=f"before (std {raw['peak_shift_kHz'].std():.0f} kHz, "
                  f"pk-pk {np.ptp(raw['peak_shift_kHz']):.0f})")
    ax.hist(clean["peak_shift_kHz"], bins=bins, histtype="step", color="tab:green",
            density=True,
            label=f"after (std {clean['peak_shift_kHz'].std():.0f} kHz, "
                  f"pk-pk {np.ptp(clean['peak_shift_kHz']):.0f})")
    ax.set_yscale("log")
    ax.set_xlabel("peak shift (kHz)"); ax.set_ylabel("density")
    ax.set_title("Distribution -- the artefact tail is what should disappear")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(path.stem.replace("_clean", ""), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="twopoint_lockin_live_*.csv to clean")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="where to write outputs (default: next to the input)")
    ap.add_argument("--keep-first-sample", action="store_true",
                    help="do not drop sample 0 of each batch")
    ap.add_argument("--keep-stale", action="store_true",
                    help="do not drop duplicate batches (diagnostic only)")
    ap.add_argument("--no-despike", action="store_true",
                    help="skip the Hampel stage; only remove structural artefacts")
    ap.add_argument("--window", type=int, default=11, help="Hampel trailing window")
    ap.add_argument("--k-sigma", type=float, default=4.0, help="Hampel threshold")
    ap.add_argument("--sigma-floor", type=float, default=None,
                    help="override the auto-derived sigma floor (ADC counts)")
    ap.add_argument("--sigma-cap", type=float, default=None,
                    help="override the auto-derived sigma cap (ADC counts)")
    args = ap.parse_args()

    if not args.csv.is_file():
        print(f"error: no such file: {args.csv}", file=sys.stderr)
        return 1
    outdir = args.outdir or args.csv.parent
    outdir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.csv)
    print(f"input   : {args.csv.name}  ({len(raw)} rows)")

    if not qc.is_burst(raw):
        print("\nThis file has one sample per acquire (averaged mode), not burst mode.")
        print("The stale-batch and row-0 artefacts are specific to burst mode, and the")
        print("2026-08-06 averaged runs showed none of them. Nothing to clean.")
        return 0

    table = qc.batch_table(raw)
    cadence = qc.fpga_cadence_seconds(raw, table)
    conv = recover_conversion(raw)
    print(f"cadence : {cadence * 1e6:.1f} us/rep  ({1 / cadence:.0f} Hz raw)")
    print(f"transform recovered from the file: counts/z = {conv['counts_per_z']:.4f}, "
          f"denom = {conv['denom']:.8f}, zero = {conv['zero']:+.3e} "
          f"(fit residual {conv['fit_residual']:.1e})")
    if conv["fit_residual"] > 1e-9:
        print("  WARNING: the lockin/delta_f relation is not exactly linear in this file; "
              "recomputed shifts may differ slightly from the recorded ones.")

    stale_rows = qc.stale_sample_mask(raw, table)
    first_rows = qc.first_sample_mask(raw)
    duration = float(raw["time_s"].iloc[-1])

    print(f"\nstage 1 -- stale batches : {int(table.stale.sum())}/{len(table.batch)} "
          f"({100 * table.stale.mean():.0f}%), {int(stale_rows.sum())} rows")
    print(f"stage 2 -- row-0 transient: {int(first_rows.sum())} rows "
          f"(mean shift {np.nanmean(raw['peak_shift_kHz'][first_rows & stale_rows]):+.0f} kHz "
          f"in stale batches)")

    keep = np.ones(len(raw), dtype=bool)
    if not args.keep_stale:
        keep &= ~stale_rows
    if not args.keep_first_sample:
        keep &= ~first_rows

    clean = raw[keep].reset_index(drop=True)
    if clean.empty:
        print("error: nothing survived the filters", file=sys.stderr)
        return 1

    # Stage 3: rebuild the time axis on the cadence and mark contiguous acquisitions.
    # Reuse the table built from the UNFILTERED frame. Recomputing it here would derive
    # each batch's start from its first surviving row -- which is sample 1, not sample 0,
    # once the transient is dropped -- and shift every batch by half a sample.
    times, segment = qc.retime(clean, table, cadence)
    clean["time_s"] = times
    clean["segment"] = segment
    print(f"stage 3 -- re-timed on the cadence; {len(np.unique(segment))} contiguous "
          f"segments, dead time between them left as gaps")

    # Stage 4: impulsive noise, per segment.
    if args.no_despike:
        print("stage 4 -- despiking skipped (--no-despike)")
        flags = np.zeros((len(clean), 2), dtype=bool)
    else:
        cols = qc.peak_columns(clean)
        clean, flags, scales = despike_by_segment(
            clean, cols, window=args.window, k_sigma=args.k_sigma,
            sigma_floor=args.sigma_floor, sigma_cap=args.sigma_cap)
        clean = recompute_from_counts(clean, conv)
        print(f"stage 4 -- Hampel (window={args.window}, k={args.k_sigma}, "
              f"sigma floor/cap {scales['sigma_floor']:.1f}/{scales['sigma_cap']:.1f} ADC "
              f"from a measured {scales['sigma_typical']:.1f} ADC): "
              f"{int(flags.sum())} of {flags.size} cells replaced "
              f"({100 * flags.mean():.2f}%)")

    before = noise_report(raw["peak_shift_kHz"].to_numpy(float),
                          np.zeros(len(raw), dtype=int), cadence, "before")
    after = noise_report(clean["peak_shift_kHz"].to_numpy(float),
                         clean["segment"].to_numpy(), cadence, "after")

    print("\n" + "=" * 70)
    print("NOISE")
    print("=" * 70)
    print(f"{'':10s} {'n':>8s} {'std kHz':>10s} {'MAD kHz':>10s} {'pk-pk kHz':>11s} "
          f"{'std uT':>9s} {'eta nT/rtHz':>12s}")
    for rep in (before, after):
        print(f"{rep['label']:10s} {rep['n']:8d} {rep['std_khz']:10.1f} {rep['mad_khz']:10.1f} "
              f"{rep['ptp_khz']:11.0f} {rep['std_uT']:9.2f} {rep['eta_nT_rtHz']:12.0f}")

    print(f"\nrate: {len(raw) / duration:.0f} Hz as recorded -> "
          f"{len(clean) / duration:.0f} Hz of real samples "
          f"({100 * len(clean) / len(raw):.0f}% of the rows kept)")
    print(f"      raw FPGA cadence is {1 / cadence:.0f} Hz; the shortfall is the dead time "
          f"between bursts, which streaming mode removes entirely")

    out_csv = outdir / f"{args.csv.stem}_clean.csv"
    out_png = outdir / f"{args.csv.stem}_clean.png"
    clean.to_csv(out_csv, index=False)
    make_plot(raw, clean, out_png, stale_rows, cadence)
    print(f"\nwrote {out_csv.name} and {out_png.name} to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
