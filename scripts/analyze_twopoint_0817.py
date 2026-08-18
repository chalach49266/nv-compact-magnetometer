#!/usr/bin/env python3
"""The 2026-08-17 morning session: the reference for what a working rig looks like.

    python scripts/analyze_twopoint_0817.py

Everything in `data/results/081726 (Test after Increased Sensitivity)/` was taken on
the *other* computer, running the notebook at commit 8fd303f or earlier, while the
same notebook on this laptop at 9832e07 could not complete a burst or a stream. So
this session is two things at once:

1. **A working baseline.** Its streaming runs are exactly what Step 4C should
   reproduce once the drain loop is fixed -- 31 250 samples, 30.000 s, uniform
   960 us spacing -- and the magnet was walked in from 50 cm to 30 cm to 15 cm in
   5-second holds, which gives a known step response to check the conversion
   against.

2. **A warning about its own burst files.** They are named `twopoint_lockin_live_*`
   because Step 4A and Step 4B both wrote that prefix before the rename in 9832e07,
   and they still carry the ~50% stale-replay defect. One of them records 45.7 s of
   FPGA work inside 30.1 s of wall clock, which is impossible: the per-batch time
   axes overlap. Numbers quoted straight off those files -- including the 29 962 Hz
   in the session's own `_summary.csv` -- are not sample rates. This script
   reprocesses them through the burst pipeline with `drop_stale=True` before
   quoting anything.

Outputs
-------
tables (CSV)   docs/2026-08-14_twopoint_methods/tables/0817_*.csv
figures (PNG)  docs/twopoint_master_reference/figures/0817_*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "notebook_modules"))

import twopoint_postprocess as tpp          # noqa: E402
import twopoint_spectra as spec             # noqa: E402

SESSION = REPO_ROOT / "data" / "results" / "081726 (Test after Increased Sensitivity)"
TABLE_DIR = REPO_ROOT / "docs" / "2026-08-14_twopoint_methods" / "tables"
FIG_DIR = REPO_ROOT / "docs" / "twopoint_master_reference" / "figures"

# The operator's log for this session: the magnet was held at each distance for a
# five-second window, with five seconds of travel between holds.
#
#   0-5 s   settling / far
#   5-10    50 cm
#   15-20   30 cm
#   25-30   15 cm
#
# The 10-15 s and 20-25 s windows are the magnet in transit and are excluded.
DISTANCE_WINDOWS = [
    ("far (start)", 0.0, 5.0, None),
    ("50 cm", 5.0, 10.0, 50.0),
    ("30 cm", 15.0, 20.0, 30.0),
    ("15 cm", 25.0, 30.0, 15.0),
]

# Bands to compare stream against burst in, matching the table already in
# part6_defects.tex so the two can be read side by side.
ASD_BANDS = [(10.0, 30.0), (100.0, 200.0), (350.0, 500.0)]


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #

def session_runs() -> list[Path]:
    """Every two-point run in the session, whatever prefix it was written under."""
    hits = []
    for prefix in ("live", "avg", "burst", "stream"):
        hits.extend((SESSION / "Two-point lockin").glob(
            f"twopoint_lockin_{prefix}_*.csv"))
    return sorted(p for p in hits if not p.name.endswith(
        ("_summary.csv", "_peakshift_calibration.csv", "_spectrum.csv")))


def inventory() -> pd.DataFrame:
    """What each file actually is -- decided by columns, never by filename.

    This is the table that shows the naming collision: several files called
    `twopoint_lockin_live_*` are burst runs, which is why 9832e07 split the prefix
    into `_avg_` and `_burst_`.
    """
    rows = []
    for path in session_runs():
        df = pd.read_csv(path)
        try:
            mode = tpp.detect_mode(df)
        except ValueError as exc:
            mode = f"unknown ({exc})"
        span = float(df["time_s"].iloc[-1]) if "time_s" in df else float("nan")
        rows.append({
            "file": path.name,
            "prefix": path.name.split("_")[2],
            "detected_mode": mode,
            "rows": len(df),
            "time_span_s": span,
            "rows_per_span_hz": len(df) / span if span > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


_CACHE: dict[tuple[str, str], "tpp.TwoPointResult"] = {}


def load_run(path: Path, mode: str):
    """Process one run through its own pipeline, once, and remember it.

    Burst runs go through with `drop_stale=True`: without it roughly half the rows
    are bit-identical replays of the previous batch, and every statistic computed
    over them -- rate, sigma, spectrum -- is wrong in a way that flatters the
    result.
    """
    key = (str(path), mode)
    if key not in _CACHE:
        if mode == "stream":
            _CACHE[key] = tpp.process(path, mode="stream", despike="shift",
                                      detrend=True, detrend_window=501)
        elif mode == "burst":
            _CACHE[key] = tpp.process(path, mode="burst", drop_stale=True,
                                      drop_first_sample=True, despike="shift")
        elif mode == "averaged":
            _CACHE[key] = tpp.process(path, mode="averaged", despike="shift")
        else:
            raise ValueError(f"no pipeline for mode {mode!r}")
    return _CACHE[key]


def burst_time_audit(inv: pd.DataFrame) -> pd.DataFrame:
    """Is each burst file's recorded rate a rate at all?

    Step 4B timestamps each sample as `batch_start + i * cadence`, which is right
    *within* a burst. It goes wrong between bursts whenever the host comes back
    around faster than the FPGA can produce one -- which is what the stale-replay
    defect does, since a replayed batch returns in ~12 ms instead of ~120 ms.
    Rows then accumulate faster than FPGA time advances and `n_rows / span` stops
    meaning anything.

    The numbers here come from `process_burst`, not from arithmetic on 240 us:
    it recovers the exact cadence from the value quantisation of the accumulator
    (`recover_readout_quantum`), which is immune to the wall-clock stretch, and
    counts replayed batches by byte-comparing consecutive ones.
    """
    rows = []
    for _, r in inv[inv["detected_mode"] == "burst"].iterrows():
        path = SESSION / "Two-point lockin" / r["file"]
        rep = load_run(path, "burst").report
        rows.append({
            "file": r["file"],
            "n_batches": rep["n_batches"],
            "stale_pct": 100.0 * rep["stale_fraction"],
            "cadence_us": rep["cadence_us"],
            "cadence_src": rep["cadence_source"],
            "recorded_rate_hz": rep["recorded_rate_hz"],
            "true_rate_hz": rep["true_rate_hz"],
            "within_burst_hz": rep["within_burst_rate_hz"],
            "duty_pct": 100.0 * rep["duty_cycle"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# The magnet-distance steps
# --------------------------------------------------------------------------- #

def distance_steps(path: Path) -> pd.DataFrame:
    """Mean shift and spread in each distance hold of one stream run.

    Processed with `detrend=False`, deliberately, and so NOT through `load_run`:
    the step between distances *is* a DC shift, and the rolling-median high-pass
    that the streaming pipeline applies by default (to remove the PL droop) would
    take the signal out along with the droop. The cost is that the droop is still
    in these numbers -- which is why every row is also reported relative to the
    opening window rather than in absolute terms.
    """
    res = tpp.process(path, mode="stream", despike="shift",
                      detrend=False, notch_mains=False)
    df = res.clean
    t = df["time_s"].to_numpy(float)
    shift = df["peak_shift_kHz"].to_numpy(float)

    rows = []
    for label, lo, hi, cm in DISTANCE_WINDOWS:
        m = (t >= lo) & (t < hi) & np.isfinite(shift)
        if not m.any():
            continue
        seg = shift[m]
        rows.append({
            "file": path.name,
            "window": label,
            "distance_cm": cm,
            "t_start_s": lo,
            "t_end_s": hi,
            "n": int(m.sum()),
            "shift_mean_kHz": float(seg.mean()),
            "shift_std_kHz": float(seg.std()),
            "B_mean_nT": float(spec.khz_to_nt(seg.mean())),
            "B_std_nT": float(spec.khz_to_nt(seg.std())),
        })
    out = pd.DataFrame(rows)
    if len(out):
        # Referenced to the opening window, so the column is the field the magnet
        # actually contributed rather than the absolute parked offset.
        base = out.iloc[0]["shift_mean_kHz"]
        out["shift_vs_start_kHz"] = out["shift_mean_kHz"] - base
        out["B_vs_start_nT"] = spec.khz_to_nt(out["shift_vs_start_kHz"].to_numpy())
    return out


def dipole_check(steps: pd.DataFrame) -> dict:
    """Does the step size fall off like a dipole?

    A point dipole on axis gives |B| ~ r^-3. Three distances is barely enough to
    fit an exponent, so this is a sanity check on the sign and the order of
    magnitude, not a measurement of anything. It is reported with its own caveat
    because the standard deviation *also* grows as the magnet comes in, which a
    dipole does not predict -- see `notes` in the printed output.
    """
    d = steps.dropna(subset=["distance_cm"])
    d = d[np.abs(d["shift_vs_start_kHz"]) > 0]
    if len(d) < 3:
        return {}
    r = d["distance_cm"].to_numpy(float)
    b = np.abs(d["B_vs_start_nT"].to_numpy(float))
    slope, intercept = np.polyfit(np.log(r), np.log(b), 1)
    return {"exponent": float(slope), "prefactor": float(np.exp(intercept)),
            "expected_exponent": -3.0}


# --------------------------------------------------------------------------- #
# Stream against burst
# --------------------------------------------------------------------------- #

QUIET_WINDOW_S = (0.0, 5.0)


def _asd_over(res, window=None):
    """ASD of one run, optionally restricted to a time window.

    Burst data is transformed per contiguous burst and never across the dead time
    between them, which is what `segmented_asd_nt` is for; streaming has no gaps,
    so a plain Welch estimate is correct there.
    """
    df = res.clean
    m = np.ones(len(df), dtype=bool)
    if window is not None:
        t = df["time_s"].to_numpy(float)
        m = (t >= window[0]) & (t < window[1])
    if m.sum() < 64:
        return None, None
    shift = df["peak_shift_kHz"].to_numpy(float)[m]
    fs = res.spectrum.fs_hz
    if res.mode == "burst" and "segment" in df.columns:
        seg = df["segment"].to_numpy()[m]
        # A window can clip a burst; segments with too few samples are dropped by
        # the estimator itself.
        return spec.segmented_asd_nt(shift, seg, fs)
    return spec.asd_nt(shift, fs)


def band_asd(res, window=None, suffix="") -> dict:
    """Median ASD in each comparison band, in nT/sqrt(Hz)."""
    f, asd = _asd_over(res, window)
    out = {}
    for lo, hi in ASD_BANDS:
        key = f"asd{suffix}_{lo:.0f}_{hi:.0f}"
        out[key] = (spec.white_floor(f, asd, (lo, hi))
                    if f is not None else float("nan"))
    return out


def compare_read_paths(inv: pd.DataFrame) -> pd.DataFrame:
    """Per-file ASD for every stream and (de-staled) burst run in the session.

    Two windows per file, because the whole session was a magnet test and the
    magnet was moved by hand:

      full   the entire 30 s, which for the stream runs contains the deliberate
             50/30/15 cm steps and whatever hand motion came with them
      quiet  the first 5 s, before the magnet was brought in at all

    A comparison that only quoted `full` would be reading the operator's arm as
    sensor noise. The two agreeing is what makes the comparison worth anything.
    """
    rows = []
    for _, r in inv.iterrows():
        path = SESSION / "Two-point lockin" / r["file"]
        mode = r["detected_mode"]
        if mode not in ("stream", "burst", "averaged"):
            continue
        try:
            res = load_run(path, mode)
        except Exception as exc:                      # noqa: BLE001
            rows.append({"file": r["file"], "mode": mode, "error": str(exc)[:120]})
            continue
        sp = res.spectrum
        row = {"file": r["file"], "mode": mode, "fs_hz": sp.fs_hz,
               "sigma_kHz": sp.sigma_khz}
        row.update(band_asd(res, None, ""))
        row.update(band_asd(res, QUIET_WINDOW_S, "_quiet"))
        if mode == "burst":
            row["stale_pct"] = 100.0 * res.report["stale_fraction"]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def figure_distance(path: Path, steps: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = tpp.process(path, mode="stream", despike="shift", detrend=False)
    df = res.clean

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2),
                             gridspec_kw={"width_ratios": [2.2, 1]})

    ax = axes[0]
    ax.plot(df["time_s"], df["peak_shift_kHz"], lw=0.4, color="0.65")
    for label, lo, hi, cm in DISTANCE_WINDOWS:
        ax.axvspan(lo, hi, color="tab:blue" if cm else "0.85", alpha=0.14)
        sub = steps[steps["window"] == label]
        if len(sub):
            ax.hlines(sub["shift_mean_kHz"].iloc[0], lo, hi,
                      color="crimson", lw=2.0, zorder=5)
            ax.text(0.5 * (lo + hi), sub["shift_mean_kHz"].iloc[0],
                    f" {label}", fontsize=8, va="bottom", ha="center", color="crimson")
    ax.set(xlabel="time (s)", ylabel="peak shift (kHz)",
           title=f"Magnet walked in -- {path.name}")
    ax.grid(alpha=0.3)

    ax = axes[1]
    d = steps.dropna(subset=["distance_cm"])
    ax.plot(d["distance_cm"], np.abs(d["B_vs_start_nT"]), "o-", color="tab:blue")
    ax.set(xscale="log", yscale="log", xlabel="magnet distance (cm)",
           ylabel="|dB| vs start (nT)", title="Step size against distance")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def figure_read_paths(cmp: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(ASD_BANDS))
    labels = [f"{lo:.0f}-{hi:.0f} Hz" for lo, hi in ASD_BANDS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, (suffix, title) in zip(axes, [
            ("", "full 30 s (magnet moving)"),
            ("_quiet", f"first {QUIET_WINDOW_S[1]:.0f} s (magnet away)")]):
        cols = [f"asd{suffix}_{lo:.0f}_{hi:.0f}" for lo, hi in ASD_BANDS]
        if cols[0] not in cmp.columns:
            continue
        for mode, colour in (("burst", "tab:green"), ("stream", "tab:purple")):
            sub = cmp[(cmp["mode"] == mode) & cmp[cols[0]].notna()]
            if sub.empty:
                continue
            ax.plot(x, [sub[c].median() for c in cols], "o-", color=colour,
                    label=f"{mode} (median of {len(sub)})")
            for c, xi in zip(cols, x):
                ax.scatter([xi] * len(sub), sub[c], s=14, color=colour, alpha=0.4)
        ax.set_xticks(x, labels)
        ax.set(yscale="log", title=title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("ASD (nT/sqrt(Hz))")
    fig.suptitle("2026-08-17 morning: streaming vs de-staled burst, same rig")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main() -> int:
    if not SESSION.exists():
        print(f"session folder not found: {SESSION}")
        return 1
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("1. What is actually in the session (mode from columns, not filename)")
    print("=" * 78)
    inv = inventory()
    cols = ["file", "prefix", "detected_mode", "rows", "time_span_s", "rows_per_span_hz"]
    print(inv[cols].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    inv.to_csv(TABLE_DIR / "0817_inventory.csv", index=False)

    mism = inv[(inv["prefix"] == "live") & (inv["detected_mode"] == "burst")]
    if len(mism):
        print(f"\n  {len(mism)} file(s) named `_live_` are BURST runs. That collision is "
              f"why\n  9832e07 split the prefix into `_avg_` and `_burst_`.")

    if (inv["detected_mode"] == "burst").any():
        print("\n" + "=" * 78)
        print("2. The burst files' recorded rate is not a rate")
        print("=" * 78)
        audit = burst_time_audit(inv)
        print(audit.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
        audit.to_csv(TABLE_DIR / "0817_burst_audit.csv", index=False)
        print(f"\n  The replay defect is still fully present: "
              f"{audit['stale_pct'].mean():.0f}% of batches on average are "
              f"bit-identical\n  copies of the batch before. Recorded "
              f"{audit['recorded_rate_hz'].mean():,.0f} Hz on average; the honest "
              f"figure is\n  {audit['true_rate_hz'].mean():,.0f} Hz "
              f"({audit['within_burst_hz'].mean():,.0f} Hz inside a burst at "
              f"{audit['duty_pct'].mean():.0f}% duty cycle).")
        print(f"  The session's own *_summary.csv quotes ~29,962 Hz for "
              f"{audit['file'].iloc[0]},\n  which is that artefact taken at face "
              f"value. Do not carry it forward.")

    print("\n" + "=" * 78)
    print("3. Magnet distance steps (stream runs)")
    print("=" * 78)
    streams = inv[inv["detected_mode"] == "stream"]["file"].tolist()
    all_steps = []
    for name in streams:
        path = SESSION / "Two-point lockin" / name
        steps = distance_steps(path)
        if steps.empty:
            continue
        all_steps.append(steps)
        print(f"\n{name}")
        print(steps[["window", "distance_cm", "n", "shift_mean_kHz", "shift_std_kHz",
                     "shift_vs_start_kHz", "B_vs_start_nT"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

        # Does this run actually follow the logged schedule? Closer magnet must mean
        # a larger |shift|. A run that fails this was mistimed against the 5 s marks
        # -- the windows are a hand-operated convention, not something recorded in
        # the file -- and nothing quantitative should be taken from it.
        held = steps.dropna(subset=["distance_cm"]).sort_values(
            "distance_cm", ascending=False)
        mag = np.abs(held["shift_vs_start_kHz"].to_numpy())
        follows = bool(np.all(np.diff(mag) > 0))
        steps["follows_schedule"] = follows
        if follows:
            fit = dipole_check(steps)
            if fit:
                print(f"  monotonic in distance. |dB| ~ r^{fit['exponent']:.2f} "
                      f"(a point dipole on axis would give r^{fit['expected_exponent']:.0f}; "
                      f"this magnet is not a point,\n  and only its projection on one NV "
                      f"axis is measured, so the exponent is indicative only)")
        else:
            print("  NOT monotonic in distance -- this run does not follow the 5 s "
                  "schedule.\n  Mistimed against the hand-operated marks; excluded "
                  "from the fit.")

    if all_steps:
        steps_df = pd.concat(all_steps, ignore_index=True)
        steps_df.to_csv(TABLE_DIR / "0817_distance_steps.csv", index=False)

        held = steps_df.dropna(subset=["distance_cm"])
        near = held[held["distance_cm"] == held["distance_cm"].min()]
        far = held[held["distance_cm"] == held["distance_cm"].max()]
        print("\n  NOTE, and it is not a detail: the spread grows with the signal --")
        print(f"  sigma is {far['shift_std_kHz'].mean():.0f} kHz at "
              f"{far['distance_cm'].iloc[0]:.0f} cm and "
              f"{near['shift_std_kHz'].mean():.0f} kHz at "
              f"{near['distance_cm'].iloc[0]:.0f} cm.")
        print("  A static dipole does not do that. Either the magnet was still being")
        print("  handled during the near holds, or the gradient across the sensor is")
        print("  large enough to matter. This has to be settled with a clamped magnet")
        print("  before any sensitivity claim is made from the near-field windows.")

        good = steps_df[steps_df["follows_schedule"]]["file"].unique().tolist()
        print(f"\n  {len(good)} of {len(streams)} stream runs follow the schedule: "
              f"{', '.join(good) if good else 'none'}")
        if good:
            # The operator named twopoint_lockin_stream_20260817_121002.csv as the
            # worked example; prefer it when it qualifies, else the last that does.
            named = "twopoint_lockin_stream_20260817_121002.csv"
            best = named if named in good else good[-1]
            figure_distance(SESSION / "Two-point lockin" / best,
                            steps_df[steps_df["file"] == best],
                            FIG_DIR / "0817_distance_steps.png")
            print(f"  figure drawn from {best}")

    print("\n" + "=" * 78)
    print("4. Stream against burst, same rig, same morning")
    print("=" * 78)
    cmp = compare_read_paths(inv)
    cmp.to_csv(TABLE_DIR / "0817_read_paths.csv", index=False)
    show = [c for c in cmp.columns if c != "error"]
    print(cmp[show].to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    s = cmp[cmp["mode"] == "stream"]
    b = cmp[cmp["mode"] == "burst"]
    if len(s) and len(b):
        print("\n  ASD in nT/sqrt(Hz), median over runs. `quiet` = first 5 s only,")
        print("  before the magnet was brought in.\n")
        print("  band             full: stream  burst  ratio | quiet: stream  burst  ratio")
        for lo, hi in ASD_BANDS:
            cf = f"asd_{lo:.0f}_{hi:.0f}"
            cq = f"asd_quiet_{lo:.0f}_{hi:.0f}"
            sf, bf = s[cf].median(), b[cf].median()
            sq, bq = s[cq].median(), b[cq].median()
            print(f"  {lo:5.0f}-{hi:4.0f} Hz  {sf:12.1f} {bf:6.1f} {sf/bf:6.2f} |"
                  f" {sq:12.1f} {bq:6.1f} {sq/bq:6.2f}")
        ratios = [s[f"asd_quiet_{lo:.0f}_{hi:.0f}"].median()
                  / b[f"asd_quiet_{lo:.0f}_{hi:.0f}"].median() for lo, hi in ASD_BANDS]
        print(f"\n  Streaming sits {min(ratios):.1f}-{max(ratios):.1f}x above burst on "
              f"the quiet window, flat across\n  the band. That is the same *kind* of "
              f"excess the 2026-08-14 session recorded, but\n  far smaller than the "
              f"5.3-6.3x quoted there -- so part6_defects.tex needs updating.")
        print("  Caveats, both real: these are different runs ten minutes apart, not an")
        print("  interleaved A/B, and the run-to-run scatter within each mode is a factor")
        print("  of ~2. `profile_twopoint_acquire.py --compare-read-paths` is still the")
        print("  measurement that settles it.")
        figure_read_paths(cmp, FIG_DIR / "0817_read_paths.png")

    print(f"\ntables  -> {TABLE_DIR}")
    print(f"figures -> {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
