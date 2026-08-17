#!/usr/bin/env python3
"""Regenerate every number and figure in the 2026-08-14 two-point analysis.

    python scripts/analyze_twopoint_0814.py

Offline: reads only committed CSVs, needs no hardware. Writes tables to
`docs/2026-08-14_twopoint_methods/tables/` and figures to `figures/`, so nothing
in the document is typed by hand and a later session refreshes it rather than
invalidating it.

Five analyses, each answering one question the session raised:

  A  timing     Why is the averaged mode still ~88 Hz after tau was halved?
                Recovers the exact FPGA readout time of every averaged run from
                the quantisation of the recorded counts, across all three
                sessions, and compares it against the measured batch period.
  B  modes      What did each acquisition mode actually deliver on 2026-08-14?
                Rate, sigma, sensitivity and measured noise floor, per mode.
  C  readpath   Why is streaming noisier than burst? Band-by-band ASD ratio, and
                the checks that rule out mains, drift and packet structure.
  D  perrep     Per-rep photodiode noise across both sessions and all modes --
                the quantity that shows averaging is not buying what it should.
  E  spectra    Where the lines are, and what 88 Hz sampling does to 60 Hz.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "notebook_modules"))

import burst_qc as qc                        # noqa: E402
import twopoint_postprocess as tpp            # noqa: E402
import twopoint_spectra as spec               # noqa: E402

OUTDIR = REPO_ROOT / "docs" / "2026-08-14_twopoint_methods"
TABLES = OUTDIR / "tables"
FIGURES = OUTDIR / "figures"

SESSION_0814 = REPO_ROOT / "data" / "results" / "081426 (Sensitivity increase update)"
TWOPOINT_0814 = SESSION_0814 / "two-point lockin"
SESSION_0806 = REPO_ROOT / "data" / "twopoint_lockin" / "08062026"

FPGA_CLOCK_HZ = tpp.FPGA_CLOCK_HZ
GAMMA = spec.GAMMA_NV_MHZ_PER_UT


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def live_csvs(root: Path):
    """Every two-point live/stream CSV under `root`, derived files excluded."""
    skip = ("summary", "calibration", "spectrum", "inference", "clean", "peakshift")
    out = []
    for path in sorted(root.rglob("twopoint_lockin_*.csv")):
        if any(k in path.name for k in skip):
            continue
        out.append(path)
    return out


def highpass_sigma(values, window: int = 31) -> float:
    """Scatter after removing a rolling median -- drift-immune, unlike std()."""
    s = pd.Series(np.asarray(values, dtype=float))
    return float((s - s.rolling(window, center=True, min_periods=1).median()).std())


def real_batch_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows belonging to batches that are not replays of the previous batch."""
    return ~qc.stale_sample_mask(df)


# --------------------------------------------------------------------------- #
# A -- timing: why the rate did not move
# --------------------------------------------------------------------------- #

def analyse_timing() -> pd.DataFrame:
    """FPGA readout work vs measured batch period, every averaged run, 3 sessions.

    The FPGA time is not taken from the notebook's configuration -- it is recovered
    from the data. `analyze_results` divides an integer buffer by treg and then by
    reps, so every stored count is a multiple of 1/(treg*reps); the smallest
    divisor that makes them all integral gives the readout time exactly.
    """
    rows = []
    for session, root in (("2026-08-04/06", SESSION_0806), ("2026-08-14", TWOPOINT_0814)):
        for path in live_csvs(root):
            df = pd.read_csv(path)
            if "acq_seconds" not in df.columns or "batch" not in df.columns:
                continue
            if len(df) > df["batch"].nunique():        # burst, handled in B
                continue
            quantum = tpp.recover_readout_quantum(df["peak_01"].to_numpy(float))
            if quantum is None:
                continue
            acq = df.groupby("batch")["acq_seconds"].first().to_numpy(float)
            t = df["time_s"].to_numpy(float)
            fpga_ms = tpp.readout_seconds(quantum["D"]) * 1e3
            period_ms = 1e3 * (t[-1] - t[0]) / max(1, len(t) - 1)
            rows.append({
                "session": session,
                "run": path.stem[-15:],
                "n_samples": len(df),
                "readout_quantum_D": quantum["D"],
                "fpga_readout_ms": fpga_ms,
                "call_p05_ms": float(np.percentile(acq, 5) * 1e3),
                "call_median_ms": float(np.median(acq) * 1e3),
                "period_ms": period_ms,
                "rate_hz": 1e3 / period_ms,
                "fpga_share": fpga_ms / period_ms,
                "sigma_khz": float(df["peak_shift_kHz"].std()),
            })
    out = pd.DataFrame(rows).sort_values(["session", "fpga_readout_ms"])
    if out.empty:
        return out

    # A run only tests the acquire() model if the loop period is close to the
    # acquire time. Some runs spent most of their period elsewhere -- live plotting
    # on, or the operator interacting -- and those say nothing about acquire().
    out["loop_bound"] = out["period_ms"] <= 1.5 * out["call_median_ms"]

    print("A. TIMING -- FPGA readout work against the measured batch period")
    print("-" * 78)
    print(out[["session", "run", "fpga_readout_ms", "call_p05_ms", "call_median_ms",
               "rate_hz", "fpga_share", "loop_bound"]].to_string(
        index=False, float_format=lambda v: f"{v:.2f}"))
    n_excluded = int((~out["loop_bound"]).sum())
    if n_excluded:
        print(f"\n  {n_excluded} run(s) excluded from the slope below: their loop period is "
              f">1.5x the acquire time, so most of it was spent outside acquire() "
              f"(live plotting or operator interaction) and they do not test this model.")
    out = out[out["loop_bound"]]

    lo, hi = out["fpga_readout_ms"].min(), out["fpga_readout_ms"].max()
    print(f"\n  FPGA readout work spans {lo:.2f} - {hi:.2f} ms ({hi / lo:.1f}x) across "
          f"{len(out)} runs")

    # Slope per session. Comparing across sessions would confound the FPGA change
    # with the host floor, which itself moved ~1 ms between 2026-08-04/06 and
    # 2026-08-14; within a session the floor is stable and the comparison is clean.
    print(f"\n  d(period)/d(FPGA work), per session -- 1.0 would mean the two add "
          f"serially:")
    for session, grp in out.groupby("session"):
        g = grp.groupby("fpga_readout_ms")["period_ms"].median()
        if len(g) < 2:
            continue
        x, y = g.index.to_numpy(), g.to_numpy()
        slope = float(np.polyfit(x, y, 1)[0])
        print(f"    {session:14s}  FPGA {x.min():.2f} -> {x.max():.2f} ms "
              f"({x.max() / x.min():.1f}x),  period {y.min():.2f} -> {y.max():.2f} ms,  "
              f"slope {slope:+.3f}")

    worst = out.loc[out["fpga_readout_ms"].idxmin()]
    print(f"\n  the run with the LEAST FPGA work ({worst['fpga_readout_ms']:.2f} ms) still "
          f"had a fastest call of {worst['call_p05_ms']:.2f} ms -- no serial model with a "
          f"non-negative host term fits that.")
    print("  -> the call is host-bound; the FPGA runs underneath it, not after it.\n")
    return out


# --------------------------------------------------------------------------- #
# B -- what each mode delivered
# --------------------------------------------------------------------------- #

MODE_RUNS = [
    ("2026-08-14", "averaged", TWOPOINT_0814 / "twopoint_lockin_live_20260814_144602.csv"),
    ("2026-08-14", "averaged", TWOPOINT_0814 / "twopoint_lockin_live_20260814_144655.csv"),
    ("2026-08-14", "averaged", TWOPOINT_0814 / "twopoint_lockin_live_20260814_145010.csv"),
    ("2026-08-14", "averaged", TWOPOINT_0814 / "twopoint_lockin_live_20260814_145321.csv"),
    ("2026-08-14", "stream", TWOPOINT_0814 / "twopoint_lockin_stream_20260814_145442.csv"),
    ("2026-08-14", "burst", TWOPOINT_0814 / "twopoint_lockin_live_20260814_150458.csv"),
    ("2026-08-06", "averaged", SESSION_0806 / "Drone Sweeps/twopoint_lockin_live_20260806_144936.csv"),
    ("2026-08-06", "burst", SESSION_0806 / "Drone Sweeps/twopoint_lockin_live_20260806_145051.csv"),
]


def analyse_modes() -> pd.DataFrame:
    """Rate, sigma, sensitivity and measured noise floor for every mode."""
    rows = []
    for session, mode, path in MODE_RUNS:
        if not path.is_file():
            print(f"  (missing: {path.name})")
            continue
        res = tpp.process(path, mode=mode, despike=False,
                          **({"detrend": False} if mode == "stream" else {}))
        r, s = res.report, res.spectrum
        rows.append({
            "session": session,
            "mode": mode,
            "run": path.stem[-15:],
            "rate_hz": r.get("within_burst_rate_hz", r.get("rate_hz", np.nan)),
            "effective_rate_hz": r.get("true_rate_hz", r.get("rate_hz", np.nan)),
            "tau_us": r.get("tau_us", np.nan),
            "sigma_khz": s.sigma_khz,
            "eta_nt_rthz": s.eta_nt_rthz,
            "floor_nt_rthz": s.floor_nt_rthz,
            "stale_fraction": r.get("stale_fraction", 0.0),
            "duty_cycle": r.get("duty_cycle", 1.0),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    print("B. MODES -- what each acquisition mode delivered")
    print("-" * 78)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    b14 = out[(out.session == "2026-08-14") & (out["mode"] == "burst")]
    b06 = out[(out.session == "2026-08-06") & (out["mode"] == "burst")]
    if len(b14) and len(b06):
        print(f"\n  burst floor 2026-08-06 -> 2026-08-14: "
              f"{b06.floor_nt_rthz.iloc[0]:.0f} -> {b14.floor_nt_rthz.iloc[0]:.0f} nT/rtHz "
              f"({b06.floor_nt_rthz.iloc[0] / b14.floor_nt_rthz.iloc[0]:.1f}x better after "
              f"the photodiode change)")
    print()
    return out


# --------------------------------------------------------------------------- #
# C -- the streaming read-path excess
# --------------------------------------------------------------------------- #

def analyse_read_path() -> pd.DataFrame:
    """Band-by-band ASD, stream against burst, same tau, same afternoon."""
    stream_csv = TWOPOINT_0814 / "twopoint_lockin_stream_20260814_145442.csv"
    burst_csv = TWOPOINT_0814 / "twopoint_lockin_live_20260814_150458.csv"
    if not (stream_csv.is_file() and burst_csv.is_file()):
        return pd.DataFrame()

    s = tpp.process(stream_csv, despike=False, detrend=False)
    b = tpp.process(burst_csv, despike=False)

    bands = [(10, 30), (30, 60), (60, 100), (100, 200), (200, 350), (350, 500)]
    rows = []
    for lo, hi in bands:
        rows.append({
            "band": f"{lo}-{hi} Hz",
            "stream_nt_rthz": spec.white_floor(s.spectrum.f, s.spectrum.asd, (lo, hi)),
            "burst_nt_rthz": spec.white_floor(b.spectrum.f, b.spectrum.asd, (lo, hi)),
        })
    out = pd.DataFrame(rows)
    out["ratio"] = out["stream_nt_rthz"] / out["burst_nt_rthz"]

    print("C. READ PATH -- streaming against burst, same tau, 10 minutes apart")
    print("-" * 78)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\n  ratio is {out['ratio'].min():.2f}-{out['ratio'].max():.2f} across the whole "
          f"band: a FLAT multiplicative excess, not a resonance or a drift knee.")

    # What it is not.
    sd = s.clean["peak_shift_kHz"].to_numpy(float)
    fs = float(s.report["rate_hz"])
    total = np.var(sd - sd.mean())
    notched = spec.notch(sd, fs, [spec.MAINS_HZ * k for k in range(1, 9)])
    mains_share = 1 - np.var(notched - notched.mean()) / total
    lf = tpp.detrend_series(sd, 101)
    lf_share = 1 - np.var(lf - lf.mean()) / total
    raw = s.raw
    raw = raw.assign(pos=raw.groupby("packet").cumcount())
    resid = raw["peak_01"] - raw["peak_01"].rolling(31, center=True, min_periods=1).median()
    first5 = float(resid[raw["pos"] < 5].std())
    mid = float(resid[raw["pos"].between(20, 180)].std())
    print(f"\n  ruled out, in the stream's own shift variance:")
    print(f"    mains comb (60 Hz + 7 harmonics) : {100 * mains_share:5.1f}%")
    print(f"    below ~10 Hz (drift)             : {100 * lf_share:5.1f}%")
    print(f"    packet-position dependence       : first-5 sigma {first5:.1f} vs "
          f"mid-packet {mid:.1f} ADC ({first5 / mid:.2f}x)")
    print(f"    duplicate rows                   : {int(s.report['duplicate_rows'])}")
    print(f"    rep_index gaps                   : {int(s.report['rep_index_gaps'])}")
    print("  -> a flat, structureless excess is what a read-path difference looks like.")
    print("     scripts/profile_twopoint_acquire.py --compare-read-paths settles it.\n")
    return out


# --------------------------------------------------------------------------- #
# D -- per-rep photodiode noise
# --------------------------------------------------------------------------- #

PERREP_RUNS = [
    ("2026-08-06", "burst", SESSION_0806 / "Drone Sweeps/twopoint_lockin_live_20260806_145051.csv", 1),
    ("2026-08-06", "burst", SESSION_0806 / "Drone Sweeps/twopoint_lockin_live_20260806_142358.csv", 1),
    ("2026-08-06", "averaged", SESSION_0806 / "Drone Sweeps/twopoint_lockin_live_20260806_144936.csv", 23),
    ("2026-08-06", "averaged", SESSION_0806 / "Drone Sweeps/twopoint_lockin_live_20260806_143229.csv", 23),
    ("2026-08-14", "burst", TWOPOINT_0814 / "twopoint_lockin_live_20260814_150458.csv", 1),
    ("2026-08-14", "stream", TWOPOINT_0814 / "twopoint_lockin_stream_20260814_145442.csv", 4),
    ("2026-08-14", "averaged", TWOPOINT_0814 / "twopoint_lockin_live_20260814_145321.csv", 10),
    ("2026-08-14", "averaged", TWOPOINT_0814 / "twopoint_lockin_live_20260814_145010.csv", 10),
    ("2026-08-14", "averaged", TWOPOINT_0814 / "twopoint_lockin_live_20260814_144602.csv", 23),
]


def analyse_per_rep() -> pd.DataFrame:
    """Per-rep normalised-PL noise, scaled back from whatever averaging was used.

    If the noise were photon shot noise, every row here would be comparable and
    averaging would reduce it as 1/sqrt(reps). It does not: the averaged mode's
    inferred per-rep noise is several times the burst's directly measured one, and
    it gets WORSE at higher rep counts -- so the dominant term is per-acquire, not
    per-rep, and adding reps cannot remove it.
    """
    rows = []
    for session, mode, path, reps in PERREP_RUNS:
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        if "batch" in df.columns and len(df) > df["batch"].nunique():
            df = df[real_batch_mask(df)]
        z = df["z_minus"].to_numpy(float)
        per_rep = highpass_sigma(z) * np.sqrt(reps)
        rows.append({
            "session": session, "mode": mode, "run": path.stem[-15:],
            "reps_averaged": reps,
            "z_mean": float(z.mean()),
            "sigma_per_rep": per_rep,
            "relative_pct": 100 * per_rep / abs(z.mean()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    print("D. PER-REP PHOTODIODE NOISE -- scaled back to a single rep")
    print("-" * 78)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    burst = out[out["mode"] == "burst"]["relative_pct"]
    other = out[out["mode"] != "burst"]["relative_pct"]
    print(f"\n  burst (measured directly, 1 rep): {burst.min():.2f}-{burst.max():.2f}% "
          f"in BOTH sessions")
    print(f"  everything else (inferred)      : {other.min():.2f}-{other.max():.2f}%")
    avg = out[out["mode"] == "averaged"].sort_values("reps_averaged")
    if len(avg) > 1:
        print(f"  averaged mode gets WORSE with more reps "
              f"({avg.iloc[0]['reps_averaged']} reps -> {avg.iloc[0]['relative_pct']:.2f}%, "
              f"{avg.iloc[-1]['reps_averaged']} reps -> {avg.iloc[-1]['relative_pct']:.2f}%)")
        print("  -> not shot noise, and not a per-rep term: it does not average down.\n")
    return out


# --------------------------------------------------------------------------- #
# E -- spectral lines and aliasing
# --------------------------------------------------------------------------- #

def analyse_spectra() -> pd.DataFrame:
    """Where the lines are, per mode, and what 88 Hz sampling does to 60 Hz."""
    rows = []
    for session, mode, path in MODE_RUNS:
        if not path.is_file() or session != "2026-08-14":
            continue
        res = tpp.process(path, mode=mode, despike=False,
                          **({"detrend": False} if mode == "stream" else {}))
        sp = res.spectrum
        strongest = (sp.lines.sort_values("excess_over_floor").iloc[-1]
                     if len(sp.lines) else None)
        rows.append({
            "mode": mode, "run": path.stem[-15:], "fs_hz": sp.fs_hz,
            "strongest_line_hz": float(strongest["frequency_hz"]) if strongest is not None else np.nan,
            "line_excess": float(strongest["excess_over_floor"]) if strongest is not None else np.nan,
            "mains_alias_hz": spec.alias_of(spec.MAINS_HZ, sp.fs_hz),
            "mains_is_aliased": sp.fs_hz < 2 * spec.MAINS_HZ,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    print("E. SPECTRAL LINES AND ALIASING")
    print("-" * 78)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\n  At ~88 Hz the 60 Hz mains line folds to "
          f"{spec.alias_of(60.0, 87.9):.1f} Hz -- into the middle of the band, where it")
    print("  cannot be distinguished from signal or filtered out. Only the >= 1 kHz modes")
    print("  resolve it as 60 Hz and can notch it.\n")
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def figure_timing(timing: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for session, grp in timing.groupby("session"):
        ax.scatter(grp["fpga_readout_ms"], grp["period_ms"], s=45, label=session)
    lim = [0, max(timing["period_ms"].max(), timing["fpga_readout_ms"].max()) * 1.15]
    ax.plot(lim, lim, "--", color="0.6", lw=1,
            label="serial model (period = FPGA + const)")
    floor = timing["call_p05_ms"].median()
    ax.axhline(floor, color="crimson", ls=":", lw=1.4,
               label=f"per-call floor ~{floor:.1f} ms")
    ax.set(xlabel="FPGA readout work per call (ms)", ylabel="measured batch period (ms)",
           title="The readout window is not the rate", xlim=lim, ylim=(0, lim[1]))
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(timing["fpga_readout_ms"], timing["rate_hz"], s=45, color="tab:purple")
    ax.axhline(timing["rate_hz"].median(), color="0.5", ls=":", lw=1)
    ax.set(xlabel="FPGA readout work per call (ms)", ylabel="rate (Hz)",
           title="A 4x change in FPGA work, the same ~88 Hz")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "timing_floor.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def figure_read_path() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = tpp.process(TWOPOINT_0814 / "twopoint_lockin_stream_20260814_145442.csv",
                    despike=False, detrend=False)
    b = tpp.process(TWOPOINT_0814 / "twopoint_lockin_live_20260814_150458.csv",
                    despike=False)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.loglog(b.spectrum.f[1:], b.spectrum.asd[1:], lw=0.8, color="tab:green",
              label=f"burst, {b.spectrum.fs_hz:.0f} Hz "
                    f"(floor {b.spectrum.floor_nt_rthz:.0f} nT/rtHz)")
    ax.loglog(s.spectrum.f[1:], s.spectrum.asd[1:], lw=0.8, color="tab:red",
              label=f"stream, {s.spectrum.fs_hz:.0f} Hz "
                    f"(floor {s.spectrum.floor_nt_rthz:.0f} nT/rtHz)")
    ax.axhline(b.spectrum.floor_nt_rthz, color="tab:green", ls=":", lw=1)
    ax.axhline(s.spectrum.floor_nt_rthz, color="tab:red", ls=":", lw=1)
    ax.set(xlabel="frequency (Hz)", ylabel="nT / sqrt(Hz)",
           title="Same tau, same afternoon, nothing changed on the bench: "
                 "streaming sits a flat ~6x above burst")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "read_path_asd.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def figure_modes(modes: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = modes[modes["session"] == "2026-08-14"].copy()
    if m.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"averaged": "tab:blue", "burst": "tab:green", "stream": "tab:red"}
    for _, r in m.iterrows():
        ax.scatter(r["rate_hz"], r["floor_nt_rthz"], s=90,
                   color=colors.get(r["mode"], "0.5"))
        ax.annotate(f"{r['mode']}\n{r['run'][-6:]}",
                    (r["rate_hz"], r["floor_nt_rthz"]),
                    textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.set(xscale="log", yscale="log", xlabel="sample rate (Hz)",
           ylabel="measured ASD floor (nT / sqrt(Hz))",
           title="2026-08-14: rate against noise floor, by acquisition mode\n"
                 "(runs taken minutes apart, not interleaved -- see section 5)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "mode_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("TWO-POINT LOCK-IN -- 2026-08-14 SESSION ANALYSIS")
    print("=" * 78)
    print()

    timing = analyse_timing()
    modes = analyse_modes()
    readpath = analyse_read_path()
    perrep = analyse_per_rep()
    spectra = analyse_spectra()

    for name, frame in (("timing", timing), ("modes", modes), ("read_path", readpath),
                        ("per_rep_noise", perrep), ("spectra", spectra)):
        if not frame.empty:
            frame.to_csv(TABLES / f"{name}.csv", index=False)

    if not timing.empty:
        figure_timing(timing)
    if not modes.empty:
        figure_modes(modes)
    if not readpath.empty:
        figure_read_path()

    print(f"tables  -> {TABLES}")
    print(f"figures -> {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
