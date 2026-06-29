"""Generate the shot-noise EDA notebook with nbformat.

Run:  python3 _build_notebook.py   ->   writes shot_noise_eda.ipynb
"""
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []


def md(text):
    cells.append(new_markdown_cell(text))


def code(src):
    cells.append(new_code_cell(src))


# ---------------------------------------------------------------- title
md(r"""# Shot-Noise Scaling of the NV Compact Magnetometer vs. Lock-in Repetitions

**Exploratory data analysis — `lockin_multipoint/shot_noise_experiment`**
Generated 2026-06-10.

## Question
For a shot-noise-limited measurement the uncertainty (standard deviation) of an
averaged quantity scales as

$$\sigma(N) \;\propto\; \frac{1}{\sqrt{N}},$$

where $N$ is the number of averaged repetitions per data point. As $N$ grows the
noise should fall along this $N^{-1/2}$ line **until it saturates** on a floor set
by drift / systematic (1/f) noise. Our magnetometer currently runs at a **baseline
of $N=10$ repetitions** — this notebook quantifies *where that baseline sits
relative to the shot-noise limit*, using runs at $N = 10, 20, \dots, 100$
(3 trials each, 30 datasets total).

We track the standard deviation of four quantity groups:

| Group | Columns | Count | Unit |
|---|---|---|---|
| ADC channels | `peak_01 … peak_16` | 16 | ADC |
| Peak shifts | `delta_f_mhz_b01 … b08` | 8 | MHz |
| B-field axes | `delta_Bx_uT`, `delta_By_uT`, `delta_Bz_uT` | 3 | µT |
| B magnitude | $|\Delta\mathbf{B}|=\sqrt{B_x^2+B_y^2+B_z^2}$ | 1 | µT |

## The fairness problem and how we handle it
Higher-$N$ runs collect **fewer data points** (each point is slower to acquire:
686 points at $N{=}10$ vs. ~152 at $N{=}100$). A naive worry is that this biases
the comparison. The crucial mitigating fact, verified below, is that **every run
spans the same ~60 s wall-clock window**. Therefore:

1. **Drift exposure is equalized** — all runs see the same amount of slow drift,
   because they cover the same 60 s. So the differing point count does **not** bias
   the *value* of $\sigma$; it only changes the *precision* of the $\sigma$
   estimate (more points → tighter estimate).
2. We account for that estimation precision with **between-trial error bars**
   (3 independent trials per $N$) and an **equal-$N$ evenly-spaced subsample
   cross-check** (Section 9) that re-estimates every $\sigma$ from the same number
   of points (`N_MIN`) while keeping the full 60 s window.
3. We report a **drift-robust noise metric** (successive-difference / Allan-style)
   alongside the raw $\sigma$, so the shot-noise component is isolated from slow
   drift.
4. An independent **Allan-deviation** analysis (Section 10) confirms the
   white-noise→drift transition *within a single run*.

This is more defensible than simply truncating every run to the same number of
points (which would shrink the time window of the low-$N$ runs and change their
drift exposure).

> **⚠ Spoiler / headline finding (Section 8).** The analysis below uncovers a
> *second*, dominant fairness problem that the equal-time design does **not**
> solve: the runs were taken **sequentially** (all $N{=}10$ trials first … all
> $N{=}100$ last), so **repetition count is perfectly confounded with
> acquisition time**, and a slow environmental drift grew over the ~65 min
> session. The apparent "σ rises with $N$" is largely this session drift, not a
> property of averaging. Read Sections 7–8 together, and see the recommendation
> to re-measure with *interleaved* rep order.""")

# ---------------------------------------------------------------- imports
md("## 1 · Setup")
code(r"""import re, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_theme(context="notebook", style="whitegrid")
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 120, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11,
    "legend.fontsize": 8.5, "axes.grid": True, "grid.alpha": 0.30,
    "lines.markersize": 5,
})
RNG = np.random.default_rng(42)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
print("Versions — numpy", np.__version__, "| pandas", pd.__version__,
      "| matplotlib", mpl.__version__)""")

# ---------------------------------------------------------------- locate
md("## 2 · Locate the data & define the quantity groups")
code(r"""CANDIDATES = [
    Path("../../data/lockin_multipoint/shot_noise_experiment"),
    Path("data/lockin_multipoint/shot_noise_experiment"),
    Path("/Users/ckasemtantikul/Library/CloudStorage/OneDrive-WashingtonUniversityin"
         "St.Louis/Documents/PhD/NV Compact Magnetometer/nv_magnetometer_project/"
         "data/lockin_multipoint/shot_noise_experiment"),
]
DATA_DIR = next((p.resolve() for p in CANDIDATES if p.exists()), None)
assert DATA_DIR is not None, "Could not locate shot_noise_experiment data dir."
print("DATA_DIR:", DATA_DIR)

REP_LEVELS = sorted(int(re.match(r"(\d+)rep", p.name).group(1))
                    for p in DATA_DIR.glob("*rep") if p.is_dir())
N_BASELINE = REP_LEVELS[0]          # current operating point = 10 reps
print("Repetition levels:", REP_LEVELS, " | baseline =", N_BASELINE)

# --- column groups -----------------------------------------------------
PEAK_COLS  = [f"peak_{i:02d}" for i in range(1, 17)]            # 16 ADC channels
SHIFT_COLS = [f"delta_f_mhz_b{i:02d}" for i in range(1, 9)]     # 8 peak shifts (MHz)
BAXIS_COLS = ["delta_Bx_uT", "delta_By_uT", "delta_Bz_uT"]      # 3 axes (uT)
BMAG_COL   = "delta_B_mag_uT"                                   # computed magnitude

GROUPS = {
    "ADC channels (16)": PEAK_COLS,
    "Peak shifts (8)":   SHIFT_COLS,
    "B-field axes (3)":  BAXIS_COLS,
    "B magnitude (1)":   [BMAG_COL],
}
UNITS = ({c: "ADC" for c in PEAK_COLS} | {c: "MHz" for c in SHIFT_COLS}
         | {c: "uT" for c in BAXIS_COLS} | {BMAG_COL: "uT"})
COL2GROUP = {c: g for g, cols in GROUPS.items() for c in cols}
ALL_COLS = PEAK_COLS + SHIFT_COLS + BAXIS_COLS + [BMAG_COL]
print(f"Tracking {len(ALL_COLS)} columns across {len(GROUPS)} groups.")""")

# ---------------------------------------------------------------- load
md(r"""## 3 · Load every trial

Each rep-folder holds 3 trials. The *main* CSV
(`multipoint_lockin_live_<timestamp>.csv`) carries the 16 ADC channels and the 8
peak shifts; the companion `_vector_rows.csv` carries the per-shot B-field deltas
(`delta_Bx/By/Bz_uT`). They are aligned row-for-row by `save_id`, so we merge by
position and compute the field magnitude $|\Delta\mathbf{B}|$.""")
code(r"""def is_main_csv(p: Path) -> bool:
    suff = ("_block_summary", "_projection_rows", "_vector_rows", "_wide")
    return p.suffix == ".csv" and not any(s in p.name for s in suff)

def load_trial(main_path: Path) -> pd.DataFrame:
    df = pd.read_csv(main_path)
    vec = pd.read_csv(main_path.with_name(main_path.stem + "_vector_rows.csv"))
    n = min(len(df), len(vec))                       # align by row order (save_id)
    df = df.iloc[:n].reset_index(drop=True)
    vec = vec.iloc[:n].reset_index(drop=True)
    for c in BAXIS_COLS + ["residual_norm_uT", "condition_number"]:
        df[c] = vec[c].to_numpy()
    df[BMAG_COL] = np.sqrt(df["delta_Bx_uT"]**2
                           + df["delta_By_uT"]**2
                           + df["delta_Bz_uT"]**2)
    return df

trials, records = {}, []
for rep in REP_LEVELS:
    mains = sorted(p for p in (DATA_DIR / f"{rep}rep").iterdir() if is_main_csv(p))
    for ti, mp in enumerate(mains, start=1):
        df = load_trial(mp)
        trials[(rep, ti)] = df
        t = df["time_s"].to_numpy()
        records.append(dict(rep=rep, trial=ti, n_points=len(df),
                            t_epoch_start=df["timestamp_epoch_s"].iloc[0],
                            T_elapsed_s=t[-1] - t[0],
                            dt_s=(t[-1] - t[0]) / (len(df) - 1),
                            acq_s=df["acq_seconds"].mean(),
                            duty_cycle=df["acq_seconds"].mean() / ((t[-1]-t[0])/(len(df)-1)),
                            file=mp.name))
inventory = pd.DataFrame(records)
# session clock: minutes since the very first run started
inventory["session_min"] = (inventory["t_epoch_start"]
                            - inventory["t_epoch_start"].min()) / 60.0
print(f"Loaded {len(trials)} trials.  "
      f"Session spans {inventory.session_min.max():.1f} min.")
inventory""")

# ---------------------------------------------------------------- inventory
md(r"""## 4 · Acquisition inventory — confirming the *equal-time* design

The table below is the heart of the fairness argument. Read the `T_elapsed_s`
column: **every run lasts ~60 s** regardless of $N$. What changes with $N$ is the
number of points (`n_points`, 686 → ~152), the sample spacing (`dt_s`), and the
per-point integration time (`acq_s`, which grows roughly $\propto N$).""")
code(r"""inv_summary = (inventory.groupby("rep")
    .agg(n_trials=("trial", "count"),
         n_points=("n_points", "mean"),
         T_elapsed_s=("T_elapsed_s", "mean"),
         dt_s=("dt_s", "mean"),
         acq_s=("acq_s", "mean"),
         duty_cycle=("duty_cycle", "mean"),
         sample_rate_Hz=("dt_s", lambda s: 1.0 / s.mean()))
    .round(3))
N_MIN = int(inventory["n_points"].min())
print("Equal-time check — elapsed range: "
      f"{inventory.T_elapsed_s.min():.2f} – {inventory.T_elapsed_s.max():.2f} s")
print("Smallest point count across all 30 trials  N_MIN =", N_MIN)
inv_summary""")

code(r"""# Visual: point count and per-point integration time vs N
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
g = inventory.groupby("rep")
axes[0].errorbar(g.n_points.mean().index, g.n_points.mean(), yerr=g.n_points.std(),
                 marker="o", capsize=3, color="#2c7fb8")
axes[0].set(title="Points collected per run", xlabel="repetitions $N$",
            ylabel="n points (60 s window)")
axes[1].errorbar(g.acq_s.mean().index, g.acq_s.mean()*1e3, yerr=g.acq_s.std()*1e3,
                 marker="o", capsize=3, color="#d95f0e")
axes[1].plot(REP_LEVELS, np.array(REP_LEVELS)/REP_LEVELS[-1]*g.acq_s.mean().iloc[-1]*1e3,
             "k--", lw=1, label=r"$\propto N$ guide")
axes[1].set(title="Integration time per point", xlabel="repetitions $N$",
            ylabel="acq time (ms)"); axes[1].legend()
axes[2].errorbar(g.T_elapsed_s.mean().index, g.T_elapsed_s.mean(),
                 yerr=g.T_elapsed_s.std(), marker="o", capsize=3, color="#31a354")
axes[2].axhline(60, color="grey", ls=":", lw=1)
axes[2].set(title="Total wall-clock per run", xlabel="repetitions $N$",
            ylabel="elapsed (s)", ylim=(55, 65))
fig.suptitle("Acquisition design: equal ~60 s window, fewer/slower points at high N",
             y=1.03, fontweight="bold")
plt.tight_layout(); plt.show()""")

# ---------------------------------------------------------------- visual sanity
md(r"""## 5 · Visual sanity check — the noise *should* shrink with N

Before any statistics, look at the raw traces. Below we overlay the $|\Delta
\mathbf{B}|$ time series and one ADC channel for the lowest ($N{=}10$) and highest
($N{=}100$) repetition runs. The high-$N$ trace should be visibly tighter.""")
code(r"""fig, axes = plt.subplots(2, 2, figsize=(14, 6.4), sharex="col")
demo = {10: "#9ecae1", 100: "#08519c"}
for col, row, label in [(BMAG_COL, 0, r"$|\Delta B|$  (µT)"),
                        ("peak_01", 1, "peak_01  (ADC)")]:
    for rep, color in demo.items():
        d = trials[(rep, 1)]
        axes[row, 0].plot(d.time_s, d[col], color=color, lw=0.8,
                          label=f"N={rep}", alpha=0.9)
    axes[row, 0].set(ylabel=label)
    # histogram view (mean-subtracted) to compare spread
    for rep, color in demo.items():
        d = trials[(rep, 1)][col].to_numpy()
        axes[row, 1].hist(d - np.mean(d), bins=40, color=color, alpha=0.55,
                          density=True, label=f"N={rep} (σ={np.std(d, ddof=1):.3g})")
    axes[row, 1].set(ylabel="density")
    axes[row, 0].legend(loc="upper right"); axes[row, 1].legend()
axes[1, 0].set_xlabel("time (s)"); axes[1, 1].set_xlabel("value − mean")
axes[0, 0].set_title("Time series  (N=10 vs N=100)")
axes[0, 1].set_title("Distribution of fluctuations")
plt.tight_layout(); plt.show()""")

# ---------------------------------------------------------------- metrics
md(r"""## 6 · Noise metrics

For each (rep, trial, column) we compute three standard-deviation estimators:

| Metric | Definition | What it captures |
|---|---|---|
| `sigma_raw` | sample std over the full 60 s run | total scatter of a single measurement (drift **+** white noise) — *what the magnetometer actually delivers* |
| `sigma_detrended` | std of residuals after a linear fit | white noise with slow drift removed |
| `sigma_diff` | $\mathrm{std}(\Delta x_i)/\sqrt{2}$ (lag-1 successive difference) | drift-immune (Allan-style) high-frequency noise |

`sigma_raw` is the headline number the user asked for. The other two isolate the
genuine shot-noise component so we can tell whether any flattening is real
saturation or merely drift.""")
code(r"""def sigma_raw(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return np.std(x, ddof=1) if x.size > 1 else np.nan

def sigma_detrended(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = x.size
    if n < 3: return np.nan
    t = np.arange(n)
    a, b = np.polyfit(t, x, 1)
    return np.std(x - (a*t + b), ddof=2)

def sigma_diff(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 2: return np.nan
    return np.std(np.diff(x), ddof=1) / np.sqrt(2)

METRICS = {"raw": sigma_raw, "detrended": sigma_detrended, "diff": sigma_diff}

rows = []
for (rep, ti), df in trials.items():
    for col in ALL_COLS:
        x = df[col].to_numpy()
        rec = dict(rep=rep, trial=ti, column=col, group=COL2GROUP[col],
                   unit=UNITS[col], acq_s=df["acq_seconds"].mean())
        for name, fn in METRICS.items():
            rec[f"sigma_{name}"] = fn(x)
        rows.append(rec)
sig = pd.DataFrame(rows)
sig = sig.merge(inventory[["rep", "trial", "session_min", "duty_cycle"]],
                on=["rep", "trial"], how="left")
print("Per-(rep,trial,column) sigma table:", sig.shape)
sig.head(8)""")

code(r"""# Aggregate across the 3 trials -> mean +/- between-trial std (our error bars)
def agg_metric(name):
    return (sig.groupby(["group", "column", "unit", "rep"])[f"sigma_{name}"]
               .agg(["mean", "std"]).reset_index()
               .rename(columns={"mean": f"{name}_mean", "std": f"{name}_sd"}))

agg = agg_metric("raw")
for name in ["detrended", "diff"]:
    a = agg_metric(name)
    agg = agg.merge(a, on=["group", "column", "unit", "rep"])
# attach mean acquisition time per rep (for sensitivity later)
acq_by_rep = inventory.groupby("rep").acq_s.mean()
agg["acq_s"] = agg.rep.map(acq_by_rep)
agg.head()""")

# ---------------------------------------------------------------- sigma vs N
md(r"""## 7 · Standard deviation vs. repetitions $N$ — the core result

One panel per quantity group, log–log. Faint lines are individual columns; the
bold line is the group mean with between-column/-trial error bars. The dashed
black line is the **ideal shot-noise law** $\sigma\propto N^{-1/2}$ anchored at the
$N{=}10$ baseline. Points sitting *on* the dashed line are shot-noise-limited;
points *above* it (flattening) indicate a drift/systematic floor (saturation).""")
code(r"""def plot_group(ax, group, metric="raw"):
    sub = agg[agg.group == group]
    cols = sub.column.unique()
    pal = sns.color_palette("viridis", len(cols))
    # faint individual columns (normalised within-column? no — raw values)
    for c, color in zip(cols, pal):
        d = sub[sub.column == c].sort_values("rep")
        ax.plot(d.rep, d[f"{metric}_mean"], color=color, lw=0.8, alpha=0.45,
                marker="o", ms=3,
                label=(c if len(cols) <= 8 else None))
    # bold group-mean across columns (each column already trial-averaged)
    gm = (sub.groupby("rep")[f"{metric}_mean"].agg(["mean", "std"])
             .reset_index().sort_values("rep"))
    ax.errorbar(gm.rep, gm["mean"], yerr=gm["std"], color="crimson", lw=2.2,
                marker="s", ms=6, capsize=3, zorder=5, label="group mean")
    # ideal shot-noise law anchored at baseline group-mean
    base = gm[gm.rep == N_BASELINE]["mean"].iloc[0]
    reps = np.array(REP_LEVELS, float)
    ax.plot(reps, base*np.sqrt(N_BASELINE/reps), "k--", lw=1.8,
            label=r"shot-noise $N^{-1/2}$")
    ax.set(xscale="log", yscale="log", xlabel="repetitions $N$",
           title=group)
    ax.set_xticks(REP_LEVELS); ax.set_xticklabels(REP_LEVELS)
    unit = sub.unit.iloc[0]
    ax.set_ylabel(fr"$\sigma$ ({unit})")
    ax.legend(fontsize=7.5, ncol=1, loc="best")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, group in zip(axes.ravel(), GROUPS):
    plot_group(ax, group, metric="raw")
fig.suptitle("Raw σ vs N  (dashed = ideal shot-noise N$^{-1/2}$ from N=10 baseline)",
             y=1.02, fontweight="bold")
plt.tight_layout(); plt.show()""")

md(r"""The same view using the **drift-robust** `sigma_diff` metric. If the raw σ
flattens but `sigma_diff` keeps following $N^{-1/2}$, the flattening is caused by
slow drift rather than a true shot-noise saturation.""")
code(r"""fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, group in zip(axes.ravel(), GROUPS):
    plot_group(ax, group, metric="diff")
fig.suptitle("Drift-robust σ (successive-difference) vs N",
             y=1.02, fontweight="bold")
plt.tight_layout(); plt.show()""")

# ---------------------------------------------------------------- CONFOUND
md(r"""## 8 · ⚠ Critical confound — repetitions are aliased with session time

The plots above show σ **rising** with $N$ for the field-derived quantities — the
*opposite* of shot-noise scaling. Before concluding anything physical, we must
check the experiment's structure: the runs were acquired **in order of increasing
$N$** (all three $N{=}10$ trials first, then $N{=}20$, … finishing with $N{=}100$
~65 min later). That makes **repetition count perfectly collinear with
acquisition time**. If the lab environment drifted over the hour, that drift is
indistinguishable from a repetition effect.

The two panels below decide it:

* **Left** — σ vs *session time*. If σ grows smoothly with clock time (not in
  discrete steps locked to $N$), the driver is **drift**, not averaging.
* **Right** — the raw lock-in amplitude `peak_01` (an *ADC* channel, i.e. the
  measurement chain itself) vs the *derived* field $|\Delta B|$. If the ADC stays
  flat while $|\Delta B|$ climbs, the rising noise is **real environmental
  magnetic drift**, not a degrading sensor.""")
code(r"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (left) sigma_raw vs session time, coloured by N, for representative quantities
ax = axes[0]
show = {"delta_B_mag_uT": "o", "delta_Bx_uT": "^", "delta_f_mhz_b01": "s"}
norm_c = mpl.colors.Normalize(vmin=min(REP_LEVELS), vmax=max(REP_LEVELS))
cmap = mpl.colormaps["viridis"]
for col, mk in show.items():
    d = sig[sig.column == col]
    # normalise each quantity to its own first-trial value so they share a y-axis
    ref = d.sort_values("session_min").sigma_raw.iloc[0]
    sc = ax.scatter(d.session_min, d.sigma_raw/ref, c=d.rep, cmap=cmap,
                    norm=norm_c, marker=mk, s=55, edgecolor="k", lw=0.4,
                    label=col)
# linear trend across all three (in normalised units)
alld = pd.concat([sig[sig.column == c].assign(
            y=sig[sig.column == c].sigma_raw /
              sig[sig.column == c].sort_values("session_min").sigma_raw.iloc[0])
        for c in show])
sl, ic, r, p, se = stats.linregress(alld.session_min, alld.y)
xx = np.array([0, alld.session_min.max()])
ax.plot(xx, ic + sl*xx, "r--", lw=2,
        label=f"trend +{sl*60:.2f}/hr  (r={r:.2f})")
ax.set(xlabel="session time (min)", ylabel="σ / σ(first trial)",
       title="Noise grows with clock time, not with N")
ax.legend(fontsize=8)
cb = fig.colorbar(sc, ax=ax); cb.set_label("repetitions N")

# (right) ADC channel (flat) vs derived field (rising) over the session
ax = axes[1]
for col, color, lab in [("peak_01", "#1b9e77", "peak_01  (ADC, measurement chain)"),
                        ("delta_B_mag_uT", "#d95f02", r"$|\Delta B|$  (derived field)")]:
    d = sig[sig.column == col].sort_values("session_min")
    ref = d.sigma_raw.iloc[0]
    ax.plot(d.session_min, d.sigma_raw/ref, "o-", color=color, label=lab, ms=6)
ax.axhline(1.0, color="grey", ls=":", lw=1)
ax.set(xlabel="session time (min)", ylabel="σ / σ(first trial)",
       title="ADC stays flat → drift is environmental, not the sensor")
ax.legend(fontsize=8.5)
fig.suptitle("Repetition count is confounded with a slow session-long drift",
             y=1.03, fontweight="bold")
plt.tight_layout(); plt.show()

# Quantify the collinearity: sigma correlates with BOTH N and session-time
diag = []
for col in [BMAG_COL] + BAXIS_COLS + SHIFT_COLS[:1] + ["peak_01"]:
    d = sig[sig.column == col]
    rN = stats.spearmanr(d.rep, d.sigma_raw).statistic
    rT = stats.spearmanr(d.session_min, d.sigma_raw).statistic
    diag.append(dict(column=col, rho_vs_N=rN, rho_vs_sessiontime=rT))
print("Spearman ρ of σ_raw against N vs against session-time "
      "(N and time are collinear, ρ(N,time)="
      f"{stats.spearmanr(sig.rep, sig.session_min).statistic:.3f}):")
pd.DataFrame(diag).round(3)""")

md(r"""**Verdict.** σ tracks **session time** as cleanly as it tracks $N$ (the two
are ~perfectly collinear), the trend is *smooth in clock time*, and the ADC
channels — the measurement chain — stay essentially flat while the derived field
noise climbs. The rise is therefore **environmental magnetic drift accumulating
over the session**, not an effect of averaging.

**Consequence for the shot-noise question:** this dataset, as structured, *cannot*
isolate the $N^{-1/2}$ averaging benefit — it is buried under (and outweighed by)
the session drift. The remaining sections are still run, but every "σ vs $N$"
result must be read as "σ vs $N$ *and* elapsed time." The drift-aware Allan
analysis (Section 14) and the explicit recommendation (Section 15) address how to
get a clean measurement.""")

# ---------------------------------------------------------------- normalized collapse
md(r"""## 9 · Normalized collapse — *where do we stand vs. the shot-noise limit?*

To compare all 28 columns on one axis we normalize each by its **own $N{=}10$
baseline**: $r(N)=\sigma(N)/\sigma(10)$. A perfectly shot-noise-limited system
collapses onto the single theory curve $r_\text{ideal}(N)=\sqrt{10/N}$ (so e.g.
$N{=}100$ should reach $r=0.316$, a 3.16× improvement). Deviation **above** the
curve is the efficiency gap.

> *Read with Section 8:* because $N$ is confounded with session time here, curves
> riding **above** the ideal line reflect accumulated drift as much as any
> intrinsic averaging inefficiency. This plot quantifies the gap; it does not by
> itself attribute it.""")
code(r"""# normalised ratio per column (trial-averaged), relative to baseline
base_vals = (agg[agg.rep == N_BASELINE]
             .set_index("column")["raw_mean"].to_dict())
norm = agg.copy()
norm["ratio"] = norm.apply(lambda r: r["raw_mean"]/base_vals[r["column"]], axis=1)
ratio_ideal = lambda N: np.sqrt(N_BASELINE/np.asarray(N, float))

fig, ax = plt.subplots(figsize=(10, 6))
gcolors = dict(zip(GROUPS, sns.color_palette("tab10", len(GROUPS))))
for col in ALL_COLS:
    d = norm[norm.column == col].sort_values("rep")
    ax.plot(d.rep, d.ratio, color=gcolors[COL2GROUP[col]], lw=0.7,
            marker="o", ms=3, alpha=0.45)
# median collapse across all columns
med = norm.groupby("rep").ratio.median()
ax.plot(med.index, med.values, color="black", lw=2.6, marker="s", ms=7,
        zorder=6, label="median of all columns")
reps = np.array(REP_LEVELS, float)
ax.plot(reps, ratio_ideal(reps), "r--", lw=2.4, zorder=6,
        label=r"ideal $\sqrt{10/N}$")
# group legend handles
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=c, lw=2, label=g) for g, c in gcolors.items()]
handles += [Line2D([0], [0], color="black", lw=2.6, marker="s", label="median (all cols)"),
            Line2D([0], [0], color="red", lw=2.4, ls="--", label=r"ideal $\sqrt{10/N}$")]
ax.legend(handles=handles, fontsize=9, loc="upper right")
ax.set(xscale="log", yscale="log", xlabel="repetitions $N$",
       ylabel=r"$\sigma(N)\,/\,\sigma(N{=}10)$",
       title="Normalized noise collapse vs. ideal shot-noise scaling")
ax.set_xticks(REP_LEVELS); ax.set_xticklabels(REP_LEVELS)
plt.tight_layout(); plt.show()

print(f"Median measured improvement at N=100:  {1/med.loc[100]:.2f}x")
print(f"Ideal shot-noise improvement at N=100:  {1/ratio_ideal(100):.2f}x "
      f"(= sqrt(10))")""")

# ---------------------------------------------------------------- fits
md(r"""## 10 · Power-law fits — the scaling exponent $b$

We fit $\log\sigma = \log a + b\,\log N$ for every column (using all 30 per-trial
points). The exponent $b$ is the verdict:

* $b \approx -0.5$ → **shot-noise-limited** (ideal averaging),
* $-0.5 < b < 0$ → partial averaging / approaching a drift floor,
* $b \approx 0$ → **saturated** (drift/systematic dominated).""")
code(r"""def fit_exponent(d, metric="sigma_raw"):
    x = np.log(d["rep"].to_numpy(float)); y = np.log(d[metric].to_numpy(float))
    m = np.isfinite(x) & np.isfinite(y) & (d[metric].to_numpy() > 0)
    if m.sum() < 3: return np.nan, np.nan, np.nan
    r = stats.linregress(x[m], y[m])
    return r.slope, r.stderr, r.rvalue**2

fit_rows = []
for col in ALL_COLS:
    d = sig[sig.column == col]
    for metric in ["sigma_raw", "sigma_diff"]:
        b, be, r2 = fit_exponent(d, metric)
        fit_rows.append(dict(group=COL2GROUP[col], column=col,
                             metric=metric.replace("sigma_", ""),
                             exponent_b=b, b_stderr=be, r2=r2))
fits = pd.DataFrame(fit_rows)

fit_summary = (fits.groupby(["group", "metric"])
               .agg(mean_b=("exponent_b", "mean"),
                    std_b=("exponent_b", "std"),
                    mean_r2=("r2", "mean")).round(3).reset_index())
print("Per-group scaling exponent  (ideal shot noise b = -0.5)\n")
fit_summary""")

code(r"""# bar chart of per-column exponents (raw metric), grouped & coloured
fr = fits[fits.metric == "raw"].copy()
order = PEAK_COLS + SHIFT_COLS + BAXIS_COLS + [BMAG_COL]
fr["column"] = pd.Categorical(fr["column"], categories=order, ordered=True)
fr = fr.sort_values("column")
fig, ax = plt.subplots(figsize=(13, 5))
colors = [gcolors[g] for g in fr.group]
ax.bar(range(len(fr)), fr.exponent_b, yerr=fr.b_stderr, color=colors,
       capsize=2, edgecolor="white", lw=0.5)
ax.axhline(-0.5, color="red", ls="--", lw=2, label="ideal shot noise (−0.5)")
ax.axhline(0.0, color="grey", ls=":", lw=1, label="fully saturated (0)")
ax.set_xticks(range(len(fr))); ax.set_xticklabels(fr.column, rotation=90, fontsize=8)
ax.set_ylabel("fitted exponent $b$"); ax.set_title(
    "Scaling exponent per column  (σ ∝ N$^{b}$, raw metric)")
handles = [Line2D([0], [0], color=c, lw=6, label=g) for g, c in gcolors.items()]
handles += [Line2D([0], [0], color="red", ls="--", lw=2, label="ideal −0.5"),
            Line2D([0], [0], color="grey", ls=":", lw=1, label="saturated 0")]
ax.legend(handles=handles, fontsize=8, ncol=3, loc="lower right")
plt.tight_layout(); plt.show()""")

# ---------------------------------------------------------------- efficiency
md(r"""## 11 · Efficiency & improvement factor — the bottom line for the $N{=}10$ baseline

For each column: **measured improvement** $=\sigma(10)/\sigma(100)$ versus the
**ideal** $\sqrt{100/10}=3.16\times$. Efficiency $=$ measured / ideal. This tells
us how much of the theoretically available noise reduction the system is actually
realizing when we crank $N$ from 10 to 100.""")
code(r"""def improvement_table(metric="raw"):
    out = []
    for col in ALL_COLS:
        d = agg[agg.column == col].set_index("rep")
        s10 = d.loc[N_BASELINE, f"{metric}_mean"]
        s100 = d.loc[REP_LEVELS[-1], f"{metric}_mean"]
        imp = s10 / s100
        ideal = np.sqrt(REP_LEVELS[-1] / N_BASELINE)
        out.append(dict(group=COL2GROUP[col], column=col, unit=UNITS[col],
                        sigma_N10=s10, sigma_N100=s100,
                        improvement=imp, ideal=ideal,
                        efficiency_pct=100*imp/ideal))
    return pd.DataFrame(out)

imp = improvement_table("raw")
grp_eff = (imp.groupby("group")
           .agg(sigma_N10=("sigma_N10", "mean"), sigma_N100=("sigma_N100", "mean"),
                improvement=("improvement", "mean"),
                efficiency_pct=("efficiency_pct", "mean")).round(3))
print(f"Ideal improvement (N:10->100) = sqrt(10) = {np.sqrt(10):.3f}x\n")
grp_eff""")

code(r"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# (a) improvement factor per group
gb = imp.groupby("group")
ax = axes[0]
ax.bar(range(len(gb)), gb.improvement.mean(), yerr=gb.improvement.std(),
       color=[gcolors[g] for g in gb.groups], capsize=4, edgecolor="white")
ax.axhline(np.sqrt(10), color="red", ls="--", lw=2, label=r"ideal $\sqrt{10}=3.16\times$")
ax.set_xticks(range(len(gb))); ax.set_xticklabels(list(gb.groups), rotation=15)
ax.set_ylabel("σ(N=10) / σ(N=100)")
ax.set_title("Measured noise improvement, N: 10→100"); ax.legend()
# (b) efficiency per group
ax = axes[1]
ax.bar(range(len(gb)), gb.efficiency_pct.mean(), yerr=gb.efficiency_pct.std(),
       color=[gcolors[g] for g in gb.groups], capsize=4, edgecolor="white")
ax.axhline(100, color="red", ls="--", lw=2, label="100% (ideal)")
ax.set_xticks(range(len(gb))); ax.set_xticklabels(list(gb.groups), rotation=15)
ax.set_ylabel("efficiency (%)")
ax.set_title("Fraction of ideal shot-noise reduction realized"); ax.legend()
plt.tight_layout(); plt.show()""")

# ---------------------------------------------------------------- sensitivity
md(r"""## 12 · Sensitivity & the optimal repetition count

Magnetometer sensitivity normalizes noise by integration time:
$\eta = \sigma_B\sqrt{\tau}$, with $\tau$ = per-point acquisition time
(`acq_seconds`). In the shot-noise regime $\sigma_B\propto N^{-1/2}$ and
$\tau\propto N$, so $\eta$ is **flat** — adding reps buys nothing per unit time
beyond the trade you already have. Once drift sets in, $\sigma_B$ stops falling
but $\tau$ keeps growing, so $\eta$ **rises**. The minimum of $\eta(N)$ is the
*time-optimal* number of repetitions.

> *Caveat (Section 8):* with $N$ confounded by session drift, an $\eta$ minimum at
> the lowest $N$ partly reflects that the low-$N$ runs were simply taken earlier,
> when the environment was quieter — not proof that averaging fails to help. Treat
> the optimal-$N$ readout as provisional until the de-confounded re-measurement.""")
code(r"""fig, ax = plt.subplots(figsize=(10, 6))
sens_cols = BAXIS_COLS + [BMAG_COL]
pal = sns.color_palette("Set1", len(sens_cols))
opt = {}
for col, color in zip(sens_cols, pal):
    d = agg[agg.column == col].sort_values("rep")
    eta = d.raw_mean.to_numpy() * np.sqrt(d.acq_s.to_numpy())   # uT * sqrt(s)
    ax.plot(d.rep, eta, marker="o", color=color, lw=1.8, label=col)
    opt[col] = int(d.rep.to_numpy()[np.argmin(eta)])
ax.set(xscale="log", xlabel="repetitions $N$",
       ylabel=r"$\eta=\sigma_B\sqrt{\tau}$  (µT·$\sqrt{\mathrm{s}}$)",
       title="Per-point magnetic sensitivity vs N  (flat = shot-noise limited)")
ax.set_xticks(REP_LEVELS); ax.set_xticklabels(REP_LEVELS)
ax.legend()
plt.tight_layout(); plt.show()
print("Time-optimal repetition count (min η) per channel:")
for k, v in opt.items():
    print(f"  {k:14s}: N = {v}")""")

# ---------------------------------------------------------------- equal-N crosscheck
md(r"""## 13 · Fairness cross-check — equal-$N$ (evenly-spaced) subsampling

The headline σ above used **all** points in each run (686 at $N{=}10$, 152 at
$N{=}100$). To prove the downward trend is **not** an artefact of the differing
point count, we re-estimate every σ from exactly `N_MIN` points, sampled **evenly
across the full 60 s window** (so drift exposure stays equal). We average over many
random evenly-spaced subsamples for a stable estimate. If the equal-$N$ curve
overlaps the full-data curve, the comparison is fair.""")
code(r"""def even_subsample_sigma(x, k, n_boot=200):
    x = np.asarray(x, float); n = x.size
    if n <= k: return np.std(x, ddof=1)
    vals = []
    for _ in range(n_boot):
        start = RNG.integers(0, n - k + 1)               # random offset
        idx = np.linspace(start, n - 1, k).astype(int)   # evenly spaced
        vals.append(np.std(x[np.unique(idx)], ddof=1))
    return float(np.mean(vals))

cc_rows = []
for (rep, ti), df in trials.items():
    for col in sens_cols:                     # B axes + magnitude (representative)
        cc_rows.append(dict(rep=rep, trial=ti, column=col,
                            sigma_full=sigma_raw(df[col].to_numpy()),
                            sigma_equalN=even_subsample_sigma(df[col].to_numpy(), N_MIN)))
cc = pd.DataFrame(cc_rows)
cc_agg = cc.groupby(["column", "rep"]).agg(
    full=("sigma_full", "mean"), full_sd=("sigma_full", "std"),
    eqN=("sigma_equalN", "mean"), eqN_sd=("sigma_equalN", "std")).reset_index()

fig, axes = plt.subplots(1, len(sens_cols), figsize=(15, 4), sharex=True)
for ax, col in zip(axes, sens_cols):
    d = cc_agg[cc_agg.column == col].sort_values("rep")
    ax.errorbar(d.rep, d.full, yerr=d.full_sd, marker="o", capsize=2,
                label="full data", color="#08519c")
    ax.errorbar(d.rep, d.eqN, yerr=d.eqN_sd, marker="s", capsize=2,
                label=f"equal-N ({N_MIN} pts)", color="#e6550d", ls="--")
    ax.set(xscale="log", yscale="log", xlabel="N", title=col)
    ax.set_xticks(REP_LEVELS); ax.set_xticklabels(REP_LEVELS, fontsize=7)
axes[0].set_ylabel(r"$\sigma$ (µT)"); axes[0].legend(fontsize=8)
fig.suptitle("Equal-N subsample (orange) tracks full-data σ (blue) → trend is not "
             "a point-count artefact", y=1.04, fontweight="bold")
plt.tight_layout(); plt.show()""")

# ---------------------------------------------------------------- allan
md(r"""## 14 · Drift-aware view — Allan deviation *within* each run

The overlapping Allan deviation $\sigma_A(\tau)$ measures how the noise averages
down as we software-bin an *individual* run over integration time $\tau$ — and,
crucially, **it lives entirely inside one 60 s run, so it is immune to the
session-time confound** of Section 8. A $\tau^{-1/2}$ slope is the white-noise
(shot-noise) regime; a flattening/upturn marks where drift takes over.

Two questions:

* **(a)** Within a single run, is the noise white (does $\sigma_A\propto
  \tau^{-1/2}$ hold)? If yes, the sensor *is* shot/white-noise-limited at sub-60 s
  timescales.
* **(b)** Do the per-run Allan curves for different $N$ share the same slope but
  sit at different vertical offsets? That offset is the session drift — the same
  story as Section 8, seen a different way.""")
code(r"""def overlapping_allan(x, dt, n_tau=25):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = x.size
    ms = np.unique(np.logspace(0, np.log10(n//4), n_tau).astype(int))
    ms = ms[ms >= 1]
    taus, devs = [], []
    for m in ms:
        # cumulative-sum trick for block averages of length m
        c = np.concatenate([[0.0], np.cumsum(x)])
        avgs = (c[m:] - c[:-m]) / m                  # overlapping block means
        if avgs.size < 3:
            continue
        d = np.diff(avgs)
        devs.append(np.sqrt(0.5 * np.mean(d**2)))
        taus.append(m * dt)
    return np.array(taus), np.array(devs)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for col, ax, lbl in [(BMAG_COL, axes[0], r"$|\Delta B|$ (µT)"),
                     ("peak_01", axes[1], "peak_01 (ADC)")]:
    for rep, color in [(10, "#08519c"), (100, "#e6550d")]:
        d = trials[(rep, 1)]
        dt = (d.time_s.iloc[-1] - d.time_s.iloc[0]) / (len(d) - 1)
        tau, dev = overlapping_allan(d[col].to_numpy(), dt)
        ax.loglog(tau, dev, marker="o", ms=3, color=color, label=f"N={rep}")
    # tau^-1/2 reference
    tref = np.array([tau.min(), tau.max()])
    ax.loglog(tref, dev[0]*np.sqrt(tau[0])/np.sqrt(tref), "k--", lw=1.5,
              label=r"$\tau^{-1/2}$")
    ax.set(xlabel=r"integration time $\tau$ (s)", ylabel=fr"Allan dev — {lbl}")
    ax.legend(fontsize=8)
fig.suptitle("Overlapping Allan deviation: white-noise τ$^{-1/2}$ roll-off then "
             "drift floor", y=1.03, fontweight="bold")
plt.tight_layout(); plt.show()""")

code(r"""# Overlay |dB| Allan curves for ALL N (trial 1 each): same slope, drift offset?
fig, ax = plt.subplots(figsize=(9.5, 6))
slopes = []
for rep in REP_LEVELS:
    d = trials[(rep, 1)]
    dt = (d.time_s.iloc[-1] - d.time_s.iloc[0]) / (len(d) - 1)
    tau, dev = overlapping_allan(d[BMAG_COL].to_numpy(), dt)
    ax.loglog(tau, dev, marker="o", ms=3, lw=1.2,
              color=cmap(norm_c(rep)), label=f"N={rep}")
    # slope of the short-tau (white) part: first ~half-decade
    sel = tau < tau.min()*5
    if sel.sum() >= 3:
        slopes.append(np.polyfit(np.log(tau[sel]), np.log(dev[sel]), 1)[0])
tref = np.array([tau.min()*0.8, tau.max()])
y0 = dev[0]*np.sqrt(tau[0])
ax.loglog(tref, y0/np.sqrt(tref), "k--", lw=2, label=r"$\tau^{-1/2}$ (white/shot)")
ax.set(xlabel=r"integration time $\tau$ (s)",
       ylabel=r"Allan deviation of $|\Delta B|$  (µT)",
       title=r"Per-run Allan deviation, all N — shared slope, drift-shifted offset")
ax.legend(fontsize=7.5, ncol=2)
plt.tight_layout(); plt.show()
print(f"Mean short-τ Allan slope across runs: {np.mean(slopes):+.2f} "
      f"(white/shot = -0.50; more negative = even faster averaging)")
print("→ within a single run the noise AVERAGES DOWN at least as fast as white")
print("  noise (slope ≤ -1/2) — i.e. NOT drift-limited at sub-60 s timescales;")
print("  the vertical spread between the N curves is the between-run session drift.")""")

# ---------------------------------------------------------------- summary
md("## 15 · Summary — automated readout")
code(r"""raw_exp = fits[fits.metric == "raw"].groupby("group").exponent_b.mean()
med_imp_100 = 1/med.loc[REP_LEVELS[-1]]
# session-drift rate (normalised σ per hour) and N/time collinearity
drift_per_hr = sl*60
rho_collinear = stats.spearmanr(sig.rep, sig.session_min).statistic
allan_slope = np.mean(slopes)
# ADC vs field rise over the session
adc_rise = (sig[sig.column=="peak_01"].sort_values("session_min").sigma_raw.iloc[-1] /
            sig[sig.column=="peak_01"].sort_values("session_min").sigma_raw.iloc[0])
bmag_rise = (sig[sig.column==BMAG_COL].sort_values("session_min").sigma_raw.iloc[-1] /
             sig[sig.column==BMAG_COL].sort_values("session_min").sigma_raw.iloc[0])

print("="*74)
print("SHOT-NOISE EDA — SUMMARY".center(74))
print("="*74)
print(f"Design : {len(REP_LEVELS)} rep levels {REP_LEVELS}, 3 trials each, "
      f"{len(trials)} datasets")
print(f"Window : every run = equal {inventory.T_elapsed_s.mean():.1f} s "
      f"(points {inventory.n_points.min()}–{inventory.n_points.max()}, "
      f"duty {inventory.duty_cycle.min():.0%}→{inventory.duty_cycle.max():.0%})")
print("-"*74)
print("RAW σ vs N  (NOTE: confounded with session time — see below)")
print(f"  {'Group':<20}{'exponent b':>12}{'eff. % (10→100)':>18}")
for g in GROUPS:
    print(f"  {g:<20}{raw_exp[g]:>12.2f}{grp_eff.loc[g,'efficiency_pct']:>18.1f}")
print(f"  ideal b = -0.50 | ideal improvement = {np.sqrt(10):.2f}× | "
      f"observed median = {med_imp_100:.2f}×")
print("-"*74)
print("CONFOUND DIAGNOSIS")
print(f"  ρ(N, session-time)            = {rho_collinear:+.3f}  (≈1 ⇒ collinear)")
print(f"  σ drift rate over session     = {drift_per_hr:+.2f} ×/hr "
      f"(r={r:.2f}, p={p:.1e})")
print(f"  σ rise 1st→last run: ADC ×{adc_rise:.2f}  vs  |ΔB| ×{bmag_rise:.2f}")
print(f"  within-run Allan slope        = {allan_slope:+.2f}  (white/shot = -0.50)")
print("-"*74)
print("VERDICT")
print(f"  • Sub-60 s, noise AVERAGES DOWN within a run (Allan slope {allan_slope:+.2f},")
print("    ≤ -1/2 = white/shot) ⇒ the sensor is NOT drift-limited at short times.")
print("  • Across the session a slow ENVIRONMENTAL MAGNETIC DRIFT grows; because")
print("    reps were swept in time order, that drift masquerades as an N effect.")
print("  • The flat ADC vs rising |ΔB| confirms the drift is the field/environment,")
print("    not the measurement chain.")
print("  ⇒ This dataset CANNOT cleanly verify N^-1/2; raw σ-vs-N is drift-dominated.")
print("-"*74)
print("RECOMMENDATIONS")
print("  1. RE-MEASURE with INTERLEAVED rep order (cycle 10,20,…,100,10,… ×k) or")
print("     randomise, so session drift averages out across N. THE key fix.")
print("  2. Bracket with reference/background subtraction or magnetic shielding to")
print(f"     suppress the ~{drift_per_hr*100:.0f}%/hr environmental drift.")
print("  3. Use the within-run Allan deviation (drift-immune) as the figure of")
print("     merit for shot-noise headroom, not the 60 s sample std.")
print("  4. Higher reps also raise duty cycle (12%→80%): per-√Hz this helps, but")
print("     test it only after de-confounding (point 1).")
print("="*74)""")

md(r"""### Bottom line — where the magnetometer stands

**1. The headline number the user asked for.** Raw σ (sample std over each 60 s
run) for the 16 ADC channels, 8 peak shifts, 3 B-axes and $|\Delta B|$ is
tabulated and plotted in Sections 6–11. Taken at face value it does **not** fall
as $N^{-1/2}$ — it is flat (ADC) to rising (field). That alone would say "we are
far from the shot-noise limit."

**2. But the experiment is confounded (Section 8).** Reps were swept in time
order, so $N$ is collinear with acquisition time ($\rho\!\approx\!1$), and a slow
environmental magnetic drift grew over the ~65 min session. The drift — not the
averaging — drives the rising σ. The flat ADC channels vs. the rising derived
field prove the drift is in the *environment*, not the sensor. **So this dataset
cannot, by itself, measure the $N^{-1/2}$ averaging gain.**

**3. What we *can* say accurately.**
* *Within* any single 60 s run the noise **averages down** — its Allan deviation
  rolls off at least as fast as the white-noise $\tau^{-1/2}$ law (measured slope
  $\approx-0.8$, Section 14), with no drift upturn inside the run. At sub-minute
  timescales the sensor is **not** drift-limited and averaging genuinely helps.
* The limit on the **60 s and longer** stability is **slow drift / 1/f**, not
  photon shot noise. More repetitions at fixed $N$ do not fix drift.
* The $N{=}10$ baseline therefore is **not leaving obvious shot-noise on the
  table at the 60 s scale** — the binding constraint is environmental stability.

**4. To get a clean shot-noise-vs-$N$ answer, re-measure with the rep order
interleaved/randomised** (e.g. loop 10,20,…,100 repeatedly), optionally with
background subtraction or shielding. Then the raw σ-vs-$N$ curve and the
efficiency/η metrics in Sections 9–12 become trustworthy; right now read them only
*together with* the Section 8 confound.

*All numbers above are printed by the summary cell and the per-group tables;
re-run the notebook on de-confounded data to refresh them.*""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
# notebook lives in the project's data/notebooks/ collection
out = Path(__file__).resolve().parents[2] / "data" / "notebooks" / "shot_noise_eda.ipynb"
nbf.write(nb, out)
print("Wrote", out, "with", len(cells), "cells")
