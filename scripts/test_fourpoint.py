"""Four-point notebook: does the analysis path actually run, end to end?

Builds a synthetic four-point streamed run with a KNOWN field step and a KNOWN
thermal drift, then executes Step 5C's own source out of the .ipynb inside a real
IPython shell. Checks that each peak gets its own result and FFT PNG, and that the
combined channels recover the two inputs separately -- which is the entire reason
the second peak is there.

Run:  python scripts/test_fourpoint.py      (exit 0 = all checks passed)
"""
from __future__ import annotations

import ast, contextlib, io, json, shutil, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "notebook_modules"))

RESULTS = []
def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

import fourpoint_runner as fr

GAMMA = 28.024e-3          # MHz per uT
F0 = (2870.0, 2950.0)
FS = 2083.333
N = 6000

# ------------------------------------------------------------------ synthetic
# A field step of +300 nT at the halfway point, on top of a linear D drift of
# -20 kHz over the record. The two must come back out separately.
t = np.arange(N) / FS
B_nt = np.where(t > t[-1] / 2, 300.0, 0.0)
dD_khz = np.linspace(0.0, -20.0, N)
rng = np.random.default_rng(0)
noise = rng.normal(0, 2.0, (2, N))                       # 2 kHz per peak, white

df_low = (dD_khz - B_nt * 1e-3 * GAMMA * 1e3 + noise[0]) * 1e-3     # MHz
df_high = (dD_khz + B_nt * 1e-3 * GAMMA * 1e3 + noise[1]) * 1e-3

rows = []
for k in range(N):
    row = {"packet": k // 200, "rep_index": k * 4, "reps_averaged": 4,
           "time_s": float(t[k]), "timestamp_epoch_s": 0.0}
    for j in range(4):
        row[f"peak_{j+1:02d}"] = 900.0 + 5 * rng.normal()
        row[f"peak_{j+1:02d}_freq_mhz"] = [2866., 2874., 2946., 2954.][j]
    for i, (d, f0) in enumerate(((df_low, F0[0]), (df_high, F0[1])), start=1):
        row.update({f"pk{i}_z_minus": 0.9, f"pk{i}_z_plus": 0.9,
                    f"pk{i}_lockin_signal": 0.0,
                    f"pk{i}_delta_f_mhz": float(d[k]),
                    f"pk{i}_peak_shift_kHz": float(d[k]) * 1e3,
                    f"pk{i}_f0_mhz": f0,
                    f"pk{i}_f_new_mhz": f0 + float(d[k]),
                    f"pk{i}_B_shift_uT": float(d[k]) / GAMMA})
    row.update({"z_minus": row["pk1_z_minus"], "z_plus": row["pk1_z_plus"],
                "lockin_signal": row["pk1_lockin_signal"],
                "delta_f_mhz": row["pk1_delta_f_mhz"],
                "peak_shift_kHz": row["pk1_peak_shift_kHz"],
                "f_new_mhz": row["pk1_f_new_mhz"], "B_shift_uT": row["pk1_B_shift_uT"]})
    rows.append(row)
df = pd.DataFrame(rows)

print("\n1. the splitters and the combined channels")
check("n_pairs sees two peaks", fr.n_pairs(df) == 2, str(fr.n_pairs(df)))
sub = fr.split_pair(df, 2)
check("split_pair gives a two-point frame",
      {"peak_01", "peak_02", "z_minus", "z_plus", "peak_shift_kHz"} <= set(sub.columns))
check("and takes peak 2's parked points",
      float(sub["peak_01_freq_mhz"].iloc[0]) == 2946.0, str(sub["peak_01_freq_mhz"].iloc[0]))

dsub = fr.split_difference(df)
d_khz = dsub["peak_shift_kHz"].to_numpy()
got_split = d_khz[N // 2 + 100:].mean() - d_khz[:N // 2 - 100].mean()
check("split_difference tracks the splitting, = 2*gamma*B",
      abs(got_split - 2 * 300.0 * 1e-3 * GAMMA * 1e3) < 1.5,
      f"{got_split:+.2f} kHz for a 300 nT step (expected "
      f"{2 * 300.0 * 1e-3 * GAMMA * 1e3:+.2f})")
check("and it is a frame process() accepts",
      {"time_s", "peak_01", "peak_02", "z_minus", "z_plus", "delta_f_mhz",
       "peak_shift_kHz"} <= set(dsub.columns))
dd_only = dsub["peak_shift_kHz"].to_numpy()
check("the D drift is gone from the difference",
      abs(dd_only[-200:].mean() - dd_only[:200].mean() - got_split) < 1.5,
      "no residual thermal ramp")

comb = fr.combine(df, GAMMA)
step = comb["B_par_uT"].to_numpy() * 1e3
got_B = step[N // 2 + 100:].mean() - step[:N // 2 - 100].mean()
check("combine() recovers the 300 nT field step", abs(got_B - 300.0) < 15.0,
      f"{got_B:+.1f} nT")
dd = comb["dD_khz"].to_numpy()
got_D = dd[-200:].mean() - dd[:200].mean()
check("combine() recovers the -20 kHz D drift", abs(got_D + 20.0) < 1.0, f"{got_D:+.2f} kHz")
lone = comb["pk1_peak_shift_kHz"].to_numpy()
check("a single peak conflates them", abs((lone[-200:].mean() - lone[:200].mean())) > 5.0,
      f"peak 1 alone drifts {lone[-200:].mean() - lone[:200].mean():+.1f} kHz -- "
      "field and D together")

# ---------------------------------------------------- 2. run the real cell
print("\n2. Step 5C, executed out of the .ipynb")
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.clear_instance()
ip = InteractiveShell.instance()
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    ip.run_cell("%matplotlib inline")

tmp = Path(tempfile.mkdtemp(prefix="fourpt_"))
csv = tmp / "fourpoint_lockin_stream_20260818_120000.csv"
df.to_csv(csv, index=False)

nb = json.loads((REPO / "Modules" / "Fourpoint_Lockin_module.ipynb").read_text())
cell = next(("".join(c["source"]) for c in nb["cells"]
             if c["cell_type"] == "code" and "Step 5C" in "".join(c["source"])))
ip.run_cell(f"""
import sys
sys.path.insert(0, {str(REPO / 'notebook_modules')!r})
from pathlib import Path
import matplotlib.pyplot as plt
TWOPOINT_DIR = Path({str(tmp)!r})
STREAM_LAST_CSV = TWOPOINT_DIR / {csv.name!r}
GAMMA_NV_MHZ_PER_UT = {GAMMA!r}
import figure_autosave
figure_autosave.enable(TWOPOINT_DIR, verbose=False)
""")
with contextlib.redirect_stdout(io.StringIO()) as out:
    res = ip.run_cell(cell)
if res.error_in_exec:
    check("Step 5C runs", False, repr(res.error_in_exec))
else:
    check("Step 5C runs", True)
    pngs = sorted(p.name for p in tmp.glob("*.png"))
    stem = csv.stem
    want = {f"{stem}_peak1_result.png", f"{stem}_peak1_fft.png",
            f"{stem}_peak2_result.png", f"{stem}_peak2_fft.png",
            f"{stem}_diff_result.png", f"{stem}_diff_fft.png"}
    check("all THREE channels get a shift plot and an FFT plot",
          want <= set(pngs), f"{len(pngs)} PNGs: {sorted(set(pngs) & want)}")
    check("plus a combined figure",
          any(p.startswith(f"{stem}_combined") for p in pngs),
          ", ".join(p for p in pngs if "combined" in p))
    text = out.getvalue()
    check("it reports both peaks separately", text.count("PEAK ") >= 2)
    check("and the difference as its own channel", "DIFFERENCE" in text)
    check("with the 2*gamma caveat stated", "2*gamma" in text or "2x the field" in text)
    check("and the two combined channels",
          "dB_par" in text and "dD" in text)

shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
sys.exit(0 if all(RESULTS) else 1)
