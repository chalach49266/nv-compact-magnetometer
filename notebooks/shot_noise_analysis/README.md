# Shot-Noise Scaling EDA — lock-in repetitions

Analyzes how the measurement standard deviation scales with the number of lock-in
repetitions `N` (10 → 100), for the dataset
`data/lockin_multipoint/shot_noise_experiment/` (10 rep levels × 3 trials = 30 runs).

## Files
- **`../../data/notebooks/shot_noise_eda.ipynb`** — the executed EDA notebook
  (open this). Self-contained; re-run top-to-bottom with the `python3` kernel in
  `qickdawg_venv`. Its data path resolves relative to that location.
- **`_build_notebook.py`** — generator that rebuilds the notebook from scratch and
  writes it to `data/notebooks/shot_noise_eda.ipynb` (`python3 _build_notebook.py`).
  Edit here, not the `.ipynb`, for structural changes.

## What it measures
Standard deviation of four quantity groups vs `N`:
16 ADC channels (`peak_01..16`), 8 peak shifts (`delta_f_mhz_b01..08`),
3 B-field axes (`delta_B{x,y,z}_uT`), and the field magnitude `|ΔB|`.

## Headline finding
Every run spans the **same ~60 s window**, so point-count differences only affect
estimator precision (handled with between-trial error bars + an equal-N subsample
cross-check). The **dominant issue is a confound**: reps were swept in time order,
so `N` is collinear (ρ≈0.996) with acquisition time, and a slow **environmental
magnetic drift (~+70 %/hr)** grew over the ~65 min session. Raw σ therefore *rises*
with `N` — that is the drift, not averaging.

- Within any single run the noise **averages down** (Allan slope ≈ −0.8 ≤ −½) → the
  sensor is **not** drift-limited at sub-60 s timescales; averaging genuinely helps.
- The flat ADC channels vs. rising derived-field σ confirm the drift is
  environmental, not in the measurement chain.

**To get a clean `N^-1/2` measurement, re-acquire with interleaved/randomised rep
order** so session drift averages out across `N`. See the notebook's Section 8
(confound), Section 14 (drift-immune Allan view), and Section 15 (summary).
