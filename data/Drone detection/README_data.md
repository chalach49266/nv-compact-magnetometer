# Raw data — 2026-08-19 drone detection test

A copy of every stream the deck uses, kept here so this folder reproduces the
deck on its own. **The originals live in `nv_magnetometer_project/data/` and are
untouched** — this is a copy, not a move. `make_figs.py` reads from this folder
if it is present and falls back to the project tree if it is not.

Total ≈ 449 MB.

## `lockin_multipoint/` — 16-point streaming, 30 s each

13 runs × 7 files (`.csv`, `_wide.csv`, `_vector_rows.csv`,
`_projection_rows.csv`, `_result.png`, `_run_result.png`, `_fft.png`):

| condition | runs |
|---|---|
| drone | 173633, 174657, 174908, 175409, 180601, 181031, 181300, 182456, 183014, 183316 |
| no drone | 174515, 175626, 181817 |

Plus the eight `parked_plan_odmr_sweep_20260819_*` calibrations (173557, 174412,
175122, 180438, 180947, 181738, 182417, 182924), one of which precedes each run
and defines the zero of `ΔB`.

## `twopoint_lockin/` — two-point streaming, 30 s each

9 runs × 4 files (2 of them lack the `_fft.png` / `_result.png` pair):

| condition | runs |
|---|---|
| drone in front, whole run | 171802, 172711, 173311 |
| no drone | 172841, 173017 |
| unlabelled, same session | 171057, 171419, 171955, 173426 |

Plus `twopoint_calibration_odmr_sweep_20260819_*.json` (170624, 171632, 171739,
172650) and the matching `twopoint_plan_odmr_sweep_*.csv`.

## `odmr_sweeps/`

The four full ODMR sweeps the two-point calibrations point at, named in each
calibration JSON's `odmr_csv` field.

## Not copied

Five 16-point streams exist in the project tree for 2026-08-19 but were never
labelled at the bench (175232, 180912, 182302, 182704, 183208). They are
excluded from every drone / no-drone comparison and are not copied here.

See `../run_log_2026-08-19.md` for the bench conditions and
`../METHODS.md` for how each file is used.
