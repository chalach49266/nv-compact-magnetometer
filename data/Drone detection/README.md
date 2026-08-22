# Drone detection test — 2026-08-19

Everything from the 2026-08-19 drone-detection session: the raw streams, the
analysis, and the deck built from them.

```
Drone detection/
  README.md                  this file
  README_data.md             what each raw file is, and what was deliberately left out
  prompt drone.md            the original request and the bench notes it carried
  run_log_2026-08-19.md      condition recorded for every run
  METHODS.md                 full method — filtering, detector, clock check, physics
  lockin_multipoint/         13 runs × 7 files + 8 parked-plan calibrations
  twopoint_lockin/           9 runs + 4 calibrations + plan sweeps
  odmr_sweeps/               the 4 sweeps the two-point calibrations point at
  slides/
    drone_detection_2026-08-19.pdf    30-slide deck
    drone_detection_2026-08-19.tex    source
    make_figs.py                      regenerates every figure and number
    numbers.txt                       audit trail for every number the deck quotes
    figs/                             22 figures, PDF + PNG
```

## Rebuild

```bash
cd slides
python3 make_figs.py                      # -> figs/, numbers.txt
pdflatex drone_detection_2026-08-19.tex   # twice, for the frame counter
```

`make_figs.py` finds the raw streams by looking, in order, for a `data/`
subfolder next to itself, then its parent directory (which is how it resolves
here), then `nv_magnetometer_project/data/Drone detection`, then the live
project tree. It works unmoved or copied elsewhere.

## What the session showed

* The drone is detected as a **static magnetic dipole**. It emits nothing — the
  drone/no-drone spectral ratio is flat at 1.2 across 0–2 kHz.
* Parked on top: a **16–27 µT** step in `|ΔB|`, held for seconds, no motion needed.
* In front at 10 / 20 / 30 cm: **21 / 4.6 / 1.1 µT**, falling as `r^(−2.2±0.2)`.
* Detector thresholded on the three drone-free runs: **10/10 flagged, 0 false
  alarms**.
* Useful static range today ≈ **13 cm**, growing only as `(m/σ)^(1/3)`.
* The ~60 Hz line is the **building mains**, not the cryostat pump. A sample-clock
  error makes it read 61.72 Hz (two-point) and 61.02 Hz (16-point); correcting
  each stream by its own wall clock puts both at 60.1 Hz. See METHODS.md §3a.

## Note on size and git

This folder is ~469 MB and sits inside the git-tracked `nv_magnetometer_project`
repo, where `data/` is **not** gitignored. Committing it would add ~469 MB to
history — worth deciding deliberately before the next `git add`. Largest single
file is a 30 MB two-point CSV, so nothing trips GitHub's 100 MB per-file limit,
but the repo total would grow a lot. Options: add
`nv_magnetometer_project/data/Drone detection/` to `.gitignore`, keep only
`slides/` under version control, or use Git LFS for the CSVs.

A second copy of the deck (without the raw data, ~20 MB) lives at
`output/slides/2026-08-19_drone_detection/`.
