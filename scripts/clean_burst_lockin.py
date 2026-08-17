#!/usr/bin/env python3
"""Recover a usable time series from a burst-mode two-point lock-in CSV.

Burst runs contain three artefacts that are not measurements, traced in
`docs/2026-08-06_twopoint_timing/` and still present on 2026-08-14:

  1. **Stale batches.** The accumulated buffer is read before the new run refills
     it, so every second acquire returns a bit-identical replay of the previous
     one. Half the samples are duplicates, which doubles the apparent rate and
     makes every real glitch appear twice about one burst period apart -- the
     "smaller, less periodic" spikes.
  2. **A row-0 transient.** Row 0 of each batch is not a copy; it reads high
     (+17 ADC on the 2026-08-14 run, several percent on 2026-08-06) and lands
     well off the trace. This is the large, strictly periodic spike.
  3. **A broken time axis.** Samples are stamped across the *measured* acquire
     window, which alternates ~12 ms and ~120 ms, so half the data is compressed
     ~10x in time. The dead time between bursts also appears as a gap -- the
     visible "break".

    python scripts/clean_burst_lockin.py <live_csv>
    python scripts/clean_burst_lockin.py <live_csv> --no-despike --outdir analysis/
    python scripts/clean_burst_lockin.py <live_csv> --k-sigma 5 --window 21

Outputs `<name>_clean.csv`, `<name>_clean.png` and a before/after noise report.
The cleaned file is self-describing: `segment` marks contiguous acquisitions and
must never be filtered or transformed across, and `time_s` is rebuilt on the FPGA
cadence rather than on host arrival times.

What it cannot do is invent the samples lost to the dead time between bursts: it
recovers a correct ~3.1 kHz record from a file that claimed 6.2 kHz, it does not
recover 6.2 kHz.

Implementation note
-------------------
All of the logic now lives in `notebook_modules/twopoint_postprocess.py`, which
also serves the notebook's per-mode analysis cells and handles averaged and
streaming files. This script is the burst-specific CLI on top of it, kept because
the flags are in muscle memory and in the 2026-08-06 document. For the general
entry point see:

    python -m twopoint_postprocess <csv> [--outdir DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "notebook_modules"))

import burst_qc as qc                       # noqa: E402
import twopoint_postprocess as tpp           # noqa: E402

# Re-exported for callers that imported them from here before the move.
recover_conversion = tpp.recover_conversion
recompute_from_counts = tpp.recompute_from_counts
robust_sigma = tpp.robust_sigma
despike_by_segment = tpp.despike_by_segment
GAMMA_NV_MHZ_PER_UT = tpp.GAMMA_NV_MHZ_PER_UT


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
    ap.add_argument("--despike-channels", action="store_true",
                    help="filter the two parked channels independently instead of "
                         "the differential. Usually RAISES sigma -- see "
                         "twopoint_postprocess.despike_shift. Diagnostic only.")
    ap.add_argument("--notch-mains", action="store_true",
                    help="notch 60 Hz and harmonics inside each burst segment")
    ap.add_argument("--window", type=int, default=11, help="Hampel trailing window")
    ap.add_argument("--k-sigma", type=float, default=4.0, help="Hampel threshold")
    ap.add_argument("--sigma-floor", type=float, default=None,
                    help="override the auto-derived sigma floor")
    ap.add_argument("--sigma-cap", type=float, default=None,
                    help="override the auto-derived sigma cap")
    args = ap.parse_args()

    if not args.csv.is_file():
        print(f"error: no such file: {args.csv}", file=sys.stderr)
        return 1
    outdir = args.outdir or args.csv.parent
    outdir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.csv)
    print(f"input   : {args.csv.name}  ({len(raw)} rows)")

    if not qc.is_burst(raw):
        mode = tpp.detect_mode(raw)
        print(f"\nThis file is a {mode} run, not burst mode. The stale-batch and row-0")
        print("artefacts are specific to burst mode. Use the general entry point, which")
        print("has a pipeline for this mode:")
        print(f"\n    python -m twopoint_postprocess {args.csv}\n")
        return 0

    despike = False if args.no_despike else ("channels" if args.despike_channels
                                             else "shift")
    result = tpp.process_burst(
        raw,
        drop_stale=not args.keep_stale,
        drop_first_sample=not args.keep_first_sample,
        despike=despike,
        notch_mains=args.notch_mains,
        source=args.csv,
        window=args.window,
        k_sigma=args.k_sigma,
        sigma_floor=args.sigma_floor,
        sigma_cap=args.sigma_cap,
    )

    r = result.report
    print(f"cadence : {r['cadence_us']:.1f} us/rep  ({r['within_burst_rate_hz']:.0f} Hz "
          f"inside a burst), from the {r['cadence_source']}")
    if r.get("cadence_source") == "readout quantum":
        print(f"          wall-clock would say {r['cadence_wallclock_us']:.1f} us; the "
              f"difference is the {100 * r['host_share_of_burst']:.0f}% host share of "
              f"each acquire, which is not FPGA time")
    print(f"\nstage 1 -- stale batches  : {r['n_stale_batches']}/{r['n_batches']} "
          f"({100 * r['stale_fraction']:.0f}%)")
    print(f"stage 2 -- row-0 transient: {r['first_sample_transients_dropped']} rows "
          f"dropped from real batches")
    print(f"stage 3 -- re-timed on the cadence; "
          f"{result.clean['segment'].nunique()} contiguous segments, gaps left as gaps")
    print(f"stage 4 -- despike        : {r.get('despiked_samples', 0)} samples "
          f"({'differential' if despike == 'shift' else despike})")

    raw_sigma = float(raw["peak_shift_kHz"].std())
    print(f"\n            {'before':>12s} {'after':>12s}")
    print(f"  rows      {len(raw):12d} {len(result.clean):12d}")
    print(f"  rate (Hz) {r['recorded_rate_hz']:12.0f} {r['true_rate_hz']:12.0f}")
    print(f"  sigma(kHz){raw_sigma:12.1f} {result.spectrum.sigma_khz:12.1f}")
    print()
    print(result.spectrum.describe())
    for note in result.notes:
        print(f"  NOTE: {note}")

    out_csv = outdir / f"{args.csv.stem}_clean.csv"
    result.clean.to_csv(out_csv, index=False)
    out_png = outdir / f"{args.csv.stem}_clean.png"
    result.figure(out_png)
    print(f"\nwrote {out_csv}")
    print(f"wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
