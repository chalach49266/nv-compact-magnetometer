# Two-point lock-in — master reference

`TWOPOINT_MASTER_REFERENCE.pdf` (44 pp.) is the single reference for the two-point
parked lock-in: how each acquisition method works, where every microsecond of the
acquisition time goes, what sensitivity each method delivers, which defects are
open, and what has been changed in the code.

It merges and supersedes the two earlier markdown analyses, which remain in the
repository as the record of what was believed when:

- `../2026-08-06_twopoint_timing/TIMING_AND_NOISE_ANALYSIS.md`
- `../2026-08-14_twopoint_methods/TWOPOINT_METHODS_AND_TIMING.md`

Chapter 11 of the PDF lists every point on which this document contradicts them.

## Rebuild

```
latexmk -pdf TWOPOINT_MASTER_REFERENCE.tex
```

Needs TeX Live (tested on 2025) with `booktabs`, `siunitx`, `tcolorbox`,
`titlesec`, `underscore`, `enumitem`.

## Regenerate the numbers first

Every measured value in the document comes from:

```
python ../../scripts/analyze_twopoint_0814.py
```

which writes `../2026-08-14_twopoint_methods/tables/*.csv` and `figures/*.png`.
Copy any updated figures into `figures/` here before rebuilding.

## Source layout

| File | Chapters |
|---|---|
| `TWOPOINT_MASTER_REFERENCE.tex` | title page, contents, includes |
| `preamble.tex` | document class, packages, macros, callout boxes |
| `part1_orientation.tex` | 1 how to read this · 2 the estimator |
| `part2_common.tex` | 3 what all three methods share |
| `part3_averaged.tex` | 4 Method A — averaged |
| `part3_burst.tex` | 5 Method B — burst |
| `part3_stream.tex` | 6 Method C — streaming |
| `part4_timing.tex` | 7 timing in full |
| `part5_sensitivity.tex` | 8 sensitivity and noise |
| `part6_defects.tex` | 9 open defects |
| `part8_multipoint_stream.tex` | 10 streaming the multipoint lock-in |
| `part7_worklog.tex` | 11 work log · 12 to do · 13 corrections · 14 reproducing |
