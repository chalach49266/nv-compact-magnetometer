# NV Toolkit

`nv_toolkit/` is the reusable layer that sits between the older `phase0/`
prototype scripts and the newer `phase1/` experiment handoffs.

Use it when you want one callable path for:

- full-spectrum CSV loading
- parked-frequency CSV loading
- base-field calibration resolution
- local-field fitting
- parked-frequency Jacobian inversion back to `dBx, dBy, dBz`

## Command-line entry point

Use `./mag` for the peak-first reconstruction workflow on a single ODMR sweep,
including raw April 29 style exports.

Examples:

```bash
./mag phase1/data/raw_scan_exports_20260429_lockin/8_peaks_8_stacks.csv

./mag \
  phase1/data/raw_scan_exports_20260429_lockin/8_peaks_8_stacks.csv \
  --axis-model stage-x \
  --bias-seed 0,0,-4.2 \
  --output-json phase1/results/0429_stage_x_reconstruct.json

./mag \
  phase1/data/raw_scan_exports_20260429_lockin/full_scan_8_peaks_8_stacks_toolkit.csv \
  --input-format toolkit \
  --plot phase1/results/0429_qdm_fit.png
```

Key options:

- `--axis-model {canonical,111,qdm,stage-x}`
- `--nv-axes-inline "1,0,0; -1/3,2*sqrt(2)/3,0; ..."` for an explicit 4x3 axis matrix
- `--bias-seed Bx,By,Bz`
- `--skip-spectrum-fit` to stop after peak-only inversion
- `--output-json path/to/summary.json`
- `--plot path/to/fit.png`

## Terminal workflow viewer

Use `mag-tui` for a terminal UI that walks the Phase 1 sensing protocol using
committed result artifacts as demos.

Examples:

```bash
mag-tui
mag-tui --demo april29-lockin
mag-tui --list-demos
```

The default demo is the committed April 17 real parked-frequency validation.
Use left/right or `d` to switch demos, up/down to move through workflow stages,
and `q` to quit.

## Live operator console

Use `mag-tui --operator` to run the full operator flow:

1. provide a full ODMR spectrum CSV
2. let the tool propose 16 parked frequencies from the detected transitions
3. point it at a live parked-data CSV
4. watch the local field and rolling vector trace update every second

Examples:

```bash
mag-tui --operator
mag-tui --operator --full-scan path/to/full_scan.csv --parked-data path/to/live_parked.csv
mag-tui --operator --full-scan path/to/full_scan.csv --parked-data path/to/live_parked.csv --parked-format long
```

Accepted live parked-data formats:

- wide: `time_s, peak_01, peak_02, ... peak_16`
- long: `time_s, frequency_mhz, intensity`
- toolkit: `save_id, timestamp_epoch_s, block_index, point_in_block, point_index, frequency_mhz, normalized_intensity`

The operator console writes three helper files next to the parked-data target:

- `parked_plan_16freq.csv`
- `parked_template_wide.csv`
- `parked_template_long.csv`

## Browser Operator

Use `mag-web` for a browser-based operator window.

```bash
mag-web --no-open
```

For Tailscale or other remote access, bind explicitly:

```bash
mag-web --host 0.0.0.0 --no-open
```

For the current live-like April 29 demo, use the manual in
[phase1/operator_test_fixture/README.md](../phase1/operator_test_fixture/README.md).
It walks through the browser operator, 16 parked-frequency plan generation, and
the noisy synthetic stream used for real-time reconstruction.

## FPGA Operator CLI

Use `mag-operator` when acquisition hardware owns data collection and Python
should only write/read CSV contracts. The command is wired in `pyproject.toml`
as:

```text
mag-operator = "nv_toolkit.operator_cli:main"
```

The implementation file is [operator_cli.py](operator_cli.py). Keep that file as
a compact orchestration layer; reusable algorithms should live in the lower-level
modules listed below.

Step 1: generate the 16 parked microwave frequencies from a full ODMR spectrum.

```bash
mag-operator plan \
  --full-scan path/to/full_odmr.csv \
  --output path/to/parked_plan_16freq.csv
```

Step 2: reconstruct B from the collected parked-intensity CSV.

```bash
mag-operator reconstruct \
  --full-scan path/to/full_odmr.csv \
  --plan path/to/parked_plan_16freq.csv \
  --parked-data path/to/collected_parked_data.csv \
  --output path/to/operator_vector_rows.csv \
  --projections-output path/to/operator_projection_rows.csv
```

`--parked-format` defaults to `auto`; pass `wide`, `long`, or `toolkit` only if
auto-detection fails.

Plan CSV contract:

| Column | Meaning |
|---|---|
| `block_index` | ODMR transition block, one minus/plus pair per block |
| `point_in_block` | `0` for first flank point, `1` for second flank point |
| `point_index` | acquisition order; wide collected data follows this order |
| `transition_index` | transition index from detected ODMR centers |
| `transition_center_mhz` | fitted/detected transition center |
| `linewidth_mhz` | local linewidth estimate used for parked-point choice |
| `frequency_mhz` | microwave frequency for FPGA/acquisition to park at |
| `slope_per_mhz` | local ODMR slope at that parked point |

Collected-data formats accepted by `reconstruct`:

- Wide: one time column named `time_s`, `time`, `timestamp`, or `timestamp_epoch_s`, followed by 16 intensity columns in `point_index` order.
- Long: one row per parked frequency with time, frequency, and intensity columns.
- Toolkit: expanded rows with `save_id`, `timestamp_epoch_s`, `block_index`, `point_in_block`, `frequency_mhz`, and `normalized_intensity`.

Output CSVs:

- `operator_vector_rows.csv`: one reconstructed vector per timestamp, including `delta_Bx_uT`, `delta_By_uT`, `delta_Bz_uT`, residual, rank, condition number, and status.
- `operator_projection_rows.csv`: optional per-block projection/debug rows before vector inversion.

File map for the reusable operator path:

| File | Role |
|---|---|
| `nv_toolkit/operator_cli.py` | Compact command-line control surface for FPGA plan/reconstruct |
| `nv_toolkit/tui.py` | Current shared operator APIs: ODMR loading, plan generation, parked-data loading, live snapshot reconstruction |
| `nv_toolkit/intensity_tracking.py` | Parked-frequency calibration, per-block projections, vector inversion |
| `nv_toolkit/operator_web.py` | Browser wrapper around the same concepts; not required for FPGA CLI |
| `tests/test_nv_toolkit.py` | Regression tests for plan/reconstruct behavior and CSV format conversion |

## Base-field CSV formats

Supported base-field CSV formats include:

- direct field vector
  - columns such as `Bx_uT`, `By_uT`, `Bz_uT`
- current plus slope calibration
  - current values together with diagonal or full-matrix `uT/A` calibration

Examples live in:

- [phase1/base_field_examples/direct_vector_uT.csv](../phase1/base_field_examples/direct_vector_uT.csv)
- [phase1/base_field_examples/current_slope_uT_per_A.csv](../phase1/base_field_examples/current_slope_uT_per_A.csv)

## Python API

Install as an editable package from the repo root:

```bash
pip install -e .
```

All public symbols are importable directly from `nv_toolkit`:

```python
import nv_toolkit as nv
```

### Module overview

| Module | What it does |
|---|---|
| `model` | NV physics: resonance frequencies, gradients, ODMR spectrum |
| `fitting` | Fit Bx/By/Bz from a batch of ODMR spectra |
| `static_fit` | Fit a single static ODMR spectrum |
| `peaks` | Find dip positions, smooth spectra, collapse hyperfine doublets |
| `two_point` | Two-point (parked-frequency) calibration per transition |
| `intensity_tracking` | Multi-block parked-frequency calibration and vector inversion |
| `parked` | Low-level per-row pair metrics (difference, normalized difference) |
| `io` | Load full-scan CSVs and parked-frequency CSVs |
| `base_field` | Load and resolve base-field vectors from CSV or inline strings |

### Full-spectrum fitting

```python
import numpy as np
import nv_toolkit as nv

batch = nv.load_spectrum_csv("spectrum.csv")
base_field = nv.resolve_base_field("15.0,980.0,500.0", unit="uT")

for i, spectrum in enumerate(batch.spectra):
    result = nv.fit_local_field(
        batch.freqs_mhz,
        spectrum,
        B_base_mT=base_field,
    )
    print(result.B_local_mT)
```

### Parked-frequency (two-point) vector tracking

This is the fast-acquisition mode. You park two frequencies on the flanks of
each ODMR transition and track intensity changes instead of sweeping the full
spectrum. One calibration block per NV axis transition. Three or more
independent blocks are needed to invert to a 3D vector.

```python
import numpy as np
import nv_toolkit as nv

# One-call entry point — returns projection rows, vector rows, and inversion records
result = nv.fit_parked_intensity_csv(
    "full_scan_reference.csv",   # single reference spectrum for calibration
    "parked_intensity.csv",      # time-series of parked-frequency intensities
    B_bias_mT=np.array([0.015, 0.980, 0.500]),
    nv_axes_preset="qdm",
)

for row in result["vector_rows"]:
    print(row["save_id"], row["delta_Bx_uT"], row["delta_By_uT"], row["delta_Bz_uT"])
```

#### Parked-frequency CSV format

The parked-frequency CSV must have these columns:

| Column | Description |
|---|---|
| `save_id` | unique identifier per acquisition time-step |
| `timestamp_epoch_s` | Unix timestamp |
| `frequency_mhz` | parked frequency |
| `normalized_intensity` | measured intensity (normalized to off-resonance = 1) |
| `block_index` | which transition block (0, 1, 2, …) |
| `point_in_block` | 0 = lower flank, 1 = upper flank |

Each `block_index` must contain exactly two rows per `save_id` (the minus and
plus flank frequencies). Three or more blocks give a full 3D vector inversion;
fewer give only projected field changes.

#### Building calibrations manually

```python
import numpy as np
import nv_toolkit as nv

ref_batch = nv.load_spectrum_csv("full_scan_reference.csv")
parked_series = nv.load_parked_frequency_csv("parked_intensity.csv")

B_bias = np.array([0.015, 0.980, 0.500])  # mT

calibrations = nv.build_blockwise_calibrations(
    ref_batch.freqs_mhz,
    ref_batch.spectra[0],
    parked_series.freqs_mhz,
    parked_series.block_indices,
    parked_series.point_in_block,
    B_bias_mT=B_bias,
)

# calibrations is a dict: block_index -> TwoPointCalibration
for block, cal in calibrations.items():
    print(f"block {block}: f0={cal.resonance_frequency_mhz:.2f} MHz, "
          f"axis={cal.measurement_axis}")
```

### Static single-spectrum fit

```python
import numpy as np
import nv_toolkit as nv

freqs, spectrum = ...  # 1D arrays
result = nv.fit_static_spectrum(freqs, spectrum)
print(result)          # fitted Bx, By, Bz and spectrum parameters
```

### NV model utilities

```python
import numpy as np
import nv_toolkit as nv

B_mT = np.array([0.1, 0.5, 0.3])
freqs_mhz = nv.odmr_resonances(B_mT)           # shape (4, 2): four axes, lower/upper
spectrum = nv.odmr_spectrum(
    np.linspace(2800, 2920, 500),
    B_mT,
    linewidths=3.0,
    contrasts=0.05,
    hyperfine_splitting_mhz=2.16,
)
```

## Phase split

- `phase0/`
  - synthetic data, prototype scripts, earlier result archives
- `phase1/`
  - real data handoffs, Phase 1-style exported CSVs, validation outputs
- `nv_toolkit/`
  - reusable code shared by both

That split keeps post-processing and experiment-facing workflows reusable without
rewriting the older prototype code.
