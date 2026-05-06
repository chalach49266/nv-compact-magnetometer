# NV Compact Magnetometer

NV-center compact magnetometer running on RFSoC4x2 + qick-dawg. Includes lock-in ODMR sweeps, two-point lock-in modulation, single-program N-frequency parked acquisition (`MultipointLockinODMR`), and B-field reconstruction via the `nv_toolkit` parked-frequency inversion.

## Repository layout

```
nv_magnetometer_project/
├── README.md                    this file
├── pyproject.toml               package metadata + abstract dependencies
├── requirements.txt             pinned versions from working venv (qickdawg_venv, Python 3.13.5)
├── .gitignore
├── notebooks/                   Jupyter notebooks
│   ├── 01_basic_nv_testing.ipynb              main workflow notebook
│   ├── 01b_noise_cancelled_nv_testing.ipynb
│   ├── 01c_dual_channel_fast_pl_validation.ipynb
│   ├── 01e_dual_channel_fast_pl_outlier_aware_workflow.ipynb
│   ├── 02_microwave_fixed_frequency.ipynb
│   ├── 02_rabi_vector_magnetometry.ipynb
│   ├── 02b_noise_cancelled_rabi_vector_magnetometry.ipynb
│   ├── Plot and sensitivity_csv.ipynb
│   └── Modules/
│       ├── Lockin_module.ipynb                plan → acquire (one FPGA upload) → reconstruct
│       ├── ODMR_module.ipynb
│       └── PL_readout_module.ipynb
├── src/                         Python modules (importable)
│   ├── multipoint_lockin_program.py    MultipointLockinODMR (one-program N-frequency lock-in)
│   ├── twopoint_lockin.py              run_twopoint_lockin / single_shot_twopoint
│   ├── odmr_sensitivity.py             ODMR Lorentzian fits, sensitivity estimates
│   ├── lockin_extensions.py            sinusoidal FM, feedback, slope analysis
│   └── nv_magnetometry_analysis.py
├── data/                        CSV data files (ODMR sweeps, lock-in time series, parked plans)
├── docs/                        markdown notes, change logs, troubleshooting guides
├── scripts/                     standalone scripts
└── qick-dawg-patch/             our modification to upstream qick-dawg
    ├── lockinodmr.py                   full modified file
    └── lockinodmr_offres_reference.patch  unified diff against upstream
```

## Setup — replicating the Python environment

The original working environment is `qickdawg_venv` (Python 3.13.5, 120 packages). To replicate it on a fresh machine:

### Option A — exact replica from `requirements.txt` (recommended)

```bash
# 1. Make sure Python 3.13 is installed (e.g., via pyenv / brew install python@3.13)
python3.13 -m venv qickdawg_venv
source qickdawg_venv/bin/activate
pip install --upgrade pip

# 2. Install pinned versions
pip install -r requirements.txt
```

### Option B — abstract install from `pyproject.toml`

```bash
python3.13 -m venv qickdawg_venv
source qickdawg_venv/bin/activate
pip install --upgrade pip
pip install -e .          # installs this project + minimum deps from pyproject.toml
```

Option A reproduces the exact package versions that work today. Option B picks the latest compatible versions — use it if you're upgrading.

### Step 2 — install qick and qick-dawg from the upstream repos

These are vendor libraries (Xilinx Quantum Instruments Control Kit + qick-dawg). They're not pip-published, so install from source:

```bash
# clone the upstream repos somewhere (e.g., next to this project)
cd ..
git clone https://github.com/openquantumhardware/qick.git
git clone https://github.com/sandialabs/qick-dawg.git

# install both into the active venv
pip install -e ./qick
pip install -e ./qick-dawg
```

### Step 3 — apply the qick-dawg patch

This project depends on a small modification to `qick-dawg/src/qickdawg/nvpulsing/lockinodmr.py` that adds an `odmr_reference_offres_mhz` config flag. The reference shot fires the MW at an explicit off-resonance frequency (default 2650 MHz) instead of relying on gain control, which fixes a persistent leakage issue.

Apply the patch one of two ways:

```bash
# (a) apply the unified diff
cd ../qick-dawg
patch -p0 < ../nv_magnetometer_project/qick-dawg-patch/lockinodmr_offres_reference.patch

# (b) or copy the full modified file
cp ../nv_magnetometer_project/qick-dawg-patch/lockinodmr.py src/qickdawg/nvpulsing/lockinodmr.py
```

### Step 4 — register the venv with Jupyter

```bash
python -m ipykernel install --user --name nv-magnetometer --display-name "NV Magnetometer (qickdawg_venv)"
```

When opening the notebooks, select the **"NV Magnetometer (qickdawg_venv)"** kernel.

## Hardware setup

- **RFSoC4x2** board running the qick-dawg firmware — bitstream lives in `<qick-dawg>/firmware/photon_counting/qick_4x2.bit` (and friends).
- ADC channels: `ADC_D=0` (default), `ADC_C=1`.
- MW channel `1`, NQZ `1`, default gain `5000`.
- Set `RFSOC_IP` in the first cell of any notebook to the board's IP (e.g., `192.168.0.103`).

## Typical workflow

1. **ODMR sweep** — `notebooks/Modules/ODMR_module.ipynb` cell that runs `qd.LockinODMR` over 2600–3100 MHz writes `odmr_sweep_<timestamp>.csv` into `data/` and sets `LAST_ODMR_CSV`.
2. **Plan parked frequencies** — `notebooks/Modules/Lockin_module.ipynb` cell `lockin_plan` calls `nv_toolkit.tui._suggest_parked_frequencies` on `LAST_ODMR_CSV` → 16 parked frequencies + bias estimate.
3. **Acquire** — cell `lockin_acquire` builds **one** `MultipointLockinODMR` program for all 16 frequencies (one FPGA upload, ~same overhead as a single sweep) and writes a wide-format CSV.
4. **Reconstruct B-field** — cell `lockin_reconstruct` calls `nv_toolkit.tui._compute_live_snapshot` and writes vector + projection CSVs.

Steps 2 and 4 use the same algorithms exposed by the `mag-operator plan` / `mag-operator reconstruct` CLI commands in `nv_toolkit`. The notebook calls them directly so the workflow is end-to-end inside one Jupyter kernel.

## Why the single-program multipoint matters

Naively, measuring 16 non-uniform parked frequencies means building 16 separate `LockinODMR` programs and uploading each to the FPGA — ~300 ms per upload, so ~5 s of wasted overhead per shot. `MultipointLockinODMR` works around that by using `mw_frequency_register.set_to(f_mhz)` inside `body()`, so one program holds the ASM for all 16 frequencies and runs them at hardware speed inside the FPGA. Effective runtime is comparable to a single ODMR sweep with the same total readout count.

## Provenance

This is a copy of the Python source tree from `<phd>/NV Compact Magnetometer/Initial Test/` (notebooks, modules) plus the modified `qick-dawg` vendor file. The originals remain in place; this folder is the canonical version for git tracking.
