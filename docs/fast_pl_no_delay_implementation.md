# Fast PL Acquisition — Implementation Guide

**Purpose:** This document explains why the original live PL had ~0.15 s/point delay, how we fixed it, and every design decision made along the way. Use this to debug or extend `run_fast_pl_no_display`.

---

## File Locations

| File | Role |
|---|---|
| `qick-dawg/src/qickdawg/nvpulsing/nv_live_helpers.py` | All custom helpers: `PLIntensityDual`, `run_fast_pl_no_display`, `run_live_odmr` |
| `qick-dawg/src/qickdawg/nvpulsing/__init__.py` | Exports: `from .nv_live_helpers import PLIntensityDual, run_fast_pl_no_display, run_live_odmr` |
| `Initial Test/01_basic_nv_testing.ipynb` cell `11de2946` | Import: `from qickdawg.nvpulsing.nv_live_helpers import run_fast_pl_no_display, run_live_odmr` |
| `Initial Test/01_basic_nv_testing.ipynb` cell `b7a56e27` | Calls `run_fast_pl_no_display(...)` |

---

## 1. What Was the Original Problem?

The original slow live PL loop (still in notebook cell `edf871f4`) acquires one point per channel like this:

```python
val_c = prog_c.acquire()   # PLIntensity.acquire() on ADC_C (ch=1)
val_d = prog_d.acquire()   # PLIntensity.acquire() on ADC_D (ch=0)
# then: fig.canvas.draw_idle() + handle.update(fig)
```

**Measured gap between consecutive timestamps:** ~0.15 s per displayed point.

### Where does 0.15 s come from?

The hardware time per point with `readout_integration_treg = 65535` (2^16 − 1) and `relax_delay_tns = 500` is:

```
integration_time = 65535 ticks × (1 / 614.4 MHz) ≈ 106.7 µs
relax_delay      = 500 ns ≈ 0.5 µs
─────────────────────────────────────────────────
hardware time per rep ≈ 107.2 µs
```

With `reps = 200` (default PLIntensity), that is `200 × 107.2 µs ≈ 21.4 ms` of FPGA time.

The remaining ~130 ms comes entirely from **matplotlib**:

- `fig.canvas.draw_idle()` — redraws the figure canvas
- `handle.update(fig)` — sends the rendered image to the Jupyter frontend

These two calls together take ~130–160 ms per loop iteration regardless of how fast the FPGA is. This is a fixed Python/Jupyter display overhead that cannot be reduced by changing hardware settings.

**Summary:** The FPGA hardware is 1000× faster than Python was delivering. Removing the display collapses the delay to near-zero.

---

## 2. Why Is ODMR Fast?

`LockinODMR.acquire()` sweeps over e.g. 301 frequency points and returns a full spectrum in one Python call. It is fast *per frequency point* for the same reason — the FPGA firmware loops over all reps and all sweep points internally, and Python only waits once for the entire batch. There is no `draw_idle()` inside the acquisition loop.

The `run_fast_pl_no_display` function replicates this pattern exactly for PL: the FPGA runs `batch_size` reps continuously in firmware with no Python round-trips between individual points.

---

## 3. PLIntensityDual — Design

### Why a new class instead of PLIntensity twice?

`PLIntensity` triggers one ADC channel per call. Running it twice (once for ADC_C, once for ADC_D) means two sequential FPGA batches — C and D are offset in time by one full batch (~76.5 ms). They are not simultaneous.

`PLIntensityDual` issues a single `trigger(adcs=[0, 1], ...)` which fires both ADC channels at the same clock tick. C and D in every row of the output DataFrame are truly simultaneous.

### Comparison to LockinODMR body

`LockinODMR.body()` does **two** readout windows per rep:
1. MW on + laser on → signal window → triggers ADC
2. MW off + laser on → reference window → triggers ADC

`PLIntensityDual.body()` does **one** readout window per rep:
1. Laser always on (not gated by PMOD) → triggers **both** ADC channels at once

The laser is not connected to PMOD in this setup — it is always on. This is why `PLIntensityDual` has no `pins=[self.cfg.laser_gate_pmod]` in its trigger call, while `LockinODMR` does.

### Class definition

```python
class PLIntensityDual(NVAveragerProgram):
    required_cfg = ["readout_integration_treg", "relax_delay_treg", "reps"]

    def initialize(self):
        self.check_cfg()
        for ch in (0, 1):
            self.declare_readout(
                ch=ch, freq=0,
                length=self.cfg.readout_integration_treg,
                sel="input",
            )
        self.cfg.adcs = [0, 1]
        self.synci(200)

    def body(self):
        self.trigger(
            adcs=self.cfg.adcs,
            width=self.cfg.readout_integration_treg,
            t=0,
        )
        self.wait_all()
        self.sync_all(self.cfg.relax_delay_treg)
```

**Key points:**
- `declare_readout(ch=0, ...)` → allocates `d_buf[0]` for ADC_D
- `declare_readout(ch=1, ...)` → allocates `d_buf[1]` for ADC_C
- No `pins=` argument → laser is not gated, always on
- `self.cfg.adcs = [0, 1]` must be set in `initialize()` before `body()` uses it

---

## 4. Reading Back Data — The d_buf Bug

### What NVAveragerProgram.acquire() returns

When you call the parent `NVAveragerProgram.acquire(program, progress=False)`, the firmware runs all `reps` and stores raw ADC data in `program.d_buf`.

`d_buf` is a list indexed by declared channel order:
- `d_buf[0]` → first declared channel = ch=0 = ADC_D
- `d_buf[1]` → second declared channel = ch=1 = ADC_C

Each entry has shape `(reps, 1, 2)` where the last dimension is `[I, Q]` (in-phase and quadrature).

### The bug that was present

Naive extraction:

```python
values_d = np.asarray(program.d_buf[0], dtype=float).reshape(-1)
```

This flattens the entire `(reps, 1, 2)` array — giving `reps × 1 × 2 = 2×reps` values, alternating I and Q. The result is wrong: twice as many points as expected, with every other value being the Q component (meaningless for DC photoluminescence).

### The correct extraction

```python
values_d = np.asarray(program.d_buf[0][..., 0], dtype=float).reshape(-1) / scale
values_c = np.asarray(program.d_buf[1][..., 0], dtype=float).reshape(-1) / scale
```

`[..., 0]` selects only the I component from the last dimension, giving shape `(reps, 1)`, then `.reshape(-1)` flattens to `(reps,)`.

### Why divide by scale?

`scale = float(integration_treg)` = 65535 by default.

The raw ADC sum returned by the firmware is accumulated over `readout_integration_treg` clock ticks. Dividing by the integration time normalises to ADC units per tick — the same normalisation that `PLIntensity.acquire()` and `LockinODMR.analyze_data()` use:

```python
# PLIntensity.acquire():
val = raw_sum / self.cfg.readout_integration_treg

# LockinODMR.analyze_data():
data = data / self.cfg.readout_integration_treg
```

---

## 5. run_fast_pl_no_display — How It Works

```python
def run_fast_pl_no_display(default_config, *, duration_sec=None,
                            batch_size=500, integration_treg=2**16-1,
                            csv_filename=None) -> pd.DataFrame:
```

### Flow

1. `_pl_program_dual(default_config, ...)` — copies `default_config`, overrides `readout_integration_treg` and `reps=batch_size`, builds a `PLIntensityDual` instance.
2. Outer `while True` loop:
   - Records `t_start = perf_counter() - t0`
   - Calls `_pl_batch_dual(prog, ...)` — runs `batch_size` reps on FPGA, returns `(values_c, values_d)` each of length `batch_size`
   - Records `t_end = perf_counter() - t0`
   - `_batch_times(t_start, t_end, batch_size)` — linearly interpolates timestamps across the batch (since the FPGA runs all reps with no Python visibility into individual rep timing)
   - Extends `all_t`, `all_c`, `all_d`
   - Breaks if `t_end >= duration_sec`
3. Builds DataFrame with columns `time_s`, `PL_C`, `PL_D`
4. Optionally saves to CSV

### Why batch_size=500?

Each Python call to `_pl_batch_dual` has a fixed overhead (ZMQ round-trip to the RFSoC board, result serialisation). Making `batch_size` large amortises this overhead over many points.

With `batch_size=500` and `integration_treg=65535`:
```
hardware time per batch = 500 × 107.2 µs ≈ 53.6 ms
Python overhead per batch ≈ a few ms (negligible vs 53.6 ms)
effective rate ≈ 500 / 0.054 ≈ 9,300 pts/s
```

For a 2-minute run: `9,300 × 120 ≈ 1,116,000 points`.

Compare to the original slow loop: `1 / 0.15 ≈ 6.7 pts/s` → `6.7 × 120 ≈ 800 points` in 2 minutes.

---

## 6. Timestamp Interpolation

The FPGA runs all `batch_size` reps internally — Python cannot observe when each individual rep completes. We only know `t_start` (before the batch call) and `t_end` (after it returns).

`_batch_times` assigns uniformly spaced timestamps across the batch:

```python
def _batch_times(t_start, t_stop, count):
    step = (t_stop - t_start) / float(count)
    return t_start + step * (np.arange(count, dtype=float) + 1.0)
```

The `+ 1.0` offset means the first timestamp is at the end of rep 1 (not the start), and the last is at `t_stop`. This is a best-effort interpolation; consecutive `time_s` diffs within a batch will be equal by construction (~107 µs each at default settings).

---

## 7. Configuration Requirements

`PLIntensityDual.required_cfg = ["readout_integration_treg", "relax_delay_treg", "reps"]`

These are all set automatically by `_pl_program_dual` — you do not need to set them on `default_config`. The function copies `default_config` and overrides only these three fields.

`default_config` must already have all fields required by `NVConfiguration` and `NVAveragerProgram`. In the notebook, this is set up in cell `default-config`.

---

## 8. Common Errors and Fixes

### `AttributeError: 'NVConfiguration' object has no attribute 'adcs'`

`PLIntensityDual.initialize()` sets `self.cfg.adcs = [0, 1]` at the end of initialize. If body() runs before initialize() completes, this can fail. This should not happen under normal qickdawg usage — if it does, check the firmware version.

### `d_buf[1]` IndexError or empty

Both channels are only populated if `declare_readout` was called for both `ch=0` and `ch=1` in `initialize()`. If the firmware only exposes one ADC channel, `d_buf` will have length 1. Check that the RFSoC4x2 firmware loaded is the dual-channel variant.

### Output has 2× more points than expected

You hit the I/Q flattening bug — make sure extraction uses `[..., 0]`:

```python
values_d = np.asarray(program.d_buf[0][..., 0], dtype=float).reshape(-1) / scale
```

### PL values are ~65535× too large

Missing the `/scale` division. `scale = float(integration_treg)` must be applied.

### Import error: `cannot import name 'run_fast_pl_no_display'`

Check that `nvpulsing/__init__.py` contains:

```python
from .nv_live_helpers import PLIntensityDual, run_fast_pl_no_display, run_live_odmr
```

And that `nv_live_helpers.py` exists at `qick-dawg/src/qickdawg/nvpulsing/nv_live_helpers.py`.

### `ModuleNotFoundError: No module named 'qickdawg'`

The qickdawg package must be installed (editable install). From the repo root:

```bash
pip install -e qick-dawg/
```

Or check that the virtual environment active in the notebook kernel is the one with qickdawg installed (`qickdawg_venv` or `qickdawg_py310`).

---

## 9. DataFrame Output

```
df_fast_pl columns:
  time_s   float64   seconds since acquisition start (linearly interpolated within each batch)
  PL_C     float64   ADC_C (ch=1) intensity in normalised ADC units (raw I sum / integration_treg)
  PL_D     float64   ADC_D (ch=0) intensity in normalised ADC units
```

Each row corresponds to one FPGA rep. Both PL_C and PL_D in the same row were acquired at the same clock tick — they are truly simultaneous.

---

## 10. Relationship to Other qickdawg Classes

```
NVAveragerProgram (base)
│
├── PLIntensity          — single channel, laser gated via PMOD, acquire() returns one float
├── LockinODMR           — single channel, MW on/off reference, acquire() returns spectrum
│     body: [MW on → trigger ADC] [MW off → trigger ADC]   (2 readouts/rep)
│
└── PLIntensityDual      — dual channel simultaneously, laser always on, d_buf read directly
      body: [trigger both ADC ch=0 and ch=1]               (1 readout/rep, 2 channels)
```

`PLIntensityDual` is structurally the same as the **reference window** in `LockinODMR` (laser on, no MW pulse), run on both ADC channels at once instead of one.
