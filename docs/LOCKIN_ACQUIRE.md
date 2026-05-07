# Lock-in acquire — single-program multipoint implementation

This doc explains how `Modules/Lockin_module.ipynb` and `Modules/multipoint_lockin_program.py` measure 16 parked frequencies in one FPGA program upload, and why that is roughly 5 s faster per shot than the naive per-frequency loop.

## The pipeline

```text
ODMR sweep (LAST_ODMR_CSV)
        │
        ▼
┌──────────────────────────┐
│ lockin_plan (cell 10)    │   nv_toolkit.tui._suggest_parked_frequencies
│   16 parked freqs        │   → writes parked_plan_<stem>.csv to data/lockin_multipoint/
│   bias-field estimate    │   → defines MULTIPOINT_FREQS_MHZ (list of 16 floats)
└──────────────┬───────────┘
               ▼
┌──────────────────────────┐
│ lockin_acquire (cell 12) │   MultipointLockinODMR(cfg).acquire()
│   ONE FPGA program       │   → wide CSV peak_01…peak_16 + freqs + refs
│   reps × 16 freqs × 2 RO │   → data/lockin_multipoint/multipoint_lockin_collected.csv
└──────────────┬───────────┘
               ▼
┌──────────────────────────┐
│ lockin_reconstruct (14)  │   nv_toolkit.tui._compute_live_snapshot
│   B-field vectors        │   → reconstructed_vector_rows.csv
│   per-block projections  │   → reconstructed_projection_rows.csv
└──────────────────────────┘
```

`lockin_plan` and `lockin_reconstruct` reuse the algorithms exposed by the
`mag-operator plan` / `mag-operator reconstruct` CLI commands in `nv_toolkit`.
The notebook calls them directly so the whole workflow runs in one Jupyter
kernel without subprocess shelling.

## The bottleneck we are working around

Naive per-frequency approach builds a fresh `LockinODMR` for every parked
point:

```python
# DON'T DO THIS — 16 separate FPGA program uploads
for f in MULTIPOINT_FREQS_MHZ:
    cfg.mw_start_fMHz = f
    cfg.mw_end_fMHz = f
    prog = qd.LockinODMR(cfg)        # compiles + uploads ASM each iteration
    data = prog.acquire()            # ~300 ms of upload overhead per call
```

Each `qd.LockinODMR(cfg)` triggers an FPGA program compile + upload, ~300 ms
of Python ↔ board round-trip. With 16 frequencies that is ~5 s of pure
overhead per shot, on top of the actual pulse-sequence work. Worse, it is
inconsistent — board upload latency varies, so the same shot can take 4–8 s.

## How `MultipointLockinODMR` avoids that

The trick is to put **all 16 frequencies into one program's `body()` method**
and switch the MW frequency register at runtime instead of recompiling.

The relevant code in [`multipoint_lockin_program.py`](multipoint_lockin_program.py):

```python
def body(self):
    """Generate ASM that loops through all N frequencies in one body call.

    Each frequency contributes one signal+reference readout pair.
    Total readouts per body = 2 * N.
    """
    ref_offres_mhz = float(self.cfg.odmr_reference_offres_mhz)
    freqs_mhz = list(self.cfg.multipoint_freqs_mhz)

    for freq_mhz in freqs_mhz:
        # 1. Park MW at this signal frequency
        self.mw_frequency_register.set_to(float(freq_mhz))
        self.pulse(ch=self.cfg.mw_channel, t=0)
        self.trigger_no_off(adcs=self.cfg.adcs, ...)            # signal ADC read
        self.trigger_no_off(pins=[laser_gate_pmod], ...)        # readout-relax pad

        # 2. Park MW off-resonance, take reference shot
        self.mw_frequency_register.set_to(ref_offres_mhz)
        self.pulse(ch=self.cfg.mw_channel, t=t_ref)
        self.trigger(adcs=self.cfg.adcs, ...)                   # reference ADC read

        self.wait_all()
        self.sync_all(self.cfg.relax_delay_treg)
```

Three things make this fast:

1. **The Python `for` loop runs once at program-compile time**, not at
   acquisition time. Each iteration emits ASM into the FPGA program. So 16
   frequencies become ~16 × 26 = ~430 ASM ops in one compiled program.
2. **`mw_frequency_register.set_to(f_mhz)`** is a single FPGA register-write
   instruction. It costs ~1 ASM cycle, vs ~300 ms for a full program rebuild.
3. **`acquire()` runs ONE Python ↔ FPGA round trip** for all reps. The FPGA
   does the inner sweep (over 16 frequencies) and the outer sweep (over reps)
   internally at hardware speed.

The hardware loop structure becomes:

```text
outer: reps                        (e.g. 1000)
  inner: body call                 (one call covers all 16 freqs)
    for each of 16 freqs:          (unrolled in ASM)
      set MW = f_signal            (signal pulse)
      pulse + ADC read             → signal[i]
      set MW = 2650 MHz            (off-resonance reference)
      pulse + ADC read             → reference[i]
      sync
```

After `acquire()` returns, the data buffer has shape `(reps, 1, 2*N)` with
the readouts axis in the order `[sig0, ref0, sig1, ref1, …, sigN-1, refN-1]`.
`analyze_results()` slices out even/odd indices and averages over the rep
axis to produce `.signal`, `.reference`, `.contrast`, and
`.contrast_percent` arrays of length `N`.

## Wiring in the notebook

`Lockin_module.ipynb` cell `lockin_acquire` configures and runs the program:

```python
from multipoint_lockin_program import MultipointLockinODMR

cfg = copy(default_config)
cfg.multipoint_freqs_mhz = list(MULTIPOINT_FREQS_MHZ)        # 16 floats from the plan
cfg.odmr_reference_offres_mhz = 2650.0                       # off-resonance reference
cfg.mw_start_fMHz = float(MULTIPOINT_FREQS_MHZ[0])           # placeholder for QickSweep
cfg.mw_end_fMHz   = float(MULTIPOINT_FREQS_MHZ[0])
cfg.nsweep_points = 1
cfg.reps          = 1000

prog = MultipointLockinODMR(cfg)            # ONE compile + upload
for batch in range(N_BATCHES):
    d_batch = prog.acquire(progress=False)  # FPGA does all 16 freqs × reps internally
    # d_batch.signal     → length-16 array of MW-on PL averages
    # d_batch.reference  → length-16 array of MW-off PL averages
    # d_batch.contrast   → signal − reference
```

The off-resonance reference at 2650 MHz fires the MW far from every NV
transition, so the reference shot has no NV excitation. This replaces the
older approach of trying to cut MW gain to zero, which leaked at low gain.

## Why off-resonance instead of MW-off

Older `LockinODMR` body code attempted to disable the MW for the reference
shot by setting gain near zero. In practice the MW chain still leaked enough
power to drive a partial dip in the reference trace, contaminating the
contrast. Switching to an off-resonance frequency (2650 MHz, well below the
~2.87 GHz zero-field splitting) keeps full MW power on the channel but
prevents NV excitation, giving a clean baseline.

This is implemented in `qickdawg/nvpulsing/lockinodmr.py` (the patched copy
vendored in this repo) via the `cfg.odmr_reference_offres_mhz` flag, and
`MultipointLockinODMR` uses the same flag — the reference shot inside its
unrolled body simply calls `self.mw_frequency_register.set_to(2650.0)`
before the reference pulse.

## Concrete timing budget

### What's deterministic (FPGA pulse work)

For the project defaults (`readout_integration_tus = 213`,
`relax_delay_tns = 500` ≈ 0.5 µs, 16 parked frequencies):

```text
cycle per freq pair  = 2 × 213 µs + 2 × 0.5 µs ≈ 427 µs
body() per rep       = 16 × 427 µs            ≈ 6.83 ms
```

So each rep does ~6.83 ms of FPGA pulse work, regardless of `reps` or batch
count. This number is exact — it's set by `readout_integration_tus`,
`relax_delay`, and the body's pulse pattern. Verify in the printed
`prog.total_time()` line right after the build.

### What scales with reps

| Setting                       | FPGA pulse work / batch | Update rate (FPGA-only)  |
| ----------------------------- | ----------------------- | ------------------------ |
| `reps = 1`                    | ~7 ms                   | ~143 Hz                  |
| `reps = 10` (current default) | ~68 ms                  | ~14.7 Hz                 |
| `reps = 100`                  | ~683 ms                 | ~1.5 Hz                  |
| `reps = 1000`                 | ~6.8 s                  | ~0.15 Hz                 |

These are floor times — what the FPGA actually spends generating pulses.
Real per-batch wall-time is *higher* by the Python ↔ board overhead below.

### What actually adds up at runtime (less deterministic)

| Source                                     | Typical cost             |
| ------------------------------------------ | ------------------------ |
| Pyro4 RPC round-trip (PYNQ ↔ host)         | 30–80 ms per `acquire()` |
| d_buf transfer (320 int64s for 16×2×10)    | ~5 ms                    |
| `analyze_results` numpy reshape + averaging | ~1–3 ms                 |
| Python list appends / DataFrame row build  | ~1 ms                    |
| `_handle.update(fig)` (if plot enabled)    | 5–30 ms (grows with N)   |

The Pyro4 round-trip dominates the non-FPGA time and is **not deterministic**
— network/board state can spike it to 200–500 ms occasionally. This is the
single largest source of jitter and the reason a "14 Hz" prediction often
lands at 5–10 Hz in practice.

### Realistic per-batch wall time at the current default (`reps = 10`)

```text
FPGA pulse work       ≈ 68 ms                (deterministic)
Pyro4 + transfer      ≈ 50–100 ms typical    (network-dependent)
Python overhead       ≈ 10–20 ms             (plot + appends)
                      ─────────────────────────
Total per batch       ≈ 130–190 ms typical   ≈ 5–8 Hz update rate
                      ≈ 200–500 ms occasionally on jitter
```

So the "7 Hz" floor in earlier docs is realistic; "14 Hz" is the unloaded
upper bound and you should not expect to see it consistently.

### One-time costs (paid once per cell run)

| Step                                          | Cost              |
| --------------------------------------------- | ----------------- |
| `MultipointLockinODMR(cfg)` build + upload    | ~300–500 ms       |
| Priming acquire (only in `lockin_live`)       | ~70 ms (`reps=10`) |

Total cold-start before the live loop begins: ~400–600 ms.

### Compared to the naive 17-program approach (the old 8-peak cell in `01_basic`)

The old cell built `qd.LockinODMR(cfg)` 17 times per batch (one per
frequency including the off-resonance baseline). Each build adds an
upload, and each acquire does a full `nsweep_points = 2` zero-span. With
`reps_per_point = 100`:

```text
old_per_batch ≈ 17 × (300 ms upload + 100 reps × 2 nsweep × 427 µs)
             ≈ 17 × (300 ms + 86 ms)
             ≈ ~6.6 s per batch (good case)
             ≈ 15-30 s per batch (bad case with network jitter)
```

vs the new multipoint cell at `reps=100`:

```text
new_per_batch ≈ 1 × (300 ms upload, ONCE) + 100 reps × 16 × 427 µs
             ≈ 300 ms (one-time) + 683 ms / batch
             ≈ ~700 ms per batch on average
```

For multi-batch / live mode the win is even bigger because the build
happens once total, not once per batch.

## Output schema

`data/lockin_multipoint/multipoint_lockin_collected.csv` is wide-format with
one row per batch:

| column                      | meaning                                             |
| --------------------------- | --------------------------------------------------- |
| `batch`                     | batch index (0-based)                              |
| `time_s`                    | seconds since the run started                      |
| `timestamp_epoch_s`         | wall-clock epoch                                   |
| `acq_seconds`               | how long this batch took inside `acquire()`        |
| `peak_<NN>` (NN = 01..16)   | MW-on PL, normalized by `readout_integration_treg` |
| `peak_<NN>_ref`             | MW-off (off-resonance) PL                          |
| `peak_<NN>_freq_mhz`        | the parked frequency for this column               |

The column order matches `point_index` in the plan CSV, which is the
contract `nv_toolkit._compute_live_snapshot` expects when given
`parked_format="wide"`.

## When the runtime grows

The single-program trick only helps when frequencies are non-uniform (so a
QickSweep cannot cover them with a linear ramp). If all 16 frequencies were
evenly spaced, a plain `LockinODMR` with `nsweep_points=16` would already do
this in one upload. The reason the project needs a custom class is that the
8 ODMR transitions are at irregular spacings, so the parked frequencies
(±FWHM/2 around each transition) are not on a uniform grid.

If you ever drop the off-resonance reference (set
`cfg.odmr_reference_offres_mhz = None`) the body will still work but the
reference column will be all zeros, since no reference pulse fires.

## From signal difference to new peak frequency

The acquire cell only gives raw PL at 16 parked frequencies. The notebook cell
`lockin_to_peak_frequency` (inserted between `lockin_acquire` and
`lockin_reconstruct`) converts those into a **per-transition frequency shift**
$\Delta f_k$ and a **new peak frequency** $f_{\text{new},k} = f_{0,k} + \Delta f_k$
using the standard slope-based two-point lock-in error signal.

### The math

For each ODMR transition $k$ (block index 1…8) at center $f_{0,k}$, with parked
frequencies $f_{-,k}, f_{+,k}$, baselines $b_{-,k}, b_{+,k}$ (sampled from the
reference ODMR), and slopes $m_{-,k}, m_{+,k}$ (from `np.gradient(spectrum, freqs)`):

```text
Δd_now    = S₊(t) − S₋(t)              # current intensity difference (this batch)
Δd_ref    = b₊ − b₋                    # reference intensity difference (calibration)
Δf_k      = (Δd_now − Δd_ref) / (m₋ − m₊)
f_new,k   = f₀,k + Δf_k
ΔB_proj,k = Δf_k / ‖df/dB‖_k
```

Because $m_-$ is negative (left flank of dip) and $m_+$ is positive, the
denominator $m_- - m_+ \approx -2|m|$, and the formula reduces to the textbook
$\Delta f \approx -\Delta d / (2|m|)$. Keeping the asymmetric form preserves
accuracy when the slopes are not perfectly equal in magnitude (which is common
when transitions overlap or sit on the wing of a neighbour).

### Inputs reused from earlier cells

The cell does not recompute anything from scratch. It pulls:

- `freqs_mhz`, `measured`, `transition_centers` — produced by `lockin_plan` via
  `nv_toolkit.tui._load_operator_full_scan` (the reference ODMR sweep).
- `parked_plan`, `BIAS_MT` — also from `lockin_plan` (one
  `ParkedFrequencyPlanEntry` per transition + the bias-field estimate).
- `df_multipoint`, `MULTIPOINT_DATA_CSV` — from `lockin_acquire` (the wide
  CSV with `peak_NN` / `peak_NN_ref`).

### Existing toolkit functions used

- `nv_toolkit.intensity_tracking.build_blockwise_calibrations(...)` — builds 8
  `TwoPointCalibration` objects (slope, baseline, df/dB per transition) from the
  reference ODMR.
- `nv_toolkit.two_point.estimate_delta_f_mhz(s_minus, s_plus, calibration)` —
  the one-line core formula.
- `nv_toolkit.two_point.normalised_signal_from_counts(signal, reference)` —
  converts raw ADC counts into the normalized intensity scale that the
  calibration expects (matches how the reference spectrum is normalized).

### Output

`data/lockin_multipoint/multipoint_lockin_peak_inference.csv` (wide), one row
per `(batch, block)` pair:

| column                    | meaning                                                  |
| ------------------------- | -------------------------------------------------------- |
| `batch`, `block`          | batch index, block (transition) index 1..8               |
| `f0_old_mhz`              | reference resonance frequency from the ODMR fit          |
| `f_minus_mhz`, `f_plus_mhz` | the two parked frequencies                              |
| `slope_minus_per_mhz`, `slope_plus_per_mhz` | calibration slopes (intensity / MHz) |
| `baseline_minus`, `baseline_plus` | reference normalized intensities at the parked freqs |
| `S_minus_norm`, `S_plus_norm` | current normalized intensities (signal/reference)    |
| `current_diff`, `baseline_diff`, `delta_d` | Δd built up step by step           |
| `delta_f_mhz`             | inferred frequency shift                                 |
| `f_new_mhz`               | inferred new peak frequency                              |
| `delta_B_proj_mT`, `delta_B_proj_uT` | projected B-field shift onto block's NV axis  |
| `sensitivity_mhz_per_mT`  | ‖df/dB‖ for this transition at BIAS_MT                   |

### Sanity checks the cell prints

- For the first batch acquired right after `lockin_plan`, every $|\Delta f_k|$
  should be $\lesssim 1$ MHz (within ADC noise + linewidth-relative drift) →
  the calibration is consistent. A larger value means the parked-frequency
  ADC read does not match the reference ODMR sweep — drift, gain change, or
  sign error.
- The slope-sign invariant `m_- < 0 < m_+` is checked for every block. A
  violation usually means two ODMR transitions are degenerate at the current
  bias field and the parked-plan algorithm picked overlapping windows; the
  affected block's $\Delta f$ is unreliable until the bias field is adjusted
  to lift the degeneracy.

### Cited literature

- El-Ella *et al.*, *Opt. Express* 2017 — lock-in slope is the calibration
  anchor.
- Clevenson *et al.*, *APL* 2018 — per-transition $\Delta f$ as the
  closed-loop error signal.
- Schloss *et al.*, *Phys. Rev. Applied* 2018 — multi-channel parked-frequency
  vector reconstruction.
- Hu *et al.*, *Micromachines* 2023 — multipoint DDS-based FSK lock-in.
- Zhang *et al.*, *Diam. Relat. Mater.* 2021 — slope-difference denominator
  with baseline-subtracted intensity differences.

## Live (continuous) time-series mode

The cells above run a single batch and stop. The `lockin_live` cell extends
this into a continuous loop: keep calling `acquire()`, Δf-convert each batch
on the fly, and stream both the wide-format peaks and the per-block Δf to a
timestamped CSV under `data/lockin_multipoint/`.

### The peaking-batching issue

`MultipointLockinODMR.initialize()` includes an optional polarization
pre-pulse:

```python
if self.cfg.pre_init:
    self.pulse(ch=self.cfg.mw_channel)              # MW polarization pulse
    self.trigger(pins=[laser_gate_pmod], ...)       # laser repump
    self.sync_all(readout_integration + relax_delay)
```

Every `prog.acquire()` call restarts the FPGA program from the top, which
means it re-runs `initialize()`. With `pre_init = True`, every batch begins
with a fresh polarization pulse → the first few reps of each batch sit at an
anomalously high PL because the spin state is over-polarized → a visible
peaking jump at every batch boundary in continuous mode.

### The fix the live cell applies

1. **Prime once** with a one-shot acquire on a throwaway program built with
   `pre_init=True`. The result is discarded; its only purpose is to drive the
   NV ensemble into the polarized steady state before live data starts.
2. **Run the live loop with `pre_init=False`** so each subsequent
   `acquire()` jumps straight into the unrolled body without firing the
   polarization pulse + sync_all again. The polarization-pulse ASM is removed
   from the compiled program entirely, not just skipped at runtime.
3. Continuous laser pumping during the readout windows of body iterations
   re-polarizes between freq pairs without a dedicated repump pulse.

### What the fix actually guarantees (be honest about scope)

What's structurally guaranteed by the code:

- **Per-batch repolarization pulse is gone.** With `pre_init=False`, the ASM
  for the polarization pulse is not in the compiled program. Every
  `acquire()` enters the body loop after only `synci(100)` waits — no MW
  pulse, no laser-gated repump, no sync_all delay. This eliminates the
  *programmed* over-polarization transient that the old `pre_init=True`
  pattern emitted at every batch boundary.
- **No FPGA program rebuild between batches.** The same `prog_live` runs
  every iteration of the live loop, so there's no upload jitter and no
  initialization beyond what's strictly needed.

What's *not* guaranteed by the code, and depends on hardware behavior:

- **The first batch after a long Python-side gap may still show a
  transient.** Between the priming acquire and the live loop starting,
  Python is doing housekeeping (deleting the priming program, building the
  live program, setting up the live plot, printing). During that window
  (~tens to hundreds of ms) the laser is OFF. NV spins partially relax
  toward thermal equilibrium. The very first batch of the live loop fires
  on partially-depolarized spins → contrast can be slightly elevated for
  the first ~few reps until the in-body laser pumping rebuilds steady
  state. With `reps = 10`, those few reps are most of the batch, so batch
  zero may genuinely sit a bit higher than batches 1, 2, 3, … The signal
  difference Δd that drives Δf still cancels common-mode offset, so this
  shows up as a small Δf glitch at t = 0, not a lasting bias.
- **Recurring per-batch peaks** (the old 8-peak cell's pathology) require
  a polarization pulse to fire at every acquire. That pulse is gone. So
  the *recurring* spike pattern is structurally impossible under the
  current code. If you still see one, it isn't `pre_init` — likely
  candidates: (a) a different cell rebuilt the program between batches,
  (b) network jitter dropping pulse timing, (c) laser gating misconfigured
  so the laser actually goes off between iterations.

How to verify on real hardware:

1. Run `lockin_live` with `LIVE_REPS_PER_BATCH = 10` and
   `LIVE_DURATION_SEC = 30`.
2. Open the saved CSV (`data/lockin_multipoint/multipoint_lockin_live_*.csv`)
   and plot `peak_01` (or any peak column) vs `batch`.
3. Expected: batch 0 might be slightly higher than the rest (the gap
   transient described above), but batches 1, 2, 3, … should sit on a flat
   noise floor. **No recurring sawtooth.**
4. If you see a recurring sawtooth, log `acq_seconds` per batch — if some
   batches are ~150 ms and others are ~500 ms, that's network jitter, not
   a polarization issue. The fix for that is `LIVE_PLOT_REFRESH_EVERY = 5`
   or `10` to reduce per-batch plotting overhead.

### What the live cell saves

`data/lockin_multipoint/multipoint_lockin_live_<YYYYMMDD_HHMMSS>.csv`, one
row per batch, with the standard wide-format peak columns **plus** the
per-block Δf and inferred new peak frequency:

| column                    | meaning                                                            |
| ------------------------- | ------------------------------------------------------------------ |
| `batch`, `time_s`, `timestamp_epoch_s`, `acq_seconds` | per-batch metadata (time_s used by the toolkit) |
| `peak_<NN>`, `peak_<NN>_ref`, `peak_<NN>_freq_mhz` | wide-format raw (toolkit-compatible) |
| `delta_f_mhz_b<NN>`       | inferred frequency shift for block NN                              |
| `f_new_mhz_b<NN>`         | inferred new peak frequency for block NN                           |

The cell live-plots `delta_f_mhz_b<NN>` for every block over time, flushes
the CSV every 5 s, and stops on either `LIVE_DURATION_SEC` elapsed or
KeyboardInterrupt. After it stops it prints a per-block summary with
`mean(Δf)`, `std(Δf)` and `std(ΔB_proj_uT) = std(Δf) / sensitivity` —
the latter is a rough sensitivity-floor read for the run.

### Why the fix doesn't apply to the single-batch cells

`lockin_acquire` and `lockin_to_peak_frequency` only call `acquire()` once
each. The single pre_init pulse at the start polarizes the spins into a
clean state, and the result is what you actually want — there's no later
batch to be "second" relative to. So those cells leave `pre_init=True`.

## Source files referenced

- [`multipoint_lockin_program.py`](multipoint_lockin_program.py)
  &nbsp;— `MultipointLockinODMR` class
- [`Lockin_module.ipynb`](Lockin_module.ipynb)
  &nbsp;— the four workflow cells (`lockin_plan`, `lockin_acquire`,
  `lockin_to_peak_frequency`, `lockin_reconstruct`)
- `../qickdawg/nvpulsing/lockinodmr.py`
  &nbsp;— the patched upstream class with `odmr_reference_offres_mhz` support, used by ODMR sweeps
- `../nv_toolkit/tui.py`
  &nbsp;— `_suggest_parked_frequencies` and `_compute_live_snapshot` helpers
- `../nv_toolkit/two_point.py`
  &nbsp;— `TwoPointCalibration`, `estimate_delta_f_mhz`,
  `normalised_signal_from_counts`
- `../nv_toolkit/intensity_tracking.py`
  &nbsp;— `build_blockwise_calibrations`
