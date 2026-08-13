# Two-Point Lock-In Modulation — Implementation Notes

**Project:** NV Compact Magnetometer  
**Date:** April 2026  
**Purpose:** Document the physics, timing calculations, and implementation details of the two-point lock-in modulation for preliminary magnetometry testing.

> **Superseded in places (2026-08-12).** The timing model here predates the 2026-08-06
> measurements. Two corrections matter: a rep costs `n_freqs x readout_integration_tus`
> (the relax windows are absorbed, not serialised), and the acquisition rate is FPGA-bound,
> not host-bound. The readout window was also moved 213 -> 120 us. See
> [`2026-08-06_twopoint_timing/TIMING_AND_NOISE_ANALYSIS.md`](2026-08-06_twopoint_timing/TIMING_AND_NOISE_ANALYSIS.md).

---

---

## 1. Background: Why Lock-In Modulation

### 1.1 The Problem with Simple ODMR Sweeping

The standard approach (Section 3 of the notebook) sweeps the microwave frequency from 2700 to 3000 MHz, collecting one PL value per frequency point. This gives a complete spectrum but is very slow — a full sweep takes 1–10 seconds depending on configuration. Any magnetic field change that happens between the start and end of a sweep is missed.

For navigation, we need to track the Earth's magnetic field in real time. The Earth's field varies at frequencies below ~10 Hz, so we need a measurement bandwidth of at least 10–100 Hz. A full sweep cannot achieve this.

### 1.2 The Lock-In Solution

Instead of sweeping the full spectrum, we park at two fixed frequencies on either slope of the ODMR resonance dip and measure the **differential photoluminescence (PL) contrast** continuously over time. This differential signal — the lock-in output — has two key properties:

1. **It is proportional to magnetic field shifts.** When the magnetic field changes, the ODMR resonance shifts, and the difference in PL between the two frequencies changes proportionally.
2. **It rejects common-mode noise.** Laser intensity fluctuations, vibration, and ambient light changes affect both measurement frequencies equally and cancel in the subtraction.

### 1.3 Naming: Square-Wave vs Sinusoidal Lock-In

A **sinusoidal FM lock-in** (what commercial lock-in amplifiers do) continuously modulates the microwave frequency as a sine wave:

```
f(t) = f₀ + δ·sin(2π·f_mod·t)
```

and demodulates the PL signal by multiplying by a reference sine. This gives the best possible noise rejection because only the component at exactly `f_mod` survives the demodulation.

Our implementation uses **square-wave frequency switching** — alternating between two fixed frequencies f₁ and f₂. This is the discrete approximation of the sinusoidal approach and is the standard first step before implementing full FM. For NV magnetometry it is well-established in the literature (El-Ella et al., Optics Express 2017) and for square-wave modulation with NV hyperfine structure, the slope can be steeper than the sinusoidal case.

---

## 2. Frequency Selection

### 2.1 The ODMR Resonance

From our measured data (Section 4 sensitivity analysis, April 2026):

```
Center frequency:  f₀ = 2870.91 MHz
FWHM:              Γ  = 17.47 MHz
Lorentzian shape:  L(f) = offset + A·γ² / ((f - f₀)² + γ²)
                   where γ = FWHM/2 = 8.735 MHz
```

The resonance is a **dip** in PL (PL decreases when MW is on resonance because NV spins are driven to the dark ms = ±1 state).

### 2.2 Choosing f₁ and f₂

We place the two measurement points symmetrically about the center, at a distance of `slope_fraction × FWHM` from center:

```
f₁ = f₀ - slope_fraction × FWHM   (left slope, PL decreasing toward center)
f₂ = f₀ + slope_fraction × FWHM   (right slope, PL increasing away from center)
```

**Default (`slope_fraction = 0.5`):**
```
f₁ = 2870.91 - 0.5 × 17.47 = 2862.18 MHz
f₂ = 2870.91 + 0.5 × 17.47 = 2879.64 MHz
```

These are the **half-maximum points** of the Lorentzian — the points where PL = offset + A/2.

**Alternative (`slope_fraction = 0.289`):**
```
f₁ = 2870.91 - 0.289 × 17.47 = 2865.86 MHz
f₂ = 2870.91 + 0.289 × 17.47 = 2875.96 MHz
```

These are the **inflection points** of the Lorentzian — where the slope `dL/df` is maximum. For a Lorentzian, the inflection points are at f₀ ± γ/√3 = f₀ ± FWHM/(2√3) ≈ f₀ ± 0.289·FWHM. This gives the steepest response to a field shift and is technically the sensitivity-optimal choice.

For a preliminary test, the half-max points (0.5) are the intuitive starting point and can be easily seen on the ODMR plot.

### 2.3 Physical Meaning of the Lock-In Signal

```
ODMR dip (zero field, symmetric):

PL ─────────────────────────────────────────────────────────
                    ╲             ╱
                     ╲           ╱
          f₁ ●────────×         ×────────● f₂
                      ↓         ↓
                   contrast   contrast
                   at f₁      at f₂
                   (equal)    (equal)
                   → lock-in signal ≈ 0
```

```
After magnetic field B shifts the resonance to the right by Δf = γ_NV × B:

PL ─────────────────────────────────────────────────────────
                         ╲             ╱
                          ╲           ╱
          f₁ ●─────────────×         ×──● f₂
                           ↑           ↑
                        larger      smaller
                        contrast    contrast
                        → lock-in signal > 0  (proportional to B)
```

The lock-in signal is **linear in the magnetic field** for small shifts (Δf << FWHM), with sensitivity determined by the slope of the ODMR line at the operating point.

---

## 3. FPGA Implementation — Detailed Timing

### 3.1 The body() Pulse Sequence

`LockinODMR.body()` runs the following sequence for each rep at each frequency. The sequence is defined in the FPGA assembly language (QICK QAP ISA) and runs entirely in hardware with nanosecond timing precision:

```
Time →    0                213.33 µs    213.83 µs             427.16 µs    427.66 µs
          │                │            │                      │            │
MW:       ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
          │←── MW ON ────→│            │                      │            │
          │                             │←─────── MW OFF ───────────────→│
          │                │            │                      │            │
Laser:    ████████████████ █████████████████████████████████████ ███████████
          (always on during readout)
          │                │            │                      │            │
ADC:      ████████████████             ████████████████████████
          │←─ signal  ───→│            │←──── reference ──────────────→│
          │  (MW on PL)   │ ← 0.5 µs → │     (MW off PL)              │ ← 0.5 µs →
          │                │   relax    │                              │   relax
```

**Per body() duration:**
```
Signal window:    213.33 µs   (readout_integration_tus = max_int_time_tus)
Relax gap:          0.50 µs   (relax_delay_tns = 500 ns)
Reference window: 213.33 µs
Relax gap:          0.50 µs
─────────────────────────────
Total:            427.66 µs per body() execution
```

**What each window measures:**
- **Signal window (MW ON):** PL with microwaves on. At resonance, PL drops because spins are driven to the dark ms = ±1 state.
- **Reference window (MW OFF):** PL with microwaves off. Serves as the baseline. Any laser fluctuation within a single body() call cancels when we compute contrast = signal − reference.

### 3.2 The FPGA Loop Structure

From `NVAveragerProgram.make_program()` (confirmed by reading the source code, line 76):

```python
# reps loop is the outer loop, first-added sweep is innermost loop
loop_dims = [cfg['reps'], *self.sweep_axes[::-1]]
```

The FPGA assembly loop is:

```
LOOP_rep:        ← outer loop, runs reps_per_batch = 1000 times
  LOOP_sweep:    ← inner loop, runs 2 times (f1, then f2)
    body()       ← 427.66 µs of pulse + readout
    update frequency register  ← instantaneous in FPGA
  loopnz LOOP_sweep
loopnz LOOP_rep
```

This means **within each rep, the FPGA alternates f1 → f2 naturally**, without any Python involvement. The frequency update between f1 and f2 is a single register write — effectively instantaneous (nanosecond timescale) compared to the 427 µs body duration.

### 3.3 One Rep = One Hardware Switching Cycle

```
Rep 0:
  [body@f1: 427.66 µs] → [body@f2: 427.66 µs]
  └── f1↔f2 transition: ~nanoseconds (FPGA register update)

Rep 1:
  [body@f1: 427.66 µs] → [body@f2: 427.66 µs]

...

Rep 999:
  [body@f1: 427.66 µs] → [body@f2: 427.66 µs]
```

**Per-rep time (one complete f1+f2 cycle):**
```
= 2 × 427.66 µs = 855.32 µs
```

**Hardware f1↔f2 switching rate:**
```
= 1 / 855.32 µs = 1,169 Hz ≈ 1.17 kHz
```

This is within the 1–10 kHz range used in published NV lock-in magnetometers (El-Ella et al. 2017, Parashar et al. 2021, Li et al. 2023).

### 3.4 One Batch = 1000 Reps

```
Total batch time = 1000 reps × 855.32 µs/rep = 855.32 ms ≈ 0.855 s
```

Python calls `prog.acquire(raw_data=True)` once, the FPGA runs for 855 ms, then Python receives all 1000 data points at once — identical in structure to how `run_fast_pl_no_display` uses `d_buf`.

---

## 4. The Raw Data Buffer

### 4.1 Buffer Shape

`acquire(raw_data=True)` returns a NumPy array of shape:

```
raw.shape = (1000, 2, 2)
              │     │  └── readout index: 0 = MW on (signal), 1 = MW off (reference)
              │     └───── sweep index:   0 = f1,             1 = f2
              └─────────── rep index:     0 → 999
```

Values are raw ADC integer counts (int64), not yet normalised.

### 4.2 Buffer Contents — Explicit Mapping

```
raw[i, 0, 0]  →  ADC counts, MW on,  at f1, rep i  (signal at f1)
raw[i, 0, 1]  →  ADC counts, MW off, at f1, rep i  (reference at f1)
raw[i, 1, 0]  →  ADC counts, MW on,  at f2, rep i  (signal at f2)
raw[i, 1, 1]  →  ADC counts, MW off, at f2, rep i  (reference at f2)
```

### 4.3 Normalisation

The ADC integrates over the readout window, so the accumulated count is proportional to `integration_time × photon_rate`. To get a rate (ADC units/time), divide by the integration register value:

```python
scale = readout_integration_treg = 65535   (for max_int_time_tus = 213.33 µs)

signal_f1[i]    = raw[i, 0, 0] / scale
reference_f1[i] = raw[i, 0, 1] / scale
signal_f2[i]    = raw[i, 1, 0] / scale
reference_f2[i] = raw[i, 1, 1] / scale
```

---

## 5. Computing the Lock-In Signal

### 5.1 Step-by-Step for Rep i

```
Step 1 — Contrast at f1 (eliminates laser noise within one body() call):
  contrast_f1[i] = signal_f1[i] - reference_f1[i]

Step 2 — Contrast at f2:
  contrast_f2[i] = signal_f2[i] - reference_f2[i]

Step 3 — Lock-in output (eliminates common-mode drift between the two frequencies):
  lockin[i] = contrast_f1[i] - contrast_f2[i]
```

### 5.2 What Each Subtraction Cancels

| Subtraction | What it removes |
|---|---|
| `signal - reference` (within one body) | Laser fluctuations faster than ~214 µs (photon shot noise remains) |
| `contrast_f1 - contrast_f2` (across one rep, ~428 µs apart) | Slow laser drift, thermal drift, common vibration |
| Lock-in demodulation over many reps | Noise at frequencies other than the switching frequency (1.17 kHz) |

### 5.3 Noise That Remains

- **Photon shot noise:** fundamental quantum limit, cannot be eliminated. Sets the sensitivity floor.
- **Noise at the switching frequency (1.17 kHz):** any periodic disturbance at exactly 1.17 kHz would appear as a false signal. In practice this is rare.
- **Spin projection noise:** fundamental for ensemble NV measurements.

---

## 6. Full Numerical Summary

### 6.1 Configuration Parameters

| Parameter | Value | Source |
|---|---|---|
| `readout_integration_tus` | 213.33 µs | `qd.max_int_time_tus` |
| `readout_integration_treg` | 65535 | auto-converted |
| `relax_delay_tns` | 500 ns | set in default_config |
| `relax_delay_tus` | 0.5 µs | auto-converted |
| `mw_gain` | 10500 | from ODMR config |
| `nsweep_points` | 2 | set in twopoint_lockin.py |
| `reps_per_batch` | 1000 | set in notebook |
| `center_fMHz` | 2870.91 MHz | from Lorentzian fit |
| `fwhm_mhz` | 17.47 MHz | from Lorentzian fit |
| `slope_fraction` | 0.5 (default) | notebook parameter |
| `f1` | 2862.18 MHz | center − 0.5 × FWHM |
| `f2` | 2879.64 MHz | center + 0.5 × FWHM |

### 6.2 Timing Numbers

| Quantity | Calculation | Value |
|---|---|---|
| Per body() | 2 × 213.33 + 2 × 0.5 µs | 427.66 µs |
| Per rep (f1 + f2) | 2 × 427.66 µs | 855.32 µs |
| Hardware switching rate | 1 / 855.32 µs | **1,169 Hz** |
| Per batch (1000 reps) | 1000 × 855.32 µs | **855.3 ms** |
| Points per batch | 1000 | |
| Points per 60 s run | 70 batches × 1000 | **~70,000 pts** |
| Effective sample rate | 1000 / 0.855 s | **1,169 pts/s** |
| Nyquist bandwidth | 1169 / 2 | **585 Hz** |
| Earth-field bandwidth needed | < 10 Hz | satisfied with 58× margin |

### 6.3 Sensitivity Estimate (Preliminary)

From the Section 4 ODMR measurement:

```
Max slope of Lorentzian:   dL/df|_max  = 10,919 ADC / GHz   (at inflection point)
Noise std (reference):     σ           = 1.48 ADC
Point time (per freq):     τ           = 855.32 µs / 2 = 427.66 µs

Noise density:  σ_density = σ × √τ = 1.48 × √(427.66×10⁻⁶) = 0.0306 ADC·√s

For two-point lock-in (both slopes contribute, response is doubled):
  Response = 2 × γ_NV × max_slope = 2 × 28 GHz/T × 10,919 ADC/GHz = 611,464 ADC/T

Sensitivity ≈ σ_density / Response = 0.0306 / 611,464 ≈ 50 nT/√Hz  (estimate)
```

This is comparable to the single-point ODMR sensitivity (58–71 nT/√Hz from Section 4) because the two-point method doubles both the signal response and the noise. The advantage of two-point lock-in is **noise rejection** (eliminating 1/f and common-mode noise), not a fundamental SNR improvement over shot noise.

---

## 7. Switching Rate vs Integration Time Trade-off

The switching rate is set entirely by `readout_integration_tus`. Reducing it increases the switching rate but decreases SNR per point (less photon collection per window):

| `readout_integration_tus` | Body time | Cycle time | Switching rate | SNR factor |
|---|---|---|---|---|
| 213.33 µs (max, current) | 427.66 µs | 855.32 µs | **1,169 Hz** | 1.0× (baseline) |
| 100 µs | 201.0 µs | 402.0 µs | **2,488 Hz** | 0.69× |
| 50 µs | 101.0 µs | 202.0 µs | **4,950 Hz** | 0.48× |
| 10 µs | 21.0 µs | 42.0 µs | **23,810 Hz** | 0.22× |

SNR factor = √(integration_tus / 213.33) because shot noise scales as √(integration time).

For navigation (< 10 Hz field changes), the current 1.17 kHz switching rate with maximum integration is the optimal choice: it sits well above the 1/f noise corner while maximising photon collection per point.

---

## 8. Implementation Details

### 8.1 File Structure

```
Initial Test/
├── 01_basic_nv_testing.ipynb   — main notebook (Section 6 added)
├── twopoint_lockin.py          — NEW: lock-in acquisition functions
├── odmr_sensitivity.py         — existing: Lorentzian fit, sensitivity calculation
└── twopoint_lockin_notes.md    — this file
```

### 8.2 Key Functions in twopoint_lockin.py

**`run_twopoint_lockin(config, *, center_fMHz, fwhm_mhz, ...)`**
- Clones config, sets `mw_start_fMHz=f1`, `mw_end_fMHz=f2`, `nsweep_points=2`
- Creates `LockinODMR(cfg)` once before the loop
- Calls `prog.acquire(raw_data=True)` per batch → gets `(reps, 2, 2)` raw buffer
- Normalises, computes contrast at f1 and f2, computes lock-in difference
- Appends all per-rep rows to a DataFrame
- Returns DataFrame with 1000 × n_batches rows

**`twopoint_lockin_from_odmr(odmr_result, config, **kwargs)`**
- Calls `estimate_odmr_sensitivity()` to extract center and FWHM from a prior ODMR sweep
- Passes to `run_twopoint_lockin()`
- Convenience wrapper so the user doesn't need to manually enter the fit parameters

### 8.3 What Was NOT Changed

- `lockinodmr.py` — unchanged. Used as-is with `nsweep_points=2`.
- `nvaverageprogram.py` — unchanged. The loop structure (reps outer, sweep inner) already gives us kHz switching.
- `nvconfiguration.py` — unchanged. Unit conversion system handles all register values.
- No FPGA firmware changes. The bitstream is unchanged.

### 8.4 Why raw_data=True Works

`LockinODMR.acquire(raw_data=True)` skips the `analyze_results()` step that averages over reps. Instead it returns `super().acquire()` directly, which is `self.d_buf[0][..., 0]` — the full `(reps, nsweep_points, readouts_per_experiment)` buffer. This is identical in principle to how `nv_live_helpers.py` reads `program.d_buf[0]` and `program.d_buf[1]` directly for the fast PL readout.

---

## 9. Connection to Full Lock-In Theory

### 9.1 What We Have vs Full Analog Lock-In

| Aspect | Our Implementation | Full Analog Lock-In |
|---|---|---|
| Modulation | Square-wave f₁↔f₂ switching at 1.17 kHz | Sinusoidal FM: f₀ + δ·sin(ωₘt) at 1–100 kHz |
| Demodulation | Per-rep subtraction: contrast(f₁) − contrast(f₂) | Multiply raw PL × reference sine, low-pass filter |
| Noise rejection | Common-mode (laser drift, vibration) | Common-mode + all noise except at ωₘ |
| 1/f rejection | Partial (modulated above ~1 kHz) | Full (bandwidth set by LPF cutoff) |
| Implementation | Pure Python, no firmware changes | Requires FPGA FM generation + on-FPGA demodulation |

### 9.2 The Professor's Sine Wave Comment

The "adding a sine function" refers to sinusoidal FM modulation where the MW frequency varies continuously as:
```
f(t) = f₀ + δ·sin(2π·f_mod·t)
```

The resulting PL signal also varies sinusoidally. A lock-in amplifier then multiplies by the same reference sine and low-passes — this is **synchronous demodulation** and rejects all noise not at exactly `f_mod`. Our square-wave switching is the discrete version: multiply by a square reference (±1) and sum, which is mathematically equivalent to demodulation at `f_mod` plus its odd harmonics.

For NV centers with hyperfine structure (14N nucleus creates three closely-spaced lines), El-Ella et al. (2017) showed that **square-wave modulation can outperform sinusoidal** when the linewidth-to-hyperfine-separation ratio is above 1/4 — which is often the case in bulk diamond ensemble measurements.

### 9.3 Next Steps Toward a Full Lock-In

1. **Current (done):** Square-wave two-point switching at 1.17 kHz using existing LockinODMR + raw_data=True.
2. **Next:** Implement sinusoidal FM within the FPGA pulse sequence (modify body() to ramp frequency sinusoidally within one rep). Requires new QAP assembly code in a new NVAveragerProgram subclass.
3. **After that:** On-FPGA demodulation — multiply the accumulated PL by a stored reference sine in firmware, low-pass filter in FPGA fabric. Latency < 10 µs, bandwidth > 100 kHz (as described in the patent).
4. **Full locking mode:** Use the demodulated output as an error signal for a PID feedback loop that continuously adjusts f₀ to track the resonance in real time.

---

## 10. Literature Context

| Reference | Modulation Rate | Bandwidth | Notes |
|---|---|---|---|
| El-Ella et al., Opt. Exp. 2017 | — | — | Compares square vs sine FM for NV; square preferred with hyperfine |
| Li et al., Micromachines 2023 | kHz modulation | 75 Hz | Geomagnetic navigation; optimal bandwidth 75 Hz with CMR noise cancellation |
| Parashar et al., Sci. Rep. 2021 | few kHz | 50–200 fps | Widefield lock-in camera imaging |
| Wang et al., Rev. Sci. Inst. 2022 | FPGA frequency-locked | 10 kHz | Closed-loop tracking, 4.2 nT/√Hz |
| Fu et al., Opt. Exp. 2025 | Multi-channel FM | real-time | 4-axis simultaneous, 0.54–0.93 nT/√Hz, 28× faster than hopping |

Our 1.17 kHz switching rate is within the range used by Li et al. (2023) for geomagnetic navigation and Parashar et al. (2021) for imaging. The 75 Hz optimal measurement bandwidth from Li et al. is well within our 585 Hz Nyquist limit.
