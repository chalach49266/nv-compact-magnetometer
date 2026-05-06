# Lock-In Extensions — Implementation Notes

**File:** `lockin_extensions.py`  
**Notebook section:** Section 7 of `01_basic_nv_testing.ipynb`  
**Depends on:** `twopoint_lockin.py`, `odmr_sensitivity.py`  
**Date:** April 2026

These three extensions build on the two-point hardware-rate lock-in in `twopoint_lockin.py`. None require FPGA firmware changes.

---

## Extension 1 — Sinusoidal FM Lock-In

### What it is

Instead of switching between exactly 2 frequencies (Section 6), the sinusoidal FM method sweeps **N equally-spaced frequencies** across the range [f₀−δ, f₀+δ] and applies **sine-weighted demodulation** to each batch of per-rep raw data.

The two-point method is mathematically the N=2 special case of this — proven below.

### Physics: why sine weighting

In a true analog sinusoidal FM lock-in:
```
MW frequency:  f(t) = f₀ + δ·sin(ωₘt)
PL response:   S(t) ≈ L(f₀) + L'(f₀)·δ·sin(ωₘt) + higher harmonics
```

The lock-in amplifier multiplies S(t) by the reference sin(ωₘt) and integrates:
```
X = (2/T)·∫ S(t)·sin(ωₘt)dt ≈ L'(f₀)·δ
```

The output X is proportional to the **derivative of the ODMR line at f₀**, which is proportional to a B-field induced shift of the resonance.

### Discrete approximation with linear sweep

With a linear sweep of N frequencies across [f₀−δ, f₀+δ]:
```
f[k] = f₀ + δ·(2k/(N-1) − 1)   for k = 0, 1, ..., N-1
```

The sine value at each frequency (mapping frequency to phase via f = f₀ + δ·sin(θ)):
```
sin(θ[k]) = (f[k] − f₀)/δ = 2k/(N-1) − 1
```

So the **in-phase (X) demodulation weights** are simply:
```
w_sin[k] = 2k/(N-1) − 1       ranges from −1 to +1 linearly
```

The **quadrature (Y) demodulation weights** (cosine of same phase):
```
w_cos[k] = √(1 − w_sin[k]²)   0 at edges, 1 at center
```

### Verification: N=2 recovers the two-point result

For N=2:
```
w_sin = [2·0/(2-1) − 1,  2·1/(2-1) − 1] = [−1, +1]
X = contrast[k=0]·(−1) + contrast[k=1]·(+1)
  = −contrast_f1 + contrast_f2
  = −(contrast_f1 − contrast_f2)
  = −lockin_signal  (from two-point, up to sign convention)
```

✓ Confirmed identical.

### What X and Y measure

| Output | Formula | Physical meaning |
|--------|---------|-----------------|
| X | Σ contrast[k] × w_sin[k] | ∝ dPL/df at f₀ ∝ B-field shift |
| Y | Σ contrast[k] × w_cos[k] | ∝ average ODMR contrast depth (not field-sensitive) |
| R | √(X²+Y²) | total lock-in magnitude |

When the resonance is centred at f₀: X ≈ 0 (no asymmetry), Y ≈ maximum (deepest dip at centre). When B shifts the resonance: X ≠ 0 (asymmetry appears), Y decreases slightly.

### Timing with N points

Increasing N slows the per-cycle rate because each cycle visits N frequencies:

| N | Cycle time | Switching rate | Nyquist BW |
|---|-----------|---------------|-----------|
| 2  | 855 µs  | 1,169 Hz | 585 Hz |
| 4  | 1,711 µs | 585 Hz  | 292 Hz |
| 8  | 3,421 µs | 292 Hz  | 146 Hz |
| 16 | 6,842 µs | 146 Hz  | 73 Hz  |

For navigation (< 10 Hz signal): N=8 (default) gives 14× more bandwidth than needed while providing a good sine approximation.

### Low-pass filter

After the run, a zero-phase Butterworth filter (scipy `sosfiltfilt`) can be applied to X. This is the digital equivalent of the LPF stage in an analog lock-in amplifier. The cutoff sets the effective **measurement bandwidth**:
- `lowpass_cutoff_hz = 50` → 50 Hz bandwidth (good for navigation)
- `lowpass_cutoff_hz = 10` → 10 Hz bandwidth (tighter, more noise rejection)
- `lowpass_cutoff_hz = None` → no filtering (raw per-cycle signal)

### Implementation key points

```python
# Raw data shape: (reps, N, 2) — same as two-point but N=n_points instead of 2
raw = prog.acquire(raw_data=True)

# Contrast per rep per frequency: shape (reps, N)
contrast = (raw[:, :, 0] - raw[:, :, 1]) / scale

# Demodulate: matrix multiply with weight vectors
w_sin, w_cos = _sinewave_weights(n_points)
X = contrast @ w_sin   # shape (reps,)
Y = contrast @ w_cos   # shape (reps,)
```

The entire demodulation is a single NumPy matrix multiply — negligible compute time compared to the FPGA acquisition.

---

## Extension 2 — PID Feedback Loop (Resonance Tracking)

### What it is

An **open-loop** lock-in (Section 6 and 7A) measures the PL difference at two fixed frequencies. If the magnetic field shifts the resonance by more than ~FWHM/4, the measurement frequencies are no longer on the optimal part of the slope and sensitivity degrades.

A **closed-loop** feedback lock-in continuously adjusts the centre frequency f₀ to keep the measurement points on the steepest part of the slope, even as the field changes. The tracked f₀ then directly encodes the magnetic field.

### Feedback loop schematic

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   f₁ = f₀ − slope_frac × FWHM                              │
│   f₂ = f₀ + slope_frac × FWHM                              │
│                ↓                                            │
│   LockinODMR.acquire(raw_data=True)  ← FPGA ~855 ms        │
│                ↓                                            │
│   error = mean(contrast_f1 − contrast_f2)                  │
│                ↓                                            │
│   PID:  correction = kp·error + ki·∫error + kd·d(error)/dt │
│                ↓                                            │
│   f₀ ← f₀ + correction                                     │
│                ↓                                            │
│   B(t) = (f₀ − f₀_initial) / γ_NV     [γ_NV = 28 GHz/T]  │
│                ↓                                            │
│   loop → next batch                                         │
└─────────────────────────────────────────────────────────────┘
```

### Sign convention (critical)

When B increases (resonance shifts to higher frequency by Δf):
- Dip moves right
- f₁ is now further from resonance → contrast_f1 becomes less negative (increases)
- f₂ is now closer to resonance → contrast_f2 becomes more negative (decreases)
- error = contrast_f1 − contrast_f2 **becomes positive**
- Desired correction: **increase f₀** to follow the dip right
- `f₀ += kp × error` → positive error → positive correction ✓

### Magnetic field conversion

The tracked frequency f₀(t) encodes the field via the NV gyromagnetic ratio:
```
ΔB(t) = Δf₀(t) / γ_NV
       = (f₀_tracked(t) − f₀_initial) / 28 GHz/T
       = (f₀_tracked(t) − f₀_initial) / (28×10⁻⁶ MHz/nT)
```

Example:
- f₀ shifts by 0.1 MHz
- ΔB = 0.1 MHz / 28×10⁻³ MHz/µT = 3.57 µT

### PID tuning procedure

The system is a first-order feedback loop. The plant gain (lock-in signal per unit frequency shift) is:
```
G = 2 × slope_at_f1 [ADC/MHz]
  ≈ 2 × 0.77 × max_slope [ADC/GHz] × 1e-3 [GHz/MHz]
```

From our measured data: max_slope ≈ 10,919 ADC/GHz (from Section 4)
```
G ≈ 2 × 0.77 × 10,919 × 1e-3 = 16.8 ADC/MHz
```

The proportional gain kp (units: MHz/ADC) should satisfy `kp × G < 1` for stability:
```
kp < 1/G ≈ 1/16.8 ≈ 0.060 MHz/ADC
```

Starting value: `kp = 0.05` (just below the instability boundary).
If the system is stable but slow to converge: increase toward 0.05.
If f₀ oscillates: decrease kp.

**Tuning sequence:**
1. Run with kp=0.01, ki=0, kd=0. Observe convergence.
2. Increase kp until the error converges in a few batches without oscillating.
3. If there is a steady-state offset after convergence, add small ki (start at kp/10).
4. kd is typically not needed for this batch-level loop.

### Implementation note: program re-creation per batch

Unlike the two-point and sinusoidal FM methods (which create the program once), the feedback loop must **create a new LockinODMR per batch** because f₀ changes and the QICK sweep frequency registers are set at compile time. The compilation overhead is ~10–50 ms per batch (small compared to the ~855 ms acquisition time).

```python
# Per batch: recompile with updated f1, f2
cfg.mw_start_fMHz = f0 - slope_fraction * fwhm_mhz
cfg.mw_end_fMHz   = f0 + slope_fraction * fwhm_mhz
prog = qd.LockinODMR(cfg)         # recompile (~10-50 ms)
raw  = prog.acquire(raw_data=True) # acquire (~855 ms)
```

### Outputs

| Column | Description |
|--------|-------------|
| `time_s` | Batch mid-point timestamp |
| `f0_tracked_mhz` | Current centre frequency (the "measurement" output) |
| `f1_mhz`, `f2_mhz` | Measurement frequencies used in this batch |
| `lockin_mean` | Mean lock-in signal — the PID error (→ 0 when locked) |
| `lockin_std` | Std of lock-in within batch — noise floor indicator |
| `correction_mhz` | Frequency correction applied to f₀ this batch |
| `B_field_nT` | Inferred B-field shift from initial (nT) |

---

## Extension 3 — Sensitivity-Optimal Slope Fraction

### What it is

The two measurement frequencies f₁ and f₂ are placed at `slope_fraction × FWHM` from the resonance centre. The default is `slope_fraction = 0.5` (half-max points). The sensitivity-optimal choice places them at the **inflection points** of the Lorentzian where |dL/df| is maximum.

### Mathematics

For a Lorentzian:
```
L(f) = offset − A·γ² / ((f−f₀)² + γ²)
```

The slope:
```
dL/df = 2A·γ²·(f−f₀) / ((f−f₀)² + γ²)²
```

Setting d²L/df² = 0 to find the inflection points:
```
(f−f₀)² + γ² = 4·(f−f₀)² / something...
```

Solving gives inflection points at:
```
f_inf = f₀ ± γ/√3 = f₀ ± FWHM/(2√3)
slope_fraction_opt = 1/(2√3) ≈ 0.2887
```

### Slope comparison at different fractions

| slope_fraction | offset (MHz) | |dL/df| (% of max) |
|---------------|-------------|-------------------|
| 0.289 (optimal) | 5.03 MHz | **100%** (maximum) |
| 0.5 (half-max)  | 8.74 MHz | **77%** |
| 0.1             | 1.75 MHz | 48% |
| 0.7             | 12.2 MHz | 46% |

Using slope_fraction = 0.289 instead of 0.5 gives **~30% more signal** for the same noise → ~30% better sensitivity.

### Numerical values for our measured ODMR

From the April 2026 measurement (center = 2870.91 MHz, FWHM = 17.47 MHz, γ = 8.735 MHz):

```
γ/√3 = 8.735 / 1.732 = 5.042 MHz

Optimal positions:
  f1_opt = 2870.91 − 5.042 = 2865.868 MHz
  f2_opt = 2870.91 + 5.042 = 2875.952 MHz
  slope_fraction = 5.042 / 17.47 = 0.2887

Half-max positions (default):
  f1_hm  = 2870.91 − 8.735 = 2862.175 MHz
  f2_hm  = 2870.91 + 8.735 = 2879.645 MHz
  slope_fraction = 0.5

Slope at optimal / slope at half-max ≈ 1.30  (30% improvement)
```

### Experimental verification

Section 7C of the notebook runs both `slope_fraction=0.289` and `slope_fraction=0.5` for 30 seconds each and compares the `lockin_signal` standard deviation. The expected result:

- If **shot-noise limited**: the std scales with the number of photons, which is roughly the same at both positions (near the half-max, PL is similar). The std ratio ≈ 1, and the **sensitivity** ratio (std/slope) ≈ 1/1.30 = 0.77× better at optimal.
- If **technical noise dominated**: the std at optimal may be lower because the higher slope means the same noise in PL translates to smaller apparent field noise.

In either case, using the optimal slope_fraction is never worse than the half-max choice and is typically 10–30% better for field sensitivity.

### The compare_slope_fractions plot

Two panels:

**Left — ODMR reference trace:** Shows the measured ODMR dip with vertical dashed lines at each slope_fraction's f₁ and f₂ positions. Visually confirms that 0.289 places the lines closer to the resonance centre (steeper slope region) while 0.5 places them at the half-max level.

**Right — Lorentzian slope profile:** Plots |dL/df| as a function of offset from centre. The maximum is at γ/√3 (the optimal), and the curve shows how the slope drops away on either side. Vertical dashed lines mark the selected slope_fractions.

---

## Summary: Relationship Between the Three Methods

```
Two-point lock-in (twopoint_lockin.py, Section 6)
        ↓ generalise N → 2→8 points
Sinusoidal FM (Extension 1, Section 7A)
        ↓ add frequency tracking
PID feedback loop (Extension 2, Section 7B)
        ↓ optimise operating point
Sensitivity-optimal slope fraction (Extension 3, Section 7C)
        ↓ (future) move demodulation to FPGA + true continuous FM
Full analog lock-in (requires new QICK assembly program)
```

All four implemented methods share the same FPGA infrastructure (LockinODMR + raw_data=True) and differ only in Python-level signal processing.

---

## File Summary

| File | Purpose |
|------|---------|
| `lockin_extensions.py` | All three extension functions |
| `twopoint_lockin.py` | Base two-point method (used by Extensions 1 and 3) |
| `odmr_sensitivity.py` | Lorentzian fit, used by all `_from_odmr` helpers |
| `twopoint_lockin_notes.md` | Detailed timing/physics notes for the base method |
| `lockin_extensions_notes.md` | This file — implementation notes for the three extensions |
