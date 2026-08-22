# Drone-detection analysis — method, 2026-08-19

Everything in `drone_detection_2026-08-19.pdf` is produced by one script:

```bash
cd "output/slides/2026-08-19_drone_detection"
python3 make_figs.py                      # -> figs/*.pdf, figs/*.png, numbers.txt
pdflatex drone_detection_2026-08-19.tex   # twice, for the frame counter
```

`numbers.txt` is the audit trail: every number quoted on a slide is printed there
by the same line of code that draws it. No number is typed by hand into the
LaTeX.

---

## 1. Data

### 16-point lock-in, streaming — 146.71 Hz, 4401 samples, 30.0 s

`nv_magnetometer_project/data/lockin_multipoint/multipoint_lockin_stream_20260819_<HHMMSS>*`

| file | what is used |
|---|---|
| `..._vector_rows.csv` | `delta_Bx_uT`, `delta_By_uT`, `delta_Bz_uT` — the 3-axis reconstruction |
| `..._wide.csv` | `time_s` and the 16 normalised flank points (used only for the sample rate and the raw-point spectra) |
| `..._projection_rows.csv` | the 8 measurement axes and their sensitivities (used for the geometry check) |
| `..._.csv` | raw ADC per point plus `peak_NN_ref` (used for the live-reference stability check) |

| condition | runs |
|---|---|
| drone | 173633, 174657, 174908, 175409, 180601, 181031, 181300, 182456, 183014, 183316 |
| no drone | 174515, 175626, 181817 |

Each run is preceded by its own `parked_plan_odmr_sweep_20260819_*` calibration,
so **`ΔB` is the change since that calibration, not an absolute field.**

### Two-point lock-in, streaming — 4166.67 Hz, 125 000 samples, 30.0 s

`nv_magnetometer_project/data/twopoint_lockin/twopoint_lockin_stream_20260819_<HHMMSS>.csv`

| condition | runs |
|---|---|
| drone in front, present the whole run | 171802, 172711, 173311 |
| no drone | 172841, 173017 |
| unlabelled, same session | 171057, 171419, 171955, 173426 |

The unlabelled streams are carried through only to measure the session's
run-to-run scatter. They are drawn in grey and never counted as either class.

Conversion chain, read back out of `notebook_modules/twopoint_postprocess.py`:

```
z          = counts / ref_norm_counts          # ref_norm_counts is FIXED at calibration
lockin     = z_plus - z_minus
delta_f    = (lockin - zero) / denom           # zero, denom from the calibration JSON
B_shift_uT = delta_f / 0.028024                # gamma = 28.024 kHz/uT
```

The fixed `ref_norm_counts` is the source of the gain problem described in §6.

---

## 2. Filtering, and what the "shadow" is

* **Display filter:** 4th-order Butterworth, zero phase (`scipy.signal.filtfilt`),
  **low-pass 3 Hz**. The hand-driven motion was measured at 0.30–2.30 Hz across
  all runs, so 3 Hz keeps all of it while removing the ~61 Hz mains line and the
  white floor.
* **Detector filter:** the same, at **2 Hz** (ENBW 2.05 Hz).
* **Shadow** on every time-series panel = the raw 146.71 Hz samples at
  `alpha = 0.20`. The solid line is the filtered trace.
* **`|ΔB|` caveat:** the shadow is `‖X_raw‖` and the line is `‖X_filt‖`. A norm of
  a noisy vector has a positive pedestal of `σ√3` (≈ 20 µT here), so the shadow
  sits well above the line at baseline. This is arithmetic, not a signal.

## 3. Reconstruction geometry (sanity check, not a figure)

The 8 measurement axes from `projection_rows` give a well-conditioned inverse:

```
cond(A)                       = 1.019
singular values               = 1.651, 1.626, 1.621
sqrt(diag((AᵀA)⁻¹)) [x, y, z] = 0.614, 0.609, 0.615
```

All three axes therefore carry **equal** noise weight. Any observed
`σ_y > σ_x, σ_z` (seen in most runs) is a real anisotropic background, not a
reconstruction artefact.

## 3a. The ~60 Hz line, and a sample-clock error

**The line is the building mains, not the cryostat pump.** None of this data was
taken in the pump room. The identification is confirmed by a cross-check between
the two instruments:

| | nominal time base | wall-clock corrected |
|---|---|---|
| two-point, 9 streams | 61.723 ± 0.011 Hz | **60.079 ± 0.050 Hz** |
| 16-point, 7 streams | 61.019 ± 0.007 Hz | **60.133 ± 0.007 Hz** |

On the nominal time base the two instruments disagree by **0.70 Hz** about a
single physical line, which is impossible. The cause is that `time_s` in every
CSV is *computed* from the assumed rep time, not measured. Each file also carries
a per-packet wall-clock column, `timestamp_epoch_s`. Fitting

```
slope = d(timestamp_epoch_s) / d(time_s)      # over the packet boundaries
```

gives 1.0274 for the two-point stream and 1.0147 for the 16-point stream — the
assumed rate is too fast by **2.7 %** and **1.5 %** respectively. Rescaling each
stream by its own slope brings both lines onto **60.1 Hz** and into agreement
with each other to 0.05 Hz.

True rates are therefore ≈ 4056 Hz (nominal 4166.67) and ≈ 144.6 Hz
(nominal 146.71), with Nyquist ≈ 2028 Hz and ≈ 72.3 Hz.

**Frequencies are quoted on the nominal time base throughout the deck**, because
that is what every other deck and every CSV uses; the correction factor is in
`numbers.txt`. Nothing else in the analysis moves: amplitudes, field values, DC
steps, the ladder fit and the detector are not functions of the time base, and
the filter cut-offs shift by <3 %, well inside their own roll-off.

*This is worth re-checking against the 2026-08-18 four-point deck, which
attributed a 61.60 Hz line to the cryostat pump on the grounds that it was
48 FFT bins off mains. The same clock error would account for that offset.*

## 4. Window statistics (distance ladders)

* Hold windows are the nominal intervals from the run log: 0–5, 10–15, 20–25 s
  (or 0–10 / 20–30 s for 180601).
* The reference is the mean of the **drone-free windows between the holds**
  (6–9, 16–19, 26–30 s), on the 3 Hz filtered vector.
* Reported `|ΔB|` is the norm of (window mean − reference mean).
* **Empirical noise on a 5 s window mean**, per axis, from the 3 drone-free runs
  cut into 18 windows: **σ = 2.64 µT**, so the 3σ level on `|ΔB|` is **13.7 µT**.

Power-law fit: 8 points from the three in-front ladders, straight line in
log|ΔB| vs log r, standard error from the residual spread.

## 5. Detection statistic

```
S(t) = LP2Hz(X) − median(LP2Hz(X))            # slow field excursion, 3 axes
σ    = rms(BP10-30Hz(X)) · sqrt(2.05 / 20)    # own-run noise, carried to 2 Hz ENBW
z(t) = ‖S(t)/σ‖ / sqrt(3)
```

Why 10–30 Hz is the right noise reference: it contains **no drone signal**
(all hand motion is below 3 Hz) and **no mains line** (~61 Hz), and it scales
with exactly the same transducer gain as the signal. Every ratio in the deck is
therefore immune to the optical-gain drift of §6.

Threshold `z* = 3.83` = the **99.9th percentile of z pooled over the three
drone-free runs** (13 203 samples). The drone-free runs define the false-alarm
rate; nothing about the drone runs enters the threshold.

Two read-outs:

| statistic | drone (10 runs) | no drone (3 runs) |
|---|---|---|
| duty above `z*` | 0.1 – 42.1 % | 0.0 – 0.2 % |
| longest contiguous dwell | 0.04 – 4.78 s | 0.00 – 0.05 s |
| max z | 4.7 – 14.9 | 3.5 – 7.1 |

**Longest dwell is the sharper of the two**: every run in which the drone was
*parked* near the sensor gives ≥ 4.4 s, against ≤ 0.05 s for every drone-free
run — a 90× margin. Swept runs break the dwell because the field crosses zero
each half cycle, which is why the duty statistic is kept alongside it.

*Caveat, stated on the slide as well:* the band edges (0.2–3 Hz signal,
10–30 Hz reference) were chosen with these 13 runs in view, so the margins are
mildly optimistic. The threshold itself is not — it comes only from the
drone-free data.

## 6. The two-point gain problem

The apparent 2× suppression of the ~61 Hz mains line when the drone is present
**is not established as magnetic**. Five independent reasons:

1. `A(61.7 Hz)` is anticorrelated with the run's optical gain
   `z̄_run / z̄_calib` at **r = −0.98** over the five labelled runs; the gain
   itself swung 34 %.
2. Normalised to each run's own white floor (median ASD 300–800 Hz — the same
   gain scales both), the SNRs overlap: drone 11–42, no drone 36.
3. The four unlabelled streams from the same session span
   `A(61.7) = 0.92 – 1.60 µT`, straddling the drone group.
4. The suppression is uniform across `f₀`, `2f₀`, `3f₀` **and** the broadband
   floor. That is a gain signature; a magnetic effect would not touch the
   floor identically.
5. Flux shunting by the drone's motor steel is ~0.2 % at 10 cm
   (`3V/4πr³` with V ≈ 4 cm³, r = 10 cm) — three orders of magnitude too small.

**Root cause:** the two-point stream normalises against a *fixed*
`ref_norm_counts` written at calibration time, so any drift in PL or contrast
rescales every reported µT. Its `z̄` wandered 0.72 → 0.99 over the evening.
The 16-point stream carries a *live per-shot* reference (`peak_NN_ref`) and its
`z̄` held 0.927 – 0.964 across all 13 runs.

**Fix:** give the two-point stream the same live reference readout, then
interleave drone in/out on a 5 s cadence *inside a single run* so the gain
is common-mode.

**What does survive** the gain problem: the **sign** of the DC offset. A positive
gain factor cannot flip a sign, and 3/3 drone runs are positive (+5.8 to
+23.5 µT) against 6/6 non-drone negative (−0.7 to −4.6 µT). Under a
random-label null that arrangement has probability 1/C(9,3) = 1/84.

## 7. Physics conclusions

**The drone is detected as a static magnetic dipole.** Three independent proofs:

* Run 174657: the field steps and *holds* with no hand motion at all.
* The signal frequency is always the hand frequency (0.3–2.3 Hz), never a motor
  or ESC frequency.
* The two-point drone/no-drone ASD ratio is featureless from 0 to 2 kHz
  (median 1.21 over 5–2000 Hz, 1.23 over 200–2000 Hz where only gain can act).

The magnetometer is operating as a DC instrument; **the hand is the chopper**.
This is exactly the behaviour of a permanent magnet, and matches the suspicion
that prompted the analysis.

**Falloff:** `|ΔB| ∝ r^(−2.17 ± 0.24)`, flatter than the point-dipole −3, for two
reasons: (a) at 10–30 cm the sensor is not in the far field — the four motors are
~20 cm apart, comparable to r, so the array falls off more slowly; (b) the 20 and
30 cm points sit near the floor and are biased upward by the `|ΔB|` noise
pedestal. Fitted `|ΔB|(10 cm) = 17.3 µT` gives an *effective* moment
`m ≈ 0.17 A·m²`, valid near 10 cm only.

**Why "on top, swept L–R" beats "in front, bobbed U–D":** two measured factors,
multiplying to ≈ 3×.

1. *Geometry.* On the dipole axis `B = 2k/r³`; on the equator `B = k/r³`.
   Measured static step at ~10 cm: on top 16.5, 24.2 µT; in front 11.8, 16.5,
   21.0 µT → ratio 1.24. Below the ideal 2, but inside the ±2 cm hand-held
   distance error, which alone is ±60 % in field at 10 cm.
2. *Stroke.* The measured gradient at 10 cm is `3B/r = 5.2 µT/cm`, so the
   0.3–4 Hz band power converts directly to a hand travel: L–R on top delivered
   1.1–2.0 cm rms, up–down in front 0.6–0.9 cm rms — 1.5–2× more travel.

Since the modulation scales as `∂B/∂r ∝ r⁻⁴`, **stand-off, not placement, is the
dominant lever.** "On top" wins mostly because it lets you get closer and swing
further, not because the axis is special.

**Range today:** the static field crosses the 3σ (5 s) level of 13.7 µT at
≈ 13 cm. Range grows only as `(m/σ)^(1/3)`, so 1 m would need ~300× less noise.

## 8. Next experiments

1. Live reference on the two-point stream (§6).
2. Interleave drone in/out every 5 s *within* one stream — every systematic
   becomes common-mode and the control is the same run.
3. Fix the stand-off with a rail or tripod. The ±2 cm hand-held error dominates
   the `r^-2.2` exponent.
4. **Power the drone.** Every run tonight found permanent-magnet field only.
   Spinning rotors put lines at the rotor rate (~80 Hz) and the ESC electrical
   rate (~600 Hz) — both inside the two-point Nyquist of 2083 Hz, neither inside
   the 16-point Nyquist of 73.4 Hz.
5. Raise the 16-point rate: τ 213 → 120 µs takes 146.7 Hz to 260 Hz and moves
   the mains line well inside band.
6. Calibrate `time_s` against the wall clock (§3a) — it is 1.5 % fast on the
   16-point stream and 2.7 % fast on the two-point stream.
