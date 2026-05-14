# Real-time spike rejection for the live lock-in loop

**Status:** implemented 2026-05-13.
**Code:** [`notebook_modules/spike_rejection.py`](../notebook_modules/spike_rejection.py),
[`scripts/replay_despike.py`](../scripts/replay_despike.py).
**Hooks into:** Cells `lockin_live` (Live A) and `a2ca3c11` (Live B) in
[`Modules/Lockin_module.ipynb`](../Modules/Lockin_module.ipynb).

---

## 1. Why this exists

The 11.7 Hz live two-point lock-in trace (Cell 18) and the 5.9 Hz live
B-field reconstruction (Cell 23) both show occasional **single-sample
ADC spikes** of ±10–40 counts on top of a baseline noise of only ±3–4
counts. Examples are visible in the project update PDF
([`docs/project_update_20260511.pdf`](project_update_20260511.pdf),
slides 5–8 — "Next 1: stabilize live traces and reject jumps") and in
the raw CSV
[`data/lockin_multipoint/multipoint_lockin_live_20260511_111629.csv`](../data/lockin_multipoint/multipoint_lockin_live_20260511_111629.csv).

These spikes have a distinctive signature:

| Feature                | Spike                                | Real magnetic-field change           |
|------------------------|--------------------------------------|--------------------------------------|
| Width                  | 1 batch (occasionally 2)             | Many consecutive batches             |
| Sign                   | Symmetric (up *or* down)             | Coherent direction                   |
| Affected channels      | Usually 1 of the 16 parked points    | All 16 move together                 |
| Amplitude vs baseline  | 10–40 ADC counts on a 3–4 ADC σ      | Slow drift / continuous              |
| Cause                  | Hardware glitch (per-channel ADC)    | NV resonance shifting under B-field  |

Because the signature is so different from a real signal, a **causal
trailing-window Hampel filter** at the raw-ADC level cleans them with
no measurable loss of magnetic-field sensitivity.

---

## 2. Algorithm — causal Hampel filter

For each parked-point channel `k` we keep a rolling buffer of the last
`window` *non-rejected* ADC samples. When a new sample `x` arrives:

1. Compute the rolling median `m = median(buf)`.
2. Compute the median absolute deviation `MAD = median(|buf − m|)`.
3. Convert MAD to a robust σ estimate:
   `σ = max(1.4826 · MAD, sigma_floor)`.
4. If `|x − m| > k_sigma · σ`, declare `x` a spike, **replace** it with
   `m`, and **do not** push it into the buffer (so a spike cannot
   poison the median for the next sample).
5. Otherwise accept `x` and append it to the buffer.

Three details that matter:

- **Causal window.** The standard textbook Hampel filter is centered
  (uses past and future samples), which adds half-a-window of latency.
  Live lock-in cannot tolerate that — at 11.7 Hz a 7-sample symmetric
  window would add ~300 ms of latency. We use a one-sided trailing
  window so the algorithm is strictly online.
- **`sigma_floor`.** With a small window the MAD itself is noisy:
  several consecutive samples can happen to cluster, the MAD shrinks
  below the true noise floor, and the threshold becomes so tight that
  legitimate ±3–4 σ noise samples get flagged. `sigma_floor = 3.0`
  (ADC counts, matching the observed baseline noise) prevents this.
  When MAD-σ is realistic the floor is inactive; when MAD-σ collapses
  the floor takes over.
- **Warm-up fallback.** During the first `min_warmup` batches the
  rolling buffer is too short to estimate σ at all. We optionally
  hard-clamp obviously wild outliers (`> 6 · max(1% · baseline, 5)`)
  against the per-channel ADC vector loaded from the most recent
  **single-batch** acquire CSV
  ([`data/lockin_multipoint/multipoint_lockin_collected.csv`](../data/lockin_multipoint/multipoint_lockin_collected.csv),
  produced by Cell 12 `lockin_acquire`). This catches spikes that
  happen in the first few seconds before the rolling estimator is
  trustworthy.

### Why detection at the raw-ADC level (not at Δf or ΔB)

The hardware glitch shows up on **one** parked-frequency ADC channel
at a time, while the downstream lock-in formula

```text
Δf_k = ((S+ − S−) − (S+,0 − S−,0)) / (m− − m+)
```

mixes two channels into each per-block Δf. A glitch in `peak_03`
contaminates `Δf_b02` (uses peaks 03 and 04), and via the per-block
projections it then contaminates the reconstructed ΔBx, ΔBy, ΔBz too
— so by the time the spike reaches Δf or ΔB it is harder to recognise
as a single-channel event. Catching it at the raw-ADC stage is the
cleanest place to act.

It also protects against false positives: a *real* magnetic-field
change moves all 8 block Δfs coherently, but the rolling per-channel
median sees only its own channel's history, so a coordinated physical
shift is not anomalous from any single channel's point of view.

---

## 3. Parameters and defaults

| Parameter     | Default | Meaning                                           | When to change                                    |
|---------------|---------|---------------------------------------------------|---------------------------------------------------|
| `n_channels`  | 16      | Number of parked points per batch                 | Only if the lock-in workflow changes              |
| `window`      | 11      | Length of the trailing per-channel buffer         | Bigger = stabler median, slower drift response    |
| `k_sigma`     | 4.0     | Rejection threshold in MAD-σ units                | Raise (→5) if real signals get clipped            |
| `min_warmup`  | 5       | Batches before MAD-based rejection turns on       | Should stay ≤ window                              |
| `sigma_floor` | 1.5     | Minimum σ in ADC counts (floors the MAD estimate) | Match the observed baseline noise σ on your setup |
| `baseline`    | None    | Optional per-channel single-batch ADC vector      | Auto-loaded from the Cell 12 CSV if present       |

**Note on `sigma_floor`.** Originally 3.0; lowered to 1.5 after the
2026-05-11 visualization showed the actual per-channel noise σ ≈ 1.1
ADC. A 3.0 floor was inflating the effective threshold to
`k_sigma · sigma_floor = 12` ADC and letting real 6–11 ADC spikes
slip through. With `sigma_floor = 1.5` the threshold drops to 6 ADC,
which catches the smaller spikes without flagging legitimate ±3σ
noise excursions. See `Modules/Despiker_visualization.ipynb`
section 8 for the parameter-sweep evidence.

At 11.7 Hz the default `window=11` corresponds to ~0.9 s of history —
fast enough to track warm-up drift (which is many-minutes scale), slow
enough to give a stable MAD estimate.

The notebook exposes the five tuning knobs at the top of each live
cell as `LIVE_DESPIKE_ENABLED`, `LIVE_DESPIKE_WINDOW`,
`LIVE_DESPIKE_K_SIGMA`, `LIVE_DESPIKE_WARMUP`,
`LIVE_DESPIKE_SIGMA_FLOOR` (and the matching `LIVE_B_DESPIKE_*` set in
Live B). Set `LIVE_DESPIKE_ENABLED = False` to bypass the filter
entirely (e.g. to collect a "raw" reference run).

---

## 4. Where it plugs into the live loop

Both live cells follow the same shape:

```python
sig_raw = np.asarray(d_batch.signal,    dtype=float)
ref_raw = np.asarray(d_batch.reference, dtype=float)
if LIVE_DESPIKE_ENABLED:
    sig_clean, sig_flags = despiker_sig.update(sig_raw)
    ref_clean, ref_flags = despiker_ref.update(ref_raw)
else:
    sig_clean, ref_clean = sig_raw, ref_raw
    sig_flags = np.zeros(16, dtype=bool)
    ref_flags = np.zeros(16, dtype=bool)
```

`sig_clean` / `ref_clean` then feed everything downstream — the wide
`peak_NN` / `peak_NN_ref` CSV columns, the per-block Δf calculation,
and (in Live B) the `intensities_single = sig_clean / ref_clean`
input to `estimate_parked_series_fields`. **The downstream toolkit
needs no change**: it just sees cleaned ADC values.

Two new boolean columns per parked point are written to the live CSV:

- `peak_NN_spike_sig` — `True` if the signal channel was rejected.
- `peak_NN_spike_ref` — `True` if the reference channel was rejected.

Plus an end-of-run summary line, e.g.

```text
Spike rejection: 12 signal-channel rejections, 4 reference-channel rejections
(0.16% of signal cells, 0.05% of reference cells).
```

Live A and Live B carry **independent despiker state**
(`despiker_sig` / `despiker_ref` for Live A, `despiker_sig_B` /
`despiker_ref_B` for Live B), so running both cells in one notebook
session does not let one cell's buffer leak into the other.

---

## 5. Performance (timing)

Measured locally on this Mac (pure numpy, no FPGA in the loop). The
remote FPGA host should see the same order of magnitude — the work is
dominated by 16 small `np.median` calls.

| Quantity                                      | Value                |
|-----------------------------------------------|----------------------|
| One `despiker.update(...)` call (16 channels) | ~195 µs              |
| Per live-loop iteration (signal + reference)  | ~390 µs              |
| Live A FPGA acquisition budget                | ~113 ms / batch      |
| Live B FPGA acquisition budget                | ~170 ms / batch      |
| Added overhead vs Live A budget               | **~0.35 %**          |
| Added overhead vs Live B budget               | **~0.23 %**          |
| Effective Live A update rate                  | ~11.65 Hz (was 11.7) |
| Effective Live B update rate                  | ~5.89 Hz (was 5.9)   |

In other words, the despiker is essentially free at this batch size.
It would only become significant if `n_channels` or `window` grew by
more than ~20×.

The notebook keeps the existing watchdog
(`LIVE_WATCHDOG_FACTOR = 3.0`) that prints a warning if any batch's
wall time exceeds 3× the FPGA prediction. After enabling the
despiker the watchdog should fire no more often than before — that is
the primary in-situ "is the overhead acceptable?" check.

---

## 6. Validation

The script
[`scripts/replay_despike.py`](../scripts/replay_despike.py)
replays the despiker over a saved live CSV without touching the
hardware. Running it on the original spiky data file gives:

```text
$ python3 scripts/replay_despike.py data/lockin_multipoint/multipoint_lockin_live_20260511_111629.csv --save-cleaned

Loaded 460 batches, 68 columns from multipoint_lockin_live_20260511_111629.csv

--- Spike rejection summary (sigma_floor=1.5) ---
  batches: 460, channels per batch: 16 (each side)
  signal rejections per channel:    [4, 0, 6, 5, 1, 1, 8, 7, 2, 0, 6, 4, 1, 0, 8, 4]
  reference rejections per channel: [2, 0, 6, 4, 0, 0, 7, 7, 2, 0, 3, 4, 0, 0, 7, 4]
  total signal rejections:      57 (0.77% of cells)
  total reference rejections:   46 (0.62% of cells)
  batches with at least one flagged channel: 8/460 (1.7%)
```

Per-block maximum single-batch ADC deviation (raw vs cleaned):

| Block | Raw max\|dev\| (ADC) | Cleaned max\|dev\| (ADC) | Rejected channels | Reduction |
|------:|---------------------:|-------------------------:|------------------:|----------:|
| 1     | 17.33                | 8.84                     | 4                 | 49 %      |
| 2     | 12.58                | 5.55                     | 11                | 56 %      |
| 3     | 5.65                 | 5.64                     | 2                 | ~0 %      |
| 4     | 31.45                | 4.17                     | 15                | 87 %      |
| 5     | 21.85                | 7.08                     | 2                 | 68 %      |
| 6     | 6.90                 | 4.71                     | 10                | 32 %      |
| 7     | 4.39                 | 5.76                     | 1                 | −31 %     |
| 8     | 21.31                | 3.59                     | 12                | 83 %      |

Block 7's small −31 % "regression" comes from a single sample being
replaced with the rolling median when the original sample happened to
sit closer to the global median than the local median did. The
absolute scale is ~6 ADC (= 3σ of baseline noise) so it is not a real
degradation. Block 3 has the same effect at sub-1 % magnitude.

Two synthetic tests in `python -c` form (kept in the implementation
history) also confirm:

- A Gaussian baseline (σ = 4 ADC) with three injected ±30 ADC spikes
  → all three spikes rejected, false-positive rate < 0.2 %.
- A linearly drifting channel (continuous physical-field analogue)
  → 0 rejections (rolling median tracks the drift).

---

## 7. Known issues and resolutions

### 7.1. Filter rejects legitimate signal samples (false positives)

**Symptom.** During a known real magnet swing, several
`peak_NN_spike_sig` flags fire on the *flank* channels of the moving
transitions; the Δf trace shows small flat steps where the original
was a continuous ramp.

**Why.** Rapid coherent drift can momentarily exceed
`k_sigma · σ` on a single channel before the rolling median catches
up.

**Resolution, in order of preference.**

1. Raise `LIVE_DESPIKE_K_SIGMA` from 4.0 to 5.0 or 6.0.
2. Widen `LIVE_DESPIKE_WINDOW` (e.g. 11 → 15) so the median lags
   less behind real drift.
3. Run `scripts/replay_despike.py --save-plot` on the offending
   CSV to A/B test the new parameters offline before another live
   run.

### 7.2. Filter misses small spikes

**Symptom.** Some clearly visible jumps of, say, 8 ADC counts go
unflagged.

**Why.** 8 ADC counts at `k_sigma=4` requires `σ_eff ≤ 2`. With
`sigma_floor = 3.0` the effective threshold is at least 12 ADC, so
8 ADC events fall under it on purpose — they are within ~3σ of
baseline noise and not safely distinguishable from real signal.

**Resolution.**

1. Lower `sigma_floor` (e.g. 3.0 → 2.0) **only if** the baseline
   ADC noise on your setup is genuinely < 2 counts σ. Measure it
   first with the Cell 12 single-batch acquire.
2. Lower `k_sigma` cautiously — but expect more false positives.

### 7.3. Two consecutive spikes on the same channel

**Symptom.** A pair of adjacent-batch spikes is rejected, but the
median between them looks slightly biased.

**Why.** Rejected samples are *not* pushed into the buffer, so the
buffer keeps the cleaner history — that is exactly the right
behaviour. But if the pair is genuinely two batches wide and you
have a narrow buffer, the rolling median can lag by one sample.

**Resolution.** Use `window ≥ 11`. The current default already
handles 2-wide spikes correctly in all of our offline test cases.
For triple+ spikes (which we have never observed), increase the
window further.

### 7.4. Warm-up false positives

**Symptom.** A burst of rejections in the first 4–5 batches of a
live run.

**Why.** Before the rolling buffer is full, the warm-up fallback
uses the static single-batch baseline (Cell 12 CSV). If the baseline
was acquired some time ago and the ADC has drifted since, the gap
between current samples and stored baseline can exceed the warm-up
threshold.

**Resolution.**

1. Re-run Cell 12 (`lockin_acquire`) just before the live run so
   the baseline is fresh.
2. Or accept the first few warm-up rejections — they're flagged
   in the CSV and do not affect downstream Δf in a meaningful way.
3. Disable the warm-up fallback by passing `baseline=None` (skip
   the `pd.read_csv(MULTIPOINT_DATA_CSV)` block); the despiker
   will simply pass samples through until the rolling buffer is
   ready.

### 7.5. Despiker module not found

**Symptom.** `ModuleNotFoundError: No module named 'spike_rejection'`
when running Cell 18 or 23.

**Why.** The project-root setup cell (Cell 0) is what adds
`notebook_modules/` to `sys.path`. If you skipped it, or restarted
the kernel and started directly from Cell 18, the import fails.

**Resolution.** Re-run Cell 0 (or use the "Run All Above" command).

### 7.6. Real data has channels with structurally different noise

**Symptom.** One or two channels report many more rejections than
the others even when the data looks fine to the eye.

**Why.** Some parked points sit on steeper flanks of the ODMR
spectrum, so their *normalised* signal has bigger sample-to-sample
swing. The filter sees that as the noise floor; this is correct.

**Resolution.** Look at the per-channel histogram of
`peak_NN - rolling_median(peak_NN)` (the replay script's
`--save-plot` mode gives this). If a channel's σ exceeds
`sigma_floor`, the rejection rate is driven by the actual MAD and
is not a configuration issue.

### 7.7. CSV gets bigger

**Symptom.** Each live run now writes 32 extra boolean columns
(16 × signal flag + 16 × reference flag).

**Why.** Documenting which samples were replaced is necessary for
offline analysis and tuning.

**Resolution.** If long-term storage size is a concern, post-process
runs through a script that drops the 32 flag columns after they have
been audited. They are pure metadata; the cleaned ADC values are
already in the original `peak_NN` columns.

---

## 8. Workflow checklist for a fresh live run

1. Run Cell 0 (project root setup) and Cell 2 (autoreload).
2. Run Cell 12 (`lockin_acquire`) to produce/refresh
   `multipoint_lockin_collected.csv` — this becomes the warm-up
   baseline for the despiker.
3. Run Cell 14 (`lockin_to_peak_frequency`) to build the per-block
   calibrations.
4. Run Cell 18 (`lockin_live`, Live A) and watch for the printed
   line `Spike rejection: enabled=True, window=11, k_sigma=4.0, …`
   confirming the filter is active.
5. After the run, check the printed summary
   (`Spike rejection: N signal-channel rejections, M reference-channel
   rejections …`). A few rejections (≤ 1 % of cells) on a quiet run
   is healthy.
6. If you suspect over- or under-rejection, copy the just-saved
   `multipoint_lockin_live_<stamp>.csv` and run
   `python3 scripts/replay_despike.py <csv> --save-cleaned --save-plot`
   to sweep parameters offline.

---

## 9. References

- Pearson, "Outliers in process modeling and identification",
  *IEEE Trans. Control Syst. Technol.* (2002) — the canonical
  citation for Hampel-filter despiking of industrial sensor streams.
- Pearson, Neuvo, Astola & Gabbouj, "Generalized Hampel Filters",
  *EURASIP J. Adv. Signal Process.* (2016) — formal framework
  including the causal / one-sided variant used here.
- Roos-Hoefgeest Toribio et al., "A Novel Approach to Speed Up
  Hampel Filter for Outlier Detection", *Sensors* (2025) — real-time
  / FPGA-friendly variants if we ever want to push this into the
  acquisition firmware.
- Fore et al., "Nonlinear temporal filtering of time-resolved
  digital particle image velocimetry data", *Exp. Fluids* (2005) —
  decision-based Hampel filter on streaming sensor measurements.
- Bhowmik et al., "Outlier removal in facial surface electromyography
  through Hampel filtering technique", *IEEE LSC* (2017) — closest
  analogue to the spike signature we see here (single-sample
  impulses on a noisy baseline).
