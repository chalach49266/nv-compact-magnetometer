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

For the project defaults
(`readout_integration_tus = 213`, `relax_delay_treg = 500`,
`reps_per_batch = 1000`, 16 parked frequencies):

```text
time_per_rep    = 16 × (2 × 213 µs + 2 × relax_delay_µs)
                ≈ 16 × ~860 µs
                ≈ 13.8 ms

total_per_batch ≈ 1000 reps × 13.8 ms        ≈ 13.8 s of pulse work
program upload  ≈ 300 ms                     (one-time, before the rep loop)
```

Compared to the naive approach:

```text
naive_total ≈ 16 × (300 ms upload + 1000 reps × ~860 µs)
           ≈ 16 × (300 ms + 0.86 s)
           ≈ ~18.6 s, of which ~4.8 s is upload overhead
```

So the single-program approach saves ~5 s per shot and removes the
upload-latency variance. The remaining runtime scales linearly with reps and
frequency count.

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

## Source files referenced

- [`multipoint_lockin_program.py`](multipoint_lockin_program.py)
  &nbsp;— `MultipointLockinODMR` class
- [`Lockin_module.ipynb`](Lockin_module.ipynb)
  &nbsp;— the three workflow cells (`lockin_plan`, `lockin_acquire`, `lockin_reconstruct`)
- `../qickdawg/nvpulsing/lockinodmr.py`
  &nbsp;— the patched upstream class with `odmr_reference_offres_mhz` support, used by ODMR sweeps
- `../nv_toolkit/tui.py`
  &nbsp;— `_suggest_parked_frequencies` and `_compute_live_snapshot` helpers
