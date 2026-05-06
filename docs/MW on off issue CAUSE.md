Found it! Looking at your Section 4 output:


Sensitivity (reference): 70.73 nT/sqrt(Hz)  FWHM=17.47 MHz
Sensitivity (signal):    58.45 nT/sqrt(Hz)  FWHM=16.84 MHz
Both signal (MW-on) and reference (MW-off) are fitting Lorentzian dips with near-identical FWHM and centers. The reference is showing ~85% of the signal's slope (10918 vs 12898 ADC/GHz). This is the symptom you're describing.

This is not a code bug — it's a physics/timing issue with the pulse sequence.

Root cause: relax_delay_treg is too short
Let me verify the sequence from lockinodmr.py:157-205:


t=0                    : MW pulse fires (length = readout_integration_treg)
t=0 .. 213 µs          : Shot 1 — ADC + laser  (MW ON)      → "signal"
t=213 .. 214.6 µs      : laser ON, MW OFF, NO ADC  (relax)  ← THE GAP
t=214.6 .. 427.9 µs    : Shot 2 — ADC + laser  (MW OFF)     → "reference"
Your config:

readout_integration_tus = 213.33 µs (65535 treg)
relax_delay_treg = 500 → only ~1.6 µs gap between shots
During Shot 1 the MW drives the NV spin into ms=±1 (depleted PL). After MW turns off, the laser must re-pump the spin back to ms=0 before Shot 2, otherwise Shot 2 inherits the same depleted state.

NV optical repolarization takes ~300 ns to 3 µs depending on laser intensity. With only 1.6 µs — and likely a sub-saturating laser — the spin is still mostly in ms=±1 when Shot 2 begins, so the "MW-off" reference sees the same ODMR dip.

This also explains why d.contrast_percent is tiny in your runs — the common-mode dip cancels in signal - reference.

Fix: increase the relax delay
In both Section 3 and Section 5 cells:


ODMR_RELAX_DELAY_TREG = 500     # ← this is the problem
Try progressively:

3000 (~10 µs) — first attempt
10000 (~33 µs) — if still showing dip in reference
30000 (~100 µs) — should be fully polarized
You can also check laser power. If the 532 nm power at the NV is <100 µW focused, repolarization is slow. A stronger laser reduces the required relax_delay.

Impact on the lock-in implementation
Our twopoint_lockin.py and lockin_extensions.py inherit relax_delay_tus from default_config (~1.6 µs). The lock-in signal will be severely attenuated for exactly the same reason: both contrast_f1 and contrast_f2 are nearly zero because reference ≈ signal at each frequency.

Recommendation: before running any lock-in tests, fix the ODMR first by finding a relax_delay_treg value where:

d.reference is ~flat (no dip)
d.signal shows a clear dip
d.contrast shows the full dip structure
Then use that same relax_delay_treg in default_config so the two-point/sinewave/feedback lock-ins inherit it.

Would you like me to add a diagnostic cell that sweeps relax_delay_treg and shows at what value the reference flattens out?