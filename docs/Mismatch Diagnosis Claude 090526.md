What the comparison CSV says
Mean offsets across all 16 parked points (live_vs_batch_vs_odmr_adc_20260508_151729.csv):

Comparison	Mean ΔADC (signal)	Mean ΔADC (reference)	Sign
batch − ODMR	−9.1	−9.1	always negative
live − ODMR	−16.2	−12.6	always negative
live − batch	−7.1	—	always negative
Two crucial facts:

The offset hits the reference column too, by almost exactly the same amount as the signal column. So this is not a frequency-dependent or NV-physics effect — it's a DC pedestal shift between the two programs.
The signal/reference ratio is preserved to ~1e-4 (e.g. peak_03: live 0.99899, batch 0.99885, ODMR 0.99909). So the dip contrast measurement is fine.
That decomposes the puzzle into two separate causes.

Cause 1: ODMR vs Multipoint — the ~9 ADC pedestal
I just compared the two body() methods. They are identical EXCEPT for one critical line.

In lockinodmr.py:186-197:


ref_offres_mhz = getattr(self.cfg, "odmr_reference_offres_mhz", None)
if ref_offres_mhz is not None:
    self.mw_frequency_register.set_to(float(ref_offres_mhz))
    self.pulse(ch=self.cfg.mw_channel, t=t_ref)   # ← only fires if offres is set
self.trigger(adcs=..., t=t_ref)
Your ODMR cell does not set odmr_reference_offres_mhz on config_odmr, so ref_offres_mhz is None → no MW pulse fires during the ODMR reference window. Just laser + ADC.

In multipoint_lockin_program.py:120-129 the corresponding section is unconditional:


self.mw_frequency_register.set_to(ref_offres_mhz)
self.pulse(ch=self.cfg.mw_channel, t=t_ref)        # ← always fires at full gain
self.trigger(adcs=..., t=t_ref)
So Multipoint always fires a full-gain MW pulse at 2700 MHz during the reference window.

Consequence — duty cycles differ:

Program	MW on (per cycle)	MW duty
LockinODMR	213 µs (signal only)	~50%
MultipointLockinODMR	213 µs (signal) + 213 µs (ref)	~100%
A near-100% MW duty cycle vs ~50% causes:

RF pickup on the photodiode preamp — when the MW chain is active, the preamp picks up a small DC-equivalent offset. Multipoint always has MW on, ODMR has it on half the time → pedestal differs.
Slight chamber/PCB heating that shifts the photodiode bias point, and possibly some weak power-broadened excitation if 2700 MHz isn't truly off-resonance under your bias field.
This is why the offset shows up identically in both signal and reference columns — both shots of Multipoint are "MW-on" shots from the preamp's point of view, while only one of ODMR's two shots is.

Cause 2: Live vs batch — the additional ~7 ADC
Live mode runs with pre_init=False. The pre_init block in multipoint_lockin_program.py:81-88 is the polarization pre-pulse. Without it, each acquire() jumps straight into the body — so the FIRST few reps of every batch start with NV in whatever residual state the previous batch left it. The laser repolarization is fast (~3 µs) so the bulk of the integration is fine, but the first ~few µs of integration each batch contributes to the average and pulls it down.

The notebook does prime once with pre_init=True before the loop, but acquire() re-runs initialize() every batch, so that priming only helps the very first acquire, not subsequent ones. Hence live sits ~7 ADC below batch consistently.

What this means physically
Your magnetometry is not broken. Δf and ΔB use signal/reference ratios, and those ratios match ODMR to ~1e-4 across all 16 points. The contrast you measure is the real contrast.
The absolute ADC mismatch is a program-asymmetry artifact, not bad NV physics, not bad parking, not bad timing.
The variation in the offset across peaks (it ranges from −2 to −30) is mostly statistical noise on the ODMR side — ODMR has 10 reps per freq with ~16 ADC stddev/freq, while live has hundreds of samples averaged, so the live reading is the more reliable absolute number.
Things to try (in order of expected impact, no edits yet)
Mirror ODMR's reference behavior in MultipointLockinODMR: skip the set_to + pulse in the reference window, just trigger(adcs=..., t=t_ref). This is the single change most likely to close the 9 ADC pedestal. It also matches the ODMR program byte-for-byte in its DC pickup behavior.
Diagnostic test: temporarily set MULTIPOINT_OFF_RESONANCE_MHZ = 3500 (clearly outside any NV transition under your bias field) and re-run batch. If the offset shrinks substantially, 2700 MHz was contaminated by tail driving. If the offset is unchanged, it's pure RF pickup from the duty cycle and step 1 will fix it.
For live mode, run with pre_init=True unless you can show the per-batch peaking transient still appears. The peaking story in the markdown was about the first rep of each batch being abnormally bright; with pre_init=False that's swapped for a small DC droop in the opposite direction. Both are batch-edge artifacts; you're choosing which one to live with.
Always work in normalized units (signal/reference or (signal − reference)/reference) for any cross-program comparison or for logging. The ratios are correct; the absolute ADC isn't directly comparable across program structures.
Want me to make change 1 first (drop the MW reference pulse in MultipointLockinODMR) and then run your diagnostic test 2?

Lockin_module.ipynb
for the 1. can you tell me where to edit without you editting
Edit multipoint_lockin_program.py:90-132 — the body() method.

Specifically, delete lines 120-122 (the three lines that fire MW during the reference window):


# Set MW to off-resonance (reference pulse — no NV excitation).
self.mw_frequency_register.set_to(ref_offres_mhz)
self.pulse(ch=self.cfg.mw_channel, t=t_ref)
Leave everything else in body() untouched. The self.trigger(adcs=..., t=t_ref) block immediately below those lines (lines 123-129) stays — that's the laser+ADC capture, which is exactly what you want for a true MW-off reference.

After deletion, the second half of the per-iteration block should look like:


t_ref = self.cfg.readout_integration_treg + self.cfg.relax_delay_treg

self.trigger(
    adcs=self.cfg.adcs,
    pins=[self.cfg.laser_gate_pmod],
    width=self.cfg.readout_integration_treg,
    adc_trig_offset=0,
    t=t_ref,
)

self.wait_all()
self.sync_all(self.cfg.relax_delay_treg)
Things you do not need to change:

ref_offres_mhz = float(self.cfg.odmr_reference_offres_mhz) on line 96 can stay (it'll just become an unused local) or you can delete it for cleanliness.
required_cfg on line 40-52 — odmr_reference_offres_mhz can stay listed; it's harmless. Or remove it if you want it strictly clean.
Nothing changes in initialize(), acquire(), analyze_results(), time_per_rep(), or total_time().
Nothing changes in Lockin_module.ipynb — you can leave MULTIPOINT_OFF_RESONANCE_MHZ = 2700.0 as-is; with the pulse removed it just becomes a no-op.
One thing to be aware of after the change:

time_per_rep() on line 177-180 currently estimates time as 2 * readout_integration + 2 * relax_delay per frequency — that includes the (now-deleted) reference MW pulse window. The actual hardware timing won't change because the trigger window length is unchanged, so the existing estimate is still correct. Don't touch it.

After the edit, restart the kernel (so importlib.reload(multipoint_lockin_program) picks up the new bytecode), re-run the multipoint cell, and compare against ODMR — you should see the ~9 ADC batch-vs-ODMR pedestal close substantially.