"""Single-program multi-frequency lock-in for NV magnetometry.

Standard LockinODMR uses one QickSweep over a register, so a single program
can only measure evenly-spaced frequencies. The 16-point parked-frequency
plan from `_suggest_parked_frequencies` is non-uniform, so naively running
it with one LockinODMR per frequency triggers 16 FPGA program uploads
(~300 ms each → ~5 s of overhead per acquisition).

This class works around that by writing all 16 frequencies into one program's
body() method. Each body iteration uses `mw_frequency_register.set_to(f_mhz)`
to switch the MW frequency at runtime, then pulses + reads the ADC. The
reference shot fires MW at an off-resonance frequency (default 2650 MHz).

Result: ONE FPGA program upload, then `reps` averaged measurements at all
N frequencies in tight hardware sequence — same speed as a single ODMR sweep.
"""

from __future__ import annotations

import numpy as np
from itemattribute import ItemAttribute

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram


class MultipointLockinODMR(NVAveragerProgram):
    """Lock-in ODMR at a list of explicit (non-uniform) frequencies.

    Required cfg attributes:
        multipoint_freqs_mhz : sequence of float
            The N MW frequencies (MHz) to park at. One signal+reference
            readout pair per frequency, per rep.
        odmr_reference_offres_mhz : float
            Off-resonance frequency (MHz) for the reference shot.
        Other LockinODMR cfg attributes (mw_channel, adc_channel, mw_gain,
        readout_integration_treg, relax_delay_treg, laser_gate_pmod,
        mw_nqz, pre_init, reps).

    Optional cfg attribute:
        multipoint_skip_reference : bool, default False
            Drop the per-frequency reference readout, halving the readouts per rep
            (and so roughly halving the FPGA time). Use when the MW cannot actually
            be gated off, which makes the "MW-off" reference a contaminated copy of
            the signal rather than a clean normaliser. With it set, `.reference`,
            `.contrast` and `.contrast_percent` come back None and the caller must
            normalise against a constant instead. Defaults to False so existing
            callers (Lockin_module) are unaffected.
    """

    required_cfg = [
        "readout_integration_treg",
        "adc_channel",
        "laser_gate_pmod",
        "relax_delay_treg",
        "mw_channel",
        "mw_nqz",
        "mw_gain",
        "pre_init",
        "reps",
        "multipoint_freqs_mhz",
        "odmr_reference_offres_mhz",
    ]

    def initialize(self):
        self.check_cfg()

        if self.cfg.mw_gain > 30000:
            assert 0, "Microwave gain exceeds maximum value"

        self.setup_readout()
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        first_freq_mhz = float(self.cfg.multipoint_freqs_mhz[0])
        self.default_pulse_registers(
            ch=self.cfg.mw_channel,
            style="const",
            freq=self.freq2reg(first_freq_mhz, gen_ch=self.cfg.mw_channel),
            gain=self.cfg.mw_gain,
            length=self.cfg.readout_integration_treg,
            phase=0,
        )
        self.set_pulse_registers(ch=self.cfg.mw_channel)

        self.mw_frequency_register = self.get_gen_reg(self.cfg.mw_channel, "freq")

        self.synci(100)
        if (self.cfg.ddr4 is True) or (self.cfg.mr is True):
            self.trigger(ddr4=self.cfg.ddr4, mr=self.cfg.mr, adc_trig_offset=0)
        self.synci(100)

        if self.cfg.pre_init:
            self.pulse(ch=self.cfg.mw_channel)
            self.trigger(
                pins=[self.cfg.laser_gate_pmod],
                width=self.cfg.readout_integration_treg,
                adc_trig_offset=0,
            )
            self.sync_all(self.cfg.readout_integration_treg + self.cfg.relax_delay_treg)

    def body(self):
        """Generate ASM that loops through all N frequencies in one body call.

        Each frequency contributes one signal+reference readout pair, or just the
        signal readout when cfg.multipoint_skip_reference is set.
        Total readouts per body = (1 or 2) * N.
        """
        ref_offres_mhz = float(self.cfg.odmr_reference_offres_mhz)
        freqs_mhz = list(self.cfg.multipoint_freqs_mhz)
        skip_reference = bool(getattr(self.cfg, "multipoint_skip_reference", False))

        for freq_mhz in freqs_mhz:
            # Set MW to this parked frequency (signal pulse).
            self.mw_frequency_register.set_to(float(freq_mhz))
            self.pulse(ch=self.cfg.mw_channel, t=0)

            self.trigger_no_off(
                adcs=self.cfg.adcs,
                pins=[self.cfg.laser_gate_pmod],
                width=self.cfg.readout_integration_treg,
                adc_trig_offset=0,
                t=0,
            )
            if not skip_reference:
                self.trigger_no_off(
                    pins=[self.cfg.laser_gate_pmod],
                    width=self.cfg.relax_delay_treg,
                    adc_trig_offset=0,
                    t=self.cfg.readout_integration_treg,
                )

            if skip_reference:
                # No reference readout. The relax window becomes the closing trigger
                # (plain `trigger`, not `trigger_no_off`) so the laser gate is still
                # driven low at the end of this frequency's block.
                self.trigger(
                    pins=[self.cfg.laser_gate_pmod],
                    width=self.cfg.relax_delay_treg,
                    adc_trig_offset=0,
                    t=self.cfg.readout_integration_treg,
                )
            else:
                t_ref = self.cfg.readout_integration_treg + self.cfg.relax_delay_treg

                # Set MW to off-resonance (reference pulse — no NV excitation).
                # self.mw_frequency_register.set_to(ref_offres_mhz) # remove this line may fix the mismatch
                # self.pulse(ch=self.cfg.mw_channel, t=t_ref) # remove this line may fix the mismatch
                self.trigger(
                    adcs=self.cfg.adcs,
                    pins=[self.cfg.laser_gate_pmod],
                    width=self.cfg.readout_integration_treg,
                    adc_trig_offset=0,
                    t=t_ref,
                )

            self.wait_all()
            self.sync_all(self.cfg.relax_delay_treg)

    def acquire(self, raw_data=False, per_rep=False, *args, **kwargs):
        n_freqs = len(self.cfg.multipoint_freqs_mhz)
        per_freq = 1 if getattr(self.cfg, "multipoint_skip_reference", False) else 2
        data = super().acquire(readouts_per_experiment=per_freq * n_freqs, *args, **kwargs)

        if raw_data is False:
            data = self.analyze_results(data, per_rep=per_rep)

        return data

    def analyze_results(self, data, per_rep=False):
        """Reshape and split into per-frequency signal/reference arrays.

        per_rep=False (default) averages over the rep axis and returns one sample:
            .signal / .reference : (N,) arrays over the N parked frequencies

        per_rep=True keeps the rep axis, so every rep becomes an independent TIME
        sample spaced by time_per_rep():
            .signal / .reference : (reps, N)

        The rep axis is already there in the raw buffer -- loop_dims = [reps, ...] --
        so averaging it away throws away time resolution that cost nothing to acquire.
        Keeping it lets one acquire() call return many samples, which is the only way
        to beat the fixed per-call host overhead (~10 ms of Pyro RPC: config_all,
        reload_mem, start_readout and the poll_data loop) that otherwise sets the rate.

        Returns ItemAttribute with .frequencies_mhz, .signal, .reference, .contrast,
        .contrast_percent and .per_rep.
        """
        n_freqs = len(self.cfg.multipoint_freqs_mhz)
        skip_reference = bool(getattr(self.cfg, "multipoint_skip_reference", False))
        # data_shape = (reps, nsweep, (1 or 2)*n_freqs); nsweep is 1 here
        data = np.asarray(data, dtype=float).reshape(self.data_shape)
        data = data / self.cfg.readout_integration_treg

        if skip_reference:
            # readouts axis: [sig0, sig1, ..., sigN-1]
            signal_all, reference_all = data, None
        else:
            # readouts axis: [sig0, ref0, sig1, ref1, ..., sigN-1, refN-1]
            signal_all = data[..., 0::2]
            reference_all = data[..., 1::2]

        if per_rep:
            # Keep axis 0 (reps) and the trailing frequency axis; collapse anything
            # in between (a trivial sweep axis) so the result is always (reps, N).
            def _to_rep_freq(a):
                if a is None:
                    return None
                a = np.atleast_2d(a)
                while a.ndim > 2:
                    a = np.mean(a, axis=1)
                return a
            signal_all = _to_rep_freq(signal_all)
            reference_all = _to_rep_freq(reference_all)
        else:
            # Average over rep + sweep axes.
            n_reduce = len(signal_all.shape) - 1
            for _ in range(n_reduce):
                signal_all = np.mean(signal_all, axis=0)
                if reference_all is not None:
                    reference_all = np.mean(reference_all, axis=0)

        d = ItemAttribute()
        d.per_rep = bool(per_rep)
        d.frequencies_mhz = np.asarray(self.cfg.multipoint_freqs_mhz, dtype=float)
        d.signal = np.asarray(signal_all, dtype=float)
        if reference_all is None:
            # No reference was taken; the caller must normalise against a constant.
            d.reference = None
            d.contrast = None
            d.contrast_percent = None
        else:
            d.reference = np.asarray(reference_all, dtype=float)
            d.contrast = d.signal - d.reference
            with np.errstate(divide="ignore", invalid="ignore"):
                d.contrast_percent = np.where(d.reference != 0, d.contrast / d.reference * 100.0, np.nan)
        return d

    def time_per_rep(self):
        n_freqs = len(self.cfg.multipoint_freqs_mhz)
        n_readouts = 1 if getattr(self.cfg, "multipoint_skip_reference", False) else 2
        # Per frequency: n_readouts readout windows, plus the in-block relax window
        # and the closing sync_all(relax) -- two relax delays either way.
        t_us = n_readouts * self.cfg.readout_integration_tus + 2 * self.cfg.relax_delay_tus
        return n_freqs * t_us / 1e6

    def total_time(self):
        return self.time_per_rep() * self.cfg.reps
