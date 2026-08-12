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

import hashlib

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

    Optional cfg attributes:
        multipoint_skip_reference : bool, default False
            Drop the per-frequency reference readout, halving the readouts per rep
            (and so roughly halving the FPGA time). Use when the MW cannot actually
            be gated off, which makes the "MW-off" reference a contaminated copy of
            the signal rather than a clean normaliser. With it set, `.reference`,
            `.contrast` and `.contrast_percent` come back None and the caller must
            normalise against a constant instead. Defaults to False so existing
            callers (Lockin_module) are unaffected.

        multipoint_pair_order : {"forward", "abba"}, default "forward"
            Order in which the parked frequencies are visited inside one rep.

            "forward" visits them once each, in list order: A B. The two points are
            then measured one readout window apart, always in the same direction, so
            any drift in the common PL level between them enters their difference to
            first order -- the estimator is effectively differentiating the common
            mode. On 2026-08-06 the common mode swung 13% peak-to-peak during a
            heat-gun test, so this term is not academic.

            "abba" visits them out and back: A B B A. A linear drift contributes
            +d, +2d, +3d... across the four slots, and folding slot 0 with slot 3 and
            slot 1 with slot 2 cancels it exactly. This doubles the readouts per rep,
            so one "abba" rep costs the same as two "forward" reps and averages the
            same number of readouts -- the drift cancellation is free, paid for in
            rep count rather than in time.

        multipoint_on_stale : {"warn", "raise", "ignore"}, default "warn"
            What to do when an acquire returns a buffer bit-identical to the previous
            one. See `acquire`.
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

    def emission_order(self):
        """The frequencies as actually visited inside one rep, in order.

        "forward" gives [A, B]; "abba" gives [A, B, B, A]. Everything downstream
        (readout count, timing model, result folding) derives from this list, so
        adding another ordering only means adding a branch here.
        """
        freqs = [float(f) for f in self.cfg.multipoint_freqs_mhz]
        order = str(getattr(self.cfg, "multipoint_pair_order", "forward")).lower()
        if order in ("forward", "ab", ""):
            return freqs
        if order in ("abba", "out_and_back"):
            return freqs + freqs[::-1]
        raise ValueError(f"unknown multipoint_pair_order {order!r}; "
                         "expected 'forward' or 'abba'")

    def slot_to_frequency_index(self):
        """Which parked frequency each readout slot belongs to.

        For "forward" this is [0, 1]; for "abba" it is [0, 1, 1, 0]. `analyze_results`
        averages the slots that share a frequency, which is what performs the
        drift cancellation.
        """
        n = len(self.cfg.multipoint_freqs_mhz)
        order = str(getattr(self.cfg, "multipoint_pair_order", "forward")).lower()
        if order in ("abba", "out_and_back"):
            return list(range(n)) + list(range(n))[::-1]
        return list(range(n))

    def body(self):
        """Generate ASM that visits every emission slot in one body call.

        Each slot contributes one signal+reference readout pair, or just the signal
        readout when cfg.multipoint_skip_reference is set.
        Total readouts per body = (1 or 2) * len(emission_order()).
        """
        ref_offres_mhz = float(self.cfg.odmr_reference_offres_mhz)
        freqs_mhz = self.emission_order()
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
        """Run the program once and return the parked-frequency results.

        Also checks that the board actually gave us new data. `qick` reads the
        accumulated buffer over DMA while the tProc writes it; if a run is still in
        flight when the next one is configured, the readout can return the previous
        run's contents. On the 2026-08-06 burst data that happened on every second
        call -- 95.7% of the rows identical to the batch before, at 13 ms instead of
        425 ms -- and half of every burst run was silently a replay.

        The check is a byte comparison against the previous raw buffer, so it costs
        microseconds and catches the failure whatever its mechanism. `n_stale_acquires`
        counts hits; `cfg.multipoint_on_stale` chooses "warn" (default, reports the
        first few), "raise", or "ignore".
        """
        n_slots = len(self.emission_order())
        per_freq = 1 if getattr(self.cfg, "multipoint_skip_reference", False) else 2
        data = super().acquire(readouts_per_experiment=per_freq * n_slots, *args, **kwargs)

        self._check_freshness(data)

        if raw_data is False:
            data = self.analyze_results(data, per_rep=per_rep)

        return data

    # Set by _check_freshness; read it after a run to see whether any batch was a replay.
    n_stale_acquires = 0
    _previous_digest = None
    _stale_warnings_emitted = 0

    def reset_freshness_counters(self):
        """Zero the stale-acquire tally.

        Call this immediately before a timed run so that setup acquires -- priming,
        auto-zero -- do not contribute to the count the run is judged on.
        """
        self.n_stale_acquires = 0
        self._previous_digest = None
        self._stale_warnings_emitted = 0

    def _check_freshness(self, data):
        """Flag an acquire whose raw buffer repeats the previous one exactly."""
        policy = str(getattr(self.cfg, "multipoint_on_stale", "warn")).lower()
        if policy == "ignore":
            return

        digest = hashlib.blake2b(np.ascontiguousarray(data).tobytes(), digest_size=16).digest()
        if digest == self._previous_digest:
            self.n_stale_acquires += 1
            message = (
                f"acquire() returned a buffer identical to the previous call "
                f"(stale count {self.n_stale_acquires}). The accumulated buffer was read "
                f"before the new run refilled it. Pass reset_tproc=True, or use stream() "
                f"for high-rate acquisition."
            )
            if policy == "raise":
                raise RuntimeError(message)
            if self._stale_warnings_emitted < 3:
                self._stale_warnings_emitted += 1
                print(f"  WARNING: {message}")
                if self._stale_warnings_emitted == 3:
                    print("  (further stale-batch warnings suppressed; "
                          "check prog.n_stale_acquires at the end of the run)")
        self._previous_digest = digest

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
        # data_shape = (reps, nsweep, (1 or 2)*n_slots); nsweep is 1 here
        data = np.asarray(data, dtype=float).reshape(self.data_shape)
        data = data / self.cfg.readout_integration_treg

        if skip_reference:
            # readouts axis: [slot0, slot1, ..., slotM-1]
            signal_all, reference_all = data, None
        else:
            # readouts axis: [sig0, ref0, sig1, ref1, ..., sigM-1, refM-1]
            signal_all = data[..., 0::2]
            reference_all = data[..., 1::2]

        # Fold emission slots back onto frequencies. This is a no-op for "forward",
        # where slots and frequencies are the same thing. For "abba" it averages
        # slot 0 with slot 3 and slot 1 with slot 2, which is what cancels a linear
        # drift across the rep: the two visits to each frequency straddle the two
        # visits to the other one, so the drift contributions are equal and opposite.
        slot_map = np.asarray(self.slot_to_frequency_index())
        if len(slot_map) != n_freqs:
            def _fold(arr):
                if arr is None:
                    return None
                return np.stack([arr[..., slot_map == k].mean(axis=-1)
                                 for k in range(n_freqs)], axis=-1)
            signal_all = _fold(signal_all)
            reference_all = _fold(reference_all)

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

    def time_per_rep(self, include_relax=False):
        """Seconds of FPGA pulse work per rep.

        The rep is the ADC integration windows and essentially nothing else. This
        used to also add the in-block relax window and the closing sync_all(relax),
        two relax delays per frequency, but the measurement does not support it: on
        four 2026-08-06 burst runs the cadence is 423.8-425.4 us against 435.3 us
        for the relax-inclusive sum, and 2 x readout_integration_tus = 426 us for
        this one. The relax windows overlap the readout rather than following it.

        `include_relax=True` restores the old sum, which is a conservative upper
        bound if you want headroom rather than an estimate.
        """
        n_slots = len(self.emission_order())
        n_readouts = 1 if getattr(self.cfg, "multipoint_skip_reference", False) else 2
        t_us = n_readouts * self.cfg.readout_integration_tus
        if include_relax:
            t_us += 2 * self.cfg.relax_delay_tus
        return n_slots * t_us / 1e6

    def total_time(self):
        """Seconds of FPGA pulse work in one acquire (all reps)."""
        return self.time_per_rep() * self.cfg.reps

    # Host round-trip per acquire() call, measured on 2026-08-06: batch period
    # 11.37 ms against 9.77 ms of FPGA work at 23 reps. It is charged per CALL,
    # which is why it, not the FPGA, sets the ceiling for small rep counts.
    HOST_OVERHEAD_S = 1.6e-3

    def predicted_rate_hz(self, reps=None, host_overhead_s=None):
        """Samples per second this configuration can sustain in averaged mode.

        One acquire yields one averaged sample, so the rate is
        1 / (reps * time_per_rep + host_overhead). Use this before a run to check
        that the operating point does what you expect -- the readout window is the
        dominant term and it is easy to pick one that quietly costs half the rate.
        """
        reps = int(self.cfg.reps if reps is None else reps)
        overhead = self.HOST_OVERHEAD_S if host_overhead_s is None else host_overhead_s
        return 1.0 / (reps * self.time_per_rep() + overhead)

    def describe_timing(self, reps=None):
        """One-line human summary of the rate this configuration will deliver."""
        reps = int(self.cfg.reps if reps is None else reps)
        order = str(getattr(self.cfg, "multipoint_pair_order", "forward")).lower()
        n_slots = len(self.emission_order())
        return (
            f"{len(self.cfg.multipoint_freqs_mhz)} freqs, order={order} "
            f"({n_slots} slots/rep), tau={self.cfg.readout_integration_tus:.1f} us, "
            f"{reps} reps -> {self.time_per_rep() * 1e6:.0f} us/rep, "
            f"{reps * self.time_per_rep() * 1e3:.2f} ms FPGA + "
            f"{self.HOST_OVERHEAD_S * 1e3:.1f} ms host "
            f"-> {self.predicted_rate_hz(reps):.1f} Hz"
        )
