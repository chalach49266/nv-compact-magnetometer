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

Three ways to get data out, in increasing order of rate:

    acquire()                 one averaged sample per call. The call costs ~10-11.4 ms
                              of host round-trip **whatever the program does**, and the
                              FPGA work runs underneath it rather than adding to it, so
                              this caps out near 88 Hz and no readout-window change
                              moves it. The flip side is that averaging is free up to
                              that budget -- see `reps_that_fit()`.
    acquire(per_rep=True)     one call returns every rep as its own time sample
                              ("burst"), so the host cost is paid once per burst instead
                              of once per sample. Fast, but it re-arms the accumulated
                              buffer once per burst and roughly half the calls come back
                              as a bit-identical replay of the previous one. Forcing the
                              tProc stop does **not** fix it (tried 2026-08-14: still
                              50.0% replays). Set `cfg.multipoint_on_stale = "drop"` so
                              the guard discards them instead of recording duplicates.
    stream()                  one configuration, one tProc start, then nothing but
                              draining the streamer queue. No per-batch reconfiguration,
                              so no replay, no duty-cycle hole, and timestamps that
                              follow the FPGA cadence exactly. Delivered 1041.7 Hz with
                              zero dropped reps on 2026-08-14. **Open defect:** its
                              broadband noise floor measured ~6x the cleaned burst floor
                              at the same tau, flat across 10-500 Hz. Unresolved -- see
                              `stream()` and the interleaved A/B in
                              scripts/profile_twopoint_acquire.py.

See docs/2026-08-14_twopoint_methods/TWOPOINT_METHODS_AND_TIMING.md for the measurements
behind those statements, and docs/2026-08-06_twopoint_timing/ for the earlier analysis
whose timing conclusion this one corrects.
"""

from __future__ import annotations

import hashlib
from time import perf_counter as _perf_counter, sleep as _sleep

import numpy as np
from itemattribute import ItemAttribute

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram

try:
    # Present when the board is reached over rpyc; unwraps remote objects into
    # local ones. Pyro-only setups return plain data already.
    from rpyc.utils.classic import obtain
except ModuleNotFoundError:
    def obtain(x):
        return x


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

        multipoint_on_stale : {"warn", "drop", "raise", "ignore"}, default "warn"
            What to do when an acquire returns a buffer bit-identical to the previous
            one. "drop" retries and then returns None so the caller can skip the
            batch -- the right setting for burst mode, where ~50% of calls replay.
            See `acquire`.

        multipoint_stale_retries : int, default 2
            Re-acquire attempts before the "drop" policy gives up on a batch.
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

    def acquire(self, raw_data=False, per_rep=False, stale_retries=None,
                *args, **kwargs):
        """Run the program once and return the parked-frequency results.

        Also checks that the board actually gave us new data. `qick` reads the
        accumulated buffer over DMA while the tProc writes it; if a run is still in
        flight when the next one is configured, the readout can return the previous
        run's contents. On the 2026-08-06 burst data that happened on every second
        call -- 95.7% of the rows identical to the batch before, at 13 ms instead of
        425 ms -- and half of every burst run was silently a replay. **It still
        happens**: the 2026-08-14 run, taken with the corrected `config_all` and
        `reset_tproc=True`, replayed exactly 50.0% of its batches (185/371). So
        forcing the tProc stop is not the cure and the guard is not optional.

        The check is a byte comparison against the previous raw buffer, so it costs
        microseconds and catches the failure whatever its mechanism.
        `n_stale_acquires` counts hits; `cfg.multipoint_on_stale` chooses:

            "warn"   (default) report the first few and return the data anyway
            "drop"   re-acquire up to `stale_retries` times; return None if every
                     attempt is stale, so the caller can skip the batch instead of
                     writing a duplicate into the CSV
            "raise"  stop the run
            "ignore" no check at all

        `stale_retries` defaults to `cfg.multipoint_stale_retries` (2).

        Returns None only under the "drop" policy, when every attempt came back
        stale. Callers in a loop should treat that as "skip this batch".
        """
        n_slots = len(self.emission_order())
        per_freq = 1 if getattr(self.cfg, "multipoint_skip_reference", False) else 2
        readouts = per_freq * n_slots

        policy = str(getattr(self.cfg, "multipoint_on_stale", "warn")).lower()
        if stale_retries is None:
            stale_retries = int(getattr(self.cfg, "multipoint_stale_retries", 2))
        attempts = (int(stale_retries) + 1) if policy == "drop" else 1

        data = None
        for attempt in range(attempts):
            data = super().acquire(readouts_per_experiment=readouts, *args, **kwargs)
            if not self._check_freshness(data):
                break
            # Stale. Under "drop" we try again; the race is transient, so a repeat
            # call usually lands on a completed run.
            self.n_stale_dropped += 1
            if attempt == attempts - 1 and policy == "drop":
                return None

        if raw_data is False:
            data = self.analyze_results(data, per_rep=per_rep)

        return data

    # Set by _check_freshness; read them after a run to see how much was a replay.
    n_stale_acquires = 0     # buffers that repeated the previous one
    n_stale_dropped = 0      # of those, ones the "drop" policy discarded or retried
    _previous_digest = None
    _stale_warnings_emitted = 0

    def reset_freshness_counters(self):
        """Zero the stale-acquire tally.

        Call this immediately before a timed run so that setup acquires -- priming,
        auto-zero -- do not contribute to the count the run is judged on.
        """
        self.n_stale_acquires = 0
        self.n_stale_dropped = 0
        self._previous_digest = None
        self._stale_warnings_emitted = 0

    def _check_freshness(self, data):
        """Flag an acquire whose raw buffer repeats the previous one exactly.

        Returns True if this buffer is a replay of the previous one.
        """
        policy = str(getattr(self.cfg, "multipoint_on_stale", "warn")).lower()
        if policy == "ignore":
            return False

        digest = hashlib.blake2b(np.ascontiguousarray(data).tobytes(), digest_size=16).digest()
        stale = digest == self._previous_digest
        if stale:
            self.n_stale_acquires += 1
            message = (
                f"acquire() returned a buffer identical to the previous call "
                f"(stale count {self.n_stale_acquires}). The accumulated buffer was read "
                f"before the new run refilled it. reset_tproc=True does NOT fix this -- "
                f"it was tried on 2026-08-14 and 50% of batches still replayed. Set "
                f"cfg.multipoint_on_stale='drop' to discard them, or use stream()."
            )
            if policy == "raise":
                raise RuntimeError(message)
            if policy != "drop" and self._stale_warnings_emitted < 3:
                self._stale_warnings_emitted += 1
                print(f"  WARNING: {message}")
                if self._stale_warnings_emitted == 3:
                    print("  (further stale-batch warnings suppressed; "
                          "check prog.n_stale_acquires at the end of the run)")
        self._previous_digest = digest
        return stale

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

        signal_all = self._fold_slots(signal_all)
        reference_all = self._fold_slots(reference_all)

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

    def _fold_slots(self, arr):
        """Average the emission slots that share a parked frequency.

        A no-op for "forward", where slots and frequencies are the same thing. For
        "abba" it averages slot 0 with slot 3 and slot 1 with slot 2, which is what
        cancels a linear drift across the rep: the two visits to each frequency
        straddle the two visits to the other one, so the drift contributions are
        equal and opposite. Operates on the trailing axis, so it works for both the
        averaged and the per-rep shapes.
        """
        if arr is None:
            return None
        n_freqs = len(self.cfg.multipoint_freqs_mhz)
        slot_map = np.asarray(self.slot_to_frequency_index())
        if len(slot_map) == n_freqs:
            return arr
        return np.stack([arr[..., slot_map == k].mean(axis=-1)
                         for k in range(n_freqs)], axis=-1)

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

    # Per-CALL floor of acquire(), measured across three sessions. This used to be
    # HOST_OVERHEAD_S = 1.6e-3 under a serial `period = host + FPGA` model, which
    # the 2026-08-14 data disproves:
    #
    #   session   FPGA readout   fastest call   median period
    #   08-06        9.80 ms       10.2 ms        11.38 ms
    #   08-06        4.26 ms        9.4 ms        10.25 ms
    #   08-14        5.52 ms       10.3 ms        11.38 ms
    #   08-14        2.40 ms       10.1 ms        11.34 ms
    #
    # A 4x change in FPGA work moves the period by ~1 ms, and the fastest call is
    # ~10 ms even when only 2.40 ms of FPGA work was requested. So the host chain
    # is ~10 ms and the FPGA work runs underneath it -- free until it exceeds the
    # floor. Two consequences that reverse the old advice:
    #
    #   * shortening tau does not raise the averaged-mode rate, at any tau;
    #   * reps is FREE up to the floor, so under-filling it wastes sensitivity.
    #
    # Two numbers, because the host cost is a distribution and the two uses want
    # different ends of it:
    #
    #   HOST_CALL_S   the TYPICAL call. Predicts the rate. 11.4 ms reproduces the
    #                 measured 87.5 Hz at tau=213/23 reps and 87.8 Hz at
    #                 tau=120/23 reps -- the same answer for both, which is the
    #                 whole point.
    #   HOST_FLOOR_S  the FASTEST call observed. Sizes the free averaging: fill
    #                 past this and even the quickest calls become FPGA-bound.
    #
    # These are rig- and session-dependent -- the 2026-08-04/06 sessions ran a
    # ~10.25 ms typical call (94.9 Hz) against 11.4 ms on 2026-08-14. Calibrate
    # with measure_host_floor() at the start of a session rather than trusting
    # these defaults, which are simply the most recent measurement.
    HOST_CALL_S = 11.4e-3
    HOST_FLOOR_S = 10.1e-3

    # Kept as an alias so older notebooks that print it still import. It used to
    # be 1.6e-3 under the disproved serial model.
    HOST_OVERHEAD_S = HOST_CALL_S

    def predicted_rate_hz(self, reps=None, host_call_s=None):
        """Samples per second this configuration can sustain in averaged mode.

        `period = max(host_call, reps * time_per_rep)`. The two terms overlap
        rather than adding, so the rate is flat against `reps` until the FPGA work
        outgrows the host call and only then falls.
        """
        reps = int(self.cfg.reps if reps is None else reps)
        call = self.HOST_CALL_S if host_call_s is None else float(host_call_s)
        return 1.0 / max(call, reps * self.time_per_rep())

    def reps_that_fit(self, host_floor_s=None):
        """Largest rep count whose FPGA work still hides under the per-call floor.

        Averaging this many reps costs nothing in rate. Running fewer is pure
        loss: the call takes the same wall-clock time either way, so the unused
        milliseconds are photons not collected. On 2026-08-14 the runs used 10
        reps at tau = 120 us -- 2.40 ms of a ~10 ms budget, so ~4x the averaging
        was available for free.

        Sized against the *fastest* call rather than the typical one, so that
        filling the budget does not make the quick calls FPGA-bound and drag the
        rate down.
        """
        floor = self.HOST_FLOOR_S if host_floor_s is None else float(host_floor_s)
        per_rep = self.time_per_rep()
        return max(1, int(floor / per_rep)) if per_rep > 0 else 1

    @staticmethod
    def measure_host_floor(program=None, n=20, reps=1):
        """Measure the per-call floor on this rig, in seconds.

        Runs `n` acquires of a deliberately tiny program: with `reps=1` the FPGA
        work is a few hundred microseconds, so whatever the call still costs is
        the host chain. Takes about `n * floor` seconds -- a fifth of a second for
        the default.

        Returns a dict with the 5th percentile (the cleanest estimate of the
        floor, since jitter only ever adds), the median and the spread.
        """
        from copy import copy

        if program is None:
            raise ValueError("measure_host_floor needs a configured program to clone")

        cfg = copy(program.cfg)
        cfg.reps = int(reps)
        probe = type(program)(cfg)

        times = []
        probe.acquire(progress=False)          # discard the first, it warms caches
        for _ in range(int(n)):
            t0 = _perf_counter()
            probe.acquire(progress=False)
            times.append(_perf_counter() - t0)

        times = np.asarray(times, dtype=float)
        fpga = probe.time_per_rep() * cfg.reps
        return {
            "n": int(n),
            "reps": int(reps),
            "fpga_s": float(fpga),
            "floor_s": float(np.percentile(times, 5)),
            "median_s": float(np.median(times)),
            "p95_s": float(np.percentile(times, 95)),
            "jitter_s": float(np.percentile(times, 95) - np.percentile(times, 5)),
        }

    def stream(self, duration_s=None, total_reps=None, poll_timeout_s=10.0,
               progress=False):
        """Yield per-rep samples continuously from one uninterrupted FPGA run.

        This is the high-rate path, and the one to use for anything above a few
        hundred Hz. `acquire()` pays the host round-trip once per call, so calling
        it per sample caps the rate at ~1/(1.6 ms) however short the program is,
        and calling it per burst leaves a dead gap between bursts and re-arms the
        accumulated buffer under a possibly-still-running program -- the race that
        made burst mode record 50% duplicates.

        `stream()` configures once, starts the tProc once, and then only drains the
        streamer queue. There is no per-batch reconfiguration, so:

          * the race cannot occur -- nothing is re-armed mid-run;
          * there is no duty-cycle hole -- the FPGA runs continuously;
          * timestamps follow the FPGA cadence exactly, because the samples are
            consecutive reps of a single program. Use
            `t0 + (packet.first_rep + arange(packet.n_reps)) * time_per_rep()`,
            never wall-clock arrival times, which only measure when the host
            happened to drain the queue.

        Parameters
        ----------
        duration_s : float, optional
            Approximate run length. Converted to a rep count via `time_per_rep()`.
        total_reps : int, optional
            Exact rep count; overrides `duration_s`. One of the two is required.
        poll_timeout_s : float
            Give up if the board sends nothing for this long.
        progress : bool
            Print a line per packet.

        Yields
        ------
        ItemAttribute with:
            first_rep        index of this packet's first rep within the run
            n_reps           reps in this packet
            signal           (n_reps, n_freqs) normalised counts
            reference        (n_reps, n_freqs) or None
            frequencies_mhz  the parked frequencies
            t_offset_s       first_rep * time_per_rep(), seconds from the run start

        Notes
        -----
        The board-side worker transfers in strides of ~10% of the accumulated
        buffer and raises if unread samples ever reach `avg_maxlen`. It runs
        autonomously, so it keeps up on its own; at these rates (a few kHz, one
        read per shot) the DMA has orders of magnitude of headroom. `stream_headroom()`
        reports the actual margin for a given configuration.
        """
        import qickdawg as qd

        if total_reps is None:
            if duration_s is None:
                raise ValueError("stream() needs duration_s or total_reps")
            total_reps = max(1, int(round(float(duration_s) / self.time_per_rep())))

        n_slots = len(self.emission_order())
        per_freq = 1 if getattr(self.cfg, "multipoint_skip_reference", False) else 2
        reads_per_shot = per_freq * n_slots

        # The program loops `reps` times and ends, so the rep count IS the run
        # length. Rebuild the program at the requested size.
        if int(self.cfg.reps) != int(total_reps):
            raise ValueError(
                f"stream() needs cfg.reps == total_reps ({total_reps}), got {self.cfg.reps}. "
                f"Build the program with reps=total_reps -- the tProc loop count is the "
                f"run length, and it cannot be changed after compilation."
            )

        # --- Configuration sequence.
        #
        # This deliberately mirrors NVAveragerProgram.acquire() step for step, so
        # that the only difference between the two paths is what happens in the
        # polling loop below. It did not, before 2026-08-16: `reload_mem()` was
        # missing entirely and `config_bufs` ran after `start_src` rather than
        # before `reload_mem`.
        #
        # That matters because the streaming path is measurably noisier than the
        # burst path on the same hardware at the same tau: on 2026-08-14, ten
        # minutes apart with nothing changed on the bench, the stream's amplitude
        # spectral density was 5.3-6.3x the burst's, flat across 10-500 Hz, with no
        # lines, no drift and no packet-boundary structure. A flat multiplicative
        # excess like that is what a readout-path difference looks like rather
        # than a physical one, and this was the only difference in the path.
        #
        # Whether it is THE cause is not established -- it needs the interleaved
        # burst/stream A/B in scripts/profile_twopoint_acquire.py --compare-read-paths.
        # Aligning the sequences costs nothing either way.
        self.set_reads_per_shot(reads_per_shot)
        self.config_all(qd.soc, load_envelopes=True, reset=True, load_mem=True)
        qd.soc.start_src("internal")
        self.get_data_shape(reads_per_shot)
        self.config_bufs(qd.soc, enable_avg=True, enable_buf=False)
        qd.soc.reload_mem()

        # One start for the whole run. Everything after this is just draining.
        qd.soc.start_readout(total_reps, counter_addr=self.counter_addr,
                             ch_list=list(self.ro_chs), reads_per_shot=self.reads_per_shot)

        scale = float(self.cfg.readout_integration_treg)
        cadence = self.time_per_rep()
        freqs = np.asarray(self.cfg.multipoint_freqs_mhz, dtype=float)
        collected = 0
        idle_since = _perf_counter()

        # QC, read back through `stream_qc` after the run. The point of streaming
        # is that the rep axis is exact; these counters are what prove it on the
        # data rather than assuming it.
        self._stream_qc = {"packets": 0, "packet_sizes": [], "short_reads": 0,
                           "expected_reps": int(total_reps), "collected": 0,
                           "wall_start": _perf_counter()}

        try:
            while collected < total_reps:
                # An explicit `timeout` is what makes `poll_timeout_s` below mean
                # anything. qick's poll_data() defaults to timeout=None, which becomes
                # data_queue.get(block=True, timeout=None) on the board: if the tProc
                # counter stops advancing, that RPC never returns, and the idle check
                # under it is unreachable. Worse, Ctrl-C then unblocks only the client
                # -- the board-side Pyro thread stays blocked on the queue forever and
                # steals the first packet of whatever runs next, which survives a
                # kernel restart. See recover().
                packets = obtain(qd.soc.poll_data(
                    totaltime=self.POLL_TOTALTIME_S, timeout=self.POLL_TIMEOUT_S))
                if not packets:
                    if _perf_counter() - idle_since > poll_timeout_s:
                        raise RuntimeError(
                            f"stream() received no data for {poll_timeout_s:.0f} s after "
                            f"{collected}/{total_reps} reps. The tProc may have stopped early."
                        )
                    continue
                idle_since = _perf_counter()

                for n_reps, (acc_buf, _stats) in packets:
                    if n_reps <= 0 or acc_buf is None:
                        continue
                    # acc_buf[0] is (n_reps * reads_per_shot, 2); NV readouts are DC, so
                    # only the I column carries signal.
                    block = np.asarray(acc_buf[0], dtype=float)[:, 0]
                    if block.size != n_reps * reads_per_shot:
                        # A packet whose payload does not match its own rep count means
                        # the reshape below would silently misalign every subsequent
                        # sample, mixing the two parked frequencies together. Refuse
                        # rather than emit scrambled data.
                        self._stream_qc["short_reads"] += 1
                        raise RuntimeError(
                            f"stream() packet claims {n_reps} reps x {reads_per_shot} reads "
                            f"= {n_reps * reads_per_shot} values but carries {block.size}. "
                            f"Refusing to reshape; the parked frequencies would be mixed.")
                    block = block.reshape(n_reps, reads_per_shot) / scale
                    self._stream_qc["packets"] += 1
                    self._stream_qc["packet_sizes"].append(int(n_reps))

                    if per_freq == 1:
                        signal, reference = block, None
                    else:
                        signal, reference = block[:, 0::2], block[:, 1::2]

                    out = ItemAttribute()
                    out.per_rep = True
                    out.first_rep = collected
                    out.n_reps = int(n_reps)
                    out.t_offset_s = collected * cadence
                    out.frequencies_mhz = freqs
                    out.signal = self._fold_slots(signal)
                    out.reference = self._fold_slots(reference)
                    collected += int(n_reps)
                    self._stream_qc["collected"] = collected

                    if progress:
                        print(f"  stream: +{n_reps} reps ({collected}/{total_reps}, "
                              f"t = {out.t_offset_s:.3f} s)")
                    yield out
        finally:
            # This is a generator, so `finally` also runs on GeneratorExit -- the
            # `break`, the Ctrl-C and the garbage collection of a partly-consumed
            # stream. Without it, an interrupted run left the tProc looping and the
            # board worker transferring into a queue nobody was draining, and the
            # *next* acquire() inherited that: qick's start_readout() would find a
            # readout still running, reset the tProc and flush the queue, and any
            # packet still in flight from the abandoned run would land in the new
            # one. Stopping here costs a few tens of milliseconds at the end of a
            # normal run and removes a whole class of cross-run contamination.
            self._abort_readout()
            self._stream_qc["aborted"] = collected < total_reps

    _stream_qc = None

    def stream_qc(self):
        """Quality report for the last `stream()` run.

        `duty_cycle` is the fraction of wall-clock time the FPGA actually spent
        integrating. Streaming should sit very close to 1.0 -- that is the whole
        point of it, and a low value means the host could not drain the queue.
        """
        if not self._stream_qc:
            return {}
        qc = dict(self._stream_qc)
        sizes = np.asarray(qc.pop("packet_sizes"), dtype=float)
        wall = _perf_counter() - qc.pop("wall_start")
        fpga = qc["collected"] * self.time_per_rep()
        qc.update({
            "reps_missing": int(qc["expected_reps"] - qc["collected"]),
            "packet_reps_median": float(np.median(sizes)) if sizes.size else float("nan"),
            "packet_reps_min": float(sizes.min()) if sizes.size else float("nan"),
            "packet_reps_max": float(sizes.max()) if sizes.size else float("nan"),
            "wall_s": float(wall),
            "fpga_s": float(fpga),
            "duty_cycle": float(fpga / wall) if wall > 0 else float("nan"),
        })
        return qc

    @staticmethod
    def ensure_idle(verbose=True):
        """Pre-flight for the top of an acquisition cell: is the board actually free?

        Costs one RPC when the board is clean, which is why this is the thing to
        call at the start of every run rather than `recover()`. A full recover()
        per run would work but leaks a board-side worker thread each time.

        Returns True if the board was already idle, False if it had to be
        recovered. A False here means the *previous* cell did not finish cleanly,
        so treat any data it wrote as suspect.
        """
        import qickdawg as qd

        try:
            running = bool(qd.soc.streamer.readout_running())
        except Exception as exc:            # noqa: BLE001 - never block a run on the probe
            if verbose:
                print(f"  (could not probe the streamer: {exc}) -- continuing")
            return True

        if not running:
            return True

        if verbose:
            print("Board still has a readout running from a previous cell -- recovering "
                  "before starting, so this run does not inherit its data.")
        MultipointLockinODMR.recover(verbose=verbose)
        return False

    @staticmethod
    def recover(verbose=True):
        """Unwedge the board after an interrupted or hung acquisition.

        Run this whenever a cell had to be interrupted during `acquire()`,
        `stream()` or a burst, or when a run that used to work starts hanging,
        returning short, or producing data that looks like the previous run's.

        **Restarting the notebook kernel does not do this.** The failure lives in
        the board's Pyro daemon, not in the client:

        `qick`'s `poll_data()` defaults to `timeout=None`, which becomes
        `data_queue.get(block=True, timeout=None)` on the board. If the tProc shot
        counter stops advancing -- a program that never started, a tProc left
        running by a previous cell, an abandoned `stream()` generator -- that
        `get` never returns, so the RPC never replies. Ctrl-C raises in the client
        while it waits in `sock.recv(..., MSG_WAITALL)`, but the board-side thread
        stays blocked on that queue for the lifetime of the daemon. It is then a
        second consumer on the streamer's data queue, and the next run's first
        packet can go to it instead of to you.

        `DataStreamer.start_worker()` is the fix: it builds fresh queues, flags and
        a fresh worker thread, so anything blocked on the old queue is orphaned
        against an object nothing will ever write to again. That thread leaks --
        Pyro's pool is ~40 threads, so a handful of incidents is harmless, but if
        `recover()` stops helping, restart the board's Pyro server.

        Returns a dict describing what was found and done.
        """
        import qickdawg as qd

        report = {"readout_was_running": None, "steps": [], "errors": []}

        def _step(name, fn):
            try:
                result = fn()
                report["steps"].append(name)
                return result
            except Exception as exc:            # noqa: BLE001 - best effort by design
                report["errors"].append(f"{name}: {exc}")
                return None

        report["readout_was_running"] = _step(
            "query streamer", lambda: bool(qd.soc.streamer.readout_running()))

        # Order matters. Stop the tProc first so the shot counter stops moving,
        # then break the worker out of its transfer loop, then replace the queues.
        _step("stop tProc", qd.soc.stop_tproc)
        _step("stop readout", qd.soc.streamer.stop_readout)
        _step("settle", lambda: _sleep(0.2))
        _step("restart streamer worker", qd.soc.streamer.start_worker)
        _step("reset generators", qd.soc.reset_gens)

        if verbose:
            print("Board recovery:")
            print(f"  readout was running : {report['readout_was_running']}")
            print(f"  completed           : {', '.join(report['steps'])}")
            if report["errors"]:
                print("  FAILED              : " + "; ".join(report["errors"]))
                print("  If these persist, restart the Pyro server on the board.")
            else:
                print("  board is idle with a fresh streamer; re-run your acquisition cell.")
        return report

    def stream_headroom(self):
        """How much margin the streaming path has against buffer overflow.

        Returns a dict with the accumulated-buffer length, the per-shot read count,
        and how many seconds of acquisition fit in the buffer. Values well above the
        board worker's own transfer period mean the configuration is safe.
        """
        import qickdawg as qd

        n_slots = len(self.emission_order())
        per_freq = 1 if getattr(self.cfg, "multipoint_skip_reference", False) else 2
        reads_per_shot = per_freq * n_slots
        ch = int(self.cfg.adc_channel)
        avg_maxlen = int(qd.soc['readouts'][ch]['avg_maxlen'])
        shots = avg_maxlen // reads_per_shot
        return {
            "avg_maxlen": avg_maxlen,
            "reads_per_shot": reads_per_shot,
            "shots_that_fit": shots,
            "seconds_that_fit": shots * self.time_per_rep(),
            "worker_stride_shots": max(1, int(0.1 * shots)),
        }

    def describe_timing(self, reps=None, host_call_s=None, host_floor_s=None):
        """Human summary of the rate and the averaging headroom.

        The second line is the one that matters in averaged mode: it says how
        much of the per-call floor the FPGA work is actually using, and therefore
        how much free averaging is being left on the table.
        """
        reps = int(self.cfg.reps if reps is None else reps)
        call = self.HOST_CALL_S if host_call_s is None else float(host_call_s)
        floor = self.HOST_FLOOR_S if host_floor_s is None else float(host_floor_s)
        order = str(getattr(self.cfg, "multipoint_pair_order", "forward")).lower()
        n_slots = len(self.emission_order())
        fpga = reps * self.time_per_rep()
        fits = self.reps_that_fit(floor)

        lines = [
            f"{len(self.cfg.multipoint_freqs_mhz)} freqs, order={order} "
            f"({n_slots} slots/rep), tau={self.cfg.readout_integration_tus:.1f} us, "
            f"{reps} reps -> {self.time_per_rep() * 1e6:.0f} us/rep, "
            f"{fpga * 1e3:.2f} ms FPGA work",
        ]
        if fpga < floor:
            lines.append(
                f"  averaged mode: {fpga * 1e3:.2f} ms of FPGA work hides under the "
                f"{floor * 1e3:.1f} ms per-call floor ({100 * fpga / floor:.0f}% used) "
                f"-> {self.predicted_rate_hz(reps, call):.1f} Hz")
            if fits > reps:
                lines.append(
                    f"  FREE AVERAGING: {fits} reps also fit inside the floor. Raising "
                    f"{reps} -> {fits} costs no rate and buys up to "
                    f"{np.sqrt(fits / reps):.1f}x in sigma if the noise is white.")
        else:
            lines.append(
                f"  averaged mode: FPGA work exceeds the {floor * 1e3:.1f} ms floor, so "
                f"reps now trades against rate one for one "
                f"-> {self.predicted_rate_hz(reps, call):.1f} Hz")
        lines.append(
            f"  burst/stream : {1 / self.time_per_rep():.0f} Hz per rep "
            f"({self.time_per_rep() * 1e6:.0f} us cadence); the floor is paid once per "
            f"burst instead of once per sample")
        return "\n".join(lines)
