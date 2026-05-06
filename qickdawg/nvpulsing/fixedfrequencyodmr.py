"""
FixedFrequencyODMR
=======================================================================
An NVAveragerProgram that runs ODMR-style measurement (MW on / MW off) at a
**single fixed frequency** — no frequency sweep. Use config.mw_fMHz and config.mw_gain
directly. Useful for microwave gain calibration and power testing.
"""

from .nvaverageprogram import NVAveragerProgram
from itemattribute import ItemAttribute
from ..util import apply_on_axis_0_n_times

import numpy as np


class FixedFrequencyODMR(NVAveragerProgram):
    """
    ODMR-style measurement at one fixed microwave frequency. No add_sweep — uses
    config.mw_fMHz and config.mw_gain directly.

    Required config: readout_integration_treg, adc_channel, laser_gate_pmod,
    relax_delay_treg, mw_channel, mw_nqz, mw_gain, mw_fMHz, pre_init, reps
    """

    required_cfg = [
        "readout_integration_treg",
        "adc_channel",
        "laser_gate_pmod",
        "relax_delay_treg",
        "mw_channel",
        "mw_nqz",
        "mw_gain",
        "mw_fMHz",
        "pre_init",
        "reps",
    ]

    def initialize(self):
        self.check_cfg()

        if self.cfg.mw_gain > 30000:
            raise ValueError("config.mw_gain should be <= 30000")

        self.setup_readout()

        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        self.default_pulse_registers(
            ch=self.cfg.mw_channel,
            style="const",
            freq=self.cfg.mw_freg,
            gain=self.cfg.mw_gain,
            length=self.cfg.readout_integration_treg,
            phase=0,
        )
        self.set_pulse_registers(ch=self.cfg.mw_channel)

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
        self.pulse(ch=self.cfg.mw_channel, t=0)

        self.trigger_no_off(
            adcs=self.cfg.adcs,
            pins=[self.cfg.laser_gate_pmod],
            width=self.cfg.readout_integration_treg,
            adc_trig_offset=0,
            t=0,
        )

        self.trigger_no_off(
            pins=[self.cfg.laser_gate_pmod],
            width=self.cfg.relax_delay_treg,
            adc_trig_offset=0,
            t=self.cfg.readout_integration_treg,
        )

        t_ref = self.cfg.readout_integration_treg + self.cfg.relax_delay_treg
        if getattr(self.cfg, "odmr_reference_zero_gain_pulse", False):
            ref_gain = int(getattr(self.cfg, "odmr_reference_pulse_gain", 1))
            self.set_pulse_registers(
                ch=self.cfg.mw_channel,
                style="const",
                gain=ref_gain,
                length=self.cfg.readout_integration_treg,
                phase=0,
            )
            self.pulse(ch=self.cfg.mw_channel, t=t_ref)

        self.trigger(
            adcs=self.cfg.adcs,
            pins=[self.cfg.laser_gate_pmod],
            width=self.cfg.readout_integration_treg,
            adc_trig_offset=0,
            t=t_ref,
        )

        self.wait_all()
        self.sync_all(self.cfg.relax_delay_treg)

    def acquire(self, raw_data=False, *arg, **kwarg):
        data = super().acquire(readouts_per_experiment=2, *arg, **kwarg)

        if raw_data is False:
            data = self.analyze_results(data)

        return data

    def analyze_results(self, data):
        data = np.reshape(data, self.data_shape)

        d = self.analyze_data(data)

        d.frequencies = np.atleast_1d(self.cfg.mw_fMHz)
        d.signal = np.atleast_1d(d.signal)
        d.reference = np.atleast_1d(d.reference)
        d.contrast = np.atleast_1d(d.contrast)
        if hasattr(d, "contrast_percent"):
            d.contrast_percent = np.atleast_1d(d.contrast_percent)

        return d

    def analyze_data(self, data):
        if self.cfg.edge_counting is False:
            data = data / self.cfg.readout_integration_treg

        signal = data[..., 0]
        reference = data[..., 1]
        if getattr(self.cfg, "odmr_lockin_swap_signal_reference", False):
            signal, reference = reference, signal

        contrast = signal - reference
        if self.cfg.edge_counting is False:
            contrast_percent = (signal - reference) / reference * 100

        n = len(contrast.shape) - 1

        d = ItemAttribute()

        if self.cfg.edge_counting is False:
            func = np.mean
        else:
            func = np.sum

        d.contrast = apply_on_axis_0_n_times(contrast, func, n)
        d.signal = apply_on_axis_0_n_times(signal, func, n)
        d.reference = apply_on_axis_0_n_times(reference, func, n)

        if self.cfg.edge_counting is False:
            d.contrast_percent = apply_on_axis_0_n_times(contrast_percent, func, n)

        return d

    def total_time(self):
        t = self.cfg.readout_integration_tus * 2 + self.cfg.relax_delay_tus * 2
        return t * self.cfg.reps / 1e6
