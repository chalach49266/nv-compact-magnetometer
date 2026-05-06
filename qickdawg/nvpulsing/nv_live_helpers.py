"""Helpers for lower-latency live NV measurements in notebooks."""

from __future__ import annotations

from copy import copy
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from .nvaverageprogram import NVAveragerProgram


def _batch_times(t_start: float, t_stop: float, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=float)
    step = (t_stop - t_start) / float(count)
    return t_start + step * (np.arange(count, dtype=float) + 1.0)



class PLIntensityDual(NVAveragerProgram):
    """
    Triggers both ADC channels (ch=0 = ADC_D, ch=1 = ADC_C) simultaneously
    in a single firmware pulse sequence. Laser is assumed always-on (no PMOD gating).
    """

    required_cfg = [
        "readout_integration_treg",
        "relax_delay_treg",
        "reps",
    ]

    def initialize(self):
        self.check_cfg()
        for ch in (0, 1):
            self.declare_readout(
                ch=ch,
                freq=0,
                length=self.cfg.readout_integration_treg,
                sel="input",
            )
        self.cfg.adcs = [0, 1]
        self.synci(200)

    def body(self):
        self.trigger(
            adcs=self.cfg.adcs,
            width=self.cfg.readout_integration_treg,
            t=0,
        )
        self.wait_all()
        self.sync_all(self.cfg.relax_delay_treg)


def _pl_program_dual(default_config, *, integration_treg: int, batch_size: int):
    cfg = copy(default_config)
    cfg.readout_integration_treg = int(integration_treg)
    cfg.reps = int(batch_size)
    return cfg, PLIntensityDual(cfg)


def _pl_batch_dual(program, integration_treg: int):
    """Returns (values_c, values_d) acquired simultaneously."""
    NVAveragerProgram.acquire(program, progress=False)
    scale = float(integration_treg)
    # d_buf[i] has shape (reps, 1, 2) — last dim is [I, Q].
    # Select [..，0] for I component before reshape, otherwise both I and Q get flattened.
    # d_buf[0] = first declared channel = ch=0 = ADC_D
    # d_buf[1] = second declared channel = ch=1 = ADC_C
    values_d = np.asarray(program.d_buf[0][..., 0], dtype=float).reshape(-1) / scale
    values_c = np.asarray(program.d_buf[1][..., 0], dtype=float).reshape(-1) / scale
    return values_c, values_d


def run_fast_pl_no_display(
    default_config,
    *,
    duration_sec: float | None = None,
    batch_size: int = 500,
    integration_treg: int = 2**16 - 1,
    csv_filename: str | None = None,
) -> pd.DataFrame:
    """
    Acquire PL on both ADC channels simultaneously at hardware rate — no display overhead.

    Uses PLIntensityDual: a single firmware trigger fires both ADC channels at the
    same clock tick, so PL_C and PL_D in each row are truly simultaneous.
    Acquisition time is halved vs sequential (one 76.5 ms batch instead of two).
    Typical throughput: ~150 µs/point, ~784k rows in 2 minutes.
    Stop with Interrupt (■) or set duration_sec.
    """
    config, prog = _pl_program_dual(
        default_config,
        integration_treg=integration_treg,
        batch_size=batch_size,
    )

    all_t: list[float] = []
    all_c: list[float] = []
    all_d: list[float] = []

    t0 = perf_counter()

    try:
        while True:
            t_start = perf_counter() - t0
            values_c, values_d = _pl_batch_dual(prog, config.readout_integration_treg)
            t_end = perf_counter() - t0

            times = _batch_times(t_start, t_end, len(values_c))
            all_t.extend(times.tolist())
            all_c.extend(values_c.tolist())
            all_d.extend(values_d.tolist())

            if duration_sec is not None and t_end >= duration_sec:
                break

    except KeyboardInterrupt:
        print(f"Fast PL stopped — {len(all_t)} points collected.")

    t_arr = np.asarray(all_t)
    df = pd.DataFrame({
        "time_s": t_arr,
        "PL_C":   np.asarray(all_c),
        "PL_D":   np.asarray(all_d),
    })

    if csv_filename is not None:
        df.to_csv(csv_filename, index=False)
        print(f"Saved {len(df)} points to {csv_filename}")

    return df


def run_live_odmr(
    prog_odmr,
    *,
    os_mhz: float = 0.0,
    xlim: tuple[float | None, float | None] | None = None,
    title: str = "Live ODMR — MW ON PL vs Frequency",
) -> None:
    """
    Plot live ODMR with lightweight figure updates.

    Refresh rate is still limited by one full ODMR sweep per `prog_odmr.acquire()`
    call. To speed it up, narrow the sweep range and/or reduce `config_odmr.reps`.
    """

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot([], [], color="steelblue", linewidth=1.5)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("PL Intensity (ADC units)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    handle = display(fig, display_id=True)

    try:
        while True:
            d_live = prog_odmr.acquire(progress=False)
            freqs_display = np.asarray(d_live.frequencies, dtype=float) + float(os_mhz)
            signal = np.asarray(d_live.signal, dtype=float)

            line.set_data(freqs_display, signal)

            if xlim is None:
                ax.set_xlim(float(freqs_display[0]), float(freqs_display[-1]))
            else:
                left, right = xlim
                ax.set_xlim(
                    float(freqs_display[0]) if left is None else float(left),
                    float(freqs_display[-1]) if right is None else float(right),
                )

            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

            fig.canvas.draw_idle()
            handle.update(fig)

    except KeyboardInterrupt:
        print("Live plot stopped.")
    finally:
        plt.ioff()
        plt.close(fig)
