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


def _live_odmr_sensitivity(d_live, prog_odmr):
    """Structure-aware η_B from one ODMR sweep. Soft-imports notebook helper."""
    try:
        from odmr_sensitivity import estimate_sensitivity
    except ImportError:
        return None

    cfg = getattr(prog_odmr, "cfg", None)
    point_time_s = None
    if cfg is not None and hasattr(prog_odmr, "total_time") and getattr(cfg, "nsweep_points", 0):
        try:
            point_time_s = float(prog_odmr.total_time()) / float(cfg.nsweep_points)
        except Exception:
            point_time_s = None

    try:
        res = estimate_sensitivity(d_live, point_time_s=point_time_s, config=cfg)
    except Exception:
        return None

    best = res.get("best")
    if not best or best not in res:
        return None
    r = res[best]
    return {
        "best": best,
        "eta_white_nT": float(r["sensitivity_t_rt_hz_white"]) * 1e9,
        "eta_base_nT": float(r["sensitivity_t_rt_hz"]) * 1e9,
        "f_op_mhz": float(r["f_at_slope_mhz"]),
        "slope": float(r["slope_adc_per_mhz"]),
        "noise_white": float(r["noise_white"]),
        "noise_baseline": float(r["noise_baseline"]),
    }


def run_live_odmr(
    prog_odmr,
    *,
    os_mhz: float = 0.0,
    xlim: tuple[float | None, float | None] | None = None,
    title: str = "Live ODMR — MW ON PL vs Frequency",
    show_sensitivity: bool = True,
) -> None:
    """
    Plot live ODMR with lightweight figure updates.

    Refresh rate is still limited by one full ODMR sweep per `prog_odmr.acquire()`
    call. To speed it up, narrow the sweep range and/or reduce `config_odmr.reps`.

    When ``show_sensitivity`` is True (default), each sweep also runs the
    structure-aware ODMR sensitivity estimator and overlays η_B on the title
    plus a marker at the steepest-slope operating frequency.
    """

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    line, = ax.plot([], [], color="steelblue", linewidth=1.5, label="MW ON PL")
    op_line = ax.axvline(
        0.0,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        label="max |dS/df|",
        visible=False,
    )
    sens_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("PL Intensity (ADC units)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    handle = display(fig, display_id=True)
    last_sens_label = "η: waiting for first sweep…"

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

            if show_sensitivity:
                sens = _live_odmr_sensitivity(d_live, prog_odmr)
                if sens is not None:
                    f_op_display = sens["f_op_mhz"] + float(os_mhz)
                    op_line.set_xdata([f_op_display, f_op_display])
                    op_line.set_visible(True)
                    last_sens_label = (
                        f"η = {sens['eta_white_nT']:.1f}–{sens['eta_base_nT']:.1f} nT/√Hz"
                        f"  ({sens['best']})\n"
                        f"op @ {f_op_display:.1f} MHz"
                        f"  slope={sens['slope']:.1f} ADC/MHz\n"
                        f"noise w/b = {sens['noise_white']:.3f}/{sens['noise_baseline']:.3f}"
                    )
                    ax.set_title(
                        f"{title}\n"
                        f"η = {sens['eta_white_nT']:.1f}–{sens['eta_base_nT']:.1f} nT/√Hz"
                        f" ({sens['best']}) @ {f_op_display:.1f} MHz"
                    )
                else:
                    op_line.set_visible(False)
                    last_sens_label = "η: fit failed / unavailable"
                    ax.set_title(f"{title}\nη: fit failed / unavailable")
                sens_text.set_text(last_sens_label)
            else:
                op_line.set_visible(False)
                sens_text.set_text("")
                ax.set_title(title)

            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

            fig.canvas.draw_idle()
            handle.update(fig)

    except KeyboardInterrupt:
        print("Live plot stopped.")
        if show_sensitivity and last_sens_label:
            print(last_sens_label.replace("\n", " | "))
    finally:
        plt.ioff()
        plt.close(fig)
