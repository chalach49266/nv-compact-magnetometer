"""Post-processing for the three two-point lock-in acquisition modes.

One pipeline per mode, because the three modes fail in different ways and a
single generic cleaner ends up doing the wrong thing to two of them:

    averaged   one sample per acquire() call, ~88 Hz. Uniformly sampled, so
               nothing needs re-timing -- but 60 Hz mains aliases to ~28 Hz at
               this rate and the pipeline says so loudly, because no amount of
               filtering can undo it.
    burst      one acquire() returns `reps` samples at the full FPGA cadence.
               Fast, but ~50% of the batches are replays of the previous one
               (see `burst_qc`), the recorded timestamps compress each batch into
               its measured wall-clock window, and there is a real transient on
               row 0. All three are repaired here.
    stream     one FPGA run, drained continuously, ~1 kHz. Structurally the
               cleanest -- gapless rep indices, no duplicates -- but the PL
               droops ~25% over a 30 s run because there is no duty-cycle gap,
               and its broadband noise floor is currently ~6x the burst floor.

Every pipeline returns a `TwoPointResult`: the cleaned frame, a flat report dict
suitable for a CSV row, and a `twopoint_spectra.SpectrumSummary`. The notebook's
analysis cells are thin wrappers around `process()`.

Usage
-----
    from twopoint_postprocess import process
    res = process("data/twopoint_lockin/twopoint_lockin_stream_20260814_145442.csv")
    print(res.describe())
    res.figure("out.png")

or from a shell:

    python -m twopoint_postprocess <csv> [--outdir DIR] [--notch-mains]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import burst_qc as qc                        # noqa: E402
import twopoint_spectra as spec              # noqa: E402
from spike_rejection import HampelDespiker    # noqa: E402

GAMMA_NV_MHZ_PER_UT = spec.GAMMA_NV_MHZ_PER_UT

# tProc fabric clock, from the recovered readout registers: 213 us <-> 65434 treg
# and 120 us <-> 36864 treg both give 307.2 MHz.
FPGA_CLOCK_HZ = 307.2e6


# --------------------------------------------------------------------------- #
# Mode detection
# --------------------------------------------------------------------------- #

def detect_mode(df: pd.DataFrame) -> str:
    """Which acquisition mode wrote this frame: averaged / burst / stream.

    Decided by columns rather than by filename, so a renamed or concatenated file
    still lands in the right pipeline. `packet` only ever comes from `stream()`;
    `batch` with more than one row per batch only ever comes from
    `acquire(per_rep=True)`.
    """
    if "packet" in df.columns and "rep_index" in df.columns:
        return "stream"
    if qc.is_burst(df):
        return "burst"
    if "batch" in df.columns:
        return "averaged"
    raise ValueError("frame has none of 'packet', 'batch' -- not a two-point live CSV")


# --------------------------------------------------------------------------- #
# Conversion recovery (moved here from scripts/clean_burst_lockin.py)
# --------------------------------------------------------------------------- #

def recover_conversion(df: pd.DataFrame) -> dict:
    """Recover the counts->z scale and the z->frequency map from the file itself.

    Both are exact linear maps applied by the acquisition cell, so they can be
    read back off the recorded columns without the calibration JSON:

        z            = counts / ref_norm_counts
        lockin       = z_plus - z_minus
        delta_f_mhz  = (lockin - zero) / denom

    Fitting `lockin` against `delta_f_mhz` recovers `denom` and `zero` to machine
    precision (residual ~1e-16 on the measured runs). Doing it this way means the
    cleaner works on any live CSV, including ones whose calibration file has moved.
    """
    scale = float(np.median(df["peak_01"].to_numpy() / df["z_minus"].to_numpy()))

    x = df["delta_f_mhz"].to_numpy(float)
    y = df["lockin_signal"].to_numpy(float)
    good = np.isfinite(x) & np.isfinite(y)
    denom, zero = np.polyfit(x[good], y[good], 1)
    residual = float(np.max(np.abs(y[good] - (denom * x[good] + zero))))

    return {"counts_per_z": scale, "denom": float(denom), "zero": float(zero),
            "fit_residual": residual}


def recompute_from_counts(df: pd.DataFrame, conv: dict) -> pd.DataFrame:
    """Rebuild z, lock-in and shift columns from (possibly cleaned) raw counts."""
    out = df.copy()
    out["z_minus"] = out["peak_01"] / conv["counts_per_z"]
    out["z_plus"] = out["peak_02"] / conv["counts_per_z"]
    out["lockin_signal"] = out["z_plus"] - out["z_minus"]
    out["delta_f_mhz"] = (out["lockin_signal"] - conv["zero"]) / conv["denom"]
    out["peak_shift_kHz"] = out["delta_f_mhz"] * 1e3
    out["B_shift_uT"] = out["delta_f_mhz"] / GAMMA_NV_MHZ_PER_UT
    if "f_new_mhz" in df.columns:
        f0 = float((df["f_new_mhz"] - df["delta_f_mhz"]).median())
        out["f_new_mhz"] = f0 + out["delta_f_mhz"]
    return out


def recover_readout_quantum(values: np.ndarray, max_reps: int = 512) -> Optional[dict]:
    """Recover `treg x reps` from the quantisation of the recorded counts.

    `analyze_results` divides the accumulated buffer -- an integer -- by
    `readout_integration_treg` and then averages over `reps`, so every stored
    count is an integer multiple of `1 / (treg * reps)`. Finding the smallest
    divisor that makes all values integral therefore recovers the *exact* FPGA
    readout time of the run, `n_slots * D / f_clk`, with no reliance on what the
    notebook says it configured.

    This is what showed that the 2026-08-14 runs really were at tau = 120 us
    while still taking 11.4 ms per batch.

    Only the product `D` is identifiable, not its factors: 36864 is
    (treg 36864) x (reps 1) and (treg 768) x (reps 48) equally well. Callers get
    `D` and must interpret it using what they know about the mode --
    `readout_seconds()` below needs only `D`, which is why every downstream
    number here depends on `D` alone.

    Returns None if no candidate fits, e.g. for a file that has been rescaled.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)][:2000]
    if v.size < 16:
        return None
    tregs = sorted({int(round(t * FPGA_CLOCK_HZ / 1e6)) for t in np.arange(1, 301, 0.5)})
    best = None
    for treg in tregs:
        for reps in range(1, max_reps + 1):
            D = treg * reps
            if best is not None and D >= best["D"]:
                break
            x = v * D
            if np.max(np.abs(x - np.round(x))) < 2e-6:
                best = {"D": D}
                break
    return best


def readout_seconds(D: int, n_slots: int = 2) -> float:
    """Seconds of ADC integration in one acquire (averaged) or one rep (per-rep).

    `D = treg * reps` counts readout clock cycles per emission slot, so the total
    is `n_slots * D / f_clk`. Independent of how `D` factorises.
    """
    return n_slots * float(D) / FPGA_CLOCK_HZ


def infer_slots(D: int, wall_cadence_s: float,
                candidates: Sequence[int] = (2, 4, 8)) -> tuple[int, float]:
    """Emission slots per rep, from wall-clock cadence against the exact quantum.

    "forward" order with `skip_reference` gives 2 readouts per rep; "abba", or
    keeping the reference readout, gives 4. The measured cadence always exceeds
    the FPGA time by the host share, so the ratio sits slightly above the true
    slot count and rounding to the nearest candidate is safe as long as the host
    share stays well under 100%.
    """
    quantum = float(D) / FPGA_CLOCK_HZ
    ratio = wall_cadence_s / quantum if quantum > 0 else float("nan")
    if not np.isfinite(ratio):
        return candidates[0], ratio
    return int(min(candidates, key=lambda k: abs(k - ratio))), ratio


# --------------------------------------------------------------------------- #
# Despiking
# --------------------------------------------------------------------------- #

def robust_sigma(values: np.ndarray) -> float:
    """Noise scale from successive differences -- unaffected by drift or offsets."""
    d = np.diff(np.asarray(values, dtype=float)[np.isfinite(values)])
    if d.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def despike_by_segment(df: pd.DataFrame, columns: Sequence[str], *, window: int = 11,
                       k_sigma: float = 4.0, sigma_floor: Optional[float] = None,
                       sigma_cap: Optional[float] = None
                       ) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Run the causal Hampel filter within each segment, never across a gap.

    `HampelDespiker` is stateful over a trailing window, so feeding it across the
    dead time between bursts would compare samples tens of ms apart as if they
    were adjacent. Each segment gets its own filter instance.

    The filter's default sigma_floor/sigma_cap (1.5 / 2.5 ADC) were tuned for a
    setup whose per-channel noise was ~1 ADC. Measured noise here ranges from
    7 to 90 ADC depending on mode, and a fixed cap of 2.5 would put the threshold
    far below the real noise and reject almost everything. So unless the caller
    overrides them, both are derived from this file's own robust noise estimate.

    **Do not use this on the two parked channels in post-processing.** They are
    0.84-0.97 correlated because the PL common mode dominates both, but the
    filter treats them independently: on the 2026-08-14 runs it flagged 30 and 23
    samples in the two channels with only 4 in common. Every one-sided
    replacement breaks the correlation the two-point estimator relies on and
    *injects* differential noise -- measured sigma rose from 54.6 to 55.8 kHz
    (averaged) and 160.8 to 175.5 kHz (stream). Use `despike_shift`, which
    filters the differential itself. This function stays for the live-loop case,
    where the raw channels must be cleaned before the conversion exists.
    """
    columns = list(columns)
    sigmas = [robust_sigma(df[c].to_numpy(float)) for c in columns]
    typical = float(np.median(sigmas)) if sigmas else 1.0
    floor = 0.5 * typical if sigma_floor is None else float(sigma_floor)
    cap = 3.0 * typical if sigma_cap is None else float(sigma_cap)

    out = df.copy()
    flags = np.zeros((len(df), len(columns)), dtype=bool)
    # `.to_numpy()` can hand back a read-only view of the block manager when the
    # selected columns share a dtype, and the filter writes back in place.
    values = np.array(df[columns].to_numpy(float), dtype=float, copy=True)
    segments = (df["segment"].to_numpy() if "segment" in df.columns
                else np.zeros(len(df), dtype=int))

    for seg in np.unique(segments):
        rows = np.where(segments == seg)[0]
        despiker = HampelDespiker(n_channels=len(columns), window=window,
                                  k_sigma=k_sigma, sigma_floor=floor, sigma_cap=cap)
        for r in rows:
            clean, flag = despiker.update(values[r])
            values[r] = clean
            flags[r] = flag

    for i, c in enumerate(columns):
        out[c] = values[:, i]
        out[f"{c}_despiked"] = flags[:, i]
    return out, flags, {"sigma_typical": typical, "sigma_floor": floor,
                        "sigma_cap": cap, "n_flagged": int(flags.sum())}


def despike_shift(df: pd.DataFrame, column: str = "peak_shift_kHz", *,
                  window: int = 11, k_sigma: float = 4.0,
                  sigma_floor: Optional[float] = None,
                  sigma_cap: Optional[float] = None
                  ) -> tuple[pd.DataFrame, dict]:
    """Hampel-filter the differential shift, within segments.

    This is the right place to despike two-point data. The estimator's whole
    point is that common-mode PL excursions cancel in `z+ - z-`, so a spike that
    matters is one that survives the subtraction -- and that is what this sees.
    Filtering the parked channels separately instead breaks the cancellation
    (see `despike_by_segment`).

    Measured effect on 2026-08-14: sigma 54.6 -> 53.6 kHz (averaged, 19 samples
    flagged) and 160.8 -> 155.2 kHz (stream, 389 flagged).
    """
    values = df[column].to_numpy(float).copy()
    sigma = robust_sigma(values)
    floor = 0.5 * sigma if sigma_floor is None else float(sigma_floor)
    cap = 3.0 * sigma if sigma_cap is None else float(sigma_cap)

    segments = (df["segment"].to_numpy() if "segment" in df.columns
                else np.zeros(len(df), dtype=int))
    flags = np.zeros(len(df), dtype=bool)

    for seg in np.unique(segments):
        rows = np.where(segments == seg)[0]
        despiker = HampelDespiker(n_channels=1, window=window, k_sigma=k_sigma,
                                  sigma_floor=floor, sigma_cap=cap)
        for r in rows:
            clean, flag = despiker.update(np.array([values[r]]))
            values[r] = clean[0]
            flags[r] = bool(flag[0])

    out = df.copy()
    out[column] = values
    out[f"{column}_despiked"] = flags
    if "delta_f_mhz" in out.columns:
        out["delta_f_mhz"] = values * 1e-3
    if "B_shift_uT" in out.columns:
        out["B_shift_uT"] = values * 1e-3 / GAMMA_NV_MHZ_PER_UT
    return out, {"sigma_robust": sigma, "sigma_floor": floor, "sigma_cap": cap,
                 "n_flagged": int(flags.sum())}


def detrend_series(values: np.ndarray, window: int) -> np.ndarray:
    """Remove slow drift with a centred rolling median, preserving the mean.

    Used for the streaming PL droop, which is a ~25% ramp over 30 s -- far too
    large to leave in before quoting a standard deviation, and not a polynomial
    of any fixed order. The window sets the high-pass corner at roughly
    `fs / window` Hz.
    """
    s = pd.Series(np.asarray(values, dtype=float))
    baseline = s.rolling(int(window), center=True, min_periods=1).median()
    return (s - baseline + s.mean()).to_numpy()


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

@dataclass
class TwoPointResult:
    mode: str
    raw: pd.DataFrame
    clean: pd.DataFrame
    report: dict
    spectrum: spec.SpectrumSummary
    notes: list[str] = field(default_factory=list)
    source: Optional[Path] = None

    def describe(self) -> str:
        head = f"{self.mode.upper()} run"
        if self.source is not None:
            head += f" -- {Path(self.source).name}"
        lines = [head, "-" * len(head)]
        for key, value in self.report.items():
            if isinstance(value, float):
                lines.append(f"  {key:24s}: {value:,.4g}")
            else:
                lines.append(f"  {key:24s}: {value}")
        lines.append("")
        lines.append(self.spectrum.describe())
        if self.notes:
            lines.append("")
            lines.extend(f"  NOTE: {n}" for n in self.notes)
        return "\n".join(lines)

    def summary_row(self) -> dict:
        row = {"mode": self.mode,
               "source": Path(self.source).name if self.source else ""}
        row.update(self.report)
        row.update(self.spectrum.to_dict())
        return row

    def spectrum_figure(self, path=None, show: bool = False, annotate: bool = True,
                        fft_fmax_hz=None):
        """Both spectra of the run, one above the other.

        Top --- Welch amplitude spectral density, log-log, in nT/sqrt(Hz). Averaged
        over segments, so the broadband floor is stable and this is the panel a
        sensitivity is quoted from. A coherent line is smeared here, and its height
        depends on the segment length rather than on the signal.

        Bottom --- the plain single-sided FFT of the whole record, linear axes, in
        nT. One bin per fs/N with nothing averaged, so a periodic component reads
        its actual amplitude: a 60 Hz line at 500 nT is 500 nT tall. The broadband
        floor here is hashy by construction -- a single periodogram has ~100%
        scatter in every bin -- so do not read a noise floor off this panel.

        Quote sensitivity from the top panel; identify and size a line on the
        bottom one.

        `fft_fmax_hz` limits the bottom panel's frequency axis (default: the whole
        band up to Nyquist).
        """
        import matplotlib.pyplot as plt

        sp = self.spectrum
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.4))
        fig.set_label("fft")

        # ------------------------------------------------ top: Welch ASD, log-log
        if sp.f is None or sp.asd is None:
            ax.text(0.5, 0.5, "no spectrum: the run was too short to transform",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            ax.loglog(sp.f[1:], sp.asd[1:], lw=0.8, color="tab:purple")
            if np.isfinite(sp.floor_nt_rthz):
                ax.axhline(sp.floor_nt_rthz, color="0.35", ls="--", lw=1.1,
                           label=f"white floor {sp.floor_nt_rthz:.1f} nT/rtHz "
                                 f"({sp.floor_band_hz[0]:.0f}-{sp.floor_band_hz[1]:.0f} Hz)")
            if np.isfinite(sp.eta_nt_rthz):
                ax.axhline(sp.eta_nt_rthz, color="0.65", ls=":", lw=1.1,
                           label=f"eta from sigma {sp.eta_nt_rthz:.1f} nT/rtHz")
            if annotate:
                for _, r in sp.lines.iterrows():
                    if r["excess_over_floor"] > 2.0:
                        ax.axvline(r["frequency_hz"], color="tab:red", ls=":", lw=0.8)
                        ax.annotate(f"{r['frequency_hz']:.0f} Hz  "
                                    f"{r['excess_over_floor']:.0f}x",
                                    xy=(r["frequency_hz"], 0.03),
                                    xycoords=("data", "axes fraction"),
                                    fontsize=7, color="tab:red", ha="center",
                                    va="bottom", rotation=90,
                                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.6))
                for a in sp.aliases:
                    ax.annotate(str(a), xy=(0.99, 0.02), xycoords="axes fraction",
                                ha="right", va="bottom", fontsize=7, color="tab:red")
            ax.set_xlim(sp.f[1], sp.f[-1])
            ax.legend(fontsize=8, loc="upper right")
        ax.set(xlabel="frequency (Hz)", ylabel="nT / sqrt(Hz)",
               title=f"Amplitude spectral density (Welch) -- {self.mode}, "
                     f"fs = {sp.fs_hz:.1f} Hz"
                     + (" (per burst segment)" if self.mode == "burst" else ""))
        ax.grid(alpha=0.3, which="both")

        # -------------------------------------- bottom: plain FFT, linear axes
        f2, a2 = sp.fft_f, sp.fft_amp_nt
        if f2 is None or a2 is None or len(f2) < 2:
            why = ("burst records are not contiguous, so a transform across the "
                   "whole run would put a frequency scale on the inter-burst gaps"
                   if self.mode == "burst" else "the run was too short to transform")
            ax2.text(0.5, 0.5, f"no plain FFT: {why}", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=9, wrap=True)
            ax2.set_axis_off()
        else:
            fmax = float(fft_fmax_hz) if fft_fmax_hz else float(f2[-1])
            keep = (f2 > 0) & (f2 <= fmax)
            ax2.plot(f2[keep], a2[keep], lw=0.6, color="tab:blue")
            if annotate and keep.any():
                # Label the tallest few bins -- this panel is amplitude-correct,
                # so the number beside each peak is the size of that component.
                amp = a2[keep]
                freq = f2[keep]
                order = np.argsort(amp)[::-1]
                picked = []
                for j in order:
                    if len(picked) >= 4:
                        break
                    # one label per line, not per bin of the same line
                    if all(abs(freq[j] - freq[k]) > 0.02 * fmax for k in picked):
                        picked.append(j)
                for j in picked:
                    ax2.annotate(f"{freq[j]:.1f} Hz\n{amp[j]:.0f} nT",
                                 xy=(freq[j], amp[j]), xytext=(0, 6),
                                 textcoords="offset points", ha="center",
                                 fontsize=7, color="tab:red")
                ax2.set_ylim(0, float(amp.max()) * 1.35)
            ax2.set_xlim(0, fmax)
        ax2.set(xlabel="frequency (Hz)", ylabel="amplitude (nT)",
                title="Plain FFT of the whole record -- amplitude, not density; "
                      "read line sizes here, not the noise floor")
        ax2.grid(alpha=0.3)

        fig.suptitle(Path(self.source).name if self.source else self.mode, y=1.0)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=130, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def figure(self, path=None, show: bool = False):
        """Three panels: cleaned shift, the two z channels, and the ASD."""
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(13, 9))
        fig.set_label("result")          # figure_autosave names the PNG from this
        raw, cln = self.raw, self.clean

        ax = axes[0]
        if self.mode == "burst" and "_dropped" in raw.columns:
            drop = raw["_dropped"].to_numpy(bool)
            ax.scatter(raw["time_s"][~drop], raw["peak_shift_kHz"][~drop], s=0.4,
                       color="0.7", alpha=0.5, rasterized=True, label="as recorded")
            ax.scatter(raw["time_s"][drop], raw["peak_shift_kHz"][drop], s=0.4,
                       color="tab:red", alpha=0.5, rasterized=True, label="dropped")
        else:
            ax.plot(raw["time_s"], raw["peak_shift_kHz"], lw=0.4, color="0.7",
                    label="as recorded")
        seg = cln["segment"].to_numpy() if "segment" in cln.columns else np.zeros(len(cln))
        for s in np.unique(seg):
            sub = cln[seg == s]
            ax.plot(sub["time_s"], sub["peak_shift_kHz"], lw=0.5, color="crimson")
        ax.set(ylabel="peak shift (kHz)", xlabel="time (s)",
               title=f"{self.mode} -- cleaned peak shift (red) over raw (grey)")
        ax.legend(fontsize=8, markerscale=8, loc="upper right")
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(cln["time_s"], cln["z_minus"], lw=0.4, color="tab:orange", label="z at f-")
        ax.plot(cln["time_s"], cln["z_plus"], lw=0.4, color="tab:blue", label="z at f+")
        ax.set(ylabel="normalised PL (z)", xlabel="time (s)",
               title="Parked channels -- common-mode drift shows up here, "
                     "not in the difference")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[2]
        sp = self.spectrum
        if sp.f is not None:
            ax.loglog(sp.f[1:], sp.asd[1:], lw=0.8, color="tab:purple")
            if np.isfinite(sp.floor_nt_rthz):
                ax.axhline(sp.floor_nt_rthz, color="0.4", ls="--", lw=1.0,
                           label=f"floor {sp.floor_nt_rthz:.0f} nT/rtHz "
                                 f"({sp.floor_band_hz[0]:.0f}-{sp.floor_band_hz[1]:.0f} Hz)")
            for _, r in sp.lines.iterrows():
                if r["excess_over_floor"] > 2.0:
                    ax.axvline(r["frequency_hz"], color="tab:red", ls=":", lw=0.8)
            ax.legend(fontsize=8)
        ax.set(xlabel="frequency (Hz)", ylabel="nT / sqrt(Hz)",
               title="Amplitude spectral density"
                     + (" (per burst segment)" if self.mode == "burst" else ""))
        ax.grid(alpha=0.3, which="both")

        fig.suptitle(Path(self.source).name if self.source else self.mode, y=1.0)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=130, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #

def _finish(mode, raw, clean, report, notes, fs, segment=None, source=None,
            floor_band=None) -> TwoPointResult:
    summary = spec.summarise(clean["peak_shift_kHz"].to_numpy(float), fs,
                             segment=segment, floor_band=floor_band)
    return TwoPointResult(mode=mode, raw=raw, clean=clean, report=report,
                          spectrum=summary, notes=notes, source=source)


def _apply_despike(clean, raw, despike, report, notes, **kwargs):
    """Dispatch the despiking mode. See `despike_shift` for why "shift" is default."""
    if despike in (False, None, "none"):
        return clean
    if despike in (True, "shift"):
        clean, info = despike_shift(clean, **kwargs)
        report["despiked_samples"] = info["n_flagged"]
        return clean
    if despike == "channels":
        clean, _, info = despike_by_segment(clean, qc.peak_columns(clean), **kwargs)
        clean = recompute_from_counts(clean, recover_conversion(raw))
        report["despiked_samples"] = info["n_flagged"]
        notes.append("despiked the parked channels independently: this breaks the "
                     "common-mode cancellation and typically RAISES sigma. Use "
                     "despike='shift' unless you are chasing raw-channel glitches.")
        return clean
    raise ValueError(f"despike must be 'shift', 'channels' or False, got {despike!r}")


def process_averaged(df: pd.DataFrame, *, despike="shift", detrend: bool = False,
                     detrend_window: int = 201, notch_mains: bool = False,
                     slots_per_rep: int = 2, source=None,
                     **despike_kwargs) -> TwoPointResult:
    """Averaged mode: one sample per acquire call.

    Nothing structural to repair -- the sampling is uniform and duplicates do not
    occur here. What this pipeline is for is the *diagnosis*: it reports the
    per-call floor that sets the rate, and it warns about the mains aliasing that
    the rate makes unavoidable.
    """
    raw = df.copy()
    notes: list[str] = []

    t = raw["time_s"].to_numpy(float)
    fs = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else float("nan")

    report = {"n_samples": len(raw), "duration_s": float(t[-1] - t[0]),
              "rate_hz": float(fs)}
    n_slots = int(slots_per_rep)

    if "acq_seconds" in raw.columns:
        acq = raw["acq_seconds"].to_numpy(float)
        report.update({
            "acq_p05_ms": float(np.percentile(acq, 5) * 1e3),
            "acq_median_ms": float(np.median(acq) * 1e3),
            "acq_p95_ms": float(np.percentile(acq, 95) * 1e3),
            "acq_outliers_gt3x": int((acq > 3 * np.median(acq)).sum()),
        })
        # The fastest call in the run bounds the per-call floor from above: no
        # call can be quicker than the host chain it has to walk.
        notes.append(f"per-call host floor <= {np.percentile(acq, 5) * 1e3:.2f} ms "
                     f"(fastest batch in the run)")

    quantum = recover_readout_quantum(raw["peak_01"].to_numpy(float))
    if quantum is not None:
        # In averaged mode only the product treg*reps is identifiable, so the
        # readout window and the rep count cannot be separated -- but the total
        # FPGA readout time depends only on the product, which is what matters
        # for the rate. n_slots=2 assumes "forward" order with skip_reference;
        # "abba" or a kept reference doubles it, and is flagged below if the
        # result would not fit inside the fastest observed call.
        fpga_ms = readout_seconds(quantum["D"], n_slots) * 1e3
        report["readout_quantum_D"] = int(quantum["D"])
        report["fpga_readout_ms"] = float(fpga_ms)
        report["fpga_share_of_period"] = float(fpga_ms / (1e3 / fs)) if fs > 0 else np.nan
        notes.append(f"FPGA readout work is {fpga_ms:.2f} ms of the "
                     f"{1e3 / fs:.2f} ms period ({100 * fpga_ms * fs / 1e3:.0f}%); "
                     f"the rest is the per-call host floor, which no readout-window "
                     f"change can shorten")
        if "acq_seconds" in raw.columns and 2 * fpga_ms > np.percentile(acq, 5) * 1e3:
            notes.append("(this assumes 2 emission slots per rep; 4 slots would not "
                         "fit inside the fastest observed call, so 2 is confirmed)")

    table = qc.batch_table(raw)
    n_stale = int(table.stale.sum())
    report["stale_batches"] = n_stale
    if n_stale:
        notes.append(f"{n_stale} batches repeat the previous one exactly -- unexpected "
                     f"in averaged mode; check prog.n_stale_acquires")

    clean = raw.copy()
    clean["segment"] = 0

    clean = _apply_despike(clean, raw, despike, report, notes, **despike_kwargs)

    if detrend:
        clean["peak_shift_kHz"] = detrend_series(clean["peak_shift_kHz"].to_numpy(),
                                                 detrend_window)
    if notch_mains:
        clean["peak_shift_kHz"] = spec.notch(clean["peak_shift_kHz"].to_numpy(),
                                             fs, [spec.MAINS_HZ * k for k in (1, 2, 3)])
        notes.append("mains comb notched -- at this rate the notch removes an "
                     "ALIASED band, taking real signal with it. Diagnostic only.")

    return _finish("averaged", raw, clean, report, notes, fs, source=source)


def process_burst(df: pd.DataFrame, *, drop_stale: bool = True,
                  drop_first_sample: bool = True, despike="shift",
                  notch_mains: bool = False, source=None,
                  **despike_kwargs) -> TwoPointResult:
    """Burst mode: repair the replay, the row-0 transient and the time axis.

    Order matters. Staleness is judged on the *unfiltered* frame, because the
    batch table reconstructs each batch's start time from its first row; dropping
    row 0 first would shift every start by half a sample.
    """
    raw = df.copy()
    notes: list[str] = []

    table = qc.batch_table(raw)

    # Cadence, two ways. The wall-clock estimate (acq_seconds / n_samples over the
    # real batches) includes the host share of each acquire, so on the 2026-08-14
    # run it reads 269 us against a true 240 us -- a 12% stretch that would go
    # straight into the time axis, the rate and the frequency scale of the ASD.
    # The readout quantum recovered from the value quantisation is exact, so use
    # it whenever it is available and keep the wall-clock number as a fallback.
    wall_cadence = qc.fpga_cadence_seconds(raw, table)
    cadence, cadence_source = wall_cadence, "wall-clock"
    quantum = recover_readout_quantum(raw["peak_01"].to_numpy(float))
    n_slots, slot_ratio = (np.nan, np.nan)
    if quantum is not None:
        n_slots, slot_ratio = infer_slots(quantum["D"], wall_cadence)
        cadence = readout_seconds(quantum["D"], n_slots)
        cadence_source = "readout quantum"

    time_s, segment = qc.retime(raw, table, cadence_s=cadence)
    raw["_retimed_s"] = time_s
    raw["_segment"] = segment

    stale_rows = qc.stale_sample_mask(raw, table)
    first_rows = qc.first_sample_mask(raw)

    drop = np.zeros(len(raw), dtype=bool)
    if drop_stale:
        drop |= stale_rows
    if drop_first_sample:
        drop |= first_rows
    raw["_dropped"] = drop

    clean = raw.loc[~drop].copy()
    clean["time_s"] = clean.pop("_retimed_s")
    clean["segment"] = clean.pop("_segment")
    clean = clean.drop(columns=["_dropped"])

    span = float(table.t_start[-1] + table.acq_seconds[-1] - table.t_start[0])
    report = {
        "n_batches": int(len(table.batch)),
        "n_stale_batches": int(table.stale.sum()),
        "stale_fraction": float(table.stale.mean()),
        "samples_recorded": int(len(raw)),
        "samples_kept": int(len(clean)),
        "first_sample_transients_dropped": int((first_rows & ~stale_rows).sum()),
        "cadence_us": float(cadence * 1e6),
        "cadence_source": cadence_source,
        "cadence_wallclock_us": float(wall_cadence * 1e6),
        "within_burst_rate_hz": float(1.0 / cadence),
        "recorded_rate_hz": float(len(raw) / span) if span > 0 else np.nan,
        "true_rate_hz": float(len(clean) / span) if span > 0 else np.nan,
        "duty_cycle": float(qc.duty_cycle(table, cadence)),
    }
    if quantum is not None:
        # per_rep=True does not average over reps, so D is treg alone and the
        # readout window follows directly -- unlike averaged mode, where only the
        # product treg*reps is identifiable.
        report["slots_per_rep"] = int(n_slots)
        report["tau_us"] = float(quantum["D"] / (FPGA_CLOCK_HZ / 1e6))
        report["host_share_of_burst"] = float(1.0 - cadence / wall_cadence)
    if report["n_stale_batches"]:
        notes.append(f"{report['stale_fraction']*100:.1f}% of batches were replays of the "
                     f"previous one. The recorded rate "
                     f"({report['recorded_rate_hz']:.0f} Hz) is not real; the honest "
                     f"figure is {report['true_rate_hz']:.0f} Hz "
                     f"({report['within_burst_rate_hz']:.0f} Hz inside a burst, "
                     f"{100*report['duty_cycle']:.0f}% duty cycle).")

    clean = _apply_despike(clean, raw, despike, report, notes, **despike_kwargs)

    fs = 1.0 / cadence
    if notch_mains:
        # Each burst is its own segment, so the notch has to be applied inside
        # one. A 500-rep burst at 240 us spans 120 ms, giving 8.3 Hz bins -- the
        # notch cannot be narrower than that, so it takes a wide bite out of the
        # spectrum. Only worth it for a long burst.
        seg = clean["segment"].to_numpy()
        shift = clean["peak_shift_kHz"].to_numpy(float).copy()
        for s_id in np.unique(seg):
            rows = np.where(seg == s_id)[0]
            if rows.size >= 32:
                shift[rows] = spec.notch(shift[rows], fs,
                                         [spec.MAINS_HZ * k for k in range(1, 9)],
                                         halfwidth_hz=max(1.5, 2.0 * fs / rows.size))
        clean["peak_shift_kHz"] = shift
        notes.append(f"mains comb notched per burst segment; the shortest segment "
                     f"gives {fs / clean.groupby('segment').size().min():.1f} Hz bins, "
                     f"so the notch is wide")
    return _finish("burst", raw, clean, report, notes, fs,
                   segment=clean["segment"].to_numpy(), source=source)


def process_stream(df: pd.DataFrame, *, despike="shift", detrend: bool = True,
                   detrend_window: int = 501, notch_mains: bool = False,
                   slots_per_rep: int = 2, source=None,
                   **despike_kwargs) -> TwoPointResult:
    """Streaming mode: verify continuity, then deal with the droop.

    There is nothing to drop here -- `stream()` cannot replay a batch, because it
    never re-arms anything mid-run. The checks below prove that on the data
    rather than assuming it, and the droop correction handles the one problem the
    mode does have: with no gap between acquires, the sample heats and the PL
    falls monotonically through the run.
    """
    raw = df.copy()
    notes: list[str] = []

    t = raw["time_s"].to_numpy(float)
    fs = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else float("nan")

    report = {"n_samples": len(raw), "duration_s": float(t[-1] - t[0]),
              "rate_hz": float(fs)}

    # --- continuity: the whole point of streaming is that the rep axis is exact
    if "rep_index" in raw.columns:
        ri = raw["rep_index"].to_numpy(int)
        step = int(raw["reps_averaged"].iloc[0]) if "reps_averaged" in raw.columns else 1
        gaps = np.diff(ri) != step
        report["rep_index_gaps"] = int(gaps.sum())
        report["reps_averaged"] = step
        report["reps_covered"] = int(ri[-1] - ri[0] + step)
        if gaps.any():
            notes.append(f"{int(gaps.sum())} discontinuities in rep_index -- reps were "
                         f"dropped, so the cadence timestamps are wrong after the first")

    peaks = qc.peak_columns(raw)
    dup = np.all(raw[peaks].to_numpy()[1:] == raw[peaks].to_numpy()[:-1], axis=1)
    report["duplicate_rows"] = int(dup.sum())

    # D = treg * reps_averaged here, so dividing by the host binning factor
    # recovers the readout window exactly.
    quantum = recover_readout_quantum(raw["peak_01"].to_numpy(float))
    if quantum is not None:
        step = int(report.get("reps_averaged", 1))
        report["readout_quantum_D"] = int(quantum["D"])
        report["tau_us"] = float(quantum["D"] / step / (FPGA_CLOCK_HZ / 1e6))
        report["cadence_us_exact"] = float(
            readout_seconds(quantum["D"] // step, slots_per_rep) * 1e6)
        report["rate_hz_exact"] = float(
            1.0 / (readout_seconds(quantum["D"] // step, slots_per_rep) * step))

    if "packet" in raw.columns:
        sizes = raw.groupby("packet").size()
        report["n_packets"] = int(len(sizes))
        report["rows_per_packet_median"] = float(sizes.median())

    # --- the droop
    pl = raw["peak_01"].to_numpy(float)
    n_edge = max(10, len(pl) // 50)
    start, end = float(np.median(pl[:n_edge])), float(np.median(pl[-n_edge:]))
    droop = (end - start) / start if start else float("nan")
    report["pl_droop_fraction"] = float(droop)
    if abs(droop) > 0.05:
        notes.append(f"PL {'fell' if droop < 0 else 'rose'} {abs(droop)*100:.0f}% over the "
                     f"run ({start:.0f} -> {end:.0f} ADC). Continuous illumination has no "
                     f"duty-cycle gap, so the sample heats. Mostly common-mode, but it "
                     f"walks the operating point off the calibrated slope.")

    clean = raw.copy()
    clean["segment"] = 0

    clean = _apply_despike(clean, raw, despike, report, notes, **despike_kwargs)

    if detrend:
        clean["peak_shift_kHz"] = detrend_series(clean["peak_shift_kHz"].to_numpy(),
                                                 detrend_window)
        report["detrend_window_samples"] = int(detrend_window)
        notes.append(f"shift high-passed at ~{fs / detrend_window:.1f} Hz "
                     f"(rolling-median window {detrend_window})")

    if notch_mains:
        clean["peak_shift_kHz"] = spec.notch(
            clean["peak_shift_kHz"].to_numpy(), fs,
            [spec.MAINS_HZ * k for k in range(1, 9)])
        notes.append("mains comb (60 Hz and 7 harmonics) notched")

    return _finish("stream", raw, clean, report, notes, fs, source=source)


_PIPELINES = {"averaged": process_averaged, "burst": process_burst,
              "stream": process_stream}


def process(csv_or_df, mode: Optional[str] = None, **kwargs) -> TwoPointResult:
    """Load a two-point CSV (or take a frame), detect the mode, and process it."""
    source = None
    if csv_or_df is None:
        # The notebook's Step 5x cells pass `<MODE>_LAST_CSV`, which is None when
        # the matching Step 4x cell never got as far as writing a file. Without
        # this the failure surfaced from inside pathlib as
        # "TypeError: argument should be a str or an os.PathLike object where
        # __fspath__ returns a str, not 'NoneType'", which says nothing about
        # what to do -- see the 2026-08-17 session, where Step 5B reported that
        # instead of "the burst run you are trying to analyse never finished".
        raise ValueError(
            "process() was given no run to analyse (csv_or_df is None). The "
            "acquisition cell that should have produced it did not finish, so "
            "there is no CSV. Re-run the Step 4 cell for this mode, or pass an "
            "explicit path to an older run.")
    if isinstance(csv_or_df, pd.DataFrame):
        df = csv_or_df
    else:
        source = Path(csv_or_df)
        if not source.exists():
            raise FileNotFoundError(f"no such two-point run: {source}")
        df = pd.read_csv(source)
    mode = mode or detect_mode(df)
    if mode not in _PIPELINES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(_PIPELINES)}")
    return _PIPELINES[mode](df, source=source, **kwargs)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", type=Path, nargs="+", help="two-point live/stream CSV(s)")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="where to write *_clean.csv and *_clean.png "
                         "(default: alongside the input)")
    ap.add_argument("--mode", choices=sorted(_PIPELINES), default=None,
                    help="override mode detection")
    ap.add_argument("--despike", choices=["shift", "channels", "none"],
                    default="shift",
                    help="'shift' (default) filters the differential; 'channels' "
                         "filters the parked channels independently and usually "
                         "raises sigma -- diagnostic only")
    ap.add_argument("--notch-mains", action="store_true",
                    help="notch 60 Hz and harmonics (offline diagnostic only)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    rows = []
    for path in args.csv:
        kwargs = {"notch_mains": args.notch_mains,
                  "despike": False if args.despike == "none" else args.despike}
        res = process(path, mode=args.mode, **kwargs)
        print(res.describe())
        print()

        outdir = args.outdir or path.parent
        outdir.mkdir(parents=True, exist_ok=True)
        out_csv = outdir / f"{path.stem}_clean.csv"
        res.clean.to_csv(out_csv, index=False)
        print(f"  wrote {out_csv}")
        if not args.no_plot:
            out_png = outdir / f"{path.stem}_clean.png"
            res.figure(out_png)
            print(f"  wrote {out_png}")
        print()
        rows.append(res.summary_row())

    if len(rows) > 1 and args.outdir:
        summary = args.outdir / "postprocess_summary.csv"
        pd.DataFrame(rows).to_csv(summary, index=False)
        print(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
