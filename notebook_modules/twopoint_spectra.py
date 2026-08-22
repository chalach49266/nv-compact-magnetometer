"""Spectral analysis for two-point lock-in time series.

Every FFT in the two-point workflow goes through this module, so the notebook's
three acquisition modes cannot end up quoting spectra computed three different
ways. `twopoint_postprocess` calls it; the notebook analysis cells call it
through `twopoint_postprocess`.

Conventions, fixed once here
----------------------------
**One-sided PSD.** For a real signal sampled at `fs`, the power in [f, f+df) is
`P(f) df` with `P` defined only for `0 <= f <= fs/2`, and

    var(x) = integral of P(f) df    over 0 .. fs/2

so a white signal of standard deviation `sigma` has `P = sigma^2 / (fs/2)` and

    ASD = sqrt(P) = sigma * sqrt(2 / fs) = sigma * sqrt(2 * dt).

**Sensitivity.** `eta = sigma_B * sqrt(2 * dt)`, i.e. the white-noise ASD implied
by a measured standard deviation at a given sample spacing. This is the
convention Step 5 of the notebook has always printed. It is a factor sqrt(2)
larger than the `sigma * sqrt(dt)` used in the 2026-08-06 analysis document;
that document's numbers are restated in this convention in
`docs/2026-08-14_twopoint_methods/`.

Quoting `eta` from `sigma` is only meaningful if the trace is actually white.
`white_floor()` reads the flat part of the measured ASD instead and needs no such
assumption -- prefer it, and use the two together as a consistency check.

**Field units.** Peak shifts are recorded in kHz. The NV gyromagnetic ratio along
one axis is 28.024 kHz/uT (`gamma_nv_mhz_per_uT = 0.028024` in the calibration
JSON), so field-referred quantities are `kHz / 28.024` uT.

Aliasing
--------
The averaged mode runs at ~88 Hz, where 60 Hz mains folds to ~28 Hz -- into the
middle of the band, indistinguishable from signal and impossible to filter.
`alias_report()` detects this and is called automatically by the post-processing
pipelines. It is the main reason to prefer streaming for anything where mains
pickup matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# MHz per microtesla along one NV axis; matches TWOPOINT_CALIB["gamma_nv_mhz_per_uT"].
GAMMA_NV_MHZ_PER_UT = 0.028024

# Mains fundamental. Odd harmonics dominate in the measured spectra (61.6, 185,
# 308, 465 Hz on the 2026-08-14 stream run).
MAINS_HZ = 60.0


def khz_to_nt(shift_khz, gamma_mhz_per_ut: float = GAMMA_NV_MHZ_PER_UT):
    """Peak shift in kHz -> equivalent field along the tracked NV axis, in nT."""
    return np.asarray(shift_khz, dtype=float) / (gamma_mhz_per_ut * 1e3) * 1e3


def sensitivity_nt_rthz(sigma_khz: float, dt_s: float,
                        gamma_mhz_per_ut: float = GAMMA_NV_MHZ_PER_UT) -> float:
    """White-noise sensitivity `eta = sigma_B * sqrt(2 dt)`, in nT/sqrt(Hz).

    Valid only where the trace is white; cross-check against `white_floor()`.
    """
    if not np.isfinite(dt_s) or dt_s <= 0:
        return float("nan")
    return float(abs(khz_to_nt(sigma_khz, gamma_mhz_per_ut)) * np.sqrt(2.0 * dt_s))


# --------------------------------------------------------------------------- #
# PSD / ASD
# --------------------------------------------------------------------------- #

def _detrend(x: np.ndarray, how: str) -> np.ndarray:
    if how in (None, "none", False):
        return x
    if how in ("constant", "mean", True):
        return x - x.mean()
    if how == "linear":
        t = np.arange(x.size, dtype=float)
        return x - np.polyval(np.polyfit(t, x, 1), t)
    raise ValueError(f"unknown detrend {how!r}; expected none/constant/linear")


def welch_psd(values, fs: float, nseg: int = 8, detrend: str = "linear",
              overlap: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """One-sided PSD by Welch averaging, in (units of `values`)^2 per Hz.

    Splitting into `nseg` overlapping Hann-windowed segments trades frequency
    resolution for a stable estimate: a single periodogram of N points has 100%
    standard error at every bin, which is useless for reading a noise floor.

    Non-finite samples make the whole estimate undefined, so they are dropped
    first; if that leaves gaps in time the caller should be using
    `segmented_psd` instead.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 16:
        raise ValueError(f"need at least 16 finite samples, got {x.size}")

    nseg = max(1, int(nseg))
    step_div = 1.0 + (nseg - 1) * (1.0 - overlap)
    nperseg = max(16, int(x.size / step_div))
    nperseg = min(nperseg, x.size)
    step = max(1, int(nperseg * (1.0 - overlap)))

    window = np.hanning(nperseg)
    # sum(w^2) normalisation keeps the estimate unbiased for the window's
    # power loss; the leading 2 makes it one-sided.
    norm = fs * np.sum(window ** 2)

    starts = list(range(0, x.size - nperseg + 1, step))
    if not starts:
        starts = [0]
    acc = np.zeros(nperseg // 2 + 1)
    for s in starts:
        seg = _detrend(x[s:s + nperseg], detrend)
        acc += np.abs(np.fft.rfft(seg * window)) ** 2
    psd = 2.0 * acc / (len(starts) * norm)
    # DC and Nyquist are not doubled -- they have no negative-frequency twin.
    psd[0] /= 2.0
    if nperseg % 2 == 0:
        psd[-1] /= 2.0
    return np.fft.rfftfreq(nperseg, 1.0 / fs), psd


def asd_nt(shift_khz, fs: float, nseg: int = 8, detrend: str = "linear",
           gamma_mhz_per_ut: float = GAMMA_NV_MHZ_PER_UT
           ) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude spectral density of a peak-shift trace, in nT/sqrt(Hz)."""
    f, psd = welch_psd(khz_to_nt(shift_khz, gamma_mhz_per_ut), fs,
                       nseg=nseg, detrend=detrend)
    return f, np.sqrt(psd)


def fft_amplitude_nt(shift_khz, fs: float, detrend: str = "linear",
                     window: str = "hann",
                     gamma_mhz_per_ut: float = GAMMA_NV_MHZ_PER_UT
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Plain single-sided FFT amplitude spectrum, in nT.

    This is the raw transform of the whole record -- no segmenting, no Welch
    averaging -- so the y axis is an *amplitude in nT*, not a density in
    nT/sqrt(Hz), and it answers a different question from `asd_nt`:

        asd_nt            how much broadband noise per root hertz. Averaged over
                          `nseg` segments, so the floor is stable but a coherent
                          line is smeared across the segment bandwidth and its
                          height depends on the window length.
        fft_amplitude_nt  how many nT a periodic component actually is. One bin
                          per fs/N, every bin kept, so a 60 Hz line reads its
                          true amplitude -- at the cost of ~100% scatter on the
                          broadband floor, which is what a single periodogram
                          always has.

    Use the ASD to quote a sensitivity, this to identify and size a line.

    Scaling is amplitude-correct for a sinusoid: the one-sided spectrum is
    2/sum(w) * |rfft(w*x)|, so a pure tone of amplitude A reads A regardless of
    record length or window. DC and Nyquist are not doubled.
    """
    x = np.asarray(khz_to_nt(shift_khz, gamma_mhz_per_ut), dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return np.array([]), np.array([])
    x = _detrend(x, detrend)

    if window == "hann":
        w = np.hanning(n)
    elif window in (None, "none", "boxcar"):
        w = np.ones(n)
    else:
        raise ValueError(f"unknown window {window!r}")

    spec = np.fft.rfft(x * w)
    amp = 2.0 * np.abs(spec) / np.sum(w)
    amp[0] /= 2.0
    if n % 2 == 0:
        amp[-1] /= 2.0
    return np.fft.rfftfreq(n, d=1.0 / fs), amp


def segmented_fft_amplitude_nt(shift_khz, segment, fs: float,
                               detrend: str = "linear", window: str = "hann",
                               gamma_mhz_per_ut: float = GAMMA_NV_MHZ_PER_UT
                               ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Plain FFT amplitude of burst data: one transform per burst, then averaged.

    The reason `fft_amplitude_nt` is not simply pointed at a burst record is that
    the record is not contiguous -- the host spends ~26% of the wall clock
    reconfiguring between bursts, and transforming across that dead time puts a
    frequency scale on a gap. This routine does the same thing `segment_psd` does
    for the Welch panel: each burst is transformed on its own, never across a
    boundary, and only the resulting spectra are combined.

    Amplitude scaling is the same as `fft_amplitude_nt`, so a tone still reads its
    true size in nT. Bursts are combined in quadrature (rms of the per-burst
    amplitudes), which leaves a coherent line at its amplitude while averaging the
    broadband hash down by sqrt(n_bursts).

    The cost is resolution: the bin spacing is `fs / n`, where `n` is the *shortest*
    burst, not the whole run. A 500-rep burst at 240 us spans 120 ms and therefore
    gives 8.3 Hz bins -- enough to see a 60 Hz line, not enough to separate it from
    52 or 68 Hz. Use `gapfilled_fft_amplitude_nt` when the resolution matters.
    """
    x = np.asarray(khz_to_nt(shift_khz, gamma_mhz_per_ut), dtype=float)
    seg = np.asarray(segment)
    if seg.size != x.size:
        raise ValueError(f"segment has {seg.size} entries for {x.size} samples")

    ids = [s for s in np.unique(seg) if np.all(np.isfinite(x[seg == s]))]
    info = {"n_segments": len(ids), "n_per_segment": 0, "bin_hz": float("nan")}
    if not ids:
        return np.array([]), np.array([]), info

    n = min(int((seg == s).sum()) for s in ids)
    if n < 8:
        return np.array([]), np.array([]), info

    if window == "hann":
        w = np.hanning(n)
    elif window in (None, "none", "boxcar"):
        w = np.ones(n)
    else:
        raise ValueError(f"unknown window {window!r}")

    acc = np.zeros(n // 2 + 1)
    for s in ids:
        y = _detrend(x[seg == s][:n], detrend)
        a = 2.0 * np.abs(np.fft.rfft(y * w)) / w.sum()
        a[0] /= 2.0
        if n % 2 == 0:
            a[-1] /= 2.0
        acc += a ** 2
    amp = np.sqrt(acc / len(ids))

    info.update(n_per_segment=int(n), bin_hz=float(fs / n))
    return np.fft.rfftfreq(n, d=1.0 / fs), amp, info


def gapfilled_fft_amplitude_nt(shift_khz, time_s, fs: float,
                               detrend: str = "linear", window: str = "hann",
                               gamma_mhz_per_ut: float = GAMMA_NV_MHZ_PER_UT
                               ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Plain FFT of a burst record over its *true* time axis, gaps left empty.

    Every sample is placed in the grid slot its timestamp says it belongs in, and
    the dead time between bursts stays as zeros instead of being closed up. The
    frequency scale is therefore the real one -- a 60 Hz line lands at 60 Hz --
    and the resolution is that of the whole run (`fs / N` over the full duration),
    not of one burst.

    Normalisation divides by the window sum over the *occupied* slots only, so the
    duty-cycle hole does not scale a tone down: a real 500 nT line still reads
    500 nT, to within the usual Hann scalloping loss (up to 15% for a line that
    falls between two bins -- measured 454 nT for an injected 500 nT tone at
    77 Hz on the 2026-08-19 burst run, at the correct 77.04 Hz).

    What it costs, and it is not nothing: multiplying by the burst pattern
    convolves the spectrum with the pattern's transform, so every real line comes
    with a comb of ghosts at +/- k x (burst repetition rate), typically a few Hz
    apart. The tallest member of each comb sits at the true frequency; the ghosts
    are an artefact of the duty cycle, not signal. Read line *positions* here, and
    cross-check a line's size against the per-burst panel.
    """
    x = np.asarray(khz_to_nt(shift_khz, gamma_mhz_per_ut), dtype=float)
    t = np.asarray(time_s, dtype=float)
    if t.size != x.size:
        raise ValueError(f"time_s has {t.size} entries for {x.size} samples")

    good = np.isfinite(x) & np.isfinite(t)
    x, t = x[good], t[good]
    info = {"n_samples": int(x.size), "n_grid": 0, "fill_fraction": float("nan"),
            "bin_hz": float("nan"), "burst_rate_hz": float("nan"),
            "ghost_fraction": float("nan")}
    if x.size < 8:
        return np.array([]), np.array([]), info

    x = _detrend(x, detrend)

    # Slot index on the exact FPGA cadence. `time_s` is already rebuilt on that
    # cadence by `burst_qc.retime`, so rounding is exact inside a burst and only
    # the burst *starts* (wall-clock) carry any rounding at all.
    k = np.rint((t - t[0]) * fs).astype(np.int64)
    n = int(k.max()) + 1
    if n <= 0 or n > 50_000_000:
        return np.array([]), np.array([]), info

    total = np.bincount(k, weights=x, minlength=n)
    count = np.bincount(k, minlength=n).astype(float)
    filled = count > 0
    y = np.zeros(n)
    y[filled] = total[filled] / count[filled]

    if window == "hann":
        w = np.hanning(n)
    elif window in (None, "none", "boxcar"):
        w = np.ones(n)
    else:
        raise ValueError(f"unknown window {window!r}")

    norm = float(w[filled].sum())
    if norm <= 0:
        return np.array([]), np.array([]), info

    amp = 2.0 * np.abs(np.fft.rfft(y * w)) / norm
    amp[0] /= 2.0
    if n % 2 == 0:
        amp[-1] /= 2.0

    # Size the artefact rather than just warning about it. Gating a signal with a
    # periodic burst pattern of duty d replicates every line at +/- k x f_rep with
    # relative height |sinc(k d)|; at 87% duty the first ghost is ~15% of its
    # parent, at 50% it is 64% and the panel needs reading with care.
    duty = float(filled.mean())
    n_bursts = int(np.count_nonzero(np.diff(t) > 1.5 / fs)) + 1
    span = float(t[-1] - t[0])
    info.update(n_grid=int(n), fill_fraction=duty, bin_hz=float(fs / n),
                burst_rate_hz=float(n_bursts / span) if span > 0 and n_bursts > 1
                              else float("nan"),
                ghost_fraction=float(abs(np.sinc(duty))) if n_bursts > 1 else 0.0)
    return np.fft.rfftfreq(n, d=1.0 / fs), amp, info


def segmented_asd_nt(shift_khz, segment, fs: float, detrend: bool = True,
                     gamma_mhz_per_ut: float = GAMMA_NV_MHZ_PER_UT
                     ) -> tuple[np.ndarray, np.ndarray]:
    """ASD of burst data, averaged per contiguous segment and never across gaps.

    Delegates to `burst_qc.segment_psd`, which is also what the burst cleaner
    uses, so the spectrum in the report and the spectrum in the cleaner's plot
    are the same estimator.
    """
    from burst_qc import segment_psd

    f, psd = segment_psd(khz_to_nt(shift_khz, gamma_mhz_per_ut),
                         np.asarray(segment), fs, detrend=detrend)
    return f, np.sqrt(psd)


def white_floor(f: np.ndarray, asd: np.ndarray,
                band: tuple[float, float]) -> float:
    """Median ASD inside a frequency band -- the measured noise floor.

    Median rather than mean so a mains line inside the band does not lift it.
    Returns NaN if the band is empty, which happens when a run is too short or
    the rate too low to reach it.
    """
    lo, hi = band
    m = (f > lo) & (f < hi) & np.isfinite(asd)
    return float(np.median(asd[m])) if m.any() else float("nan")


def default_floor_band(fs: float) -> tuple[float, float]:
    """A band that is above the drift knee and below Nyquist, for this rate.

    Drift dominates below ~10 Hz in every run measured so far, so the floor is
    read from 10 Hz upward, stopping short of Nyquist where the anti-alias
    behaviour of the accumulator is not characterised.
    """
    return (10.0, max(12.0, 0.4 * fs))


# --------------------------------------------------------------------------- #
# Spectral lines and aliasing
# --------------------------------------------------------------------------- #

def alias_of(line_hz: float, fs: float) -> float:
    """Where a tone at `line_hz` appears after sampling at `fs`.

    Returns `line_hz` itself when it is below Nyquist. Otherwise it folds into
    [0, fs/2] -- and once folded there is no way to tell it from a real signal
    at the folded frequency.
    """
    if fs <= 0:
        return float("nan")
    r = float(line_hz) % fs
    return r if r <= fs / 2 else fs - r


@dataclass
class AliasWarning:
    line_hz: float
    alias_hz: float

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return (f"{self.line_hz:.0f} Hz aliases to {self.alias_hz:.1f} Hz "
                f"and cannot be separated from signal there")


def alias_report(fs: float, lines: Sequence[float] = (MAINS_HZ, 2 * MAINS_HZ,
                                                      3 * MAINS_HZ)
                 ) -> list[AliasWarning]:
    """Which of `lines` are above Nyquist at this rate, and where they land."""
    out = []
    for line in lines:
        if line > fs / 2:
            out.append(AliasWarning(float(line), alias_of(line, fs)))
    return out


def find_lines(f: np.ndarray, asd: np.ndarray, base: float = MAINS_HZ,
               n_harmonics: int = 8, halfwidth_hz: float = 2.0,
               floor_band: Optional[tuple[float, float]] = None) -> pd.DataFrame:
    """Amplitude of a harmonic comb, each line relative to the local floor.

    `excess` is the line peak divided by the broadband floor, so a value near 1
    means "no line here". Only harmonics below Nyquist are reported; the caller
    should pair this with `alias_report` to learn about the ones that are not.
    """
    floor = white_floor(f, asd, floor_band or default_floor_band(2 * f[-1]))
    rows = []
    for k in range(1, int(n_harmonics) + 1):
        target = base * k
        if target >= f[-1]:
            break
        m = np.abs(f - target) <= halfwidth_hz
        if not m.any():
            continue
        peak = float(np.max(asd[m]))
        rows.append({
            "harmonic": k,
            "frequency_hz": float(f[m][np.argmax(asd[m])]),
            "asd_nt_rthz": peak,
            "excess_over_floor": peak / floor if np.isfinite(floor) and floor > 0
                                 else float("nan"),
        })
    return pd.DataFrame(rows)


def notch(values, fs: float, lines: Iterable[float], halfwidth_hz: float = 1.5,
          detrend: str = "constant"):
    """Zero out narrow frequency bands -- an ideal (non-causal) comb notch.

    Offline only. It assumes the whole record is stationary and it will ring at
    the edges, so it is for reading a floor underneath mains pickup, not for
    producing a trace to quote transients from. A real-time notch belongs in the
    acquisition path, not here.

    Non-finite samples are passed through unchanged.
    """
    x = np.asarray(values, dtype=float)
    good = np.isfinite(x)
    if good.sum() < 16:
        return x.copy()

    y = x[good]
    mean = y.mean()
    y = _detrend(y, detrend)

    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(y.size, 1.0 / fs)
    for line in lines:
        target = alias_of(float(line), fs)
        spec[np.abs(freqs - target) <= halfwidth_hz] = 0.0

    out = x.copy()
    out[good] = np.fft.irfft(spec, n=y.size) + (mean if detrend != "none" else 0.0)
    return out


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #

@dataclass
class SpectrumSummary:
    """Everything a report needs to say about one trace's noise."""

    fs_hz: float
    sigma_khz: float
    eta_nt_rthz: float
    floor_nt_rthz: float
    floor_band_hz: tuple[float, float]
    lines: pd.DataFrame
    aliases: list[AliasWarning] = field(default_factory=list)
    f: Optional[np.ndarray] = None
    asd: Optional[np.ndarray] = None
    # Plain single-sided FFT of the same trace, amplitude in nT. Kept alongside
    # the Welch ASD because the two answer different questions -- see
    # `fft_amplitude_nt`.
    fft_f: Optional[np.ndarray] = None
    fft_amp_nt: Optional[np.ndarray] = None
    # How that plain FFT was computed, for the figure to label itself honestly:
    # "record" (contiguous run), "per-burst", "gap-filled" or "concatenated".
    fft_kind: str = "record"
    fft_note: str = ""

    def to_dict(self) -> dict:
        strongest = (self.lines.sort_values("excess_over_floor").iloc[-1].to_dict()
                     if len(self.lines) else {})
        return {
            "fs_hz": self.fs_hz,
            "sigma_khz": self.sigma_khz,
            "eta_nt_rthz": self.eta_nt_rthz,
            "floor_nt_rthz": self.floor_nt_rthz,
            "floor_band_lo_hz": self.floor_band_hz[0],
            "floor_band_hi_hz": self.floor_band_hz[1],
            "strongest_line_hz": strongest.get("frequency_hz", float("nan")),
            "strongest_line_excess": strongest.get("excess_over_floor", float("nan")),
            "n_aliased_lines": len(self.aliases),
        }

    def describe(self) -> str:
        out = [
            f"  sample rate        : {self.fs_hz:.1f} Hz",
            f"  sigma(peak shift)  : {self.sigma_khz:.1f} kHz "
            f"({khz_to_nt(self.sigma_khz):.2f} nT)",
            f"  eta = sigma*sqrt(2dt): {self.eta_nt_rthz:.1f} nT/sqrt(Hz)",
            f"  measured ASD floor : {self.floor_nt_rthz:.1f} nT/sqrt(Hz) "
            f"({self.floor_band_hz[0]:.0f}-{self.floor_band_hz[1]:.0f} Hz)",
        ]
        if np.isfinite(self.floor_nt_rthz) and np.isfinite(self.eta_nt_rthz) \
                and self.floor_nt_rthz > 0:
            ratio = self.eta_nt_rthz / self.floor_nt_rthz
            if ratio > 1.5:
                out.append(f"  -> eta is {ratio:.1f}x the flat floor: the trace is not "
                           f"white, drift dominates sigma. Trust the floor.")
        if len(self.lines):
            strong = self.lines[self.lines["excess_over_floor"] > 2.0]
            for _, r in strong.iterrows():
                out.append(f"  line               : {r['frequency_hz']:.1f} Hz, "
                           f"{r['excess_over_floor']:.1f}x the floor")
        for a in self.aliases:
            out.append(f"  ALIAS WARNING      : {a}")
        return "\n".join(out)


def summarise(shift_khz, fs: float, segment=None, nseg: int = 8,
              detrend: str = "linear", floor_band: Optional[tuple] = None,
              gamma_mhz_per_ut: float = GAMMA_NV_MHZ_PER_UT,
              burst_fft: str = "gapfill", time_s=None) -> SpectrumSummary:
    """Full spectral summary of a peak-shift trace.

    Pass `segment` (from `burst_qc.retime`) for burst data so the Welch transform
    never straddles the dead time between bursts.

    `burst_fft` picks how the *plain* FFT panel is built for burst data, where a
    single transform of the whole record is not defined. It is ignored for the
    contiguous modes, which always get the straight transform of the run.

        "gapfill"   (default) one transform on the true time axis with the dead
                    time left as zeros. Needs `time_s`. Full-run resolution and
                    true line frequencies, at the price of a ghost comb at the
                    burst repetition rate whose height the panel reports.
        "segments"  one plain FFT per burst, combined in quadrature. No gap
                    artefacts at all, but the bins are as wide as one burst is
                    short (4-8 Hz), and a bin that wide collects so much
                    broadband noise that a real line barely clears it. Use it to
                    confirm a line gapfill already found.
        "concat"    the naive transform that pretends the bursts are contiguous.
                    DIAGNOSTIC ONLY: removing the gaps rescales the time axis by
                    the duty cycle and breaks the phase at every burst boundary,
                    so lines land at the wrong frequency and are smeared. It is
                    here because it is what "just FFT the column" does.
        "none"      leave the panel empty, the behaviour before 2026-08-19.
    """
    x = np.asarray(shift_khz, dtype=float)
    finite = x[np.isfinite(x)]
    sigma = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")

    if segment is not None:
        f, asd = segmented_asd_nt(x, segment, fs, gamma_mhz_per_ut=gamma_mhz_per_ut)
    else:
        f, asd = asd_nt(x, fs, nseg=nseg, detrend=detrend,
                        gamma_mhz_per_ut=gamma_mhz_per_ut)

    band = tuple(floor_band) if floor_band else default_floor_band(fs)

    # The plain transform, alongside the Welch estimate. A contiguous record is
    # transformed whole; a burst record cannot be, so `burst_fft` says which of
    # the three well-defined substitutes to use instead of dropping the panel.
    fft_kind, fft_note = "record", ""
    if segment is None:
        fft_f, fft_amp = fft_amplitude_nt(x, fs, detrend=detrend,
                                          gamma_mhz_per_ut=gamma_mhz_per_ut)
        if fft_f is not None and len(fft_f) > 1:
            fft_note = f"{finite.size} samples, {fs / finite.size:.3g} Hz bins"
    else:
        mode = str(burst_fft or "none").lower()
        if mode in ("segment", "segments", "per_burst", "per-burst"):
            fft_f, fft_amp, info = segmented_fft_amplitude_nt(
                x, segment, fs, detrend=detrend, gamma_mhz_per_ut=gamma_mhz_per_ut)
            fft_kind = "per-burst"
            fft_note = (f"{info['n_segments']} bursts x {info['n_per_segment']} "
                        f"samples, quadrature-averaged, {info['bin_hz']:.3g} Hz bins")
        elif mode in ("gapfill", "gap_fill", "gap-fill", "zerofill", "zero-fill"):
            if time_s is None:
                raise ValueError("burst_fft='gapfill' needs time_s (the retimed "
                                 "axis from burst_qc.retime)")
            fft_f, fft_amp, info = gapfilled_fft_amplitude_nt(
                x, time_s, fs, detrend=detrend, gamma_mhz_per_ut=gamma_mhz_per_ut)
            fft_kind = "gap-filled"
            fft_note = (f"{info['n_samples']} samples on a {info['n_grid']}-slot "
                        f"grid ({100 * info['fill_fraction']:.0f}% filled), "
                        f"{info['bin_hz']:.3g} Hz bins")
            if np.isfinite(info["burst_rate_hz"]):
                fft_note += (f"; each line ghosts at +/-{info['burst_rate_hz']:.2g} Hz "
                             f"x k at ~{100 * info['ghost_fraction']:.0f}% height")
        elif mode in ("concat", "concatenate", "naive"):
            fft_f, fft_amp = fft_amplitude_nt(x, fs, detrend=detrend,
                                              gamma_mhz_per_ut=gamma_mhz_per_ut)
            fft_kind = "concatenated"
            fft_note = (f"{finite.size} samples with the gaps closed up -- the "
                        f"frequency axis is stretched by 1/duty and the phase "
                        f"breaks at every burst edge. Do not quote frequencies.")
        elif mode in ("none", "off", "false"):
            fft_f, fft_amp = None, None
            fft_kind = "none"
        else:
            raise ValueError(f"unknown burst_fft {burst_fft!r}; expected "
                             f"segments/gapfill/concat/none")
        if fft_f is not None and len(fft_f) < 2:
            fft_note = "burst too short to transform"

    return SpectrumSummary(
        fs_hz=float(fs),
        sigma_khz=sigma,
        eta_nt_rthz=sensitivity_nt_rthz(sigma, 1.0 / fs, gamma_mhz_per_ut),
        floor_nt_rthz=white_floor(f, asd, band),
        floor_band_hz=band,
        lines=find_lines(f, asd, floor_band=band),
        aliases=alias_report(fs),
        f=f,
        asd=asd,
        fft_f=fft_f,
        fft_amp_nt=fft_amp,
        fft_kind=fft_kind,
        fft_note=fft_note,
    )
