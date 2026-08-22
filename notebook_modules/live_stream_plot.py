"""Live display for a streamed lock-in run: peak shift over time, plus a live FFT.

Why this is a module and not another block of notebook cell
-----------------------------------------------------------
Two notebooks need the identical picture -- `Twopoint_Lockin_module` with one
tracked peak, `Fourpoint_Lockin_module` with three channels (peak 1, peak 2 and
the splitting between them) -- and the awkward parts are the same for both.

What the awkward parts are:

* **A redraw is expensive and the board does not wait for it.** `stream()` starts
  the tProc once and it free-runs; the host only drains the queue. Host work
  therefore costs *buffer occupancy*, never sample rate -- until the buffer
  overflows, at which point the run ABORTS rather than slows. Measured on the
  2026-08-20 multipoint runs: a live cell spent 25.2% of a 30 s run inside
  redraws, changed the sample rate by 0.007%, and pushed peak buffer occupancy
  from 10.3% to 30.1%. So the rule is not "be fast", it is "be bounded".
* **The two-point stream runs at 4166.7 Hz.** A 20 s window is 83k points. Drawing
  83k points per redraw costs more than everything else in the loop put together,
  so the time panel is fed a block-MEAN decimated to `display_hz` (default 100 Hz)
  rather than the raw samples. The mean is an anti-aliasing filter, so the 3 Hz
  low-pass drawn over it is the same curve it would be at full rate. The file on
  disk is untouched: this decimation exists only inside the picture.
* **The FFT wants the opposite.** Its whole point is the lines above the signal
  band -- mains at 60 Hz, the 61.6 Hz cryostat pump -- so it is computed from a
  separate full-rate ring buffer holding the last `fft_window_s` seconds. Short
  window, full rate: 4 s at 4166.7 Hz is 16k points, 0.25 Hz bins, Nyquist
  2083 Hz, and about 2 ms to transform.

Usage
-----
::

    live = LiveShiftFFT(["peak shift"], nt_per_khz=1/gamma, lowpass_hz=3.0)
    live.start()
    for packet in prog.stream(total_reps=n):
        ...
        live.add(t_array, shift_khz_array)      # shape (n,) or (n, n_channels)
        if time_to_redraw:
            live.redraw()
    live.final_figure()
"""
from __future__ import annotations

import numpy as np

# Same conversion the rest of the pipeline uses: MHz per microtesla, one NV axis.
GAMMA_NV_MHZ_PER_UT = 0.028024
NT_PER_KHZ_ONE_AXIS = 1.0 / GAMMA_NV_MHZ_PER_UT      # 35.68 nT per kHz of shift

_DEFAULT_COLOURS = ("crimson", "tab:blue", "tab:green", "tab:orange", "tab:purple")


def amp_spectrum(x, dt):
    """Plain single-sided amplitude spectrum, in the units of `x`.

    Hann window, normalised by the window sum, so a pure tone of amplitude A
    reads A at its own bin. This is the same transform the drone deck's
    `make_figs.amp_spectrum` uses -- not a PSD, and not the Welch average: a
    live panel wants the actual line height, not a smoothed noise density.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 8:
        return np.array([0.0]), np.array([0.0])
    w = np.hanning(n)
    A = 2.0 * np.abs(np.fft.rfft(x * w)) / w.sum()
    A[0] /= 2.0
    if n % 2 == 0:
        A[-1] /= 2.0
    return np.fft.rfftfreq(n, dt), A


def lowpass(x, fc_hz, fs_hz, order=4):
    """Zero-phase Butterworth low-pass. None when the series cannot carry it."""
    from scipy.signal import butter, filtfilt

    x = np.asarray(x, dtype=float)
    padlen = 3 * (2 * order + 1)
    if (not fc_hz or not np.isfinite(fs_hz) or fs_hz <= 0
            or fc_hz >= 0.45 * fs_hz or x.size <= padlen + 1):
        return None
    good = np.isfinite(x)
    if good.sum() < 2:
        return None
    if not good.all():
        idx = np.arange(x.size)
        x = x.copy()
        x[~good] = np.interp(idx[~good], idx[good], x[good])
    b, a = butter(order, fc_hz / (fs_hz / 2.0), btype="low")
    return filtfilt(b, a, x)


def top_line(f, A, fmin_hz, fmax_hz, *, min_excess=4.0, floor_bins=101):
    """The tallest REAL line in `A`, or None.

    Not `argmax`. On a lock-in trace the per-bin noise is Rayleigh-distributed, so
    the largest bin of a few thousand is about 4x the mean noise amplitude *with no
    line present at all* -- picking `argmax` reports a noise spike and its "height"
    wanders every refresh. Measured on a synthetic 4166.7 Hz run with 150 kHz/sample
    of white noise: argmax returned 700.0 Hz at 9.0 kHz while the only real line was
    60.0 Hz at 3.0 kHz.

    So the line is chosen by its excess over a RUNNING median of the spectrum -- the
    local noise floor, which also tracks the 1/f rise at the bottom of the band --
    and it is reported only when that excess clears `min_excess`. The returned
    amplitude is still the honest line height `A[j]`, not the excess.
    """
    band = (f >= fmin_hz) & (f <= fmax_hz)
    if not band.any():
        return None
    idx = np.flatnonzero(band)
    a = A[idx]
    w = int(min(floor_bins, max(11, (a.size // 8) | 1)))
    try:
        from scipy.ndimage import median_filter
        # "reflect", NOT "nearest". With "nearest" the window at either end fills
        # with copies of the single edge bin, so the floor there IS that bin and its
        # own excess goes to 1 while its neighbours' explodes -- measured as a
        # phantom "line" at 2083 Hz, 5x floor, on pure noise.
        floor = median_filter(a, size=w, mode="reflect")
    except Exception:                       # SciPy is always present here; belt and braces
        floor = np.full_like(a, np.median(a))
    floor = np.maximum(floor, 1e-30)
    excess = a / floor
    j = int(np.argmax(excess))
    if excess[j] < min_excess:
        return None
    return {"freq_hz": float(f[idx[j]]), "amp_khz": float(a[j]),
            "excess": float(excess[j]), "floor_khz": float(floor[j])}


def _decimate_for_display(f, A, max_points=2500):
    """Thin a spectrum for drawing by taking the MAX of each block.

    A line is one or two bins wide, so plain striding drops it half the time while
    the block maximum always keeps it. Only the picture is thinned; every number
    quoted comes from the full spectrum.
    """
    n = f.size
    if n <= max_points:
        return f, A
    k = int(np.ceil(n / max_points))
    m = (n // k) * k
    fb = f[:m].reshape(-1, k)
    ab = A[:m].reshape(-1, k)
    j = np.argmax(ab, axis=1)
    return fb[np.arange(fb.shape[0]), j], ab[np.arange(ab.shape[0]), j]


class LiveShiftFFT:
    """Two live panels: peak shift against time, and the plain FFT of it.

    Panel 1 -- peak shift in kHz on the left, the equivalent field in nT on the
    right (a `secondary_yaxis`, so it stays locked through any autoscale). Raw as
    the shadow, the `lowpass_hz` zero-phase filter as the line: the drone-deck
    convention.

    Panel 2 -- the plain single-sided amplitude spectrum of the last
    `fft_window_s` seconds at FULL rate, with the tallest line above `fft_min_hz`
    marked and its height printed in kHz and nT.

    Parameters
    ----------
    channels : sequence of str
        One label per channel. Two-point passes one; four-point passes three
        (peak 1, peak 2, splitting).
    nt_per_khz : float
        Field per kHz for the right-hand axis. `NT_PER_KHZ_ONE_AXIS` for a single
        tracked peak. NOTE for four-point: the splitting moves at 2*gamma, so on
        an axis built for one peak that channel reads TWICE the true field. That
        is the same convention `fourpoint_runner.plot_run` uses, and
        `channel_field_scale` lets the printed numbers correct for it even though
        the shared axis cannot.
    channel_field_scale : sequence of float, optional
        Per-channel multiplier applied to the *printed* field numbers only
        (1.0 for a single peak, 0.5 for a splitting channel). Never applied to the
        plotted kHz values, which are always what was measured.
    """

    def __init__(self, channels, *, nt_per_khz=NT_PER_KHZ_ONE_AXIS, window_s=20.0,
                 display_hz=100.0, lowpass_hz=3.0, lowpass_order=4,
                 fft_window_s=16.0, fft_min_hz=0.5, fft_fmax_hz=None,
                 fft_min_excess=6.0, channel_field_scale=None, field_label=None, title="",
                 colours=None, figsize=(13, 8), fs_hint=None):
        self.channels = list(channels)
        self.n_ch = len(self.channels)
        self.nt_per_khz = float(nt_per_khz)
        self.window_s = None if window_s is None else float(window_s)
        self.display_hz = float(display_hz)
        self.lowpass_hz = lowpass_hz
        self.lowpass_order = int(lowpass_order)
        self.fft_window_s = float(fft_window_s)
        self.fft_min_hz = float(fft_min_hz)
        self.fft_fmax_hz = fft_fmax_hz
        self.fft_min_excess = float(fft_min_excess)
        self.field_scale = (np.ones(self.n_ch) if channel_field_scale is None
                            else np.asarray(channel_field_scale, dtype=float))
        self.field_label = field_label or "equivalent ΔB (nT)"
        self.title = title
        self.colours = list(colours or _DEFAULT_COLOURS)[:self.n_ch] or ["crimson"]
        self.figsize = figsize
        self.fs = float(fs_hint) if fs_hint else float("nan")

        # Display history: block-mean of the incoming samples down to display_hz.
        self._dt: list[float] = []
        self._dv: list[np.ndarray] = []
        self._carry_t = np.empty(0)
        self._carry_v = np.empty((0, self.n_ch))
        self._bin = 0                      # samples per display point, set on first add

        # Full-rate ring for the FFT, as a preallocated array rather than a deque:
        # at 4166.7 Hz a 16 s window is 66k rows, and rebuilding that from a deque
        # of row arrays on every redraw costs more than the transform it feeds.
        self._ring: np.ndarray | None = None
        self._ring_n = 0
        self._ring_pos = 0
        self._ring_filled = 0

        self.n_samples = 0
        self.last = {}
        self.fig = None
        self._handle = None

    # ------------------------------------------------------------------ setup
    def start(self, display_handle=True):
        """Build the figure and register it for live updates."""
        import matplotlib.pyplot as plt

        plt.ion()
        self.fig, axes = plt.subplots(2, 1, figsize=self.figsize,
                                      constrained_layout=True)
        self.fig.set_label("live")
        self.ax_t, self.ax_f = axes

        self._raw_lines, self._flt_lines = [], []
        for name, colour in zip(self.channels, self.colours):
            raw, = self.ax_t.plot([], [], lw=0.5, color=colour, alpha=0.22)
            flt, = self.ax_t.plot([], [], lw=1.5, color=colour, label=name)
            self._raw_lines.append(raw)
            self._flt_lines.append(flt)
        self.ax_t.axhline(0.0, color="0.4", lw=0.7)
        self.ax_t.set(xlabel="time (s)", ylabel="peak shift (kHz)")
        self.ax_t.grid(alpha=0.3)
        self.ax_t.legend(fontsize=8, loc="upper right", ncol=max(1, self.n_ch))
        self._sec_t = self.ax_t.secondary_yaxis(
            "right", functions=(lambda k: k * self.nt_per_khz,
                                lambda n: n / self.nt_per_khz))
        self._sec_t.set_ylabel(self.field_label)

        self._fft_lines = []
        for name, colour in zip(self.channels, self.colours):
            line, = self.ax_f.semilogy([], [], lw=0.8, color=colour, label=name)
            self._fft_lines.append(line)
        self._peak_marker, = self.ax_f.plot([], [], "o", ms=7, mfc="none",
                                            mew=1.8, color="k")
        self._peak_text = self.ax_f.text(
            0.985, 0.93, "", transform=self.ax_f.transAxes, ha="right", va="top",
            fontsize=11, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.9))
        self.ax_f.set(xlabel="frequency (Hz)", ylabel="amplitude (kHz)")
        self.ax_f.grid(alpha=0.3, which="both")
        self._sec_f = self.ax_f.secondary_yaxis(
            "right", functions=(lambda k: k * self.nt_per_khz,
                                lambda n: n / self.nt_per_khz))
        self._sec_f.set_ylabel("amplitude (nT)")
        if self.title:
            self.fig.suptitle(self.title)

        if display_handle:
            try:
                from IPython.display import display
                self._handle = display(self.fig, display_id=True)
            except Exception:
                self._handle = None
        return self.fig

    # ------------------------------------------------------------- ring buffer
    def _ring_push(self, v):
        n = v.shape[0]
        if n >= self._ring_n:
            self._ring[:] = v[-self._ring_n:]
            self._ring_pos, self._ring_filled = 0, self._ring_n
            return
        end = self._ring_pos + n
        if end <= self._ring_n:
            self._ring[self._ring_pos:end] = v
        else:
            k = self._ring_n - self._ring_pos
            self._ring[self._ring_pos:] = v[:k]
            self._ring[:end - self._ring_n] = v[k:]
        self._ring_pos = end % self._ring_n
        self._ring_filled = min(self._ring_n, self._ring_filled + n)

    def _ring_view(self):
        if self._ring is None or self._ring_filled == 0:
            return None
        if self._ring_filled < self._ring_n:
            return self._ring[:self._ring_filled]
        return np.concatenate([self._ring[self._ring_pos:], self._ring[:self._ring_pos]])

    # ------------------------------------------------------------------- feed
    def add(self, t, values):
        """Take `n` new samples: `t` shape (n,), `values` (n,) or (n, n_channels).

        Cheap by construction -- this runs inside the drain loop, once per packet.
        Everything expensive waits for `redraw()`, which the caller rate-limits.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        v = np.asarray(values, dtype=float)
        if v.ndim == 1:
            v = v.reshape(-1, 1)
        if v.shape[0] != t.shape[0]:
            raise ValueError(f"t has {t.shape[0]} samples, values has {v.shape[0]}")
        if v.shape[1] != self.n_ch:
            raise ValueError(f"expected {self.n_ch} channels, got {v.shape[1]}")
        self.n_samples += t.shape[0]

        # Sample rate from the timestamps themselves; they are the FPGA cadence.
        if t.size > 1:
            dt = float(np.median(np.diff(t)))
            if dt > 0:
                self.fs = 1.0 / dt
        if self._ring is None and np.isfinite(self.fs):
            self._ring_n = max(256, int(round(self.fft_window_s * self.fs)))
            self._ring = np.zeros((self._ring_n, self.n_ch), dtype=float)
        if self._ring is not None:
            self._ring_push(v)

        # Block-mean down to display_hz, vectorised. Leftovers carry to the next
        # packet so a block never straddles a packet boundary -- the boundary is an
        # artefact of when the host drained, nothing physical.
        if not self._bin and np.isfinite(self.fs):
            self._bin = max(1, int(round(self.fs / self.display_hz)))
        k = self._bin or 1
        if self._carry_t.size:
            t = np.concatenate([self._carry_t, t])
            v = np.concatenate([self._carry_v, v], axis=0)
        n_full = (t.shape[0] // k) * k
        self._carry_t, self._carry_v = t[n_full:], v[n_full:]
        if n_full:
            self._dt.extend(t[:n_full].reshape(-1, k).mean(axis=1).tolist())
            self._dv.extend(v[:n_full].reshape(-1, k, self.n_ch).mean(axis=1))

        if self.window_s and self._dt:
            cut = self._dt[-1] - self.window_s
            while len(self._dt) > 2 and self._dt[0] < cut:
                self._dt.pop(0)
                self._dv.pop(0)

    # ----------------------------------------------------------------- redraw
    def redraw(self):
        """Recompute both panels and push the figure. THE expensive call."""
        if self.fig is None or not self._dt:
            return
        t = np.asarray(self._dt, dtype=float)
        V = np.asarray(self._dv, dtype=float)
        fs_disp = 1.0 / np.median(np.diff(t)) if t.size > 2 else float("nan")

        stats = {}
        lo_all, hi_all = [], []
        for c in range(self.n_ch):
            y = V[:, c]
            yf = lowpass(y, self.lowpass_hz, fs_disp, self.lowpass_order)
            self._raw_lines[c].set_data(t, y)
            if yf is None:
                self._flt_lines[c].set_data(t, y)
                self._raw_lines[c].set_alpha(0.0)
                ref = y
            else:
                self._flt_lines[c].set_data(t, yf)
                self._raw_lines[c].set_alpha(0.22)
                ref = yf
            # Scale to the filtered trace: a 3 Hz line inside a wide noise band is
            # otherwise flat on the axis, and the shadow is meant to run off it.
            edge = int(np.ceil(fs_disp / self.lowpass_hz)) if (
                yf is not None and self.lowpass_hz and np.isfinite(fs_disp)) else 0
            core = ref[edge:-edge] if (edge and ref.size > 2 * edge + 2) else ref
            lo_all.append(float(np.nanmin(core)))
            hi_all.append(float(np.nanmax(core)))
            stats[self.channels[c]] = {
                "sigma_khz": float(np.nanstd(core)),
                "sigma_nt": float(np.nanstd(core) * self.nt_per_khz * self.field_scale[c]),
                "last_khz": float(ref[-1]),
            }
        lo, hi = min(lo_all), max(hi_all)
        pad = max(0.25 * (hi - lo), 1e-9)
        self.ax_t.set_xlim(t[0], t[-1])
        self.ax_t.set_ylim(lo - pad, hi + pad)
        self.ax_t.set_title(
            f"{self.lowpass_hz:g} Hz low-pass (line) over raw block-averaged to "
            f"{fs_disp:.0f} Hz (shadow)   |   "
            + "   ".join(f"{k}: σ {v['sigma_khz']:.2f} kHz = {v['sigma_nt']:.0f} nT"
                         for k, v in stats.items()),
            fontsize=9)

        # --- FFT panel: full rate, short window ---
        top = None
        R = self._ring_view()
        if R is not None and R.shape[0] >= 64 and np.isfinite(self.fs):
            fmax = self.fft_fmax_hz or self.fs / 2.0
            ymin, ymax = np.inf, 0.0
            for c in range(self.n_ch):
                f, A = amp_spectrum(R[:, c], 1.0 / self.fs)
                m = f <= fmax
                self._fft_lines[c].set_data(*_decimate_for_display(f[m], A[m]))
                band = (f >= self.fft_min_hz) & m
                hit = top_line(f, A, self.fft_min_hz, fmax,
                               min_excess=self.fft_min_excess)
                if hit is not None and (top is None or hit["excess"] > top["excess"]):
                    top = dict(hit, channel=self.channels[c],
                               amp_nt=hit["amp_khz"] * self.nt_per_khz * self.field_scale[c])
                if band.any():
                    pos = A[m][A[m] > 0]
                    if pos.size:
                        ymin = min(ymin, float(np.percentile(pos, 5)))
                        ymax = max(ymax, float(A[band].max()))
            self.ax_f.set_xlim(0, fmax)
            if np.isfinite(ymin) and ymax > 0:
                self.ax_f.set_ylim(max(ymin * 0.5, ymax * 1e-5), ymax * 3.0)
            self.ax_f.set_title(
                f"plain FFT of the last {len(R) / self.fs:.1f} s at {self.fs:.0f} Hz "
                f"({self.fs / len(R):.2f} Hz bins)", fontsize=9)
            if top is not None:
                self._peak_marker.set_data([top["freq_hz"]], [top["amp_khz"]])
                self._peak_text.set_text(
                    f"top line  {top['freq_hz']:8.2f} Hz\n"
                    f"          {top['amp_khz']:8.3f} kHz\n"
                    f"          {top['amp_nt']:8.1f} nT\n"
                    f"          {top['excess']:8.1f}x floor"
                    + (f"\n          {top['channel']}" if self.n_ch > 1 else ""))
            else:
                self._peak_marker.set_data([], [])
                self._peak_text.set_text(
                    f"no line above\n{self.fft_min_excess:.0f}x the local\nnoise floor")
        self.last = {"stats": stats, "top_line": top, "fs_hz": self.fs,
                     "display_hz": fs_disp, "n_samples": self.n_samples}

        self.fig.canvas.draw_idle()
        if self._handle is not None:
            self._handle.update(self.fig)

    # ------------------------------------------------------------------ close
    def close(self):
        import matplotlib.pyplot as plt
        plt.ioff()

    def summary(self) -> str:
        if not self.last:
            return "no samples displayed"
        out = [f"Live display: {self.last['n_samples']} samples at "
               f"{self.last['fs_hz']:.1f} Hz "
               f"(shown block-averaged to {self.last['display_hz']:.0f} Hz)"]
        for name, s in self.last["stats"].items():
            out.append(f"  {name:<28s} sigma {s['sigma_khz']:8.3f} kHz = "
                       f"{s['sigma_nt']:8.1f} nT   last {s['last_khz']:+8.3f} kHz")
        t = self.last["top_line"]
        if t:
            out.append(f"  top FFT line: {t['freq_hz']:.2f} Hz at {t['amp_khz']:.3f} kHz "
                       f"= {t['amp_nt']:.1f} nT" + (f"  ({t['channel']})" if self.n_ch > 1 else ""))
        return "\n".join(out)
