"""Real-time spike rejection for the multipoint lock-in live loop.

The live two-point lock-in time series produces occasional single-sample
spikes of order tens of ADC counts on top of a few-ADC baseline. They are
sign-symmetric and one (rarely two) batches wide -- a hardware-origin
per-channel glitch, not a physical magnetic-field change.

`HampelDespiker` is a causal (trailing-window) Hampel filter operating on
each parked-frequency channel independently. For every new sample x[n,k]
on channel k it computes the median m and MAD of the last `window`
*non-rejected* samples and rejects x if |x - m| > k_sigma * 1.4826 * MAD.
Rejected samples are replaced with m (the rolling median).

Until the window has at least `min_warmup` samples, an optional
single-batch baseline vector is used to hard-clamp obviously wild
samples so they do not poison the buffer.

Per-batch cost for 16 channels and W=7 is ~30-50 us in pure numpy --
negligible against the 113 ms FPGA acquisition budget.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_MAD_SCALE = 1.4826  # consistent estimator of sigma from MAD for gaussian noise


@dataclass
class HampelDespiker:
    n_channels: int
    window: int = 11
    k_sigma: float = 4.0
    min_warmup: int = 5
    baseline: Optional[np.ndarray] = None
    warmup_k_sigma: float = 6.0
    # Lower bound on the MAD-derived sigma estimate. With small windows the
    # MAD can transiently shrink to zero (e.g. when consecutive samples
    # happen to collide on integer ADC counts), so we floor it. The right
    # value is roughly the true baseline noise sigma. On the 2026-05-11
    # live run we measured per-channel sigma ~ 1.0-1.2 ADC, so 1.5 is a
    # safe floor that does not over-protect and still catches single-batch
    # MAD collapses. Override per setup if the noise floor differs.
    sigma_floor: float = 1.5
    _buffers: list = field(init=False, repr=False)
    _count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be >= 2")
        if self.min_warmup < 1 or self.min_warmup > self.window:
            raise ValueError("min_warmup must be in [1, window]")
        if self.sigma_floor < 0.0:
            raise ValueError("sigma_floor must be >= 0")
        self._buffers = [deque(maxlen=self.window) for _ in range(self.n_channels)]
        if self.baseline is not None:
            self.baseline = np.asarray(self.baseline, dtype=float)
            if self.baseline.shape != (self.n_channels,):
                raise ValueError(
                    f"baseline shape {self.baseline.shape} != ({self.n_channels},)"
                )

    @property
    def count(self) -> int:
        return self._count

    def update(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Process one batch.

        Parameters
        ----------
        x : np.ndarray, shape (n_channels,)
            Raw ADC values for this batch.

        Returns
        -------
        clean : np.ndarray, shape (n_channels,)
            Despiked ADC values. Equal to x where no spike was detected,
            equal to the rolling median (or baseline, during warm-up) where
            it was.
        flags : np.ndarray of bool, shape (n_channels,)
            True on channels that were rejected and replaced.
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n_channels,):
            raise ValueError(
                f"x shape {x.shape} != ({self.n_channels},)"
            )
        clean = x.copy()
        flags = np.zeros(self.n_channels, dtype=bool)

        for k in range(self.n_channels):
            buf = self._buffers[k]
            replaced = False

            if len(buf) >= self.min_warmup:
                arr = np.fromiter(buf, dtype=float, count=len(buf))
                med = float(np.median(arr))
                mad = float(np.median(np.abs(arr - med)))
                sigma = max(_MAD_SCALE * mad, self.sigma_floor)
                if sigma > 0.0 and abs(clean[k] - med) > self.k_sigma * sigma:
                    clean[k] = med
                    flags[k] = True
                    replaced = True
            elif self.baseline is not None:
                tol = self.warmup_k_sigma * max(abs(self.baseline[k]) * 0.01, 5.0)
                if abs(clean[k] - self.baseline[k]) > tol:
                    clean[k] = self.baseline[k]
                    flags[k] = True
                    replaced = True

            if not replaced:
                buf.append(float(clean[k]))

        self._count += 1
        return clean, flags
