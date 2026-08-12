"""Quality control for two-point lock-in live CSVs, especially burst mode.

Both `scripts/analyze_twopoint_session.py` and `scripts/clean_burst_lockin.py`
build on this module so the analysis and the cleaner cannot disagree about what
counts as a bad sample.

Background -- what goes wrong in burst mode
-------------------------------------------
`Modules/Twopoint_Lockin_module.ipynb` Step 4 calls `prog.acquire(per_rep=True)`
in a loop. On the 2026-08-06 data every second call comes back in ~13 ms instead
of the ~430 ms the FPGA actually needs, and 95-97% of its rows are bit-identical
to the previous call's rows: the accumulated buffer is read before the new run
has refilled it, so the batch is a replay of the previous one. Three artefacts
follow, and this module detects all three:

1. **Stale batches.** Half the recorded samples are duplicates. They inflate the
   apparent rate ~2x and make every real glitch appear twice, roughly one burst
   period apart -- which is what reads as "periodic noise" in the plots.
2. **A first-sample transient.** Row 0 of a stale batch is not a copy; it reads
   ~4.6% high in `z_minus`, i.e. ~+1500 kHz (~+54 uT). This is the large,
   strictly periodic spike, one per batch pair.
3. **A broken time axis.** The live cell spreads a burst's samples across the
   *measured* acquire window, so a stale 13 ms batch and a real 430 ms batch
   both get 1000 timestamps -- a 33x compression alternating batch to batch.
   The FPGA cadence is in fact constant, so timestamps should be rebuilt from it.

Nothing here is specific to the 2026-08-06 run; the detectors are threshold-based
and report what they found, so a clean file passes through untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# A batch whose rows repeat the previous batch this closely is a stale read.
# Observed duplicate fractions are 0.95-0.97 for stale batches and 0.06-0.12 for
# real ones, so anything in the middle of that gap separates them cleanly.
STALE_DUPLICATE_FRACTION = 0.90

# Second, independent staleness test: a batch cannot contain more FPGA pulse work
# than its own wall-clock duration. A real burst spends `n_samples * cadence`
# seconds on the FPGA; a stale read returns in a fraction of that. This catches
# batches the duplicate test misses -- on 2026-08-06, batch 24 is 82% duplicate
# (below the fraction threshold, because the run's one CSV-flush stall shifted
# the alignment) but returned in 12.9 ms against 424 ms of claimed pulse work.
STALE_DURATION_FRACTION = 0.5


def peak_columns(df: pd.DataFrame) -> list[str]:
    """The raw per-frequency ADC columns, in frequency order (`peak_01`, ...)."""
    cols = [c for c in df.columns if c.startswith("peak_") and c[5:].isdigit()]
    return sorted(cols)


def is_burst(df: pd.DataFrame) -> bool:
    """True if the file has more than one sample per acquire call."""
    return "batch" in df.columns and len(df) > df["batch"].nunique()


@dataclass
class BatchTable:
    """Per-acquire-call summary of a live CSV.

    Attributes
    ----------
    batch : (n_batches,) int      batch index as recorded
    n_samples : (n_batches,) int  rows contributed by that batch
    acq_seconds : (n_batches,)    measured wall-clock duration of the acquire call
    t_start : (n_batches,)        reconstructed start of the acquire window
    dup_fraction : (n_batches,)   fraction of rows identical to the previous batch
    stale : (n_batches,) bool     dup_fraction >= STALE_DUPLICATE_FRACTION
    """

    batch: np.ndarray
    n_samples: np.ndarray
    acq_seconds: np.ndarray
    t_start: np.ndarray
    dup_fraction: np.ndarray
    stale: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "batch": self.batch,
                "n_samples": self.n_samples,
                "acq_seconds": self.acq_seconds,
                "t_start": self.t_start,
                "dup_fraction": self.dup_fraction,
                "stale": self.stale,
            }
        )


def batch_table(df: pd.DataFrame, threshold: float = STALE_DUPLICATE_FRACTION) -> BatchTable:
    """Summarise a live CSV batch by batch and flag stale reads.

    `t_start` is reconstructed rather than read, because the live cell records
    only sample timestamps. It stamps sample k of an n-sample batch at
    `t_start + (k + 0.5) / n * acq`, so the first sample sits half a slot in:

        t_start = t_first - (0.5 / n) * acq

    That reduces to `t_mid - acq/2` for the averaged mode (n = 1), so one
    expression covers both modes.
    """
    if "batch" not in df.columns:
        raise ValueError("live CSV has no 'batch' column")

    grouped = df.groupby("batch", sort=True)
    batches = np.asarray(grouped.size().index, dtype=int)
    n_samples = grouped.size().to_numpy(dtype=int)
    acq = grouped["acq_seconds"].first().to_numpy(dtype=float)
    t_first = grouped["time_s"].first().to_numpy(dtype=float)
    t_start = t_first - (0.5 / n_samples) * acq

    cols = peak_columns(df)
    blocks = [g[cols].to_numpy(dtype=float) for _, g in grouped]

    dup = np.zeros(len(batches), dtype=float)
    for i in range(1, len(blocks)):
        prev, cur = blocks[i - 1], blocks[i]
        n = min(len(prev), len(cur))
        if n:
            dup[i] = float((prev[:n] == cur[:n]).all(axis=1).mean())

    # Two independent tests, either of which condemns a batch. See the module
    # docstring and the STALE_* constants for why one is not enough.
    cadence = _cadence_from_acq(acq, n_samples)
    too_fast = acq < STALE_DURATION_FRACTION * n_samples * cadence
    stale = (dup >= threshold) | too_fast

    return BatchTable(
        batch=batches,
        n_samples=n_samples,
        acq_seconds=acq,
        t_start=t_start,
        dup_fraction=dup,
        stale=stale,
    )


def _cadence_from_acq(acq: np.ndarray, n_samples: np.ndarray) -> float:
    """Per-rep period estimated from wall-clock durations alone.

    Real batches give `acq / n_samples == cadence`; stale ones give something much
    smaller, and nothing ever gives something larger except host jitter. So the
    upper quartile of `acq / n_samples` lands inside the real cluster whether the
    file is half stale (2026-08-06 burst runs), fully clean, or averaged mode
    (where every batch is real and the quartile is just the batch period).
    """
    per_rep = np.asarray(acq, dtype=float) / np.maximum(n_samples, 1)
    return float(np.percentile(per_rep, 75))


def stale_sample_mask(df: pd.DataFrame, table: Optional[BatchTable] = None,
                      threshold: float = STALE_DUPLICATE_FRACTION) -> np.ndarray:
    """Row mask marking every sample that belongs to a stale batch."""
    table = batch_table(df, threshold=threshold) if table is None else table
    stale_batches = set(table.batch[table.stale].tolist())
    return df["batch"].isin(stale_batches).to_numpy()


def first_sample_mask(df: pd.DataFrame) -> np.ndarray:
    """Row mask marking sample 0 of every batch (the post-idle transient)."""
    if "sample" in df.columns:
        return (df["sample"].to_numpy() == 0)
    return np.zeros(len(df), dtype=bool)


def fpga_cadence_seconds(df: pd.DataFrame, table: Optional[BatchTable] = None) -> float:
    """Seconds per rep, measured from the batches that really ran.

    A real burst acquire spends essentially all of its wall-clock time on FPGA
    pulse work, so `acq_seconds / n_samples` over the non-stale batches recovers
    the per-rep period directly. Verified against the program's own model on the
    2026-08-06 data: 500-rep bursts give 429.4 us/rep and 1000-rep bursts
    428.6 us/rep, against a predicted 435 us.
    """
    table = batch_table(df) if table is None else table
    good = ~table.stale & (table.n_samples > 1)
    if not good.any():
        good = table.n_samples > 1
    if not good.any():
        raise ValueError("no burst batches in this file; cadence is undefined")
    return float(np.median(table.acq_seconds[good] / table.n_samples[good]))


def retime(df: pd.DataFrame, table: Optional[BatchTable] = None,
           cadence_s: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild timestamps on the true FPGA cadence and label contiguous segments.

    Within a burst the FPGA runs at a fixed period, so sample k of a batch that
    began at `t_start` belongs at `t_start + k * cadence`. Between bursts the
    host spends time reconfiguring and no samples exist -- that dead time is a
    real gap, so each batch becomes its own segment. Callers should never filter
    or transform across a segment boundary.

    Returns
    -------
    time_s : (n_rows,) float   cadence-based timestamps
    segment : (n_rows,) int    contiguous-acquisition id, one per surviving batch
    """
    table = batch_table(df) if table is None else table
    cadence = fpga_cadence_seconds(df, table) if cadence_s is None else float(cadence_s)

    start_by_batch = dict(zip(table.batch.tolist(), table.t_start.tolist()))
    seg_by_batch = {b: i for i, b in enumerate(table.batch.tolist())}

    batch_of_row = df["batch"].to_numpy()
    sample_of_row = (df["sample"].to_numpy() if "sample" in df.columns
                     else np.zeros(len(df), dtype=int))

    starts = np.array([start_by_batch[b] for b in batch_of_row], dtype=float)
    segment = np.array([seg_by_batch[b] for b in batch_of_row], dtype=int)
    return starts + sample_of_row * cadence, segment


def duty_cycle(table: BatchTable, cadence_s: float) -> float:
    """Fraction of wall-clock time the FPGA spends producing real samples."""
    live = float((table.n_samples[~table.stale] * cadence_s).sum())
    span = float(table.t_start[-1] + table.acq_seconds[-1] - table.t_start[0])
    return live / span if span > 0 else float("nan")


def segment_psd(values: np.ndarray, segment: np.ndarray, fs: float,
                detrend: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """One-sided PSD averaged over equal-length segments, never across gaps.

    Averaging periodograms across the dead time between bursts would fold the
    gap into the spectrum as a spurious low-frequency line, so each contiguous
    segment is transformed on its own and only then averaged.
    """
    lengths = [int((segment == s).sum()) for s in np.unique(segment)]
    n = min(lengths)
    if n < 8:
        raise ValueError(f"segments too short for a PSD (shortest is {n} samples)")

    window = np.hanning(n)
    norm = fs * (window ** 2).sum()
    acc = None
    count = 0
    for s in np.unique(segment):
        x = values[segment == s][:n].astype(float)
        if not np.all(np.isfinite(x)):
            continue
        if detrend:
            x = x - x.mean()
        power = np.abs(np.fft.rfft(x * window)) ** 2 / norm
        acc = power if acc is None else acc + power
        count += 1
    if count == 0:
        raise ValueError("no finite segments to transform")
    return np.fft.rfftfreq(n, 1.0 / fs), acc / count


def block_average_sigma(values: np.ndarray, segment: np.ndarray,
                        block_sizes: Sequence[int]) -> pd.DataFrame:
    """Standard deviation after averaging in blocks of N, within segments only.

    White noise gives sigma proportional to 1/sqrt(N). Any flattening shows where
    correlated drift takes over, which is the quantity that decides whether
    raising the rep count actually buys sensitivity.
    """
    rows = []
    for n in block_sizes:
        chunks = []
        for s in np.unique(segment):
            x = values[segment == s]
            x = x[np.isfinite(x)]
            k = len(x) // n
            if k:
                chunks.append(x[: k * n].reshape(k, n).mean(axis=1))
        if not chunks:
            continue
        pooled = np.concatenate(chunks)
        if pooled.size < 2:
            continue
        rows.append({"block": n, "n_points": pooled.size, "sigma": float(pooled.std(ddof=1))})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["sigma_white_model"] = out["sigma"].iloc[0] / np.sqrt(out["block"])
        out["excess"] = out["sigma"] / out["sigma_white_model"]
    return out
