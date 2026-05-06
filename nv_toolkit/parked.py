from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import ParkedFrequencySeries


@dataclass(frozen=True)
class ParkedPairMetric:
    save_id: str
    timestamp_epoch_s: float
    block_index: int
    frequency_minus_mhz: float
    frequency_plus_mhz: float
    intensity_minus: float
    intensity_plus: float
    difference: float
    normalized_difference: float


def compute_pair_metrics(series: ParkedFrequencySeries) -> list[ParkedPairMetric]:
    metrics: list[ParkedPairMetric] = []
    for row_index, save_id in enumerate(series.save_ids):
        row = np.asarray(series.intensities[row_index], dtype=float)
        timestamp = float(series.timestamps_epoch_s[row_index])
        for block in np.unique(series.block_indices):
            mask = series.block_indices == block
            block_freqs = series.freqs_mhz[mask]
            block_values = row[mask]
            point_order = np.argsort(series.point_in_block[mask])
            block_freqs = block_freqs[point_order]
            block_values = block_values[point_order]
            if len(block_values) != 2:
                raise ValueError(f"Block {block} does not contain exactly two parked points")
            minus_value = float(block_values[0])
            plus_value = float(block_values[1])
            denominator = plus_value + minus_value
            normalized_difference = 0.0 if denominator == 0.0 else (plus_value - minus_value) / denominator
            metrics.append(
                ParkedPairMetric(
                    save_id=save_id,
                    timestamp_epoch_s=timestamp,
                    block_index=int(block),
                    frequency_minus_mhz=float(block_freqs[0]),
                    frequency_plus_mhz=float(block_freqs[1]),
                    intensity_minus=minus_value,
                    intensity_plus=plus_value,
                    difference=plus_value - minus_value,
                    normalized_difference=normalized_difference,
                )
            )
    return metrics
