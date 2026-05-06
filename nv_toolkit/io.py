from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FREQUENCY_COLUMN_CANDIDATES = (
    "frequency_mhz",
    "freq_mhz",
    "frequency",
    "freq",
)
INTENSITY_COLUMN_CANDIDATES = (
    "normalized_intensity",
    "intensity",
    "signal",
    "norm_signal",
    "norm_intensity",
)
GROUP_COLUMN_CANDIDATES = (
    "save_id",
    "spectrum_id",
    "scan_id",
    "run_id",
)
ORDER_COLUMN_CANDIDATES = (
    "point_index",
    "frequency_mhz",
    "frequency",
)


@dataclass(frozen=True)
class SpectrumBatch:
    freqs_mhz: np.ndarray
    spectra: np.ndarray
    spectrum_ids: list[str]
    metadata: list[dict[str, str]]


@dataclass(frozen=True)
class ParkedFrequencySeries:
    freqs_mhz: np.ndarray
    intensities: np.ndarray
    save_ids: list[str]
    timestamps_epoch_s: np.ndarray
    block_indices: np.ndarray
    point_in_block: np.ndarray
    metadata: list[dict[str, str]]


def _normalized(text: str) -> str:
    return "".join(ch for ch in text.strip().lower() if ch.isalnum())


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {_normalized(name): name for name in fieldnames}
    for candidate in candidates:
        match = normalized.get(_normalized(candidate))
        if match is not None:
            return match
    return None


def _load_numeric_matrix(path: Path) -> SpectrumBatch:
    matrix = np.loadtxt(path, delimiter=",", dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two numeric columns")
    freqs = np.asarray(matrix[:, 0], dtype=float)
    spectra = np.asarray(matrix[:, 1:].T, dtype=float)
    spectrum_ids = [f"spectrum_{idx:03d}" for idx in range(spectra.shape[0])]
    metadata = [{"source": str(path), "format": "numeric_matrix"} for _ in spectrum_ids]
    return SpectrumBatch(freqs_mhz=freqs, spectra=spectra, spectrum_ids=spectrum_ids, metadata=metadata)


def _grouped_rows_to_batch(
    rows: list[dict[str, str]],
    *,
    freq_col: str,
    intensity_col: str,
    group_col: str | None,
    order_col: str | None,
    source: Path,
) -> SpectrumBatch:
    grouped: dict[str, list[dict[str, str]]] = {}
    ordered_group_ids: list[str] = []

    for row in rows:
        group_id = row[group_col] if group_col is not None else "spectrum_000"
        if group_id not in grouped:
            grouped[group_id] = []
            ordered_group_ids.append(group_id)
        grouped[group_id].append(row)

    spectra: list[np.ndarray] = []
    metadata: list[dict[str, str]] = []
    base_freqs: np.ndarray | None = None
    for group_id in ordered_group_ids:
        group_rows = grouped[group_id]
        if order_col is not None:
            group_rows = sorted(group_rows, key=lambda item: float(item[order_col]))

        freqs = np.asarray([float(item[freq_col]) for item in group_rows], dtype=float)
        values = np.asarray([float(item[intensity_col]) for item in group_rows], dtype=float)
        if base_freqs is None:
            base_freqs = freqs
        elif not np.allclose(base_freqs, freqs, atol=1e-9, rtol=0.0):
            raise ValueError(f"{source} contains grouped spectra with mismatched frequency axes")
        spectra.append(values)
        metadata.append(dict(group_rows[0]))

    if base_freqs is None:
        raise ValueError(f"{source} did not contain any usable spectrum rows")

    return SpectrumBatch(
        freqs_mhz=base_freqs,
        spectra=np.asarray(spectra, dtype=float),
        spectrum_ids=ordered_group_ids,
        metadata=metadata,
    )


def load_spectrum_csv(path: str | Path, *, group_column: str | None = None) -> SpectrumBatch:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        has_header = csv.Sniffer().has_header(sample)
        if not has_header:
            return _load_numeric_matrix(path)

        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    freq_col = _pick_column(fieldnames, FREQUENCY_COLUMN_CANDIDATES)
    intensity_col = _pick_column(fieldnames, INTENSITY_COLUMN_CANDIDATES)
    if freq_col is None or intensity_col is None:
        return _load_numeric_matrix(path)

    resolved_group_col = group_column or _pick_column(fieldnames, GROUP_COLUMN_CANDIDATES)
    order_col = _pick_column(fieldnames, ORDER_COLUMN_CANDIDATES)
    return _grouped_rows_to_batch(
        rows,
        freq_col=freq_col,
        intensity_col=intensity_col,
        group_col=resolved_group_col,
        order_col=order_col,
        source=path,
    )


def load_parked_frequency_csv(path: str | Path) -> ParkedFrequencySeries:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    required = {
        "save_id",
        "timestamp_epoch_s",
        "block_index",
        "point_in_block",
        "frequency_mhz",
        "normalized_intensity",
    }
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"{path} is missing parked-frequency columns: {sorted(missing)}")

    batch = _grouped_rows_to_batch(
        rows,
        freq_col="frequency_mhz",
        intensity_col="normalized_intensity",
        group_col="save_id",
        order_col="point_index",
        source=path,
    )
    timestamps = np.asarray([float(item["timestamp_epoch_s"]) for item in batch.metadata], dtype=float)

    first_group_rows = sorted(
        [row for row in rows if row["save_id"] == batch.spectrum_ids[0]],
        key=lambda item: float(item["point_index"]),
    )
    block_indices = np.asarray([int(item["block_index"]) for item in first_group_rows], dtype=int)
    point_in_block = np.asarray([int(item["point_in_block"]) for item in first_group_rows], dtype=int)

    return ParkedFrequencySeries(
        freqs_mhz=batch.freqs_mhz,
        intensities=batch.spectra,
        save_ids=batch.spectrum_ids,
        timestamps_epoch_s=timestamps,
        block_indices=block_indices,
        point_in_block=point_in_block,
        metadata=batch.metadata,
    )
