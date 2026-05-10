from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .model import named_nv_axes, odmr_resonances
from .two_point import (
    TwoPointCalibration,
    _resonance_gradient_mhz_per_mT,
    estimate_delta_B_projection_mT,
    estimate_delta_B_vector_mT,
)
from .io import load_parked_frequency_csv, load_spectrum_csv


def build_parked_pair_calibration(
    reference_freqs_mhz: np.ndarray,
    reference_spectrum: np.ndarray,
    *,
    parked_frequencies_mhz: tuple[float, float],
    B_bias_mT: np.ndarray,
    transition_index: int | None = None,
    nv_axes_preset: str = "qdm",
) -> TwoPointCalibration:
    freqs = np.asarray(reference_freqs_mhz, dtype=float)
    spectrum = np.asarray(reference_spectrum, dtype=float)
    if freqs.ndim != 1 or spectrum.ndim != 1 or freqs.shape != spectrum.shape:
        raise ValueError("Reference frequency and spectrum arrays must be 1D with matching shape")

    # Normalize the reference spectrum so off-resonance bins sit at 1.0.
    # This corrects for DC baseline drift between the full-scan and later parked runs.
    # Use a data-driven top-quantile approach rather than a model-based mask so
    # the correction is robust to any linewidth, contrast, or baseline level.
    off_res_threshold = float(np.percentile(spectrum, 80))
    off_res_mask = spectrum >= off_res_threshold
    if int(np.sum(off_res_mask)) >= 10:
        ref_baseline = float(np.mean(spectrum[off_res_mask]))
        if 0.9 < ref_baseline < 1.1:
            spectrum = spectrum / ref_baseline

    f_minus_mhz, f_plus_mhz = sorted((float(parked_frequencies_mhz[0]), float(parked_frequencies_mhz[1])))
    midpoint_mhz = 0.5 * (f_minus_mhz + f_plus_mhz)
    nv_axes = named_nv_axes(nv_axes_preset)
    resonances = np.sort(odmr_resonances(np.asarray(B_bias_mT, dtype=float), nv_axes=nv_axes).ravel())
    resolved_transition = int(np.argmin(np.abs(resonances - midpoint_mhz))) if transition_index is None else int(transition_index)

    slope_axis = np.gradient(spectrum, freqs)
    baseline_minus = float(np.interp(f_minus_mhz, freqs, spectrum))
    baseline_plus = float(np.interp(f_plus_mhz, freqs, spectrum))
    slope_minus = float(np.interp(f_minus_mhz, freqs, slope_axis))
    slope_plus = float(np.interp(f_plus_mhz, freqs, slope_axis))
    gradient = _resonance_gradient_mhz_per_mT(np.asarray(B_bias_mT, dtype=float), resolved_transition, nv_axes=nv_axes)

    return TwoPointCalibration(
        B_bias_mT=np.asarray(B_bias_mT, dtype=float),
        transition_index=resolved_transition,
        resonance_frequency_mhz=float(resonances[resolved_transition]),
        f_minus_mhz=f_minus_mhz,
        f_plus_mhz=f_plus_mhz,
        baseline_minus=baseline_minus,
        baseline_plus=baseline_plus,
        slope_minus_per_mhz=slope_minus,
        slope_plus_per_mhz=slope_plus,
        df_dB_vector_mhz_per_mT=gradient,
    )


def build_blockwise_calibrations(
    reference_freqs_mhz: np.ndarray,
    reference_spectrum: np.ndarray,
    parked_freqs_mhz: np.ndarray,
    block_indices: np.ndarray,
    point_in_block: np.ndarray,
    *,
    B_bias_mT: np.ndarray,
    nv_axes_preset: str = "qdm",
) -> dict[int, TwoPointCalibration]:
    calibrations: dict[int, TwoPointCalibration] = {}
    for block_index in np.unique(np.asarray(block_indices, dtype=int)):
        mask = np.asarray(block_indices, dtype=int) == int(block_index)
        block_freqs = np.asarray(parked_freqs_mhz, dtype=float)[mask]
        block_order = np.argsort(np.asarray(point_in_block, dtype=int)[mask])
        block_freqs = block_freqs[block_order]
        if len(block_freqs) != 2:
            raise ValueError(f"Block {block_index} must contain exactly two parked frequencies")
        calibrations[int(block_index)] = build_parked_pair_calibration(
            reference_freqs_mhz,
            reference_spectrum,
            parked_frequencies_mhz=(float(block_freqs[0]), float(block_freqs[1])),
            B_bias_mT=B_bias_mT,
            nv_axes_preset=nv_axes_preset,
        )
    return calibrations


def estimate_parked_series_fields(
    parked_intensities: np.ndarray,
    timestamps_epoch_s: np.ndarray,
    save_ids: list[str],
    block_indices: np.ndarray,
    point_in_block: np.ndarray,
    parked_freqs_mhz: np.ndarray,
    calibrations: dict[int, TwoPointCalibration],
    *,
    hyperfine_splitting_mhz: float | None = 2.16,
    nv_axes_preset: str = "qdm",
    search_radius_mT: float = 0.0,
    subtract_baseline: bool = True,
) -> list[dict[str, object]]:
    block_indices = np.asarray(block_indices, dtype=int)
    point_in_block = np.asarray(point_in_block, dtype=int)
    parked_freqs_mhz = np.asarray(parked_freqs_mhz, dtype=float)
    intensities = np.asarray(parked_intensities, dtype=float)
    timestamps = np.asarray(timestamps_epoch_s, dtype=float)
    rows: list[dict[str, object]] = []
    nv_axes = named_nv_axes(nv_axes_preset)

    # Fast-path (search_radius_mT == 0): vectorize over all timestamps and blocks.
    # Pre-sort each block's columns once; then the inner body is just arithmetic.
    if float(search_radius_mT) == 0.0:
        # Build per-block index arrays (sorted by point_in_block within each block)
        block_info: list[tuple[int, TwoPointCalibration, np.ndarray, np.ndarray]] = []
        for block_index, calibration in calibrations.items():
            mask = block_indices == int(block_index)
            order = np.argsort(point_in_block[mask])
            cols = np.flatnonzero(mask)[order]  # indices into the measurement columns
            if len(cols) != 2:
                raise ValueError(f"Block {block_index} does not contain exactly two parked values")
            block_info.append((int(block_index), calibration, cols, parked_freqs_mhz[cols]))

        for row_index, save_id in enumerate(save_ids):
            spectrum = intensities[row_index]
            timestamp = float(timestamps[row_index])
            for block_index, calibration, cols, block_freqs in block_info:
                s_minus = float(spectrum[cols[0]])
                s_plus = float(spectrum[cols[1]])
                # estimate_delta_f_mhz inlined using cached fields. The
                # subtract_baseline kwarg gates the per-block baseline differential
                # subtraction; default True preserves the original behaviour.
                d_current = s_plus - s_minus
                if subtract_baseline:
                    delta_d = d_current - calibration._d_reference  # type: ignore[attr-defined]
                else:
                    delta_d = d_current
                denom = calibration._denom  # type: ignore[attr-defined]
                if abs(denom) < 1e-12:
                    raise ValueError("Calibration slopes are too small for a stable delta_f estimate")
                delta_f_mhz = delta_d / denom
                sensitivity = calibration.sensitivity_norm_mhz_per_mT
                delta_proj_mT = float(delta_f_mhz / sensitivity)
                axis = calibration.measurement_axis
                delta_vector_mT = axis * delta_proj_mT
                rows.append(
                    {
                        "save_id": save_id,
                        "timestamp_epoch_s": timestamp,
                        "block_index": block_index,
                        "frequency_minus_mhz": float(block_freqs[0]),
                        "frequency_plus_mhz": float(block_freqs[1]),
                        "intensity_minus": s_minus,
                        "intensity_plus": s_plus,
                        "delta_B_projection_mT": delta_proj_mT,
                        "delta_B_projection_uT": delta_proj_mT * 1000.0,
                        "delta_Bx_uT": float(delta_vector_mT[0] * 1000.0),
                        "delta_By_uT": float(delta_vector_mT[1] * 1000.0),
                        "delta_Bz_uT": float(delta_vector_mT[2] * 1000.0),
                        "measurement_axis_x": float(axis[0]),
                        "measurement_axis_y": float(axis[1]),
                        "measurement_axis_z": float(axis[2]),
                        "sensitivity_norm_mhz_per_mT": sensitivity,
                        "slope_denom_abs": float(abs(calibration.slope_minus_per_mhz - calibration.slope_plus_per_mhz)),
                        "transition_index": int(calibration.transition_index),
                    }
                )
        return rows

    # Nonlinear-refinement branch: per-sample loop (search_radius_mT > 0)
    for row_index, save_id in enumerate(save_ids):
        spectrum = intensities[row_index]
        timestamp = float(timestamps[row_index])
        for block_index, calibration in calibrations.items():
            mask = block_indices == int(block_index)
            block_values = spectrum[mask]
            order = np.argsort(point_in_block[mask])
            block_values = block_values[order]
            block_freqs = parked_freqs_mhz[mask][order]
            if len(block_values) != 2:
                raise ValueError(f"Block {block_index} does not contain exactly two parked values")
            delta_proj_mT = estimate_delta_B_projection_mT(
                float(block_values[0]),
                float(block_values[1]),
                calibration,
                search_radius_mT=float(search_radius_mT),
            )
            delta_vector_mT = estimate_delta_B_vector_mT(
                float(block_values[0]),
                float(block_values[1]),
                calibration,
            )
            rows.append(
                {
                    "save_id": save_id,
                    "timestamp_epoch_s": timestamp,
                    "block_index": int(block_index),
                    "frequency_minus_mhz": float(block_freqs[0]),
                    "frequency_plus_mhz": float(block_freqs[1]),
                    "intensity_minus": float(block_values[0]),
                    "intensity_plus": float(block_values[1]),
                    "delta_B_projection_mT": float(delta_proj_mT),
                    "delta_B_projection_uT": float(delta_proj_mT * 1000.0),
                    "delta_Bx_uT": float(delta_vector_mT[0] * 1000.0),
                    "delta_By_uT": float(delta_vector_mT[1] * 1000.0),
                    "delta_Bz_uT": float(delta_vector_mT[2] * 1000.0),
                    "measurement_axis_x": float(calibration.measurement_axis[0]),
                    "measurement_axis_y": float(calibration.measurement_axis[1]),
                    "measurement_axis_z": float(calibration.measurement_axis[2]),
                    "sensitivity_norm_mhz_per_mT": float(calibration.sensitivity_norm_mhz_per_mT),
                    "slope_denom_abs": float(abs(calibration.slope_minus_per_mhz - calibration.slope_plus_per_mhz)),
                    "transition_index": int(calibration.transition_index),
                }
            )
    return rows


def reconstruct_vector_timeseries(
    projection_rows: list[dict[str, object]],
    *,
    min_channels: int = 3,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    ordered_ids: list[str] = []
    for row in projection_rows:
        save_id = str(row["save_id"])
        if save_id not in grouped:
            grouped[save_id] = []
            ordered_ids.append(save_id)
        grouped[save_id].append(row)

    # Cache (WA)† pseudo-inverse per unique block-set so that across a long time
    # series we run one SVD total instead of one per timestamp.
    # Key: (sorted block_indices, A-matrix rows as tuples, weight tuple)
    _pinv_cache: dict[tuple, tuple] = {}

    def _get_or_compute_pinv(
        rows_for_id: list[dict[str, object]],
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        """Return (A, P, rank, condition_number) where P = (WA)† W."""
        A = np.asarray(
            [[float(r["measurement_axis_x"]), float(r["measurement_axis_y"]), float(r["measurement_axis_z"])]
             for r in rows_for_id],
            dtype=float,
        )
        slope_denoms = np.asarray(
            [float(r.get("slope_denom_abs", r.get("sensitivity_norm_mhz_per_mT", 1.0))) for r in rows_for_id],
            dtype=float,
        )
        weights = slope_denoms ** 2
        weights = weights / float(np.max(weights)) if float(np.max(weights)) > 0.0 else np.ones_like(weights)

        cache_key = (
            tuple(int(r.get("block_index", i)) for i, r in enumerate(rows_for_id)),
            tuple(A.ravel().tolist()),
            tuple(weights.tolist()),
        )
        cached = _pinv_cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        W = np.diag(weights)
        WA = W @ A
        # Compute weighted pseudo-inverse: P such that solution = P @ b
        # lstsq(WA, W·b) = (WA)† W b, so P = (WA)† W
        WA_pinv, _, rank, singular_values = np.linalg.lstsq(WA, W, rcond=None)
        if singular_values.size == 0 or float(np.min(np.abs(singular_values))) == 0.0:
            condition_number = float("inf")
        else:
            condition_number = float(np.max(np.abs(singular_values)) / np.min(np.abs(singular_values)))

        result = (A, WA_pinv, int(rank), condition_number)
        _pinv_cache[cache_key] = result  # type: ignore[assignment]
        return result

    vector_rows: list[dict[str, object]] = []
    for save_id in ordered_ids:
        rows = grouped[save_id]
        timestamp = float(rows[0]["timestamp_epoch_s"])
        if len(rows) < int(min_channels):
            vector_rows.append(
                {
                    "save_id": save_id,
                    "timestamp_epoch_s": timestamp,
                    "n_channels": int(len(rows)),
                    "rank": -1,
                    "condition_number": float("inf"),
                    "residual_norm_uT": float("nan"),
                    "delta_Bx_uT": float("nan"),
                    "delta_By_uT": float("nan"),
                    "delta_Bz_uT": float("nan"),
                    "status": "insufficient_channels",
                }
            )
            continue

        A, P, rank, condition_number = _get_or_compute_pinv(rows)
        b = np.asarray([float(row["delta_B_projection_mT"]) for row in rows], dtype=float)
        solution = P @ b
        fitted = A @ solution
        residual_norm_uT = float(np.linalg.norm(fitted - b) * 1000.0)

        vector_rows.append(
            {
                "save_id": save_id,
                "timestamp_epoch_s": timestamp,
                "n_channels": int(len(rows)),
                "rank": rank,
                "condition_number": condition_number,
                "residual_norm_uT": residual_norm_uT,
                "delta_Bx_uT": float(solution[0] * 1000.0),
                "delta_By_uT": float(solution[1] * 1000.0),
                "delta_Bz_uT": float(solution[2] * 1000.0),
                "status": "ok" if rank >= 3 else "rank_deficient",
            }
        )
    return vector_rows


def build_inversion_records(
    projection_rows: list[dict[str, object]],
    vector_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in projection_rows:
        grouped.setdefault(str(row["save_id"]), []).append(row)
    vector_by_id = {str(row["save_id"]): row for row in vector_rows}

    records: list[dict[str, object]] = []
    for save_id, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda item: int(item["block_index"]))
        vector_row = vector_by_id.get(save_id, {})
        jacobian_rows = [
            [
                float(row["measurement_axis_x"]),
                float(row["measurement_axis_y"]),
                float(row["measurement_axis_z"]),
            ]
            for row in rows_sorted
        ]
        projection_vector_uT = [float(row["delta_B_projection_uT"]) for row in rows_sorted]
        records.append(
            {
                "save_id": save_id,
                "timestamp_epoch_s": float(rows_sorted[0]["timestamp_epoch_s"]),
                "block_indices": [int(row["block_index"]) for row in rows_sorted],
                "frequencies_minus_mhz": [float(row["frequency_minus_mhz"]) for row in rows_sorted],
                "frequencies_plus_mhz": [float(row["frequency_plus_mhz"]) for row in rows_sorted],
                "jacobian_rows": jacobian_rows,
                "projection_vector_uT": projection_vector_uT,
                "solution_delta_B_uT": [
                    float(vector_row.get("delta_Bx_uT", float("nan"))),
                    float(vector_row.get("delta_By_uT", float("nan"))),
                    float(vector_row.get("delta_Bz_uT", float("nan"))),
                ],
                "residual_norm_uT": float(vector_row.get("residual_norm_uT", float("nan"))),
                "rank": int(vector_row.get("rank", -1)),
                "condition_number": float(vector_row.get("condition_number", float("inf"))),
                "status": str(vector_row.get("status", "missing")),
            }
        )
    return records


def fit_parked_intensity_csv(
    full_scan_csv: str | Path,
    parked_csv: str | Path,
    *,
    B_bias_mT: np.ndarray,
    nv_axes_preset: str = "qdm",
    hyperfine_splitting_mhz: float | None = 2.16,
    search_radius_mT: float = 0.0,
) -> dict[str, object]:
    reference_batch = load_spectrum_csv(full_scan_csv)
    if reference_batch.spectra.shape[0] != 1:
        raise ValueError("The full-scan reference CSV must resolve to exactly one spectrum")

    parked_series = load_parked_frequency_csv(parked_csv)
    calibrations = build_blockwise_calibrations(
        reference_batch.freqs_mhz,
        reference_batch.spectra[0],
        parked_series.freqs_mhz,
        parked_series.block_indices,
        parked_series.point_in_block,
        B_bias_mT=np.asarray(B_bias_mT, dtype=float),
        nv_axes_preset=nv_axes_preset,
    )
    rows = estimate_parked_series_fields(
        parked_series.intensities,
        parked_series.timestamps_epoch_s,
        parked_series.save_ids,
        parked_series.block_indices,
        parked_series.point_in_block,
        parked_series.freqs_mhz,
        calibrations,
        hyperfine_splitting_mhz=hyperfine_splitting_mhz,
        nv_axes_preset=nv_axes_preset,
        search_radius_mT=search_radius_mT,
    )
    vector_rows = reconstruct_vector_timeseries(rows)
    return {
        "reference_csv": str(full_scan_csv),
        "parked_csv": str(parked_csv),
        "B_bias_mT": np.asarray(B_bias_mT, dtype=float),
        "calibrations": {key: asdict(value) for key, value in calibrations.items()},
        "rows": rows,
        "vector_rows": vector_rows,
        "inversion_records": build_inversion_records(rows, vector_rows),
    }
