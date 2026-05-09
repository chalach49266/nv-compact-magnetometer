from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

from .model import invert_odmr_pair, named_nv_axes, odmr_resonances, odmr_spectrum
from .static_fit import fit_static_spectrum


@dataclass(frozen=True)
class SpectrumParameterFitResult:
    field_estimate: dict[str, object]
    linewidths_mhz: np.ndarray
    contrasts: np.ndarray
    baseline_offset: float
    hyperfine_mode: str
    hyperfine_splitting_mhz: float | None
    cost: float
    aic: float
    success: bool
    message: str


@dataclass(frozen=True)
class SpectrumSeededFitSummary:
    peak_fit: dict[str, object]
    ranked_candidates: list[dict[str, object]]
    best: dict[str, object]


def _resolve_nv_axes(nv_axes_preset: str, nv_axes: np.ndarray | None = None) -> np.ndarray:
    if nv_axes is None:
        return named_nv_axes(nv_axes_preset)
    return np.asarray(nv_axes, dtype=float)


def fit_field_from_peak_positions(
    transition_centers_mhz: np.ndarray,
    B_base_mT: np.ndarray,
    *,
    nv_axes_preset: str = "canonical",
    nv_axes: np.ndarray | None = None,
    pred_axis_centers: np.ndarray | None = None,
) -> dict[str, object]:
    """Fast analytical field estimate from exactly 8 transition-center frequencies.

    For 16-peak (hyperfine-resolved) spectra call ``find_dip_positions`` with
    ``hyperfine_splitting_mhz`` first, or manually call
    ``collapse_hyperfine_doublets`` — both return transition centers that this
    function expects.

    The approach:
    1. Pair i-th lowest with i-th highest center (0↔7, 1↔6, 2↔5, 3↔4).
    2. Match each pair to a NV axis via predicted per-axis center frequencies.
    3. Call ``invert_odmr_pair`` to get |B_axial| for each pair analytically.
    4. Sign from the known bias-field projection on the matched axis.
    5. Solve the overdetermined 4×3 system via least-squares for B_total;
       return B_local = B_total − B_base.

    Note: this path does not fit linewidths, contrasts, or baseline.  It is
    fast and works well when peaks are cleanly resolved.  Use
    ``fit_spectrum_parameters`` when peaks overlap or SNR is low.
    """
    has_custom_nv_axes = nv_axes is not None
    centers = np.sort(np.asarray(transition_centers_mhz, dtype=float))
    if len(centers) != 8:
        raise ValueError(f"Expected 8 transition-center frequencies, got {len(centers)}")
    b_base = np.asarray(B_base_mT, dtype=float)
    if b_base.shape != (3,):
        raise ValueError(f"B_base_mT must have shape (3,), got {b_base.shape}")
    nv_axes = _resolve_nv_axes(nv_axes_preset, nv_axes=nv_axes)

    pairs = [(float(centers[i]), float(centers[7 - i])) for i in range(4)]
    obs_pair_centers = np.array([0.5 * (p[0] + p[1]) for p in pairs])

    if pred_axis_centers is None:
        res_by_axis = odmr_resonances(b_base, nv_axes=nv_axes)  # (4, 2)
        pred_axis_centers = 0.5 * (res_by_axis[:, 0] + res_by_axis[:, 1])  # (4,)
    cost_matrix = np.abs(obs_pair_centers[:, None] - np.asarray(pred_axis_centers, dtype=float)[None, :])
    obs_order, axis_order = linear_sum_assignment(cost_matrix)

    A = np.zeros((4, 3), dtype=float)
    b_vec = np.zeros(4, dtype=float)
    for obs_i, axis_j in zip(obs_order, axis_order):
        f_low, f_high = pairs[obs_i]
        b_axial_mag, _, _ = invert_odmr_pair(f_low, f_high)
        sign = float(np.sign(np.dot(b_base, nv_axes[axis_j])))
        if sign == 0.0:
            sign = 1.0
        A[obs_i] = nv_axes[axis_j]
        b_vec[obs_i] = sign * b_axial_mag

    b_total_est, _, rank, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    b_local_est = b_total_est - b_base
    residual_norm_uT = float(np.linalg.norm(A @ b_total_est - b_vec) * 1000.0)

    return {
        "B_base_mT": b_base,
        "B_local_mT": b_local_est,
        "B_total_mT": b_total_est,
        "B_base_uT": b_base * 1000.0,
        "B_local_uT": b_local_est * 1000.0,
        "B_total_uT": b_total_est * 1000.0,
        "nv_axes_preset": nv_axes_preset,
        "nv_axes_source": "custom" if has_custom_nv_axes else nv_axes_preset,
        "rank": int(rank),
        "residual_norm_uT": residual_norm_uT,
    }


def fit_total_field_from_peak_positions(
    transition_centers_mhz: np.ndarray,
    *,
    nv_axes_preset: str = "canonical",
    nv_axes: np.ndarray | None = None,
    max_candidates: int = 16,
) -> dict[str, object]:
    """Estimate total field from transition centers without a known bias vector.

    The input must be transition centers, not individual hyperfine components.
    The sorted lower-half centers are paired with the sorted upper-half centers
    in reverse order.  Each pair analytically gives ``|B dot n_j|`` and
    ``|B|`` through ``invert_odmr_pair``.

    Without a known bias vector the projection signs are ambiguous.  For three
    or more pairs this function enumerates axis assignments and sign choices,
    solves for candidate ``B_total`` vectors, and ranks them by projection
    residual plus consistency with the per-pair total-field magnitudes.  For one
    or two pairs it still returns the pair-level projections and total-field
    magnitudes, but does not claim a unique 3D vector.
    """
    has_custom_nv_axes = nv_axes is not None
    centers = np.sort(np.asarray(transition_centers_mhz, dtype=float))
    if len(centers) < 2:
        raise ValueError("Expected at least 2 transition-center frequencies")
    if len(centers) % 2 != 0:
        raise ValueError(f"Expected an even number of transition centers, got {len(centers)}")
    if len(centers) > 8:
        raise ValueError(f"Expected at most 8 transition centers, got {len(centers)}")

    nv_axes = _resolve_nv_axes(nv_axes_preset, nv_axes=nv_axes)
    n_pairs = len(centers) // 2
    pairs = [(float(centers[i]), float(centers[-1 - i])) for i in range(n_pairs)]

    pair_rows: list[dict[str, object]] = []
    axial_abs = np.zeros(n_pairs, dtype=float)
    total_mags = np.zeros(n_pairs, dtype=float)
    for pair_index, (f_low, f_high) in enumerate(pairs):
        b_axial_abs, b_trans_abs, b_total_abs = invert_odmr_pair(f_low, f_high)
        axial_abs[pair_index] = b_axial_abs
        total_mags[pair_index] = b_total_abs
        pair_rows.append(
            {
                "pair_index": pair_index,
                "frequency_low_mhz": f_low,
                "frequency_high_mhz": f_high,
                "B_axis_projection_abs_mT": float(b_axial_abs),
                "B_transverse_to_axis_abs_mT": float(b_trans_abs),
                "B_total_abs_mT": float(b_total_abs),
            }
        )

    total_mean = float(np.mean(total_mags))
    total_std = float(np.std(total_mags))
    candidates: list[dict[str, object]] = []
    if n_pairs >= 3:
        import itertools

        for axis_indices in itertools.permutations(range(4), n_pairs):
            A_unsigned = nv_axes[np.asarray(axis_indices, dtype=int)]
            for signs in itertools.product([-1.0, 1.0], repeat=n_pairs):
                signs_arr = np.asarray(signs, dtype=float)
                A = A_unsigned
                b_vec = signs_arr * axial_abs
                b_total, _, rank, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                fitted = A @ b_total
                projection_residual_mT = float(np.linalg.norm(fitted - b_vec))
                norm_residual_mT = abs(float(np.linalg.norm(b_total)) - total_mean)
                score = projection_residual_mT + norm_residual_mT + total_std
                candidates.append(
                    {
                        "B_total_mT": b_total,
                        "B_total_uT": b_total * 1000.0,
                        "B_total_norm_mT": float(np.linalg.norm(b_total)),
                        "axis_indices": list(axis_indices),
                        "projection_signs": signs_arr,
                        "signed_axis_projections_mT": b_vec,
                        "all_axis_projections_mT": nv_axes @ b_total,
                        "rank": int(rank),
                        "projection_residual_mT": projection_residual_mT,
                        "norm_residual_mT": norm_residual_mT,
                        "score": float(score),
                    }
                )
        candidates = sorted(candidates, key=lambda item: float(item["score"]))[: int(max_candidates)]

    return {
        "transition_centers_mhz": centers,
        "pairs": pair_rows,
        "B_total_abs_mT_mean": total_mean,
        "B_total_abs_mT_std": total_std,
        "nv_axes_preset": nv_axes_preset,
        "nv_axes_source": "custom" if has_custom_nv_axes else nv_axes_preset,
        "n_pairs": int(n_pairs),
        "candidates": candidates,
        "status": "vector_candidates" if candidates else "partial_projection_only",
    }


def rank_spectrum_field_candidates(
    freqs_mhz: np.ndarray,
    spectrum: np.ndarray,
    transition_centers_mhz: np.ndarray,
    *,
    nv_axes_preset: str = "canonical",
    nv_axes: np.ndarray | None = None,
    max_candidates: int = 8,
    linewidth_mode: str = "per_axis",
    contrast_mode: str = "per_axis",
    hyperfine_mode: str = "fixed",
    initial_hyperfine_splitting_mhz: float = 2.16,
    local_field_bounds_mT: float = 1.5,
    preferred_field_mT: np.ndarray | None = None,
    preference_weight: float = 0.0,
) -> SpectrumSeededFitSummary:
    freqs = np.asarray(freqs_mhz, dtype=float)
    measured = np.asarray(spectrum, dtype=float)
    peak_fit = fit_total_field_from_peak_positions(
        np.asarray(transition_centers_mhz, dtype=float),
        nv_axes_preset=nv_axes_preset,
        nv_axes=nv_axes,
        max_candidates=max_candidates,
    )
    candidates: list[dict[str, object]] = []
    preferred_field = None if preferred_field_mT is None else np.asarray(preferred_field_mT, dtype=float)
    for candidate_index, candidate in enumerate(peak_fit.get("candidates", [])[: int(max_candidates)]):
        result = fit_spectrum_parameters(
            freqs,
            measured,
            B_base_mT=np.asarray(candidate["B_total_mT"], dtype=float),
            nv_axes_preset=nv_axes_preset,
            nv_axes=nv_axes,
            linewidth_mode=linewidth_mode,
            contrast_mode=contrast_mode,
            hyperfine_mode=hyperfine_mode,
            initial_hyperfine_splitting_mhz=initial_hyperfine_splitting_mhz,
            local_field_bounds_mT=local_field_bounds_mT,
        )
        b_total_mT = np.asarray(result.field_estimate["B_total_mT"], dtype=float)
        resolved_axes = _resolve_nv_axes(nv_axes_preset, nv_axes=nv_axes)
        predicted = odmr_spectrum(
            freqs,
            b_total_mT,
            linewidths=result.linewidths_mhz,
            contrasts=result.contrasts,
            hyperfine_splitting_mhz=result.hyperfine_splitting_mhz,
            nv_axes=resolved_axes,
        ) + float(result.baseline_offset)
        residual = measured - predicted
        preferred_field_penalty = 0.0
        if preferred_field is not None and np.linalg.norm(preferred_field) > 0.0 and np.linalg.norm(b_total_mT) > 0.0:
            direction_penalty = 1.0 - float(
                np.dot(b_total_mT, preferred_field)
                / (np.linalg.norm(b_total_mT) * np.linalg.norm(preferred_field))
            )
            magnitude_penalty = abs(float(np.linalg.norm(b_total_mT) - np.linalg.norm(preferred_field)))
            preferred_field_penalty = direction_penalty + 0.25 * magnitude_penalty
        selection_score = float(result.cost) + float(preference_weight) * preferred_field_penalty
        candidates.append(
            {
                "candidate_index": candidate_index,
                "success": bool(result.success),
                "cost": float(result.cost),
                "selection_score": selection_score,
                "aic": float(result.aic),
                "B_base_mT": np.asarray(candidate["B_total_mT"], dtype=float).tolist(),
                "B_total_mT": b_total_mT.tolist(),
                "B_total_norm_mT": float(np.linalg.norm(b_total_mT)),
                "preferred_field_penalty": float(preferred_field_penalty),
                "linewidths_mhz": np.asarray(result.linewidths_mhz, dtype=float).tolist(),
                "contrasts": np.asarray(result.contrasts, dtype=float).tolist(),
                "baseline_offset": float(result.baseline_offset),
                "hyperfine_splitting_mhz": result.hyperfine_splitting_mhz,
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "max_abs_residual": float(np.max(np.abs(residual))),
                "predicted": predicted,
                "residual": residual,
            }
        )

    if not candidates:
        raise ValueError("No vector candidates available from supplied transition centers")

    best = min(candidates, key=lambda item: float(item["selection_score"]))
    serializable_candidates = [
        {key: value for key, value in candidate.items() if key not in {"predicted", "residual"}}
        for candidate in sorted(candidates, key=lambda item: float(item["selection_score"]))
    ]
    return SpectrumSeededFitSummary(
        peak_fit=peak_fit,
        ranked_candidates=serializable_candidates,
        best={key: value for key, value in best.items() if key not in {"predicted", "residual"}},
    )


def fit_local_field(
    freqs_mhz: np.ndarray,
    spectrum: np.ndarray,
    *,
    B_base_mT: np.ndarray,
    nv_axes_preset: str = "canonical",
    nv_axes: np.ndarray | None = None,
    linewidth_mhz: float = 3.0,
    contrast: float = 0.035,
    hyperfine_splitting_mhz: float | None = 2.16,
    max_dips: int = 18,
    min_dips: int = 8,
    local_field_l2_penalty: float = 0.02,
    local_field_bounds_mT: float = 0.2,
) -> dict[str, object]:
    has_custom_nv_axes = nv_axes is not None
    nv_axes = _resolve_nv_axes(nv_axes_preset, nv_axes=nv_axes)
    diagnostics = fit_static_spectrum(
        freqs=np.asarray(freqs_mhz, dtype=float),
        measured_norm=np.asarray(spectrum, dtype=float),
        B_base_mT=np.asarray(B_base_mT, dtype=float),
        linewidths=float(linewidth_mhz),
        contrasts=float(contrast),
        hyperfine_splitting_mhz=hyperfine_splitting_mhz,
        max_dips=int(max_dips),
        min_dips=int(min_dips),
        local_field_l2_penalty=float(local_field_l2_penalty),
        local_field_bounds_mT=float(local_field_bounds_mT),
        return_diagnostics=True,
        nv_axes=nv_axes,
    )
    freqs = np.asarray(freqs_mhz, dtype=float)
    measured = np.asarray(spectrum, dtype=float)
    b_base = np.asarray(B_base_mT, dtype=float)

    def _local_cost(b_local: np.ndarray) -> float:
        predicted = odmr_spectrum(
            freqs,
            b_base + np.asarray(b_local, dtype=float),
            linewidths=float(linewidth_mhz),
            contrasts=float(contrast),
            hyperfine_splitting_mhz=hyperfine_splitting_mhz,
            nv_axes=nv_axes,
        )
        regularization = float(local_field_l2_penalty) * float(np.dot(b_local, b_local))
        return float(np.sum((predicted - measured) ** 2) + regularization)

    _EARLY_EXIT_THRESHOLD = 1e-4

    raw_initial = np.asarray(diagnostics["B_local_mT"], dtype=float)
    bound = abs(float(local_field_bounds_mT))
    starts = [
        raw_initial,
        np.zeros(3, dtype=float),
        0.5 * raw_initial,
        np.array([0.05, 0.0, 0.0], dtype=float),
        np.array([-0.05, 0.0, 0.0], dtype=float),
        np.array([0.0, 0.05, 0.0], dtype=float),
        np.array([0.0, -0.05, 0.0], dtype=float),
        np.array([0.0, 0.0, 0.05], dtype=float),
        np.array([0.0, 0.0, -0.05], dtype=float),
    ]
    best_x = raw_initial.copy()
    best_cost = _local_cost(best_x)
    best_message = str(diagnostics["message"])
    best_success = bool(diagnostics["success"])
    best_nfev = int(diagnostics["nfev"])
    for start in starts:
        if float(best_cost) < _EARLY_EXIT_THRESHOLD and bool(best_success):
            break
        result = minimize(
            lambda x: _local_cost(np.asarray(x, dtype=float)),
            np.clip(np.asarray(start, dtype=float), -bound, bound),
            method="L-BFGS-B",
            bounds=[(-bound, bound)] * 3,
            options={"ftol": 1e-12, "maxiter": 20000},
        )
        if float(result.fun) < float(best_cost):
            best_x = np.asarray(result.x, dtype=float)
            best_cost = float(result.fun)
            best_message = str(result.message)
            best_success = bool(result.success)
            best_nfev = int(getattr(result, "nfev", -1))

    # If the peak-finder-based initialization failed to find enough dips, try a
    # coarse 3×3×3 grid of starting points to escape the peak-detection dependency.
    # This is the merged-peaks / broad-linewidth fallback.
    n_obs = len(np.asarray(diagnostics.get("observed_dips_mhz", []), dtype=float))
    if not best_success and n_obs < int(min_dips):
        half = abs(float(local_field_bounds_mT))
        grid_vals = np.linspace(-half, half, 3)
        for gx in grid_vals:
            for gy in grid_vals:
                for gz in grid_vals:
                    if float(best_cost) < _EARLY_EXIT_THRESHOLD and bool(best_success):
                        break
                    grid_start = np.array([gx, gy, gz], dtype=float)
                    result = minimize(
                        lambda x: _local_cost(np.asarray(x, dtype=float)),
                        grid_start,
                        method="L-BFGS-B",
                        bounds=[(-half, half)] * 3,
                        options={"ftol": 1e-12, "maxiter": 20000},
                    )
                    if float(result.fun) < float(best_cost):
                        best_x = np.asarray(result.x, dtype=float)
                        best_cost = float(result.fun)
                        best_message = str(result.message)
                        best_success = bool(result.success)
                        best_nfev += int(getattr(result, "nfev", 0))

    diagnostics["B_local_mT"] = best_x
    diagnostics["B_total_mT"] = b_base + best_x
    diagnostics["cost"] = best_cost
    diagnostics["message"] = best_message
    diagnostics["success"] = best_success
    diagnostics["nfev"] = best_nfev

    # Compute a linewidth regime indicator to flag potential merged-peak conditions.
    resonances_sorted = np.sort(odmr_resonances(b_base + best_x, nv_axes=nv_axes).ravel())
    min_spacing_mhz = float(np.min(np.diff(resonances_sorted))) if len(resonances_sorted) > 1 else float("inf")
    lw = float(linewidth_mhz)
    if lw < min_spacing_mhz / 3.0:
        linewidth_regime = "normal"
    elif lw < min_spacing_mhz:
        linewidth_regime = "broad"
    else:
        linewidth_regime = "merged"

    b_local = np.asarray(diagnostics["B_local_mT"], dtype=float)
    b_total = np.asarray(diagnostics["B_total_mT"], dtype=float)
    b_base = np.asarray(diagnostics["B_base_mT"], dtype=float)
    return {
        "B_base_mT": b_base,
        "B_local_mT": b_local,
        "B_total_mT": b_total,
        "B_base_uT": b_base * 1000.0,
        "B_local_uT": b_local * 1000.0,
        "B_total_uT": b_total * 1000.0,
        "observed_dips_mhz": np.asarray(diagnostics["observed_dips_mhz"], dtype=float),
        "initialization_dips_mhz": np.asarray(diagnostics["initialization_dips_mhz"], dtype=float),
        "success": bool(diagnostics["success"]),
        "message": str(diagnostics["message"]),
        "cost": float(diagnostics["cost"]),
        "nfev": int(diagnostics["nfev"]),
        "nv_axes_preset": nv_axes_preset,
        "nv_axes_source": "custom" if has_custom_nv_axes else nv_axes_preset,
        "linewidth_regime": linewidth_regime,
        "min_resonance_spacing_mhz": min_spacing_mhz,
    }


def _expand_parameter_block(values: np.ndarray, *, mode: str) -> np.ndarray:
    if mode == "shared":
        return np.full((4, 2), float(values[0]), dtype=float)
    if mode == "per_axis":
        axis_values = np.asarray(values, dtype=float)
        return np.repeat(axis_values[:, None], 2, axis=1)
    if mode == "per_peak":
        return np.asarray(values, dtype=float).reshape(4, 2)
    raise ValueError(f"Unsupported parameter mode: {mode}")


def _parameter_count(mode: str) -> int:
    if mode == "shared":
        return 1
    if mode == "per_axis":
        return 4
    if mode == "per_peak":
        return 8
    raise ValueError(f"Unsupported parameter mode: {mode}")


def _aic_from_cost(cost: float, n_points: int, n_params: int) -> float:
    rss = max(float(cost), 1e-18)
    return float(n_points * np.log(rss / max(1, n_points)) + 2.0 * n_params)


def fit_spectrum_parameters(
    freqs_mhz: np.ndarray,
    spectrum: np.ndarray,
    *,
    B_base_mT: np.ndarray,
    nv_axes_preset: str = "canonical",
    nv_axes: np.ndarray | None = None,
    linewidth_mode: str = "per_axis",
    contrast_mode: str = "per_axis",
    hyperfine_mode: str = "auto",
    initial_hyperfine_splitting_mhz: float = 2.16,
    local_field_bounds_mT: float = 0.3,
) -> SpectrumParameterFitResult:
    has_custom_nv_axes = nv_axes is not None
    freqs = np.asarray(freqs_mhz, dtype=float)
    measured = np.asarray(spectrum, dtype=float)
    b_base = np.asarray(B_base_mT, dtype=float)
    if b_base.shape != (3,):
        raise ValueError(f"B_base_mT must have shape (3,), got {b_base.shape}")

    field_estimate = fit_local_field(
        freqs,
        measured,
        B_base_mT=b_base,
        nv_axes_preset=nv_axes_preset,
        nv_axes=nv_axes,
        hyperfine_splitting_mhz=initial_hyperfine_splitting_mhz if hyperfine_mode != "none" else None,
        local_field_bounds_mT=local_field_bounds_mT,
    )
    initial_local = np.asarray(field_estimate["B_local_mT"], dtype=float)
    nv_axes = _resolve_nv_axes(nv_axes_preset, nv_axes=nv_axes)

    linewidth_count = _parameter_count(linewidth_mode)
    contrast_count = _parameter_count(contrast_mode)

    def run_model(mode_name: str) -> SpectrumParameterFitResult:
        include_hyperfine = mode_name in {"fit", "fixed"}
        fit_hyperfine = mode_name == "fit"

        x0 = np.concatenate(
            [
                initial_local,
                np.full(linewidth_count, 3.0, dtype=float),
                np.full(contrast_count, 0.035, dtype=float),
                np.array([0.0], dtype=float),
                np.array([initial_hyperfine_splitting_mhz], dtype=float) if fit_hyperfine else np.array([], dtype=float),
            ]
        )
        bounds = [(-abs(local_field_bounds_mT), abs(local_field_bounds_mT))] * 3
        bounds += [(0.1, 20.0)] * linewidth_count
        bounds += [(1e-4, 0.2)] * contrast_count
        bounds += [(-0.05, 0.05)]
        if fit_hyperfine:
            bounds += [(1.0, 4.0)]

        def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float | None]:
            idx = 0
            b_local = params[idx:idx + 3]
            idx += 3
            linewidth_values = params[idx:idx + linewidth_count]
            idx += linewidth_count
            contrast_values = params[idx:idx + contrast_count]
            idx += contrast_count
            baseline_offset = float(params[idx])
            idx += 1
            splitting = float(params[idx]) if fit_hyperfine else (
                float(initial_hyperfine_splitting_mhz) if include_hyperfine else None
            )
            return (
                b_local,
                _expand_parameter_block(linewidth_values, mode=linewidth_mode),
                _expand_parameter_block(contrast_values, mode=contrast_mode),
                baseline_offset,
                splitting,
            )

        def cost_fn(params: np.ndarray) -> float:
            b_local, linewidths, contrasts, baseline_offset, splitting = unpack(np.asarray(params, dtype=float))
            predicted = odmr_spectrum(
                freqs,
                b_base + b_local,
                linewidths=linewidths,
                contrasts=contrasts,
                hyperfine_splitting_mhz=splitting,
                nv_axes=nv_axes,
            ) + baseline_offset
            return float(np.sum((predicted - measured) ** 2))

        result = minimize(
            cost_fn,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12, "maxiter": 20000},
        )
        b_local, linewidths, contrasts, baseline_offset, splitting = unpack(np.asarray(result.x, dtype=float))
        cost = float(result.fun)
        return SpectrumParameterFitResult(
            field_estimate={
                "B_base_mT": b_base,
                "B_local_mT": b_local,
                "B_total_mT": b_base + b_local,
                "B_base_uT": b_base * 1000.0,
                "B_local_uT": b_local * 1000.0,
                "B_total_uT": (b_base + b_local) * 1000.0,
                "nv_axes_preset": nv_axes_preset,
                "nv_axes_source": "custom" if has_custom_nv_axes else nv_axes_preset,
            },
            linewidths_mhz=linewidths,
            contrasts=contrasts,
            baseline_offset=baseline_offset,
            hyperfine_mode=mode_name,
            hyperfine_splitting_mhz=splitting,
            cost=cost,
            aic=_aic_from_cost(cost, len(freqs), len(x0)),
            success=bool(result.success),
            message=str(result.message),
        )

    if hyperfine_mode == "none":
        return run_model("none")
    if hyperfine_mode == "fixed":
        return run_model("fixed")
    if hyperfine_mode == "fit":
        return run_model("fit")
    if hyperfine_mode != "auto":
        raise ValueError(f"Unsupported hyperfine mode: {hyperfine_mode}")

    candidates = [run_model("none"), run_model("fit")]
    return min(candidates, key=lambda item: item.aic)
