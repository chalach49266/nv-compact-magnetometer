from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.signal import find_peaks

from .model import odmr_resonances, odmr_resonances_and_gradients, odmr_spectrum, optimal_bias_direction


def _find_dip_positions(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    max_dips: int = 10,
    min_dips: int = 4,
) -> np.ndarray:
    inverted = 1.0 - spectrum
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    chosen_peaks = None
    for prom in [0.001, 0.003, 0.005, 0.01, 0.02]:
        peaks, props = find_peaks(
            inverted,
            prominence=prom,
            distance=max(1, int(0.5 / df)),
        )
        if len(peaks) == 0:
            continue

        prominences = np.asarray(props.get("prominences", np.zeros(len(peaks))), dtype=float)
        if len(peaks) > max_dips:
            keep = np.argsort(prominences)[-max_dips:]
            peaks = np.sort(peaks[keep])

        chosen_peaks = peaks
        if len(peaks) >= min_dips:
            break

    if chosen_peaks is None or len(chosen_peaks) == 0:
        raise RuntimeError("No ODMR dips found in spectrum")

    dip_freqs: list[float] = []
    for peak_idx in chosen_peaks:
        lo = max(0, peak_idx - 2)
        hi = min(len(freqs), peak_idx + 3)
        f_local = freqs[lo:hi]
        s_local = inverted[lo:hi]
        if len(f_local) >= 3:
            coeffs = np.polyfit(f_local, s_local, 2)
            f_peak = -coeffs[1] / (2 * coeffs[0])
            if f_local[0] <= f_peak <= f_local[-1]:
                dip_freqs.append(float(f_peak))
            else:
                dip_freqs.append(float(freqs[peak_idx]))
        else:
            dip_freqs.append(float(freqs[peak_idx]))

    return np.sort(np.asarray(dip_freqs, dtype=float))


def _collapse_doublet_dips_to_centers(
    observed_dips_mhz: np.ndarray,
    hyperfine_splitting_mhz: float,
    tolerance_mhz: float = 0.75,
) -> np.ndarray:
    dips = np.sort(np.asarray(observed_dips_mhz, dtype=float))
    candidates: list[tuple[float, int, int]] = []
    for i in range(len(dips) - 1):
        error = abs(float(dips[i + 1] - dips[i]) - float(hyperfine_splitting_mhz))
        if error <= tolerance_mhz:
            candidates.append((error, i, i + 1))

    centers: list[float] = []
    used: set[int] = set()
    for _error, i, j in sorted(candidates, key=lambda item: item[0]):
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        centers.append(float(0.5 * (dips[i] + dips[j])))

    for i, dip in enumerate(dips):
        if i not in used:
            centers.append(float(dip))

    return np.sort(np.asarray(centers, dtype=float))


def _compute_jacobian(
    B_total: np.ndarray,
    eps: float = 1e-5,
    nv_axes: np.ndarray | None = None,
) -> np.ndarray:
    # Analytic Jacobian via odmr_resonances_and_gradients — replaces finite differences.
    # Returns (8, 3) matrix: d(resonances_sorted)/d(B_total).
    _, gradients = odmr_resonances_and_gradients(B_total, nv_axes=nv_axes)
    # gradients shape (4, 2, 3); flatten to (8, 3) matching ravel() order of resonances
    return gradients.reshape(8, 3)


def _match_and_solve(
    observed_dips: np.ndarray,
    B_bias: np.ndarray,
    nv_axes: np.ndarray | None = None,
) -> np.ndarray:
    predicted = odmr_resonances(B_bias, nv_axes=nv_axes).ravel()
    pred_sorted_idx = np.argsort(predicted)
    pred_sorted = predicted[pred_sorted_idx]

    J_full = _compute_jacobian(B_bias, nv_axes=nv_axes)
    J_sorted = J_full[pred_sorted_idx]

    delta_f = np.zeros(8, dtype=float)
    weight = np.zeros(8, dtype=float)
    used_obs: set[int] = set()

    for ip in range(8):
        best_io = None
        best_dist = np.inf
        for io in range(len(observed_dips)):
            if io in used_obs:
                continue
            dist = abs(observed_dips[io] - pred_sorted[ip])
            if dist < best_dist:
                best_dist = dist
                best_io = io
        if best_io is not None and best_dist < 20.0:
            delta_f[ip] = observed_dips[best_io] - pred_sorted[ip]
            weight[ip] = 1.0
            used_obs.add(best_io)

    W = np.diag(weight)
    B_local_est, _, _, _ = np.linalg.lstsq(W @ J_sorted, W @ delta_f, rcond=None)
    return B_local_est


def _cost_with_base(
    B_local_flat: np.ndarray,
    freqs: np.ndarray,
    measured_norm: np.ndarray,
    B_base_mT: np.ndarray,
    linewidths: np.ndarray | float,
    contrasts: np.ndarray | float,
    hyperfine_splitting_mhz: float | None,
    nv_axes: np.ndarray | None,
    local_field_l2_penalty: float,
) -> float:
    B_local = np.asarray(B_local_flat, dtype=float)
    B_total = np.asarray(B_base_mT, dtype=float) + B_local
    predicted = odmr_spectrum(
        freqs,
        B_total,
        linewidths=linewidths,
        contrasts=contrasts,
        hyperfine_splitting_mhz=hyperfine_splitting_mhz,
        nv_axes=nv_axes,
    )
    spectrum_cost = float(np.sum((predicted - measured_norm) ** 2))
    regularization_cost = float(local_field_l2_penalty) * float(np.dot(B_local, B_local))
    return spectrum_cost + regularization_cost


def _residuals_with_base(
    B_local_flat: np.ndarray,
    freqs_sq: np.ndarray,
    freqs: np.ndarray,
    measured_norm: np.ndarray,
    B_base_mT: np.ndarray,
    linewidths: np.ndarray | float,
    contrasts: np.ndarray | float,
    hyperfine_splitting_mhz: float | None,
    nv_axes: np.ndarray | None,
    local_field_l2_penalty: float,
    sqrt_reg: np.ndarray,
) -> np.ndarray:
    """Residuals for least_squares: [predicted - measured, sqrt(penalty)*B_local]."""
    B_local = np.asarray(B_local_flat, dtype=float)
    B_total = np.asarray(B_base_mT, dtype=float) + B_local
    predicted = odmr_spectrum(
        freqs,
        B_total,
        linewidths=linewidths,
        contrasts=contrasts,
        hyperfine_splitting_mhz=hyperfine_splitting_mhz,
        nv_axes=nv_axes,
    )
    spectrum_res = predicted - measured_norm
    if float(local_field_l2_penalty) > 0.0:
        return np.concatenate([spectrum_res, sqrt_reg * B_local])
    return spectrum_res


def _jac_residuals_with_base(
    B_local_flat: np.ndarray,
    freqs_sq: np.ndarray,
    freqs: np.ndarray,
    measured_norm: np.ndarray,
    B_base_mT: np.ndarray,
    linewidths: np.ndarray | float,
    contrasts: np.ndarray | float,
    hyperfine_splitting_mhz: float | None,
    nv_axes: np.ndarray | None,
    local_field_l2_penalty: float,
    sqrt_reg: np.ndarray,
) -> np.ndarray:
    """Analytic Jacobian of residuals w.r.t. B_local via chain rule through Lorentzians."""
    B_local = np.asarray(B_local_flat, dtype=float)
    B_total = np.asarray(B_base_mT, dtype=float) + B_local
    resonances, grads = odmr_resonances_and_gradients(B_total, nv_axes=nv_axes)
    # grads: (4, 2, 3) = d(resonance[j,k])/d(B_total[i])

    from .model import D0 as _D0
    lw_arr = np.atleast_1d(np.asarray(linewidths, dtype=float))
    ct_arr = np.atleast_1d(np.asarray(contrasts, dtype=float))
    res_shape = (4, 2)
    if lw_arr.shape == (4,):
        lw_arr = lw_arr[:, None]
    if ct_arr.shape == (4,):
        ct_arr = ct_arr[:, None]
    lw_full = np.broadcast_to(lw_arr, res_shape).copy()
    ct_full = np.broadcast_to(ct_arr, res_shape).copy()

    n_freq = len(freqs)
    # d(predicted)/d(B_local) shape: (n_freq, 3)
    dpred_dB = np.zeros((n_freq, 3), dtype=float)
    for j in range(4):
        for k in range(2):
            hw = lw_full[j, k] / 2.0
            ct = ct_full[j, k]
            center = resonances[j, k]
            d_center_dB = grads[j, k]  # (3,)
            if hyperfine_splitting_mhz is None:
                df_sq = freqs_sq - 2.0 * center * freqs + center**2  # (freqs - center)^2
                denom = df_sq + hw**2
                # d(Lorentzian)/d(center) = ct*hw²*2(f-c)/denom²
                dL_dcenter = ct * hw**2 * 2.0 * (freqs - center) / (denom**2)
                # d(spectrum)/d(B) = -sum over peaks of dL/dcenter * dcenter/dB
                dpred_dB -= dL_dcenter[:, None] * d_center_dB[None, :]
            else:
                half_split = 0.5 * float(hyperfine_splitting_mhz)
                half_ct = 0.5 * ct
                for shift in (-half_split, half_split):
                    c2 = center + shift
                    df_sq = freqs_sq - 2.0 * c2 * freqs + c2**2
                    denom = df_sq + hw**2
                    dL_dcenter = half_ct * hw**2 * 2.0 * (freqs - c2) / (denom**2)
                    dpred_dB -= dL_dcenter[:, None] * d_center_dB[None, :]

    # Clip gradient to match odmr_spectrum's clip(signal, 0, 1) effect: zero gradient
    # where signal is clipped (rare in normal operating regime, defensive).
    jac_spectrum = dpred_dB  # (n_freq, 3)

    if float(local_field_l2_penalty) > 0.0:
        jac_reg = np.diag(sqrt_reg)  # (3, 3)
        return np.vstack([jac_spectrum, jac_reg])
    return jac_spectrum


def fit_static_field(
    freqs: np.ndarray,
    undampened: np.ndarray,
    dampened: np.ndarray,
    B_bias_mag: float = 2.0,
    nv_axes: np.ndarray | None = None,
) -> np.ndarray:
    measured_norm = dampened / undampened
    return fit_static_spectrum(
        freqs=freqs,
        measured_norm=measured_norm,
        B_bias_mag=B_bias_mag,
        nv_axes=nv_axes,
    )


def fit_static_spectrum(
    freqs: np.ndarray,
    measured_norm: np.ndarray,
    B_base_mT: np.ndarray | None = None,
    B_bias_mag: float = 2.0,
    linewidths: np.ndarray | float = 3.0,
    contrasts: np.ndarray | float = 0.05,
    hyperfine_splitting_mhz: float | None = None,
    hyperfine_collapse_tolerance_mhz: float = 0.75,
    max_dips: int = 12,
    min_dips: int = 4,
    local_field_l2_penalty: float = 0.0,
    local_field_bounds_mT: float | tuple[float, float] | None = None,
    return_diagnostics: bool = False,
    nv_axes: np.ndarray | None = None,
) -> np.ndarray | dict[str, object]:
    freqs = np.asarray(freqs, dtype=float)
    measured_norm = np.asarray(measured_norm, dtype=float)
    if freqs.shape != measured_norm.shape:
        raise ValueError(f"freqs shape {freqs.shape} does not match measured_norm shape {measured_norm.shape}")

    B_base = B_bias_mag * optimal_bias_direction() if B_base_mT is None else np.asarray(B_base_mT, dtype=float)
    if B_base.shape != (3,):
        raise ValueError(f"B_base_mT must have shape (3,), got {B_base.shape}")

    if hyperfine_splitting_mhz is not None:
        max_dips = max(max_dips, 24)
    observed_dips = _find_dip_positions(freqs, measured_norm, max_dips=max_dips, min_dips=min_dips)
    init_dips = (
        _collapse_doublet_dips_to_centers(
            observed_dips,
            hyperfine_splitting_mhz=hyperfine_splitting_mhz,
            tolerance_mhz=hyperfine_collapse_tolerance_mhz,
        )
        if hyperfine_splitting_mhz is not None
        else observed_dips
    )

    B_local_init = _match_and_solve(init_dips, B_base, nv_axes=nv_axes)

    # Precompute freqs² once — reused in every residual and Jacobian evaluation (R5).
    freqs_sq = freqs ** 2
    sqrt_reg = np.full(3, float(np.sqrt(float(local_field_l2_penalty))), dtype=float)

    if local_field_bounds_mT is not None:
        if np.isscalar(local_field_bounds_mT):
            bound_value = abs(float(local_field_bounds_mT))
            lower_bound = -bound_value
            upper_bound = bound_value
        else:
            lower_bound, upper_bound = local_field_bounds_mT
            lower_bound = float(lower_bound)
            upper_bound = float(upper_bound)
        B_local_init = np.clip(B_local_init, lower_bound, upper_bound)
        args = (freqs_sq, freqs, measured_norm, B_base, linewidths, contrasts,
                hyperfine_splitting_mhz, nv_axes, local_field_l2_penalty, sqrt_reg)
        result = least_squares(
            _residuals_with_base,
            B_local_init,
            jac=_jac_residuals_with_base,
            args=args,
            method="trf",
            bounds=([lower_bound] * 3, [upper_bound] * 3),
            ftol=1e-12,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=20000,
        )
        x_opt = np.asarray(result.x, dtype=float)
        success = bool(result.success)
        message = str(result.message)
        nfev = int(result.nfev)
        cost = float(np.sum(result.fun ** 2))
    else:
        result = minimize(
            _cost_with_base,
            B_local_init,
            args=(freqs, measured_norm, B_base, linewidths, contrasts,
                  hyperfine_splitting_mhz, nv_axes, local_field_l2_penalty),
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 20000},
        )
        x_opt = np.asarray(result.x, dtype=float)
        success = bool(result.success)
        message = str(result.message)
        nfev = int(getattr(result, "nfev", -1))
        cost = float(result.fun)

    if not return_diagnostics:
        return x_opt

    return {
        "B_local_mT": x_opt,
        "B_total_mT": B_base + x_opt,
        "B_base_mT": B_base,
        "observed_dips_mhz": observed_dips,
        "initialization_dips_mhz": init_dips,
        "n_observed_dips": int(len(observed_dips)),
        "local_field_l2_penalty": float(local_field_l2_penalty),
        "local_field_bounds_mT": None if local_field_bounds_mT is None else local_field_bounds_mT,
        "success": success,
        "message": message,
        "nfev": nfev,
        "cost": cost,
    }
