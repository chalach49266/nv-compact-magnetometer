from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .model import odmr_resonances, odmr_resonances_and_gradients, odmr_spectrum


def _flatten_resonances(B_total_mT: np.ndarray, nv_axes: np.ndarray | None = None) -> np.ndarray:
    return np.sort(odmr_resonances(B_total_mT, nv_axes=nv_axes).ravel())


def _resonance_gradient_mhz_per_mT(
    B_total_mT: np.ndarray,
    transition_index: int,
    nv_axes: np.ndarray | None = None,
) -> np.ndarray:
    B_total_mT = np.asarray(B_total_mT, dtype=float)
    if B_total_mT.shape != (3,):
        raise ValueError(f"B_total_mT must have shape (3,), got {B_total_mT.shape}")
    if not 0 <= int(transition_index) < 8:
        raise ValueError(f"transition_index must be in [0, 7], got {transition_index}")

    resonances, gradients = odmr_resonances_and_gradients(B_total_mT, nv_axes=nv_axes)
    freqs_flat = resonances.ravel()
    grads_flat = gradients.reshape(8, 3)
    order = np.argsort(freqs_flat)
    return grads_flat[order[int(transition_index)]]


@dataclass(frozen=True)
class TwoPointCalibration:
    B_bias_mT: np.ndarray
    transition_index: int
    resonance_frequency_mhz: float
    f_minus_mhz: float
    f_plus_mhz: float
    baseline_minus: float
    baseline_plus: float
    slope_minus_per_mhz: float
    slope_plus_per_mhz: float
    df_dB_vector_mhz_per_mT: np.ndarray

    def __post_init__(self) -> None:
        # Precompute constants that would otherwise be recomputed on every sample call.
        norm = float(np.linalg.norm(self.df_dB_vector_mhz_per_mT))
        object.__setattr__(self, "_sensitivity_norm", norm)
        if norm == 0.0:
            object.__setattr__(self, "_measurement_axis", np.zeros(3, dtype=float))
        else:
            object.__setattr__(self, "_measurement_axis", self.df_dB_vector_mhz_per_mT / norm)
        object.__setattr__(self, "_d_reference", self.baseline_plus - self.baseline_minus)
        object.__setattr__(self, "_denom", self.slope_minus_per_mhz - self.slope_plus_per_mhz)

    @property
    def sensitivity_norm_mhz_per_mT(self) -> float:
        return self._sensitivity_norm  # type: ignore[attr-defined]

    @property
    def measurement_axis(self) -> np.ndarray:
        if self._sensitivity_norm == 0.0:  # type: ignore[attr-defined]
            raise ValueError("Tracked transition has zero magnetic sensitivity at this bias point")
        return self._measurement_axis  # type: ignore[attr-defined]


def build_two_point_calibration(
    B_bias_mT: np.ndarray,
    transition_index: int,
    linewidths: np.ndarray | float = 3.0,
    contrasts: np.ndarray | float = 0.05,
    hyperfine_splitting_mhz: float | None = None,
    search_half_window_mhz: float = 8.0,
    search_points: int = 4001,
    nv_axes: np.ndarray | None = None,
) -> TwoPointCalibration:
    B_bias_mT = np.asarray(B_bias_mT, dtype=float)
    if B_bias_mT.shape != (3,):
        raise ValueError(f"B_bias_mT must have shape (3,), got {B_bias_mT.shape}")
    if search_half_window_mhz <= 0:
        raise ValueError(f"search_half_window_mhz must be > 0, got {search_half_window_mhz}")
    if search_points < 21:
        raise ValueError(f"search_points must be >= 21, got {search_points}")

    centers = _flatten_resonances(B_bias_mT, nv_axes=nv_axes)
    f0 = float(centers[transition_index])
    freqs = np.linspace(f0 - float(search_half_window_mhz), f0 + float(search_half_window_mhz), int(search_points))
    spectrum = odmr_spectrum(
        freqs,
        B_bias_mT,
        linewidths=linewidths,
        contrasts=contrasts,
        hyperfine_splitting_mhz=hyperfine_splitting_mhz,
        nv_axes=nv_axes,
    )
    slopes = np.gradient(spectrum, freqs)

    left_mask = freqs < f0
    right_mask = freqs > f0
    if not np.any(left_mask) or not np.any(right_mask):
        raise RuntimeError("Slope search window does not contain points on both sides of the resonance")

    left_indices = np.flatnonzero(left_mask)
    right_indices = np.flatnonzero(right_mask)
    i_minus = int(left_indices[int(np.argmax(np.abs(slopes[left_mask])))])
    i_plus = int(right_indices[int(np.argmax(np.abs(slopes[right_mask])))])

    grad = _resonance_gradient_mhz_per_mT(B_bias_mT, transition_index, nv_axes=nv_axes)
    return TwoPointCalibration(
        B_bias_mT=B_bias_mT.copy(),
        transition_index=int(transition_index),
        resonance_frequency_mhz=f0,
        f_minus_mhz=float(freqs[i_minus]),
        f_plus_mhz=float(freqs[i_plus]),
        baseline_minus=float(spectrum[i_minus]),
        baseline_plus=float(spectrum[i_plus]),
        slope_minus_per_mhz=float(slopes[i_minus]),
        slope_plus_per_mhz=float(slopes[i_plus]),
        df_dB_vector_mhz_per_mT=grad,
    )


def estimate_delta_f_mhz(
    signal_minus: float,
    signal_plus: float,
    calibration: TwoPointCalibration,
    *,
    subtract_baseline: bool = True,
) -> float:
    # subtract_baseline=True (default) preserves the original behaviour: subtract the
    # per-block baseline differential captured at calibration time so delta_f reports
    # the SHIFT relative to that baseline. Set False to skip the subtraction and
    # observe the raw differential / slope ratio (useful for experiments comparing
    # whether the baseline term contributes meaningful information).
    d_current = float(signal_plus) - float(signal_minus)
    if subtract_baseline:
        delta_d = d_current - calibration._d_reference  # type: ignore[attr-defined]
    else:
        delta_d = d_current
    denom = calibration._denom  # type: ignore[attr-defined]
    if abs(denom) < 1e-12:
        raise ValueError("Calibration slopes are too small for a stable delta_f estimate")
    return float(delta_d / denom)


def estimate_delta_B_projection_mT(
    signal_minus: float,
    signal_plus: float,
    calibration: TwoPointCalibration,
    linewidths: np.ndarray | float = 3.0,
    contrasts: np.ndarray | float = 0.05,
    hyperfine_splitting_mhz: float | None = None,
    nv_axes: np.ndarray | None = None,
    search_radius_mT: float = 0.2,
) -> float:
    delta_f_mhz = estimate_delta_f_mhz(signal_minus, signal_plus, calibration)
    sensitivity = calibration.sensitivity_norm_mhz_per_mT
    if sensitivity == 0.0:
        raise ValueError("Calibration has zero magnetic sensitivity")
    delta_B_linear_mT = float(delta_f_mhz / sensitivity)

    if search_radius_mT <= 0:
        return delta_B_linear_mT

    target = np.array([float(signal_minus), float(signal_plus)], dtype=float)
    axis = calibration.measurement_axis
    freqs = np.array([calibration.f_minus_mhz, calibration.f_plus_mhz], dtype=float)
    lo = delta_B_linear_mT - float(search_radius_mT)
    hi = delta_B_linear_mT + float(search_radius_mT)

    # Expand linewidth/contrast to shape (4,2) once; resonances always has shape (4,2).
    _res_shape = (4, 2)
    lw_arr = np.atleast_1d(np.asarray(linewidths, dtype=float))
    ct_arr = np.atleast_1d(np.asarray(contrasts, dtype=float))
    if lw_arr.shape == (4,):
        lw2d_pre = np.repeat(lw_arr, 2).reshape(4, 2)
    else:
        lw2d_pre = np.broadcast_to(lw_arr, _res_shape).copy()
    if ct_arr.shape == (4,):
        ct2d_pre = np.repeat(ct_arr, 2).reshape(4, 2)
    else:
        ct2d_pre = np.broadcast_to(ct_arr, _res_shape).copy()

    def _predicted_and_derivative(delta_proj_mT: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (pred[2], dpred_ddelta[2]) using one resonances+gradients call.

        Gauss-Newton needs J = dpred/ddelta analytically.  Each Lorentzian dip at
        center contributes  -c·hw²/((f-center)²+hw²), whose derivative w.r.t. center
        is  +2c·hw²·(f-center)/((f-center)²+hw²)².  Chain through dcenter/ddelta
        = (grads @ axis)[j,k] to get dpred_k/ddelta summed over all resonances.
        Fully vectorized over all 8 resonances to avoid Python-level loops.
        """
        B_total = calibration.B_bias_mT + float(delta_proj_mT) * axis
        resonances, grads = odmr_resonances_and_gradients(B_total, nv_axes=nv_axes)
        # dres_ddelta[j,k] = d(center_jk)/d(delta), shape (4,2) → flatten to (8,)
        dres_ddelta_flat = (grads @ axis).ravel()   # (8,)
        centers_flat = resonances.ravel()            # (8,)

        # hw2d and ct2d: shape (8,) after broadcast+flatten
        hw_flat = (lw2d_pre / 2.0).ravel()          # (8,)
        ct_flat = ct2d_pre.ravel()                   # (8,)

        if hyperfine_splitting_mhz is None:
            # centers shape (8,1), freqs shape (1,2): detuning u = f_k - ctr_j, shape (8,2)
            d_all = freqs[None, :] - centers_flat[:, None]         # (8, 2)
            denom_all = d_all ** 2 + hw_flat[:, None] ** 2        # (8, 2)
            lor_all = hw_flat[:, None] ** 2 / denom_all            # (8, 2)
            # pred = 1 - sum_over_8(c * lor)  (2,)
            pred = np.ones(2, dtype=float) - (ct_flat[:, None] * lor_all).sum(axis=0)
            # dpred/dctr = -c * 2*u*hw²/denom²; chain: dpred/ddelta = dpred/dctr * dctr/ddelta
            dpred = -(ct_flat[:, None] * 2.0 * hw_flat[:, None] ** 2
                      * d_all / (denom_all ** 2)
                      * dres_ddelta_flat[:, None]).sum(axis=0)
        else:
            half_split = 0.5 * float(hyperfine_splitting_mhz)
            # Two sub-peaks per resonance: centers ± half_split, contrast halved.
            # Both sub-peaks translate together with the resonance center, so dc_sub/ddelta = dres_ddelta.
            ctr_lo = centers_flat - half_split    # (8,)
            ctr_hi = centers_flat + half_split    # (8,)
            ct_hf = ct_flat * 0.5                 # (8,)
            pred = np.ones(2, dtype=float)
            dpred = np.zeros(2, dtype=float)
            for ctr_sub in (ctr_lo, ctr_hi):
                d_all = freqs[None, :] - ctr_sub[:, None]             # (8, 2)
                denom_all = d_all ** 2 + hw_flat[:, None] ** 2        # (8, 2)
                pred -= (ct_hf[:, None] * hw_flat[:, None] ** 2 / denom_all).sum(axis=0)
                dpred -= (ct_hf[:, None] * 2.0 * hw_flat[:, None] ** 2
                          * d_all / (denom_all ** 2)
                          * dres_ddelta_flat[:, None]).sum(axis=0)

        pred = np.clip(pred, 0.0, 1.0)
        return pred, dpred

    # Gauss-Newton from linear seed: δ ← δ − (J·r)/(J·J)
    # Typically converges in 1–2 steps; cap at 5.
    delta = delta_B_linear_mT
    cost_seed = None
    converged = False
    tol = 1e-10
    tiny = 1e-30
    for _ in range(5):
        delta_clamped = float(np.clip(delta, lo, hi))
        pred, J = _predicted_and_derivative(delta_clamped)
        residuals = pred - target       # (2,)
        Jr = float(np.dot(J, residuals))
        JJ = float(np.dot(J, J))
        if cost_seed is None:
            cost_seed = float(np.sum(residuals ** 2))
        if abs(Jr) < 1e-12:
            converged = True
            delta = delta_clamped
            break
        step = -Jr / max(JJ, tiny)
        delta_new = float(np.clip(delta_clamped + step, lo, hi))
        if abs(delta_new - delta_clamped) < tol:
            converged = True
            delta = delta_new
            break
        delta = delta_new

    delta_final = float(np.clip(delta, lo, hi))

    # Only fall back to minimize_scalar when Newton failed (didn't converge or cost rose).
    if not converged:
        pred_final, _ = _predicted_and_derivative(delta_final)
        cost_final = float(np.sum((pred_final - target) ** 2))
        if cost_seed is None or cost_final > cost_seed:
            def _cost(d: float) -> float:
                B_t = calibration.B_bias_mT + float(d) * axis
                p = odmr_spectrum(
                    freqs, B_t,
                    linewidths=linewidths, contrasts=contrasts,
                    hyperfine_splitting_mhz=hyperfine_splitting_mhz, nv_axes=nv_axes,
                )
                return float(np.sum((p - target) ** 2))

            result = minimize_scalar(_cost, bounds=(lo, hi), method="bounded",
                                     options={"xatol": 1e-8})
            if result.success and float(result.fun) < cost_final:
                return float(result.x)

    return delta_final


def estimate_delta_B_vector_mT(
    signal_minus: float,
    signal_plus: float,
    calibration: TwoPointCalibration,
) -> np.ndarray:
    delta_B_proj_mT = estimate_delta_B_projection_mT(signal_minus, signal_plus, calibration)
    return calibration.measurement_axis * delta_B_proj_mT


def normalised_signal_from_counts(
    signal_counts: float,
    reference_counts: float,
) -> float:
    if reference_counts <= 0:
        raise ValueError(f"reference_counts must be > 0, got {reference_counts}")
    return float(signal_counts) / float(reference_counts)


def estimate_delta_B_projection_from_counts_mT(
    signal_minus_counts: float,
    reference_minus_counts: float,
    signal_plus_counts: float,
    reference_plus_counts: float,
    calibration: TwoPointCalibration,
) -> float:
    signal_minus = normalised_signal_from_counts(signal_minus_counts, reference_minus_counts)
    signal_plus = normalised_signal_from_counts(signal_plus_counts, reference_plus_counts)
    return estimate_delta_B_projection_mT(signal_minus, signal_plus, calibration)


def simulate_two_point_signals(
    B_total_mT: np.ndarray,
    calibration: TwoPointCalibration,
    linewidths: np.ndarray | float = 3.0,
    contrasts: np.ndarray | float = 0.05,
    hyperfine_splitting_mhz: float | None = None,
    nv_axes: np.ndarray | None = None,
) -> tuple[float, float]:
    freqs = np.array([calibration.f_minus_mhz, calibration.f_plus_mhz], dtype=float)
    signal = odmr_spectrum(
        freqs,
        np.asarray(B_total_mT, dtype=float),
        linewidths=linewidths,
        contrasts=contrasts,
        hyperfine_splitting_mhz=hyperfine_splitting_mhz,
        nv_axes=nv_axes,
    )
    return float(signal[0]), float(signal[1])
