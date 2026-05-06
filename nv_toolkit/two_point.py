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

    @property
    def sensitivity_norm_mhz_per_mT(self) -> float:
        return float(np.linalg.norm(self.df_dB_vector_mhz_per_mT))

    @property
    def measurement_axis(self) -> np.ndarray:
        norm = self.sensitivity_norm_mhz_per_mT
        if norm == 0.0:
            raise ValueError("Tracked transition has zero magnetic sensitivity at this bias point")
        return self.df_dB_vector_mhz_per_mT / norm


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
) -> float:
    d_current = float(signal_plus) - float(signal_minus)
    d_reference = calibration.baseline_plus - calibration.baseline_minus
    delta_d = d_current - d_reference
    denom = calibration.slope_minus_per_mhz - calibration.slope_plus_per_mhz
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

    def _cost(delta_proj_mT: float) -> float:
        predicted = odmr_spectrum(
            freqs,
            calibration.B_bias_mT + float(delta_proj_mT) * axis,
            linewidths=linewidths,
            contrasts=contrasts,
            hyperfine_splitting_mhz=hyperfine_splitting_mhz,
            nv_axes=nv_axes,
        )
        return float(np.sum((predicted - target) ** 2))

    result = minimize_scalar(
        _cost,
        bounds=(delta_B_linear_mT - float(search_radius_mT), delta_B_linear_mT + float(search_radius_mT)),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success:
        return delta_B_linear_mT
    return float(result.x)


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
