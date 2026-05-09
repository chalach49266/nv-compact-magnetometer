from __future__ import annotations

import numpy as np


D0 = 2870.0
GAMMA = 28.03

NV_AXES = np.array(
    [
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ],
    dtype=float,
) / np.sqrt(3.0)

QDM_NV_AXES = np.array(
    [
        [np.sqrt(2.0), 0.0, 1.0],
        [0.0, np.sqrt(2.0), -1.0],
        [-np.sqrt(2.0), 0.0, 1.0],
        [0.0, -np.sqrt(2.0), -1.0],
    ],
    dtype=float,
) / np.sqrt(3.0)

def _resolved_nv_axes(nv_axes: np.ndarray | None = None) -> np.ndarray:
    # Fast path: module-level constants are already normalized
    if nv_axes is None or nv_axes is NV_AXES:
        return NV_AXES
    if nv_axes is QDM_NV_AXES:
        return QDM_NV_AXES
    axes = np.asarray(nv_axes, dtype=float)
    if axes.shape != (4, 3):
        raise ValueError(f"nv_axes must have shape (4, 3), got {axes.shape}")
    norms = np.linalg.norm(axes, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("nv_axes must not contain zero-length vectors")
    return axes / norms[:, None]


def named_nv_axes(name: str | None = None) -> np.ndarray:
    if name is None:
        return NV_AXES.copy()

    normalized = str(name).strip().lower()
    if normalized in {"111", "canonical", "default"}:
        return NV_AXES.copy()
    if normalized == "qdm":
        return QDM_NV_AXES.copy()
    raise ValueError(f"Unsupported NV-axis preset: {name}")


def _cubic_roots(B_mag_sq: float, B_axial: float) -> np.ndarray:
    # Viète's trigonometric formula for the depressed cubic.
    # Polynomial: λ³ − 2D·λ² + (D²−γ²|B|²)·λ + γ²(|B|²−B∥²)·D = 0
    p2 = D0**2 - GAMMA**2 * float(B_mag_sq)
    p3 = GAMMA**2 * (float(B_mag_sq) - float(B_axial) ** 2) * D0

    # Depress via λ = t + 2D/3 → t³ + pt + q = 0
    shift = 2.0 * D0 / 3.0
    p = p2 - (D0**2) * 4.0 / 3.0
    q = (2.0 * D0**3) / 13.5 - D0 * p2 / 3.0 + p3
    discriminant = -(4.0 * p**3 + 27.0 * q**2)
    if discriminant >= 0.0:
        # Three distinct real roots via arccosine form
        m = 2.0 * np.sqrt(-p / 3.0)
        arg = np.clip(3.0 * q / (p * m), -1.0, 1.0)
        theta = np.arccos(arg)
        roots = m * np.cos((theta - 2.0 * np.pi * np.arange(3)) / 3.0) + shift
        return np.sort(roots)
    # Fallback for non-real-root regime (should not occur in normal NV operating range)
    coeffs = [1.0, -2.0 * D0, p2, p3]
    return np.sort(np.real(np.roots(coeffs)))


def odmr_resonances_and_gradients(
    B_total: np.ndarray,
    nv_axes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    B_total = np.asarray(B_total, dtype=float)
    B_mag_sq = float(np.dot(B_total, B_total))
    axes = _resolved_nv_axes(nv_axes)  # (4, 3)

    resonances = np.zeros((4, 2), dtype=float)
    gradients = np.zeros((4, 2, 3), dtype=float)

    for j in range(4):
        n_j = axes[j]
        # Use scalar dot product per axis for consistent FP results with odmr_resonances
        B_axial = float(np.dot(B_total, n_j))
        roots = _cubic_roots(B_mag_sq, B_axial)

        resonances[j, 0] = roots[1] - roots[0]
        resonances[j, 1] = roots[2] - roots[0]

        root_grads = np.zeros((3, 3), dtype=float)
        for k, lam in enumerate(roots):
            dP_dlam = 3 * lam**2 - 4 * D0 * lam + (D0**2 - GAMMA**2 * B_mag_sq)
            if abs(dP_dlam) < 1e-30:
                continue
            # dP/dBi = 2γ²(Bi(D−λ) − D·B∥·n_j_i), shape (3,)
            dP_dB = 2 * GAMMA**2 * (B_total * (D0 - lam) - D0 * B_axial * n_j)
            root_grads[k] = -dP_dB / dP_dlam

        gradients[j, 0] = root_grads[1] - root_grads[0]
        gradients[j, 1] = root_grads[2] - root_grads[0]

    return resonances, gradients


def invert_odmr_pair(
    f1_mhz: float,
    f2_mhz: float,
) -> tuple[float, float, float]:
    f1, f2 = float(f1_mhz), float(f2_mhz)
    lam_min = (2 * D0 - f1 - f2) / 3.0
    lam_mid = lam_min + f1
    lam_max = lam_min + f2

    sum_pairs = lam_min * lam_mid + lam_min * lam_max + lam_mid * lam_max
    B_total_sq = (D0**2 - sum_pairs) / GAMMA**2
    B_trans_sq = -lam_min * lam_mid * lam_max / (GAMMA**2 * D0)
    B_axial_sq = max(B_total_sq - B_trans_sq, 0.0)

    return (
        float(np.sqrt(max(B_axial_sq, 0.0))),
        float(np.sqrt(max(B_trans_sq, 0.0))),
        float(np.sqrt(max(B_total_sq, 0.0))),
    )


def odmr_resonances(B_total: np.ndarray, nv_axes: np.ndarray | None = None) -> np.ndarray:
    B_total = np.asarray(B_total, dtype=float)
    B_mag_sq = float(np.dot(B_total, B_total))
    axes = _resolved_nv_axes(nv_axes)
    resonances = np.zeros((4, 2), dtype=float)
    for j in range(4):
        roots = _cubic_roots(B_mag_sq, float(np.dot(B_total, axes[j])))
        resonances[j, 0] = roots[1] - roots[0]
        resonances[j, 1] = roots[2] - roots[0]
    return resonances


def odmr_spectrum(
    freqs: np.ndarray,
    B_total: np.ndarray,
    linewidths: np.ndarray | float = 3.0,
    contrasts: np.ndarray | float = 0.05,
    hyperfine_splitting_mhz: float | None = None,
    nv_axes: np.ndarray | None = None,
) -> np.ndarray:
    res = odmr_resonances(B_total, nv_axes=nv_axes)

    def _expand(p: object) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(p, dtype=float))
        if arr.shape == (4,):
            arr = arr[:, None]
        return np.broadcast_to(arr, res.shape).copy()

    lw = _expand(linewidths)
    ct = _expand(contrasts)

    signal = np.ones(len(freqs))
    for j in range(4):
        for k in range(2):
            hw = lw[j, k] / 2.0
            if hyperfine_splitting_mhz is None:
                signal -= ct[j, k] * hw**2 / ((freqs - res[j, k]) ** 2 + hw**2)
            else:
                half_split = 0.5 * float(hyperfine_splitting_mhz)
                half_contrast = 0.5 * ct[j, k]
                for center in (res[j, k] - half_split, res[j, k] + half_split):
                    signal -= half_contrast * hw**2 / ((freqs - center) ** 2 + hw**2)

    return np.clip(signal, 0.0, 1.0)


def optimal_bias_direction() -> np.ndarray:
    direction = np.array([1.0, 2.0, 4.0], dtype=float)
    return direction / np.linalg.norm(direction)


def bias_projections(B_bias_mag: float = 1.0, nv_axes: np.ndarray | None = None) -> np.ndarray:
    axes = _resolved_nv_axes(nv_axes)
    return np.abs(axes @ (float(B_bias_mag) * optimal_bias_direction()))
