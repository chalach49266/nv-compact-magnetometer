"""Variant A: Fit local B field and NV axis orientation from multiple ODMR spectra.

This module implements the "know nothing" workflow: given 4+ ODMR spectra
recorded at different known applied bias fields, simultaneously solve for
the local B field (3 params) and NV axis orientation (3 Euler angles).

Reference: Solve_Bulk_Diamond_Four_Mag.m from NV_Centers.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# --- Physical constants ---
MU_E = 28.04  # electron gyromagnetic ratio, GHz/T (mu_e = gamma_e / 2π)
D0 = 2.8692   # zero-field splitting, GHz


def _tetrahedral_axes(theta: float, phi: float, alpha: float) -> np.ndarray:
    """Build 4 NV axes from 3 Euler angles (tetrahedral parametrization).

    Parameters
    ----------
    theta : float
        Polar angle of axis 1 (radians).
    phi : float
        Azimuthal angle of axis 1 (radians).
    alpha : float
        Rotation angle around axis 1 controlling the other 3 axes (radians).

    Returns
    -------
    axes : np.ndarray, shape (4, 3)
        Unit vectors for the 4 NV axes.
    """
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    ca, sa = np.cos(alpha), np.sin(alpha)

    # Perpendicular direction in the xy-plane
    perp = np.array([sp, -cp, 0.0])
    # Third orthogonal direction
    cross = np.array([-st * cp, -st * sp, ct])

    ax1 = np.array([ct * cp, ct * sp, st])
    ax2 = -ax1 / 3.0 - (2.0 * np.sqrt(2.0) / 3.0) * (ca * perp + sa * cross)
    ax3 = -ax1 / 3.0 - (2.0 * np.sqrt(2.0) / 3.0) * (
        np.cos(alpha - 2.0 * np.pi / 3.0) * perp
        + np.sin(alpha - 2.0 * np.pi / 3.0) * cross
    )
    ax4 = -ax1 / 3.0 - (2.0 * np.sqrt(2.0) / 3.0) * (
        np.cos(alpha + 2.0 * np.pi / 3.0) * perp
        + np.sin(alpha + 2.0 * np.pi / 3.0) * cross
    )

    axes = np.stack([ax1, ax2, ax3, ax4], axis=0)
    # Normalize (should already be unit, but ensure)
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    return axes / norms


def _odmr_splittings(
    b_total_tesla: np.ndarray,
    axes: np.ndarray,
) -> np.ndarray:
    """Compute the 4 ODMR splittings for a given total B field and NV axes.

    Parameters
    ----------
    b_total_tesla : np.ndarray, shape (3,)
        Total magnetic field in Tesla (= B_applied + B_local).
    axes : np.ndarray, shape (4, 3)
        NV axis unit vectors.

    Returns
    -------
    splittings : np.ndarray, shape (4,)
        4 ODMR splittings (f_high - f_low) in GHz, sorted descending.
    """
    b_norm_sq = float(np.dot(b_total_tesla, b_total_tesla))
    b_axial = axes @ b_total_tesla  # shape (4,) — projection onto each axis

    deltas = np.zeros(4, dtype=float)
    for j in range(4):
        bz_sq = float(b_axial[j]) ** 2
        # Cubic: Δ³ - 2D·Δ² + (D² - μ²|B|²)·Δ + μ²(|B|² - Bz²)·D = 0
        coeffs = [
            1.0,
            -2.0 * D0,
            D0**2 - MU_E**2 * b_norm_sq,
            MU_E**2 * (b_norm_sq - bz_sq) * D0,
        ]
        roots = np.roots(coeffs)
        # The two physically meaningful roots are the transition energies
        real_roots = np.sort(np.real(roots[np.isreal(roots)]))
        if len(real_roots) >= 2:
            deltas[j] = real_roots[-1] - real_roots[-2]
        else:
            deltas[j] = 0.0

    return -np.sort(-deltas)  # descending


def _residual(
    params: np.ndarray,
    applied_fields_tesla: list[np.ndarray],
    observed_splittings_ghz: list[np.ndarray],
) -> np.ndarray:
    """Compute residual vector for least-squares fitting.

    Parameters
    ----------
    params : np.ndarray, shape (6,)
        [dBx_uT, dBy_uT, dBz_uT, theta, phi, alpha]
        B_local in microtesla, angles in radians.
    applied_fields_tesla : list of np.ndarray
        Applied bias fields in Tesla, one per spectrum.
    observed_splittings_ghz : list of np.ndarray
        Observed 4 splittings per spectrum in GHz.

    Returns
    -------
    residual : np.ndarray
        Flattened residual vector (predicted - observed), scaled by 1e7.
    """
    dBx, dBy, dBz, theta, phi, alpha = params
    b_local_tesla = np.array([dBx, dBy, dBz], dtype=float) * 1e-6
    axes = _tetrahedral_axes(float(theta), float(phi), float(alpha))

    residual_parts = []
    for b_applied, observed in zip(applied_fields_tesla, observed_splittings_ghz):
        b_total = np.asarray(b_applied, dtype=float) + b_local_tesla
        predicted = _odmr_splittings(b_total, axes)
        residual_parts.append(predicted - np.asarray(observed, dtype=float))

    return np.concatenate(residual_parts) * 1e7


def fit_blind(
    applied_fields_tesla: list[np.ndarray],
    observed_splittings_ghz: list[np.ndarray],
    *,
    b_local_initial_uT: tuple[float, float, float] = (100.0, 100.0, 100.0),
    angles_initial: tuple[float, float, float] = (0.6155, 1.000, 0.5236),
    b_local_bounds_uT: tuple[float, float] = (-1000.0, 1000.0),
    angle_bounds: tuple[float, float] = (-np.pi, np.pi),
    num_restarts: int = 50,
    seed: int | None = None,
) -> dict[str, object]:
    """Fit local B field and NV axis orientation from multiple ODMR spectra.

    This is the "know nothing" workflow: no axis model, no bias field
    guess — just 4+ spectra at different known applied fields.

    Parameters
    ----------
    applied_fields_tesla : list of np.ndarray
        Applied bias field for each spectrum, each shape (3,), in Tesla.
    observed_splittings_ghz : list of np.ndarray
        Observed 4 ODMR splittings per spectrum, each shape (4,), in GHz,
        sorted descending.
    b_local_initial_uT : tuple
        Initial guess for local B field in microtesla.
    angles_initial : tuple
        Initial guess for (theta, phi, alpha) in radians.
    b_local_bounds_uT : tuple
        (lower, upper) bounds for each B_local component in microtesla.
    angle_bounds : tuple
        (lower, upper) bounds for each angle in radians.
    num_restarts : int
        Number of random restarts for MultiStart-like global optimization.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Keys: b_local_uT, b_local_tesla, theta, phi, alpha, axes,
        cost, success, message, num_spectra, residual_rms_ghz.
    """
    n_spectra = len(applied_fields_tesla)
    if n_spectra < 3:
        raise ValueError(f"Need at least 3 spectra, got {n_spectra}")
    if len(observed_splittings_ghz) != n_spectra:
        raise ValueError(
            f"Mismatch: {n_spectra} applied fields vs {len(observed_splittings_ghz)} splitting sets"
        )

    rng = np.random.RandomState(seed)
    b_lo, b_hi = float(b_local_bounds_uT[0]), float(b_local_bounds_uT[1])
    a_lo, a_hi = float(angle_bounds[0]), float(angle_bounds[1])

    x0 = np.array(
        [*b_local_initial_uT, *angles_initial], dtype=float
    )
    bounds = [
        (b_lo, b_hi), (b_lo, b_hi), (b_lo, b_hi),
        (a_lo, a_hi), (a_lo, a_hi), (a_lo, a_hi),
    ]

    def cost(params: np.ndarray) -> float:
        r = _residual(params, applied_fields_tesla, observed_splittings_ghz)
        return float(np.dot(r, r))

    best_x = x0.copy()
    best_cost = cost(x0)
    best_success = False
    best_message = ""

    # Try the initial guess
    res = minimize(cost, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 20000})
    if res.fun < best_cost:
        best_x, best_cost = res.x, res.fun
        best_success, best_message = res.success, res.message

    # Random restarts
    for _ in range(num_restarts):
        x_rand = np.array([
            rng.uniform(b_lo, b_hi),
            rng.uniform(b_lo, b_hi),
            rng.uniform(b_lo, b_hi),
            rng.uniform(a_lo, a_hi),
            rng.uniform(a_lo, a_hi),
            rng.uniform(a_lo, a_hi),
        ])
        res = minimize(cost, x_rand, method="L-BFGS-B", bounds=bounds, options={"maxiter": 20000})
        if res.fun < best_cost:
            best_x, best_cost = res.x, res.fun
            best_success, best_message = res.success, res.message

    dBx, dBy, dBz, theta, phi, alpha = best_x
    axes = _tetrahedral_axes(float(theta), float(phi), float(alpha))
    b_local_uT = np.array([dBx, dBy, dBz], dtype=float)
    b_local_tesla = b_local_uT * 1e-6

    # Compute final residuals
    n_obs = 4 * n_spectra
    rms_ghz = np.sqrt(best_cost / n_obs) * 1e-7 if n_obs > 0 else float("inf")

    return {
        "b_local_uT": b_local_uT,
        "b_local_tesla": b_local_tesla,
        "theta": float(theta),
        "phi": float(phi),
        "alpha": float(alpha),
        "axes": axes,
        "cost": float(best_cost),
        "success": bool(best_success),
        "message": str(best_message),
        "num_spectra": n_spectra,
        "residual_rms_ghz": rms_ghz,
    }


def splittings_from_transition_frequencies(
    freq_pairs_mhz: list[tuple[float, float, float, float]],
) -> np.ndarray:
    """Convert 8 transition-center frequencies to 4 ODMR splittings.

    Following the MATLAB convention: the 8 resonances are paired as
    (2-1, 4-3, 6-5, 8-7) and each splitting is f_high - f_low.

    Parameters
    ----------
    freq_pairs_mhz : list of (float, float, float, float)
        Four splitting values in MHz, one per NV axis.

    Returns
    -------
    splittings_ghz : np.ndarray, shape (4,)
        Four splittings in GHz, sorted descending.
    """
    splits = np.asarray(freq_pairs_mhz, dtype=float) * 1e-3  # MHz → GHz
    return -np.sort(-splits)  # descending


def extract_splittings_from_spectrum_csv(
    csv_path: str,
    *,
    max_dips: int = 8,
    min_dips: int = 4,
    min_distance_mhz: float = 8.0,
) -> np.ndarray:
    """Load a toolkit-format spectrum CSV and extract 4 ODMR splittings.

    Convenience wrapper that runs peak detection + transition-center
    extraction on a single spectrum file, returning the 4 splittings
    suitable for ``fit_blind``.

    Parameters
    ----------
    csv_path : str
        Path to toolkit-format CSV (columns: frequency_mhz, normalized_intensity).
    max_dips, min_dips, min_distance_mhz :
        Passed through to peak detection.

    Returns
    -------
    splittings_ghz : np.ndarray, shape (4,)
        Four ODMR splittings in GHz, sorted descending.
    """
    from .io import load_spectrum_csv
    from .peaks import find_refined_dip_positions

    batch = load_spectrum_csv(csv_path)
    freqs = np.asarray(batch.freqs_mhz, dtype=float)
    spectrum = np.asarray(batch.spectra[0], dtype=float)

    dips = find_refined_dip_positions(
        freqs, spectrum,
        max_dips=max_dips, min_dips=min_dips,
        min_distance_mhz=min_distance_mhz,
    )
    dips = np.sort(dips)

    if len(dips) < 8:
        raise ValueError(
            f"Need 8 transition centers for blind fitting, got {len(dips)} from {csv_path}"
        )
    if len(dips) > 8:
        # Take the 8 strongest (by prominence proxy)
        dips = dips[:8]

    # Pair: (2-1, 4-3, 6-5, 8-7) → 4 splittings in MHz
    splits_mhz = np.array([
        dips[1] - dips[0],
        dips[3] - dips[2],
        dips[5] - dips[4],
        dips[7] - dips[6],
    ])
    return -np.sort(-splits_mhz * 1e-3)  # descending, GHz
