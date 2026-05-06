"""
Three extensions to the two-point lock-in modulation in twopoint_lockin.py.

Extension 1 — Sinusoidal FM approximation  (run_sinewave_lockin)
Extension 2 — PID feedback / resonance tracking  (run_feedback_lockin)
Extension 3 — Sensitivity-optimal slope fraction  (optimal_slope_fraction,
                                                    compare_slope_fractions)

None of these require FPGA firmware changes. All use LockinODMR with
raw_data=True to access per-rep data at hardware rate.
"""

from __future__ import annotations

from copy import copy
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import qickdawg as qd


# ══════════════════════════════════════════════════════════════════════════════
# EXTENSION 1 — Sinusoidal FM Lock-In Approximation
# ══════════════════════════════════════════════════════════════════════════════

def _sinewave_weights(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute sine (in-phase) and cosine (quadrature) demodulation weights for
    N linearly-spaced frequencies covering [f₀-δ, f₀+δ].

    Physics:
        For true sinusoidal FM: f(t) = f₀ + δ·sin(ωₘt).
        At time t, the phase is θ = ωₘt, so sin(θ) = (f(t)-f₀)/δ.
        Uniform frequency sampling maps f[k] = f₀ + δ·u[k] where u[k] = 2k/(N-1)-1.
        In-phase  weight: w_sin[k]  = u[k]           = sin(arcsin(u[k]))
        Quadrature weight: w_cos[k] = √(1 - u[k]²)   = cos(arcsin(u[k]))

    For N=2: w_sin = [-1, +1], which gives X = -lockin_signal from twopoint.
    For N→∞: approaches the continuous FM demodulation.

    Returns
    -------
    w_sin : shape (N,)  in-phase weights,   antisymmetric, field-sensitive
    w_cos : shape (N,)  quadrature weights, symmetric,    proportional to ODMR depth
    """
    k = np.arange(n_points, dtype=float)
    u = 2.0 * k / (n_points - 1) - 1.0   # ranges from -1 to +1
    w_sin = u
    w_cos = np.sqrt(np.maximum(0.0, 1.0 - u ** 2))
    return w_sin, w_cos


def run_sinewave_lockin(
    config,
    *,
    center_fMHz: float = 2870.91,
    modulation_depth_mhz: float | None = None,
    fwhm_mhz: float = 17.47,
    n_points: int = 8,
    duration_sec: float | None = 60.0,
    reps_per_batch: int = 200,
    lowpass_cutoff_hz: float | None = 50.0,
    csv_filename: str | None = None,
) -> pd.DataFrame:
    """
    Sinusoidal FM lock-in: N-point sweep demodulated with sine weights.

    Sweeps N equally-spaced frequencies across [f₀-δ, f₀+δ] each rep and
    applies sine-weighted demodulation to extract the in-phase (X) component,
    which is proportional to dPL/df at f₀ and therefore proportional to B.

    The two-point method (twopoint_lockin.py) is the special case N=2.
    For N≥4 the sine approximation improves and the X output better reflects
    the derivative of the ODMR line rather than a point difference.

    Parameters
    ----------
    config : NVConfiguration
        Base hardware configuration. Not modified.
    center_fMHz : float
        ODMR resonance center f₀ in MHz.
    modulation_depth_mhz : float | None
        δ — half-range of FM modulation. If None, defaults to FWHM/2.
    fwhm_mhz : float
        ODMR FWHM (used only when modulation_depth_mhz is None).
    n_points : int
        Number of frequency steps per modulation cycle (N≥2).
        N=2 replicates two-point; N=8 (default) is a good approximation.
    duration_sec : float | None
        Run time. None → run until interrupted.
    reps_per_batch : int
        FPGA reps per acquire() call. Each rep = one N-point sweep cycle.
    lowpass_cutoff_hz : float | None
        Cutoff for zero-phase Butterworth LPF applied to X at end of run.
        None → no filtering.
    csv_filename : str | None
        Save DataFrame to this path after the run.

    Returns
    -------
    pd.DataFrame with columns:
        time_s          hardware timestamp
        X               in-phase demodulated signal (field-sensitive, ADC units)
        Y               quadrature demodulated signal (ODMR depth, ADC units)
        R               magnitude √(X²+Y²)
        X_filtered      low-pass-filtered X (if lowpass_cutoff_hz is set)
        contrast_k*     per-frequency contrast (k=0..N-1, for diagnostics)
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    if modulation_depth_mhz is None:
        modulation_depth_mhz = fwhm_mhz / 2.0

    delta = modulation_depth_mhz
    freqs = np.linspace(center_fMHz - delta, center_fMHz + delta, n_points)
    w_sin, w_cos = _sinewave_weights(n_points)

    cfg = copy(config)
    cfg.mw_start_fMHz = float(freqs[0])
    cfg.mw_end_fMHz   = float(freqs[-1])
    cfg.nsweep_points = n_points
    cfg.reps          = reps_per_batch
    cfg.pre_init      = getattr(config, "pre_init", True)

    prog = qd.LockinODMR(cfg)

    body_tus  = 2.0 * cfg.readout_integration_tus + 2.0 * cfg.relax_delay_tus
    cycle_tus = n_points * body_tus
    batch_sec = reps_per_batch * cycle_tus / 1e6
    cycle_hz  = 1e6 / cycle_tus

    print("Sinusoidal FM lock-in")
    print(f"  Center f₀:          {center_fMHz:.2f} MHz")
    print(f"  Modulation depth δ: {delta:.2f} MHz  (sweep {freqs[0]:.2f} → {freqs[-1]:.2f} MHz)")
    print(f"  N points per cycle: {n_points}")
    print(f"  Frequencies: {np.round(freqs, 2).tolist()}")
    print(f"  Cycle rate:         {cycle_hz:.1f} Hz  ({cycle_tus:.0f} µs/cycle)")
    print(f"  Batch: {reps_per_batch} reps → {batch_sec*1e3:.0f} ms, {reps_per_batch} demodulated pts")
    print("Press Interrupt (■) to stop early.\n")

    scale = float(cfg.readout_integration_treg)
    all_rows: list[dict] = []
    t0 = perf_counter()
    n_total = 0

    try:
        while True:
            t_start = perf_counter() - t0
            # raw shape: (reps, n_points, 2)  [reps, sweep_pt, readout(on/off)]
            raw = prog.acquire(raw_data=True)
            t_end = perf_counter() - t0

            times = t_start + np.linspace(0, t_end - t_start, reps_per_batch, endpoint=False)

            # Respect LockinODMR swap flag (raw_data=True bypasses analyze_data).
            if getattr(cfg, "odmr_lockin_swap_signal_reference", False):
                sig_idx, ref_idx = 1, 0
            else:
                sig_idx, ref_idx = 0, 1

            # contrast per rep per frequency: shape (reps, n_points)
            contrast = (raw[:, :, sig_idx] - raw[:, :, ref_idx]) / scale

            # demodulate: dot product with weights along frequency axis
            X = contrast @ w_sin   # shape (reps,) — in-phase, field-sensitive
            Y = contrast @ w_cos   # shape (reps,) — quadrature, proportional to ODMR depth
            R = np.sqrt(X ** 2 + Y ** 2)

            for i in range(reps_per_batch):
                row: dict = {
                    "time_s": float(times[i]),
                    "X":      float(X[i]),
                    "Y":      float(Y[i]),
                    "R":      float(R[i]),
                }
                for k in range(n_points):
                    row[f"contrast_f{k}"] = float(contrast[i, k])
                all_rows.append(row)

            n_total += reps_per_batch
            if duration_sec is not None and t_end >= duration_sec:
                break

    except KeyboardInterrupt:
        print(f"\nSinusoidal FM stopped — {n_total} points collected.")

    df = pd.DataFrame(all_rows)

    # Optional low-pass filter on X
    if lowpass_cutoff_hz is not None and len(df) > 10:
        from scipy.signal import butter, sosfiltfilt
        nyq = cycle_hz / 2.0
        if lowpass_cutoff_hz < nyq:
            sos = butter(4, lowpass_cutoff_hz / nyq, btype="low", output="sos")
            df["X_filtered"] = sosfiltfilt(sos, df["X"].to_numpy())
        else:
            df["X_filtered"] = df["X"]

    if not df.empty:
        elapsed = float(df["time_s"].iloc[-1])
        print(f"\nSummary ({n_total} points in {elapsed:.1f} s, {n_total/elapsed:.1f} pts/s):")
        print(f"  X (in-phase):  mean={df['X'].mean():.5f}  std={df['X'].std():.5f} ADC")
        print(f"  Y (quadrature):mean={df['Y'].mean():.5f}  std={df['Y'].std():.5f} ADC")

    if csv_filename is not None and not df.empty:
        df.to_csv(csv_filename, index=False)
        print(f"Saved {len(df)} rows to {csv_filename}")

    return df


def sinewave_lockin_from_odmr(odmr_result, config, **kwargs) -> pd.DataFrame:
    """Auto-extract center and FWHM from a prior ODMR sweep and call run_sinewave_lockin."""
    from odmr_sensitivity import estimate_odmr_sensitivity
    sens = estimate_odmr_sensitivity(odmr_result, trace="reference")
    ref = sens["reference"] if isinstance(sens, dict) and "reference" in sens else sens
    center = float(ref["center_mhz"])
    fwhm   = float(ref["fwhm_mhz"])
    print(f"Lorentzian fit → center = {center:.2f} MHz,  FWHM = {fwhm:.2f} MHz")
    return run_sinewave_lockin(config, center_fMHz=center, fwhm_mhz=fwhm, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# EXTENSION 2 — PID Feedback / Resonance Tracking
# ══════════════════════════════════════════════════════════════════════════════

def run_feedback_lockin(
    config,
    *,
    center_fMHz: float = 2870.91,
    fwhm_mhz: float = 17.47,
    slope_fraction: float = 0.5,
    kp: float = 0.05,
    ki: float = 0.0,
    kd: float = 0.0,
    max_correction_mhz: float = 2.0,
    duration_sec: float | None = 60.0,
    reps_per_batch: int = 200,
    csv_filename: str | None = None,
) -> pd.DataFrame:
    """
    PID feedback lock-in: continuously track the ODMR resonance in real time.

    Each batch measures the two-point lock-in signal. Its mean is used as the
    PID error signal (should be zero when the tracking frequency f₀ is centred
    on the resonance). The PID output adjusts f₀ before the next batch,
    causing the measurement to follow the resonance as it shifts with the
    magnetic field.

    The tracked f₀ therefore encodes the magnetic field:
        B(t) = (f₀_tracked(t) - f₀_initial) / γ_NV
        γ_NV = 28 MHz/mT = 28 GHz/T

    PID gains (kp, ki, kd):
        Units: MHz / (ADC contrast unit)
        Rough starting value for kp:
            kp ≈ 1 / (2 × slope_at_f1 [ADC/MHz])
            slope_at_f1 ≈ 0.77 × max_slope [ADC/GHz] × 1e-3 [GHz/MHz]
        Start with kp small (0.01–0.1) and increase until oscillation, then halve.
        Leave ki=kd=0 until the P-loop is stable.

    Parameters
    ----------
    config : NVConfiguration
    center_fMHz : float
        Starting f₀ (the initial best estimate of the resonance center).
    fwhm_mhz : float
        ODMR FWHM (used to set f1, f2 around tracked f₀).
    slope_fraction : float
        Fraction of FWHM to offset f1 and f2 from f₀ (default 0.5).
    kp, ki, kd : float
        PID gains. Start with ki=kd=0.
    max_correction_mhz : float
        Hard clamp on per-batch frequency correction (prevents runaway).
    duration_sec : float | None
        Run time.
    reps_per_batch : int
        FPGA reps per batch. Fewer reps → faster loop, noisier error signal.

    Returns
    -------
    pd.DataFrame with columns:
        time_s              batch mid-point timestamp
        f0_tracked_mhz      current tracked resonance frequency
        f1_mhz, f2_mhz      measurement frequencies used this batch
        lockin_mean         mean lock-in signal (the PID error)
        lockin_std          std of lock-in signal (noise within batch)
        correction_mhz      PID correction applied to f₀
        B_field_nT          estimated B-field shift from initial (nT)
    """
    GAMMA_NV_MHZ_PER_NT = 28e-6        # 28 GHz/T = 28e-6 MHz/nT → 1 MHz = 1/28e-6 nT

    f0         = float(center_fMHz)
    f0_initial = float(center_fMHz)
    integral   = 0.0
    prev_error = 0.0

    body_tus  = 2.0 * config.readout_integration_tus + 2.0 * config.relax_delay_tus
    cycle_tus = 2.0 * body_tus
    batch_sec = reps_per_batch * cycle_tus / 1e6

    print("PID feedback lock-in — resonance tracking")
    print(f"  Initial f₀: {f0:.2f} MHz")
    print(f"  slope_fraction: {slope_fraction}  (f1/f2 at ±{slope_fraction:.3f}×FWHM from f₀)")
    print(f"  PID gains: kp={kp}  ki={ki}  kd={kd}")
    print(f"  Max correction per batch: ±{max_correction_mhz} MHz")
    print(f"  Batch: {reps_per_batch} reps → ~{batch_sec*1e3:.0f} ms")
    print(f"  Duration: {duration_sec} s\n")
    print(f"  Tip: if f₀ drifts quickly, reduce kp. If it doesn't converge, increase kp.")
    print("Press Interrupt (■) to stop early.\n")

    rows: list[dict] = []
    t0 = perf_counter()

    try:
        while True:
            f1 = f0 - slope_fraction * fwhm_mhz
            f2 = f0 + slope_fraction * fwhm_mhz

            cfg = copy(config)
            cfg.mw_start_fMHz = f1
            cfg.mw_end_fMHz   = f2
            cfg.nsweep_points = 2
            cfg.reps          = reps_per_batch
            cfg.pre_init      = getattr(config, "pre_init", True)

            prog = qd.LockinODMR(cfg)
            scale = float(cfg.readout_integration_treg)

            t_start = perf_counter() - t0
            raw = prog.acquire(raw_data=True)   # shape (reps, 2, 2)
            t_end = perf_counter() - t0

            if getattr(cfg, "odmr_lockin_swap_signal_reference", False):
                sig_idx, ref_idx = 1, 0
            else:
                sig_idx, ref_idx = 0, 1

            c_f1   = (raw[:, 0, sig_idx] - raw[:, 0, ref_idx]) / scale
            c_f2   = (raw[:, 1, sig_idx] - raw[:, 1, ref_idx]) / scale
            lockin = c_f1 - c_f2

            error = float(np.mean(lockin))   # PID error: 0 when locked on resonance

            # PID update
            integral   += error
            derivative  = error - prev_error
            correction  = kp * error + ki * integral + kd * derivative
            correction  = float(np.clip(correction, -max_correction_mhz, max_correction_mhz))

            # Sign convention:
            #   lockin > 0  →  resonance shifted right (higher B)
            #   → increase f₀ to follow  →  f0 += positive correction
            f0 += correction
            prev_error = error

            # Convert f₀ shift to B-field
            delta_f_mhz = f0 - f0_initial
            B_nT = delta_f_mhz / GAMMA_NV_MHZ_PER_NT

            rows.append({
                "time_s":           0.5 * (t_start + t_end),
                "f0_tracked_mhz":   f0,
                "f1_mhz":           f1,
                "f2_mhz":           f2,
                "lockin_mean":      error,
                "lockin_std":       float(np.std(lockin)),
                "correction_mhz":   correction,
                "B_field_nT":       B_nT,
            })

            elapsed = t_end
            if len(rows) % 5 == 0:
                print(f"  t={elapsed:.1f}s  f₀={f0:.3f} MHz  error={error:.5f}  "
                      f"corr={correction:.4f} MHz  B≈{B_nT:.2f} nT")

            if duration_sec is not None and t_end >= duration_sec:
                break

    except KeyboardInterrupt:
        print(f"\nFeedback lock-in stopped — {len(rows)} batches collected.")

    df = pd.DataFrame(rows)

    if not df.empty:
        print(f"\nSummary ({len(df)} batches):")
        print(f"  f₀ range: {df['f0_tracked_mhz'].min():.3f} → {df['f0_tracked_mhz'].max():.3f} MHz")
        print(f"  B-field range: {df['B_field_nT'].min():.2f} → {df['B_field_nT'].max():.2f} nT")

    if csv_filename is not None and not df.empty:
        df.to_csv(csv_filename, index=False)
        print(f"Saved {len(df)} rows to {csv_filename}")

    return df


def feedback_lockin_from_odmr(odmr_result, config, **kwargs) -> pd.DataFrame:
    """Auto-extract center and FWHM from a prior ODMR sweep and call run_feedback_lockin."""
    from odmr_sensitivity import estimate_odmr_sensitivity
    sens = estimate_odmr_sensitivity(odmr_result, trace="reference")
    ref  = sens["reference"] if isinstance(sens, dict) and "reference" in sens else sens
    center = float(ref["center_mhz"])
    fwhm   = float(ref["fwhm_mhz"])
    print(f"Lorentzian fit → center = {center:.2f} MHz,  FWHM = {fwhm:.2f} MHz")
    return run_feedback_lockin(config, center_fMHz=center, fwhm_mhz=fwhm, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# EXTENSION 3 — Sensitivity-Optimal Slope Fraction
# ══════════════════════════════════════════════════════════════════════════════

def optimal_slope_fraction(odmr_result=None, *, center_fMHz: float | None = None,
                           fwhm_mhz: float | None = None) -> dict:
    """
    Compute the sensitivity-optimal slope_fraction for a Lorentzian ODMR line.

    For a Lorentzian L(f) the maximum |dL/df| (maximum field sensitivity)
    occurs at the inflection points:
        f_opt = f₀ ± γ/√3 = f₀ ± FWHM / (2√3)
        slope_fraction_opt = 1 / (2√3) ≈ 0.2887

    Parameters
    ----------
    odmr_result : optional
        Result from LockinODMR.acquire() — if provided, Lorentzian parameters
        are extracted automatically.
    center_fMHz : float, optional
        Provide manually if odmr_result is not available.
    fwhm_mhz : float, optional
        Provide manually if odmr_result is not available.

    Returns
    -------
    dict with keys:
        slope_fraction_optimal   float ≈ 0.2887
        slope_fraction_halfmax   float = 0.5  (for comparison)
        center_mhz               float
        fwhm_mhz                 float
        f1_optimal_mhz           float
        f2_optimal_mhz           float
        f1_halfmax_mhz           float
        f2_halfmax_mhz           float
        slope_ratio              float — ratio of slope at 0.289 vs 0.5 (always > 1)
    """
    FRAC_OPT = 1.0 / (2.0 * np.sqrt(3.0))   # ≈ 0.28868

    if odmr_result is not None:
        from odmr_sensitivity import estimate_odmr_sensitivity
        sens   = estimate_odmr_sensitivity(odmr_result, trace="reference")
        ref    = sens["reference"] if isinstance(sens, dict) and "reference" in sens else sens
        center = float(ref["center_mhz"])
        fwhm   = float(ref["fwhm_mhz"])
    elif center_fMHz is not None and fwhm_mhz is not None:
        center = float(center_fMHz)
        fwhm   = float(fwhm_mhz)
    else:
        raise ValueError("Provide either odmr_result or both center_fMHz and fwhm_mhz.")

    gamma = fwhm / 2.0   # Lorentzian half-width

    # Lorentzian slope at a given offset u (in units of gamma):
    # |dL/df| = 2·A·gamma²·|u| / (u²+gamma²)²
    # At u = gamma/sqrt(3):   → maximum
    # At u = gamma  (FWHM/2): → slightly less than maximum
    def lorentz_slope(u_mhz):
        return 2 * gamma ** 2 * abs(u_mhz) / (u_mhz ** 2 + gamma ** 2) ** 2

    u_opt     = gamma / np.sqrt(3.0)
    u_halfmax = gamma                     # = FWHM/2 × slope_fraction=0.5

    slope_opt     = lorentz_slope(u_opt)
    slope_halfmax = lorentz_slope(u_halfmax)
    slope_ratio   = slope_opt / slope_halfmax    # always > 1 (≈ 1.30)

    result = {
        "slope_fraction_optimal": FRAC_OPT,
        "slope_fraction_halfmax": 0.5,
        "center_mhz":            center,
        "fwhm_mhz":              fwhm,
        "f1_optimal_mhz":        center - FRAC_OPT * fwhm,
        "f2_optimal_mhz":        center + FRAC_OPT * fwhm,
        "f1_halfmax_mhz":        center - 0.5 * fwhm,
        "f2_halfmax_mhz":        center + 0.5 * fwhm,
        "slope_ratio":           slope_ratio,
        "sensitivity_gain_pct":  (slope_ratio - 1.0) * 100.0,
    }

    print("Sensitivity-optimal slope fraction")
    print(f"  Optimal (inflection):  slope_fraction = {FRAC_OPT:.4f}")
    print(f"    f1 = {result['f1_optimal_mhz']:.3f} MHz")
    print(f"    f2 = {result['f2_optimal_mhz']:.3f} MHz")
    print(f"  Half-max (default):    slope_fraction = 0.5")
    print(f"    f1 = {result['f1_halfmax_mhz']:.3f} MHz")
    print(f"    f2 = {result['f2_halfmax_mhz']:.3f} MHz")
    print(f"  Slope ratio (opt/halfmax): {slope_ratio:.4f}")
    print(f"  Sensitivity gain from using optimal fraction: {result['sensitivity_gain_pct']:.1f}%")

    return result


def compare_slope_fractions(odmr_result, fractions=(0.289, 0.5), config=None):
    """
    Plot the ODMR reference trace with multiple slope_fraction positions marked,
    and display the Lorentzian slope (sensitivity proxy) at each position.

    Parameters
    ----------
    odmr_result : ItemAttribute
        Result from LockinODMR.acquire().
    fractions : tuple of float
        slope_fraction values to compare. Default: (0.289, 0.5).
    config : optional
        If provided, estimates sensitivity in nT/√Hz using estimate_odmr_sensitivity.
    """
    from odmr_sensitivity import estimate_odmr_sensitivity, lorentzian
    from scipy.optimize import curve_fit

    # odmr_result.frequencies is in MHz per LockinODMR (qick_sweeps.get_sweep_pts).
    freqs = np.asarray(odmr_result.frequencies, dtype=float)
    ref   = np.asarray(odmr_result.reference,   dtype=float)

    sens    = estimate_odmr_sensitivity(odmr_result, trace="reference")
    ref_inf = sens["reference"] if isinstance(sens, dict) and "reference" in sens else sens
    center  = float(ref_inf["center_mhz"])
    fwhm    = float(ref_inf["fwhm_mhz"])
    gamma   = fwhm / 2.0

    # Re-fit Lorentzian in MHz units for plotting overlay.
    fit_f = np.linspace(freqs.min(), freqs.max(), 1000)
    try:
        amp0  = -(ref.max() - ref.min())
        p0    = [center, gamma, amp0, float(ref.mean())]
        popt, _ = curve_fit(lorentzian, freqs, ref, p0=p0, maxfev=20000)
        fit_y = lorentzian(fit_f, *popt)
    except Exception:
        fit_y = None

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(fractions)))

    fig, (ax_odmr, ax_slope) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: ODMR trace with marked positions
    ax_odmr.plot(freqs, ref, color="steelblue", lw=1.5, alpha=0.7, label="ODMR reference (raw)")
    if fit_y is not None:
        ax_odmr.plot(fit_f, fit_y, "k--", lw=1.2, alpha=0.5, label="Lorentzian fit")
    ax_odmr.axvline(center, color="gray", ls=":", lw=1.0, label=f"center {center:.1f} MHz")

    for frac, color in zip(fractions, colors):
        f1 = center - frac * fwhm
        f2 = center + frac * fwhm
        ax_odmr.axvline(f1, color=color, ls="--", lw=1.8,
                        label=f"slope={frac:.3f}: f1={f1:.1f}")
        ax_odmr.axvline(f2, color=color, ls="--", lw=1.8,
                        label=f"slope={frac:.3f}: f2={f2:.1f}")

    ax_odmr.set_xlabel("Frequency (MHz)")
    ax_odmr.set_ylabel("PL (ADC units)")
    ax_odmr.set_title("ODMR reference — slope_fraction comparison")
    ax_odmr.legend(fontsize=8)
    ax_odmr.grid(True, alpha=0.3)

    # Right: Lorentzian slope vs frequency offset
    u_range = np.linspace(0.01, fwhm * 0.8, 400)   # offset from center in MHz

    def lorentz_slope_profile(u):
        return 2 * gamma**2 * u / (u**2 + gamma**2)**2

    slope_profile = lorentz_slope_profile(u_range)
    ax_slope.plot(u_range, slope_profile / slope_profile.max(), "k-", lw=2,
                  label="|dL/df| (normalised)")

    for frac, color in zip(fractions, colors):
        u = frac * fwhm
        slope_val = lorentz_slope_profile(u) / slope_profile.max()
        ax_slope.axvline(u, color=color, ls="--", lw=1.8,
                         label=f"frac={frac:.3f}  slope={slope_val*100:.1f}% of max")
        ax_slope.plot(u, slope_val, "o", color=color, ms=8)

    ax_slope.set_xlabel("Frequency offset from center |f - f₀| (MHz)")
    ax_slope.set_ylabel("|dL/df| (normalised to max)")
    ax_slope.set_title("Lorentzian slope — field sensitivity proxy")
    ax_slope.legend(fontsize=9)
    ax_slope.grid(True, alpha=0.3)

    plt.suptitle(f"Slope fraction comparison  (center={center:.1f} MHz, FWHM={fwhm:.2f} MHz)",
                 fontsize=12)
    plt.tight_layout()
    plt.show()

    # Print summary table
    base_slope = lorentz_slope_profile(fractions[0] * fwhm)
    print(f"\n{'slope_fraction':>15}  {'offset (MHz)':>12}  {'slope (% of max)':>17}  "
          f"{'vs baseline':>12}")
    print("-" * 65)
    for frac in fractions:
        u   = frac * fwhm
        s_pct = lorentz_slope_profile(u) / slope_profile.max() * 100
        if frac == fractions[0]:
            tag = "(baseline)"
        else:
            tag = f"{lorentz_slope_profile(u) / base_slope:.3f}×"
        print(f"{frac:>15.4f}  {u:>12.3f}  {s_pct:>16.2f}%  {tag:>12}")
