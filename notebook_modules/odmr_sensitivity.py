import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


GYROMAGNETIC_RATIO_GHZ_PER_T = 28.0
DEFAULT_POINT_TIME_S = 213e-6


def lorentzian(x, x0, gamma, amplitude, offset):
    return offset + amplitude * gamma**2 / ((x - x0) ** 2 + gamma**2)


def lorentzian_derivative(x, x0, gamma, amplitude):
    denom = (x - x0) ** 2 + gamma**2
    return -2.0 * amplitude * gamma**2 * (x - x0) / denom**2


def _format_sensitivity(value_t_rt_hz):
    abs_value = abs(value_t_rt_hz)
    if abs_value < 1e-6:
        return f"{value_t_rt_hz * 1e9:.2f} nT/sqrt(Hz)"
    if abs_value < 1e-3:
        return f"{value_t_rt_hz * 1e6:.2f} uT/sqrt(Hz)"
    return f"{value_t_rt_hz:.3e} T/sqrt(Hz)"


def _get_point_time_s(point_time_s=None, qd_module=None, config=None):
    if point_time_s is not None:
        return float(point_time_s)
    if qd_module is not None and config is not None:
        total_time_s = float(qd_module.LockinODMR(config).total_time())
        return total_time_s / float(config.nsweep_points)
    return DEFAULT_POINT_TIME_S


def estimate_odmr_sensitivity(
    data,
    *,
    trace="reference",
    noise_window_ghz=(2.70, 2.80),
    point_time_s=None,
    qd_module=None,
    config=None,
):
    # Calculate for primary trace (default: 'reference')
    x = np.asarray(data.frequencies, dtype=float) / 1e3
    y = np.asarray(getattr(data, trace), dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        raise ValueError("No valid points to estimate sensitivity.")

    sort_idx = np.argsort(x)
    xf = x[sort_idx]
    yf = y[sort_idx]

    n = len(yf)
    win = min(101, n if n % 2 == 1 else n - 1)
    if win < 5:
        win = 5
    baseline = savgol_filter(yf, window_length=win, polyorder=3)

    yf_detrended = yf - baseline
    peak_idx = int(np.argmax(np.abs(yf_detrended)))

    x0_guess = float(xf[peak_idx])
    amplitude_guess = float(yf_detrended[peak_idx])
    gamma_guess = float((xf.max() - xf.min()) / 50.0)
    try:
        half = 0.5 * np.abs(amplitude_guess)
        mask_half = np.abs(yf_detrended) >= half
        if mask_half.sum() >= 2:
            xf_half = xf[mask_half]
            gamma_guess = max(float((xf_half.max() - xf_half.min()) / 2.0), gamma_guess)
    except Exception:
        pass
    if np.isclose(amplitude_guess, 0.0):
        amplitude_guess = float(np.max(yf_detrended) - np.min(yf_detrended))
    offset_guess = float(np.median(baseline))

    initial_guess = [x0_guess, gamma_guess, amplitude_guess, offset_guess]
    range_x = float(xf.max() - xf.min())
    lower_bounds = [float(xf.min()), 1e-12, -np.inf, -np.inf]
    upper_bounds = [float(xf.max()), max(range_x, 1e-6), np.inf, np.inf]

    try:
        popt, _ = curve_fit(
            lorentzian,
            xf,
            yf,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=200000,
        )
    except Exception:
        initial_guess[1] = max(gamma_guess, range_x / 20.0)
        popt, _ = curve_fit(
            lorentzian,
            xf,
            yf,
            p0=initial_guess,
            maxfev=200000,
        )

    x0_fit, gamma_fit, amplitude_fit, offset_fit = [float(v) for v in popt]
    fit_y = lorentzian(xf, x0_fit, gamma_fit, amplitude_fit, offset_fit)
    deriv_fit = lorentzian_derivative(xf, x0_fit, gamma_fit, amplitude_fit)
    response_slope = float(np.max(np.abs(deriv_fit)))
    if response_slope <= 0:
        raise ValueError("Fitted slope is zero; choose a clearer ODMR dip.")

    low_ghz, high_ghz = noise_window_ghz
    noise_mask = (xf >= float(low_ghz)) & (xf <= float(high_ghz))
    if noise_mask.sum() >= 2:
        noise_std = float(np.std(yf[noise_mask], ddof=1))
        noise_window_used = (float(low_ghz), float(high_ghz))
    else:
        noise_std = float(np.std(yf - fit_y, ddof=1))
        noise_window_used = None

    point_time_s = _get_point_time_s(point_time_s=point_time_s, qd_module=qd_module, config=config)
    noise_density = noise_std * np.sqrt(point_time_s)
    response = GYROMAGNETIC_RATIO_GHZ_PER_T * response_slope
    sensitivity_t_rt_hz = noise_density / response

    result_dict = {
        "trace": trace,
        "center_ghz": x0_fit,
        "center_mhz": x0_fit * 1e3,
        "gamma_ghz": gamma_fit,
        "fwhm_mhz": 2.0 * gamma_fit * 1e3,
        "noise_std": noise_std,
        "noise_density": float(noise_density),
        "max_slope_per_ghz": response_slope,
        "point_time_s": float(point_time_s),
        "sensitivity_t_rt_hz": float(sensitivity_t_rt_hz),
        "sensitivity_text": _format_sensitivity(sensitivity_t_rt_hz),
        "noise_window_ghz": noise_window_used,
    }

    # Add results for both 'reference' and 'signal' traces, if available
    def get_trace_result(trace_name):
        if not hasattr(data, trace_name):
            return None
        y_local = np.asarray(getattr(data, trace_name), dtype=float)
        valid_local = np.isfinite(x) & np.isfinite(y_local)
        if valid_local.sum() == 0:
            return None
        sort_idx_local = np.argsort(x)
        yf_local = y_local[valid_local][sort_idx_local]
        n_local = len(yf_local)
        win_local = min(101, n_local if n_local % 2 == 1 else n_local - 1)
        if win_local < 5:
            win_local = 5
        baseline_local = savgol_filter(yf_local, window_length=win_local, polyorder=3)
        yf_detrended_local = yf_local - baseline_local
        peak_idx_local = int(np.argmax(np.abs(yf_detrended_local)))
        x0_guess_local = float(x[valid_local][sort_idx_local][peak_idx_local])
        amplitude_guess_local = float(yf_detrended_local[peak_idx_local])
        gamma_guess_local = float((x[valid_local].max() - x[valid_local].min()) / 50.0)
        try:
            half = 0.5 * np.abs(amplitude_guess_local)
            mask_half = np.abs(yf_detrended_local) >= half
            if mask_half.sum() >= 2:
                xf_half = x[valid_local][sort_idx_local][mask_half]
                gamma_guess_local = max(float((xf_half.max() - xf_half.min()) / 2.0), gamma_guess_local)
        except Exception:
            pass
        if np.isclose(amplitude_guess_local, 0.0):
            amplitude_guess_local = float(np.max(yf_detrended_local) - np.min(yf_detrended_local))
        offset_guess_local = float(np.median(baseline_local))
        initial_guess_local = [
            x0_guess_local, gamma_guess_local, amplitude_guess_local, offset_guess_local
        ]
        range_x_local = float(x[valid_local].max() - x[valid_local].min())
        lower_bounds_local = [float(x[valid_local].min()), 1e-12, -np.inf, -np.inf]
        upper_bounds_local = [float(x[valid_local].max()), max(range_x_local, 1e-6), np.inf, np.inf]

        try:
            popt_local, _ = curve_fit(
                lorentzian,
                x[valid_local][sort_idx_local],
                yf_local,
                p0=initial_guess_local,
                bounds=(lower_bounds_local, upper_bounds_local),
                maxfev=200000,
            )
        except Exception:
            initial_guess_local[1] = max(gamma_guess_local, range_x_local / 20.0)
            popt_local, _ = curve_fit(
                lorentzian,
                x[valid_local][sort_idx_local],
                yf_local,
                p0=initial_guess_local,
                maxfev=200000,
            )

        x0_fit_local, gamma_fit_local, amplitude_fit_local, offset_fit_local = [float(v) for v in popt_local]
        fit_y_local = lorentzian(x[valid_local][sort_idx_local], x0_fit_local, gamma_fit_local, amplitude_fit_local, offset_fit_local)
        deriv_fit_local = lorentzian_derivative(x[valid_local][sort_idx_local], x0_fit_local, gamma_fit_local, amplitude_fit_local)
        response_slope_local = float(np.max(np.abs(deriv_fit_local)))

        low_ghz, high_ghz = noise_window_ghz
        noise_mask_local = (x[valid_local][sort_idx_local] >= float(low_ghz)) & (x[valid_local][sort_idx_local] <= float(high_ghz))
        if noise_mask_local.sum() >= 2:
            noise_std_local = float(np.std(yf_local[noise_mask_local], ddof=1))
            noise_window_used_local = (float(low_ghz), float(high_ghz))
        else:
            noise_std_local = float(np.std(yf_local - fit_y_local, ddof=1))
            noise_window_used_local = None

        noise_density_local = noise_std_local * np.sqrt(point_time_s)
        response_local = GYROMAGNETIC_RATIO_GHZ_PER_T * response_slope_local
        sensitivity_t_rt_hz_local = noise_density_local / response_local

        return {
            "trace": trace_name,
            "center_ghz": x0_fit_local,
            "center_mhz": x0_fit_local * 1e3,
            "gamma_ghz": gamma_fit_local,
            "fwhm_mhz": 2.0 * gamma_fit_local * 1e3,
            "noise_std": noise_std_local,
            "noise_density": float(noise_density_local),
            "max_slope_per_ghz": response_slope_local,
            "point_time_s": float(point_time_s),
            "sensitivity_t_rt_hz": float(sensitivity_t_rt_hz_local),
            "sensitivity_text": _format_sensitivity(sensitivity_t_rt_hz_local),
            "noise_window_ghz": noise_window_used_local,
        }

    # Prepare dictionary with both reference and signal analysis
    result = {
        "reference": result_dict
    }
    # Attempt to add signal, if present (and not already the chosen 'trace')
    if hasattr(data, "signal") and trace != "signal":
        result["signal"] = get_trace_result("signal")
    # If they asked for signal, still return reference too for parity
    if hasattr(data, "reference") and trace != "reference":
        result["reference"] = get_trace_result("reference")

    # Also leave root-level keys for convenience if only one trace is requested
    if len(result) == 1:
        return list(result.values())[0]
    else:
        return result


# Backwards-compatible alias: the single-Lorentzian estimator is the "legacy" method.
estimate_odmr_sensitivity_legacy = estimate_odmr_sensitivity


# =============================================================================
# Structure-aware sensitivity (handles Zeeman doublets / hyperfine).
#
# The legacy estimator above fits ONE Lorentzian to the whole resonance, so a
# split/hyperfine line is fitted as a single broad envelope -> the slope is
# roughly halved and the sensitivity comes out ~2x worse than reality. The
# functions below instead:
#   * fit a sum of Lorentzians (one per detected line) over the central
#     resonance and take the analytic max |dS/df| on the steepest flank,
#   * estimate noise on the genuinely off-resonance points two ways: the
#     successive-difference white floor and a sigma-clipped detrended baseline
#     (the clip removes broad off-axis lines the central mask misses).
# The sensitivity formula itself is unchanged.
# =============================================================================


def _multi_lorentzian(x, off, *params):
    y = np.full_like(x, off, dtype=float)
    for i in range(len(params) // 3):
        c, g, a = params[3 * i:3 * i + 3]
        y = y + a * g ** 2 / ((x - c) ** 2 + g ** 2)
    return y


def _multi_lorentzian_deriv(x, params):  # params = flat [c, g, a] * n
    d = np.zeros_like(x, dtype=float)
    for i in range(len(params) // 3):
        c, g, a = params[3 * i:3 * i + 3]
        d = d + a * g ** 2 * (-2.0 * (x - c)) / ((x - c) ** 2 + g ** 2) ** 2
    return d


def _detect_dips(freq, trace, step, prom_frac=0.06, min_sep_mhz=3.0):
    from scipy.signal import find_peaks
    y = savgol_filter(trace, min(9, len(trace) - (len(trace) + 1) % 2), 3)
    dip = -(y - np.median(y))
    span = float(dip.max() - dip.min())
    if span <= 0:
        return np.array([])
    pk, _ = find_peaks(dip, prominence=max(span * prom_frac, 1e-9),
                       distance=max(int(min_sep_mhz / step), 1))
    return freq[pk]


def _resonance_window(freq, trace, halfwidth_mhz):
    center = float(freq[int(np.argmax(np.abs(trace - np.median(trace))))])
    return center - halfwidth_mhz, center + halfwidth_mhz


def _fit_max_slope(freq, y, window, step, max_lines=6):
    """Multi-Lorentzian fit over `window`; return analytic max |dS/df| (ADC/MHz)."""
    lo, hi = window
    m = (freq >= lo) & (freq <= hi)
    xc, yc = freq[m], y[m]
    if len(xc) < 6:
        xc, yc, m = freq, y, np.ones_like(freq, bool)
    centers = _detect_dips(xc, yc, step, prom_frac=0.05)
    if len(centers) == 0:
        centers = np.array([xc[int(np.argmin(yc))]])
    centers = centers[:max_lines]
    off0 = float(np.median(np.concatenate([yc[:3], yc[-3:]])))
    p0 = [off0]
    lb = [-np.inf]
    ub = [np.inf]
    for c in centers:
        p0 += [float(c), 3.0, float(yc.min() - off0)]
        lb += [float(c) - 6.0, 0.5, -np.inf]
        ub += [float(c) + 6.0, 18.0, np.inf]
    popt = None
    try:
        popt, _ = curve_fit(_multi_lorentzian, xc, yc, p0=p0,
                            bounds=(lb, ub), maxfev=300000)
    except Exception:
        try:
            popt, _ = curve_fit(_multi_lorentzian, xc, yc, p0=p0, maxfev=300000)
        except Exception:
            popt = None

    # Model-free numeric slope (cross-check / overfit guard / fallback).
    win = min(11, len(y) if len(y) % 2 == 1 else len(y) - 1)
    win = max(win, 5)
    deriv_num = savgol_filter(y, win, 3, deriv=1, delta=step)
    numeric_slope = float(np.max(np.abs(deriv_num[m])))

    if popt is None:
        return {"slope": numeric_slope, "f_at_slope": float(freq[m][int(np.argmax(np.abs(deriv_num[m])))]),
                "n_lines": 0, "popt": None, "window": window, "numeric_slope": numeric_slope}

    xfine = np.linspace(xc.min(), xc.max(), 6000)
    der = _multi_lorentzian_deriv(xfine, popt[1:])
    k = int(np.argmax(np.abs(der)))
    fit_slope = float(np.abs(der[k]))
    # Guard against fitting a too-narrow line to a noise spike.
    if not np.isfinite(fit_slope) or fit_slope > 3.0 * max(numeric_slope, 1e-9):
        return {"slope": numeric_slope, "f_at_slope": float(freq[m][int(np.argmax(np.abs(deriv_num[m])))]),
                "n_lines": len(centers), "popt": popt, "window": window, "numeric_slope": numeric_slope}
    return {"slope": fit_slope, "f_at_slope": float(xfine[k]), "n_lines": len(centers),
            "popt": popt, "window": window, "numeric_slope": numeric_slope}


def _noise_estimates(freq, y, init_mask, k=4.0, n_iter=6):
    """White (successive-difference) noise + sigma-clipped detrended baseline."""
    valid = np.isfinite(freq) & np.isfinite(y)
    base_mask = np.asarray(init_mask, dtype=bool) & valid
    if base_mask.sum() < 2:
        base_mask = valid
    if base_mask.sum() < 2:
        return {"sdiff": np.nan, "baseline": np.nan, "n_clean": int(base_mask.sum())}

    if base_mask.sum() > 3:
        sdiff = float(np.std(np.diff(y[base_mask]), ddof=1) / np.sqrt(2))
    else:
        sdiff = float(np.std(y[base_mask], ddof=1))
    if not np.isfinite(sdiff) or sdiff <= 0:
        sdiff = float(np.std(y[base_mask], ddof=1))

    m = base_mask.copy()
    deg = min(3 if base_mask.sum() > 8 else 1, int(base_mask.sum()) - 1)
    for _ in range(n_iter):
        if m.sum() <= deg:
            break
        coef = np.polyfit(freq[m], y[m], deg)
        resid = y - np.polyval(coef, freq)
        threshold = k * max(sdiff if np.isfinite(sdiff) else 0.0, 1e-9)
        m_new = base_mask & (np.abs(resid) < threshold)
        if m_new.sum() < 15 or np.array_equal(m_new, m):
            m = m_new if m_new.sum() >= 15 else m
            break
        m = m_new

    deg = min(deg, int(m.sum()) - 1)
    if m.sum() <= deg:
        baseline = float(np.std(y[base_mask], ddof=1))
        return {"sdiff": sdiff, "baseline": baseline, "n_clean": int(base_mask.sum())}

    coef = np.polyfit(freq[m], y[m], deg)
    baseline = float(np.std((y - np.polyval(coef, freq))[m], ddof=1))
    return {"sdiff": sdiff, "baseline": baseline, "n_clean": int(m.sum())}


def estimate_sensitivity(
    data,
    *,
    traces=("signal", "reference", "contrast"),
    point_time_s=None,
    qd_module=None,
    config=None,
    gamma_hz_per_t=GYROMAGNETIC_RATIO_GHZ_PER_T * 1e9,
    resonance_window_mhz=None,
    resonance_halfwidth_mhz=30.0,
    noise_guard_mhz=18.0,
):
    """Structure-aware ODMR magnetic sensitivity for every available trace.

    Returns a dict keyed by trace name, each with the fitted slope, both noise
    estimates and sensitivities (white floor and conservative baseline), plus a
    ``best`` key naming the trace with the lowest baseline sensitivity.

    eta_B = sigma * sqrt(t_point) / (gamma * |dS/df|_max)   [T / sqrt(Hz)]
    """
    point_time_s = _get_point_time_s(point_time_s=point_time_s, qd_module=qd_module, config=config)

    x = np.asarray(data.frequencies, dtype=float)  # MHz
    order = np.argsort(x)
    x = x[order]
    step = float(np.median(np.diff(x))) if len(x) > 1 else 1.0

    available = {}
    for name in traces:
        if hasattr(data, name):
            y = np.asarray(getattr(data, name), dtype=float)[order]
            if np.isfinite(y).sum() >= 5:
                available[name] = y
    if not available:
        raise ValueError("No usable traces found on data (expected signal/reference/contrast).")

    # Common resonance window from the clearest trace.
    if resonance_window_mhz is not None:
        window = tuple(resonance_window_mhz)
    else:
        if "reference" in available:
            ref_trace = available["reference"]
        elif "contrast" in available:
            ref_trace = available["contrast"]
        else:
            ref_trace = next(iter(available.values()))
        window = _resonance_window(x, ref_trace, resonance_halfwidth_mhz)

    # Off-resonance mask: exclude +/- guard around every detected line on every trace.
    features = []
    for y in available.values():
        features.append(_detect_dips(x, y, step))
    features = np.unique(np.concatenate(features)) if features else np.array([])
    init_mask = np.ones_like(x, bool)
    for c in features:
        init_mask &= np.abs(x - c) > noise_guard_mhz
    min_noise_points = min(15, max(5, len(x) // 5))
    if init_mask.sum() < min_noise_points:
        outside_window = (x < window[0]) | (x > window[1])
        if outside_window.sum() >= min_noise_points:
            init_mask = outside_window
        else:
            init_mask = np.ones_like(x, bool)

    results = {}
    for name, y in available.items():
        fit = _fit_max_slope(x, y, window, step)
        noise = _noise_estimates(x, y, init_mask)
        slope_per_hz = fit["slope"] / 1e6  # ADC per Hz
        denom = gamma_hz_per_t * slope_per_hz
        eta_white = (noise["sdiff"] * np.sqrt(point_time_s) / denom) if denom > 0 else np.nan
        eta_base = (noise["baseline"] * np.sqrt(point_time_s) / denom) if denom > 0 else np.nan
        results[name] = {
            "trace": name,
            "slope_adc_per_mhz": fit["slope"],
            "slope_numeric_adc_per_mhz": fit["numeric_slope"],
            "f_at_slope_mhz": fit["f_at_slope"],
            "n_lines": fit["n_lines"],
            "noise_white": noise["sdiff"],
            "noise_baseline": noise["baseline"],
            "n_clean_points": noise["n_clean"],
            "point_time_s": float(point_time_s),
            "sensitivity_t_rt_hz_white": float(eta_white),
            "sensitivity_t_rt_hz": float(eta_base),  # baseline = conservative headline
            "sensitivity_text": _format_sensitivity(eta_base),
            "resonance_window_mhz": tuple(float(w) for w in window),
        }

    best = min(results, key=lambda n: results[n]["sensitivity_t_rt_hz"])
    results["best"] = best
    results["point_time_s"] = float(point_time_s)
    return results


def quick_sensitivity_legacy(
    data,
    *,
    trace="reference",
    noise_window_ghz=(2.70, 2.80),
    point_time_s=None,
    qd_module=None,
    config=None,
):
    result = estimate_odmr_sensitivity(
        data,
        trace=trace,
        noise_window_ghz=noise_window_ghz,
        point_time_s=point_time_s,
        qd_module=qd_module,
        config=config,
    )
    # Print information for both reference and signal if both are present
    if isinstance(result, dict) and "reference" in result and "signal" in result and result["signal"] is not None:
        for key in ["reference", "signal"]:
            print(f"Sensitivity ({key}): {result[key]['sensitivity_text']}")
            print(
                f"trace={result[key]['trace']}, center={result[key]['center_mhz']:.2f} MHz, "
                f"FWHM={result[key]['fwhm_mhz']:.2f} MHz"
            )
    else:
        # Single result (dict or structure)
        print(f"Sensitivity: {result['sensitivity_text']}")
        print(
            f"trace={result['trace']}, center={result['center_mhz']:.2f} MHz, "
            f"FWHM={result['fwhm_mhz']:.2f} MHz"
        )
    return result


def quick_sensitivity(
    data,
    *,
    point_time_s=None,
    qd_module=None,
    config=None,
    show_legacy=True,
    **legacy_kwargs,
):
    """Structure-aware ODMR sensitivity for every trace (signal / reference / contrast).

    Drop-in replacement for the old single-Lorentzian ``quick_sensitivity``: call
    ``quick_sensitivity(d)`` after an ODMR sweep. Handles Zeeman-split / hyperfine
    resonances that the legacy estimator mis-fit as one broad line. Reports each
    trace as a range (white-noise floor -> conservative detrended baseline) and
    highlights the best trace. Pass ``show_legacy=False`` to hide the old number.

    Returns the dict from :func:`estimate_sensitivity`.
    """
    res = estimate_sensitivity(
        data, point_time_s=point_time_s, qd_module=qd_module, config=config
    )
    pt_us = res["point_time_s"] * 1e6
    best = res["best"]

    print(f"ODMR sensitivity (structure-aware)  |  t_point = {pt_us:.0f} us")
    for name in ("signal", "reference", "contrast"):
        if name not in res:
            continue
        r = res[name]
        eta_lo = r["sensitivity_t_rt_hz_white"] * 1e9
        eta_hi = r["sensitivity_t_rt_hz"] * 1e9
        star = "  <-- best" if name == best else ""
        print(
            f"  {name:18s} slope={r['slope_adc_per_mhz']:6.2f} ADC/MHz @ {r['f_at_slope_mhz']:7.1f} MHz "
            f"| noise w/b={r['noise_white']:.3f}/{r['noise_baseline']:.3f} "
            f"| eta = {eta_lo:5.1f} - {eta_hi:5.1f} nT/sqrt(Hz){star}"
        )
    rb = res[best]
    print(
        f"  BEST: {best}  eta = {rb['sensitivity_t_rt_hz_white']*1e9:.1f} - "
        f"{rb['sensitivity_t_rt_hz']*1e9:.1f} nT/sqrt(Hz) "
        f"(operate at {rb['f_at_slope_mhz']:.1f} MHz)"
    )

    if show_legacy:
        try:
            leg = estimate_odmr_sensitivity_legacy(data, trace="reference",
                                                   point_time_s=point_time_s,
                                                   qd_module=qd_module, config=config)
            leg = leg["reference"] if isinstance(leg, dict) and "reference" in leg else leg
            print(f"  (legacy single-Lorentzian, reference: {leg['sensitivity_text']})")
        except Exception:
            pass
    return res
