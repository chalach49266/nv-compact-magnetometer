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


def quick_sensitivity(
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
