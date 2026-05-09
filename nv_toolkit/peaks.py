from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths, savgol_filter


@dataclass(frozen=True)
class DipFitResult:
    center_mhz: float
    linewidth_mhz: float
    contrast: float
    baseline: float
    model: str
    eta: float | None
    cost: float
    success: bool
    n_points: int


@dataclass(frozen=True)
class DetectedDipCandidate:
    center_mhz: float
    prominence: float
    width_mhz: float
    baseline: float
    depth: float
    noise_sigma: float
    snr: float
    confidence: float
    fit_cost: float
    fit_success: bool
    fit_linewidth_mhz: float
    fit_contrast: float


def smooth_spectrum(
    spectrum: np.ndarray,
    *,
    window_length: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    values = np.asarray(spectrum, dtype=float)
    if len(values) < max(window_length, polyorder + 2):
        return values.copy()
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(values, window_length=window_length, polyorder=polyorder, mode="interp")


def find_dip_positions(
    freqs_mhz: np.ndarray,
    spectrum: np.ndarray,
    *,
    prominence: float = 0.002,
    min_distance_mhz: float = 0.5,
    max_dips: int | None = None,
    smooth: bool = True,
    hyperfine_splitting_mhz: float | None = None,
) -> np.ndarray:
    """Find dip positions in a normalised ODMR spectrum.

    If ``hyperfine_splitting_mhz`` is provided, the search cap is raised to 24
    to accommodate resolved doublets, and the returned positions are transition
    *centers* after collapsing each hyperfine doublet.  Unmatched singlets are
    passed through unchanged, so the output is always in transition-center space
    regardless of whether hyperfine is resolved.

    Use this parameter consistently with ``odmr_spectrum``'s
    ``hyperfine_splitting_mhz`` argument: if you collapse here, the downstream
    cost function must also use the same splitting to generate doublets.
    """
    effective_max = max_dips
    if hyperfine_splitting_mhz is not None and (max_dips is None or max_dips < 24):
        effective_max = 24

    freqs = np.asarray(freqs_mhz, dtype=float)
    values = np.asarray(spectrum, dtype=float)
    working = smooth_spectrum(values) if smooth else values
    spacing_mhz = float(np.median(np.diff(freqs))) if len(freqs) > 1 else 1.0
    peaks, props = find_peaks(
        1.0 - working,
        prominence=float(prominence),
        distance=max(1, int(round(min_distance_mhz / spacing_mhz))),
    )
    if effective_max is not None and len(peaks) > effective_max:
        prominences = np.asarray(props.get("prominences", np.zeros(len(peaks))), dtype=float)
        keep = np.argsort(prominences)[-int(effective_max):]
        peaks = np.sort(peaks[keep])
    result = freqs[peaks]
    if hyperfine_splitting_mhz is not None:
        result = collapse_hyperfine_doublets(result, splitting_mhz=float(hyperfine_splitting_mhz))
    return result


def find_refined_dip_positions(
    freqs_mhz: np.ndarray,
    spectrum: np.ndarray,
    *,
    prominences: tuple[float, ...] = (0.001, 0.003, 0.005, 0.01, 0.02),
    min_distance_mhz: float = 0.5,
    max_dips: int = 10,
    min_dips: int = 4,
) -> np.ndarray:
    """Find strong ODMR dip positions with local quadratic refinement."""
    freqs = np.asarray(freqs_mhz, dtype=float)
    values = np.asarray(spectrum, dtype=float)
    if freqs.shape != values.shape:
        raise ValueError(f"freqs shape {freqs.shape} does not match spectrum shape {values.shape}")

    inverted = 1.0 - values
    spacing_mhz = float(np.median(np.diff(freqs))) if len(freqs) > 1 else 1.0
    chosen_peaks: np.ndarray | None = None
    for prominence in prominences:
        peaks, props = find_peaks(
            inverted,
            prominence=float(prominence),
            distance=max(1, int(round(float(min_distance_mhz) / spacing_mhz))),
        )
        if len(peaks) == 0:
            continue

        prominences_found = np.asarray(props.get("prominences", np.zeros(len(peaks))), dtype=float)
        if len(peaks) > int(max_dips):
            keep = np.argsort(prominences_found)[-int(max_dips):]
            peaks = np.sort(peaks[keep])

        chosen_peaks = peaks
        if len(peaks) >= int(min_dips):
            break

    if chosen_peaks is None or len(chosen_peaks) == 0:
        return np.asarray([], dtype=float)

    dip_freqs: list[float] = []
    for peak_idx in chosen_peaks:
        lo = max(0, int(peak_idx) - 2)
        hi = min(len(freqs), int(peak_idx) + 3)
        f_local = freqs[lo:hi]
        s_local = inverted[lo:hi]
        if len(f_local) >= 3:
            coeffs = np.polyfit(f_local, s_local, 2)
            if coeffs[0] != 0.0:
                f_peak = float(-coeffs[1] / (2 * coeffs[0]))
                if float(f_local[0]) <= f_peak <= float(f_local[-1]):
                    dip_freqs.append(f_peak)
                    continue
        dip_freqs.append(float(freqs[int(peak_idx)]))

    return np.sort(np.asarray(dip_freqs, dtype=float))


def find_baseline_corrected_dip_positions(
    freqs_mhz: np.ndarray,
    spectrum: np.ndarray,
    *,
    baseline_window_mhz: float = 101.0,
    smooth_window_mhz: float = 9.0,
    min_prominence_sigma: float = 3.0,
    min_prominence: float | None = None,
    min_width_mhz: float = 2.0,
    max_width_mhz: float = 20.0,
    min_distance_mhz: float = 8.0,
    max_dips: int | None = 8,
) -> np.ndarray:
    """Find negative-going dips after removing slow baseline structure.

    This is intended for raw ADC-like ODMR traces where absolute intensity drift
    can be larger than weak ODMR contrast.  The detector works on
    ``baseline - spectrum`` and filters candidates by prominence and FWHM before
    doing a local quadratic center refinement.
    """
    candidates = detect_baseline_corrected_dip_candidates(
        freqs_mhz,
        spectrum,
        baseline_window_mhz=baseline_window_mhz,
        smooth_window_mhz=smooth_window_mhz,
        min_prominence_sigma=min_prominence_sigma,
        min_prominence=min_prominence,
        min_width_mhz=min_width_mhz,
        max_width_mhz=max_width_mhz,
        min_distance_mhz=min_distance_mhz,
        max_dips=max_dips,
    )
    return np.sort(np.asarray([candidate.center_mhz for candidate in candidates], dtype=float))


def _baseline_corrected_peak_data(
    freqs_mhz: np.ndarray,
    spectrum: np.ndarray,
    *,
    baseline_window_mhz: float,
    smooth_window_mhz: float,
    min_prominence_sigma: float,
    min_prominence: float | None,
    min_width_mhz: float,
    max_width_mhz: float,
    min_distance_mhz: float,
    max_dips: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    freqs = np.asarray(freqs_mhz, dtype=float)
    values = np.asarray(spectrum, dtype=float)
    if freqs.shape != values.shape:
        raise ValueError(f"freqs shape {freqs.shape} does not match spectrum shape {values.shape}")
    if len(freqs) < 7:
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            0.0,
        )

    spacing_mhz = float(np.median(np.diff(freqs))) if len(freqs) > 1 else 1.0

    def odd_window(width_mhz: float, minimum: int) -> int:
        points = max(minimum, int(round(float(width_mhz) / spacing_mhz)))
        if points % 2 == 0:
            points += 1
        return min(points, len(freqs) if len(freqs) % 2 == 1 else len(freqs) - 1)

    baseline_window = odd_window(baseline_window_mhz, 7)
    smooth_window = odd_window(smooth_window_mhz, 5)
    baseline = savgol_filter(values, window_length=baseline_window, polyorder=2, mode="interp")
    residual_dip_signal = baseline - values
    if smooth_window >= 5:
        residual_dip_signal = savgol_filter(residual_dip_signal, window_length=smooth_window, polyorder=2, mode="interp")

    noise_sigma = 1.4826 * float(np.median(np.abs(residual_dip_signal - np.median(residual_dip_signal))))
    prominence_threshold = float(min_prominence) if min_prominence is not None else float(min_prominence_sigma) * noise_sigma
    if prominence_threshold <= 0.0:
        prominence_threshold = float(np.finfo(float).eps)

    peaks, props = find_peaks(
        residual_dip_signal,
        prominence=prominence_threshold,
        distance=max(1, int(round(float(min_distance_mhz) / spacing_mhz))),
    )
    if len(peaks) == 0:
        return peaks, np.asarray([], dtype=float), residual_dip_signal, baseline, noise_sigma

    widths_mhz = peak_widths(residual_dip_signal, peaks, rel_height=0.5)[0] * spacing_mhz
    prominences = np.asarray(props.get("prominences", np.zeros(len(peaks))), dtype=float)
    keep_mask = (widths_mhz >= float(min_width_mhz)) & (widths_mhz <= float(max_width_mhz))
    peaks = peaks[keep_mask]
    widths_mhz = widths_mhz[keep_mask]
    prominences = prominences[keep_mask]
    if len(peaks) == 0:
        return peaks, widths_mhz, residual_dip_signal, baseline, noise_sigma

    if max_dips is not None and len(peaks) > int(max_dips):
        keep = np.argsort(prominences)[-int(max_dips):]
        peaks = np.sort(peaks[keep])
        widths_mhz = widths_mhz[keep]

    return peaks, widths_mhz, residual_dip_signal, baseline, noise_sigma


def detect_baseline_corrected_dip_candidates(
    freqs_mhz: np.ndarray,
    spectrum: np.ndarray,
    *,
    baseline_window_mhz: float = 101.0,
    smooth_window_mhz: float = 9.0,
    min_prominence_sigma: float = 3.0,
    min_prominence: float | None = None,
    min_width_mhz: float = 2.0,
    max_width_mhz: float = 20.0,
    min_distance_mhz: float = 8.0,
    max_dips: int | None = 8,
    local_fit_window_mhz: float = 8.0,
) -> list[DetectedDipCandidate]:
    freqs = np.asarray(freqs_mhz, dtype=float)
    values = np.asarray(spectrum, dtype=float)
    peaks, widths_mhz, residual_dip_signal, baseline, noise_sigma = _baseline_corrected_peak_data(
        freqs,
        values,
        baseline_window_mhz=baseline_window_mhz,
        smooth_window_mhz=smooth_window_mhz,
        min_prominence_sigma=min_prominence_sigma,
        min_prominence=min_prominence,
        min_width_mhz=min_width_mhz,
        max_width_mhz=max_width_mhz,
        min_distance_mhz=min_distance_mhz,
        max_dips=max_dips,
    )
    if len(peaks) == 0:
        return []

    dip_freqs: list[float] = []
    for peak_idx in peaks:
        lo = max(0, int(peak_idx) - 2)
        hi = min(len(freqs), int(peak_idx) + 3)
        f_local = freqs[lo:hi]
        s_local = residual_dip_signal[lo:hi]
        if len(f_local) >= 3:
            coeffs = np.polyfit(f_local, s_local, 2)
            if coeffs[0] < 0.0:
                f_peak = float(-coeffs[1] / (2.0 * coeffs[0]))
                if float(f_local[0]) <= f_peak <= float(f_local[-1]):
                    dip_freqs.append(f_peak)
                    continue
        dip_freqs.append(float(freqs[int(peak_idx)]))

    local_fits = fit_local_dips(
        freqs,
        values,
        np.asarray(dip_freqs, dtype=float),
        window_mhz=local_fit_window_mhz,
        model="pseudo_voigt",
    )
    fit_by_center = {round(fit.center_mhz, 6): fit for fit in local_fits}

    candidates: list[DetectedDipCandidate] = []
    for dip_freq, peak_idx, width_mhz in zip(dip_freqs, peaks, widths_mhz):
        center_key = min(fit_by_center, key=lambda key: abs(key - round(dip_freq, 6))) if fit_by_center else None
        fit = fit_by_center.get(center_key) if center_key is not None else None
        baseline_level = float(np.interp(dip_freq, freqs, baseline))
        raw_level = float(np.interp(dip_freq, freqs, values))
        depth = max(baseline_level - raw_level, 0.0)
        snr = depth / max(noise_sigma, float(np.finfo(float).eps))
        width_score = float(np.exp(-abs(float(width_mhz) - 4.0) / 10.0))
        fit_score = 0.0
        fit_cost = float("inf")
        fit_success = False
        fit_linewidth_mhz = float(width_mhz)
        fit_contrast = 0.0
        if fit is not None:
            fit_cost = float(fit.cost)
            fit_success = bool(fit.success)
            fit_linewidth_mhz = float(fit.linewidth_mhz)
            fit_contrast = float(fit.contrast)
            fit_score = float(np.exp(-fit.cost * max(fit.n_points, 1) * 1e3))
        confidence = float(np.clip(0.55 * min(snr / 8.0, 1.0) + 0.20 * width_score + 0.25 * fit_score, 0.0, 1.0))
        candidates.append(
            DetectedDipCandidate(
                center_mhz=float(dip_freq),
                prominence=float(residual_dip_signal[int(peak_idx)]),
                width_mhz=float(width_mhz),
                baseline=baseline_level,
                depth=depth,
                noise_sigma=float(noise_sigma),
                snr=float(snr),
                confidence=confidence,
                fit_cost=fit_cost,
                fit_success=fit_success,
                fit_linewidth_mhz=fit_linewidth_mhz,
                fit_contrast=fit_contrast,
            )
        )
    return sorted(candidates, key=lambda item: item.center_mhz)


def cluster_dip_candidates(
    candidates: list[DetectedDipCandidate],
    *,
    merge_distance_mhz: float = 4.0,
) -> list[list[DetectedDipCandidate]]:
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda item: item.center_mhz)
    clusters: list[list[DetectedDipCandidate]] = [[sorted_candidates[0]]]
    for candidate in sorted_candidates[1:]:
        previous = clusters[-1][-1]
        if float(candidate.center_mhz - previous.center_mhz) < float(merge_distance_mhz):
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])
    return clusters


def collapse_candidate_clusters(
    clusters: list[list[DetectedDipCandidate]],
    *,
    max_centers: int | None = 8,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    cluster_rows: list[dict[str, object]] = []
    for cluster_index, cluster in enumerate(clusters):
        weights = np.asarray(
            [max(candidate.prominence, candidate.depth) * max(candidate.confidence, 1e-6) for candidate in cluster],
            dtype=float,
        )
        if np.sum(weights) <= 0.0:
            weights = np.ones(len(cluster), dtype=float)
        centers = np.asarray([candidate.center_mhz for candidate in cluster], dtype=float)
        collapsed_center = float(np.average(centers, weights=weights))
        cluster_rows.append(
            {
                "cluster_index": cluster_index,
                "collapsed_center_mhz": collapsed_center,
                "n_members": int(len(cluster)),
                "mean_confidence": float(np.mean([candidate.confidence for candidate in cluster])),
                "score": float(np.sum(weights)),
                "members": [
                    {
                        "center_mhz": float(candidate.center_mhz),
                        "snr": float(candidate.snr),
                        "confidence": float(candidate.confidence),
                        "width_mhz": float(candidate.width_mhz),
                        "fit_success": bool(candidate.fit_success),
                    }
                    for candidate in cluster
                ],
            }
        )

    if max_centers is not None and len(cluster_rows) > int(max_centers):
        keep_indices = {
            row["cluster_index"]
            for row in sorted(cluster_rows, key=lambda row: float(row["score"]), reverse=True)[: int(max_centers)]
        }
        cluster_rows = [row for row in cluster_rows if row["cluster_index"] in keep_indices]

    cluster_rows = sorted(cluster_rows, key=lambda row: float(row["collapsed_center_mhz"]))
    centers = np.asarray([float(row["collapsed_center_mhz"]) for row in cluster_rows], dtype=float)
    return centers, cluster_rows


def _estimate_local_fwhm_mhz(
    local_freqs: np.ndarray,
    local_values: np.ndarray,
    window_mhz: float = 8.0,
) -> float:
    """Estimate FWHM of the dominant dip in a local spectral segment.

    Uses scipy.signal.peak_widths on the inverted trace.  Falls back to
    ``min(window_mhz / 4, 3.0)`` when the estimate would be degenerate.
    """
    if len(local_freqs) < 5:
        return min(window_mhz / 4.0, 3.0)
    spacing_mhz = float(np.median(np.diff(local_freqs)))
    if spacing_mhz <= 0.0:
        return min(window_mhz / 4.0, 3.0)
    inverted = -np.asarray(local_values, dtype=float)
    inverted -= inverted.min()
    if inverted.max() == 0.0:
        return min(window_mhz / 4.0, 3.0)
    inverted /= inverted.max()
    peak_idx = int(np.argmax(inverted))
    try:
        widths_samples, _, _, _ = peak_widths(inverted, [peak_idx], rel_height=0.5)
        fwhm_mhz = float(widths_samples[0]) * spacing_mhz
    except Exception:
        return min(window_mhz / 4.0, 3.0)
    if fwhm_mhz < 0.2 or fwhm_mhz >= window_mhz * 2.0:
        return min(window_mhz / 4.0, 3.0)
    return fwhm_mhz


def fit_local_dips(
    freqs_mhz: np.ndarray,
    spectrum: np.ndarray,
    dip_centers_mhz: np.ndarray,
    *,
    window_mhz: float = 8.0,
    model: str = "lorentzian",
    hyperfine_merge_mhz: float = 3.0,
) -> list[DipFitResult]:
    """Fit local dip profile parameters around supplied center guesses.

    Supported models are ``lorentzian``, ``gaussian``, and ``pseudo_voigt``.
    ``linewidth_mhz`` is FWHM for all three models.  For ``pseudo_voigt``,
    ``eta`` is the Lorentzian mixing fraction.

    Candidates separated by less than ``hyperfine_merge_mhz`` are collapsed
    into a single midpoint before fitting — they are almost certainly hyperfine
    lines of the same NV transition (~2.16 MHz splitting) rather than distinct
    transitions.  Set to 0 to disable.
    """
    model_name = str(model).strip().lower().replace("-", "_")
    if model_name not in {"lorentzian", "gaussian", "pseudo_voigt"}:
        raise ValueError(f"Unsupported dip model: {model}")

    freqs = np.asarray(freqs_mhz, dtype=float)
    values = np.asarray(spectrum, dtype=float)
    raw_centers = np.sort(np.asarray(dip_centers_mhz, dtype=float))

    # Collapse pairs within hyperfine_merge_mhz into a single midpoint.
    merged: list[float] = []
    skip_next = False
    for i, c in enumerate(raw_centers):
        if skip_next:
            skip_next = False
            continue
        if i + 1 < len(raw_centers) and (raw_centers[i + 1] - c) < float(hyperfine_merge_mhz):
            merged.append(float(0.5 * (c + raw_centers[i + 1])))
            skip_next = True
        else:
            merged.append(float(c))
    centers = np.asarray(merged, dtype=float)
    if freqs.shape != values.shape:
        raise ValueError(f"freqs shape {freqs.shape} does not match spectrum shape {values.shape}")

    # Per-center windows: shrink when a neighbour is closer than the default
    # window to avoid contaminating fits with the adjacent peak's data.
    if len(centers) > 1:
        center_windows = []
        for i, c in enumerate(centers):
            gaps = [abs(c - centers[j]) for j in range(len(centers)) if j != i]
            min_gap = min(gaps)
            center_windows.append(min(float(window_mhz), 0.9 * min_gap / 2.0))
    else:
        center_windows = [float(window_mhz)]

    fits: list[DipFitResult] = []
    for center_guess, win in zip(centers, center_windows):
        mask = np.abs(freqs - float(center_guess)) <= win
        local_freqs = freqs[mask]
        local_values = values[mask]
        if len(local_freqs) < 5:
            continue

        baseline0 = float(np.percentile(local_values, 90.0))
        nearest_idx = int(np.argmin(np.abs(local_freqs - float(center_guess))))
        center0 = float(center_guess)
        contrast0 = max(float(baseline0 - local_values[nearest_idx]), 1e-5)
        linewidth0 = _estimate_local_fwhm_mhz(local_freqs, local_values, win)

        lower = [center_guess - win, 0.2, 0.0, float(np.min(local_values) - 0.05)]
        upper = [center_guess + win, 29.0, 0.2, float(np.max(local_values) + 0.05)]
        x0 = [center0, linewidth0, contrast0, baseline0]
        if model_name == "pseudo_voigt":
            x0.append(0.5)
            lower.append(0.0)
            upper.append(1.0)

        if model_name == "lorentzian":
            def _curve_func(f: np.ndarray, center: float, linewidth: float, contrast: float, baseline: float) -> np.ndarray:
                hw = linewidth / 2.0
                return baseline - contrast * hw**2 / ((f - center) ** 2 + hw**2)
        elif model_name == "gaussian":
            def _curve_func(f: np.ndarray, center: float, linewidth: float, contrast: float, baseline: float) -> np.ndarray:
                return baseline - contrast * np.exp(-4.0 * np.log(2.0) * ((f - center) / linewidth) ** 2)
        else:
            def _curve_func(f: np.ndarray, center: float, linewidth: float, contrast: float, baseline: float, eta: float) -> np.ndarray:
                hw = linewidth / 2.0
                lorentz = hw**2 / ((f - center) ** 2 + hw**2)
                gaussian = np.exp(-4.0 * np.log(2.0) * ((f - center) / linewidth) ** 2)
                return baseline - contrast * (eta * lorentz + (1.0 - eta) * gaussian)

        success = False
        cost_val = float("inf")
        center = center0
        linewidth = linewidth0
        contrast = contrast0
        baseline = baseline0
        eta: float | None = None
        try:
            popt, _ = curve_fit(
                _curve_func,
                local_freqs,
                local_values,
                p0=x0,
                bounds=(lower, upper),
                method="trf",
                maxfev=50000,
            )
            center, linewidth, contrast, baseline = float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3])
            eta = float(popt[4]) if model_name == "pseudo_voigt" else None
            residuals = local_values - _curve_func(local_freqs, *popt)
            cost_val = float(np.sum(residuals**2))
            rms = float(np.sqrt(np.mean(residuals**2)))
            peak_to_peak = float(np.max(local_values) - np.min(local_values))
            center_ok = abs(center - float(center_guess)) <= win
            width_ok = linewidth > 0.2 and linewidth < 29.0
            residual_ok = peak_to_peak == 0.0 or rms <= 0.5 * peak_to_peak
            success = center_ok and width_ok and residual_ok
        except Exception:
            pass

        fits.append(
            DipFitResult(
                center_mhz=float(center),
                linewidth_mhz=float(linewidth),
                contrast=float(contrast),
                baseline=float(baseline),
                model=model_name,
                eta=eta,
                cost=cost_val,
                success=success,
                n_points=int(len(local_freqs)),
            )
        )

    return fits


def fit_lorentzian_dips(
    freqs_mhz: np.ndarray,
    spectrum: np.ndarray,
    dip_centers_mhz: np.ndarray,
    *,
    window_mhz: float = 8.0,
) -> list[DipFitResult]:
    return fit_local_dips(
        freqs_mhz,
        spectrum,
        dip_centers_mhz,
        window_mhz=window_mhz,
        model="lorentzian",
    )


def reject_unpaired_peaks(
    fits: list[DipFitResult],
    sweep_range_mhz: tuple[float, float],
    *,
    d0_mhz: float | None = None,
    pair_tolerance_mhz: float = 20.0,
) -> tuple[list[DipFitResult], list[DipFitResult]]:
    """Separate fitted peaks into physically paired (kept) and spurious (rejected).

    Two rejection criteria applied in order:

    1. **Mirror outside sweep range**: if 2*D0 - center falls outside the measured
       frequency window, the peak cannot have a visible partner → rejected.
    2. **No close partner**: if the mirror is in range but no detected peak lies
       within pair_tolerance_mhz, the peak is rejected.

    D0 is estimated from the inner pairs (excluding the outermost pair) to avoid
    bias from per-axis strain shifts (typically ±3–6 MHz in real samples).
    If fewer than 4 peaks are present, falls back to 2870.0 MHz.

    Returns (kept, rejected).
    """
    if len(fits) == 0:
        return [], []

    sorted_fits = sorted(fits, key=lambda f: f.center_mhz)
    cs = np.array([f.center_mhz for f in sorted_fits])
    n = len(cs)
    f_lo, f_hi = float(sweep_range_mhz[0]), float(sweep_range_mhz[1])

    if d0_mhz is None:
        # Estimate D0 from inner pairs only (skip outermost pair on each side)
        inner_midpoints = [
            0.5 * (cs[i] + cs[n - 1 - i])
            for i in range(1, n // 2)  # skip i=0 (outermost)
        ]
        d0_mhz = float(np.mean(inner_midpoints)) if inner_midpoints else 2870.0

    kept: list[DipFitResult] = []
    rejected: list[DipFitResult] = []

    for fit in sorted_fits:
        mirror = 2.0 * d0_mhz - fit.center_mhz
        if not (f_lo <= mirror <= f_hi):
            rejected.append(fit)
        elif float(np.min(np.abs(cs - mirror))) > pair_tolerance_mhz:
            rejected.append(fit)
        else:
            kept.append(fit)

    return kept, rejected


def collapse_hyperfine_doublets(
    dip_positions_mhz: np.ndarray,
    *,
    splitting_mhz: float,
    tolerance_mhz: float = 0.75,
) -> np.ndarray:
    dips = np.sort(np.asarray(dip_positions_mhz, dtype=float))
    centers: list[float] = []
    used: set[int] = set()
    for i in range(len(dips) - 1):
        if i in used or (i + 1) in used:
            continue
        separation = float(dips[i + 1] - dips[i])
        if abs(separation - float(splitting_mhz)) <= float(tolerance_mhz):
            used.add(i)
            used.add(i + 1)
            centers.append(0.5 * float(dips[i] + dips[i + 1]))
    for idx, dip in enumerate(dips):
        if idx not in used:
            centers.append(float(dip))
    return np.sort(np.asarray(centers, dtype=float))
