#!/usr/bin/env python
"""
Email-compatible April 29 ODMR reconstruction using the cloned official repo.

This file is intentionally a thin local wrapper around
``NV_Magnetometry_Software-main/nv_toolkit``.  It follows the phase1 workflow
from the cloned repo rather than the earlier standalone fallback:

1. read the raw full-scan CSV exactly as exported,
2. detect transition centers from the raw MW-on trace,
3. fit the normalized MW-on/MW-off spectrum from a bias seed, and
4. optionally write a JSON summary and fit plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent
REPO_CANDIDATES = (
    ROOT / "NV_Magnetometry_Software-main",
    ROOT / "NV_magnetometry_software-main",
    ROOT / "NV_Magnetometry_Software",
)


def _find_official_repo() -> Path:
    for candidate in REPO_CANDIDATES:
        if (candidate / "nv_toolkit").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    searched = ", ".join(str(path.name) for path in REPO_CANDIDATES)
    raise FileNotFoundError(f"Could not find cloned NV magnetometry repo. Searched: {searched}")


OFFICIAL_REPO = _find_official_repo()
sys.path.insert(0, str(OFFICIAL_REPO))

import numpy as np  # noqa: E402
import nv_toolkit as nv  # noqa: E402


DEFAULT_BIAS_SEED_MT = np.array([0.0, 0.0, -6.0], dtype=float)


def _parse_vector(text: str) -> np.ndarray:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected a 3-vector like '0,0,-6'.")
    return np.asarray(values, dtype=float)


def _load_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No rows found in {path}")

    columns: dict[str, np.ndarray] = {}
    for field in rows[0]:
        if field:
            columns[field] = np.asarray([float(row[field]) for row in rows], dtype=float)
    return columns


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def detect_transition_centers(freqs_mhz: np.ndarray, mw_on_adc: np.ndarray, max_dips: int) -> np.ndarray:
    """Use the same raw-MW-on dip detector as the repo's phase1 April 29 script."""

    return np.asarray(
        nv.find_baseline_corrected_dip_positions(
            freqs_mhz,
            mw_on_adc,
            baseline_window_mhz=101.0,
            smooth_window_mhz=9.0,
            min_prominence_sigma=3.0,
            min_width_mhz=2.0,
            max_width_mhz=20.0,
            min_distance_mhz=8.0,
            max_dips=max_dips,
        ),
        dtype=float,
    )


def fit_normalized_spectrum(
    freqs_mhz: np.ndarray,
    normalized_intensity: np.ndarray,
    *,
    bias_seed_mT: np.ndarray,
    axis_model: str,
    local_field_bounds_mT: float,
) -> tuple[Any, np.ndarray]:
    result = nv.fit_spectrum_parameters(
        freqs_mhz,
        normalized_intensity,
        B_base_mT=np.asarray(bias_seed_mT, dtype=float),
        nv_axes_preset=axis_model,
        linewidth_mode="per_axis",
        contrast_mode="per_axis",
        hyperfine_mode="fixed",
        initial_hyperfine_splitting_mhz=2.16,
        local_field_bounds_mT=local_field_bounds_mT,
    )
    b_total_mT = np.asarray(result.field_estimate["B_total_mT"], dtype=float)
    predicted = (
        nv.odmr_spectrum(
            freqs_mhz,
            b_total_mT,
            linewidths=result.linewidths_mhz,
            contrasts=result.contrasts,
            hyperfine_splitting_mhz=result.hyperfine_splitting_mhz,
            nv_axes=nv.named_nv_axes(axis_model),
        )
        + float(result.baseline_offset)
    )
    return result, predicted


def analyze_csv(
    csv_path: Path,
    *,
    bias_seed_mT: np.ndarray,
    axis_model: str,
    max_dips: int,
    local_field_bounds_mT: float,
) -> dict[str, Any]:
    columns = _load_columns(csv_path)
    required = {
        "frequency_MHz",
        "photoluminescence_mw_on_ADC",
        "photoluminescence_mw_off_ADC",
    }
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required column(s): {', '.join(missing)}")

    freqs = columns["frequency_MHz"]
    mw_on = columns["photoluminescence_mw_on_ADC"]
    mw_off = columns["photoluminescence_mw_off_ADC"]
    normalized = mw_on / mw_off

    transition_centers = detect_transition_centers(freqs, mw_on, max_dips=max_dips)
    fit_result, predicted = fit_normalized_spectrum(
        freqs,
        normalized,
        bias_seed_mT=bias_seed_mT,
        axis_model=axis_model,
        local_field_bounds_mT=local_field_bounds_mT,
    )
    b_total_mT = np.asarray(fit_result.field_estimate["B_total_mT"], dtype=float)
    residual = normalized - predicted

    return {
        "source_csv": csv_path,
        "official_repo": OFFICIAL_REPO,
        "assumptions": {
            "workflow": "phase1_april29_raw_mw_on_centers_and_normalized_spectrum_fit",
            "axis_model": axis_model,
            "bias_seed_mT": bias_seed_mT,
            "hyperfine_mode": "fixed",
            "hyperfine_splitting_mhz": fit_result.hyperfine_splitting_mhz,
            "local_field_bounds_mT": local_field_bounds_mT,
        },
        "frequency_mhz": freqs,
        "normalized_intensity": normalized,
        "predicted_normalized_intensity": predicted,
        "residual": residual,
        "detected_transition_centers_mhz": transition_centers,
        "fit": {
            "success": bool(fit_result.success),
            "message": str(fit_result.message),
            "cost": float(fit_result.cost),
            "aic": float(fit_result.aic),
            "B_total_mT": b_total_mT,
            "B_total_norm_mT": float(np.linalg.norm(b_total_mT)),
            "B_local_mT": np.asarray(fit_result.field_estimate["B_local_mT"], dtype=float),
            "B_base_mT": np.asarray(fit_result.field_estimate["B_base_mT"], dtype=float),
            "linewidths_mhz": np.asarray(fit_result.linewidths_mhz, dtype=float),
            "contrasts": np.asarray(fit_result.contrasts, dtype=float),
            "baseline_offset": float(fit_result.baseline_offset),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "max_abs_residual": float(np.max(np.abs(residual))),
        },
    }


def write_json(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compact = {
        key: value
        for key, value in result.items()
        if key not in {"frequency_mhz", "normalized_intensity", "predicted_normalized_intensity", "residual"}
    }
    path.write_text(json.dumps(_json_safe(compact), indent=2) + "\n", encoding="utf-8")


def write_plot(path: Path, result: dict[str, Any]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(OFFICIAL_REPO / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freqs = np.asarray(result["frequency_mhz"], dtype=float)
    measured = np.asarray(result["normalized_intensity"], dtype=float)
    predicted = np.asarray(result["predicted_normalized_intensity"], dtype=float)
    centers = np.asarray(result["detected_transition_centers_mhz"], dtype=float)
    residual = np.asarray(result["residual"], dtype=float)
    b_total = np.asarray(result["fit"]["B_total_mT"], dtype=float)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axes[0].plot(freqs, measured, color="#1f5b70", linewidth=1.1, label="MW-on / MW-off")
    axes[0].plot(freqs, predicted, color="#b23a2e", linewidth=1.2, label="best fit")
    for center in centers:
        axes[0].axvline(float(center), color="#8d6e63", linestyle="--", linewidth=0.8, alpha=0.65)
    axes[0].set_ylabel("Normalized intensity")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[0].set_title(
        f"{Path(result['source_csv']).name}: "
        f"B=[{b_total[0]:.4f}, {b_total[1]:.4f}, {b_total[2]:.4f}] mT"
    )

    axes[1].plot(freqs, residual, color="#333333", linewidth=1.0)
    axes[1].axhline(0.0, color="#888888", linewidth=0.8)
    axes[1].set_xlabel("Microwave frequency (MHz)")
    axes[1].set_ylabel("Residual")
    axes[1].grid(alpha=0.25)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _format_centers(centers: np.ndarray) -> str:
    return ", ".join(f"{float(center):.3f}" for center in centers)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mag",
        description="Run the official April 29 NV ODMR reconstruction workflow on a raw full-scan CSV.",
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--axis-model", choices=["qdm", "canonical", "111"], default="qdm")
    parser.add_argument("--bias-seed", type=_parse_vector, default=DEFAULT_BIAS_SEED_MT)
    parser.add_argument("--max-dips", type=int, default=8)
    parser.add_argument("--local-field-bounds-mt", type=float, default=8.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = analyze_csv(
        args.csv_path,
        bias_seed_mT=np.asarray(args.bias_seed, dtype=float),
        axis_model=args.axis_model,
        max_dips=args.max_dips,
        local_field_bounds_mT=float(args.local_field_bounds_mt),
    )

    centers = np.asarray(result["detected_transition_centers_mhz"], dtype=float)
    b_total = np.asarray(result["fit"]["B_total_mT"], dtype=float)
    print(f"Transition centers: {_format_centers(centers)} MHz")
    print(f"B_total = [{b_total[0]: .4f}, {b_total[1]: .4f}, {b_total[2]: .4f}] mT")
    print(
        f"|B| = {float(result['fit']['B_total_norm_mT']):.4f} mT; "
        f"RMSE = {float(result['fit']['rmse']):.3e}"
    )

    if args.output_json is not None:
        write_json(args.output_json, result)
        print(f"Wrote {args.output_json}")
    if args.plot is not None:
        write_plot(args.plot, result)
        print(f"Wrote {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
