from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from ast import Add, BinOp, Call, Constant, Div, Expression, Load, Mult, Name, NodeVisitor, Pow, Sub, UAdd, USub, UnaryOp, parse
from pathlib import Path

import numpy as np

from .fitting import fit_spectrum_parameters, fit_total_field_from_peak_positions
from .io import load_spectrum_csv
from .model import D0, odmr_spectrum
from .peaks import find_baseline_corrected_dip_positions, find_refined_dip_positions


STAGE_X_AXES = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0 / 3.0, 2.0 * math.sqrt(2.0) / 3.0, 0.0],
        [-1.0 / 3.0, -math.sqrt(2.0) / 3.0, math.sqrt(2.0 / 3.0)],
        [-1.0 / 3.0, -math.sqrt(2.0) / 3.0, -math.sqrt(2.0 / 3.0)],
    ],
    dtype=float,
)

_AXIS_PRESET_NAMES = ["canonical", "qdm", "stage-x"]


def _axes_for_preset(preset: str) -> np.ndarray:
    """Return the NV axis matrix for a named preset."""
    norm = str(preset).strip().lower()
    if norm in {"canonical", "111", "default"}:
        from .model import NV_AXES
        return np.asarray(NV_AXES, dtype=float).copy()
    if norm == "qdm":
        from .model import QDM_NV_AXES
        return np.asarray(QDM_NV_AXES, dtype=float).copy()
    if norm == "stage-x":
        return np.asarray(STAGE_X_AXES, dtype=float).copy()
    raise ValueError(f"Unsupported axis preset: {preset}")

class _ScalarExpressionEvaluator(NodeVisitor):
    def visit_Expression(self, node: Expression) -> float:
        return float(self.visit(node.body))

    def visit_BinOp(self, node: BinOp) -> float:
        left = float(self.visit(node.left))
        right = float(self.visit(node.right))
        if isinstance(node.op, Add):
            return left + right
        if isinstance(node.op, Sub):
            return left - right
        if isinstance(node.op, Mult):
            return left * right
        if isinstance(node.op, Div):
            return left / right
        if isinstance(node.op, Pow):
            return left**right
        raise ValueError("Unsupported operator in scalar expression")

    def visit_UnaryOp(self, node: UnaryOp) -> float:
        operand = float(self.visit(node.operand))
        if isinstance(node.op, UAdd):
            return operand
        if isinstance(node.op, USub):
            return -operand
        raise ValueError("Unsupported unary operator in scalar expression")

    def visit_Call(self, node: Call) -> float:
        if not isinstance(node.func, Name) or node.func.id != "sqrt" or len(node.args) != 1:
            raise ValueError("Only sqrt(...) calls are supported in custom axis expressions")
        return math.sqrt(float(self.visit(node.args[0])))

    def visit_Name(self, node: Name) -> float:
        if node.id == "pi":
            return math.pi
        raise ValueError(f"Unsupported name in scalar expression: {node.id}")

    def visit_Constant(self, node: Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numeric constants are supported in scalar expressions")

    def generic_visit(self, node: object) -> float:
        raise ValueError("Unsupported syntax in scalar expression")


def _safe_eval_scalar(expr: str) -> float:
    tree = parse(expr.strip(), mode="eval")
    return float(_ScalarExpressionEvaluator().visit(tree))


def _parse_inline_nv_axes(text: str) -> np.ndarray:
    rows = [row.strip() for row in text.split(";") if row.strip()]
    matrix = [[_safe_eval_scalar(value) for value in row.split(",")] for row in rows]
    axes = np.asarray(matrix, dtype=float)
    if axes.shape != (4, 3):
        raise ValueError(f"Custom NV axes must have shape (4, 3), got {axes.shape}")
    return axes


def _parse_vector(text: str) -> np.ndarray:
    values = [_safe_eval_scalar(part) for part in text.split(",")]
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got shape {vector.shape}")
    return vector


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run peak-first ODMR reconstruction from a raw full-scan or toolkit-style spectrum CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  mag phase1/data/raw_scan_exports_20260429_lockin/8_peaks_8_stacks.csv\n"
            "  mag phase1/data/raw_scan_exports_20260429_lockin/8_peaks_8_stacks.csv "
            "--axis-model stage-x --bias-seed 0,0,-4.2\n"
            "  mag phase1/data/raw_scan_exports_20260429_lockin/full_scan_8_peaks_8_stacks_toolkit.csv "
            "--input-format toolkit --skip-spectrum-fit"
        ),
    )
    parser.add_argument("spectrum_csv", type=Path)
    parser.add_argument("--input-format", choices=["auto", "raw", "toolkit"], default="auto")
    parser.add_argument("--fit-signal", choices=["auto", "mw-on", "normalized"], default="auto")
    parser.add_argument("--frequency-column", default="frequency_MHz")
    parser.add_argument("--signal-column", default="photoluminescence_mw_on_ADC")
    parser.add_argument("--reference-column", default="photoluminescence_mw_off_ADC")
    parser.add_argument("--axis-model", choices=["auto", "canonical", "111", "qdm", "stage-x"], default="qdm")
    parser.add_argument(
        "--nv-axes-inline",
        default=None,
        help="Semicolon-separated 4x3 axis matrix. Expressions like sqrt(2)/3 are supported.",
    )
    parser.add_argument(
        "--bias-seed",
        default=None,
        help="Optional base-field seed as 'Bx,By,Bz' in the unit given by --bias-unit.",
    )
    parser.add_argument("--bias-unit", choices=["mT", "uT"], default="mT")
    parser.add_argument("--max-dips", type=int, default=8)
    parser.add_argument("--min-dips", type=int, default=2)
    parser.add_argument("--min-distance-mhz", type=float, default=8.0)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-spectrum-candidates", type=int, default=4)
    parser.add_argument("--hyperfine-mode", choices=["auto", "fixed", "fit", "none"], default="fixed")
    parser.add_argument("--initial-hyperfine-splitting-mhz", type=float, default=2.16)
    parser.add_argument("--local-field-bounds-mt", type=float, default=1.5)
    parser.add_argument("--skip-spectrum-fit", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    return parser


def _read_numeric_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return {
        field: np.asarray([float(row[field]) for row in rows], dtype=float)
        for field in rows[0]
        if field
    }


def _resolve_axes(args: argparse.Namespace) -> tuple[str, np.ndarray | None]:
    if args.nv_axes_inline:
        return "qdm", _parse_inline_nv_axes(args.nv_axes_inline)
    if args.axis_model == "auto":
        return "auto", None
    if args.axis_model == "stage-x":
        return "qdm", np.asarray(STAGE_X_AXES, dtype=float)
    return args.axis_model, None


def _load_reconstruction_input(args: argparse.Namespace) -> dict[str, object]:
    path = args.spectrum_csv
    if args.input_format == "toolkit":
        batch = load_spectrum_csv(path)
        return {
            "input_format": "toolkit",
            "freqs_mhz": np.asarray(batch.freqs_mhz, dtype=float),
            "measured": np.asarray(batch.spectra[0], dtype=float),
            "peak_trace": np.asarray(batch.spectra[0], dtype=float),
            "peak_mode": "normalized",
            "fit_signal": "normalized",
        }

    if args.input_format == "raw":
        columns = _read_numeric_columns(path)
    else:
        columns = _read_numeric_columns(path)
        if "normalized_intensity" in columns and "frequency_mhz" in columns:
            batch = load_spectrum_csv(path)
            return {
                "input_format": "toolkit",
                "freqs_mhz": np.asarray(batch.freqs_mhz, dtype=float),
                "measured": np.asarray(batch.spectra[0], dtype=float),
                "peak_trace": np.asarray(batch.spectra[0], dtype=float),
                "peak_mode": "normalized",
                "fit_signal": "normalized",
            }

    if args.frequency_column not in columns:
        raise ValueError(f"Missing frequency column '{args.frequency_column}' in {path}")
    if args.signal_column not in columns:
        raise ValueError(f"Missing signal column '{args.signal_column}' in {path}")

    freqs = np.asarray(columns[args.frequency_column], dtype=float)
    signal = np.asarray(columns[args.signal_column], dtype=float)
    if args.fit_signal == "auto":
        fit_signal = "mw-on"
    else:
        fit_signal = args.fit_signal

    if fit_signal == "normalized":
        if args.reference_column not in columns:
            raise ValueError(f"Missing reference column '{args.reference_column}' required for normalized fitting")
        reference = np.asarray(columns[args.reference_column], dtype=float)
        measured = signal / reference
        peak_trace = measured
        peak_mode = "normalized"
    else:
        measured = signal / float(np.median(signal))
        peak_trace = signal
        peak_mode = "baseline_corrected_raw_mw_on"

    return {
        "input_format": "raw",
        "freqs_mhz": freqs,
        "measured": measured,
        "peak_trace": peak_trace,
        "peak_mode": peak_mode,
        "fit_signal": fit_signal,
    }


def _detect_transition_centers(
    freqs_mhz: np.ndarray,
    measured: np.ndarray,
    peak_trace: np.ndarray,
    peak_mode: str,
    *,
    max_dips: int,
    min_dips: int,
    min_distance_mhz: float,
) -> np.ndarray:
    if peak_mode == "baseline_corrected_raw_mw_on":
        dips = find_baseline_corrected_dip_positions(
            freqs_mhz,
            peak_trace,
            min_distance_mhz=min_distance_mhz,
            max_dips=max_dips,
        )
    else:
        dips = find_refined_dip_positions(
            freqs_mhz,
            measured,
            max_dips=max_dips,
            min_dips=min_dips,
            min_distance_mhz=min_distance_mhz,
        )

    dips = np.sort(np.asarray(dips, dtype=float))
    if len(dips) >= 3 and len(dips) % 2 != 0:
        center_index = int(np.argmin(np.abs(dips - D0)))
        dips = np.delete(dips, center_index)
    return dips


def _serialize(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_serialize(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _fit_candidate_spectra(
    freqs_mhz: np.ndarray,
    measured: np.ndarray,
    peak_fit: dict[str, object],
    *,
    preset: str,
    nv_axes: np.ndarray | None,
    bias_seed_mT: np.ndarray | None,
    max_candidates: int,
    hyperfine_mode: str,
    initial_hyperfine_splitting_mhz: float,
    local_field_bounds_mT: float,
) -> list[dict[str, object]]:
    seeds: list[tuple[str, np.ndarray]] = []
    if bias_seed_mT is not None:
        seeds.append(("user_bias_seed", np.asarray(bias_seed_mT, dtype=float)))
    for index, candidate in enumerate(peak_fit.get("candidates", [])[:max_candidates]):
        seeds.append((f"peak_candidate_{index}", np.asarray(candidate["B_total_mT"], dtype=float)))

    fitted: list[dict[str, object]] = []
    seen: set[tuple[float, float, float]] = set()
    for label, seed in seeds:
        key = tuple(np.round(seed, decimals=9))
        if key in seen:
            continue
        seen.add(key)
        result = fit_spectrum_parameters(
            freqs_mhz,
            measured,
            B_base_mT=seed,
            nv_axes_preset=preset,
            nv_axes=nv_axes,
            linewidth_mode="per_axis",
            contrast_mode="per_axis",
            hyperfine_mode=hyperfine_mode,
            initial_hyperfine_splitting_mhz=initial_hyperfine_splitting_mhz,
            local_field_bounds_mT=local_field_bounds_mT,
        )
        b_total = np.asarray(result.field_estimate["B_total_mT"], dtype=float)
        predicted = odmr_spectrum(
            freqs_mhz,
            b_total,
            linewidths=result.linewidths_mhz,
            contrasts=result.contrasts,
            hyperfine_splitting_mhz=result.hyperfine_splitting_mhz,
            nv_axes=nv_axes,
        ) + float(result.baseline_offset)
        residual = np.asarray(measured, dtype=float) - predicted
        fitted.append(
            {
                "seed_label": label,
                "B_base_mT": np.asarray(seed, dtype=float),
                "success": bool(result.success),
                "message": str(result.message),
                "cost": float(result.cost),
                "aic": float(result.aic),
                "B_total_mT": b_total,
                "B_local_mT": np.asarray(result.field_estimate["B_local_mT"], dtype=float),
                "B_total_norm_mT": float(np.linalg.norm(b_total)),
                "linewidths_mhz": np.asarray(result.linewidths_mhz, dtype=float),
                "contrasts": np.asarray(result.contrasts, dtype=float),
                "baseline_offset": float(result.baseline_offset),
                "hyperfine_mode": str(result.hyperfine_mode),
                "hyperfine_splitting_mhz": result.hyperfine_splitting_mhz,
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "max_abs_residual": float(np.max(np.abs(residual))),
                "predicted": predicted,
            }
        )
    return sorted(fitted, key=lambda item: (float(item["aic"]), float(item["cost"])))


def _write_plot(path: Path, freqs_mhz: np.ndarray, measured: np.ndarray, predicted: np.ndarray, dips: np.ndarray, title: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    residual = np.asarray(measured, dtype=float) - np.asarray(predicted, dtype=float)
    axes[0].plot(freqs_mhz, measured, color="#1f5b70", linewidth=1.1, label="measured")
    axes[0].plot(freqs_mhz, predicted, color="#b23a2e", linewidth=1.2, label="best fit")
    for dip in np.asarray(dips, dtype=float):
        axes[0].axvline(float(dip), color="#8d6e63", linestyle="--", linewidth=0.8, alpha=0.65)
    axes[0].set_ylabel("Signal")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[0].set_title(title)
    axes[1].plot(freqs_mhz, residual, color="#333333", linewidth=1.0)
    axes[1].axhline(0.0, color="#888888", linewidth=0.8)
    axes[1].set_xlabel("Microwave frequency (MHz)")
    axes[1].set_ylabel("Residual")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _run_auto_reconstruct(args: argparse.Namespace) -> None:
    """Option 3: try all axis models, pick best by AIC — no setup knowledge needed."""
    bias_seed_mT = None if args.bias_seed is None else _parse_vector(args.bias_seed) * (1e-3 if args.bias_unit == "uT" else 1.0)
    loaded = _load_reconstruction_input(args)
    freqs_mhz = np.asarray(loaded["freqs_mhz"], dtype=float)
    measured = np.asarray(loaded["measured"], dtype=float)
    peak_trace = np.asarray(loaded["peak_trace"], dtype=float)
    dips = _detect_transition_centers(
        freqs_mhz, measured, peak_trace, str(loaded["peak_mode"]),
        max_dips=args.max_dips, min_dips=args.min_dips,
        min_distance_mhz=args.min_distance_mhz,
    )
    if len(dips) < 2:
        raise ValueError(f"Need at least 2 detected transition centers, got {len(dips)}")

    print(f"Detected {len(dips)} transition centers: {', '.join(f'{float(dip):.3f}' for dip in dips)}")
    print(f"--- Auto mode: trying axis models: {', '.join(_AXIS_PRESET_NAMES)} ---\n")

    all_trials: list[dict[str, object]] = []
    for model_name in _AXIS_PRESET_NAMES:
        axes = _axes_for_preset(model_name)
        peak_fit = fit_total_field_from_peak_positions(
            dips, nv_axes_preset=model_name, nv_axes=axes,
            max_candidates=args.max_candidates,
        )
        spectrum_fits: list[dict[str, object]] = []
        if not args.skip_spectrum_fit:
            spectrum_fits = _fit_candidate_spectra(
                freqs_mhz, measured, peak_fit,
                preset=model_name, nv_axes=axes,
                bias_seed_mT=bias_seed_mT,
                max_candidates=args.max_spectrum_candidates,
                hyperfine_mode=args.hyperfine_mode,
                initial_hyperfine_splitting_mhz=args.initial_hyperfine_splitting_mhz,
                local_field_bounds_mT=args.local_field_bounds_mt,
            )

        best = spectrum_fits[0] if spectrum_fits else None
        aic = float(best["aic"]) if best is not None else float("inf")
        rmse = float(best["rmse"]) if best is not None else float("inf")
        b_norm = float(best["B_total_norm_mT"]) if best is not None else None
        b_vector = np.asarray(best["B_total_mT"], dtype=float) if best is not None else None

        tag = " ★ BEST" if not all_trials or aic < min(float(t.get("aic", float("inf"))) for t in all_trials) else ""
        print(
            f"  {model_name:>10s}: aic={aic:8.3f}  rmse={rmse:.3e}"
            + (f"  |B|={b_norm:.4f} mT" if b_norm is not None else "  (no fit)")
            + tag
        )

        all_trials.append({
            "axis_model": model_name,
            "nv_axes_xyz": axes,
            "aic": aic,
            "rmse": rmse,
            "peak_inversion": peak_fit,
            "spectrum_fits": [{k: v for k, v in fit.items() if k != "predicted"} for fit in spectrum_fits],
            "best_fit": {k: v for k, v in best.items() if k != "predicted"} if best is not None else None,
            "predicted": np.asarray(best["predicted"], dtype=float) if best is not None else None,
        })

    # Pick winner by AIC
    winner = min(all_trials, key=lambda t: float(t.get("aic", float("inf"))))
    winner_model = str(winner["axis_model"])
    winner_best = winner["best_fit"]
    winner_predicted = np.asarray(winner["predicted"], dtype=float) if winner["predicted"] is not None else None

    print(f"\n→ Selected axis model: {winner_model} (lowest AIC)")

    if winner_best is not None:
        bv = np.asarray(winner_best["B_total_mT"], dtype=float)
        print(
            f"  B_total_mT = ({bv[0]:.4f}, {bv[1]:.4f}, {bv[2]:.4f})  "
            f"rmse={float(winner_best['rmse']):.3e}  aic={float(winner_best['aic']):.3f}"
        )
        if args.plot is not None and winner_predicted is not None:
            _write_plot(
                args.plot, freqs_mhz, measured, winner_predicted, dips,
                f"{args.spectrum_csv.name}: auto→{winner_model} | "
                f"|B|={float(winner_best['B_total_norm_mT']):.4f} mT, "
                f"RMSE={float(winner_best['rmse']):.2e}"
            )

    if args.output_json is not None:
        summary = {
            "source_csv": args.spectrum_csv,
            "mode": "auto",
            "assumptions": {
                "input_format": loaded["input_format"],
                "fit_signal": loaded["fit_signal"],
                "peak_mode": loaded["peak_mode"],
                "axis_model": "auto (enumerated)",
                "bias_seed_mT": bias_seed_mT,
                "hyperfine_mode": args.hyperfine_mode,
                "initial_hyperfine_splitting_mhz": args.initial_hyperfine_splitting_mhz,
                "local_field_bounds_mT": args.local_field_bounds_mt,
            },
            "detected_transition_centers_mhz": dips,
            "trials": [
                {k: v for k, v in t.items() if k not in ("predicted", "nv_axes_xyz")}
                for t in all_trials
            ],
            "selected_model": winner_model,
            "selected_axes_xyz": winner["nv_axes_xyz"],
            "best_fit": winner_best,
            "plot": str(args.plot) if args.plot is not None else None,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(_serialize(summary), handle, indent=2)
        print(f"Wrote {args.output_json}")


def _run_reconstruct_cli(args: argparse.Namespace) -> None:
    preset, nv_axes = _resolve_axes(args)
    if preset == "auto":
        _run_auto_reconstruct(args)
        return
    bias_seed_mT = None if args.bias_seed is None else _parse_vector(args.bias_seed) * (1e-3 if args.bias_unit == "uT" else 1.0)
    loaded = _load_reconstruction_input(args)
    freqs_mhz = np.asarray(loaded["freqs_mhz"], dtype=float)
    measured = np.asarray(loaded["measured"], dtype=float)
    peak_trace = np.asarray(loaded["peak_trace"], dtype=float)
    dips = _detect_transition_centers(
        freqs_mhz,
        measured,
        peak_trace,
        str(loaded["peak_mode"]),
        max_dips=args.max_dips,
        min_dips=args.min_dips,
        min_distance_mhz=args.min_distance_mhz,
    )
    if len(dips) < 2:
        raise ValueError(f"Need at least 2 detected transition centers, got {len(dips)}")

    peak_fit = fit_total_field_from_peak_positions(
        dips,
        nv_axes_preset=preset,
        nv_axes=nv_axes,
        max_candidates=args.max_candidates,
    )

    spectrum_fits: list[dict[str, object]] = []
    if not args.skip_spectrum_fit:
        spectrum_fits = _fit_candidate_spectra(
            freqs_mhz,
            measured,
            peak_fit,
            preset=preset,
            nv_axes=nv_axes,
            bias_seed_mT=bias_seed_mT,
            max_candidates=args.max_spectrum_candidates,
            hyperfine_mode=args.hyperfine_mode,
            initial_hyperfine_splitting_mhz=args.initial_hyperfine_splitting_mhz,
            local_field_bounds_mT=args.local_field_bounds_mt,
        )

    summary = {
        "source_csv": args.spectrum_csv,
        "assumptions": {
            "input_format": loaded["input_format"],
            "fit_signal": loaded["fit_signal"],
            "peak_mode": loaded["peak_mode"],
            "axis_model": args.axis_model,
            "nv_axes_source": "custom" if nv_axes is not None else preset,
            "nv_axes_xyz": np.asarray(nv_axes, dtype=float) if nv_axes is not None else None,
            "bias_seed_mT": bias_seed_mT,
            "hyperfine_mode": args.hyperfine_mode,
            "initial_hyperfine_splitting_mhz": args.initial_hyperfine_splitting_mhz,
            "local_field_bounds_mT": args.local_field_bounds_mt,
        },
        "detected_transition_centers_mhz": dips,
        "peak_inversion": peak_fit,
        "spectrum_fits": [{key: value for key, value in fit.items() if key != "predicted"} for fit in spectrum_fits],
    }

    best_fit = spectrum_fits[0] if spectrum_fits else None
    if best_fit is not None:
        summary["best_fit"] = {key: value for key, value in best_fit.items() if key != "predicted"}
        if args.plot is not None:
            _write_plot(
                args.plot,
                freqs_mhz,
                measured,
                np.asarray(best_fit["predicted"], dtype=float),
                dips,
                (
                    f"{args.spectrum_csv.name}: {args.axis_model} | "
                    f"|B|={float(best_fit['B_total_norm_mT']):.4f} mT, RMSE={float(best_fit['rmse']):.2e}"
                ),
            )
            summary["plot"] = args.plot

    print(f"Detected {len(dips)} transition centers: {', '.join(f'{float(dip):.3f}' for dip in dips)}")
    if peak_fit.get("candidates"):
        best_peak = peak_fit["candidates"][0]
        best_vector = np.asarray(best_peak["B_total_mT"], dtype=float)
        print(
            "Best peak-only candidate: "
            f"B_total_mT=({best_vector[0]:.4f}, {best_vector[1]:.4f}, {best_vector[2]:.4f}) "
            f"score={float(best_peak['score']):.4g}"
        )
    else:
        print("Peak inversion returned partial-pair information only.")
    if best_fit is not None:
        best_vector = np.asarray(best_fit["B_total_mT"], dtype=float)
        print(
            "Best spectrum fit: "
            f"B_total_mT=({best_vector[0]:.4f}, {best_vector[1]:.4f}, {best_vector[2]:.4f}) "
            f"rmse={float(best_fit['rmse']):.3e} aic={float(best_fit['aic']):.3f} "
            f"seed={best_fit['seed_label']}"
        )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(_serialize(summary), handle, indent=2)
        print(f"Wrote {args.output_json}")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(argv)
    _run_reconstruct_cli(args)


if __name__ == "__main__":
    main()
