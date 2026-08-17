from __future__ import annotations

import argparse
import csv
import json
import math
import os
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import curses
except ModuleNotFoundError as exc:
    if exc.name not in {"curses", "_curses"}:
        raise
    curses = None  # type: ignore[assignment]
    _CURSES_IMPORT_ERROR = exc
else:
    _CURSES_IMPORT_ERROR = None

import numpy as np

from .base_field import load_base_field_csv
from .cli import _build_parser as _build_reconstruct_parser
from .cli import _detect_transition_centers, _load_reconstruction_input, _parse_vector, _read_numeric_columns
from .fitting import rank_spectrum_field_candidates
from .intensity_tracking import build_blockwise_calibrations, estimate_parked_series_fields, reconstruct_vector_timeseries
from .io import ParkedFrequencySeries, load_spectrum_csv
from .peaks import fit_local_dips
from .peaks import cluster_dip_candidates, collapse_candidate_clusters, detect_baseline_corrected_dip_candidates, reject_unpaired_peaks


REPO_DIR = Path(__file__).resolve().parents[1]
TIME_COLUMN_CANDIDATES = ("timestamp_epoch_s", "time_s", "time", "timestamp")
FREQUENCY_COLUMN_CANDIDATES = ("frequency_mhz", "peak_frequency_mhz", "frequency", "freq")
INTENSITY_COLUMN_CANDIDATES = ("normalized_intensity", "intensity", "signal", "norm_intensity")
SAVE_ID_COLUMN_CANDIDATES = ("save_id", "scan_id", "run_id", "sample_id")
RAW_FREQUENCY_COLUMN = "frequency_MHz"
RAW_MW_ON_COLUMN = "photoluminescence_mw_on_ADC"


def _require_curses() -> Any:
    if curses is None:
        raise RuntimeError(
            "Interactive curses UI is not available in this Python environment. "
            "On Windows, install the 'windows-curses' package or run operator mode with --popup."
        ) from _CURSES_IMPORT_ERROR
    return curses


@dataclass(frozen=True)
class WorkflowStage:
    title: str
    status: str
    summary: str
    details: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowDemo:
    key: str
    title: str
    subtitle: str
    stages: tuple[WorkflowStage, ...]
    source_paths: tuple[str, ...]


@dataclass(frozen=True)
class ParkedFrequencyPlanEntry:
    transition_index: int
    center_mhz: float
    linewidth_mhz: float
    frequency_minus_mhz: float
    frequency_plus_mhz: float
    slope_minus_per_mhz: float
    slope_plus_per_mhz: float


@dataclass(frozen=True)
class OperatorConfig:
    full_scan_path: Path
    parked_data_path: Path
    parked_format: str
    bias_field_mT: np.ndarray
    reference_freqs_mhz: np.ndarray
    reference_spectrum: np.ndarray
    transition_centers_mhz: np.ndarray
    parked_plan: tuple[ParkedFrequencyPlanEntry, ...]
    plan_csv_path: Path
    wide_template_path: Path
    long_template_path: Path
    poll_interval_s: float


@dataclass(frozen=True)
class LiveMonitorSnapshot:
    status: str
    message: str
    sample_count: int
    ok_count: int
    latest_timestamp_s: float | None
    latest_local_uT: tuple[float, float, float] | None
    latest_total_uT: tuple[float, float, float] | None
    latest_residual_uT: float | None
    latest_rank: int | None
    latest_channels: int | None
    vector_rows: tuple[dict[str, object], ...]
    projection_rows: tuple[dict[str, object], ...]


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _pick_column(fieldnames: Iterable[str], candidates: Sequence[str]) -> str | None:
    normalized = {"".join(ch for ch in name.strip().lower() if ch.isalnum()): name for name in fieldnames}
    for candidate in candidates:
        resolved = normalized.get("".join(ch for ch in candidate.strip().lower() if ch.isalnum()))
        if resolved is not None:
            return resolved
    return None


def _safe_float(row: dict[str, object], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _format_optional_float(value: object, *, precision: int = 2, unit: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    suffix = f" {unit}" if unit else ""
    return f"{number:.{precision}f}{suffix}"


def _format_vector(values: Sequence[float], *, precision: int = 2, unit: str = "uT") -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + f"] {unit}"


def _summarize_vector_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    parsed_rows = list(rows)
    if not parsed_rows:
        return {
            "count": 0,
            "ok_count": 0,
            "first_vector_uT": None,
            "last_vector_uT": None,
            "residual_median_uT": float("nan"),
            "time_span_s": 0.0,
        }

    ok_rows = [row for row in parsed_rows if str(row.get("status", "")) == "ok"]
    residuals = sorted(
        value
        for value in (_safe_float(row, "residual_norm_uT") for row in ok_rows)
        if math.isfinite(value)
    )
    median_residual = float("nan")
    if residuals:
        median_residual = residuals[len(residuals) // 2]

    timestamps = [_safe_float(row, "timestamp_epoch_s") for row in parsed_rows]
    finite_timestamps = [value for value in timestamps if math.isfinite(value)]
    time_span_s = max(finite_timestamps) - min(finite_timestamps) if finite_timestamps else 0.0

    def _vector(row: dict[str, object] | None) -> tuple[float, float, float] | None:
        if row is None:
            return None
        return (
            _safe_float(row, "delta_Bx_uT"),
            _safe_float(row, "delta_By_uT"),
            _safe_float(row, "delta_Bz_uT"),
        )

    first_ok = ok_rows[0] if ok_rows else None
    last_ok = ok_rows[-1] if ok_rows else None
    return {
        "count": len(parsed_rows),
        "ok_count": len(ok_rows),
        "first_vector_uT": _vector(first_ok),
        "last_vector_uT": _vector(last_ok),
        "residual_median_uT": median_residual,
        "time_span_s": time_span_s,
    }


def _build_april17_real_demo() -> WorkflowDemo:
    summary_path = REPO_DIR / "phase1" / "results" / "real_export_validation_summary.json"
    projection_csv = REPO_DIR / "phase1" / "results" / "real_parked_frequency_projection_rows.csv"
    vector_csv = REPO_DIR / "phase1" / "results" / "real_parked_frequency_vector_rows.csv"

    summary = _load_json(summary_path)
    vector_rows = _load_csv_rows(vector_csv)
    projection_rows = _load_csv_rows(projection_csv)
    vector_summary = _summarize_vector_rows(vector_rows)

    base_field_mT = summary["base_field_mT"]
    full_scan_local_uT = summary["full_scan_local_uT"]
    linewidths = summary["calibrated_linewidths_mhz"]
    contrasts = summary["calibrated_contrasts"]

    stages = (
        WorkflowStage(
            title="1. Reference ODMR",
            status="ready",
            summary="Committed April 17 full-scan calibration is available.",
            details=(
                f"Base field: {_format_vector(base_field_mT, precision=4, unit='mT')}",
                f"Recovered local field: {_format_vector(full_scan_local_uT, precision=2)}",
                f"Fit success: {bool(summary['full_scan_success'])}",
                f"Fit cost: {float(summary['full_scan_cost']):.3e}",
            ),
        ),
        WorkflowStage(
            title="2. Parked-Frequency Plan",
            status="ready",
            summary="The parked-frequency export has eight independent two-point blocks.",
            details=(
                f"Projection rows: {len(projection_rows)}",
                f"Vector rows expected: {int(summary['n_vector_rows'])}",
                "Two-point blocks per timestamp: 8",
                f"Mean parked linewidth: {float(summary['parked_calibrated_linewidth_mhz']):.3f} MHz",
                f"Mean parked contrast: {float(summary['parked_calibrated_contrast']):.5f}",
            ),
        ),
        WorkflowStage(
            title="3. Lock-In Projection Solve",
            status="ready",
            summary="Each parked block resolves a projected field channel from the intensity pair.",
            details=(
                f"Calibrated linewidth examples: {float(linewidths[0][0]):.3f}, {float(linewidths[1][0]):.3f} MHz",
                f"Calibrated contrast examples: {float(contrasts[0][0]):.5f}, {float(contrasts[1][0]):.5f}",
                f"Projection output rows: {len(projection_rows)}",
                f"Projection CSV: {projection_csv.relative_to(REPO_DIR)}",
            ),
        ),
        WorkflowStage(
            title="4. Vector Time Series",
            status="ready",
            summary="The committed result already demonstrates a full-rank 3D time series.",
            details=(
                f"Vector rows: {vector_summary['count']}",
                f"Rank-ok rows: {vector_summary['ok_count']}",
                f"Median residual: {float(summary['vector_residual_uT_median']):.2f} uT",
                f"First vector: {_format_vector(vector_summary['first_vector_uT'] or (0.0, 0.0, 0.0))}",
                f"Last vector: {_format_vector(vector_summary['last_vector_uT'] or (0.0, 0.0, 0.0))}",
            ),
        ),
        WorkflowStage(
            title="5. Downstream Local B",
            status="ready",
            summary="This dataset is the cleanest demonstration target for the sensing-protocol TUI.",
            details=(
                f"delta_Bx range: {float(summary['delta_Bx_uT_min']):.2f} to {float(summary['delta_Bx_uT_max']):.2f} uT",
                f"delta_By range: {float(summary['delta_By_uT_min']):.2f} to {float(summary['delta_By_uT_max']):.2f} uT",
                f"delta_Bz range: {float(summary['delta_Bz_uT_min']):.2f} to {float(summary['delta_Bz_uT_max']):.2f} uT",
                f"Vector CSV: {vector_csv.relative_to(REPO_DIR)}",
                f"Summary JSON: {summary_path.relative_to(REPO_DIR)}",
            ),
        ),
    )
    return WorkflowDemo(
        key="april17-real",
        title="April 17 Real Parked Validation",
        subtitle="Best end-to-end demonstration of the committed sensing workflow.",
        stages=stages,
        source_paths=(
            str(summary_path.relative_to(REPO_DIR)),
            str(projection_csv.relative_to(REPO_DIR)),
            str(vector_csv.relative_to(REPO_DIR)),
        ),
    )


def _build_april29_lockin_demo() -> WorkflowDemo:
    summary_path = REPO_DIR / "phase1" / "results" / "20260429_lockin_fit_summary.json"
    summary = _load_json(summary_path)
    full_scan_fits = summary["full_scan_fits"]
    lockin_notes = summary["lockin_row_notes"]
    lockin_fit = summary["lockin_row_fit"]
    projection_rows = lockin_fit["projection_rows"]
    vector_rows = lockin_fit["vector_rows"]

    old_fit = full_scan_fits["8_peaks_8_stacks"]
    new_fit = full_scan_fits["8_peaks_8_stacks_new_tdy"]
    only_vector = vector_rows[0]

    stages = (
        WorkflowStage(
            title="1. Reference ODMR",
            status="mixed",
            summary="Two April 29 full scans are committed and can seed parked-frequency choices.",
            details=(
                f"Legacy scan magnitude: {_format_vector(old_fit['B_total_uT'], precision=1)}",
                f"New scan magnitude: {_format_vector(new_fit['B_total_uT'], precision=1)}",
                f"Legacy trust level: {old_fit['trust_level']}",
                f"New trust level: {new_fit['trust_level']}",
            ),
        ),
        WorkflowStage(
            title="2. Parked-Frequency Plan",
            status="limited",
            summary="The handoff captures only one lock-in row and two parked blocks.",
            details=(
                f"Raw lower peak input: {float(lockin_notes['raw_lower_peak_input_mhz']):.3f} MHz",
                f"Raw upper peak input: {float(lockin_notes['raw_upper_peak_input_mhz']):.3f} MHz",
                f"Two-point blocks: {int(lockin_notes['n_two_point_blocks'])}",
                str(lockin_notes["limitation"]),
            ),
        ),
        WorkflowStage(
            title="3. Lock-In Projection Solve",
            status="limited",
            summary="The existing export supports projections, but not a durable time series.",
            details=(
                f"Projection rows generated: {len(projection_rows)}",
                f"Reference full scan: {Path(lockin_fit['reference_full_scan']).relative_to(REPO_DIR)}",
                f"Parked CSV: {Path(lockin_fit['parked_csv']).relative_to(REPO_DIR)}",
            ),
        ),
        WorkflowStage(
            title="4. Vector Attempt",
            status="limited",
            summary="The current solver emits a single rank-deficient vector row from the two-block export.",
            details=(
                f"Vector rows generated: {len(vector_rows)}",
                f"Status: {only_vector['status']}",
                f"Rank: {int(only_vector['rank'])}",
                f"Channels: {int(only_vector['n_channels'])}",
                "Recovered delta_B: unavailable because the solve is underdetermined.",
                f"Residual norm: {_format_optional_float(only_vector['residual_norm_uT'], precision=2, unit='uT')}",
            ),
        ),
        WorkflowStage(
            title="5. Recommended Next Acquisition",
            status="next",
            summary="This is the right dataset for demonstrating the limitation and motivating a richer export contract.",
            details=(
                "Target at least 3 independent parked blocks per timestamp.",
                "Prefer 4 to 8 blocks for stable least-squares inversion.",
                "Keep the full-scan reference CSV and parked CSV as separate durable outputs.",
                f"Summary JSON: {summary_path.relative_to(REPO_DIR)}",
            ),
        ),
    )
    return WorkflowDemo(
        key="april29-lockin",
        title="April 29 Lock-In Handoff",
        subtitle="Useful for showing the workflow boundary: projection-capable, time-series-limited.",
        stages=stages,
        source_paths=(str(summary_path.relative_to(REPO_DIR)),),
    )


def _build_synthetic_demo() -> WorkflowDemo:
    summary_path = REPO_DIR / "phase1" / "results" / "synthetic_phase0_parked_vector_summary.json"
    vector_csv = REPO_DIR / "phase1" / "results" / "synthetic_phase0_parked_vector_rows.csv"
    summary = _load_json(summary_path)
    vector_rows = _load_csv_rows(vector_csv)
    vector_summary = _summarize_vector_rows(vector_rows)

    stages = (
        WorkflowStage(
            title="1. Synthetic Reference",
            status="ready",
            summary="Synthetic Phase 0 exports can exercise the same Phase 1 interface.",
            details=(
                f"n_vector_rows: {int(summary['n_vector_rows'])}",
                f"Recovered local field: {_format_vector(summary['full_scan_local_uT'], precision=2)}",
                f"Mean absolute vector error: {float(summary['mae_vector_uT']):.2f} uT",
                f"RMSE vector error: {float(summary['rmse_vector_uT']):.2f} uT",
            ),
        ),
        WorkflowStage(
            title="2. Time-Series Replay",
            status="ready",
            summary="This dataset is ideal for deterministic UI demos because it is dense and repeatable.",
            details=(
                f"First vector: {_format_vector(vector_summary['first_vector_uT'] or (0.0, 0.0, 0.0))}",
                f"Last vector: {_format_vector(vector_summary['last_vector_uT'] or (0.0, 0.0, 0.0))}",
                f"Residual median: {float(summary['median_lsq_residual_uT']):.2f} uT",
                f"Vector CSV: {vector_csv.relative_to(REPO_DIR)}",
            ),
        ),
    )
    return WorkflowDemo(
        key="synthetic-phase0",
        title="Synthetic Phase 0 Replay",
        subtitle="Deterministic regression-style demo using the Phase 1 wiring.",
        stages=stages,
        source_paths=(
            str(summary_path.relative_to(REPO_DIR)),
            str(vector_csv.relative_to(REPO_DIR)),
        ),
    )


def load_demo_catalog() -> tuple[WorkflowDemo, ...]:
    return (
        _build_april17_real_demo(),
        _build_april29_lockin_demo(),
        _build_synthetic_demo(),
    )


def _wrap_lines(text: str, width: int) -> list[str]:
    if width <= 1:
        return [text[: max(width, 0)]]
    wrapped = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    return wrapped or [""]


def _draw_bordered_panel(stdscr: curses.window, y: int, x: int, height: int, width: int, title: str) -> None:
    if height < 3 or width < 4:
        return
    stdscr.addstr(y, x, "+" + "-" * (width - 2) + "+")
    for row in range(y + 1, y + height - 1):
        stdscr.addstr(row, x, "|")
        stdscr.addstr(row, x + width - 1, "|")
    stdscr.addstr(y + height - 1, x, "+" + "-" * (width - 2) + "+")
    label = f" {title} "
    if len(label) < width - 2:
        stdscr.addstr(y, x + 2, label[: width - 4])


def _add_lines(
    stdscr: curses.window,
    lines: Sequence[str],
    y: int,
    x: int,
    width: int,
    max_lines: int,
    *,
    highlight_index: int | None = None,
) -> None:
    row = y
    for index, line in enumerate(lines[:max_lines]):
        attr = curses.A_REVERSE | curses.A_BOLD if highlight_index is not None and index == highlight_index else curses.A_NORMAL
        stdscr.addnstr(row, x, line.ljust(width), width, attr)
        row += 1


def _render_demo(stdscr: curses.window, demos: Sequence[WorkflowDemo], demo_index: int, stage_index: int) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    if height < 16 or width < 72:
        warning = "Terminal too small for the workflow TUI. Resize to at least 72x16."
        stdscr.addnstr(0, 0, warning, max(width - 1, 0), curses.A_BOLD)
        stdscr.refresh()
        return

    demo = demos[demo_index]
    stage = demo.stages[stage_index]
    header = f"NV Magnetometry Workflow TUI | demo {demo_index + 1}/{len(demos)} | {demo.title}"
    stdscr.addnstr(0, 0, header.ljust(width - 1), width - 1, curses.A_BOLD)
    stdscr.addnstr(1, 0, demo.subtitle.ljust(width - 1), width - 1)

    left_width = max(28, min(38, width // 3))
    right_x = left_width + 2
    right_width = width - right_x - 1
    body_height = height - 5

    _draw_bordered_panel(stdscr, 3, 0, body_height, left_width, "Workflow")
    _draw_bordered_panel(stdscr, 3, right_x, body_height, right_width, "Detail")

    nav_lines = [f"[{item.status}] {item.title}" for item in demo.stages]
    _add_lines(
        stdscr,
        [line[: left_width - 4] for line in nav_lines],
        4,
        2,
        left_width - 4,
        body_height - 2,
        highlight_index=stage_index,
    )

    detail_lines: list[str] = []
    detail_lines.extend(_wrap_lines(stage.summary, right_width - 4))
    detail_lines.append("")
    for item in stage.details:
        wrapped = _wrap_lines(f"- {item}", right_width - 4)
        detail_lines.extend(wrapped)
    detail_lines.append("")
    detail_lines.append("Sources:")
    for source_path in demo.source_paths:
        detail_lines.extend(_wrap_lines(f"  {source_path}", right_width - 4))
    _add_lines(stdscr, detail_lines, 4, right_x + 2, right_width - 4, body_height - 2)

    footer = "Up/Down: move  Left/Right or d: change demo  Home/End: jump  q: quit"
    stdscr.addnstr(height - 1, 0, footer.ljust(width - 1), width - 1, curses.A_DIM)
    stdscr.refresh()


def _run_demo_tui(stdscr: curses.window, demos: Sequence[WorkflowDemo], starting_demo: str) -> None:
    curses.curs_set(0)
    stdscr.keypad(True)

    demo_index = 0
    for index, demo in enumerate(demos):
        if demo.key == starting_demo:
            demo_index = index
            break
    stage_index = 0

    while True:
        stage_index = max(0, min(stage_index, len(demos[demo_index].stages) - 1))
        _render_demo(stdscr, demos, demo_index, stage_index)
        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            return
        if key in (curses.KEY_UP, ord("k")):
            stage_index = max(0, stage_index - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            stage_index = min(len(demos[demo_index].stages) - 1, stage_index + 1)
        elif key in (curses.KEY_LEFT, ord("h")):
            demo_index = (demo_index - 1) % len(demos)
            stage_index = 0
        elif key in (curses.KEY_RIGHT, ord("l"), ord("d"), ord("D")):
            demo_index = (demo_index + 1) % len(demos)
            stage_index = 0
        elif key == curses.KEY_HOME:
            stage_index = 0
        elif key == curses.KEY_END:
            stage_index = len(demos[demo_index].stages) - 1


def _load_operator_full_scan(path: Path, input_format: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if input_format in {"auto", "raw"}:
        try:
            raw_columns = _read_numeric_columns(path)
            if RAW_FREQUENCY_COLUMN in raw_columns and RAW_MW_ON_COLUMN in raw_columns:
                freqs = np.asarray(raw_columns[RAW_FREQUENCY_COLUMN], dtype=float)
                mw_on = np.asarray(raw_columns[RAW_MW_ON_COLUMN], dtype=float)
                measured = mw_on / abs(float(np.median(mw_on)))
                # Direct port of the phase1 compute_parking_simple pipeline.
                candidates = detect_baseline_corrected_dip_candidates(
                    freqs, mw_on, min_prominence_sigma=1.0,
                )
                clusters = cluster_dip_candidates(candidates, merge_distance_mhz=3.0)
                centers, _ = collapse_candidate_clusters(clusters, max_centers=8)
                sweep = (float(freqs.min()), float(freqs.max()))
                candidate_fits = fit_local_dips(
                    freqs, measured, centers,
                    window_mhz=8.0, model="lorentzian", hyperfine_merge_mhz=0.0,
                )
                kept_fits, _ = reject_unpaired_peaks(candidate_fits, sweep)
                kept_fits = sorted(kept_fits, key=lambda f: f.center_mhz)
                centers = np.asarray([f.center_mhz for f in kept_fits], dtype=float)
                return freqs, measured, centers
        except Exception:
            if input_format == "raw":
                raise

    if input_format in {"auto", "toolkit"}:
        try:
            batch = load_spectrum_csv(path)
            freqs = np.asarray(batch.freqs_mhz, dtype=float)
            measured = np.asarray(batch.spectra[0], dtype=float)

            # Direct port of phase1 compute_parking_apr17: detect all sub-peaks,
            # pair into doublets (gap 1.5–5 MHz), return 8 doublet midpoints.
            candidates = detect_baseline_corrected_dip_candidates(
                freqs, measured,
                min_prominence_sigma=1.5,
                baseline_window_mhz=12.0,
                smooth_window_mhz=1.5,
                min_width_mhz=0.4,
                max_width_mhz=8.0,
                min_distance_mhz=2.0,
                max_dips=None,
            )
            clusters = cluster_dip_candidates(candidates, merge_distance_mhz=1.5)
            sub_centers, _ = collapse_candidate_clusters(clusters, max_centers=None)
            sub_fits = fit_local_dips(
                freqs, measured, sub_centers,
                window_mhz=6.0, model="lorentzian", hyperfine_merge_mhz=0.0,
            )
            sub_fits = sorted(sub_fits, key=lambda f: f.center_mhz)

            used = [False] * len(sub_fits)
            midpoints: list[float] = []
            for i, fl in enumerate(sub_fits):
                if used[i]:
                    continue
                best_j, best_gap = None, np.inf
                for j, fr in enumerate(sub_fits):
                    if j <= i or used[j]:
                        continue
                    gap = fr.center_mhz - fl.center_mhz
                    if 1.5 <= gap <= 5.0 and gap < best_gap:
                        best_gap, best_j = gap, j
                if best_j is not None:
                    used[i] = used[best_j] = True
                    midpoints.append(0.5 * (fl.center_mhz + sub_fits[best_j].center_mhz))

            if len(midpoints) == 8:
                return freqs, measured, np.sort(np.asarray(midpoints, dtype=float))
        except Exception:
            if input_format == "toolkit":
                raise

    parser = _build_reconstruct_parser()
    args = parser.parse_args(
        [
            str(path),
            "--input-format",
            input_format,
            "--fit-signal",
            "normalized",
            "--max-dips",
            "8",
            "--min-dips",
            "2",
            "--min-distance-mhz",
            "3.0",
        ]
    )
    try:
        loaded = _load_reconstruction_input(args)
    except Exception:
        args = parser.parse_args(
            [
                str(path),
                "--input-format",
                input_format,
                "--fit-signal",
                "mw-on",
                "--max-dips",
                "8",
                "--min-dips",
                "2",
                "--min-distance-mhz",
                "3.0",
            ]
        )
        loaded = _load_reconstruction_input(args)

    freqs = np.asarray(loaded["freqs_mhz"], dtype=float)
    measured = np.asarray(loaded["measured"], dtype=float)
    peak_trace = np.asarray(loaded["peak_trace"], dtype=float)
    centers = _detect_transition_centers(
        freqs,
        measured,
        peak_trace,
        str(loaded["peak_mode"]),
        max_dips=8,
        min_dips=2,
        min_distance_mhz=3.0,
    )
    return freqs, measured, np.asarray(centers, dtype=float)


def _estimate_bias_field_mT(freqs_mhz: np.ndarray, measured: np.ndarray, transition_centers_mhz: np.ndarray) -> np.ndarray:
    summary = rank_spectrum_field_candidates(
        freqs_mhz,
        measured,
        np.asarray(transition_centers_mhz, dtype=float),
        nv_axes_preset="qdm",
        linewidth_mode="per_axis",
        contrast_mode="per_axis",
        hyperfine_mode="fixed",
        initial_hyperfine_splitting_mhz=2.16,
        local_field_bounds_mT=1.5,
    )
    return np.asarray(summary.best["B_total_mT"], dtype=float)


_FWHM_FLOOR_MHZ = 0.200
_FWHM_FALLBACK_MHZ = 0.500


def _safe_fwhm(fwhm: float) -> float:
    """Return fwhm, replacing pathologically narrow values with a fixed fallback."""
    return float(fwhm) if float(fwhm) > _FWHM_FLOOR_MHZ + 0.01 else _FWHM_FALLBACK_MHZ


_PLACEMENT_HALF_MAX = "half_max"
_PLACEMENT_MAX_SLOPE = "max_slope"
_PLACEMENT_EMPIRICAL = "empirical"
_PLACEMENT_EMPIRICAL_HALF_MAX = "empirical_half_max"
_PLACEMENT_AUTO = "auto"
# Analytical inflection-point factors (used by half_max and max_slope):
#   Gaussian:   center ± FWHM/(2√(2 ln 2)) ≈ 0.4247 · FWHM
#   Half-max:   center ± FWHM/2            = 0.5000 · FWHM
import math as _math
_PLACEMENT_FACTORS = {
    _PLACEMENT_HALF_MAX: 0.5,
    _PLACEMENT_MAX_SLOPE: 1.0 / (2.0 * _math.sqrt(2.0 * _math.log(2.0))),  # ≈ 0.4247 Gaussian
}
_WEAK_CONTRAST_FOR_FIT_HALF_MAX = 0.008


def _crossing(f_arr: np.ndarray, v_arr: np.ndarray, target: float, decreasing: bool) -> float | None:
    """Linear-interpolation crossing of `target` in a monotone segment."""
    for i in range(len(v_arr) - 1):
        a, b = float(v_arr[i]), float(v_arr[i + 1])
        if decreasing and a >= target >= b:
            t = (target - a) / (b - a)
            return float(f_arr[i]) + t * (float(f_arr[i + 1]) - float(f_arr[i]))
        if not decreasing and a <= target <= b:
            t = (target - a) / (b - a)
            return float(f_arr[i]) + t * (float(f_arr[i + 1]) - float(f_arr[i]))
    return None


def _local_linear_slope(freqs: np.ndarray, values: np.ndarray, f_center: float, half_width: float) -> float:
    """Slope (PL/MHz) from linear fit to raw data in f_center ± half_width."""
    mask = (freqs >= f_center - half_width) & (freqs <= f_center + half_width)
    f_loc, v_loc = freqs[mask], values[mask]
    if len(f_loc) < 2:
        return 0.0
    return float(np.polyfit(f_loc, v_loc, 1)[0])


def _quadratic_vertex_x(x: np.ndarray, y: np.ndarray, index: int) -> float:
    """Sub-sample x position of a local extremum from three neighboring samples."""
    if index <= 0 or index >= len(x) - 1:
        return float(x[index])
    x3 = np.asarray(x[index - 1:index + 2], dtype=float)
    y3 = np.asarray(y[index - 1:index + 2], dtype=float)
    try:
        a, b, _ = np.polyfit(x3, y3, 2)
    except Exception:
        return float(x[index])
    if a >= 0.0 or abs(a) < 1e-18:
        return float(x[index])
    vertex = float(-b / (2.0 * a))
    if float(x3[0]) <= vertex <= float(x3[-1]):
        return vertex
    return float(x[index])


def _target_crossing_range(v_arr: np.ndarray, decreasing: bool) -> tuple[float, float] | None:
    """ADC target interval reachable by a monotone crossing scan."""
    if len(v_arr) < 2:
        return None
    return float(np.min(v_arr)), float(np.max(v_arr))


def _empirical_parking_pair(
    freqs: np.ndarray,
    values: np.ndarray,
    center: float,
    search_half_window: float,
    other_peak_centers: Sequence[float] = (),
    expected_minus_mhz: float | None = None,
    expected_plus_mhz: float | None = None,
    target_adc: float | None = None,
    seed_search_half_width_mhz: float = 1.0,
    slope_fit_half_width: float = 1.5,
    smooth_points: int = 5,
    min_neighbor_clearance_mhz: float = 1.0,
) -> tuple[float | None, float | None, float, float]:
    """Empirical max-slope pair corrected to equal ADC count.

    First find continuous-frequency max-slope seeds on each side from a smoothed
    raw spectrum.  Then choose one reachable ADC target that minimizes the
    frequency displacement from those seeds, so equal count is enforced without
    discarding the slope optimum.
    """
    from scipy.ndimage import uniform_filter1d

    mask = (freqs >= center - search_half_window) & (freqs <= center + search_half_window)
    f_loc = freqs[mask]
    v_loc = values[mask]
    if len(f_loc) < 6:
        return None, None, 0.0, 0.0

    left_mask  = f_loc <= center
    right_mask = f_loc >= center
    f_l, v_l = f_loc[left_mask],  v_loc[left_mask]
    f_r, v_r = f_loc[right_mask], v_loc[right_mask]

    if len(f_l) < 3 or len(f_r) < 3:
        return None, None, 0.0, 0.0

    smooth_size = max(3, int(smooth_points))
    if smooth_size % 2 == 0:
        smooth_size += 1
    if len(v_loc) < smooth_size:
        smooth_size = len(v_loc) if len(v_loc) % 2 == 1 else len(v_loc) - 1
        smooth_size = max(3, smooth_size)
    v_smooth = uniform_filter1d(v_loc.astype(float), size=smooth_size, mode="nearest")
    grad = np.gradient(v_smooth, f_loc)
    g_l = grad[left_mask]
    g_r = grad[right_mask]

    left_seed_mask = np.ones(len(f_l), dtype=bool)
    right_seed_mask = np.ones(len(f_r), dtype=bool)
    if expected_minus_mhz is not None:
        left_seed_mask = np.abs(f_l - float(expected_minus_mhz)) <= float(seed_search_half_width_mhz)
        if not np.any(left_seed_mask):
            left_seed_mask = np.ones(len(f_l), dtype=bool)
    if expected_plus_mhz is not None:
        right_seed_mask = np.abs(f_r - float(expected_plus_mhz)) <= float(seed_search_half_width_mhz)
        if not np.any(right_seed_mask):
            right_seed_mask = np.ones(len(f_r), dtype=bool)

    left_candidates = np.flatnonzero(left_seed_mask)
    right_candidates = np.flatnonzero(right_seed_mask)
    left_idx = int(left_candidates[np.argmax(np.abs(g_l[left_seed_mask]))])
    right_idx = int(right_candidates[np.argmax(np.abs(g_r[right_seed_mask]))])
    f_db_seed = _quadratic_vertex_x(f_l, np.abs(g_l), left_idx)
    f_ib_seed = _quadratic_vertex_x(f_r, np.abs(g_r), right_idx)
    v_db_seed = float(np.interp(f_db_seed, f_l, v_l))
    v_ib_seed = float(np.interp(f_ib_seed, f_r, v_r))
    slope_db_seed = _local_linear_slope(freqs, values, f_db_seed, slope_fit_half_width)
    slope_ib_seed = _local_linear_slope(freqs, values, f_ib_seed, slope_fit_half_width)
    if abs(slope_db_seed) < 1e-15 or abs(slope_ib_seed) < 1e-15:
        return None, None, 0.0, 0.0

    left_range = _target_crossing_range(v_l, decreasing=True)
    right_range = _target_crossing_range(v_r, decreasing=False)
    if left_range is None or right_range is None:
        return None, None, 0.0, 0.0
    target_low = max(left_range[0], right_range[0])
    target_high = min(left_range[1], right_range[1])
    if target_low > target_high:
        return None, None, 0.0, 0.0

    if target_adc is None:
        w_db = 1.0 / (slope_db_seed * slope_db_seed)
        w_ib = 1.0 / (slope_ib_seed * slope_ib_seed)
        target = (w_db * v_db_seed + w_ib * v_ib_seed) / (w_db + w_ib)
        target = min(target, v_db_seed, v_ib_seed)
    else:
        target = float(target_adc)
    target = float(np.clip(target, target_low, target_high))

    # If a point is close to another fitted peak, scan nearby equal-ADC targets
    # and choose the one with better clearance while staying near max slope.
    other_centers = np.asarray([c for c in other_peak_centers if abs(float(c) - center) > 1e-6], dtype=float)
    if len(other_centers) > 0:
        candidates = np.linspace(target_low, target_high, 41)
        best_score = -float("inf")
        best_target = target
        max_shift = max(abs(target_high - target_low), 1e-15)
        for candidate_target in candidates:
            cand_db = _crossing(f_l, v_l, float(candidate_target), decreasing=True)
            cand_ib = _crossing(f_r, v_r, float(candidate_target), decreasing=False)
            if cand_db is None or cand_ib is None:
                continue
            clearance = min(
                float(np.min(np.abs(other_centers - cand_db))),
                float(np.min(np.abs(other_centers - cand_ib))),
            )
            shift_penalty = abs(float(candidate_target) - target) / max_shift
            clearance_score = min(clearance, min_neighbor_clearance_mhz) / min_neighbor_clearance_mhz
            score = clearance_score - 0.25 * shift_penalty
            if score > best_score:
                best_score = score
                best_target = float(candidate_target)
        target = best_target

    f_db = _crossing(f_l, v_l, target, decreasing=True)
    f_ib = _crossing(f_r, v_r, target, decreasing=False)
    if f_db is None or f_ib is None:
        return None, None, 0.0, 0.0

    slope_db = _local_linear_slope(freqs, values, f_db, slope_fit_half_width)
    slope_ib = _local_linear_slope(freqs, values, f_ib, slope_fit_half_width)
    return f_db, f_ib, slope_db, slope_ib


def _auto_placement_for_fit(contrast: float) -> str:
    """Choose a robust parked placement from fitted peak strength.

    Weak dips make empirical gradients unstable and their raw equal-ADC crossing
    can overfit sweep noise.  For those, use the fit-derived half-width to keep
    range.  Stronger dips can afford the empirical half-height equal-ADC
    correction.
    """
    if float(contrast) < _WEAK_CONTRAST_FOR_FIT_HALF_MAX:
        return _PLACEMENT_HALF_MAX
    return _PLACEMENT_EMPIRICAL_HALF_MAX


def _suggest_parked_frequencies(
    freqs_mhz: np.ndarray,
    spectrum: np.ndarray,
    transition_centers_mhz: np.ndarray,
    placement: str = _PLACEMENT_AUTO,
) -> tuple[ParkedFrequencyPlanEntry, ...]:
    freqs = np.asarray(freqs_mhz, dtype=float)
    values = np.asarray(spectrum, dtype=float)
    centers = np.sort(np.asarray(transition_centers_mhz, dtype=float))
    gradient = np.gradient(values, freqs)
    if placement in {_PLACEMENT_EMPIRICAL_HALF_MAX, _PLACEMENT_AUTO}:
        placement_factor = _PLACEMENT_FACTORS[_PLACEMENT_HALF_MAX]
    else:
        placement_factor = _PLACEMENT_FACTORS.get(placement, _PLACEMENT_FACTORS[_PLACEMENT_MAX_SLOPE])

    # Detect all sub-peaks without merging so resolved hyperfine components
    # (e.g. 15N doublets) are found as separate candidates.
    sub_candidates = detect_baseline_corrected_dip_candidates(
        freqs, values,
        min_prominence_sigma=1.0,
        baseline_window_mhz=30.0,
        smooth_window_mhz=1.5,
        min_width_mhz=0.3,
        max_width_mhz=10.0,
        min_distance_mhz=1.0,
        max_dips=None,
    )
    sub_clusters = cluster_dip_candidates(sub_candidates, merge_distance_mhz=0.5)
    sub_centers_arr, _ = collapse_candidate_clusters(sub_clusters, max_centers=None)

    successful_fits: list = []
    if len(sub_centers_arr) > 0:
        raw_fits = fit_local_dips(freqs, values, sub_centers_arr,
                                   window_mhz=8.0, model="gaussian", hyperfine_merge_mhz=0.0)
        successful_fits = [f for f in raw_fits if f.success]

    plan: list[ParkedFrequencyPlanEntry] = []
    for transition_index, center_guess in enumerate(centers[:8]):
        # Search window: no wider than half the gap to the nearest other transition
        # center, capped at 4 MHz to avoid cross-transition contamination.
        if len(centers) > 1:
            gaps = [abs(float(center_guess) - float(c)) for c in centers
                    if abs(float(center_guess) - float(c)) > 1e-6]
            search_window = min(4.0, min(gaps) / 2.0)
        else:
            search_window = 4.0

        local_fits = sorted(
            [f for f in successful_fits if abs(f.center_mhz - float(center_guess)) <= search_window],
            key=lambda f: f.center_mhz,
        )

        if local_fits:
            fl = local_fits[0]   # leftmost sub-peak  → DB side
            fr = local_fits[-1]  # rightmost sub-peak → IB side
            fwhm_l = _safe_fwhm(fl.linewidth_mhz)
            fwhm_r = _safe_fwhm(fr.linewidth_mhz)
            center = float(0.5 * (fl.center_mhz + fr.center_mhz))
            linewidth = float(0.5 * (fwhm_l + fwhm_r))
            contrast = float(np.median([fl.contrast, fr.contrast]))
            effective_placement = _auto_placement_for_fit(contrast) if placement == _PLACEMENT_AUTO else placement
            if effective_placement == _PLACEMENT_EMPIRICAL_HALF_MAX:
                local_factor = _PLACEMENT_FACTORS[_PLACEMENT_HALF_MAX]
            else:
                local_factor = _PLACEMENT_FACTORS.get(effective_placement, _PLACEMENT_FACTORS[_PLACEMENT_MAX_SLOPE])
            target_adc = None
            if effective_placement == _PLACEMENT_EMPIRICAL_HALF_MAX:
                target_adc = float(np.median([
                    fl.baseline - 0.5 * fl.contrast,
                    fr.baseline - 0.5 * fr.contrast,
                ]))
            f_minus = float(fl.center_mhz) - fwhm_l * local_factor
            f_plus = float(fr.center_mhz) + fwhm_r * local_factor
        else:
            # Fallback: direct fit at the given transition center.
            fallback = fit_local_dips(freqs, values, np.asarray([float(center_guess)]),
                                       window_mhz=8.0, model="gaussian", hyperfine_merge_mhz=0.0)
            if fallback and fallback[0].success:
                f0 = fallback[0]
                center = float(f0.center_mhz)
                linewidth = _safe_fwhm(f0.linewidth_mhz)
                contrast = float(f0.contrast)
                effective_placement = _auto_placement_for_fit(contrast) if placement == _PLACEMENT_AUTO else placement
                target_adc = float(f0.baseline - 0.5 * f0.contrast) if effective_placement == _PLACEMENT_EMPIRICAL_HALF_MAX else None
            else:
                center = float(center_guess)
                linewidth = _FWHM_FALLBACK_MHZ
                effective_placement = _PLACEMENT_HALF_MAX if placement == _PLACEMENT_AUTO else placement
                target_adc = None
            if effective_placement == _PLACEMENT_EMPIRICAL_HALF_MAX:
                local_factor = _PLACEMENT_FACTORS[_PLACEMENT_HALF_MAX]
            else:
                local_factor = _PLACEMENT_FACTORS.get(effective_placement, _PLACEMENT_FACTORS[_PLACEMENT_MAX_SLOPE])
            f_minus = center - linewidth * local_factor
            f_plus = center + linewidth * local_factor

        # Empirical override: seed at continuous-frequency max slope, then
        # correct to one equal-ADC target with minimal frequency displacement.
        slope_minus = float(np.interp(f_minus, freqs, gradient))
        slope_plus  = float(np.interp(f_plus,  freqs, gradient))
        if effective_placement in {_PLACEMENT_EMPIRICAL, _PLACEMENT_EMPIRICAL_HALF_MAX}:
            em_db, em_ib, em_slope_db, em_slope_ib = _empirical_parking_pair(
                freqs,
                values,
                center,
                search_window,
                other_peak_centers=[float(f.center_mhz) for f in successful_fits],
                expected_minus_mhz=f_minus,
                expected_plus_mhz=f_plus,
                target_adc=target_adc,
            )
            if em_db is not None and em_ib is not None:
                f_minus, f_plus = em_db, em_ib
                slope_minus, slope_plus = em_slope_db, em_slope_ib

        plan.append(
            ParkedFrequencyPlanEntry(
                transition_index=int(transition_index),
                center_mhz=center,
                linewidth_mhz=linewidth,
                frequency_minus_mhz=f_minus,
                frequency_plus_mhz=f_plus,
                slope_minus_per_mhz=slope_minus,
                slope_plus_per_mhz=slope_plus,
            )
        )

    return tuple(plan)


def _prompt_text(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value if value else (default or "")


def _resolve_bias_input(text: str, auto_bias_mT: np.ndarray) -> np.ndarray:
    if not text.strip():
        return np.asarray(auto_bias_mT, dtype=float)
    candidate = Path(text)
    if candidate.exists():
        return np.asarray(load_base_field_csv(candidate).vector_mT, dtype=float)
    return np.asarray(_parse_vector(text), dtype=float)


def _write_parked_plan_csv(path: Path, plan: Sequence[ParkedFrequencyPlanEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "block_index",
                "point_in_block",
                "point_index",
                "transition_index",
                "transition_center_mhz",
                "linewidth_mhz",
                "frequency_mhz",
                "slope_per_mhz",
            ],
        )
        writer.writeheader()
        point_index = 0
        for block_index, entry in enumerate(plan, start=1):
            writer.writerow(
                {
                    "block_index": block_index,
                    "point_in_block": 0,
                    "point_index": point_index,
                    "transition_index": entry.transition_index,
                    "transition_center_mhz": entry.center_mhz,
                    "linewidth_mhz": entry.linewidth_mhz,
                    "frequency_mhz": entry.frequency_minus_mhz,
                    "slope_per_mhz": entry.slope_minus_per_mhz,
                }
            )
            point_index += 1
            writer.writerow(
                {
                    "block_index": block_index,
                    "point_in_block": 1,
                    "point_index": point_index,
                    "transition_index": entry.transition_index,
                    "transition_center_mhz": entry.center_mhz,
                    "linewidth_mhz": entry.linewidth_mhz,
                    "frequency_mhz": entry.frequency_plus_mhz,
                    "slope_per_mhz": entry.slope_plus_per_mhz,
                }
            )
            point_index += 1


def _write_wide_template_csv(path: Path, plan: Sequence[ParkedFrequencyPlanEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["time_s"] + [f"peak_{index:02d}" for index in range(1, 2 * len(plan) + 1)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)


def _write_long_template_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "frequency_mhz", "intensity"])


def _infer_parked_format(path: Path) -> str:
    if not path.exists():
        return "wide"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            fieldnames = next(reader)
        except StopIteration:
            return "wide"
    lowered = [item.strip().lower() for item in fieldnames]
    if {"save_id", "timestamp_epoch_s", "block_index", "point_in_block", "frequency_mhz", "normalized_intensity"}.issubset(lowered):
        return "toolkit"
    if len(fieldnames) >= 17 and _pick_column(fieldnames, TIME_COLUMN_CANDIDATES) is not None:
        return "wide"
    if _pick_column(fieldnames, TIME_COLUMN_CANDIDATES) and _pick_column(fieldnames, FREQUENCY_COLUMN_CANDIDATES) and _pick_column(fieldnames, INTENSITY_COLUMN_CANDIDATES):
        return "long"
    return "wide"


def _parked_plan_points(plan: Sequence[ParkedFrequencyPlanEntry]) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    point_index = 0
    for block_index, entry in enumerate(plan, start=1):
        points.append(
            {
                "block_index": block_index,
                "point_in_block": 0,
                "point_index": point_index,
                "frequency_mhz": entry.frequency_minus_mhz,
            }
        )
        point_index += 1
        points.append(
            {
                "block_index": block_index,
                "point_in_block": 1,
                "point_index": point_index,
                "frequency_mhz": entry.frequency_plus_mhz,
            }
        )
        point_index += 1
    return points


def _convert_wide_rows_to_toolkit_rows(rows: Sequence[dict[str, str]], plan: Sequence[ParkedFrequencyPlanEntry]) -> list[dict[str, object]]:
    if not rows:
        return []
    time_col = _pick_column(rows[0].keys(), TIME_COLUMN_CANDIDATES)
    if time_col is None:
        raise ValueError("Wide parked file needs a time column such as time_s or timestamp_epoch_s")
    value_columns = [column for column in rows[0].keys() if column != time_col][: 2 * len(plan)]
    expected_count = 2 * len(plan)
    if len(value_columns) < expected_count:
        raise ValueError(f"Wide parked file needs at least {expected_count} peak columns after the time column")

    toolkit_rows: list[dict[str, object]] = []
    plan_points = _parked_plan_points(plan)
    for sample_index, row in enumerate(rows):
        timestamp = float(row[time_col])
        save_id = f"{sample_index:06d}"
        valid = True
        sample_rows: list[dict[str, object]] = []
        for point_index, plan_point in enumerate(plan_points):
            value = row[value_columns[point_index]]
            if value == "":
                valid = False
                break
            sample_rows.append(
                {
                    "save_id": save_id,
                    "timestamp_epoch_s": timestamp,
                    "block_index": int(plan_point["block_index"]),
                    "point_in_block": int(plan_point["point_in_block"]),
                    "point_index": int(plan_point["point_index"]),
                    "frequency_mhz": float(plan_point["frequency_mhz"]),
                    "normalized_intensity": float(value),
                }
            )
        if valid:
            toolkit_rows.extend(sample_rows)
    return toolkit_rows


def _convert_long_rows_to_toolkit_rows(rows: Sequence[dict[str, str]], plan: Sequence[ParkedFrequencyPlanEntry]) -> list[dict[str, object]]:
    if not rows:
        return []
    time_col = _pick_column(rows[0].keys(), TIME_COLUMN_CANDIDATES)
    freq_col = _pick_column(rows[0].keys(), FREQUENCY_COLUMN_CANDIDATES)
    intensity_col = _pick_column(rows[0].keys(), INTENSITY_COLUMN_CANDIDATES)
    save_id_col = _pick_column(rows[0].keys(), SAVE_ID_COLUMN_CANDIDATES)
    if time_col is None or freq_col is None or intensity_col is None:
        raise ValueError("Long parked file needs time, frequency, and intensity columns")

    plan_points = _parked_plan_points(plan)
    plan_freqs = np.asarray([float(point["frequency_mhz"]) for point in plan_points], dtype=float)
    spacing = np.diff(np.sort(plan_freqs))
    min_spacing = float(np.min(spacing)) if spacing.size > 0 else 3.0
    tolerance_mhz = max(0.75, 0.25 * min_spacing)

    grouped: dict[str, list[dict[str, str]]] = {}
    ordered_group_ids: list[str] = []
    for row in rows:
        group_id = str(row[save_id_col]) if save_id_col is not None else str(row[time_col])
        if group_id not in grouped:
            grouped[group_id] = []
            ordered_group_ids.append(group_id)
        grouped[group_id].append(row)

    toolkit_rows: list[dict[str, object]] = []
    sample_index = 0
    for group_id in ordered_group_ids:
        group_rows = grouped[group_id]
        assigned: dict[int, dict[str, object]] = {}
        timestamp = float(group_rows[0][time_col])
        for row in group_rows:
            freq_value = float(row[freq_col])
            nearest_index = int(np.argmin(np.abs(plan_freqs - freq_value)))
            if abs(float(plan_freqs[nearest_index]) - freq_value) > tolerance_mhz:
                continue
            plan_point = plan_points[nearest_index]
            assigned[nearest_index] = {
                "save_id": f"{sample_index:06d}",
                "timestamp_epoch_s": timestamp,
                "block_index": int(plan_point["block_index"]),
                "point_in_block": int(plan_point["point_in_block"]),
                "point_index": int(plan_point["point_index"]),
                "frequency_mhz": float(plan_point["frequency_mhz"]),
                "normalized_intensity": float(row[intensity_col]),
            }
        if len(assigned) == len(plan_points):
            for point_index in range(len(plan_points)):
                toolkit_rows.append(assigned[point_index])
            sample_index += 1
            continue

        if len(group_rows) != len(plan_points):
            continue

        for point_index, row in enumerate(group_rows):
            plan_point = plan_points[point_index]
            toolkit_rows.append(
                {
                    "save_id": f"{sample_index:06d}",
                    "timestamp_epoch_s": timestamp,
                    "block_index": int(plan_point["block_index"]),
                    "point_in_block": int(plan_point["point_in_block"]),
                    "point_index": int(plan_point["point_index"]),
                    "frequency_mhz": float(plan_point["frequency_mhz"]),
                    "normalized_intensity": float(row[intensity_col]),
                }
            )
        sample_index += 1
    return toolkit_rows


def _filter_complete_toolkit_rows(rows: Sequence[dict[str, str]], plan: Sequence[ParkedFrequencyPlanEntry]) -> list[dict[str, object]]:
    plan_length = 2 * len(plan)
    grouped: dict[str, list[dict[str, str]]] = {}
    ordered_ids: list[str] = []
    for row in rows:
        group_id = str(row["save_id"])
        if group_id not in grouped:
            grouped[group_id] = []
            ordered_ids.append(group_id)
        grouped[group_id].append(row)

    toolkit_rows: list[dict[str, object]] = []
    sample_index = 0
    for group_id in ordered_ids:
        group_rows = sorted(grouped[group_id], key=lambda item: float(item["point_index"]))
        if len(group_rows) < plan_length:
            continue
        complete = group_rows[:plan_length]
        toolkit_rows.extend(
            {
                "save_id": f"{sample_index:06d}",
                "timestamp_epoch_s": float(row["timestamp_epoch_s"]),
                "block_index": int(row["block_index"]),
                "point_in_block": int(row["point_in_block"]),
                "point_index": int(row["point_index"]),
                "frequency_mhz": float(row["frequency_mhz"]),
                "normalized_intensity": float(row["normalized_intensity"]),
            }
            for row in complete
        )
        sample_index += 1
    return toolkit_rows


def _toolkit_rows_to_series(rows: Sequence[dict[str, object]]) -> ParkedFrequencySeries:
    if not rows:
        raise ValueError("No complete parked-frequency samples found yet")

    grouped: dict[str, list[dict[str, object]]] = {}
    ordered_ids: list[str] = []
    for row in rows:
        group_id = str(row["save_id"])
        if group_id not in grouped:
            grouped[group_id] = []
            ordered_ids.append(group_id)
        grouped[group_id].append(row)

    first_rows = sorted(grouped[ordered_ids[0]], key=lambda item: int(item["point_index"]))
    freqs_mhz = np.asarray([float(row["frequency_mhz"]) for row in first_rows], dtype=float)
    block_indices = np.asarray([int(row["block_index"]) for row in first_rows], dtype=int)
    point_in_block = np.asarray([int(row["point_in_block"]) for row in first_rows], dtype=int)

    intensities: list[np.ndarray] = []
    timestamps: list[float] = []
    metadata: list[dict[str, str]] = []
    save_ids: list[str] = []
    for save_id in ordered_ids:
        sample_rows = sorted(grouped[save_id], key=lambda item: int(item["point_index"]))
        if len(sample_rows) != len(first_rows):
            continue
        sample_freqs = np.asarray([float(row["frequency_mhz"]) for row in sample_rows], dtype=float)
        if not np.allclose(sample_freqs, freqs_mhz, atol=1e-6, rtol=0.0):
            continue
        intensities.append(np.asarray([float(row["normalized_intensity"]) for row in sample_rows], dtype=float))
        timestamps.append(float(sample_rows[0]["timestamp_epoch_s"]))
        metadata.append({str(key): str(value) for key, value in sample_rows[0].items()})
        save_ids.append(str(save_id))

    if not intensities:
        raise ValueError("No complete parked-frequency samples matched the suggested plan")

    return ParkedFrequencySeries(
        freqs_mhz=freqs_mhz,
        intensities=np.asarray(intensities, dtype=float),
        save_ids=save_ids,
        timestamps_epoch_s=np.asarray(timestamps, dtype=float),
        block_indices=block_indices,
        point_in_block=point_in_block,
        metadata=metadata,
    )


def _load_operator_parked_series(path: Path, parked_format: str, plan: Sequence[ParkedFrequencyPlanEntry]) -> ParkedFrequencySeries:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = _load_csv_rows(path)
    if not rows:
        raise ValueError("Parked file exists but contains no rows yet")

    resolved_format = parked_format if parked_format != "auto" else _infer_parked_format(path)
    if resolved_format == "wide":
        toolkit_rows = _convert_wide_rows_to_toolkit_rows(rows, plan)
    elif resolved_format == "long":
        toolkit_rows = _convert_long_rows_to_toolkit_rows(rows, plan)
    elif resolved_format == "toolkit":
        toolkit_rows = _filter_complete_toolkit_rows(rows, plan)
    else:
        raise ValueError(f"Unsupported parked format: {resolved_format}")
    return _toolkit_rows_to_series(toolkit_rows)


def _compute_live_snapshot(config: OperatorConfig) -> LiveMonitorSnapshot:
    try:
        parked_series = _load_operator_parked_series(config.parked_data_path, config.parked_format, config.parked_plan)
    except FileNotFoundError:
        return LiveMonitorSnapshot(
            status="waiting",
            message=f"Waiting for parked data file: {config.parked_data_path}",
            sample_count=0,
            ok_count=0,
            latest_timestamp_s=None,
            latest_local_uT=None,
            latest_total_uT=None,
            latest_residual_uT=None,
            latest_rank=None,
            latest_channels=None,
            vector_rows=(),
            projection_rows=(),
        )
    except Exception as exc:
        return LiveMonitorSnapshot(
            status="error",
            message=str(exc),
            sample_count=0,
            ok_count=0,
            latest_timestamp_s=None,
            latest_local_uT=None,
            latest_total_uT=None,
            latest_residual_uT=None,
            latest_rank=None,
            latest_channels=None,
            vector_rows=(),
            projection_rows=(),
        )

    calibrations = build_blockwise_calibrations(
        config.reference_freqs_mhz,
        config.reference_spectrum,
        parked_series.freqs_mhz,
        parked_series.block_indices,
        parked_series.point_in_block,
        B_bias_mT=np.asarray(config.bias_field_mT, dtype=float),
        nv_axes_preset="qdm",
    )
    projection_rows = estimate_parked_series_fields(
        parked_series.intensities,
        parked_series.timestamps_epoch_s,
        parked_series.save_ids,
        parked_series.block_indices,
        parked_series.point_in_block,
        parked_series.freqs_mhz,
        calibrations,
        nv_axes_preset="qdm",
        search_radius_mT=0.0,
    )
    vector_rows = reconstruct_vector_timeseries(projection_rows)
    ok_rows = [row for row in vector_rows if str(row.get("status", "")) == "ok"]
    latest_row = ok_rows[-1] if ok_rows else (vector_rows[-1] if vector_rows else None)
    latest_local = None
    latest_total = None
    latest_residual = None
    latest_rank = None
    latest_channels = None
    latest_timestamp = None
    if latest_row is not None:
        latest_timestamp = _safe_float(latest_row, "timestamp_epoch_s")
        latest_rank = int(latest_row["rank"])
        latest_channels = int(latest_row["n_channels"])
        latest_residual = _safe_float(latest_row, "residual_norm_uT")
        if math.isfinite(_safe_float(latest_row, "delta_Bx_uT")):
            latest_local = (
                _safe_float(latest_row, "delta_Bx_uT"),
                _safe_float(latest_row, "delta_By_uT"),
                _safe_float(latest_row, "delta_Bz_uT"),
            )
            bias_uT = np.asarray(config.bias_field_mT, dtype=float) * 1000.0
            latest_total = (
                float(bias_uT[0] + latest_local[0]),
                float(bias_uT[1] + latest_local[1]),
                float(bias_uT[2] + latest_local[2]),
            )

    return LiveMonitorSnapshot(
        status="ok" if ok_rows else "limited",
        message="Live reconstruction running." if ok_rows else "Samples loaded, but the current solve is not full-rank yet.",
        sample_count=len(vector_rows),
        ok_count=len(ok_rows),
        latest_timestamp_s=latest_timestamp,
        latest_local_uT=latest_local,
        latest_total_uT=latest_total,
        latest_residual_uT=latest_residual,
        latest_rank=latest_rank,
        latest_channels=latest_channels,
        vector_rows=tuple(vector_rows),
        projection_rows=tuple(projection_rows),
    )


def _render_plan_table(plan: Sequence[ParkedFrequencyPlanEntry]) -> str:
    lines = ["Suggested parked frequencies (MHz):"]
    for block_index, entry in enumerate(plan, start=1):
        lines.append(
            f"  block {block_index:02d}: {entry.frequency_minus_mhz:9.3f}, {entry.frequency_plus_mhz:9.3f}"
            f"  center={entry.center_mhz:9.3f}  FWHM={entry.linewidth_mhz:6.3f}"
        )
    return "\n".join(lines)


def _prepare_operator_config(args: argparse.Namespace) -> OperatorConfig:
    full_scan_text = str(args.full_scan) if args.full_scan else _prompt_text("May I have the path to the full ODMR spectra CSV")
    full_scan_path = Path(full_scan_text).expanduser().resolve()
    if not full_scan_path.exists():
        raise FileNotFoundError(f"Full ODMR spectra file not found: {full_scan_path}")

    freqs_mhz, measured, transition_centers = _load_operator_full_scan(full_scan_path, args.input_format)
    parked_plan = _suggest_parked_frequencies(freqs_mhz, measured, transition_centers)
    auto_bias_mT = _estimate_bias_field_mT(freqs_mhz, measured, transition_centers)

    print()
    print(_render_plan_table(parked_plan))
    print()
    print(f"Estimated bias field from the full scan: {_format_vector(auto_bias_mT, precision=4, unit='mT')}")
    print("Press Enter to accept that estimate, or enter either:")
    print("  1. a Bx,By,Bz vector in mT")
    print("  2. a path to a base-field CSV")
    bias_text = args.bias_field or _prompt_text("Bias field override (Enter = accept estimate)", default="")
    bias_field_mT = _resolve_bias_input(bias_text, auto_bias_mT)

    parked_path_text = str(args.parked_data) if args.parked_data else _prompt_text("Path to the live parked-data CSV")
    parked_data_path = Path(parked_path_text).expanduser().resolve()
    parked_format = args.parked_format if args.parked_format != "auto" else _infer_parked_format(parked_data_path)
    print(f"Using parked-data format: {parked_format}")

    output_dir = parked_data_path.parent if parked_data_path.parent.exists() else full_scan_path.parent
    plan_csv_path = output_dir / "parked_plan_16freq.csv"
    wide_template_path = output_dir / "parked_template_wide.csv"
    long_template_path = output_dir / "parked_template_long.csv"
    _write_parked_plan_csv(plan_csv_path, parked_plan)
    _write_wide_template_csv(wide_template_path, parked_plan)
    _write_long_template_csv(long_template_path)

    print(f"Parked plan CSV written to: {plan_csv_path}")
    print(f"Wide-format template written to: {wide_template_path}")
    print(f"Long-format template written to: {long_template_path}")
    print()
    print("Accepted parked-data formats:")
    print("  1. Wide: time_s, peak_01, peak_02, ... peak_16")
    print("  2. Long: time_s, frequency_mhz, intensity")
    print("  3. Toolkit: save_id, timestamp_epoch_s, block_index, point_in_block, point_index, frequency_mhz, normalized_intensity")
    print()
    input("Press Enter to start live monitoring...")

    return OperatorConfig(
        full_scan_path=full_scan_path,
        parked_data_path=parked_data_path,
        parked_format=parked_format,
        bias_field_mT=np.asarray(bias_field_mT, dtype=float),
        reference_freqs_mhz=np.asarray(freqs_mhz, dtype=float),
        reference_spectrum=np.asarray(measured, dtype=float),
        transition_centers_mhz=np.asarray(transition_centers, dtype=float),
        parked_plan=tuple(parked_plan),
        plan_csv_path=plan_csv_path,
        wide_template_path=wide_template_path,
        long_template_path=long_template_path,
        poll_interval_s=float(args.poll_interval),
    )


def _draw_ascii_plot(
    stdscr: curses.window,
    rows: Sequence[dict[str, object]],
    y: int,
    x: int,
    height: int,
    width: int,
) -> None:
    if height < 4 or width < 10:
        return
    ok_rows = [row for row in rows if str(row.get("status", "")) == "ok"]
    if not ok_rows:
        stdscr.addnstr(y, x, "Waiting for rank-ok vector rows...", width)
        return

    series = {
        "x": [_safe_float(row, "delta_Bx_uT") for row in ok_rows][-width:],
        "y": [_safe_float(row, "delta_By_uT") for row in ok_rows][-width:],
        "z": [_safe_float(row, "delta_Bz_uT") for row in ok_rows][-width:],
    }
    finite_values = [value for values in series.values() for value in values if math.isfinite(value)]
    if not finite_values:
        stdscr.addnstr(y, x, "Latest vector rows do not contain finite values.", width)
        return

    min_value = min(min(finite_values), 0.0)
    max_value = max(max(finite_values), 0.0)
    if math.isclose(min_value, max_value):
        min_value -= 1.0
        max_value += 1.0

    grid = [[" " for _ in range(width)] for _ in range(height)]
    zero_row = int(round((max_value - 0.0) / (max_value - min_value) * (height - 1)))
    zero_row = max(0, min(height - 1, zero_row))
    for col in range(width):
        grid[zero_row][col] = "-"

    for name, values in series.items():
        offset = width - len(values)
        for index, value in enumerate(values):
            if not math.isfinite(value):
                continue
            row_index = int(round((max_value - value) / (max_value - min_value) * (height - 1)))
            row_index = max(0, min(height - 1, row_index))
            col_index = offset + index
            current = grid[row_index][col_index]
            grid[row_index][col_index] = name if current in {" ", "-"} else "*"

    for row_offset, line in enumerate(grid):
        stdscr.addnstr(y + row_offset, x, "".join(line), width)


def _render_operator(stdscr: curses.window, config: OperatorConfig, snapshot: LiveMonitorSnapshot, paused: bool) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    if height < 24 or width < 90:
        stdscr.addnstr(0, 0, "Terminal too small for live mode. Resize to at least 90x24.", max(width - 1, 0), curses.A_BOLD)
        stdscr.refresh()
        return

    header = f"NV Magnetometry Live Console | full scan: {config.full_scan_path.name} | parked: {config.parked_data_path.name}"
    stdscr.addnstr(0, 0, header.ljust(width - 1), width - 1, curses.A_BOLD)
    status_line = (
        f"status={snapshot.status}  samples={snapshot.sample_count}  rank-ok={snapshot.ok_count}  "
        f"poll={config.poll_interval_s:.1f}s  mode={'paused' if paused else 'live'}"
    )
    stdscr.addnstr(1, 0, status_line.ljust(width - 1), width - 1)

    left_width = max(34, width // 3)
    right_width = width - left_width - 2
    top_height = 12
    bottom_height = height - top_height - 4

    _draw_bordered_panel(stdscr, 3, 0, top_height, left_width, "Current Field")
    _draw_bordered_panel(stdscr, 3, left_width + 2, top_height, right_width, "Tracking Plot")
    _draw_bordered_panel(stdscr, top_height + 4, 0, bottom_height, width - 1, "Plan And Status")

    field_lines = [
        f"Bias field: {_format_vector(config.bias_field_mT, precision=4, unit='mT')}",
        f"Latest timestamp: {_format_optional_float(snapshot.latest_timestamp_s, precision=3, unit='s')}",
        f"Latest local dB: {_format_vector(snapshot.latest_local_uT, precision=2) if snapshot.latest_local_uT else 'n/a'}",
        f"Latest total B: {_format_vector(snapshot.latest_total_uT, precision=2) if snapshot.latest_total_uT else 'n/a'}",
        f"Residual: {_format_optional_float(snapshot.latest_residual_uT, precision=2, unit='uT')}",
        f"Rank: {snapshot.latest_rank if snapshot.latest_rank is not None else 'n/a'}",
        f"Channels: {snapshot.latest_channels if snapshot.latest_channels is not None else 'n/a'}",
    ]
    _add_lines(stdscr, field_lines, 4, 2, left_width - 4, top_height - 2)

    _draw_ascii_plot(stdscr, snapshot.vector_rows, 4, left_width + 4, top_height - 3, right_width - 4)
    stdscr.addnstr(3 + top_height - 2, left_width + 4, "legend: x=Bx  y=By  z=Bz  -=0", right_width - 4)

    plan_lines = [snapshot.message, ""]
    for block_index, entry in enumerate(config.parked_plan, start=1):
        plan_lines.append(
            f"block {block_index:02d}: {entry.frequency_minus_mhz:9.3f}, {entry.frequency_plus_mhz:9.3f} MHz"
            f"  center={entry.center_mhz:9.3f}"
        )
    plan_lines.extend(
        [
            "",
            f"plan csv: {config.plan_csv_path}",
            f"wide template: {config.wide_template_path}",
            f"long template: {config.long_template_path}",
        ]
    )
    _add_lines(stdscr, plan_lines, top_height + 5, 2, width - 5, bottom_height - 2)

    footer = "q: quit  p: pause/resume polling  r: force refresh"
    stdscr.addnstr(height - 1, 0, footer.ljust(width - 1), width - 1, curses.A_DIM)
    stdscr.refresh()


def _run_live_operator(stdscr: curses.window, config: OperatorConfig) -> None:
    curses.curs_set(0)
    stdscr.keypad(True)
    stdscr.timeout(200)

    paused = False
    snapshot = _compute_live_snapshot(config)
    last_refresh = time.monotonic()

    while True:
        if not paused and time.monotonic() - last_refresh >= config.poll_interval_s:
            snapshot = _compute_live_snapshot(config)
            last_refresh = time.monotonic()

        _render_operator(stdscr, config, snapshot, paused)
        key = stdscr.getch()
        if key == -1:
            continue
        if key in (ord("q"), ord("Q")):
            return
        if key in (ord("p"), ord("P")):
            paused = not paused
        elif key in (ord("r"), ord("R")):
            snapshot = _compute_live_snapshot(config)
            last_refresh = time.monotonic()


def _run_popup_operator(config: OperatorConfig) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_DIR / ".matplotlib-cache"))
    import matplotlib.pyplot as plt

    plt.ion()
    fig, (ax_current, ax_trace) = plt.subplots(1, 2, figsize=(12.5, 5.8))
    fig.canvas.manager.set_window_title("NV Magnetometry Live Operator")

    last_refresh = 0.0
    latest_snapshot = _compute_live_snapshot(config)

    while plt.fignum_exists(fig.number):
        now = time.monotonic()
        if now - last_refresh >= config.poll_interval_s:
            latest_snapshot = _compute_live_snapshot(config)
            last_refresh = now

            ax_current.clear()
            ax_trace.clear()

            if latest_snapshot.latest_local_uT is not None:
                components = ["Bx", "By", "Bz"]
                local_values = np.asarray(latest_snapshot.latest_local_uT, dtype=float)
                total_values = np.asarray(latest_snapshot.latest_total_uT, dtype=float) if latest_snapshot.latest_total_uT is not None else local_values
                positions = np.arange(len(components), dtype=float)
                ax_current.bar(positions - 0.18, local_values, width=0.36, label="local dB (uT)", color="#1f5b70")
                ax_current.bar(positions + 0.18, total_values, width=0.36, label="total B (uT)", color="#b23a2e")
                ax_current.set_xticks(positions, components)
                ax_current.axhline(0.0, color="#555555", linewidth=0.8)
                ax_current.set_title("Current Field")
                ax_current.set_ylabel("uT")
                ax_current.legend(loc="best")
            else:
                ax_current.text(0.5, 0.5, latest_snapshot.message, ha="center", va="center", transform=ax_current.transAxes)
                ax_current.set_title("Current Field")
                ax_current.set_xticks([])
                ax_current.set_yticks([])

            ok_rows = [row for row in latest_snapshot.vector_rows if str(row.get("status", "")) == "ok"]
            if ok_rows:
                timestamps = np.asarray([_safe_float(row, "timestamp_epoch_s") for row in ok_rows], dtype=float)
                t0 = float(timestamps[0])
                rel_t = timestamps - t0
                bx = np.asarray([_safe_float(row, "delta_Bx_uT") for row in ok_rows], dtype=float)
                by = np.asarray([_safe_float(row, "delta_By_uT") for row in ok_rows], dtype=float)
                bz = np.asarray([_safe_float(row, "delta_Bz_uT") for row in ok_rows], dtype=float)
                ax_trace.plot(rel_t, bx, label="Bx", color="#0f7173", linewidth=1.5)
                ax_trace.plot(rel_t, by, label="By", color="#d16b00", linewidth=1.5)
                ax_trace.plot(rel_t, bz, label="Bz", color="#7a3e9d", linewidth=1.5)
                ax_trace.axhline(0.0, color="#555555", linewidth=0.8)
                ax_trace.set_title("Live Local Field Tracking")
                ax_trace.set_xlabel("time since first sample (s)")
                ax_trace.set_ylabel("uT")
                ax_trace.legend(loc="best")
            else:
                ax_trace.text(0.5, 0.5, latest_snapshot.message, ha="center", va="center", transform=ax_trace.transAxes)
                ax_trace.set_title("Live Local Field Tracking")
                ax_trace.set_xticks([])
                ax_trace.set_yticks([])

            latest_local_text = _format_vector(latest_snapshot.latest_local_uT, precision=2) if latest_snapshot.latest_local_uT is not None else "n/a"
            fig.suptitle(
                "NV Magnetometry Live Operator\n"
                f"status={latest_snapshot.status}  samples={latest_snapshot.sample_count}  "
                f"rank-ok={latest_snapshot.ok_count}  latest_local={latest_local_text}",
                fontsize=11,
            )
            fig.tight_layout()

        plt.pause(0.05)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Terminal workflow viewer and live operator console for NV magnetometry.")
    parser.add_argument(
        "--demo",
        choices=("april17-real", "april29-lockin", "synthetic-phase0"),
        default="april17-real",
        help="Which committed workflow artifact to open first in demo mode.",
    )
    parser.add_argument(
        "--list-demos",
        action="store_true",
        help="Print the available demos and exit.",
    )
    parser.add_argument(
        "--operator",
        action="store_true",
        help="Start the full operator console: ingest a full ODMR file, propose parked frequencies, then poll a live parked-data CSV.",
    )
    parser.add_argument("--full-scan", type=Path, default=None, help="Full ODMR spectrum CSV for operator mode.")
    parser.add_argument("--parked-data", type=Path, default=None, help="Live parked-data CSV to poll in operator mode.")
    parser.add_argument("--input-format", choices=("auto", "raw", "toolkit"), default="auto", help="How to interpret the full-scan CSV.")
    parser.add_argument("--parked-format", choices=("auto", "wide", "long", "toolkit"), default="auto", help="How to interpret the live parked-data CSV.")
    parser.add_argument("--bias-field", default="", help="Optional bias-field override as Bx,By,Bz in mT or a path to a base-field CSV.")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Live parked-data polling interval in seconds.")
    parser.add_argument("--popup", action="store_true", help="In operator mode, open a real popup plot window instead of the curses console.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    demos = load_demo_catalog()

    if args.list_demos:
        for demo in demos:
            print(f"{demo.key}: {demo.title}")
        return 0

    if args.operator:
        config = _prepare_operator_config(args)
        if args.popup:
            _run_popup_operator(config)
        else:
            _require_curses().wrapper(_run_live_operator, config)
        return 0

    _require_curses().wrapper(_run_demo_tui, demos, args.demo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
