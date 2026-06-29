"""Compact command-line entrypoint for the FPGA parked-frequency workflow.

This file is intentionally the top-level control surface for collaborators who
want to read the Python rather than use the browser UI. It does not plot or open
windows. It only writes CSVs.

Current April 29 data path:
    phase1/data/raw_scan_exports_20260429_lockin/8_peaks_8_stacks.csv

Typical outputs:
    phase1/operator_test_fixture/parked_plan_16freq.csv
    phase1/results/operator_vector_rows.csv
    phase1/results/operator_projection_rows.csv

High-level wiring:
    plan:
        full ODMR CSV
        -> _load_operator_full_scan
        -> _suggest_parked_frequencies
        -> _write_parked_plan_csv

    reconstruct:
        full ODMR CSV + parked plan CSV + collected parked-data CSV
        -> _load_operator_full_scan
        -> _read_parked_plan_csv
        -> _compute_live_snapshot
        -> vector/projection CSVs

Engineering rule for collaborators:
    Code changes are expected, but keep this file as a small orchestration layer.
    Put algorithm changes in the underlying API modules, add/update tests, and
    push changes through GitHub so the workflow remains reproducible.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np

# These are example/default paths for the current dataset. They are deliberately
# not used as hidden defaults in argparse: production calls should pass paths
# explicitly so scripts and FPGA handoffs are reproducible.
CURRENT_FULL_SCAN_CSV = Path("phase1/data/raw_scan_exports_20260429_lockin/8_peaks_8_stacks.csv")
CURRENT_PLAN_CSV = Path("phase1/operator_test_fixture/parked_plan_16freq.csv")
CURRENT_VECTOR_OUTPUT_CSV = Path("phase1/results/operator_vector_rows.csv")
CURRENT_PROJECTION_OUTPUT_CSV = Path("phase1/results/operator_projection_rows.csv")

from .tui import (
    ParkedFrequencyPlanEntry,
    _compute_live_snapshot,
    _estimate_bias_field_mT,
    _load_operator_full_scan,
    _resolve_bias_input,
    _suggest_parked_frequencies,
    _write_parked_plan_csv,
    OperatorConfig,
)


APRIL29_BIAS_FALLBACKS_MT = {
    "8_peaks_8_stacks.csv": np.asarray([0.0, 0.0, -4.611225501956753], dtype=float),
    "full_scan_8_peaks_8_stacks_toolkit.csv": np.asarray([0.0, 0.0, -4.611225501956753], dtype=float),
    "8_peaks_8_stacks_new_tdy.csv": np.asarray([0.0, 0.0, -4.651440416301873], dtype=float),
    "full_scan_8_peaks_8_stacks_new_tdy_toolkit.csv": np.asarray([0.0, 0.0, -4.651440416301873], dtype=float),
}


def _estimate_bias_with_fallback(
    full_scan_path: Path,
    freqs_mhz: np.ndarray,
    measured: np.ndarray,
    transition_centers: np.ndarray,
) -> np.ndarray:
    """Estimate the bias field, using known April 29 fallbacks only if fitting fails."""
    try:
        return np.asarray(_estimate_bias_field_mT(freqs_mhz, measured, transition_centers), dtype=float)
    except Exception:
        fallback = APRIL29_BIAS_FALLBACKS_MT.get(full_scan_path.name)
        if fallback is None:
            raise
        return np.asarray(fallback, dtype=float)


def _read_parked_plan_csv(path: Path) -> tuple[ParkedFrequencyPlanEntry, ...]:
    """Load the 16-row plan CSV that was handed to FPGA/acquisition."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No parked-frequency rows found in {path}")

    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["block_index"]), []).append(row)

    plan: list[ParkedFrequencyPlanEntry] = []
    for block_index in sorted(grouped):
        block_rows = sorted(grouped[block_index], key=lambda item: int(item["point_in_block"]))
        if len(block_rows) != 2:
            raise ValueError(f"Block {block_index} in {path} must contain exactly two parked frequencies")
        minus, plus = block_rows
        plan.append(
            ParkedFrequencyPlanEntry(
                transition_index=int(minus["transition_index"]),
                center_mhz=float(minus["transition_center_mhz"]),
                linewidth_mhz=float(minus["linewidth_mhz"]),
                frequency_minus_mhz=float(minus["frequency_mhz"]),
                frequency_plus_mhz=float(plus["frequency_mhz"]),
                slope_minus_per_mhz=float(minus["slope_per_mhz"]),
                slope_plus_per_mhz=float(plus["slope_per_mhz"]),
            )
        )
    return tuple(plan)


def _write_rows_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write list-of-dict API rows to CSV without changing field names."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_summary_csv(path: Path, *, full_scan: Path, plan: Sequence[ParkedFrequencyPlanEntry], bias_mT: np.ndarray) -> None:
    """Write optional one-row metadata for humans; not required by acquisition."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "full_scan_path",
                "n_blocks",
                "n_parked_frequencies",
                "bias_Bx_mT",
                "bias_By_mT",
                "bias_Bz_mT",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "full_scan_path": str(full_scan),
                "n_blocks": len(plan),
                "n_parked_frequencies": 2 * len(plan),
                "bias_Bx_mT": float(bias_mT[0]),
                "bias_By_mT": float(bias_mT[1]),
                "bias_Bz_mT": float(bias_mT[2]),
            }
        )


def _cmd_plan(args: argparse.Namespace) -> int:
    """ODMR spectrum -> 16 parked frequencies CSV.

    This is the command FPGA/acquisition should run before data collection.
    The output CSV is the contract: acquire intensities in point_index order.
    """
    full_scan = args.full_scan.expanduser().resolve()
    output_plan = args.output.expanduser().resolve()
    freqs_mhz, measured, transition_centers = _load_operator_full_scan(full_scan, args.input_format)
    plan = _suggest_parked_frequencies(
        freqs_mhz,
        measured,
        transition_centers,
        placement=args.placement,
    )
    bias_mT = _estimate_bias_with_fallback(full_scan, freqs_mhz, measured, transition_centers)

    _write_parked_plan_csv(output_plan, plan)
    if args.summary is not None:
        _write_summary_csv(args.summary.expanduser().resolve(), full_scan=full_scan, plan=plan, bias_mT=bias_mT)

    print(f"wrote parked plan: {output_plan}")
    print(f"parked frequencies: {2 * len(plan)}")
    print(f"estimated bias mT: {bias_mT[0]:.6f},{bias_mT[1]:.6f},{bias_mT[2]:.6f}")
    return 0


def _cmd_reconstruct(args: argparse.Namespace) -> int:
    """Collected parked intensities -> reconstructed B-field CSVs.

    This reuses the same plan CSV that was handed to acquisition, so the B-field
    solve is tied to the exact parked frequencies that were measured.
    """
    full_scan = args.full_scan.expanduser().resolve()
    parked_data = args.parked_data.expanduser().resolve()
    plan_csv = args.plan.expanduser().resolve()
    output_vectors = args.output.expanduser().resolve()

    freqs_mhz, measured, transition_centers = _load_operator_full_scan(full_scan, args.input_format)
    estimated_bias_mT = _estimate_bias_with_fallback(full_scan, freqs_mhz, measured, transition_centers)
    bias_field_mT = _resolve_bias_input(args.bias_override, estimated_bias_mT)
    parked_plan = _read_parked_plan_csv(plan_csv)

    config = OperatorConfig(
        full_scan_path=full_scan,
        parked_data_path=parked_data,
        parked_format=args.parked_format,
        bias_field_mT=np.asarray(bias_field_mT, dtype=float),
        reference_freqs_mhz=np.asarray(freqs_mhz, dtype=float),
        reference_spectrum=np.asarray(measured, dtype=float),
        transition_centers_mhz=np.asarray(transition_centers, dtype=float),
        parked_plan=tuple(parked_plan),
        plan_csv_path=plan_csv,
        wide_template_path=plan_csv.with_name("parked_template_wide.csv"),
        long_template_path=plan_csv.with_name("parked_template_long.csv"),
        poll_interval_s=0.0,
    )
    snapshot = _compute_live_snapshot(config)
    if snapshot.status == "error":
        raise ValueError(snapshot.message)

    _write_rows_csv(output_vectors, snapshot.vector_rows)
    if args.projections_output is not None:
        _write_rows_csv(args.projections_output.expanduser().resolve(), snapshot.projection_rows)

    print(f"status: {snapshot.status}")
    print(f"samples: {snapshot.sample_count}")
    print(f"rank-ok samples: {snapshot.ok_count}")
    print(f"wrote vector rows: {output_vectors}")
    if args.projections_output is not None:
        print(f"wrote projection rows: {args.projections_output.expanduser().resolve()}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the small command surface; algorithmic knobs live in API modules."""
    parser = argparse.ArgumentParser(
        description="Non-interactive operator commands for FPGA parked-frequency workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Input a full ODMR spectrum and output the 16 parked frequencies CSV for FPGA acquisition.",
    )
    plan_parser.add_argument("--full-scan", type=Path, required=True, help="Input full ODMR spectrum CSV.")
    plan_parser.add_argument("--output", type=Path, required=True, help="Output parked-frequency plan CSV.")
    plan_parser.add_argument("--input-format", choices=("auto", "raw", "toolkit"), default="auto")
    plan_parser.add_argument("--summary", type=Path, default=None, help="Optional one-row summary CSV with estimated bias.")
    plan_parser.add_argument(
        "--placement",
        choices=("auto", "empirical_half_max", "empirical", "max_slope", "half_max"),
        default="auto",
        help="Lock-in parking placement: auto (default, contrast-based hybrid), empirical_half_max, empirical, max_slope, or half_max.",
    )
    plan_parser.set_defaults(func=_cmd_plan)

    reconstruct_parser = subparsers.add_parser(
        "reconstruct",
        help="Input collected parked data and output reconstructed B-field rows.",
    )
    reconstruct_parser.add_argument("--full-scan", type=Path, required=True, help="The same reference full ODMR spectrum CSV.")
    reconstruct_parser.add_argument("--plan", type=Path, required=True, help="Parked-frequency plan CSV handed to acquisition.")
    reconstruct_parser.add_argument("--parked-data", type=Path, required=True, help="Collected parked-intensity CSV.")
    reconstruct_parser.add_argument("--output", type=Path, required=True, help="Output reconstructed B-field vector CSV.")
    reconstruct_parser.add_argument("--projections-output", type=Path, default=None, help="Optional per-block projection CSV.")
    reconstruct_parser.add_argument("--input-format", choices=("auto", "raw", "toolkit"), default="auto")
    reconstruct_parser.add_argument("--parked-format", choices=("auto", "wide", "long", "toolkit"), default="auto")
    reconstruct_parser.add_argument(
        "--bias-override",
        default="",
        help="Optional bias field as Bx,By,Bz in mT, or a base-field CSV path. Empty uses estimated bias.",
    )
    reconstruct_parser.set_defaults(func=_cmd_reconstruct)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
