from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nv_toolkit.model import named_nv_axes, odmr_resonances


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_JSON = Path("data/results/official/8_peaks_8_stacks_fit.json")


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_DIR / candidate


def _nv_axes_preset(summary: dict[str, Any]) -> str:
    peak_inversion = summary.get("peak_inversion", {})
    if isinstance(peak_inversion, dict):
        preset = peak_inversion.get("nv_axes_preset")
        if isinstance(preset, str):
            try:
                named_nv_axes(preset)
                return preset
            except ValueError:
                pass
    return "qdm"


def _transition_centers(summary: dict[str, Any]) -> np.ndarray:
    centers = np.asarray(summary.get("detected_transition_centers_mhz", []), dtype=float)
    if centers.size < 8:
        peak_inversion = summary.get("peak_inversion", {})
        if isinstance(peak_inversion, dict):
            centers = np.asarray(peak_inversion.get("transition_centers_mhz", []), dtype=float)
    centers = np.sort(centers[np.isfinite(centers)])
    if centers.size < 8:
        raise ValueError(
            "The reference summary must contain at least 8 transition centers "
            f"for the 8-peak parked-frequency workflow; found {centers.size}."
        )
    return centers[:8]


def _best_fit(summary: dict[str, Any]) -> dict[str, Any]:
    best = summary.get("best_fit")
    if not isinstance(best, dict):
        raise ValueError("The reference summary does not contain a best_fit block.")
    return best


def ensure_reference_outputs(reference_json: str | Path | None = None) -> dict[str, Any]:
    """Load the committed 8-peak reference fit used by the notebook workflow.

    The historical notebook expected a phase1 helper script to prepare this
    summary. In this repository the fit outputs are already committed, so this
    function validates and returns that reference data without regenerating it.
    """

    path = _resolve_repo_path(reference_json or DEFAULT_REFERENCE_JSON)
    if not path.exists():
        raise FileNotFoundError(f"Reference fit JSON not found: {path}")

    summary = json.loads(path.read_text(encoding="utf-8"))
    best = _best_fit(summary)
    centers = _transition_centers(summary)
    b_total_mT = np.asarray(best.get("B_total_mT", []), dtype=float)
    if b_total_mT.shape != (3,):
        raise ValueError("The reference best_fit block must contain a 3-vector B_total_mT.")

    preset = _nv_axes_preset(summary)
    modeled = odmr_resonances(b_total_mT, nv_axes=named_nv_axes(preset))

    summary["_reference_json"] = str(path)
    summary["_nv_axes_preset"] = preset
    summary["_transition_centers_mhz"] = centers.tolist()
    summary["_modeled_resonances_mhz"] = modeled.tolist()
    return summary


def build_multipeak_channel_table(reference_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the 8 transition rows consumed by the parked-frequency notebook."""

    summary = dict(reference_summary)
    best = _best_fit(summary)
    centers = _transition_centers(summary)

    b_total_mT = np.asarray(best.get("B_total_mT", []), dtype=float)
    preset = str(summary.get("_nv_axes_preset") or _nv_axes_preset(summary))
    modeled = np.asarray(
        summary.get("_modeled_resonances_mhz")
        or odmr_resonances(b_total_mT, nv_axes=named_nv_axes(preset)),
        dtype=float,
    )
    linewidths = np.asarray(best.get("linewidths_mhz", 4.0), dtype=float)
    linewidths = np.broadcast_to(linewidths, modeled.shape)

    rows: list[dict[str, Any]] = []
    used_model_indices: set[tuple[int, int]] = set()
    for block_index, center in enumerate(centers, start=1):
        distances = np.abs(modeled - float(center))
        for used in used_model_indices:
            distances[used] = np.inf
        if not np.isfinite(distances).any():
            distances = np.abs(modeled - float(center))

        axis_index, side_index = np.unravel_index(int(np.argmin(distances)), modeled.shape)
        used_model_indices.add((int(axis_index), int(side_index)))

        rows.append(
            {
                "block_index": int(block_index),
                "transition_center_mhz": float(center),
                "modeled_center_mhz": float(modeled[axis_index, side_index]),
                "axis_index": int(axis_index),
                "transition_side": -1 if int(side_index) == 0 else 1,
                "transition_side_index": int(side_index),
                "linewidth_mhz": float(linewidths[axis_index, side_index]),
            }
        )

    return rows
