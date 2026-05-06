from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BaseFieldResolution:
    vector_mT: np.ndarray
    source: str
    metadata: dict[str, str]


def _normalized(text: str) -> str:
    return "".join(ch for ch in text.strip().lower() if ch.isalnum())


def _pick_key(row: dict[str, str], *candidates: str) -> str | None:
    normalized = {_normalized(key): key for key in row.keys()}
    for candidate in candidates:
        match = normalized.get(_normalized(candidate))
        if match is not None:
            return match
    return None


def _unit_scale_from_key(key: str, *, default_unit: str) -> float:
    token = _normalized(key)
    if "mtpera" in token or token.endswith("mt") or "millitesla" in token:
        return 1.0
    if "utpera" in token or token.endswith("ut") or "microtesla" in token:
        return 1e-3
    if default_unit.lower() == "mt":
        return 1.0
    return 1e-3


def parse_base_field_vector(value: str, *, unit: str = "uT") -> BaseFieldResolution:
    tokens = [item for item in value.replace(";", ",").replace(" ", ",").split(",") if item]
    if len(tokens) != 3:
        raise ValueError("Base field vector text must contain exactly three comma- or space-separated values")
    vector = np.asarray([float(item) for item in tokens], dtype=float)
    scale = 1.0 if unit.lower() == "mt" else 1e-3
    return BaseFieldResolution(
        vector_mT=vector * scale,
        source="inline_vector",
        metadata={"unit": unit},
    )


def load_base_field_csv(path: str | Path, *, default_unit: str = "uT") -> BaseFieldResolution:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path} is empty")
    row = rows[0]

    bx_key = _pick_key(row, "Bx_uT", "Bx_mT", "bx", "B_x", "field_x")
    by_key = _pick_key(row, "By_uT", "By_mT", "by", "B_y", "field_y")
    bz_key = _pick_key(row, "Bz_uT", "Bz_mT", "bz", "B_z", "field_z")
    if bx_key and by_key and bz_key:
        vector = np.asarray(
            [
                float(row[bx_key]) * _unit_scale_from_key(bx_key, default_unit=default_unit),
                float(row[by_key]) * _unit_scale_from_key(by_key, default_unit=default_unit),
                float(row[bz_key]) * _unit_scale_from_key(bz_key, default_unit=default_unit),
            ],
            dtype=float,
        )
        return BaseFieldResolution(
            vector_mT=vector,
            source=str(path),
            metadata={"mode": "direct_vector"},
        )

    ix_key = _pick_key(row, "Ix_A", "current_x_a", "ix")
    iy_key = _pick_key(row, "Iy_A", "current_y_a", "iy")
    iz_key = _pick_key(row, "Iz_A", "current_z_a", "iz")
    if ix_key and iy_key and iz_key:
        currents = np.asarray([float(row[ix_key]), float(row[iy_key]), float(row[iz_key])], dtype=float)
        matrix_keys = [
            [
                _pick_key(row, "dBx_dIx_uT_per_A", "dBx_dIx_mT_per_A"),
                _pick_key(row, "dBx_dIy_uT_per_A", "dBx_dIy_mT_per_A"),
                _pick_key(row, "dBx_dIz_uT_per_A", "dBx_dIz_mT_per_A"),
            ],
            [
                _pick_key(row, "dBy_dIx_uT_per_A", "dBy_dIx_mT_per_A"),
                _pick_key(row, "dBy_dIy_uT_per_A", "dBy_dIy_mT_per_A"),
                _pick_key(row, "dBy_dIz_uT_per_A", "dBy_dIz_mT_per_A"),
            ],
            [
                _pick_key(row, "dBz_dIx_uT_per_A", "dBz_dIx_mT_per_A"),
                _pick_key(row, "dBz_dIy_uT_per_A", "dBz_dIy_mT_per_A"),
                _pick_key(row, "dBz_dIz_uT_per_A", "dBz_dIz_mT_per_A"),
            ],
        ]
        if all(all(item is not None for item in row_keys) for row_keys in matrix_keys):
            matrix = np.asarray(
                [
                    [
                        float(row[matrix_keys[i][j]])
                        * _unit_scale_from_key(matrix_keys[i][j], default_unit=default_unit)
                        for j in range(3)
                    ]
                    for i in range(3)
                ],
                dtype=float,
            )
            return BaseFieldResolution(
                vector_mT=matrix @ currents,
                source=str(path),
                metadata={"mode": "current_times_full_matrix"},
            )

        diagonal = [
            _pick_key(row, "Bx_per_Ix_uT_per_A", "Bx_per_Ix_mT_per_A", "dBx_dIx_uT_per_A", "dBx_dIx_mT_per_A"),
            _pick_key(row, "By_per_Iy_uT_per_A", "By_per_Iy_mT_per_A", "dBy_dIy_uT_per_A", "dBy_dIy_mT_per_A"),
            _pick_key(row, "Bz_per_Iz_uT_per_A", "Bz_per_Iz_mT_per_A", "dBz_dIz_uT_per_A", "dBz_dIz_mT_per_A"),
        ]
        if all(diagonal):
            slopes = np.asarray(
                [
                    float(row[diagonal[0]]) * _unit_scale_from_key(diagonal[0], default_unit=default_unit),
                    float(row[diagonal[1]]) * _unit_scale_from_key(diagonal[1], default_unit=default_unit),
                    float(row[diagonal[2]]) * _unit_scale_from_key(diagonal[2], default_unit=default_unit),
                ],
                dtype=float,
            )
            return BaseFieldResolution(
                vector_mT=currents * slopes,
                source=str(path),
                metadata={"mode": "current_times_diagonal_slopes"},
            )

    raise ValueError(
        f"{path} does not match a supported base-field CSV format. "
        "Use direct Bx/By/Bz columns or currents plus slope columns."
    )


def resolve_base_field(value: str, *, default_unit: str = "uT") -> BaseFieldResolution:
    path = Path(value)
    if path.exists():
        return load_base_field_csv(path, default_unit=default_unit)
    return parse_base_field_vector(value, unit=default_unit)
