#!/usr/bin/env python
"""CLI for blind NV axis + local B field reconstruction (Variant A).

Usage:
    python -m nv_toolkit.blind_reconstruct manifest.json

Manifest format (JSON):
{
  "spectra": [
    {"csv": "spec1.csv", "B_applied_mT": [1.0, 0.5, 0.0]},
    {"csv": "spec2.csv", "B_applied_mT": [0.0, 1.2, 0.0]},
    {"csv": "spec3.csv", "B_applied_mT": [0.0, 0.0, 1.5]},
    {"csv": "spec4.csv", "B_applied_mT": [0.5, 0.0, 1.0]}
  ],
  "num_restarts": 50,
  "output_json": "result.json"
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .blind_fit import extract_splittings_from_spectrum_csv, fit_blind


def _parse_manifest(path: str) -> dict:
    with open(path, "r") as f:
        manifest = json.load(f)

    if "spectra" not in manifest or len(manifest["spectra"]) < 3:
        raise ValueError("Manifest must contain 'spectra' list with at least 3 entries")

    applied_fields_tesla = []
    observed_splittings_ghz = []
    for entry in manifest["spectra"]:
        b_mT = np.asarray(entry["B_applied_mT"], dtype=float)
        applied_fields_tesla.append(b_mT * 1e-3)  # mT → T

        csv_path = entry["csv"]
        splits_ghz = extract_splittings_from_spectrum_csv(csv_path)
        observed_splittings_ghz.append(splits_ghz)

        print(f"  {csv_path}: splits = {np.round(splits_ghz * 1e3, 1)} MHz  | B_applied = {b_mT} mT")

    return {
        "applied_fields_tesla": applied_fields_tesla,
        "observed_splittings_ghz": observed_splittings_ghz,
        "num_restarts": manifest.get("num_restarts", 50),
        "output_json": manifest.get("output_json"),
    }


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Blind NV reconstruction: fit local B field + axis orientation from 4+ spectra."
    )
    parser.add_argument("manifest", type=str, help="JSON manifest file")
    parser.add_argument("--num-restarts", type=int, default=50, help="Random restarts (default: 50)")
    parser.add_argument("--output-json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args(argv)

    data = _parse_manifest(args.manifest)
    if args.num_restarts is not None:
        data["num_restarts"] = args.num_restarts

    print(f"\nFitting with {len(data['applied_fields_tesla'])} spectra, "
          f"{data['num_restarts']} restarts...\n")

    result = fit_blind(
        data["applied_fields_tesla"],
        data["observed_splittings_ghz"],
        num_restarts=data["num_restarts"],
    )

    B_loc = result["b_local_uT"]
    axes = result["axes"]

    print(f"  Local B field:  ({B_loc[0]:.1f}, {B_loc[1]:.1f}, {B_loc[2]:.1f}) µT")
    print(f"  |B_local|:      {np.linalg.norm(B_loc):.1f} µT")
    print(f"  Theta:          {np.degrees(result['theta']):.1f}°")
    print(f"  Phi:            {np.degrees(result['phi']):.1f}°")
    print(f"  Alpha:          {np.degrees(result['alpha']):.1f}°")
    print(f"  Residual RMS:   {result['residual_rms_ghz'] * 1e3:.3f} MHz")
    print(f"  Success:        {result['success']}")
    print(f"\n  Recovered axes:")
    for i in range(4):
        ax = axes[i]
        print(f"    axis {i+1}: ({ax[0]:.6f}, {ax[1]:.6f}, {ax[2]:.6f})")

    output_path = args.output_json or data.get("output_json")
    if output_path:
        serializable = {
            "b_local_uT": result["b_local_uT"].tolist(),
            "b_local_tesla": result["b_local_tesla"].tolist(),
            "theta_deg": float(np.degrees(result["theta"])),
            "phi_deg": float(np.degrees(result["phi"])),
            "alpha_deg": float(np.degrees(result["alpha"])),
            "axes": axes.tolist(),
            "residual_rms_ghz": float(result["residual_rms_ghz"]),
            "success": result["success"],
            "message": result["message"],
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
