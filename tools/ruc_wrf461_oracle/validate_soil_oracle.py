#!/usr/bin/env python3
"""Structural checks for the unmodified-WRF snow-free RUC ``soil`` step."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


CASES = ("warm_forest_rain", "dry_grass", "frozen_crop_rain", "humid_bare_dew")


def _load_profiles(
    path: str, expected_cases: tuple[str, ...], *,
    shape: tuple[int, int] = (4, 9), excluded: tuple[str, ...] = ("case", "k"),
) -> dict[str, np.ndarray]:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    expected_rows = shape[0] * shape[1]
    if len(rows) != expected_rows:
        raise SystemExit(f"expected {expected_rows} rows, got {len(rows)}")
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    if cases != expected_cases:
        raise SystemExit(f"unexpected case inventory {cases!r}")
    numeric = tuple(key for key in rows[0] if key not in excluded)
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(shape)
        for name in numeric
    }


def main(path: str) -> None:
    values = _load_profiles(path, CASES)
    if not all(np.all(np.isfinite(field)) for field in values.values()):
        raise SystemExit("RUC soil oracle contains non-finite output")
    if not np.array_equal(values["nroot"][:, 0], [8.0, 6.0, 6.0, 6.0]):
        raise SystemExit("RUC soil oracle lost variable root depths")
    if not all(np.any(values["tso_after"][case] != values["tso_before"][case])
               for case in range(4)):
        raise SystemExit("RUC soil oracle failed to evolve temperature")
    if not np.any(values["soilmois_after"][0] != values["soilmois_before"][0]):
        raise SystemExit("warm-rain water state did not evolve")
    if not values["qfx"][1, 0] > 0.0:
        raise SystemExit("dry evaporation branch did not produce upward latent heat")
    if not np.any(values["soilice"][2] > 0.0):
        raise SystemExit("frozen-soil branch did not retain ice")
    if not values["dew"][3, 0] > 0.0:
        raise SystemExit("humid condensation branch did not produce dew")
    print("RUC soil oracle: PASS (four evolving snow-free land regimes)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_soil_oracle.py RUC_SOIL.csv")
    main(sys.argv[1])
