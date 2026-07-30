#!/usr/bin/env python3
"""Structural and regime checks for an unmodified-WRF RUC step oracle."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


CASES = ("warm_rain", "cold_snow", "water", "sea_ice")


def main(path: str) -> None:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 36:
        raise SystemExit(f"expected 36 rows, got {len(rows)}")
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    if cases != CASES:
        raise SystemExit(f"unexpected case inventory {cases!r}")
    numeric = tuple(key for key in rows[0] if key not in ("case", "k"))
    values = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(4, 9)
        for name in numeric
    }
    if not all(np.all(np.isfinite(field)) for field in values.values()):
        raise SystemExit("RUC step oracle contains non-finite output")

    warm, cold, water, ice = range(4)
    if np.array_equal(values["tso_after"][warm], values["tso_before"][warm]):
        raise SystemExit("warm-rain soil temperature did not evolve")
    if not (values["sfcrunoff"][warm, 0] > 0.0 and values["qfx"][warm, 0] > 0.0):
        raise SystemExit("warm-rain water-flux branch did not execute")
    if not (
        values["snow"][cold, 0] > 20.0
        and values["precipfr"][cold, 0] > 0.0
        and values["snowfallac"][cold, 0] > 0.0
    ):
        raise SystemExit("cold-snow accumulation branch did not execute")
    for name in ("tso", "soilmois", "sh2o", "smfr"):
        before = values[f"{name}_before"][water]
        after = values[f"{name}_after"][water]
        if not np.array_equal(before.view(np.uint32), after.view(np.uint32)):
            raise SystemExit(f"water branch unexpectedly changed {name}")
    if not (
        np.all(values["sh2o_after"][ice] == np.float32(0.0))
        and np.all(values["smfr_after"][ice] == np.float32(1.0))
    ):
        raise SystemExit("sea-ice frozen-state branch did not execute")
    print("RUC full-step oracle: PASS (four finite discriminating regimes)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_step_oracle.py RUC_STEP.csv")
    main(sys.argv[1])
