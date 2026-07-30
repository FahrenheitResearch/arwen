#!/usr/bin/env python3
"""Structural and branch checks for unmodified-WRF RUC ``soilvegin``."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


CASES = (
    "evergreen_cold",
    "evergreen_warm",
    "crop_midseason",
    "water_preserve_znt",
    "lai2d_preserve",
    "grass_short_season",
)


def main(path: str) -> None:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    cases = tuple(row["case"] for row in rows)
    if cases != CASES:
        raise SystemExit(f"unexpected case inventory {cases!r}")
    numeric = tuple(key for key in rows[0] if key != "case")
    values = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        for name in numeric
    }
    if not all(np.all(np.isfinite(field)) for field in values.values()):
        raise SystemExit("RUC soilvegin oracle contains non-finite output")

    cold, warm, crop, water, lai2d, short = range(6)
    if not values["lai"][cold] < values["lai"][warm]:
        raise SystemExit("forest seasonal LAI branch did not discriminate")
    if not values["znt"][crop] < np.float32(0.2):
        raise SystemExit("crop seasonal roughness branch did not execute")
    if values["znt"][water].view(np.uint32) != values["znt_before"][water].view(np.uint32):
        raise SystemExit("water roughness was not preserved")
    for name in (
        "qwrtz", "rhocs", "bclh", "dqm", "ksat", "psis", "qmin",
        "ref", "wilt",
    ):
        if values[name][water] != np.float32(0.0):
            raise SystemExit(f"water soil parameter {name} was not zero")
    if values["lai"][lai2d].view(np.uint32) != values["lai_before"][lai2d].view(np.uint32):
        raise SystemExit("rdlai2d did not preserve caller LAI")
    if not values["lai"][short] < np.float32(2.9):
        raise SystemExit("short greenness-range branch did not use cold season")
    print("RUC soilvegin oracle: PASS (six finite discriminating cases)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_soilvegin_oracle.py RUC_SOILVEGIN.csv")
    main(sys.argv[1])
