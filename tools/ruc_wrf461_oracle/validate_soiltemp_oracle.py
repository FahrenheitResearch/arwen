#!/usr/bin/env python3
"""Structural checks for unmodified-WRF RUC ``soiltemp``."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


CASES = (
    "moist_saturated",
    "dry_second_solve",
    "warm_rain",
    "humid_condensing",
)


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
        raise SystemExit("RUC soiltemp oracle contains non-finite output")
    if not all(np.any(values["tso_after"][case] != values["tso_before"][case])
               for case in range(4)):
        raise SystemExit("soiltemp failed to evolve every temperature profile")
    if not values["qvg_after"][1, 0] < values["qsg_after"][1, 0]:
        raise SystemExit("dry second-solve regime did not remain unsaturated")
    if not values["qcg_after"][3, 0] > 0.0:
        raise SystemExit("humid regime did not enter the condensing branch")
    if not values["rainf"][2, 0] == 1.0 or not values["prcpms"][2, 0] > 0.0:
        raise SystemExit("rain-heating regime was not pinned")
    print("RUC soiltemp oracle: PASS (four finite nonlinear heat regimes)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_soiltemp_oracle.py RUC_SOILTEMP.csv")
    main(sys.argv[1])
