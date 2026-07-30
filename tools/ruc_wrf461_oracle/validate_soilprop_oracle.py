#!/usr/bin/env python3
"""Structural and branch checks for unmodified-WRF RUC ``soilprop``."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


CASES = ("warm_wet", "warm_dry", "frozen", "deep_frozen_keep")


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
        raise SystemExit("RUC soilprop oracle contains non-finite output")
    if not np.all(values["hydro"][0] > np.float32(0.0)):
        raise SystemExit("warm-wet hydraulic branch did not execute")
    if not np.any(values["hydro"][1] == np.float32(0.0)):
        raise SystemExit("warm-dry conductivity floor did not execute")
    if not (
        np.all(values["diffu"][3, :-1] == np.float32(0.0))
        and np.all(values["hydro"][3] == np.float32(0.0))
    ):
        raise SystemExit("deep-frozen conductivity shutoff did not execute")
    for name in ("thdif", "diffu", "cap"):
        if not np.all(values[name][:, -1] == np.float32(0.0)):
            raise SystemExit(f"WRF last-level {name} initialization drifted")
    print("RUC soilprop oracle: PASS (four finite discriminating regimes)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_soilprop_oracle.py RUC_SOILPROP.csv")
    main(sys.argv[1])
