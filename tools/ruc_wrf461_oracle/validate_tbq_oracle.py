#!/usr/bin/env python3
"""Structural checks for the unmodified-WRF RUC saturation table."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


def main(path: str) -> None:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 5001:
        raise SystemExit(f"expected 5001 rows, got {len(rows)}")
    indices = np.asarray([int(row["k"]) for row in rows])
    values = np.asarray([float(row["tbq"]) for row in rows], dtype=np.float32)
    if not np.array_equal(indices, np.arange(1, 5002)):
        raise SystemExit("RUC tbq indices are not exactly 1..5001")
    if not np.all(np.isfinite(values)) or not np.all(values > 0.0):
        raise SystemExit("RUC tbq contains invalid values")
    if not np.all(np.diff(values) > 0.0):
        raise SystemExit("RUC tbq is not strictly increasing")
    print("RUC tbq oracle: PASS (5001 finite increasing entries)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_tbq_oracle.py RUC_TBQ.csv")
    main(sys.argv[1])
