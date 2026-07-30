#!/usr/bin/env python3
"""Validate the default-option MYNN mixing-length oracle."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


EXPECTED = ("stable", "convective", "high_shear", "edmf_active")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_mixlength_oracle.py ORACLE.csv")
    with Path(sys.argv[1]).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 48:
        raise SystemExit(f"expected 48 rows, found {len(rows)}")
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    if cases != EXPECTED:
        raise SystemExit(f"unexpected cases: {cases}")
    for name in EXPECTED:
        selected = [row for row in rows if row["case"] == name]
        el = [float(row["el"]) for row in selected]
        qkw = [float(row["qkw"]) for row in selected]
        if not all(math.isfinite(value) and value >= 0.0 for value in el + qkw):
            raise SystemExit(f"invalid mixing length in {name}")
        if el[0] != 0.0 or not all(value > 0.0 for value in el[1:] + qkw):
            raise SystemExit(f"boundary/interior identity failed in {name}")
    print({"status": "PASS", "rows": len(rows), "cases": list(cases)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
