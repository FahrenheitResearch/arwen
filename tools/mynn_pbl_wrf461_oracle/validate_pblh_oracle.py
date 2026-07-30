#!/usr/bin/env python3
"""Validate the MYNN PBL-height and scale-aware oracle."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


EXPECTED = ("convective_land", "stable_land", "marine", "cold_pool")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_pblh_oracle.py ORACLE.csv")
    with Path(sys.argv[1]).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 40:
        raise SystemExit(f"expected 40 rows, found {len(rows)}")
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    if cases != EXPECTED:
        raise SystemExit(f"unexpected cases: {cases}")
    outputs = {}
    for name in EXPECTED:
        selected = [row for row in rows if row["case"] == name]
        values = {
            (float(row["zi"]), int(row["kzi"]), float(row["psig_bl"]),
             float(row["psig_shcu"]))
            for row in selected
        }
        if len(values) != 1:
            raise SystemExit(f"nonconstant column output for {name}")
        zi, kzi, psig_bl, psig_shcu = values.pop()
        if not all(math.isfinite(value) for value in (zi, psig_bl, psig_shcu)):
            raise SystemExit(f"non-finite output for {name}")
        if zi <= 0.0 or not 1 <= kzi <= 10:
            raise SystemExit(f"invalid PBL result for {name}: zi={zi}, kzi={kzi}")
        if not 0.0 <= psig_bl <= 1.0 or not 0.0 <= psig_shcu <= 1.0:
            raise SystemExit(f"invalid scale-aware factors for {name}")
        outputs[name] = (zi, kzi, psig_bl, psig_shcu)
    if outputs["convective_land"][0] == outputs["stable_land"][0]:
        raise SystemExit("convective and stable PBL heights unexpectedly match")
    print({"status": "PASS", "rows": len(rows), "outputs": outputs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
