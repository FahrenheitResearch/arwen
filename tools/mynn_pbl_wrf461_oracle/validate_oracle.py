#!/usr/bin/env python3
"""Validate the shape and physical branches of the MYNN level-2 oracle."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


EXPECTED = ("stable_dry", "convective_dry", "neutral_shear", "moist_cloud")
FIELDS = ("dtl", "dqw", "dtv", "gm", "gh", "sm", "sh")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_oracle.py ORACLE.csv")
    path = Path(sys.argv[1])
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 28:
        raise SystemExit(f"expected 28 rows, found {len(rows)}")
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    if cases != EXPECTED:
        raise SystemExit(f"unexpected cases: {cases}")
    for row in rows:
        for name in FIELDS:
            if not math.isfinite(float(row[name])):
                raise SystemExit(f"non-finite {name} in {row['case']} k={row['k']}")
        if float(row["gm"]) <= 0.0 or float(row["sm"]) <= 0.0 \
                or float(row["sh"]) <= 0.0:
            raise SystemExit(f"non-positive mixing term in {row['case']} k={row['k']}")
    by_case = {
        name: [row for row in rows if row["case"] == name]
        for name in EXPECTED
    }
    if not all(float(row["gh"]) < 0.0 for row in by_case["stable_dry"]):
        raise SystemExit("stable profile did not produce negative GH")
    if not all(float(row["gh"]) > 0.0 for row in by_case["convective_dry"]):
        raise SystemExit("convective profile did not produce positive GH")
    if not all(float(row["gh"]) == 0.0 for row in by_case["neutral_shear"]):
        raise SystemExit("neutral profile did not preserve exact zero GH")
    print({"status": "PASS", "rows": len(rows), "cases": list(cases)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
