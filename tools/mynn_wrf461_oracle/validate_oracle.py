from __future__ import annotations

import csv
import math
from pathlib import Path
import sys


EXPECTED_CASES = (
    "stable_land",
    "unstable_land",
    "neutral_land",
    "snow_land",
    "stable_water",
    "unstable_water",
)
EXPECTED_REGIMES = {
    "stable_land": {1.0, 2.0},
    "unstable_land": {4.0},
    "neutral_land": {3.0},
    "snow_land": {1.0, 2.0},
    "stable_water": {1.0, 2.0},
    "unstable_water": {4.0},
}
POSITIVE_FIELDS = (
    "rho1", "ust", "ustm", "chs", "ch", "flhc", "flqc", "qgh",
    "gz1oz0", "wspd", "ck", "cka", "cd", "cda", "cpm",
)


def validate(path: Path) -> None:
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    names = tuple(row["case"] for row in rows)
    if names != EXPECTED_CASES:
        raise ValueError(f"oracle cases are {names}, expected {EXPECTED_CASES}")
    for row in rows:
        name = row["case"]
        numeric = {
            key: float(value) for key, value in row.items() if key != "case"
        }
        bad = [key for key, value in numeric.items() if not math.isfinite(value)]
        if bad:
            raise ValueError(f"{name}: non-finite fields {bad}")
        if numeric["regime"] not in EXPECTED_REGIMES[name]:
            raise ValueError(f"{name}: unexpected regime {numeric['regime']}")
        bad = [key for key in POSITIVE_FIELDS if numeric[key] <= 0.0]
        if bad:
            raise ValueError(f"{name}: non-positive fields {bad}")
        if numeric["xland"] == 1.0 and numeric["ust"] < 0.005:
            raise ValueError(f"{name}: land friction-velocity floor failed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_oracle.py surface-layer.csv")
    validate(Path(sys.argv[1]))
    print(f"PASS {sys.argv[1]}")
