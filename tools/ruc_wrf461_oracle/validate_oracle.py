#!/usr/bin/env python3
"""Validate a freshly generated unmodified-WRF RUC initialization CSV."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.ruc import ruc_initialize_cold_start


def main(path: str) -> None:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 36:
        raise SystemExit(f"expected 36 rows, got {len(rows)}")
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    if cases != ("warm_land", "frozen_land", "water", "sea_ice"):
        raise SystemExit(f"unexpected case inventory {cases!r}")
    fields = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(4, 9).T
        for name in ("tslb", "smois", "sh2o", "smfr3d")
    }
    horizontal = {
        name: np.asarray(
            [float(rows[index * 9][name]) for index in range(4)],
            dtype=np.float32,
        )
        for name in ("xice", "mavail", "znt")
    }
    soil = np.asarray(
        [int(rows[index * 9]["isltyp"]) for index in range(4)],
        dtype=np.int32,
    )
    vegetation = np.asarray(
        [int(rows[index * 9]["ivgtyp"]) for index in range(4)],
        dtype=np.int32,
    )
    actual = ruc_initialize_cold_start(
        fields["tslb"], fields["smois"], soil, vegetation, horizontal["xice"]
    )
    for name, expected in (
        ("sh2o", fields["sh2o"]),
        ("smfr3d", fields["smfr3d"]),
        ("mavail", horizontal["mavail"]),
        ("znt", horizontal["znt"]),
    ):
        candidate = getattr(actual, name)
        if not np.array_equal(candidate.view(np.uint32), expected.view(np.uint32)):
            difference = np.abs(
                candidate.view(np.int32).astype(np.int64)
                - expected.view(np.int32).astype(np.int64)
            )
            raise SystemExit(
                f"{name} differs from WRF: max integer-bit distance "
                f"{int(difference.max())}"
            )
    print("RUC initialization oracle: PASS (bit-for-bit)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_oracle.py RUC_INIT.csv")
    main(sys.argv[1])
