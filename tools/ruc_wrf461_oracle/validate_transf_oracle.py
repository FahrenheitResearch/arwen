#!/usr/bin/env python3
"""Structural and branch checks for unmodified-WRF RUC ``transf``."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


CASES = ("wet_forest", "wilt_dark_grass", "mixed_hot_crop", "bare_high_sun")


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
        raise SystemExit("RUC transf oracle contains non-finite output")
    expected_roots = np.asarray([8, 6, 6, 4], dtype=np.float32)
    if not np.array_equal(values["nroot"][:, 0], expected_roots):
        raise SystemExit("unexpected variable root-zone inventory")
    for case, root_count in enumerate(expected_roots.astype(int)):
        if not np.all(values["tranf"][case, root_count:] == np.float32(0.0)):
            raise SystemExit(f"transpiration escaped case {CASES[case]} root zone")
    if not np.all(values["tranf"][0, :4] > values["tranf"][1, :4]):
        raise SystemExit("wet/high-light and wilt/dark branches did not discriminate")
    if len(np.unique(values["tranf"][2, :6])) < 3:
        raise SystemExit("mixed-moisture root branches did not discriminate")
    for case, root_count in enumerate(expected_roots.astype(int)):
        if not np.isclose(
            np.sum(values["tranf"][case, :root_count], dtype=np.float32),
            values["transum"][case, 0],
            rtol=2.0e-7,
            atol=1.0e-9,
        ):
            raise SystemExit(f"case {CASES[case]} transum is inconsistent")
    print("RUC transf oracle: PASS (four finite variable-root regimes)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_transf_oracle.py RUC_TRANSF.csv")
    main(sys.argv[1])
