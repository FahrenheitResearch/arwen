#!/usr/bin/env python3
"""Bitwise comparison of two Ferrier column-oracle result sets.

usage: compare_ferrier.py REF_DIR TEST_DIR [INPUTS_CSV]

Each directory holds ``levels.csv`` and ``surface.csv`` in the format
``run_ferrier.F90`` writes (floats as binary32 hex).  A port of the scheme
is graded by writing the same two files from the same ``columns_inputs.csv``
and running this script against the Fortran reference.

Prints, per output field: rows compared, rows that differ in any bit, max
ULP distance and max absolute difference, and the worst column with its
regime.  Exit status 0 only if every field is bitwise equal.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gpuwm.core.fp32_ulp import monotone_fp32_key  # noqa: E402


def _load(path: Path) -> tuple[list[str], dict[tuple, np.ndarray]]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    nkey = 3 if head[2] == "k" else 2
    fields = head[nkey:]
    data = {tuple(int(x) for x in r[:nkey]):
            np.array([int(x, 16) for x in r[nkey:]], dtype=np.uint32)
            for r in body}
    return fields, data


def _ordered(bits: np.ndarray) -> np.ndarray:
    return monotone_fp32_key(np.ascontiguousarray(bits, dtype=np.uint32).view(np.float32))


def _regimes(inputs: Path | None) -> dict[int, str]:
    if inputs is None or not inputs.exists():
        return {}
    out = {}
    with open(inputs) as f:
        for line in f:
            if line.startswith("C,"):
                parts = line.split(",")
                out[int(parts[1])] = parts[2]
    return out


def compare(ref: Path, test: Path, inputs: Path | None = None) -> bool:
    regimes = _regimes(inputs)
    exact = True
    for name in ("levels.csv", "surface.csv"):
        fields, a = _load(ref / name)
        fields_b, b = _load(test / name)
        if fields != fields_b or a.keys() != b.keys():
            print(f"{name}: layouts differ")
            return False
        keys = sorted(a)
        A = np.stack([a[k] for k in keys])
        B = np.stack([b[k] for k in keys])
        fa = A.view(np.float32).astype(np.float64)
        fb = B.view(np.float32).astype(np.float64)
        ulp = np.abs(_ordered(A) - _ordered(B))
        for j, fld in enumerate(fields):
            diff = A[:, j] != B[:, j]
            n = int(diff.sum())
            exact &= n == 0
            mu = int(ulp[:, j].max())
            mad = float(np.nanmax(np.abs(fa[:, j] - fb[:, j])))
            worst = keys[int(ulp[:, j].argmax())]
            tag = regimes.get(worst[0], "?") if n else ""
            print(f"{name:12s} {fld:8s} rows={len(keys):6d} differ={n:6d} "
                  f"max_ulp={mu:12d} max_abs={mad:.3e}"
                  + (f" worst={worst} ({tag})" if n else ""))
    print("RESULT", "BITWISE EQUAL" if exact else "NOT EQUAL")
    return exact


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    inp = Path(sys.argv[3]) if len(sys.argv) == 4 else \
        Path(sys.argv[1]).resolve().parent / "columns_inputs.csv"
    sys.exit(0 if compare(Path(sys.argv[1]), Path(sys.argv[2]), inp) else 1)
