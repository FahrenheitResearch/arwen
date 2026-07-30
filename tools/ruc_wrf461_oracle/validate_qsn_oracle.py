#!/usr/bin/env python3
"""Parity check for the gpuwm ``qsn`` port against unmodified WRF v4.6.1.

Loads the freshly generated ``qsn`` CSV, feeds every sampled temperature to
``gpuwm.core.ruc.ruc_qsn`` (which reads the pinned ``tbq`` fixture, not the
harness's in-memory copy) and reports ``max_abs``/``max_rel``/``max_ulp``.
The achieved bar is bitwise, so the script fails on any nonzero ULP.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.fp32_ulp import monotone_fp32_key
from gpuwm.core.ruc import ruc_qsn


CASES = ("clamp_low", "node_low", "interior", "node_high", "clamp_high")


#: Monotonic integer ordering of float32 bits; +0 and -0 collapse.  Shared
#: rather than re-derived: see gpuwm/core/fp32_ulp.py.
_ordinal = monotone_fp32_key


def _report(name: str, actual: np.ndarray, expected: np.ndarray) -> int:
    difference = np.abs(
        actual.astype(np.float64) - expected.astype(np.float64)
    )
    denominator = np.abs(expected.astype(np.float64))
    relative = np.zeros_like(difference)
    np.divide(difference, denominator, out=relative, where=denominator > 0.0)
    ulp = np.abs(_ordinal(actual) - _ordinal(expected))
    signed_zero = int(
        np.count_nonzero(
            (actual == 0.0)
            & (expected == 0.0)
            & (np.signbit(actual) != np.signbit(expected))
        )
    )
    print(
        f"  {name:<12s} max_abs={float(np.max(difference)):.6e} "
        f"max_rel={float(np.max(relative)):.6e} "
        f"max_ulp={int(np.max(ulp))} signed_zero_mismatch={signed_zero}"
    )
    return int(np.max(ulp)) + signed_zero


def main(path: str) -> None:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 81:
        raise SystemExit(f"expected 81 rows, got {len(rows)}")
    seen = set(row["case"] for row in rows)
    if seen != set(CASES):
        raise SystemExit(f"unexpected case inventory {sorted(seen)!r}")

    temperature = np.asarray([float(row["tn"]) for row in rows], dtype=np.float32)
    expected = np.asarray([float(row["qsn"]) for row in rows], dtype=np.float32)
    raw_index = np.asarray([float(row["r_raw"]) for row in rows], dtype=np.float32)
    if not np.all(np.isfinite(expected)) or not np.all(expected > 0.0):
        raise SystemExit("RUC qsn oracle contains invalid saturation values")

    labels = np.asarray([row["case"] for row in rows])
    if not np.all(raw_index[labels == "clamp_low"] < 1.0):
        raise SystemExit("clamp_low rows did not fall below the first node")
    if not np.all(raw_index[labels == "clamp_high"] > 5001.0):
        raise SystemExit("clamp_high rows did not exceed the last node")
    low = expected[labels == "clamp_low"]
    high = expected[labels == "clamp_high"]
    if not np.all(low == low[0]) or not np.all(high == high[0]):
        raise SystemExit("the qsn end clamps are not flat")
    if not np.all(np.diff(expected[labels == "interior"]) > 0.0):
        raise SystemExit("the qsn interior sweep is not monotonically increasing")

    actual = ruc_qsn(temperature)
    print("RUC qsn oracle parity (gpuwm.core.ruc.ruc_qsn):")
    failures = _report("qsn", actual, expected)
    if failures:
        raise SystemExit("RUC qsn oracle: FAIL (not bitwise)")
    print(f"RUC qsn oracle: PASS (bitwise over {len(rows)} samples, "
          f"{len(set(labels))} branches)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_qsn_oracle.py RUC_QSN.csv")
    main(sys.argv[1])
