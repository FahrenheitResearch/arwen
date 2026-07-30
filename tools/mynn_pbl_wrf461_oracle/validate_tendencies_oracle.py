#!/usr/bin/env python3
"""Compare the CPU MYNN tendency solve with the official WRF CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.fp32_ulp import monotone_fp32_key
from gpuwm.core.mynn_pbl import (
    MYNN_TENDENCIES_INTERFACE_INPUTS,
    MYNN_TENDENCIES_LAYER_INPUTS,
    MYNN_TENDENCIES_SCALAR_INPUTS,
    mynn_retrieve_exchange_coeffs,
    mynn_tendencies_nomf,
)


# CSV column names that differ from the argument names.
RENAMED = {
    "thl": "thl_before",
    "sqv": "sqv_before",
    "sqc": "sqc_before",
    "sqi": "sqi_before",
    "sqs": "sqs_before",
}
OUTPUT_COLUMNS = (
    "du", "dv", "dth", "dqv", "dqc", "dqi", "dqs", "dozone",
)
STATE_COLUMNS = {"thl": "thl_after"}
EXCHANGE_COLUMNS = ("k_m", "k_h")

#: The inventory ``run_tendencies.F90`` emits: a fixed case tuple at a fixed
#: column depth.  Every strict check below lives inside the per-case loop, so
#: without these the validator certifies a header-only CSV as a pass and a
#: truncated or failed regeneration goes unnoticed.  Pin the shape first.
EXPECTED_CASES = ("stable", "convective", "cloudy", "depleted_moisture")
NZ = 16
EXPECTED_ROWS = len(EXPECTED_CASES) * NZ


#: The total-order FP32 key is shared rather than re-derived here.
#: Thirteen local copies of this two-line bit trick carried the same
#: sign error, which reported -0.0 as 2**32 ULP from +0.0; see
#: gpuwm/core/fp32_ulp.py for the measurement and the live case.
_ordered_bits = monotone_fp32_key


def _report(got: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    absolute = np.abs(got.astype(np.float64) - expected.astype(np.float64))
    relative = absolute / np.maximum(
        np.abs(expected.astype(np.float64)), 1.0e-30
    )
    ulp = np.abs(_ordered_bits(got) - _ordered_bits(expected))
    return {
        "max_abs": float(absolute.max(initial=0.0)),
        "max_rel": float(relative.max(initial=0.0)),
        "max_ulp": int(ulp.max(initial=0)),
    }


def main(path: str) -> None:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    if cases != EXPECTED_CASES:
        raise SystemExit(f"unexpected cases: {cases}")
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for case in cases:
        selected = [row for row in rows if row["case"] == case]
        if len(selected) != NZ:
            raise SystemExit(
                f"{case} has {len(selected)} levels, expected {NZ}"
            )
        values: dict[str, np.ndarray] = {
            name: np.asarray(
                [[np.float32(row[RENAMED.get(name, name)])
                  for row in selected]],
                dtype=np.float32,
            )
            for name in MYNN_TENDENCIES_LAYER_INPUTS
        }
        for name in MYNN_TENDENCIES_INTERFACE_INPUTS:
            values[name] = np.asarray([[
                *[np.float32(row[name]) for row in selected],
                np.float32(selected[-1][f"{name}_next"]),
            ]], dtype=np.float32)
        for name in MYNN_TENDENCIES_SCALAR_INPUTS:
            values[name] = np.asarray(
                [np.float32(selected[0][name])], dtype=np.float32
            )
        actual = mynn_tendencies_nomf(values)
        exchange = mynn_retrieve_exchange_coeffs(values)
        case_summary: dict[str, dict[str, float | int]] = {}
        for name in OUTPUT_COLUMNS:
            expected = np.asarray(
                [np.float32(row[name]) for row in selected], dtype=np.float32
            )
            got = actual[name][0]
            case_summary[name] = _report(got, expected)
            np.testing.assert_array_equal(got, expected, err_msg=f"{case}/{name}")
        for name, csv_name in STATE_COLUMNS.items():
            expected = np.asarray(
                [np.float32(row[csv_name]) for row in selected],
                dtype=np.float32,
            )
            got = actual[name][0]
            case_summary[csv_name] = _report(got, expected)
            np.testing.assert_array_equal(
                got, expected, err_msg=f"{case}/{csv_name}"
            )
        for name in EXCHANGE_COLUMNS:
            expected = np.asarray(
                [np.float32(row[name]) for row in selected], dtype=np.float32
            )
            got = exchange[name][0]
            case_summary[name] = _report(got, expected)
            np.testing.assert_array_equal(
                got, expected, err_msg=f"{case}/{name}"
            )
        # The identically-zero tendencies are checked separately so a
        # regression that made them nonzero cannot hide behind a passing
        # relative tolerance.
        for name in ("dqnc", "dqni", "dqnwfa", "dqnifa", "dqnbca"):
            got = actual[name][0]
            if np.any(got != 0.0):
                raise AssertionError(
                    f"{case}/{name} must stay zero under bl_mynn_mixscalars=0"
                )
        summary[case] = case_summary
    print(json.dumps({
        "status": "PASS", "rows": len(rows), "cases": summary,
    }, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: validate_tendencies_oracle.py tendencies-nomf.csv"
        )
    main(sys.argv[1])
