#!/usr/bin/env python3
"""Compare the CPU MYNN prognostic predictor with official WRF CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.fp32_ulp import monotone_fp32_key
from gpuwm.core.mynn_pbl import mynn_predict_default


ARRAY_COLUMNS = (
    "dz", "rho", "dfq", "pdk", "pdt", "pdq", "pdc", "el",
    "qke_before", "tsq_before", "qsq_before", "cov_before",
)
SCALAR_COLUMNS = ("ust", "flt", "flq", "pmz", "phh", "delt")
OUTPUT_COLUMNS = ("qke", "tsq", "qsq", "cov")

#: The inventory ``run_predict.F90`` emits: a fixed case tuple at a fixed
#: column depth.  Every strict check below lives inside the per-case loop, so
#: without these the validator certifies a header-only CSV as a pass and a
#: truncated or failed regeneration goes unnoticed.  Pin the shape first.
EXPECTED_CASES = ("stable", "convective", "cloudy", "edmf_active")
NZ = 12
EXPECTED_ROWS = len(EXPECTED_CASES) * NZ


#: The total-order FP32 key is shared rather than re-derived here.
#: Thirteen local copies of this two-line bit trick carried the same
#: sign error, which reported -0.0 as 2**32 ULP from +0.0; see
#: gpuwm/core/fp32_ulp.py for the measurement and the live case.
_ordered_bits = monotone_fp32_key


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
        values = {
            name.removesuffix("_before"): np.asarray(
                [[np.float32(row[name]) for row in selected]],
                dtype=np.float32,
            )
            for name in ARRAY_COLUMNS
        }
        for name, next_name in (
            ("s_aw", "s_aw_next"), ("s_awqke", "s_awqke_next"),
        ):
            values[name] = np.asarray([[
                *[np.float32(row[name]) for row in selected],
                np.float32(selected[-1][next_name]),
            ]], dtype=np.float32)
        for name in SCALAR_COLUMNS:
            values[name] = np.asarray(
                [np.float32(selected[0][name])], dtype=np.float32
            )
        actual = mynn_predict_default(values)
        case_summary: dict[str, dict[str, float | int]] = {}
        for name in OUTPUT_COLUMNS:
            expected = np.asarray(
                [np.float32(row[f"{name}_after"]) for row in selected],
                dtype=np.float32,
            )
            got = actual[name][0]
            absolute = np.abs(got.astype(np.float64) - expected.astype(np.float64))
            relative = absolute / np.maximum(
                np.abs(expected.astype(np.float64)), 1.0e-30
            )
            ulp = np.abs(_ordered_bits(got) - _ordered_bits(expected))
            case_summary[name] = {
                "max_abs": float(absolute.max(initial=0.0)),
                "max_rel": float(relative.max(initial=0.0)),
                "max_ulp": int(ulp.max(initial=0)),
            }
            # mynn_predict_default is already exact on every field of every
            # case.  The former gate was rtol=3e-5, which admits ~251 FP32
            # ULP, so a regression of that size would have passed silently.
            # Assert what the code actually achieves.
            worst = int(ulp.max(initial=0))
            if worst:
                raise AssertionError(
                    f"{case}/{name}: {worst} ULP from the unmodified WRF "
                    f"oracle; this routine is bitwise and must stay bitwise"
                )
        summary[case] = case_summary
    print(json.dumps({
        "status": "PASS", "rows": len(rows), "cases": summary,
    }, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_predict_oracle.py predict.csv")
    main(sys.argv[1])
