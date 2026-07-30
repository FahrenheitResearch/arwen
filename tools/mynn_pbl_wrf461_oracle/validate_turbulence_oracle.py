#!/usr/bin/env python3
"""Compare the CPU MYNN turbulence transcription with official WRF CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.fp32_ulp import monotone_fp32_key
from gpuwm.core.mynn_pbl import mynn_turbulence_default


INPUT_COLUMNS = (
    "dz", "u", "v", "thl", "thetav", "ql", "qw", "qke", "tsq",
    "qsq", "cov", "vt", "vq", "theta", "cldfra", "edmf_w",
    "edmf_a", "tkeprodtd",
)
SCALAR_COLUMNS = (
    "xland", "dx", "rmo", "flt", "fltv", "flq", "zi", "psig_bl",
    "psig_shcu",
)
OUTPUT_COLUMNS = (
    "dfm", "dfh", "dfq", "tcd", "qcd", "pdk", "pdt", "pdq", "pdc",
    "el", "sm", "sh",
)

#: The inventory ``run_turbulence.F90`` emits: a fixed case tuple at a fixed
#: column depth.  Every strict check below lives inside the per-case loop, so
#: without these the validator certifies a header-only CSV as a pass and a
#: truncated or failed regeneration goes unnoticed.  Pin the shape first.
EXPECTED_CASES = ("stable", "convective", "cloudy", "edmf_active")
NZ = 12
EXPECTED_ROWS = len(EXPECTED_CASES) * NZ


#: Measured worst-case FP32 ULP distance from the unmodified WRF oracle.
#: mym_turbulence is now bitwise on all 48 case/field pairs.  The former
#: 3-ULP residue was not a transcendental floor after all: ``el`` (the only
#: output of the two libm calls on this path) was already bitwise, and the
#: whole residue came from two Fortran subexpressions this port had widened
#: to FP64 -- ``3.0*c1*e5c`` and ``a2fac**2``, both real(kind_phys) at
#: module_bl_mynn.F:2834/2845 with kind_phys = kind(1.0).  Keep it at 0.
TURBULENCE_ULP_BUDGET = 0


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
            name: np.asarray(
                [[np.float32(row[name]) for row in selected]],
                dtype=np.float32,
            )
            for name in INPUT_COLUMNS
        }
        values["zw"] = np.asarray([[
            *[np.float32(row["zw"]) for row in selected],
            np.float32(selected[-1]["zw_next"]),
        ]], dtype=np.float32)
        for name in SCALAR_COLUMNS:
            values[name] = np.asarray(
                [np.float32(selected[0][name])], dtype=np.float32
            )
        actual = mynn_turbulence_default(values)
        case_summary: dict[str, dict[str, float | int]] = {}
        for name in OUTPUT_COLUMNS:
            expected = np.asarray(
                [np.float32(row[name]) for row in selected], dtype=np.float32
            )
            got = actual[name][0]
            absolute = np.abs(got.astype(np.float64) - expected.astype(np.float64))
            ulp = np.abs(_ordered_bits(got) - _ordered_bits(expected))
            scale = np.maximum(np.abs(expected.astype(np.float64)), 1.0e-30)
            relative = absolute / scale
            case_summary[name] = {
                "max_abs": float(absolute.max(initial=0.0)),
                "max_rel": float(relative.max(initial=0.0)),
                "max_ulp": int(ulp.max(initial=0)),
            }
            # The former gate was rtol=2e-5, which admits ~167 FP32 ULP --
            # 56x the error actually present, so it could not have caught a
            # regression.  Bound the quantity being measured instead.
            worst = int(ulp.max(initial=0))
            if worst > TURBULENCE_ULP_BUDGET:
                raise AssertionError(
                    f"{case}/{name}: {worst} ULP exceeds the measured budget "
                    f"of {TURBULENCE_ULP_BUDGET}"
                )
        summary[case] = case_summary
    print(json.dumps({
        "status": "PASS", "rows": len(rows), "cases": summary,
    }, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_turbulence_oracle.py turbulence.csv")
    main(sys.argv[1])

