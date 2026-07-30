#!/usr/bin/env python3
"""Compare the CPU MYNN mass-flux plume scheme with the official WRF CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.fp32_ulp import monotone_fp32_key
from gpuwm.core.mynn_pbl import (
    MYNN_DMP_MF_COLUMN_INPUTS,
    MYNN_DMP_MF_INTERFACE_OUTPUTS,
    MYNN_DMP_MF_LAYER_OUTPUTS,
    MYNN_DMP_MF_SCALAR_INPUTS,
    MYNN_DMP_MF_ZERO_INTERFACE_OUTPUTS,
    MYNN_DMP_MF_ZERO_OUTPUTS,
    mynn_dmp_mf,
)


# CSV names for the arguments WRF declares intent(inout).
RENAMED = {
    "qc_bl": "qc_bl_before", "cldfra_bl": "cldfra_bl_before",
    "vt": "vt_before", "vq": "vq_before",
}

#: The inventory ``run_dmp_mf.F90`` emits: a fixed case tuple at a fixed
#: column depth.  Every strict check below lives inside the per-case loop, so
#: without these the validator certifies a header-only CSV as a pass and a
#: truncated or failed regeneration goes unnoticed.  Pin the shape first.
EXPECTED_CASES = (
    "land_dry", "land_cumulus", "water_cumulus", "stable_off",
    "resolved_w", "flux_limited", "stochastic", "high_wind_thin",
    "deep_pblh", "cloud_base_capped", "fine_grid", "dead_probe",
)
NZ = 30
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


def case_inputs(selected: list[dict[str, str]]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {
        name: np.asarray(
            [[np.float32(row[RENAMED.get(name, name)]) for row in selected]],
            dtype=np.float32,
        )
        for name in MYNN_DMP_MF_COLUMN_INPUTS
    }
    values["zw"] = np.asarray([[
        *[np.float32(row["zw"]) for row in selected],
        np.float32(selected[-1]["zw_next"]),
    ]], dtype=np.float32)
    for name in MYNN_DMP_MF_SCALAR_INPUTS:
        values[name] = np.asarray(
            [np.float32(selected[0][name])], dtype=np.float32
        )
    return values


def expected_outputs(selected: list[dict[str, str]]) -> dict[str, np.ndarray]:
    wanted: dict[str, np.ndarray] = {
        name: np.asarray([np.float32(row[name]) for row in selected],
                         dtype=np.float32)
        for name in (*MYNN_DMP_MF_LAYER_OUTPUTS, *MYNN_DMP_MF_ZERO_OUTPUTS)
    }
    for name in MYNN_DMP_MF_INTERFACE_OUTPUTS:
        wanted[name] = np.asarray([
            *[np.float32(row[name]) for row in selected],
            np.float32(selected[-1][f"{name}_next"]),
        ], dtype=np.float32)
    for name in MYNN_DMP_MF_ZERO_INTERFACE_OUTPUTS:
        wanted[name] = np.asarray([
            *[np.float32(row[name]) for row in selected], np.float32(0.0),
        ], dtype=np.float32)
    for name in ("maxwidth", "ztop", "maxmf"):
        wanted[name] = np.asarray([np.float32(selected[0][name])],
                                  dtype=np.float32)
    wanted["ktop"] = np.asarray([int(selected[0]["ktop"])], dtype=np.int32)
    return wanted


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
        actual = mynn_dmp_mf(case_inputs(selected))
        wanted = expected_outputs(selected)
        case_summary: dict[str, dict[str, float | int]] = {}
        for name, expected in wanted.items():
            got = actual[name][0] if actual[name].ndim == 2 else actual[name]
            if name == "ktop":
                if int(got[0]) != int(expected[0]):
                    raise AssertionError(
                        f"{case}/ktop = {int(got[0])}, WRF {int(expected[0])}"
                    )
                case_summary[name] = {"max_abs": 0.0, "max_rel": 0.0,
                                      "max_ulp": 0}
                continue
            case_summary[name] = _report(got, expected)
            np.testing.assert_array_equal(
                got, expected, err_msg=f"{case}/{name}"
            )
        summary[case] = case_summary
    print(json.dumps({
        "status": "PASS", "rows": len(rows), "cases": summary,
    }, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_dmp_mf_oracle.py dmp-mf.csv")
    main(sys.argv[1])
