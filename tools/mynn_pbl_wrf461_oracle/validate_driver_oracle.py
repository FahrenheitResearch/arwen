#!/usr/bin/env python3
"""Compare the assembled CPU MYNN PBL driver with the official WRF CSV.

Both cold and warm calls are bitwise against WRF. The cold call must pass
vapor sqv, not total water sqw, into mym_initialize's qw argument.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.fp32_ulp import monotone_fp32_key
from gpuwm.core.mynn_pbl import (
    MYNN_DRIVER_LAYER_INPUTS,
    MYNN_DRIVER_SCALAR_INPUTS,
    MYNN_DRIVER_STATE,
    mynn_bl_driver,
)

EXPECTED_CASES = (
    "convective_land", "marine_cumulus", "stable_land", "cloudy_deep",
    "snow_anvil",
)
NZ = 30
NSTEP = 2
EXPECTED_ROWS = len(EXPECTED_CASES) * NZ * NSTEP

#: CSV column names for the incoming state.
STATE_IN = {name: f"{name}_in" for name in MYNN_DRIVER_STATE}
STATE_IN["el"] = "el_in"
STATE_IN["sh"] = "sh_in"
STATE_IN["sm"] = "sm_in"
STATE_IN["qc_bl"] = "qc_bl_in"
STATE_IN["qi_bl"] = "qi_bl_in"
STATE_IN["cldfra_bl"] = "cldfra_bl_in"

LAYER_CSV = {
    "sqv": "sqv3d", "sqc": "sqc3d", "sqi": "sqi3d", "sqs": "sqs3d",
    "tk": "t3d",
}
#: Output names that differ from the WRF 3-D array names.
OUTPUT_CSV = {"el": "el_pbl", "sh": "sh3d", "sm": "sm3d"}

#: Every profile output the driver writes back.
PROFILE_OUTPUTS = (
    "rublten", "rvblten", "rthblten", "rqvblten", "rqcblten", "rqiblten",
    "dozone", "exch_h", "exch_m", "qke", "tsq", "qsq", "cov", "el",
    "sh", "sm", "qc_bl", "qi_bl", "cldfra_bl",
)
SCALAR_OUTPUTS = ("pblh", "rmol", "maxwidth", "maxmf", "ztop_plume")
INT_OUTPUTS = ("kpbl", "ktop_plume")

#: The resolved cold-cloud initialization defect no longer needs a tolerance.
ULP_BUDGET: dict[tuple[int, str], int] = {}


def _ulp(got: np.ndarray, want: np.ndarray) -> int:
    return int(np.abs(monotone_fp32_key(got) - monotone_fp32_key(want)).max())


def main(path: str) -> None:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    if cases != EXPECTED_CASES:
        raise SystemExit(f"unexpected cases: {cases}")

    summary: dict[str, dict[str, int]] = {}
    worst_overall = 0
    for step in range(1, NSTEP + 1):
        selected = [row for row in rows if int(row["step"]) == step]
        if len(selected) != len(EXPECTED_CASES) * NZ:
            raise SystemExit(f"step {step} has {len(selected)} rows")
        blocks = [
            [row for row in selected if row["case"] == case]
            for case in EXPECTED_CASES
        ]
        values: dict[str, np.ndarray] = {}
        for name in MYNN_DRIVER_LAYER_INPUTS:
            key = LAYER_CSV.get(name, name)
            values[name] = np.asarray(
                [[np.float32(row[key]) for row in block] for block in blocks],
                dtype=np.float32,
            )
        for name in MYNN_DRIVER_STATE:
            values[name] = np.asarray(
                [[np.float32(row[STATE_IN[name]]) for row in block]
                 for block in blocks],
                dtype=np.float32,
            )
        for name in MYNN_DRIVER_SCALAR_INPUTS:
            values[name] = np.asarray(
                [np.float32(block[0][name]) for block in blocks],
                dtype=np.float32,
            )
        values["pblh"] = np.asarray(
            [np.float32(block[0]["pblh_in"]) for block in blocks],
            dtype=np.float32,
        )
        values["rmol"] = np.asarray(
            [np.float32(block[0]["rmol_in"]) for block in blocks],
            dtype=np.float32,
        )
        values["kpbl"] = np.asarray(
            [int(block[0]["kpbl_in"]) for block in blocks], dtype=np.int32
        )
        initflag = int(blocks[0][0]["initflag"])
        delt = np.float32(blocks[0][0]["delt"])
        actual = mynn_bl_driver(
            values, initflag=initflag, delt=delt, flag_qs=True,
        )

        step_summary: dict[str, int] = {}
        for name in PROFILE_OUTPUTS:
            key = OUTPUT_CSV.get(name, name)
            want = np.asarray(
                [[np.float32(row[key]) for row in block] for block in blocks],
                dtype=np.float32,
            )
            got = np.asarray(actual[name], dtype=np.float32)
            distance = _ulp(got, want)
            step_summary[name] = distance
            worst_overall = max(worst_overall, distance)
            budget = ULP_BUDGET.get((step, name), 0)
            if distance > budget:
                raise AssertionError(
                    f"step {step} {name}: {distance} ULP exceeds the measured "
                    f"budget of {budget}"
                )
        for name in SCALAR_OUTPUTS:
            key = name
            want = np.asarray(
                [np.float32(block[0][key]) for block in blocks],
                dtype=np.float32,
            )
            got = np.asarray(actual[name], dtype=np.float32).reshape(-1)
            distance = _ulp(got, want)
            step_summary[name] = distance
            worst_overall = max(worst_overall, distance)
            budget = ULP_BUDGET.get((step, name), 0)
            if distance > budget:
                raise AssertionError(
                    f"step {step} {name}: {distance} ULP exceeds the measured "
                    f"budget of {budget}"
                )
        for name in INT_OUTPUTS:
            want = np.asarray(
                [int(block[0][name]) for block in blocks], dtype=np.int32
            )
            got = np.asarray(actual[name], dtype=np.int32).reshape(-1)
            if not np.array_equal(got, want):
                raise AssertionError(f"step {step} {name}: {got} != {want}")
            step_summary[name] = 0
        summary[f"step{step}"] = step_summary

    print(json.dumps({
        "status": "PASS", "rows": len(rows), "worst_ulp": worst_overall,
        "steps": summary,
    }, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_driver_oracle.py driver.csv")
    main(sys.argv[1])
