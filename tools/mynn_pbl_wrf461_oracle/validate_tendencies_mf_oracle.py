#!/usr/bin/env python3
"""Compare the CPU MYNN tendency solve with the mass-flux WRF CSV."""

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
    mynn_tendencies_default,
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

#: The inventory ``run_tendencies_mf.F90`` emits.  Every strict check below
#: lives inside the per-case loop, so without this shape pin the validator
#: would certify a header-only CSV as a pass.
EXPECTED_CASES = (
    "land_cumulus", "water_cumulus", "deep_plume", "fine_grid",
    "momentum_off", "momentum_off_probe", "downdraft_probe",
    "moisture_repair", "subsidence_probe",
)
NZ = 30
EXPECTED_ROWS = len(EXPECTED_CASES) * NZ

#: Columns whose mass flux must be live.  A fixture that regenerated with a
#: dead plume model would otherwise reproduce the mass-flux-free lane and pass.
LIVE_MASS_FLUX = (
    "land_cumulus", "water_cumulus", "deep_plume", "fine_grid",
    "momentum_off", "momentum_off_probe", "downdraft_probe",
    "moisture_repair", "subsidence_probe",
)

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


def _case_values(selected: list[dict[str, str]]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {
        name: np.asarray(
            [[np.float32(row[RENAMED.get(name, name)]) for row in selected]],
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
    return values


def main(path: str) -> None:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    if cases != EXPECTED_CASES:
        raise SystemExit(f"unexpected cases: {cases}")
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    momentum: dict[str, np.ndarray] = {}
    for case in cases:
        selected = [row for row in rows if row["case"] == case]
        if len(selected) != NZ:
            raise SystemExit(
                f"{case} has {len(selected)} levels, expected {NZ}"
            )
        edmf_mom = {int(row["bl_mynn_edmf_mom"]) for row in selected}
        if len(edmf_mom) != 1:
            raise SystemExit(f"{case} mixes bl_mynn_edmf_mom values")
        values = _case_values(selected)
        if case in LIVE_MASS_FLUX and not np.any(values["s_aw"] != 0.0):
            raise SystemExit(
                f"{case} has an identically-zero s_aw; the fixture no longer "
                "exercises the mass flux"
            )
        actual = mynn_tendencies_default(
            values, bl_mynn_edmf_mom=edmf_mom.pop()
        )
        exchange = mynn_retrieve_exchange_coeffs(values)
        case_summary: dict[str, dict[str, float | int]] = {}
        for name in OUTPUT_COLUMNS:
            expected = np.asarray(
                [np.float32(row[name]) for row in selected], dtype=np.float32
            )
            got = actual[name][0]
            case_summary[name] = _report(got, expected)
            np.testing.assert_array_equal(
                got, expected, err_msg=f"{case}/{name}"
            )
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
        for name in ("dqnc", "dqni", "dqnwfa", "dqnifa", "dqnbca"):
            if np.any(actual[name][0] != 0.0):
                raise AssertionError(
                    f"{case}/{name} must stay zero under bl_mynn_mixscalars=0"
                )
        momentum[case] = np.asarray(
            [np.float32(row["du"]) for row in selected], dtype=np.float32
        )
        summary[case] = case_summary

    # Negative controls on the onoff factor.  momentum_off_probe carries the
    # same nonzero s_awu as land_cumulus but bl_mynn_edmf_mom=0, so it must
    # agree with momentum_off (whose s_awu is zero) and differ from
    # land_cumulus.  Without both, a port that ignored onoff would pass.
    if not np.array_equal(
        momentum["momentum_off"], momentum["momentum_off_probe"]
    ):
        raise AssertionError(
            "onoff=0 did not erase the momentum mass flux: momentum_off and "
            "momentum_off_probe disagree"
        )
    if np.array_equal(momentum["land_cumulus"], momentum["momentum_off"]):
        raise AssertionError(
            "onoff made no difference: land_cumulus and momentum_off agree"
        )
    print(json.dumps({
        "status": "PASS", "rows": len(rows), "cases": summary,
    }, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: validate_tendencies_mf_oracle.py tendencies-mf.csv"
        )
    main(sys.argv[1])
