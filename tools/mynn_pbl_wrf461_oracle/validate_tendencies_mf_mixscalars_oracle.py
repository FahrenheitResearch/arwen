#!/usr/bin/env python3
"""Compare the CPU MYNN mixscalars tendency solve with the anchored WRF CSV.

W4 full-admission lane (stock half).  Fixture: tendencies-mf-mixscalars.csv
(w4-oracle-fixtures, SHA-256 564c867a...841c), built from WRF
v4.6.1 module_bl_mynn.F b36c8b93...49452 with bl_mynn_mixscalars=1 and the
five FLAG_QN* true.  Every shared base column is bit-identical to the
admitted tendencies-mf fixture (manifest, measured), so this validator
gates ONLY correctly if the base outputs stay bit-exact too — both are
asserted.
"""

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
    MYNN_TENDENCIES_QN_INTERFACE_INPUTS,
    MYNN_TENDENCIES_QN_LAYER_INPUTS,
    MYNN_TENDENCIES_SCALAR_INPUTS,
    mynn_tendencies_default,
)

RENAMED = {
    "thl": "thl_before",
    "sqv": "sqv_before",
    "sqc": "sqc_before",
    "sqi": "sqi_before",
    "sqs": "sqs_before",
}
#: The base outputs that must STAY bit-exact with mixscalars=1 (turning the
#: option on changes no admitted-path output — the fixture family's own
#: differential guarantee) plus the five live qn tendencies this lane gates.
BASE_OUTPUT_COLUMNS = (
    "du", "dv", "dth", "dqv", "dqc", "dqi", "dqs", "dozone",
)
QN_OUTPUT_COLUMNS = ("dqnc", "dqni", "dqnwfa", "dqnifa", "dqnbca")

#: The inventory ``run_tendencies_mf_mixscalars.F90`` emits.  Shape pinned
#: first so a header-only or truncated CSV cannot pass.
EXPECTED_CASES = (
    "land_cumulus", "water_cumulus", "deep_plume", "fine_grid",
    "momentum_off", "momentum_off_probe", "downdraft_probe",
    "moisture_repair", "subsidence_probe",
)
NZ = 30
EXPECTED_ROWS = len(EXPECTED_CASES) * NZ

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
        for name in (*MYNN_TENDENCIES_LAYER_INPUTS,
                     *MYNN_TENDENCIES_QN_LAYER_INPUTS)
    }
    for name in (*MYNN_TENDENCIES_INTERFACE_INPUTS,
                 *MYNN_TENDENCIES_QN_INTERFACE_INPUTS):
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
        # Liveness pins: a fixture regenerated with a dead plume or dead
        # scalar path would reproduce the mixscalars-free lane and pass.
        if not np.any(values["s_aw"] != 0.0):
            raise SystemExit(f"{case} has an identically-zero s_aw")
        if not any(np.any(values[name] != 0.0)
                   for name in MYNN_TENDENCIES_QN_INTERFACE_INPUTS):
            raise SystemExit(f"{case} has identically-zero s_awqn* inputs")
        actual = mynn_tendencies_default(
            values, bl_mynn_edmf_mom=edmf_mom.pop(),
            bl_mynn_mixscalars=1,
            flag_qnc=True, flag_qni=True, flag_qnwfa=True,
            flag_qnifa=True, flag_qnbca=True,
        )
        case_summary: dict[str, dict[str, float | int]] = {}
        for name in (*BASE_OUTPUT_COLUMNS, *QN_OUTPUT_COLUMNS):
            expected = np.asarray(
                [np.float32(row[name]) for row in selected], dtype=np.float32
            )
            got = actual[name][0]
            case_summary[name] = _report(got, expected)
            np.testing.assert_array_equal(
                got, expected, err_msg=f"{case}/{name}"
            )
        if all(not np.any(actual[name][0] != 0.0)
               for name in QN_OUTPUT_COLUMNS):
            raise AssertionError(
                f"{case}: every qn tendency is zero; the mixscalars arm "
                "did not run"
            )
        summary[case] = case_summary
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("PASS tendencies-mf-mixscalars", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tendencies-mf-mixscalars.csv")
