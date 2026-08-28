#!/usr/bin/env python3
"""Compare the CPU MYNN mass-flux scalar_opt=1 lane with the anchored WRF CSV.

W4 full-admission lane (stock half).  Fixture: dmp-mf-mixscalars.csv
(w4-oracle-fixtures, SHA-256 4740a0f4...de73), built from WRF
v4.6.1 module_bl_mynn.F b36c8b93...49452 with scalar_opt=1 live at
module_bl_mynn.F:6447-6456.  All 77 columns shared with the admitted base
dmp-mf.csv are bit-identical (manifest, measured), so this validator gates
both the live s_awqn* accumulation AND that the base outputs stay bit-exact.
"""

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
    MYNN_DMP_MF_QN_COLUMN_INPUTS,
    MYNN_DMP_MF_SCALAR_INPUTS,
    MYNN_DMP_MF_ZERO_OUTPUTS,
    mynn_dmp_mf,
)

RENAMED = {
    "qc_bl": "qc_bl_before", "cldfra_bl": "cldfra_bl_before",
    "vt": "vt_before", "vq": "vq_before",
}

#: s_awqn* families now live (per-level + _next top interface); s_awqke
#: keeps its tke_opt=0 structural zero.
QN_INTERFACE_OUTPUTS = (
    "s_awqnc", "s_awqni", "s_awqnwfa", "s_awqnifa", "s_awqnbca",
)

#: The inventory ``run_dmp_mf_mixscalars.F90`` emits.  Shape pinned first.
#: dead_probe is NO LONGER a dead probe under scalar_opt=1 (the qn
#: arguments are read); it rides as a qn-magnitude variation case.
EXPECTED_CASES = (
    "land_dry", "land_cumulus", "water_cumulus", "stable_off",
    "resolved_w", "flux_limited", "stochastic", "high_wind_thin",
    "deep_pblh", "cloud_base_capped", "fine_grid", "dead_probe",
)
#: Cases whose plume is dead in the BASE fixture too (ktop=0, s_aw all
#: zero) — their s_awqn* must be exactly zero, and every other case must
#: carry a live s_awqn*.
DEAD_PLUME_CASES = ("stable_off", "resolved_w")
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


def case_inputs(selected: list[dict[str, str]]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {
        name: np.asarray(
            [[np.float32(row[RENAMED.get(name, name)]) for row in selected]],
            dtype=np.float32,
        )
        for name in (*MYNN_DMP_MF_COLUMN_INPUTS,
                     *MYNN_DMP_MF_QN_COLUMN_INPUTS)
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
        values = case_inputs(selected)
        if not any(np.any(values[name] != 0.0)
                   for name in MYNN_DMP_MF_QN_COLUMN_INPUTS):
            raise SystemExit(f"{case} has identically-zero qn inputs")
        actual = mynn_dmp_mf(values, bl_mynn_mixscalars=1)
        case_summary: dict[str, dict[str, float | int]] = {}
        for name in (*MYNN_DMP_MF_LAYER_OUTPUTS, *MYNN_DMP_MF_ZERO_OUTPUTS):
            expected = np.asarray(
                [np.float32(row[name]) for row in selected],
                dtype=np.float32,
            )
            got = actual[name][0]
            case_summary[name] = _report(got, expected)
            np.testing.assert_array_equal(
                got, expected, err_msg=f"{case}/{name}"
            )
        for name in (*MYNN_DMP_MF_INTERFACE_OUTPUTS, *QN_INTERFACE_OUTPUTS):
            expected = np.asarray([
                *[np.float32(row[name]) for row in selected],
                np.float32(selected[-1][f"{name}_next"]),
            ], dtype=np.float32)
            got = actual[name][0]
            case_summary[name] = _report(got, expected)
            np.testing.assert_array_equal(
                got, expected, err_msg=f"{case}/{name}"
            )
        s_awqke = np.asarray(
            [np.float32(row["s_awqke"]) for row in selected],
            dtype=np.float32,
        )
        if np.any(s_awqke != 0.0) or np.any(actual["s_awqke"][0] != 0.0):
            raise AssertionError(
                f"{case}: s_awqke must stay a tke_opt=0 structural zero"
            )
        live = any(np.any(actual[name][0] != 0.0)
                   for name in QN_INTERFACE_OUTPUTS)
        if case in DEAD_PLUME_CASES and live:
            raise AssertionError(f"{case}: dead-plume case grew s_awqn*")
        if case not in DEAD_PLUME_CASES and not live:
            raise AssertionError(
                f"{case}: every s_awqn* is zero; the scalar_opt arm did "
                "not run"
            )
        for name in ("maxwidth", "ztop", "maxmf"):
            expected = np.asarray([np.float32(selected[0][name])],
                                  dtype=np.float32)
            case_summary[name] = _report(actual[name], expected)
            np.testing.assert_array_equal(
                actual[name], expected, err_msg=f"{case}/{name}"
            )
        if int(actual["ktop"][0]) != int(selected[0]["ktop"]):
            raise AssertionError(f"{case}/ktop mismatch")
        summary[case] = case_summary
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("PASS dmp-mf-mixscalars", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dmp-mf-mixscalars.csv")
