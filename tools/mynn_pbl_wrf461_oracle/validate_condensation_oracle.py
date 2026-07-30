#!/usr/bin/env python3
"""Compare the CPU MYNN cloud-PDF=2 condensation with official WRF CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.fp32_ulp import monotone_fp32_key
from gpuwm.core.mynn_pbl import (
    F, QPCT_PBL, QPCT_SFC, QPCT_TRP, mynn_condensation_default,
    mynn_qsat_blend,
)


ARRAY_COLUMNS = (
    "dz", "th", "thl", "qw", "qv", "qc", "qi", "qs", "p", "exner",
    "tsq", "qsq", "cov", "sh", "el", "rstoch",
    "vt_before", "vq_before", "sgm_before",
)
SCALAR_COLUMNS = ("xland", "dx", "pblh", "hfx", "rmo")
OUTPUT_COLUMNS = ("qc_bl", "qi_bl", "cldfra", "vt", "vq", "sgm")
# The first four columns are the original floor-bound regimes and must stay
# byte-for-byte where they are; the last three are the high-variance columns
# that exercise the square root, the qsat_tk*0.666 clip, and the coarse-dz
# inflation ramp.
FLOOR_CASES = ("dry_land", "humid_land", "liquid_cloud", "ice_anvil_water")
VARIANCE_CASES = (
    "high_variance_fine_grid",
    "high_variance_transition_grid",
    "high_variance_coarse_grid",
)
NZ = 20


#: The total-order FP32 key is shared rather than re-derived here.
#: Thirteen local copies of this two-line bit trick carried the same
#: sign error, which reported -0.0 as 2**32 ULP from +0.0; see
#: gpuwm/core/fp32_ulp.py for the measurement and the live case.
_ordered_bits = monotone_fp32_key


def _sigma_branch_census(fields: dict[str, np.ndarray], row: int) -> dict:
    """Count which of the three sigma constraints binds at each level.

    Mirrors the ordering in ``mym_condensation`` CASE(2): square root, then
    the ``qsat_tk*0.666`` clip, then the coarse-``dz`` inflation, then the
    ``qsat_tk*qpct`` floor.
    """

    zagl = F(0.0)
    dzm1 = F(0.0)
    census = {"clip": 0, "sqrt": 0, "floor": 0}
    weights = set()
    for k in range(NZ - 1):
        dz = F(fields["dz"][row, k])
        zagl = F(zagl + F(F(0.5) * F(dz + dzm1)))
        dzm1 = dz
        t = F(fields["th"][row, k] * fields["exner"][row, k])
        qsat_tk = mynn_qsat_blend(t, F(fields["p"][row, k]))
        raw = F(np.sqrt(np.float64(max(F(fields["qsq"][row, k]), F(0.0)))))
        clipped = min(raw, F(qsat_tk * F(0.666)))
        wt = F(max(F(F(500.0) - max(F(dz - F(100.0)), F(0.0))), F(0.0))
               / F(500.0))
        weights.add(float(wt))
        inflated = F(clipped + F(F(clipped * F(0.2)) * F(F(1.0) - wt)))
        qpct = F(F(QPCT_PBL * wt) + F(QPCT_TRP * F(F(1.0) - wt)))
        qpct = min(qpct, max(QPCT_SFC, F(F(QPCT_PBL * zagl) / F(500.0))))
        if F(qsat_tk * qpct) >= inflated:
            census["floor"] += 1
        elif clipped < raw:
            census["clip"] += 1
        else:
            census["sqrt"] += 1
    census["dz_weights"] = sorted(weights)
    return census


def _sanity(fields: dict[str, np.ndarray], ncase: int) -> None:
    assert all(np.isfinite(value).all() for value in fields.values())
    for name in ("qc_bl", "qi_bl", "cldfra"):
        assert np.all(fields[name] >= 0.0), name
        np.testing.assert_array_equal(fields[name][:, -1], 0.0)
    assert np.max(fields["cldfra"][0]) < 0.05
    assert np.max(fields["cldfra"][1]) > 0.20
    assert np.max(fields["qc_bl"][2]) > 0.0
    assert np.max(fields["qi_bl"][3]) > 0.0
    assert not np.array_equal(fields["vt_after"], fields["vt_before"])
    assert not np.array_equal(fields["vq_after"], fields["vq_before"])
    assert np.all(fields["sgm_after"][:, :-1] > 0.0)
    # The original four columns hand in sgm(kte)=0, which cannot separate
    # "untouched" from "zeroed"; the high-variance columns hand in nonzero
    # values that must come back out bit for bit.
    np.testing.assert_array_equal(fields["sgm_before"][:4, -1], 0.0)
    assert np.all(fields["sgm_before"][4:ncase, -1] > 0.0)
    np.testing.assert_array_equal(
        fields["sgm_after"][4:ncase, -1], fields["sgm_before"][4:ncase, -1]
    )


def main(path: str) -> None:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    assert cases == FLOOR_CASES + VARIANCE_CASES, cases
    ncase = len(cases)
    assert len(rows) == ncase * NZ, len(rows)
    numeric = tuple(name for name in rows[0] if name != "case")
    fields = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(ncase, NZ)
        for name in numeric
    }
    _sanity(fields, ncase)

    summary: dict[str, dict] = {}
    inputs: dict[str, np.ndarray] = {}
    for case_index, case in enumerate(cases):
        selected = [row for row in rows if row["case"] == case]
        if len(selected) != NZ:
            raise AssertionError(f"{case} has {len(selected)} levels")
        values = {
            name.removesuffix("_before"): np.asarray(
                [[np.float32(row[name]) for row in selected]],
                dtype=np.float32,
            )
            for name in ARRAY_COLUMNS
        }
        values["zw"] = np.asarray([[
            *[np.float32(row["zw"]) for row in selected],
            np.float32(selected[-1]["zw_next"]),
        ]], dtype=np.float32)
        for name in SCALAR_COLUMNS:
            values[name] = np.asarray(
                [np.float32(selected[0][name])], dtype=np.float32
            )
        if not inputs:
            inputs = {name: [value] for name, value in values.items()}
        else:
            for name, value in values.items():
                inputs[name].append(value)
        actual = mynn_condensation_default(values)
        case_summary: dict[str, dict[str, float | int]] = {}
        for name in OUTPUT_COLUMNS:
            recorded = f"{name}_after" if name in ("vt", "vq", "sgm") else name
            expected = np.asarray(
                [np.float32(row[recorded]) for row in selected],
                dtype=np.float32,
            )
            got = actual[name][0]
            absolute = np.abs(
                got.astype(np.float64) - expected.astype(np.float64)
            )
            relative = absolute / np.maximum(
                np.abs(expected.astype(np.float64)), 1.0e-30
            )
            ulp = np.abs(_ordered_bits(got) - _ordered_bits(expected))
            case_summary[name] = {
                "max_abs": float(absolute.max(initial=0.0)),
                "max_rel": float(relative.max(initial=0.0)),
                "max_ulp": int(ulp.max(initial=0)),
            }
            np.testing.assert_array_equal(
                got, expected, err_msg=f"{case}/{name}"
            )
        summary[case] = {
            "outputs": case_summary,
            "sigma_branches": _sigma_branch_census(fields, case_index),
        }

    # Coverage gate: the original columns are wholly floor-bound, and each new
    # column must exercise both the clip and the bare square root.
    for case in FLOOR_CASES:
        census = summary[case]["sigma_branches"]
        assert census["floor"] == NZ - 1, (case, census)
    for case in VARIANCE_CASES:
        census = summary[case]["sigma_branches"]
        assert census["clip"] > 0 and census["sqrt"] > 0, (case, census)
    weights = {
        case: summary[case]["sigma_branches"]["dz_weights"]
        for case in VARIANCE_CASES
    }
    assert weights[VARIANCE_CASES[0]] == [1.0], weights
    assert 0.0 < weights[VARIANCE_CASES[1]][0] < 1.0, weights
    assert weights[VARIANCE_CASES[2]] == [0.0], weights

    # Discrimination gate: driving qsq to exactly zero must move every output
    # on the high-variance columns.  Before they existed it moved none.
    stacked = {
        name: np.concatenate(value, axis=0) for name, value in inputs.items()
    }
    baseline = mynn_condensation_default(stacked)
    zeroed = dict(stacked)
    zeroed["qsq"] = np.zeros_like(stacked["qsq"])
    probe = mynn_condensation_default(zeroed)
    discrimination = {}
    for name in OUTPUT_COLUMNS:
        moved = ~np.all(baseline[name] == probe[name], axis=1)
        discrimination[name] = [
            case for case, flag in zip(cases, moved.tolist()) if flag
        ]
        # Every output has to become qsq-sensitive somewhere.  qi_bl only
        # responds where liq_frac < 1, so the warm fine-grid column cannot
        # carry it; the two colder high-variance columns must.
        assert set(discrimination[name]) & set(VARIANCE_CASES), (
            name, discrimination[name]
        )
    for case in VARIANCE_CASES:
        responsive = {
            name for name in OUTPUT_COLUMNS if case in discrimination[name]
        }
        assert responsive >= {"sgm", "cldfra", "vt", "vq"}, (case, responsive)
    print(json.dumps({
        "status": "PASS",
        "cases": summary,
        "qsq_zero_changes": discrimination,
    }, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: validate_condensation_oracle.py condensation.csv"
        )
    main(sys.argv[1])
