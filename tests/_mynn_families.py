"""Multi-step MYNN driver oracle over six column families (WRF v4.6.1).

The fixtures ``gpuwm/data/mynn/oracle/driver-families{,2}.csv.gz`` come from
``tools/mynn_pbl_wrf461_oracle/run_driver_families.F90`` run against the
unmodified WRF v4.6.1 ``module_bl_mynn.F`` (gfortran 13.3.0 -O0
-ffp-contract=off, glibc 2.39).  Six families -- convective day, stable night
with dew, marine stratocumulus, shallow cumulus, a sub-freezing cold pool
with fog, ice and frost, and high wind -- are integrated for twelve 20 s
steps: a cold start, then eleven warm steps.  Between steps the harness
applies the driver's own tendencies (``x = x + delt*tend``, species floored
at zero) and recomputes ``t3d = th*exner`` and ``rho = p/(287*t3d)``.

:func:`integrate` repeats that loop around any driver with the
``mynn_bl_driver`` call shape, so a port is graded both per step (each step
started from the oracle's own recorded inputs, ``replay=True``) and as a
free-running integration from step 1 alone (``replay=False``), where any
rounding difference compounds.
"""

from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path

import numpy as np

from gpuwm.core.mynn_pbl import (
    MYNN_DRIVER_LAYER_INPUTS, MYNN_DRIVER_SCALAR_INPUTS, MYNN_DRIVER_STATE,
)

ORACLE_DIR = Path(__file__).parents[1] / "gpuwm" / "data" / "mynn" / "oracle"
FAMILIES = ("convective_day", "stable_night", "stratocumulus",
            "shallow_cumulus", "cold_pool", "high_wind")
NZ = 30
LAYER_CSV = {"sqv": "sqv3d", "sqc": "sqc3d", "sqi": "sqi3d", "sqs": "sqs3d",
             "tk": "t3d"}
OUTPUT_CSV = {"el": "el_pbl", "sh": "sh3d", "sm": "sm3d"}
PROFILE_OUTPUTS = (
    "rublten", "rvblten", "rthblten", "rqvblten", "rqcblten", "rqiblten",
    "rqsblten", "dozone",
    "exch_h", "exch_m", "qke", "tsq", "qsq", "cov", "el", "sh", "sm",
    "qc_bl", "qi_bl", "cldfra_bl",
)
COLUMN_OUTPUTS = ("pblh", "rmol", "maxwidth", "maxmf", "ztop_plume")
INDEX_OUTPUTS = ("kpbl", "ktop_plume")
#: Tendency applied to each advanced layer field, in the harness's order.
ADVANCE = (("u", "rublten", False), ("v", "rvblten", False),
           ("th", "rthblten", False), ("sqv", "rqvblten", True),
           ("sqc", "rqcblten", True), ("sqi", "rqiblten", True),
           ("sqs", "rqsblten", True))


def oracle_path(mixlength: int) -> Path:
    suffix = "" if mixlength == 1 else str(mixlength)
    return ORACLE_DIR / f"driver-families{suffix}.csv.gz"


def load(mixlength: int, *, path: Path | None = None) -> dict[int, list[list[dict]]]:
    """``{step: [rows of family 0, ..., rows of family 5]}``."""
    with gzip.open(oracle_path(mixlength) if path is None else path, "rb") as stream:
        rows = list(csv.DictReader(io.TextIOWrapper(stream, "ascii")))
    assert tuple(dict.fromkeys(row["case"] for row in rows)) == FAMILIES
    steps: dict[int, list[list[dict]]] = {}
    for step in sorted({int(row["step"]) for row in rows}):
        selected = [row for row in rows if int(row["step"]) == step]
        steps[step] = [[row for row in selected if row["case"] == case]
                       for case in FAMILIES]
        assert all(len(block) == NZ for block in steps[step])
    return steps


def _f32(blocks, key):
    return np.asarray([[np.float32(row[key]) for row in block]
                       for block in blocks], dtype=np.float32)


def _col(blocks, key, dtype=np.float32):
    if dtype is np.int32:
        return np.asarray([int(block[0][key]) for block in blocks], np.int32)
    return np.asarray([np.float32(block[0][key]) for block in blocks], dtype)


def step_inputs(blocks) -> dict[str, np.ndarray]:
    """The recorded incoming values of one step, in driver-input form."""
    values = {name: _f32(blocks, LAYER_CSV.get(name, name))
              for name in MYNN_DRIVER_LAYER_INPUTS}
    values.update({name: _f32(blocks, f"{name}_in")
                   for name in MYNN_DRIVER_STATE})
    values.update({name: _col(blocks, name)
                   for name in MYNN_DRIVER_SCALAR_INPUTS})
    values["pblh"] = _col(blocks, "pblh_in")
    values["rmol"] = _col(blocks, "rmol_in")
    values["kpbl"] = _col(blocks, "kpbl_in", np.int32)
    return values


def step_outputs(blocks) -> dict[str, np.ndarray]:
    """What WRF wrote back on one step."""
    out = {name: _f32(blocks, OUTPUT_CSV.get(name, name))
           for name in PROFILE_OUTPUTS}
    out.update({name: _col(blocks, name) for name in COLUMN_OUTPUTS})
    out.update({name: _col(blocks, name, np.int32) for name in INDEX_OUTPUTS})
    return out


def advance(values: dict, out: dict, delt: np.float32) -> dict:
    """The harness's between-step update, FP32 operation for operation."""
    nxt = {name: np.array(value, copy=True) for name, value in values.items()}
    zero = np.float32(0.0)
    for field, tend, floor in ADVANCE:
        updated = (nxt[field] + delt * np.asarray(out[tend], np.float32)
                   ).astype(np.float32)
        nxt[field] = np.maximum(updated, zero) if floor else updated
    nxt["tk"] = (nxt["th"] * nxt["exner"]).astype(np.float32)
    nxt["rho"] = (nxt["p"] / (np.float32(287.0) * nxt["tk"])
                  ).astype(np.float32)
    for name in MYNN_DRIVER_STATE:
        nxt[name] = np.asarray(out[name], np.float32).copy()
    nxt["pblh"] = np.asarray(out["pblh"], np.float32).reshape(-1).copy()
    nxt["rmol"] = np.asarray(out["rmol"], np.float32).reshape(-1).copy()
    nxt["kpbl"] = np.asarray(out["kpbl"], np.int32).reshape(-1).copy()
    return nxt


def integrate(driver, mixlength: int, *, replay: bool, to_host=np.asarray,
              steps: dict | None = None, **driver_kwargs):
    """Run ``driver`` over every recorded step.

    Returns ``[(step, actual, expected), ...]`` with host arrays.  With
    ``replay`` every step starts from the oracle's recorded inputs; without
    it only step 1 does and later steps start from the port's own output.
    """
    steps = load(mixlength) if steps is None else steps
    results = []
    values = None
    for step, blocks in steps.items():
        recorded = step_inputs(blocks)
        if replay or values is None:
            values = recorded
        initflag = int(blocks[0][0]["initflag"])
        delt = np.float32(blocks[0][0]["delt"])
        raw = driver(values, initflag=initflag, delt=delt, flag_qs=True,
                     bl_mynn_mixlength=mixlength, **driver_kwargs)
        actual = {name: np.asarray(to_host(value)) for name, value in raw.items()}
        results.append((step, actual, step_outputs(blocks)))
        values = advance(values, actual, delt)
    return results


def max_ulp_table(results) -> dict[str, list[int]]:
    """Per output, the worst FP32 ULP distance on each step."""
    from gpuwm.core.fp32_ulp import fp32_ulp_distance

    table: dict[str, list[int]] = {}
    for _, actual, expected in results:
        for name in (*PROFILE_OUTPUTS, *COLUMN_OUTPUTS):
            got = np.asarray(actual[name], np.float32).reshape(
                expected[name].shape)
            table.setdefault(name, []).append(
                int(fp32_ulp_distance(got, expected[name]).max()))
        for name in INDEX_OUTPUTS:
            got = np.asarray(actual[name]).astype(np.int32).reshape(-1)
            table.setdefault(name, []).append(
                int(np.abs(got - expected[name]).max()))
    return table


def bit_mismatch_table(results) -> dict[str, list[int]]:
    """Count differing words for every recorded output, without zero folding."""
    table: dict[str, list[int]] = {}
    for _, actual, expected in results:
        for name, want in expected.items():
            got = np.asarray(actual[name])
            assert got.dtype == want.dtype, (name, got.dtype, want.dtype)
            assert got.shape == want.shape, (name, got.shape, want.shape)
            got_words = np.ascontiguousarray(got).view(np.uint32)
            want_words = np.ascontiguousarray(want).view(np.uint32)
            table.setdefault(name, []).append(
                int(np.count_nonzero(got_words != want_words)))
    return table
