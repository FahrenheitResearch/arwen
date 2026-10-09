"""Multi-step oracle of the operational fork's MYNN driver (GSD MYNN v4.1).

Fixtures ``gpuwm/data/mynn/oracle/driver-families-gsd41{,-asis}.csv.gz``
come from ``tools/mynn_pbl_gsd41_oracle/run_driver_families_gsd41.F90`` run
against NOAA-EMC/HRRR v4.1.21 ``module_bl_mynn.F`` (``-asis``: the file
unmodified; without the suffix: the one :995 line build.sh squares, the
``bl_mynn_gsd41_unsquared_qtke = false`` default), with HRRR's namelist
identity, over the six column families of the WRF v4.6.1 families oracle.

The fork reads mixing ratios and returns mixing-ratio tendencies.
:func:`integrate` hands the port what the runtime hands it
(gpuwm.core.mynn_pbl_runtime: ``mynn_wrapper_to_specific``'s ``q/(1+qv)``
beside the original mixing ratios) and applies the harness's between-step
update itself.  The fork carries no Sm3D, so the port's own ``sm`` rides
from step to step in both modes.
"""

from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path

import numpy as np

from _mynn_families import bit_mismatch_table

ORACLE_DIR = Path(__file__).parents[1] / "gpuwm" / "data" / "mynn" / "oracle"
FAMILIES = ("convective_day", "stable_night", "stratocumulus",
            "shallow_cumulus", "cold_pool", "high_wind")
NZ = 30
F = np.float32
PROFILE_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
                   "rqiblten", "exch_h", "exch_m", "qke", "tsq", "qsq", "cov",
                   "el", "sh", "qc_bl", "cldfra_bl")
OUTPUT_CSV = {"el": "el_pbl", "sh": "sh3d"}
COLUMN_OUTPUTS = ("pblh", "maxmf")
INDEX_OUTPUTS = ("kpbl", "ktop_plume")


def oracle_path(asis: bool, mixlength: int = 2) -> Path:
    if mixlength == 1:
        return ORACLE_DIR / "driver-families-gsd41-ml1.csv.gz"
    return ORACLE_DIR / f"driver-families-gsd41{'-asis' if asis else ''}.csv.gz"


def load(asis: bool, mixlength: int = 2) -> dict[int, list[list[dict]]]:
    with gzip.open(oracle_path(asis, mixlength), "rb") as stream:
        rows = list(csv.DictReader(io.TextIOWrapper(stream, "ascii")))
    assert tuple(dict.fromkeys(row["case"] for row in rows)) == FAMILIES
    steps = {}
    for step in sorted({int(row["step"]) for row in rows}):
        selected = [row for row in rows if int(row["step"]) == step]
        steps[step] = [[row for row in selected if row["case"] == case]
                       for case in FAMILIES]
    return steps


def _f32(blocks, key):
    return np.asarray([[F(row[key]) for row in block] for block in blocks], F)


def _col(blocks, key, dtype=F):
    if dtype is np.int32:
        return np.asarray([int(block[0][key]) for block in blocks], np.int32)
    return np.asarray([F(block[0][key]) for block in blocks], F)


def step_inputs(blocks) -> dict[str, np.ndarray]:
    """The fork's recorded inputs of one step (mixing ratios, fork state)."""
    v = {name: _f32(blocks, name) for name in (
        "dz", "u", "v", "w", "th", "qv", "qc", "qi", "p", "exner", "rho",
        "t3d", "rthraten")}
    v.update({name: _f32(blocks, f"{name}_in") for name in (
        "qke", "tsq", "qsq", "cov", "el", "sh", "qc_bl", "cldfra_bl")})
    v.update({name: _col(blocks, name) for name in (
        "dx", "xland", "ts", "ps", "ust", "hfx", "qfx", "wspd", "rmol")})
    v["pblh"] = _col(blocks, "pblh_in")
    v["kpbl"] = _col(blocks, "kpbl_in", np.int32)
    return v


def step_outputs(blocks) -> dict[str, np.ndarray]:
    out = {name: _f32(blocks, OUTPUT_CSV.get(name, name))
           for name in PROFILE_OUTPUTS}
    out.update({name: _col(blocks, name) for name in COLUMN_OUTPUTS})
    out.update({name: _col(blocks, name, np.int32) for name in INDEX_OUTPUTS})
    return out


def driver_values(fork: dict, sm) -> dict[str, np.ndarray]:
    """What gpuwm.core.mynn_pbl_runtime hands mynn_bl_driver_cuda."""
    denominator = (F(1.0) + fork["qv"]).astype(F)
    values = {
        "dz": fork["dz"], "u": fork["u"], "v": fork["v"], "w": fork["w"],
        "th": fork["th"], "p": fork["p"], "exner": fork["exner"],
        "rho": fork["rho"], "tk": fork["t3d"],
        "sqv": (fork["qv"] / denominator).astype(F),
        "sqc": (fork["qc"] / denominator).astype(F),
        "sqi": (fork["qi"] / denominator).astype(F),
        "sqs": np.zeros_like(fork["qv"]),
        "qv": fork["qv"], "qc": fork["qc"], "qi": fork["qi"],
        "qke": fork["qke"], "tsq": fork["tsq"], "qsq": fork["qsq"],
        "cov": fork["cov"], "el": fork["el"], "sh": fork["sh"], "sm": sm,
        "qc_bl": fork["qc_bl"], "qi_bl": np.zeros_like(fork["qv"]),
        "cldfra_bl": fork["cldfra_bl"],
        "uoce": np.zeros_like(fork["dx"]), "voce": np.zeros_like(fork["dx"]),
        "pblh": fork["pblh"], "rmol": fork["rmol"], "kpbl": fork["kpbl"],
    }
    for name in ("dx", "xland", "ts", "ps", "ust", "hfx", "qfx", "wspd"):
        values[name] = fork[name]
    return values


def advance(fork: dict, out: dict, delt) -> dict:
    """The harness's between-step update, FP32 operation for operation."""
    nxt = {name: np.array(value, copy=True) for name, value in fork.items()}
    zero = F(0.0)
    for field, tend, floor in (("u", "rublten", False), ("v", "rvblten", False),
                               ("th", "rthblten", False),
                               ("qv", "rqvblten", True),
                               ("qc", "rqcblten", True),
                               ("qi", "rqiblten", True)):
        value = (nxt[field] + delt * np.asarray(out[tend], F)).astype(F)
        nxt[field] = np.maximum(value, zero) if floor else value
    nxt["t3d"] = (nxt["th"] * nxt["exner"]).astype(F)
    nxt["rho"] = (nxt["p"] / (F(287.0) * nxt["t3d"])).astype(F)
    for name in ("qke", "tsq", "qsq", "cov", "el", "sh", "qc_bl", "cldfra_bl"):
        nxt[name] = np.asarray(out[name], F).copy()
    nxt["pblh"] = np.asarray(out["pblh"], F).reshape(-1).copy()
    nxt["kpbl"] = np.asarray(out["kpbl"], np.int32).reshape(-1).copy()
    return nxt


def integrate(driver, *, asis: bool, replay: bool, to_host=np.asarray,
              steps: dict | None = None, bl_mynn_mixlength: int = 2,
              **driver_kwargs):
    """``[(step, actual, expected), ...]`` with host arrays."""
    results, fork, sm = [], None, None
    for step, blocks in (load(asis, bl_mynn_mixlength) if steps is None else steps).items():
        recorded = step_inputs(blocks)
        if replay or fork is None:
            fork = recorded
        if sm is None:
            sm = np.zeros_like(recorded["qv"])
        initflag = int(blocks[0][0]["initflag"])
        delt = F(blocks[0][0]["delt"])
        raw = driver(driver_values(fork, sm), initflag=initflag, delt=delt,
                     flag_qs=False, bl_mynn_mixlength=bl_mynn_mixlength,
                     bl_mynn_version="gsd_41",
                     bl_mynn_gsd41_unsquared_qtke=asis, **driver_kwargs)
        actual = {name: np.asarray(to_host(value)) for name, value in raw.items()}
        sm = np.asarray(actual["sm"], F).copy()
        results.append((step, actual, step_outputs(blocks)))
        fork = advance(fork, actual, delt)
    return results


def max_ulp_table(results) -> dict[str, list[int]]:
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    table: dict[str, list[int]] = {}
    for _, actual, expected in results:
        for name in (*PROFILE_OUTPUTS, *COLUMN_OUTPUTS):
            got = np.asarray(actual[name], F).reshape(expected[name].shape)
            table.setdefault(name, []).append(
                int(fp32_ulp_distance(got, expected[name]).max()))
        for name in INDEX_OUTPUTS:
            got = np.asarray(actual[name]).astype(np.int32).reshape(-1)
            table.setdefault(name, []).append(
                int(np.abs(got - expected[name]).max()))
    return table
