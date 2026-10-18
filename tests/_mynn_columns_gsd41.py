"""84-column oracle of the operational fork's MYNN driver (GSD MYNN v4.1).

Fixtures ``tests/data/oracles/mynn/driver-columns-gsd41{,-asis}.bin.gz``
come from ``tools/mynn_pbl_gsd41_oracle/build_columns.sh``: the fork's own
``mynn_bl_driver`` (NOAA-EMC/HRRR v4.1.21 ``module_bl_mynn.F``; ``-asis``
the file unmodified, without the suffix the one :995 line squared, the
``bl_mynn_gsd41_unsquared_qtke = false`` default), HRRR's namelist identity,
over the twenty-one regimes x four variants of
``tools/mynn_pbl_gsd41_oracle/columns_gsd41.py``, twelve 20 s steps.

The file is float32 bits, little endian, no record markers:

* header: int32 magic, ncol, nz, nstep; float32 delt
* scal (ncol, 12): dx znt xland ts qsfc ps ust ch hfx qfx wspd rmol
* prof (ncol, 14, nz): dz u v w th qv qc qi p exner rho t3d rthraten qs
* per step: int32 initflag; the outputs (ncol, 16, nz) in
  ``_mynn_families_gsd41.PROFILE_OUTPUTS`` order; pblh float32, kpbl int32,
  maxmf float32, ktop_plume int32 (ncol each).

Each step's inputs are not stored: they are the starting columns, then the
previous step's outputs carried and the harness's between-step update
(``_mynn_families_gsd41.advance``, FP32 operation for operation as the
Fortran writes it).  A reader that rebuilt them differently would hand the
port inputs the fork never saw, and replay would fail rather than pass.

The snow profile ``qs`` never reaches the fork (its driver has no snow
argument).  :func:`integrate` hands it to the CUDA driver as
``sqs = qs/(1+qv)`` under ``flag_qs = True`` (the HRRR door's Thompson
configuration) unless told otherwise; it is held across steps, since the
fork returns no snow tendency.
"""

from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path

import numpy as np

import _mynn_families_gsd41 as G

ROOT = Path(__file__).parents[1]
ORACLE_DIR = ROOT / "tests" / "data" / "oracles" / "mynn"
F = np.float32
MAGIC = 0x4D594E31
SCALARS = ("dx", "znt", "xland", "ts", "qsfc", "ps", "ust", "ch", "hfx",
           "qfx", "wspd", "rmol")
PROFILES = ("dz", "u", "v", "w", "th", "qv", "qc", "qi", "p", "exner", "rho",
            "t3d", "rthraten", "qs")
STEP_STATE = ("qke", "tsq", "qsq", "cov", "el", "sh", "qc_bl", "cldfra_bl")
PROFILE_OUTPUTS = G.PROFILE_OUTPUTS
COLUMN_OUTPUTS = G.COLUMN_OUTPUTS
INDEX_OUTPUTS = G.INDEX_OUTPUTS


def column_names() -> list[str]:
    path = ROOT / "tools" / "mynn_pbl_gsd41_oracle" / "columns_gsd41.py"
    spec = importlib.util.spec_from_file_location("_columns_gsd41", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.NAMES)


def oracle_path(asis: bool) -> Path:
    return ORACLE_DIR / f"driver-columns-gsd41{'-asis' if asis else ''}.bin.gz"


class _Reader:
    def __init__(self, raw: bytes):
        self.raw, self.at = raw, 0

    def take(self, dtype, count, shape=None):
        value = np.frombuffer(self.raw, dtype, count, self.at)
        self.at += value.nbytes
        return value.reshape(shape) if shape else value


def load(asis: bool | None = None, path: Path | None = None) -> dict:
    """``{"delt", "ncol", "nz", "scal", "prof", "steps": [...]}``."""
    raw = gzip.decompress((path or oracle_path(bool(asis))).read_bytes())
    r = _Reader(raw)
    magic, ncol, nz, nstep = (int(x) for x in r.take("<i4", 4))
    assert magic == MAGIC, hex(magic)
    delt = F(r.take("<f4", 1)[0])
    scal = r.take("<f4", ncol * 12, (ncol, 12)).astype(F)
    prof = r.take("<f4", ncol * 14 * nz, (ncol, 14, nz)).astype(F)
    columns = {name: np.ascontiguousarray(prof[:, i])
               for i, name in enumerate(PROFILES)}
    columns.update({name: np.ascontiguousarray(scal[:, i])
                    for i, name in enumerate(SCALARS)})
    state = dict(columns)
    for name in STEP_STATE:
        state[name] = np.zeros((ncol, nz), F)
    state["pblh"] = np.zeros(ncol, F)
    state["kpbl"] = np.ones(ncol, np.int32)
    steps = []
    for _ in range(nstep):
        initflag = int(r.take("<i4", 1)[0])
        out = r.take("<f4", ncol * 16 * nz, (ncol, 16, nz)).astype(F)
        outputs = {name: np.ascontiguousarray(out[:, i])
                   for i, name in enumerate(PROFILE_OUTPUTS)}
        outputs["pblh"] = r.take("<f4", ncol).astype(F)
        outputs["kpbl"] = r.take("<i4", ncol).astype(np.int32)
        outputs["maxmf"] = r.take("<f4", ncol).astype(F)
        outputs["ktop_plume"] = r.take("<i4", ncol).astype(np.int32)
        steps.append((initflag, state, outputs))
        state = G.advance(state, outputs, delt)
    assert r.at == len(raw), (r.at, len(raw))
    return {"delt": delt, "ncol": ncol, "nz": nz, "scal": scal,
            "prof": prof, "steps": steps}


def integrate(driver, *, asis: bool, replay: bool, to_host=np.asarray,
              snow: bool = True, oracle: dict | None = None,
              **driver_kwargs):
    """``[(step, actual, expected), ...]`` with host arrays.

    ``snow``: hand the column's snow to the CUDA driver under
    ``flag_qs = True``; otherwise ``flag_qs = False`` and no snow.
    """
    oracle = oracle or load(asis)
    delt = oracle["delt"]
    results, fork, sm = [], None, None
    for step, (initflag, recorded, expected) in enumerate(oracle["steps"], 1):
        if replay or fork is None:
            fork = {name: np.array(value, copy=True)
                    for name, value in recorded.items()}
        if sm is None:
            sm = np.zeros_like(recorded["qv"])
        values = G.driver_values(fork, sm)
        if snow:
            values["sqs"] = (fork["qs"] / (F(1.0) + fork["qv"])).astype(F)
        raw = driver(values, initflag=initflag, delt=delt,
                     flag_qs=snow, bl_mynn_mixlength=2,
                     bl_mynn_version="gsd_41",
                     bl_mynn_gsd41_unsquared_qtke=asis, **driver_kwargs)
        actual = {name: np.asarray(to_host(value))
                  for name, value in raw.items()}
        sm = np.asarray(actual["sm"], F).copy()
        results.append((step, actual, expected))
        fork = G.advance(fork, actual, delt)
    return results


def _bits(x) -> np.ndarray:
    return np.ascontiguousarray(x, F).view(np.uint32)


def compare(results, names: list[str] | None = None) -> dict:
    """Per output: max ULP, max |diff|, bit-differing values, worst column.

    ``bits`` counts values whose float32 bit patterns differ (a -0 against
    +0 counts here and is 0 ULP); ``ulp`` is
    gpuwm.core.fp32_ulp.fp32_ulp_distance.
    """
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    table: dict[str, dict] = {}
    for step, actual, expected in results:
        for name in (*PROFILE_OUTPUTS, *COLUMN_OUTPUTS, *INDEX_OUTPUTS):
            want = expected[name]
            got = np.asarray(actual[name]).reshape(want.shape)
            row = table.setdefault(name, {"ulp": 0, "absdiff": 0.0,
                                          "bits": 0, "values": 0,
                                          "worst_column": None,
                                          "worst_step": None,
                                          "ulp_by_step": []})
            if name in INDEX_OUTPUTS:
                d = np.abs(got.astype(np.int64) - want.astype(np.int64))
                ulp, absd, bits = d, d.astype(float), d != 0
            else:
                got = got.astype(F)
                ulp = fp32_ulp_distance(got, want)
                absd = np.abs(got.astype(np.float64) - want.astype(np.float64))
                bits = _bits(got) != _bits(want)
            m = int(ulp.max())
            row["ulp_by_step"].append(m)
            row["bits"] += int(bits.sum())
            row["values"] += int(bits.size)
            row["absdiff"] = max(row["absdiff"], float(np.nanmax(absd)))
            if m > row["ulp"]:
                row["ulp"] = m
                col = int(np.unravel_index(int(np.argmax(ulp)), ulp.shape)[0])
                row["worst_column"] = names[col] if names else col
                row["worst_step"] = step
    return table


def max_ulp_table(results) -> dict[str, list[int]]:
    return {name: row["ulp_by_step"] for name, row in compare(results).items()}
