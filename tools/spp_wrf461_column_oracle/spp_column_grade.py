"""Grade WOOF's SPP consumers against the SPP column oracle, word for word.

Used by ``compare.py`` (the box run that writes the receipts) and by
``tests/test_spp_wrf461_column_oracle_gpu.py`` (the gate).  Every output
field is compared as float32 words; the result for each field is its worst
ULP distance, its worst absolute difference, the number of differing words
and the worst ULP per step and per regime.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _path in (str(HERE), str(ROOT), str(ROOT / "tests"),
              str(ROOT / "tools" / "gf_wrf461_oracle")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import spp_column_fixtures as columns  # noqa: E402


def _ulp(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    return np.asarray(fp32_ulp_distance(np.asarray(actual, np.float32),
                                        np.asarray(expected, np.float32)), np.int64)


def _absdiff(actual, expected) -> float:
    a = np.asarray(actual, np.float64)
    e = np.asarray(expected, np.float64)
    both = np.isfinite(a) & np.isfinite(e)
    return float(np.max(np.abs(a - e)[both], initial=0.0))


class _Ledger:
    def __init__(self):
        self.fields: dict[str, dict] = {}

    def add(self, name, actual, expected, *, step=None, groups=None, integer=False):
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        if integer:
            distance = np.abs(actual.astype(np.int64) - expected.astype(np.int64))
            absdiff = float(distance.max(initial=0))
        else:
            distance = _ulp(actual.reshape(expected.shape), expected)
            absdiff = _absdiff(actual.reshape(expected.shape), expected)
        entry = self.fields.setdefault(name, {
            "max_ulp": 0, "max_abs": 0.0, "words": 0, "words_differ": 0,
            "per_step": {}, "per_group": {}})
        entry["max_ulp"] = max(entry["max_ulp"], int(distance.max(initial=0)))
        entry["max_abs"] = max(entry["max_abs"], absdiff)
        entry["words"] += int(distance.size)
        entry["words_differ"] += int(np.count_nonzero(distance))
        if step is not None:
            entry["per_step"][str(step)] = max(entry["per_step"].get(str(step), 0),
                                               int(distance.max(initial=0)))
        if groups is not None:
            per_column = distance.reshape(distance.shape[0], -1).max(axis=1)
            for label, mask in groups.items():
                worst = int(per_column[mask].max(initial=0))
                entry["per_group"][label] = max(entry["per_group"].get(label, 0), worst)

    def summary(self) -> dict:
        return {"max_ulp": max((f["max_ulp"] for f in self.fields.values()), default=0),
                "max_abs": max((f["max_abs"] for f in self.fields.values()), default=0.0),
                "words": sum(f["words"] for f in self.fields.values()),
                "words_differ": sum(f["words_differ"] for f in self.fields.values()),
                "fields": self.fields}


def mynn_groups(cases: list[str]) -> dict[str, np.ndarray]:
    labels = [columns.regime(case) for case in cases]
    groups: dict[str, np.ndarray] = {}
    for axis in ("family", "terrain", "pattern"):
        for value in dict.fromkeys(label[axis] for label in labels):
            groups[f"{axis}:{value}"] = np.asarray([label[axis] == value for label in labels])
    return groups


def mynn_driver(backend: str):
    if backend == "gpu":
        import cupy as cp
        from gpuwm.core.mynn_pbl_gpu import mynn_bl_driver_cuda

        def driver(values, **kwargs):
            return mynn_bl_driver_cuda({name: cp.asarray(np.ascontiguousarray(value))
                                        for name, value in values.items()}, **kwargs)
        return driver, cp.asnumpy
    from gpuwm.core.mynn_pbl import mynn_bl_driver
    return mynn_bl_driver, np.asarray


def grade_mynn(npz: Path, mixlength: int, *, replay: bool, backend: str = "gpu") -> dict:
    """WRF v4.6.1 ``mynn_bl_driver`` with spp_pbl=1 against WOOF's driver.

    ``replay``: every step starts from WRF's own recorded inputs.  Otherwise
    only step 1 does and WOOF integrates the rest itself, so any rounding
    difference compounds.
    """
    import _mynn_families as F
    cases, steps = columns.mynn_steps(npz)
    first = steps[min(steps)]
    rstoch = np.asarray([[np.float32(row["rstoch"]) for row in block] for block in first],
                        np.float32)
    for blocks in steps.values():
        again = np.asarray([[np.float32(row["rstoch"]) for row in block] for block in blocks])
        assert np.array_equal(again.view(np.uint32), rstoch.view(np.uint32))
    base, to_host = mynn_driver(backend)

    def driver(values, **kwargs):
        return base({**values, "rstoch": rstoch}, spp_pbl=1, **kwargs)

    results = F.integrate(driver, mixlength, replay=replay, to_host=to_host, steps=steps)
    groups = mynn_groups(cases)
    ledger = _Ledger()
    for step, actual, expected in results:
        for name in (*F.PROFILE_OUTPUTS, "rqsblten", *F.COLUMN_OUTPUTS):
            ledger.add(name, actual[name], expected[name], step=step, groups=groups)
        for name in F.INDEX_OUTPUTS:
            ledger.add(name, np.asarray(actual[name]).reshape(-1), expected[name],
                       step=step, groups=groups, integer=True)
    out = ledger.summary()
    out.update(columns=len(cases), steps=len(steps), mixlength=mixlength,
               replay=replay, backend=backend,
               pattern_range=[float(rstoch.min()), float(rstoch.max())])
    return out


def gf_launch(module, fixture, patterns: np.ndarray) -> dict[str, np.ndarray]:
    """One ``gf_gfdrv_stage`` launch of the SPP-specialised GF unit.

    The GF fixture's own inputs, WRF's captured fzu words in the override
    slots (as tests/test_gf_gfdrv_cuda.py does), and the raw per-column
    closure patterns, which the kernel clips exactly as GFDRV does.
    """
    import cupy as cp
    from gf_field_lists import (DRV_IN_LEV, DRV_ISCA_FIELDS, DRV_LEV_FIELDS,
                                DRV_SCA_FIELDS, drv_scalar_inputs)
    from gpuwm.core.gf import gf_workspace_floats
    n, nz = fixture.ncol, columns.GF_NZ
    lvin = np.ascontiguousarray(np.stack([fixture.levels[name] for name in DRV_IN_LEV],
                                         axis=1).astype(np.float32))
    scin = np.ascontiguousarray(np.concatenate(
        [drv_scalar_inputs(fixture, True), np.asarray(patterns, np.float32)], axis=1))
    iin = np.ascontiguousarray(np.stack([fixture.surface[name] for name in
                                         ("kpbl", "ishallow", "ichoice")], axis=1).astype(np.int32))
    lev = cp.zeros((n, len(DRV_LEV_FIELDS), nz), dtype=cp.float32)
    sca = cp.zeros((n, len(DRV_SCA_FIELDS)), dtype=cp.float32)
    isc = cp.zeros((n, len(DRV_ISCA_FIELDS)), dtype=cp.int32)
    module.get_function("gf_gfdrv_stage")(((n + 63) // 64,), (64,), (
        cp.asarray(lvin), cp.asarray(scin), cp.asarray(iin), lev, sca, isc,
        cp.empty(gf_workspace_floats(nz, n), dtype=cp.float32),
        np.int32(1), np.int32(n), np.int32(nz)))
    cp.cuda.Stream.null.synchronize()
    lev, sca, isc = cp.asnumpy(lev), cp.asnumpy(sca), cp.asnumpy(isc)
    out = {name: lev[:, DRV_LEV_FIELDS.index(name), :] for name in DRV_LEV_FIELDS}
    out.update({name: sca[:, DRV_SCA_FIELDS.index(name)] for name in DRV_SCA_FIELDS})
    out.update({name: isc[:, DRV_ISCA_FIELDS.index(name)] for name in DRV_ISCA_FIELDS})
    return out


def gf_groups(fixture, oracle) -> dict[str, np.ndarray]:
    groups = {"converging_deep": oracle["ierr"] == 0,
              "rejected_deep": oracle["ierr"] != 0,
              "pattern_clipped": np.any(np.abs(oracle["spp"]) >= 1.0, axis=1),
              "pattern_inside": np.all(np.abs(oracle["spp"]) < 1.0, axis=1)}
    land = fixture.surface["xland"] < 1.5
    groups["land"], groups["water"] = land, ~land
    for arm in (0, 1):
        groups[f"ishallow:{arm}"] = fixture.surface["ishallow"] == arm
    return groups


def grade_gf(npz: Path) -> dict:
    """GF with a per-column closure pattern against CUP_gf, word for word.

    Graded at the CUP_gf seam (the out* tendencies after neg_check, pret and
    prets), plus kbcon/ktop of the deep cloud, which is where the closure
    pattern acts.
    """
    from gpuwm.core.spp_kernel_sources import load_spp_module
    from gpuwm.verify.gf_oracle import load_gf_oracle
    fixture = load_gf_oracle()
    oracle = columns.load_gf(npz)
    if not np.array_equal(oracle["key"], fixture.key.astype(np.int32)):
        raise ValueError("GF SPP fixture is not in the GF fixture's column order")
    actual = gf_launch(load_spp_module("gf"), fixture, oracle["spp"])
    groups = gf_groups(fixture, oracle)
    ledger = _Ledger()
    for name in columns.GF_SEAM_LEVELS:
        ledger.add(name, actual[name], oracle[name], groups=groups)
    for name in ("pret", "prets"):
        ledger.add(name, actual[name], oracle[name], groups=groups)
    for name in ("kbcon", "ktop"):
        ledger.add(name, actual[name], oracle[name], groups=groups, integer=True)
    out = ledger.summary()
    out.update(columns=int(fixture.ncol),
               converging_deep=int(np.count_nonzero(oracle["ierr"] == 0)),
               clipped_columns=int(np.count_nonzero(groups["pattern_clipped"])))
    return out


def gf_zero_pattern_identity() -> int:
    """Words where the SPP unit with a zero pattern differs from the plain unit."""
    from gpuwm.core.kernels import load_module
    from gpuwm.core.spp_kernel_sources import load_spp_module
    from gpuwm.verify.gf_oracle import load_gf_oracle
    import test_gf_gfdrv_cuda as gf
    fixture = load_gf_oracle()
    plain = gf._launch(load_module("gf"), fixture, 1)
    zero = gf_launch(load_spp_module("gf"), fixture, np.zeros((fixture.ncol, 4), np.float32))
    from gf_field_lists import DRV_ISCA_FIELDS, DRV_LEV_FIELDS, DRV_SCA_FIELDS
    differ = 0
    for key, names in (("lev", DRV_LEV_FIELDS), ("sca", DRV_SCA_FIELDS), ("isc", DRV_ISCA_FIELDS)):
        for j, name in enumerate(names):
            want = plain[key][:, j]
            got = zero[name]
            differ += int(np.count_nonzero(np.asarray(got).view(np.uint32)
                                           != np.asarray(want).view(np.uint32)))
    return differ
