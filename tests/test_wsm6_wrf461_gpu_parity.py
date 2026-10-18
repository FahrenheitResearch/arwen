"""The CUDA WSM6 kernel against unmodified WRF v4.6.1, in ULP.

THE BREAKAGE THIS PREVENTS
--------------------------
WSM6 (mp_physics = 6) carries the registry label ``supported``, whose
definition is agreement with WRF settled by committed oracle parity.  The
oracle existed (``tools/wsm6_wrf_oracle.F90`` linked against WRF's own
``mp_wsm6.F90``, frozen in ``tests/data/wsm6_wrf461_oracle.json``) and
``tools/wsm6_gpu_oracle.py`` grades the kernel against it, but no test ran
that grading and no ULP table was committed, so the kernel could move away
from WRF with nothing red (2026-10-05 parity sweep, row 30).

This file runs the four oracle columns (warm cloud, cold cloud, mixed with
graupel, mixed with hail) through the production launchers and holds every
output field to the per-field ULP ceiling committed in
``tests/data/wsm6_gpu_wrf461_ulp.json``, measured on an RTX 5090 (sm_120).
A ceiling is a measurement, not a tolerance chosen to pass: a new kernel
that moves a field further from WRF than the table fails, and one that
moves closer should lower the table in the same commit.

Large ULP counts in the table are near-zero cells, where one float32 ULP
of the input mass is many ULP of a 1e-11 kg/kg residue, and one
effective-radius branch: WRF's effectRad has a hard positive-mass test,
and a residue at the float32 floor can put one implementation on the
background radius (scenario 2, snow, 15.01 micron).  The absolute column
of the table states each.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests" / "data" / "wsm6_wrf461_oracle.json"
TABLE = ROOT / "tests" / "data" / "wsm6_gpu_wrf461_ulp.json"
SURFACE = ("rainnc", "rainncv", "snownc", "snowncv",
           "graupelnc", "graupelncv", "sr")


def _ordered(values) -> np.ndarray:
    bits = np.asarray(values, np.float32).view(np.int32).astype(np.int64)
    return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)


def _ulp(a, b) -> np.ndarray:
    return np.abs(_ordered(a) - _ordered(b))


def test_the_ulp_table_covers_every_oracle_scenario_and_field():
    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    fields = set(oracle["provenance"]["columns"]) | {"surface",
                                                     "reflectivity_before"}
    assert set(table["scenarios"]) == set(oracle["scenarios"])
    for key, rows in table["scenarios"].items():
        assert set(rows) == fields, key
        for name, row in rows.items():
            assert row["max_ulp"] >= 0 and row["max_abs"] >= 0.0, (key, name)
    assert table["authority"] == oracle["provenance"]["authority"]


@requires_gpu
def test_cuda_wsm6_is_within_the_committed_ulp_table_of_wrf461():
    import cupy as cp

    from gpuwm.core.refl import launch_refl10cm_wsm6
    from gpuwm.core.wsm6 import launch_wsm6
    from tools.wsm6_gpu_oracle import _initial_column

    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    columns = oracle["provenance"]["columns"]
    worse = []
    measured = {}
    for key, reference in oracle["scenarios"].items():
        initial = _initial_column(int(key))
        dev = {name: cp.asarray(value.reshape(8, 1, 1), dtype=cp.float32)
               for name, value in initial.items() if name != "temperature"}
        hail = int(reference["hail_opt"])
        surface = {name: cp.zeros((1, 1), dtype=cp.float32)
                   for name in SURFACE}
        effc = cp.full((8, 1, 1), np.float32(2.49), dtype=cp.float32)
        effi = cp.full((8, 1, 1), np.float32(4.99), dtype=cp.float32)
        effs = cp.full((8, 1, 1), np.float32(9.99), dtype=cp.float32)
        refl = cp.empty((8, 1, 1), dtype=cp.float32)
        temperature = cp.asarray(initial["temperature"].reshape(8, 1, 1),
                                 dtype=cp.float32)
        launch_refl10cm_wsm6(dev["qv"], dev["qr"], dev["qs"], dev["qg"],
                             temperature, dev["pressure"], refl,
                             hail_opt=hail)
        launch_wsm6(
            dev["theta"], dev["qv"], dev["qc"], dev["qr"], dev["qi"],
            dev["qs"], dev["qg"], dev["rho"], dev["pii"], dev["pressure"],
            dev["dz"], surface["rainnc"], surface["rainncv"],
            surface["snownc"], surface["snowncv"], surface["graupelnc"],
            surface["graupelncv"], surface["sr"], 30.0,
            effc=effc, effi=effi, effs=effs, hail_opt=hail)
        cp.cuda.runtime.deviceSynchronize()
        got = {"temperature": dev["theta"] * dev["pii"], "qv": dev["qv"],
               "qc": dev["qc"], "qi": dev["qi"], "qr": dev["qr"],
               "qs": dev["qs"], "qg": dev["qg"], "effc_micron": effc,
               "effi_micron": effi, "effs_micron": effs}
        levels = np.asarray(reference["levels"], np.float32)
        pairs = {name: (got[name][:, 0, 0].get(), levels[:, j])
                 for j, name in enumerate(columns)}
        pairs["surface"] = (
            np.asarray([surface[n].get().reshape(()) for n in SURFACE],
                       np.float32),
            np.asarray(reference["surface"], np.float32))
        pairs["reflectivity_before"] = (
            refl[:, 0, 0].get(),
            np.asarray(reference["reflectivity_before"], np.float32))
        measured[key] = {}
        for name, (actual, expected) in pairs.items():
            actual = np.asarray(actual, np.float32)
            assert np.isfinite(actual).all(), (key, name)
            ulp = int(_ulp(actual, expected).max())
            gap = float(np.abs(actual.astype(np.float64)
                               - expected.astype(np.float64)).max())
            measured[key][name] = (ulp, gap)
            ceiling = table["scenarios"][key][name]
            if ulp > ceiling["max_ulp"] or gap > ceiling["max_abs"]:
                worse.append(f"scenario {key} {name}: {ulp} ULP / {gap:.3g} "
                             f"against the table's {ceiling['max_ulp']} ULP / "
                             f"{ceiling['max_abs']:.3g}")
    assert not worse, (
        "the CUDA WSM6 kernel moved further from WRF v4.6.1 than the "
        "committed table: " + "; ".join(worse) + f"\nmeasured: {measured}")
