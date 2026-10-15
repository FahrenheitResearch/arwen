"""Wordwise grading against the unmodified WRF 4.8 ROOT_UPTAKE.

Build libraries on a CPU box with tools/urban_wrf480_oracle/check.py.
No fixture lookup or reimplementation of Fortran determines expected values.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pytest


@lru_cache(maxsize=None)
def function(path, name):
    lib = ctypes.CDLL(str(path))
    fn = getattr(lib, name)
    f, p = ctypes.c_float, ctypes.c_void_p
    fn.argtypes = [p, ctypes.c_int, p, f, f, f, f, p, p, f, f, f]
    fn.restype = None
    return fn


def invoke(path, name, values):
    fn = function(path,name)
    smc, maximum, dry, radius, lai, dz, zsoil, energy, fraction, width = values
    out = np.full(len(smc), np.float32(-123))
    fn(out.ctypes.data, len(smc), smc.ctypes.data, maximum, dry, radius, lai,
       dz.ctypes.data, zsoil.ctypes.data, energy, fraction, width)
    return out


def cases():
    f = np.float32
    rng = np.random.default_rng(480)
    for n in (1, 2, 3, 4, 8, 16):
        dz = np.asarray([f(0.05)] * n, np.float32)
        z = -np.cumsum(dz, dtype=np.float32)
        for moisture in (-0.1, 0, 0.066, 0.084, 0.2, 0.2898, 0.439, 0.6):
            center = f(moisture)
            for m in (np.nextafter(center, f(-np.inf)), center,
                      np.nextafter(center, f(np.inf))):
                yield (np.full(n,m,f), f(0.439),f(0.066),f(2),f(1.5),dz,z,f(400),f(0.25),f(12))
        for knob in (3, 4, 8, 9):
            for value in (0, -1, np.nextafter(f(0), f(1))):
                v = [np.full(n,f(0.2)),f(0.439),f(0.066),f(2),f(1.5),dz,z,f(400),f(0.25),f(12)]
                v[knob] = f(value)
                yield tuple(v)
        # ALPHABAR transition around 1e-12 and very small forcing/fractions.
        for scale in (0, 1e-14, 1e-12, 1e-10):
            yield (np.full(n,f(0.2)),f(0.439),f(0.066),f(2),f(1.5),dz*f(scale),z,f(400),f(0.25),f(12))
        for energy in (0, -1, 1e-38, 1e-20, 1, 400):
            yield (np.full(n,f(0.2)),f(0.439),f(0.066),f(2),f(1.5),dz,z,f(energy),f(0.25),f(12))
        for _ in range(400):
            dz = rng.uniform(0.001, 0.5, n).astype(f)
            z = -np.cumsum(dz, dtype=f)
            yield (rng.uniform(-0.1,0.6,n).astype(f),f(0.439),f(0.066),
                   f(rng.uniform(0.01,4)),f(rng.uniform(0.01,8)),dz,z,
                   f(rng.uniform(-100,1000)),f(rng.uniform(0.001,1)),f(rng.uniform(0.1,50)))


@pytest.fixture(scope="module")
def libraries():
    root = os.environ.get("GPUWM_URBAN480_ORACLE_BUILD")
    if not root:
        pytest.skip("CPU oracle libraries not built; run tools/urban_wrf480_oracle/check.py on a box")
    root = Path(root)
    return root / "root-host.so", root / "root-ref.so", root / "root-mutant.so"


def test_every_defined_root_word(libraries):
    host, ref, _ = libraries
    for index, values in enumerate(cases()):
        want = invoke(ref, "root_ref", values)
        got = invoke(host, "root_host", values)
        np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32),
                                      err_msg=f"root case={index}")


def test_planted_one_bit_is_rejected(libraries):
    _, ref, mutant = libraries
    values = next(v for v in cases() if np.any(
        invoke(ref,"root_ref",v).view(np.uint32) != invoke(mutant,"root_host",v).view(np.uint32)))
    want = invoke(ref,"root_ref",values).view(np.uint32)
    got = invoke(mutant,"root_host",values).view(np.uint32)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(got,want)
