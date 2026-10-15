"""Every defined tendency, coefficient, renewed state and water diagnostic."""
from __future__ import annotations
import ctypes
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pytest

FIELDS = tuple(f"{name}{k}" for name in ("rhstt","ai","bi","ci","smc") for k in (1,2,3)) + (
    "cmc","runoff1","runoff2","runoff3","drip","srt_runoff1","srt_runoff2")


@lru_cache(maxsize=None)
def function(path,name):
    lib=ctypes.CDLL(str(path))
    fn=getattr(lib,name)
    fn.argtypes=[ctypes.c_void_p]*6
    fn.restype=None
    return fn


def invoke(path,name,values):
    out=np.full(22,np.float32(-123))
    function(path,name)(out.ctypes.data,*[a.ctypes.data for a in values])
    return out


def cases():
    f=np.float32
    z=np.asarray([-0.05,-0.15,-0.30,-0.50],f)
    rng=np.random.default_rng(480)
    for _ in range(400):
        smc=rng.uniform(0.084,0.439,4).astype(f)
        et=rng.uniform(0,1e-8,4).astype(f)
        root=rng.uniform(0,1e-7,4).astype(f)
        v=np.asarray([rng.uniform(1,120),rng.choice([0,1e-8,0.1,2,20]),0.439,5.25,0.084,
                      0.245e-5,0.228e-5,rng.uniform(0,1),0.5e-3,rng.uniform(0,1e-8),
                      rng.uniform(0,1e-8),rng.uniform(0,0.5e-3),rng.choice([0,1e-9,1e-6])],f)
        yield smc,z,et,root,v
    for sm in (0.066,0.084,0.2,0.439):
        for rain in (0,1e-38,1e-8,0.1,20):
            for uptake in (0,1e-38,1e-10,1e-7):
                v=np.asarray([30,rain,0.439,5.25,0.084,0.245e-5,0.228e-5,0.8,0.5e-3,
                              0,0,0,0],f)
                yield np.full(4,f(sm)),z,np.zeros(4,f),np.full(4,f(uptake)),v


@pytest.fixture(scope="module")
def libraries():
    root=os.environ.get("GPUWM_URBAN480_ORACLE_BUILD")
    if not root:pytest.skip("CPU oracle libraries not built; run the box check.py")
    return Path(root)/"water-host.so",Path(root)/"water-ref.so"


def test_all_defined_water_words(libraries):
    host,ref=libraries
    for index,v in enumerate(cases()):
        got=invoke(host,"water_host",v).view(np.uint32)
        want=invoke(ref,"water_ref",v).view(np.uint32)
        bad=np.flatnonzero(got!=want)
        assert not bad.size, {"case":index,"fields":[FIELDS[k] for k in bad],
                              "got":got[bad].tolist(),"want":want[bad].tolist()}


def test_six_carried_water_steps(libraries):
    host,ref=libraries
    for index,initial in enumerate(cases()):
        hg=[a.copy() for a in initial];wg=[a.copy() for a in initial]
        for step in range(6):
            got=invoke(host,"water_host",hg);want=invoke(ref,"water_ref",wg)
            np.testing.assert_array_equal(got.view(np.uint32),want.view(np.uint32),
                                          err_msg=f"water case={index} step={step}")
            hg[0][:3]=got[12:15];wg[0][:3]=want[12:15]
            hg[4][11]=got[15];wg[4][11]=want[15]
