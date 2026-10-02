"""The native explicit launch path faces compiled WRF, every output word.

This freezes measured outputs and their exact word-level discrepancy from
the compiled Fortran fixture. No acceptance tolerance is applied. The
full-theta diagnostic case separates the engine's scalar coordinate from
WRF's canonical theta-minus-300 coordinate; headline cases use the latter.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.verify.smallstep_oracle import ORACLE_DIR
from gpuwm.verify.smallstep_horizontal_oracle import (
    horizontal_port_outputs, load_horizontal_fixture,
)


def test_horizontal_compiled_fixture_pins():
    pins=json.loads((ORACLE_DIR/"horizontal.sha256.json").read_text(encoding="utf-8"))
    assert set(pins)=={"horizontal.npz","horizontal.json","horizontal-baseline.json"}
    for filename,expected in pins.items():
        assert hashlib.sha256((ORACLE_DIR/filename).read_bytes()).hexdigest()==expected,filename


@pytest.fixture(scope="module")
def horizontal_measurements():
    baseline=json.loads((ORACLE_DIR/"horizontal-baseline.json").read_text(encoding="utf-8"))
    with np.load(ORACLE_DIR/"horizontal.npz",allow_pickle=False) as fixture:
        result=[]
        for name,data,cfg,reference,full_theta in load_horizontal_fixture(ORACLE_DIR/"horizontal.npz"):
            actual=horizontal_port_outputs(data,cfg,full_theta_probe=full_theta)
            frozen={key:fixture[name+"__gpu_"+key].copy() for key in actual}
            result.append((name,actual,reference,frozen,baseline[name]))
    return result


def _assert_outputs(measurements,keys):
    for name,actual,reference,frozen,baseline in measurements:
        for key in keys:
            got,want=actual[key],reference[key]
            np.testing.assert_array_equal(got.view(np.uint32),frozen[key].view(np.uint32),
                                          err_msg=name+" "+key+" GPU output words drifted")
            d=fp32_ulp_distance(got,want)
            assert int(d.max())==baseline[key]["max_ulp"],(name,key,"ULP measurement drifted")
            assert int(np.count_nonzero(got.view(np.uint32)!=want.view(np.uint32)))==baseline[key]["different_words"],(name,key,"word count drifted")


@pytest.mark.gpu
@requires_gpu
def test_advance_uv_against_compiled_wrf471(horizontal_measurements):
    _assert_outputs(horizontal_measurements,("u_pp","v_pp"))


@pytest.mark.gpu
@requires_gpu
def test_advance_mu_t_against_compiled_wrf471(horizontal_measurements):
    _assert_outputs(horizontal_measurements,("mu_pp","mudf","th_pp","ww_pp","t_ave","muave","muts"))


@pytest.mark.gpu
@requires_gpu
def test_horizontal_word_gate_rejects_missing_pressure_term():
    # Prove the word gate can see a dynamical defect: remove the native
    # full-geopotential pressure-gradient term only in a diagnostic compile.
    for name,data,cfg,reference,full_theta in load_horizontal_fixture(ORACLE_DIR/"horizontal.npz"):
        if name=="real_initial_map":
            break
    else:
        raise AssertionError("real-state pressure-gradient fixture missing")
    actual=horizontal_port_outputs(data,cfg,full_theta_probe=full_theta,drop_pressure_term=True)
    baseline=json.loads((ORACLE_DIR/"horizontal-baseline.json").read_text(encoding="utf-8"))[name]
    with np.load(ORACLE_DIR/"horizontal.npz",allow_pickle=False) as fixture:
        frozen={key:fixture[name+"__gpu_"+key].copy() for key in actual}
    with pytest.raises(AssertionError,match="GPU output words drifted"):
        _assert_outputs([(name,actual,reference,frozen,baseline)],("u_pp","v_pp"))
