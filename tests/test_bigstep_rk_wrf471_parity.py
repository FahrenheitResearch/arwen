"""Every-output word measurements against compiled WRF v4.7.1 RK routines.

The asserted ULP values are exact measured pins, not acceptance tolerances.
Changed halos, skipped output rows, NaN payloads, or improved arithmetic all
change the table. The numerical comparison runs through production launchers.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.verify.bigstep_rk_oracle import (RK_ORACLE_DIR,load_rk_oracle,
                                           measure_rk_case,load_rk_tendency_oracle,
                                           measure_rk_tendency_case)


def test_rk_wrf471_fixture_words_and_provenance():
    for filename in ("rk-cases.npz","rk-tendency.npz","rk-tendency-stored-theta.npz"):
        meta=json.loads((RK_ORACLE_DIR/filename).with_suffix(".json").read_text())
        assert hashlib.sha256((RK_ORACLE_DIR/filename).read_bytes()).hexdigest()==meta["fixture_sha256"]
        assert meta["state_sha256"]==hashlib.sha256((RK_ORACLE_DIR/"state-real.npz").read_bytes()).hexdigest()
    arrays,meta=load_rk_oracle()
    assert sum(case["kind"]=="dry" for case in meta["cases"])==5
    assert sum(case["kind"]=="scalar" for case in meta["cases"])==7
    for case in meta["cases"]:
        prefix=f"{case['kind']}{case['id']}_"
        names=("a","b","mu") if case["kind"]=="dry" else ("s1","s2","decomp")
        for name in names:
            expected=arrays[prefix+name+"_wrf"]
            assert expected.dtype==np.float32
            assert expected.shape==arrays[prefix+name].shape


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("case_id",range(5))
def test_rk_addtend_dry_matches_compiled_wrf471(case_id):
    arrays,meta=load_rk_oracle()
    case=next(c for c in meta["cases"] if c["kind"]=="dry" and c["id"]==case_id)
    pin=json.loads((RK_ORACLE_DIR/"rk-ulp-table.json").read_text())
    assert measure_rk_case(arrays,case)==pin["measurements"]["dry:"+case["name"]]


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("case_id",range(7))
def test_rk_update_scalar_matches_compiled_wrf471(case_id):
    arrays,meta=load_rk_oracle()
    case=next(c for c in meta["cases"] if c["kind"]=="scalar" and c["id"]==case_id)
    pin=json.loads((RK_ORACLE_DIR/"rk-ulp-table.json").read_text())
    assert measure_rk_case(arrays,case)==pin["measurements"]["scalar:"+case["name"]]


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("case_id",range(3))
@pytest.mark.parametrize("filename,key",[
    ("rk-tendency.npz","rk_tendency_measurements"),
    ("rk-tendency-stored-theta.npz","rk_tendency_stored_theta_measurements"),
])
def test_rk_tendency_records_every_compiled_wrf471_output(case_id,filename,key):
    """Record the unresolved arithmetic association and cqw representation.

    This is a drift gate, not a claim that the full routine is bit-identical.
    The complete difference is declared alongside the fixture and receipt.
    """
    arrays,meta=load_rk_tendency_oracle(filename)
    case=next(c for c in meta["cases"] if c["id"]==case_id)
    pin=json.loads((RK_ORACLE_DIR/"rk-ulp-table.json").read_text())
    assert measure_rk_tendency_case(arrays,case)==pin[key][case["name"]]


@requires_gpu
@pytest.mark.gpu
def test_rk_oracle_detects_changed_timestep():
    """The compiled reference catches a different scalar update interval."""
    arrays,meta=load_rk_oracle()
    case=next(c for c in meta["cases"] if c["kind"]=="scalar" and c["id"]==1).copy()
    case["dt"]+=0.125
    pin=json.loads((RK_ORACLE_DIR/"rk-ulp-table.json").read_text())
    measured=measure_rk_case(arrays,case)
    assert measured!=pin["measurements"]["scalar:"+case["name"]]
    assert measured["s2"]["different_words"]>0
