"""Sixty complete fire-model steps and exact device continuation controls."""

import json
import numpy as np
import pytest
from conftest import requires_gpu

from tools.sfire_wrf471_oracle.fixture import load, words


def _native_state(*, tiles=None):
    import cupy as cp
    from gpuwm.core.sfire import FireState, FireOptions
    from gpuwm.core.sfire_core import CoreOptions, IgnitionLine
    case = load("model_initial")
    xlo,xhi,ylo,yhi,xmem,_,ymem,_ = map(int,case["bounds"])
    domain = (xlo-xmem,xhi-xmem,ylo-ymem,yhi-ymem)
    state = FireState.from_static(*[case[key] for key in ("nfuel_cat","zsf","dzdxf","dzdyf")],
            float(case["dx"]),float(case["dy"]),domain=domain,tiles=tiles,
            coord_xf=case["coord_xf"],coord_yf=case["coord_yf"],
            options=FireOptions(core=CoreOptions(),fire_fmc_read=0),fmc_g=case["fmc_g"],
            ignitions=[IgnitionLine(*map(float,case["ignition"]))])
    for key,value in case.items():
        if key in state.data:
            state.data[key] = cp.asarray(value)
    return state,case


def _exact(actual, expected, name):
    import cupy as cp
    result = words(cp.asnumpy(actual),expected)
    print(json.dumps({name:result}))
    assert result["different_words"] == 0, (name,result)
    assert result["nonfinite_differences"] == 0, (name,result)


@requires_gpu
def test_native_full_fire_model_sixty_steps():
    state,initial = _native_state()
    for step in range(1,61):
        state.advance(float(initial["dt"]))
        if step not in (1,10,30,60):
            continue
        expected = load(f"model_step{step}")
        for key,value in expected.items():
            if key not in state.data:
                continue
            _exact(state.data[key][state.interior],value[state.interior],f"model_step{step}/{key}")


@requires_gpu
def test_fire_checkpoint_continuation_every_state_word():
    import cupy as cp
    from gpuwm.core.sfire import FireState
    state,case = _native_state()
    for _ in range(30):
        state.advance(float(case["dt"]))
    # A host checkpoint copy, then a fresh device restoration, exercises
    # serialization words without aliasing the original device buffers.
    arrays = {key:cp.asnumpy(value) for key,value in state.arrays().items()}
    metadata = json.loads(json.dumps(state.metadata()))
    resumed = FireState.from_restart(arrays,metadata)
    for key,value in state.arrays().items():
        _exact(resumed.arrays()[key],cp.asnumpy(value),"checkpoint_restore/"+key)
    for _ in range(30):
        state.advance(float(case["dt"]))
        resumed.advance(float(case["dt"]))
    assert state.metadata() == resumed.metadata()
    for key,value in state.arrays().items():
        _exact(resumed.arrays()[key],cp.asnumpy(value),"checkpoint_continue/"+key)


@requires_gpu
def test_fire_tile_stage_barriers_are_exact():
    import cupy as cp
    single,case = _native_state()
    tiled,_ = _native_state(tiles=[(4,23,4,20),(24,43,4,20),(4,23,21,37),(24,43,21,37)])
    for _ in range(30):
        single.advance(float(case["dt"]))
        tiled.advance(float(case["dt"]))
    for key,value in single.arrays().items():
        _exact(tiled.arrays()[key],cp.asnumpy(value),"tile_barriers/"+key)
    assert single.time_seconds == tiled.time_seconds


@requires_gpu
def test_freeze_correction_advances_before_and_consumes_after():
    import cupy as cp
    from dataclasses import replace
    before,case = _native_state()
    original_before,original_after = load("model_freeze_before"),load("model_freeze_after")
    selection = before.interior
    assert np.array_equal(original_before["lfn_in"][selection].view(np.uint32),
                          original_before["lfn_out"][selection].view(np.uint32))
    assert not np.array_equal(original_after["lfn_in"][selection].view(np.uint32),
                              original_after["lfn_out"][selection].view(np.uint32))
    before.options = replace(before.options,const_time=20.0)
    before.data["lfn"] = cp.asarray(original_before["lfn_in"])
    before.data["tign"].fill(0)
    before.time_seconds = 19.0
    start = before.data["lfn"].copy()
    before.advance(1.0)
    assert not bool(cp.array_equal(before.data["lfn"][selection],start[selection]))
    after,_ = _native_state()
    after.options = replace(after.options,const_time=20.0)
    after.data["lfn"] = cp.asarray(original_after["lfn_in"])
    after.data["tign"].fill(0)
    after.time_seconds = 21.0
    start = after.data["lfn"].copy()
    after.advance(1.0)
    _exact(after.data["lfn"][selection],cp.asnumpy(start[selection]),"corrected_freeze/spread")
    assert float(after.data["burnt_area_dt"][selection].max().item()) > 0
    print(json.dumps({"corrected_freeze":dict(before_threshold="advances",
          after_threshold="spread unchanged, fuel consumed",native_condition="inverted")}))
