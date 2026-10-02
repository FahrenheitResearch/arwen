"""Direct compiled WRF v4.7.1 diffusion receipts and boundary regressions.

The fixture contains the actual WRF routine outputs, not NumPy expressions
derived from the same CUDA transcription. Numerical differences are measured
word for word by vertical_compare.py; a tolerance does not erase them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from conftest import requires_gpu

FIXTURES = Path(__file__).parent / "data/wrf471_diffusion/vertical"
TOOLS = Path(__file__).resolve().parents[1] / "tools/wrf_diffusion_oracle"


def _fixture(name):
    receipt = json.loads((FIXTURES/"vertical-fixtures.json").read_text())
    case = next(c for c in receipt["cases"] if c["case"]==name)
    with np.load(FIXTURES/case["file"],allow_pickle=False) as data:
        arrays = {k[3:]:data[k].copy() for k in data.files if k.startswith("in_")}
        coefficients = {k:data["coefficient_"+k].copy() for k in ("kmh","kmv","khv")}
        deformation = {k:data["deformation_"+k].copy() for k in ("d11","d22","d12")}
        reference = {k:data[k].copy() for k in data.files if k.startswith("ref_")}
        bn2 = data["bn2"].copy()
    return arrays,case["metadata"],coefficients,deformation,bn2,reference


def _outer(a):
    return np.concatenate((a[:,0,:].ravel(),a[:,-1,:].ravel(),
                           a[:,:,0].ravel(),a[:,:,-1].ravel()))


def test_vertical_diffusion_wrf471_fixture_hashes_and_nonzero_motion():
    receipt = json.loads((FIXTURES/"vertical-fixtures.json").read_text())
    assert receipt["wrf_release"]=="4.7.1"
    assert len(receipt["cases"])==14
    for case in receipt["cases"]:
        path = FIXTURES/case["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==case["sha256"]
    # A zero W fixture cannot distinguish a broken vertical-W operator.
    arrays,_,_,_,_,reference = _fixture("evolved_real_periodic")
    assert np.count_nonzero(arrays["w"])
    assert np.count_nonzero(reference["ref_vertical_w"])


@pytest.mark.gpu
@requires_gpu
def test_vertical_diffusion_wrf471_all_output_word_pins(tmp_path):
    sys.path.insert(0,str(TOOLS))
    from vertical_compare import compare
    expected = [json.loads(p.read_text()) for p in sorted(FIXTURES.glob("vertical-comparison*.json"))]
    got = compare(FIXTURES,tmp_path/"comparison.json")
    assert expected
    assert all(len(got["cases"])==len(pin["cases"]) for pin in expected)
    for actual in got["cases"]:
        pins=[next(case for case in variant["cases"] if case["case"]==actual["case"]) for variant in expected]
        # The entire word array must match its measured GPU pin. The
        # separately recorded WRF mismatch counts and ULP distances remain
        # visible and exact; there is no blanket numerical tolerance.
        assert any(actual["array_sha256"]==pin["array_sha256"] and actual["fields"]==pin["fields"]
                   for pin in pins),(actual["case"],actual["fields"])


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name",["real_open","steep_open","southern_open","evolved_real_open"])
@pytest.mark.parametrize("isfflx",[0,1,2])
def test_tke_rhs_wrf471_open_boundary_words(name,isfflx):
    sys.path.insert(0,str(TOOLS))
    from vertical_gpu import tke_gpu
    arrays,meta,coef,deform,bn2,reference = _fixture(name)
    got = tke_gpu(arrays,meta,deform,coef,bn2,isfflx=isfflx,c_k=meta["c_k"],dt=meta["dt"])
    # Exact WRF words on every excluded mass row, including signed zero.
    for routine in ("tke_shear","tke_buoyancy","tke_dissip","tke_rhs"):
        expected = reference[f"ref_{routine}_flux{isfflx}"]
        assert not np.any(_outer(expected).view(np.uint32))
        np.testing.assert_array_equal(_outer(got[routine]).view(np.uint32),
                                      _outer(expected).view(np.uint32))
    assert np.count_nonzero(got["tke_rhs"][:,1:-1,1:-1])


@pytest.mark.gpu
@requires_gpu
def test_tke_vertical_self_diffusion_wrf471_open_boundary_words():
    import cupy as cp
    from gpuwm.core import tke_budget
    from gpuwm.core.dycore import prepare_fixed_tendencies
    from gpuwm.verify.diffusion_oracle import device_state
    sys.path.insert(0,str(TOOLS))
    from vertical_gpu import _config
    arrays,meta,_,_,_,reference = _fixture("real_open")
    cfg = _config(meta,c_k=meta["c_k"],dt=meta["dt"],tke_heat_flux=.24,
                  tke_drag_coefficient=.0013,diff_6th_opt=0)
    state = device_state(arrays,cf=tuple(meta[k] for k in ("cf1","cf2","cf3")))
    state.mub2d = state.mut
    state.mup = cp.zeros_like(state.mut)
    state.mup0 = cp.zeros_like(state.mut)
    state.has_msf = True
    state.qr = cp.zeros_like(state.alt)
    state.qs = cp.zeros_like(state.alt)
    state.qg = cp.zeros_like(state.alt)
    for key in ("u","v","w","php","thp","qv","qc","qi","qr","qs","qg","tke"):
        setattr(state,key+"0",getattr(state,key).copy())
    for field,target in (("u","ru_t"),("v","rv_t"),("w","rw_t"),("thp","rth_t")):
        setattr(state,target,cp.zeros_like(getattr(state,field)))
    prepare_fixed_tendencies(state,cfg)
    rtke = cp.asnumpy(state.scratch(state.p.shape,"smag_rtke"))
    vertical = cp.asnumpy(tke_budget.term(state,cfg,"diffusion_v"))
    expected = _outer(reference["ref_vertical_s_tke"]).view(np.uint32)
    assert not np.any(expected)
    np.testing.assert_array_equal(_outer(vertical).view(np.uint32),expected)
    np.testing.assert_array_equal(_outer(rtke).view(np.uint32),expected)
    assert np.count_nonzero(vertical[:,1:-1,1:-1])
