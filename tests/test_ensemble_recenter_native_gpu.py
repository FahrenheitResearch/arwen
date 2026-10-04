"""Exact native CPU/CUDA recentering, independent of member partitions."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from gpuwm.ensemble.recentered import FieldBounds, recenter_field


@pytest.mark.parametrize("population",[3,7,30,37])
def test_native_recenter_matches_cuda_and_every_output_partition(population):
    cp = pytest.importorskip("cupy")
    bridge = os.environ.get("GPUWM_ENSEMBLE_PREPARATION_BRIDGE")
    if not bridge:
        pytest.skip("requires the built native ensemble preparation bridge")
    random = np.random.default_rng(137+population)
    base = random.uniform(-1,1,(7,11)).astype("f4")
    base.flat[:6] = [-0.,0.,-1.,1.,np.nextafter(np.float32(1),np.float32(0)),np.float32(-.1)]
    donors = random.uniform(-4,4,(population,7,11)).astype("f4")
    ids = tuple(f"p{index:03d}" for index in range(population))
    bounds = FieldBounds("m s-1",.1,-1.,1.,1.7)
    kwargs = dict(donor_ids=ids,bounds=bounds,mapped_grid_sha256="a"*64)
    cpu,receipt = recenter_field(base,donors,selected_ids=ids,array_module=np,cpu_bridge=bridge,
                                 workers=2,**kwargs)
    device_base,device_donors = cp.asarray(base),cp.asarray(donors)
    words = 0
    for selection in (ids,tuple(reversed(ids[:min(20,population)])),(ids[-1],)):
        gpu,gpu_receipt = recenter_field(device_base,device_donors,selected_ids=selection,**kwargs)
        assert gpu_receipt["compiler_backend"] == "direct-nvrtc"
        assert gpu_receipt["compiler_options"] == ["-std=c++17","--fmad=false","--ftz=false"]
        assert receipt["native_bridge_sha256"] == hashlib.sha256(Path(bridge).read_bytes()).hexdigest()
        expected = cpu[[ids.index(member) for member in selection]]
        np.testing.assert_array_equal(cp.asnumpy(gpu).view("u4"),expected.view("u4"))
        words += expected.size
        single_cpu,_ = recenter_field(base,donors,selected_ids=selection,array_module=np,
                                      cpu_bridge=bridge,workers=1,**kwargs)
        np.testing.assert_array_equal(single_cpu.view("u4"),expected.view("u4"))
    for quiet in (FieldBounds("m s-1",.1,-1.,1.,0.),FieldBounds("m s-1",0.,-1.,1.,1.7)):
        expected,_ = recenter_field(base,donors,selected_ids=(ids[-1],),array_module=np,
                                    cpu_bridge=bridge,workers=1,**{**kwargs,"bounds":quiet})
        actual,_ = recenter_field(device_base,device_donors,selected_ids=(ids[-1],),
                                  **{**kwargs,"bounds":quiet})
        np.testing.assert_array_equal(expected[0].view("u4"),base.view("u4"))
        np.testing.assert_array_equal(cp.asnumpy(actual[0]).view("u4"),base.view("u4"))
    np.testing.assert_array_equal(cp.asnumpy(device_base).view("u4"),base.view("u4"))
    np.testing.assert_array_equal(cp.asnumpy(device_donors).view("u4"),donors.view("u4"))
    print("RECENTER_NATIVE_CUDA " + json.dumps({"population":population,"compared_words":words,
        "different_words":0,"native_bridge_sha256":hashlib.sha256(Path(bridge).read_bytes()).hexdigest(),
        "cuda_source_sha256":receipt["kernel_sha256"]},sort_keys=True))


@pytest.mark.parametrize("scale",[1e-38,1e30])
def test_native_recenter_preserves_float32_extremes_on_cuda(scale):
    cp = pytest.importorskip("cupy")
    bridge = os.environ.get("GPUWM_ENSEMBLE_PREPARATION_BRIDGE")
    if not bridge:
        pytest.skip("requires the built native ensemble preparation bridge")
    base = np.asarray([-1.,-.25,-0.,0.,.25,1.],dtype="f4")*np.float32(scale)
    donors = np.stack([base+np.float32(delta*scale) for delta in (-3.,-.5,2.,4.,6.)])
    bounds = FieldBounds("m s-1",scale*.13,-scale*10,scale*10,1.7)
    ids = tuple(f"p{index}" for index in range(5))
    kwargs = dict(donor_ids=ids,selected_ids=ids,bounds=bounds,mapped_grid_sha256="b"*64)
    expected,_ = recenter_field(base,donors,array_module=np,cpu_bridge=bridge,workers=1,**kwargs)
    actual,_ = recenter_field(cp.asarray(base),cp.asarray(donors),**kwargs)
    np.testing.assert_array_equal(cp.asnumpy(actual).view("u4"),expected.view("u4"))


def test_native_humidity_input_undershoots_freeze_without_creating_negative_members():
    cp = pytest.importorskip("cupy")
    bridge = os.environ.get("GPUWM_ENSEMBLE_PREPARATION_BRIDGE")
    if not bridge:
        pytest.skip("requires the built native ensemble preparation bridge")
    base = np.asarray([-.004,-.0001,-0.,0.,.0001,.002,.006],dtype="f4")
    donors = np.stack([np.zeros_like(base),np.full_like(base,.006),np.full_like(base,.008)])
    ids = ("p01","p02","p03")
    bounds = FieldBounds("kg kg-1",.005,0.,.1,1.,input_lower=-.005)
    kwargs = dict(donor_ids=ids,selected_ids=ids,bounds=bounds,mapped_grid_sha256="c"*64)
    expected,_ = recenter_field(base,donors,array_module=np,cpu_bridge=bridge,workers=2,**kwargs)
    actual,_ = recenter_field(cp.asarray(base),cp.asarray(donors),**kwargs)
    host = cp.asnumpy(actual)
    np.testing.assert_array_equal(host.view("u4"),expected.view("u4"))
    below = base < 0
    np.testing.assert_array_equal(host[:,below].view("u4"),np.broadcast_to(base[below],host[:,below].shape).view("u4"))
    assert np.all(host[:,~below] >= 0)
    assert np.any(host[:,base > 0] != base[base > 0])
    from dataclasses import replace
    for xp,source,donor in ((np,base,donors),(cp,cp.asarray(base),cp.asarray(donors))):
        off,_ = recenter_field(source,donor,array_module=xp,cpu_bridge=bridge,
                               **{**kwargs,"bounds":replace(bounds,amplitude=0.)})
        off_host = off if xp is np else cp.asnumpy(off)
        np.testing.assert_array_equal(off_host.view("u4"),np.broadcast_to(base,off_host.shape).view("u4"))


def test_native_relative_humidity_clipped_tails_preserve_cpu_cuda_words():
    cp = pytest.importorskip("cupy")
    bridge = os.environ.get("GPUWM_ENSEMBLE_PREPARATION_BRIDGE")
    if not bridge:pytest.skip("requires the built native ensemble preparation bridge")
    base=np.asarray([-4.,-0.,0.,20.,80.,100.,104.],"f4")
    donors=np.stack([np.full_like(base,10.),np.full_like(base,50.),np.full_like(base,90.)])
    bounds=FieldBounds("%",20.,0.,100.,1.5,input_lower=-4.,input_upper=104.)
    kwargs=dict(donor_ids=("p01","p02","p03"),selected_ids=("p01","p02","p03"),
                bounds=bounds,mapped_grid_sha256="d"*64)
    expected,_=recenter_field(base,donors,array_module=np,cpu_bridge=bridge,workers=2,**kwargs)
    actual,_=recenter_field(cp.asarray(base),cp.asarray(donors),**kwargs)
    np.testing.assert_array_equal(cp.asnumpy(actual).view("u4"),expected.view("u4"))
    for i in (0,6):assert all(row[i].tobytes()==base[i].tobytes() for row in expected)
    assert np.all((expected[:,1:6]>=0.) & (expected[:,1:6]<=100.))
