"""Bit oracle for the hydrometeor-free finalization path and fallback."""
from pathlib import Path
import numpy as np
import pytest

def test_finalize_bits():
    cp=pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount(): pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    from gpuwm.core.kernels import module_source,get_kernel
    from gpuwm.core.morrison_constants import rimed_ice_constants
    source=(Path(__file__).parent / "data/morrison_finalize_base.cu").read_text()
    module=cp.RawModule(code=module_source("morrison")+source,options=("-std=c++17",))
    rng=np.random.default_rng(994);shape=(2,2,16)
    arrays=[cp.full(shape,280,dtype="f4"),cp.full(shape,0.005,dtype="f4")]
    for _ in range(5):
        q=rng.uniform(1e-6,0.001,shape).astype("f4");q.ravel()[:32]=0
        arrays.append(cp.asarray(q))
    arrays += [cp.asarray(rng.uniform(1e3,1e7,shape).astype("f4")) for _ in range(5)]
    arrays += [cp.full(shape,x,dtype="f4") for x in (1.1,0.9,85000,0,2.5e6,1004,0,0)]
    arrays[1].flat[0]=np.float32(0);arrays[1].flat[1]=np.float32(-0.0)
    arrays[1].flat[2]=np.float32(-0.001)
    for q in arrays[2:7]: q.flat[3]=np.float32(-0.0)
    arrays[0].flat[4]=np.float32(225);arrays[0].flat[5]=np.float32(310)
    expected=[x.copy() for x in arrays]
    tail=(np.float32(rimed_ice_constants(1).rhog),np.int32(64))
    module.get_function("base_morrison_finalize_levels")((1,),(64,),tuple(expected)+tail)
    get_kernel("morrison","morrison_finalize_levels")((1,),(64,),tuple(arrays)+tail)
    for index,(actual,want) in enumerate(zip(arrays,expected)):
        np.testing.assert_array_equal(cp.asnumpy(actual).view("u4"),cp.asnumpy(want).view("u4"),err_msg=f"buffer {index}")
