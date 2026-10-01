"""Bit oracle against readable base sedimentation functions."""
from pathlib import Path
import numpy as np
import pytest

@pytest.mark.parametrize("nz,ncol", [(49,65), (65,33)])
def test_sediment_bits(nz, ncol):
    cp = pytest.importorskip("cupy")
    from gpuwm.core.kernels import module_source, get_kernel
    from gpuwm.core.morrison_constants import rimed_ice_constants
    from gpuwm.core.morrison import _SEDIMENT_TPB
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device unavailable")
    source = (Path(__file__).parent / "data/morrison_sediment_base.cu").read_text()
    module = cp.RawModule(code=module_source("morrison") + source, options=("-std=c++17",))
    name = "morrison_sediment_64" if nz <= 64 else "morrison_sediment_256"
    rng = np.random.default_rng(746)
    shape = (nz,1,ncol)
    fields = []
    for category in range(5):
        q = rng.uniform(0,0.003,shape).astype("f4")
        q[rng.random(shape)<0.4] = 0
        fields.append(cp.asarray(q))
    fields += [cp.asarray(rng.uniform(1e3,1e7,shape).astype("f4")) for _ in range(5)]
    fields += [cp.full(shape,2.5e8,dtype="f4"), cp.full(shape,290,dtype="f4"),
               cp.full(shape,0.9,dtype="f4"), cp.full(shape,85000,dtype="f4"),
               cp.full(shape,1.1,dtype="f4"),
               cp.asarray(rng.uniform(50,500,shape).astype("f4"))]
    fields += [cp.zeros((1,ncol),dtype="f4") for _ in range(7)]
    reference = [a.copy() for a in fields]
    rimed = rimed_ice_constants(1)
    scalars = tuple(np.float32(x) for x in (15,rimed.ag,rimed.bg,rimed.rhog))
    scalars += (np.int32(nz),np.int32(1),np.int32(ncol))
    blocks = ((ncol+31)//32,)
    module.get_function("base_" + name)(blocks,(32,),tuple(reference)+scalars)
    get_kernel("morrison",name)(blocks,(_SEDIMENT_TPB,),tuple(fields)+scalars)
    for index,(actual,expected) in enumerate(zip(fields,reference)):
        np.testing.assert_array_equal(cp.asnumpy(actual).view("u4"),
                                      cp.asnumpy(expected).view("u4"),
                                      err_msg=f"buffer {index}")
