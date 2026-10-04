"""Complete CUDA lake columns against the byte-unmodified WRF oracle."""
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ORACLE = Path(__file__).resolve().parents[1] / "gpuwm/data/lake/oracle"


def _assert_words(actual, expected, *, err_msg=""):
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32), err_msg=err_msg)


@pytest.mark.gpu
@requires_gpu
def test_complete_lake_cuda_column_history_matches_wrf():
    import cupy as cp
    from gpuwm.core.kernels import get_kernel, module_options, module_source

    # Breakage: x<0?-x:x preserves -0 and signed NaNs, and an FTZ ABS
    # instruction can erase a smallest subnormal. Probe the production
    # helper under the production flags against native Fortran words.
    abs_words=np.loadtxt(ORACLE / "abs-fortran.txt",dtype=str)
    bits32=np.asarray([int(row[0],16) for row in abs_words],np.uint32)
    bits64=np.asarray([int(row[2],16) for row in abs_words],np.uint64)
    expected32=np.asarray([int(row[1],16) for row in abs_words],np.uint32)
    expected64=np.asarray([int(row[3],16) for row in abs_words],np.uint64)
    probe=r'''
extern "C" __global__ void lake_abs_probe(const unsigned int* a,
    const unsigned long long* b,unsigned int* c,unsigned long long* d) {
    int i=threadIdx.x;
    if(i>=12)return;
    c[i]=__float_as_uint(lake_abs(__uint_as_float(a[i])));
    d[i]=__double_as_longlong(lake_abs(__longlong_as_double(b[i])));
}
'''
    module=cp.RawModule(code=module_source("lake")+probe,options=module_options("lake"))
    actual32=cp.empty(12,cp.uint32);actual64=cp.empty(12,cp.uint64)
    module.get_function("lake_abs_probe")((1,),(32,),
        (cp.asarray(bits32),cp.asarray(bits64),actual32,actual64))
    np.testing.assert_array_equal(cp.asnumpy(actual32),expected32)
    np.testing.assert_array_equal(cp.asnumpy(actual64),expected64)

    with np.load(ORACLE / "columns.npz") as reference:
        seed = cp.asarray(reference["seed"])
        forcing = cp.asarray(reference["forcing"])
        n = seed.shape[1]
        columns = cp.zeros((131, n), cp.float32)
        static = cp.zeros((71, n), cp.float32)
        output = cp.zeros((9, n), cp.float32)
        errors = cp.zeros(n, cp.int32)
        init = get_kernel("lake", "lake_init_columns")
        step = get_kernel("lake", "lake_step_columns")
        init((1,), (32,), (n, seed, columns, static, 1, 1, np.float32(50), errors))
        assert not cp.asnumpy(errors).any()
        _assert_words(cp.asnumpy(columns), reference["initial_state"])
        _assert_words(cp.asnumpy(static), reference["reference_static"])
        for i in range(300):
            step((1,), (32,), (n, forcing, columns, static, output, np.float32(30), errors))
            assert not cp.asnumpy(errors).any()
            _assert_words(cp.asnumpy(columns), reference["trace_state"][i], err_msg=f"step {i+1} state")
            _assert_words(cp.asnumpy(output), reference["trace_output"][i], err_msg=f"step {i+1} output")


@pytest.mark.gpu
@requires_gpu
def test_complete_lake_cuda_default_depth_controls_match_wrf():
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    with np.load(ORACLE / "initialization.npz") as reference:
        seed=cp.asarray(reference["seed"]);n=seed.shape[1]
        columns=cp.zeros((131,n),cp.float32);static=cp.zeros((71,n),cp.float32);errors=cp.zeros(n,cp.int32)
        for i,default in enumerate(reference["defaults"]):
            get_kernel("lake","lake_init_columns")((1,),(32,),
                (n,seed,columns,static,0,0,np.float32(default),errors))
            assert not cp.asnumpy(errors).any()
            _assert_words(cp.asnumpy(columns),reference["reference_state"][i])
            _assert_words(cp.asnumpy(static),reference["reference_static"][i])


@pytest.mark.gpu
@requires_gpu
def test_lake_runtime_preserves_nonlake_cells_and_frozen_mask_conversion():
    import cupy as cp
    from gpuwm.core.lake import initialize_lake

    with np.load(ORACLE / "columns.npz") as reference:
        seed,forcing=reference["seed"],reference["forcing"]
        shape=(1,16)

        def field(values, fill=0):
            result=cp.full(shape,fill,cp.float32)
            result[0,:12]=cp.asarray(values)
            return result

        f={"lakemask":field(np.ones(12)),"isltyp":field(seed[0],3),
           "tsk":field(seed[2],299),"snow":field(seed[3]),"xice":field(seed[4]),
           "ivgtyp":cp.full(shape,15,cp.int32),"xland":cp.ones(shape,cp.float32),
           "glw":field(forcing[7]),"emiss":field(forcing[8]),
           "swdown":field(forcing[10]),"albedo":field(forcing[11],0.21)}
        for name in ("hfx","lh","grdflx","qfx","t2","th2","q2"):
            f[name]=cp.full(shape,-999,cp.float32)
        before={name:value.copy() for name,value in f.items()}
        owner=initialize_lake(f,latitude=field(forcing[12]),lake_depth=field(seed[1]),iswater=16)
        frozen=seed[4]>.5
        np.testing.assert_array_equal(cp.asnumpy(f["ivgtyp"])[0,:12],np.where(frozen,16,15))
        np.testing.assert_array_equal(cp.asnumpy(f["xland"])[0,:12],np.where(frozen,2,1))
        np.testing.assert_array_equal(cp.asnumpy(f["xice"])[0,:12],np.where(frozen,0,seed[4]))
        atmosphere={name:field(forcing[row])[None] for name,row in
                    (("temperature",0),("dz",3),("qv",4),("u",5),("v",6))}
        atmosphere["p_interface"]=cp.stack((field(forcing[1]),field(forcing[2])))
        owner.step(f,atmosphere,dt=30,precipitation=field(forcing[9]))
        for name,original in before.items():
            _assert_words(cp.asnumpy(f[name])[:,12:],cp.asnumpy(original)[:,12:],err_msg=name)
        for row,name in enumerate(("hfx","lh","grdflx","tsk","qfx","t2","th2","q2","albedo")):
            _assert_words(cp.asnumpy(f[name])[0,:12],reference["trace_output"][0,row],err_msg=name)
        _assert_words(cp.asnumpy(f["lake_columns"])[:,0,:12],reference["trace_state"][0])
        work_names=("columns","static","latitude","forcing","output","errors")
        work_before={name:getattr(owner,name).data.ptr for name in work_names}
        owner.refresh_columns(f)
        assert {name:getattr(owner,name).data.ptr for name in work_names} == work_before
        # Breakage: a tile's changed occupancy must not reuse the previous
        # tile's sparse indices, geometry or latitude.
        f["lakemask"][...]=0
        owner.invalidate_columns()
        owner.step(f,{},dt=30,precipitation=cp.zeros(shape,cp.float32))
        assert owner.column_count==0
