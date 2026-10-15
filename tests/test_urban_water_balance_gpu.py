"""Run the production SLUCM entry and its coupled surface driver."""
import numpy as np
import pytest
from conftest import requires_gpu


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("canopy",["wrf_471","wrf_480"])
@pytest.mark.parametrize("variant",["ucm-nlcd-imp2","ucm-lcz-imp2ahalh"])
def test_bare_default_evaporates_during_weak_and_subnormal_rain(variant,canopy):
    from gpuwm.core.urban_ucm import run_columns
    from gpuwm.verify.urban_ucm_oracle import load_variant,port_params,_inputs,_state_from,_flatten
    rows,table,switches = load_variant(variant)
    params = port_params(table,switches)
    if canopy == "wrf_480":
        from dataclasses import replace
        from gpuwm.core.urban_tables import load_urban_params
        from gpuwm.core.urban_ucm import pack_params
        full = pack_params(load_urban_params(1,int("lcz" in variant),generation="wrf_480"))
        flags = full.switches.copy()
        flags[:10] = params.switches[:10]
        flags[11:13] = 0
        params = replace(full,switches=flags)
    index = np.repeat(np.flatnonzero((rows["scenario"]==1)&(rows["step"]==1))[:1],8)
    incoming = _inputs(rows,index)
    if canopy == "wrf_480":
        incoming.update(declin=np.zeros(8,np.float32),cosz=np.full(8,.8,np.float32),
                        xlat=np.full(8,40.,np.float32))
    incoming["rain"] = np.asarray([0.,np.uint32(1).view(np.float32),1.e-8,0.5,4.,100.,1.e-8,1.e-8],np.float32)
    state = _state_from(rows,index,"_in")
    for key in ("drelr_urb2d","drelb_urb2d","drelg_urb2d"):
        state[key].fill(np.float32(0.0001))
    for key in ("flxhumr_urb2d","flxhumb_urb2d","flxhumg_urb2d"):
        state[key].fill(np.float32(0.00003))
        state[key][7] = np.float32(-0.00003)
    for key in ("drelr_urb2d","drelb_urb2d","drelg_urb2d"):
        state[key][6] = np.float32(1.e-7)
    out,new,codes = run_columns(params,incoming,state)
    oldout,old,oldcodes = run_columns(params,incoming,state,generation="wrf_471")
    assert not np.any(codes) and not np.any(oldcodes)
    for i,surface in enumerate("rbg"):
        key = f"drel{surface}_urb2d"
        assert np.all(new[key][:3] < state[key][:3])
        assert old[key][1:3].tobytes() == state[key][1:3].tobytes()
        assert new[key][0].tobytes() == old[key][0].tobytes()
        cap = params.globals_[19+i]
        assert np.all((new[key]>=np.float32(0.)) & (new[key]<=cap))
        assert new[key][6] == np.float32(0.)
        assert new[key][7] >= state[key][7]
        assert np.isfinite(new[f"flxhum{surface}_urb2d"]).all()
        assert not np.array_equal(new[f"flxhum{surface}_urb2d"][1:3],old[f"flxhum{surface}_urb2d"][1:3])
    for key,a in _flatten(out,new).items():
        b = _flatten(oldout,old)[key]
        assert a[0].tobytes() == b[0].tobytes(),key
    assert np.isfinite(out["lh"]).all()


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("lsm",[2,4])
@pytest.mark.parametrize("generation",["wrf_480","wrf_471"])
def test_real_surface_driver_reaches_the_store_generation(lsm,generation):
    import cupy as cp
    from dataclasses import replace
    from test_urban_default_off_identity import _small_driver
    state,cfg,driver = _small_driver(sf_urban_physics=1,sf_surface_physics=lsm,
                                    urban_water_balance=generation)
    urban = driver.urban
    values = dict(urban.params.values)
    values["IMP_SCHEME"] = np.asarray(2,np.int32)
    urban.params = replace(urban.params,values=values)
    for surface in "rbg":
        urban.fields[f"drel{surface}_urb2d"].fill(cp.float32(0.0001))
        urban.fields[f"flxhum{surface}_urb2d"].fill(cp.float32(0.00003))
    driver.fields["rainbl"].fill(cp.float32(1.e-10))
    driver.compute(state,cfg)
    mask = cp.asnumpy(urban.urban_mask).astype(bool)
    for surface in "rbg":
        water = cp.asnumpy(urban.fields[f"drel{surface}_urb2d"])[mask]
        if generation == "wrf_480":
            assert np.all(water < np.float32(0.0001))
        else:
            assert np.all(water == np.float32(0.0001))
    assert np.isfinite(cp.asnumpy(driver.fields["lh"])).all()
