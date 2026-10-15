"""Admission stays closed until the WRF aerosol producer is present."""
from gpuwm.core.aer_opt2 import load_tables, validate_options


def test_aer_opt2_tables_are_pinned():
    tables = load_tables()
    assert tables['rrtmg']['raod'].shape == (3, 8, 14)
    assert tables['goddard']['raod'].shape == (3, 8, 11)


def test_options_match_wrf_bounds():
    validate_options(aer_type=2, options=(1, 1, 1, 1), values=(0.12, 1.3, 0.85, 0.9))

import json
import pytest


def test_one_bit_spectral_coefficient_change_is_refused(tmp_path):
    from gpuwm.core.aer_opt2 import TABLE_PATH
    import numpy as np
    data=json.loads(TABLE_PATH.read_bytes())
    value=np.asarray([data['rrtmg']['raod'][0][0][0]],np.float32)
    value.view(np.uint32)[0] ^= np.uint32(1)
    data['rrtmg']['raod'][0][0][0]=float(value[0])
    path=tmp_path/'changed.json'
    path.write_text(json.dumps(data,separators=(',',':'))+'\n')
    with pytest.raises(ValueError,match='spectral table hash differs'):
        load_tables(path)


@pytest.mark.parametrize('typ', [1,2,3])
@pytest.mark.parametrize('opts', [(1,1,1,1),(2,2,2,2),(2,3,3,3),(1,2,3,2)])
def test_wrf_option_combinations(typ,opts):
    validate_options(aer_type=typ,options=opts)


@pytest.mark.parametrize('opts,values', [((0,1,1,1),(0.1,1.3,0.85,0.9)),
                                      ((1,4,1,1),(0.1,1.3,0.85,0.9)),
                                      ((1,1,1,1),(-0.1,1.3,0.85,0.9)),
                                      ((1,1,1,1),(0.1,1.3,1.01,0.9)),
                                      ((1,1,1,1),(0.1,1.3,0.85,-0.1))])
def test_invalid_wrf_selectors_and_fixed_values(opts,values):
    with pytest.raises(ValueError):
        validate_options(options=opts,values=values)


def test_fixed_aerosol_is_admitted_on_legacy_rrtmg():
    from gpuwm.config import RunConfig,validate_radiation_driver_options
    validate_radiation_driver_options(RunConfig(nx=8,ny=8,nz=8,dx=1000,dy=1000,ztop=10000,dt=6,run_seconds=60,aer_opt=2,ra_physics=4,ra_rrtmg_variant='rrtmg_legacy'))


def test_gridded_aerosol_requires_the_auxiliary_source():
    from gpuwm.config import RunConfig,validate_radiation_driver_options
    with pytest.raises(ValueError,match='auxinput15_inname'):
        validate_radiation_driver_options(RunConfig(nx=8,ny=8,nz=8,dx=1000,dy=1000,ztop=10000,dt=6,run_seconds=60,aer_opt=2,aer_aod550_opt=2,ra_physics=4,ra_rrtmg_variant='rrtmg_legacy'))


def test_aerosol_cannot_be_discarded_by_another_shortwave_solver():
    from gpuwm.config import RunConfig,validate_radiation_driver_options
    with pytest.raises(ValueError,match='discard'):
        validate_radiation_driver_options(RunConfig(nx=8,ny=8,nz=8,dx=1000,dy=1000,ztop=10000,dt=6,run_seconds=60,aer_opt=2,ra_physics=4))


def test_namelist_selects_legacy_for_user_aerosol():
    from gpuwm.namelist_import import _default_rrtmg_variant_for_namelist
    assert _default_rrtmg_variant_for_namelist({'physics':{'aer_opt':[2]}})=='rrtmg_legacy'


def test_auxinput15_uses_the_rust_reader_and_updates_the_time_record():
    from datetime import datetime
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    from gpuwm.core.aer_opt2 import auxiliary_inputs,INPUT_FIELDS
    root=Path(__file__).resolve().parents[1]/'tools/aer_opt2_wrf461_oracle/fixtures'
    cfg=SimpleNamespace(grid_id=2,aer_aod550_opt=2,aer_angexp_opt=2,
        aer_ssa_opt=2,aer_asy_opt=2,aer_auxinput15=str(root/'auxinput15_d<domain>.nc'),
        aer_auxinput15_interval=0)
    first=auxiliary_inputs(cfg,datetime(2025,6,1,18),0,(2,3))
    second=auxiliary_inputs(cfg,datetime(2025,6,1,18),600,(2,3))
    for i,key in enumerate(INPUT_FIELDS):
        assert np.all(first[key]==np.float32(np.float32(0.1)*np.float32(i+1)+np.float32(0.01)))
        assert np.all(second[key]==np.float32(np.float32(0.1)*np.float32(i+1)+np.float32(0.02)))
    with pytest.raises(ValueError,match='shape'):
        auxiliary_inputs(cfg,datetime(2025,6,1,18),0,(3,2))
    with pytest.raises(ValueError,match='no record'):
        auxiliary_inputs(cfg,datetime(2025,6,1,17),0,(2,3))


@pytest.mark.gpu
def test_zero_aod_and_subnormal_numerator_control():
    from tools.aer_opt2_wrf461_oracle.validate_edges import validate
    result=validate()
    assert result['zero_aod_columns']==192 and result['zero_optics_max_ulp']==0
    assert result['declared_zero_aod_angstrom_diagnostics']==48


@pytest.mark.gpu
def test_gpu_optics_and_sw_fluxes_match_the_wrf_oracle():
    from tools.aer_opt2_wrf461_oracle.validate_flux import validate
    result=validate()
    assert result['columns']==96
    assert all(row['max_ulp']==0 for row in result['fields'].values())


@pytest.mark.gpu
def test_real_physics_driver_allocates_and_owns_the_user_aerosol_history():
    import cupy as cp
    import numpy as np
    from datetime import datetime
    from gpuwm.config import RunConfig
    from gpuwm.core.state import DomainState
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.radiation_composition import make_radiation
    cfg=RunConfig(nx=6,ny=5,nz=8,dx=3000,dy=3000,ztop=10000,dt=6,run_seconds=60,
        moist=False,mp_physics=0,sf_sfclay_physics=0,sf_surface_physics=0,bl_pbl_physics=0,
        aer_opt=2,ra_physics=4,ra_rrtmg_variant='rrtmg_legacy')
    state=DomainState(cfg)
    lat=np.full((5,6),35,np.float32);lon=np.full((5,6),-97,np.float32)
    radiation=make_radiation(cfg,datetime(2025,6,1,18),lat,lon)
    driver=initialize_physics(state,cfg,radiation=radiation,
        radiation_start_time=datetime(2025,6,1,18),radiation_latitude=lat,radiation_longitude=lon)
    assert driver.aod5503d.shape==(8,5,6)
    for key,value in driver.aerosol2d.items():
        assert value.shape==(5,6) and value.dtype==cp.float32
        assert driver.fields[key] is value
    from gpuwm.io.restart import _driver_manifest
    manifest=_driver_manifest(driver)
    for key in driver.aerosol2d:
        assert any(value is driver.fields[key] for value in manifest.values())


@pytest.mark.gpu
def test_user_aerosol_driver_updates_all_64_day_and_night_columns():
    import cupy as cp
    import numpy as np
    from test_rrtmg_legacy_wiring import profile,_bundle,_env_from,START,_call
    from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation
    from gpuwm.core.aer_opt2 import OPTION_KEYS,VALUE_KEYS,DEFAULT_VALUES
    donor=profile.__wrapped__()
    bundle=_bundle(donor,64,seed=1072)
    env=_env_from(bundle,np.arange(64),8,8,np.linspace(20,45,64),np.linspace(-170,170,64))
    env.state.ht=cp.zeros((8,8),cp.float32)
    env.cfg.aer_opt=2;env.cfg.aer_type=2
    for key in OPTION_KEYS:setattr(env.cfg,key,1 if key=='aer_aod550_opt' else 3)
    for key,value in zip(VALUE_KEYS,DEFAULT_VALUES):setattr(env.cfg,key,value)
    env.cfg.aer_auxinput15='';env.cfg.aer_auxinput15_interval=0;env.cfg.grid_id=1
    adapter=RRTMGLegacyRadiation(START,env.lat.reshape(8,8),env.lon.reshape(8,8),p_top=env.p_top,aer_opt=2)
    result=_call(adapter,env)
    night=cp.asnumpy(result.coszen)<=0
    assert night.any() and (~night).any()
    assert np.all(cp.asnumpy(result.swdown)[night]==0)
    assert np.all(cp.asnumpy(result.rthratensw)[:,night]==0)
    assert np.all(cp.asnumpy(result.aod5503d)>0)
    assert np.all(cp.asnumpy(result.aerosol2d['AOD5502D'])==np.float32(0.12))


@pytest.mark.gpu
def test_held_aerosol_diagnostics_survive_a_real_restart_file(tmp_path):
    import cupy as cp
    import numpy as np
    from datetime import datetime
    from gpuwm.config import RunConfig
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.radiation_composition import make_radiation
    from gpuwm.io import restart
    from test_physics import _balanced_state
    cfg=RunConfig(nx=6,ny=5,nz=8,dx=3000,dy=3000,ztop=10000,dt=6,run_seconds=60,
        moist=True,mp_physics=0,sf_sfclay_physics=0,sf_surface_physics=0,bl_pbl_physics=0,
        aer_opt=2,ra_physics=4,ra_rrtmg_variant='rrtmg_legacy')
    lat=np.full((5,6),35,np.float32);lon=np.full((5,6),-97,np.float32)
    def build():
        state=_balanced_state(cp,cfg)
        radiation=make_radiation(cfg,datetime(2025,6,1,18),lat,lon)
        driver=initialize_physics(state,cfg,radiation=radiation,
            radiation_start_time=datetime(2025,6,1,18),radiation_latitude=lat,radiation_longitude=lon)
        return state,driver
    source,driver=build()
    driver.aod5503d.fill(np.float32(0.01))
    for i,value in enumerate(driver.aerosol2d.values()):value.fill(np.float32(0.2+0.1*i))
    path=restart.write_restart(tmp_path/'user-aerosol.npz',source,cfg)
    restored,live=build()
    restart.restore_restart(path,restored,cfg)
    assert cp.asnumpy(live.aod5503d).tobytes()==cp.asnumpy(driver.aod5503d).tobytes()
    for key,value in live.aerosol2d.items():
        assert value is live.fields[key]
        assert cp.asnumpy(value).tobytes()==cp.asnumpy(driver.aerosol2d[key]).tobytes()


def test_aerosol_shared_loader_preserves_subnormal_operands():
    from gpuwm.core.kernels import module_options
    assert module_options("aer_opt2") == ("-std=c++17", "--fmad=false", "--ftz=false")
