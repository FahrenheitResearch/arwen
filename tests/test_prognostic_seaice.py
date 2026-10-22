"""Conservation, snow insulation, complete melt and restart of CUDA columns."""
import os
from types import SimpleNamespace
import numpy as np
import pytest
from gpuwm.ocean.seaice import validate

def config(**kw):
    d=dict(seaice_prognostic=True, seaice_initial_thickness=1.,
           seaice_new_ice_thickness=.5, seaice_ocean_heat_flux=2., sf_surface_physics=2)
    return SimpleNamespace(**(d|kw))

@pytest.fixture
def cp():
    if os.environ.get('GPUWM_NO_LOCAL_GPU'):
        pytest.skip('GPU disabled by test farm')
    cp=pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount()==0: pytest.skip('no CUDA card')
    except cp.cuda.runtime.CUDARuntimeError: pytest.skip('no CUDA runtime')
    return cp

def fields(cp,*,area=1.,snow=0.,air=250.,sw=0.,lw=180.,size=(64,64)):
    values=dict(tsk=260.,sst=271.35,xice=area,xland=2.,lakemask=0.,snow=snow,
        snowh=snow/330.,psfc=100000.,chs=.002,swdown=sw,glw=lw,
        hfx=0.,lh=0.,qfx=0.,qsfc=.002,albedo=.65,emiss=.98,znt=.001,sr=0.)
    f={k:cp.full(size,v,cp.float32) for k,v in values.items()}
    from gpuwm.ocean.seaice import initialize
    initialize(f,config())
    driver=SimpleNamespace(fields=f,bldt_seconds=600.)
    atm={'temperature':cp.full((1,*size),air,cp.float32),
         'qv':cp.full((1,*size),.0002,cp.float32)}
    return driver,atm

def advance(d,atm,steps=432):
    from gpuwm.ocean.seaice import step
    for k in range(steps):step(d,atm,config())

@pytest.mark.gpu
def test_three_day_freezing_energy_budget_and_snow_insulation(cp):
    d,a=fields(cp); snowy,_=fields(cp,snow=50.)
    start={k:d.fields[k].copy() for k in ('ice_volume','ice_snow_mass')}
    advance(d,a);advance(snowy,a)
    assert float(d.fields['ice_volume'].min())>1.
    assert float(snowy.fields['ice_volume'].mean())<float(d.fields['ice_volume'].mean())
    energy=-333700*(917*(d.fields['ice_volume']-start['ice_volume'])+
                     d.fields['ice_snow_mass']-start['ice_snow_mass'])
    assert float(cp.max(cp.abs(energy-d.fields['ice_energy'])))<5000.
    assert float(cp.max(cp.abs(d.fields['ice_energy_error'])))<100.
    assert float(cp.min(d.fields['xice']))==1.

@pytest.mark.gpu
def test_three_day_melt_concentration_and_complete_melt(cp):
    d,a=fields(cp,area=.6,air=290.,sw=600.,lw=380.)
    d.fields['ice_thickness'].fill(.05)
    d.fields['ice_volume'].fill(.03)
    advance(d,a)
    assert float(d.fields['xice'].max())==0.
    assert float(d.fields['ice_volume'].max())==0.
    assert float(d.fields['ice_ocean_exchange'].min())>0.
    assert all(bool(cp.all(cp.isfinite(v))) for v in d.fields.values())

@pytest.mark.gpu
def test_precipitation_phase_and_restart_exact(cp):
    from gpuwm.ocean.seaice import capture_precipitation
    d,a=fields(cp,area=.7)
    d._pending_rainbl=cp.full_like(d.fields['tsk'],2.)
    d.fields['rainbl']=cp.full_like(d.fields['tsk'],1.)
    d.mp_physics=1 # Kessler has SR=0, even with subfreezing air.
    d.microphysics=SimpleNamespace(sr=cp.zeros_like(d.fields['tsk']))
    capture_precipitation(d,a)
    assert float(d.fields['ice_snowfall'].max())==0.
    advance(d,a,144)
    restored=SimpleNamespace(fields={k:cp.asarray(cp.asnumpy(v)) for k,v in d.fields.items()},
                             bldt_seconds=600.)
    advance(d,a,288);advance(restored,a,288)
    for key in d.fields:
        assert bool(cp.array_equal(d.fields[key],restored.fields[key])),key

@pytest.mark.gpu
def test_land_lake_and_ice_free_domains_unchanged(cp):
    from gpuwm.ocean.seaice import initialize
    d,a=fields(cp)
    d.fields['ice_ocean_mask'][:,:32]=0
    saved={k:v[:,:32].copy() for k,v in d.fields.items()}
    advance(d,a)
    for key in saved: assert bool(cp.array_equal(d.fields[key][:,:32],saved[key])),key
    f={k:cp.zeros((4,4),cp.float32) for k in ('tsk','xice','lakemask')}
    initialize(f,config())
    assert 'ice_volume' not in f

@pytest.mark.gpu
def test_coupled_ocean_heat_and_separate_water_skin(cp):
    from gpuwm.core.surface_ocean import initialize_ocean, launch_ocean_step
    from gpuwm.ocean.seaice import reconcile_ocean, step
    from gpuwm.config import RunConfig
    d,a=fields(cp,area=.7)
    f=d.fields
    for key,val in dict(gsw=0.,uair=0.,vair=0.,ust=0.,f=.00014).items():
        f[key]=cp.full_like(f['tsk'],val)
    cfg=RunConfig(nx=64,ny=64,nz=16,dx=25000.,dy=25000.,ztop=12000.,
                  dt=60.,run_seconds=259200.,sf_ocean_physics=1,oml_hml0=50.)
    forcing=dict(f,tsk=f['sst'])
    initialize_ocean(forcing,cfg)
    f.update({k:v for k,v in forcing.items() if k not in f})
    step(d,a,config())
    launch_ocean_step(dict(f,tsk=f['sst']),dt=600.,gamma=cfg.oml_gamma,
                      relaxation_time=cfg.oml_relaxation_time)
    reconcile_ocean(f,cfg)
    assert bool(cp.all(f['sst']==cp.float32(271.35)))
    assert bool(cp.all(f['ice_volume']>0.7))
    assert bool(cp.all(f['tsk']<f['sst']))
    assert bool(cp.allclose(f['tsk'], f['xice']*f['ice_skin']+(1-f['xice'])*cp.maximum(f['sst'],271.35)))

@pytest.mark.gpu
def test_three_day_masked_analysis_sst_and_polar_entrainment(cp):
    from gpuwm.core.surface_ocean import initialize_ocean, launch_ocean_step, STATE_FIELDS
    from gpuwm.ocean.seaice import initialize, reconcile_ocean, step
    from gpuwm.config import RunConfig
    d,a=fields(cp,area=.7)
    f=d.fields
    # The real analysis masks SST to zero under ice whose XLAND is land.
    f['sst'].fill(0.); f['xland'].fill(1.)
    f['sst'][:,32:]=275.
    initialize(f,config())
    assert float(cp.min(f['sst']))==float(cp.float32(271.35))
    assert float(cp.max(f['sst']))==275.
    v0=f['ice_volume'].astype(cp.float64).copy()
    for key,val in dict(gsw=0.,uair=5.,vair=0.,ust=.1,f=.00014).items():
        f[key]=cp.full_like(f['tsk'],val)
    cfg=RunConfig(nx=64,ny=64,nz=16,dx=25000.,dy=25000.,ztop=12000.,
                  dt=60.,run_seconds=259200.,sf_ocean_physics=1,oml_hml0=50.)
    forcing=dict(f,tsk=f['sst'])
    initialize_ocean(forcing,cfg)
    f.update({k:forcing[k] for k in STATE_FIELDS})
    for _ in range(432):
        step(d,a,config())
        launch_ocean_step(dict(f,tsk=f['sst']),dt=600.,gamma=cfg.oml_gamma,
                          relaxation_time=cfg.oml_relaxation_time)
        reconcile_ocean(f,cfg)
    assert float(cp.max(f['ice_thickness']))<1.5
    assert float(cp.min(f['ice_volume'][:,:32]))>.7
    assert bool(cp.all(f['sst']>=cp.float32(271.35)))
    actual=-333700*(917*(f['ice_volume'].astype(cp.float64)-v0)
                      +f['ice_snow_mass'].astype(cp.float64))
    assert float(cp.max(cp.abs(actual-f['ice_energy'])))<5000.

@pytest.mark.gpu
def test_thin_ice_sublimation_cannot_exceed_available_mass(cp):
    d,a=fields(cp,area=1.,air=290.,sw=0.,lw=400.)
    d.fields['ice_volume'].fill(1.e-8)
    d.fields['ice_snow_mass'].fill(0.)
    d.fields['chs'].fill(.1)
    a['qv'].fill(0.)
    advance(d,a,1)
    assert float(cp.min(d.fields['ice_volume']))>=0
    assert float(cp.max(cp.abs(d.fields['ice_energy_error'])))<1.
