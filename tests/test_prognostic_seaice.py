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

@pytest.mark.gpu
def test_pwp_owner_freezes_its_mixed_layer_without_an_oml_slab(cp):
    """Multi-day runs default to the 3D PWP ocean (sf_ocean_physics=2), which
    has no OML tml/hml. reconcile_ocean read the slab and died with KeyError
    'tml' at the first surface step of an 84 h 12 km CONUS run with Great
    Lakes ice (GFS 2025-04-02T00). The PWP mixed layer must freeze into the
    ice column with its heat conserved, and columns PWP does not own stay put."""
    from gpuwm.core.surface_ocean import initialize_ocean, launch_ocean_step
    from gpuwm.ocean.seaice import reconcile_ocean, step
    from gpuwm.config import RunConfig
    d,a=fields(cp,area=.7)
    f=d.fields
    f['xland'][:,:16]=1.     # ice classified as land: no PWP profile there
    # Open water the ice model owns, but with no ice on it: the atmosphere's
    # own fluxes act there, so the PWP step must keep applying them.
    for key in ('xice','ice_volume','ice_thickness'): f[key][:,56:]=0.
    for key,val in dict(gsw=0.,uair=5.,vair=0.,ust=.1,f=.00014).items():
        f[key]=cp.full_like(f['tsk'],val)
    cfg=RunConfig(nx=64,ny=64,nz=16,dx=25000.,dy=25000.,ztop=12000.,
                  dt=60.,run_seconds=302400.,sf_ocean_physics=2,ocean_levels=10,
                  ocean_mld_climatology='configured_uniform_mld',oml_hml0=30.)
    forcing=dict(f,tsk=f['sst'])
    with pytest.warns(RuntimeWarning, match='configured_uniform_mld'):
        initialize_ocean(forcing,cfg)
    f.update({k:v for k,v in forcing.items() if k not in f})
    assert 'tml' not in f and 'om_tmp' in f
    # Cold start: liquid water is never seeded below Tf where this model owns
    # it (the fallback lapse runs to 268.55 K at 50 m), so no ice forms from
    # the initial profile; columns PWP does not own keep theirs.
    owned=f['_pwp_land']>1.5
    assert float(cp.min(f['om_tmp'][:,owned]))==float(cp.float32(271.35))
    assert float(cp.min(f['om_tini'][:,owned]))==float(cp.float32(271.35))
    assert float(cp.min(f['om_tmp'][:,~owned]))<271.
    step(d,a,config())
    # Under ice the ice step owns the surface energy balance and hands the
    # ocean ICE_OCEAN_HEAT, as on the OML route. The PWP step must not also
    # apply the night-time atmospheric loss to those columns (that counted the
    # ice-covered surface twice and never took the 2 W m-2 ocean-to-ice flux out).
    coupled=f['ice_ocean_coupled']>0
    assert bool(cp.all(coupled[:,16:56])) and not bool(cp.any(coupled[:,56:]))
    z=f['om_depth'].astype(cp.float64)
    dz=cp.concatenate([z[:1],z[1:]-z[:-1]])
    capacity=1024.*4183.3
    content=lambda: (f['om_tmp'].astype(cp.float64)*dz*capacity).sum(axis=0)
    before=content()
    launch_ocean_step(dict(f,tsk=f['sst']),dt=600.,gamma=cfg.oml_gamma,
                      relaxation_time=cfg.oml_relaxation_time,dx=cfg.dx,dy=cfg.dy,
                      ocean_timestep_minutes=cfg.omdt)
    wet=f['_pwp_land']>1.5
    # (columns 16-31 lie further from the open water than the step's
    # transport stencil reaches in its 20 stages)
    far=cp.zeros_like(wet); far[:,16:32]=True
    assert bool(cp.all(f['om_tmp'][0][far]==cp.float32(271.35)))
    assert float(cp.max(cp.abs(content()-before)[far]))==0.
    assert float(cp.max((content()-before)[:,56:]))<-5e4  # open water lost its night heat (~150 W m-2)
    # The ocean's heat this step under ice: the open-water night loss and the
    # ocean-to-ice flux, out of water already at Tf.
    qice=f['ice_ocean_heat'].astype(cp.float64)*600.*coupled
    assert float(cp.max(qice[far]))<-2.*600.
    t0=f['om_tmp'].astype(cp.float64).copy()
    v0=f['ice_volume'].astype(cp.float64).copy()
    x0=f['ice_ocean_exchange'].astype(cp.float64).copy()
    e0=f['ice_energy_error'].astype(cp.float64).copy()
    o0=f['ice_ocean_owed'].astype(cp.float64).copy()
    with pytest.raises(ValueError, match='surface step dt'):
        reconcile_ocean(f,cfg)
    reconcile_ocean(f,cfg,dt=600.)
    gained=((f['om_tmp'].astype(cp.float64)-t0)*dz*capacity).sum(axis=0)
    exchanged=f['ice_ocean_exchange'].astype(cp.float64)-x0
    owed=f['ice_ocean_owed'].astype(cp.float64)-o0
    # Ice-ocean budget: the column changes by ICE_OCEAN_HEAT*dt plus the heat
    # its supercooled layers gave the ice (ICEOCEANEXCHANGE), less what waits.
    assert float(cp.min(exchanged[wet]))>0
    # (to the float32 storage of ICEOCEANEXCHANGE: 3e-8 of the heat moved)
    assert float(cp.max(cp.abs(gained-qice-exchanged+owed)[wet]))<1e-6*float(cp.max(cp.abs(exchanged)))
    grown=f['ice_volume'].astype(cp.float64)-v0
    assert float(cp.max(cp.abs(grown-exchanged/(917.*333700.))))<1e-6
    # The ledger moves only by the float32 rounding of the stored ice volume.
    v=f['ice_volume']
    quantum=917.*333700.*(cp.nextafter(v,cp.float32(cp.inf))-v).astype(cp.float64)
    assert bool(cp.all(cp.abs(f['ice_energy_error']-e0)<=quantum))
    assert bool(cp.all(f['sst'][wet]==cp.float32(271.35)))
    assert bool(cp.all(f['om_tmp'][0][wet]==cp.float32(271.35)))
    # Columns PWP does not own are neither frozen nor warmed: the ice step's
    # ocean heat is their prescribed boundary, as with no ocean model.
    assert bool(cp.all(gained[~wet]==0)) and bool(cp.all(grown[~wet]==0))
    assert bool(cp.all(f['ice_ocean_owed'][~wet]==0))
    assert bool(cp.allclose(f['tsk'][wet], (f['xice']*f['ice_skin']+(1-f['xice'])*f['sst'])[wet]))

@pytest.mark.gpu
def test_pwp_under_full_ice_gives_up_the_ocean_to_ice_heat_flux(cp):
    """The ice step's 2 W m-2 ocean-to-ice flux must leave a 3D PWP column
    whose mixed layer sits above Tf. Per 600 s step it is 1e-5 K over the
    mixed layer, under half a float32 unit at 271 K: applied naively it would
    round away every step and leak forever. Over a day the column gives up the
    whole flux, and the ice-ocean budget closes to the float32 owed carrier."""
    from gpuwm.core.surface_ocean import initialize_ocean, launch_ocean_step
    from gpuwm.ocean.seaice import reconcile_ocean, step
    from gpuwm.config import RunConfig
    d,a=fields(cp,area=1.)
    f=d.fields
    for key,val in dict(gsw=0.,uair=0.,vair=0.,ust=0.,f=.00014).items():
        f[key]=cp.full_like(f['tsk'],val)
    cfg=RunConfig(nx=64,ny=64,nz=16,dx=25000.,dy=25000.,ztop=12000.,
                  dt=60.,run_seconds=302400.,sf_ocean_physics=2,ocean_levels=10,
                  ocean_mld_climatology='configured_uniform_mld',oml_hml0=30.)
    forcing=dict(f,tsk=f['sst'])
    with pytest.warns(RuntimeWarning, match='configured_uniform_mld'):
        initialize_ocean(forcing,cfg)
    f.update({k:v for k,v in forcing.items() if k not in f})
    # Water under the ice 0.2 K above freezing, the whole column alike.
    f['om_tmp']+=cp.float32(.2); f['om_tini']+=cp.float32(.2)
    f['sst'][...]=f['om_tmp'][0]
    z=f['om_depth'].astype(cp.float64)
    dz=cp.concatenate([z[:1],z[1:]-z[:-1]])
    capacity=1024.*4183.3
    content=lambda: (f['om_tmp'].astype(cp.float64)*dz*capacity).sum(axis=0)
    h0=content(); v0=f['ice_volume'].astype(cp.float64).copy()
    x0=f['ice_ocean_exchange'].astype(cp.float64).copy()
    supplied=cp.zeros_like(h0)
    for _ in range(144):
        step(d,a,config())
        supplied+=f['ice_ocean_heat'].astype(cp.float64)*600.
        launch_ocean_step(dict(f,tsk=f['sst']),dt=600.,gamma=cfg.oml_gamma,
                          relaxation_time=cfg.oml_relaxation_time,dx=cfg.dx,dy=cfg.dy,
                          ocean_timestep_minutes=cfg.omdt)
        reconcile_ocean(f,cfg,dt=600.)
    gained=content()-h0
    exchanged=f['ice_ocean_exchange'].astype(cp.float64)-x0
    owed=f['ice_ocean_owed'].astype(cp.float64)
    # Full ice and no melt: the ocean's net is exactly the ocean-to-ice flux.
    assert bool(cp.allclose(supplied,-2.*86400.,rtol=1e-5))
    # Nothing froze: the water stays above Tf (the exchange holds only the ice
    # step's round-off melt excess, already inside ICE_OCEAN_HEAT).
    assert float(cp.max(exchanged))<1e-9
    assert bool(cp.all(f['om_tmp'][0]>cp.float32(271.35)))
    # The column gave up the day's flux; only a sub-unit remainder is owed.
    assert float(cp.max(cp.abs(gained-supplied+owed)))<1e-2
    ml_capacity=float(capacity*(f['om_ml'].max()+5.))
    ulp=float(cp.nextafter(cp.float32(271.6),cp.float32(300))-cp.float32(271.6))
    assert float(cp.max(cp.abs(owed)))<=ulp*ml_capacity
    assert float(cp.max(cp.abs(gained-supplied)))<=ulp*ml_capacity
    assert float(cp.max(gained))<-.9*2.*86400.
    assert bool(cp.all(f['sst']==f['om_tmp'][0]))

@pytest.mark.gpu
def test_oml_open_ocean_keeps_the_atmosphere_flux_with_ice_elsewhere(cp):
    """sf_ocean_physics=1 (OML) with ice in one column: the ice model's ocean
    mask covers all non-lake water, and oml_step swapped ICE_OCEAN_HEAT (the
    ice model's own open-water flux) into every masked column. So open ocean
    far from the ice lost the fluxes the atmosphere sees. Only the columns
    whose surface fluxes the ice step set (ICE_OCEAN_COUPLED) take its heat;
    ice-free open ocean must step exactly as with no ice on the grid."""
    from gpuwm.core.surface_ocean import (initialize_ocean, launch_ocean_step,
                                          STATE_FIELDS, FORCING_FIELDS)
    from gpuwm.ocean.seaice import initialize, step
    from gpuwm.config import RunConfig
    d,a=fields(cp,area=0.)
    f=d.fields
    assert 'ice_volume' not in f
    f['sst'].fill(278.); f['xice'][10,10]=1.; f['sst'][10,10]=271.35
    initialize(f,config())
    assert float(f['ice_ocean_mask'].min())==1.  # every water column is masked
    for key,val in dict(gsw=0.,glw=300.,hfx=40.,lh=80.,uair=5.,vair=0.,ust=.2,f=.0001).items():
        f[key]=cp.full_like(f['tsk'],val)
    cfg=RunConfig(nx=64,ny=64,nz=16,dx=25000.,dy=25000.,ztop=12000.,
                  dt=60.,run_seconds=259200.,sf_ocean_physics=1,oml_hml0=50.)
    forcing=dict(f,tsk=f['sst'])
    initialize_ocean(forcing,cfg)
    f.update({k:forcing[k] for k in STATE_FIELDS})
    step(d,a,config())
    coupled=f['ice_ocean_coupled']>0
    assert bool(coupled[10,10]) and int(coupled.sum())==1
    # Ice-free reference: the same ocean, no sea-ice carriers.
    ref={k:f[k].copy() for k in (*STATE_FIELDS,*FORCING_FIELDS)}
    ref['tsk']=f['sst'].copy()
    launch_ocean_step(ref,dt=600.,gamma=cfg.oml_gamma,relaxation_time=cfg.oml_relaxation_time)
    t0=f['tml'].copy()
    launch_ocean_step(dict(f,tsk=f['sst']),dt=600.,gamma=cfg.oml_gamma,
                      relaxation_time=cfg.oml_relaxation_time)
    open_=~coupled
    for key in (*STATE_FIELDS,):
        assert bool(cp.array_equal(f[key][open_],ref[key][open_])),key
    assert bool(cp.array_equal(f['sst'][open_],ref['tsk'][open_]))
    # The atmosphere's night loss cooled the open ocean ...
    assert float(cp.max((f['tml']-t0)[open_]))<0
    # ... and the ice column still took the ice step's heat, not the atmosphere's.
    assert float(f['tml'][10,10])!=float(ref['tml'][10,10])
