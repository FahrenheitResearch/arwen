"""Supplied aerosol IC/BCs reach the common transport and identity owners."""
from dataclasses import replace
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from gpuwm.boundary_fields import external_scalar_fields, potential_external_scalar_fields
from gpuwm.config import RunConfig
from gpuwm.ingest import wrfinput as wi
from wrf_input_fixtures import _small_wrfinput


def _cfg(**changes):
    values = dict(nx=13, ny=12, nz=4, dx=12000., dy=12000., ztop=10000.,
                  dt=1., run_seconds=120., moist=True, mp_physics=28,
                  specified=True, aer_init_opt=1, wif_input_opt=1)
    return RunConfig(**(values | changes))


def _dimensions(cfg):
    return dict(west_east=cfg.nx, west_east_stag=cfg.nx+1,
                south_north=cfg.ny, south_north_stag=cfg.ny+1,
                bottom_top=cfg.nz, bottom_top_stag=cfg.nz+1,
                soil_layers_stag=4)


def _input(path, cfg):
    _small_wrfinput(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                    moisture_names=tuple(wi._active_moisture_map(cfg)))
    with netCDF4.Dataset(path, 'a') as ds:
        ds.createDimension('soil_layers_stag', 4)
        for name in ('QNWFA2D', 'QNIFA2D'):
            ds.createVariable(name, 'f4', ('Time','south_north','west_east'))[:] = (
                np.arange(cfg.ny*cfg.nx).reshape(cfg.ny,cfg.nx) + (32 if name == 'QNWFA2D' else 2))
        ds.createVariable('QNBCA', 'f4', ('Time','bottom_top','south_north','west_east'))[:] = 0
        for name, dim, value in [('C1H','bottom_top',1.),('C2H','bottom_top',0.),
                                 ('C1F','bottom_top_stag',1.),('C2F','bottom_top_stag',0.)]:
            ds.createVariable(name,'f4',('Time',dim))[:] = value
        for name, dims in [('MAPFAC_M',('south_north','west_east')),
                           ('MAPFAC_U',('south_north','west_east_stag')),
                           ('MAPFAC_V',('south_north_stag','west_east'))]:
            ds.createVariable(name,'f4',('Time',*dims))[:] = 1.
        ds.setncatts(dict(MP_PHYSICS=cfg.mp_physics,SF_SURFACE_PHYSICS=cfg.sf_surface_physics,
                         GRID_ID=1,DX=cfg.dx,DY=cfg.dy,MAP_PROJ=1,TRUELAT1=30.,
                         TRUELAT2=60.,STAND_LON=-100.,CEN_LAT=35.,CEN_LON=-100.,
                         HYBRID_OPT=2,ETAC=.2,USE_THETA_M=0,START_DATE='2026-08-25_18:00:00'))
    return path


def _read(path, cfg):
    return wi.read_wrfinput(path, expected_dimensions=_dimensions(cfg),
                            cfg=cfg, require_complete=False)


def _boundary(path, initial, cfg):
    # Independent fixture writer: constant mass13, no map-factor scaling.
    # Only actual aerosol tables are written; number moments are absent.
    layouts = wi._WRFBDY_FIELDS | {
        'nwfa':('QNWFA','bottom_top','south_north','west_east'),
        'nifa':('QNIFA','bottom_top','south_north','west_east')}
    with netCDF4.Dataset(path,'w') as ds:
        for name,size in _dimensions(cfg).items(): ds.createDimension(name,size)
        ds.createDimension('Time',2); ds.createDimension('bdy_width',5)
        ds.createDimension('DateStrLen',19)
        ds.createVariable('Times','S1',('Time','DateStrLen'))[:] = np.array([
            list(b'2026-08-25_18:00:00'),list(b'2026-08-25_18:01:00')],dtype='u1').view('S1')
        ds.setncatts(dict(initial.global_attributes))
        for name,(wrf,zdim,ydim,xdim) in layouts.items():
            a=initial.raw[wrf]
            a=a[None] if zdim is None else a*13.
            for side,suffix in [('west','XS'),('east','XE'),('south','YS'),('north','YE')]:
                if side=='west': slab=a[:,:,:5].transpose(2,0,1)
                elif side=='east': slab=a[:,:,-5:][:,:,::-1].transpose(2,0,1)
                elif side=='south': slab=a[:,:5,:].transpose(1,0,2)
                else: slab=a[:,-5:,:][:,::-1,:].transpose(1,0,2)
                if zdim is None: slab=slab[:,0,:]
                dims=('Time','bdy_width',*(() if zdim is None else (zdim,)),
                      ydim if side in ('west','east') else xdim)
                for marker in ('B','BT'):
                    var=ds.createVariable(f'{wrf}_{marker}{suffix}','f4',dims)
                    rate=np.float32(.125 if name in ('nwfa','nifa') else 0.)
                    var[0]=slab if marker=='B' else rate
                    var[1]=slab+60*rate if marker=='B' else rate
    return path


def _read_boundary(path, initial, cfg):
    return wi.read_wrfbdy(path, restored=initial, run_seconds=120,
                          forcing_interval_seconds=60, cfg=cfg)


def test_aerosol_reader_and_surface_restore_preserve_supplied_words(tmp_path):
    cfg=_cfg(); path=_input(tmp_path/'wrfinput',cfg); initial=_read(path,cfg)
    from gpuwm.core.state import DomainState
    state=DomainState(cfg,array_module=np)
    wi._restore_active_moisture(state,initial.raw,cfg,np)
    for wrf,name in wi._active_moisture_map(cfg).items():
        np.testing.assert_array_equal(getattr(state,name),initial.raw[wrf])
        np.testing.assert_array_equal(getattr(state,name+'0'),initial.raw[wrf])
    for wrf,name in [('QNWFA2D','nwfa2d'),('QNIFA2D','nifa2d')]:
        np.testing.assert_array_equal(getattr(state,name),initial.raw[wrf])
    assert 'QNBCA' in initial.raw and 'QNBCA' not in initial.mapped_variables


def test_black_carbon_input_is_retained_when_inactive_and_names_missing_active_consumer(tmp_path):
    cfg=_cfg(); path=_input(tmp_path/'input',cfg)
    with netCDF4.Dataset(path,'a') as ds:
        ds['QNBCA'][:]=123.
    initial=_read(path,cfg)
    np.testing.assert_array_equal(initial.raw['QNBCA'],123.)
    with pytest.raises(NotImplementedError,match='QNBCA.*wif_input_opt=2.*consumer'):
        _read(path,replace(cfg,wif_input_opt=2))


@pytest.mark.parametrize('poison',['shape','declared_missing','nan'])
def test_added_aerosol_geometry_and_values_are_validated(tmp_path,poison):
    cfg=_cfg(); path=_input(tmp_path/'wrfinput',cfg)
    with netCDF4.Dataset(path,'a') as ds:
        if poison=='shape':
            ds.renameVariable('QNWFA','unused')
            ds.createVariable('QNWFA','f4',('Time','bottom_top','south_north','west_east_stag'))[:] = 1
            # Exercise this field's common geometry validator directly;
            # an unrelated name must not weaken the closed reader inventory.
            variable=ds['QNWFA']
            with pytest.raises(ValueError,match='dimensions|shape'):
                wi._validate_wrfinput_geometry('QNWFA',variable,_dimensions(cfg),np.asarray(variable[0]))
            return
        if poison=='declared_missing':
            ds['QNIFA'].missing_value=np.float32(9e20)
            ds['QNIFA'][0,0,0,0]=np.float32(9e20)
        else:
            ds['QNIFA'][0,0,0,0]=np.nan
    with pytest.raises(ValueError,match='masked|non-finite'):
        _read(path,cfg)


def test_aerosol_boundary_values_tendencies_and_identity_survive(tmp_path):
    cfg=_cfg(); initial=_read(_input(tmp_path/'input',cfg),cfg)
    path=_boundary(tmp_path/'boundary',initial,cfg)
    bc=_read_boundary(path,initial,cfg)
    assert set(bc.intervals[0].fields)=={'u','v','theta','phi','mu','qv','nwfa','nifa'}
    from gpuwm.io.restart import lateral_boundary_prefix_identity
    before=lateral_boundary_prefix_identity(SimpleNamespace(lateral_boundaries=bc))
    for name,wrf in [('nwfa','QNWFA'),('nifa','QNIFA')]:
        expected=np.asarray(initial.raw[wrf][:,:,:5]*13,np.float32).astype(np.float64)
        np.testing.assert_array_equal(bc.intervals[0].fields[name].west.value,expected)
        start=np.float32(np.float32(initial.raw[wrf][0,0,0]*13)+60*np.float32(.125))
        end=np.float32(start+60*np.float32(.125))
        np.testing.assert_array_equal(bc.intervals[1].fields[name].north.tendency,(float(end)-float(start))/60.)
    with netCDF4.Dataset(path,'a') as ds: ds['QNIFA_BTYE'][1,0,0,0] += .25
    after=lateral_boundary_prefix_identity(SimpleNamespace(lateral_boundaries=_read_boundary(path,initial,cfg)))
    assert before!=after


@pytest.mark.parametrize('poison',['missing','late_nan','pair','identity'])
def test_supplied_aerosol_boundary_failures_are_not_dropped(tmp_path,poison):
    cfg=_cfg(); initial=_read(_input(tmp_path/'input',cfg),cfg)
    path=_boundary(tmp_path/'boundary',initial,cfg)
    with netCDF4.Dataset(path,'a') as ds:
        if poison=='missing': ds.renameVariable('QNWFA_BTXS','absent')
        elif poison=='late_nan': ds['QNIFA_BTYE'][1,0,0,0]=np.nan
        elif poison=='pair': ds['QNWFA_BXS'][0,0,0,0]+=100
        else: ds.DX=cfg.dx+1
    with pytest.raises(ValueError,match='QNWFA_BTXS|non-finite|does not match initial|DX'):
        _read_boundary(path,initial,cfg)


@pytest.mark.parametrize('mp',[6,8,9,10,16,18,28,50])
def test_default_ordinary_number_moments_do_not_require_boundary_tables(mp):
    cfg=_cfg(mp_physics=mp,aer_init_opt=0,wif_input_opt=0,mp28_aerosol_source='synthetic')
    assert external_scalar_fields(cfg)==('qv',)


def test_resolved_zero_aerosol_is_still_supplied_and_cold_pricing_covers_it():
    cfg=_cfg(aer_init_opt=0,wif_input_opt=0)
    assert external_scalar_fields(cfg)==('qv',)
    assert external_scalar_fields(cfg,aerosol_from_input=True)==('qv','nwfa','nifa')
    assert potential_external_scalar_fields(cfg)==('qv','nwfa','nifa')
    from gpuwm.core.preflight import lbc_interval_values,scratch_slot_registry
    synthetic=replace(cfg,mp28_aerosol_source='synthetic')
    assert lbc_interval_values(cfg)-lbc_interval_values(synthetic)==8*cfg.nz*5*(cfg.nx+cfg.ny)
    slots=scratch_slot_registry(cfg)
    for name in ('lbc_nwfa_held','lbc_nifa_held'):
        assert slots[name]==(cfg.nz,cfg.ny,cfg.nx)
        assert name not in scratch_slot_registry(synthetic)


def test_snapshot_inventory_uses_resolved_input_presence_without_testing_values():
    from gpuwm.core.state import DomainState
    from gpuwm.ingest.lateral_bc import domain_boundary_snapshot
    cfg=_cfg(aer_init_opt=0,wif_input_opt=0)
    state=DomainState(cfg,array_module=np)
    # A tiny independent base suffices for coupling; zero aerosol is actual
    # supplied data here, not a cue to select synthetic initialization.
    state.c1h[:]=1;state.c2h[:]=0;state.c1f[:]=1;state.c2f[:]=0
    state.mub2d[:]=10000
    assert set(domain_boundary_snapshot(state))=={'u','v','theta','phi','mu','qv'}
    state._external_scalar_boundary_fields=external_scalar_fields(cfg,aerosol_from_input=True)
    snapshot=domain_boundary_snapshot(state)
    assert set(snapshot)=={'u','v','theta','phi','mu','qv','nwfa','nifa'}
    np.testing.assert_array_equal(snapshot['nwfa'],0.)
    np.testing.assert_array_equal(snapshot['nifa'],0.)


@pytest.mark.parametrize('has_grid', [False, True])
def test_real_initialization_records_final_wif_resolution(monkeypatch, tmp_path, has_grid):
    from gpuwm.ingest import wif_climatology as wif
    from gpuwm.ingest.lateral_bc import domain_boundary_snapshot
    from test_real_init import _analyzed_hrrr_real_init

    source = tmp_path / 'declared-wif-fixture'
    source.write_bytes(b'unit provider: valid all-zero aerosol fields')
    resolution = wif.WifSourceResolution(source, 'unit-provider', (str(source),))
    monkeypatch.setattr(wif, 'resolve_wif_climatology', lambda *a, **k: resolution)
    reads = []

    def load(path):
        reads.append(path)
        return object()

    def fields(dataset, lat, lon, date, pressure, phb):
        return ({'nwfa': np.zeros_like(pressure), 'nifa': np.zeros_like(pressure),
                 'nwfa2d': np.zeros_like(lat), 'nifa2d': np.zeros_like(lat)},
                {'schema': 'unit-supplied-aerosol'})

    monkeypatch.setattr(wif, 'load_wif_climatology', load)
    monkeypatch.setattr(wif, 'wif_fields_for_grid', fields)
    geometry = (np.full((2, 3), 35.), np.full((2, 3), -97.)) if has_grid else None
    result, cfg = _analyzed_hrrr_real_init(28, wif_grid_latlon=geometry)
    snapshot = domain_boundary_snapshot(result.state)
    assert reads == ([source] if has_grid else [])
    assert result.state._external_scalar_boundary_fields == (
        ('qv', 'nwfa', 'nifa') if has_grid else ('qv',))
    assert {'nwfa', 'nifa'} & set(snapshot) == ({'nwfa', 'nifa'} if has_grid else set())
    assert result.aerosol_initialization['dataset']['resolved'] is has_grid
    # Both branches have zero fields, but only the one actually read is a
    # supplied scalar boundary: numeric values cannot make the decision.
    np.testing.assert_array_equal(result.state.nwfa, 0.)
    np.testing.assert_array_equal(result.state.nifa, 0.)


@pytest.mark.gpu
@pytest.mark.parametrize('final',[False,True])
def test_actual_cuda_scalar_stage_consumes_aerosol_tables_and_preserves_number_flow(final):
    cp=pytest.importorskip('cupy')
    from gpuwm.core.grid import make_base_state,make_vertical_coord
    from gpuwm.core.state import init_at_rest
    from gpuwm.core.moist import advance_scalars_stage
    from gpuwm.ingest.lateral_bc import domain_boundary_snapshot,build_lateral_boundaries,attach_lateral_boundaries
    cfg=_cfg(nz=6,dt=1.)
    coord=make_vertical_coord(cfg.nz)
    base=make_base_state(coord,lambda z:np.full_like(z,300.),cfg.p_surf,cfg.ztop)
    state=init_at_rest(cfg,coord,base)
    state.qv[:]=.01; state.qv0[:]=state.qv
    for name,value in [('nwfa',2e8),('nifa',2e5),('nc',1e5)]:
        getattr(state,name)[:]=value;getattr(state,name+'0')[:]=value
    state.mup0[:]=state.mup
    first=domain_boundary_snapshot(state)
    for name in ('nwfa','nifa'): getattr(state,name)[:]*=1.25
    second=domain_boundary_snapshot(state)
    for name in ('nwfa','nifa'): getattr(state,name)[:]=getattr(state,name+'0')
    bc=build_lateral_boundaries([first,second],[0.,60.])
    attach_lateral_boundaries(state,bc)
    # Nonzero inflow distinguishes supplied boundary forcing from the old
    # zero-inflow branch; nc has no external table and still takes that branch.
    ru=cp.full_like(state.u,1.);rv=cp.zeros_like(state.v);ww=cp.zeros_like(state.w)
    advance_scalars_stage(state,cfg,ru,rv,ww,dt_eff=1.,final=final,apply_relax=True)
    for name,value in [('nwfa',2e8),('nifa',2e5)]:
        expected=value*(1.+.25/60.)
        np.testing.assert_allclose(cp.asnumpy(getattr(state,name))[:,:,0],expected,rtol=3e-7)
    np.testing.assert_array_equal(cp.asnumpy(state.nc)[:,:,0],0.)
