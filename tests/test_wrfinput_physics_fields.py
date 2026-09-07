"""Supplied RUC/MYNN fields reach their selected existing physics carriers."""
from dataclasses import replace
import sys
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from gpuwm.config import RunConfig, soil_layer_count
from gpuwm.ingest import wrfinput as wi
from wrf_input_fixtures import _small_wrfinput


def _cfg(**changes):
    return RunConfig(**(dict(nx=6, ny=4, nz=3, dx=3000., dy=3000., ztop=10000.,
                             dt=1., run_seconds=0., moist=True, mp_physics=6,
                             sf_surface_physics=3, num_soil_layers=9,
                             sf_sfclay_physics=5, bl_pbl_physics=5) | changes))


def _dimensions(cfg):
    return dict(west_east=cfg.nx, west_east_stag=cfg.nx+1,
                south_north=cfg.ny, south_north_stag=cfg.ny+1,
                bottom_top=cfg.nz, bottom_top_stag=cfg.nz+1,
                soil_layers_stag=soil_layer_count(cfg))


def _values(cfg, qke_name='qke'):
    shape = cfg.ny, cfg.nx
    column = np.arange(cfg.ny*cfg.nx, dtype=np.float32).reshape(shape)
    return {'ACRUNOFF': column+.25, 'RHOSNF': column+100.,
            'SNOWFALLAC': column+.125, 'SOILT1': column+270.,
            qke_name: np.broadcast_to(column*.125+.25, (cfg.nz, *shape)).copy(),
            'qke_adv': np.full((cfg.nz, *shape), 37., dtype=np.float32)}


def _input(path, cfg, values, *, dimensions_override=None):
    _small_wrfinput(path, nz=cfg.nz, ny=cfg.ny, nx=cfg.nx,
                    moisture_names=tuple(wi._active_moisture_map(cfg)))
    with netCDF4.Dataset(path, 'a') as ds:
        ds.createDimension('soil_layers_stag', soil_layer_count(cfg))
        ds.setncatts(dict(MP_PHYSICS=cfg.mp_physics,
                         SF_SURFACE_PHYSICS=cfg.sf_surface_physics,
                         BL_PBL_PHYSICS=cfg.bl_pbl_physics,
                         MMINLU='MODIFIED_IGBP_MODIS_NOAH'))
        for name, value in values.items():
            dims = (dimensions_override or {}).get(name, wi.WRFINPUT_DIMENSIONS[name])
            dtype = 'i4' if np.asarray(value).dtype.kind in 'iu' else 'f4'
            ds.createVariable(name, dtype, ('Time', *dims))[:] = value
    return path


def _read(path, cfg):
    return wi.read_wrfinput(path, cfg=cfg, expected_dimensions=_dimensions(cfg),
                            require_complete=False)


@pytest.mark.parametrize('qke_name', ['qke', 'QKE'])
def test_reader_retains_exact_ruc_and_mynn_words_and_inactive_advected_record(tmp_path, qke_name):
    cfg = _cfg(); values = _values(cfg, qke_name)
    result = _read(_input(tmp_path/'input', cfg, values), cfg)
    for name, value in values.items():
        # The Rust bridge exposes decoded reals as float64. Every source
        # float32 value is represented exactly and restores to the same word.
        np.testing.assert_array_equal(result.raw[name], value)
        np.testing.assert_array_equal(result.raw[name].astype('f4').view('u4'), value.view('u4'))
        assert (name in result.mapped_variables) is (name != 'qke_adv')


@pytest.mark.parametrize('name', ['ACRUNOFF', 'RHOSNF', 'SNOWFALLAC', 'SOILT1', 'qke', 'QKE', 'qke_adv'])
@pytest.mark.parametrize('poison', ['shape', 'nan', 'masked'])
def test_new_physics_fields_keep_closed_geometry_and_value_checks(tmp_path, name, poison):
    cfg = _cfg()
    dimensions = ({name: ('bottom_top', 'south_north', 'west_east_stag')}
                  if poison == 'shape' else None)
    value = 5. if dimensions else np.full(
        tuple(_dimensions(cfg)[d] for d in wi.WRFINPUT_DIMENSIONS[name]), 5., np.float32)
    path = _input(tmp_path/'input', cfg, {name: value}, dimensions_override=dimensions)
    if poison != 'shape':
        with netCDF4.Dataset(path, 'a') as ds:
            if poison == 'nan': ds[name][:] = np.nan
            else:
                ds[name].missing_value = np.float32(9.e20)
                ds[name][:] = np.float32(9.e20)
    with pytest.raises(ValueError, match='shape mismatch|non-finite|masked'):
        _read(path, cfg)


def test_new_fields_require_their_selected_consumer_and_qke_aliases_must_agree(tmp_path):
    cfg = _cfg(); values = _values(cfg)
    path = _input(tmp_path/'input', cfg, values)
    with pytest.raises(ValueError, match='RUC.*sf_surface_physics=3'):
        _read(path, replace(cfg, sf_surface_physics=2, num_soil_layers=4))
    with pytest.raises(ValueError, match='MYNN.*bl_pbl_physics=5'):
        _read(path, replace(cfg, bl_pbl_physics=1))
    with netCDF4.Dataset(path, 'a') as ds:
        ds.createVariable('QKE', 'f4', ('Time', *wi.WRFINPUT_DIMENSIONS['QKE']))[:] = values['qke']
    _read(path, cfg)
    with netCDF4.Dataset(path, 'a') as ds: ds['QKE'][0,0,0,0] += 1.
    with pytest.raises(ValueError, match='conflicting aliases'):
        _read(path, cfg)


def test_qke_adv_names_unsupported_operation_for_active_or_unresolved_selection(tmp_path):
    cfg = _cfg(); path = _input(tmp_path/'input', cfg, _values(cfg))
    with pytest.raises(NotImplementedError, match='qke_adv transport.*MYNN feedback'):
        _read(path, replace(cfg, bl_mynn_tkeadvect=True))
    with pytest.raises(ValueError, match='explicit bl_mynn_tkeadvect=False'):
        wi.read_wrfinput(path, cfg=None, expected_dimensions=_dimensions(cfg),
                         require_complete=False)


def _surface_values(cfg):
    shape = cfg.ny, cfg.nx
    defaults = dict(LANDMASK=1., XICE=0., ALBBCK=.2, LAI=2., TSK=290.,
                    LU_INDEX=10, ISLTYP=6, VEGFRA=60., TMN=288., SNOW=0.,
                    SNOWH=0., GLW=300.)
    values = {name: np.full(shape, value, np.int32 if name in ('LU_INDEX','ISLTYP')
                           else np.float32) for name, value in defaults.items()}
    soil = soil_layer_count(cfg), *shape
    values.update(TSLB=np.full(soil, 288., np.float32),
                  SMOIS=np.full(soil, .3, np.float32), SH2O=np.full(soil, .3, np.float32))
    return values


@pytest.mark.parametrize('supplied', [True, False])
def test_actual_initializer_forwards_supplied_fields_and_preserves_absent_cold_state(
        tmp_path, monkeypatch, supplied):
    cfg = _cfg(); values = _values(cfg)
    raw = _surface_values(cfg) | (values if supplied else {})
    restored = _read(_input(tmp_path/'input', cfg, raw), cfg)
    fields = {name.lower(): np.full(value.shape, -77., np.float32)
              for name, value in values.items() if name != 'qke_adv'}
    fields.update(albbck=np.zeros((cfg.ny,cfg.nx), np.float32),
                  lai=np.zeros((cfg.ny,cfg.nx), np.float32))
    driver = SimpleNamespace(fields=fields, rainc=None)
    monkeypatch.setitem(sys.modules, 'cupy', np)
    monkeypatch.setitem(sys.modules, 'gpuwm.core.physics', SimpleNamespace(
        initialize_physics=lambda *args, **kwargs: driver))
    assert wi.initialize_wrfinput_physics(object(), restored, cfg) is driver
    for name, value in values.items():
        if name == 'qke_adv':
            assert name not in driver.fields
        else:
            expected = value if supplied else np.full(value.shape, -77., np.float32)
            np.testing.assert_array_equal(fields[name.lower()].view('u4'), expected.view('u4'))


def _from_driver(driver, cp):
    raw = {aliases[0]: cp.asnumpy(driver.fields[field])
           for field, aliases in wi.PHYSICS_FIELD_ALIASES.items() if field in driver.fields}
    raw['LU_INDEX'] = raw['IVGTYP']
    return raw


class _ConsumerReached(Exception):
    pass


@pytest.mark.gpu
def test_file_values_reach_actual_ruc_driver_arguments(tmp_path, monkeypatch):
    cp = pytest.importorskip('cupy')
    from test_ruc_runtime import _build
    from gpuwm.core import ruc_runtime
    from gpuwm.core.dycore import step
    state, cfg, template = _build(nx=6, ny=4, nz=20)
    values = {name: value for name, value in _values(cfg).items() if name in wi.RUC_INPUT_FIELDS}
    raw = _from_driver(template, cp) | values
    restored = _read(_input(tmp_path/'input', cfg, raw), cfg)
    driver = wi.initialize_wrfinput_physics(state, restored, cfg)
    driver.set_forcing(gsw=0.)
    for name, value in values.items():
        np.testing.assert_array_equal(cp.asnumpy(driver.fields[name.lower()]).view('u4'), value.view('u4'))

    def consume(arguments, **kwargs):
        for name, value in values.items():
            np.testing.assert_array_equal(cp.asnumpy(arguments[name.lower()]).view('u4'), value.view('u4'))
        raise _ConsumerReached

    monkeypatch.setattr(ruc_runtime, 'ruc_land_surface_step', consume)
    with pytest.raises(_ConsumerReached): step(state, cfg)


@pytest.mark.gpu
@pytest.mark.parametrize('qke_name', ['qke', 'QKE'])
def test_file_qke_reaches_actual_mynn_column_arguments(tmp_path, monkeypatch, qke_name):
    cp = pytest.importorskip('cupy')
    from test_mynn_pbl_runtime import _build
    from gpuwm.core import mynn_pbl_runtime
    from gpuwm.core.dycore import step
    state, cfg, template = _build(nx=6, ny=4)
    values = _values(cfg, qke_name)
    raw = _from_driver(template, cp)
    raw.pop('qke', None)
    raw.update({qke_name: values[qke_name], 'qke_adv': values['qke_adv']})
    restored = _read(_input(tmp_path/'input', cfg, raw), cfg)
    driver = wi.initialize_wrfinput_physics(state, restored, cfg)
    np.testing.assert_array_equal(cp.asnumpy(driver.fields['qke']).view('u4'), values[qke_name].view('u4'))

    def consume(arguments, **kwargs):
        expected = values[qke_name].reshape(cfg.nz, -1).T
        np.testing.assert_array_equal(cp.asnumpy(arguments['qke']).view('u4'), expected.view('u4'))
        assert kwargs['initflag'] == 1  # Solver retains stock cold-start authority.
        raise _ConsumerReached

    monkeypatch.setattr(mynn_pbl_runtime, 'mynn_bl_driver_cuda', consume)
    with pytest.raises(_ConsumerReached): step(state, cfg)
