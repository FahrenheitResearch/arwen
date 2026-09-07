"""Nonlinear forcing shares the existing clock, tile and identity contracts."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import os

import numpy as np
import pytest

from gpuwm.ingest import lateral_bc as lbc


def _boundaries(*, nonlinear=True, intervals=2):
    rng = np.random.default_rng(91)
    frame = {name: rng.normal(size=shape) for name, shape in {
        'u': (2, 16, 19), 'v': (2, 17, 18), 'theta': (2, 16, 18),
        'phi': (3, 16, 18), 'mu': (1, 16, 18), 'qv': (2, 16, 18)}.items()}
    frames = [{name: array + 0.1*i for name, array in frame.items()}
              for i in range(intervals+1)]
    result = lbc.build_lateral_boundaries(frames, [100.*i for i in range(intervals+1)])
    if not nonlinear:
        return result
    output = []
    for interval in result.intervals:
        fields = dict(interval.fields)
        sides = {}
        for name in ('west', 'east', 'south', 'north'):
            side = getattr(fields['theta'], name)
            law = lbc.RationalTimeLaw(np.full(side.value.shape, 0.003),
                                      np.full(side.value.shape, 0.0005))
            sides[name] = replace(side, time_law=law)
        fields['theta'] = lbc.FieldBoundary(**sides)
        output.append(replace(interval, fields=fields))
    return replace(result, intervals=tuple(output))


def _host_state():
    pool = {}
    def scratch(shape, slot):
        assert slot not in pool or pool[slot].shape == tuple(shape)
        return pool.setdefault(slot, np.zeros(shape, np.float32))
    return SimpleNamespace(_host_setup_state=True, _scratch=pool, scratch=scratch)


def test_rational_value_and_derivative_against_independent_quotient():
    # This caller is not WRF: numerator 3+5t+7t², denominator 2+0.25t.
    side = lbc.SideBoundary(np.array([1.5]), np.array([2.3125]),
        lbc.RationalTimeLaw(np.array([3.5]), np.array([0.125])))
    for t in (0., 0.125, 3., 19.):
        value, derivative = lbc.evaluate_boundary_side(side, t)
        numerator, denominator = 3+5*t+7*t*t, 2+0.25*t
        np.testing.assert_allclose(value, numerator/denominator, rtol=1e-15)
        np.testing.assert_allclose(derivative,
            ((5+14*t)*denominator-0.25*numerator)/(denominator*denominator), rtol=1e-15)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_coefficient_is_refused(bad):
    with pytest.raises(ValueError, match='finite'):
        lbc.RationalTimeLaw(np.array([bad]), np.array([0.]))


def test_denominator_crossing_and_malformed_shapes_are_refused():
    side = lbc.SideBoundary(np.ones((1, 10, 5)), np.zeros((1, 10, 5)),
        lbc.RationalTimeLaw(np.zeros((1, 10, 5)), np.full((1, 10, 5), -0.01)))
    with pytest.raises(ValueError, match='denominator crosses zero'):
        lbc.BoundaryInterval(0, 100, {'theta': lbc.FieldBoundary(side, side, side, side)})
    with pytest.raises(ValueError, match='shapes differ'):
        lbc.SideBoundary(np.ones(3), np.zeros(3),
                         lbc.RationalTimeLaw(np.zeros(2), np.zeros(2)))
    with pytest.raises(ValueError):
        side.time_law.quadratic.setflags(write=True)


@pytest.mark.parametrize('streaming', [False, True])
def test_registered_evaluation_uses_bound_clock_and_resets_without_double_dtbc(streaming):
    boundaries = _boundaries()
    state = _host_state()
    attach = lbc.attach_streaming_lateral_boundaries if streaming else lbc.attach_lateral_boundaries
    attach(state, boundaries)
    resident = state._lateral_boundary_device
    clock = SimpleNamespace(elapsed_seconds=0., dtbc_launch_fp32=np.float32(12.))
    lbc.bind_lateral_boundary_clock(state, clock)
    cfg = SimpleNamespace(nested=False, dt=12., spec_exp=0.)
    for elapsed, offset in ((0., 12.), (88., 100.), (100., 12.), (124., 36.)):
        clock.elapsed_seconds, clock.dtbc_launch_fp32 = elapsed, np.float32(offset)
        device, dtbc, _, _ = lbc._active_device_interval(state, cfg)
        assert dtbc == 0, 'already evaluated forcing must not interpolate again'
        host = boundaries.interval_at(elapsed).fields['theta'].west
        rounded = lbc.SideBoundary(host.value.astype(np.float32), host.tendency.astype(np.float32),
            lbc.RationalTimeLaw(host.time_law.quadratic.astype(np.float32),
                                host.time_law.denominator_rate.astype(np.float32)))
        expected = lbc.evaluate_boundary_side(rounded, offset)
        np.testing.assert_array_equal(device.fields['theta'].west.value, expected[0].astype(np.float32))
        np.testing.assert_array_equal(device.fields['theta'].west.tendency, expected[1].astype(np.float32))
        assert lbc._active_device_interval(state, cfg)[0] is device
    shapes = lbc.boundary_storage_shapes(boundaries, streaming=streaming)
    assert {name: value.shape for name, value in state._scratch.items()} == shapes
    assert resident.device_nbytes == sum(value.nbytes for value in state._scratch.values())
    assert resident.external_reload_count == (2 if streaming else 0)


def test_linear_attachment_keeps_original_arrays_offset_and_identity_schema():
    state = _host_state()
    boundaries = _boundaries(nonlinear=False)
    lbc.attach_lateral_boundaries(state, boundaries)
    state.elapsed_seconds = 17.
    original = state._lateral_boundary_device.intervals[0]
    device, offset, *_ = lbc._active_device_interval(
        state, SimpleNamespace(nested=False, dt=12., spec_exp=0.))
    assert device is original and offset == 17.
    assert 'lbc_evaluated_tables' not in state._scratch
    from gpuwm.state_serialization_contract import lateral_boundary_prefix_identity
    assert lateral_boundary_prefix_identity(state)['schema'] == 'gpuwm-lateral-boundary-prefix-v2'


def test_time_coefficients_affect_restart_identity_and_survive_tile_windows():
    from gpuwm.core.streaming import window_boundaries, owned_edges
    from gpuwm.state_serialization_contract import lateral_boundary_prefix_identity
    from gpuwm.io.restart import _validated_forcing_prefix
    from tilestream.spec import plan_tiles
    boundaries = _boundaries(intervals=1)
    state = SimpleNamespace(lateral_boundaries=boundaries)
    identity = lateral_boundary_prefix_identity(state)
    assert identity['schema'] == 'gpuwm-lateral-boundary-prefix-v3'
    _validated_forcing_prefix(identity, label='test', path=Path('checkpoint'))
    linear = lateral_boundary_prefix_identity(SimpleNamespace(lateral_boundaries=_boundaries(nonlinear=False)))
    assert identity['intervals'][0]['sha256'] != linear['intervals'][0]['sha256']
    assert identity['intervals'][0]['end_frame_sha256'] != linear['intervals'][0]['end_frame_sha256']
    for spec in plan_tiles(18, 16, 9, 8, 4, periodic=False):
        tile = window_boundaries(boundaries, spec)
        owns = owned_edges(spec)
        for name in ('west', 'east', 'south', 'north'):
            source = getattr(boundaries.intervals[0].fields['theta'], name)
            result = getattr(tile.intervals[0].fields['theta'], name)
            if owns[name]:
                index = ((slice(None), slice(spec.cj0, spec.cj0+spec.cny), slice(None))
                    if name in ('west', 'east') else
                    (slice(None), slice(None), slice(spec.ci0, spec.ci0+spec.cnx)))
                np.testing.assert_array_equal(lbc.evaluate_boundary_side(result, 43.)[0],
                    lbc.evaluate_boundary_side(source, 43.)[0][index])
            else:
                assert not np.any(result.time_law.quadratic)
                assert not np.any(result.time_law.denominator_rate)


def test_preflight_prices_actual_coefficients_and_evaluation_scratch():
    from gpuwm.config import RunConfig
    from gpuwm.experiment import experiment_from_run_config
    from gpuwm.core.preflight import estimate_experiment
    cfg = RunConfig(nx=18, ny=16, nz=2, dx=3000, dy=3000, ztop=10000, dt=12,
                    run_seconds=24, specified=True, moist=True)
    exp = experiment_from_run_config(cfg, datetime(2021, 12, 30, 17))
    bounds = _boundaries()
    estimate = estimate_experiment(exp, forcing_interval_seconds=100,
        forcing_intervals=2, lateral_boundaries=bounds)
    shapes = lbc.boundary_storage_shapes(bounds)
    items = {item.name:item for item in estimate.domains[0].items}
    for name, shape in shapes.items():
        assert items[name].shape == shape
        assert items[name].nbytes == 4*np.prod(shape)


@pytest.fixture(scope='module')
def moist_pair():
    directory = os.environ.get('GPUWM_TEST_WRF_MOIST_DIRECTORY')
    if not directory:
        pytest.skip('set GPUWM_TEST_WRF_MOIST_DIRECTORY to a real.exe moist-boundary pair')
    from gpuwm.wrfinput_door import resolve_wrfinput_run
    from gpuwm.ingest.wrfinput import read_wrfinput, read_wrfbdy
    from gpuwm.config import soil_layer_count
    run = resolve_wrfinput_run(Path(directory))
    cfg = run.experiment.root.run
    dimensions = dict(west_east=cfg.nx, west_east_stag=cfg.nx+1,
        south_north=cfg.ny, south_north_stag=cfg.ny+1,
        bottom_top=cfg.nz, bottom_top_stag=cfg.nz+1, soil_layers_stag=soil_layer_count(cfg))
    restored = read_wrfinput(run.wrfinput_paths[1], expected_dimensions=dimensions, cfg=cfg)
    assert int(restored.global_attributes['USE_THETA_M']) == 1
    bounds = read_wrfbdy(run.wrfbdy_path, run_seconds=60, restored=restored,
        forcing_interval_seconds=run.coverage.forcing_interval_seconds,
        spec_bdy_width=cfg.spec_bdy_width)
    return run, bounds


def test_real_moist_forcing_against_independent_file_thermodynamics(moist_pair):
    netCDF4 = pytest.importorskip('netCDF4')
    run, bounds = moist_pair
    with netCDF4.Dataset(run.wrfinput_paths[1]) as f:
        mub = np.asarray(f['MUB'][0], np.float32)
        c1 = np.asarray(f['C1H'][0], np.float32)[None, :, None]
        c2 = np.asarray(f['C2H'][0], np.float32)[None, :, None]
    a = float((np.float32(461.6)/np.float32(287.0)))
    with netCDF4.Dataset(run.wrfbdy_path) as f:
        errors = []
        for name, suffix, strip in [('west','XS',mub[:,:5].T),('east','XE',mub[:,-5:][:,::-1].T),
                                    ('south','YS',mub[:5]),('north','YE',mub[-5:][::-1])]:
            raw = lambda key: np.asarray(f[key+suffix][0], np.float32)
            A, Ad = raw('T_B').astype(float), raw('T_BT').astype(float)
            Q, Qd = raw('QVAPOR_B').astype(float), raw('QVAPOR_BT').astype(float)
            mu, mud = raw('MU_B')[:,None,:], raw('MU_BT')[:,None,:]
            M = (c1*mu+(c1*strip[:,None,:]+c2)).astype(float)
            G = (c1*(strip[:,None,:]+mu)+c2).astype(float)
            Md = c1.astype(float)*mud.astype(float)
            def direct(t):
                mass, target = M+t*Md, G+t*Md
                theta_m = (A+t*Ad)/mass+300
                qv = (Q+t*Qd)/mass
                return target*(theta_m/(1+a*qv)-300)
            side = getattr(bounds.intervals[0].fields['theta'], name)
            transpose = ((1,2,0) if name in ('west','east') else (1,0,2))
            for t in (12., 900., 1800., 2700., 3600.):
                expected = direct(t).transpose(transpose)
                target = (G+t*Md).transpose(transpose)
                actual, derivative = lbc.evaluate_boundary_side(side, t)
                assert np.max(np.abs(actual-expected)/target) < 3e-12
                difference = ((direct(t+0.1)-direct(t-0.1))/0.2).transpose(transpose)
                assert np.max(np.abs(derivative-difference)/target) < 3e-11
                linear = direct(0)+t/3600*(direct(3600)-direct(0))
                errors.append(float(np.max(np.abs(linear-direct(t))/(G+t*Md))))
        assert max(errors) > 0.001, 'this fixture must distinguish endpoint-only conversion'


@pytest.mark.gpu
def test_gpu_evaluator_matches_host_for_resident_and_tile_attachments():
    from conftest import HAS_GPU
    if not HAS_GPU:
        pytest.skip('no CUDA GPU / cupy')
    import cupy as cp
    from gpuwm.core.streaming import window_boundaries
    from tilestream.spec import plan_tiles
    boundaries = _boundaries()
    tiles = plan_tiles(18, 16, 9, 8, 4, periodic=False)
    for tables in [boundaries, *(window_boundaries(boundaries, spec) for spec in tiles)]:
        pool = {}
        def scratch(shape, slot):
            return pool.setdefault(slot, cp.zeros(shape, cp.float32))
        state = SimpleNamespace(_scratch=pool, scratch=scratch)
        lbc.attach_streaming_lateral_boundaries(state, tables)
        cfg = SimpleNamespace(nested=False, dt=12., spec_exp=0.)
        for seconds in (12., 43., 112., 143.):
            state.elapsed_seconds = seconds
            device, offset, *_ = lbc._active_device_interval(state, cfg)
            assert offset == 0
            interval = tables.interval_at(seconds)
            source = lbc._resident_interval(state, interval)
            for name in ('west', 'east', 'south', 'north'):
                side = getattr(source.fields['theta'], name)
                law = side.time_law
                host = lbc.SideBoundary(cp.asnumpy(side.value), cp.asnumpy(side.tendency),
                    None if law is None else lbc.RationalTimeLaw(
                        cp.asnumpy(law.quadratic), cp.asnumpy(law.denominator_rate)))
                expected = lbc.evaluate_boundary_side(host, seconds-interval.start_seconds)
                result = getattr(device.fields['theta'], name)
                np.testing.assert_allclose(cp.asnumpy(result.value), expected[0], rtol=2e-7, atol=1e-7)
                np.testing.assert_allclose(cp.asnumpy(result.tendency), expected[1], rtol=2e-7, atol=1e-7)
