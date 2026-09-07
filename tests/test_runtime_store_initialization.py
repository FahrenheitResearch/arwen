"""Ordinary case initialization reuses prepared store and forcing contracts."""
from dataclasses import replace

import numpy as np
import pytest

from gpuwm.ingest.lateral_bc import (
    FieldBoundary, RationalTimeLaw, evaluate_boundary_side,
)
from gpuwm.ingest.prepared_cache import (
    PreparedCacheCorruptError, PreparedCacheReader, write_prepared_cache,
)
from gpuwm.ingest.prepared_store import _boundaries_from_cache
from test_prepared_cache import _fixture


@pytest.mark.parametrize('nonlinear', [False, True])
def test_store_loader_retains_boundary_values_and_derivatives(tmp_path, nonlinear):
    initial, met, boundaries = _fixture()
    old = boundaries.intervals[0]
    side = old.fields['u'].west
    if nonlinear:
        side = replace(side, time_law=RationalTimeLaw(
            np.full(side.value.shape, 0.003),
            np.full(side.value.shape, 0.0005)))
    boundaries = replace(boundaries, intervals=(replace(
        old, fields={'u': FieldBoundary(side, side, side, side)}),))
    root = tmp_path / 'cache'
    write_prepared_cache(root, identity={'case': 'forcing-control'},
                         initial_result=initial, met=met, boundaries=boundaries)
    reader = PreparedCacheReader(root, expected_identity={'case': 'forcing-control'})
    restored = _boundaries_from_cache(reader, reader.header['metadata'])
    result = restored.intervals[0].fields['u'].west
    for t in (0., 13., 1499., 3600.):
        for got, expected in zip(evaluate_boundary_side(result, t),
                                 evaluate_boundary_side(side, t)):
            np.testing.assert_array_equal(got, expected)
    assert (result.time_law is not None) is nonlinear


def test_store_loader_refuses_incomplete_time_law(tmp_path):
    initial, met, boundaries = _fixture()
    root = tmp_path / 'cache'
    write_prepared_cache(root, identity={}, initial_result=initial,
                         met=met, boundaries=boundaries)
    reader = PreparedCacheReader(root, expected_identity={})
    # A header with one coefficient must not silently fall back to linear.
    reader.arrays['lbc/0/u/west/rational_time_v1/quadratic'] = {}
    with pytest.raises(PreparedCacheCorruptError, match='incomplete rational'):
        _boundaries_from_cache(reader, reader.header['metadata'])


def test_store_health_checks_rows_outside_the_template(monkeypatch):
    from types import SimpleNamespace
    from gpuwm.core import streaming
    from gpuwm.core.health import HealthCheckError, StoreHealthValidator
    from test_store_health_gate import _state, _store, NZ, NY, NX, ROWS

    state, inventory = _state(ROWS)
    store = _store()
    # The fixture helper returns the full-domain mapping and its auxiliary
    # columns; use its field values as a real descriptor-gate control.
    if isinstance(store, tuple):
        store = store[0]
    bundle = SimpleNamespace(template=state, store=store,
        base=SimpleNamespace(thb=np.full((NZ, NY, NX), 296., np.float32),
                             mub=np.full((NY, NX), 90000., np.float32),
                             p_top=10000.))
    monkeypatch.setattr(streaming, 'streamed_store_inventory',
                        lambda: lambda state, names: inventory)
    health = StoreHealthValidator(bundle, SimpleNamespace(ny=NY, nx=NX))
    assert health.require_healthy(phase='initial').ok
    store['state/qv'][-1, -1, -1] = np.nan
    with pytest.raises(HealthCheckError, match='qv'):
        health.require_healthy(phase='last-domain-row')


def test_case_row_context_preserves_resolved_soil_and_explicit_physics(monkeypatch):
    from types import SimpleNamespace
    from gpuwm import runtime
    from gpuwm.ingest.case_store import CasePhysicsContext

    fields = {name: np.arange(12, dtype=np.float32).reshape(3, 4)
              for name in ('TSK', 'TSLB', 'SMOIS', 'SH2O', 'TMN',
                           'SEAICE', 'SNOW', 'SNOWH')}
    soil_type = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    sst = np.arange(12, dtype=np.float32).reshape(3, 4) + 270.
    context = CasePhysicsContext('explicit-eta', soil_type, sst,
                                 {'co2': 0.00051}, 47)
    captured = {}
    def initialize(result, cfg, met, soil, soil_fields, *args, **kwargs):
        captured.update(kwargs, soil=soil, soil_fields=soil_fields)
        result.state.physics = SimpleNamespace()
    monkeypatch.setattr(runtime, '_initialize_real_case_physics', initialize)
    monkeypatch.setattr(runtime, 'apply_single_domain_pbl_cadence',
                        lambda driver, cfg: captured.update(cadence=cfg))
    cfg = SimpleNamespace(nx=4, ny=2)
    context(SimpleNamespace(state=SimpleNamespace()), cfg, object(),
            SimpleNamespace(fields={k: v[1:] for k, v in fields.items()}),
            {}, {}, object(), object(), row_start=1, domain_rows=3,
            center_lat=-0.25, constant_glw_wm2=333.)
    np.testing.assert_array_equal(captured['reconciled_soil_type'], soil_type[1:])
    np.testing.assert_array_equal(captured['soil_fields']['SST'], sst[1:])
    np.testing.assert_array_equal(captured['soil'].soil_temperature, fields['TSLB'][1:])
    assert captured['vertical'] == 'explicit-eta'
    assert captured['trace_gas_overrides'] == {'co2': 0.00051}
    assert captured['radiation_column_chunk'] == 47
    assert captured['center_lat'] == -0.25
    assert captured['constant_glw_wm2'] == 333.
    assert captured['cadence'] is cfg


@pytest.mark.gpu
@pytest.mark.parametrize("species", [(), ("QC", "QR", "QI", "QS", "QG")])
def test_cuda_transforms_host_state_is_byte_identical_to_cuda_state(species):
    cp = pytest.importorskip('cupy')
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_vertical_coord
    from gpuwm.ingest.real import initialize_real
    from gpuwm.ingest.lateral_bc import domain_boundary_snapshot
    from gpuwm.state_serialization_contract import STATE_SERIALIZED_ATTRS
    from test_real_init import _synthetic_horizontal_snapshot

    ny, nx = 12, 13
    eta = np.array([1., .92, .78, .60, .40, .22, .09, 0.])
    cfg = RunConfig(nx=nx, ny=ny, nz=7, dx=12000., dy=12000.,
        ztop=16000., dt=30., run_seconds=1800., hybrid_opt=2,
        etac=.2, moist=True, terrain_opt=1, base_temp=290.)
    snapshot = _synthetic_horizontal_snapshot(cp, ny, nx)
    if species:
        cfg = replace(cfg, mp_physics=6)
        fields = dict(snapshot.fields)
        for index, name in enumerate(species):
            fields[name] = cp.full_like(snapshot.fields["TT"], 0.0001*(index+1))
        snapshot = replace(snapshot, fields=fields)
    source_orography = np.linspace(200., 800., ny*nx).reshape(ny, nx)
    terrain = source_orography + 120.*np.sin(np.arange(nx))[None, :]
    def initialize(state_backend):
        return initialize_real(snapshot, cfg, make_vertical_coord(
            cfg.nz, hybrid_opt=2, etac=.2, eta_levels=eta), terrain,
            source_orography=source_orography, p_top=10000.,
            preprocess_backend='cuda', state_backend=state_backend,
            analyzed_species=species if species else None)
    resident = initialize('cuda')
    host = initialize('cpu')
    assert isinstance(host.state.u, np.ndarray)
    for name in STATE_SERIALIZED_ATTRS:
        device_array = getattr(resident.state, name, None)
        host_array = getattr(host.state, name, None)
        if device_array is not None:
            assert isinstance(host_array, np.ndarray), name
            np.testing.assert_array_equal(host_array, cp.asnumpy(device_array),
                                          err_msg=name)
    original = domain_boundary_snapshot(resident.state)
    stored = domain_boundary_snapshot(host.state)
    for name in original:
        np.testing.assert_array_equal(stored[name], original[name], err_msg=name)


def test_host_state_estimate_uses_actual_fields_and_retained_times():
    from gpuwm.config import RunConfig
    from gpuwm.core.preflight import estimate_host_state_initialization
    cfg = RunConfig(nx=31, ny=27, nz=9, dx=10000., dy=10000., ztop=16000., dt=30., run_seconds=60.)
    shapes = {'arbitrary-field': (17, 27, 31), 'wind-face': (17, 27, 32)}
    short = estimate_host_state_initialization(cfg, analysis_shapes=shapes,
                                               forcing_times=2)
    long = estimate_host_state_initialization(cfg, analysis_shapes=shapes,
                                              forcing_times=7)
    assert short.device.category_bytes('analysis') == 4*17*27*(31+32)
    assert short.device.category_bytes('state') == 0
    assert short.device.category_bytes('lbc') == 0
    assert short.host_state_bytes > 0
    assert long.host_boundary_bytes > short.host_boundary_bytes
    assert long.device.peak_envelope_bytes == short.device.peak_envelope_bytes


@pytest.mark.parametrize('capacity', ['device_budget_bytes', 'host_available_bytes'])
def test_remaining_initialization_refuses_only_measured_capacity(tmp_path, capacity):
    from types import SimpleNamespace
    from gpuwm.config import RunConfig
    from gpuwm.ingest.case_store import CaseStoreRequest, admit_case_initialization
    cfg = RunConfig(nx=31, ny=27, nz=9, dx=10000., dy=10000., ztop=16000., dt=30., run_seconds=60.)
    met = SimpleNamespace(fields={'actual': np.zeros((17, 27, 31))})
    request = CaseStoreRequest(tmp_path/'cache', resources={capacity: 1})
    with pytest.raises(MemoryError, match='initialization'):
        admit_case_initialization(request, cfg, met, (0, 1, 2))
    assert request.backend == 'cuda'
    unknown = CaseStoreRequest(tmp_path/'unknown')
    report = admit_case_initialization(unknown, cfg, met, (0, 1, 2))
    assert report['device_budget_bytes'] is None
    assert report['host_available_bytes'] is None
    assert 'unpriced' in report['scope']


@pytest.mark.gpu
def test_actual_case_store_releases_prior_slabs_before_allocating(tmp_path, monkeypatch):
    import gc
    import os
    import weakref
    from pathlib import Path
    from gpuwm import runtime
    from gpuwm.case_data import load_experiment_case
    from gpuwm.core import streaming
    from gpuwm.ingest import prepared_store
    from gpuwm.ingest.case_store import CaseStoreRequest, build_case_store

    path = Path(os.environ.get('GPUWM_TEST_RUNTIME_STORE_CASE', 'fixture-unset'))
    if not path.is_file():
        pytest.skip('real ordinary-case input fixture is not configured')
    exp, data = load_experiment_case(path)
    prepared = runtime.prepare_experiment_case(exp, data,
        store_request=CaseStoreRequest(tmp_path/'cache'))
    references = []
    original = prepared_store._slab_state
    def allocate(*args, **kwargs):
        gc.collect()
        assert all(ref() is None for ref in references), 'previous slab remains live'
        result = original(*args, **kwargs)
        references.append(weakref.ref(result))
        return result
    monkeypatch.setattr(prepared_store, '_slab_state', allocate)
    options = streaming.options_for_domain(exp.root, exp.tiles)
    prepared, bundle = build_case_store(prepared, valid_time=exp.start_time,
        decision=streaming.decide(exp.root.run, options), options=options)
    assert len(references) > 1
    assert all(ref() is None for ref in references[:-1])
    assert references[-1]() is bundle.template


def test_ordinary_dispatch_decides_before_preparation_and_keeps_elapsed_clock(tmp_path, monkeypatch):
    from dataclasses import dataclass
    from datetime import timedelta
    from types import SimpleNamespace
    from gpuwm import runtime
    from gpuwm.core import streaming
    from gpuwm.ingest import case_store, preflight
    from test_runtime import _fixture_pair

    exp, data = _fixture_pair(tmp_path)
    options = streaming.StreamingOptions(mode='on', tile_nx=8, tile_ny=8)
    exp = replace(exp, tiles=options)
    events = []
    decision = streaming.StreamingDecision(True, 'test', 8, 8, 2, 16)
    def decide(cfg, options):
        events.append('decide')
        return decision
    monkeypatch.setattr(streaming, 'decide', decide)
    catalog = SimpleNamespace(valid_times=(exp.start_time, exp.start_time+timedelta(hours=1)),
                              excluded_valid_times=())
    monkeypatch.setattr(preflight, 'build_input_catalog',
                        lambda data: events.append('catalog') or catalog)
    monkeypatch.setattr(runtime, 'forcing_snapshots',
                        lambda *a: events.append('decode') or {})
    monkeypatch.setattr(runtime, 'forcing_schedule', lambda *a: catalog.valid_times)
    monkeypatch.setattr(case_store, 'initialization_resources',
                        lambda options: events.append('resources') or {})
    state = SimpleNamespace()
    prepared = SimpleNamespace(cfg=exp.root.run,
        initial_result=SimpleNamespace(state=state, initial_perturbation=None))
    def prepare(*args, **kwargs):
        events.append('prepare')
        assert kwargs['store_request'].backend == 'cuda'
        return prepared
    monkeypatch.setattr(runtime, 'prepare_experiment_case', prepare)
    monkeypatch.setattr(runtime, 'clear_forcing_caches', lambda: events.append('release'))
    monkeypatch.setattr(runtime, 'release_backend_memory', lambda backend: None)
    bundle = object()
    monkeypatch.setattr(case_store, 'build_case_store',
        lambda *a, **kw: events.append('store') or (prepared, bundle))
    def builder(value, *, clock):
        assert value is bundle
        assert clock is None
        return object()
    monkeypatch.setattr(streaming, 'store_domain_builder', builder)
    stepper = object()
    def make_stepper(*args, **kwargs):
        assert kwargs['decision'] is decision
        return stepper
    monkeypatch.setattr(streaming, 'make_stepper', make_stepper)
    @dataclass(frozen=True)
    class Summary:
        wrfout_paths: tuple = ()
        trajectory_digest: object = None
        frame_records: tuple | None = None
    def integrate(*args, **kwargs):
        assert kwargs['stepper'] is stepper
        assert state._streamed_domain is stepper
        assert kwargs['run_seconds'] == exp.run_seconds
        events.append('integrate')
        return Summary()
    monkeypatch.setattr(runtime, 'integrate_prepared_case', integrate)
    runtime.run_experiment(exp, data, tmp_path/'out')
    assert events == ['decide', 'catalog', 'decode', 'resources', 'prepare',
                      'release', 'store', 'integrate']
