"""Bit identity of boundary-only receipts and perimeter selection."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ingest import real, lateral_bc
from test_real_init import _analyzed_hrrr_real_init
from test_real_init_forcing_time_cost import _assert_same_state


@pytest.mark.parametrize('scheme', [8, 28])
def test_boundary_only_does_not_fingerprint_or_read_geopotential(monkeypatch, scheme):
    root, _ = _analyzed_hrrr_real_init(scheme)
    def forbidden(value):
        raise AssertionError('boundary receipt fingerprint was evaluated')
    monkeypatch.setattr(real, 'array_correspondence_fingerprint', forbidden)
    boundary, _ = _analyzed_hrrr_real_init(
        scheme, init_kwargs={'boundary_only': True})
    _assert_same_state(root, boundary)
    assert boundary.total_geopotential is None
    assert boundary.hydrometeor_initialization['vertical_disposition'] == {}
    assert getattr(root.state, '_external_scalar_boundary_fields', None) == getattr(
        boundary.state, '_external_scalar_boundary_fields', None)


@pytest.mark.parametrize('width', [3, 5])
def test_frame_owner_selection_matches_full_snapshot(monkeypatch, width):
    rng = np.random.default_rng(281)
    fields = {name: rng.normal(size=shape).astype(np.float32)
              for name, shape in {'mu': (12, 16), 'u': (4, 12, 17),
                                  'v': (4, 13, 16), 'phi': (5, 12, 16)}.items()}
    monkeypatch.setattr(lateral_bc, '_coupled_device_fields', lambda state: fields)
    old = lateral_bc.StateBoundaryFrames(spec_bdy_width=width, spec_zone=1, relax_zone=2)
    new = lateral_bc.StateBoundaryFrames(spec_bdy_width=width, spec_zone=1, relax_zone=2)
    state = SimpleNamespace()
    old.add_snapshot(lateral_bc.domain_boundary_snapshot(state))
    new.add_state(state)
    for side in old._frames[0]:
        for name in fields:
            assert old._frames[0][side][name].tobytes() == new._frames[0][side][name].tobytes()


def test_selection_copies_only_perimeter(monkeypatch):
    copied = []
    class Owner:
        def __init__(self, value):
            self.value, self.shape, self.ndim = value, value.shape, value.ndim
        def __getitem__(self, selection):
            return Owner(self.value[selection])
        def get(self):
            copied.append(self.value.shape)
            return self.value.copy()
    field = Owner(np.arange(2*10*12, dtype=np.float32).reshape(2, 10, 12))
    for side in ('west', 'east', 'south', 'north'):
        lateral_bc.extract_lateral_side({'qv': field}, side, 2)
    assert copied == [(2, 10, 2), (2, 10, 2), (2, 2, 12), (2, 2, 12)]


@pytest.mark.parametrize('scheme', [8, 28])
def test_gathered_closure_matches_frozen_full_domain(scheme):
    from host_setup_reference import namespace
    from gpuwm.boundary_fields import COLD_START_SEEDED_NUMBERS
    rng = np.random.default_rng(281)
    shape = (4, 7, 9)
    original, gathered = SimpleNamespace(), SimpleNamespace()
    for mass, number in COLD_START_SEEDED_NUMBERS[scheme]:
        q = rng.uniform(0, 0.003, shape).astype(np.float32)
        q.ravel()[::3] = 0
        q.ravel()[1::11] = np.float32(5e-13)
        n = rng.uniform(1, 1e6, shape).astype(np.float32)
        n.ravel()[::2] = 0
        n.ravel()[1::7] = -1
        for state in (original, gathered):
            setattr(state, mass, q.copy())
            setattr(state, number, n.copy())
    # Non-negative installed numbers keep the original whole-array refusal.
    for _, number in COLD_START_SEEDED_NUMBERS[scheme]:
        for state in (original, gathered):
            getattr(state, number)[...] = np.maximum(getattr(state, number), 0)
    cfg = SimpleNamespace(mp_physics=scheme)
    alt = rng.uniform(0.3, 3, shape).astype(np.float32)
    temp = rng.uniform(220, 310, shape).astype(np.float32)
    kw = dict(aerosol_number=rng.uniform(0, 1e8, shape).astype(np.float32),
              landmask=rng.integers(0, 2, shape[-2:]), temperature=temp)
    expected = namespace()['_thompson_cold_start_moment_closure'](
        original, np, cfg, alt, **kw)
    actual = real._thompson_cold_start_moment_closure(gathered, np, cfg, alt, **kw)
    assert actual == expected
    for _, number in COLD_START_SEEDED_NUMBERS[scheme]:
        assert getattr(original, number).tobytes() == getattr(gathered, number).tobytes()


def test_gathering_preserves_chunked_power_elements():
    from gpuwm.core import portable_math as pm, noahmp_libm
    rng = np.random.default_rng(281)
    size = noahmp_libm._POWF_ARRAY_BLOCK * 2 + 37
    base = rng.uniform(1e-12, 30, size).astype(np.float32)
    exponent = rng.uniform(-4, 4, size).astype(np.float32)
    indices = np.flatnonzero(np.arange(size) % 3 != 0)
    full = noahmp_libm.powf_array(base, exponent)
    gathered = noahmp_libm.powf_array(base[indices], exponent[indices])
    assert full[indices].tobytes() == gathered.tobytes()
    base64 = base.astype(np.float64)
    for workers in (1, 3):
        full = pm.power(base64, 0.2854, workers=workers)
        gathered = pm.power(base64[indices], 0.2854, workers=workers)
        assert full[indices].tobytes() == gathered.tobytes()

    # Special powf branches and the last partial block must also be local.
    base = rng.integers(0, 2**32, size, dtype=np.uint32).view(np.float32)
    exponent = rng.choice(np.array([-3., -0.5, 0., 0.5, 2., np.inf, np.nan],
                                   dtype=np.float32), size)
    with np.errstate(all='ignore'):
        full = noahmp_libm.powf_array(base, exponent)
        gathered = noahmp_libm.powf_array(base[indices], exponent[indices])
    assert full[indices].tobytes() == gathered.tobytes()


@pytest.mark.parametrize('inverse', [0., -0., np.nan, np.inf, -np.inf,
                                    np.nextafter(np.float32(0.), np.float32(1.))])
def test_inverse_density_refusal_matches_base(inverse):
    from host_setup_reference import namespace
    states = [SimpleNamespace(qr=np.zeros(1, np.float32), qi=np.zeros(1, np.float32),
                              nr=np.zeros(1, np.float32), ni=np.zeros(1, np.float32))
              for _ in range(2)]
    alt = np.array([inverse], dtype=np.float32)
    cfg = SimpleNamespace(mp_physics=8)
    def run(function, state):
        with np.errstate(all='ignore'):
            try:
                return function(state, np, cfg, alt)
            except ValueError as error:
                return str(error)
    assert run(namespace()['_thompson_cold_start_moment_closure'], states[0]) == run(
        real._thompson_cold_start_moment_closure, states[1])


def test_base_cache_is_exact_and_preparation_owned(monkeypatch):
    from gpuwm.core.grid import make_vertical_coord
    from gpuwm.ingest.preparation_setup import current_setup
    frames = lateral_bc.StateBoundaryFrames()
    coord = make_vertical_coord(4)
    terrain = np.arange(10*12, dtype=np.float64).reshape(10, 12)
    expected = real._make_real_base_uncached(coord, terrain, 5000., 290.)
    builds = []
    original = real._make_real_base_uncached
    def counted(*args, **kwargs):
        builds.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(real, '_make_real_base_uncached', counted)
    actual = real._make_real_base(coord, terrain, 5000., 290.)
    for name in ('mub', 'pb', 'alb', 'thb', 'phb', 'terrain_z'):
        assert getattr(expected, name).tobytes() == getattr(actual, name).tobytes()
    actual.pb[0, 0, 0] = np.nextafter(actual.pb[0, 0, 0], np.inf)
    again = real._make_real_base(coord, terrain.copy(), 5000., 290.)
    assert again.pb.tobytes() == expected.pb.tobytes()
    assert len(builds) == 1
    changed = terrain.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], 1.)
    real._make_real_base(coord, changed, 5000., 290.)
    assert len(builds) == 2 and len(frames._setup.base) == 1
    frames.add_snapshot({'mu': terrain}, index=0)
    frames.add_snapshot({'mu': terrain + 1}, index=1)
    frames.interval(0, [0., 60.])
    assert current_setup() is None
    assert not frames._setup.base and not frames._setup.horizontal


def test_abandoned_setup_has_no_global_owner():
    import gc
    import weakref
    from gpuwm.ingest.preparation_setup import current_setup
    frames = lateral_bc.StateBoundaryFrames()
    reference = weakref.ref(frames._setup)
    del frames
    gc.collect()
    assert reference() is None and current_setup() is None


def test_child_setup_returns_to_parent_when_closed():
    from gpuwm.ingest.preparation_setup import current_setup
    parent = lateral_bc.StateBoundaryFrames()
    child = lateral_bc.StateBoundaryFrames()
    assert current_setup() is child._setup
    child._setup.close()
    assert current_setup() is parent._setup
    parent._setup.close()
    assert current_setup() is None


def test_horizontal_setup_hits_only_exact_inputs(monkeypatch):
    from gpuwm.ingest import horiz
    from types import MappingProxyType
    calls = []
    class Engine:
        __hash__ = None
        def regular_plan(self, *args):
            calls.append(1)
            return object()
    engine = Engine()
    frames = lateral_bc.StateBoundaryFrames()
    projection = MappingProxyType({'name': 'lambert',
                                   'parameters': MappingProxyType({'axis_unit_m': 1.})})
    monkeypatch.setattr(horiz, 'declared_source_projection', lambda snapshot: projection)
    monkeypatch.setattr(horiz, 'source_coordinate_transform',
                        lambda snapshot: (lambda lat, lon: (lat, lon), False))
    monkeypatch.setattr(horiz, '_refuse_uncovered_in_source_plane', lambda *args: None)
    snap = SimpleNamespace(latitude=np.arange(5.), longitude=np.arange(6.))
    targets = tuple(np.full((2, 3), float(i)) for i in range(6))
    first = horiz._horizontal_domain_setup(snap, targets, engine)
    second = horiz._horizontal_domain_setup(snap, tuple(x.copy() for x in targets), engine)
    assert first is second and len(calls) == 3
    snap.latitude[0] = np.nextafter(0., 1.)
    assert horiz._horizontal_domain_setup(snap, targets, engine) is not first
    assert len(calls) == 6
    frames._setup.close()


def test_parallel_receipt_json_matches_frozen_base(monkeypatch):
    import json
    from host_setup_reference import namespace
    frozen = namespace()['build_hrrr_hydrometeor_vertical_disposition']
    expected, _ = _analyzed_hrrr_real_init(8)
    baseline = expected.hydrometeor_initialization['vertical_disposition']
    def compare(*args, **kwargs):
        before = frozen(*args, **kwargs)
        after = original(*args, **kwargs)
        assert json.dumps(before, sort_keys=True).encode() == json.dumps(after, sort_keys=True).encode()
        return after
    original = real.build_hrrr_hydrometeor_vertical_disposition
    monkeypatch.setattr(real, '_RECEIPT_HASH_MIN_CELLS', 0)
    monkeypatch.setattr(real, 'build_hrrr_hydrometeor_vertical_disposition', compare)
    actual, _ = _analyzed_hrrr_real_init(8)
    assert actual.hydrometeor_initialization['vertical_disposition'] == baseline


def test_retained_plan_and_receipt_allocation_inventory_is_inside_priced_margin():
    from test_preparation_price import _cfg
    from gpuwm.ingest import preparation_price as pp
    for nx, ny, nz in ((800, 600, 50), (1797, 1057, 50)):
        cfg = _cfg(nx, ny, nz)
        inventory = pp.SourceInventory(levels=39, level_fields=11, surface_planes=29)
        terms = pp._build(cfg, inventory)
        mass, u, v = pp._columns(cfg)
        # RegularGpuPlan owns x/y and supported x/y for each staggering.
        retained_plans = 4 * 4 * (mass + u + v)
        replay_cells = nz * mass
        # uint32 magnitude, nonzero mask, comparison and subnormal mask.
        receipt_mask_and_packing = 8 * replay_cells + (replay_cells + 7) // 8
        assert retained_plans + receipt_mask_and_packing <= terms['setup_residual']
    shapes = {'T': (3, 4, 7), 'U': (3, 4, 7), 'V': (3, 4, 7),
              'RH': (3, 4, 7), 'GHT': (3, 4, 7)}
    assert not pp.SourceInventory.from_shapes(shapes).hydrometeor_replays
    shapes['QR'] = (3, 4, 7)
    assert pp.SourceInventory.from_shapes(shapes).hydrometeor_replays
