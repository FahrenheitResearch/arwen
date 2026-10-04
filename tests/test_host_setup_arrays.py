"""Large native setup copies retain base-state bytes and owned caches."""
from types import SimpleNamespace
import numpy as np
import pytest

from gpuwm.core import portable_math as pm
from gpuwm.ingest import host_arrays


@pytest.fixture(autouse=True)
def native_arrays():
    if not hasattr(pm._load(), "gpuwm_host_geopotential"):
        pytest.skip("CPU bridge predates native host setup arrays")


@pytest.mark.parametrize("workers", [1, 8, 24])
def test_native_cast_copy_and_geopotential_match_numpy(workers):
    from gpuwm.core import constants as c
    rng = np.random.default_rng(940)
    for dtype in (np.float32, np.float64):
        values = rng.uniform(-1e7, 1e7, 100_003).astype(dtype)
        values[:6] = [0, -0., np.nan, np.inf, -np.inf, np.nextafter(dtype(0), dtype(1))]
        output = np.empty(values.shape, np.float32)
        with pm.worker_limit(workers):
            assert host_arrays.copy_float32(output, values)
        assert output.tobytes() == values.astype(np.float32).tobytes()
    host = np.cumsum(rng.uniform(10, 2000, (9, 67, 71)), axis=0)
    for special in (None, np.nan, -0.0):
        source = host.copy()
        if special is not None:
            source[:, 3, 5] = special
        state = SimpleNamespace(phb=np.empty(source.shape, np.float32),
                                dphb_resid=np.empty(source[1:].shape, np.float32))
        stored = source.astype(np.float32)
        with np.errstate(all="ignore"), pm.worker_limit(workers):
            residual = (np.diff(source, axis=0) - np.diff(stored, axis=0).astype(np.float64)).astype(np.float32)
            minimum = float(np.diff(0.5 * (source[:-1] + source[1:]) / c.G, axis=0).min())
            assert host_arrays.geopotential_cache(state, source, c.G)
        assert state.phb.tobytes() == stored.tobytes()
        assert state.dphb_resid.tobytes() == residual.tobytes()
        assert state._phb_host.tobytes() == source.tobytes()
        assert np.float64(state._dz_min).tobytes() == np.float64(minimum).tobytes()
        source[0, 0, 0] = -1
        assert state._phb_host[0, 0, 0] != -1


def test_native_load_base_is_opted_in_only_by_preparation(monkeypatch):
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import BaseState, make_vertical_coord
    from gpuwm.core.state import DomainState
    cfg = RunConfig(nx=65, ny=67, nz=4, dx=3000., dy=3000., ztop=15000.,
                    dt=10., run_seconds=3600., moist=True, mp_physics=6, terrain_opt=1)
    coord = make_vertical_coord(4, hybrid_opt=0)
    shape = (4, 67, 65)
    base = BaseState(mub=np.full(shape[1:], 90000.), p_top=10000.,
                     pb=np.linspace(95000., 20000., 4)[:, None, None] * np.ones(shape),
                     alb=np.full(shape, 0.9), thb=np.full(shape, 300.),
                     phb=np.linspace(0., 1.4e5, 5)[:, None, None] * np.ones((5, 67, 65)),
                     terrain_z=np.zeros(shape[1:]))
    expected, actual = (DomainState(cfg, array_module=np) for _ in range(2))
    invoked = []
    original = host_arrays.geopotential_cache
    def record(*args):
        invoked.append(True)
        return original(*args)
    monkeypatch.setattr(host_arrays, "geopotential_cache", record)
    expected.load_base(coord, base)
    assert not invoked
    with pm.worker_limit(8):
        actual.load_base(coord, base, native_host=True)
    assert invoked
    for name, value in vars(expected).items():
        other = getattr(actual, name)
        if isinstance(value, np.ndarray):
            assert other.tobytes() == value.tobytes(), name
    assert actual._dz_min == expected._dz_min
    assert actual.height_half().tobytes() == expected.height_half().tobytes()


def test_copy_refuses_broadcast_alias_and_readonly_destinations():
    source = np.arange(20000, dtype=np.float64)
    assert not host_arrays.copy_float32(np.empty((2, 20000), np.float32), source)
    target = source.astype(np.float32)
    assert not host_arrays.copy_float32(target, target)
    target.flags.writeable = False
    assert not host_arrays.copy_float32(target, source)


def test_geopotential_declines_aliases_and_wrong_buffer_contracts():
    from gpuwm.core import constants as c
    source = np.arange(20000, dtype=np.float64).reshape(4, 50, 100)
    stored = np.empty(source.shape, np.float32)
    residual = np.empty(source[1:].shape, np.float32)
    states = (
        SimpleNamespace(phb=source.view(np.float32).reshape(-1)[:source.size].reshape(source.shape), dphb_resid=residual),
        SimpleNamespace(phb=stored, dphb_resid=source.view(np.float32).reshape(-1)[:residual.size].reshape(residual.shape)),
        SimpleNamespace(phb=stored, dphb_resid=stored[1:]),
        SimpleNamespace(phb=stored.astype(np.float64), dphb_resid=residual),
        SimpleNamespace(phb=stored, dphb_resid=np.empty((3, 100, 50), np.float32)),
    )
    before = source.tobytes()
    for state in states:
        assert not host_arrays.geopotential_cache(state, source, c.G)
        assert source.tobytes() == before


@pytest.mark.parametrize("workers", [1, 8, 24])
def test_staggering_sums_and_difference_preserve_numpy_words(workers):
    from gpuwm.ingest.real import _pressure_at_u, _pressure_at_v
    if not hasattr(pm._load(), "gpuwm_host_stagger_f64"):
        pytest.skip("CPU bridge predates native setup arithmetic")
    rng = np.random.default_rng(381)
    source = rng.uniform(-1e7, 1e7, (7, 53, 67))
    special = np.array([0., -0., np.inf, -np.inf, np.nan,
                        np.nextafter(0., 1.), np.finfo(np.float64).max])
    source.ravel()[:special.size] = special
    with np.errstate(all="ignore"):
        for axis, reference in ((1, _pressure_at_v), (2, _pressure_at_u)):
            actual = host_arrays.stagger_pressure(source, axis, workers=workers)
            assert actual.tobytes() == reference(source).tobytes()
            # Copied outer faces preserve payloads and signed zeros exactly.
            for shape in ((1, 1, 1), (2, 1, 7), (2, 7, 1)):
                small = np.resize(source, shape)
                assert host_arrays.stagger_pressure(small, axis, workers=workers).tobytes() == reference(small).tobytes()
        for count in (1, 2, 5):
            fields = [rng.uniform(-1e7, 1e7, source.shape).astype(
                np.float64 if index % 2 else np.float32) for index in range(count)]
            for field in fields:
                field.ravel()[:special.size] = special
            expected = fields[0].astype(np.float64)
            for field in fields[1:]:
                expected = expected + field.astype(np.float64)
            actual = host_arrays.sum_fields(fields, workers=workers)
            assert actual.tobytes() == expected.tobytes()
        for second in (source.copy(), -source, rng.uniform(-1e7, 1e7, source.shape)):
            expected = (source - second).astype(np.float32)
            actual = np.empty(source.shape, np.float32)
            assert host_arrays.difference_float32(actual, source, second, workers=workers)
            assert actual.tobytes() == expected.tobytes()
        raw_fields = [rng.integers(0, 2**64, size=100_003, dtype=np.uint64).view(np.float64)
                      for _ in range(3)]
        raw_fields[0][:4] = np.array([0x7ff8123456789abc, 0xfff8456789abcdef,
                                     0x7ff0000000000001, 0xfff0000000000001], dtype=np.uint64).view(np.float64)
        raw_fields[1][:4] = raw_fields[0][:4][::-1]
        expected = (raw_fields[0] + raw_fields[1]) + raw_fields[2]
        assert host_arrays.sum_fields(raw_fields, workers=workers).tobytes() == expected.tobytes()
        actual = np.empty(raw_fields[0].shape, np.float32)
        assert host_arrays.difference_float32(actual, raw_fields[0], raw_fields[1], workers=workers)
        assert actual.tobytes() == (raw_fields[0] - raw_fields[1]).astype(np.float32).tobytes()


def test_setup_arithmetic_falls_back_for_unsupported_buffer_contracts():
    source = np.arange(4 * 51 * 67, dtype=np.float64).reshape(4, 51, 67)
    target = np.empty(source.shape, np.float32)
    assert host_arrays.stagger_pressure(source[:, ::2], 1) is None
    assert host_arrays.stagger_pressure(source.astype(np.float32), 1) is None
    assert host_arrays.stagger_pressure(source, 0) is None
    assert host_arrays.stagger_pressure(np.empty((2, 0, 3)), 2) is None
    assert host_arrays.sum_fields([]) is None
    assert host_arrays.sum_fields([source, source[0]]) is None
    assert host_arrays.sum_fields([source[:, ::2]]) is None
    assert host_arrays.sum_fields([source.astype(np.int64)]) is None
    assert not host_arrays.difference_float32(target, source, source[0])
    assert not host_arrays.difference_float32(target, source.astype(np.float32), source)
    target.flags.writeable = False
    assert not host_arrays.difference_float32(target, source, source)
    aliased = source.view(np.float32).reshape(-1)[:source.size].reshape(source.shape)
    before = source.tobytes()
    assert not host_arrays.difference_float32(aliased, source, source)
    assert source.tobytes() == before
    unaligned = np.ndarray(source.shape, dtype=np.float64,
                           buffer=bytearray(source.nbytes + 1), offset=1)
    assert unaligned.flags.c_contiguous and not unaligned.flags.aligned
    assert host_arrays.stagger_pressure(unaligned, 1) is None
    assert host_arrays.sum_fields([unaligned]) is None
    target.flags.writeable = True
    assert not host_arrays.copy_float32(target, unaligned)
    assert not host_arrays.difference_float32(target, source, unaligned)
    state = SimpleNamespace(phb=target, dphb_resid=np.empty(source[1:].shape, np.float32))
    assert not host_arrays.geopotential_cache(state, unaligned, 9.81)


def test_owned_host_float64_preserves_layout_bits_and_independent_storage():
    from gpuwm.ingest.real import _host, _host_owned_float64
    rng = np.random.default_rng(3309)
    for dtype in (np.float32, np.float64, ">f4", ">f8"):
        base = rng.uniform(-10, 10, (5, 7, 11)).astype(dtype)
        base.flat[:6] = [0., -0., np.nan, np.inf, -np.inf, np.nextafter(np.float32(0), np.float32(1))]
        for value in (base, np.asfortranarray(base), base[:, ::2, ::-1],
                      base.transpose(2, 0, 1), np.array(-0., dtype=dtype)):
            value.flags.writeable = False
            with np.errstate(all="ignore"):
                expected = _host(value).astype(np.float64)
                actual = _host_owned_float64(value)
            assert actual.dtype == np.float64
            assert actual.shape == expected.shape
            assert actual.strides == expected.strides
            assert actual.tobytes() == expected.tobytes()
            assert actual.flags.owndata and actual.flags.writeable
            assert not np.shares_memory(actual, value)
            before = value.tobytes()
            actual.flat[0] = 12345.
            assert value.tobytes() == before
    class HostTransfer:
        def __init__(self):
            self.values = np.arange(21, dtype=np.float64)
            self.calls = 0
        def get(self):
            self.calls += 1
            return self.values
    transfer = HostTransfer()
    actual = _host_owned_float64(transfer)
    assert transfer.calls == 1
    transfer.values[:] = -1
    assert actual.tobytes() == np.arange(21, dtype=np.float64).tobytes()


def test_cpu_real_owned_widening_matches_previous_copy_contract(monkeypatch):
    from gpuwm.ingest import real
    from test_real_init import _analyzed_hrrr_real_init
    original = real._host_owned_float64
    calls = []
    def observed(value):
        calls.append((value.shape, value.dtype))
        return original(value)
    monkeypatch.setattr(real, "_host_owned_float64", observed)
    actual, _cfg = _analyzed_hrrr_real_init(6, preprocess_backend="cpu",
        init_kwargs={"preprocess_workers": 8})
    # This fixture selects direct qv, so the absent RH retains None.
    assert len(calls) == 2
    assert all(dtype == np.float32 for _shape, dtype in calls)
    monkeypatch.setattr(real, "_host_owned_float64", lambda value: real._host(value).astype(np.float64))
    expected, _cfg = _analyzed_hrrr_real_init(6, preprocess_backend="cpu",
        init_kwargs={"preprocess_workers": 8})
    assert actual.hydrometeor_initialization == expected.hydrometeor_initialization
    for container_name in ("state", "base"):
        for name, value in vars(getattr(expected, container_name)).items():
            if isinstance(value, np.ndarray):
                assert getattr(getattr(actual, container_name), name).tobytes() == value.tobytes(), name
    for name, value in vars(expected).items():
        if isinstance(value, np.ndarray):
            assert getattr(actual, name).tobytes() == value.tobytes(), name


@pytest.mark.parametrize("workers", [1, 8, 24])
@pytest.mark.parametrize("pressure_dtype", [np.float32, np.float64])
def test_native_deepest_level_is_the_numpy_selection(workers, pressure_dtype):
    """The number fields' surface pseudo-level, selected in Rust.

    Guards the data path the native number fields added to ingest: each
    column takes the field at its level of greatest pressure, and the
    bytes are the ones ``take_along_axis(field, argmax(pressure))`` gives,
    for either level order, for ties (the first level wins) and for NaN
    (the first NaN is the maximum).
    """
    if not hasattr(pm._load(), "gpuwm_host_deepest_level_f32"):
        pytest.skip("CPU bridge predates the native deepest-level selection")
    rng = np.random.default_rng(2850)
    shape = (9, 67, 71)
    pressure = np.sort(rng.uniform(5000, 105000, shape), axis=0).astype(pressure_dtype)
    # Half the columns run surface-first, half top-first; some are shuffled.
    pressure[:, ::2] = pressure[::-1, ::2]
    shuffled = rng.permuted(pressure[:, 5:9], axis=0)
    pressure[:, 5:9] = shuffled
    pressure[3, 11, 13] = pressure[6, 11, 13] = pressure[:, 11, 13].max() + 1
    pressure[[2, 5], 20, 21] = np.nan
    pressure[0, 22, 23] = np.nan
    pressure[:, 30, 31] = 0.0
    pressure[4, 30, 31] = -0.0
    field = rng.uniform(0, 1e9, shape).astype(np.float32)
    field[:, 40, 41] = [0, -0., np.nan, np.inf, -np.inf, 1e-45, 1, 2, 3]
    with np.errstate(all="ignore"):
        expected = np.take_along_axis(
            field, np.argmax(pressure, axis=0)[None, ...], axis=0)[0]
    with pm.worker_limit(workers):
        actual = host_arrays.deepest_level(pressure, field)
    assert actual is not None
    assert actual.dtype == np.float32 and actual.shape == shape[1:]
    assert actual.tobytes() == expected.tobytes()
    # The reference path keeps what the library does not take.
    assert host_arrays.deepest_level(pressure[:, ::2], field[:, ::2]) is None
    assert host_arrays.deepest_level(pressure, field.astype(np.float64)) is None
    assert host_arrays.deepest_level(pressure[0], field[0]) is None
