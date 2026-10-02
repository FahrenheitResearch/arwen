"""Bitwise REAL helper gates. No tolerance comparisons are used."""
import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.grid import make_vertical_coord
from gpuwm.ingest import real as host
from gpuwm.ingest import real_device as dev

pytestmark = requires_gpu


def identical(a, b):
    if isinstance(a, tuple):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            identical(left, right)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            identical(a[key], b[key])
    else:
        if hasattr(b, "get"):
            b = b.get()
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert a.tobytes() == b.tobytes()


@pytest.fixture
def columns():
    rng = np.random.default_rng(281)
    shape = (9, 7, 11)
    p = np.broadcast_to(np.linspace(102000, 5000, 9)[:, None, None], shape).copy()
    p += rng.uniform(-30, 30, shape)
    q = rng.uniform(-0.01, 0.025, shape)
    t = rng.uniform(215, 305, shape)
    z = np.broadcast_to(np.linspace(0, 17000, 9)[:, None, None], shape).copy()
    ps = rng.uniform(80000, 100000, shape[1:])
    ts = t[0].copy()
    qs = q[0].copy()
    zs = np.zeros(shape[1:])
    return q, p, t, z, ps, ts, qs, zs


def test_arithmetic_twins(columns):
    q, p, t, z, ps, ts, qs, zs = columns
    special = np.array([-0.028126, -0.0, 0.0, np.nextafter(0.0, 1.0),
                        1e-40, 0.01, np.nextafter(1.0, 0.0)])
    identical(host._specific_humidity_to_mixing_ratio(special, allow_wps_undershoot=True),
              dev._specific_humidity_to_mixing_ratio(special, allow_wps_undershoot=True))
    p[0, 0, :6] = [9999.0, 10000.0, np.nan, -0.0, np.inf, -np.inf]
    q[0, 0, :6] = [1e-5, np.nextafter(1e-5, np.inf), -0.0, np.nan, np.inf, -np.inf]
    identical(host._cap_stratospheric_qv(q, p), dev._cap_stratospheric_qv(q, p))
    for name in ("_pressure_at_u", "_pressure_at_v"):
        identical(getattr(host, name)(p), getattr(dev, name)(p))


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("values", [[np.nan, 0.0], [np.inf, -np.inf], [-0.04, 1.0], [1.0, 0.1]])
def test_specific_refusal_message(values, workers):
    q = np.asarray(values)[:, None, None]
    with pytest.raises(ValueError) as expected:
        host._specific_humidity_to_mixing_ratio(q, column_workers=workers)
    with pytest.raises(ValueError) as observed:
        dev._specific_humidity_to_mixing_ratio(q, column_workers=workers)
    assert str(observed.value) == str(expected.value)


@pytest.mark.parametrize("reverse", [False, True])
def test_integrate(columns, reverse):
    args = columns
    if reverse:
        args = tuple(a[::-1].copy() if a.ndim == 3 else a for a in args)
    identical(host._integrate_moisture(*args), dev._integrate_moisture(*args))


def test_fallback_and_floors(columns):
    q, p, _, _, ps, _, qs, _ = columns
    for fallback in (None, False, True):
        identical(host._wrf_flag_sh_surface_specific_humidity(qs, q, p, force_fallback=fallback),
                  dev._wrf_flag_sh_surface_specific_humidity(qs, q, p, force_fallback=fallback))
    identical(host._floor_flag_sh_surface_mixing_ratio(qs, ps),
              dev._floor_flag_sh_surface_mixing_ratio(qs, ps))
    identical(host._floor_sh_vertical_undershoot(q, p), dev._floor_sh_vertical_undershoot(q, p))


def test_rebalance_ladder_and_split(columns):
    q, _, _, _, ps, _, _, zs = columns
    coord = make_vertical_coord(q.shape[0], hybrid_opt=2)
    base = host._make_real_base(coord, zs, 5000.0, 290.0)
    mu = ps - 5000.0
    p = coord.c3h[:, None, None] * mu + coord.c4h[:, None, None] + 5000.0
    identical(p, dev.dry_pressure_ladder(mu, coord.c3h, coord.c4h, 5000.0))
    identical(host._rebalance_moist_pressure(p, q, mu, base, coord),
              dev._rebalance_moist_pressure(p, q, mu, base, coord))
    rng = np.random.default_rng(282)
    alpha = rng.uniform(0.5, 20.0, q.shape)
    # Exercise conversion and nextafter around the FP32 subnormal range.
    alpha[0, 0, :5] = [0.0, -0.0, 1e-45, 1e-40, np.finfo(np.float32).tiny]
    identical(host._fp32_geopotential_split(base, coord, mu, alpha),
              dev._fp32_geopotential_split(base, coord, mu, alpha))


def test_missing_surface_refusal(columns):
    args = list(columns)
    args[4] = np.full_like(args[4], 1.0)
    with pytest.raises(ValueError) as expected:
        host._integrate_moisture(*args)
    with pytest.raises(ValueError) as observed:
        dev._integrate_moisture(*args)
    assert str(expected.value) == str(observed.value)


def test_device_base_install(columns):
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.state import DomainState
    q, _, _, _, _, _, _, terrain = columns
    terrain = np.arange(terrain.size, dtype=np.float64).reshape(terrain.shape) * 27.5
    coord = make_vertical_coord(q.shape[0], hybrid_opt=2)
    base = host._make_real_base(coord, terrain, 5000.0, 290.0)
    cfg = RunConfig(nx=q.shape[2], ny=q.shape[1], nz=q.shape[0], dx=12000,
                    dy=12000, ztop=16000, dt=30, run_seconds=30, moist=True,
                    terrain_opt=1)
    reference = DomainState(cfg, array_module=np)
    reference.load_base(coord, base)
    observed = DomainState(cfg, array_module=cp)
    dev.load_base(observed, coord, base, dev.upload_base(base))
    for name, value in vars(reference).items():
        other = getattr(observed, name)
        if hasattr(value, "dtype") and hasattr(value, "shape"):
            identical(value, other)
        elif name in ("_dz_min", "cf1", "cf2", "cf3", "cfn", "cfn1", "p_top"):
            assert value == other


def test_stagger_adversarial_bit_words():
    rng = np.random.default_rng(28167)
    words = rng.integers(0, np.iinfo(np.uint64).max, (3, 7, 101), dtype=np.uint64)
    words[0, 0, :8] = [0x7ff8000000001234, 0x7ff8000000005678,
        0xfff8000000004321, 0x7ff0000000000001, 0, 0x8000000000000000,
        1, 0x8000000000000001]
    pressure = words.view(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        identical(host._pressure_at_u(pressure), dev._pressure_at_u(pressure))
        identical(host._pressure_at_v(pressure), dev._pressure_at_v(pressure))


def test_pipeline_widening_and_narrowing():
    import cupy as cp
    rng = np.random.default_rng(28168)
    words = rng.integers(0, np.iinfo(np.uint32).max, 8192, dtype=np.uint32)
    # Exact subnormal words, signed zero, and the normal boundary.
    words[:8] = [0, 0x80000000, 1, 2, 0x80000001, 0x007fffff, 0x00800000, 0x807fffff]
    values = words.view(np.float32)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        identical(values.astype(np.float64), dev.widen(cp.asarray(values)))
        wide = values.astype(np.float64)
        # Half-ulp ties around the smallest subnormal and smallest normal.
        wide[:6] = [2.0**-150, np.nextafter(2.0**-150, np.inf),
                    -2.0**-150, np.nextafter(-2.0**-150, -np.inf),
                    2.0**-126 - 2.0**-150, 2.0**-126 + 2.0**-150]
        identical(wide.astype(np.float32), dev.float32(cp.asarray(wide)))


def test_quantizer_fp32_primitives():
    rng = np.random.default_rng(28169)
    a = rng.integers(0, np.iinfo(np.uint32).max, 8192, dtype=np.uint32).view(np.float32)
    b = rng.integers(0, np.iinfo(np.uint32).max, 8192, dtype=np.uint32).view(np.float32)
    a[:8] = np.array([1, 0x80000001, 0x007fffff, 0x00800000,
                     0, 0x7f800000, 0xff800000, 0x7fc01234], np.uint32).view(np.float32)
    b[:8] = np.array([0, 0x3f800000, 1, 0x007fffff,
                     0, 0xff800000, 0xff800000, 0x7fc05678], np.uint32).view(np.float32)
    with np.errstate(invalid="ignore", over="ignore", under="ignore", divide="ignore"):
        expected = np.stack([a + b, a - b, a * b, a / b])
    identical(expected, dev._fp32_probe(a, b))


def test_root_export_retains_reference_base():
    import cupy as cp
    coord = make_vertical_coord(7, hybrid_opt=2)
    base = host._make_real_base(coord, np.zeros((2, 3)), 5000.0, 290.0)
    rng = np.random.default_rng(28170)
    values = rng.integers(0, np.iinfo(np.uint64).max, 256,
                          dtype=np.uint64).view(np.float64)
    names = ("surface_pressure", "surface_qv", "dry_mass", "dry_pressure",
             "total_pressure", "total_geopotential", "total_specific_volume",
             "integrated_moisture_pressure")
    result = host.RealInitResult(None, coord, dev.upload_base(base),
                                **{name: cp.asarray(values) for name in names})
    exported = dev.export_result(result, base=base)
    assert exported.base is base
    assert exported.coord is coord
    for name in names:
        actual = getattr(exported, name)
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == values.dtype and actual.tobytes() == values.tobytes()


@pytest.mark.parametrize("values", [
    [0.0, -0.0, 0.0, -0.0, 0.0, -0.0],
    [-0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, -0.0]])
def test_surface_floor_receipt_signed_zero(values):
    q = np.array(values).reshape(2, 3)
    p = np.full_like(q, 100000.0)
    identical(host._floor_flag_sh_surface_mixing_ratio(q, p),
              dev._floor_flag_sh_surface_mixing_ratio(q, p))
