"""Bit identity prevents cold-start seeds or receipts changing on the card."""
from types import SimpleNamespace

import numpy as np
import pytest


def _cupy():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("no CUDA card")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA driver")
    return cp


@pytest.mark.parametrize("mp", [8, 28])
@pytest.mark.parametrize("device_alt", [False, True])
def test_cold_start_numbers_and_receipt_bits(mp, device_alt, monkeypatch):
    cp = _cupy()
    from gpuwm.ingest.real import _thompson_cold_start_moment_closure as host
    from gpuwm.ingest.closure_device import thompson_cold_start_moment_closure as device
    monkeypatch.setattr("gpuwm.ingest.closure_device.COLD_START_CHUNK_CELLS", 127)
    from gpuwm.core.thompson_entry import R1, R2

    rng = np.random.default_rng(571)
    shape = (4, 17, 19)
    size = np.prod(shape)
    masses = np.array([0, np.nextafter(np.float32(R1), np.float32(0)), R1,
                       np.nextafter(np.float32(R1), np.float32(np.inf)),
                       R2, 1e-8, 1e-4, 0.1], dtype=np.float32)
    temperatures = np.array([-1e10, 170, 179, 179.5, 200, 271.15, 272,
                             273.15, 273, 300, 1e10], dtype=np.float32)
    arrays = {}
    for q, n in (("qr", "nr"), ("qi", "ni"), ("qc", "nc")):
        arrays[q] = rng.choice(masses, size).reshape(shape)
        arrays[n] = rng.choice(np.array([0, -1, R2, 1e7], dtype=np.float32), size).reshape(shape)
        # Negative existing numbers without mass refuse, so keep that arm valid.
        arrays[n][arrays[q] == 0] = 0
    alt = (10.0 ** rng.uniform(-12, 12, size)).reshape(shape)
    temp = rng.choice(temperatures, size).reshape(shape)
    aero = rng.choice(np.array([0, -1, 99e6, 1e9, 1e10, 5e10], dtype=np.float32), size).reshape(shape)
    land = rng.integers(0, 2, shape[-2:]).astype(np.float32)
    h = SimpleNamespace(**{k: v.copy() for k, v in arrays.items()})
    d = SimpleNamespace(**{k: cp.asarray(v) for k, v in arrays.items()})
    cfg = SimpleNamespace(mp_physics=mp)
    expected = host(h, np, cfg, alt, temperature=temp, aerosol_number=aero, landmask=land)
    actual = device(d, cp, cfg, cp.asarray(alt) if device_alt else alt,
                    temperature=temp, aerosol_number=aero, landmask=land)
    for field in ("nr", "ni") + (("nc",) if mp == 28 else ()):
        np.testing.assert_array_equal(getattr(h, field).view(np.uint32),
                                      getattr(d, field).get().view(np.uint32))
    assert actual == expected
    import json
    assert json.dumps(actual) == json.dumps(expected)


@pytest.mark.parametrize("species", ["rain", "ice", "cloud"])
def test_cold_start_extreme_seed_bits(species):
    cp = _cupy()
    from gpuwm.ingest.closure_device import gathered_numbers
    from gpuwm.ingest.real import _cold_start_rain_ice_number, _cold_start_droplet_number
    from gpuwm.core.thompson_entry import np_thompson_entry_numbers, R1

    rng = np.random.default_rng(991)
    n = 50000
    mass = (10.0 ** rng.uniform(-42, 2, n)).astype(np.float32)
    number = np.zeros(n, dtype=np.float32)
    alt = (10.0 ** rng.uniform(-30, 30, n)).astype(np.float32)
    temp = rng.uniform(150, 330, n).astype(np.float32)
    aerosol = (10.0 ** rng.uniform(0, 12, n)).astype(np.float32)
    aerosol[::3] = 0
    land = rng.integers(0, 2, n).astype(np.float32)
    xland = np.where(land >= 0.5, 1.0, 2.0).astype(np.float32)
    with np.errstate(all="ignore"):
        if species == "cloud":
            seeded, _, _ = _cold_start_droplet_number(mass, number, alt, aerosol, land)
        else:
            seeded, _, _ = _cold_start_rain_ice_number(species, mass, number, alt, temp)
        active = mass > R1
        expected = seeded.copy()
        expected[active] = np_thompson_entry_numbers(
            species, mass[active], seeded[active], 1.0 / alt[active].astype(np.float64)).astype(np.float32)
        actual, _ = gathered_numbers(species, mass, number, alt, temp, aerosol, xland)
        actual = actual.get()
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(actual[finite].view(np.uint32), expected[finite].view(np.uint32))


@pytest.mark.parametrize("alt", [0.0, np.nan])
def test_inverse_density_refusal_text(alt):
    cp = _cupy()
    from gpuwm.ingest.real import _thompson_cold_start_moment_closure as host
    from gpuwm.ingest.closure_device import thompson_cold_start_moment_closure as device
    cfg = SimpleNamespace(mp_physics=8)
    with pytest.raises(ValueError) as h:
        host(None, np, cfg, np.array([alt], dtype=np.float32))
    with pytest.raises(ValueError) as d:
        device(None, cp, cfg, np.array([alt], dtype=np.float32))
    assert str(d.value) == str(h.value)


def test_subnormal_masks_and_aerosol_branch():
    cp = _cupy()
    import json
    from gpuwm.ingest.real import _thompson_cold_start_moment_closure as host
    from gpuwm.ingest.closure_device import thompson_cold_start_moment_closure as device
    mass = np.array([1e-40, 1e-40, 1e-4, 1e-4], dtype=np.float32)
    num = np.array([0, 1e-44, 0, 1e-44], dtype=np.float32)
    h = SimpleNamespace(**{k: v.copy() for k, v in dict(
        qc=mass, qr=mass, qi=mass, nc=num, nr=num, ni=num).items()})
    d = SimpleNamespace(**{k: cp.asarray(v) for k, v in vars(h).items()})
    cfg = SimpleNamespace(mp_physics=28)
    alt = np.ones(4, dtype=np.float64)
    temp = np.full(4, 270, dtype=np.float32)
    aero = np.full(4, 1e-40, dtype=np.float32)
    with np.errstate(all="ignore"):
        expected = host(h, np, cfg, alt, temperature=temp, aerosol_number=aero)
        actual = device(d, cp, cfg, cp.asarray(alt), temperature=temp, aerosol_number=cp.asarray(aero))
    for field in ("nr", "ni", "nc"):
        np.testing.assert_array_equal(getattr(h, field).view(np.uint32), getattr(d, field).get().view(np.uint32))
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


@pytest.mark.parametrize("species", ["rain", "ice"])
def test_nonfinite_seed_keeps_the_refusal(species):
    cp = _cupy()
    from gpuwm.ingest.real import _thompson_cold_start_moment_closure as host
    from gpuwm.ingest.closure_device import thompson_cold_start_moment_closure as device
    h = SimpleNamespace(**{field: np.zeros(1, dtype=np.float32)
                           for field in ("qr", "qi", "nr", "ni")})
    getattr(h, {"rain": "qr", "ice": "qi"}[species])[0] = 1e-4
    d = SimpleNamespace(**{k: cp.asarray(v) for k, v in vars(h).items()})
    cfg = SimpleNamespace(mp_physics=8)
    alt = np.ones(1, dtype=np.float32)
    # Infinite specific volume gives a finite zero entry density, but the
    # seed is non-finite and must still refuse instead of being floor-repaired.
    alt[0] = np.inf
    temp = np.full(1, 270, dtype=np.float32)
    with np.errstate(all="ignore"):
        with pytest.raises(ValueError) as expected:
            host(h, np, cfg, alt, temperature=temp)
        with pytest.raises(ValueError) as actual:
            device(d, cp, cfg, alt, temperature=temp)
    assert str(actual.value) == str(expected.value)


def test_gathered_temperature_matches_the_column_reference():
    cp = _cupy()
    from gpuwm.ingest.real import _temperature_from_potential_temperature
    from gpuwm.ingest.closure_device import make_temperature_provider
    rng = np.random.default_rng(612)
    theta = rng.uniform(250, 400, (9, 17, 19))
    pressure = rng.uniform(5000, 105000, theta.shape)
    idx = rng.integers(0, theta.size, 2017)
    full = _temperature_from_potential_temperature(theta, pressure, column_workers=4).astype(np.float32)
    before = cp.get_default_memory_pool().used_bytes()
    provider = make_temperature_provider(theta, pressure)
    assert cp.get_default_memory_pool().used_bytes() == before
    gathered = provider(cp.asarray(idx)).get()
    np.testing.assert_array_equal(gathered.view(np.uint32), full.ravel()[idx].view(np.uint32))


def test_boundary_receipt_does_not_read_back_seed_arrays(monkeypatch):
    cp = _cupy()
    from gpuwm.ingest import closure_device as device
    from gpuwm.ingest.real import _thompson_cold_start_moment_closure as host
    monkeypatch.setattr(device, "COLD_START_CHUNK_CELLS", 2)
    fields = dict(qr=np.array([1e-4, 1e-40, 0, 2e-4], dtype=np.float32),
                  qi=np.array([0, 1e-4, 2e-4, 1e-40], dtype=np.float32),
                  nr=np.zeros(4, dtype=np.float32), ni=np.zeros(4, dtype=np.float32))
    h = SimpleNamespace(**{k: v.copy() for k, v in fields.items()})
    d = SimpleNamespace(**{k: cp.asarray(v) for k, v in fields.items()})
    cfg = SimpleNamespace(mp_physics=8)
    alt = np.ones(4, dtype=np.float64)
    temperature = np.full(4, 270, dtype=np.float32)
    original = device._seed_receipt
    calls = []

    def count_only(name, count, *arrays, build_receipt=True):
        assert not build_receipt
        assert all(array is None for array in arrays)
        calls.append((name, count))
        return original(name, count, *arrays, build_receipt=False)

    monkeypatch.setattr(device, "_seed_receipt", count_only)
    expected = host(h, np, cfg, alt, temperature=temperature, receipt=False)
    actual = device.thompson_cold_start_moment_closure(
        d, cp, cfg, cp.asarray(alt), temperature=cp.asarray(temperature), receipt=False)
    assert calls
    assert actual == expected
    for field in ("nr", "ni"):
        np.testing.assert_array_equal(getattr(d, field).get().view(np.uint32),
                                      getattr(h, field).view(np.uint32))


@pytest.mark.parametrize("mask_shape", [(3, 1), (2, 1, 1)])
def test_cloud_landmask_broadcast_matches_host(mask_shape):
    cp = _cupy()
    from gpuwm.ingest.real import _thompson_cold_start_moment_closure as host
    from gpuwm.ingest.closure_device import thompson_cold_start_moment_closure as device
    shape = (2, 3, 4)
    fields = {q: np.zeros(shape, dtype=np.float32) for q in ("qc", "qr", "qi", "nc", "nr", "ni")}
    fields["qc"][:] = 1e-4
    h = SimpleNamespace(**{k: v.copy() for k, v in fields.items()})
    d = SimpleNamespace(**{k: cp.asarray(v) for k, v in fields.items()})
    cfg = SimpleNamespace(mp_physics=28)
    alt = np.ones(shape, dtype=np.float32)
    land = (np.arange(np.prod(mask_shape)) % 2).reshape(mask_shape).astype(np.float32)
    expected = host(h, np, cfg, alt, landmask=land)
    actual = device(d, cp, cfg, cp.asarray(alt), landmask=cp.asarray(land))
    assert actual == expected
    np.testing.assert_array_equal(d.nc.get().view(np.uint32), h.nc.view(np.uint32))


def test_unseeded_nan_number_preserves_host_refusal():
    cp = _cupy()
    from gpuwm.ingest.real import _thompson_cold_start_moment_closure as host
    from gpuwm.ingest.closure_device import thompson_cold_start_moment_closure as device
    fields = {q: np.zeros(2, dtype=np.float32) for q in ("qc", "qr", "qi", "nc", "nr", "ni")}
    fields["qc"][:] = 1e-4
    fields["nc"][0] = np.nan
    h = SimpleNamespace(**{k: v.copy() for k, v in fields.items()})
    d = SimpleNamespace(**{k: cp.asarray(v) for k, v in fields.items()})
    cfg = SimpleNamespace(mp_physics=28)
    alt, aerosol = np.ones(2, dtype=np.float32), np.full(2, 1e9, dtype=np.float32)
    with pytest.raises(ValueError) as expected:
        host(h, np, cfg, alt, aerosol_number=aerosol)
    with pytest.raises(ValueError) as actual:
        device(d, cp, cfg, cp.asarray(alt), aerosol_number=cp.asarray(aerosol))
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("missing_temperature", [False, True])
def test_refusal_order_and_counts_span_chunks(monkeypatch, missing_temperature):
    cp = _cupy()
    from gpuwm.ingest.real import _thompson_cold_start_moment_closure as host
    from gpuwm.ingest.closure_device import thompson_cold_start_moment_closure as device
    monkeypatch.setattr("gpuwm.ingest.closure_device.COLD_START_CHUNK_CELLS", 2)
    h = SimpleNamespace(qr=np.array([0, 0, 0, 1e-4], dtype=np.float32),
                        qi=np.array([np.inf, 0, 0, 0], dtype=np.float32),
                        nr=np.array([-1, 0, 0, 0], dtype=np.float32),
                        ni=np.zeros(4, dtype=np.float32))
    if missing_temperature:
        h.qr[:] = 1e-4
        h.nr[:] = 0
    d = SimpleNamespace(**{k: cp.asarray(v) for k, v in vars(h).items()})
    cfg = SimpleNamespace(mp_physics=8)
    alt = np.ones(4, dtype=np.float32)
    temp = None if missing_temperature else np.full(4, 270, dtype=np.float32)
    with np.errstate(all="ignore"):
        with pytest.raises(ValueError) as expected:
            host(h, np, cfg, alt, temperature=temp)
        with pytest.raises(ValueError) as actual:
            device(d, cp, cfg, alt, temperature=temp)
    assert str(actual.value) == str(expected.value)


def test_nan_aerosol_is_refused_by_name_on_host_and_device():
    """The device says the host's sentence: the field, the cells, the row.

    Both used to raise NumPy's ``IndexError: index 9223372036854775807 is
    out of bounds for axis 0 with size 15``, the device by copying it.
    """
    cp = _cupy()
    from gpuwm.ingest.real import _thompson_cold_start_moment_closure as host
    from gpuwm.ingest.closure_device import thompson_cold_start_moment_closure as device
    h = SimpleNamespace(**{k: np.zeros(1, dtype=np.float32)
                           for k in ("qc", "qr", "qi", "nc", "nr", "ni")})
    h.qc[0] = 1e-4
    d = SimpleNamespace(**{k: cp.asarray(v) for k, v in vars(h).items()})
    cfg = SimpleNamespace(mp_physics=28)
    alt = np.ones(1, dtype=np.float32)
    aerosol = np.full(1, np.nan, dtype=np.float32)
    with np.errstate(all="ignore"):
        with pytest.raises(ValueError, match="1 cloudy cell.*QNWFA") as expected:
            host(h, np, cfg, alt, aerosol_number=aerosol)
        with pytest.raises(ValueError) as actual:
            device(d, cp, cfg, alt, aerosol_number=aerosol)
    assert str(actual.value) == str(expected.value)
    assert "out of bounds" not in str(actual.value)


def test_a_nan_droplet_number_at_the_entry_block_is_refused_by_name(monkeypatch):
    """Infinite inverse density under cloud water: zero density, NaN number.

    The host mirror failed with ``IndexError: index -9223372036854775808
    is out of bounds for axis 0 with size 16``; the device reached the same
    mirror.  Two offending cells sit in different two-cell chunks, so the
    device must count the whole domain, as the host does, and publish
    nothing.
    """
    cp = _cupy()
    from gpuwm.ingest.real import _thompson_cold_start_moment_closure as host
    from gpuwm.ingest.closure_device import thompson_cold_start_moment_closure as device
    monkeypatch.setattr("gpuwm.ingest.closure_device.COLD_START_CHUNK_CELLS", 2)
    h = SimpleNamespace(**{k: np.zeros(6, dtype=np.float32)
                           for k in ("qc", "qr", "qi", "nc", "nr", "ni")})
    h.qc[:] = 1e-4
    d = SimpleNamespace(**{k: cp.asarray(v) for k, v in vars(h).items()})
    cfg = SimpleNamespace(mp_physics=28)
    alt = np.ones(6, dtype=np.float32)
    alt[[1, 4]] = np.inf
    aerosol = np.full(6, 1.0e8, dtype=np.float32)
    land = np.ones(6, dtype=np.float32)
    with np.errstate(all="ignore"):
        with pytest.raises(ValueError, match="2 cloudy cell.*entry block") as expected:
            host(h, np, cfg, alt, aerosol_number=aerosol, landmask=land)
        with pytest.raises(ValueError) as actual:
            device(d, cp, cfg, alt, aerosol_number=aerosol, landmask=land,
                   chunk_cells=2)
    assert str(actual.value) == str(expected.value)
    assert "out of bounds" not in str(actual.value)
    assert not cp.asnumpy(d.nc).any() and not h.nc.any()


def test_all_seeded_cloud_scratch_fits_the_preparation_allowance():
    cp = _cupy()
    from gpuwm.ingest.closure_device import (
        thompson_cold_start_moment_closure, COLD_START_CHUNK_CELLS,
        make_temperature_provider)
    pool = cp.cuda.MemoryPool()
    high = [0]

    def allocate(size):
        pointer = pool.malloc(size)
        high[0] = max(high[0], pool.used_bytes())
        return pointer

    size = 65536
    with cp.cuda.using_allocator(allocate):
        state = SimpleNamespace(**{q: cp.full(size, 1e-4, dtype=cp.float32)
                                   for q in ("qc", "qr", "qi")},
                                **{n: cp.zeros(size, dtype=cp.float32)
                                   for n in ("nc", "nr", "ni")})
        alt = cp.ones(size, dtype=cp.float64)
        temp = make_temperature_provider(np.full(size, 270.0), np.full(size, 100000.0))
        aerosol = cp.full(size, 1e9, dtype=cp.float32)
        land = cp.ones(size, dtype=cp.float32)
        baseline = pool.used_bytes()
        high[0] = baseline
        thompson_cold_start_moment_closure(
            state, cp, SimpleNamespace(mp_physics=28), alt,
            temperature=temp, aerosol_number=aerosol, landmask=land)
    measured = high[0] - baseline
    allowance = 28 * size + 112 * min(size, COLD_START_CHUNK_CELLS)
    assert measured <= allowance, (measured, allowance)
