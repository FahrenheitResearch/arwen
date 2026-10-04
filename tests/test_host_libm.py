"""gpuwm.core.host_libm is the C library, element by element, with NumPy's
domain answers.

Its callers (the static-fields projection transcription) need the bits the
native Rust crate computes through the same C library, on every machine;
NumPy 2.5's AVX-512 loops for these functions round differently.  These
tests hold each function to :mod:`math` on ordinary arguments, to NumPy's
NaN and infinity answers outside the domain, and to the input's shape.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gpuwm.core import host_libm

_RNG = np.random.default_rng(29)
_ANGLES = _RNG.uniform(-1.5, 1.5, 2000)
_POSITIVE = _RNG.uniform(1e-6, 1e4, 2000)
_UNIT = _RNG.uniform(-1.0, 1.0, 2000)

_CASES = (
    (host_libm.exp, math.exp, _RNG.uniform(-700.0, 700.0, 2000)),
    (host_libm.log, math.log, _POSITIVE),
    (host_libm.log10, math.log10, _POSITIVE),
    (host_libm.tan, math.tan, _ANGLES),
    (host_libm.arctan, math.atan, _RNG.uniform(-1e3, 1e3, 2000)),
    (host_libm.arcsin, math.asin, _UNIT),
    (host_libm.arccos, math.acos, _UNIT),
)


@pytest.mark.parametrize("ours,libm,values", _CASES,
                         ids=[c[0].__name__ for c in _CASES])
def test_each_function_is_the_c_library_element_by_element(ours, libm, values):
    got = ours(values.reshape(40, 50))
    assert got.shape == (40, 50) and got.dtype == np.float64
    want = np.array([libm(float(v)) for v in values]).reshape(40, 50)
    assert np.array_equal(got.view(np.uint64), want.view(np.uint64))


def test_power_is_pow_element_for_element_and_broadcasts():
    base = _POSITIVE.reshape(40, 50)
    got = host_libm.power(base, 0.7159)
    want = np.array([math.pow(float(v), 0.7159) for v in _POSITIVE])
    assert np.array_equal(got.reshape(-1).view(np.uint64),
                          want.view(np.uint64))
    exponents = _RNG.uniform(-3.0, 3.0, 50)
    got = host_libm.power(base, exponents)
    assert got.shape == (40, 50)
    assert got[3, 7] == math.pow(float(base[3, 7]), float(exponents[7]))


def test_arctan2_is_atan2_element_for_element_and_broadcasts():
    """The Lambert inverse's longitude takes ``arctan2``, whose NumPy 2.5
    AVX-512 float64 loop differs from the C library's atan2 on about 8% of
    random argument pairs, so it is held to :func:`math.atan2` here, on
    ordinary pairs, on Lambert-inverse-shaped offsets and on the signed
    zeros, infinities and NaNs."""
    y = _RNG.uniform(-5.0, 5.0, 2000).reshape(40, 50)
    x = _RNG.uniform(-3.0, 3.0, 2000).reshape(40, 50)
    got = host_libm.arctan2(y, x)
    assert got.shape == (40, 50) and got.dtype == np.float64
    want = np.array([math.atan2(float(a), float(b))
                     for a, b in zip(y.reshape(-1), x.reshape(-1))])
    assert np.array_equal(got.reshape(-1).view(np.uint64),
                          want.view(np.uint64))
    offsets = _RNG.uniform(-600.0, 600.0, 2000)
    got = host_libm.arctan2(offsets, 2400.0)
    want = np.array([math.atan2(float(a), 2400.0) for a in offsets])
    assert np.array_equal(got.view(np.uint64), want.view(np.uint64))
    specials = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan])
    a, b = np.meshgrid(specials, specials)
    ours = host_libm.arctan2(a, b).reshape(-1)
    theirs = np.array([math.atan2(float(u), float(v))
                       for u, v in zip(a.reshape(-1), b.reshape(-1))])
    number = ~np.isnan(theirs)
    assert np.array_equal(np.isnan(ours), ~number)
    assert np.array_equal(ours[number].view(np.uint64),
                          theirs[number].view(np.uint64))
    assert host_libm.arctan2(1.0, 2.0).shape == ()


def test_a_scalar_argument_gives_a_zero_dimensional_answer():
    assert host_libm.tan(0.3).shape == ()
    assert float(host_libm.tan(0.3)) == math.tan(0.3)
    assert host_libm.power(2.0, 0.5).shape == ()


def test_outside_the_domain_the_answers_are_numpys():
    with np.errstate(all="ignore"):
        cases = (
            (host_libm.log, np.log, [0.0, -0.0, -1.0, np.inf, np.nan]),
            (host_libm.log10, np.log10, [0.0, -1.0, np.inf, np.nan]),
            (host_libm.exp, np.exp, [1000.0, -1000.0, np.inf, -np.inf,
                                     np.nan]),
            (host_libm.tan, np.tan, [np.inf, -np.inf, np.nan]),
            (host_libm.arcsin, np.arcsin, [1.5, -1.5, np.inf, np.nan]),
            (host_libm.arccos, np.arccos, [1.5, -1.5, -np.inf, np.nan]),
            (host_libm.arctan, np.arctan, [np.inf, -np.inf, np.nan]),
        )
        for ours, numpy_function, values in cases:
            values = np.array(values)
            np.testing.assert_array_equal(ours(values), numpy_function(values))
        np.testing.assert_array_equal(
            host_libm.power(np.array([0.0, -2.0, 1e300]),
                            np.array([-1.0, 0.5, 2.5])),
            np.power(np.array([0.0, -2.0, 1e300]), np.array([-1.0, 0.5, 2.5])))


@pytest.mark.parametrize("workers", [1, 8, 32])
def test_native_host_math_preserves_scalar_bits_and_special_values(workers):
    from gpuwm.core import portable_math as pm
    library = pm._load()
    if not hasattr(library, "gpuwm_host_binary_f64"):
        pytest.skip("CPU bridge predates the separate host-libm ABI")
    rng = np.random.default_rng(719)
    values = rng.uniform(-700, 700, 65_539)
    specials = np.array([
        0, 0x8000000000000000, 1, 0x8000000000000001,
        0x7ff0000000000000, 0xfff0000000000000,
        0x7ff8000000000042, 0xfff8000000000019,
        0x7ff0000000000031, 0xfff0000000000027,
    ], dtype=np.uint64).view(np.float64)
    values[:len(specials)] = specials
    references = (
        (host_libm.exp, host_libm._exp), (host_libm.log, host_libm._LOG),
        (host_libm.log10, host_libm._LOG10), (host_libm.tan, host_libm._tan),
        (host_libm.arctan, math.atan), (host_libm.arcsin, host_libm._ASIN),
        (host_libm.arccos, host_libm._ACOS),
    )
    with pm.worker_limit(workers):
        for function, reference in references:
            expected = np.array([reference(float(value)) for value in values])
            assert function(values).tobytes() == expected.tobytes(), function.__name__
        bases = rng.integers(0, np.iinfo(np.uint64).max, 65_539, dtype=np.uint64).view(np.float64)
        exponents = rng.uniform(-4, 4, bases.size)
        bases[:len(specials)] = specials
        for power in (-3.0, -0.5, 0.0, 0.5, 2.0, 3.0, np.inf, -np.inf, np.nan):
            exponents[:len(specials)] = power
            expected = np.array([host_libm._pow(float(x), float(y))
                                 for x, y in zip(bases, exponents)])
            assert host_libm.power(bases, exponents).tobytes() == expected.tobytes()
        for y, x in ((values, 0.0), (values, values[::-1])):
            a, b = np.broadcast_arrays(y, x)
            expected = np.array([math.atan2(float(u), float(v))
                                 for u, v in zip(a.flat, b.flat)])
            assert host_libm.arctan2(y, x).tobytes() == expected.tobytes()


def test_native_host_math_broadcast_blocks_and_older_bridge_fallback(monkeypatch):
    from gpuwm.core import portable_math as pm
    base = np.linspace(0.01, 10, 513 * 257).reshape(513, 257)[:, ::2]
    exponents = np.linspace(-0.75, 1.25, base.shape[1])
    expected = np.array([host_libm._pow(float(x), float(y)) for x, y in
                         zip(base.flat, np.broadcast_to(exponents, base.shape).flat)]).reshape(base.shape)
    assert host_libm.power(base, exponents).tobytes() == expected.tobytes()
    assert host_libm.power(np.empty((0, 3)), 2).shape == (0, 3)
    unaligned = np.ndarray((1003,), dtype=np.float64,
                           buffer=bytearray(1003 * 8 + 1), offset=1)
    unaligned[:] = np.linspace(0.01, 1.0, 1003)
    assert not unaligned.flags.aligned
    assert host_libm.exp(unaligned).tobytes() == np.array([math.exp(x) for x in unaligned]).tobytes()
    assert host_libm.power(unaligned, 0.7159).tobytes() == np.array([math.pow(x, 0.7159) for x in unaligned]).tobytes()
    monkeypatch.setattr(pm, "_load", lambda: None)
    assert host_libm.power(base, exponents).tobytes() == expected.tobytes()
    assert host_libm.exp(np.array([1.0, 2.0])).tobytes() == np.array([math.exp(1), math.exp(2)]).tobytes()


def test_geometry_callers_keep_reference_without_native_assets(monkeypatch):
    from gpuwm.core import portable_math as pm
    def absent():
        raise FileNotFoundError("CPU bridge is not installed")
    monkeypatch.setattr(pm, "_load", absent)
    values = np.array([0.1, 1.0, 2.0])
    assert host_libm.log(values).tobytes() == np.array([math.log(x) for x in values]).tobytes()
    assert host_libm.power(values, 0.7159).tobytes() == np.array([math.pow(x, 0.7159) for x in values]).tobytes()
    def broken():
        raise OSError("native ABI cannot be loaded")
    monkeypatch.setattr(pm, "_load", broken)
    with pytest.raises(OSError, match="native ABI"):
        host_libm.power(values, 0.7159)
