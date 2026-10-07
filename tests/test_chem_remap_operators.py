"""Bit oracles for the two generic Rust remap operators."""
import ctypes
import os
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from gpuwm.ingest.cpu_backend import CpuPreprocessBackend, _library_names, CPU_BRIDGE_ENV, CPU_BRIDGE_ENV

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def backend():
    return CpuPreprocessBackend(os.environ.get(CPU_BRIDGE_ENV, ROOT / "tools/grib1_bridge/target/release" / _library_names()[0]))

def bits_equal(actual, expected):
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

def test_hybrid_full_pressure_single_round(backend):
    a = np.array([0, 2.125, 3000.3, 0], dtype=np.float64)
    b = np.array([0, 0.017, 0.4321, 1], dtype=np.float64)
    ps = np.array([[100000.1, 91234.5], [82341, 101323]], dtype=np.float32)
    levels = [1, 3]
    expected = np.empty((2, 2, 2), dtype=np.float32)
    for index, k in enumerate(levels):
        expected[index] = (0.5 * ((a[k-1] + b[k-1]*ps.astype(np.float64)) +
                                 (a[k] + b[k]*ps.astype(np.float64)))).astype(np.float32)
    bits_equal(backend.hybrid_full_pressure(a, b, levels, ps), expected)
    fn = backend._library.gpuwm_hybrid_full_pressure_f32
    lev = np.array(levels, dtype=np.int32)
    for workers in [0, 1, 4, 17]:
        out = np.empty_like(expected)
        assert fn(a.ctypes.data, b.ctypes.data, a.size, lev.ctypes.data, lev.size,
                  ps.ctypes.data, ps.size, out.ctypes.data, workers) == 0
        bits_equal(out, expected)

@pytest.mark.parametrize("levels,ps", [([0], [100000]), ([4], [100000]), ([2,1], [100000]),
                                      ([1,1], [100000]), ([1], [0]), ([1], [-1]),
                                      ([1], [np.nan]), ([1], [np.inf])])
def test_hybrid_refusals(backend, levels, ps):
    with pytest.raises(ValueError, match="gpuwm_hybrid_full_pressure_f32"):
        backend.hybrid_full_pressure([0,0,0,0], [0,.2,.5,1], levels, ps)

@pytest.mark.parametrize("humidity", [None, np.array([[0.0,.5],[.12345,.99]], dtype=np.float32)])
def test_weighted_combination_bits_and_workers(backend, humidity):
    fields = np.array([[[1.2345, 3.23456], [5.125, np.nan]],
                       [[2.3478, -1.875], [1.e-5, 9]],
                       [[-1.111, .00125], [1.0e10, 4]]], dtype=np.float32)
    weights = np.array([.123456789, -.8123456789, .999999991], dtype=np.float64)
    total = np.zeros(fields.shape[1:], dtype=np.float64)
    for field, weight in zip(fields, weights):
        total += weight * field.astype(np.float64)
    expected = 1.e6 * total
    if humidity is not None:
        expected /= 1.0 - humidity.astype(np.float64)
    expected = expected.astype(np.float32)
    bits_equal(backend.weighted_combination(fields, weights, scale=1.e6, humidity=humidity), expected)
    fn = backend._library.gpuwm_weighted_combination_f32
    for workers in [0, 1, 4, 17]:
        out = np.empty_like(expected)
        assert fn(fields.ctypes.data, weights.ctypes.data, weights.size, out.size, 1.e6,
                  None if humidity is None else humidity.ctypes.data, out.ctypes.data, workers) == 0
        bits_equal(out, expected)

@pytest.mark.parametrize("q", [-.001, 1, 1.01, np.nan, np.inf])
def test_invalid_humidity_refuses(backend, q):
    with pytest.raises(ValueError, match="humidity must be finite and in"):
        backend.weighted_combination([[1]], [1], humidity=[q])

def test_linear_interpolation_and_unit_conversion(backend):
    fields = np.array([[1.25, 2.75], [3.5, 4.25]], dtype=np.float32)
    alpha = .123456789
    expected = (1.e9 * ((1-alpha)*fields[0].astype(np.float64) + alpha*fields[1].astype(np.float64))).astype(np.float32)
    bits_equal(backend.weighted_combination(fields, [1-alpha,alpha], scale=1.e9), expected)


def test_field_order_is_preserved_through_cancellation(backend):
    fields = np.array([[2**60, 2**60], [1, -2**60], [-2**60, 1]], dtype=np.float32)
    expected = np.zeros(2, dtype=np.float64)
    for field in fields:
        expected += field.astype(np.float64)
    bits_equal(backend.weighted_combination(fields, [1, 1, 1]), expected.astype(np.float32))
    np.testing.assert_array_equal(expected, [0, 1])

@pytest.mark.parametrize("method,symbol,args", [
    ("hybrid_full_pressure", "gpuwm_hybrid_full_pressure_f32", ([0,0],[0,1],[1],[100000])),
    ("weighted_combination", "gpuwm_weighted_combination_f32", ([[1]], [1]))])
def test_old_library_names_missing_symbol(method, symbol, args):
    backend = CpuPreprocessBackend.__new__(CpuPreprocessBackend)
    backend._library = SimpleNamespace()
    with pytest.raises(RuntimeError, match=symbol):
        getattr(backend, method)(*args)

def test_shapes_are_checked_before_native_call(backend):
    with pytest.raises(ValueError, match="integer"):
        backend.hybrid_full_pressure([0,0], [0,1], [1.5], [100000])
    with pytest.raises(ValueError, match="matching"):
        backend.hybrid_full_pressure([0,0], [0], [1], [100000])
    with pytest.raises(ValueError, match="weights"):
        backend.weighted_combination([[1]], [1,2])
    with pytest.raises(ValueError, match="humidity"):
        backend.weighted_combination([[1]], [1], humidity=[[0]])
