"""Bitwise grading against the loaded host library, including NaN payloads."""
import numpy as np
import pytest


def test_portable_libm64_generated_arithmetic_source():
    import importlib.util
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("plm_header_generator", root / "tools/portable_libm64_header.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    header = (root / "gpuwm/core/kernels/portable_libm64.cuh").read_text(encoding="utf-8")
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", header, flags=re.S)
    assert not re.search(r"\b(?:exp|log|log1p|pow|log1pf|fma)\s*\(", code)
    assert "/" not in code
    assert "\u2014" not in header
    # Independent AST examples guard association and prevent contraction.
    assert generator.expand("D(a * b * c / d)") == "__ddiv_rn(__dmul_rn(__dmul_rn(a, b), c), d)"
    assert generator.expand("F(a - (b + c * d))") == "__fsub_rn(a, __fadd_rn(b, __fmul_rn(c, d)))"


def grading_arguments():
    rng = np.random.default_rng(641)
    words = rng.bit_generator.random_raw(500_000)
    finite = np.ldexp(rng.uniform(0.5, 1.0, 500_000), rng.integers(-1073, 1025, 500_000))
    finite[rng.integers(0, 2, finite.size).astype(bool)] *= -1
    special = np.array([0, 1, 2, 0x000fffffffffffff, 0x0010000000000000,
        0x3ff0000000000000, 0x7fefffffffffffff, 0x7ff0000000000000,
        0x7ff0000000000001, 0x7ff8000000001234], dtype=np.uint64)
    special = np.concatenate([special, special | np.uint64(1 << 63)])
    near = []
    for value in (1.0, -1.0, 709.782712893383973096, -708.39641853226410622,
                  -745.13321910194110842, 0.5, -0.5, 2.0, -2.0):
        word = np.array(value).view(np.uint64).item()
        near.extend(range(word - 8, word + 9))
    special = np.concatenate([special, np.array(near, dtype=np.uint64)]).view(np.float64)
    exponents = np.concatenate([special, np.arange(-64.0, 64.5, 0.5), np.array(
        [1023.0, 1023.5, 1024.0, 1024.5, -1074.0, -1074.5, -1075.0, -1075.5,
         2**31, 2**53, 2**64], dtype=np.float64)])
    sx = np.repeat(special, exponents.size)
    sy = np.tile(exponents, special.size)
    x = np.concatenate([words.view(np.float64), finite, sx])
    y = np.concatenate([rng.bit_generator.random_raw(500_000).view(np.float64), rng.uniform(-2048.0, 2048.0, 500_000), sy])
    return x, y


def test_portable_libm64_matches_host_gpu():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA driver")
    from gpuwm.core import portable_math as pm
    if pm.implementation() != pm.IMPLEMENTATION:
        pytest.skip("Rust portable host library unavailable")
    from gpuwm.core.kernels import load_module
    from gpuwm.core.noahmp_libm import log1pf_array
    rng = np.random.default_rng(641)
    x, y = grading_arguments()
    module = load_module("portable_libm64_grade")
    kernel = module.get_function("plm_grade64")
    dx, dy, out = cp.asarray(x), cp.asarray(y), cp.empty(x.size, dtype=cp.float64)
    for op, fn in enumerate((pm.exp, pm.log, pm.log1p, pm.power)):
        kernel(((x.size + 255) // 256,), (256,), (dx, dy, out, np.uint64(x.size), np.int32(op)))
        expected = fn(x, y) if op == 3 else fn(x)
        np.testing.assert_array_equal(out.get().view(np.uint64), expected.view(np.uint64))
    special32 = np.array([0, 1, 2, 0x007fffff, 0x00800000, 0x31000000,
        0x3ed413d7, 0x3f800000, 0x5a000000, 0x7f7fffff, 0x7f800000,
        0x7f800001, 0x7fc01234], dtype=np.uint32)
    x32 = np.concatenate([rng.integers(0, 1 << 32, 1_000_000, dtype=np.uint32),
        special32, special32 | np.uint32(1 << 31)]).view(np.float32)
    d32, o32 = cp.asarray(x32), cp.empty(x32.size, dtype=cp.float32)
    module.get_function("plm_grade32")(((x32.size + 255) // 256,), (256,), (d32, o32, np.uint64(x32.size)))
    with np.errstate(all="ignore"):
        expected = log1pf_array(x32)
    np.testing.assert_array_equal(o32.get().view(np.uint32), expected.view(np.uint32))
