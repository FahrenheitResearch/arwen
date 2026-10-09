"""mp=28 evaluates WRF's EXP/LOG/LOG10/** with WOOF's own libm words.

The breakage this file prevents.  WRF v4.6.1's module_mp_thompson.F is
compiled by gfortran, which lowers a REAL(4) EXP/ALOG/ALOG10/** to the C
library's expf/logf/log10f/powf and a DOUBLE PRECISION one to
exp/log/log10/pow; WRF's phy_prep forms the Exner function mp_gt_driver
receives as a REAL(4) power, ``(p/p1000mb)**rcp``.  The mp=28 units called
CUDA's expf/logf/powf/log10f/exp/log/pow/log10 and the adapter formed the
Exner function with CuPy's power (CUDA's powf).  Those are different
functions: the 0 ULP column oracle (tools/thompson_aerosol_column_oracle,
153 columns) measured the Exner function 1 ULP off WRF's at 1131 of 7497
cells and no column bit-identical to WRF.  WOOF's own words
(gpuwm/core/kernels/thompson_aerosol_libm.cuh) return the oracle host's
words; the adapter's Exner kernel uses WOOF's powf.

CPU tests read the sources.  The GPU tests compare the device's words with
the C library of the host they run on, which is WRF's reference only on the
oracle host's library (glibc 2.39, x86-64 with FMA, the path the library
selects there); on any other C library they skip and say so, because a
different library is a different reference, not a defect here.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import pathlib
import platform
import re

import numpy as np
import pytest

from conftest import requires_gpu

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_KERNELS = _ROOT / "gpuwm" / "core" / "kernels"
_UNITS = ("thompson_aerosol_cold", "thompson_aerosol_warm",
          "thompson_aerosol_sat", "thompson_aerosol_sed",
          "thompson_aerosol_state", "thompson_aerosol_probe")
_LIBM_HEADERS = ("glibc_flt32.cuh", "glibc_flt64.cuh",
                 "thompson_aerosol_libm.cuh", "thompson_aerosol_common.cuh")
#: A bare call of a C-library or CUDA transcendental that WRF's gfortran
#: build lowers to the C library.
_BARE_CALL = re.compile(
    r"(?<![A-Za-z0-9_.])(expf|logf|powf|log10f|exp|log|pow|log10|exp2f?"
    r"|exp10f?|__expf|__logf|__powf|__log10f|__exp10f"
    r"|hypotf?|log1pf?|atan2f?|expm1f?|cbrtf?)\s*\(")


def _code_only(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def test_every_mp28_unit_prepends_the_libm_words():
    from gpuwm.core.kernels import EXTRA_HEADERS
    for unit in _UNITS:
        assert EXTRA_HEADERS[unit] == _LIBM_HEADERS, unit


@pytest.mark.parametrize("name", [f"{u}.cu" for u in _UNITS]
                         + ["thompson_aerosol_common.cuh",
                            "thompson_aerosol_libm.cuh"])
def test_no_mp28_source_calls_a_cuda_transcendental(name):
    code = _code_only((_KERNELS / name).read_text(encoding="utf-8"))
    calls = sorted({m.group(1) for m in _BARE_CALL.finditer(code)})
    assert calls == [], f"{name} calls {calls}; use thompson_aa_* words"


def test_the_adapter_forms_the_exner_function_with_the_libm_word():
    src = (_ROOT / "gpuwm" / "core" / "microphysics_aerosol.py").read_text(
        encoding="utf-8")
    assert "launch_aerosol_exner(state.p, pii)" in src
    assert "cp.power(state.p" not in src


# ---------------------------------------------------------------------------
# GPU: the device's words against the host's C library.
# ---------------------------------------------------------------------------

def _oracle_host_libm():
    """The C library, if this host is the oracle's kind; else a skip reason."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        return None, "not x86-64 Linux, so not the oracle host's C library"
    libc, version = platform.libc_ver()
    if libc != "glibc" or version != "2.39":
        return None, f"C library {libc} {version}, the oracle host's is glibc 2.39"
    try:
        flags = pathlib.Path("/proc/cpuinfo").read_text()
    except OSError:
        return None, "cannot read /proc/cpuinfo to confirm the FMA path"
    if " fma" not in flags:
        return None, "CPU without FMA: glibc takes its non-FMA path here"
    return ctypes.CDLL(ctypes.util.find_library("m")), None


def _host(libm, name, x, y=None):
    fn = getattr(libm, name)
    single = name.endswith("f")
    ctype = ctypes.c_float if single else ctypes.c_double
    fn.restype = ctype
    fn.argtypes = ((ctype, ctype) if name in ("powf", "pow", "hypot", "atan2")
                   else (ctype,))
    out = np.empty(x.shape, np.float32 if single else np.float64)
    for i in range(x.size):
        args = (float(x.flat[i]),) if y is None else (float(x.flat[i]),
                                                      float(y.flat[i]))
        out.flat[i] = fn(*args)
    return out


def _samples(name, n, rng):
    if name == "expf":
        return rng.uniform(-104.0, 89.0, n).astype(np.float32), None
    if name in ("logf", "log10f"):
        # Default arithmetic compiles with flush-to-zero, which flushes a
        # subnormal ARGUMENT before the word sees it; the strict build
        # (GPUWM_WRF_EXACT=1) does not, and there the subnormal band is
        # sampled too.  The host exhaustive check (libm_check.cpp) covers
        # every input.  No mp=28 operand of a logarithm is subnormal: every
        # mass and number reaching one is floored far above 1e-38.
        from gpuwm import wrf_exact
        low = 1 if wrf_exact.ENABLED else 0x00800000
        bits = rng.integers(low, 0x7F800000, n, dtype=np.int64)
        return bits.astype(np.uint32).view(np.float32), None
    if name == "powf":
        x = rng.uniform(1.0e-6, 1.0e6, n).astype(np.float32)
        return x, rng.uniform(-12.0, 12.0, n).astype(np.float32)
    if name in ("hypot", "atan2"):
        # Seven decades of each argument, both signs: the reflectivity's
        # refractive indices, their squares and quotients sit inside.
        def leg():
            return rng.choice([-1.0, 1.0], n) * np.power(
                10.0, rng.uniform(-3.5, 3.5, n))
        return leg(), leg()
    if name == "log1p":
        tiny = n - n // 2
        return np.concatenate([
            rng.uniform(-0.5, 1.0, n // 2),
            rng.choice([-1.0, 1.0], tiny)
            * np.power(10.0, rng.uniform(-16, 0, tiny))]), None
    if name == "exp":
        return rng.uniform(-745.0, 709.0, n), None
    if name in ("log", "log10"):
        return np.ldexp(rng.uniform(1.0, 2.0, n), rng.integers(-1074, 1023, n)), None
    x = np.ldexp(rng.uniform(1.0, 2.0, n), rng.integers(-30, 30, n))
    return x, rng.uniform(-16.0, 16.0, n)


@requires_gpu
@pytest.mark.parametrize("name", ["expf", "logf", "log10f", "powf",
                                  "exp", "log", "log10", "pow",
                                  "hypot", "log1p"])
def test_device_libm_words_equal_the_oracle_host_library(name):
    libm, why = _oracle_host_libm()
    if libm is None:
        pytest.skip(why)
    import cupy as cp
    from gpuwm.core.thompson_aerosol_launch import probe_libm
    rng = np.random.default_rng(20261007)
    x, y = _samples(name, 40000, rng)
    want = _host(libm, name, x, y)
    got = cp.asnumpy(probe_libm(name, cp.asarray(x),
                                None if y is None else cp.asarray(y)))
    width = np.int32 if got.dtype == np.float32 else np.int64
    bad = np.flatnonzero(got.view(width) != want.view(width))
    assert bad.size == 0, (name, x.flat[bad[:3]], got.flat[bad[:3]],
                           want.flat[bad[:3]])


@requires_gpu
def test_the_exner_kernel_returns_wrfs_powf_word_and_cupy_power_does_not():
    libm, why = _oracle_host_libm()
    if libm is None:
        pytest.skip(why)
    import cupy as cp
    from gpuwm.core.thompson_aerosol_state import launch_aerosol_exner
    rng = np.random.default_rng(7)
    p = rng.uniform(5000.0, 105000.0, 20000).astype(np.float32)
    base = (p / np.float32(1.0e5)).astype(np.float32)
    rcp = np.float32(np.float32(287.0) / np.float32(1004.5))
    want = _host(libm, "powf", base, np.full_like(base, rcp))
    out = cp.empty(p.shape, cp.float32)
    launch_aerosol_exner(cp.asarray(p), out)
    got = cp.asnumpy(out)
    assert np.array_equal(got.view(np.int32), want.view(np.int32))
    # The negative control: the CuPy line this kernel replaced is a
    # different function, so the gate above would catch its return.
    cupy_word = cp.asnumpy(cp.power(cp.asarray(p) / np.float32(1.0e5), rcp))
    assert not np.array_equal(cupy_word.view(np.int32), want.view(np.int32))


@requires_gpu
def test_device_atan2_is_within_one_and_a_half_ulp_and_mostly_the_hosts():
    """WOOF's atan2 is its own routine, not the host's: it is graded on its
    error, not bit for bit.  It feeds only the argument of the complex
    logarithm inside the melting-snow backscatter, a binary64 chain that
    calc_refl10cm rounds to REAL(4) (tests/test_thompson_aerosol_sedim_refl.py
    holds that echo to WRF's word on every melting cell of its fixture)."""
    libm, why = _oracle_host_libm()
    if libm is None:
        pytest.skip(why)
    import cupy as cp
    from gpuwm.core.thompson_aerosol_launch import probe_libm
    rng = np.random.default_rng(20261007)
    y, x = _samples("atan2", 40000, rng)
    want = _host(libm, "atan2", y, x)
    got = cp.asnumpy(probe_libm("atan2", cp.asarray(y), cp.asarray(x)))
    exact = np.arctan2(y.astype(np.longdouble), x.astype(np.longdouble))
    ulp = np.spacing(np.abs(want))
    err = np.abs((got.astype(np.longdouble) - exact) / ulp)
    assert float(err.max()) <= 1.5, float(err.max())
    agree = np.mean(got.view(np.int64) == want.view(np.int64))
    assert agree >= 0.75, agree
