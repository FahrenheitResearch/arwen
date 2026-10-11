"""TEMPO's device arithmetic gives TEMPO's Fortran words.

Two defects the TEMPO column oracle (tools/tempo_ref/oracle) found, each
fixed in gpuwm/core/kernels/tempo/tempo_common.cuh:

* Elementary functions.  The kernels called CUDA's exp, log, pow and log10,
  which differ from the C library TEMPO's Fortran links in the last place on
  some arguments; over 90 steps that grew to 1.9e-10 relative.  The device
  build now calls WOOF's own float64 exp/log/pow (kernels/glibc_flt64.cuh)
  and a log10 built on that log, through tempo_exp / tempo_log / tempo_pow /
  tempo_log10.
* NINT.  tnint was floor(x + 0.5), which turns 0.49999999999999994 into 1
  (x + 0.5 rounds to 1.0 first).  Graupel at the 50 kg m-3 density bound
  computes that ratio in its density index and took the wrong density
  class: a 1.1 K temperature error in one step.

Breakage these tests prevent: a TEMPO kernel edit that calls a bare libm
name on the device, or a NINT that is not Fortran's, moving the GPU off
TEMPO's Fortran.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math
import platform
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

KERNEL_DIR = Path(__file__).resolve().parents[1] / "gpuwm" / "core" / "kernels" / "tempo"
BARE = re.compile(r"(?<![\w.:])(exp|log|log10|pow)\s*\(")


def _code_lines(text):
    for n, line in enumerate(text.split("\n"), start=1):
        yield n, line.split("//", 1)[0]


def test_tempo_kernels_call_no_bare_libm_name():
    offenders = []
    for header in sorted(KERNEL_DIR.glob("*.cuh")):
        text = header.read_text(encoding="utf-8")
        if header.name == "tempo_common.cuh":
            # the four wrappers themselves call the host C library by name
            start = text.index("TEMPO_HD double tempo_exp(double x)")
            end = text.index("#ifndef TEMPO_NZ_MAX")
            text = text[:start] + "\n" * text[start:end].count("\n") + text[end:]
        for n, code in _code_lines(text):
            if BARE.search(code):
                offenders.append(f"{header.name}:{n}: {code.strip()}")
    assert not offenders, "\n".join(offenders)


def test_tnint_is_not_floor_of_x_plus_half():
    text = (KERNEL_DIR / "tempo_common.cuh").read_text(encoding="utf-8")
    body = text[text.index("TEMPO_HD int tnint(double x)"):]
    body = body[:body.index("\n}") + 2]
    assert "x + 0.5" not in body and "trunc(x)" in body


def _fortran_nint(x: float) -> int:
    f = Fraction(x)
    m = math.floor(abs(f) + Fraction(1, 2))
    return m if f >= 0 else -m


NINT_CASES = np.array([
    0.49999999999999994, -0.49999999999999994, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5,
    0.0, -0.0, 1.0, 0.4999999999999999, 1.4999999999999998, 2.4999999999999996,
    10.0 * 0.05, 10.0 * (1.0 / 20.0), 1e9 + 0.5, -(1e9 + 0.5), 2.0 ** 30 + 0.5,
    123456.49999999999, 3.4999999999999996, 7.0000000000000009, 0.1, -0.7,
], dtype=np.float64)


cp = None


def _cupy():
    global cp
    if cp is None:
        cp = pytest.importorskip("cupy")
    return cp


_KERNEL = r"""
#include "tempo_common.cuh"
extern "C" __global__ void tempo_math_probe(const double* x, const double* y, int n,
                                            double* e, double* l, double* p, double* l10,
                                            int* ni) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  e[i] = tempo_exp(x[i]);
  l[i] = tempo_log(x[i]);
  p[i] = tempo_pow(x[i], y[i]);
  l10[i] = tempo_log10(x[i]);
  ni[i] = (x[i] > -2.0e9 && x[i] < 2.0e9) ? tnint(x[i]) : 0;
}
"""


def _probe(x, y):
    from gpuwm.core.tempo_runtime import composed_source
    cpy = _cupy()
    mod = cpy.RawModule(code=composed_source(_KERNEL),
                        options=("-std=c++17", "-DTEMPO_NZ_MAX=8", "--fmad=false"))
    n = x.size
    xd, yd = cpy.asarray(x), cpy.asarray(y)
    outs = [cpy.empty(n, dtype=cpy.float64) for _ in range(4)]
    ni = cpy.empty(n, dtype=cpy.int32)
    mod.get_function("tempo_math_probe")(((n + 127) // 128,), (128,),
                                         (xd, yd, np.int32(n), *outs, ni))
    return [o.get() for o in outs] + [ni.get()]


@pytest.mark.gpu
def test_device_tnint_is_fortran_nint():
    got = _probe(NINT_CASES, np.ones_like(NINT_CASES))[4]
    want = np.array([_fortran_nint(float(v)) for v in NINT_CASES], dtype=np.int64)
    assert got.tolist() == want.tolist()


def _host_libm():
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        pytest.skip("the reference words are the x86-64 Linux C library's")
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        pytest.skip("no /proc/cpuinfo")
    if " fma" not in cpuinfo or " avx2" not in cpuinfo:
        pytest.skip("the C library takes its FMA code path only on FMA+AVX2 CPUs")
    lib = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
    for name in ("exp", "log", "log10"):
        getattr(lib, name).restype = ctypes.c_double
        getattr(lib, name).argtypes = (ctypes.c_double,)
    lib.pow.restype = ctypes.c_double
    lib.pow.argtypes = (ctypes.c_double, ctypes.c_double)
    return lib


def _arguments(n=1 << 17, seed=20261007):
    """TEMPO's working ranges (masses, numbers, temperatures, slopes and the
    10**x and x**b forms the kernels use) plus random positive doubles."""
    rng = np.random.default_rng(seed)
    q = n // 4
    x = np.concatenate([
        10.0 ** rng.uniform(-14.0, 12.0, q),              # mixing ratios .. numbers
        rng.uniform(-60.0, 60.0, q),                      # exp arguments
        rng.uniform(150.0, 330.0, q) / 273.15,            # temperature ratios
        np.abs(rng.standard_normal(n - 3 * q)) * 10.0 ** rng.integers(-300, 300, n - 3 * q),
    ])
    y = np.concatenate([
        rng.uniform(-3.0, 3.0, q),
        rng.uniform(0.1, 4.0, q),
        rng.uniform(1.0, 2.5, q),
        rng.uniform(-1.0, 1.0, n - 3 * q),
    ])
    return x.astype(np.float64), y.astype(np.float64)


@pytest.mark.gpu
def test_device_elementary_functions_are_the_host_c_library_words():
    lib = _host_libm()
    x, y = _arguments()
    e, lg, p, l10, _ = _probe(x, y)
    want_e = np.array([lib.exp(v) for v in x])
    want_l = np.array([lib.log(v) for v in x])
    want_p = np.array([lib.pow(a, b) for a, b in zip(x, y)])
    want_l10 = np.array([lib.log10(v) for v in x])
    pos = x > 0.0
    for name, got, want, sel in (("exp", e, want_e, np.ones_like(pos)),
                                 ("log", lg, want_l, pos), ("pow", p, want_p, pos),
                                 ("log10", l10, want_l10, pos)):
        g, w = got[sel], want[sel]
        bad = np.nonzero(g.view(np.int64) != w.view(np.int64))[0]
        assert bad.size == 0, (name, bad.size, [(float(x[sel][i]).hex(), float(g[i]).hex(),
                                                 float(w[i]).hex()) for i in bad[:5]])
