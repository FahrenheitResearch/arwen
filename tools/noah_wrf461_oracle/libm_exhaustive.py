#!/usr/bin/env python3
"""The Noah column's float32 libm words against the host C library, exhaustively.

    python tools/noah_wrf461_oracle/libm_exhaustive.py OUT.json

Needs a CUDA device and a Linux host with gcc and the C library the WRF
oracle linked (glibc 2.39 for the committed fixtures).  It compiles the
exact source ``kernels.load_module("noah")`` hands NVRTC, plus one probe
kernel, and compares, bit for bit:

* ``noah_log10f`` against the host ``log10f`` on every positive finite
  float32 (2,139,095,039 inputs) and on +-0, +-inf, NaN and the negatives'
  sign class;
* ``gfk_log`` against ``logf`` on the same inputs;
* ``gfk_exp`` against ``expf`` on every float32 bit pattern (2^32 inputs);
* ``gfk_pow`` against ``powf`` on the exponents the Noah column uses (the
  REDPRM/GENPARM table values, 2, 3, 4, 1.5, CAPA and the -1/BX family)
  over a dense base sweep, and on 2^26 pseudo-random (base, exponent) pairs;
* ``noah_powi`` against libgcc's ``__powisf2`` for exponents 1..4 on every
  positive finite base.

It writes the counts of differing words (and the first few) to OUT.json and
exits nonzero if any count is nonzero.
"""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HOST_C = r"""
#include <math.h>
#include <stddef.h>
float __powisf2(float, int);
void h_log10f(const float* x, float* y, size_t n){for(size_t i=0;i<n;i++)y[i]=log10f(x[i]);}
void h_logf(const float* x, float* y, size_t n){for(size_t i=0;i<n;i++)y[i]=logf(x[i]);}
void h_expf(const float* x, float* y, size_t n){for(size_t i=0;i<n;i++)y[i]=expf(x[i]);}
void h_powf(const float* x, const float* e, float* y, size_t n){for(size_t i=0;i<n;i++)y[i]=powf(x[i],e[i]);}
void h_powi(const float* x, int m, float* y, size_t n){for(size_t i=0;i<n;i++)y[i]=__powisf2(x[i],m);}
"""

PROBE = r"""
extern "C" __global__ void noah_libm_probe(const float* x, const float* e,
                                           float* y, int which, int m,
                                           long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float r;
    if (which == 0) r = noah_log10f(v);
    else if (which == 1) r = gfk_log(v);
    else if (which == 2) r = gfk_exp(v);
    else if (which == 3) r = gfk_pow(v, e[i]);
    else r = noah_powi(v, m);
    y[i] = r;
}
"""

CHUNK = 1 << 26


def _host_library(workdir: Path) -> ctypes.CDLL:
    src = workdir / "noah_host_libm.c"
    lib = workdir / "noah_host_libm.so"
    src.write_text(HOST_C, encoding="ascii")
    # -O0 and no builtins: every call is the C library's (libgcc's for powi)
    subprocess.run(["gcc", "-O0", "-fno-builtin", "-fPIC", "-shared",
                    "-o", str(lib), str(src), "-lm"], check=True)
    return ctypes.CDLL(str(lib))


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.c_void_p)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    import cupy as cp

    from gpuwm.core import kernels

    src = kernels.module_source("noah") + PROBE
    module = cp.RawModule(code=src, options=kernels.module_options("noah"))
    probe = module.get_function("noah_libm_probe")
    out: dict = {"strict": __import__("os").environ.get("GPUWM_WRF_EXACT")
                 == "1", "options": list(kernels.module_options("noah")),
                 "checks": {}}
    with tempfile.TemporaryDirectory() as tmp:
        host = _host_library(Path(tmp))
        for fn in ("h_log10f", "h_logf", "h_expf"):
            getattr(host, fn).argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t]
        host.h_powf.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t]
        host.h_powi.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_size_t]

        def run(name, which, bits_iter, host_call, m=0, exps=None):
            bad, total, first = 0, 0, []
            for bits, e in bits_iter:
                x = bits.view(np.float32)
                want = np.empty_like(x)
                host_call(x, e, want)
                dx = cp.asarray(x)
                de = cp.asarray(e if e is not None else x)
                dy = cp.empty_like(dx)
                n = x.size
                probe(((n + 255) // 256,), (256,),
                      (dx, de, dy, np.int32(which), np.int32(m),
                       np.int64(n)))
                got = cp.asnumpy(dy)
                gw, ww = got.view(np.uint32), want.view(np.uint32)
                same = (gw == ww) | (np.isnan(got) & np.isnan(want))
                nb = int((~same).sum())
                if nb and len(first) < 8:
                    for j in np.nonzero(~same)[0][:8 - len(first)]:
                        first.append({"x": f"{int(bits[j]):08x}",
                                      "e": (None if e is None
                                            else f"{int(e.view(np.uint32)[j]):08x}"),
                                      "got": f"{int(gw[j]):08x}",
                                      "want": f"{int(ww[j]):08x}"})
                bad += nb
                total += n
            out["checks"][name] = {"inputs": total, "differing": bad,
                                   "first": first}
            print(f"{name}: {bad} differing of {total}", flush=True)

        def positive_finite():
            for lo in range(0, 0x7f800000, CHUNK):
                hi = min(lo + CHUNK, 0x7f800000)
                yield np.arange(lo, hi, dtype=np.uint32), None
            yield np.asarray([0x80000000, 0x7f800000, 0xff800000,
                              0x7fc00000, 0xbf800000, 0x80000001],
                             np.uint32), None

        def every_word():
            for lo in range(0, 1 << 32, CHUNK):
                yield np.arange(lo, lo + CHUNK, dtype=np.uint64).astype(
                    np.uint32), None

        run("noah_log10f vs log10f", 0, positive_finite(),
            lambda x, e, y: host.h_log10f(_ptr(x), _ptr(y), x.size))
        run("gfk_log vs logf", 1, positive_finite(),
            lambda x, e, y: host.h_logf(_ptr(x), _ptr(y), x.size))
        run("gfk_exp vs expf", 2, every_word(),
            lambda x, e, y: host.h_expf(_ptr(x), _ptr(y), x.size))

        rng = np.random.default_rng(20261007)
        # The exponents the Noah column passes to powf: GENPARM/SOILPARM
        # values reach it through BEXP, FXEXP, CFACTR, SNACCB, QZ and the
        # derived -1/BX, 2*BEXP+3, BEXP+2; the literals are 2, 3, 4, 1.5,
        # 0.2857 (CAPA) and 2.25*DSNOW's base 10.
        exps = np.asarray([2.0, 3.0, 4.0, 1.5, 0.58, 0.5, 2.0 / 7.0,
                           0.28571430, 4.05, 4.26, 4.74, 5.33, 5.25, 6.66,
                           8.72, 11.55, 10.39, 7.12, 12.0, 2.79, 4.0 + 2.0,
                           2 * 4.05 + 3, 2 * 11.55 + 3, -1.0 / 4.05,
                           -1.0 / 5.5, 0.07, 0.25, 0.92, 0.69, 0.08],
                          np.float32)

        def pow_table():
            bases = np.arange(0x3c23d70a, 0x47c35000, 97, dtype=np.uint32)
            for ex in exps:
                for lo in range(0, bases.size, CHUNK):
                    b = bases[lo:lo + CHUNK]
                    yield b, np.full(b.size, ex, np.float32)
            for _ in range(4):
                b = rng.integers(0, 0x7f800000, CHUNK, dtype=np.uint32)
                e = rng.uniform(-30.0, 30.0, CHUNK).astype(np.float32)
                yield b, e

        run("gfk_pow vs powf", 3, pow_table(),
            lambda x, e, y: host.h_powf(_ptr(x), _ptr(e), _ptr(y), x.size))
        for m in (1, 2, 3, 4):
            run(f"noah_powi(x,{m}) vs __powisf2", 4, positive_finite(),
                lambda x, e, y, m=m: host.h_powi(_ptr(x), m, _ptr(y),
                                                 x.size), m=m)
    Path(argv[1]).write_text(json.dumps(out, indent=1) + "\n",
                             encoding="utf-8")
    bad = sum(c["differing"] for c in out["checks"].values())
    print(f"total differing words: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
