"""TEMPO device elementary functions against the host C library on the
argument ranges the lane's probe (tests/test_tempo_device_libm.py) samples
thinly or not at all:

  exp    subnormal results (x in [-745.2, -708.3]), near overflow
         ([709, 709.79]), |x| < 2^-54, negative arguments below -60
  log    subnormal inputs, 1 +- tiny, huge inputs
  pow    subnormal and near-overflow results, subnormal bases, huge |y|,
         integral y with negative bases, bases next to 1 with large y,
         10**y for y in [-330, 310] (TEMPO's 10.0**loga form)
  log10  exact powers of ten 1e-323 .. 1e308 and their neighbours,
         subnormal inputs, powers of two, values next to 1

Every function is compared bitwise (NaN payloads ignored, NaN vs NaN is a
match).  Writes a JSON summary with counts and the first mismatches.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import json
import sys
from pathlib import Path

import numpy as np

KERNEL = r"""
#include "tempo_common.cuh"
extern "C" __global__ void probe(const double* x, const double* y, int n, int which, double* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double r;
  if (which == 0) r = tempo_exp(x[i]);
  else if (which == 1) r = tempo_log(x[i]);
  else if (which == 2) r = tempo_pow(x[i], y[i]);
  else r = tempo_log10(x[i]);
  out[i] = r;
}
"""


def nextafter_n(x, n, toward):
    out = [x]
    v = x
    for _ in range(n):
        v = np.nextafter(v, toward)
        out.append(v)
    return out


def args(rng):
    sets = {}
    N = 1 << 21
    sets["exp"] = np.concatenate([
        rng.uniform(-745.2, -708.3, N), np.linspace(-745.13321910194122, -708.39641853226408, N),
        rng.uniform(709.0, 709.782712893384, N // 4), rng.uniform(-2.0 ** -54, 2.0 ** -54, N // 8),
        rng.uniform(-708.3, -60.0, N), -np.abs(rng.standard_normal(N // 4)) * 300.0,
    ])
    sub = (rng.integers(1, 1 << 52, N, dtype=np.uint64)).view(np.float64)
    near1 = 1.0 + rng.uniform(-1e-8, 1e-8, N // 2)
    sets["log"] = np.concatenate([sub, near1, 10.0 ** rng.uniform(250, 308, N // 4),
                                  rng.uniform(0.9, 1.1, N // 2)])
    px = np.concatenate([
        rng.uniform(1e-310, 1e-300, N // 4),                 # tiny and subnormal bases
        (rng.integers(1, 1 << 52, N // 4, dtype=np.uint64)).view(np.float64),
        rng.uniform(0.5, 2.0, N // 2),                       # moderate base, huge |y|
        1.0 + rng.uniform(-1e-10, 1e-10, N // 4),            # base next to 1, large y
        -rng.uniform(0.1, 10.0, N // 4),                     # negative base, integral y
        np.full(N // 2, 10.0),                               # 10**y
        rng.uniform(1e-6, 1e-2, N // 4),                     # x**y -> subnormal
    ])
    py = np.concatenate([
        rng.uniform(0.5, 1.2, N // 4),
        rng.uniform(0.2, 3.0, N // 4),
        rng.uniform(-1100.0, 1100.0, N // 2),
        rng.uniform(1e9, 1e12, N // 4) * rng.choice([-1.0, 1.0], N // 4),
        rng.integers(-60, 60, N // 4).astype(np.float64),
        rng.uniform(-330.0, 310.0, N // 2),
        rng.uniform(55.0, 170.0, N // 4),
    ])
    sets["pow"] = (px, py)
    pw = [10.0 ** k for k in range(-323, 309)]
    pw = np.array([float(f"1e{k}") for k in range(-323, 309)])
    nb = np.concatenate([np.nextafter(pw, 0.0), pw, np.nextafter(pw, np.inf)])
    sets["log10"] = np.concatenate([
        nb, sub[: N // 2], 2.0 ** np.arange(-1074, 1024, dtype=np.float64),
        1.0 + rng.uniform(-1e-12, 1e-12, N // 4), 10.0 ** rng.uniform(-30, 30, N)])
    return sets


def main():
    out_path = Path(sys.argv[1])
    import cupy as cp
    from gpuwm.core.tempo_runtime import composed_source
    mod = cp.RawModule(code=composed_source(KERNEL),
                       options=("-std=c++17", "-DTEMPO_NZ_MAX=8", "--fmad=false"))
    fn = mod.get_function("probe")
    lib = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
    for name in ("exp", "log", "log10"):
        getattr(lib, name).restype = ctypes.c_double
        getattr(lib, name).argtypes = (ctypes.c_double,)
    lib.pow.restype = ctypes.c_double
    lib.pow.argtypes = (ctypes.c_double, ctypes.c_double)
    rng = np.random.default_rng(20261009)
    sets = args(rng)
    summary = {}
    for which, name in enumerate(("exp", "log", "pow", "log10")):
        if name == "pow":
            x, y = sets["pow"]
        else:
            x = sets[name]
            y = np.ones_like(x)
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        n = x.size
        o = cp.empty(n, dtype=cp.float64)
        fn(((n + 255) // 256,), (256,), (cp.asarray(x), cp.asarray(y), np.int32(n),
                                         np.int32(which), o))
        got = o.get()
        f = getattr(lib, name)
        want = (np.array([f(a, b) for a, b in zip(x, y)]) if name == "pow"
                else np.array([f(a) for a in x]))
        both_nan = np.isnan(got) & np.isnan(want)
        bad = np.nonzero((got.view(np.int64) != want.view(np.int64)) & ~both_nan)[0]
        summary[name] = dict(n=int(n), mismatches=int(bad.size),
                             examples=[dict(x=float(x[i]).hex(), y=float(y[i]).hex(),
                                            device=float(got[i]).hex(), host=float(want[i]).hex())
                                       for i in bad[:20]])
        print(name, n, "mismatches", bad.size)
    out_path.write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
