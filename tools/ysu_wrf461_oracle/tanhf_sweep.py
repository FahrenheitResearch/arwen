#!/usr/bin/env python3
"""Hold the YSU kernel's tanhf to the reference platform's, float by float.

    python tools/ysu_wrf461_oracle/tanhf_sweep.py [--write-sample OUT.bin]

get_pblh (bl_ysu.F90:1684) weights two PBL heights by
``0.5*tanh((zi-200)/400)+0.5``; zi runs from 0 to the column top, so the
argument runs from -0.5 up, and tanhf is exactly 1 beyond about 9.01.  This
sweep evaluates ``ysu_tanhf`` (kernels/ysu_topo.cuh) on the GPU and the C
library's ``tanhf`` on the CPU for EVERY float32 in [-0.5, 22] -- 2.17e9
arguments, in chunks -- and counts the words that differ.  It also counts
how many of the C library's words differ from the correctly rounded tanh
(long double tanhl, rounded once), which is what the kernel computed before
lane/verify-ysu, so the size of what was fixed is measured on the same run.

Run it on the reference platform (box W1: Ubuntu 24.04, glibc 2.39).  On a
glibc with a correctly rounded tanhf (2.41 and later) the library column
changes and the kernel, which keeps the classic words, will differ there.

``--write-sample`` stores (argument, library word) pairs -- every argument
where the library differs from the correctly rounded value up to 4096, plus
4096 spread across the range -- for the GPU test that runs anywhere.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

C_SOURCE = r"""
#include <math.h>
#include <stdint.h>
#include <string.h>
void libm_tanhf(const float *x, float *out, int64_t n) {
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i) out[i] = tanhf(x[i]);
}
void cr_tanhf(const float *x, float *out, int64_t n) {
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i) out[i] = (float)tanhl((long double)x[i]);
}
"""

KERNEL = r"""
extern "C" __global__ void ysu_tanhf_probe(const float *x, float *o, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = ysu_tanhf(x[i]);
}
"""


def _clib():
    tmp = Path(tempfile.mkdtemp(prefix="tanhf_sweep_"))
    (tmp / "t.c").write_text(C_SOURCE)
    so = tmp / "t.so"
    subprocess.run(["gcc", "-O2", "-fopenmp", "-fPIC", "-shared", "-o",
                    str(so), str(tmp / "t.c"), "-lm"], check=True)
    lib = ctypes.CDLL(str(so))
    for name in ("libm_tanhf", "cr_tanhf"):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_int64]
    return lib


def _call(lib, name, x):
    out = np.empty_like(x)
    getattr(lib, name)(x.ctypes.data, out.ctypes.data, x.size)
    return out


def _kernel():
    import cupy as cp

    from gpuwm.core.kernels import _extra_header_text, _preamble, module_options
    src = _preamble() + _extra_header_text("ysu") + KERNEL
    mod = cp.RawModule(code=src, options=module_options("ysu"))
    return mod.get_function("ysu_tanhf_probe")


def gpu_tanhf(fn, x: np.ndarray) -> np.ndarray:
    import cupy as cp
    d = cp.asarray(x)
    o = cp.empty_like(d)
    n = d.size
    fn(((n + 255) // 256,), (256,), (d, o, np.int64(n)))
    return cp.asnumpy(o)


def all_floats(lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Bit-pattern ranges for every float32 in [lo, hi] (lo < 0 < hi)."""
    neg = (np.uint32(0x80000000), np.float32(lo).view(np.uint32))
    pos = (np.uint32(0), np.float32(hi).view(np.uint32))
    return neg, pos


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=-0.5)
    ap.add_argument("--hi", type=float, default=22.0)
    ap.add_argument("--chunk", type=int, default=1 << 26)
    ap.add_argument("--write-sample")
    args = ap.parse_args(argv)
    lib = _clib()
    fn = _kernel()
    total = kernel_vs_libm = libm_vs_cr = kernel_vs_cr = 0
    lib_bad: list[np.ndarray] = []
    first = []
    for start, stop in all_floats(args.lo, args.hi):
        b = int(start)
        while b <= int(stop):
            e = min(b + args.chunk, int(stop) + 1)
            x = np.arange(b, e, dtype=np.uint32).view(np.float32)
            got = gpu_tanhf(fn, x)
            ref = _call(lib, "libm_tanhf", x)
            cr = _call(lib, "cr_tanhf", x)
            gv, rv, cv = got.view(np.uint32), ref.view(np.uint32), cr.view(np.uint32)
            d = gv != rv
            kernel_vs_libm += int(d.sum())
            if d.any() and len(first) < 8:
                for i in np.flatnonzero(d)[:8 - len(first)]:
                    first.append((float(x[i]).hex(), float(got[i]).hex(),
                                  float(ref[i]).hex()))
            lc = rv != cv
            libm_vs_cr += int(lc.sum())
            kernel_vs_cr += int((gv != cv).sum())
            if lc.any() and sum(a.size for a in lib_bad) < 4096:
                lib_bad.append(x[lc][:4096])
            total += x.size
            b = e
    libc = os.confstr("CS_GNU_LIBC_VERSION") if hasattr(os, "confstr") else "?"
    print(f"C library: {libc}")
    print(f"floats checked in [{args.lo}, {args.hi}]: {total}")
    print(f"ysu_tanhf != library tanhf: {kernel_vs_libm}")
    print(f"library tanhf != correctly rounded: {libm_vs_cr}")
    print(f"ysu_tanhf != correctly rounded: {kernel_vs_cr}")
    for row in first:
        print("  differs:", row)
    if args.write_sample:
        rng = np.random.default_rng(1684)
        spread = rng.uniform(args.lo, 10.0, 4096).astype(np.float32)
        bad = np.concatenate(lib_bad)[:4096] if lib_bad else np.empty(0, np.float32)
        xs = np.concatenate([bad, spread]).astype(np.float32)
        ys = _call(lib, "libm_tanhf", xs)
        header = np.asarray([bad.size, spread.size], dtype="<i4")
        with open(args.write_sample, "wb") as fh:
            fh.write(header.tobytes())
            fh.write(xs.astype("<f4").tobytes())
            fh.write(ys.astype("<f4").tobytes())
        print(f"sample written: {bad.size} library-vs-correctly-rounded "
              f"arguments + {spread.size} spread, to {args.write_sample}")
    return 0 if kernel_vs_libm == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
