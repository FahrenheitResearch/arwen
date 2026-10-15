"""Sweep gfk_log10 (log10_flt32.cuh) against the host libm's log10f.

    python sweep_log10.py            # every positive finite float, plus specials

Run on the oracle box (Linux, gfortran's libm): gfortran lowers default-REAL
LOG10 to a call of the C library's log10f, so this is the function the
Fortran oracle evaluated.  The host side is a small C helper compiled here
with gcc and called through ctypes (OpenMP over the chunk); the device side
compiles WOOF's header the way the loader does (glibc_flt32.cuh then
log10_flt32.cuh) under the strict options.  Prints the count of differing
results; exit 0 only when there are none.  Also reports how many results
CUDA's own log10f gets wrong against the same reference, as the control
that the sweep can see a difference at all.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HOST_C = r"""
#include <math.h>
#include <stdint.h>
#include <string.h>
void host_log10f(uint32_t start, long long n, float *out) {
  #pragma omp parallel for schedule(static)
  for (long long j = 0; j < n; ++j) {
    uint32_t b = start + (uint32_t)j; float x; memcpy(&x, &b, 4);
    out[j] = log10f(x);
  }
}
"""

DEVICE = r"""
extern "C" __global__ void sweep(unsigned int start, long long n,
                                 float* woof, float* cuda) {
  long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  float x = __uint_as_float(start + (unsigned int)j);
  woof[j] = gfk_log10(x);
  cuda[j] = log10f(x);
}
"""


def main():
    # The strict build's compile hook: CuPy appends -ftz=true after caller
    # options, and only gpuwm.wrf_exact (GPUWM_WRF_EXACT=1) normalizes that
    # at the NVRTC boundary.  Without it subnormal inputs are flushed.
    import os
    if os.environ.get("GPUWM_WRF_EXACT") != "1":
        raise SystemExit("run with GPUWM_WRF_EXACT=1 (strict build)")
    import gpuwm  # noqa: F401  installs the strict-arithmetic hook
    import cupy as cp
    root = Path(__file__).resolve().parents[2]
    kdir = root / "gpuwm" / "core" / "kernels"
    src = ((kdir / "glibc_flt32.cuh").read_text(encoding="utf-8")
           + (kdir / "log10_flt32.cuh").read_text(encoding="utf-8") + DEVICE)
    mod = cp.RawModule(code=src, options=("-std=c++17", "--fmad=false",
                                          "--ftz=false", "--prec-div=true",
                                          "--prec-sqrt=true"))
    kern = mod.get_function("sweep")
    with tempfile.TemporaryDirectory() as td:
        c = Path(td) / "h.c"
        so = Path(td) / "h.so"
        c.write_text(HOST_C)
        subprocess.run(["gcc", "-O2", "-fopenmp", "-fno-tree-vectorize", "-shared", "-fPIC", str(c),
                        "-o", str(so), "-lm"], check=True)
        lib = ctypes.CDLL(str(so))
        lib.host_log10f.argtypes = (ctypes.c_uint32, ctypes.c_longlong,
                                    ctypes.c_void_p)
        chunk = 1 << 27
        bad = 0
        cuda_bad = 0
        total = 0
        first = None
        # every positive finite float, plus +0, -0, +inf, NaN and a negative
        ranges = [(0x00000000, 0x7F800000), (0x7F800000, 0x7F800001),
                  (0x7FC00000, 0x7FC00001), (0x80000000, 0x80000001),
                  (0xBF800000, 0xBF800001)]
        for lo, hi in ranges:
            start = lo
            while start < hi:
                n = min(chunk, hi - start)
                host = np.empty(n, np.float32)
                lib.host_log10f(start, n, host.ctypes.data)
                woof = cp.empty(n, cp.float32)
                cuda = cp.empty(n, cp.float32)
                kern(((n + 255) // 256,), (256,),
                     (np.uint32(start), np.int64(n), woof, cuda))
                w = cp.asnumpy(woof).view(np.uint32)
                cu = cp.asnumpy(cuda).view(np.uint32)
                h = host.view(np.uint32)
                # NaN payloads: compare NaN-ness, not bits
                hn = np.isnan(host)
                diff = (w != h) & ~(hn & np.isnan(w.view(np.float32)))
                cdiff = (cu != h) & ~(hn & np.isnan(cu.view(np.float32)))
                bad += int(diff.sum())
                cuda_bad += int(cdiff.sum())
                if first is None and diff.any():
                    j = int(np.argmax(diff))
                    first = (start + j, host[j], w.view(np.float32)[j])
                total += n
                start += n
    print(f"swept {total} inputs: gfk_log10 differs from host log10f in "
          f"{bad}; CUDA log10f differs in {cuda_bad}")
    if first:
        print(f"first difference at bits {first[0]:#010x}: host={first[1]!r} "
              f"woof={first[2]!r}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
