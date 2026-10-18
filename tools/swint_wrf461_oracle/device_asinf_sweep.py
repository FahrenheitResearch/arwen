"""WOOF's device asinf over every float32 in [-0.39875, 0.39875].

Usage (GPU, through the box's GPU mutex):
    python tools/swint_wrf461_oracle/device_asinf_sweep.py

0.39875 is sin(23.5 deg), the largest argument radconst hands ASIN.  Each
word is compared with asin((double)x) rounded once to float on the device.
asinf_sweep.c shows on the CPU that rounding asinl (80-bit) and asin
(double) to float agree on every argument in this range, so that rounded
double is the correctly rounded float32 asin here.  Subnormal arguments are
skipped by the comparison (the device double reference flushes them); asin
of a subnormal is the argument itself, which WOOF and the C library both
return.  Prints the count of normal arguments where WOOF differs.
"""
import numpy as np
import cupy as cp

from gpuwm.core.kernels import module_source

SRC = module_source("swint") + r"""
extern "C" __global__ void sweep(unsigned int top, unsigned int sgn,
                                 unsigned long long* nbad,
                                 unsigned int* first) {
  unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x
                         + threadIdx.x;
  if (i > top || i < 0x00800000ull) return;          // normals only
  unsigned int w = (unsigned int)i | sgn;
  float x = __uint_as_float(w);
  float a = glibc_asinf(x);
  float c = __double2float_rn(asin((double)x));
  if (__float_as_uint(a) != __float_as_uint(c)) {
    atomicAdd(nbad, 1ull);
    atomicMin(first, w);
  }
}"""


def main():
    mod = cp.RawModule(code=SRC, options=("-std=c++17", "--fmad=false"))
    k = mod.get_function("sweep")
    top = int(np.float32(0.39875).view(np.uint32))
    normals = 0
    bad = 0
    for sgn in (0, 0x80000000):
        nbad = cp.zeros(1, cp.uint64)
        first = cp.full(1, 0xFFFFFFFF, cp.uint32)
        k(((top + 1 + 255) // 256,), (256,),
          (np.uint32(top), np.uint32(sgn), nbad, first))
        cp.cuda.Device().synchronize()
        normals += top + 1 - 0x00800000
        bad += int(nbad.get()[0])
        if int(nbad.get()[0]):
            print("first differing word", hex(int(first.get()[0])))
    print(f"normal arguments {normals}; WOOF asinf differs from the "
          f"correctly rounded asin on {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
