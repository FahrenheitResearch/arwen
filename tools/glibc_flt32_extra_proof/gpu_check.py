"""glibc_flt32_extra.cuh on the device against the host C library, every float32.

The header is compiled as the Morrison unit compiles it
(glibc_flt32.cuh first; -std=c++17 --fmad=false, CuPy's --ftz=true), and
also with --fmad=true.  The 2^32 inputs run in 16 chunks; the reference is
the host's libm.so.6 (gpu_oracle.so).  NaN results compare as NaN.  The
device flushes subnormal RESULTS: log10f never returns one, cbrtf returns
one only for a subnormal argument, and those are counted apart.
"""
from pathlib import Path
import ctypes
import hashlib

import cupy as cp
import numpy as np

root = Path(__file__).resolve().parent
kdir = root.parents[1] / "gpuwm" / "core" / "kernels"
lib = ctypes.CDLL(str(root / "gpu_oracle.so"))
lib.reference.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p,
                          ctypes.c_uint64]
header = (kdir / "glibc_flt32_extra.cuh").read_text(encoding="utf-8")
pre = (kdir / "glibc_flt32.cuh").read_text(encoding="utf-8")
names = ["log10", "cbrt"]
src = pre + "\n" + header + "\n"
for n in names:
    src += ('extern "C" __global__ void check_' + n
            + '(unsigned long long start, unsigned int* y, unsigned long long'
            ' count) { unsigned long long k = (unsigned long long)blockIdx.x'
            ' * blockDim.x + threadIdx.x; if (k < count) y[k] ='
            ' __float_as_uint(gfk_' + n + '(__uint_as_float((unsigned int)'
            '(start + k)))); }\n')
print("CuPy", cp.__version__, "NVRTC", cp.cuda.nvrtc.getVersion(), "GPU",
      cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), flush=True)
print("header_sha256", hashlib.sha256(header.encode()).hexdigest(),
      flush=True)
chunk = 1 << 28
for options in (("-std=c++17", "--fmad=false"), ("-std=c++17", "--fmad=true")):
    module = cp.RawModule(code=src, options=options)
    for f, n in enumerate(names):
        bad = flushed = 0
        examples = []
        dy = cp.empty(chunk, dtype=cp.uint32)
        expected = np.empty(chunk, dtype=np.uint32)
        for start in range(0, 1 << 32, chunk):
            lib.reference(f, start, expected.ctypes.data, chunk)
            module.get_function("check_" + n)(
                ((chunk + 255) // 256,), (256,),
                (np.uint64(start), dy, np.uint64(chunk)))
            got = dy.get()
            ex_f = expected.view(np.float32)
            got_f = got.view(np.float32)
            diff = (got != expected) & ~(np.isnan(ex_f) & np.isnan(got_f))
            if diff.any():
                idx = np.flatnonzero(diff)
                arg = (np.uint64(start) + idx.astype(np.uint64))
                sub_arg = (((arg & np.uint64(0x7f800000)) == 0)
                           & ((arg & np.uint64(0x7fffff)) != 0))
                sub_res = (expected[idx] & 0x7f800000) == 0
                flush = sub_arg & sub_res & ((got[idx] & 0x7fffffff) == 0)
                flushed += int(flush.sum())
                bad += int((~flush).sum())
                for j in idx[~flush][: max(0, 8 - len(examples))]:
                    examples.append((hex(start + int(j)),
                                     hex(int(expected[j])), hex(int(got[j]))))
        print(options, n, "inputs 4294967296 mismatches", bad,
              "flushed_subnormal_results", flushed, "examples", examples,
              flush=True)
