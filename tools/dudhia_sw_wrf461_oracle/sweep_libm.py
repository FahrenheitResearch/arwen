#!/usr/bin/env python3
"""Exhaustive word-for-word check of the Dudhia kernel's two libm calls.

    python sweep_libm.py BUILD_DIR [--json OUT]

SWPARA calls ALOG10(WGM+1.) and (1.+141.5*UGCM)**0.635.  Both arguments are
>= 1, so the domain is every float32 from 1.0 up to and including +inf
(1,073,741,825 words each).  This runs the kernel's own device functions
(dsw_log10f and gfk_pow with y = 0.635f, compiled from the same assembled
source the production loader builds) on every word and compares each
result bitwise with the C library's log10f/powf from build.sh's
libm_sweep.so, called in this process on this CPU.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIRST = 0x3F800000          # 1.0f
LAST = 0x7F800000           # +inf
CHUNK = 1 << 26

_SWEEP = r'''
extern "C" __global__ void dsw_sweep(unsigned int first, unsigned int n,
                                     int which, float* __restrict__ out)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = __uint_as_float(first + i);
    out[i] = which == 0 ? dsw_log10f(x) : gfk_pow(x, 0.635f);
}
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("build_dir", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    import gpuwm  # noqa: F401  (installs the strict compiler hook if set)
    import cupy as cp
    from gpuwm.core.kernels import module_source, module_options

    lib = ctypes.CDLL(str(args.build_dir / "libm_sweep.so"))
    lib.sweep_log10f.argtypes = (ctypes.c_uint32, ctypes.c_uint64,
                                 ctypes.c_void_p)
    lib.sweep_powf.argtypes = (ctypes.c_uint32, ctypes.c_uint64,
                               ctypes.c_float, ctypes.c_void_p)
    src = module_source("dudhia_sw") + _SWEEP
    mod = cp.RawModule(code=src, options=module_options("dudhia_sw"))
    kern = mod.get_function("dsw_sweep")

    report = {"strict": os.environ.get("GPUWM_WRF_EXACT", "0") == "1",
              "domain": "every float32 in [1.0, +inf]",
              "words": LAST - FIRST + 1}
    host = np.empty(CHUNK, np.float32)
    dev = cp.empty(CHUNK, cp.float32)
    for which, name in ((0, "log10f"), (1, "powf(x, 0.635f)")):
        misses, first_miss = 0, None
        start = FIRST
        while start <= LAST:
            n = min(CHUNK, LAST - start + 1)
            kern(((n + 255) // 256,), (256,),
                 (np.uint32(start), np.uint32(n), np.int32(which), dev))
            if which == 0:
                lib.sweep_log10f(start, n, host.ctypes.data)
            else:
                lib.sweep_powf(start, n, ctypes.c_float(0.635),
                               host.ctypes.data)
            got = cp.asnumpy(dev[:n]).view(np.uint32)
            want = host[:n].view(np.uint32)
            bad = np.nonzero(got != want)[0]
            if bad.size:
                misses += int(bad.size)
                if first_miss is None:
                    j = int(bad[0])
                    first_miss = {
                        "x_bits": hex(start + j),
                        "woof_bits": hex(int(got[j])),
                        "libm_bits": hex(int(want[j]))}
            start += n
        report[name] = {"not_bit_identical": misses,
                        "first_miss": first_miss}
        print(name, misses, first_miss, flush=True)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if all(report[k]["not_bit_identical"] == 0
                    for k in ("log10f", "powf(x, 0.635f)")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
