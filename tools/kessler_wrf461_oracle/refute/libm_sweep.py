#!/usr/bin/env python3
"""Every float32 input: WOOF's gfk_pow(x, c) for Kessler's five constant
exponents and gfk_exp(x), on the device, against the box's C library
powf/expf (the functions the gfortran -O0 Kessler object calls).

    python libm_sweep.py LIBREF.so OUT.json

Device code is compiled from the repo's own glibc_flt32.cuh text through the
same preamble the loader uses, with the Kessler unit's options
("-std=c++17", "--fmad=false"); under GPUWM_WRF_EXACT=1 the strict hook
rewrites them at the NVRTC boundary exactly as for the production unit.
"""
from __future__ import annotations

import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

import gpuwm  # noqa: F401  (installs the strict hook when selected)
from gpuwm.core import kernels
from gpuwm import wrf_exact

F32 = np.float32
EXPONENTS = {"0.1364": F32(0.1364), "0.875": F32(0.875),
             "1004/287": F32(F32(1004.0) / F32(287.0)),
             "0.2046": F32(0.2046), "0.525": F32(0.525)}
KSRC = r"""
extern "C" __global__ void sweep_pow(float y, unsigned int start,
                                      unsigned int n, float* out) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    out[t] = gfk_pow(__uint_as_float(start + t), y);
}
extern "C" __global__ void sweep_exp(unsigned int start, unsigned int n,
                                      float* out) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    out[t] = gfk_exp(__uint_as_float(start + t));
}
"""


def main():
    import cupy as cp
    lib = ctypes.CDLL(sys.argv[1])
    lib.ref_pow.argtypes = [ctypes.c_float, ctypes.c_uint32, ctypes.c_uint32,
                            ctypes.c_void_p]
    lib.ref_exp.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    kd = Path(kernels.__file__).parent
    src = (kernels._preamble() + (kd / "glibc_flt32.cuh").read_text("utf-8")
           + KSRC)
    mod = cp.RawModule(code=src, options=kernels.module_options("kessler"))
    kpow, kexp = mod.get_function("sweep_pow"), mod.get_function("sweep_exp")
    chunk = 1 << 26
    dout = cp.empty(chunk, cp.float32)
    hout = np.empty(chunk, np.float32)
    report = {"strict": wrf_exact.ENABLED,
              "options_seen": [r["options"] for r in
                               wrf_exact.compile_receipt()["records"]][-1:],
              "functions": {}}
    jobs = [("exp", None)] + [("pow", k) for k in EXPONENTS]
    for kind, key in jobs:
        t0 = time.time()
        mism = nan_payload = 0
        examples = []
        for c in range(1 << 32 >> 26):
            start = c * chunk
            grid = ((chunk + 255) // 256,)
            if kind == "exp":
                kexp(grid, (256,), (np.uint32(start), np.uint32(chunk), dout))
                lib.ref_exp(start, chunk, hout.ctypes.data)
            else:
                kpow(grid, (256,), (EXPONENTS[key], np.uint32(start),
                                    np.uint32(chunk), dout))
                lib.ref_pow(float(EXPONENTS[key]), start, chunk,
                            hout.ctypes.data)
            g = cp.asnumpy(dout)
            gb, hb = g.view(np.uint32), hout.view(np.uint32)
            diff = gb != hb
            if diff.any():
                both_nan = np.isnan(g) & np.isnan(hout)
                nan_payload += int((diff & both_nan).sum())
                real = diff & ~both_nan
                mism += int(real.sum())
                for i in np.nonzero(real)[0][:5]:
                    if len(examples) < 20:
                        examples.append({"x_bits": hex(start + int(i)),
                                         "x": float(np.uint32(start + int(i))
                                                    .view(np.float32)),
                                         "device": hex(int(gb[i])),
                                         "libm": hex(int(hb[i]))})
        name = "expf" if kind == "exp" else f"powf(x, {key})"
        report["functions"][name] = {
            "inputs": 1 << 32, "mismatch_non_nan": mism,
            "nan_payload_only": nan_payload, "examples": examples,
            "seconds": round(time.time() - t0, 1)}
        print(name, report["functions"][name], flush=True)
    report["options_seen"] = [r["options"] for r in
                              wrf_exact.compile_receipt()["records"]][-3:]
    Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
