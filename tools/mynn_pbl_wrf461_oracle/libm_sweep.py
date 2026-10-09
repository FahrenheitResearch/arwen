"""Every float32 input of the MYNN unit's device libm against the live glibc.

The WRF v4.6.1 MYNN oracles are gfortran -O0 builds linked against glibc
2.39, so a real-argument EXP, LOG, LOG10, ATAN, TANH or ``x**y`` in
``module_bl_mynn.F`` is that glibc's ``expf``/``logf``/``log10f``/``atanf``/
``tanhf``/``powf``.  ``gpuwm/core/kernels/mynn_pbl.cu`` carries device
transcriptions of those six functions (``gfk_exp`` from glibc_flt32.cuh for
expf).  This tool evaluates the device words through the unit's two probe
kernels on every one of the 2**32 float32 bit patterns (``powf`` on every
``x`` for each exponent the unit uses) and compares them, bit for bit,
with the host's own glibc called through a small C shim built here with
gcc.  NaN results compare equal when both sides are NaN.

Run on a box whose glibc is the oracle's (2.39) with one card::

    PYTHONPATH=<checkout> python tools/mynn_pbl_wrf461_oracle/libm_sweep.py \
        [--chunk-bits 28] [--out sweep.json]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

#: Every exponent the unit hands powf, by value (named constants resolved):
#: phim/phih 2.5, 1/2.5, 1/2.5 - 1, 1/1.1, 1/1.1 - 1, 0.25, 0.5, 0.333333,
#: -0.6666667; the length laws 1/3, 0.2, 2/3, 1.5, 0.667, 0.190; plus 1, 2.
POWF_EXPONENTS = (
    2.5, np.float32(1.0) / np.float32(2.5),
    np.float32(np.float32(1.0) / np.float32(2.5)) - np.float32(1.0),
    1.1, np.float32(1.0) / np.float32(1.1),
    np.float32(np.float32(1.0) / np.float32(1.1)) - np.float32(1.0),
    0.25, 0.5, 0.333333, -0.6666667, 0.33333334, 0.2, 0.6666667, 1.5,
    0.667, 0.190, 1.0, 2.0, 3.0,
)

_C_SHIM = r"""
#include <math.h>
#include <stdint.h>
#include <string.h>
typedef float (*unary)(float);
static unary pick(int fn) {
    switch (fn) { case 0: return logf; case 1: return atanf; case 2: return tanhf;
                  case 3: return log10f; case 4: return expf; }
    return 0;
}
void sweep_unary(int fn, uint64_t start, uint64_t count, float *out) {
    unary f = pick(fn);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)count; ++i) {
        uint32_t bits = (uint32_t)(start + (uint64_t)i); float x;
        memcpy(&x, &bits, 4); out[i] = f(x);
    }
}
void sweep_pow(float y, uint64_t start, uint64_t count, float *out) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)count; ++i) {
        uint32_t bits = (uint32_t)(start + (uint64_t)i); float x;
        memcpy(&x, &bits, 4); out[i] = powf(x, y);
    }
}
"""

UNARY = ("logf", "atanf", "tanhf", "log10f", "expf")


def _shim():
    build = Path(tempfile.mkdtemp(prefix="mynn-libm-"))
    (build / "shim.c").write_text(_C_SHIM)
    subprocess.run(["gcc", "-O1", "-fno-builtin", "-fopenmp", "-fPIC",
                    "-shared", "-o", str(build / "shim.so"),
                    str(build / "shim.c"), "-lm"], check=True)
    lib = ctypes.CDLL(str(build / "shim.so"))
    lib.sweep_unary.argtypes = (ctypes.c_int, ctypes.c_uint64,
                                ctypes.c_uint64, ctypes.c_void_p)
    lib.sweep_pow.argtypes = (ctypes.c_float, ctypes.c_uint64,
                              ctypes.c_uint64, ctypes.c_void_p)
    return lib


def _mismatch(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    same = (got.view(np.uint32) == want.view(np.uint32))
    same |= np.isnan(got) & np.isnan(want)
    return ~same


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunk-bits", type=int, default=28)
    parser.add_argument("--out")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args(argv)
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    lib = _shim()
    chunk = 1 << args.chunk_bits
    probe1 = get_kernel("mynn_pbl", "mynn_glibc_libm_probe")
    probe2 = get_kernel("mynn_pbl", "mynn_glibc_libm_probe2")
    host = np.empty(chunk, dtype=np.float32)
    dev_out = cp.empty(3 * chunk, dtype=np.float32)
    dev_y = cp.empty(chunk, dtype=np.float32)
    report = {"glibc": platform.libc_ver()[1], "functions": {}}
    tasks = [(name, None) for name in UNARY]
    tasks += [("powf", float(np.float32(y))) for y in POWF_EXPONENTS]
    if args.only:
        tasks = [task for task in tasks if task[0] in args.only]
    threads = 256
    for name, y in tasks:
        label = name if y is None else f"powf(x, {np.float32(y)!r})"
        mismatches, examples = 0, []
        for start in range(0, 1 << 32, chunk):
            x = (cp.arange(chunk, dtype=cp.uint64) + start).astype(
                cp.uint32).view(cp.float32)
            grid = ((chunk + threads - 1) // threads,)
            if name in ("logf", "atanf") or y is not None:
                dev_y.fill(np.float32(0.0 if y is None else y))
                probe1(grid, (threads,), (x, dev_y, dev_out, np.int32(chunk)))
                column = {"logf": 0, "atanf": 1}.get(name, 2)
            else:
                probe2(grid, (threads,), (x, dev_out, np.int32(chunk)))
                column = {"tanhf": 0, "log10f": 1, "expf": 2}[name]
            got = cp.asnumpy(dev_out.reshape(chunk, 3)[:, column])
            if y is None:
                lib.sweep_unary(UNARY.index(name), start, chunk,
                                host.ctypes.data)
            else:
                lib.sweep_pow(ctypes.c_float(np.float32(y)), start, chunk,
                              host.ctypes.data)
            bad = np.flatnonzero(_mismatch(got, host))
            mismatches += int(bad.size)
            for index in bad[: max(0, 5 - len(examples))]:
                examples.append({
                    "x_bits": f"{start + int(index):08x}",
                    "device_bits": f"{int(got[index:index+1].view(np.uint32)[0]):08x}",
                    "glibc_bits": f"{int(host[index:index+1].view(np.uint32)[0]):08x}"})
        report["functions"][label] = {"inputs": 1 << 32,
                                      "mismatches": mismatches,
                                      "examples": examples}
        print(label, mismatches, examples[:2], flush=True)
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0 if all(row["mismatches"] == 0
                    for row in report["functions"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
