"""Exhaustive check of the two libm words the MYJ kernel calls.

    GPUWM_WRF_EXACT=1 python tools/myjpbl_wrf461_oracle/libm_sweep.py [--out FILE]

module_bl_myjpbl.F's -O0 object calls exactly two libm entry points (see
build.sh's undefined-O0.txt): ``expf`` (PRODQ2 :1104 and :1129, the sea
QSFC at :563) and ``powf`` with the one exponent ``CAPA`` (:477 and :561;
``B1**(2./3.)`` is folded at compile time).  The kernel calls WOOF's own
float32 ``gfk_exp_fma``/``gfk_pow`` (gpuwm/core/kernels/) for
them (``gfk_exp_fma`` from flt32_expf_fma.cuh for EXP).  This sweep
compares, over EVERY float32 bit pattern x,

* ``gfk_exp_fma(x)``     against this host's C-library ``expf(x)``
  (and, for the record, the unfused ``gfk_exp(x)`` too), and
* ``gfk_pow(x, CAPA)``   against this host's C-library ``powf(x, CAPA)``,

bit for bit (two NaNs count as equal whatever their payloads).  The C side
is compiled at -O2 with -fno-builtin and -fno-tree-vectorize so every call
is the scalar library routine the gfortran -O0 oracle calls.  Needs a CUDA
GPU and gcc.  Prints the mismatching inputs, if any.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CAPA = np.float32(np.float32(287.0) / np.float32(1004.5))

C_SRC = r"""
#include <math.h>
#include <stdint.h>
void sweep_expf(const float* x, float* y, int64_t n)
{ for (int64_t i = 0; i < n; ++i) y[i] = expf(x[i]); }
void sweep_powf(const float* x, float* y, int64_t n, float e)
{ for (int64_t i = 0; i < n; ++i) y[i] = powf(x[i], e); }
"""

CU_SRC = r"""
extern "C" __global__ void sweep(unsigned int start, unsigned int n,
                                 float capa, float* e, float* p, float* u)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = __uint_as_float(start + i);
    e[i] = gfk_exp_fma(x);
    p[i] = gfk_pow(x, capa);
    u[i] = gfk_exp(x);
}
"""


def host_lib(tmp: Path):
    src = tmp / "sweep.c"
    lib = tmp / "libsweep.so"
    src.write_text(C_SRC)
    subprocess.run(["gcc", "-O2", "-fno-builtin", "-fno-tree-vectorize",
                    "-fPIC", "-shared", "-o", str(lib), str(src), "-lm"],
                   check=True)
    so = ctypes.CDLL(str(lib))
    fp = ctypes.POINTER(ctypes.c_float)
    so.sweep_expf.argtypes = (fp, fp, ctypes.c_int64)
    so.sweep_powf.argtypes = (fp, fp, ctypes.c_int64, ctypes.c_float)
    return so


def same(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    nan = np.isnan(a) & np.isnan(b)
    return (a.view(np.uint32) == b.view(np.uint32)) | nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--chunk-log2", type=int, default=26)
    args = ap.parse_args()

    import gpuwm  # noqa: F401  (installs the strict compile hook if asked)
    import cupy as cp

    kdir = ROOT / "gpuwm" / "core" / "kernels"
    header = "".join((kdir / name).read_text(encoding="utf-8")
                     for name in ("glibc_flt32.cuh", "flt32_expf_fma.cuh"))
    module = cp.RawModule(code=header + CU_SRC,
                          options=("-std=c++17", "--fmad=false"))
    kern = module.get_function("sweep")
    with tempfile.TemporaryDirectory() as tmp:
        so = host_lib(Path(tmp))
        chunk = 1 << args.chunk_log2
        miss = {"expf": [], "powf_capa": [], "expf_unfused_gfk_exp": []}
        counts = {"expf": 0, "powf_capa": 0, "expf_unfused_gfk_exp": 0}
        checked = 0
        for start in range(0, 1 << 32, chunk):
            bits = np.arange(start, start + chunk, dtype=np.uint64)
            x = bits.astype(np.uint32).view(np.float32)
            ge = cp.empty(chunk, dtype=cp.float32)
            gp = cp.empty(chunk, dtype=cp.float32)
            gu = cp.empty(chunk, dtype=cp.float32)
            kern(((chunk + 255) // 256,), (256,),
                 (np.uint32(start), np.uint32(chunk), CAPA, ge, gp, gu))
            he = np.empty(chunk, dtype=np.float32)
            hp = np.empty(chunk, dtype=np.float32)
            fp = ctypes.POINTER(ctypes.c_float)
            so.sweep_expf(x.ctypes.data_as(fp), he.ctypes.data_as(fp), chunk)
            so.sweep_powf(x.ctypes.data_as(fp), hp.ctypes.data_as(fp), chunk,
                          ctypes.c_float(float(CAPA)))
            ge, gp, gu = cp.asnumpy(ge), cp.asnumpy(gp), cp.asnumpy(gu)
            for name, g, h in (("expf", ge, he), ("powf_capa", gp, hp),
                               ("expf_unfused_gfk_exp", gu, he)):
                bad = np.nonzero(~same(g, h))[0]
                counts[name] += int(bad.size)
                for i in bad[:64]:
                    miss[name].append({
                        "x_bits": f"0x{int(start + i):08X}",
                        "x": float(x[i]),
                        "gfk": f"0x{int(g[i].view(np.uint32)):08X}",
                        "libc": f"0x{int(h[i].view(np.uint32)):08X}"})
            checked += chunk
    result = {"inputs_checked_each": checked,
              "strict": os.environ.get("GPUWM_WRF_EXACT") == "1",
              "capa_bits": f"0x{int(CAPA.view(np.uint32)):08X}",
              "mismatches": counts,
              "first_mismatches": miss}
    text = json.dumps(result, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
