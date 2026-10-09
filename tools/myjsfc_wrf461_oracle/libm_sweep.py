"""Exhaustive check of the libm words the Eta similarity surface layer calls.

    GPUWM_WRF_EXACT=1 python tools/myjsfc_wrf461_oracle/libm_sweep.py [--out FILE]

module_sf_myjsfc.F's -O0 object calls logf, expf, powf and atanf (see the
fixture's compiler.txt).  ATAN only builds the MYJSFCINIT tables on the
host, which the oracle's own table dump grades word for word.  On the
device the kernel calls WOOF's own float32 routines: gfk_log and gfk_pow
(glibc_flt32.cuh) and gfk_exp_fma (flt32_expf_fma.cuh), with the three
exponents the source uses: CAPA (THSK and APESFC), RCAP (the shelter
pressures P02P/P10P) and 2./3. (the Beljaars WSTAR2).  This sweep
compares, over EVERY float32 bit pattern x,

* gfk_log(x)                                     against this host's logf(x)
* gfk_exp_fma(x) and the unfused gfk_exp(x)      against this host's expf(x)
* gfk_pow(x, e), e in CAPA, RCAP, 2./3.          against this host's powf(x, e)

bit for bit (two NaNs count as equal whatever their payloads).  The C side
is compiled at -O2 with -fno-builtin and -fno-tree-vectorize, so every call
is the scalar library routine the gfortran oracle calls.  Needs a CUDA GPU
and gcc.

Exit 0 only if the forms the kernel calls match everywhere except at the
inputs DECLARED below, each with the reason the scheme cannot reach it.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

F = np.float32
CAPA = F(F(287.0) / F(F(F(7.0) * F(287.0)) / F(2.0)))
RCAP = F(F(1.0) / CAPA)
TWO_THIRDS = F(F(2.0) / F(3.0))
EXPONENTS = {"capa": CAPA, "rcap": RCAP, "two_thirds": TWO_THIRDS}

#: Inputs where a form the kernel calls may differ from the host, with why
#: MYJSFC cannot reach them.  gfk_pow(x, RCAP) is 1 ULP off at x = 0.00825:
#: x is RAPA02 or RAPA10 = (PSFC/1E5)**CAPA - GOCP/TH (module_sf_myjsfc.F
#: :330-333), which is that small only for PSFC below 0.01 Pa.
DECLARED = {"powf_rcap": {"0x3C072A38"}}

C_SRC = r"""
#include <math.h>
#include <stdint.h>
void sweep_logf(uint32_t start, float* y, int64_t n)
{ for (int64_t i = 0; i < n; ++i) { uint32_t b = start + (uint32_t)i; float x;
  __builtin_memcpy(&x, &b, 4); y[i] = logf(x); } }
void sweep_expf(uint32_t start, float* y, int64_t n)
{ for (int64_t i = 0; i < n; ++i) { uint32_t b = start + (uint32_t)i; float x;
  __builtin_memcpy(&x, &b, 4); y[i] = expf(x); } }
void sweep_powf(uint32_t start, float* y, int64_t n, float e)
{ for (int64_t i = 0; i < n; ++i) { uint32_t b = start + (uint32_t)i; float x;
  __builtin_memcpy(&x, &b, 4); y[i] = powf(x, e); } }
"""

CU_SRC = r"""
extern "C" __global__ void sweep(unsigned int start, unsigned int n,
                                 float e0, float e1, float e2,
                                 float* lu, float* ef, float* eu,
                                 float* q0, float* q1, float* q2)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = __uint_as_float(start + i);
    lu[i] = gfk_log(x);
    ef[i] = gfk_exp_fma(x);
    eu[i] = gfk_exp(x);
    q0[i] = gfk_pow(x, e0);
    q1[i] = gfk_pow(x, e1);
    q2[i] = gfk_pow(x, e2);
}
"""


def _clib(tmp: Path):
    src = tmp / "sweep.c"
    lib = tmp / "sweep.so"
    src.write_text(C_SRC)
    subprocess.run(["gcc", "-O2", "-fno-builtin", "-fno-tree-vectorize",
                    "-shared", "-fPIC", "-o", str(lib), str(src), "-lm"],
                   check=True)
    clib = ctypes.CDLL(str(lib))
    ptr = ctypes.c_void_p
    for name in ("sweep_logf", "sweep_expf"):
        getattr(clib, name).argtypes = [ctypes.c_uint32, ptr, ctypes.c_int64]
    clib.sweep_powf.argtypes = [ctypes.c_uint32, ptr, ctypes.c_int64,
                                ctypes.c_float]
    return clib


def _mismatch(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    differ = got.view(np.uint32) != want.view(np.uint32)
    return differ & ~(np.isnan(got) & np.isnan(want))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--chunk-log2", type=int, default=26)
    args = ap.parse_args()

    import cupy as cp
    from gpuwm.core.kernels import _KDIR, _preamble
    source = (_preamble()
              + (_KDIR / "glibc_flt32.cuh").read_text(encoding="utf-8")
              + (_KDIR / "flt32_expf_fma.cuh").read_text(encoding="utf-8")
              + CU_SRC)
    module = cp.RawModule(code=source, options=("-std=c++17", "--fmad=false"))
    kernel = module.get_function("sweep")

    names = ("logf", "expf_fma", "expf_unfused", "powf_capa", "powf_rcap",
             "powf_two_thirds")
    bad = {name: [] for name in names}
    counts = {name: 0 for name in names}
    chunk = 1 << args.chunk_log2
    with tempfile.TemporaryDirectory() as tmp:
        clib = _clib(Path(tmp))
        dev = [cp.empty(chunk, dtype=F) for _ in names]
        host_ref = {k: np.empty(chunk, dtype=F)
                    for k in ("log", "exp", "capa", "rcap", "two_thirds")}
        for start in range(0, 1 << 32, chunk):
            kernel(((chunk + 255) // 256,), (256,),
                   (np.uint32(start), np.uint32(chunk), CAPA, RCAP,
                    TWO_THIRDS, *dev))
            clib.sweep_logf(start, host_ref["log"].ctypes.data, chunk)
            clib.sweep_expf(start, host_ref["exp"].ctypes.data, chunk)
            for key, e in EXPONENTS.items():
                clib.sweep_powf(start, host_ref[key].ctypes.data, chunk,
                                float(e))
            got = [cp.asnumpy(d) for d in dev]
            refs = (host_ref["log"], host_ref["exp"], host_ref["exp"],
                    host_ref["capa"], host_ref["rcap"], host_ref["two_thirds"])
            for name, g, want in zip(names, got, refs):
                miss = np.flatnonzero(_mismatch(g, want))
                counts[name] += int(miss.size)
                for i in miss[:64]:
                    if len(bad[name]) < 64:
                        bad[name].append({
                            "x_bits": f"0x{start + int(i):08X}",
                            "x": float(np.uint32(start + int(i)).view(F)),
                            "got_bits": f"0x{int(g[i].view(np.uint32)):08X}",
                            "want_bits":
                                f"0x{int(want[i].view(np.uint32)):08X}"})
    report = {"inputs_per_function": 1 << 32,
              "exponents": {k: f"0x{int(v.view(np.uint32)):08X} ({float(v)!r})"
                            for k, v in EXPONENTS.items()},
              "mismatches": counts, "first_mismatches": bad}
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    used = ("logf", "expf_fma", "powf_capa", "powf_rcap", "powf_two_thirds")
    undeclared = [name for name in used
                  if counts[name] != len(DECLARED.get(name, ()))
                  or {row["x_bits"] for row in bad[name]}
                  != DECLARED.get(name, set())]
    return 1 if undeclared else 0


if __name__ == "__main__":
    sys.exit(main())
