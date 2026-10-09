#!/usr/bin/env python3
"""RUC's float32 libm words against the C library gfortran links, swept.

The column oracle proves WOOF's RUC equals WRF on the arguments its columns
reach.  This sweeps the routines themselves far past those arguments, on the
same box and C library the oracle's Fortran ran against:

* the DEVICE words RUC calls (ruc.cu's ruc_tanhf_glibc / ruc_expm1f_glibc /
  ruc_expf_glibc / ruc_log10f_rn / ruc_powf_rn, glibc_cosf, gfk_log), compiled
  from the exact RUC translation unit NVRTC receives;
* the HOST mirrors in gpuwm.core.ruc (_f32_cos exhaustively over its claimed
  range; the noahmp_libm-backed ones on a sample, they are swept by their own
  lane);
* the C library's cosf/tanhf/expf/log10f/logf/powf, called through a tiny
  shared object built with -fno-builtin so every call is the library's.

Writes ``libm-sweep.json`` into OUTDIR and exits nonzero on any mismatch.
Run on a card through the mutex, under GPUWM_WRF_EXACT=1 and without it.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

C_SOURCE = r"""
#include <math.h>
void v_cosf(const float* x, float* y, long n){for(long i=0;i<n;++i) y[i]=cosf(x[i]);}
void v_tanhf(const float* x, float* y, long n){for(long i=0;i<n;++i) y[i]=tanhf(x[i]);}
void v_expf(const float* x, float* y, long n){for(long i=0;i<n;++i) y[i]=expf(x[i]);}
void v_expm1f(const float* x, float* y, long n){for(long i=0;i<n;++i) y[i]=expm1f(x[i]);}
void v_log10f(const float* x, float* y, long n){for(long i=0;i<n;++i) y[i]=log10f(x[i]);}
void v_logf(const float* x, float* y, long n){for(long i=0;i<n;++i) y[i]=logf(x[i]);}
void v_powf(const float* x, const float* e, float* y, long n){for(long i=0;i<n;++i) y[i]=powf(x[i],e[i]);}
"""

KERNEL = r"""
extern "C" __global__ void ruc_libm_sweep(const float* x, const float* e,
                                          float* y, int which, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    switch (which) {
    case 0: y[i] = glibc_cosf(v); break;
    case 1: y[i] = ruc_tanhf_glibc(v); break;
    case 2: y[i] = ruc_expf_glibc(v); break;
    case 3: y[i] = ruc_expm1f_glibc(v); break;
    case 4: y[i] = ruc_log10f_rn(v); break;
    case 5: y[i] = gfk_log(v); break;
    case 6: y[i] = ruc_powf_rn(v, e[i]); break;
    case 7: y[i] = ruc_expf_rn(v); break;
    }
}
"""

WHICH = {"cosf": 0, "tanhf": 1, "expf": 2, "expm1f": 3, "log10f": 4,
         "logf": 5, "powf": 6}


def libc():
    work = Path(tempfile.mkdtemp(prefix="ruc-libm-"))
    (work / "s.c").write_text(C_SOURCE)
    subprocess.run(["gcc", "-O1", "-fno-builtin", "-shared", "-fPIC",
                    str(work / "s.c"), "-o", str(work / "s.so"), "-lm"],
                   check=True)
    lib = ctypes.CDLL(str(work / "s.so"))
    return lib


def call_c(lib, name, x, e=None):
    y = np.empty_like(x)
    p = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    if name == "powf":
        lib.v_powf(p(x), p(e), p(y), ctypes.c_long(x.size))
    else:
        getattr(lib, "v_" + name)(p(x), p(y), ctypes.c_long(x.size))
    return y


def device_kernel():
    import cupy as cp
    from gpuwm.core.kernels import module_source
    source = module_source("ruc") + KERNEL
    module = cp.RawModule(code=source, options=("-std=c++17",))
    return module.get_function("ruc_libm_sweep")


def call_device(kernel, which, x, e=None):
    import cupy as cp
    dx = cp.asarray(x)
    de = cp.asarray(e if e is not None else x)
    dy = cp.empty_like(dx)
    n = x.size
    kernel(((n + 255) // 256,), (256,),
           (dx, de, dy, np.int32(which), np.int64(n)))
    return cp.asnumpy(dy)


def float_range(lo, hi):
    """Every float32 in [lo, hi], as (first_bits, last_bits, sign) spans."""
    def bits(v):
        return int(np.float32(v).view(np.int32))
    spans = []
    if lo < 0:
        spans.append((bits(0.0), bits(-lo), -1))
        lo = 0.0
    if hi > lo:
        spans.append((bits(lo), bits(hi), 1))
    return spans


def chunks(spans, size=1 << 25):
    """The spans' floats, generated a chunk at a time (never all at once)."""
    for first, last, sign in spans:
        for start in range(first, last + 1, size):
            stop = min(start + size, last + 1)
            x = np.arange(start, stop, dtype=np.int64).astype(np.int32)
            x = x.view(np.float32)
            yield np.ascontiguousarray(x * np.float32(sign))


def same(a, b):
    a = np.asarray(a, np.float32).view(np.uint32)
    b = np.asarray(b, np.float32).view(np.uint32)
    nan = (np.isnan(a.view(np.float32)) & np.isnan(b.view(np.float32)))
    return (a == b) | nan


#: numpy float64 stand-ins, only to pick arguments where the C library's
#: float32 word is NOT the float64 value rounded once -- the arguments that
#: discriminate a real float32 libm from a rounded one.
_ROUND_ONCE = {"cosf": np.cos, "tanhf": np.tanh, "expf": np.exp,
               "expm1f": np.expm1, "log10f": np.log10, "logf": np.log}
WORDS: dict[str, list] = {}


def _collect_words(name, x, ref, limit=256):
    bucket = WORDS.setdefault(name, [])
    if len(bucket) >= limit:
        return
    with np.errstate(all="ignore"):
        rounded = _ROUND_ONCE[name](x.astype(np.float64)).astype(np.float32)
    for i in np.nonzero(~same(rounded, ref))[0][: limit - len(bucket)]:
        bucket.append([int(x.view(np.uint32)[i]), int(ref.view(np.uint32)[i])])


def _host_chunk(job):
    """Worker: one host helper over one slice of arguments (by name)."""
    from gpuwm.core import ruc
    name, x = job
    fn = getattr(ruc, name)
    return np.array([fn(v) for v in x], dtype=np.float32)


_POOL = None


def host_all_map(host, x):
    """``host`` over every element of ``x`` on a process pool."""
    global _POOL
    import multiprocessing as mp
    if _POOL is None:
        _POOL = mp.get_context("fork").Pool(
            int(os.environ.get("RUC_SWEEP_WORKERS", "48")))
    parts = np.array_split(x, 4 * _POOL._processes)
    out = _POOL.map(_host_chunk, [(host.__name__, p) for p in parts])
    return np.concatenate(out)


def sweep_unary(lib, kernel, name, domain, record, host=None, host_all=False):
    total = bad = hbad = hcount = 0
    examples = []
    for x in chunks(domain):
        ref = call_c(lib, name, x)
        _collect_words(name, x, ref)
        dev = call_device(kernel, WHICH[name], x)
        ok = same(dev, ref)
        total += x.size
        bad += int((~ok).sum())
        for i in np.nonzero(~ok)[0][:3]:
            examples.append((float(x[i]), int(ref.view(np.uint32)[i]),
                             int(dev.view(np.uint32)[i])))
        if host is not None:
            pick = x if host_all else x[:: max(1, x.size // 4096)]
            refh = ref if host_all else ref[:: max(1, x.size // 4096)]
            got = (host_all_map(host, pick) if host_all else
                   np.array([host(v) for v in pick], dtype=np.float32))
            hcount += pick.size
            hbad += int((~same(got, refh)).sum())
    record[name] = {"arguments": total, "device_mismatches": bad,
                    "host_checked": hcount, "host_mismatches": hbad,
                    "examples": examples[:6]}
    print(f"{name}: {total} args, device mismatches {bad}, host {hbad}/{hcount}",
          flush=True)
    return bad + hbad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    from gpuwm.core import ruc
    lib = libc()
    kernel = device_kernel()
    record = {"wrf_exact": os.environ.get("GPUWM_WRF_EXACT", "0"),
              "libc": subprocess.run(["ldd", "--version"], capture_output=True,
                                     text=True).stdout.splitlines()[0]}
    bad = 0
    pi = np.float32(3.141592653589793)
    # SOILRES: cos(piconst*fex), fex in [0.01, 1]: every float32 in range.
    bad += sweep_unary(lib, kernel, "cosf",
                       float_range(float(np.float32(pi * np.float32(0.01))),
                                   float(pi)),
                       record, host=ruc._f32_cos, host_all=not args.quick)
    span = 1.0 if args.quick else 12.0
    bad += sweep_unary(lib, kernel, "tanhf", float_range(-span, span), record,
                       host=ruc._f32_tanh)
    bad += sweep_unary(lib, kernel, "expm1f", float_range(-span, span), record,
                       host=ruc._f32_expm1)
    bad += sweep_unary(lib, kernel, "expf",
                       float_range(-2.0 if args.quick else -40.0,
                                   2.0 if args.quick else 40.0),
                       record, host=ruc._f32_exp)
    bad += sweep_unary(lib, kernel, "log10f",
                       float_range(1.0e-3, 10.0 if args.quick else 1.0e8),
                       record, host=ruc._f32_log10)
    bad += sweep_unary(lib, kernel, "logf", float_range(0.5, 2.0), record,
                       host=ruc._f32_log)
    # REAL**REAL: the RUC shapes -- freezing curve base**(-1/b) with b in
    # [2.7, 11.6], conductivity k**w with w in [0, 1], ratios**3 and **(2b+2),
    # Exner (p0/p)**0.2857, canopy (cst/sat)**cn.
    rng = np.random.default_rng(20261007)
    n = 1 << (20 if args.quick else 26)
    base = np.concatenate([
        rng.uniform(1.0, 3.0e4, n), rng.uniform(1.0e-3, 1.0, n),
        rng.uniform(0.5, 9.0, n)]).astype(np.float32)
    expo = np.concatenate([
        -1.0 / rng.uniform(2.7, 11.6, n), rng.uniform(0.0, 1.0, n),
        rng.choice([3.0, 0.2857143, 2.0], n)]).astype(np.float32)
    total = pbad = hbad = 0
    for start in range(0, base.size, 1 << 25):
        x = np.ascontiguousarray(base[start:start + (1 << 25)])
        e = np.ascontiguousarray(expo[start:start + (1 << 25)])
        ref = call_c(lib, "powf", x, e)
        dev = call_device(kernel, WHICH["powf"], x, e)
        pbad += int((~same(dev, ref)).sum())
        total += x.size
        k = slice(None, None, max(1, x.size // 4096))
        got = np.array([ruc._f32_pow(a, b) for a, b in zip(x[k], e[k])],
                       np.float32)
        hbad += int((~same(got, ref[k])).sum())
    record["powf"] = {"arguments": total, "device_mismatches": pbad,
                      "host_mismatches": hbad}
    print(f"powf: {total} pairs, device mismatches {pbad}, host {hbad}", flush=True)
    bad += pbad + hbad
    probe = np.float32(0.9974991083145142)
    record["tanhf_probe"] = {
        "x": float(probe),
        "libc": float(call_c(lib, "tanhf", np.array([probe]))[0]),
        "device": float(call_device(kernel, 1, np.array([probe]))[0]),
        "host": float(ruc._f32_tanh(probe))}
    WORDS["tanhf"].append([int(probe.view(np.uint32)),
                           int(np.float32(record["tanhf_probe"]["libc"]).view(np.uint32))])
    k = slice(None, None, max(1, base.size // 256))
    WORDS["powf"] = [[int(a), int(b), int(c)] for a, b, c in zip(
        base[k].view(np.uint32), expo[k].view(np.uint32),
        call_c(lib, "powf", np.ascontiguousarray(base[k]),
               np.ascontiguousarray(expo[k])).view(np.uint32))]
    (args.outdir / "libm-words.json").write_text(json.dumps(
        {"libc": record["libc"],
         "note": "[x_bits, libc_bits] (powf: [x, y, libc]); unary arguments "
                 "are ones where the C library's float32 word is not the "
                 "float64 value rounded once",
         "words": WORDS}) + chr(10))
    record["total_mismatches"] = bad
    (args.outdir / f"libm-sweep-exact{record['wrf_exact']}.json").write_text(
        json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["tanhf_probe"]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
