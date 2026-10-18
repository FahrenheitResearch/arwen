#!/usr/bin/env python3
"""Hold Shin-Hong's tanhf to the reference platform's, float by float.

    python tools/topo_wind_wrf461_oracle/sh_tanhf_sweep.py [--lo -10] [--hi 10]

module_bl_shinhong.F:847 forms ``csfac = 0.5*(tanh(uwstx)+3.)`` with
``uwstx = -80*|ust/wstar - 0.5| + 14``, so the argument runs from 14 down;
tanhf is exactly +-1 beyond about 9.01.  This sweep evaluates the Shin-Hong
unit's own ``sh_tanhf`` through its probe kernel (``shinhong_tanhf_probe``,
compiled exactly as the column kernel is) on the GPU and the C library's
``tanhf`` on the CPU for EVERY float32 in [lo, hi], and counts the words
that differ.  Run it on the reference platform (box W1: Ubuntu 24.04, glibc
2.39).  The C-library and correctly-rounded helpers are the YSU sweep's
(tools/ysu_wrf461_oracle/tanhf_sweep.py).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _ysu_sweep():
    path = ROOT / "tools" / "ysu_wrf461_oracle" / "tanhf_sweep.py"
    spec = importlib.util.spec_from_file_location("_ysu_tanhf_sweep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=-10.0)
    ap.add_argument("--hi", type=float, default=10.0)
    ap.add_argument("--chunk", type=int, default=1 << 26)
    args = ap.parse_args(argv)

    from gpuwm.core.kernels import get_kernel

    ysu = _ysu_sweep()
    lib = ysu._clib()
    fn = get_kernel("shinhong", "shinhong_tanhf_probe")
    total = kernel_vs_libm = libm_vs_cr = 0
    first = []
    for start, stop in ysu.all_floats(args.lo, args.hi):
        b = int(start)
        while b <= int(stop):
            e = min(b + args.chunk, int(stop) + 1)
            x = np.arange(b, e, dtype=np.uint32).view(np.float32)
            got = ysu.gpu_tanhf(fn, x)
            ref = ysu._call(lib, "libm_tanhf", x)
            cr = ysu._call(lib, "cr_tanhf", x)
            gv, rv = got.view(np.uint32), ref.view(np.uint32)
            d = gv != rv
            kernel_vs_libm += int(d.sum())
            libm_vs_cr += int((rv != cr.view(np.uint32)).sum())
            if d.any() and len(first) < 8:
                for i in np.flatnonzero(d)[:8 - len(first)]:
                    first.append((float(x[i]).hex(), float(got[i]).hex(),
                                  float(ref[i]).hex()))
            total += x.size
            b = e
    libc = os.confstr("CS_GNU_LIBC_VERSION") if hasattr(os, "confstr") else "?"
    print(f"C library: {libc}")
    print(f"floats checked in [{args.lo}, {args.hi}]: {total}")
    print(f"sh_tanhf != library tanhf: {kernel_vs_libm}")
    print(f"library tanhf != correctly rounded: {libm_vs_cr}")
    for row in first:
        print("  differs:", row)
    return 0 if kernel_vs_libm == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
