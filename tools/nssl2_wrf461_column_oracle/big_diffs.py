"""List cells whose WRF/WOOF relative difference exceeds a threshold
(default 1e-3): differences too large to be rounding, i.e. where to look
for a logic difference rather than an arithmetic one.

  big_diffs.py PREFIX [--stage end|gs|nucond|diag] [--rel 1e-3]
"""
import argparse
from pathlib import Path

import numpy as np

from compare_bitwise import OUT3, read_fortran
from compare_stages import read_dumps

p = argparse.ArgumentParser()
p.add_argument("prefix", type=Path)
p.add_argument("--stage", default="end")
p.add_argument("--rel", type=float, default=1e-3)
p.add_argument("--examples", type=int, default=3)
a = p.parse_args()
base = a.prefix.parent
reg = np.load(base / "columns.npz")["regime"]
ncol = reg.size
nz = np.load(base / "columns.npz")["theta"].shape[1]
f = lambda s: a.prefix.with_name(a.prefix.name + s)
if a.stage == "end":
    want = read_fortran(f(".wrf_scalar.bin"), ncol, nz)
    got = dict(np.load(f(".gpu.npz")))
else:
    iso = np.load(f(".isolated.npz"))
    key = {"gs": "gs", "nucond": "nucond", "diag": "radar/radii/scatter"}[a.stage]
    got = {k.split("__", 1)[1]: iso[k] for k in iso.files if k.startswith(key + "__")}
    if a.stage == "diag":
        want = read_fortran(f(".wrf_scalar.bin"), ncol, nz)
    else:
        want = read_dumps(f(".wrf_stages.bin"), ncol)[{"gs": 2, "nucond": 3}[a.stage]]
for n in got:
    if n not in want or got[n].ndim != 2:
        continue
    x = want[n].astype(np.float64); y = got[n].astype(np.float64)
    den = np.maximum(np.abs(x), np.abs(y))
    rel = np.where(den > 0, np.abs(x - y) / np.where(den > 0, den, 1), 0)
    big = rel > a.rel
    if big.any():
        idx = np.argwhere(big)
        regs = {}
        for c, k in idx:
            regs[str(reg[c])] = regs.get(str(reg[c]), 0) + 1
        print(f"{n:9s} rel>{a.rel:g}: {big.sum():4d} {regs}")
        for c, k in idx[:a.examples]:
            print(f"      col {c} k {k+1} {reg[c]} wrf {x[c,k]:.9g} woof {y[c,k]:.9g}")
