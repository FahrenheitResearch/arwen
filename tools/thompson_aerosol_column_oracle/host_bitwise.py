#!/usr/bin/env python3
"""Bitwise view of a ``real_column_parity.py --dump`` run: every one of the
sixty-four process rates (float64 in both codes) and every final output
(float32), as counts of differing cells and the largest ULP distance.

This locates a GPU-oracle miss without the GPU: the host build of the
kernels runs with IEEE per-operation rounding, no FMA and the oracle host's
scalar libm, so a rate that differs here differs because of the port's
operation order or logic, not its device arithmetic.

usage: host_bitwise.py DUMP_DIR [--col-labels COLUMNS.npz] [--show RATE]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_io as io  # noqa: E402


#: The rates the condensation and rain-evaporation blocks own, published at
#: WRF's second checkpoint.
LATE = ("prw_vcd", "pnc_wcd", "prv_rev", "pnr_rev")


def ulp64(a, b):
    def o(x):
        v = np.asarray(x, np.float64).view(np.int64)
        return np.where(v < 0, -(v & 0x7FFFFFFFFFFFFFFF), v).astype(object)
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = np.abs(a.view(np.int64).astype(np.float64)
               - b.view(np.int64).astype(np.float64))
    same = a.view(np.int64) == b.view(np.int64)
    return np.where(same, 0.0, d)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--cols", default=None)
    ap.add_argument("--show", action="append", default=[])
    a = ap.parse_args(argv)
    z = np.load(Path(a.dump) / "arrays.npz")
    labels = None
    if a.cols:
        labels = np.load(a.cols)["labels"]
    present = z["cp1_present"]
    print("RATES (float64; cells where either side is non-zero)")
    for k in sorted(x for x in z.files if x.startswith("portrate_")):
        name = k[len("portrate_"):]
        cp = "cp2_" if name in LATE else "cp1_"
        if cp + name not in z.files:
            continue
        here = z[cp + "present"]
        p = z[k][here]
        w = z[cp + name][here]
        act = (p != 0) | (w != 0)
        same = p.view(np.int64) == w.view(np.int64)
        bad = act & ~same & np.isfinite(w)
        n = int(bad.sum())
        if n == 0 and name not in a.show:
            print(f"  {name:9s} exact on {int(act.sum())} active cells")
            continue
        rel = np.abs(p - w) / np.maximum(np.abs(w), 1e-300)
        u = ulp64(p, w)
        print(f"  {name:9s} DIFFER {n:5d}/{int(act.sum()):5d}  max rel "
              f"{rel[bad].max() if n else 0:.3e}  max ulp64 "
              f"{u[bad].max() if n else 0:.3e}")
        if name in a.show and n:
            cols = np.flatnonzero(here)
            idx = np.argwhere(bad)
            for c, lev in idx[:12]:
                lab = labels[cols[c]] if labels is not None else cols[c]
                print(f"      col {cols[c]} '{lab}' lev {lev}: port "
                      f"{p[c, lev]!r} wrf {w[c, lev]!r}")
    print("FINAL (float32 bitwise)")
    for name in io.FIELDS:
        if "portfinal_" + name not in z.files:
            continue
        res, bad = io.bitwise(z["portfinal_" + name],
                              io.wrf_view(name, z["wrf_" + name]))
        print(f"  {name:10s} differ {res['differ']:5d}/{res['cells']:5d} "
              f"max ulp {res['max_ulp']}")


if __name__ == "__main__":
    main()
