"""Compare WRF's GS variable dump (instrument_gs.py) with WOOF's (gs_rates_probe.py).

  gs_rates_compare.py WRF_GS.bin NAMES.json WOOF.npz COLUMNS [--first N] [--show NAME ...]

Lists, for every shared variable, how many gather points differ bitwise, the
max ULP and the first differing (column, level), and per point the first
variable (in WOOF's declaration order, which follows WRF's source order)
that differs: that is where to look.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_bitwise import ulp


def read_gs(path: Path):
    raw = path.read_bytes()
    off, recs = 0, []
    while off < len(raw):
        col, i, k, n = np.frombuffer(raw, "<i4", 4, off)
        off += 16
        v = np.frombuffer(raw, "<f4", n, off)
        off += 4 * n
        recs.append((int(col) - 1, int(k) - 1, v))
    return recs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wrf", type=Path)
    p.add_argument("names", type=Path)
    p.add_argument("woof", type=Path)
    p.add_argument("columns", type=Path)
    p.add_argument("--show", nargs="*", default=[])
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--ignore-zero-sign", action="store_true")
    a = p.parse_args()
    keys = json.loads(a.names.read_text())["names"]
    recs = read_gs(a.wrf)
    regime = np.load(a.columns)["regime"]
    woof = np.load(a.woof)
    cols = np.array([r[0] for r in recs])
    levs = np.array([r[1] for r in recs])
    W = np.stack([r[2] for r in recs])
    G = np.stack([woof[k][cols, levs] for k in keys], axis=1).astype(np.float32)
    gath = woof["__gathered"]
    print(f"{len(recs)} WRF gather points; WOOF gathered {int((gath > 0).sum())}; "
          f"WRF points WOOF did not gather: {int((gath[cols, levs] <= 0).sum())}")
    diff = W.view(np.uint32) != G.view(np.uint32)
    if a.ignore_zero_sign:
        diff &= ~((W == 0) & (G == 0))
    print(f"{int(diff.any(axis=1).sum())} points with any differing variable")
    firsts = {}
    for p_ in np.nonzero(diff.any(axis=1))[0]:
        j = int(np.nonzero(diff[p_])[0][0])
        firsts[keys[j]] = firsts.get(keys[j], 0) + 1
    print("first differing variable per point:",
          dict(sorted(firsts.items(), key=lambda t: -t[1])))
    for j, k in enumerate(keys):
        d = diff[:, j]
        if not d.any():
            continue
        u = ulp(W[:, j], G[:, j])
        p0 = int(np.nonzero(d)[0][0])
        print(f"  {k:12s} n={int(d.sum()):5d} max_ulp={int(u[d].max()):<11d} "
              f"first col={cols[p0]} k={levs[p0] + 1} ({regime[cols[p0]]}) "
              f"wrf={W[p0, j]!r} woof={G[p0, j]!r}")
    for k in a.show:
        j = keys.index(k)
        for p_ in np.nonzero(diff[:, j])[0][:15]:
            print(f"   {k} col={cols[p_]} k={levs[p_] + 1} wrf={W[p_, j]!r} woof={G[p_, j]!r}")
    shown = 0
    for p_ in np.nonzero(diff.any(axis=1))[0]:
        if shown >= a.first:
            break
        shown += 1
        print(f"-- col={cols[p_]} k={levs[p_] + 1} ({regime[cols[p_]]})")
        for j in np.nonzero(diff[p_])[0]:
            print(f"     {keys[j]:12s} wrf={W[p_, j]!r} woof={G[p_, j]!r}")


if __name__ == "__main__":
    main()
