"""Spread ratios with a set of columns masked out, at chosen frames.

    python mask_ratio.py WRF_SCALAR_DIR WRF_VECTOR_DIR CAND_DIR OUT.json
                         [--exclude-lu 15] [--frames 1,15]

The interior (5-cell rim excluded) RMS of (cand - WRF scalar) over the RMS of
(WRF vectorised - WRF scalar), as in summarize.py, but over the columns whose
LU_INDEX is not in --exclude-lu.  Used to separate a known, attributed
column class (the land-use category whose surface physics differs) from the
rest of the domain.  3-D fields are masked by column.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import frames, RIM  # noqa: E402

FIELDS = ("W", "T", "PH", "MU", "P", "QVAPOR", "QCLOUD", "QRAIN",
          "RAINC", "RAINNC", "T2", "Q2", "U10", "V10", "PSFC", "HFX", "LH",
          "TSK", "SWDOWN", "GLW", "OLR", "PBLH", "UST")


def _get(index, t, name):
    from netCDF4 import Dataset
    path, i = index[t]
    with Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        v = ds.variables[name]
        return np.asarray(v[i] if v.dimensions[0] == "Time" else v[:], np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scalar"); ap.add_argument("vector"); ap.add_argument("cand")
    ap.add_argument("out")
    ap.add_argument("--exclude-lu", default="15")
    ap.add_argument("--frames", default="1,-1")
    a = ap.parse_args()
    rs, rv, rc = frames(a.scalar), frames(a.vector), frames(a.cand)
    times = sorted(set(rs) & set(rv) & set(rc))
    lu = _get(rs, times[0], "LU_INDEX")
    keep = ~np.isin(lu, [float(x) for x in a.exclude_lu.split(",")])
    keep = keep[RIM:-RIM, RIM:-RIM]
    out = {"excluded_lu": a.exclude_lu, "columns_kept": int(keep.sum()),
           "columns_excluded": int((~keep).sum()), "frames": {}}
    for f in (int(x) for x in a.frames.split(",")):
        t = times[f]
        row = {}
        for name in FIELDS:
            try:
                s = _get(rs, t, name); v = _get(rv, t, name); c = _get(rc, t, name)
            except KeyError:
                continue
            if s.shape != c.shape or s.ndim < 2:
                continue
            ds = (c - s)[..., RIM:-RIM, RIM:-RIM]; dv = (v - s)[..., RIM:-RIM, RIM:-RIM]
            if ds.shape[-2:] != keep.shape:      # U/V staggered: mask the mass grid only
                continue
            m = np.broadcast_to(keep, ds.shape)
            woof = float(np.sqrt(np.mean(ds[m] ** 2))); wrf = float(np.sqrt(np.mean(dv[m] ** 2)))
            row[name] = {"woof_rms": woof, "wrf_spread_rms": wrf,
                         "ratio": (woof / wrf) if wrf else (0.0 if woof == 0 else None)}
        out["frames"][t] = row
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for t, row in out["frames"].items():
        print(t, " ".join(f"{k}:{v['ratio']:.1f}" if v["ratio"] is not None else f"{k}:inf"
                          for k, v in row.items()))


if __name__ == "__main__":
    main()
