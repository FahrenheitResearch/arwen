#!/usr/bin/env python3
"""CTOPO, CTOPO2 and LAP_HGT from a real wrf.exe against WOOF's, word for word.

    python tools/topo_wind_wrf461_oracle/wrfexe_static.py WRFOUT.nc [--json OUT]

WRFOUT is a WRF v4.6.1 history file written at the initial time by a run
with topo_wind = 1 or 2 and an iofields file adding ``ctopo``, ``ctopo2``
and ``lap_hgt`` to history (``+:h:0:ctopo,ctopo2,lap_hgt``), converted to
NetCDF-3 (``nccopy -k '64-bit offset'``) so SciPy can read it.  The script
reads the run's own HGT, XLAND, VAR_SSO and VAR from the same file, forms
WOOF's coefficients with :func:`gpuwm.core.terrain_drag.topo_wind_coefficients`
on the GPU, and compares all three planes bitwise over the whole domain.

This is the stock executable, not a column cut: dyn_em/start_em.F compiled
by WRF's own build (-O2 -ftree-vectorize -funroll-loops, gfortran 13.3,
glibc 2.39), on a real 425 x 300 CONUS domain.  A multi-rank run computes
each tile's laplacian from its halo, so its LAP_HGT shows what WRF's
decomposition does at tile edges; WOOF's single-domain answer is the
serial run's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wrfout")
    ap.add_argument("--json")
    ap.add_argument("--topo-wind", type=int, choices=(1, 2),
                    help="the run's topo_wind; WRF writes the TOPO_WIND "
                         "global attribute only for 1 (output_wrf.F:733)")
    args = ap.parse_args(argv)

    import cupy as cp
    from scipy.io import netcdf_file

    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    from gpuwm.core.terrain_drag import topo_wind_coefficients

    nc = netcdf_file(args.wrfout, "r", mmap=False)
    opt = args.topo_wind if args.topo_wind else int(nc.TOPO_WIND)
    get = lambda n: np.ascontiguousarray(  # noqa: E731
        np.asarray(nc.variables[n][0], dtype=np.float32))
    ht, xland, var_sso, var2d = (get("HGT"), get("XLAND"), get("VAR_SSO"),
                                 get("VAR"))
    ctopo, ctopo2, lap = topo_wind_coefficients(
        cp.asarray(ht), cp.asarray(xland), topo_wind=opt,
        var_sso=cp.asarray(var_sso), var2d=cp.asarray(var2d))
    got = {"CTOPO": cp.asnumpy(ctopo), "CTOPO2": cp.asnumpy(ctopo2),
           "LAP_HGT": cp.asnumpy(lap)}
    report = {"file": args.wrfout, "topo_wind": opt, "shape": list(ht.shape),
              "fields": {}}
    print(f"{args.wrfout}: topo_wind = {opt}, {ht.shape[1]} x {ht.shape[0]}"
          f" = {ht.size} columns")
    for name, g in got.items():
        w = get(name)
        diff = g.view(np.uint32) != w.view(np.uint32)
        ulp = np.asarray(fp32_ulp_distance(g, w))
        n = int(diff.sum())
        if n:
            idx = np.argwhere(diff)
            print(f"  {name:<8} differing {n}/{g.size}, max ULP "
                  f"{int(ulp.max())}, first {idx[:5].tolist()}")
        else:
            print(f"  {name:<8} bitwise equal on all {g.size} columns")
        report["fields"][name] = {"differing": n, "total": int(g.size),
                                  "max_ulp": int(ulp.max())}
    c = got["CTOPO"]
    report["ctopo_range"] = [float(c.min()), float(c.max())]
    report["ctopo_not_one"] = int((c != 1).sum())
    report["ctopo2_below_one"] = int((got["CTOPO2"] < 1).sum())
    print(f"  CTOPO range {c.min()} .. {c.max()}, != 1 on "
          f"{report['ctopo_not_one']} columns; CTOPO2 < 1 on "
          f"{report['ctopo2_below_one']}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
