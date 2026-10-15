"""Dump one column's KF tendencies from whichever kernel the tree on
PYTHONPATH carries, for the before/after evidence chart.

    python dump_profiles.py COLUMNS.bin INDEX OUT.npz {port|preport}

``port`` runs the current API (``advance_w0avg`` + ``launch_kf``);
``preport`` runs the replaced kernel's API exactly as its adapter did (see
compare_baseline.py).  Both write the six tendencies and PRATEC.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

try:
    from . import kf_columns as K
except ImportError:                       # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import kf_columns as K  # noqa: E402

NAMES = ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten",
         "rqscuten")


def main(columns_path, index, out_path, which):
    import cupy as cp
    from gpuwm import wrf_exact
    wrf_exact.install()
    from gpuwm.core import kf

    col = K.read_columns(columns_path)[int(index)]
    nz, warm, fqr, fqi, fqs, stepcu, adapt, _ = col["ints"]
    dt, dx, cudt = (float(x) for x in col["scal"][:3])
    mode = (kf.KFPhaseMode.WARM_RAIN if warm else
            kf.KFPhaseMode.NO_SEPARATE_SNOW if not fqs else
            kf.KFPhaseMode.SEPARATE_SNOW if not fqi else
            kf.KFPhaseMode.SEPARATE_ICE_SNOW)

    def dev(name, levels=nz):
        return cp.asarray(col[name].reshape(levels, 1, 1).astype(np.float32))

    w = dev("w", nz + 1)
    w0 = dev("w0avg")
    if which == "port":
        kf.advance_w0avg(w0, w, factors=kf.w0avg_factors(
            cudt_minutes=cudt, dt=dt, stepcu=stepcu, adaptive=bool(adapt)))
        res = kf.launch_kf(dev("u"), dev("v"), dev("t"), dev("qv"), dev("p"),
                           dev("rho"), dev("pi"), dev("dz"), w0, dx=dx,
                           dt=dt, cudt_minutes=cudt, phase_mode=mode,
                           f_qr=bool(fqr))
        pratec = float(res["pratec"][0, 0])
    else:
        tst = np.float32(2 * stepcu)
        inst = cp.ascontiguousarray(np.float32(0.5) * (w[:-1] + w[1:]))
        w0[...] = (w0 * (tst - np.float32(1.0)) + inst) / tst
        steps = 1 if cudt <= 0 else max(int(math.floor(cudt * 60.0 / dt + 0.5)), 1)
        cudt_s = dt if cudt <= 0 else steps * dt
        res = kf.launch_kf(dev("u"), dev("v"), dev("t"), dev("qv"),
                           cp.zeros_like(w0), dev("p"), dev("pi"), dev("dz"),
                           w0, dx=dx, dt=dt, cudt=cudt_s, phase_mode=mode)
        nca = float(res["nca_seconds"][0, 0])
        div = min(cudt_s, nca)
        pratec = float(res["rainc"][0, 0]) / div if div > 0 else 0.0
    out = {name: cp.asnumpy(res[name])[:, 0, 0] for name in NAMES}
    out["pratec"] = np.float32(pratec)
    out["p"] = col["p"]
    np.savez(out_path, **out)
    print(which, "saved", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:5])
