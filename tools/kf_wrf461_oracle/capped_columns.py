"""Write the hex capped-downdraft KF columns in the oracle's column format.

tests/data/kf_downdraft_takes_all_rain_columns.npz holds the seven deep
columns a global 60 km hex forecast (GFS 2026-10-10 00Z, 55 levels, dt
360 s) handed Kain-Fritsch on its first three calls, where the capped
downdraft took all the precipitation (d547dc8c3).  It was recorded for the
pre-port kernel's argument list (u, v, T, qv, qc, p, PI, dz, W0AVG), so this
writes them as KF_eta_CPS inputs: RHO is WRF's (1+qv)/alt from the moist
equation of state, and full-level w is built so that its level means equal
the recorded W0AVG, with STEPCU = 1 (the running mean then returns the
recorded W0AVG).

    python capped_columns.py OUT.bin [--npz PATH]

Run the Fortran oracle (build.sh) and the GPU path (compare_woof.py,
floor_probe.py --columns) on OUT.bin.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    from . import kf_columns as K
except ImportError:                       # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import kf_columns as K  # noqa: E402

DEFAULT_NPZ = (Path(__file__).resolve().parents[2] / "tests" / "data"
               / "kf_downdraft_takes_all_rain_columns.npz")
#: KFPhaseMode -> (warm_rain, f_qr, f_qi, f_qs), the kf_columns BRANCHES rows.
PHASE_BRANCH = {0: (1, 1, 0, 0), 1: (0, 1, 0, 0), 2: (0, 1, 0, 1),
                3: (0, 1, 1, 1)}


def columns(npz_path=DEFAULT_NPZ):
    d = np.load(npz_path)
    f32 = np.float32
    nz, ncol = d["temperature"].shape
    warm, fqr, fqi, fqs = PHASE_BRANCH[int(d["phase_mode"])]
    dt, dx = float(d["dt"]), float(d["dx"])
    out = []
    for j in range(ncol):
        t = d["temperature"][:, j].astype(np.float64)
        qv = d["qv"][:, j].astype(np.float64)
        p = d["pressure"][:, j].astype(np.float64)
        rho = p / (K.RD * t * (1.0 + qv * K.RV / K.RD)) * (1.0 + qv)
        w0 = d["w"][:, j].astype(f32)
        w = np.zeros(nz + 1, f32)
        for k in range(nz):
            w[k + 1] = f32(f32(2.0) * w0[k] - w[k])
        out.append(dict(
            ints=(nz, warm, fqr, fqi, fqs, 1, 0, 1),
            scal=tuple(f32(x) for x in (dt, dx, 0.0, -100.0)),
            u=d["u"][:, j].astype(f32), v=d["v"][:, j].astype(f32),
            t=t.astype(f32), qv=qv.astype(f32), p=p.astype(f32),
            dz=d["dz"][:, j].astype(f32), rho=rho.astype(f32),
            pi=d["exner"][:, j].astype(f32), w0avg=w0, w=w,
            meta=dict(regime="hex-capped-downdraft", branch="qs_qi", nz=nz,
                      dx=dx, dt=dt, cudt=0.0, stepcu=1,
                      mesh_column=int(d["mesh_column"][j]),
                      call=int(d["call"][j]))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--npz", default=str(DEFAULT_NPZ))
    args = ap.parse_args(argv)
    K.write_columns(args.out, columns(args.npz))


if __name__ == "__main__":
    main()
