"""WOOF side of the Milbrandt-Yau (mp_physics=9) column oracle.

    python woof_run.py IN.bin OUT.bin

Runs the shipped CUDA kernels through the shipped launcher
``gpuwm.core.milbrandt2.launch_milbrandt2`` on the same float32 inputs the
Fortran runner reads, NSTEPS calls in a row, and writes the same output
stream ``run_my2.F90`` writes.  The two wrapper statements around the
launcher are the ones ``gpuwm.core.milbrandt2.apply`` itself executes:
``temperature = theta * pii`` before the call (WRF :3644) and
``theta = temperature / pii`` after it (:3671).  ``psfc`` is WRF's
``p8w(i,kms,j)`` (:3646), taken from the input as the Fortran takes it.

Set GPUWM_WRF_EXACT=1 in the environment for the strict build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def run(in_path, out_path, layout=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cupy as cp
    import gpuwm  # noqa: F401  (installs the strict-arithmetic hook first)
    from gen_columns import read_stream
    from compare import write_output, LEVEL_FIELDS, SURFACE_FIELDS
    from gpuwm.core.milbrandt2 import launch_milbrandt2

    ncol, nz, nsteps, dt, arr, sfc = read_stream(in_path)
    ny, nx = layout if layout else (1, ncol)
    assert ny * nx == ncol

    def dev3(a):  # (ncol, nz) -> (nz, ny, nx) C-contiguous float32
        return cp.ascontiguousarray(
            cp.asarray(np.ascontiguousarray(a.T).reshape(nz, ny, nx)))

    def dev2(a):
        return cp.ascontiguousarray(cp.asarray(a.reshape(ny, nx)))

    f = {k: dev3(arr[k]) for k in
         ("qv", "qc", "qr", "qi", "qs", "qg", "qh",
          "nc", "nr", "ni", "ns", "ng", "nh", "th", "pii", "p", "w")}
    psfc = dev2(sfc["p8w"])
    acc = {k: dev2(sfc[k]) for k in ("rainnc", "snownc", "grplnc", "hailnc")}
    zero2 = lambda: cp.zeros((ny, nx), cp.float32)  # noqa: E731
    rainncv, snowncv, grplncv, hailncv, sr = (zero2() for _ in range(5))
    zet = cp.zeros((nz, ny, nx), cp.float32)
    scratch = {k: cp.zeros((nz, ny, nx), cp.float32) for k in
               ("pres", "de", "ide", "dz", "idz", "gamfact", "qsw", "qsi",
                "qc_in", "qr_in", "nc_in", "nr_in")}
    temperature = cp.empty((nz, ny, nx), cp.float32)

    out = {k: np.empty((nsteps, ncol, nz), np.float32) for k in LEVEL_FIELDS}
    out.update({k: np.empty((nsteps, ncol), np.float32)
                for k in SURFACE_FIELDS})

    def host3(a):
        return cp.asnumpy(a).reshape(nz, ncol).T

    def host2(a):
        return cp.asnumpy(a).reshape(ncol)

    for s in range(nsteps):
        cp.multiply(f["th"], f["pii"], out=temperature)          # :3644
        launch_milbrandt2(
            temperature, f["qv"], f["qc"], f["qr"], f["qi"], f["qs"],
            f["qg"], f["qh"], f["nc"], f["nr"], f["ni"], f["ns"], f["ng"],
            f["nh"], f["w"], f["p"], psfc, float(dt),
            acc["rainnc"], rainncv, acc["snownc"], snowncv,
            acc["grplnc"], grplncv, acc["hailnc"], hailncv, sr, zet,
            **scratch)
        cp.divide(temperature, f["pii"], out=f["th"])             # :3671
        for k in ("qv", "qc", "qr", "qi", "qs", "qg", "qh",
                  "nc", "nr", "ni", "ns", "ng", "nh", "th"):
            out[k][s] = host3(f[k])
        out["zet"][s] = host3(zet)
        out["rainnc"][s] = host2(acc["rainnc"])
        out["rainncv"][s] = host2(rainncv)
        out["snownc"][s] = host2(acc["snownc"])
        out["snowncv"][s] = host2(snowncv)
        out["grplnc"][s] = host2(acc["grplnc"])
        out["grplncv"][s] = host2(grplncv)
        out["hailnc"][s] = host2(acc["hailnc"])
        out["hailncv"][s] = host2(hailncv)
        out["sr"][s] = host2(sr)
    write_output(out_path, out)
    return out


if __name__ == "__main__":
    layout = None
    if len(sys.argv) > 3:
        layout = tuple(int(v) for v in sys.argv[3].split("x"))
    run(sys.argv[1], sys.argv[2], layout)
