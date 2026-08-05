#!/usr/bin/env python3
"""Where does a candidate moist sounding put its LCL?  (P1, ideal.exe only.)

A moist LES case that never saturates verifies nothing about the saturated
half of the closure, and whether a sounding saturates is decided by its LCL
against the CBL depth the forcing actually builds.  This reads a `wrfinput_d01`
-- i.e. WRF's own base state, produced by WRF's own ideal.exe, including its
moisture correction -- and reports the LCL under WRF's own saturation formula.

It costs one ideal.exe (about a second) per candidate, so a sounding can be
screened before anything is integrated.  The CBL depth it is compared against
is a MEASURED number from a committed dry receipt, never an estimate.

Usage: lcl_probe.py <wrfinput_d01> [--zi 1695] [--json out.json]
"""
import sys
import json

import numpy as np
from netCDF4 import Dataset

# WRF v4.6.1 share/module_model_constants.F
G = 9.81
T0 = 300.0
R_D = 287.0
R_V = 461.6
CP = 1004.5
P1000MB = 1.0e5
SVP1, SVP2, SVP3, SVPT0 = 0.6112, 17.67, 29.65, 273.15
EP_2 = R_D / R_V
RCP = R_D / CP


def qvs_wrf(t_k, p_pa):
    """dyn_em/module_diffusion_em.F:1626-1631, verbatim."""
    tc = t_k - SVPT0
    es = 1000.0 * SVP1 * np.exp(SVP2 * tc / (t_k - SVP3))
    return EP_2 * es / (p_pa - es)


def main():
    path = sys.argv[1]
    zi = None
    if "--zi" in sys.argv:
        zi = float(sys.argv[sys.argv.index("--zi") + 1])

    d = Dataset(path)
    th = d.variables["T"][0].astype(np.float64) + T0
    p = (d.variables["P"][0] + d.variables["PB"][0]).astype(np.float64)
    ph = (d.variables["PH"][0] + d.variables["PHB"][0]).astype(np.float64)
    qv = (d.variables["QVAPOR"][0].astype(np.float64)
          if "QVAPOR" in d.variables else np.zeros_like(th))
    d.close()

    zw = (ph / G).mean(axis=(1, 2))
    zm = 0.5 * (zw[:-1] + zw[1:])
    th_b = th.mean(axis=(1, 2))
    p_b = p.mean(axis=(1, 2))
    qv_b = qv.mean(axis=(1, 2))

    # Parcel: the lowest model level's slab-mean theta and qv, lifted dry
    # adiabatically along the model's own pressure profile.
    th_p, qv_p = float(th_b[0]), float(qv_b[0])
    t_par = th_p * (p_b / P1000MB) ** RCP
    qvs_par = qvs_wrf(t_par, p_b)
    hit = np.where(qv_p >= qvs_par)[0]
    lcl = float(zm[hit[0]]) if len(hit) else None

    # In-place saturation of the initial column (a sounding can be saturated
    # where it stands, which is a different statement from "a surface parcel
    # would saturate if lifted").
    t_env = th_b * (p_b / P1000MB) ** RCP
    qvs_env = qvs_wrf(t_env, p_b)
    rh = qv_b / np.maximum(qvs_env, 1e-30)

    out = dict(
        wrfinput=path,
        nz=int(len(zm)), ztop_m=float(zw[-1]),
        parcel_theta_K=th_p, parcel_qv_kg_kg=qv_p,
        parcel_qv_g_kg=qv_p * 1000.0,
        lcl_m=lcl,
        lcl_note=("the surface parcel does not saturate anywhere below the "
                  "model top" if lcl is None else "first level at which "
                  "qv_parcel >= qvs(T_parcel, p)"),
        qvs_min_in_column_g_kg=float(qvs_par.min() * 1000.0),
        rh_max_initial=float(rh.max()),
        rh_surface_initial=float(rh[0]),
    )
    if zi is not None:
        out["zi_reference_m"] = zi
        out["lcl_over_zi"] = (lcl / zi) if lcl is not None else None
        out["verdict"] = (
            "LCL above the model top: no condensation is reachable"
            if lcl is None else
            ("LCL inside the measured CBL depth" if lcl <= zi else
             "LCL above the measured CBL depth: the CBL must deepen past it, "
             "and warming raises the LCL while it does"))
    for k, v in out.items():
        print("%-26s %s" % (k, v))
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w",
                  newline="\n") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
