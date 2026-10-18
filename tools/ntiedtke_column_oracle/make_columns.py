"""Build a New Tiedtke column-oracle case from a wrfout file.

A case is a directory of raw little-endian float32 arrays, one file per
driver-boundary field, plus ``meta.txt``.  Both implementations read the
SAME words: ``run_nt_oracle.F90`` (WRF v4.6.1's own cu_ntiedtke_driver) and
``woof_side.py`` (WOOF's NewTiedtke adapter on the GPU).  Realism matters
only for which code paths the columns reach; the bitwise claim rests on the
inputs being identical, which they are by construction.

Array layout on disk is WRF memory order, Fortran ``a(i, k, j)``, which is
C order ``(ny, nz, nx)``.  Interface fields carry nz + 1 levels.

THE FORCING LANES.  WRF folds three lanes into what New Tiedtke receives
(module_cumulus_driver.F:879-881)::

    RTHFTEN = (RTHFTEN + RTHRATEN + RTHBLTEN) * pi
    RQVFTEN =  RQVFTEN + RQVBLTEN

wrfout carries none of the advective lanes, so they are built here from the
same state, as plain finite differences of the resolved flow:

* ``rthdynten`` / ``rqvdynten``: -(u d/dx + v d/dy + w d/dz) of dry theta
  and qv, centred differences on the mass grid (one-sided at the edges).
* ``rthratenlw`` / ``rthratensw``: the file's own RTHRATLW / RTHRATSW when it
  carries them (WOOF's wrfout does), otherwise a clear-sky-like profile
  (-1.5 K/day longwave scaled by sqrt(p/p0), with extra cooling at cloudy
  levels, and +0.6 K/day shortwave scaled by qv).
* ``rthblten`` / ``rqvblten``: the surface fluxes HFX / QFX spread uniformly
  through the boundary layer depth PBLH (or 1000 m when absent), as a theta
  and a mixing-ratio tendency.

The lanes are inputs to both codes, so how they are built cannot make the
two disagree; they exist so the fold and the forcing-driven closure paths
see realistic non-zero values instead of the zeros a wrfout would give.

    python make_columns.py WRFOUT OUTDIR [--coeffs FILE] [--dt SECONDS]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from netCDF4 import Dataset

HERE = pathlib.Path(__file__).resolve().parent
# wrfout_columns is the reconstruction the live-column arm proved against
# ArWen's own dump; it is imported where it is used so make_subset and the
# comparator can share this module's names without netCDF reads.
_STAGE_ORACLE = HERE.parent / "ntiedtke_wrf461_oracle"

G = 9.81
RD = 287.0
CP = 7.0 * RD / 2.0
P1000 = 100000.0

LEV = ("t3d", "qv3d", "qc3d", "qi3d", "u3d", "v3d", "pcps", "dz8w",
       "rho3d", "exner", "rthdynten", "rthratenlw", "rthratensw",
       "rthblten", "rqvdynten", "rqvblten")
IFACE = ("p8w", "w")
SFC = ("xland", "hfx", "qfx")


def _has(d, name):
    return name in d.variables


def build_fallback(path):
    """Driver fields for a wrfout that carries no C1H/C2H.

    The hydrostatic reconstruction in wrfout_columns needs the hybrid
    coefficients.  Without them the full pressure P+PB is used on mass
    levels and the interfaces are PSFC at the bottom, P_TOP at the top and
    the mean of the adjacent mass levels between.  Same formulas as
    wrfout_columns for everything else.
    """
    d = Dataset(path)
    v = lambda n: np.asarray(d.variables[n][0], dtype=np.float64)
    p = v("P") + v("PB")
    nz = p.shape[0]
    theta = v("T") + 300.0
    exner = (p / P1000) ** (RD / CP)
    qv, qc, qi = v("QVAPOR"), v("QCLOUD"), v("QICE")
    U, V = v("U"), v("V")
    zw = (v("PH") + v("PHB")) / G
    dz8w = np.zeros_like(p)
    dz8w[:nz - 1] = zw[1:nz] - zw[:nz - 1]
    dz8w[nz - 1] = 0.0
    gamma = CP / (CP - RD)
    theta_m = theta * (1.0 + (461.6 / 287.0) * qv)
    alt = (RD * theta_m / P1000) * (p / P1000) ** (-1.0 / gamma)
    p8w = np.zeros_like(zw)
    p8w[0] = v("PSFC")
    p8w[nz] = float(np.asarray(d.variables["P_TOP"][0]))
    p8w[1:nz] = 0.5 * (p[:nz - 1] + p[1:nz])
    lev = {
        "t3d": theta * exner, "qv3d": qv, "qc3d": qc, "qi3d": qi,
        "u3d": 0.5 * (U[:, :, :-1] + U[:, :, 1:]),
        "v3d": 0.5 * (V[:, :-1, :] + V[:, 1:, :]),
        "pcps": p, "dz8w": dz8w, "rho3d": (1.0 + qv) / alt, "exner": exner,
    }
    iface = {"p8w": p8w, "w": v("W")}
    sfc = {"xland": 2.0 - v("LANDMASK"), "hfx": v("HFX"), "qfx": v("QFX")}
    dx = float(getattr(d, "DX"))
    d.close()
    return lev, iface, sfc, dx


def _ddx(a, dx, axis):
    return np.gradient(a, dx, axis=axis)


def forcing_lanes(path, lev, iface, sfc, dx):
    """The six lanes WRF folds, built from the state (see module doc)."""
    d = Dataset(path)
    v = lambda n: np.asarray(d.variables[n][0], dtype=np.float64)
    nz = lev["t3d"].shape[0]
    theta = lev["t3d"] / lev["exner"]
    qv = lev["qv3d"]
    u, w_if = lev["u3d"], iface["w"]
    vv = lev["v3d"]
    zw = (v("PH") + v("PHB")) / G
    zm = 0.5 * (zw[:nz] + zw[1:nz + 1])
    wm = 0.5 * (w_if[:nz] + w_if[1:nz + 1])

    def advect(a):
        dadx = _ddx(a, dx, 2)
        dady = _ddx(a, dx, 1)
        dadz = np.gradient(a, axis=0) / np.gradient(zm, axis=0)
        return -(u * dadx + vv * dady + wm * dadz)

    lanes = {"rthdynten": advect(theta), "rqvdynten": advect(qv)}

    p = lev["pcps"]
    if _has(d, "RTHRATLW") and _has(d, "RTHRATSW"):
        lanes["rthratenlw"] = v("RTHRATLW")
        lanes["rthratensw"] = v("RTHRATSW")
        lanes["_radiation_source"] = "file"
    else:
        cloud = (lev["qc3d"] + lev["qi3d"]) > 1.0e-6
        lw = -1.5 / 86400.0 * np.sqrt(p / P1000)
        lw = np.where(cloud, lw - 2.0 / 86400.0, lw)
        sw = 0.6 / 86400.0 * np.minimum(qv / 0.01, 1.5)
        lanes["rthratenlw"] = lw
        lanes["rthratensw"] = sw
        lanes["_radiation_source"] = "synthetic"

    pblh = v("PBLH") if _has(d, "PBLH") else np.full(sfc["hfx"].shape, 1000.0)
    pblh = np.maximum(pblh, 50.0)
    zagl = zm - zw[0][None]
    inside = zagl < pblh[None]
    rho = lev["rho3d"]
    lanes["rthblten"] = np.where(
        inside, sfc["hfx"][None] / (rho * CP * pblh[None]) / lev["exner"], 0.0)
    lanes["rqvblten"] = np.where(
        inside, sfc["qfx"][None] / (rho * pblh[None]), 0.0)
    d.close()
    return lanes


def write_case(out, lev, iface, sfc, meta):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in LEV:
        a = np.asarray(lev[name], dtype=np.float32)          # (nz, ny, nx)
        np.ascontiguousarray(a.transpose(1, 0, 2)).astype("<f4").tofile(
            out / f"{name}.f32")
    for name in IFACE:
        a = np.asarray(iface[name], dtype=np.float32)        # (nz+1, ny, nx)
        np.ascontiguousarray(a.transpose(1, 0, 2)).astype("<f4").tofile(
            out / f"{name}.f32")
    for name in SFC:
        np.asarray(sfc[name], dtype=np.float32).astype("<f4").tofile(
            out / f"{name}.f32")
    with open(out / "meta.txt", "w", newline="\n") as f:
        for k, val in meta.items():
            f.write(f"{k} {val}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wrfout")
    ap.add_argument("outdir")
    ap.add_argument("--coeffs", default=None,
                    help="file to take C1H/C2H from when the input has none")
    ap.add_argument("--dt", type=float, default=None,
                    help="model time step, seconds (default: the file's DT)")
    a = ap.parse_args()

    d = Dataset(a.wrfout)
    has_c1h = _has(d, "C1H")
    dt = float(a.dt if a.dt is not None else getattr(d, "DT"))
    d.close()
    if has_c1h or a.coeffs:
        sys.path.insert(0, str(_STAGE_ORACLE))
        import wrfout_columns

        lev, iface, sfc, dx = wrfout_columns.build(a.wrfout, a.coeffs)
        sfc.pop("_hgt", None)
        pressure = "hydrostatic (wrfout_columns)"
    else:
        lev, iface, sfc, dx = build_fallback(a.wrfout)
        pressure = "full P+PB, interfaces averaged (no C1H in file)"
    lanes = forcing_lanes(a.wrfout, lev, iface, sfc, dx)
    source = lanes.pop("_radiation_source")
    lev = {k: val for k, val in lev.items() if k not in ("qvften", "thften")}
    lev.update(lanes)
    nz, ny, nx = lev["t3d"].shape
    for name in LEV + IFACE:
        arr = lev.get(name, iface.get(name))
        if not np.all(np.isfinite(arr)):
            raise SystemExit(f"non-finite values in {name}")
    write_case(a.outdir, lev, iface, sfc, {
        "nx": nx, "ny": ny, "nz": nz, "dx_native": dx, "dt": dt,
        "source": pathlib.Path(a.wrfout).name,
        "pressure": pressure.replace(" ", "_"),
        "radiation_lanes": source,
    })
    print(f"case {a.outdir}: {nx} x {ny} columns, nz {nz}, dx {dx}, "
          f"radiation lanes {source}, pressure {pressure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
