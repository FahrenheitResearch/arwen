"""Column cases for the WRF v4.7.1 UW PBL (bl_pbl_physics=9) oracle.

Writes case files for run_camuwpbl.F90: a header (magic 'UWPB', version 1,
ncol, nk, nsteps, dt) then float32 arrays in the order the Fortran reads
them.  Each file holds one vertical grid; columns are drawn from six regime
families, each with its own random perturbations from a fixed seed, so the
same command always writes the same bytes on any host with the same NumPy
major version (the profiles are built in float64 with only +, -, *, /,
exp and log, and then rounded once to float32; the files, not the
generator, are the fixture of record).

Families (the regimes the scheme's branches are keyed on):
  convective  daytime dry convective boundary layer over land
  stable      clear-sky nocturnal surface inversion with a low-level jet
  stratocu    marine stratocumulus-topped mixed layer, cloud-top LW cooling
  coldpool    valley cold pool: elevated terrain, strong surface inversion,
              near-calm wind, some with radiation fog
  icecloud    cold mixed-phase column: cloud ice with ice number aloft,
              a supercooled liquid layer
  shallowcu   moist layer with partial cloud cover over a mixed layer

Usage:
  python make_cases.py OUTDIR [--columns-per-family N] [--nsteps S] [--seed X]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MAGIC = 1431785538  # arbitrary tag 0x55575042
R_D = 287.0
CP = 7.0 * R_D / 2.0      # WRF module_model_constants cp
RCP = R_D / CP
G = 9.81
P0 = 1.0e5
EP = 0.622

FAMILIES = ("convective", "stable", "stratocu", "coldpool", "icecloud",
            "shallowcu")

MASS_FIELDS = ("u", "v", "th", "rho", "qv", "qc", "qi", "qnc", "qni", "p",
               "z", "t", "cldfra", "exner", "rthratenlw", "wsedl3d")
FULL_FIELDS = ("p8w", "z_at_w")
SFC_FIELDS = ("hfx", "qfx", "ust", "ht")


def interface_heights(nk: int, ztop: float, dz0: float) -> np.ndarray:
    """nk+1 interface heights above ground, stretched from dz0 to ztop."""
    # geometric stretch with a cap, then rescaled so the top lands on ztop
    dz = np.empty(nk)
    d = dz0
    for k in range(nk):
        dz[k] = d
        d = min(d * 1.13, 1000.0)
    z = np.concatenate([[0.0], np.cumsum(dz)])
    return z * (ztop / z[-1])


def esat_water(t):
    return 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))


def build_column(rng, family: str, nk: int, ztop: float):
    zi = interface_heights(nk, ztop, dz0=rng.uniform(22.0, 45.0))
    zm = 0.5 * (zi[1:] + zi[:-1])
    th = np.empty(nk)
    qv = np.empty(nk)
    qc = np.zeros(nk)
    qi = np.zeros(nk)
    qni = np.zeros(nk)
    cf = np.zeros(nk)
    lw = np.full(nk, -1.5 / 86400.0)
    u = np.empty(nk)
    v = np.empty(nk)
    free_lapse = rng.uniform(3.2e-3, 4.5e-3)

    if family == "convective":
        ht = rng.uniform(50.0, 600.0)
        psfc = 101325.0 * np.exp(-ht / 8400.0)
        pbl = rng.uniform(900.0, 2200.0)
        ths = rng.uniform(296.0, 306.0)
        jump = rng.uniform(1.5, 6.0)
        q0 = rng.uniform(6e-3, 14e-3)
        for k, z in enumerate(zm):
            if z < 60.0:
                th[k] = ths + 1.2 * (1.0 - z / 60.0)  # superadiabatic skin
            elif z < pbl:
                th[k] = ths
            else:
                th[k] = ths + jump + free_lapse * (z - pbl)
            qv[k] = q0 * (1.0 - 0.1 * z / pbl) if z < pbl else \
                0.45 * q0 * np.exp(-(z - pbl) / 2500.0)
        us = rng.uniform(2.0, 9.0)
        u[:] = us + 2.5e-3 * zm
        v[:] = rng.uniform(-3.0, 3.0) + 5e-4 * zm
        hfx = rng.uniform(120.0, 420.0)
        qfx = rng.uniform(4e-5, 1.6e-4)
        ust = rng.uniform(0.25, 0.7)
    elif family == "stable":
        ht = rng.uniform(20.0, 400.0)
        psfc = 101325.0 * np.exp(-ht / 8400.0)
        ths = rng.uniform(278.0, 292.0)
        depth = rng.uniform(120.0, 350.0)
        inv = rng.uniform(3.0, 9.0)
        jet_z = rng.uniform(180.0, 450.0)
        jet = rng.uniform(6.0, 16.0)
        q0 = rng.uniform(3e-3, 9e-3)
        for k, z in enumerate(zm):
            if z < depth:
                th[k] = ths + inv * (z / depth) ** 0.6
            else:
                th[k] = ths + inv + free_lapse * (z - depth)
            qv[k] = q0 * np.exp(-z / 3000.0)
        u[:] = jet * np.exp(-((zm - jet_z) / (0.8 * jet_z)) ** 2) + \
            3.0 * (1.0 - np.exp(-zm / 2000.0)) + 0.4
        v[:] = rng.uniform(-2.0, 2.0) * (1.0 - np.exp(-zm / 500.0)) + 0.2
        hfx = -rng.uniform(8.0, 60.0)
        qfx = rng.uniform(0.0, 3e-6)
        ust = rng.uniform(0.05, 0.3)
        lw[zm < depth] = -rng.uniform(1.0, 4.0) / 86400.0
    elif family == "stratocu":
        ht = 0.0
        psfc = rng.uniform(100800.0, 102500.0)
        pbl = rng.uniform(650.0, 1200.0)
        ths = rng.uniform(285.0, 292.0)
        jump = rng.uniform(6.0, 11.0)
        qt = rng.uniform(7.5e-3, 10.5e-3)
        cb = pbl - rng.uniform(200.0, 450.0)
        lwc = rng.uniform(2e-4, 7e-4)
        ktop = int(np.searchsorted(zi, pbl)) - 1
        for k, z in enumerate(zm):
            if z < pbl:
                th[k] = ths
                if z > cb:
                    qc[k] = lwc * (z - cb) / (pbl - cb)
                    cf[k] = 1.0
                qv[k] = qt - qc[k]
            else:
                th[k] = ths + jump + free_lapse * (z - pbl)
                qv[k] = 0.35 * qt * np.exp(-(z - pbl) / 2500.0)
        if 0 <= ktop < nk:
            lw[ktop] = -rng.uniform(4e-4, 1.8e-3)
            if ktop >= 1:
                lw[ktop - 1] = -rng.uniform(5e-5, 3e-4)
        u[:] = rng.uniform(4.0, 10.0) + 1e-3 * zm
        v[:] = rng.uniform(-4.0, 0.0)
        hfx = rng.uniform(2.0, 25.0)
        qfx = rng.uniform(1.5e-5, 5e-5)
        ust = rng.uniform(0.15, 0.4)
    elif family == "coldpool":
        ht = rng.uniform(350.0, 1100.0)
        psfc = 101325.0 * np.exp(-ht / 8400.0)
        ths = rng.uniform(270.0, 282.0)
        depth = rng.uniform(150.0, 450.0)
        inv = rng.uniform(7.0, 15.0)
        q0 = rng.uniform(2.5e-3, 5.5e-3)
        fog = rng.uniform() < 0.5
        for k, z in enumerate(zm):
            if z < depth:
                th[k] = ths + inv * (z / depth) ** 1.3
            else:
                th[k] = ths + inv + free_lapse * (z - depth)
            qv[k] = q0 * np.exp(-z / 2500.0)
        nfog = int(rng.integers(2, 5))
        if fog:
            qc[:nfog] = rng.uniform(3e-5, 2.5e-4, size=nfog)
            cf[:nfog] = 1.0
            lw[nfog - 1] = -rng.uniform(5e-5, 2e-4)
        speed = rng.uniform(0.2, 2.0)
        ang = rng.uniform(0.0, 2.0 * np.pi)
        u[:] = speed * np.cos(ang) * (1.0 + zm / 800.0)
        v[:] = speed * np.sin(ang) * (1.0 + zm / 800.0)
        hfx = -rng.uniform(1.0, 18.0)
        qfx = rng.uniform(0.0, 1e-6)
        ust = rng.uniform(0.015, 0.09)
    elif family == "icecloud":
        ht = rng.uniform(0.0, 800.0)
        psfc = 101325.0 * np.exp(-ht / 8400.0)
        ths = rng.uniform(262.0, 272.0)
        pbl = rng.uniform(300.0, 1200.0)
        q0 = rng.uniform(1.2e-3, 3e-3)
        ice_lo = rng.uniform(1500.0, 3000.0)
        ice_hi = ice_lo + rng.uniform(800.0, 2500.0)
        sc_lo = pbl - rng.uniform(100.0, 250.0)
        for k, z in enumerate(zm):
            th[k] = ths + (0.0 if z < pbl else 2.0 + free_lapse * (z - pbl))
            qv[k] = q0 * np.exp(-z / 3000.0)
            if ice_lo < z < ice_hi:
                qi[k] = rng.uniform(5e-6, 6e-5)
                qni[k] = rng.uniform(5e3, 2e5)
                cf[k] = 1.0
            if sc_lo < z < pbl:
                qc[k] = rng.uniform(2e-5, 1.5e-4)
                qi[k] += rng.uniform(1e-6, 1e-5)
                qni[k] += rng.uniform(1e3, 3e4)
                cf[k] = 1.0
        u[:] = rng.uniform(3.0, 14.0) + 2e-3 * zm
        v[:] = rng.uniform(-6.0, 6.0)
        hfx = rng.uniform(-15.0, 60.0)
        qfx = rng.uniform(0.0, 2e-5)
        ust = rng.uniform(0.1, 0.55)
    elif family == "shallowcu":
        ht = rng.uniform(0.0, 300.0)
        psfc = 101325.0 * np.exp(-ht / 8400.0)
        pbl = rng.uniform(450.0, 800.0)
        ths = rng.uniform(296.0, 300.0)
        q0 = rng.uniform(14e-3, 18e-3)
        cu_top = pbl + rng.uniform(900.0, 1800.0)
        for k, z in enumerate(zm):
            if z < pbl:
                th[k] = ths
                qv[k] = q0
            elif z < cu_top:
                th[k] = ths + 3.0e-3 * (z - pbl) + 0.3
                qv[k] = q0 * (1.0 - 0.45 * (z - pbl) / (cu_top - pbl))
                cf[k] = rng.uniform(0.05, 0.45)
                qc[k] = cf[k] * rng.uniform(1e-4, 6e-4)
            else:
                th[k] = ths + 3.0e-3 * (cu_top - pbl) + 2.5 + \
                    free_lapse * (z - cu_top)
                qv[k] = 0.4 * q0 * np.exp(-(z - cu_top) / 2000.0)
        u[:] = -rng.uniform(4.0, 9.0) + 1e-3 * zm
        v[:] = rng.uniform(-2.0, 2.0)
        hfx = rng.uniform(8.0, 30.0)
        qfx = rng.uniform(8e-5, 1.6e-4)
        ust = rng.uniform(0.18, 0.4)
    else:
        raise ValueError(family)

    # small random structure so no two columns share a branch history
    th += rng.normal(0.0, 0.03, nk)
    u += rng.normal(0.0, 0.08, nk)
    v += rng.normal(0.0, 0.08, nk)
    qv = np.maximum(qv * (1.0 + rng.normal(0.0, 0.01, nk)), 1e-7)

    # hydrostatic pressure on the interfaces, integrated upward from psfc
    p8w = np.empty(nk + 1)
    p8w[0] = psfc
    for k in range(nk):
        dz = zi[k + 1] - zi[k]
        pm = p8w[k]
        for _ in range(3):
            pmid = 0.5 * (p8w[k] + pm)
            t_mid = th[k] * (pmid / P0) ** RCP
            tv = t_mid * (1.0 + 0.608 * qv[k])
            pm = p8w[k] * np.exp(-G * dz / (R_D * tv))
        p8w[k + 1] = pm
    p = 0.5 * (p8w[1:] + p8w[:-1])
    exner = (p / P0) ** RCP
    t = th * exner
    rho = p / (R_D * t * (1.0 + 0.608 * qv))
    qnc = np.where(qc > 0.0, 1.0e8 * (1.0 + 0.2 * rng.uniform(size=nk)), 0.0)
    wsedl = np.zeros(nk)
    # the scheme divides by the level-1 wind speed; WRF never sees exactly 0
    if abs(u[0]) < 1e-3 and abs(v[0]) < 1e-3:
        u[0] = 1e-3

    col = {
        "u": u, "v": v, "th": th, "rho": rho, "qv": qv, "qc": qc, "qi": qi,
        "qnc": qnc, "qni": qni, "p": p, "z": zm + ht, "t": t, "cldfra": cf,
        "exner": exner, "rthratenlw": lw, "wsedl3d": wsedl,
        "p8w": p8w, "z_at_w": zi + ht,
        "hfx": np.array([hfx]), "qfx": np.array([qfx]), "ust": np.array([ust]),
        "ht": np.array([ht]),
    }
    # WRF stores t_phy = th_phy * pi_phy in single precision; do the same
    col = {k: np.asarray(a, dtype=np.float64).astype(np.float32)
           for k, a in col.items()}
    col["t"] = (col["th"] * col["exner"]).astype(np.float32)
    return col


def write_case_file(path: Path, columns, nk: int, nsteps: int, dt: float):
    ncol = len(columns)
    with open(path, "wb") as fh:
        np.array([MAGIC, 1, ncol, nk, nsteps], dtype="<i4").tofile(fh)
        np.array([dt], dtype="<f4").tofile(fh)
        for name in MASS_FIELDS + FULL_FIELDS + SFC_FIELDS:
            arr = np.stack([c[name] for c in columns]).astype("<f4")
            arr.tofile(fh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--columns-per-family", type=int, default=3)
    ap.add_argument("--nsteps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    # grids: (name, nk, model top above ground, dt)
    grids = (("g44", 44, 16500.0, 60.0), ("g61", 61, 20000.0, 20.0),
             ("g35", 35, 14000.0, 150.0))
    index = {}
    rng = np.random.default_rng(args.seed)
    for gname, nk, ztop, dt in grids:
        columns, meta = [], []
        for family in FAMILIES:
            for j in range(args.columns_per_family):
                columns.append(build_column(rng, family, nk, ztop))
                meta.append({"family": family, "member": j})
        path = args.outdir / f"cases-{gname}.bin"
        write_case_file(path, columns, nk, args.nsteps, dt)
        index[gname] = {"file": path.name, "nk": nk, "dt": dt,
                        "nsteps": args.nsteps, "columns": meta}
    (args.outdir / "cases-index.json").write_text(
        json.dumps(index, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
