"""Deterministic input domains for the Grell 3D (cu_physics = 5) oracle.

Each configuration is one 16 x 16 x 40 domain handed WHOLE to WRF v4.6.1's
G3DRV and then to conv_grell_spread3d, exactly as module_cumulus_driver.F
calls them for a single-tile, single-patch run.  Outputs are graded on the
8 x 8 interior band WRF writes (ids+4 .. ide-5, both axes, non-periodic):
64 columns per configuration, every one of which sits on the 3 x 3 (and,
under cugd_avedx > 1, 5 x 5) neighbour stencil G3DRV reads its ensemble
forcing from, so the stencil and the spreading are inside the check rather
than beside it.

Column families (by band position, with jitter, and around the ring):

  0 deep convective, land        4 dry
  1 deep convective, ocean       5 high terrain (psfc ~ 700 hPa)
  2 shallow-only                 6 moist-neutral maritime
  3 capped (strong inversion)    7 ensemble edge cases (negative forcing,
                                   near-zero shear, saturated BL)

The binary layout (stream, little-endian) is read by run_g3.F90:

  int32  nx ny nz ishallow avedx ichoice fqi itimestep
  real32 dt dx
  real32 [ny][nz][nx] u v t q p pi rho dz8w rthften rqvften rthblten rqvblten
  real32 [ny][nz+1][nx] w p8w
  real32 [ny][nx] ht xland gsw
  int32  [ny][nx] kpbl
  real32 [ny][nx] apr_gr apr_w apr_mc apr_st apr_as apr_capma apr_capme
                  apr_capmi   (prior accumulator values; G3 adds to them)

i.e. Fortran (i,k,j) order, WRF's memory order.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

NX = NY = 16
NZ = 40
R_D = 287.0
CP = 1004.5
G = 9.81
P_TOP = 5000.0

#: The configurations.  dt/dx/avedx/ishallow/ichoice/fqi are the namelist
#: axes G3DRV reads; itimestep only seeds the (dead, irandom = 0) RNG.
CONFIGS = {
    "a": dict(ishallow=0, avedx=1, ichoice=0, fqi=1, itimestep=3, dt=60.0, dx=12000.0, day=True, seed=11),
    "b": dict(ishallow=1, avedx=1, ichoice=0, fqi=1, itimestep=7, dt=60.0, dx=12000.0, day=True, seed=22),
    "c": dict(ishallow=1, avedx=3, ichoice=0, fqi=1, itimestep=12, dt=18.0, dx=3000.0, day=True, seed=33),
    "d": dict(ishallow=0, avedx=1, ichoice=5, fqi=0, itimestep=2, dt=45.0, dx=9000.0, day=True, seed=44),
    "e": dict(ishallow=1, avedx=3, ichoice=0, fqi=1, itimestep=40, dt=20.0, dx=4000.0, day=False, seed=55),
    "f": dict(ishallow=1, avedx=1, ichoice=13, fqi=1, itimestep=5, dt=90.0, dx=27000.0, day=True, seed=66),
    "g": dict(ishallow=0, avedx=3, ichoice=7, fqi=0, itimestep=9, dt=30.0, dx=6000.0, day=True, seed=77),
}


def _eta_w(nz: int) -> np.ndarray:
    """WRF-like full eta levels, dense near the surface (nz+1 values, 1 -> 0)."""
    x = np.linspace(0.0, 1.0, nz + 1)
    eta = 1.0 - (0.55 * x + 0.45 * x ** 2.2)
    eta[0], eta[-1] = 1.0, 0.0
    return eta


def _qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    es = np.minimum(es, 0.5 * p)
    return 0.622 * es / (p - es)


def _column(fam: int, rng: np.random.Generator):
    """One column: dict of profiles in physical units (float64)."""
    eta = _eta_w(NZ)
    if fam == 5:
        psfc = 70000.0 + rng.uniform(-2000, 2000)
        ht = 3000.0 + rng.uniform(-300, 300)
    elif fam in (1, 6):
        psfc = 101000.0 + rng.uniform(-500, 500)
        ht = 0.0
    else:
        psfc = 99500.0 + rng.uniform(-1500, 1500)
        ht = rng.uniform(50.0, 600.0)
    pw = P_TOP + eta * (psfc - P_TOP)
    pm = 0.5 * (pw[:-1] + pw[1:])

    # temperature
    tsfc = {0: 303.0, 1: 301.0, 2: 297.0, 3: 304.0, 4: 300.0, 5: 290.0,
            6: 300.5, 7: 299.0}[fam] + rng.uniform(-1.5, 1.5)
    if fam == 5:
        tsfc -= 8.0
    pbl_top = psfc - {0: 15000, 1: 8000, 2: 12000, 3: 10000, 4: 25000,
                      5: 12000, 6: 7000, 7: 9000}[fam] * rng.uniform(0.8, 1.2)
    t = np.empty(NZ)
    gamma_m = {0: 0.0062, 1: 0.0058, 2: 0.0070, 3: 0.0066, 4: 0.0080,
               5: 0.0064, 6: 0.0056, 7: 0.0060}[fam]
    t_pbl = tsfc * (pbl_top / psfc) ** (R_D / CP)
    for k, p in enumerate(pm):
        if p >= pbl_top:
            t[k] = tsfc * (p / psfc) ** (R_D / CP)
        else:
            t[k] = t_pbl * (p / pbl_top) ** (R_D * gamma_m / G)
    t = np.maximum(t, 208.0 + rng.uniform(-3, 3))
    if fam in (3, 2):
        # inversion just above the boundary layer
        lo = pbl_top
        hi = pbl_top - (4000.0 if fam == 3 else 2500.0)
        dT = (5.5 if fam == 3 else 3.0) * rng.uniform(0.7, 1.3)
        sel = (pm < lo) & (pm >= hi)
        t[sel] += dT
        t[pm < hi] += dT * 0.6
    if fam == 7:
        t += rng.normal(0, 0.4, NZ)

    # moisture
    rh = np.empty(NZ)
    for k, p in enumerate(pm):
        if p >= pbl_top:
            rh[k] = {0: 0.80, 1: 0.92, 2: 0.85, 3: 0.82, 4: 0.12, 5: 0.75,
                     6: 0.965, 7: 0.97}[fam]
        else:
            frac = (pbl_top - p) / (pbl_top - P_TOP)
            top = {0: 0.70, 1: 0.80, 2: 0.25, 3: 0.55, 4: 0.05, 5: 0.6,
                   6: 0.85, 7: 0.75}[fam]
            rh[k] = top * (1.0 - frac) + 0.05 * frac
    rh = np.clip(rh + rng.normal(0, 0.02, NZ), 0.01, 0.995)
    q = rh * _qsat(t, pm)
    q[pm < 10000.0] = np.maximum(q[pm < 10000.0] * 0.02, 2e-6)
    if fam == 4:
        q *= 0.3

    # winds: shear depends on family
    z_like = np.log(psfc / pm) * 8000.0
    shear = {0: 2.5e-3, 1: 1.5e-3, 2: 1.0e-3, 3: 3.0e-3, 4: 2.0e-3,
             5: 2.5e-3, 6: 0.8e-3, 7: 1.0e-5}[fam]
    u = 2.0 + shear * z_like + rng.normal(0, 0.3, NZ)
    v = -1.0 + 0.5 * shear * z_like + rng.normal(0, 0.3, NZ)
    if fam == 7:
        u = np.full(NZ, 3.0) + 1e-4 * rng.normal(0, 1, NZ)
        v = np.full(NZ, -2.0)

    # vertical velocity on full levels
    zw = np.log(psfc / pw) * 8000.0
    amp = {0: 0.35, 1: 0.25, 2: 0.05, 3: 0.20, 4: 0.02, 5: 0.25, 6: 0.15,
           7: -0.2}[fam] * rng.uniform(0.5, 1.5)
    w = amp * np.sin(np.pi * np.clip(zw / 12000.0, 0, 1)) + rng.normal(0, 0.03, NZ + 1)
    w[0] = 0.0

    tv = t * (1.0 + 0.608 * q)
    rho = pm / (R_D * tv)
    dz8w = R_D * tv / G * np.log(pw[:-1] / pw[1:])
    pi = (pm / 1.0e5) ** (R_D / CP)

    # forcing: rthften is G3DRV's folded TEMPERATURE forcing (K/s)
    low = pm > (psfc - 30000.0)
    rthften = rng.normal(0, 2e-5, NZ)
    rqvften = np.where(low, 1.0, 0.1) * rng.uniform(0.0, 4e-7, NZ)
    if fam in (0, 1, 5, 6):
        rthften += np.where(pm < 50000.0, -6e-5, 1.5e-5)
        rqvften += np.where(low, 6e-7, 0.0)
    if fam == 7:
        rthften += 2.5e-4
        rqvften = -np.abs(rqvften) * 3.0
    blsel = pm >= pbl_top
    rthblten = np.where(blsel, 1.0, 0.0) * rng.uniform(1e-4, 6e-4)
    rqvblten = np.where(blsel, 1.0, 0.0) * rng.uniform(0.0, 2e-7)
    if fam == 4:
        rqvblten *= 0.0
    kpbl = int(np.sum(blsel)) + (1 if fam != 4 else 0)
    if fam == 7 and rng.uniform() < 0.5:
        kpbl = 0
    xland = 2.0 if fam in (1, 6) else 1.0
    return dict(u=u, v=v, t=t, q=q, p=pm, pi=pi, rho=rho, dz8w=dz8w,
                rthften=rthften, rqvften=rqvften, rthblten=rthblten,
                rqvblten=rqvblten, w=w, p8w=pw, ht=ht, xland=xland,
                kpbl=kpbl)


def build_domain(name: str, cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    lev = ("u", "v", "t", "q", "p", "pi", "rho", "dz8w", "rthften",
           "rqvften", "rthblten", "rqvblten")
    out = {k: np.zeros((NY, NZ, NX)) for k in lev}
    out.update({k: np.zeros((NY, NZ + 1, NX)) for k in ("w", "p8w")})
    out.update({k: np.zeros((NY, NX)) for k in ("ht", "xland", "gsw")})
    out["kpbl"] = np.zeros((NY, NX), dtype=np.int32)
    fam = np.zeros((NY, NX), dtype=np.int32)
    for j in range(NY):
        for i in range(NX):
            # band columns cycle the families; the ring repeats them so
            # every band column's stencil sees a mixed neighbourhood.
            f = (i + 3 * j + (ord(name[0]) - 97)) % 8
            fam[j, i] = f
            c = _column(f, rng)
            for k in lev:
                out[k][j, :, i] = c[k]
            out["w"][j, :, i] = c["w"]
            out["p8w"][j, :, i] = c["p8w"]
            out["ht"][j, i] = c["ht"]
            out["xland"][j, i] = c["xland"]
            out["kpbl"][j, i] = c["kpbl"]
            out["gsw"][j, i] = (rng.uniform(150, 850) if cfg["day"]
                                else (0.0 if rng.uniform() < 0.8 else 0.5))
    for k in ("apr_gr", "apr_w", "apr_mc", "apr_st", "apr_as",
              "apr_capma", "apr_capme", "apr_capmi"):
        out[k] = rng.uniform(0.0, 3.0, (NY, NX)) * (rng.uniform(size=(NY, NX)) < 0.5)
    out["family"] = fam
    return out


LEV_ORDER = ("u", "v", "t", "q", "p", "pi", "rho", "dz8w", "rthften",
             "rqvften", "rthblten", "rqvblten")
IFACE_ORDER = ("w", "p8w")
SFC_ORDER = ("ht", "xland", "gsw")
APR_ORDER = ("apr_gr", "apr_w", "apr_mc", "apr_st", "apr_as", "apr_capma",
             "apr_capme", "apr_capmi")


def write_input(path: Path, cfg: dict, dom: dict) -> None:
    with open(path, "wb") as fh:
        fh.write(struct.pack("<8i", NX, NY, NZ, cfg["ishallow"], cfg["avedx"],
                             cfg["ichoice"], cfg["fqi"], cfg["itimestep"]))
        fh.write(struct.pack("<2f", cfg["dt"], cfg["dx"]))
        for k in LEV_ORDER + IFACE_ORDER + SFC_ORDER:
            fh.write(np.ascontiguousarray(dom[k], dtype="<f4").tobytes())
        fh.write(np.ascontiguousarray(dom["kpbl"], dtype="<i4").tobytes())
        for k in APR_ORDER:
            fh.write(np.ascontiguousarray(dom[k], dtype="<f4").tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Run every closure choice on the same physical domain.
    for choice in range(1, 17):
        CONFIGS[f"closure{choice:02d}"] = dict(CONFIGS["a"], ichoice=choice)
    CONFIGS["dims-wide"] = dict(CONFIGS["a"], maxiens=1, maxens=4,
                               maxens2=2, maxens3=17, ensdim=136)
    CONFIGS["dims-outer"] = dict(CONFIGS["a"], maxiens=2, maxens=3,
                                maxens2=3, maxens3=16, ensdim=288)
    CONFIGS["periodic-x"] = dict(CONFIGS["b"], periodic_x=1)
    CONFIGS["periodic-y"] = dict(CONFIGS["c"], periodic_y=1)
    CONFIGS["periodic-xy"] = dict(CONFIGS["c"], periodic_x=1, periodic_y=1)
    (out / "configurations.json").write_text(json.dumps(CONFIGS, sort_keys=True, indent=2))
    for name, cfg in CONFIGS.items():
        dom = build_domain(name, cfg)
        write_input(out / f"g3-in-{name}.bin", cfg, dom)
        np.save(out / f"g3-family-{name}.npy", dom["family"])
    print(f"wrote {len(CONFIGS)} configurations to {out}")


if __name__ == "__main__":
    main()
