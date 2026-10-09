"""Regime columns for the upper-damping oracle (damp_opt=3 and w_damp).

One 12 x 8 periodic domain (96 columns), 49 levels, on the real hybrid
vertical coordinate of the committed WRF small-step fixture
(tests/data/wrf471_smallstep/real-state.npz, case ``real_initial``).
Each row of columns is one regime; every column is built from its own
surface height, dry surface pressure and theta profile by integrating
WRF's hydrostatic relation, so the column top (and therefore the damping
layer ``htop - zdamp``) differs from column to column:

    j = 0, 1  convective      warm moist columns, deep updrafts and
                              downdrafts, overshooting tops in the layer
    j = 2     stable night    surface inversion, weak w, upper gravity waves
    j = 3     snow / ice      cold column, snow, ice and graupel loading
    j = 4     ocean           sea-level surface (ht = 0 exactly)
    j = 5, 6  high terrain    1.5 to 4.3 km surfaces, steep slopes,
                              mountain waves reaching the damping layer
    j = 7     edge cases      a column at rest, signed zeros, tiny and huge
                              w, alternating signs, a large phi perturbation

Everything here is input construction (float64 then rounded to float32
once); nothing decides an acceptance bound.  ``edge_zdamp`` picks a
``zdamp`` whose ``htop - zdamp`` equals one level's height exactly in
float32, so the ``hk .ge. hbot`` boundary of WRF's damper is exercised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

F32 = np.float32
NX, NY = 12, 8
G, RD, CP, P1000, PT = 9.81, 287.0, 7.0 * 287.0 / 2.0, 1.0e5, 5000.0
CVPM = -(CP - RD) / CP
REGIMES = ("convective", "convective", "stable_night", "snow_ice", "ocean",
           "high_terrain", "high_terrain", "edge_cases")
VECTORS = ("ZNW", "ZNU", "C1F", "C2F", "C1H", "C2H", "C3F", "C4F", "C3H",
           "C4H", "RDNW", "RDN", "DNW", "FNM", "FNP", "CF1", "CF2", "CF3")
EDGE_COLUMN = (7, 5)          # (j, i) whose level height sets edge_zdamp


def _fixture_vectors(root: Path):
    path = root / "tests" / "data" / "wrf471_smallstep" / "real-state.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive["case0_" + name].copy() for name in VECTORS}


def regime_of(j: int) -> str:
    return REGIMES[j]


def build_raw(root: Path, *, map_factors: bool = False, seed: int = 4610):
    """WRF-named fields for the 96-column domain (see module docstring)."""
    rng = np.random.default_rng(seed)
    vec = _fixture_vectors(root)
    nz = vec["ZNU"].size
    ny, nx = NY, NX
    f64 = lambda name: vec[name].astype(np.float64)
    c1h, c2h, c3h, c4h = f64("C1H"), f64("C2H"), f64("C3H"), f64("C4H")
    c3f, c4f, dnw = f64("C3F"), f64("C4F"), f64("DNW")

    ht = np.zeros((ny, nx))
    x = np.arange(nx) / nx * 2 * np.pi
    ht[0:2] = 250.0 + 120.0 * np.sin(x)[None]
    ht[2] = 180.0 + 40.0 * np.cos(x)
    ht[3] = 600.0 + 300.0 * np.sin(2 * x)
    ht[4] = 0.0
    ht[5] = 2900.0 + 1400.0 * np.sin(3 * x)                # 1.5 .. 4.3 km
    ht[6] = 2600.0 + 1300.0 * np.cos(3 * x + 0.4)
    ht[7] = 900.0 + 500.0 * np.sin(x)
    ht[7, 0] = 0.0

    psfc = 101325.0 * (1.0 - 2.25577e-5 * ht) ** 5.25588
    psfc[0:2] -= 300.0                                      # warm-core lows
    psfc[3] += 900.0                                        # cold high
    mub = psfc - PT                                         # dry column mass
    mu = rng.normal(0.0, 250.0, (ny, nx))
    mu[0:2] += -400.0 + 200.0 * np.sin(2 * x)[None]
    mu[7, 0] = 0.0
    mu[7, 1] = -0.0
    mu[7, 6] = 1800.0

    # Full-level pressure for z; half-level total pressure for alpha.
    mtot = mub + mu
    p_half = c3h[:, None, None] * mtot[None] + c4h[:, None, None] + PT
    pb_half = c3h[:, None, None] * mub[None] + c4h[:, None, None] + PT

    # Theta profiles (K) on the base-state half-level pressure.
    lnp = np.log(pb_half / P1000)
    zapprox = -7400.0 * lnp                                 # metres, rough
    theta = np.empty((nz, ny, nx))
    for j in range(ny):
        z = zapprox[:, j, :]
        trop = 12000.0 if REGIMES[j] != "snow_ice" else 9500.0
        if REGIMES[j] == "convective":
            th0 = 303.0 + 3.0 * np.sin(x)[None] + 0.0 * z
            lapse = 3.2e-3
        elif REGIMES[j] == "stable_night":
            th0 = 282.0 + 8.0 * np.clip(z / 400.0, 0, 1)
            lapse = 3.8e-3
        elif REGIMES[j] == "snow_ice":
            th0 = 268.0 + 0.0 * z
            lapse = 4.5e-3
        elif REGIMES[j] == "ocean":
            th0 = 298.0 + 0.0 * z
            lapse = 3.6e-3
        elif REGIMES[j] == "high_terrain":
            th0 = 300.0 + 0.0 * z
            lapse = 3.9e-3
        else:
            th0 = 295.0 + 0.0 * z
            lapse = 3.5e-3
        tropo = th0 + lapse * np.minimum(z, trop)
        theta[:, j, :] = tropo + 0.022 * np.maximum(z - trop, 0.0)
    theta += rng.normal(0.0, 0.15, theta.shape)
    theta[:, 7, 0] = 300.0                                  # at rest: t = 0

    alt = (RD / P1000) * theta * (p_half / P1000) ** CVPM
    alb_t = (RD / P1000) * (theta - (theta - 300.0) * 0.0) \
        * (pb_half / P1000) ** CVPM

    # Hydrostatic base geopotential on full levels.
    phb = np.empty((nz + 1, ny, nx))
    phb[0] = G * ht
    mass_h = c1h[:, None, None] * mub[None] + c2h[:, None, None]
    for k in range(nz):
        phb[k + 1] = phb[k] - dnw[k] * mass_h[k] * alb_t[k]
    zf = phb / G                                            # metres

    # Geopotential perturbation from the column's own alpha difference.
    mass_t = c1h[:, None, None] * mtot[None] + c2h[:, None, None]
    ph = np.zeros((nz + 1, ny, nx))
    for k in range(nz):
        ph[k + 1] = ph[k] - dnw[k] * (mass_t[k] * alt[k] - mass_h[k] * alb_t[k])
    ph *= 0.02                                              # keep it modest
    ph[:, 7, 0] = 0.0
    ph[:, 7, 1] = -0.0
    ph[1:, 7, 6] += 2000.0 * np.linspace(0.1, 1.0, nz)      # top raised ~200 m

    # Vertical velocity (m/s) on full levels.
    zn = zf - zf[0]
    top = zf[-1]
    w = np.zeros((nz + 1, ny, nx))
    for j in range(ny):
        for i in range(nx):
            zc = zn[:, j, i]
            reg = REGIMES[j]
            if reg == "convective":
                amp = (5.0 + 25.0 * rng.random()) * (1 if (i + j) % 3 else -0.4)
                core = amp * np.sin(np.pi * np.clip(zc / 13000.0, 0, 1))
                over = (2.0 + 6.0 * rng.random()) * np.exp(-((zc - 15500.0) / 1800.0) ** 2)
                w[:, j, i] = core + over * np.sign(amp)
            elif reg == "stable_night":
                w[:, j, i] = (0.02 * np.sin(zc / 300.0 + i)
                              + 0.6 * np.sin(zc / 1500.0 + 0.5 * i)
                              * np.clip((zc - 9000.0) / 6000.0, 0, 1))
            elif reg == "snow_ice":
                w[:, j, i] = 0.3 * np.sin(np.pi * np.clip(zc / 6000.0, 0, 1)) \
                    + 0.15 * np.sin(zc / 900.0 + i)
            elif reg == "ocean":
                w[:, j, i] = 0.05 * np.sin(zc / 700.0 + i) \
                    + 0.4 * np.sin(zc / 2500.0) * np.clip((zc - 8000.0) / 8000.0, 0, 1)
            elif reg == "high_terrain":
                w[:, j, i] = (3.0 + 5.0 * rng.random()) * np.cos(zc / 2200.0 + 0.7 * i) \
                    * np.exp(-zc / 40000.0)
            else:
                w[:, j, i] = rng.normal(0.0, 2.0, nz + 1)
    w[0] = 0.0
    # Edge-case row.
    w[:, 7, 0] = 0.0                                        # at rest
    w[:, 7, 1] = -0.0                                       # signed zero
    w[:, 7, 2] = 1.0e-30 * np.where(np.arange(nz + 1) % 2, 1.0, -1.0)
    w[:, 7, 3] = 60.0 * np.clip((zn[:, 7, 3] - 12000.0) / 8000.0, 0, 1)
    w[:, 7, 4] = np.where(np.arange(nz + 1) % 2, 4.0, -4.0)
    w[1:, 7, 7] = rng.normal(0.0, 8.0, nz)
    w[-1, 7, 8] = -0.0
    w[-6:, 7, 9] = 0.0

    u = np.zeros((nz, ny, nx + 1))
    v = np.zeros((nz, ny + 1, nx))
    zh = 0.5 * (zn[1:] + zn[:-1])
    jet = 10.0 + 35.0 * np.exp(-((zh - 11000.0) / 3500.0) ** 2)
    u[:, :, :nx] = jet + rng.normal(0.0, 2.0, (nz, ny, nx))
    v[:, :ny, :] = 0.4 * jet + rng.normal(0.0, 2.0, (nz, ny, nx))
    u[:, 7, :2] = 0.0
    u[:, :, nx] = u[:, :, 0]                                # periodic alias
    v[:, ny, :] = v[:, 0, :]

    q = {name: np.zeros((nz, ny, nx)) for name in
         ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")}
    rh_q = 0.018 * np.exp(-np.maximum(zh, 0) / 2500.0)
    for j in range(ny):
        reg = REGIMES[j]
        scale = {"convective": 1.0, "stable_night": 0.45, "snow_ice": 0.12,
                 "ocean": 0.9, "high_terrain": 0.35, "edge_cases": 0.6}[reg]
        q["QVAPOR"][:, j, :] = scale * rh_q[:, j, :]
        if reg == "convective":
            cl = np.exp(-((zh[:, j, :] - 4000.0) / 2500.0) ** 2)
            q["QCLOUD"][:, j, :] = 1.5e-3 * cl
            q["QRAIN"][:, j, :] = 2.0e-3 * np.exp(-((zh[:, j, :] - 1500.0) / 1500.0) ** 2)
            q["QGRAUP"][:, j, :] = 3.0e-3 * np.exp(-((zh[:, j, :] - 7000.0) / 2000.0) ** 2)
            q["QICE"][:, j, :] = 4.0e-4 * np.exp(-((zh[:, j, :] - 10500.0) / 2000.0) ** 2)
            q["QSNOW"][:, j, :] = 1.0e-3 * np.exp(-((zh[:, j, :] - 8500.0) / 2000.0) ** 2)
        elif reg == "snow_ice":
            q["QSNOW"][:, j, :] = 2.0e-3 * np.exp(-zh[:, j, :] / 3000.0)
            q["QICE"][:, j, :] = 5.0e-4 * np.exp(-((zh[:, j, :] - 3000.0) / 2000.0) ** 2)
            q["QGRAUP"][:, j, :] = 3.0e-4 * np.exp(-zh[:, j, :] / 1500.0)
    q["QVAPOR"][:, 7, 0] = 0.0

    p = p_half - PT                                         # WRF P + PB - pt
    pb = pb_half - PT
    raw = {name: vec[name].copy() for name in VECTORS}
    raw.update(
        T=(theta - 300.0).astype(F32), PH=ph.astype(F32), PHB=phb.astype(F32),
        P=(p - pb).astype(F32), PB=pb.astype(F32), ALT=alt.astype(F32),
        ALB=alb_t.astype(F32), AL=(alt - alb_t).astype(F32),
        MU=mu.astype(F32), MUB=mub.astype(F32), U=u.astype(F32),
        V=v.astype(F32), W=w.astype(F32), HGT=ht.astype(F32),
        **{name: value.astype(F32) for name, value in q.items()})
    # Exact signed zeros survive the float64 construction only if set last.
    raw["W"][:, 7, 1] = F32(-0.0)
    raw["W"][-1, 7, 8] = F32(-0.0)
    raw["PH"][:, 7, 1] = F32(-0.0)
    raw["MU"][7, 1] = F32(-0.0)
    if map_factors:
        msf = 1.0 + 0.035 * np.cos(np.linspace(-1.2, 1.2, ny))[:, None] \
            * np.cos(x)[None] - 0.02
        msfu = np.concatenate((msf, msf[:, :1]), axis=1)
        msfv = np.concatenate((msf, msf[:1]), axis=0)
    else:
        msf = np.ones((ny, nx))
        msfu = np.ones((ny, nx + 1))
        msfv = np.ones((ny + 1, nx))
    for name, value in (("MAPFAC_M", msf), ("MAPFAC_U", msfu), ("MAPFAC_V", msfv)):
        raw[name] = value.astype(F32)
    return raw


def column_heights(raw):
    """WRF's damper heights, float32 in its own operation order."""
    g = F32(G)
    ph = raw["PH"].astype(F32) + raw["PHB"].astype(F32)
    return (ph / g).astype(F32)


def edge_zdamp(raw, level: int = 38):
    """A zdamp for which float32 ``htop - zdamp`` equals one level's hk."""
    j, i = EDGE_COLUMN
    h = column_heights(raw)[:, j, i]
    htop = h[-1]
    for k in range(level, 5, -1):
        zd = F32(np.float64(htop) - np.float64(h[k]))
        if F32(htop - zd) == h[k]:
            return float(zd), k
    raise ValueError("no level gives an exactly representable zdamp")


def damping_layer_levels(raw, zdamp: float):
    """Per column, how many w levels sit at or above hbot (WRF's test)."""
    h = column_heights(raw)
    hbot = (h[-1] - F32(zdamp)).astype(F32)
    return (h[1:] >= hbot[None]).sum(axis=0)
