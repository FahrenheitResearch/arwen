"""Deterministic input columns for the Milbrandt-Yau (mp_physics=9) oracle.

Writes one little-endian stream file the Fortran runner reads and the GPU
runner reads, so both sides start from the same float32 bits.  Layout
(every array Fortran order, shape (ncol, nz) unless noted):

    int32  ncol, nz, nsteps
    float32 dt
    float32 qv qc qr qi qs qg qh nc nr ni ns ng nh th pii p w dz   (ncol, nz)
    float32 p8w_sfc                                                (ncol,)
    float32 rainnc snownc grplnc hailnc                            (ncol,)

The column set spans the regimes the scheme's branches split on: a deep
convective updraft core and its downdraft, stratiform rain with a melting
layer, a stable night (clear and foggy), winter snow, freezing rain under a
warm nose, maritime shallow cloud with drizzle, high terrain, cirrus anvil,
and a block of edge cases (thresholds epsQ/epsN exactly and one ULP either
side, Q without N and N without Q, small negatives, subnormals, signed
zeros, huge hail and rain drops, a 176 K tropopause, qv = 0, supersaturation
with no condensate, extreme vertical velocity).  Temperatures stay inside
WRF's live 173 K .. 323 K stop (module_mp_milbrandt2mom.F:2819-2823).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

F = np.float32
NZ = 50
FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg", "qh",
          "nc", "nr", "ni", "ns", "ng", "nh", "th", "pii", "p", "w", "dz")
SURFACE = ("p8w", "rainnc", "snownc", "grplnc", "hailnc")
RD, CPD, G = 287.0, 1004.0, 9.81


def _levels(ztop=20000.0, nz=None):
    # Stretched: thin near the ground, ~600 m aloft.
    nz = NZ if nz is None else nz
    eta = np.linspace(0.0, 1.0, nz + 1)
    zw = ztop * (0.35 * eta + 0.65 * eta ** 2)
    return zw, 0.5 * (zw[1:] + zw[:-1])


def _esw(t):
    tc = t - 273.15
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def _esi(t):
    tc = t - 273.15
    return 611.2 * np.exp(22.46 * tc / (tc + 272.62))


def _hydrostatic(psfc, t, zw, z):
    """Pressure at mass levels from the surface value, virtual-free."""
    p = np.empty_like(z)
    pk = psfc
    zlast = zw[0]
    tlast = t[0]
    for k in range(z.size):
        dzk = z[k] - zlast
        tm = 0.5 * (t[k] + tlast)
        pk = pk * np.exp(-G * dzk / (RD * tm))
        p[k] = pk
        zlast, tlast = z[k], t[k]
    return p


def _number_for(q, dmean, rho_x, rho_air):
    """Per-kg number giving mean diameter dmean for a sphere of density rho_x."""
    m = np.pi / 6.0 * rho_x * dmean ** 3
    return np.where(q > 0, q / m, 0.0)


class Column:
    def __init__(self, name, regime, psfc, tsfc, lapse, tropo_t, rh, rng):
        zw, z = _levels()
        self.name, self.regime = name, regime
        t = tsfc - lapse * z
        t = np.maximum(t, tropo_t)
        self.zw, self.z = zw, z
        self.t = t
        self.p = _hydrostatic(psfc, t, zw, z)
        e = rh * np.where(t > 273.15, _esw(t), _esi(t))
        self.qv = 0.622 * e / (self.p - e)
        self.psfc = psfc
        zero = np.zeros_like(z)
        self.q = {k: zero.copy() for k in ("qc", "qr", "qi", "qs", "qg", "qh")}
        self.n = {k: zero.copy() for k in ("nc", "nr", "ni", "ns", "ng", "nh")}
        self.w = zero.copy()
        self.rng = rng

    def band(self, zlo, zhi):
        return (self.z >= zlo) & (self.z <= zhi)

    def put(self, species, mask, qmax, dmean, rho_x, shape="hump"):
        z = self.z
        if not mask.any():
            return
        zl, zh = z[mask].min(), z[mask].max()
        if shape == "hump" and zh > zl:
            prof = np.clip(np.sin(np.pi * (z - zl) / (zh - zl + 1e-9)), 0.0, None) ** 1.5
        else:
            prof = np.ones_like(z)
        jitter = 1.0 + 0.25 * self.rng.standard_normal(z.size)
        q = np.where(mask, np.maximum(qmax * prof * jitter, 0.0), 0.0)
        self.q["q" + species] += q
        rho = self.p / (RD * self.t)
        dvar = dmean * np.exp(0.3 * self.rng.standard_normal(z.size))
        self.n["n" + species] += _number_for(q, dvar, rho_x, rho)

    def saturate(self, mask, over=1.0, ice=False):
        e = np.where(ice, _esi(self.t), _esw(self.t)) * over
        qs = 0.622 * e / (self.p - e)
        self.qv = np.where(mask, qs, self.qv)


def build_columns(seed=20261007):
    rng = np.random.default_rng(seed)
    cols = []

    def new(name, regime, **kw):
        c = Column(name, regime, rng=rng, **kw)
        cols.append(c)
        return c

    # Deep convective updraft cores (supercell-like).
    for v in range(10):
        c = new(f"conv_core_{v}", "convective",
                psfc=rng.uniform(96000, 101000), tsfc=rng.uniform(298, 308),
                lapse=rng.uniform(0.0062, 0.0072), tropo_t=rng.uniform(205, 215),
                rh=np.clip(0.85 - 0.00003 * _levels()[1], 0.2, 1))
        cb, ct = rng.uniform(800, 1500), rng.uniform(11000, 15000)
        core = c.band(cb, ct)
        c.saturate(core, over=1.0 + rng.uniform(0.0, 0.01))
        c.w = np.where(core, rng.uniform(10, 45) * np.sin(
            np.pi * (c.z - cb) / (ct - cb)).clip(0), 0.5)
        c.put("c", c.band(cb, 0.8 * ct), rng.uniform(1e-3, 4e-3), 14e-6, 1000)
        c.put("r", c.band(0, 0.6 * ct), rng.uniform(1e-3, 6e-3), 1.2e-3, 1000)
        c.put("i", c.band(0.5 * ct, ct), rng.uniform(1e-4, 1e-3), 60e-6, 500)
        c.put("s", c.band(0.4 * ct, ct), rng.uniform(5e-4, 3e-3), 1.5e-3, 100)
        c.put("g", c.band(0.25 * ct, 0.8 * ct), rng.uniform(1e-3, 6e-3), 2.5e-3, 400)
        c.put("h", c.band(0.15 * ct, 0.7 * ct), rng.uniform(5e-4, 8e-3), 8e-3, 900)

    # Convective downdraft / outflow: sub-saturated, evaporating rain, melting hail.
    for v in range(6):
        c = new(f"conv_down_{v}", "convective",
                psfc=rng.uniform(97000, 100500), tsfc=rng.uniform(296, 304),
                lapse=0.0068, tropo_t=210, rh=rng.uniform(0.45, 0.75))
        c.w = -rng.uniform(3, 20) * np.exp(-((c.z - 2500) / 2000) ** 2)
        c.put("r", c.band(0, 6000), rng.uniform(5e-4, 4e-3), 1.8e-3, 1000)
        c.put("h", c.band(0, 7000), rng.uniform(1e-4, 3e-3), 12e-3, 900)
        c.put("g", c.band(1000, 8000), rng.uniform(1e-4, 2e-3), 3e-3, 400)
        c.put("s", c.band(4000, 9000), rng.uniform(1e-4, 1e-3), 2e-3, 100)

    # Stratiform rain with a melting layer (bright band).
    for v in range(8):
        c = new(f"strat_melt_{v}", "stratiform",
                psfc=rng.uniform(98000, 102000), tsfc=rng.uniform(278, 290),
                lapse=0.0062, tropo_t=213, rh=0.95)
        fl = (c.t[0] - 273.15) / 0.0062
        c.saturate(c.band(fl - 1500, fl + 4000), over=1.0)
        c.w = 0.2 + 0.1 * rng.standard_normal(NZ)
        c.put("s", c.band(fl - 400, fl + 5000), rng.uniform(2e-4, 1.5e-3), 2.5e-3, 100)
        c.put("i", c.band(fl + 2000, fl + 7000), rng.uniform(1e-5, 2e-4), 80e-6, 500)
        c.put("r", c.band(0, fl + 200), rng.uniform(1e-4, 1e-3), 0.9e-3, 1000)
        c.put("c", c.band(fl - 800, fl + 1200), rng.uniform(5e-5, 4e-4), 12e-6, 1000)
        c.put("g", c.band(fl - 300, fl + 2000), rng.uniform(1e-5, 3e-4), 1.5e-3, 400)

    # Stable night: clear-sky inversion, and radiation fog.
    for v in range(8):
        c = new(f"stable_night_{v}", "stable",
                psfc=rng.uniform(99000, 103000), tsfc=rng.uniform(262, 285),
                lapse=0.0055, tropo_t=215, rh=rng.uniform(0.3, 0.7))
        inv = c.band(0, 400)
        c.t = np.where(inv, c.t + 6.0 * c.z / 400.0, c.t + 6.0 * (c.z > 400))
        c.p = _hydrostatic(c.psfc, c.t, c.zw, c.z)
        c.w = 0.01 * rng.standard_normal(NZ)
        if v % 2 == 0:  # fog
            fog = c.band(0, 250)
            c.saturate(fog, over=1.0 + rng.uniform(0, 0.003),
                       ice=False)
            c.put("c", fog, rng.uniform(5e-5, 3e-4), 8e-6, 1000, shape="flat")
            if c.t[0] < 270:
                c.put("i", fog, rng.uniform(1e-6, 2e-5), 40e-6, 500, shape="flat")
        else:
            c.qv = c.qv * rng.uniform(0.2, 0.6)

    # Winter snow (all sub-freezing), some with graupel.
    for v in range(10):
        c = new(f"winter_snow_{v}", "snow",
                psfc=rng.uniform(97000, 103500), tsfc=rng.uniform(250, 271),
                lapse=0.0055, tropo_t=rng.uniform(200, 215), rh=0.9)
        c.saturate(c.band(300, 6000), over=1.0, ice=True)
        c.w = rng.uniform(0.05, 1.5) * np.exp(-((c.z - 3000) / 2500) ** 2)
        c.put("s", c.band(0, 7000), rng.uniform(1e-4, 2e-3), 3e-3, 100)
        c.put("i", c.band(2000, 9000), rng.uniform(1e-5, 3e-4), 100e-6, 500)
        if v % 3 == 0:
            c.put("g", c.band(0, 3000), rng.uniform(5e-5, 6e-4), 1.2e-3, 400)
        if v % 2 == 0:
            c.put("c", c.band(500, 2500), rng.uniform(2e-5, 2e-4), 10e-6, 1000)

    # Freezing rain / drizzle under a warm nose.
    for v in range(6):
        c = new(f"frz_rain_{v}", "freezing",
                psfc=rng.uniform(98000, 102500), tsfc=rng.uniform(264, 271),
                lapse=0.004, tropo_t=210, rh=0.92)
        nose = c.band(600, 2200)
        c.t = np.where(nose, c.t + rng.uniform(5, 9) * np.sin(
            np.pi * (c.z - 600) / 1600).clip(0), c.t)
        c.p = _hydrostatic(c.psfc, c.t, c.zw, c.z)
        c.saturate(c.band(0, 5000), over=1.0)
        c.put("r", c.band(0, 2200), rng.uniform(1e-4, 1.2e-3),
              (0.2e-3 if v % 2 else 1.0e-3), 1000)
        c.put("s", c.band(1800, 6000), rng.uniform(1e-4, 1e-3), 2e-3, 100)
        c.put("i", c.band(1500, 2500), rng.uniform(1e-6, 5e-5), 50e-6, 500)
        c.put("c", c.band(0, 500), rng.uniform(1e-5, 1e-4), 15e-6, 1000)

    # Maritime shallow cloud with drizzle (low droplet number, warm SST).
    for v in range(10):
        c = new(f"ocean_{v}", "ocean",
                psfc=rng.uniform(100500, 102000), tsfc=rng.uniform(290, 302),
                lapse=0.0065, tropo_t=205, rh=0.82)
        cb, ct = rng.uniform(400, 900), rng.uniform(1500, 3500)
        c.saturate(c.band(cb, ct), over=1.0 + rng.uniform(0, 0.004))
        c.w = 1.5 * np.sin(np.pi * np.clip((c.z - cb) / (ct - cb), 0, 1))
        c.put("c", c.band(cb, ct), rng.uniform(2e-4, 1.5e-3), 22e-6, 1000)
        c.put("r", c.band(0, ct), rng.uniform(1e-5, 4e-4),
              (0.12e-3 if v % 2 else 0.6e-3), 1000)

    # High terrain: low surface pressure, cold, snow/graupel, orographic lift.
    for v in range(8):
        c = new(f"high_terrain_{v}", "terrain",
                psfc=rng.uniform(58000, 72000), tsfc=rng.uniform(255, 285),
                lapse=0.0065, tropo_t=205, rh=0.88)
        c.saturate(c.band(200, 4000), over=1.0, ice=bool(v % 2))
        c.w = rng.uniform(0.5, 4.0) * np.exp(-((c.z - 1500) / 1500) ** 2)
        c.put("s", c.band(0, 5000), rng.uniform(1e-4, 1.5e-3), 2.5e-3, 100)
        c.put("g", c.band(0, 3000), rng.uniform(5e-5, 1e-3), 1.5e-3, 400)
        c.put("c", c.band(200, 2500), rng.uniform(5e-5, 6e-4), 12e-6, 1000)
        c.put("i", c.band(2000, 6000), rng.uniform(1e-5, 1e-4), 70e-6, 500)
        if v >= 4:
            c.put("r", c.band(0, 1200), rng.uniform(1e-5, 4e-4), 0.8e-3, 1000)

    # Cirrus / anvil at very cold temperatures.
    for v in range(4):
        c = new(f"anvil_{v}", "anvil",
                psfc=rng.uniform(99000, 101500), tsfc=rng.uniform(295, 305),
                lapse=0.0068, tropo_t=rng.uniform(190, 200), rh=0.6)
        c.saturate(c.band(9000, 14000), over=1.0 + rng.uniform(0, 0.1), ice=True)
        c.put("i", c.band(9000, 14000), rng.uniform(5e-5, 5e-4), 50e-6, 500)
        c.put("s", c.band(8000, 13000), rng.uniform(1e-4, 8e-4), 1.2e-3, 100)

    # ---------------- edge cases ----------------
    def base_edge(name, **kw):
        args = dict(psfc=100000.0, tsfc=290.0, lapse=0.0065, tropo_t=210,
                    rh=0.7)
        args.update(kw)
        return new(name, "edge", **args)

    c = base_edge("edge_dry_zero", rh=0.0)          # qv = 0, nothing at all
    c.qv[:] = 0.0
    c = base_edge("edge_clear", rh=0.4)             # nothing but vapour
    c = base_edge("edge_supersat_water", rh=1.0)   # supersaturated, no condensate
    c.saturate(c.band(0, 6000), over=1.02)
    c.w[:] = 2.0
    c = base_edge("edge_supersat_ice", tsfc=255.0, rh=1.0)
    c.saturate(c.band(0, 9000), over=1.25, ice=True)
    c.w[:] = 0.5

    eps = {"q": 1.0e-14, "n": 1.0e-3}
    for label, factor in (("at", 1.0), ("above", float(np.nextafter(F(1.0), F(2.0)))),
                          ("below", float(np.nextafter(F(1.0), F(0.0))))):
        c = base_edge(f"edge_epsQ_{label}", tsfc=280.0, rh=0.95)
        qeps = float(F(eps["q"]) * F(factor))
        for s in ("c", "r", "i", "s", "g", "h"):
            c.q["q" + s][:] = qeps
            c.n["n" + s][:] = 1.0
    for label, factor in (("at", 1.0), ("above", 1.0000001), ("below", 0.9999999)):
        c = base_edge(f"edge_epsN_{label}", tsfc=280.0, rh=0.95)
        for s, q in zip("crisgh", (1e-4, 1e-4, 1e-5, 1e-4, 1e-4, 1e-4)):
            c.q["q" + s][:] = q
            # number per kg: epsN is per m3, so use a column of 1/rho.
            rho = c.p / (RD * c.t)
            c.n["n" + s][:] = eps["n"] * factor / rho

    c = base_edge("edge_q_without_n", tsfc=285.0, rh=0.98)
    for s in "crisgh":
        c.q["q" + s][:] = 3e-4
    c = base_edge("edge_n_without_q", tsfc=285.0, rh=0.98)
    for s in "crisgh":
        c.n["n" + s][:] = 5e4
    c = base_edge("edge_negatives", tsfc=285.0, rh=0.9)
    for s in "crisgh":
        c.q["q" + s][:] = -1e-9
        c.n["n" + s][:] = -10.0
    c.q["qr"][::3] = 2e-4
    c.n["nr"][::3] = 2e3
    c = base_edge("edge_subnormal", tsfc=275.0, rh=0.9)
    for s in "crisgh":
        c.q["q" + s][::2] = 1e-40
        c.q["q" + s][1::2] = -0.0
        c.n["n" + s][::2] = 1e-42
    c = base_edge("edge_huge_hail", tsfc=300.0, rh=0.8)
    c.q["qh"][:30] = 0.02
    c.n["nh"][:30] = 1.0      # D >> Dh_large
    c.q["qr"][:20] = 0.01
    c.n["nr"][:20] = 10.0     # D >> DrMax
    c = base_edge("edge_tiny_drops", tsfc=295.0, rh=0.97)
    c.q["qr"][:25] = 1e-4
    c.n["nr"][:25] = 1e10     # D << Dr_large
    c.q["qc"][:25] = 2e-3
    c.n["nc"][:25] = 1e12
    c = base_edge("edge_cold_tropopause", tsfc=300.0, lapse=0.0085,
                  tropo_t=176.0, rh=1.0)
    c.saturate(c.band(10000, 20000), over=1.4, ice=True)
    c.put("i", c.band(10000, 18000), 1e-4, 30e-6, 500)
    c = base_edge("edge_triple_point", tsfc=273.16, lapse=0.0, tropo_t=273.16,
                  rh=1.0)
    c.t[:] = np.where(np.arange(NZ) % 2 == 0, 273.16, 273.15)
    for s, d, r in (("c", 12e-6, 1000), ("r", 1e-3, 1000), ("s", 2e-3, 100),
                    ("g", 2e-3, 400), ("h", 6e-3, 900), ("i", 60e-6, 500)):
        c.put(s, c.band(0, 20000), 5e-4, d, r, shape="flat")
    c = base_edge("edge_extreme_w", tsfc=303.0, rh=1.0)
    c.saturate(c.band(0, 14000), over=1.0)
    c.w = 60.0 * np.sin(np.arange(NZ) * 0.7)
    c.put("c", c.band(500, 12000), 3e-3, 16e-6, 1000)
    c.put("g", c.band(4000, 12000), 4e-3, 4e-3, 400)
    c = base_edge("edge_all_species_every_level", tsfc=278.0, rh=1.0)
    c.saturate(c.band(0, 20000), over=1.0)
    for s, d, r in (("c", 12e-6, 1000), ("r", 1e-3, 1000), ("s", 2e-3, 100),
                    ("g", 2e-3, 400), ("h", 6e-3, 900), ("i", 60e-6, 500)):
        c.put(s, c.band(0, 20000), 3e-4, d, r, shape="flat")
    c = base_edge("edge_low_rh_tiny_q", tsfc=283.0, rh=0.2)
    for s in "crisgh":
        c.q["q" + s][:] = 5e-7       # < epsQ2 with RH < 0.9 / 0.8: clipped
        c.n["n" + s][:] = 100.0
    return cols


def build_stress_columns(ncol=256, seed=97531):
    """Randomised columns for branch coverage beyond the named regimes.

    Each column draws its own sounding and, per species, a presence flag,
    a vertical band, a peak mass (log-uniform 1e-7 .. 1e-2) and a mean
    diameter (log-uniform over a range wider than the scheme's limiters);
    some draws deliberately break moment consistency (mass with no number,
    number with no mass).  Vertical velocity spans -20 .. +40 m/s.
    """
    rng = np.random.default_rng(seed)
    cols = []
    sizes = {"c": (3e-6, 60e-6, 1000), "r": (5e-5, 8e-3, 1000),
             "i": (5e-6, 1e-3, 500), "s": (1e-4, 1.5e-2, 100),
             "g": (2e-4, 1e-2, 400), "h": (1e-3, 8e-2, 900)}
    for v in range(ncol):
        c = Column(f"stress_{v}", "stress",
                   psfc=rng.uniform(55000, 103500), tsfc=rng.uniform(248, 312),
                   lapse=rng.uniform(0.004, 0.0095),
                   tropo_t=rng.uniform(186, 222), rh=rng.uniform(0.3, 1.0),
                   rng=rng)
        zl = rng.uniform(0, 12000)
        zh = zl + rng.uniform(500, 9000)
        c.saturate(c.band(zl, zh), over=rng.uniform(0.97, 1.25),
                   ice=bool(rng.integers(2)))
        amp = rng.uniform(-20, 40)
        zc = rng.uniform(1000, 12000)
        c.w = amp * np.exp(-((c.z - zc) / rng.uniform(800, 5000)) ** 2)             + rng.normal(0, 0.3, c.z.size)
        for sp, (dlo, dhi, rho) in sizes.items():
            if rng.uniform() < 0.35:
                continue
            a = rng.uniform(0, 15000)
            b = a + rng.uniform(300, 10000)
            q = float(np.exp(rng.uniform(np.log(1e-7), np.log(1e-2))))
            d = float(np.exp(rng.uniform(np.log(dlo), np.log(dhi))))
            c.put(sp, c.band(a, b), q, d, rho,
                  shape="flat" if rng.uniform() < 0.3 else "hump")
            u = rng.uniform()
            if u < 0.05:
                c.n["n" + sp][:] = 0.0
            elif u < 0.10:
                c.n["n" + sp] = np.where(c.q["q" + sp] > 0, 0.0,
                                         rng.uniform(1, 1e4))
                c.q["q" + sp][:] = 0.0
        cols.append(c)
    return cols


def pack(cols, dt, nsteps):
    ncol = len(cols)
    arr = {k: np.zeros((ncol, NZ), dtype=F) for k in FIELDS}
    sfc = {k: np.zeros(ncol, dtype=F) for k in SURFACE}
    rng = np.random.default_rng(7)
    for i, c in enumerate(cols):
        p = F(c.p)
        pii = np.power(p / F(1.0e5), F(RD / CPD)).astype(F)
        arr["p"][i] = p
        arr["pii"][i] = pii
        arr["th"][i] = (F(c.t) / pii).astype(F)
        arr["qv"][i] = F(c.qv)
        for k, v in c.q.items():
            arr[k][i] = F(v)
        for k, v in c.n.items():
            arr[k][i] = F(v)
        arr["w"][i] = F(c.w)
        arr["dz"][i] = F(np.diff(c.zw))
        # WRF's p8w(i,1,j): linear-in-height extrapolation of the two lowest
        # mass pressures to the surface (moist_physics_prep_em).
        w1 = (c.zw[0] - c.z[1]) / (c.z[0] - c.z[1])
        sfc["p8w"][i] = F(w1 * c.p[0] + (1.0 - w1) * c.p[1])
        acc = rng.uniform(0, 30) if i % 3 else 0.0
        sfc["rainnc"][i] = F(acc)
        sfc["snownc"][i] = F(acc * 0.3)
        sfc["grplnc"][i] = F(acc * 0.1)
        sfc["hailnc"][i] = F(acc * 0.05)
    return arr, sfc


def write_stream(path, arr, sfc, dt, nsteps):
    ncol = arr["qv"].shape[0]
    with open(path, "wb") as fh:
        np.asarray([ncol, NZ, nsteps], dtype="<i4").tofile(fh)
        np.asarray([dt], dtype="<f4").tofile(fh)
        for k in FIELDS:
            np.asfortranarray(arr[k]).astype("<f4").ravel(order="F").tofile(fh)
        for k in SURFACE:
            sfc[k].astype("<f4").tofile(fh)


def read_stream(path):
    raw = Path(path).read_bytes()
    ncol, nz, nsteps = np.frombuffer(raw[:12], dtype="<i4")
    dt = float(np.frombuffer(raw[12:16], dtype="<f4")[0])
    off = 16
    arr = {}
    for k in FIELDS:
        n = ncol * nz
        arr[k] = np.frombuffer(raw[off:off + 4 * n], dtype="<f4").reshape(
            (ncol, nz), order="F").copy()
        off += 4 * n
    sfc = {}
    for k in SURFACE:
        sfc[k] = np.frombuffer(raw[off:off + 4 * ncol], dtype="<f4").copy()
        off += 4 * ncol
    assert off == len(raw)
    return int(ncol), int(nz), int(nsteps), dt, arr, sfc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=20.0)
    ap.add_argument("--nsteps", type=int, default=10)
    ap.add_argument("--set", choices=("regimes", "stress"), default="regimes")
    ap.add_argument("--nz", type=int, default=50)
    args = ap.parse_args()
    global NZ
    NZ = args.nz
    cols = build_columns() if args.set == "regimes" else build_stress_columns()
    arr, sfc = pack(cols, args.dt, args.nsteps)
    for k, v in arr.items():
        if not np.all(np.isfinite(v)):
            raise SystemExit(f"non-finite input in {k}")
    t = arr["th"] * arr["pii"]
    assert t.min() > 173.0 and t.max() < 323.0, (t.min(), t.max())
    write_stream(args.out, arr, sfc, args.dt, args.nsteps)
    names = Path(args.out).with_suffix(".columns.txt")
    names.write_text("".join(f"{i} {c.regime} {c.name}\n"
                             for i, c in enumerate(cols)))
    print(f"{len(cols)} columns x {NZ} levels, dt={args.dt}, "
          f"nsteps={args.nsteps}, T {t.min():.2f}..{t.max():.2f} K")


if __name__ == "__main__":
    main()
