#!/usr/bin/env python3
"""Build the mp=28 column-oracle input set.

Three sources, all written as raw float32 microphysics inputs (the
``extract_columns.py`` layout: ``p th`` (ncol, nz), ``geop w``
(ncol, nz+1), the eleven moments, ``nwfa2d nifa2d`` (ncol,), plus
``labels`` and ``regime``):

1. ``real``: the 42 columns of ``tests/data/thompson_real_columns_wrf461.npz``
   (one convective real-data case, chosen per process regime from WRF's own
   checkpoints).
2. ``synthetic``: soundings built here for regimes the real case lacks --
   deep convection, tropical anvil, a stable night with fog and drizzle, a
   cold snow/ice column, marine stratocumulus over the ocean with clean air,
   high terrain (surface near 640 hPa) with orographic cloud, freezing rain
   under a warm nose, and a polluted continental cumulus.  Each regime has
   four seeded variants.  Every variant enters raw and after spin-up by WRF
   itself: the column is stepped through UNMODIFIED WRF v4.6.1 (the
   pristine batch driver) 10 and 45 times at 20 s, so the spun-up states
   are WRF's own self-consistent mass/number/aerosol distributions.
3. ``edge``: hand-made edge cases (all-zero hydrometeors, values exactly at
   and one unit around R1, orphan numbers, masses without numbers, zero and
   extreme aerosol, very heavy rain and graupel, supersaturation with no
   condensate, a column of subsaturated cloud, strong up/down drafts,
   vapour near zero, small negative moments as advection leaves them, and
   four columns that empty the cloud in one step by every path WRF has:
   evaporation, the cloud-water limiter under rain and under riming, the
   freeze below HGFR and the melt above 0 C).  Edge cases are appended
   last, so the earlier columns keep their indices.

usage: make_columns.py WRF_BUILD_DIR OUT.npz [--seed S]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import oracle_io as io  # noqa: E402

f32 = np.float32
NZ = 49
G, RD, RV, CP, P0 = 9.81, 287.0, 461.6, 1004.5, 1.0e5
RCP = RD / CP
EPS = RD / RV
P_TOP = 5000.0
R1 = f32(1.0e-12)


# ---------------------------------------------------------------------------
# Soundings.
# ---------------------------------------------------------------------------

def esat_w(t):
    return 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))


def esat_i(t):
    return 611.2 * np.exp(21.8745584 * (t - 273.16) / (t - 7.66))


def qsat(t, p, ice=False):
    e = np.where((t < 273.15) & ice, esat_i(t), esat_w(t))
    return EPS * e / np.maximum(p - e, 1.0)


def eta_levels(nz=NZ):
    """WRF-like stretched full levels, 1 at the surface, 0 at the top."""
    s = np.linspace(0.0, 1.0, nz + 1)
    eta = 1.0 - (0.35 * s + 0.65 * s ** 2.2)
    eta[0], eta[-1] = 1.0, 0.0
    return eta


def column(ps, z0, temp_of_p, rh_of_p, w_of_z, ice_rh=False):
    """A hydrostatic column from a surface pressure and height, a
    temperature T(p) and a relative humidity RH(p) function.  Returns the
    raw fields plus mass-level temperature, pressure and height."""
    eta = eta_levels()
    pf = P_TOP + eta * (ps - P_TOP)                 # faces, nz+1
    pm = 0.5 * (pf[:-1] + pf[1:])                    # mass levels
    t = temp_of_p(pm)
    qv = rh_of_p(pm) * qsat(t, pm, ice=ice_rh)
    tv = t * (1.0 + 0.61 * qv)
    zf = np.empty(NZ + 1)
    zf[0] = z0
    for k in range(NZ):
        zf[k + 1] = zf[k] + RD * tv[k] / G * np.log(pf[k] / pf[k + 1])
    zm = 0.5 * (zf[:-1] + zf[1:])
    th = t * (P0 / pm) ** RCP
    out = {"p": pm, "th": th, "geop": G * zf, "w": w_of_z(zf),
           "qv": qv}
    return out, t, pm, zm


def lapse_profile(ts, zs_fn, lapse, trop_t, inv=None):
    """T(p) from a surface temperature and a constant lapse rate in height
    (approximated through the hydrostatic scale height), with an isothermal
    stratosphere at ``trop_t`` and an optional surface inversion
    ``(depth_pa, delta_t)``."""
    def f(p):
        z = 7400.0 * np.log(zs_fn / p)
        t = ts - lapse * z
        if inv is not None:
            depth, dt = inv
            t = t + dt * np.clip((zs_fn - p) / depth, 0.0, 1.0)
        return np.maximum(t, trop_t)
    return f


def layer(zm, base, top, peak, shape="hump"):
    """A smooth layer profile between heights ``base`` and ``top``."""
    x = (zm - base) / max(top - base, 1.0)
    inside = (x >= 0.0) & (x <= 1.0)
    if shape == "hump":
        v = np.sin(np.pi * np.clip(x, 0.0, 1.0)) ** 1.5
    elif shape == "bottom":
        v = 1.0 - np.clip(x, 0.0, 1.0) ** 2
    else:
        v = np.ones_like(x)
    return np.where(inside, peak * v, 0.0)


def numbers(c, rho, *, nc_cm3, ni_l, nwfa_cc, nifa_l, rng):
    """Numbers to go with masses: cloud droplets from a CCN-like count,
    rain from a Marshall-Palmer slope, ice from a per-litre count, the
    water- and ice-friendly aerosol from a per-cc / per-litre count decaying
    with height.  Per kg throughout."""
    zfac = np.exp(-np.arange(NZ) / 18.0)
    c["nc"] = np.where(c["qc"] > 0, nc_cm3 * 1.0e6 / rho
                       * rng.uniform(0.6, 1.2, NZ), 0.0)
    lam = (np.pi * 1000.0 * 8.0e6 / np.maximum(rho * c["qr"], 1e-12)) ** 0.25
    c["nr"] = np.where(c["qr"] > 0, 8.0e6 / lam / rho
                       * rng.uniform(0.5, 2.0, NZ), 0.0)
    c["ni"] = np.where(c["qi"] > 0, ni_l * 1.0e3 / rho
                       * rng.uniform(0.3, 3.0, NZ), 0.0)
    c["nwfa"] = nwfa_cc * 1.0e6 / rho * (0.15 + 0.85 * zfac)
    c["nifa"] = nifa_l * 1.0e3 / rho * (0.05 + 0.95 * zfac)


# Regimes: each returns one raw column from a seeded generator.

def regime_convective(rng):
    ts = rng.uniform(298, 306)
    ps = rng.uniform(96000, 101000)
    tp = lapse_profile(ts, ps, rng.uniform(6.5e-3, 7.5e-3), 210.0)
    rh = lambda p: np.where(p > 70000, rng.uniform(0.85, 0.97),
                            np.where(p > 25000, 0.95, 0.3))
    wmax = rng.uniform(5, 25)
    c, t, pm, zm = column(ps, rng.uniform(0, 400), tp, rh,
                          lambda z: wmax * np.sin(np.pi * np.clip(
                              (z - z[0]) / 13000.0, 0, 1)))
    c["qc"] = layer(zm, 1200, 7500, rng.uniform(1e-3, 3e-3))
    c["qr"] = layer(zm, zm[0], 5000, rng.uniform(1e-3, 6e-3), "bottom")
    c["qg"] = layer(zm, 3500, 10000, rng.uniform(5e-4, 4e-3))
    c["qs"] = layer(zm, 6000, 12500, rng.uniform(2e-4, 1.5e-3))
    c["qi"] = layer(zm, 8000, 13500, rng.uniform(5e-5, 5e-4))
    return c, t, pm, dict(nc_cm3=rng.uniform(150, 600), ni_l=100,
                          nwfa_cc=rng.uniform(800, 3000), nifa_l=1500)


def regime_tropical_anvil(rng):
    ts = rng.uniform(299, 302)
    ps = rng.uniform(100500, 101300)
    tp = lapse_profile(ts, ps, 6.0e-3, 192.0)
    rh = lambda p: np.where(p > 15000, 0.97, 0.5)
    c, t, pm, zm = column(ps, 0.0, tp, rh, lambda z: rng.uniform(0.3, 3.0)
                          * np.sin(np.pi * np.clip(z / 16000.0, 0, 1)))
    c["qi"] = layer(zm, 10500, 16000, rng.uniform(1e-4, 8e-4))
    c["qs"] = layer(zm, 7000, 14000, rng.uniform(2e-4, 1e-3))
    c["qc"] = layer(zm, 4000, 6500, rng.uniform(1e-4, 6e-4))
    c["qr"] = layer(zm, 0, 4500, rng.uniform(1e-4, 1.5e-3), "bottom")
    return c, t, pm, dict(nc_cm3=80, ni_l=rng.uniform(50, 500),
                          nwfa_cc=300, nifa_l=500)


def regime_stable_night(rng):
    ts = rng.uniform(266, 280)
    ps = rng.uniform(97000, 102500)
    tp = lapse_profile(ts, ps, 6.0e-3, 215.0,
                       inv=(rng.uniform(1500, 4000), rng.uniform(3, 8)))
    rh = lambda p: np.where(ps - p < 1500, 1.01, np.where(p > 60000, 0.7,
                                                           0.35))
    c, t, pm, zm = column(ps, rng.uniform(0, 300), tp, rh,
                          lambda z: rng.uniform(-0.05, 0.05) * np.ones_like(z))
    c["qc"] = layer(zm, zm[0], zm[0] + 180, rng.uniform(5e-5, 3e-4),
                    "bottom")
    c["qr"] = layer(zm, zm[0], zm[0] + 150, rng.uniform(1e-6, 3e-5),
                    "bottom")
    return c, t, pm, dict(nc_cm3=rng.uniform(100, 400), ni_l=10,
                          nwfa_cc=rng.uniform(1000, 5000), nifa_l=2000)


def regime_snow_ice(rng):
    ts = rng.uniform(250, 266)
    ps = rng.uniform(95000, 102000)
    tp = lapse_profile(ts, ps, 5.0e-3, 210.0)
    rh = lambda p: np.where(p > 40000, 1.0, 0.6)
    c, t, pm, zm = column(ps, rng.uniform(0, 600), tp, rh,
                          lambda z: rng.uniform(0.05, 0.6)
                          * np.sin(np.pi * np.clip((z - z[0]) / 8000.0, 0, 1)),
                          ice_rh=True)
    c["qs"] = layer(zm, zm[0], 6000, rng.uniform(1e-4, 1e-3), "bottom")
    c["qi"] = layer(zm, 1500, 8000, rng.uniform(1e-5, 2e-4))
    c["qc"] = layer(zm, 800, 2200, rng.uniform(0, 1.5e-4))
    c["qg"] = layer(zm, zm[0], 2000, rng.uniform(0, 1e-4), "bottom")
    return c, t, pm, dict(nc_cm3=60, ni_l=rng.uniform(30, 300),
                          nwfa_cc=300, nifa_l=rng.uniform(500, 5000))


def regime_marine_sc(rng):
    ts = rng.uniform(286, 294)
    ps = rng.uniform(101000, 102500)
    tp = lapse_profile(ts, ps, 9.0e-3, 210.0, inv=(9000, 0.0))

    def tfun(p, base=tp):
        t = base(p)
        return np.where(ps - p > 11000, t + 8.0, t)
    rh = lambda p: np.where(ps - p < 11000, 1.005, 0.25)
    c, t, pm, zm = column(ps, 0.0, tfun, rh,
                          lambda z: rng.uniform(-0.3, 0.3)
                          * np.sin(np.pi * np.clip(z / 1000.0, 0, 1)))
    c["qc"] = layer(zm, 500, 950, rng.uniform(2e-4, 6e-4))
    c["qr"] = layer(zm, 0, 900, rng.uniform(1e-6, 5e-5), "bottom")
    return c, t, pm, dict(nc_cm3=rng.uniform(25, 90), ni_l=1,
                          nwfa_cc=rng.uniform(60, 200), nifa_l=50)


def regime_high_terrain(rng):
    ts = rng.uniform(262, 285)
    ps = rng.uniform(60000, 70000)
    tp = lapse_profile(ts, ps, 7.0e-3, 212.0)
    rh = lambda p: np.where(p > 45000, rng.uniform(0.95, 1.02), 0.4)
    c, t, pm, zm = column(ps, rng.uniform(3000, 4200), tp, rh,
                          lambda z: rng.uniform(0.5, 4.0)
                          * np.sin(np.pi * np.clip((z - z[0]) / 6000.0, 0, 1)))
    c["qc"] = layer(zm, zm[0] + 200, zm[0] + 2500, rng.uniform(1e-4, 1e-3))
    c["qs"] = layer(zm, zm[0], zm[0] + 4500, rng.uniform(1e-4, 8e-4),
                    "bottom")
    c["qi"] = layer(zm, zm[0] + 2000, zm[0] + 5500, rng.uniform(1e-5, 1e-4))
    c["qg"] = layer(zm, zm[0], zm[0] + 1500, rng.uniform(0, 3e-4), "bottom")
    return c, t, pm, dict(nc_cm3=rng.uniform(100, 300), ni_l=50,
                          nwfa_cc=500, nifa_l=3000)


def regime_freezing_rain(rng):
    ts = rng.uniform(266, 271)
    ps = rng.uniform(98000, 102000)
    base = lapse_profile(ts, ps, 6.0e-3, 212.0)

    def tfun(p):
        t = base(p)
        dz = 7400.0 * np.log(ps / p)
        nose = rng.uniform(4, 8) * np.exp(-((dz - 1500.0) / 600.0) ** 2)
        return t + nose
    rh = lambda p: np.where(p > 50000, 0.99, 0.6)
    c, t, pm, zm = column(ps, rng.uniform(0, 300), tfun, rh,
                          lambda z: rng.uniform(0.0, 0.3) * np.ones_like(z))
    c["qs"] = layer(zm, 2500, 6000, rng.uniform(2e-4, 1e-3))
    c["qr"] = layer(zm, zm[0], 2600, rng.uniform(2e-4, 1.5e-3), "bottom")
    c["qi"] = layer(zm, 4000, 7000, rng.uniform(1e-5, 1e-4))
    c["qg"] = layer(zm, zm[0], 800, rng.uniform(0, 2e-4), "bottom")
    return c, t, pm, dict(nc_cm3=100, ni_l=100, nwfa_cc=800, nifa_l=1500)


def regime_polluted_cumulus(rng):
    ts = rng.uniform(293, 300)
    ps = rng.uniform(95000, 99000)
    tp = lapse_profile(ts, ps, 7.5e-3, 212.0)
    rh = lambda p: np.where(p > 60000, 0.995, 0.4)
    c, t, pm, zm = column(ps, rng.uniform(200, 1200), tp, rh,
                          lambda z: rng.uniform(1.0, 6.0)
                          * np.sin(np.pi * np.clip((z - z[0]) / 5000.0, 0, 1)))
    c["qc"] = layer(zm, zm[0] + 900, zm[0] + 3500, rng.uniform(5e-4, 2e-3))
    c["qr"] = layer(zm, zm[0], zm[0] + 1500, rng.uniform(0, 2e-4), "bottom")
    return c, t, pm, dict(nc_cm3=rng.uniform(800, 2000), ni_l=10,
                          nwfa_cc=rng.uniform(1e4, 4e4), nifa_l=8000)


REGIMES = {
    "convective": regime_convective,
    "tropical anvil": regime_tropical_anvil,
    "stable night fog": regime_stable_night,
    "snow/ice": regime_snow_ice,
    "ocean marine Sc": regime_marine_sc,
    "high terrain": regime_high_terrain,
    "freezing rain": regime_freezing_rain,
    "polluted cumulus": regime_polluted_cumulus,
}


def finish(c, t, pm, nums, rng):
    for s in io.SPECIES:
        c.setdefault(s, np.zeros(NZ))
    rho = pm / (RD * t * (1.0 + 0.61 * c["qv"]))
    numbers(c, rho, rng=rng, **nums)
    c["nwfa2d"] = np.float64(rng.uniform(0, 3e6))
    c["nifa2d"] = np.float64(rng.choice([0.0, rng.uniform(0, 5e4)]))
    return c


# ---------------------------------------------------------------------------
# Edge cases, on one ordinary convective sounding.
# ---------------------------------------------------------------------------

def edge_cases(rng):
    out = []
    base_rng = np.random.default_rng(12345)
    base, t, pm, nums = regime_convective(base_rng)
    base = finish(base, t, pm, nums, base_rng)
    zero = {s: np.zeros(NZ) for s in io.SPECIES if s != "qv"}

    def mk(label, **over):
        c = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
             for k, v in base.items()}
        for k, v in over.items():
            c[k] = v
        out.append((label, c))

    mk("edge dry clear (no condensate)", **zero)
    at_r1 = {s: np.full(NZ, 1.0e-12) for s in ("qc", "qr", "qi", "qs", "qg")}
    mk("edge masses exactly R1", **at_r1,
       nc=np.full(NZ, 1.0e3), nr=np.full(NZ, 1.0), ni=np.full(NZ, 1.0))
    up = float(np.nextafter(R1, f32(1)))
    dn = float(np.nextafter(R1, f32(0)))
    alt = np.where(np.arange(NZ) % 2 == 0, up, dn)
    mk("edge masses one unit about R1",
       **{s: alt.copy() for s in ("qc", "qr", "qi", "qs", "qg")},
       nc=np.full(NZ, 10.0), nr=np.full(NZ, 1.0), ni=np.full(NZ, 1.0))
    mk("edge orphan numbers (n > 0, q = 0)",
       **{**zero, "nc": np.full(NZ, 5.0e7), "nr": np.full(NZ, 3.0e3),
          "ni": np.full(NZ, 2.0e4)})
    mk("edge masses without numbers", nc=np.zeros(NZ), nr=np.zeros(NZ),
       ni=np.zeros(NZ))
    mk("edge zero aerosol", nwfa=np.zeros(NZ), nifa=np.zeros(NZ),
       nwfa2d=0.0, nifa2d=0.0)
    mk("edge extreme aerosol", nwfa=np.full(NZ, 5.0e11),
       nifa=np.full(NZ, 1.0e9), nwfa2d=5.0e7, nifa2d=1.0e6)
    mk("edge heavy rain few drops", qr=base["qr"] * 3.0 + 2e-3 *
       (np.arange(NZ) < 12), nr=np.where(np.arange(NZ) < 12, 50.0, 0.0))
    mk("edge heavy graupel", qg=layer(np.arange(NZ) * 300.0, 1000, 9000,
                                      1.2e-2))
    ss = dict(zero)
    ss["qv"] = base["qv"] * 1.06
    mk("edge supersaturated, no condensate", **ss)
    sub = dict(qv=base["qv"] * 0.5, qc=np.full(NZ, 2.0e-4),
               nc=np.full(NZ, 1.0e8))
    mk("edge subsaturated cloud everywhere", **sub)
    mk("edge strong updraft", w=np.linspace(0, 45.0, NZ + 1))
    mk("edge strong downdraft", w=np.linspace(0, -30.0, NZ + 1))
    qv = base["qv"].copy()
    qv[30:] = 1.0e-10
    qv[40:] = 0.0
    mk("edge vapour near zero aloft", qv=qv)
    neg = {s: base[s] - 2.0e-9 * (np.arange(NZ) % 3 == 0)
           for s in ("qc", "qr", "qi", "qs", "qg")}
    neg.update(nc=base["nc"] - 5.0 * (np.arange(NZ) % 4 == 0),
               nr=base["nr"] - 1.0 * (np.arange(NZ) % 5 == 0))
    mk("edge small negative moments", **neg)

    # The residue cells.  Where one step removes the whole cloud, :3975
    # forms qc1d + qcten*DT as the difference of two nearly equal float32
    # numbers and :4007 zeroes it only at or below R1, so the answer is
    # exactly zero or a residue of one unit in the last place of the entry
    # cloud (2^-35 kg/kg near 3e-4) with a floor droplet number and a
    # radius; which one depends on every bit of the tendency.  These four
    # columns walk the cloud through each path that empties it: whole
    # evaporation (:3472-3475) and droplet evaporation (:3423-3471) in dry
    # air, the cloud-water limiter (:2878-2890) under heavy rain and under
    # rime-heavy snow and graupel, the freeze below HGFR (:3955-3965) and
    # the melt above T_0 (:3946-3953).  The entry cloud steps by a factor of
    # 1.37 per level so its mantissa differs at every level.
    lev = np.arange(NZ)
    sweep = 1.0e-9 * 1.37 ** lev                 # 1e-9 .. 3.6e-3 kg/kg
    sweep = np.minimum(sweep, 4.0e-4)
    drops = np.full(NZ, 1.5e8)
    # The base sounding's aerosol stays; only the hydrometeors are reset.
    bare = {s: np.zeros(NZ) for s in ("qc", "qr", "qi", "qs", "qg", "ni",
                                      "nr", "nc")}
    mk("edge cloud evaporating whole in dry air",
       **{**bare, "qv": base["qv"] * 0.55, "qc": sweep, "nc": drops})
    # 20 g/kg of rain at a 0.6 mm volume diameter collects the cloud
    # faster than one 20 s step: accretion meets its own cap (rc*odts) and
    # autoconversion on top of it trips the limiter.
    mk("edge cloud drained by heavy rain",
       **{**bare, "qc": sweep[::-1].copy(), "nc": drops,
          "qr": np.full(NZ, 2.0e-2),
          "nr": np.full(NZ, 2.0e-2 * (3.672 / 6.0e-4) ** 3
                        / (np.pi * 1000.0))})
    mk("edge cloud drained by riming snow and graupel",
       **{**bare, "qc": sweep, "nc": np.full(NZ, 4.0e7),
          "qs": np.full(NZ, 3.0e-3), "qg": np.full(NZ, 5.0e-3)})
    mk("edge ice melting into cloud and cloud freezing",
       **{**bare, "qc": sweep[::-1].copy(), "nc": drops,
          "qi": sweep, "ni": np.full(NZ, 5.0e4)})
    return out


# ---------------------------------------------------------------------------
# WRF spin-up.
# ---------------------------------------------------------------------------

def wrf_inputs(cols):
    p = cols["p"].astype(f32)
    z8w = cols["geop"].astype(f32) / f32(G)
    inp = {"p": p, "th": cols["th"].astype(f32),
           "pii": np.power(p / f32(P0), f32(RCP)),
           "dz": z8w[:, 1:] - z8w[:, :-1], "hgt": z8w[:, :-1],
           "w": cols["w"].astype(f32),
           "nwfa2d": cols["nwfa2d"].astype(f32),
           "nifa2d": cols["nifa2d"].astype(f32)}
    for s in io.SPECIES:
        inp[s] = cols[s].astype(f32)
    return inp


def spin_up(cols, wrf_build, steps, dt=20.0):
    """Step the columns through the pristine WRF driver ``steps`` times;
    every snapshot in ``steps`` is returned as a column set."""
    binary = Path(wrf_build) / "pristine" / "run_columns_aero"
    run_dir = Path(wrf_build) / "run"
    cur = {k: np.array(v, copy=True) for k, v in cols.items()}
    snaps = {}
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        for n in range(1, max(steps) + 1):
            inp = wrf_inputs(cur)
            io.write_wrf_input(fin, inp, dt)
            io.run_wrf(binary, run_dir, fin, fout)
            out = io.read_wrf_output(fout, *inp["p"].shape)
            cur["th"] = out["th"]
            for s in io.SPECIES:
                cur[s] = out[s]
            if n in steps:
                snaps[n] = {k: np.array(v, copy=True) for k, v in cur.items()}
    return snaps


def stack(entries):
    keys = ("p", "th", "geop", "w", *io.SPECIES, "nwfa2d", "nifa2d")
    out = {k: np.stack([np.asarray(c[k], np.float64) for _, _, c in entries]
                       ).astype(f32) for k in keys}
    out["labels"] = np.array([lab for lab, _, _ in entries])
    out["regime"] = np.array([reg for _, reg, _ in entries])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("wrf_build")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--variants", type=int, default=4)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)

    entries = []
    real = np.load(ROOT / "tests" / "data" / "thompson_real_columns_wrf461.npz")
    for i, lab in enumerate(real["labels"]):
        c = {k: real["col_" + k][i] for k in
             ("p", "th", "geop", "w", *io.SPECIES, "nwfa2d", "nifa2d")}
        entries.append((f"real {real['origin'][i]} {lab}", "real convective",
                        c))

    synth = []
    for name, fn in REGIMES.items():
        for v in range(a.variants):
            c, t, pm, nums = fn(rng)
            synth.append((f"{name} v{v}", name, finish(c, t, pm, nums, rng)))
    raw = stack(synth)
    snaps = spin_up({k: raw[k] for k in raw if k not in ("labels",
                                                         "regime")},
                    a.wrf_build, steps=(10, 45))
    for lab, reg, c in synth:
        entries.append((lab + " raw", reg, c))
    for n, snap in snaps.items():
        for i, (lab, reg, _) in enumerate(synth):
            entries.append((f"{lab} WRF spin-up {n} x 20 s", reg,
                            {k: snap[k][i] for k in snap}))
    for lab, c in edge_cases(rng):
        entries.append((lab, "edge", c))

    out = stack(entries)
    bad = [k for k in out if out[k].dtype == f32 and
           not np.isfinite(out[k]).all()]
    if bad:
        raise SystemExit(f"non-finite inputs in {bad}")
    np.savez(a.out, **out)
    regs, counts = np.unique(out["regime"], return_counts=True)
    print(f"{len(out['labels'])} columns x {NZ} levels -> {a.out}")
    for r, n in zip(regs, counts):
        print(f"  {n:4d}  {r}")


if __name__ == "__main__":
    main()
