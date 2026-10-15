#!/usr/bin/env python3
"""Build the Dudhia shortwave (ra_sw_physics=1) column-oracle inputs.

Writes one stream-unformatted little-endian file that run_swrad.F90 reads
and one ``.npz`` holding the same float32 words for the GPU comparison:

    int32   ncol, nz, ncase
    float32 solcon[ncase], swrad_scat[ncase]
    int32   icloud[ncase]
    float32 t, p, qv, qc, qr, qi, qs, qg, dz8w, pii, rho   each (nz, ncol)
    float32 coszen[ncol], albedo[ncol]

C order (nz, ncol) is Fortran order (ncol, nz): WRF's (i, k) with j = 1 and
k = 1 the lowest half level, which is also gpuwm's (nz, 1, ncol) layout.
``t`` is temperature (WRF's t_phy), ``p`` full pressure, ``pii`` the Exner
function SWRAD divides by, ``dz8w`` the layer depth.  ``rho`` is passed
because SWRAD takes it; SWPARA recomputes its own density from p and t.

The columns are built in float64 from a hydrostatic profile per regime and
rounded once to float32.  The committed ``.npz`` is the fixture of record:
numpy's float64 libm may round the last bit of a profile differently on
another platform, so regenerating is not guaranteed to reproduce the same
words, and nothing depends on it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

NZ = 50
SEED = 20261007
RD, RV, CP, G = 287.0, 461.6, 1004.5, 9.81
EP2 = RD / RV

# (solcon W m-2, swrad_scat, icloud).  A: WRF defaults at a mean-orbit
# solar constant.  B: perihelion-like solcon, weaker scattering.  C: aphelion,
# icloud = 0 (clouds ignored).  D: strong clear-air scattering, which drives
# the available flux negative deep in the column (the XABS/XALB clamps) and
# the TRANS0 < 1 rescale through XSCA alone.
CASES = ((1367.0, 1.0, 1), (1412.3, 0.7, 1), (1321.8, 1.0, 0),
         (1360.0, 6.0, 1))


def _qvs(T, p):
    es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
    return EP2 * es / np.maximum(p - es, 1.0)


def _grid(zs, ztop, b):
    s = np.linspace(0.0, 1.0, NZ + 1)
    zf = zs + (ztop - zs) * np.expm1(b * s) / np.expm1(b)
    return 0.5 * (zf[:-1] + zf[1:]), np.diff(zf)


def _thermo(z, zs, ps, tsfc, lapse, ttrop, inv_dt=0.0, inv_h=1.0):
    agl = z - zs
    T = tsfc - lapse * agl
    if inv_dt:
        T = np.where(agl < inv_h, tsfc + inv_dt * agl / inv_h,
                     tsfc + inv_dt - lapse * (agl - inv_h))
    T = np.maximum(T, ttrop)
    p = np.empty_like(z)
    zprev, pprev, tprev = zs, ps, tsfc
    for k in range(z.size):
        tm = 0.5 * (tprev + T[k])
        p[k] = pprev * np.exp(-G * (z[k] - zprev) / (RD * tm))
        zprev, pprev, tprev = z[k], p[k], T[k]
    return T, p


def _bump(z, z0, z1):
    """Smooth 0..1..0 bump between heights z0 and z1."""
    x = (z - z0) / max(z1 - z0, 1.0)
    return np.where((x > 0) & (x < 1),
                    np.sin(np.pi * np.clip(x, 0, 1)) ** 2, 0.0)


def _column(rng, *, zs=0.0, ps=101325.0, tsfc=288.0, lapse=0.0065,
            ttrop=215.0, rh_low=0.7, rh_high=0.2, inv_dt=0.0, inv_h=1.0,
            ztop=20000.0, b=2.2, clouds=()):
    z, dz = _grid(zs, ztop, b)
    T, p = _thermo(z, zs, ps, tsfc, lapse, ttrop, inv_dt, inv_h)
    agl = z - zs
    rh = rh_high + (rh_low - rh_high) * np.exp(-agl / 2500.0)
    rh = rh * rng.uniform(0.92, 1.08, NZ)
    qv = np.clip(rh, 0.0, 1.0) * _qvs(T, p)
    qv = np.where(p < 10000.0, np.minimum(qv, 4.0e-6), qv)
    q = {name: np.zeros(NZ) for name in ("qc", "qr", "qi", "qs", "qg")}
    for name, amp, z0, z1 in clouds:
        jitter = rng.uniform(0.8, 1.2, NZ)
        q[name] += amp * jitter * _bump(agl, z0, z1)
    pii = (p / 1.0e5) ** (RD / CP)
    rho = p / (RD * T * (1.0 + 0.61 * qv))
    return dict(t=T, p=p, qv=qv, dz8w=dz, pii=pii, rho=rho, **q)


def _f32(x):
    return np.float32(x)


def _mu_edges():
    """Cosine-zenith words on the branch points of SWPARA."""
    up = lambda x: np.nextafter(_f32(x), _f32(2.0))
    dn = lambda x: np.nextafter(_f32(x), _f32(-2.0))
    return [
        ("night_mu_eq_1e-9", _f32(1.0e-9)),          # CSZA .LE. 1.E-9: night
        ("mu_just_above_1e-9", up(1.0e-9)),          # first day word
        ("mu_2e-9", _f32(2.0e-9)),
        ("mu_1e-6", _f32(1.0e-6)),
        ("mu_0p01", _f32(0.01)),
        ("mu_eq_0p2", _f32(0.2)),                    # XMUVAL(2): not .GT.
        ("mu_above_0p2", up(0.2)),
        ("mu_below_0p2", dn(0.2)),
        ("mu_eq_0p5", _f32(0.5)),                    # XMUVAL(3)
        ("mu_above_0p5", up(0.5)),
        ("mu_below_0p5", dn(0.5)),
        ("mu_eq_1", _f32(1.0)),
        ("night_mu_zero", _f32(0.0)),
        ("night_mu_neg", _f32(-0.35)),
    ]


def build(rng):
    cols = []

    def add(regime, mu, alb, **kw):
        col = _column(rng, **kw)
        col["coszen"] = _f32(mu)
        col["albedo"] = _f32(alb)
        col["regime"] = regime
        cols.append(col)

    # Clear mid-latitude day.
    for mu in np.linspace(0.25, 0.97, 8):
        add("clear_midlat", mu, rng.uniform(0.12, 0.25),
            tsfc=rng.uniform(285, 300), rh_low=rng.uniform(0.4, 0.8))
    # Deep convection: thick liquid with rain, graupel and an ice/snow anvil.
    # Thick first cloud layers at low sun drive ALW to its 3.999 cap and the
    # TRANS0 < 1 rescale.
    for mu in np.linspace(0.08, 1.0, 10):
        s = rng.uniform(0.7, 1.6)
        add("deep_convective", mu, rng.uniform(0.14, 0.2), tsfc=302.0,
            lapse=0.0062, ttrop=195.0, rh_low=0.92, rh_high=0.6,
            clouds=(("qc", 2.2e-3 * s, 600, 7000),
                    ("qr", 1.5e-3 * s, 0, 5000),
                    ("qg", 2.0e-3 * s, 3500, 9500),
                    ("qi", 6.0e-4 * s, 7000, 15000),
                    ("qs", 1.2e-3 * s, 5500, 12500)))
    # Stable night: radiation inversion, low cloud; every column is dark.
    for mu in (-0.9, -0.5, -0.2, -0.05, -1.0e-6, 0.0, 5.0e-10, 1.0e-9):
        add("stable_night", mu, rng.uniform(0.15, 0.3), tsfc=268.0,
            inv_dt=9.0, inv_h=300.0, rh_low=0.85,
            clouds=(("qc", 3.0e-4, 50, 400),))
    # Snow and ice: polar, high albedo, low sun, ice and snow clouds.
    for mu in np.linspace(0.04, 0.42, 8):
        add("snow_ice_polar", mu, rng.uniform(0.75, 0.9), tsfc=246.0,
            lapse=0.004, ttrop=205.0, inv_dt=6.0, inv_h=500.0,
            rh_low=0.9, rh_high=0.5,
            clouds=(("qi", 2.5e-4, 300, 4500), ("qs", 4.0e-4, 0, 3500),
                    ("qc", 6.0e-5, 800, 1500)))
    # Ocean: dark surface, marine stratocumulus under a subsidence inversion.
    for mu in np.linspace(0.15, 0.99, 8):
        add("ocean_stratocumulus", mu, rng.uniform(0.05, 0.08),
            tsfc=291.0, inv_dt=7.0, inv_h=900.0, rh_low=0.95,
            clouds=(("qc", 6.0e-4 * rng.uniform(0.5, 1.5), 500, 950),
                    ("qr", 2.0e-5, 0, 800)))
    # High terrain: low surface pressure, thin layers, bright snow patches.
    for mu in np.linspace(0.35, 0.99, 8):
        zs = rng.uniform(3000, 4800)
        add("high_terrain", mu, rng.choice([0.2, 0.55, 0.8]), zs=zs,
            ps=101325.0 * np.exp(-zs / 8000.0), tsfc=rng.uniform(262, 284),
            rh_low=0.6, clouds=(("qc", 1.5e-4, 800, 2500),
                                ("qi", 1.0e-4, 3000, 7000)))
    # Cirrus and anvil remnants only.
    for mu in np.linspace(0.12, 0.9, 6):
        add("cirrus", mu, rng.uniform(0.12, 0.3), tsfc=296.0,
            clouds=(("qi", 1.5e-4 * rng.uniform(0.3, 2.0), 8500, 13000),
                    ("qs", 5.0e-5, 8000, 12000)))
    # Moist tropics, clear: large vapour path.
    for mu in np.linspace(0.2, 1.0, 6):
        add("moist_tropics_clear", mu, rng.uniform(0.08, 0.14),
            tsfc=304.0, ttrop=192.0, rh_low=0.95, rh_high=0.55)
    # Branch-point words of the cosine zenith, on one cloudy profile.
    for name, mu in _mu_edges():
        add("edge_" + name, mu, 0.2, tsfc=295.0,
            clouds=(("qc", 4.0e-4, 1000, 3000), ("qi", 8.0e-5, 6000, 9000)))
    # Other edges: negative species (SWRAD clamps them), a dry column,
    # albedo 0 and 0.99, a giant water path that pins ALW at 3.999 in the
    # first cloudy layer.
    neg = _column(rng, clouds=(("qc", 2.0e-4, 1000, 3000),))
    for name in ("qc", "qr", "qi", "qs", "qg"):
        neg[name] = neg[name] - 3.0e-6 * (np.arange(NZ) % 3 == 0)
    # Vapour below zero on every third level (SWRAD's max(0.,QV1D)).
    neg["qv"] = np.where(np.arange(NZ) % 3 == 0, -2.0e-5, neg["qv"])
    neg.update(coszen=_f32(0.61), albedo=_f32(0.18),
               regime="edge_negative_species")
    cols.append(neg)
    add("edge_dry_column", 0.7, 0.3, tsfc=280.0, rh_low=0.0, rh_high=0.0)
    cols[-1]["qv"] = np.zeros(NZ)
    add("edge_albedo_zero", 0.66, 0.0, clouds=(("qc", 3.0e-4, 1500, 2500),))
    add("edge_albedo_0p99", 0.66, 0.99, clouds=(("qc", 3.0e-4, 1500, 2500),))
    add("edge_giant_water_path", 0.31, 0.2, tsfc=300.0,
        clouds=(("qc", 2.0e-2, 9000, 16000),))
    add("edge_giant_water_path_low_sun", 0.03, 0.2, tsfc=300.0,
        clouds=(("qc", 2.0e-2, 9000, 16000), ("qg", 1.0e-2, 4000, 8000)))
    return cols


def build_stress(rng, n):
    """Randomised columns over every regime parameter, for the stress set.

    Night, grazing and ordinary sun, albedo 0..0.95, up to five species
    with log-uniform amounts (1e-7..2e-2 kg/kg) at random heights, and
    occasional negative species words that SWRAD clamps.
    """
    cols = []
    species = ("qc", "qr", "qi", "qs", "qg")
    for j in range(n):
        clouds = []
        for name in species:
            if rng.random() < 0.5:
                z0 = rng.uniform(0.0, 14000.0)
                clouds.append((name, 10.0 ** rng.uniform(-7.0, np.log10(2e-2)),
                               z0, z0 + rng.uniform(200.0, 8000.0)))
        zs = rng.choice([0.0, rng.uniform(0.0, 4500.0)])
        inv = rng.random() < 0.3
        col = _column(rng, zs=zs, ps=101325.0 * np.exp(-zs / 8000.0),
                      tsfc=rng.uniform(235.0, 310.0),
                      lapse=rng.uniform(0.003, 0.0085),
                      ttrop=rng.uniform(190.0, 225.0),
                      rh_low=rng.uniform(0.0, 1.0),
                      rh_high=rng.uniform(0.0, 0.8),
                      inv_dt=rng.uniform(2.0, 12.0) if inv else 0.0,
                      inv_h=rng.uniform(100.0, 1200.0), clouds=clouds)
        if rng.random() < 0.05:
            for name in ("qv",) + species:
                col[name] = col[name] - 1.0e-6 * (rng.random(NZ) < 0.2)
        u = rng.random()
        if u < 0.1:
            mu = rng.uniform(-1.0, 1.0e-9)
        elif u < 0.2:
            mu = 10.0 ** rng.uniform(-9.0, -2.0)
        else:
            mu = rng.uniform(0.0, 1.0)
        col["coszen"] = _f32(mu)
        col["albedo"] = _f32(rng.uniform(0.0, 0.95))
        col["regime"] = "stress"
        cols.append(col)
    return cols


FIELDS = ("t", "p", "qv", "qc", "qr", "qi", "qs", "qg", "dz8w", "pii",
          "rho")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--stress", type=int, default=0,
                    help="write N randomised columns instead of the "
                         "named regimes")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cols = build_stress(rng, args.stress) if args.stress else build(rng)
    ncol = len(cols)
    arrays = {name: np.ascontiguousarray(
        np.stack([c[name] for c in cols], axis=1).astype(np.float32))
        for name in FIELDS}
    coszen = np.asarray([c["coszen"] for c in cols], np.float32)
    albedo = np.asarray([c["albedo"] for c in cols], np.float32)
    regime = np.asarray([c["regime"] for c in cols])
    solcon = np.asarray([c[0] for c in CASES], np.float32)
    scat = np.asarray([c[1] for c in CASES], np.float32)
    icloud = np.asarray([c[2] for c in CASES], np.int32)
    with open(args.out_dir / "dudhia-inputs.bin", "wb") as fh:
        np.asarray([ncol, NZ, len(CASES)], "<i4").tofile(fh)
        solcon.astype("<f4").tofile(fh)
        scat.astype("<f4").tofile(fh)
        icloud.astype("<i4").tofile(fh)
        for name in FIELDS:
            arrays[name].astype("<f4").tofile(fh)
        coszen.astype("<f4").tofile(fh)
        albedo.astype("<f4").tofile(fh)
    np.savez_compressed(args.out_dir / "dudhia-inputs.npz", **arrays,
                        coszen=coszen, albedo=albedo, regime=regime,
                        solcon=solcon, swrad_scat=scat, icloud=icloud)
    print(f"{ncol} columns x {NZ} levels x {len(CASES)} cases; "
          f"{len(set(regime))} regimes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
