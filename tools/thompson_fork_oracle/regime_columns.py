"""Synthetic model columns for the fork Thompson column oracle.

Each regime is an analytic sounding (hydrostatic, float64, then rounded to
float32 once) with a hydrometeor and aerosol profile typical of that
weather, plus seeded variants so every regime contributes several distinct
columns.  The saved real-data columns of
tests/data/thompson_real_columns_wrf461.npz (42 columns of one convective
case, chosen per process regime) are added by ``all_columns``.

Regimes: deep convection, stable night with fog, snow, ocean stratocumulus,
high terrain mixed phase, melting layer, cirrus anvil with homogeneous
freezing, dry desert, and an edge-case family (condensate exactly at and one
float32 step either side of R1 = 1e-12, supersaturated clear air, orphan
numbers, zero aerosol, a huge droplet number, rain without number, warm
graupel).

Column arrays follow tools/thompson_real_column_parity's column-file
contract: ``p, th, qv..nifa`` are (ncol, nz) on mass levels, ``geop`` and
``w`` are (ncol, nz + 1) on full levels, ``nwfa2d`` and ``nifa2d`` are
(ncol,).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "tests" / "data" / "thompson_real_columns_wrf461.npz"

NZ = 49
G = 9.81
RD = 287.0
CP = 1004.5
RCP = RD / CP
P0 = 1.0e5
R1 = np.float32(1.0e-12)

SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc",
           "nwfa", "nifa")


def _levels(z_sfc):
    """Full-level heights (m above sea level), stretched: ~40 m at the
    ground to ~900 m near the 20 km top."""
    s = np.linspace(0.0, 1.0, NZ + 1)
    depth = 20000.0
    z = depth * (0.35 * s ** 2.0 + 0.65 * s ** 1.0) ** 1.6
    return z_sfc + z * (depth - 0.0) / z[-1]


def _sat_mixing_ratio(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    esi = 611.2 * np.exp(21.8745584 * (t - 273.16) / (t - 7.66))
    es = np.where(t < 273.15, esi, es)
    return 0.622 * es / np.maximum(p - es, 1.0)


def _sounding(z_full, ps, t_sfc, lapse, inversion=None, tropo=11000.0,
              rh=None):
    """Mass-level p, T, qv from a temperature profile and RH profile."""
    zc = 0.5 * (z_full[1:] + z_full[:-1])
    zr = zc - z_full[0]
    t = t_sfc - lapse * np.minimum(zr, tropo - z_full[0])
    t = np.where(zr > tropo - z_full[0], t, t)
    if inversion is not None:
        base, top, dtemp = inversion
        frac = np.clip((zr - base) / max(top - base, 1.0), 0.0, 1.0)
        t = t + dtemp * frac * (zr >= base) - lapse * 0.0
        t = np.where(zr < base, t_sfc - (lapse * 0.3) * zr, t)
    t = np.maximum(t, 205.0)
    # Hydrostatic pressure on full then mass levels.
    pf = np.empty_like(z_full)
    pf[0] = ps
    tf = np.interp(z_full, zc, t)
    for k in range(1, len(z_full)):
        tm = 0.5 * (tf[k] + tf[k - 1])
        pf[k] = pf[k - 1] * np.exp(-G * (z_full[k] - z_full[k - 1]) / (RD * tm))
    p = np.sqrt(pf[1:] * pf[:-1])
    if rh is None:
        rh = np.full_like(t, 0.5)
    qv = rh * _sat_mixing_ratio(t, p)
    return zc, p, t, qv


def _bump(z, centre, width, peak):
    return peak * np.exp(-0.5 * ((z - centre) / width) ** 2)


def _column(z_full, p, t, qv, w, hyd, nwfa2d, nifa2d):
    th = t * (P0 / p) ** RCP
    col = {"p": p, "th": th, "geop": z_full * G, "w": w, "qv": qv}
    for name in SPECIES[1:]:
        col[name] = hyd.get(name, np.zeros_like(p))
    col["nwfa2d"] = np.float64(nwfa2d)
    col["nifa2d"] = np.float64(nifa2d)
    return col


def _aerosol(zc, z_sfc, surface, scale_height):
    return surface * np.exp(-(zc - z_sfc) / scale_height)


def regime(name, rng):
    """One column of the named regime; ``rng`` varies amplitudes."""
    u = lambda lo, hi: float(rng.uniform(lo, hi))  # noqa: E731
    if name == "convective":
        z_sfc = u(0.0, 400.0)
        zf = _levels(z_sfc)
        rh = np.clip(np.interp(_levels(0)[:-1], [0, 3000, 9000, 20000],
                               [0.95, 0.9, 0.7, 0.2]), 0, 1.02)
        zc, p, t, qv = _sounding(zf, u(97000, 100500), u(298, 306), 0.0068,
                                 rh=rh)
        zr = zc - z_sfc
        hyd = {
            "qc": _bump(zr, u(2500, 4500), 1500, u(5e-4, 2.5e-3)),
            "qr": _bump(zr, u(1000, 2500), 1600, u(5e-4, 4e-3)),
            "qg": _bump(zr, u(5000, 8000), 1800, u(5e-4, 5e-3)),
            "qs": _bump(zr, u(8000, 10500), 1500, u(2e-4, 2e-3)),
            "qi": _bump(zr, u(10500, 12500), 1200, u(2e-5, 3e-4)),
        }
        hyd["nc"] = hyd["qc"] * u(1e11, 6e11)
        hyd["nr"] = hyd["qr"] * u(5e6, 1e8)
        hyd["ni"] = hyd["qi"] * u(1e9, 3e10)
        aer = u(0.6e9, 3e9)
        hyd["nwfa"] = _aerosol(zc, z_sfc, aer, 2500)
        hyd["nifa"] = _aerosol(zc, z_sfc, u(1e5, 2e6), 3000)
        wf = _bump(zf - z_sfc, 7000, 3500, u(8, 30))
        return _column(zf, p, t, qv, wf, hyd, aer * u(0.5, 1.5) * 1e-3, 0.0)
    if name == "stable_night":
        z_sfc = u(100, 600)
        zf = _levels(z_sfc)
        zc0 = 0.5 * (zf[1:] + zf[:-1]) - z_sfc
        rh = np.where(zc0 < 150, u(1.0, 1.006), np.where(zc0 < 2500, 0.6, 0.3))
        zc, p, t, qv = _sounding(zf, u(95000, 99000), u(268, 284), 0.006,
                                 inversion=(0.0, 300.0, u(4, 10)), rh=rh)
        zr = zc - z_sfc
        hyd = {"qc": np.where(zr < u(80, 200), u(5e-5, 4e-4), 0.0)}
        hyd["nc"] = hyd["qc"] * u(2e11, 1e12)
        if t[0] < 273.15:
            hyd["qi"] = np.where(zr < 300, u(1e-7, 3e-6), 0.0)
            hyd["ni"] = hyd["qi"] * u(1e9, 5e10)
        aer = u(2e9, 8e9)
        hyd["nwfa"] = _aerosol(zc, z_sfc, aer, 800)
        hyd["nifa"] = _aerosol(zc, z_sfc, u(5e5, 5e6), 900)
        wf = rng.normal(0.0, 0.02, zf.shape)
        return _column(zf, p, t, qv, wf, hyd, aer * 2e-3, 0.0)
    if name == "snow":
        z_sfc = u(0, 800)
        zf = _levels(z_sfc)
        zc, p, t, qv = _sounding(zf, u(96000, 101500), u(250, 266), 0.005,
                                 rh=np.full(NZ, u(0.85, 1.03)))
        zr = zc - z_sfc
        hyd = {
            "qs": _bump(zr, u(500, 2500), 2000, u(2e-4, 1.5e-3)),
            "qi": _bump(zr, u(3000, 6000), 1500, u(1e-5, 2e-4)),
            "qc": _bump(zr, u(300, 1200), 400, u(0, 2e-4)),
            "qg": _bump(zr, u(500, 1500), 600, u(0, 2e-4)),
        }
        hyd["ni"] = hyd["qi"] * u(5e8, 1e10)
        hyd["nc"] = hyd["qc"] * u(5e10, 3e11)
        aer = u(2e8, 1.5e9)
        hyd["nwfa"] = _aerosol(zc, z_sfc, aer, 1500)
        hyd["nifa"] = _aerosol(zc, z_sfc, u(1e6, 1e7), 2000)
        wf = _bump(zf - z_sfc, 2000, 1500, u(0.05, 1.0))
        return _column(zf, p, t, qv, wf, hyd, aer * 1e-3, 0.0)
    if name == "ocean":
        zf = _levels(0.0)
        zc0 = 0.5 * (zf[1:] + zf[:-1])
        top = u(700, 1300)
        rh = np.where(zc0 < top, u(0.97, 1.004), u(0.2, 0.4))
        zc, p, t, qv = _sounding(zf, u(100800, 102500), u(286, 301), 0.0065,
                                 inversion=(top, top + 300, u(4, 9)), rh=rh)
        hyd = {"qc": np.where((zc > top - 450) & (zc < top), u(1e-4, 8e-4), 0.0),
               "qr": np.where(zc < top - 100, u(1e-7, 5e-5), 0.0)}
        hyd["nc"] = hyd["qc"] * u(3e10, 2e11)
        hyd["nr"] = hyd["qr"] * u(1e7, 5e8)
        aer = u(1e8, 6e8)
        hyd["nwfa"] = _aerosol(zc, 0.0, aer, 1200)
        hyd["nifa"] = _aerosol(zc, 0.0, u(1e4, 2e5), 2000)
        wf = _bump(zf, top - 300, 300, u(0.05, 0.6))
        return _column(zf, p, t, qv, wf, hyd, aer * 5e-3, 0.0)
    if name == "high_terrain":
        z_sfc = u(2500, 4200)
        zf = _levels(z_sfc)
        ps = 101325.0 * np.exp(-z_sfc / 8000.0)
        zc, p, t, qv = _sounding(zf, ps, u(266, 280), 0.0065,
                                 rh=np.full(NZ, u(0.8, 1.02)))
        zr = zc - z_sfc
        hyd = {
            "qc": _bump(zr, u(400, 1200), 500, u(1e-4, 1e-3)),
            "qs": _bump(zr, u(800, 2500), 1000, u(1e-4, 1e-3)),
            "qg": _bump(zr, u(500, 1500), 700, u(5e-5, 1e-3)),
            "qi": _bump(zr, u(2500, 4500), 1000, u(1e-6, 1e-4)),
            "qr": _bump(zr, u(0, 400), 300, u(0, 3e-4)),
        }
        hyd["nc"] = hyd["qc"] * u(5e10, 3e11)
        hyd["ni"] = hyd["qi"] * u(5e8, 2e10)
        hyd["nr"] = hyd["qr"] * u(1e6, 1e8)
        aer = u(3e8, 1e9)
        hyd["nwfa"] = _aerosol(zc, z_sfc, aer, 1500)
        hyd["nifa"] = _aerosol(zc, z_sfc, u(5e5, 5e6), 2000)
        wf = _bump(zf - z_sfc, 1500, 1200, u(0.5, 4.0))
        return _column(zf, p, t, qv, wf, hyd, aer * 1e-3, 0.0)
    if name == "melting_layer":
        z_sfc = u(0, 500)
        zf = _levels(z_sfc)
        t_sfc = u(276, 283)
        zc, p, t, qv = _sounding(zf, u(97000, 101000), t_sfc, 0.0065,
                                 rh=np.full(NZ, u(0.9, 1.0)))
        zr = zc - z_sfc
        z0 = (t_sfc - 273.15) / 0.0065
        hyd = {
            "qs": np.where(zr > z0 - 400, _bump(zr, z0 + 800, 1500, u(3e-4, 2e-3)), 0.0),
            "qg": np.where(zr > z0 - 600, _bump(zr, z0 + 500, 1000, u(1e-4, 2e-3)), 0.0),
            "qr": np.where(zr < z0 + 200, u(2e-4, 2e-3) * np.ones_like(zr), 0.0),
            "qc": _bump(zr, z0, 400, u(0, 3e-4)),
        }
        hyd["nr"] = hyd["qr"] * u(1e6, 3e7)
        hyd["nc"] = hyd["qc"] * u(5e10, 3e11)
        aer = u(5e8, 2e9)
        hyd["nwfa"] = _aerosol(zc, z_sfc, aer, 1500)
        hyd["nifa"] = _aerosol(zc, z_sfc, u(5e5, 5e6), 2000)
        wf = _bump(zf - z_sfc, 3000, 2000, u(0.1, 2.0))
        return _column(zf, p, t, qv, wf, hyd, aer * 1e-3, 0.0)
    if name == "cirrus_anvil":
        z_sfc = u(0, 300)
        zf = _levels(z_sfc)
        zc0 = 0.5 * (zf[1:] + zf[:-1]) - z_sfc
        rh = np.where(zc0 > 9000, u(1.0, 1.25), 0.4)
        zc, p, t, qv = _sounding(zf, u(98000, 101000), u(295, 303), 0.0072,
                                 tropo=13500.0, rh=rh)
        zr = zc - z_sfc
        hyd = {
            "qi": _bump(zr, u(10000, 12500), 1200, u(5e-5, 6e-4)),
            "qs": _bump(zr, u(9000, 11000), 1200, u(1e-5, 5e-4)),
            # Supercooled cloud right at the homogeneous-freezing level.
            "qc": np.where((t > 233.0) & (t < 238.5), u(1e-5, 4e-4), 0.0),
        }
        hyd["ni"] = hyd["qi"] * u(5e9, 1e11)
        hyd["nc"] = hyd["qc"] * u(5e10, 5e11)
        aer = u(5e8, 2e9)
        hyd["nwfa"] = _aerosol(zc, z_sfc, aer, 2500)
        hyd["nifa"] = _aerosol(zc, z_sfc, u(1e6, 1e7), 4000)
        wf = _bump(zf - z_sfc, 11000, 2000, u(0.2, 6.0))
        return _column(zf, p, t, qv, wf, hyd, aer * 1e-3, 0.0)
    if name == "dry_desert":
        z_sfc = u(200, 1500)
        zf = _levels(z_sfc)
        ps = 101325.0 * np.exp(-z_sfc / 8000.0)
        zc, p, t, qv = _sounding(zf, ps, u(305, 318), 0.0095,
                                 rh=np.full(NZ, u(0.03, 0.15)))
        hyd = {"nwfa": _aerosol(zc, z_sfc, u(5e9, 1e10), 3000),
               "nifa": _aerosol(zc, z_sfc, u(1e7, 5e7), 3000)}
        wf = rng.normal(0.0, 0.5, zf.shape)
        return _column(zf, p, t, qv, wf, hyd, 1e7, 0.0)
    raise KeyError(name)


REGIMES = ("convective", "stable_night", "snow", "ocean", "high_terrain",
           "melting_layer", "cirrus_anvil", "dry_desert")


def edge_columns(rng):
    """Edge cases built on a convective and a snow column."""
    out = []
    base = regime("convective", rng)
    nxt_up = lambda x: np.nextafter(np.float32(x), np.float32(1))  # noqa: E731
    nxt_dn = lambda x: np.nextafter(np.float32(x), np.float32(0))  # noqa: E731
    # Condensate at R1 and one float32 step either side, every species.
    for label, value in (("at R1", R1), ("one step above R1", nxt_up(R1)),
                         ("one step below R1", nxt_dn(R1))):
        c = {k: (v.copy() if isinstance(v, np.ndarray) else v)
             for k, v in base.items()}
        for s in ("qc", "qr", "qi", "qs", "qg"):
            c[s] = np.where(np.arange(NZ) % 5 == ("qc", "qr", "qi", "qs", "qg").index(s),
                            np.float32(value), c[s])
        out.append((f"edge: condensate {label}", c))
    # Supersaturated clear air: activation from nothing.
    c = regime("convective", rng)
    for s in ("qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni"):
        c[s] = np.zeros_like(c[s])
    t = c["th"] * (c["p"] / P0) ** RCP
    c["qv"] = 1.04 * _sat_mixing_ratio(t, c["p"])
    out.append(("edge: supersaturated clear air", c))
    # Orphan numbers: number without mass.
    c = regime("snow", rng)
    for s, n in (("qi", "ni"), ("qr", "nr"), ("qc", "nc")):
        c[n] = np.where(c[s] > 1e-6, c[n], 1.0e5)
        c[s] = np.where(np.arange(NZ) % 2 == 0, 0.0, c[s])
    out.append(("edge: orphan numbers", c))
    # Zero aerosol everywhere.
    c = regime("convective", rng)
    c["nwfa"] = np.zeros_like(c["nwfa"])
    c["nifa"] = np.zeros_like(c["nifa"])
    c["nwfa2d"] = np.float64(0.0)
    out.append(("edge: zero aerosol", c))
    # Huge droplet number, rain with zero number, graupel in warm air.
    c = regime("convective", rng)
    c["nc"] = np.where(c["qc"] > 1e-8, 2.0e10, c["nc"])
    c["nr"] = np.zeros_like(c["nr"])
    c["qg"] = c["qg"] + _bump(np.arange(NZ, dtype=float), 2, 2, 1e-3)
    out.append(("edge: huge nc, rain without number, warm graupel", c))
    # Aerosol at the fork's ceiling and floor.
    c = regime("ocean", rng)
    c["nwfa"] = np.where(np.arange(NZ) % 2 == 0, 9999.0e6, 11.0e6 * 0.5)
    c["nifa"] = np.where(np.arange(NZ) % 3 == 0, 9999.0e6, 0.0)
    out.append(("edge: aerosol at the ceiling and floor", c))
    # Heavy rain with huge number (size bound) and tiny number.
    c = regime("melting_layer", rng)
    c["nr"] = np.where(np.arange(NZ) % 2 == 0, c["qr"] * 1e12, c["qr"] * 1e2)
    out.append(("edge: rain number at both size bounds", c))
    # Ice with huge and tiny numbers (terminal ice bound).
    c = regime("cirrus_anvil", rng)
    c["ni"] = np.where(np.arange(NZ) % 2 == 0, c["qi"] * 1e14, c["qi"] * 1e3)
    out.append(("edge: ice number at both size bounds", c))
    return out


def synthetic_columns(per_regime=4, seed=20261007):
    rng = np.random.default_rng(seed)
    cols = []
    for name in REGIMES:
        for i in range(per_regime):
            cols.append((f"{name} #{i}", regime(name, rng)))
    cols.extend(edge_columns(rng))
    return cols


def stack(named):
    """[(label, col)] -> column-file dict of float32 arrays plus labels."""
    labels = [label for label, _ in named]
    out = {}
    for key in ("p", "th", "geop", "w", *SPECIES):
        out[key] = np.stack([np.asarray(c[key], np.float64)
                             for _, c in named]).astype(np.float32)
    for key in ("nwfa2d", "nifa2d"):
        out[key] = np.array([float(c[key]) for _, c in named], np.float32)
    for s in SPECIES[1:]:
        out[s] = np.maximum(out[s], np.float32(0.0))
    return out, labels


def real_columns():
    z = np.load(REAL)
    cols = {k[4:]: z[k] for k in z.files if k.startswith("col_")}
    labels = [f"real: {s}" for s in z["labels"]]
    return cols, labels, float(z["dt"])


def all_columns(per_regime=4, seed=20261007):
    """The real 42 columns followed by the synthetic set, one batch."""
    real, real_labels, _ = real_columns()
    syn, syn_labels = stack(synthetic_columns(per_regime, seed))
    cols = {k: np.concatenate([np.asarray(real[k], np.float32),
                               syn[k]]) for k in syn}
    return cols, real_labels + syn_labels
