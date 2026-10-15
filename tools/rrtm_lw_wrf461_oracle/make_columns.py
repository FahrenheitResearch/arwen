"""Deterministic input columns for the RRTM LW (ra_lw_physics=1) oracle.

Every column is built from a small physical recipe (surface pressure, model
top, a temperature profile with optional inversion and tropopause, a relative
humidity profile and a list of cloud layers) and then rounded once to
float32, so WRF and WOOF read the same bits.  Regimes covered:

* deep convection (warm moist surface, liquid + ice + rain + snow + graupel
  through the depth of the storm, overcast);
* stable nights (surface inversions over cold land, clear and with fog);
* snow and ice (arctic winter, ice cloud, falling snow, ~235 K surfaces);
* ocean (tropical, subtropical stratocumulus, mid-latitude storm track);
* high terrain (600 hPa and 700 hPa surfaces, Andes/Tibet/Rockies);
* edge cases: a 1085 hPa surface (JP clamps to 1), a 342 K surface (WRF's
  TBOUND reset to 339.99 K), a surface at exactly 340 K, a desert surface
  layer hot enough to clamp JT and INDSELF, layer pressures whose float32
  logarithm lands exactly on SETCOEF's 4.56 / 5.76 / 6.62 switches, cloud
  water putting TAUCLOUD on both sides of MM5ATM's 0.01 overcast test,
  fractional CLDFRA with no condensate, negative and subnormal mixing
  ratios, integer-kelvin temperatures (Planck fraction exactly 0),
  emissivities from 0.85 to 1.0.

Column sets vary ``nz`` and ``p_top`` so the Cavallo buffer above the model
top takes 25, 13, 3, 1 and 0 layers; one set runs ``icloud = 0`` and one
runs Kessler's registry flags (no ice species, ``warm_rain``).

usage: python make_columns.py OUT.bin [labels.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_io import ColumnSet, write_inputs  # noqa: E402

RD = 287.0
CP = 7.0 * RD / 2.0
G = 9.81
F = np.float32


def _eta_levels(nz: int, stretch: float = 1.6) -> np.ndarray:
    """nz+1 full-level eta values from 1 (surface) to 0 (top), stretched."""
    s = np.linspace(0.0, 1.0, nz + 1)
    eta = 1.0 - s ** stretch
    eta[0], eta[-1] = 1.0, 0.0
    return eta


def _esat_water(t):
    tc = t - 273.15
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def _esat_ice(t):
    tc = t - 273.15
    return 611.2 * np.exp(21.8745584 * tc / (t - 7.66))


def build_column(*, nz, p_top, ps, t_sfc, lapse=6.5e-3, t_tropo=215.0,
                 strat_lapse=-1.5e-3, inversion=None, rh=None, clouds=(),
                 cldfra=(), tsk=None, emiss=0.98, stretch=1.6,
                 qv_override=None):
    """Return a dict of BOTTOM-UP float32 fields for one column."""
    eta = _eta_levels(nz, stretch)
    p8w = p_top + eta * (ps - p_top)
    p = 0.5 * (p8w[:-1] + p8w[1:])

    # Temperature: dry-ish lapse to the tropopause, then a gentle warming.
    z_like = RD * t_sfc / G * np.log(ps / p)
    t = t_sfc - lapse * z_like
    tropo = t < t_tropo
    if np.any(tropo):
        k0 = int(np.argmax(tropo))
        z0 = z_like[k0]
        t[k0:] = t_tropo - strat_lapse * (z_like[k0:] - z0)
    if inversion is not None:
        depth_pa, delta = inversion
        mask = p > ps - depth_pa
        frac = (ps - p[mask]) / depth_pa
        t[mask] = t[mask] - delta * (1.0 - frac)
    t = np.clip(t, 175.0, 335.0)

    if rh is None:
        rh = lambda pp: 0.8 * (pp / ps) ** 2
    rel = np.clip(rh(p), 0.0, 1.0)
    e = rel * np.where(t > 273.15, _esat_water(t), _esat_ice(t))
    e = np.minimum(e, 0.5 * p)
    qv = 0.622 * e / (p - e)
    if qv_override is not None:
        qv = qv_override(p, qv)

    tv = t * (1.0 + 0.608 * qv)
    dz = RD * tv / G * np.log(p8w[:-1] / p8w[1:])
    rho = p / (RD * tv)
    pi = (p / 1.0e5) ** (RD / CP)

    # t8w: phy_prep's interior average and linear end extrapolation.
    t8w = np.empty(nz + 1)
    t8w[1:nz] = 0.5 * (t[1:] + t[:-1])
    t8w[0] = t[0] + (t[0] - t[1]) * 0.5
    t8w[nz] = t[nz - 1] + (t[nz - 1] - t[nz - 2]) * 0.5

    q = {name: np.zeros(nz) for name in ("qc", "qr", "qi", "qs", "qg")}
    for spec in clouds:
        lo, hi = spec["p"]            # pressure band in Pa (lo < hi)
        mask = (p >= lo) & (p <= hi)
        for name in ("qc", "qr", "qi", "qs", "qg"):
            if name in spec:
                q[name][mask] = spec[name]
    cf = np.zeros(nz)
    for lo, hi, value in cldfra:
        cf[(p >= lo) & (p <= hi)] = value
    for name in q:
        cf = np.where(q[name] > 1.0e-6, np.maximum(cf, 1.0), cf)

    fields = {
        "p8w": p8w, "t8w": t8w, "p3d": p, "pi3d": pi, "dz8w": dz, "t3d": t,
        "rho3d": rho, "qv": qv, "cldfra": cf, **q,
        "tsk": np.asarray(t8w[0] if tsk is None else tsk),
        "emiss": np.asarray(emiss),
    }
    return {name: np.asarray(value, dtype=np.float64).astype(np.float32)
            for name, value in fields.items()}


def _storm(ps):
    return [
        {"p": (ps - 15000.0, ps - 2000.0), "qc": 1.2e-3, "qr": 2.5e-3},
        {"p": (55000.0, ps - 15000.0), "qc": 1.8e-3, "qr": 1.2e-3,
         "qg": 2.0e-3},
        {"p": (30000.0, 55000.0), "qi": 4.0e-4, "qs": 1.5e-3, "qg": 1.0e-3,
         "qc": 3.0e-4},
        {"p": (15000.0, 30000.0), "qi": 2.0e-4, "qs": 3.0e-4},
    ]


def regime_columns(nz: int, p_top: float):
    """The physical regimes, as (label, kwargs) pairs."""
    cols = []
    add = lambda label, **kw: cols.append((label, kw))
    moist = lambda top: (lambda pp: np.where(pp > top, 0.85, 0.15))
    # deep convection
    add("convective-plains", ps=96500.0, t_sfc=303.0, tsk=309.5,
        rh=moist(25000.0), clouds=_storm(96500.0), t_tropo=208.0)
    add("convective-tropical-mcs", ps=100800.0, t_sfc=300.5, tsk=301.2,
        rh=lambda pp: np.full_like(pp, 0.92), clouds=_storm(100800.0),
        t_tropo=195.0, emiss=0.98)
    add("convective-anvil-only", ps=99000.0, t_sfc=297.0, tsk=296.0,
        rh=moist(20000.0),
        clouds=[{"p": (18000.0, 30000.0), "qi": 8.0e-5, "qs": 4.0e-5}],
        t_tropo=205.0)
    # stable nights
    add("stable-night-clear", ps=98500.0, t_sfc=268.0, tsk=262.0,
        inversion=(6000.0, 9.0), rh=lambda pp: 0.5 * (pp / 98500.0) ** 3,
        emiss=0.95, t_tropo=218.0)
    add("stable-night-fog", ps=101500.0, t_sfc=276.0, tsk=274.5,
        inversion=(4000.0, 5.0), rh=lambda pp: np.where(pp > 97000.0, 1.0,
                                                         0.3),
        clouds=[{"p": (98000.0, 101500.0), "qc": 2.0e-4}], emiss=0.97)
    add("stable-night-dry-desert", ps=91000.0, t_sfc=282.0, tsk=271.0,
        inversion=(8000.0, 12.0), rh=lambda pp: np.full_like(pp, 0.08),
        emiss=0.85)
    # snow and ice
    add("arctic-winter-clear", ps=103000.0, t_sfc=236.0, tsk=233.0,
        inversion=(9000.0, -14.0), rh=lambda pp: np.full_like(pp, 0.7),
        t_tropo=212.0, emiss=0.99)
    add("arctic-ice-cloud-snow", ps=101000.0, t_sfc=248.0, tsk=246.0,
        rh=lambda pp: np.full_like(pp, 0.9),
        clouds=[{"p": (70000.0, 95000.0), "qi": 6.0e-5, "qs": 2.0e-4}],
        emiss=0.99)
    add("midlat-snowstorm", ps=99500.0, t_sfc=268.5, tsk=268.0,
        rh=lambda pp: np.where(pp > 40000.0, 0.97, 0.4),
        clouds=[{"p": (60000.0, 99500.0), "qc": 1.0e-4, "qs": 6.0e-4,
                 "qi": 1.0e-4},
                {"p": (40000.0, 60000.0), "qi": 2.0e-4, "qs": 3.0e-4}],
        emiss=0.99)
    add("freezing-rain", ps=100000.0, t_sfc=271.0, tsk=271.5,
        inversion=(12000.0, -9.0), rh=lambda pp: np.full_like(pp, 0.95),
        clouds=[{"p": (60000.0, 100000.0), "qc": 3.0e-4, "qr": 5.0e-4}])
    # ocean
    add("ocean-tropical-clear", ps=101200.0, t_sfc=300.0, tsk=302.0,
        rh=lambda pp: np.where(pp > 80000.0, 0.85, 0.25), t_tropo=192.0,
        emiss=0.98)
    add("ocean-stratocumulus", ps=101800.0, t_sfc=289.0, tsk=290.5,
        inversion=(9000.0, -8.0),
        rh=lambda pp: np.where(pp > 92000.0, 0.95, 0.2),
        clouds=[{"p": (90000.0, 94000.0), "qc": 4.0e-4}],
        cldfra=[(94000.0, 96000.0, 0.4)], emiss=0.98)
    add("ocean-storm-track", ps=98000.0, t_sfc=281.0, tsk=283.0,
        rh=lambda pp: np.full_like(pp, 0.9),
        clouds=[{"p": (50000.0, 90000.0), "qc": 5.0e-4, "qr": 2.0e-4},
                {"p": (30000.0, 50000.0), "qi": 1.0e-4, "qs": 1.0e-4}],
        emiss=0.98)
    # high terrain
    add("high-terrain-tibet", ps=58000.0, t_sfc=278.0, tsk=290.0,
        rh=lambda pp: np.full_like(pp, 0.35), t_tropo=205.0, emiss=0.93)
    add("high-terrain-rockies-cloud", ps=70500.0, t_sfc=272.0, tsk=270.0,
        rh=lambda pp: np.full_like(pp, 0.85),
        clouds=[{"p": (45000.0, 65000.0), "qi": 5.0e-5, "qs": 2.5e-4,
                 "qc": 5.0e-5}], emiss=0.97)
    add("high-terrain-andes-night", ps=61000.0, t_sfc=262.0, tsk=255.0,
        inversion=(5000.0, 8.0), rh=lambda pp: np.full_like(pp, 0.2),
        emiss=0.96)
    # edge cases
    add("edge-1085hPa-anticyclone", ps=108500.0, t_sfc=228.0, tsk=226.0,
        inversion=(10000.0, -10.0), rh=lambda pp: np.full_like(pp, 0.6),
        emiss=0.99)
    add("edge-tbound-reset-342K", ps=97000.0, t_sfc=318.0, tsk=342.0,
        rh=lambda pp: np.full_like(pp, 0.1), emiss=0.9, lapse=9.0e-3)
    add("edge-tbound-exactly-340K", ps=97000.0, t_sfc=316.0, tsk=340.0,
        rh=lambda pp: np.full_like(pp, 0.12), emiss=0.9, lapse=9.0e-3)
    add("edge-hot-desert-jt-clamp", ps=100500.0, t_sfc=334.0, tsk=338.0,
        rh=lambda pp: np.full_like(pp, 0.05), emiss=0.86, lapse=9.5e-3)
    add("edge-taucloud-threshold", ps=100000.0, t_sfc=285.0, tsk=285.0,
        rh=lambda pp: np.full_like(pp, 0.7),
        clouds=[{"p": (80000.0, 82000.0), "qc": 1.0e-7},
                {"p": (70000.0, 72000.0), "qc": 4.0e-7},
                {"p": (60000.0, 62000.0), "qi": 9.0e-7}],
        cldfra=[(50000.0, 55000.0, 0.25), (40000.0, 45000.0, 0.6)])
    add("edge-negative-subnormal-q", ps=99000.0, t_sfc=290.0, tsk=290.0,
        rh=lambda pp: np.full_like(pp, 0.5),
        clouds=[{"p": (85000.0, 90000.0), "qc": -1.0e-6, "qr": 1.0e-40,
                 "qi": 1.4e-45, "qs": -0.0},
                {"p": (70000.0, 75000.0), "qs": 1.0e-38, "qr": 3.0e-39}],
        qv_override=lambda pp, qv: np.where(pp < 30000.0, -1.0e-7,
                                            np.where(pp < 40000.0, 1.0e-13,
                                                     qv)))
    add("edge-overcast-everywhere", ps=100000.0, t_sfc=280.0, tsk=280.0,
        rh=lambda pp: np.full_like(pp, 0.99),
        cldfra=[(0.0, 2.0e5, 1.0)])
    add("edge-integer-kelvin", ps=100000.0, t_sfc=290.0, tsk=290.0,
        rh=lambda pp: np.full_like(pp, 0.5))
    return cols


def _assemble(name, nz, p_top, specs, *, icloud=1, warm_rain=False,
              flags=(True,) * 6, yr=2026, post=None):
    labels, columns = [], []
    for label, kw in specs:
        col = build_column(nz=nz, p_top=p_top, **kw)
        if post is not None:
            post(label, col)
        labels.append(label)
        columns.append(col)
    fields = {key: np.stack([c[key] for c in columns])
              for key in columns[0]}
    return ColumnSet(name=name, p_top=float(np.float32(p_top)), yr=yr,
                     icloud=icloud, warm_rain=warm_rain, flags=flags,
                     fields=fields, labels=labels)


def _logf_boundary_pressure(target: float) -> np.float32:
    """A float32 layer pressure (mb) whose float32 log is exactly ``target``.

    WOOF's own float32 logf is the function SETCOEF's LOG resolves to on
    this toolchain; the search walks float32 neighbours of exp(target).
    """
    from gpuwm.core.noahmp_libm import logf
    want = np.float32(target)
    p = np.float32(np.exp(np.float64(target)))
    for _ in range(64):
        got = logf(p)
        if got == want:
            return p
        p = np.nextafter(p, np.float32(np.inf if got < want else 0.0))
    raise RuntimeError(f"no float32 pressure has logf == {target}")


def _post_edges(label, col):
    if label == "edge-integer-kelvin":
        col["t3d"][:] = np.round(col["t3d"])
        col["t8w"][:] = np.round(col["t8w"])
        col["tsk"][...] = np.float32(290.0)


def _post_switch_pressures(label, col):
    """Put SETCOEF's three PLOG switches exactly on three layer pressures."""
    if label != "edge-plog-switches":
        return
    p = col["p3d"]
    for target in (6.62, 5.76, 4.56):
        mb = _logf_boundary_pressure(target)
        pa = np.float32(mb * np.float32(100.0))
        # RRTMLWRAD forms P1D = P3D/100.; nudge P3D until that division
        # returns the boundary pressure exactly.
        for _ in range(64):
            back = np.float32(pa / np.float32(100.0))
            if back == mb:
                break
            pa = np.nextafter(pa, np.float32(np.inf if back < mb else 0.0))
        k = int(np.argmin(np.abs(p.astype(np.float64) - float(pa))))
        p[k] = pa


def build_sets() -> list[ColumnSet]:
    sets = []
    base = regime_columns(50, 5000.0)
    base.append(("edge-plog-switches",
                 dict(ps=100000.0, t_sfc=288.0, tsk=288.0,
                      rh=lambda pp: np.full_like(pp, 0.6))))

    def post(label, col):
        _post_edges(label, col)
        _post_switch_pressures(label, col)

    sets.append(_assemble("A-nz50-ptop50hPa", 50, 5000.0, base, post=post))
    pick = lambda names: [c for c in regime_columns(0, 0) if c[0] in names]
    sets.append(_assemble(
        "B-nz40-ptop100hPa", 40, 10000.0,
        pick({"convective-plains", "stable-night-clear", "arctic-winter-clear",
              "ocean-stratocumulus", "high-terrain-tibet",
              "edge-1085hPa-anticyclone", "midlat-snowstorm",
              "ocean-tropical-clear", "edge-tbound-reset-342K",
              "stable-night-fog", "freezing-rain",
              "convective-anvil-only"}), post=post))
    sets.append(_assemble(
        "C-nz60-ptop10hPa", 60, 1000.0,
        pick({"convective-tropical-mcs", "stable-night-dry-desert",
              "arctic-ice-cloud-snow", "ocean-storm-track",
              "high-terrain-rockies-cloud", "edge-hot-desert-jt-clamp",
              "edge-overcast-everywhere", "high-terrain-andes-night",
              "edge-taucloud-threshold", "convective-plains",
              "edge-integer-kelvin", "stable-night-clear"}), post=post))
    sets.append(_assemble(
        "D-nz35-ptop1hPa-nobuffer", 35, 100.0,
        pick({"convective-plains", "arctic-winter-clear", "ocean-tropical-clear",
              "high-terrain-tibet", "midlat-snowstorm",
              "stable-night-clear", "edge-negative-subnormal-q",
              "ocean-stratocumulus"}), post=post, yr=2019))
    sets.append(_assemble(
        "E-nz45-ptop2hPa-onebuffer", 45, 200.0,
        pick({"convective-tropical-mcs", "stable-night-fog",
              "arctic-ice-cloud-snow", "ocean-storm-track",
              "high-terrain-andes-night", "edge-tbound-exactly-340K",
              "freezing-rain", "edge-1085hPa-anticyclone"}), post=post,
        yr=2031))
    sets.append(_assemble(
        "F-icloud0", 50, 5000.0,
        pick({"convective-plains", "midlat-snowstorm", "ocean-storm-track",
              "edge-overcast-everywhere", "stable-night-fog",
              "arctic-ice-cloud-snow", "high-terrain-rockies-cloud",
              "convective-tropical-mcs"}), icloud=0, post=post))

    def kessler(label, col):
        for name in ("qi", "qs", "qg"):
            col[name][:] = 0.0
    sets.append(_assemble(
        "G-kessler-flags", 50, 5000.0,
        pick({"convective-plains", "ocean-storm-track", "freezing-rain",
              "stable-night-fog", "ocean-stratocumulus",
              "convective-tropical-mcs"}),
        warm_rain=True, flags=(True, True, True, False, False, False),
        post=kessler))
    return sets


def main(argv):
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    sets = build_sets()
    write_inputs(argv[1], sets)
    total = sum(cs.ncol for cs in sets)
    labels = {cs.name: {"p_top": cs.p_top, "nz": cs.nz, "icloud": cs.icloud,
                        "warm_rain": cs.warm_rain, "yr": cs.yr,
                        "columns": cs.labels} for cs in sets}
    if len(argv) == 3:
        Path(argv[2]).write_text(json.dumps(labels, indent=1) + "\n")
    print(f"wrote {len(sets)} sets, {total} columns to {argv[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
