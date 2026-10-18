"""Regime decks for the gsl_wrf39 MYNN surface-layer column oracle.

Writes the text decks run_columns.F90 reads.  One deck per (ISFTCFLX, DX)
pair.  Header line ``NCOL NSTEP ISFTCFLX DX``, then one line per column::

    xland snowh u1 v1 t1 qv1 p1 rho1 dz1 u2 v2 dz2 psfc tsk pblh mavail
    hfx qfx znt qsfc ust mol

Every value is a float32 written with 9 significant digits, so the Fortran
list-directed read and :func:`load_deck` give the same words.  Step 1 enters
with the SFCLAY_mynn wrapper's itimestep == 1 seeding (fork :344-351)
already applied: UST = max(0.04*|V1|, 0.001), MOL = 0, QSFC = QV1/(1+QV1).
A column that sets ``qsfc`` itself (the land QSFC <= 0 edge) keeps it.

Regimes (deck ``main``, ISFTCFLX = 0, DX = 3000, the HRRR door's setting):
daytime convective land over four roughness classes, stable night land
from weakly to very stable (the Richardson clamp at 50 and the z/L cap at
50), snow cover and sea ice (Andreas 2002 z_t/z_q, the ice-saturation
branch), open ocean from calm to 25 m/s, cold and tropical (Charnock
z0 and the COARE 3.0 z_t/z_q), high terrain at 650-750 hPa, and edge
cases: first level below 7 m with and without a second level inside
7-13 m, first level above 13 m, a land column entering with QSFC <= 0,
XLAND exactly 1.5, an exactly neutral column with a nonzero MOL, zero
moisture availability, a skin colder than freezing under air warmer than
freezing, and near-calm air.

Decks ``isftcflx1/2/3`` rerun the ocean columns (plus XLAND = 1.5) through
the other over-water roughness forms, and deck ``dx13km`` reruns a
cross-section at DX = 13545 m (RAP), where the Mahrt-Sun subgrid wind is
nonzero.

    python gsl_wrf39_columns.py OUTDIR
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

F = np.float32
R, RV = F(287.0), F(461.6)
EP1 = F(RV / R - F(1.0))
FIELDS = ("xland", "snowh", "u1", "v1", "t1", "qv1", "p1", "rho1", "dz1",
          "u2", "v2", "dz2", "psfc", "tsk", "pblh", "mavail", "hfx", "qfx",
          "znt", "qsfc", "ust", "mol")
NSTEP = 8


def _qsat(t, p):
    """Bolton saturation mixing ratio, only to pick realistic humidities."""
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / (p - es)


def _column(*, xland, znt, u1, t1, tsk, rh=0.6, qv1=None, p1=99000.0,
            psfc=None, dz1=16.0, dz2=20.0, snowh=0.0, pblh=None,
            mavail=None, hfx=None, qfx=0.0, ust=None, mol=0.0, qsfc=None,
            v1=None):
    v1 = 0.3 * u1 if v1 is None else v1
    stable = tsk < t1
    if qv1 is None:
        qv1 = rh * _qsat(t1, p1)
    if psfc is None:
        # Hydrostatic over half the first layer, rounded to a whole Pa.
        psfc = round(p1 * (1.0 + 9.81 * 0.5 * dz1 / (287.0 * t1)))
    col = {
        "xland": xland, "snowh": snowh, "u1": u1, "v1": v1, "t1": t1,
        "qv1": qv1, "p1": p1, "dz1": dz1, "u2": 1.3 * u1, "v2": 1.3 * v1,
        "dz2": dz2, "psfc": psfc, "tsk": tsk,
        "pblh": (150.0 if stable else 1200.0) if pblh is None else pblh,
        "mavail": (1.0 if xland > 1.5 else 0.4) if mavail is None else mavail,
        "hfx": (-15.0 if stable else 150.0) if hfx is None else hfx,
        "qfx": qfx, "znt": znt, "mol": mol,
    }
    col = {k: F(v) for k, v in col.items()}
    col["rho1"] = F(col["p1"] / (R * col["t1"] * (
        F(1.0) + EP1 * col["qv1"] / (F(1.0) + col["qv1"]))))
    wsp = F(np.sqrt(col["u1"] * col["u1"] + col["v1"] * col["v1"]))
    col["ust"] = F(max(F(0.04) * wsp, F(0.001))) if ust is None else F(ust)
    col["qsfc"] = (F(col["qv1"] / (F(1.0) + col["qv1"])) if qsfc is None
                   else F(qsfc))
    return col


LAND = (("forest", 0.8), ("crop", 0.2), ("grass", 0.075), ("bare", 0.01))


def main_deck():
    names, cols = [], []

    def add(name, **kw):
        names.append(name)
        cols.append(_column(**kw))

    # Daytime convective land: skin warmer than air, surface heat flux in.
    for label, z0 in LAND:
        for u1 in (0.5, 2.0, 6.0):
            for dt, hfx in ((3.0, 120.0), (8.0, 280.0), (15.0, 450.0)):
                add(f"convective_{label}_u{u1:g}_dt+{dt:g}", xland=1.0,
                    znt=z0, u1=u1, t1=300.0, tsk=300.0 + dt, rh=0.45,
                    hfx=hfx, qfx=1.0e-4, pblh=1800.0)
    # Stable night land, weakly to very stable.
    for label, z0 in LAND:
        for u1 in (0.5, 1.5, 4.0, 8.0):
            for dt in (-1.0, -3.0, -8.0, -12.0):
                add(f"stable_{label}_u{u1:g}_dt{dt:g}", xland=1.0, znt=z0,
                    u1=u1, t1=285.0, tsk=285.0 + dt, rh=0.8, hfx=-20.0,
                    pblh=120.0)
    # Snow cover (SNOWH >= 0.1 takes Andreas 2002) and sea ice.
    for snowh in (0.1, 0.45):
        for u1, t1, dt in ((1.0, 266.0, -6.0), (4.0, 262.0, -2.0),
                           (8.0, 270.0, -1.0), (3.0, 268.0, 2.0),
                           (0.4, 255.0, -10.0)):
            add(f"snow_h{snowh:g}_u{u1:g}_t{t1:g}_dt{dt:+g}", xland=1.0,
                znt=0.01, u1=u1, t1=t1, tsk=t1 + dt, rh=0.85,
                snowh=snowh, hfx=-10.0 if dt < 0 else 30.0)
    for u1, t1, tsk in ((5.0, 248.0, 252.0), (2.0, 255.0, 245.0),
                        (12.0, 240.0, 250.0)):
        add(f"sea_ice_u{u1:g}_t{t1:g}_tsk{tsk:g}", xland=1.0, znt=0.001,
            u1=u1, t1=t1, tsk=tsk, rh=0.9, snowh=0.2, mavail=1.0)
    # Open ocean: Charnock z0 and COARE 3.0 z_t/z_q.
    for u1 in (0.3, 3.0, 8.0, 15.0, 25.0):
        for t1, dt in ((290.0, -3.0), (290.0, -0.5), (290.0, 1.0),
                       (299.0, 3.0)):
            add(f"ocean_u{u1:g}_t{t1:g}_dt{dt:+g}", xland=2.0, znt=2.0e-4,
                u1=u1, t1=t1, tsk=t1 + dt, rh=0.8, hfx=10.0 * dt)
    add("ocean_cold", xland=2.0, znt=1.0e-4, u1=10.0, t1=272.0, tsk=274.0,
        rh=0.7)
    # High terrain: 650-750 hPa, cold, convective and stable.
    for p1, t1 in ((65000.0, 268.0), (72000.0, 280.0), (75000.0, 290.0)):
        for dt, u1, z0 in ((10.0, 3.0, 0.3), (-6.0, 2.0, 0.3),
                           (-2.0, 9.0, 0.05), (4.0, 7.0, 0.5)):
            add(f"terrain_p{p1/100:g}_t{t1:g}_dt{dt:+g}", xland=1.0,
                znt=z0, u1=u1, t1=t1, tsk=t1 + dt, p1=p1, rh=0.5,
                pblh=2500.0 if dt > 0 else 80.0,
                hfx=300.0 if dt > 0 else -30.0)
    # Edge cases.
    add("za_below_7_za2_inside", xland=1.0, znt=0.1, u1=4.0, t1=290.0,
        tsk=294.0, dz1=10.0, dz2=8.0)
    add("za_below_7_za2_above", xland=1.0, znt=0.1, u1=4.0, t1=290.0,
        tsk=286.0, dz1=10.0, dz2=12.0)
    add("za_above_13_unstable", xland=1.0, znt=0.3, u1=6.0, t1=295.0,
        tsk=301.0, dz1=40.0, dz2=50.0)
    add("za_above_13_stable", xland=1.0, znt=0.3, u1=6.0, t1=295.0,
        tsk=290.0, dz1=40.0, dz2=50.0)
    add("ocean_za_above_13", xland=2.0, znt=2.0e-4, u1=12.0, t1=288.0,
        tsk=290.0, dz1=36.0, dz2=44.0)
    add("land_qsfc_zero", xland=1.0, znt=0.2, u1=3.0, t1=292.0, tsk=296.0,
        qsfc=0.0)
    add("land_qsfc_zero_stable", xland=1.0, znt=0.2, u1=3.0, t1=292.0,
        tsk=289.0, qsfc=0.0)
    add("xland_1p5_unstable", xland=1.5, znt=2.0e-4, u1=6.0, t1=290.0,
        tsk=292.0)
    add("xland_1p5_stable", xland=1.5, znt=2.0e-4, u1=6.0, t1=290.0,
        tsk=287.0)
    add("neutral_with_mol", xland=1.0, znt=0.1, u1=4.0, v1=0.0, t1=295.0,
        tsk=295.0, p1=100000.0, psfc=100000.0, qv1=0.008, hfx=0.0, ust=0.2,
        mol=-0.05)
    add("neutral_calm_ust", xland=1.0, znt=0.1, u1=0.1, v1=0.0, t1=295.0,
        tsk=295.0, p1=100000.0, psfc=100000.0, qv1=0.008, hfx=0.0,
        ust=0.005, mol=0.02)
    add("dry_mavail_zero", xland=1.0, znt=0.05, u1=5.0, t1=305.0,
        tsk=318.0, rh=0.1, mavail=0.0, hfx=400.0)
    add("skin_freezing_air_warm", xland=1.0, znt=0.1, u1=3.0, t1=276.0,
        tsk=271.0, rh=0.9)
    add("skin_warm_air_freezing", xland=1.0, znt=0.1, u1=3.0, t1=271.0,
        tsk=275.0, rh=0.9)
    add("near_calm_unstable", xland=1.0, znt=0.1, u1=0.05, v1=0.02,
        t1=300.0, tsk=306.0, hfx=200.0)
    add("near_calm_inversion", xland=1.0, znt=0.1, u1=0.3, v1=0.1,
        t1=290.0, tsk=280.0)
    add("small_pblh_unstable", xland=1.0, znt=0.1, u1=2.0, t1=290.0,
        tsk=293.0, pblh=20.0)
    add("urban_rough", xland=1.0, znt=2.0, u1=5.0, t1=295.0, tsk=300.0,
        hfx=250.0)
    # The fork's bulk Richardson clamp at +50 (:613-614): a strong inversion
    # under near-calm air, the first level at 15 m and at 8 m.
    add("calm_inversion_br_clamp_za15", xland=1.0, znt=0.1, u1=0.2, v1=0.0,
        t1=285.0, tsk=273.0, rh=0.9, dz1=30.0, dz2=40.0)
    add("calm_inversion_br_clamp_za8", xland=1.0, znt=0.075, u1=0.1,
        v1=0.05, t1=283.0, tsk=277.0, rh=0.9)
    return names, cols


def ocean_deck():
    names, cols = [], []
    for u1 in (0.3, 3.0, 8.0, 15.0, 25.0):
        for t1, dt in ((290.0, -3.0), (290.0, 1.0), (299.0, 3.0)):
            names.append(f"ocean_u{u1:g}_t{t1:g}_dt{dt:+g}")
            cols.append(_column(xland=2.0, znt=2.0e-4, u1=u1, t1=t1,
                                tsk=t1 + dt, rh=0.8, hfx=10.0 * dt))
    for dt in (-2.0, 2.0):
        names.append(f"xland_1p5_dt{dt:+g}")
        cols.append(_column(xland=1.5, znt=2.0e-4, u1=6.0, t1=290.0,
                            tsk=290.0 + dt))
    names.append("land_beside")
    cols.append(_column(xland=1.0, znt=0.1, u1=4.0, t1=290.0, tsk=294.0))
    return names, cols


def dx13km_deck():
    names, cols = main_deck()
    keep = [i for i in range(0, len(names), 4)]
    return [names[i] for i in keep], [cols[i] for i in keep]


#: deck name -> (column builder, ISFTCFLX, DX)
DECKS = {
    "main": (main_deck, 0, 3000.0),
    "isftcflx1": (ocean_deck, 1, 3000.0),
    "isftcflx2": (ocean_deck, 2, 3000.0),
    "isftcflx3": (ocean_deck, 3, 3000.0),
    "dx13km": (dx13km_deck, 0, 13545.0),
}


def write_deck(path, deck: str, nstep: int = NSTEP) -> None:
    builder, isftcflx, dx = DECKS[deck]
    _, cols = builder()
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(f"{len(cols)} {nstep} {isftcflx} {dx:.9g}\n")
        for col in cols:
            fh.write(" ".join(f"{float(col[k]):.9g}" for k in FIELDS) + "\n")


def load_deck(path):
    """(nstep, isftcflx, dx, {field: float32 array}) as the Fortran reads it."""
    with open(path, encoding="ascii") as fh:
        head = fh.readline().split()
        n, nstep, isftcflx = int(head[0]), int(head[1]), int(head[2])
        dx = float(np.float32(float(head[3])))
        rows = [list(map(float, fh.readline().split())) for _ in range(n)]
    data = np.asarray(rows, dtype=np.float64).astype(np.float32)
    return nstep, isftcflx, dx, {k: data[:, i].copy()
                                 for i, k in enumerate(FIELDS)}


def deck_names(deck: str) -> list[str]:
    return DECKS[deck][0]()[0]


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for deck in DECKS:
        write_deck(out / f"{deck}.txt", deck)
        print(deck, len(deck_names(deck)), "columns")
