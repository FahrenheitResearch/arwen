"""Write the input columns for the classic MM5 surface-layer oracle.

``sf_sfclay_physics = 91`` runs ``module_sf_sfclay.F:SFCLAY``.  This script
writes the inputs both sides read: the WRF column driver
(``run_sfclay_classic.F90``) and the WOOF kernel validator
(``validate_sfclay_classic_oracle.py``).  It imports nothing from the engine,
so the column set cannot be shaped by the code it grades.

Every value is a float32 written as its unsigned 32-bit word, so both readers
see exactly the same bits.  The file is plain text:

    NGROUP NSTEP
    for each group:
        NAME NCOL ISFFLX ISFTCFLX IZ0TLND DX_WORD
        NCOL lines of 20 words, in INPUT_FIELDS order

DX is per group because the WOOF launcher takes one grid spacing per launch
(WRF takes a 2-D DX, filled uniformly by every real WRF domain).

Usage: python make_columns.py OUT_FILE
"""

from __future__ import annotations

import sys

import numpy as np

INPUT_FIELDS = ("u", "v", "t", "qv", "p", "dz8w", "psfc", "tsk", "znt",
                "ust", "ustm", "pblh", "mavail", "xland", "lakemask",
                "mol", "hfx", "qfx", "qsfc", "zol")

#: Calls per column.  Each step feeds every WRF inout field (ZNT, UST, USTM,
#: MOL, HFX, QFX, QSFC, ZOL) back into the next call, so step 2 and 3 run on
#: the scheme's own state: the old-MOL sign clamp, the old-UST z/L, the
#: Beljaars flux from the previous HFX/QFX and the Charnock ZNT update.
NSTEP = 3

_BASE = dict(u=5.0, v=2.0, t=295.0, qv=0.010, p=99500.0, dz8w=50.0,
             psfc=100000.0, tsk=297.0, znt=0.1, ust=0.3, ustm=0.3,
             pblh=800.0, mavail=0.5, xland=1.0, lakemask=0.0, mol=0.0,
             hfx=0.0, qfx=0.0, qsfc=0.0, zol=0.0)


def col(**kw):
    unknown = set(kw) - set(_BASE)
    if unknown:
        raise KeyError(sorted(unknown))
    out = dict(_BASE)
    out.update(kw)
    return out


def ocean(**kw):
    base = dict(xland=2.0, lakemask=0.0, znt=2.0e-4, mavail=1.0)
    base.update(kw)
    return col(**base)


def hand_columns():
    """Named regimes and edges, each a realistic WRF lowest-level state."""
    c = []
    # --- land, convective (regime 4) -------------------------------------
    c.append(col(u=3.0, v=1.0, t=303.0, qv=0.012, p=99400.0, tsk=318.0,
                 ust=0.4, ustm=0.35, pblh=1800.0, mavail=0.3, mol=-0.8,
                 hfx=350.0, qfx=1.2e-4, qsfc=0.015, zol=-1.5))
    c.append(col(u=0.5, v=0.3, t=300.0, qv=0.011, tsk=325.0, ust=0.15,
                 ustm=0.1, pblh=2500.0, mol=-1.5, hfx=450.0, qfx=5e-5,
                 qsfc=0.013))                       # z/L clamps at -9.9999
    c.append(col(u=1.0, v=0.0, t=301.0, qv=0.012, tsk=312.0, ust=0.005,
                 ustm=0.004, mol=-1.0, hfx=200.0, qfx=8e-5, qsfc=0.014,
                 pblh=1500.0))                      # UST < 0.01: z/L = Br*gz1
    c.append(col(u=12.0, v=-6.0, t=290.0, qv=0.008, tsk=290.5, ust=0.6,
                 ustm=0.6, mol=-0.02, hfx=20.0, qfx=3e-5, qsfc=0.009))
    # --- land, stable night (regimes 1, 2, 3) -----------------------------
    c.append(col(u=1.5, v=0.5, t=278.0, qv=0.005, tsk=268.0, ust=0.1,
                 ustm=0.08, mol=0.3, hfx=-30.0, qfx=-1e-6, pblh=100.0,
                 qsfc=0.004, zol=2.0))              # Br >= 0.2, regime 1
    c.append(col(u=2.0, v=0.0, t=280.0, qv=0.005, tsk=272.0, ust=0.005,
                 ustm=0.005, mol=0.2, hfx=-10.0, pblh=60.0, qsfc=0.004,
                 zol=0.7))                          # regime 1, UST < 0.01
    c.append(col(u=6.0, v=2.0, t=285.0, qv=0.006, tsk=283.5, ust=0.3,
                 ustm=0.3, mol=0.05, hfx=-15.0, qsfc=0.006))   # regime 2
    c.append(col(u=3.0, v=0.0, t=285.0, qv=0.006, tsk=284.0, ust=0.2,
                 ustm=0.2, mol=0.08, hfx=-12.0, qsfc=0.006))   # z/L > 0.5
    c.append(col(u=4.0, v=1.0, t=285.0, qv=0.006, tsk=281.0, ust=0.25,
                 ustm=0.2, mol=-0.1, hfx=40.0, qsfc=0.006))    # Br -> 0, reg 3
    c.append(col(u=4.0, v=1.0, t=285.0, qv=0.006, tsk=281.0, ust=0.005,
                 ustm=0.005, mol=-0.1, hfx=40.0, qsfc=0.006))  # reg 3, UST<.01
    c.append(col(u=5.0, v=3.0, t=297.0, qv=0.010, tsk=300.0, ust=0.3,
                 mol=-0.2, hfx=80.0, qfx=6e-5, qsfc=0.012))    # MOL<0, unstable
    # --- roughness and layer depth ----------------------------------------
    c.append(col(u=4.0, v=2.0, t=300.0, qv=0.012, tsk=310.0, znt=2.0,
                 dz8w=20.0, ust=0.7, ustm=0.6, mol=-0.5, hfx=300.0,
                 qfx=1e-4, qsfc=0.016, pblh=1600.0))  # forest: psih clamp, psit floor
    c.append(col(u=3.0, v=1.0, t=282.0, qv=0.006, tsk=278.0, znt=2.0,
                 dz8w=20.0, ust=0.4, mol=0.1, hfx=-20.0, qsfc=0.006))
    c.append(col(u=6.0, v=1.0, t=296.0, qv=0.011, tsk=303.0, znt=0.5,
                 dz8w=8.0, ust=0.5, mol=-0.3, hfx=150.0, qsfc=0.013))  # za = 4 m
    c.append(col(u=9.0, v=4.0, t=292.0, qv=0.009, tsk=296.0, dz8w=120.0,
                 ust=0.45, mol=-0.15, hfx=120.0, qsfc=0.010))          # za = 60 m
    # --- snow, sea ice and cold land ---------------------------------------
    c.append(col(u=2.5, v=1.0, t=255.0, qv=5e-4, tsk=248.0, znt=0.002,
                 ust=0.2, ustm=0.18, mol=0.1, hfx=-25.0, qsfc=3e-4,
                 mavail=1.0, pblh=150.0, p=101000.0, psfc=101600.0))
    c.append(col(u=3.0, v=-1.0, t=265.0, qv=1.5e-3, tsk=270.0, znt=0.002,
                 ust=0.25, mol=-0.1, hfx=60.0, qsfc=2e-3, mavail=1.0))
    c.append(col(u=7.0, v=3.0, t=245.0, qv=3e-4, tsk=250.0, znt=0.001,
                 ust=0.3, mol=-0.05, hfx=40.0, qsfc=0.0, mavail=1.0,
                 p=101500.0, psfc=102000.0))        # sea ice, qsfc recomputed
    c.append(col(u=1.0, v=0.5, t=240.0, qv=2e-4, tsk=229.0, znt=0.001,
                 ust=0.08, mol=0.4, hfx=-15.0, qsfc=1e-4, mavail=1.0,
                 pblh=40.0, p=103000.0, psfc=103500.0))  # polar night inversion
    # --- high terrain -------------------------------------------------------
    c.append(col(u=6.0, v=3.0, t=285.0, qv=0.006, p=61000.0, psfc=61500.0,
                 tsk=305.0, ust=0.45, mol=-0.6, hfx=400.0, qfx=6e-5,
                 qsfc=0.007, pblh=3000.0, mavail=0.2))
    c.append(col(u=2.0, v=1.0, t=270.0, qv=0.003, p=62000.0, psfc=62400.0,
                 tsk=262.0, ust=0.15, mol=0.2, hfx=-20.0, qsfc=0.002,
                 pblh=80.0))
    # --- moisture extremes --------------------------------------------------
    c.append(col(u=5.0, v=0.0, t=310.0, qv=0.002, tsk=335.0, mavail=0.0,
                 ust=0.5, mol=-1.0, hfx=500.0, qfx=0.0, qsfc=0.003,
                 pblh=3500.0))                       # desert, MAVAIL 0
    c.append(col(u=4.0, v=2.0, t=300.0, qv=0.020, tsk=302.0, mavail=1.0,
                 ust=0.3, mol=-0.1, hfx=60.0, qfx=1.5e-4, qsfc=0.025))
    c.append(col(u=4.0, v=2.0, t=290.0, qv=0.0, tsk=294.0, mavail=0.0,
                 ust=0.3, mol=-0.1, hfx=60.0, qsfc=0.0))  # dry: QV 0, QSFC recomputed
    # --- calm and odd winds -------------------------------------------------
    c.append(col(u=0.0, v=0.0, t=302.0, qv=0.012, tsk=320.0, ust=0.2,
                 mol=-1.0, hfx=300.0, qfx=1e-4, qsfc=0.014, pblh=2000.0))
    c.append(col(u=0.0, v=0.0, t=280.0, qv=0.005, tsk=275.0, ust=0.1,
                 mol=0.1, hfx=-5.0, qsfc=0.005, pblh=50.0))  # WSPD floor 0.1
    c.append(col(u=-7.0, v=-9.0, t=293.0, qv=0.009, tsk=295.0, ust=0.5,
                 mol=-0.05, hfx=50.0, qsfc=0.010))
    c.append(col(u=3.0, v=2.0, t=299.0, qv=0.011, tsk=306.0, pblh=0.0,
                 ust=0.3, mol=-0.3, hfx=150.0, qsfc=0.013))  # PBLH 0
    c.append(col(u=3.0, v=2.0, t=299.0, qv=0.011, tsk=303.0, ust=0.3,
                 mol=0.1, hfx=-40.0, qfx=-1e-5, qsfc=0.013))  # old HFX < 0
    c.append(col(u=3.0, v=2.0, t=299.0, qv=0.011, tsk=303.0, ust=0.0,
                 ustm=0.0, mol=0.0, qsfc=0.013))             # UST 0, first call
    c.append(col(u=4.0, v=1.0, t=290.0, qv=0.008, p=100000.0, psfc=100000.0,
                 tsk=290.0, ust=0.3, qsfc=0.008))            # THX == THGB
    # --- ocean (XLAND 2, LAKEMASK 0) ------------------------------------------
    c.append(ocean(u=9.0, v=3.0, t=298.0, qv=0.017, tsk=301.0, ust=0.35,
                   ustm=0.33, mol=-0.05, hfx=25.0, qfx=8e-5))
    c.append(ocean(u=0.3, v=0.1, t=299.0, qv=0.017, tsk=302.0, ust=0.05,
                   mol=-0.3, hfx=15.0, qfx=3e-5))           # calm: VCONV = sqrt(-dthv)
    c.append(ocean(u=7.0, v=-2.0, t=290.0, qv=0.009, tsk=284.0, ust=0.2,
                   mol=0.1, hfx=-20.0))                     # warm air over cold sea
    c.append(ocean(u=35.0, v=20.0, t=299.0, qv=0.019, tsk=301.0, znt=2.8e-3,
                   ust=1.4, ustm=1.4, mol=-0.02, hfx=100.0, qfx=3e-4,
                   p=96000.0, psfc=96600.0))                # hurricane wind
    c.append(ocean(u=6.0, v=0.0, t=295.0, qv=0.014, tsk=296.0, ust=0.0,
                   ustm=0.0))                               # UST 0: RESTAR 0
    c.append(ocean(u=4.0, v=4.0, t=295.0, qv=0.014, tsk=297.0, znt=1.0e-6,
                   ust=0.15, mol=-0.02))                    # tiny roughness
    c.append(ocean(u=5.0, v=1.0, t=294.0, qv=0.012, tsk=296.0, lakemask=1.0,
                   znt=1e-4, ust=0.2, mol=-0.05, hfx=20.0))  # lake: no salinity
    c.append(ocean(u=3.0, v=1.0, t=285.0, qv=0.006, tsk=279.0, lakemask=1.0,
                   ust=0.1, mol=0.15, hfx=-10.0))           # cold lake, stable
    c.append(ocean(u=6.0, v=2.0, t=291.0, qv=0.010, tsk=288.0, ust=0.2,
                   mol=-0.02, hfx=5.0))                     # MOL < 0: regime 3
    c.append(ocean(u=8.0, v=1.0, t=296.0, qv=0.015, tsk=298.0, ust=0.3,
                   qsfc=0.05, mol=-0.03))                   # QSFC input overwritten
    return c


def random_columns(n, seed):
    """Seeded plausible states: land/ocean mix, day and night."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        water = k % 4 == 3
        t = rng.uniform(250.0, 310.0)
        day = rng.random() < 0.5
        dt_sfc = rng.uniform(0.5, 18.0) if day else -rng.uniform(0.5, 12.0)
        if water:
            dt_sfc = float(np.clip(dt_sfc, -6.0, 4.0))
        psfc = rng.uniform(65000.0, 103000.0)
        dz = rng.uniform(10.0, 110.0)
        p = psfc - 1.1 * 0.5 * dz * 9.81 * rng.uniform(0.9, 1.1)
        es = 0.6112 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
        qsat = 0.622 * es / (p / 1000.0 - es)
        qv = qsat * rng.uniform(0.1, 0.95)
        u = rng.uniform(-15.0, 15.0)
        v = rng.uniform(-10.0, 10.0)
        ust = rng.uniform(0.0, 0.8)
        mol = rng.uniform(-1.0, 0.0) if day else rng.uniform(0.0, 0.4)
        kw = dict(u=u, v=v, t=t, qv=qv, p=p, dz8w=dz, psfc=psfc,
                  tsk=t + dt_sfc, ust=ust, ustm=ust * rng.uniform(0.7, 1.0),
                  pblh=rng.uniform(50.0, 3000.0) if day else rng.uniform(20.0, 400.0),
                  mol=mol, hfx=rng.uniform(0.0, 400.0) if day else rng.uniform(-60.0, 0.0),
                  qfx=rng.uniform(0.0, 2e-4) if day else rng.uniform(-1e-5, 2e-5),
                  qsfc=qv * rng.uniform(1.0, 1.4) if rng.random() < 0.8 else 0.0,
                  zol=rng.uniform(-3.0, 0.0) if day else rng.uniform(0.0, 3.0))
        if water:
            out.append(ocean(znt=rng.uniform(1e-5, 2e-3),
                             lakemask=float(rng.random() < 0.25), **kw))
        else:
            out.append(col(znt=float(np.exp(rng.uniform(np.log(1e-3), np.log(2.0)))),
                           mavail=rng.uniform(0.0, 1.0), **kw))
    return out


def groups():
    hand = hand_columns()
    land = [c for c in hand if c["xland"] < 1.5]
    sea = [c for c in hand if c["xland"] > 1.5]
    pick = lambda seq, idx: [seq[i] for i in idx]          # noqa: E731
    return [
        # name, isfflx, isftcflx, iz0tlnd, dx, columns
        ("base-3km", 1, 0, 0, 3000.0, hand),
        ("random-3km", 1, 0, 0, 3000.0, random_columns(40, 20261007)),
        ("vsgd-12km", 1, 0, 0, 12000.0,
         pick(land, (0, 1, 4, 6, 8, 24, 25)) + pick(sea, (0, 1, 2))),
        ("vsgd-zero-5km", 1, 0, 0, 5000.0, pick(land, (0, 4)) + pick(sea, (0, 2))),
        ("isftcflx1", 1, 1, 0, 3000.0,
         pick(sea, (0, 1, 2, 3, 4, 5, 8)) + pick(land, (0,))),
        ("isftcflx2", 1, 2, 0, 3000.0,
         pick(sea, (0, 1, 2, 3, 4, 5, 8)) + pick(land, (0,))),
        ("iz0tlnd1", 1, 0, 1, 3000.0,
         pick(land, (0, 2, 4, 6, 11, 15, 19)) + pick(sea, (0,))),
        ("iz0tlnd2", 1, 0, 2, 3000.0, pick(land, (0, 4, 6, 11, 19)) + pick(sea, (2,))),
        ("isfflx0", 0, 0, 0, 3000.0, pick(land, (0, 4, 6)) + pick(sea, (0, 2, 3))),
        ("isftcflx1-iz0tlnd1", 1, 1, 1, 3000.0, pick(land, (0, 4)) + pick(sea, (0, 3))),
        ("random-12km", 1, 0, 0, 12000.0, random_columns(16, 7)),
    ]


def word(x) -> int:
    return int(np.float32(x).view(np.uint32))


def main(path: str) -> int:
    gs = groups()
    lines = [f"{len(gs)} {NSTEP}"]
    total = 0
    for name, isfflx, isftcflx, iz0tlnd, dx, cols in gs:
        lines.append(f"{name} {len(cols)} {isfflx} {isftcflx} {iz0tlnd} {word(dx)}")
        for c in cols:
            lines.append(" ".join(str(word(c[f])) for f in INPUT_FIELDS))
        total += len(cols)
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path}: {len(gs)} groups, {total} columns, {NSTEP} steps")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
