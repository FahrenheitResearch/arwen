"""Column deck for the GSL WRF 3.9 fork MYNN surface-layer oracle.

Writes the text file run_surface_layer_fork.F90 reads: a header line
``NCOL NSTEP`` and one line per column::

    xland snowh u1 v1 t1 qv1 p1 rho1 dz1 u2 v2 dz2 psfc tsk pblh mavail
    hfx qfx znt qsfc ust mol

Every value is a float32 written with 9 significant digits, so the Fortran
read and :func:`load_columns` give the same words.  Step 1 enters with the
SFCLAY_mynn wrapper's itimestep == 1 seeding (fork :329-337): UST =
max(0.04*|V1|, 0.001), MOL = 0, QSFC = QV1/(1+QV1).

The deck is a grid over roughness (forest 0.8 m, urban 0.5, crop/mosaic
0.3 and 0.2, grass 0.075, bare 0.01, water), first-level wind and
skin-minus-air temperature, with the first level at 8 m (HRRR's lowest
layer is about 16 m deep), plus four special columns: snow, a near-calm
inversion that drives the bulk Richardson number past 4 (the 50 against 4
clamp and the 50 against 20 z/L cap), an exactly neutral column entering
with a nonzero MOL (the fork's neutral-branch z/L), and a water column at
exactly XLAND = 1.5.

    python make_columns.py OUT.txt [NSTEP]
"""
from __future__ import annotations

import sys

import numpy as np

F = np.float32
R, RV = F(287.0), F(461.6)
EP1 = F(RV / R - F(1.0))
FIELDS = ("xland", "snowh", "u1", "v1", "t1", "qv1", "p1", "rho1", "dz1",
          "u2", "v2", "dz2", "psfc", "tsk", "pblh", "mavail", "hfx", "qfx",
          "znt", "qsfc", "ust", "mol")
ROUGHNESS = (("forest", 1.0, 0.8), ("urban", 1.0, 0.5), ("mosaic", 1.0, 0.3),
             ("crop", 1.0, 0.2), ("grass", 1.0, 0.075), ("bare", 1.0, 0.01),
             ("water", 2.0, 2.0e-4))
WINDS = (1.0, 2.0, 3.0, 5.0, 8.0)
DT_SKIN = (-8.0, -4.0, -2.0, -1.0, -0.5, 1.0, 3.0, 8.0)


def _column(*, xland, znt, u1, t1, tsk, qv1=0.007, p1=99000.0,
            psfc=100000.0, dz1=16.0, dz2=20.0, snowh=0.0, pblh=None,
            mavail=None, hfx=None, ust=None, mol=0.0, qsfc=None, v1=None):
    v1 = 0.3 * u1 if v1 is None else v1
    stable = tsk < t1
    col = {
        "xland": xland, "snowh": snowh, "u1": u1, "v1": v1, "t1": t1,
        "qv1": qv1, "p1": p1, "dz1": dz1, "u2": 1.3 * u1, "v2": 1.3 * v1,
        "dz2": dz2, "psfc": psfc, "tsk": tsk,
        "pblh": (150.0 if stable else 1200.0) if pblh is None else pblh,
        "mavail": (1.0 if xland > 1.5 else 0.4) if mavail is None else mavail,
        "hfx": (-15.0 if stable else 150.0) if hfx is None else hfx,
        "qfx": 0.0, "znt": znt, "mol": mol,
    }
    col = {k: F(v) for k, v in col.items()}
    col["rho1"] = F(col["p1"] / (R * col["t1"] * (
        F(1.0) + EP1 * col["qv1"] / (F(1.0) + col["qv1"]))))
    wsp = F(np.sqrt(col["u1"] * col["u1"] + col["v1"] * col["v1"]))
    col["ust"] = F(max(F(0.04) * wsp, F(0.001))) if ust is None else F(ust)
    col["qsfc"] = (F(col["qv1"] / (F(1.0) + col["qv1"])) if qsfc is None
                   else F(qsfc))
    return col


def columns() -> tuple[list[str], list[dict]]:
    names, cols = [], []
    for label, xland, znt in ROUGHNESS:
        for u1 in WINDS:
            for dt in DT_SKIN:
                names.append(f"{label}_u{u1:g}_dt{dt:+g}")
                cols.append(_column(xland=xland, znt=znt, u1=u1, t1=290.0,
                                    tsk=290.0 + dt))
    names.append("snow_stable")
    cols.append(_column(xland=1.0, znt=0.01, u1=3.0, t1=268.0, tsk=263.0,
                        snowh=0.2, qv1=0.0025))
    names.append("calm_inversion")
    cols.append(_column(xland=1.0, znt=0.1, u1=0.3, v1=0.1, t1=290.0,
                        tsk=284.0))
    names.append("neutral_with_mol")
    cols.append(_column(xland=1.0, znt=0.1, u1=4.0, v1=0.0, t1=295.0,
                        tsk=295.0, p1=100000.0, qv1=0.008, hfx=0.0,
                        ust=0.2, mol=-0.05))
    names.append("water_xland_1p5")
    cols.append(_column(xland=1.5, znt=2.0e-4, u1=6.0, t1=290.0, tsk=292.0))
    return names, cols


def write_deck(path, nstep: int = 6) -> None:
    names, cols = columns()
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(f"{len(cols)} {nstep}\n")
        for col in cols:
            fh.write(" ".join(f"{float(col[k]):.9g}" for k in FIELDS) + "\n")


def load_columns(path) -> tuple[int, dict[str, np.ndarray]]:
    with open(path, encoding="ascii") as fh:
        n, nstep = (int(v) for v in fh.readline().split())
        rows = [list(map(float, fh.readline().split())) for _ in range(n)]
    data = np.asarray(rows, dtype=np.float64).astype(np.float32)
    return nstep, {k: data[:, i].copy() for i, k in enumerate(FIELDS)}


if __name__ == "__main__":
    write_deck(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
