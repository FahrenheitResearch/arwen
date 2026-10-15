#!/usr/bin/env python3
"""Ferrier-Aligo (WRF mp_physics=5) oracle columns for the branches the lane
and refute column sets never executed.

A gcov build of the byte-unmodified module over the 72 lane and 65 refute
columns left these computational lines of ``phys/module_mp_fer_hires.F``
unrun:

* 2307-2308  the post-fallout heavy-rain cap (``INDEXR=MDRmax`` in
  rain_pass).  It needs a rain content above RQR_DRmax (about 25 g m-3)
  after fallout AND the drizzle flag off, which only melting ice in the
  same column turns off; the refute rain_cap columns had no ice, so the
  drizzle branch caught every one of them first.
* 2621-2622  CONDENSE's fallback when ten Asai passes do not converge.
  The Asai step assumes dWS/dT from Clausius-Clapeyron alone; the true
  slope carries the factor PP/(PP-ESW), so once the saturation vapour
  pressure is a large part of the pressure (ESW/PP above about 0.6) each
  pass overshoots by more than it corrects.  Such air holds over 0.5 kg/kg
  of vapour at saturation: legal input, never a forecast state.
* 2685-2686  DEPOSIT's fallback, the same overshoot for ice below -40 C,
  which needs pressures of a few pascals.
* 2213  ``RimeF=RimeF1`` when the ice budget DUM2=TOT_ICE+THICK*DELS stays
  at or below CLIMIT while TOT_ICEnew exceeds it: new ice nucleating
  (PINIT>0, so deposition does not count toward DELS) in a layer of a few
  kg m-2 with no ice in it or falling into it, and a trace of rain whose
  Biggs freezing makes DELS just exceed EPSQ.
* 3292  FPVS's water formula above the 330 K table top on the ICE path:
  homogeneous freezing of 0.4 kg/kg of cloud water at -50 C lifts the
  dummy temperature DUM1=TK+XLF1*PIACW past 330 K.

1627 (VEL_RF at Nrime) stays unreachable in binary32: RimeF1 is capped at
RFmax = 1.1**40 = 45.2593 before XRF=10.492*ALOG(RimeF1) is formed, and
that product rounds to 39.9997863770, so INT(XRF) never reaches Nrime = 40.

Every level of these columns is its own trial: no precipitation falls into
a level that the trial does not put there, pressures need not decrease
upward (the scheme reads each level's own P_PHY), and the layer depth is
set per level.  Output format is make_columns.py's.  No random numbers.

usage: make_columns_coverage.py OUT_DIR
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_ferrier_make_columns", _HERE.parent / "make_columns.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

NZ = mc.NZ
T0C = 273.15
RD = 287.04
RCP = 287.04 / 1004.6
EPS = 287.04 / 461.5


def esw(t):
    """Water saturation vapour pressure (Pa), Murray's Magnus form."""
    return 610.78 * np.exp(17.2694 * (t - 273.16) / (t - 35.86))


def esi(t):
    return 610.78 * np.exp(21.8746 * (t - 273.16) / (t - 7.66))


class Trial:
    """One column whose levels are independent trials."""

    def __init__(self, regime: str, idx: int, *, dt: float, gid: int,
                 nsteps: int = 10):
        self.regime = regime
        self.idx = idx
        self.dt = dt
        self.gid = gid
        self.nsteps = nsteps
        self.dx = 12000.0
        self.rainnc = 0.0
        self.p = np.full(NZ, 50000.0)
        self.t = np.full(NZ, 250.0)
        self.qv = np.full(NZ, 1.0e-6)
        self.qc = np.zeros(NZ)
        self.qr = np.zeros(NZ)
        self.qi = np.zeros(NZ)
        self.dz = np.full(NZ, 200.0)
        self.f_ice = np.zeros(NZ)
        self.f_rain = np.zeros(NZ)
        self.f_rimef = np.ones(NZ)

    def arrays(self) -> dict[str, np.ndarray]:
        pi = (self.p / 1.0e5) ** RCP
        rho = self.p / (RD * self.t * (1.0 + 0.608 * self.qv))
        return {"p": self.p, "pi": pi, "th": self.t / pi, "qv": self.qv,
                "qc": self.qc, "qr": self.qr, "qi": self.qi,
                "dz8w": self.dz, "rho": rho, "f_ice": self.f_ice,
                "f_rain": self.f_rain, "f_rimef": self.f_rimef}


def _spread(lo: float, hi: float, n: int = NZ, log: bool = False):
    if log:
        return np.exp(np.linspace(math.log(lo), math.log(hi), n))
    return np.linspace(lo, hi, n)


def build() -> list[Trial]:
    cols: list[Trial] = []

    def add(c):
        cols.append(c)
        return c

    k = np.arange(NZ)

    # rain_cap_melt: ice melting into very heavy rain in thick warm layers,
    # so the drizzle flag is off when rain_pass meets RQRnew > RQR_DRmax.
    for j, dt in enumerate((6.0, 20.0, 36.0, 72.0)):
        c = add(Trial("rain_cap_melt", len(cols), dt=dt, gid=1 + j % 2))
        c.p = _spread(100000.0, 70000.0)
        c.t = 276.0 + 4.0 * np.sin(0.7 * k + j)
        c.qv = 0.98 * EPS * esw(c.t) / (c.p - esw(c.t))
        c.qr = _spread(30.0e-3, 90.0e-3) * (1.0 + 0.1 * j)
        c.qi = _spread(0.5e-3, 4.0e-3)[::-1]
        c.dz = np.full(NZ, 400.0 + 400.0 * j)
        c.f_rimef = np.where(c.qi > 0, 1.0 + 3.0 * (k % 4), 1.0)

    # condense_nc_lowp: warm saturated air where ESW is a large part of the
    # pressure, with and without cloud water.
    for j in range(4):
        c = add(Trial("condense_nc_lowp", len(cols), dt=(20.0, 60.0)[j % 2],
                      gid=1 + j % 2))
        c.t = 285.0 + 10.0 * j + 3.0 * (k % 5)
        ratio = _spread(0.35, 0.95)
        c.p = esw(c.t) / ratio
        ss = 1.002 + 0.02 * (k % 7)
        e = np.minimum(esw(c.t), 0.99 * c.p)
        c.qv = ss * EPS * e / (c.p - e)
        c.qc = np.where(k % 2 == 0, 1.0e-3 * (1 + j), 0.0)
        c.dz = np.full(NZ, 300.0)

    # deposit_nc_lowp: below -40 C with the ice saturation vapour pressure a
    # large part of the pressure, cloud water present (homogeneous freezing
    # then DEPOSIT).
    for j in range(4):
        c = add(Trial("deposit_nc_lowp", len(cols), dt=(20.0, 60.0)[j % 2],
                      gid=1 + j % 2))
        c.t = 200.0 + 6.0 * j + 2.0 * (k % 5)
        ratio = _spread(0.30, 0.95)
        c.p = esi(c.t) / ratio
        ss = 1.01 + 0.05 * (k % 6)
        e = np.minimum(esi(c.t), 0.99 * c.p)
        c.qv = ss * EPS * e / (c.p - e)
        c.qc = 1.0e-4 * (1 + k % 3)
        c.dz = np.full(NZ, 2000.0)

    # rime_reset: one supercooled, water-supersaturated, ice-free trial
    # layer of a few decimetres at L = 2 (the level under the model top,
    # which FER_HIRES leaves as it is), dry clear air elsewhere, and just
    # enough rain that its Biggs freezing makes DELS exceed EPSQ while
    # THICK*DELS stays under CLIMIT.  The nucleated ice (PINIT > 0) is what
    # carries TOT_ICEnew over CLIMIT.  PIACR ~ min(1.03*DT*D, 1)*QR with
    # D = (EXP(ABFR*TC)-1)*(0.05 mm)**7 sets the rain for each trial.
    for dt in (6.0, 36.0):
        for tc in (-13.0, -20.0, -30.0, -35.0):
            c = add(Trial("rime_reset", len(cols), dt=dt, gid=2))
            c.p = np.full(NZ, 60000.0)
            c.t = np.full(NZ, 250.0)
            kk = NZ - 2
            t = T0C + tc
            c.t[kk] = t
            c.qv[kk] = 1.02 * EPS * esw(t) / (60000.0 - esw(t))
            rate = min(1.03 * dt * (math.exp(-0.66 * tc) - 1.0) * 0.05 ** 7,
                       1.0)
            c.qr[kk] = max(2.0e-12 / rate, 1.1e-12)
            c.dz[kk] = 0.3

    # fpvs_hot_ice: homogeneous freezing of 0.2-0.6 kg/kg of cloud water
    # below -40 C, lifting DUM1 past the 330 K table top.
    for j in range(2):
        c = add(Trial("fpvs_hot_ice", len(cols), dt=(20.0, 60.0)[j],
                      gid=1 + j))
        c.p = _spread(40000.0, 20000.0)
        c.t = 215.0 + 3.0 * (k % 5) + 4.0 * j
        c.qc = _spread(0.2, 0.6)
        c.qv = 1.05 * EPS * esi(c.t) / (c.p - esi(c.t))
        c.dz = np.full(NZ, 300.0)

    return cols


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cols = build()
    f32hex = mc.f32hex
    with open(out / "columns_inputs.csv", "w", newline="\n") as fc, \
            open(out / "columns.in", "w", newline="\n") as fi:
        fc.write("# Ferrier-Aligo coverage columns; floats are binary32 bit "
                 "patterns (hex)\n")
        fc.write("# column rows: C,col,regime,gid,nsteps,dt,dx,rainnc\n")
        fc.write("# level rows (k=1 is the surface): L,col,k,"
                 + ",".join(mc.FIELDS) + "\n")
        fi.write(f"{len(cols):6d} {NZ:6d}\n")
        for c in cols:
            a = c.arrays()
            fc.write(f"C,{c.idx},{c.regime},{c.gid},{c.nsteps},"
                     f"{f32hex(c.dt)},{f32hex(c.dx)},{f32hex(c.rainnc)}\n")
            fi.write(f"{c.gid:6d} {c.nsteps:6d} {f32hex(c.dt)} "
                     f"{f32hex(c.dx)} {f32hex(c.rainnc)}\n")
            for kk in range(NZ):
                words = [f32hex(a[f][kk]) for f in mc.FIELDS]
                fc.write(f"L,{c.idx},{kk + 1}," + ",".join(words) + "\n")
                fi.write(" ".join(words) + "\n")
    print(f"wrote {len(cols)} columns x {NZ} levels to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
