#!/usr/bin/env python3
"""Write the Ferrier-Aligo (WRF mp_physics=5) column-oracle inputs.

The inputs are built deterministically from closed-form soundings (no random
number generator, so the bits do not depend on the NumPy version) and are
written twice:

* ``columns_inputs.csv`` -- the canonical, human-readable input set.  Every
  float is its IEEE-754 binary32 bit pattern in hex, so a GPU port can be fed
  exactly the bits the Fortran saw.
* ``columns.in`` -- the same values in the fixed format ``run_ferrier.F90``
  reads.

Usage: ``python3 make_columns.py OUT_DIR``.

Regimes (8 columns each, 9 groups = 72 columns):

``convective``    warm moist deep convection, supersaturated updraft layers,
                  cloud water, rain, rimed ice aloft
``stable_night``  surface inversion with radiation fog, dry aloft
``snow``          whole column below freezing, ice-supersaturated, light
                  supercooled water
``ocean``         marine stratocumulus with drizzle under a trade inversion
``high_terrain``  surface pressure 600-720 hPa, orographic mixed-phase cloud
``freezing_rain`` warm nose aloft over a sub-freezing surface layer
``cold_anvil``    tropopause near 195-205 K, ice and liquid colder than -40 C
``edge``          threshold and limit cases (see ``_edge``)
``mixed``         blends of the above on the inner-domain path (GID 2)

Columns alternate the domain id between 1 (outer: RHgrd_out = 0.975 above
850 hPa) and 2 (inner: RHgrd_in = 1) and cycle the time step over
6, 20, 36 and 72 s.
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np

NZ = 50
P_TOP = 5000.0
R_D = 287.0
CP = 7.0 * R_D / 2.0
RCP = R_D / CP
G = 9.81
T0C = 273.15
NSTEPS = 10
DTS = (6.0, 20.0, 36.0, 72.0)
DX = (12000.0, 3000.0, 1000.0, 4000.0)

FIELDS = ("p", "pi", "th", "qv", "qc", "qr", "qi", "dz8w", "rho",
          "f_ice", "f_rain", "f_rimef")


def f32hex(x: float) -> str:
    return struct.pack(">f", np.float32(x)).hex().upper()


def esat_w(t: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure over water (Pa), Bolton."""
    tc = t - T0C
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def esat_i(t: np.ndarray) -> np.ndarray:
    tc = t - T0C
    return 611.2 * np.exp(21.8745584 * tc / (t - 7.66))


def qsat(t: np.ndarray, p: np.ndarray, ice: bool = False) -> np.ndarray:
    e = esat_i(t) if ice else esat_w(t)
    e = np.minimum(e, 0.5 * p)
    return 0.622 * e / (p - e)


def eta_levels(psfc: float) -> tuple[np.ndarray, np.ndarray]:
    """Interface and mass-level pressures, bottom to top, WRF-like spacing."""
    n = np.arange(NZ + 1) / NZ
    eta = 1.0 - (0.35 * n + 0.65 * n ** 1.6)
    eta[-1] = 0.0
    pint = P_TOP + eta * (psfc - P_TOP)
    pm = 0.5 * (pint[:-1] + pint[1:])
    return pint, pm


def bump(p: np.ndarray, pc: float, half: float) -> np.ndarray:
    """A smooth 0..1 bump centred on pressure pc (Pa) of half-width half."""
    x = (p - pc) / half
    return np.where(np.abs(x) < 1.0, np.cos(0.5 * math.pi * x) ** 2, 0.0)


class Column:
    def __init__(self, regime: str, idx: int, psfc: float):
        self.regime = regime
        self.idx = idx
        self.gid = 1 if idx % 2 == 0 else 2
        self.dt = DTS[idx % 4]
        self.dx = DX[(idx // 4) % 4]
        self.rainnc = 0.0 if idx % 3 else 1.25 * (idx % 7)
        self.pint, self.p = eta_levels(psfc)
        self.t = np.zeros(NZ)
        self.qv = np.zeros(NZ)
        self.qc = np.zeros(NZ)
        self.qr = np.zeros(NZ)
        self.qi = np.zeros(NZ)
        self.f_ice = np.zeros(NZ)
        self.f_rain = np.zeros(NZ)
        self.f_rimef = np.ones(NZ)

    def temperature(self, tsfc: float, lapse: float, ttrop: float,
                    inversion: float = 0.0, inv_depth: float = 0.0,
                    nose: float = 0.0, nose_p: float = 0.0) -> None:
        """Lapse-rate sounding in log-pressure height with optional features."""
        z = 7400.0 * np.log(self.pint[0] / self.p)  # rough height, m
        t = tsfc - lapse * z
        t = np.maximum(t, ttrop)
        if inversion:
            frac = np.clip(1.0 - z / max(inv_depth, 1.0), 0.0, 1.0)
            t = t - inversion * frac
            t = np.maximum(t, tsfc - inversion)
        if nose:
            t = t + nose * bump(self.p, nose_p, 9000.0)
        self.t = t

    def humidity(self, rh_low: float, rh_high: float, p_break: float,
                 ice_above: bool = False) -> None:
        rh = np.where(self.p > p_break, rh_low, rh_high)
        q = qsat(self.t, self.p, ice=False)
        if ice_above:
            qi_s = qsat(self.t, self.p, ice=True)
            q = np.where(self.t < T0C, qi_s, q)
        self.qv = np.maximum(rh * q, 1.0e-7)

    def arrays(self) -> dict[str, np.ndarray]:
        tv = self.t * (1.0 + 0.608 * self.qv)
        rho = self.p / (R_D * tv)
        dz = R_D * tv / G * np.log(self.pint[:-1] / self.pint[1:])
        pi = (self.p / 1.0e5) ** RCP
        th = self.t / pi
        return {
            "p": self.p, "pi": pi, "th": th, "qv": self.qv, "qc": self.qc,
            "qr": self.qr, "qi": self.qi, "dz8w": dz, "rho": rho,
            "f_ice": self.f_ice, "f_rain": self.f_rain,
            "f_rimef": self.f_rimef,
        }


def _convective(i: int, j: int) -> Column:
    c = Column("convective", i, 100800.0 - 400.0 * j)
    c.temperature(299.5 + 0.9 * j, 0.0068 + 0.0002 * j, 196.0 + j)
    c.humidity(0.93 + 0.01 * j, 0.75, 45000.0)
    up = bump(c.p, 70000.0, 25000.0)
    c.qv = c.qv * (1.0 + (0.01 + 0.005 * j) * up)  # 1-4.5 % supersaturated
    warm = c.t > T0C
    c.qc = (0.4e-3 + 0.35e-3 * j) * bump(c.p, 75000.0, 20000.0)
    c.qr = np.where(warm, (0.5e-3 + 0.7e-3 * j) * bump(c.p, 85000.0, 20000.0),
                    0.0)
    c.qi = np.where(~warm, (0.1e-3 + 0.25e-3 * j) * bump(c.p, 40000.0, 18000.0),
                    0.0)
    c.f_rimef = np.where(c.qi > 0, 1.0 + 2.5 * j, 1.0)
    return c


def _stable_night(i: int, j: int) -> Column:
    c = Column("stable_night", i, 101500.0 - 600.0 * j)
    c.temperature(272.0 + 2.5 * j, 0.0055, 210.0, inversion=5.0 + 0.7 * j,
                  inv_depth=250.0 + 40.0 * j)
    c.humidity(0.35 + 0.05 * j, 0.25, 98000.0)
    fog = np.arange(NZ) < 3
    c.qv = np.where(fog, qsat(c.t, c.p) * (1.0 + 0.002 * j), c.qv)
    c.qc = np.where(fog, (0.04e-3 + 0.04e-3 * j), 0.0)
    if j >= 6:  # ice fog in the coldest members
        c.qi = np.where(fog & (c.t < T0C), 0.01e-3, 0.0)
    return c


def _snow(i: int, j: int) -> Column:
    c = Column("snow", i, 99000.0 - 1500.0 * j)
    c.temperature(268.5 - 1.6 * j, 0.0062, 205.0)
    c.humidity(0.97, 0.7, 50000.0, ice_above=True)
    c.qv = c.qv * (1.0 + (0.02 + 0.01 * j) * bump(c.p, 75000.0, 20000.0))
    c.qi = (0.05e-3 + 0.15e-3 * j) * bump(c.p, 78000.0, 25000.0)
    c.qc = (0.02e-3 * j) * bump(c.p, 85000.0, 8000.0)
    c.f_rimef = np.where(c.qi > 0, 1.0 + 0.6 * j, 1.0)
    return c


def _ocean(i: int, j: int) -> Column:
    c = Column("ocean", i, 101200.0 + 100.0 * j)
    c.temperature(288.0 + 1.5 * j, 0.0062, 198.0, nose=4.0 + 0.5 * j,
                  nose_p=86000.0)
    c.humidity(0.92 + 0.008 * j, 0.3, 88000.0)
    deck = bump(c.p, 92000.0, 3500.0)
    c.qv = np.where(deck > 0, qsat(c.t, c.p) * (1.0 + 0.001 * j * deck), c.qv)
    c.qc = (0.2e-3 + 0.08e-3 * j) * deck
    c.qr = (10.0 ** (-6.0 + 0.3 * j)) * bump(c.p, 95000.0, 5000.0)
    return c


def _high_terrain(i: int, j: int) -> Column:
    c = Column("high_terrain", i, 60000.0 + 1500.0 * j)
    c.temperature(270.0 + 3.0 * j, 0.0070, 200.0)
    c.humidity(0.96, 0.6, 45000.0)
    c.qv = c.qv * (1.0 + 0.015 * bump(c.p, 52000.0, 6000.0))
    cold = c.t < T0C
    c.qc = (0.1e-3 + 0.1e-3 * j) * bump(c.p, 55000.0, 6000.0)
    c.qi = np.where(cold, (0.05e-3 + 0.1e-3 * j) * bump(c.p, 45000.0, 8000.0),
                    0.0)
    c.qr = np.where(~cold, 0.2e-3 * j * bump(c.p, 58000.0, 4000.0), 0.0)
    c.f_rimef = np.where(c.qi > 0, 1.0 + 4.0 * j, 1.0)
    return c


def _freezing_rain(i: int, j: int) -> Column:
    c = Column("freezing_rain", i, 100500.0 - 300.0 * j)
    c.temperature(270.5 - 0.5 * j, 0.0050, 205.0, inversion=6.0,
                  inv_depth=700.0, nose=7.0 + 0.6 * j, nose_p=82000.0)
    c.humidity(0.99, 0.85, 60000.0)
    warm = c.t > T0C
    c.qr = (0.3e-3 + 0.2e-3 * j) * (bump(c.p, 85000.0, 12000.0)
                                    + bump(c.p, 97000.0, 5000.0))
    c.qi = np.where(~warm, (0.1e-3 + 0.05e-3 * j) * bump(c.p, 65000.0, 12000.0),
                    0.0)
    c.qc = 0.05e-3 * bump(c.p, 90000.0, 6000.0)
    c.f_rimef = np.where(c.qi > 0, 1.0 + 1.5 * j, 1.0)
    return c


def _cold_anvil(i: int, j: int) -> Column:
    c = Column("cold_anvil", i, 100000.0)
    c.temperature(303.0, 0.0076 + 0.0001 * j, 192.0 + 1.6 * j)
    c.humidity(0.9, 0.95, 70000.0, ice_above=True)
    very_cold = c.t < T0C - 40.0
    c.qc = np.where(very_cold, (0.05e-3 + 0.05e-3 * j)
                    * bump(c.p, 25000.0, 10000.0), 0.0)
    c.qi = (0.2e-3 + 0.3e-3 * j) * bump(c.p, 22000.0, 12000.0)
    c.f_rimef = np.where(c.qi > 0, 1.0 + 5.0 * j, 1.0)
    return c


def _edge(i: int, j: int) -> Column:
    c = Column("edge", i, 100000.0)
    c.temperature(285.0, 0.0065, 205.0)
    c.humidity(0.5, 0.3, 70000.0)
    k = np.arange(NZ)
    if j == 0:  # bone dry, tiny vapour everywhere
        c.qv = np.full(NZ, 1.0e-9)
    elif j == 1:  # condensate on both sides of EPSQ (1e-12) and EPSQ1
        vals = np.array([0.0, 1.0e-12, 1.001e-12, 1.0011e-12, 2.0e-12,
                         1.0e-11, 1.0e-10])
        c.qc = vals[k % 7]
        c.qr = vals[(k + 2) % 7]
        c.qi = vals[(k + 4) % 7]
    elif j == 2:  # ice in warm air (melting) with very large rime factors
        c.qi = 0.8e-3 * bump(c.p, 90000.0, 12000.0)
        c.f_rimef = np.where(c.qi > 0, 30.0 + 0.5 * (k % 40), 1.0)
    elif j == 3:  # temperatures exactly at 0 C and at T_ICE (-40 C)
        c.t = np.where(k % 3 == 0, T0C, np.where(k % 3 == 1, T0C - 40.0,
                                                 c.t))
        c.qc = np.full(NZ, 0.1e-3)
        c.qi = np.where(k % 2 == 0, 0.05e-3, 0.0)
        c.qv = qsat(c.t, c.p)
    elif j == 4:  # very heavy rain and cloud water
        c.qr = 15.0e-3 * bump(c.p, 80000.0, 20000.0)
        c.qc = 5.0e-3 * bump(c.p, 70000.0, 20000.0)
        c.qv = qsat(c.t, c.p)
    elif j == 5:  # 20 % supersaturation over water near the ground
        c.qv = np.where(k < 8, 1.2 * qsat(c.t, c.p), c.qv)
    elif j == 6:  # 181 K (just inside the 180 K table floor) aloft
        c.temperature(240.0, 0.0090, 181.0)
        c.humidity(0.8, 0.9, 80000.0, ice_above=True)
        c.qi = np.where(c.p < 30000.0, 1.0e-6, 0.0)
        c.qc = np.where(c.p < 20000.0, 1.0e-7, 0.0)
    else:  # rime factor below 1 on input, F_ICE / F_RAIN inconsistent
        c.qi = 0.3e-3 * bump(c.p, 50000.0, 15000.0)
        c.qc = 0.3e-3 * bump(c.p, 80000.0, 15000.0)
        c.qr = 0.2e-3 * bump(c.p, 90000.0, 10000.0)
        c.f_rimef = np.where(c.qi > 0, 0.5, 0.0)
        c.f_ice = np.full(NZ, 0.37)
        c.f_rain = np.full(NZ, 0.91)
    return c


def _mixed(i: int, j: int) -> Column:
    makers = (_convective, _snow, _ocean, _high_terrain, _freezing_rain,
              _cold_anvil, _stable_night, _convective)
    c = makers[j](i, (j * 3 + 1) % 8)
    c.regime = "mixed"
    c.gid = 2
    c.qc = c.qc * 1.7
    c.qr = c.qr * 0.6
    return c


GROUPS = (_convective, _stable_night, _snow, _ocean, _high_terrain,
          _freezing_rain, _cold_anvil, _edge, _mixed)


def build() -> list[Column]:
    cols = []
    for g, maker in enumerate(GROUPS):
        for j in range(8):
            cols.append(maker(len(cols), j))
    return cols


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cols = build()
    with open(out / "columns_inputs.csv", "w", newline="\n") as fc, \
            open(out / "columns.in", "w", newline="\n") as fi:
        fc.write("# Ferrier-Aligo (WRF 4.6.1 mp_physics=5) column-oracle "
                 "inputs; floats are binary32 bit patterns (hex)\n")
        fc.write("# column rows: C,col,regime,gid,nsteps,dt,dx,rainnc\n")
        fc.write("# level rows (k=1 is the surface): L,col,k,"
                 + ",".join(FIELDS) + "\n")
        fi.write(f"{len(cols):6d} {NZ:6d}\n")
        for c in cols:
            a = c.arrays()
            fc.write(f"C,{c.idx},{c.regime},{c.gid},{NSTEPS},{f32hex(c.dt)},"
                     f"{f32hex(c.dx)},{f32hex(c.rainnc)}\n")
            fi.write(f"{c.gid:6d} {NSTEPS:6d} {f32hex(c.dt)} {f32hex(c.dx)} "
                     f"{f32hex(c.rainnc)}\n")
            for k in range(NZ):
                words = [f32hex(a[f][k]) for f in FIELDS]
                fc.write(f"L,{c.idx},{k + 1}," + ",".join(words) + "\n")
                fi.write(" ".join(words) + "\n")
    print(f"wrote {len(cols)} columns x {NZ} levels to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: make_columns.py OUT_DIR")
    main(Path(sys.argv[1]))
