#!/usr/bin/env python3
"""Extra Ferrier-Aligo (WRF mp_physics=5) oracle columns aimed at the code
paths the lane's 72 columns never executed.

A gcov build of the byte-unmodified module, run over the lane's column set,
left these computational branches of ``module_mp_fer_hires.F`` at zero
executions:

* FPVS / FPVS0 outside the 180-330 K table (the closed-form fallbacks);
* the CONDENSE and DEPOSIT non-convergence fallbacks (``CONDENSE=DWVinp``,
  ``DEPOSIT=DWVinp``);
* the extremely-heavy-rain cap (``INDEXR=MDRmax``);
* the large-ice size cap (``INDEXS=MDImax``) on the first sizing pass;
* the ``RimeF=RimeF1`` arm for a vanishing ice budget;
* homogeneous freezing below the 180 K table floor.

The columns below aim at those, and add inputs a WRF run really hands the
scheme that the lane did not use: long time steps (90 to 300 s), small
negative mixing ratios after advection, subnormal values, very thin and very
thick layers, a 10 hPa model top with stratospheric condensate, domain id 3,
a level exactly at the 850 hPa RHgrd switch, and 60-step integrations.

Output format is identical to ``make_columns.py`` so ``run_ferrier.F90`` and
``compare_ferrier.py`` are reused unchanged.  No random numbers.

usage: make_columns_refute.py OUT_DIR
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_ferrier_make_columns", _HERE.parent / "make_columns.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

NZ = mc.NZ
T0C = mc.T0C
bump = mc.bump
qsat = mc.qsat


class Col(mc.Column):
    """A lane Column with per-column steps, dt, gid and a dz multiplier."""

    def __init__(self, regime, idx, psfc, *, dt, gid, nsteps=10,
                 p_top=None):
        if p_top is not None:
            saved = mc.P_TOP
            mc.P_TOP = p_top
            try:
                super().__init__(regime, idx, psfc)
            finally:
                mc.P_TOP = saved
        else:
            super().__init__(regime, idx, psfc)
        self.dt = dt
        self.gid = gid
        self.nsteps = nsteps
        self.dz_scale = np.ones(NZ)
        self.overrides: dict[str, np.ndarray] = {}

    def arrays(self):
        a = super().arrays()
        a["dz8w"] = a["dz8w"] * self.dz_scale
        a.update(self.overrides)
        return a


def build() -> list[Col]:
    cols: list[Col] = []
    k = np.arange(NZ)

    def add(c):
        cols.append(c)
        return c

    # 1. table_hot: air above the 330 K table top near the ground (FPVS and
    #    FPVS0 closed-form arms), with cloud and rain so CONDENSE runs there.
    for j, tsfc in enumerate((331.0, 334.0, 338.0, 345.0)):
        c = Col("table_hot", len(cols), 100500.0, dt=(20.0, 60.0)[j % 2],
                gid=1 + j % 2)
        c.temperature(tsfc, 0.0098, 205.0)
        c.humidity(0.6 + 0.1 * j, 0.4, 85000.0)
        c.qc = np.where(k < 4, 0.2e-3, 0.0)
        c.qr = np.where(k < 6, 0.5e-3 * (j + 1), 0.0)
        add(c)

    # 2. table_cold: polar-winter stratosphere below the 180 K table floor,
    #    10 hPa top, ice and supercooled liquid there (FPVS cold arm,
    #    homogeneous freezing below XMIN, WARN1).
    for j, tmin in enumerate((178.0, 174.0, 168.0, 160.0)):
        c = Col("table_cold", len(cols), 99000.0, dt=(36.0, 90.0)[j % 2],
                gid=1 + j % 2, p_top=1000.0)
        c.temperature(250.0, 0.0085, tmin)
        c.humidity(0.7, 0.9, 70000.0, ice_above=True)
        cold = c.t < 185.0
        c.qi = np.where(cold, 2.0e-6 * (j + 1), 0.0)
        c.qc = np.where(cold, 1.0e-6 * (j + 1), 0.0)
        # stratospheric condensate above 75 hPa (the WARN4 arm)
        c.qi = np.where(c.p < 7500.0, 2.0e-3, c.qi)
        add(c)

    # 3. condense_nc: huge water supersaturation where the Asai iteration
    #    may not converge in 10 passes (CONDENSE fallback).
    for j, ss in enumerate((1.5, 2.5, 4.0, 8.0)):
        c = Col("condense_nc", len(cols), 100000.0, dt=(6.0, 72.0)[j % 2],
                gid=1 + j % 2)
        c.temperature(300.0 - 4.0 * j, 0.0065, 200.0)
        c.humidity(0.8, 0.5, 70000.0)
        warm = c.t > T0C + 2.0
        c.qv = np.where(warm & (k % 2 == 0), ss * qsat(c.t, c.p), c.qv)
        c.qc = np.where(warm & (k % 3 == 0), 1.0e-3, 0.0)
        add(c)

    # 4. deposit_nc: huge ice supersaturation, cold and low pressure
    #    (DEPOSIT fallback), with and without existing ice.
    for j, ss in enumerate((2.0, 4.0, 10.0, 30.0)):
        c = Col("deposit_nc", len(cols), 100000.0, dt=(20.0, 120.0)[j % 2],
                gid=1 + j % 2)
        c.temperature(270.0, 0.0080, 195.0 + 3.0 * j)
        c.humidity(0.9, 0.6, 60000.0, ice_above=True)
        cold = c.t < T0C - 15.0
        c.qv = np.where(cold, ss * qsat(c.t, c.p, ice=True), c.qv)
        c.qi = np.where(cold & (k % 2 == 0), 0.2e-3, 0.0)
        c.f_rimef = np.where(c.qi > 0, 3.0, 1.0)
        add(c)

    # 5. extreme_rain: rain water at and beyond the MDRmax cap.
    for j, qr in enumerate((25.0e-3, 35.0e-3, 50.0e-3, 80.0e-3)):
        c = Col("extreme_rain", len(cols), 100800.0, dt=(36.0, 150.0)[j % 2],
                gid=1 + j % 2)
        c.temperature(302.0, 0.0065, 198.0)
        c.humidity(0.95, 0.8, 50000.0)
        c.qr = qr * bump(c.p, 80000.0, 25000.0)
        c.qc = 3.0e-3 * bump(c.p, 65000.0, 15000.0)
        add(c)

    # 6. extreme_ice: very large ice contents (hail-like), rime factors at
    #    and above RFmax, near and above 0 C, sizing caps.
    for j in range(6):
        qi = (8.0e-3, 15.0e-3, 30.0e-3, 50.0e-3, 12.0e-3, 20.0e-3)[j]
        rf = (1.0, 45.26, 60.0, 100.0, 1000.0, 46.0)[j]
        c = Col("extreme_ice", len(cols), 100000.0,
                dt=(20.0, 90.0, 180.0)[j % 3], gid=1 + j % 3)
        c.temperature(285.0 - 3.0 * j, 0.0070, 205.0)
        c.humidity(0.95, 0.9, 50000.0, ice_above=True)
        c.qi = qi * bump(c.p, 60000.0, 30000.0)
        c.qc = 1.5e-3 * bump(c.p, 70000.0, 15000.0)
        c.f_rimef = np.where(c.qi > 0, rf, 1.0)
        add(c)

    # 7. long_dt: ordinary convective and snow columns at coarse-grid steps.
    for j, dt in enumerate((90.0, 120.0, 180.0, 300.0)):
        c = mc._convective(len(cols), 2 * j + 1)
        c2 = Col("long_dt", len(cols), 100000.0, dt=dt, gid=1 + j % 2)
        c2.__dict__.update({k_: v for k_, v in c.__dict__.items()
                            if k_ not in ("regime", "idx", "dt", "gid")})
        add(c2)
        s = mc._snow(len(cols), 2 * j)
        s2 = Col("long_dt", len(cols), 100000.0, dt=dt, gid=2 - j % 2)
        s2.__dict__.update({k_: v for k_, v in s.__dict__.items()
                            if k_ not in ("regime", "idx", "dt", "gid")})
        add(s2)

    # 8. negative: small negative mixing ratios as advection leaves them,
    #    including negative vapour and negative total condensate.
    for j in range(4):
        c = Col("negative", len(cols), 100500.0, dt=(20.0, 60.0)[j % 2],
                gid=1 + j % 2)
        c.temperature(290.0, 0.0065, 205.0)
        c.humidity(0.85, 0.5, 60000.0)
        c.qc = 0.3e-3 * bump(c.p, 80000.0, 15000.0)
        c.qr = 0.4e-3 * bump(c.p, 90000.0, 10000.0)
        c.qi = np.where(c.t < T0C, 0.2e-3 * bump(c.p, 45000.0, 15000.0), 0.0)
        neg = -(10.0 ** (-9.0 + j))
        c.qc = np.where(k % 5 == 1, neg, c.qc)
        c.qr = np.where(k % 5 == 2, neg, c.qr)
        c.qi = np.where(k % 5 == 3, neg, c.qi)
        if j >= 2:
            c.qv = np.where(k % 11 == 4, -1.0e-6, c.qv)
        add(c)

    # 9. subnormal: binary32 subnormals in every mass field and a tiny
    #    rime factor (FTZ-sensitive for a port).
    for j in range(2):
        c = Col("subnormal", len(cols), 100000.0, dt=(20.0, 36.0)[j],
                gid=1 + j)
        c.temperature(280.0, 0.0065, 205.0)
        c.humidity(0.7, 0.4, 70000.0)
        tiny = np.float32(1.0e-40) * (j + 1)
        c.qc = np.where(k % 3 == 0, tiny, 0.0)
        c.qr = np.where(k % 3 == 1, tiny, 0.0)
        c.qi = np.where(k % 3 == 2, tiny, 0.0)
        c.qv = np.where(k > 40, tiny, c.qv)
        add(c)

    # 10. thin_layers: 2-10 m layers with heavy rain and tiny ice (the
    #     RimeF=RimeF1 arm, sedimentation through thin layers).
    for j in range(4):
        c = Col("thin_layers", len(cols), 100800.0,
                dt=(6.0, 20.0, 60.0, 120.0)[j], gid=1 + j % 2)
        c.temperature(276.0 - 2.0 * j, 0.0060, 205.0)
        c.humidity(0.95, 0.7, 70000.0)
        c.dz_scale = np.where(k < 20, (2.0 + 2.0 * j) / 150.0, 1.0)
        c.qr = 2.0e-3 * bump(c.p, 98000.0, 4000.0)
        c.qi = np.where(c.t < T0C, 2.0e-12 + 1.0e-12 * (k % 4), 0.0)
        c.qi = np.where((k > 25) & (k < 35), 0.5e-3, c.qi)
        add(c)

    # 11. thick_layers: 1-2 km layers (coarse vertical grids).
    for j in range(2):
        c = Col("thick_layers", len(cols), 100800.0, dt=(60.0, 180.0)[j],
                gid=1 + j)
        c.temperature(295.0, 0.0065, 205.0)
        c.humidity(0.9, 0.6, 50000.0)
        c.dz_scale = np.full(NZ, 6.0 + 4.0 * j)
        c.qr = 1.0e-3 * bump(c.p, 85000.0, 15000.0)
        c.qi = np.where(c.t < T0C, 1.0e-3 * bump(c.p, 40000.0, 15000.0), 0.0)
        c.qc = 0.5e-3 * bump(c.p, 70000.0, 20000.0)
        add(c)

    # 12. rhgrd_switch: a level exactly at P_RHgrd_out = 850 hPa, ice
    #     supersaturated, for domain ids 1, 2 and 3.
    for j, gid in enumerate((1, 2, 3)):
        c = Col("rhgrd_switch", len(cols), 100000.0, dt=36.0, gid=gid)
        c.p = c.p.copy()
        kk = int(np.argmin(np.abs(c.p - 85000.0)))
        c.p[kk] = 85000.0
        c.p[kk + 1] = np.nextafter(np.float32(85000.0), np.float32(0.0))
        c.temperature(268.0, 0.0065, 205.0)
        c.humidity(1.0, 0.9, 80000.0, ice_above=True)
        c.qv = c.qv * 1.03
        c.qi = 0.1e-3 * bump(c.p, 85000.0, 5000.0)
        c.qc = 0.1e-3 * bump(c.p, 85000.0, 5000.0)
        add(c)

    # 13. long_run: 60 steps so sedimented rain and snow reach the surface
    #     and the state leaves the initial regime.
    for j, maker in enumerate((mc._convective, mc._freezing_rain,
                               mc._snow, mc._cold_anvil)):
        base = maker(len(cols), 7 - j)
        c = Col("long_run", len(cols), 100000.0,
                dt=(20.0, 36.0, 60.0, 72.0)[j], gid=1 + j % 2, nsteps=60)
        c.__dict__.update({k_: v for k_, v in base.__dict__.items()
                           if k_ not in ("regime", "idx", "dt", "gid")})
        add(c)

    # 14. rain_cap: very thick layers holding 30-60 g/kg of rain at short
    #     steps, so the rain content after fallout stays above RQR_DRmax
    #     (DSD2 ``INDEXR=MDRmax``).
    for j in range(4):
        c = Col("rain_cap", len(cols), 100800.0, dt=(6.0, 6.0, 20.0, 36.0)[j],
                gid=1 + j % 2)
        c.temperature(300.0, 0.0065, 200.0)
        c.humidity(0.97, 0.8, 50000.0)
        c.dz_scale = np.full(NZ, (10.0, 20.0, 15.0, 30.0)[j])
        c.qr = (30.0e-3 + 10.0e-3 * j) * bump(c.p, 85000.0, 20000.0)
        add(c)

    # 15. clamped_esw: warm cloudy air at very low pressure, where the
    #     saturation vapour pressure is clamped at 0.99*p and the Asai
    #     iteration in CONDENSE converges slowly (its 10-pass fallback).
    for j in range(4):
        c = Col("clamped_esw", len(cols), 99000.0, dt=(20.0, 60.0)[j % 2],
                gid=1 + j % 2, p_top=500.0)
        c.temperature(300.0, 0.0065, 205.0)
        c.humidity(0.7, 0.5, 70000.0)
        top = c.p < 3000.0
        c.t = np.where(top, 280.0 + 10.0 * j, c.t)
        c.qc = np.where(top, (5.0e-3, 10.0e-3, 20.0e-3, 40.0e-3)[j], 0.0)
        c.qv = np.where(top, 1.0e-6, c.qv)
        add(c)

    # 16. vanishing_ice: sub-metre layers with ice just above EPSQ beside
    #     cloud water (the ``RimeF=RimeF1`` arm for a vanishing ice budget).
    for j in range(4):
        c = Col("vanishing_ice", len(cols), 100000.0,
                dt=(6.0, 20.0, 36.0, 72.0)[j], gid=1 + j % 2)
        c.temperature(268.0, 0.0060, 205.0)
        c.humidity(1.0, 0.9, 70000.0)
        c.dz_scale = np.where(k < 30, (0.2, 0.5, 1.0, 0.1)[j] / 150.0, 1.0)
        c.qi = np.where(k < 30, 1.1e-12 + 0.5e-12 * (k % 5), 0.0)
        c.qc = np.where(k < 30, 0.3e-3, 0.0)
        c.f_rimef = np.where(c.qi > 0, 2.0, 1.0)
        add(c)

    return cols


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cols = build()
    f32hex = mc.f32hex
    with open(out / "columns_inputs.csv", "w", newline="\n") as fc, \
            open(out / "columns.in", "w", newline="\n") as fi:
        fc.write("# Ferrier-Aligo refute columns; floats are binary32 bit "
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
