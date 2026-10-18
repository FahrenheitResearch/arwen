"""MYNN wrapper column oracle: WRF v4.6.1 ``mynnedmf_wrapper_run`` against WOOF.

The other MYNN oracles in this folder call ``mynn_bl_driver`` (or one of its
leaves) with the driver's own specific-humidity inputs.  WRF's PBL driver
does not: ``module_pbl_driver.F`` calls ``mynnedmf_wrapper_run``
(``phys/module_bl_mynn_wrapper.F``), which converts the mixing ratios to
specific humidity, calls the driver, and converts the moisture tendencies
and the subgrid cloud water back.  WOOF's production entry is
:func:`gpuwm.core.mynn_pbl_runtime.mynn_pbl_step`, which owns the same
conversions.  This oracle grades that pair, entry to entry.

What it does
------------
* ``gen``: build a catalogue of realistic columns (convective and stable
  land, ocean, high terrain, snow and ice, moist convection, high wind, and
  edge cases) and write them as raw float32 words to an input file.  The
  Fortran harness ``run_wrapper_columns.F90`` reads the same words, so both
  sides start from identical bits.
* The harness (``build_columns.sh``) compiles the pinned, unmodified WRF
  v4.6.1 sources -- ``ccpp_kind_types.F``, ``module_model_constants.F``,
  ``module_bl_mynn_common.F``, ``module_bl_mynn.F`` and
  ``module_bl_mynn_wrapper.F`` -- with no stub, and integrates every column
  for ``nstep`` steps (a cold start, then warm steps), applying the
  wrapper's own tendencies between steps.  Arrays are laid out the way WRF
  lays them out for physics: ``kme = kde = nz + 1`` with ``kte = nz``; the
  unused ``kme`` slot of every input is NaN, so a read of it would show.
* ``pack``: fold the inputs and the harness's binary output into one
  compressed ``.npz`` fixture.
* ``grade``: run WOOF's ``mynn_pbl_step`` on the GPU over the same columns
  and report per-output, per-regime max ULP, both step by step from the
  oracle's recorded inputs (``replay``) and free-running from step 1
  (``free``), where any rounding difference compounds.

Columns are grouped by (dx, delt, FLAG_QS) because WRF passes one ``dxc``
and one ``delt`` per call and derives FLAG_QS from the microphysics; both
sides make the same calls over the same groups.

Only numpy is imported at module level: ``gen``/``pack`` run on a box that
has no CUDA.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = 0x4D594E57  # "MYNW"
VERSION = 1
NZ = 40

#: Column groups: one WRF call each per step.  (dx m, delt s, FLAG_QS).
GROUPS = {
    "A": (3000.0, 20.0, True),    # HRRR-like convection-allowing
    "B": (1000.0, 6.0, True),
    "C": (12000.0, 60.0, True),
    "D": (27000.0, 150.0, True),  # mesoscale, scale-aware weight near 1
    "E": (250.0, 1.5, True),      # grey zone
    "F": (100.0, 0.5, True),      # LES-like grey zone
    "G": (3000.0, 20.0, False),   # a microphysics with no snow (FLAG_QS off)
}
GROUP_ORDER = tuple(GROUPS)

SCALAR_INPUTS = ("xland", "ts", "qsfc", "ps", "ust", "ch", "hfx", "qfx",
                 "wspd", "znt")
PROFILE_INPUTS = ("dz", "u", "v", "w", "th", "qv", "qc", "qi", "qs", "p",
                  "exner", "rho", "t3d", "rthraten", "qnc", "qni", "qnwfa",
                  "qnifa", "spp")
#: Recorded each step before the call (the state that step started from).
STEP_PROFILE_IN = ("u", "v", "th", "qv", "qc", "qi", "qs", "t3d", "rho",
                   "qnc", "qni", "qnwfa", "qnifa")
#: Harness variants (run_wrapper_columns.F90 VARIANT argument).
VARIANTS = ("base", "mixscalars", "spp")
QN_TENDENCIES = (("qnc", "rqncblten", "nc", "dnc"),
                 ("qni", "rqniblten", "ni", "dni"),
                 ("qnwfa", "rqnwfablten", "nwfa", "dnwfa"),
                 ("qnifa", "rqnifablten", "nifa", "dnifa"))
STATE_PROFILE = ("qke", "tsq", "qsq", "cov", "el_pbl", "sh3d", "sm3d",
                 "qc_bl", "qi_bl", "cldfra_bl", "exch_h", "exch_m")
STEP_COLUMN_IN = ("pblh", "rmol", "ust", "hfx", "qfx", "wspd", "znt")
STEP_COLUMN_IN_INT = ("kpbl",)
#: Written by the wrapper.
PROFILE_OUT = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
               "rqiblten", "exch_h", "exch_m", "rqncblten", "rqniblten",
               "rqnwfablten", "rqnifablten", "qke", "tsq", "qsq", "cov",
               "el_pbl", "sh3d", "sm3d", "qc_bl", "qi_bl", "cldfra_bl")
COLUMN_OUT = ("pblh", "rmol", "maxwidth", "maxmf", "ztop_plume", "ust",
              "hfx", "qfx", "wspd", "znt")
COLUMN_OUT_INT = ("kpbl", "ktop_plume")

R_D, CP, G = 287.0, 1004.5, 9.81
KAPPA = R_D / CP


# ---------------------------------------------------------------------------
# Column catalogue
# ---------------------------------------------------------------------------

@dataclass
class Spec:
    """One column, in physical terms; :func:`build_column` makes the words."""

    name: str
    regime: str
    group: str = "A"
    xland: float = 1.0
    ps: float = 100000.0
    grid: tuple = (40.0, 1.10, 900.0)     # dz1, stretch, dzmax
    th0: float = 295.0                    # theta of the lowest layer
    zi: float = 1000.0                    # mixed-layer top
    gml: float = 0.0                      # mixed-layer dtheta/dz (K/m)
    dthinv: float = 3.0                   # inversion jump (K)
    gft: float = 0.004                    # free-troposphere dtheta/dz
    ztp: float = 11000.0                  # tropopause
    gst: float = 0.02                     # stratosphere dtheta/dz
    qv0: float = 0.010                    # mixing ratio of the lowest layer
    gqml: float = -2.0e-7                 # mixed-layer dqv/dz (per m)
    dqinv: float = 0.4                    # fractional drop at the inversion
    hq: float = 2500.0                    # qv scale height above zi
    rhmax: float = 0.97                   # cap outside cloud layers
    clouds: list = field(default_factory=list)   # (zb, zt, q, phase[, rh])
    snow: list = field(default_factory=list)     # (zb, zt, qs)
    u0: float = 5.0
    du: float = 0.002
    v0: float = -1.0
    dv: float = 0.0008
    jet: tuple | None = None              # (zj, uj, width)
    w: tuple = (0.0, 0.0, 0.0, 1.0)       # (const, amp, zc, width)
    rad: list = field(default_factory=list)      # (zb, zt, rate K/s)
    ts: float | None = None               # skin temperature (default th0*exner1+dT)
    dts: float = 0.0
    hfx: float = 0.0
    qfx: float = 0.0
    ust: float = 0.3
    znt: float = 0.1
    ch: float = 0.01
    wspd: float | None = None
    qsfc: float | None = None


def _esat(tk: float) -> float:
    if tk >= 273.15:
        return 611.2 * math.exp(17.67 * (tk - 273.15) / (tk - 29.65))
    return 611.2 * math.exp(21.8745584 * (tk - 273.16) / (tk - 7.66))


def _qsat(tk: float, p: float) -> float:
    es = min(_esat(tk), 0.5 * p)
    return 0.622 * es / (p - es)


def _grid(spec: Spec) -> np.ndarray:
    dz1, ratio, dzmax = spec.grid
    return np.array([min(dz1 * ratio ** k, dzmax) for k in range(NZ)])


def _theta(spec: Spec, z: float) -> float:
    if z <= spec.zi:
        return spec.th0 + spec.gml * z
    th = spec.th0 + spec.gml * spec.zi + spec.dthinv
    if z <= spec.ztp:
        return th + spec.gft * (z - spec.zi)
    return th + spec.gft * (spec.ztp - spec.zi) + spec.gst * (z - spec.ztp)


def _qv_target(spec: Spec, z: float) -> float:
    if z <= spec.zi:
        return max(spec.qv0 + spec.gqml * z, 1.0e-6)
    top = max(spec.qv0 + spec.gqml * spec.zi, 1.0e-6) * (1.0 - spec.dqinv)
    return max(top * math.exp(-(z - spec.zi) / spec.hq), 2.0e-6)


def build_column(spec: Spec) -> dict:
    """Physical profiles (float64) for one spec."""

    dz = _grid(spec)
    zw = np.concatenate([[0.0], np.cumsum(dz)])
    zm = 0.5 * (zw[:-1] + zw[1:])
    out = {name: np.zeros(NZ) for name in PROFILE_INPUTS}
    out["dz"] = dz
    pw = spec.ps
    for k in range(NZ):
        th = _theta(spec, zm[k])
        qv = _qv_target(spec, zm[k])
        # Hydrostatic: two fixed-point passes for the half-layer and layer.
        tv = th * (pw / 1.0e5) ** KAPPA * (1.0 + 0.608 * qv)
        for _ in range(3):
            pm = pw * math.exp(-G * 0.5 * dz[k] / (R_D * tv))
            tk = th * (pm / 1.0e5) ** KAPPA
            tv = tk * (1.0 + 0.608 * qv)
        qs_ = _qsat(tk, pm)
        qc = qi = 0.0
        in_cloud = False
        for cloud in spec.clouds:
            zb, zt, q, phase = cloud[:4]
            rh = cloud[4] if len(cloud) > 4 else 1.0
            if zb <= zm[k] <= zt:
                in_cloud = True
                qv = rh * qs_
                if phase == "liq":
                    qc += q
                elif phase == "ice":
                    qi += q
                elif phase == "mix":
                    qc += 0.5 * q
                    qi += 0.5 * q
                else:  # "auto": by temperature
                    frac = min(max((tk - 248.0) / 20.0, 0.0), 1.0)
                    qc += frac * q
                    qi += (1.0 - frac) * q
        if not in_cloud:
            qv = min(qv, spec.rhmax * qs_)
        qsn = sum(q for zb, zt, q in spec.snow if zb <= zm[k] <= zt)
        u = spec.u0 + spec.du * zm[k]
        if spec.jet is not None:
            zj, uj, width = spec.jet
            u += uj * math.exp(-((zm[k] - zj) / width) ** 2)
        v = spec.v0 + spec.dv * zm[k]
        wc, wa, wz, ww = spec.w
        w = wc + wa * math.exp(-((zw[k] - wz) / ww) ** 2)
        rad = sum(r for zb, zt, r in spec.rad if zb <= zm[k] <= zt)
        out["u"][k], out["v"][k], out["w"][k] = u, v, w
        out["th"][k], out["qv"][k], out["qc"][k] = th, qv, qc
        out["qi"][k], out["qs"][k], out["p"][k] = qi, qsn, pm
        out["rthraten"][k] = rad
        land = spec.xland < 1.5
        out["qnc"][k] = ((3.0e8 if land else 8.0e7) * min(qc / 1.0e-4, 1.0)
                         + 1.0e6) if qc > 0.0 else 0.0
        out["qni"][k] = (min(5.0e4 * math.sqrt(qi / 1.0e-5), 5.0e5)
                         if qi > 0.0 else 0.0)
        out["qnwfa"][k] = ((2.0e9 if land else 5.0e8)
                           * math.exp(-zm[k] / 1800.0) + 5.0e7)
        out["qnifa"][k] = 1.0e6 * math.exp(-zm[k] / 3000.0) + 1.0e4
        phase = (zlib.crc32(spec.name.encode()) % 6283) / 1000.0
        out["spp"][k] = (0.9 * math.sin(2.0 * math.pi * zm[k] / 3000.0
                                        + phase)
                         * math.exp(-zm[k] / 9000.0)
                         + 0.1 * math.cos(3.0 * phase))
        pw = pw * math.exp(-G * dz[k] / (R_D * tv))
    # WRF's own fp32 relations for the derived thermodynamics.
    f32 = np.float32
    p32 = out["p"].astype(f32)
    th32 = out["th"].astype(f32)
    exner = (p32 / f32(1.0e5)) ** f32(R_D / (7.0 * R_D / 2.0))
    t3d = (th32 * exner).astype(f32)
    out["exner"] = exner
    out["t3d"] = t3d
    out["rho"] = (p32 / (f32(R_D) * t3d)).astype(f32)
    tk1 = float(t3d[0])
    ts = spec.ts if spec.ts is not None else tk1 + spec.dts
    if spec.wspd is None:
        wspd = max(math.hypot(out["u"][0], out["v"][0]), 0.1)
    else:
        wspd = spec.wspd
    if spec.qsfc is None:
        qsfc = _qsat(ts, spec.ps) if spec.xland > 1.5 else out["qv"][0]
    else:
        qsfc = spec.qsfc
    scalars = dict(xland=spec.xland, ts=ts, qsfc=qsfc, ps=spec.ps,
                   ust=spec.ust, ch=spec.ch, hfx=spec.hfx, qfx=spec.qfx,
                   wspd=wspd, znt=spec.znt)
    return {"profiles": {k: np.asarray(v, np.float32) for k, v in out.items()},
            "scalars": {k: np.float32(v) for k, v in scalars.items()}}


def catalogue() -> list[Spec]:
    """The graded columns.  Deterministic; jitter comes from one seed."""

    rng = np.random.default_rng(461)
    j = lambda lo, hi: float(rng.uniform(lo, hi))  # noqa: E731
    cols: list[Spec] = []
    add = cols.append

    # --- convective land ----------------------------------------------------
    for n, (hfx, zi, qv0, grp) in enumerate([
            (320, 1600, 0.012, "A"), (180, 900, 0.016, "A"),
            (450, 2600, 0.006, "A"), (550, 3200, 0.004, "C"),
            (250, 1300, 0.014, "B"), (120, 700, 0.018, "B"),
            (380, 2000, 0.010, "D"), (290, 1500, 0.011, "E"),
            (220, 1200, 0.013, "F"), (400, 1800, 0.009, "G")]):
        add(Spec(f"convective_land_{n}", "convective", grp,
                 th0=j(298, 310), zi=zi, gml=-0.0002, dthinv=j(1, 4),
                 qv0=qv0, dqinv=j(0.2, 0.6), u0=j(1, 6), du=j(0, 0.003),
                 dts=j(4, 12), hfx=hfx, qfx=hfx * j(1.5e-7, 6e-7),
                 ust=j(0.25, 0.6), znt=j(0.03, 0.5), ch=j(0.008, 0.02),
                 w=(0.0, j(0.02, 0.2), zi * 0.5, zi)))

    # --- stable night over land --------------------------------------------
    for n, (hfx, ust, gml, grp) in enumerate([
            (-38, 0.12, 0.016, "A"), (-8, 0.05, 0.03, "A"),
            (-75, 0.30, 0.008, "A"), (-20, 0.08, 0.02, "C"),
            (-55, 0.22, 0.012, "B"), (-3, 0.03, 0.04, "E"),
            (-30, 0.15, 0.01, "D"), (-12, 0.07, 0.025, "F"),
            (-45, 0.18, 0.015, "G"), (-90, 0.40, 0.006, "A")]):
        th0 = j(280, 295)
        add(Spec(f"stable_night_{n}", "stable", grp,
                 th0=th0, zi=j(150, 450), gml=gml, dthinv=j(0.5, 3),
                 qv0=j(0.004, 0.013), dqinv=j(0.05, 0.3),
                 u0=j(0.5, 3), du=0.0, jet=(j(150, 400), j(6, 14), 150.0),
                 dts=-j(1, 6), hfx=hfx, qfx=(-1.0e-5 if n % 2 == 0 else 1e-6),
                 ust=ust, znt=j(0.05, 0.8), ch=j(0.002, 0.006)))

    # --- ocean --------------------------------------------------------------
    for n in range(4):  # marine stratocumulus
        zi = j(600, 1200)
        add(Spec(f"marine_sc_{n}", "ocean", "AACD"[n], xland=2.0,
                 th0=j(286, 292), zi=zi, dthinv=j(6, 11), qv0=j(0.009, 0.011),
                 gqml=-1.0e-7, dqinv=j(0.4, 0.6),
                 clouds=[(zi * j(0.55, 0.75), zi, j(2e-4, 6e-4), "liq")],
                 rad=[(zi - 120.0, zi, -1.5e-4)], u0=j(5, 9), du=0.001,
                 w=(-0.004, 0.0, 0.0, 1.0), dts=j(0, 1.5), hfx=j(5, 20),
                 qfx=j(2e-5, 4e-5), ust=j(0.18, 0.3), znt=2e-4,
                 ch=j(0.009, 0.012)))
    for n in range(3):  # trade cumulus
        add(Spec(f"trade_cumulus_{n}", "ocean", "ABC"[n], xland=2.0,
                 th0=j(298, 301), zi=j(500, 700), dthinv=0.8, gft=0.0045,
                 qv0=j(0.016, 0.019), dqinv=j(0.1, 0.2), ztp=15500.0,
                 clouds=[(800.0, 1800.0, j(2e-5, 1e-4), "liq", 1.0)],
                 u0=-j(6, 10), du=0.0005, dts=j(0.5, 1.5), hfx=j(8, 25),
                 qfx=j(6e-5, 1.2e-4), ust=j(0.2, 0.35), znt=2e-4, ch=0.012))
    for n in range(2):  # cold-air outbreak over warm water
        add(Spec(f"cold_air_outbreak_{n}", "ocean", "AC"[n], xland=2.0,
                 th0=j(266, 272), zi=j(800, 1500), dthinv=4.0,
                 qv0=j(0.0015, 0.003), dqinv=0.5, gml=-0.0004,
                 clouds=[(500.0, 1300.0, j(1e-4, 3e-4), "auto")],
                 snow=[(200.0, 1400.0, j(1e-5, 8e-5))],
                 u0=j(10, 16), du=0.002, ts=j(276, 281), hfx=j(250, 420),
                 qfx=j(8e-5, 1.5e-4), ust=j(0.5, 0.8), znt=1.5e-3, ch=0.018))
    for n in range(2):  # warm advection fog over cold sea
        th0 = j(284, 288)
        add(Spec(f"sea_fog_{n}", "ocean", "AB"[n], xland=2.0,
                 th0=th0, zi=j(120, 250), gml=0.01, dthinv=j(3, 6),
                 qv0=0.0095, gqml=0.0, dqinv=0.3, rhmax=0.95,
                 clouds=[(0.0, 180.0, j(5e-5, 2e-4), "liq")],
                 u0=j(5, 8), du=0.003, dts=-j(2, 4), hfx=-j(10, 35),
                 qfx=-j(1e-6, 5e-6), ust=j(0.1, 0.2), znt=1e-4, ch=0.004))
    add(Spec("tropical_ocean", "ocean", "A", xland=2.0, th0=300.5, zi=550.0,
             dthinv=0.5, gft=0.003, ztp=16000.0, qv0=0.0205, hq=3500.0,
             dqinv=0.1, clouds=[(700.0, 4500.0, 2e-4, "auto", 1.0)],
             u0=4.0, du=0.0, dts=1.2, hfx=15.0, qfx=1.4e-4, ust=0.22,
             znt=2e-4, ch=0.012))

    # --- high terrain -------------------------------------------------------
    for n, (ps, hfx, th0, regime_flags) in enumerate([
            (60000.0, 520.0, 318.0, "day"), (62000.0, -45.0, 300.0, "night"),
            (70000.0, 380.0, 312.0, "day"), (75000.0, -25.0, 296.0, "night"),
            (58000.0, 60.0, 290.0, "snowfield"), (78000.0, 250.0, 308.0, "day"),
            (66000.0, 140.0, 304.0, "valley"), (72000.0, -60.0, 294.0, "night")]):
        day = regime_flags in ("day", "valley")
        add(Spec(f"high_terrain_{regime_flags}_{n}", "high_terrain",
                 "ABCAACBG"[n], ps=ps, grid=(35.0, 1.10, 850.0),
                 th0=th0, zi=(j(1500, 3500) if day else j(100, 400)),
                 gml=(0.0 if day else 0.015), dthinv=j(1, 4), ztp=7000.0,
                 qv0=j(0.002, 0.007), dqinv=0.4, hq=2000.0,
                 clouds=([(2500.0, 3200.0, 5e-5, "auto")]
                         if regime_flags == "valley" else []),
                 snow=([(0.0, 600.0, 2e-5)]
                       if regime_flags == "snowfield" else []),
                 u0=j(3, 12), du=j(0.001, 0.004),
                 dts=(j(10, 20) if day else -j(3, 8)), hfx=hfx,
                 qfx=(j(2e-5, 8e-5) if hfx > 0 else -2e-6),
                 ust=j(0.15, 0.6), znt=j(0.05, 1.0), ch=j(0.004, 0.02)))

    # --- snow and ice -------------------------------------------------------
    add(Spec("arctic_inversion", "snow_ice", "A", th0=245.0, zi=60.0,
             gml=0.05, dthinv=8.0, gft=0.006, ztp=8500.0, qv0=3e-4,
             dqinv=0.0, rhmax=0.9, u0=2.0, du=0.004, dts=-2.0, hfx=-15.0,
             qfx=-5e-7, ust=0.06, znt=1e-3, ch=0.003))
    add(Spec("ice_fog", "snow_ice", "A", th0=238.0, zi=80.0, gml=0.04,
             dthinv=6.0, ztp=8000.0, qv0=2e-4, clouds=[(0.0, 120.0, 3e-5,
                                                         "ice")],
             u0=1.0, du=0.002, dts=-1.5, hfx=-8.0, qfx=-2e-7, ust=0.04,
             znt=1e-3, ch=0.002))
    for n in range(2):
        add(Spec(f"arctic_mixed_phase_sc_{n}", "snow_ice", "AC"[n],
                 xland=float(1 + n), th0=j(258, 263), zi=j(600, 900),
                 dthinv=j(4, 7), qv0=j(1.5e-3, 2.2e-3), ztp=8500.0,
                 clouds=[(350.0, 800.0, j(1e-4, 2e-4), "mix")],
                 snow=[(0.0, 800.0, j(2e-5, 6e-5))],
                 rad=[(650.0, 800.0, -8e-5)], u0=j(3, 7), dts=-1.0,
                 hfx=j(-5, 15), qfx=j(0, 5e-6), ust=j(0.1, 0.25),
                 znt=1e-3, ch=0.006))
    add(Spec("deep_snowfall", "snow_ice", "A", th0=271.0, zi=900.0,
             gml=0.002, dthinv=1.0, gft=0.0035, qv0=3.5e-3, ztp=9000.0,
             clouds=[(300.0, 5000.0, 1.5e-4, "auto", 1.0)],
             snow=[(0.0, 6000.0, 4e-4)], u0=6.0, du=0.004, dts=-0.5,
             hfx=-5.0, qfx=1e-6, ust=0.3, znt=0.05, ch=0.008))
    add(Spec("polar_night", "snow_ice", "D", th0=228.0, zi=40.0,
             gml=0.08, dthinv=10.0, ztp=7500.0, qv0=5e-5, rhmax=0.85,
             u0=1.5, du=0.003, dts=-3.0, hfx=-20.0, qfx=-1e-7, ust=0.05,
             znt=5e-4, ch=0.002))
    add(Spec("freezing_drizzle", "snow_ice", "B", th0=268.0, zi=1100.0,
             dthinv=3.0, qv0=2.8e-3, ztp=9000.0,
             clouds=[(300.0, 1100.0, 3e-4, "liq")], u0=7.0, du=0.002,
             dts=-0.3, hfx=-3.0, qfx=2e-6, ust=0.3, znt=0.1, ch=0.008))
    add(Spec("lake_effect_snow", "snow_ice", "A", xland=2.0, th0=263.0,
             zi=1800.0, gml=-0.0003, dthinv=2.0, qv0=1.8e-3, ztp=8500.0,
             clouds=[(700.0, 1800.0, 2.5e-4, "auto")],
             snow=[(0.0, 2000.0, 2e-4)], u0=12.0, du=0.002, ts=276.0,
             hfx=320.0, qfx=1.1e-4, ust=0.55, znt=1e-3, ch=0.016))
    add(Spec("blowing_snow_wind", "snow_ice", "C", th0=255.0, zi=700.0,
             dthinv=4.0, qv0=8e-4, ztp=8500.0, snow=[(0.0, 300.0, 1e-4)],
             u0=18.0, du=0.004, dts=-1.0, hfx=-10.0, qfx=1e-6, ust=0.9,
             znt=5e-3, ch=0.01))
    add(Spec("snow_cold_front", "snow_ice", "E", th0=266.0, zi=500.0,
             dthinv=2.0, qv0=2.2e-3, ztp=9000.0,
             clouds=[(200.0, 3000.0, 1e-4, "auto", 1.0)],
             snow=[(0.0, 3000.0, 1.2e-4)], u0=9.0, du=0.003, dts=-0.5,
             hfx=-6.0, qfx=0.0, ust=0.4, znt=0.08, ch=0.009))
    add(Spec("cold_pool_fog_ice", "snow_ice", "F", th0=268.5, zi=400.0,
             gml=0.006, dthinv=5.0, qv0=2.9e-3, ztp=9000.0,
             clouds=[(0.0, 300.0, 1e-4, "auto")],
             snow=[(1500.0, 3500.0, 3e-5)], u0=1.5, du=0.004, dts=-2.5,
             hfx=-12.0, qfx=-3e-6, ust=0.08, znt=0.05, ch=0.003))
    add(Spec("snow_no_flagqs", "snow_ice", "G", th0=267.0, zi=800.0,
             dthinv=2.0, qv0=2.5e-3, ztp=9000.0,
             clouds=[(300.0, 2500.0, 1e-4, "auto", 1.0)],
             snow=[(0.0, 2500.0, 2e-4)], u0=6.0, du=0.003, dts=-0.5,
             hfx=-4.0, qfx=1e-6, ust=0.3, znt=0.1, ch=0.008))

    # --- moist convection and cloud ----------------------------------------
    for n in range(3):
        zi = j(900, 1300)
        add(Spec(f"shallow_cumulus_{n}", "moist", "ABE"[n], th0=j(299, 302),
                 zi=zi, dthinv=1.0, gft=0.0035, qv0=j(0.013, 0.016),
                 dqinv=0.15, clouds=[(zi + 100, zi + 1100, j(2e-5, 6e-5),
                                      "liq", 1.0)],
                 u0=j(3, 6), du=0.001, w=(0.05, 0.0, 0.0, 1.0),
                 dts=j(3, 6), hfx=j(150, 230), qfx=j(1.5e-4, 2.5e-4),
                 ust=j(0.3, 0.45), znt=0.15, ch=0.011))
    add(Spec("congestus", "moist", "A", th0=301.0, zi=1000.0, dthinv=0.5,
             gft=0.003, ztp=14000.0, qv0=0.017, dqinv=0.05, hq=3000.0,
             clouds=[(1100.0, 5000.0, 6e-4, "auto", 1.0)],
             snow=[(4000.0, 6000.0, 1e-4)], u0=3.0, du=0.002,
             w=(0.0, 2.5, 3000.0, 1500.0), dts=4.0, hfx=200.0, qfx=2.4e-4,
             ust=0.35, znt=0.2, ch=0.012))
    add(Spec("saturated_stratiform", "moist", "C", th0=288.0, zi=600.0,
             dthinv=0.5, qv0=0.0095, clouds=[(0.0, 4000.0, 3e-4, "auto",
                                               1.0)],
             snow=[(2500.0, 5000.0, 1e-4)], u0=8.0, du=0.003, dts=0.0,
             hfx=0.5, qfx=1e-6, ust=0.35, znt=0.1, ch=0.01))
    add(Spec("tropical_land_moist", "moist", "B", th0=303.0, zi=1400.0,
             dthinv=0.6, gft=0.003, ztp=16000.0, qv0=0.021, dqinv=0.1,
             hq=3500.0, u0=2.0, du=0.001, dts=5.0, hfx=260.0, qfx=3e-4,
             ust=0.4, znt=0.5, ch=0.014))
    add(Spec("evaporating_cloud", "moist", "D", th0=293.0, zi=1200.0,
             dthinv=2.0, qv0=0.008, clouds=[(800.0, 1600.0, 2e-4, "liq",
                                             0.85)],
             u0=6.0, du=0.002, dts=1.0, hfx=60.0, qfx=4e-5, ust=0.3,
             znt=0.1, ch=0.01))
    add(Spec("elevated_mixed_layer", "moist", "A", th0=300.0, zi=600.0,
             dthinv=4.0, gft=0.001, ztp=12000.0, qv0=0.016, dqinv=0.7,
             u0=8.0, du=0.004, jet=(800.0, 12.0, 300.0), dts=3.0,
             hfx=120.0, qfx=1.2e-4, ust=0.35, znt=0.1, ch=0.012))

    # --- high wind and shear ------------------------------------------------
    for n in range(2):
        add(Spec(f"high_wind_neutral_{n}", "high_wind", "AD"[n],
                 grid=(30.0, 1.10, 900.0), th0=j(292, 297), zi=j(1000, 1400),
                 dthinv=2.0, qv0=j(0.007, 0.01), u0=j(15, 22), du=0.006,
                 dts=0.5, hfx=j(15, 40), qfx=6e-5, ust=j(0.85, 1.2),
                 znt=j(0.3, 0.6), ch=0.02))
    add(Spec("hurricane_boundary_layer", "high_wind", "B", xland=2.0,
             th0=299.0, zi=900.0, dthinv=0.5, qv0=0.019, ztp=16000.0,
             clouds=[(500.0, 6000.0, 5e-4, "auto", 1.0)], u0=40.0,
             du=-0.002, v0=-15.0, dv=0.0, dts=1.5, hfx=120.0, qfx=3e-4,
             ust=2.2, znt=3e-3, ch=0.02, wspd=45.0))
    add(Spec("upper_jet_shear", "high_wind", "C", th0=294.0, zi=1100.0,
             dthinv=2.0, qv0=0.008, u0=6.0, du=0.002,
             jet=(9500.0, 45.0, 1500.0), dts=2.0, hfx=90.0, qfx=5e-5,
             ust=0.4, znt=0.1, ch=0.012))
    add(Spec("overcast_windy", "high_wind", "A", th0=285.0, zi=1200.0,
             dthinv=1.5, qv0=0.0085, clouds=[(400.0, 1200.0, 2e-4, "liq")],
             u0=13.0, du=0.004, dts=0.0, hfx=2.0, qfx=1e-5, ust=0.7,
             znt=0.2, ch=0.015))
    add(Spec("low_level_jet_night", "high_wind", "B", th0=294.0, zi=250.0,
             gml=0.012, dthinv=1.0, qv0=0.013, u0=2.0, du=0.0,
             jet=(350.0, 18.0, 180.0), dts=-3.0, hfx=-25.0, qfx=-2e-6,
             ust=0.2, znt=0.1, ch=0.005))

    # --- edge cases ---------------------------------------------------------
    add(Spec("edge_zero_flux", "edge", "A", th0=290.0, zi=800.0, dthinv=2.0,
             qv0=0.008, u0=5.0, dts=0.0, hfx=0.0, qfx=0.0, ust=0.25,
             znt=0.1, ch=0.01))
    add(Spec("edge_calm_floor_ust", "edge", "A", th0=288.0, zi=200.0,
             gml=0.02, dthinv=1.0, qv0=0.007, u0=0.05, du=0.0, v0=0.0,
             dv=0.0, dts=-2.0, hfx=-1.0, qfx=0.0, ust=0.001, znt=0.01,
             ch=0.001, wspd=0.1))
    add(Spec("edge_extreme_heating", "edge", "C", th0=318.0, zi=4500.0,
             gml=-0.0005, dthinv=1.0, qv0=0.003, dqinv=0.3, ztp=12000.0,
             u0=4.0, du=0.001, dts=25.0, hfx=800.0, qfx=1e-5, ust=0.7,
             znt=0.05, ch=0.02))
    add(Spec("edge_extreme_cooling", "edge", "A", th0=285.0, zi=150.0,
             gml=0.04, dthinv=2.0, qv0=0.006, u0=10.0, du=0.004, dts=-8.0,
             hfx=-250.0, qfx=-3e-5, ust=0.5, znt=0.3, ch=0.006))
    add(Spec("edge_superadiabatic", "edge", "B", th0=312.0, zi=1500.0,
             gml=-0.01, dthinv=1.0, qv0=0.01, u0=3.0, dts=15.0, hfx=420.0,
             qfx=8e-5, ust=0.4, znt=0.1, ch=0.015))
    add(Spec("edge_supersaturated_clear", "edge", "A", th0=285.0,
             zi=900.0, dthinv=2.0, qv0=0.0125, rhmax=1.03, dqinv=0.1,
             u0=5.0, dts=0.5, hfx=20.0, qfx=3e-5, ust=0.25, znt=0.1,
             ch=0.01))
    add(Spec("edge_bone_dry", "edge", "C", th0=305.0, zi=2500.0, dthinv=1.0,
             qv0=2e-6, gqml=0.0, dqinv=0.0, u0=6.0, dts=12.0, hfx=350.0,
             qfx=0.0, ust=0.45, znt=0.05, ch=0.015))
    add(Spec("edge_inversion_in_first_layer", "edge", "A", th0=280.0,
             zi=10.0, gml=0.0, dthinv=12.0, qv0=0.005, u0=1.5, du=0.004,
             dts=-6.0, hfx=-30.0, qfx=-1e-6, ust=0.05, znt=0.02,
             ch=0.002))
    add(Spec("edge_strong_cloudtop_cooling", "edge", "A", xland=2.0,
             th0=288.0, zi=900.0, dthinv=10.0, qv0=0.0098, gqml=-1e-7,
             dqinv=0.5, clouds=[(500.0, 900.0, 8e-4, "liq")],
             rad=[(700.0, 900.0, -1e-3)], u0=6.0, dts=0.5, hfx=10.0,
             qfx=3e-5, ust=0.25, znt=2e-4, ch=0.01))
    add(Spec("edge_large_vertical_velocity", "edge", "B", th0=302.0,
             zi=1500.0, dthinv=1.0, qv0=0.016, gft=0.003,
             clouds=[(1600.0, 7000.0, 1e-3, "auto", 1.0)], u0=5.0,
             w=(0.0, -3.0, 600.0, 300.0), dts=5.0, hfx=150.0, qfx=1.5e-4,
             ust=0.4, znt=0.2, ch=0.012))
    add(Spec("edge_thin_lowest_layer", "edge", "F", grid=(5.0, 1.18, 800.0),
             th0=296.0, zi=600.0, dthinv=2.0, qv0=0.011, u0=4.0, dts=6.0,
             hfx=200.0, qfx=1e-4, ust=0.35, znt=0.05, ch=0.014))
    add(Spec("edge_thick_lowest_layer", "edge", "D",
             grid=(200.0, 1.04, 900.0), th0=293.0, zi=1200.0, dthinv=2.0,
             qv0=0.009, u0=7.0, dts=3.0, hfx=150.0, qfx=6e-5, ust=0.4,
             znt=0.2, ch=0.012))
    add(Spec("edge_heavy_condensate", "edge", "A", th0=296.0, zi=800.0,
             dthinv=0.5, qv0=0.015, clouds=[(0.0, 3000.0, 3e-3, "auto",
                                              1.0)],
             snow=[(2000.0, 5000.0, 1e-3)], u0=5.0, dts=0.0, hfx=5.0,
             qfx=1e-5, ust=0.3, znt=0.1, ch=0.01))
    add(Spec("edge_tiny_znt_ocean_calm", "edge", "E", xland=2.0, th0=297.0,
             zi=400.0, dthinv=1.0, qv0=0.016, u0=0.5, du=0.0, v0=0.2,
             dv=0.0, dts=1.0, hfx=4.0, qfx=2e-5, ust=0.01, znt=1e-5,
             ch=0.001, wspd=0.6))
    return cols


def build_inputs() -> dict:
    """All columns, ordered by group, as float32 arrays + metadata."""

    specs = catalogue()
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("column names must be unique")
    specs = sorted(specs, key=lambda spec: GROUP_ORDER.index(spec.group))
    built = [build_column(spec) for spec in specs]
    ncol = len(specs)
    profiles = {name: np.stack([b["profiles"][name] for b in built], 1)
                .astype(np.float32) for name in PROFILE_INPUTS}  # (nz, ncol)
    scalars = {name: np.array([b["scalars"][name] for b in built],
                              np.float32) for name in SCALAR_INPUTS}
    groups = []
    for key in GROUP_ORDER:
        members = [i for i, spec in enumerate(specs) if spec.group == key]
        if members:
            assert members == list(range(members[0], members[-1] + 1))
            dx, delt, flag_qs = GROUPS[key]
            groups.append((key, members[0], len(members), dx, delt, flag_qs))
    return {"ncol": ncol, "names": [s.name for s in specs],
            "regimes": [s.regime for s in specs], "groups": groups,
            "profiles": profiles, "scalars": scalars}


def write_input(path: Path, inputs: dict, nstep: int) -> None:
    ncol = inputs["ncol"]
    with open(path, "wb") as stream:
        np.array([MAGIC, VERSION, ncol, NZ, nstep, len(inputs["groups"])],
                 "<i4").tofile(stream)
        for _, start, count, dx, delt, flag_qs in inputs["groups"]:
            np.array([start, count, int(flag_qs)], "<i4").tofile(stream)
            np.array([dx, delt], "<f4").tofile(stream)
        for name in SCALAR_INPUTS:
            inputs["scalars"][name].astype("<f4").tofile(stream)
        for name in PROFILE_INPUTS:
            # Fortran P(nz, ncol): level fastest.
            np.asfortranarray(inputs["profiles"][name]).astype("<f4") \
                .T.tofile(stream)


def read_output(path: Path, ncol: int, nstep: int) -> dict:
    """The harness's records: ``{name: array (nstep, nz, ncol) or (nstep, ncol)}``."""

    raw = np.fromfile(path, "<u4")
    header = raw[:6].view("<i4")
    assert tuple(header[:2]) == (MAGIC, VERSION), header
    assert int(header[2]) == ncol and int(header[3]) == NZ
    assert int(header[4]) == nstep
    pos = 6
    records: dict[str, list] = {}

    def take(count, dtype):
        nonlocal pos
        piece = raw[pos:pos + count].view(dtype)
        pos += count
        return piece

    layout = ([("in_" + n, "p", "<f4") for n in STEP_PROFILE_IN]
              + [("in_" + n, "p", "<f4") for n in STATE_PROFILE]
              + [("in_" + n, "c", "<f4") for n in STEP_COLUMN_IN]
              + [("in_" + n, "c", "<i4") for n in STEP_COLUMN_IN_INT]
              + [("out_" + n, "p", "<f4") for n in PROFILE_OUT]
              + [("out_" + n, "c", "<f4") for n in COLUMN_OUT]
              + [("out_" + n, "c", "<i4") for n in COLUMN_OUT_INT])
    for _ in range(nstep):
        step_marker = int(take(1, "<i4")[0])
        assert step_marker == 0x53544550, hex(step_marker)
        for name, kind, dtype in layout:
            if kind == "p":
                value = take(NZ * ncol, dtype).reshape(ncol, NZ).T
            else:
                value = take(ncol, dtype)
            records.setdefault(name, []).append(np.array(value))
    assert pos == raw.size, (pos, raw.size)
    return {name: np.stack(values) for name, values in records.items()}


def _delt_by_column(groups, ncol: int) -> np.ndarray:
    delt = np.zeros(ncol, np.float32)
    for _, start, count, _, step_delt, _ in groups:
        delt[start:start + count] = np.float32(step_delt)
    return delt[None, :]


def replay_inputs(fixture: dict, step0: dict) -> dict:
    """Every step's incoming state from step 1's and WRF's own outputs.

    The harness carries the state exactly this way (the wrapper's outputs
    straight back; the atmosphere advanced by :func:`_advance`), so the
    recorded inputs of steps 2..N are redundant with the outputs and are
    not stored.  :func:`pack` proves the reconstruction is bitwise the
    recording before it drops them.
    """

    nstep = fixture["meta"]["nstep"]
    delt = _delt_by_column(fixture["groups"], len(fixture["names"]))
    held = {k: fixture[f"prof_{k}"] for k in ("p", "exner")}
    prof = {k: step0[f"in_{k}"][0] for k in STEP_PROFILE_IN}
    full = {name: [value[0]] for name, value in step0.items()}
    for step in range(1, nstep):
        out = {k: fixture[f"out_{k}"][step - 1]
               for k in (*PROFILE_OUT, *COLUMN_OUT, *COLUMN_OUT_INT)}
        prof = _advance({**prof, **held}, out, delt)
        for k in STEP_PROFILE_IN:
            full[f"in_{k}"].append(prof[k])
        for k in (*STATE_PROFILE, *STEP_COLUMN_IN, *STEP_COLUMN_IN_INT):
            full[f"in_{k}"].append(out[k])
    return {name: np.stack(values) for name, values in full.items()}


_IN_NAMES = tuple(f"in_{k}" for k in (*STEP_PROFILE_IN, *STATE_PROFILE,
                                      *STEP_COLUMN_IN, *STEP_COLUMN_IN_INT))


def pack(inputs: dict, records: dict, mixlength: int, nstep: int,
         path: Path, provenance: dict, variant: str = "base") -> None:
    """Write the fixture: inputs, step-1 state, every output (LZMA zip)."""

    import io
    import zipfile

    arrays = {f"prof_{k}": v for k, v in inputs["profiles"].items()}
    arrays.update({f"col_{k}": v for k, v in inputs["scalars"].items()})
    arrays.update({k: v for k, v in records.items() if k.startswith("out_")})
    arrays["names"] = np.array(inputs["names"])
    arrays["regimes"] = np.array(inputs["regimes"])
    arrays["groups"] = np.array(json.dumps(inputs["groups"]))
    arrays["meta"] = np.array(json.dumps(
        {"mixlength": mixlength, "nstep": nstep, "nz": NZ,
         "variant": variant, **provenance}))
    step0 = {k: records[k][:1] for k in _IN_NAMES}
    check = _decode(dict(arrays))
    rebuilt = replay_inputs(check, step0)
    for k in _IN_NAMES:
        if not np.array_equal(rebuilt[k].view(np.uint32),
                              records[k].view(np.uint32)):
            raise AssertionError(f"replay reconstruction differs on {k}")
    arrays.update(step0)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_LZMA) as archive:
        for name, value in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            archive.writestr(name + ".npy", buffer.getvalue())


def _decode(fixture: dict) -> dict:
    fixture["groups"] = [tuple(g) for g in json.loads(str(fixture["groups"]))]
    fixture["meta"] = json.loads(str(fixture["meta"]))
    fixture["names"] = [str(n) for n in fixture["names"]]
    fixture["regimes"] = [str(r) for r in fixture["regimes"]]
    return fixture


def load_fixture(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        fixture = _decode({name: data[name] for name in data.files})
    fixture.update(replay_inputs(
        fixture, {k: fixture[k] for k in _IN_NAMES}))
    return fixture


# ---------------------------------------------------------------------------
# WOOF side
# ---------------------------------------------------------------------------

def _advance(state: dict, out: dict, delt: np.float32) -> dict:
    """The harness's between-step update, fp32 operation for operation."""

    nxt = {k: np.array(v, copy=True) for k, v in state.items()}
    zero = np.float32(0.0)
    for fld, tend, floor in (("u", "rublten", False), ("v", "rvblten", False),
                             ("th", "rthblten", False),
                             ("qv", "rqvblten", True),
                             ("qc", "rqcblten", True),
                             ("qi", "rqiblten", True),
                             *((q, t, True) for q, t, _, _ in QN_TENDENCIES)):
        updated = (nxt[fld] + delt * out[tend]).astype(np.float32)
        nxt[fld] = np.maximum(updated, zero) if floor else updated
    nxt["t3d"] = (nxt["th"] * nxt["exner"]).astype(np.float32)
    nxt["rho"] = (nxt["p"] / (np.float32(287.0) * nxt["t3d"])
                  ).astype(np.float32)
    return nxt


def run_woof(fixture: dict, *, replay: bool, mp_physics_qs=(8, 1),
             column_chunk: int | None = None) -> dict:
    """Run ``mynn_pbl_step`` over every group and step.

    Returns ``{"out_<name>": (nstep, nz, ncol) | (nstep, ncol)}`` in the
    harness's record layout.  ``mp_physics_qs``: the microphysics ids used
    for FLAG_QS true / false groups (Thompson 8 has snow, Kessler 1 none).
    """

    import cupy as cp
    from gpuwm.core.mynn_pbl_runtime import mynn_flag_qs, mynn_pbl_step

    meta = fixture["meta"]
    nstep, nz = meta["nstep"], meta["nz"]
    mixlength = meta["mixlength"]
    variant = meta.get("variant", "base")
    if variant == "mixscalars":
        mp_physics_qs = (28, mp_physics_qs[1])
    ncol = len(fixture["names"])
    mp_on, mp_off = mp_physics_qs
    assert mynn_flag_qs(mp_on) and not mynn_flag_qs(mp_off)
    f32 = np.float32
    held = {k: fixture[f"prof_{k}"] for k in ("dz", "w", "p", "exner",
                                                "spp")}
    results: dict[str, list] = {}
    carried = None   # WOOF's own state (free mode)
    for step in range(nstep):
        recorded = {k: fixture[f"in_{k}"][step] for k in STEP_PROFILE_IN}
        recorded_state = {k: fixture[f"in_{k}"][step] for k in STATE_PROFILE}
        recorded_cols = {k: fixture[f"in_{k}"][step]
                         for k in (*STEP_COLUMN_IN, *STEP_COLUMN_IN_INT)}
        if replay or carried is None:
            prof = {**recorded, **held}
            state = dict(recorded_state)
            cols = dict(recorded_cols)
        else:
            prof, state, cols = carried
        step_out = {f"out_{k}": np.zeros((nz, ncol), f32)
                    for k in PROFILE_OUT}
        step_out.update({f"out_{k}": np.zeros(ncol, f32) for k in COLUMN_OUT})
        step_out.update({f"out_{k}": np.zeros(ncol, np.int32)
                         for k in COLUMN_OUT_INT})
        for _, start, count, dx, delt, flag_qs in fixture["groups"]:
            sl = slice(start, start + count)

            def dev3(a):
                return cp.asarray(np.ascontiguousarray(
                    a[:, sl].reshape(nz, 1, count), dtype=f32))

            def dev2(a, dtype=f32):
                return cp.asarray(np.ascontiguousarray(
                    a[sl].reshape(1, count), dtype=dtype))

            atmosphere = {
                "dz": dev3(prof["dz"]), "u": dev3(prof["u"]),
                "v": dev3(prof["v"]), "theta": dev3(prof["th"]),
                "pressure": dev3(prof["p"]), "exner": dev3(prof["exner"]),
                "rho": dev3(prof["rho"]), "temperature": dev3(prof["t3d"]),
                "qv": dev3(prof["qv"]), "qc": dev3(prof["qc"]),
                "qi": dev3(prof["qi"]), "qs": dev3(prof["qs"]),
            }
            fields = {k: dev3(state[k]) for k in STATE_PROFILE}
            fields.update({
                "xland": dev2(fixture["col_xland"]),
                "tsk": dev2(fixture["col_ts"]),
                "psfc": dev2(fixture["col_ps"]),
                "ust": dev2(cols["ust"]), "hfx": dev2(cols["hfx"]),
                "qfx": dev2(cols["qfx"]), "wspd": dev2(cols["wspd"]),
                "pblh": dev2(cols["pblh"]), "rmol": dev2(cols["rmol"]),
                "kpbl": dev2(cols["kpbl"], np.int32),
                "maxwidth": cp.zeros((1, count), f32),
                "maxmf": cp.zeros((1, count), f32),
                "ztop_plume": cp.zeros((1, count), f32),
                "ktop_plume": cp.zeros((1, count), np.int32),
            })
            w = dev3(prof["w"])
            extra = {}
            if variant == "mixscalars":
                extra["qn_scalars"] = {state: dev3(prof[q]) for q, _, state, _
                                       in QN_TENDENCIES}
            if variant == "spp":
                extra.update(spp_pbl=1, pattern_spp_pbl=dev3(prof["spp"]))
            tend = mynn_pbl_step(
                atmosphere, fields, w=w, dx=float(dx), delt=float(delt),
                itimestep=step + 1,
                mp_physics=(mp_on if flag_qs else mp_off),
                column_chunk=column_chunk, **extra,
                closure=2.6, bl_mynn_cloudpdf=2, bl_mynn_mixlength=mixlength,
                bl_mynn_edmf=1, bl_mynn_edmf_mom=1, bl_mynn_edmf_tke=0,
                bl_mynn_mixscalars=int(variant == "mixscalars"),
                bl_mynn_cloudmix=1, bl_mynn_mixqt=0,
                bl_mynn_output=0, bl_mynn_tkeadvect=False, icloud_bl=1)
            host = lambda a: cp.asnumpy(a).reshape(-1, count)  # noqa: E731
            for name, key in (("rublten", "du"), ("rvblten", "dv"),
                              ("rthblten", "dtheta"), ("rqvblten", "dqv"),
                              ("rqcblten", "dqc"), ("rqiblten", "dqi")):
                step_out[f"out_{name}"][:, sl] = host(tend[key])
            if variant == "mixscalars":
                for _, name, _, key in QN_TENDENCIES:
                    step_out[f"out_{name}"][:, sl] = host(tend[key])
            for name in STATE_PROFILE:
                step_out[f"out_{name}"][:, sl] = host(fields[name])
            for name in ("pblh", "rmol", "maxwidth", "maxmf", "ztop_plume",
                         "ust", "hfx", "qfx", "wspd"):
                step_out[f"out_{name}"][sl] = host(fields[name])[0]
            # WOOF holds no znt in the PBL call; WRF's wrapper passes it
            # through untouched (graded as "unchanged" below).
            step_out["out_znt"][sl] = cols["znt"][sl]
            for name in COLUMN_OUT_INT:
                step_out[f"out_{name}"][sl] = host(fields[name])[0]
        for name, value in step_out.items():
            results.setdefault(name, []).append(value)
        # Free-running carry: WOOF's own outputs drive the next step.
        delt_by_col = np.zeros(ncol, f32)
        for _, start, count, _, delt, _ in fixture["groups"]:
            delt_by_col[start:start + count] = f32(delt)
        nxt = _advance({**prof}, {k.removeprefix("out_"): v
                                  for k, v in step_out.items()},
                       delt_by_col[None, :])
        carried = (nxt,
                   {k: step_out[f"out_{k}"] for k in STATE_PROFILE},
                   {"pblh": step_out["out_pblh"], "rmol": step_out["out_rmol"],
                    "ust": cols["ust"], "hfx": cols["hfx"],
                    "qfx": cols["qfx"], "wspd": cols["wspd"],
                    "znt": cols["znt"], "kpbl": step_out["out_kpbl"]})
    return {name: np.stack(values) for name, values in results.items()}


def grade(fixture: dict, actual: dict) -> dict:
    """Per output: max ULP over all, per regime, per step; bitwise counts."""

    from gpuwm.core.fp32_ulp import fp32_ulp_distance

    regimes = np.array(fixture["regimes"])
    report = {}
    for name in (*PROFILE_OUT, *COLUMN_OUT, *COLUMN_OUT_INT):
        want = fixture[f"out_{name}"]
        got = actual[f"out_{name}"]
        if name in COLUMN_OUT_INT:
            dist = np.abs(got.astype(np.int64) - want.astype(np.int64))
        else:
            dist = np.asarray(fp32_ulp_distance(got.astype(np.float32),
                                                want.astype(np.float32)),
                              np.int64)
        col_axis = dist.reshape(dist.shape[0], -1, dist.shape[-1]).max(1)
        words_differ = (int(np.count_nonzero(
            got.view(np.uint32) != want.view(np.uint32)))
            if name not in COLUMN_OUT_INT else int(np.count_nonzero(dist)))
        absdiff = (np.abs(got.astype(np.float64) - want.astype(np.float64))
                   if name not in COLUMN_OUT_INT else dist)
        report[name] = {
            "max_ulp": int(dist.max()),
            "max_abs": float(np.nanmax(absdiff)) if absdiff.size else 0.0,
            "words_differ": words_differ,
            "words": int(dist.size),
            "per_step": [int(v) for v in dist.reshape(dist.shape[0], -1)
                         .max(1)],
            "per_regime": {r: int(col_axis[:, regimes == r].max())
                           for r in sorted(set(regimes))},
        }
    return report


def _cmd_gen(args) -> None:
    inputs = build_inputs()
    write_input(Path(args.out), inputs, args.nstep)
    print(f"{inputs['ncol']} columns, nz={NZ}, groups:",
          [(g[0], g[2]) for g in inputs["groups"]])


def _cmd_pack(args) -> None:
    inputs = build_inputs()
    records = read_output(Path(args.records), inputs["ncol"], args.nstep)
    prov = json.loads(Path(args.provenance).read_text()) \
        if args.provenance else {}
    pack(inputs, records, args.mixlength, args.nstep, Path(args.out), prov,
         args.variant)
    nonfinite = {k: int(np.count_nonzero(~np.isfinite(v)))
                 for k, v in records.items() if v.dtype.kind == "f"
                 and not np.isfinite(v).all()}
    print("packed", args.out, "non-finite words:", nonfinite or "none")


def _cmd_grade(args) -> None:
    fixture = load_fixture(Path(args.fixture))
    summary = {}
    for mode in args.modes.split(","):
        actual = run_woof(fixture, replay=(mode == "replay"))
        summary[mode] = grade(fixture, actual)
    text = json.dumps(summary, indent=1)
    if args.json:
        Path(args.json).write_text(text)
    for mode, report in summary.items():
        worst = {k: v["max_ulp"] for k, v in report.items() if v["max_ulp"]}
        words = sum(v["words"] for v in report.values())
        differ = sum(v["words_differ"] for v in report.values())
        print(f"[{mode}] ml={fixture['meta']['mixlength']} words={words} "
              f"differ={differ} nonzero-ULP outputs: {worst or 'none'}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("gen")
    p.add_argument("out")
    p.add_argument("--nstep", type=int, default=12)
    p.set_defaults(func=_cmd_gen)
    p = sub.add_parser("pack")
    p.add_argument("records")
    p.add_argument("out")
    p.add_argument("--mixlength", type=int, required=True)
    p.add_argument("--nstep", type=int, default=12)
    p.add_argument("--provenance")
    p.add_argument("--variant", choices=VARIANTS, default="base")
    p.set_defaults(func=_cmd_pack)
    p = sub.add_parser("grade")
    p.add_argument("fixture")
    p.add_argument("--modes", default="replay,free")
    p.add_argument("--json")
    p.set_defaults(func=_cmd_grade)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
