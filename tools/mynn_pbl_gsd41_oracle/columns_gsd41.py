"""Column set of the gsd_41 MYNN driver column oracle (84 columns).

Twenty-one regimes, four variants each, written as one little-endian stream
file that ``run_driver_columns_gsd41.F90`` reads.  Every value is a float32
bit pattern, so the Fortran driver and the CUDA driver start from the same
bits; nothing here has to agree with anything to the last digit, because
the oracle file records the bits both sides actually ran.

Regimes: convective land day, deep dry convective (desert), stable night
with dew, very stable weak-wind night, marine stratocumulus, shallow
cumulus, tropical ocean, cold-air outbreak over warm water, cold pool with
fog, ice and frost, arctic snow-covered surface, mixed-phase cloud in the
253..273 K blend, high terrain day (700 hPa surface), high terrain night
(650 hPa surface), high wind, neutral with zero surface fluxes, saturated
fog layer, capped elevated mixed layer, deep moist tropical land, a very dry
column, cirrus over a clear boundary layer, and float32-subnormal
condensate and snow over a clear mixed layer.

Variants: (0) dx 3 km, the base column; (1) dx 750 m and a stronger wind;
(2) dx 13 km on a finer grid (20 m first layer) with weaker fluxes;
(3) dx 25 km with a weak-wind, near-floor surface layer.

Snow: the fork's mynn_bl_driver has no snow argument, so the snow profile
written here (``qs``) is read by the Fortran harness and never handed to
the fork.  The CUDA driver receives it under ``flag_qs = True``, the HRRR
door's Thompson configuration, and must not let it change anything.

usage: python columns_gsd41.py OUT.bin [NSTEP] [DELT]
"""

from __future__ import annotations

import struct
import sys

import numpy as np

F = np.float32
NZ = 40
MAGIC = 0x4D594E31  # "MYN1"
SCALARS = ("dx", "znt", "xland", "ts", "qsfc", "ps", "ust", "ch", "hfx",
           "qfx", "wspd", "rmol")
PROFILES = ("dz", "u", "v", "w", "th", "qv", "qc", "qi", "p", "exner", "rho",
            "t3d", "rthraten", "qs")

# name, th_ml, qsurf, zinv, dthml, dthinv, dqinv, gam, ts, ust, hfx, qfx,
# znt, ch, uref, ushear, xland, ps, cldtop, extras
_BASE = dict(dthml=0.0, dthinv=4.0, dqinv=2.0e-3, gam=0.004, xland=1.0,
             ps=100000.0, cldtop=-1.0, w=0.0, ice=None, liq=None, snow=None,
             rad=None, qfloor=1.0e-4, ch=0.010, vref=-1.0, mixed=False,
             rmol=None, supersat=None, trace=None)
REGIMES = [
    dict(name="convective_day", th_ml=303.0, qsurf=0.0120, zinv=1600.0,
         ts=308.0, ust=0.45, hfx=320.0, qfx=9.0e-5, znt=0.10, ch=0.012,
         uref=4.0, ushear=0.0015),
    dict(name="desert_deep_dry", th_ml=312.0, qsurf=0.0040, zinv=3200.0,
         dthinv=2.0, dqinv=1.0e-3, gam=0.005, ts=325.0, ust=0.55, hfx=520.0,
         qfx=1.0e-5, znt=0.05, ch=0.014, uref=6.0, ushear=0.001,
         ps=92000.0),
    dict(name="stable_night_dew", th_ml=290.0, qsurf=0.0105, zinv=250.0,
         dthml=0.016, dthinv=1.0, dqinv=5.0e-4, ts=286.5, ust=0.12,
         hfx=-38.0, qfx=-1.2e-5, znt=0.50, ch=0.004, uref=2.0,
         ushear=0.012, jet=True),
    dict(name="very_stable_calm", th_ml=280.0, qsurf=0.0055, zinv=120.0,
         dthml=0.05, dthinv=3.0, dqinv=4.0e-4, ts=272.0, ust=0.03,
         hfx=-8.0, qfx=-2.0e-6, znt=0.03, ch=0.001, uref=0.5,
         ushear=0.004, rmol=1.5),
    dict(name="marine_stratocumulus", xland=2.0, th_ml=289.0, qsurf=0.0102,
         zinv=850.0, dthinv=9.0, dqinv=5.0e-3, ts=289.5, ust=0.25,
         hfx=12.0, qfx=3.5e-5, znt=0.0002, ch=0.010, uref=7.0,
         ushear=0.001, cldtop=850.0, w=-0.004,
         rad=(-120.0, 0.0, -1.5e-4)),
    dict(name="shallow_cumulus", th_ml=300.0, qsurf=0.0130, zinv=1100.0,
         dthinv=1.0, dqinv=1.5e-3, gam=0.0035, ts=303.5, ust=0.35,
         hfx=190.0, qfx=2.1e-4, znt=0.15, ch=0.011, uref=5.0,
         ushear=0.0010, cldtop=2200.0, w=0.05, liq=(1100.0, 2200.0, 4.0e-5)),
    dict(name="tropical_ocean", xland=2.0, th_ml=299.5, qsurf=0.0175,
         zinv=600.0, dthinv=1.5, dqinv=2.5e-3, gam=0.0035, ts=301.5,
         ust=0.30, hfx=15.0, qfx=1.1e-4, znt=0.0003, ch=0.011, uref=8.0,
         ushear=0.0008, cldtop=1800.0, liq=(600.0, 1800.0, 2.0e-5)),
    dict(name="cold_air_outbreak", xland=2.0, th_ml=268.0, qsurf=0.0018,
         zinv=1400.0, dthinv=5.0, dqinv=6.0e-4, ts=280.0, ust=0.60,
         hfx=380.0, qfx=1.6e-4, znt=0.0015, ch=0.016, uref=15.0,
         ushear=0.002, cldtop=1400.0, mixed=True,
         snow=(0.0, 1400.0, 8.0e-5)),
    dict(name="cold_pool_fog_ice", th_ml=268.5, qsurf=0.0029, zinv=400.0,
         dthml=0.006, dthinv=5.0, dqinv=8.0e-4, ts=266.0, ust=0.08,
         hfx=-12.0, qfx=-3.0e-6, znt=0.05, ch=0.003, uref=1.5,
         ushear=0.004, ps=98000.0, cldtop=300.0,
         ice=(1500.0, 3500.0, 2.0e-5), snow=(0.0, 3500.0, 3.0e-5)),
    dict(name="arctic_snowpack", th_ml=246.0, qsurf=0.00035, zinv=300.0,
         dthml=0.02, dthinv=8.0, dqinv=1.0e-4, gam=0.006, ts=241.0,
         ust=0.15, hfx=-25.0, qfx=-1.0e-6, znt=0.001, ch=0.004, uref=4.0,
         ushear=0.006, qfloor=2.0e-6, ice=(0.0, 900.0, 6.0e-6),
         snow=(0.0, 900.0, 1.5e-5)),
    dict(name="mixed_phase_cloud", th_ml=276.0, qsurf=0.0042, zinv=1700.0,
         dthinv=3.0, dqinv=1.0e-3, gam=0.0045, ts=275.5, ust=0.30,
         hfx=20.0, qfx=1.5e-5, znt=0.20, ch=0.009, uref=9.0, ushear=0.003,
         cldtop=1700.0, mixed=True, snow=(0.0, 1700.0, 1.2e-4),
         rad=(-150.0, 0.0, -1.0e-4)),
    dict(name="high_terrain_day", th_ml=314.0, qsurf=0.0060, zinv=2400.0,
         dthinv=1.5, dqinv=1.5e-3, gam=0.0045, ts=318.0, ust=0.50,
         hfx=420.0, qfx=6.0e-5, znt=0.25, ch=0.013, uref=7.0,
         ushear=0.002, ps=70000.0),
    dict(name="high_terrain_night", th_ml=296.0, qsurf=0.0030, zinv=200.0,
         dthml=0.03, dthinv=2.0, dqinv=4.0e-4, ts=287.0, ust=0.18,
         hfx=-45.0, qfx=-4.0e-6, znt=0.60, ch=0.005, uref=3.0,
         ushear=0.008, ps=65000.0, snow=(0.0, 400.0, 1.0e-5)),
    dict(name="high_wind", th_ml=295.0, qsurf=0.0090, zinv=1300.0,
         dthinv=2.0, dqinv=1.0e-3, ts=295.6, ust=0.95, hfx=25.0, qfx=6.0e-5,
         znt=0.40, ch=0.020, uref=18.0, ushear=0.006),
    dict(name="neutral_zero_flux", th_ml=292.0, qsurf=0.0080, zinv=900.0,
         dthinv=2.0, dqinv=1.0e-3, ts=292.0, ust=0.30, hfx=0.0, qfx=0.0,
         znt=0.10, ch=0.008, uref=8.0, ushear=0.002, rmol=0.0),
    dict(name="saturated_fog", th_ml=283.0, qsurf=0.0090, zinv=200.0,
         dthml=0.004, dthinv=4.0, dqinv=1.5e-3, ts=282.0, ust=0.10,
         hfx=-10.0, qfx=-1.0e-6, znt=0.10, ch=0.004, uref=1.5,
         ushear=0.004, cldtop=220.0, rad=(-80.0, 0.0, -2.0e-4)),
    dict(name="capped_elevated_mixed_layer", th_ml=301.0, qsurf=0.0165,
         zinv=900.0, dthinv=6.0, dqinv=9.0e-3, gam=0.003, ts=306.0,
         ust=0.40, hfx=250.0, qfx=1.4e-4, znt=0.08, ch=0.012, uref=10.0,
         ushear=0.004, w=0.01),
    dict(name="deep_moist_tropical_land", th_ml=301.5, qsurf=0.0200,
         zinv=800.0, dthinv=0.8, dqinv=1.5e-3, gam=0.0032, ts=304.0,
         ust=0.30, hfx=150.0, qfx=2.6e-4, znt=0.80, ch=0.010, uref=3.0,
         ushear=0.0012, cldtop=3500.0, liq=(800.0, 3500.0, 1.5e-4),
         ice=(5500.0, 8000.0, 5.0e-5), snow=(4500.0, 8000.0, 2.0e-4),
         w=0.15),
    dict(name="very_dry_column", th_ml=305.0, qsurf=0.0008, zinv=2000.0,
         dthinv=2.0, dqinv=6.0e-4, gam=0.005, ts=312.0, ust=0.40,
         hfx=300.0, qfx=1.0e-6, znt=0.02, ch=0.012, uref=5.0,
         ushear=0.001, qfloor=1.0e-6),
    dict(name="cirrus_over_clear_bl", th_ml=297.0, qsurf=0.0100,
         zinv=1200.0, dthinv=2.0, dqinv=2.0e-3, ts=300.0, ust=0.35,
         hfx=150.0, qfx=8.0e-5, znt=0.12, ch=0.011, uref=6.0,
         ushear=0.003, ice=(7000.0, 9500.0, 3.0e-5),
         snow=(6500.0, 9500.0, 4.0e-5), supersat=(7000.0, 9500.0, 1.05)),
    # float32 subnormal condensate and snow (the CUDA compile flushes
    # subnormals unless the arithmetic is written around it; gfortran
    # keeps them) in and above a cloud-free mixed layer
    dict(name="subnormal_trace_condensate", th_ml=298.0, qsurf=0.0110,
         zinv=1000.0, dthinv=2.0, dqinv=2.0e-3, ts=301.0, ust=0.35,
         hfx=120.0, qfx=7.0e-5, znt=0.10, ch=0.010, uref=5.0,
         ushear=0.002, trace=(300.0, 4000.0, 3.0e-40, 2.0e-41),
         snow=(2000.0, 4000.0, 5.0e-42)),
]


def _es(tk: float) -> float:
    """The families oracle's liquid/ice saturation vapour pressure (Pa)."""
    if tk > 253.0:
        return 611.2 * np.exp(17.67 * (tk - 273.15) / (tk - 29.65))
    return 611.2 * np.exp(21.8745584 * (tk - 273.16) / (tk - 7.66))


def _rmol(th: float, ust: float, hfx: float, qfx: float, ps: float) -> float:
    """1/L from the surface fluxes (Obukhov), the surface layer's value."""
    rho = ps / (287.0 * th)
    thv_flux = hfx / (rho * 1004.5) + 0.61 * th * qfx / rho
    return -0.4 * 9.81 * thv_flux / (th * max(ust, 0.01) ** 3)


def column(regime: dict, variant: int) -> tuple[dict, dict]:
    r = dict(_BASE)
    r.update(regime)
    dx, dz1, dzslope = 3000.0, 40.0, 12.0
    wind, flux, ust = 1.0, 1.0, r["ust"]
    if variant == 1:
        dx, wind = 750.0, 1.5
        ust = r["ust"] * 1.3
    elif variant == 2:
        dx, dz1, dzslope, flux = 13000.0, 20.0, 15.0, 0.5
    elif variant == 3:
        dx, wind, flux = 25000.0, 0.35, 0.7
        ust = max(r["ust"] * 0.4, 0.02)
    hfx, qfx = r["hfx"] * flux, r["qfx"] * flux
    ps = r["ps"]
    rmol = r["rmol"]
    if rmol is None:
        rmol = _rmol(r["th_ml"], ust, hfx, qfx, ps)
    elif variant:
        rmol = rmol * (1.0 + 0.5 * variant)
    s = {"dx": dx, "znt": r["znt"], "xland": r["xland"], "ts": r["ts"],
         "ps": ps, "ust": ust, "ch": r["ch"], "hfx": hfx, "qfx": qfx,
         "rmol": rmol}
    p = {name: np.zeros(NZ) for name in PROFILES}
    zw = 0.0
    for k in range(NZ):
        dz = dz1 + dzslope * k
        zm = zw + 0.5 * dz
        pk = ps * np.exp(-zm / 8500.0)
        exner = (pk / 100000.0) ** (287.0 / (7.0 * 287.0 / 2.0))
        zinv = r["zinv"]
        if zm <= zinv:
            th = r["th_ml"] + r["dthml"] * zm
            qt = r["qsurf"] - 1.0e-7 * zm
        else:
            th = (r["th_ml"] + r["dthml"] * zinv + r["dthinv"]
                  + r["gam"] * (zm - zinv))
            qt = max(r["qsurf"] - 1.0e-7 * zinv - r["dqinv"]
                     - 2.5e-6 * (zm - zinv), r["qfloor"])
        tk = th * exner
        es = _es(tk)
        qs = 0.622 * es / (pk - 0.378 * es)
        sqc = sqi = 0.0
        if zm <= r["cldtop"] and qt > qs:
            sqv = qs
            if r["mixed"] and 253.0 < tk < 273.16:
                frac = (tk - 253.0) / 20.16
                sqc, sqi = frac * (qt - qs), (1.0 - frac) * (qt - qs)
            elif tk > 263.0:
                sqc = qt - qs
            else:
                sqc = sqi = 0.5 * (qt - qs)
        else:
            sqv = min(qt, qs)
        if r["supersat"] and r["supersat"][0] < zm < r["supersat"][1]:
            sqv = min(qt, qs * r["supersat"][2])
        if r["liq"] and r["liq"][0] < zm < r["liq"][1] and sqc == 0.0:
            sqc = r["liq"][2]
        if r["ice"] and r["ice"][0] < zm < r["ice"][1]:
            sqi = max(sqi, r["ice"][2])
        if r["trace"] and r["trace"][0] < zm < r["trace"][1]:
            sqc, sqi = r["trace"][2], r["trace"][3]
        sqs = 0.0
        if r["snow"] and r["snow"][0] <= zm < r["snow"][1]:
            sqs = r["snow"][2] * (0.5 + 0.5 * zm / r["snow"][1])
        u = (r["uref"] + r["ushear"] * zm) * wind
        if r.get("jet") and zm < 600.0:
            u = (r["uref"] + 9.0 * zm / 600.0) * wind
        v = (r["vref"] + 0.0008 * zm) * wind
        rad = 0.0
        if r["rad"] and (r["cldtop"] + r["rad"][0] < zm <= r["cldtop"]
                         + r["rad"][1]):
            rad = r["rad"][2]
        p["dz"][k], p["u"][k], p["v"][k] = dz, u, v
        p["w"][k] = r["w"] if zm < 6000.0 else 0.0
        p["th"][k], p["p"][k], p["exner"][k] = th, pk, exner
        # the fork's inputs are mixing ratios
        p["qv"][k] = sqv / (1.0 - sqv)
        p["qc"][k] = sqc / (1.0 - sqv)
        p["qi"][k] = sqi / (1.0 - sqv)
        p["qs"][k] = sqs / (1.0 - sqv)
        p["rthraten"][k] = rad
        zw += dz
    prof = {name: np.asarray(value, F) for name, value in p.items()}
    # t3d and rho as the harness's between-step update forms them
    prof["t3d"] = (prof["th"] * prof["exner"]).astype(F)
    prof["rho"] = (prof["p"] / (F(287.0) * prof["t3d"])).astype(F)
    sca = {name: F(value) for name, value in s.items()}
    sca["qsfc"] = F(prof["qv"][0] / (F(1.0) + prof["qv"][0]))
    sca["wspd"] = F(max(np.sqrt(F(prof["u"][0]) ** 2
                                + F(prof["v"][0]) ** 2), F(1.0)))
    return sca, prof


def columns() -> tuple[list[str], dict, dict]:
    names, scal, prof = [], {n: [] for n in SCALARS}, {n: [] for n in PROFILES}
    for regime in REGIMES:
        for variant in range(4):
            sca, pro = column(regime, variant)
            names.append(f"{regime['name']}/v{variant}")
            for n in SCALARS:
                scal[n].append(sca[n])
            for n in PROFILES:
                prof[n].append(pro[n])
    return (names, {n: np.asarray(v, F) for n, v in scal.items()},
            {n: np.asarray(v, F) for n, v in prof.items()})


NAMES = columns()[0]


def write(path: str, nstep: int = 12, delt: float = 20.0) -> None:
    names, scal, prof = columns()
    ncol = len(names)
    with open(path, "wb") as out:
        out.write(struct.pack("<4i", MAGIC, ncol, NZ, nstep))
        out.write(np.asarray([delt], "<f4").tobytes())
        # scal(12, ncol) and prof(nz, 14, ncol), Fortran order
        out.write(np.stack([scal[n] for n in SCALARS]).T.astype("<f4")
                  .tobytes())
        out.write(np.stack([prof[n] for n in PROFILES], axis=1)
                  .astype("<f4").tobytes())


if __name__ == "__main__":
    write(sys.argv[1], *(int(a) for a in sys.argv[2:3]),
          *(float(a) for a in sys.argv[3:4]))
