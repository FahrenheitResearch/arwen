"""Adversarial column sets for the TEMPO column oracle (refute pass).

The lane's oracle (make_columns.py) uses one 60-level grid (25 m at the
ground, top 18.6 km), dt 20/60/150 s and the RRFS-v2 configuration.  This
file builds column sets the lane did not use:

  A  deep grid: 100 levels to 38.6 km (nz equals the runtime's default
     TEMPO_NZ_MAX of 100), stratospheric pressure down to a few hundred Pa,
     polar-stratospheric ice at 188 K, overshooting-top ice and snow; the
     first column triggers TEMPO's first-call aerosol and graupel init on
     this grid.  dt 30 s, 20 steps.
  B  thin bottom layers: 4 m at the ground stretching 12 % per level
     (many sedimentation sub-steps), heavy rain and hail.  dt 60 s, 15 steps.
  C  coarse grid: 10 levels of 1600 m, every regime template, dt 300 s
     (twice the lane's largest), 8 steps.
  D  standard grid, physics extremes the lane's edge cases do not reach:
     supercooled cloud and rain at 228-235 K, ice supersaturation 1.6 at
     213 K under high aerosol (homogeneous aerosol freezing), temperature
     exactly at the homogeneous-freezing threshold, graupel at 50 kg m-3
     everywhere with a huge number, giant and tiny rain drops, heavy snow
     melting at 278 K, 20 % water supersaturation at 298 K, a -40 m/s
     downdraft through warm rain, zero and above-limit aerosol in a column
     that is not the first call, 1e9/kg ice number, snow just above r1,
     T = 273.15 K in every level with every species, a 330 K saturated
     surface, and qv = 0 under ice cloud.  dt 20 s, 30 steps.

Same binary layout as make_columns.py, except that each set carries its own
nz (the Fortran driver and the readers already take nz per set).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import make_columns as mc  # noqa: E402

FIELDS = mc.FIELDS


def set_grid(dz):
    dz = np.asarray(dz, dtype=np.float64)
    zf = np.concatenate([[0.0], np.cumsum(dz)])
    mc.NZ = dz.size
    mc.DZ = dz
    mc.ZM = 0.5 * (zf[1:] + zf[:-1])


def stretched(nz, d0, growth, cap):
    out, d = [], d0
    for _ in range(nz):
        out.append(d)
        d = min(d * growth, cap)
    return np.array(out)


def templates():
    for regime, fns in mc.REGIMES.items():
        for fn in fns:
            yield regime, fn


def set_a(b):
    set_grid(stretched(100, 20.0, 1.06, 700.0))
    cols, cat = [], []
    c = mc.conv_supercell(b)
    for f in ("nwfa", "nifa", "ng", "qb"):
        c[f][:] = 0.0
    cols.append(c)
    cat.append(("deep", "deep_first_call_init_aerosol_and_graupel"))
    for regime, fn in templates():
        cols.append(fn(b))
        cat.append(("deep", f"deep_{fn.__name__}"))
    # polar stratospheric ice: 188 K between 20 and 24 km, ice supersaturated
    c = mc.winter_snow(b)
    z = mc.ZM
    psc = (z > 20000.0) & (z < 24000.0)
    c["t"] = np.where(psc, 188.0, c["t"])
    c["qv"] = np.where(psc, 1.3 * mc.qsat(c["t"], c["p"], ice=True), c["qv"])
    c["qi"] = np.where(psc, 1.0e-6, c["qi"])
    c["ni"] = np.where(psc, 1.0e5, c["ni"])
    cols.append(c)
    cat.append(("deep", "deep_polar_stratospheric_ice_188K"))
    # overshooting top: ice, snow and graupel above the tropopause
    c = mc.conv_supercell(b)
    over = (z > 13000.0) & (z < 18000.0)
    c["qi"] = np.where(over, 8.0e-4, c["qi"])
    c["ni"] = np.where(over, mc.number_from_mass(np.full(mc.NZ, 8.0e-4), 40e-6, 890.0), c["ni"])
    c["qs"] = np.where(over, 1.0e-3, c["qs"])
    c["qg"] = np.where(over, 5.0e-4, c["qg"])
    c["ng"] = np.where(over, 5.0e-4 / (math.pi / 6.0 * 300.0 * 2.0e-3 ** 3), c["ng"])
    c["qb"] = np.where(over, 5.0e-4 * 1000.0 / 300.0, c["qb"])
    c["w"] = mc.updraft(800.0, 17000.0, 45.0)
    cols.append(c)
    cat.append(("deep", "deep_overshooting_top"))
    return dict(dt=30.0, nsteps=20, cols=cols, cat=cat, nz=mc.NZ)


def set_b(b):
    set_grid(stretched(60, 4.0, 1.12, 600.0))
    cols, cat = [], []
    for fn in (mc.conv_supercell, mc.conv_elevated_hail, mc.conv_pulse, mc.ocean_eyewall,
               mc.ocean_trop_warm_rain, mc.terrain_summer_storm, mc.sleet, mc.freezing_rain):
        for v in range(2):
            cols.append(fn(b))
            cat.append(("thin", f"thin_{fn.__name__}_v{v}"))
    c = mc.conv_elevated_hail(b)
    c["qg"] = mc.layer(0.0, 6000.0, 4.0e-2)
    c["ng"] = c["qg"] / (math.pi / 6.0 * 900.0 * 3.0e-2 ** 3)
    c["qb"] = c["qg"] * 1000.0 / 900.0
    cols.append(c)
    cat.append(("thin", "thin_giant_hail_at_ground"))
    c = mc.ocean_trop_warm_rain(b)
    c["qr"] = mc.layer(0.0, 4000.0, 2.5e-2)
    c["nr"] = mc.number_from_mass(c["qr"], 4.0e-3, 1000.0)
    cols.append(c)
    cat.append(("thin", "thin_extreme_rain_at_ground"))
    return dict(dt=60.0, nsteps=15, cols=cols, cat=cat, nz=mc.NZ)


def set_c(b):
    set_grid(np.full(10, 1600.0))
    cols, cat = [], []
    for regime, fn in templates():
        cols.append(fn(b))
        cat.append(("coarse", f"coarse_{fn.__name__}"))
    return dict(dt=300.0, nsteps=8, cols=cols, cat=cat, nz=mc.NZ)


def set_d(b):
    set_grid(stretched(60, 25.0, 1.08, 600.0))
    z = mc.ZM
    cols, cat = [], []

    def add(name, c):
        cols.append(c)
        cat.append(("extreme", name))

    c = b.column(ts=245.0, psfc=100000.0, lapse_low=6.0e-3, lapse_high=6.0e-3, z_tp=9000.0,
                 rh_low=1.0, rh_mid=0.9, rh_high=0.5, rh_cloud=[(1000.0, 3000.0, 1.0)],
                 qc=mc.layer(1000.0, 3000.0, 5.0e-4), qr=mc.layer(0.0, 3000.0, 2.0e-4),
                 w=mc.updraft(0.0, 4000.0, 2.0))
    add("supercooled_cloud_and_rain_228_to_245K", c)

    c = b.column(ts=290.0, psfc=100000.0, lapse_low=7.0e-3, lapse_high=7.0e-3, z_tp=12500.0,
                 rh_low=0.6, rh_mid=0.6, rh_high=0.6, ice_rh=True,
                 rh_cloud=[(9000.0, 11500.0, 1.6)], w=mc.updraft(8000.0, 12000.0, 3.0),
                 nwfa_sfc=5.0e9, nwfa_h=20000.0, nifa_sfc=5.0e6, nifa_h=20000.0)
    add("ice_supersaturation_1p6_homogeneous_aerosol_freezing", c)

    c = mc.winter_snow(b)
    c["t"][10:30] = 273.15 - 38.0
    c["qc"][10:30] = 2.0e-4
    c["nc"][10:30] = 1.0e8
    add("t_exactly_hgfrz_with_cloud_water", c)

    c = mc.conv_pulse(b)
    c["qb"] = np.where(c["qg"] > 0.0, c["qg"] * 1000.0 / 50.0, 0.0)
    c["ng"] = np.where(c["qg"] > 0.0, 1.0e7, 0.0)
    add("graupel_50_kgm3_everywhere_huge_number", c)

    c = mc.ocean_trop_warm_rain(b)
    c["qr"] = mc.layer(0.0, 4000.0, 1.0e-2)
    c["nr"] = np.where(c["qr"] > 0.0, 1.0, 0.0)
    add("giant_rain_drops_nr_1_per_kg", c)

    c = mc.ocean_trop_warm_rain(b)
    c["qr"] = mc.layer(0.0, 4000.0, 1.0e-3)
    c["nr"] = np.where(c["qr"] > 0.0, 1.0e10, 0.0)
    add("tiny_rain_drops_nr_1e10", c)

    c = mc.winter_snow(b)
    c["t"] = np.where(z < 1500.0, 278.0, c["t"])
    c["qs"] = mc.layer(0.0, 3000.0, 5.0e-3)
    add("heavy_snow_melting_at_278K", c)

    c = mc.stable_night_clear(b)
    c["t"] = mc.temperature(298.0, 6.5e-3, 6.5e-3, 3000.0, 15000.0)
    c["qv"] = np.where(z < 3000.0, 1.2 * mc.qsat(c["t"], c["p"]), c["qv"])
    c["w"] = mc.updraft(0.0, 3000.0, 5.0)
    add("water_supersaturation_1p2_no_condensate", c)

    c = mc.ocean_trop_warm_rain(b)
    c["w"] = -mc.updraft(0.0, 6000.0, 40.0)
    add("downdraft_minus_40_through_warm_rain", c)

    c = mc.conv_mcs_stratiform(b)
    c["nwfa"][:] = 0.0
    c["nifa"][:] = 0.0
    add("zero_aerosol_not_first_call", c)

    c = mc.conv_mcs_stratiform(b)
    c["nwfa"][:] = 1.0e14
    c["nifa"][:] = 1.0e12
    add("aerosol_above_limits", c)

    c = mc.arctic_ice_fog(b)
    c["qi"] = mc.layer(0.0, 3000.0, 1.0e-3)
    c["ni"] = np.where(c["qi"] > 0.0, 1.0e9, 0.0)
    add("ice_number_1e9_per_kg", c)

    c = mc.winter_snow(b)
    for f in ("qc", "qr", "qi", "qg"):
        c[f][:] = 0.0
    for f in ("nc", "nr", "ni", "ng", "qb"):
        c[f][:] = 0.0
    c["qs"] = np.full(mc.NZ, 2.0e-12)
    add("snow_just_above_r1_everywhere", c)

    c = mc.conv_mcs_stratiform(b)
    c["t"][:] = 273.15
    for f, v in (("qc", 3e-4), ("qr", 5e-4), ("qi", 1e-4), ("qs", 5e-4), ("qg", 3e-4)):
        c[f] = np.full(mc.NZ, v)
    c["nc"] = np.full(mc.NZ, 1.0e8)
    c["nr"] = np.full(mc.NZ, 1.0e4)
    c["ni"] = np.full(mc.NZ, 1.0e5)
    c["ng"] = np.full(mc.NZ, 1.0e3)
    c["qb"] = np.full(mc.NZ, 3e-4 * 1000.0 / 400.0)
    add("t_273p15_every_level_every_species", c)

    c = b.column(ts=330.0, psfc=101000.0, lapse_low=9.0e-3, lapse_high=8.0e-3, z_tp=16000.0,
                 rh_low=1.0, rh_mid=0.9, rh_high=0.5, rh_cloud=[(500.0, 6000.0, 1.01)],
                 qc=mc.layer(500.0, 6000.0, 4.0e-3), qr=mc.layer(0.0, 5000.0, 1.0e-2),
                 w=mc.updraft(500.0, 12000.0, 20.0))
    add("hot_330K_saturated_heavy_rain", c)

    c = mc.arctic_ice_fog(b)
    c["qv"][:] = 0.0
    c["qi"] = mc.layer(2000.0, 5000.0, 1.0e-4)
    c["ni"] = mc.number_from_mass(c["qi"], 60e-6, 890.0)
    add("qv_zero_under_ice_cloud", c)
    return dict(dt=20.0, nsteps=30, cols=cols, cat=cat, nz=mc.NZ)


def build(seed=20261009):
    b = mc.Builder(np.random.default_rng(seed))
    return [set_a(b), set_b(b), set_c(b), set_d(b)]


def write(sets, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = []
    with open(out_dir / "columns_in.bin", "wb") as f:
        np.array([len(sets)], dtype="<i8").tofile(f)
        for si, s in enumerate(sets, start=1):
            ncol, nz = len(s["cols"]), s["nz"]
            np.array([ncol, nz, s["nsteps"]], dtype="<i8").tofile(f)
            np.array([s["dt"]], dtype="<f8").tofile(f)
            for name in FIELDS:
                a = np.stack([np.asarray(c[name], dtype=np.float64) for c in s["cols"]])
                assert a.shape == (ncol, nz), (name, a.shape)
                assert np.all(np.isfinite(a)), name
                a.astype("<f8").tofile(f)
            for ci, (regime, variant) in enumerate(s["cat"], start=1):
                c = s["cols"][ci - 1]
                catalog.append(dict(set=si, column=ci, dt=s["dt"], nsteps=s["nsteps"], nz=nz,
                                    regime=regime, variant=variant,
                                    t_sfc=float(c["t"][0]), p_sfc=float(c["p"][0]),
                                    t_min=float(np.min(c["t"])), p_top=float(c["p"][-1]),
                                    dz_min=float(np.min(c["dz"])), dz_max=float(np.max(c["dz"])),
                                    w_max=float(np.max(c["w"])), w_min=float(np.min(c["w"])),
                                    q_total_max=float(np.max(c["qc"] + c["qr"] + c["qi"]
                                                             + c["qs"] + c["qg"]))))
    (out_dir / "columns_catalog.json").write_text(json.dumps(catalog, indent=1) + "\n")
    return catalog


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    cat = write(build(), out)
    by = {}
    for row in cat:
        by[(row["set"], row["regime"], row["nz"])] = by.get((row["set"], row["regime"], row["nz"]), 0) + 1
    print(f"columns: {len(cat)}  by set: {by}")


if __name__ == "__main__":
    main()
