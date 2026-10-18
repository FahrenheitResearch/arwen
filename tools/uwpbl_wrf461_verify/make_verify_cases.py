"""Column cases for the WRF v4.6.1 UW PBL (bl_pbl_physics=9) verification.

    python make_verify_cases.py OUTDIR

Writes two case sets in run_camuwpbl.F90's format (tools/uwpbl_wrf471_oracle),
fresh columns that the 4.7.1 fixtures of record never saw:

  cases-v461-g35/g44/g61  nine regime families x 8 columns on each of the
      three vertical grids (35 levels dt 150 s, 44 levels dt 60 s, 61
      levels dt 20 s), four consecutive steps: the six record families
      (convective day, stable night, marine stratocumulus over ocean,
      valley cold pool with fog, cold mixed-phase ice cloud, shallow
      cumulus) plus high terrain, arctic snow/ice surface with ice fog,
      and a cold-air outbreak over open water.  216 columns.
  cases-v461edge          24 edge columns on the 44-level grid, four steps:
      calm, extreme surface heating and cooling, a bone-dry and a
      saturated column, very thin and very thick first layers, terrain
      below sea level and at 5.5 km, deep superadiabatic layers, a strong
      low-level jet, trace condensate, extreme cloud-top cooling, huge ice
      number, free convection under minimal friction velocity, 60 m/s
      winds, the level-1 wind floor, zero fluxes, negative moisture flux
      (deposition), and cloud fraction exactly 1 through the column.

The index files name each column's family and member.  Profiles are built
in float64 with +, -, *, /, exp, log and power, rounded once to float32;
the files, not the generator, are what the oracle and the port read.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "make_cases", _HERE.parent / "uwpbl_wrf471_oracle" / "make_cases.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

SEED = 20261007
FAMILIES = m.FAMILIES + m.EXTRA_FAMILIES
GRIDS = (("g35", 35, 14000.0, 150.0), ("g44", 44, 16500.0, 60.0),
         ("g61", 61, 20000.0, 20.0))
PER_FAMILY = 8
NSTEPS = 4


def _refresh(c):
    """Recompute t and rho after an edit, as make_cases does."""
    c["t"] = (c["th"] * c["exner"]).astype(np.float32)
    c["rho"] = (c["p"] / (m.R_D * c["t"].astype(np.float64)
                          * (1.0 + 0.608 * c["qv"].astype(np.float64)))
                ).astype(np.float32)
    return c


def _f32(a):
    return np.asarray(a, dtype=np.float64).astype(np.float32)


def edge_columns(rng, nk, ztop):
    cols, meta = [], []

    def add(name, c):
        cols.append(_refresh(c))
        meta.append({"family": "edge", "probe": name})

    c = m.build_column(rng, "stable", nk, ztop)
    c["u"][:] = _f32(np.linspace(0.0, 2.0, nk)); c["u"][0] = np.float32(1e-3)
    c["v"][:] = 0.0
    c["hfx"][:] = 0.0; c["qfx"][:] = 0.0; c["ust"][:] = np.float32(1e-3)
    add("calm: level-1 wind at the 1e-3 floor, zero fluxes, ust 1e-3", c)

    c = m.build_column(rng, "convective", nk, ztop)
    c["hfx"][:] = 800.0; c["qfx"][:] = np.float32(3e-4); c["ust"][:] = 1.2
    add("extreme surface heating hfx 800 W/m2", c)

    c = m.build_column(rng, "stable", nk, ztop)
    c["hfx"][:] = -250.0; c["ust"][:] = 0.6
    add("extreme surface cooling hfx -250 W/m2, windy", c)

    c = m.build_column(rng, "convective", nk, ztop)
    c["qv"][:] = np.float32(1e-7)
    c["qfx"][:] = 0.0
    add("bone-dry column qv 1e-7", c)

    c = m.build_column(rng, "shallowcu", nk, ztop)
    z = c["z"] - c["ht"][0]
    sat = z < 8000.0
    c["cldfra"][sat] = 1.0
    c["qc"][sat & (z < 4500.0)] = np.float32(5e-4)
    c["qnc"][sat & (z < 4500.0)] = np.float32(1e8)
    c["qi"][sat & (z >= 3000.0)] = np.float32(3e-5)
    c["qni"][sat & (z >= 3000.0)] = np.float32(1e5)
    es = m.esat_water(c["t"].astype(np.float64))
    qs = m.EP * es / (c["p"].astype(np.float64) - es)
    c["qv"][sat] = _f32(qs[sat])
    add("saturated deep cloud column to 8 km", c)

    c = m.build_column(rng, "convective", nk, ztop, dz0=4.0)
    add("very thin first layer, dz0 4 m", c)

    c = m.build_column(rng, "stable", nk, ztop, dz0=120.0)
    add("very thick first layer, dz0 120 m", c)

    c = m.build_column(rng, "convective", nk, ztop)
    c["z"] = c["z"] - np.float32(c["ht"][0] + 420.0)
    c["z_at_w"] = c["z_at_w"] - np.float32(c["ht"][0] + 420.0)
    c["ht"][:] = np.float32(-420.0)
    add("terrain below sea level, ht -420 m", c)

    c = m.build_column(rng, "highterrain", nk, ztop)
    shift = np.float32(5500.0) - c["ht"][0]
    c["z"] = c["z"] + shift; c["z_at_w"] = c["z_at_w"] + shift
    c["ht"][:] = np.float32(5500.0)
    add("very high terrain, ht 5500 m", c)

    c = m.build_column(rng, "convective", nk, ztop)
    z = c["z"] - c["ht"][0]
    deep = z < 4000.0
    c["th"][deep] = _f32(c["th"][0] - 1.5e-3 * z[deep])
    add("superadiabatic through 4 km", c)

    c = m.build_column(rng, "stable", nk, ztop)
    z = (c["z"] - c["ht"][0]).astype(np.float64)
    c["u"][:] = _f32(30.0 * np.exp(-((z - 150.0) / 120.0) ** 2) + 2.0)
    c["ust"][:] = 0.5
    add("low-level jet 30 m/s at 150 m over a stable surface", c)

    c = m.build_column(rng, "convective", nk, ztop)
    c["qc"][:] = np.float32(1e-9); c["qnc"][:] = np.float32(1e6)
    c["cldfra"][:] = np.float32(0.01)
    add("trace condensate qc 1e-9 everywhere", c)

    c = m.build_column(rng, "stratocu", nk, ztop)
    k = int(np.argmax(c["qc"]))
    c["rthratenlw"][k] = np.float32(-1e-2)
    add("cloud-top cooling -0.01 K/s", c)

    c = m.build_column(rng, "icecloud", nk, ztop)
    c["qni"][c["qi"] > 0] = np.float32(1e7)
    add("ice number 1e7 /kg where ice", c)

    c = m.build_column(rng, "convective", nk, ztop)
    c["ust"][:] = np.float32(1e-3); c["hfx"][:] = 300.0
    add("free convection, ust 1e-3 with hfx 300", c)

    c = m.build_column(rng, "stable", nk, ztop)
    c["u"][:] = 60.0; c["v"][:] = -10.0; c["ust"][:] = 1.0
    add("uniform 60 m/s wind", c)

    c = m.build_column(rng, "coldpool", nk, ztop)
    c["u"][0] = np.float32(1e-3); c["v"][0] = 0.0
    add("level-1 wind exactly at the 1e-3 floor in a cold pool", c)

    c = m.build_column(rng, "stratocu", nk, ztop)
    c["hfx"][:] = 0.0; c["qfx"][:] = 0.0
    add("zero surface fluxes under stratocumulus", c)

    c = m.build_column(rng, "arctic", nk, ztop)
    c["qfx"][:] = np.float32(-5e-6)
    add("negative moisture flux (deposition) on snow", c)

    c = m.build_column(rng, "stratocu", nk, ztop)
    c["cldfra"][:] = 1.0
    add("cloud fraction 1 in every layer", c)

    c = m.build_column(rng, "convective", nk, ztop)
    c["wsedl3d"][:] = np.float32(0.02)
    add("nonzero liquid sedimentation velocity", c)

    c = m.build_column(rng, "cao", nk, ztop)
    c["hfx"][:] = 650.0; c["qfx"][:] = np.float32(4e-4)
    add("extreme cold-air outbreak fluxes", c)

    c = m.build_column(rng, "stable", nk, ztop)
    c["hfx"][:] = np.float32(-0.01); c["ust"][:] = np.float32(0.01)
    add("near-neutral, nearly zero fluxes", c)

    c = m.build_column(rng, "icecloud", nk, ztop)
    c["qc"][:] = 0.0; c["qnc"][:] = 0.0
    add("ice-only cloud, no liquid", c)
    return cols, meta


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    index = {}
    for gname, nk, ztop, dt in GRIDS:
        cols, meta = [], []
        for family in FAMILIES:
            for j in range(PER_FAMILY):
                cols.append(m.build_column(rng, family, nk, ztop))
                meta.append({"family": family, "member": j})
        name = f"v461-{gname}"
        path = out / f"cases-{name}.bin"
        m.write_case_file(path, cols, nk, NSTEPS, dt)
        index[name] = {"file": path.name, "nk": nk, "dt": dt,
                       "nsteps": NSTEPS, "seed": SEED, "columns": meta}
    cols, meta = edge_columns(rng, 44, 16500.0)
    path = out / "cases-v461edge.bin"
    m.write_case_file(path, cols, 44, NSTEPS, 60.0)
    index["v461edge"] = {"file": path.name, "nk": 44, "dt": 60.0,
                         "nsteps": NSTEPS, "seed": SEED, "columns": meta}
    (out / "cases-v461-index.json").write_text(
        json.dumps(index, indent=1) + "\n", encoding="utf-8")
    total = sum(len(v["columns"]) for v in index.values())
    print(f"wrote {total} columns in {len(index)} case files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
