"""Run the RUC soil ingest over a real metgrid file, the way real.exe would.

The oracle fixture is twenty synthetic columns.  This is the other half of the
question: does the proved remap survive a real grid?  It reads a met_em file,
assembles the source exactly as WRF's own reader does
(``share/module_optional_input.F:1330-1372``), calls
:func:`gpuwm.ingest.ruc_soil.remap_soil_to_ruc_levels` over every column, and
reports the resulting nine-level column plus every input WRF would have
refused.

Usage::

    python tools/ruc_soil_ingest_wrf461_oracle/exercise_on_metgrid.py MET_EM.nc

It does NOT write anything.  Wiring this into the run path is a separate job:
``gpuwm.ingest.soil.preprocess_noah_soil`` and ``NoahSoilState`` are what the
ERA5/GFS/nest initialisers actually call, and a RUC run needs its own state
object and its own ``ruc_cold_start`` handoff.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gpuwm.ingest.ruc_soil import remap_soil_to_ruc_levels  # noqa: E402


def _layer_midpoints_cm(bottoms_deep_first) -> np.ndarray:
    """``share/module_optional_input.F:1339-1352``, in INTEGER arithmetic."""

    bottoms = [int(round(float(v))) for v in bottoms_deep_first][::-1]
    above = 0
    midpoints = []
    for bottom in bottoms:
        midpoints.append((above + bottom) // 2)
        above = bottom
    return np.asarray(midpoints, dtype=np.int64)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    from netCDF4 import Dataset

    path = Path(argv[1])
    data = Dataset(path)
    flag_layers = int(getattr(data, "FLAG_SOIL_LAYERS", 0))
    flag_levels = int(getattr(data, "FLAG_SOIL_LEVELS", 0))
    if (flag_layers, flag_levels) != (1, 0):
        print(f"{path.name}: FLAG_SOIL_LAYERS={flag_layers} "
              f"FLAG_SOIL_LEVELS={flag_levels}; only the layer arm is wired here")
        return 2

    levels = _layer_midpoints_cm(np.asarray(data.variables["SOIL_LAYERS"][0, :, 0, 0]))
    # :1364-1370 flips the profile so k = 1 is closest to the surface.
    temperature = np.asarray(data.variables["ST"][0], dtype=np.float32)[::-1]
    moisture = np.asarray(data.variables["SM"][0], dtype=np.float32)[::-1]
    skin = np.asarray(data.variables["SKINTEMP"][0], dtype=np.float32)
    deep = np.asarray(data.variables["SOILTEMP"][0], dtype=np.float32)
    landsea = np.asarray(data.variables["LANDSEA"][0], dtype=np.float32)
    sst = np.asarray(data.variables["SST"][0], dtype=np.float32)

    water = landsea < 0.5
    unusable = water & ((sst < 170.0) | (sst > 400.0))
    print(f"{path.name}: {temperature.shape[1]}x{temperature.shape[2]} "
          f"= {landsea.size} columns, {int(water.sum())} water")
    print(f"source sample depths (cm, WRF integer midpoints): {list(levels)}")
    print(f"water columns with nonphysical SST: {int(unusable.sum())} "
          f"({100.0 * unusable.sum() / max(1, water.sum()):.1f}% of water)")

    # dyn_em/module_initialize_real.F:3282-3297 repairs an unusable TSK from
    # SST; gpuwm's Noah path repairs an unusable SST from the skin temperature
    # (gpuwm/ingest/soil.py:275-277).  The second is the one that works here.
    repaired_sst = np.where(unusable, skin, sst).astype(np.float32)
    # :3300-3322: an out-of-range land TMN falls back to TSK.
    bad_deep = ~np.isfinite(deep) | (deep < 170.0) | (deep > 400.0)
    repaired_deep = np.where(bad_deep, skin, deep).astype(np.float32)
    print(f"land columns with nonphysical SOILTEMP: "
          f"{int((bad_deep & ~water).sum())}")

    result = remap_soil_to_ruc_levels(
        source_temperature=temperature,
        source_moisture=moisture,
        source_levels_cm=levels,
        source_geometry="layers",
        skin_temperature=skin,
        deep_temperature=repaired_deep,
        landmask=landsea,
        sea_surface_temperature=repaired_sst,
        num_soil_layers=9,
        moisture_adjustment=False,
    )

    tslb = result.soil_temperature
    smois = result.soil_moisture
    print()
    print(f"{'level':>5s} {'depth_m':>8s} "
          f"{'TSLB min':>9s} {'TSLB max':>9s} "
          f"{'SMOIS min':>10s} {'SMOIS max':>10s}")
    for index, depth in enumerate(result.level_depths):
        print(f"{index + 1:5d} {float(depth):8.3f} "
              f"{float(tslb[index].min()):9.3f} {float(tslb[index].max()):9.3f} "
              f"{float(smois[index].min()):10.5f} {float(smois[index].max()):10.5f}")

    problems = 0
    if not np.isfinite(tslb).all() or tslb.min() < 170.0 or tslb.max() > 400.0:
        print("FAIL: TSLB left 170..400 K")
        problems += 1
    if not np.isfinite(smois).all() or smois.min() < 0.0 or smois.max() > 1.0:
        print("FAIL: SMOIS left 0..1")
        problems += 1
    distinct = len({tslb[k].tobytes() for k in range(tslb.shape[0])})
    print(f"\ndistinct TSLB levels: {distinct} of {tslb.shape[0]}")
    if distinct != tslb.shape[0]:
        print("FAIL: the nine levels are not nine distinct fields")
        problems += 1
    print("OK" if problems == 0 else f"{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
