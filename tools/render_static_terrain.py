"""Render baseline-vs-Copernicus terrain through the REAL Rust renderer."""
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent / "wt-intl"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "wt-intl" / "tools"))
from gpuwm import rustwx
from gpuwm.static.lambert import LambertGrid
from terrain_source_crossvalidation import build_domain, default_geog_root

SP = Path(__file__).resolve().parent
CACHE = SP / "xval-cache"
OUT = Path(sys.argv[1])
WORK = SP / "renderwork"
shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir(parents=True)
GEOG = default_geog_root()
if GEOG is None:
    raise SystemExit(
        "set WPS_GEOG (or GPUWM_CASE_DATA_ROOT) to a staged geography tree")
STAMP = "2024-06-01_00:00:00"


def write_wrfout(path, grid, hgt):
    ny, nx = hgt.shape
    xc, yc = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5)
    lat, lon = grid.ij_to_latlon(xc, yc)
    with Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as f:
        f.createDimension("Time", None)
        f.createDimension("DateStrLen", 19)
        f.createDimension("west_east", nx)
        f.createDimension("south_north", ny)
        f.createDimension("bottom_top", 1)
        f.createDimension("west_east_stag", nx + 1)
        f.createDimension("south_north_stag", ny + 1)
        f.createDimension("bottom_top_stag", 2)
        t = f.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[0] = np.array(list(STAMP), dtype="S1")
        for name, data in (("XLAT", lat), ("XLONG", lon), ("HGT", hgt)):
            v = f.createVariable(name, "f4",
                                 ("Time", "south_north", "west_east"))
            v[0] = data.astype("f4")
        v = f.createVariable("T", "f4", ("Time", "bottom_top",
                                         "south_north", "west_east"))
        v[0] = np.zeros((1, ny, nx), dtype="f4")
        f.TITLE = "OUTPUT FROM GPUWM STATIC BUILDER"
        f.SIMULATION_START_DATE = STAMP
        f.START_DATE = STAMP
        setattr(f, "WEST-EAST_GRID_DIMENSION", nx + 1)
        setattr(f, "SOUTH-NORTH_GRID_DIMENSION", ny + 1)
        setattr(f, "BOTTOM-TOP_GRID_DIMENSION", 2)
        f.GRIDTYPE = "C"
        f.DX = np.float32(grid.dx); f.DY = np.float32(grid.dy)
        f.MAP_PROJ = np.int32(1)
        f.CEN_LAT = np.float32(grid.ref_lat)
        f.CEN_LON = np.float32(grid.ref_lon)
        f.TRUELAT1 = np.float32(grid.truelat1)
        f.TRUELAT2 = np.float32(grid.truelat2)
        f.MOAD_CEN_LAT = np.float32(grid.ref_lat)
        f.STAND_LON = np.float32(grid.stand_lon)
        f.POLE_LAT = np.float32(90.0); f.POLE_LON = np.float32(0.0)
        f.MMINLU = "MODIFIED_IGBP_MODIS_NOAH"
        f.ISWATER = np.int32(17); f.ISLAKE = np.int32(21)
        f.ISICE = np.int32(15); f.ISURBAN = np.int32(13)
        f.GRID_ID = np.int32(1); f.PARENT_ID = np.int32(0)
        f.I_PARENT_START = np.int32(1); f.J_PARENT_START = np.int32(1)
        f.PARENT_GRID_RATIO = np.int32(1)


renderer = rustwx.find_renderer()
print("renderer:", renderer)
assert renderer and "bmwork" not in str(renderer).lower(), renderer
print("probe:", rustwx.probe_renderer(renderer))

CASES = [("alps", 47.2, 13.6), ("frontrange", 39.55, -105.55)]
for tag, lat0, lon0 in CASES:
    grid = LambertGrid(ref_lat=lat0, ref_lon=lon0, truelat1=30.0,
                       truelat2=60.0, stand_lon=lon0, dx=500.0, dy=500.0,
                       e_we=101, e_sn=101)
    kw = dict(lat=lat0, lon=lon0, dx=500.0, n=100, geog_root=GEOG,
              cache_root=CACHE, case_date=date(2024, 6, 1))
    for label, src in (("baseline-900m", "baseline"),
                       ("copernicus-30m", "copernicus-dem-glo30")):
        hgt, _, _ = build_domain(source=src, **kw)
        name = f"wrfout_d01_2024-06-01_00_00_00"
        run = WORK / f"{tag}_{label}"
        run.mkdir(parents=True, exist_ok=True)
        wrfout = run / name
        write_wrfout(wrfout, grid, hgt)
        store = run / "store"; odir = run / "png"
        written, failures, skipped = rustwx.run_renderer(
            renderer, wrfout, store_root=store, out_dir=odir,
            products="terrain_height", frames="all", width=1500,
            height=1100,
            source_label=f"{tag} {label} (gpuwm static builder)")
        print(tag, label, "written:", written, "failures:", failures)
        for p in written:
            dest = OUT / f"renderer-{tag}-{label}.png"
            shutil.copy2(p, dest)
            print("  ->", dest)
