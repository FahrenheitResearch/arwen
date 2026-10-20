"""Render one field of an Earth2Studio NetCDF (time, lead_time, lat, lon) with the Rust renderer.

rw_wrfbatch reads WRF-shaped history. The field's exact float32 values and
its latitude/longitude are written into a WRF-shaped cylindrical-equidistant
view (MAP_PROJ 6) and drawn with the real renderer through gpuwm.rustwx.
    python render_latlon.py FILE.nc VAR LEAD_HOURS OUT_DIR LABEL
VAR t2m -> product 2m_temperature; u10m/v10m -> 10m_wind_speed_and_direction;
anything else -> var:<VAR upper>.
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from gpuwm.rustwx import find_renderer, run_renderer

path, var, lead_h, out, label = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
out.mkdir(parents=True, exist_ok=True)
ds = xr.open_dataset(path)
lead = np.timedelta64(lead_h, "h")
tix = int(sys.argv[6]) if len(sys.argv) > 6 else 0
sel = dict(time=tix)
lt = ds["lead_time"].values
i = int(np.nonzero(lt == lead)[0][0]) if "lead_time" in ds.dims else None
if i is not None:
    sel["lead_time"] = i


def plane(name):
    return ds[name].isel(**sel).values.astype(np.float32)


lat = ds["lat"].values
lon = ds["lon"].values
lon = np.where(lon > 180, lon - 360, lon)
order = np.argsort(lat)
lon2, lat2 = np.meshgrid(lon, lat[order])
init = ds["time"].values[tix]
valid = np.datetime_as_string(init + (lt[i] if i is not None else np.timedelta64(0, "h")), unit="s").replace("T", "_")
start = np.datetime_as_string(init, unit="s").replace("T", "_")
dims = ("Time", "south_north", "west_east")
fields = {}
if var == "t2m":
    fields["T2"] = (plane("t2m")[order], "K")
    product = "2m_temperature"
elif var in ("u10m", "v10m", "ws10m"):
    fields["U10"] = (plane("u10m")[order], "m s-1")
    fields["V10"] = (plane("v10m")[order], "m s-1")
    product = "10m_wind_speed_and_direction"
else:
    fields[var.upper()] = (plane(var)[order], "m s-1")
    product = "var:wrf_" + var.lower()
data = {k: (dims, v[None], {"units": u}) for k, (v, u) in fields.items()}
data["XLAT"] = (dims, lat2[None].astype(np.float32))
data["XLONG"] = (dims, lon2[None].astype(np.float32))
data["Times"] = (("Time",), np.array([valid], dtype="S19"))
dx = float(abs(lon[1] - lon[0]) * 111194.9)
view = xr.Dataset(data, attrs={
    "TITLE": "view of an Earth2Studio tensor for the Rust renderer", "MAP_PROJ": 6,
    "MAP_PROJ_CHAR": "Cylindrical Equidistant", "START_DATE": start,
    "SIMULATION_START_DATE": start, "CEN_LAT": float(lat.mean()), "CEN_LON": float(lon.mean()),
    "STAND_LON": 0.0, "TRUELAT1": 0.0, "TRUELAT2": 0.0, "POLE_LAT": 90.0, "POLE_LON": 0.0,
    "DX": dx, "DY": dx})
vpath = out / f"view-{var}-f{lead_h:03d}.nc"
view.to_netcdf(vpath, format="NETCDF3_64BIT")
written, failures, skipped = run_renderer(find_renderer(), vpath, store_root=out / "store",
                                          out_dir=out, products=product, frames="all",
                                          width=1100, height=700, source_label=label)
print("written", [str(p) for p in written], "failures", failures)
