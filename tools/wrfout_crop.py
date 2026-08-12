"""Cut a rectangular sub-window out of a wrfout, as its own wrfout.

A close-up of one storm is NOT a different forecast and must not be produced
by a different code path.  This copies bytes: every variable is sliced on the
``west_east``/``south_north`` axes (and their staggered partners), every
attribute is carried over, and nothing is interpolated, smoothed or
recomputed.  ``gpuwm render`` then renders the crop exactly as it renders the
parent, so the two figures are the same renderer on the same numbers.

    python -m tools.wrfout_crop SRC DST --i0 480 --i1 800 --j0 320 --j1 640

Indices are mass-point, ``[i0, i1)`` / ``[j0, j1)``.  ``--lon/--lat/--half-km``
picks the window from the file's own XLONG/XLAT instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def crop(src: Path, dst: Path, i0: int, i1: int, j0: int, j1: int) -> Path:
    import netCDF4

    nx, ny = i1 - i0, j1 - j0
    with netCDF4.Dataset(src) as a:
        dims = {name: len(dim) for name, dim in a.dimensions.items()}
        new_dims = dict(dims)
        for name, size in dims.items():
            if name == "west_east":
                new_dims[name] = nx
            elif name == "south_north":
                new_dims[name] = ny
            elif name == "west_east_stag":
                new_dims[name] = nx + 1
            elif name == "south_north_stag":
                new_dims[name] = ny + 1
        tmp = dst.with_name("." + dst.name + ".tmp")
        with netCDF4.Dataset(tmp, "w", format="NETCDF4_CLASSIC") as b:
            for name, size in new_dims.items():
                b.createDimension(name,
                                  None if a.dimensions[name].isunlimited()
                                  else size)
            attrs = {k: a.getncattr(k) for k in a.ncattrs()}
            attrs["WEST-EAST_GRID_DIMENSION"] = np.int32(nx + 1)
            attrs["SOUTH-NORTH_GRID_DIMENSION"] = np.int32(ny + 1)
            attrs["GPUWM_CROP_OF"] = src.name
            attrs["GPUWM_CROP_INDICES"] = f"i[{i0}:{i1}] j[{j0}:{j1}]"
            b.setncatts(attrs)
            for name, var in a.variables.items():
                out = b.createVariable(name, var.datatype, var.dimensions)
                out.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                sl = []
                for dim in var.dimensions:
                    if dim == "west_east":
                        sl.append(slice(i0, i1))
                    elif dim == "west_east_stag":
                        sl.append(slice(i0, i1 + 1))
                    elif dim == "south_north":
                        sl.append(slice(j0, j1))
                    elif dim == "south_north_stag":
                        sl.append(slice(j0, j1 + 1))
                    else:
                        sl.append(slice(None))
                out[...] = var[tuple(sl)]
        tmp.replace(dst)
    return dst


def window_from_lonlat(src: Path, lon: float, lat: float, half_km: float,
                       dx_km: float) -> tuple[int, int, int, int]:
    import netCDF4

    with netCDF4.Dataset(src) as a:
        xlat = np.asarray(a.variables["XLAT"][0])
        xlong = np.asarray(a.variables["XLONG"][0])
    d = (xlat - lat) ** 2 + ((xlong - lon)
                             * np.cos(np.deg2rad(lat))) ** 2
    j, i = np.unravel_index(int(np.argmin(d)), d.shape)
    half = int(round(half_km / dx_km))
    ny, nx = xlat.shape
    i0 = max(0, min(i - half, nx - 2 * half))
    j0 = max(0, min(j - half, ny - 2 * half))
    return i0, min(nx, i0 + 2 * half), j0, min(ny, j0 + 2 * half)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--i0", type=int)
    ap.add_argument("--i1", type=int)
    ap.add_argument("--j0", type=int)
    ap.add_argument("--j1", type=int)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--lat", type=float)
    ap.add_argument("--half-km", type=float, default=225.0)
    ap.add_argument("--dx-km", type=float, default=3.0)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if args.lon is not None:
        i0, i1, j0, j1 = window_from_lonlat(src, args.lon, args.lat,
                                            args.half_km, args.dx_km)
    else:
        i0, i1, j0, j1 = args.i0, args.i1, args.j0, args.j1
    dst.parent.mkdir(parents=True, exist_ok=True)
    crop(src, dst, i0, i1, j0, j1)
    print(f"cropped i[{i0}:{i1}] j[{j0}:{j1}] -> {dst} "
          f"({dst.stat().st_size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
