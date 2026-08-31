#!/usr/bin/env python3
"""Regenerate the NetCDF-4/HDF5 fixtures the rw-netcdf suite reads.

The classic-format fixtures are built at test runtime by the in-workspace
`netcdf-writer` crate, so they need no files here.  NetCDF-4 is different:
nothing in the Rust stack WRITES HDF5 (deliberately -- see
crates/netcdf-writer/src/lib.rs), and the code under test exists exactly
because real NetCDF-4 producers store coordinate variables as HDF5
dimension scales that the vendored reader's `variables()` omits.  So the
fixtures are produced by the reference implementations themselves --
netCDF4-python (the C library) and h5py -- and checked in as bytes.

axes.nc4 -- a well-formed NetCDF-4 file, the `recover_dimension_scales`
    subject:
    * dims time(4), lon(3), lat(3), bnds(2); lon and lat share a length
      on purpose, so only the coordinate-variable NAME rule can tell
      their scales apart;
    * coordinate variables time/lon/lat (HDF5 dimension scales, absent
      from the reader's variable table) with CF string attributes;
    * `bnds` has NO coordinate variable: the C library writes the
      "This is a netCDF dimension but not a netCDF variable." NAME
      sentinel, and recovering it would invent a variable netCDF4-python
      does not report;
    * a data variable t2(time, lat, lon) that carries a real
      DIMENSION_LIST, so strict metadata reconstruction succeeds;
    * an extra 1-D dimension scale `zvals` (length 6, added with h5py);
      the reader registers a 1-D scale as a dimension named after
      itself, so it resolves by the coordinate-variable NAME rule;
    * a 2-D dimension scale `corners` (5 x 6, added with h5py).  A 2-D
      dataset can never take the name rule, so each of its axes is
      resolvable only by the unique-length rule: 5 -> the scale's own
      registered dimension, 6 -> zvals.

sizes.nc4 -- a bare h5py file with no netCDF metadata at all: no
    DIMENSION_LIST anywhere.  Strict reconstruction must fail and the
    size-inferred fallback must open it, which is the `open()`
    provenance contract.

mixed.nc4 -- a netCDF4-python file whose dimensions row(3) and col(3)
    share a length, plus an h5py dataset `orphan` with no DIMENSION_LIST
    (the ERA5 `number`/`expver` shape).  Strict reconstruction fails on
    `orphan`, the fallback opens it, and the file's real dimension table
    survives into the fallback -- so `dimension_lengths_ambiguous` must
    come back true.

Run from this directory:  python make_fixtures.py
"""

from pathlib import Path

import h5py
import netCDF4
import numpy as np

HERE = Path(__file__).resolve().parent


def make_axes() -> None:
    path = HERE / "axes.nc4"
    path.unlink(missing_ok=True)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.title = "rw-netcdf axes fixture"
        ds.createDimension("time", 4)
        ds.createDimension("lon", 3)
        ds.createDimension("lat", 3)
        ds.createDimension("bnds", 2)

        time = ds.createVariable("time", "f8", ("time",))
        time.units = "hours since 2024-03-01"
        time.standard_name = "time"
        time.calendar = "standard"
        time.axis = "T"
        time[:] = [0.0, 1.0, 2.0, 3.0]

        lon = ds.createVariable("lon", "f8", ("lon",))
        lon.units = "degrees_east"
        lon.standard_name = "longitude"
        lon[:] = [10.0, 11.0, 12.0]

        lat = ds.createVariable("lat", "f8", ("lat",))
        lat.units = "degrees_north"
        lat.standard_name = "latitude"
        lat[:] = [40.0, 41.0, 42.0]

        t2 = ds.createVariable("t2", "f8", ("time", "lat", "lon"))
        t2.units = "K"
        t2[:] = np.arange(4 * 3 * 3, dtype="f8").reshape(4, 3, 3)

    # Scales netCDF4-python never writes, from the producers that do.
    # `zvals` (1-D) becomes a dimension named after itself and takes the
    # name rule; `corners` (2-D) can only take the unique-length rule.
    with h5py.File(path, "a") as f:
        zvals = f.create_dataset("zvals", data=np.array(
            [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]))
        zvals.attrs["units"] = np.bytes_("m")
        zvals.make_scale("zvals")
        corners = f.create_dataset(
            "corners", data=np.arange(30.0).reshape(5, 6))
        corners.attrs["long_name"] = np.bytes_("cell corner offsets")
        corners.make_scale("corners")
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def make_sizes() -> None:
    path = HERE / "sizes.nc4"
    path.unlink(missing_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("first", data=np.arange(6.0).reshape(2, 3))
        f.create_dataset("second", data=np.arange(3.0))
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def make_mixed() -> None:
    path = HERE / "mixed.nc4"
    path.unlink(missing_ok=True)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("row", 3)
        ds.createDimension("col", 3)
        grid = ds.createVariable("grid", "f8", ("row", "col"))
        grid[:] = np.arange(9.0).reshape(3, 3)
    with h5py.File(path, "a") as f:
        f.create_dataset("orphan", data=np.arange(3.0))
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    make_axes()
    make_sizes()
    make_mixed()
