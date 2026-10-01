#!/usr/bin/env bash
# Regenerates the wrfout time-axis fixtures beside this script.
#
# Needs python3 with netCDF4 and NCO's ncks on PATH (conda-forge nco 5.4.0
# wrote the committed copies).  Each fixture is a tiny two-by-two grid cut
# by ncks out of a two-record source whose records are valid at
# 2025-03-15 23:00Z and 2025-03-16 00:00Z and whose START_DATE is
# 2025-03-15 12:00Z, so the cut record is hour 12 and a reader that falls
# back to the start time is visibly wrong.
#
# Every ncks call passes -h: without it ncks appends its command line, with
# the absolute output path, to the global history attribute, and 2.8.0
# shipped a home folder inside these three files that way.  The committed
# copies had that attribute deleted through the NetCDF library and were
# rewritten by nccopy (netCDF 4.9.3); every variable is unchanged.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

python3 - "$work" <<'PY'
import sys
import numpy as np
import netCDF4

work = sys.argv[1]
labels = ["2025-03-15_23:00:00", "2025-03-16_00:00:00"]


def base(path, fmt, string_times):
    ds = netCDF4.Dataset(path, "w", format=fmt)
    ds.START_DATE = "2025-03-15_12:00:00"
    ds.SIMULATION_START_DATE = "2025-03-15_12:00:00"
    ds.MAP_PROJ = np.int32(1)
    ds.TRUELAT1 = np.float32(38.5)
    ds.TRUELAT2 = np.float32(38.5)
    ds.STAND_LON = np.float32(-97.5)
    ds.CEN_LAT = np.float32(38.5)
    ds.CEN_LON = np.float32(-97.5)
    ds.DX = np.float32(3000.0)
    ds.DY = np.float32(3000.0)
    ds.createDimension("Time", None)
    ds.createDimension("DateStrLen", 19)
    ds.createDimension("south_north", 2)
    ds.createDimension("west_east", 2)
    ds.createDimension("bottom_top", 1)
    if string_times:
        times = ds.createVariable("Times", str, ("Time",))
        for index, label in enumerate(labels):
            times[index] = label
    else:
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[:] = np.array([list(label.encode("ascii")) for label in labels], dtype="u1").view("S1")
    lat = ds.createVariable("XLAT", "f4", ("Time", "south_north", "west_east"))
    lon = ds.createVariable("XLONG", "f4", ("Time", "south_north", "west_east"))
    t2 = ds.createVariable("T2", "f4", ("Time", "south_north", "west_east"))
    t2.units = "K"
    # A four-dimensional T is what the raw-wrfout reader keys on, so the
    # NetCDF-4 cuts exercise both import routes.
    theta = ds.createVariable(
        "T", "f4", ("Time", "bottom_top", "south_north", "west_east")
    )
    theta.units = "K"
    for record in range(2):
        lat[record] = [[38.0, 38.0], [38.03, 38.03]]
        lon[record] = [[-97.5, -97.47], [-97.5, -97.47]]
        t2[record] = [[290.0 + record, 291.0], [292.0, 293.0]]
        theta[record] = [[[1.0, 2.0], [3.0, 4.0]]]
    ds.close()


base(f"{work}/char_source.nc", "NETCDF3_64BIT_OFFSET", False)
base(f"{work}/string_source.nc", "NETCDF4", True)
PY

# A cut that drops Times (and T, so only the generic reader takes it): the
# valid time survives only in the file name.
ncks -O -h -6 -d Time,1 -x -v Times,T "$work/char_source.nc" "$here/no_times_cut.nc"
# The same record cut to NetCDF-4: ncks keeps Times as char[Time, DateStrLen].
ncks -O -h -4 -d Time,1 "$work/char_source.nc" "$here/times_char_netcdf4_cut.nc"
# A NetCDF-4 source whose Times is NC_STRING[Time], cut by ncks.
ncks -O -h -4 -d Time,1 "$work/string_source.nc" "$here/times_string_netcdf4_cut.nc"
