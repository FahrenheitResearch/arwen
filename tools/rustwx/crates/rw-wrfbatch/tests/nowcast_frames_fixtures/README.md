# Nowcast frames converter fixtures

`netcdf4-dimension-scale-nowcast.nc` (11 KB) is a NetCDF-4 (HDF5) file shaped
like a StormScope run: `refc[valid_time, member, y, x]` in dBZ with NaN as the
fill value, `valid_time` a dimension-scale coordinate carrying CF
`units = "hours since 2026-08-19 00:00:00"`, `lat`/`lon` cell centres (longitude
in [0, 360)), and the run's facts as global attributes (`init`, `causal`,
`seed`, `variant`, `precision`). Hour 0 is 20 dBZ above hour 1, and member 1
of hour 0 holds NaN at cell (0, 0). Values are `5 + 2 x` at hour 1.

It exists because the NetCDF index netcrust answers `variable()` from omits
HDF5 dimension-scale datasets, so a converter that reads the time
coordinate's units only through `variable()` refuses every real StormScope
file as having no time dimension. The classic-format fixtures the other
tests write cannot reproduce that.

Written on 2026-10-06 with netCDF4-python (NETCDF4 format, no compression).
