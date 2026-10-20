# Regular latitude-longitude pair sheets

`rw_compare --mode latlon-pair` reads two CF NetCDF fields through the native
Rust reader and renders A, B and A minus B through the production panel path.
A whole globe uses the existing Robinson projection. Both field panels share
one product ladder. The difference ladder has a neutral bin centred on zero;
its open ends and adaptive step are recorded in the JSON receipt.

```sh
rw_compare --mode latlon-pair --file analysis.nc --file reference.nc \
  --label A=Analysis --label B=Reference \
  --field z500 --field t850 --field t2m --field msl \
  --field u10m_v10m_speed --field tcwv \
  --region global --valid 2024-09-24T12:00Z --out-dir renders
```

The default variable table reads the Earth-2 vocabulary name in both files
(`z500`, `t850`, `t2m`, `msl`, `tcwv`; wind speed from `u10m` and `v10m`). A file
that prefixes or renames its variables passes `--variable-table TABLE.json`
(export the default with `--print-variable-table` and edit the names). Height is geopotential divided
by 9.80665 and then 10, pressure is Pa divided by 100, temperatures are K minus
273.15, and column water in kg/m2 is numerically mm. Temperature legends keep
their own product colours with levels expressed in Celsius. Wind is m/s.
Pressure uses the existing neutral generic ramp because the production
pressure product contours pressure and fills wind. Height uses the existing
500 hPa comparison ladder and palette.

Source names and conversions are table data. Export a table, edit its variable
names, affine conversions, product slugs or styles, then pass the JSON back:

```sh
rw_compare --mode latlon-pair --print-variable-table > variables.json
rw_compare --mode latlon-pair --variable-table variables.json \
  --file a.nc --file b.nc --field t2m --valid 2024-09-24T12:00Z --out-dir renders
```

Each row has `field`, `a`, `b`, `product_slug`, `title`, `display_units`, `style`
and `difference_step`. A plane has one scalar variable or two components, then
`factor` and `offset`. A production style names a canonical selector, stored
units and a conversion of its legend levels. Height ladders and fixed neutral
ranges are also available. Unknown fields, missing variables, duplicate output
slugs and incompatible dimensions fail with the offending name.

Inputs have regular monotonic 1-D `lat` and `lon` axes and 2-D variables in
`(lat,lon)` or `(lon,lat)` order. Latitude may ascend or descend. Longitudes may
use 0..360 or -180..180. The reader reorders axes and cells together, handles
CF fill values and packing, and keeps a duplicate cyclic endpoint once.
Two grids are joined on coincident coordinates within 0.00001 degree; no
interpolation is hidden in a score. A region selects cell centres and may
cross the date line: `--region -30,30,150,-150`.

The header and receipt report cosine-latitude weighted mean bias, RMSE and
spatial Pearson correlation over common finite cells in the selected region.
Correlation is undefined for constant fields. The receipt contains cell count,
weight sum, paired extrema, difference extrema and ladder step, per-stage wall
times, parallelism and deleted intermediate panel paths with their sizes.

`--threads` defaults to 12. Fields, paired input reads, cell conversions,
statistics chunks and production raster work share one Rayon pool. Finished
sheets are filed under `<out>/latlon-global|latlon-region/<product>/<day>/`.
Only final sheets and a small JSON receipt remain after successful rendering.
