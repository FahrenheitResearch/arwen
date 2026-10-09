# Snowfall products

`rw_wrfbatch --list-products` lists `snowfall_window`, `snow_10to1_window`, and
`snow_kuchera_window`. They are ordinary named products included in `all` and
`direct`. They need no opt-in switch. Values are stored in mm of snow and the
map legend displays inches.

The default baseline is model initialization. Set `GPUWM_SNOW_SINCE` to an
exact history time in `YYYY-MM-DD_HH:MM:SS` form for a different window. Keep
all intervening history records in the current frame's directory. Both
colon and underscore timestamp filenames are accepted, and every time
record in a multi-record file is included. The integration uses the actual
available history cadence. Removing intermediate records changes the
temperature weighting, so retain the cadence used for a comparison.

`snowfall_window` reads the model's `SNOWFALLAC` snowfall depth. Its schema
defines mm even when the units attribute is blank. It does not substitute
snow depth, snow water, or a fixed density when this field is unavailable.
`snow_10to1_window` uses changes in `SNOWNC + GRAUPELNC` times 10.
`snow_kuchera_window` uses the same frozen liquid equivalent times the ratio
at each interval's end, then sums intervals in time order. Graupel is
included to match the supplied implementation. Missing `GRAUPELNC` is
allowed, but a present malformed graupel field is an error.

For the maximum native mass-level temperature `Tmax` in K from the lowest
model level through 500 hPa, inclusive, the ratio is:

```
Tmax > 271.16 K: ratio = max(0, 12 + 2 * (271.16 - Tmax))
otherwise:      ratio = max(0, 12 +     (271.16 - Tmax))
```

This is the formula in the [NOAA forecast operations guide](https://vlab.noaa.gov/web/forecast-guide/fog?page=numerical-methods-for-determining-snow-accumulation),
with the nonnegative floor used in the supplied source patch. The native
levels are not interpolated to create another sample at 500 hPa. Surface
`T2` is not substituted for a model-level temperature. This empirical
diagnostic does not model snow settling, melting, or crystal breakage.

The exact baseline must exist for a liquid-equivalent integration. A
different grid or run origin, duplicate valid time, or malformed source
plane refuses the product with a concrete availability note. Unknown
temperatures and accumulator resets produce missing cells rather than a
zero snowfall estimate. Neighboring history files and the chosen baseline
participate in the import cache identity.

## Point extraction

```
rw_points --point site,47.5,-111.5 --out points.csv wrfout...
```

Repeat `--point ID,LAT,LON` for more sites. Every record is extracted by
default; `--timeidx N` selects a record in each file. The CSV includes
geographic coordinates, record index, grid indices, nearest-cell distance
in km, raw surface fields, and `TMAXCOL`, `KUCH_SLR`, `SNOWLVL_FT`, `GUST10`.
The last two follow the source patch's wet-bulb-zero height and boundary
layer gust diagnostics. Their units are ft MSL and m/s; Tmax is K. Missing
data are blank. Raw wind components retain their native grid rotation.
Extraction uses great-circle distance, so longitude wrapping works.
All inputs are validated before the output file is created.
