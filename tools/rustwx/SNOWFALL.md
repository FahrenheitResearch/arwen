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
Only the RUC land model (`sf_surface_physics = 3`) writes `SNOWFALLAC`, as in
WRF, so a history from any other land model does not offer `snowfall_window`:
`--list-products` lists it as `missing-fields`, and `all` leaves it out without
a skip. The same holds for a product whose `SNOWNC` or `SNOWH` the history does
not store.
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

## Standing depth, water equivalent and ensemble probabilities

`snowfall_liquid_equivalent` integrates the saved `SNOWNC` plus optional
`GRAUPELNC` counter increments without an SLR multiplier. It stores mm of
water and displays inches of water. Counter resets are tested independently,
so a reset in one counter cannot be hidden by growth in the other.
`snow_depth_total` reads the land model's `SNOWH` in metres, converts to mm
and displays inches. `snow_depth_change` subtracts `SNOWH` at the exact
baseline on the same run/grid; it retains negative settling and melt on a
signed colour scale. All three are ordinary default `all` products when
the required snow fields and baseline exist.

`gpuwm enprod ENS_ROOT --field snow10_win --engine rust` and `snowkuchera_win` reach
`rw_ensbatch`. With no threshold override, snow probability products include
1, 4, 8 and 12 inches, expressed internally as 25.4, 101.6, 203.2 and
304.8 mm. `snow_depth` applies the same thresholds to standing depth.
Every probability retains the uncalibrated ensemble label and missing-member
policy. A single member is not a qualified ensemble probability forecast.

## Public analysis comparisons

`gpuwm render INPUT --compare nohrsc --products snow10,snowkuchera` uses the
NOHRSC national 24-hour snowfall analysis. The model window must be exactly
24 hours and end at the analysis valid time. An exact saved baseline is
required for another start. Analysis `Data:start_date`, `stop_date`, units,
coordinates and missing flags are checked, independent of filenames.
The analysis combines reported snowfall and a background estimate; it is
not an independent direct observation at every cell.

`gpuwm render INPUT --compare snodas --products snow_depth` uses the public
masked daily SNODAS archive. Standing depth is valid at 06 UTC, despite the
05 timestamp in the data member name. The reader checks the header's actual
start/stop time, quantity, units, dimensions and coordinate benchmark; it
reads signed big-endian 16-bit cells without unpacking archive paths. SNODAS
is a modeled analysis with assimilated observations, not fresh snowfall.
It cannot validate an SLR or a daily snowfall accumulation by depth alone.
Both references support the comparison door's offline and explicit-file
options, and receipts contain the source SHA-256 and missing coverage.

The NetCDF row requires the decoded `thickness_of_snowfall_amount` quantity,
metre units and exact start/end attributes. The archive row requires the decoded
snow-layer description, signed 16-bit representation, header dimensions and the
actual 06 UTC validity. Downloaded rasters are validated before cache publication;
a corrupt owned cache is reacquired once without replacing explicit input files.

Data attribution: National Operational Hydrologic Remote Sensing Center (2004),
Snow Data Assimilation System (SNODAS) Data Products at NSIDC, Version 1,
National Snow and Ice Data Center, https://doi.org/10.7265/N5TB14TC.
The proof uses the masked 2022-12-24 snow-depth analysis, accessed 2026-10-20.
NOAA/NWS public-domain data remain attributed to their source and do not imply
NOAA endorsement: https://www.weather.gov/disclaimer. The source distribution's
MIT notice remains in `LICENSE`; no external implementation was imported.
