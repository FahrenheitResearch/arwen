# Published static sources

A configuration can select a published static file through a row in
`gpuwm/data/static_sources/static-sources.v1.toml`. The row pins its URL,
SHA-256, size, grid, attributes, field inventory and field rename map.
Decoding, cropping and same-spacing fractional sampling run in Rust. Aligned
windows store exact float64 promotions of source float32 values. A row can
declare nearest sampling for categories and bilinear sampling for continuous,
monthly and fractional fields when the target starts between source cells.
Derived deep-soil temperature and projection fields retain their existing
definitions.

```toml
[static]
source = "hrrr-conus-v4"
```

The `hrrr`, `hrrr-prs` and `hrrr-native` configuration metadata selects this
row by default. A defaulted row is used only on its own projection. When a
configuration names no `[static] source` and its `[projection]` differs from
the row's (another `map_proj`, `truelat1`, `truelat2` or `stand_lon`), the
statics build from WPS_GEOG exactly as they did before the row existed. The
static receipt then carries `static_source_fallback` with the reason
"defaulted source did not match; WPS_GEOG used" and the differing values.
A source the configuration names itself is never set aside: a projection
that differs from the row's is refused, naming the row.

The domain planner aligns native-spacing configurations to source mass
points. A hand-written configuration at that spacing can keep a fractional
centre when its row declares field sampling. Its offset must remain
constant at all four corners, and every sampled point must remain inside
the source extent. The receipt records the fractional offset and each
field's method. File coordinates are checked on the original source lattice
before sampling. At other spacings, the existing WPS geography builder runs.

The full selection copies land use and soil fractions and categories,
terrain and slope category, monthly green fraction, LAI and albedo,
maximum snow albedo, soil temperature, lake depth and sub-grid drag fields.
The lake category comes from the source land-use legend. Large-scale drag
names are renamed from `CON`, `VAR`, `OA*` and `OL*` to their `LS` names;
the `SS` names remain unchanged. The pinned file carries equal large- and
small-scale sets. Runtime vegetation fraction comes from the analyzed
surface GRIB in percent, rather than substituting monthly green fraction.
Native acquisition selects that runtime record through source metadata.
If an index cannot prove safe subset coverage, the Rust fetcher takes the
full file and the Rust inventory reader extracts the original record by
its numeric selector. A selected field sharing an envelope with unrelated
fields is refused rather than appending those extra quantities.

For controlled comparisons, `source_fields` can select `soil`, `landuse`,
`terrain`, `vegetation`, `albedo`, `soil_temperature`, `lake_depth` or
`drag`. Its default is `all`.

```console
gpuwm fetch-geog --static-source hrrr-conus-v4 --root GEOG_ROOT
```

Files are staged at `GEOG_ROOT/static_sources/<id>/<filename>` and verified
against the pin on every resolution. `GPUWM_STATIC_SOURCE_ROOT` can name
a shared cache with the same `<id>/<filename>` layout. Ordinary geography
indexes remain required by preparation catalog checks. The publisher's
version folder can rotate; the table currently has no published mirror.
A retained local copy preserves the pin while a durable mirror is pending.

Configurations whose source metadata declares no static source, and
configurations whose defaulted row is on another projection, retain the
existing static builder and configuration output.

The WPS geography builder also accepts the `bnu_soil_30s` dataset token.
It selects the alternative 16-category top and bottom soil maps distributed
by NCAR, with 30 arc-second scalar categorical pixels. The existing Rust
category sampler builds the soil fractions and dominant categories.

```console
gpuwm fetch-geog --datasets bnu_soiltype_top,bnu_soiltype_bot --root GEOG_ROOT
```

Select it independently for each domain in `namelist.wps`:

```fortran
geog_data_res = 'bnu_soil_30s+default', 'bnu_soil_30s+default',
```

Token priority is field-specific. This selection changes only soil categories
and fractions; the other fields use the later token or the established
default. It does not override a published static source that supplies soil
on its own grid. The archive rows use the upstream NCAR source automatically
because the mirror does not carry these archives.

When high-resolution terrain and land cover are enabled, retain the selected
WPS soil maps with `soil_source = "wps-geog"` in `[static.highres]`.
The overlay skips SoilGrids downloads and reconciles soil water categories
with its land mask in Rust. Land cells with valid soil retain the source
fractions exactly; newly resolved land with water soil takes the nearest
valid land-soil cell. The omitted option keeps the established SoilGrids
overlay and its existing configuration bytes.
