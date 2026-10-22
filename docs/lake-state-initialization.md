# Lake initialization sources

`RunConfig.lake_state_source` defaults to `auto`. Before the land-surface cold
start, public NOAA GLSEA temperature and ice concentration replace valid lake
cells within the source raster. Other cells retain their forcing. The table
`gpuwm/data/surface_obs/lake-state-sources.v1.toml` owns identifiers, dates,
variables, units, conversions, URLs, coverage, age limits and licences. Additional
regular CF sources are table rows. Downloads are cropped and cached; no
credentials are discovered. GLSEA is free to use and redistribute under its
NOAA metadata terms.

Use these shared experiment settings for the coupled lake model:

```toml
[shared]
sf_lake_physics = 1
lake_state_source = "auto"
```

The ERA5-Land row accepts an acquired CF NetCDF file (CC-BY-4.0; the receipt
carries its citation):

```toml
[shared]
sf_lake_physics = 1
lake_state_source = "era5-land"
lake_state_path = "/inputs/lake-fields.nc"
```

The file contains `valid_time`, `latitude`, `longitude`, and the fields `lmlt`,
`lblt`, `lmld`, `lshf`, `licd`, `lict` on those axes, in K, K, m, 1, m, K.
Longitude may be in either conventional encoding. The latest analysis at or
before initialization is selected; future analyses and stale files are refused.
An explicit source refuses missing fields, bad units and invalid physical
values. Automatic acquisition reports partial coverage and errors without
replacing absent values with zero. Each field has a file hash, selected time,
age, URL and coverage count in the initialization receipt.

Public ARCO ERA5 atmospheric forcing is available without a CDS key through
`gpuwm fetch --source era5 --era5-provider arco`. ERA5 is a separate product;
it is not evidence that ERA5-Land profile fields have been acquired. The local
ERA5-Land door does not download authenticated CDS files.

Rust reads CF masks/packing, samples finite local bilinear donors and converts
to SI. It does not extrapolate beyond the raster. CUDA binds TSK/SST and the
horizontal ice reservoirs before LSM initialization. With `sf_lake_physics=1`,
it also binds the sparse CLM columns after lakeini:

* GLSEA constrains the surface. Deeper temperature and unobserved ice thickness
  retain lakeini priors. Open-water concentration clears column ice. A GLSEA
  ice fraction alone cannot supply an observed thickness; the receipt counts
  those icy cells explicitly.
* ERA5-Land mixed-layer temperature, bottom temperature, depth and integral
  shape factor reconstruct a power profile. Below the mixed layer, normalized
  depth `r` gives `T = Tmixed + (Tbottom - Tmixed) r^(1/shape - 1)`. Its depth
  mean shape is the supplied factor. This is an explicit transfer between
  lake-model representations, not a measured vertical temperature profile.
  A mixed layer deeper than model bathymetry yields an isothermal column.
* Positive ERA5-Land ice thickness identifies an ice-covered column. When a
  concentration is supplied, thickness times concentration is conserved as
  equivalent ice volume distributed from the top downward. Equivalent ice
  thicker than model bathymetry is refused before binding. Ice temperature
  caps frozen layers at melting. Sediment temperatures remain lakeini priors.

Existing serialized `lake_columns`, `lake_static`, `lake_latitude` and surface
fields retain prognostic state across restart. Source acquisition runs only on
cold start; provenance is run-local metadata and never overwrites a restart.

`GPUWM_LAKE_STATE_CACHE` sets the download cache. `GPUWM_LAKE_RECEIPT_DIR` sets
the JSON receipt directory. The native preprocess library must be rebuilt from
`tools/grib1_bridge/Cargo.toml`; there is no Python decoder fallback.

Verification helpers decode NDBC WTMP and missing sentinels in Rust, select a
bounded time window, match only nearby finite wet model cells and reduce paired
errors in parallel. Missing buoy reports stay unavailable. GLSEA initialization
agreement measures binding closure, not independent forecast skill. Buoy bulk
water temperatures and model skin temperatures are distinct measurements.
`write_diagnostic_cf` masks land and converts units in Rust, then writes through
the existing Rust NetCDF seam. Render its products using `gpuwm.rustwx` and the
production `rw_wrfbatch`; no alternate weather renderer is used.

The CF renderer table includes lake temperature in degF, lake ice fraction,
accumulated snowfall water equivalent in inches and snow depth in inches.
SNOWNC is water equivalent, not snowfall depth; SNOWH is the current snowpack
depth, not a storm accumulation. The event proof keeps those products separate.
The native renderer can load the checked-in CF table with `--cf-table` when
using an older prebuilt renderer executable.

Regular longitude axes may ascend or descend and use either conventional longitude encoding. Sampling chooses an equivalent coordinate within the actual source extent. In particular, a descending axis crossing zero does not imply a 0-to-360 encoding. No donor is extrapolated beyond a regional source boundary.

The source table may declare `zero_roundoff_tolerance` in SI units. Rust snaps only values whose absolute magnitude is within that tolerance to zero. ERA5-Land ice depth uses 1e-12 m because converted GRIB zero depth can be -2.1684e-19 m. The receipt records the tolerance; material negative depths still fail admission.
