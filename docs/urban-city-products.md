# Urban surfaces in city forecasts

The single-layer urban canopy model runs with `sf_urban_physics = 1` and
Noah (`sf_surface_physics = 2`, `num_soil_layers = 4`). Its CUDA implementation
has a compiled C++ host twin that includes the same numerical source. The
default URBPARM equations are checked against the original WRF 4.6.1
Fortran. The optional impervious roof-water scheme retains three evaporation
depth corrections from WRF 4.7.1. See the column fixture provenance in
`tests/data/oracles/urban/ucm461/PROVENANCE.md` for those differences and the
undefined green-roof dew read.

High-resolution land cover retains urban types when a canopy is selected.
Annual NLCD maps developed classes to three URBPARM types with
`use_wudapt_lcz = 0`. CGLC-MODIS-LCZ retains the eleven LCZ types with
`use_wudapt_lcz = 1`. Under SLUCM, a retained urban legend also supplies `FRC_URB2D`:
Rust sums each mapped urban category's grid-cell area fraction times that
type's URBPARM built fraction. It keeps WRF's dominant urban mask. This is
an estimate from classified land cover and table parameters, not an
impervious-surface observation. The static receipt records its algorithm,
coefficients, counts and range.

The prepared SLUCM driver reads this plane from its static carrier.
An explicit fraction from a WRF input takes precedence. Static inputs
without the plane retain the existing table fallback. Disabled canopy
configurations do not enter this computation or allocate urban state.
BEP and BEP+BEM retain their existing fraction behavior.

The city recipe settings are:

| Key | Setting |
| --- | --- |
| `sf_urban_physics` | `1` |
| `sf_surface_physics` | `2` |
| `num_soil_layers` | `4` |
| `use_wudapt_lcz` | `0` for Annual NLCD, `1` for CGLC-MODIS-LCZ |
| `[static.highres] enabled` | `true` |
| `[static.highres] fields` | `auto` |
| `[static.highres] landcover_source` | `annual-nlcd` or `cglc-modis-lcz` |
| `[static.highres] on_refuse` | `error` |

The existing first-level geometry check still applies. An LCZ canopy must
fit below the first model level. Selecting LCZ therefore requires a vertical
grid that passes that check. RUC does not call an urban canopy model.

These are recipe settings, not global defaults. The HRRR configuration
keeps `sf_urban_physics = 0`. Column agreement establishes implementation
correctness; observation scores from a paired forecast establish its effect
on that case. Neither establishes broad forecast skill by itself.
