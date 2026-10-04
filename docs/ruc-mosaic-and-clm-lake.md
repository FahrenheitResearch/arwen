# RUC mosaic and CLM lake

The experiment and WRF namelist doors accept RUC `mosaic_lu = 1`,
`mosaic_soil = 1`, and `sf_lake_physics = 1`. All three default to zero,
as in WRF. Selecting RUC does not turn the mosaic or lake model on.

```toml
[shared]
sf_surface_physics = 3
num_soil_layers = 9
mosaic_lu = 1
mosaic_soil = 1
sf_lake_physics = 1
use_lakedepth = 1
```

RUC mosaic follows WRF v4.6.1 `module_sf_ruclsm.F`: vegetation and soil
parameters are accumulated in source category order, the covered area is
capped at one, roughness uses the logarithmic blend at 5 m, and the soil
mixture excludes water category 14. An empty nonwater soil mixture uses
the dominant soil. The dominant vegetation still controls root depth and
forest class. These are mixed parameters for one prognostic soil column,
not separate prognostic land-use tiles.

The `LSMRUC` crop and natural-cover irrigation block runs after `SFCTMP`.
It adds moisture in root layers under WRF's greenness threshold. WRF does
not add that water to a separate irrigation budget accumulator, and the
port retains that behavior. `LANDUSEF` and `SOILCTOP` must contain the
source category fractions. A dominant-category map cannot reconstruct them.

The CLM lake port transcribes the complete WRF v4.6.1 `module_sf_lake.F`,
including ten water layers, ten sediment layers and up to five snow
layers. Internal calculations use double precision and persisted state
uses WRF's single precision. The lake call follows the land-surface call,
uses its accumulated precipitation before that buffer is cleared, and
replaces lake skin temperature, surface fluxes, albedo and 2 m diagnostics
before the PBL consumes them. RUC bypasses a lake column only while the
lake model is selected. With the lake model off, RUC uses its water arm.

The source-pinned Fortran harnesses and CUDA comparisons are in
`tools/ruc_mosaic_wrf461_oracle`, `tools/lake_wrf461_oracle`,
`tests/test_ruc_mosaic.py`, `tests/test_ruc_mosaic_gpu.py` and the lake
oracle tests. `tests/test_ruc_lake_runtime.py` covers coupled stepping,
restart and the actual resident-rank path. These are implementation
comparisons, not observations-based forecast skill or whole-forecast
identity with operational WRF.

Lake bathymetry follows WRF's defaults: `use_lakedepth = 1` requires the
`LAKE_DEPTH` input dataset. The static builder uses WPS v4.6.0's
`lake_depth/` dataset, `average_gcell(1.0)+search(5)`, land masking and
10 m missing-value fill. Fetch this optional dataset into the same
geography root as the land-use data:

```sh
gpuwm fetch-geog --datasets lake_depth --source ncar --root /path/to/WPS_GEOG
```

The existing `--datasets wrf` set is unchanged. The catalogue pins the
official NCAR archive and its extracted index; `--datasets all` includes it.
A supplied nonpositive depth falls back to
`lakedepth_default`, whose WRF default is 50 m. Explicit
`use_lakedepth = 0` uses that constant depth everywhere. Missing bathymetry
is not silently replaced with a constant. A nonpositive
`lakedepth_default` selects WRF's reference layer geometry depth.
Without a supplied lake mask, or with WRF's explicit `LAKEFLAG=0`,
WRF's elevation rule uses `lake_min_elev`, default 5 m.

Persistent lake state keeps horizontal grid axes so restart, tile and
resident-rank copies preserve the same cells. Only lake cells enter the
CUDA column solve. The dense persistent state costs 203 float32 words per
grid cell; sparse work is additional. Configuration-only memory estimates
bound the sparse work by the full grid and RUC fractions by the largest
supported category tables.

Lake domains must be fixed and present from the start. Moving and spawned
nests are refused before preparation because their land-state rebuild does
not yet transfer lake water, ice, snow and sediment heat storage.

The existing real-data land-use initializer derives sea ice from the
supplied ice fraction. It does not synthesize WRF's additional cold-water
skin-temperature trigger; the RUC fractions follow the same classification.
The separate stochastic RUC inputs and `flag_sm_adj` remain outside this
runtime change. WRF's existing defined-value treatment for uninitialized
thin-snow `ilnb` is unchanged.

WRF and WPS source attribution and notices are in `NOTICE`,
`licenses/LICENSE-WRF-public-domain.txt`, and beside the oracle sources.
The WPS download page points to the same public-domain notice:
https://www2.mmm.ucar.edu/wrf/site/wps_source.html.
