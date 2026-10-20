# Terrain source handoff

LICENSE: NOAA-OWP/ngen `be11a180565a78ff069d855b0b5dbfc23845e7d8` and
evapotranspiration `6c410f2a8bc843043ad25e13239928031b8c5e64`: actual LICENSE
headings “Apache License”, “Version 2.0, January 2004”. Full notices retained.
No withdrawn WRF-Hydro code, inputs or interface is used.

## Availability at 2026-10-18

This lane has no actual Lower Colorado case(b) surface-excess/recharge
provider or matching time series. Hydrofabric has now supplied the final
north-first geometry and raw catchment-to-cell overlaps described below;
these still require a physical producer and an explicit overlap/source
partition policy. Existing
native forcing/PET resources cover case(a), catchments67/27/52, December2015.
They cannot supply the case(b) terrain run. Meteorological `RAINRATE` and PET
are inputs to land physics, not terrain surface-excess or aquifer recharge.

Lead's retained `merged-routing-16/prepare.py` describes case(b) captured
reach laterals in m3/s, starting 2023-04-02T00:00:00Z, 12 intervals of300s.
That routing-only composition does not run weather, PET, Noah or CFE. Reach
discharge cannot define a physical surface/recharge split or its raster
distribution. Its epoch/cadence are not a prepared land-source contract.

## Existing numerical owner and binding

The Noah owner publishes its current API in
`<scratch>/nextgen-noahmp-couple/CONTRACT.md`; driver assembly remains
owned by coupled-driver. The existing parent path is
`gpuwm.hydro.noahowp.factory:prepare_terrain_exchange`, followed by
`advance(dt_seconds, *, stream=None)`. It consumes the actual parent LSM
producer arrays and parent-to-terrain CSR; it needs no extra atmospheric
forcing/PET stage. This is a parent exchange, not an automatic rasterizer for
standalone catchment means. No duplicate exchange is added by forcing.

The owner contract requires `mapping_arrays` references for signed-i64
offsets, i32 indices and f32 weights, `destination_shape`, and
`parent_grid_sha256`, `terrain_grid_sha256`, `weights_sha256`, `fabric_sha256`.
It publishes stable same-card f32 C(ny,nx) `surface_source` and `recharge`
views in m/s. The actual LSM period is prepared once; the launch argument
must match it. Inputs, weights, producer state and all cross-step carriers
must restore together. This CSR differs from forcing's u64/u32/f64
meteorological aggregation contract.

Parent surface excess is `infxsrt`/`runsfxy` in mm per LSM step. Genuine bottom
drainage is `soldrain`, also mm per LSM step. The exchange converts weighted
depth to m/s using the prepared period. `subsurface_runoff_m_s` includes bottom
drainage: adding recharge to that total would count water twice. Terrain
recharge needs an explicit compatible aquifer budget. Water sent to terrain
must be removed from any parallel direct-channel runoff contribution.

For standalone Noah-OWP, the numerical owner's 2026-10-18 handoff identifies
`water.runsrf` and `water.qdrain` in mm/s as potential surface/recharge
terms, with a factor of .001 to m/s. These are not the parent's mm/step
fields. `QINSUR` in m/s is pre-infiltration soil surface input currently fed
to CFE, not terrain surface excess. Total `runsub` includes bottom drainage
and snow flow; it is not a second disjoint recharge source. No initialized
case(b) producer or numerical catchment-to-terrain binding is available
from that owner yet; this inventory does not create one.

## Exact target grid and remaining inputs

The terrain owner's actual manifest is
`<scratch>/hydro-overland/nextgen/lowercolorado-dem-01/manifest.json`.
It declares EPSG5070, shape[582,737], dx1000m, x0=-730254.9996999999m,
y0=612815.0003m at the SOUTHERN edge; storage is north-first C order, x east.
The finite-DEM active rectangle is not a catchment basin mask. An equal-shape
or nearest-centroid mapping does not supply that provenance.

Use the final north-first hydrofabric manifest, not the superseded
south-first weights:

`<scratch>/hydro-preprocess/nextgen/prepared/lowercolorado-north/6a8a6af4d40cf0bb6438/manifest.json`

Its SHA256 is
`919dba63435f7d9b6670cfb84c9f5284b59600b9db69efc47dac7e522212f886`.
It seals 6971 ordered catchments, the grid above, 180005 polygon/cell
intersections, actual flowline `channel_node` and `stream_order` rasters.
Native parser agreement reported by its owner does not establish the
Muskingum-Cunge upstream reduction or waterbody topology; its upstream order
is explicitly `flowpaths_sqlite_scan_order_UNVERIFIED_NATIVE_REDUCTION`.

Three mappings have different directions and normalization:

| Mapping | CSR rows and indices | Weights and intended use |
| --- | --- | --- |
| Meteorological aggregation | Catchment rows; u64 offsets, u32 source-grid indices | f64 catchment-mean area weights; forcing's prepared meteorological input |
| Parent LSM exchange | Destination rows; signed i64 offsets, i32 parent-grid indices | f32 weights; actual parent geometry required by the Noah factory |
| Catchment-to-terrain overlap | Terrain-cell rows; signed i64 offsets, i32 ordered-catchment indices | f64 intersection/full-cell-area weights, with explicit f32 aliases; conditional catchment-depth distribution |

The final fabric's `weights_*` rows aggregate grid values into catchments;
they cannot be passed unchanged to a cell-row catchment runoff consumer.
The `exchange_*` aliases likewise do not establish that this DEM grid is an
executed parent atmosphere grid.

Hydrofabric's separately sealed startup transpose is:

`<scratch>/hydro-preprocess/nextgen/cell-overlap/prepared/b52c05e04d48fd22f386/manifest.json`

SHA256 `8041e9a22ded38a7a88d0a5d05cf484c8858b6b44fb9b4bd694418faaa470bcf`.
It binds the final fabric seal, ordered catchment IDs and the terrain DEM
manifest SHA256
`ba1f68c2d4ac32c2d85f293a2b3f2b83611627a55ad1521b4458e278f9717347`.
The export contains `cell_catchment_offsets` i64[428935],
`cell_catchment_indices` i32[180005], and `cell_catchment_weights`
f64[180005] (plus f32 aliases), on the north-first C[582,737] destination.
Rows are ordered by catchment-axis index. Its Rust preparation API is
`nextgen_hydrofabric::cell_overlap::prepare`, exposed through
`gpuwm.hydrofabric.prepare_cell_overlap` and
`gpuwm hydro prepare --cell-overlap CONFIG.json`. This is geometry, with no
registered numerical source factory or new runtime Stage.

These are raw polygon intersections, not a disjoint basin mask. The owner
reports 100962 positive cells, 8035 cells with exact coverage above one
(including roundoff), and maximum coverage 2.0. Nothing is clipped or
renormalized. If a producer explicitly declares uniform depth over each
catchment, summing depth times these weights yields full-cell mean depth
and preserves the sum of covered polygon volumes, including overlaps. It
does not establish a unique physical land area or a disjoint water budget.
Do not silently normalize overlapping sources, fill uncovered cells, or
pass this transpose as an actual parent-grid mapping. Numerical/source
owners must declare the overlap policy, units, epoch, cadence, source-axis
order and removal from direct-channel runoff before using it physically.

Still needed for a physical run: an actual case(b) initialized parent or
standalone land model with declared physics/parameters, weather and time
origin; its runoff/recharge producer and prepared period; either the actual
parent-to-terrain mapping or an explicit uniform-catchment-depth producer
and overlap/partition policy for the supplied transpose; and terrain's
physical initial stores, aquifer parameters, receiving levels/outlets and
disjoint runoff budget. The standalone alternative needs a separately
declared numerical mapping owned by Noah/hydrofabric, not an inferred reuse
of the parent factory. No observed-flood, native event replay or terrain
speed result follows from the DEM, weather acquisition or captured laterals.
Case(b) input DATA terms remain unresolved; these manifests/payloads are
private owner artifacts, not bundled or cleared for public/default use.
The projected DEM weights also do not supply case(c)'s geographic AORC
aggregation: source latitude rows ascend south-to-north, and its CRS,
sampling convention, local subset and catchment-area CSR need separate
qualification by hydrofabric and referee.
