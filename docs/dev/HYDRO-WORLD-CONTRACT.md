# HYDRO-WORLD fabric contract (`arwen.hydro-world.fabric.v1`)

Owner: lane hydro-world-fabric. Code: `tools/hydro_world/` (Rust crate `hydro-world`, CLI `rw_hydro_world`).
Ruling: Drew 2026-10-18 08:40Z "hydro is default on", 08:50Z "just put opus lanes and lets make this worldwide".

This is the one fabric shape every hydro fabric builder writes: the CONUS NHDPlus recipe (owned by
astra-c-hydro-data) and the global builder (owned by hydro-world-fabric: the open Copernicus GLO-90
row, the worldwide default, and the opt-in HydroSHEDS v1 cross-check row). It is the NextGen
hydrofabric v2.0.1 GeoPackage the engine already reads, pinned column by column, plus two tables the
runoff modules and the audit trail need. A fabric that passes `rw_hydro_world check` goes through the
engine's existing preparers unchanged, and two builders that describe the same rivers produce the same
prepared bytes.

## 1. What the engine consumes, end to end

| Contract part | Engine consumer (already in lane/nextgen-integrate) | Engine output |
|---|---|---|
| divides, flowpaths, nexus, network, flowpath_attributes, hydrolocations, lakes, model_attributes | `gpuwm hydro prepare CONFIG.json` = `rw_hydrofabric prepare` (`tools/nextgen_hydrofabric`) | `arwen.nextgen.hydrofabric.v1` manifest: reach graph, level orders, MC parameters `f32[N,8]`, parent-grid weights, Noah exchange CSR, channel raster, `network.json`, `attributes.json`, `gauges.json`; loaded once by `gpuwm.hydrofabric.load_cuda` |
| model_attributes + initial storage | `hydro_world::derive::cfe_ini` -> request `arwen.cfe.prepare/v1` -> `nextgen-cfe-prepare` | `cfe.initial.pack/v1` for `gpuwm.hydro.cfe:prepare` |
| model_attributes | `hydro_world::derive::noahowp_overrides` -> `arwen-noahowp-init` request `overrides` | Noah-OWP resident slabs for `gpuwm.hydro.noahowp.factory:prepare_slab` |
| lakes | today: `nextgen-reservoir-prepare` reads native t-route captures only | named gap 13.3: a Rust packer from the `lakes` table |

The engine reads tables by role. `rw_hydrofabric` matches a role to a table named exactly the role (a `-`
in a table name is read as `_`). It requires divides, flowpaths and nexus, and reads the others when
present. `fabric_metadata` is not read by the engine.

## 2. Proof that the engine cannot tell builders apart

`tools/hydro_world/tests/roundtrip.rs` reads two real CONUS fabrics into the contract structs, writes
them back with the contract DDL, and runs the engine's own `rw_hydrofabric prepare` on the source and on
the rewritten file:

| Fixture | Contract round trip | Engine prepare, source vs contract-written |
|---|---|---|
| case (a): ngen be11a180 `data/gauge_01073000/gauge_01073000.gpkg` (5 reaches) | lossless (struct equality, every f64 bit and geometry byte) | 10 of 10 payloads identical; network.json, parsed_network.tsv, gauges.json, attributes.json identical |
| case (b): t-route `LowerColorado_NGEN_v201.gpkg` (6971 reaches, 20 lakes) | lossless | without grid 10 of 10, with the case (b) 737 x 582 EPSG:5070 grid 21 of 21 payloads identical (weights, overlaps, channel raster included) and all 21 SHA-256 equal to the landed case (b) manifest; network.json, parsed_network.tsv, gauges.json identical; attributes.json differs only because the source declares `mainstem` REAL (see 6) |

So the contract is exactly what the engine reads, and the only engine-visible degree of freedom left to a
builder is the JSON type of a column value, which the contract pins.

## 3. Container

- GeoPackage 1.2: `PRAGMA application_id=1196444487` ('GPKG'), `PRAGMA user_version=10200`, the three
  core tables `gpkg_spatial_ref_sys`, `gpkg_contents`, `gpkg_geometry_columns` with the spec DDL.
- `gpkg_spatial_ref_sys` holds the spec rows -1, 0, 4326 and the one fabric CRS.
- Every contract table is created with the DDL in section 7, verbatim, in that column order. The writer
  (`hydro_world::gpkg::write`) is the only sanctioned way to produce the container.
- No rtree index and no GDAL triggers: they need GDAL's `ST_*` SQL functions, and the engine reader does
  not use them. Native ngen reading a contract-written file is checked in 13.1.
- Tables outside the contract are not written. Readers ignore them (`layer_styles` in NOAA files).

## 4. CRS, geometry and units

- One horizontal CRS for every feature table, projected, in metres, equal-area. CONUS: EPSG:5070 (as the
  NOAA fabrics). Global rows: Lambert Azimuthal Equal Area on WGS84 centred on the domain centroid,
  written as a `gpkg_spatial_ref_sys` row with `organization` `ARWEN`, `srs_id` = `organization_coordsys_id`
  >= 100000 and a WKT2 `definition` (the engine reader uses the definition when the organization is not
  EPSG; a custom-WKT end-to-end run is open item 13.2).
- Geometry: GeoPackage binary (spec 2.1.3), version 0, little-endian header, standard (not extended)
  geometry, header `srs_id` = the fabric CRS, 2D little-endian WKB. Envelope: XY for POLYGON and
  LINESTRING, none for POINT. Polygons closed, exterior ring counter-clockwise, holes clockwise. Flowpath
  vertices run upstream to downstream.
- Areas (`areasqkm`, `tot_drainage_areasqkm`, `LkArea`) are planar areas in the equal-area fabric CRS, in
  km2. Lengths (`lengthkm`, `length_m`) are geodesic on the WGS84 ellipsoid.
- Elevations (`LkMxE`, `WeirE`, `OrificeE`, `elevation_mean_m`) use the row's `vertical_datum`
  (NAVD88 CONUS, EGM2008 global, the Copernicus DEM datum), recorded in `fabric_metadata`.

## 5. ID scheme

| Feature | ID | Rule |
|---|---|---|
| flowpath | `wb-<n>` | `n` unique in the fabric |
| divide | `cat-<n>` | same `n` as its flowpath; `divides.id` = `wb-<n>`, `flowpaths.divide_id` = `cat-<n>` |
| nexus | `nex-<m>` | `m` = `n` of the flowpath the nexus drains into (`nexus.toid` = `wb-<m>`) |
| terminal nexus | `tnx-<k>` | `toid` = `wb-0` (the v2.0.1 sentinel; the engine resolves it to downstream -1) |
| lake | `lakes.id` = outlet `nex-<m>`, `lakes.toid` = `wb-<m>`, `hl_link` = waterbody id, `hl_uri` = `WBOut-<hl_link>` | `hl_link` unique in the fabric |
| gauge | hydrolocation `hl_reference` `Gages`; `hl_uri` = `Gages-<station>` for USGS (v2.0.1 form, what `gauges.json` and the NWIS door read), `Gages-<AGENCY>:<station>` for every other agency (`AGENCY` from the source row's gauge feed: NRFA, PEGEL, BOM, ...) | the gauge lane owns the station tables |

`n`, `m`, `k` and `hl_link` are positive decimal integers without leading zeros and below 2^53 (exact in
JSON and in the engine's i64 `node_ids`). Numbering: `n = id_namespace * 10^10 + local_id`, where
`id_namespace` comes from the source row (CONUS NHDPlus = 0, so `n` is the NHDPlus COMID, as
astra-c-hydro-data proposed; HydroSHEDS v1 = 1, so `n = 10^10 + local reach id`; Copernicus GLO-90
D8 = 2, so `n = 2*10^10 + local reach id`). Namespaces never collide,
so a domain that crosses the US border can hold CONUS and global reaches in one fabric without renaming.
A builder that splits a source reach assigns new local ids inside its namespace and records the source id
in `network.hf_id`.

## 6. Canonical order and values

- Row order is part of the bytes the engine sees (node order, upstream reduction order, attributes.json).
  Builders call `hydro_world::derive::canonicalize`: every table sorted by (prefix rank wb < cat < nex <
  tnx < cnx, numeric suffix), network additionally by `hf_id`, `toid`; `fid` = rank + 1.
- Declared types are the section 7 DDL; values are stored in that storage class: MEDIUMINT/INTEGER
  columns hold integers, REAL columns hold finite doubles, BOOLEAN holds 0/1, TEXT holds UTF-8. NULL only
  where section 7 says nullable. `-0.0` is written as `0.0`.
- Pre-contract NOAA files deviate in two places, both lossless and reported by `rw_hydro_world check`:
  case (b) declares `flowpaths.mainstem` and `network.mainstem` REAL (case (a) declares MEDIUMINT, as the
  contract), and declares `lakes.geom` POINT (contract GEOMETRY, as case (a)).
- `hydroseq`: builders write a topological rank strictly larger downstream. That is the majority v2.0.1
  convention; NOAA files break it (73 of 6970 edges in case (b)), the engine does not use it.

## 7. Tables

Generated from `rw_hydro_world schema` (the code is the source of truth). Units: `-` dimensionless,
`id` identifier, `enum` coded value.

### divides

data_type features; geometry POLYGON; engine role divides; required in a builder fabric: yes.

| Column | Declared type | Unit | Nullable | Meaning |
|---|---|---|---|---|
| `fid` | INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL | id | no | 1-based row key; equals canonical row rank |
| `geom` | POLYGON | m (fabric CRS) | no | catchment polygon, GPKG binary, fabric CRS |
| `divide_id` | TEXT | id | no | cat-&lt;n&gt; |
| `toid` | TEXT | id | no | nexus receiving this catchment's water: nex-&lt;m&gt; or tnx-&lt;k&gt; |
| `type` | TEXT | enum | no | network | terminal (v2.0.1 values) |
| `ds_id` | REAL | id | yes | reserved by v2.0.1; NULL |
| `areasqkm` | REAL | km2 | no | planar area in the equal-area fabric CRS |
| `id` | TEXT | id | yes | wb-&lt;n&gt; of the paired flowpath; NULL if has_flowline=0 |
| `lengthkm` | REAL | km | yes | length of the paired flowpath |
| `tot_drainage_areasqkm` | REAL | km2 | no | total upstream area including this catchment |
| `has_flowline` | BOOLEAN | - | no | 1 when a flowpath drains this catchment |

### flowpaths

data_type features; geometry LINESTRING; engine role flowpaths; required in a builder fabric: yes.

| Column | Declared type | Unit | Nullable | Meaning |
|---|---|---|---|---|
| `fid` | INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL | id | no | 1-based row key; equals canonical row rank |
| `geom` | LINESTRING | m (fabric CRS) | no | reach line, upstream to downstream vertex order |
| `id` | TEXT | id | no | wb-&lt;n&gt; |
| `toid` | TEXT | id | no | downstream nexus nex-&lt;m&gt; or terminal tnx-&lt;k&gt; |
| `mainstem` | MEDIUMINT | id | yes | level-path identifier shared along a main stem |
| `order` | REAL | - | no | Strahler stream order |
| `hydroseq` | MEDIUMINT | - | no | hydrologic sequence; builders write a topological rank strictly larger downstream |
| `lengthkm` | REAL | km | no | reach length (geodesic) |
| `areasqkm` | REAL | km2 | no | local catchment area |
| `tot_drainage_areasqkm` | REAL | km2 | no | total upstream area |
| `has_divide` | BOOLEAN | - | no | 1 when cat-&lt;n&gt; exists |
| `divide_id` | TEXT | id | yes | cat-&lt;n&gt; or NULL |

### nexus

data_type features; geometry POINT; engine role nexus; required in a builder fabric: yes.

| Column | Declared type | Unit | Nullable | Meaning |
|---|---|---|---|---|
| `fid` | INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL | id | no | 1-based row key; equals canonical row rank |
| `geom` | POINT | m (fabric CRS) | no | confluence / outlet point |
| `id` | TEXT | id | no | nex-&lt;m&gt; (m = n of the receiving flowpath) or tnx-&lt;k&gt; |
| `toid` | TEXT | id | no | wb-&lt;m&gt;; terminal tnx- uses the v2.0.1 sentinel wb-0 |
| `hl_id` | TEXT | id | yes | hydrolocation id at this nexus |
| `hl_uri` | TEXT | id | no | &lt;hl_reference&gt;-&lt;hl_link&gt;, or NA |
| `type` | TEXT | enum | no | nexus | poi (poi = carries a hydrolocation; v2.0.1 values) |

### network

data_type attributes; geometry none; engine role network; required in a builder fabric: yes.

| Column | Declared type | Unit | Nullable | Meaning |
|---|---|---|---|---|
| `fid` | INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL | id | no | 1-based row key; equals canonical row rank |
| `id` | TEXT | id | no | wb-&lt;n&gt;, nex-&lt;m&gt; or tnx-&lt;k&gt;; a flowpath repeats once per source member reach |
| `toid` | TEXT | id | yes | downstream feature id |
| `divide_id` | TEXT | id | yes | cat-&lt;n&gt; draining to this feature |
| `ds_id` | REAL | id | yes | reserved by v2.0.1; NULL |
| `mainstem` | MEDIUMINT | id | yes | level-path identifier |
| `hl_id` | TEXT | id | yes | hydrolocation id |
| `hydroseq` | MEDIUMINT | - | yes | hydrologic sequence |
| `hl_uri` | TEXT | id | yes | Gages-&lt;station&gt; links a gauge (engine gauges.json) |
| `hf_source` | TEXT | enum | yes | source network name (source table hf_source) |
| `hf_id` | REAL | id | yes | native id of the source reach |
| `lengthkm` | REAL | km | yes | reach length |
| `areasqkm` | REAL | km2 | yes | local catchment area |
| `tot_drainage_areasqkm` | REAL | km2 | yes | total upstream area |
| `type` | TEXT | enum | yes | nexus | terminal | coastal | network |
| `vpu` | TEXT | id | yes | processing region code |

### flowpath_attributes

data_type attributes; geometry none; engine role flowpath_attributes; required in a builder fabric: yes.

| Column | Declared type | Unit | Nullable | Meaning |
|---|---|---|---|---|
| `fid` | INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL | id | no | 1-based row key; equals canonical row rank |
| `id` | TEXT | id | no | wb-&lt;n&gt;; exactly one row per flowpath |
| `rl_gages` | TEXT | id | yes | comma list of gauge station ids on the reach |
| `rl_NHDWaterbodyComID` | TEXT | id | yes | lake hl_link the reach belongs to |
| `Qi` | REAL | m3 s-1 | no | initial discharge |
| `MusK` | REAL | s | no | Muskingum K |
| `MusX` | REAL | - | no | Muskingum X |
| `n` | REAL | s m-1/3 | no | Manning n, main channel (MC n) |
| `So` | REAL | m m-1 | no | bed slope (MC s0), &gt; 0 |
| `ChSlp` | REAL | m m-1 | no | channel side slope, horizontal per vertical (MC cs) |
| `BtmWdth` | REAL | m | no | bottom width (MC bw) |
| `time` | REAL | - | yes | reserved by v2.0.1; 0 |
| `Kchan` | REAL | mm h-1 | yes | channel loss conductance; 0 = none |
| `nCC` | REAL | s m-1/3 | no | Manning n, compound channel (MC ncc) |
| `TopWdthCC` | REAL | m | no | compound channel top width (MC twcc) |
| `TopWdth` | REAL | m | no | bankfull top width (MC tw) |
| `length_m` | REAL | m | no | reach length (MC dx); = 1000 * flowpaths.lengthkm |

### hydrolocations

data_type features; geometry POINT; engine role hydrolocations; required in a builder fabric: yes.

| Column | Declared type | Unit | Nullable | Meaning |
|---|---|---|---|---|
| `fid` | INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL | id | no | 1-based row key; equals canonical row rank |
| `geom` | POINT | m (fabric CRS) | no | location point |
| `hl_id` | TEXT | id | no | hydrolocation id, unique in fabric |
| `hl_reference` | TEXT | enum | no | Gages | WBOut | WBIn | HUC12 | ... |
| `hl_link` | TEXT | id | no | native id in hl_reference's namespace |
| `hl_uri` | TEXT | id | no | &lt;hl_reference&gt;-&lt;hl_link&gt; |
| `hl_position` | TEXT | enum | no | outflow | inflow |
| `id` | TEXT | id | no | nexus id the location sits on |

### lakes

data_type features; geometry GEOMETRY; engine role lakes; required in a builder fabric: yes.

| Column | Declared type | Unit | Nullable | Meaning |
|---|---|---|---|---|
| `fid` | INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL | id | no | 1-based row key; equals canonical row rank |
| `geom` | GEOMETRY | m (fabric CRS) | yes | outlet point |
| `id` | TEXT | id | no | outlet nexus nex-&lt;m&gt; |
| `toid` | TEXT | id | no | outlet flowpath wb-&lt;m&gt; |
| `hl_id` | TEXT | id | yes | hydrolocation id |
| `hl_reference` | TEXT | enum | no | WBOut |
| `hl_link` | REAL | id | no | waterbody id (t-route lake id); integer valued |
| `hl_uri` | TEXT | id | no | WBOut-&lt;hl_link&gt; |
| `Dam_Length` | REAL | m | no | dam length (t-route dam_length_m) |
| `ifd` | REAL | - | no | initial fractional depth (0..1) |
| `LkArea` | REAL | km2 | no | lake area (area_km2) |
| `LkMxE` | REAL | m (fabric vertical datum) | no | maximum water elevation (maximum_m) |
| `OrificeA` | REAL | m2 | no | orifice area |
| `OrificeC` | REAL | - | no | orifice coefficient |
| `OrificeE` | REAL | m (fabric vertical datum) | no | orifice elevation |
| `time` | REAL | - | yes | reserved by v2.0.1; 0 |
| `WeirC` | REAL | - | no | weir coefficient |
| `WeirE` | REAL | m (fabric vertical datum) | no | weir crest elevation |
| `WeirL` | REAL | m | no | weir length |

### model_attributes

data_type attributes; geometry none; engine role model_attributes; required in a builder fabric: yes.

| Column | Declared type | Unit | Nullable | Meaning |
|---|---|---|---|---|
| `fid` | INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL | id | no | 1-based row key; equals canonical row rank |
| `divide_id` | TEXT | id | no | cat-&lt;n&gt;; one row per divide, divides order |
| `elevation_mean_m` | REAL | m (fabric vertical datum) | no | mean DEM elevation |
| `slope_mean_deg` | REAL | deg | no | mean terrain slope (Noah-OWP terrain_slope) |
| `aspect_mean_deg` | REAL | deg clockwise from north | no | circular mean aspect (Noah-OWP azimuth) |
| `centroid_lat_deg` | REAL | deg north, WGS84 | no | Noah-OWP lat |
| `centroid_lon_deg` | REAL | deg east, WGS84 | no | Noah-OWP lon |
| `soil_class` | INTEGER | enum | no | index into fabric_metadata soil_class_table (Noah-OWP isltyp) |
| `veg_class` | INTEGER | enum | no | index into fabric_metadata veg_class_table (Noah-OWP vegtyp) |
| `soil_color` | INTEGER | enum | no | Noah-OWP soilcolor (1..20) |
| `impervious_fraction` | REAL | - | no | 0..1 (CFE urban_decimal_fraction) |
| `soil_depth_m` | REAL | m | no | CFE soil_params.depth |
| `soil_b` | REAL | - | no | CFE soil_params.b (Clapp-Hornberger b) |
| `soil_satdk_m_s` | REAL | m s-1 | no | CFE soil_params.satdk |
| `soil_satpsi_m` | REAL | m | no | CFE soil_params.satpsi |
| `soil_slop` | REAL | m m-1 | no | CFE soil_params.slop |
| `soil_smcmax` | REAL | m3 m-3 | no | CFE soil_params.smcmax |
| `soil_wltsmc` | REAL | m3 m-3 | no | CFE soil_params.wltsmc |
| `refkdt` | REAL | - | no | CFE refkdt |
| `alpha_fc` | REAL | - | no | CFE alpha_fc |
| `max_gw_storage_m` | REAL | m | no | CFE max_gw_storage |
| `cgw_m_h` | REAL | m h-1 | no | CFE Cgw |
| `gw_expon` | REAL | - | no | CFE expon |
| `k_lf` | REAL | - | no | CFE K_lf |
| `k_nash` | REAL | - | no | CFE K_nash |
| `giuh_ordinates` | TEXT | - per CFE step | no | comma list summing to 1 (CFE giuh_ordinates) |

### fabric_metadata

data_type attributes; geometry none; engine role not read by the engine; required in a builder fabric: yes.

| Column | Declared type | Unit | Nullable | Meaning |
|---|---|---|---|---|
| `fid` | INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL | id | no | 1-based row key; equals canonical row rank |
| `key` | TEXT | id | no | metadata key (see contract doc) |
| `value` | TEXT | - | no | metadata value |


## 8. `fabric_metadata` keys

Every builder writes one row per key: `contract` (`arwen.hydro-world.fabric.v1`), `builder`
(`rw_hydro_world` or the CONUS tool), `builder_revision` (git SHA), `source_row` (source table row id),
`source_table_sha256`, `data_terms` (attribution and licence names of every input, from the row's
verified licence quotes), `id_namespace`, `horizontal_crs`, `vertical_datum`, `soil_class_table`
(`STAS` = the Noah-OWP SOILPARM.TBL STAS classes), `veg_class_table` (`USGS` or
`MODIFIED_IGBP_MODIS_NOAH`, the Noah-OWP MPTABLE.TBL sections), `domain` (WGS84 bbox or polygon WKT the
fabric covers), `build_utc`.

## 9. Derived engine packets

CFE INI per divide (`derive::cfe_ini`), keys in the ngen example order, units as annotated there. The
`cfe_packet_from_model_attributes_matches_ngen_example` test reproduces ngen's `cat-11223` packet value
for value.

| CFE key | Source |
|---|---|
| `forcing_file`, `surface_partitioning_scheme` | `BMI`, `Schaake` |
| `soil_params.depth` [m], `.b` [], `.satdk` [m s-1], `.satpsi` [m], `.slop` [m/m], `.smcmax` [m/m], `.wltsmc` [m/m] | `soil_depth_m`, `soil_b`, `soil_satdk_m_s`, `soil_satpsi_m`, `soil_slop`, `soil_smcmax`, `soil_wltsmc` |
| `soil_params.expon`, `soil_params.expon_secondary` | `1.0[]` (CFE initializer defaults) |
| `refkdt`, `alpha_fc` | `refkdt`, `alpha_fc` |
| `max_gw_storage` [m], `Cgw` [m h-1], `expon` [] | `max_gw_storage_m`, `cgw_m_h`, `gw_expon` |
| `K_nash` [], `K_lf` [] | `k_nash`, `k_lf` |
| `soil_storage` [m/m], `gw_storage` [m/m] | initial-state provider (lane hydro-world-init), fractions 0..1 |
| `nash_storage`, `num_timesteps`, `verbosity`, `DEBUG` | `0.0,0.0`, `1`, `1`, `0` |
| `giuh_ordinates` | `giuh_ordinates` (must sum to 1) |

Noah-OWP overrides (`derive::noahowp_overrides`): `lat`, `lon` = `centroid_lat_deg`, `centroid_lon_deg`;
`terrain_slope` = `slope_mean_deg`; `azimuth` = `aspect_mean_deg`; `isltyp` = `soil_class`; `vegtyp` =
`veg_class`; `soilcolor` = `soil_color`. Soil/veg table names come from `fabric_metadata`. Initial
`sh2o`, `sice`, `zwt` come from the initial-state provider, not the fabric.

LGAR parameters are not in v1: the LGAR packet is not landed (lgar unit: physical parameters held). They
arrive as additional `model_attributes` columns in contract v1.1 with the LGAR owner, not as a new table.

## 10. The builder: table-driven

`tools/hydro_world/data/sources.v1.json`, schema `arwen.hydro-world.sources.v1`. A new region or dataset is
a new row, never a new code path. Each row names:

| Field | Meaning |
|---|---|
| `id`, `owner` | row id (written to `fabric_metadata.source_row`), owning lane |
| `id_namespace`, `hf_source` | section 5 numbering; value for `network.hf_source` |
| `crs`, `vertical_datum` | section 4 policy (`EPSG:<code>` or `laea-domain-centroid`) |
| `inputs.<name>` | dataset, version, `url` (tile inputs: pattern with `{NS}{lat2}{EW}{lon3}`), `local` path under the data root (per-continent datasets: `{region}` with `regions`), format, source column -> meaning, licence |
| `inputs.river_network`, `inputs.catchments` | given (vector-basins) or derived from `inputs.dem` (dem-d8) |
| `inputs.dem` | grid for D8 derivation (dem-d8), reach slopes and terrain statistics |
| `inputs.lakes` | HydroLAKES points: id, area, mean depth, catchment area, pour point |
| `inputs.soil`, `inputs.land_cover` | consumed by hydro-world-params for `model_attributes` |
| `attribute_mapping` | `model_attributes` column -> recipe (parameter rows owned by hydro-world-params) |
| `gauges` | gauge feeds (agency code, service, licence) for hydrolocations |
| `build` | method (`dem-d8` or `vector-basins`) and every number it uses: grid resolution, fill tile, sea value, stream threshold, minimum reach length, lake minimum area and snap rules, recipe table, soil/veg class table names |
| `licence` on every input and feed | `status` unverified / verified / conditional / research-only / excluded, `name`, verbatim `quote`, `url`, `read_utc`, `read_by` |

Rows today (licence verdicts from hydro-world-licences, tip 5c1bbd89eb4):

| Row | Namespace | Method | Inputs | Admitted |
|---|---|---|---|---|
| `global-copernicus-glo90` (worldwide default) | 2 | dem-d8 | Copernicus GLO-90 tiles, HydroLAKES points (SoilGrids, WorldCover for params) | yes, every input verified |
| `global-hydrosheds-v1` (cross-check) | 1 | vector-basins | HydroRIVERS + HydroBASINS level 12 (research-only), Copernicus GLO-90, HydroLAKES | private research only with `--accept-licence 'WWF HydroSHEDS v1 License Agreement'`; never bundled, fetched by default or shipped |
| `conus-nhdplus-v21` | 0 | owner's tool (astra-c-hydro-data), build spec pending | NHDPlus V2.1, NAVD88 USGS 3DEP, SoilGrids 2.0, WorldCover 2021, HydroLAKES | source terms yes (DATA 2026-10-18); physical build/covering fabric/parameters/realization pending |

Licence law in code: default `source::admit(row)` requires verified inputs. A limited source marked
`research-only` remains refused with its licence named. `source::admit_with(row, accept)` allows an
explicit private standalone cross-check when the caller names that licence; the resulting fabric records
`data_usage=research-only` and must never be bundled, fetched by default or shipped. HydroSHEDS v1 has
restricted redistribution, end-user EULA/record requirements and a licence terminable by WWF. The
2026-10-18T11:50Z ruling therefore keeps its Rhine fabric as a pipeline research artifact, not a
shippable fabric or a skill qualification. The worldwide default derives hydrography from Copernicus DEM.

Recipe table `tools/hydro_world/data/recipes.v1.json` (`arwen.hydro-world.recipes.v1`): channel geometry
(top width 2.44 DA^0.34, bottom width 0.6 top width, compound width 3 top width, side slope
0.7836 DA^-0.1456, Manning n by order 0.06 / 0.055 / 0.05, compound n = 2 n, bed slope floor 1e-5, MusK 3600,
MusX 0.2) and level-pool constants (WeirC 0.4, WeirL 10 m, OrificeC 0.1, OrificeA 1 m2, Dam_Length 10,
ifd 0.9; WeirE = DEM water surface at the lake outlet, pool depth D = 2 x HydroLAKES mean depth,
LkMxE = WeirE + 0.15 D, OrificeE = LkMxE - D), each with its provenance in the table. hydro-world-params
and the reservoirs owner may replace these rows; the builder reads only the table.

Builder stages, as implemented (`rw_hydro_world build`, Rust, all cores unless named):

1. domain -> tiles: `rw_hydro_world tiles` lists the row's tile files for the box; the orchestrator
   `tools/hydro_world/hydro_world_build.py` fetches missing ones (SOURCE.md row per file; a 404 tile, open
   sea in the Copernicus tiling, goes to `ABSENT.txt` and is sea). Tiles are decoded in parallel (own
   GeoTIFF reader: deflate, floating-point predictor) and mosaicked onto a lat/lon grid whose cell centres
   are integer multiples of the resolution, clipped to the box (`south < lat <= north`, `west <= lon < east`).
2. dem-d8 hydrography: sea mask (cells at the dataset's sea value connected to the box edge or to absent
   tiles; 1 thread, breadth-first, about 10 ms for the Rhine box); tiled parallel Priority-Flood depression
   filling (Barnes 2016); D8 on the filled surface; flats drained towards lower and away from higher terrain
   (Barnes, Lehman and Mulla 2014), one flat per task; flow accumulation of exact ellipsoidal cell areas in
   integer units by level-synchronous Kahn order (deterministic). Box edges and the sea are the outlets: the
   fabric covers the box, and a river that leaves the box ends at a `tnx-` there.
3. reaches and catchments: stream cells (upstream area >= `stream_threshold_km2`, 10 km2) cut at confluences
   and right after every lake outlet; reaches shorter than `min_reach_length_km` (1 km) merged into their
   downstream reach (NextGen-style refactoring; lake-outlet and terminal reaches kept); every cell labelled
   with its reach by pointer jumping; catchments polygonised along exact cell edges, traced 8-connected
   (a D8 catchment is always 8-connected, so each is one ring that may touch itself at a cell corner and
   contains every cell of the catchment).
   vector-basins instead keeps level-12 basins with at least `basin_min_inside_fraction` of their area in
   the box (clipped to it), and takes as each basin's flowpath the main stem of its HydroRIVERS reaches.
4. topology tables with section 5 ids; Strahler order, main stems (largest drainage upstream),
   hydroseq = topological rank (strictly larger downstream), geodesic lengths, bed slope = least-squares fit
   of DEM elevation along the reach (floored by the recipe), `network.hf_id` = global 3-arc-second index of
   the reach's outlet cell (stable across boxes for the same DEM and threshold).
5. flowpath_attributes from the channel recipe.
6. terrain statistics per catchment (area-weighted mean elevation, Horn slope, circular mean aspect,
   centroid), written to `FABRIC.gpkg.terrain.json`; `model_attributes` is written when a parameter packet
   (`--params`, keyed by divide id, from hydro-world-params) covers every divide, else the table is absent
   and `check` names that one gap.
7. lakes: HydroLAKES pour points with area >= `lake_min_area_km2` snapped to the stream cell within
   `lake_snap_radius` whose upstream area is within `lake_snap_area_ratio` of the lake's catchment area;
   level-pool rows from the recipe; WBOut hydrolocations; the lake's reach carries
   `rl_NHDWaterbodyComID`. Lakes at a domain outlet or sharing an outlet with a larger lake are dropped and
   counted.
8. gauges: not yet (hydro-world-gauges owns the station tables; open item 6).
9. `canonicalize`, `gpkg::write` (row encoding on all threads, one SQLite transaction), read back (tables
   on parallel read-only connections) and require equality, `check`; the receipt lists every stage with
   its time and parallelism.

## 11. What `check` says about the existing CONUS fabrics

Both are lossless through the contract and have zero errors. Their contract gaps are what a builder must
add: no `model_attributes`, no `fabric_metadata`, rows not in canonical order; case (b) also has 73
hydroseq inversions and one lake (`nex-2420985`) with no level-pool parameters (the native case (b) run carries 19
reservoirs, 7 level pool and 12 RFC, for its 20 lakes). Each has one exterior outlet (case (a) `nex-11225` -> `wb-11225` outside the
subset; case (b) `tnx-1000006226` -> `wb-0`).

## 12. Commands

```
# on the build host, from tools/hydro_world (CARGO_TARGET_DIR and TMPDIR under your scratch)
cargo test --release --offline -j 24
# with the engine equivalence test (build tools/nextgen_hydrofabric first)
RW_HYDROFABRIC=/path/to/rw_hydrofabric cargo test --release --offline -j 24 -- --include-ignored
rw_hydro_world schema | check FABRIC.gpkg | roundtrip IN.gpkg OUT.gpkg | sources TABLE.json | admit TABLE.json ROW
# build a fabric for any box (orchestrator fetches missing tiles, then builds; optional engine prepare)
python3 tools/hydro_world/hydro_world_build.py --bbox=W,S,E,N --out FABRIC.gpkg --data-root DATA \
    --exe rw_hydro_world [--engine rw_hydrofabric --grid-dx 1000] [--accept-licence NAME]
rw_hydro_world build --sources data/sources.v1.json --row global-copernicus-glo90 --bbox=W,S,E,N \
    --data-root DATA --out FABRIC.gpkg --receipt RECEIPT.json
rw_hydro_world compare --fabric FABRIC.gpkg --reference NOAA_EPSG5070.gpkg --radius-km 3 --out OUT.json
rw_hydro_world render --fabric FABRIC.gpkg --png OUT.png [--lonlat W,S,E,N] [--reference NOAA_EPSG5070.gpkg]
```

## 13. Open items (owner)

1. Native ngen parse of a contract-written file (no rtree/GDAL triggers): run the referee's network probe
   on case (a) rewritten (hydro-world-fabric with the referee recipe).
2. CLOSED 2026-10-18: the Rhine and CONUS-box fabrics (custom-WKT LAEA) go through `rw_hydrofabric
   prepare` and `gpuwm hydro prepare` with a 1 km parent grid in their own CRS: 21 of 21 payloads, coverage
   fraction 1 for every catchment, engine polygon area = fabric area (receipts/hydro-world-fabric/).
3. `lakes` -> reservoir packer in Rust so a builder fabric needs no native capture (hydro-world-fabric with
   the reservoirs owner).
4. CONUS NHDPlus recipe writes this contract through `hydro_world::gpkg::write` (astra-c-hydro-data).
5. Licence verification of every input in `sources.v1.json` (hydro-world-licences).
6. Parameter recipes behind `attribute_mapping` (hydro-world-params); initial storage (hydro-world-init);
   gauge station tables (hydro-world-gauges).
7. Lake polygons: with HydroLAKES points only the reach ending at a lake outlet is a lake member; reaches
   crossing a large lake (e.g. Lake Constance) are channels across a flat, drawn as straight parallel paths.
   Burning HydroLAKES polygons (CC BY 4.0, a global archive with no tile service) would make every reach in a
   lake a member (hydro-world-fabric).
8. Delta bifurcations: a D8 network is a tree, so all Rhine water takes one branch through the delta (it
   leaves the Rhine box at its north edge near Utrecht, 159,778 km2); the Waal/IJssel split needs divergence
   rows, which the v2.0.1 shape does not carry (hydro-world-fabric with the routing owners).
