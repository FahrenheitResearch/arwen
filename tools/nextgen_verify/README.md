# NextGen gauge verification

Original Apache-2.0 Rust observation fetch, table-based JSON/CSV/NetCDF import,
NSE, KGE (Gupta 2009), RMSE and peak timing. Python only launches the executable.
No hydrologic model source or native runtime is included.

Build with `cargo build --release --locked --manifest-path tools/nextgen_verify/Cargo.toml`.
Set `GPUWM_NEXTGEN_VERIFY` to the resulting `rw_nextgen_verify` executable,
or use the packaged bridge. The bridge roster includes this executable; a
release must build and stage it like the other Rust tools.

```
python -m gpuwm.hydro.verify_usgs fetch --table gauges.json \
  --start 2023-01-05T00:00:00Z --end 2023-01-17T00:00:00Z --out observations
python -m gpuwm.hydro.verify_usgs import --table gauges.json \
  --layout feature-long --files model-files.json --out model
python -m gpuwm.hydro.verify_usgs score \
  --model model/hydrographs.json --reference native/hydrographs.json \
  --observations observations/observations.json --out comparison
python -m gpuwm.hydro.verify_usgs compare \
  --prediction archive/hydrographs.json --label "NWM v3" \
  --observations observations/observations.json --out archive-observed
```

`model-files.json` is an array of input paths. A table station row has
`station_id`, optional `feature_id` for the routing history join, and optional
`time_series_id` to select among conflicting USGS series. Station identifiers
are strings so leading zeroes survive. Histories use the canonical rows:

```json
[{"station_id":"11462500","time":"2023-01-05T00:00:00Z","flow_m3s":1.0}]
```

The example values above illustrate the schema; they are not event evidence.
`feature-long` imports `feature_id,time,discharge,units` columns. A layout's
JSON pointers, wildcard record path, declared units/factors and quality filters
also describe other formats without adding a source-specific adapter. `stations`
extracts station/feature rows from attribute metadata using four pointer/field
arguments; it refuses conflicting gauge mappings rather than picking a reach.

NetCDF input uses the same `import` command and a layout row such as:

```json
{"unit_factors":{"m3 s-1":1.0},"netcdf":{
 "feature":"feature_id","time":"time","flow":"flow",
 "epoch":"2023-01-01T00:00:00Z","time_unit_seconds":1.0,
 "expected_time_units":"seconds since 2023-01-01 00:00:00"}}
```

The row declares the exact expected file time units and corresponding RFC3339
epoch/conversion. The importer checks the attribute, preserves integer feature
IDs without f64 conversion, and uses coordinate dimension names to distinguish
`flow(feature,time)` from `flow(time,feature)`, including equal-length axes.
Feature-only discharge is accepted only with one time coordinate. Flow units
come from the file and must have a declared conversion factor. Packing and fill
metadata are decoded in Rust; nonfinite/negative flows are rejected and counted.
JSON record filters/time-series selectors are not applicable to this feature-axis
layout and are rejected. Missing requested features retain a partial receipt.
The existing vendored MIT/Apache-2.0 `netcdf-reader` supports CDF-1/2/5 and
NetCDF-4/HDF5, with no NetCDF-C, netCDF-Fortran or model runtime dependency.

The supplied table uses USGS's continuous OGC API (the replacement service for
NWIS instantaneous values), discharge parameter 00060, Approved/Provisional
records, and rejects unsupported qualifiers, negative fill values and missing
values. Pagination is sequential within each station, with stations fetched
in parallel. `USGS_API_KEY` is optional and only sent to the official API host.
Raw response pages, source URLs, SHA256 and timing are saved beside normalized
observations. The source table is retained in the fetch receipt. See
https://api.waterdata.usgs.gov/docs/ogcapi/migration/ .

Times must have an RFC3339 offset; matching uses exact UTC milliseconds, with
no interpolation. Duplicate UTC times with different values are an error.
Both observation skill comparisons use the same model/native/observation
intersection. Reports retain each series count and full-support model/native
f64 word equality; this is equality of imported values, not a native module
replay.
Missing observations still return a nonzero result and are reported separately
from model/native value equality; they do not turn an exact native match into
a false numerical mismatch.
NSE/KGE and peak timing are null with fewer than two matching samples;
constant-reference NSE and constant-series KGE are null with a reason.
Peak timing is also undefined when either series is constant: choosing the
earliest tied sample would manufacture a lag for an event with no peak.
Peak timing is model minus reference, positive when late; the earliest tied
peak wins on the shared comparison support.

`compare` scores one labelled prediction against observations on their two-way
exact UTC intersection. It reports unmatched counts for both series and retains
stations with missing pairs as failures. This door permits an archive/observation
sanity comparison before native/GPU histories exist, without inventing a native
series or a module replay claim. Chart labels use the supplied prediction label.

`observations --observations FILE --out DIR` produces observed-only SVG
hydrographs and peak receipts. `replay --table TABLE --station ID --raw PAGE
--expected ROWS --out DIR` replays retained actual USGS decode/normalization;
its scope is not hydrologic physics. Score charts are analysis SVGs with
UTC and m³/s units, independent of spatial weather/terrain rendering.

Parsing, joins and scoring run in Rust. Rayon parallelizes stations, files,
and the three input series; reductions within each station are ordered f64.
Set `RAYON_NUM_THREADS` to select the worker budget; the default uses all
available cores. These are setup/output operations, never work in a routing
step. Timings include the named CPU stage, not GPU kernel or whole-step speed.

`gpuwm.hydro.verify_usgs.register_cli(subparsers)` lets the hydro CLI owner
register the same door. Until that owner's CLI is merged, the module command
above is the available source-checkout door. Native/GPU case histories and
unambiguous hydrofabric gauge mapping are required before reporting skill.
Hydrofabric database licences and attribution remain separate from software.

## Worldwide gauge rows (HYDRO-WORLD)

The same door fetches gauges outside the US. Each national service is a source
row in `gpuwm/hydro/usgs_table.json`; there is no per-agency code path. A row
can declare, as data: a POST form request with a same-origin download redirect,
a zip container (`records` from the JSON member, `info` from a `key=value`
member), a station metadata request (location, name, drainage area and any
value later templates need, used as `{meta:<pointer>}`), document-level layout
pointers (a leading `^`), member filters in pointers (`/timeseries/[shortname=Q]/longname`),
a per-owner licence gate, a discovery request, requests in flight per service, and a
licence verdict copied from `docs/public/HYDRO-WORLD-LICENCES.md`.

| Row | Service | Licence (verdict) | Values | Drainage area |
|---|---|---|---|---|
| `usgs-continuous` | USGS OGC API | US public domain (allowed) | continuous, UTC | `drainage_area` (sq mi to km2) |
| `pegelonline-archive` | WSV PEGELONLINE long-term download (from 2000-01-01) | dl-de/zero-2-0 (allowed) | 15 min raw, `quality=unchecked` | not published |
| `pegelonline-rest` | WSV PEGELONLINE REST (about the last 31 days) | dl-de/zero-2-0 (allowed) | 15 min raw | not published |
| `eccc-hydat-daily` | ECCC OGC API, HYDAT daily means | OGL-Canada + ECCC end-use licence (allowed) | daily mean, local standard day | `DRAINAGE_AREA_GROSS` km2 |
| `eccc-realtime` | ECCC OGC API, real-time | same (allowed) | provisional, UTC | `DRAINAGE_AREA_GROSS` km2 |
| `bom-wdo` | BOM Water Data Online (KiWIS) | CC BY 4.0 per data owner (allowed for 43 owners) | hourly mean, quality code | not published |
| `uk-nrfa` | UK NRFA | refused: no sublicensing, no download sites, 30-day cache | never fetched | |
| `grdc` | GRDC | refused: no commercial use, no redistribution | never fetched | |

Station ids are the bare number for USGS (`Gages-<n>`, the v2.0.1 form) and
`AGENCY:<number>` for every other agency (`Gages-PEGEL:2730010`), matching the
HYDRO-WORLD contract's hydrolocation ids. A station row names its `source`.
HYDAT daily values carry a local date only, so their station rows declare
`utc_offset`; the verifier never guesses a timezone. A BOM station whose data
owner is not on the CC BY owner list (for example Snowy Hydro, w00205) is
refused by name. NRFA and GRDC rows exist only to refuse with their clause.
One station's failure (for example a PEGELONLINE series with no long-term
download) is recorded in `fetch.json` `failed_stations` with the saved response;
the other stations still land and the command exits non-zero. A rate limit's
`Retry-After` is honoured up to 60 s and reported beyond that.

Common schema. `fetch` writes `observations.json` (the canonical rows above,
now with `quality`) and `stations.json`, one record per gauge: `station_id`,
`agency`, `source`, `name`, `latitude`, `longitude`, `drainage_area_km2`,
`drainage_area_origin` (`source metadata ...`, `table row: <citation>` or
`not published by source`), `hl_uri`, `licence`, `attribution`. `ATTRIBUTION.txt`
lists every line the data terms require. A table can take its source rows from
another table with `"sources_from": "<path>"`, so a station list is just rows.

Commands added:

```
rw_nextgen_verify describe TABLE SOURCE OUT          # stations.json without values
rw_nextgen_verify discover TABLE SOURCE MINLON,MINLAT,MAXLON,MAXLAT OUT.json
rw_nextgen_verify match TABLE FABRIC.gpkg STATIONS.json OUT [OBSERVATIONS.json]
rw_nextgen_verify match-check FABRIC.gpkg OUT/matches.json OUT/check.json
python -m gpuwm.hydro.verify_usgs match --fabric F.gpkg --stations S.json --out M
```

Matching (`match`) reads any HYDRO-WORLD contract GeoPackage through
`tools/hydro_world`, projects each gauge into the fabric's own CRS (Albers or
Lambert azimuthal equal area read from its WKT; checked against the IOGP
worked examples) and assigns the nearest flowpath inside `radius_m` whose area
range (inflow area at its top, total area at its outlet) agrees with the
gauge's published drainage area within `area_tolerance`. Without a published
area (PEGELONLINE, BOM) and with the gauge's observations given as the fifth
argument, the default rule `nearest_plausible` takes the nearest flowpath whose
area can carry the observed mean flow at no more than
`max_specific_discharge_m3s_km2` (0.2, i.e. 17 mm of runoff a day over the
window): a 1,950 km2 tributary cannot carry the Rhine's 970 m3/s at Kehl, while
a tributary gauge still lands on its own river when it is the nearest. Without
observations it falls back to the largest flowpath within `no_area_radius_m`
(1 km). A gauge with no agreeing flowpath is left unmatched with the reason;
nothing is forced. Rules live in the table's `matching` block. Outputs: `matches.json` (candidates and
area errors per gauge), `hydrolocations.json` (contract rows for builder
stage 8), `match-map.svg`, `match.json` (receipt with stage timings).
`match-check` compares the matches with a fabric's own gauge links
(`flowpath_attributes.rl_gages`, `Gages` hydrolocations at nexuses).

`observations` also writes `observed-all.svg`, every station on one chart with
names from `stations.json`. Scoring (`compare`, `score`) is unchanged and takes
the worldwide rows as they are: station ids are strings.

`rw_gauge_proof_fabric TILE_DIR MINLON MINLAT MAXLON MAXLAT CELLS_PER_DEG
CHANNEL_KM2 OUT.gpkg` builds a coarse D8 reach network from Copernicus GLO-90
tiles (minimum per cell, Priority-Flood, junction-to-junction reaches) as a
contract GeoPackage with flowpaths, nexus and network only. It is a test
fixture for matching before the hydro-world fabric builder lands, not a product
fabric, and uses id namespace 9.

Tests: `cargo test --release --offline` (24 tests: projection worked examples,
GeoPackage geometry, zip container, templates, pointer member filter, owner gate, matcher rules, and
every world row decoding a response shaped like the service's own, plus the
original scoring tests).
