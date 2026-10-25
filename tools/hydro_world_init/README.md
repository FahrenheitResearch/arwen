# rw_hydro_init: initial water state for gpuwm hydro

Builds the initial water state of every catchment of a HYDRO-WORLD fabric
(`arwen.hydro-world.fabric.v1`, any CRS) for any date, from a reanalysis row,
plus an optional CFE spin-up. Rust does every data step on all cores; Python
(`gpuwm/hydro/initial_state.py`) only fetches and chains the stages.

Door: `gpuwm hydro init-state --fabric F.gpkg --date 2023-01-01T00:00:00Z [--source era5-arco] [--spinup-days N] [--params ROW.json] --outdir DIR`

Source fetch reuse compares the saved extraction requests, including bounds,
dates, store, fields and units, and requires each NetCDF payload to exist.
New fetch receipts record each payload's byte count and SHA-256. Reuse checks
these opaque bytes in parallel before native mapping; a changed or truncated
payload, or a malformed integrity descriptor, selects a fresh sibling fetch
while preserving the old attempt. Legacy receipts without either integrity
field remain readable and unmodified, with no payload integrity claim. This
detects changes since extraction; it is not upstream source authentication.
When an API caller shares a `fetch_dir` across domains or dates, a different
request uses a sibling directory named by its request hash. Existing files
stay untouched, and repeating that request reuses its completed extraction.
An incomplete extraction gets a fresh sibling attempt. Request and integrity
checks do not decode scientific data; `rw_zarr` performs source reads and crops.

The door accepts an empty existing output directory. If the requested address
contains a previous result, is a file or symlink, or belongs to another live
run, it preserves that address and chooses a fresh sibling for its results.
The output lease lasts through preparation; the printed path and receipt's
`output_directory` name the actual result directory. Default source extractions
live beside the requested address in `DIR-source`, so repeated preparations
reuse matching downloads even when their results need separate directories.
Matching legacy `DIR/source` extractions remain usable.

Concurrent preparations sharing a source-cache address wait for its current
writer, then reuse the completed matching extraction. Requests with changed
settings keep their separate cached inputs. The lease covers cache selection
and extraction; the OS releases it if a preparation is interrupted. A retry
preserves the partial attempt and writes a fresh sibling. Lock sidecars are
permanent so another process cannot replace a live lock's inode. Extraction
slices still run concurrently within each fetch; independent cache addresses
can fetch in parallel. No numerical stage runs under the cache lease.

## Sources are rows

`gpuwm/data/hydro/init-state-sources.v1.json` (`arwen.hydro-world.init-sources.v1`).
A row names its access (store, protocol, account), licence (SPDX, quote read at
the source, attribution), grid kind, state variables with layer bounds and
units, the soil moisture transfer and the forcing bindings (terms, sign,
temporal kind, time stamp convention). Adding a source is a row.

| Row | Status | Why |
|---|---|---|
| era5-arco | admitted | ERA5 hourly 0.25 deg via Google ARCO-ERA5 Zarr; CC-BY-4.0; no account |
| era5-land-cds | pending-ruling | CC-BY-4.0 but the CDS needs an account (Drew's ruling on account-gated sources) |
| gldas-noah025-3h-2.1 | pending-ruling | CC0 but Earthdata Login needed |
| nwm-retrospective-v3 | reader-pending | public, no restrictions, but on a projected 1 km grid; rw_zarr reads only lat/lon stores |

`rw_hydro_init` and `initial_state.admit` refuse any row that is not admitted,
naming the row's own reason (licence law in code).

## What it produces

`state` (one time): per catchment, land-weighted area means (weights times
land fraction; a catchment with no land cell takes the nearest land cells and
is counted), the row's soil moisture transfer, depth-overlap remap onto the
Noah layers (0-0.1, 0.1-0.4, 0.4-1.0, 1.0-2.0 m), the Noah-OWP supercooled
liquid split (opt_frz=1) into `sh2o`/`sice`, soil temperature, snow water
equivalent, depth and temperature, and CFE `soil_storage` (mean water over the
CFE soil depth / smcmax) and `gw_storage` (cold start from the parameter row).
Every array is sealed in `manifest.json` with SHA-256; `cfe-initial.json` and
`noahowp-overrides.json` are the engine hand-offs (CFE INI values for
`hydro_world::derive::cfe_ini`, namelist overrides for `arwen-noahowp-init`).
Bounds are checked and counted in the receipt.

Soil moisture transfer `field_capacity_anchored` (default for ERA5): an index
0 at wilting point, 1 at field capacity, 2 at saturation, per cell with the
source soil class (IFS Cy43r1 Part IV Table 8.9), read back with the
catchment's wltsmc, CFE field capacity and smcmax. `wetness_index` (wilting
and saturation only) and `volumetric` stay selectable per row.

`spinup` (N days): CFE runs per catchment on the row's forcing (rain plus
snowmelt, PET), from the state at date minus N days to the date, then writes
the spun-up CFE storages (soil, groundwater, Nash, GIUH queue, carried
percolation), per-catchment mass balance, daily domain series, and a channel
initial state: steady flow accumulated down the fabric network from each
catchment's last-day runoff, with Manning depth and velocity on each reach's
trapezoid (MC convention z = 1/ChSlp), as the `routing_state` (Q, v, h) triple
the engine's MC component takes. New spin-up packets bind that array's ordered
flowpath axis in the same manifest with `routing_ids: {file: "routing_ids.json",
sha256: "<SHA-256 of exact JSON bytes>", count: N}`. Regenerating an older
unsealed packet through `spinup` creates this binding; hashing an old ID file
alone cannot recover its original association to the state rows. Divides-only
fabrics emit neither routing state nor an ID descriptor.
Reservoirs keep the reservoir preparer's
existing `ifd` cold-start rule (tools/nextgen_reservoir_prepare).

The CFE here is a CPU Rust port of NOAA-OWP/cfe
f9182dfb3d81c407c66c5b782114173893c30cf2 (Apache-2.0; LICENSE "Apache License",
"Version 2.0, January 2004"), Schaake partitioning and GIUH only, with the C
order of operations. It is checked word for word against the native build:
`referee/cfe_trace_native.c` (referee tool, gcc, never on a user path) and
`rw_hydro_init cfe-trace` print every state word in hex for the same INI and
BMI inputs; `spinup` writes those inputs for chosen catchments (`trace`).
The run-time module stays the engine's CUDA CFE.

Weights: each divide is moved into the grid's coordinates by PROJ (lon/lat
for a lat/lon grid, the grid CRS for a projected grid), clipped to every cell,
and measured on the WGS84 ellipsoid through the authalic sphere (lat/lon) or
in the plane (projected). Coverage fraction is reported per catchment.

## Commands

Each takes one REQUEST.json, prints a JSON receipt and keeps it in its output
directory (directories must be new).

| Command | Request keys |
|---|---|
| tessellate | name, south, west, north, east, cell_m, output (proof fixture: equal-area squares, not hydrography) |
| bbox | fabric, pad_degrees |
| weights | fabric, grid_file, table, row, output, densify_m (default 1000) |
| state | table, row, fabric, weights, source_file, valid_time, params, output |
| spinup | table, row, fabric, weights, initial_state, land_file, forcing_files, start, hours, params, output, trace, initial_override |
| cfe-trace | ini, forcing_csv (precip mm/h, PET m/s per line), output |
| engine-packets | fabric, state, spinup, noah_namelist, time_origin, output |

## Build and test (build host)

```
cd tools/hydro_world_init
CARGO_TARGET_DIR=<scratch>/target TMPDIR=<scratch>/tmp cargo test --release --offline -j 24   # 8 tests
CARGO_TARGET_DIR=<scratch>/target TMPDIR=<scratch>/tmp cargo build --release --offline -j 24
export GPUWM_RW_HYDRO_INIT=<scratch>/target/release/rw_hydro_init GPUWM_RW_ZARR=<rw_zarr from tools/zarr_bridge>
```

Links pinned static PROJ 9.7.1 and SQLite 3.46.1 through
[`hydro_proj`](../hydro_proj/README.md); the projection database is embedded and
network access is disabled. No system PROJ or SQLite installation is required.

## Proof receipts

`receipts/hydro-world-init/RECEIPT.json` and the per-stage receipts beside it.

## Exact continuation boundary

New spin-ups write raw f64 `cfe_soil_storage_m` and `cfe_gw_storage_m`, in
addition to the existing fraction arrays. Their manifest's `cfe_raw_initial`
descriptor (schema `arwen.hydro-world.cfe-raw-initial.v1`) seals the exact ordered
`cfe-parameters.json` INIs by fixed filename, SHA-256 and column count. These are
the INIs actually used by spin-up, so their per-column Nash/GIUH counts bind the
padded arrays. The raw depths, subsurface Nash stores, GIUH queue (excluding the
extra zero sentinel), and carried percolation all reach `nextgen-cfe-prepare`
through its five `initial_*` keys, without a fraction round trip or clamping.
The parameter descriptor and array descriptors share one manifest identity.

`engine-packets` checks ordered parameter IDs and static INI text against the
same fabric, together with the existing manifest fabric/epoch/ID checks, typed
array seals, bounds, counts and zero padding, before creating output. A present
malformed raw descriptor never falls back. Descriptor-absent v1 spin-ups retain
their legacy fraction/Nash behavior; the receipt explicitly names missing raw
stores, GIUH and percolation, and calls for regeneration. Hashing legacy files
does not manufacture producer provenance. `cfe-initial.json` remains a legacy
fraction summary, not the complete handoff.

This is physical initialization at a fresh native diagnostic/budget epoch.
The consumer records initial Nash/surface inventory offsets and treats carried
percolation as a flux, not additional stored water. Complete native restart
counters, connected first/next update and disjoint coupled budget qualification
remain separate obligations; this transport proof is not a restart claim.


`engine-packets` requires the state and spin-up manifests to match the fabric
hash, exact catchment roster, and requested epoch. It checks sealed soil and snow
array dtype, shape, units, byte count and finite values. Declared four-layer soil
bounds are retained. The Noah request includes soil temperature, liquid/ice and
aggregate snow (SWE converted m -> mm before native REAL rounding at the producer boundary), for the
consumer's explicit all-ice, uniform-temperature DIVIDE initialization recipe.
Legacy CFE INIs use the native `nash_storage_subsurface` key; complete raw
initialization uses the sealed contract above.
