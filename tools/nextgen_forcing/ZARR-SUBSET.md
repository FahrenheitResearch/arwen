# Local packed weather subsets

`forcing-zarr-subset REQUEST.json` is a setup-only Rust/Rayon command. It
extracts explicitly indexed weather windows from **sealed local chunks**,
unpacks CF int16 values and writes `weather.nc` through the project's
`netcdf-writer`. New field names, units and source axes are request metadata;
there is no provider adapter. It fetches nothing, infers no geographic box,
constructs no catchment weights and adds no resident stage.

Build with scratch `CARGO_HOME`, `CARGO_TARGET_DIR` and `TMPDIR`:

```bash
cargo build --offline --locked --release --manifest-path tools/nextgen_forcing/Cargo.toml \
  --features zarr-subset --bin forcing-zarr-subset -j 8
RAYON_NUM_THREADS=8 forcing-zarr-subset REQUEST.json
```

The feature is off by default. Zstd versions/notices remain those in
`ZSTD-NOTICES/`. The shared dependency-free NetCDF writer is MIT, from
`tools/rustwx/crates/netcdf-writer`, with the full grant in
`tools/rustwx/LICENSE` at base `64fc403dfd4305510ca478b2473c263b24886e73`:
“Permission is hereby granted, free of charge, to any person obtaining a copy”.
No new third-party codec or native runtime is added.

```json
{
  "schema": "nextgen.forcing.zarr-subset-request/v1",
  "metadata": "/owned-scratch/store.zmetadata",
  "metadata_sha256": "ACTUAL_SHA256",
  "coordinate_manifest": "/owned-scratch/coordinates/manifest.json",
  "coordinate_manifest_sha256": "ACTUAL_SHA256",
  "chunks_root": "/owned-scratch/chunks",
  "output_dir": "/owned-scratch/fresh-window",
  "axes": [
    {"name": "time", "start": 97, "end": 99},
    {"name": "latitude", "start": 2250, "end": 2252},
    {"name": "longitude", "start": 840, "end": 843}
  ],
  "fields": [{"source": "TMP_2maboveground", "units": "K"}],
  "chunk_sha256": {"TMP_2maboveground/0.17.3": "ACTUAL_SHA256"}
}
```

This numerical example is a small transport window, **not a basin definition
or a complete event request**. The axes name the source's exact dimension
order; ranges are half-open source indices, with no flip, interpolation,
padding, wrapping or timestamp shift. Supply the existing
[coordinate export](ZARR-COORDINATES.md) manifest from the same metadata and
its raw coordinate payloads. Their hashes, shape and source attributes must
agree. Inferred coordinate bounds are not copied into the weather file or
promoted to provider authority.

Supported weather encoding: consolidated Zarr2, rank-three C-order `<i2`,
Zstd, no filters, dot-separated chunk indices. Array dimensions must match
all three requested coordinates. Every intersecting chunk must be present
with its SHA256; unused seals fail rather than hiding a mistaken window.
Chunks retain their declared extent at source edges; only selected in-array
values are kept. Limits: 16 fields, 64MiB decoded per chunk, 128MiB total
selected float64 weather; split larger requests at preparation. Up to the
Rayon pool's field count run concurrently; chunks within a field are serial.

Missing values are checked against `_FillValue`, `missing_value` and the
array fill value **before** unpacking. A selected missing value fails; missing
padding outside the selection is irrelevant. No spatial or temporal fill is
invented. Float64 `raw * scale_factor + add_offset` is applied once. Output
variables retain their source names/units and contain unpacked doubles without
packing attributes. Selected time coordinates retain units/calendar exactly.
The manifest seals source metadata, coordinate export, selected chunks and
output, with timings. A fresh output directory is required; the manifest is
written only after the complete NetCDF file finishes.

Pass `weather.nc` to `arwen-nextgen-forcing` with the source's existing profile
and matching **local source-grid** CSR. For AORC, `aorc-grid` divides hourly
precipitation amounts by 3600 and applies its existing -3600-second coordinate
alignment. Do not shift coordinates or unpack values again. For a run starting
00Z, that row requires the first source stamp 01Z. Geographic weights/native
sampling and actual catchment order remain hydrofabric/referee decisions.

The generic atmosphere-oriented `tools/zarr_bridge` remains unchanged. Its
current route requires pressure-level/coverage metadata and lacks int16
numeric retrieval. This narrow local transport uses the already selected
Zstd codec and shared NetCDF writer; it introduces no competing fetch path,
provider-specific adapter or atmospheric derivation.

## Measured scope

A real NOAA AORC diagnostic uses the exact metadata
`eec224d45eb7621ff8f13621f22c733be19c491389d55ab3ceda62530ef76e09`
from `https://noaa-nws-aorc-v1-1-1km.s3.amazonaws.com/2023.zarr` and the
coordinate export `941ccd77c824959c65e3896edfc6a0e9f9a7f98bbebdca0141bec8f6fa339114`.
Eight source fields, two hours and six source cells were extracted. Test-only
identity cell weights produce a `[2,8,6]` packet; these are diagnostic cell IDs,
not qualified physical catchments. A separate C witness using system libzstd
and netCDF-C compares raw selected source words/unpacking to NetCDF and both
packet precisions, including the rain amount/rate conversion. It is a referee
test tool only, never a product dependency or shipped binary.

Exact scripts, command arguments, hashes and measured results are in the lane
receipt `delivery/code/evidence/zarr-subset`. This is startup transport
verification, **not native BMI physics replay, event acquisition/skill, an
initialized runoff source, a short coupled run, or whole-network4090 speed**.
Raw GRIB and geographic event CSR remain open. Source data/fabric restrictions
are separate; this command does not acquire or redistribute a hydrofabric.

Focused checks (scratch environment as above):

```bash
RAYON_NUM_THREADS=8 cargo test --offline --locked --release \
  --manifest-path tools/nextgen_forcing/Cargo.toml --features zarr-subset \
  --bin forcing-zarr-subset -j 8
```

They use arbitrary field/axis names, a window crossing eight padded chunks,
negative/descending coordinates and an independent literal expected subset;
bad selected values, seals, units, unused chunks and existing outputs fail.
