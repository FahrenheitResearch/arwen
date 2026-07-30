# gpuwm native GRIB bridges

One Rust crate (`grib1_bridge`, library name `gpuwm_preprocess_cpu`) holding
every CPU-side GRIB decode boundary gpuwm uses.  `gpuwm.ingest.grib`,
`gpuwm.ingest.hrrr`, and `gpuwm.gfs_direct` consume the raw little-endian
arrays and JSON/TSV metadata these binaries write; no ecCodes, cfgrib, or C
runtime is required anywhere.

## Prerequisites

- A Rust toolchain (stable; `cargo` on PATH).  <https://rustup.rs> installs
  it on every platform gpuwm targets; no nightly features are used.
- Nothing else -- no network access included.  The hardened `grib-core`
  decoder and every crates.io transitive dependency are checked in under
  `vendor/` (provenance in `vendor/VENDOR.md`), so a clean clone builds
  without a user-specific path and without touching the network.

## Build

From the repository root:

```powershell
Push-Location tools/grib1_bridge
cargo build --release --locked --offline
Pop-Location
```

The command is intentionally run *in* `tools/grib1_bridge`, where Cargo finds
the checked-in source-replacement configuration in `.cargo/config.toml`
(crates.io is replaced by the `vendor/crates-io` directory).

## Where the binaries land

Everything is written to `tools/grib1_bridge/target/release/` (with an
`.exe` suffix on Windows).  Binaries are **not** committed -- `target/` is
gitignored -- so `cargo build --release` is a required step of any clean
clone that ingests GRIB sources.  The clean-clone rule is:
`pip install gpuwm` + this one cargo build + the documented data fetch.

| Binary | Purpose |
|---|---|
| `grib1_bridge` | ERA5 GRIB1 decode boundary.  Validates every concatenated GRIB1 envelope (declared length, edition, `7777` terminator, exact EOF coverage) before decode, writes every decoded message to one little-endian float64 stream, and records message/grid metadata in JSON.  `gpuwm.ingest.grib` applies `Vtable.ERA5_CDO` and assembles snapshots.  Usage: `grib1_bridge INPUT.grb OUTPUT_DIR`. |
| `gfs_grib2_bridge` | Fail-closed companion for GFS `pgrb2.0p25` series.  Accepts a `HOUR<TAB>GRIB2` manifest, inventories the whole series before publication, selects 124 records per time by raw GRIB2 identifiers, validates the NCEP/table/process identity, exact 0.25-degree grid endpoints, cycle/cadence/packing/missing-value policies, WPS-compatible RH2 and LANDN-with-LAND-fallback semantics, and both fixed surfaces for GFS's four exact Noah soil layers.  Terrain and the selected land mask must remain bit-identical across the series.  Writes little-endian FP32 regular-grid arrays consumed directly by `gpuwm.gfs_direct`; no `.rws`, WPS, or `real.exe` step is involved. |
| `hrrr_grib2_bridge` | Fail-closed native HRRR GRIB2 subset bridge for gpuwm initialization.  Accepts one contiguous public source-lead window from a single HRRR cycle, proves the exact atmosphere/surface/soil inventory the initialization lane needs (no field selected by display name or message position), and writes a south-to-north row-major FP32 source window read by `gpuwm.ingest.hrrr`. |
| `grib2_inventory` | Strict GRIB2 inventory/decode probe.  Emits one TSV row per message (raw identifiers, grid definition, packing, decode statistics) -- the manifest/receipt tool for auditing what a downloaded GRIB2 file actually contains before anything ingests it. |
| `grib2_dump` | Dumps selected GRIB2 fields as little-endian float64 with a TSV header, for decoder cross-checks against independent readers. |

## Validation posture

Every bridge is fail-closed: unexpected editions, grids, packing, missing
records, or inventory drift abort with a receipt instead of publishing a
partial product.  Upstream provenance of the vendored decoder is recorded in
`vendor/VENDOR.md`.
