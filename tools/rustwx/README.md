# gpuwm rust render engine (vendored Rusty Weather)

One Rust workspace holding the production Rusty Weather renderer that
`gpuwm render --engine rust` drives: `rw_wrfbatch` imports wrfout
NetCDF files (pure-Rust reader, no netCDF-C) and renders the
production catalog (324 vendored entries; 151 implicit-render
candidates evaluated per file, the remainder being explicit-opt-in
ensemble/probabilistic families) -- composite/1 km reflectivity,
2 m temperature/dewpoint/RH (each with 10 m wind variants), MSLP +
10 m winds, precipitable water, low/middle/high cloud cover, the
200/250/300/500/700/850 mb chart families (height, temperature,
dewpoint, RH, absolute vorticity, each with winds), SB/ML/MU CAPE and
CIN, 0-1/0-3 km SRH, 0-1/0-6 km bulk shear, fixed-layer STP, the
heavy ECAPE family (`--heavy`), total QPF, and the multi-hour
windowed accumulations (run-total QPF, UH/wind maxima, 24/48 h
temperature/RH/dewpoint statistics) on whole-hour multi-frame runs --
with coast, state, and county basemaps, at the campaign product
sheets' quality.  Every product a store's fields prove out renders;
`--list-products` prints all 151 candidate rows with per-file
availability and a field-level reason for anything unavailable.  Provenance and
licensing: `VENDOR.md` (the code is Drew's own rusty-weather
workspace, MIT).

## Prerequisites

- A Rust toolchain (stable; `cargo` on PATH).  <https://rustup.rs>
  installs it on every platform gpuwm targets; no nightly features are
  used.
- Nothing else -- no network access included.  Every crates.io and git
  transitive dependency is checked in under `vendor/crates-io`
  (provenance in `VENDOR.md`), so a clean clone builds without a
  user-specific path and without touching the network.

## Build

From the repository root:

```powershell
Push-Location tools/rustwx
cargo build --release --locked --offline
Pop-Location
```

The command is intentionally run *in* `tools/rustwx`, where Cargo finds
the checked-in source-replacement configuration in `.cargo/config.toml`
(crates.io and the pinned git sources are replaced by the
`vendor/crates-io` directory).

## Where the binary lands

`tools/rustwx/target/release/rw_wrfbatch` (`.exe` on Windows).  It is
**not** committed -- `target/` is gitignored.  `gpuwm render` resolves
it exactly like the GRIB bridges (`gpuwm.rustwx`): the
`GPUWM_RW_WRFBATCH` environment variable, this checkout's
`target/{release,debug}`, `libexec/bridges` beside the package, then
`~/.gpuwm/bridges`.  Once found and probe-verified it becomes the
default render engine; without it `gpuwm render` falls back to the
matplotlib engine and says so.

## Basemaps

`assets/basemap/` carries the Natural Earth 10m/110m and US Census
county layers (public domain) the charts draw.  A binary running from
this checkout finds them by walking its own ancestors; for a relocated
binary `gpuwm render` pins `RUSTWX_BASEMAP_DIR` to this directory when
it exists, and an explicit `RUSTWX_BASEMAP_DIR`/`RUSTWX_ASSETS_DIR` in
your environment always wins.

## Direct use

```
rw_wrfbatch --store-root SCRATCH --out-dir PNGDIR \
    [--products all|SLUGS] [--frames all|N] [--width N] [--height N] \
    [--heavy] [--list-products] wrfout...
```

`--list-products` imports and then, instead of rendering, prints one
`PRODUCT\t<slug>\t<kind>\t<status>\t<detail>` row per catalog entry
(statuses: renderable / missing-fields / gated / blocked / excluded,
each with its reason) plus a `CATALOG total=...` tally.

The importer stages each run into an rw-store tree under
`--store-root` (scratch space; `gpuwm render` uses a per-file temporary
directory and cleans it up).  Sub-hourly output cadences are fully
supported: frames land on an exact-time axis and every product PNG
carries its `valid_..._lead_...` stamp in both filename and subtitle.
`--heavy` additionally computes the heavy ECAPE product family during
import (minutes per frame; off by default, exactly like the campaign
flow).
