# recast-radar-tools

[![PyPI](https://img.shields.io/pypi/v/recast-radar)](https://pypi.org/project/recast-radar/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/recast-radar/)
[![Docs](https://img.shields.io/badge/docs-field%20guide-6BB8B8)](https://recastsystems.github.io/recast-radar-docs/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

```sh
pip install recast-radar
```

Prebuilt for Windows, Linux and macOS; no Rust toolchain needed. The
[field guide](https://recastsystems.github.io/recast-radar-docs/) walks
through Python, the command line and Rust with real data.

Pure-Rust weather radar libraries. They read NEXRAD Level II, NEXRAD and TDWR
Level III, ODIM_H5, CfRadial 1 and 2 (classic netCDF and netCDF-4), DORADE,
JMA radar GRIB2 and Meteo-France radar BUFR files into one data model that follows WMO FM301 (CfRadial 2),
and write that model as NEXRAD Level II, CfRadial 1, CfRadial 2 / FM301 and
ODIM_H5. They download NEXRAD Level II
volumes and real-time chunks from AWS, and data from other public feeds. They
dealias Doppler velocity, filter gates, compute derived products, build
composites and cross sections, track storm cells and render sweeps to PNG.
There is no unsafe code, and without the `net` feature no C in the build.
There is no GUI. The `recast-radar` command-line tool
([`crates/recast-radar-cli`](crates/recast-radar-cli), guide
[docs/guide/cli.md](docs/guide/cli.md)) runs them from a shell, and the
Python package `recast_radar` ([`crates/recast-radar-py`](crates/recast-radar-py),
guide [docs/guide/python.md](docs/guide/python.md)) opens radar files as
xarray DataTrees and Py-ART radars.

Status: version 0.1.3. The Python package is on
[PyPI](https://pypi.org/project/recast-radar/); the crates are not on
crates.io yet, and the API is not stable
(see [CHANGELOG.md](CHANGELOG.md)). The data model is checked against xradar
and Py-ART goldens. Design notes: [docs/design](docs/design).

## Quick start

`recast-radar-tools` re-exports the other crates as modules, each behind a
Cargo feature (see [Features](#features)). Until the crates are published,
depend on it by path or by git:

```toml
[dependencies]
recast-radar-tools = { path = "../recast-radar-tools/crates/recast-radar-tools" }
```

Minimum Rust version: 1.94 (edition 2024).

This program reads a Level II file and lists its sweeps:

<!-- example: crates/recast-radar-tools/examples/decode_level2.rs -->
```rust
//! Decode a NEXRAD Level II file and list its sweeps.
//!
//! cargo run --release -p recast-radar-tools --example decode_level2 -- <level2-file>

use std::error::Error;
use std::path::PathBuf;

use recast_radar_tools::nexrad;

fn main() -> Result<(), Box<dyn Error>> {
    let path = PathBuf::from(std::env::args_os().nth(1).ok_or("usage: <level2-file>")?);

    // Uncompressed, gzip, bzip2 and LDM block-bzip2 archives all decode here.
    let volume = nexrad::read_volume_from_path(&path)?;

    println!(
        "{} at {}",
        volume.attrs.instrument_name, volume.time_reference
    );
    if let Some(vcp) = volume.scan.vcp_pattern {
        println!("VCP {vcp}");
    }
    for (index, sweep) in volume.sweeps.iter().enumerate() {
        let fields: Vec<&str> = sweep.fields.iter().map(|f| f.name.as_str()).collect();
        println!(
            "sweep {index:>2}: {:>5.2} deg, {} rays, {}",
            sweep.fixed_angle_deg,
            sweep.nrays(),
            fields.join(" ")
        );
    }
    Ok(())
}
```

On [`KTLX20240315_000217_V06`](https://unidata-nexrad-level2.s3.amazonaws.com/2024/03/15/KTLX/KTLX20240315_000217_V06)
from the public `unidata-nexrad-level2` bucket (testdata id
`l2-ktlx-20240315-000217`), the first lines are:

<!-- output-head: decode_level2 testdata:l2-ktlx-20240315-000217 -->
```text
KTLX at 2024-03-15 00:02:17 UTC
VCP 212
sweep  0:  0.48 deg, 720 rays, DBZH ZDR PHIDP RHOHV CCORH
sweep  1:  0.48 deg, 720 rays, DBZH VRADH WRADH
sweep  2:  0.88 deg, 720 rays, DBZH ZDR PHIDP RHOHV CCORH
```

For a file of any supported format, `io::read_supported_volume_bytes(&bytes)`
sniffs the format and calls the matching decoder: DORADE, an HDF5 container by
content (ODIM_H5, netCDF-4 CfRadial 1 or CfRadial 2), classic-netCDF CfRadial
1, JMA GRIB2 tar, Meteo-France BUFR, NEXRAD Level III, or Level II. It also unwraps gzip and
single-file ZIP archives. A Level III product becomes one sweep per data array;
`io::read_supported_volume_with_metadata` also returns the decoded product.
`level3::decode_message(&bytes)` (feature `level3`, part of `io`) decodes any
Level III file: a product, including the graphic and tabular products that have
no data array, a General Status Message or a plain-text message (`NOUS`
headings); `level3::decode_product(&bytes)` decodes products only and returns
`Level3Error::TextOnly` or `Level3Error::NotAProduct` for the other two.

## User guide

The [user guide](docs/guide/README.md) covers the common tasks, each with a
runnable program from
[`crates/recast-radar-tools/examples/`](crates/recast-radar-tools/examples/):

| Task | Guide | Example |
|---|---|---|
| Read a file of any format, list sweeps and fields | [Reading radar files](docs/guide/reading.md) | `read_any` |
| Write Level II, CfRadial 1, FM301 and ODIM_H5 | [Writing radar files](docs/guide/writing.md) | `write_formats` |
| Get physical values, gate positions and heights | [The data model](docs/guide/data-model.md) | `physical_values` |
| Download Level II from AWS | [Fetching data](docs/guide/fetching.md) | `fetch_aws` |
| Dealias velocity; composites, echo tops, VIL | [Processing](docs/guide/processing.md) | `dealias_velocity`, `composite` |
| Render a sweep to PNG | [Rendering](docs/guide/rendering.md) | `render_png` |
| Features, errors, naming, limits, WebAssembly | [Conventions](docs/guide/conventions.md) | |
| The `recast-radar` command | [Command-line tool](docs/guide/cli.md) | |
| The `recast_radar` Python package | [Python](docs/guide/python.md) | |

Run an example with
`cargo run --release -p recast-radar-tools --features full --example <name> -- <args>`.
The programs are shown in full in the guide, and
`crates/recast-radar-tools/tests/readme.rs` fails when a copy differs from its
file.

## Crates

<!-- crate-map:start -->
| Crate | Module | Contents |
|---|---|---|
| `recast-radar-tools` | | The facade: re-exports the crates below as modules behind features |
| `recast-radar-core` | `model` | FM301 data model: volumes, sweeps, ray coordinates, fields with CF packing, the FM301 group view, beam geometry, field names, decode limits |
| `recast-radar-io-nexrad` | `nexrad` | NEXRAD Archive II (Level II), Message 31 and legacy Message 1, uncompressed, gzip, bzip2 or LDM block-bzip2, with the metadata messages; writes any volume as Archive II, real-time chunks or a GR2Analyst polling directory ([docs/level2/writer.md](docs/level2/writer.md)) |
| `recast-radar-bzip2` | | bzip2 compressor and decompressor without unsafe code or required dependencies (rayon with the `rayon` feature), written for LDM records: the decoder takes about a quarter of the instructions of C libbzip2; the encoder writes libbzip2 1.0.8's streams byte for byte (except, in a block that repeats a shorter string, which of its identical rows `origPtr` names; the stream decodes the same) in 0.6 to 0.8 of its time on LDM records, and in less time than it at every input size measured, from 16 bytes up |
| `recast-radar-io-level3` | `level3` | NEXRAD and TDWR Level III products: NOAAPort/WMO framing, message and product description headers, symbology, graphic and tabular blocks, display packets, data levels, the VAD Wind Profile |
| `recast-radar-hdf5` | `hdf5` | HDF5 reader without unsafe code (superblocks 0-3, old and new-style groups, dense links and attributes, fractal heaps, v2 B-trees, every chunk index, deflate/shuffle/Fletcher-32, the netCDF-4 data model) and an HDF5 / netCDF-4 writer |
| `recast-radar-io-odim` | `odim` | ODIM_H5 polar volumes and Cartesian products, through `recast-radar-hdf5`; an ODIM_H5 polar volume writer |
| `recast-radar-io-cfradial` | `cfradial` | CfRadial 1 (classic netCDF CDF-1/CDF-2 or netCDF-4) and CfRadial 2 / FM301 (netCDF-4), through readers written in Rust; CfRadial 1 (CDF-2) and CfRadial 2 / FM301 writers |
| `recast-radar-io-dorade` | `dorade` | DORADE sweepfiles and mobile-radar (DOW, COW, RaXPol) archives |
| `recast-radar-io-jma` | `jma` | Japan Meteorological Agency polar-coordinate radar GRIB2 tar archives |
| `recast-radar-io-bufr` | `bufr` | WMO BUFR (editions 2 to 4, written from the WMO specification, with the WMO and Meteo-France tables embedded) and Meteo-France PAG and PAM polar radar files (gzip and compress members) |
| `recast-radar-io` | `io` | Format sniffing: routes a byte buffer to the matching decoder |
| `recast-radar-data` | `data` | NEXRAD Level II archive and real-time chunks on AWS, site catalogs, international and community feeds |
| `recast-radar-correct` | `correct` | Doppler velocity dealiasing |
| `recast-radar-filters` | `filters` | Gate filters, polar smoothing, display interpolation |
| `recast-radar-retrieve` | `retrieve` | Sweep and column products (KDP by windowed regression, Vulpiani or Maesaka; PHIDP-linear or Z-PHI attenuation correction), CAPPI, azimuthal shear and rotation, GBVTD, VAD wind profile |
| `recast-radar-map` | `map` | Composites, echo tops, VIL, hail, cross sections, RHI panels, volume resampling, Cartesian gridding of one or more volumes (Barnes, Cressman) |
| `recast-radar-track` | `track` | Storm cell identification and tracking, rotation tracks, swaths, temporal grids |
| `recast-radar-render` | `render` | CPU rendering to RGBA and PNG, color tables, GR `.pal` palettes |
| `recast-radar-scattering` | `scattering` | Radar-scattering primitives and offline lookup tables |
| `recast-radar-cli` | | The `recast-radar` command: info, dump, render, fetch, validate, bench, convert, publish, serve |
| `recast-radar-py` | | The Python package `recast_radar` (PyO3 extension module, built into wheels by maturin, not published): FM301 DataTrees, Py-ART radars, writers, fetchers |
| `recast-radar-bench` | | Decode and render benchmark with output checksums (binary, not published) |
| `recast-radar-testdata` | | Real test files: manifests, committed fixtures, a SHA-256-verified download cache and the Level II trim tool (for tests only, not published) |
<!-- crate-map:end -->

Not in the workspace: `fuzz/` (cargo-fuzz targets, its own workspace; see
[fuzz/README.md](fuzz/README.md)).

`crates/recast-radar-tools/tests/readme.rs` checks that the map lists exactly
the crates under `crates/` and that the Module column matches the facade's
re-exports. It also checks every crate's package metadata (description,
keywords, categories, readme, license, a version on every internal
dependency, and a LICENSE file in the package).

## Features

<!-- features:start -->
| Feature | Module | Crate | Also enables | Default |
|---|---|---|---|---|
| (always on) | `model` | `recast-radar-core` | | yes |
| `nexrad` | `nexrad` | `recast-radar-io-nexrad` | | via `io` |
| `write` | | | `nexrad` | |
| `level3` | `level3` | `recast-radar-io-level3` | | via `io` |
| `odim` | `odim` | `recast-radar-io-odim` | `hdf5` | via `io` |
| `cfradial` | `cfradial` | `recast-radar-io-cfradial` | `hdf5` | via `io` |
| `dorade` | `dorade` | `recast-radar-io-dorade` | | via `io` |
| `jma` | `jma` | `recast-radar-io-jma` | | via `io` |
| `bufr` | `bufr` | `recast-radar-io-bufr` | | via `io` |
| `hdf5` | `hdf5` | `recast-radar-hdf5` | | via `io` |
| `io` | `io` | `recast-radar-io` | `nexrad` `level3` `odim` `cfradial` `dorade` `jma` `bufr` | yes |
| `net` | `data` | `recast-radar-data` | | |
| `correct` | `correct` | `recast-radar-correct` | | yes |
| `filters` | `filters` | `recast-radar-filters` | | yes |
| `retrieve` | `retrieve` | `recast-radar-retrieve` | `correct` | yes |
| `map` | `map` | `recast-radar-map` | `correct` `filters` | yes |
| `track` | `track` | `recast-radar-track` | `correct` `map` `retrieve` | |
| `render` | `render` | `recast-radar-render` | `correct` | |
| `scattering` | `scattering` | `recast-radar-scattering` | | |
| `serde` | | | | |
| `full` | | | `io` `write` `net` `correct` `filters` `retrieve` `map` `track` `render` `scattering` `serde` | |
<!-- features:end -->

- In the Default column, "yes" means the feature is listed in `default`, and
  "via `io`" means `io` turns it on.
- A feature also enables the features of the member crates its crate depends
  on, so the types a module's API uses can be named through the facade.
- `nexrad` alone gives the Level II decoder without the other formats or the
  router. To take only some modules, turn the defaults off, for example
  `default-features = false, features = ["nexrad", "correct"]`. Each member
  crate can also be used on its own.
- `net` (also part of `full`) is the only feature that makes HTTPS requests
  and the only one that compiles C (see [Pure Rust](#pure-rust)).
- `serde` turns on `recast-radar-core/serde`: `Serialize` and `Deserialize`
  for the data model (`Volume`, `Sweep`, `Field`, names and codings; field
  names serialize as their FM301 spelling, `"DBZH"`). Without it the model
  does not depend on serde. `recast-radar-data` and `recast-radar-scattering`
  use serde for their own file formats, so `net` and `scattering` still
  compile it.
- `crates/recast-radar-tools/tests/readme.rs` checks this table against the
  facade's `[features]` and `src/lib.rs`. It also checks the dependency rule,
  and its exception, against every dependency table of the member crates'
  manifests.

## Pure Rust

Without `net`, nothing in the build compiles C or C++, for any target:

- Level II bzip2 (LDM records and whole-file) through `recast-radar-bzip2`,
  this repository's decoder without unsafe code; Level III bzip2 through the
  same crate; gzip and zlib through `flate2` with `zlib-rs`.
- HDF5 (for ODIM_H5 and netCDF-4, through `recast-radar-hdf5`) and classic
  netCDF (for CfRadial 1) are read by parsers in this repository, not by the C
  libraries.
- chrono is built with its `now` feature instead of `clock`. `clock` would add
  `iana-time-zone`, which compiles C++ when the target is Haiku.

The one exception is `net`: `recast-radar-data` uses reqwest with rustls, and
its crypto provider, `ring`, compiles C and assembly. The test-only
`recast-radar-testdata` also uses rustls (through ureq) to download test
files, behind its default `download` feature; the facade's examples use it
without that feature, so `cargo run --example` builds no C unless the example
needs `net`. CI runs [`tools/ci/pure-rust-check.sh`](tools/ci/pure-rust-check.sh).
It fails if `cc` or `cmake` is in the dependency graph, for any target, of
the facade with every feature except `net` and `full`.

The same split applies to WebAssembly. Every library crate except
`recast-radar-data` (which builds for wasm32 without its `net` feature) and
the test-only `recast-radar-testdata` passes
`cargo check --target wasm32-unknown-unknown`, and so do the facade with any
single feature other than `net` and `full`, and the benchmark binary. The
command-line tool and the Python extension module are native only. CI
checks this with [`tools/ci/wasm-check.sh`](tools/ci/wasm-check.sh). On that
target, use the byte-slice entry points (such as
`nexrad::read_volume_from_bytes`), because the path-based ones return I/O
errors. rayon runs everything on the calling thread there. Details:
[docs/design/wasm.md](docs/design/wasm.md).

## No unsafe

The workspace sets `unsafe_code = "forbid"`, and every crate opts in with
`[lints] workspace = true`. There are no exceptions:

<!-- lint-exceptions:start -->
<!-- lint-exceptions:end -->

That includes the Python bindings (`recast-radar-py`): PyO3 0.29's
`#[pymodule]`, `#[pyfunction]`, `#[pyclass]` and `#[pymethods]` expand to code
the forbidden lint accepts, and field buffers reach NumPy through
`numpy::PyArray::from_vec`, which takes ownership of the `Vec` without
`unsafe` (design note `docs/design/fm301-model.md` 12.2).

Library code also denies `clippy::unwrap_used` and `clippy::expect_used`, and
the workspace denies `missing_docs`: every public item is documented.
`crates/recast-radar-tools/tests/readme.rs` checks the exception list above
against the crate manifests.

## Tests and CI

`cargo test --workspace` runs the tests. New tests read real radar files only.
Some tests carried over from the original code base still build synthetic
input, and they will be converted to real files. The real files are listed in
`testdata/manifest.toml` and `testdata/*/manifest.toml`. Small ones are
committed under `testdata/files/`. The others are downloaded on first use into
a cache (`$RECAST_RADAR_TESTDATA`, by default in the user's cache directory)
and checked against their SHA-256.

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs:

- rustfmt, and clippy with `-D warnings` on every target and feature, on
  Rust 1.94.0, the minimum supported version;
- the workspace tests on the latest stable Rust on Linux, Windows and macOS,
  plus the `serde` round-trip test, and the workspace tests on Rust 1.94.0 on
  Linux, keeping the testdata download cache between runs;
- the documentation build with warnings denied, and `cargo package` for every
  crate (a dry run: nothing is uploaded);
- a `cargo hack check` of each facade feature on its own, and
  [`tools/ci/pure-rust-check.sh`](tools/ci/pure-rust-check.sh);
- the wasm32 check in [`tools/ci/wasm-check.sh`](tools/ci/wasm-check.sh);
- [`tools/ci/level2-golden-check.sh`](tools/ci/level2-golden-check.sh), which
  checks that MetPy 1.7.1 and Py-ART 2.2.5 reproduce the Level II goldens
  byte for byte;
- [`tools/check_example_outputs.py`](tools/check_example_outputs.py), which
  runs the examples whose output this README and the user guide show and
  fails when a shown output differs;
- the fuzz harness and the replay of every fuzz regression input on stable,
  and a 60-second run of every fuzz target on nightly
  ([`tools/ci/fuzz-smoke.sh`](tools/ci/fuzz-smoke.sh)).

[`.github/workflows/python-wheels.yml`](.github/workflows/python-wheels.yml)
is set to build the Python package as abi3 wheels for Linux (manylinux2014),
Windows and macOS (Apple silicon and Intel) on pushes to `main` and on
manual dispatch, run the pytest suite against xradar, Py-ART and MetPy with
each wheel (Python 3.10 and 3.12 on Linux), and keep the wheels as workflow
artifacts. A version tag (`v*`) also publishes the wheels and a source
distribution to PyPI as `recast-radar`, through trusted publishing. The suite
(`crates/recast-radar-py/pytests`) also runs locally after
`maturin develop` or with a built wheel installed.

[`.github/workflows/cli-binaries.yml`](.github/workflows/cli-binaries.yml)
is set to build the `recast-radar` command with the release profile for
Linux, Windows and macOS (Apple silicon and Intel) on pushes to `main` and on
manual dispatch, run it on committed test files, and keep the binaries as
workflow artifacts, which signed-in GitHub users can
download. Nothing is released.

Neither of these two workflows has run on GitHub yet: the branch that adds
them has not been pushed. What was checked without GitHub: both pass
`actionlint`; the Windows x64 binary and wheel build; the Linux binary builds
with the release profile (fat LTO) on stable Rust and passes the workflow's
smoke commands (Ubuntu 24.04); the manylinux2014
wheel builds in the maturin manylinux2014 image, and the pytest suite passes
with it on Linux under Python 3.12 (all of it) and 3.10 (without the xradar
and Py-ART comparisons, whose pinned readers need 3.11). The macOS jobs
(native Apple silicon, and the x86_64 cross build on the Apple silicon
runner) have not run anywhere.

## License

Licensed under either of MIT ([LICENSE-MIT](LICENSE-MIT)) or Apache-2.0
([LICENSE-APACHE](LICENSE-APACHE)) at your option. Each crate's package
carries both files.
