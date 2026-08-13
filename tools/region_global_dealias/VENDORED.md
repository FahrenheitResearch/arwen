# Vendored: `region-global-dealias`

This directory is a **verbatim copy** of Drew's crate, not a fork.

| | |
|---|---|
| Upstream | <https://github.com/FahrenheitResearch/region-global-dealias> |
| Commit | `a7d4baf6b8a11ca5602fe44a533efd8200ef6cea` |
| Crate version | `0.2.0` |
| Vendored | 2026-08-12 |

## Why vendored rather than a git dependency

The house offline-build pattern: every Rust artifact gpuwm needs is built
from a tree inside the checkout with `cargo build --release --locked
--offline`, so a build never depends on the network being up or on a
remote ref still pointing where it did. `tools/grib1_bridge/vendor/` does
the same for `grib-core`. The recorded commit above is what makes the copy
auditable: `git diff` against that ref in the upstream clone is the proof
that nothing here diverges.

The crate has **no dependencies**, so nothing else came with it.

## What was and was not copied

Copied: `Cargo.toml`, `Cargo.lock`, `src/`, `tests/`, `test/fixtures/`,
`examples/`, `region_global_dealias.h`, and the licence and notice files.

Not copied: the WebAssembly build outputs (`dealias.wasm`,
`wasm-inline.mjs`), the JavaScript wrapper and worker pool (`index.mjs`,
`pool*.mjs`, `package.json`, type declarations), the browser demo, the
build script and the CI workflow. gpuwm drives the **native C ABI**
(`bw_dealias`, `bw_dealias_rift_v1`) through `ctypes`, so the JavaScript
half of upstream is not part of this integration. Dropping it is a
subtraction, never an edit: no file kept here differs from upstream by a
byte.

## Nothing in here is modified

The solver, the FFI layer and the tests are Drew's. Changes belong
upstream and arrive here by re-vendoring at a newer commit and updating
the sha above, `gpuwm.obs.dealias_region.UPSTREAM_COMMIT`, and the
Py-ART parity receipt. `tests/test_obs_dealias_region.py` pins the ABI
version and the three structure layouts, so a re-vendor that changes the
contract fails a test rather than returning different numbers.

## Building

```
cd tools/region_global_dealias
cargo build --release --locked --offline
```

produces `target/release/region_global_dealias.dll` (`.so` / `.dylib`
elsewhere), which `gpuwm.obs.dealias_region` finds on its own -- or set
`GPUWM_DEALIAS_REGION_BRIDGE` to it.

`cargo test --release` runs upstream's own suite (45 tests at this
commit, including the real-sweep RIFT cases in `tests/rift_real_cases.rs`
and its pinned golden checksum).

## Licence

`(MIT OR Apache-2.0) AND BSD-3-Clause`. The solver is a derivative of
Py-ART, whose BSD-3-Clause terms travel with it and cannot be removed:
`PYART-LICENSE.txt` must ship with any redistribution of this directory,
in source or compiled form. See `THIRD-PARTY-NOTICES.md` for the full
notice, the list of modifications the port makes relative to Py-ART, and
the citation to use in published work (Helmus & Collis 2016).
