# Native WRF initialization: public-release acceptance

This repository is an internal integration tree, not a public release.  The
goal of the native initialization product is a standalone, reproducible tool
that turns supported raw meteorological data plus standard WRF namelists and
`WPS_GEOG` into stock-WRF-ready inputs without running WPS or `real.exe`.
Passing one proof case is not completion.

The machine-readable statement of current support is
`gpuwm/native_wrf_support_v1.json`.  Unsupported choices must fail with an
actionable error; no source, physics, projection, or initialization scheme may
be silently substituted.

## Required end-state interface

One documented command must accept:

- `namelist.wps` and `namelist.input`;
- a supported raw meteorological source or a resumable source manifest;
- a `WPS_GEOG` installation or a provenance-bound static cache; and
- an output directory and explicit CPU/CUDA backend policy.

It must atomically emit every required `wrfinput_d0N`, root `wrfbdy_d01`, any
required lower-boundary auxiliary files, and a deterministic receipt binding
all inputs, resolved defaults, substitutions, outputs, versions, and hashes.
Normal runtime must not invoke `geogrid.exe`, `ungrib.exe`, `metgrid.exe`, or
`real.exe`.

## Acceptance gates

- [x] Extract a model-independent package boundary. The dedicated `rw-wps`
  wheel has a separate distribution identity, omits forecast/model/controller
  entry points and forecast-only data, and passes an isolated no-CuPy import
  gate plus a clean Linux CPU install. Generic initialization state shared
  with the preprocessor remains under the internal `gpuwm.*` namespace.
- [ ] Add a clean-clone CPU-only build/install test on Linux and Windows.
- [ ] Add a clean-clone CUDA build/install test on a documented CUDA baseline.
- [x] Make the all-Rust GRIB decoder and its pinned provenance reproducibly
  buildable offline from the release source tree. At commit `15c0923`, two
  independently located clean checkouts with different absolute path lengths
  produced byte-identical complete Linux archives (SHA-256
  `a09c473559de19e39c0cdab84eb1bc381f9ccab3a1f25d9d58809e7286f5384d`),
  and the clean CPU install receipt passed (SHA-256
  `6e459d52200f4de93eeb1a5854ecc13fcc9ed1b535e251161e895e44cd99af56`).
  This is a same-host reproducibility gate; a second-machine gate remains.
  The reconciled implementation commit `d8c73e0`, including both 20CR routes,
  repeated
  this gate with complete archive SHA-256
  `366405e5674b64233eb5ac32ae74a06f33e604a79e921ea6fda54169bf4649bd`
  and clean-install receipt SHA-256
  `624170956ec1981ad15bfad1442881657d6481f123bcc975094723e358aee96b`.
- [ ] Support arbitrary static one-way Lambert hierarchies accepted by the
  compatibility matrix: movable centers, sizes, spacings, parent ratios,
  starts, times, forcing cadences, and supported explicit vertical grids.
- [ ] Build every domain's static fields directly from `WPS_GEOG`; remove the
  requirement for prebuilt private `static.npz` inputs.
- [ ] Run source decode and independent per-domain interpolation concurrently,
  with deterministic worker limits, bounded memory, cancellation, and a
  sequential dependency phase only where WRF child initialization requires it.
- [ ] Emit `wrfinput_d01..dNN` and root `wrfbdy_d01`; validate hierarchy,
  staggered dimensions, domain metadata, finite state, cadence, and hashes
  before atomically publishing the directory.
- [ ] Cover all state inventories required by every physics combination marked
  supported, including soil layers, hydrometeor/number fields, PBL/TKE fields,
  radiation gases, lake/sea-ice state, and restart-sensitive setup.
- [ ] Add unchanged-stock-WRF acceptance for at least two locations, two
  horizontal geometries, two supported vertical grids, two forcing cadences,
  and a nested hierarchy; each gate must complete model steps, not merely open
  the NetCDF files.
- [ ] Add deterministic CPU-versus-CUDA numerical reports with frozen fieldwise
  tolerances and non-finite mismatch checks.
- [ ] Add resumable downloads, content-addressed caches, checksum validation,
  retry/backoff, stale-part cleanup, and disk/memory preflight.
- [ ] Add redistributable small fixtures and eliminate tests that require a
  developer Downloads directory, rented-node paths, or private case bundles.
- [x] Add a root README, install guide, CLI reference, compatibility/migration
  guide, reproducible benchmark recipe, changelog, contribution guide,
  security policy, and release checklist.
- [ ] Remove or quarantine machine paths, node addresses, credentials, private
  evidence, case-specific controller notes, and unrelated forecast-model code
  from source distributions and wheels.
- [ ] Record third-party source, data, and algorithm attribution; audit all
  bundled tables, WRF-derived declarations, and Rust/CUDA components.
- [x] Owner authorized Apache-2.0; the root license and package metadata now
  carry that selection. Third-party attribution and redistribution auditing
  remain separate blocking work.
- [ ] Produce reproducible sdist/wheel/container artifacts, SBOM, dependency
  lock, checksums, signatures, and a clean-room install receipt.
- [ ] Complete API/CLI threat review: path traversal, archive extraction,
  malformed GRIB/NetCDF/namelist input, resource exhaustion, and overwrite
  semantics must fail safely.

## Internal milestone ledger

### M0: one-domain direct export

Implemented and proof-tested for the narrow source/physics slices enumerated in
the support matrix.  This established that native preprocessing can produce
files accepted by unchanged stock WRF.  It is not a public product boundary.

### M1: namelist-bound hierarchy final export

The exporter can now validate all WPS/experiment domain geometries, consume a
strict relocatable per-domain artifact manifest, and atomically write
`wrfinput_d01..dNN` plus root `wrfbdy_d01` with per-domain WRF nest metadata.
This is the final-export half of multi-domain operation.  Per-source creation
of all domain artifacts, parallel scheduling, child terrain/state dependency
handling, and unchanged-stock-WRF nested acceptance remain release blockers.
