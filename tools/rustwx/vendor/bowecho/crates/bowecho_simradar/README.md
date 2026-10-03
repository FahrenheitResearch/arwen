# BowEcho simulation library

This crate exposes the native virtual radar without a desktop UI dependency.
The reference source is BowEcho commit
`eea07fc0a032d4de439d0bfe2def9098afd8b5a8`.

The app and library compile the same `wrf_radar_core.rs`. The app keeps its
interactive acquisition adapters and UI tests. Pure physics, estimator,
validation, temporal, and scattering modules live in this crate and the app
reexports them. `model_grid.rs` is shared with the app map sampler.

- `ModelRadarFields` accepts normalized physical arrays from any model.
- `WrfRadarFields::from_model_fields` validates array lengths and builds the
  existing geolocation lookup.
- `read_wrf_radar_fields_for_config` reads WRF-format history and populates
  scheme-supported microphysical fields.
- `try_build_synthetic_volume` produces one `radar_core::RadarVolume`.
- `build_synthetic_volume_reporting_temporal` uses the existing adjacent-scene
  plan, with the existing no-extrapolation and missing-neighbor rules.
- `all_sites`, `SiteKind`, and `SiteRef` expose the existing typed site catalog.

Use `SyntheticRadarComputePreference::Cpu` to prohibit GPU selection. Construct
`SyntheticRadarConfig::default()` and set public fields as needed. The generic
input expects horizontal Y/X arrays and volume Z/Y/X arrays, metres MSL, dBZ,
and earth-relative winds in metres per second. Velocity toward the antenna is
negative.

The local extraction also fixes the scattering crate's JSON feature declaration.
Its LUT contract compares binary header axes with JSON axes bit for bit, so
`serde_json/float_roundtrip` is required explicitly instead of relying on feature
unification from the desktop dependency tree.

The library and required workspace crates use MIT OR Apache-2.0. Keep the root
licence files and the Py-ART notice with vendored sources. Research scattering
LUTs are downloaded and hash-checked by the existing asset loader on demand;
they are not included by this crate. Their existing notices and applicability
checks are retained. No CUDA build or runtime invocation is needed for CPU
simulation.
