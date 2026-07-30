# rustwx-render-verify

`rustwx-render-verify` is a standalone verification crate for the weather-native plot engine direction.

It is not a normal workspace member. It exists so rendering behavior can be tested in isolation against the shipped `rustwx-render` crate without pulling the whole `rustwx` workspace into every check.

The intended split is:

- product layers own data fetch, crop/window choice, units, and science
- `rustwx-render` owns request shaping, palettes, overlays, layout, and PNG/image output
- this crate verifies that render-facing contract with synthetic but realistic requests

## What it covers

- synthetic filled-map requests on projected domains
- contour-only height/wind overlays
- contour-fill/line alignment on a shared scalar field
- projected contour topology on a slanted synthetic domain
- mixed panel composition
- PNG save/write smoke coverage
- runnable examples for quick visual checks

## Current limits

- this is still a narrow harness, not a second production API
- the fixtures are synthetic; end-to-end product proofs still live under `proof/`
- projected contour verification currently exercises the public `projected_domain + contours`
  request path; if a dedicated projected contour-fill API lands, extend this harness through that
  public surface instead of reaching into renderer internals

## Commands

```powershell
cargo test --manifest-path crates/rustwx-render/verify/Cargo.toml
cargo run --manifest-path crates/rustwx-render/verify/Cargo.toml --example synthetic_sbecape
cargo run --manifest-path crates/rustwx-render/verify/Cargo.toml --example synthetic_panel
cargo run --manifest-path crates/rustwx-render/verify/Cargo.toml --example synthetic_contour_alignment
```

The examples write smoke PNGs under `target/rustwx-render-verify/` at the workspace root and
print the output path.
