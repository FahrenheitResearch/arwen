# rustwx-calc

`rustwx-calc` is the Rust-first diagnostics layer. It wraps `metrust` for gridded severe and thermodynamic products and exposes APIs shaped for Rust callers.

## What is implemented

- ECAPE-family grid APIs
- SB/ML/MU ECAPE triplet APIs
- failure-mask variants for ECAPE verification/debug
- CAPE/CIN wrappers, including explicit `SB` / `ML` / `MU` convenience entrypoints
- 2 m surface thermo wrappers for dewpoint, RH, theta-e, heat index, wind chill, and apparent temperature
- lifted index and lapse-rate wrappers (`700-500 mb`, `0-3 km`)
- temperature-advection wrappers for generic 2-D layers plus `700 mb` / `850 mb` aliases
- conservative window reducers for multi-hour sums and run-max operations
- SRH and bulk shear wrappers, including explicit `0-1 km`, `0-3 km`, and `0-6 km` helpers
- EHI wrappers with explicit `0-1 km` and `0-3 km` depth distinctions
- STP, SCP, EHI, local-proxy SHIP, and BRI wrappers
- bundled "supported severe" proof outputs

## Important notes

- the fixed STP path is the real fixed-layer form
- the current bundled SCP/EHI proof path is intentionally conservative
- the `0-1 km` / `0-3 km` EHI helpers are explicit depth wrappers around the same underlying EHI math; they are not effective-layer diagnostics
- the 2 m wind-chill wrapper uses 10 m wind speed because that is the standard input carried by the shared surface bundle
- the `700 mb` / `850 mb` temperature-advection wrappers are just explicit aliases around the same horizontal temperature-advection kernel; they do not change the underlying math
- the current SHIP wrapper matches the local `wrf-rust` hail-proxy formula and should not be treated as a canonical SHARPpy-style SHIP implementation yet
- full effective-layer severe support still depends on broader upstream profile logic

## Current limits

- this crate does not ingest model data
- it assumes the caller already has the required grid and profile inputs
- some severe products are still "supported proof fields" rather than final operational APIs

## Minimal example

```rust
use rustwx_calc::{
    EcapeGridInputs, EcapeOptions, GridShape, SurfaceInputs, compute_2m_theta_e, compute_ecape,
};

let ecape = compute_ecape(inputs, &EcapeOptions::default())?;
let theta_e = compute_2m_theta_e(
    GridShape::new(1, 1)?,
    SurfaceInputs {
        psfc_pa: &[100000.0],
        t2_k: &[303.15],
        q2_kgkg: &[0.014],
        u10_ms: &[5.0],
        v10_ms: &[0.0],
    },
)?;
# let _ = ecape;
# let _ = theta_e;
# Ok::<(), Box<dyn std::error::Error>>(())
```
