# ecape-rs

`ecape-rs` is a Rust implementation of the `ecape-parcel`-style full
entraining parcel-path ECAPE calculation. Its job is compatibility and speed:
reproduce the public Python `ecape-parcel` parcel-path behavior closely enough
for science work, then make the calculation cheap enough for HRRR grids,
large profile catalogs, and archive-scale verification.

This crate is not a new ECAPE theory and does not replace the Peters analytic
ECAPE formulation. The intended lineage is:

- Peters et al. provide the ECAPE theory and analytic approximation.
- `ecape-parcel` provides a public Python parcel-path implementation checked
  against that framework.
- `ecape-rs` provides a high-throughput Rust implementation of the same class
  of parcel-path calculation.

## What It Computes

- Surface-based, mixed-layer, and most-unstable parcel sources.
- Entraining and non-entraining parcel ascents.
- Pseudoadiabatic and irreversible ascent modes.
- Density-temperature parcel paths, buoyancy, CAPE/CIN/LFC/EL-style path
  diagnostics, and ECAPE-family grid wrappers used by `rustwx`.
- Storm-motion options: `right_moving`, `left_moving`, `mean_wind`, and
  `user_defined`.

Internally, winds use Cartesian components: `u > 0` eastward and `v > 0`
northward.

## Validation Status

Current validation is centered on direct parity with Python `ecape-parcel`, not
comparison to a separate JavaScript implementation.

The parity harness compares raw 20 m parcel paths and scalar outputs between
Python and Rust, including:

- parcel pressure, height, temperature, water vapor, total water, and density
  temperature;
- environmental density temperature and buoyancy along the path;
- integrated parcel-path energy;
- CIN, LFC, EL, zero/nonzero path behavior, and first divergent step;
- runtime distributions.

The latest `rustwx` validation sweep using this crate reports:

- complete no-entrainment-limit checks for SB/ML/MU parcels under
  pseudoadiabatic and irreversible ascent;
- a 252-profile / 3,024-configuration first event-sample parity sweep with no
  first-stage failures;
- a bounded random/stress/storm-motion v2 sweep with 5,368 comparable
  configurations, zero first-stage parity failures, and eight Python reference
  exceptions where `ecape-parcel` did not return a parcel path.

First-stage parity criteria are:

- `|delta E_parcel| < 2 J kg^-1`;
- `max |delta T_rho| < 0.01 K`;
- no zero/nonzero LFC, EL, empty-path, or path-energy mismatch.

These tolerances are intended to validate implementation compatibility. They
are not operational severe-weather thresholds.

## Performance Role

`ecape-rs` is designed for package-level throughput in workflows where Python
parcel-path calls are too slow for grid-scale diagnostics. In `rustwx`, the
solver is used to generate HRRR ECAPE, ECAPE/CAPE ratio, ECAPE-EHI, ECAPE-STP,
and related research fields over cropped HRRR swaths and profile catalogs.

The practical goal is to move full parcel-path ECAPE from individual sounding
workflows to map-scale and archive-scale research.

## Build

```bash
cargo build --release
```

## License

MIT
