# rustwx-core

`rustwx-core` defines the shared domain model for the workspace.

## Responsibilities

- model and source identifiers
- cycle and timestep request types
- grid and field containers
- canonical field names
- vertical selectors
- validation and common errors

## What is implemented

- `GridShape`, `LatLonGrid`, and typed 2D/3D field containers
- `CanonicalField` and `FieldSelector`
- model/source/time request types used by the fetch and registry layers
- semantic selector types consumed by the registry and I/O layers
- a growing direct-product vocabulary for:
  - upper-air pressure fields
  - near-surface thermodynamics and winds
  - column/surface fields like MSLP, PWAT, cloud cover, and visibility
  - native radar/convective fields like reflectivity and UH

## Current limits

- semantic selector coverage is broader than extractor support; `rustwx-models`
  and `rustwx-io` still decide what is actually fetchable today
- projection metadata is still lighter than the eventual end-state
- this crate does not know anything about fetch or rendering

## Minimal example

```rust
use rustwx_core::{CanonicalField, FieldSelector};

let selector = FieldSelector::isobaric(CanonicalField::Temperature, 500);
assert_eq!(selector.to_string(), "temperature_500_mb");
```
