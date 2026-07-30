# rustwx-models

`rustwx-models` is the model registry and source-planning crate.

## Responsibilities

- built-in model definitions
- source ordering and source metadata
- URL builders
- latest-run probing
- forecast-hour availability probes
- plot-recipe fetch planning

## Built-in models

- `HRRR`
- `GFS`
- `ECMWF IFS Cycle 50r1 open data`
- `ECMWF AIFS Single v2 open data`
- `RRFS-A`

## What is implemented

- URL resolution for all built-in models, including ECMWF IFS/AIFS operational
  `oper` and `wave` streams
- NOAA-style latest/probe/hour checks
- recipe planning for selector-backed upper-air plots
- direct-plot recipe registry coverage for:
  - pressure-level height/wind and temperature/height/wind products
  - near-surface thermodynamics and wind-combo products
  - direct surface/column products like MSLP, PWAT, cloud cover, and visibility
  - native radar products like 1 km reflectivity and composite reflectivity
- registry-owned selector/model support policy for wired recipes
- explicit support/blocker reporting when a recipe is not wired for a model

## Current limits

- ECMWF open data is whole-file fetch only because `.idx` sidecars are not
  published with the feed
- recipe coverage is not uniform across all models
- many direct surface/radar recipes are cataloged before extractor/render support
  is complete, so they currently resolve to explicit blockers rather than fetch plans
- some native convective and severe recipes are still HRRR/RRFS-A only

## Minimal example

```rust
use rustwx_core::{CycleSpec, ModelId, ModelRunRequest};

let request = ModelRunRequest::new(
    ModelId::Gfs,
    CycleSpec::new("20260414", 18)?,
    0,
    "pgrb2.0p25",
)?;
let urls = rustwx_models::resolve_urls(&request)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```
