# rustwx-io

`rustwx-io` is the fetch, probe, extract, and cache layer.

## Responsibilities

- source probing
- forecast-hour discovery
- full-file and indexed fetch paths, depending on model/operator policy
- cached fetch results
- selector-backed GRIB extraction
- cached extracted fields

## What is implemented

- live source probes for supported models
- cached byte fetches for both full-family files and indexed subsets
- structured GRIB extraction for the selector subset used by current proofs and
  the current direct catalog, including:
  - 200/300/500/700/850 hPa height, temperature, RH, absolute vorticity, and winds
  - 2 m AGL temperature, dewpoint, and relative humidity
  - 10 m AGL u/v wind plus operational 10 m gust matching
  - MSLP, precipitable water, total cloud cover, and visibility
  - native composite reflectivity plus 1 km AGL reflectivity where present
- batch selector extraction from one parsed GRIB where possible
- field cache layout organized by model/date/cycle/fhr/product/source/patterns
- shared cached grid geometry so repeated fields from the same timestep do not
  re-serialize identical lat/lon arrays per field

## Current limits

- extraction is still selector-driven rather than a broad general decoder
- volume-level APIs are not yet the default path
- simulated IR remains intentionally unsupported here until the GRIB signature is
  pinned down cleanly enough to keep selector semantics exact
- relative vorticity, lightning, smoke, and derived surface fields still need
  their own explicit selector/provenance work
- the fetch/index helpers are vendored in-repo for self-contained
  builds while the longer-term internal GRIB contract continues to
  evolve

## Minimal example

```rust
use rustwx_core::{CanonicalField, FieldSelector};
use rustwx_io::{extract_field_from_bytes, extract_fields_from_bytes};

let selector = FieldSelector::isobaric(CanonicalField::Temperature, 500);
let field = extract_field_from_bytes(&bytes, selector)?;
let fields = extract_fields_from_bytes(
    &bytes,
    &[
        FieldSelector::isobaric(CanonicalField::Temperature, 500),
        FieldSelector::isobaric(CanonicalField::UWind, 500),
        FieldSelector::isobaric(CanonicalField::VWind, 500),
    ],
)?;
# let _ = field;
# let _ = fields;
# Ok::<(), Box<dyn std::error::Error>>(())
```
