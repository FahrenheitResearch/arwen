# rustwx-sounding

`rustwx-sounding` bridges generic sounding columns into `sharprs` and renders sounding PNGs.

## What is implemented

- validated generic sounding column types with conservative physical QC
- conversion to `sharprs` profiles
- native sounding rendering
- rustwx-owned sounding table/title rendering with Source Sans text
- verified SB/ML/MU analytic ECAPE and paired NCAPE table values from
  `ecape_rs::calc_ecape_ncape`
- optional external ECAPE annotation block that can be appended to the rendered product

## Input validation

`SoundingColumn::validate()` checks the basic shape constraints and also rejects:

- non-finite values in the required profile vectors and optional omega vector
- pressure profiles that are not monotonic non-increasing
- height profiles that are not monotonic non-decreasing
- dewpoints that exceed temperature, while still allowing saturated levels

## Important note

`sharprs` is not being misrepresented here as an ECAPE engine. Analytic ECAPE
and paired NCAPE values shown in the native table are computed by
`rustwx-sounding` through `ecape_rs::calc_ecape_ncape`; `sharprs` remains the
profile/legacy sounding calculation dependency. The table does not publish the
post-path `calc_ecape_parcel(...).ecape_jkg` field as standard ECAPE.

## Current limits

- no direct model/observation ingest
- TEHI, TTS, SigSvr, and LHP are table placeholders until rustwx has verified native formulas for them
- sounding input assembly still belongs to higher-level crates

## Minimal example

```rust
use rustwx_sounding::write_full_sounding_png;

write_full_sounding_png(&column, "sounding.png")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```
