# Freezing rain and ice accretion

Native forecast ensemble output includes these fields by default when the
ordinary output supplies their inputs:

| Field | Stored units | Default probability |
| --- | --- | --- |
| temperature2_below_freezing | K | T2 < 273.15 K |
| freezing_rain_rate | mm h-1 | >= 0.1, 1, 5 mm/h |
| freezing_rain_total | mm liquid equivalent | >= 1, 5, 10 mm |
| ice_accretion | mm equivalent radial ice | > 2.54, 6.35, 12.7 mm (0.10, 0.25, 0.50 in) |
| ice_accretion_horizontal | mm elevated horizontal ice | > 2.54, 6.35, 12.7 mm |

The ice fields are **saved-frame accumulations**, evaluated at the right
endpoint of each interval. They are not per-model-step accumulations. The
NetCDF attribute ice_accretion_sampling and ensemble manifest retain this
method, its input and missing-value policy, and interval lengths in hours.
A first sample after the forecast start cannot reconstruct prior type or
temperature, so its cumulative ice is missing. A missing input sample or a
counter reset invalidates the cumulative ice rather than producing a zero.

For each interval the liquid precipitation is the difference of RAINNC +
RAINC + RAINSH. The native precipitation-type classifier uses the lowest-level
microphysics fall fluxes (rain 5, snow 1, graupel 3 m/s), a total flux of at
least 0.1 mm/h, and a liquid fraction of at least 0.7. The surface gate is
T2 < 273.15 K or, when supplied, TSK < 273.15 K. A cold surface alone does
not convert snow or ice pellets into freezing rain. Missing QSNOW is not
interpreted as rain.

FRAM uses interval-average liquid rate, endpoint 10 m wind magnitude and a
pressure-aware psychrometric wet-bulb temperature from T2/Q2/PSFC. Predictor
units are inches/hour, knots and Celsius. Equations (1)-(8) from the 2016
NWS FRAM paper are implemented independently in Rust and CUDA. The cold
predictor is capped at -7 C. Horizontal ice is converted to equivalent
radial ice by 0.394. Wet-bulb temperatures >= 1 C yield zero ice.
The published calibration used hourly continuous freezing rain with
precipitation >= 0.5 mm/h. Other intervals and rates are extrapolations.
This is gross elevated accretion with no melting, shedding, sublimation,
ground icing or structural-load calculation.

The public [ensemble] request accepts comparison = "ge", "gt", "le" or
"lt", applying it to all threshold fields. A field table overrides only its
named fields, preserving the native defaults on other fields:

```toml
[ensemble]
members = 36
[ensemble.thresholds]
temperature2 = [273.15]
[ensemble.comparison]
temperature2 = "lt"
```

The original below-freezing comparison request now produces less-than
probabilities. Unknown fields and comparison names fail validation before
the run. The existing temperature2 field retains its default greater-than-
or-equal comparisons. All thresholds are rounded once to float32, matching
the resident input words and strict boundary semantics.

Ice maps and probabilities use rw_wrfbatch through gpuwm.rustwx. Ice maps
display inches on fixed steps including 0.10, 0.25 and 0.50; probability
titles display those same inch thresholds. Comparisons and stored units
remain explicit in the native data.

Source: NOAA-hosted original paper, DOI 10.1175/WAF-D-15-0118.1:
https://repository.library.noaa.gov/view/noaa/15309/noaa_15309_DS1.pdf
No external implementation was copied, and the paper is not redistributed.
The implementation independently expresses its published mathematical model.
NOAA's repository copyright policy distinguishes government works from
copyrighted journal content; repository availability alone is not a blanket
redistribution licence. The Rust implementation uses the workspace MIT
licence; CUDA and Python driver changes use the engine Apache-2.0 licence.

## Observation binding

`python -m gpuwm.ensemble.ice_accretion verify REQUEST.json OUTDIR` (the
packaged `rw_ice_verify` Rust binary) binds the native ensemble NetCDF files to
ASOS CSV reports and an optional mPING archive export. The request names
`frames`, `asos_csv`, `mping_json` and `max_distance_km` (default 15 km).
IEM CSV input must include station, valid, lat, lon and wxcodes or metar;
tmpf and ice_accretion_1hr are optional. mPING input accepts a JSON report
array or an API results array, with descriptive labels, observation_time or
obtime, and GeoJSON geometry/geom coordinates. Numeric mPING codes without
their authoritative label table are refused.

The binding groups ASOS reports by station and native output interval and
records whether any report contains FZRA. It samples the first native
freezing_rain_rate probability threshold (>= 0.1 mm/h), using the nearest
native point within the requested distance. It retains every pair, source
and product SHA-256, matching policy, observed-positive count and Brier
score by source. This is an occurrence check over the labelled sampling
interval; it does not validate a numerical ice amount. Below-freezing
probabilities bind only to temperature reports within 10 minutes of the
frame. ASOS hourly horizontal ice, when supplied, binds only to an hourly
model horizontal-ice increment with an observation within 10 minutes of
its endpoint. The output never converts an ASOS horizontal amount into an
observed radial amount. Missing mPING or no matched reports yields PARTIAL.

The Rust observation binary is built by `cargo build --release -p rw-obs
--bin rw_ice_verify` in tools/rustwx. The public mPING archive requires
access approval. A dated export and its data terms are required to close
that part of verification; access to the live viewer alone is insufficient.
