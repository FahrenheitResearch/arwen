# Stateful fire-danger diagnostics

`python -m gpuwm.fire_danger_state run REQUEST.json [--render]` launches
`rw_fire_indices`. Rust owns all weather/NetCDF/CSV/bzip2 I/O, scientific
calculations and state. CUDA accelerates Nelson radial diffusion by default
when available; the receipt identifies the actual backend. Python orchestrates.

Products: Canadian FFMC, DMC, DC, ISI, BUI, FWI; KBDI; NFDRS dead fuel moisture
(1/10/100/1000 hour), live herb/wood moisture, GSI, ERC, BI and SC. All seventeen
have native `rw_wrfbatch` CF product rows, physical levels and maps through
`gpuwm.rustwx`. Moisture outputs use percent of dry mass; GSI and indices use 1.

Build in `tools/rustwx`: `cargo build --release -p fire-indices`.
Set `GPUWM_FIRE_INDICES` for an external target directory. Normal packaging uses
the registered native artifact and ABI contract.

A run request supplies ordered `inputs`, `member`, `utc_offset_hours` (fixed
standard time), `annual_rain_source`, exactly one of `annual_rain_mm` or
`annual_rain_grid`, and distinct `output`, `state_out`, `receipt` paths. A new
state requires explicit `initial_kbdi` (0..800 hundredths of an inch).
`initial_fwi` defaults to FFMC 85, DMC 6, DC 15. Nelson initializes each radial
node to 20 percent dry-mass moisture. Spin up over preceding weeks before using
forecast-day output. Use `state_in` to continue. Explicit ensemble forks name
the old member in `fork_member` and a new `member`; parent hashes are retained.
Version-2 state carries all four 21-node profiles, temperatures, wet-event
history, pending hourly weather, GSI and precipitation windows and annual-herb
phenology. Older FWI-only state is rejected rather than inventing Nelson state.

Fields are [time,latitude,longitude]: T2/D2 in K, U10/V10 in m s-1, TP in m and
SSRD in J m-2, both accumulated over the **preceding one hour**, never running
forecast accumulations. Sub-zero SSRD packing noise down to -0.5 J m-2
is clamped to zero and counted in the receipt; larger negatives are rejected. `time` is interval-end seconds since
1970-01-01 00:00:00. Regular geographic axes use longitude -180..180. Initial
interval ends one hour after standard noon; all subsequent hours must be
contiguous across files and runs. Daily valid time is standard noon. FWI uses
noon weather and 24-hour rain; KBDI uses noon-to-noon maximum temperature/rain.
NFDRS uses hourly Nelson moisture and updates live fuels/GSI at that same noon.
A 13:00 station service therefore has a different sampling hour.

`nfdrs` config is metadata: `fuel_model` V/W/X/Y/Z (default Y), `slope_class`
1..5 (default 1), `gsi_days` 28, `greenup_threshold` 0.2,
`precipitation_days` 28, `use_precipitation` true, `annual_herbs` false.
It is carried and checked on restart. Default GSI settings follow the published
v4/2016 modernization configuration. Older 21-day/0.5/no-precipitation settings
can be explicitly declared. All five 2016 fuel beds are supported without a
model-specific code path. `compute_backend` is auto (default), cuda (required)
or cpu. CPU fallback is recorded, never represented as GPU execution.

Nelson uses nonlinear bound-water/vapor diffusion, capillary diffusivity in its
valid saturation range, cylindrical geometry, rain/dew/evaporation surface
conditions, sorption exchange and solar/convection/latent surface heat balance.
Independent implicit finite-volume numerics use 21 radial nodes, 60 steps/hour
and a tridiagonal solve. Published adsorption constants, radii and maximum local
moisture are used. Positive latent conductance follows Nelson eq.8; the sign in
the paper's printed eq.9 would produce negative conductance. This independent
solver is **not binary parity with the GPL reference solver**. Its numerical
and calibration differences must be assessed through station validation.
Rothermel/NFDRS ERC/BI/SC equations are independently implemented and tested
against separately compiled public-domain USFS index-module vectors for all
five fuel beds. CPU/CUDA transport is checked against the independent Rust
solver; full monolithic/chained runs check every state and daily field.

`spinup CONFIG.json` acquires at least fourteen days of all six hourly fields
through the generic Rust ERA5 Zarr route. It requires start/days/area/out,
member/time/rain/startup settings and explicit
`research_era5_license: "CC-BY-4.0"`. Acquisition hashes bind reusable inputs.
This active-season implementation does not implement snow shutdown, snow cover
or overwinter DC. Weeks of spin-up do not establish deep drought equilibrium.
These limitations must accompany interpretation of fire danger.

`climatology REQUEST.json` weights twelve January–December monthly mm/day normal
planes by declared calendar lengths and bilinearly interpolates cyclic longitude
to write an annual-rainfall grid. Supply source/template paths, variable/axes,
days_per_month, provenance, license, output and receipt.

`verify REQUEST.json` compares FWI with the CWFIS station archive, using explicit
research_station_license=OGL-Canada-2.0, dates, offset and maximum distance.
Only active finite nonnegative indices are used. Site identity includes name and
coordinates; identical duplicates collapse and every conflicting site/day is
quarantined and counted. The archive's own licence remains to be established
before default redistribution; the OGL metadata applies to CWFIS daily maps.

`verify-nfdrs REQUEST.json` compares native daily products with a CSV reference.
Supply reference/forecast/rows/receipt, dates/offset, date_column, a
product-to-column `columns` mapping, reference_name, fuel_model, slope_class,
provenance, and explicit research_reference_license. `aggregation` is either
nearest_station (latitude/longitude required) or regional_mean (cosine-area
weighted grid mean). A station-group reference is labelled as such; it is not
represented as a single station. Receipts retain bias, MAE, RMSE and correlation,
source hashes and differences in sampling. These are weather-derived station
indices, not measurements of fire behaviour. No unverifiable skill threshold or
claim of operational agency readiness is inferred from running a calculation.

`compare REQUEST.json` verifies bitwise equality of all time/product values and
exact carried scientific state between monolithic and concatenated chained
outputs. Completed request replay checks input/state/output/climate hashes.
Existing unrelated artifacts are preserved. Failed temporary output is removed
with its path/size reported. Individual artifacts publish atomically; the whole
output/state/receipt triplet is not a crash transaction.

Mathematical sources: Van Wagner (1987) FTR35; Van Wagner/Pickett (1985) FTR33;
Keetch/Byram (1968) SE-38; Nelson (2000), CJFR 30:1071–1087;
Carlson et al. (2007), IJWF 16:204–216; Jolly et al. (2024), NFDRS v4
modernization. See SOURCES.md for licences. No GPL source is incorporated.
