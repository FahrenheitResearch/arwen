# GPU storm surge

`python -m gpuwm.ocean.surge run CONFIG.json --out OUTPUT` runs the entire
regular latitude/longitude grid solver, writes NetCDF water levels and coastal
inundation, verifies supplied NOAA gauges, and renders maps through
`gpuwm.rustwx` and `rw_wrfbatch`. `prepare` exposes the same Rust input contract. If rendering is interrupted,
`finish CONFIG.json --out OUTPUT` rechecks input hashes and controls and
completes NetCDF products and maps from the completed native integration.
Maxima are stored at the final time and cover the complete run period.

Build `tools/rw-bathy` with `cargo build --release --manifest-path
tools/rw-bathy/Cargo.toml`, then place `rw_bathy` on PATH or set
`GPUWM_RW_BATHY`. A CUDA toolkit and GPU are required. The door caches a native
library built for the installed GPU; a prebuilt library can be selected with
`GPUWM_SURGE_LIBRARY`. The existing renderer resolver selects `rw_wrfbatch`.
No atmospheric model has a special adapter.

Inputs are local NetCDF files and explicit variable names in JSON. Paths are
relative to the JSON file. Bathymetry is finite elevation positive upward in
metres on monotone one-dimensional latitude and longitude axes; it must cover
the solver grid. Supply its actual vertical datum, source and license. Forcing
is U10 and V10 in metres per second and sea-level pressure in Pa with dimensions
`[time,latitude,longitude]`, regular axes, epoch seconds, and complete temporal
coverage. Source and license are required. Rust reads, checks, normalizes,
regrids and writes fields; Python only runs stages and passes native buffers.
The CUDA solver interpolates atmospheric forcing in space and time.

The grid JSON specifies `nx`, `ny`, `west`, `east`, `south`, `north` (cell
centres, degrees). Required top-level fields are `grid`, `bathymetry`, `forcing`
and `duration_seconds`. Defaults: dry depth 0.001 m, Manning 0.025, CFL 0.4,
maximum step 60 s, output interval 21600 s, initial water level 0 m, reference
pressure 101325 Pa. Atmospheric forcing is mandatory, including calm cases.
`open_boundary_mask` has west/east/south/north bits 1/2/4/8, default all open;
zero gives reflecting walls. Initially dry boundary cells reflect. Supply
`tide_levels_m` at every atmospheric time for an optional spatially uniform
astronomical open-boundary tide. An omitted tide is zero. Open wet cells use
Flather radiation with prescribed tide and inverse-barometer elevation.

The original finite-volume implementation uses hydrostatic MUSCL
reconstruction, HLL fluxes, SSPRK2, conservative draining of shared faces,
wind stress, pressure gradients, Coriolis, spherical metrics and implicit
Manning drag. The timestep is reduced on the GPU from the full wet state.
The native driver enqueues 64 explicit timesteps per batch without CUDA graphs;
Python never steps cells or advances time.
Mass, boundary exchange, negative depth, wet land and maximum speed are
recorded every 600 simulated seconds and at the end.

`surge.nc` contains `water_level`, `inundation_depth`, `max_surge`,
`max_inundation_depth`, geolocation and times. Dry water levels are missing;
inundation is water depth on initially dry terrain only. Maximum surge is
maximum water level minus initial water level minus imposed astronomical tide.
It includes pressure setup and excludes wave setup. Initial water is at rest
at the specified level wherever bed is below that level. A single regular grid
is supported; no nested-grid conservation or wave/erosion coupling is claimed.
The initial condition, finite domain, grid resolution, atmospheric resolution
and terrain datum affect skill and must be assessed using observations.

Optional `gauges` rows contain station, latitude, longitude, observations and
predictions JSON paths from the public NOAA API. The nearest initially wet
cell within 5 km is selected and its distance reported. Verification uses
verified hourly observations minus NOAA astronomical predictions in the same
MSL datum, matched only at common model times. Missing observed storm peaks
are not reconstructed. Full model series and matched-time CSVs remain
available. Receipts include input hashes, controls, stage walls, actual
parallelism, allocation bytes and physical diagnostics.

`examples/fetch_public_case.py` explicitly downloads a 2022 hurricane research
case's public ETOPO and three NOAA gauges and writes `case.json`. Extract
`examples/forcing-request.json` with the existing native `rw_zarr extract`
door to `era5.nc` beside that JSON. The example explicitly selects the
CC-BY-4.0 ERA5 research input; there is no automatic default data download.
See SOURCES.md for source terms and attribution.

## Measured research case

The complete 72-hour Ian 2022 research example ran on an RTX PRO 6000
Blackwell Server GPU at 800x780 cells (about 1 km), with hourly ERA5 wind
and pressure and NOAA ETOPO terrain. CUDA graphs were disabled. The full
door, including native preparation, integration, NOAA verification and all
15 Rust maps, completed in 1274.03 s under a shared 1024 MiB GPU lease;
sampled process peak was 652 MiB. It advanced 244080 steps with maximum
relative mass error 9.09e-12, zero negative depth and domain maximum surge
2.46 m. Five complete 12-hour physical regressions passed in 440.32 s.

At verified common hours, Fort Myers residual RMSE was 0.7505 m and its
modeled peak was 0.4294 m versus 2.18 m observed, eight hours late.
Naples RMSE was 0.3385 m across 42 matched hours, with missing observed
storm-peak intervals; St Petersburg RMSE was 0.3544 m across 73 hours.
The coarse-grid example substantially underpredicts Fort Myers surge.
Initial water is at rest without spin-up; the terrain datum approximates
MSL, forcing is 0.25 degrees, and coastal channels have grid-scale limits.
Operational forecasting skill remains unqualified. Exact receipts, source
hashes, gauge series and inspected maps accompany the lane delivery.

## Coupled device ABI

`gpuwm.ocean.surge:create_solver(grid, bathymetry, config={})` implements
`gpuwm.ocean.device/v1` from the shared `gpuwm.ocean.interface` module supplied
by the ocean coupling package. Its `set_forcing(forcing)`, `set_waves(waves)`
and `step(dt)` operate on CUDA float32 C-contiguous planes. `outputs` is the
shared `SurgeOutputs` object; `eta` and `inundation` retain their allocations.
Use the creation device and the caller's current stream. Arrays must remain
alive until that stream completes; order stream changes with caller events.
There are no per-step allocations, host field copies, synchronization calls
or CUDA graphs. Initial construction reads the bathymetric depth bound and
builds/caches the native library; `nvcc` and CuPy are required. Set
`GPUWM_SURGE_DEVICE_CACHE` to choose its build cache.

The incremental solver reuses the standalone hydrostatic MUSCL/HLL fluxes
and positive draining limiter, with metric `grid.dx/dy`, SSPRK2, implicit
Manning friction, local-latitude Coriolis, pressure gradients and the
**supplied** wind stresses (no second wind-drag diagnosis). Earth-relative
stress is rotated using `cosalpha/sinalpha`. Finite-depth radiation stress
uses wave height, period and towards-direction, solving dispersion on CUDA
with a 0.78 depth-breaking limit. Initially wet water is selected by the
ocean-connected `bathymetry.wet` mask; other cells may subsequently flood.
Dry `eta` equals bed elevation so coupled wave depth remains zero, rather
than becoming NaN. Inundation is nonnegative depth on initially dry cells.

Scalar options and defaults: `dry=.001`, `manning=.025`, `initial_level=0`,
`boundary_level=initial_level`, `reference_pressure=101325`, `boundary_mask=15`
(open west/east/south/north bits 1/2/4/8), `cfl=.4`, `max_dt=30`,
`max_velocity=100`, `max_surface_rise=100`. Open ocean-connected wet boundaries
use Flather radiation with inverse-barometer pressure and the supplied scalar
boundary level. The scalar boundary level is constant in this ABI; the
standalone door retains its time-varying tide series.

Each requested interval is partitioned exactly into conservative fixed
substeps, bounded using maximum bed depth plus the configured velocity and
surface-rise envelope. No host CFL polling is needed. The device int32
`diagnostics` counters are, in order: nonfinite state, negative depth beyond
1e-6 m, actual CFL exceeding its bound, and invalid forcing/wave input.
Inspect them when collecting output; nonzero counts invalidate that run.
For extreme non-weather flows, increase the velocity/rise envelopes before
constructing the solver. This incremental metric-grid implementation does
not claim bitwise identity to the standalone spherical-grid integration.

`tests/test_device.py` exercises the real interface on a non-default stream:
six 192x256 states advanced for 40 s each, wind/pressure analytic checks,
wave stress, rotation, wetting/drying, persistent outputs and rejected host
or wrong-dtype inputs. The optional `tests/device_guard.c` preload guard
intercepts native host transfers and synchronization; the delivered proof
enabled it and checked that every intercepted copy used the caller stream.
