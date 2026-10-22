Hub-height wind and turbine power
================================

`gpuwm render WRFOUT... --series --products hub_speed,hub_power,hub_capacity_factor,hub_max_abs_ramp`
reads every native WRF time record, interpolates wind vectors at 80, 100, 120 and
150 m above local ground, and produces native Rust maps and numeric NetCDF files.
`gpuwm go forecast.toml --products hub_all` reaches the same path when its history
frames arrive. `hub_all` selects every hub product; ordinary `all` includes them.
The final pass accumulates the complete run, while the first live frame supplies
instantaneous fields. Run statistics require at least two distinct valid times.

Put settings in the forecast TOML, or pass that file with `render --hubwind-config`:

```toml
[hubwind]
heights_m = [80, 100, 120, 150, 175.5]
turbine = "iec_class_2"
rated_mw = 2.0
exceedance_ms = 12.0
ramp_window_seconds = 3600.0
ramp_ms = 3.0
power_ramp_fraction = 0.2
max_gap_seconds = 5400.0
```

A standalone JSON options object also works. Advanced JSON or TOML settings may
use an `options` object/table plus `turbines` and `observations` arrays, with
`compute_backend` (`auto`, `cuda`, `cpu`), `verify_cuda` (compare every GPU output
with the Rust reference), and `render_frames` (`all`, `last`) alongside them.
`auto` uses the installed CUDA driver and NVRTC when available, otherwise Rayon.
No CUDA runtime or compiler is bundled. GPU calculations require the caller's
normal device allocation; CPU installations need no CUDA library.

Discover product, turbine and observation rows with `gpuwm hubwind catalog`, or
`gpuwm render --list-products`. An explicit native JSON request runs with
`gpuwm hubwind products --request request.json`. It accepts `inputs`, `out`,
`options`, `products`, `domain`, `width`, `height`, and the advanced fields above.
Empty `products` means every hub product. One request describes one stationary
grid and one member. Mixed domains are grouped by the render front door.

Wind vectors are destaggered, rotated to true east/north and interpolated linearly
in geometric AGL height. A 10 m vector anchors the lowest layer where available.
Higher extrapolation is missing. Calm direction and TI are missing. The shear
product uses adjacent requested heights sorted by height, independently of the
order of the configuration. Direction is meteorological direction **from** true
north. Missing source values are masked before destaggering.

TI is `sqrt(2*TKE/3) / speed`, an isotropic estimate from unresolved PBL turbulence.
It is not a measured IEC turbulence intensity. The source table handles MYNN's
`QKE/2`, scheme-qualified PBL TKE, interface-level `TKE_PBL` and native `TKE`.
Missing or negative TKE produces missing TI; it never becomes a zero estimate.
Interface TKE is averaged to mass levels before interpolation.

Run mean wind and energy are weighted by elapsed time. Speed varies linearly
between output times; energy integrates the entire piecewise linear power curve,
including the cut-out discontinuity. Below-cut-in, cut-in-to-rated,
rated-to-cut-out and cut-out hours partition valid time. Exceedance includes
speed equal to the configured threshold. Missing data or intervals longer than
`max_gap_seconds` interrupt both integration and ramp episodes. A ramp is a signed
change over the trailing configured window, evaluated at each output timestamp.
Consecutive exceedances of the same signed wind or power threshold count as one
episode. Output cadence limits which shorter ramps can be detected.

The three packaged IEC class rows are normalized theoretical industry composites
from NREL/turbine-models, revision `e9e4ccc6a87f50c0c65d236154a2897e752c47c9`,
under BSD-3-Clause. The full notice is in
`licenses/turbine-models-BSD-3-Clause.txt`. They are not certified manufacturer
curves. Cut-in 3 m/s and cut-out 25 m/s are explicit operating assumptions at the
endpoints of the published tables; rated wind is the first normalized rated point.
The 1.225 kg/m3 reference density is a declared modelling assumption. Power scales
by `rated_mw`; density correction, wakes, electrical losses, icing and availability
are not inferred. Capacity factor and energy describe this ideal single turbine.
A new turbine is a table row with ordered `[speed_ms, normalized_power]` points,
operating limits, density, source and licence; no new algorithm is needed.

Every run writes a receipt with source paths, source record times, grid size,
configuration, turbine provenance, TKE availability, spatial statistics, strongest
ramp traces and per-stage wall times/parallelism. It also writes CF NetCDF
instantaneous speed/direction/TI/power in `<domain>/hub_fields/<day>/`, and all
run-statistic planes in `<domain>-hub-statistics.nc`. Rust maps follow
`<domain>/<product>_<height>m/<day>/`. No Python numerical processing or weather
plotting is involved.

Public tower verification
-------------------------

The observation table supplies NOAA BAO 100 m and 300 m rows, HTTPS daily
acquisition paths, coordinates, heights, CSV layouts and public-domain terms.

```
gpuwm hubwind fetch-observations --network noaa_bao_100m --date 2015-11-01 --out observations
```

The acquisition receipt retains the exact URL, byte count, licence URL and NOAA
acknowledgement. The 100 m sensor was absent before 2010-09-14 18:18 UTC. The 300 m
row uses the corrected true direction, not boom-relative direction. NREL M2 and
FINO1 are not default data sources: their access does not itself establish one of
the permitted distribution licences.

Add an observation row to a request, and include its exact height in `heights_m`:

```json
{"network":"noaa_bao_100m", "path":"/data/BAO_100_2015305.dat"}
```

Rust finds the nearest model cell within `observation_max_distance_km` (default
25 km), interpolates the model vector in time without bridging gaps, and emits
paired CSV values plus speed bias/MAE/RMSE and circular direction MAE. Observation
samples outside the model timeline or missing samples are excluded. A receipt
with zero matched pairs is not observational verification. Tower measurements
validate wind; turbine power remains a curve-based estimate.
