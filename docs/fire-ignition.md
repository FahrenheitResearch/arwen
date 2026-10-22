# Public fire observations

`gpuwm fire-ignition REQUEST --experiment FORECAST.toml --domain 1` admits
reported incident discovery, a dated observed perimeter, and optional NASA FIRMS
VIIRS/MODIS CSV detections. Rust validates source hashes, decodes all geometry
and CSV records, projects coordinates, preserves polygon holes, and prepares
`LFN_HIST` and `HISTORICAL_TIGN` on the forecast's refined fire grid. Python only
acquires bytes and drives the native interface.

Use `configs/fire-observations/request.example.json` as the request template.
Replace its incident UUID and UTC epoch milliseconds, supply the admitted
`gpuwm fire-static` bundle, and choose a different output bundle path. The
experiment supplies the grid and refinement; conflicting request geometry is
refused. Point-only initialization requires an explicit positive
`incident_radius_m`; a discovery point has no measured fire size.

The default routes are public NIFC WFIGS incident locations and perimeters.
`nifc-geomac-2018` also exposes dated historical observations. A new compatible
GeoJSON provider requires a row in `fire-observation-routes.v1.json`, including
its ID and observation-time field names, source terms and licence policy.
Acquisition retains every raw response page and seals their manifest with SHA256.
For captured files, replace `route` with `path`, `sha256`, `source_url`, `license`,
`id_property` and `time_property`. Coordinates must be EPSG:4326, dates UTC epoch
milliseconds, and the incident UUID must match. Null measurement times are
refused; catalog modification times are never substituted.

Initialization selects the latest measured perimeter at or before the declared
initialization time. Later polygons remain held out. Incident discovery is a
reported discovery time, not measured physical ignition. The request's explicit
`assumed_burn_age_seconds` is a uniform assumption at the perimeter observation
time; its age increases by the interval to initialization. Neither a perimeter
nor satellite detections measure the ignition history inside the burned area.
The receipt preserves these assumptions and both observation and model times.

Optional `hotspots` supplies a captured FIRMS CSV with `path`, `sha256`,
`source_url`, `license`, `earliest_utc_epoch_ms`, `bounds` in west/south/east/north
order, `minimum_modis_confidence` (0 to 100), and
`include_low_confidence_viirs`. Records later than initialization, outside that
incident search box, or failing confidence/type filters are excluded. Pixel
sizes use the CSV scan/track dimensions in km, represented by circles with the
same nominal projected area because orientation is unavailable. Accepted
footprints join the perimeter's burned mask. Detections describe active pixels,
not a complete burned perimeter or an observed burn-age map. Public FIRMS
country archives can be downloaded without an API key; API endpoints require
the provider's own key. No credentials are searched or inferred.

An admitted output bundle activates real-perimeter initialization automatically,
including streamed compute buffers. A moist atmosphere, positive
`fire_atm_feedback`, and `fmoist_only=false` are required for the coupled forecast.
The existing SFIRE path returns sensible and moisture fluxes to the atmosphere.
History attributes retain the observation receipt and
`FIRE_PYROCONVECTION_ENABLED=1`, which means thermodynamic feedback is enabled.
It does not assert that a pyrocumulonimbus cloud forms. A later atmospheric
restart must restore its coupled fire state rather than replay an older fresh
initialization.

For verification, set `GPUWM_FIRE_OBSERVATIONS` to the overlay request template
and render the matching valid-time history file through `gpuwm.rustwx` using
`fire_perimeter`, `fire_ros`, or `fire_heat_flux`. The native renderer draws the
model's zero contour in dark red and the independently dated perimeter in cyan.
An observation/forecast time mismatch fails rather than implying skill at an
unrelated time. Each PNG has a sibling `.fire-verification.json` receipt with
cell-center intersection-over-union, missed/false-alarm counts, nominal projected
areas, and finite ranges for `FGRNHFX`, `FGRNQFX`, `GRNHFX`, `GRNQFX`, `RTHFRTEN`
and `RQVFRTEN`. Areas do not include map-factor correction. GeoMAC perimeter
terms state that these are informational observations, not final official
incident perimeters; check the source terms before interpreting skill.

The permanent `configs/fire-observations/camp2018.toml` case uses a 192 by 160 by
49 atmospheric grid at 300 m, a 1920 by 1600 fire grid at 30 m, and a six-hour
forecast from 2018-11-08 18 UTC. From the repository root, acquire its pre-event
fuels, attach the public observations, and run the forecast:

```sh
gpuwm fire-static configs/fire-observations/camp2018.toml --domain 1 \
  --request configs/fire-observations/camp2018-static.json \
  --output configs/fire-observations/fire-static.npz \
  --receipt configs/fire-observations/static-receipt.json
gpuwm fire-ignition configs/fire-observations/camp2018-ignition.json \
  --experiment configs/fire-observations/camp2018.toml --domain 1
gpuwm go configs/fire-observations/camp2018.toml --outdir fire-observation-proof
```

This demo needs the regional fire-input implementation for the 2016 LANDFIRE
canopy bundle. The provider's 17:54 UTC perimeter is carried forward six minutes
without an invented growth correction. The explicit uniform burn age is 900 s at
that observation, hence 1260 s at model initialization. The dated 00 UTC polygon
is available for verification after the forecast; it is not assimilated at start.

The complete six-hour demonstration was run on 2026-10-20 with 14,400 coupled
steps and seven hourly history files. The initial history contains the admitted
perimeter before the first fire step, with no retroactive heat release. Against
the held-out 00 UTC GeoMAC perimeter, the native cell-center verification gave
IoU 0.743771: 276,050 intersecting cells, 54,595 false alarms and 40,504 misses.
Nominal projected model/observed areas were 297.5805/284.8986 km². Northern and
eastern flanks broadly follow the observation; the western extension is missed
and southern spread is too large. This single event is pipeline evidence, not a
claim of general forecast skill.

All six coupling fields were finite and nonzero at the final time. The initial
perimeter, categorical fuels, terrain, final perimeter overlay, heat flux and
spread-rate panels were rendered by `rw_wrfbatch` through `gpuwm.rustwx` and
inspected. The spread-rate panel's narrow front was not visually discernible at
the full-domain scale, so that image alone does not quantify spread speed.
The run took 1,974.30 s process wall on a shared RTX PRO 6000 Blackwell card,
with 8,310 MiB peak PID-matched VRAM sampled every five seconds. Shared-card wall
is not a controlled performance comparison. Native admission and rendering used
14 Rayon threads; CPU serialization in the forecast is recorded in the run
receipt. Hydro activation reported missing admitted provider inputs explicitly;
this proof establishes coupled fire/atmosphere execution, not coupled hydro.
