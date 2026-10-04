# Native ensemble observation matching

The `rw_precip` and `rw_ensemble_match` binaries live in the existing `rw-obs`
crate. They prepare precipitation observations and match individual forecast
members to the input seam of `ensemble_calibration_score.rs`. They do not run
forecasts, fit amplitudes or qualify an ensemble as calibrated.

Build on the authorized CPU host:

```sh
cargo build --offline --locked --release -p rw-obs --bin rw_precip --bin rw_ensemble_match
```

Precipitation product definitions are data in
`tools/rustwx/crates/rw-obs/data/precipitation-products.json` and
`precipitation-hdf-products.json`. They define source identity, variable names,
units, masks and accumulation duration. MRMS Pass2 GRIB is decoded through the
existing native GRIB core. IMERG Final V07B HDF is decoded through the existing
native HDF reader. Native time bounds and independent granule timestamps must
agree. Rate data are integrated over their actual interval. Every output is an
observation pack in mm with explicit start, end and raw source hashes.

```sh
rw_precip decode --file hourly.grib2.gz --product MultiSensor_QPE_01H_Pass2_00.00 \
  --bbox -100,30,-90,40 --valid-time 2000-01-01T01:00:00Z \
  --geometry geometry.obspack --out hourly.obspack
rw_precip decode-hdf --file halfhour.HDF5 --product imerg-final-v07b \
  --bbox 140,-25,155,-10 --valid-time 2000-01-01T00:30:00Z \
  --geometry geometry.obspack --out halfhour.obspack
rw_precip accumulate --input first.obspack --input second.obspack \
  --start 2000-01-01T00:00:00Z --end 2000-01-01T01:00:00Z --out total.obspack
rw_precip verify --file total.obspack
```

These filenames and dates illustrate the CLI shape. They are not supplied data.
Accumulation sorts intervals and refuses gaps, overlap, mixed geometry or wrong
endpoints. Cumulative validity is the intersection of every component mask.
Missing observations never become dry weather. IMERG quality index and its mask
remain in the companion `.quality.obspack`; there is no invented error estimate.

Freeze a matching plan once for each actual model grid. The request is JSON:

```json
{
  "grid": "/absolute/path/wrfinput_d01",
  "surface": "/absolute/path/asos/surface.json",
  "precipitation_geometry": "/absolute/path/geometry.obspack",
  "precipitation_direction": "observation-to-model",
  "interior_rim_m": 45000.0,
  "boundary_rows": 5,
  "elevation_tolerance_m": 100.0,
  "max_regrid_distance_m": 5000.0
}
```

`boundary_rows` is the actual specified plus relaxation width from the run
configuration. It is not inferred from a missing file attribute. The extra
interior rim is applied after those rows. The station file is the native
`rw_asos` v2 output, with its registered QC and nearest-time matching. Terrain
admission uses the nearest model terrain cell, the land mask and the supplied
elevation tolerance. Surface values use bilinear interpolation. The existing
`static-fields` WRF projection must reproduce every grid coordinate within
0.01 cell before any station is admitted.

```sh
rw_ensemble_match prepare prepare.json plan.json > plan-receipt.json
```

The plan stores source hashes, frozen station positions and drop reasons, grid
interior masks, and the integer mapping from the existing Rust `obs-regrid`
cell-average operator. That operator averages source centres assigned to each
destination centre; it is not an exact area-overlap conservative remap.

Use `observation-to-model` for finer observations such as MRMS. Use
`model-to-observation` when observations have coarser native support, such as
0.1-degree IMERG. In that direction, a satellite target touching any excluded
model source cell is dropped. Satellite observations are never repeated on an
upsampled fine grid. Set and record the distance bound before looking at scores.

Match one quantity and valid time with this request:

```json
{
  "plan": "/absolute/path/plan.json",
  "quantity": "precipitation_accumulation",
  "valid_time": "2000-01-01T12:00:00Z",
  "precipitation_observation": "/absolute/path/cumulative/lead-12.obspack",
  "members": [
    {
      "id": "member-001",
      "file": "/absolute/path/member-001/end.nc",
      "initial_file": "/absolute/path/member-001/start.nc"
    }
  ]
}
```

```sh
rw_ensemble_match match match.json matched.tsv > match-receipt.json
```

Supported quantities are `temperature_2m`, `wind_speed_10m`, and
`precipitation_accumulation`. The pinned `wrf-core` computes temperature and
wind diagnostics. For surface fields, omit precipitation observation and
initial-file entries. Wind is sustained speed, not gust. Precipitation is each
member's `RAINNC + RAINC` difference between the observation interval's exact
endpoints. Bucketed counters and decreasing accumulators are refused. The
history files must each contain one requested valid time and exactly the frozen
grid coordinates. A surface-only history is sufficient.

The optional production surface archive uses `match-diagnostics` instead of
`match`. Pin a complete manifest after the required members and times have
arrived. The manifest may still be receiving later hours, so the SHA-256 pin is
mandatory and binds precisely the snapshot scored:

```json
{
  "plan": "/absolute/path/plan.json",
  "quantity": "precipitation_accumulation",
  "valid_time": "2000-01-01T12:00:00Z",
  "precipitation_observation": "/absolute/path/cumulative/lead-12.obspack",
  "archive": {
    "root": "/absolute/path/forecast-output",
    "manifest": {
      "path": "/absolute/path/forecast-output/member-diagnostics/manifest.json",
      "sha256": "the actual 64-character lowercase SHA-256"
    },
    "grid_id": 1,
    "episode": 0
  },
  "members": [
    {"id": "member-0019", "member_id": 19, "seed": 9007199254740993}
  ]
}
```

The displayed seed illustrates an integer beyond float64's exact integer
range. Use the run's actual member IDs and seeds. The requested order must
equal the complete archive roster. Missing and explicitly unavailable members
are errors; the matcher never silently reduces the ensemble size.

```sh
rw_ensemble_match match-diagnostics match.json matched.tsv > match-receipt.json
```

The native reader checks `gpuwm-ensemble-member-diagnostics.v1`, CDF5 format,
file byte counts and hashes, exact uint64 member IDs and seeds, provenance,
grid/episode/time selection, float32 types, dimension names, units and both
coordinate hashes. The original float32 coordinates must equal the reference
coordinates of the frozen matching plan. The archive does not need terrain or
projection metadata because the plan already binds those authorities. ISO
times with an offset are normalized to UTC; naive engine times explicitly mean
UTC, independently of the host timezone.

Temperature uses the original `T2` words. Wind uses separate float32 squares,
addition and square root of `U10` and `V10`, then the frozen bilinear operator.
Each wind operation flushes subnormal inputs and outputs to signed zero,
matching the production CuPy RawModule's effective `-ftz=true` option. Each
operation rounds to nearest with ties to even, without FMA contraction. The
receipt names this numerical contract. Nonfinite wind values are unavailable
member values; their NaN payload bits do not affect matching or scores.
`RAIN_TOTAL` is the original float32 total since the declared forecast start,
including the producer's rain components. The observation accumulation must
start at that exact instant. No initial value is subtracted and no rain
components are reconstructed. Legacy WRF-history scores and production
diagnostic scores record distinct readers; a comparison must use a consistent
reader for its control and candidates.

Member labels must be unique and retain the declared ensemble order. The TSV
uses the scorer's `sample_id`, `weight`, `observed`, member-columns contract.
Every target has unit weight. Missing observations are written as `NaN` and
counted, preserving the same observation eligibility across recipes. The
receipt records individual file hashes, exact accumulation endpoints,
observation timestamps, plan hash, request hash and executable hash.

Run the precipitation tests and matching tests on a CPU host:

```sh
cargo test --offline --locked --release -p rw-obs --lib precipitation
cargo test --offline --locked --release -p rw-obs --bin rw_ensemble_match
```

The end-to-end matching fixture is explicitly analytical. It writes native
NetCDF history files, native precipitation packs and a surface record, then
checks independent expected temperature, wind and precipitation values, masks,
member identities and time refusals. It is an implementation test, not weather
skill evidence.

## Campaign orchestration

`ensemble_campaign_score.py` runs the native diagnostic matcher and scorer for
each supplied case, recipe, quantity and lead. It does not choose amplitudes.
Use one request for the full comparison so every recipe has the same observed
sample IDs, weights, values and missing-observation mask. An unavailable member
value is an error, even at a location where the observation is missing.

```sh
python tools/ensemble_campaign_score.py --request campaign.json \
  --matcher /path/rw_ensemble_match --scorer /path/ensemble_calibration_score \
  --out /new/path/campaign-scores
```

The campaign request has `lead_hours` (normally `[3,4,5,6,7,8,9,10,11,12]`),
`spinup_seconds` and a `cases` list. Each case supplies `case_id`, `held_out`,
`start_time`, a pinned `plan: {path, sha256}`, `precipitation_observations`
(`{lead_hour, path, sha256}` entries), and `runs`. Each run supplies `recipe`,
the same `archive` and `members` objects as the native request above, and a
pinned `production_manifest: {path, sha256}` pointing to its actual
`gpuwm-ensemble-output.v2` domain manifest. Put the ordinary control first.
Every run must include all three supported quantities.

Thresholds come directly from the production manifest's `temperature2`,
`wind10` and `rain_total` products. They must already contain the normalized
float32 numbers, with matching units and `ge` comparisons. The tool never
substitutes nominal decimal thresholds. Every scored production frame must
be complete with the full member roster. Final manifest pins are rechecked
after scoring. The native matcher independently verifies individual archive
file bytes, units, identities and coordinates.

Outputs include each native request, match table and receipt, pooled native
scores, the common observation-column hashes, `campaign-scores.json` and
`score-table.tsv`. All member IDs and seeds, actual production thresholds,
binary hashes and the frozen request are retained. Brier scores, CRPS, rank
histograms and reliability bins refer to the matched observation support;
they are not pixelwise comparisons to the original model-grid probability
maps after spatial interpolation or cell averaging.

## Frozen fitting and analysis charts

`ensemble_amplitude_select.py --request selection-request.json --out new-selection.json`
applies the predeclared normalized empirical-CRPS objective. Its request binds
`selection_plan`, `campaign_plan` and each `score_summaries` entry as
`{path, sha256}`. `arms` explicitly maps each summary's `score_recipe` label to
`kind`, `recenter_amplitude` and `stochastic_amplitude`. One `ordinary-control`
arm supplies the N=1 MAE denominator. The product alias
`forecast_total_precipitation` means `precipitation_accumulation`.

Stage `recenter` requires every declared recenter amplitude with stochastic
physics off. Stage `stochastic-and-default` additionally pins
`previous_selection` and requires each campaign recipe with stochastic off
and at both declared stochastic amplitudes. The selected recenter amplitude
cannot change. Use complete phase-specific score summaries. Undeclared arms,
held-out rows, changed member counts or observation masks, and missing
arm/case/product cells are errors. No poorly performing arm can disappear from
the objective by omission. Zero control denominators are excluded identically
for all candidates and listed explicitly; all raw scores remain in the receipt.
Exact objective ties choose the lower amplitude, then the frozen recipe
preference when selecting the recipe. The result is a training selection and
does not assert successful held-out validation.

`ensemble_campaign_plot.py --scores campaign-scores.json --out new-chart-directory`
writes PNG and SVG analysis charts for CRPS/RMSE, raw spread-skill, normalized
rank histograms and threshold reliability. Plot sources, hashes and captions
are retained beside the images. It only reads score summaries and does not
plot weather fields. N=1 rows remain in error and reliability charts but are
omitted from ensemble spread and rank charts. Review the figures on actual
campaign output before using them as scientific evidence.
