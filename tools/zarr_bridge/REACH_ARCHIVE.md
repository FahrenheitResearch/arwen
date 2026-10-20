Reach archive analysis is original Apache-2.0 Rust code using the existing
Zarr bridge; it contains no hydrological model source. It compares the public
NWM v3 retrospective DATA with explicit candidate point flows. Python only
configures requests and launches Rust. No Fortran/runtime or GPU is needed
by this offline analysis. It adds no work to a forecast routing step.

Build the existing bridge from tools/zarr_bridge with the pinned offline
Cargo.lock, placing CARGO_TARGET_DIR and TMPDIR on your scratch disk:
`cargo build --release --locked --offline --jobs 16`. Point GPUWM_RW_ZARR at
that rw_zarr binary. `python -m gpuwm.hydro_archive --output NEW_DIRECTORY`
inspects actual archive time coverage.
`gpuwm hydro retrospective` (also `python -m gpuwm hydro retrospective`)
uses `gpuwm.hydro_retrospective.register_cli(hydro_subparsers)` to register
the `retrospective` subcommand and its `func(args)` callback. It delegates
to the existing `gpuwm.hydro_archive` parser and launcher; register either
entry point once per parser.
`python -m gpuwm.hydro_retrospective` also runs the same standalone parser.
The common door needs no second parser implementation.

Event example (choose dates within coverage, using real case mappings):

```
python -m gpuwm.hydro_archive --start 2023-01-01T00:00:00Z \
  --end 2023-01-02T00:00:00Z --mapping mapping.json \
  --candidate point-flow.json --cache SCRATCH/archive \
  --output SCRATCH/event
```

Mapping is explicit JSON configuration, not an index assumption:
`[{"archive_feature_id":123,"candidate_feature_id":"wb-456","station_id":"01234567"}]`.
These example identifiers are placeholders, not a gauged case. The referee
supplies case(c)'s authoritative crosswalk. Hydrofabric identifiers must not
be assumed equal to NWM NHDPlus reach IDs.

Candidate interchange (Rust producers should emit it directly):

```json
{"schema":"arwen.point-flow.v1","units":"m3 s-1",
 "provenance":{"forcing":"identity","initial_state":"identity"},
 "samples":[{"feature_id":"wb-456","time_utc":"2023-01-01T00:00:00Z",
             "streamflow_m3_s":1.25}]}
```

Missing flow is null. Duplicate candidate feature/time keys are errors.
Only exact feature and UTC timestamps join. Incomplete events retain paired
scores but do not claim a peak lag. NSE is null for constant reference flow.
For independently measured flows, set `"kind":"observations"` and optionally
`"label":"USGS observations"` in the point-flow document. The archive then
becomes the prediction: bias is archive minus observation, NSE uses the
observation variance, and peak lag is archive minus observed peak time. The
report records these roles explicitly. Default kind is `model`, preserving
candidate-minus-archive scores. Unknown kinds are rejected. Observation input
does not supply a missing native/NextGen run or establish a model crosswalk.
Very short, nearly constant windows are poor skill measures. The published
CF scale/offset is preserved; decoded word identity is decoder evidence,
not bitwise hydrological physics. Source/candidate provenance declarations
are reported as context, never isolated causal explanations.

Sources are rows in gpuwm/data/reach-archives.v1.json. `--source-table` and
`--source` select another compatible Zarr row. No provider-name code branch
exists. `--store CAPTURED_DIRECTORY` replays exact compressed objects offline.
Remote captures record their origin URL and reject mixing datasets. Archive
metadata, CF timestamps and cadence are validated before extracting flow.
Out-of-coverage events fail with actual endpoints; operational output is never
substituted for retrospective output.

Results: coverage.json, report.json, reference.json point series, paired.csv,
and one SVG hydrograph per requested archive reach. Axes show m3/s and UTC;
missing samples stay gaps. An existing output directory receives a new `run-*`
subfolder, preserving all prior files. The report names the actual
`output_directory`. The first run
into a new directory retains the direct layout above. Concurrent launches
reserve separate output directories atomically.
Every picture must be opened and reviewed in the
receipt. These are analysis time series. Hydro/weather field maps continue
through rw_wrfbatch/gpuwm.rustwx. Per-stage wall times name task concurrency;
no isolated-card timing or routing-step performance is implied.

NOAA data terms/coverage: https://registry.opendata.aws/nwm-archive/.
The dataset declares open data without use restrictions, AORC1.1 forcing,
no streamflow assimilation and February1979–January2023 coverage. Actual
CF endpoints are measured by the coverage command. Any requested event must
overlap that coverage for a direct NWM v3 comparison; a later flood cannot
silently be relabelled as a retrospective comparison.
The hydrofabric software/data licences are separate and owned by referee.
