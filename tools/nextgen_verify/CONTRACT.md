# Gauge verification boundary

Owner: verify-usgs. Original Apache-2.0 verification code; no native model
runtime or hydrologic algorithm is included.

The hydro CLI owner calls
`gpuwm.hydro.verify_usgs.register_cli(hydro_subparsers)` once. It registers
`gpuwm hydro verify-usgs` with `fetch`, `import`, `compare`, `score`, `replay`,
`observations`, `stations`, `describe`, `discover`, `match` and `match-check`
(worldwide gauge rows and station-to-reach matching; see README). Every operation takes `--out`. Python only
parses arguments, resolves the packaged Rust executable, and returns its exit
status. `GPUWM_NEXTGEN_VERIFY` can select an explicit executable.

This is an offline setup/output boundary. It is not a prepared resident
component: it has no `advance`, `.arrays`, `.restart_identity`, or checkpoint
namespace. Do not register it in the numerical stage sequence or invoke it
per routing step. No card or CUDA stream is required. A routing-stack loader
is a hydrofabric/assembly dependency, not a dependency for observation decode.

Canonical input is an array of rows:

```json
[{"station_id":"00123456","time":"2023-01-01T00:00:00Z","flow_m3s":1.0}]
```

Identifiers are strings. Times require an explicit RFC3339 offset and match
at exact UTC milliseconds. Flows are finite nonnegative m³/s. Conflicting
duplicate station/timestamps fail. Table-described import joins explicit
feature IDs to stations and converts declared units in Rust; a new source
or history format adds table metadata, not a Python adapter. NHDPlus and
HYFeatures IDs cannot be substituted for each other.

A date without an offset is rejected with the required format in the error;
the verifier never assigns a timezone to an unannotated source date.

An optional `netcdf` layout row names feature/time/flow variables, an explicit
RFC3339 epoch, `time_unit_seconds`, and `expected_time_units` matching the file.
CDF-1/2/5 and NetCDF-4/HDF5 decode through the existing pure Rust reader. Integer
IDs remain exact. Coordinate dimension names determine time/feature order;
feature-only flow requires a single time. Flow units and packing/fill attributes
come from the file; supported unit factors remain table metadata. No Python
field export or new model adapter is required. See README for the complete row.

`score --model M --reference R --observations O --out DIR` writes
`scores.json` plus analysis SVGs. Model/native equality uses their complete
station/time support. Both observation comparisons use the same exact
three-way intersection; there is no interpolation.

- `complete_support_bitwise_f64` requires equal imported f64 words and
  complete model/native support at every requested station. It does not
  depend on observations and does not constitute a native module replay.
- `all_stations_have_observation_pairs` requires at least one common
  model/native pair and one three-way observation pair at each station.
- Exit 1 retains the receipt when any requested station lacks these pairs.
  Exit 0 means scoring coverage succeeded; it does not assert numerical
  equality, positive skill, or hydrologic acceptance. Read the equality flag
  and per-station metrics separately. Decode, I/O, or validation failures
  also return nonzero and may occur before a receipt can be written.

`compare --prediction P --label LABEL --observations O --out DIR` writes
`comparison.json` and scores the exact two-way UTC intersection, with both
unmatched counts. A station without pairs yields exit 1 with the partial
receipt. Exit 0 establishes coverage, not a skill threshold.

NSE, KGE2009, RMSE and peak timing use ordered f64 reductions within each
station. Peak offset is prediction minus reference, positive when late;
the earliest tied maximum wins. With fewer than two pairs, NSE/KGE/peak
timing are null. Constant-reference NSE and constant-series KGE/peak timing
are null with reasons. Rayon parallelizes independent files and stations;
`RAYON_NUM_THREADS` sets the worker budget. Receipts name stage timings and
parallelism; they are not GPU step timings.

`replay` checks actual retained observation-page decoding, UTC and unit
normalization against retained canonical rows. Its bitwise flag is scoped
to that operation. Native/GPU routing proofs remain the numerical owners'
responsibility. Analysis SVGs do not replace product hydrographs through
`gpuwm.rustwx`/`rw_wrfbatch`.

Source software and data licences remain separate. No hydrofabric database,
research-only crosswalk, native model runtime, or automatic data acquisition
belongs in the CLI registration. Fetch occurs only through the requested
`fetch` operation. Private research fixtures do not define a public demo.
