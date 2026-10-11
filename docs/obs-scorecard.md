The standing observation scorecard uses the existing native verification engine.
`rw_verify` reads, collocates, scores and renders fields. `rw_scorecard` pools
station residuals across leads within each case, averages FSS across leads,
and compares arms by resampling independent case blocks.

Run a manifest with one command:

```text
gpuwm obs scorecard campaign.json --out scorecard
```

Build both native binaries with
`cargo build --release -p rw-wrfbatch --bin rw_scorecard --bin rw_verify`
from `tools/rustwx`. The front door resolves the built or packaged binaries;
`GPUWM_RW_SCORECARD` and `GPUWM_RW_VERIFY` can name explicit builds. All native
steps hide GPUs. This harness evaluates finished runs and starts no forecasts.

The manifest schema is `gpuwm.obs-scorecard.manifest.v1`. It declares candidate
and reference labels, cases, metrics, and bootstrap configuration. Each case
has an id, an independent block id, and either native `requests`, native
`receipts`, or a digest-bound `pairs` CSV table. Requests and receipts can be
combined across leads. Related cycles, domains and overlapping events share a
block. Cases inside a block have equal weight; blocks have equal weight.
Each case must have the same two arm labels. A missing required metric writes
an incomplete scorecard and returns exit 3. Invalid input or a failed native
step returns exit 1. Missing optional measurements stay visible in the table
and give `coverage_status:partial`, with exit 0 if the required rows are present.

Native requests use `gpuwm.verify-visuals.request.v1`, the existing verification
format. Arm `kind`, `path`, `previous_path`, `fields` and point extracts are
metadata. WRF history, GRIB2, native stores, NPZ grids and point extracts share
this reader seam. Paths resolve relative to their containing manifest or request.
`stations_path` is the decoded native ASOS record. `radar` lists native observation
packs and geometry packs. Use the existing `gpuwm obs asos`, `gpuwm obs mrms`
and observation front doors to acquire and decode inputs. No acquisition,
post-processing or per-model adapter is duplicated here.

`--acquire` schedules those existing front doors from the manifest's acquisition
rows. A station row is `{"kind":"stations","source":"asos"}`. A radar row
declares `kind:radar`, `source:mrms`, `quantity` and the archive `product`.
A source outage stays in the receipt notes and missing rows; available sources
still score, with exit 3 if a required row cannot be measured. Private acquired
inputs are deleted after the command, with a file/size deletion receipt.
Caller-owned model files and observation inputs are never deleted.
`--data-root DIR` substitutes `${DATA_ROOT}` in request paths.

For example, a case entry is
`{"id":"case-01","block":"event-01","requests":["case-01/hour-01.json"]}`.
A metric entry is
`{"id":"temperature-rmse","source":"asos","quantity":"temperature_2m","units":"K","required":true}`.
FSS entries set `statistic` to `fss`, `direction` to `higher`, and supply
`threshold` and `width_km` matching the native verification request. RMSE and
MAE use `lower`. Signed bias uses `diagnostic`, because increasingly negative
bias is not an improvement. Units must match exactly; no unit guessing occurs.

The ASOS decoder carries reported gusts in `wind_gust_10m` (m s-1) and numeric
hourly precipitation in `precipitation_1h` (mm). Blank, missing or trace values
remain absent. Sustained wind never substitutes for a missing gust. A forecast
gust field must be mapped explicitly through `fields`; no gust diagnostic is
invented. Model hourly precipitation differences the native accumulation
components across two frames exactly one hour apart. Station precipitation
must set `exact_time:true` in its station quantity metadata, because an hourly
accumulation ending at a nearest METAR time is a different interval.

Native `station_quantities` entries declare `quantity`, `label`, `units`,
`minimum`, `maximum` and optional `exact_time`. The existing three-quantity
native defaults retain their prior values and scoring. Additional networks
use the generic decoded `gpuwm-obs.station-observations.v1` schema: `source`,
`units` by quantity, original-source `provenance.sha256`, a `stations` table
and `reports` with station id, target `valid_time`, actual `observation_time`
and `values`. `station_source` in the request must match that network. A
rawinsonde station/sample id includes its pressure level; the supplied native
forecast point extraction must use that same level. SNOTEL quantities use
declared snow or precipitation units and matching intervals. This seam accepts
already decoded and collocated profiles and snow observations. It does not
implement new raw IGRA or SNOTEL acquisition or profile interpolation. Reuse
the existing Rust IGRA reader and Rust isobaric extractor upstream.

Outputs are one JSON receipt, one case table CSV, one campaign table Markdown,
and the native Rust sheets for every raw request. The receipt binds input
digests, coverage, counts, case/block ids, metrics and bootstrap settings.
Positive improvement favors candidate. SplitMix64 and linear percentile
quantiles are specified in the Rust implementation. Fewer than four independent
blocks produces exploratory CIs. One block produces no CI. Pixel or station
counts never inflate the independent case count. No automatic model promotion
is made from this scorecard.

The raw two-experiment demonstration uses the checked-in recipe:

```text
gpuwm obs scorecard configs/verification/scorecard-recorded/campaign.json --data-root RECORDED_INPUTS --acquire --out raw-scorecard
```

The input root contains each case's A and B folders named in the request files.
Case-01 uses recorded microphysics arms and case-02 uses recorded PBL arms.
They are separate regional experiments, not one common treatment. Their CIs
are exploratory. The demo writes five production field sheets per case, reads
real WOOF history and public ASOS/MRMS, and exposes missing gust, station
precipitation, profile and snow data. Continuous MRMS errors use common finite
coverage and remain measured on observed dry grids. FSS with no observed events
remains unmeasured. The source's declared no-echo floor affects all-grid dBZ RMSE.

The retained demo runs with
`gpuwm obs scorecard tests/fixtures/obs_scorecard/demo.json --out demo-scorecard`.
It replays two real station exports with their original source hashes and QC
support. They are two forecast cycles of one event, so the correct bootstrap
has one block and declines to infer significance. It exercises recorded-pair
replay, not fresh raw-field reading or a complete two-event observation campaign.
