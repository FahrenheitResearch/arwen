# NextGen product requests

`python -m gpuwm.nextgen_products --request products.json` launches the real
`rw_wrfbatch` through `gpuwm.nextgen_products.run_nextgen_products`. Python handles the
command and receipt path; Rust reads, transforms and renders every numeric
field. `gpuwm hydro` registers no products subcommand on this tree; a common
hydro form needs the resident driver to register this module. Coupled forecast
history invokes the same launcher automatically.

An ordinary renderer build exposes the products ABI and accepts these requests:
`cargo build --offline --locked --release -p rw-wrfbatch --bin rw_wrfbatch`
in `tools/rustwx`. No extra feature is required. The launcher uses
`gpuwm.rustwx` for renderer discovery and environment setup.

Both request schemas preserve earlier products on repeated runs. An absent or
empty `out_dir` keeps the requested layout. An occupied directory, file or
symlink selects a fresh sibling named `NAME-hydro-products-UNIQUE_ID`;
an `out_dir` symlink remains untouched. The command reports that location and prints the
actual receipt path. Receipts contain `requested_output_directory` and
`output_directory`; scripts should use those paths rather than assume a rerun
replaces the original. This handles ordinary sequential repeats; it does not
reserve an existing empty directory against external concurrent writers.

This is new original Apache-2.0 product code based on the public integration
tree. No withdrawn WRF-Hydro source, port, fixture, ABI or runtime is used.
The existing Rusty Weather renderer retains its MIT notice. HAND follows the
published terrain definition in [Rennó et al. (2008)](https://doi.org/10.1016/j.rse.2008.03.018):
height relative to the first drainage cell along the supplied downstream flow
graph. Rust computes it with parallel pointer jumping, once per request.
Masked elevation and paths leaving the domain without drainage remain missing;
channel-free cycles are reported as malformed terrain topology.

A request is metadata identifying files and node IDs:

```json
{
  "schema": "gpuwm.nextgen-products.v1",
  "out_dir": "products",
  "domain": "parent-1km",
  "source_label": "NextGen run name and numerics",
  "node_ids": ["nex-17", "wb-29"],
  "gauges": ["nex-17"],
  "frames": [{
    "valid_time": "2024-05-04T12:00:00Z",
    "discharge_m3_s": {"path": "routing-q.f32le", "indices": "node-map.i64le"},
    "reference_m3_s": {"path": "native-q.f32le"},
    "observed_m3_s": {"path": "usgs-q.f32le"}
  }]
}
```

`node_ids` is an ordered, unique roster and may name gauges, flowpaths or
nexuses. Every flow/depth vector must contain exactly that many values after
its explicit mapping. Rust never truncates padded resident arrays or guesses
their relationship to feature IDs. `indices` is a little-endian int64 vector
of zero-based positions into the source buffer, in roster order. It also
expresses grid transposes and row reversal when the producer supplies those
indices. Out-of-range and negative indices fail before plotting. No data
adapter is added for a new provider; use the same metadata contract.

Buffers accept `path`, optional `dtype` (`<f4` default, `<f8`, `<i4`, `<i8`),
optional `missing` sentinel, and optional `indices`. A NetCDF source supplies
`variable` and optionally `record` instead of a raw dtype. `record` explicitly
selects a zero-based index on the leading variable axis, including one- and
two-dimensional arrays such as `[time, station]`; every remaining axis is
read in full. Scalars and out-of-bounds records fail. Omitting `record` reads
the whole variable. All sources resolve
relative to the request file; absolute paths also work. Missing/negative
discharge is blank in CSV and breaks plotted lines. Reference and USGS points
are drawn with larger markers than the model, including isolated samples, so
coincident samples remain visible. Only supplied times are sampled.

For maps add `terrain` with `nx`, `ny` and buffer fields `latitude`,
`longitude`, `elevation_m`, `downstream_cell`, `channel_node`, `stream_order`.
All raster fields have the same row-major layout. `downstream_cell` is a flat
zero-based cell index, -1 at an outlet. `channel_node` is a zero-based position
in `node_ids` at drainage cells and -1 off the network. This is an explicit
cell-to-node relationship, independent of native array offsets. Stream order
is positive integer order on drainage cells. The maps preserve cell values
using nearest-cell sampling, including categorical order and probabilities.

Each frame can add these buffers:

| Field | Layout | Meaning |
|---|---|---|
| `terrain_depth_m` | terrain cells | Actual terrain-2d dynamic water depth; renders depth and wet extent |
| `channel_depth_m` | node roster | Above-bed channel water depth; renders a separately named HAND estimate |
| `member_terrain_depth_m` | list of cell buffers | Complete same-time member roster of dynamic depths |
| `member_channel_depth_m` | list of node buffers | Complete same-time member roster of above-bed depths for HAND estimates |

HAND depth is `max(channel_depth_m - HAND, 0)`. It is an estimate and is never
labelled as the terrain-2d dynamic solution. Discharge is never converted into
water depth. Probability is the fraction of the complete declared roster
whose depth is strictly greater than `flood_threshold_m` (default 0.1 m).
Any negative/missing member masks the cell; dry members remain in the
denominator. Missing files fail the job. Dynamic probability and HAND
probability have different product names. Channel member probabilities do
not materialize one full terrain raster per member.

Outputs follow `<out>/<domain>/<product>/<valid-day>/`. Maps carry metres,
dimensionless order or probability fraction; chart and CSV discharge is m3/s.
Raw f32le map and display-flow buffers accompany each product. Streamflow also
has f64le exports and full-precision CSV: native float64 input is preserved,
and float32 input is exactly promoted. Reference word comparison uses this
float64 representation before display narrowing. Shape/dtype and
node roster in `nextgen-products-receipt.json`. The receipt includes per-stage
wall times, worker count and finite-reference bitwise word counts. This
transport comparison does not establish physics parity. HAND uses 0–100 m,
depth 0–2 m, probability 0–1; captions identify saturation, while raw fields
remain unbounded. Products execute only at history/output boundaries and add
no host work, synchronization, allocations or CUDA graphs to routing steps.

Native NextGen fixture replay, an opened short run and the RTX4090 step-speed
receipt are required separately before unit acceptance.

Exact output names below are relative to `out_dir`. `DAY` is the UTC date;
`TIME` is the sanitized UTC timestamp, `ID` is the sanitized node ID, and
`INDEX` is its zero-based roster position. Paths use lowercase slugs. The
receipt lists the actual paths produced by each request.

| Path | Units / contents |
|---|---|
| `DOMAIN/hydrograph/FIRST_DAY/INDEX-ID.png` | Streamflow in m3/s versus UTC; only supplied model/reference/observed series |
| `DOMAIN/hydrograph/FIRST_DAY/streamflow.csv` | `valid_time,node_id,discharge_m3_s,reference_m3_s,observed_m3_s`; absent series are empty |
| `DOMAIN/streamflow/DAY/streamflow_TIME.{f32le,f64le}` | Ordered node discharge, m3/s; f64 retains source precision |
| `DOMAIN/hand/DAY/hand_TIME.{png,f32le}` | HAND, metres; static terrain rendered at the first frame time |
| `DOMAIN/stream_order/DAY/stream_order_TIME.{png,f32le}` | Dimensionless stream order, static |
| `DOMAIN/inundation_depth/DAY/inundation_depth_TIME.{png,f32le}` | Dynamic terrain water depth, metres |
| `DOMAIN/inundation_extent/DAY/inundation_extent_TIME.{png,f32le}` | Dimensionless 0 dry / 1 wet, depth > 0 |
| `DOMAIN/hand_depth/DAY/hand_depth_TIME.{png,f32le}` | Estimated water depth from HAND, metres |
| `DOMAIN/flood_probability/DAY/flood_probability_TIME.{png,f32le}` | Dynamic terrain depth exceedance, fraction 0–1 |
| `DOMAIN/hand_flood_probability/DAY/hand_flood_probability_TIME.{png,f32le}` | HAND-estimated depth exceedance, fraction 0–1 |
| `nextgen-products-receipt.json` | Paths, shapes, worker count, per-stage milliseconds and finite reference comparisons |

The withdrawn Croton/WRF-Hydro case is not a public example for this unit.
The internal Lower Colorado routing replay has aligned GPU/native/observed
charts. It uses captured lateral forcing and does not establish runoff,
terrain or observational skill. Dataset-specific evidence is separate from
this generic request contract.

Depth-only requests may omit `node_ids`, `discharge_m3_s` and all four drainage
fields (`elevation_m`, `downstream_cell`, `channel_node`, `stream_order`). They
need only terrain dimensions/coordinates and frame `terrain_depth_m`. No HAND,
stream order, Q export or CSV is invented. To enable HAND/order, supply all four
drainage fields together, with the explicit `node_ids` roster. This supports
real terrain engineering runs before a channel crosswalk or Q is available.
`depth_scale_max_m` sets a positive display maximum in metres (default 2); use
a range that shows the supplied depths. Full raw values remain untouched.
A frame may supply `time_label`, such as `Elapsed 2400s from imposed reset`,
for an engineering experiment. ISO `valid_time` still orders frames and names
files; the caption must identify its simulation clock rather than imply an
observed forecast.

At an ensemble output deadline, the existing parent history collector calls:

```python
current_capture().submit_hydro(
    valid_time=valid_time, reach_ids=sample["reach_id"],
    discharge=sample["discharge_m3_s"], stream=sample.get("stream"),
)
```

`current_capture` comes from `gpuwm.ensemble.runtime_context`. The capture
context supplies the member ID. This is a keyword-only API. The ordinary and
mass history wrappers own this call; do not add another forecast callback or
call it from routing steps. Child-domain output does not recapture hydro.
The borrowed arrays are contiguous, on the parent card, and retain stable
int64 IDs, native f32/f64 discharge in m3/s and the producer CUDA stream.
Output-cadence capture is device resident.

Channel depth has its own optional `driver.channel_output()` mapping with
`reach_id`, `channel_depth_m` and producer `stream`. Its native f32/f64 values
are metres above the channel bed. At the same deadline the channel-depth sink
receives `member_id`, `valid_time`, `feature_id`, `channel_depth_m` and `stream`
as keyword arguments, before the producer reuses the buffers. Its ordered IDs may
differ from the full discharge roster, which can include reservoirs. Match
IDs explicitly; reservoir surface elevations are not above-bed channel depth.
The renderer's `node_ids` and `channel_node` mapping must use the depth roster
when producing a HAND estimate. Supply discharge separately with an explicit
gather if both products share a request.
If a terrain drainage ID has no corresponding above-bed depth, exclude that
drainage cell from `channel_node` and stop its `downstream_cell` path at -1.
Paths reaching that missing-stage boundary remain masked rather than borrowing
the stage of another channel downstream.

Reduced reach exceedance carries `threshold_m3_s[T]`, `exceedance_count[T,R]`
uint32, `finite_count[R]` uint32, `probability[T,R]` float32 fraction, ordered IDs,
requested member IDs and UTC. Flow comparison is `>=`; probability is missing
when any requested member is absent/nonfinite. Preserve counts, including dry
members, and declare the fixed roster denominator. Read exact paths, variable
names, dtypes and shapes from `nextgen/reach-ensemble.json`; do not infer them
from the display quantities or reinterpret thresholds at a different precision.
Current `arwen.nextgen.reach-ensemble/v2` publishes float32 thresholds. The
collector rounds each configured threshold to float32 once, then uses that
same value for comparison and publication; float64 discharge retains its native
precision and compares against the exactly promoted float32 threshold. Historical
v1 files carry float64 thresholds and retain their original meaning; their
checkpoint counts cannot be reused as v2 counts. Read the schema before plotting.
The collector refuses publication of an incomplete member cohort. This is a
transport contract for the ensemble owner; the depth/HAND renderer uses its
separately documented strict depth comparison.

The same module command accepts `gpuwm.component-series.v1` for component
diagnostics whose quantities are not discharge. It requires explicit units,
sample positions and source time labels; it never infers a timezone or
converts a storage amount into a rate. For example:

```json
{
  "schema": "gpuwm.component-series.v1",
  "out_dir": "component-products",
  "domain": "column-run",
  "source_label": "Native and GPU component replay",
  "charts": [{
    "id": "topsoil-temperature",
    "quantity": "Topsoil temperature",
    "units": "K",
    "clock_label": "source-time",
    "axis_label": "Source labels; timezone unverified; hourly samples",
    "positions": [1, 2, 3],
    "sample_labels": ["source hour 1", "source hour 2", "source hour 3"],
    "zero_based": false,
    "series": [
      {"label": "referee", "values": {"path": "native-temperature.f64le", "dtype": "<f8"}},
      {"label": "model", "values": {"path": "gpu-temperature.f64le", "dtype": "<f8"}}
    ]
  }]
}
```

Positions must be finite and strictly increasing. Every trace has exactly one
value per position after its optional explicit gather. Buffer metadata is the
same as for flow requests. Signed finite values are supported; missing values
break lines and appear as blank CSV values. `zero_based` defaults to false,
giving a tight display range; true includes zero. Chart ticks adapt to the
range, while full-precision CSV remains unchanged. Labels must be unique
within each chart; sanitized chart paths must also be unique.

Output is `OUT/DOMAIN/component_series/CLOCK_LABEL/ID.{png,csv}`. The CSV
columns are `position,sample_label,series,value`, with source f64 precision.
`OUT/nextgen-products-receipt.json` has schema
`gpuwm.component-series.receipt.v1`, per-chart load/render/CSV times, worker
count, and finite f64 comparisons for traces named `model` and `referee`.
These comparisons establish transport/replay agreement only. Charts and
traces run in parallel at output boundaries. Hydro field maps continue to
use the map renderer described above.

The component chart also accepts reach exceedance probabilities: use explicit
`fraction` units and identify the UTC output time, complete member roster and
`Q >= threshold` comparison. Read probability and threshold precision from the
producer manifest. For discrete threshold categories, supply ordered positions
and label every threshold, including any middle samples omitted by endpoint
ticks; state that spacing is categorical. Retain counts and valid-member counts
beside the chart. A flow exceedance chart does not imply depth or flood skill,
and identical control members do not establish ensemble spread.

The renderer ABI is
`gpuwm.nextgen-products.v1 node-map HAND terrain-depth ensemble-depth hydrograph component-series`.
Rebuild `rw_wrfbatch` together with its Python launcher when importing this
change; the launcher checks that exact capability string before execution.
