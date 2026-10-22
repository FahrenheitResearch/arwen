# Implemented NextGen output and restart boundary

This documents the implementation in `src/capi.rs`, `src/checkpoint.rs`,
`src/series.rs`, `src/netcdf.rs` and `gpuwm/hydro/output.py`. It adds no API,
model state, conversion or per-step callback. The withdrawn WRF-Hydro interfaces
and captures are not inputs to this contract.

## Driver binding

Prepare `gpuwm.hydro.output:prepare_history` after the actual routing producer,
before the resident driver seals its inventory. Its `source` option names the
prepared component with `reach_output()`:

| Producer key | Required native array |
| --- | --- |
| `reach_id` | contiguous int64[R], explicit publication IDs |
| `discharge_m3_s` | contiguous float32/float64[R] |
| `velocity_m_s`, `depth_m`, `nudge_m3_s` | same precision/order/card; required for native NetCDF |

All sources must remain allocated on the driver's card. Above-bed channel depth
is distinct from reservoir elevation. IDs/units/conversions and any lake-to-outlet
join belong to the producer. The writer never invents nudge values or IDs.

History options include `source`, `capacity_steps`, `cadence_steps`, `origin`,
the complete ordered `types` roster and `formats`. Optional `nexus_ids`,
`candidate_ids`, `provenance`, `parquet` and `renderer` configure the corresponding exports.
Supported formats are `analysis_csv`, `troute_csv`, `troute_netcdf`,
`troute_parquet`, `nexus_csv`, `candidate_json`, `renderer_request`. If `formats` is
omitted, the Python writer selects `analysis_csv` and `renderer_request`. Explicit
format lists retain their meaning; include `renderer_request` to emit its request.
Capacity includes the restored
prefix and planned continuation; it is part of restart identity.

After construction, the only history work in a timed step is a bound CUDA copy
launch (256 threads/block) and scalar cursor launch. Cadence skips capture on
the device. No I/O, D2H, scalar reads, allocation, synchronization or graphs.

```python
from gpuwm.hydro import output

# driver already includes history and the complete native carrier inventory.
writer = output.prepare(settings=settings, driver=driver, outdir=outdir)
if restart_path is not None:
    writer.restore(restart_path, driver)  # Before stepping; existing arrays.
# Execute the driver's already-bound before_lsm/advance launch sequence here.
result = writer.finish(driver, receipt={})  # After the timed sequence.
```

Standalone restore without a reach writer is
`output.restore_checkpoint(path, driver)`. The driver owns complete arrays,
identity and the checkpoint-metadata/validation/restore hook trio. Rust validates
its envelope and digest; driver validation checks the complete composition,
shapes, dtypes, clocks and continuation metadata before any destination write.
CUDA copies into original allocations, including strided views, then restores
metadata. The caller supplies the ordered producer stream at the boundary;
the current Python implementation uses the current stream on `driver.device_id`.
Complete source carriers must be inventoried; a diagnostic history cannot
substitute for missing module state or qualify full physics continuation.

The driver owns ensemble member identity as well. With driver revision
`d2f662947b467362a9acb1d2f01fca4746c7cc76`, construction inside the existing
member capture scope seals its nonnegative `member_id` in `restart_identity`.
Unscoped construction leaves that key absent. Output transports the complete
metadata unchanged and delegates validation before any device copy; it does not
remove member identity to admit another member's checkpoint. Prepare a fresh
driver in the original member scope before restoring. This adds no output
schema, callback or checkpoint namespace. Atmosphere checkpoint composition
and classification of prepared parent geometry remain driver-owned.

`finish` snapshots once, writes `restart.nc`, then requested products. Returned
and saved `receipt.json` include actual/requested output directories and boundary
timings. Occupied paths select fresh siblings preserving prior files; selection
is rechecked before snapshot. Individual Rust files publish without replacing
existing files. This is neither concurrent directory reservation nor a
multi-product transaction: a later export failure can leave a valid checkpoint
and earlier complete products.

## Renderer history publication

`renderer_request` writes `renderer-request.json` and
`renderer-history/flow-NNNNNN.{f32le,f64le}` at the final boundary. This uses
all captured frames, including any restored prefix, rather than repeating the
final routing vector. Each file is exactly one frame of native-precision Q
words, in the actual int64 publication order. There is no implicit trimming,
reindexing, narrowing or fabricated missing field. Rust owns slicing, ID
formatting and UTC clock conversion. Requests use `gpuwm.nextgen-products.v1`:
`node_ids` contains decimal strings from the actual int64 roster and every
frame supplies `valid_time` (UTC RFC3339 interval end) and `discharge_m3_s`
with a relative `path` and explicit `<f4` or `<f8` dtype. All paths remain
valid when the complete history directory is moved together.

The optional `renderer` object has `domain`, `source_label`, and `gauges`
(an explicit unique subset of the string publication IDs). Gauges default to
empty, so no site identity or thousands of chart requests are guessed. Domain
and source label default to generic routing-history labels; provide physical
run scope and numerics in the source label for evidence. The request targets
its own `rendered-products` directory; rendering is a separate output-boundary
command and never a routing callback. Returned series receipts identify the
actual request path, native dtype, units and frame/reach counts.

```json
{
  "formats": ["analysis_csv", "troute_netcdf", "renderer_request"],
  "renderer": {
    "domain": "parent-routing",
    "source_label": "Standalone routing history; captured laterals; no terrain",
    "gauges": []
  }
}
```

Use the renderer owner's existing module command with `--request` pointing to
that returned path. Its inputs resolve relative to the JSON. Reference and
observation fields require separately supplied files, exact aligned clocks and
explicit mappings; none is invented by output. The renderer can read native
NetCDF `flow[feature_id,time]`, but its `record` selector addresses the leading
feature axis. Selecting a time frame from that layout requires an explicit
int64 gather; the emitted per-frame raw files avoid that ambiguity.

`depth` in the combined t-route history includes reservoir elevations.
Consequently the generic request deliberately emits Q only. A HAND request
requires the producer's separate above-bed channel-depth roster, an explicit
join to terrain drainage and actual terrain fields. Full checkpoint/depth
history remains retained by the existing writer; Q never becomes depth.
Neither software permission nor a generated renderer request clears a dataset's
redistribution terms or proves physical/event skill.

## Checkpoint-only reuse inside the driver

There is no NextGen `HydroOutputWriter`/`Tape` API. The existing reach Writer
and driver-owned ModuleHistory Writer both reuse `_snapshot`, `_call` and
`restore_checkpoint` from `gpuwm.hydro.output`. A driver-owned boundary without
a reach history can use that same internal sequence:

```python
from gpuwm.hydro import output
import cupy as cp

# Between completed launch pairs. stream is the actual producer stream.
# driver.arrays already contains the complete sealed composition.
metadata = driver.checkpoint_metadata()
with cp.cuda.Device(driver.device_id), stream:
    roster, offsets, payload = output._snapshot(driver.arrays, stream)
checkpoint_receipt, _ = output._call({
    "operation": "checkpoint-write",
    "path": str(checkpoint_path),
    "metadata": metadata,
    "arrays": roster,
}, payload)

# In a fresh, identically prepared driver, before any new advance:
with cp.cuda.Device(fresh_driver.device_id), fresh_producer_stream:
    restored = output.restore_checkpoint(checkpoint_path, fresh_driver)
```

This is existing internal driver integration, not a second codec or new public
writer. The caller owns output-directory recovery/publication policy when it
does not use either existing Writer. Rust refuses replacement of an existing
checkpoint. `_snapshot` synchronizes the supplied stream at the boundary;
`restore_checkpoint` uses the current stream on the driver's card, validates
all incoming metadata before CUDA mutation, copies in place and restores clocks.
Validation is not rollback after a device-copy failure.

Register LSM/runoff/module state, backing slabs and views, forcing cursors,
driver workspaces and diagnostic history before the driver seals `.arrays`.
Do not append an unvalidated dictionary of LSM arrays at save time: the fresh
driver must validate the same complete inventory and identity. The metadata
hook includes component continuation state and driver clocks; source, network,
parameter, producer-order and member identities remain intact. In particular,
the current loader's verified `network_sha256` cannot be removed to accept an
older unbound identity. A native routing restart or diagnostic file alone is
not a complete combined checkpoint.

NetCDF-safe storage names are already generated by the Rust envelope described
below: arbitrary resident paths stay in `resident_name` and `arwen_arrays`,
while variables/dimensions use indexed names. No flattening, slash substitution
or manually inferred field-to-variable mapping is required.

For SFT/SMP, the soil owner's inventory at revision
`846c7cd324a3b98967d2190b7614ea18ee7263ed` names all 12 SFT and nine SMP
arrays, including native f64 packed scalars/scratch and i32 controls. Keep the
actual flat array shapes and native column/slot order. Producer arrays, forcing
cursors and history are additional carriers; these modules require no extra
metadata hook trio and publish no reach output. Their packed vectors do not
have one common physical unit. The soil owner's `RESIDENT-STATE.md` and
`layout.json` supply the roster/slot meanings; selecting physical producers
and native BMI exchanges remains assembly work. Module checkpoint evidence
does not establish combined physical forecast continuation.

## C ABI and ownership

`librw_nextgen_output.so` exports the following existing ABI. Use platform C
layout/alignment; pointer/size widths follow the target process.

```c
#include <stdint.h>
#include <stddef.h>

struct rw_nextgen_response {
    int32_t status;
    uint8_t *json;
    size_t json_len;
    uint8_t *payload;
    size_t payload_len;
};
struct rw_nextgen_response *rw_nextgen_output_call(
    const uint8_t *json, size_t json_len,
    const uint8_t *payload, size_t payload_len);
void rw_nextgen_output_free(struct rw_nextgen_response *response);
```

Requests are UTF-8 JSON with explicit lengths, not NUL-terminated strings.
The JSON pointer must be nonnull/readable; payload may be null only when length
is zero. Complete/synchronize a CUDA snapshot before entry. Input buffers stay
valid throughout the synchronous call. Rust does not retain them.

Response status is 0 on success, 1 on an ordinary error, 2 on a caught panic;
errors return JSON `{"error":"..."}`. Read/copy both response spans before
calling `rw_nextgen_output_free` exactly once on the response pointer. Never
free the spans separately, use C `free`, or retain borrowed data after release.
The Python `_call(request, payload)` wrapper implements this ownership and
raises on nonzero status. Its bytearray copy is opaque transport, not a numerical
conversion. The library is loaded from the sibling file by default; development
may explicitly set `GPUWM_NEXTGEN_OUTPUT_LIBRARY` before first use.

## JSON requests and payload order

Every request has a required `operation` discriminator. Unknown top-level fields
reject. The implemented request forms are:

| Operation | Other fields | Response metadata / payload |
| --- | --- | --- |
| `checkpoint-write` | `path`, `metadata`, `arrays` | timing/digest receipt; empty payload |
| `checkpoint-read` | `path` | metadata/arrays/SHA/read time; packed native payload |
| `dataset-write` | `path`, `schema` | container timing receipt; empty payload |
| `dataset-read` | `path` | `schema`; packed native payload |
| `series-write` | `plan` | export receipts; empty payload |
| `module-series-write` | `plan` | typed diagnostic-series receipt; empty payload |
| `raw-read` | `path` | source path; opaque file bytes |

Checkpoint arrays are an ordered list of
`{"name":"component/key","dtype":"f64le","shape":[N]}`. Payload concatenates
each array in list order, logical C order, with no alignment padding. Shape `[]`
means one scalar. A zero extent contributes zero bytes and remains in the JSON
inventory without a NetCDF data variable. Preserve aliases as declared carriers;
do not infer carrier counts from the number of physical variables. Supported
numeric types are `i8`, `u8`, `i16le`, `u16le`, `i32le`, `u32le`, `i64le`,
`u64le`, `f32le`, `f64le`. No narrowing or NaN/signed-zero normalization.

The CDF5 envelope is `arwen.nextgen.checkpoint/v1`. Ordered global attributes
are `arwen_schema`, `arwen_metadata`, `arwen_arrays`, `arwen_payload_sha256`.
The digest covers the complete concatenated payload. Nonempty array i becomes
`a000000` etc. with `resident_name`; dimensions are `d000000_0` etc. Numeric
words are restored by inventory rather than an inferred variable name. Full
container schema equality is checked on read, not only the digest. There is no
WRF-style restart schema or claim that this envelope is native t-route state.

Dataset schemas contain ordered `dimensions`, `variables`, `attributes` and
`format` (`netcdf4` or `cdf5`). Dimensions have `name`, `len`,
`unlimited` (default false). Variables have `name`, `dtype`, ordered dimension
names, ordered attributes, byte `offset` (default0), and optional `strings`.
Each attribute has `name`, `dtype` and either UTF-8 `text` for char or typed
little-endian `bytes`. NetCDF string variable values use the strings list,
not a pointer table in payload. Primitive variable ranges must cover the whole
payload without gaps/overlap. All dimensions/variables/attributes define in
supplied order, including the position and original type of `_FillValue`.

Series plans use int64 IDs at `ids_offset` and native f32/f64 history at named
flow/velocity/depth/nudge byte offsets. Histories are C[capacity,R], while native
NetCDF fields are (feature_id,time), time fastest; Rust transposes. `frames`
selects the completed prefix. `origin`, integer `interval_seconds` and
`cadence_steps` determine interval-end timestamps. RFC3339 explicit offsets
normalize to UTC; native UTC spelling also works. Nonzero fractional seconds
and ambiguous T/nozone timestamps reject. Full plan fields and format-specific
rosters are in `src/series.rs::Plan`; module plans are in `src/module_series.rs`.

## Point-series exchange with offline analysis

`analysis_csv` writes `streamflow.csv` with the exact header
`feature_id,time_utc,streamflow_m3_s`. Rows are time-major, then the producer's
publication order. `feature_id` is the decimal spelling of the producer's int64
publication ID; consumers retain that spelling as an identifier, not a floating
point number. Times are whole-second UTC RFC3339 with `Z`, at interval ends:
`origin + (frame_index + 1) * interval_seconds * cadence_steps`. Discharge is
m³/s, with native float32/float64 values preserved through decimal formatting.
NaN is an empty CSV field; infinities retain Rust's `inf`/`-inf` spelling and
must not be admitted as finite observations. CSV carries no provenance envelope.

`candidate_json` writes `candidate.json`, directly consumable by the current
Rust retrospective reader:

```json
{
  "schema": "arwen.point-flow.v1",
  "units": "m3 s-1",
  "provenance": {"identifier_namespace": "explicit producer namespace"},
  "samples": [
    {"feature_id": "producer-ID", "time_utc": "2023-01-05T00:05:00Z", "streamflow_m3_s": null}
  ]
}
```

This is a schema illustration, not an event sample. Supply unique, nonempty
`candidate_ids` in publication order and nonempty `provenance`; use string
provenance values for retrospective compatibility. JSON IDs may differ from the
numeric CSV IDs only through that explicit roster. JSON nonfinite values become
null. Both exports use the same captured discharge and completed frame prefix;
neither interpolates, aggregates or invents observations.

Archive mapping is a separate, authoritative JSON array of rows containing
`archive_feature_id` (int64) and `candidate_feature_id` (exact string), with
optional `station_id`. The writer does not infer HYFeatures/NHDPlus/gauge
identity or mapping cardinality. In particular, case(b) LowerColorado April
2023 output is not the January case(c) event or an archive-overlap substitute.
Actual physical case(c) model histories and their crosswalk remain dependencies.
Serialization and archive comparison run after the timed launch sequence.

## Evidence and deployment limits

Actual t-route case(b) output matches all7variables/1,348,496typed bytes and
ordered typed headers. The 1,180-byte literal ncdump header matches after only
the filename declaration line is removed. Physical HDF5 header/file bytes
differ with actual container-library versions/properties; those are not spoofed.
Native Parquet ordered schema/data match; footer/keyvalue/created_by bytes differ.

Current148-carrier MC+7LP+12RFC fresh6+6 continuation and uninterrupted12-step
whole CDF5 files match in both modes. Strict native QVD is0/242,712; fast retains
52,731 differences and is not strict acceptance. This is routing/history
continuation, not full Noah/CFE/terrain/atmosphere restart. Exact source pins,
commands, per-stage timings and hashes are delivered as scoped receipts, not
claims about every future inventory.

Reservoir qualification at runner `eabe22e8878b7609fea926a9f5364046f7c53504`
also calls production `Writer.finish` on that fresh restored assembly. Complete
CDF5 and 80,904-row analysis CSV match uninterrupted output in both modes.
Within the same writer/backend build, resumed and uninterrupted NetCDF files
also match, including all7variables/1,348,496typed bytes and ordered headers.
Resumed strict NetCDF matches those native typed data/headers; its physical
HDF5 file differs from native. This closes resumed publication for that scoped
148-carrier routing composition, using driver5509567, Rust20e74/Python2df892.
It is separate from the newer driver's member-identity instrumentation and
does not qualify full atmospheric or hydrologic composition.

Reservoir owner `1bdaac555997702bd9035674c0d6b63132f5c4b4` subsequently
executes the real root `gpuwm hydro run` door: 3600 seconds uninterrupted versus
1800 seconds, a fresh process restoring `restart.nc`, then 1800 seconds.
All six commands succeed across default-fast and explicit-strict modes. Each
final 148-carrier CDF5 and complete analysis CSV matches its uninterrupted
counterpart; NetCDF typed headers/data and same-build files match. Native strict
NetCDF typed identity remains separate from physical HDF5 identity. This uses
driver d2f6629 with the same Rust20e74/Python2df output, not an installed wheel.

Initialized catchment-only continuation is now separately measured by driver
`43f8cea2e84ce3e3eb28bcacce0e807fa22f43b2`. Its ModuleHistory writer reuses
this owner's `_snapshot` and `restore_checkpoint` functions without a serializer
fork, imported from 8edfccbd635. `restart.nc` contains the complete inventory
and capacity; `history.nc` contains only the completed diagnostic prefix and
does not satisfy the checkpoint envelope. A fresh resume must prepare the same
capacity, ordering, parameters and complete identity before validation/copy.

The actual initialized strict Noah/PET/CFE case(a), three catchments and 48 hourly
steps, matches fresh-process 24h + restore + 24h: all 466 declared carriers,
232,708 logical bytes, whole final CDF5 checkpoint and diagnostic history.
Independent Rust replay of resumed outputs matches all 864 stored native BMI
words. This closes that standalone declared-state continuation; it does not
establish parent/atmosphere/network/terrain/event continuation. Root commands,
source hashes, native verdicts, pictures and separate boundary/stage times are
retained in the output delivery's `evidence/root-restart-review/`. No per-step
output work or wire schema changes accompany this documentation.

Rust emits NetCDF-4/HDF5 fixed snapshots through netcdf-writer::netcdf4,
including NC_STRING reach labels, dimension scales and native little-endian
payloads. CDF5 remains the restart container. No netCDF-C, HDF5 C, zlib C or
Fortran runtime is linked. NetCDF-4 append/unlimited snapshots are outside this
complete fixed-history interface and are rejected. User metadata and data are
preserved; physical HDF5 bytes and generated backend metadata are not a file
identity claim. Native referees remain verification tools only.
