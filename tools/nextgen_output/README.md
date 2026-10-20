# NextGen output boundary

Source pins and notices are in NOTICE. This crate contains no hydrologic
numerics or Fortran runtime. Python transports opaque snapshots, Rust formats
and transposes fields, and CUDA captures output history without host reads.

Build from `tools/nextgen_output`, with CARGO_HOME,
CARGO_TARGET_DIR and TMPDIR set to a data-disk scratch directory:

```
cargo build --release --locked
```

The default developer build requires the system netCDF-C development library. The shared library is
`librw_nextgen_output.so`; supply its path in `GPUWM_NEXTGEN_OUTPUT_LIBRARY`
during development.

For a Linux stage without system netCDF/HDF5, use the pinned local-file-only
backend recipe. Choose a new scratch stage and keep Cargo caches/targets on
the data disk:

```
python3 tools/nextgen_output/build_local_backend.py \
  --scratch "$LANE_SCRATCH/output-backend" \
  --cargo-home "$LANE_SCRATCH/cargo" \
  --cargo-target "$LANE_SCRATCH/target" --jobs 24 --fetch
```

The recipe verifies archive and licence hashes in `backend-sources.json`,
builds PIC static netCDF-C/HDF5/zlib, and stages the CLI plus a sibling Rust
library beside `gpuwm/hydro/output.py`, matching the existing loader contract.
No output/checkpoint wire changes are needed. C symbols stay private to the
library. DAP, remote access, NCZarr, optional compression plugins, MPI and
Fortran are disabled; local NetCDF-4, deflate and CDF5 remain available.
`BACKEND-BUILD.json` records actual commands, times, dependencies, artifacts
and notices. An unexpected dynamic dependency beyond the system C/math/GCC
runtime fails staging. Source archives, expanded sources and C build trees
are removed after the receipt unless `--keep-work` is explicit. An existing
stage is preserved; retries use a new scratch path or clear their own failed
stage after recording its failure.
Rust builds use `--offline --locked`; the supplied Cargo home must already
contain the locked crate dependencies. `--fetch` fetches the three C archives.

This is a host-built Linux stage, not a manylinux wheel qualification.
Public bridge roster integration and installed-product verification remain
pending. System glibc/libm and GCC support libraries remain runtime requirements;
the recipe does not bundle those platform libraries.

The realization adds `gpuwm.hydro.output:prepare_history` after its routing and
reservoir stages, before the resident inventory is sealed. Options are:

```
source = "routing"
capacity_steps = 288
cadence_steps = 1
origin = "2023-04-01 00:00:00"
types = ["wb", "wb"] # explicit roster in source output order
formats = ["analysis_csv", "troute_netcdf", "renderer_request"]
```

The source binds `reach_output()` with ordered contiguous same-card int64
`reach_id` and native-real `discharge_m3_s`; NetCDF additionally requires
`velocity_m_s`, `depth_m` and `nudge_m3_s` of identical shape and precision.
`nexus_csv` also requires an explicit `nexus_ids` roster. No fabric ID mapping,
zero nudging field or extra reach is inferred. The caller sets capacity for the
whole planned continuation; it is part of checkpoint identity.

`renderer_request` emits `renderer-request.json` and one exact native-precision
raw discharge buffer per captured frame. The existing Rust product renderer
reads this request directly: explicit string publication IDs, UTC interval ends,
relative buffer paths and dtype. `renderer` options supply `domain`,
`source_label` and an explicit `gauges` subset; absent gauges request no charts.
References, observations and above-bed depth/terrain mappings are supplied
separately, never guessed from the mixed routing depth field. If `formats` is
omitted, `analysis_csv` and `renderer_request` are the Python writer defaults.
Rebuild the Rust library together with `output.py` for this additive plan field.
See CONTRACT.md for exact schema and boundary behavior.

The history stage launches one thread per reach and a separate scalar cursor
kernel, including when cadence skips a sample. All arrays, including the device
cursor, are checkpoint carriers. There are no CUDA graphs, host copies, scalar
reads, I/O or device allocations in `advance`. History is bounded; buffer
exhaustion fails at the output boundary rather than silently wrapping.

`prepare(settings=...,driver=...,outdir=...)` returns a writer whose `restore`
validates SHA256, the complete typed inventory, driver identity and continuation
metadata before mutating any existing device array. `finish(driver,receipt=...)`
copies all carriers once, writes a CDF5 checkpoint and requested products, and
records boundary timings. An absent or empty requested output directory keeps
its exact layout. Repeating a run into an occupied directory, regular file or
symlink automatically selects a fresh sibling `hydro-run-<unique id>` directory,
preserving previous files. The writer reports that location on stderr and in
`writer.outdir`; the returned and saved receipt includes `output_directory` and
`requested_output_directory`. The finish boundary checks again before the
snapshot in case another caller populated the selected path during the run.
Resume with `--restart PREVIOUS/restart.nc`; the previous checkpoint remains
readable even when the same `--outdir` is requested. Selection does not reserve
the directory against concurrent writers; individual Rust file publications
still refuse replacement of existing files. Driver-owned module-history
diagnostic writers also use this selection policy in driver revision
`9801c4ed3c045b9998416452ccc7d6752d479d64`. With driver revision
`43f8cea2e84ce3e3eb28bcacce0e807fa22f43b2`, ModuleHistory writes a separate
full-state `restart.nc` through this owner's shared snapshot/restore boundary.
Its `history.nc` remains diagnostic and cannot be used as a checkpoint.
Partial diagnostic histories contain the completed prefix; the checkpoint
retains the prepared capacity and complete driver identity. Prepare the same
capacity, ordering and parameters in a fresh process before restoring.
The initialized strict three-catchment Noah/PET/CFE case matches an uninterrupted
48-hour run after 24 hours plus a fresh 24-hour continuation: all 466 declared
carriers and whole final checkpoint/history files match; independent Rust replay
matches all 864 stored native BMI words. This establishes that standalone
composition, with parent/atmosphere/network/terrain/event continuation separate.
The implemented lifecycle, wire schemas, scoped evidence and C ABI
ownership rules are documented in [CONTRACT.md](CONTRACT.md).

`analysis_csv` has `feature_id,time_utc,streamflow_m3_s`, using UTC RFC3339.
It is an analysis interchange and does not claim Pandas byte formatting.
`candidate_json` writes `arwen.point-flow.v1`, units `m3 s-1`, for retrospective
comparison. It requires explicit unique `candidate_ids` in capture order and a
nonempty `provenance` map in the history options. Non-finite flow becomes null;
no archive-to-candidate crosswalk is inferred.
`troute_csv` implements the native tuple-column header and native-real shortest
decimals. Its full file matches the actual patched-native LowerColorado
12-step output. `troute_netcdf` matches all seven variables, including every
typed header and native data byte, in that same case. Native netCDF4-Python
represents an empty `code_version` attribute as one NUL byte; this is retained.
Physical HDF5 header/file bytes differ from the native file; typed header/data
identity does not establish physical file identity.

The Rust series boundary validates the complete format list before creating
any output directory or file. An empty list, unknown format or repeated format
fails without publishing an earlier requested product. This check does not
make the whole multi-format export a transaction: later format-specific or
I/O failures can still leave completed earlier products.

NetCDF-4 is Rust-owned serialization through the **netCDF-C / HDF5 C**
container backend, not a pure-Rust NetCDF-4 encoder. The CDF5 checkpoint
contract remains `arwen.nextgen.checkpoint/v1`, with unchanged array inventory,
metadata and payload SHA validation before GPU mutation. No netCDF-Fortran,
Fortran runtime or native hydrologic model is used. netCDF-C and HDF5 notices
are retained in `NETCDF-C-NOTICE` and `HDF5-NOTICE.txt`.

The writer defines all dimensions, variables and attributes in supplied order,
ends definition once, then writes native arrays. In particular, `_FillValue`
stays in its declared attribute position while still being defined before data.
This also avoids repeatedly relocating CDF5 data as a checkpoint header grows.
Receipts report header definition, payload write and close/fsync/publication
times, format and actual netCDF-C library version. Container calls are
serialized under the library lock; upstream numerical marshalling uses Rust
and Rayon at the output boundary, with no work added per routing step.

Header evidence distinguishes ordered typed metadata and literal `ncdump -h`
bytes from physical HDF5 file bytes. The measured native file uses
netCDF-C 4.10.1/HDF5 2.2.0; this system backend uses netCDF-C 4.9.3/HDF5 1.14.6.
Their generated `_NCProperties` therefore differ. No version attribute is
spoofed, and physical header/file identity is not claimed from a logical dump.

The Rust series boundary validates the complete format list before creating
any output directory or file. An empty list, unknown format or repeated format
fails without publishing an earlier requested product. This check does not
make the whole multi-format export a transaction: later format-specific or
I/O failures can still leave completed earlier products.

NetCDF-4 is Rust-owned serialization through the **netCDF-C / HDF5 C**
container backend, not a pure-Rust NetCDF-4 encoder. The CDF5 checkpoint
contract remains `arwen.nextgen.checkpoint/v1`, with unchanged array inventory,
metadata and payload SHA validation before GPU mutation. No netCDF-Fortran,
Fortran runtime or native hydrologic model is used. netCDF-C and HDF5 notices
are retained in `NETCDF-C-NOTICE` and `HDF5-NOTICE.txt`.

The writer defines all dimensions, variables and attributes in supplied order,
ends definition once, then writes native arrays. In particular, `_FillValue`
stays in its declared attribute position while still being defined before data.
This also avoids repeatedly relocating CDF5 data as a checkpoint header grows.
Receipts report header definition, payload write and close/fsync/publication
times, format and actual netCDF-C library version. Container calls are
serialized under the library lock; upstream numerical marshalling uses Rust
and Rayon at the output boundary, with no work added per routing step.

Header evidence distinguishes ordered typed metadata and literal `ncdump -h`
bytes from physical HDF5 file bytes. The measured native file uses
netCDF-C 4.10.1/HDF5 2.2.0; this system backend uses netCDF-C 4.9.3/HDF5 1.14.6.
Their generated `_NCProperties` therefore differ. No version attribute is
spoofed, and physical header/file identity is not claimed from a logical dump.

Catchment/module histories use the separate Rust boundary operation
`module-series-write`. It accepts native fields without requiring reach
discharge, numerical reach IDs or an area conversion:

```json
{"operation":"module-series-write","plan":{
  "output_directory":"/data/run/catchment-history",
  "origin":"2023-04-02 00:00:00","interval_seconds":3600,
  "cadence_steps":1,"frames":48,"capacity":48,
  "feature_ids":["cat-67","cat-27","cat-52"],
  "fields":[{"component":"cfe","name":"flux_Qout_m","units":"m/step",
             "dtype":"f64le","offset":0,"element_shape":[]}],
  "provenance":{"clock_timezone":"UTC","source_identity":"producer supplied"}
}}
```

Each field occupies contiguous native words with shape
`[capacity, feature_ids.len(), *element_shape]` at its byte offset in the common
snapshot. Empty `element_shape` means one scalar per feature. Fields can have
different primitive integer/float types. Explicit string IDs, field names and
units are preserved; the producer owns their scientific identity. The clock
uses the interval-end convention below. CUDA capture, cursor/overflow state and
restart identity belong to the driver; this operation runs only at the output
boundary through the existing CLI `request` or C ABI/Python `_call`.

The output is `module-fields.csv` with columns
`time,feature_id,component,field,units,element,dtype,value,bits_hex` and a JSON
completion manifest, `module-fields.json`, schema
`arwen.module-field-series.v1`. Rows follow declared field, then frame, feature
and flattened trailing element. `bits_hex` preserves the exact native word,
including signed zero and NaN payload; its most significant byte is first.
The manifest retains the complete plan and CSV SHA256. Rayon formats fields
concurrently. All extents and metadata validate before output creation; an
existing target directory is preserved. The manifest publishes atomically
after CSV fsync and marks the series complete. This interchange is distinct
from native t-route output and carries no fabricated routing semantics.

The native NetCDF filename is `flowveldepth.nc`. Its reader contract is:

| Variable | Native type | Dimensions, in order | Units |
| --- | --- | --- | --- |
| `feature_id` | int64 | `(feature_id)` | explicit segment ID |
| `time` | float64 | `(time)` | `seconds since <origin>` |
| `flow` | float32 | `(feature_id, time)` | `m3 s-1` |
| `velocity` | float32 | `(feature_id, time)` | `m/s` |
| `depth` | float32 | `(feature_id, time)` | `m` |
| `nudge` | float32 | `(feature_id, time)` | `m3 s-1` |

`type(feature_id)` carries the explicitly supplied feature classes. Both axes
have fixed lengths. Time varies fastest in each field; an offline reader maps
native `flow` to its canonical `streamflow` name through metadata. IDs require
an explicit join, including any lake-to-outlet output-ID mapping supplied by
the routing provider. They are never row numbers or inferred gauge IDs.

Sample `j`, starting at zero, labels the interval end at
`(j + 1) * cadence_steps * interval_seconds` after the history origin. There
is no sample at the initial time. The native header spells the origin
`YYYY-MM-DD HH:MM:SS`, and `file_reference_time` uses `YYYY-MM-DD_HH:MM:SS`.
The writer accepts that legacy UTC spelling or RFC3339 with an explicit offset,
including `Z`, and normalizes offsets to UTC before formatting all products.
Native NetCDF units retain the legacy UTC spelling. A timestamp with `T` but
no offset is ambiguous and rejected; nonzero fractional seconds are rejected because
the native headers cannot preserve them.
It has no calendar or timezone attribute; UTC must come from the realization's
clock provenance. The qualified 300-second run starts at UTC origin
`2023-04-02 00:00:00` and has times 300 through 3600 seconds, or
00:05 through 01:00 UTC. Field `_FillValue` is float32 -9999;
`missing_value` is float64 -9999. Time uses float64 -9999 for both.

The captured LowerColorado run has -9999 in all 80,904 native nudge words
(6,742 features by 12 times), including ungauged features. A DA-off producer
must preserve that captured missing-value policy for this configuration;
zero is a different value. Other DA configurations require their own explicit
producer binding and identity. The writer never synthesizes nudge values.

History and its cursor are checkpointed. A resumed finish writes all captured
frames, including the restored prefix, rather than only the new suffix.
Offline consumers should select the final file or join by explicit ID and
valid time when combining partial and resumed files. Comparison and export
are Rust output-boundary work, with no routing-step host processing. The JSON
export remains `arwen.point-flow.v1`; a consumer's reach-series schema is a
separate interchange, not a change to native t-route headers.

`gpuwm.hydro.output.restore_checkpoint(path, driver)` also exposes the complete
checkpoint restore independently of the reach-series writer. It validates the
Rust envelope, digest, full driver inventory/identity and continuation metadata
before mutating existing device arrays. Snapshotting packs noncontiguous views
in logical C order on CUDA; restoration uses CUDA copies into those original
views. Backing slabs, their aliases and empty arrays all remain in the declared
inventory. These temporary copies occur only at output/restore boundaries.
A focused RTX 5070 Ti transport check on an explicit nonblocking stream
preserved all 3,840 native CFE payload bytes across a slab, transpose, sliced and
negative-stride aliases, and an empty view; whole CDF5 files also matched.
Wrong composition rejected before mutation. This checks the strided boundary,
without claiming a full Noah or coupled-catchment restart.

`troute_parquet` implements the native TEEHR long-form output. Set history option
`parquet = {"prefix_ids":"wb", "configuration":"short_range"}` explicitly;
optional `segments` selects numerical feature IDs. Rust sorts the native IDs,
retains the full-roster dataframe indices after filtering, and writes the eight
native nullable columns, microsecond timestamps and reference dates. All eight
columns are formatted, dictionary encoded and Snappy compressed concurrently;
only the ordered container splice is serial. The actual LowerColorado capture
matches all 242,712 rows and the complete ordered physical/logical schema, also
confirmed with the independent native PyArrow/Pandas reader. Rust identifies its
own creator and carries Pandas index metadata; the PyArrow IPC/footer metadata
and whole file bytes differ. Complete native footer byte identity is not claimed.
Apache parquet-rs is pinned in Cargo.lock; its notice is PARQUET-NOTICE.txt.
The older rustwx vendor roster does not contain this added dependency. Fetch the
locked dependencies once into a scratch CARGO_HOME; subsequent builds work offline.
The driver's actual strict case (b) MC+19-level-pool run now also passes
12 continuous steps versus 6+CDF5 restore+6: complete 140-carrier,
11,272,268-byte payload and whole checkpoint/CSV files identical. Its full
6,742-feature, 12-frame Q/V/D history matches all 242,712 native float32 words.
This uses captured native lateral forcing, without runoff or terrain, and
does not qualify a complete coupled forecast. That physical proof is separate
from the focused GPU transport check below, which copies native output words.

Focused referee tools:

```
rw_nextgen_output capture-nexus-csv upstream.csv NEW_FIXTURE_DIR
rw_nextgen_output capture-native-nexus ID_output.raw NEW_FIXTURE_DIR
rw_nextgen_output capture-troute-netcdf NATIVE.nc NEW_FIXTURE_DIR
rw_nextgen_output nexus NEW_FIXTURE_DIR/plan.json
rw_nextgen_output replay-netcdf upstream.nc NEW_OUTPUT.nc
rw_nextgen_output compare-netcdf LEFT.nc RIGHT.nc
rw_nextgen_output compare-parquet LEFT.parquet RIGHT.parquet
rw_nextgen_output compare-series NATIVE_REQUEST.json NATIVE.raw GPU_REQUEST.json GPU.raw flow,velocity,depth,nudge
rw_nextgen_output pack-history INPUT_PLAN.json FIELDS.json NEW_DIRECTORY
rw_nextgen_output remap-ids INPUT.i64 CROSSWALK.json NEW_OUTPUT.i64
rw_nextgen_output request REQUEST.json INPUT.raw OUTPUT.raw
```

The first command captures already-rounded text, not native module inputs.
The second consumes the referee's packed native `receive_data_entry` capture
(i64le index, 19-byte timestamp, unrounded f64le flow). Numerical words are
copied unchanged before Rust formatting.
The third command copies actual native f32 Q/V/D/nudge words into time-major
history, without decimal conversion, then builds a series request. Decimal
candidate IDs are derived in Rust from the native feature_id values; another
provider or crosswalk must supply its own explicit candidate IDs.
NetCDF replay compares every ordered header field and every native-type data
byte and separately reports whole-file identity. It never widens fields through
float64 or excludes variables. A shipped upstream output is a transport fixture,
not a fresh coupled run or case (b) restart/resume acceptance.

`compare-series` compares native-precision words after aligning the complete
explicit int64 feature rosters, requiring equal clocks and cadences. The
clock comparison uses the writer's UTC parser: legacy UTC, RFC3339 Z and explicit
offsets compare by instant. Invalid or nonzero fractional origins reject even
when both strings are identical; the receipt records the canonical `origin_utc`.
The field list is explicit; any unselected native fields are listed in the receipt.
Mismatches retain numerical words/feature IDs/time indices and exit nonzero.
This offline tool does not infer a reach/lake crosswalk or treat a missing native
feature as an exclusion. A focused check against the prior GPU output transport
compares all four fields, 323,616 words exactly; a deliberately wrong flow binding
fails with witnesses. This validates the comparator, not routing physics.

`pack-history` takes a series request and a JSON object mapping `reach_id`,
`flow`, and optional `velocity`, `depth`, `nudge` to raw-array paths. Rust checks
the declared native extents, copies the original words into `snapshot.raw`, and
writes `request.json` with corrected offsets. IDs are explicit int64 values;
candidate decimal IDs are decoded in Rust. Its new directory also receives
products when the resulting request is executed. No absent field is invented.

`remap-ids` applies an explicit JSON integer-to-integer metadata table to the
int64 roster, preserving order. It rejects missing source IDs and duplicate
output IDs. This supports captured native lake-to-outlet output labels without
changing routing state or numerical history. Production components should bind
their prepared immutable output-ID vector directly.

`troute_transport_replay.py --capture CAPTURE_DIR --out NEW_DIR` uses the fresh
resident-driver contract and the production output component on a GPU. A CUDA
copy stage advances the captured native words; 12 continuous steps are compared
with a 6+6-step checkpoint continuation across the complete carrier inventory.
This verifies output transport at the real feature count, not routing numerics.

`gpuwm.hydro.output.restore_checkpoint(path, driver)` also exposes the complete
checkpoint restore independently of the reach-series writer. It validates the
Rust envelope, digest, full driver inventory/identity and continuation metadata
before mutating existing device arrays. Snapshotting packs noncontiguous views
in logical C order on CUDA; restoration uses CUDA copies into those original
views. Backing slabs, their aliases and empty arrays all remain in the declared
inventory. These temporary copies occur only at output/restore boundaries.
A focused RTX 5070 Ti transport check on an explicit nonblocking stream
preserved all 3,840 native CFE payload bytes across a slab, transpose, sliced and
negative-stride aliases, and an empty view; whole CDF5 files also matched.
Wrong composition rejected before mutation. This checks the strided boundary,
without claiming a full Noah or coupled-catchment restart.
