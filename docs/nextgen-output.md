# NextGen output and restart scope

This is an implementation and evidence snapshot, dated 2026-10-18. It describes
output owner `20e74aca3d1e047f45b117ae0a000597c8487ab8`, Python boundary
`2df892bd8063ac101c10f7ac76f2a2e30c32c5d9`, and the owner's contract commit
`e73b0b73e2c6901b0e0ba748b774f6d6386de0ed`. The original strict standalone
continuation separately pins driver `43f8cea2e84ce3e3eb28bcacce0e807fa22f43b2` and output
owner `8edfccbd63523c9bc4412bce1665d124feba1150`, imported by that driver as
`9d741ac1b20`. The PET v2 fast continuation below pins driver
`15b52b6157be07baa3e2db594c9415d4de91997c`. These implementations are
not imported by this documentation change. A deployment needs the matching
writer and driver; the measurements below do not qualify an installed wheel.

## Publication and recovery

The reach writer returns `restart.nc`, selected history products and
`receipt.json`. An absent or empty output directory keeps the requested
layout. An occupied directory, file or symlink selects a fresh sibling and
preserves earlier results. The writer checks again before taking the final
snapshot so a late-arriving file is preserved too. Returned and saved receipts
record `requested_output_directory` and `output_directory`; use the actual
path when chaining commands.

The diagnostic `ModuleHistory` writer has a separate recovery implementation
at driver commit `9801c4ed3c045b9998416452ccc7d6752d479d64`. Its actual root
CLI/CUDA/Rust regression preserves prior output and late-arriving files, with
zero differences across 336 typed field bytes. That test uses instrumented
state and does not establish physical forecast or installed-product acceptance.

Individual Rust files publish without replacing existing files. Reach output
does not reserve a whole directory for concurrent writers or publish every
product as one transaction. A later export failure can leave a valid
checkpoint and earlier completed products. Read the returned receipt and
the command result before treating the requested export set as complete.

## Histories and timestamps

History capture is allocated and registered before the driver's inventory is
sealed. The timed step only launches CUDA copies and a cursor update; transfer
to the host, serialization and file I/O happen at the output boundary.

Native reach NetCDF requires explicit int64 publication IDs and flow,
velocity, above-bed depth and nudge arrays in their native float32 or float64
precision. A driver that has no nudge array cannot produce that native layout;
the writer does not invent missing values. Other supported exports include
analysis CSV, t-route CSV/Parquet, nexus CSV and point-flow candidate JSON.

History timestamps are interval ends. The output implementation accepts native
UTC spelling and RFC3339 origins with `Z` or an explicit offset, normalizing
the latter to UTC. Ambiguous timestamps without an offset and nonzero
fractional seconds are rejected because native headers retain whole seconds.

Measured t-route output matches seven variables containing 1,348,496 typed
bytes and their ordered typed headers. The 1,180-byte literal ncdump header
also matches after removing only the filename declaration. Physical HDF5
files differ with container versions and properties. Parquet ordered schema
and values match; footer metadata is not byte-identical.

## Measured routing continuation

Reservoir owner `90787ee07d485c2ba4146fea5f77ab815866609e` records candidate
`ca07e99de864e8fbd97d1fb224cdff9b706e3dcc`, runner
`6ec17695a7d8d22d25e83d8698511a16d7993b2b`, MC
`bab7f09df8c2878048adecff0d64379592988ac0` and driver
`5509567df428297fc7a2b116e9fd21326795e94a` with the output versions above.

For that exact 148-array routing/history inventory, six steps followed by
restore and six more produce a checkpoint byte-identical to an uninterrupted
12-step run in both fast and strict modes. Original device pointers remain
stable. The composition contains Muskingum-Cunge routing, seven level-pool
and twelve RFC reservoirs. Strict flow/velocity/depth comparison against
native is zero differing words out of 242,712; fast retains 52,731 differences.
Native t-route uses the disclosed secant-seed repair, rather than untouched
upstream. Default fast mode is not qualified as bitwise native agreement.

This proves continuation of that routing/history inventory. The initialized
catchment composition below is a separate measurement; the two results do
not establish a combined runoff, routing, terrain and atmosphere restart.

The later reservoir carrier correction at integration
`2c9b0295f99ac132a67759fa1515d838fef882f5` preserves native float32 assignment
inside float64 restart storage. Its fresh root-CLI check writes all 148 arrays
and reaches step 12 / 3,600 seconds; it does not repeat this resume comparison.
See [carrier precision and current routing
measurements](nextgen-development-evidence.md#reservoir-carrier-precision).

## Initialized standalone continuation

At the separately pinned driver version above, `ModuleHistory.Writer` writes
a full-state `restart.nc` alongside diagnostic `history.nc`, using the shared
output checkpoint serializer and reader. Resume uses `restart.nc`;
`history.nc` remains diagnostic and is refused as a checkpoint before state
mutation. Partial histories publish only completed steps. The checkpoint
retains every declared carrier and the full prepared history capacity.

Preparing the destination must reproduce the checkpoint's component order,
parameters, numerics, cadence, origin and history capacity. A run stopping
after 24 hours of a prepared 48-hour history keeps that 48-hour capacity on
resume. Shortening the first command's duration does not change the identity
or shrink the prepared state.

The actual strict case uses native initialization and forcing for ordered
catchments `cat-67`, `cat-27`, `cat-52`, hourly intervals and PET methods
1, 5, 3. An uninterrupted 48-hour run and a 24-hour run followed by a fresh
process restoring and advancing another 24 hours agree across all 466
declared carriers, containing 232,708 logical bytes. The final 427,024-byte
checkpoint and 63,268-byte history are each whole-file byte-identical.
Resumed history has zero differing words out of 864 stored native BMI words.
The five Rust-rendered diagnostics show coincident native and resumed runoff
recessions, PET peaks and small Noah pulses, with no visible 24-hour jump.

The owner's shared-pool RTX 4090 measurement is 0.189709 ms per hourly model
step, over 47 steps after one retained warmup step. Each stage handles three
catchments in one 128-thread block, except Noah's 64-thread block; history
also updates its cursor. A separate fresh-state profile after one warmup gives:

| Stage | Milliseconds |
|---|---:|
| Native float64 forcing | 0.004096 |
| Float32 forcing | 0.004096 |
| PET | 0.014336 |
| Noah | 0.151552 |
| CFE | 0.013312 |
| History capture | 0.006144 |

These profile samples include enqueue gaps and are separate from the batch
timing. Production uses ordinary launches, two batch waits and no CUDA graphs
or timing events. Checkpoint transfer and Rust publication occur outside
the step. The shared-pool result is not an isolated-card benchmark or the
complete hydro speed line.

This closes declared-state continuation for initialized standalone Noah/PET/CFE.
It supplies no reach or terrain routing, parent atmosphere, event warmup or
USGS skill evidence. Physical parent and complete forecast continuation remain
pending. See the [coupling contract](nextgen-coupling.md).

## PET v2 fast continuation

Driver `15b52b6157be07baa3e2db594c9415d4de91997c` repeats the initialized
three-catchment test in fast numerics with corrected PET identity
`noaa-owp-pet-cuda-v2`. A 24-hour run followed by a fresh process restoring
`restart.nc` and advancing another 24 hours matches the same-version
uninterrupted 48-hour run. All 466 carriers and 232,708 logical bytes agree;
the final 427,484-byte checkpoint and 63,628-byte history each match as whole
files. This closes continuation for the new PET identity. Old PET v1
checkpoints still require fresh preparation; identity validation stays intact.

Restart agreement and native agreement have separate results. The restored
fast history differs from native in 88 of 864 words: five Noah runoff words,
five CFE rainfall words and 26 words each in CFE runoff, soil storage and
groundwater storage. PET matches all 144 words. The first differences occur
at hour 36 in catchment 67. No tolerance or maths-library witness qualifies
these differences as bitwise agreement. The earlier strict result remains
at its own version and numerics.

The five existing Rust-rendered fast diagnostics were inspected again. Model
and native curves visually overlap through CFE recessions, PET daytime peaks
and small Noah runoff pulses. Because the resumed history is whole-file
identical to the uninterrupted history, these pictures also represent the
restored result; they were not rendered from a new trajectory. Visual overlap
does not close the differing-word result or establish USGS skill.

The owner's shared-pool RTX 4090 batches include the first enqueue, with no
retained warmup:

| Run | Steps | Batch milliseconds | Milliseconds per step |
|---|---:|---:|---:|
| Uninterrupted | 48 | 8.295582 | 0.172825 |
| First half | 24 | 5.248065 | 0.218669 |
| Fresh-process resumed half | 24 | 7.740047 | 0.322502 |

Each batch has two boundary waits, host work limited to the prepared launch
sequence, and no production timing events or CUDA graphs. Three catchments
use 128-thread blocks except Noah's 64-thread block; history also advances
its cursor. A separate cold first-interval profile at the same source gives:

| Stage | Milliseconds |
|---|---:|
| Float64 forcing | 0.033984 |
| Float32 forcing | 0.009248 |
| PET | 0.018176 |
| Noah | 2.416736 |
| CFE | 0.013216 |
| History capture | 0.007168 |

These event brackets include enqueue gaps and are not resumed throughput or
an additive step budget. Output boundaries take 234.745177 and 155.384223 ms
for the first and resumed halves, outside the step; Rust marshals the data
and serialized netCDF-C calls write the containers. This is a pooled-card
standalone result. It adds no routing, terrain, parent-atmosphere or complete
hydro speed measurement.

## Serialization and packaging

Rust owns serialization using permissive netCDF-C/HDF5-C/zlib containers;
NetCDF-4 is not a pure Rust implementation. No model or Fortran runtime is
installed on the user path. The measured static Linux stage requires
GLIBC 2.38 and has not qualified manylinux or an installed product. Output
boundary timings are separate from the GPU routing-step speed target.
