# Experimental native launch recorder

This is a low-level launch-plan prototype. It is not connected to the model
step dispatcher and is not a qualified whole-model optimization.

Build with `python3 -B tools/build_native_step.py --output PATH/libawstep.so`.
Start a separate forecast child with that absolute library path in both
`LD_PRELOAD` and `GPUWM_NATIVE_STEP_LIBRARY`, before CuPy imports. Linux/glibc
and a CUDA driver exporting parameter-layout queries are required. CUDA 13
context-independent kernel handles use `cuKernelGetParamInfo`.

`Plan.record(*resources)` executes the original Python commands. Kernel
parameters are copied by their actual driver-reported byte layout. A private
CuPy pool permits temporary storage reuse during recording and reserves the
backing allocations for subsequent native submissions. Keep modules, streams,
events, state objects and any externally allocated arrays in `resources`.

`Plan.bind(operation, argument, Frame, field)` replaces one complete kernel
argument with an exactly sized ctypes structure field. `Plan.when(require=N,
reject=M)` records a command group under explicit per-submission predicate
bits. `Plan.submit(frame, flags=N)` makes one native FFI call and issues ordinary
CUDA launches, copies, fills and event operations. It preserves captured synchronizations and native guards; it adds no final
synchronization. Complete the relevant streams before closing the plan or destroying
referenced resources. A plan belongs to one host thread and CUDA context.

The recorder does not infer model time, boundary changes, host state mutations,
branch semantics, external allocation lifetimes or data-dependent array sizes.
A model adapter must supply those contracts. Unannotated host transfers are rejected in a
replayable recording. `Plan(diagnostic=True)` records their call stacks but is
permanently non-replayable. Known unsupported launch/copy APIs are rejected;
this interposer is not a universal CUDA interception or completeness guarantee.
Driver errors after partial replay invalidate the plan. A caller must not
retry an already partially executed model step from its mutated state.

The real CUDA contract test is `tools/native_step/probe_contract.py`. It checks
conditional scalar binding, cross-stream event ordering, device copies,
pre-submission frame rejection, rejected host transfers and storage reuse.
Those are backend contracts, not weather-field identity or speed evidence.

Explicit `constant_uploads()` retains only caller-declared lifetime-invariant
host bytes in owned pinned storage. `private_events=True` additionally declares
recorded events in that scope to be private upload bookkeeping with no external
observer. Replay owns independent events, so a CuPy cleanup cannot invalidate
its event handles. Public events keep their caller-owned lifetime contract;
external waits or reuse of a declared private event refuse capture.

`zero_guard(error, word_bytes=1, mask=None)` records
one host readback, stream completion and native zero-status validation. It
stops before downstream submissions, invalidates the plan on failure, and
constructs the caller-specified original exception. This is not rollback.
Do not publish candidate state before validation. Predicate changes inside a
host scope are rejected. `tools/native_step/probe_host_contracts.py` checks
real model exceptions, immutable source lifetime and downstream write stops.
An optional uint64 mask restricts zero checks to selected words of a result
with at most 64 words. `finite_guard(error, word_bytes=4, mask=...)` checks
selected IEEE float32 or float64 words for NaN or infinity without changing
signed zeros or subnormals. Its failure callback receives the raw failing bits.
This predicate does not validate derived arithmetic or scientific thresholds.

Place any required cleanup inside the guard scope. For example, the pinned
HealthLedger adapter includes the ordinary device clear before the native guard
raises, preserving the ledger's clear-before-refusal behavior. These explicit
consumers preserve existing checks; they do not authorize omitting a health
check or publishing partially submitted state.

The AC diagnostic now annotates three uploads and six native validation
guards. Its callsite annotations are an audit harness, not a model adapter.
The diagnostic now admits only unchanged Noah-MP layouts; the fixed-width
vegetation path is described below. Explicit calendar and boundary slots have
separate advancing-domain proofs, and ordinary radiation fallback has a
forty-interval STRICT/FAST proof. Those examples are not a production admission
adapter: changing layouts, general cadence, scheduling and error publication
still require model-level contracts. Do not
replace `dycore.step` with this plan until those dependencies and the required
strict/FAST, same-shape wall and separate profile receipts are complete.

The fixed-width vegetation path removes the data-dependent host cardinality
from the assembled Noah-MP column call. `tools/native_step/measure_vegetation.py`
compares real AC inputs against the prior assembled column and tests live mask
changes through one recorded leaf sequence. `probe_domain_step.py` additionally
replays complete parent/child domain CUDA sequences at fixed captured time.
Its Python metadata is held from capture and its source includes this vegetation
candidate. It is not a changing-time model adapter or a completed parent/nest
qualification. No production native dispatch is enabled by these probes.

`Plan.numeric_guard` is an explicit one-readback binary64 consumer for an
admitted adapter. Its copied per-word rule vector supports numeric nonzero and
below-threshold checks with negative infinity excluded; signed zero and NaN
follow IEEE comparisons. Rules must cover the readback exactly. It stops before
downstream operations and invalidates a failed plan. The error factory runs only
on failure and receives the first failing word's original bits.

The new deferred-radiation diagnostic pins six original verdict sources and
retains the concatenate descriptor upload. It preserves original verdict order
and calls the original converters/verdicts only on failure. This is not native
radiation: solar, root climatology and day/night topology remain unresolved.
CPU predicate evidence alone does not qualify its CUDA recording/replay path.

## Compiled host consumers

`Plan.readback()` records exactly one D2H transfer and its stream completion.
`Plan.host_buffer()` borrows that plan-owned pinned payload for a later compiled
consumer. An unborrowed payload refuses replay before any operation.
`Plan.native_host(library, symbol, ctypes_struct, *owners, error=...)` records
`int(const void *, size_t)` and retains the provider library and declared owners.
The helper runs at capture and replay. Argument 0 may bind a same-sized frame
field, so calendar scalars need no recapture. Return -1007 reconstructs the
original exception after submission stops. Other nonzero results are recorder
errors. A failed partial plan is invalid, with no rollback.

A helper cannot enter Python, call CUDA, or read outstanding device work. Its
readback must precede it and execute under compatible predicates. Pointer
owners must remain alive. These are explicit adapter obligations. Mutable upload sources must use the explicit native_upload contract below.
No radiation dispatcher uses this API yet. The CPU and CUDA integration
fixtures in tools/native_step exercise transport and callback semantics only.

## Native-produced uploads and live counts

`Plan.native_upload()` retains exactly one H2D buffer and private copy events.
`host_buffer()` borrows it for an earlier native producer, bound through that
producer's frame struct. The original Python source is used only at capture.
Each recorded upload completes on its stream before a later submission can
overwrite the same buffer. Missing ownership acknowledgement refuses replay
before any operations. Producer ordering and compatible predicates remain
explicit caller contracts.

`count_predicate(count, greater_than=threshold, bit=index)` records a live
uint64 comparison after the native count producer. It sets or clears only
that bit for later operation groups. It does not make padded physics chunks
safe: an adapter must still guard every read and output publication using
current counts. The CUDA fixture proves transfer, lifetime and group selection,
not radiation integration or model performance.

A synchronized readback can supply the exact bytes of a later kernel scalar with `bind_host_argument(operation, argument, readback, byte_offset=0)`. The source must precede the kernel, cover its predicate, and contain the entire argument. Intervening count predicates cannot alter the source predicate bits. The binding retains owned readback storage and acknowledges its consumer. A later frame binding replaces it, and a later readback binding replaces a frame slot. This does not admit arbitrary host arithmetic or native producer output as a scalar source.

## Experimental composed radiation witness

The source-pinned AC adapter in `tools/native_step/radiation_capture.py` is a
component prototype. It preserves the original vertical-weight route and the
canonical child ozone snapshot body. It binds calendar scalars, uses a native
root climatology producer, and compacts daylight indices with live counts.
Padded shortwave chunks use count-guarded output writes and conditional deferred
verdicts. Immutable upload events have explicit private ownership.

The port admits only the named fixed-geometry LW/SW legacy configuration,
canonical device vertical weights, and the existing parent-interpolated child
ozone carrier. A different provider, rebound carrier, changed vertical binding,
unsupported diagnostic provider or changed trace-gas year refuses. This is not
admission for moving geometry or general radiation options. No capability or
production default is disabled.

`probe_radiation_replay.py` is the pending composed GPU witness. It compares
ordinary and captured outputs at actual parent/child radiation calls, then
replays current night, daytime and night again at fixed sampled atmospheric
inputs. The child also receives an owned changed-ozone control. Model inputs
must stay unchanged. Every replay uses one `Plan.submit` call, followed by stream
completion and explicit night-metadata publication. Adapter update counters and
whole-model carrier/history publication are outside this component witness.
See its result JSON for actual execution status; CPU checks alone are not a
successful GPU replay.

Build the recorder with `python3 tools/build_native_step.py --output <library>`.
The separate component host helper builds without CUDA initialization:

```sh
cc -shared -fPIC -O2 -std=c11 -Wall -Wextra -Werror -fopenmp   tools/native_step/radiation_composed_host.c -o <host-library>
```

The lane runner names retained library paths and uses the existing prepared
input binding. GPU execution must follow the current pool and time-slice rules.
The full model still needs held-buffer publication, independent parent/child
cadence, failure handling and original-base qualification before production use.
