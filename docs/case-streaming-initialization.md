# Ordinary case initialization with a host store

`gpuwm run CONFIG --outdir OUT` resolves the configured `[tiles]` plan before
initializing a domain. A resident run keeps the original preparation path.
A host-store run uses the same CUDA preprocessing transforms, writes the
initialized host state through the prepared-cache contract, and builds GPU
physics through the existing row-slab loader. The ordinary forecast loop,
output writer, and restart implementation remain the execution path.

The internal cache is temporary. `OUT/initialization.json` retains its content
identity, actual initialization memory inputs, slab/store inventory, and
whole-domain health coverage. The memory record states its scope: remaining
initialization after horizontal interpolation, with an explicit known host
allocation floor. It does not claim an upper bound on horizontal workspace or
all initializer temporaries. The pinned-store allocator separately checks the
complete store request before its first pinned allocation. A streamed forecast
does not imply that every preprocessing allocation will fit.

Storage does not change the configured physics, eta levels, land-surface
solution, reconciled soil category, SST, radiation gas override, column chunk,
or forcing times. The ordinary single-domain loop retains its elapsed-time
boundary semantics. The state-free store builder is passed `clock=None`
deliberately; domain-tree clock semantics are a separate existing contract.

The validation witness uses a public two-time ERA5 input on a 64×64×49 domain
with legacy radiation 4/4. The resident and host-store ordinary commands each
complete 120 model seconds and produce three byte-identical WRF output files.
Restarting the host-store run from 60 seconds reproduces the final file and all
136 common checkpoint arrays exactly. The host-store health gate covers 132
fields over the whole domain with no missing store fields. These measurements
establish the storage equivalence at this size; acceptance above physical VRAM
is a separate system measurement.

Permanent tests cover CUDA transforms feeding both CUDA and host states,
including all five analyzed mass species, exact serialized-state and coupled
boundary values, nonlinear forcing hydration, row-context preservation,
allocation order, whole-store health, and capacity refusal controls. The actual
case slab-lifetime test uses `GPUWM_TEST_RUNTIME_STORE_CASE` to locate a staged
ordinary-case fixture and reports a skip when it is absent.
