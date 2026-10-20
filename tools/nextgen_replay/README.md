# NextGen sealed raw verification

Original Apache-2.0 verification infrastructure, written fresh from clean
`integrate/2.9.2` at `64fc403dfd4`. Project LICENSE at that pin says “Apache
License”, “Version 2.0, January 2004”. No withdrawn port code, UCAR program,
source, fixture, or ABI is used. This tool is not a numerical referee.

The terrain-2d-verify lane owns this shared comparator and pinned-host staging
path. Replay consumers own their module-specific native-capture decoding and
CUDA runners in their own tool directories; the referee owns capture schemas.
Emit the descriptor below rather than adding module names or adapters here.
Request shared contract changes from this lane to avoid concurrent edits.
The fresh generic implementation is commit
`8f6a99aba0010f7ef5f2af99feafc2aec47ab086`; withdrawn tooling is separate.

Build from this directory with all generated artifacts under lane scratch:

```sh
export CARGO_HOME="$scratch/cargo-home" CARGO_TARGET_DIR="$scratch/target"
export TMPDIR="$scratch/tmp"
cargo build --release --offline --locked -j 8
```

The checked-in vendor provides Rayon 1.12.0, serde 1.0.229, serde_json 1.0.151,
sha2 0.10.9, each licensed `MIT OR Apache-2.0`; their LICENSE-APACHE and
LICENSE-MIT files are retained in `tools/rustwx/vendor/crates-io`. The Cargo.lock
pins transitive dependencies. No native/Fortran library is linked or installed.

## Comparison contract

`nextgen-replay REQUEST.json NEW_VERDICT.json`: exit 0 exact, 1 differing stored
words, 2 malformed request, missing/incorrect seal, wrong size, or I/O failure.
The verdict must be a new file. All paths are explicit, resolved relative to the
process working directory; absolute paths are recommended for receipts.

Request schema `nextgen.raw.compare/v1`:

```json
{
  "schema": "nextgen.raw.compare/v1",
  "unit": "module-name",
  "case": "case-and-step-id",
  "numerics": "strict",
  "candidate_revision": "actual-full-candidate-commit",
  "command": ["actual-command", "actual-arguments"],
  "provenance": {
    "origin": "native-bmi",
    "sources": [{"name": "upstream-repository", "revision": "actual-pin", "licence": "actual-permissive-licence"}],
    "evidence": [{"path": "/sealed/native-capture-index.json", "sha256": "ACTUAL_64_HEX_DIGITS"}]
  },
  "fields": [{
    "name": "step-0/output-name",
    "dtype": "f64",
    "shape": [12, 4],
    "order": "fortran",
    "units": "m",
    "expected": {"path": "/sealed/expected.raw", "sha256": "ACTUAL_64_HEX_DIGITS"},
    "actual": "/new/candidate.raw"
  }]
}
```

This example is a schema illustration, not replay evidence. The module-owned
sealed capture index must record source/build pins and flags, process count,
entry/exit names, timestep, BMI variables, dtype, shape, lower bounds, order and
units. Referee owns native capture schemas. No BMI names or guessed interfaces
are embedded in this generic comparator. Preserve native word streams/order;
`shape: []` means one scalar. `units` is optional metadata. Supported dtype is
f32, f64, i32, i64 or u8; all files are packed little-endian, with no record
markers. Order is `fortran` or `c`, with no conversion. Exact words include
signed zero and NaN payloads. There is no tolerance, field mask, ULP waiver, or
implicit exclusion. Invalid evidence produces no successful verdict. Different
fields and word chunks run on Rayon workers; `RAYON_NUM_THREADS` selects the
offline worker count. Verdicts include both hashes and the first differing
word's index and stored bit patterns.

Use `origin: "tool-self-test"` for synthetic integrity checks, never
`native-bmi`. Terrain-2d uses observed USGS/NWM verification, not UCAR replay.

## Pinned host staging C ABI

`include/nextgen_raw.h` is the authoritative C declaration. Build produces
`libnextgen_replay.so`. `nextgen_raw_load`, `nextgen_raw_store`, and
`nextgen_raw_sha256` handle individual files. `nextgen_raw_read_many` and
`nextgen_raw_write_many` parallelize independent fields using Rayon. No Python
payload decode, hash, packing or comparison is needed. Host byte buffers may be
CUDA pinned allocations. These functions take **host pointers**, never device
pointers; the caller enqueues CUDA copies at setup/export boundaries only.
All buffers, paths and strings must remain alive until the synchronous call
returns. Read destinations must not overlap. A failed batch can leave other
items completed: discard all candidate input state from a failed batch. Write
destinations must be new; failed writes are not transactional. C calls return
0/2 and put diagnostics in an optional NUL-terminated error buffer. Setup and
offline I/O are outside the resident GPU numerical step.

The crate provides raw I/O as well as comparison; consumers do not need another
I/O cdylib. For a caller-owned pinned host buffer, use
`nextgen_raw_load(path, sha256, host_destination, bytes, error, error_cap)` and
`nextgen_raw_store(new_path, host_source, bytes, error, error_cap)`.
The load checks the seal and exact byte count before copying into that buffer.
There is no direct-device-pointer entry point. The orchestration bridge below
owns pinned staging and enqueues native CUDA copies; Rust performs payload I/O.

For packed native snapshots, use the additive host-buffer API:

```c
int nextgen_raw_load_slice(const char *path, const char *sha256, size_t offset,
                           uint8_t *destination, size_t bytes,
                           char *error, size_t error_cap);
```

The SHA seals the **entire snapshot**, not the selected range. This call reads
and hashes the whole file, checks offset/length overflow and file bounds, then
copies the selected bytes without alignment requirements or numeric conversion.
Any validation failure leaves the destination unchanged. A zero-length range
at EOF is valid with a non-null destination. Module owners supply capture
offsets, types, shapes, bounds and strides; the common crate parses no native
metadata format. For many fields from one snapshot, `load_owned` once followed
by host-pointer offsets avoids repeated reads/hashes. These are offline/setup
operations, never resident numerical-step work. The existing ABI and comparison
schema remain unchanged.

Consumers that need Rust-owned host memory can use the additive ABI:

```c
int nextgen_raw_alloc(size_t bytes, uint8_t **output, char *error, size_t error_cap);
int nextgen_raw_load_owned(const char *path, const char *sha256, size_t bytes,
                           uint8_t **output, char *error, size_t error_cap);
int nextgen_raw_free(uint8_t *pointer, char *error, size_t error_cap);
```

All return 0/2 like the existing ABI. `alloc` zero-initializes the requested
bytes; `load_owned` validates SHA and exact length before returning ownership.
With a valid output argument, failure sets it to NULL. Do not pass an output
slot that already owns a buffer. Even a zero-byte success returns a distinct
non-null pointer. Save using the existing `nextgen_raw_store(new_path, pointer,
bytes, error, error_cap)`; no additional save implementation is needed.

These are **pageable host bytes**, not CUDA-pinned allocations. Python may pass
the pointer to native CUDA memcpy, without viewing/decoding the payload. Complete
every CUDA copy before freeing; synchronous memcpy is suitable for this optional
setup/export path. For pinned asynchronous staging, keep using `RawCuda` below
and the caller-owned ABI. Free with this same loaded library, exactly once;
never use Python/C/CUDA deallocators. `free(NULL)` succeeds; foreign/non-live
addresses return 2. As with any pointer ABI, a stale pointer must never be reused
after a subsequent allocation, which might have recycled its address. The
registry retains memory until explicit free or process exit. Its ownership lock
is only for setup/export, never a GPU step. Neither path changes the comparator
schema or numerical processing.

## Actual tool checks

```sh
NEXTGEN_TEST_ROOT="$scratch/tmp" NEXTGEN_TEST_REVISION=ACTUAL_COMMIT \
RAYON_NUM_THREADS=8 cargo test --release --offline --locked -j 8 \
  --test raw_contract -- --nocapture
```

These execute the real CLI and C ABI: NaN/signed-zero exactness and mismatch,
bad seals, shape mismatch, duplicate fields, parallel byte staging, failed-seal
destination preservation, overlapping destinations and overwrite prevention.
They create tiny synthetic artifacts only under `NEXTGEN_TEST_ROOT`, which must
already exist. They establish tool integrity, not native physics or obs skill.

## CUDA staging at replay boundaries

`gpuwm.nextgen.raw_cuda` is the canonical product import. The local
`raw_cuda.py` re-exports the same `RawCuda` class for existing replay scripts;
it contains no second implementation. Stage the product module alongside the
compatibility import in isolated snapshots. Production factories import from
`gpuwm.nextgen.raw_cuda` without depending on a tools directory on `sys.path`.
Both expose `RawCuda(library).load_many(specs, stream=...)`,
`store_many([(device_array, new_path), ...], stream=...)`, and `sha256(path)`.
Each input spec has explicit `path`, `sha256`, `dtype`, `shape` and `order`
as above. It allocates CUDA arrays and pinned host buffers at setup, calls the
Rust batch ABI, enqueues byte copies, then synchronizes once before releasing
the pins. Export copies into pinned buffers, synchronizes once, and writes
through Rust. No Python payload conversion, hashing or comparison occurs.
Call only at setup/export boundaries, on the arrays' card and ordered stream.
Caller owns module-specific native capture decoding/coalescing; no generic
BMI variable names, lower-bound conversion or packed-capture parser is implied.

For example, with a sealed descriptor `spec` and a new output path:

```python
from gpuwm.nextgen.raw_cuda import RawCuda

raw = RawCuda(library_path)  # target/release/libnextgen_replay.so
arrays = raw.load_many({"state": spec}, stream=stream)
# Enqueue the module's CUDA launches on this same stream.
raw.store_many([(arrays["state"], new_output_path)], stream=stream)
```

Here `spec` has `path`, `sha256`, `dtype`, full `shape` and `order` (`c` or
`fortran`). This is API usage, not a native replay receipt. No NumPy payload,
`fromfile`, `tobytes`, Python hash or host array conversion is required.

The generic Rust comparator and host ABI were introduced in commit
`8f6a99aba0010f7ef5f2af99feafc2aec47ab086`. The CUDA bridge was introduced
in `7c69b0b689097b813718b80c257ae65c91e70c9e`, which also contains terrain
diagnostic files. Consumers adopting only the bridge should extract just
`tools/nextgen_replay/raw_cuda.py` from that commit and make their own forward
commit; do not cherry-pick the mixed commit merely to obtain staging. The
bridge uses this fresh NextGen host ABI, not the withdrawn hydro replay ABI.
The canonical product module retains the original bridge's exact bytes
(SHA256 `c1442a0e90412be2771cdf38804330c67206e3dc2e91a3bf7127b2f21090782c`),
matching the SFT owner's product adoption. For current consumers, adopt both
`gpuwm/nextgen/raw_cuda.py` and the tools compatibility import together rather
than copying the implementation into another module. Existing sealed historical
snapshots retain their originally executed bridge and source hashes.

The fresh terrain diagnostic's executed synthetic fixture and invocation are
documented in [NEXTGEN-INUNDATION.md](../../docs/technical/NEXTGEN-INUNDATION.md).
