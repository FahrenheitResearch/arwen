# Compiler-family kernel packs

The normal kernel-plan startup also installs the prebuilt-image loader. It
selects `gpuwm/data/kernel-pack-cuda12.json` or `kernel-pack-cuda13.json` from
the loaded NVRTC major version, with `kernel-pack.json` as the legacy fallback.
`GPUWM_AOT_PACK` selects an explicit pack. The filename alone never establishes
compatibility: the loader checks the compiler packages, NVRTC library digest,
source, numerical options, target architecture, request metadata and image
digest. A miss uses the ordinary compiler.

The codec is the same `gpuwm.kernel-pack/v1` contract used by hex. Both CuPy's
cached module route, ArWen's direct NVRTC route and the final NVRTC funnel use it. Template symbol
aliases are restored on load. Strict and fast recipes remain separate, even
when they are built in one pool whose environment defaults to fast.
`GPUWM_WRF_EXACT=1` also normalizes its arithmetic controls before direct-cache,
warm-plan and pack lookup. A fast image cannot satisfy that request. Packs or
catalogs recorded without those controls need separate exact-mode entries;
otherwise the loader falls back to compilation.

Final-funnel recipes use `raw-nvrtc`. Their options already contain CuPy's
additions: the builder preserves them and the exact program filename rather
than appending flags or recomputing a cache name. Ordered template names and
their lowered mappings are preserved. PTX, linking and memory-only requests
retain the ordinary compiler path. A retained plan from a different compiler
is an input seed; it must be reconstructed or captured for the target identity
and target filenames before it becomes that compiler's production catalog.

## Build and stage

Record the actual selected forecast or `warm-kernels` requests in each pinned
compiler environment. Use a data-disk scratch directory for every output,
including `TMPDIR`, `CUPY_CACHE_DIR` and `GPUWM_KERNEL_PLAN_DIR`. A `{pid}`
filename template gives parallel workers separate catalogs. For example,
with `$scratch` and a valid `$PROFILE` already selected:

```sh
GPUWM_AOT_RECORD="$scratch/strict-{pid}.json" GPUWM_NUMERICS=strict \
  python -m gpuwm warm-kernels --profile "$PROFILE" --json
GPUWM_AOT_RECORD="$scratch/fast-{pid}.json" GPUWM_NUMERICS=fast \
  python -m gpuwm warm-kernels --profile "$PROFILE" --json
python tools/build_kernel_aot.py \
  --recipes "$scratch"/strict-*.json "$scratch"/fast-*.json \
  --output "$scratch/kernel-pack-cuda12.json"
```

Repeat with the CUDA 13 environment and a separate scratch directory to build
`kernel-pack-cuda13.json`. Catalogs are compiler-bound; a CUDA 13 catalog is
not a CUDA 12 build recipe. The CPU-only builder uses spawned NVRTC processes,
defaults to all available CPUs, and builds sm_89/90/100/120. `--workers` caps
build concurrency; `--base-pack` reuses matching verified images. Stage the
two completed family packs beside each other in package data for the release
build. Generated packs and catalogs do not belong in the source checkout.

## Warm plan and receipts

Supported cubin requests, including first-start static captures, compile in
CPU-only processes. GPU module loads stay in the parent thread pool. Every
process compiles the parent's final options and exact program name. The pool
honors `GPUWM_KERNEL_PLAN_WORKERS`. Rebinding a configuration drains its old
jobs before replacing the state used by their callbacks.
Recorded requests register the process cancellation proxy directly in the
single-flight table, leaving GPU warm-up threads available while CPU compilers
run. A foreground request cancels a queued job only when the process job itself
can be cancelled; otherwise it joins that job's image and symbol mapping.

Pack-covered recorded requests and static captures are skipped before scheduling. This is only a
scheduling hint: the actual loader still verifies the complete pack contract.
`kernel_plan.receipt()` reports `pool_skipped_aot`, `static_skipped_aot`, worker CPU seconds and
per-process work; `kernel_aot.receipt()` reports image hits and JIT misses.
These records describe startup and add no work to a warm simulation step.

Build coverage is the recorded request inventory. Before release, verify the
intended configuration loads its images without JIT misses, replay its strict
outputs, and capture a separate device timeline to check step launch counts.

## Automatic warm images and compiler metadata

The ordinary compile plan now retains successful cubins in its own
`images/<binding digest>/index.json`. The binding includes the full loaded
NVRTC library digest, CuPy package/cache identity, shape/options/numerics tag,
and target SM. Each request includes exact source, filename, ordered final
options and symbol aliases. Blobs and index rows carry digests. First use
uses the existing parallel compiler; background image writes retain its
actual results. Missing or corrupt entries warn and use that same compiler.
Concurrent publishers merge index entries under a process lock.

An unknown shape reads only its own plan filename. It does not enumerate
other shapes' plans. Grid dimensions and floating configuration values are
part of the tag. Composed Noah-MP and legacy radiation units use their
runtime factories during static warming.

CuPy's empty-source NVRTC preprocessing and the device-probe compiler banner
also have persistent metadata. These records contain measured output, not
inferred version strings. Exact options/target/compiler bytes bind empty
preprocessing; loaded compiler bytes bind the banner. The bare subprocess
loads stdlib-only helpers by path. If a standalone preparation package lacks
those helpers, its original measurement path remains available.

These caches are enabled with the existing compile-plan defaults. A cache
hit changes neither a kernel's source nor its arithmetic flags. Component
zero-NVRTC checks do not establish full CLI startup or forecast-word parity;
those require the corresponding forecast receipts.

A shape first encountered with an existing CuPy disk cache can have a partial
automatic image index: requests served before the NVRTC funnel produce no new
compiler result to retain. Removing that CuPy cache can therefore trigger
compilation of those missing requests. Their successful results populate the
index. The missing-image fallback must not be counted as a zero-NVRTC warm
run. Both caches contribute to ordinary warm startup.

External-header invalidation: automatic images bind literal headers and nested
literal includes by actual file digest. Macro-selected includes, conditional
header discovery and forced includes cannot be bound by this parser, so they
warn and use the ordinary compiler/cache route. The existing shipped-pack
format has no external-header binding and declines such requests. This is
artifact admission, not a numerical capability switch. Ordinary CuPy cache
invalidation for arbitrary changing external headers is not qualified here.
The measured AC warm load uses one conditional-header fallback but still
performs zero NVRTC work with its ordinary caches intact. The earlier AC
empty-CuPy-cache 59-image result predates this safety guard and is not a claim
for the final candidate's complete image-only coverage.
