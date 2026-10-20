# t-route MC native replay

Source licence read before implementation: **“Apache License”, “Version 2.0,
January 2004”**, NOAA-OWP/t-route at
`12a8eae0cdfed437143c590659fa7077605a5e70`.
The exact upstream LICENSE and TERMS accompany every sealed fixture. Only
the fresh Apache source and the existing integration-base IEEE CUDA compiler
are used. No withdrawn Hydro code, ABI or fixture is reused.

This dependency-free Rust tool captures the **actual compiled**
`c_muskingcungenwm` calls in the shipped HYFeatures executor. Native Fortran
is a reference tool only, built in a scratch directory; it never enters a product
installation or wheel. Build/run recipes are in
`tools/native_nextgen_reference/troute_mc/`.

## Observe without perturbing native scratch

The external x86_64 Linux `trace` command launches its own traced child. It
finds the reference extension's executable mapping, temporarily places
debugger breakpoints at native entry/return, copies arguments through
`/proc/<child>/mem`, restores instructions and continues the original process.
It adds no target stack frame and edits no source or on-disk native executable.
Only the child is traced; the file must be new; the explicit call limit bounds
capture size. A failed observation terminates that child.

Each raw call record has 27 little-endian float32 words: 15 inputs, six caller
initialization words, six returned outputs. The input order is
`dt,qup,quc,qdp,ql,dx,bw,tw,twcc,n,ncc,cs,s0,velp,depthp`;
outputs are `qdc,velc,depthc,ck,cn,X`. The native Cython caller initializes
all six outputs to positive zero; `split` rejects any other initialization.
All six returned words, including NaN payloads and diagnostics, are preserved
and compared without exclusions or tolerances.

The discarded LD_PRELOAD experiment changed native outputs by adding a target
frame. Do not restore that method: the pinned native secant routine reads
uninitialized `Qj` before setting it. Whole-case byte equivalence between clean
and externally observed native results is required by `seal_fixture.py`.

An optional bounded secant witness uses `MC_TRACE_SECANT_OFFSET`,
`MC_TRACE_WITNESS_FIRST`, `MC_TRACE_WITNESS_CALLS`, `MC_TRACE_WITNESS_LIMIT`,
`MC_TRACE_WITNESS_PATH`. It records native secant arguments without a target
frame. Its argument order is
`z,bw,bfd,twcc,s0,n,ncc,dt,dx,qdp,ql,qup,quc,h,interval,Qj,C1,C2,C3,C4,X`;
`interval` is int32, all others float32. This is diagnostic evidence, never
permission to insert captured stack values in production or waive differences.

`MC_TRACE_POWF_OFFSET` instead names the extension's `powf@plt` offset;
`MC_TRACE_POWF_CALL`, `MC_TRACE_POWF_LIMIT`, and `MC_TRACE_POWF_PATH` bound one
call's math observation. The tool records raw XMM0/XMM1 inputs and XMM0 return,
the native return-site offset, resolved implementation address, and process
maps. It never changes floating-point registers. The native case runner exposes
this as `--powf-call ROW` and seals mapped libm/libc files. Whole-case success
and byte equivalence remain necessary; a later PyArrow teardown crash makes
that extended case witness unqualified.

For the exact isolated issue, `tools/native_nextgen_reference/troute_mc/row.rs`
is a private Rust reference driver: it reads one actual 15-word input row,
seeds the six outputs as the native Cython caller does, calls the bind(C)
routine and writes the six raw words. Its optional expected-array argument
extracts that captured row's six expected words without floating-point math.
Build a tiny native shared tool from the **same patched** `mc_single_seg.o`
and `pymc_single_seg.o` with `gfortran -shared -o LIBMC ...`; compile the driver
with `rustc --edition 2021 -O row.rs -o DRIVER`. Both outputs and `TMPDIR`
must be under a scratch directory. Run `DRIVER LIBMC INPUTS ROW NEW_OUTPUT [EXPECTED]`.
Compare all six words against the actual case and against an externally
observed invocation. With this short driver set `MC_TRACE_STARTUP_STRIDE=1`
so the observer installs its entry breakpoint at the first mapped syscall.

The qualified row-68066 witness matched the case and the clean driver in all
six words. Native `powf(0x3d99d83a, 0x3f000000)` returned `0x3e8c541f`;
`sqrtf` rounded one word lower, changing the computed radius and final
velocity. Preserving the native power operation fixes the port; no ULP waiver
is needed. The complete strict replay then matches 484,056/484,056 words.

The reference recipe applies the pinned `define-secant-seed.patch` by default.
It defines `qdc`, `X`, and first-secant `Qj_0` as zero, and declares the secant
`Qj/C1..C4` read/write arguments `intent(inout)`. The original source reads
`Qj_0` before initialization. Actual call 6724 entered with `0x3f606d79`;
substituting only that measured seed in a diagnostic GPU call reproduced all
six native words. The original full replay failed 57,601/484,056 words and is
retained. This is an explicit native-tool correctness patch, never an
untouched-upstream claim or a tolerance. Build receipts seal original and
patched source, patch, compiler and flags; execution receipts seal the binary.
The recipe forces extension relinking because setuptools does not track
changes in the linked Fortran archives. Native source/binaries stay private
reference tools and are never part of a wheel.

The native graph manifest also records `assume_short_ts` from the bound actual
`compute_nhd_routing_v02` call. Consumers must preserve that boolean when
preparing the resident router. The shipped case enables it: upstream flow at
both endpoints comes from the previous period. Omitting the flag and using
the factory's false default caused 61,033/242,712 network Q/V/D words to differ,
despite the exact leaf replay. At actual call 2, reach 2420802 receives native
`quc=0`; the false-mode network instead supplies its upstream reach's current
`0.037910074`. After binding the captured true mode, the same 12-period network
matches all 242,712 Q/V/D words, including all 19 reservoirs. The graph-04
schema correction retained every graph and raw-column hash from graph-03;
its complete native output is byte-identical. Earlier captures remain sealed.

## Build and run

Use a scratch directory on the data disk for every output/cache/target.
Set `TMPDIR`, `CARGO_HOME` and `CARGO_TARGET_DIR` there before:

```
cargo build --offline --release --manifest-path tools/troute_mc_replay/Cargo.toml
```

Prepare a private Python3.11 reference environment with the dependency versions
in its receipt and extract the pinned upstream archive under that scratch.
Then run the native build and a clean/observed pair (paths are explicit):

```
python tools/native_nextgen_reference/troute_mc/build.py SOURCE SCRATCH --workers 24
python tools/native_nextgen_reference/troute_mc/run.py SOURCE CLEAN --build-receipt BUILD_JSON
python tools/native_nextgen_reference/troute_mc/run.py SOURCE OBSERVED --build-receipt BUILD_JSON --trace-tool TOOL --limit 100000
python tools/troute_mc_replay/seal_fixture.py OBSERVED CLEAN/flowveldepth.pkl FIXTURE --tool TOOL
```

The run uses the shipped `test/LowerColorado_TX_v4` no-DA realization, shortened
to 12×300 seconds with native serial scheduling for observation. Native
`return_courant=True` has an unrelated upstream writer NameError; retain its
shipped false setting. The leaf still computes/captures all six diagnostics.
The reference-only serial schedule is never a product execution path.

Under the canonical GPU pool, with `GPUWM_NUMERICS=strict` and scratch caches:

```
python tools/troute_mc_replay/replay.py FIXTURE/fixture.json NEW_PROOF --library RAW_SO --tool TOOL --port-module PORT/gpuwm/hydro/troute_mc.py
```

Python validates metadata, allocates CUDA/pinned buffers and launches kernels.
Rust reads/writes raw pinned bytes, packs captures and compares words across
available CPU workers. CUDA performs all model arithmetic. The replay records
the actual compiler options/source hash, card, block/grid, all compared columns
and CUDA event timing. Independent-call batch timing is not a routing-step
throughput measurement. The production network and short standalone/coupled
run are separate acceptance requirements. No CUDA graphs are used.

## Native routing graph observation

`tools/native_nextgen_reference/troute_mc/graph.rs` is a separate private
Rust cdylib using the running reference interpreter's CPython API. Build with
`rustc --edition 2021 --crate-type cdylib -O graph.rs -o GRAPH_SO` under
scratch. Pass `--graph-library GRAPH_SO --raw-library RAW_SO` to the native
case runner. Its opt-in `sitecustomize.py` passes existing native object and
array handles; Rust traverses graph dictionaries/sequences, preserves every
native upstream-list order, packs strided raw columns across available CPU
workers, and writes the captures. The original native constructor and compute
execute unchanged. No Python graph packing or model arithmetic is added.

The manifest seals native connections, reverse network, reservoir membership,
lake-outlet replacement crosswalk, reaches and independent networks, both
after construction and at the actual compute boundary. It also seals each
parameter/reservoir/initial-state/lateral-forcing column and its row IDs, dtype,
shape and source/output strides. The entire clean-vs-observed native result
must again compare byte-identically with successful process exit. The native
6971-row parameter frame is not by itself the MC-only collapsed roster.
