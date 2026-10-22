# blusc

[![Latest Version](https://img.shields.io/crates/v/blusc.svg)](https://crates.io/crates/blusc)
![msrv](https://img.shields.io/crates/msrv/blusc)
[![blusc documentation](https://docs.rs/blusc/badge.svg)](https://docs.rs/blusc/latest/)


This is a pure Rust implementation of [c-blosc2](https://github.com/Blosc/c-blosc2) compression and decompression.


Blusc is not intended to be as performant as the reference C implementation, as the goal here is to enable easy compilation to WASM targets (so we avoid optimizations like multi-threading, hardware acceleration, etc). I have not performed any benchmarks.

## Background

The Blosc algorithm wraps other compression methods, including zstd, snappy, lz4, and zlib. We will use pure Rust implementations of these inner compression methods:

- ruzstd crate for zstd: https://github.com/KillingSpark/zstd-rs
- snap crate for snappy: https://github.com/BurntSushi/rust-snappy
- lz4_flex crate for lz4: https://github.com/pseitz/lz4_flex
- flate2 crate for zlib: https://github.com/rust-lang/flate2-rs

This crate offers pure rust implementations of the following functions and constants:

- blosc1_cbuffer_metainfo
- blosc1_cbuffer_validate
- blosc1_cbuffer_sizes
- blosc1_getitem
- blosc2_get_complib_info
- blosc2_compress
- blosc2_decompress
- BLOSC_NOSHUFFLE
- BLOSC2_MAX_OVERHEAD
- BLOSC2_CPARAMS_DEFAULTS
- BLOSC2_DPARAMS_DEFAULTS
- blosc2_create_cctx
- blosc2_compress_ctx
- blosc2_cbuffer_sizes
- blosc2_create_dctx
- blosc2_decompress_ctx



### Motivation

I tried and [failed](https://github.com/mulimoen/rust-blosc-src/compare/main...keller-mark:rust-blosc-src:keller-mark/shims) to compile existing Rust bindings to c-blosc to WASM (requiring [shims](https://github.com/mulimoen/rust-blosc-src/issues/23) for C standard library things), which motivated this pure Rust implementation.
Another aspect that complicates things is that c-blosc internally uses C libraries such as zstd, which can result in function name conflicts when used in codebases that depend on both c-blosc and zstd bindings:

```
some_crate
  - rust-bindings-to-c-blosc
    - c-blosc (C implementation)
      - zstd (C implementation)
  - rust-bindings-to-zstd
    - zstd (C implementation)
```

More broadly, it makes sense to have alternative implementations (especially in Rust) of widely-used compression algorithms.



## Development

```sh
cargo build
```

### Testing

```sh
cargo test
```



For reference during development, this repository contains the C implementations in the `c-blosc` and `c-blosc2` directories as git submodules.


## AI usage disclaimer

This README, `Cargo.toml`, the public interface/function signatures, initial unit tests, and the contents of `src/internal/constants.rs` were written by me, a human, or manually ported.
The rest of the Rust code and other unit tests were largely LLM-generated/ported (see `AGENTS.md` for more details).


## Citing Blosc

Copied from the [citing-blosc](https://github.com/Blosc/c-blosc2/tree/main?tab=readme-ov-file#citing-blosc) section of the c-blosc2 README.

```bibtex
@ONLINE{blosc,
    author = {{Blosc Development Team}},
    title = "{A fast, compressed and persistent data store library}",
    year = {2009-2025},
    note = {https://blosc.org}
}
```

## Related work

- [numcodecs.js](https://github.com/manzt/numcodecs.js/tree/main/codecs/blosc) compiles c-blosc to WebAssembly via Emscripten.
- https://github.com/maiteko/blosc2-src-rs
- https://github.com/mulimoen/rust-blosc-src
- https://github.com/asomers/blosc-rs
- https://github.com/barakugav/blosc-rs
- https://github.com/milesgranger/blosc2-rs
- https://github.com/bfffs/byteshuffle
