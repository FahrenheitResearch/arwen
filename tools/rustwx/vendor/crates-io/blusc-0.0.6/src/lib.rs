//! Pure Rust implementation of the [Blosc/Blosc2](https://www.blosc.org/) compression library.
//!
//! Blosc is a high-performance, chunked compression library designed for binary data.
//! It splits input into blocks, applies filters (shuffle, bitshuffle), and then compresses
//! each block with a codec such as BloscLZ, LZ4, Zstd, Zlib, or Snappy.
//!
//! # Quick start
//!
//! ```rust
//! use blusc::{blosc2_compress, blosc2_decompress, BLOSC_SHUFFLE, BLOSC2_MAX_OVERHEAD};
//!
//! let input: Vec<u8> = vec![0u8; 1024];
//! let mut compressed = vec![0u8; input.len() + BLOSC2_MAX_OVERHEAD];
//! let cbytes = blosc2_compress(5, BLOSC_SHUFFLE as i32, 4, &input, &mut compressed);
//!
//! let mut output = vec![0u8; 1024];
//! let nbytes = blosc2_decompress(&compressed[..cbytes as usize], &mut output);
//! assert_eq!(input, output);
//! ```

/// High-level compress/decompress API, context management, and header introspection.
pub mod api;
/// Idiomatic Rust convenience wrappers: bytes-in, `Result<Vec<u8>>` out.
pub mod convenience;
/// Compression codec implementations (BloscLZ, etc.).
pub mod codecs;
/// Pre-compression filter implementations (byte shuffle, bitshuffle).
pub mod filters;
/// Low-level compression/decompression internals and protocol constants.
pub mod internal;

pub use crate::internal::constants::*;
pub use api::*;
