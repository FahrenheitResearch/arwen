use crate::api::{blosc2_cbuffer_sizes, BLOSC2_CPARAMS_DEFAULTS};
use crate::internal;
use crate::internal::constants::*;

/// Error returned by the convenience compress/decompress functions.
#[derive(Debug)]
pub enum BloscError {
    /// The compressed buffer header is too short or malformed.
    InvalidHeader,
    /// Compression or decompression failed.
    Failed,
}

impl std::fmt::Display for BloscError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BloscError::InvalidHeader => write!(f, "invalid or truncated blosc header"),
            BloscError::Failed => write!(f, "blosc compression/decompression failed"),
        }
    }
}

impl std::error::Error for BloscError {}

/// Compresses `src` using Blosc1 format with default parameters.
///
/// Uses BloscLZ codec, compression level 5, typesize 8, and byte shuffle.
/// Returns the compressed bytes on success.
pub fn blosc1_compress(src: &[u8]) -> Result<Vec<u8>, BloscError> {
    let clevel = BLOSC2_CPARAMS_DEFAULTS.clevel as i32;
    let typesize = BLOSC2_CPARAMS_DEFAULTS.typesize as usize;
    let doshuffle = BLOSC_SHUFFLE as i32;
    let compressor = BLOSC_BLOSCLZ;

    let max_out = src.len() + BLOSC_MIN_HEADER_LENGTH;
    let mut dest = vec![0u8; max_out];

    internal::compress(clevel, doshuffle, typesize, src, &mut dest, compressor)
        .map(|n| {
            dest.truncate(n);
            dest
        })
        .map_err(|_| BloscError::Failed)
}

/// Decompresses a Blosc1 compressed buffer.
///
/// The uncompressed size is read from the header automatically.
/// Returns the decompressed bytes on success.
pub fn blosc1_decompress(src: &[u8]) -> Result<Vec<u8>, BloscError> {
    let (nbytes, _, _) = blosc2_cbuffer_sizes(src);
    if nbytes == 0 && src.len() < BLOSC_MIN_HEADER_LENGTH {
        return Err(BloscError::InvalidHeader);
    }

    let mut dest = vec![0u8; nbytes];
    internal::decompress(src, &mut dest)
        .map(|_| dest)
        .map_err(|_| BloscError::Failed)
}

/// Compresses `src` using Blosc2 format with default parameters.
///
/// Uses BloscLZ codec, compression level 5, typesize 8, and byte shuffle.
/// Returns the compressed bytes on success.
pub fn blosc2_compress(src: &[u8]) -> Result<Vec<u8>, BloscError> {
    let clevel = BLOSC2_CPARAMS_DEFAULTS.clevel as i32;
    let typesize = BLOSC2_CPARAMS_DEFAULTS.typesize as usize;
    let doshuffle = BLOSC_SHUFFLE as i32;
    let compressor = BLOSC_BLOSCLZ;

    let filters = BLOSC2_CPARAMS_DEFAULTS.filters;
    let filters_meta = BLOSC2_CPARAMS_DEFAULTS.filters_meta;

    let mut actual_filters = filters;
    if typesize > 1 {
        actual_filters[5] = BLOSC_SHUFFLE;
    }

    let max_out = src.len() + BLOSC2_MAX_OVERHEAD;
    let mut dest = vec![0u8; max_out];

    internal::compress_extended(
        clevel,
        doshuffle,
        typesize,
        src,
        &mut dest,
        compressor,
        &actual_filters,
        &filters_meta,
    )
    .map(|n| {
        dest.truncate(n);
        dest
    })
    .map_err(|_| BloscError::Failed)
}

/// Decompresses a Blosc2 compressed buffer.
///
/// The uncompressed size is read from the header automatically.
/// Returns the decompressed bytes on success.
pub fn blosc2_decompress(src: &[u8]) -> Result<Vec<u8>, BloscError> {
    let (nbytes, _, _) = blosc2_cbuffer_sizes(src);
    if nbytes == 0 && src.len() < BLOSC_MIN_HEADER_LENGTH {
        return Err(BloscError::InvalidHeader);
    }

    let mut dest = vec![0u8; nbytes];
    internal::decompress(src, &mut dest)
        .map(|_| dest)
        .map_err(|_| BloscError::Failed)
}
