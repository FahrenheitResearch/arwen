use crate::internal;
use crate::internal::constants::*;
use std::os::raw::c_void;

/// Holds compression and decompression parameters for a Blosc2 operation.
///
/// Created via [`blosc2_create_cctx`] or [`blosc2_create_dctx`].
#[repr(C)]
pub struct Blosc2Context {
    /// Compression parameters.
    pub cparams: Blosc2Cparams,
    /// Decompression parameters.
    pub dparams: Blosc2Dparams,
}

/// Parameters controlling Blosc2 compression behavior.
///
/// Use [`BLOSC2_CPARAMS_DEFAULTS`] as a starting point and override individual fields.
#[repr(C)]
pub struct Blosc2Cparams {
    /// Compressor codec identifier (e.g. [`BLOSC_BLOSCLZ`], [`BLOSC_LZ4`], [`BLOSC_ZSTD`]).
    pub compcode: u8,
    /// Codec-specific metadata byte.
    pub compcode_meta: u8,
    /// Compression level, 0 (no compression) through 9 (maximum).
    pub clevel: u8,
    /// Whether to use a dictionary for compression.
    pub use_dict: i32,
    /// Size in bytes of the atomic data type (e.g. 4 for `f32`).
    pub typesize: i32,
    /// Number of threads for compression (currently only 1 is supported).
    pub nthreads: i16,
    /// Internal block size in bytes. 0 means automatic selection.
    pub blocksize: i32,
    /// Block split mode (e.g. [`BLOSC_FORWARD_COMPAT_SPLIT`], [`BLOSC_ALWAYS_SPLIT`]).
    pub splitmode: i32,
    /// Pointer to an associated super-chunk, if any.
    pub schunk: *mut c_void,
    /// Filter pipeline — up to [`BLOSC2_MAX_FILTERS`] filter codes applied before compression.
    pub filters: [u8; BLOSC2_MAX_FILTERS as usize],
    /// Per-filter metadata bytes corresponding to each entry in `filters`.
    pub filters_meta: [u8; BLOSC2_MAX_FILTERS as usize],
    /// Reserved: pre-filter callback pointer.
    pub prefilter: *mut c_void,
    /// Reserved: pre-filter parameters pointer.
    pub preparams: *mut c_void,
    /// Reserved: tuner parameters pointer.
    pub tuner_params: *mut c_void,
    /// Tuner identifier.
    pub tuner_id: i32,
    /// Whether to instrument the codec (for development/debugging).
    pub instr_codec: bool,
    /// Reserved: codec-specific parameters pointer.
    pub codec_params: *mut c_void,
    /// Reserved: per-filter parameters pointers.
    pub filter_params: [*mut c_void; BLOSC2_MAX_FILTERS as usize],
}

/// Parameters controlling Blosc2 decompression behavior.
///
/// Use [`BLOSC2_DPARAMS_DEFAULTS`] as a starting point and override individual fields.
#[repr(C)]
pub struct Blosc2Dparams {
    /// Number of threads for decompression (currently only 1 is supported).
    pub nthreads: i16,
    /// Pointer to an associated super-chunk, if any.
    pub schunk: *mut c_void,
    /// Reserved: post-filter callback pointer.
    pub postfilter: *mut c_void,
    /// Reserved: post-filter parameters pointer.
    pub postparams: *mut c_void,
}

/// Default compression parameters: BloscLZ codec, compression level 5, typesize 8, byte shuffle.
pub const BLOSC2_CPARAMS_DEFAULTS: Blosc2Cparams = Blosc2Cparams {
    compcode: BLOSC_BLOSCLZ,
    compcode_meta: 0,
    clevel: 5,
    use_dict: 0,
    typesize: 8,
    nthreads: 1,
    blocksize: 0,
    splitmode: BLOSC_FORWARD_COMPAT_SPLIT as i32,
    schunk: std::ptr::null_mut(),
    filters: [BLOSC_NOFILTER; BLOSC2_MAX_FILTERS as usize],
    filters_meta: [0; BLOSC2_MAX_FILTERS as usize],
    prefilter: std::ptr::null_mut(),
    preparams: std::ptr::null_mut(),
    tuner_params: std::ptr::null_mut(),
    tuner_id: 0,
    instr_codec: false,
    codec_params: std::ptr::null_mut(),
    filter_params: [std::ptr::null_mut(); BLOSC2_MAX_FILTERS as usize],
};

/// Default decompression parameters: single-threaded, no postfilter.
pub const BLOSC2_DPARAMS_DEFAULTS: Blosc2Dparams = Blosc2Dparams {
    nthreads: 1,
    schunk: std::ptr::null_mut(),
    postfilter: std::ptr::null_mut(),
    postparams: std::ptr::null_mut(),
};

/// Extracts the typesize and flags byte from a Blosc1 compressed buffer's header.
///
/// Returns `Some((typesize, flags))` on success, or `None` if the buffer is too short.
pub fn blosc1_cbuffer_metainfo(cbuffer: &[u8]) -> Option<(usize, i32)> {
    if cbuffer.len() < BLOSC_MIN_HEADER_LENGTH {
        return None;
    }
    let ts = cbuffer[3] as usize;
    let fl = cbuffer[2] as i32;

    Some((ts, fl))
}

/// Validates a Blosc1 compressed buffer by checking that the declared compressed
/// size in the header matches `cbytes`.
///
/// Returns the uncompressed size on success.
pub fn blosc1_cbuffer_validate(cbuffer: &[u8], cbytes: usize) -> Result<usize, ()> {
    if cbuffer.len() < BLOSC_MIN_HEADER_LENGTH {
        return Err(());
    }

    let cb = u32::from_le_bytes([cbuffer[12], cbuffer[13], cbuffer[14], cbuffer[15]]) as usize;
    if cbytes != cb {
        return Err(());
    }

    let nb = u32::from_le_bytes([cbuffer[4], cbuffer[5], cbuffer[6], cbuffer[7]]) as usize;
    Ok(nb)
}

/// Returns the `(uncompressed_size, compressed_size, block_size)` stored in a
/// Blosc1 compressed buffer's header. Equivalent to [`blosc2_cbuffer_sizes`].
pub fn blosc1_cbuffer_sizes(cbuffer: &[u8]) -> (usize, usize, usize) {
    blosc2_cbuffer_sizes(cbuffer)
}

/// Extracts a slice of items from a Blosc1 compressed buffer without fully decompressing it.
///
/// Decompresses `nitems` elements starting at item index `start` into `dest`.
/// Returns the number of bytes written, or 0 on error.
pub fn blosc1_getitem(cbuffer: &[u8], start: i32, nitems: i32, dest: &mut [u8]) -> i32 {
    let cbytes = u32::from_le_bytes([cbuffer[12], cbuffer[13], cbuffer[14], cbuffer[15]]) as usize;

    let src_slice = if cbuffer.len() >= cbytes {
        &cbuffer[..cbytes]
    } else {
        cbuffer
    };

    match internal::getitem(src_slice, start as usize, nitems as usize, dest) {
        Ok(size) => size as i32,
        Err(_) => 0,
    }
}

/// Compresses `src` into `dest` using Blosc1 format (16-byte header, version 2).
///
/// This produces output compatible with the C blosc v1 library (`blosc_compress`).
///
/// - `clevel`: compression level (0–9).
/// - `doshuffle`: shuffle mode ([`BLOSC_NOSHUFFLE`], [`BLOSC_SHUFFLE`], or [`BLOSC_BITSHUFFLE`]).
/// - `typesize`: element size in bytes (used by shuffle filters).
///
/// `dest` must be at least `src.len() + BLOSC_MIN_HEADER_LENGTH` bytes.
/// Returns the number of compressed bytes written, or 0 on error.
pub fn blosc1_compress(
    clevel: i32,
    doshuffle: i32,
    typesize: usize,
    src: &[u8],
    dest: &mut [u8],
) -> i32 {
    let compressor = BLOSC_BLOSCLZ;

    match internal::compress(clevel, doshuffle, typesize, src, dest, compressor) {
        Ok(size) => size as i32,
        Err(_) => 0,
    }
}

/// Compresses `src` into `dest` using Blosc1 format with the codec specified in `context`.
///
/// This produces output compatible with the C blosc v1 library (`blosc_compress_ctx`).
///
/// Returns the number of compressed bytes written, or 0 on error.
pub fn blosc1_compress_ctx(context: &Blosc2Context, src: &[u8], dest: &mut [u8]) -> i32 {
    let clevel = context.cparams.clevel as i32;
    let typesize = context.cparams.typesize as usize;
    let compressor = context.cparams.compcode;

    let mut doshuffle = BLOSC_NOSHUFFLE as i32;
    for &f in context.cparams.filters.iter() {
        if f == BLOSC_SHUFFLE {
            doshuffle = BLOSC_SHUFFLE as i32;
        }
        if f == BLOSC_BITSHUFFLE {
            doshuffle = BLOSC_BITSHUFFLE as i32;
        }
    }

    match internal::compress(clevel, doshuffle, typesize, src, dest, compressor) {
        Ok(size) => size as i32,
        Err(_) => 0,
    }
}

/// Decompresses a Blosc1 or Blosc2 compressed buffer.
///
/// This is functionally identical to [`blosc2_decompress`] — both formats are
/// auto-detected from the header version byte.
///
/// `dest` must be large enough to hold the uncompressed data (see [`blosc1_cbuffer_sizes`]).
/// Returns the number of decompressed bytes, or -1 on error.
pub fn blosc1_decompress(src: &[u8], dest: &mut [u8]) -> i32 {
    match internal::decompress(src, dest) {
        Ok(size) => size as i32,
        Err(_) => -1,
    }
}

/// Looks up information about a compression library by its name (e.g. `"blosclz"`, `"lz4"`).
///
/// Returns `Some((library_name, version_string, library_code))` or `None` if unknown.
pub fn blosc2_get_complib_info(compcode: &str) -> Option<(&'static str, &'static str, i32)> {
    // TODO: use constants
    match compcode {
        "blosclz" => Some(("BloscLZ", "2.5.1", 0)),
        "lz4" => Some(("LZ4", "1.9.4", 1)),
        "lz4hc" => Some(("LZ4", "1.9.4", 1)),
        "snappy" => Some(("Snappy", "1.1.9", 2)),
        "zlib" => Some(("Zlib", "1.2.11", 3)),
        "zstd" => Some(("Zstd", "1.5.2", 4)),
        _ => None,
    }
}

/// Compresses `src` into `dest` using the default BloscLZ codec.
///
/// - `clevel`: compression level (0–9).
/// - `doshuffle`: shuffle mode ([`BLOSC_NOSHUFFLE`], [`BLOSC_SHUFFLE`], or [`BLOSC_BITSHUFFLE`]).
/// - `typesize`: element size in bytes (used by shuffle filters).
///
/// `dest` must be at least `src.len() + BLOSC2_MAX_OVERHEAD` bytes.
/// Returns the number of compressed bytes written, or 0 on error.
pub fn blosc2_compress(
    clevel: i32,
    doshuffle: i32,
    typesize: usize,
    src: &[u8],
    dest: &mut [u8],
) -> i32 {
    // Default compressor: BLOSCLZ (0)
    let compressor = BLOSC_BLOSCLZ;

    let mut filters = [0u8; BLOSC2_MAX_FILTERS as usize];
    let filters_meta = [0u8; BLOSC2_MAX_FILTERS as usize];

    if doshuffle == BLOSC_SHUFFLE as i32 && typesize > 1 {
        filters[5] = BLOSC_SHUFFLE;
    } else if doshuffle == BLOSC_BITSHUFFLE as i32 {
        filters[5] = BLOSC_BITSHUFFLE;
    }

    match internal::compress_extended(
        clevel,
        doshuffle,
        typesize,
        src,
        dest,
        compressor,
        &filters,
        &filters_meta,
    ) {
        Ok(size) => size as i32,
        Err(_) => 0,
    }
}

/// Decompresses a Blosc2 compressed buffer.
///
/// `dest` must be large enough to hold the uncompressed data (see [`blosc2_cbuffer_sizes`]).
/// Returns the number of decompressed bytes, or -1 on error.
pub fn blosc2_decompress(src: &[u8], dest: &mut [u8]) -> i32 {
    match internal::decompress(src, dest) {
        Ok(size) => size as i32,
        Err(_) => -1,
    }
}

/// Creates a compression context from the given parameters.
///
/// Use this when you need to specify a non-default codec or custom filter pipeline.
pub fn blosc2_create_cctx(cparams: Blosc2Cparams) -> Blosc2Context {
    Blosc2Context {
        cparams,
        dparams: BLOSC2_DPARAMS_DEFAULTS,
    }
}

/// Compresses `src` into `dest` using the codec and filters specified in `context`.
///
/// Returns the number of compressed bytes written, or 0 on error.
pub fn blosc2_compress_ctx(context: &Blosc2Context, src: &[u8], dest: &mut [u8]) -> i32 {
    let clevel = context.cparams.clevel as i32;
    let typesize = context.cparams.typesize as usize;
    let compressor = context.cparams.compcode;

    let mut doshuffle = BLOSC_NOSHUFFLE as i32;
    for &f in context.cparams.filters.iter() {
        if f == BLOSC_SHUFFLE {
            doshuffle = BLOSC_SHUFFLE as i32;
        }
        if f == BLOSC_BITSHUFFLE {
            doshuffle = BLOSC_BITSHUFFLE as i32;
        }
    }

    // Filters are already u8 array
    let filters = context.cparams.filters;
    let filters_meta = context.cparams.filters_meta;

    match internal::compress_extended(
        clevel,
        doshuffle,
        typesize,
        src,
        dest,
        compressor,
        &filters,
        &filters_meta,
    ) {
        Ok(size) => size as i32,
        Err(_) => 0,
    }
}

/// Returns the `(uncompressed_size, compressed_size, block_size)` stored in a
/// Blosc2 compressed buffer's header.
///
/// Returns `(0, 0, 0)` if the buffer is too short to contain a valid header.
pub fn blosc2_cbuffer_sizes(cbuffer: &[u8]) -> (usize, usize, usize) {
    if cbuffer.len() < BLOSC_MIN_HEADER_LENGTH {
        return (0, 0, 0);
    }
    let nb = u32::from_le_bytes([cbuffer[4], cbuffer[5], cbuffer[6], cbuffer[7]]) as usize;
    let bs = u32::from_le_bytes([cbuffer[8], cbuffer[9], cbuffer[10], cbuffer[11]]) as usize;
    let cb = u32::from_le_bytes([cbuffer[12], cbuffer[13], cbuffer[14], cbuffer[15]]) as usize;

    (nb, cb, bs)
}

/// Creates a decompression context from the given parameters.
pub fn blosc2_create_dctx(dparams: Blosc2Dparams) -> Blosc2Context {
    Blosc2Context {
        cparams: BLOSC2_CPARAMS_DEFAULTS,
        dparams,
    }
}

/// Decompresses a Blosc2 compressed buffer using the given context.
///
/// Returns the number of decompressed bytes, or -1 on error.
pub fn blosc2_decompress_ctx(_context: &Blosc2Context, src: &[u8], dest: &mut [u8]) -> i32 {
    match internal::decompress(src, dest) {
        Ok(size) => size as i32,
        Err(_) => -1,
    }
}
