//! Protocol constants ported from `c-blosc2/include/blosc2.h`.

// Version numbers

/// Blosc2 major version — incremented for major interface/format changes.
pub const BLOSC2_VERSION_MAJOR: u8 = 2;
/// Blosc2 minor version — incremented for minor interface/format changes.
pub const BLOSC2_VERSION_MINOR: u8 = 22;
/// Blosc2 release tag.
pub const BLOSC2_VERSION_RELEASE: &str = "1.dev";
/// Full Blosc2 version string.
pub const BLOSC2_VERSION_STRING: &str = "2.22.1.dev";
/// Blosc2 version date.
pub const BLOSC2_VERSION_DATE: &str = "$Date:: 2025-10-28 #$";

/// Maximum number of dimensions for Blosc2 NDim arrays.
pub const BLOSC2_MAX_DIM: u8 = 8;

// Chunk format version identifiers (1 byte each).
//
// History:
// - 1: Blosc pre-1.0
// - 2: Blosc 1.x stable
// - 3: Blosc 2-alpha.x
// - 4: Blosc 2.x beta.1
// - 5: Blosc 2.x stable

/// Chunk format version for Blosc pre-1.0.
pub const BLOSC1_VERSION_FORMAT_PRE1: u8 = 1;
/// Chunk format version for Blosc 1.x stable.
pub const BLOSC1_VERSION_FORMAT: u8 = 2;
/// Chunk format version for Blosc 2-alpha.
pub const BLOSC2_VERSION_FORMAT_ALPHA: u8 = 3;
/// Chunk format version for Blosc 2 beta.1.
pub const BLOSC2_VERSION_FORMAT_BETA1: u8 = 4;
/// Chunk format version for Blosc 2 stable.
pub const BLOSC2_VERSION_FORMAT_STABLE: u8 = 5;
/// Current chunk format version.
pub const BLOSC2_VERSION_FORMAT: u8 = BLOSC2_VERSION_FORMAT_STABLE;

// Frame format version identifiers (4-bit).

/// Frame format version introduced in 2.0.0-beta2.
pub const BLOSC2_VERSION_FRAME_FORMAT_BETA2: u8 = 1;
/// Frame format version introduced in 2.0.0-rc1.
pub const BLOSC2_VERSION_FRAME_FORMAT_RC1: u8 = 2;
/// Current frame format version.
pub const BLOSC2_VERSION_FRAME_FORMAT: u8 = BLOSC2_VERSION_FRAME_FORMAT_RC1;

/// Minimum header length in bytes (Blosc1 header).
pub const BLOSC_MIN_HEADER_LENGTH: usize = 16;

/// Extended header length in bytes (Blosc2 header).
pub const BLOSC_EXTENDED_HEADER_LENGTH: usize = 32;

/// Maximum compression overhead in bytes added by the Blosc2 framing.
///
/// Allocate at least `src.len() + BLOSC2_MAX_OVERHEAD` for the destination buffer.
pub const BLOSC2_MAX_OVERHEAD: usize = BLOSC_EXTENDED_HEADER_LENGTH;

/// Maximum value of a 32-bit signed integer.
pub const INT_MAX: usize = 2147483647;
/// Maximum value of an unsigned 8-bit integer.
pub const UINT8_MAX: usize = 255;

/// Maximum source buffer size (in bytes) that can be compressed.
pub const BLOSC2_MAX_BUFFERSIZE: usize = INT_MAX - BLOSC2_MAX_OVERHEAD;

/// Maximum typesize in bytes before the source is treated as a raw byte stream.
pub const BLOSC_MAX_TYPESIZE: usize = UINT8_MAX;

/// Minimum buffer size in bytes that Blosc will attempt to compress.
pub const BLOSC_MIN_BUFFERSIZE: usize = 32;

/// Typical L1 cache size used for block-size heuristics.
pub const L1: usize = 32 * 1024;
/// Typical L2 cache size used for block-size heuristics.
pub const L2: usize = 256 * 1024;

/// Maximum block size in bytes.
pub const BLOSC_MAX_BLOCKSIZE: usize = BLOSC2_MAX_BUFFERSIZE;

// Not used here, but exported for compatibility reasons.
// Reference: https://github.com/mulimoen/rust-blosc-src/blob/8fb7864e2b2ce03dff400e444bf34005c0447357/src/bindgen.rs#L20
pub const BLOSC_MAX_THREADS: u32 = 256;

// Tuner ID ranges

/// Start of Blosc-defined tuner IDs.
pub const BLOSC2_DEFINED_TUNER_START: u8 = 0;
/// End of Blosc-defined tuner IDs.
pub const BLOSC2_DEFINED_TUNER_STOP: u8 = 31;

/// Start of globally-registered tuner IDs.
pub const BLOSC2_GLOBAL_REGISTERED_TUNER_START: u8 = 32;
/// End of globally-registered tuner IDs.
pub const BLOSC2_GLOBAL_REGISTERED_TUNER_STOP: u8 = 159;

/// Number of globally-registered tuners.
pub const BLOSC2_GLOBAL_REGISTERED_TUNERS: u8 = 0;

/// Start of user-defined tuner IDs.
pub const BLOSC2_USER_REGISTERED_TUNER_START: u8 = 160;
/// End of user-defined tuner IDs.
pub const BLOSC2_USER_REGISTERED_TUNER_STOP: u8 = 255;

// Built-in tuner codes

/// Blosc built-in simple tuner (stune).
pub const BLOSC_STUNE: u8 = 0;
/// Sentinel: one past the last built-in tuner.
pub const BLOSC_LAST_TUNER: u8 = 1;
/// Last globally-registered tuner ID.
pub const BLOSC_LAST_REGISTERED_TUNER: u8 =
    BLOSC2_GLOBAL_REGISTERED_TUNER_START + BLOSC2_GLOBAL_REGISTERED_TUNERS - 1;

// Filter ID ranges

/// Start of Blosc-defined filter IDs.
pub const BLOSC2_DEFINED_FILTERS_START: u8 = 0;
/// End of Blosc-defined filter IDs.
pub const BLOSC2_DEFINED_FILTERS_STOP: u8 = 31;

/// Start of globally-registered filter IDs.
pub const BLOSC2_GLOBAL_REGISTERED_FILTERS_START: u8 = 32;
/// End of globally-registered filter IDs.
pub const BLOSC2_GLOBAL_REGISTERED_FILTERS_STOP: u8 = 159;

/// Number of globally-registered filters.
pub const BLOSC2_GLOBAL_REGISTERED_FILTERS: u8 = 5;

/// Start of user-defined filter IDs.
pub const BLOSC2_USER_REGISTERED_FILTERS_START: u8 = 160;
/// End of user-defined filter IDs.
pub const BLOSC2_USER_REGISTERED_FILTERS_STOP: u8 = 255;

/// Maximum number of filters in a single filter pipeline.
pub const BLOSC2_MAX_FILTERS: u8 = 6;

/// Maximum number of user-defined filters that can be registered.
pub const BLOSC2_MAX_UDFILTERS: u8 = 16;

// Filter codes — used in the `filters` array of [`Blosc2Cparams`].

/// No shuffle (Blosc1 compatibility alias for [`BLOSC_NOFILTER`]).
pub const BLOSC_NOSHUFFLE: u8 = 0;

/// No filter applied.
pub const BLOSC_NOFILTER: u8 = 0;

/// Byte-wise shuffle filter. Rearranges bytes by significance across elements
/// to improve codec compression ratios.
pub const BLOSC_SHUFFLE: u8 = 1;

/// Bit-wise shuffle filter. Rearranges individual bits for maximum
/// compression of data with low bit-entropy.
pub const BLOSC_BITSHUFFLE: u8 = 2;

/// Delta coding filter.
pub const BLOSC_DELTA: u8 = 3;

/// Truncate mantissa precision filter. Positive `filters_meta` keeps that
/// many bits; negative values zero that many bits.
pub const BLOSC_TRUNC_PREC: u8 = 4;

/// Sentinel: one past the last built-in filter.
pub const BLOSC_LAST_FILTER: u8 = 5;

/// Last globally-registered filter ID.
pub const BLOSC_LAST_REGISTERED_FILTER: u8 =
    BLOSC2_GLOBAL_REGISTERED_FILTERS_START + BLOSC2_GLOBAL_REGISTERED_FILTERS - 1;

// Internal header flags (stored in the flags byte of the chunk header).

/// Header flag: byte-wise shuffle was applied.
pub const BLOSC_DOSHUFFLE: u8 = 0x1;
/// Header flag: data was memcpy'd (not compressed).
pub const BLOSC_MEMCPYED: u8 = 0x2;
/// Header flag: bit-wise shuffle was applied.
pub const BLOSC_DOBITSHUFFLE: u8 = 0x4;
/// Header flag: delta coding was applied.
pub const BLOSC_DODELTA: u8 = 0x8;

// Blosc2-specific flags (stored in the extended header).

/// Blosc2 flag: dictionary-aided compression was used.
pub const BLOSC2_USEDICT: u8 = 0x1;
/// Blosc2 flag: data is stored in big-endian byte order.
pub const BLOSC2_BIGENDIAN: u8 = 0x2;
/// Blosc2 flag: codec instrumentation is enabled (development use).
pub const BLOSC2_INSTR_CODEC: u8 = 0x80;

// Blosc2 capability limits

/// Maximum size in bytes for a compression dictionary.
pub const BLOSC2_MAXDICTSIZE: u32 = 128 * 1024;
/// Maximum block size in bytes (absolute limit).
pub const BLOSC2_MAXBLOCKSIZE: u32 = 536866816;
/// Maximum typesize in bytes (absolute limit).
pub const BLOSC2_MAXTYPESIZE: u32 = BLOSC2_MAXBLOCKSIZE;

// Codec ID ranges

/// Start of Blosc-defined codec IDs.
pub const BLOSC2_DEFINED_CODECS_START: u8 = 0;
/// End of Blosc-defined codec IDs.
pub const BLOSC2_DEFINED_CODECS_STOP: u8 = 31;

/// Start of globally-registered codec IDs.
pub const BLOSC2_GLOBAL_REGISTERED_CODECS_START: u8 = 32;
/// End of globally-registered codec IDs.
pub const BLOSC2_GLOBAL_REGISTERED_CODECS_STOP: u8 = 159;

/// Number of globally-registered codecs.
pub const BLOSC2_GLOBAL_REGISTERED_CODECS: u8 = 5;

/// Start of user-defined codec IDs.
pub const BLOSC2_USER_REGISTERED_CODECS_START: u8 = 160;
/// End of user-defined codec IDs.
pub const BLOSC2_USER_REGISTERED_CODECS_STOP: u8 = 255;

// Compressor codec identifiers — pass these to [`Blosc2Cparams::compcode`].

/// BloscLZ codec (default). A fast, LZ77-based compressor tuned for Blosc.
pub const BLOSC_BLOSCLZ: u8 = 0;
/// LZ4 codec. Very fast compression/decompression.
pub const BLOSC_LZ4: u8 = 1;
/// LZ4HC codec. Higher compression ratio variant of LZ4 (slower compression, same decompression speed).
pub const BLOSC_LZ4HC: u8 = 2;
/// Snappy codec.
pub const BLOSC_SNAPPY: u8 = 3;
/// Zlib (deflate) codec. Good compression ratio, moderate speed.
pub const BLOSC_ZLIB: u8 = 4;
/// Zstandard codec. Excellent compression ratio with fast decompression.
pub const BLOSC_ZSTD: u8 = 5;

/// Sentinel: one past the last built-in codec.
pub const BLOSC_LAST_CODEC: u8 = 6;

/// Last globally-registered codec ID.
pub const BLOSC_LAST_REGISTERED_CODEC: u8 =
    BLOSC2_GLOBAL_REGISTERED_CODECS_START + BLOSC2_GLOBAL_REGISTERED_CODECS - 1;

// Compressor name strings

/// Name string for the BloscLZ compressor.
pub const BLOSC_BLOSCLZ_COMPNAME: &str = "blosclz";
/// Name string for the LZ4 compressor.
pub const BLOSC_LZ4_COMPNAME: &str = "lz4";
/// Name string for the LZ4HC compressor.
pub const BLOSC_LZ4HC_COMPNAME: &str = "lz4hc";
/// Name string for the Snappy compressor.
pub const BLOSC_SNAPPY_COMPNAME: &str = "snappy";
/// Name string for the Zlib compressor.
pub const BLOSC_ZLIB_COMPNAME: &str = "zlib";
/// Name string for the Zstd compressor.
pub const BLOSC_ZSTD_COMPNAME: &str = "zstd";

// Compression library codes (stored in header flags bits 5–7, must be < 8).

/// Library code for BloscLZ.
pub const BLOSC_BLOSCLZ_LIB: u8 = 0;
/// Library code for LZ4/LZ4HC.
pub const BLOSC_LZ4_LIB: u8 = 1;
/// Library code for Snappy (blosc1 only; blosc2 deprecated snappy).
pub const BLOSC_SNAPPY_LIB: u8 = 2;
/// Library code for Zlib.
pub const BLOSC_ZLIB_LIB: u8 = 3;
/// Library code for Zstd.
pub const BLOSC_ZSTD_LIB: u8 = 4;

/// Library code for user-defined codecs.
pub const BLOSC_UDCODEC_LIB: u8 = 6;
/// Library code for super-chunk header.
pub const BLOSC_SCHUNK_LIB: u8 = 7;

// Compression library display names

/// Display name for the BloscLZ library.
pub const BLOSC_BLOSCLZ_LIBNAME: &str = "BloscLZ";
/// Display name for the LZ4 library.
pub const BLOSC_LZ4_LIBNAME: &str = "LZ4";
/// Display name for the Zlib library.
pub const BLOSC_ZLIB_LIBNAME: &str = "Zlib";
/// Display name for the Zstd library.
pub const BLOSC_ZSTD_LIBNAME: &str = "Zstd";

// Compressor format codes — written into the header flags byte to identify
// which library produced the compressed data. Note that LZ4 and LZ4HC share
// the same format.

/// Header format code for BloscLZ.
pub const BLOSC_BLOSCLZ_FORMAT: u8 = BLOSC_BLOSCLZ_LIB;
/// Header format code for LZ4.
pub const BLOSC_LZ4_FORMAT: u8 = BLOSC_LZ4_LIB;
/// Header format code for LZ4HC (same as LZ4).
pub const BLOSC_LZ4HC_FORMAT: u8 = BLOSC_LZ4_LIB;
/// Header format code for Snappy in blosc1 headers.
pub const BLOSC_SNAPPY_FORMAT: u8 = BLOSC_SNAPPY_LIB;
/// Header format code for Zlib.
pub const BLOSC_ZLIB_FORMAT: u8 = BLOSC_ZLIB_LIB;
/// Header format code for Zstd.
pub const BLOSC_ZSTD_FORMAT: u8 = BLOSC_ZSTD_LIB;

/// Header format code for user-defined codecs.
pub const BLOSC_UDCODEC_FORMAT: u8 = BLOSC_UDCODEC_LIB;

// Compressor internal format versions (all start at 1).

/// Internal format version for BloscLZ.
pub const BLOSC_BLOSCLZ_VERSION_FORMAT: u8 = 1;
/// Internal format version for LZ4.
pub const BLOSC_LZ4_VERSION_FORMAT: u8 = 1;
/// Internal format version for LZ4HC (same format as LZ4).
pub const BLOSC_LZ4HC_VERSION_FORMAT: u8 = 1;
/// Internal format version for Zlib.
pub const BLOSC_ZLIB_VERSION_FORMAT: u8 = 1;
/// Internal format version for Zstd.
pub const BLOSC_ZSTD_VERSION_FORMAT: u8 = 1;

/// Internal format version for user-defined codecs.
pub const BLOSC_UDCODEC_VERSION_FORMAT: u8 = 1;

// Block split modes — control whether compressed blocks are split into
// sub-streams for each byte significance level.

/// Always split blocks into sub-streams (experimental).
pub const BLOSC_ALWAYS_SPLIT: u8 = 1;
/// Never split blocks (experimental).
pub const BLOSC_NEVER_SPLIT: u8 = 2;
/// Automatically decide whether to split based on heuristics.
pub const BLOSC_AUTO_SPLIT: u8 = 3;
/// Forward-compatible split mode (default). Behaves like `ALWAYS_SPLIT` for
/// codecs that benefit from it, ensuring older Blosc versions can still read the output.
pub const BLOSC_FORWARD_COMPAT_SPLIT: u8 = 4;

// Byte offsets for fields in the Blosc2 chunk header.

/// Offset of the chunk format version byte.
pub const BLOSC2_CHUNK_VERSION: u8 = 0x0;
/// Offset of the internal codec format version byte.
pub const BLOSC2_CHUNK_VERSIONLZ: u8 = 0x1;
/// Offset of the flags byte (filter flags + codec format in upper bits).
pub const BLOSC2_CHUNK_FLAGS: u8 = 0x2;
/// Offset of the typesize byte.
pub const BLOSC2_CHUNK_TYPESIZE: u8 = 0x3;
/// Offset of the uncompressed size field (i32, little-endian).
pub const BLOSC2_CHUNK_NBYTES: u8 = 0x4;
/// Offset of the block size field (i32, little-endian).
pub const BLOSC2_CHUNK_BLOCKSIZE: u8 = 0x8;
/// Offset of the compressed size field (i32, little-endian, includes header).
pub const BLOSC2_CHUNK_CBYTES: u8 = 0xc;
/// Offset of the 6-byte filter codes array (Blosc2 extended header).
pub const BLOSC2_CHUNK_FILTER_CODES: u8 = 0x10;
/// Offset of the 6-byte filter metadata array (Blosc2 extended header).
pub const BLOSC2_CHUNK_FILTER_META: u8 = 0x18;
/// Offset of the Blosc2-specific flags byte.
pub const BLOSC2_CHUNK_BLOSC2_FLAGS: u8 = 0x1F;

// Run-length encoding markers for special chunk values.

/// No special value — normal compressed data.
pub const BLOSC2_NO_SPECIAL: u8 = 0x0;
/// Special value: all zeros.
pub const BLOSC2_SPECIAL_ZERO: u8 = 0x1;
/// Special value: all NaN.
pub const BLOSC2_SPECIAL_NAN: u8 = 0x2;
/// Special value: a single repeated byte value.
pub const BLOSC2_SPECIAL_VALUE: u8 = 0x3;
/// Special value: uninitialized data.
pub const BLOSC2_SPECIAL_UNINIT: u8 = 0x4;
/// Last valid special value ID.
pub const BLOSC2_SPECIAL_LASTID: u8 = 0x4;
/// Bitmask for extracting the special value type from a flags byte.
pub const BLOSC2_SPECIAL_MASK: u8 = 0x7;

// Error codes returned by Blosc2 operations.

/// Operation completed successfully.
pub const BLOSC2_ERROR_SUCCESS: i32 = 0;
/// Generic failure.
pub const BLOSC2_ERROR_FAILURE: i32 = -1;
/// Bad stream.
pub const BLOSC2_ERROR_STREAM: i32 = -2;
/// Invalid data.
pub const BLOSC2_ERROR_DATA: i32 = -3;
/// Memory allocation failure.
pub const BLOSC2_ERROR_MEMORY_ALLOC: i32 = -4;
/// Not enough space to read.
pub const BLOSC2_ERROR_READ_BUFFER: i32 = -5;
/// Not enough space to write.
pub const BLOSC2_ERROR_WRITE_BUFFER: i32 = -6;
/// Codec not supported.
pub const BLOSC2_ERROR_CODEC_SUPPORT: i32 = -7;
/// Invalid parameter supplied to codec.
pub const BLOSC2_ERROR_CODEC_PARAM: i32 = -8;
/// Codec dictionary error.
pub const BLOSC2_ERROR_CODEC_DICT: i32 = -9;
/// Version not supported.
pub const BLOSC2_ERROR_VERSION_SUPPORT: i32 = -10;
/// Invalid value in header.
pub const BLOSC2_ERROR_INVALID_HEADER: i32 = -11;
/// Invalid parameter supplied to function.
pub const BLOSC2_ERROR_INVALID_PARAM: i32 = -12;
/// File read failure.
pub const BLOSC2_ERROR_FILE_READ: i32 = -13;
/// File write failure.
pub const BLOSC2_ERROR_FILE_WRITE: i32 = -14;
/// File open failure.
pub const BLOSC2_ERROR_FILE_OPEN: i32 = -15;
/// Not found.
pub const BLOSC2_ERROR_NOT_FOUND: i32 = -16;
/// Bad run-length encoding.
pub const BLOSC2_ERROR_RUN_LENGTH: i32 = -17;
/// Filter pipeline error.
pub const BLOSC2_ERROR_FILTER_PIPELINE: i32 = -18;
/// Chunk insert failure.
pub const BLOSC2_ERROR_CHUNK_INSERT: i32 = -19;
/// Chunk append failure.
pub const BLOSC2_ERROR_CHUNK_APPEND: i32 = -20;
/// Chunk update failure.
pub const BLOSC2_ERROR_CHUNK_UPDATE: i32 = -21;
/// Sizes larger than 2 GB not supported.
pub const BLOSC2_ERROR_2GB_LIMIT: i32 = -22;
/// Super-chunk copy failure.
pub const BLOSC2_ERROR_SCHUNK_COPY: i32 = -23;
/// Wrong type for frame.
pub const BLOSC2_ERROR_FRAME_TYPE: i32 = -24;
/// File truncate failure.
pub const BLOSC2_ERROR_FILE_TRUNCATE: i32 = -25;
/// Thread or thread context creation failure.
pub const BLOSC2_ERROR_THREAD_CREATE: i32 = -26;
/// Postfilter failure.
pub const BLOSC2_ERROR_POSTFILTER: i32 = -27;
/// Special frame failure.
pub const BLOSC2_ERROR_FRAME_SPECIAL: i32 = -28;
/// Special super-chunk failure.
pub const BLOSC2_ERROR_SCHUNK_SPECIAL: i32 = -29;
/// IO plugin error.
pub const BLOSC2_ERROR_PLUGIN_IO: i32 = -30;
/// File remove failure.
pub const BLOSC2_ERROR_FILE_REMOVE: i32 = -31;
/// Null pointer.
pub const BLOSC2_ERROR_NULL_POINTER: i32 = -32;
/// Invalid index.
pub const BLOSC2_ERROR_INVALID_INDEX: i32 = -33;
/// Metalayer not found.
pub const BLOSC2_ERROR_METALAYER_NOT_FOUND: i32 = -34;
/// Max buffer size exceeded.
pub const BLOSC2_ERROR_MAX_BUFSIZE_EXCEEDED: i32 = -35;
/// Tuner failure.
pub const BLOSC2_ERROR_TUNER: i32 = -36;
