/// Unit tests for the idiomatic Rust convenience API in `blusc::convenience`.
///
/// These tests exercise the four public functions:
///   - `blosc1_compress` / `blosc1_decompress`
///   - `blosc2_compress` / `blosc2_decompress`
///
/// All functions take only bytes and return `Result<Vec<u8>, BloscError>`.
/// No C bindings are required — correctness is verified by roundtrip.
use blusc::api::{
    blosc1_compress as api_blosc1_compress, blosc1_decompress as api_blosc1_decompress,
    blosc2_cbuffer_sizes, blosc2_compress as api_blosc2_compress,
    blosc2_decompress as api_blosc2_decompress,
};
use blusc::convenience::{blosc1_compress, blosc1_decompress, blosc2_compress, blosc2_decompress};
use blusc::{BLOSC2_MAX_OVERHEAD, BLOSC_SHUFFLE};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sequential(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 256) as u8).collect()
}

fn repeated(byte: u8, n: usize) -> Vec<u8> {
    vec![byte; n]
}

// ---------------------------------------------------------------------------
// blosc1 roundtrip
// ---------------------------------------------------------------------------

#[test]
fn blosc1_roundtrip_small() {
    let src = sequential(256);
    let compressed = blosc1_compress(&src).expect("blosc1_compress failed");
    assert!(!compressed.is_empty());
    let recovered = blosc1_decompress(&compressed).expect("blosc1_decompress failed");
    assert_eq!(src, recovered);
}

#[test]
fn blosc1_roundtrip_large() {
    let src = sequential(100_000);
    let compressed = blosc1_compress(&src).expect("blosc1_compress failed");
    let recovered = blosc1_decompress(&compressed).expect("blosc1_decompress failed");
    assert_eq!(src, recovered);
}

#[test]
fn blosc1_roundtrip_all_zeros() {
    let src = repeated(0, 64_000);
    let compressed = blosc1_compress(&src).expect("blosc1_compress failed");
    assert!(
        compressed.len() < src.len() / 10,
        "all-zeros should compress well: compressed={}, src={}",
        compressed.len(),
        src.len()
    );
    let recovered = blosc1_decompress(&compressed).expect("blosc1_decompress failed");
    assert_eq!(src, recovered);
}

#[test]
fn blosc1_roundtrip_single_byte() {
    let src = vec![42u8];
    let compressed = blosc1_compress(&src).expect("blosc1_compress failed");
    let recovered = blosc1_decompress(&compressed).expect("blosc1_decompress failed");
    assert_eq!(src, recovered);
}

#[test]
fn blosc1_roundtrip_runs() {
    // Runs of repeated bytes — exercises run-length encoding paths.
    let mut src = Vec::with_capacity(10_000);
    for b in 0u8..100 {
        src.extend(std::iter::repeat(b).take(100));
    }
    let compressed = blosc1_compress(&src).expect("blosc1_compress failed");
    let recovered = blosc1_decompress(&compressed).expect("blosc1_decompress failed");
    assert_eq!(src, recovered);
}

// ---------------------------------------------------------------------------
// blosc2 roundtrip
// ---------------------------------------------------------------------------

#[test]
fn blosc2_roundtrip_small() {
    let src = sequential(256);
    let compressed = blosc2_compress(&src).expect("blosc2_compress failed");
    assert!(!compressed.is_empty());
    let recovered = blosc2_decompress(&compressed).expect("blosc2_decompress failed");
    assert_eq!(src, recovered);
}

#[test]
fn blosc2_roundtrip_large() {
    let src = sequential(100_000);
    let compressed = blosc2_compress(&src).expect("blosc2_compress failed");
    let recovered = blosc2_decompress(&compressed).expect("blosc2_decompress failed");
    assert_eq!(src, recovered);
}

#[test]
fn blosc2_roundtrip_all_zeros() {
    let src = repeated(0, 64_000);
    let compressed = blosc2_compress(&src).expect("blosc2_compress failed");
    assert!(
        compressed.len() < src.len() / 10,
        "all-zeros should compress well: compressed={}, src={}",
        compressed.len(),
        src.len()
    );
    let recovered = blosc2_decompress(&compressed).expect("blosc2_decompress failed");
    assert_eq!(src, recovered);
}

#[test]
fn blosc2_roundtrip_single_byte() {
    let src = vec![0xABu8];
    let compressed = blosc2_compress(&src).expect("blosc2_compress failed");
    let recovered = blosc2_decompress(&compressed).expect("blosc2_decompress failed");
    assert_eq!(src, recovered);
}

#[test]
fn blosc2_roundtrip_runs() {
    let mut src = Vec::with_capacity(10_000);
    for b in 0u8..100 {
        src.extend(std::iter::repeat(b).take(100));
    }
    let compressed = blosc2_compress(&src).expect("blosc2_compress failed");
    let recovered = blosc2_decompress(&compressed).expect("blosc2_decompress failed");
    assert_eq!(src, recovered);
}

// ---------------------------------------------------------------------------
// Cross-format: blosc1 compressed -> blosc2_decompress (and vice versa).
// Both decompressors auto-detect the format from the header version byte.
// ---------------------------------------------------------------------------

#[test]
fn blosc1_compress_blosc2_decompress() {
    let src = sequential(10_000);
    let compressed = blosc1_compress(&src).expect("blosc1_compress failed");
    let recovered = blosc2_decompress(&compressed).expect("blosc2_decompress on blosc1 data failed");
    assert_eq!(src, recovered);
}

#[test]
fn blosc2_compress_blosc1_decompress() {
    let src = sequential(10_000);
    let compressed = blosc2_compress(&src).expect("blosc2_compress failed");
    let recovered = blosc1_decompress(&compressed).expect("blosc1_decompress on blosc2 data failed");
    assert_eq!(src, recovered);
}

// ---------------------------------------------------------------------------
// Compressed output properties
// ---------------------------------------------------------------------------

#[test]
fn blosc1_compressed_size_is_smaller_for_compressible_data() {
    let src = repeated(0xAA, 50_000);
    let compressed = blosc1_compress(&src).expect("blosc1_compress failed");
    assert!(
        compressed.len() < src.len(),
        "compressed={} should be smaller than src={}",
        compressed.len(),
        src.len()
    );
}

#[test]
fn blosc2_compressed_size_is_smaller_for_compressible_data() {
    let src = repeated(0xAA, 50_000);
    let compressed = blosc2_compress(&src).expect("blosc2_compress failed");
    assert!(
        compressed.len() < src.len(),
        "compressed={} should be smaller than src={}",
        compressed.len(),
        src.len()
    );
}

// ---------------------------------------------------------------------------
// Cross-API: convenience encode → non-convenience decode, and vice versa.
// Verifies that both APIs produce and consume identical wire formats.
// ---------------------------------------------------------------------------

/// Decompress with the non-convenience API: read nbytes from header, allocate, call.
fn api_decompress_blosc1(compressed: &[u8]) -> Vec<u8> {
    let (nbytes, _, _) = blosc2_cbuffer_sizes(compressed);
    let mut dest = vec![0u8; nbytes];
    let n = api_blosc1_decompress(compressed, &mut dest);
    assert!(n >= 0, "api_blosc1_decompress failed: {}", n);
    dest
}

fn api_decompress_blosc2(compressed: &[u8]) -> Vec<u8> {
    let (nbytes, _, _) = blosc2_cbuffer_sizes(compressed);
    let mut dest = vec![0u8; nbytes];
    let n = api_blosc2_decompress(compressed, &mut dest);
    assert!(n >= 0, "api_blosc2_decompress failed: {}", n);
    dest
}

#[test]
fn convenience_blosc1_compress_then_api_blosc1_decompress() {
    let src = sequential(10_000);
    let compressed = blosc1_compress(&src).expect("blosc1_compress failed");
    let recovered = api_decompress_blosc1(&compressed);
    assert_eq!(src, recovered);
}

#[test]
fn api_blosc1_compress_then_convenience_blosc1_decompress() {
    let src = sequential(10_000);
    let mut dest = vec![0u8; src.len() + BLOSC2_MAX_OVERHEAD];
    let n = api_blosc1_compress(5, BLOSC_SHUFFLE as i32, 8, &src, &mut dest);
    assert!(n > 0, "api_blosc1_compress failed: {}", n);
    dest.truncate(n as usize);
    let recovered = blosc1_decompress(&dest).expect("blosc1_decompress failed");
    assert_eq!(src, recovered);
}

#[test]
fn convenience_blosc2_compress_then_api_blosc2_decompress() {
    let src = sequential(10_000);
    let compressed = blosc2_compress(&src).expect("blosc2_compress failed");
    let recovered = api_decompress_blosc2(&compressed);
    assert_eq!(src, recovered);
}

#[test]
fn api_blosc2_compress_then_convenience_blosc2_decompress() {
    let src = sequential(10_000);
    let mut dest = vec![0u8; src.len() + BLOSC2_MAX_OVERHEAD];
    let n = api_blosc2_compress(5, BLOSC_SHUFFLE as i32, 8, &src, &mut dest);
    assert!(n > 0, "api_blosc2_compress failed: {}", n);
    dest.truncate(n as usize);
    let recovered = blosc2_decompress(&dest).expect("blosc2_decompress failed");
    assert_eq!(src, recovered);
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn blosc1_decompress_truncated_header_returns_error() {
    let bad = vec![0u8; 8]; // too short to be a valid header
    assert!(
        blosc1_decompress(&bad).is_err(),
        "should fail on truncated header"
    );
}

#[test]
fn blosc2_decompress_truncated_header_returns_error() {
    let bad = vec![0u8; 8];
    assert!(
        blosc2_decompress(&bad).is_err(),
        "should fail on truncated header"
    );
}

#[test]
fn blosc1_decompress_empty_input_returns_error() {
    assert!(blosc1_decompress(&[]).is_err(), "should fail on empty input");
}

#[test]
fn blosc2_decompress_empty_input_returns_error() {
    assert!(blosc2_decompress(&[]).is_err(), "should fail on empty input");
}
