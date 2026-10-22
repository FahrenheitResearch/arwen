/// Ported from c-blosc2/tests/test_compress_roundtrip.c
/// Parametrized compress + decompress round trip tests.
/// Verifies byte-exact match of compressed output between blusc and blosc2-src (C bindings).
///
/// Test matrix from test_compress_roundtrip.csv:
///   type_sizes: [1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 22, 30, 32, 42, 48, 52, 53, 64, 80]
///   num_elements: [7, 192, 1792, 500, 8000, 100000, 702713]
///   shuffle: [0 (noshuffle), 1 (shuffle)]
///   clevel: 5
///   compressor: blosclz (default)
use blosc2_src::{
    blosc2_compress as bound_blosc2_compress, blosc2_decompress as bound_blosc2_decompress,
    blosc2_destroy as bound_blosc2_destroy, blosc2_init as bound_blosc2_init,
    BLOSC2_MAX_OVERHEAD as BOUND_BLOSC2_MAX_OVERHEAD, BLOSC_NOSHUFFLE as BOUND_BLOSC_NOSHUFFLE,
};
use blusc::api::{
    blosc2_compress as blusc_blosc2_compress, blosc2_decompress as blusc_blosc2_decompress,
};
use blusc::BLOSC2_MAX_OVERHEAD;

use ctor::{ctor, dtor};

#[ctor]
fn blosc2_init() {
    unsafe {
        bound_blosc2_init();
    }
}

#[dtor]
fn blosc2_cleanup() {
    unsafe {
        bound_blosc2_destroy();
    }
}

/// Fill buffer with sequential byte values (matching C blosc_test_fill_seq)
fn fill_seq(buf: &mut [u8]) {
    for (k, byte) in buf.iter_mut().enumerate() {
        *byte = k as u8;
    }
}

/// Run a single compress roundtrip test case, comparing blusc output byte-for-byte
/// with the C blosc2 reference implementation.
fn run_compress_roundtrip(type_size: usize, num_elements: usize, clevel: i32, doshuffle: i32) {
    let buffer_size = type_size * num_elements;
    let dest_size = buffer_size + BLOSC2_MAX_OVERHEAD;

    let mut original = vec![0u8; buffer_size];
    fill_seq(&mut original);

    // Compress with blusc (Rust)
    let mut compressed_blusc = vec![0u8; dest_size];
    let csize_blusc = blusc_blosc2_compress(
        clevel,
        doshuffle,
        type_size,
        &original,
        &mut compressed_blusc,
    );

    // Compress with blosc2-src (C)
    let mut compressed_bound = vec![0u8; dest_size];
    let csize_bound = unsafe {
        bound_blosc2_compress(
            clevel,
            doshuffle,
            type_size as i32,
            original.as_ptr().cast(),
            original.len() as i32,
            compressed_bound.as_mut_ptr().cast(),
            compressed_bound.len() as i32,
        )
    };

    assert!(
        csize_blusc > 0,
        "blusc compression failed for type_size={}, num_elements={}, clevel={}, doshuffle={}",
        type_size,
        num_elements,
        clevel,
        doshuffle
    );
    assert!(
        csize_bound > 0,
        "blosc2 (C) compression failed for type_size={}, num_elements={}, clevel={}, doshuffle={}",
        type_size,
        num_elements,
        clevel,
        doshuffle
    );

    // Compare compressed sizes
    assert_eq!(
        csize_blusc as i32, csize_bound,
        "Compressed size mismatch for type_size={}, num_elements={}, clevel={}, doshuffle={}: blusc={}, bound={}",
        type_size, num_elements, clevel, doshuffle, csize_blusc, csize_bound
    );

    // Compare compressed bytes
    compressed_blusc.truncate(csize_blusc as usize);
    compressed_bound.truncate(csize_bound as usize);
    assert_eq!(
        compressed_blusc, compressed_bound,
        "Compressed data mismatch for type_size={}, num_elements={}, clevel={}, doshuffle={}",
        type_size, num_elements, clevel, doshuffle
    );

    // Decompress with blusc and verify roundtrip
    let mut decompressed = vec![0u8; buffer_size];
    let dsize = blusc_blosc2_decompress(&compressed_blusc, &mut decompressed);
    assert_eq!(
        dsize as usize, buffer_size,
        "Decompression size mismatch for type_size={}, num_elements={}, clevel={}, doshuffle={}",
        type_size, num_elements, clevel, doshuffle
    );
    assert_eq!(
        original, decompressed,
        "Roundtrip data mismatch for type_size={}, num_elements={}, clevel={}, doshuffle={}",
        type_size, num_elements, clevel, doshuffle
    );
}

// Type sizes from the CSV
const TYPE_SIZES: &[usize] = &[
    1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 22, 30, 32, 42, 48, 52, 53, 64, 80,
];
const NUM_ELEMENTS: &[usize] = &[7, 192, 1792, 500, 8000, 100000, 702713];

// --- noshuffle tests (doshuffle=0) ---

#[test]
fn compress_roundtrip_noshuffle() {
    for &ts in TYPE_SIZES {
        for &ne in NUM_ELEMENTS {
            run_compress_roundtrip(ts, ne, 5, 0);
        }
    }
}

// --- shuffle tests (doshuffle=1) ---

#[test]
fn compress_roundtrip_shuffle() {
    for &ts in TYPE_SIZES {
        for &ne in NUM_ELEMENTS {
            run_compress_roundtrip(ts, ne, 5, 1);
        }
    }
}

// --- Additional clevel tests not in CSV but important for coverage ---

#[test]
fn compress_roundtrip_clevel_0() {
    for &ts in &[1usize, 4, 8] {
        run_compress_roundtrip(ts, 8000, 0, 0);
        run_compress_roundtrip(ts, 8000, 0, 1);
    }
}

#[test]
fn compress_roundtrip_clevel_1() {
    for &ts in &[1usize, 4, 8] {
        run_compress_roundtrip(ts, 8000, 1, 0);
        run_compress_roundtrip(ts, 8000, 1, 1);
    }
}

#[test]
fn compress_roundtrip_clevel_9() {
    for &ts in &[1usize, 4, 8] {
        run_compress_roundtrip(ts, 8000, 9, 0);
        run_compress_roundtrip(ts, 8000, 9, 1);
    }
}
