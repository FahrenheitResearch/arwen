/// Ported from c-blosc2/tests/test_compressor.c (test_small_blocksize, test_small_buffer)
/// and other edge case tests.
/// Tests small buffers, small blocksizes, empty-ish data, and boundary conditions.
use blosc2_src::{
    blosc2_compress as bound_blosc2_compress, blosc2_decompress as bound_blosc2_decompress,
    blosc2_destroy as bound_blosc2_destroy, blosc2_init as bound_blosc2_init,
    BLOSC2_MAX_OVERHEAD as BOUND_BLOSC2_MAX_OVERHEAD,
};
use blusc::api::{
    blosc1_cbuffer_metainfo as blusc_blosc1_cbuffer_metainfo,
    blosc1_cbuffer_sizes as blusc_blosc1_cbuffer_sizes,
    blosc1_cbuffer_validate as blusc_blosc1_cbuffer_validate,
    blosc2_compress as blusc_blosc2_compress, blosc2_compress_ctx as blusc_blosc2_compress_ctx,
    blosc2_create_cctx as blusc_blosc2_create_cctx, blosc2_create_dctx as blusc_blosc2_create_dctx,
    blosc2_decompress as blusc_blosc2_decompress,
    blosc2_decompress_ctx as blusc_blosc2_decompress_ctx,
    BLOSC2_CPARAMS_DEFAULTS as BLUSC_BLOSC2_CPARAMS_DEFAULTS,
    BLOSC2_DPARAMS_DEFAULTS as BLUSC_BLOSC2_DPARAMS_DEFAULTS,
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

/// Ported from test_compressor.c: test_small_buffer
/// Compress and decompress a 2-byte buffer.
#[test]
fn small_buffer_2_bytes() {
    let src: Vec<u8> = vec![0, 1];
    let mut compressed = vec![0u8; src.len() + BLOSC2_MAX_OVERHEAD];

    // Context-based
    let mut cparams = BLUSC_BLOSC2_CPARAMS_DEFAULTS;
    cparams.typesize = 1;
    let cctx = blusc_blosc2_create_cctx(cparams);
    let csize = blusc_blosc2_compress_ctx(&cctx, &src, &mut compressed);
    assert!(
        csize > 0,
        "Small buffer compression failed: csize={}",
        csize
    );
    compressed.truncate(csize as usize);

    let dparams = BLUSC_BLOSC2_DPARAMS_DEFAULTS;
    let dctx = blusc_blosc2_create_dctx(dparams);
    let mut decompressed = vec![0u8; src.len()];
    let dsize = blusc_blosc2_decompress_ctx(&dctx, &compressed, &mut decompressed);
    assert_eq!(
        dsize as usize,
        src.len(),
        "Small buffer decompression size mismatch"
    );
    assert_eq!(src, decompressed, "Small buffer data mismatch");
}

/// Ported from test_compressor.c: test_small_buffer (blosc2_compress path)
#[test]
fn small_buffer_2_bytes_direct() {
    let src: Vec<u8> = vec![0, 1];
    let mut compressed = vec![0u8; src.len() + BLOSC2_MAX_OVERHEAD];

    let csize = blusc_blosc2_compress(9, 1, 1, &src, &mut compressed);
    assert!(csize > 0, "Small buffer direct compression failed");
    compressed.truncate(csize as usize);

    let mut decompressed = vec![0u8; src.len()];
    let dsize = blusc_blosc2_decompress(&compressed, &mut decompressed);
    assert_eq!(dsize as usize, src.len());
    assert_eq!(src, decompressed);
}

/// Test minimum BLOSC_MIN_BUFFERSIZE boundary (32 bytes).
#[test]
fn buffer_at_min_buffersize() {
    let src = vec![42u8; 32];
    let mut compressed = vec![0u8; src.len() + BLOSC2_MAX_OVERHEAD];

    let csize = blusc_blosc2_compress(5, 0, 1, &src, &mut compressed);
    assert!(csize > 0, "Compression at min buffersize failed");
    compressed.truncate(csize as usize);

    let mut decompressed = vec![0u8; src.len()];
    let dsize = blusc_blosc2_decompress(&compressed, &mut decompressed);
    assert_eq!(dsize as usize, 32);
    assert_eq!(src, decompressed);
}

/// Test buffers just below and at the BLOSC_MIN_BUFFERSIZE boundary.
#[test]
fn buffer_below_min_buffersize() {
    for size in [1, 2, 4, 8, 15, 16, 31] {
        let src: Vec<u8> = (0..size as u8).collect();
        let mut compressed = vec![0u8; src.len() + BLOSC2_MAX_OVERHEAD];

        let csize = blusc_blosc2_compress(5, 0, 1, &src, &mut compressed);
        assert!(csize > 0, "Compression failed for size={}", size);
        compressed.truncate(csize as usize);

        let mut decompressed = vec![0u8; src.len()];
        let dsize = blusc_blosc2_decompress(&compressed, &mut decompressed);
        assert_eq!(
            dsize as usize, size,
            "Decompression size mismatch for size={}",
            size
        );
        assert_eq!(src, decompressed, "Data mismatch for size={}", size);
    }
}

/// Test all-zeros buffer (should compress very well).
#[test]
fn all_zeros_buffer() {
    let src = vec![0u8; 100_000];
    let mut compressed = vec![0u8; src.len() + BLOSC2_MAX_OVERHEAD];

    let csize = blusc_blosc2_compress(5, 1, 4, &src, &mut compressed);
    assert!(csize > 0, "All-zeros compression failed");
    assert!(
        (csize as usize) < src.len() / 10,
        "All-zeros should compress very well: csize={}, src_len={}",
        csize,
        src.len()
    );
    compressed.truncate(csize as usize);

    let mut decompressed = vec![0u8; src.len()];
    let dsize = blusc_blosc2_decompress(&compressed, &mut decompressed);
    assert_eq!(dsize as usize, src.len());
    assert_eq!(src, decompressed);
}

/// Test all-ones buffer.
#[test]
fn all_ones_buffer() {
    let src = vec![0xFFu8; 100_000];
    let mut compressed = vec![0u8; src.len() + BLOSC2_MAX_OVERHEAD];

    let csize = blusc_blosc2_compress(5, 1, 4, &src, &mut compressed);
    assert!(csize > 0);
    compressed.truncate(csize as usize);

    let mut decompressed = vec![0u8; src.len()];
    let dsize = blusc_blosc2_decompress(&compressed, &mut decompressed);
    assert_eq!(dsize as usize, src.len());
    assert_eq!(src, decompressed);
}

/// Test cbuffer_sizes, cbuffer_metainfo, cbuffer_validate on compressed data.
#[test]
fn cbuffer_info_functions() {
    let src: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
    let mut compressed = vec![0u8; src.len() + BLOSC2_MAX_OVERHEAD];

    let csize = blusc_blosc2_compress(5, 1, 4, &src, &mut compressed);
    assert!(csize > 0);
    compressed.truncate(csize as usize);

    // cbuffer_sizes
    let (nbytes, cbytes, blocksize) = blusc_blosc1_cbuffer_sizes(&compressed);
    assert_eq!(nbytes, 1000, "nbytes mismatch");
    assert_eq!(cbytes, csize as usize, "cbytes mismatch");
    assert!(blocksize > 0, "blocksize should be > 0");

    // cbuffer_metainfo
    let info = blusc_blosc1_cbuffer_metainfo(&compressed);
    assert!(info.is_some(), "metainfo should succeed");
    let (typesize, flags) = info.unwrap();
    assert_eq!(typesize, 4, "typesize mismatch");

    // cbuffer_validate
    let valid = blusc_blosc1_cbuffer_validate(&compressed, csize as usize);
    assert!(valid.is_ok(), "validate should succeed");
}

/// Cross-validate compressed bytes between blusc and C for various data patterns.
#[test]
fn cross_validate_patterns() {
    let patterns: Vec<(&str, Vec<u8>)> = vec![
        ("sequential", (0..10000).map(|i| (i % 256) as u8).collect()),
        ("repeated_block", {
            let mut v = Vec::with_capacity(10000);
            for _ in 0..100 {
                v.extend_from_slice(&[
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                    63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
                    83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                ]);
            }
            v
        }),
        ("runs", {
            let mut v = Vec::with_capacity(10000);
            for i in 0..100u8 {
                v.extend(std::iter::repeat(i).take(100));
            }
            v
        }),
    ];

    for (name, src) in &patterns {
        let dest_size = src.len() + BLOSC2_MAX_OVERHEAD;

        let mut compressed_blusc = vec![0u8; dest_size];
        let csize_blusc = blusc_blosc2_compress(5, 0, 1, src, &mut compressed_blusc);

        let mut compressed_c = vec![0u8; dest_size];
        let csize_c = unsafe {
            bound_blosc2_compress(
                5,
                0,
                1,
                src.as_ptr().cast(),
                src.len() as i32,
                compressed_c.as_mut_ptr().cast(),
                compressed_c.len() as i32,
            )
        };

        assert!(csize_blusc > 0, "blusc failed on pattern '{}'", name);
        assert!(csize_c > 0, "C failed on pattern '{}'", name);
        assert_eq!(
            csize_blusc as i32, csize_c,
            "Compressed size mismatch on pattern '{}': blusc={}, c={}",
            name, csize_blusc, csize_c
        );

        compressed_blusc.truncate(csize_blusc as usize);
        compressed_c.truncate(csize_c as usize);
        assert_eq!(
            compressed_blusc, compressed_c,
            "Compressed bytes mismatch on pattern '{}'",
            name
        );

        // Roundtrip
        let mut decompressed = vec![0u8; src.len()];
        let dsize = blusc_blosc2_decompress(&compressed_blusc, &mut decompressed);
        assert_eq!(
            dsize as usize,
            src.len(),
            "Decompression size mismatch on pattern '{}'",
            name
        );
        assert_eq!(
            src, &decompressed,
            "Roundtrip mismatch on pattern '{}'",
            name
        );
    }
}
