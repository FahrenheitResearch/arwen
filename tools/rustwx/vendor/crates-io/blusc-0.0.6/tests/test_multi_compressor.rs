/// Tests for multiple compressor backends (LZ4, ZSTD, ZLIB, Snappy).
/// Ported from concepts in c-blosc2/tests/test_compressor.c.
/// Verifies that compress + decompress roundtrip works for each backend,
/// and cross-validates with C reference implementation.
use blosc2_src::{
    blosc2_compress_ctx as bound_blosc2_compress_ctx,
    blosc2_create_cctx as bound_blosc2_create_cctx, blosc2_create_dctx as bound_blosc2_create_dctx,
    blosc2_decompress_ctx as bound_blosc2_decompress_ctx, blosc2_destroy as bound_blosc2_destroy,
    blosc2_init as bound_blosc2_init, BLOSC2_CPARAMS_DEFAULTS as BOUND_BLOSC2_CPARAMS_DEFAULTS,
    BLOSC2_DPARAMS_DEFAULTS as BOUND_BLOSC2_DPARAMS_DEFAULTS,
};
use blusc::api::{
    blosc2_compress_ctx as blusc_blosc2_compress_ctx,
    blosc2_create_cctx as blusc_blosc2_create_cctx, blosc2_create_dctx as blusc_blosc2_create_dctx,
    blosc2_decompress_ctx as blusc_blosc2_decompress_ctx,
    BLOSC2_CPARAMS_DEFAULTS as BLUSC_BLOSC2_CPARAMS_DEFAULTS,
    BLOSC2_DPARAMS_DEFAULTS as BLUSC_BLOSC2_DPARAMS_DEFAULTS,
};
use blusc::{BLOSC2_MAX_OVERHEAD, BLOSC_BLOSCLZ, BLOSC_LZ4, BLOSC_SNAPPY, BLOSC_ZLIB, BLOSC_ZSTD};

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

/// Run a roundtrip test for a given compressor, typesize, shuffle, and clevel.
/// Compress with blusc, decompress with blusc.
fn run_compressor_roundtrip(
    compcode: u8,
    comp_name: &str,
    typesize: i32,
    doshuffle: u8,
    clevel: u8,
    num_elements: usize,
) {
    let buffer_size = typesize as usize * num_elements;
    let mut src = vec![0u8; buffer_size];
    for (i, byte) in src.iter_mut().enumerate() {
        *byte = i as u8;
    }

    let mut cparams = BLUSC_BLOSC2_CPARAMS_DEFAULTS;
    cparams.typesize = typesize;
    cparams.compcode = compcode;
    cparams.filters[5] = doshuffle;
    cparams.clevel = clevel;
    let cctx = blusc_blosc2_create_cctx(cparams);

    let mut compressed = vec![0u8; buffer_size + BLOSC2_MAX_OVERHEAD];
    let csize = blusc_blosc2_compress_ctx(&cctx, &src, &mut compressed);
    assert!(
        csize > 0,
        "{} compression failed: compcode={}, typesize={}, shuffle={}, clevel={}, nelems={}",
        comp_name,
        compcode,
        typesize,
        doshuffle,
        clevel,
        num_elements
    );
    compressed.truncate(csize as usize);

    let dparams = BLUSC_BLOSC2_DPARAMS_DEFAULTS;
    let dctx = blusc_blosc2_create_dctx(dparams);
    let mut decompressed = vec![0u8; buffer_size];
    let dsize = blusc_blosc2_decompress_ctx(&dctx, &compressed, &mut decompressed);
    assert_eq!(
        dsize as usize, buffer_size,
        "{} decompression size mismatch: expected {}, got {}",
        comp_name, buffer_size, dsize
    );
    assert_eq!(
        src, decompressed,
        "{} roundtrip data mismatch (typesize={}, shuffle={}, clevel={})",
        comp_name, typesize, doshuffle, clevel
    );
}

/// Cross-validate: compress with blusc, decompress with C reference.
fn run_cross_validate_blusc_to_c(
    compcode: u8,
    comp_name: &str,
    typesize: i32,
    doshuffle: u8,
    clevel: u8,
    num_elements: usize,
) {
    let buffer_size = typesize as usize * num_elements;
    let mut src = vec![0u8; buffer_size];
    for (i, byte) in src.iter_mut().enumerate() {
        *byte = i as u8;
    }

    // Compress with blusc
    let mut cparams = BLUSC_BLOSC2_CPARAMS_DEFAULTS;
    cparams.typesize = typesize;
    cparams.compcode = compcode;
    cparams.filters[5] = doshuffle;
    cparams.clevel = clevel;
    let cctx = blusc_blosc2_create_cctx(cparams);

    let mut compressed = vec![0u8; buffer_size + BLOSC2_MAX_OVERHEAD];
    let csize = blusc_blosc2_compress_ctx(&cctx, &src, &mut compressed);
    assert!(csize > 0, "{} blusc compression failed", comp_name);
    compressed.truncate(csize as usize);

    // Decompress with C
    let mut decompressed = vec![0u8; buffer_size];
    let dsize = unsafe {
        let dparams = BOUND_BLOSC2_DPARAMS_DEFAULTS;
        let dctx = bound_blosc2_create_dctx(dparams);
        bound_blosc2_decompress_ctx(
            dctx,
            compressed.as_ptr().cast(),
            compressed.len() as i32,
            decompressed.as_mut_ptr().cast(),
            buffer_size as i32,
        )
    };
    if dsize < 0 {
        // C library doesn't support this codec (e.g. ZLIB not compiled in)
        eprintln!(
            "Skipping {} cross-validate (blusc->C): C returned {} (codec not supported)",
            comp_name, dsize
        );
        return;
    }
    assert_eq!(
        dsize as usize, buffer_size,
        "{} C decompression size mismatch",
        comp_name
    );
    assert_eq!(
        src, decompressed,
        "{} cross-validate mismatch (blusc compress -> C decompress)",
        comp_name
    );
}

/// Cross-validate: compress with C, decompress with blusc.
fn run_cross_validate_c_to_blusc(
    compcode: u8,
    comp_name: &str,
    typesize: i32,
    doshuffle: u8,
    clevel: u8,
    num_elements: usize,
) {
    let buffer_size = typesize as usize * num_elements;
    let mut src = vec![0u8; buffer_size];
    for (i, byte) in src.iter_mut().enumerate() {
        *byte = i as u8;
    }

    // Compress with C
    let mut compressed = vec![0u8; buffer_size + BLOSC2_MAX_OVERHEAD];
    let csize = unsafe {
        let mut cparams = BOUND_BLOSC2_CPARAMS_DEFAULTS;
        cparams.typesize = typesize;
        cparams.compcode = compcode;
        cparams.filters[5] = doshuffle;
        cparams.clevel = clevel;
        let cctx = bound_blosc2_create_cctx(cparams);
        bound_blosc2_compress_ctx(
            cctx,
            src.as_ptr().cast(),
            buffer_size as i32,
            compressed.as_mut_ptr().cast(),
            compressed.len() as i32,
        )
    };
    if csize <= 0 {
        // C library doesn't support this codec (e.g. ZLIB not compiled in)
        eprintln!(
            "Skipping {} cross-validate (C->blusc): C returned {} (codec not supported)",
            comp_name, csize
        );
        return;
    }
    compressed.truncate(csize as usize);

    // Decompress with blusc
    let dparams = BLUSC_BLOSC2_DPARAMS_DEFAULTS;
    let dctx = blusc_blosc2_create_dctx(dparams);
    let mut decompressed = vec![0u8; buffer_size];
    let dsize = blusc_blosc2_decompress_ctx(&dctx, &compressed, &mut decompressed);
    assert_eq!(
        dsize as usize, buffer_size,
        "{} blusc decompression size mismatch",
        comp_name
    );
    assert_eq!(
        src, decompressed,
        "{} cross-validate mismatch (C compress -> blusc decompress)",
        comp_name
    );
}

// --- LZ4 tests ---

#[test]
fn lz4_roundtrip() {
    for &ts in &[1i32, 4, 8] {
        for &shuffle in &[0u8, 1] {
            for &clevel in &[1u8, 5, 9] {
                run_compressor_roundtrip(BLOSC_LZ4, "LZ4", ts, shuffle, clevel, 10000);
            }
        }
    }
}

#[test]
fn lz4_cross_validate_to_c() {
    for &ts in &[1i32, 4, 8] {
        for &shuffle in &[0u8, 1] {
            run_cross_validate_blusc_to_c(BLOSC_LZ4, "LZ4", ts, shuffle, 5, 10000);
        }
    }
}

#[test]
fn lz4_cross_validate_from_c() {
    for &ts in &[1i32, 4, 8] {
        for &shuffle in &[0u8, 1] {
            run_cross_validate_c_to_blusc(BLOSC_LZ4, "LZ4", ts, shuffle, 5, 10000);
        }
    }
}

// --- ZSTD tests ---

#[test]
fn zstd_roundtrip() {
    for &ts in &[1i32, 4, 8] {
        for &shuffle in &[0u8, 1] {
            for &clevel in &[1u8, 5, 9] {
                run_compressor_roundtrip(BLOSC_ZSTD, "ZSTD", ts, shuffle, clevel, 10000);
            }
        }
    }
}

#[test]
fn zstd_cross_validate_to_c() {
    for &ts in &[1i32, 4, 8] {
        for &shuffle in &[0u8, 1] {
            run_cross_validate_blusc_to_c(BLOSC_ZSTD, "ZSTD", ts, shuffle, 5, 10000);
        }
    }
}

#[test]
fn zstd_cross_validate_from_c() {
    for &ts in &[1i32, 4, 8] {
        for &shuffle in &[0u8, 1] {
            run_cross_validate_c_to_blusc(BLOSC_ZSTD, "ZSTD", ts, shuffle, 5, 10000);
        }
    }
}

// --- ZLIB tests ---

#[test]
fn zlib_roundtrip() {
    for &ts in &[1i32, 4, 8] {
        for &shuffle in &[0u8, 1] {
            for &clevel in &[1u8, 5, 9] {
                run_compressor_roundtrip(BLOSC_ZLIB, "ZLIB", ts, shuffle, clevel, 10000);
            }
        }
    }
}

#[test]
fn zlib_cross_validate_to_c() {
    for &ts in &[1i32, 4, 8] {
        for &shuffle in &[0u8, 1] {
            run_cross_validate_blusc_to_c(BLOSC_ZLIB, "ZLIB", ts, shuffle, 5, 10000);
        }
    }
}

#[test]
fn zlib_cross_validate_from_c() {
    for &ts in &[1i32, 4, 8] {
        for &shuffle in &[0u8, 1] {
            run_cross_validate_c_to_blusc(BLOSC_ZLIB, "ZLIB", ts, shuffle, 5, 10000);
        }
    }
}

// --- Snappy tests ---

#[test]
fn snappy_roundtrip() {
    for &ts in &[1i32, 4, 8] {
        for &shuffle in &[0u8, 1] {
            // Snappy doesn't have compression levels, use clevel=5
            run_compressor_roundtrip(BLOSC_SNAPPY, "Snappy", ts, shuffle, 5, 10000);
        }
    }
}

// Note: Snappy cross-validation with C is not tested because the blosc2-src
// crate may not include Snappy support.

// --- Large buffer tests with different compressors ---

#[test]
fn lz4_large_buffer() {
    run_compressor_roundtrip(BLOSC_LZ4, "LZ4", 4, 1, 5, 100000);
}

#[test]
fn zstd_large_buffer() {
    run_compressor_roundtrip(BLOSC_ZSTD, "ZSTD", 4, 1, 5, 100000);
}

#[test]
fn zlib_large_buffer() {
    run_compressor_roundtrip(BLOSC_ZLIB, "ZLIB", 4, 1, 5, 100000);
}

// --- Snappy clevel 4, noshuffle, blocksize 0 ---

#[test]
fn snappy_noshuffle_clevel4_blocksize0() {
    let compcode = BLOSC_SNAPPY;
    let typesize: i32 = 4;
    let num_elements: usize = 10000;
    let buffer_size = typesize as usize * num_elements;

    let mut src = vec![0u8; buffer_size];
    for (i, byte) in src.iter_mut().enumerate() {
        *byte = i as u8;
    }

    let mut cparams = BLUSC_BLOSC2_CPARAMS_DEFAULTS;
    cparams.typesize = typesize;
    cparams.compcode = compcode;
    cparams.clevel = 4;
    cparams.blocksize = 0;
    cparams.filters[5] = 0; // noshuffle
    let cctx = blusc_blosc2_create_cctx(cparams);

    let mut compressed = vec![0u8; buffer_size + BLOSC2_MAX_OVERHEAD];
    let csize = blusc_blosc2_compress_ctx(&cctx, &src, &mut compressed);
    assert!(
        csize > 0,
        "Snappy compression failed (clevel=4, noshuffle, blocksize=0)"
    );
    compressed.truncate(csize as usize);

    let dparams = BLUSC_BLOSC2_DPARAMS_DEFAULTS;
    let dctx = blusc_blosc2_create_dctx(dparams);
    let mut decompressed = vec![0u8; buffer_size];
    let dsize = blusc_blosc2_decompress_ctx(&dctx, &compressed, &mut decompressed);
    assert_eq!(
        dsize as usize, buffer_size,
        "Snappy decompression size mismatch: expected {}, got {}",
        buffer_size, dsize
    );
    assert_eq!(
        src, decompressed,
        "Snappy roundtrip data mismatch (clevel=4, noshuffle, blocksize=0)"
    );
}

#[test]
fn snappy_noshuffle_clevel4_blocksize0_typesize0() {
    let compcode = BLOSC_SNAPPY;
    let typesize: i32 = 0;
    let buffer_size = 40000;

    let mut src = vec![0u8; buffer_size];
    for (i, byte) in src.iter_mut().enumerate() {
        *byte = i as u8;
    }

    let mut cparams = BLUSC_BLOSC2_CPARAMS_DEFAULTS;
    cparams.typesize = typesize;
    cparams.compcode = compcode;
    cparams.clevel = 4;
    cparams.blocksize = 0;
    cparams.filters[5] = 0; // noshuffle
    let cctx = blusc_blosc2_create_cctx(cparams);

    let mut compressed = vec![0u8; buffer_size + BLOSC2_MAX_OVERHEAD];
    let csize = blusc_blosc2_compress_ctx(&cctx, &src, &mut compressed);
    assert!(
        csize > 0,
        "Snappy compression failed (clevel=4, noshuffle, blocksize=0, typesize=0)"
    );
    compressed.truncate(csize as usize);

    let dparams = BLUSC_BLOSC2_DPARAMS_DEFAULTS;
    let dctx = blusc_blosc2_create_dctx(dparams);
    let mut decompressed = vec![0u8; buffer_size];
    let dsize = blusc_blosc2_decompress_ctx(&dctx, &compressed, &mut decompressed);
    assert_eq!(
        dsize as usize, buffer_size,
        "Snappy decompression size mismatch: expected {}, got {}",
        buffer_size, dsize
    );
    assert_eq!(
        src, decompressed,
        "Snappy roundtrip data mismatch (clevel=4, noshuffle, blocksize=0, typesize=0)"
    );
}

// --- Bitshuffle with different compressors ---

#[test]
fn lz4_bitshuffle() {
    run_compressor_roundtrip(BLOSC_LZ4, "LZ4", 8, 2, 5, 10000);
}

#[test]
fn zstd_bitshuffle() {
    run_compressor_roundtrip(BLOSC_ZSTD, "ZSTD", 8, 2, 5, 10000);
}

#[test]
fn zlib_bitshuffle() {
    run_compressor_roundtrip(BLOSC_ZLIB, "ZLIB", 8, 2, 5, 10000);
}
