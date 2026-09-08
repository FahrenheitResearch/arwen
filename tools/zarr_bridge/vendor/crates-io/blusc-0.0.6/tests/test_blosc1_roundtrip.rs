/// Parametrized cross-validation tests between blosc-src (C blosc v1) and blusc.
///
/// Tests both directions:
/// - blosc-src compress → blusc decompress (blosc1 format, version=2, 16-byte header)
/// - blusc compress → blosc-src decompress (blosc2 format, version=5, 32-byte header)

use blosc_src::{
    blosc_init as bound_blosc1_init,
    blosc_compress_ctx as bound_blosc1_compress_ctx,
    blosc_decompress_ctx as bound_blosc1_decompress_ctx,
    blosc_destroy as bound_blosc1_destroy,
    BLOSC_NOSHUFFLE as BOUND_BLOSC1_NOSHUFFLE,
    BLOSC_SHUFFLE as BOUND_BLOSC1_SHUFFLE,
    BLOSC_BITSHUFFLE as BOUND_BLOSC1_BITSHUFFLE,
    BLOSC_MAX_OVERHEAD as BOUND_BLOSC1_MAX_OVERHEAD,
    BLOSC_BLOSCLZ_COMPNAME,
    BLOSC_LZ4_COMPNAME,
    BLOSC_LZ4HC_COMPNAME,
    BLOSC_SNAPPY_COMPNAME,
    BLOSC_ZLIB_COMPNAME,
    BLOSC_ZSTD_COMPNAME,
};
use blusc::{
    blosc2_decompress as blusc_blosc2_decompress,
    blosc2_create_cctx as blusc_blosc2_create_cctx,
    blosc1_compress_ctx as blusc_blosc1_compress_ctx,
    BLOSC2_CPARAMS_DEFAULTS as BLUSC_BLOSC2_CPARAMS_DEFAULTS,
    BLOSC_MIN_HEADER_LENGTH as BLUSC_BLOSC1_MAX_OVERHEAD,
    BLOSC_BLOSCLZ as BLUSC_BLOSCLZ,
    BLOSC_LZ4 as BLUSC_LZ4,
    BLOSC_LZ4HC as BLUSC_LZ4HC,
    BLOSC_SNAPPY as BLUSC_SNAPPY,
    BLOSC_ZLIB as BLUSC_ZLIB,
    BLOSC_ZSTD as BLUSC_ZSTD,
    BLOSC_NOSHUFFLE as BLUSC_NOSHUFFLE,
    BLOSC_SHUFFLE as BLUSC_SHUFFLE,
    BLOSC_BITSHUFFLE as BLUSC_BITSHUFFLE,
};

use ctor::{ctor, dtor};

#[ctor]
fn blosc1_init() {
    unsafe {
        bound_blosc1_init();
    }
}

#[dtor]
fn blosc1_cleanup() {
    unsafe {
        bound_blosc1_destroy();
    }
}

/// Generate test data of a given typesize and element count.
fn generate_test_data(typesize: usize, num_elements: usize) -> Vec<u8> {
    let nbytes = typesize * num_elements;
    let mut data = vec![0u8; nbytes];
    for i in 0..nbytes {
        // Pseudo-random pattern that produces compressible data
        data[i] = ((i * 8923 + i / 7) % 251) as u8;
    }
    data
}

/// Map blusc compressor code to the corresponding blosc-src compressor name.
fn compcode_to_compname(code: u8) -> &'static [u8] {
    match code {
        0 => BLOSC_BLOSCLZ_COMPNAME,
        1 => BLOSC_LZ4_COMPNAME,
        2 => BLOSC_LZ4HC_COMPNAME,
        3 => BLOSC_SNAPPY_COMPNAME,
        4 => BLOSC_ZLIB_COMPNAME,
        5 => BLOSC_ZSTD_COMPNAME,
        _ => panic!("Unknown compressor code: {code}"),
    }
}

fn shuffle_name(doshuffle: i32) -> &'static str {
    match doshuffle {
        0 => "noshuffle",
        1 => "shuffle",
        2 => "bitshuffle",
        _ => "unknown",
    }
}

// ============================================================
// Direction 1: blosc-src compress → blusc decompress
// ============================================================

/// Compress with blosc-src, decompress with blusc. Must always succeed.
fn blosc1_to_blusc(
    compressor: u8,
    clevel: i32,
    doshuffle: i32,
    typesize: usize,
    num_elements: usize,
) {
    let src = generate_test_data(typesize, num_elements);
    let compname = compcode_to_compname(compressor);

    let mut compressed = vec![0u8; src.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress_ctx(
            clevel,
            doshuffle,
            typesize,
            src.len(),
            src.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
            compname.as_ptr().cast(),
            0, // auto blocksize
            1, // single thread
        )
    };

    assert!(
        csize > 0,
        "blosc1 compress_ctx failed: {csize} \
         (compressor={compressor}, clevel={clevel}, shuffle={}, typesize={typesize}, n={num_elements})",
        shuffle_name(doshuffle)
    );
    compressed.truncate(csize as usize);

    let mut result = vec![0u8; src.len()];
    let dsize = blusc_blosc2_decompress(&compressed, &mut result);
    assert!(
        dsize > 0,
        "blusc decompress failed: {dsize} \
         (compressor={compressor}, clevel={clevel}, shuffle={}, typesize={typesize}, n={num_elements})",
        shuffle_name(doshuffle)
    );

    assert_eq!(
        src, result,
        "Data mismatch after blosc1→blusc roundtrip \
         (compressor={compressor}, clevel={clevel}, shuffle={}, typesize={typesize}, n={num_elements})",
        shuffle_name(doshuffle)
    );
}

/// Compress with blusc, decompress with blosc-src.
fn blusc_to_blosc1(
    compressor: u8,
    clevel: i32,
    doshuffle: i32,
    typesize: usize,
    num_elements: usize,
) {
    let src = generate_test_data(typesize, num_elements);

    let mut compressed = vec![0u8; src.len() + BLUSC_BLOSC1_MAX_OVERHEAD];

    let mut cparams = BLUSC_BLOSC2_CPARAMS_DEFAULTS;
    cparams.compcode = compressor;
    cparams.clevel = clevel as u8;
    cparams.typesize = typesize as i32;
    // Set up filter pipeline
    cparams.filters = [0u8; 6];
    if doshuffle == BLUSC_SHUFFLE as i32 && typesize > 1 {
        cparams.filters[5] = BLUSC_SHUFFLE;
    } else if doshuffle == BLUSC_BITSHUFFLE as i32 {
        cparams.filters[5] = BLUSC_BITSHUFFLE;
    }

    let cctx = blusc_blosc2_create_cctx(cparams);
    let csize = blusc_blosc1_compress_ctx(&cctx, &src, &mut compressed);

    assert!(
        csize > 0,
        "blusc compress failed: {csize} \
         (compressor={compressor}, clevel={clevel}, shuffle={}, typesize={typesize}, n={num_elements})",
        shuffle_name(doshuffle)
    );
    compressed.truncate(csize as usize);

    let mut result = vec![0u8; src.len()];
    let dsize = unsafe {
        bound_blosc1_decompress_ctx(
            compressed.as_ptr().cast(),
            result.as_mut_ptr().cast(),
            result.len(),
            1, // single thread
        )
    };

    assert!(
        dsize > 0,
        "blosc1 decompress of blusc data failed: {dsize} \
         (compressor={compressor}, clevel={clevel}, shuffle={}, typesize={typesize}, n={num_elements})",
        shuffle_name(doshuffle)
    );

    assert_eq!(
        src, result,
        "Data mismatch after blusc→blosc1 roundtrip \
         (compressor={compressor}, clevel={clevel}, shuffle={}, typesize={typesize}, n={num_elements})",
        shuffle_name(doshuffle)
    );
}

// ============================================================
// Parametrized tests: blosc-src → blusc (various codecs)
// ============================================================

#[test]
fn blosc1_to_blusc_blosclz_matrix() {
    let typesizes = [1, 2, 4, 8, 16];
    let shuffles = [
        BOUND_BLOSC1_NOSHUFFLE as i32,
        BOUND_BLOSC1_SHUFFLE as i32,
        BOUND_BLOSC1_BITSHUFFLE as i32,
    ];
    let clevels = [0, 1, 5, 9];
    let num_elements_list = [192, 1792, 8000];

    for &typesize in &typesizes {
        for &doshuffle in &shuffles {
            for &clevel in &clevels {
                for &num_elements in &num_elements_list {
                    blosc1_to_blusc(BLUSC_BLOSCLZ, clevel, doshuffle, typesize, num_elements);
                }
            }
        }
    }
}

#[test]
fn blosc1_to_blusc_lz4_matrix() {
    let typesizes = [1, 2, 4, 8];
    let shuffles = [
        BOUND_BLOSC1_NOSHUFFLE as i32,
        BOUND_BLOSC1_SHUFFLE as i32,
        BOUND_BLOSC1_BITSHUFFLE as i32,
    ];
    let clevels = [0, 5, 9];
    let num_elements_list = [192, 8000];

    for &typesize in &typesizes {
        for &doshuffle in &shuffles {
            for &clevel in &clevels {
                for &num_elements in &num_elements_list {
                    blosc1_to_blusc(BLUSC_LZ4, clevel, doshuffle, typesize, num_elements);
                }
            }
        }
    }
}

#[test]
fn blosc1_to_blusc_lz4hc_matrix() {
    let typesizes = [1, 4, 8];
    let shuffles = [
        BOUND_BLOSC1_NOSHUFFLE as i32,
        BOUND_BLOSC1_SHUFFLE as i32,
    ];
    let clevels = [5, 9];
    let num_elements_list = [192, 8000];

    for &typesize in &typesizes {
        for &doshuffle in &shuffles {
            for &clevel in &clevels {
                for &num_elements in &num_elements_list {
                    blosc1_to_blusc(BLUSC_LZ4HC, clevel, doshuffle, typesize, num_elements);
                }
            }
        }
    }
}

#[test]
fn blosc1_to_blusc_snappy_matrix() {
    let typesizes = [1, 4, 8];
    let shuffles = [
        BOUND_BLOSC1_NOSHUFFLE as i32,
        BOUND_BLOSC1_SHUFFLE as i32,
    ];
    let clevels = [1, 5, 9];
    let num_elements_list = [192, 8000];

    for &typesize in &typesizes {
        for &doshuffle in &shuffles {
            for &clevel in &clevels {
                for &num_elements in &num_elements_list {
                    blosc1_to_blusc(BLUSC_SNAPPY, clevel, doshuffle, typesize, num_elements);
                }
            }
        }
    }
}

#[test]
fn blosc1_to_blusc_zlib_matrix() {
    let typesizes = [1, 4, 8];
    let shuffles = [
        BOUND_BLOSC1_NOSHUFFLE as i32,
        BOUND_BLOSC1_SHUFFLE as i32,
    ];
    let clevels = [1, 5, 9];
    let num_elements_list = [192, 8000];

    for &typesize in &typesizes {
        for &doshuffle in &shuffles {
            for &clevel in &clevels {
                for &num_elements in &num_elements_list {
                    blosc1_to_blusc(BLUSC_ZLIB, clevel, doshuffle, typesize, num_elements);
                }
            }
        }
    }
}

#[test]
fn blosc1_to_blusc_zstd_matrix() {
    let typesizes = [1, 4, 8];
    let shuffles = [
        BOUND_BLOSC1_NOSHUFFLE as i32,
        BOUND_BLOSC1_SHUFFLE as i32,
    ];
    let clevels = [1, 5, 9];
    let num_elements_list = [192, 8000];

    for &typesize in &typesizes {
        for &doshuffle in &shuffles {
            for &clevel in &clevels {
                for &num_elements in &num_elements_list {
                    blosc1_to_blusc(BLUSC_ZSTD, clevel, doshuffle, typesize, num_elements);
                }
            }
        }
    }
}

// ============================================================
// Parametrized tests: blusc → blosc-src (various codecs)
// ============================================================

#[test]
fn blusc_to_blosc1_blosclz_matrix() {
    let typesizes = [1, 2, 4, 8, 16];
    let shuffles = [
        BLUSC_NOSHUFFLE as i32,
        BLUSC_SHUFFLE as i32,
        BLUSC_BITSHUFFLE as i32,
    ];
    let clevels = [0, 1, 5, 9];
    let num_elements_list = [192, 1792, 8000];

    for &typesize in &typesizes {
        for &doshuffle in &shuffles {
            for &clevel in &clevels {
                for &num_elements in &num_elements_list {
                    blusc_to_blosc1(BLUSC_BLOSCLZ, clevel, doshuffle, typesize, num_elements);
                }
            }
        }
    }
}

#[test]
fn blusc_to_blosc1_lz4_matrix() {
    let typesizes = [1, 2, 4, 8];
    let shuffles = [
        BLUSC_NOSHUFFLE as i32,
        BLUSC_SHUFFLE as i32,
        BLUSC_BITSHUFFLE as i32,
    ];
    let clevels = [0, 5, 9];
    let num_elements_list = [192, 8000];

    for &typesize in &typesizes {
        for &doshuffle in &shuffles {
            for &clevel in &clevels {
                for &num_elements in &num_elements_list {
                    blusc_to_blosc1(BLUSC_LZ4, clevel, doshuffle, typesize, num_elements);
                }
            }
        }
    }
}

#[test]
fn blusc_to_blosc1_lz4hc_matrix() {
    let typesizes = [1, 4, 8];
    let shuffles = [
        BLUSC_NOSHUFFLE as i32,
        BLUSC_SHUFFLE as i32,
    ];
    let clevels = [5, 9];
    let num_elements_list = [192, 8000];

    for &typesize in &typesizes {
        for &doshuffle in &shuffles {
            for &clevel in &clevels {
                for &num_elements in &num_elements_list {
                    blusc_to_blosc1(BLUSC_LZ4HC, clevel, doshuffle, typesize, num_elements);
                }
            }
        }
    }
}

#[test]
fn blusc_to_blosc1_snappy_matrix() {
    let typesizes = [1, 4, 8];
    let shuffles = [
        BLUSC_NOSHUFFLE as i32,
        BLUSC_SHUFFLE as i32,
    ];
    let clevels = [1, 5, 9];
    let num_elements_list = [192, 8000];

    for &typesize in &typesizes {
        for &doshuffle in &shuffles {
            for &clevel in &clevels {
                for &num_elements in &num_elements_list {
                    blusc_to_blosc1(BLUSC_SNAPPY, clevel, doshuffle, typesize, num_elements);
                }
            }
        }
    }
}

#[test]
fn blusc_to_blosc1_zlib_matrix() {
    let typesizes = [1, 4, 8];
    let shuffles = [
        BLUSC_NOSHUFFLE as i32,
        BLUSC_SHUFFLE as i32,
    ];
    let clevels = [1, 5, 9];
    let num_elements_list = [192, 8000];

    for &typesize in &typesizes {
        for &doshuffle in &shuffles {
            for &clevel in &clevels {
                for &num_elements in &num_elements_list {
                    blusc_to_blosc1(BLUSC_ZLIB, clevel, doshuffle, typesize, num_elements);
                }
            }
        }
    }
}

#[test]
fn blusc_to_blosc1_zstd_matrix() {
    let typesizes = [1, 4, 8];
    let shuffles = [
        BLUSC_NOSHUFFLE as i32,
        BLUSC_SHUFFLE as i32,
    ];
    let clevels = [1, 5, 9];
    let num_elements_list = [192, 8000];

    for &typesize in &typesizes {
        for &doshuffle in &shuffles {
            for &clevel in &clevels {
                for &num_elements in &num_elements_list {
                    blusc_to_blosc1(BLUSC_ZSTD, clevel, doshuffle, typesize, num_elements);
                }
            }
        }
    }
}

// ============================================================
// Edge cases
// ============================================================

#[test]
fn blosc1_to_blusc_odd_typesize() {
    // Non-power-of-2 type sizes
    for typesize in [3, 5, 7, 11, 13] {
        blosc1_to_blusc(BLUSC_BLOSCLZ, 5, BOUND_BLOSC1_SHUFFLE as i32, typesize, 1000);
        blosc1_to_blusc(BLUSC_BLOSCLZ, 5, BOUND_BLOSC1_NOSHUFFLE as i32, typesize, 1000);
    }
}

#[test]
fn blosc1_to_blusc_small_buffers() {
    // Very small input sizes
    for num_elements in [1, 2, 7, 15, 16, 17] {
        blosc1_to_blusc(BLUSC_BLOSCLZ, 5, BOUND_BLOSC1_NOSHUFFLE as i32, 4, num_elements);
        blosc1_to_blusc(BLUSC_LZ4, 5, BOUND_BLOSC1_SHUFFLE as i32, 4, num_elements);
    }
}

#[test]
fn blosc1_to_blusc_large_buffer() {
    // Large buffer with 100k f64 elements (800KB)
    blosc1_to_blusc(BLUSC_BLOSCLZ, 5, BOUND_BLOSC1_SHUFFLE as i32, 8, 100000);
    blosc1_to_blusc(BLUSC_LZ4, 9, BOUND_BLOSC1_SHUFFLE as i32, 8, 100000);
    blosc1_to_blusc(BLUSC_ZSTD, 5, BOUND_BLOSC1_SHUFFLE as i32, 8, 100000);
}

#[test]
fn blosc1_to_blusc_uniform_data() {
    // All zeros — tests run-length encoding paths
    let src = vec![0u8; 40000]; // 10000 x 4-byte elements worth
    let typesize = 4;

    let mut compressed = vec![0u8; src.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress_ctx(
            5,
            BOUND_BLOSC1_SHUFFLE as i32,
            typesize,
            src.len(),
            src.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
            BLOSC_BLOSCLZ_COMPNAME.as_ptr().cast(),
            0,
            1,
        )
    };
    assert!(csize > 0, "blosc1 compress (zeros) failed: {csize}");
    compressed.truncate(csize as usize);

    let mut result = vec![0xFFu8; src.len()];
    let dsize = blusc_blosc2_decompress(&compressed, &mut result);
    assert!(dsize > 0, "blusc decompress (zeros) failed: {dsize}");

    assert_eq!(src, result);
}

#[test]
fn blosc1_to_blusc_incompressible_data() {
    // Random-looking data that resists compression
    let src: Vec<u8> = (0..40000)
        .map(|i| {
            let x = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (x >> 33) as u8
        })
        .collect();
    let typesize = 4;

    let mut compressed = vec![0u8; src.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress_ctx(
            5,
            BOUND_BLOSC1_NOSHUFFLE as i32,
            typesize,
            src.len(),
            src.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
            BLOSC_BLOSCLZ_COMPNAME.as_ptr().cast(),
            0,
            1,
        )
    };
    assert!(csize > 0, "blosc1 compress (random) failed: {csize}");
    compressed.truncate(csize as usize);

    let mut result = vec![0u8; src.len()];
    let dsize = blusc_blosc2_decompress(&compressed, &mut result);
    assert!(dsize > 0, "blusc decompress (random) failed: {dsize}");

    assert_eq!(src, result);
}
