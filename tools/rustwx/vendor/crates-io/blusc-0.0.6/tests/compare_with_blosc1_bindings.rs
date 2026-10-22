use blosc_src::{
    blosc_init as bound_blosc1_init,
    blosc_compress as bound_blosc1_compress,
    blosc_compress_ctx as bound_blosc1_compress_ctx,
    blosc_cbuffer_sizes as bound_blosc1_cbuffer_sizes,
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
    blosc2_create_dctx as blusc_blosc2_create_dctx,
    blosc2_decompress_ctx as blusc_blosc2_decompress_ctx,
    blosc2_cbuffer_sizes as blusc_blosc2_cbuffer_sizes,
    BLOSC2_DPARAMS_DEFAULTS as BLUSC_BLOSC2_DPARAMS_DEFAULTS,
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

// ============================================================
// blosc-src (C blosc1) compress → blusc decompress
// ============================================================

#[test]
fn blosc1_compress_blusc_decompress_text() {
    let text =
        "I am here writing some very cool and novel words which I will compress and decompress";
    let bytes = text.as_bytes();

    let mut compressed = vec![0u8; bytes.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress(
            5,
            BOUND_BLOSC1_NOSHUFFLE as _,
            std::mem::size_of::<u8>(),
            bytes.len(),
            bytes.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
        )
    };
    assert!(csize > 0, "blosc1 compression failed: {csize}");
    compressed.truncate(csize as usize);

    let mut decompressed = vec![0u8; bytes.len()];
    let dsize = blusc_blosc2_decompress(&compressed, &mut decompressed);
    assert!(dsize > 0, "blusc decompression of blosc1 data failed: {dsize}");

    assert_eq!(text, std::str::from_utf8(&decompressed).unwrap());
}

#[test]
fn blosc1_compress_blusc_decompress_floats() {
    let src: Vec<f32> = (0..10000)
        .map(|num| ((num * 8923) % 100) as f32 / 2.0)
        .collect();
    let typesize = std::mem::size_of::<f32>();
    let src_bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * typesize)
    };

    let mut compressed = vec![0u8; src_bytes.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress(
            5,
            BOUND_BLOSC1_SHUFFLE as _,
            typesize,
            src_bytes.len(),
            src_bytes.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
        )
    };
    assert!(csize > 0, "blosc1 compression failed: {csize}");
    compressed.truncate(csize as usize);

    let mut result = vec![0f32; src.len()];
    let result_bytes = unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, result.len() * typesize)
    };
    let dsize = blusc_blosc2_decompress(&compressed, result_bytes);
    assert!(dsize > 0, "blusc decompression of blosc1 data failed: {dsize}");

    assert_eq!(src, result);
}

#[test]
fn blosc1_compress_blusc_decompress_doubles() {
    let src: Vec<f64> = (0..5000)
        .map(|num| ((num * 7919) % 1000) as f64 / 3.0)
        .collect();
    let typesize = std::mem::size_of::<f64>();
    let src_bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * typesize)
    };

    let mut compressed = vec![0u8; src_bytes.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress(
            5,
            BOUND_BLOSC1_SHUFFLE as _,
            typesize,
            src_bytes.len(),
            src_bytes.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
        )
    };
    assert!(csize > 0, "blosc1 compression failed: {csize}");
    compressed.truncate(csize as usize);

    let mut result = vec![0f64; src.len()];
    let result_bytes = unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, result.len() * typesize)
    };
    let dsize = blusc_blosc2_decompress(&compressed, result_bytes);
    assert!(dsize > 0, "blusc decompression of blosc1 data failed: {dsize}");

    assert_eq!(src, result);
}

#[test]
fn blosc1_compress_ctx_blusc_decompress_blosclz() {
    let src: Vec<u16> = (0..8000).map(|i| (i * 31) as u16).collect();
    let typesize = std::mem::size_of::<u16>();
    let src_bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * typesize)
    };

    let mut compressed = vec![0u8; src_bytes.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress_ctx(
            5,
            BOUND_BLOSC1_SHUFFLE as _,
            typesize,
            src_bytes.len(),
            src_bytes.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
            BLOSC_BLOSCLZ_COMPNAME.as_ptr().cast(),
            0, // auto blocksize
            1, // single thread
        )
    };
    assert!(csize > 0, "blosc1 compress_ctx failed: {csize}");
    compressed.truncate(csize as usize);

    let mut result = vec![0u16; src.len()];
    let result_bytes = unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, result.len() * typesize)
    };
    let dsize = blusc_blosc2_decompress(&compressed, result_bytes);
    assert!(dsize > 0, "blusc decompression failed: {dsize}");

    assert_eq!(src, result);
}

#[test]
fn blosc1_compress_ctx_blusc_decompress_lz4() {
    let src: Vec<i32> = (0..5000).map(|i| (i * 127 - 2500) as i32).collect();
    let typesize = std::mem::size_of::<i32>();
    let src_bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * typesize)
    };

    let mut compressed = vec![0u8; src_bytes.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress_ctx(
            5,
            BOUND_BLOSC1_SHUFFLE as _,
            typesize,
            src_bytes.len(),
            src_bytes.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
            BLOSC_LZ4_COMPNAME.as_ptr().cast(),
            0,
            1,
        )
    };
    assert!(csize > 0, "blosc1 compress_ctx (lz4) failed: {csize}");
    compressed.truncate(csize as usize);

    let mut result = vec![0i32; src.len()];
    let result_bytes = unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, result.len() * typesize)
    };
    let dsize = blusc_blosc2_decompress(&compressed, result_bytes);
    assert!(dsize > 0, "blusc decompression failed: {dsize}");

    assert_eq!(src, result);
}

#[test]
fn blosc1_compress_ctx_blusc_decompress_lz4hc() {
    let src: Vec<i32> = (0..5000).map(|i| (i * 127 - 2500) as i32).collect();
    let typesize = std::mem::size_of::<i32>();
    let src_bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * typesize)
    };

    let mut compressed = vec![0u8; src_bytes.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress_ctx(
            5,
            BOUND_BLOSC1_SHUFFLE as _,
            typesize,
            src_bytes.len(),
            src_bytes.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
            BLOSC_LZ4HC_COMPNAME.as_ptr().cast(),
            0,
            1,
        )
    };
    assert!(csize > 0, "blosc1 compress_ctx (lz4hc) failed: {csize}");
    compressed.truncate(csize as usize);

    let mut result = vec![0i32; src.len()];
    let result_bytes = unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, result.len() * typesize)
    };
    let dsize = blusc_blosc2_decompress(&compressed, result_bytes);
    assert!(dsize > 0, "blusc decompression failed: {dsize}");

    assert_eq!(src, result);
}

#[test]
fn blosc1_compress_ctx_blusc_decompress_snappy() {
    let src: Vec<i32> = (0..5000).map(|i| (i * 127 - 2500) as i32).collect();
    let typesize = std::mem::size_of::<i32>();
    let src_bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * typesize)
    };

    let mut compressed = vec![0u8; src_bytes.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress_ctx(
            5,
            BOUND_BLOSC1_SHUFFLE as _,
            typesize,
            src_bytes.len(),
            src_bytes.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
            BLOSC_SNAPPY_COMPNAME.as_ptr().cast(),
            0,
            1,
        )
    };
    assert!(csize > 0, "blosc1 compress_ctx (snappy) failed: {csize}");
    compressed.truncate(csize as usize);

    let mut result = vec![0i32; src.len()];
    let result_bytes = unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, result.len() * typesize)
    };
    let dsize = blusc_blosc2_decompress(&compressed, result_bytes);
    assert!(dsize > 0, "blusc decompression failed: {dsize}");

    assert_eq!(src, result);
}

#[test]
fn blosc1_compress_ctx_blusc_decompress_zlib() {
    let src: Vec<i32> = (0..5000).map(|i| (i * 127 - 2500) as i32).collect();
    let typesize = std::mem::size_of::<i32>();
    let src_bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * typesize)
    };

    let mut compressed = vec![0u8; src_bytes.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress_ctx(
            5,
            BOUND_BLOSC1_SHUFFLE as _,
            typesize,
            src_bytes.len(),
            src_bytes.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
            BLOSC_ZLIB_COMPNAME.as_ptr().cast(),
            0,
            1,
        )
    };
    if csize <= 0 {
        eprintln!("Skipping zlib test: blosc1 compress_ctx returned {csize}");
        return;
    }
    compressed.truncate(csize as usize);

    let mut result = vec![0i32; src.len()];
    let result_bytes = unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, result.len() * typesize)
    };
    let dsize = blusc_blosc2_decompress(&compressed, result_bytes);
    assert!(dsize > 0, "blusc decompression failed: {dsize}");

    assert_eq!(src, result);
}

#[test]
fn blosc1_compress_ctx_blusc_decompress_zstd() {
    let src: Vec<i32> = (0..5000).map(|i| (i * 127 - 2500) as i32).collect();
    let typesize = std::mem::size_of::<i32>();
    let src_bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * typesize)
    };

    let mut compressed = vec![0u8; src_bytes.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress_ctx(
            5,
            BOUND_BLOSC1_SHUFFLE as _,
            typesize,
            src_bytes.len(),
            src_bytes.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
            BLOSC_ZSTD_COMPNAME.as_ptr().cast(),
            0,
            1,
        )
    };
    if csize <= 0 {
        eprintln!("Skipping zstd test: blosc1 compress_ctx returned {csize}");
        return;
    }
    compressed.truncate(csize as usize);

    let mut result = vec![0i32; src.len()];
    let result_bytes = unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, result.len() * typesize)
    };
    let dsize = blusc_blosc2_decompress(&compressed, result_bytes);
    assert!(dsize > 0, "blusc decompression failed: {dsize}");

    assert_eq!(src, result);
}

// ============================================================
// blosc-src compress → blusc decompress: shuffle modes
// ============================================================

#[test]
fn blosc1_compress_blusc_decompress_noshuffle() {
    let src: Vec<f32> = (0..10000).map(|i| i as f32 * 1.5).collect();
    let typesize = std::mem::size_of::<f32>();
    let src_bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * typesize)
    };

    let mut compressed = vec![0u8; src_bytes.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress(
            5,
            BOUND_BLOSC1_NOSHUFFLE as _,
            typesize,
            src_bytes.len(),
            src_bytes.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
        )
    };
    assert!(csize > 0, "blosc1 compression (noshuffle) failed: {csize}");
    compressed.truncate(csize as usize);

    let mut result = vec![0f32; src.len()];
    let result_bytes = unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, result.len() * typesize)
    };
    let dsize = blusc_blosc2_decompress(&compressed, result_bytes);
    assert!(dsize > 0, "blusc decompression failed: {dsize}");

    assert_eq!(src, result);
}

#[test]
fn blosc1_compress_blusc_decompress_bitshuffle() {
    let src: Vec<f32> = (0..10000).map(|i| i as f32 * 1.5).collect();
    let typesize = std::mem::size_of::<f32>();
    let src_bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * typesize)
    };

    let mut compressed = vec![0u8; src_bytes.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress(
            5,
            BOUND_BLOSC1_BITSHUFFLE as _,
            typesize,
            src_bytes.len(),
            src_bytes.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
        )
    };
    assert!(csize > 0, "blosc1 compression (bitshuffle) failed: {csize}");
    compressed.truncate(csize as usize);

    let mut result = vec![0f32; src.len()];
    let result_bytes = unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, result.len() * typesize)
    };
    let dsize = blusc_blosc2_decompress(&compressed, result_bytes);
    assert!(dsize > 0, "blusc decompression failed: {dsize}");

    assert_eq!(src, result);
}

// ============================================================
// blosc-src compress → blusc decompress: compression levels
// ============================================================

#[test]
fn blosc1_compress_blusc_decompress_clevel0() {
    let src: Vec<u32> = (0..10000).collect();
    let typesize = std::mem::size_of::<u32>();
    let src_bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * typesize)
    };

    let mut compressed = vec![0u8; src_bytes.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress(
            0, // clevel 0 = no compression (memcpy)
            BOUND_BLOSC1_NOSHUFFLE as _,
            typesize,
            src_bytes.len(),
            src_bytes.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
        )
    };
    assert!(csize > 0, "blosc1 compression (clevel 0) failed: {csize}");
    compressed.truncate(csize as usize);

    let mut result = vec![0u32; src.len()];
    let result_bytes = unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, result.len() * typesize)
    };
    let dsize = blusc_blosc2_decompress(&compressed, result_bytes);
    assert!(dsize > 0, "blusc decompression failed: {dsize}");

    assert_eq!(src, result);
}

#[test]
fn blosc1_compress_blusc_decompress_clevel9() {
    let src: Vec<u32> = (0..10000).map(|i| (i * 37) % 256).collect();
    let typesize = std::mem::size_of::<u32>();
    let src_bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * typesize)
    };

    let mut compressed = vec![0u8; src_bytes.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress(
            9, // maximum compression
            BOUND_BLOSC1_SHUFFLE as _,
            typesize,
            src_bytes.len(),
            src_bytes.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
        )
    };
    assert!(csize > 0, "blosc1 compression (clevel 9) failed: {csize}");
    compressed.truncate(csize as usize);

    let mut result = vec![0u32; src.len()];
    let result_bytes = unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, result.len() * typesize)
    };
    let dsize = blusc_blosc2_decompress(&compressed, result_bytes);
    assert!(dsize > 0, "blusc decompression failed: {dsize}");

    assert_eq!(src, result);
}

// ============================================================
// blosc-src compress → blusc decompress: various type sizes
// ============================================================

#[test]
fn blosc1_compress_blusc_decompress_u8() {
    let src: Vec<u8> = (0..50000).map(|i| (i % 256) as u8).collect();
    let typesize = 1;

    let mut compressed = vec![0u8; src.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress(
            5,
            BOUND_BLOSC1_SHUFFLE as _,
            typesize,
            src.len(),
            src.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
        )
    };
    assert!(csize > 0, "blosc1 compression (u8) failed: {csize}");
    compressed.truncate(csize as usize);

    let mut result = vec![0u8; src.len()];
    let dsize = blusc_blosc2_decompress(&compressed, &mut result);
    assert!(dsize > 0, "blusc decompression failed: {dsize}");

    assert_eq!(src, result);
}

#[test]
fn blosc1_compress_blusc_decompress_u64() {
    let src: Vec<u64> = (0..10000).map(|i| i * 123456789).collect();
    let typesize = std::mem::size_of::<u64>();
    let src_bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * typesize)
    };

    let mut compressed = vec![0u8; src_bytes.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress(
            5,
            BOUND_BLOSC1_SHUFFLE as _,
            typesize,
            src_bytes.len(),
            src_bytes.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
        )
    };
    assert!(csize > 0, "blosc1 compression (u64) failed: {csize}");
    compressed.truncate(csize as usize);

    let mut result = vec![0u64; src.len()];
    let result_bytes = unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, result.len() * typesize)
    };
    let dsize = blusc_blosc2_decompress(&compressed, result_bytes);
    assert!(dsize > 0, "blusc decompression failed: {dsize}");

    assert_eq!(src, result);
}

// ============================================================
// blosc-src compress → blusc decompress_ctx
// ============================================================

#[test]
fn blosc1_compress_blusc_decompress_ctx_floats() {
    let src: Vec<f32> = (0..10000)
        .map(|num| ((num * 8923) % 100) as f32 / 2.0)
        .collect();
    let typesize = std::mem::size_of::<f32>();
    let src_bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * typesize)
    };

    let mut compressed = vec![0u8; src_bytes.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress(
            5,
            BOUND_BLOSC1_SHUFFLE as _,
            typesize,
            src_bytes.len(),
            src_bytes.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
        )
    };
    assert!(csize > 0, "blosc1 compression failed: {csize}");
    compressed.truncate(csize as usize);

    let mut result = vec![0f32; src.len()];
    let result_bytes = unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, result.len() * typesize)
    };

    let dparams = BLUSC_BLOSC2_DPARAMS_DEFAULTS;
    let dctx = blusc_blosc2_create_dctx(dparams);
    let dsize = blusc_blosc2_decompress_ctx(&dctx, &compressed, result_bytes);
    assert!(dsize > 0, "blusc decompress_ctx failed: {dsize}");

    assert_eq!(src, result);
}

// ============================================================
// blosc-src compress → blusc cbuffer_sizes
// ============================================================

#[test]
fn blosc1_compress_blusc_cbuffer_sizes() {
    let src: Vec<u32> = (0..10000).collect();
    let typesize = std::mem::size_of::<u32>();
    let src_bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * typesize)
    };

    let mut compressed = vec![0u8; src_bytes.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress(
            5,
            BOUND_BLOSC1_SHUFFLE as _,
            typesize,
            src_bytes.len(),
            src_bytes.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
        )
    };
    assert!(csize > 0, "blosc1 compression failed: {csize}");
    compressed.truncate(csize as usize);

    // Read sizes via blosc-src
    let mut nbytes_c: usize = 0;
    let mut cbytes_c: usize = 0;
    let mut blocksize_c: usize = 0;
    unsafe {
        bound_blosc1_cbuffer_sizes(
            compressed.as_ptr().cast(),
            &mut nbytes_c,
            &mut cbytes_c,
            &mut blocksize_c,
        );
    }

    // Read sizes via blusc
    let (nbytes_r, cbytes_r, blocksize_r) = blusc_blosc2_cbuffer_sizes(&compressed);

    assert_eq!(nbytes_c, nbytes_r, "nbytes mismatch");
    assert_eq!(cbytes_c, cbytes_r, "cbytes mismatch");
    assert_eq!(blocksize_c, blocksize_r, "blocksize mismatch");
}

// ============================================================
// blosc-src compress → blusc decompress: large buffer
// ============================================================

#[test]
fn blosc1_compress_blusc_decompress_large() {
    let src: Vec<f64> = (0..100000)
        .map(|i| (i as f64).sin() * 1000.0)
        .collect();
    let typesize = std::mem::size_of::<f64>();
    let src_bytes = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * typesize)
    };

    let mut compressed = vec![0u8; src_bytes.len() + BOUND_BLOSC1_MAX_OVERHEAD as usize];
    let csize = unsafe {
        bound_blosc1_compress(
            5,
            BOUND_BLOSC1_SHUFFLE as _,
            typesize,
            src_bytes.len(),
            src_bytes.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            compressed.len(),
        )
    };
    assert!(csize > 0, "blosc1 compression (large) failed: {csize}");
    compressed.truncate(csize as usize);

    let mut result = vec![0f64; src.len()];
    let result_bytes = unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, result.len() * typesize)
    };
    let dsize = blusc_blosc2_decompress(&compressed, result_bytes);
    assert!(dsize > 0, "blusc decompression failed: {dsize}");

    assert_eq!(src, result);
}
