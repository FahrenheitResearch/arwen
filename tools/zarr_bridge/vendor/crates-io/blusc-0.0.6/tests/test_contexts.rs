/// Ported from c-blosc2/tests/test_contexts.c
/// Tests context-based compression and decompression.
/// Uses blosc2_create_cctx / blosc2_compress_ctx / blosc2_create_dctx / blosc2_decompress_ctx.
use blosc2_src::{
    blosc2_compress_ctx as bound_blosc2_compress_ctx,
    blosc2_create_cctx as bound_blosc2_create_cctx, blosc2_create_dctx as bound_blosc2_create_dctx,
    blosc2_decompress_ctx as bound_blosc2_decompress_ctx, blosc2_destroy as bound_blosc2_destroy,
    blosc2_init as bound_blosc2_init, BLOSC2_CPARAMS_DEFAULTS as BOUND_BLOSC2_CPARAMS_DEFAULTS,
    BLOSC2_DPARAMS_DEFAULTS as BOUND_BLOSC2_DPARAMS_DEFAULTS,
};
use blusc::api::{
    blosc1_getitem as blusc_blosc1_getitem, blosc2_compress_ctx as blusc_blosc2_compress_ctx,
    blosc2_create_cctx as blusc_blosc2_create_cctx, blosc2_create_dctx as blusc_blosc2_create_dctx,
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

const SIZE: usize = 500 * 1000;

/// Ported from test_contexts.c: compress with ctx, decompress with ctx, getitem with ctx.
/// Uses int32 data filled with sequential values.
#[test]
fn context_compress_decompress_i32() {
    let src: Vec<i32> = (0..SIZE as i32).collect();
    let src_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, SIZE * 4) };
    let isize = SIZE * std::mem::size_of::<i32>();

    // Compress with blusc context
    let mut cparams = BLUSC_BLOSC2_CPARAMS_DEFAULTS;
    cparams.typesize = std::mem::size_of::<i32>() as i32;
    cparams.compcode = 0; // BLOSCLZ
    cparams.filters[5] = 1; // SHUFFLE
    cparams.clevel = 5;
    let cctx = blusc_blosc2_create_cctx(cparams);

    let mut compressed = vec![0u8; isize + BLOSC2_MAX_OVERHEAD];
    let csize = blusc_blosc2_compress_ctx(&cctx, src_bytes, &mut compressed);
    assert!(csize > 0, "Context compression failed: csize={}", csize);
    assert!((csize as usize) < isize, "Data should compress");
    compressed.truncate(csize as usize);

    // Decompress with blusc context
    let dparams = BLUSC_BLOSC2_DPARAMS_DEFAULTS;
    let dctx = blusc_blosc2_create_dctx(dparams);
    let mut decompressed = vec![0u8; isize];
    let dsize = blusc_blosc2_decompress_ctx(&dctx, &compressed, &mut decompressed);
    assert_eq!(dsize as usize, isize, "Decompression size mismatch");
    assert_eq!(src_bytes, &decompressed[..], "Decompressed data mismatch");

    // Test getitem on a subset (items 5..10)
    let subset_start = 5;
    let subset_count = 5;
    let subset_bytes = subset_count * std::mem::size_of::<i32>();
    let mut subset = vec![0u8; subset_bytes];
    let gsize = blusc_blosc1_getitem(&compressed, subset_start, subset_count as i32, &mut subset);
    assert!(gsize > 0, "getitem failed");

    let expected: Vec<i32> = (5..10).collect();
    let expected_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(expected.as_ptr() as *const u8, subset_bytes) };
    assert_eq!(&subset[..], expected_bytes, "getitem data mismatch");
}

/// Cross-validate: compress with blusc ctx, decompress with C ctx.
#[test]
fn context_blusc_compress_c_decompress() {
    let src: Vec<i32> = (0..SIZE as i32).collect();
    let src_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, SIZE * 4) };
    let isize = SIZE * std::mem::size_of::<i32>();

    // Compress with blusc
    let mut cparams = BLUSC_BLOSC2_CPARAMS_DEFAULTS;
    cparams.typesize = 4;
    cparams.compcode = 0;
    cparams.filters[5] = 1;
    cparams.clevel = 5;
    let cctx = blusc_blosc2_create_cctx(cparams);

    let mut compressed = vec![0u8; isize + BLOSC2_MAX_OVERHEAD];
    let csize = blusc_blosc2_compress_ctx(&cctx, src_bytes, &mut compressed);
    assert!(csize > 0, "blusc compression failed");
    compressed.truncate(csize as usize);

    // Decompress with C
    let mut decompressed = vec![0i32; SIZE];
    let dsize = unsafe {
        let dparams = BOUND_BLOSC2_DPARAMS_DEFAULTS;
        let dctx = bound_blosc2_create_dctx(dparams);
        bound_blosc2_decompress_ctx(
            dctx,
            compressed.as_ptr().cast(),
            compressed.len() as i32,
            decompressed.as_mut_ptr().cast(),
            isize as i32,
        )
    };
    assert_eq!(dsize as usize, isize, "C decompression size mismatch");
    assert_eq!(src, decompressed, "C decompressed data mismatch");
}

/// Cross-validate: compress with C ctx, decompress with blusc ctx.
#[test]
fn context_c_compress_blusc_decompress() {
    let src: Vec<i32> = (0..SIZE as i32).collect();
    let isize = SIZE * std::mem::size_of::<i32>();

    // Compress with C
    let mut compressed = vec![0u8; isize + BLOSC2_MAX_OVERHEAD];
    let csize = unsafe {
        let mut cparams = BOUND_BLOSC2_CPARAMS_DEFAULTS;
        cparams.typesize = 4;
        cparams.compcode = 0;
        cparams.filters[5] = 1;
        cparams.clevel = 5;
        let cctx = bound_blosc2_create_cctx(cparams);
        bound_blosc2_compress_ctx(
            cctx,
            src.as_ptr().cast(),
            isize as i32,
            compressed.as_mut_ptr().cast(),
            compressed.len() as i32,
        )
    };
    assert!(csize > 0, "C compression failed");
    compressed.truncate(csize as usize);

    // Decompress with blusc
    let dparams = BLUSC_BLOSC2_DPARAMS_DEFAULTS;
    let dctx = blusc_blosc2_create_dctx(dparams);
    let mut decompressed = vec![0u8; isize];
    let dsize = blusc_blosc2_decompress_ctx(&dctx, &compressed, &mut decompressed);
    assert_eq!(dsize as usize, isize, "blusc decompression size mismatch");

    let result: &[i32] =
        unsafe { std::slice::from_raw_parts(decompressed.as_ptr() as *const i32, SIZE) };
    assert_eq!(&src[..], result, "blusc decompressed data mismatch");
}

/// Test different compression levels via context.
#[test]
fn context_clevels() {
    let src: Vec<i32> = (0..10000i32).collect();
    let src_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4) };
    let isize = src.len() * 4;

    for clevel in 0..=9u8 {
        let mut cparams = BLUSC_BLOSC2_CPARAMS_DEFAULTS;
        cparams.typesize = 4;
        cparams.compcode = 0;
        cparams.filters[5] = 1;
        cparams.clevel = clevel;
        let cctx = blusc_blosc2_create_cctx(cparams);

        let mut compressed = vec![0u8; isize + BLOSC2_MAX_OVERHEAD];
        let csize = blusc_blosc2_compress_ctx(&cctx, src_bytes, &mut compressed);
        assert!(csize > 0, "Compression failed at clevel={}", clevel);
        compressed.truncate(csize as usize);

        let dparams = BLUSC_BLOSC2_DPARAMS_DEFAULTS;
        let dctx = blusc_blosc2_create_dctx(dparams);
        let mut decompressed = vec![0u8; isize];
        let dsize = blusc_blosc2_decompress_ctx(&dctx, &compressed, &mut decompressed);
        assert_eq!(
            dsize as usize, isize,
            "Decompression size mismatch at clevel={}",
            clevel
        );
        assert_eq!(
            src_bytes,
            &decompressed[..],
            "Roundtrip failed at clevel={}",
            clevel
        );
    }
}

/// Test bitshuffle via context.
#[test]
fn context_bitshuffle() {
    // 10000 elements of f64 = 80000 bytes, blocksize/typesize = multiple of 8
    let src: Vec<f64> = (0..10000).map(|i| i as f64).collect();
    let src_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 8) };
    let isize = src.len() * 8;

    let mut cparams = BLUSC_BLOSC2_CPARAMS_DEFAULTS;
    cparams.typesize = 8;
    cparams.compcode = 0;
    cparams.filters[5] = 2; // BITSHUFFLE
    cparams.clevel = 5;
    let cctx = blusc_blosc2_create_cctx(cparams);

    let mut compressed = vec![0u8; isize + BLOSC2_MAX_OVERHEAD];
    let csize = blusc_blosc2_compress_ctx(&cctx, src_bytes, &mut compressed);
    assert!(csize > 0, "Bitshuffle compression failed");
    compressed.truncate(csize as usize);

    let dparams = BLUSC_BLOSC2_DPARAMS_DEFAULTS;
    let dctx = blusc_blosc2_create_dctx(dparams);
    let mut decompressed = vec![0u8; isize];
    let dsize = blusc_blosc2_decompress_ctx(&dctx, &compressed, &mut decompressed);
    assert_eq!(
        dsize as usize, isize,
        "Bitshuffle decompression size mismatch"
    );
    assert_eq!(src_bytes, &decompressed[..], "Bitshuffle roundtrip failed");
}

#[test]
fn test_codec_blosc_round_trip_snappy() {
    // blosc_compress_bytes(src len: 64, clevel: BloscCompressionLevel(4), shuffle_mode: NoShuffle, typesize: 0, compressor: Snappy, blocksize: 0, numinternalthreads: 1)

    let src: Vec<u8> = vec![0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31, 0];
    let numinternalthreads = 1;
    let blocksize = 0;
    let compressor = 3; // Snappy
    let shuffle_mode = 0; // NoShuffle
    let clevel = 4;
    let typesize = 0;
    let destsize = src.len() + BLOSC2_MAX_OVERHEAD as usize;
    let mut dest: Vec<u8> = vec![0; destsize];
    let destsize = {
        let mut cparams = BLUSC_BLOSC2_CPARAMS_DEFAULTS;
        cparams.typesize = typesize as i32;
        cparams.clevel = clevel.into();
        cparams.nthreads = numinternalthreads as i16;
        cparams.blocksize = blocksize as i32;
        cparams.compcode = compressor;
        cparams.filters[5] = shuffle_mode;
        let context = blusc_blosc2_create_cctx(cparams);

        blusc_blosc2_compress_ctx(&context, &src, &mut dest)
    };

    assert!(destsize > 0, "Compression failed");
}
