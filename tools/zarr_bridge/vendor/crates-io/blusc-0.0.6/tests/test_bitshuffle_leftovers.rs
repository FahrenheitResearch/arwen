/// Ported from c-blosc2/tests/test_bitshuffle_leftovers.c
/// Tests bitshuffle compression with non-aligned buffer sizes (not evenly
/// divisible by 8 or typesize), verifying correct handling of leftover bytes.
use blosc2_src::{
    blosc2_compress as bound_blosc2_compress, blosc2_destroy as bound_blosc2_destroy,
    blosc2_init as bound_blosc2_init,
};
use blusc::api::{
    blosc2_compress as blusc_blosc2_compress, blosc2_compress_ctx as blusc_blosc2_compress_ctx,
    blosc2_create_cctx as blusc_blosc2_create_cctx, blosc2_decompress as blusc_blosc2_decompress,
    BLOSC2_CPARAMS_DEFAULTS as BLUSC_BLOSC2_CPARAMS_DEFAULTS,
};
use blusc::{BLOSC2_MAX_FILTERS, BLOSC2_MAX_OVERHEAD, BLOSC_BITSHUFFLE, BLOSC_BLOSCLZ};

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

/// Fill buffer with sequential 32-bit integers, then fill leftover bytes.
/// Matches the C test's data initialization pattern.
fn fill_buffer(buf: &mut [u8], typesize: usize) {
    let num_ints = buf.len() / 4;
    for i in 0..num_ints {
        let val = i as u32;
        let offset = i * 4;
        if offset + 4 <= buf.len() {
            buf[offset..offset + 4].copy_from_slice(&val.to_ne_bytes());
        }
    }
    // Fill remaining bytes
    let leftover_start = num_ints * 4;
    for i in leftover_start..buf.len() {
        buf[i] = i as u8;
    }
    let _ = typesize; // typesize used by C for alignment, not needed in fill
}

/// Bitshuffle roundtrip with typesize=4, buffer size not divisible by 8.
/// Ported from test_bitshuffle_leftovers.c: test_roundtrip_bitshuffle4
#[test]
fn bitshuffle_leftovers_typesize4() {
    let buffer_size = 641_092; // Not divisible by 8 or 4
    let typesize = 4;

    let mut src = vec![0u8; buffer_size];
    fill_buffer(&mut src, typesize);

    let dest_size = buffer_size + BLOSC2_MAX_OVERHEAD;
    let mut compressed = vec![0u8; dest_size];
    let csize = blusc_blosc2_compress(9, BLOSC_BITSHUFFLE as i32, typesize, &src, &mut compressed);
    assert!(
        csize > 0,
        "Bitshuffle typesize=4 compression failed: csize={}",
        csize
    );
    compressed.truncate(csize as usize);

    let mut decompressed = vec![0u8; buffer_size];
    let dsize = blusc_blosc2_decompress(&compressed, &mut decompressed);
    assert_eq!(
        dsize as usize, buffer_size,
        "Bitshuffle typesize=4 decompression size mismatch"
    );
    assert_eq!(
        src, decompressed,
        "Bitshuffle typesize=4 roundtrip data mismatch"
    );
}

/// Bitshuffle roundtrip with typesize=8, buffer size not divisible by 8.
/// Ported from test_bitshuffle_leftovers.c: test_roundtrip_bitshuffle8
#[test]
fn bitshuffle_leftovers_typesize8() {
    let buffer_size = 641_092;
    let typesize = 8;

    let mut src = vec![0u8; buffer_size];
    fill_buffer(&mut src, typesize);

    let dest_size = buffer_size + BLOSC2_MAX_OVERHEAD;
    let mut compressed = vec![0u8; dest_size];
    let csize = blusc_blosc2_compress(9, BLOSC_BITSHUFFLE as i32, typesize, &src, &mut compressed);
    assert!(
        csize > 0,
        "Bitshuffle typesize=8 compression failed: csize={}",
        csize
    );
    compressed.truncate(csize as usize);

    let mut decompressed = vec![0u8; buffer_size];
    let dsize = blusc_blosc2_decompress(&compressed, &mut decompressed);
    assert_eq!(
        dsize as usize, buffer_size,
        "Bitshuffle typesize=8 decompression size mismatch"
    );
    assert_eq!(
        src, decompressed,
        "Bitshuffle typesize=8 roundtrip data mismatch"
    );
}

/// Test bitshuffle with various non-aligned buffer sizes.
#[test]
fn bitshuffle_leftovers_various_sizes() {
    // Sizes chosen to have different remainder patterns mod typesize and mod 8
    let sizes = [
        100, 101, 255, 256, 257, 500, 511, 512, 513, 1000, 1023, 1024, 1025, 4095, 4096, 4097,
        10001, 65535,
    ];

    for &size in &sizes {
        for typesize in [1, 2, 4, 8] {
            let mut src = vec![0u8; size];
            for (i, byte) in src.iter_mut().enumerate() {
                *byte = (i % 251) as u8; // Use prime modulus for variety
            }

            let dest_size = size + BLOSC2_MAX_OVERHEAD;
            let mut compressed = vec![0u8; dest_size];
            let csize =
                blusc_blosc2_compress(5, BLOSC_BITSHUFFLE as i32, typesize, &src, &mut compressed);
            assert!(
                csize > 0,
                "Compression failed for size={}, typesize={}",
                size,
                typesize
            );
            compressed.truncate(csize as usize);

            let mut decompressed = vec![0u8; size];
            let dsize = blusc_blosc2_decompress(&compressed, &mut decompressed);
            assert_eq!(
                dsize as usize, size,
                "Decompression size mismatch for size={}, typesize={}",
                size, typesize
            );
            assert_eq!(
                src, decompressed,
                "Data mismatch for size={}, typesize={}",
                size, typesize
            );
        }
    }
}

/// Cross-validate bitshuffle with non-aligned sizes against C reference.
#[test]
fn bitshuffle_leftovers_cross_validate() {
    let test_cases = [(641_092, 4usize, 9), (641_092, 8, 9)];

    for &(size, typesize, clevel) in &test_cases {
        let mut src = vec![0u8; size];
        fill_buffer(&mut src, typesize);

        let dest_size = size + BLOSC2_MAX_OVERHEAD;
        let mut compressed_blusc = vec![0u8; dest_size];
        let csize_blusc = blusc_blosc2_compress(
            clevel,
            BLOSC_BITSHUFFLE as i32,
            typesize,
            &src,
            &mut compressed_blusc,
        );

        let mut compressed_c = vec![0u8; dest_size];
        let csize_c = unsafe {
            bound_blosc2_compress(
                clevel,
                BLOSC_BITSHUFFLE as i32,
                typesize as i32,
                src.as_ptr().cast(),
                src.len() as i32,
                compressed_c.as_mut_ptr().cast(),
                compressed_c.len() as i32,
            )
        };

        assert!(
            csize_blusc > 0,
            "blusc failed for size={}, ts={}",
            size,
            typesize
        );
        assert!(csize_c > 0, "C failed for size={}, ts={}", size, typesize);
        assert_eq!(
            csize_blusc as i32, csize_c,
            "Compressed size mismatch for size={}, typesize={}: blusc={}, C={}",
            size, typesize, csize_blusc, csize_c
        );

        compressed_blusc.truncate(csize_blusc as usize);
        compressed_c.truncate(csize_c as usize);
        assert_eq!(
            compressed_blusc, compressed_c,
            "Compressed bytes mismatch for size={}, typesize={}",
            size, typesize
        );
    }
}

/// Test bitshuffle via context API with non-aligned sizes.
#[test]
fn bitshuffle_leftovers_context_api() {
    let buffer_size = 641_092;

    for typesize in [4usize, 8] {
        let mut src = vec![0u8; buffer_size];
        fill_buffer(&mut src, typesize);

        let mut cparams = BLUSC_BLOSC2_CPARAMS_DEFAULTS;
        cparams.clevel = 9;
        cparams.typesize = typesize as i32;
        cparams.compcode = BLOSC_BLOSCLZ;
        cparams.filters[BLOSC2_MAX_FILTERS as usize - 1] = BLOSC_BITSHUFFLE;
        let cctx = blusc_blosc2_create_cctx(cparams);

        let dest_size = buffer_size + BLOSC2_MAX_OVERHEAD;
        let mut compressed = vec![0u8; dest_size];
        let csize = blusc_blosc2_compress_ctx(&cctx, &src, &mut compressed);
        assert!(
            csize > 0,
            "Context bitshuffle compression failed for typesize={}",
            typesize
        );
        compressed.truncate(csize as usize);

        let mut decompressed = vec![0u8; buffer_size];
        let dsize = blusc_blosc2_decompress(&compressed, &mut decompressed);
        assert_eq!(
            dsize as usize, buffer_size,
            "Context bitshuffle decompression size mismatch for typesize={}",
            typesize
        );
        assert_eq!(
            src, decompressed,
            "Context bitshuffle roundtrip mismatch for typesize={}",
            typesize
        );
    }
}
