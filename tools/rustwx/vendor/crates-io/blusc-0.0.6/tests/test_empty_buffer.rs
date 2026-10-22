/// Ported from c-blosc2/tests/test_empty_buffer.c
/// Tests edge cases with empty, tiny, and zero-filled buffers.
use blosc2_src::{
    blosc2_compress as bound_blosc2_compress, blosc2_destroy as bound_blosc2_destroy,
    blosc2_init as bound_blosc2_init,
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

/// Compress an empty (0-byte) buffer.
/// Ported from test_empty_buffer.c
#[test]
fn empty_buffer_compress() {
    let src: Vec<u8> = vec![];
    let mut compressed = vec![0u8; BLOSC2_MAX_OVERHEAD];

    // Compress empty buffer with blusc
    let csize = blusc_blosc2_compress(3, 0, 1, &src, &mut compressed);
    assert!(
        csize > 0,
        "Empty buffer compression should produce header-only output, got csize={}",
        csize
    );

    // Compress with C reference and compare
    let mut c_compressed = vec![0u8; BLOSC2_MAX_OVERHEAD];
    let c_csize = unsafe {
        bound_blosc2_compress(
            3,
            0,
            1,
            src.as_ptr().cast(),
            0,
            c_compressed.as_mut_ptr().cast(),
            c_compressed.len() as i32,
        )
    };
    assert!(c_csize > 0, "C empty buffer compression failed");

    assert_eq!(
        csize as i32, c_csize,
        "Empty buffer compressed size mismatch: blusc={}, C={}",
        csize, c_csize
    );

    compressed.truncate(csize as usize);
    c_compressed.truncate(c_csize as usize);
    assert_eq!(
        compressed, c_compressed,
        "Empty buffer compressed bytes differ"
    );
}

/// Decompress empty buffer back to zero bytes.
#[test]
fn empty_buffer_roundtrip() {
    let src: Vec<u8> = vec![];
    let mut compressed = vec![0u8; BLOSC2_MAX_OVERHEAD];

    let csize = blusc_blosc2_compress(3, 0, 1, &src, &mut compressed);
    assert!(csize > 0);
    compressed.truncate(csize as usize);

    let mut decompressed = vec![0u8; 0];
    let dsize = blusc_blosc2_decompress(&compressed, &mut decompressed);
    assert_eq!(dsize, 0, "Decompressing empty buffer should return 0 bytes");
}

/// Test single-byte buffer.
#[test]
fn single_byte_buffer() {
    let src = vec![42u8];
    let mut compressed = vec![0u8; src.len() + BLOSC2_MAX_OVERHEAD];

    let csize = blusc_blosc2_compress(5, 0, 1, &src, &mut compressed);
    assert!(csize > 0, "Single byte compression failed");
    compressed.truncate(csize as usize);

    let mut decompressed = vec![0u8; 1];
    let dsize = blusc_blosc2_decompress(&compressed, &mut decompressed);
    assert_eq!(dsize, 1);
    assert_eq!(src, decompressed);
}

/// Test buffers of various small sizes from 0 to 64 bytes, comparing with C.
#[test]
fn small_buffer_sizes_cross_validate() {
    for size in [0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 63, 64] {
        let src: Vec<u8> = (0..size as u8).collect();
        let dest_size = src.len() + BLOSC2_MAX_OVERHEAD;

        let mut compressed_blusc = vec![0u8; dest_size];
        let csize_blusc = blusc_blosc2_compress(5, 0, 1, &src, &mut compressed_blusc);

        let mut compressed_c = vec![0u8; dest_size];
        let csize_c = unsafe {
            bound_blosc2_compress(
                5,
                0,
                1,
                src.as_ptr().cast(),
                size as i32,
                compressed_c.as_mut_ptr().cast(),
                compressed_c.len() as i32,
            )
        };

        assert!(
            csize_blusc > 0,
            "blusc compression failed for size={}",
            size
        );
        assert!(csize_c > 0, "C compression failed for size={}", size);
        assert_eq!(
            csize_blusc as i32, csize_c,
            "Compressed size mismatch for size={}: blusc={}, C={}",
            size, csize_blusc, csize_c
        );

        compressed_blusc.truncate(csize_blusc as usize);
        compressed_c.truncate(csize_c as usize);
        assert_eq!(
            compressed_blusc, compressed_c,
            "Compressed bytes mismatch for size={}",
            size
        );

        // Roundtrip
        if size > 0 {
            let mut decompressed = vec![0u8; size];
            let dsize = blusc_blosc2_decompress(&compressed_blusc, &mut decompressed);
            assert_eq!(
                dsize as usize, size,
                "Decompression size mismatch for size={}",
                size
            );
            assert_eq!(
                src, decompressed,
                "Data mismatch after roundtrip for size={}",
                size
            );
        }
    }
}
