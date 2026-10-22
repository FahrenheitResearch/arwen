/// Ported from c-blosc2/tests/test_noinit.c
/// Tests that blosc compression/decompression works without explicit init/destroy.
/// In Rust, there is no global state requiring initialization, so this verifies
/// the API works correctly when called directly.
use blusc::api::{
    blosc2_compress as blusc_blosc2_compress, blosc2_decompress as blusc_blosc2_decompress,
};
use blusc::BLOSC2_MAX_OVERHEAD;

/// Test that compression produces smaller output than original for compressible data.
/// Ported from test_noinit.c: test_compress
#[test]
fn noinit_compress() {
    let typesize = 4;
    let num_elements = 100_000;
    let size = typesize * num_elements;

    // Fill with sequential 32-bit integers
    let mut src = vec![0u8; size];
    for i in 0..num_elements {
        let val = i as u32;
        src[i * 4..(i + 1) * 4].copy_from_slice(&val.to_ne_bytes());
    }

    let mut compressed = vec![0u8; size + BLOSC2_MAX_OVERHEAD];
    let csize = blusc_blosc2_compress(5, 1, typesize, &src, &mut compressed);
    assert!(csize > 0, "Compression failed: csize={}", csize);
    assert!(
        (csize as usize) < size,
        "Compressed size {} should be less than original size {}",
        csize,
        size
    );
}

/// Test full roundtrip (compress + decompress) without initialization.
/// Ported from test_noinit.c: test_compress_decompress
#[test]
fn noinit_compress_decompress() {
    let typesize = 4;
    let num_elements = 1_000_000;
    let size = typesize * num_elements;

    let mut src = vec![0u8; size];
    for i in 0..num_elements {
        let val = i as u32;
        src[i * 4..(i + 1) * 4].copy_from_slice(&val.to_ne_bytes());
    }

    let mut compressed = vec![0u8; size + BLOSC2_MAX_OVERHEAD];
    let csize = blusc_blosc2_compress(1, 1, typesize, &src, &mut compressed);
    assert!(csize > 0, "Compression failed");
    compressed.truncate(csize as usize);

    let mut decompressed = vec![0u8; size];
    let dsize = blusc_blosc2_decompress(&compressed, &mut decompressed);
    assert_eq!(
        dsize as usize, size,
        "Decompression size mismatch: got {}, expected {}",
        dsize, size
    );
    assert_eq!(src, decompressed, "Data mismatch after roundtrip");
}

/// Test multiple compression levels without initialization.
#[test]
fn noinit_multiple_clevels() {
    let typesize = 4;
    let num_elements = 10_000;
    let size = typesize * num_elements;

    let mut src = vec![0u8; size];
    for i in 0..num_elements {
        let val = i as u32;
        src[i * 4..(i + 1) * 4].copy_from_slice(&val.to_ne_bytes());
    }

    for clevel in 0..=9 {
        let mut compressed = vec![0u8; size + BLOSC2_MAX_OVERHEAD];
        let csize = blusc_blosc2_compress(clevel, 1, typesize, &src, &mut compressed);
        assert!(
            csize > 0,
            "Compression failed at clevel={}: csize={}",
            clevel,
            csize
        );
        compressed.truncate(csize as usize);

        let mut decompressed = vec![0u8; size];
        let dsize = blusc_blosc2_decompress(&compressed, &mut decompressed);
        assert_eq!(
            dsize as usize, size,
            "Decompression size mismatch at clevel={}",
            clevel
        );
        assert_eq!(
            src, decompressed,
            "Data mismatch after roundtrip at clevel={}",
            clevel
        );
    }
}
