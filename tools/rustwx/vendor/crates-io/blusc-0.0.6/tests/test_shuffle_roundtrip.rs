/// Ported from c-blosc2/tests/test_shuffle_roundtrip.c
/// Tests shuffle/unshuffle and bitshuffle/bitunshuffle roundtrip without compression,
/// verifying that data is preserved exactly through the filter pipeline.
use blusc::filters::{bitshuffle, bitunshuffle, shuffle, unshuffle};

/// Test shuffle → unshuffle roundtrip for a given typesize and number of elements.
fn run_shuffle_roundtrip(typesize: usize, num_elements: usize) {
    let blocksize = typesize * num_elements;

    // Fill with sequential data
    let mut original = vec![0u8; blocksize];
    for (i, byte) in original.iter_mut().enumerate() {
        *byte = i as u8;
    }

    let mut shuffled = vec![0u8; blocksize];
    let mut unshuffled = vec![0u8; blocksize];

    shuffle(typesize, blocksize, &original, &mut shuffled);
    unshuffle(typesize, blocksize, &shuffled, &mut unshuffled);

    assert_eq!(
        original, unshuffled,
        "Shuffle roundtrip failed for typesize={}, num_elements={}",
        typesize, num_elements
    );
}

/// Test bitshuffle → bitunshuffle roundtrip for a given typesize and number of elements.
fn run_bitshuffle_roundtrip(typesize: usize, num_elements: usize) {
    let blocksize = typesize * num_elements;

    let mut original = vec![0u8; blocksize];
    for (i, byte) in original.iter_mut().enumerate() {
        *byte = i as u8;
    }

    let mut shuffled = vec![0u8; blocksize];
    let mut unshuffled = vec![0u8; blocksize];

    bitshuffle(typesize, blocksize, &original, &mut shuffled).expect("bitshuffle failed");
    bitunshuffle(typesize, blocksize, &shuffled, &mut unshuffled).expect("bitunshuffle failed");

    assert_eq!(
        original, unshuffled,
        "Bitshuffle roundtrip failed for typesize={}, num_elements={}",
        typesize, num_elements
    );
}

/// Shuffle roundtrip across many type sizes.
/// Ported from test_shuffle_roundtrip.c test matrix.
#[test]
fn shuffle_roundtrip_various_typesizes() {
    let typesizes = [
        1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 22, 30, 32, 42, 48, 52, 53, 64, 80,
    ];
    let num_elements = 1024;

    for &ts in &typesizes {
        run_shuffle_roundtrip(ts, num_elements);
    }
}

/// Shuffle roundtrip with various element counts.
#[test]
fn shuffle_roundtrip_various_counts() {
    let typesize = 4;
    let counts = [
        1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 100, 128, 255, 256, 512, 1000, 1024, 4096,
    ];

    for &count in &counts {
        run_shuffle_roundtrip(typesize, count);
    }
}

/// Bitshuffle roundtrip across many type sizes.
#[test]
fn bitshuffle_roundtrip_various_typesizes() {
    let typesizes = [
        1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 22, 30, 32, 42, 48, 52, 53, 64, 80,
    ];
    let num_elements = 1024;

    for &ts in &typesizes {
        run_bitshuffle_roundtrip(ts, num_elements);
    }
}

/// Bitshuffle roundtrip with various element counts.
#[test]
fn bitshuffle_roundtrip_various_counts() {
    let typesize = 4;
    let counts = [
        1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 100, 128, 255, 256, 512, 1000, 1024, 4096,
    ];

    for &count in &counts {
        run_bitshuffle_roundtrip(typesize, count);
    }
}

/// Shuffle preserves data identity for typesize=1 (should be a no-op).
#[test]
fn shuffle_typesize1_is_identity() {
    let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
    let mut shuffled = vec![0u8; 256];

    shuffle(1, 256, &data, &mut shuffled);
    assert_eq!(data, shuffled, "Shuffle with typesize=1 should be identity");
}

/// Test that shuffle actually reorders bytes for typesize > 1.
#[test]
fn shuffle_actually_shuffles() {
    let typesize = 4;
    let num_elements = 4;
    let blocksize = typesize * num_elements;

    // 4 elements of 4 bytes each: [0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]
    let original: Vec<u8> = (0..blocksize as u8).collect();
    let mut shuffled = vec![0u8; blocksize];
    let mut unshuffled = vec![0u8; blocksize];

    shuffle(typesize, blocksize, &original, &mut shuffled);

    // After byte-shuffle with typesize=4:
    // Stream 0 (byte 0 of each element): [0, 4, 8, 12]
    // Stream 1 (byte 1 of each element): [1, 5, 9, 13]
    // Stream 2 (byte 2 of each element): [2, 6, 10, 14]
    // Stream 3 (byte 3 of each element): [3, 7, 11, 15]
    let expected_shuffled: Vec<u8> = vec![0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];
    assert_eq!(
        shuffled, expected_shuffled,
        "Shuffle did not produce expected byte interleaving"
    );

    unshuffle(typesize, blocksize, &shuffled, &mut unshuffled);
    assert_eq!(
        original, unshuffled,
        "Unshuffle did not recover original data"
    );
}

/// Test with large typesize that exceeds typical SIMD widths.
#[test]
fn shuffle_large_typesize() {
    for typesize in [128, 256] {
        let num_elements = 16;
        run_shuffle_roundtrip(typesize, num_elements);
    }
}

/// Test bitshuffle with large typesize.
#[test]
fn bitshuffle_large_typesize() {
    for typesize in [128, 256] {
        let num_elements = 16;
        run_bitshuffle_roundtrip(typesize, num_elements);
    }
}

/// Test shuffle/unshuffle with random-ish data patterns.
#[test]
fn shuffle_roundtrip_random_pattern() {
    let typesize = 8;
    let num_elements = 500;
    let blocksize = typesize * num_elements;

    // Pseudo-random fill using a simple LCG
    let mut original = vec![0u8; blocksize];
    let mut state: u32 = 12345;
    for byte in original.iter_mut() {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        *byte = (state >> 16) as u8;
    }

    let mut shuffled = vec![0u8; blocksize];
    let mut unshuffled = vec![0u8; blocksize];

    shuffle(typesize, blocksize, &original, &mut shuffled);
    unshuffle(typesize, blocksize, &shuffled, &mut unshuffled);

    assert_eq!(
        original, unshuffled,
        "Shuffle roundtrip with random data failed"
    );
}

/// Test bitshuffle/bitunshuffle with random-ish data patterns.
#[test]
fn bitshuffle_roundtrip_random_pattern() {
    let typesize = 8;
    let num_elements = 500;
    let blocksize = typesize * num_elements;

    let mut original = vec![0u8; blocksize];
    let mut state: u32 = 67890;
    for byte in original.iter_mut() {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        *byte = (state >> 16) as u8;
    }

    let mut shuffled = vec![0u8; blocksize];
    let mut unshuffled = vec![0u8; blocksize];

    bitshuffle(typesize, blocksize, &original, &mut shuffled).expect("bitshuffle failed");
    bitunshuffle(typesize, blocksize, &shuffled, &mut unshuffled).expect("bitunshuffle failed");

    assert_eq!(
        original, unshuffled,
        "Bitshuffle roundtrip with random data failed"
    );
}
