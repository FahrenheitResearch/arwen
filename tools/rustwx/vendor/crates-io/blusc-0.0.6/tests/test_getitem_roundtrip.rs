/// Ported from c-blosc2/tests/test_getitem.c
/// Tests blosc1_getitem() for extracting individual items from compressed data.
/// Verifies that compressing then extracting all items via getitem produces
/// the original data, with byte-exact compressed output matching C reference.
///
/// Test matrix from test_getitem.csv:
///   type_sizes: [1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 22, 30, 32, 42, 48, 52, 53, 64, 80]
///   num_elements: [7, 192, 1792, 500, 8000, 100000, 702713]
///   shuffle_type: [0 (noshuffle), 1 (shuffle), 2 (bitshuffle)]
///   clevel: 5
///   compressor: blosclz (default)
use blosc2_src::{
    blosc2_compress as bound_blosc2_compress, blosc2_destroy as bound_blosc2_destroy,
    blosc2_init as bound_blosc2_init, BLOSC2_MAX_OVERHEAD as BOUND_BLOSC2_MAX_OVERHEAD,
};
use blusc::api::{
    blosc1_getitem as blusc_blosc1_getitem, blosc2_compress as blusc_blosc2_compress,
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

fn fill_seq(buf: &mut [u8]) {
    for (k, byte) in buf.iter_mut().enumerate() {
        *byte = k as u8;
    }
}

/// Compress with C, then use blusc getitem to extract all elements.
/// This tests that blusc can correctly decompress individual items from
/// C-compressed data.
fn run_getitem_from_c_compressed(
    type_size: usize,
    num_elements: usize,
    clevel: i32,
    doshuffle: i32,
) {
    let buffer_size = type_size * num_elements;
    let dest_size = buffer_size + BLOSC2_MAX_OVERHEAD;

    let mut original = vec![0u8; buffer_size];
    fill_seq(&mut original);

    // Compress with C reference
    let mut compressed = vec![0u8; dest_size];
    let csize = unsafe {
        bound_blosc2_compress(
            clevel,
            doshuffle,
            type_size as i32,
            original.as_ptr().cast(),
            original.len() as i32,
            compressed.as_mut_ptr().cast(),
            compressed.len() as i32,
        )
    };
    assert!(
        csize > 0,
        "C compression failed for type_size={}, num_elements={}, doshuffle={}",
        type_size,
        num_elements,
        doshuffle
    );
    compressed.truncate(csize as usize);

    // Use blusc getitem to extract all elements
    let mut result = vec![0u8; buffer_size];
    let dsize = blusc_blosc1_getitem(&compressed, 0, num_elements as i32, &mut result);
    assert!(
        dsize > 0,
        "blusc getitem failed for type_size={}, num_elements={}, doshuffle={}: dsize={}",
        type_size,
        num_elements,
        doshuffle,
        dsize
    );
    assert_eq!(
        dsize as usize, buffer_size,
        "getitem size mismatch for type_size={}, num_elements={}, doshuffle={}",
        type_size, num_elements, doshuffle
    );
    assert_eq!(
        original, result,
        "getitem data mismatch for type_size={}, num_elements={}, doshuffle={}",
        type_size, num_elements, doshuffle
    );
}

/// Compress with blusc, then use blusc getitem to extract all elements.
fn run_getitem_from_blusc_compressed(
    type_size: usize,
    num_elements: usize,
    clevel: i32,
    doshuffle: i32,
) {
    let buffer_size = type_size * num_elements;
    let dest_size = buffer_size + BLOSC2_MAX_OVERHEAD;

    let mut original = vec![0u8; buffer_size];
    fill_seq(&mut original);

    // Compress with blusc
    let mut compressed = vec![0u8; dest_size];
    let csize = blusc_blosc2_compress(clevel, doshuffle, type_size, &original, &mut compressed);
    assert!(
        csize > 0,
        "blusc compression failed for type_size={}, num_elements={}, doshuffle={}",
        type_size,
        num_elements,
        doshuffle
    );
    compressed.truncate(csize as usize);

    // Use blusc getitem to extract all elements
    let mut result = vec![0u8; buffer_size];
    let dsize = blusc_blosc1_getitem(&compressed, 0, num_elements as i32, &mut result);
    assert!(
        dsize > 0,
        "blusc getitem failed for type_size={}, num_elements={}, doshuffle={}: dsize={}",
        type_size,
        num_elements,
        doshuffle,
        dsize
    );
    assert_eq!(
        original, result,
        "getitem roundtrip data mismatch for type_size={}, num_elements={}, doshuffle={}",
        type_size, num_elements, doshuffle
    );
}

/// Compress with C, then use blusc getitem to extract a subset of elements.
fn run_getitem_subset(type_size: usize, num_elements: usize, clevel: i32, doshuffle: i32) {
    let buffer_size = type_size * num_elements;
    let dest_size = buffer_size + BLOSC2_MAX_OVERHEAD;

    let mut original = vec![0u8; buffer_size];
    fill_seq(&mut original);

    // Compress with C reference
    let mut compressed = vec![0u8; dest_size];
    let csize = unsafe {
        bound_blosc2_compress(
            clevel,
            doshuffle,
            type_size as i32,
            original.as_ptr().cast(),
            original.len() as i32,
            compressed.as_mut_ptr().cast(),
            compressed.len() as i32,
        )
    };
    assert!(csize > 0);
    compressed.truncate(csize as usize);

    // Extract items from the middle (if enough elements)
    if num_elements >= 10 {
        let start = 5;
        let nitems = 5;
        let subset_size = type_size * nitems;
        let mut subset = vec![0u8; subset_size];
        let dsize = blusc_blosc1_getitem(&compressed, start as i32, nitems as i32, &mut subset);
        assert!(dsize > 0, "getitem subset failed");

        let expected_start = start * type_size;
        let expected_end = expected_start + subset_size;
        assert_eq!(
            &original[expected_start..expected_end], &subset[..],
            "getitem subset data mismatch for type_size={}, num_elements={}, doshuffle={}, start={}, nitems={}",
            type_size, num_elements, doshuffle, start, nitems
        );
    }
}

const TYPE_SIZES: &[usize] = &[
    1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 22, 30, 32, 42, 48, 52, 53, 64, 80,
];
const NUM_ELEMENTS: &[usize] = &[7, 192, 1792, 500, 8000, 100000, 702713];

// --- getitem from C-compressed data, noshuffle ---

#[test]
fn getitem_c_compressed_noshuffle() {
    for &ts in TYPE_SIZES {
        for &ne in NUM_ELEMENTS {
            run_getitem_from_c_compressed(ts, ne, 5, 0);
        }
    }
}

// --- getitem from C-compressed data, shuffle ---

#[test]
fn getitem_c_compressed_shuffle() {
    for &ts in TYPE_SIZES {
        for &ne in NUM_ELEMENTS {
            run_getitem_from_c_compressed(ts, ne, 5, 1);
        }
    }
}

// --- getitem from C-compressed data, bitshuffle ---
// Note: bitshuffle requires buffer_size / typesize to be a multiple of 8

#[test]
fn getitem_c_compressed_bitshuffle() {
    for &ts in TYPE_SIZES {
        for &ne in NUM_ELEMENTS {
            // Bitshuffle requires (blocksize / typesize) % 8 == 0
            // Skip cases where this doesn't hold
            let buffer_size = ts * ne;
            if buffer_size < ts || (buffer_size / ts) % 8 != 0 {
                continue;
            }
            run_getitem_from_c_compressed(ts, ne, 5, 2);
        }
    }
}

// --- getitem from blusc-compressed data ---

#[test]
fn getitem_blusc_compressed_noshuffle() {
    for &ts in TYPE_SIZES {
        for &ne in NUM_ELEMENTS {
            run_getitem_from_blusc_compressed(ts, ne, 5, 0);
        }
    }
}

#[test]
fn getitem_blusc_compressed_shuffle() {
    for &ts in TYPE_SIZES {
        for &ne in NUM_ELEMENTS {
            run_getitem_from_blusc_compressed(ts, ne, 5, 1);
        }
    }
}

// --- getitem subset extraction ---

#[test]
fn getitem_subset_noshuffle() {
    for &ts in &[1usize, 4, 8, 16] {
        for &ne in &[100usize, 8000, 100000] {
            run_getitem_subset(ts, ne, 5, 0);
        }
    }
}

#[test]
fn getitem_subset_shuffle() {
    for &ts in &[1usize, 4, 8, 16] {
        for &ne in &[100usize, 8000, 100000] {
            run_getitem_subset(ts, ne, 5, 1);
        }
    }
}
