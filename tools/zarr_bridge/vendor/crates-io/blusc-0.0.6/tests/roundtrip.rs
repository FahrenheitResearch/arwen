use blusc::api::{
    blosc2_compress as blusc_blosc2_compress,
    blosc2_decompress as blusc_blosc2_decompress,
};
use blusc::BLOSC2_MAX_OVERHEAD;

use blosc2_src::{
    blosc2_init as bound_blosc2_init,
    blosc2_compress as bound_blosc2_compress,
    blosc2_destroy as bound_blosc2_destroy,
};

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

struct TestCase {
    type_size: usize,
    num_elements: usize,
    clevel: i32,
    doshuffle: i32,
}

// Small buffers
#[test]
fn test_roundtrip_small_1() {
    run_roundtrip(&TestCase { type_size: 1, num_elements: 7, clevel: 5, doshuffle: 0 });
}

#[test]
fn test_roundtrip_small_2() {
    run_roundtrip(&TestCase { type_size: 2, num_elements: 7, clevel: 5, doshuffle: 0 });
}

#[test]
fn test_roundtrip_small_4() {
    run_roundtrip(&TestCase { type_size: 4, num_elements: 7, clevel: 5, doshuffle: 0 });
}

#[test]
fn test_roundtrip_small_8() {
    run_roundtrip(&TestCase { type_size: 8, num_elements: 7, clevel: 5, doshuffle: 0 });
}

// Larger buffers
#[test]
fn test_roundtrip_large_1() {
    run_roundtrip(&TestCase { type_size: 1, num_elements: 10000, clevel: 5, doshuffle: 0 });
}

#[test]
fn test_roundtrip_large_shuffle() {
    run_roundtrip(&TestCase { type_size: 4, num_elements: 10000, clevel: 5, doshuffle: 1 });
}

#[test]
fn test_roundtrip_large_bitshuffle() {
    run_roundtrip(&TestCase { type_size: 8, num_elements: 10000, clevel: 5, doshuffle: 2 });
}

// Different compression levels
#[test]
fn test_roundtrip_clevel_1() {
    run_roundtrip(&TestCase { type_size: 4, num_elements: 5000, clevel: 1, doshuffle: 1 });
}

#[test]
fn test_roundtrip_clevel_9() {
    run_roundtrip(&TestCase { type_size: 4, num_elements: 5000, clevel: 9, doshuffle: 1 });
}

// Odd sizes
#[test]
fn test_roundtrip_odd_3() {
    run_roundtrip(&TestCase { type_size: 3, num_elements: 1000, clevel: 5, doshuffle: 1 });
}

#[test]
fn test_roundtrip_odd_16() {
    run_roundtrip(&TestCase { type_size: 16, num_elements: 1000, clevel: 5, doshuffle: 1 });
}

#[test]
fn test_roundtrip_odd_33() {
    run_roundtrip(&TestCase { type_size: 33, num_elements: 100, clevel: 5, doshuffle: 0 });
}

#[test]
fn test_roundtrip_csv_case() {
    run_roundtrip(&TestCase { type_size: 1, num_elements: 702713, clevel: 5, doshuffle: 0 });
}

fn run_roundtrip(case: &TestCase) {
    let buffer_size = case.type_size * case.num_elements;
    let mut original = vec![0u8; buffer_size];
    
    // Fill with sequential data
    for j in 0..buffer_size {
        original[j] = (j % 255) as u8;
    }

    let dest_size = buffer_size + BLOSC2_MAX_OVERHEAD as usize;
    let mut intermediate = vec![0u8; dest_size];
    let mut result = vec![0u8; buffer_size];

    let csize = blusc_blosc2_compress(
        case.clevel,
        case.doshuffle,
        case.type_size,
        &original,
        &mut intermediate,
    );

    assert!(csize > 0, "Compression failed");
    intermediate.truncate(csize as usize);

    // Compare blusc intermediate to bound intermediate
    let mut bound_intermediate = vec![0; dest_size];
    let bound_csize = unsafe {
        bound_blosc2_compress(
            case.clevel,
            case.doshuffle,
            case.type_size as i32,
            original.as_ptr().cast(),
            original.len() as i32,
            bound_intermediate.as_mut_ptr().cast(),
            bound_intermediate.len() as i32,
        )
    };
    assert!(bound_csize > 0);
    bound_intermediate.truncate(bound_csize as usize);

    // Debug output for case 6
    if case.type_size == 8 && case.num_elements == 10000 && case.doshuffle == 2 {
        println!("\n=== DEBUG: Case 6 compression ===");
        println!("blusc compressed size: {}", csize);
        println!("bound compressed size: {}", bound_csize);
        
        let bound_blocksize = u32::from_le_bytes(bound_intermediate[8..12].try_into().unwrap());
        println!("bound blocksize: {}", bound_blocksize);
        
        println!("\nblusc first 100 compressed bytes (after header):");
        for i in (32..(csize.min(132) as usize)).step_by(10) {
            if i + 9 < csize as usize {
                println!("{:3}: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
                    i-32,
                    intermediate[i], intermediate[i+1], intermediate[i+2], intermediate[i+3], intermediate[i+4],
                    intermediate[i+5], intermediate[i+6], intermediate[i+7], intermediate[i+8], intermediate[i+9]);
            }
        }
        println!("\nbound first 100 compressed bytes (after header):");
        for i in (32..(bound_csize.min(132) as usize)).step_by(10) {
            if i + 9 < bound_csize as usize {
                println!("{:3}: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
                    i-32,
                    bound_intermediate[i], bound_intermediate[i+1], bound_intermediate[i+2], bound_intermediate[i+3], bound_intermediate[i+4],
                    bound_intermediate[i+5], bound_intermediate[i+6], bound_intermediate[i+7], bound_intermediate[i+8], bound_intermediate[i+9]);
            }
        }
    }
    
    assert_eq!(csize as usize, bound_csize as usize, "Compressed size mismatch between blusc and bound");
    assert_eq!(intermediate, bound_intermediate, "Compressed data mismatch between blusc and bound");

    let dsize = blusc_blosc2_decompress(
        &intermediate,
        &mut result,
    );

    assert_eq!(dsize, buffer_size as i32, "Decompression size mismatch");

    if original != result {
        for (i, (a, b)) in original.iter().zip(result.iter()).enumerate() {
            if a != b {
                println!("Mismatch at index {}: original={}, result={}", i, a, b);
                // Print surrounding values
                let start = if i > 10 { i - 10 } else { 0 };
                let end = if i + 10 < original.len() { i + 10 } else { original.len() };
                println!("Original context: {:?}", &original[start..end]);
                println!("Result context:   {:?}", &result[start..end]);
                break;
            }
        }
    }

    assert_eq!(original, result, "Data mismatch after roundtrip");
}
