//! Exact receipt digests without whole-array byte or boolean-mask copies.
//! Each SHA-256 stream stays sequential. Independent arrays own independent
//! result slots and may run on the preparation pool in parallel.

use std::panic::{catch_unwind, AssertUnwindSafe};
use sha2::{Digest, Sha256};
use crate::{ERR_DIMENSION, ERR_NULL, ERR_PANIC, OK};

const HASH_BLOCK: usize = 128 * 1024;
const MASK_BLOCK: usize = 8 * 1024;

#[repr(C)]
pub struct FingerprintInput {
    pub data: *const u8,
    pub data_bytes: usize,
    pub length: usize,
    pub kind: u32,
    /// Zero hashes bytes only; one additionally computes numeric metadata.
    pub metadata: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Fingerprint {
    pub digest: [u8; 32],
    pub nonzero_mask_digest: [u8; 32],
    pub nonzero_count: u64,
    pub minimum: f64,
    pub maximum: f64,
    /// NumPy keeps its own reduction order for NaNs and signed-zero ties.
    /// Bit zero requests its minimum; bit one requests its maximum.
    pub numpy_extrema: u32,
}

impl Default for Fingerprint {
    fn default() -> Self {
        Self { digest: [0; 32], nonzero_mask_digest: [0; 32], nonzero_count: 0,
            minimum: f64::INFINITY, maximum: f64::NEG_INFINITY, numpy_extrema: 0 }
    }
}

trait Number: Copy {
    fn value(self) -> f64;
    fn nonzero(self) -> bool;
}
macro_rules! integer {
    ($($kind:ty),+) => { $(impl Number for $kind {
        fn value(self) -> f64 { self as f64 }
        fn nonzero(self) -> bool { self != 0 }
    })+ };
}
integer!(i8, u8, i16, u16, i32, u32, i64, u64);
impl Number for f32 {
    fn value(self) -> f64 { self as f64 }
    fn nonzero(self) -> bool { self.to_bits() & 0x7fff_ffff != 0 }
}
impl Number for f64 {
    fn value(self) -> f64 { self }
    fn nonzero(self) -> bool { self.to_bits() & 0x7fff_ffff_ffff_ffff != 0 }
}

unsafe fn metadata<T: Number>(data: *const u8, length: usize, boolean: bool, result: &mut Fingerprint) {
    let mut packed = [0_u8; MASK_BLOCK];
    let mut mask_hash = Sha256::new();
    let mut saw_nan = false;
    for start in (0..length).step_by(MASK_BLOCK * 8) {
        let count = (length - start).min(MASK_BLOCK * 8);
        let packed_length = count.div_ceil(8);
        packed[..packed_length].fill(0);
        for offset in 0..count {
            // Contiguous NumPy buffers may still have an unaligned origin.
            let value = std::ptr::read_unaligned(data.cast::<T>().add(start + offset));
            let nonzero = value.nonzero();
            if nonzero {
                packed[offset / 8] |= 1 << (offset % 8);
                result.nonzero_count += 1;
            }
            let number = if boolean { u8::from(nonzero) as f64 } else { value.value() };
            if number.is_nan() { saw_nan = true; }
            else {
                result.minimum = result.minimum.min(number);
                result.maximum = result.maximum.max(number);
            }
        }
        mask_hash.update(&packed[..packed_length]);
    }
    result.nonzero_mask_digest.copy_from_slice(&mask_hash.finalize());
    if saw_nan { result.numpy_extrema = 3; }
    else {
        if result.minimum == 0.0 { result.numpy_extrema |= 1; }
        if result.maximum == 0.0 { result.numpy_extrema |= 2; }
    }
}

fn item_size(kind: u32) -> Option<usize> {
    Some(match kind { 1..=3 => 1, 4..=5 => 2, 6..=7 | 10 => 4, 8..=9 | 11 => 8,
        _ => return None })
}

unsafe fn fingerprint(input: &FingerprintInput) -> Fingerprint {
    let mut result = Fingerprint::default();
    let mut hash = Sha256::new();
    if input.data_bytes != 0 {
        for block in std::slice::from_raw_parts(input.data, input.data_bytes).chunks(HASH_BLOCK) {
            hash.update(block);
        }
    }
    result.digest.copy_from_slice(&hash.finalize());
    if input.metadata != 0 {
        match input.kind {
            1 => metadata::<u8>(input.data, input.length, true, &mut result),
            2 => metadata::<i8>(input.data, input.length, false, &mut result),
            3 => metadata::<u8>(input.data, input.length, false, &mut result),
            4 => metadata::<i16>(input.data, input.length, false, &mut result),
            5 => metadata::<u16>(input.data, input.length, false, &mut result),
            6 => metadata::<i32>(input.data, input.length, false, &mut result),
            7 => metadata::<u32>(input.data, input.length, false, &mut result),
            8 => metadata::<i64>(input.data, input.length, false, &mut result),
            9 => metadata::<u64>(input.data, input.length, false, &mut result),
            10 => metadata::<f32>(input.data, input.length, false, &mut result),
            11 => metadata::<f64>(input.data, input.length, false, &mut result),
            _ => unreachable!(),
        }
    }
    result
}

/// Hash immutable contiguous arrays in input order with bounded scratch.
///
/// # Safety
/// `inputs` and `results` contain `count` elements. Each data buffer stays
/// readable and immutable for `data_bytes` bytes throughout this call. Result
/// slots are disjoint from the input buffers and from each other. Kind codes
/// describe native-endian primitive numeric arrays when metadata is requested.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_preparation_fingerprints(
    inputs: *const FingerprintInput, count: usize, workers: usize, results: *mut Fingerprint,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if workers == 0 { return ERR_DIMENSION; }
        if count == 0 { return OK; }
        if inputs.is_null() || results.is_null() { return ERR_NULL; }
        for index in 0..count {
            let input = &*inputs.add(index);
            if input.data.is_null() && input.data_bytes != 0 { return ERR_NULL; }
            if input.metadata > 1 { return ERR_DIMENSION; }
            if input.metadata != 0 {
                let Some(size) = item_size(input.kind) else { return ERR_DIMENSION; };
                if input.length == 0 || input.length.checked_mul(size) != Some(input.data_bytes) {
                    return ERR_DIMENSION;
                }
            }
        }
        let inputs = inputs as usize;
        let results = results as usize;
        crate::parallel::run_ranges(count, workers, |start, stop| {
            for index in start..stop {
                *(results as *mut Fingerprint).add(index) = fingerprint(
                    &*(inputs as *const FingerprintInput).add(index));
            }
        });
        OK
    })).unwrap_or(ERR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn little_mask_padding_and_special_values() {
        let values = [0.0_f32, -0.0, 1.0, f32::NAN, f32::INFINITY,
            f32::NEG_INFINITY, f32::from_bits(1), 0.0, -2.0];
        let input = FingerprintInput { data: values.as_ptr().cast(), data_bytes: values.len() * 4,
            length: values.len(), kind: 10, metadata: 1 };
        let mut result = Fingerprint::default();
        assert_eq!(unsafe { gpuwm_preparation_fingerprints(&input, 1, 8, &mut result) }, OK);
        assert_eq!(result.nonzero_count, 6);
        assert_eq!(result.numpy_extrema, 3);
        assert_eq!(result.nonzero_mask_digest.as_slice(), Sha256::digest([0x7c, 0x01]).as_slice());
        assert_eq!(result.digest.as_slice(), Sha256::digest(unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4)
        }).as_slice());
    }

    #[test]
    fn unaligned_input_and_mask_block_boundary() {
        let count = MASK_BLOCK * 8 + 11;
        let mut storage = vec![0_u8; count * 4 + 1];
        for index in [0, MASK_BLOCK * 8 - 1, MASK_BLOCK * 8, count - 1] {
            storage[1 + index * 4..1 + index * 4 + 4].copy_from_slice(&1.0_f32.to_ne_bytes());
        }
        let input = FingerprintInput { data: unsafe { storage.as_ptr().add(1) },
            data_bytes: count * 4, length: count, kind: 10, metadata: 1 };
        let mut result = Fingerprint::default();
        assert_eq!(unsafe { gpuwm_preparation_fingerprints(&input, 1, 1, &mut result) }, OK);
        let mut mask = vec![0_u8; count.div_ceil(8)];
        for index in [0, MASK_BLOCK * 8 - 1, MASK_BLOCK * 8, count - 1] { mask[index / 8] |= 1 << (index % 8); }
        assert_eq!(result.nonzero_count, 4);
        assert_eq!(result.nonzero_mask_digest.as_slice(), Sha256::digest(mask).as_slice());
        assert_eq!(result.minimum, 0.0);
        assert_eq!(result.maximum, 1.0);
    }
}
