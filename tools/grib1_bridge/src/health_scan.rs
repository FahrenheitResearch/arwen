//! Read-only host health scans with deterministic descriptor/index attribution.
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use crate::{parallel, ERR_DIMENSION, ERR_NULL, ERR_PANIC, OK};

const INDEX_BITS: u32 = 48;
const INDEX_LIMIT: usize = 1usize << INDEX_BITS;
const CHUNK: usize = 1 << 20;
const LOWER: u32 = 1;
const UPPER: u32 = 2;
const STRICT_LOWER: u32 = 4;

#[repr(C)]
pub struct HealthDescriptor {
    pub values: *const u8,
    pub auxiliary: *const u8,
    pub size: usize,
    pub plane: usize,
    pub value_type: u32,
    pub auxiliary_type: u32,
    pub result_type: u32,
    pub auxiliary_mode: u32,
    pub flags: u32,
    pub status_bit: u32,
    pub lower: f64,
    pub upper: f64,
}

#[repr(C)]
pub struct HealthResult {
    pub status_bits: u32,
    pub first_field: usize,
    pub first_index: usize,
    pub first_value: f64,
}

unsafe fn read(pointer: *const u8, dtype: u32, index: usize) -> f64 {
    if dtype == 1 { std::ptr::read_unaligned((pointer as *const f32).add(index)) as f64 }
    else { std::ptr::read_unaligned((pointer as *const f64).add(index)) }
}

unsafe fn value(desc: &HealthDescriptor, index: usize) -> f64 {
    let original = read(desc.values, desc.value_type, index);
    if desc.auxiliary_mode == 0 { return original; }
    let auxiliary_index = if desc.auxiliary_mode == 1 { index }
                          else { index / desc.plane };
    let auxiliary = read(desc.auxiliary, desc.auxiliary_type, auxiliary_index);
    // NumPy's float32 addition rounds before the finite/bound checks.
    if desc.result_type == 1 { ((original as f32) + (auxiliary as f32)) as f64 }
    else { original + auxiliary }
}

fn bad(desc: &HealthDescriptor, value: f64) -> bool {
    if !value.is_finite() { return true; }
    let lower = if desc.result_type == 1 { (desc.lower as f32) as f64 } else { desc.lower };
    let upper = if desc.result_type == 1 { (desc.upper as f32) as f64 } else { desc.upper };
    ((desc.flags & LOWER != 0) && if desc.flags & STRICT_LOWER != 0 {
        value <= lower
    } else { value < lower }) || ((desc.flags & UPPER != 0) && value > upper)
}

/// Scan immutable dense host arrays without creating masks or derived arrays.
///
/// # Safety
/// Descriptors and inputs remain readable and unchanged until return. Each
/// values buffer holds `size` native-endian f32/f64 elements. Direct auxiliaries
/// hold `size` elements; level auxiliaries hold `ceil(size/plane)` elements.
/// The writable result slot overlaps no input. Types are 1=f32 and 2=f64.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_validate_health_fields(
    descriptors: *const HealthDescriptor, count: usize, workers: usize,
    result: *mut HealthResult,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if descriptors.is_null() || result.is_null() { return ERR_NULL; }
        if workers == 0 || count >= 1 << (64 - INDEX_BITS) { return ERR_DIMENSION; }
        let descriptors = std::slice::from_raw_parts(descriptors, count);
        let mut starts = Vec::with_capacity(count + 1);
        starts.push(0usize);
        for desc in descriptors {
            if desc.size >= INDEX_LIMIT || !(1..=2).contains(&desc.value_type)
                || !(1..=2).contains(&desc.result_type) || desc.auxiliary_mode > 2
                || desc.flags & !(LOWER | UPPER | STRICT_LOWER) != 0
                || (desc.size > 0 && desc.values.is_null())
                || (desc.auxiliary_mode != 0 && (!(1..=2).contains(&desc.auxiliary_type)
                    || desc.auxiliary.is_null() || (desc.auxiliary_mode == 2 && desc.plane == 0))) {
                return ERR_DIMENSION;
            }
            let chunks = desc.size / CHUNK + usize::from(desc.size % CHUNK != 0);
            match starts.last().unwrap().checked_add(chunks) {
                Some(next) => starts.push(next), None => return ERR_DIMENSION,
            }
        }
        let status = AtomicU32::new(0);
        let first = AtomicU64::new(u64::MAX);
        let addresses = descriptors.as_ptr() as usize;
        parallel::run_ranges(*starts.last().unwrap(), workers, |start, stop| {
            let descriptors = std::slice::from_raw_parts(addresses as *const HealthDescriptor, count);
            for chunk in start..stop {
                let field = starts.partition_point(|&offset| offset <= chunk) - 1;
                let desc = &descriptors[field];
                let begin = (chunk - starts[field]) * CHUNK;
                let end = desc.size.min(begin + CHUNK);
                for index in begin..end {
                    if bad(desc, value(desc, index)) {
                        status.fetch_or(desc.status_bit, Ordering::Relaxed);
                        first.fetch_min(((field as u64) << INDEX_BITS) | index as u64,
                                        Ordering::Relaxed);
                        break;
                    }
                }
            }
        });
        let encoded = first.load(Ordering::Relaxed);
        let (field, index, first_value) = if encoded == u64::MAX {
            (usize::MAX, usize::MAX, 0.0)
        } else {
            let field = (encoded >> INDEX_BITS) as usize;
            let index = (encoded & ((1u64 << INDEX_BITS) - 1)) as usize;
            (field, index, value(&descriptors[field], index))
        };
        *result = HealthResult { status_bits: status.load(Ordering::Relaxed),
                                first_field: field, first_index: index, first_value };
        OK
    })).unwrap_or(ERR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earliest_field_and_index_and_all_status_classes_are_stable() {
        let first = [1.0f32, -0.0, 2.0, f32::NAN];
        let second = [f64::INFINITY];
        let descriptors = [HealthDescriptor { values: first.as_ptr() as *const u8,
            auxiliary: std::ptr::null(), size: 4, plane: 0, value_type: 1,
            auxiliary_type: 0, result_type: 1, auxiliary_mode: 0,
            flags: LOWER | STRICT_LOWER, status_bit: 2, lower: 0.0, upper: 0.0 },
            HealthDescriptor { values: second.as_ptr() as *const u8,
                auxiliary: std::ptr::null(), size: 1, plane: 0, value_type: 2,
                auxiliary_type: 0, result_type: 2, auxiliary_mode: 0,
                flags: 0, status_bit: 4, lower: 0.0, upper: 0.0 }];
        for workers in [1, 4, 32] {
            let mut result = HealthResult { status_bits: 0, first_field: 0,
                                           first_index: 0, first_value: 0.0 };
            assert_eq!(unsafe { gpuwm_validate_health_fields(descriptors.as_ptr(), 2,
                workers, &mut result) }, OK);
            assert_eq!((result.status_bits, result.first_field, result.first_index), (6, 0, 1));
            assert_eq!(result.first_value.to_bits(), (-0.0f64).to_bits());
        }
    }
}
