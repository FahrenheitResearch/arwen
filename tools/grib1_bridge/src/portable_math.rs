//! Portable elementwise transcendentals for the host preparation.
//!
//! NumPy's float64 and float32 `exp`, `log`, `power`, `arcsin`, `tan`,
//! `arctan` and friends take the C library on most hosts but their own
//! vector loops on an AVX-512 Linux host, and the C library itself differs
//! between glibc releases and between glibc and the MSVC runtime.  A
//! prepared state built from them therefore changes in its last bits with
//! the machine.  These entries take the vendored `libm` crate, a pure Rust
//! port of musl's libm: the same instructions on every x86-64 host
//! whatever its vector unit, with no call into the C library.  Its only
//! hardware routines (`sqrt` and `fma` on SSE2) are correctly rounded by
//! IEEE 754, so they cannot differ between CPUs either, and Rust never
//! contracts a multiply and an add into an `fma` on its own.
//!
//! Every entry is elementwise and splits the flattened arrays into fixed
//! contiguous ranges, so the worker count never changes an element.
//! Output may be the input buffer itself (element `i` is read before it is
//! written); any other overlap is the caller's error.

use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::{worker_ranges, ERR_DIMENSION, ERR_NULL, ERR_PANIC, OK};

/// Unary operation codes, shared with `gpuwm/core/portable_math.py`.
pub const UNARY_EXP: u32 = 0;
pub const UNARY_LOG: u32 = 1;
pub const UNARY_LOG1P: u32 = 2;
pub const UNARY_SIN: u32 = 3;
pub const UNARY_COS: u32 = 4;
pub const UNARY_TAN: u32 = 5;
pub const UNARY_ASIN: u32 = 6;
pub const UNARY_ATAN: u32 = 7;
pub const UNARY_LOG10: u32 = 8;
pub const UNARY_ACOS: u32 = 9;
/// Binary operation codes.
pub const BINARY_POW: u32 = 0;
pub const BINARY_ATAN2: u32 = 1;

/// Below this many elements a call stays on the calling thread: spawning
/// costs more than the work.
const MIN_ELEMENTS_PER_WORKER: usize = 16_384;

fn unary_f64(op: u32) -> Option<fn(f64) -> f64> {
    Some(match op {
        UNARY_EXP => libm::exp,
        UNARY_LOG => libm::log,
        UNARY_LOG1P => libm::log1p,
        UNARY_SIN => libm::sin,
        UNARY_COS => libm::cos,
        UNARY_TAN => libm::tan,
        UNARY_ASIN => libm::asin,
        UNARY_ATAN => libm::atan,
        UNARY_LOG10 => libm::log10,
        UNARY_ACOS => libm::acos,
        _ => return None,
    })
}

fn unary_f32(op: u32) -> Option<fn(f32) -> f32> {
    Some(match op {
        UNARY_EXP => libm::expf,
        UNARY_LOG => libm::logf,
        UNARY_LOG1P => libm::log1pf,
        UNARY_SIN => libm::sinf,
        UNARY_COS => libm::cosf,
        UNARY_TAN => libm::tanf,
        UNARY_ASIN => libm::asinf,
        UNARY_ATAN => libm::atanf,
        UNARY_LOG10 => libm::log10f,
        UNARY_ACOS => libm::acosf,
        _ => return None,
    })
}

fn binary_f64(op: u32) -> Option<fn(f64, f64) -> f64> {
    Some(match op {
        BINARY_POW => libm::pow,
        BINARY_ATAN2 => libm::atan2,
        _ => return None,
    })
}

fn binary_f32(op: u32) -> Option<fn(f32, f32) -> f32> {
    Some(match op {
        BINARY_POW => libm::powf,
        BINARY_ATAN2 => libm::atan2f,
        _ => return None,
    })
}

fn workers_for(length: usize, workers: usize) -> usize {
    workers.min(length.div_ceil(MIN_ELEMENTS_PER_WORKER)).max(1)
}

/// Run `body(start, stop)` over fixed contiguous ranges of `0..length`.
fn run_ranges(length: usize, workers: usize, body: &(dyn Fn(usize, usize) + Sync)) {
    let workers = workers_for(length, workers);
    if workers == 1 {
        body(0, length);
        return;
    }
    std::thread::scope(|scope| {
        for (start, stop) in worker_ranges(length, workers) {
            scope.spawn(move || body(start, stop));
        }
    });
}

unsafe fn unary<T: Copy + Send + Sync>(
    function: fn(T) -> T,
    input: *const T,
    output: *mut T,
    length: usize,
    workers: usize,
) {
    let input = input as usize;
    let output = output as usize;
    run_ranges(length, workers, &|start, stop| {
        let source = input as *const T;
        let target = output as *mut T;
        for index in start..stop {
            // SAFETY: the caller guarantees `length` readable and writable
            // elements; each index is read before it is written.
            unsafe { *target.add(index) = function(*source.add(index)) };
        }
    });
}

/// `left_length` and `right_length` are each `length` or 1 (a scalar
/// operand broadcast over the other).
unsafe fn binary<T: Copy + Send + Sync>(
    function: fn(T, T) -> T,
    left: *const T,
    left_length: usize,
    right: *const T,
    right_length: usize,
    output: *mut T,
    length: usize,
    workers: usize,
) {
    let left = left as usize;
    let right = right as usize;
    let output = output as usize;
    let left_step = usize::from(left_length != 1);
    let right_step = usize::from(right_length != 1);
    run_ranges(length, workers, &|start, stop| {
        let a = left as *const T;
        let b = right as *const T;
        let target = output as *mut T;
        for index in start..stop {
            // SAFETY: as in `unary`; a scalar operand is read at index 0.
            unsafe {
                *target.add(index) =
                    function(*a.add(index * left_step), *b.add(index * right_step))
            };
        }
    });
}

fn operand_ok(operand_length: usize, length: usize) -> bool {
    operand_length == length || operand_length == 1
}

/// The generation of the four entries below.  Python looks the entries up
/// by name (an older library simply lacks them) and reads this to refuse a
/// library whose operation codes it does not know.
#[no_mangle]
pub extern "C" fn gpuwm_portable_math_version() -> u32 {
    1
}

/// Elementwise float64 unary transcendental through the vendored libm.
///
/// # Safety
/// `input` and `output` address `length` f64 values; `output` is writable
/// and is either `input` itself or does not overlap it.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_portable_unary_f64(
    op: u32,
    input: *const f64,
    output: *mut f64,
    length: usize,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let Some(function) = unary_f64(op) else {
            return ERR_DIMENSION;
        };
        if length == 0 {
            return OK;
        }
        if input.is_null() || output.is_null() {
            return ERR_NULL;
        }
        if workers == 0 {
            return ERR_DIMENSION;
        }
        unary(function, input, output, length, workers);
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

/// Elementwise float32 unary transcendental through the vendored libm.
///
/// # Safety
/// As [`gpuwm_portable_unary_f64`], for f32 values.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_portable_unary_f32(
    op: u32,
    input: *const f32,
    output: *mut f32,
    length: usize,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let Some(function) = unary_f32(op) else {
            return ERR_DIMENSION;
        };
        if length == 0 {
            return OK;
        }
        if input.is_null() || output.is_null() {
            return ERR_NULL;
        }
        if workers == 0 {
            return ERR_DIMENSION;
        }
        unary(function, input, output, length, workers);
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

/// Elementwise float64 `pow` or `atan2` through the vendored libm.
///
/// # Safety
/// `left` addresses `left_length` and `right` addresses `right_length`
/// f64 values, each `length` or 1; `output` addresses `length` writable
/// f64 values and is an operand of full length itself or overlaps neither.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_portable_binary_f64(
    op: u32,
    left: *const f64,
    left_length: usize,
    right: *const f64,
    right_length: usize,
    output: *mut f64,
    length: usize,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let Some(function) = binary_f64(op) else {
            return ERR_DIMENSION;
        };
        if length == 0 {
            return OK;
        }
        if left.is_null() || right.is_null() || output.is_null() {
            return ERR_NULL;
        }
        if workers == 0 || !operand_ok(left_length, length) || !operand_ok(right_length, length) {
            return ERR_DIMENSION;
        }
        binary(function, left, left_length, right, right_length, output, length, workers);
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

/// Elementwise float32 `powf` or `atan2f` through the vendored libm.
///
/// # Safety
/// As [`gpuwm_portable_binary_f64`], for f32 values.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_portable_binary_f32(
    op: u32,
    left: *const f32,
    left_length: usize,
    right: *const f32,
    right_length: usize,
    output: *mut f32,
    length: usize,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let Some(function) = binary_f32(op) else {
            return ERR_DIMENSION;
        };
        if length == 0 {
            return OK;
        }
        if left.is_null() || right.is_null() || output.is_null() {
            return ERR_NULL;
        }
        if workers == 0 || !operand_ok(left_length, length) || !operand_ok(right_length, length) {
            return ERR_DIMENSION;
        }
        binary(function, left, left_length, right, right_length, output, length, workers);
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(length: usize) -> Vec<f64> {
        (0..length).map(|i| 1.0e-3 + i as f64 * 7.3e-4).collect()
    }

    #[test]
    fn worker_count_never_changes_an_element() {
        let input = ramp(100_003);
        let mut one = vec![0.0; input.len()];
        let mut many = vec![0.0; input.len()];
        for op in [UNARY_EXP, UNARY_LOG, UNARY_LOG1P, UNARY_SIN, UNARY_TAN, UNARY_ATAN] {
            unsafe {
                assert_eq!(gpuwm_portable_unary_f64(op, input.as_ptr(), one.as_mut_ptr(), input.len(), 1), OK);
                assert_eq!(gpuwm_portable_unary_f64(op, input.as_ptr(), many.as_mut_ptr(), input.len(), 7), OK);
            }
            assert!(one.iter().zip(&many).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[test]
    fn a_scalar_operand_broadcasts_and_matches_the_scalar_call() {
        let base = ramp(40_000);
        let exponent = [0.2857142857142857_f64];
        let mut out = vec![0.0; base.len()];
        unsafe {
            assert_eq!(
                gpuwm_portable_binary_f64(BINARY_POW, base.as_ptr(), base.len(), exponent.as_ptr(), 1,
                                          out.as_mut_ptr(), base.len(), 4),
                OK
            );
        }
        for (x, y) in base.iter().zip(&out) {
            assert_eq!(libm::pow(*x, exponent[0]).to_bits(), y.to_bits());
        }
    }

    #[test]
    fn in_place_output_is_allowed_and_bad_calls_are_refused() {
        let mut values = vec![0.5_f32, 1.0, 2.0];
        let expected: Vec<f32> = values.iter().map(|v| libm::logf(*v)).collect();
        unsafe {
            let pointer = values.as_mut_ptr();
            assert_eq!(gpuwm_portable_unary_f32(UNARY_LOG, pointer, pointer, 3, 2), OK);
            assert_eq!(gpuwm_portable_unary_f32(99, pointer, pointer, 3, 2), ERR_DIMENSION);
            assert_eq!(gpuwm_portable_unary_f32(UNARY_LOG, std::ptr::null(), pointer, 3, 2), ERR_NULL);
            assert_eq!(
                gpuwm_portable_binary_f32(BINARY_POW, pointer, 2, pointer, 3, pointer, 3, 1),
                ERR_DIMENSION
            );
        }
        assert_eq!(values, expected);
    }
}
