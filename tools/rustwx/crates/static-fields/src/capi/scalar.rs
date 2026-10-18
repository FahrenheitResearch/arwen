//! Scalar rounding at the strict terrain initialization boundary.
//!
//! This is initialization of a static rotation plane, once per domain.
//! It does not replace the device math routines or any time-step producer.

use super::{clear_error, guard, set_error, ERR, OK};

/// Evaluate the native scalar inverse sine on input binary32 words.
/// The caller validates the scalar-rounding contract before using this
/// optional entry point in a strict forecast.
///
/// # Safety
/// Input and output describe `count` aligned, nonoverlapping binary32 words.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_scalar_asin_f32(
    input: *const f32,
    output: *mut f32,
    count: usize,
) -> i32 {
    guard(ERR, || {
        clear_error();
        if count == 0 {
            return OK;
        }
        if input.is_null() || output.is_null()
            || count > isize::MAX as usize / std::mem::size_of::<f32>()
        {
            return set_error("invalid scalar rotation input/output buffer");
        }
        let input = unsafe { std::slice::from_raw_parts(input, count) };
        let output = unsafe { std::slice::from_raw_parts_mut(output, count) };
        for (&value, result) in input.iter().zip(output.iter_mut()) {
            *result = value.asin();
        }
        OK
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_rotation_preserves_zero_sign_and_endpoints() {
        let input = std::hint::black_box([0.0_f32, -0.0, 1.0, -1.0]);
        let mut output = [0.0; 4];
        assert_eq!(unsafe {
            gpuwm_static_scalar_asin_f32(input.as_ptr(), output.as_mut_ptr(), input.len())
        }, OK);
        assert_eq!(output.map(f32::to_bits), [0, 0x8000_0000, 0x3fc9_0fdb, 0xbfc9_0fdb]);
    }

    #[test]
    fn scalar_rotation_refuses_nonempty_null_buffers() {
        assert_eq!(unsafe {
            gpuwm_static_scalar_asin_f32(std::ptr::null(), std::ptr::null_mut(), 1)
        }, ERR);
    }
}
