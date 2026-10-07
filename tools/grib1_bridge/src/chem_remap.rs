//! Generic single-round pressure and linear combination operators.
use crate::{ERR_DIMENSION, ERR_NONFINITE, ERR_NULL, ERR_PANIC, OK};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// # Safety
/// All buffers must address their declared lengths and output must not alias input.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_hybrid_full_pressure_f32(
    a_half: *const f64,
    b_half: *const f64,
    nhalf: usize,
    levels: *const i32,
    nlevel: usize,
    surface_pressure: *const f32,
    ncol: usize,
    out: *mut f32,
    _workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if a_half.is_null()
            || b_half.is_null()
            || levels.is_null()
            || surface_pressure.is_null()
            || out.is_null()
        {
            return ERR_NULL;
        }
        if nhalf < 2
            || nlevel == 0
            || ncol == 0
            || nlevel
                .checked_mul(ncol)
                .filter(|n| *n <= isize::MAX as usize / 4)
                .is_none()
            || nhalf > isize::MAX as usize / 8
        {
            return ERR_DIMENSION;
        }
        let a = std::slice::from_raw_parts(a_half, nhalf);
        let b = std::slice::from_raw_parts(b_half, nhalf);
        let lev = std::slice::from_raw_parts(levels, nlevel);
        let ps = std::slice::from_raw_parts(surface_pressure, ncol);
        if lev.iter().any(|k| *k < 1 || *k as usize >= nhalf)
            || lev.windows(2).any(|p| p[0] >= p[1])
        {
            return ERR_DIMENSION;
        }
        if a.iter().chain(b).any(|v| !v.is_finite())
            || ps.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return ERR_NONFINITE;
        }
        let dest = std::slice::from_raw_parts_mut(out, nlevel * ncol);
        for (l, k) in lev.iter().enumerate() {
            let k = *k as usize;
            for (c, p) in ps.iter().enumerate() {
                let upper = a[k - 1] + b[k - 1] * f64::from(*p);
                let lower = a[k] + b[k] * f64::from(*p);
                dest[l * ncol + c] = (0.5 * (upper + lower)) as f32;
            }
        }
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

/// # Safety
/// All buffers must address their declared lengths and output must not alias input.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_weighted_combination_f32(
    fields: *const f32,
    weights: *const f64,
    nfield: usize,
    n: usize,
    scale: f64,
    humidity: *const f32,
    out: *mut f32,
    _workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if fields.is_null() || weights.is_null() || out.is_null() {
            return ERR_NULL;
        }
        if nfield == 0
            || n == 0
            || nfield
                .checked_mul(n)
                .filter(|v| *v <= isize::MAX as usize / 4)
                .is_none()
            || nfield > isize::MAX as usize / 8
        {
            return ERR_DIMENSION;
        }
        let f = std::slice::from_raw_parts(fields, nfield * n);
        let w = std::slice::from_raw_parts(weights, nfield);
        let q = if humidity.is_null() {
            None
        } else {
            Some(std::slice::from_raw_parts(humidity, n))
        };
        if !scale.is_finite() || w.iter().any(|v| !v.is_finite()) {
            return ERR_NONFINITE;
        }
        if q.map_or(false, |q| {
            q.iter().any(|v| !v.is_finite() || *v < 0.0 || *v >= 1.0)
        }) {
            return ERR_NONFINITE;
        }
        let dest = std::slice::from_raw_parts_mut(out, n);
        for i in 0..n {
            let mut sum = 0.0f64;
            for j in 0..nfield {
                sum += w[j] * f64::from(f[j * n + i]);
            }
            let value = scale * sum;
            dest[i] = (match q {
                Some(q) => value / (1.0 - f64::from(q[i])),
                None => value,
            }) as f32;
        }
        OK
    }))
    .unwrap_or(ERR_PANIC)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_pressure_and_levels_leave_output_untouched() {
        let a = [0.0, 0.0];
        let b = [0.0, 1.0];
        let levels = [1];
        let mut out = [123.0];
        for ps in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let code = unsafe {
                gpuwm_hybrid_full_pressure_f32(
                    a.as_ptr(),
                    b.as_ptr(),
                    2,
                    levels.as_ptr(),
                    1,
                    &ps,
                    1,
                    out.as_mut_ptr(),
                    4,
                )
            };
            assert_eq!(code, ERR_NONFINITE);
            assert_eq!(out, [123.0]);
        }
        for k in [0, 2, -1] {
            let ps = 100000.0;
            assert_eq!(
                unsafe {
                    gpuwm_hybrid_full_pressure_f32(
                        a.as_ptr(),
                        b.as_ptr(),
                        2,
                        &k,
                        1,
                        &ps,
                        1,
                        out.as_mut_ptr(),
                        4,
                    )
                },
                ERR_DIMENSION
            );
            assert_eq!(out, [123.0]);
        }
    }
    #[test]
    fn invalid_divisor_weights_and_scale_leave_output_untouched() {
        let fields = [1.0];
        let weights = [1.0];
        let mut out = [123.0];
        for q in [-0.01, 1.0, 1.01, f32::NAN, f32::INFINITY] {
            assert_eq!(
                unsafe {
                    gpuwm_weighted_combination_f32(
                        fields.as_ptr(),
                        weights.as_ptr(),
                        1,
                        1,
                        1.0,
                        &q,
                        out.as_mut_ptr(),
                        4,
                    )
                },
                ERR_NONFINITE
            );
            assert_eq!(out, [123.0]);
        }
        assert_eq!(
            unsafe {
                gpuwm_weighted_combination_f32(
                    fields.as_ptr(),
                    weights.as_ptr(),
                    1,
                    1,
                    f64::NAN,
                    std::ptr::null(),
                    out.as_mut_ptr(),
                    4,
                )
            },
            ERR_NONFINITE
        );
        assert_eq!(
            unsafe {
                gpuwm_weighted_combination_f32(
                    fields.as_ptr(),
                    [f64::INFINITY].as_ptr(),
                    1,
                    1,
                    1.0,
                    std::ptr::null(),
                    out.as_mut_ptr(),
                    4,
                )
            },
            ERR_NONFINITE
        );
        assert_eq!(out, [123.0]);
    }
    #[test]
    fn null_and_overflow_dimensions_refuse() {
        let mut out = [0.0];
        let f = [1.0];
        let w = [1.0];
        assert_eq!(
            unsafe {
                gpuwm_weighted_combination_f32(
                    std::ptr::null(),
                    w.as_ptr(),
                    1,
                    1,
                    1.0,
                    std::ptr::null(),
                    out.as_mut_ptr(),
                    1,
                )
            },
            ERR_NULL
        );
        assert_eq!(
            unsafe {
                gpuwm_weighted_combination_f32(
                    f.as_ptr(),
                    w.as_ptr(),
                    usize::MAX,
                    2,
                    1.0,
                    std::ptr::null(),
                    out.as_mut_ptr(),
                    1,
                )
            },
            ERR_DIMENSION
        );
        assert_eq!(
            unsafe {
                gpuwm_hybrid_full_pressure_f32(
                    w.as_ptr(),
                    w.as_ptr(),
                    2,
                    [1].as_ptr(),
                    usize::MAX,
                    f.as_ptr(),
                    2,
                    out.as_mut_ptr(),
                    1,
                )
            },
            ERR_DIMENSION
        );
    }
}
