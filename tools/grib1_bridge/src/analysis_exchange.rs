//! Host analysis arithmetic. Python owns buffers and metadata, not their values.
use crate::{ERR_DIMENSION, ERR_NONFINITE, ERR_NULL, ERR_PANIC, OK};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Elementwise sum/difference with optional nonnegative clipping. Statistics are
/// [negative count, minimum negative, nonzero count, RMS, max absolute].
/// # Safety
/// Inputs and output have `length` f64 words; stats has five. No overlap.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_analysis_binary_f64(
    first: *const f64,
    second: *const f64,
    output: *mut f64,
    stats: *mut f64,
    length: usize,
    subtract: u32,
    clip: u32,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if subtract > 1 || clip > 2 {
            return ERR_DIMENSION;
        }
        if [first, second, output, stats].iter().any(|p| p.is_null()) {
            return ERR_NULL;
        }
        let (a, b, out) = (
            std::slice::from_raw_parts(first, length),
            std::slice::from_raw_parts(second, length),
            std::slice::from_raw_parts_mut(output, length),
        );
        let mut negative = 0usize;
        let mut minimum = 0.0_f64;
        let mut nonzero = 0usize;
        let mut maximum = 0.0_f64;
        for i in 0..length {
            let value = if subtract == 1 {
                a[i] - b[i]
            } else {
                a[i] + b[i]
            };
            if !value.is_finite() {
                return ERR_NONFINITE;
            }
            if value < 0.0 {
                negative += 1;
                minimum = minimum.min(value);
            }
            nonzero += usize::from(value != 0.0);
            maximum = maximum.max(value.abs());
            out[i] = if (clip == 1 && value < 0.0) || (clip == 2 && value <= 0.0) {
                0.0
            } else {
                value
            };
        }
        // NumPy's contiguous pairwise reduction tree, including negative zero.
        fn sum_squares(a: &[f64], b: &[f64], subtract: bool) -> f64 {
            let n = a.len();
            let square = |i: usize| {
                let v = if subtract { a[i] - b[i] } else { a[i] + b[i] };
                v * v
            };
            if n < 8 {
                let mut s = -0.0;
                for i in 0..n {
                    s += square(i);
                }
                return s;
            }
            if n <= 128 {
                let mut r = [0.0; 8];
                for i in 0..8 {
                    r[i] = square(i);
                }
                let mut i = 8;
                while i + 7 < n {
                    for j in 0..8 {
                        r[j] += square(i + j);
                    }
                    i += 8;
                }
                let mut s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
                while i < n {
                    s += square(i);
                    i += 1;
                }
                return s;
            }
            let half = (n / 2) / 8 * 8;
            sum_squares(&a[..half], &b[..half], subtract)
                + sum_squares(&a[half..], &b[half..], subtract)
        }
        let values = [
            negative as f64,
            minimum,
            nonzero as f64,
            if length == 0 {
                0.0
            } else {
                (sum_squares(a, b, subtract == 1) / length as f64).sqrt()
            },
            maximum,
        ];
        std::slice::from_raw_parts_mut(stats, 5).copy_from_slice(&values);
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

/// Hydrostatic before/after integration with the existing fixed column tree.
/// The buffers are field-major: p, theta-before, theta-after, vapor-before,
/// vapor-after, total-loading-before, total-loading-after, base-theta (all 3D);
/// base-mass, perturbation-mass-before/after (2D); c1f,c2f,c1h,c2h,c3f,c4f,
/// c3h,c4h,rdnw,rdn,dnw (profiles). `constants`: Rd,Rv/Rd,p0,Rd/cp,p_top.
/// # Safety
/// The 22 input pointers have the lengths above. Two disjoint outputs have
/// (levels+1)*columns and levels*columns words. Constants has five words.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_analysis_hydro_f64(
    inputs: *const *const f64,
    constants: *const f64,
    phi: *mut f64,
    pressure: *mut f64,
    levels: usize,
    columns: usize,
    option: u32,
    mass: u32,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if levels == 0
            || columns == 0
            || levels
                .checked_add(1)
                .and_then(|n| n.checked_mul(columns))
                .is_none()
            || !(option == 1 || option == 2)
            || mass > 1
        {
            return ERR_DIMENSION;
        }
        if inputs.is_null() || constants.is_null() || phi.is_null() || pressure.is_null() {
            return ERR_NULL;
        }
        let pointers = std::slice::from_raw_parts(inputs, 22);
        if pointers.iter().any(|p| p.is_null()) {
            return ERR_NULL;
        }
        let c = std::slice::from_raw_parts(constants, 5);
        let get = |field: usize, index: usize| *pointers[field].add(index);
        let mut pb = vec![0.0; levels];
        let mut pa = vec![0.0; levels];
        for col in 0..columns {
            let mub = get(8, col);
            let mupb = get(9, col);
            let mupa = get(10, col);
            for (qfield, mup, p) in [(5, mupb, &mut pb), (6, mupa, &mut pa)] {
                let top = levels - 1;
                let q = get(qfield, top * columns + col);
                let cq = 1.0 / (1.0 + q);
                let load = q * cq;
                p[top] = -0.5
                    * (get(11, levels) * mup + load * (get(11, levels) * mub + get(12, levels)))
                    / get(19, top)
                    / cq;
                for k in (0..top).rev() {
                    let qbar = 0.5
                        * (get(qfield, k * columns + col) + get(qfield, (k + 1) * columns + col));
                    let cq = 1.0 / (1.0 + qbar);
                    let load = qbar * cq;
                    p[k] = p[k + 1]
                        - (get(11, k + 1) * mup + load * (get(11, k + 1) * mub + get(12, k + 1)))
                            / cq
                            / get(20, k + 1);
                }
            }
            *phi.add(col) = 0.0;
            let mut integrated = 0.0;
            for k in 0..levels {
                let i = k * columns + col;
                let dp = pa[k] - pb[k];
                *pressure.add(i) = dp;
                let p = get(0, i);
                let analysed = p + dp;
                if !analysed.is_finite() || analysed <= 0.0 {
                    return ERR_NONFINITE;
                }
                let alt = |theta: f64, qv: f64, p: f64| {
                    c[0] * theta * (1.0 + c[1] * qv) * libm::pow(p / c[2], c[3]) / p
                };
                let aa = alt(get(7, i) + get(2, i), get(4, i), analysed);
                let ab = alt(get(7, i) + get(1, i), get(3, i), p);
                let operator = |mu: f64| {
                    if option == 1 {
                        -get(21, k) * (get(13, k) * mu + get(14, k))
                    } else {
                        let up = get(15, k + 1) * mu + get(16, k + 1) + c[4];
                        let down = get(15, k) * mu + get(16, k) + c[4];
                        let mid = get(17, k) * mu + get(18, k) + c[4];
                        mid * libm::log(down / up)
                    }
                };
                let dd = if mass == 1 {
                    operator(mub + mupa) * aa - operator(mub + mupb) * ab
                } else {
                    operator(mub + mupb) * (aa - ab)
                };
                integrated = if k == 0 { dd } else { integrated + dd };
                if !integrated.is_finite() {
                    return ERR_NONFINITE;
                }
                *phi.add((k + 1) * columns + col) = integrated;
            }
        }
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn clipping_statistics_and_signed_zero() {
        let a = [-0.0, -3.0, 2.0];
        let b = [0.0, 1.0, 2.0];
        let mut out = [0.0; 3];
        let mut stats = [0.0; 5];
        assert_eq!(
            unsafe {
                gpuwm_analysis_binary_f64(
                    a.as_ptr(),
                    b.as_ptr(),
                    out.as_mut_ptr(),
                    stats.as_mut_ptr(),
                    3,
                    1,
                    1,
                )
            },
            OK
        );
        assert_eq!(out[0].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(out[1], 0.0);
        assert_eq!(&stats[..3], &[1.0, -4.0, 1.0]);
    }
    #[test]
    fn refuses_nonfinite_and_bad_operation() {
        let a = [f64::INFINITY];
        let mut out = [0.0];
        let mut stats = [0.0; 5];
        assert_eq!(
            unsafe {
                gpuwm_analysis_binary_f64(
                    a.as_ptr(),
                    a.as_ptr(),
                    out.as_mut_ptr(),
                    stats.as_mut_ptr(),
                    1,
                    0,
                    0,
                )
            },
            ERR_NONFINITE
        );
        assert_eq!(
            unsafe {
                gpuwm_analysis_binary_f64(
                    a.as_ptr(),
                    a.as_ptr(),
                    out.as_mut_ptr(),
                    stats.as_mut_ptr(),
                    1,
                    2,
                    0,
                )
            },
            ERR_DIMENSION
        );
    }
}
