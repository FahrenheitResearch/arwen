//! The one maths library of the post-processor (specification D30).
//!
//! Every elementary function any group needs, on the CPU and (through the
//! twin `kernels/woof_math.cuh`) on the GPU.  Both sides are built from
//! IEEE `+ - * /`, `sqrt` and exact bit operations only, in the same order,
//! with no fused multiply-add: Rust never contracts `a * b + c`, and the
//! kernels are compiled with `--fmad=false`, `--prec-div=true` and
//! `--prec-sqrt=true`.  No platform C library, no `libm` crate, no CUDA
//! libdevice transcendental is called, so a frame returns the same bits on
//! Linux, Windows and every CUDA architecture.
//!
//! Method (textbook, written for WOOF): the natural logarithm reduces
//! x = m 2^e with m in [sqrt(1/2), sqrt(2)) and sums 2 atanh(s),
//! s = (m - 1) / (m + 1), as a polynomial in s^2; the exponential reduces
//! y = n ln 2 + r with the two-part split of ln 2 and sums a nested Taylor
//! series; sine and cosine reduce by multiples of pi/2 (three-part split)
//! and sum nested Taylor series on |r| <= pi/4; the arctangent reduces to
//! |u| <= tan(pi/12) and sums its alternating series.  Accuracy is a few
//! units in the last place of binary64 over the ranges the post-processor
//! uses, far below what any f32 output resolves.
//!
//! The binary32 entry points (`expf`, `logf`, `sinf`, `cosf`, `powf_pos`)
//! widen to binary64, call the function above and round once to binary32,
//! so there is a single algorithm for both precisions.  A NaN argument is
//! returned unchanged; a NaN produced from a non-NaN argument is the
//! canonical quiet NaN of each width on both devices.

const SQRT2: f64 = 1.4142135623730951;
const LN2_HI: f64 = 6.93147180369123816490e-01;
const LN2_LO: f64 = 1.90821492927058770002e-10;
const INV_LN2: f64 = 1.44269504088896338700e+00;
const TWO_OVER_PI: f64 = 6.36619772367581382433e-01;
const PIO2_1: f64 = 1.57079632673412561417e+00;
const PIO2_2: f64 = 6.07710050650619224932e-11;
const PIO2_3: f64 = 2.02226624871116645580e-21;
const PI: f64 = 3.14159265358979311600e+00;
const PIO2: f64 = 1.57079632679489655800e+00;
const PIO6: f64 = 5.23598775598298815659e-01;
const SQRT3: f64 = 1.73205080756887719318e+00;
const TAN_PIO12: f64 = 2.67949192431122696320e-01;

/// Canonical binary64 quiet NaN (the CUDA twin returns the same word).
pub const NAN64: f64 = f64::from_bits(0x7ff8_0000_0000_0000);
/// Canonical binary32 quiet NaN.
pub const NAN32: f32 = f32::from_bits(0x7fc0_0000);

/// floor(v) for |v| < 2^52 by truncation through an integer (exact, no
/// library call).  Twin of `wm_floor`.
#[inline]
#[must_use]
pub fn floor_exact(v: f64) -> f64 {
    let t = v as i64 as f64;
    if t > v { t - 1.0 } else { t }
}

/// Natural logarithm.  Twin of `wm_ln`.
#[must_use]
pub fn ln(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return NAN64;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x > f64::MAX {
        return x;
    }
    let mut bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i64;
    if e == 0 {
        let scaled = x * f64::from_bits(((1023 + 54) as u64) << 52);
        bits = scaled.to_bits();
        e = ((bits >> 52) & 0x7ff) as i64 - 54;
    }
    e -= 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > SQRT2 {
        m = m * 0.5;
        e += 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut p = 1.0 / 23.0;
    let mut k: i32 = 10;
    while k >= 0 {
        p = p * s2 + 1.0 / f64::from(2 * k + 1);
        k -= 1;
    }
    let lnm = 2.0 * s * p;
    let de = e as f64;
    de * LN2_HI + (de * LN2_LO + lnm)
}

/// Exponential.  Twin of `wm_exp`.
#[must_use]
pub fn exp(y: f64) -> f64 {
    if y.is_nan() {
        return NAN64;
    }
    if y > 709.78 {
        return f64::INFINITY;
    }
    if y < -745.2 {
        return 0.0;
    }
    let n = floor_exact(y * INV_LN2 + 0.5);
    let r = (y - n * LN2_HI) - n * LN2_LO;
    let mut p = 1.0;
    let mut k: i32 = 13;
    while k >= 1 {
        p = 1.0 + r * p / f64::from(k);
        k -= 1;
    }
    let mut ni = n as i64;
    if ni > 1023 {
        p = p * f64::from_bits(((1023 + 1023) as u64) << 52);
        ni -= 1023;
    } else if ni < -1022 {
        p = p * f64::from_bits(((1023 - 1022) as u64) << 52);
        ni += 1022;
        if ni < -1022 {
            p = p * f64::from_bits(((1023 - 1022) as u64) << 52);
            ni += 1022;
        }
    }
    p * f64::from_bits(((ni + 1023) as u64) << 52)
}

/// x^y as exp(y ln x).  x < 0 or NaN: NaN; x = 0: 0 for y > 0, 1 for
/// y = 0, +inf for y < 0.  Twin of `wm_pow`.
#[must_use]
pub fn pow(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() || x < 0.0 {
        return NAN64;
    }
    if x == 0.0 {
        return if y > 0.0 {
            0.0
        } else if y == 0.0 {
            1.0
        } else {
            f64::INFINITY
        };
    }
    exp(y * ln(x))
}

/// Real cube root.  Twin of `wm_cbrt`.
#[must_use]
pub fn cbrt(x: f64) -> f64 {
    if x.is_nan() {
        return NAN64;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < 0.0 {
        return -exp(ln(-x) / 3.0);
    }
    exp(ln(x) / 3.0)
}

/// IEEE square root (correctly rounded on both devices).
#[inline]
#[must_use]
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// sin(r) and cos(r) on |r| <= pi/4 by nested Taylor sums.
fn sin_cos_reduced(r: f64) -> (f64, f64) {
    let r2 = r * r;
    let mut s = 1.0;
    let mut c = 1.0;
    let mut k: i32 = 9;
    while k >= 1 {
        s = 1.0 - r2 * s / f64::from((2 * k) * (2 * k + 1));
        c = 1.0 - r2 * c / f64::from((2 * k - 1) * (2 * k));
        k -= 1;
    }
    (r * s, c)
}

/// Reduce x by multiples of pi/2: (quadrant mod 4, remainder).
fn reduce_pio2(x: f64) -> (i64, f64) {
    let n = floor_exact(x * TWO_OVER_PI + 0.5);
    let r = ((x - n * PIO2_1) - n * PIO2_2) - n * PIO2_3;
    let q = (n as i64) & 3;
    (q, r)
}

/// Sine (radians).  Accurate for |x| below about 1e6.  Twin of `wm_sin`.
#[must_use]
pub fn sin(x: f64) -> f64 {
    if !x.is_finite() {
        return NAN64;
    }
    let (q, r) = reduce_pio2(x);
    let (s, c) = sin_cos_reduced(r);
    match q {
        0 => s,
        1 => c,
        2 => -s,
        _ => -c,
    }
}

/// Cosine (radians).  Twin of `wm_cos`.
#[must_use]
pub fn cos(x: f64) -> f64 {
    if !x.is_finite() {
        return NAN64;
    }
    let (q, r) = reduce_pio2(x);
    let (s, c) = sin_cos_reduced(r);
    match q {
        0 => c,
        1 => -s,
        2 => -c,
        _ => s,
    }
}

/// Arctangent of u >= 0.
fn atan_pos(t: f64) -> f64 {
    let (mut u, mut base, mut flip) = (t, 0.0, false);
    if u > 1.0 {
        u = 1.0 / u;
        flip = true;
    }
    if u > TAN_PIO12 {
        u = (u * SQRT3 - 1.0) / (u + SQRT3);
        base = PIO6;
    }
    let u2 = u * u;
    let mut p = 0.0;
    let mut k: i32 = 14;
    while k >= 0 {
        p = 1.0 / f64::from(2 * k + 1) - u2 * p;
        k -= 1;
    }
    let a = base + u * p;
    if flip { PIO2 - a } else { a }
}

/// Arctangent (radians).
#[must_use]
pub fn atan(x: f64) -> f64 {
    if x.is_nan() {
        return NAN64;
    }
    if x < 0.0 { -atan_pos(-x) } else { atan_pos(x) }
}

/// Four-quadrant arctangent of y / x (radians, in (-pi, pi]).
#[must_use]
pub fn atan2(y: f64, x: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return NAN64;
    }
    if x > 0.0 {
        return atan(y / x);
    }
    if x < 0.0 {
        let a = atan(y / x);
        return if y < 0.0 { a - PI } else { a + PI };
    }
    if y > 0.0 {
        PIO2
    } else if y < 0.0 {
        -PIO2
    } else {
        0.0
    }
}

/// Arcsine (radians), |x| <= 1.
#[must_use]
pub fn asin(x: f64) -> f64 {
    if x.is_nan() || x > 1.0 || x < -1.0 {
        return NAN64;
    }
    atan2(x, sqrt((1.0 - x) * (1.0 + x)))
}

/// Round a binary64 result once to binary32, NaN to the canonical word.
#[inline]
#[must_use]
pub fn to_f32(v: f64) -> f32 {
    if v.is_nan() { NAN32 } else { v as f32 }
}

/// e^x in binary32 (one rounding of the binary64 [`exp`]).  Twin of `woof_expf`.
#[inline]
#[must_use]
pub fn expf(x: f32) -> f32 {
    if x.is_nan() { x } else { to_f32(exp(f64::from(x))) }
}

/// ln x in binary32.  Twin of `woof_logf`.
#[inline]
#[must_use]
pub fn logf(x: f32) -> f32 {
    if x.is_nan() { x } else { to_f32(ln(f64::from(x))) }
}

/// sin x in binary32.  Twin of `woof_sinf`.
#[inline]
#[must_use]
pub fn sinf(x: f32) -> f32 {
    if x.is_nan() { x } else { to_f32(sin(f64::from(x))) }
}

/// cos x in binary32.  Twin of `woof_cosf`.
#[inline]
#[must_use]
pub fn cosf(x: f32) -> f32 {
    if x.is_nan() { x } else { to_f32(cos(f64::from(x))) }
}

/// x^y for x > 0 in binary32, exp(y ln x) evaluated in binary64.  Twin of
/// `woof_powpos`.
#[inline]
#[must_use]
pub fn powf_pos(x: f32, y: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if y.is_nan() {
        return y;
    }
    to_f32(pow(f64::from(x), f64::from(y)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, ulps: f64) -> bool {
        (a - b).abs() <= ulps * f64::EPSILON * b.abs().max(f64::MIN_POSITIVE)
    }

    #[test]
    fn ln_and_exp_match_the_reference_to_a_few_ulp() {
        let mut x = 1.0e-300_f64;
        while x < 1.0e300 {
            for &v in &[x, x * 1.37, x * 2.9, x * 0.7071] {
                let b = v.ln();
                assert!((ln(v) - b).abs() <= 4.0 * f64::EPSILON * b.abs().max(1.0), "ln({v})");
            }
            x *= 3.7;
        }
        let mut y = -700.0_f64;
        while y < 700.0 {
            assert!(close(exp(y), y.exp(), 4.0), "exp({y})");
            y += 0.913;
        }
        assert_eq!(ln(1.0), 0.0);
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(ln(-1.0).to_bits(), NAN64.to_bits());
        assert_eq!(ln(0.0), f64::NEG_INFINITY);
        assert!((ln(f64::from_bits(1)) - f64::from_bits(1).ln()).abs() < 1e-12);
    }

    #[test]
    fn pow_cbrt_sqrt() {
        for &(x, y) in &[(0.9, 0.2857142857142857), (1.03, 5.2558), (300.0, -0.5), (1e-5, 2.0)] {
            assert!(close(pow(x, y), f64::powf(x, y), 64.0), "pow({x},{y})");
        }
        assert_eq!(pow(0.0, 2.0), 0.0);
        assert!(pow(-1.0, 2.0).is_nan());
        for &x in &[1.0e-9, 0.001, 0.27, 1.0, 8.0, 27.0, 1234.5, -8.0] {
            assert!(close(cbrt(x), f64::cbrt(x), 8.0), "cbrt({x})");
        }
    }

    #[test]
    fn trig_matches_the_reference() {
        let mut x = -20.0_f64;
        while x < 20.0 {
            assert!((sin(x) - x.sin()).abs() < 4e-16, "sin({x})");
            assert!((cos(x) - x.cos()).abs() < 4e-16, "cos({x})");
            assert!((atan(x) - x.atan()).abs() < 4e-16, "atan({x})");
            x += 0.0137;
        }
        for &(y, x) in &[(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0), (0.3, -2.0), (2.0, 0.0)] {
            assert!((atan2(y, x) - f64::atan2(y, x)).abs() < 4e-16, "atan2({y},{x})");
        }
        for &x in &[-1.0, -0.5, 0.0, 0.3, 0.97, 1.0] {
            assert!((asin(x) - x.asin()).abs() < 1e-15, "asin({x})");
        }
    }

    #[test]
    fn binary32_entry_points_round_once() {
        assert_eq!(expf(0.0), 1.0);
        assert_eq!(logf(1.0), 0.0);
        assert_eq!(logf(-1.0).to_bits(), NAN32.to_bits());
        let nan = f32::from_bits(0x7fc0_1234);
        assert_eq!(expf(nan).to_bits(), nan.to_bits());
        for &x in &[0.1f32, 1.5, 17.3, -3.25] {
            assert_eq!(expf(x), (f64::from(x)).exp() as f32);
            assert!((sinf(x) - x.sin()).abs() <= f32::EPSILON);
        }
    }
}
