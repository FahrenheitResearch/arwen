//! The numpy scalar semantics the reference relies on, spelled out.

/// `numpy.remainder` on float64 (the `%` the reference writes): `fmod`,
/// moved into the divisor's sign, and an exact zero carrying the divisor's
/// sign (`npy_divmod`).
#[inline]
pub fn py_mod(a: f64, b: f64) -> f64 {
    let m = a % b;
    if b == 0.0 {
        return m;
    }
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) { m + b } else { m }
    } else {
        0.0f64.copysign(b)
    }
}

/// `numpy.clip(x, lo, hi)`: NaN passes through.
#[inline]
pub fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    if x.is_nan() {
        x
    } else if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// `numpy.rint(x).astype(numpy.intp)`: round half to even, and a value
/// that is not a finite integer-sized number becomes the most negative
/// index (what the C cast numpy performs yields on x86-64), which every
/// bounds test then rejects.
#[inline]
pub fn rint_index(x: f64) -> i64 {
    let r = x.round_ties_even();
    if r.is_finite() && r >= -9.2e18 && r <= 9.2e18 { r as i64 } else { i64::MIN }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remainder_matches_numpy() {
        assert_eq!(py_mod(-1.0, 360.0), 359.0);
        assert_eq!(py_mod(361.0, 360.0), 1.0);
        assert!(py_mod(-360.0, 360.0).is_sign_positive());
        assert_eq!(py_mod(720.0, 360.0), 0.0);
    }

    #[test]
    fn rint_is_half_even() {
        assert_eq!(rint_index(0.5), 0);
        assert_eq!(rint_index(1.5), 2);
        assert_eq!(rint_index(-0.5), 0);
        assert_eq!(rint_index(f64::NAN), i64::MIN);
    }
}
