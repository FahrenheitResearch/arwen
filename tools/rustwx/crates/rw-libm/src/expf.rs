//! binary32 exponential as glibc 2.43 computes it: Arm optimized-routines
//! math/expf.c and math/exp2f_data.c, in glibc's copy
//! (sysdeps/ieee754/flt-32/e_expf.c, e_exp2f_data.c) and in the x86-64 FMA
//! IFUNC variant (`__expf_fma`): every multiply-add GCC fuses there is
//! `mul_add` here.  Notice in lib.rs.  Not correctly rounded (glibc's own
//! count: 170,635 wrong results with FMA); bit for bit glibc 2.43's answer.

use crate::{hf, hf32};

/// Shared with powf: tab[i] = uint(2^(i/32)) - (i << 47).
pub(crate) const EXP2F_TAB: [u64; 32] = [
    0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
    0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
    0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
    0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
    0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
    0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
    0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
];
/// exp2f_data.poly, the 2^r polynomial for |r| <= 1/64.
pub(crate) const EXP2F_POLY: [f64; 3] =
    [hf("0x1.c6af84b912394p-5"), hf("0x1.ebfce50fac4f3p-3"), hf("0x1.62e42ff0c52d6p-1")];
/// exp2f_data.poly_scaled: EXP2F_POLY[i] / 32^(3-i), exact.
const POLY_SCALED: [f64; 3] = [
    EXP2F_POLY[0] / 32.0 / 32.0 / 32.0,
    EXP2F_POLY[1] / 32.0 / 32.0,
    EXP2F_POLY[2] / 32.0,
];
const INVLN2_SCALED: f64 = hf("0x1.71547652b82fep+0") * 32.0;
const SHIFT: f64 = hf("0x1.8p+52");

const fn top12(x: f32) -> u32 {
    x.to_bits() >> 20
}

/// e^x for binary32, bit for bit as glibc 2.43 returns it.
#[inline]
pub fn expf(x: f32) -> f32 {
    let xd = x as f64;
    let abstop = top12(x) & 0x7ff;
    if abstop >= top12(88.0) {
        // |x| >= 88 or x is NaN
        if x.to_bits() == f32::NEG_INFINITY.to_bits() {
            return 0.0;
        }
        if abstop >= top12(f32::INFINITY) {
            return x + x;
        }
        if x > const { hf32("0x1.62e42ep6") } {
            // overflow: __math_oflowf(0)
            return const { hf32("0x1p97") } * const { hf32("0x1p97") };
        }
        if x < const { hf32("-0x1.9fe368p6") } {
            // underflow: __math_uflowf(0)
            return const { hf32("0x1p-95") } * const { hf32("0x1p-95") };
        }
        if x < const { hf32("-0x1.9d1d9ep6") } {
            // __math_may_uflowf(0): exp(x) is within [2^-150, 2^-149)
            return const { hf32("0x1.4p-75") } * const { hf32("0x1.4p-75") };
        }
    }
    // x*N/Ln2 = k + r with r in [-1/2, 1/2] and int k.  The C computes
    // z = InvLn2N * x once and uses it only in an addition and a
    // subtraction, so GCC's FMA build fuses the product into both.
    let kd = INVLN2_SCALED.mul_add(xd, SHIFT);
    let ki = kd.to_bits();
    let kd = kd - SHIFT;
    let r = INVLN2_SCALED.mul_add(xd, -kd);
    // exp(x) = 2^(k/N) * 2^(r/N) ~= s * (C0*r^3 + C1*r^2 + C2*r + 1)
    let t = EXP2F_TAB[(ki % 32) as usize].wrapping_add(ki << (52 - 5));
    let s = f64::from_bits(t);
    let z = POLY_SCALED[0].mul_add(r, POLY_SCALED[1]);
    let r2 = r * r;
    let y = POLY_SCALED[2].mul_add(r, 1.0);
    let y = z.mul_add(r2, y);
    (y * s) as f32
}
