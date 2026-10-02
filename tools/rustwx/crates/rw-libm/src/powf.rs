//! binary32 power as glibc 2.43 computes it: Arm optimized-routines
//! math/powf.c, math/powf_log2_data.c and math/exp2f_data.c, in glibc's copy
//! (sysdeps/ieee754/flt-32/e_powf.c, e_powf_log2_data.c, e_exp2f_data.c)
//! and in the x86-64 FMA IFUNC variant (`__powf_fma`): every multiply-add
//! GCC fuses there is `mul_add` here.  Notice in lib.rs.  Not correctly
//! rounded (glibc's own worst case is 0.82 ULP); bit for bit glibc 2.43's
//! answer, including its overflow, underflow and special-value results.

use crate::expf::{EXP2F_POLY, EXP2F_TAB};
use crate::{hf, hf32};

/// (1/c, log2(c)) for 16 subintervals of [OFF, 2*OFF] (POWF_SCALE = 1).
const TAB: [[f64; 2]; 16] = [
    [hf("0x1.661ec79f8f3bep+0"), hf("-0x1.efec65b963019p-2")],
    [hf("0x1.571ed4aaf883dp+0"), hf("-0x1.b0b6832d4fca4p-2")],
    [hf("0x1.49539f0f010bp+0"), hf("-0x1.7418b0a1fb77bp-2")],
    [hf("0x1.3c995b0b80385p+0"), hf("-0x1.39de91a6dcf7bp-2")],
    [hf("0x1.30d190c8864a5p+0"), hf("-0x1.01d9bf3f2b631p-2")],
    [hf("0x1.25e227b0b8eap+0"), hf("-0x1.97c1d1b3b7afp-3")],
    [hf("0x1.1bb4a4a1a343fp+0"), hf("-0x1.2f9e393af3c9fp-3")],
    [hf("0x1.12358f08ae5bap+0"), hf("-0x1.960cbbf788d5cp-4")],
    [hf("0x1.0953f419900a7p+0"), hf("-0x1.a6f9db6475fcep-5")],
    [hf("0x1p+0"), hf("0x0p+0")],
    [hf("0x1.e608cfd9a47acp-1"), hf("0x1.338ca9f24f53dp-4")],
    [hf("0x1.ca4b31f026aap-1"), hf("0x1.476a9543891bap-3")],
    [hf("0x1.b2036576afce6p-1"), hf("0x1.e840b4ac4e4d2p-3")],
    [hf("0x1.9c2d163a1aa2dp-1"), hf("0x1.40645f0c6651cp-2")],
    [hf("0x1.886e6037841edp-1"), hf("0x1.88e9c2c1b9ff8p-2")],
    [hf("0x1.767dcf5534862p-1"), hf("0x1.ce0a44eb17bccp-2")],
];
const A: [f64; 5] = [
    hf("0x1.27616c9496e0bp-2"),
    hf("-0x1.71969a075c67ap-2"),
    hf("0x1.ec70a6ca7baddp-2"),
    hf("-0x1.7154748bef6c8p-1"),
    hf("0x1.71547652ab82bp0"),
];
const OFF: u32 = 0x3f33_0000;
/// Sign bit of the result, carried into the exp2 table index.
const SIGN_BIAS: u32 = 1 << (5 + 11);
/// exp2f_data.shift_scaled = 0x1.8p+52 / 32.
const SHIFT_SCALED: f64 = hf("0x1.8p+52") / 32.0;

/// log2(x) for a normalised encoding (a subnormal has a negative exponent).
#[inline(always)]
fn log2_inline(ix: u32) -> f64 {
    let tmp = ix.wrapping_sub(OFF);
    let i = ((tmp >> (23 - 4)) % 16) as usize;
    let top = tmp & 0xff80_0000;
    let iz = ix.wrapping_sub(top);
    let k = (top as i32) >> 23;
    let [invc, logc] = TAB[i];
    let z = f32::from_bits(iz) as f64;
    // log2(x) = log1p(z/c - 1)/ln2 + log2(c) + k
    let r = z.mul_add(invc, -1.0);
    let y0 = logc + k as f64;
    let r2 = r * r;
    let y = A[0].mul_add(r, A[1]);
    let p = A[2].mul_add(r, A[3]);
    let r4 = r2 * r2;
    let q = A[4].mul_add(r, y0);
    let q = p.mul_add(r2, q);
    y.mul_add(r4, q)
}

/// 2^xd with the result's sign carried in sign_bias, for xd in [-1021, 1023].
#[inline(always)]
fn exp2_inline(xd: f64, sign_bias: u32) -> f64 {
    let kd = xd + SHIFT_SCALED;
    let ki = kd.to_bits();
    let kd = kd - SHIFT_SCALED; // k/N
    let r = xd - kd;
    // exp2(x) = 2^(k/N) * 2^r ~= s * (C0*r^3 + C1*r^2 + C2*r + 1)
    let ski = ki.wrapping_add(sign_bias as u64);
    let t = EXP2F_TAB[(ki % 32) as usize].wrapping_add(ski << (52 - 5));
    let s = f64::from_bits(t);
    let z = EXP2F_POLY[0].mul_add(r, EXP2F_POLY[1]);
    let r2 = r * r;
    let y = EXP2F_POLY[2].mul_add(r, 1.0);
    let y = z.mul_add(r2, y);
    y * s
}

/// 0 if not an integer, 1 if an odd integer, 2 if an even integer, for the
/// encoding of a non-zero finite value.
#[inline(always)]
fn checkint(iy: u32) -> i32 {
    let e = ((iy >> 23) & 0xff) as i32;
    if e < 0x7f {
        return 0;
    }
    if e > 0x7f + 23 {
        return 2;
    }
    if iy & ((1 << (0x7f + 23 - e)) - 1) != 0 {
        return 0;
    }
    if iy & (1 << (0x7f + 23 - e)) != 0 {
        return 1;
    }
    2
}

#[inline(always)]
fn zeroinfnan(ix: u32) -> bool {
    ix.wrapping_mul(2).wrapping_sub(1) >= 2 * 0x7f80_0000 - 1
}

#[inline(always)]
fn issignaling(x: f32) -> bool {
    ((x.to_bits() ^ 0x0040_0000) & 0x7fff_ffff) > 0x7fc0_0000
}

/// glibc's xflowf: a product that overflows or underflows to the right
/// signed result in the current (nearest) rounding.
#[inline(always)]
fn xflowf(sign: bool, y: f32) -> f32 {
    (if sign { -y } else { y }) * y
}

/// x^y for binary32, bit for bit as glibc 2.43 returns it.
#[inline]
pub fn powf(x: f32, y: f32) -> f32 {
    let mut sign_bias = 0u32;
    let mut ix = x.to_bits();
    let iy = y.to_bits();
    if ix.wrapping_sub(0x0080_0000) >= 0x7f80_0000 - 0x0080_0000 || zeroinfnan(iy) {
        // x < 0x1p-126 or infinity or NaN, or y is 0, infinity or NaN
        if zeroinfnan(iy) {
            if iy.wrapping_mul(2) == 0 {
                return if issignaling(x) { x + y } else { 1.0 };
            }
            if ix == 0x3f80_0000 {
                return if issignaling(y) { x + y } else { 1.0 };
            }
            if ix.wrapping_mul(2) > 2 * 0x7f80_0000 || iy.wrapping_mul(2) > 2 * 0x7f80_0000 {
                return x + y;
            }
            if ix.wrapping_mul(2) == 2 * 0x3f80_0000 {
                return 1.0;
            }
            if (ix.wrapping_mul(2) < 2 * 0x3f80_0000) == (iy & 0x8000_0000 == 0) {
                return 0.0; // |x| < 1 && y == inf, or |x| > 1 && y == -inf
            }
            return y * y;
        }
        if zeroinfnan(ix) {
            let mut x2 = x * x;
            if ix & 0x8000_0000 != 0 && checkint(iy) == 1 {
                x2 = -x2;
                sign_bias = 1;
            }
            if ix.wrapping_mul(2) == 0 && iy & 0x8000_0000 != 0 {
                // __math_divzerof(sign_bias)
                return (if sign_bias != 0 { -1.0f32 } else { 1.0 }) / 0.0;
            }
            return if iy & 0x8000_0000 != 0 { 1.0 / x2 } else { x2 };
        }
        // x and y are non-zero finite
        if ix & 0x8000_0000 != 0 {
            // finite x < 0
            let yint = checkint(iy);
            if yint == 0 {
                return (x - x) / (x - x); // __math_invalidf
            }
            if yint == 1 {
                sign_bias = SIGN_BIAS;
            }
            ix &= 0x7fff_ffff;
        }
        if ix < 0x0080_0000 {
            // normalise a subnormal x so its exponent becomes negative
            ix = (x * const { hf32("0x1p23") }).to_bits();
            ix &= 0x7fff_ffff;
            ix = ix.wrapping_sub(23 << 23);
        }
    }
    // y * log2(x) cannot overflow since y is single precision
    let ylogx = y as f64 * log2_inline(ix);
    // |y*log(x)| >= 126?
    if (ylogx.to_bits() >> 47 & 0xffff) >= 126.0f64.to_bits() >> 47 {
        if ylogx <= -150.0 {
            return xflowf(sign_bias != 0, const { hf32("0x1p-95") }); // __math_uflowf
        }
        if ylogx < -149.0 {
            return xflowf(sign_bias != 0, const { hf32("0x1.4p-75") }); // __math_may_uflowf
        }
        if ylogx > const { hf("0x1.fffffffa3aae2p+6") } {
            if ylogx > const { hf("0x1.fffffffd1d571p+6") } {
                return xflowf(sign_bias != 0, const { hf32("0x1p97") }); // __math_oflowf
            }
            // |x^y| > 0x1.fffffep127 and rounding is to nearest: the
            // largest finite value
            let max = const { hf32("0x1.fffffep127") };
            return if sign_bias != 0 { -max } else { max };
        }
    }
    exp2_inline(ylogx, sign_bias) as f32
}
