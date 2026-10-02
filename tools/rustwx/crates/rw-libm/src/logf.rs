//! binary32 natural logarithm as glibc 2.43 computes it: Arm
//! optimized-routines math/logf.c and math/logf_data.c, in glibc's copy
//! (sysdeps/ieee754/flt-32/e_logf.c, e_logf_data.c) and in the x86-64 FMA
//! IFUNC variant (`__logf_fma`): every multiply-add GCC fuses there is
//! `mul_add` here.  Notice in lib.rs.  Not correctly rounded (glibc's own
//! worst case is 0.82 ULP); bit for bit glibc 2.43's answer.

use crate::{hf, hf32};

/// (1/c, log(c)) near the centre of each of 16 subintervals of [OFF, 2*OFF].
const TAB: [[f64; 2]; 16] = [
    [hf("0x1.661ec79f8f3bep+0"), hf("-0x1.57bf7808caadep-2")],
    [hf("0x1.571ed4aaf883dp+0"), hf("-0x1.2bef0a7c06ddbp-2")],
    [hf("0x1.49539f0f010bp+0"), hf("-0x1.01eae7f513a67p-2")],
    [hf("0x1.3c995b0b80385p+0"), hf("-0x1.b31d8a68224e9p-3")],
    [hf("0x1.30d190c8864a5p+0"), hf("-0x1.6574f0ac07758p-3")],
    [hf("0x1.25e227b0b8eap+0"), hf("-0x1.1aa2bc79c81p-3")],
    [hf("0x1.1bb4a4a1a343fp+0"), hf("-0x1.a4e76ce8c0e5ep-4")],
    [hf("0x1.12358f08ae5bap+0"), hf("-0x1.1973c5a611cccp-4")],
    [hf("0x1.0953f419900a7p+0"), hf("-0x1.252f438e10c1ep-5")],
    [hf("0x1p+0"), hf("0x0p+0")],
    [hf("0x1.e608cfd9a47acp-1"), hf("0x1.aa5aa5df25984p-5")],
    [hf("0x1.ca4b31f026aap-1"), hf("0x1.c5e53aa362eb4p-4")],
    [hf("0x1.b2036576afce6p-1"), hf("0x1.526e57720db08p-3")],
    [hf("0x1.9c2d163a1aa2dp-1"), hf("0x1.bc2860d22477p-3")],
    [hf("0x1.886e6037841edp-1"), hf("0x1.1058bc8a07ee1p-2")],
    [hf("0x1.767dcf5534862p-1"), hf("0x1.4043057b6ee09p-2")],
];
const LN2: f64 = hf("0x1.62e42fefa39efp-1");
const A: [f64; 3] = [hf("-0x1.00ea348b88334p-2"), hf("0x1.5575b0be00b6ap-2"), hf("-0x1.ffffef20a4123p-2")];
const OFF: u32 = 0x3f33_0000;

/// log(x) for binary32, bit for bit as glibc 2.43 returns it.
#[inline]
pub fn logf(x: f32) -> f32 {
    let mut ix = x.to_bits();
    if ix == 0x3f80_0000 {
        return 0.0;
    }
    if ix.wrapping_sub(0x0080_0000) >= 0x7f80_0000 - 0x0080_0000 {
        // x < 0x1p-126 or infinity or NaN
        if ix.wrapping_mul(2) == 0 {
            return f32::NEG_INFINITY; // __math_divzerof(1)
        }
        if ix == 0x7f80_0000 {
            return x; // log(inf) = inf
        }
        if (ix & 0x8000_0000) != 0 || ix.wrapping_mul(2) >= 0xff00_0000 {
            return (x - x) / (x - x); // __math_invalidf
        }
        // subnormal: normalise
        ix = (x * const { hf32("0x1p23") }).to_bits();
        ix = ix.wrapping_sub(23 << 23);
    }
    // x = 2^k z with z in [OFF, 2*OFF] and exact
    let tmp = ix.wrapping_sub(OFF);
    let i = ((tmp >> (23 - 4)) % 16) as usize;
    let k = (tmp as i32) >> 23;
    let iz = ix.wrapping_sub(tmp & 0xff80_0000);
    let [invc, logc] = TAB[i];
    let z = f32::from_bits(iz) as f64;
    // log(x) = log1p(z/c - 1) + log(c) + k*Ln2
    let r = z.mul_add(invc, -1.0);
    let y0 = (k as f64).mul_add(LN2, logc);
    let r2 = r * r;
    let y = A[1].mul_add(r, A[2]);
    let y = A[0].mul_add(r2, y);
    let y = y.mul_add(r2, y0 + r);
    y as f32
}
