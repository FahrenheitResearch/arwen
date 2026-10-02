//! Correctly rounded binary32 tangent: CORE-MATH src/binary32/tan/tanf.c
//! (notice in lib.rs).

use crate::{fmaf, hf, hf32, roundeven};

/// Reduction for |x| < 2^28: 2/pi * x = q + r.
#[inline(always)]
fn rltl(z: f32) -> (f64, i32) {
    let x = z as f64;
    // idh is representable on 28 bits and x on 24, so idh * x is exact.
    let idl = const { hf("-0x1.b1bbead603d8bp-32") } * x;
    let idh = const { hf("0x1.45f306ep-1") } * x;
    let id = roundeven(idh);
    ((idh - id) + idl, id as i64 as i32)
}

/// Payne-Hanek reduction for |x| >= 2^28 (tan variant, k = e - 127).
#[inline(never)]
fn rbig(u: u32) -> (f64, i32) {
    const IPI: [u64; 4] =
        [0xfe5163abdebbc562, 0xdb6295993c439041, 0xfc2757d1f534ddc0, 0xa2f9836e4e441529];
    let e = ((u >> 23) & 0xff) as i32;
    let m = ((u & (!0u32 >> 9)) | (1 << 23)) as u64;
    let p0 = m as u128 * IPI[0] as u128;
    let p1 = m as u128 * IPI[1] as u128 + (p0 >> 64);
    let p2 = m as u128 * IPI[2] as u128 + (p1 >> 64);
    let p3 = m as u128 * IPI[3] as u128 + (p2 >> 64);
    let (p3h, p3l, p2l, p1l) = ((p3 >> 64) as u64, p3 as u64, p2 as u64, p1 as u64);
    let k = e - 127;
    let s = (k - 23) as u32;
    let (mut i, a): (i32, i64) = if s < 64 {
        ((p3h << s | p3l >> (64 - s)) as i32, (p3l << s | p2l >> (64 - s)) as i64)
    } else if s == 64 {
        (p3l as i32, p2l as i64)
    } else {
        ((p3l << (s - 64) | p2l >> (128 - s)) as i32, (p2l << (s - 64) | p1l >> (128 - s)) as i64)
    };
    let sgn = (u as i32) >> 31;
    let sm = a >> 63;
    i = (i as i64 - sm) as i32;
    let z = (a ^ sgn as i64) as f64 * const { hf("0x1p-64") };
    i = (i ^ sgn).wrapping_sub(sgn);
    (z, i)
}

/// Correctly rounded tan(x) for binary32 (round to nearest).
#[inline]
pub fn tanf(x: f32) -> f32 {
    let t = x.to_bits();
    let e = ((t >> 23) & 0xff) as i32;
    let (z, i);
    if e < 127 + 28 {
        // |x| < 2^28
        if e < 115 {
            // |x| < 2^-13
            if e < 102 {
                // |x| < 2^-26
                return fmaf(x, x.abs(), x);
            }
            let x2 = x * x;
            return fmaf(x, const { hf32("0x1.555556p-2") } * x2, x);
        }
        (z, i) = rltl(x);
    } else if e < 0xff {
        (z, i) = rbig(t);
    } else {
        if t << 9 != 0 {
            return x + x; // NaN
        }
        return f32::NAN; // infinity
    }
    let z2 = z * z;
    let z4 = z2 * z2;
    const CN: [f64; 4] = [
        hf("0x1.921fb54442d18p+0"),
        hf("-0x1.fd226e573289fp-2"),
        hf("0x1.b7a60c8dac9f6p-6"),
        hf("-0x1.725beb40f33e5p-13"),
    ];
    const CD: [f64; 4] = [
        hf("0x1p+0"),
        hf("-0x1.2395347fb829dp+0"),
        hf("0x1.2313660f29c36p-3"),
        hf("-0x1.9a707ab98d1c1p-9"),
    ];
    const S: [f64; 2] = [0.0, 1.0];
    let mut n = CN[0] + z2 * CN[1];
    let n2 = CN[2] + z2 * CN[3];
    n += z4 * n2;
    let mut d = CD[0] + z2 * CD[1];
    let d2 = CD[2] + z2 * CD[3];
    d += z4 * d2;
    n *= z;
    let s0 = S[(i & 1) as usize];
    let s1 = S[(1 - (i & 1)) as usize];
    let r1 = (n * s1 - d * s0) / (n * s0 + d * s1);
    let tail = r1.to_bits().wrapping_add(7) & (u64::MAX >> 35);
    if tail <= 14 {
        const ST: [(f32, f32, f32); 8] = [
            (hf32("0x1.143ec4p+0"), hf32("0x1.ddf9f6p+0"), hf32("-0x1.891d24p-52")),
            (hf32("0x1.ada6aap+27"), hf32("0x1.e80304p-3"), hf32("0x1.419f46p-58")),
            (hf32("0x1.af61dap+48"), hf32("0x1.60d1c8p-2"), hf32("-0x1.2d6c3ap-55")),
            (hf32("0x1.0088bcp+52"), hf32("0x1.ca1edp+0"), hf32("0x1.f6053p-53")),
            (hf32("0x1.f90dfcp+72"), hf32("0x1.597f9cp-1"), hf32("0x1.925978p-53")),
            (hf32("0x1.cc4e22p+85"), hf32("-0x1.f33584p+1"), hf32("0x1.d7254ap-51")),
            (hf32("0x1.a6ce12p+86"), hf32("-0x1.c5612ep-1"), hf32("-0x1.26c33ep-53")),
            (hf32("0x1.6a0b76p+102"), hf32("-0x1.e42a1ep+0"), hf32("-0x1.1dc906p-52")),
        ];
        let ax = t & (!0u32 >> 1);
        let negative = t >> 31 != 0;
        for (arg, rh, rl) in ST {
            if arg.to_bits() == ax {
                return if negative { -rh - rl } else { rh + rl };
            }
        }
    }
    r1 as f32
}
