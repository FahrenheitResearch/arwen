//! binary32 sine and cosine as glibc 2.43 computes them: Arm
//! optimized-routines math/sinf.c, math/cosf.c, math/sincosf.h and
//! math/sincosf_data.c, in glibc's copy (sysdeps/ieee754/flt-32/s_sinf.c,
//! s_cosf.c, s_sincosf.h, sincosf_poly.h, s_sincosf_data.c) and in the x86-64
//! FMA IFUNC variant glibc selects on an FMA machine (`__sinf_fma`,
//! `__cosf_fma`): every multiply-add GCC fuses there is `mul_add` here.
//! Notice in lib.rs.  Not correctly rounded (glibc's own worst case is 0.56
//! ULP); bit for bit glibc 2.43's answer, on every platform.

use crate::{hf, hf32};

struct SinCos {
    sign: [f64; 4],
    hpi_inv: f64,
    hpi: f64,
    c0: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    s1: f64,
    s2: f64,
    s3: f64,
}

/// The second entry computes -cos(x) to get the negation for free.
const TABLE: [SinCos; 2] = [
    SinCos {
        sign: [1.0, -1.0, -1.0, 1.0],
        hpi_inv: hf("0x1.45F306DC9C883p+23"),
        hpi: hf("0x1.921FB54442D18p0"),
        c0: hf("0x1p0"),
        c1: hf("-0x1.ffffffd0c621cp-2"),
        c2: hf("0x1.55553e1068f19p-5"),
        c3: hf("-0x1.6c087e89a359dp-10"),
        c4: hf("0x1.99343027bf8c3p-16"),
        s1: hf("-0x1.555545995a603p-3"),
        s2: hf("0x1.1107605230bc4p-7"),
        s3: hf("-0x1.994eb3774cf24p-13"),
    },
    SinCos {
        sign: [1.0, -1.0, -1.0, 1.0],
        hpi_inv: hf("0x1.45F306DC9C883p+23"),
        hpi: hf("0x1.921FB54442D18p0"),
        c0: hf("-0x1p0"),
        c1: hf("0x1.ffffffd0c621cp-2"),
        c2: hf("-0x1.55553e1068f19p-5"),
        c3: hf("0x1.6c087e89a359dp-10"),
        c4: hf("-0x1.99343027bf8c3p-16"),
        s1: hf("-0x1.555545995a603p-3"),
        s2: hf("0x1.1107605230bc4p-7"),
        s3: hf("-0x1.994eb3774cf24p-13"),
    },
];

/// 4/pi to 192 bits, eight new bits per entry.
const INV_PIO4: [u32; 24] = [
    0xa2, 0xa2f9, 0xa2f983, 0xa2f9836e, 0xf9836e4e, 0x836e4e44, 0x6e4e4415, 0x4e441529,
    0x441529fc, 0x1529fc27, 0x29fc2757, 0xfc2757d1, 0x2757d1f5, 0x57d1f534, 0xd1f534dd, 0xf534ddc0,
    0x34ddc0db, 0xddc0db62, 0xc0db6295, 0xdb629599, 0x6295993c, 0x95993c43, 0x993c4390, 0x3c439041,
];

/// 2pi * 2^-64.
const PI63: f64 = hf("0x1.921FB54442D18p-62");

/// Top 12 bits of the encoding with the sign cleared.
const fn abstop12(x: f32) -> u32 {
    (x.to_bits() >> 20) & 0x7ff
}
const TOP_PIO4: u32 = abstop12(hf32("0x1.921FB6p-1"));
const TOP_TINY: u32 = abstop12(hf32("0x1p-12"));
const TOP_120: u32 = abstop12(120.0);
const TOP_INF: u32 = abstop12(f32::INFINITY);

/// sinf_poly: the sine polynomial for even quadrants, the cosine one for odd.
#[inline(always)]
fn sinf_poly(x: f64, x2: f64, p: &SinCos, n: i32) -> f32 {
    if n & 1 == 0 {
        let x3 = x * x2;
        let s1 = x2.mul_add(p.s3, p.s2);
        let x7 = x3 * x2;
        let s = x3.mul_add(p.s1, x);
        x7.mul_add(s1, s) as f32
    } else {
        let x4 = x2 * x2;
        let c2 = x2.mul_add(p.c4, p.c3);
        let c1 = x2.mul_add(p.c1, p.c0);
        let x6 = x4 * x2;
        let c = x4.mul_add(p.c2, c1);
        x6.mul_add(c2, c) as f32
    }
}

/// One multiply-subtract reduction, accurate for |x| <= 120.
#[inline(always)]
fn reduce_fast(x: f64, p: &SinCos) -> (f64, i32) {
    let r = x * p.hpi_inv;
    // hpi_inv is prescaled by 2^24, so the quadrant lands in bits 24..31
    let n = ((r as i32) + 0x80_0000) >> 24;
    ((-(n as f64)).mul_add(p.hpi, x), n)
}

/// 32x96->128-bit reduction modulo pi/2 for |x| >= 120 (sign ignored).
#[inline(always)]
fn reduce_large(xi: u32) -> (f64, i32) {
    let arr = &INV_PIO4[((xi >> 26) & 15) as usize..];
    let shift = (xi >> 23) & 7;
    let xi = ((xi & 0xff_ffff) | 0x80_0000) << shift;
    let mut res0 = xi.wrapping_mul(arr[0]) as u64;
    let res1 = xi as u64 * arr[4] as u64;
    let res2 = xi as u64 * arr[8] as u64;
    res0 = (res2 >> 32) | (res0 << 32);
    res0 = res0.wrapping_add(res1);
    let n = res0.wrapping_add(1 << 61) >> 62;
    res0 = res0.wrapping_sub(n << 62);
    let x = res0 as i64 as f64;
    (x * PI63, n as i32)
}

#[inline(always)]
fn sincos(y: f32, cosine: i32) -> f32 {
    let x = y as f64;
    let top = abstop12(y);
    let p = &TABLE[0];
    if top < TOP_PIO4 {
        let x2 = x * x;
        if top < TOP_TINY {
            return if cosine != 0 { 1.0 } else { y };
        }
        return sinf_poly(x, x2, p, cosine);
    }
    let (x, n, quadrant) = if top < TOP_120 {
        let (x, n) = reduce_fast(x, p);
        (x, n, n)
    } else if top < TOP_INF {
        let xi = y.to_bits();
        let sign = (xi >> 31) as i32;
        let (x, n) = reduce_large(xi);
        // include the original sign
        (x, n, n + sign)
    } else {
        return (y - y) / (y - y); // infinity or NaN: invalid
    };
    let s = p.sign[(quadrant & 3) as usize];
    let p = if quadrant & 2 != 0 { &TABLE[1] } else { p };
    sinf_poly(x * s, x * x, p, n ^ cosine)
}

/// sin(x) for binary32, bit for bit as glibc 2.43 returns it.
#[inline]
pub fn sinf(x: f32) -> f32 {
    sincos(x, 0)
}

/// cos(x) for binary32, bit for bit as glibc 2.43 returns it.
#[inline]
pub fn cosf(x: f32) -> f32 {
    sincos(x, 1)
}
