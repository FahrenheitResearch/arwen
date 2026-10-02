//! Correctly rounded binary32 two-argument arctangent: CORE-MATH
//! src/binary32/atan2/atan2f.c (notice in lib.rs).

use crate::hf;

/// Double-double product (xh + xl) * (ch + cl).
#[inline(always)]
fn muldd(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let ahlh = ch * xl;
    let alhh = cl * xh;
    let ahhh = ch * xh;
    let mut ahhl = ch.mul_add(xh, -ahhh);
    ahhl += alhh + ahlh;
    let h = ahhh + ahhl;
    (h, (ahhh - h) + ahhl)
}

/// Horner evaluation of a double-double polynomial at xh + xl.
fn polydd(xh: f64, xl: f64, c: &[[f64; 2]]) -> (f64, f64) {
    let mut i = c.len() - 1;
    let mut ch = c[i][0];
    let mut cl = c[i][1];
    while i > 0 {
        i -= 1;
        (ch, cl) = muldd(xh, xl, ch, cl);
        let th = ch + c[i][0];
        let tl = (c[i][0] - th) + ch;
        ch = th;
        cl += tl + c[i][1];
    }
    (ch, cl)
}

/// atan2 for tiny y/x: the Taylor approximation z - z^3/3, z = y/x.
#[inline(never)]
#[cold]
fn atan2f_tiny(y: f32, x: f32) -> f32 {
    let dy = y as f64;
    let dx = x as f64;
    let z = dy / dx;
    let mut e = (-z).mul_add(dx, dy);
    // z * x + e = y thus y/x = z + e/x
    const C: f64 = hf("-0x1.5555555555555p-2"); // -1/3 rounded to nearest
    let zz = z * z;
    let cz = C * z;
    e = e / dx + cz * zz;
    let mut t = z.to_bits();
    if t & 0xfff_ffff == 0 {
        // boundary case: move the significand by one toward the true value
        // to avoid a double-rounding error when rounding to binary32
        if z * e > 0.0 {
            t = t.wrapping_add(1);
        } else {
            t = t.wrapping_sub(1);
        }
    }
    f64::from_bits(t) as f32
}

const PI: f64 = hf("0x1.921fb54442d18p+1");
const PI2: f64 = hf("0x1.921fb54442d18p+0");
const PI2L: f64 = hf("0x1.1a62633145c07p-54");

/// Correctly rounded atan2(y, x) for binary32 (round to nearest).
#[inline]
pub fn atan2f(y: f32, x: f32) -> f32 {
    const CN: [f64; 7] = [
        hf("0x1p+0"),
        hf("0x1.40e0698f94c35p+1"),
        hf("0x1.248c5da347f0dp+1"),
        hf("0x1.d873386572976p-1"),
        hf("0x1.46fa40b20f1dp-3"),
        hf("0x1.33f5e041eed0fp-7"),
        hf("0x1.546bbf28667c5p-14"),
    ];
    const CD: [f64; 7] = [
        hf("0x1p+0"),
        hf("0x1.6b8b143a3f6dap+1"),
        hf("0x1.8421201d18ed5p+1"),
        hf("0x1.8221d086914ebp+0"),
        hf("0x1.670657e3a07bap-2"),
        hf("0x1.0f4951fd1e72dp-5"),
        hf("0x1.b3874b8798286p-11"),
    ];
    const M: [f64; 2] = [0.0, 1.0];
    const OFF: [f64; 8] = [0.0, PI2, PI, PI2, -0.0, -PI2, -PI, -PI2];
    const OFFL: [f64; 8] = [0.0, PI2L, 2.0 * PI2L, PI2L, -0.0, -PI2L, -2.0 * PI2L, -PI2L];
    const SGN: [f64; 2] = [1.0, -1.0];
    let ux = x.to_bits();
    let uy = y.to_bits();
    let ax = ux & (!0u32 >> 1);
    let ay = uy & (!0u32 >> 1);
    if ay >= 0xff << 23 || ax >= 0xff << 23 {
        // x or y is NaN or infinity
        if ay > 0xff << 23 {
            return x + y; // y NaN
        }
        if ax > 0xff << 23 {
            return x + y; // x NaN
        }
        let yinf = ay == 0xff << 23;
        let xinf = ax == 0xff << 23;
        if yinf && xinf {
            return if ux >> 31 != 0 {
                (const { hf("0x1.2d97c7f3321d2p+1") } * SGN[(uy >> 31) as usize]) as f32 // +/-3pi/4
            } else {
                (const { hf("0x1.921fb54442d18p-1") } * SGN[(uy >> 31) as usize]) as f32 // +/-pi/4
            };
        }
        if xinf {
            return if ux >> 31 != 0 {
                (PI * SGN[(uy >> 31) as usize]) as f32
            } else {
                (0.0 * SGN[(uy >> 31) as usize]) as f32
            };
        }
        if yinf {
            return (PI2 * SGN[(uy >> 31) as usize]) as f32;
        }
    }
    if ay == 0 {
        if ax == 0 {
            let i = ((uy >> 31) * 4 + (ux >> 31) * 2) as usize;
            return if ux >> 31 != 0 { (OFF[i] + OFFL[i]) as f32 } else { OFF[i] as f32 };
        }
        if ux >> 31 == 0 {
            return (0.0 * SGN[(uy >> 31) as usize]) as f32;
        }
    }
    let gt = (ay > ax) as usize;
    let i = ((uy >> 31) * 4 + (ux >> 31) * 2) as usize + gt;

    let zx = x as f64;
    let zy = y as f64;
    // z = x/y if |y| > |x|, and z = y/x otherwise
    let mut z = (M[gt] * zx + M[1 - gt] * zy) / (M[gt] * zy + M[1 - gt] * zx);
    let mut r;
    let d = ax as i32 - ay as i32;
    if d < (27 << 23) && d > -(27 << 23) {
        let z2 = z * z;
        let z4 = z2 * z2;
        let z8 = z4 * z4;
        let mut cn0 = CN[0] + z2 * CN[1];
        let cn2 = CN[2] + z2 * CN[3];
        let mut cn4 = CN[4] + z2 * CN[5];
        let cn6 = CN[6];
        cn0 += z4 * cn2;
        cn4 += z4 * cn6;
        cn0 += z8 * cn4;
        let mut cd0 = CD[0] + z2 * CD[1];
        let cd2 = CD[2] + z2 * CD[3];
        let mut cd4 = CD[4] + z2 * CD[5];
        let cd6 = CD[6];
        cd0 += z4 * cd2;
        cd4 += z4 * cd6;
        cd0 += z8 * cd4;
        r = cn0 / cd0;
    } else {
        r = 1.0;
    }
    z *= SGN[gt];
    r = z * r + OFF[i];
    if (r.to_bits().wrapping_add(8) & 0xfff_ffff) <= 16 {
        // check tiny y/x
        if ay < ax && ((ax - ay) >> 23 >= 25) {
            return atan2f_tiny(y, x);
        }
        let (mut zh, mut zl);
        if gt == 0 {
            zh = zy / zx;
            zl = zh.mul_add(-zx, zy) / zx;
        } else {
            zh = zx / zy;
            zl = zh.mul_add(-zy, zx) / zy;
        }
        let (z2h, z2l) = muldd(zh, zl, zh, zl);
        let (mut ph, mut pl) = polydd(z2h, z2l, &C32);
        zh *= SGN[gt];
        zl *= SGN[gt];
        (ph, pl) = muldd(zh, zl, ph, pl);
        let sh = ph + OFF[i];
        let sl = ((OFF[i] - sh) + ph) + pl + OFFL[i];
        let rf = sh as f32;
        let th = rf as f64;
        let dh = sh - th;
        let mut tm = dh + sl;
        if th + th * const { hf("0x1p-60") } == th - th * const { hf("0x1p-60") } {
            let mut tth = th.to_bits();
            tth &= 0x7ff << 52;
            tth = tth.wrapping_sub(24 << 52);
            if tm.abs() > f64::from_bits(tth) {
                tm *= 1.25;
            } else {
                tm *= 0.75;
            }
        }
        r = th + tm;
    }
    r as f32
}

/// Double-double Taylor coefficients of atan(sqrt(t))/sqrt(t) (atan2f.c `c`).
const C32: [[f64; 2]; 32] = [
    [hf("0x1p+0"), hf("-0x1.8c1dac5492248p-87")],
    [hf("-0x1.5555555555555p-2"), hf("-0x1.55553bf3a2abep-56")],
    [hf("0x1.999999999999ap-3"), hf("-0x1.99deed1ec9071p-57")],
    [hf("-0x1.2492492492492p-3"), hf("-0x1.fd99c8d18269ap-58")],
    [hf("0x1.c71c71c71c717p-4"), hf("-0x1.651eee4c4d9dp-61")],
    [hf("-0x1.745d1745d1649p-4"), hf("-0x1.632683d6c44a6p-58")],
    [hf("0x1.3b13b13b11c63p-4"), hf("0x1.bf69c1f8af41dp-58")],
    [hf("-0x1.11111110e6338p-4"), hf("0x1.3c3e431e8bb68p-61")],
    [hf("0x1.e1e1e1dc45c4ap-5"), hf("-0x1.be2db05c77bbfp-59")],
    [hf("-0x1.af286b8164b4fp-5"), hf("0x1.a4673491f0942p-61")],
    [hf("0x1.86185e9ad4846p-5"), hf("0x1.e12e32d79fceep-59")],
    [hf("-0x1.642c6d5161faep-5"), hf("0x1.3ce76c1ca03fp-59")],
    [hf("0x1.47ad6f277e5bfp-5"), hf("-0x1.abd8d85bdb714p-60")],
    [hf("-0x1.2f64a2ee8896dp-5"), hf("0x1.ef87d4b615323p-61")],
    [hf("0x1.1a6a2b31741b5p-5"), hf("0x1.a5d9d973547eep-62")],
    [hf("-0x1.07fbdad65e0a6p-5"), hf("-0x1.65ac07f5d35f4p-61")],
    [hf("0x1.ee9932a9a5f8bp-6"), hf("0x1.f8b9623f6f55ap-61")],
    [hf("-0x1.ce8b5b9584dc6p-6"), hf("0x1.fe5af96e8ea2dp-61")],
    [hf("0x1.ac9cb288087b7p-6"), hf("-0x1.450cdfceaf5cap-60")],
    [hf("-0x1.84b025351f3e6p-6"), hf("0x1.579561b0d73dap-61")],
    [hf("0x1.52f5b8ecdd52bp-6"), hf("0x1.036bd2c6fba47p-60")],
    [hf("-0x1.163a8c44909dcp-6"), hf("0x1.18f735ffb9f16p-60")],
    [hf("0x1.a400dce3eea6fp-7"), hf("-0x1.c90569c0c1b5cp-61")],
    [hf("-0x1.1caa78ae6db3ap-7"), hf("-0x1.4c60f8161ea09p-61")],
    [hf("0x1.52672453c0731p-8"), hf("0x1.834efb598c338p-62")],
    [hf("-0x1.5850c5be137cfp-9"), hf("-0x1.445fc150ca7f5p-63")],
    [hf("0x1.23eb98d22e1cap-10"), hf("-0x1.388fbaf1d783p-64")],
    [hf("-0x1.8f4e974a40741p-12"), hf("0x1.271198a97da34p-66")],
    [hf("0x1.a5cf2e9cf76e5p-14"), hf("-0x1.887eb4a63b665p-68")],
    [hf("-0x1.420c270719e32p-16"), hf("0x1.efd595b27888bp-71")],
    [hf("0x1.3ba2d69b51677p-19"), hf("-0x1.4fb06829cdfc7p-73")],
    [hf("-0x1.29b7e6f676385p-23"), hf("-0x1.a783b6de718fbp-77")],
];
