//! Correctly rounded binary32 arctangent: CORE-MATH
//! src/binary32/atan/atanf.c (notice in lib.rs).

use crate::{fmaf, hf, hf32_rounded};

/// Correctly rounded atan(x) for binary32 (round to nearest).
#[inline]
pub fn atanf(x: f32) -> f32 {
    const PI2: f64 = hf("0x1.921fb54442d18p+0");
    let t = x.to_bits();
    let e = ((t >> 23) & 0xff) as i32;
    let gt = e >= 127;
    let ta = t & 0x7fff_ffff;
    if ta >= 0x4c70_0518 {
        // |x| >= 0x1.e00a3p+25
        if ta > 0x7f80_0000 {
            return x + x; // NaN
        }
        return PI2.copysign(x as f64) as f32;
    }
    if e < 127 - 13 {
        // |x| < 2^-13
        if e < 127 - 25 {
            // |x| < 2^-25
            if t << 1 == 0 {
                return x;
            }
            return fmaf(-x, x.abs(), x);
        }
        // C: __builtin_fmaf(-0x1.5555555555555p-2f*x, x*x, x), where the
        // suffixed literal is that value rounded to binary32.
        return fmaf(const { hf32_rounded("-0x1.5555555555555p-2") } * x, x * x, x);
    }
    // now |x| >= 0x1p-13
    let mut z = x as f64;
    if gt {
        z = 1.0 / z;
    }
    let z2 = z * z;
    let z4 = z2 * z2;
    let z8 = z4 * z4;
    // Rational approximation from rminimax; CN[0] is the original value and
    // CD[0] was slightly reduced upstream to avoid an exceptional case.
    const CN: [f64; 7] = [
        hf("0x1.51eccde075d67p-2"),
        hf("0x1.a76bb5637f2f2p-1"),
        hf("0x1.81e0eed20de88p-1"),
        hf("0x1.376c8ca67d11dp-2"),
        hf("0x1.aec7b69202ac6p-5"),
        hf("0x1.9561899acc73ep-9"),
        hf("0x1.bf9fa5b67e6p-16"),
    ];
    const CD: [f64; 7] = [
        hf("0x1.51eccde075d66p-2"),
        hf("0x1.dfbdd7b392d28p-1"),
        hf("0x1p+0"),
        hf("0x1.fd22bf0e89b54p-2"),
        hf("0x1.d91ff8b576282p-4"),
        hf("0x1.653ea99fc9bbp-7"),
        hf("0x1.1e7fcc202340ap-12"),
    ];
    let mut cn0 = CN[0] + z2 * CN[1];
    let cn2 = CN[2] + z2 * CN[3];
    let mut cn4 = CN[4] + z2 * CN[5];
    let cn6 = CN[6];
    cn0 += z4 * cn2;
    cn4 += z4 * cn6;
    cn0 += z8 * cn4;
    cn0 *= z;
    let mut cd0 = CD[0] + z2 * CD[1];
    let cd2 = CD[2] + z2 * CD[3];
    let mut cd4 = CD[4] + z2 * CD[5];
    let cd6 = CD[6];
    cd0 += z4 * cd2;
    cd4 += z4 * cd6;
    cd0 += z8 * cd4;
    let r = cn0 / cd0;
    if !gt {
        return r as f32; // for |x| < 1, (float) r is correctly rounded
    }
    // r approximates atan(1/x); atan(x) + atan(1/x) = sign(x) * pi/2.
    const PI_OVER2_H: f64 = hf("0x1.9p0");
    const PI_OVER2_L: f64 = hf("0x1.0fdaa22168c23p-7");
    let r = (PI_OVER2_L.copysign(z) - r) + PI_OVER2_H.copysign(z);
    r as f32
}
