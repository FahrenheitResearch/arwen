// THIRD-PARTY NOTICE
//
// The function bodies in this crate are transcriptions of two bodies of
// third-party work, each under MIT, chosen function by function to be the
// code glibc 2.43 itself runs for that function.
//
// 1. Arm optimized-routines (https://github.com/ARM-software/optimized-routines),
//    as glibc 2.43 carries it in sysdeps/ieee754/flt-32/ and compiles it for
//    its x86-64 FMA IFUNC variants:
//
//      sinf, cosf  math/sinf.c, math/cosf.c, math/sincosf.h, math/sincosf_data.c
//      expf        math/expf.c, math/exp2f_data.c
//      logf        math/logf.c, math/logf_data.c
//      powf        math/powf.c, math/powf_log2_data.c, math/exp2f_data.c
//
//      Copyright (c) 2017-2018, Arm Limited.        (expf, logf, powf)
//      Copyright (c) 2018-2024, Arm Limited.        (sinf, cosf)
//      Copyright (c) 2018-2019, Arm Limited.        (sincosf data)
//      SPDX-License-Identifier: MIT
//
// 2. CORE-MATH (https://core-math.gitlabpages.inria.fr/), revision
//    284b3b0e198042c38f5c30316f696786b10816b0 of
//    https://gitlab.inria.fr/core-math/core-math, whose binary32 versions are
//    the ones glibc 2.43 distributes for these functions:
//
//      tanf        src/binary32/tan/tanf.c
//      atanf       src/binary32/atan/atanf.c
//      atan2f      src/binary32/atan2/atan2f.c
//      log10f      src/binary32/log10/log10f.c
//      asin, acos  src/binary64/asin/asin.c, src/binary64/acos/acos.c
//
//      Copyright (c) 2022-2025 Alexei Sibidanov.                     (tanf, atanf)
//      Copyright (c) 2022-2025 Alexei Sibidanov and Paul Zimmermann. (atan2f)
//      Copyright (c) 2022-2026 Alexei Sibidanov <sibid@uvic.ca>.     (log10f, asin)
//      Copyright (c) 2024-2025 Alexei Sibidanov.                     (acos)
//
// Both carry the same MIT permission notice:
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy
//   of this software and associated documentation files (the "Software"), to deal
//   in the Software without restriction, including without limitation the rights
//   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//   copies of the Software, and to permit persons to whom the Software is
//   furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in all
//   copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//   SOFTWARE.
//
// The texts are also at licenses/LICENSE-Arm-optimized-routines-MIT.txt and
// licenses/LICENSE-CORE-MATH-rw-libm-MIT.txt, and in the repository NOTICE
// ("Portable libm in Rust -- Arm optimized-routines and CORE-MATH").  glibc's
// own IBM binary64 asin and acos (LGPL) are not transcribed.

//! The C library's elementary functions, returning the same bits on every
//! platform.
//!
//! `f32::sin`, `f32::powf`, `f64::asin` and the rest call the platform C
//! library, and C libraries disagree in the last bit: glibc 2.43, glibc 2.39
//! and the Windows UCRT each return their own answer.  Code graded bit for
//! bit against an oracle captured with glibc 2.43 therefore passed on one
//! machine and failed on another, and its product output differed by
//! platform (public CI run 37036982597, the terrain-drag orographic
//! projection).
//!
//! Every binary32 function here returns exactly what glibc 2.43 returns on
//! an x86-64 machine with FMA, for every argument, on every platform:
//!
//! * `tanf`, `atanf`, `atan2f`, `log10f` are CORE-MATH's, which glibc 2.43
//!   distributes; they are correctly rounded.
//! * `sinf`, `cosf`, `expf`, `logf`, `powf` are Arm's, which glibc 2.43
//!   still uses; they are NOT correctly rounded (up to 0.82 ULP), and the
//!   WPS oracle bits carry their rounding.  A correctly rounded
//!   replacement would differ from glibc on 29.4 million sinf and 28.2
//!   million cosf arguments and fails the WPS oracle, so it is not used.
//! * `asin` and `acos` (binary64) are CORE-MATH's correctly rounded
//!   functions.  glibc's binary64 asin/acos are IBM's LGPL code and are not
//!   transcribed.  They are not correctly rounded on about 0.08 % (asin) and
//!   0.05 % (acos) of random arguments, but over 10^9 draws of the polar
//!   projection's arguments the binary32 results it keeps never differed.
//!
//! How the transcription stays platform independent:
//!
//! * only IEEE 754 basic operations (+, -, *, /, sqrt), which Rust never
//!   contracts or reassociates, and integer arithmetic;
//! * `f64::mul_add` wherever the C calls `__builtin_fma`, and wherever GCC
//!   fuses a multiply-add in glibc's FMA build: a fused multiply-add is
//!   correctly rounded by definition on every platform;
//! * the binary32 fused multiply-add CORE-MATH uses on a few tiny-argument
//!   paths is [`fmaf`] here, computed exactly through binary64 rather than
//!   through the platform `fmaf`;
//! * round-to-nearest-even of a small double by the 1.5 * 2^52 shift,
//!   instead of a `roundeven` library call;
//! * no floating-point environment: rounding is always to nearest, and the
//!   exception flags and errno the C code can maintain are not reproduced.
//!
//! Constants keep the upstream hexadecimal spelling through [`hf`], which
//! parses a C99 hex-float literal at compile time and refuses (as a build
//! error) any literal that would round.

mod asincos;
mod atan2f;
mod atanf;
mod expf;
mod log10f;
mod logf;
mod powf;
mod sincosf;
mod tanf;

pub use asincos::{acos, asin};
pub use atan2f::atan2f;
pub use atanf::atanf;
pub use expf::expf;
pub use log10f::log10f;
pub use logf::logf;
pub use powf::powf;
pub use sincosf::{cosf, sinf};
pub use tanf::tanf;

const fn hex_digit(c: u8) -> u64 {
    match c {
        b'0'..=b'9' => (c - b'0') as u64,
        b'a'..=b'f' => (c - b'a' + 10) as u64,
        b'A'..=b'F' => (c - b'A' + 10) as u64,
        _ => panic!("hf: not a hexadecimal digit"),
    }
}

/// The binary64 value of a C99 hexadecimal floating literal such as
/// `"-0x1.45f306dc9c883p+2"`, evaluated at compile time when used in a
/// `const`.  Only normal binary64 values and zero are accepted, and only
/// when the literal is exactly representable: a literal that would round
/// is a build error, never a silently different constant.
pub(crate) const fn hf(s: &str) -> f64 {
    let b = s.as_bytes();
    let mut i = 0;
    let mut negative = false;
    if b[0] == b'-' {
        negative = true;
        i = 1;
    } else if b[0] == b'+' {
        i = 1;
    }
    assert!(b[i] == b'0' && (b[i + 1] == b'x' || b[i + 1] == b'X'), "hf: no 0x prefix");
    i += 2;
    let mut mantissa: u64 = 0;
    let mut scale: i32 = 0;
    let mut fraction = false;
    while i < b.len() && b[i] != b'p' && b[i] != b'P' {
        if b[i] == b'.' {
            assert!(!fraction, "hf: two radix points");
            fraction = true;
        } else {
            assert!(mantissa >> 60 == 0, "hf: too many digits");
            mantissa = (mantissa << 4) | hex_digit(b[i]);
            if fraction {
                scale -= 4;
            }
        }
        i += 1;
    }
    assert!(i < b.len(), "hf: no binary exponent");
    i += 1;
    let mut exponent_negative = false;
    if b[i] == b'-' {
        exponent_negative = true;
        i += 1;
    } else if b[i] == b'+' {
        i += 1;
    }
    let mut exponent: i32 = 0;
    assert!(i < b.len(), "hf: empty exponent");
    while i < b.len() {
        assert!(b[i] >= b'0' && b[i] <= b'9', "hf: bad exponent digit");
        exponent = exponent * 10 + (b[i] - b'0') as i32;
        i += 1;
    }
    if exponent_negative {
        exponent = -exponent;
    }
    let sign = if negative { 1u64 << 63 } else { 0 };
    if mantissa == 0 {
        return f64::from_bits(sign);
    }
    let top = 63 - mantissa.leading_zeros() as i32;
    let bottom = mantissa.trailing_zeros() as i32;
    assert!(top - bottom < 53, "hf: literal is not exact in binary64");
    let unbiased = scale + exponent + top;
    assert!(unbiased >= -1022 && unbiased <= 1023, "hf: outside the normal binary64 range");
    let significand = if top >= 52 { mantissa >> (top - 52) } else { mantissa << (52 - top) };
    f64::from_bits(sign | (((unbiased + 1023) as u64) << 52) | (significand & ((1u64 << 52) - 1)))
}

/// The binary32 value of a hexadecimal literal that is exact in binary32.
pub(crate) const fn hf32(s: &str) -> f32 {
    let v = hf(s);
    let f = v as f32;
    assert!(f as f64 == v, "hf32: literal is not exact in binary32");
    f
}

/// The binary32 rounding of an exact binary64 hexadecimal literal: what a C
/// compiler makes of a `...f`-suffixed literal carrying more than 24 bits.
pub(crate) const fn hf32_rounded(s: &str) -> f32 {
    hf(s) as f32
}

/// Round a double to the nearest integer, ties to even, for |x| < 2^51.
///
/// Adding and removing 1.5 * 2^52 moves the binary point to the units
/// place, and the addition rounds to nearest-even like every IEEE
/// operation.  This is exact and needs no library call; CORE-MATH's
/// `roundeven_finite` only ever sees arguments well inside the range.
#[inline(always)]
pub(crate) fn roundeven(x: f64) -> f64 {
    const SHIFT: f64 = hf("0x1.8p52");
    debug_assert!(x.abs() < const { hf("0x1p51") });
    (x + SHIFT) - SHIFT
}

/// Correctly rounded binary32 fused multiply-add, rounding to nearest.
///
/// The product of two binary32 values is exact in binary64, so only the
/// addition rounds; when that rounding lands exactly halfway between two
/// binary32 values the second rounding could go the wrong way, and the
/// halfway case is nudged toward the exact sum (the musl method).  Used by
/// CORE-MATH only on tiny arguments, where the sum is dominated by `z`.
#[inline]
pub(crate) fn fmaf(x: f32, y: f32, z: f32) -> f32 {
    let xy = x as f64 * y as f64;
    let zd = z as f64;
    let result = xy + zd;
    let mut u = result.to_bits();
    let e = (u >> 52) & 0x7ff;
    if (u & 0x1fff_ffff) != 0x1000_0000
        || e == 0x7ff
        || (result - xy == zd && result - zd == xy)
    {
        return result as f32;
    }
    let negative = (u >> 63) != 0;
    let err = if negative == (zd > xy) { xy - result + zd } else { zd - result + xy };
    if negative == (err < 0.0) {
        u += 1;
    } else {
        u -= 1;
    }
    f64::from_bits(u) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_literals_parse_exactly() {
        assert_eq!(const { hf("0x1p+0") }, 1.0);
        assert_eq!(const { hf("-0x1.8p52") }.to_bits(), (-6755399441055744.0f64).to_bits());
        assert_eq!(const { hf("0x1.921fb54442d18p+1") }, std::f64::consts::PI);
        assert_eq!(const { hf("0x1.62e42fefa39efp-1") }, std::f64::consts::LN_2);
        assert_eq!(const { hf("0x1.71547652b82fep+0") }, std::f64::consts::LOG2_E);
        assert_eq!(const { hf("0x0p+0") }.to_bits(), 0);
        assert_eq!(const { hf("-0x0p+0") }.to_bits(), 1u64 << 63);
        assert_eq!(const { hf("0x1p-1022") }, f64::MIN_POSITIVE);
        assert_eq!(const { hf("0x1.fffffffffffffp+1023") }, f64::MAX);
        assert_eq!(const { hf("0x1.f7d70599926c4p-98") }.to_bits(), 0x39df7d70599926c4);
        assert_eq!(const { hf32("0x1.555556p-2") }.to_bits(), 0x3eaaaaab);
        assert_eq!(const { hf32("0x1p-149") }.to_bits(), 1);
        assert_eq!(const { hf32_rounded("-0x1.5555555555555p-2") }.to_bits(), 0xbeaaaaab);
    }

    #[test]
    fn roundeven_breaks_ties_to_even() {
        for (x, r) in [(0.5, 0.0), (1.5, 2.0), (2.5, 2.0), (-0.5, -0.0), (-2.5, -2.0), (3.49, 3.0)] {
            assert_eq!(roundeven(x), r, "{x}");
        }
        assert_eq!(roundeven(const { hf("0x1.fffffffffffffp+49") }), const { hf("0x1p+50") });
        assert_eq!(roundeven(const { hf("0x1.0000000000002p+50") }), const { hf("0x1p+50") });
        assert_eq!(roundeven(const { hf("0x1.0000000000006p+50") }), const { hf("0x1.0000000000008p+50") });
    }

    #[test]
    fn fmaf_rounds_once() {
        // 1 + 2^-24 is a binary32 halfway case; the tail decides the direction.
        let one = 1.0f32;
        let half_ulp = const { hf32("0x1p-24") };
        let odd = const { hf32("0x1.000002p+0") };
        assert_eq!(fmaf(one, one, half_ulp), 1.0);
        assert_eq!(fmaf(odd, one, half_ulp), const { hf32("0x1.000004p+0") });
        // odd + 2^-24 - 2^-60 is just below the midpoint above `odd`, but its
        // binary64 rounding IS that midpoint, which ties-to-even would carry
        // up: the single rounding stays at `odd`.
        let x = const { hf32("0x1.00004p-24") };
        let y = const { hf32("0x1.ffff8p-1") };
        assert_eq!(fmaf(x, y, odd), odd);
        assert_eq!(fmaf(-x, y, -odd), -odd);
        assert_eq!(fmaf(0.0, -0.0, -0.0).to_bits(), (-0.0f32).to_bits());
        assert!(fmaf(f32::INFINITY, 0.0, 1.0).is_nan());
    }
}
