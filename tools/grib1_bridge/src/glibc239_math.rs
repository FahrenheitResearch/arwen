//! Deterministic glibc 2.39 powf used by host initialization.
//! Exact Rust transcription of gpuwm/core/noahmp_libm.py, with no FMA
//! contraction and no dependence on the host C library. Zero condensate
//! cells take this native path instead of scalar Python calls.
//
// Copyright (c) 2017-2018, Arm Limited.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the Software),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

use std::panic::{catch_unwind, AssertUnwindSafe};
use crate::{ERR_DIMENSION, ERR_NULL, ERR_PANIC, OK};

const EXP2_TAB: [u64; 32] = [
    0x3ff0000000000000,
    0x3fefd9b0d3158574,
    0x3fefb5586cf9890f,
    0x3fef9301d0125b51,
    0x3fef72b83c7d517b,
    0x3fef54873168b9aa,
    0x3fef387a6e756238,
    0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715,
    0x3feef1a7373aa9cb,
    0x3feedea64c123422,
    0x3feece086061892d,
    0x3feebfdad5362a27,
    0x3feeb42b569d4f82,
    0x3feeab07dd485429,
    0x3feea47eb03a5585,
    0x3feea09e667f3bcd,
    0x3fee9f75e8ec5f74,
    0x3feea11473eb0187,
    0x3feea589994cce13,
    0x3feeace5422aa0db,
    0x3feeb737b0cdc5e5,
    0x3feec49182a3f090,
    0x3feed503b23e255d,
    0x3feee89f995ad3ad,
    0x3feeff76f2fb5e47,
    0x3fef199bdd85529c,
    0x3fef3720dcef9069,
    0x3fef5818dcfba487,
    0x3fef7c97337b9b5f,
    0x3fefa4afa2a490da,
    0x3fefd0765b6e4540,
];
const EXP2_POLY: [f64; 3] = [0.05550361559341535, 0.2402284522445722, 0.6931471806916203];
const POW_TAB: [(f64, f64); 16] = [
    (1.398907162146528, -0.48430022186289673),
    (1.3403141896637998, -0.42257122959194704),
    (1.286432210124115, -0.3633754347673556),
    (1.2367150214269895, -0.30651309567405577),
    (1.1906977166711752, -0.25180720160537634),
    (1.1479821020556429, -0.19910014943794563),
    (1.1082251448272158, -0.14825100623281615),
    (1.0711297413057381, -0.09913323807318392),
    (1.036437278977283, -0.051632812977629436),
    (1.0, 0.0),
    (0.9492859795739057, 0.07508531937943004),
    (0.8951049428609004, 0.15987125980713107),
    (0.8476821620351103, 0.2384046664317681),
    (0.8050314851692001, 0.31288288605863257),
    (0.7664671008843108, 0.38370422656453185),
    (0.731428603316328, 0.451211048935815),
];
const POW_A: [f64; 5] = [0.288457581109214, -0.36092606229713164, 0.480898481472577, -0.7213474675006291, 1.4426950408774342];

#[inline]
fn log2(ix: u32) -> f64 {
    let tmp = ix.wrapping_sub(0x3f33_0000);
    let i = ((tmp >> 19) % 16) as usize;
    let top = tmp & 0xff80_0000;
    let iz = ix.wrapping_sub(top);
    let k = (top as i32) >> 23;
    let (invc, logc) = POW_TAB[i];
    let z = f32::from_bits(iz) as f64;
    let r = z * invc - 1.0;
    let y0 = logc + k as f64;
    let r2 = r * r;
    let y = POW_A[0] * r + POW_A[1];
    let p = POW_A[2] * r + POW_A[3];
    let r4 = r2 * r2;
    let q = POW_A[4] * r + y0;
    let q = p * r2 + q;
    y * r4 + q
}

#[inline]
fn exp2(xd: f64, sign_bias: u64) -> f64 {
    const SHIFT: f64 = 211106232532992.0;
    let kd = xd + SHIFT;
    let ki = kd.to_bits();
    let kd = kd - SHIFT;
    let r = xd - kd;
    let t = EXP2_TAB[(ki % 32) as usize].wrapping_add(ki.wrapping_add(sign_bias) << 47);
    let s = f64::from_bits(t);
    let z = EXP2_POLY[0] * r + EXP2_POLY[1];
    let r2 = r * r;
    let y = EXP2_POLY[2] * r + 1.0;
    let y = z * r2 + y;
    y * s
}

#[inline]
fn checkint(iy: u32) -> u32 {
    let e = (iy >> 23) & 0xff;
    if e < 0x7f { return 0; }
    if e > 0x7f + 23 { return 2; }
    if iy & ((1 << (0x7f + 23 - e)) - 1) != 0 { return 0; }
    if iy & (1 << (0x7f + 23 - e)) != 0 { 1 } else { 2 }
}

#[inline]
fn zeroinfnan(ix: u32) -> bool {
    ix.wrapping_mul(2).wrapping_sub(1) >= 2 * 0x7f80_0000 - 1
}

#[inline]
pub fn powf(base: f32, exponent: f32) -> f32 {
    let mut ix = base.to_bits();
    let iy = exponent.to_bits();
    let mut sign_bias = 0u64;
    if ix.wrapping_sub(0x0080_0000) >= 0x7f80_0000 - 0x0080_0000 || zeroinfnan(iy) {
        if zeroinfnan(iy) {
            if iy.wrapping_mul(2) == 0 || ix == 0x3f80_0000 { return 1.0; }
            if ix.wrapping_mul(2) > 2 * 0x7f80_0000 || iy.wrapping_mul(2) > 2 * 0x7f80_0000 {
                return base + exponent;
            }
            if ix.wrapping_mul(2) == 2 * 0x3f80_0000 { return 1.0; }
            if (ix.wrapping_mul(2) < 2 * 0x3f80_0000) == (iy & 0x8000_0000 == 0) { return 0.0; }
            return exponent * exponent;
        }
        if zeroinfnan(ix) {
            let mut squared = base * base;
            if ix & 0x8000_0000 != 0 && checkint(iy) == 1 { squared = -squared; }
            return if iy & 0x8000_0000 != 0 { 1.0 / squared } else { squared };
        }
        if ix & 0x8000_0000 != 0 {
            let yint = checkint(iy);
            if yint == 0 { return f32::NAN; }
            if yint == 1 { sign_bias = 1 << 16; }
            ix &= 0x7fff_ffff;
        }
        if ix < 0x0080_0000 {
            ix = (base * 8388608.0).to_bits() & 0x7fff_ffff;
            ix = ix.wrapping_sub(23 << 23);
        }
    }
    let ylogx = exponent as f64 * log2(ix);
    if ((ylogx.to_bits() >> 47) & 0xffff) >= (126.0f64.to_bits() >> 47) {
        if ylogx > 127.99999995700433 {
            return if sign_bias != 0 { f32::NEG_INFINITY } else { f32::INFINITY };
        }
        if ylogx <= -150.0 { return if sign_bias != 0 { -0.0 } else { 0.0 }; }
    }
    exp2(ylogx, sign_bias) as f32
}

/// Elementwise powf. Each input has either one element or `length`.
/// # Safety
/// All pointers must address the lengths declared; output is disjoint.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_glibc239_powf_array_f32(
    base: *const f32, exponent: *const f32, output: *mut f32,
    length: usize, base_length: usize, exponent_length: usize, workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if base.is_null() || exponent.is_null() || output.is_null() { return ERR_NULL; }
        if workers == 0 || (base_length != 1 && base_length != length)
            || (exponent_length != 1 && exponent_length != length) { return ERR_DIMENSION; }
        let base_address = base as usize;
        let exponent_address = exponent as usize;
        let output_address = output as usize;
        let base_step = usize::from(base_length != 1);
        let exponent_step = usize::from(exponent_length != 1);
        crate::parallel::run_ranges(length, workers, |start, stop| {
            let base = base_address as *const f32;
            let exponent = exponent_address as *const f32;
            let output = output_address as *mut f32;
            for i in start..stop {
                // Safety: each index belongs to one disjoint range.
                unsafe { *output.add(i) = powf(*base.add(i * base_step), *exponent.add(i * exponent_step)); }
            }
        });
        OK
    })).unwrap_or(ERR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zero_and_signed_special_values_keep_the_reference_words() {
        assert_eq!(powf(0.0, 0.25).to_bits(), 0);
        assert_eq!(powf(-0.0, 3.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(powf(-0.0, -3.0).to_bits(), f32::NEG_INFINITY.to_bits());
        assert_eq!(powf(-2.0, 0.5).to_bits(), f32::NAN.to_bits());
        assert_eq!(powf(2.0, 3.0), 8.0);
    }
}
