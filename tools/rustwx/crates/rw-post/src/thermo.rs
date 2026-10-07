//! Thermodynamic helpers (specification section 4.1), binary32.
//!
//! The CPU twins of the device functions in `kernels/post_common.cuh`: the
//! same operations in the same order, and the same elementary functions
//! (src/math.rs on the CPU, its twin kernels/woof_math.cuh on the GPU), so both devices return
//! the same bits.  Sources: [R6] (improved Magnus forms AERK/AERKi), [R8]
//! (relative humidity w.r.t. water), [R2] (virtual temperature), [R9]
//! (equivalent potential temperature, eqs. 21 and 43).
//!
//! NaN in gives NaN out.  Clipping is written as comparisons, never
//! `min`/`max`, because those drop a NaN on both devices.

use crate::consts::*;

/// e^x: the shared maths library (one implementation on both devices).
pub use crate::math::{cosf, expf, logf, sinf};

/// x^y for x > 0, exp(y ln x) in the shared maths library.
#[inline]
pub fn powpos(x: f32, y: f32) -> f32 {
    crate::math::powf_pos(x, y)
}

/// Saturation vapour pressure over liquid water (Pa), AERK [R6].
#[inline]
pub fn es_water(tk: f32) -> f32 {
    let tc = tk - T0C;
    ESW_A * expf(ESW_B * tc / (tc + ESW_C))
}

/// Saturation vapour pressure over ice (Pa), AERKi [R6].
#[inline]
pub fn es_ice(tk: f32) -> f32 {
    let tc = tk - T0C;
    ESI_A * expf(ESI_B * tc / (tc + ESI_C))
}

/// Vapour pressure (Pa) from specific humidity and pressure (Pa).
#[inline]
pub fn vapour_pressure(q: f32, p: f32) -> f32 {
    q * p / (EPS + ONE_EPS * q)
}

/// Dewpoint (K) from vapour pressure (Pa): the liquid AERK form inverted.
/// No dewpoint exists for e <= 0: NaN.
#[inline]
pub fn dewpoint(e: f32) -> f32 {
    if !(e > 0.0) {
        return f32::NAN;
    }
    let l = logf(e / ESW_A);
    ESW_C * l / (ESW_B - l) + T0C
}

/// Relative humidity (%) with respect to liquid water, clipped to [0, 100].
#[inline]
pub fn rh(e: f32, tk: f32) -> f32 {
    let mut r = 100.0 * e / es_water(tk);
    if r > 100.0 {
        r = 100.0;
    }
    if r < 0.0 {
        r = 0.0;
    }
    r
}

/// Virtual temperature (K) from temperature and mixing ratio [R2].
#[inline]
pub fn virtual_temperature(tk: f32, r: f32) -> f32 {
    tk * (1.0 + r / EPS) / (1.0 + r)
}

/// Pseudo-equivalent potential temperature (K), [R9] eq. 43 with the
/// condensation temperature of eq. 21.  `p` in Pa, `r` mixing ratio.
#[inline]
pub fn theta_e(tk: f32, p: f32, r: f32) -> f32 {
    let q = r / (1.0 + r);
    let e_hpa = vapour_pressure(q, p) * HPA;
    if !(e_hpa > 0.0) {
        return f32::NAN;
    }
    let tl = BL_A / (BL_B * logf(tk) - logf(e_hpa) - BL_C) + BL_D;
    let rg = r * 1000.0;
    let expo = BK * (1.0 - BK_R * rg);
    let theta = tk * powpos(1000.0 / (p * HPA), expo);
    theta * expf((BE_A / tl - BE_B) * rg * (1.0 + BE_C * rg))
}

/// Lower bound 0 that keeps NaN.
#[inline]
pub fn floor0(x: f32) -> f32 {
    if x < 0.0 { 0.0 } else { x }
}
