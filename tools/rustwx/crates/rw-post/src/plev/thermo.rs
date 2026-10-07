//! Scalar thermodynamics shared by the column functions.
//!
//! Every function here has a twin in kernels/group_b.cu written with the
//! same operations in the same order.  Rust does not contract a * b + c
//! into a fused multiply-add and the kernels are compiled with
//! `--fmad=false`, so with one maths library (src/math.rs and its twin
//! kernels/woof_math.cuh) on both sides the results are equal
//! bit for bit.  Keep the twins in step when editing.

use crate::plev::consts::*;

#[inline]
pub fn ln(x: f64) -> f64 {
    crate::math::ln(x)
}

#[inline]
pub fn exp(x: f64) -> f64 {
    crate::math::exp(x)
}

#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    crate::math::pow(x, y)
}

/// Saturation vapour pressure over liquid water (Pa), [R6].
#[inline]
pub fn esat_water(t_k: f64) -> f64 {
    let tc = t_k - T_MELT;
    MAGNUS_A * exp(MAGNUS_B * tc / (tc + MAGNUS_C))
}

/// Vapour pressure (Pa) from specific humidity and pressure:
/// e = q p / (eps + (1 - eps) q).
#[inline]
pub fn vapour_pressure(q: f64, p: f64) -> f64 {
    let qf = if q < Q_FLOOR { Q_FLOOR } else { q };
    qf * p / (EPS + (1.0 - EPS) * qf)
}

/// Dewpoint (K) from vapour pressure by inverting the liquid Magnus form [R6].
#[inline]
pub fn dewpoint(e: f64) -> f64 {
    let l = ln(e / MAGNUS_A);
    MAGNUS_C * l / (MAGNUS_B - l) + T_MELT
}

/// Relative humidity (%) w.r.t. liquid water [R8], clipped to [0, 100].
#[inline]
pub fn rh_percent(e: f64, t_k: f64) -> f64 {
    let rh = 100.0 * e / esat_water(t_k);
    if rh > 100.0 {
        100.0
    } else if rh < 0.0 {
        0.0
    } else {
        rh
    }
}

/// Temperature (K) at pressure `p` below the ground by NCAR/TN-396 [R12]
/// (its eq. 16): `t_star` is the ground temperature extrapolated from the
/// lowest model level at the standard lapse rate, `ps` the ground pressure
/// and `zs` the ground height.  Below 2000 m the standard lapse rate
/// applies; above 2500 m the lapse rate is reduced so the sea-level
/// temperature is min(T* + gamma zs, 298 K); in between, the two blend.
/// T = T* (1 + y + y^2/2 + y^3/6), y = alpha ln(p / ps).
#[inline]
pub fn tn396_temperature(t_star: f64, ps: f64, zs: f64, p: f64) -> f64 {
    let lr = ln(p / ps);
    let y;
    if zs < TN396_Z_LOW {
        y = ALPHA * lr;
    } else {
        let t0 = t_star + GAMMA * zs;
        let tplat = if t0 < TN396_T_PLATEAU { t0 } else { TN396_T_PLATEAU };
        let tprime0 = if zs <= TN396_Z_HIGH {
            0.002 * ((TN396_Z_HIGH - zs) * t0 + (zs - TN396_Z_LOW) * tplat)
        } else {
            tplat
        };
        if tprime0 < t_star {
            y = 0.0;
        } else {
            y = RD * (tprime0 - t_star) / (G * zs) * lr;
        }
    }
    t_star * (1.0 + y + 0.5 * y * y + y * y * y / 6.0)
}

/// Height (m) of pressure `p` below the ground by NCAR/TN-396 [R12] (its
/// eq. 15), the hypsometric equation integrated down from the ground along
/// the extrapolated profile, with the published warm (290.5 K) and cold
/// (255 K) column modifications.  `tv_star` is the ground virtual
/// temperature (the hypsometric equation is exact in Tv, [R2]).
#[inline]
pub fn tn396_height(tv_star: f64, ps: f64, zs: f64, p: f64) -> f64 {
    let phis = G * zs;
    let mut ts = tv_star;
    let t0 = ts + GAMMA * zs;
    let alpha;
    if ts <= TN396_T_WARM && t0 > TN396_T_WARM {
        alpha = RD * (TN396_T_WARM - ts) / phis;
    } else if ts > TN396_T_WARM && t0 > TN396_T_WARM {
        alpha = 0.0;
        ts = 0.5 * (TN396_T_WARM + ts);
    } else {
        alpha = ALPHA;
    }
    if ts < TN396_T_COLD {
        ts = 0.5 * (TN396_T_COLD + ts);
    }
    let lr = ln(p / ps);
    let y = alpha * lr;
    (phis - RD * ts * lr * (1.0 + 0.5 * y + y * y / 6.0)) / G
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magnus_and_dewpoint_invert() {
        for t in [233.15, 263.15, 273.15, 293.15, 308.15] {
            let e = esat_water(t);
            assert!((dewpoint(e) - t).abs() < 1e-9, "{t}");
        }
        // 0 C: 610.94 Pa by construction.
        assert!((esat_water(273.15) - 610.94).abs() < 1e-9);
    }

    #[test]
    fn tn396_low_ground_is_standard_lapse() {
        let ts = 288.0;
        let ps = 100000.0;
        let p = 103000.0;
        let exact = ts * pow(p / ps, ALPHA);
        assert!((tn396_temperature(ts, ps, 100.0, p) - exact).abs() < 1e-6);
        // Height at the ground is the ground.
        assert!((tn396_height(ts, ps, 100.0, ps) - 100.0).abs() < 1e-9);
        // Isothermal limit of the hypsometric equation for a small step.
        let z = tn396_height(ts, ps, 100.0, 100100.0);
        let iso = 100.0 - RD * ts / G * ln(100100.0 / ps);
        assert!((z - iso).abs() < 0.01, "{z} {iso}");
    }

    #[test]
    fn tn396_cold_plateau_holds_temperature() {
        // T' 0 < T*: alpha 0, T stays T*.
        assert_eq!(tn396_temperature(300.0, 60000.0, 4000.0, 80000.0), 300.0);
    }
}
