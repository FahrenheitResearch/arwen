//
// Moist thermodynamics for the parcel engine (CPU reference, f64).
//
// Sources (SPEC sections 3.2, 4.1 and 6.1):
// - Constants: A Description of the Advanced Research WRF Model Version 4,
//   NCAR/TN-556+STR (2019): g, Rd, cp = 7 Rd / 2, Rv, p0.
// - Saturation vapour pressure over liquid water: improved Magnus form
//   (AERK), J. Appl. Meteor. 35 (1996) 601-609.
// - LCL temperature (eq. 21) and pseudo-equivalent potential temperature
//   (eq. 43): Mon. Wea. Rev. 108 (1980) 1046-1053.
// - Temperature on a pseudoadiabat by Newton iteration on theta-e:
//   Mon. Wea. Rev. 136 (2008) 2764-2785.
// - Virtual temperature: AMS Glossary of Meteorology.
//
// The CUDA kernel in kernels/severe_c.cu carries the same functions,
// statement for statement, so the two paths stay comparable.

use crate::math as m;

pub const G: f64 = 9.81;
pub const RD: f64 = 287.0;
pub const CP: f64 = 3.5 * RD;
pub const RV: f64 = 461.6;
pub const P0: f64 = 100_000.0;
pub const EPS: f64 = RD / RV;
pub const KAPPA: f64 = RD / CP;

/// Mixing ratios are floored here before any logarithm (a perfectly dry
/// parcel has no finite LCL temperature).  1e-10 kg/kg is far below any
/// value a model writes, so it never changes a real result.
pub const R_FLOOR: f64 = 1.0e-10;

/// Fixed Newton iteration count on the pseudoadiabat.  A fixed count (no
/// convergence test) keeps CPU and GPU control flow identical.  The first
/// guess is always the parcel temperature at the point just below, which
/// lies on the warm side of the root; theta-e is increasing and convex in T
/// there, so the iteration approaches the root monotonically and 10 steps
/// are far past double-precision convergence.
pub const NEWTON_ITERS: usize = 10;

/// Saturation vapour pressure over liquid water (Pa), AERK form.
#[inline]
pub fn es_liquid(t_k: f64) -> f64 {
    let tc = t_k - 273.15;
    610.94 * m::exp(17.625 * tc / (tc + 243.04))
}

/// Vapour pressure (Pa) from mixing ratio r (kg/kg) and pressure p (Pa).
#[inline]
pub fn vapour_pressure(r: f64, p: f64) -> f64 {
    r * p / (EPS + r)
}

/// Saturation mixing ratio over liquid water (kg/kg).
#[inline]
pub fn rsat(t_k: f64, p: f64) -> f64 {
    let es = es_liquid(t_k);
    EPS * es / (p - es).max(1.0e-3 * p)
}

/// Virtual temperature from temperature and mixing ratio.
#[inline]
pub fn virtual_temperature(t_k: f64, r: f64) -> f64 {
    t_k * (1.0 + r / EPS) / (1.0 + r)
}

/// Potential temperature.
#[inline]
pub fn theta(t_k: f64, p: f64) -> f64 {
    t_k * m::pow(P0 / p, KAPPA)
}

/// Temperature from potential temperature.
#[inline]
pub fn temperature_from_theta(th: f64, p: f64) -> f64 {
    th * m::pow(p / P0, KAPPA)
}

/// LCL temperature, Bolton (1980) eq. 21 with relative humidity
/// U = e / e_s(T) as a fraction.
#[inline]
pub fn lcl_temperature(t_k: f64, r: f64, p: f64) -> f64 {
    let r = r.max(R_FLOOR);
    let u = vapour_pressure(r, p) / es_liquid(t_k);
    1.0 / (1.0 / (t_k - 55.0) - m::ln(u) / 2840.0) + 55.0
}

/// Pseudo-equivalent potential temperature, Bolton (1980) eq. 43.
/// r in kg/kg (converted to g/kg inside, as the equation is written).
#[inline]
pub fn theta_e(t_k: f64, p: f64, r: f64, t_lcl: f64) -> f64 {
    let rg = 1000.0 * r.max(R_FLOOR);
    let exponent = 0.2854 * (1.0 - 0.28e-3 * rg);
    t_k * m::pow(P0 / p, exponent) * m::exp((3.376 / t_lcl - 0.00254) * rg * (1.0 + 0.81e-3 * rg))
}

/// Theta-e of a parcel at (T, p) with mixing ratio r, its own LCL
/// temperature.  Vapour above saturation is not vapour a parcel can carry:
/// a supersaturated start is taken as saturated where it stands, the excess
/// leaving as condensate (pseudoadiabatic), so r is capped at r_s(T, p).
#[inline]
pub fn theta_e_parcel(t_k: f64, p: f64, r: f64) -> f64 {
    let r = r.max(R_FLOOR).min(rsat(t_k, p));
    theta_e(t_k, p, r, lcl_temperature(t_k, r, p))
}

/// Temperature of the saturated pseudoadiabat with ln(theta-e) =
/// `ln_theta_e` at pressure `p`, by Newton iteration from `t_guess`.
#[inline]
pub fn pseudoadiabat_temperature(ln_theta_e: f64, p: f64, t_guess: f64) -> f64 {
    let ln_p_ratio = m::ln(P0 / p);
    let mut t = t_guess;
    for _ in 0..NEWTON_ITERS {
        let tc = t - 273.15;
        let es = 610.94 * m::exp(17.625 * tc / (tc + 243.04));
        let dlnes = 17.625 * 243.04 / ((tc + 243.04) * (tc + 243.04));
        let denom = (p - es).max(1.0e-3 * p);
        let r = EPS * es / denom;
        let dr = r * (p / denom) * dlnes;
        let rg = 1000.0 * r;
        let drg = 1000.0 * dr;
        let kap = 0.2854 * (1.0 - 0.28e-3 * rg);
        let dkap = -0.2854 * 0.28e-3 * drg;
        let a = 3.376 / t - 0.00254;
        let da = -3.376 / (t * t);
        let b = rg * (1.0 + 0.81e-3 * rg);
        let db = drg * (1.0 + 1.62e-3 * rg);
        let f = m::ln(t) + kap * ln_p_ratio + a * b - ln_theta_e;
        let df = 1.0 / t + dkap * ln_p_ratio + da * b + a * db;
        t -= f / df;
        t = t.clamp(100.0, 400.0);
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magnus_at_zero_and_twenty() {
        assert!((es_liquid(273.15) - 610.94).abs() < 1e-9);
        // Saturation over water at 20 C is 2339 Pa; AERK stays within its
        // stated 0.4 % of it.
        assert!((es_liquid(293.15) - 2339.0).abs() < 0.004 * 2339.0);
    }

    #[test]
    fn pseudoadiabat_inverts_theta_e() {
        for &(t, p) in &[(300.0, 90_000.0), (285.0, 70_000.0), (240.0, 30_000.0), (210.0, 15_000.0)] {
            let r = rsat(t, p);
            let te = theta_e(t, p, r, t);
            let back = pseudoadiabat_temperature(m::ln(te), p, t + 15.0);
            assert!((back - t).abs() < 1e-9, "{t} {p} -> {back}");
        }
    }

    #[test]
    fn lcl_of_saturated_parcel_is_itself() {
        let t = 295.0;
        let p = 95_000.0;
        let r = rsat(t, p);
        assert!((lcl_temperature(t, r, p) - t).abs() < 1e-9);
    }
}
