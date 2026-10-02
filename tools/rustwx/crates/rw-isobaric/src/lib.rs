//! Column arithmetic onto pressure levels.
//!
//! Two consumers share this crate, and the sharing is the point: the
//! renderer's pressure-level charts (`rw-wrfbatch`, `wrf_volumes.rs`) and the
//! machine-learning exporter (`rw-mlexport`) interpolate a column with the
//! same [`bracket`] and [`lerp`], so a 500 hPa map and a 500 hPa training
//! sample from one history file cannot disagree about where 500 hPa is.
//!
//! The renderer leaves a target outside the column's model pressure range
//! empty.  A training dataset cannot: every ML loader computes per-level
//! normalisation statistics, and a hole is a NaN in those.  The exporter
//! fills below ground with the rules ERA5's pressure levels are filled
//! with ([`ecmwf_temperature`], [`ecmwf_geopotential`]), which live here
//! because they are the same column arithmetic.

/// Locate the native levels bracketing `target` in a column (pressure
/// decreasing with index, level 0 nearest the surface) and return the lower
/// level index plus the log-pressure interpolation weight.  `None` when the
/// target sits below the lowest level or above the top one.
///
/// Moved whole from the renderer's `wrf_volumes.rs`; its pressure-level
/// charts call this copy.
pub fn bracket(col_p: &[f64], target: f64) -> Option<(usize, f64)> {
    for k in 0..col_p.len().saturating_sub(1) {
        let (pk, pk1) = (col_p[k], col_p[k + 1]);
        if !pk.is_finite() || !pk1.is_finite() || pk == pk1 {
            continue;
        }
        let (hi, lo) = if pk >= pk1 { (pk, pk1) } else { (pk1, pk) };
        if target <= hi && target >= lo {
            let t = (target.ln() - pk.ln()) / (pk1.ln() - pk.ln());
            return Some((k, t));
        }
    }
    None
}

/// `a + t (b - a)`, or `None` when either endpoint is not finite.
///
/// Moved whole from the renderer's `wrf_volumes.rs`.
pub fn lerp(a: f64, b: f64, t: f64) -> Option<f64> {
    (a.is_finite() && b.is_finite()).then_some(a + t * (b - a))
}

/// The constants the ECMWF below-ground rule is written with.
///
/// Trenberth, Berry and Buja (1993) state the rule with the dry-air gas
/// constant 287.04 and gravity 9.80616, and NCL (`vinth2p_ecmwf`) and GeoCAT
/// (`interp_hybrid_to_pressure(extrapolate=True)`) implement it with exactly
/// those, so a column filled here can be checked against either to round-off
/// rather than to a constant's worth of difference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EcmwfConstants {
    /// Dry-air gas constant (J kg-1 K-1).
    pub rd: f64,
    /// Gravity (m s-2).
    pub g: f64,
    /// Standard lapse rate (K m-1).
    pub lapse: f64,
}

/// The rule's own constants (see [`EcmwfConstants`]).
pub const ECMWF_RULE: EcmwfConstants = EcmwfConstants {
    rd: 287.04,
    g: 9.80616,
    lapse: 0.0065,
};

/// The quantities of one column the below-ground rule reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColumnBase {
    /// Temperature of the lowest model level (K).
    pub t_bot: f64,
    /// Pressure of the lowest model level (Pa).
    pub p_bot: f64,
    /// Surface pressure (Pa).
    pub p_sfc: f64,
    /// Surface geopotential (m2 s-2).
    pub phi_sfc: f64,
}

impl ColumnBase {
    /// The surface temperature the rule extrapolates from, T* = T_bot (1 +
    /// alpha (p_sfc / p_bot - 1)), alpha = lapse Rd / g.
    pub fn t_star(&self, c: &EcmwfConstants) -> f64 {
        let alpha = c.lapse * c.rd / c.g;
        self.t_bot * (1.0 + alpha * (self.p_sfc / self.p_bot - 1.0))
    }
}

/// Temperature at pressure `p` (Pa) below the lowest model level, by the
/// ECMWF rule (Trenberth et al. 1993, equation 16).
///
/// With hgt = phi_sfc / g, T0 = T* + lapse hgt and T_plat = min(T0, 298):
/// below 2000 m the standard lapse rate applies; above 2500 m the lapse rate
/// is reduced so the extrapolated sea-level temperature is T_plat; between
/// the two, blended.  y = alpha ln(p / p_sfc), T = T* (1 + y + y^2/2 + y^3/6).
pub fn ecmwf_temperature(base: &ColumnBase, p: f64, c: &EcmwfConstants) -> f64 {
    let alpha = c.lapse * c.rd / c.g;
    let t_star = base.t_star(c);
    let hgt = base.phi_sfc / c.g;
    let ln_ratio = (p / base.p_sfc).ln();
    let y = if hgt < 2000.0 {
        alpha * ln_ratio
    } else {
        let t0 = t_star + c.lapse * hgt;
        let t_plat = t0.min(298.0);
        let t_prime0 = if hgt <= 2500.0 {
            0.002 * ((2500.0 - hgt) * t0 + (hgt - 2000.0) * t_plat)
        } else {
            t_plat
        };
        if t_prime0 < t_star {
            0.0
        } else {
            c.rd * (t_prime0 - t_star) / base.phi_sfc * ln_ratio
        }
    };
    t_star * (1.0 + y + 0.5 * y * y + y * y * y / 6.0)
}

/// Geopotential (m2 s-2) at pressure `p` (Pa) below the lowest model level,
/// by the ECMWF rule (Trenberth et al. 1993, equation 15).
///
/// alpha is the standard value, except Rd (290.5 - T*) / phi_sfc when
/// T* <= 290.5 < T0, and 0 with T* replaced by (290.5 + T*) / 2 when both
/// exceed 290.5; then T* becomes (255 + T*) / 2 when it is under 255.
/// y = alpha ln(p / p_sfc), phi = phi_sfc - Rd T* ln(p / p_sfc) (1 + y/2 +
/// y^2/6).
pub fn ecmwf_geopotential(base: &ColumnBase, p: f64, c: &EcmwfConstants) -> f64 {
    let alpha_std = c.lapse * c.rd / c.g;
    let mut t_star = base.t_star(c);
    let hgt = base.phi_sfc / c.g;
    let t0 = t_star + c.lapse * hgt;
    let alpha = if t_star <= 290.5 && t0 > 290.5 {
        c.rd / base.phi_sfc * (290.5 - t_star)
    } else if t_star > 290.5 && t0 > 290.5 {
        t_star = 0.5 * (290.5 + t_star);
        0.0
    } else {
        alpha_std
    };
    if t_star < 255.0 {
        t_star = 0.5 * (255.0 + t_star);
    }
    let ln_ratio = (p / base.p_sfc).ln();
    let y = alpha * ln_ratio;
    base.phi_sfc - c.rd * t_star * ln_ratio * (1.0 + 0.5 * y + y * y / 6.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bracket_interpolates_in_log_pressure_and_clamps_to_range() {
        let col = [1000.0, 850.0, 700.0, 500.0];
        let (k, t) = bracket(&col, 925.0).expect("in range");
        assert_eq!(k, 0);
        let expected = (925f64.ln() - 1000f64.ln()) / (850f64.ln() - 1000f64.ln());
        assert!((t - expected).abs() < 1e-12);
        assert!(bracket(&col, 1013.0).is_none());
        assert!(bracket(&col, 300.0).is_none());
    }

    #[test]
    fn bracket_takes_the_first_pair_from_the_ground_up() {
        // A target exactly on a level is the top of the pair below it.
        let col = [1000.0, 850.0, 700.0];
        let (k, t) = bracket(&col, 850.0).expect("on a level");
        assert_eq!((k, t), (0, 1.0));
    }

    #[test]
    fn bracket_skips_non_finite_and_flat_pairs() {
        let col = [f64::NAN, 900.0, 900.0, 800.0];
        let (k, _) = bracket(&col, 850.0).expect("finite pair above");
        assert_eq!(k, 2);
    }

    #[test]
    fn lerp_skips_non_finite_endpoints() {
        assert_eq!(lerp(0.0, 10.0, 0.5), Some(5.0));
        assert_eq!(lerp(f64::NAN, 10.0, 0.5), None);
        assert_eq!(lerp(0.0, f64::INFINITY, 0.5), None);
    }

    fn low_column() -> ColumnBase {
        ColumnBase {
            t_bot: 288.0,
            p_bot: 100_000.0,
            p_sfc: 100_300.0,
            phi_sfc: 9.80616 * 50.0,
        }
    }

    #[test]
    fn both_rules_meet_the_surface_at_the_surface() {
        let base = low_column();
        let t_star = base.t_star(&ECMWF_RULE);
        assert!((ecmwf_temperature(&base, base.p_sfc, &ECMWF_RULE) - t_star).abs() < 1e-12);
        assert!((ecmwf_geopotential(&base, base.p_sfc, &ECMWF_RULE) - base.phi_sfc).abs() < 1e-9);
    }

    #[test]
    fn low_ground_follows_the_standard_lapse_rate() {
        // Below 2000 m: T = T* (1 + y + y^2/2 + y^3/6) is exp(y) to third
        // order, i.e. T = T* (p / p_sfc)^alpha, the standard atmosphere's
        // temperature-pressure relation.
        let base = low_column();
        let t_star = base.t_star(&ECMWF_RULE);
        let alpha = 0.0065 * 287.04 / 9.80616;
        let p = 105_000.0;
        let exact = t_star * (p / base.p_sfc).powf(alpha);
        let got = ecmwf_temperature(&base, p, &ECMWF_RULE);
        assert!((got - exact).abs() < 1e-6, "{got} vs {exact}");
        assert!(got > t_star);
    }

    #[test]
    fn geopotential_falls_below_ground_and_matches_the_hand_computation() {
        let base = low_column();
        let p = 101_300.0;
        let t_star = base.t_star(&ECMWF_RULE);
        let alpha = 0.0065 * 287.04 / 9.80616;
        let ln_ratio = (p / base.p_sfc).ln();
        let y = alpha * ln_ratio;
        let expected = base.phi_sfc - 287.04 * t_star * ln_ratio * (1.0 + y / 2.0 + y * y / 6.0);
        let got = ecmwf_geopotential(&base, p, &ECMWF_RULE);
        assert_eq!(got, expected);
        assert!(got < base.phi_sfc);
    }

    #[test]
    fn high_ground_uses_the_plateau_temperature() {
        // 3000 m: T' 0 is T_plat = min(T0, 298); with a warm column T0 > 298
        // so the lapse rate is reduced and the extrapolated temperature rises
        // more slowly than the standard rate would make it.
        let base = ColumnBase {
            t_bot: 290.0,
            p_bot: 70_000.0,
            p_sfc: 70_300.0,
            phi_sfc: 9.80616 * 3000.0,
        };
        let t_star = base.t_star(&ECMWF_RULE);
        let t0 = t_star + 0.0065 * 3000.0;
        assert!(t0 > 298.0);
        let p = 85_000.0;
        let got = ecmwf_temperature(&base, p, &ECMWF_RULE);
        let alpha = 287.04 * (298.0 - t_star) / base.phi_sfc;
        let y = alpha * (p / base.p_sfc).ln();
        let expected = t_star * (1.0 + y + 0.5 * y * y + y * y * y / 6.0);
        assert_eq!(got, expected);
        let standard = t_star * (p / base.p_sfc).powf(0.0065 * 287.04 / 9.80616);
        assert!(got < standard);
    }

    #[test]
    fn blended_band_lies_between_its_ends() {
        // 2250 m: T' 0 is the average of T0 and T_plat.
        let base = ColumnBase {
            t_bot: 300.0,
            p_bot: 77_000.0,
            p_sfc: 77_200.0,
            phi_sfc: 9.80616 * 2250.0,
        };
        let t_star = base.t_star(&ECMWF_RULE);
        let t0 = t_star + 0.0065 * 2250.0;
        let t_plat = t0.min(298.0);
        let t_prime0 = 0.002 * (250.0 * t0 + 250.0 * t_plat);
        assert!((t_prime0 - 0.5 * (t0 + t_plat)).abs() < 1e-9);
        let p = 90_000.0;
        let alpha = 287.04 * (t_prime0 - t_star) / base.phi_sfc;
        let y = alpha * (p / base.p_sfc).ln();
        let expected = t_star * (1.0 + y + 0.5 * y * y + y * y * y / 6.0);
        assert_eq!(ecmwf_temperature(&base, p, &ECMWF_RULE), expected);
    }

    #[test]
    fn a_cold_plateau_holds_the_lapse_rate_at_zero() {
        // T' 0 < T*: alpha is 0 and the temperature stays at T*.
        let base = ColumnBase {
            t_bot: 300.0,
            p_bot: 60_000.0,
            p_sfc: 60_200.0,
            phi_sfc: 9.80616 * 4000.0,
        };
        let t_star = base.t_star(&ECMWF_RULE);
        assert!(t_star > 298.0);
        assert_eq!(ecmwf_temperature(&base, 80_000.0, &ECMWF_RULE), t_star);
    }

    #[test]
    fn geopotential_warm_branches_follow_the_rule() {
        // T* <= 290.5 < T0: alpha = Rd (290.5 - T*) / phi_sfc.
        let base = ColumnBase {
            t_bot: 285.0,
            p_bot: 85_000.0,
            p_sfc: 85_200.0,
            phi_sfc: 9.80616 * 1500.0,
        };
        let t_star = base.t_star(&ECMWF_RULE);
        assert!(t_star <= 290.5 && t_star + 0.0065 * 1500.0 > 290.5);
        let p = 95_000.0;
        let alpha = 287.04 * (290.5 - t_star) / base.phi_sfc;
        let ln_ratio = (p / base.p_sfc).ln();
        let y = alpha * ln_ratio;
        let expected = base.phi_sfc - 287.04 * t_star * ln_ratio * (1.0 + y / 2.0 + y * y / 6.0);
        assert_eq!(ecmwf_geopotential(&base, p, &ECMWF_RULE), expected);

        // Both above 290.5: alpha = 0 and T* = (290.5 + T*) / 2.
        let hot = ColumnBase { t_bot: 305.0, ..base };
        let t_star = hot.t_star(&ECMWF_RULE);
        let t_mid = 0.5 * (290.5 + t_star);
        let expected = hot.phi_sfc - 287.04 * t_mid * ln_ratio;
        assert_eq!(ecmwf_geopotential(&hot, p, &ECMWF_RULE), expected);
    }

    #[test]
    fn geopotential_cold_columns_are_warmed_toward_255() {
        let base = ColumnBase {
            t_bot: 240.0,
            p_bot: 99_000.0,
            p_sfc: 99_200.0,
            phi_sfc: 9.80616 * 100.0,
        };
        let t_star = base.t_star(&ECMWF_RULE);
        assert!(t_star < 255.0);
        let p = 104_000.0;
        let alpha = 0.0065 * 287.04 / 9.80616;
        let ln_ratio = (p / base.p_sfc).ln();
        let y = alpha * ln_ratio;
        let t_used = 0.5 * (255.0 + t_star);
        let expected = base.phi_sfc - 287.04 * t_used * ln_ratio * (1.0 + y / 2.0 + y * y / 6.0);
        assert_eq!(ecmwf_geopotential(&base, p, &ECMWF_RULE), expected);
    }
}
