//! Severe weather composite parameters.
//!
//! Implements the significant tornado parameter (STP), supercell composite
//! parameter (SCP), and critical angle — standard indices used in operational
//! severe convective weather forecasting.
//!
//! All inputs use SI units: CAPE in J/kg, heights in meters, helicity in
//! m^2/s^2, and bulk shear in m/s.
//!
//! ## Grid-based composites
//!
//! The [`grid`] submodule re-exports grid-oriented functions from
//! `wx_math::composite` that operate on flattened 2-D/3-D arrays and use
//! `rayon` for parallelism. These have different signatures from the
//! point-based functions at the top level of this module.

use std::f64::consts::PI;

// ─────────────────────────────────────────────
// Re-exports from wx_math::composite (point-based helpers)
// ─────────────────────────────────────────────

pub use wx_math::composite::boyden_index;
pub use wx_math::composite::bulk_richardson_number;
pub use wx_math::composite::convective_inhibition_depth;
pub use wx_math::composite::dendritic_growth_zone;
pub use wx_math::composite::fosberg_fire_weather_index;
pub use wx_math::composite::freezing_rain_composite;
pub use wx_math::composite::haines_index;
pub use wx_math::composite::hot_dry_windy;
pub use wx_math::composite::warm_nose_check;

// Re-export galvez_davison_index from wx_math::thermo (it lives there, not composite)
pub use wx_math::thermo::galvez_davison_index;

/// Grid-oriented composite parameters from `wx_math::composite`.
///
/// These functions accept flattened 2-D or 3-D arrays (`&[f64]`) and grid
/// dimensions (`nx`, `ny`, `nz`) and return `Vec<f64>` result grids. They
/// use `rayon` internally for parallel computation.
///
/// Functions whose names overlap with point-based versions in the parent
/// module (e.g. `supercell_composite_parameter`, `critical_angle`) have
/// different signatures here — they take grid slices rather than scalar
/// values.
pub mod grid {
    // ── Stability indices (profile-based, different arg order from thermo) ──
    pub use wx_math::composite::showalter_index;
    pub use wx_math::composite::lifted_index;
    pub use wx_math::composite::k_index;
    pub use wx_math::composite::total_totals;
    pub use wx_math::composite::cross_totals;
    pub use wx_math::composite::vertical_totals;
    pub use wx_math::composite::sweat_index;

    // ── 3-D grid compute functions ──
    pub use wx_math::composite::compute_cape_cin;
    pub use wx_math::composite::compute_srh;
    pub use wx_math::composite::compute_shear;
    pub use wx_math::composite::compute_lapse_rate;
    pub use wx_math::composite::compute_pw;

    // ── 2-D grid composite parameters ──
    pub use wx_math::composite::compute_stp;
    pub use wx_math::composite::compute_scp;
    pub use wx_math::composite::compute_ehi;
    pub use wx_math::composite::significant_hail_parameter;
    pub use wx_math::composite::derecho_composite_parameter;
    pub use wx_math::composite::supercell_composite_parameter;
    pub use wx_math::composite::critical_angle;

    // ── Reflectivity composites ──
    pub use wx_math::composite::composite_reflectivity_from_refl;
    pub use wx_math::composite::composite_reflectivity_from_hydrometeors;
}

/// Significant Tornado Parameter (STP).
///
/// STP combines mixed-layer CAPE, LCL height, 0-1 km storm-relative helicity,
/// and 0-6 km bulk shear magnitude into a single composite favoring significant
/// (EF2+) tornadoes.
///
/// ```text
/// STP = (mlCAPE / 1500) * ((2000 - mlLCL) / 1000) * (SRH / 150) * (shear / 20)
/// ```
///
/// Each term is clamped so it cannot go below 0 (and the LCL term is also
/// capped at 1.0 when LCL <= 1000 m, per Thompson et al. 2003).
///
/// # Arguments
/// * `mlcape` — Mixed-layer CAPE (J/kg)
/// * `lcl_height_m` — Mixed-layer LCL height AGL (m)
/// * `srh_0_1km` — 0-1 km storm-relative helicity (m^2/s^2)
/// * `bulk_shear_0_6km_ms` — 0-6 km bulk wind shear magnitude (m/s)
///
/// # Returns
/// Dimensionless STP value. Values above 1.0 are increasingly favorable for
/// significant tornadoes.
///
/// # References
/// Thompson, R. L., R. Edwards, J. A. Hart, K. L. Elmore, and P. Markowski,
/// 2003: Close proximity soundings within supercell environments obtained from
/// the Rapid Update Cycle. *Wea. Forecasting*, **18**, 1243-1261.
pub fn significant_tornado_parameter(
    sbcape: f64,
    lcl_height_m: f64,
    srh_0_1km: f64,
    bulk_shear_0_6km_ms: f64,
) -> f64 {
    // CAPE term: SBCAPE / 1500, floored at 0
    // Fixed-layer STP uses surface-based CAPE (not MLCAPE)
    let cape_term = (sbcape / 1500.0).max(0.0);

    // LCL term: (2000 - LCL) / 1000
    //   - Capped at 1.0 when LCL <= 1000 m (very low LCL is always favorable)
    //   - Floored at 0.0 when LCL >= 2000 m (too high, unfavorable)
    let lcl_term = if lcl_height_m <= 1000.0 {
        1.0
    } else {
        ((2000.0 - lcl_height_m) / 1000.0).clamp(0.0, 1.0)
    };

    // SRH term: SRH / 150, floored at 0
    let srh_term = (srh_0_1km / 150.0).max(0.0);

    // Shear term: zero when < 12.5 m/s, capped at 30 m/s, then / 20
    // (per Thompson et al. 2003 / MetPy: shear < 12.5 m/s => 0)
    let shear_term = if bulk_shear_0_6km_ms < 12.5 {
        0.0
    } else {
        (bulk_shear_0_6km_ms.min(30.0) / 20.0).max(0.0)
    };

    cape_term * lcl_term * srh_term * shear_term
}

/// Supercell Composite Parameter (SCP).
///
/// SCP combines most-unstable CAPE, effective-layer storm-relative helicity,
/// and effective bulk shear magnitude.
///
/// ```text
/// SCP = (muCAPE / 1000) * (SRH / 50) * (shear / 20)
/// ```
///
/// Each term is floored at 0.
///
/// # Arguments
/// * `mucape` — Most-unstable CAPE (J/kg)
/// * `srh_eff` — Effective-layer storm-relative helicity (m^2/s^2)
/// * `bulk_shear_eff_ms` — Effective bulk shear magnitude (m/s)
///
/// # Returns
/// Dimensionless SCP value. Values >= 1.0 favor supercells.
///
/// # References
/// Thompson, R. L., R. Edwards, and C. M. Mead, 2004: An update to the
/// supercell composite and significant tornado parameters. Preprints, 22nd
/// Conf. on Severe Local Storms, Hyannis, MA.
pub fn supercell_composite_parameter(
    mucape: f64,
    srh_eff: f64,
    bulk_shear_eff_ms: f64,
) -> f64 {
    let cape_term = (mucape / 1000.0).max(0.0);
    let srh_term = (srh_eff / 50.0).max(0.0);
    // Shear term: zero when < 10 m/s, capped at 1.0 (i.e. clipped to 20 m/s then / 20)
    // (per Thompson et al. 2004 / MetPy)
    let shear_term = if bulk_shear_eff_ms < 10.0 {
        0.0
    } else {
        (bulk_shear_eff_ms.min(20.0) / 20.0).max(0.0)
    };

    cape_term * srh_term * shear_term
}

/// Critical angle between the storm-relative inflow vector and the 0-500 m
/// shear vector.
///
/// A critical angle near 90 degrees is most favorable for low-level
/// mesocyclone development because it means the storm-relative inflow is
/// perpendicular to the low-level shear, maximizing streamwise vorticity
/// tilting.
///
/// # Arguments
/// * `storm_u`, `storm_v` — Storm motion components (m/s)
/// * `u_sfc`, `v_sfc` — Surface wind components (m/s)
/// * `u_500m`, `v_500m` — Wind at 500 m AGL (m/s)
///
/// # Returns
/// The angle in degrees [0, 180] between the two vectors. Returns 0.0 if
/// either vector has near-zero magnitude.
///
/// # References
/// Esterheld, J. M., and D. J. Giuliano, 2008: Discriminating between
/// tornadic and non-tornadic supercells: A new hodograph technique. *E-Journal
/// of Severe Storms Meteorology*, **3(2)**, 1-13.
pub fn critical_angle(
    storm_u: f64,
    storm_v: f64,
    u_sfc: f64,
    v_sfc: f64,
    u_500m: f64,
    v_500m: f64,
) -> f64 {
    // Storm-relative inflow vector (surface wind relative to storm)
    let inflow_u = u_sfc - storm_u;
    let inflow_v = v_sfc - storm_v;

    // 0-500 m shear vector
    let shear_u = u_500m - u_sfc;
    let shear_v = v_500m - v_sfc;

    let mag_inflow = (inflow_u * inflow_u + inflow_v * inflow_v).sqrt();
    let mag_shear = (shear_u * shear_u + shear_v * shear_v).sqrt();

    if mag_inflow < 1e-10 || mag_shear < 1e-10 {
        return 0.0;
    }

    let cos_angle = (inflow_u * shear_u + inflow_v * shear_v) / (mag_inflow * mag_shear);
    // Clamp to [-1, 1] to avoid NaN from floating-point rounding
    cos_angle.clamp(-1.0, 1.0).acos() * (180.0 / PI)
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── significant_tornado_parameter ──

    #[test]
    fn test_stp_nominal() {
        // All terms at their normalizing values => STP = 1.0
        let stp = significant_tornado_parameter(1500.0, 1000.0, 150.0, 20.0);
        assert!(
            (stp - 1.0).abs() < 1e-10,
            "STP with nominal values = {stp}, expected 1.0"
        );
    }

    #[test]
    fn test_stp_zero_cape() {
        let stp = significant_tornado_parameter(0.0, 800.0, 200.0, 25.0);
        assert!((stp - 0.0).abs() < 1e-10, "STP should be 0 with zero CAPE");
    }

    #[test]
    fn test_stp_high_lcl_zero() {
        // LCL at 2000 m makes the LCL term 0 => STP = 0
        let stp = significant_tornado_parameter(2000.0, 2000.0, 200.0, 25.0);
        assert!((stp - 0.0).abs() < 1e-10, "STP should be 0 with LCL=2000m");
    }

    #[test]
    fn test_stp_very_low_lcl_capped() {
        // LCL <= 1000m => LCL term capped at 1.0
        let stp_500 = significant_tornado_parameter(1500.0, 500.0, 150.0, 20.0);
        let stp_1000 = significant_tornado_parameter(1500.0, 1000.0, 150.0, 20.0);
        assert!(
            (stp_500 - stp_1000).abs() < 1e-10,
            "LCL 500m and 1000m should give same STP"
        );
    }

    #[test]
    fn test_stp_shear_capped() {
        // Shear term capped at 1.5 when shear = 30 m/s (30/20 = 1.5)
        let stp_30 = significant_tornado_parameter(1500.0, 1000.0, 150.0, 30.0);
        let stp_50 = significant_tornado_parameter(1500.0, 1000.0, 150.0, 50.0);
        assert!(
            (stp_30 - stp_50).abs() < 1e-10,
            "STP should cap shear at 1.5: stp_30={stp_30}, stp_50={stp_50}"
        );
        assert!((stp_30 - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_stp_weak_shear_zero() {
        // Shear below 12.5 m/s => shear_term = 0 => STP = 0 (per MetPy)
        let stp = significant_tornado_parameter(2000.0, 800.0, 150.0, 10.0);
        assert!((stp - 0.0).abs() < 1e-10, "STP should be 0 with weak shear (<12.5 m/s)");
    }

    #[test]
    fn test_stp_negative_inputs_floored() {
        let stp = significant_tornado_parameter(-500.0, 3000.0, -100.0, -5.0);
        assert!((stp - 0.0).abs() < 1e-10, "STP should be 0 with all negative inputs");
    }

    #[test]
    fn test_stp_strong_case() {
        // High CAPE=4000, low LCL=500, strong SRH=400, strong shear=35
        let stp = significant_tornado_parameter(4000.0, 500.0, 400.0, 35.0);
        let expected = (4000.0 / 1500.0) * 1.0 * (400.0 / 150.0) * 1.5;
        assert!((stp - expected).abs() < 1e-10, "STP = {stp}, expected {expected}");
    }

    // ── supercell_composite_parameter ──

    #[test]
    fn test_scp_nominal() {
        let scp = supercell_composite_parameter(1000.0, 50.0, 20.0);
        assert!((scp - 1.0).abs() < 1e-10, "SCP with nominal values = {scp}");
    }

    #[test]
    fn test_scp_zero_cape() {
        let scp = supercell_composite_parameter(0.0, 200.0, 30.0);
        assert!((scp - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_scp_strong_case() {
        // Shear 30 m/s is capped at 20 m/s => shear_term = 1.0
        let scp = supercell_composite_parameter(3000.0, 300.0, 30.0);
        let expected = (3000.0 / 1000.0) * (300.0 / 50.0) * 1.0;
        assert!((scp - expected).abs() < 1e-10, "SCP = {scp}, expected {expected}");
    }

    #[test]
    fn test_scp_weak_shear_zero() {
        // Shear below 10 m/s => shear_term = 0 => SCP = 0 (per MetPy)
        let scp = supercell_composite_parameter(3000.0, 200.0, 8.0);
        assert!((scp - 0.0).abs() < 1e-10, "SCP should be 0 with weak shear");
    }

    #[test]
    fn test_scp_negative_inputs_floored() {
        let scp = supercell_composite_parameter(-500.0, -100.0, -10.0);
        assert!((scp - 0.0).abs() < 1e-10);
    }

    // ── critical_angle ──

    #[test]
    fn test_critical_angle_perpendicular() {
        // Inflow from east (storm east of surface obs), shear points north
        // => 90 degrees
        let angle = critical_angle(
            10.0, 0.0,  // storm at (10, 0)
            0.0, 0.0,   // sfc wind calm
            0.0, 5.0,   // 500m wind northward
        );
        assert!(
            (angle - 90.0).abs() < 1e-10,
            "expected 90 degrees, got {angle}"
        );
    }

    #[test]
    fn test_critical_angle_parallel() {
        // Inflow and shear both pointing north => 0 degrees
        let angle = critical_angle(
            0.0, -10.0, // storm south of sfc
            0.0, 0.0,   // sfc calm
            0.0, 5.0,   // 500m northward
        );
        assert!(
            angle.abs() < 1e-10,
            "expected 0 degrees, got {angle}"
        );
    }

    #[test]
    fn test_critical_angle_antiparallel() {
        // Inflow and shear 180 degrees apart
        let angle = critical_angle(
            0.0, 10.0,  // storm north of sfc
            0.0, 0.0,   // sfc calm
            0.0, 5.0,   // 500m northward
        );
        assert!(
            (angle - 180.0).abs() < 1e-10,
            "expected 180 degrees, got {angle}"
        );
    }

    #[test]
    fn test_critical_angle_zero_inflow() {
        // Storm motion equals surface wind => zero inflow vector
        let angle = critical_angle(5.0, 3.0, 5.0, 3.0, 10.0, 8.0);
        assert!((angle - 0.0).abs() < 1e-10, "zero inflow should give 0 deg");
    }

    #[test]
    fn test_critical_angle_zero_shear() {
        // 500m wind equals surface wind => zero shear vector
        let angle = critical_angle(10.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!((angle - 0.0).abs() < 1e-10, "zero shear should give 0 deg");
    }

    #[test]
    fn test_critical_angle_45_degrees() {
        // Inflow along x-axis, shear at 45 degrees
        let angle = critical_angle(
            10.0, 0.0, // storm motion
            0.0, 0.0,  // sfc
            -5.0, 5.0, // 500m: shear = (-5, 5), inflow = (0,0)-(10,0) = (-10, 0)
        );
        // inflow = (-10, 0), shear = (-5, 5)
        // cos(theta) = ((-10)*(-5) + 0*5) / (10 * sqrt(50)) = 50 / (10*sqrt(50))
        //            = 5 / sqrt(50) = sqrt(50)/10 = 1/sqrt(2)
        // theta = 45 degrees
        assert!(
            (angle - 45.0).abs() < 1e-10,
            "expected 45 degrees, got {angle}"
        );
    }
}
