//! Derived radar products.
//!
//! Algorithms for computing secondary products from base radar data,
//! including vertically integrated liquid (VIL), echo tops, and
//! storm-relative velocity. Full implementations will be migrated
//! from rustdar.

use wx_field::RadialField;

/// Compute vertically integrated liquid (VIL) from a reflectivity volume scan.
///
/// VIL integrates reflectivity through the depth of the atmosphere to estimate
/// the total liquid water content in a column (kg/m^2). Useful for hail and
/// severe storm detection.
///
/// # Arguments
/// * `reflectivity` - A multi-sweep reflectivity volume (REF product).
///
/// # Returns
/// A vector of VIL values, one per horizontal grid cell. Layout TBD.
pub fn compute_vil(_reflectivity: &RadialField) -> Vec<f64> {
    // TODO: migrate VIL algorithm from rustdar
    Vec::new()
}

/// Compute echo tops from a reflectivity volume scan.
///
/// Echo tops represent the highest altitude (km MSL) where reflectivity
/// exceeds a given threshold (typically 18 dBZ).
///
/// # Arguments
/// * `reflectivity` - A multi-sweep reflectivity volume (REF product).
/// * `threshold_dbz` - Reflectivity threshold in dBZ (default: 18.0).
///
/// # Returns
/// A vector of echo-top heights in km, one per horizontal grid cell.
pub fn compute_echo_tops(_reflectivity: &RadialField, _threshold_dbz: f64) -> Vec<f64> {
    // TODO: migrate echo tops algorithm from rustdar
    Vec::new()
}

/// Compute storm-relative velocity from base velocity and a storm motion vector.
///
/// Subtracts the storm motion component from each radial velocity gate,
/// revealing the wind field relative to the storm for mesocyclone analysis.
///
/// # Arguments
/// * `velocity` - A velocity sweep or volume (VEL product).
/// * `storm_u` - Storm motion east-west component (m/s, positive eastward).
/// * `storm_v` - Storm motion north-south component (m/s, positive northward).
///
/// # Returns
/// A new `RadialField` with storm-relative velocities.
pub fn compute_storm_relative_velocity(
    _velocity: &RadialField,
    _storm_u: f64,
    _storm_v: f64,
) -> RadialField {
    // TODO: migrate SRV algorithm from rustdar
    _velocity.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wx_field::RadarSite;

    fn empty_ref_volume() -> RadialField {
        RadialField {
            product: "REF".to_string(),
            site: RadarSite::new("KTLX", "Oklahoma City", 35.333, -97.278, 370.0),
            sweeps: vec![],
            time: None,
        }
    }

    #[test]
    fn test_vil_stub_returns_empty() {
        let result = compute_vil(&empty_ref_volume());
        assert!(result.is_empty());
    }

    #[test]
    fn test_echo_tops_stub_returns_empty() {
        let result = compute_echo_tops(&empty_ref_volume(), 18.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_srv_stub_returns_clone() {
        let vel = RadialField {
            product: "VEL".to_string(),
            site: RadarSite::new("KTLX", "Oklahoma City", 35.333, -97.278, 370.0),
            sweeps: vec![],
            time: None,
        };
        let srv = compute_storm_relative_velocity(&vel, 10.0, 5.0);
        assert_eq!(srv.product, "VEL");
    }
}
