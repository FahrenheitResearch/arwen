//! Unit-sphere geometry, spelled to match `gpuwm/verify/obs/regrid.py`.
//!
//! Every operation here is a transcription of the Python reference with
//! the SAME arithmetic in the SAME order, because the parity contract on
//! this port is bit-identical float64, not "close enough".  Where the
//! Python spells `np.deg2rad(x)`, this spells `x * (PI / 180.0)`, which
//! is what numpy's ufunc computes (measured, not assumed: the probe in
//! `golden/gen_regrid_goldens.py` checks the identity bitwise before it
//! writes a single golden byte).
//!
//! Geometry is done on the unit sphere in Cartesian coordinates so that
//! longitude wrapping and the poles are not special cases; distances
//! convert between great-circle and chord form with [`EARTH_RADIUS_M`].

use crate::error::RegridError;

/// Mean Earth radius (IUGG), used only to turn a metre bound into a
/// chord bound on the unit sphere.  Matches
/// `gpuwm.verify.obs.regrid.EARTH_RADIUS_M`.
pub const EARTH_RADIUS_M: f64 = 6371008.8;

/// numpy's `deg2rad` is one multiply by the double nearest to pi/180,
/// which is exactly what `PI / 180.0` evaluates to at compile time.
/// Named rather than inlined so the parity claim has somewhere to point.
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

/// `(n, 3)` unit-sphere positions for a lat/lon grid, row-major.
///
/// The component order is `(cos_lat * cos_lon, cos_lat * sin_lon,
/// sin_lat)`, and the multiply order inside each component matches the
/// Python: `cos_lat` is computed once and multiplied on the LEFT.
pub fn unit_vectors(latitude: &[f64], longitude: &[f64]) -> Result<Vec<[f64; 3]>, RegridError> {
    if latitude.len() != longitude.len() || latitude.is_empty() {
        return Err(RegridError::InvalidGrid(String::from(
            "latitude and longitude must be one non-empty shape",
        )));
    }
    let mut points = Vec::with_capacity(latitude.len());
    for (&lat_deg, &lon_deg) in latitude.iter().zip(longitude.iter()) {
        let lat = lat_deg * DEG_TO_RAD;
        let lon = lon_deg * DEG_TO_RAD;
        let cos_lat = lat.cos();
        points.push([cos_lat * lon.cos(), cos_lat * lon.sin(), lat.sin()]);
    }
    Ok(points)
}

/// Straight-line unit-sphere distance for a great-circle metre distance.
///
/// The refusal names the breakage rather than clamping: a non-positive
/// or non-finite bound is a caller that has not decided how far an
/// observation may reach, and silently substituting a default would let
/// a domain corner outside the observation's coverage borrow the nearest
/// edge observation and get scored.
pub fn chord_from_arc(distance_m: f64) -> Result<f64, RegridError> {
    if !distance_m.is_finite() || distance_m <= 0.0 {
        return Err(RegridError::InvalidOptions(String::from(
            "a remap distance bound must be positive and finite",
        )));
    }
    let arc = distance_m / EARTH_RADIUS_M;
    if arc >= std::f64::consts::PI {
        return Ok(2.0);
    }
    Ok(2.0 * (arc / 2.0).sin())
}

/// Great-circle metres for a unit-sphere chord length.
pub fn arc_from_chord(chord: f64) -> f64 {
    let value = (chord / 2.0).max(-1.0).min(1.0);
    2.0 * value.asin() * EARTH_RADIUS_M
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deg_to_rad_is_the_numpy_constant() {
        // 0x3F91DF46A2529D39 is the IEEE754 double numpy's deg2rad
        // multiplies by; a compiler that folded PI/180 differently would
        // move every unit vector and break the bit-identical claim.
        assert_eq!(DEG_TO_RAD.to_bits(), 0x3F91DF46A2529D39);
    }

    #[test]
    fn a_bound_that_is_not_a_distance_is_refused_by_name() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let error = chord_from_arc(bad).unwrap_err();
            assert!(error.to_string().contains("positive and finite"));
        }
    }

    #[test]
    fn a_half_turn_saturates_at_the_diameter() {
        assert_eq!(chord_from_arc(EARTH_RADIUS_M * 4.0).unwrap(), 2.0);
    }

    #[test]
    fn the_pole_is_not_a_special_case() {
        let points = unit_vectors(&[90.0, -90.0], &[0.0, 179.0]).unwrap();
        assert!((points[0][2] - 1.0).abs() < 1.0e-15);
        assert!((points[1][2] + 1.0).abs() < 1.0e-15);
    }

    #[test]
    fn an_empty_or_ragged_grid_is_refused() {
        assert!(unit_vectors(&[], &[]).is_err());
        assert!(unit_vectors(&[1.0], &[1.0, 2.0]).is_err());
    }
}
