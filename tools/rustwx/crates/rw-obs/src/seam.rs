//! The seam's own spellings: time, longitude, units, and provenance.
//!
//! `gpuwm/verify/obs/contracts.py` is the authority for every rule in this
//! module; each one is transcribed here so the producing side refuses what
//! the consuming side would refuse, at the point where it is cheap to fix
//! rather than three receipts later.

use serde::{Deserialize, Serialize};

/// The seam's instant: ISO-8601, UTC, second resolution, **no zone suffix**.
///
/// The scorer parses with `%Y-%m-%dT%H:%M:%S` and refuses a zone rather than
/// interpreting one, on the grounds that a string carrying a zone is a
/// string somebody built from a local clock. So this is not `iso8601()` from
/// the S3 client, which appends `Z`, and the difference is load-bearing.
pub const TIME_FORMAT: &str = "%Y-%m-%dT%H:%M:%S";

/// Spell an instant the way the seam reads it.
pub fn seam_time(when: chrono::DateTime<chrono::Utc>) -> String {
    when.format(TIME_FORMAT).to_string()
}

/// Wrap a longitude into `[-180, 180)`, which is what the seam's geometry
/// check demands. MRMS states its grid in `[0, 360)`, so this fires on every
/// cell of every composite.
pub fn wrap_longitude(lon_deg: f64) -> f64 {
    let wrapped = (lon_deg + 180.0).rem_euclid(360.0) - 180.0;
    // `rem_euclid` can return exactly 360.0 for inputs a hair below a
    // multiple, which would land on +180.0 and fail the half-open bound.
    if wrapped >= 180.0 {
        wrapped - 360.0
    } else {
        wrapped
    }
}

/// Seam quantity ids and their pinned units, transcribed from `GRID_UNITS`.
pub const QUANTITY_COMPOSITE_REFLECTIVITY: &str = "composite_reflectivity";
pub const QUANTITY_PRECIPITATION_ACCUMULATION: &str = "precipitation_accumulation";
pub const UNITS_DBZ: &str = "dBZ";
pub const UNITS_MM: &str = "mm";

/// `SEAM_BOUNDS`, transcribed. A field whose *valid* cells leave these
/// bounds cannot be the quantity it claims to be, and the scorer refuses it;
/// the producing side checks first so the refusal names the archive object.
pub fn seam_bounds(quantity: &str) -> Option<(f64, f64)> {
    match quantity {
        QUANTITY_COMPOSITE_REFLECTIVITY => Some((-40.0, 100.0)),
        QUANTITY_PRECIPITATION_ACCUMULATION => Some((0.0, 2000.0)),
        _ => None,
    }
}

/// Where one observation artifact came from, and whether it is real.
///
/// Mirrors `ObsProvenance` field for field, including `is_stub`, which is
/// always false here — this crate has no stub path, and the field exists so
/// the JSON a receipt carries is the same shape whoever wrote it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub source: String,
    pub product: String,
    pub uri: String,
    pub sha256: String,
    pub fetched_at: String,
    pub is_stub: bool,
    pub stub_reason: String,
}

impl Provenance {
    pub fn new(
        source: impl Into<String>,
        product: impl Into<String>,
        uri: impl Into<String>,
        sha256: impl Into<String>,
        fetched_at: impl Into<String>,
    ) -> Self {
        Self {
            source: source.into(),
            product: product.into(),
            uri: uri.into(),
            sha256: sha256.into(),
            fetched_at: fetched_at.into(),
            is_stub: false,
            stub_reason: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seam_time_carries_no_zone_suffix() {
        // The instant an archived MRMS frame carried: 2021-12-10 21:00:37Z.
        let when = chrono::DateTime::from_timestamp(1_639_170_037, 0).unwrap();
        let text = seam_time(when);
        assert_eq!(text, "2021-12-10T21:00:37");
        assert!(!text.ends_with('Z'), "the scorer refuses a zone suffix");
        assert_eq!(text.len(), 19);
    }

    #[test]
    fn longitudes_land_in_the_half_open_interval_the_scorer_checks() {
        // MRMS states its western edge as 230.005 and its eastern as
        // 299.995; both must come back negative.
        assert!((wrap_longitude(230.005) - (-129.995)).abs() < 1e-9);
        assert!((wrap_longitude(299.995) - (-60.005)).abs() < 1e-9);
        // Already-wrapped values are left alone.
        assert!((wrap_longitude(-97.5) - (-97.5)).abs() < 1e-12);
        assert!((wrap_longitude(0.0)).abs() < 1e-12);
        // The half-open end: +180 is not a legal seam longitude, -180 is.
        assert!((wrap_longitude(180.0) - (-180.0)).abs() < 1e-9);
        assert!((wrap_longitude(-180.0) - (-180.0)).abs() < 1e-9);
        for degrees in [-360.0, -270.5, 0.0, 12.25, 179.9, 180.0, 359.999, 360.0] {
            let wrapped = wrap_longitude(degrees);
            assert!(
                (-180.0..180.0).contains(&wrapped),
                "{degrees} wrapped to {wrapped}, outside [-180, 180)"
            );
        }
    }

    #[test]
    fn the_bounds_table_matches_the_two_quantities_the_seam_declares() {
        assert_eq!(seam_bounds(QUANTITY_COMPOSITE_REFLECTIVITY), Some((-40.0, 100.0)));
        assert_eq!(seam_bounds(QUANTITY_PRECIPITATION_ACCUMULATION), Some((0.0, 2000.0)));
        assert_eq!(seam_bounds("brightness_temperature"), None);
    }
}
