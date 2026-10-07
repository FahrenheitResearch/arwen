//! GRIB2 grid-coordinate encoding (specification 4.1).
//!
//! GRIB2 grid definition templates (WMO-No. 306, Manual on Codes [R10])
//! carry the latitude and longitude of grid points in units of 1e-6 degree,
//! with longitude in [0, 360).  Values round half away from zero.  The
//! exporter's sphere-radius and projected-spacing handling is separate and
//! unchanged.

/// Latitude in micro-degrees, rounded half away from zero.
pub fn latitude_microdeg(lat_deg: f64) -> i32 {
    (lat_deg * 1e6).round() as i32
}

/// Longitude in micro-degrees on [0, 360e6), rounded half away from zero.
pub fn longitude_microdeg(lon_deg: f64) -> u32 {
    let mut lon = lon_deg % 360.0;
    if lon < 0.0 {
        lon += 360.0;
    }
    let micro = (lon * 1e6).round();
    if micro >= 360e6 { (micro - 360e6) as u32 } else { micro as u32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_wmo_ranges() {
        assert_eq!(latitude_microdeg(38.5), 38_500_000);
        assert_eq!(latitude_microdeg(-0.0000005), -1);
        assert_eq!(latitude_microdeg(0.0000005), 1);
        assert_eq!(longitude_microdeg(-97.5), 262_500_000);
        assert_eq!(longitude_microdeg(-0.0), 0);
        assert_eq!(longitude_microdeg(359.9999999), 0);
        assert_eq!(longitude_microdeg(-180.0), 180_000_000);
        assert_eq!(longitude_microdeg(540.25), 180_250_000);
    }
}
