/// Radar site information.

/// Information about a radar site (NEXRAD, TDWR, etc.).
#[derive(Debug, Clone)]
pub struct RadarSite {
    /// Site identifier (e.g., "KTLX", "KOUN").
    pub id: String,
    /// Human-readable site name (e.g., "Oklahoma City").
    pub name: String,
    /// Latitude in degrees.
    pub lat: f64,
    /// Longitude in degrees.
    pub lon: f64,
    /// Elevation above sea level in meters.
    pub elevation: f64,
}

impl RadarSite {
    /// Create a new radar site.
    pub fn new(id: impl Into<String>, name: impl Into<String>, lat: f64, lon: f64, elevation: f64) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            lat,
            lon,
            elevation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radar_site() {
        let site = RadarSite::new("KTLX", "Oklahoma City", 35.333, -97.278, 370.0);
        assert_eq!(site.id, "KTLX");
        assert!((site.lat - 35.333).abs() < 0.001);
    }

    #[test]
    fn test_site_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RadarSite>();
    }
}
