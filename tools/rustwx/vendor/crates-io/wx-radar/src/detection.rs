//! Rotation detection types for mesocyclone and TVS (Tornado Vortex Signature).
//!
//! Defines output structures for vortex detection algorithms. The actual
//! detection logic will be migrated from rustdar in a future release.

/// A detected mesocyclone (rotational couplet in Doppler velocity).
#[derive(Debug, Clone)]
pub struct MesocycloneDetection {
    /// Latitude of the mesocyclone center (degrees).
    pub lat: f64,
    /// Longitude of the mesocyclone center (degrees).
    pub lon: f64,
    /// Range from the radar (km).
    pub range_km: f64,
    /// Azimuth from the radar (degrees).
    pub azimuth: f64,
    /// Rotational velocity (half the velocity difference across the couplet, m/s).
    pub rotational_velocity: f64,
    /// Diameter of the mesocyclone (km).
    pub diameter_km: f64,
    /// Strength rank (1 = weak, 5 = violent).
    pub strength_rank: u8,
    /// Elevation angle of the detection (degrees).
    pub elevation: f64,
}

/// A detected Tornado Vortex Signature (TVS).
///
/// A TVS is a very tight, intense rotational couplet in the lowest elevation
/// scans, strongly correlated with tornado occurrence.
#[derive(Debug, Clone)]
pub struct TVSDetection {
    /// Latitude of the TVS center (degrees).
    pub lat: f64,
    /// Longitude of the TVS center (degrees).
    pub lon: f64,
    /// Range from the radar (km).
    pub range_km: f64,
    /// Azimuth from the radar (degrees).
    pub azimuth: f64,
    /// Maximum gate-to-gate velocity difference (m/s).
    pub max_shear: f64,
    /// Lowest elevation angle where the TVS was detected (degrees).
    pub base_elevation: f64,
    /// Highest elevation angle where the TVS was detected (degrees).
    pub top_elevation: f64,
    /// Depth of the TVS signature through the volume (km).
    pub depth_km: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesocyclone_detection_fields() {
        let meso = MesocycloneDetection {
            lat: 35.0,
            lon: -97.0,
            range_km: 50.0,
            azimuth: 225.0,
            rotational_velocity: 25.0,
            diameter_km: 5.0,
            strength_rank: 3,
            elevation: 0.5,
        };
        assert!(meso.rotational_velocity > 0.0);
        assert!(meso.strength_rank <= 5);
    }

    #[test]
    fn test_tvs_detection_fields() {
        let tvs = TVSDetection {
            lat: 35.1,
            lon: -97.1,
            range_km: 30.0,
            azimuth: 200.0,
            max_shear: 60.0,
            base_elevation: 0.5,
            top_elevation: 3.4,
            depth_km: 4.0,
        };
        assert!(tvs.max_shear > 0.0);
        assert!(tvs.top_elevation > tvs.base_elevation);
    }

    #[test]
    fn test_detection_types_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MesocycloneDetection>();
        assert_send_sync::<TVSDetection>();
    }
}
