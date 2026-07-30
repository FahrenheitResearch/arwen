/// Radial (radar) field types -- sweeps, radials, and full volumes.

use crate::site::RadarSite;
use chrono::{DateTime, Utc};

/// A single radial (ray) of radar data at a specific azimuth.
#[derive(Debug, Clone)]
pub struct Radial {
    /// Azimuth angle in degrees (0 = north, 90 = east).
    pub azimuth: f64,
    /// Gate data values (e.g., reflectivity in dBZ, velocity in m/s).
    pub gates: Vec<f64>,
}

/// A single radar sweep (one full rotation at a fixed elevation).
#[derive(Debug, Clone)]
pub struct RadialSweep {
    /// Elevation angle in degrees above horizon.
    pub elevation: f64,
    /// Range to the first gate in meters.
    pub range_first: f64,
    /// Gate spacing in meters.
    pub gate_spacing: f64,
    /// Number of gates per radial.
    pub num_gates: usize,
    /// Individual radials (one per azimuth).
    pub radials: Vec<Radial>,
    /// Scan time.
    pub time: Option<DateTime<Utc>>,
}

impl RadialSweep {
    /// Number of radials in this sweep.
    pub fn num_radials(&self) -> usize {
        self.radials.len()
    }

    /// Range to the last gate in meters.
    pub fn max_range(&self) -> f64 {
        self.range_first + (self.num_gates as f64 - 1.0) * self.gate_spacing
    }
}

/// A complete radial field (one or more sweeps for a single product).
#[derive(Debug, Clone)]
pub struct RadialField {
    /// Product name (e.g., "REF", "VEL", "SW", "ZDR").
    pub product: String,
    /// Radar site information.
    pub site: RadarSite,
    /// Sweeps ordered by increasing elevation angle.
    pub sweeps: Vec<RadialSweep>,
    /// Volume scan time.
    pub time: Option<DateTime<Utc>>,
}

impl RadialField {
    /// Number of sweeps in this volume.
    pub fn num_sweeps(&self) -> usize {
        self.sweeps.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_site() -> RadarSite {
        RadarSite {
            id: "KTLX".to_string(),
            name: "Oklahoma City".to_string(),
            lat: 35.333,
            lon: -97.278,
            elevation: 370.0,
        }
    }

    #[test]
    fn test_radial_sweep() {
        let sweep = RadialSweep {
            elevation: 0.5,
            range_first: 2125.0,
            gate_spacing: 250.0,
            num_gates: 1832,
            radials: vec![
                Radial { azimuth: 0.0, gates: vec![0.0; 1832] },
                Radial { azimuth: 1.0, gates: vec![0.0; 1832] },
            ],
            time: None,
        };
        assert_eq!(sweep.num_radials(), 2);
        assert!((sweep.max_range() - 459875.0).abs() < 0.1);
    }

    #[test]
    fn test_radial_field() {
        let field = RadialField {
            product: "REF".to_string(),
            site: test_site(),
            sweeps: vec![],
            time: None,
        };
        assert_eq!(field.num_sweeps(), 0);
        assert_eq!(field.site.id, "KTLX");
    }

    #[test]
    fn test_radial_types_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Radial>();
        assert_send_sync::<RadialSweep>();
        assert_send_sync::<RadialField>();
    }
}
