/// Sounding (vertical profile) types for radiosonde and model soundings.

use chrono::{DateTime, Utc};

/// A single level in a sounding profile.
#[derive(Debug, Clone)]
pub struct SoundingLevel {
    /// Pressure in hPa.
    pub pressure: f64,
    /// Height above sea level in meters (may be NaN if missing).
    pub height: f64,
    /// Temperature in degrees Celsius (may be NaN if missing).
    pub temperature: f64,
    /// Dewpoint in degrees Celsius (may be NaN if missing).
    pub dewpoint: f64,
    /// Wind direction in degrees true (may be NaN if missing).
    pub wind_dir: f64,
    /// Wind speed in knots (may be NaN if missing).
    pub wind_speed: f64,
}

/// A complete vertical sounding profile.
#[derive(Debug, Clone)]
pub struct SoundingProfile {
    /// Station identifier (e.g., "OUN", "72451").
    pub station_id: String,
    /// Station latitude in degrees.
    pub lat: f64,
    /// Station longitude in degrees.
    pub lon: f64,
    /// Station elevation in meters.
    pub elevation: f64,
    /// Observation/valid time.
    pub time: Option<DateTime<Utc>>,
    /// Sounding levels, ordered from surface (highest pressure) upward.
    pub levels: Vec<SoundingLevel>,
}

impl SoundingProfile {
    /// Number of levels in the sounding.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Surface pressure in hPa (first level), or NaN if empty.
    pub fn surface_pressure(&self) -> f64 {
        self.levels.first().map(|l| l.pressure).unwrap_or(f64::NAN)
    }

    /// Find the level nearest to a given pressure.
    pub fn nearest_level(&self, pressure: f64) -> Option<&SoundingLevel> {
        self.levels.iter().min_by(|a, b| {
            let da = (a.pressure - pressure).abs();
            let db = (b.pressure - pressure).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sounding() -> SoundingProfile {
        SoundingProfile {
            station_id: "OUN".to_string(),
            lat: 35.25,
            lon: -97.47,
            elevation: 357.0,
            time: None,
            levels: vec![
                SoundingLevel {
                    pressure: 965.0, height: 357.0,
                    temperature: 25.0, dewpoint: 18.0,
                    wind_dir: 180.0, wind_speed: 10.0,
                },
                SoundingLevel {
                    pressure: 850.0, height: 1500.0,
                    temperature: 15.0, dewpoint: 10.0,
                    wind_dir: 200.0, wind_speed: 25.0,
                },
                SoundingLevel {
                    pressure: 500.0, height: 5500.0,
                    temperature: -10.0, dewpoint: -20.0,
                    wind_dir: 270.0, wind_speed: 50.0,
                },
            ],
        }
    }

    #[test]
    fn test_sounding_basics() {
        let snd = test_sounding();
        assert_eq!(snd.num_levels(), 3);
        assert!((snd.surface_pressure() - 965.0).abs() < 0.01);
    }

    #[test]
    fn test_nearest_level() {
        let snd = test_sounding();
        let level = snd.nearest_level(860.0).unwrap();
        assert!((level.pressure - 850.0).abs() < 0.01);
    }

    #[test]
    fn test_sounding_types_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SoundingLevel>();
        assert_send_sync::<SoundingProfile>();
    }
}
