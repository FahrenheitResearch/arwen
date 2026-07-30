/// Field metadata types — units, levels, data sources.

use chrono::{DateTime, Utc};

/// Metadata associated with a weather field.
#[derive(Debug, Clone)]
pub struct FieldMeta {
    /// Variable name (e.g., "TMP", "UGRD", "REFL").
    pub variable: String,
    /// Human-readable long name (e.g., "Temperature", "U-Component of Wind").
    pub long_name: Option<String>,
    /// Physical units.
    pub units: Units,
    /// Vertical level.
    pub level: Level,
    /// Data source.
    pub source: DataSource,
    /// Valid time (when this data is valid).
    pub valid_time: Option<DateTime<Utc>>,
    /// Reference/analysis time.
    pub reference_time: Option<DateTime<Utc>>,
    /// Forecast hour (hours from reference to valid time).
    pub forecast_hour: Option<u32>,
}

impl FieldMeta {
    /// Create minimal metadata with just a variable name and units.
    pub fn new(variable: impl Into<String>, units: Units) -> Self {
        Self {
            variable: variable.into(),
            long_name: None,
            units,
            level: Level::Surface,
            source: DataSource::Unknown,
            valid_time: None,
            reference_time: None,
            forecast_hour: None,
        }
    }
}

/// Physical units for field data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Units {
    Kelvin,
    Celsius,
    Fahrenheit,
    MetersPerSecond,
    Knots,
    MilesPerHour,
    Pascal,
    Hectopascal,
    Millibar,
    Meters,
    Feet,
    Kilometers,
    Percent,
    KgPerKg,
    KgPerM2,
    Dbz,
    DegreesTrue,
    Dimensionless,
    Other(String),
}

/// Vertical level specification.
#[derive(Debug, Clone, PartialEq)]
pub enum Level {
    /// Surface (2m, 10m, etc.)
    Surface,
    /// Mean sea level
    MeanSeaLevel,
    /// Pressure level in hPa
    Pressure(f64),
    /// Height above ground in meters
    HeightAboveGround(f64),
    /// Sigma level (0..1)
    Sigma(f64),
    /// Hybrid level index
    Hybrid(u32),
    /// Entire atmosphere (e.g., PWAT)
    EntireAtmosphere,
    /// Top of atmosphere
    TopOfAtmosphere,
    /// Tropopause
    Tropopause,
    /// Maximum wind level
    MaxWind,
    /// Specific elevation angle in degrees (radar)
    Elevation(f64),
    /// Other level type
    Other(String),
}

/// Data source identification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataSource {
    /// NOAA HRRR model
    Hrrr,
    /// NOAA GFS model
    Gfs,
    /// NOAA NAM model
    Nam,
    /// NOAA RAP model
    Rap,
    /// ECMWF IFS / ERA5
    Ecmwf,
    /// WRF simulation output
    Wrf,
    /// NEXRAD radar observation
    Nexrad,
    /// TDWR radar observation
    Tdwr,
    /// Radiosonde observation
    Radiosonde,
    /// Unknown source
    Unknown,
    /// Other named source
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_meta_new() {
        let meta = FieldMeta::new("TMP", Units::Kelvin);
        assert_eq!(meta.variable, "TMP");
        assert_eq!(meta.units, Units::Kelvin);
        assert_eq!(meta.level, Level::Surface);
    }

    #[test]
    fn test_meta_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FieldMeta>();
        assert_send_sync::<Units>();
        assert_send_sync::<Level>();
        assert_send_sync::<DataSource>();
    }
}
