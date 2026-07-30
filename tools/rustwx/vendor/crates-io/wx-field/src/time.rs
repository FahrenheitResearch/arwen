/// Time-related types for weather data.

use chrono::{DateTime, Utc};

/// A valid time — when weather data is representative.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ValidTime(pub DateTime<Utc>);

impl ValidTime {
    pub fn new(dt: DateTime<Utc>) -> Self {
        Self(dt)
    }

    pub fn datetime(&self) -> &DateTime<Utc> {
        &self.0
    }
}

impl From<DateTime<Utc>> for ValidTime {
    fn from(dt: DateTime<Utc>) -> Self {
        Self(dt)
    }
}

/// A forecast hour — offset from the model initialization (reference) time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ForecastHour(pub u32);

impl ForecastHour {
    pub fn new(hour: u32) -> Self {
        Self(hour)
    }

    pub fn hours(&self) -> u32 {
        self.0
    }

    /// Is this an analysis (forecast hour 0)?
    pub fn is_analysis(&self) -> bool {
        self.0 == 0
    }
}

impl From<u32> for ForecastHour {
    fn from(hour: u32) -> Self {
        Self(hour)
    }
}

/// A model run — combination of reference time and forecast hour.
#[derive(Debug, Clone)]
pub struct ModelRun {
    /// Model initialization time.
    pub reference_time: DateTime<Utc>,
    /// Forecast hour offset.
    pub forecast_hour: ForecastHour,
}

impl ModelRun {
    pub fn new(reference_time: DateTime<Utc>, forecast_hour: u32) -> Self {
        Self {
            reference_time,
            forecast_hour: ForecastHour(forecast_hour),
        }
    }

    /// Compute the valid time from reference time + forecast hour.
    pub fn valid_time(&self) -> ValidTime {
        let dt = self.reference_time + chrono::Duration::hours(self.forecast_hour.0 as i64);
        ValidTime(dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_forecast_hour() {
        let fh = ForecastHour::new(6);
        assert_eq!(fh.hours(), 6);
        assert!(!fh.is_analysis());
        assert!(ForecastHour::new(0).is_analysis());
    }

    #[test]
    fn test_model_run_valid_time() {
        let ref_time = Utc.with_ymd_and_hms(2024, 1, 15, 12, 0, 0).unwrap();
        let run = ModelRun::new(ref_time, 6);
        let vt = run.valid_time();
        let expected = Utc.with_ymd_and_hms(2024, 1, 15, 18, 0, 0).unwrap();
        assert_eq!(*vt.datetime(), expected);
    }

    #[test]
    fn test_time_types_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ValidTime>();
        assert_send_sync::<ForecastHour>();
        assert_send_sync::<ModelRun>();
    }
}
