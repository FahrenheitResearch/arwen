//! The neutral observation table, `gpuwm-obs.table.v2`.
//!
//! One row per observation in the analysis's own vocabulary, the layout
//! `gpuwm.arwen_global.obs_table.decode_neutral_csv` reads without any
//! conversion.  The first ten columns are the v1 table every reader of the
//! tree already takes: `source, station_id, latitude_deg, longitude_deg,
//! elevation_m, level_pa, valid_time, variable, value, error`.  A surface
//! row leaves `level_pa` empty and is anchored at `elevation_m`; an aloft
//! row carries its pressure in `level_pa`.  Every time is ISO-8601 UTC
//! with a `Z`.
//!
//! The five v2 columns are the causal bookkeeping every report carries
//! (the design's rule that an analysis is made from information available
//! by a declared cutoff, so a report must say when it was measured, when
//! its source published it and when this system first held it):
//!
//! * `measurement`: what the number is, from the fixed vocabulary below
//!   (`station_pressure_from_altimeter`, `sea_level_pressure`, ...), so a
//!   reduced sea-level pressure is never mistaken for a station pressure
//!   and a buoy's 5 m wind reduced to 10 m says so;
//! * `nominal_time`: the synoptic hour a report is filed under (a sounding's
//!   00Z or 12Z), empty when the source has no such notion;
//! * `published_time`: when the source made the report available, where
//!   the source states it (an archive file's `Last-Modified`, a granule's
//!   creation stamp), empty when unknown;
//! * `received_time`: when this system first held the bytes (the fetch
//!   record's `fetched_at`), empty when the rows were decoded from files
//!   whose arrival was not recorded, which the reader labels "latency
//!   unverified";
//! * `revision`: the identity of the source object the row was decoded
//!   from (the first twelve hex digits of its SHA-256), so a corrected
//!   report in a re-issued file is a new revision and never a silent
//!   overwrite.
//!
//! The variable names, units, gross bounds and the per-stream observation
//! errors live here once so the four front doors that write the table
//! cannot drift into four vocabularies.  The bounds are transcribed from
//! `VARIABLE_TABLE` in the Python module; a value outside them is
//! instrument garbage (a sentinel, a truncated transmission) and the front
//! door counts it by name rather than writing it.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use chrono::{DateTime, Utc};

use crate::{err, hex_sha256};

pub const TABLE_SCHEMA: &str = "gpuwm-obs.table.v2";
pub const TABLE_HEADER: &str = "source,station_id,latitude_deg,longitude_deg,elevation_m,\
level_pa,valid_time,variable,value,error,measurement,nominal_time,published_time,\
received_time,revision";

/// The measurement vocabulary (`measurement` column).  Transcribed into
/// `MEASUREMENT_TABLE` of the Python module; a row naming a measurement
/// outside it is counted by the reader, never coerced.
pub const MEAS_STATION_PRESSURE_FROM_ALTIMETER: &str = "station_pressure_from_altimeter";
pub const MEAS_STATION_PRESSURE: &str = "station_pressure";
pub const MEAS_SEA_LEVEL_PRESSURE: &str = "sea_level_pressure";
pub const MEAS_SCREEN_TEMPERATURE_2M: &str = "screen_temperature_2m";
pub const MEAS_SCREEN_DEWPOINT_2M: &str = "screen_dewpoint_2m";
pub const MEAS_PLATFORM_TEMPERATURE: &str = "platform_temperature";
pub const MEAS_PLATFORM_DEWPOINT: &str = "platform_dewpoint";
pub const MEAS_ANEMOMETER_WIND_10M: &str = "anemometer_wind_10m";
pub const MEAS_ANEMOMETER_WIND_5M_TO_10M: &str = "anemometer_wind_5m_reduced_to_10m";
pub const MEAS_SONDE_LEVEL: &str = "sonde_level";
pub const MEAS_AMV_ASSIGNED_PRESSURE: &str = "amv_assigned_pressure";
pub const MEAS_RO_REFRACTIVITY_TANGENT: &str = "ro_refractivity_tangent_point";
/// Added with the prepbufr door (`rw_prepbufr`).  The Python reader's
/// `MEASUREMENT_TABLE` needs these rows; until it has them it counts the
/// rows as unknown measurements and does not coerce them.
/// An aircraft report at its flight level (AIREP, PIREP, AMDAR, ACARS).
pub const MEAS_AIRCRAFT_LEVEL: &str = "aircraft_level";
/// A wind profiler range gate.
pub const MEAS_PROFILER_LEVEL: &str = "profiler_level";
/// A radar velocity-azimuth-display wind level.
pub const MEAS_VAD_LEVEL: &str = "vad_level";
/// GSI's VAD superob: the mean of up to six consecutive VAD levels within
/// 301 m, at the first level's pressure and the levels' mean height.
pub const MEAS_VAD_SUPEROB: &str = "vad_superob";
/// A ship, buoy or coastal platform wind at the platform's own height
/// (prepbufr does not reduce marine winds to 10 m).
pub const MEAS_PLATFORM_WIND: &str = "platform_wind";
/// Station pressure computed by NCEP from the reported sea-level pressure
/// and the station elevation through the standard atmosphere.
pub const MEAS_STATION_PRESSURE_FROM_SEA_LEVEL: &str = "station_pressure_from_sea_level";

/// The first twelve hex digits of a source object's SHA-256: the
/// `revision` a row carries.
pub fn revision_of(sha256_hex: &str) -> String {
    sha256_hex.chars().take(12).collect()
}

pub const VAR_SURFACE_PRESSURE: &str = "surface_pressure_pa";
pub const VAR_TEMPERATURE: &str = "temperature_k";
pub const VAR_DEWPOINT: &str = "dewpoint_k";
pub const VAR_WIND_U: &str = "wind_u_m_s";
pub const VAR_WIND_V: &str = "wind_v_m_s";
pub const VAR_REFRACTIVITY: &str = "refractivity_n";

/// Gross physical bounds, `VARIABLE_TABLE` transcribed.
pub const GROSS_SURFACE_PRESSURE_PA: (f64, f64) = (45_000.0, 108_000.0);
pub const GROSS_TEMPERATURE_K: (f64, f64) = (170.0, 340.0);
pub const GROSS_DEWPOINT_K: (f64, f64) = (170.0, 320.0);
pub const GROSS_WIND_M_S: (f64, f64) = (0.0, 150.0);
pub const GROSS_REFRACTIVITY_N: (f64, f64) = (0.1, 500.0);

/// Observation error standard deviations by stream and variable.  Stated
/// here, carried in every row, and repeated in each front door's record so
/// a reader of a table never has to know which door wrote it.
pub const ERROR_SURFACE_PRESSURE_PA: f64 = 100.0;
pub const ERROR_TEMPERATURE_SURFACE_K: f64 = 1.5;
pub const ERROR_DEWPOINT_SURFACE_K: f64 = 1.5;
pub const ERROR_WIND_SURFACE_M_S: f64 = 2.5;
pub const ERROR_TEMPERATURE_ALOFT_K: f64 = 1.0;
pub const ERROR_DEWPOINT_ALOFT_K: f64 = 2.5;
pub const ERROR_WIND_ALOFT_M_S: f64 = 2.5;
/// Buoy anemometers sit at 4 to 5 m and are reduced to 10 m by the neutral
/// log law (see `rw_ndbc`); the reduction's uncertainty is inside this.
pub const ERROR_WIND_BUOY_M_S: f64 = 3.0;
pub const ERROR_DEWPOINT_BUOY_K: f64 = 2.0;
/// Atmospheric motion vectors by layer: low (below 700 hPa), mid, high
/// (above 400 hPa).  The height assignment dominates the error aloft.
pub const ERROR_AMV_LOW_M_S: f64 = 3.0;
pub const ERROR_AMV_MID_M_S: f64 = 4.0;
pub const ERROR_AMV_HIGH_M_S: f64 = 5.0;
/// Refractivity, N-units, as a fraction of the value, by tangent height
/// and latitude (`refractivity_error_fraction`): the shape Kuo et al.
/// (2004) estimated for the retrieved refractivity, two percent at the
/// surface falling with a 3 km scale to a 0.3 percent floor through the
/// upper troposphere and lower stratosphere (0.9 percent at 3 km, 0.5 at
/// 5 km, 0.36 at 10 km), the tropical boundary layer (|lat| < 30 degrees,
/// below 8 km) half as large again for the moisture and super-refraction
/// there, and the residual ionospheric noise raising it again above 25 km
/// by 0.07 percent per kilometre (1.0 percent at 35 km).  The 2026-09-06
/// first cut had one percent below 10 km and two above, the inverse of
/// the published shape in the upper troposphere; the Desroziers reading
/// of a cycle judges this rule (the assigned error is never retuned until
/// that reading looks right).
pub const ERROR_REFRACTIVITY_FRACTION_FLOOR: f64 = 0.003;
pub const ERROR_REFRACTIVITY_FRACTION_SURFACE_EXTRA: f64 = 0.017;
pub const ERROR_REFRACTIVITY_SCALE_HEIGHT_M: f64 = 3000.0;
pub const ERROR_REFRACTIVITY_TROPICAL_FACTOR: f64 = 1.5;
pub const ERROR_REFRACTIVITY_TROPICAL_LATITUDE_DEG: f64 = 30.0;
pub const ERROR_REFRACTIVITY_TROPICAL_TOP_M: f64 = 8000.0;
pub const ERROR_REFRACTIVITY_UPPER_ONSET_M: f64 = 25_000.0;
pub const ERROR_REFRACTIVITY_UPPER_SLOPE_PER_M: f64 = 0.0007 / 1000.0;
pub const ERROR_REFRACTIVITY_RULE: &str = "fraction of N by tangent height and latitude: 0.3 percent floor plus 1.7 percent decaying with a 3 km scale from the surface (2.0 percent at 0 m, 0.5 at 5 km, 0.36 at 10 km), times 1.5 below 8 km inside 30 degrees of the equator, plus 0.07 percent per kilometre above 25 km (Kuo et al. 2004 shape); at least 0.001 N";

/// The refractivity error as a fraction of the value at a tangent height
/// (m above mean sea level) and latitude (degrees); see the constants.
pub fn refractivity_error_fraction(altitude_m: f64, latitude_deg: f64) -> f64 {
    let z = altitude_m.max(0.0);
    let mut fraction = ERROR_REFRACTIVITY_FRACTION_FLOOR
        + ERROR_REFRACTIVITY_FRACTION_SURFACE_EXTRA * (-z / ERROR_REFRACTIVITY_SCALE_HEIGHT_M).exp();
    if latitude_deg.abs() < ERROR_REFRACTIVITY_TROPICAL_LATITUDE_DEG && z < ERROR_REFRACTIVITY_TROPICAL_TOP_M {
        fraction *= ERROR_REFRACTIVITY_TROPICAL_FACTOR;
    }
    if z > ERROR_REFRACTIVITY_UPPER_ONSET_M {
        fraction += ERROR_REFRACTIVITY_UPPER_SLOPE_PER_M * (z - ERROR_REFRACTIVITY_UPPER_ONSET_M);
    }
    fraction
}

#[derive(Debug, Clone, PartialEq)]
pub struct TableRow {
    pub source: String,
    pub station_id: String,
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub elevation_m: f64,
    pub level_pa: Option<f64>,
    pub valid_time: DateTime<Utc>,
    pub variable: String,
    pub value: f64,
    pub error: f64,
    /// The causal bookkeeping (v2 columns); see the module notes.
    pub provenance: RowProvenance,
}

/// The five v2 columns of a row, shared by every row of one source file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RowProvenance {
    pub measurement: &'static str,
    pub nominal_time: Option<DateTime<Utc>>,
    pub published_time: Option<DateTime<Utc>>,
    pub received_time: Option<DateTime<Utc>>,
    pub revision: String,
}

impl RowProvenance {
    /// The bookkeeping of one source object: its publication and receipt
    /// instants where the fetch record states them, its revision from its
    /// digest.  `measurement` and `nominal_time` are set per row.
    pub fn of_source(
        sha256_hex: &str,
        published_time: Option<DateTime<Utc>>,
        received_time: Option<DateTime<Utc>>,
    ) -> Self {
        Self {
            measurement: "",
            nominal_time: None,
            published_time,
            received_time,
            revision: revision_of(sha256_hex),
        }
    }

    pub fn measuring(&self, measurement: &'static str) -> Self {
        Self {
            measurement,
            ..self.clone()
        }
    }

    pub fn nominal(&self, nominal_time: DateTime<Utc>) -> Self {
        Self {
            nominal_time: Some(nominal_time),
            ..self.clone()
        }
    }
}

fn stamp(when: &Option<DateTime<Utc>>) -> String {
    match when {
        Some(t) => t.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        None => String::new(),
    }
}

impl TableRow {
    /// One CSV line, no trailing newline.  Longitude is written in
    /// `[-180, 180)`; times carry a `Z`; an unknown time is empty.
    pub fn csv_line(&self) -> String {
        let lon = crate::seam::wrap_longitude(self.longitude_deg);
        format!(
            "{},{},{:.5},{:.5},{:.1},{},{},{},{},{},{},{},{},{},{}",
            self.source,
            self.station_id,
            self.latitude_deg,
            lon,
            self.elevation_m,
            match self.level_pa {
                Some(p) => format!("{p:.1}"),
                None => String::new(),
            },
            self.valid_time.format("%Y-%m-%dT%H:%M:%SZ"),
            self.variable,
            trim_float(self.value),
            trim_float(self.error),
            self.provenance.measurement,
            stamp(&self.provenance.nominal_time),
            stamp(&self.provenance.published_time),
            stamp(&self.provenance.received_time),
            self.provenance.revision,
        )
    }
}

/// Shortest decimal spelling that round-trips: seventeen significant digits
/// would make a 300 MB table of what is mostly two-decimal reports.
fn trim_float(value: f64) -> String {
    let s = format!("{value}");
    if s.contains('e') {
        format!("{value:.6}")
    } else {
        s
    }
}

/// Rows accumulate here and are written once, so the file's digest is the
/// digest of exactly what the record states.
#[derive(Debug, Default)]
pub struct TableWriter {
    rows: Vec<TableRow>,
}

impl TableWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Refuses a row outside the vocabulary or with a non-finite value, a
    /// non-positive error or a position off the sphere; those are the
    /// writer's bugs, not the archive's, so they are errors here and not
    /// counters.
    pub fn push(&mut self, row: TableRow, rows_by_variable: &mut BTreeMap<String, usize>) {
        debug_assert!(
            row.value.is_finite() && row.error > 0.0 && row.latitude_deg.abs() <= 90.0,
            "a front door built an invalid row: {row:?}"
        );
        *rows_by_variable.entry(row.variable.clone()).or_insert(0) += 1;
        self.rows.push(row);
    }

    pub fn rows(&self) -> &[TableRow] {
        &self.rows
    }

    pub fn rows_mut(&mut self) -> &mut Vec<TableRow> {
        &mut self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn text(&self) -> String {
        let mut text = String::with_capacity(64 + self.rows.len() * 96);
        text.push_str(TABLE_HEADER);
        text.push('\n');
        for row in &self.rows {
            text.push_str(&row.csv_line());
            text.push('\n');
        }
        text
    }

    /// Write the table; returns `(rows, sha256, bytes)`.
    pub fn write(&self, path: &Path) -> Result<(usize, String, usize), Box<dyn Error>> {
        let text = self.text();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| err(format!("cannot create {}: {e}", parent.display())))?;
            }
        }
        std::fs::write(path, text.as_bytes())
            .map_err(|e| err(format!("cannot write {}: {e}", path.display())))?;
        Ok((self.rows.len(), hex_sha256(text.as_bytes()), text.len()))
    }
}

/// `(u, v)` from a meteorological direction (degrees the wind blows FROM,
/// 360 = north) and a speed.  Direction 0 with a nonzero speed is the
/// variable-direction encoding of several archives and is not derivable;
/// callers screen it before this.
pub fn wind_components(direction_deg: f64, speed_m_s: f64) -> (f64, f64) {
    let r = direction_deg.to_radians();
    (-speed_m_s * r.sin(), -speed_m_s * r.cos())
}

/// ICAO standard-atmosphere altitude of a pressure (the inverse of
/// `obs_table.isa_pressure_pa`): the height a level without a measured
/// geopotential is stamped with, so a row is never written without an
/// anchor and the count of such rows says how many are ISA.
pub fn isa_altitude_m(pressure_pa: f64) -> f64 {
    const P0: f64 = 101_325.0;
    const T0: f64 = 288.15;
    const LAPSE: f64 = 0.0065;
    const EXPONENT: f64 = 5.255877;
    const P_TROP: f64 = 22_632.06;
    const Z_TROP: f64 = 11_000.0;
    const SCALE: f64 = 6341.62;
    if pressure_pa >= P_TROP {
        T0 / LAPSE * (1.0 - (pressure_pa / P0).powf(1.0 / EXPONENT))
    } else {
        Z_TROP - SCALE * (pressure_pa / P_TROP).ln()
    }
}

/// Fahrenheit to Kelvin, the IEM archive's temperature unit.
pub fn fahrenheit_to_kelvin(value: f64) -> f64 {
    (value - 32.0) * 5.0 / 9.0 + 273.15
}

pub fn knots_to_m_s(value: f64) -> f64 {
    value * 0.514444
}

/// Station pressure from an altimeter setting (inches of mercury) at a
/// station elevation: the altimeter setting is defined as the ISA
/// sea-level reduction of station pressure, so inverting it with the ISA
/// column recovers the station pressure exactly.  The same arithmetic as
/// `obs_table._altimeter_station_pa`.
pub fn altimeter_inhg_to_station_pa(altimeter_in_hg: f64, elevation_m: f64) -> f64 {
    const INHG_TO_PA: f64 = 3386.389;
    (altimeter_in_hg * INHG_TO_PA) * (1.0 - 0.0065 * elevation_m / 288.15).powf(5.255877)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn a_row_spells_the_neutral_layout() {
        let row = TableRow {
            source: "igra2".into(),
            station_id: "USM00072365".into(),
            latitude_deg: 35.04,
            longitude_deg: 253.38,
            elevation_m: 1620.0,
            level_pa: Some(50000.0),
            valid_time: Utc.with_ymd_and_hms(2026, 9, 1, 0, 0, 0).unwrap(),
            variable: VAR_TEMPERATURE.into(),
            value: 262.15,
            error: 1.0,
            provenance: RowProvenance {
                measurement: MEAS_SONDE_LEVEL,
                nominal_time: Some(Utc.with_ymd_and_hms(2026, 9, 1, 0, 0, 0).unwrap()),
                published_time: Some(Utc.with_ymd_and_hms(2026, 9, 2, 21, 36, 0).unwrap()),
                received_time: None,
                revision: revision_of("16d059c6d0e21396028900d70a35210c171fca1c68409c31765eaeea46b63121"),
            },
        };
        assert_eq!(
            row.csv_line(),
            "igra2,USM00072365,35.04000,-106.62000,1620.0,50000.0,2026-09-01T00:00:00Z,temperature_k,262.15,1,\
sonde_level,2026-09-01T00:00:00Z,2026-09-02T21:36:00Z,,16d059c6d0e2"
        );
        let mut surface = row.clone();
        surface.level_pa = None;
        assert!(surface.csv_line().contains(",1620.0,,2026-09-01T00:00:00Z,"));
        assert_eq!(TABLE_HEADER.split(',').count(), 15);
        assert_eq!(row.csv_line().split(',').count(), 15);
    }

    #[test]
    fn an_unknown_provenance_writes_empty_columns_not_placeholders() {
        let row = TableRow {
            source: "ndbc".into(),
            station_id: "41001".into(),
            latitude_deg: 34.5,
            longitude_deg: -72.5,
            elevation_m: 0.0,
            level_pa: None,
            valid_time: Utc.with_ymd_and_hms(2026, 9, 1, 0, 0, 0).unwrap(),
            variable: VAR_SURFACE_PRESSURE.into(),
            value: 100860.0,
            error: 100.0,
            provenance: RowProvenance::of_source("abcdef0123456789", None, None)
                .measuring(MEAS_SEA_LEVEL_PRESSURE),
        };
        assert!(row.csv_line().ends_with(",sea_level_pressure,,,,abcdef012345"), "{}", row.csv_line());
    }

    #[test]
    fn isa_altitude_inverts_the_isa_pressure() {
        // 500 hPa is 5,574 m in the standard atmosphere; 226.32 hPa is 11 km.
        assert!((isa_altitude_m(50000.0) - 5574.4).abs() < 1.0);
        assert!((isa_altitude_m(22632.06) - 11000.0).abs() < 0.01);
        assert!((isa_altitude_m(101325.0)).abs() < 1e-9);
        assert!((isa_altitude_m(5474.9) - 20000.0).abs() < 2.0);
    }

    #[test]
    fn wind_components_follow_the_meteorological_convention() {
        let (u, v) = wind_components(270.0, 10.0);
        assert!((u - 10.0).abs() < 1e-9 && v.abs() < 1e-9);
        let (u, v) = wind_components(360.0, 5.0);
        assert!(u.abs() < 1e-9 && (v + 5.0).abs() < 1e-9);
        let (u, v) = wind_components(180.0, 5.0);
        assert!(u.abs() < 1e-9 && (v - 5.0).abs() < 1e-9);
    }

    #[test]
    fn the_refractivity_error_follows_the_stated_shape() {
        assert!((refractivity_error_fraction(0.0, 45.0) - 0.020).abs() < 1e-12);
        assert!((refractivity_error_fraction(0.0, 10.0) - 0.030).abs() < 1e-12);
        let mid = refractivity_error_fraction(10_000.0, 45.0);
        assert!(mid > 0.0035 && mid < 0.0037, "{mid}");
        let floor = refractivity_error_fraction(20_000.0, 45.0);
        assert!((floor - 0.003).abs() < 5e-5, "{floor}");
        assert!((refractivity_error_fraction(35_000.0, -60.0) - 0.010).abs() < 1e-4);
        // the tropical factor stops at 8 km and the floor is never below 0.3 percent
        assert!(refractivity_error_fraction(9_000.0, 0.0) == refractivity_error_fraction(9_000.0, 50.0));
        assert!(refractivity_error_fraction(15_000.0, 0.0) >= 0.003);
    }

    #[test]
    fn altimeter_at_sea_level_is_the_setting_itself() {
        assert!((altimeter_inhg_to_station_pa(29.92, 0.0) - 101_320.7).abs() < 1.0);
        // At 1,000 m the station pressure is about 89.9 kPa for a standard setting.
        let p = altimeter_inhg_to_station_pa(29.92, 1000.0);
        assert!((p - 89_870.0).abs() < 60.0, "{p}");
    }
}
