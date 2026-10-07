//! Solar zenith cosine (specification section 4.1).
//!
//! The Astronomical Almanac's approximate solar position as published in
//! Solar Energy 40 (1988) 227-235 [R4] (about 0.01 degree for 1950 to 2050):
//!
//! ```text
//! n    = JD - 2451545.0
//! L    = 280.460 + 0.9856474 n            mean longitude, deg
//! g    = 357.528 + 0.9856003 n            mean anomaly, deg
//! l    = L + 1.915 sin g + 0.020 sin 2g   ecliptic longitude, deg
//! eps  = 23.439 - 0.0000004 n             obliquity, deg
//! ra   = atan2(cos eps sin l, cos l)      right ascension
//! dec  = asin(sin eps sin l)              declination
//! gmst = 6.697375 + 0.0657098242 n + UT   hours (UT = hours since 0 UT)
//! ha   = 15 gmst + lon - ra               hour angle, deg
//! cosz = sin dec sin lat + cos dec cos lat cos ha
//! ```
//!
//! Everything but the last two lines depends only on the valid time, so the
//! host evaluates it once per frame ([`SolarFrame`]) with binary64 basic
//! operations and the shared maths library (src/math.rs); the per-cell
//! part ([`cosz`]) is binary32 and runs identically on both devices.

use crate::consts::DEG2RAD;
use crate::thermo::{cosf, sinf};

/// A UTC instant, as written in a WRF history's `Times` record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UtcTime {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
}

impl UtcTime {
    /// Parse `YYYY-MM-DD_hh:mm:ss` (also accepts `T` or a space as the
    /// separator).
    pub fn parse_wrf(text: &str) -> Option<Self> {
        let t = text.trim_end_matches('\0').trim();
        if t.len() < 19 {
            return None;
        }
        let b = t.as_bytes();
        if b[4] != b'-' || b[7] != b'-' || !matches!(b[10], b'_' | b'T' | b' ') || b[13] != b':' || b[16] != b':' {
            return None;
        }
        let num = |a: usize, z: usize| t.get(a..z)?.parse::<u32>().ok();
        let time = Self {
            year: num(0, 4)? as i32,
            month: num(5, 7)?,
            day: num(8, 10)?,
            hour: num(11, 13)?,
            minute: num(14, 16)?,
            second: num(17, 19)?,
        };
        let ok = (1..=12).contains(&time.month)
            && (1..=31).contains(&time.day)
            && time.hour < 24
            && time.minute < 60
            && time.second < 61;
        ok.then_some(time)
    }

    /// Julian date (days, binary64) of this instant.  Gregorian calendar
    /// day number by the standard integer formula, then the
    /// fraction of the day counted from noon.
    pub fn julian_date(&self) -> f64 {
        let y = self.year as i64;
        let m = self.month as i64;
        let d = self.day as i64;
        let a = (14 - m) / 12;
        let yy = y + 4800 - a;
        let mm = m + 12 * a - 3;
        let jdn = d + (153 * mm + 2) / 5 + 365 * yy + yy / 4 - yy / 100 + yy / 400 - 32045;
        let seconds = (self.hour * 3600 + self.minute * 60 + self.second) as f64;
        jdn as f64 - 0.5 + seconds / 86400.0
    }

    /// Hours since 0 UT.
    pub fn ut_hours(&self) -> f64 {
        (self.hour * 3600 + self.minute * 60 + self.second) as f64 / 3600.0
    }
}

/// Per-frame solar scalars handed to the per-cell function and the kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolarFrame {
    /// Declination, degrees (binary64, for reports and tests).
    pub declination_deg: f64,
    /// Hour angle at longitude 0, degrees in [0, 360) (binary64).
    pub hour_angle0_deg: f64,
    pub sin_dec: f32,
    pub cos_dec: f32,
    pub ha0_deg: f32,
}

fn wrap(x: f64, period: f64) -> f64 {
    let r = x % period;
    if r < 0.0 { r + period } else { r }
}

fn sin64(rad: f64) -> f64 {
    crate::math::sin(rad)
}

fn cos64(rad: f64) -> f64 {
    crate::math::cos(rad)
}

impl SolarFrame {
    /// Evaluate [R4] at a valid time.
    pub fn at(time: &UtcTime) -> Self {
        let rad = std::f64::consts::PI / 180.0;
        let n = time.julian_date() - 2451545.0;
        let mean_lon = wrap(280.460 + 0.9856474 * n, 360.0);
        let anomaly = wrap(357.528 + 0.9856003 * n, 360.0) * rad;
        let ecl = wrap(
            mean_lon + 1.915 * sin64(anomaly) + 0.020 * sin64(2.0 * anomaly),
            360.0,
        ) * rad;
        let obliquity = (23.439 - 0.0000004 * n) * rad;
        let ra = crate::math::atan2(cos64(obliquity) * sin64(ecl), cos64(ecl));
        let ra_deg = wrap(ra / rad, 360.0);
        let dec = crate::math::asin(sin64(obliquity) * sin64(ecl));
        let gmst = wrap(6.697375 + 0.0657098242 * n + time.ut_hours(), 24.0);
        let ha0 = wrap(15.0 * gmst - ra_deg, 360.0);
        let sin_dec = sin64(dec) as f32;
        let cos_dec = cos64(dec) as f32;
        Self {
            declination_deg: dec / rad,
            hour_angle0_deg: ha0,
            sin_dec,
            cos_dec,
            ha0_deg: ha0 as f32,
        }
    }
}

/// Solar zenith cosine at one point (binary32), twin of `wp_cosz`.
#[inline]
pub fn cosz(lat_deg: f32, lon_deg: f32, sin_dec: f32, cos_dec: f32, ha0_deg: f32) -> f32 {
    let mut ha = ha0_deg + lon_deg;
    if ha > 180.0 {
        ha = ha - 360.0;
    }
    if ha < -180.0 {
        ha = ha + 360.0;
    }
    let lat = lat_deg * DEG2RAD;
    let har = ha * DEG2RAD;
    sin_dec * sinf(lat) + cos_dec * cosf(lat) * cosf(har)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(y: i32, mo: u32, d: u32, h: u32, mi: u32, s: u32) -> UtcTime {
        UtcTime { year: y, month: mo, day: d, hour: h, minute: mi, second: s }
    }

    #[test]
    fn julian_date_epochs() {
        assert_eq!(t(2000, 1, 1, 12, 0, 0).julian_date(), 2451545.0);
        assert_eq!(t(1970, 1, 1, 0, 0, 0).julian_date(), 2440587.5);
    }

    #[test]
    fn parses_wrf_times() {
        assert_eq!(UtcTime::parse_wrf("2026-10-03_22:00:00"), Some(t(2026, 10, 3, 22, 0, 0)));
        assert_eq!(UtcTime::parse_wrf("2026-10-03T22:30:15"), Some(t(2026, 10, 3, 22, 30, 15)));
        assert_eq!(UtcTime::parse_wrf("2026-13-03_22:00:00"), None);
    }

    #[test]
    fn solstice_and_equinox_declination() {
        // June solstice 2024-06-20 20:51 UT: declination = obliquity.
        let s = SolarFrame::at(&t(2024, 6, 20, 20, 51, 0));
        assert!((s.declination_deg - 23.44).abs() < 0.02, "{}", s.declination_deg);
        // March equinox 2024-03-20 03:06 UT: declination 0.
        let e = SolarFrame::at(&t(2024, 3, 20, 3, 6, 0));
        assert!(e.declination_deg.abs() < 0.02, "{}", e.declination_deg);
    }

    #[test]
    fn equation_of_time_early_november() {
        // Solar transit at Greenwich on 2024-11-03 is near 11:43:35 UT
        // (equation of time +16.4 min): the hour angle there is 0.
        let s = SolarFrame::at(&t(2024, 11, 3, 11, 43, 35));
        let ha = if s.hour_angle0_deg > 180.0 { s.hour_angle0_deg - 360.0 } else { s.hour_angle0_deg };
        assert!(ha.abs() < 0.1, "{ha}");
    }

    #[test]
    fn subsolar_point_has_the_sun_overhead() {
        let s = SolarFrame::at(&t(2026, 10, 3, 22, 0, 0));
        let lat = s.declination_deg as f32;
        let lon = -s.ha0_deg;
        let c = cosz(lat, lon, s.sin_dec, s.cos_dec, s.ha0_deg);
        assert!(c > 0.99999, "{c}");
        let anti = cosz(-lat, lon + 180.0, s.sin_dec, s.cos_dec, s.ha0_deg);
        assert!(anti < -0.99999, "{anti}");
    }
}
