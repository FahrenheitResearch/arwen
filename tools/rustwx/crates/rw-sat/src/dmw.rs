//! ABI Level 2 Derived Motion Winds (`ABI-L2-DMW{C,F,M}`): the wind
//! vectors the GOES-R ground segment tracks in successive images of one
//! band and assigns a pressure to.  gpuwm divergence from the rusty-weather
//! source tree (recorded in `tools/rustwx/VENDOR.md`): the source tree has
//! no DMW reader; this module is new and follows the cloud-product
//! module's shape, one product family, its variable table verified against
//! a downloaded granule, and a DQF gate that counts what it condemns.
//!
//! Variable table, verified 2026-09-06 against
//! `OR_ABI-L2-DMWF-M6C14_G19_s20262441800203_e20262441809511_c20262441824411.nc`
//! (38,899 vectors, dimension `nMeasures`): `lat` (F64, degrees_north),
//! `lon` (F64, degrees_east), `pressure` (F32, hPa), `wind_speed` (F32,
//! m s-1), `wind_direction` (F32, degree, the direction the wind blows
//! FROM), `temperature` (F32, K, the target's median brightness
//! temperature), `time` (F64, seconds since 2000-01-01 12:00:00, the
//! mid-point of the image pair), `local_zenith_angle` (F32, degree), `DQF`
//! (I8, 0 = good).  Fill is -999 on the floats and -128 on the DQF.
//!
//! The DQF is enumerated: 0 is the only good value (1 = bad track, 2 =
//! bad height assignment, 3 = invalid, per the product's flag meanings),
//! so the gate is the literal fail-closed rule and everything not zero is
//! counted and dropped.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, TimeZone, Utc};

use crate::goes::{parse_goes_abi_filename, GoesAbiFilename};
use crate::netcdf::open_goes_netcdf_lossy;

/// J2000: the epoch the ABI products count seconds from.
pub fn j2000_epoch() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2000, 1, 1, 12, 0, 0).unwrap()
}

/// Seconds since J2000 to an instant.  The products count SI seconds
/// without leap seconds, so the conversion is a plain offset; the five
/// leap seconds since 2000 are inside the one-second rounding of an image
/// mid-point and are not corrected for (stated, not hidden).
pub fn j2000_to_utc(seconds: f64) -> Option<DateTime<Utc>> {
    if !seconds.is_finite() || seconds < 0.0 || seconds > 4.0e9 {
        return None;
    }
    let whole = seconds.floor() as i64;
    let nanos = ((seconds - whole as f64) * 1.0e9) as i64;
    Some(j2000_epoch() + Duration::seconds(whole) + Duration::nanoseconds(nanos))
}

/// One good-quality vector.
#[derive(Debug, Clone, PartialEq)]
pub struct DmwVector {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub pressure_hpa: f64,
    pub speed_m_s: f64,
    pub direction_deg: f64,
    pub temperature_k: Option<f64>,
    pub local_zenith_angle_deg: f64,
    pub time: DateTime<Utc>,
}

/// What the gate did, vector by vector.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DmwCounts {
    /// The file carries no nMeasures dimension: a product with no vectors.
    pub empty_granule: bool,
    pub vectors_in_file: usize,
    pub dqf_good: usize,
    pub dqf_by_value: BTreeMap<i64, usize>,
    pub dqf_fill: usize,
    pub position_fill: usize,
    pub pressure_fill: usize,
    pub wind_fill: usize,
    pub time_fill: usize,
    pub kept: usize,
}

#[derive(Debug, Clone)]
pub struct DmwGranule {
    pub path: PathBuf,
    pub filename: GoesAbiFilename,
    pub band: u8,
    pub band_wavelength_um: Option<f64>,
    pub vectors: Vec<DmwVector>,
    pub counts: DmwCounts,
}

fn boxed(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()))
}

fn read_1d(file: &netcrust::File, name: &str, n: usize, subject: &str) -> Result<(Vec<f64>, f64), Box<dyn Error>> {
    let Some(variable) = file.variable(name) else {
        return Err(boxed(format!(
            "{subject}: variable {name:?} is missing; the DMW product table names it (lat, lon, \
             pressure, wind_speed, wind_direction, time, local_zenith_angle, DQF) and a file \
             without it is not a DMW granule this reader knows"
        )));
    };
    let fill = variable
        .attribute("_FillValue")
        .and_then(|a| a.as_f64())
        .unwrap_or(f64::NAN);
    let values = variable.array_f64()?.into_values();
    if values.len() != n {
        return Err(boxed(format!(
            "{subject}: {name} has {} values where nMeasures is {n}",
            values.len()
        )));
    }
    Ok((values, fill))
}

fn is_fill(value: f64, fill: f64) -> bool {
    !value.is_finite() || (fill.is_finite() && (value - fill).abs() < 1e-6)
}

/// Decode one DMW granule: every vector whose DQF is 0 and whose position,
/// pressure, wind and time are not fill.
pub fn read_dmw_granule(path: impl AsRef<Path>) -> Result<DmwGranule, Box<dyn Error>> {
    let path = path.as_ref();
    let subject = path.display().to_string();
    let filename = parse_goes_abi_filename(path)?;
    if !filename.product.to_ascii_uppercase().starts_with("ABI-L2-DMW") {
        return Err(boxed(format!(
            "{subject}: product {} is not a derived-motion-winds granule (ABI-L2-DMW*)",
            filename.product
        )));
    }
    let band = filename.channel.ok_or_else(|| {
        boxed(format!("{subject}: the filename carries no band token (C02..C16)"))
    })?;
    let file = open_goes_netcdf_lossy(path)?;
    // A granule with no vectors (the visible band at night, a band whose
    // tracker found nothing) is written WITHOUT the nMeasures dimension;
    // measured 2026-09-06 on 19 of 31 hours of the case window (C02 and
    // C07 granules).  It is an empty product, not a broken one.
    let Some(n) = file.dimension("nMeasures").map(|d| d.len()) else {
        return Ok(DmwGranule {
            path: path.to_path_buf(),
            filename,
            band,
            band_wavelength_um: None,
            vectors: Vec::new(),
            counts: DmwCounts {
                empty_granule: true,
                ..Default::default()
            },
        });
    };
    let (lat, lat_fill) = read_1d(&file, "lat", n, &subject)?;
    let (lon, lon_fill) = read_1d(&file, "lon", n, &subject)?;
    let (pressure, p_fill) = read_1d(&file, "pressure", n, &subject)?;
    let (speed, s_fill) = read_1d(&file, "wind_speed", n, &subject)?;
    let (direction, d_fill) = read_1d(&file, "wind_direction", n, &subject)?;
    let (time, t_fill) = read_1d(&file, "time", n, &subject)?;
    let (lza, lza_fill) = read_1d(&file, "local_zenith_angle", n, &subject)?;
    let (dqf, dqf_fill) = read_1d(&file, "DQF", n, &subject)?;
    let temperature = file
        .variable("temperature")
        .and_then(|v| {
            let fill = v.attribute("_FillValue").and_then(|a| a.as_f64()).unwrap_or(f64::NAN);
            v.array_f64().ok().map(|a| (a.into_values(), fill))
        })
        .filter(|(values, _)| values.len() == n);
    let band_wavelength_um = file
        .variable("band_wavelength")
        .and_then(|v| v.array_f64().ok())
        .and_then(|a| a.into_values().first().copied())
        .filter(|v| v.is_finite());

    let mut counts = DmwCounts {
        vectors_in_file: n,
        ..Default::default()
    };
    let mut vectors = Vec::with_capacity(n);
    for i in 0..n {
        if is_fill(dqf[i], dqf_fill) {
            counts.dqf_fill += 1;
            continue;
        }
        let flag = dqf[i].round() as i64;
        *counts.dqf_by_value.entry(flag).or_insert(0) += 1;
        if flag != 0 {
            continue;
        }
        counts.dqf_good += 1;
        if is_fill(lat[i], lat_fill) || is_fill(lon[i], lon_fill) || lat[i].abs() > 90.0 {
            counts.position_fill += 1;
            continue;
        }
        if is_fill(pressure[i], p_fill) || pressure[i] <= 0.0 {
            counts.pressure_fill += 1;
            continue;
        }
        if is_fill(speed[i], s_fill) || is_fill(direction[i], d_fill) || speed[i] < 0.0 {
            counts.wind_fill += 1;
            continue;
        }
        let Some(instant) = (!is_fill(time[i], t_fill)).then(|| j2000_to_utc(time[i])).flatten() else {
            counts.time_fill += 1;
            continue;
        };
        let temperature_k = temperature.as_ref().and_then(|(values, fill)| {
            let v = values[i];
            (!is_fill(v, *fill)).then_some(v)
        });
        vectors.push(DmwVector {
            latitude_deg: lat[i],
            longitude_deg: lon[i],
            pressure_hpa: pressure[i],
            speed_m_s: speed[i],
            direction_deg: direction[i],
            temperature_k,
            local_zenith_angle_deg: if is_fill(lza[i], lza_fill) { f64::NAN } else { lza[i] },
            time: instant,
        });
    }
    counts.kept = vectors.len();
    Ok(DmwGranule {
        path: path.to_path_buf(),
        filename,
        band,
        band_wavelength_um,
        vectors,
        counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn j2000_seconds_land_on_the_scan_instant() {
        // 2026-09-01 18:05:00 UTC is 9,740 days and 6 h 5 min after
        // 2000-01-01 12:00:00: 841,557,900 s.
        let t = j2000_to_utc(841_557_900.0).unwrap();
        assert_eq!(t, Utc.with_ymd_and_hms(2026, 9, 1, 18, 5, 0).unwrap());
        assert!(j2000_to_utc(-999.0).is_none());
        assert!(j2000_to_utc(f64::NAN).is_none());
    }
}
