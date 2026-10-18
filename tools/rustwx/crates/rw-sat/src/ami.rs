//! KMA GK-2A AMI Level 1B (NetCDF-4, one channel per file) decode.
//!
//! Everything is read from the file itself: the count-to-radiance gain and
//! offset, the effective-to-brightness quadratic, the physical constants
//! and the CGMS navigation (`cfac`, `lfac`, `coff`, `loff`, sub-satellite
//! longitude in radians, satellite distance from the Earth centre).
//!
//! The pixel word is 16 bits: the top 2 are the quality flag (0 good,
//! 1 conditionally usable, 2 outside the scan area, 3 error, the file's own
//! `data_quality_flag_meaning`) and the low `number_of_valid_bits_per_pixel`
//! are the count.  Only flag 0 carries a brightness temperature, the same
//! fail-closed rule as the ABI L1b DQF.

use std::error::Error;
use std::path::Path;

use chrono::{DateTime, TimeZone, Utc};

use crate::abi::{AbiFixedGrid, GoesImagerProjection};
use crate::geo_imager::{GeoImagerScan, bt_from_radiance_per_wavenumber};
use crate::geostationary::SweepAngleAxis;
use crate::netcdf::open_goes_netcdf_lossy;

/// AMI channel name -> band number (wavelength order, 1..16).
pub const AMI_CHANNELS: [(&str, u8); 16] = [
    ("VI004", 1),
    ("VI005", 2),
    ("VI006", 3),
    ("VI008", 4),
    ("NR013", 5),
    ("NR016", 6),
    ("SW038", 7),
    ("WV063", 8),
    ("WV069", 9),
    ("WV073", 10),
    ("IR087", 11),
    ("IR096", 12),
    ("IR105", 13),
    ("IR112", 14),
    ("IR123", 15),
    ("IR133", 16),
];

/// Seconds from the Unix epoch to the AMI time origin 2000-01-01T12:00:00Z
/// (`observation_start_time` 845294432.84 is 2026-10-15T00:00:32.8Z, the
/// file's own `scene_acquisition_time` 20261015_000032).
pub const AMI_EPOCH_UNIX_S: f64 = 946_728_000.0;

fn attr_f64(file: &netcrust::File, name: &str) -> Result<f64, Box<dyn Error>> {
    file.attribute(name)
        .and_then(|a| a.as_f64())
        .ok_or_else(|| format!("GK-2A L1B global attribute {name} is missing or not numeric").into())
}

fn attr_str(file: &netcrust::File, name: &str) -> Result<String, Box<dyn Error>> {
    file.attribute(name)
        .and_then(|a| a.as_string().map(|s| s.trim().to_string()))
        .ok_or_else(|| format!("GK-2A L1B global attribute {name} is missing").into())
}

/// Navigation of one AMI file without reading the pixels.
pub struct AmiNavigation {
    pub projection: GoesImagerProjection,
    pub grid: AbiFixedGrid,
}

pub fn read_ami_navigation(path: &Path) -> Result<AmiNavigation, Box<dyn Error>> {
    let file = open_goes_netcdf_lossy(path)?;
    navigation(&file)
}

fn navigation(file: &netcrust::File) -> Result<AmiNavigation, Box<dyn Error>> {
    let projection_type = attr_str(file, "projection_type")?;
    if projection_type != "GEOS" {
        return Err(format!("GK-2A projection_type {projection_type} is not GEOS").into());
    }
    let ncols = attr_f64(file, "number_of_columns")? as usize;
    let nlines = attr_f64(file, "number_of_lines")? as usize;
    let (cfac, lfac) = (attr_f64(file, "cfac")?, attr_f64(file, "lfac")?);
    let (coff, loff) = (attr_f64(file, "coff")?, attr_f64(file, "loff")?);
    let a = attr_f64(file, "earth_equatorial_radius")?;
    let b = attr_f64(file, "earth_polar_radius")?;
    let distance = attr_f64(file, "nominal_satellite_height")?;
    let sub_lon_rad = attr_f64(file, "sub_longitude")?;
    // Column c and line l are 1-based (coff = loff = 2750.5 centres the 5500
    // pixel grid); x = (c - coff) 2^16 / cfac degrees, east-positive; lfac is
    // negative in the file, so (l - loff) 2^16 / lfac is already north-positive.
    // The file's image_upperleft_x = -0.153972 rad is column 1's angle.
    let x_scan_rad = (0..ncols)
        .map(|c| ((c as f64 + 1.0 - coff) * 65536.0 / cfac).to_radians())
        .collect();
    let y_scan_rad = (0..nlines)
        .map(|l| ((l as f64 + 1.0 - loff) * 65536.0 / lfac).to_radians())
        .collect();
    Ok(AmiNavigation {
        projection: GoesImagerProjection {
            perspective_point_height_m: distance - a,
            semi_major_axis_m: a,
            semi_minor_axis_m: b,
            longitude_of_projection_origin_deg: sub_lon_rad.to_degrees(),
            sweep_angle_axis: SweepAngleAxis::Y,
        },
        grid: AbiFixedGrid {
            nx: ncols,
            ny: nlines,
            x_scan_rad,
            y_scan_rad,
        },
    })
}

/// Decode one AMI L1B channel file, optionally only `[x_start, x_count,
/// y_start, y_count]` of it.
///
/// `centroid_wavenumber_cm1` is the channel's spectral-response centroid
/// (the source table's `band_centroid_wavenumber_cm1`).  It is required: the
/// file's `channel_center_wavelength` is rounded to 0.1 um (`10.5` for a
/// 10.350 um centroid, `6.3` for 6.182 um), while KMA's `Teff_to_Tbb`
/// coefficients belong to the centroid.  Measured on the 2026-10-15 00Z full
/// disk: the file's formula at the rounded wavelength is off by +3.13 K
/// (WV063), -0.85 K (WV069), -0.53 K (WV073) and +1.57 K (IR105) against
/// the same formula at the centroid, which in turn matches CRTM's
/// band-corrected Planck relation for ami_gk2 to 0.004 K.
pub fn read_ami_l1b(
    path: &Path,
    window: Option<[usize; 4]>,
    centroid_wavenumber_cm1: Option<f64>,
) -> Result<GeoImagerScan, Box<dyn Error>> {
    let file = open_goes_netcdf_lossy(path)?;
    let nav = navigation(&file)?;
    let satellite_name = attr_str(&file, "satellite_name")?;
    let instrument = attr_str(&file, "instrument_name")?;
    if instrument != "AMI" {
        return Err(format!("{}: instrument {instrument}, not AMI", path.display()).into());
    }
    let variable = file
        .variable("image_pixel_values")
        .ok_or_else(|| format!("{}: no image_pixel_values", path.display()))?;
    let channel = variable
        .attribute("channel_name")
        .and_then(|a| a.as_string().map(|s| s.trim().to_ascii_uppercase()))
        .ok_or("image_pixel_values has no channel_name")?;
    let band = AMI_CHANNELS
        .iter()
        .find(|(n, _)| *n == channel)
        .map(|(_, b)| *b)
        .ok_or_else(|| format!("unknown AMI channel {channel}"))?;
    if band < 7 {
        return Err(format!("AMI {channel} is a visible/near-infrared channel; only the infrared channels are decoded").into());
    }
    let total_bits = variable
        .attribute("number_of_total_bits_per_pixel")
        .and_then(|a| a.as_f64())
        .unwrap_or(16.0) as u32;
    let flag_bits = variable
        .attribute("number_of_data_quality_flag_bits_per_pixel")
        .and_then(|a| a.as_f64())
        .unwrap_or(2.0) as u32;
    let valid_bits = variable
        .attribute("number_of_valid_bits_per_pixel")
        .and_then(|a| a.as_f64())
        .ok_or("image_pixel_values has no number_of_valid_bits_per_pixel")? as u32;
    if total_bits != 16 || flag_bits + valid_bits > 16 {
        return Err(format!("unexpected AMI pixel layout: {total_bits} bits, {flag_bits} flag, {valid_bits} valid").into());
    }
    let gain = attr_f64(&file, "DN_to_Radiance_Gain")?;
    let offset = attr_f64(&file, "DN_to_Radiance_Offset")?;
    let c = [
        attr_f64(&file, "Teff_to_Tbb_c0")?,
        attr_f64(&file, "Teff_to_Tbb_c1")?,
        attr_f64(&file, "Teff_to_Tbb_c2")?,
    ];
    let light_c = attr_f64(&file, "light_speed")?;
    let planck_h = attr_f64(&file, "Plank_constant_h")?;
    let boltzmann_k = attr_f64(&file, "Boltzmann_constant_k")?;
    let stated_wavelength_um: f64 = attr_str(&file, "channel_center_wavelength")?
        .parse()
        .map_err(|e| format!("channel_center_wavelength: {e}"))?;
    let centroid = centroid_wavenumber_cm1.ok_or_else(|| {
        format!(
            "AMI {channel}: no spectral-response centroid for this channel; the file's rounded \
             channel_center_wavelength ({stated_wavelength_um} um) puts brightness temperature off by up to 3 K \
             (measured), so the source table's band_centroid_wavenumber_cm1 must name it"
        )
    })?;
    let wavelength_um = 1.0e4 / centroid;
    let t0 = AMI_EPOCH_UNIX_S + attr_f64(&file, "observation_start_time")?;
    let t1 = AMI_EPOCH_UNIX_S + attr_f64(&file, "observation_end_time")?;

    let (full_nx, full_ny) = (nav.grid.nx, nav.grid.ny);
    let [xs, xc, ys, yc] = window.unwrap_or([0, full_nx, 0, full_ny]);
    if xc == 0 || yc == 0 || xs + xc > full_nx || ys + yc > full_ny {
        return Err(format!("window {:?} is not inside the {full_nx}x{full_ny} grid", [xs, xc, ys, yc]).into());
    }
    let raw: Vec<f64> = if window.is_some() {
        let sel = netcrust::NcSliceInfo {
            selections: vec![
                netcrust::NcSliceInfoElem::Slice { start: ys as u64, end: (ys + yc) as u64, step: 1 },
                netcrust::NcSliceInfoElem::Slice { start: xs as u64, end: (xs + xc) as u64, step: 1 },
            ],
        };
        variable.array_f64_slice(&sel)?.into_values()
    } else {
        variable.array_f64()?.into_values()
    };
    if raw.len() != xc * yc {
        return Err(format!("read {} pixels for a {xc}x{yc} window", raw.len()).into());
    }
    let count_mask: u32 = (1u32 << valid_bits) - 1;
    let mut radiance = Vec::with_capacity(raw.len());
    let mut dqf = Vec::with_capacity(raw.len());
    let mut bt = Vec::with_capacity(raw.len());
    for &w in &raw {
        let word = w as u32;
        let flag = word >> (16 - flag_bits);
        let count = word & count_mask;
        dqf.push(flag as f32);
        if flag != 0 {
            radiance.push(f32::NAN);
            bt.push(f32::NAN);
            continue;
        }
        let rad = gain * f64::from(count) + offset;
        radiance.push(rad as f32);
        bt.push(bt_from_radiance_per_wavenumber(rad, wavelength_um, c, planck_h, light_c, boltzmann_k) as f32);
    }
    let row_time_unix_s = (ys..ys + yc)
        .map(|r| t0 + (t1 - t0) * (r as f64 + 0.5) / full_ny as f64)
        .collect();
    let satellite = satellite_name.to_ascii_uppercase().replace(['-', ' '], "");
    let to_dt = |s: f64| -> Result<DateTime<Utc>, Box<dyn Error>> {
        Utc.timestamp_millis_opt((s * 1000.0).round() as i64)
            .single()
            .ok_or_else(|| format!("bad AMI time {s}").into())
    };
    Ok(GeoImagerScan {
        instrument: "ami".to_string(),
        satellite,
        sector: match attr_str(&file, "observation_mode")?.as_str() {
            "FD" => "F".to_string(),
            other => other.to_string(),
        },
        band,
        channel_name: channel,
        central_wavelength_um: wavelength_um,
        start_time_utc: to_dt(t0)?,
        end_time_utc: to_dt(t1)?,
        projection: nav.projection,
        fixed_grid: AbiFixedGrid {
            nx: xc,
            ny: yc,
            x_scan_rad: nav.grid.x_scan_rad[xs..xs + xc].to_vec(),
            y_scan_rad: nav.grid.y_scan_rad[ys..ys + yc].to_vec(),
        },
        row_time_unix_s,
        radiance,
        radiance_units: "mW m-2 sr-1 (cm-1)-1".to_string(),
        dqf,
        bt,
        calibration: serde_json::json!({
            "source": "GK-2A AMI L1B global attributes",
            "formula": "flag = word >> 14 (0 good, 1 conditionally usable, 2 outside scan, 3 error; only 0 is kept); count = word & (2^valid_bits - 1); L[mW m-2 sr-1 (cm-1)-1] = DN_to_Radiance_Gain * count + DN_to_Radiance_Offset; nu = the channel's spectral-response centroid (source table, not the file's rounded channel_center_wavelength); Te = (h c nu / k) / ln(2 h c^2 nu^3 / L + 1) (SI units); Tb = Teff_to_Tbb_c0 + c1 Te + c2 Te^2",
            "centroid_wavenumber_cm1": centroid,
            "file_channel_center_wavelength_um": stated_wavelength_um,
            "number_of_valid_bits_per_pixel": valid_bits,
            "DN_to_Radiance_Gain": gain,
            "DN_to_Radiance_Offset": offset,
            "Teff_to_Tbb_c0_c1_c2": c,
            "light_speed": light_c,
            "Plank_constant_h": planck_h,
            "Boltzmann_constant_k": boltzmann_k,
            "calibration_table_version": attr_str(&file, "calibration_table_version").ok(),
            "navigation": {"sweep_angle_axis": "y"},
        }),
        window: window.map(|_| [xs, xc, ys, yc]),
        full_nx,
        full_ny,
        sources: vec![path.to_path_buf()],
    })
}
