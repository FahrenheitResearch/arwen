//! JMA Himawari-8/9 AHI Himawari Standard Data (HSD) decode.
//!
//! Vendored from the project's rusty-weather workspace (`crates/rw-sat/src/
//! himawari.rs` @ 369d65c, MIT): the segment-name parser and the HSD header
//! parser (blocks #1, #2, #3, #5, #7) are that file's code.  Changes made
//! here, each for a measured reason (recorded in VENDOR.md):
//!
//! * **Counts are not shifted.**  Upstream shifts the raw word right by
//!   `bits_per_pixel - valid_bits_per_pixel` before calibrating.  The HSD
//!   count is the raw word itself: on 2026-10-15 00Z band 13 segment 5 the
//!   counts span 1381..4052 with 12 valid bits, which calibrate to a median
//!   290 K unshifted and to 326-329 K everywhere shifted.
//! * **Navigation sweeps about y** (CGMS / HSD), not x; see
//!   `geostationary::scan_angles_to_lat_lon_sweep_y`.
//! * Segments may be read straight from the bucket's `.DAT.bz2`; the output
//!   is a [`GeoImagerScan`] (radiance, quality flag, brightness temperature,
//!   row times) rather than an rw-store field; the S3 discovery, manifest and
//!   staging code is replaced by the shared source table (`geo_imager`).

use std::error::Error;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use bzip2::read::MultiBzDecoder;
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use serde::{Deserialize, Serialize};

use crate::abi::{AbiFixedGrid, GoesImagerProjection};
use crate::geo_imager::{DQF_ERROR, DQF_GOOD, DQF_OUTSIDE, GeoImagerScan, bt_from_radiance_per_um};
use crate::geostationary::SweepAngleAxis;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HimawariByteOrder {
    Little,
    Big,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HimawariHsdHeader {
    pub path: Option<String>,
    pub byte_order: HimawariByteOrder,
    pub total_header_blocks: u16,
    pub satellite_name: String,
    pub processing_center: String,
    pub observation_area: String,
    pub observation_timeline: u16,
    pub observation_start_mjd: f64,
    pub observation_end_mjd: f64,
    pub file_creation_mjd: f64,
    pub total_header_length: u32,
    pub total_data_length: u32,
    pub file_name: String,
    pub data: HimawariDataInfo,
    pub projection: Option<HimawariProjectionInfo>,
    pub calibration: Option<HimawariCalibrationInfo>,
    pub segment: Option<HimawariSegmentInfo>,
    pub file_length: u64,
    pub expected_file_length: u64,
    pub length_matches_header: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HimawariDataInfo {
    pub bits_per_pixel: u16,
    pub columns: u16,
    pub lines: u16,
    pub compression_flag: u8,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HimawariProjectionInfo {
    pub sub_lon_degrees: f64,
    pub cfac: u32,
    pub lfac: u32,
    pub coff: f32,
    pub loff: f32,
    pub satellite_distance_km: f64,
    pub equatorial_radius_km: f64,
    pub polar_radius_km: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HimawariCalibrationInfo {
    pub band_number: u16,
    pub central_wavelength_um: f64,
    pub valid_bits_per_pixel: u16,
    pub error_pixel_count: u16,
    pub outside_scan_count: u16,
    pub count_to_radiance_slope: f64,
    pub count_to_radiance_intercept: f64,
    pub planck_or_albedo_coefficients: [f64; 3],
    pub inverse_planck_coefficients: Option<[f64; 3]>,
    pub physical_constants: Option<HimawariPhysicalConstants>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HimawariPhysicalConstants {
    pub speed_of_light_m_s: f64,
    pub planck_constant_j_s: f64,
    pub boltzmann_constant_j_k: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HimawariSegmentInfo {
    pub total_segments: u8,
    pub sequence_number: u8,
    pub first_line_number: u16,
}

/// Read one segment file, decompressing a `.bz2` in memory.
pub fn read_segment_bytes(path: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    let raw = fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if raw.starts_with(b"BZh") {
        let mut out = Vec::with_capacity(raw.len() * 4);
        MultiBzDecoder::new(raw.as_slice())
            .read_to_end(&mut out)
            .map_err(|e| format!("{}: bzip2: {e}", path.display()))?;
        Ok(out)
    } else {
        Ok(raw)
    }
}

pub fn parse_hsd_header(bytes: &[u8]) -> Result<HimawariHsdHeader, Box<dyn Error>> {
    require_len(bytes, 282, "HSD basic information block")?;
    if bytes[0] != 1 {
        return Err(format!("expected HSD block #1 at offset 0, got {}", bytes[0]).into());
    }
    let byte_order = match bytes[5] {
        0 => HimawariByteOrder::Little,
        1 => HimawariByteOrder::Big,
        value => return Err(format!("unsupported HSD byte order flag {value}").into()),
    };
    let block1_len = read_u16(bytes, 1, byte_order)?;
    if block1_len < 82 {
        return Err(format!("HSD block #1 is too short: {block1_len}").into());
    }
    let total_header_blocks = read_u16(bytes, 3, byte_order)?;
    let total_header_length = read_u32(bytes, 70, byte_order)?;
    let total_data_length = read_u32(bytes, 74, byte_order)?;
    require_len(bytes, total_header_length as usize, "HSD declared header length")?;

    let mut offset = 0_usize;
    let mut data_info = None;
    let mut projection = None;
    let mut calibration = None;
    let mut segment = None;
    for _ in 0..total_header_blocks {
        require_len(bytes, offset + 3, "HSD block prefix")?;
        let block_number = bytes[offset];
        let length = if block_number == 10 {
            require_len(bytes, offset + 5, "HSD block #10 prefix")?;
            read_u32(bytes, offset + 1, byte_order)?
        } else {
            u32::from(read_u16(bytes, offset + 1, byte_order)?)
        };
        if length == 0 {
            return Err(format!("HSD block #{block_number} at offset {offset} has zero length").into());
        }
        let block_end = offset.saturating_add(length as usize);
        require_len(bytes, block_end, "HSD block body")?;
        let block = &bytes[offset..block_end];
        match block_number {
            2 => data_info = Some(parse_data_info(block, byte_order)?),
            3 => projection = Some(parse_projection_info(block, byte_order)?),
            5 => calibration = Some(parse_calibration_info(block, byte_order)?),
            7 => segment = Some(parse_segment_info(block, byte_order)?),
            _ => {}
        }
        offset = block_end;
    }
    if offset != total_header_length as usize {
        return Err(format!(
            "HSD header block lengths sum to {offset}, declared header length is {total_header_length}"
        )
        .into());
    }
    let data = data_info.ok_or("HSD header is missing block #2 data information")?;
    let expected_file_length = u64::from(total_header_length) + u64::from(total_data_length);
    Ok(HimawariHsdHeader {
        path: None,
        byte_order,
        total_header_blocks,
        satellite_name: read_ascii(bytes, 6, 16)?,
        processing_center: read_ascii(bytes, 22, 16)?,
        observation_area: read_ascii(bytes, 38, 4)?,
        observation_timeline: read_u16(bytes, 44, byte_order)?,
        observation_start_mjd: read_f64(bytes, 46, byte_order)?,
        observation_end_mjd: read_f64(bytes, 54, byte_order)?,
        file_creation_mjd: read_f64(bytes, 62, byte_order)?,
        total_header_length,
        total_data_length,
        file_name: read_ascii(bytes, 114, 128)?,
        data,
        projection,
        calibration,
        segment,
        file_length: bytes.len() as u64,
        expected_file_length,
        length_matches_header: bytes.len() as u64 == expected_file_length,
    })
}

struct DecodedSegment {
    header: HimawariHsdHeader,
    counts: Vec<u16>,
    path: PathBuf,
}

fn decode_segment(path: &Path) -> Result<DecodedSegment, Box<dyn Error>> {
    let bytes = read_segment_bytes(path)?;
    let mut header = parse_hsd_header(&bytes)?;
    header.path = Some(path.display().to_string());
    if !header.length_matches_header {
        return Err(format!(
            "{}: HSD length {} does not match the declared {}",
            path.display(),
            header.file_length,
            header.expected_file_length
        )
        .into());
    }
    if header.data.compression_flag != 0 {
        return Err(format!("{}: compressed HSD data blocks are not decoded", path.display()).into());
    }
    let pixels = usize::from(header.data.columns) * usize::from(header.data.lines);
    if header.total_data_length as usize != pixels * 2 {
        return Err(format!(
            "{}: HSD data length {} is not {}x{}x2",
            path.display(),
            header.total_data_length,
            header.data.columns,
            header.data.lines
        )
        .into());
    }
    let start = header.total_header_length as usize;
    let counts = bytes[start..start + pixels * 2]
        .chunks_exact(2)
        .map(|w| match header.byte_order {
            HimawariByteOrder::Little => u16::from_le_bytes([w[0], w[1]]),
            HimawariByteOrder::Big => u16::from_be_bytes([w[0], w[1]]),
        })
        .collect();
    Ok(DecodedSegment {
        header,
        counts,
        path: path.to_path_buf(),
    })
}

/// Decode the segments of one band of one scan (any contiguous run, in any
/// order) into a [`GeoImagerScan`] on the full-disk fixed grid rows they
/// cover.
pub fn read_ahi_segments(paths: &[PathBuf]) -> Result<GeoImagerScan, Box<dyn Error>> {
    if paths.is_empty() {
        return Err("no Himawari HSD segments given".into());
    }
    // Segments decompress and decode independently: one rayon task each.
    use rayon::prelude::*;
    let mut segs = paths
        .par_iter()
        .map(|p| decode_segment(p).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, String>>()?;
    segs.sort_by_key(|s| s.header.segment.as_ref().map(|i| i.sequence_number).unwrap_or(0));
    let first = &segs[0].header;
    let proj = first.projection.clone().ok_or("HSD header is missing projection block #3")?;
    let cal = first.calibration.clone().ok_or("HSD header is missing calibration block #5")?;
    let consts = cal
        .physical_constants
        .clone()
        .ok_or("HSD band has no infrared Planck constants (visible and near-infrared bands are not decoded here)")?;
    let first_info = first.segment.clone().ok_or("HSD header is missing segment block #7")?;
    let nx = usize::from(first.data.columns);
    let total_lines = usize::from(first.data.lines) * usize::from(first_info.total_segments);
    let mut expected_line = first_info.first_line_number;
    let mut counts: Vec<u16> = Vec::new();
    let mut row_times: Vec<f64> = Vec::new();
    for s in &segs {
        let h = &s.header;
        let c = h.calibration.as_ref().ok_or("HSD header is missing calibration block #5")?;
        let info = h.segment.as_ref().ok_or("HSD header is missing segment block #7")?;
        if h.satellite_name != first.satellite_name
            || h.observation_area != first.observation_area
            || h.data.columns != first.data.columns
            || c.band_number != cal.band_number
            || info.total_segments != first_info.total_segments
            || (h.observation_start_mjd - first.observation_start_mjd).abs() > 600.0 / 86400.0
        {
            return Err(format!("{} is not a segment of the same band and scan", s.path.display()).into());
        }
        if c != &cal {
            return Err(format!("{}: calibration differs between segments of one scan", s.path.display()).into());
        }
        if info.first_line_number != expected_line {
            return Err(format!(
                "segments are not contiguous: expected first line {expected_line}, {} starts at {}",
                s.path.display(),
                info.first_line_number
            )
            .into());
        }
        expected_line = expected_line.saturating_add(h.data.lines);
        counts.extend_from_slice(&s.counts);
        let (t0, t1) = (mjd_to_unix(h.observation_start_mjd), mjd_to_unix(h.observation_end_mjd));
        let n = usize::from(h.data.lines);
        for r in 0..n {
            row_times.push(t0 + (t1 - t0) * (r as f64 + 0.5) / n as f64);
        }
    }
    let ny = counts.len() / nx;
    let mut radiance = Vec::with_capacity(counts.len());
    let mut dqf = Vec::with_capacity(counts.len());
    let mut bt = Vec::with_capacity(counts.len());
    let coeffs = cal.planck_or_albedo_coefficients;
    for &raw in &counts {
        if raw == cal.outside_scan_count {
            radiance.push(f32::NAN);
            dqf.push(DQF_OUTSIDE);
            bt.push(f32::NAN);
            continue;
        }
        if raw == cal.error_pixel_count {
            radiance.push(f32::NAN);
            dqf.push(DQF_ERROR);
            bt.push(f32::NAN);
            continue;
        }
        let rad = cal.count_to_radiance_slope * f64::from(raw) + cal.count_to_radiance_intercept;
        let t = bt_from_radiance_per_um(
            rad,
            cal.central_wavelength_um,
            coeffs,
            consts.planck_constant_j_s,
            consts.speed_of_light_m_s,
            consts.boltzmann_constant_j_k,
        );
        radiance.push(rad as f32);
        dqf.push(DQF_GOOD);
        bt.push(t as f32);
    }
    // Column c and line l are 1-based; the fixed-grid angles are east- and north-positive.
    let x_scan_rad = (0..nx)
        .map(|c| ((c as f64 + 1.0 - f64::from(proj.coff)) * 65536.0 / f64::from(proj.cfac)).to_radians())
        .collect();
    let y_scan_rad = (0..ny)
        .map(|r| {
            let line = f64::from(first_info.first_line_number) + r as f64;
            ((f64::from(proj.loff) - line) * 65536.0 / f64::from(proj.lfac)).to_radians()
        })
        .collect();
    let last = &segs[segs.len() - 1].header;
    let satellite = match first.satellite_name.trim() {
        "Himawari-8" => "H08".to_string(),
        "Himawari-9" => "H09".to_string(),
        other => other.to_ascii_uppercase().replace([' ', '-'], ""),
    };
    let full_disk = first_info.first_line_number == 1 && ny == total_lines;
    let window = if full_disk {
        None
    } else {
        Some([0, nx, usize::from(first_info.first_line_number) - 1, ny])
    };
    Ok(GeoImagerScan {
        instrument: "ahi".to_string(),
        satellite,
        sector: if first.observation_area == "FLDK" { "F".to_string() } else { first.observation_area.clone() },
        band: u8::try_from(cal.band_number).map_err(|_| format!("band {}", cal.band_number))?,
        channel_name: format!("B{:02}", cal.band_number),
        central_wavelength_um: cal.central_wavelength_um,
        start_time_utc: mjd_to_datetime(first.observation_start_mjd)?,
        end_time_utc: mjd_to_datetime(last.observation_end_mjd)?,
        projection: GoesImagerProjection {
            perspective_point_height_m: (proj.satellite_distance_km - proj.equatorial_radius_km) * 1000.0,
            semi_major_axis_m: proj.equatorial_radius_km * 1000.0,
            semi_minor_axis_m: proj.polar_radius_km * 1000.0,
            longitude_of_projection_origin_deg: proj.sub_lon_degrees,
            sweep_angle_axis: SweepAngleAxis::Y,
        },
        fixed_grid: AbiFixedGrid {
            nx,
            ny,
            x_scan_rad,
            y_scan_rad,
        },
        row_time_unix_s: row_times,
        radiance,
        radiance_units: "W m-2 sr-1 um-1".to_string(),
        dqf,
        bt,
        calibration: serde_json::json!({
            "source": "HSD block #5 (calibration information) of the first segment; every segment's block #5 is identical",
            "formula": "L[W m-2 sr-1 um-1] = slope * count + intercept (count = the raw 16-bit word); Te = (h c / k) / (lambda ln(2 h c^2 / (L lambda^5) + 1)); Tb = c0 + c1 Te + c2 Te^2; count = error or outside-scan value is no observation",
            "central_wavelength_um": cal.central_wavelength_um,
            "valid_bits_per_pixel": cal.valid_bits_per_pixel,
            "count_to_radiance_slope": cal.count_to_radiance_slope,
            "count_to_radiance_intercept": cal.count_to_radiance_intercept,
            "te_to_tb_c0_c1_c2": coeffs,
            "error_pixel_count": cal.error_pixel_count,
            "outside_scan_count": cal.outside_scan_count,
            "speed_of_light_m_s": consts.speed_of_light_m_s,
            "planck_constant_j_s": consts.planck_constant_j_s,
            "boltzmann_constant_j_k": consts.boltzmann_constant_j_k,
            "navigation": {"cfac": proj.cfac, "lfac": proj.lfac, "coff": proj.coff, "loff": proj.loff,
                           "satellite_distance_km": proj.satellite_distance_km, "sweep_angle_axis": "y"},
        }),
        window,
        full_nx: nx,
        full_ny: total_lines,
        sources: segs.iter().map(|s| s.path.clone()).collect(),
    })
}

fn mjd_to_unix(mjd: f64) -> f64 {
    (mjd - 40587.0) * 86400.0
}

fn mjd_to_datetime(mjd: f64) -> Result<DateTime<Utc>, Box<dyn Error>> {
    if !mjd.is_finite() {
        return Err(format!("invalid MJD {mjd}").into());
    }
    let epoch = NaiveDate::from_ymd_opt(1858, 11, 17)
        .and_then(|d| d.and_hms_opt(0, 0, 0))
        .ok_or("failed to construct MJD epoch")?;
    let millis = (mjd * 86_400_000.0).round() as i64;
    let naive = epoch
        .checked_add_signed(chrono::Duration::milliseconds(millis))
        .ok_or_else(|| format!("MJD {mjd} is out of range"))?;
    Ok(Utc.from_utc_datetime(&naive))
}

fn parse_data_info(block: &[u8], o: HimawariByteOrder) -> Result<HimawariDataInfo, Box<dyn Error>> {
    require_len(block, 10, "HSD data information block")?;
    Ok(HimawariDataInfo {
        bits_per_pixel: read_u16(block, 3, o)?,
        columns: read_u16(block, 5, o)?,
        lines: read_u16(block, 7, o)?,
        compression_flag: block[9],
    })
}

fn parse_projection_info(block: &[u8], o: HimawariByteOrder) -> Result<HimawariProjectionInfo, Box<dyn Error>> {
    require_len(block, 51, "HSD projection information block")?;
    Ok(HimawariProjectionInfo {
        sub_lon_degrees: read_f64(block, 3, o)?,
        cfac: read_u32(block, 11, o)?,
        lfac: read_u32(block, 15, o)?,
        coff: read_f32(block, 19, o)?,
        loff: read_f32(block, 23, o)?,
        satellite_distance_km: read_f64(block, 27, o)?,
        equatorial_radius_km: read_f64(block, 35, o)?,
        polar_radius_km: read_f64(block, 43, o)?,
    })
}

fn parse_calibration_info(block: &[u8], o: HimawariByteOrder) -> Result<HimawariCalibrationInfo, Box<dyn Error>> {
    require_len(block, 59, "HSD calibration information block")?;
    let band_number = read_u16(block, 3, o)?;
    let infrared = band_number >= 7;
    let inverse_planck_coefficients = if infrared {
        require_len(block, 107, "HSD infrared calibration block")?;
        Some([read_f64(block, 59, o)?, read_f64(block, 67, o)?, read_f64(block, 75, o)?])
    } else {
        None
    };
    let physical_constants = if infrared {
        Some(HimawariPhysicalConstants {
            speed_of_light_m_s: read_f64(block, 83, o)?,
            planck_constant_j_s: read_f64(block, 91, o)?,
            boltzmann_constant_j_k: read_f64(block, 99, o)?,
        })
    } else {
        None
    };
    Ok(HimawariCalibrationInfo {
        band_number,
        central_wavelength_um: read_f64(block, 5, o)?,
        valid_bits_per_pixel: read_u16(block, 13, o)?,
        error_pixel_count: read_u16(block, 15, o)?,
        outside_scan_count: read_u16(block, 17, o)?,
        count_to_radiance_slope: read_f64(block, 19, o)?,
        count_to_radiance_intercept: read_f64(block, 27, o)?,
        planck_or_albedo_coefficients: [read_f64(block, 35, o)?, read_f64(block, 43, o)?, read_f64(block, 51, o)?],
        inverse_planck_coefficients,
        physical_constants,
    })
}

fn parse_segment_info(block: &[u8], o: HimawariByteOrder) -> Result<HimawariSegmentInfo, Box<dyn Error>> {
    require_len(block, 7, "HSD segment information block")?;
    Ok(HimawariSegmentInfo {
        total_segments: block[3],
        sequence_number: block[4],
        first_line_number: read_u16(block, 5, o)?,
    })
}

fn require_len(data: &[u8], needed: usize, context: &str) -> Result<(), Box<dyn Error>> {
    if data.len() < needed {
        Err(format!("{context} requires at least {needed} bytes, got {}", data.len()).into())
    } else {
        Ok(())
    }
}

fn read_u16(data: &[u8], offset: usize, o: HimawariByteOrder) -> Result<u16, Box<dyn Error>> {
    require_len(data, offset + 2, "u16 read")?;
    let b = [data[offset], data[offset + 1]];
    Ok(match o {
        HimawariByteOrder::Little => u16::from_le_bytes(b),
        HimawariByteOrder::Big => u16::from_be_bytes(b),
    })
}

fn read_u32(data: &[u8], offset: usize, o: HimawariByteOrder) -> Result<u32, Box<dyn Error>> {
    require_len(data, offset + 4, "u32 read")?;
    let b = [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]];
    Ok(match o {
        HimawariByteOrder::Little => u32::from_le_bytes(b),
        HimawariByteOrder::Big => u32::from_be_bytes(b),
    })
}

fn read_f32(data: &[u8], offset: usize, o: HimawariByteOrder) -> Result<f32, Box<dyn Error>> {
    Ok(f32::from_bits(read_u32(data, offset, o)?))
}

fn read_f64(data: &[u8], offset: usize, o: HimawariByteOrder) -> Result<f64, Box<dyn Error>> {
    require_len(data, offset + 8, "f64 read")?;
    let mut b = [0u8; 8];
    b.copy_from_slice(&data[offset..offset + 8]);
    Ok(match o {
        HimawariByteOrder::Little => f64::from_le_bytes(b),
        HimawariByteOrder::Big => f64::from_be_bytes(b),
    })
}

fn read_ascii(data: &[u8], offset: usize, len: usize) -> Result<String, Box<dyn Error>> {
    require_len(data, offset + len, "ASCII read")?;
    Ok(String::from_utf8_lossy(&data[offset..offset + len])
        .trim_matches(char::from(0))
        .trim()
        .to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mjd_epoch_is_the_unix_epoch_at_40587() {
        assert_eq!(mjd_to_unix(40587.0), 0.0);
        let t = mjd_to_datetime(40587.5).unwrap();
        assert_eq!(t.timestamp(), 43200);
    }
}
