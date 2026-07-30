//! GRIB2 file writer — creates GRIB2 files from data arrays.
//!
//! Implements simple packing (Data Representation Template 5.0) following
//! the WMO FM 92 GRIB Edition 2 specification. Output is compatible with
//! eccodes, wgrib2, and cfgrib.

use super::parser::{GridDefinition, ProductDefinition};
use chrono::NaiveDateTime;

/// Packing method for encoding data values.
#[derive(Debug, Clone)]
pub enum PackingMethod {
    /// Template 5.0: Simple grid-point packing.
    Simple {
        /// Number of bits per packed value (e.g., 16, 24).
        /// If 0, automatically chosen based on data range.
        bits_per_value: u8,
    },
}

impl Default for PackingMethod {
    fn default() -> Self {
        PackingMethod::Simple { bits_per_value: 16 }
    }
}

/// Builder for a single GRIB2 message (one field/variable).
#[derive(Debug, Clone)]
pub struct MessageBuilder {
    discipline: u8,
    center: u16,
    subcenter: u16,
    reference_time: NaiveDateTime,
    grid: GridDefinition,
    product: ProductDefinition,
    values: Vec<f64>,
    bitmap: Option<Vec<bool>>,
    packing: PackingMethod,
}

impl MessageBuilder {
    /// Create a new message builder with the given discipline and data values.
    ///
    /// - `discipline`: WMO discipline (0=Meteorological, 1=Hydrological, 2=Land surface, 10=Oceanographic)
    /// - `values`: The data values for all grid points (ny * nx elements).
    pub fn new(discipline: u8, values: Vec<f64>) -> Self {
        Self {
            discipline,
            center: 0,    // 0 = WMO Secretariat
            subcenter: 0,
            reference_time: chrono::NaiveDate::from_ymd_opt(2000, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid: GridDefinition::default(),
            product: ProductDefinition::default(),
            values,
            bitmap: None,
            packing: PackingMethod::default(),
        }
    }

    /// Set the originating center and subcenter.
    pub fn center(mut self, center: u16, subcenter: u16) -> Self {
        self.center = center;
        self.subcenter = subcenter;
        self
    }

    /// Set the reference time.
    pub fn reference_time(mut self, time: NaiveDateTime) -> Self {
        self.reference_time = time;
        self
    }

    /// Set the grid definition.
    pub fn grid(mut self, grid: GridDefinition) -> Self {
        self.grid = grid;
        self
    }

    /// Set the product definition.
    pub fn product(mut self, product: ProductDefinition) -> Self {
        self.product = product;
        self
    }

    /// Set the packing method.
    pub fn packing(mut self, method: PackingMethod) -> Self {
        self.packing = method;
        self
    }

    /// Set a bitmap (true = value present, false = missing).
    /// When a bitmap is used, only values where `bitmap[i] == true` are packed.
    /// The bitmap length must match the values length.
    pub fn bitmap(mut self, bitmap: Vec<bool>) -> Self {
        self.bitmap = Some(bitmap);
        self
    }
}

/// Builder for creating complete GRIB2 files containing one or more messages.
#[derive(Debug, Clone)]
pub struct Grib2Writer {
    messages: Vec<MessageBuilder>,
}

impl Grib2Writer {
    /// Create a new empty GRIB2 writer.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// Add a message/field to the GRIB2 file.
    pub fn add_message(mut self, msg: MessageBuilder) -> Self {
        self.messages.push(msg);
        self
    }

    /// Write the complete GRIB2 file to bytes.
    ///
    /// Each message is written as a separate GRIB2 message (with its own
    /// indicator and end sections), concatenated together as is standard
    /// for multi-message GRIB2 files.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        for msg in &self.messages {
            let msg_bytes = encode_message(msg)?;
            out.extend_from_slice(&msg_bytes);
        }
        Ok(out)
    }

    /// Write to a file.
    pub fn write_file(&self, path: &str) -> Result<(), String> {
        let data = self.to_bytes()?;
        std::fs::write(path, &data)
            .map_err(|e| format!("Failed to write GRIB2 file '{}': {}", path, e))
    }
}

impl Default for Grib2Writer {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════
// Internal encoding functions
// ═══════════════════════════════════════════════════════════

/// Encode a single GRIB2 message to bytes.
fn encode_message(msg: &MessageBuilder) -> Result<Vec<u8>, String> {
    // We build sections 1, 3, 4, 5, 6, 7 first, then prepend section 0
    // and append section 8, computing total length.
    let total_points = msg.grid.nx as usize * msg.grid.ny as usize;
    if total_points != msg.values.len() {
        return Err(format!(
            "Grid expects {} points ({}x{}), but values has {} elements",
            total_points, msg.grid.nx, msg.grid.ny, msg.values.len()
        ));
    }

    let sec1 = encode_section1(msg);
    let sec3 = encode_section3(&msg.grid)?;
    let sec4 = encode_section4(&msg.product);

    // Determine which values to pack and which bitmap to emit.
    let (bitmap, pack_values) = prepare_bitmap_and_values(msg)?;

    let (sec5, sec7) = encode_data(&pack_values, &msg.packing)?;
    let sec6 = encode_section6(&bitmap, total_points);

    // Total length = sec0(16) + sec1 + sec3 + sec4 + sec5 + sec6 + sec7 + sec8(4)
    let total_length: u64 = 16
        + sec1.len() as u64
        + sec3.len() as u64
        + sec4.len() as u64
        + sec5.len() as u64
        + sec6.len() as u64
        + sec7.len() as u64
        + 4;

    let mut out = Vec::with_capacity(total_length as usize);

    // Section 0: Indicator Section (16 bytes)
    out.extend_from_slice(b"GRIB");    // octets 1-4: "GRIB"
    out.extend_from_slice(&[0, 0]);    // octets 5-6: reserved
    out.push(msg.discipline);          // octet 7: discipline
    out.push(2);                       // octet 8: GRIB edition number = 2
    out.extend_from_slice(&total_length.to_be_bytes()); // octets 9-16: total length

    // Sections 1-7
    out.extend_from_slice(&sec1);
    out.extend_from_slice(&sec3);
    out.extend_from_slice(&sec4);
    out.extend_from_slice(&sec5);
    out.extend_from_slice(&sec6);
    out.extend_from_slice(&sec7);

    // Section 8: End Section
    out.extend_from_slice(b"7777");

    debug_assert_eq!(out.len() as u64, total_length);

    Ok(out)
}

fn prepare_bitmap_and_values(msg: &MessageBuilder) -> Result<(Option<Vec<bool>>, Vec<f64>), String> {
    match &msg.bitmap {
        Some(bitmap) => {
            if bitmap.len() != msg.values.len() {
                return Err(format!(
                    "Bitmap length {} does not match values length {}",
                    bitmap.len(),
                    msg.values.len()
                ));
            }

            let mut pack_values = Vec::with_capacity(bitmap.iter().filter(|&&present| present).count());
            for (idx, (&value, &present)) in msg.values.iter().zip(bitmap.iter()).enumerate() {
                if present {
                    if !value.is_finite() {
                        return Err(format!(
                            "Non-finite value at index {} is marked present in the bitmap",
                            idx
                        ));
                    }
                    pack_values.push(value);
                }
            }

            Ok((Some(bitmap.clone()), pack_values))
        }
        None => {
            if msg.values.iter().all(|v| v.is_finite()) {
                Ok((None, msg.values.clone()))
            } else {
                let bitmap: Vec<bool> = msg.values.iter().map(|v| v.is_finite()).collect();
                let pack_values = msg
                    .values
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .collect();
                Ok((Some(bitmap), pack_values))
            }
        }
    }
}

/// Section 1: Identification Section.
///
/// 21 bytes total (standard for GRIB2 Section 1).
fn encode_section1(msg: &MessageBuilder) -> Vec<u8> {
    let mut sec = Vec::with_capacity(21);

    // Octets 1-4: length of section (21)
    sec.extend_from_slice(&21u32.to_be_bytes());
    // Octet 5: section number = 1
    sec.push(1);
    // Octets 6-7: originating center
    sec.extend_from_slice(&msg.center.to_be_bytes());
    // Octets 8-9: originating subcenter
    sec.extend_from_slice(&msg.subcenter.to_be_bytes());
    // Octet 10: GRIB master tables version number (latest = 2)
    sec.push(2);
    // Octet 11: GRIB local tables version number
    sec.push(1);
    // Octet 12: significance of reference time (1 = start of forecast)
    sec.push(1);

    let dt = msg.reference_time;
    // Octets 13-14: year
    sec.extend_from_slice(&(dt.and_utc().timestamp() as u16).to_be_bytes());
    // Overwrite with actual year
    let year = dt.format("%Y").to_string().parse::<u16>().unwrap_or(2000);
    sec[11..13].copy_from_slice(&year.to_be_bytes());
    // Fix: rebuild properly
    sec.clear();

    sec.extend_from_slice(&21u32.to_be_bytes());    // length
    sec.push(1);                                     // section number
    sec.extend_from_slice(&msg.center.to_be_bytes());
    sec.extend_from_slice(&msg.subcenter.to_be_bytes());
    sec.push(2);  // master tables version
    sec.push(1);  // local tables version
    sec.push(1);  // significance of reference time

    let year = dt.format("%Y").to_string().parse::<u16>().unwrap_or(2000);
    sec.extend_from_slice(&year.to_be_bytes());      // octets 13-14
    sec.push(dt.format("%m").to_string().parse::<u8>().unwrap_or(1));  // 15: month
    sec.push(dt.format("%d").to_string().parse::<u8>().unwrap_or(1));  // 16: day
    sec.push(dt.format("%H").to_string().parse::<u8>().unwrap_or(0));  // 17: hour
    sec.push(dt.format("%M").to_string().parse::<u8>().unwrap_or(0));  // 18: minute
    sec.push(dt.format("%S").to_string().parse::<u8>().unwrap_or(0));  // 19: second
    // Octet 20: production status (0 = operational)
    sec.push(0);
    // Octet 21: type of processed data (1 = forecast)
    sec.push(1);

    debug_assert_eq!(sec.len(), 21);
    sec
}

/// Section 3: Grid Definition Section.
fn encode_section3(grid: &GridDefinition) -> Result<Vec<u8>, String> {
    match grid.template {
        0 => encode_grid_template_0(grid),
        30 => encode_grid_template_30(grid),
        _ => Err(format!(
            "Unsupported grid template {} for writing. Supported: 0 (lat/lon), 30 (Lambert Conformal)",
            grid.template
        )),
    }
}

/// Grid Definition Template 3.0: Latitude/Longitude (Equidistant Cylindrical).
///
/// Section 3 for template 0 is 72 bytes.
fn encode_grid_template_0(grid: &GridDefinition) -> Result<Vec<u8>, String> {
    let section_len: u32 = 72;
    let mut sec = Vec::with_capacity(section_len as usize);

    // Octets 1-4: length of section
    sec.extend_from_slice(&section_len.to_be_bytes());
    // Octet 5: section number
    sec.push(3);
    // Octet 6: source of grid definition (0 = specified in Code Table 3.1)
    sec.push(0);
    // Octets 7-10: number of data points
    let npoints = grid.nx * grid.ny;
    sec.extend_from_slice(&npoints.to_be_bytes());
    // Octet 11: number of octets for optional list of numbers
    sec.push(0);
    // Octet 12: interpretation of list of numbers
    sec.push(0);
    // Octets 13-14: grid definition template number
    sec.extend_from_slice(&0u16.to_be_bytes());

    // Template 3.0 specific fields:
    // Octet 15: shape of earth (6 = spherical, radius 6371229m)
    sec.push(6);
    // Octet 16: scale factor of radius
    sec.push(0);
    // Octets 17-20: scaled value of radius
    sec.extend_from_slice(&0u32.to_be_bytes());
    // Octet 21: scale factor of major axis
    sec.push(0);
    // Octets 22-25: scaled value of major axis
    sec.extend_from_slice(&0u32.to_be_bytes());
    // Octet 26: scale factor of minor axis
    sec.push(0);
    // Octets 27-30: scaled value of minor axis
    sec.extend_from_slice(&0u32.to_be_bytes());

    // Octets 31-34: Ni (nx)
    sec.extend_from_slice(&grid.nx.to_be_bytes());
    // Octets 35-38: Nj (ny)
    sec.extend_from_slice(&grid.ny.to_be_bytes());

    // Octets 39-42: basic angle (0)
    sec.extend_from_slice(&0u32.to_be_bytes());
    // Octets 43-46: subdivisions of basic angle (0 = use 10^6)
    sec.extend_from_slice(&0u32.to_be_bytes());

    // Octets 47-50: La1 (latitude of first grid point) — signed, microdegrees
    sec.extend_from_slice(&encode_signed_u32(grid.lat1, 1_000_000.0));
    // Octets 51-54: Lo1 (longitude of first grid point) — signed, microdegrees
    sec.extend_from_slice(&encode_signed_u32(grid.lon1, 1_000_000.0));
    // Octet 55: resolution and component flags
    sec.push(0x30); // bit 3+4 set: i and j direction increments given
    // Octets 56-59: La2 (latitude of last grid point)
    sec.extend_from_slice(&encode_signed_u32(grid.lat2, 1_000_000.0));
    // Octets 60-63: Lo2 (longitude of last grid point)
    sec.extend_from_slice(&encode_signed_u32(grid.lon2, 1_000_000.0));
    // Octets 64-67: Di (i direction increment) — unsigned microdegrees
    sec.extend_from_slice(&((grid.dx * 1_000_000.0).round() as u32).to_be_bytes());
    // Octets 68-71: Dj (j direction increment) — unsigned microdegrees
    sec.extend_from_slice(&((grid.dy * 1_000_000.0).round() as u32).to_be_bytes());
    // Octet 72: scanning mode
    sec.push(grid.scan_mode);

    debug_assert_eq!(sec.len(), section_len as usize);
    Ok(sec)
}

/// Grid Definition Template 3.30: Lambert Conformal.
///
/// Section 3 for template 30 is 81 bytes.
fn encode_grid_template_30(grid: &GridDefinition) -> Result<Vec<u8>, String> {
    let section_len: u32 = 81;
    let mut sec = Vec::with_capacity(section_len as usize);

    // Octets 1-4: length
    sec.extend_from_slice(&section_len.to_be_bytes());
    // Octet 5: section number
    sec.push(3);
    // Octet 6: source of grid definition
    sec.push(0);
    // Octets 7-10: number of data points
    let npoints = grid.nx * grid.ny;
    sec.extend_from_slice(&npoints.to_be_bytes());
    // Octet 11: optional list
    sec.push(0);
    // Octet 12: interpretation
    sec.push(0);
    // Octets 13-14: template number = 30
    sec.extend_from_slice(&30u16.to_be_bytes());

    // Template 3.30 specific fields:
    // Octet 15: shape of earth (6 = spherical 6371229m)
    sec.push(6);
    // Octet 16: scale factor of radius
    sec.push(0);
    // Octets 17-20: scaled value of radius
    sec.extend_from_slice(&0u32.to_be_bytes());
    // Octet 21: scale factor of major axis
    sec.push(0);
    // Octets 22-25: scaled value of major axis
    sec.extend_from_slice(&0u32.to_be_bytes());
    // Octet 26: scale factor of minor axis
    sec.push(0);
    // Octets 27-30: scaled value of minor axis
    sec.extend_from_slice(&0u32.to_be_bytes());

    // Octets 31-34: Nx
    sec.extend_from_slice(&grid.nx.to_be_bytes());
    // Octets 35-38: Ny
    sec.extend_from_slice(&grid.ny.to_be_bytes());
    // Octets 39-42: La1 (microdegrees)
    sec.extend_from_slice(&encode_signed_u32(grid.lat1, 1_000_000.0));
    // Octets 43-46: Lo1 (microdegrees)
    sec.extend_from_slice(&encode_signed_u32(grid.lon1, 1_000_000.0));
    // Octet 47: resolution and component flags
    sec.push(0x30);

    // Octets 48-51: LaD (latitude where Dx/Dy are specified) — for Lambert this
    // is sometimes set to latin1. Use latin1 if lad is 0.
    let lad = if grid.lad != 0.0 { grid.lad } else { grid.latin1 };
    sec.extend_from_slice(&encode_signed_u32(lad, 1_000_000.0));
    // Octets 52-55: LoV (microdegrees)
    sec.extend_from_slice(&encode_signed_u32(grid.lov, 1_000_000.0));
    // Octets 56-59: Dx (millimeters)
    sec.extend_from_slice(&((grid.dx * 1000.0).round() as u32).to_be_bytes());
    // Octets 60-63: Dy (millimeters)
    sec.extend_from_slice(&((grid.dy * 1000.0).round() as u32).to_be_bytes());
    // Octet 64: projection center flag
    sec.push(grid.projection_center_flag);
    // Octet 65: scanning mode
    sec.push(grid.scan_mode);
    // Octets 66-69: Latin1 (microdegrees)
    sec.extend_from_slice(&encode_signed_u32(grid.latin1, 1_000_000.0));
    // Octets 70-73: Latin2 (microdegrees)
    sec.extend_from_slice(&encode_signed_u32(grid.latin2, 1_000_000.0));

    // Octets 74-77: Latitude of southern pole (microdegrees)
    sec.extend_from_slice(&encode_signed_u32(-90.0, 1_000_000.0));
    // Octets 78-81: Longitude of southern pole (microdegrees)
    sec.extend_from_slice(&0u32.to_be_bytes());

    debug_assert_eq!(sec.len(), section_len as usize);
    Ok(sec)
}

/// Section 4: Product Definition Section.
///
/// Template 4.0: Analysis or forecast at a horizontal level at a point in time.
/// 34 bytes total.
fn encode_section4(prod: &ProductDefinition) -> Vec<u8> {
    let section_len: u32 = 34;
    let mut sec = Vec::with_capacity(section_len as usize);

    // Octets 1-4: length
    sec.extend_from_slice(&section_len.to_be_bytes());
    // Octet 5: section number
    sec.push(4);
    // Octets 6-7: number of coordinate values after template (0)
    sec.extend_from_slice(&0u16.to_be_bytes());
    // Octets 8-9: product definition template number
    sec.extend_from_slice(&prod.template.to_be_bytes());

    // Template 4.0 fields:
    // Octet 10: parameter category
    sec.push(prod.parameter_category);
    // Octet 11: parameter number
    sec.push(prod.parameter_number);
    // Octet 12: type of generating process (2 = forecast)
    sec.push(prod.generating_process);
    // Octet 13: background generating process identifier
    sec.push(0);
    // Octet 14: analysis or forecast generating process identified
    sec.push(0);
    // Octets 15-16: hours of observational data cutoff after reference time
    sec.extend_from_slice(&0u16.to_be_bytes());
    // Octet 17: minutes of observational data cutoff after reference time
    sec.push(0);
    // Octet 18: indicator of unit of time range
    sec.push(prod.time_range_unit);
    // Octets 19-22: forecast time in units defined by octet 18
    sec.extend_from_slice(&prod.forecast_time.to_be_bytes());
    // Octet 23: type of first fixed surface (level type)
    sec.push(prod.level_type);

    // Octets 24-28: scale factor and scaled value of first fixed surface
    let (scale_factor, scaled_value) = encode_level_value(prod.level_value);
    sec.push(scale_factor);
    sec.extend_from_slice(&scaled_value.to_be_bytes());

    // Octet 29: type of second fixed surface (255 = missing)
    sec.push(255);
    // Octet 30: scale factor of second fixed surface
    sec.push(0);
    // Octets 31-34: scaled value of second fixed surface
    sec.extend_from_slice(&0u32.to_be_bytes());

    debug_assert_eq!(sec.len(), section_len as usize);
    sec
}

/// Section 5: Data Representation Section (Template 5.0: Simple Packing).
/// Section 7: Data Section (packed values).
///
/// Returns (section5_bytes, section7_bytes).
fn encode_data(values: &[f64], packing: &PackingMethod) -> Result<(Vec<u8>, Vec<u8>), String> {
    match packing {
        PackingMethod::Simple { bits_per_value } => {
            encode_simple_packing(values, *bits_per_value)
        }
    }
}

/// Encode data using simple packing (Template 5.0).
///
/// Packing formula: X = nint((Y * 10^D - R) / 2^E)
/// Unpacking formula: Y = (R + X * 2^E) * 10^(-D)
///
/// We use D=0 (no decimal scaling) and E=0 (no binary scaling beyond
/// what's needed), and compute R = min(values). The packed integer is:
///   X = nint((Y - R) / 2^E)
/// where 2^E = (max - min) / (2^bpv - 1)
fn encode_simple_packing(
    values: &[f64],
    mut bits_per_value: u8,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let num_points = values.len() as u32;

    // Handle edge case: no values
    if values.is_empty() {
        let sec5 = encode_section5_simple(0.0f32, 0, 0, 0, 0);
        let sec7 = encode_section7(&[]);
        return Ok((sec5, sec7));
    }

    // Filter finite values for computing stats
    let finite_values: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite_values.is_empty() {
        // All values are NaN/Inf — write zeros
        let sec5 = encode_section5_simple(0.0f32, 0, 0, bits_per_value, num_points);
        let packed_bytes = vec![0u8; (num_points as usize * bits_per_value as usize + 7) / 8];
        let sec7 = encode_section7(&packed_bytes);
        return Ok((sec5, sec7));
    }

    let min_val = finite_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = finite_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Auto-determine bits_per_value if set to 0
    if bits_per_value == 0 {
        bits_per_value = if (max_val - min_val).abs() < 1e-30 {
            // All values are the same — 0 bits needed, but use at least 8
            // to avoid issues with readers.
            16
        } else {
            16
        };
    }

    let decimal_scale: i16 = 0;
    let reference_value = min_val as f32;
    let r = reference_value as f64;

    // Compute binary scale factor E such that:
    //   (max - R) / 2^E <= 2^bpv - 1
    //   2^E >= (max - R) / (2^bpv - 1)
    //   E >= log2((max - R) / (2^bpv - 1))
    let range = max_val - r;
    let max_packed = (1u64 << bits_per_value as u64) - 1;

    let binary_scale: i16 = if range < 1e-30 || max_packed == 0 {
        // All values are identical (or nearly so)
        0
    } else {
        let e = (range / max_packed as f64).log2().ceil() as i16;
        e
    };

    let two_e = 2.0_f64.powi(binary_scale as i32);

    // Pack values: X = nint((Y - R) / 2^E)
    let mut packed_ints: Vec<u64> = Vec::with_capacity(values.len());
    for &v in values {
        let y = if v.is_finite() { v } else { r }; // NaN gets min value
        let x = ((y - r) / two_e).round();
        let x = x.max(0.0).min(max_packed as f64) as u64;
        packed_ints.push(x);
    }

    // Pack integers into a bitstream
    let total_bits = values.len() * bits_per_value as usize;
    let total_bytes = (total_bits + 7) / 8;
    let mut packed_bytes = vec![0u8; total_bytes];

    let bpv = bits_per_value as usize;
    for (i, &x) in packed_ints.iter().enumerate() {
        write_bits(&mut packed_bytes, i * bpv, bpv, x);
    }

    let sec5 = encode_section5_simple(
        reference_value,
        binary_scale,
        decimal_scale,
        bits_per_value,
        num_points,
    );
    let sec7 = encode_section7(&packed_bytes);

    Ok((sec5, sec7))
}

/// Build Section 5 for simple packing (Template 5.0).
///
/// 21 bytes total.
fn encode_section5_simple(
    reference_value: f32,
    binary_scale: i16,
    decimal_scale: i16,
    bits_per_value: u8,
    num_points: u32,
) -> Vec<u8> {
    let section_len: u32 = 21;
    let mut sec = Vec::with_capacity(section_len as usize);

    // Octets 1-4: length
    sec.extend_from_slice(&section_len.to_be_bytes());
    // Octet 5: section number
    sec.push(5);
    // Octets 6-9: number of data points
    sec.extend_from_slice(&num_points.to_be_bytes());
    // Octets 10-11: data representation template number (0 = simple packing)
    sec.extend_from_slice(&0u16.to_be_bytes());

    // Template 5.0 fields:
    // Octets 12-15: reference value (IEEE 754 single precision)
    sec.extend_from_slice(&reference_value.to_be_bytes());
    // Octets 16-17: binary scale factor (signed, two's complement)
    sec.extend_from_slice(&encode_signed_u16_grib(binary_scale));
    // Octets 18-19: decimal scale factor (signed, two's complement)
    sec.extend_from_slice(&encode_signed_u16_grib(decimal_scale));
    // Octet 20: number of bits per packed value
    sec.push(bits_per_value);
    // Octet 21: type of original field values (0 = floating point)
    sec.push(0);

    debug_assert_eq!(sec.len(), section_len as usize);
    sec
}

/// Section 6: Bitmap Section.
fn encode_section6(bitmap: &Option<Vec<bool>>, total_points: usize) -> Vec<u8> {
    match bitmap {
        None => {
            // No bitmap — indicator = 255
            let section_len: u32 = 6;
            let mut sec = Vec::with_capacity(section_len as usize);
            sec.extend_from_slice(&section_len.to_be_bytes());
            sec.push(6);  // section number
            sec.push(255); // bitmap indicator: not present
            sec
        }
        Some(bm) => {
            // Bitmap present — indicator = 0
            // Pack bits: 1 byte per 8 grid points, MSB first
            let bitmap_bytes = (total_points + 7) / 8;
            let section_len = 6 + bitmap_bytes as u32;
            let mut sec = Vec::with_capacity(section_len as usize);
            sec.extend_from_slice(&section_len.to_be_bytes());
            sec.push(6);  // section number
            sec.push(0);  // bitmap indicator: bitmap follows

            let mut bytes = vec![0u8; bitmap_bytes];
            for (i, &present) in bm.iter().enumerate().take(total_points) {
                if present {
                    let byte_idx = i / 8;
                    let bit_idx = 7 - (i % 8);
                    bytes[byte_idx] |= 1 << bit_idx;
                }
            }
            sec.extend_from_slice(&bytes);

            debug_assert_eq!(sec.len(), section_len as usize);
            sec
        }
    }
}

/// Section 7: Data Section.
fn encode_section7(packed_data: &[u8]) -> Vec<u8> {
    let section_len = 5 + packed_data.len() as u32;
    let mut sec = Vec::with_capacity(section_len as usize);
    sec.extend_from_slice(&section_len.to_be_bytes());
    sec.push(7); // section number
    sec.extend_from_slice(packed_data);
    sec
}

// ═══════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════

/// Encode a floating point value as a signed 32-bit GRIB2 value
/// using sign-magnitude format (MSB = sign, rest = magnitude).
///
/// The value is multiplied by `scale` first (e.g., 1_000_000 for microdegrees).
fn encode_signed_u32(value: f64, scale: f64) -> [u8; 4] {
    let scaled = (value * scale).round() as i64;
    let sign: u32 = if scaled < 0 { 1 << 31 } else { 0 };
    let magnitude = scaled.unsigned_abs() as u32 & 0x7FFF_FFFF;
    (sign | magnitude).to_be_bytes()
}

/// Encode a signed 16-bit value in GRIB2 sign-magnitude format.
fn encode_signed_u16_grib(value: i16) -> [u8; 2] {
    let sign: u16 = if value < 0 { 1 << 15 } else { 0 };
    let magnitude = value.unsigned_abs() & 0x7FFF;
    (sign | magnitude).to_be_bytes()
}

/// Encode a level value into (scale_factor, scaled_value) for Section 4.
///
/// For integer levels (e.g., 2 m, 10 m, 100000 Pa), the scale factor is 0
/// and scaled value is the integer. For fractional levels, we find the
/// smallest scale factor that represents the value exactly.
fn encode_level_value(value: f64) -> (u8, u32) {
    // Try scale factors 0..6 until we get an integer
    for sf in 0u8..7 {
        let factor = 10.0_f64.powi(sf as i32);
        let scaled = value * factor;
        let rounded = scaled.round();
        if (scaled - rounded).abs() < 0.01 {
            return (sf, rounded as u32);
        }
    }
    // Fallback: use scale factor 0
    (0, value.round() as u32)
}

/// Write `n_bits` of `value` into `buf` starting at bit offset `bit_offset`.
/// Big-endian bit ordering (MSB first within each byte).
fn write_bits(buf: &mut [u8], bit_offset: usize, n_bits: usize, value: u64) {
    for i in 0..n_bits {
        let bit_val = (value >> (n_bits - 1 - i)) & 1;
        let pos = bit_offset + i;
        let byte_idx = pos / 8;
        let bit_idx = 7 - (pos % 8);
        if byte_idx < buf.len() {
            if bit_val == 1 {
                buf[byte_idx] |= 1 << bit_idx;
            }
            // bit_val == 0: already zero from initialization
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grib2::parser::{Grib2File, GridDefinition, ProductDefinition};
    use crate::grib2::unpack::unpack_message;

    #[test]
    fn roundtrip_simple_constant() {
        // All values are the same
        let values = vec![273.15; 9];
        let grid = GridDefinition {
            template: 0,
            nx: 3,
            ny: 3,
            lat1: 0.0,
            lon1: 0.0,
            lat2: 2.0,
            lon2: 2.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, values.clone())
                .grid(grid)
                .packing(PackingMethod::Simple { bits_per_value: 16 }),
        );

        let bytes = writer.to_bytes().unwrap();

        // Parse back
        let grib = Grib2File::from_bytes(&bytes).unwrap();
        assert_eq!(grib.messages.len(), 1);
        let msg = &grib.messages[0];
        assert_eq!(msg.grid.nx, 3);
        assert_eq!(msg.grid.ny, 3);

        let unpacked = unpack_message(msg).unwrap();
        assert_eq!(unpacked.len(), 9);
        for (i, &v) in unpacked.iter().enumerate() {
            assert!(
                (v - 273.15).abs() < 0.01,
                "Value[{}]: expected 273.15, got {}",
                i, v
            );
        }
    }

    #[test]
    fn roundtrip_simple_ramp() {
        // Values from 0.0 to 8.0
        let values: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let grid = GridDefinition {
            template: 0,
            nx: 3,
            ny: 3,
            lat1: 30.0,
            lon1: -100.0,
            lat2: 32.0,
            lon2: -98.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let product = ProductDefinition {
            template: 0,
            parameter_category: 0, // Temperature
            parameter_number: 0,   // Temperature
            generating_process: 2,
            forecast_time: 0,
            time_range_unit: 1, // Hour
            level_type: 103,    // Height above ground
            level_value: 2.0,   // 2 m
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, values.clone())
                .grid(grid)
                .product(product)
                .packing(PackingMethod::Simple { bits_per_value: 16 }),
        );

        let bytes = writer.to_bytes().unwrap();

        let grib = Grib2File::from_bytes(&bytes).unwrap();
        assert_eq!(grib.messages.len(), 1);
        let msg = &grib.messages[0];
        assert_eq!(msg.discipline, 0);
        assert_eq!(msg.product.parameter_category, 0);
        assert_eq!(msg.product.parameter_number, 0);
        assert_eq!(msg.product.level_type, 103);

        let unpacked = unpack_message(msg).unwrap();
        assert_eq!(unpacked.len(), 9);
        for (i, &v) in unpacked.iter().enumerate() {
            let expected = i as f64;
            assert!(
                (v - expected).abs() < 0.01,
                "Value[{}]: expected {}, got {}",
                i, expected, v
            );
        }
    }

    #[test]
    fn roundtrip_temperature_kelvin() {
        // Realistic 2m temperature range in Kelvin
        let values: Vec<f64> = (0..100).map(|i| 250.0 + i as f64 * 0.5).collect();
        let grid = GridDefinition {
            template: 0,
            nx: 10,
            ny: 10,
            lat1: 30.0,
            lon1: -100.0,
            lat2: 39.0,
            lon2: -91.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, values.clone())
                .grid(grid)
                .packing(PackingMethod::Simple { bits_per_value: 16 }),
        );

        let bytes = writer.to_bytes().unwrap();
        let grib = Grib2File::from_bytes(&bytes).unwrap();
        let unpacked = unpack_message(&grib.messages[0]).unwrap();
        assert_eq!(unpacked.len(), 100);

        for (i, (&orig, &unpk)) in values.iter().zip(unpacked.iter()).enumerate() {
            assert!(
                (orig - unpk).abs() < 0.01,
                "Value[{}]: expected {}, got {} (diff={})",
                i, orig, unpk, (orig - unpk).abs()
            );
        }
    }

    #[test]
    fn roundtrip_with_bitmap() {
        // 3x3 grid, center value is missing
        let values = vec![1.0, 2.0, 3.0, 4.0, f64::NAN, 6.0, 7.0, 8.0, 9.0];
        let bitmap = vec![true, true, true, true, false, true, true, true, true];

        let grid = GridDefinition {
            template: 0,
            nx: 3,
            ny: 3,
            lat1: 0.0,
            lon1: 0.0,
            lat2: 2.0,
            lon2: 2.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, values.clone())
                .grid(grid)
                .bitmap(bitmap)
                .packing(PackingMethod::Simple { bits_per_value: 16 }),
        );

        let bytes = writer.to_bytes().unwrap();
        let grib = Grib2File::from_bytes(&bytes).unwrap();
        let msg = &grib.messages[0];
        assert!(msg.bitmap.is_some());

        let unpacked = unpack_message(msg).unwrap();
        // With bitmap, unpack returns bitmap.len() entries (rounded up to byte boundary).
        // The bitmap is stored as whole bytes, so 9 bits -> 16 bits (2 bytes).
        // The first 9 entries correspond to our grid points.
        assert!(unpacked.len() >= 9);
        assert!((unpacked[0] - 1.0).abs() < 0.01);
        assert!((unpacked[1] - 2.0).abs() < 0.01);
        assert!((unpacked[3] - 4.0).abs() < 0.01);
        assert!(unpacked[4].is_nan(), "Index 4 should be NaN");
        assert!((unpacked[5] - 6.0).abs() < 0.01);
        assert!((unpacked[8] - 9.0).abs() < 0.01);
        // Trailing bits (padding) should be NaN since they are 0 in the bitmap
        for i in 9..unpacked.len() {
            assert!(unpacked[i].is_nan(), "Padding bit {} should be NaN", i);
        }
    }

    #[test]
    fn roundtrip_nonfinite_values_auto_bitmap() {
        let values = vec![1.0, f64::NAN, 3.0, f64::INFINITY];
        let grid = GridDefinition {
            template: 0,
            nx: 2,
            ny: 2,
            lat1: 0.0,
            lon1: 0.0,
            lat2: 1.0,
            lon2: 1.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, values)
                .grid(grid)
                .packing(PackingMethod::Simple { bits_per_value: 16 }),
        );

        let bytes = writer.to_bytes().unwrap();
        let grib = Grib2File::from_bytes(&bytes).unwrap();
        let msg = &grib.messages[0];
        assert!(msg.bitmap.is_some(), "non-finite values should create a bitmap");

        let unpacked = unpack_message(msg).unwrap();
        assert!((unpacked[0] - 1.0).abs() < 0.01);
        assert!(unpacked[1].is_nan());
        assert!((unpacked[2] - 3.0).abs() < 0.01);
        assert!(unpacked[3].is_nan());
    }

    #[test]
    fn bitmap_length_mismatch_is_error() {
        let grid = GridDefinition {
            template: 0,
            nx: 2,
            ny: 2,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, vec![1.0, 2.0, 3.0, 4.0])
                .grid(grid)
                .bitmap(vec![true, false, true]),
        );

        let err = writer.to_bytes().unwrap_err();
        assert!(err.contains("Bitmap length 3 does not match values length 4"));
    }

    #[test]
    fn bitmap_present_nonfinite_value_is_error() {
        let grid = GridDefinition {
            template: 0,
            nx: 2,
            ny: 2,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, vec![1.0, f64::NAN, 3.0, 4.0])
                .grid(grid)
                .bitmap(vec![true, true, true, true]),
        );

        let err = writer.to_bytes().unwrap_err();
        assert!(err.contains("Non-finite value at index 1 is marked present"));
    }

    #[test]
    fn roundtrip_lambert_grid() {
        // Lambert Conformal (HRRR-like) grid
        let values: Vec<f64> = (0..25).map(|i| 270.0 + i as f64 * 0.1).collect();
        let grid = GridDefinition {
            template: 30,
            nx: 5,
            ny: 5,
            lat1: 21.138123,
            lon1: 237.280472,
            lat2: 0.0,
            lon2: 0.0,
            dx: 3000.0,
            dy: 3000.0,
            latin1: 38.5,
            latin2: 38.5,
            lov: 262.5,
            scan_mode: 0x40,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, values.clone())
                .grid(grid)
                .packing(PackingMethod::Simple { bits_per_value: 24 }),
        );

        let bytes = writer.to_bytes().unwrap();
        let grib = Grib2File::from_bytes(&bytes).unwrap();
        assert_eq!(grib.messages.len(), 1);

        let msg = &grib.messages[0];
        assert_eq!(msg.grid.template, 30);
        assert_eq!(msg.grid.nx, 5);
        assert_eq!(msg.grid.ny, 5);
        assert!((msg.grid.latin1 - 38.5).abs() < 0.001);
        assert!((msg.grid.latin2 - 38.5).abs() < 0.001);
        assert!((msg.grid.lov - 262.5).abs() < 0.001);
    }

    #[test]
    fn roundtrip_multi_message() {
        let grid = GridDefinition {
            template: 0,
            nx: 4,
            ny: 4,
            lat1: 30.0,
            lon1: -100.0,
            lat2: 33.0,
            lon2: -97.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let temp_values: Vec<f64> = (0..16).map(|i| 273.0 + i as f64).collect();
        let wind_values: Vec<f64> = (0..16).map(|i| i as f64 * 0.5).collect();

        let writer = Grib2Writer::new()
            .add_message(
                MessageBuilder::new(0, temp_values.clone())
                    .grid(grid.clone())
                    .product(ProductDefinition {
                        parameter_category: 0,
                        parameter_number: 0,
                        level_type: 103,
                        level_value: 2.0,
                        ..ProductDefinition::default()
                    })
                    .packing(PackingMethod::Simple { bits_per_value: 16 }),
            )
            .add_message(
                MessageBuilder::new(0, wind_values.clone())
                    .grid(grid.clone())
                    .product(ProductDefinition {
                        parameter_category: 2,
                        parameter_number: 2,
                        level_type: 103,
                        level_value: 10.0,
                        ..ProductDefinition::default()
                    })
                    .packing(PackingMethod::Simple { bits_per_value: 16 }),
            );

        let bytes = writer.to_bytes().unwrap();
        let grib = Grib2File::from_bytes(&bytes).unwrap();
        assert_eq!(grib.messages.len(), 2);

        // Check first message (temperature)
        let msg0 = &grib.messages[0];
        assert_eq!(msg0.product.parameter_category, 0);
        assert_eq!(msg0.product.parameter_number, 0);
        let vals0 = unpack_message(msg0).unwrap();
        assert_eq!(vals0.len(), 16);
        assert!((vals0[0] - 273.0).abs() < 0.01);

        // Check second message (wind)
        let msg1 = &grib.messages[1];
        assert_eq!(msg1.product.parameter_category, 2);
        assert_eq!(msg1.product.parameter_number, 2);
        let vals1 = unpack_message(msg1).unwrap();
        assert_eq!(vals1.len(), 16);
        assert!((vals1[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn roundtrip_write_file() {
        let values: Vec<f64> = (0..4).map(|i| i as f64 * 10.0).collect();
        let grid = GridDefinition {
            template: 0,
            nx: 2,
            ny: 2,
            lat1: 0.0,
            lon1: 0.0,
            lat2: 1.0,
            lon2: 1.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, values.clone()).grid(grid),
        );

        let tmp = std::env::temp_dir().join("metrust_test_writer.grib2");
        let path = tmp.to_str().unwrap();
        writer.write_file(path).unwrap();

        // Read back
        let grib = Grib2File::open(path).unwrap();
        assert_eq!(grib.messages.len(), 1);
        let unpacked = unpack_message(&grib.messages[0]).unwrap();
        assert_eq!(unpacked.len(), 4);
        for (i, &v) in unpacked.iter().enumerate() {
            let expected = i as f64 * 10.0;
            assert!(
                (v - expected).abs() < 0.1,
                "Value[{}]: expected {}, got {}",
                i, expected, v
            );
        }

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn write_bits_basic() {
        let mut buf = vec![0u8; 3];

        // Write 0xFF (8 bits) at offset 0
        write_bits(&mut buf, 0, 8, 0xFF);
        assert_eq!(buf[0], 0xFF);

        // Write 0x5 (4 bits) at offset 8
        write_bits(&mut buf, 8, 4, 0x5);
        assert_eq!(buf[1] & 0xF0, 0x50);

        // Write 0xA (4 bits) at offset 12
        write_bits(&mut buf, 12, 4, 0xA);
        assert_eq!(buf[1], 0x5A);
    }

    #[test]
    fn encode_signed_u32_positive() {
        let bytes = encode_signed_u32(45.5, 1_000_000.0);
        let raw = u32::from_be_bytes(bytes);
        assert_eq!(raw & 0x80000000, 0); // positive
        assert_eq!(raw, 45_500_000);
    }

    #[test]
    fn encode_signed_u32_negative() {
        let bytes = encode_signed_u32(-45.5, 1_000_000.0);
        let raw = u32::from_be_bytes(bytes);
        assert_ne!(raw & 0x80000000, 0); // negative (sign bit set)
        assert_eq!(raw & 0x7FFFFFFF, 45_500_000);
    }

    #[test]
    fn grib2_magic_and_end_marker() {
        let values = vec![1.0; 4];
        let grid = GridDefinition {
            template: 0,
            nx: 2,
            ny: 2,
            lat1: 0.0,
            lon1: 0.0,
            lat2: 1.0,
            lon2: 1.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, values).grid(grid),
        );
        let bytes = writer.to_bytes().unwrap();

        // Check magic
        assert_eq!(&bytes[0..4], b"GRIB");
        // Check edition
        assert_eq!(bytes[7], 2);
        // Check end marker
        assert_eq!(&bytes[bytes.len() - 4..], b"7777");

        // Check total length matches
        let total_len = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(total_len as usize, bytes.len());
    }

    #[test]
    fn reference_time_roundtrip() {
        let dt = chrono::NaiveDate::from_ymd_opt(2025, 6, 15)
            .unwrap()
            .and_hms_opt(12, 30, 45)
            .unwrap();

        let values = vec![1.0; 4];
        let grid = GridDefinition {
            template: 0,
            nx: 2,
            ny: 2,
            lat1: 0.0,
            lon1: 0.0,
            lat2: 1.0,
            lon2: 1.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, values)
                .grid(grid)
                .reference_time(dt),
        );
        let bytes = writer.to_bytes().unwrap();
        let grib = Grib2File::from_bytes(&bytes).unwrap();
        let msg = &grib.messages[0];
        assert_eq!(msg.reference_time, dt);
    }
}
