//! GRIB Edition 1 message parser.
//!
//! Parses all sections of a GRIB1 message: Indicator (Section 0),
//! Product Definition (Section 1), Grid Description (Section 2, optional),
//! Bit Map (Section 3, optional), Binary Data (Section 4), and End (Section 5).

use std::path::Path;

use crate::grib1::grid::{self, LatLon};
use crate::grib1::tables;
use crate::grib1::unpack;
use crate::GribError;

// ---------------------------------------------------------------------------
// Section structures
// ---------------------------------------------------------------------------

/// Section 0: Indicator Section (8 bytes).
#[derive(Debug, Clone)]
pub struct IndicatorSection {
    /// Total length of the GRIB message in bytes.
    pub total_length: u32,
    /// GRIB edition number (always 1).
    pub edition: u8,
}

/// Section 1: Product Definition Section (PDS).
#[derive(Debug, Clone)]
pub struct ProductDefinitionSection {
    /// Section length in bytes.
    pub section_length: u32,
    /// Parameter table version number.
    pub table_version: u8,
    /// Identification of originating center.
    pub center_id: u8,
    /// Generating process identification number.
    pub process_id: u8,
    /// Grid identification number.
    pub grid_id: u8,
    /// True if a Grid Description Section (GDS) is present.
    pub gds_present: bool,
    /// True if a Bit Map Section (BMS) is present.
    pub bms_present: bool,
    /// Parameter indicator (WMO Table 2 code).
    pub parameter: u8,
    /// Level type indicator (WMO Table 3 code).
    pub level_type: u8,
    /// Level value (or top of layer for layer types).
    pub level_value: u16,
    /// Level bottom (byte 12); for single-level types this equals the full value.
    pub level_top: u8,
    /// Level bottom byte; for layer types this is the bottom of the layer.
    pub level_bottom: u8,
    /// Year of century (byte 13).
    pub year_of_century: u8,
    /// Month (byte 14).
    pub month: u8,
    /// Day (byte 15).
    pub day: u8,
    /// Hour (byte 16).
    pub hour: u8,
    /// Minute (byte 17).
    pub minute: u8,
    /// Forecast time unit indicator (WMO Table 4).
    pub time_unit: u8,
    /// Time range value P1.
    pub p1: u8,
    /// Time range value P2.
    pub p2: u8,
    /// Time range indicator (WMO Table 5).
    pub time_range_indicator: u8,
    /// Number of values included in average.
    pub num_in_average: u16,
    /// Number missing from averages.
    pub num_missing: u8,
    /// Century (byte 25). Full year = (century - 1) * 100 + year_of_century.
    pub century: u8,
    /// Sub-center identification.
    pub sub_center: u8,
    /// Decimal scale factor D (signed).
    pub decimal_scale: i16,
    /// Raw PDS bytes for access to extended/local-use fields.
    pub raw: Vec<u8>,
}

impl ProductDefinitionSection {
    /// Returns the full 4-digit year.
    pub fn year(&self) -> i32 {
        (self.century as i32 - 1) * 100 + self.year_of_century as i32
    }

    /// Returns the parameter name from WMO Table 2, if known.
    pub fn parameter_name(&self) -> Option<&'static str> {
        tables::parameter_name(self.parameter)
    }

    /// Returns the parameter units from WMO Table 2, if known.
    pub fn parameter_units(&self) -> Option<&'static str> {
        tables::parameter_units(self.parameter)
    }

    /// Returns the parameter abbreviation from WMO Table 2, if known.
    pub fn parameter_abbrev(&self) -> Option<&'static str> {
        tables::parameter_abbrev(self.parameter)
    }

    /// Returns a human-readable level description.
    pub fn level_description(&self) -> (&'static str, &'static str) {
        tables::level_description(self.level_type)
    }
}

/// The type of grid described by a GDS.
#[derive(Debug, Clone)]
pub enum GridType {
    /// Regular latitude/longitude grid (type 0).
    LatLon {
        ni: u16,
        nj: u16,
        la1: f64,
        lo1: f64,
        la2: f64,
        lo2: f64,
        di: f64,
        dj: f64,
        scanning_mode: u8,
    },
    /// Gaussian latitude/longitude grid (type 4).
    Gaussian {
        ni: u16,
        nj: u16,
        la1: f64,
        lo1: f64,
        la2: f64,
        lo2: f64,
        di: f64,
        /// Number of parallels between pole and equator.
        n: u16,
        scanning_mode: u8,
    },
    /// Lambert conformal conic projection (type 3).
    LambertConformal {
        nx: u16,
        ny: u16,
        la1: f64,
        lo1: f64,
        resolution_flags: u8,
        lov: f64,
        dx: f64,
        dy: f64,
        projection_center: u8,
        scanning_mode: u8,
        latin1: f64,
        latin2: f64,
        lat_sp: f64,
        lon_sp: f64,
    },
    /// Polar stereographic projection (type 5).
    PolarStereographic {
        nx: u16,
        ny: u16,
        la1: f64,
        lo1: f64,
        resolution_flags: u8,
        lov: f64,
        dx: f64,
        dy: f64,
        projection_center: u8,
        scanning_mode: u8,
    },
    /// Unrecognized grid type.
    Unknown(u8),
}

/// The level type parsed from PDS bytes 10-12.
#[derive(Debug, Clone)]
pub enum LevelType {
    /// A single level (e.g., surface, tropopause, specific isobaric surface).
    Single { type_code: u8, value: u16 },
    /// A layer between two levels (e.g., between two pressure surfaces).
    Layer { type_code: u8, top: u8, bottom: u8 },
}

/// Section 2: Grid Description Section (GDS).
#[derive(Debug, Clone)]
pub struct GridDescriptionSection {
    /// Section length in bytes.
    pub section_length: u32,
    /// Number of vertical coordinate parameters.
    pub nv: u8,
    /// Location (byte number) of the list of vertical coordinate parameters,
    /// or 255 if not used.
    pub pv_location: u8,
    /// Data representation type code (0=lat/lon, 3=lambert, 4=gaussian, 5=polar stereo).
    pub data_representation_type: u8,
    /// Parsed grid definition.
    pub grid_type: GridType,
    /// Raw GDS bytes.
    pub raw: Vec<u8>,
}

impl GridDescriptionSection {
    /// Returns the total number of grid points (ni * nj or nx * ny).
    pub fn num_points(&self) -> usize {
        match &self.grid_type {
            GridType::LatLon { ni, nj, .. } => (*ni as usize) * (*nj as usize),
            GridType::Gaussian { ni, nj, .. } => (*ni as usize) * (*nj as usize),
            GridType::LambertConformal { nx, ny, .. } => (*nx as usize) * (*ny as usize),
            GridType::PolarStereographic { nx, ny, .. } => (*nx as usize) * (*ny as usize),
            GridType::Unknown(_) => 0,
        }
    }
}

/// Section 3: Bit Map Section (BMS).
#[derive(Debug, Clone)]
pub struct BitMapSection {
    /// Section length in bytes.
    pub section_length: u32,
    /// Number of unused bits at end of bitmap.
    pub unused_bits: u8,
    /// Bitmap table reference. 0 means bitmap follows in this section.
    pub table_reference: u16,
    /// The bitmap data (only present when table_reference == 0).
    pub bitmap: Vec<u8>,
}

/// Section 4: Binary Data Section (BDS).
#[derive(Debug, Clone)]
pub struct BinaryDataSection {
    /// Section length in bytes.
    pub section_length: u32,
    /// Flag byte (packing type, data type, etc.).
    pub flags: u8,
    /// Binary scale factor E.
    pub binary_scale: i16,
    /// Reference value R (converted from IBM float to IEEE f64).
    pub reference_value: f64,
    /// Number of bits per packed datum.
    pub bits_per_value: u8,
    /// Raw section bytes (including header) for unpacking.
    pub raw: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Single GRIB1 message
// ---------------------------------------------------------------------------

/// A single GRIB Edition 1 message containing all parsed sections.
#[derive(Debug, Clone)]
pub struct Grib1Message {
    /// Byte offset of this message within the file/buffer.
    pub offset: usize,
    /// Section 0: Indicator.
    pub indicator: IndicatorSection,
    /// Section 1: Product Definition.
    pub pds: ProductDefinitionSection,
    /// Section 2: Grid Description (present only if PDS flag indicates).
    pub gds: Option<GridDescriptionSection>,
    /// Section 3: Bit Map (present only if PDS flag indicates).
    pub bms: Option<BitMapSection>,
    /// Section 4: Binary Data.
    pub bds: BinaryDataSection,
}

impl Grib1Message {
    /// Unpack the data values from this message.
    ///
    /// Applies binary and decimal scaling. If a bitmap is present, missing
    /// grid points are filled with `f64::NAN`.
    pub fn values(&self) -> Result<Vec<f64>, GribError> {
        let num_points = self.num_data_points();

        // Number of packed values depends on whether a bitmap is present
        let num_packed = if let Some(ref bms) = self.bms {
            // Count the number of "1" bits in the bitmap
            let mut count = 0usize;
            let total = num_points;
            for i in 0..total {
                let byte_idx = i / 8;
                let bit_idx = 7 - (i % 8);
                if byte_idx < bms.bitmap.len() && (bms.bitmap[byte_idx] >> bit_idx) & 1 == 1 {
                    count += 1;
                }
            }
            count
        } else {
            num_points
        };

        let raw_values = unpack::unpack_bds(&self.bds.raw, self.pds.decimal_scale, num_packed)?;

        if let Some(ref bms) = self.bms {
            unpack::apply_bitmap(&raw_values, &bms.bitmap, num_points)
        } else {
            Ok(raw_values)
        }
    }

    /// Generate latitude/longitude coordinates for each grid point.
    ///
    /// Requires the GDS to be present in the message.
    pub fn latlons(&self) -> Result<Vec<LatLon>, GribError> {
        match &self.gds {
            Some(gds) => grid::grid_coordinates(gds),
            None => Err(GribError::Parse(
                "Cannot generate coordinates: no Grid Description Section (GDS) present".into(),
            )),
        }
    }

    /// Generate latitude values for each grid point.
    pub fn lats(&self) -> Result<Vec<f64>, GribError> {
        Ok(self.latlons()?.iter().map(|c| c.lat).collect())
    }

    /// Generate longitude values for each grid point.
    pub fn lons(&self) -> Result<Vec<f64>, GribError> {
        Ok(self.latlons()?.iter().map(|c| c.lon).collect())
    }

    /// Returns the total number of grid points in this message.
    pub fn num_data_points(&self) -> usize {
        if let Some(ref gds) = self.gds {
            gds.num_points()
        } else {
            // Without a GDS, estimate from the BDS packed data size
            if self.bds.bits_per_value == 0 {
                0
            } else {
                let data_bytes = self.bds.section_length as usize - 11;
                (data_bytes * 8) / (self.bds.bits_per_value as usize)
            }
        }
    }

    /// Returns the parameter name, if known.
    pub fn parameter_name(&self) -> Option<&'static str> {
        self.pds.parameter_name()
    }

    /// Returns the parameter units, if known.
    pub fn parameter_units(&self) -> Option<&'static str> {
        self.pds.parameter_units()
    }

    /// Returns the parameter abbreviation, if known.
    pub fn parameter_abbrev(&self) -> Option<&'static str> {
        self.pds.parameter_abbrev()
    }

    /// Returns the level type information.
    pub fn level(&self) -> LevelType {
        // Layer types have distinct top and bottom values in bytes 11-12.
        // Single-level types use the combined 16-bit value.
        match self.pds.level_type {
            101 | 104 | 106 | 108 | 110 | 112 | 114 | 116 | 120 | 121 | 128 | 141 => {
                LevelType::Layer {
                    type_code: self.pds.level_type,
                    top: self.pds.level_top,
                    bottom: self.pds.level_bottom,
                }
            }
            _ => LevelType::Single {
                type_code: self.pds.level_type,
                value: self.pds.level_value,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// GRIB1 file
// ---------------------------------------------------------------------------

/// A collection of GRIB Edition 1 messages parsed from a file or byte buffer.
#[derive(Debug)]
pub struct Grib1File {
    /// All messages found in the file.
    pub messages: Vec<Grib1Message>,
}

impl Grib1File {
    /// Open and parse all GRIB1 messages from a file on disk.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GribError> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data)
    }

    /// Parse all GRIB1 messages from an in-memory byte buffer.
    ///
    /// Scans for the "GRIB" magic bytes and parses each complete message.
    /// Non-GRIB data between messages is skipped.
    pub fn from_bytes(data: &[u8]) -> Result<Self, GribError> {
        let mut messages = Vec::new();
        let mut pos = 0;

        while pos + 8 <= data.len() {
            // Scan for "GRIB" magic
            match find_grib_magic(data, pos) {
                Some(start) => {
                    match parse_message(data, start) {
                        Ok((msg, consumed)) => {
                            messages.push(msg);
                            pos = start + consumed;
                        }
                        Err(e) => {
                            // If parsing fails, skip past this "GRIB" and keep scanning
                            eprintln!(
                                "Warning: failed to parse GRIB1 message at offset {}: {}",
                                start, e
                            );
                            pos = start + 4;
                        }
                    }
                }
                None => break,
            }
        }

        if messages.is_empty() {
            return Err(GribError::Parse(
                "No valid GRIB1 messages found in input".into(),
            ));
        }

        Ok(Grib1File { messages })
    }

    /// Returns the number of messages in the file.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Returns true if the file contains no messages.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Returns an iterator over the messages.
    pub fn iter(&self) -> std::slice::Iter<'_, Grib1Message> {
        self.messages.iter()
    }
}

impl<'a> IntoIterator for &'a Grib1File {
    type Item = &'a Grib1Message;
    type IntoIter = std::slice::Iter<'a, Grib1Message>;

    fn into_iter(self) -> Self::IntoIter {
        self.messages.iter()
    }
}

// ---------------------------------------------------------------------------
// Parsing implementation
// ---------------------------------------------------------------------------

/// Search for the "GRIB" magic bytes starting from `pos`.
fn find_grib_magic(data: &[u8], pos: usize) -> Option<usize> {
    let magic = b"GRIB";
    if data.len() < 4 {
        return None;
    }
    let end = data.len() - 3;
    for i in pos..end {
        if &data[i..i + 4] == magic {
            return Some(i);
        }
    }
    None
}

/// Read a 3-byte big-endian unsigned integer.
#[inline]
fn read_u24(data: &[u8], offset: usize) -> crate::Result<u32> {
    if offset + 3 > data.len() {
        return Err(crate::GribError::Parse(format!(
            "read_u24: offset {} + 3 > data length {}",
            offset,
            data.len()
        )));
    }
    Ok(
        ((data[offset] as u32) << 16)
            | ((data[offset + 1] as u32) << 8)
            | (data[offset + 2] as u32),
    )
}

/// Read a 2-byte big-endian unsigned integer.
#[inline]
fn read_u16(data: &[u8], offset: usize) -> crate::Result<u16> {
    if offset + 2 > data.len() {
        return Err(crate::GribError::Parse(format!(
            "read_u16: offset {} + 2 > data length {}",
            offset,
            data.len()
        )));
    }
    Ok(((data[offset] as u16) << 8) | (data[offset + 1] as u16))
}

/// Read a 3-byte signed integer (sign-magnitude, GRIB1 convention).
/// Bit 23 is sign, bits 22-0 are magnitude.
#[inline]
fn read_signed_24(data: &[u8], offset: usize) -> crate::Result<i32> {
    let raw = read_u24(data, offset)?;
    let magnitude = (raw & 0x7F_FFFF) as i32;
    if raw & 0x80_0000 != 0 {
        Ok(-magnitude)
    } else {
        Ok(magnitude)
    }
}

/// Read a 2-byte signed integer (sign-magnitude, GRIB1 convention).
#[inline]
fn read_signed_16(data: &[u8], offset: usize) -> crate::Result<i16> {
    let raw = read_u16(data, offset)?;
    let magnitude = (raw & 0x7FFF) as i16;
    if raw & 0x8000 != 0 {
        Ok(-magnitude)
    } else {
        Ok(magnitude)
    }
}

/// Parse a complete GRIB1 message starting at `offset`.
/// Returns `(message, bytes_consumed)`.
fn parse_message(data: &[u8], offset: usize) -> Result<(Grib1Message, usize), GribError> {
    let base = offset;

    // --- Section 0: Indicator ---
    if data.len() < base + 8 {
        return Err(GribError::Parse(
            "Data too short for GRIB1 indicator section".into(),
        ));
    }

    if &data[base..base + 4] != b"GRIB" {
        return Err(GribError::Parse("Missing GRIB magic bytes".into()));
    }

    let total_length = read_u24(data, base + 4)?;
    let edition = data[base + 7];

    if edition != 1 {
        return Err(GribError::Parse(format!(
            "Expected GRIB edition 1, found edition {}",
            edition
        )));
    }

    let indicator = IndicatorSection {
        total_length,
        edition,
    };

    let msg_end = base
        .checked_add(total_length as usize)
        .ok_or_else(|| GribError::Parse("GRIB1 message length overflows usize".into()))?;
    if msg_end > data.len() {
        return Err(GribError::Parse(format!(
            "GRIB1 message claims {} bytes but only {} available",
            total_length,
            data.len() - base
        )));
    }

    // Restrict every section parser to this message. A corrupt section length
    // must not borrow template bytes from a following GRIB message.
    let message_data = &data[..msg_end];
    let mut pos = base + 8;

    // --- Section 1: Product Definition Section ---
    let pds = parse_pds(message_data, pos)?;
    pos += pds.section_length as usize;

    // --- Section 2: Grid Description Section (optional) ---
    let gds = if pds.gds_present {
        let g = parse_gds(message_data, pos)?;
        pos += g.section_length as usize;
        Some(g)
    } else {
        None
    };

    // --- Section 3: Bit Map Section (optional) ---
    let bms = if pds.bms_present {
        let b = parse_bms(message_data, pos)?;
        pos += b.section_length as usize;
        Some(b)
    } else {
        None
    };

    // --- Section 4: Binary Data Section ---
    let bds = parse_bds(message_data, pos)?;
    pos += bds.section_length as usize;

    // --- Section 5: End Section ---
    if pos.checked_add(4) != Some(msg_end) {
        return Err(GribError::Parse(format!(
            "GRIB1 section lengths end at byte {pos}, but the message end marker must start at byte {}",
            msg_end.saturating_sub(4)
        )));
    }
    if &message_data[pos..msg_end] != b"7777" {
        return Err(GribError::Parse(format!(
            "Missing GRIB1 end marker '7777' at offset {pos}"
        )));
    }

    let consumed = total_length as usize;

    Ok((
        Grib1Message {
            offset: base,
            indicator,
            pds,
            gds,
            bms,
            bds,
        },
        consumed,
    ))
}

/// Parse Section 1 (PDS) starting at `pos`.
fn parse_pds(data: &[u8], pos: usize) -> Result<ProductDefinitionSection, GribError> {
    if pos.checked_add(3).map_or(true, |end| end > data.len()) {
        return Err(GribError::Parse(
            "Data too short for GRIB1 Product Definition Section header".into(),
        ));
    }

    let section_length = read_u24(data, pos)?;
    if section_length < 28 {
        return Err(GribError::Parse(format!(
            "GRIB1 Product Definition Section is {section_length} bytes; need at least 28"
        )));
    }
    let end = pos
        .checked_add(section_length as usize)
        .ok_or_else(|| GribError::Parse("PDS section length overflows usize".into()))?;
    if end > data.len() {
        return Err(GribError::Parse(format!(
            "PDS section length {section_length} exceeds the current GRIB1 message"
        )));
    }
    let raw = data[pos..end].to_vec();

    let flag_byte = data[pos + 7];
    let gds_present = flag_byte & 0x80 != 0;
    let bms_present = flag_byte & 0x40 != 0;

    let level_top = data[pos + 10];
    let level_bottom = data[pos + 11];
    let level_value = read_u16(data, pos + 10)?;

    let decimal_scale = if section_length >= 28 {
        read_signed_16(data, pos + 26)?
    } else {
        0
    };

    Ok(ProductDefinitionSection {
        section_length,
        table_version: data[pos + 3],
        center_id: data[pos + 4],
        process_id: data[pos + 5],
        grid_id: data[pos + 6],
        gds_present,
        bms_present,
        parameter: data[pos + 8],
        level_type: data[pos + 9],
        level_value,
        level_top,
        level_bottom,
        year_of_century: data[pos + 12],
        month: data[pos + 13],
        day: data[pos + 14],
        hour: data[pos + 15],
        minute: data[pos + 16],
        time_unit: data[pos + 17],
        p1: data[pos + 18],
        p2: data[pos + 19],
        time_range_indicator: data[pos + 20],
        num_in_average: read_u16(data, pos + 21)?,
        num_missing: data[pos + 23],
        century: if section_length >= 25 {
            data[pos + 24]
        } else {
            20
        },
        sub_center: if section_length >= 26 {
            data[pos + 25]
        } else {
            0
        },
        decimal_scale,
        raw,
    })
}

/// Parse Section 2 (GDS) starting at `pos`.
fn parse_gds(data: &[u8], pos: usize) -> Result<GridDescriptionSection, GribError> {
    if pos.checked_add(6).map_or(true, |end| end > data.len()) {
        return Err(GribError::Parse(
            "Data too short for GRIB1 Grid Description Section header".into(),
        ));
    }

    let section_length = read_u24(data, pos)?;
    if section_length < 6 {
        return Err(GribError::Parse(format!(
            "GDS section is {section_length} bytes; need at least 6"
        )));
    }
    let end = pos
        .checked_add(section_length as usize)
        .ok_or_else(|| GribError::Parse("GDS section length overflows usize".into()))?;
    if end > data.len() {
        return Err(GribError::Parse(format!(
            "GDS section length {} exceeds the current GRIB1 message",
            section_length
        )));
    }

    let raw = data[pos..end].to_vec();
    let nv = data[pos + 3];
    let pv_location = data[pos + 4];
    let data_rep_type = data[pos + 5];

    // Parse templates only from their declared section bytes. Previously a
    // short grid-0/4 GDS could read its final fields out of the following BDS.
    let grid_type = match data_rep_type {
        0 => parse_gds_latlon(&raw, 0)?,
        4 => parse_gds_gaussian(&raw, 0)?,
        3 => parse_gds_lambert(&raw, 0)?,
        5 => parse_gds_polar_stereo(&raw, 0)?,
        other => GridType::Unknown(other),
    };

    Ok(GridDescriptionSection {
        section_length,
        nv,
        pv_location,
        data_representation_type: data_rep_type,
        grid_type,
        raw,
    })
}

/// Parse lat/lon grid definition (GDS type 0) from bytes starting at `pos`.
fn parse_gds_latlon(data: &[u8], pos: usize) -> Result<GridType, GribError> {
    if pos.checked_add(32).map_or(true, |end| end > data.len()) {
        return Err(GribError::Parse(
            "Data too short for lat/lon GDS template (need at least 32 section bytes)".into(),
        ));
    }

    let ni = read_u16(data, pos + 6)?;
    let nj = read_u16(data, pos + 8)?;
    let la1 = read_signed_24(data, pos + 10)? as f64 / 1000.0;
    let lo1 = read_signed_24(data, pos + 13)? as f64 / 1000.0;
    // byte 17 (pos+16) is resolution flags, skip
    let la2 = read_signed_24(data, pos + 17)? as f64 / 1000.0;
    let lo2 = read_signed_24(data, pos + 20)? as f64 / 1000.0;
    let di_raw = read_u16(data, pos + 23)?;
    let dj_raw = read_u16(data, pos + 25)?;
    let di = if di_raw == u16::MAX {
        f64::NAN
    } else {
        f64::from(di_raw) / 1000.0
    };
    let dj = if dj_raw == u16::MAX {
        f64::NAN
    } else {
        f64::from(dj_raw) / 1000.0
    };
    let scanning_mode = data[pos + 27];

    Ok(GridType::LatLon {
        ni,
        nj,
        la1,
        lo1,
        la2,
        lo2,
        di,
        dj,
        scanning_mode,
    })
}

/// Parse Gaussian grid definition (GDS type 4) from bytes starting at `pos`.
fn parse_gds_gaussian(data: &[u8], pos: usize) -> Result<GridType, GribError> {
    if pos.checked_add(32).map_or(true, |end| end > data.len()) {
        return Err(GribError::Parse(
            "Data too short for Gaussian GDS template (need at least 32 section bytes)".into(),
        ));
    }

    let ni = read_u16(data, pos + 6)?;
    let nj = read_u16(data, pos + 8)?;
    let la1 = read_signed_24(data, pos + 10)? as f64 / 1000.0;
    let lo1 = read_signed_24(data, pos + 13)? as f64 / 1000.0;
    let la2 = read_signed_24(data, pos + 17)? as f64 / 1000.0;
    let lo2 = read_signed_24(data, pos + 20)? as f64 / 1000.0;
    let di_raw = read_u16(data, pos + 23)?;
    let di = if di_raw == u16::MAX {
        f64::NAN
    } else {
        f64::from(di_raw) / 1000.0
    };
    let n = read_u16(data, pos + 25)?; // number of parallels between pole and equator
    if n == 0 {
        return Err(GribError::Parse(
            "Gaussian GDS has N=0; at least one parallel per hemisphere is required".into(),
        ));
    }
    let scanning_mode = data[pos + 27];

    Ok(GridType::Gaussian {
        ni,
        nj,
        la1,
        lo1,
        la2,
        lo2,
        di,
        n,
        scanning_mode,
    })
}

/// Parse Lambert conformal grid definition (GDS type 3) from bytes starting at `pos`.
fn parse_gds_lambert(data: &[u8], pos: usize) -> Result<GridType, GribError> {
    if data.len() < pos + 40 {
        return Err(GribError::Parse(
            "Data too short for Lambert conformal GDS (need 40 bytes)".into(),
        ));
    }

    let nx = read_u16(data, pos + 6)?;
    let ny = read_u16(data, pos + 8)?;
    let la1 = read_signed_24(data, pos + 10)? as f64 / 1000.0;
    let lo1 = read_signed_24(data, pos + 13)? as f64 / 1000.0;
    let resolution_flags = data[pos + 16];
    let lov = read_signed_24(data, pos + 17)? as f64 / 1000.0;
    let dx = read_signed_24(data, pos + 20)? as f64; // meters
    let dy = read_signed_24(data, pos + 23)? as f64; // meters
    let projection_center = data[pos + 26];
    let scanning_mode = data[pos + 27];
    let latin1 = read_signed_24(data, pos + 28)? as f64 / 1000.0;
    let latin2 = read_signed_24(data, pos + 31)? as f64 / 1000.0;
    let lat_sp = read_signed_24(data, pos + 34)? as f64 / 1000.0;
    let lon_sp = read_signed_24(data, pos + 37)? as f64 / 1000.0;

    Ok(GridType::LambertConformal {
        nx,
        ny,
        la1,
        lo1,
        resolution_flags,
        lov,
        dx,
        dy,
        projection_center,
        scanning_mode,
        latin1,
        latin2,
        lat_sp,
        lon_sp,
    })
}

/// Parse Polar Stereographic grid definition (GDS type 5) from bytes starting at `pos`.
fn parse_gds_polar_stereo(data: &[u8], pos: usize) -> Result<GridType, GribError> {
    if data.len() < pos + 28 {
        return Err(GribError::Parse(
            "Data too short for polar stereographic GDS (need 28 bytes)".into(),
        ));
    }

    let nx = read_u16(data, pos + 6)?;
    let ny = read_u16(data, pos + 8)?;
    let la1 = read_signed_24(data, pos + 10)? as f64 / 1000.0;
    let lo1 = read_signed_24(data, pos + 13)? as f64 / 1000.0;
    let resolution_flags = data[pos + 16];
    let lov = read_signed_24(data, pos + 17)? as f64 / 1000.0;
    let dx = read_signed_24(data, pos + 20)? as f64; // meters
    let dy = read_signed_24(data, pos + 23)? as f64; // meters
    let projection_center = data[pos + 26];
    let scanning_mode = data[pos + 27];

    Ok(GridType::PolarStereographic {
        nx,
        ny,
        la1,
        lo1,
        resolution_flags,
        lov,
        dx,
        dy,
        projection_center,
        scanning_mode,
    })
}

/// Parse Section 3 (BMS) starting at `pos`.
fn parse_bms(data: &[u8], pos: usize) -> Result<BitMapSection, GribError> {
    if pos.checked_add(6).map_or(true, |end| end > data.len()) {
        return Err(GribError::Parse(
            "Data too short for GRIB1 Bit Map Section header".into(),
        ));
    }

    let section_length = read_u24(data, pos)?;
    if section_length < 6 {
        return Err(GribError::Parse(format!(
            "BMS section is {section_length} bytes; need at least 6"
        )));
    }
    let end = pos
        .checked_add(section_length as usize)
        .ok_or_else(|| GribError::Parse("BMS section length overflows usize".into()))?;
    if end > data.len() {
        return Err(GribError::Parse(format!(
            "BMS section length {} exceeds the current GRIB1 message",
            section_length
        )));
    }

    let unused_bits = data[pos + 3];
    let table_reference = read_u16(data, pos + 4)?;

    let bitmap = if table_reference == 0 && section_length > 6 {
        data[pos + 6..end].to_vec()
    } else {
        Vec::new()
    };

    Ok(BitMapSection {
        section_length,
        unused_bits,
        table_reference,
        bitmap,
    })
}

/// Parse Section 4 (BDS) starting at `pos`.
fn parse_bds(data: &[u8], pos: usize) -> Result<BinaryDataSection, GribError> {
    if pos.checked_add(11).map_or(true, |end| end > data.len()) {
        return Err(GribError::Parse(
            "Data too short for GRIB1 Binary Data Section header".into(),
        ));
    }

    let section_length = read_u24(data, pos)?;
    if section_length < 11 {
        return Err(GribError::Parse(format!(
            "BDS section is {section_length} bytes; need at least 11"
        )));
    }
    let end = pos
        .checked_add(section_length as usize)
        .ok_or_else(|| GribError::Parse("BDS section length overflows usize".into()))?;
    if end > data.len() {
        return Err(GribError::Parse(format!(
            "BDS section length {} exceeds the current GRIB1 message",
            section_length
        )));
    }

    let flags = data[pos + 3];

    // Binary scale factor (sign-magnitude, 16-bit)
    let binary_scale = read_signed_16(data, pos + 4)?;

    // Reference value (IBM 32-bit float)
    let ibm_ref = ((data[pos + 6] as u32) << 24)
        | ((data[pos + 7] as u32) << 16)
        | ((data[pos + 8] as u32) << 8)
        | (data[pos + 9] as u32);
    let reference_value = unpack::ibm_to_ieee(ibm_ref);

    let bits_per_value = data[pos + 10];

    let raw = data[pos..end].to_vec();

    Ok(BinaryDataSection {
        section_length,
        flags,
        binary_scale,
        reference_value,
        bits_per_value,
        raw,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid GRIB1 message in memory for testing.
    fn build_test_message() -> Vec<u8> {
        let mut msg = Vec::new();

        // --- Section 0: Indicator (8 bytes) ---
        msg.extend_from_slice(b"GRIB");
        // Total length placeholder (filled in at end)
        msg.push(0);
        msg.push(0);
        msg.push(0);
        msg.push(1); // edition = 1

        let sec0_end = msg.len();

        // --- Section 1: PDS (28 bytes minimum) ---
        let pds_len: u32 = 28;
        msg.push((pds_len >> 16) as u8);
        msg.push((pds_len >> 8) as u8);
        msg.push(pds_len as u8);
        msg.push(2); // table version
        msg.push(7); // center = NCEP
        msg.push(0); // process
        msg.push(0); // grid id
        msg.push(0x80); // flags: GDS present, no BMS
        msg.push(11); // parameter = temperature
        msg.push(100); // level type = isobaric
        msg.push(0x01); // level value high byte (500 = 0x01F4)
        msg.push(0xF4); // level value low byte
        msg.push(24); // year of century
        msg.push(1); // month
        msg.push(15); // day
        msg.push(12); // hour
        msg.push(0); // minute
        msg.push(1); // time unit = hour
        msg.push(6); // P1
        msg.push(0); // P2
        msg.push(0); // time range indicator
        msg.push(0); // num in average high
        msg.push(0); // num in average low
        msg.push(0); // num missing
        msg.push(21); // century = 21
        msg.push(0); // sub-center
        msg.push(0); // decimal scale high (0)
        msg.push(0); // decimal scale low (0)

        // --- Section 2: GDS (32 bytes, lat/lon grid 2x2) ---
        let gds_len: u32 = 32;
        msg.push((gds_len >> 16) as u8);
        msg.push((gds_len >> 8) as u8);
        msg.push(gds_len as u8);
        msg.push(0); // NV
        msg.push(255); // PV location (not used)
        msg.push(0); // data rep type = lat/lon

        // Ni = 2
        msg.push(0);
        msg.push(2);
        // Nj = 2
        msg.push(0);
        msg.push(2);
        // La1 = 45.000 degrees = 45000 millidegrees
        let la1: i32 = 45000;
        msg.push(((la1 >> 16) & 0xFF) as u8);
        msg.push(((la1 >> 8) & 0xFF) as u8);
        msg.push((la1 & 0xFF) as u8);
        // Lo1 = -90.000 = -90000 millidegrees => sign-magnitude: 0x80_0000 | 90000
        let lo1_raw: u32 = 0x80_0000 | 90000;
        msg.push(((lo1_raw >> 16) & 0xFF) as u8);
        msg.push(((lo1_raw >> 8) & 0xFF) as u8);
        msg.push((lo1_raw & 0xFF) as u8);
        // Resolution flags
        msg.push(0x80); // direction increments given
                        // La2 = 40.000 = 40000
        let la2: i32 = 40000;
        msg.push(((la2 >> 16) & 0xFF) as u8);
        msg.push(((la2 >> 8) & 0xFF) as u8);
        msg.push((la2 & 0xFF) as u8);
        // Lo2 = -85.000 = -85000 => sign-magnitude
        let lo2_raw: u32 = 0x80_0000 | 85000;
        msg.push(((lo2_raw >> 16) & 0xFF) as u8);
        msg.push(((lo2_raw >> 8) & 0xFF) as u8);
        msg.push((lo2_raw & 0xFF) as u8);
        // Di = 5.000 degrees = 5000 millidegrees
        msg.push((5000 >> 8) as u8);
        msg.push((5000 & 0xFF) as u8);
        // Dj = 5.000 degrees = 5000 millidegrees
        msg.push((5000 >> 8) as u8);
        msg.push((5000 & 0xFF) as u8);
        // Scanning mode = 0 (east, north-to-south, i-consecutive)
        msg.push(0x00);
        // Pad GDS to 32 bytes
        while msg.len() < sec0_end + pds_len as usize + gds_len as usize {
            msg.push(0);
        }

        // --- Section 4: BDS ---
        // 4 data points, 8 bits each, reference=0 (IBM), binary scale=0
        let num_values = 4u32;
        let bits = 8u8;
        let data_bytes = (num_values * bits as u32 + 7) / 8;
        let bds_len = 11 + data_bytes;
        msg.push((bds_len >> 16) as u8);
        msg.push((bds_len >> 8) as u8);
        msg.push(bds_len as u8);
        msg.push(0); // flags
        msg.push(0); // binary scale high
        msg.push(0); // binary scale low
                     // Reference value = 270.0 in IBM float
                     // 270 = 16^2 * (270/256) = 16^2 * 1.0546875
                     // mantissa = 270/256 * 2^24 = 270 * 65536 = 17694720 = 0x10E0000
                     // Wait, let's compute properly:
                     // 270.0: find exponent e such that 270 / 16^(e-64) is in [1/16, 1)
                     // 16^2 = 256, 270/256 = 1.0546875... not in [1/16, 1)
                     // Actually IBM mantissa f is 0.f where f is fraction, so value = 0.f * 16^e
                     // 270 = 0.f * 16^e => we need 16^e > 270
                     // 16^3 = 4096, so e=67 (67-64=3): 270/4096 = 0.06591796875
                     // mantissa = 0.06591796875 * 2^24 = 1105920 = 0x10E000
                     // IBM word: sign=0, exp=67=0x43, mantissa=0x10E000
                     // => 0x4310E000
        let ibm_270: u32 = 0x4310_E000;
        msg.push((ibm_270 >> 24) as u8);
        msg.push((ibm_270 >> 16) as u8);
        msg.push((ibm_270 >> 8) as u8);
        msg.push(ibm_270 as u8);
        msg.push(bits); // bits per value
                        // Data: 4 values = 0, 1, 2, 3 => final values = 270+0, 270+1, 270+2, 270+3
        msg.push(0);
        msg.push(1);
        msg.push(2);
        msg.push(3);

        // --- Section 5: End ---
        msg.extend_from_slice(b"7777");

        // Fix total length in Section 0
        let total_len = msg.len() as u32;
        msg[4] = (total_len >> 16) as u8;
        msg[5] = (total_len >> 8) as u8;
        msg[6] = total_len as u8;

        msg
    }

    #[test]
    fn test_parse_synthetic_message() {
        let data = build_test_message();
        let file = Grib1File::from_bytes(&data).expect("Failed to parse test GRIB1 message");

        assert_eq!(file.len(), 1);
        let msg = &file.messages[0];

        // Check indicator
        assert_eq!(msg.indicator.edition, 1);

        // Check PDS
        assert_eq!(msg.pds.parameter, 11); // temperature
        assert_eq!(msg.pds.level_type, 100); // isobaric
        assert_eq!(msg.pds.level_value, 500);
        assert_eq!(msg.pds.center_id, 7); // NCEP
        assert_eq!(msg.pds.year(), 2024);
        assert_eq!(msg.pds.parameter_name(), Some("Temperature"));

        // Check GDS
        assert!(msg.gds.is_some());
        let gds = msg.gds.as_ref().unwrap();
        assert_eq!(gds.data_representation_type, 0);
        assert_eq!(gds.num_points(), 4);

        // Check data values
        let values = msg.values().expect("Failed to unpack values");
        assert_eq!(values.len(), 4);
        // Values should be 270 + {0, 1, 2, 3} = {270, 271, 272, 273}
        for (i, &v) in values.iter().enumerate() {
            let expected = 270.0 + i as f64;
            assert!(
                (v - expected).abs() < 0.01,
                "Value {} expected {}, got {}",
                i,
                expected,
                v
            );
        }
    }

    #[test]
    fn test_grid_zero_and_four_require_section_local_32_byte_templates() {
        const GDS_START: usize = 8 + 28;
        for grid_type in [0_u8, 4] {
            let mut data = build_test_message();
            data[GDS_START..GDS_START + 3].copy_from_slice(&[0, 0, 31]);
            data[GDS_START + 5] = grid_type;
            let error = parse_message(&data, 0).expect_err("short GDS must not read into BDS");
            assert!(
                error.to_string().contains("at least 32 section bytes"),
                "grid {grid_type}: {error}"
            );
        }
    }

    #[test]
    fn test_missing_di_is_preserved_and_derived_from_endpoints() {
        const GDS_START: usize = 8 + 28;
        let mut data = build_test_message();
        data[GDS_START + 23..GDS_START + 25].copy_from_slice(&u16::MAX.to_be_bytes());
        let (message, _) = parse_message(&data, 0).expect("valid missing-Di message");
        match &message.gds.as_ref().expect("GDS").grid_type {
            GridType::LatLon { di, .. } => assert!(di.is_nan()),
            other => panic!("expected lat/lon grid, got {other:?}"),
        }
        let coordinates = message.latlons().expect("derive Di from Lo1/Lo2");
        assert!((coordinates[0].lon + 90.0).abs() < 1e-9);
        assert!((coordinates[1].lon + 85.0).abs() < 1e-9);
    }

    #[test]
    fn test_gaussian_n_zero_is_rejected_during_parse() {
        const GDS_START: usize = 8 + 28;
        let mut data = build_test_message();
        data[GDS_START + 5] = 4;
        data[GDS_START + 25..GDS_START + 27].copy_from_slice(&0_u16.to_be_bytes());
        let error = parse_message(&data, 0).expect_err("N=0 Gaussian grid must fail closed");
        assert!(error.to_string().contains("N=0"), "{error}");
    }

    #[test]
    fn test_parse_latlons() {
        let data = build_test_message();
        let file = Grib1File::from_bytes(&data).unwrap();
        let msg = &file.messages[0];

        let coords = msg.latlons().expect("Failed to generate coordinates");
        assert_eq!(coords.len(), 4);

        // First point: lat=45, lon=-90
        assert!((coords[0].lat - 45.0).abs() < 0.01);
        assert!((coords[0].lon - (-90.0)).abs() < 0.01);
    }

    #[test]
    fn test_pds_year_calculation() {
        let pds = ProductDefinitionSection {
            section_length: 28,
            table_version: 2,
            center_id: 7,
            process_id: 0,
            grid_id: 0,
            gds_present: false,
            bms_present: false,
            parameter: 11,
            level_type: 100,
            level_value: 500,
            level_top: 1,
            level_bottom: 244,
            year_of_century: 24,
            month: 1,
            day: 15,
            hour: 12,
            minute: 0,
            time_unit: 1,
            p1: 0,
            p2: 0,
            time_range_indicator: 0,
            num_in_average: 0,
            num_missing: 0,
            century: 21,
            sub_center: 0,
            decimal_scale: 0,
            raw: vec![],
        };
        assert_eq!(pds.year(), 2024);
    }

    #[test]
    fn test_level_type_single() {
        let data = build_test_message();
        let file = Grib1File::from_bytes(&data).unwrap();
        let msg = &file.messages[0];

        match msg.level() {
            LevelType::Single { type_code, value } => {
                assert_eq!(type_code, 100);
                assert_eq!(value, 500);
            }
            _ => panic!("Expected single level"),
        }
    }

    #[test]
    fn test_no_grib_magic() {
        let data = vec![0u8; 100];
        let result = Grib1File::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_edition() {
        let mut data = build_test_message();
        data[7] = 2; // change edition to 2
        let result = Grib1File::from_bytes(&data);
        assert!(result.is_err());
    }
}
