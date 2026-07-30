//! NEXRAD Level-III (NIDS) product parser.
//!
//! Parses the standard NEXRAD Level 3 product format including the message
//! header, product description block, and product symbology block.  Handles
//! the most common radial data packet (type 16 / AF1F) used by reflectivity,
//! velocity, and similar products.

use chrono::Datelike;
use std::io::Read;

// ── Public types ────────────────────────────────────────────────────────

/// Parsed NEXRAD Level 3 product file.
#[derive(Debug, Clone)]
pub struct Level3File {
    /// Numeric product code (e.g. 94 = base reflectivity 0.5 deg).
    pub product_code: u16,
    /// Source (radar) ID.
    pub source_id: u16,
    /// Latitude of the radar (degrees).
    pub latitude: f64,
    /// Longitude of the radar (degrees).
    pub longitude: f64,
    /// Height of the radar (feet).
    pub height: f64,
    /// Volume scan time as "YYYY-MM-DD HH:MM:SS".
    pub volume_time: String,
    /// Flattened data values (row-major: radial * bin).
    pub data: Vec<f64>,
    /// Number of range bins per radial.
    pub num_bins: usize,
    /// Number of radials in the product.
    pub num_radials: usize,
}

// ── Internal header structs ─────────────────────────────────────────────

/// 18-byte message header.
struct MsgHeader {
    code: u16,
    _date: u16,
    _time: u32,
    _length: u32,
    source: u16,
    _dest: u16,
    _num_blocks: u16,
}

/// Product Description Block (PDB) — fixed-size portion that follows the
/// message header divider.
struct ProductDesc {
    latitude: i32,
    longitude: i32,
    height: i16,
    product_code: u16,
    volume_date: u16,
    volume_time: u32,
    symbology_offset: u32,
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Read a big-endian u16 from a slice at `off`.
#[inline]
fn be_u16(data: &[u8], off: usize) -> u16 {
    u16::from_be_bytes([data[off], data[off + 1]])
}

/// Read a big-endian i16 from a slice at `off`.
#[inline]
fn be_i16(data: &[u8], off: usize) -> i16 {
    i16::from_be_bytes([data[off], data[off + 1]])
}

/// Read a big-endian u32 from a slice at `off`.
#[inline]
fn be_u32(data: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

/// Read a big-endian i32 from a slice at `off`.
#[inline]
fn be_i32(data: &[u8], off: usize) -> i32 {
    i32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

/// Convert a Modified-Julian date + seconds-since-midnight into a string.
fn volume_datetime(date: u16, time: u32) -> String {
    // NEXRAD date is Modified Julian days since 1970-01-01 (MJD offset = 1).
    let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let day = epoch + chrono::Duration::days(i64::from(date) - 1);
    let secs = time;
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{}-{:02}-{:02} {:02}:{:02}:{:02}",
        day.format("%Y"),
        day.month(),
        day.day(),
        h, m, s,
    )
}

// ── Parsing ─────────────────────────────────────────────────────────────

fn parse_msg_header(data: &[u8]) -> Result<MsgHeader, String> {
    if data.len() < 18 {
        return Err("Data too short for message header".into());
    }
    Ok(MsgHeader {
        code: be_u16(data, 0),
        _date: be_u16(data, 2),
        _time: be_u32(data, 4),
        _length: be_u32(data, 8),
        source: be_u16(data, 12),
        _dest: be_u16(data, 14),
        _num_blocks: be_u16(data, 16),
    })
}

fn parse_product_desc(data: &[u8], base: usize) -> Result<ProductDesc, String> {
    // The PDB starts with a divider (-1 i16 = 0xFFFF) then block-ID (1).
    // Total PDB is 102 bytes starting from the divider.
    let need = base + 102;
    if data.len() < need {
        return Err(format!(
            "Data too short for product description block (need {} bytes, have {})",
            need,
            data.len()
        ));
    }

    let off = base; // offset to divider

    // Divider check (optional, some files omit).
    // Fields within PDB (offsets relative to `off`):
    //   0  divider (i16)
    //   2  latitude  (i32, 1/1000 deg)
    //   6  longitude (i32, 1/1000 deg)
    //  10  height    (i16, feet)
    //  12  product code (i16)
    //  14  operational mode (i16)
    //  ...
    //  26  volume scan date  (u16)
    //  28  volume scan time  (u32)
    //  ...
    //  52  product-dependent (halfwords 27-30)
    //  60  symbology offset  (u32, halfwords from msg header start) @ offset 60
    let latitude = be_i32(data, off + 2);
    let longitude = be_i32(data, off + 6);
    let height = be_i16(data, off + 10);
    let product_code = be_u16(data, off + 12);
    let volume_date = be_u16(data, off + 26);
    let volume_time = be_u32(data, off + 28);
    let symbology_offset = be_u32(data, off + 60);

    Ok(ProductDesc {
        latitude,
        longitude,
        height,
        product_code,
        volume_date,
        volume_time,
        symbology_offset,
    })
}

/// Parse a radial data packet (packet type 16 / 0xAF1F).
///
/// Layout:
///   0  packet code  (u16 = 0xAF1F)
///   2  index first bin (u16)
///   4  num bins      (u16)
///   6  i-center      (i16)
///   8  j-center      (i16)
///  10  scale factor  (u16)
///  12  num radials   (u16)
///  14  start of radial data
///
/// Each radial:
///   0  num halfwords in RLE (u16)
///   2  start angle (u16, 0.1 deg)
///   4  delta angle (u16, 0.1 deg)
///   6  RLE data (num_halfwords * 2 bytes)
fn parse_radial_packet(data: &[u8], off: usize) -> Result<(Vec<f64>, usize, usize), String> {
    if data.len() < off + 14 {
        return Err("Data too short for radial packet header".into());
    }
    let packet_code = be_u16(data, off);
    if packet_code != 0xAF1F {
        return Err(format!("Unexpected radial packet code 0x{:04X}", packet_code));
    }

    let num_bins = be_u16(data, off + 4) as usize;
    let num_radials = be_u16(data, off + 12) as usize;

    let mut values: Vec<f64> = Vec::with_capacity(num_radials * num_bins);
    let mut cursor = off + 14;

    for _ in 0..num_radials {
        if data.len() < cursor + 6 {
            return Err("Data truncated in radial".into());
        }
        let num_halfwords = be_u16(data, cursor) as usize;
        // start_angle and delta_angle at cursor+2, cursor+4 — skip for now
        cursor += 6;

        let rle_bytes = num_halfwords * 2;
        if data.len() < cursor + rle_bytes {
            return Err("Data truncated in RLE run".into());
        }

        // Decode RLE: each byte is (run << 4 | value) for 8-bit products,
        // or the byte itself for non-RLE products.  We use a simple approach:
        // treat each byte as run=high-nibble, code=low-nibble.
        let mut bins_decoded = 0usize;
        for &byte in &data[cursor..cursor + rle_bytes] {
            let run = (byte >> 4) as usize;
            let code = (byte & 0x0F) as f64;
            // A run of 0 means 1 bin in many implementations; handle both.
            let count = if run == 0 { 1 } else { run };
            for _ in 0..count {
                if bins_decoded < num_bins {
                    values.push(code);
                    bins_decoded += 1;
                }
            }
        }
        // Pad if we decoded fewer than num_bins (e.g., short RLE).
        while bins_decoded < num_bins {
            values.push(0.0);
            bins_decoded += 1;
        }

        cursor += rle_bytes;
    }

    Ok((values, num_bins, num_radials))
}

/// Parse a raster data packet (packet type 0xBA0F or 0xBA07).
fn parse_raster_packet(data: &[u8], off: usize) -> Result<(Vec<f64>, usize, usize), String> {
    if data.len() < off + 14 {
        return Err("Data too short for raster packet header".into());
    }
    let _packet_code = be_u16(data, off);
    // Raster header: similar to radial but rows instead of radials.
    let num_cols = be_u16(data, off + 8) as usize;
    let num_rows = be_u16(data, off + 12) as usize;

    if num_cols == 0 || num_rows == 0 {
        return Ok((Vec::new(), 0, 0));
    }

    let mut values: Vec<f64> = Vec::with_capacity(num_rows * num_cols);
    let mut cursor = off + 14;

    for _ in 0..num_rows {
        if data.len() < cursor + 2 {
            return Err("Data truncated in raster row header".into());
        }
        let row_bytes = be_u16(data, cursor) as usize;
        cursor += 2;
        if data.len() < cursor + row_bytes {
            return Err("Data truncated in raster row".into());
        }

        let mut cols_decoded = 0usize;
        for &byte in &data[cursor..cursor + row_bytes] {
            let run = (byte >> 4) as usize;
            let code = (byte & 0x0F) as f64;
            let count = if run == 0 { 1 } else { run };
            for _ in 0..count {
                if cols_decoded < num_cols {
                    values.push(code);
                    cols_decoded += 1;
                }
            }
        }
        while cols_decoded < num_cols {
            values.push(0.0);
            cols_decoded += 1;
        }

        cursor += row_bytes;
    }

    Ok((values, num_cols, num_rows))
}

// ── Public API ──────────────────────────────────────────────────────────

impl Level3File {
    /// Parse a Level 3 product from a byte slice.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        // Strip any WMO header: look for a line starting with the product
        // code.  WMO headers are ASCII followed by \r\n; the binary product
        // starts with the message code (u16 > 0).  A quick heuristic: if the
        // first byte is printable ASCII (>= 0x20), skip until we find a
        // \r\n\r\n or \n\n boundary.
        let buf = strip_wmo_header(data);

        let hdr = parse_msg_header(buf)?;
        let pdb = parse_product_desc(buf, 18)?;

        let lat = f64::from(pdb.latitude) / 1000.0;
        let lon = f64::from(pdb.longitude) / 1000.0;
        let ht = f64::from(pdb.height);
        let vt = volume_datetime(pdb.volume_date, pdb.volume_time);

        // Resolve symbology block offset (in halfwords from message start).
        let (data_vals, num_bins, num_radials) = if pdb.symbology_offset > 0 {
            let sym_off = (pdb.symbology_offset as usize) * 2;
            parse_symbology(buf, sym_off)?
        } else {
            (Vec::new(), 0, 0)
        };

        Ok(Level3File {
            product_code: if pdb.product_code != 0 {
                pdb.product_code
            } else {
                hdr.code
            },
            source_id: hdr.source,
            latitude: lat,
            longitude: lon,
            height: ht,
            volume_time: vt,
            data: data_vals,
            num_bins,
            num_radials,
        })
    }

    /// Read and parse a Level 3 product from a file path.
    pub fn from_file(path: &std::path::Path) -> Result<Self, String> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        Self::from_bytes(&buf)
    }
}

/// Strip an optional WMO/AWIPS text header preceding the binary product.
fn strip_wmo_header(data: &[u8]) -> &[u8] {
    if data.is_empty() {
        return data;
    }
    // If the first byte looks like ASCII text (>= 0x20 and < 0x7F), try to
    // skip past the header.
    if data[0] >= 0x20 && data[0] < 0x7F {
        // Look for \r\n\r\n or \n\n
        for i in 0..data.len().saturating_sub(4) {
            if data[i] == b'\r' && data[i + 1] == b'\n' && data[i + 2] == b'\r' && data[i + 3] == b'\n' {
                return &data[i + 4..];
            }
            if data[i] == b'\n' && data[i + 1] == b'\n' {
                return &data[i + 2..];
            }
        }
    }
    data
}

/// Parse the product symbology block and extract data.
fn parse_symbology(data: &[u8], sym_off: usize) -> Result<(Vec<f64>, usize, usize), String> {
    // Symbology block header:
    //   0  divider   (i16, = -1)
    //   2  block ID  (u16, = 1)
    //   4  block length (u32)
    //   8  num layers (u16)
    //  10  layer divider (i16)
    //  12  layer length (u32)
    //  16  start of data packets
    if data.len() < sym_off + 16 {
        return Err("Data too short for symbology block".into());
    }

    let _divider = be_i16(data, sym_off);
    let _block_id = be_u16(data, sym_off + 2);
    let _block_len = be_u32(data, sym_off + 4);
    let _num_layers = be_u16(data, sym_off + 8);
    // layer header
    let _layer_div = be_i16(data, sym_off + 10);
    let _layer_len = be_u32(data, sym_off + 12);

    let pkt_off = sym_off + 16;
    if data.len() < pkt_off + 2 {
        return Err("No data packets in symbology block".into());
    }

    let packet_code = be_u16(data, pkt_off);
    match packet_code {
        0xAF1F => parse_radial_packet(data, pkt_off),
        0xBA0F | 0xBA07 => parse_raster_packet(data, pkt_off),
        _ => {
            // Unknown packet type — return empty data rather than error
            // so the caller still gets the metadata.
            Ok((Vec::new(), 0, 0))
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal Level 3 binary blob (msg header + PDB, no symbology).
    fn make_minimal_l3() -> Vec<u8> {
        let mut buf = vec![0u8; 18 + 102]; // msg header + PDB

        // Message header
        buf[0] = 0; buf[1] = 94; // product code 94
        buf[12] = 0; buf[13] = 25; // source id = 25

        // PDB (starts at offset 18)
        let pdb = 18;
        // divider = -1
        buf[pdb] = 0xFF; buf[pdb + 1] = 0xFF;
        // latitude = 35123 (35.123 deg)
        let lat_bytes = 35123i32.to_be_bytes();
        buf[pdb + 2..pdb + 6].copy_from_slice(&lat_bytes);
        // longitude = -97456 (-97.456 deg)
        let lon_bytes = (-97456i32).to_be_bytes();
        buf[pdb + 6..pdb + 10].copy_from_slice(&lon_bytes);
        // height = 1200 ft
        let ht_bytes = 1200i16.to_be_bytes();
        buf[pdb + 10..pdb + 12].copy_from_slice(&ht_bytes);
        // product code = 94
        buf[pdb + 12] = 0; buf[pdb + 13] = 94;
        // volume date = 18628  (arbitrary)
        let vd_bytes = 18628u16.to_be_bytes();
        buf[pdb + 26..pdb + 28].copy_from_slice(&vd_bytes);
        // volume time = 43200 (12:00:00)
        let vt_bytes = 43200u32.to_be_bytes();
        buf[pdb + 28..pdb + 32].copy_from_slice(&vt_bytes);
        // symbology offset = 0 (no symbology)
        buf[pdb + 60..pdb + 64].copy_from_slice(&0u32.to_be_bytes());

        buf
    }

    #[test]
    fn parse_minimal_level3() {
        let data = make_minimal_l3();
        let l3 = Level3File::from_bytes(&data).unwrap();
        assert_eq!(l3.product_code, 94);
        assert_eq!(l3.source_id, 25);
        assert!((l3.latitude - 35.123).abs() < 0.001);
        assert!((l3.longitude - (-97.456)).abs() < 0.001);
        assert!((l3.height - 1200.0).abs() < 0.1);
        assert!(l3.data.is_empty());
        assert_eq!(l3.num_bins, 0);
        assert_eq!(l3.num_radials, 0);
    }

    #[test]
    fn parse_with_radial_symbology() {
        // Build a blob with 2 radials, 4 bins each.
        let mut buf = vec![0u8; 18 + 102];

        // Message header: code=94, source=1
        buf[0] = 0; buf[1] = 94;
        buf[12] = 0; buf[13] = 1;

        let pdb = 18;
        buf[pdb] = 0xFF; buf[pdb + 1] = 0xFF;
        let lat = 40000i32.to_be_bytes();
        buf[pdb + 2..pdb + 6].copy_from_slice(&lat);
        let lon = (-90000i32).to_be_bytes();
        buf[pdb + 6..pdb + 10].copy_from_slice(&lon);
        buf[pdb + 10..pdb + 12].copy_from_slice(&500i16.to_be_bytes());
        buf[pdb + 12] = 0; buf[pdb + 13] = 94;
        buf[pdb + 26..pdb + 28].copy_from_slice(&1u16.to_be_bytes());
        buf[pdb + 28..pdb + 32].copy_from_slice(&0u32.to_be_bytes());

        // Symbology offset: points to right after the PDB.
        // Offset is in halfwords from message start.
        // PDB ends at 18+102=120. Symbology block starts at 120.
        // In halfwords: 120/2 = 60.
        let sym_hw = 60u32;
        buf[pdb + 60..pdb + 64].copy_from_slice(&sym_hw.to_be_bytes());

        // Symbology block header (16 bytes)
        let sym_off = 120;
        buf.resize(sym_off + 200, 0);
        // divider = -1
        buf[sym_off] = 0xFF; buf[sym_off + 1] = 0xFF;
        // block id = 1
        buf[sym_off + 2] = 0; buf[sym_off + 3] = 1;
        // block length (placeholder, large enough)
        buf[sym_off + 4..sym_off + 8].copy_from_slice(&150u32.to_be_bytes());
        // num layers = 1
        buf[sym_off + 8] = 0; buf[sym_off + 9] = 1;
        // layer divider = -1
        buf[sym_off + 10] = 0xFF; buf[sym_off + 11] = 0xFF;
        // layer length (placeholder)
        buf[sym_off + 12..sym_off + 16].copy_from_slice(&100u32.to_be_bytes());

        // Radial packet at sym_off + 16
        let pkt = sym_off + 16;
        // packet code = 0xAF1F
        buf[pkt] = 0xAF; buf[pkt + 1] = 0x1F;
        // index first bin = 0
        buf[pkt + 2] = 0; buf[pkt + 3] = 0;
        // num bins = 4
        buf[pkt + 4] = 0; buf[pkt + 5] = 4;
        // i-center, j-center, scale factor
        buf[pkt + 10] = 0; buf[pkt + 11] = 1;
        // num radials = 2
        buf[pkt + 12] = 0; buf[pkt + 13] = 2;

        // Radial 1: 4 bins using 4 bytes of RLE (run=1 for each)
        let r1 = pkt + 14;
        buf[r1] = 0; buf[r1 + 1] = 2; // num halfwords = 2 -> 4 bytes
        buf[r1 + 2] = 0; buf[r1 + 3] = 0; // start angle
        buf[r1 + 4] = 0; buf[r1 + 5] = 10; // delta angle
        // RLE: run=1, val=5 for each bin -> 0x15 0x15 0x15 0x15
        buf[r1 + 6] = 0x15;
        buf[r1 + 7] = 0x15;
        buf[r1 + 8] = 0x15;
        buf[r1 + 9] = 0x15;

        // Radial 2
        let r2 = r1 + 10;
        buf[r2] = 0; buf[r2 + 1] = 2;
        buf[r2 + 2] = 0; buf[r2 + 3] = 10;
        buf[r2 + 4] = 0; buf[r2 + 5] = 10;
        // RLE: run=1, val=3 -> 0x13 for each
        buf[r2 + 6] = 0x13;
        buf[r2 + 7] = 0x13;
        buf[r2 + 8] = 0x13;
        buf[r2 + 9] = 0x13;

        let l3 = Level3File::from_bytes(&buf).unwrap();
        assert_eq!(l3.num_bins, 4);
        assert_eq!(l3.num_radials, 2);
        assert_eq!(l3.data.len(), 8);
        // Radial 1: all 5.0
        assert!((l3.data[0] - 5.0).abs() < 0.001);
        assert!((l3.data[3] - 5.0).abs() < 0.001);
        // Radial 2: all 3.0
        assert!((l3.data[4] - 3.0).abs() < 0.001);
        assert!((l3.data[7] - 3.0).abs() < 0.001);
    }

    #[test]
    fn strip_wmo_header_works() {
        // Binary data starting with a non-ASCII byte should be returned as-is.
        let bin = [0x00u8, 0x5E, 0xFF];
        assert_eq!(strip_wmo_header(&bin).len(), 3);

        // ASCII header followed by \r\n\r\n then binary.
        let mut with_hdr = Vec::new();
        with_hdr.extend_from_slice(b"SDUS55 KTLX 121200\r\n\r\n");
        with_hdr.extend_from_slice(&[0x00, 0x5E, 0xFF]);
        let stripped = strip_wmo_header(&with_hdr);
        assert_eq!(stripped, &[0x00, 0x5E, 0xFF]);
    }

    #[test]
    fn volume_datetime_basic() {
        // Day 1 from epoch = 1970-01-01, time=0 => 1970-01-01 00:00:00
        let s = volume_datetime(1, 0);
        assert_eq!(s, "1970-01-01 00:00:00");

        let s2 = volume_datetime(1, 43200);
        assert_eq!(s2, "1970-01-01 12:00:00");
    }

    #[test]
    fn too_short_returns_error() {
        let data = [0u8; 10];
        assert!(Level3File::from_bytes(&data).is_err());
    }

    #[test]
    fn be_helpers() {
        let d = [0x01, 0x02, 0x03, 0x04];
        assert_eq!(be_u16(&d, 0), 0x0102);
        assert_eq!(be_u32(&d, 0), 0x01020304);
        assert_eq!(be_i16(&d, 0), 0x0102);
        assert_eq!(be_i32(&d, 0), 0x01020304);
    }
}
