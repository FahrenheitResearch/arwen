//! GINI satellite image file reader.
//!
//! GINI is the NOAA/NWS satellite image format used for GOES imagery.
//! The file consists of a Product Definition Block (PDB) header followed by
//! row-major raster pixel data. Files may optionally include a WMO text
//! header and/or be zlib-compressed.
//!
//! Reference: MetPy's `metpy.io.GiniFile` implementation.

use std::io::Read;

// ── Lookup tables ──────────────────────────────────────────────────────

const CRAFTS: &[&str] = &[
    "Unknown", "Unknown", "Miscellaneous", "JERS", "ERS/QuikSCAT",
    "POES/NPOESS", "Composite", "DMSP", "GMS", "METEOSAT",
    "GOES-7", "GOES-8", "GOES-9", "GOES-10", "GOES-11", "GOES-12",
    "GOES-13", "GOES-14", "GOES-15", "GOES-16",
];

const SECTORS: &[&str] = &[
    "NH Composite", "East CONUS", "West CONUS", "Alaska Regional",
    "Alaska National", "Hawaii Regional", "Hawaii National",
    "Puerto Rico Regional", "Puerto Rico National", "Supernational",
    "NH Composite", "Central CONUS", "East Floater", "West Floater",
    "Central Floater", "Polar Floater",
];

const CHANNELS: &[&str] = &[
    "Unknown", "Visible", "IR (3.9 micron)", "WV (6.5/6.7 micron)",
    "IR (11 micron)", "IR (12 micron)", "IR (13 micron)", "IR (1.3 micron)",
    "Reserved", "Reserved", "Reserved", "Reserved", "Reserved",
    "LI (Imager)", "PW (Imager)", "Surface Skin Temp (Imager)",
    "LI (Sounder)", "PW (Sounder)", "Surface Skin Temp (Sounder)",
    "CAPE", "Land-sea Temp", "WINDEX", "Dry Microburst Potential Index",
    "Microburst Day Potential Index", "Convective Inhibition",
    "Volcano Imagery", "Scatterometer", "Cloud Top", "Cloud Amount",
    "Rainfall Rate", "Surface Wind Speed", "Surface Wetness",
    "Ice Concentration", "Ice Type", "Ice Edge", "Cloud Water Content",
    "Surface Type", "Snow Indicator", "Snow/Water Content",
    "Volcano Imagery", "Reserved",
    "Sounder (14.71 micron)", "Sounder (14.37 micron)",
    "Sounder (14.06 micron)", "Sounder (13.64 micron)",
    "Sounder (13.37 micron)", "Sounder (12.66 micron)",
    "Sounder (12.02 micron)", "Sounder (11.03 micron)",
    "Sounder (9.71 micron)", "Sounder (7.43 micron)",
    "Sounder (7.02 micron)", "Sounder (6.51 micron)",
    "Sounder (4.57 micron)", "Sounder (4.52 micron)",
    "Sounder (4.45 micron)", "Sounder (4.13 micron)",
    "Sounder (3.98 micron)", "Sounder (3.74 micron)",
    "Sounder (Visible)", "Percent Normal TPW",
];

fn lookup_name(table: &[&str], idx: u8) -> String {
    let i = idx as usize;
    if i < table.len() {
        table[i].to_string()
    } else {
        format!("Unknown({})", idx)
    }
}

// ── Projection enum ────────────────────────────────────────────────────

/// GINI projection types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GiniProjection {
    Mercator,
    LambertConformal,
    PolarStereographic,
    Unknown(u8),
}

impl GiniProjection {
    fn from_byte(b: u8) -> Self {
        match b {
            1 => GiniProjection::Mercator,
            3 => GiniProjection::LambertConformal,
            5 => GiniProjection::PolarStereographic,
            _ => GiniProjection::Unknown(b),
        }
    }

    /// Return a human-readable name for the projection.
    pub fn name(&self) -> &str {
        match self {
            GiniProjection::Mercator => "mercator",
            GiniProjection::LambertConformal => "lambert_conformal",
            GiniProjection::PolarStereographic => "polar_stereographic",
            GiniProjection::Unknown(_) => "unknown",
        }
    }
}

// ── Main struct ────────────────────────────────────────────────────────

/// A parsed GINI satellite image file.
#[derive(Debug, Clone)]
pub struct GiniFile {
    /// Source indicator byte.
    pub source: String,
    /// Creating entity (satellite platform).
    pub creating_entity: String,
    /// Image sector name.
    pub sector: String,
    /// Channel / physical element name.
    pub channel: String,
    /// Number of pixels per scan line (columns).
    pub nx: usize,
    /// Number of scan lines (rows).
    pub ny: usize,
    /// Map projection.
    pub projection: String,
    /// Latitude of the first grid point (lower-left), degrees.
    pub lat1: f64,
    /// Longitude of the first grid point (lower-left), degrees.
    pub lon1: f64,
    /// Grid spacing in x (km).
    pub dx: f64,
    /// Grid spacing in y (km).
    pub dy: f64,
    /// Longitude of orientation (lov), degrees. Lambert/PS only.
    pub lov: f64,
    /// Standard latitude (lat_in), degrees.
    pub lat_in: f64,
    /// Raw pixel values in row-major order (ny rows, nx columns).
    pub data: Vec<u8>,
    /// Image datetime as "YYYY-MM-DD HH:MM:SS".
    pub datetime: String,
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Decode a 3-byte scaled integer from the GINI PDB.
///
/// The high bit is the sign (0 = positive, 1 = negative). The remaining
/// 23 bits form an unsigned integer that is divided by 10000 to produce
/// the final value.
fn scaled_int(s: &[u8]) -> f64 {
    let sign = if s[0] & 0x80 != 0 { -1.0 } else { 1.0 };
    let int_val = (((s[0] & 0x7F) as u32) << 16) | ((s[1] as u32) << 8) | (s[2] as u32);
    sign * (int_val as f64) / 10000.0
}

/// Convert 7-byte GINI datetime fields to a string.
fn make_datetime(s: &[u8]) -> String {
    if s.len() < 7 {
        return "0000-00-00 00:00:00".to_string();
    }
    let mut year = s[0] as u32;
    if year < 70 {
        year += 100;
    }
    year += 1900;
    let month = s[1];
    let day = s[2];
    let hour = s[3];
    let minute = s[4];
    let second = s[5];
    // s[6] is centiseconds, which we ignore for the string
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        year, month, day, hour, minute, second
    )
}

/// Try to strip a WMO text header preceding the binary product.
fn strip_wmo_header(data: &[u8]) -> &[u8] {
    if data.is_empty() {
        return data;
    }
    // WMO headers are ASCII text (bytes >= 0x20, < 0x7F) followed by \r\r\n.
    // Scan for the pattern T\w{3}\d{2} ... \r\r\n which MetPy uses.
    if data[0] >= 0x20 && data[0] < 0x7F {
        // Look for \r\r\n boundary
        for i in 0..data.len().saturating_sub(3) {
            if data[i] == b'\r' && data[i + 1] == b'\r' && data[i + 2] == b'\n' {
                return &data[i + 3..];
            }
        }
        // Also try \r\n\r\n
        for i in 0..data.len().saturating_sub(4) {
            if data[i] == b'\r'
                && data[i + 1] == b'\n'
                && data[i + 2] == b'\r'
                && data[i + 3] == b'\n'
            {
                return &data[i + 4..];
            }
        }
    }
    data
}

/// Attempt to decompress zlib-compressed data. Returns the original data
/// if decompression fails or the data is not compressed.
fn try_zlib_decompress(data: &[u8]) -> Vec<u8> {
    // zlib magic: first byte is typically 0x78
    if data.len() < 2 {
        return data.to_vec();
    }
    if data[0] == 0x78 {
        let mut decoder = flate2::read::ZlibDecoder::new(data);
        let mut out = Vec::new();
        if decoder.read_to_end(&mut out).is_ok() && !out.is_empty() {
            return out;
        }
    }
    data.to_vec()
}

// ── Parsing ────────────────────────────────────────────────────────────

impl GiniFile {
    /// Read and parse a GINI file from disk.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open {}: {}", path, e))?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .map_err(|e| format!("Failed to read {}: {}", path, e))?;
        Self::from_bytes(&buf)
    }

    /// Parse a GINI image from a byte slice.
    ///
    /// Handles optional WMO headers and zlib compression.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        // Strip WMO header if present
        let data = strip_wmo_header(data);

        // Attempt zlib decompression
        let decompressed = try_zlib_decompress(data);
        let buf = if decompressed.len() > data.len() {
            // If decompression produced more data, strip WMO header again
            // (compressed data can contain a second WMO header)
            strip_wmo_header(&decompressed)
        } else {
            &decompressed
        };

        Self::parse_pdb(buf)
    }

    /// Parse the Product Definition Block and raster data.
    fn parse_pdb(buf: &[u8]) -> Result<Self, String> {
        // The PDB is at least 21 bytes for the first section, and the
        // standard PDB size is 512 bytes. We need at least 37 bytes to
        // reach the projection-dependent fields.
        if buf.len() < 37 {
            return Err(format!(
                "Data too short for GINI PDB (need >= 37 bytes, have {})",
                buf.len()
            ));
        }

        // ── PDB first section (offsets from MetPy's prod_desc_fmt) ─────
        // The struct format is: source(b) creating_entity(b) sector_id(b)
        // channel(b) num_records(H) record_len(H) datetime(7s) projection(b)
        // nx(H) ny(H) la1(3s) lo1(3s)
        let source_byte = buf[0];
        let creating_entity_byte = buf[1];
        let sector_byte = buf[2];
        let channel_byte = buf[3];
        let num_records = u16::from_be_bytes([buf[4], buf[5]]) as usize;
        let record_len = u16::from_be_bytes([buf[6], buf[7]]) as usize;
        let datetime = make_datetime(&buf[8..15]);
        let proj_byte = buf[15];
        let nx = u16::from_be_bytes([buf[16], buf[17]]) as usize;
        let ny = u16::from_be_bytes([buf[18], buf[19]]) as usize;
        let lat1 = scaled_int(&buf[20..23]);
        let lon1 = scaled_int(&buf[23..26]);

        let projection = GiniProjection::from_byte(proj_byte);

        // ── Projection-dependent fields ────────────────────────────────
        let (dx, dy, lov, lat_in) = match projection {
            GiniProjection::LambertConformal | GiniProjection::PolarStereographic => {
                // Lambert/PS format: reserved(b) lov(3s) dx(3s) dy(3s) proj_center(b)
                if buf.len() < 37 {
                    return Err("Data too short for Lambert/PS projection fields".into());
                }
                // byte 26 is reserved
                let lov = scaled_int(&buf[27..30]);
                let dx_val = scaled_int(&buf[30..33]);
                let dy_val = scaled_int(&buf[33..36]);
                // byte 36 is proj_center

                // lat_in comes from prod_desc2 section
                // scanning_mode(b, 3 bits) lat_in(3s) resolution(b)
                // compression(b) version(b) pdb_size(H) nav_cal(b)
                // The prod_desc2 starts after the projection info.
                let pd2_off = 37; // after proj-dependent fields
                let lat_in_val = if buf.len() >= pd2_off + 4 {
                    scaled_int(&buf[pd2_off + 1..pd2_off + 4])
                } else {
                    0.0
                };
                (dx_val, dy_val, lov, lat_in_val)
            }
            GiniProjection::Mercator => {
                // Mercator format: resolution(b) la2(3s) lo2(3s) di(H) dj(H)
                if buf.len() < 36 {
                    return Err("Data too short for Mercator projection fields".into());
                }
                let _resolution = buf[26];
                let _la2 = scaled_int(&buf[27..30]);
                let _lo2 = scaled_int(&buf[30..33]);
                let di = u16::from_be_bytes([buf[33], buf[34]]) as f64;
                let dj = u16::from_be_bytes([buf[35], buf[36]]) as f64;

                let pd2_off = 37;
                let lat_in_val = if buf.len() >= pd2_off + 4 {
                    scaled_int(&buf[pd2_off + 1..pd2_off + 4])
                } else {
                    0.0
                };
                (di, dj, lon1, lat_in_val)
            }
            _ => (0.0, 0.0, 0.0, 0.0),
        };

        // ── PDB size ───────────────────────────────────────────────────
        // prod_desc2 ends with: compression(b) version(b) pdb_size(H) nav_cal(b)
        // Locate pdb_size. prod_desc2 starts at offset 37.
        let pdb_size = if buf.len() >= 44 {
            let s = u16::from_be_bytes([buf[42], buf[43]]) as usize;
            if s == 0 { 512 } else { s }
        } else {
            512
        };

        // ── Raster data ────────────────────────────────────────────────
        let raster_len = num_records * record_len;
        let raster_start = pdb_size.min(buf.len());
        let raster_end = (raster_start + raster_len).min(buf.len());
        let data = buf[raster_start..raster_end].to_vec();

        Ok(GiniFile {
            source: format!("{}", source_byte),
            creating_entity: lookup_name(CRAFTS, creating_entity_byte),
            sector: lookup_name(SECTORS, sector_byte),
            channel: lookup_name(CHANNELS, channel_byte),
            nx,
            ny,
            projection: projection.name().to_string(),
            lat1,
            lon1,
            dx,
            dy,
            lov,
            lat_in,
            data,
            datetime,
        })
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaled_int_positive() {
        // 0x00_C3_50 = 50000 => 50000/10000 = 5.0
        let val = scaled_int(&[0x00, 0xC3, 0x50]);
        assert!((val - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_int_negative() {
        // Set sign bit: 0x80 | 0x00, 0xC3, 0x50 = -5.0
        let val = scaled_int(&[0x80, 0xC3, 0x50]);
        assert!((val - (-5.0)).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_int_zero() {
        let val = scaled_int(&[0x00, 0x00, 0x00]);
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_make_datetime() {
        // year=24 (2024), month=3, day=15, hour=12, min=30, sec=45, cs=0
        let dt = make_datetime(&[24, 3, 15, 12, 30, 45, 0]);
        assert_eq!(dt, "2024-03-15 12:30:45");
    }

    #[test]
    fn test_make_datetime_pre_1970() {
        // year=69 => 1900+69=1969, but <70 rule: year+100=169 => 1900+169=2069
        let dt = make_datetime(&[69, 1, 1, 0, 0, 0, 0]);
        assert_eq!(dt, "2069-01-01 00:00:00");
    }

    #[test]
    fn test_make_datetime_post_1970() {
        // year=70 => 1900+70=1970
        let dt = make_datetime(&[70, 6, 15, 18, 0, 0, 0]);
        assert_eq!(dt, "1970-06-15 18:00:00");
    }

    #[test]
    fn test_lookup_name_valid() {
        assert_eq!(lookup_name(CRAFTS, 11), "GOES-8");
        assert_eq!(lookup_name(SECTORS, 1), "East CONUS");
        assert_eq!(lookup_name(CHANNELS, 1), "Visible");
    }

    #[test]
    fn test_lookup_name_out_of_range() {
        let name = lookup_name(CRAFTS, 250);
        assert!(name.starts_with("Unknown("));
    }

    #[test]
    fn test_projection_from_byte() {
        assert_eq!(GiniProjection::from_byte(1), GiniProjection::Mercator);
        assert_eq!(
            GiniProjection::from_byte(3),
            GiniProjection::LambertConformal
        );
        assert_eq!(
            GiniProjection::from_byte(5),
            GiniProjection::PolarStereographic
        );
        assert_eq!(GiniProjection::from_byte(99), GiniProjection::Unknown(99));
    }

    #[test]
    fn test_strip_wmo_header_no_header() {
        let binary = [0x00u8, 0x0B, 0x01, 0xFF];
        assert_eq!(strip_wmo_header(&binary), &binary);
    }

    #[test]
    fn test_strip_wmo_header_with_header() {
        let mut with_hdr = Vec::new();
        with_hdr.extend_from_slice(b"TICZ99 KNES 121200\r\r\n");
        with_hdr.extend_from_slice(&[0x00, 0x0B, 0x01]);
        let stripped = strip_wmo_header(&with_hdr);
        assert_eq!(stripped, &[0x00, 0x0B, 0x01]);
    }

    #[test]
    fn test_from_bytes_minimal() {
        // Build a minimal GINI PDB: Lambert Conformal, 100x200 image
        let mut buf = vec![0u8; 512 + 100 * 200]; // PDB + raster

        // source=0, entity=16 (GOES-13), sector=1 (East CONUS), channel=1 (Visible)
        buf[0] = 0;
        buf[1] = 16;
        buf[2] = 1;
        buf[3] = 1;
        // num_records=200 (ny), record_len=100 (nx)
        buf[4..6].copy_from_slice(&200u16.to_be_bytes());
        buf[6..8].copy_from_slice(&100u16.to_be_bytes());
        // datetime: year=24, month=7, day=4, hour=18, min=0, sec=0, cs=0
        buf[8] = 24;
        buf[9] = 7;
        buf[10] = 4;
        buf[11] = 18;
        buf[12] = 0;
        buf[13] = 0;
        buf[14] = 0;
        // projection = 3 (Lambert Conformal)
        buf[15] = 3;
        // nx=100, ny=200
        buf[16..18].copy_from_slice(&100u16.to_be_bytes());
        buf[18..20].copy_from_slice(&200u16.to_be_bytes());
        // la1 = 20.0 degrees => 200000 => 0x030D40
        buf[20] = 0x03;
        buf[21] = 0x0D;
        buf[22] = 0x40;
        // lo1 = -120.0 degrees => sign bit set, 1200000 => 0x92_4F_80
        buf[23] = 0x92;
        buf[24] = 0x4F;
        buf[25] = 0x80;
        // reserved
        buf[26] = 0;
        // lov = -95.0 degrees => sign bit set, 950000 => 0x8E_7E_F0
        buf[27] = 0x8E;
        buf[28] = 0x7E;
        buf[29] = 0xF0;
        // dx = 10.0 km => 100000 => 0x01_86_A0
        buf[30] = 0x01;
        buf[31] = 0x86;
        buf[32] = 0xA0;
        // dy = 10.0 km => 100000 => 0x01_86_A0
        buf[33] = 0x01;
        buf[34] = 0x86;
        buf[35] = 0xA0;
        // proj_center
        buf[36] = 0;
        // prod_desc2: scanning_mode, lat_in = 25.0 => 250000 = 0x03_D0_90
        buf[37] = 0;
        buf[38] = 0x03;
        buf[39] = 0xD0;
        buf[40] = 0x90;
        // resolution, compression, version
        buf[41] = 0;
        // pdb_size = 512
        buf[42..44].copy_from_slice(&512u16.to_be_bytes());
        // nav_cal
        buf[44] = 0;

        // Fill raster with a pattern
        for i in 0..(100 * 200) {
            buf[512 + i] = (i % 256) as u8;
        }

        let gini = GiniFile::from_bytes(&buf).unwrap();
        assert_eq!(gini.creating_entity, "GOES-13");
        assert_eq!(gini.sector, "East CONUS");
        assert_eq!(gini.channel, "Visible");
        assert_eq!(gini.nx, 100);
        assert_eq!(gini.ny, 200);
        assert_eq!(gini.projection, "lambert_conformal");
        assert!((gini.lat1 - 20.0).abs() < 0.01);
        assert!((gini.lon1 - (-120.0)).abs() < 0.01);
        assert!((gini.dx - 10.0).abs() < 0.01);
        assert!((gini.dy - 10.0).abs() < 0.01);
        assert!((gini.lov - (-95.0)).abs() < 0.01);
        assert!((gini.lat_in - 25.0).abs() < 0.01);
        assert_eq!(gini.datetime, "2024-07-04 18:00:00");
        assert_eq!(gini.data.len(), 100 * 200);
    }

    #[test]
    fn test_too_short_returns_error() {
        let data = [0u8; 10];
        assert!(GiniFile::from_bytes(&data).is_err());
    }
}
