//! GEMPAK grid file reader.
//!
//! Parses GEMPAK Data Management (DM) grid files, the legacy binary format
//! produced by UCAR/Unidata's GEMPAK meteorological analysis package.
//!
//! The format stores gridded NWP model output in a fixed-record binary layout:
//!
//! 1. **File header** -- 28-byte "GEMPAK DATA MANAGEMENT FILE " magic string
//! 2. **Product description** -- file geometry (rows, columns, parts, pointers)
//! 3. **File keys** -- navigation block (projection info) and analysis block
//! 4. **Row/column keys and headers** -- grid index metadata
//! 5. **Data blocks** -- packed grid data
//!
//! Grid data can be stored in several packing modes. This implementation
//! supports:
//!
//! - **None** (unpacked float32 values)
//! - **GRIB** (reference + scale + bit-width integer packing)
//! - **DEC** (same algorithm as GRIB)
//! - **DIFF** (second-order differencing packing)
//!
//! Not yet supported: NMC packing, GRIB2 packing.
//!
//! ## Example
//!
//! ```no_run
//! use metrust::io::gempak::GempakGrid;
//!
//! let gempak = GempakGrid::from_file("model.gem").unwrap();
//! println!("Grid dimensions: {}x{}", gempak.nx, gempak.ny);
//! for rec in &gempak.grids {
//!     println!("  {} level={} time={}", rec.parameter, rec.level, rec.time);
//! }
//! ```

use std::collections::HashMap;
use std::io::Read;

// ── Constants ──────────────────────────────────────────────────────────

const BYTES_PER_WORD: usize = 4;
const GEMPAK_HEADER: &[u8] = b"GEMPAK DATA MANAGEMENT FILE ";
const USED_FLAG: i32 = 9999;
#[cfg(test)]
const MISSING_FLOAT: f32 = -9999.0;

// ── Public types ───────────────────────────────────────────────────────

/// Projection information extracted from the GEMPAK navigation block.
#[derive(Debug, Clone)]
pub struct NavigationInfo {
    /// GEMPAK projection code (e.g. "LCC", "CED", "MER", "STR").
    pub projection: String,
    /// Lower-left latitude of the grid domain (degrees).
    pub lower_left_lat: f64,
    /// Lower-left longitude of the grid domain (degrees).
    pub lower_left_lon: f64,
    /// Upper-right latitude of the grid domain (degrees).
    pub upper_right_lat: f64,
    /// Upper-right longitude of the grid domain (degrees).
    pub upper_right_lon: f64,
    /// First projection angle (meaning depends on projection type).
    pub angle1: f64,
    /// Second projection angle.
    pub angle2: f64,
    /// Third projection angle.
    pub angle3: f64,
}

/// Analysis block information.
#[derive(Debug, Clone)]
pub struct AnalysisInfo {
    /// Analysis type (1 = type 1, 2 = type 2).
    pub analysis_type: i32,
    /// Grid spacing parameter.
    pub delta_n: f64,
    /// Analysis area lower-left lat.
    pub garea_ll_lat: f64,
    /// Analysis area lower-left lon.
    pub garea_ll_lon: f64,
    /// Analysis area upper-right lat.
    pub garea_ur_lat: f64,
    /// Analysis area upper-right lon.
    pub garea_ur_lon: f64,
}

/// A single grid record from a GEMPAK file.
#[derive(Debug, Clone)]
pub struct GempakGridRecord {
    /// Grid number (index within the file).
    pub grid_number: usize,
    /// Parameter name (e.g. "TMPK", "HGHT", "UREL").
    pub parameter: String,
    /// Primary vertical level value.
    pub level: f64,
    /// Secondary level value (for layers), or -1 if not set.
    pub level2: f64,
    /// Vertical coordinate type (e.g. "PRES", "HGHT", "NONE").
    pub coordinate: String,
    /// Valid datetime string in "YYYYMMDD/HHMM" format.
    pub time: String,
    /// Secondary datetime string, if present.
    pub time2: String,
    /// Forecast type (e.g. "analysis", "forecast").
    pub forecast_type: String,
    /// Forecast offset as "HHH:MM".
    pub forecast_time: String,
    /// Grid data values, row-major order (ny * nx elements).
    pub data: Vec<f64>,
}

/// Parsed GEMPAK grid file.
#[derive(Debug, Clone)]
pub struct GempakGrid {
    /// Data source description.
    pub source: String,
    /// File type description.
    pub grid_type: String,
    /// Number of grid columns (x-dimension).
    pub nx: usize,
    /// Number of grid rows (y-dimension).
    pub ny: usize,
    /// Navigation (projection) information.
    pub navigation: Option<NavigationInfo>,
    /// Analysis block information.
    pub analysis: Option<AnalysisInfo>,
    /// All grid records in the file.
    pub grids: Vec<GempakGridRecord>,
}

// ── Packing types ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackingType {
    None,
    Grib,
    Nmc,
    Diff,
    Dec,
    Grib2,
    Unknown(i32),
}

impl PackingType {
    fn from_i32(v: i32) -> Self {
        match v {
            0 => Self::None,
            1 => Self::Grib,
            2 => Self::Nmc,
            3 => Self::Diff,
            4 => Self::Dec,
            5 => Self::Grib2,
            x => Self::Unknown(x),
        }
    }
}

// ── Forecast type ──────────────────────────────────────────────────────

fn forecast_type_name(code: i32) -> &'static str {
    match code {
        0 => "analysis",
        1 => "forecast",
        2 => "guess",
        3 => "initial",
        _ => "unknown",
    }
}

// ── Vertical coordinate ────────────────────────────────────────────────

fn vertical_coord_name(code: i32) -> String {
    match code {
        0 => "NONE".to_string(),
        1 => "PRES".to_string(),
        2 => "THTA".to_string(),
        3 => "HGHT".to_string(),
        4 => "SGMA".to_string(),
        5 => "DPTH".to_string(),
        6 => "HYBD".to_string(),
        7 => "PVAB".to_string(),
        8 => "PVBL".to_string(),
        _ => {
            // Try to decode as 4 ASCII bytes
            let bytes = code.to_be_bytes();
            String::from_utf8(bytes.to_vec())
                .unwrap_or_else(|_| format!("UNKN({})", code))
                .trim()
                .to_string()
        }
    }
}

// ── Data source ────────────────────────────────────────────────────────

fn data_source_name(code: i32) -> &'static str {
    match code {
        0 => "model",
        1 => "airway_surface",
        2 => "metar",
        3 => "ship",
        4 => "raob_buoy",
        5 => "synop_raob_vas",
        6 => "grid",
        7 => "watch_by_county",
        99 => "unknown",
        100 => "text",
        _ => "unknown",
    }
}

// ── Byte-swapping buffer ───────────────────────────────────────────────

/// Internal cursor over a byte buffer with endian-aware reads.
struct GemBuf {
    data: Vec<u8>,
    pos: usize,
    big_endian: bool,
}

impl GemBuf {
    fn new(data: Vec<u8>) -> Result<Self, String> {
        if data.len() < 32 {
            return Err("File too small to be a GEMPAK file".to_string());
        }
        Ok(GemBuf {
            data,
            pos: 0,
            big_endian: true, // determined after header
        })
    }

    fn jump(&mut self, offset: usize) {
        self.pos = offset;
    }

    fn read_bytes(&mut self, n: usize) -> Result<&[u8], String> {
        if self.pos + n > self.data.len() {
            return Err(format!(
                "Read past end of buffer: pos={}, need={}, len={}",
                self.pos, n, self.data.len()
            ));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn skip(&mut self, n: usize) {
        self.pos += n;
    }

    fn read_4bytes(&mut self) -> Result<[u8; 4], String> {
        if self.pos + 4 > self.data.len() {
            return Err(format!(
                "Read past end of buffer: pos={}, need=4, len={}",
                self.pos, self.data.len()
            ));
        }
        let arr = [
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ];
        self.pos += 4;
        Ok(arr)
    }

    fn read_i32(&mut self) -> Result<i32, String> {
        let arr = self.read_4bytes()?;
        Ok(if self.big_endian {
            i32::from_be_bytes(arr)
        } else {
            i32::from_le_bytes(arr)
        })
    }

    fn read_f32(&mut self) -> Result<f32, String> {
        let arr = self.read_4bytes()?;
        Ok(if self.big_endian {
            f32::from_be_bytes(arr)
        } else {
            f32::from_le_bytes(arr)
        })
    }

    fn read_string(&mut self, n: usize) -> Result<String, String> {
        let b = self.read_bytes(n)?.to_vec();
        Ok(String::from_utf8_lossy(&b).trim_end_matches('\0').trim().to_string())
    }

    /// Read a 4-byte character field (GEMPAK stores identifiers this way).
    fn read_char4(&mut self) -> Result<String, String> {
        self.read_string(4)
    }
}

/// Convert a 1-indexed word position to a byte offset.
#[inline]
fn word_to_pos(word: usize) -> usize {
    (word * BYTES_PER_WORD) - BYTES_PER_WORD
}

/// Fortran-compatible bit shifting (handles sign extension like GEMPAK's ISHIFT).
#[inline]
fn fortran_ishift(i: i32, shift: i32) -> i32 {
    if shift >= 0 {
        // Left shift, keep low 32 bits
        let result = ((i as u32) << (shift as u32)) as i32;
        result
    } else {
        // Right shift the unsigned representation
        ((i as u32) >> ((-shift) as u32)) as i32
    }
}

// ── Product Description ────────────────────────────────────────────────

/// Parsed product description block.
#[derive(Debug)]
struct ProductDesc {
    _version: i32,
    file_headers: i32,
    file_keys_ptr: i32,
    rows: i32,
    row_keys: i32,
    row_keys_ptr: i32,
    row_headers_ptr: i32,
    columns: i32,
    column_keys: i32,
    column_keys_ptr: i32,
    column_headers_ptr: i32,
    parts: i32,
    parts_ptr: i32,
    _data_mgmt_ptr: i32,
    _data_mgmt_length: i32,
    data_block_ptr: i32,
    file_type: i32,
    data_source: i32,
    _machine_type: i32,
    _missing_int: i32,
    missing_float: f32,
}

impl ProductDesc {
    fn read(buf: &mut GemBuf) -> Result<Self, String> {
        let version = buf.read_i32()?;
        let file_headers = buf.read_i32()?;
        let file_keys_ptr = buf.read_i32()?;
        let rows = buf.read_i32()?;
        let row_keys = buf.read_i32()?;
        let row_keys_ptr = buf.read_i32()?;
        let row_headers_ptr = buf.read_i32()?;
        let columns = buf.read_i32()?;
        let column_keys = buf.read_i32()?;
        let column_keys_ptr = buf.read_i32()?;
        let column_headers_ptr = buf.read_i32()?;
        let parts = buf.read_i32()?;
        let parts_ptr = buf.read_i32()?;
        let data_mgmt_ptr = buf.read_i32()?;
        let data_mgmt_length = buf.read_i32()?;
        let data_block_ptr = buf.read_i32()?;
        let file_type = buf.read_i32()?;
        let data_source = buf.read_i32()?;
        let machine_type = buf.read_i32()?;
        let missing_int = buf.read_i32()?;
        // 12 bytes padding
        buf.skip(12);
        let missing_float = buf.read_f32()?;

        Ok(ProductDesc {
            _version: version,
            file_headers,
            file_keys_ptr,
            rows,
            row_keys,
            row_keys_ptr,
            row_headers_ptr,
            columns,
            column_keys,
            column_keys_ptr,
            column_headers_ptr,
            parts,
            parts_ptr,
            _data_mgmt_ptr: data_mgmt_ptr,
            _data_mgmt_length: data_mgmt_length,
            data_block_ptr,
            file_type,
            data_source,
            _machine_type: machine_type,
            _missing_int: missing_int,
            missing_float,
        })
    }
}

// ── Part description ───────────────────────────────────────────────────

#[derive(Debug)]
struct PartDesc {
    _name: String,
    header_length: i32,
    _data_type: i32,
    _parameter_count: i32,
}

// ── Column header (grid metadata) ──────────────────────────────────────

#[derive(Debug)]
struct ColumnHeader {
    values: HashMap<String, i32>,
}

impl ColumnHeader {
    fn get(&self, key: &str) -> i32 {
        *self.values.get(key).unwrap_or(&0)
    }

    fn get_param_string(&self) -> String {
        // GPM1, GPM2, GPM3 are stored as integer representations of 4-char strings
        let mut result = String::new();
        for key in &["GPM1", "GPM2", "GPM3"] {
            let v = self.get(key);
            let bytes = v.to_be_bytes();
            let s = String::from_utf8_lossy(&bytes).trim().to_string();
            result.push_str(&s);
        }
        result.trim().to_string()
    }
}

// ── DATTIM conversion ──────────────────────────────────────────────────

/// Convert GEMPAK DATTIM integer to a formatted date string.
fn convert_dattim(dattim: i32) -> String {
    if dattim == 0 {
        return String::new();
    }
    if dattim < 100_000_000 {
        // YYMMDD format
        let yy = dattim / 10000;
        let mm = (dattim % 10000) / 100;
        let dd = dattim % 100;
        let year = if yy >= 70 { 1900 + yy } else { 2000 + yy };
        format!("{:04}{:02}{:02}", year, mm, dd)
    } else {
        // MMDDHHMM format encoded as MMDDYYHHMM
        let mm = dattim / 100_000_000;
        let dd = (dattim / 1_000_000) % 100;
        let yy = (dattim / 10_000) % 100;
        let hh = (dattim / 100) % 100;
        let mn = dattim % 100;
        let year = if yy >= 70 { 1900 + yy } else { 2000 + yy };
        format!("{:04}{:02}{:02}/{:02}{:02}", year, mm, dd, hh, mn)
    }
}

/// Convert GEMPAK forecast time integer to (type_code, hours, minutes).
fn convert_ftime(ftime: i32) -> (i32, i32, i32) {
    if ftime < 0 {
        return (0, 0, 0);
    }
    let iftype = ftime / 100_000;
    let iftime = ftime - iftype * 100_000;
    let hours = iftime / 100;
    let minutes = iftime - hours * 100;
    (iftype, hours, minutes)
}

// ── Main implementation ────────────────────────────────────────────────

impl GempakGrid {
    /// Open and parse a GEMPAK grid file from a filesystem path.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| format!("Cannot open '{}': {}", path, e))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| format!("Cannot read '{}': {}", path, e))?;
        Self::from_bytes(&data)
    }

    /// Parse a GEMPAK grid file from an in-memory byte buffer.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let mut buf = GemBuf::new(data.to_vec())?;

        // ── 1. Verify GEMPAK header ───────────────────────────────────
        let header = buf.read_bytes(28)?;
        if header != GEMPAK_HEADER {
            return Err("Not a valid GEMPAK file (header mismatch)".to_string());
        }

        // ── 2. Determine byte order ───────────────────────────────────
        // Next 4 bytes encode the integer 1 in the file's native endianness.
        let endian_check = buf.read_bytes(4)?;
        let as_be = i32::from_be_bytes([endian_check[0], endian_check[1], endian_check[2], endian_check[3]]);
        let as_le = i32::from_le_bytes([endian_check[0], endian_check[1], endian_check[2], endian_check[3]]);
        buf.big_endian = if as_be == 1 {
            true
        } else if as_le == 1 {
            false
        } else {
            // Default to big-endian (standard GEMPAK)
            true
        };

        // Back up to re-read from after the 28-byte header.
        // The endian bytes are actually the start of the product description.
        buf.jump(28);

        // ── 3. Read product description ───────────────────────────────
        let pd = ProductDesc::read(&mut buf)?;

        // Verify file type = 3 (grid file)
        if pd.file_type != 3 {
            return Err(format!(
                "Not a GEMPAK grid file (file_type={}, expected 3)",
                pd.file_type
            ));
        }

        let source = data_source_name(pd.data_source).to_string();

        // ── 4. Read navigation and analysis blocks ────────────────────
        let mut navigation: Option<NavigationInfo> = None;
        let mut analysis: Option<AnalysisInfo> = None;
        let mut kx: usize = 0;
        let mut ky: usize = 0;

        if pd.file_headers > 0 {
            // Jump to file keys pointer
            buf.jump(word_to_pos(pd.file_keys_ptr as usize));

            // Read file key entries: each file header has name(4s) + length(i) + type(i)
            // We just need to skip past them to get to the nav/analysis blocks.
            let _num_headers = pd.file_headers;
            buf.skip((pd.file_headers as usize) * 3 * BYTES_PER_WORD);

            // Navigation Block
            let navb_size = buf.read_i32()?;
            let expected_nav_words = 256; // 13 floats + 972 bytes padding = 13*4 + 972 = 1024 bytes = 256 words
            if navb_size != expected_nav_words {
                // Try reading anyway if size is close
                if navb_size < 13 {
                    return Err(format!(
                        "Navigation block too small: {} words (need at least 13)",
                        navb_size
                    ));
                }
            }

            let _grid_def_type = buf.read_f32()?;
            let proj_bytes = buf.read_bytes(4)?;
            let projection = String::from_utf8_lossy(&proj_bytes[..3])
                .trim()
                .to_string();
            let _left = buf.read_f32()?;
            let _bottom = buf.read_f32()?;
            let right = buf.read_f32()?;
            let top = buf.read_f32()?;
            let ll_lat = buf.read_f32()?;
            let ll_lon = buf.read_f32()?;
            let ur_lat = buf.read_f32()?;
            let ur_lon = buf.read_f32()?;
            let angle1 = buf.read_f32()?;
            let angle2 = buf.read_f32()?;
            let angle3 = buf.read_f32()?;

            kx = right as usize;
            ky = top as usize;

            navigation = Some(NavigationInfo {
                projection,
                lower_left_lat: ll_lat as f64,
                lower_left_lon: ll_lon as f64,
                upper_right_lat: ur_lat as f64,
                upper_right_lon: ur_lon as f64,
                angle1: angle1 as f64,
                angle2: angle2 as f64,
                angle3: angle3 as f64,
            });

            // Skip remaining navigation block padding.
            // We read 12 floats (48 bytes) + 4-byte projection = 52 bytes after the size word.
            let nav_bytes_read = 12 * 4 + 4;
            let nav_total = (navb_size as usize) * BYTES_PER_WORD;
            if nav_total > nav_bytes_read {
                buf.skip(nav_total - nav_bytes_read);
            }

            // Analysis Block
            let anlb_size = buf.read_i32()?;
            if anlb_size > 0 {
                let anlb_start = buf.pos;
                let atype = buf.read_f32()? as i32;
                let delta_n = buf.read_f32()?;

                if atype == 1 {
                    // Format 1: delta_x, delta_y, padding, then area coords
                    let _delta_x = buf.read_f32()?;
                    let _delta_y = buf.read_f32()?;
                    buf.skip(4); // padding
                    let ga_ll_lat = buf.read_f32()?;
                    let ga_ll_lon = buf.read_f32()?;
                    let ga_ur_lat = buf.read_f32()?;
                    let ga_ur_lon = buf.read_f32()?;
                    analysis = Some(AnalysisInfo {
                        analysis_type: atype,
                        delta_n: delta_n as f64,
                        garea_ll_lat: ga_ll_lat as f64,
                        garea_ll_lon: ga_ll_lon as f64,
                        garea_ur_lat: ga_ur_lat as f64,
                        garea_ur_lon: ga_ur_lon as f64,
                    });
                } else if atype == 2 {
                    // Format 2: grid extensions, then area coords
                    buf.skip(4 * 4); // 4 extension floats
                    let ga_ll_lat = buf.read_f32()?;
                    let ga_ll_lon = buf.read_f32()?;
                    let ga_ur_lat = buf.read_f32()?;
                    let ga_ur_lon = buf.read_f32()?;
                    analysis = Some(AnalysisInfo {
                        analysis_type: atype,
                        delta_n: delta_n as f64,
                        garea_ll_lat: ga_ll_lat as f64,
                        garea_ll_lon: ga_ll_lon as f64,
                        garea_ur_lat: ga_ur_lat as f64,
                        garea_ur_lon: ga_ur_lon as f64,
                    });
                }

                // Skip to end of analysis block
                let anlb_total = (anlb_size as usize) * BYTES_PER_WORD;
                let anlb_used = buf.pos - anlb_start;
                if anlb_total > anlb_used {
                    buf.skip(anlb_total - anlb_used);
                }
            }
        }

        // ── 5. Read row keys ──────────────────────────────────────────
        buf.jump(word_to_pos(pd.row_keys_ptr as usize));
        let mut row_key_names = Vec::new();
        for _ in 0..pd.row_keys {
            row_key_names.push(buf.read_char4()?);
        }

        // ── 6. Read column keys ───────────────────────────────────────
        buf.jump(word_to_pos(pd.column_keys_ptr as usize));
        let mut column_key_names = Vec::new();
        for _ in 0..pd.column_keys {
            let key = buf.read_char4()?;
            column_key_names.push(key);
        }

        // ── 7. Read parts ─────────────────────────────────────────────
        let num_parts = pd.parts as usize;
        let mut parts = Vec::new();

        // Parts are laid out as interleaved arrays at parts_ptr:
        //   [name_0..name_N-1, hdr_len_0..hdr_len_N-1,
        //    data_type_0..data_type_N-1, parm_count_0..parm_count_N-1]
        for n in 0..num_parts {
            // The parts pointer area stores:
            // [name_0, name_1, ..., name_N-1,
            //  hdr_len_0, hdr_len_1, ..., hdr_len_N-1,
            //  data_type_0, data_type_1, ..., data_type_N-1,
            //  parm_count_0, parm_count_1, ..., parm_count_N-1]
            let base = pd.parts_ptr as usize;

            buf.jump(word_to_pos(base + n));
            let name = buf.read_char4()?;

            buf.jump(word_to_pos(base + num_parts + n));
            let header_length = buf.read_i32()?;

            buf.jump(word_to_pos(base + 2 * num_parts + n));
            let data_type = buf.read_i32()?;

            buf.jump(word_to_pos(base + 3 * num_parts + n));
            let parameter_count = buf.read_i32()?;

            parts.push(PartDesc {
                _name: name,
                header_length,
                _data_type: data_type,
                _parameter_count: parameter_count,
            });
        }

        // ── 8. Read row headers ───────────────────────────────────────
        buf.jump(word_to_pos(pd.row_headers_ptr as usize));
        let num_row_keys = pd.row_keys as usize;
        let mut _row_headers = Vec::new();
        for _ in 0..pd.rows {
            let flag = buf.read_i32()?;
            if flag == USED_FLAG {
                let mut hdr = HashMap::new();
                for key in &row_key_names {
                    hdr.insert(key.clone(), buf.read_i32()?);
                }
                _row_headers.push(hdr);
            } else {
                // Skip unused row header
                buf.skip(num_row_keys * BYTES_PER_WORD);
            }
        }

        // ── 9. Read column headers ────────────────────────────────────
        buf.jump(word_to_pos(pd.column_headers_ptr as usize));
        let num_col_keys = pd.column_keys as usize;
        let mut column_headers: Vec<ColumnHeader> = Vec::new();
        let mut col_used_indices: Vec<usize> = Vec::new();

        for col_idx in 0..(pd.columns as usize) {
            let flag = buf.read_i32()?;
            if flag == USED_FLAG {
                let mut values = HashMap::new();
                for key in &column_key_names {
                    values.insert(key.clone(), buf.read_i32()?);
                }
                column_headers.push(ColumnHeader { values });
                col_used_indices.push(col_idx);
            } else {
                buf.skip(num_col_keys * BYTES_PER_WORD);
            }
        }

        // ── 10. Build grid info from column headers ───────────────────
        // Column keys for grid files are typically:
        //   GDT1, GTM1, GDT2, GTM2, GLV1, GLV2, GVCD, GPM1, GPM2, GPM3
        // The GPM* keys store the parameter name as integer-encoded 4-char strings.

        let mut grids = Vec::new();

        // Determine if GPM keys are stored as character (4-byte strings) or integer.
        // In GEMPAK grid files, GPM1/GPM2/GPM3 are 4-byte character keys.
        // The column_key_names tell us which are character vs integer.
        // For grids: GDT1=datetime, GTM1=ftime, GDT2=datetime, GTM2=ftime,
        //            GLV1=level, GLV2=level, GVCD=vertical coord,
        //            GPM1/GPM2/GPM3=parameter name characters.

        // Track which column key indices are the string-type GPM fields.
        let _string_keys: Vec<&str> = vec!["GPM1", "GPM2", "GPM3"];

        for (grid_idx, col_hdr) in column_headers.iter().enumerate() {
            // Extract parameter name from GPM1+GPM2+GPM3
            let parameter = col_hdr.get_param_string();

            // Extract datetime
            let gdt1 = col_hdr.get("GDT1");
            let time_str = convert_dattim(gdt1);

            let gdt2 = col_hdr.get("GDT2");
            let time2_str = convert_dattim(gdt2);

            // Forecast time
            let gtm1 = col_hdr.get("GTM1");
            let (ftype, fhours, fminutes) = convert_ftime(gtm1);
            let forecast_type = forecast_type_name(ftype).to_string();
            let forecast_time = format!("{:03}:{:02}", fhours, fminutes);

            // Levels
            let level1 = col_hdr.get("GLV1");
            let level2 = col_hdr.get("GLV2");

            // Vertical coordinate
            let gvcd = col_hdr.get("GVCD");
            let coordinate = vertical_coord_name(gvcd);

            // ── Extract grid data ─────────────────────────────────────
            let icol = col_used_indices[grid_idx];
            let mut grid_data: Vec<f64> = Vec::new();
            let mut data_found = false;

            for (iprt, part) in parts.iter().enumerate() {
                // Calculate data pointer
                let pointer = (pd.data_block_ptr as usize)
                    + (0 * pd.columns as usize * num_parts)   // irow = 0 for grids
                    + (icol * num_parts + iprt);

                if word_to_pos(pointer) >= buf.data.len() {
                    continue;
                }

                buf.jump(word_to_pos(pointer));
                let data_ptr = match buf.read_i32() {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                if data_ptr <= 0 || word_to_pos(data_ptr as usize) >= buf.data.len() {
                    continue;
                }

                buf.jump(word_to_pos(data_ptr as usize));
                let data_header_length = match buf.read_i32() {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                if data_header_length <= 0 {
                    continue;
                }

                let data_header_start = buf.pos;

                // Skip past the part header to reach the packing type word.
                // MetPy: jump_to(data_header, _word_to_position(part.header_length + 1))
                //   where _word_to_position(w) = w*4 - 4, so offset = header_length * 4
                buf.jump(data_header_start + (part.header_length as usize) * BYTES_PER_WORD);

                let packing_int = match buf.read_i32() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let packing_type = PackingType::from_i32(packing_int);

                match unpack_grid(
                    &mut buf,
                    packing_type,
                    data_header_length,
                    part.header_length,
                    kx,
                    ky,
                    pd.missing_float,
                ) {
                    Ok(Some(d)) => {
                        grid_data = d;
                        data_found = true;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        // Log warning but continue to next grid
                        eprintln!(
                            "Warning: failed to unpack grid {} ({}): {}",
                            parameter, packing_int, e
                        );
                    }
                }
            }

            let level_f64 = if level1 >= 0 { level1 as f64 } else { -1.0 };
            let level2_f64 = if level2 >= 0 { level2 as f64 } else { -1.0 };

            grids.push(GempakGridRecord {
                grid_number: grid_idx,
                parameter,
                level: level_f64,
                level2: level2_f64,
                coordinate,
                time: time_str,
                time2: time2_str,
                forecast_type,
                forecast_time,
                data: if data_found { grid_data } else { Vec::new() },
            });
        }

        Ok(GempakGrid {
            source,
            grid_type: "grid".to_string(),
            nx: kx,
            ny: ky,
            navigation,
            analysis,
            grids,
        })
    }

    /// Return a summary of all grids in the file.
    pub fn grid_info(&self) -> Vec<String> {
        self.grids
            .iter()
            .map(|g| {
                format!(
                    "Grid #{}: param={} level={} coord={} time={} ftype={} ftime={}",
                    g.grid_number,
                    g.parameter,
                    g.level,
                    g.coordinate,
                    g.time,
                    g.forecast_type,
                    g.forecast_time,
                )
            })
            .collect()
    }

    /// Find grids matching a parameter name (case-insensitive).
    pub fn find_grids(&self, parameter: &str) -> Vec<&GempakGridRecord> {
        let param_upper = parameter.to_uppercase();
        self.grids
            .iter()
            .filter(|g| g.parameter.to_uppercase() == param_upper)
            .collect()
    }

    /// Get a specific grid by parameter and level.
    pub fn get_grid(&self, parameter: &str, level: f64) -> Option<&GempakGridRecord> {
        let param_upper = parameter.to_uppercase();
        self.grids.iter().find(|g| {
            g.parameter.to_uppercase() == param_upper
                && (g.level - level).abs() < 0.01
        })
    }
}

// ── Grid unpacking ─────────────────────────────────────────────────────

/// Unpack grid data from the buffer according to the packing type.
fn unpack_grid(
    buf: &mut GemBuf,
    packing_type: PackingType,
    data_header_length: i32,
    part_header_length: i32,
    kx: usize,
    ky: usize,
    missing_float: f32,
) -> Result<Option<Vec<f64>>, String> {
    match packing_type {
        PackingType::None => {
            unpack_none(buf, data_header_length, part_header_length, kx, ky)
        }
        PackingType::Grib | PackingType::Dec => {
            unpack_grib(buf, data_header_length, part_header_length, kx, ky, missing_float)
        }
        PackingType::Diff => {
            unpack_diff(buf, data_header_length, part_header_length, kx, ky, missing_float)
        }
        PackingType::Nmc => {
            Err("NMC packing is not yet supported".to_string())
        }
        PackingType::Grib2 => {
            Err("GRIB2 packing is not yet supported".to_string())
        }
        PackingType::Unknown(v) => {
            Err(format!("Unknown packing type: {}", v))
        }
    }
}

/// Unpack raw (unpacked) float32 grid data.
fn unpack_none(
    buf: &mut GemBuf,
    data_header_length: i32,
    part_header_length: i32,
    kx: usize,
    ky: usize,
) -> Result<Option<Vec<f64>>, String> {
    // lendat = data_header_length - part.header_length - 1
    let lendat = data_header_length - part_header_length - 1;
    if lendat <= 1 {
        return Ok(None);
    }

    let mut data = Vec::with_capacity(lendat as usize);
    for _ in 0..lendat {
        data.push(buf.read_f32()? as f64);
    }

    // Truncate or extend to kx*ky
    data.resize(ky * kx, f64::NAN);

    Ok(Some(data))
}

/// Unpack GRIB/DEC packed grid data.
///
/// This implements the standard GEMPAK grid packing where each value is stored
/// as: value = reference + (packed_integer * scale)
fn unpack_grib(
    buf: &mut GemBuf,
    data_header_length: i32,
    part_header_length: i32,
    _kx: usize,
    _ky: usize,
    missing_float: f32,
) -> Result<Option<Vec<f64>>, String> {
    // Integer metadata: bits, missing_flag, kxky
    let bits = buf.read_i32()?;
    let missing_flag = buf.read_i32()?;
    let kxky = buf.read_i32()?;

    // Real metadata: reference, scale
    let reference = buf.read_f32()?;
    let scale = buf.read_f32()?;

    // lendat = data_header_length - part.header_length - 6
    let lendat = data_header_length - part_header_length - 6;
    if lendat <= 1 {
        return Ok(None);
    }

    // Read packed integer buffer
    let mut packed = Vec::with_capacity(lendat as usize);
    for _ in 0..lendat {
        packed.push(buf.read_i32()?);
    }

    let grid_size = kxky as usize;
    let imax = if bits < 32 {
        (1i64 << bits) - 1
    } else {
        i64::MAX
    };
    let mut grid = vec![0.0f64; grid_size];

    let mut ibit: i32 = 1;
    let mut iword: usize = 0;

    for cell in 0..grid_size {
        let jshft = bits + ibit - 33;
        let mut idat = fortran_ishift(packed[iword], jshft) as i64 & imax;

        if jshft > 0 {
            let jshft2 = jshft - 32;
            if iword + 1 < packed.len() {
                let idat2 = fortran_ishift(packed[iword + 1], jshft2) as i64 & imax;
                idat |= idat2;
            }
        }

        if idat == imax && missing_flag != 0 {
            grid[cell] = missing_float as f64;
        } else {
            grid[cell] = (reference as f64) + (idat as f64) * (scale as f64);
        }

        ibit += bits;
        if ibit > 32 {
            ibit -= 32;
            iword += 1;
        }
    }

    Ok(Some(grid))
}

/// Unpack DIFF (second-order differencing) packed grid data.
///
/// This uses first-value references plus row/column differencing to reconstruct
/// the grid, which is efficient for smoothly varying fields.
fn unpack_diff(
    buf: &mut GemBuf,
    data_header_length: i32,
    part_header_length: i32,
    kx: usize,
    ky: usize,
    missing_float: f32,
) -> Result<Option<Vec<f64>>, String> {
    // Integer metadata: bits, missing_flag, kxky, kx
    let bits = buf.read_i32()?;
    let missing_flag = buf.read_i32()?;
    let _kxky = buf.read_i32()?;
    let _kx_check = buf.read_i32()?;

    // Real metadata: reference, scale, diffmin
    let reference = buf.read_f32()?;
    let scale = buf.read_f32()?;
    let diffmin = buf.read_f32()?;

    // lendat = data_header_length - part.header_length - 8
    let lendat = data_header_length - part_header_length - 8;
    if lendat <= 1 {
        return Ok(None);
    }

    let mut packed = Vec::with_capacity(lendat as usize);
    for _ in 0..lendat {
        packed.push(buf.read_i32()?);
    }

    let imiss = if bits < 32 {
        (1i64 << bits) - 1
    } else {
        i64::MAX
    };
    let mut grid = vec![vec![0.0f64; kx]; ky];

    let mut iword: usize = 0;
    let mut ibit: i32 = 1;
    let mut first = true;
    let mut psav = 0.0f64;
    let mut plin = 0.0f64;

    for j in 0..ky {
        let mut line = false;
        for i in 0..kx {
            // Extract packed value
            let jshft = bits + ibit - 33;
            let mut idat = fortran_ishift(packed[iword], jshft) as i64 & imiss;

            if jshft > 0 {
                let jshft2 = jshft - 32;
                if iword + 1 < packed.len() {
                    let idat2 = fortran_ishift(packed[iword + 1], jshft2) as i64 & imiss;
                    idat |= idat2;
                }
            }

            ibit += bits;
            if ibit > 32 {
                ibit -= 32;
                iword += 1;
            }

            if missing_flag != 0 && idat == imiss {
                grid[j][i] = missing_float as f64;
            } else if first {
                grid[j][i] = reference as f64;
                psav = reference as f64;
                plin = reference as f64;
                line = true;
                first = false;
            } else if !line {
                grid[j][i] = plin + (diffmin as f64) + (idat as f64) * (scale as f64);
                line = true;
                plin = grid[j][i];
            } else {
                grid[j][i] = psav + (diffmin as f64) + (idat as f64) * (scale as f64);
            }
            if !(missing_flag != 0 && idat == imiss) && !first {
                psav = grid[j][i];
            }
        }
    }

    // Flatten to row-major order
    let flat: Vec<f64> = grid.into_iter().flat_map(|row| row.into_iter()).collect();
    Ok(Some(flat))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal synthetic GEMPAK grid file in memory.
    ///
    /// Layout (all pointers are 1-indexed word positions):
    ///
    ///   Offset  Content
    ///   ------  -------
    ///   0..28   GEMPAK header string (28 bytes)
    ///   28..    Product description (22 i32 + 12 skip + 1 f32 = 100 bytes)
    ///   ...     File keys (1 header: name+length+type = 12 bytes)
    ///   ...     Navigation block (size word + 256 words)
    ///   ...     Analysis block (size word = 4 bytes, size=0 means no block)
    ///   ...     Row keys
    ///   ...     Column keys
    ///   ...     Parts
    ///   ...     Row headers
    ///   ...     Column headers
    ///   ...     Data management
    ///   ...     Data block
    fn build_test_file(kx: usize, ky: usize, value: f32) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();

        // Helper closures
        let push_i32 = |out: &mut Vec<u8>, v: i32| {
            out.extend_from_slice(&v.to_be_bytes());
        };
        let push_f32 = |out: &mut Vec<u8>, v: f32| {
            out.extend_from_slice(&v.to_be_bytes());
        };

        // 0..28: GEMPAK header
        out.extend_from_slice(GEMPAK_HEADER);

        // Product description - we need to calculate pointers first
        // PD is at word 8 (byte 28), length = 20 ints + 3 skip words + 1 float = 24 words
        let pd_start_word: usize = 8; // byte 28 / 4 + 1 = 8

        // Layout plan (word positions, 1-indexed):
        let file_keys_ptr = pd_start_word + 24;  // after PD
        let file_keys_size = 3; // 1 header entry: name(1) + length(1) + type(1)
        let navb_ptr = file_keys_ptr + file_keys_size;
        let navb_size_word = 1; // size indicator
        let navb_content = 256; // standard nav block
        let anlb_ptr = navb_ptr + navb_size_word + navb_content;
        let anlb_size_word = 1;
        let row_keys_ptr = anlb_ptr + anlb_size_word;
        let row_keys_count = 0; // grid files have 0 row keys
        let col_keys_ptr = row_keys_ptr; // no row keys to skip
        let col_keys_count = 10; // GDT1 GTM1 GDT2 GTM2 GLV1 GLV2 GVCD GPM1 GPM2 GPM3
        let parts_ptr = col_keys_ptr + col_keys_count;
        let parts_count = 1;
        let parts_total = parts_count * 4; // 4 arrays: name, hdr_len, dtype, parm_count
        let row_headers_ptr = parts_ptr + parts_total;
        let rows = 1;
        let row_header_size = 1 + row_keys_count; // flag + keys per row
        let col_headers_ptr = row_headers_ptr + rows * row_header_size;
        let columns = 1;
        let col_header_size = 1 + col_keys_count; // flag + keys per column
        let data_mgmt_ptr = col_headers_ptr + columns * col_header_size;
        let data_mgmt_size = 32; // 4 + 28 free words
        let data_block_ptr = data_mgmt_ptr + data_mgmt_size;
        let data_block_entries = rows * columns * parts_count; // pointer table
        let data_start = data_block_ptr + data_block_entries;
        // Data: length(1) + header(1 word for part) + packing_type(1) + grid data
        let grid_data_length = (kx * ky) as i32;
        let data_content_length = 1 + 1 + grid_data_length; // part_hdr + packing + data

        // Version (=1 signals endianness)
        push_i32(&mut out, 1);
        // file_headers
        push_i32(&mut out, 1);
        // file_keys_ptr
        push_i32(&mut out, file_keys_ptr as i32);
        // rows
        push_i32(&mut out, rows as i32);
        // row_keys
        push_i32(&mut out, row_keys_count as i32);
        // row_keys_ptr
        push_i32(&mut out, row_keys_ptr as i32);
        // row_headers_ptr
        push_i32(&mut out, row_headers_ptr as i32);
        // columns
        push_i32(&mut out, columns as i32);
        // column_keys
        push_i32(&mut out, col_keys_count as i32);
        // column_keys_ptr
        push_i32(&mut out, col_keys_ptr as i32);
        // column_headers_ptr
        push_i32(&mut out, col_headers_ptr as i32);
        // parts
        push_i32(&mut out, parts_count as i32);
        // parts_ptr
        push_i32(&mut out, parts_ptr as i32);
        // data_mgmt_ptr
        push_i32(&mut out, data_mgmt_ptr as i32);
        // data_mgmt_length
        push_i32(&mut out, data_mgmt_size as i32);
        // data_block_ptr
        push_i32(&mut out, data_block_ptr as i32);
        // file_type = 3 (grid)
        push_i32(&mut out, 3);
        // data_source = 0 (model)
        push_i32(&mut out, 0);
        // machine_type
        push_i32(&mut out, 0);
        // missing_int
        push_i32(&mut out, -9999);
        // 12 bytes padding
        out.extend_from_slice(&[0u8; 12]);
        // missing_float
        push_f32(&mut out, MISSING_FLOAT);

        // File keys: 1 header entry
        out.extend_from_slice(b"NAVB");
        push_i32(&mut out, 256); // length in words
        push_i32(&mut out, 1);   // type

        // Navigation block
        push_i32(&mut out, 256); // size word
        push_f32(&mut out, 0.0); // grid_definition_type
        out.extend_from_slice(b"CED\0"); // projection (4 bytes)
        push_f32(&mut out, 1.0); // left grid number
        push_f32(&mut out, 1.0); // bottom grid number
        push_f32(&mut out, kx as f32); // right grid number (= kx)
        push_f32(&mut out, ky as f32); // top grid number (= ky)
        push_f32(&mut out, 20.0); // lower_left_lat
        push_f32(&mut out, -120.0); // lower_left_lon
        push_f32(&mut out, 50.0); // upper_right_lat
        push_f32(&mut out, -60.0); // upper_right_lon
        push_f32(&mut out, 0.0); // angle1
        push_f32(&mut out, -90.0); // angle2
        push_f32(&mut out, 0.0); // angle3
        // Pad to 256 words (1024 bytes) total
        let nav_written = 12 * 4 + 4; // 12 floats + projection
        let nav_pad = 256 * 4 - nav_written;
        out.extend_from_slice(&vec![0u8; nav_pad]);

        // Analysis block: size=0 (none)
        push_i32(&mut out, 0);

        // Row keys (none for grid files)
        // (col_keys_ptr and row_keys_ptr are the same since row_keys_count=0)

        // Column keys: 10 key names
        for name in &["GDT1", "GTM1", "GDT2", "GTM2", "GLV1", "GLV2", "GVCD", "GPM1", "GPM2", "GPM3"] {
            let mut key_bytes = [b' '; 4];
            for (i, b) in name.bytes().enumerate() {
                if i < 4 { key_bytes[i] = b; }
            }
            out.extend_from_slice(&key_bytes);
        }

        // Parts: 1 part
        // name array
        out.extend_from_slice(b"GRID");
        // header_length array
        push_i32(&mut out, 1); // 1 word header
        // data_type array
        push_i32(&mut out, 5); // DataTypes.grid = 5
        // parameter_count array
        push_i32(&mut out, 0);

        // Row headers: 1 row, flag = USED
        push_i32(&mut out, USED_FLAG);
        // (no row keys to write)

        // Column headers: 1 column, flag = USED
        push_i32(&mut out, USED_FLAG);
        // GDT1: encode 250101 (2025-01-01) = YYMMDD
        push_i32(&mut out, 250101);
        // GTM1: forecast type 1 (forecast) * 100000 + hours*100 = 100000 + 0
        push_i32(&mut out, 0); // analysis, 0 hours
        // GDT2
        push_i32(&mut out, 0);
        // GTM2
        push_i32(&mut out, 0);
        // GLV1: level = 500 (hPa)
        push_i32(&mut out, 500);
        // GLV2
        push_i32(&mut out, -1);
        // GVCD: 1 = PRES
        push_i32(&mut out, 1);
        // GPM1: "HGHT" as big-endian i32
        out.extend_from_slice(b"HGHT");
        // GPM2: spaces
        out.extend_from_slice(b"    ");
        // GPM3: spaces
        out.extend_from_slice(b"    ");

        // Data management (32 words)
        for _ in 0..32 {
            push_i32(&mut out, 0);
        }

        // Data block: pointer table (1 entry)
        push_i32(&mut out, data_start as i32);

        // Data content at data_start
        // Length word (total words of data including header)
        push_i32(&mut out, data_content_length);
        // Part header (1 word, content doesn't matter)
        push_i32(&mut out, 0);
        // Packing type = 0 (none)
        push_i32(&mut out, 0);
        // Grid data: kx*ky floats
        for _ in 0..(kx * ky) {
            push_f32(&mut out, value);
        }

        out
    }

    #[test]
    fn test_header_validation() {
        let bad = vec![0u8; 100];
        assert!(GempakGrid::from_bytes(&bad).is_err());
    }

    #[test]
    fn test_non_grid_file_type() {
        let mut data = build_test_file(3, 3, 1.0);
        // Overwrite file_type (word 17, byte offset 28 + 16*4 = 92..96) with 1 (surface)
        let ft_offset = 28 + 16 * 4;
        let ft_bytes = 1i32.to_be_bytes();
        data[ft_offset..ft_offset + 4].copy_from_slice(&ft_bytes);
        let result = GempakGrid::from_bytes(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("file_type=1"));
    }

    #[test]
    fn test_parse_synthetic_grid() {
        let data = build_test_file(4, 3, 5500.0);
        let gem = GempakGrid::from_bytes(&data).expect("failed to parse synthetic GEMPAK file");

        assert_eq!(gem.nx, 4);
        assert_eq!(gem.ny, 3);
        assert_eq!(gem.grid_type, "grid");
        assert_eq!(gem.source, "model");
        assert_eq!(gem.grids.len(), 1);

        let g = &gem.grids[0];
        assert_eq!(g.parameter, "HGHT");
        assert!((g.level - 500.0).abs() < 0.01);
        assert_eq!(g.coordinate, "PRES");
        assert_eq!(g.data.len(), 12); // 4*3

        // All values should be 5500.0
        for v in &g.data {
            assert!((*v - 5500.0).abs() < 0.01, "expected 5500.0, got {}", v);
        }
    }

    #[test]
    fn test_navigation_info() {
        let data = build_test_file(10, 10, 0.0);
        let gem = GempakGrid::from_bytes(&data).unwrap();

        let nav = gem.navigation.as_ref().expect("navigation should be present");
        assert_eq!(nav.projection, "CED");
        assert!((nav.lower_left_lat - 20.0).abs() < 0.01);
        assert!((nav.lower_left_lon - (-120.0)).abs() < 0.01);
        assert!((nav.upper_right_lat - 50.0).abs() < 0.01);
        assert!((nav.upper_right_lon - (-60.0)).abs() < 0.01);
    }

    #[test]
    fn test_grid_info() {
        let data = build_test_file(5, 5, 100.0);
        let gem = GempakGrid::from_bytes(&data).unwrap();
        let info = gem.grid_info();
        assert_eq!(info.len(), 1);
        assert!(info[0].contains("HGHT"));
        assert!(info[0].contains("500"));
    }

    #[test]
    fn test_find_grids() {
        let data = build_test_file(5, 5, 100.0);
        let gem = GempakGrid::from_bytes(&data).unwrap();
        assert_eq!(gem.find_grids("HGHT").len(), 1);
        assert_eq!(gem.find_grids("TMPK").len(), 0);
        assert_eq!(gem.find_grids("hght").len(), 1); // case insensitive
    }

    #[test]
    fn test_get_grid() {
        let data = build_test_file(5, 5, 273.15);
        let gem = GempakGrid::from_bytes(&data).unwrap();
        let g = gem.get_grid("HGHT", 500.0).expect("should find grid");
        assert_eq!(g.data.len(), 25);
        assert!(gem.get_grid("HGHT", 850.0).is_none());
    }

    #[test]
    fn test_convert_dattim() {
        assert_eq!(convert_dattim(250101), "20250101");
        assert_eq!(convert_dattim(990615), "19990615");
        assert_eq!(convert_dattim(0), "");
    }

    #[test]
    fn test_convert_ftime() {
        let (ftype, hours, minutes) = convert_ftime(100600);
        assert_eq!(ftype, 1); // forecast
        assert_eq!(hours, 6);
        assert_eq!(minutes, 0);

        let (ftype, hours, minutes) = convert_ftime(0);
        assert_eq!(ftype, 0); // analysis
        assert_eq!(hours, 0);
        assert_eq!(minutes, 0);
    }

    #[test]
    fn test_vertical_coord_name() {
        assert_eq!(vertical_coord_name(0), "NONE");
        assert_eq!(vertical_coord_name(1), "PRES");
        assert_eq!(vertical_coord_name(3), "HGHT");
    }

    #[test]
    fn test_fortran_ishift() {
        // Left shift
        assert_eq!(fortran_ishift(1, 4), 16);
        // Right shift
        assert_eq!(fortran_ishift(16, -4), 1);
        // Zero shift
        assert_eq!(fortran_ishift(42, 0), 42);
    }
}
