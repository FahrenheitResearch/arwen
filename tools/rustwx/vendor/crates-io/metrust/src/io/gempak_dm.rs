//! Shared GEMPAK Data Management (DM) binary file reader.
//!
//! GEMPAK files (grid, sounding, surface) share a common DM binary format:
//! - Big-endian by default (auto-detected via machine-type word)
//! - 28-byte magic header: "GEMPAK DATA MANAGEMENT FILE "
//! - Product description block (version, row/col counts, pointers, file type, ...)
//! - Row keys, column keys, parts, parameters
//! - Row headers, column headers
//! - Data management block (free-list bookkeeping)
//! - Data blocks (variable-length, pointed to from row x col x part grid)
//!
//! Data packing uses the GEMPAK "realpack" scheme:
//!   reference + scale + bit-packed integers.

use std::io::Read;

// ── Constants ───────────────────────────────────────────────────────────

pub const BYTES_PER_WORD: usize = 4;
pub const USED_FLAG: i32 = 9999;
pub const GEMPAK_HEADER: &str = "GEMPAK DATA MANAGEMENT FILE ";
pub const MISSING_FLOAT: f32 = -9999.0;
pub const MISSING_INT: i32 = -9999;

// ── Enums ───────────────────────────────────────────────────────────────

/// GEMPAK file type stored in the product description.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    Surface = 1,
    Sounding = 2,
    Grid = 3,
    Unknown = 0,
}

impl From<i32> for FileType {
    fn from(v: i32) -> Self {
        match v {
            1 => FileType::Surface,
            2 => FileType::Sounding,
            3 => FileType::Grid,
            _ => FileType::Unknown,
        }
    }
}

/// Data storage type for a part's data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Real = 1,
    Integer = 2,
    Character = 3,
    RealPack = 4,
    Grid = 5,
    Unknown = 0,
}

impl From<i32> for DataType {
    fn from(v: i32) -> Self {
        match v {
            1 => DataType::Real,
            2 => DataType::Integer,
            3 => DataType::Character,
            4 => DataType::RealPack,
            5 => DataType::Grid,
            _ => DataType::Unknown,
        }
    }
}

// ── Product Description ─────────────────────────────────────────────────

/// The fixed metadata block found in every GEMPAK DM file.
#[derive(Debug, Clone)]
pub struct ProductDescription {
    pub version: i32,
    pub file_headers: i32,
    pub file_keys_ptr: i32,
    pub rows: i32,
    pub row_keys: i32,
    pub row_keys_ptr: i32,
    pub row_headers_ptr: i32,
    pub columns: i32,
    pub column_keys: i32,
    pub column_keys_ptr: i32,
    pub column_headers_ptr: i32,
    pub parts: i32,
    pub parts_ptr: i32,
    pub data_mgmt_ptr: i32,
    pub data_mgmt_length: i32,
    pub data_block_ptr: i32,
    pub file_type: FileType,
    pub data_source: i32,
    pub machine_type: i32,
    pub missing_int: i32,
    pub missing_float: f32,
}

/// A "part" describes one logical section of data (e.g. SNDT, SFDT, TXPB, ...).
#[derive(Debug, Clone)]
pub struct PartInfo {
    pub name: String,
    pub header_length: i32,
    pub data_type: DataType,
    pub parameter_count: i32,
}

/// Parameter packing metadata for one part.
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    pub names: Vec<String>,
    pub scales: Vec<i32>,
    pub offsets: Vec<i32>,
    pub bits: Vec<i32>,
}

// ── Byte-order aware reader ─────────────────────────────────────────────

/// A cursor over a byte buffer with selectable endianness.
pub struct DmBuffer {
    pub data: Vec<u8>,
    pub pos: usize,
    pub big_endian: bool,
}

impl DmBuffer {
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            pos: 0,
            big_endian: true, // GEMPAK default
        }
    }

    /// Jump to an absolute byte offset.
    pub fn jump_to(&mut self, offset: usize) {
        self.pos = offset;
    }

    /// Convert a 1-based word pointer to a byte offset.
    pub fn word_to_offset(word: i32) -> usize {
        ((word - 1) as usize) * BYTES_PER_WORD
    }

    /// Read a signed 32-bit integer.
    pub fn read_i32(&mut self) -> Result<i32, String> {
        if self.pos + 4 > self.data.len() {
            return Err(format!(
                "read_i32: unexpected EOF at offset {} (len={})",
                self.pos,
                self.data.len()
            ));
        }
        let b = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        if self.big_endian {
            Ok(i32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        } else {
            Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        }
    }

    /// Read an unsigned 32-bit integer.
    pub fn read_u32(&mut self) -> Result<u32, String> {
        if self.pos + 4 > self.data.len() {
            return Err(format!(
                "read_u32: unexpected EOF at offset {} (len={})",
                self.pos,
                self.data.len()
            ));
        }
        let b = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        if self.big_endian {
            Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        } else {
            Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        }
    }

    /// Read a 32-bit IEEE-754 float.
    pub fn read_f32(&mut self) -> Result<f32, String> {
        if self.pos + 4 > self.data.len() {
            return Err(format!(
                "read_f32: unexpected EOF at offset {} (len={})",
                self.pos,
                self.data.len()
            ));
        }
        let b = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        if self.big_endian {
            Ok(f32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        } else {
            Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        }
    }

    /// Read `n` bytes as a UTF-8 string, trimming trailing whitespace/nulls.
    pub fn read_string(&mut self, n: usize) -> Result<String, String> {
        if self.pos + n > self.data.len() {
            return Err(format!(
                "read_string: unexpected EOF at offset {} requesting {} bytes (len={})",
                self.pos,
                n,
                self.data.len()
            ));
        }
        let b = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(String::from_utf8_lossy(b).trim_end_matches(|c: char| c.is_whitespace() || c == '\0').to_string())
    }

    /// Skip `n` bytes.
    pub fn skip(&mut self, n: usize) {
        self.pos += n;
    }

    /// Detect byte order from the machine-type word right after the 28-byte header.
    /// The first 4 bytes after the header encode the integer 1 in the file's native byte order.
    pub fn detect_byte_order(&mut self) {
        let saved = self.pos;
        // machine-type word is at offset 28 (right after the GEMPAK header)
        self.pos = 28;
        let be_val = i32::from_be_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        // If big-endian reading gives 1, the file is big-endian
        self.big_endian = be_val == 1;
        self.pos = saved;
    }
}

// ── Core DM file reader ─────────────────────────────────────────────────

/// Parsed GEMPAK DM file common structures.
#[derive(Debug, Clone)]
pub struct DmFile {
    pub prod_desc: ProductDescription,
    pub row_keys: Vec<String>,
    pub column_keys: Vec<String>,
    pub parts: Vec<PartInfo>,
    pub parameters: Vec<ParameterInfo>,
    /// Raw file bytes, kept for data extraction.
    raw: Vec<u8>,
    big_endian: bool,
}

impl DmFile {
    /// Parse a GEMPAK DM file from a path.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open {}: {}", path, e))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| format!("Failed to read {}: {}", path, e))?;
        Self::from_bytes(data)
    }

    /// Parse a GEMPAK DM file from raw bytes.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, String> {
        if data.len() < 32 {
            return Err("File too small to be a GEMPAK file".to_string());
        }

        let mut buf = DmBuffer::new(data.clone());

        // Verify GEMPAK header (28 bytes)
        let header = buf.read_string(28)?;
        if header != GEMPAK_HEADER.trim() {
            return Err(format!(
                "Not a GEMPAK file: expected '{}', got '{}'",
                GEMPAK_HEADER.trim(),
                header
            ));
        }

        // Detect byte order
        buf.detect_byte_order();

        // Now read product description starting at offset 32
        // (28-byte header + 4-byte machine type word)
        buf.jump_to(28);
        // Read the machine-type word (used above for byte-order detection)
        let _machine_detect = buf.read_i32()?;

        // Product description fields
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
        let file_type_val = buf.read_i32()?;
        let data_source = buf.read_i32()?;
        let machine_type = buf.read_i32()?;
        let missing_int = buf.read_i32()?;
        buf.skip(12); // 12 bytes padding
        let missing_float = buf.read_f32()?;

        let prod_desc = ProductDescription {
            version,
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
            data_mgmt_ptr,
            data_mgmt_length,
            data_block_ptr,
            file_type: FileType::from(file_type_val),
            data_source,
            machine_type,
            missing_int,
            missing_float,
        };

        // Read row keys
        buf.jump_to(DmBuffer::word_to_offset(prod_desc.row_keys_ptr));
        let mut row_key_names = Vec::new();
        for _ in 0..prod_desc.row_keys {
            let key = buf.read_string(4)?;
            row_key_names.push(key);
        }

        // Read column keys
        buf.jump_to(DmBuffer::word_to_offset(prod_desc.column_keys_ptr));
        let mut col_key_names = Vec::new();
        for _ in 0..prod_desc.column_keys {
            let key = buf.read_string(4)?;
            col_key_names.push(key);
        }

        // Read parts information
        // Parts are stored in a column-interleaved layout:
        //   parts_ptr + 0: name_1, name_2, ..., name_N
        //   parts_ptr + N: header_length_1, header_length_2, ..., header_length_N
        //   parts_ptr + 2N: data_type_1, data_type_2, ..., data_type_N
        //   parts_ptr + 3N: parameter_count_1, parameter_count_2, ..., parameter_count_N
        let nparts = prod_desc.parts as usize;
        let parts_base = DmBuffer::word_to_offset(prod_desc.parts_ptr);

        let mut part_names = Vec::with_capacity(nparts);
        buf.jump_to(parts_base);
        for _ in 0..nparts {
            let name = buf.read_string(4)?;
            part_names.push(name);
        }

        let mut header_lengths = Vec::with_capacity(nparts);
        buf.jump_to(parts_base + nparts * BYTES_PER_WORD);
        for _ in 0..nparts {
            header_lengths.push(buf.read_i32()?);
        }

        let mut data_types = Vec::with_capacity(nparts);
        buf.jump_to(parts_base + 2 * nparts * BYTES_PER_WORD);
        for _ in 0..nparts {
            data_types.push(buf.read_i32()?);
        }

        let mut param_counts = Vec::with_capacity(nparts);
        buf.jump_to(parts_base + 3 * nparts * BYTES_PER_WORD);
        for _ in 0..nparts {
            param_counts.push(buf.read_i32()?);
        }

        let mut part_infos = Vec::with_capacity(nparts);
        for i in 0..nparts {
            part_infos.push(PartInfo {
                name: part_names[i].clone(),
                header_length: header_lengths[i],
                data_type: DataType::from(data_types[i]),
                parameter_count: param_counts[i],
            });
        }

        // Read parameter attributes (name, scale, offset, bits)
        // These follow the parts block:
        //   parts_ptr + 4*N words: param names for all parts
        //   then scales, then offsets, then bits
        let params_base = parts_base + 4 * nparts * BYTES_PER_WORD;
        let total_params: usize = param_counts.iter().map(|c| *c as usize).sum();

        // Names
        let mut all_names = Vec::with_capacity(total_params);
        buf.jump_to(params_base);
        for _ in 0..total_params {
            let name = buf.read_string(4)?;
            all_names.push(name);
        }

        // Scales
        let mut all_scales = Vec::with_capacity(total_params);
        buf.jump_to(params_base + total_params * BYTES_PER_WORD);
        for _ in 0..total_params {
            all_scales.push(buf.read_i32()?);
        }

        // Offsets
        let mut all_offsets = Vec::with_capacity(total_params);
        buf.jump_to(params_base + 2 * total_params * BYTES_PER_WORD);
        for _ in 0..total_params {
            all_offsets.push(buf.read_i32()?);
        }

        // Bits
        let mut all_bits = Vec::with_capacity(total_params);
        buf.jump_to(params_base + 3 * total_params * BYTES_PER_WORD);
        for _ in 0..total_params {
            all_bits.push(buf.read_i32()?);
        }

        // Split parameters by part
        let mut parameters = Vec::with_capacity(nparts);
        let mut offset = 0usize;
        for i in 0..nparts {
            let count = param_counts[i] as usize;
            parameters.push(ParameterInfo {
                names: all_names[offset..offset + count].to_vec(),
                scales: all_scales[offset..offset + count].to_vec(),
                offsets: all_offsets[offset..offset + count].to_vec(),
                bits: all_bits[offset..offset + count].to_vec(),
            });
            offset += count;
        }

        Ok(DmFile {
            prod_desc,
            row_keys: row_key_names,
            column_keys: col_key_names,
            parts: part_infos,
            parameters,
            raw: data,
            big_endian: buf.big_endian,
        })
    }

    /// Create a buffer positioned at the start of the raw file data.
    pub fn buffer(&self) -> DmBuffer {
        let mut b = DmBuffer::new(self.raw.clone());
        b.big_endian = self.big_endian;
        b
    }

    /// Read a row header value from the station/time header area.
    /// Row headers start at `row_headers_ptr` and each row has a used-flag
    /// (1 word) followed by `row_keys` words of data.
    pub fn read_row_headers_raw(&self) -> Result<Vec<Option<Vec<i32>>>, String> {
        let mut buf = self.buffer();
        buf.jump_to(DmBuffer::word_to_offset(self.prod_desc.row_headers_ptr));
        let nkeys = self.prod_desc.row_keys as usize;
        let mut headers = Vec::new();
        for _ in 0..self.prod_desc.rows {
            let flag = buf.read_i32()?;
            if flag == USED_FLAG {
                let mut vals = Vec::with_capacity(nkeys);
                for _ in 0..nkeys {
                    vals.push(buf.read_i32()?);
                }
                headers.push(Some(vals));
            } else {
                // Skip the key words for unused rows
                buf.skip(nkeys * BYTES_PER_WORD);
                headers.push(None);
            }
        }
        Ok(headers)
    }

    /// Read column headers raw.
    pub fn read_column_headers_raw(&self) -> Result<Vec<Option<Vec<i32>>>, String> {
        let mut buf = self.buffer();
        buf.jump_to(DmBuffer::word_to_offset(self.prod_desc.column_headers_ptr));
        let nkeys = self.prod_desc.column_keys as usize;
        let mut headers = Vec::new();
        for _ in 0..self.prod_desc.columns {
            let flag = buf.read_i32()?;
            if flag == USED_FLAG {
                let mut vals = Vec::with_capacity(nkeys);
                for _ in 0..nkeys {
                    vals.push(buf.read_i32()?);
                }
                headers.push(Some(vals));
            } else {
                buf.skip(nkeys * BYTES_PER_WORD);
                headers.push(None);
            }
        }
        Ok(headers)
    }

    /// Read the data pointer for a given (row, col, part) triple.
    pub fn data_pointer(&self, row: usize, col: usize, part: usize) -> Result<i32, String> {
        let mut buf = self.buffer();
        let pointer_word = self.prod_desc.data_block_ptr
            + (row as i32 * self.prod_desc.columns * self.prod_desc.parts)
            + (col as i32 * self.prod_desc.parts + part as i32);
        buf.jump_to(DmBuffer::word_to_offset(pointer_word));
        buf.read_i32()
    }

    /// Read and unpack data for a (row, col, part) combination.
    /// Returns a vector of f64 values, with missing values set to NaN.
    pub fn read_data(&self, row: usize, col: usize, part_idx: usize) -> Result<Vec<f64>, String> {
        let data_ptr = self.data_pointer(row, col, part_idx)?;
        if data_ptr == 0 {
            return Ok(Vec::new());
        }

        let part = &self.parts[part_idx];
        let params = &self.parameters[part_idx];
        let mut buf = self.buffer();

        buf.jump_to(DmBuffer::word_to_offset(data_ptr));
        let data_header_length = buf.read_i32()?;
        // Skip past the part header
        let data_start = DmBuffer::word_to_offset(data_ptr) + (1 + part.header_length as usize) * BYTES_PER_WORD;
        buf.jump_to(data_start);
        let lendat = (data_header_length - part.header_length) as usize;

        if lendat == 0 {
            return Ok(Vec::new());
        }

        match part.data_type {
            DataType::Real => {
                let mut values = Vec::with_capacity(lendat);
                for _ in 0..lendat {
                    let v = buf.read_f32()? as f64;
                    if (v - self.prod_desc.missing_float as f64).abs() < 0.5 {
                        values.push(f64::NAN);
                    } else {
                        values.push(v);
                    }
                }
                Ok(values)
            }
            DataType::RealPack => {
                let mut packed = Vec::with_capacity(lendat);
                for _ in 0..lendat {
                    packed.push(buf.read_i32()?);
                }
                unpack_real(&packed, params, lendat, self.prod_desc.missing_float)
            }
            DataType::Character => {
                // Character data: read as raw bytes, return empty numeric vec
                // (handled specially by callers)
                Ok(Vec::new())
            }
            _ => Err(format!("Unsupported data type: {:?}", part.data_type)),
        }
    }

    /// Read character data for a (row, col, part) combination.
    pub fn read_char_data(&self, row: usize, col: usize, part_idx: usize) -> Result<Vec<String>, String> {
        let data_ptr = self.data_pointer(row, col, part_idx)?;
        if data_ptr == 0 {
            return Ok(Vec::new());
        }

        let part = &self.parts[part_idx];
        let mut buf = self.buffer();

        buf.jump_to(DmBuffer::word_to_offset(data_ptr));
        let data_header_length = buf.read_i32()?;
        let data_start = DmBuffer::word_to_offset(data_ptr) + (1 + part.header_length as usize) * BYTES_PER_WORD;
        buf.jump_to(data_start);
        let lendat = (data_header_length - part.header_length) as usize;

        let nparms = part.parameter_count as usize;
        let mut strings = Vec::with_capacity(nparms);
        if lendat > 0 {
            let total_bytes = lendat * BYTES_PER_WORD;
            let bytes_per_param = total_bytes / nparms;
            for _ in 0..nparms {
                let s = buf.read_string(bytes_per_param)?;
                strings.push(s);
            }
        }
        Ok(strings)
    }
}

// ── GEMPAK real data unpacking (DP_UNPK) ────────────────────────────────

/// Fortran-style signed bit shift (32-bit).
fn fortran_ishift(i: i32, shift: i32) -> i32 {
    if shift >= 0 {
        let ret = ((i as u32).wrapping_shl(shift as u32)) & 0xFFFFFFFF;
        if ret > 0x7FFFFFFF {
            -((!ret & 0x7FFFFFFF) as i32) - 1
        } else {
            ret as i32
        }
    } else {
        ((i as u32) >> ((-shift) as u32)) as i32
    }
}

/// Unpack GEMPAK realpack (bit-packed integer) data into f64 values.
/// Matches the DP_UNPK subroutine from GEMPAK.
pub fn unpack_real(
    packed: &[i32],
    params: &ParameterInfo,
    length: usize,
    missing_float: f32,
) -> Result<Vec<f64>, String> {
    let nparms = params.names.len();
    if nparms == 0 {
        return Ok(Vec::new());
    }

    let total_bits: i32 = params.bits.iter().sum();
    let pwords = ((total_bits - 1) / 32 + 1) as usize;
    let npack = (length - 1) / pwords + 1;

    let mskpat: i32 = -1; // 0xFFFFFFFF as signed i32

    let mut unpacked = vec![f64::NAN; npack * nparms];

    let mut ir = 0usize;
    let mut ii = 0usize;

    for _ in 0..npack {
        if ii + pwords > packed.len() {
            break;
        }
        let pdat = &packed[ii..ii + pwords];
        let mut itotal = 0i32;

        for idata in 0..nparms {
            let scale = 10.0_f64.powi(params.scales[idata]);
            let offset = params.offsets[idata] as f64;
            let bits = params.bits[idata];
            let imissc = fortran_ishift(mskpat, bits - 32);

            let jsbit = (itotal % 32) + 1;
            let jsword = (itotal / 32) as usize;
            let jshift = 1 - jsbit;
            let jword = pdat[jsword];
            let mask = fortran_ishift(mskpat, bits - 32);
            let mut ifield = fortran_ishift(jword, jshift) & mask;

            if (jsbit + bits - 1) > 32 {
                if jsword + 1 < pdat.len() {
                    let jword2 = pdat[jsword + 1];
                    let iword = fortran_ishift(jword2, jshift + 32) & mask;
                    ifield |= iword;
                }
            }

            if ifield == imissc {
                unpacked[ir + idata] = f64::NAN;
            } else {
                unpacked[ir + idata] = (ifield as f64 + offset) * scale;
            }
            itotal += bits;
        }
        ir += nparms;
        ii += pwords;
    }

    // Mark remaining values matching missing_float as NaN
    let mf = missing_float as f64;
    for v in unpacked.iter_mut() {
        if !v.is_nan() && (*v - mf).abs() < 0.5 {
            *v = f64::NAN;
        }
    }

    Ok(unpacked)
}

// ── Station header parsing helpers ──────────────────────────────────────

/// Parsed station information from a GEMPAK sounding or surface column/row header.
#[derive(Debug, Clone)]
pub struct StationHeader {
    pub stid: String,
    pub stnm: i32,
    pub slat: f64,
    pub slon: f64,
    pub selv: i32,
    pub stat: String,
    pub coun: String,
    pub std2: String,
}

/// Parsed date/time from a GEMPAK row/column header.
#[derive(Debug, Clone)]
pub struct DateTimeHeader {
    pub date: String,
    pub time: String,
}

/// Known column/row key names and how to interpret the raw i32 values.
pub fn parse_key_value(key: &str, raw: i32, buf: &[u8], offset: usize) -> KeyValue {
    match key {
        "STID" | "STAT" | "COUN" | "STD2" => {
            // These are 4-byte ASCII packed into an i32
            let bytes = &buf[offset..offset + 4];
            let s = String::from_utf8_lossy(bytes)
                .trim_end_matches(|c: char| c.is_whitespace() || c == '\0')
                .to_string();
            KeyValue::Str(s)
        }
        "SLAT" | "SLON" => {
            // Stored as integer * 100
            KeyValue::Float(raw as f64 / 100.0)
        }
        "DATE" => {
            KeyValue::Str(convert_dattim(raw))
        }
        "TIME" => {
            KeyValue::Str(convert_time(raw))
        }
        _ => KeyValue::Int(raw),
    }
}

/// A parsed key value.
#[derive(Debug, Clone)]
pub enum KeyValue {
    Int(i32),
    Float(f64),
    Str(String),
}

impl KeyValue {
    pub fn as_str(&self) -> String {
        match self {
            KeyValue::Str(s) => s.clone(),
            KeyValue::Int(i) => i.to_string(),
            KeyValue::Float(f) => f.to_string(),
        }
    }

    pub fn as_f64(&self) -> f64 {
        match self {
            KeyValue::Float(f) => *f,
            KeyValue::Int(i) => *i as f64,
            _ => f64::NAN,
        }
    }

    pub fn as_i32(&self) -> i32 {
        match self {
            KeyValue::Int(i) => *i,
            KeyValue::Float(f) => *f as i32,
            _ => 0,
        }
    }
}

/// Convert GEMPAK DATTIM integer to a date string "YYYY-MM-DD" or "YY-MM-DD".
pub fn convert_dattim(dattim: i32) -> String {
    if dattim == 0 {
        return String::new();
    }
    if dattim < 100_000_000 {
        // Format: YYMMDD
        let yy = dattim / 10000;
        let mm = (dattim / 100) % 100;
        let dd = dattim % 100;
        let year = if yy > 50 { 1900 + yy } else { 2000 + yy };
        format!("{:04}-{:02}-{:02}", year, mm, dd)
    } else {
        // Format: MMDDYYHHMM
        let mm = dattim / 100_000_000;
        let dd = (dattim / 1_000_000) % 100;
        let yy = (dattim / 10_000) % 100;
        let hh = (dattim / 100) % 100;
        let mn = dattim % 100;
        let year = if yy > 50 { 1900 + yy } else { 2000 + yy };
        format!("{:04}-{:02}-{:02}T{:02}:{:02}", year, mm, dd, hh, mn)
    }
}

/// Convert GEMPAK time integer (HHMM) to "HH:MM".
pub fn convert_time(t: i32) -> String {
    let hh = t / 100;
    let mm = t % 100;
    format!("{:02}:{:02}", hh, mm)
}

/// Parse station headers from raw i32 values using key names.
pub fn parse_station_header(keys: &[String], raw_values: &[i32], raw_bytes: &[u8], header_byte_offset: usize) -> StationHeader {
    let mut stid = String::new();
    let mut stnm = 0i32;
    let mut slat = 0.0f64;
    let mut slon = 0.0f64;
    let mut selv = 0i32;
    let mut stat = String::new();
    let mut coun = String::new();
    let mut std2 = String::new();

    for (i, key) in keys.iter().enumerate() {
        if i >= raw_values.len() {
            break;
        }
        let byte_off = header_byte_offset + i * BYTES_PER_WORD;
        match key.as_str() {
            "STID" => {
                if byte_off + 4 <= raw_bytes.len() {
                    stid = String::from_utf8_lossy(&raw_bytes[byte_off..byte_off + 4])
                        .trim()
                        .to_string();
                }
            }
            "STD2" => {
                if byte_off + 4 <= raw_bytes.len() {
                    std2 = String::from_utf8_lossy(&raw_bytes[byte_off..byte_off + 4])
                        .trim()
                        .to_string();
                }
            }
            "STAT" => {
                if byte_off + 4 <= raw_bytes.len() {
                    stat = String::from_utf8_lossy(&raw_bytes[byte_off..byte_off + 4])
                        .trim()
                        .to_string();
                }
            }
            "COUN" => {
                if byte_off + 4 <= raw_bytes.len() {
                    coun = String::from_utf8_lossy(&raw_bytes[byte_off..byte_off + 4])
                        .trim()
                        .to_string();
                }
            }
            "STNM" => stnm = raw_values[i],
            "SLAT" => slat = raw_values[i] as f64 / 100.0,
            "SLON" => slon = raw_values[i] as f64 / 100.0,
            "SELV" => selv = raw_values[i],
            _ => {}
        }
    }

    StationHeader {
        stid,
        stnm,
        slat,
        slon,
        selv,
        stat,
        coun,
        std2,
    }
}

/// Parse date/time from raw header values using key names.
pub fn parse_datetime_header(keys: &[String], raw_values: &[i32]) -> DateTimeHeader {
    let mut date = String::new();
    let mut time = String::new();

    for (i, key) in keys.iter().enumerate() {
        if i >= raw_values.len() {
            break;
        }
        match key.as_str() {
            "DATE" => date = convert_dattim(raw_values[i]),
            "TIME" => time = convert_time(raw_values[i]),
            _ => {}
        }
    }

    DateTimeHeader { date, time }
}
