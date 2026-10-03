//! Minimal read-only classic netCDF parser — just enough for CfRadial 1.x.
//!
//! Implements the netCDF classic file format per the Unidata "NetCDF Classic
//! Format Specification" (a.k.a. CDF-1) and its 64-bit-offset variant
//! (CDF-2): <https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html>.
//! CDF-5 (64-bit data, `CDF\x05`) is detected and rejected with a clear
//! error — radar moment files do not use it. All values are big-endian.
//!
//! Supported: dimension/attribute/variable lists, the six classic types
//! (byte, char, short, int, float, double), fixed-size variables, and
//! record variables (unlimited dimension) including the single-record-
//! variable no-padding special case. netCDF-4/HDF5 files never reach this
//! module (they carry the HDF5 magic, not `CDF`).

use std::collections::{BTreeMap, HashMap};

use crate::{CfRadialError, Result};

const NC_DIMENSION: u32 = 0x0A;
const NC_VARIABLE: u32 = 0x0B;
const NC_ATTRIBUTE: u32 = 0x0C;
const MAX_NC_DIMENSIONS: usize = 1024;
const MAX_NC_VARIABLES: usize = 4096;
const MAX_NC_ATTRIBUTES: usize = 4096;
const MAX_NC_VAR_DIMS: usize = 32;
const MAX_NC_NAME_BYTES: usize = 64 * 1024;
const MAX_NC_DIMENSION_LEN: usize = 100 * 1024 * 1024;
const MAX_NC_ATTRIBUTE_BYTES: usize = 16 * 1024 * 1024;
const MAX_NC_ARRAY_BYTES: usize = 256 * 1024 * 1024;

/// `true` for classic netCDF magic (`CDF\x01` or `CDF\x02`). CDF-5 sniffs
/// true as well so the decoder can reject it with a useful message instead
/// of the Level II decoder's.
pub fn looks_like_netcdf3_bytes(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && &bytes[..3] == b"CDF" && matches!(bytes[3], 1 | 2 | 5)
}

/// An attribute value (global or per-variable).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum NcValue {
    /// Text (`NC_CHAR`, or a one-element `NC_STRING` in netCDF-4).
    Str(String),
    /// A netCDF-4 `NC_STRING` array of more than one element.
    Strings(Vec<String>),
    /// `NC_FLOAT` values, kept at their width: xarray derives the decoded
    /// dtype of a packed variable from the width of its `scale_factor`.
    Floats(Vec<f32>),
    /// `NC_DOUBLE` values (and netCDF-4 `uint64` values above `i64::MAX`).
    Doubles(Vec<f64>),
    /// Integer values, widened to `i64`, with the stored type.
    Ints(Vec<i64>, IntKind),
}

/// Attributes of a variable or a file, in file order (netCDF names are
/// unique; inserting a name again replaces its value in place). Lookups and
/// inserts go through a name index, so an object with many attributes (HDF5
/// allows 65,536) costs linear time, not quadratic.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NcAttrs {
    entries: Vec<(String, NcValue)>,
    index: HashMap<String, usize>,
}

impl NcAttrs {
    /// The value of attribute `name`.
    pub fn get(&self, name: &str) -> Option<&NcValue> {
        let position = *self.index.get(name)?;
        self.entries.get(position).map(|(_, value)| value)
    }

    /// Whether attribute `name` exists.
    pub fn contains_key(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Add `name` at the end, or replace its value where it is.
    pub fn insert(&mut self, name: String, value: NcValue) {
        if let Some(&position) = self.index.get(&name)
            && let Some(slot) = self.entries.get_mut(position)
        {
            slot.1 = value;
            return;
        }
        self.index.insert(name.clone(), self.entries.len());
        self.entries.push((name, value));
    }

    /// `(name, value)` pairs in file order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &NcValue)> {
        self.entries.iter().map(|(name, value)| (name, value))
    }

    /// Number of attributes.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when there are none.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl<'a> IntoIterator for &'a NcAttrs {
    type Item = (&'a String, &'a NcValue);
    type IntoIter = std::iter::Map<
        std::slice::Iter<'a, (String, NcValue)>,
        fn(&'a (String, NcValue)) -> (&'a String, &'a NcValue),
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter().map(|(name, value)| (name, value))
    }
}

impl FromIterator<(String, NcValue)> for NcAttrs {
    fn from_iter<I: IntoIterator<Item = (String, NcValue)>>(iter: I) -> Self {
        let mut attrs = Self::default();
        for (name, value) in iter {
            attrs.insert(name, value);
        }
        attrs
    }
}

/// The stored type of an integer attribute (classic `byte`, `short`, `int`;
/// netCDF-4 also `ubyte`, `ushort`, `uint`, `int64`, `uint64`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum IntKind {
    /// `byte` (int8).
    I8,
    /// `ubyte` (uint8).
    U8,
    /// `short` (int16).
    I16,
    /// `ushort` (uint16).
    U16,
    /// `int` (int32).
    I32,
    /// `uint` (uint32).
    U32,
    /// `int64`.
    I64,
    /// `uint64` (values that fit in `i64`; larger ones are `Doubles`).
    U64,
}

impl NcValue {
    /// The text of a `Str` value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(value) => Some(value),
            _ => None,
        }
    }

    /// The first element of a numeric value, widened to f64.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Floats(values) => values.first().map(|value| f64::from(*value)),
            Self::Doubles(values) => values.first().copied(),
            Self::Ints(values, _) => values.first().map(|value| *value as f64),
            _ => None,
        }
    }
}

/// Variable data in its stored type (classic files store big-endian; the
/// unsigned, 64-bit and string types occur in netCDF-4 files only).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum NcArray {
    /// `byte`.
    I8(Vec<i8>),
    /// `char`: raw bytes.
    Char(Vec<u8>),
    /// `short`.
    I16(Vec<i16>),
    /// `int`.
    I32(Vec<i32>),
    /// `float`.
    F32(Vec<f32>),
    /// `double`.
    F64(Vec<f64>),
    /// `ubyte`.
    U8(Vec<u8>),
    /// `ushort`.
    U16(Vec<u16>),
    /// `uint`.
    U32(Vec<u32>),
    /// `int64`.
    I64(Vec<i64>),
    /// `uint64`.
    U64(Vec<u64>),
    /// `string`: one text per element.
    Str(Vec<String>),
}

impl NcArray {
    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Self::I8(values) => values.len(),
            Self::Char(values) => values.len(),
            Self::I16(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
            Self::U8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::Str(values) => values.len(),
        }
    }

    /// True when there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Element as f64 (numeric types only).
    pub fn get_f64(&self, index: usize) -> Option<f64> {
        match self {
            Self::I8(values) => values.get(index).map(|value| f64::from(*value)),
            Self::Char(_) | Self::Str(_) => None,
            Self::I16(values) => values.get(index).map(|value| f64::from(*value)),
            Self::I32(values) => values.get(index).map(|value| f64::from(*value)),
            Self::F32(values) => values.get(index).map(|value| f64::from(*value)),
            Self::F64(values) => values.get(index).copied(),
            Self::U8(values) => values.get(index).map(|value| f64::from(*value)),
            Self::U16(values) => values.get(index).map(|value| f64::from(*value)),
            Self::U32(values) => values.get(index).map(|value| f64::from(*value)),
            Self::I64(values) => values.get(index).map(|value| *value as f64),
            Self::U64(values) => values.get(index).map(|value| *value as f64),
        }
    }

    /// True for text (`char` or `string`) data.
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Char(_) | Self::Str(_))
    }
}

/// Where a variable's data lives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Location {
    /// Classic netCDF: the `begin` offset of the data (or first record).
    Classic(u64),
    /// netCDF-4: the HDF5 dataset path.
    Hdf5(String),
    /// netCDF-4: one member of a compound (user-defined) variable: the
    /// HDF5 dataset path and the member names down to it.
    Hdf5Member {
        /// HDF5 dataset path.
        path: String,
        /// Member names, outermost first.
        members: Vec<String>,
    },
    /// netCDF-4: an opaque variable's bytes (the HDF5 dataset path), over
    /// the variable's dimensions and one of the opaque size.
    Hdf5Bytes(String),
    /// netCDF-4: the lengths of a variable-length variable's sequences.
    Hdf5VlenLengths(String),
    /// netCDF-4: a variable-length variable's values, back to back.
    Hdf5VlenValues(String),
}

/// A variable of a classic file or of one netCDF-4 group.
#[derive(Clone, Debug)]
pub struct NcVar {
    /// Variable name.
    pub name: String,
    /// Dimension indices into the file's (or group view's) `dims`.
    pub dim_ids: Vec<usize>,
    /// Attributes, in file order.
    pub attrs: NcAttrs,
    /// Position in the file header's (or group's) variable list.
    pub index: usize,
    pub(crate) nc_type: u32,
    pub(crate) location: Location,
}

impl NcVar {
    /// The classic data offset.
    fn begin(&self) -> Result<u64> {
        match &self.location {
            Location::Classic(begin) => Ok(*begin),
            _ => Err(invalid(
                0,
                format!("netCDF variable '{}' is not in a classic file", self.name),
            )),
        }
    }

    /// The netCDF external type code (1 byte, 2 char, 3 short, 4 int, 5
    /// float, 6 double; netCDF-4 adds 7 ubyte, 8 ushort, 9 uint, 10 int64,
    /// 11 uint64, 12 string, and 0 here for a user-defined type).
    pub fn nc_type(&self) -> u32 {
        self.nc_type
    }

    /// A text attribute.
    pub fn attr_str(&self, name: &str) -> Option<&str> {
        self.attrs.get(name).and_then(NcValue::as_str)
    }

    /// The first element of a numeric attribute, widened to f64.
    pub fn attr_f64(&self, name: &str) -> Option<f64> {
        self.attrs.get(name).and_then(NcValue::as_f64)
    }

    /// True for a `char` or `string` variable.
    pub fn is_text(&self) -> bool {
        matches!(self.nc_type, 2 | 12)
    }
}

/// Parsed header of a classic netCDF file plus the backing bytes.
pub struct Nc3File<'a> {
    bytes: &'a [u8],
    /// (name, length) — the record dimension stores its per-file length
    /// (`numrecs`), not zero.
    pub dims: Vec<(String, usize)>,
    /// Index of the record (unlimited) dimension in `dims`, if the file has one.
    pub record_dim: Option<usize>,
    /// Number of records.
    pub numrecs: usize,
    /// Global attributes, in file order.
    pub gattrs: NcAttrs,
    /// Variables, by name.
    pub vars: BTreeMap<String, NcVar>,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
    offset64: bool,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let start = self.at;
        let end = start
            .checked_add(len)
            .ok_or_else(|| invalid(start, "netCDF cursor overflow"))?;
        let raw = self
            .bytes
            .get(start..end)
            .ok_or_else(|| truncated(start, len, self.bytes.len()))?;
        self.at = end;
        Ok(raw)
    }

    fn take_padded_4(&mut self, len: usize) -> Result<&'a [u8]> {
        let padded = len
            .checked_add(3)
            .map(|value| value / 4 * 4)
            .ok_or_else(|| invalid(self.at, "netCDF padded length overflow"))?;
        let start = self.at;
        let end = start
            .checked_add(padded)
            .ok_or_else(|| invalid(start, "netCDF cursor overflow"))?;
        let data_end = start
            .checked_add(len)
            .ok_or_else(|| invalid(start, "netCDF data length overflow"))?;
        if end > self.bytes.len() {
            return Err(truncated(start, padded, self.bytes.len()));
        }
        self.at = end;
        Ok(&self.bytes[start..data_end])
    }

    fn take_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let start = self.at;
        let raw = self.take(N)?;
        raw.first_chunk::<N>()
            .copied()
            .ok_or_else(|| truncated(start, N, self.bytes.len()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.take_array()?))
    }

    fn offset(&mut self) -> Result<u64> {
        if self.offset64 {
            Ok(u64::from_be_bytes(self.take_array()?))
        } else {
            Ok(u64::from(self.u32()?))
        }
    }

    fn name(&mut self) -> Result<String> {
        let len = self.u32()? as usize;
        if len > MAX_NC_NAME_BYTES {
            return Err(limit(format!(
                "netCDF name is {len} bytes (limit {MAX_NC_NAME_BYTES})"
            )));
        }
        let raw = self.take_padded_4(len)?;
        Ok(String::from_utf8_lossy(raw).into_owned())
    }

    fn attrs(&mut self) -> Result<NcAttrs> {
        let tag = self.u32()?;
        let count = self.u32()? as usize;
        if tag != NC_ATTRIBUTE && (tag != 0 || count != 0) {
            return Err(invalid(self.at, "malformed attribute list tag"));
        }
        if count > MAX_NC_ATTRIBUTES {
            return Err(limit(format!(
                "netCDF attribute count is {count} (limit {MAX_NC_ATTRIBUTES})"
            )));
        }
        let mut attrs = NcAttrs::default();
        for _ in 0..count {
            let name = self.name()?;
            let nc_type = self.u32()?;
            let nelems = self.u32()? as usize;
            let elem_size = type_size(nc_type, self.at)?;
            let byte_len = nelems
                .checked_mul(elem_size)
                .ok_or_else(|| invalid(self.at, "netCDF attribute size overflow"))?;
            if byte_len > MAX_NC_ATTRIBUTE_BYTES {
                return Err(limit(format!(
                    "netCDF attribute is {byte_len} bytes (limit {MAX_NC_ATTRIBUTE_BYTES})"
                )));
            }
            let raw = self.take_padded_4(byte_len)?;
            let value = match nc_type {
                2 => NcValue::Str(
                    String::from_utf8_lossy(raw)
                        .trim_end_matches('\0')
                        .to_owned(),
                ),
                1 => {
                    let mut values = reserve_vec(nelems, "netCDF byte attribute")?;
                    values.extend(raw.iter().map(|byte| i64::from(*byte as i8)));
                    NcValue::Ints(values, IntKind::I8)
                }
                3 => {
                    let mut values = reserve_vec(nelems, "netCDF short attribute")?;
                    values.extend(
                        raw.chunks_exact(2)
                            .map(|pair| i64::from(i16::from_be_bytes([pair[0], pair[1]]))),
                    );
                    NcValue::Ints(values, IntKind::I16)
                }
                4 => {
                    let mut values = reserve_vec(nelems, "netCDF int attribute")?;
                    values.extend(
                        raw.as_chunks::<4>()
                            .0
                            .iter()
                            .map(|quad| i64::from(i32::from_be_bytes(*quad))),
                    );
                    NcValue::Ints(values, IntKind::I32)
                }
                5 => {
                    let mut values = reserve_vec(nelems, "netCDF float attribute")?;
                    values.extend(
                        raw.as_chunks::<4>()
                            .0
                            .iter()
                            .map(|quad| f32::from_be_bytes(*quad)),
                    );
                    NcValue::Floats(values)
                }
                6 => {
                    let mut values = reserve_vec(nelems, "netCDF double attribute")?;
                    values.extend(
                        raw.as_chunks::<8>()
                            .0
                            .iter()
                            .map(|oct| f64::from_be_bytes(*oct)),
                    );
                    NcValue::Doubles(values)
                }
                other => return Err(invalid(self.at, format!("attribute type {other}"))),
            };
            attrs.insert(name, value);
        }
        Ok(attrs)
    }
}

impl<'a> Nc3File<'a> {
    /// Parse the header of a classic netCDF file (CDF-1 or CDF-2). Variable data
    /// is read on request.
    pub fn open(bytes: &'a [u8]) -> Result<Self> {
        if !looks_like_netcdf3_bytes(bytes) {
            return Err(invalid(0, "missing netCDF classic magic"));
        }
        let version = bytes[3];
        if version == 5 {
            return Err(invalid(
                3,
                "CDF-5 (64-bit data) netCDF is unsupported; convert with `nccopy -k classic`",
            ));
        }
        let mut cursor = Cursor {
            bytes,
            at: 4,
            offset64: version == 2,
        };
        let numrecs = {
            let raw = cursor.u32()?;
            // 0xFFFFFFFF = STREAMING sentinel; treat as zero records.
            if raw == u32::MAX { 0 } else { raw as usize }
        };
        if numrecs > MAX_NC_DIMENSION_LEN {
            return Err(limit(format!(
                "netCDF record count is {numrecs} (limit {MAX_NC_DIMENSION_LEN})"
            )));
        }

        // Dimension list.
        let tag = cursor.u32()?;
        let dim_count = cursor.u32()? as usize;
        if tag != NC_DIMENSION && (tag != 0 || dim_count != 0) {
            return Err(invalid(cursor.at, "malformed dimension list tag"));
        }
        if dim_count > MAX_NC_DIMENSIONS {
            return Err(limit(format!(
                "netCDF dimension count is {dim_count} (limit {MAX_NC_DIMENSIONS})"
            )));
        }
        let mut dims = Vec::with_capacity(dim_count);
        let mut record_dim = None;
        for index in 0..dim_count {
            let name = cursor.name()?;
            let len = cursor.u32()? as usize;
            if len > MAX_NC_DIMENSION_LEN {
                return Err(limit(format!(
                    "netCDF dimension '{name}' length is {len} (limit {MAX_NC_DIMENSION_LEN})"
                )));
            }
            if len == 0 {
                if record_dim.is_some() {
                    return Err(invalid(
                        cursor.at,
                        "netCDF declares multiple record dimensions",
                    ));
                }
                record_dim = Some(index);
                dims.push((name, numrecs));
            } else {
                dims.push((name, len));
            }
        }

        let gattrs = cursor.attrs()?;

        // Variable list.
        let tag = cursor.u32()?;
        let var_count = cursor.u32()? as usize;
        if tag != NC_VARIABLE && (tag != 0 || var_count != 0) {
            return Err(invalid(cursor.at, "malformed variable list tag"));
        }
        if var_count > MAX_NC_VARIABLES {
            return Err(limit(format!(
                "netCDF variable count is {var_count} (limit {MAX_NC_VARIABLES})"
            )));
        }
        let mut vars = BTreeMap::new();
        for index in 0..var_count {
            let name = cursor.name()?;
            let ndims = cursor.u32()? as usize;
            if ndims > MAX_NC_VAR_DIMS {
                return Err(limit(format!(
                    "netCDF variable '{name}' rank is {ndims} (limit {MAX_NC_VAR_DIMS})"
                )));
            }
            let mut dim_ids = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                let dim_id = cursor.u32()? as usize;
                if dim_id >= dims.len() {
                    return Err(invalid(
                        cursor.at,
                        format!("netCDF variable '{name}' references dimension {dim_id}"),
                    ));
                }
                dim_ids.push(dim_id);
            }
            let attrs = cursor.attrs()?;
            let nc_type = cursor.u32()?;
            let _vsize = cursor.u32()?; // recomputed below; unreliable for big vars
            let begin = cursor.offset()?;
            vars.insert(
                name.clone(),
                NcVar {
                    name,
                    dim_ids,
                    attrs,
                    index,
                    nc_type,
                    location: Location::Classic(begin),
                },
            );
        }

        Ok(Self {
            bytes,
            dims,
            record_dim,
            numrecs,
            gattrs,
            vars,
        })
    }

    /// A global text attribute.
    pub fn gattr_str(&self, name: &str) -> Option<&str> {
        self.gattrs.get(name).and_then(NcValue::as_str)
    }

    /// The first value of a numeric global attribute, as `f64`.
    pub fn gattr_f64(&self, name: &str) -> Option<f64> {
        self.gattrs.get(name).and_then(NcValue::as_f64)
    }

    /// Resolved dimension lengths of a variable (record dim → numrecs).
    pub fn var_dims(&self, var: &NcVar) -> Vec<usize> {
        var.dim_ids
            .iter()
            .map(|id| self.dims.get(*id).map(|(_, len)| *len).unwrap_or(0))
            .collect()
    }

    fn is_record_var(&self, var: &NcVar) -> bool {
        matches!((var.dim_ids.first(), self.record_dim), (Some(first), Some(record)) if *first == record)
    }

    /// Per-record slab size in bytes for a record variable (or the full
    /// size for a fixed variable), before padding.
    fn slab_bytes(&self, var: &NcVar) -> Result<usize> {
        let elem = type_size(var.nc_type, 0)?;
        let skip_record = usize::from(self.is_record_var(var));
        let count = var.dim_ids[skip_record..]
            .iter()
            .try_fold(1usize, |count, id| {
                let len = self
                    .dims
                    .get(*id)
                    .map(|(_, len)| *len)
                    .ok_or_else(|| invalid(0, format!("invalid netCDF dimension id {id}")))?;
                count
                    .checked_mul(len)
                    .ok_or_else(|| invalid(0, "netCDF variable element-count overflow"))
            })?;
        let bytes = count
            .checked_mul(elem)
            .ok_or_else(|| invalid(0, "netCDF variable byte-size overflow"))?;
        if bytes > MAX_NC_ARRAY_BYTES {
            return Err(limit(format!(
                "netCDF variable '{}' is {bytes} bytes per slab (limit {MAX_NC_ARRAY_BYTES})",
                var.name
            )));
        }
        Ok(bytes)
    }

    /// Read the full data array of `name`, de-interleaving record slabs.
    pub fn read_var(&self, name: &str) -> Result<NcArray> {
        let var = self
            .vars
            .get(name)
            .ok_or_else(|| invalid(0, format!("netCDF variable '{name}' not found")))?;
        let slab = self.slab_bytes(var)?;
        let raw: Vec<u8> = if self.is_record_var(var) {
            // recsize = sum over record vars of their padded slabs; the
            // single-record-variable case is unpadded per the spec.
            let record_vars: Vec<&NcVar> = self
                .vars
                .values()
                .filter(|candidate| self.is_record_var(candidate))
                .collect();
            let recsize: usize = if record_vars.len() == 1 {
                slab
            } else {
                record_vars
                    .iter()
                    .map(|candidate| {
                        self.slab_bytes(candidate).and_then(|bytes| {
                            bytes
                                .checked_add(3)
                                .map(|value| value / 4 * 4)
                                .ok_or_else(|| invalid(0, "netCDF record padding overflow"))
                        })
                    })
                    .try_fold(0usize, |total, bytes| {
                        total
                            .checked_add(bytes?)
                            .ok_or_else(|| invalid(0, "netCDF record size overflow"))
                    })?
            };
            let total = slab
                .checked_mul(self.numrecs)
                .ok_or_else(|| invalid(0, "netCDF record variable size overflow"))?;
            if total > MAX_NC_ARRAY_BYTES {
                return Err(limit(format!(
                    "netCDF variable '{name}' expands to {total} bytes (limit {MAX_NC_ARRAY_BYTES})"
                )));
            }
            let begin = usize::try_from(var.begin()?)
                .map_err(|_| invalid(0, "netCDF variable offset overflows usize"))?;
            // Reserve only after proving every record slab lies in the file,
            // so a record count the bytes cannot back allocates nothing.
            if let Some(last_record) = self.numrecs.checked_sub(1) {
                let last_end = last_record
                    .checked_mul(recsize)
                    .and_then(|offset| begin.checked_add(offset))
                    .and_then(|start| start.checked_add(slab))
                    .ok_or_else(|| invalid(0, "netCDF record offset overflow"))?;
                if last_end > self.bytes.len() {
                    return Err(truncated(
                        begin,
                        last_end - begin,
                        self.bytes.len().saturating_sub(begin),
                    ));
                }
            }
            let mut raw = reserve_vec(total, "netCDF record variable")?;
            for record in 0..self.numrecs {
                let record_offset = record
                    .checked_mul(recsize)
                    .ok_or_else(|| invalid(0, "netCDF record offset overflow"))?;
                let start = begin
                    .checked_add(record_offset)
                    .ok_or_else(|| invalid(0, "netCDF record offset overflow"))?;
                let end = start
                    .checked_add(slab)
                    .ok_or_else(|| invalid(start, "netCDF record range overflow"))?;
                let chunk = self
                    .bytes
                    .get(start..end)
                    .ok_or_else(|| truncated(start, slab, self.bytes.len()))?;
                raw.extend_from_slice(chunk);
            }
            raw
        } else {
            let start = usize::try_from(var.begin()?)
                .map_err(|_| invalid(0, "netCDF variable offset overflows usize"))?;
            let end = start
                .checked_add(slab)
                .ok_or_else(|| invalid(start, "netCDF variable range overflow"))?;
            let bytes = self
                .bytes
                .get(start..end)
                .ok_or_else(|| truncated(start, slab, self.bytes.len()))?;
            let mut raw = reserve_vec(bytes.len(), "netCDF variable")?;
            raw.extend_from_slice(bytes);
            raw
        };
        decode_array(&raw, var.nc_type)
    }
}

fn decode_array(raw: &[u8], nc_type: u32) -> Result<NcArray> {
    Ok(match nc_type {
        1 => {
            let mut values = reserve_vec(raw.len(), "netCDF i8 array")?;
            values.extend(raw.iter().map(|byte| *byte as i8));
            NcArray::I8(values)
        }
        2 => {
            let mut values = reserve_vec(raw.len(), "netCDF char array")?;
            values.extend_from_slice(raw);
            NcArray::Char(values)
        }
        3 => {
            let mut values = reserve_vec(raw.len() / 2, "netCDF i16 array")?;
            values.extend(
                raw.chunks_exact(2)
                    .map(|pair| i16::from_be_bytes([pair[0], pair[1]])),
            );
            NcArray::I16(values)
        }
        4 => {
            let mut values = reserve_vec(raw.len() / 4, "netCDF i32 array")?;
            values.extend(
                raw.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|quad| i32::from_be_bytes(*quad)),
            );
            NcArray::I32(values)
        }
        5 => {
            let mut values = reserve_vec(raw.len() / 4, "netCDF f32 array")?;
            values.extend(
                raw.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|quad| f32::from_be_bytes(*quad)),
            );
            NcArray::F32(values)
        }
        6 => {
            let mut values = reserve_vec(raw.len() / 8, "netCDF f64 array")?;
            values.extend(
                raw.as_chunks::<8>()
                    .0
                    .iter()
                    .map(|oct| f64::from_be_bytes(*oct)),
            );
            NcArray::F64(values)
        }
        other => return Err(invalid(0, format!("netCDF type {other} unsupported"))),
    })
}

fn reserve_vec<T>(count: usize, context: &'static str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|err| {
        invalid(
            0,
            format!("cannot reserve {context} with {count} elements: {err}"),
        )
    })?;
    Ok(values)
}

fn type_size(nc_type: u32, offset: usize) -> Result<usize> {
    match nc_type {
        1 | 2 => Ok(1),
        3 => Ok(2),
        4 | 5 => Ok(4),
        6 => Ok(8),
        other => Err(invalid(offset, format!("netCDF type {other} unsupported"))),
    }
}

fn limit(reason: String) -> CfRadialError {
    CfRadialError::LimitExceeded(reason)
}

fn invalid(offset: usize, reason: impl Into<String>) -> CfRadialError {
    CfRadialError::InvalidMessage {
        offset,
        reason: reason.into(),
    }
}

fn truncated(offset: usize, needed: usize, available: usize) -> CfRadialError {
    CfRadialError::Truncated {
        what: "netCDF structure",
        offset,
        needed,
        available,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden values: tools/golden_io_formats.py, section `cfradial`
    // (netCDF4-python reading the same files with mask/scale disabled).
    const XSAPR_CLASSIC: &str = "cfrad1-xsapr-sgp-20110520-ppi-classic";
    const IRENE: &str = "cfrad1-irene-sr2-20110827-120420-sur-sweeps01";

    fn corpus(id: &str) -> Vec<u8> {
        recast_radar_testdata::bytes(id).unwrap_or_else(|err| panic!("{err}"))
    }

    #[test]
    fn magic_sniffer_accepts_classic_versions() {
        // netCDF classic format specification: magic "CDF" + version byte
        // 1 (CDF-1), 2 (64-bit offset) or 5 (64-bit data).
        let classic = corpus(XSAPR_CLASSIC);
        // golden cfradial.xsapr_classic.magic = 43444601 ("CDF", version 1)
        assert_eq!(classic[..4], [0x43, 0x44, 0x46, 0x01]);
        assert!(looks_like_netcdf3_bytes(&classic));
        assert!(looks_like_netcdf3_bytes(&corpus(IRENE)));
        for (version, accepted) in [(2u8, true), (5, true), (3, false), (0, false)] {
            let mut mutated = classic.clone();
            mutated[3] = version;
            assert_eq!(
                looks_like_netcdf3_bytes(&mutated),
                accepted,
                "version byte {version}"
            );
        }
        // The published netCDF-4 container is HDF5 (signature 89 48 44 46).
        let netcdf4 = corpus("cfrad1-xsapr-sgp-20110520-ppi-netcdf4");
        assert_eq!(&netcdf4[1..4], b"HDF");
        assert!(!looks_like_netcdf3_bytes(&netcdf4));
        assert!(!looks_like_netcdf3_bytes(&classic[..3]));
    }

    #[test]
    fn parses_real_classic_cdf1_header_and_record_variables() {
        let bytes = corpus(XSAPR_CLASSIC);
        let file = Nc3File::open(&bytes).expect("open X-SAPR classic");
        // golden cfradial.xsapr_classic.dims: time is the UNLIMITED record
        // dimension with 40 records.
        assert_eq!(
            file.dims,
            vec![
                ("time".to_owned(), 40),
                ("range".to_owned(), 42),
                ("sweep".to_owned(), 1),
                ("string_length".to_owned(), 32),
            ]
        );
        assert_eq!(file.record_dim, Some(0));
        assert_eq!(file.numrecs, 40);
        assert_eq!(file.gattr_str("instrument_name"), Some("xsapr-sgp"));

        let var = file.vars.get("reflectivity_horizontal").expect("field");
        assert_eq!(file.var_dims(var), vec![40, 42]);
        assert_eq!(var.attr_str("units"), Some("dBZ"));
        assert_eq!(var.attr_f64("_FillValue"), Some(-9999.0));
        let NcArray::F32(values) = file.read_var("reflectivity_horizontal").expect("data") else {
            panic!("reflectivity_horizontal is float32");
        };
        assert_eq!(values.len(), 40 * 42);
        // golden cfradial.xsapr_classic.refl_raw (record-interleaved rows).
        for ((ray, gate), expected) in [
            ((0, 0), -6.05f32),
            ((0, 21), 23.3),
            ((10, 14), 25.23),
            ((39, 41), 19.68),
        ] {
            assert_eq!(values[ray * 42 + gate], expected, "[{ray},{gate}]");
        }
        // golden refl_fill_count 15, first at [3, 37].
        assert_eq!(values.iter().filter(|value| **value == -9999.0).count(), 15);
        assert_eq!(values[3 * 42 + 37], -9999.0);

        // prt(time) is a record variable too: golden ray 0/3/39 values.
        let NcArray::F32(prt) = file.read_var("prt").expect("prt") else {
            panic!("prt is float32");
        };
        assert_eq!(prt.len(), 40);
        assert!(prt.iter().all(|value| *value == 0.000_450_045_02));
        let NcArray::F64(time) = file.read_var("time").expect("time") else {
            panic!("time is float64");
        };
        assert_eq!((time[0], time[3], time[39]), (8.0, 9.0, 7.0));
    }

    #[test]
    fn parses_real_packed_int8_fixed_dimension_variables() {
        let bytes = corpus(IRENE);
        let file = Nc3File::open(&bytes).expect("open Irene");
        assert_eq!(file.record_dim, None, "golden: no unlimited dimension");
        assert_eq!(file.dims.len(), 8);
        assert_eq!(file.dims[0], ("time".to_owned(), 719));
        assert_eq!(file.dims[1], ("range".to_owned(), 1107));
        assert_eq!(file.gattr_str("version"), Some("CF-Radial-1.3"));
        let var = file.vars.get("DBZ").expect("DBZ");
        // golden cfradial.irene.dbz_attrs
        assert_eq!(var.attr_f64("scale_factor"), Some(0.5));
        assert_eq!(var.attr_f64("add_offset"), Some(32.0));
        assert_eq!(var.attr_f64("_FillValue"), Some(-128.0));
        let NcArray::I8(raw) = file.read_var("DBZ").expect("DBZ data") else {
            panic!("DBZ is int8");
        };
        assert_eq!(raw.len(), 719 * 1107);
        // golden cfradial.irene.dbz_raw
        for ((ray, gate), expected) in [
            ((0, 0), -127i8),
            ((0, 100), -10),
            ((10, 200), -21),
            ((180, 50), -32),
            ((359, 1106), -128),
            ((360, 10), 20),
            ((500, 300), -18),
            ((718, 700), 5),
        ] {
            assert_eq!(raw[ray * 1107 + gate], expected, "DBZ[{ray},{gate}]");
        }
    }

    #[test]
    fn cdf5_is_rejected_with_guidance() {
        let mut bytes = corpus(XSAPR_CLASSIC);
        bytes[3] = 5; // real CDF-1 header relabelled as CDF-5
        let Err(err) = Nc3File::open(&bytes) else {
            panic!("CDF-5 must be rejected");
        };
        assert!(err.to_string().contains("CDF-5"), "{err}");
    }

    #[test]
    fn rejects_absurd_header_counts_before_allocating() {
        // netCDF classic header: magic (4) | numrecs (4) | NC_DIMENSION tag (4)
        // | dimension count (4) | ... (golden dim_list_tag 10, dim_count 4).
        let mut bytes = corpus(XSAPR_CLASSIC);
        assert_eq!(
            u32::from_be_bytes(bytes[8..12].try_into().unwrap()),
            NC_DIMENSION
        );
        assert_eq!(u32::from_be_bytes(bytes[12..16].try_into().unwrap()), 4);
        bytes[12..16].copy_from_slice(&u32::MAX.to_be_bytes());

        let Err(err) = Nc3File::open(&bytes) else {
            panic!("dimension bomb must fail");
        };
        assert!(matches!(err, CfRadialError::LimitExceeded(_)), "{err}");
        assert!(err.to_string().contains("dimension count"), "{err}");
    }

    #[test]
    fn cursor_rejects_overflowing_ranges() {
        let mut cursor = Cursor {
            bytes: &[],
            at: usize::MAX,
            offset64: false,
        };
        let err = cursor.u32().expect_err("overflowing cursor must fail");
        assert!(err.to_string().contains("cursor overflow"));
    }
}
