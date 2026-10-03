//! The ODIM decoders' view of an HDF5 file ([`recast_radar_hdf5`]).
//!
//! ODIM_H5 attributes are strings, 64-bit integers ("long") and doubles, as
//! scalars or 1-D arrays; writers disagree about widths, so numeric values
//! widen to `i64`/`f64` for the typed readers ([`H5Attr`]). Data planes keep
//! their stored `u8`, `i8`, `u16`, `i16`, `i32`, `f32` and `f64` storage;
//! `u32`, `i64` and `u64` planes, which the model has no integer storage
//! for, widen to `f64` ([`H5Data`]). An enumerated plane (h5py writes
//! quality flags as a `bool` enum) keeps its base integers and names its
//! members ([`H5Dataset::enum_members`]).
//!
//! Passthrough never drops an attribute: [`H5File::attr_entries`] turns
//! every attribute of an object into model attributes, whatever its
//! datatype (string arrays, compound members as `name.member`, object
//! references as the target's path, enum values as their member names,
//! uninterpreted data as bytes).

use recast_radar_core::model::{ArrayBuf, AttrValue, Scalar};
use recast_radar_hdf5::{Attribute, Datatype, Values};

use crate::{OdimError, Result};

impl From<recast_radar_hdf5::Error> for OdimError {
    fn from(err: recast_radar_hdf5::Error) -> Self {
        match err {
            recast_radar_hdf5::Error::Truncated {
                what,
                offset,
                needed,
                available,
            } => Self::Truncated {
                what,
                offset,
                needed,
                available,
            },
            recast_radar_hdf5::Error::Invalid { offset, reason } => {
                Self::InvalidMessage { offset, reason }
            }
            recast_radar_hdf5::Error::LimitExceeded(reason) => Self::LimitExceeded(reason),
            other => Self::InvalidMessage {
                offset: 0,
                reason: other.to_string(),
            },
        }
    }
}

/// A decoded scalar or 1-D attribute value, for the typed readers.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum H5Attr {
    Str(String),
    F64(f64),
    I64(i64),
    F64Array(Vec<f64>),
    I64Array(Vec<i64>),
}

impl H5Attr {
    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(value) => Some(value),
            _ => None,
        }
    }

    /// Numeric view: integers widen to f64 (ODIM writers disagree about
    /// whether e.g. `nodata` is a long or a double).
    pub(crate) fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F64(value) => Some(*value),
            Self::I64(value) => Some(*value as f64),
            _ => None,
        }
    }

    pub(crate) fn as_i64(&self) -> Option<i64> {
        match self {
            Self::I64(value) => Some(*value),
            Self::F64(value) => (value.fract() == 0.0).then_some(*value as i64),
            _ => None,
        }
    }

    /// The typed view of an attribute: the first string of a string
    /// attribute, numbers widened; `None` for any other datatype (the
    /// passthrough keeps those, see [`H5File::attr_entries`]).
    fn from_attribute(attribute: &Attribute) -> Option<Self> {
        let values = attribute.values();
        if let Some(strings) = values.strings() {
            return Some(Self::Str(strings.into_iter().next().unwrap_or_default()));
        }
        let count = values.len();
        if let Some(ints) = int_values(values) {
            return Some(if count == 1 {
                Self::I64(ints[0])
            } else {
                Self::I64Array(ints)
            });
        }
        let floats: Vec<f64> = match values {
            Values::F32(v) => v.iter().map(|x| f64::from(*x)).collect(),
            Values::F64(v) => v.clone(),
            _ => return None,
        };
        Some(if count == 1 {
            Self::F64(floats[0])
        } else {
            Self::F64Array(floats)
        })
    }
}

/// Integer values widened to `i64`; 64-bit unsigned bit for bit, as the
/// earlier reader did.
fn int_values(values: &Values) -> Option<Vec<i64>> {
    Some(match values {
        Values::I8(v) => v.iter().map(|x| i64::from(*x)).collect(),
        Values::U8(v) => v.iter().map(|x| i64::from(*x)).collect(),
        Values::I16(v) => v.iter().map(|x| i64::from(*x)).collect(),
        Values::U16(v) => v.iter().map(|x| i64::from(*x)).collect(),
        Values::I32(v) => v.iter().map(|x| i64::from(*x)).collect(),
        Values::U32(v) => v.iter().map(|x| i64::from(*x)).collect(),
        Values::I64(v) => v.clone(),
        Values::U64(v) => v.iter().map(|x| *x as i64).collect(),
        _ => return None,
    })
}

/// Raw dataset elements in the closest storage.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum H5Data {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl H5Data {
    /// Bytes per stored element.
    pub(crate) fn word_bytes(&self) -> usize {
        match self {
            Self::U8(_) | Self::I8(_) => 1,
            Self::U16(_) | Self::I16(_) => 2,
            Self::I32(_) | Self::F32(_) => 4,
            Self::F64(_) => 8,
        }
    }
}

/// A dataset: dimension sizes (row-major) plus the element array.
#[derive(Clone, Debug)]
pub(crate) struct H5Dataset {
    pub(crate) dims: Vec<usize>,
    pub(crate) data: H5Data,
    /// `(value, name)` of each member of an enumerated datatype (empty
    /// otherwise).
    pub(crate) enum_members: Vec<(i64, String)>,
}

/// Read-only ODIM view over [`recast_radar_hdf5::H5File`].
pub(crate) struct H5File<'a>(recast_radar_hdf5::H5File<'a>);

impl<'a> H5File<'a> {
    pub(crate) fn open(bytes: &'a [u8]) -> Result<Self> {
        Self::from_hdf5(recast_radar_hdf5::H5File::open(bytes)?)
    }

    /// The ODIM view of an opened HDF5 file.
    pub(crate) fn from_hdf5(file: recast_radar_hdf5::H5File<'a>) -> Result<Self> {
        if !file.has_object("/what") {
            // netCDF-4 (CfRadial 1.x and 2) is HDF5 too, with no ODIM
            // `/what` group.
            let hint = if recast_radar_hdf5::netcdf4::is_netcdf4(&file) {
                "this is a netCDF-4 file: if it is CfRadial 1.x or 2, decode it with \
                 recast_radar_io_cfradial::read_cfradial_volume (the recast_radar_io \
                 router does so for every netCDF-4 file without a /what group)"
            } else {
                "ODIM_H5 files carry /what, /where and /how groups"
            };
            return Err(OdimError::InvalidMessage {
                offset: 0,
                reason: format!(
                    "HDF5 file has no /what group, so it is not ODIM_H5 (netCDF-4 CfRadial and \
                     other HDF5 radar formats do not decode here); {hint}"
                ),
            });
        }
        Ok(Self(file))
    }

    /// Names of the links of the group at `path`.
    pub(crate) fn child_names(&self, path: &str) -> Vec<String> {
        self.0.child_names(path)
    }

    /// True when `path` names an object.
    pub(crate) fn has_object(&self, path: &str) -> bool {
        self.0.has_object(path)
    }

    /// One attribute of the object at `path`, typed (see [`H5Attr`]).
    pub(crate) fn attr(&self, path: &str, name: &str) -> Option<H5Attr> {
        self.0.attr(path, name).and_then(H5Attr::from_attribute)
    }

    /// Every attribute of the object at `path` as model attributes, in
    /// storage order and whatever its datatype (module docs). Empty when the
    /// object does not exist.
    pub(crate) fn attr_entries(&self, path: &str) -> Vec<(Box<str>, AttrValue)> {
        let mut entries = Vec::new();
        for attribute in self.0.attrs(path) {
            self.push_entries(
                attribute.name(),
                attribute.datatype(),
                attribute.values(),
                &mut entries,
            );
        }
        entries
    }

    /// `values` of `datatype` as entries named `name` (compound members as
    /// `name.member`, the elements of a sequence array as `name.<i>`).
    fn push_entries(
        &self,
        name: &str,
        datatype: &Datatype,
        values: &Values,
        out: &mut Vec<(Box<str>, AttrValue)>,
    ) {
        if let Datatype::Enum { members, .. } = datatype
            && let Some(codes) = int_values(values)
        {
            out.push((name.into(), enum_value(members, &codes)));
            return;
        }
        let value = match values {
            Values::Compound {
                members: columns, ..
            } => {
                let member_types: Vec<&Datatype> = match datatype {
                    Datatype::Compound { members, .. } => {
                        members.iter().map(|member| &member.datatype).collect()
                    }
                    _ => Vec::new(),
                };
                for (index, (member, column)) in columns.iter().enumerate() {
                    let member_type = member_types.get(index).copied().unwrap_or(datatype);
                    self.push_entries(&format!("{name}.{member}"), member_type, column, out);
                }
                return;
            }
            Values::Sequences(sequences) => {
                let base = match datatype {
                    Datatype::VarLenSequence { base, .. } => base.as_ref(),
                    other => other,
                };
                if let [only] = sequences.as_slice() {
                    self.push_entries(name, base, only, out);
                } else {
                    for (index, sequence) in sequences.iter().enumerate() {
                        self.push_entries(&format!("{name}.{index}"), base, sequence, out);
                    }
                }
                return;
            }
            Values::References(targets) => {
                let paths: Vec<Box<str>> = targets
                    .iter()
                    .map(|target| match target {
                        Some(address) => self
                            .0
                            .object_at(*address)
                            .map_or_else(
                                || format!("@{address:#x}"),
                                |object| object.path().to_owned(),
                            )
                            .into(),
                        None => "".into(),
                    })
                    .collect();
                text_value(paths)
            }
            Values::Raw { bytes, .. } => AttrValue::Array(ArrayBuf::U8(bytes.clone())),
            Values::FixedStrings { .. } | Values::VarStrings(_) => {
                let strings: Vec<Box<str>> = values
                    .strings()
                    .unwrap_or_default()
                    .into_iter()
                    .map(String::into_boxed_str)
                    .collect();
                text_value(strings)
            }
            Values::F32(_) | Values::F64(_) => {
                let floats = values.to_f64_vec().unwrap_or_default();
                match floats.as_slice() {
                    [only] => AttrValue::Scalar(Scalar::F64(*only)),
                    _ => AttrValue::Array(ArrayBuf::F64(floats)),
                }
            }
            numeric => {
                let ints = int_values(numeric).unwrap_or_default();
                match ints.as_slice() {
                    [only] => AttrValue::Scalar(Scalar::I64(*only)),
                    _ => AttrValue::Array(ArrayBuf::I64(ints)),
                }
            }
        };
        out.push((name.into(), value));
    }

    /// Read the full dataset at `path` in its stored element type.
    pub(crate) fn dataset(&self, path: &str) -> Result<H5Dataset> {
        let dataset = self.0.dataset(path)?;
        let enum_members = match &dataset.datatype {
            Datatype::Enum { members, .. } => members
                .iter()
                .filter_map(|member| {
                    i64::try_from(member.value)
                        .ok()
                        .map(|value| (value, member.name.clone()))
                })
                .collect(),
            _ => Vec::new(),
        };
        let widen = |values: Vec<f64>| H5Data::F64(values);
        let data = match dataset.values {
            Values::U8(values) => H5Data::U8(values),
            Values::I8(values) => H5Data::I8(values),
            Values::U16(values) => H5Data::U16(values),
            Values::I16(values) => H5Data::I16(values),
            Values::I32(values) => H5Data::I32(values),
            Values::F32(values) => H5Data::F32(values),
            Values::F64(values) => H5Data::F64(values),
            Values::U32(v) => widen(v.into_iter().map(f64::from).collect()),
            Values::I64(v) => widen(v.into_iter().map(|x| x as f64).collect()),
            Values::U64(v) => widen(v.into_iter().map(|x| x as f64).collect()),
            _ => {
                return Err(OdimError::InvalidMessage {
                    offset: 0,
                    reason: format!(
                        "dataset '{path}': unsupported dataset element type {}",
                        describe(&dataset.datatype)
                    ),
                });
            }
        };
        Ok(H5Dataset {
            dims: dataset.dims,
            data,
            enum_members,
        })
    }

    /// The dataset at `path` as stored: its dimensions, its values (`u64`
    /// widened to `f64`, which [`ArrayBuf`] has no type for) and the
    /// members of an enumerated datatype.
    #[allow(clippy::type_complexity)]
    pub(crate) fn dataset_raw(
        &self,
        path: &str,
    ) -> Result<(Vec<usize>, ArrayBuf, Vec<(i64, String)>)> {
        let dataset = self.0.dataset(path)?;
        let enum_members = match &dataset.datatype {
            Datatype::Enum { members, .. } => members
                .iter()
                .filter_map(|member| {
                    i64::try_from(member.value)
                        .ok()
                        .map(|value| (value, member.name.clone()))
                })
                .collect(),
            _ => Vec::new(),
        };
        let raw = match dataset.values {
            Values::U8(values) => ArrayBuf::U8(values),
            Values::I8(values) => ArrayBuf::I8(values),
            Values::U16(values) => ArrayBuf::U16(values),
            Values::I16(values) => ArrayBuf::I16(values),
            Values::I32(values) => ArrayBuf::I32(values),
            Values::U32(values) => ArrayBuf::U32(values),
            Values::I64(values) => ArrayBuf::I64(values),
            Values::F32(values) => ArrayBuf::F32(values),
            Values::F64(values) => ArrayBuf::F64(values),
            Values::U64(values) => ArrayBuf::F64(values.into_iter().map(|v| v as f64).collect()),
            _ => {
                return Err(OdimError::InvalidMessage {
                    offset: 0,
                    reason: format!(
                        "dataset '{path}': unsupported dataset element type {}",
                        describe(&dataset.datatype)
                    ),
                });
            }
        };
        Ok((dataset.dims, raw, enum_members))
    }

    /// An ODIM `legend` dataset as `(code, class name)` pairs, in file
    /// order: the ODIM_H5 v2.4 layout (compound of `key` and `value`
    /// strings, the class name and its code as text) or FMI's (compound of
    /// an integer `code` and a string `class`). `None` for anything else.
    pub(crate) fn legend(&self, path: &str) -> Option<Vec<(i64, String)>> {
        let dataset = self.0.dataset(path).ok()?;
        let Values::Compound {
            members: columns, ..
        } = &dataset.values
        else {
            return None;
        };
        // Rows by the dataspace: an array member (`char[64]`) decodes to
        // `rows * 64` values.
        let rows: usize = dataset.dims.iter().product();
        let column = |name: &str| {
            columns
                .iter()
                .find(|(member, _)| member == name)
                .map(|(_, values)| values)
        };
        if let (Some(keys), Some(values)) = (column("key"), column("value")) {
            let keys = column_texts(keys, rows)?;
            let codes = column_texts(values, rows)?;
            return keys
                .into_iter()
                .zip(codes)
                .map(|(key, code)| code.trim().parse::<i64>().ok().map(|code| (code, key)))
                .collect();
        }
        if let (Some(codes), Some(classes)) = (column("code"), column("class")) {
            let classes = column_texts(classes, rows)?;
            return classes
                .into_iter()
                .enumerate()
                .map(|(row, class)| codes.get_i64(row).map(|code| (code, class)))
                .collect();
        }
        None
    }
}

/// `rows` strings from a compound column: strings, or `rows` equal runs of
/// 8-bit characters (a `char[n]` member, NUL-padded).
fn column_texts(values: &Values, rows: usize) -> Option<Vec<String>> {
    if let Some(strings) = values.strings() {
        return (strings.len() == rows).then_some(strings);
    }
    let bytes: Vec<u8> = match values {
        Values::I8(v) => v.iter().map(|byte| *byte as u8).collect(),
        Values::U8(v) => v.clone(),
        _ => return None,
    };
    if rows == 0 || !bytes.len().is_multiple_of(rows) {
        return None;
    }
    Some(
        bytes
            .chunks(bytes.len() / rows)
            .map(|chunk| {
                let text = chunk.split(|byte| *byte == 0).next().unwrap_or_default();
                String::from_utf8_lossy(text).trim().to_owned()
            })
            .collect(),
    )
}

/// One text, or an array of them.
fn text_value(mut texts: Vec<Box<str>>) -> AttrValue {
    if texts.len() == 1 {
        AttrValue::Text(texts.remove(0))
    } else {
        AttrValue::Array(ArrayBuf::Text(texts))
    }
}

/// Enumerated values by member name (h5py's `bool` enum as a bool); a code
/// with no member keeps its number.
fn enum_value(members: &[recast_radar_hdf5::EnumMember], codes: &[i64]) -> AttrValue {
    let name = |code: i64| {
        members
            .iter()
            .find(|member| member.value == i128::from(code))
            .map(|member| member.name.as_str())
    };
    let is_bool = members.len() == 2 && name(0) == Some("FALSE") && name(1) == Some("TRUE");
    if is_bool && let [code] = codes {
        return AttrValue::Bool(*code != 0);
    }
    let texts: Option<Vec<Box<str>>> = codes
        .iter()
        .map(|code| name(*code).map(Box::from))
        .collect();
    match texts {
        Some(texts) => text_value(texts),
        None => match codes {
            [only] => AttrValue::Scalar(Scalar::I64(*only)),
            _ => AttrValue::Array(ArrayBuf::I64(codes.to_vec())),
        },
    }
}

fn describe(datatype: &Datatype) -> String {
    match datatype {
        Datatype::Other { class, size } => format!("class {class} ({size} bytes)"),
        Datatype::Compound { .. } => "class 6 (compound)".to_owned(),
        Datatype::Reference { .. } => "class 7 (reference)".to_owned(),
        Datatype::VarLenSequence { .. } => "class 9 (variable-length sequence)".to_owned(),
        Datatype::Opaque { .. } => "class 5 (opaque)".to_owned(),
        Datatype::Array { .. } => "class 10 (array)".to_owned(),
        other => format!("{other:?}"),
    }
}
