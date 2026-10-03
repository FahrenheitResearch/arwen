//! Decoded attribute and dataset values.

use crate::bytes::{normalize_addr, read_le_u32, read_uint};
use crate::datatype::{ByteOrder, Datatype, StringPadding, int_value};
use crate::error::{Result, invalid, limit};
use crate::heap::GlobalHeaps;
use crate::limits::MAX_VLEN_BYTES;
use crate::space::Space;

/// Elements of an attribute or dataset, in the stored type.
///
/// Integers and floats keep their stored width and signedness (big-endian
/// storage is converted to native values). Enumerations decode to their base
/// integer type; bit fields to unsigned integers; array datatypes flatten
/// into their base type (the [`Datatype::Array`] dimensions say how).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Values {
    /// 8-bit signed integers.
    I8(Vec<i8>),
    /// 8-bit unsigned integers.
    U8(Vec<u8>),
    /// 16-bit signed integers.
    I16(Vec<i16>),
    /// 16-bit unsigned integers.
    U16(Vec<u16>),
    /// 32-bit signed integers.
    I32(Vec<i32>),
    /// 32-bit unsigned integers.
    U32(Vec<u32>),
    /// 64-bit signed integers (also 3-, 5-, 6- and 7-byte signed storage).
    I64(Vec<i64>),
    /// 64-bit unsigned integers (also 3-, 5-, 6- and 7-byte unsigned storage).
    U64(Vec<u64>),
    /// IEEE binary32.
    F32(Vec<f32>),
    /// IEEE binary64.
    F64(Vec<f64>),
    /// Fixed-length strings, `size` bytes each, padding kept.
    FixedStrings {
        /// Bytes per string.
        size: usize,
        /// Padding convention, for [`Values::strings`].
        padding: StringPadding,
        /// All strings back to back.
        bytes: Vec<u8>,
    },
    /// Variable-length strings (lossy UTF-8; a null string is empty).
    VarStrings(Vec<String>),
    /// Object references: the target object header address, `None` for a
    /// null reference.
    References(Vec<Option<u64>>),
    /// Variable-length sequences, one inner array per element.
    Sequences(Vec<Values>),
    /// Compound elements as one column per member, in member order. The
    /// column of an array-typed member holds each element's array
    /// flattened (`elements * 64` values for a `char[64]` member).
    Compound {
        /// Number of compound elements.
        elements: usize,
        /// Member name and column, in member order.
        members: Vec<(String, Values)>,
    },
    /// Elements this crate does not interpret, `size` bytes each (opaque,
    /// time, region references, non-IEEE floats, integers wider than 8
    /// bytes).
    Raw {
        /// Bytes per element.
        size: usize,
        /// All elements back to back.
        bytes: Vec<u8>,
    },
}

impl Values {
    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Self::I8(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::FixedStrings { size, bytes, .. } | Self::Raw { size, bytes } => {
                if *size == 0 {
                    0
                } else {
                    bytes.len() / size
                }
            }
            Self::VarStrings(v) => v.len(),
            Self::References(v) => v.len(),
            Self::Sequences(v) => v.len(),
            Self::Compound { elements, .. } => *elements,
        }
    }

    /// True when there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Element `index` of a numeric array widened to `f64` (exact for
    /// integers up to 2^53).
    pub fn get_f64(&self, index: usize) -> Option<f64> {
        Some(match self {
            Self::I8(v) => f64::from(*v.get(index)?),
            Self::U8(v) => f64::from(*v.get(index)?),
            Self::I16(v) => f64::from(*v.get(index)?),
            Self::U16(v) => f64::from(*v.get(index)?),
            Self::I32(v) => f64::from(*v.get(index)?),
            Self::U32(v) => f64::from(*v.get(index)?),
            Self::I64(v) => *v.get(index)? as f64,
            Self::U64(v) => *v.get(index)? as f64,
            Self::F32(v) => f64::from(*v.get(index)?),
            Self::F64(v) => *v.get(index)?,
            _ => return None,
        })
    }

    /// Element `index` of an integer array as `i64` (`None` for floats and
    /// for unsigned values above `i64::MAX`).
    pub fn get_i64(&self, index: usize) -> Option<i64> {
        Some(match self {
            Self::I8(v) => i64::from(*v.get(index)?),
            Self::U8(v) => i64::from(*v.get(index)?),
            Self::I16(v) => i64::from(*v.get(index)?),
            Self::U16(v) => i64::from(*v.get(index)?),
            Self::I32(v) => i64::from(*v.get(index)?),
            Self::U32(v) => i64::from(*v.get(index)?),
            Self::I64(v) => *v.get(index)?,
            Self::U64(v) => i64::try_from(*v.get(index)?).ok()?,
            _ => return None,
        })
    }

    /// Every element of a numeric array widened to `f64`.
    pub fn to_f64_vec(&self) -> Option<Vec<f64>> {
        if !self.is_numeric() {
            return None;
        }
        (0..self.len()).map(|index| self.get_f64(index)).collect()
    }

    /// True for the integer and floating-point variants.
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Self::I8(_)
                | Self::U8(_)
                | Self::I16(_)
                | Self::U16(_)
                | Self::I32(_)
                | Self::U32(_)
                | Self::I64(_)
                | Self::U64(_)
                | Self::F32(_)
                | Self::F64(_)
        )
    }

    /// The elements of a string array as text: fixed-length strings cut at
    /// their first NUL (space-padded ones also lose trailing spaces), lossy
    /// UTF-8. `None` for non-string values.
    pub fn strings(&self) -> Option<Vec<String>> {
        match self {
            Self::VarStrings(strings) => Some(strings.clone()),
            Self::FixedStrings {
                size,
                padding,
                bytes,
            } => {
                if *size == 0 {
                    return Some(Vec::new());
                }
                Some(
                    bytes
                        .chunks(*size)
                        .map(|raw| fixed_string(raw, *padding))
                        .collect(),
                )
            }
            _ => None,
        }
    }
}

/// Text of one fixed-length string element.
pub(crate) fn fixed_string(raw: &[u8], padding: StringPadding) -> String {
    let text = raw.split(|byte| *byte == 0).next().unwrap_or_default();
    let text = if padding == StringPadding::SpacePad {
        let end = text
            .iter()
            .rposition(|byte| *byte != b' ')
            .map_or(0, |last| last + 1);
        &text[..end]
    } else {
        text
    };
    String::from_utf8_lossy(text).into_owned()
}

/// Shared state for decoding values: the file and a global heap cache.
pub(crate) struct Decoder<'a, 'h> {
    pub(crate) space: Space<'a>,
    pub(crate) heaps: &'h mut GlobalHeaps<'a>,
    /// Variable-length bytes still allowed for this value.
    pub(crate) vlen_budget: usize,
}

impl<'a, 'h> Decoder<'a, 'h> {
    pub(crate) fn new(space: Space<'a>, heaps: &'h mut GlobalHeaps<'a>) -> Self {
        Self {
            space,
            heaps,
            vlen_budget: MAX_VLEN_BYTES,
        }
    }

    /// Decode `count` elements of `datatype` from `raw` (at least
    /// `count * datatype.size()` bytes).
    pub(crate) fn decode(
        &mut self,
        datatype: &Datatype,
        raw: &[u8],
        count: usize,
    ) -> Result<Values> {
        let size = datatype.size();
        let needed = count
            .checked_mul(size)
            .ok_or_else(|| invalid(0, "HDF5 value size overflow"))?;
        let raw = raw.get(..needed).ok_or_else(|| {
            invalid(
                0,
                format!("HDF5 value data too short: {} < {needed}", raw.len()),
            )
        })?;
        match datatype {
            Datatype::Integer {
                size,
                signed,
                order,
                bit_offset,
                precision,
            } => Ok(integers(
                raw,
                *size,
                *signed,
                *order,
                *bit_offset,
                *precision,
            )),
            Datatype::Bitfield {
                size,
                order,
                bit_offset,
                precision,
            } => Ok(integers(raw, *size, false, *order, *bit_offset, *precision)),
            Datatype::Enum { base, .. } => self.decode(base, raw, count),
            Datatype::Float { size: 4, order } => Ok(Values::F32(
                raw.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| {
                        f32::from_bits(if *order == ByteOrder::BigEndian {
                            u32::from_be_bytes(*word)
                        } else {
                            u32::from_le_bytes(*word)
                        })
                    })
                    .collect(),
            )),
            Datatype::Float { size: 8, order } => Ok(Values::F64(
                raw.as_chunks::<8>()
                    .0
                    .iter()
                    .map(|word| {
                        f64::from_bits(if *order == ByteOrder::BigEndian {
                            u64::from_be_bytes(*word)
                        } else {
                            u64::from_le_bytes(*word)
                        })
                    })
                    .collect(),
            )),
            Datatype::FixedString { size, padding, .. } => Ok(Values::FixedStrings {
                size: *size,
                padding: *padding,
                bytes: raw.to_vec(),
            }),
            Datatype::VarLenString { size, .. } => {
                let mut strings = Vec::with_capacity(count);
                for element in raw.chunks_exact((*size).max(1)).take(count) {
                    let bytes = self.vlen_bytes(element, 1)?;
                    let text = bytes.split(|byte| *byte == 0).next().unwrap_or_default();
                    strings.push(String::from_utf8_lossy(text).into_owned());
                }
                Ok(Values::VarStrings(strings))
            }
            Datatype::VarLenSequence { size, base } => {
                let mut sequences = Vec::with_capacity(count);
                let base_size = base.size();
                for element in raw.chunks_exact((*size).max(1)).take(count) {
                    let length = read_le_u32(element, 0)? as usize;
                    let bytes = self.vlen_bytes(element, base_size)?;
                    sequences.push(self.decode(base, bytes, length)?);
                }
                Ok(Values::Sequences(sequences))
            }
            Datatype::Reference {
                size,
                kind: crate::datatype::ReferenceKind::Object,
            } if *size == self.space.offset_size => Ok(Values::References(
                raw.chunks_exact(*size)
                    .map(|chunk| {
                        let address = normalize_addr(crate::bytes::le_uint(chunk), *size);
                        (address != crate::bytes::UNDEFINED_ADDR && address != 0).then_some(address)
                    })
                    .collect(),
            )),
            Datatype::Compound { size, members } => {
                let mut columns = Vec::with_capacity(members.len());
                for member in members {
                    let member_size = member.datatype.size();
                    let mut column = Vec::with_capacity(member_size.saturating_mul(count));
                    for element in raw.chunks_exact((*size).max(1)).take(count) {
                        let bytes = element
                            .get(member.offset..member.offset + member_size)
                            .ok_or_else(|| invalid(0, "HDF5 compound member out of range"))?;
                        column.extend_from_slice(bytes);
                    }
                    let values = self.decode(&member.datatype, &column, count)?;
                    columns.push((member.name.clone(), values));
                }
                Ok(Values::Compound {
                    elements: raw.chunks_exact((*size).max(1)).take(count).len(),
                    members: columns,
                })
            }
            Datatype::Array { dims, base } => {
                let per_element = dims
                    .iter()
                    .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
                    .ok_or_else(|| invalid(0, "HDF5 array datatype overflow"))?;
                let total = count
                    .checked_mul(per_element)
                    .ok_or_else(|| invalid(0, "HDF5 array element count overflow"))?;
                self.decode(base, raw, total)
            }
            _ => Ok(Values::Raw {
                size,
                bytes: raw.to_vec(),
            }),
        }
    }

    /// The heap bytes of one variable-length element: sequence length (u32,
    /// in base elements), collection address (O), object index (u32).
    fn vlen_bytes(&mut self, element: &[u8], base_size: usize) -> Result<&'a [u8]> {
        let offset_size = self.space.offset_size;
        let length = read_le_u32(element, 0)? as usize;
        let collection = normalize_addr(read_uint(element, 4, offset_size)?, offset_size);
        let index = read_le_u32(element, 4 + offset_size)?;
        if length == 0 || collection == crate::bytes::UNDEFINED_ADDR || collection == 0 {
            return Ok(&[]);
        }
        let bytes = length
            .checked_mul(base_size)
            .ok_or_else(|| invalid(0, "HDF5 variable-length size overflow"))?;
        if bytes > self.vlen_budget {
            return Err(limit(format!(
                "HDF5 variable-length data exceeds {MAX_VLEN_BYTES} bytes (limit)"
            )));
        }
        self.vlen_budget -= bytes;
        let object = self.heaps.object(&self.space, collection, index)?;
        object.get(..bytes).ok_or_else(|| {
            invalid(
                0,
                format!(
                    "global heap object {index} holds {} bytes, element needs {bytes}",
                    object.len()
                ),
            )
        })
    }
}

/// Integers of 1-8 bytes into the matching [`Values`] variant; a bit offset
/// or precision narrower than the storage is applied with shift and mask.
fn integers(
    raw: &[u8],
    size: usize,
    signed: bool,
    order: ByteOrder,
    bit_offset: u16,
    precision: u16,
) -> Values {
    let big = order == ByteOrder::BigEndian;
    let full = bit_offset == 0 && usize::from(precision) == size * 8;
    if !full {
        let precision = u32::from(precision).clamp(1, 64);
        let shift = u32::from(bit_offset).min(63);
        let values: Vec<u128> = raw
            .chunks_exact(size)
            .map(|chunk| int_value(chunk, false, big) as u128)
            .collect();
        let mask: u128 = (1u128 << precision) - 1;
        let decoded = values.into_iter().map(|value| {
            let value = (value >> shift) & mask;
            if signed && value & (1u128 << (precision - 1)) != 0 {
                (value | !mask) as i128
            } else {
                value as i128
            }
        });
        return if signed {
            Values::I64(decoded.map(|value| value as i64).collect())
        } else {
            Values::U64(decoded.map(|value| value as u64).collect())
        };
    }
    macro_rules! fixed {
        ($variant:ident, $ty:ty, $n:literal) => {
            Values::$variant(
                raw.as_chunks::<$n>()
                    .0
                    .iter()
                    .map(|word| {
                        if big {
                            <$ty>::from_be_bytes(*word)
                        } else {
                            <$ty>::from_le_bytes(*word)
                        }
                    })
                    .collect(),
            )
        };
    }
    match (size, signed) {
        (1, false) => Values::U8(raw.to_vec()),
        (1, true) => Values::I8(raw.iter().map(|byte| *byte as i8).collect()),
        (2, false) => fixed!(U16, u16, 2),
        (2, true) => fixed!(I16, i16, 2),
        (4, false) => fixed!(U32, u32, 4),
        (4, true) => fixed!(I32, i32, 4),
        (8, false) => fixed!(U64, u64, 8),
        (8, true) => fixed!(I64, i64, 8),
        (_, true) => Values::I64(
            raw.chunks_exact(size)
                .map(|chunk| int_value(chunk, true, big) as i64)
                .collect(),
        ),
        (_, false) => Values::U64(
            raw.chunks_exact(size)
                .map(|chunk| int_value(chunk, false, big) as u64)
                .collect(),
        ),
    }
}
