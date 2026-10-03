//! Datatype messages (HDF5 File Format Specification, section IV.A.2.d).

use crate::bytes::{Cursor, limit_enc_size};
use crate::error::{Result, invalid, limit};
use crate::limits::{MAX_ATTRIBUTE_BYTES, MAX_DATASPACE_RANK, MAX_DATATYPE_DEPTH};

/// Byte order of a numeric datatype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ByteOrder {
    /// Least significant byte first.
    LittleEndian,
    /// Most significant byte first.
    BigEndian,
}

/// How a string fills its storage (string datatype bits 0-3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StringPadding {
    /// NUL-terminated; bytes after the terminator are undefined.
    NullTerminate,
    /// Padded with NUL bytes.
    NullPad,
    /// Padded with spaces (Fortran style).
    SpacePad,
    /// A padding code this crate does not know (kept verbatim).
    Other(u8),
}

/// Character set of a string (string datatype bits 4-7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CharSet {
    /// US-ASCII.
    Ascii,
    /// UTF-8.
    Utf8,
    /// A character set code this crate does not know.
    Other(u8),
}

/// What a reference datatype points at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReferenceKind {
    /// An object (its object header address).
    Object,
    /// A dataset region (a global heap entry).
    Region,
    /// A reference type of the revised (1.12+) encoding, by its type code.
    Other(u8),
}

/// One member of a compound datatype.
#[derive(Clone, Debug, PartialEq)]
pub struct CompoundMember {
    /// Member name.
    pub name: String,
    /// Byte offset of the member inside each element.
    pub offset: usize,
    /// Member datatype.
    pub datatype: Datatype,
}

/// One named value of an enumeration datatype.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumMember {
    /// Member name.
    pub name: String,
    /// Member value, decoded with the base integer type.
    pub value: i128,
}

/// An HDF5 datatype.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Datatype {
    /// Fixed-point integer (class 0).
    Integer {
        /// Bytes per element.
        size: usize,
        /// Two's complement signed.
        signed: bool,
        /// Byte order.
        order: ByteOrder,
        /// First significant bit.
        bit_offset: u16,
        /// Number of significant bits.
        precision: u16,
    },
    /// IEEE 754 binary32 or binary64 floating point (class 1).
    Float {
        /// Bytes per element (4 or 8).
        size: usize,
        /// Byte order.
        order: ByteOrder,
    },
    /// Fixed-length string (class 3).
    FixedString {
        /// Bytes per element.
        size: usize,
        /// Padding convention.
        padding: StringPadding,
        /// Character set.
        charset: CharSet,
    },
    /// Bit field (class 4); values read as unsigned integers.
    Bitfield {
        /// Bytes per element.
        size: usize,
        /// Byte order.
        order: ByteOrder,
        /// First significant bit.
        bit_offset: u16,
        /// Number of significant bits.
        precision: u16,
    },
    /// Opaque bytes (class 5).
    Opaque {
        /// Bytes per element.
        size: usize,
        /// Description tag.
        tag: String,
    },
    /// Compound (class 6).
    Compound {
        /// Bytes per element.
        size: usize,
        /// Members in storage order.
        members: Vec<CompoundMember>,
    },
    /// Reference (class 7).
    Reference {
        /// Bytes per element.
        size: usize,
        /// What the reference points at.
        kind: ReferenceKind,
    },
    /// Enumeration (class 8) over an integer base type.
    Enum {
        /// Base integer type.
        base: Box<Datatype>,
        /// Named values.
        members: Vec<EnumMember>,
    },
    /// Variable-length sequence of a base type (class 9, type 0).
    VarLenSequence {
        /// Bytes per stored element (length + global heap ID).
        size: usize,
        /// Element type of each sequence.
        base: Box<Datatype>,
    },
    /// Variable-length string (class 9, type 1).
    VarLenString {
        /// Bytes per stored element (length + global heap ID).
        size: usize,
        /// Padding convention.
        padding: StringPadding,
        /// Character set.
        charset: CharSet,
    },
    /// Fixed-size array of a base type (class 10).
    Array {
        /// Array dimensions.
        dims: Vec<usize>,
        /// Element type.
        base: Box<Datatype>,
    },
    /// A datatype this crate stores but does not interpret: the time class
    /// (2), non-IEEE floating point, or a class code above 10. Values come
    /// back as raw bytes.
    Other {
        /// Datatype class code.
        class: u8,
        /// Bytes per element.
        size: usize,
    },
}

impl Datatype {
    /// Bytes per element as stored.
    pub fn size(&self) -> usize {
        match self {
            Self::Integer { size, .. }
            | Self::Float { size, .. }
            | Self::FixedString { size, .. }
            | Self::Bitfield { size, .. }
            | Self::Opaque { size, .. }
            | Self::Compound { size, .. }
            | Self::Reference { size, .. }
            | Self::VarLenSequence { size, .. }
            | Self::VarLenString { size, .. }
            | Self::Other { size, .. } => *size,
            Self::Enum { base, .. } => base.size(),
            Self::Array { dims, base } => dims
                .iter()
                .try_fold(base.size(), |acc, dim| acc.checked_mul(*dim))
                .unwrap_or(usize::MAX),
        }
    }

    /// True for fixed-length and variable-length strings.
    pub fn is_string(&self) -> bool {
        matches!(self, Self::FixedString { .. } | Self::VarLenString { .. })
    }
}

fn order(bits: u32, at: usize) -> Result<ByteOrder> {
    if bits & 0x40 != 0 {
        return Err(invalid(at, "VAX byte order is unsupported"));
    }
    Ok(if bits & 1 != 0 {
        ByteOrder::BigEndian
    } else {
        ByteOrder::LittleEndian
    })
}

fn padding(code: u8) -> StringPadding {
    match code {
        0 => StringPadding::NullTerminate,
        1 => StringPadding::NullPad,
        2 => StringPadding::SpacePad,
        other => StringPadding::Other(other),
    }
}

fn charset(code: u8) -> CharSet {
    match code {
        0 => CharSet::Ascii,
        1 => CharSet::Utf8,
        other => CharSet::Other(other),
    }
}

/// Parse a complete datatype message body.
pub(crate) fn parse(body: &[u8], offset: usize, offset_size: usize) -> Result<Datatype> {
    let mut cursor = Cursor::new(body, offset);
    parse_at(&mut cursor, offset_size, 0)
}

/// Parse one datatype at the cursor (nested types recurse).
pub(crate) fn parse_at(
    cursor: &mut Cursor<'_>,
    offset_size: usize,
    depth: usize,
) -> Result<Datatype> {
    if depth > MAX_DATATYPE_DEPTH {
        return Err(limit(format!(
            "HDF5 datatype nesting deeper than {MAX_DATATYPE_DEPTH} (limit)"
        )));
    }
    let at = cursor.pos();
    let class_version = cursor.u8()?;
    let class = class_version & 0x0F;
    let version = class_version >> 4;
    let bits =
        u32::from(cursor.u8()?) | (u32::from(cursor.u8()?) << 8) | (u32::from(cursor.u8()?) << 16);
    let size = cursor.u32()? as usize;
    if size > MAX_ATTRIBUTE_BYTES && !matches!(class, 6 | 10) {
        return Err(limit(format!(
            "HDF5 datatype element is {size} bytes (limit {MAX_ATTRIBUTE_BYTES})"
        )));
    }
    let datatype = match class {
        0 => {
            let bit_offset = cursor.u16()?;
            let precision = cursor.u16()?;
            if size == 0 || size > 8 {
                return finish(Datatype::Other { class, size }, offset_size, at);
            }
            Datatype::Integer {
                size,
                signed: bits & 0x08 != 0,
                order: order(bits, at)?,
                bit_offset,
                precision,
            }
        }
        1 => {
            let bit_offset = cursor.u16()?;
            let precision = cursor.u16()?;
            let exponent_location = cursor.u8()?;
            let exponent_size = cursor.u8()?;
            let mantissa_location = cursor.u8()?;
            let mantissa_size = cursor.u8()?;
            let exponent_bias = cursor.u32()?;
            let sign_location = (bits >> 8) & 0xFF;
            let ieee = match size {
                4 => {
                    (bit_offset, precision, exponent_location, exponent_size) == (0, 32, 23, 8)
                        && (
                            mantissa_location,
                            mantissa_size,
                            exponent_bias,
                            sign_location,
                        ) == (0, 23, 127, 31)
                }
                8 => {
                    (bit_offset, precision, exponent_location, exponent_size) == (0, 64, 52, 11)
                        && (
                            mantissa_location,
                            mantissa_size,
                            exponent_bias,
                            sign_location,
                        ) == (0, 52, 1023, 63)
                }
                _ => false,
            };
            if ieee {
                Datatype::Float {
                    size,
                    order: order(bits, at)?,
                }
            } else {
                Datatype::Other { class, size }
            }
        }
        2 => {
            // Time: bit precision (u16).
            cursor.skip(2)?;
            Datatype::Other { class, size }
        }
        3 => Datatype::FixedString {
            size,
            padding: padding((bits & 0x0F) as u8),
            charset: charset(((bits >> 4) & 0x0F) as u8),
        },
        4 => {
            let bit_offset = cursor.u16()?;
            let precision = cursor.u16()?;
            if size == 0 || size > 8 {
                return finish(Datatype::Other { class, size }, offset_size, at);
            }
            Datatype::Bitfield {
                size,
                order: order(bits, at)?,
                bit_offset,
                precision,
            }
        }
        5 => {
            let tag_len = (bits & 0xFF) as usize;
            let tag = cursor.take(tag_len)?;
            let tag = tag.split(|byte| *byte == 0).next().unwrap_or_default();
            Datatype::Opaque {
                size,
                tag: String::from_utf8_lossy(tag).into_owned(),
            }
        }
        6 => parse_compound(cursor, version, bits, size, offset_size, depth)?,
        7 => {
            let code = (bits & 0x0F) as u8;
            let kind = if version >= 4 {
                ReferenceKind::Other(code)
            } else {
                match code {
                    0 => ReferenceKind::Object,
                    1 => ReferenceKind::Region,
                    other => ReferenceKind::Other(other),
                }
            };
            Datatype::Reference { size, kind }
        }
        8 => parse_enum(cursor, version, bits, offset_size, depth)?,
        9 => {
            let base = parse_at(cursor, offset_size, depth + 1)?;
            match bits & 0x0F {
                0 => Datatype::VarLenSequence {
                    size,
                    base: Box::new(base),
                },
                1 => Datatype::VarLenString {
                    size,
                    padding: padding(((bits >> 4) & 0x0F) as u8),
                    charset: charset(((bits >> 8) & 0x0F) as u8),
                },
                other => {
                    return Err(invalid(
                        at,
                        format!("HDF5 variable-length type {other} unknown"),
                    ));
                }
            }
        }
        10 => parse_array(cursor, version, offset_size, depth)?,
        other => Datatype::Other { class: other, size },
    };
    finish(datatype, offset_size, at)
}

/// Final consistency checks shared by every class.
fn finish(datatype: Datatype, offset_size: usize, at: usize) -> Result<Datatype> {
    if let Datatype::VarLenSequence { size, .. } | Datatype::VarLenString { size, .. } = datatype
        && size != 4 + offset_size + 4
    {
        return Err(invalid(
            at,
            format!(
                "HDF5 variable-length element of {size} bytes (need {})",
                8 + offset_size
            ),
        ));
    }
    Ok(datatype)
}

/// Names of compound and enum members: NUL-terminated, padded to a
/// multiple of eight bytes before version 3.
fn member_name(cursor: &mut Cursor<'_>, version: u8) -> Result<String> {
    let start = cursor.pos();
    let name = cursor.c_string()?;
    if version < 3 {
        let used = cursor.pos() - start;
        cursor.skip(used.div_ceil(8) * 8 - used)?;
    }
    Ok(String::from_utf8_lossy(name).into_owned())
}

fn parse_compound(
    cursor: &mut Cursor<'_>,
    version: u8,
    bits: u32,
    size: usize,
    offset_size: usize,
    depth: usize,
) -> Result<Datatype> {
    let count = (bits & 0xFFFF) as usize;
    let mut members = Vec::with_capacity(count.min(cursor.remaining()));
    for _ in 0..count {
        let at = cursor.pos();
        let name = member_name(cursor, version)?;
        let offset = match version {
            1 | 2 => cursor.u32()? as usize,
            _ => cursor.length(limit_enc_size(size as u64), "compound member offset")?,
        };
        let datatype = if version == 1 {
            let rank = usize::from(cursor.u8()?);
            cursor.skip(3 + 4 + 4)?;
            let mut dims = Vec::with_capacity(4);
            for _ in 0..4 {
                dims.push(cursor.u32()? as usize);
            }
            let base = parse_at(cursor, offset_size, depth + 1)?;
            if rank == 0 {
                base
            } else {
                if rank > 4 {
                    return Err(invalid(at, format!("compound member rank {rank} > 4")));
                }
                dims.truncate(rank);
                Datatype::Array {
                    dims,
                    base: Box::new(base),
                }
            }
        } else {
            parse_at(cursor, offset_size, depth + 1)?
        };
        let end = offset.checked_add(datatype.size());
        if end.is_none_or(|end| end > size) {
            return Err(invalid(
                at,
                format!("compound member '{name}' lies outside its {size}-byte element"),
            ));
        }
        members.push(CompoundMember {
            name,
            offset,
            datatype,
        });
    }
    Ok(Datatype::Compound { size, members })
}

fn parse_enum(
    cursor: &mut Cursor<'_>,
    version: u8,
    bits: u32,
    offset_size: usize,
    depth: usize,
) -> Result<Datatype> {
    let at = cursor.pos();
    let count = (bits & 0xFFFF) as usize;
    let base = parse_at(cursor, offset_size, depth + 1)?;
    let (base_size, signed, big_endian) = match &base {
        Datatype::Integer {
            size,
            signed,
            order,
            ..
        } => (*size, *signed, *order == ByteOrder::BigEndian),
        _ => return Err(invalid(at, "HDF5 enumeration over a non-integer base type")),
    };
    let mut names = Vec::with_capacity(count.min(cursor.remaining()));
    for _ in 0..count {
        names.push(member_name(cursor, version)?);
    }
    let mut members = Vec::with_capacity(names.len());
    for name in names {
        let raw = cursor.take(base_size)?;
        members.push(EnumMember {
            name,
            value: int_value(raw, signed, big_endian),
        });
    }
    Ok(Datatype::Enum {
        base: Box::new(base),
        members,
    })
}

fn parse_array(
    cursor: &mut Cursor<'_>,
    version: u8,
    offset_size: usize,
    depth: usize,
) -> Result<Datatype> {
    let at = cursor.pos();
    let rank = usize::from(cursor.u8()?);
    if rank == 0 || rank > MAX_DATASPACE_RANK {
        return Err(invalid(at, format!("HDF5 array datatype of rank {rank}")));
    }
    if version < 3 {
        cursor.skip(3)?;
    }
    let mut dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        dims.push(cursor.u32()? as usize);
    }
    if version < 3 {
        cursor.skip(4 * rank)?;
    }
    let base = parse_at(cursor, offset_size, depth + 1)?;
    let datatype = Datatype::Array {
        dims,
        base: Box::new(base),
    };
    if datatype.size() > MAX_ATTRIBUTE_BYTES {
        return Err(limit(format!(
            "HDF5 array datatype element exceeds {MAX_ATTRIBUTE_BYTES} bytes (limit)"
        )));
    }
    Ok(datatype)
}

/// An integer of up to 16 bytes, sign-extended when `signed`.
pub(crate) fn int_value(raw: &[u8], signed: bool, big_endian: bool) -> i128 {
    let mut value: u128 = 0;
    let len = raw.len().min(16);
    if big_endian {
        for byte in &raw[..len] {
            value = (value << 8) | u128::from(*byte);
        }
    } else {
        for (index, byte) in raw[..len].iter().enumerate() {
            value |= u128::from(*byte) << (8 * index);
        }
    }
    if signed && len > 0 && len < 16 {
        let sign_bit = 1u128 << (8 * len - 1);
        if value & sign_bit != 0 {
            value |= !((1u128 << (8 * len)) - 1);
        }
    }
    value as i128
}
