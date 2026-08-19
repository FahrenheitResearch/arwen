//! The classic format's external types, its three container versions, and
//! the two value carriers (attribute values and variable data).

/// Which classic container to emit.
///
/// The three differ in exactly two axes -- the width of the `begin`
/// offset field and the width of every count field -- plus the extended
/// type set CDF-5 adds.  Everything else in the grammar is shared, which
/// is why one serializer covers all three.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcFormat {
    /// CDF-1, "classic": 32-bit offsets, 32-bit counts.
    Classic,
    /// CDF-2, "64-bit offset": 64-bit offsets, 32-bit counts.
    Offset64,
    /// CDF-5, "64-bit data": 64-bit offsets, 64-bit counts, and the
    /// unsigned/64-bit external types.
    Cdf5,
}

impl NcFormat {
    /// The version byte after the `CDF` magic.
    pub(crate) fn magic_byte(self) -> u8 {
        match self {
            NcFormat::Classic => 1,
            NcFormat::Offset64 => 2,
            NcFormat::Cdf5 => 5,
        }
    }

    /// Are count fields (`nelems`, name lengths, dim lengths, `vsize`,
    /// `numrecs`, `ndims`, dimids) 8 bytes rather than 4?
    pub(crate) fn wide_counts(self) -> bool {
        matches!(self, NcFormat::Cdf5)
    }

    /// Is the `begin` offset field 8 bytes rather than 4?
    pub(crate) fn wide_offsets(self) -> bool {
        !matches!(self, NcFormat::Classic)
    }

    /// The name this format is known by in the wild, for messages.
    pub(crate) fn label(self) -> &'static str {
        match self {
            NcFormat::Classic => "CDF-1",
            NcFormat::Offset64 => "CDF-2",
            NcFormat::Cdf5 => "CDF-5",
        }
    }
}

/// A classic external data type.
///
/// The first six exist in every container version; the last five are
/// CDF-5 only, and writing one into a CDF-1/CDF-2 file produces a type
/// code older readers do not know.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcType {
    Byte,
    Char,
    Short,
    Int,
    Float,
    Double,
    UByte,
    UShort,
    UInt,
    Int64,
    UInt64,
}

impl NcType {
    /// The on-disk type code.
    pub(crate) fn code(self) -> u32 {
        match self {
            NcType::Byte => 1,
            NcType::Char => 2,
            NcType::Short => 3,
            NcType::Int => 4,
            NcType::Float => 5,
            NcType::Double => 6,
            NcType::UByte => 7,
            NcType::UShort => 8,
            NcType::UInt => 9,
            NcType::Int64 => 10,
            NcType::UInt64 => 11,
        }
    }

    /// Bytes per element on disk.
    pub fn size(self) -> usize {
        match self {
            NcType::Byte | NcType::Char | NcType::UByte => 1,
            NcType::Short | NcType::UShort => 2,
            NcType::Int | NcType::Float | NcType::UInt => 4,
            NcType::Double | NcType::Int64 | NcType::UInt64 => 8,
        }
    }

    /// The `NC_*` spelling, for messages that have to be greppable
    /// against the spec and against netCDF-C's own diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            NcType::Byte => "NC_BYTE",
            NcType::Char => "NC_CHAR",
            NcType::Short => "NC_SHORT",
            NcType::Int => "NC_INT",
            NcType::Float => "NC_FLOAT",
            NcType::Double => "NC_DOUBLE",
            NcType::UByte => "NC_UBYTE",
            NcType::UShort => "NC_USHORT",
            NcType::UInt => "NC_UINT",
            NcType::Int64 => "NC_INT64",
            NcType::UInt64 => "NC_UINT64",
        }
    }

    /// Does this type exist only in CDF-5?
    pub(crate) fn is_cdf5_only(self) -> bool {
        matches!(
            self,
            NcType::UByte | NcType::UShort | NcType::UInt | NcType::Int64 | NcType::UInt64
        )
    }

    /// Reconstruct from an on-disk type code, for the C ABI seam.
    pub fn from_code(code: u32) -> Option<NcType> {
        Some(match code {
            1 => NcType::Byte,
            2 => NcType::Char,
            3 => NcType::Short,
            4 => NcType::Int,
            5 => NcType::Float,
            6 => NcType::Double,
            7 => NcType::UByte,
            8 => NcType::UShort,
            9 => NcType::UInt,
            10 => NcType::Int64,
            11 => NcType::UInt64,
            _ => return None,
        })
    }
}

/// An attribute value: a typed vector, or text.
///
/// `Text` is `NC_CHAR` with `nelems` equal to the BYTE count -- the same
/// encoding netCDF-C uses for a string attribute, which is why a WRF
/// `TITLE` written here reads back as a string and not as an array of
/// one-character strings.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    Text(String),
    Bytes(Vec<i8>),
    Shorts(Vec<i16>),
    Ints(Vec<i32>),
    Floats(Vec<f32>),
    Doubles(Vec<f64>),
    UBytes(Vec<u8>),
    UShorts(Vec<u16>),
    UInts(Vec<u32>),
    Int64s(Vec<i64>),
    UInt64s(Vec<u64>),
}

impl AttrValue {
    /// The external type this value is written as.
    pub fn nc_type(&self) -> NcType {
        match self {
            AttrValue::Text(_) => NcType::Char,
            AttrValue::Bytes(_) => NcType::Byte,
            AttrValue::Shorts(_) => NcType::Short,
            AttrValue::Ints(_) => NcType::Int,
            AttrValue::Floats(_) => NcType::Float,
            AttrValue::Doubles(_) => NcType::Double,
            AttrValue::UBytes(_) => NcType::UByte,
            AttrValue::UShorts(_) => NcType::UShort,
            AttrValue::UInts(_) => NcType::UInt,
            AttrValue::Int64s(_) => NcType::Int64,
            AttrValue::UInt64s(_) => NcType::UInt64,
        }
    }

    /// `nelems` as the file records it: the byte count for text, the
    /// element count for everything else.
    pub fn len(&self) -> usize {
        match self {
            // BYTES, not characters: the file records `nelems` for an
            // NC_CHAR attribute as its byte length.
            AttrValue::Text(s) => s.len(),
            AttrValue::Bytes(v) => v.len(),
            AttrValue::Shorts(v) => v.len(),
            AttrValue::Ints(v) => v.len(),
            AttrValue::Floats(v) => v.len(),
            AttrValue::Doubles(v) => v.len(),
            AttrValue::UBytes(v) => v.len(),
            AttrValue::UShorts(v) => v.len(),
            AttrValue::UInts(v) => v.len(),
            AttrValue::Int64s(v) => v.len(),
            AttrValue::UInt64s(v) => v.len(),
        }
    }

    /// Is this a zero-length value?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append the big-endian value bytes, unpadded.
    pub(crate) fn encode_be(&self, buf: &mut Vec<u8>) {
        macro_rules! push {
            ($vals:expr) => {
                for value in $vals {
                    buf.extend_from_slice(&value.to_be_bytes());
                }
            };
        }
        match self {
            AttrValue::Text(s) => buf.extend_from_slice(s.as_bytes()),
            AttrValue::Bytes(v) => push!(v),
            AttrValue::Shorts(v) => push!(v),
            AttrValue::Ints(v) => push!(v),
            AttrValue::Floats(v) => push!(v),
            AttrValue::Doubles(v) => push!(v),
            AttrValue::UBytes(v) => push!(v),
            AttrValue::UShorts(v) => push!(v),
            AttrValue::UInts(v) => push!(v),
            AttrValue::Int64s(v) => push!(v),
            AttrValue::UInt64s(v) => push!(v),
        }
    }
}

/// One variable's values, borrowed, in the caller's native endianness.
///
/// Borrowed rather than owned because the product tape hands over slabs
/// that are already tens of megabytes; a writer that took `Vec` would
/// double the peak footprint of every frame.
#[derive(Debug, Clone, Copy)]
pub enum VarData<'a> {
    I8(&'a [i8]),
    Char(&'a [u8]),
    I16(&'a [i16]),
    I32(&'a [i32]),
    F32(&'a [f32]),
    F64(&'a [f64]),
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    I64(&'a [i64]),
    U64(&'a [u64]),
}

impl VarData<'_> {
    /// The external type this payload can be written as.
    pub fn nc_type(&self) -> NcType {
        match self {
            VarData::I8(_) => NcType::Byte,
            VarData::Char(_) => NcType::Char,
            VarData::I16(_) => NcType::Short,
            VarData::I32(_) => NcType::Int,
            VarData::F32(_) => NcType::Float,
            VarData::F64(_) => NcType::Double,
            VarData::U8(_) => NcType::UByte,
            VarData::U16(_) => NcType::UShort,
            VarData::U32(_) => NcType::UInt,
            VarData::I64(_) => NcType::Int64,
            VarData::U64(_) => NcType::UInt64,
        }
    }

    /// Element count.
    pub fn len(&self) -> usize {
        match self {
            VarData::I8(v) => v.len(),
            VarData::Char(v) => v.len(),
            VarData::I16(v) => v.len(),
            VarData::I32(v) => v.len(),
            VarData::F32(v) => v.len(),
            VarData::F64(v) => v.len(),
            VarData::U8(v) => v.len(),
            VarData::U16(v) => v.len(),
            VarData::U32(v) => v.len(),
            VarData::I64(v) => v.len(),
            VarData::U64(v) => v.len(),
        }
    }

    /// Is this an empty payload?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
