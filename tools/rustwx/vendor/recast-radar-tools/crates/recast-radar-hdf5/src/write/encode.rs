//! Byte encodings of the structures the writer emits (HDF5 File Format
//! Specification Version 3.0): datatype, dataspace, attribute, link, link
//! info, group info, attribute info, fill value, layout and filter pipeline
//! messages, version 2 object headers and the version 2 superblock.

use super::{CharSet, Data, Shape, StringPadding};
use crate::checksum::lookup3;

/// The undefined address (all bits set, 8-byte offsets).
pub(super) const UNDEF: u64 = u64::MAX;

/// Size of the version 2 superblock with 8-byte offsets and lengths.
pub(super) const SUPERBLOCK_SIZE: usize = 48;

/// Header message types.
pub(super) const MSG_DATASPACE: u8 = 0x01;
pub(super) const MSG_LINK_INFO: u8 = 0x02;
pub(super) const MSG_DATATYPE: u8 = 0x03;
pub(super) const MSG_FILL: u8 = 0x05;
pub(super) const MSG_LINK: u8 = 0x06;
pub(super) const MSG_LAYOUT: u8 = 0x08;
pub(super) const MSG_GROUP_INFO: u8 = 0x0A;
pub(super) const MSG_FILTERS: u8 = 0x0B;
pub(super) const MSG_ATTRIBUTE: u8 = 0x0C;
pub(super) const MSG_ATTRIBUTE_INFO: u8 = 0x15;

/// Header message flag bit 0: the message is constant.
pub(super) const FLAG_CONSTANT: u8 = 0x01;

/// One header message.
pub(super) struct Message {
    pub(super) kind: u8,
    pub(super) flags: u8,
    /// Creation order (attribute messages: the attribute's index).
    pub(super) creation_order: u16,
    pub(super) body: Vec<u8>,
}

impl Message {
    pub(super) fn new(kind: u8, body: Vec<u8>) -> Self {
        Self {
            kind,
            flags: 0,
            creation_order: 0,
            body,
        }
    }

    pub(super) fn constant(kind: u8, body: Vec<u8>) -> Self {
        Self {
            kind,
            flags: FLAG_CONSTANT,
            creation_order: 0,
            body,
        }
    }
}

fn padding_code(padding: StringPadding) -> u8 {
    match padding {
        StringPadding::NullTerminate => 0,
        StringPadding::NullPad => 1,
        StringPadding::SpacePad => 2,
        StringPadding::Other(code) => code & 0x0F,
    }
}

fn charset_code(charset: CharSet) -> u8 {
    match charset {
        CharSet::Ascii => 0,
        CharSet::Utf8 => 1,
        CharSet::Other(code) => code & 0x0F,
    }
}

/// Fixed-point datatype (class 0, version 1), little-endian.
pub(super) fn int_type(size: u32, signed: bool) -> Vec<u8> {
    let mut out = vec![0x10, if signed { 0x08 } else { 0x00 }, 0, 0];
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&((size * 8) as u16).to_le_bytes());
    out
}

/// IEEE floating-point datatype (class 1, version 1), little-endian.
pub(super) fn float_type(size: u32) -> Vec<u8> {
    // Sign bit, exponent location and size, mantissa size, exponent bias.
    let (sign, exponent_at, exponent_bits, mantissa_bits, bias) = if size == 4 {
        (31u8, 23u8, 8u8, 23u8, 127u32)
    } else {
        (63, 52, 11, 52, 1023)
    };
    // Bits 4-5 = 2: the mantissa's most significant bit is implied.
    let mut out = vec![0x11, 0x20, sign, 0];
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&((size * 8) as u16).to_le_bytes());
    out.extend_from_slice(&[exponent_at, exponent_bits, 0, mantissa_bits]);
    out.extend_from_slice(&bias.to_le_bytes());
    out
}

/// Fixed-length string datatype (class 3, version 1).
pub(super) fn string_type(size: u32, padding: StringPadding, charset: CharSet) -> Vec<u8> {
    let mut out = vec![
        0x13,
        padding_code(padding) | (charset_code(charset) << 4),
        0,
        0,
    ];
    out.extend_from_slice(&size.to_le_bytes());
    out
}

/// Object reference datatype (class 7, version 1).
pub(super) fn object_reference_type() -> Vec<u8> {
    let mut out = vec![0x17, 0, 0, 0];
    out.extend_from_slice(&8u32.to_le_bytes());
    out
}

/// Variable-length string datatype (class 9, version 1): NUL-terminated,
/// over `unsigned char` as the HDF5 library encodes it.
pub(super) fn var_string_type(charset: CharSet) -> Vec<u8> {
    let mut out = vec![0x19, 0x01, charset_code(charset), 0];
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&int_type(1, false));
    out
}

/// Variable-length sequence of object references (class 9, version 1).
pub(super) fn reference_sequence_type() -> Vec<u8> {
    let mut out = vec![0x19, 0x00, 0, 0];
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&object_reference_type());
    out
}

/// The `REFERENCE_LIST` compound of the HDF5 dimension scale convention
/// (class 6, version 3): `dataset` (object reference) at 0, `dimension`
/// (int32) at 8, 16 bytes per element (the C struct's size).
pub(super) fn reference_list_type() -> Vec<u8> {
    let mut out = vec![0x36, 2, 0, 0];
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(b"dataset\0");
    out.push(0);
    out.extend_from_slice(&object_reference_type());
    out.extend_from_slice(b"dimension\0");
    out.push(8);
    out.extend_from_slice(&int_type(4, true));
    out
}

/// The datatype message body of `data`.
pub(super) fn datatype(data: &Data) -> Vec<u8> {
    match data {
        Data::I8(_) => int_type(1, true),
        Data::U8(_) => int_type(1, false),
        Data::I16(_) => int_type(2, true),
        Data::U16(_) => int_type(2, false),
        Data::I32(_) => int_type(4, true),
        Data::U32(_) => int_type(4, false),
        Data::I64(_) => int_type(8, true),
        Data::U64(_) => int_type(8, false),
        Data::F32(_) => float_type(4),
        Data::F64(_) => float_type(8),
        Data::FixedStrings {
            size,
            padding,
            charset,
            ..
        } => string_type(*size as u32, *padding, *charset),
        Data::VarStrings { charset, .. } => var_string_type(*charset),
        Data::ObjectRefs(_) => object_reference_type(),
        Data::ObjectRefLists(_) => reference_sequence_type(),
        Data::DimensionScaleRefs(_) => reference_list_type(),
        Data::Bools(_) => bool_enum_type(),
        Data::Compound(members) => compound_type(members),
    }
}

/// h5py's boolean: an enumeration (class 8, version 3) of `int8` with the
/// members `FALSE` = 0 and `TRUE` = 1.
pub(super) fn bool_enum_type() -> Vec<u8> {
    let mut out = vec![0x38, 2, 0, 0];
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&int_type(1, true));
    out.extend_from_slice(b"FALSE\0TRUE\0");
    out.extend_from_slice(&[0, 1]);
    out
}

/// A packed compound (class 6, version 3) of `members`.
pub(super) fn compound_type(members: &[(String, Data)]) -> Vec<u8> {
    let size: usize = members
        .iter()
        .map(|(_, column)| column.element_size())
        .sum();
    let offset_bytes = if size < 1 << 8 {
        1
    } else if size < 1 << 16 {
        2
    } else if size < 1 << 24 {
        3
    } else {
        4
    };
    let count = members.len() as u16;
    let mut out = vec![0x36, count.to_le_bytes()[0], count.to_le_bytes()[1], 0];
    out.extend_from_slice(&(size as u32).to_le_bytes());
    let mut offset = 0usize;
    for (name, column) in members {
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        out.extend_from_slice(&(offset as u32).to_le_bytes()[..offset_bytes]);
        out.extend_from_slice(&datatype(column));
        offset += column.element_size();
    }
    out
}

/// Dataspace message body (version 2).
pub(super) fn dataspace(shape: &Shape) -> Vec<u8> {
    match shape {
        Shape::Scalar => vec![2, 0, 0, 0],
        Shape::Null => vec![2, 0, 0, 2],
        Shape::Simple(dims) => {
            let mut out = vec![2, dims.len() as u8, 0, 1];
            for dim in dims {
                out.extend_from_slice(&dim.to_le_bytes());
            }
            out
        }
    }
}

/// Attribute message body (version 3).
pub(super) fn attribute(
    name: &str,
    datatype: &[u8],
    dataspace: &[u8],
    utf8_name: bool,
    data: &[u8],
) -> Vec<u8> {
    let mut out =
        Vec::with_capacity(9 + name.len() + 1 + datatype.len() + dataspace.len() + data.len());
    out.push(3);
    out.push(0);
    out.extend_from_slice(&((name.len() + 1) as u16).to_le_bytes());
    out.extend_from_slice(&(datatype.len() as u16).to_le_bytes());
    out.extend_from_slice(&(dataspace.len() as u16).to_le_bytes());
    out.push(u8::from(utf8_name));
    out.extend_from_slice(name.as_bytes());
    out.push(0);
    out.extend_from_slice(datatype);
    out.extend_from_slice(dataspace);
    out.extend_from_slice(data);
    out
}

/// Link message body (version 1): a hard link with its creation order.
pub(super) fn hard_link(name: &str, creation_order: u64, address: u64) -> Vec<u8> {
    let utf8 = !name.is_ascii();
    let length_bytes: usize = if name.len() < 256 {
        1
    } else if name.len() < 65_536 {
        2
    } else {
        4
    };
    let size_code: u8 = match length_bytes {
        1 => 0,
        2 => 1,
        _ => 2,
    };
    // Bit 2: creation order present; bit 4: character set present.
    let flags = size_code | 0x04 | if utf8 { 0x10 } else { 0 };
    let mut out = vec![1, flags];
    out.extend_from_slice(&creation_order.to_le_bytes());
    if utf8 {
        out.push(1);
    }
    out.extend_from_slice(&(name.len() as u64).to_le_bytes()[..length_bytes]);
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(&address.to_le_bytes());
    out
}

/// Link info message body (version 0): compact storage, creation order
/// tracked (not indexed).
pub(super) fn link_info(links: u64) -> Vec<u8> {
    let mut out = vec![0, 0x01];
    out.extend_from_slice(&links.to_le_bytes());
    out.extend_from_slice(&UNDEF.to_le_bytes());
    out.extend_from_slice(&UNDEF.to_le_bytes());
    out
}

/// Group info message body (version 0). Groups with more links than the
/// library default of 8 store a compact limit that holds them all.
pub(super) fn group_info(links: usize) -> Vec<u8> {
    if links <= 8 {
        return vec![0, 0];
    }
    let mut out = vec![0, 0x01];
    out.extend_from_slice(&(links as u16).to_le_bytes());
    out.extend_from_slice(&6u16.to_le_bytes());
    out
}

/// Attribute info message body (version 0): compact storage, creation order
/// tracked (not indexed).
pub(super) fn attribute_info(attributes: u16) -> Vec<u8> {
    let mut out = vec![0, 0x01];
    out.extend_from_slice(&attributes.to_le_bytes());
    out.extend_from_slice(&UNDEF.to_le_bytes());
    out.extend_from_slice(&UNDEF.to_le_bytes());
    out
}

/// Space allocation times of the fill value message.
#[derive(Clone, Copy)]
pub(super) enum AllocTime {
    Early = 1,
    Late = 2,
    Incremental = 3,
}

/// Fill value message body (version 3): fill written "if set"; the value
/// when defined, else the library default (zeros).
pub(super) fn fill_value(alloc: AllocTime, value: Option<&[u8]>) -> Vec<u8> {
    // Bits 0-1 allocation time; bits 2-3 fill write time (2 = if set);
    // bit 5 a value follows.
    let mut flags = alloc as u8 | (2 << 2);
    let mut out = Vec::new();
    if let Some(value) = value {
        flags |= 0x20;
        out.push(3);
        out.push(flags);
        out.extend_from_slice(&(value.len() as u32).to_le_bytes());
        out.extend_from_slice(value);
    } else {
        out.push(3);
        out.push(flags);
    }
    out
}

/// Contiguous layout message body (version 3).
pub(super) fn contiguous_layout(address: u64, size: u64) -> Vec<u8> {
    let mut out = vec![3, 1];
    out.extend_from_slice(&address.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
    out
}

/// Compact layout message body (version 3).
pub(super) fn compact_layout(raw: &[u8]) -> Vec<u8> {
    let mut out = vec![3, 0];
    out.extend_from_slice(&(raw.len() as u16).to_le_bytes());
    out.extend_from_slice(raw);
    out
}

/// Chunked layout message body (version 3, v1 B-tree index).
pub(super) fn chunked_layout(btree: u64, chunk: &[u64], element_size: u32) -> Vec<u8> {
    let mut out = vec![3, 2, (chunk.len() + 1) as u8];
    out.extend_from_slice(&btree.to_le_bytes());
    for dim in chunk {
        out.extend_from_slice(&(*dim as u32).to_le_bytes());
    }
    out.extend_from_slice(&element_size.to_le_bytes());
    out
}

/// Filter pipeline message body (version 2): shuffle then deflate, both
/// optional as the HDF5 library registers them.
pub(super) fn filter_pipeline(shuffle: Option<u32>, deflate: Option<u32>) -> Vec<u8> {
    let count = u8::from(shuffle.is_some()) + u8::from(deflate.is_some());
    let mut out = vec![2, count];
    for (id, value) in [(2u16, shuffle), (1u16, deflate)] {
        if let Some(value) = value {
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&1u16.to_le_bytes());
            out.extend_from_slice(&1u16.to_le_bytes());
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    out
}

/// A version 2 object header ("OHDR") with attribute creation order
/// tracked, so every message carries a creation order field.
pub(super) fn object_header(messages: &[Message]) -> Vec<u8> {
    let body: usize = messages.iter().map(|m| 6 + m.body.len()).sum();
    let (size_code, size_bytes) = if body < 256 {
        (0u8, 1usize)
    } else if body < 65_536 {
        (1, 2)
    } else {
        (2, 4)
    };
    let mut out = Vec::with_capacity(6 + size_bytes + body + 4);
    out.extend_from_slice(b"OHDR");
    out.push(2);
    // Bits 0-1: chunk size width; bit 2: attribute creation order tracked.
    out.push(size_code | 0x04);
    out.extend_from_slice(&(body as u64).to_le_bytes()[..size_bytes]);
    for message in messages {
        out.push(message.kind);
        out.extend_from_slice(&(message.body.len() as u16).to_le_bytes());
        out.push(message.flags);
        out.extend_from_slice(&message.creation_order.to_le_bytes());
        out.extend_from_slice(&message.body);
    }
    let checksum = lookup3(&out);
    out.extend_from_slice(&checksum.to_le_bytes());
    out
}

/// Encoded size of an object header holding messages with these body
/// lengths.
pub(super) fn object_header_size(body_lengths: impl Iterator<Item = usize>) -> usize {
    let body: usize = body_lengths.map(|len| 6 + len).sum();
    let size_bytes = if body < 256 {
        1
    } else if body < 65_536 {
        2
    } else {
        4
    };
    6 + size_bytes + body + 4
}

/// The version 2 superblock.
pub(super) fn superblock(end_of_file: u64, root: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(SUPERBLOCK_SIZE);
    out.extend_from_slice(b"\x89HDF\r\n\x1a\n");
    out.extend_from_slice(&[2, 8, 8, 0]);
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&UNDEF.to_le_bytes());
    out.extend_from_slice(&end_of_file.to_le_bytes());
    out.extend_from_slice(&root.to_le_bytes());
    let checksum = lookup3(&out);
    out.extend_from_slice(&checksum.to_le_bytes());
    out
}
