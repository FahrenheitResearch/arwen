//! Object headers, versions 1 and 2 (HDF5 File Format Specification,
//! section IV.A), with their continuation blocks.

use std::collections::BTreeSet;

use crate::bytes::Cursor;
use crate::checksum::lookup3;
use crate::error::{Result, invalid, limit, truncated};
use crate::limits::{MAX_HEADER_BLOCKS, MAX_OBJECT_MESSAGE_BYTES, MAX_OBJECT_MESSAGES};
use crate::space::Space;

/// Version 2 object header signature.
const OHDR: &[u8; 4] = b"OHDR";
/// Version 2 continuation block signature.
const OCHK: &[u8; 4] = b"OCHK";

pub(crate) const MSG_NIL: u16 = 0x0000;
pub(crate) const MSG_DATASPACE: u16 = 0x0001;
pub(crate) const MSG_LINK_INFO: u16 = 0x0002;
pub(crate) const MSG_DATATYPE: u16 = 0x0003;
pub(crate) const MSG_FILL_OLD: u16 = 0x0004;
pub(crate) const MSG_FILL: u16 = 0x0005;
pub(crate) const MSG_LINK: u16 = 0x0006;
pub(crate) const MSG_EXTERNAL_FILES: u16 = 0x0007;
pub(crate) const MSG_LAYOUT: u16 = 0x0008;
pub(crate) const MSG_GROUP_INFO: u16 = 0x000A;
pub(crate) const MSG_FILTERS: u16 = 0x000B;
pub(crate) const MSG_ATTRIBUTE: u16 = 0x000C;
pub(crate) const MSG_CONTINUATION: u16 = 0x0010;
pub(crate) const MSG_SYMBOL_TABLE: u16 = 0x0011;
pub(crate) const MSG_ATTRIBUTE_INFO: u16 = 0x0015;

/// Message flag bit 1: the body is a shared-message reference.
pub(crate) const FLAG_SHARED: u8 = 0x02;

/// One header message; `body` borrows the file bytes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Message<'a> {
    pub(crate) kind: u16,
    pub(crate) flags: u8,
    /// Creation order (version 2 headers that track it).
    pub(crate) creation_order: Option<u16>,
    pub(crate) body: &'a [u8],
    /// Absolute file offset of `body`.
    pub(crate) offset: usize,
}

/// A parsed object header: its messages in storage order.
#[derive(Clone, Debug)]
pub(crate) struct ObjectHeader<'a> {
    pub(crate) version: u8,
    pub(crate) messages: Vec<Message<'a>>,
}

impl<'a> ObjectHeader<'a> {
    /// Messages of one type, in storage order.
    pub(crate) fn of_kind(&self, kind: u16) -> impl Iterator<Item = &Message<'a>> {
        self.messages
            .iter()
            .filter(move |message| message.kind == kind)
    }

    /// The first message of one type.
    pub(crate) fn first(&self, kind: u16) -> Option<&Message<'a>> {
        self.of_kind(kind).next()
    }
}

/// Parse the object header at a relative address.
pub(crate) fn parse<'a>(space: &Space<'a>, address: u64) -> Result<ObjectHeader<'a>> {
    if space.has_signature(address, OHDR) {
        parse_v2(space, address)
    } else {
        parse_v1(space, address)
    }
}

fn continuation(space: &Space<'_>, body: &[u8], at: usize) -> Result<(usize, usize)> {
    let mut cursor = Cursor::new(body, at);
    let offset = cursor.addr(space.offset_size)?;
    let length = cursor.length(space.length_size, "HDF5 continuation length")?;
    if length > MAX_OBJECT_MESSAGE_BYTES {
        return Err(limit(format!(
            "HDF5 continuation block is {length} bytes (limit {MAX_OBJECT_MESSAGE_BYTES})"
        )));
    }
    Ok((space.abs(offset)?, length))
}

/// Version 1: version (1) = 1, reserved (1), total message count (u16),
/// reference count (u32), size of the first message block (u32), 4 bytes of
/// alignment padding; then messages of type (u16), size (u16), flags (u8),
/// 3 reserved bytes and an 8-byte aligned body.
fn parse_v1<'a>(space: &Space<'a>, address: u64) -> Result<ObjectHeader<'a>> {
    let head = space.slice(address, 16)?;
    let start = space.abs(address)?;
    if head[0] != 1 {
        return Err(invalid(
            start,
            format!("object header version {} is unsupported", head[0]),
        ));
    }
    let total_messages = usize::from(u16::from_le_bytes([head[2], head[3]]));
    if total_messages > MAX_OBJECT_MESSAGES {
        return Err(limit(format!(
            "HDF5 object header declares {total_messages} messages (limit {MAX_OBJECT_MESSAGES})"
        )));
    }
    let block_size = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
    if block_size > MAX_OBJECT_MESSAGE_BYTES {
        return Err(limit(format!(
            "HDF5 object-header message block is {block_size} bytes (limit {MAX_OBJECT_MESSAGE_BYTES})"
        )));
    }
    let first_block = start + 16;
    let mut blocks = vec![(first_block, block_size)];
    let mut scheduled = BTreeSet::from([first_block]);
    let mut messages = Vec::with_capacity(total_messages);
    let mut message_bytes = 0usize;
    let mut parsed = 0usize;
    let mut index = 0;
    while index < blocks.len() && parsed < total_messages {
        let (block_start, block_len) = blocks[index];
        index += 1;
        let end = block_start
            .checked_add(block_len)
            .ok_or_else(|| invalid(block_start, "HDF5 object-header block overflow"))?;
        let block = space
            .bytes
            .get(block_start..end)
            .ok_or_else(|| truncated(block_start, block_len, space.bytes.len()))?;
        let mut cursor = Cursor::new(block, block_start);
        while cursor.remaining() >= 8 && parsed < total_messages {
            let kind = cursor.u16()?;
            let size = usize::from(cursor.u16()?);
            let flags = cursor.u8()?;
            cursor.skip(3)?;
            let offset = block_start + cursor.pos();
            let body = cursor.take(size)?;
            parsed += 1;
            if kind == MSG_CONTINUATION {
                let (next, length) = continuation(space, body, offset)?;
                if blocks.len() >= MAX_HEADER_BLOCKS {
                    return Err(limit(format!(
                        "HDF5 object header has more than {MAX_HEADER_BLOCKS} continuation blocks (limit)"
                    )));
                }
                if !scheduled.insert(next) {
                    return Err(invalid(next, "cycle in HDF5 object-header continuations"));
                }
                blocks.push((next, length));
                continue;
            }
            message_bytes = message_bytes
                .checked_add(size)
                .ok_or_else(|| invalid(offset, "HDF5 object-header message size overflow"))?;
            if message_bytes > MAX_OBJECT_MESSAGE_BYTES {
                return Err(limit(format!(
                    "HDF5 object header holds more than {MAX_OBJECT_MESSAGE_BYTES} bytes of messages (limit)"
                )));
            }
            if kind != MSG_NIL {
                messages.push(Message {
                    kind,
                    flags,
                    creation_order: None,
                    body,
                    offset,
                });
            }
        }
    }
    Ok(ObjectHeader {
        version: 1,
        messages,
    })
}

/// Version 2 ("OHDR"): signature, version (1) = 2, flags (1),
/// [access/modification/change/birth times, 4 x u32, when flags bit 5],
/// [maximum compact / minimum dense attribute counts, 2 x u16, when flags
/// bit 4], size of chunk 0 (1/2/4/8 bytes per flags bits 0-1), messages,
/// lookup3 checksum of the chunk from the signature on. Messages are type
/// (u8), size (u16), flags (u8), [creation order (u16) when flags bit 2],
/// body, with no alignment; a gap smaller than a message header may precede
/// the checksum. Continuation blocks ("OCHK") hold signature, messages and a
/// checksum; their stored length includes both.
fn parse_v2<'a>(space: &Space<'a>, address: u64) -> Result<ObjectHeader<'a>> {
    let start = space.abs(address)?;
    let mut cursor = space.cursor(address)?;
    cursor.skip(4)?;
    let version = cursor.u8()?;
    if version != 2 {
        return Err(invalid(
            start,
            format!("OHDR object header version {version} unsupported (need 2)"),
        ));
    }
    let flags = cursor.u8()?;
    if flags & 0x20 != 0 {
        cursor.skip(16)?;
    }
    if flags & 0x10 != 0 {
        cursor.skip(4)?;
    }
    let size_width = 1usize << (flags & 0x03);
    let chunk0_size = cursor.length(size_width, "HDF5 v2 chunk size")?;
    if chunk0_size > MAX_OBJECT_MESSAGE_BYTES {
        return Err(limit(format!(
            "HDF5 v2 header message block is {chunk0_size} bytes (limit {MAX_OBJECT_MESSAGE_BYTES})"
        )));
    }
    let tracks_order = flags & 0x04 != 0;
    let message_header = if tracks_order { 6 } else { 4 };
    // (message region start, length, checksummed chunk start)
    let mut blocks = vec![(start + cursor.pos(), chunk0_size, start)];
    let mut scheduled = BTreeSet::from([start]);
    let mut messages = Vec::new();
    let mut message_bytes = 0usize;
    let mut index = 0;
    while index < blocks.len() {
        let (region, len, chunk_start) = blocks[index];
        index += 1;
        let end = region
            .checked_add(len)
            .ok_or_else(|| invalid(region, "HDF5 v2 message block overflow"))?;
        let chunk = space
            .bytes
            .get(chunk_start..end)
            .ok_or_else(|| truncated(chunk_start, end.saturating_sub(chunk_start), 0))?;
        let stored = u32::from_le_bytes(
            space
                .bytes
                .get(end..)
                .and_then(|tail| tail.first_chunk::<4>())
                .copied()
                .ok_or_else(|| truncated(end, 4, space.bytes.len().saturating_sub(end)))?,
        );
        space.check("HDF5 v2 object header", chunk_start, stored, || {
            lookup3(chunk)
        })?;
        let region_bytes = &chunk[region - chunk_start..];
        let mut cursor = Cursor::new(region_bytes, region);
        while cursor.remaining() >= message_header {
            let kind = u16::from(cursor.u8()?);
            let size = usize::from(cursor.u16()?);
            let message_flags = cursor.u8()?;
            let creation_order = if tracks_order {
                Some(cursor.u16()?)
            } else {
                None
            };
            let offset = region + cursor.pos();
            let body = cursor.take(size)?;
            if kind == MSG_CONTINUATION {
                let (next, length) = continuation(space, body, offset)?;
                if length < 8 {
                    return Err(invalid(offset, "HDF5 v2 continuation block too short"));
                }
                if blocks.len() >= MAX_HEADER_BLOCKS {
                    return Err(limit(format!(
                        "HDF5 v2 header has more than {MAX_HEADER_BLOCKS} continuation blocks (limit)"
                    )));
                }
                if space.bytes.get(next..next.saturating_add(4)) != Some(&OCHK[..]) {
                    return Err(invalid(next, "expected OCHK signature"));
                }
                if !scheduled.insert(next) {
                    return Err(invalid(next, "cycle in HDF5 v2 header continuations"));
                }
                blocks.push((next + 4, length - 8, next));
                continue;
            }
            if messages.len() >= MAX_OBJECT_MESSAGES {
                return Err(limit(format!(
                    "HDF5 v2 object header has more than {MAX_OBJECT_MESSAGES} messages (limit)"
                )));
            }
            message_bytes = message_bytes
                .checked_add(size)
                .ok_or_else(|| invalid(offset, "HDF5 v2 message byte count overflow"))?;
            if message_bytes > MAX_OBJECT_MESSAGE_BYTES {
                return Err(limit(format!(
                    "HDF5 v2 object header holds more than {MAX_OBJECT_MESSAGE_BYTES} bytes of messages (limit)"
                )));
            }
            if kind != MSG_NIL {
                messages.push(Message {
                    kind,
                    flags: message_flags,
                    creation_order,
                    body,
                    offset,
                });
            }
        }
    }
    Ok(ObjectHeader {
        version: 2,
        messages,
    })
}
