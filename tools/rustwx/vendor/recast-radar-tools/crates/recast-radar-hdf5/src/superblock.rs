//! Superblock versions 0-3 (HDF5 File Format Specification, section II).

use crate::bytes::Cursor;
use crate::checksum::lookup3;
use crate::error::{Error, Result, invalid, truncated};
use crate::space::check_checksum;

/// The 8-byte HDF5 format signature.
pub(crate) const SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', b'\r', b'\n', 0x1a, b'\n'];

/// A parsed superblock.
#[derive(Clone, Debug)]
pub(crate) struct Superblock {
    pub(crate) version: u8,
    /// File offset of the signature; every address in the file is relative
    /// to it (a user block may precede it).
    pub(crate) base: usize,
    pub(crate) offset_size: usize,
    pub(crate) length_size: usize,
    /// Object header address of the root group.
    pub(crate) root: u64,
}

/// Longest prefix searched for a signature at an offset section II does not
/// allow: a WMO bulletin heading (`IRVX40 CWAO 012240`) or another short
/// text header that a distributor puts before the file.
pub(crate) const MAX_PREFIX: usize = 64 * 1024;

/// Offset of the HDF5 signature: 0, 512, 1024, 2048, ... (section II); else
/// the first offset within [`MAX_PREFIX`] bytes where it starts, for files
/// some distributor has put a header in front of (ECCC volume scans carry a
/// text heading). Every address is relative to the signature either way.
pub(crate) fn signature_offset(bytes: &[u8]) -> Option<usize> {
    standard_signature_offset(bytes).or_else(|| {
        let end = bytes.len().min(MAX_PREFIX + SIGNATURE.len());
        bytes[..end]
            .windows(SIGNATURE.len())
            .position(|window| window == SIGNATURE)
    })
}

fn standard_signature_offset(bytes: &[u8]) -> Option<usize> {
    let mut at = 0usize;
    loop {
        if bytes.get(at..at.checked_add(SIGNATURE.len())?) == Some(&SIGNATURE[..]) {
            return Some(at);
        }
        at = if at == 0 { 512 } else { at.checked_mul(2)? };
        if at.saturating_add(SIGNATURE.len()) > bytes.len() {
            return None;
        }
    }
}

/// Find the superblock (at 0, 512, 1024, 2048, ... per section II, or after
/// a short header; see [`signature_offset`]) and parse it; `verify` checks
/// the version 2/3 checksum.
pub(crate) fn find(bytes: &[u8], verify: bool) -> Result<Superblock> {
    match signature_offset(bytes) {
        Some(at) => parse(bytes, at, verify),
        None => Err(invalid(0, "missing HDF5 superblock signature")),
    }
}

fn parse(bytes: &[u8], base: usize, verify: bool) -> Result<Superblock> {
    let tail = &bytes[base..];
    let version = *tail.get(8).ok_or_else(|| truncated(base + 8, 1, 0))?;
    match version {
        0 | 1 => parse_v0_v1(tail, base, version),
        2 | 3 => parse_v2_v3(tail, base, version, verify),
        other => Err(Error::Unsupported(format!(
            "HDF5 superblock version {other}"
        ))),
    }
}

fn check_sizes(offset_size: usize, length_size: usize, at: usize) -> Result<()> {
    if !matches!(offset_size, 2 | 4 | 8) || !matches!(length_size, 2 | 4 | 8) {
        return Err(invalid(
            at,
            format!("unsupported HDF5 offset/length sizes {offset_size}/{length_size}"),
        ));
    }
    Ok(())
}

/// Version 0/1: signature, versions (free space, root symbol table entry,
/// reserved, shared header), offset and length sizes, reserved, group K
/// values (2 x u16), consistency flags (u32), [v1: indexed storage K (u16) +
/// reserved (u16)], base / free-space / end-of-file / driver addresses, then
/// the root group symbol table entry (link name offset, object header
/// address, cache type, reserved, scratch pad).
fn parse_v0_v1(tail: &[u8], base: usize, version: u8) -> Result<Superblock> {
    let mut cursor = Cursor::new(tail, base);
    cursor.skip(13)?;
    let offset_size = usize::from(cursor.u8()?);
    let length_size = usize::from(cursor.u8()?);
    check_sizes(offset_size, length_size, base + 13)?;
    cursor.skip(1)?;
    let _group_leaf_k = cursor.u16()?;
    let _group_internal_k = cursor.u16()?;
    let _consistency_flags = cursor.u32()?;
    if version == 1 {
        cursor.skip(4)?;
    }
    let _base_address = cursor.addr(offset_size)?;
    let _free_space = cursor.addr(offset_size)?;
    let _end_of_file = cursor.addr(offset_size)?;
    let _driver_info = cursor.addr(offset_size)?;
    let _link_name_offset = cursor.uint(offset_size)?;
    let root = cursor.addr(offset_size)?;
    Ok(Superblock {
        version,
        base,
        offset_size,
        length_size,
        root,
    })
}

/// Version 2/3: signature, version, offset size, length size, consistency
/// flags (u8), base / extension / end-of-file / root object header
/// addresses, lookup3 checksum over everything before it.
fn parse_v2_v3(tail: &[u8], base: usize, version: u8, verify: bool) -> Result<Superblock> {
    let mut cursor = Cursor::new(tail, base);
    cursor.skip(9)?;
    let offset_size = usize::from(cursor.u8()?);
    let length_size = usize::from(cursor.u8()?);
    check_sizes(offset_size, length_size, base + 9)?;
    let _consistency_flags = cursor.u8()?;
    let _base_address = cursor.addr(offset_size)?;
    let _extension = cursor.addr(offset_size)?;
    let _end_of_file = cursor.addr(offset_size)?;
    let root = cursor.addr(offset_size)?;
    let covered = cursor.pos();
    let stored = cursor.u32()?;
    check_checksum(verify, "superblock", base, stored, || {
        lookup3(&tail[..covered])
    })?;
    Ok(Superblock {
        version,
        base,
        offset_size,
        length_size,
        root,
    })
}
