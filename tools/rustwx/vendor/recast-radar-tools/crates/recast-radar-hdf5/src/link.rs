//! Link messages (section IV.A.2.g), link info (IV.A.2.c) and attribute
//! info (IV.A.2.v) messages.

use crate::bytes::Cursor;
use crate::error::{Result, invalid, limit};
use crate::limits::MAX_LINK_NAME_BYTES;

/// Where a link points.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LinkTarget {
    /// A hard link: the object header address.
    Hard(u64),
    /// A soft link: a path in this file.
    Soft(String),
    /// An external link: a file name and an object path in that file.
    External {
        /// Target file name.
        file: String,
        /// Object path inside the target file.
        path: String,
    },
    /// A user-defined link type, by its type code, with its raw data.
    UserDefined {
        /// Link type code (65-255).
        kind: u8,
        /// Link data.
        data: Vec<u8>,
    },
}

/// One link of a group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Link {
    /// Link name.
    pub name: String,
    /// What the link points at.
    pub target: LinkTarget,
    /// Creation order, when the group tracks it.
    pub creation_order: Option<u64>,
}

/// Link message: version (1) = 1, flags (u8), [link type (u8) when flags
/// bit 3], [creation order (u64) when bit 2], [name character set (u8) when
/// bit 4], name length (1/2/4/8 bytes per bits 0-1), name, link
/// information (hard: address; soft: u16 length + path; external: u16
/// length + flags byte + file name + NUL + object path + NUL; other: u16
/// length + data).
pub(crate) fn parse_link(body: &[u8], offset: usize, offset_size: usize) -> Result<Link> {
    let mut cursor = Cursor::new(body, offset);
    let version = cursor.u8()?;
    if version != 1 {
        return Err(invalid(
            offset,
            format!("link message version {version} unsupported"),
        ));
    }
    let flags = cursor.u8()?;
    let kind = if flags & 0x08 != 0 { cursor.u8()? } else { 0 };
    let creation_order = if flags & 0x04 != 0 {
        Some(cursor.u64()?)
    } else {
        None
    };
    if flags & 0x10 != 0 {
        cursor.skip(1)?;
    }
    let name_len = cursor.length(1 << (flags & 0x03), "link name length")?;
    if name_len > MAX_LINK_NAME_BYTES {
        return Err(limit(format!(
            "HDF5 link name of {name_len} bytes (limit {MAX_LINK_NAME_BYTES})"
        )));
    }
    let name = String::from_utf8_lossy(cursor.take(name_len)?).into_owned();
    let target = match kind {
        0 => LinkTarget::Hard(cursor.addr(offset_size)?),
        1 => {
            let len = usize::from(cursor.u16()?);
            LinkTarget::Soft(String::from_utf8_lossy(cursor.take(len)?).into_owned())
        }
        64 => {
            let len = usize::from(cursor.u16()?);
            let data = cursor.take(len)?;
            let mut inner = Cursor::new(data, offset);
            inner.skip(1)?;
            let file = String::from_utf8_lossy(inner.c_string()?).into_owned();
            let path = String::from_utf8_lossy(inner.c_string()?).into_owned();
            LinkTarget::External { file, path }
        }
        other if other >= 65 => {
            let len = usize::from(cursor.u16()?);
            LinkTarget::UserDefined {
                kind: other,
                data: cursor.take(len)?.to_vec(),
            }
        }
        other => return Err(invalid(offset, format!("link type {other} is reserved"))),
    };
    Ok(Link {
        name,
        target,
        creation_order,
    })
}

/// Dense storage locations of links or attributes.
pub(crate) struct DenseInfo {
    /// Creation order is tracked.
    pub(crate) tracked: bool,
    pub(crate) heap: u64,
    pub(crate) name_index: u64,
    /// Creation-order index (v2 B-tree), when indexed.
    pub(crate) order_index: Option<u64>,
}

/// Link info: version (0), flags, [maximum creation index (u64) when bit
/// 0], fractal heap address, name index v2 B-tree address, [creation order
/// index address when bit 1].
pub(crate) fn parse_link_info(body: &[u8], offset: usize, offset_size: usize) -> Result<DenseInfo> {
    parse_dense_info(body, offset, offset_size, 8, "link info")
}

/// Attribute info: as link info, with a u16 maximum creation index.
pub(crate) fn parse_attribute_info(
    body: &[u8],
    offset: usize,
    offset_size: usize,
) -> Result<DenseInfo> {
    parse_dense_info(body, offset, offset_size, 2, "attribute info")
}

fn parse_dense_info(
    body: &[u8],
    offset: usize,
    offset_size: usize,
    max_index_len: usize,
    what: &str,
) -> Result<DenseInfo> {
    let mut cursor = Cursor::new(body, offset);
    let version = cursor.u8()?;
    if version != 0 {
        return Err(invalid(
            offset,
            format!("{what} message version {version} unsupported"),
        ));
    }
    let flags = cursor.u8()?;
    if flags & 0x01 != 0 {
        cursor.skip(max_index_len)?;
    }
    let heap = cursor.addr(offset_size)?;
    let name_index = cursor.addr(offset_size)?;
    let order_index = if flags & 0x02 != 0 {
        Some(cursor.addr(offset_size)?)
    } else {
        None
    };
    Ok(DenseInfo {
        tracked: flags & 0x01 != 0,
        heap,
        name_index,
        order_index,
    })
}
