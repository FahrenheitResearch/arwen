//! Version 1 B-trees (HDF5 File Format Specification, section III.A.1):
//! old-style group symbol tables (node type 0) and the chunk index of
//! layout messages before version 4 (node type 1).

use std::collections::BTreeSet;

use crate::bytes::Cursor;
use crate::error::{Result, invalid, limit};
use crate::limits::{MAX_BTREE_NODES, MAX_DATA_CHUNKS, MAX_GROUP_ENTRIES};
use crate::space::Space;

/// One symbol table entry: link name offset in the local heap and object
/// header address.
pub(crate) struct SymbolEntry {
    pub(crate) name_offset: u64,
    pub(crate) header: u64,
}

/// One chunk: file address, stored size, filter mask and element offsets
/// (one per dataset dimension).
#[derive(Clone, Debug)]
pub(crate) struct ChunkRecord {
    pub(crate) address: u64,
    pub(crate) size: usize,
    pub(crate) filter_mask: u32,
    pub(crate) offsets: Vec<u64>,
}

/// Node header: "TREE", node type (u8), level (u8), entries used (u16),
/// left and right sibling addresses; keys and child pointers alternate.
fn node_header<'a>(
    space: &Space<'a>,
    address: u64,
    node_type: u8,
    visited: &mut BTreeSet<u64>,
) -> Result<(u8, usize, Cursor<'a>)> {
    let start = space.abs(address)?;
    if !visited.insert(address) {
        let what = if node_type == 0 { "group" } else { "chunk" };
        return Err(invalid(start, format!("cycle in HDF5 {what} B-tree")));
    }
    if visited.len() > MAX_BTREE_NODES {
        return Err(limit(format!(
            "HDF5 B-tree has more than {MAX_BTREE_NODES} nodes (limit)"
        )));
    }
    space.expect_signature(address, b"TREE")?;
    let mut cursor = space.cursor(address)?;
    cursor.skip(4)?;
    if cursor.u8()? != node_type {
        let what = if node_type == 0 { "group" } else { "chunk" };
        return Err(invalid(start, format!("expected {what} B-tree node")));
    }
    let level = cursor.u8()?;
    let entries = usize::from(cursor.u16()?);
    cursor.skip(2 * space.offset_size)?;
    Ok((level, entries, cursor))
}

/// Every entry of an old-style group, in B-tree (name) order.
pub(crate) fn group_entries(space: &Space<'_>, root: u64) -> Result<Vec<SymbolEntry>> {
    let mut out = Vec::new();
    let mut visited = BTreeSet::new();
    group_node(space, root, &mut out, &mut visited)?;
    Ok(out)
}

fn group_node(
    space: &Space<'_>,
    address: u64,
    out: &mut Vec<SymbolEntry>,
    visited: &mut BTreeSet<u64>,
) -> Result<()> {
    let (level, entries, mut cursor) = node_header(space, address, 0, visited)?;
    for _ in 0..entries {
        cursor.skip(space.length_size)?; // key: heap offset of the first name
        let child = cursor.addr(space.offset_size)?;
        if level == 0 {
            symbol_node(space, child, out)?;
        } else {
            group_node(space, child, out, visited)?;
        }
    }
    Ok(())
}

/// "SNOD", version (1), reserved (1), symbol count (u16), then entries of
/// link name offset (O), object header address (O), cache type (u32),
/// reserved (u32) and a 16-byte scratch pad.
fn symbol_node(space: &Space<'_>, address: u64, out: &mut Vec<SymbolEntry>) -> Result<()> {
    space.expect_signature(address, b"SNOD")?;
    let mut cursor = space.cursor(address)?;
    cursor.skip(6)?;
    let count = usize::from(cursor.u16()?);
    for index in 0..count {
        let name_offset = cursor.uint(space.offset_size)?;
        let header = cursor.addr(space.offset_size)?;
        // Cache type, reserved word and scratch pad are unused; the last
        // entry's may be cut off.
        if index + 1 < count {
            cursor.skip(24)?;
        }
        if out.len() >= MAX_GROUP_ENTRIES {
            return Err(limit(format!(
                "HDF5 group has more than {MAX_GROUP_ENTRIES} entries (limit)"
            )));
        }
        out.push(SymbolEntry {
            name_offset,
            header,
        });
    }
    Ok(())
}

/// Every chunk of a v1 chunk B-tree whose keys hold `rank` dataset
/// dimensions (plus the element-size dimension).
pub(crate) fn chunks(space: &Space<'_>, root: u64, rank: usize) -> Result<Vec<ChunkRecord>> {
    let mut out = Vec::new();
    let mut visited = BTreeSet::new();
    chunk_node(space, root, rank, &mut out, &mut visited)?;
    Ok(out)
}

/// Chunk keys: stored chunk size (u32), filter mask (u32), and `rank + 1`
/// element offsets (u64 each); the last offset is always zero.
fn chunk_node(
    space: &Space<'_>,
    address: u64,
    rank: usize,
    out: &mut Vec<ChunkRecord>,
    visited: &mut BTreeSet<u64>,
) -> Result<()> {
    let (level, entries, mut cursor) = node_header(space, address, 1, visited)?;
    for _ in 0..entries {
        let size = cursor.u32()? as usize;
        let filter_mask = cursor.u32()?;
        let mut offsets = Vec::with_capacity(rank);
        for _ in 0..rank {
            offsets.push(cursor.u64()?);
        }
        cursor.skip(8)?;
        let child = cursor.addr(space.offset_size)?;
        if level == 0 {
            if out.len() >= MAX_DATA_CHUNKS {
                return Err(limit(format!(
                    "HDF5 dataset has more than {MAX_DATA_CHUNKS} chunks (limit)"
                )));
            }
            out.push(ChunkRecord {
                address: child,
                size,
                filter_mask,
                offsets,
            });
        } else {
            chunk_node(space, child, rank, out, visited)?;
        }
    }
    Ok(())
}
