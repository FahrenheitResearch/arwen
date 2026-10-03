//! Version 2 B-trees (HDF5 File Format Specification, section III.A.2):
//! dense link and attribute name/creation-order indexes, huge fractal heap
//! objects, and the v2 B-tree chunk index.

use std::collections::HashSet;

use crate::bytes::{Cursor, UNDEFINED_ADDR, limit_enc_size};
use crate::checksum::lookup3;
use crate::error::{Result, invalid, limit};
use crate::limits::{MAX_BTREE_NODES, MAX_BTREE2_DEPTH, MAX_BTREE2_NODE_BYTES, MAX_GROUP_ENTRIES};
use crate::space::Space;

/// Signature, version, type and checksum around every node's content.
const PREFIX: usize = 4 + 1 + 1 + 4;

/// A parsed v2 B-tree header ("BTHD").
pub(crate) struct Header {
    pub(crate) record_type: u8,
    pub(crate) record_size: usize,
    depth: usize,
    root: u64,
    root_records: usize,
    /// Bytes of a child's record count in internal nodes.
    max_nrec_size: usize,
    /// Bytes of a child's total record count, per depth.
    cum_nrec_size: Vec<usize>,
    /// Most records a node holds, per depth (`H5B2__hdr_init`).
    node_max: Vec<usize>,
}

/// Header: signature, version (0), type (u8), node size (u32), record size
/// (u16), depth (u16), split and merge percents (u8 each), root node
/// address (O), records in the root (u16), total records (L), checksum.
pub(crate) fn header(space: &Space<'_>, address: u64) -> Result<Header> {
    space.expect_signature(address, b"BTHD")?;
    let start = space.abs(address)?;
    let mut cursor = space.cursor(address)?;
    cursor.skip(4)?;
    let version = cursor.u8()?;
    if version != 0 {
        return Err(invalid(
            start,
            format!("v2 B-tree version {version} unsupported"),
        ));
    }
    let record_type = cursor.u8()?;
    let node_size = cursor.u32()? as usize;
    let record_size = usize::from(cursor.u16()?);
    let depth = usize::from(cursor.u16()?);
    cursor.skip(2)?;
    let root = cursor.addr(space.offset_size)?;
    let root_records = usize::from(cursor.u16()?);
    let _total = cursor.uint(space.length_size)?;
    let covered = cursor.pos();
    let stored = cursor.u32()?;
    let checked = space.slice(address, covered)?;
    space.check("v2 B-tree header", start, stored, || lookup3(checked))?;
    if depth > MAX_BTREE2_DEPTH {
        return Err(limit(format!(
            "v2 B-tree depth {depth} (limit {MAX_BTREE2_DEPTH})"
        )));
    }
    if record_size == 0 || node_size <= PREFIX {
        return Err(invalid(start, "v2 B-tree node or record size is invalid"));
    }
    if node_size > MAX_BTREE2_NODE_BYTES {
        return Err(limit(format!(
            "v2 B-tree nodes of {node_size} bytes (limit {MAX_BTREE2_NODE_BYTES})"
        )));
    }
    // H5B2__hdr_init: leaf capacity sets the child record-count width;
    // each internal level's capacity follows from its pointer size.
    let leaf_max = ((node_size - PREFIX) / record_size) as u64;
    let max_nrec_size = limit_enc_size(leaf_max);
    let mut cum_max = vec![leaf_max];
    let mut cum_nrec_size = vec![0usize];
    let mut node_max = vec![leaf_max as usize];
    for level in 1..=depth {
        let pointer = space.offset_size
            + max_nrec_size
            + if level > 1 {
                cum_nrec_size[level - 1]
            } else {
                0
            };
        let available = node_size.saturating_sub(PREFIX + pointer);
        let max = (available / (record_size + pointer)) as u64;
        node_max.push(max as usize);
        let cumulative = max
            .saturating_add(1)
            .saturating_mul(cum_max[level - 1])
            .saturating_add(max);
        cum_max.push(cumulative);
        cum_nrec_size.push(limit_enc_size(cumulative));
    }
    Ok(Header {
        record_type,
        record_size,
        depth,
        root,
        root_records,
        max_nrec_size,
        cum_nrec_size,
        node_max,
    })
}

/// Visit every record of the tree at `address` in key order. The tree must
/// hold one of `record_types`, with records of at least `min_record_size`
/// bytes; both are checked before any record is visited.
pub(crate) fn for_each_record(
    space: &Space<'_>,
    address: u64,
    record_types: &[u8],
    min_record_size: usize,
    visit: &mut dyn FnMut(&[u8]) -> Result<()>,
) -> Result<Header> {
    let header = header(space, address)?;
    let start = space.abs(address)?;
    if !record_types.contains(&header.record_type) {
        return Err(invalid(
            start,
            format!(
                "v2 B-tree holds records of type {} (need {record_types:?})",
                header.record_type
            ),
        ));
    }
    if header.record_size < min_record_size {
        return Err(invalid(
            start,
            format!(
                "v2 B-tree records of type {} are {} bytes (need at least {min_record_size})",
                header.record_type, header.record_size
            ),
        ));
    }
    if header.root != UNDEFINED_ADDR && header.root_records > 0 {
        let mut walk = Walk {
            space,
            header: &header,
            visited: HashSet::new(),
            records: 0,
        };
        walk.node(header.root, header.root_records, header.depth, visit)?;
    }
    Ok(header)
}

struct Walk<'s, 'a> {
    space: &'s Space<'a>,
    header: &'s Header,
    visited: HashSet<u64>,
    records: usize,
}

impl Walk<'_, '_> {
    fn node(
        &mut self,
        address: u64,
        records: usize,
        depth: usize,
        visit: &mut dyn FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        let start = self.space.abs(address)?;
        if !self.visited.insert(address) {
            return Err(invalid(start, "cycle in v2 B-tree"));
        }
        if self.visited.len() > MAX_BTREE_NODES {
            return Err(limit(format!(
                "v2 B-tree has more than {MAX_BTREE_NODES} nodes (limit)"
            )));
        }
        self.records = self.records.saturating_add(records);
        if self.records > MAX_GROUP_ENTRIES {
            return Err(limit(format!(
                "v2 B-tree has more than {MAX_GROUP_ENTRIES} records (limit)"
            )));
        }
        let capacity = self.header.node_max.get(depth).copied().unwrap_or(0);
        if records > capacity {
            return Err(invalid(
                start,
                format!("v2 B-tree node claims {records} records (capacity {capacity})"),
            ));
        }
        let signature: &[u8; 4] = if depth == 0 { b"BTLF" } else { b"BTIN" };
        self.space.expect_signature(address, signature)?;
        let head = self.space.slice(address, 6)?;
        if head[4] != 0 || head[5] != self.header.record_type {
            return Err(invalid(start, "v2 B-tree node version or type mismatch"));
        }
        let record_bytes = records
            .checked_mul(self.header.record_size)
            .ok_or_else(|| invalid(start, "v2 B-tree node size overflow"))?;
        let pointer_size = if depth == 0 {
            0
        } else {
            self.space.offset_size
                + self.header.max_nrec_size
                + if depth > 1 {
                    self.header
                        .cum_nrec_size
                        .get(depth - 1)
                        .copied()
                        .unwrap_or(0)
                } else {
                    0
                }
        };
        let pointer_bytes = if depth == 0 {
            0
        } else {
            (records + 1)
                .checked_mul(pointer_size)
                .ok_or_else(|| invalid(start, "v2 B-tree node size overflow"))?
        };
        let content = record_bytes
            .checked_add(pointer_bytes)
            .and_then(|bytes| bytes.checked_add(6))
            .ok_or_else(|| invalid(start, "v2 B-tree node size overflow"))?;
        let node = self.space.slice(address, content + 4)?;
        let stored = u32::from_le_bytes(
            *node[content..]
                .first_chunk::<4>()
                .ok_or_else(|| invalid(start, "v2 B-tree node checksum"))?,
        );
        self.space.check("v2 B-tree node", start, stored, || {
            lookup3(&node[..content])
        })?;
        // `content` covers every record: the node slice holds them all.
        let record_area = &node[6..6 + record_bytes];
        let mut records_of = record_area.chunks_exact(self.header.record_size);
        if depth == 0 {
            for record in records_of {
                visit(record)?;
            }
            return Ok(());
        }
        let mut children = Vec::with_capacity(records + 1);
        let mut cursor = Cursor::new(&node[6 + record_bytes..content], start + 6 + record_bytes);
        for _ in 0..=records {
            let child = cursor.addr(self.space.offset_size)?;
            let child_records =
                cursor.length(self.header.max_nrec_size, "v2 B-tree child records")?;
            if depth > 1 {
                cursor.skip(
                    self.header
                        .cum_nrec_size
                        .get(depth - 1)
                        .copied()
                        .unwrap_or(0),
                )?;
            }
            children.push((child, child_records));
        }
        for (child, child_records) in children {
            self.node(child, child_records, depth - 1, visit)?;
            if let Some(record) = records_of.next() {
                visit(record)?;
            }
        }
        Ok(())
    }
}
