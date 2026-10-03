//! Local heaps (old-style group link names, section III.D) and global heap
//! collections (variable-length data, section III.E).

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::bytes::{Cursor, to_usize};
use crate::error::{Result, invalid};
use crate::space::Space;

/// A local heap's data segment.
pub(crate) struct LocalHeap<'a> {
    data: &'a [u8],
    address: u64,
}

impl<'a> LocalHeap<'a> {
    /// "HEAP", version (1) = 0, reserved (3), data segment size (L), free
    /// list head offset (L), data segment address (O).
    pub(crate) fn parse(space: &Space<'a>, address: u64) -> Result<Self> {
        space.expect_signature(address, b"HEAP")?;
        let mut cursor = space.cursor(address)?;
        cursor.skip(8)?;
        let size = cursor.length(space.length_size, "HDF5 local heap size")?;
        let _free_list = cursor.uint(space.length_size)?;
        let data_address = cursor.addr(space.offset_size)?;
        // The segment is resolved lazily so that a bad address surfaces
        // with the name that needs it.
        let data = space.slice(data_address, size).unwrap_or_default();
        Ok(Self {
            data,
            address: data_address,
        })
    }

    /// The NUL-terminated string at `offset` in the data segment.
    pub(crate) fn string(&self, space: &Space<'a>, offset: u64) -> Result<&'a [u8]> {
        let start = self
            .address
            .checked_add(offset)
            .ok_or_else(|| invalid(0, "HDF5 local heap name offset overflow"))?;
        let local = to_usize(offset, 0, "HDF5 local heap name offset")?;
        let tail = match self.data.get(local..) {
            Some(tail) if !tail.is_empty() => tail,
            _ => space.tail(start)?,
        };
        let end = tail
            .iter()
            .position(|byte| *byte == 0)
            .ok_or_else(|| invalid(0, "unterminated HDF5 local heap name"))?;
        Ok(&tail[..end])
    }
}

/// Parsed global heap collections: collection address to object index to
/// object bytes.
#[derive(Default)]
pub(crate) struct GlobalHeaps<'a> {
    collections: HashMap<u64, HashMap<u16, &'a [u8]>>,
}

impl<'a> GlobalHeaps<'a> {
    /// The bytes of object `index` in the collection at `collection`.
    pub(crate) fn object(
        &mut self,
        space: &Space<'a>,
        collection: u64,
        index: u32,
    ) -> Result<&'a [u8]> {
        let objects = match self.collections.entry(collection) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(parse_collection(space, collection)?),
        };
        u16::try_from(index)
            .ok()
            .and_then(|index| objects.get(&index))
            .copied()
            .ok_or_else(|| {
                invalid(
                    space.abs(collection).unwrap_or(0),
                    format!("global heap object {index} not found"),
                )
            })
    }
}

/// Round up to the global heap's 8-byte alignment (`H5HG_ALIGN`).
fn align8(len: usize) -> usize {
    len.div_ceil(8) * 8
}

/// "GCOL", version (1) = 1, reserved (3), collection size (L), padded to 8
/// bytes; then objects: index (u16), reference count (u16), reserved (4),
/// size (L), padded to 8 bytes, and the data padded to 8 bytes. Index 0 is
/// the free space that ends the collection. The header paddings
/// (`H5HG_SIZEOF_HDR`, `H5HG_SIZEOF_OBJHDR`) matter only for 4-byte lengths.
fn parse_collection<'a>(space: &Space<'a>, address: u64) -> Result<HashMap<u16, &'a [u8]>> {
    space.expect_signature(address, b"GCOL")?;
    let start = space.abs(address)?;
    let mut head = space.cursor(address)?;
    head.skip(8)?;
    let total = head.length(space.length_size, "HDF5 global heap size")?;
    let available = space.bytes.len() - start;
    let collection = &space.bytes[start..start + total.min(available)];
    let object_header = align8(8 + space.length_size);
    let mut cursor = Cursor::new(collection, start);
    cursor.skip(align8(8 + space.length_size))?;
    let mut objects = HashMap::new();
    while cursor.remaining() >= object_header {
        let index = cursor.u16()?;
        cursor.skip(6)?;
        let size = cursor.length(space.length_size, "HDF5 global heap object size")?;
        if index == 0 {
            break;
        }
        cursor.skip(object_header - 8 - space.length_size)?;
        let data = cursor.take(size)?;
        objects.insert(index, data);
        let padding = size.div_ceil(8) * 8 - size;
        if cursor.remaining() < padding {
            break;
        }
        cursor.skip(padding)?;
    }
    Ok(objects)
}
