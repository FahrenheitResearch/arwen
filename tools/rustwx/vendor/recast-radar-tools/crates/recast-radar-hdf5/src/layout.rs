//! Data layout messages, versions 1-4 (section IV.A.2.i).

use crate::bytes::Cursor;
use crate::error::{Result, invalid, limit, unsupported};
use crate::limits::{MAX_DATASET_BYTES, MAX_DATASPACE_DIM, MAX_DATASPACE_RANK};

/// How a chunked dataset indexes its chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChunkIndexKind {
    /// Version 1 B-tree (layout messages before version 4).
    BTreeV1,
    /// One chunk covering the whole dataset.
    SingleChunk,
    /// No index: chunks stored back to back (fixed size, no filters).
    Implicit,
    /// Fixed array (fixed maximum dimensions).
    FixedArray,
    /// Extensible array (one unlimited dimension).
    ExtensibleArray,
    /// Version 2 B-tree (several unlimited dimensions).
    BTreeV2,
}

/// Storage layout of a dataset.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StorageLayout {
    /// Data stored inside the object header.
    Compact,
    /// One contiguous block.
    Contiguous,
    /// Fixed-size chunks.
    Chunked {
        /// Chunk dimensions (dataset rank).
        chunk_dims: Vec<usize>,
        /// The chunk index.
        index: ChunkIndexKind,
    },
    /// A virtual dataset (mapping of other datasets); not readable here.
    Virtual,
}

/// Chunk index location and parameters.
#[derive(Clone, Debug)]
pub(crate) enum ChunkIndex {
    BTreeV1 {
        address: u64,
    },
    SingleChunk {
        address: u64,
        filtered: Option<(usize, u32)>,
    },
    Implicit {
        address: u64,
    },
    FixedArray {
        address: u64,
    },
    ExtensibleArray {
        address: u64,
    },
    BTreeV2 {
        address: u64,
    },
}

impl ChunkIndex {
    pub(crate) fn kind(&self) -> ChunkIndexKind {
        match self {
            Self::BTreeV1 { .. } => ChunkIndexKind::BTreeV1,
            Self::SingleChunk { .. } => ChunkIndexKind::SingleChunk,
            Self::Implicit { .. } => ChunkIndexKind::Implicit,
            Self::FixedArray { .. } => ChunkIndexKind::FixedArray,
            Self::ExtensibleArray { .. } => ChunkIndexKind::ExtensibleArray,
            Self::BTreeV2 { .. } => ChunkIndexKind::BTreeV2,
        }
    }
}

/// A parsed layout message.
#[derive(Clone, Debug)]
pub(crate) enum Layout<'a> {
    Compact(&'a [u8]),
    Contiguous {
        address: u64,
        /// Stored size; `None` for version 1-2 messages (computed from the
        /// dataspace).
        size: Option<u64>,
    },
    Chunked {
        /// Chunk dimensions without the trailing element-size dimension.
        chunk_dims: Vec<usize>,
        index: ChunkIndex,
        /// Layout flags (version 4): bit 0 = partial edge chunks are not
        /// filtered.
        flags: u8,
    },
    Virtual,
}

impl Layout<'_> {
    pub(crate) fn describe(&self) -> StorageLayout {
        match self {
            Self::Compact(_) => StorageLayout::Compact,
            Self::Contiguous { .. } => StorageLayout::Contiguous,
            Self::Chunked {
                chunk_dims, index, ..
            } => StorageLayout::Chunked {
                chunk_dims: chunk_dims.clone(),
                index: index.kind(),
            },
            Self::Virtual => StorageLayout::Virtual,
        }
    }
}

fn chunk_dims_checked(mut dims: Vec<u64>, at: usize) -> Result<Vec<usize>> {
    if dims.len() < 2 || dims.len() > MAX_DATASPACE_RANK + 1 {
        return Err(invalid(at, "invalid HDF5 chunk dimensionality"));
    }
    // The trailing entry is the element size.
    dims.pop();
    dims.into_iter()
        .map(|dim| {
            if dim == 0 {
                Err(invalid(at, "invalid HDF5 chunk dimension"))
            } else if dim > MAX_DATASPACE_DIM {
                Err(limit(format!(
                    "HDF5 chunk dimension is {dim} (limit {MAX_DATASPACE_DIM})"
                )))
            } else {
                Ok(dim as usize)
            }
        })
        .collect()
}

pub(crate) fn parse(
    body: &[u8],
    offset: usize,
    offset_size: usize,
    length_size: usize,
) -> Result<Layout<'_>> {
    let mut cursor = Cursor::new(body, offset);
    let version = cursor.u8()?;
    match version {
        1 | 2 => parse_v1_v2(&mut cursor, offset, offset_size),
        3..=5 => {
            let class = cursor.u8()?;
            match class {
                0 => {
                    let size = usize::from(cursor.u16()?);
                    Ok(Layout::Compact(cursor.take(size)?))
                }
                1 => Ok(Layout::Contiguous {
                    address: cursor.addr(offset_size)?,
                    size: Some(cursor.uint(length_size)?),
                }),
                2 if version == 3 => {
                    let rank = usize::from(cursor.u8()?);
                    let address = cursor.addr(offset_size)?;
                    let mut dims = Vec::with_capacity(rank);
                    for _ in 0..rank {
                        dims.push(u64::from(cursor.u32()?));
                    }
                    Ok(Layout::Chunked {
                        chunk_dims: chunk_dims_checked(dims, offset)?,
                        index: ChunkIndex::BTreeV1 { address },
                        flags: 0,
                    })
                }
                2 => parse_v4_chunked(&mut cursor, offset, offset_size, length_size),
                3 if version >= 4 => Ok(Layout::Virtual),
                other => Err(invalid(
                    offset,
                    format!("data layout class {other} unsupported"),
                )),
            }
        }
        other => Err(invalid(
            offset,
            format!("data layout message version {other} unsupported"),
        )),
    }
}

/// Versions 1-2: version, dimensionality, class, 5 reserved bytes, [address
/// unless compact], dimensionality x u32 sizes (for chunked storage the
/// last is the element size), [compact size (u32) and data].
fn parse_v1_v2<'a>(
    cursor: &mut Cursor<'a>,
    offset: usize,
    offset_size: usize,
) -> Result<Layout<'a>> {
    let rank = usize::from(cursor.u8()?);
    let class = cursor.u8()?;
    cursor.skip(5)?;
    if rank > MAX_DATASPACE_RANK + 1 {
        return Err(invalid(offset, "invalid HDF5 layout dimensionality"));
    }
    let address = if class == 0 {
        None
    } else {
        Some(cursor.addr(offset_size)?)
    };
    let mut dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        dims.push(u64::from(cursor.u32()?));
    }
    match (class, address) {
        (0, _) => {
            let size = cursor.u32()? as usize;
            if size > MAX_DATASET_BYTES {
                return Err(limit(format!(
                    "HDF5 compact dataset is {size} bytes (limit {MAX_DATASET_BYTES})"
                )));
            }
            Ok(Layout::Compact(cursor.take(size)?))
        }
        (1, Some(address)) => Ok(Layout::Contiguous {
            address,
            size: None,
        }),
        (2, Some(address)) => Ok(Layout::Chunked {
            chunk_dims: chunk_dims_checked(dims, offset)?,
            index: ChunkIndex::BTreeV1 { address },
            flags: 0,
        }),
        (other, _) => Err(invalid(
            offset,
            format!("data layout class {other} unsupported"),
        )),
    }
}

/// Version 4 chunked: flags, dimensionality, dimension-size encoded length,
/// dimension sizes, chunk index type, index parameters, index address.
fn parse_v4_chunked<'a>(
    cursor: &mut Cursor<'a>,
    offset: usize,
    offset_size: usize,
    length_size: usize,
) -> Result<Layout<'a>> {
    let flags = cursor.u8()?;
    let rank = usize::from(cursor.u8()?);
    let encoded = usize::from(cursor.u8()?);
    if !(1..=8).contains(&encoded) {
        return Err(invalid(
            offset,
            format!("chunk dimension size of {encoded} bytes"),
        ));
    }
    if rank > MAX_DATASPACE_RANK + 1 {
        return Err(invalid(offset, "invalid HDF5 chunk dimensionality"));
    }
    let mut dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        dims.push(cursor.uint(encoded)?);
    }
    let index_type = cursor.u8()?;
    let index = match index_type {
        1 => {
            let filtered = if flags & 0x02 != 0 {
                let size = cursor.length(length_size, "single chunk size")?;
                Some((size, cursor.u32()?))
            } else {
                None
            };
            ChunkIndex::SingleChunk {
                address: cursor.addr(offset_size)?,
                filtered,
            }
        }
        2 => ChunkIndex::Implicit {
            address: cursor.addr(offset_size)?,
        },
        3 => {
            cursor.skip(1)?;
            ChunkIndex::FixedArray {
                address: cursor.addr(offset_size)?,
            }
        }
        4 => {
            cursor.skip(5)?;
            ChunkIndex::ExtensibleArray {
                address: cursor.addr(offset_size)?,
            }
        }
        5 => {
            cursor.skip(6)?;
            ChunkIndex::BTreeV2 {
                address: cursor.addr(offset_size)?,
            }
        }
        other => {
            return Err(unsupported(format!("chunk index type {other}")));
        }
    };
    Ok(Layout::Chunked {
        chunk_dims: chunk_dims_checked(dims, offset)?,
        index,
        flags,
    })
}
