//! Chunked storage: splitting a dataset into chunks, the shuffle and
//! deflate filters, and the version 1 B-tree chunk index (node type 1).

use std::io::Write as _;

use flate2::Compression;
use flate2::write::ZlibEncoder;

use super::WriteError;
use super::encode::UNDEF;

/// `K` of chunk B-tree nodes: the library default (`HDF5_BTREE_CHUNK_IK_DEF`)
/// every reader assumes when the superblock does not store one. A node holds
/// at most `2K` children and is always allocated at its full size.
const CHUNK_BTREE_K: usize = 32;

/// One stored chunk: element offsets and its filtered bytes.
pub(super) struct StoredChunk {
    pub(super) offsets: Vec<u64>,
    pub(super) bytes: Vec<u8>,
}

/// Split `raw` (row-major elements of `element_size` bytes, `dims` extent)
/// into `chunk`-shaped chunks in row-major chunk order. Edge chunks are
/// stored whole, padded with `fill` (one element). Each chunk is shuffled
/// and deflated when asked.
pub(super) fn split(
    raw: &[u8],
    dims: &[u64],
    chunk: &[u64],
    element_size: usize,
    fill: &[u8],
    shuffle: bool,
    deflate: Option<u32>,
) -> Result<Vec<StoredChunk>, WriteError> {
    let rank = dims.len();
    let counts: Vec<u64> = dims
        .iter()
        .zip(chunk)
        .map(|(dim, chunk)| dim.div_ceil(*chunk))
        .collect();
    let total: u64 = counts.iter().product();
    let chunk_elements: usize = chunk
        .iter()
        .try_fold(1usize, |product, dim| {
            usize::try_from(*dim)
                .ok()
                .and_then(|dim| product.checked_mul(dim))
        })
        .ok_or_else(|| WriteError::TooLarge("chunk element count".to_owned()))?;
    let chunk_bytes = chunk_elements
        .checked_mul(element_size)
        .filter(|bytes| u32::try_from(*bytes).is_ok())
        .ok_or_else(|| WriteError::TooLarge("chunk larger than 4 GiB".to_owned()))?;
    // Element strides of the dataset, row-major.
    let mut strides = vec![1u64; rank];
    for axis in (0..rank.saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * dims[axis + 1];
    }
    let last = rank - 1;
    let mut chunks = Vec::with_capacity(usize::try_from(total).unwrap_or(0));
    let mut index = vec![0u64; rank];
    for _ in 0..total {
        let offsets: Vec<u64> = index.iter().zip(chunk).map(|(i, c)| i * c).collect();
        let mut buffer = Vec::with_capacity(chunk_bytes);
        // Walk the rows (last axis runs) of the chunk box.
        let rows: u64 = chunk[..last].iter().product();
        let mut row = vec![0u64; last];
        for _ in 0..rows {
            let inside = (0..last).all(|axis| offsets[axis] + row[axis] < dims[axis]);
            let run = if inside {
                (dims[last] - offsets[last]).min(chunk[last]) as usize
            } else {
                0
            };
            if run > 0 {
                let start: u64 = (0..last)
                    .map(|axis| (offsets[axis] + row[axis]) * strides[axis])
                    .sum::<u64>()
                    + offsets[last];
                let start = start as usize * element_size;
                buffer.extend_from_slice(&raw[start..start + run * element_size]);
            }
            for _ in run..chunk[last] as usize {
                buffer.extend_from_slice(fill);
            }
            // Next row, odometer over the leading axes of the chunk.
            for axis in (0..last).rev() {
                row[axis] += 1;
                if row[axis] < chunk[axis] {
                    break;
                }
                row[axis] = 0;
            }
        }
        let mut bytes = if shuffle && element_size > 1 {
            shuffle_bytes(&buffer, element_size)
        } else {
            buffer
        };
        if let Some(level) = deflate {
            bytes = deflate_bytes(&bytes, level)?;
        }
        if u32::try_from(bytes.len()).is_err() {
            return Err(WriteError::TooLarge(
                "stored chunk larger than 4 GiB".to_owned(),
            ));
        }
        chunks.push(StoredChunk { offsets, bytes });
        for axis in (0..rank).rev() {
            index[axis] += 1;
            if index[axis] < counts[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Ok(chunks)
}

/// The HDF5 shuffle filter: byte `b` of every element, for each `b` in turn.
fn shuffle_bytes(bytes: &[u8], element_size: usize) -> Vec<u8> {
    let elements = bytes.len() / element_size;
    let mut out = vec![0u8; bytes.len()];
    for (index, element) in bytes.chunks_exact(element_size).enumerate() {
        for (byte, value) in element.iter().enumerate() {
            out[byte * elements + index] = *value;
        }
    }
    // A trailing partial element (never produced here) stays in place.
    let tail = elements * element_size;
    out[tail..].copy_from_slice(&bytes[tail..]);
    out
}

/// zlib-wrapped deflate, as the HDF5 deflate filter (`compress2`) writes.
fn deflate_bytes(bytes: &[u8], level: u32) -> Result<Vec<u8>, WriteError> {
    let mut encoder = ZlibEncoder::new(
        Vec::with_capacity(bytes.len() / 2),
        Compression::new(level.min(9)),
    );
    encoder
        .write_all(bytes)
        .map_err(|err| WriteError::Compression(err.to_string()))?;
    encoder
        .finish()
        .map_err(|err| WriteError::Compression(err.to_string()))
}

/// One node of the chunk B-tree, in the order nodes are laid out.
pub(super) struct Node {
    pub(super) level: u8,
    /// Children: chunk indices (level 0) or node indices (level > 0).
    pub(super) children: Vec<usize>,
}

/// The chunk B-tree of `chunk_count` chunks: leaves of up to `2K` chunks,
/// parents of up to `2K` nodes, up to one root. Returns the nodes (root
/// last) or none for zero chunks.
pub(super) fn btree(chunk_count: usize) -> Vec<Node> {
    let fan = 2 * CHUNK_BTREE_K;
    let mut nodes = Vec::new();
    if chunk_count == 0 {
        return nodes;
    }
    let mut level_nodes: Vec<usize> = Vec::new();
    for start in (0..chunk_count).step_by(fan) {
        level_nodes.push(nodes.len());
        nodes.push(Node {
            level: 0,
            children: (start..(start + fan).min(chunk_count)).collect(),
        });
    }
    let mut level = 0u8;
    while level_nodes.len() > 1 {
        level += 1;
        let mut parents = Vec::new();
        for group in level_nodes.chunks(fan) {
            parents.push(nodes.len());
            nodes.push(Node {
                level,
                children: group.to_vec(),
            });
        }
        level_nodes = parents;
    }
    nodes
}

/// Bytes of one key: chunk size, filter mask, `rank + 1` offsets.
fn key_size(rank: usize) -> usize {
    8 + 8 * (rank + 1)
}

/// Allocated size of every chunk B-tree node of a `rank`-dimensional
/// dataset: header, `2K` children and `2K + 1` keys.
pub(super) fn node_size(rank: usize) -> usize {
    24 + 2 * CHUNK_BTREE_K * 8 + (2 * CHUNK_BTREE_K + 1) * key_size(rank)
}

fn push_key(out: &mut Vec<u8>, stored: u32, offsets: &[u64], element_size: u64) {
    out.extend_from_slice(&stored.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for offset in offsets {
        out.extend_from_slice(&offset.to_le_bytes());
    }
    out.extend_from_slice(&element_size.to_le_bytes());
}

/// Encode node `index` of `nodes`. `node_addresses` and `chunk_addresses`
/// are file addresses; keys follow the HDF5 library: a child's key is its
/// first chunk's offsets, the last key one chunk past the last chunk.
pub(super) fn encode_node(
    nodes: &[Node],
    index: usize,
    node_addresses: &[u64],
    chunks: &[StoredChunk],
    chunk_addresses: &[u64],
    chunk: &[u64],
    element_size: u64,
) -> Vec<u8> {
    let node = &nodes[index];
    let rank = chunk.len();
    let mut out = Vec::with_capacity(node_size(rank));
    out.extend_from_slice(b"TREE");
    out.push(1);
    out.push(node.level);
    out.extend_from_slice(&(node.children.len() as u16).to_le_bytes());
    // Siblings: the neighbouring nodes of the same level.
    let left = index
        .checked_sub(1)
        .filter(|left| nodes[*left].level == node.level)
        .map_or(UNDEF, |left| node_addresses[left]);
    let right = Some(index + 1)
        .filter(|right| nodes.get(*right).is_some_and(|n| n.level == node.level))
        .map_or(UNDEF, |right| node_addresses[right]);
    out.extend_from_slice(&left.to_le_bytes());
    out.extend_from_slice(&right.to_le_bytes());
    let first_chunk = |node: usize| -> usize {
        let mut at = node;
        while nodes[at].level > 0 {
            at = nodes[at].children[0];
        }
        nodes[at].children[0]
    };
    let last_chunk = |node: usize| -> usize {
        let mut at = node;
        while nodes[at].level > 0 {
            at = *nodes[at].children.last().unwrap_or(&at);
        }
        *nodes[at].children.last().unwrap_or(&0)
    };
    for child in &node.children {
        let (chunk_index, stored, address) = if node.level == 0 {
            (
                *child,
                chunks[*child].bytes.len() as u32,
                chunk_addresses[*child],
            )
        } else {
            let first = first_chunk(*child);
            (
                first,
                chunks[first].bytes.len() as u32,
                node_addresses[*child],
            )
        };
        push_key(&mut out, stored, &chunks[chunk_index].offsets, 0);
        out.extend_from_slice(&address.to_le_bytes());
    }
    let last = &chunks[last_chunk(index)].offsets;
    let past: Vec<u64> = last.iter().zip(chunk).map(|(o, c)| o + c).collect();
    push_key(&mut out, 0, &past, element_size);
    out.resize(node_size(rank), 0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_groups_bytes_by_position() {
        assert_eq!(
            shuffle_bytes(&[1, 2, 3, 4, 5, 6], 2),
            vec![1, 3, 5, 2, 4, 6]
        );
    }

    #[test]
    fn btree_fans_out_at_64() {
        assert_eq!(btree(0).len(), 0);
        assert_eq!(btree(64).len(), 1);
        let nodes = btree(65);
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[2].level, 1);
        assert_eq!(nodes[2].children, vec![0, 1]);
    }
}
