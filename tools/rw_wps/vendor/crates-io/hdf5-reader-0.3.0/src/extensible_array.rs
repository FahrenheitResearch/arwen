//! HDF5 Extensible Array (EA) chunk index.
//!
//! This is the default chunk index for datasets with one unlimited dimension
//! and `libver='latest'`. It uses a three-level hierarchy:
//!
//! - `EAHD` — Extensible Array Header
//! - `EAIB` — Extensible Array Index Block
//! - `EADB` — Extensible Array Data Block
//! - `EASB` — Extensible Array Secondary Block

use crate::checksum::jenkins_lookup3;
use crate::chunk_index::ChunkEntry;
use crate::error::{Error, Result};
use crate::io::Cursor;
use crate::storage::Storage;

const EAHD_SIGNATURE: [u8; 4] = *b"EAHD";
const EAIB_SIGNATURE: [u8; 4] = *b"EAIB";
const EADB_SIGNATURE: [u8; 4] = *b"EADB";
const EASB_SIGNATURE: [u8; 4] = *b"EASB";

/// Parsed Extensible Array Header.
#[derive(Debug)]
struct EaHeader {
    client_id: u8,
    element_size: u8,
    _max_nelmts_bits: u8,
    idx_blk_elmts: u8,
    data_blk_min_elmts: u8,
    sec_blk_min_data_ptrs: u8,
    max_dblk_page_nelmts_bits: u8,
    _nelmts: u64,
    index_block_address: u64,
}

/// Parse the Extensible Array Header.
///
/// On-disk layout (from H5EA_HEADER_SIZE):
/// sig(4) + ver(1) + client_id(1) +
/// element_size(1) + max_nelmts_bits(1) + idx_blk_elmts(1) +
/// data_blk_min_elmts(1) + sec_blk_min_data_ptrs(1) + max_dblk_page_nelmts_bits(1) +
/// 6 statistics fields (each length_size) +
/// index_block_address(offset_size) + checksum(4)
fn parse_header(data: &[u8], address: u64, offset_size: u8, length_size: u8) -> Result<EaHeader> {
    let mut cursor = Cursor::new(data);
    cursor.set_position(address);

    let sig = cursor.read_bytes(4)?;
    if sig != EAHD_SIGNATURE {
        return Err(Error::InvalidExtensibleArraySignature {
            context: "header signature mismatch",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::Other(format!(
            "unsupported extensible array header version {}",
            version
        )));
    }

    let client_id = cursor.read_u8()?;
    let element_size = cursor.read_u8()?;
    let max_nelmts_bits = cursor.read_u8()?;
    let idx_blk_elmts = cursor.read_u8()?;
    let data_blk_min_elmts = cursor.read_u8()?;
    let sec_blk_min_data_ptrs = cursor.read_u8()?;
    let max_dblk_page_nelmts_bits = cursor.read_u8()?;

    // Statistics (6 fields, each length_size bytes)
    let _nsuper_blks = cursor.read_length(length_size)?;
    let _super_blk_size = cursor.read_length(length_size)?;
    let _ndata_blks = cursor.read_length(length_size)?;
    let _data_blk_size = cursor.read_length(length_size)?;
    let _max_idx_set = cursor.read_length(length_size)?;
    let nelmts = cursor.read_length(length_size)?;

    let index_block_address = cursor.read_offset(offset_size)?;

    // Checksum
    let header_end = cursor.position();
    let header_bytes = &data[address as usize..header_end as usize];
    let stored_checksum = cursor.read_u32_le()?;
    let computed = jenkins_lookup3(header_bytes);
    if stored_checksum != computed {
        return Err(Error::ChecksumMismatch {
            expected: stored_checksum,
            actual: computed,
        });
    }

    Ok(EaHeader {
        client_id,
        element_size,
        _max_nelmts_bits: max_nelmts_bits,
        idx_blk_elmts,
        data_blk_min_elmts,
        sec_blk_min_data_ptrs,
        max_dblk_page_nelmts_bits,
        _nelmts: nelmts,
        index_block_address,
    })
}

fn parse_header_storage(
    storage: &dyn Storage,
    address: u64,
    offset_size: u8,
    length_size: u8,
) -> Result<EaHeader> {
    let header_len = 4
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
        + 6 * usize::from(length_size)
        + usize::from(offset_size)
        + 4;
    let bytes = storage.read_range(address, header_len)?;
    parse_header(bytes.as_ref(), 0, offset_size, length_size)
}

/// Compute the super block layout.
///
/// Returns a vec of (elements_per_data_block, num_data_blocks) for each super block.
/// Stops generating entries once cumulative capacity exceeds `nelmts`.
fn compute_super_block_layout(header: &EaHeader) -> Vec<(u64, u64)> {
    let mut layout = Vec::new();
    let dblk_min = header.data_blk_min_elmts as u64;
    let sblk_min = header.sec_blk_min_data_ptrs as u64;
    let nelmts = header._nelmts;
    let mut cumulative = header.idx_blk_elmts as u64;

    for sb_idx in 0u32..64 {
        if cumulative >= nelmts {
            break;
        }
        let elmts_per_dblk = dblk_min * (1u64 << (sb_idx / 2));
        let num_dblks = sblk_min * (1u64 << (sb_idx.div_ceil(2)));
        layout.push((elmts_per_dblk, num_dblks));
        cumulative += elmts_per_dblk * num_dblks;
    }

    layout
}

/// A single raw entry.
struct EaRawEntry {
    address: u64,
    chunk_size: u64,
    filter_mask: u32,
}

/// Read `count` entries from the cursor.
fn read_entries(
    cursor: &mut Cursor<'_>,
    count: usize,
    is_filtered: bool,
    offset_size: u8,
    entry_size: u8,
) -> Result<Vec<EaRawEntry>> {
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let address = cursor.read_offset(offset_size)?;
        let (chunk_size, filter_mask) = if is_filtered {
            let chunk_size_len = entry_size
                .checked_sub(offset_size)
                .and_then(|remaining| remaining.checked_sub(4))
                .ok_or_else(|| Error::InvalidData("invalid extensible array entry size".into()))?;
            let cs = cursor.read_length(chunk_size_len)?;
            let fm = cursor.read_u32_le()?;
            (cs, fm)
        } else {
            (0, 0)
        };
        entries.push(EaRawEntry {
            address,
            chunk_size,
            filter_mask,
        });
    }
    Ok(entries)
}

/// Parse a data block and return its entries.
///
/// `sizeof_nelmts` is `ceil(max_nelmts_bits / 8)` — used for the block_off field.
#[allow(clippy::too_many_arguments)]
fn parse_data_block(
    data: &[u8],
    address: u64,
    num_entries: usize,
    is_filtered: bool,
    max_page_bits: u8,
    offset_size: u8,
    entry_size: u8,
    sizeof_nelmts: usize,
) -> Result<Vec<EaRawEntry>> {
    let mut cursor = Cursor::new(data);
    cursor.set_position(address);

    let sig = cursor.read_bytes(4)?;
    if sig != EADB_SIGNATURE {
        return Err(Error::InvalidExtensibleArraySignature {
            context: "data block signature mismatch",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::Other(format!(
            "unsupported extensible array data block version {}",
            version
        )));
    }

    let _client_id = cursor.read_u8()?;
    let _header_address = cursor.read_offset(offset_size)?;

    // Block offset: sizeof_nelmts bytes indicating this block's element index offset.
    cursor.skip(sizeof_nelmts)?;

    // Paging is used only when nelmts exceeds 2^page_bits.
    let page_nelmts = if max_page_bits > 0 {
        1usize << max_page_bits
    } else {
        0
    };

    if page_nelmts > 0 && num_entries > page_nelmts {
        // Paged data block
        let num_pages = num_entries.div_ceil(page_nelmts);
        let bitmap_bytes = num_pages.div_ceil(8);
        let page_bitmap = cursor.read_bytes(bitmap_bytes)?.to_vec();

        let mut all_entries = Vec::with_capacity(num_entries);
        for page_idx in 0..num_pages {
            let byte_idx = page_idx / 8;
            let bit_idx = page_idx % 8;
            let page_initialized =
                byte_idx < page_bitmap.len() && (page_bitmap[byte_idx] & (1 << bit_idx)) != 0;

            let entries_in_page = if page_idx == num_pages - 1 {
                let remainder = num_entries % page_nelmts;
                if remainder == 0 {
                    page_nelmts
                } else {
                    remainder
                }
            } else {
                page_nelmts
            };

            if page_initialized {
                let page_entries = read_entries(
                    &mut cursor,
                    entries_in_page,
                    is_filtered,
                    offset_size,
                    entry_size,
                )?;
                let _page_checksum = cursor.read_u32_le()?;
                all_entries.extend(page_entries);
            } else {
                for _ in 0..entries_in_page {
                    all_entries.push(EaRawEntry {
                        address: u64::MAX,
                        chunk_size: 0,
                        filter_mask: 0,
                    });
                }
            }
        }
        Ok(all_entries)
    } else {
        // Non-paged data block
        let entries = read_entries(
            &mut cursor,
            num_entries,
            is_filtered,
            offset_size,
            entry_size,
        )?;
        let _checksum = cursor.read_u32_le()?;
        Ok(entries)
    }
}

/// Parse a secondary block and return its data block addresses.
fn parse_secondary_block(
    data: &[u8],
    address: u64,
    num_dblk_addrs: usize,
    offset_size: u8,
    sizeof_nelmts: usize,
    page_bitmap_bytes: usize,
) -> Result<Vec<u64>> {
    let mut cursor = Cursor::new(data);
    cursor.set_position(address);

    let sig = cursor.read_bytes(4)?;
    if sig != EASB_SIGNATURE {
        return Err(Error::InvalidExtensibleArraySignature {
            context: "secondary block signature mismatch",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::Other(format!(
            "unsupported extensible array secondary block version {}",
            version
        )));
    }

    let _client_id = cursor.read_u8()?;
    let _header_address = cursor.read_offset(offset_size)?;
    cursor.skip(sizeof_nelmts)?;

    if page_bitmap_bytes > 0 {
        cursor.skip(page_bitmap_bytes)?;
    }

    let mut addrs = Vec::with_capacity(num_dblk_addrs);
    for _ in 0..num_dblk_addrs {
        addrs.push(cursor.read_offset(offset_size)?);
    }

    // Skip checksum
    let _checksum = cursor.read_u32_le()?;

    Ok(addrs)
}

fn parse_secondary_block_storage(
    storage: &dyn Storage,
    address: u64,
    num_dblk_addrs: usize,
    offset_size: u8,
    sizeof_nelmts: usize,
    page_bitmap_bytes: usize,
) -> Result<Vec<u64>> {
    let _len = 4
        + 1
        + 1
        + usize::from(offset_size)
        + sizeof_nelmts
        + page_bitmap_bytes
        + num_dblk_addrs * usize::from(offset_size)
        + 4;
    let read_len = usize::try_from(storage.len().saturating_sub(address)).map_err(|_| {
        Error::InvalidData(
            "extensible array secondary block exceeds platform usize capacity".into(),
        )
    })?;
    let bytes = storage.read_range(address, read_len)?;
    parse_secondary_block(
        bytes.as_ref(),
        0,
        num_dblk_addrs,
        offset_size,
        sizeof_nelmts,
        page_bitmap_bytes,
    )
}

fn read_entry_at(
    data: &[u8],
    position: u64,
    is_filtered: bool,
    offset_size: u8,
    entry_size: u8,
) -> Result<EaRawEntry> {
    let mut cursor = Cursor::new(data);
    cursor.set_position(position);
    let mut entries = read_entries(&mut cursor, 1, is_filtered, offset_size, entry_size)?;
    entries
        .pop()
        .ok_or_else(|| Error::InvalidData("missing extensible array entry".into()))
}

fn read_entry_at_storage(
    storage: &dyn Storage,
    position: u64,
    is_filtered: bool,
    offset_size: u8,
    entry_size: u8,
) -> Result<EaRawEntry> {
    let bytes = storage.read_range(position, usize::from(entry_size))?;
    let mut cursor = Cursor::new(bytes.as_ref());
    let mut entries = read_entries(&mut cursor, 1, is_filtered, offset_size, entry_size)?;
    entries
        .pop()
        .ok_or_else(|| Error::InvalidData("missing extensible array entry".into()))
}

fn linear_target_offsets(
    dataset_shape: &[u64],
    chunk_dims: &[u32],
    chunk_bounds: Option<(&[u64], &[u64])>,
) -> Vec<(usize, Vec<u64>)> {
    let ndim = dataset_shape.len();
    let chunks_per_dim: Vec<u64> = (0..ndim)
        .map(|i| dataset_shape[i].div_ceil(chunk_dims[i] as u64))
        .collect();

    if ndim == 0 {
        return vec![(0, Vec::new())];
    }

    let (first_chunk, last_chunk): (Vec<u64>, Vec<u64>) = match chunk_bounds {
        Some((first, last)) => (first.to_vec(), last.to_vec()),
        None => (
            vec![0u64; ndim],
            chunks_per_dim
                .iter()
                .map(|count| count.saturating_sub(1))
                .collect(),
        ),
    };

    let mut targets = Vec::new();
    let mut chunk_indices = first_chunk.clone();
    loop {
        let mut linear_idx = 0u64;
        for (dim, chunk_index) in chunk_indices.iter().enumerate() {
            linear_idx = linear_idx * chunks_per_dim[dim] + chunk_index;
        }
        let offsets = chunk_indices
            .iter()
            .enumerate()
            .map(|(dim, chunk_index)| chunk_index * u64::from(chunk_dims[dim]))
            .collect();
        targets.push((linear_idx as usize, offsets));

        let mut advanced = false;
        for dim in (0..ndim).rev() {
            if chunk_indices[dim] < last_chunk[dim] {
                chunk_indices[dim] += 1;
                if dim + 1 < ndim {
                    chunk_indices[(dim + 1)..ndim].copy_from_slice(&first_chunk[(dim + 1)..ndim]);
                }
                advanced = true;
                break;
            }
        }

        if !advanced {
            break;
        }
    }

    targets
}

#[allow(clippy::too_many_arguments)]
fn read_data_block_entry(
    data: &[u8],
    address: u64,
    num_entries: usize,
    local_idx: usize,
    is_filtered: bool,
    max_page_bits: u8,
    offset_size: u8,
    entry_size: u8,
    sizeof_nelmts: usize,
) -> Result<EaRawEntry> {
    let mut cursor = Cursor::new(data);
    cursor.set_position(address);

    let sig = cursor.read_bytes(4)?;
    if sig != EADB_SIGNATURE {
        return Err(Error::InvalidExtensibleArraySignature {
            context: "data block signature mismatch",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::Other(format!(
            "unsupported extensible array data block version {}",
            version
        )));
    }

    let _client_id = cursor.read_u8()?;
    let _header_address = cursor.read_offset(offset_size)?;
    cursor.skip(sizeof_nelmts)?;

    let page_nelmts = if max_page_bits > 0 {
        1usize << max_page_bits
    } else {
        0
    };

    if page_nelmts > 0 && num_entries > page_nelmts {
        let num_pages = num_entries.div_ceil(page_nelmts);
        let bitmap_bytes = num_pages.div_ceil(8);
        let page_bitmap = cursor.read_bytes(bitmap_bytes)?.to_vec();
        let data_start = cursor.position();

        let target_page = local_idx / page_nelmts;
        let within_page = local_idx % page_nelmts;
        let byte_idx = target_page / 8;
        let bit_idx = target_page % 8;
        let page_initialized =
            byte_idx < page_bitmap.len() && (page_bitmap[byte_idx] & (1 << bit_idx)) != 0;
        if !page_initialized {
            return Ok(EaRawEntry {
                address: u64::MAX,
                chunk_size: 0,
                filter_mask: 0,
            });
        }

        let mut page_start = data_start;
        for page_idx in 0..target_page {
            let entries_in_page = if page_idx == num_pages - 1 {
                let remainder = num_entries % page_nelmts;
                if remainder == 0 {
                    page_nelmts
                } else {
                    remainder
                }
            } else {
                page_nelmts
            };
            let page_byte_idx = page_idx / 8;
            let page_bit_idx = page_idx % 8;
            let initialized = page_byte_idx < page_bitmap.len()
                && (page_bitmap[page_byte_idx] & (1 << page_bit_idx)) != 0;
            if initialized {
                page_start += (entries_in_page * entry_size as usize + 4) as u64;
            }
        }

        let position = page_start + (within_page * entry_size as usize) as u64;
        return read_entry_at(data, position, is_filtered, offset_size, entry_size);
    }

    let position = cursor.position() + (local_idx * entry_size as usize) as u64;
    read_entry_at(data, position, is_filtered, offset_size, entry_size)
}

#[allow(clippy::too_many_arguments)]
fn read_data_block_entry_storage(
    storage: &dyn Storage,
    address: u64,
    num_entries: usize,
    local_idx: usize,
    is_filtered: bool,
    max_page_bits: u8,
    offset_size: u8,
    entry_size: u8,
    sizeof_nelmts: usize,
) -> Result<EaRawEntry> {
    let header_len = 4 + 1 + 1 + usize::from(offset_size) + sizeof_nelmts;
    let header = storage.read_range(address, header_len)?;
    let mut cursor = Cursor::new(header.as_ref());

    let sig = cursor.read_bytes(4)?;
    if sig != EADB_SIGNATURE {
        return Err(Error::InvalidExtensibleArraySignature {
            context: "data block signature mismatch",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::Other(format!(
            "unsupported extensible array data block version {}",
            version
        )));
    }

    let _client_id = cursor.read_u8()?;
    let _header_address = cursor.read_offset(offset_size)?;
    cursor.skip(sizeof_nelmts)?;

    let base =
        address + u64::try_from(header_len).map_err(|_| Error::OffsetOutOfBounds(address))?;
    let page_nelmts = if max_page_bits > 0 {
        1usize << max_page_bits
    } else {
        0
    };

    if page_nelmts > 0 && num_entries > page_nelmts {
        let num_pages = num_entries.div_ceil(page_nelmts);
        let bitmap_bytes = num_pages.div_ceil(8);
        let page_bitmap = storage.read_range(base, bitmap_bytes)?;
        let data_start = base
            + u64::try_from(bitmap_bytes)
                .map_err(|_| Error::InvalidData("EA bitmap size exceeds u64 capacity".into()))?;

        let target_page = local_idx / page_nelmts;
        let within_page = local_idx % page_nelmts;
        let byte_idx = target_page / 8;
        let bit_idx = target_page % 8;
        let page_initialized =
            byte_idx < page_bitmap.len() && (page_bitmap[byte_idx] & (1 << bit_idx)) != 0;
        if !page_initialized {
            return Ok(EaRawEntry {
                address: u64::MAX,
                chunk_size: 0,
                filter_mask: 0,
            });
        }

        let mut page_start = data_start;
        for page_idx in 0..target_page {
            let entries_in_page = if page_idx == num_pages - 1 {
                let remainder = num_entries % page_nelmts;
                if remainder == 0 {
                    page_nelmts
                } else {
                    remainder
                }
            } else {
                page_nelmts
            };
            let page_byte_idx = page_idx / 8;
            let page_bit_idx = page_idx % 8;
            let initialized = page_byte_idx < page_bitmap.len()
                && (page_bitmap[page_byte_idx] & (1 << page_bit_idx)) != 0;
            if initialized {
                page_start += u64::try_from(entries_in_page * usize::from(entry_size) + 4)
                    .map_err(|_| Error::InvalidData("EA page size exceeds u64 capacity".into()))?;
            }
        }

        let position = page_start
            + u64::try_from(within_page * usize::from(entry_size)).map_err(|_| {
                Error::InvalidData("EA page entry offset exceeds u64 capacity".into())
            })?;
        return read_entry_at_storage(storage, position, is_filtered, offset_size, entry_size);
    }

    let position = base
        + u64::try_from(local_idx * usize::from(entry_size))
            .map_err(|_| Error::InvalidData("EA entry offset exceeds u64 capacity".into()))?;
    read_entry_at_storage(storage, position, is_filtered, offset_size, entry_size)
}

#[allow(clippy::too_many_arguments)]
fn collect_extensible_array_chunk_entries_bounded(
    data: &[u8],
    header: &EaHeader,
    offset_size: u8,
    dataset_shape: &[u64],
    chunk_dims: &[u32],
    chunk_bounds: (&[u64], &[u64]),
    sb_layout: &[(u64, u64)],
    sizeof_nelmts: usize,
) -> Result<Vec<ChunkEntry>> {
    let is_filtered = header.client_id == 1;
    let targets = linear_target_offsets(dataset_shape, chunk_dims, Some(chunk_bounds));

    let mut cursor = Cursor::new(data);
    cursor.set_position(header.index_block_address);

    let sig = cursor.read_bytes(4)?;
    if sig != EAIB_SIGNATURE {
        return Err(Error::InvalidExtensibleArraySignature {
            context: "index block signature mismatch",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::Other(format!(
            "unsupported extensible array index block version {}",
            version
        )));
    }

    let _client_id = cursor.read_u8()?;
    let _header_address = cursor.read_offset(offset_size)?;

    let num_inline = header.idx_blk_elmts as usize;
    let inline_start = cursor.position();
    cursor.skip(num_inline * header.element_size as usize)?;

    let ndblk_addrs = 2 * header.sec_blk_min_data_ptrs as usize;
    let mut direct_dblk_addrs = Vec::with_capacity(ndblk_addrs);
    for _ in 0..ndblk_addrs {
        direct_dblk_addrs.push(cursor.read_offset(offset_size)?);
    }

    let nsblks = sb_layout.len();
    let nsblk_addrs = nsblks.saturating_sub(ndblk_addrs);
    let mut sec_block_addrs = Vec::with_capacity(nsblk_addrs);
    for _ in 0..nsblk_addrs {
        sec_block_addrs.push(cursor.read_offset(offset_size)?);
    }

    let mut secondary_block_cache: Vec<Option<Vec<u64>>> = vec![None; sec_block_addrs.len()];
    let mut entries = Vec::new();

    for (linear_idx, offsets) in targets {
        let raw = if linear_idx < num_inline {
            read_entry_at(
                data,
                inline_start + (linear_idx * header.element_size as usize) as u64,
                is_filtered,
                offset_size,
                header.element_size,
            )?
        } else {
            let mut relative_idx = (linear_idx - num_inline) as u64;
            let mut sb_idx = None;
            for (candidate_idx, (elmts_per_dblk, num_dblks)) in sb_layout.iter().enumerate() {
                let capacity = elmts_per_dblk * num_dblks;
                if relative_idx < capacity {
                    sb_idx = Some(candidate_idx);
                    break;
                }
                relative_idx -= capacity;
            }

            let Some(sb_idx) = sb_idx else {
                continue;
            };
            let (elmts_per_dblk, _) = sb_layout[sb_idx];
            let dblk_idx = (relative_idx / elmts_per_dblk) as usize;
            let local_idx = (relative_idx % elmts_per_dblk) as usize;

            let dblk_addr = if sb_idx < 2 {
                let base = sb_layout[..sb_idx]
                    .iter()
                    .map(|(_, num_dblks)| *num_dblks as usize)
                    .sum::<usize>();
                *direct_dblk_addrs.get(base + dblk_idx).unwrap_or(&u64::MAX)
            } else {
                let sec_cache_idx = sb_idx - 2;
                if secondary_block_cache[sec_cache_idx].is_none() {
                    let sec_addr = sec_block_addrs
                        .get(sec_cache_idx)
                        .copied()
                        .unwrap_or(u64::MAX);
                    if Cursor::is_undefined_offset(sec_addr, offset_size) {
                        secondary_block_cache[sec_cache_idx] = Some(Vec::new());
                    } else {
                        let (_, num_dblks) = sb_layout[sb_idx];
                        let page_bitmap_bytes = if header.max_dblk_page_nelmts_bits > 0
                            && elmts_per_dblk > (1u64 << header.max_dblk_page_nelmts_bits)
                        {
                            let page_nelmts = 1usize << header.max_dblk_page_nelmts_bits;
                            let pages_per_dblk = (elmts_per_dblk as usize).div_ceil(page_nelmts);
                            (num_dblks as usize * pages_per_dblk).div_ceil(8)
                        } else {
                            0
                        };
                        secondary_block_cache[sec_cache_idx] = Some(parse_secondary_block(
                            data,
                            sec_addr,
                            num_dblks as usize,
                            offset_size,
                            sizeof_nelmts,
                            page_bitmap_bytes,
                        )?);
                    }
                }

                secondary_block_cache[sec_cache_idx]
                    .as_ref()
                    .and_then(|addrs| addrs.get(dblk_idx))
                    .copied()
                    .unwrap_or(u64::MAX)
            };

            if Cursor::is_undefined_offset(dblk_addr, offset_size) {
                continue;
            }

            read_data_block_entry(
                data,
                dblk_addr,
                elmts_per_dblk as usize,
                local_idx,
                is_filtered,
                header.max_dblk_page_nelmts_bits,
                offset_size,
                header.element_size,
                sizeof_nelmts,
            )?
        };

        if Cursor::is_undefined_offset(raw.address, offset_size) {
            continue;
        }

        entries.push(ChunkEntry {
            address: raw.address,
            size: raw.chunk_size,
            filter_mask: raw.filter_mask,
            offsets,
        });
    }

    Ok(entries)
}

/// Collect chunk entries from an Extensible Array index.
///
/// Walks the EAHD → EAIB → (EADB / EASB → EADB) hierarchy and converts
/// linear entry indices to multi-dimensional chunk offsets.
pub fn collect_extensible_array_chunk_entries(
    data: &[u8],
    header_address: u64,
    offset_size: u8,
    length_size: u8,
    dataset_shape: &[u64],
    chunk_dims: &[u32],
    chunk_bounds: Option<(&[u64], &[u64])>,
) -> Result<Vec<ChunkEntry>> {
    let header = parse_header(data, header_address, offset_size, length_size)?;

    if Cursor::is_undefined_offset(header.index_block_address, offset_size) {
        return Ok(Vec::new());
    }

    let is_filtered = header.client_id == 1;
    let sb_layout = compute_super_block_layout(&header);
    let sizeof_nelmts = (header._max_nelmts_bits as usize).div_ceil(8);

    if let Some(bounds) = chunk_bounds {
        return collect_extensible_array_chunk_entries_bounded(
            data,
            &header,
            offset_size,
            dataset_shape,
            chunk_dims,
            bounds,
            &sb_layout,
            sizeof_nelmts,
        );
    }

    // Parse the index block.
    let mut cursor = Cursor::new(data);
    cursor.set_position(header.index_block_address);

    let sig = cursor.read_bytes(4)?;
    if sig != EAIB_SIGNATURE {
        return Err(Error::InvalidExtensibleArraySignature {
            context: "index block signature mismatch",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::Other(format!(
            "unsupported extensible array index block version {}",
            version
        )));
    }

    let _client_id = cursor.read_u8()?;
    let _header_address = cursor.read_offset(offset_size)?;

    // 1. Inline elements (idx_blk_elmts entries stored directly).
    let num_inline = header.idx_blk_elmts as usize;
    let inline_entries = read_entries(
        &mut cursor,
        num_inline,
        is_filtered,
        offset_size,
        header.element_size,
    )?;

    // 2. Data block addresses stored directly in the index block.
    // The number is 2 * sec_blk_min_data_ptrs (from HDF5: EA_IBLOCK_NDBLK_ADDRS).
    let ndblk_addrs = 2 * header.sec_blk_min_data_ptrs as usize;
    let mut direct_dblk_addrs = Vec::with_capacity(ndblk_addrs);
    for _ in 0..ndblk_addrs {
        direct_dblk_addrs.push(cursor.read_offset(offset_size)?);
    }

    // 3. Secondary block addresses for super blocks 2+.
    // nsblk_addrs = max(0, nsblks - ndblk_addrs) where nsblks is the total
    // number of super blocks needed to cover nelmts.
    // compute_super_block_layout already stops once capacity >= nelmts,
    // so sb_layout.len() is the total number of super blocks needed.
    let nsblks = sb_layout.len();

    let nsblk_addrs = nsblks.saturating_sub(ndblk_addrs);
    let mut sec_block_addrs = Vec::with_capacity(nsblk_addrs);
    for _ in 0..nsblk_addrs {
        sec_block_addrs.push(cursor.read_offset(offset_size)?);
    }

    // Skip checksum at end of index block
    let _checksum = cursor.read_u32_le()?;

    // Now collect all entries.
    let mut all_entries: Vec<EaRawEntry> = Vec::new();

    // Inline entries
    all_entries.extend(inline_entries);

    // Data blocks from direct addresses (super blocks 0-1)
    let mut dblk_addr_idx = 0;
    for sb_idx_iter in 0..2usize.min(nsblks) {
        if sb_idx_iter >= sb_layout.len() {
            break;
        }
        let (elmts_per_dblk, num_dblks) = sb_layout[sb_idx_iter];
        for _ in 0..num_dblks {
            if dblk_addr_idx >= direct_dblk_addrs.len() {
                break;
            }
            let dblk_addr = direct_dblk_addrs[dblk_addr_idx];
            dblk_addr_idx += 1;

            if Cursor::is_undefined_offset(dblk_addr, offset_size) {
                for _ in 0..elmts_per_dblk {
                    all_entries.push(EaRawEntry {
                        address: u64::MAX,
                        chunk_size: 0,
                        filter_mask: 0,
                    });
                }
            } else {
                let dblk_entries = parse_data_block(
                    data,
                    dblk_addr,
                    elmts_per_dblk as usize,
                    is_filtered,
                    header.max_dblk_page_nelmts_bits,
                    offset_size,
                    header.element_size,
                    sizeof_nelmts,
                )?;
                all_entries.extend(dblk_entries);
            }
        }
    }

    // Data blocks from super blocks 2+ (via secondary blocks)
    for (sec_idx, &sec_addr) in sec_block_addrs.iter().enumerate() {
        let sb_idx_iter = sec_idx + 2;
        if sb_idx_iter >= sb_layout.len() {
            break;
        }
        let (elmts_per_dblk, num_dblks) = sb_layout[sb_idx_iter];

        if Cursor::is_undefined_offset(sec_addr, offset_size) {
            for _ in 0..(elmts_per_dblk * num_dblks) {
                all_entries.push(EaRawEntry {
                    address: u64::MAX,
                    chunk_size: 0,
                    filter_mask: 0,
                });
            }
            continue;
        }

        // Per HDF5 spec III.H "Extensible Array Secondary Block", the secondary
        // block contains a page initialization bitmap when data blocks are paged.
        // Bitmap size = ceil(num_dblks * pages_per_dblk / 8).
        let page_bitmap_bytes = if header.max_dblk_page_nelmts_bits > 0
            && elmts_per_dblk > (1u64 << header.max_dblk_page_nelmts_bits)
        {
            let page_nelmts = 1usize << header.max_dblk_page_nelmts_bits;
            let pages_per_dblk = (elmts_per_dblk as usize).div_ceil(page_nelmts);
            (num_dblks as usize * pages_per_dblk).div_ceil(8)
        } else {
            0
        };
        let dblk_addrs = parse_secondary_block(
            data,
            sec_addr,
            num_dblks as usize,
            offset_size,
            sizeof_nelmts,
            page_bitmap_bytes,
        )?;

        for &dblk_addr in &dblk_addrs {
            if Cursor::is_undefined_offset(dblk_addr, offset_size) {
                for _ in 0..elmts_per_dblk {
                    all_entries.push(EaRawEntry {
                        address: u64::MAX,
                        chunk_size: 0,
                        filter_mask: 0,
                    });
                }
            } else {
                let dblk_entries = parse_data_block(
                    data,
                    dblk_addr,
                    elmts_per_dblk as usize,
                    is_filtered,
                    header.max_dblk_page_nelmts_bits,
                    offset_size,
                    header.element_size,
                    sizeof_nelmts,
                )?;
                all_entries.extend(dblk_entries);
            }
        }
    }

    // Convert linear indices to chunk offsets.
    let ndim = dataset_shape.len();
    let chunks_per_dim: Vec<u64> = (0..ndim)
        .map(|i| dataset_shape[i].div_ceil(chunk_dims[i] as u64))
        .collect();

    let mut entries = Vec::new();
    for (linear_idx, raw) in all_entries.iter().enumerate() {
        if Cursor::is_undefined_offset(raw.address, offset_size) {
            continue;
        }

        let mut remaining = linear_idx as u64;
        let mut offsets = vec![0u64; ndim];
        for d in (0..ndim).rev() {
            offsets[d] = (remaining % chunks_per_dim[d]) * chunk_dims[d] as u64;
            remaining /= chunks_per_dim[d];
        }

        if let Some((first_chunk, last_chunk)) = chunk_bounds {
            let overlaps = offsets.iter().enumerate().all(|(dim, offset)| {
                let chunk_index = *offset / u64::from(chunk_dims[dim]);
                chunk_index >= first_chunk[dim] && chunk_index <= last_chunk[dim]
            });
            if !overlaps {
                continue;
            }
        }

        entries.push(ChunkEntry {
            address: raw.address,
            size: raw.chunk_size,
            filter_mask: raw.filter_mask,
            offsets,
        });
    }

    Ok(entries)
}

/// Collect chunk entries from an Extensible Array index using random-access storage.
pub fn collect_extensible_array_chunk_entries_storage(
    storage: &dyn Storage,
    header_address: u64,
    offset_size: u8,
    length_size: u8,
    dataset_shape: &[u64],
    chunk_dims: &[u32],
    chunk_bounds: Option<(&[u64], &[u64])>,
) -> Result<Vec<ChunkEntry>> {
    let header = parse_header_storage(storage, header_address, offset_size, length_size)?;

    if Cursor::is_undefined_offset(header.index_block_address, offset_size) {
        return Ok(Vec::new());
    }

    let is_filtered = header.client_id == 1;
    let sb_layout = compute_super_block_layout(&header);
    let sizeof_nelmts = (header._max_nelmts_bits as usize).div_ceil(8);

    if let Some(bounds) = chunk_bounds {
        let targets = linear_target_offsets(dataset_shape, chunk_dims, Some(bounds));
        let _index_block_len = 4
            + 1
            + 1
            + usize::from(offset_size)
            + usize::from(header.idx_blk_elmts) * usize::from(header.element_size)
            + (2 * usize::from(header.sec_blk_min_data_ptrs)) * usize::from(offset_size)
            + sb_layout
                .len()
                .saturating_sub(2 * usize::from(header.sec_blk_min_data_ptrs))
                * usize::from(offset_size)
            + 4;
        let index_block = storage.read_range(
            header.index_block_address,
            usize::try_from(storage.len().saturating_sub(header.index_block_address)).map_err(
                |_| {
                    Error::InvalidData(
                        "extensible array index block exceeds platform usize capacity".into(),
                    )
                },
            )?,
        )?;
        let mut cursor = Cursor::new(index_block.as_ref());
        let sig = cursor.read_bytes(4)?;
        if sig != EAIB_SIGNATURE {
            return Err(Error::InvalidExtensibleArraySignature {
                context: "index block signature mismatch",
            });
        }
        let version = cursor.read_u8()?;
        if version != 0 {
            return Err(Error::Other(format!(
                "unsupported extensible array index block version {}",
                version
            )));
        }
        let _client_id = cursor.read_u8()?;
        let _header_address = cursor.read_offset(offset_size)?;
        let num_inline = header.idx_blk_elmts as usize;
        let inline_start = cursor.position();
        cursor.skip(num_inline * header.element_size as usize)?;

        let ndblk_addrs = 2 * header.sec_blk_min_data_ptrs as usize;
        let mut direct_dblk_addrs = Vec::with_capacity(ndblk_addrs);
        for _ in 0..ndblk_addrs {
            direct_dblk_addrs.push(cursor.read_offset(offset_size)?);
        }

        let nsblks = sb_layout.len();
        let nsblk_addrs = nsblks.saturating_sub(ndblk_addrs);
        let mut sec_block_addrs = Vec::with_capacity(nsblk_addrs);
        for _ in 0..nsblk_addrs {
            sec_block_addrs.push(cursor.read_offset(offset_size)?);
        }

        let mut secondary_block_cache: Vec<Option<Vec<u64>>> = vec![None; sec_block_addrs.len()];
        let mut entries = Vec::new();

        for (linear_idx, offsets) in targets {
            let raw = if linear_idx < num_inline {
                let inline_offset = inline_start
                    + u64::try_from(linear_idx * usize::from(header.element_size)).map_err(
                        |_| {
                            Error::InvalidData("EA inline entry offset exceeds u64 capacity".into())
                        },
                    )?;
                let position = header.index_block_address + inline_offset;
                read_entry_at_storage(
                    storage,
                    position,
                    is_filtered,
                    offset_size,
                    header.element_size,
                )?
            } else {
                let mut relative_idx = (linear_idx - num_inline) as u64;
                let mut sb_idx = None;
                for (candidate_idx, (elmts_per_dblk, num_dblks)) in sb_layout.iter().enumerate() {
                    let capacity = elmts_per_dblk * num_dblks;
                    if relative_idx < capacity {
                        sb_idx = Some(candidate_idx);
                        break;
                    }
                    relative_idx -= capacity;
                }

                let Some(sb_idx) = sb_idx else {
                    continue;
                };
                let (elmts_per_dblk, _) = sb_layout[sb_idx];
                let dblk_idx = (relative_idx / elmts_per_dblk) as usize;
                let local_idx = (relative_idx % elmts_per_dblk) as usize;

                let dblk_addr = if sb_idx < 2 {
                    let base = sb_layout[..sb_idx]
                        .iter()
                        .map(|(_, num_dblks)| *num_dblks as usize)
                        .sum::<usize>();
                    *direct_dblk_addrs.get(base + dblk_idx).unwrap_or(&u64::MAX)
                } else {
                    let sec_cache_idx = sb_idx - 2;
                    if secondary_block_cache[sec_cache_idx].is_none() {
                        let sec_addr = sec_block_addrs
                            .get(sec_cache_idx)
                            .copied()
                            .unwrap_or(u64::MAX);
                        if Cursor::is_undefined_offset(sec_addr, offset_size) {
                            secondary_block_cache[sec_cache_idx] = Some(Vec::new());
                        } else {
                            let (_, num_dblks) = sb_layout[sb_idx];
                            let page_bitmap_bytes = if header.max_dblk_page_nelmts_bits > 0
                                && elmts_per_dblk > (1u64 << header.max_dblk_page_nelmts_bits)
                            {
                                let page_nelmts = 1usize << header.max_dblk_page_nelmts_bits;
                                let pages_per_dblk =
                                    (elmts_per_dblk as usize).div_ceil(page_nelmts);
                                (num_dblks as usize * pages_per_dblk).div_ceil(8)
                            } else {
                                0
                            };
                            secondary_block_cache[sec_cache_idx] =
                                Some(parse_secondary_block_storage(
                                    storage,
                                    sec_addr,
                                    num_dblks as usize,
                                    offset_size,
                                    sizeof_nelmts,
                                    page_bitmap_bytes,
                                )?);
                        }
                    }

                    secondary_block_cache[sec_cache_idx]
                        .as_ref()
                        .and_then(|addrs| addrs.get(dblk_idx))
                        .copied()
                        .unwrap_or(u64::MAX)
                };

                if Cursor::is_undefined_offset(dblk_addr, offset_size) {
                    continue;
                }

                read_data_block_entry_storage(
                    storage,
                    dblk_addr,
                    elmts_per_dblk as usize,
                    local_idx,
                    is_filtered,
                    header.max_dblk_page_nelmts_bits,
                    offset_size,
                    header.element_size,
                    sizeof_nelmts,
                )?
            };

            if Cursor::is_undefined_offset(raw.address, offset_size) {
                continue;
            }

            entries.push(ChunkEntry {
                address: raw.address,
                size: raw.chunk_size,
                filter_mask: raw.filter_mask,
                offsets,
            });
        }

        return Ok(entries);
    }

    let index_block_len = 4
        + 1
        + 1
        + usize::from(offset_size)
        + usize::from(header.idx_blk_elmts) * usize::from(header.element_size)
        + (2 * usize::from(header.sec_blk_min_data_ptrs)) * usize::from(offset_size)
        + sb_layout
            .len()
            .saturating_sub(2 * usize::from(header.sec_blk_min_data_ptrs))
            * usize::from(offset_size)
        + 4;
    let data = storage.read_range(header.index_block_address, index_block_len)?;
    let mut cursor = Cursor::new(data.as_ref());
    cursor.set_position(0);

    let sig = cursor.read_bytes(4)?;
    if sig != EAIB_SIGNATURE {
        return Err(Error::InvalidExtensibleArraySignature {
            context: "index block signature mismatch",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::Other(format!(
            "unsupported extensible array index block version {}",
            version
        )));
    }

    let _client_id = cursor.read_u8()?;
    let _header_address = cursor.read_offset(offset_size)?;

    let num_inline = header.idx_blk_elmts as usize;
    let inline_entries = read_entries(
        &mut cursor,
        num_inline,
        is_filtered,
        offset_size,
        header.element_size,
    )?;

    let ndblk_addrs = 2 * header.sec_blk_min_data_ptrs as usize;
    let mut direct_dblk_addrs = Vec::with_capacity(ndblk_addrs);
    for _ in 0..ndblk_addrs {
        direct_dblk_addrs.push(cursor.read_offset(offset_size)?);
    }

    let nsblks = sb_layout.len();
    let nsblk_addrs = nsblks.saturating_sub(ndblk_addrs);
    let mut sec_block_addrs = Vec::with_capacity(nsblk_addrs);
    for _ in 0..nsblk_addrs {
        sec_block_addrs.push(cursor.read_offset(offset_size)?);
    }
    let _checksum = cursor.read_u32_le()?;

    let mut all_entries: Vec<EaRawEntry> = Vec::new();
    all_entries.extend(inline_entries);

    let mut dblk_addr_idx = 0;
    for sb_idx_iter in 0..2usize.min(nsblks) {
        if sb_idx_iter >= sb_layout.len() {
            break;
        }
        let (elmts_per_dblk, num_dblks) = sb_layout[sb_idx_iter];
        for _ in 0..num_dblks {
            if dblk_addr_idx >= direct_dblk_addrs.len() {
                break;
            }
            let dblk_addr = direct_dblk_addrs[dblk_addr_idx];
            dblk_addr_idx += 1;

            if Cursor::is_undefined_offset(dblk_addr, offset_size) {
                for _ in 0..elmts_per_dblk {
                    all_entries.push(EaRawEntry {
                        address: u64::MAX,
                        chunk_size: 0,
                        filter_mask: 0,
                    });
                }
            } else {
                let dblk_entries = {
                    let page_nelmts = if header.max_dblk_page_nelmts_bits > 0 {
                        1usize << header.max_dblk_page_nelmts_bits
                    } else {
                        0
                    };
                    let _dblk_len = if page_nelmts > 0 && elmts_per_dblk as usize > page_nelmts {
                        let num_pages = (elmts_per_dblk as usize).div_ceil(page_nelmts);
                        let bitmap_bytes = num_pages.div_ceil(8);
                        let mut len =
                            4 + 1 + 1 + usize::from(offset_size) + sizeof_nelmts + bitmap_bytes;
                        for page_idx in 0..num_pages {
                            let entries_in_page = if page_idx == num_pages - 1 {
                                let remainder = elmts_per_dblk as usize % page_nelmts;
                                if remainder == 0 {
                                    page_nelmts
                                } else {
                                    remainder
                                }
                            } else {
                                page_nelmts
                            };
                            len += entries_in_page * usize::from(header.element_size) + 4;
                        }
                        len
                    } else {
                        4 + 1
                            + 1
                            + usize::from(offset_size)
                            + sizeof_nelmts
                            + elmts_per_dblk as usize * usize::from(header.element_size)
                            + 4
                    };
                    let block = storage.read_range(
                        dblk_addr,
                        usize::try_from(storage.len().saturating_sub(dblk_addr)).map_err(|_| {
                            Error::InvalidData(
                                "extensible array data block exceeds platform usize capacity"
                                    .into(),
                            )
                        })?,
                    )?;
                    parse_data_block(
                        block.as_ref(),
                        0,
                        elmts_per_dblk as usize,
                        is_filtered,
                        header.max_dblk_page_nelmts_bits,
                        offset_size,
                        header.element_size,
                        sizeof_nelmts,
                    )?
                };
                all_entries.extend(dblk_entries);
            }
        }
    }

    for (sb_idx_iter, &(elmts_per_dblk, num_dblks)) in sb_layout.iter().enumerate().skip(2) {
        let sec_idx = sb_idx_iter - 2;
        let sec_addr = *sec_block_addrs.get(sec_idx).unwrap_or(&u64::MAX);
        if Cursor::is_undefined_offset(sec_addr, offset_size) {
            for _ in 0..(elmts_per_dblk * num_dblks) {
                all_entries.push(EaRawEntry {
                    address: u64::MAX,
                    chunk_size: 0,
                    filter_mask: 0,
                });
            }
            continue;
        }

        let page_bitmap_bytes = if header.max_dblk_page_nelmts_bits > 0
            && elmts_per_dblk > (1u64 << header.max_dblk_page_nelmts_bits)
        {
            let page_nelmts = 1usize << header.max_dblk_page_nelmts_bits;
            let pages_per_dblk = (elmts_per_dblk as usize).div_ceil(page_nelmts);
            (num_dblks as usize * pages_per_dblk).div_ceil(8)
        } else {
            0
        };
        let dblk_addrs = parse_secondary_block_storage(
            storage,
            sec_addr,
            num_dblks as usize,
            offset_size,
            sizeof_nelmts,
            page_bitmap_bytes,
        )?;

        for dblk_addr in dblk_addrs {
            if Cursor::is_undefined_offset(dblk_addr, offset_size) {
                for _ in 0..elmts_per_dblk {
                    all_entries.push(EaRawEntry {
                        address: u64::MAX,
                        chunk_size: 0,
                        filter_mask: 0,
                    });
                }
            } else {
                let page_nelmts = if header.max_dblk_page_nelmts_bits > 0 {
                    1usize << header.max_dblk_page_nelmts_bits
                } else {
                    0
                };
                let _dblk_len = if page_nelmts > 0 && elmts_per_dblk as usize > page_nelmts {
                    let num_pages = (elmts_per_dblk as usize).div_ceil(page_nelmts);
                    let bitmap_bytes = num_pages.div_ceil(8);
                    let mut len =
                        4 + 1 + 1 + usize::from(offset_size) + sizeof_nelmts + bitmap_bytes;
                    for page_idx in 0..num_pages {
                        let entries_in_page = if page_idx == num_pages - 1 {
                            let remainder = elmts_per_dblk as usize % page_nelmts;
                            if remainder == 0 {
                                page_nelmts
                            } else {
                                remainder
                            }
                        } else {
                            page_nelmts
                        };
                        len += entries_in_page * usize::from(header.element_size) + 4;
                    }
                    len
                } else {
                    4 + 1
                        + 1
                        + usize::from(offset_size)
                        + sizeof_nelmts
                        + elmts_per_dblk as usize * usize::from(header.element_size)
                        + 4
                };
                let block = storage.read_range(
                    dblk_addr,
                    usize::try_from(storage.len().saturating_sub(dblk_addr)).map_err(|_| {
                        Error::InvalidData(
                            "extensible array data block exceeds platform usize capacity".into(),
                        )
                    })?,
                )?;
                let dblk_entries = parse_data_block(
                    block.as_ref(),
                    0,
                    elmts_per_dblk as usize,
                    is_filtered,
                    header.max_dblk_page_nelmts_bits,
                    offset_size,
                    header.element_size,
                    sizeof_nelmts,
                )?;
                all_entries.extend(dblk_entries);
            }
        }
    }

    let ndim = dataset_shape.len();
    let chunks_per_dim: Vec<u64> = (0..ndim)
        .map(|i| dataset_shape[i].div_ceil(chunk_dims[i] as u64))
        .collect();

    let mut entries = Vec::new();
    for (linear_idx, raw) in all_entries.iter().enumerate() {
        if Cursor::is_undefined_offset(raw.address, offset_size) {
            continue;
        }

        let mut remaining = linear_idx as u64;
        let mut offsets = vec![0u64; ndim];
        for d in (0..ndim).rev() {
            offsets[d] = (remaining % chunks_per_dim[d]) * chunk_dims[d] as u64;
            remaining /= chunks_per_dim[d];
        }

        if let Some((first_chunk, last_chunk)) = chunk_bounds {
            let overlaps = offsets.iter().enumerate().all(|(dim, offset)| {
                let chunk_index = *offset / u64::from(chunk_dims[dim]);
                chunk_index >= first_chunk[dim] && chunk_index <= last_chunk[dim]
            });
            if !overlaps {
                continue;
            }
        }

        entries.push(ChunkEntry {
            address: raw.address,
            size: raw.chunk_size,
            filter_mask: raw.filter_mask,
            offsets,
        });
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eahd_bad_signature() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"XXXX");
        let err = parse_header(&data, 0, 8, 8).unwrap_err();
        assert!(matches!(err, Error::InvalidExtensibleArraySignature { .. }));
    }

    #[test]
    fn test_compute_super_block_layout() {
        let header = EaHeader {
            client_id: 0,
            element_size: 8,
            _max_nelmts_bits: 32,
            idx_blk_elmts: 2,
            data_blk_min_elmts: 2,
            sec_blk_min_data_ptrs: 2,
            max_dblk_page_nelmts_bits: 0,
            _nelmts: 100,
            index_block_address: 0,
        };
        let layout = compute_super_block_layout(&header);
        // sb 0: elmts_per_dblk = 2 * 2^0 = 2, num_dblks = 2 * 2^0 = 2  (cap = 4 elements)
        assert_eq!(layout[0], (2, 2));
        // sb 1: elmts_per_dblk = 2 * 2^0 = 2, num_dblks = 2 * 2^1 = 4  (cap = 8 elements)
        assert_eq!(layout[1], (2, 4));
        // sb 2: elmts_per_dblk = 2 * 2^1 = 4, num_dblks = 2 * 2^1 = 4  (cap = 16 elements)
        assert_eq!(layout[2], (4, 4));
        // sb 3: elmts_per_dblk = 2 * 2^1 = 4, num_dblks = 2 * 2^2 = 8  (cap = 32 elements)
        assert_eq!(layout[3], (4, 8));
    }
}
