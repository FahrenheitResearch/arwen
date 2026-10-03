//! Chunk indexes (sections III.A.1, III.A.2, VII-VIII "Fixed Array" and
//! "Extensible Array" of the HDF5 File Format Specification) and chunk
//! assembly into a dataset buffer.

use std::collections::{HashMap, HashSet};

use crate::btree1;
use crate::btree2;
use crate::bytes::{Cursor, UNDEFINED_ADDR, log2_floor, to_usize};
use crate::checksum::lookup3;
use crate::dataspace::UNLIMITED;
use crate::error::{Result, invalid, limit};
use crate::filters::{self, Filter};
use crate::layout::ChunkIndex;
use crate::limits::{MAX_DATA_CHUNKS, MAX_DATASET_BYTES};
use crate::space::Space;

/// One stored chunk: address, stored size, filter mask and chunk
/// coordinates (element offset / chunk dimension).
#[derive(Clone, Debug)]
pub(crate) struct Chunk {
    pub(crate) address: u64,
    pub(crate) size: usize,
    pub(crate) filter_mask: u32,
    pub(crate) scaled: Vec<u64>,
}

/// Dataset geometry the chunk indexes need.
pub(crate) struct Geometry<'g> {
    pub(crate) dims: &'g [u64],
    pub(crate) max_dims: &'g [u64],
    pub(crate) chunk_dims: &'g [usize],
    pub(crate) element_size: usize,
}

impl Geometry<'_> {
    fn chunk_bytes(&self) -> Result<usize> {
        self.chunk_dims
            .iter()
            .try_fold(self.element_size, |acc, dim| acc.checked_mul(*dim))
            .filter(|bytes| *bytes <= MAX_DATASET_BYTES)
            .ok_or_else(|| {
                limit(format!(
                    "HDF5 chunk exceeds {MAX_DATASET_BYTES} bytes (limit)"
                ))
            })
    }

    /// Chunks per dimension covering the current dimensions.
    fn chunk_counts(&self) -> Vec<u64> {
        self.dims
            .iter()
            .zip(self.chunk_dims)
            .map(|(dim, chunk)| dim.div_ceil(*chunk as u64))
            .collect()
    }

    /// Chunks per dimension covering the maximum dimensions
    /// ([`UNLIMITED`] stays unlimited).
    fn max_chunk_counts(&self) -> Vec<u64> {
        self.max_dims
            .iter()
            .zip(self.chunk_dims)
            .map(|(dim, chunk)| {
                if *dim == UNLIMITED {
                    UNLIMITED
                } else {
                    dim.div_ceil(*chunk as u64)
                }
            })
            .collect()
    }
}

/// `H5VM_array_down`: row-major strides of a chunk-count array.
fn down(counts: &[u64]) -> Vec<u64> {
    let mut out = vec![1u64; counts.len()];
    for index in (0..counts.len().saturating_sub(1)).rev() {
        out[index] = out[index + 1].saturating_mul(counts[index + 1]);
    }
    out
}

/// Visit every chunk coordinate inside the current dimensions, row-major.
fn for_each_coordinate(counts: &[u64], mut visit: impl FnMut(&[u64]) -> Result<()>) -> Result<()> {
    let total = counts
        .iter()
        .try_fold(1u64, |acc, count| acc.checked_mul(*count))
        .ok_or_else(|| invalid(0, "HDF5 chunk count overflow"))?;
    if total > MAX_DATA_CHUNKS as u64 {
        return Err(limit(format!(
            "HDF5 dataset has {total} chunks (limit {MAX_DATA_CHUNKS})"
        )));
    }
    if total == 0 {
        return Ok(());
    }
    let mut current = vec![0u64; counts.len()];
    loop {
        visit(&current)?;
        let mut dim = counts.len();
        loop {
            if dim == 0 {
                return Ok(());
            }
            dim -= 1;
            current[dim] += 1;
            if current[dim] < counts[dim] {
                break;
            }
            current[dim] = 0;
        }
    }
}

fn linear(scaled: &[u64], strides: &[u64]) -> u64 {
    scaled
        .iter()
        .zip(strides)
        .fold(0u64, |acc, (coordinate, stride)| {
            acc.saturating_add(coordinate.saturating_mul(*stride))
        })
}

/// Every allocated chunk of a dataset.
pub(crate) fn collect(
    space: &Space<'_>,
    index: &ChunkIndex,
    geometry: &Geometry<'_>,
) -> Result<Vec<Chunk>> {
    let chunk_bytes = geometry.chunk_bytes()?;
    let rank = geometry.chunk_dims.len();
    if geometry.dims.len() != rank || geometry.max_dims.len() != rank {
        return Err(invalid(
            0,
            format!(
                "HDF5 chunks of rank {rank} in a dataspace of rank {} (maximum rank {})",
                geometry.dims.len(),
                geometry.max_dims.len()
            ),
        ));
    }
    match index {
        ChunkIndex::BTreeV1 { address } => {
            if *address == UNDEFINED_ADDR {
                return Ok(Vec::new());
            }
            btree1::chunks(space, *address, rank)?
                .into_iter()
                .map(|record| {
                    let scaled = record
                        .offsets
                        .iter()
                        .zip(geometry.chunk_dims)
                        .map(|(offset, dim)| offset / *dim as u64)
                        .collect();
                    Ok(Chunk {
                        address: record.address,
                        size: record.size,
                        filter_mask: record.filter_mask,
                        scaled,
                    })
                })
                .collect()
        }
        ChunkIndex::SingleChunk { address, filtered } => {
            if *address == UNDEFINED_ADDR {
                return Ok(Vec::new());
            }
            let (size, filter_mask) = filtered.unwrap_or((chunk_bytes, 0));
            Ok(vec![Chunk {
                address: *address,
                size,
                filter_mask,
                scaled: vec![0; rank],
            }])
        }
        ChunkIndex::Implicit { address } => {
            if *address == UNDEFINED_ADDR {
                return Ok(Vec::new());
            }
            let strides = down(&geometry.max_chunk_counts());
            let mut out = Vec::new();
            for_each_coordinate(&geometry.chunk_counts(), |scaled| {
                let offset = linear(scaled, &strides)
                    .checked_mul(chunk_bytes as u64)
                    .and_then(|offset| address.checked_add(offset))
                    .ok_or_else(|| invalid(0, "implicit chunk address overflow"))?;
                out.push(Chunk {
                    address: offset,
                    size: chunk_bytes,
                    filter_mask: 0,
                    scaled: scaled.to_vec(),
                });
                Ok(())
            })?;
            Ok(out)
        }
        ChunkIndex::FixedArray { address } => {
            if *address == UNDEFINED_ADDR {
                return Ok(Vec::new());
            }
            let max_counts = geometry.max_chunk_counts();
            if max_counts.contains(&UNLIMITED) {
                return Err(invalid(
                    0,
                    "fixed array chunk index on an unlimited dimension",
                ));
            }
            let strides = down(&max_counts);
            let mut array = FixedArray::parse(space, *address)?;
            let mut out = Vec::new();
            for_each_coordinate(&geometry.chunk_counts(), |scaled| {
                if let Some(entry) = array.entry(space, linear(scaled, &strides))? {
                    out.push(chunk_from_entry(
                        entry,
                        space.offset_size,
                        chunk_bytes,
                        scaled.to_vec(),
                    )?);
                }
                Ok(())
            })?;
            Ok(out)
        }
        ChunkIndex::ExtensibleArray { address } => {
            if *address == UNDEFINED_ADDR {
                return Ok(Vec::new());
            }
            let max_counts = geometry.max_chunk_counts();
            let unlimited: Vec<usize> = (0..rank)
                .filter(|dim| max_counts[*dim] == UNLIMITED)
                .collect();
            let [unlim_dim] = unlimited[..] else {
                return Err(invalid(
                    0,
                    "extensible array chunk index needs exactly one unlimited dimension",
                ));
            };
            // H5D__earray_idx_resize: the unlimited dimension moves to the
            // front ("swizzle"), the others keep their order.
            let swizzle = |values: &[u64]| {
                let mut out = Vec::with_capacity(values.len());
                out.push(values[unlim_dim]);
                out.extend(
                    values
                        .iter()
                        .enumerate()
                        .filter(|(dim, _)| *dim != unlim_dim)
                        .map(|(_, value)| *value),
                );
                out
            };
            let strides = down(&swizzle(&max_counts));
            let mut array = ExtensibleArray::parse(space, *address)?;
            let mut out = Vec::new();
            for_each_coordinate(&geometry.chunk_counts(), |scaled| {
                let index = linear(&swizzle(scaled), &strides);
                if let Some(entry) = array.entry(space, index)? {
                    out.push(chunk_from_entry(
                        entry,
                        space.offset_size,
                        chunk_bytes,
                        scaled.to_vec(),
                    )?);
                }
                Ok(())
            })?;
            Ok(out)
        }
        ChunkIndex::BTreeV2 { address } => {
            if *address == UNDEFINED_ADDR {
                return Ok(Vec::new());
            }
            let mut out = Vec::new();
            let offset_size = space.offset_size;
            let header = btree2::header(space, *address)?;
            let filtered = match header.record_type {
                10 => false,
                11 => true,
                other => {
                    return Err(invalid(
                        0,
                        format!("v2 B-tree chunk index with record type {other}"),
                    ));
                }
            };
            let size_len = if filtered {
                header
                    .record_size
                    .checked_sub(offset_size + 4 + 8 * rank)
                    .filter(|len| (1..=8).contains(len))
                    .ok_or_else(|| invalid(0, "v2 B-tree chunk record size"))?
            } else {
                0
            };
            // Address (O), [stored size, filter mask (u32)], scaled
            // offsets (u64 per dimension).
            let min_record = offset_size.saturating_add(8 * rank);
            btree2::for_each_record(
                space,
                *address,
                &[header.record_type],
                min_record,
                &mut |record| {
                    if out.len() >= MAX_DATA_CHUNKS {
                        return Err(limit(format!(
                            "HDF5 dataset has more than {MAX_DATA_CHUNKS} chunks (limit)"
                        )));
                    }
                    let mut cursor = Cursor::new(record, 0);
                    let address = cursor.addr(offset_size)?;
                    let (size, filter_mask) = if filtered {
                        (cursor.length(size_len, "chunk size")?, cursor.u32()?)
                    } else {
                        (chunk_bytes, 0)
                    };
                    let mut scaled = Vec::with_capacity(rank);
                    for _ in 0..rank {
                        scaled.push(cursor.u64()?);
                    }
                    out.push(Chunk {
                        address,
                        size,
                        filter_mask,
                        scaled,
                    });
                    Ok(())
                },
            )?;
            Ok(out)
        }
    }
}

/// A fixed or extensible array element for a chunk: address, then for
/// filtered chunks the stored size (entry size - O - 4 bytes) and mask.
fn chunk_from_entry(
    entry: &[u8],
    offset_size: usize,
    chunk_bytes: usize,
    scaled: Vec<u64>,
) -> Result<Chunk> {
    let mut cursor = Cursor::new(entry, 0);
    let address = cursor.addr(offset_size)?;
    let (size, filter_mask) = if entry.len() > offset_size {
        let size_len = entry
            .len()
            .checked_sub(offset_size + 4)
            .filter(|len| (1..=8).contains(len))
            .ok_or_else(|| invalid(0, "filtered chunk entry size"))?;
        (cursor.length(size_len, "chunk size")?, cursor.u32()?)
    } else {
        (chunk_bytes, 0)
    };
    Ok(Chunk {
        address,
        size,
        filter_mask,
        scaled,
    })
}

fn verify(
    space: &Space<'_>,
    bytes: &[u8],
    covered: usize,
    structure: &'static str,
    offset: usize,
) -> Result<()> {
    let stored = u32::from_le_bytes(
        *bytes
            .get(covered..)
            .and_then(|tail| tail.first_chunk::<4>())
            .ok_or_else(|| invalid(offset, format!("{structure} checksum missing")))?,
    );
    let checked = bytes
        .get(..covered)
        .ok_or_else(|| invalid(offset, format!("{structure} checksum missing")))?;
    space.check(structure, offset, stored, || lookup3(checked))
}

/// `true` when bit `index` of an MSB-first bitmap is set (`H5VM_bit_get`).
fn bit(bitmap: &[u8], index: usize) -> bool {
    bitmap
        .get(index / 8)
        .is_some_and(|byte| byte & (0x80 >> (index % 8)) != 0)
}

/// Element bytes, or `None` for an unallocated chunk.
fn element(entries: &[u8], index: usize, size: usize, offset_size: usize) -> Result<Option<&[u8]>> {
    let start = index
        .checked_mul(size)
        .ok_or_else(|| invalid(0, "array element offset overflow"))?;
    let entry = entries
        .get(start..start + size)
        .ok_or_else(|| invalid(0, "array element out of range"))?;
    let address = crate::bytes::normalize_addr(
        crate::bytes::le_uint(&entry[..offset_size.min(entry.len())]),
        offset_size,
    );
    Ok((address != UNDEFINED_ADDR).then_some(entry))
}

/// Fixed array ("FAHD" header, "FADB" data block, optional pages).
struct FixedArray<'a> {
    entry_size: usize,
    page_elements: usize,
    total: usize,
    /// Unpaged entries, or the page bitmap and page area for paged arrays.
    entries: &'a [u8],
    page_bitmap: &'a [u8],
    pages_start: u64,
    paged: bool,
    /// Pages whose checksum already matched.
    verified_pages: HashSet<usize>,
}

impl<'a> FixedArray<'a> {
    /// Header: signature, version (0), client ID (0 plain / 1 filtered
    /// chunks), entry size (u8), page bits (u8), maximum entries (L), data
    /// block address (O), checksum. Data block: signature, version, client
    /// ID, header address (O), [page bitmap], [entries], checksum; pages of
    /// `2^page_bits` entries plus a checksum follow a paged data block.
    fn parse(space: &Space<'a>, address: u64) -> Result<Self> {
        space.expect_signature(address, b"FAHD")?;
        let start = space.abs(address)?;
        let mut cursor = space.cursor(address)?;
        cursor.skip(6)?;
        let entry_size = usize::from(cursor.u8()?);
        let page_bits = cursor.u8()?;
        let total = cursor.length(space.length_size, "fixed array entries")?;
        let block = cursor.addr(space.offset_size)?;
        let covered = cursor.pos();
        verify(
            space,
            space.slice(address, covered + 4)?,
            covered,
            "fixed array header",
            start,
        )?;
        if total > MAX_DATA_CHUNKS {
            return Err(limit(format!(
                "fixed array of {total} entries (limit {MAX_DATA_CHUNKS})"
            )));
        }
        if entry_size < space.offset_size || page_bits > 30 {
            return Err(invalid(
                start,
                "fixed array entry size or page bits invalid",
            ));
        }
        space.expect_signature(block, b"FADB")?;
        let block_start = space.abs(block)?;
        let page_elements = 1usize << page_bits;
        let paged = total > page_elements;
        let prefix = 6 + space.offset_size;
        if paged {
            let pages = total.div_ceil(page_elements);
            let bitmap_len = pages.div_ceil(8);
            let bytes = space.slice(block, prefix + bitmap_len + 4)?;
            verify(
                space,
                bytes,
                prefix + bitmap_len,
                "fixed array data block",
                block_start,
            )?;
            Ok(Self {
                entry_size,
                page_elements,
                total,
                entries: &[],
                page_bitmap: &bytes[prefix..prefix + bitmap_len],
                pages_start: block + (prefix + bitmap_len + 4) as u64,
                paged,
                verified_pages: HashSet::new(),
            })
        } else {
            let entries_len = total * entry_size;
            let bytes = space.slice(block, prefix + entries_len + 4)?;
            verify(
                space,
                bytes,
                prefix + entries_len,
                "fixed array data block",
                block_start,
            )?;
            Ok(Self {
                entry_size,
                page_elements,
                total,
                entries: &bytes[prefix..prefix + entries_len],
                page_bitmap: &[],
                pages_start: 0,
                paged,
                verified_pages: HashSet::new(),
            })
        }
    }

    fn entry(&mut self, space: &Space<'a>, index: u64) -> Result<Option<&'a [u8]>> {
        let index = to_usize(index, 0, "fixed array index")?;
        if index >= self.total {
            return Err(invalid(
                0,
                format!("fixed array index {index} >= {}", self.total),
            ));
        }
        if !self.paged {
            return element(self.entries, index, self.entry_size, space.offset_size);
        }
        let page = index / self.page_elements;
        if !bit(self.page_bitmap, page) {
            return Ok(None);
        }
        let overflow = || invalid(0, "fixed array page address overflow");
        let full_page = self
            .page_elements
            .checked_mul(self.entry_size)
            .and_then(|bytes| bytes.checked_add(4))
            .ok_or_else(overflow)?;
        let elements = (self.total - page * self.page_elements).min(self.page_elements);
        let page_address = (page as u64)
            .checked_mul(full_page as u64)
            .and_then(|offset| self.pages_start.checked_add(offset))
            .ok_or_else(overflow)?;
        let bytes = space.slice(page_address, elements * self.entry_size + 4)?;
        // Each page's checksum is verified once, not once per chunk.
        if self.verified_pages.insert(page) {
            verify(
                space,
                bytes,
                elements * self.entry_size,
                "fixed array page",
                space.abs(page_address)?,
            )?;
        }
        element(
            &bytes[..elements * self.entry_size],
            index % self.page_elements,
            self.entry_size,
            space.offset_size,
        )
    }
}

/// Per-super-block layout of an extensible array (`H5EA__hdr_init`).
#[derive(Clone, Copy)]
struct SuperInfo {
    data_blocks: usize,
    block_elements: usize,
    start_index: u64,
    start_block: usize,
}

/// Extensible array ("EAHD" header, "EAIB" index block, "EASB" secondary
/// blocks, "EADB" data blocks and their pages).
struct ExtensibleArray<'a> {
    element_size: usize,
    index_elements: usize,
    data_min_elements: usize,
    page_elements: usize,
    array_offset_size: usize,
    supers: Vec<SuperInfo>,
    index_supers: usize,
    index_block_entries: &'a [u8],
    data_block_addresses: Vec<u64>,
    super_block_addresses: Vec<u64>,
    /// Parsed secondary blocks: address to (page bitmap, data block
    /// addresses).
    secondary: HashMap<u64, (&'a [u8], Vec<u64>)>,
    /// Data blocks and data block pages whose checksum already matched.
    verified: HashSet<u64>,
}

impl<'a> ExtensibleArray<'a> {
    /// Header: signature, version (0), client ID, element size (u8),
    /// maximum element count bits (u8), index block elements (u8), data
    /// block minimum elements (u8), secondary block minimum data block
    /// pointers (u8), data block page element bits (u8), six statistics
    /// (L each), index block address (O), checksum.
    fn parse(space: &Space<'a>, address: u64) -> Result<Self> {
        space.expect_signature(address, b"EAHD")?;
        let start = space.abs(address)?;
        let mut cursor = space.cursor(address)?;
        cursor.skip(6)?;
        let element_size = usize::from(cursor.u8()?);
        let max_bits = u32::from(cursor.u8()?);
        let index_elements = usize::from(cursor.u8()?);
        let data_min_elements = usize::from(cursor.u8()?);
        let min_pointers = usize::from(cursor.u8()?);
        let page_bits = u32::from(cursor.u8()?);
        for _ in 0..6 {
            cursor.uint(space.length_size)?;
        }
        let index_block = cursor.addr(space.offset_size)?;
        let covered = cursor.pos();
        verify(
            space,
            space.slice(address, covered + 4)?,
            covered,
            "extensible array header",
            start,
        )?;
        let valid = element_size >= space.offset_size
            && (1..=64).contains(&max_bits)
            && data_min_elements.is_power_of_two()
            && min_pointers.is_power_of_two()
            && page_bits <= 30
            && log2_floor(data_min_elements as u64) < max_bits;
        if !valid {
            return Err(invalid(start, "extensible array parameters are invalid"));
        }
        let super_count = 1 + (max_bits - log2_floor(data_min_elements as u64)) as usize;
        let mut supers = Vec::with_capacity(super_count);
        let (mut start_index, mut start_block) = (0u64, 0usize);
        for index in 0..super_count {
            let data_blocks = 1usize
                .checked_shl((index / 2) as u32)
                .ok_or_else(|| invalid(start, "extensible array super block overflow"))?;
            let block_elements = 1usize
                .checked_shl(index.div_ceil(2) as u32)
                .and_then(|count| count.checked_mul(data_min_elements))
                .ok_or_else(|| invalid(start, "extensible array super block overflow"))?;
            supers.push(SuperInfo {
                data_blocks,
                block_elements,
                start_index,
                start_block,
            });
            start_index = start_index
                .saturating_add((data_blocks as u64).saturating_mul(block_elements as u64));
            start_block = start_block.saturating_add(data_blocks);
        }
        let index_supers = 2 * log2_floor(min_pointers as u64) as usize;
        let data_pointers = 2 * (min_pointers - 1);
        let super_pointers = super_count.saturating_sub(index_supers);
        let mut array = Self {
            element_size,
            index_elements,
            data_min_elements,
            page_elements: 1usize << page_bits,
            array_offset_size: (max_bits as usize).div_ceil(8),
            supers,
            index_supers,
            index_block_entries: &[],
            data_block_addresses: Vec::new(),
            super_block_addresses: Vec::new(),
            secondary: HashMap::new(),
            verified: HashSet::new(),
        };
        if index_block == UNDEFINED_ADDR {
            return Ok(array);
        }
        // Index block: signature, version, client ID, header address (O),
        // elements, data block addresses, secondary block addresses,
        // checksum.
        space.expect_signature(index_block, b"EAIB")?;
        let block_start = space.abs(index_block)?;
        let prefix = 6 + space.offset_size;
        let entries_len = index_elements * element_size;
        let content = prefix + entries_len + (data_pointers + super_pointers) * space.offset_size;
        let bytes = space.slice(index_block, content + 4)?;
        verify(
            space,
            bytes,
            content,
            "extensible array index block",
            block_start,
        )?;
        array.index_block_entries = &bytes[prefix..prefix + entries_len];
        let mut cursor = Cursor::new(&bytes[prefix + entries_len..content], block_start);
        for _ in 0..data_pointers {
            array
                .data_block_addresses
                .push(cursor.addr(space.offset_size)?);
        }
        for _ in 0..super_pointers {
            array
                .super_block_addresses
                .push(cursor.addr(space.offset_size)?);
        }
        Ok(array)
    }

    fn entry(&mut self, space: &Space<'a>, index: u64) -> Result<Option<&'a [u8]>> {
        let offset_size = space.offset_size;
        if index < self.index_elements as u64 {
            return element(
                self.index_block_entries,
                index as usize,
                self.element_size,
                offset_size,
            );
        }
        let relative = index - self.index_elements as u64;
        // H5EA__dblock_sblk_idx
        let super_index = log2_floor(relative / self.data_min_elements as u64 + 1) as usize;
        let info = *self.supers.get(super_index).ok_or_else(|| {
            invalid(
                0,
                format!("extensible array index {index} beyond its super blocks"),
            )
        })?;
        let within = to_usize(relative - info.start_index, 0, "extensible array offset")?;
        let block_elements = info.block_elements;
        let paged = block_elements > self.page_elements;
        if super_index < self.index_supers {
            let block_index = info.start_block + within / block_elements;
            let Some(&address) = self.data_block_addresses.get(block_index) else {
                return Err(invalid(0, "extensible array data block index out of range"));
            };
            if address == UNDEFINED_ADDR {
                return Ok(None);
            }
            return self.data_block_element(
                space,
                address,
                block_elements,
                within % block_elements,
                paged,
                None,
            );
        }
        let super_offset = super_index - self.index_supers;
        let Some(&super_address) = self.super_block_addresses.get(super_offset) else {
            return Err(invalid(
                0,
                "extensible array super block index out of range",
            ));
        };
        if super_address == UNDEFINED_ADDR {
            return Ok(None);
        }
        let pages_per_block = if paged {
            block_elements / self.page_elements
        } else {
            0
        };
        if !self.secondary.contains_key(&super_address) {
            // Secondary block: signature, version, client ID, header
            // address (O), block offset, [page bitmaps], data block
            // addresses, checksum. Each data block has its own page-init
            // bitmap of whole bytes (H5EA__sblock_alloc: `ndblks *
            // ((dblk_npages + 7) / 8)` bytes).
            space.expect_signature(super_address, b"EASB")?;
            let start = space.abs(super_address)?;
            let prefix = 6 + offset_size + self.array_offset_size;
            let overflow = || invalid(start, "extensible array secondary block size overflow");
            let bitmap_len = if paged {
                info.data_blocks
                    .checked_mul(pages_per_block.div_ceil(8))
                    .ok_or_else(overflow)?
            } else {
                0
            };
            let content = info
                .data_blocks
                .checked_mul(offset_size)
                .and_then(|bytes| bytes.checked_add(prefix + bitmap_len))
                .ok_or_else(overflow)?;
            let bytes = space.slice(super_address, content + 4)?;
            verify(
                space,
                bytes,
                content,
                "extensible array secondary block",
                start,
            )?;
            let mut cursor = Cursor::new(&bytes[prefix + bitmap_len..content], start);
            let mut addresses = Vec::with_capacity(info.data_blocks);
            for _ in 0..info.data_blocks {
                addresses.push(cursor.addr(offset_size)?);
            }
            self.secondary.insert(
                super_address,
                (&bytes[prefix..prefix + bitmap_len], addresses),
            );
        }
        let (bitmap, addresses) = self
            .secondary
            .get(&super_address)
            .ok_or_else(|| invalid(0, "extensible array cache"))?;
        let block_index = within / block_elements;
        let Some(&address) = addresses.get(block_index) else {
            return Err(invalid(0, "extensible array data block index out of range"));
        };
        if address == UNDEFINED_ADDR {
            return Ok(None);
        }
        let within_block = within % block_elements;
        // H5EA__lookup_elmt: bit `dblk_idx * dblk_npages + page_idx` of
        // the super block's bitmaps (MSB first).
        let page_init = paged.then(|| {
            bit(
                bitmap,
                block_index * pages_per_block + within_block / self.page_elements,
            )
        });
        self.data_block_element(
            space,
            address,
            block_elements,
            within_block,
            paged,
            page_init,
        )
    }

    /// Data block: signature, version, client ID, header address (O), block
    /// offset, [elements], checksum; pages of elements plus a checksum
    /// follow a paged block.
    fn data_block_element(
        &mut self,
        space: &Space<'a>,
        address: u64,
        block_elements: usize,
        within: usize,
        paged: bool,
        page_initialized: Option<bool>,
    ) -> Result<Option<&'a [u8]>> {
        space.expect_signature(address, b"EADB")?;
        let start = space.abs(address)?;
        let prefix = 6 + space.offset_size + self.array_offset_size;
        let overflow = || invalid(start, "extensible array data block size overflow");
        if !paged {
            let entries_len = block_elements
                .checked_mul(self.element_size)
                .filter(|len| *len <= MAX_DATASET_BYTES)
                .ok_or_else(overflow)?;
            let bytes = space.slice(address, prefix + entries_len + 4)?;
            // Each block's checksum is verified once, not once per chunk.
            if self.verified.insert(address) {
                verify(
                    space,
                    bytes,
                    prefix + entries_len,
                    "extensible array data block",
                    start,
                )?;
            }
            return element(
                &bytes[prefix..prefix + entries_len],
                within,
                self.element_size,
                space.offset_size,
            );
        }
        if page_initialized == Some(false) {
            return Ok(None);
        }
        let page = within / self.page_elements;
        let page_len = self.page_elements * self.element_size;
        let page_address = (page as u64)
            .checked_mul(page_len as u64 + 4)
            .and_then(|offset| offset.checked_add((prefix + 4) as u64))
            .and_then(|offset| address.checked_add(offset))
            .ok_or_else(overflow)?;
        let bytes = space.slice(page_address, page_len + 4)?;
        if self.verified.insert(page_address) {
            verify(
                space,
                bytes,
                page_len,
                "extensible array data block page",
                space.abs(page_address)?,
            )?;
        }
        element(
            &bytes[..page_len],
            within % self.page_elements,
            self.element_size,
            space.offset_size,
        )
    }
}

/// Read every chunk into a dataset buffer pre-filled with the fill value.
pub(crate) fn assemble(
    space: &Space<'_>,
    chunks: &[Chunk],
    geometry: &Geometry<'_>,
    filters: &[Filter],
    partial_unfiltered: bool,
    out: &mut [u8],
) -> Result<()> {
    let chunk_bytes = geometry.chunk_bytes()?;
    let rank = geometry.chunk_dims.len();
    if geometry.dims.len() != rank {
        return Err(invalid(
            0,
            "HDF5 chunk dimensionality does not match dataset rank",
        ));
    }
    let dims: Vec<usize> = geometry.dims.iter().map(|dim| *dim as usize).collect();
    let counts = geometry.chunk_counts();
    for chunk in chunks {
        if chunk.scaled.len() != rank
            || chunk
                .scaled
                .iter()
                .zip(&counts)
                .any(|(scaled, count)| scaled >= count)
        {
            continue; // outside the current extent
        }
        if chunk.size > MAX_DATASET_BYTES {
            return Err(limit(format!(
                "HDF5 stored chunk is {} bytes (limit {MAX_DATASET_BYTES})",
                chunk.size
            )));
        }
        let stored = space.slice(chunk.address, chunk.size)?;
        let partial = chunk
            .scaled
            .iter()
            .zip(geometry.chunk_dims)
            .zip(&dims)
            .any(|((scaled, chunk_dim), dim)| (*scaled as usize + 1) * chunk_dim > *dim);
        let decoded;
        let raw: &[u8] = if filters.is_empty() || (partial_unfiltered && partial) {
            stored
        } else {
            decoded = filters::apply_inverse(
                stored,
                filters,
                chunk.filter_mask,
                geometry.element_size,
                chunk_bytes,
            )?;
            &decoded
        };
        if raw.len() < chunk_bytes {
            return Err(invalid(
                space.abs(chunk.address).unwrap_or(0),
                "decoded chunk shorter than chunk dimensions",
            ));
        }
        let origin: Vec<usize> = chunk
            .scaled
            .iter()
            .zip(geometry.chunk_dims)
            .map(|(scaled, dim)| *scaled as usize * dim)
            .collect();
        copy_chunk(
            out,
            &raw[..chunk_bytes],
            &dims,
            geometry.chunk_dims,
            &origin,
            geometry.element_size,
        );
    }
    Ok(())
}

/// Copy one decoded chunk into the dataset buffer, clipping edge chunks.
fn copy_chunk(
    out: &mut [u8],
    chunk: &[u8],
    dims: &[usize],
    chunk_dims: &[usize],
    origin: &[usize],
    element_size: usize,
) {
    let rank = dims.len();
    if rank == 0 {
        let len = element_size.min(chunk.len()).min(out.len());
        out[..len].copy_from_slice(&chunk[..len]);
        return;
    }
    let last = rank - 1;
    let columns = chunk_dims[last].min(dims[last].saturating_sub(origin[last]));
    if columns == 0 {
        return;
    }
    let row_bytes = columns * element_size;
    // Output strides (elements) per dimension.
    let mut strides = vec![1usize; rank];
    for dim in (0..last).rev() {
        strides[dim] = strides[dim + 1] * dims[dim + 1];
    }
    // Rows of the chunk that land inside the dataset, per leading dimension.
    let extents: Vec<usize> = (0..last)
        .map(|dim| chunk_dims[dim].min(dims[dim].saturating_sub(origin[dim])))
        .collect();
    if extents.contains(&0) {
        return;
    }
    let mut index = vec![0usize; last];
    loop {
        // Source row: row-major position inside the chunk.
        let mut source_row = 0usize;
        let mut target = origin[last];
        for dim in 0..last {
            source_row = source_row * chunk_dims[dim] + index[dim];
            target += (origin[dim] + index[dim]) * strides[dim];
        }
        let source = source_row * chunk_dims[last] * element_size;
        let target = target * element_size;
        if let (Some(from), Some(to)) = (
            chunk.get(source..source + row_bytes),
            out.get_mut(target..target + row_bytes),
        ) {
            to.copy_from_slice(from);
        }
        // Next row.
        let mut dim = last;
        loop {
            if dim == 0 {
                return;
            }
            dim -= 1;
            index[dim] += 1;
            if index[dim] < extents[dim] {
                break;
            }
            index[dim] = 0;
        }
    }
}
