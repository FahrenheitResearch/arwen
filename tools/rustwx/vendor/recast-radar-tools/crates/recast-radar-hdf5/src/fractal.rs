//! Fractal heaps (HDF5 File Format Specification, section III.G): the
//! storage behind dense links and dense attributes.
//!
//! Objects are addressed by heap IDs. Managed objects live in direct blocks
//! reached through a doubling table of indirect blocks; huge objects live
//! outside the heap (directly addressed, or through a v2 B-tree keyed by an
//! ID); tiny objects are stored inside the heap ID itself.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::btree2;
use crate::bytes::{Cursor, UNDEFINED_ADDR, le_uint, limit_enc_size, log2_floor, to_usize};
use crate::checksum::lookup3;
use crate::error::{Result, invalid, limit};
use crate::filters::{self, Filter};
use crate::limits::{MAX_FILE_HEAP_BYTES, MAX_HEAP_BLOCK_BYTES, MAX_HEAP_DEPTH};
use crate::space::Space;

/// Add `bytes` of block checksumming, inflation or index reading to a
/// file's running total, failing past [`MAX_FILE_HEAP_BYTES`].
pub(crate) fn charge(work: &mut usize, bytes: usize) -> Result<()> {
    *work = work.saturating_add(bytes);
    if *work > MAX_FILE_HEAP_BYTES {
        return Err(limit(format!(
            "fractal heaps exceed {MAX_FILE_HEAP_BYTES} checksummed, inflated or indexed bytes per file (limit)"
        )));
    }
    Ok(())
}

/// Where a huge object is stored: address, stored length and, for filtered
/// heaps, the filter mask and unfiltered size.
type HugeObject = (u64, usize, Option<(u32, usize)>);

/// A parsed fractal heap header ("FRHP").
pub(crate) struct FractalHeap {
    /// Blocks whose checksum already matched (direct and indirect).
    verified: RefCell<HashSet<u64>>,
    /// Filtered direct blocks already inflated, by address.
    inflated: RefCell<HashMap<u64, Vec<u8>>>,
    /// The huge-object B-tree by huge object ID, read on first use.
    huge_index: RefCell<Option<HashMap<u64, HugeObject>>>,
    heap_id_len: usize,
    checksum_direct: bool,
    huge_btree: u64,
    table_width: u64,
    start_block_size: u64,
    root: u64,
    root_rows: usize,
    filtered_root_size: Option<(usize, u32)>,
    filters: Vec<Filter>,
    /// Bytes of a heap offset, `ceil(max heap size bits / 8)`.
    heap_off_size: usize,
    /// Bytes of a managed object length inside a heap ID.
    heap_len_size: usize,
    max_direct_rows: usize,
    first_row_bits: u32,
    offset_size: usize,
    length_size: usize,
}

impl FractalHeap {
    /// Header layout: signature, version (0), heap ID length (u16), I/O
    /// filter encoded length (u16), flags (u8), maximum managed object size
    /// (u32), next huge ID (L), huge-object v2 B-tree (O), free space (L),
    /// free-space manager (O), managed space (L), allocated managed space
    /// (L), direct-block allocation iterator offset (L), managed object
    /// count (L), huge object size (L), huge object count (L), tiny object
    /// size (L), tiny object count (L), table width (u16), starting block
    /// size (L), maximum direct block size (L), maximum heap size in bits
    /// (u16), starting root rows (u16), root block (O), current root rows
    /// (u16), [filtered root direct block size (L), filter mask (u32),
    /// filter pipeline], checksum.
    pub(crate) fn parse(space: &Space<'_>, address: u64) -> Result<Self> {
        space.expect_signature(address, b"FRHP")?;
        let start = space.abs(address)?;
        let mut cursor = space.cursor(address)?;
        cursor.skip(4)?;
        let version = cursor.u8()?;
        if version != 0 {
            return Err(invalid(
                start,
                format!("fractal heap version {version} unsupported"),
            ));
        }
        let (o, l) = (space.offset_size, space.length_size);
        let heap_id_len = usize::from(cursor.u16()?);
        let filter_len = usize::from(cursor.u16()?);
        let flags = cursor.u8()?;
        let max_managed = u64::from(cursor.u32()?);
        let _next_huge_id = cursor.uint(l)?;
        let huge_btree = cursor.addr(o)?;
        let _free_space = cursor.uint(l)?;
        let _free_space_manager = cursor.addr(o)?;
        let _managed_space = cursor.uint(l)?;
        let _allocated_space = cursor.uint(l)?;
        let _iterator_offset = cursor.uint(l)?;
        let _managed_objects = cursor.uint(l)?;
        let _huge_size = cursor.uint(l)?;
        let _huge_count = cursor.uint(l)?;
        let _tiny_size = cursor.uint(l)?;
        let _tiny_count = cursor.uint(l)?;
        let table_width = u64::from(cursor.u16()?);
        let start_block_size = cursor.uint(l)?;
        let max_direct_block_size = cursor.uint(l)?;
        let max_heap_bits = cursor.u16()?;
        let _start_root_rows = cursor.u16()?;
        let root = cursor.addr(o)?;
        let root_rows = usize::from(cursor.u16()?);
        let (filtered_root_size, filters) = if filter_len > 0 {
            let size = cursor.length(l, "fractal heap filtered root size")?;
            let mask = cursor.u32()?;
            let pipeline = cursor.take(filter_len)?;
            let filters = filters::parse(pipeline, start + cursor.pos() - filter_len)?;
            (Some((size, mask)), filters)
        } else {
            (None, Vec::new())
        };
        let covered = cursor.pos();
        let stored = cursor.u32()?;
        let checked = space.slice(address, covered)?;
        space.check("fractal heap header", start, stored, || lookup3(checked))?;
        let power_of_two = |value: u64| value != 0 && value.is_power_of_two();
        if !power_of_two(table_width)
            || !power_of_two(start_block_size)
            || !power_of_two(max_direct_block_size)
            || max_direct_block_size < start_block_size
            || max_heap_bits == 0
            || max_heap_bits > 64
            || heap_id_len == 0
        {
            return Err(invalid(
                start,
                "fractal heap doubling table parameters are invalid",
            ));
        }
        if max_direct_block_size > MAX_HEAP_BLOCK_BYTES as u64 {
            return Err(limit(format!(
                "fractal heap direct blocks of {max_direct_block_size} bytes (limit {MAX_HEAP_BLOCK_BYTES})"
            )));
        }
        let max_direct_rows =
            (log2_floor(max_direct_block_size) - log2_floor(start_block_size) + 2) as usize;
        let heap_off_size = usize::from(max_heap_bits).div_ceil(8);
        let direct_offset_size = (log2_floor(max_direct_block_size) as usize).div_ceil(8);
        let heap_len_size = direct_offset_size.min(limit_enc_size(max_managed));
        let _ = flags & 0x01; // huge IDs wrapped: only affects writers
        Ok(Self {
            verified: RefCell::new(HashSet::new()),
            inflated: RefCell::new(HashMap::new()),
            huge_index: RefCell::new(None),
            heap_id_len,
            checksum_direct: flags & 0x02 != 0,
            huge_btree,
            table_width,
            start_block_size,
            root,
            root_rows,
            filtered_root_size,
            filters,
            heap_off_size,
            heap_len_size,
            max_direct_rows,
            first_row_bits: log2_floor(start_block_size) + log2_floor(table_width),
            offset_size: o,
            length_size: l,
        })
    }

    /// Length of this heap's IDs.
    pub(crate) fn id_len(&self) -> usize {
        self.heap_id_len
    }

    /// Size of the blocks in doubling-table row `row`.
    fn row_block_size(&self, row: usize) -> Result<u64> {
        if row == 0 {
            return Ok(self.start_block_size);
        }
        let shift = u32::try_from(row - 1)
            .ok()
            .filter(|shift| *shift < 64)
            .ok_or_else(|| invalid(0, "fractal heap row size overflow"))?;
        self.start_block_size
            .checked_mul(1u64 << shift)
            .ok_or_else(|| invalid(0, "fractal heap row size overflow"))
    }

    /// Row and column of heap offset `offset` in a doubling table
    /// (`H5HF__dtable_lookup`).
    fn lookup(&self, offset: u64) -> Result<(usize, u64)> {
        let first_row_span = self.start_block_size * self.table_width;
        if offset < first_row_span {
            return Ok((0, offset / self.start_block_size));
        }
        let high_bit = log2_floor(offset);
        let row = (high_bit - self.first_row_bits + 1) as usize;
        let column = (offset - (1u64 << high_bit)) / self.row_block_size(row)?;
        Ok((row, column))
    }

    /// Heap offset of the first block in `row`.
    fn row_offset(&self, row: usize) -> Result<u64> {
        if row == 0 {
            return Ok(0);
        }
        let shift = u32::try_from(row - 1)
            .ok()
            .filter(|shift| *shift < 64)
            .ok_or_else(|| invalid(0, "fractal heap row offset overflow"))?;
        (self.start_block_size * self.table_width)
            .checked_mul(1u64 << shift)
            .ok_or_else(|| invalid(0, "fractal heap row offset overflow"))
    }

    /// The object a heap ID names. Block checksums, inflation and the
    /// huge-object index are charged to `work` (see [`charge`]).
    pub(crate) fn object<'a>(
        &self,
        space: &Space<'a>,
        id: &[u8],
        work: &mut usize,
    ) -> Result<Cow<'a, [u8]>> {
        let first = *id
            .first()
            .ok_or_else(|| invalid(0, "empty fractal heap ID"))?;
        if first >> 6 != 0 {
            return Err(invalid(
                0,
                format!("fractal heap ID version {}", first >> 6),
            ));
        }
        match (first >> 4) & 0x03 {
            0 => self.managed(space, id, work),
            1 => self.huge(space, id, work),
            2 => self.tiny(id),
            _ => Err(invalid(0, "fractal heap ID type 3 is reserved")),
        }
    }

    fn tiny<'a>(&self, id: &[u8]) -> Result<Cow<'a, [u8]>> {
        // H5HF__tiny_init: IDs longer than 18 bytes carry a 12-bit length.
        let (length, start) = if self.heap_id_len.saturating_sub(1) <= 17 {
            (usize::from(id[0] & 0x0F) + 1, 1)
        } else {
            let low = *id.get(1).ok_or_else(|| invalid(0, "short tiny heap ID"))?;
            ((usize::from(id[0] & 0x0F) << 8 | usize::from(low)) + 1, 2)
        };
        let data = id
            .get(start..start + length)
            .ok_or_else(|| invalid(0, "tiny fractal heap object overruns its ID"))?;
        Ok(Cow::Owned(data.to_vec()))
    }

    fn managed<'a>(&self, space: &Space<'a>, id: &[u8], work: &mut usize) -> Result<Cow<'a, [u8]>> {
        let mut cursor = Cursor::new(id, 0);
        cursor.skip(1)?;
        let offset = cursor.uint(self.heap_off_size)?;
        let length = cursor.length(self.heap_len_size, "managed object length")?;
        if length > MAX_HEAP_BLOCK_BYTES {
            return Err(limit(format!(
                "fractal heap object of {length} bytes (limit {MAX_HEAP_BLOCK_BYTES})"
            )));
        }
        if self.root == UNDEFINED_ADDR {
            return Err(invalid(
                0,
                "managed object in a fractal heap without a root block",
            ));
        }
        // Walk indirect blocks down to the direct block holding `offset`.
        let mut block = self.root;
        let mut block_rows = self.root_rows;
        let mut block_offset = 0u64;
        let mut block_size = self.start_block_size;
        let mut filtered = self.filtered_root_size;
        let mut depth = 0usize;
        while block_rows > 0 {
            depth += 1;
            if depth > MAX_HEAP_DEPTH {
                return Err(limit(format!(
                    "fractal heap deeper than {MAX_HEAP_DEPTH} indirect blocks (limit)"
                )));
            }
            let relative = offset
                .checked_sub(block_offset)
                .ok_or_else(|| invalid(0, "fractal heap offset precedes its block"))?;
            let (row, column) = self.lookup(relative)?;
            if row >= block_rows || column >= self.table_width {
                return Err(invalid(
                    0,
                    format!("fractal heap offset {offset} outside its table"),
                ));
            }
            let entry = self.indirect_entry(space, block, block_rows, row, column, work)?;
            let child_offset = block_offset
                .checked_add(self.row_offset(row)?)
                .and_then(|value| {
                    value.checked_add(column.checked_mul(self.row_block_size(row).ok()?)?)
                })
                .ok_or_else(|| invalid(0, "fractal heap block offset overflow"))?;
            if entry.0 == UNDEFINED_ADDR {
                return Err(invalid(
                    0,
                    format!("fractal heap object at offset {offset} is not allocated"),
                ));
            }
            block = entry.0;
            block_offset = child_offset;
            block_size = self.row_block_size(row)?;
            filtered = entry.1;
            block_rows = if row < self.max_direct_rows {
                0
            } else {
                // H5HF__man_dblock_locate: rows of a child indirect block.
                (log2_floor(block_size) - self.first_row_bits + 1) as usize
            };
        }
        let within = to_usize(offset - block_offset, 0, "fractal heap object offset")?;
        let end = within
            .checked_add(length)
            .ok_or_else(|| invalid(0, "fractal heap object range overflow"))?;
        let prefix =
            5 + self.offset_size + self.heap_off_size + if self.checksum_direct { 4 } else { 0 };
        let object = |direct: &[u8]| {
            if within < prefix || end > direct.len() {
                return Err(invalid(
                    0,
                    format!(
                        "fractal heap object [{within}, {end}) outside its {}-byte block",
                        direct.len()
                    ),
                ));
            }
            Ok(within..end)
        };
        match filtered {
            Some(filtered) if !self.filters.is_empty() => {
                if !self.inflated.borrow().contains_key(&block) {
                    let direct = self.inflate_direct(space, block, block_size, filtered, work)?;
                    self.inflated.borrow_mut().insert(block, direct);
                }
                let inflated = self.inflated.borrow();
                let direct = inflated
                    .get(&block)
                    .ok_or_else(|| invalid(0, "fractal heap direct block cache"))?;
                Ok(Cow::Owned(direct[object(direct)?].to_vec()))
            }
            _ => {
                let direct = self.direct_block(space, block, block_size, work)?;
                Ok(Cow::Borrowed(&direct[object(direct)?]))
            }
        }
    }

    /// Child entry (`row`, `column`) of the indirect block at `address`
    /// with `rows` rows: address and, for filtered heaps, the stored size
    /// and filter mask of a direct block.
    fn indirect_entry(
        &self,
        space: &Space<'_>,
        address: u64,
        rows: usize,
        row: usize,
        column: u64,
        work: &mut usize,
    ) -> Result<(u64, Option<(usize, u32)>)> {
        space.expect_signature(address, b"FHIB")?;
        let start = space.abs(address)?;
        let direct_rows = rows.min(self.max_direct_rows);
        let indirect_rows = rows - direct_rows;
        let width = to_usize(self.table_width, start, "fractal heap width")?;
        let direct_entry = self.offset_size
            + if self.filters.is_empty() {
                0
            } else {
                self.length_size + 4
            };
        let header = 5 + self.offset_size + self.heap_off_size;
        let entries_len = direct_rows
            .checked_mul(width)
            .and_then(|count| count.checked_mul(direct_entry))
            .and_then(|bytes| {
                indirect_rows
                    .checked_mul(width)?
                    .checked_mul(self.offset_size)?
                    .checked_add(bytes)
            })
            .ok_or_else(|| invalid(start, "fractal heap indirect block size overflow"))?;
        if entries_len > MAX_HEAP_BLOCK_BYTES {
            return Err(limit(format!(
                "fractal heap indirect block of {entries_len} bytes (limit {MAX_HEAP_BLOCK_BYTES})"
            )));
        }
        let block = space.slice(address, header + entries_len + 4)?;
        if space.verify_checksums && !self.verified.borrow().contains(&address) {
            charge(work, block.len())?;
            let stored = u32::from_le_bytes(
                *block[header + entries_len..]
                    .first_chunk::<4>()
                    .ok_or_else(|| invalid(start, "fractal heap indirect block checksum"))?,
            );
            space.check("fractal heap indirect block", start, stored, || {
                lookup3(&block[..header + entries_len])
            })?;
            self.verified.borrow_mut().insert(address);
        }
        let column = to_usize(column, start, "fractal heap column")?;
        let mut cursor = Cursor::new(&block[header..header + entries_len], start + header);
        if row < direct_rows {
            cursor.skip((row * width + column) * direct_entry)?;
            let child = cursor.addr(self.offset_size)?;
            let filtered = if self.filters.is_empty() {
                None
            } else {
                let size = cursor.length(self.length_size, "filtered direct block size")?;
                Some((size, cursor.u32()?))
            };
            Ok((child, filtered))
        } else {
            cursor.skip(direct_rows * width * direct_entry)?;
            cursor.skip(((row - direct_rows) * width + column) * self.offset_size)?;
            Ok((cursor.addr(self.offset_size)?, None))
        }
    }

    /// The bytes of an unfiltered direct block ("FHDB", version, heap
    /// header address, block offset, [checksum], objects), checksum-checked.
    fn direct_block<'a>(
        &self,
        space: &Space<'a>,
        address: u64,
        size: u64,
        work: &mut usize,
    ) -> Result<&'a [u8]> {
        let size = to_usize(size, 0, "fractal heap direct block size")?;
        let block = space.slice(address, size)?;
        self.check_direct(space, address, block, work)?;
        Ok(block)
    }

    /// A filtered direct block, inflated and checksum-checked.
    fn inflate_direct(
        &self,
        space: &Space<'_>,
        address: u64,
        size: u64,
        (stored_size, mask): (usize, u32),
        work: &mut usize,
    ) -> Result<Vec<u8>> {
        let size = to_usize(size, 0, "fractal heap direct block size")?;
        if stored_size > MAX_HEAP_BLOCK_BYTES {
            return Err(limit(format!(
                "filtered fractal heap block of {stored_size} bytes (limit {MAX_HEAP_BLOCK_BYTES})"
            )));
        }
        charge(work, stored_size.saturating_add(size))?;
        let stored = space.slice(address, stored_size)?;
        let mut raw = filters::apply_inverse(stored, &self.filters, mask, 1, size)?;
        raw.truncate(size);
        self.check_direct(space, address, &raw, work)?;
        Ok(raw)
    }

    /// Signature and (when the heap checksums direct blocks) checksum of a
    /// direct block, each block checked once.
    fn check_direct(
        &self,
        space: &Space<'_>,
        address: u64,
        block: &[u8],
        work: &mut usize,
    ) -> Result<()> {
        let start = space.abs(address)?;
        if block.get(..4) != Some(&b"FHDB"[..]) {
            return Err(invalid(start, "expected FHDB signature"));
        }
        if self.checksum_direct
            && space.verify_checksums
            && !self.verified.borrow().contains(&address)
        {
            charge(work, block.len())?;
            let at = 5 + self.offset_size + self.heap_off_size;
            let stored = u32::from_le_bytes(
                *block
                    .get(at..)
                    .and_then(|tail| tail.first_chunk::<4>())
                    .ok_or_else(|| invalid(start, "fractal heap direct block checksum"))?,
            );
            space.check("fractal heap direct block", start, stored, || {
                let mut zeroed = block.to_vec();
                zeroed[at..at + 4].fill(0);
                lookup3(&zeroed)
            })?;
            self.verified.borrow_mut().insert(address);
        }
        Ok(())
    }

    /// Huge objects: directly addressed when the ID holds the address and
    /// length (plus filter mask and size for filtered heaps), otherwise
    /// looked up by ID in the huge-object v2 B-tree (record type 1, or 2
    /// for filtered heaps), which is read once per heap.
    fn huge<'a>(&self, space: &Space<'a>, id: &[u8], work: &mut usize) -> Result<Cow<'a, [u8]>> {
        let (o, l) = (self.offset_size, self.length_size);
        let filtered = !self.filters.is_empty();
        let direct = if filtered {
            self.heap_id_len > o + l + 4 + l
        } else {
            self.heap_id_len > o + l
        };
        let (address, length, filter) = if direct {
            let mut cursor = Cursor::new(id, 0);
            cursor.skip(1)?;
            let address = cursor.addr(o)?;
            let length = cursor.length(l, "huge object length")?;
            let filter = if filtered {
                let mask = cursor.u32()?;
                let size = cursor.length(l, "huge object size")?;
                Some((mask, size))
            } else {
                None
            };
            (address, length, filter)
        } else {
            let key_len = (self.heap_id_len - 1).min(8);
            let key = le_uint(
                id.get(1..1 + key_len)
                    .ok_or_else(|| invalid(0, "short huge heap ID"))?,
            );
            if self.huge_index.borrow().is_none() {
                let index = self.read_huge_index(space, work)?;
                *self.huge_index.borrow_mut() = Some(index);
            }
            self.huge_index
                .borrow()
                .as_ref()
                .and_then(|index| index.get(&key).copied())
                .ok_or_else(|| invalid(0, format!("huge fractal heap object {key} not found")))?
        };
        if length > MAX_HEAP_BLOCK_BYTES {
            return Err(limit(format!(
                "huge fractal heap object of {length} bytes (limit {MAX_HEAP_BLOCK_BYTES})"
            )));
        }
        let stored = space.slice(address, length)?;
        match filter {
            None => Ok(Cow::Borrowed(stored)),
            Some((mask, size)) => {
                if size > MAX_HEAP_BLOCK_BYTES {
                    return Err(limit(format!(
                        "huge fractal heap object of {size} bytes (limit {MAX_HEAP_BLOCK_BYTES})"
                    )));
                }
                charge(work, length.saturating_add(size))?;
                let mut raw = filters::apply_inverse(stored, &self.filters, mask, 1, size)?;
                raw.truncate(size);
                Ok(Cow::Owned(raw))
            }
        }
    }

    /// Every record of the huge-object v2 B-tree by ID: address (O),
    /// length (L), [filter mask (u32), unfiltered size (L), for type 2], ID
    /// (L). The first record of an ID wins.
    fn read_huge_index(
        &self,
        space: &Space<'_>,
        work: &mut usize,
    ) -> Result<HashMap<u64, HugeObject>> {
        let (o, l) = (self.offset_size, self.length_size);
        let filtered = !self.filters.is_empty();
        if self.huge_btree == UNDEFINED_ADDR {
            return Err(invalid(
                0,
                "huge fractal heap object without a huge-object B-tree",
            ));
        }
        let (record_type, record_size) = if filtered {
            (2u8, o + l + 4 + l + l)
        } else {
            (1u8, o + l + l)
        };
        let mut index = HashMap::new();
        btree2::for_each_record(
            space,
            self.huge_btree,
            &[record_type],
            record_size,
            &mut |record| {
                charge(work, record.len())?;
                let mut cursor = Cursor::new(record, 0);
                let address = cursor.addr(o)?;
                let length = cursor.length(l, "huge object length")?;
                let (filter, id) = if filtered {
                    let mask = cursor.u32()?;
                    let size = cursor.length(l, "huge object size")?;
                    (Some((mask, size)), cursor.uint(l)?)
                } else {
                    (None, cursor.uint(l)?)
                };
                index.entry(id).or_insert((address, length, filter));
                Ok(())
            },
        )?;
        Ok(index)
    }
}
