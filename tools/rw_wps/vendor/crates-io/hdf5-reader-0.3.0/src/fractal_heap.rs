//! HDF5 Fractal Heap (FRHP).
//!
//! Fractal heaps are the storage mechanism for link messages and attribute
//! messages in new-style (v2) groups and datasets. They use a doubling-table
//! scheme with direct and indirect blocks. Objects are addressed by a
//! heap ID that encodes the offset and length within the heap.
//!
//! This module parses the heap header and provides managed-object extraction
//! from direct blocks. Huge and tiny object types are recognized but not
//! fully implemented yet.

use crate::checksum::jenkins_lookup3;
use crate::error::{Error, Result};
use crate::io::Cursor;
use crate::storage::Storage;

/// Signature bytes for a fractal heap header: ASCII `FRHP`.
const FRHP_SIGNATURE: [u8; 4] = *b"FRHP";

/// Signature bytes for a direct block: ASCII `FHDB`.
const _FHDB_SIGNATURE: [u8; 4] = *b"FHDB";

/// Signature bytes for an indirect block: ASCII `FHIB`.
const FHIB_SIGNATURE: [u8; 4] = *b"FHIB";

/// Parsed fractal heap header.
#[derive(Debug, Clone)]
pub struct FractalHeap {
    /// Size in bytes of heap IDs used to reference objects.
    pub heap_id_len: u16,
    /// Size in bytes of I/O filter info (0 if none).
    pub io_filters_len: u16,
    /// Maximum size of a managed object (larger objects become "huge").
    pub max_managed_object_size: u64,
    /// Next huge object ID to assign.
    pub next_huge_id: u64,
    /// Address of the B-tree v2 used for huge objects.
    pub btree_huge_objects_address: u64,
    /// Address of the free-space manager for managed blocks.
    pub free_space_managed_address: u64,
    /// Total managed space in bytes.
    pub managed_space_amount: u64,
    /// Total managed allocated space in bytes.
    pub managed_alloc_amount: u64,
    /// Iterator offset for managed free-space.
    pub managed_iter_offset: u64,
    /// Number of managed objects.
    pub managed_objects_count: u64,
    /// Total size of huge objects in bytes.
    pub huge_objects_size: u64,
    /// Number of huge objects.
    pub huge_objects_count: u64,
    /// Total size of tiny objects in bytes.
    pub tiny_objects_size: u64,
    /// Number of tiny objects.
    pub tiny_objects_count: u64,
    /// Width of the doubling table (number of direct blocks per row).
    pub table_width: u16,
    /// Size in bytes of the starting (smallest) direct block.
    pub starting_block_size: u64,
    /// Maximum direct block size before switching to indirect blocks.
    pub max_direct_block_size: u64,
    /// Log2 of the maximum managed heap size (used for heap ID encoding).
    pub max_heap_size: u16,
    /// Starting row of the root indirect block (for doubling table).
    pub starting_row_root_indirect: u16,
    /// Address of the root block (direct or indirect).
    pub root_block_address: u64,
    /// Current number of rows in the root indirect block.
    pub current_rows_in_root_indirect: u16,
    /// Filtered root direct block size (present only when io_filters_len > 0).
    pub io_filter_size: Option<u64>,
    /// Filter mask for root direct block (present only when io_filters_len > 0).
    pub io_filter_mask: Option<u32>,
}

impl FractalHeap {
    /// Parse a fractal heap header at the current cursor position.
    ///
    /// Format:
    /// - Signature: `FRHP` (4 bytes)
    /// - Version: 0 (1 byte)
    /// - Heap ID length (u16 LE)
    /// - I/O filters encoded length (u16 LE)
    /// - Flags (u8)
    /// - Max managed object size (u32 LE)
    /// - Next huge object ID (`length_size` bytes)
    /// - B-tree address for huge objects (`offset_size` bytes)
    /// - Free-space managed objects address (`length_size` bytes)
    /// - Managed space amount (`length_size` bytes)
    /// - Managed alloc amount (`length_size` bytes)
    /// - Managed free-space iterator offset (`length_size` bytes)
    /// - Managed objects count (`length_size` bytes)
    /// - Huge objects size (`length_size` bytes)
    /// - Huge objects count (`length_size` bytes)
    /// - Tiny objects size (`length_size` bytes)
    /// - Tiny objects count (`length_size` bytes)
    /// - Table width (u16 LE)
    /// - Starting block size (`length_size` bytes)
    /// - Maximum direct block size (`length_size` bytes)
    /// - Max heap size (u16 LE)
    /// - Starting row of root indirect block (u16 LE)
    /// - Root block address (`offset_size` bytes)
    /// - Current rows in root indirect block (u16 LE)
    /// - If io_filters_len > 0: filtered root direct block size (`length_size`), filter mask (u32 LE)
    /// - Checksum (u32 LE)
    pub fn parse(cursor: &mut Cursor, offset_size: u8, length_size: u8) -> Result<Self> {
        let start = cursor.position();

        let sig = cursor.read_bytes(4)?;
        if sig != FRHP_SIGNATURE {
            return Err(Error::InvalidFractalHeapSignature);
        }

        let version = cursor.read_u8()?;
        if version != 0 {
            return Err(Error::UnsupportedFractalHeapVersion(version));
        }

        let heap_id_len = cursor.read_u16_le()?;
        let io_filters_len = cursor.read_u16_le()?;
        let _flags = cursor.read_u8()?;

        let max_managed_object_size = cursor.read_u32_le()? as u64;
        let next_huge_id = cursor.read_length(length_size)?;
        let btree_huge_objects_address = cursor.read_offset(offset_size)?;
        let free_space_managed_address = cursor.read_length(length_size)?;
        let managed_space_amount = cursor.read_length(length_size)?;
        let managed_alloc_amount = cursor.read_length(length_size)?;
        let managed_iter_offset = cursor.read_length(length_size)?;
        let managed_objects_count = cursor.read_length(length_size)?;
        let huge_objects_size = cursor.read_length(length_size)?;
        let huge_objects_count = cursor.read_length(length_size)?;
        let tiny_objects_size = cursor.read_length(length_size)?;
        let tiny_objects_count = cursor.read_length(length_size)?;

        let table_width = cursor.read_u16_le()?;
        let starting_block_size = cursor.read_length(length_size)?;
        let max_direct_block_size = cursor.read_length(length_size)?;
        let max_heap_size = cursor.read_u16_le()?;
        let starting_row_root_indirect = cursor.read_u16_le()?;
        let root_block_address = cursor.read_offset(offset_size)?;
        let current_rows_in_root_indirect = cursor.read_u16_le()?;

        let (io_filter_size, io_filter_mask) = if io_filters_len > 0 {
            let size = cursor.read_length(length_size)?;
            let mask = cursor.read_u32_le()?;
            (Some(size), Some(mask))
        } else {
            (None, None)
        };

        // Verify checksum.
        let checksum_end = cursor.position();
        let stored_checksum = cursor.read_u32_le()?;
        let computed = jenkins_lookup3(&cursor.data()[start as usize..checksum_end as usize]);
        if computed != stored_checksum {
            return Err(Error::ChecksumMismatch {
                expected: stored_checksum,
                actual: computed,
            });
        }

        Ok(FractalHeap {
            heap_id_len,
            io_filters_len,
            max_managed_object_size,
            next_huge_id,
            btree_huge_objects_address,
            free_space_managed_address,
            managed_space_amount,
            managed_alloc_amount,
            managed_iter_offset,
            managed_objects_count,
            huge_objects_size,
            huge_objects_count,
            tiny_objects_size,
            tiny_objects_count,
            table_width,
            starting_block_size,
            max_direct_block_size,
            max_heap_size,
            starting_row_root_indirect,
            root_block_address,
            current_rows_in_root_indirect,
            io_filter_size,
            io_filter_mask,
        })
    }

    /// Parse a fractal heap header from random-access storage.
    pub fn parse_at_storage(
        storage: &dyn Storage,
        address: u64,
        offset_size: u8,
        length_size: u8,
    ) -> Result<Self> {
        let max_header_len = 256usize;
        let available = storage.len().saturating_sub(address);
        let len = usize::try_from(available.min(max_header_len as u64)).map_err(|_| {
            Error::InvalidData("fractal heap header exceeds platform usize capacity".into())
        })?;
        let bytes = storage.read_range(address, len)?;
        let mut cursor = Cursor::new(bytes.as_ref());
        Self::parse(&mut cursor, offset_size, length_size)
    }

    /// Extract a managed object given a heap ID.
    ///
    /// The heap ID for managed objects (type nibble = 0) encodes:
    /// - Bits for the version/type (first nibble: 0 = managed)
    /// - Offset within the heap (variable number of bits based on `max_heap_size`)
    /// - Length of the object
    ///
    /// This implementation handles the common case where the root block is a
    /// single direct block (i.e., `current_rows_in_root_indirect == 0`).
    /// Indirect block traversal is provided for single-level indirect blocks.
    pub fn get_managed_object(
        &self,
        heap_id: &[u8],
        file_data: &[u8],
        offset_size: u8,
        _length_size: u8,
    ) -> Result<Vec<u8>> {
        if heap_id.is_empty() {
            return Err(Error::InvalidData("empty fractal heap ID".into()));
        }

        // First nibble is the type: 0 = managed, 1 = tiny, 2 = huge.
        let id_type = (heap_id[0] >> 4) & 0x03;
        match id_type {
            0 => {} // managed — handled below
            1 => {
                return Err(Error::Other(
                    "fractal heap tiny objects not yet supported".to_string(),
                ));
            }
            2 => {
                return Err(Error::Other(
                    "fractal heap huge objects not yet supported".to_string(),
                ));
            }
            other => {
                return Err(Error::InvalidData(format!(
                    "unknown fractal heap ID type {}",
                    other
                )));
            }
        }

        // Decode the offset and length from the heap ID.
        // The offset uses `max_heap_size` bits, and the length uses the remaining
        // bits in the heap ID.
        let offset_bits = self.max_heap_size as usize;

        // Build a u64 from the heap ID bytes (skipping the type nibble).
        // The first 4 bits are the type/version nibble. The remaining bits
        // are: offset (offset_bits) then length.
        let total_bits = (heap_id.len() * 8) - 4; // minus 4 for type nibble
        let length_bits = total_bits.saturating_sub(offset_bits);

        // Extract bits from the heap ID.
        let (heap_offset, obj_length) = decode_managed_heap_id(heap_id, offset_bits, length_bits)?;

        if obj_length == 0 {
            return Ok(Vec::new());
        }

        // Find the direct block containing this offset.
        let (block_address, block_offset_in_heap, _block_size) =
            self.find_direct_block(heap_offset, file_data, offset_size)?;

        // Parse the direct block header to find where object data begins.
        let db_header_size = self.direct_block_header_size(offset_size);

        // The object's position within the direct block.
        let offset_in_block = heap_offset - block_offset_in_heap;
        let data_start = block_address as usize + db_header_size + offset_in_block as usize;
        let data_end = data_start + obj_length as usize;

        if data_end > file_data.len() {
            return Err(Error::UnexpectedEof {
                offset: data_start as u64,
                needed: obj_length,
                available: file_data.len().saturating_sub(data_start) as u64,
            });
        }

        Ok(file_data[data_start..data_end].to_vec())
    }

    /// Extract a managed object from random-access storage.
    pub fn get_managed_object_storage(
        &self,
        heap_id: &[u8],
        storage: &dyn Storage,
        offset_size: u8,
        _length_size: u8,
    ) -> Result<Vec<u8>> {
        if heap_id.is_empty() {
            return Err(Error::InvalidData("empty fractal heap ID".into()));
        }

        let id_type = (heap_id[0] >> 4) & 0x03;
        match id_type {
            0 => {}
            1 => {
                return Err(Error::Other(
                    "fractal heap tiny objects not yet supported".to_string(),
                ));
            }
            2 => {
                return Err(Error::Other(
                    "fractal heap huge objects not yet supported".to_string(),
                ));
            }
            other => {
                return Err(Error::InvalidData(format!(
                    "unknown fractal heap ID type {}",
                    other
                )));
            }
        }

        let offset_bits = self.max_heap_size as usize;
        let total_bits = (heap_id.len() * 8) - 4;
        let length_bits = total_bits.saturating_sub(offset_bits);
        let (heap_offset, obj_length) = decode_managed_heap_id(heap_id, offset_bits, length_bits)?;

        if obj_length == 0 {
            return Ok(Vec::new());
        }

        let (block_address, block_offset_in_heap, _block_size) =
            self.find_direct_block_storage(heap_offset, storage, offset_size)?;
        let db_header_size = self.direct_block_header_size(offset_size);
        let offset_in_block = heap_offset - block_offset_in_heap;
        let data_start = block_address
            .checked_add(u64::try_from(db_header_size).map_err(|_| {
                Error::InvalidData("fractal heap direct block header too large".into())
            })?)
            .and_then(|value| value.checked_add(offset_in_block))
            .ok_or(Error::OffsetOutOfBounds(block_address))?;
        let len = usize::try_from(obj_length).map_err(|_| {
            Error::InvalidData("fractal heap object exceeds platform usize capacity".into())
        })?;
        Ok(storage.read_range(data_start, len)?.to_vec())
    }

    /// Find the direct block containing a given heap offset.
    ///
    /// Returns (block_file_address, block_offset_within_heap, block_size).
    fn find_direct_block(
        &self,
        heap_offset: u64,
        file_data: &[u8],
        offset_size: u8,
    ) -> Result<(u64, u64, u64)> {
        if Cursor::is_undefined_offset(self.root_block_address, offset_size) {
            return Err(Error::UndefinedAddress);
        }

        if self.current_rows_in_root_indirect == 0 {
            // Root block is a direct block.
            // The entire managed space is in this one block.
            Ok((self.root_block_address, 0, self.starting_block_size))
        } else {
            // Root block is an indirect block — traverse the doubling table.
            self.find_direct_block_via_indirect(
                self.root_block_address,
                heap_offset,
                file_data,
                offset_size,
                self.current_rows_in_root_indirect,
            )
        }
    }

    fn find_direct_block_storage(
        &self,
        heap_offset: u64,
        storage: &dyn Storage,
        offset_size: u8,
    ) -> Result<(u64, u64, u64)> {
        if Cursor::is_undefined_offset(self.root_block_address, offset_size) {
            return Err(Error::UndefinedAddress);
        }

        if self.current_rows_in_root_indirect == 0 {
            Ok((self.root_block_address, 0, self.starting_block_size))
        } else {
            self.find_direct_block_via_indirect_storage(
                self.root_block_address,
                heap_offset,
                storage,
                offset_size,
                self.current_rows_in_root_indirect,
            )
        }
    }

    /// Traverse an indirect block to find the direct block for a given offset.
    fn find_direct_block_via_indirect(
        &self,
        indirect_address: u64,
        heap_offset: u64,
        file_data: &[u8],
        offset_size: u8,
        nrows: u16,
    ) -> Result<(u64, u64, u64)> {
        // Validate FHIB signature
        let addr = indirect_address as usize;
        if addr + 4 > file_data.len() {
            return Err(Error::OffsetOutOfBounds(indirect_address));
        }
        if file_data[addr..addr + 4] != FHIB_SIGNATURE {
            return Err(Error::InvalidData(format!(
                "expected FHIB signature at offset {:#x}, got {:?}",
                indirect_address,
                &file_data[addr..addr + 4]
            )));
        }

        // The doubling table has `table_width` entries per row.
        // Row 0 and 1 have blocks of size `starting_block_size`.
        // Row r (for r >= 1) has blocks of size `starting_block_size * 2^(r-1)`.
        //
        // We iterate through the rows to find which block contains the
        // target offset, then read the block address from the indirect block.

        let width = self.table_width as u64;
        let mut running_offset: u64 = 0;

        for row in 0..nrows as u64 {
            let block_size = self.block_size_for_row(row);
            let is_direct = block_size <= self.max_direct_block_size;

            for col in 0..width {
                let block_end = running_offset + block_size;
                if heap_offset >= running_offset && heap_offset < block_end {
                    // This is the block we want. Read its address from the
                    // indirect block.
                    let entry_index = row * width + col;

                    // Indirect block layout: signature(4) + version(1) +
                    // heap_header_addr(offset_size) + block_offset(max_heap_size/8 rounded up)
                    // Then entry_index * offset_size bytes to the address.
                    let iblock_header_size =
                        4 + 1 + offset_size as u64 + (self.max_heap_size as u64).div_ceil(8);
                    let entry_addr_pos =
                        indirect_address + iblock_header_size + entry_index * offset_size as u64;

                    if entry_addr_pos as usize + offset_size as usize > file_data.len() {
                        return Err(Error::OffsetOutOfBounds(entry_addr_pos));
                    }

                    let mut cursor = Cursor::new(file_data);
                    cursor.set_position(entry_addr_pos);
                    let block_address = cursor.read_offset(offset_size)?;

                    if Cursor::is_undefined_offset(block_address, offset_size) {
                        return Err(Error::UndefinedAddress);
                    }

                    if is_direct {
                        return Ok((block_address, running_offset, block_size));
                    } else {
                        // Need to recurse into a sub-indirect block.
                        // Determine how many rows the sub-indirect has.
                        let sub_rows = self.rows_for_block_size(block_size);
                        return self.find_direct_block_via_indirect(
                            block_address,
                            heap_offset - running_offset,
                            file_data,
                            offset_size,
                            sub_rows,
                        );
                    }
                }
                running_offset = block_end;
            }
        }

        Err(Error::InvalidData(format!(
            "fractal heap offset {} not found in doubling table",
            heap_offset
        )))
    }

    fn find_direct_block_via_indirect_storage(
        &self,
        indirect_address: u64,
        heap_offset: u64,
        storage: &dyn Storage,
        offset_size: u8,
        nrows: u16,
    ) -> Result<(u64, u64, u64)> {
        let sig = storage.read_range(indirect_address, 4)?;
        if sig.as_ref() != FHIB_SIGNATURE {
            return Err(Error::InvalidData(format!(
                "expected FHIB signature at offset {:#x}, got {:?}",
                indirect_address,
                sig.as_ref()
            )));
        }

        let width = self.table_width as u64;
        let mut running_offset = 0u64;

        for row in 0..u64::from(nrows) {
            let block_size = self.block_size_for_row(row);
            let is_direct = block_size <= self.max_direct_block_size;

            for col in 0..width {
                let block_end = running_offset + block_size;
                if heap_offset >= running_offset && heap_offset < block_end {
                    let entry_index = row * width + col;
                    let iblock_header_size = 4
                        + 1
                        + u64::from(offset_size)
                        + (u64::from(self.max_heap_size)).div_ceil(8);
                    let entry_addr_pos = indirect_address
                        + iblock_header_size
                        + entry_index * u64::from(offset_size);
                    let entry = storage.read_range(entry_addr_pos, usize::from(offset_size))?;
                    let mut cursor = Cursor::new(entry.as_ref());
                    let block_address = cursor.read_offset(offset_size)?;

                    if Cursor::is_undefined_offset(block_address, offset_size) {
                        return Err(Error::UndefinedAddress);
                    }

                    if is_direct {
                        return Ok((block_address, running_offset, block_size));
                    }

                    let sub_rows = self.rows_for_block_size(block_size);
                    return self.find_direct_block_via_indirect_storage(
                        block_address,
                        heap_offset - running_offset,
                        storage,
                        offset_size,
                        sub_rows,
                    );
                }
                running_offset = block_end;
            }
        }

        Err(Error::InvalidData(format!(
            "fractal heap offset {} not found in doubling table",
            heap_offset
        )))
    }

    /// Compute the block size for a given row in the doubling table.
    fn block_size_for_row(&self, row: u64) -> u64 {
        if row == 0 {
            self.starting_block_size
        } else {
            self.starting_block_size * (1u64 << (row - 1))
        }
    }

    /// Compute how many rows of the doubling table fit in a block of the
    /// given total size.
    fn rows_for_block_size(&self, total_size: u64) -> u16 {
        let mut rows = 0u16;
        let mut accum = 0u64;
        let width = self.table_width as u64;
        loop {
            let bs = self.block_size_for_row(rows as u64);
            let row_total = bs * width;
            if accum + row_total > total_size {
                break;
            }
            accum += row_total;
            rows += 1;
            if rows > 1000 {
                break; // safety
            }
        }
        rows
    }

    /// Size in bytes of a direct block header (including checksum when applicable).
    ///
    /// Per the HDF5 spec, direct blocks include a checksum when the heap has
    /// NO I/O filters. When I/O filters are present, the filters handle
    /// integrity and the checksum is omitted.
    fn direct_block_header_size(&self, offset_size: u8) -> usize {
        // Signature(4) + Version(1) + Heap header address(offset_size) +
        // Block offset within heap (max_heap_size bits, rounded up to bytes)
        let offset_bytes = (self.max_heap_size as usize).div_ceil(8);
        let base = 4 + 1 + offset_size as usize + offset_bytes;
        if self.io_filters_len == 0 {
            base + 4 // checksum present when no I/O filters
        } else {
            base
        }
    }
}

/// Decode the offset and length from a managed-object heap ID.
///
/// The first nibble (4 bits) is the type (already checked). The remaining
/// bits contain the offset (offset_bits wide) followed by the length.
fn decode_managed_heap_id(
    heap_id: &[u8],
    offset_bits: usize,
    length_bits: usize,
) -> Result<(u64, u64)> {
    // Convert heap_id to a bit stream, skipping the first 4 bits (type nibble).
    // We work with the bytes directly.
    let total_bits = offset_bits + length_bits;
    let available_bits = heap_id.len() * 8 - 4;
    if total_bits > available_bits {
        return Err(Error::InvalidData(format!(
            "fractal heap ID too short: need {} bits for offset+length, have {}",
            total_bits, available_bits
        )));
    }

    let offset = extract_bits(heap_id, 4, offset_bits);
    let length = extract_bits(heap_id, 4 + offset_bits, length_bits);

    Ok((offset, length))
}

/// Extract `num_bits` bits starting at bit position `start_bit` from a byte
/// slice (MSB-first bit ordering within each byte).
fn extract_bits(data: &[u8], start_bit: usize, num_bits: usize) -> u64 {
    if num_bits == 0 {
        return 0;
    }
    let mut value: u64 = 0;
    for i in 0..num_bits {
        let bit_pos = start_bit + i;
        let byte_idx = bit_pos / 8;
        let bit_idx = 7 - (bit_pos % 8); // MSB first
        if byte_idx < data.len() {
            let bit = (data[byte_idx] >> bit_idx) & 1;
            value = (value << 1) | bit as u64;
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_bits() {
        // 0xAB = 1010_1011
        let data = [0xAB];
        assert_eq!(extract_bits(&data, 0, 4), 0b1010); // 0xA
        assert_eq!(extract_bits(&data, 4, 4), 0b1011); // 0xB
        assert_eq!(extract_bits(&data, 0, 8), 0xAB);
    }

    #[test]
    fn test_extract_bits_cross_byte() {
        let data = [0xFF, 0x00];
        assert_eq!(extract_bits(&data, 4, 8), 0b1111_0000);
    }

    #[test]
    fn test_decode_managed_heap_id() {
        // Type nibble = 0 (managed), offset = 0, length = 100
        // offset_bits = 16, length_bits = 12
        // Total bits needed: 4 + 16 + 12 = 32 = 4 bytes
        // Byte layout:
        //   byte 0: type(4 bits=0000) + offset_hi(4 bits=0000)
        //   byte 1: offset_lo(8 bits=0000_0000)
        //   byte 2: offset_lo(4 bits=0000) + length_hi(4 bits=0000)
        //   byte 3: length_lo(8 bits=0110_0100) = 100
        // Wait, let's compute more carefully.
        // offset = 0 -> 16 bits all zero
        // length = 100 -> 12 bits: 0000_0110_0100
        // Full bit stream after type nibble: 0000_0000_0000_0000 | 0000_0110_0100
        // = 0x00 0x00 0x06 0x40 (but only 28 bits)
        // With type nibble 0: 0000 | 0000_0000_0000_0000 | 0000_0110_0100
        // Packed into bytes: 0000_0000 0000_0000 0000_0000 0110_0100
        //                    0x00       0x00       0x00       0x64
        let heap_id = [0x00, 0x00, 0x00, 0x64];
        let (offset, length) = decode_managed_heap_id(&heap_id, 16, 12).unwrap();
        assert_eq!(offset, 0);
        assert_eq!(length, 100);
    }

    #[test]
    fn test_block_size_for_row() {
        let heap = FractalHeap {
            heap_id_len: 8,
            io_filters_len: 0,
            max_managed_object_size: 0,
            next_huge_id: 0,
            btree_huge_objects_address: 0,
            free_space_managed_address: 0,
            managed_space_amount: 0,
            managed_alloc_amount: 0,
            managed_iter_offset: 0,
            managed_objects_count: 0,
            huge_objects_size: 0,
            huge_objects_count: 0,
            tiny_objects_size: 0,
            tiny_objects_count: 0,
            table_width: 4,
            starting_block_size: 256,
            max_direct_block_size: 4096,
            max_heap_size: 16,
            starting_row_root_indirect: 0,
            root_block_address: 0,
            current_rows_in_root_indirect: 0,
            io_filter_size: None,
            io_filter_mask: None,
        };

        assert_eq!(heap.block_size_for_row(0), 256);
        assert_eq!(heap.block_size_for_row(1), 256); // 256 * 2^0
        assert_eq!(heap.block_size_for_row(2), 512); // 256 * 2^1
        assert_eq!(heap.block_size_for_row(3), 1024); // 256 * 2^2
    }

    #[test]
    fn test_direct_block_header_size() {
        let heap = FractalHeap {
            heap_id_len: 8,
            io_filters_len: 0,
            max_managed_object_size: 0,
            next_huge_id: 0,
            btree_huge_objects_address: 0,
            free_space_managed_address: 0,
            managed_space_amount: 0,
            managed_alloc_amount: 0,
            managed_iter_offset: 0,
            managed_objects_count: 0,
            huge_objects_size: 0,
            huge_objects_count: 0,
            tiny_objects_size: 0,
            tiny_objects_count: 0,
            table_width: 4,
            starting_block_size: 256,
            max_direct_block_size: 4096,
            max_heap_size: 16,
            starting_row_root_indirect: 0,
            root_block_address: 0,
            current_rows_in_root_indirect: 0,
            io_filter_size: None,
            io_filter_mask: None,
        };

        // No I/O filters => checksum present.
        // sig(4) + ver(1) + addr(8) + offset_bytes(2) + checksum(4) = 19
        assert_eq!(heap.direct_block_header_size(8), 19);

        // With 4-byte offsets: sig(4) + ver(1) + addr(4) + offset_bytes(2) + checksum(4) = 15
        assert_eq!(heap.direct_block_header_size(4), 15);
    }

    #[test]
    fn test_get_managed_object_direct_root() {
        // Set up a fractal heap where the root is a direct block.
        let offset_size: u8 = 8;
        let max_heap_size: u16 = 16;
        let starting_block_size: u64 = 256;

        // Direct block header size: sig(4) + ver(1) + addr(8) + offset_bytes(2) + checksum(4) = 19
        // (no I/O filters => checksum present)
        let db_header_size = 19usize;

        // Place the direct block at file offset 1000.
        let block_address: u64 = 1000;

        let heap = FractalHeap {
            heap_id_len: 8,
            io_filters_len: 0,
            max_managed_object_size: 128,
            next_huge_id: 0,
            btree_huge_objects_address: u64::MAX,
            free_space_managed_address: 0,
            managed_space_amount: starting_block_size,
            managed_alloc_amount: starting_block_size,
            managed_iter_offset: 0,
            managed_objects_count: 1,
            huge_objects_size: 0,
            huge_objects_count: 0,
            tiny_objects_size: 0,
            tiny_objects_count: 0,
            table_width: 4,
            starting_block_size,
            max_direct_block_size: 4096,
            max_heap_size,
            starting_row_root_indirect: 0,
            root_block_address: block_address,
            current_rows_in_root_indirect: 0,
            io_filter_size: None,
            io_filter_mask: None,
        };

        // Build file data with the direct block.
        let file_size = block_address as usize + starting_block_size as usize + 100;
        let mut file_data = vec![0u8; file_size];

        // Write direct block header at block_address.
        let ba = block_address as usize;
        file_data[ba..ba + 4].copy_from_slice(b"FHDB");
        file_data[ba + 4] = 0; // version
                               // heap header address (8 bytes) — doesn't matter for this test
                               // block offset (2 bytes) — 0

        // Write object data at offset 0 within the heap.
        let obj_data = b"test object data";
        let obj_start = ba + db_header_size; // offset 0 within heap
        file_data[obj_start..obj_start + obj_data.len()].copy_from_slice(obj_data);

        // Build heap ID for managed object at offset=0, length=16.
        // Type nibble = 0, offset = 0 (16 bits), length = 16 (12 bits)
        // offset_bits = 16, length_bits = (8*4 - 4) - 16 = 12
        // Bit stream: 0000 | 0000_0000_0000_0000 | 0000_0001_0000
        //           = 0x00, 0x00, 0x00, 0x10
        let heap_id = [0x00, 0x00, 0x00, 0x10];

        let result = heap
            .get_managed_object(&heap_id, &file_data, offset_size, 8)
            .unwrap();
        assert_eq!(result, obj_data);
    }
}
