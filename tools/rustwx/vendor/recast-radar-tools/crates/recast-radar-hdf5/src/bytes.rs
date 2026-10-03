//! Bounds-checked little-endian readers. HDF5 metadata is always
//! little-endian; raw data byte order comes from the datatype.

use crate::error::{Result, invalid, truncated};

/// The canonical "undefined address" value (all bits set).
pub(crate) const UNDEFINED_ADDR: u64 = u64::MAX;

pub(crate) fn checked_range(bytes: &[u8], at: usize, len: usize) -> Result<&[u8]> {
    let end = at
        .checked_add(len)
        .ok_or_else(|| invalid(at, "HDF5 byte range overflow"))?;
    bytes
        .get(at..end)
        .ok_or_else(|| truncated(at, len, bytes.len()))
}

pub(crate) fn read_le_u32(bytes: &[u8], at: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(array_at(bytes, at)?))
}

/// The `N` bytes of `bytes` starting at `at`.
pub(crate) fn array_at<const N: usize>(bytes: &[u8], at: usize) -> Result<[u8; N]> {
    bytes
        .get(at..)
        .and_then(|tail| tail.first_chunk::<N>())
        .copied()
        .ok_or_else(|| truncated(at, N, bytes.len()))
}

/// Little-endian unsigned integer of `size` (0..=8) bytes. An all-ones value
/// is a value here (sizes, lengths, counts), unlike [`normalize_addr`].
pub(crate) fn read_uint(bytes: &[u8], at: usize, size: usize) -> Result<u64> {
    if size > 8 {
        return Err(invalid(
            at,
            format!("{size}-byte integer field exceeds 8 bytes"),
        ));
    }
    let raw = checked_range(bytes, at, size)?;
    Ok(le_uint(raw))
}

/// Little-endian value of up to eight bytes.
pub(crate) fn le_uint(raw: &[u8]) -> u64 {
    raw.iter()
        .take(8)
        .enumerate()
        .fold(0u64, |value, (index, byte)| {
            value | (u64::from(*byte) << (8 * index))
        })
}

pub(crate) fn normalize_addr(value: u64, size: usize) -> u64 {
    if (1..8).contains(&size) && value == (1u64 << (8 * size)) - 1 {
        UNDEFINED_ADDR
    } else {
        value
    }
}

pub(crate) fn to_usize(value: u64, offset: usize, what: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| invalid(offset, format!("{what} {value} overflows usize")))
}

/// Number of bytes needed to encode `value` (HDF5 `H5VM_limit_enc_size`):
/// `log2(value) / 8 + 1`.
pub(crate) fn limit_enc_size(value: u64) -> usize {
    (log2_floor(value) / 8 + 1) as usize
}

/// `floor(log2(value))`, 0 for 0 (HDF5 `H5VM_log2_gen`).
pub(crate) fn log2_floor(value: u64) -> u32 {
    if value == 0 {
        0
    } else {
        63 - value.leading_zeros()
    }
}

/// A forward reader over a byte slice that knows the file's offset and
/// length sizes. `origin` is the file offset of `bytes[0]`, used in errors.
pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    origin: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(bytes: &'a [u8], origin: usize) -> Self {
        Self {
            bytes,
            pos: 0,
            origin,
        }
    }

    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    pub(crate) fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn here(&self) -> usize {
        self.origin.saturating_add(self.pos)
    }

    pub(crate) fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| invalid(self.here(), "HDF5 field length overflow"))?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| truncated(self.here(), len, self.remaining()))?;
        self.pos = end;
        Ok(slice)
    }

    pub(crate) fn skip(&mut self, len: usize) -> Result<()> {
        self.take(len).map(|_| ())
    }

    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16> {
        Ok(le_uint(self.take(2)?) as u16)
    }

    pub(crate) fn u32(&mut self) -> Result<u32> {
        Ok(le_uint(self.take(4)?) as u32)
    }

    pub(crate) fn u64(&mut self) -> Result<u64> {
        Ok(le_uint(self.take(8)?))
    }

    /// Unsigned little-endian integer of `size` (0..=8) bytes.
    pub(crate) fn uint(&mut self, size: usize) -> Result<u64> {
        if size > 8 {
            return Err(invalid(
                self.here(),
                format!("{size}-byte integer field exceeds 8 bytes"),
            ));
        }
        Ok(le_uint(self.take(size)?))
    }

    /// File address of `size` bytes (all ones = undefined).
    pub(crate) fn addr(&mut self, size: usize) -> Result<u64> {
        let value = self.uint(size)?;
        Ok(normalize_addr(value, size))
    }

    /// A length or size field of `size` bytes as `usize`.
    pub(crate) fn length(&mut self, size: usize, what: &str) -> Result<usize> {
        let at = self.here();
        to_usize(self.uint(size)?, at, what)
    }

    /// Bytes up to (not including) the next NUL; the cursor moves past it.
    pub(crate) fn c_string(&mut self) -> Result<&'a [u8]> {
        let rest = self.bytes.get(self.pos..).unwrap_or_default();
        let len = rest
            .iter()
            .position(|byte| *byte == 0)
            .ok_or_else(|| invalid(self.here(), "unterminated HDF5 string"))?;
        let text = self.take(len)?;
        self.skip(1)?;
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undefined_addresses_normalize_across_offset_sizes() {
        assert_eq!(normalize_addr(0xFFFF_FFFF, 4), UNDEFINED_ADDR);
        assert_eq!(normalize_addr(u64::MAX, 8), UNDEFINED_ADDR);
        assert_eq!(normalize_addr(0x10, 4), 0x10);
    }

    /// Unlike `normalize_addr`, `read_uint` keeps all-ones values: 0xFF is a
    /// legal one-byte chunk size.
    #[test]
    fn read_uint_keeps_all_ones_values() {
        assert_eq!(read_uint(&[0xFF], 0, 1).unwrap(), 0xFF);
        assert_eq!(read_uint(&[0xFF, 0xFF], 0, 2).unwrap(), 0xFFFF);
        assert_eq!(read_uint(&[0x83, 0x01], 0, 2).unwrap(), 0x0183);
        assert!(read_uint(&[0; 9], 0, 9).is_err());
    }

    #[test]
    fn encoded_sizes_match_hdf5_formulas() {
        // H5VM_limit_enc_size: 4096 -> 2 bytes, 255 -> 1, 256 -> 2.
        assert_eq!(limit_enc_size(4096), 2);
        assert_eq!(limit_enc_size(255), 1);
        assert_eq!(limit_enc_size(256), 2);
        assert_eq!(log2_floor(65536), 16);
        assert_eq!(log2_floor(0), 0);
    }
}
