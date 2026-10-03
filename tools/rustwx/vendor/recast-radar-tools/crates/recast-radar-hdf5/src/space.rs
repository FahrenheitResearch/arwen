//! Address space of one HDF5 file: the bytes, the base address and the
//! offset and length sizes every structure reader needs.

use crate::bytes::{Cursor, UNDEFINED_ADDR};
use crate::error::{Error, Result, invalid, truncated};

/// Relative addresses resolve against `base` (the superblock position).
#[derive(Clone, Copy)]
pub(crate) struct Space<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) base: usize,
    pub(crate) offset_size: usize,
    pub(crate) length_size: usize,
    /// [`crate::OpenOptions::verify_metadata_checksums`].
    pub(crate) verify_checksums: bool,
}

impl<'a> Space<'a> {
    /// Check a stored lookup3 metadata checksum against `computed` (run
    /// only when the file is opened with checksum verification).
    pub(crate) fn check(
        &self,
        structure: &'static str,
        offset: usize,
        stored: u32,
        computed: impl FnOnce() -> u32,
    ) -> Result<()> {
        check_checksum(self.verify_checksums, structure, offset, stored, computed)
    }

    /// Absolute file offset of a relative address.
    pub(crate) fn abs(&self, address: u64) -> Result<usize> {
        if address == UNDEFINED_ADDR {
            return Err(invalid(0, "HDF5 undefined address dereferenced"));
        }
        usize::try_from(address)
            .ok()
            .and_then(|address| address.checked_add(self.base))
            .ok_or_else(|| invalid(0, format!("HDF5 address {address:#x} overflows")))
    }

    /// `len` bytes at a relative address.
    pub(crate) fn slice(&self, address: u64, len: usize) -> Result<&'a [u8]> {
        let start = self.abs(address)?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| invalid(start, "HDF5 byte range overflow"))?;
        self.bytes
            .get(start..end)
            .ok_or_else(|| truncated(start, len, self.bytes.len().saturating_sub(start)))
    }

    /// Everything from a relative address to the end of the file.
    pub(crate) fn tail(&self, address: u64) -> Result<&'a [u8]> {
        let start = self.abs(address)?;
        self.bytes
            .get(start..)
            .ok_or_else(|| truncated(start, 1, 0))
    }

    /// A cursor at a relative address, reading to the end of the file.
    pub(crate) fn cursor(&self, address: u64) -> Result<Cursor<'a>> {
        let start = self.abs(address)?;
        Ok(Cursor::new(self.tail(address)?, start))
    }

    /// True when `len` bytes at the address begin with `signature`.
    pub(crate) fn has_signature(&self, address: u64, signature: &[u8; 4]) -> bool {
        self.slice(address, 4).is_ok_and(|bytes| bytes == signature)
    }

    /// Require a 4-byte signature at a relative address.
    pub(crate) fn expect_signature(&self, address: u64, signature: &[u8; 4]) -> Result<()> {
        if self.has_signature(address, signature) {
            Ok(())
        } else {
            let at = self.abs(address).unwrap_or(0);
            Err(invalid(
                at,
                format!("expected {} signature", String::from_utf8_lossy(signature)),
            ))
        }
    }
}

/// [`Space::check`] for structures read before the address space exists
/// (the superblock).
pub(crate) fn check_checksum(
    verify: bool,
    structure: &'static str,
    offset: usize,
    stored: u32,
    computed: impl FnOnce() -> u32,
) -> Result<()> {
    if !verify {
        return Ok(());
    }
    let computed = computed();
    if stored == computed {
        Ok(())
    } else {
        Err(Error::Checksum {
            structure,
            offset,
            stored,
            computed,
        })
    }
}
