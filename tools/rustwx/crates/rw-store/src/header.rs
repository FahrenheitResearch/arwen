//! Binary file header for rw-store hour files.
//!
//! Byte layout (little-endian, 64 bytes):
//! ```text
//!  0- 7  magic          b"RWSTORE1"
//!  8-11  version        u32
//! 12-15  meta_len       u32
//! 16-23  index_count    u64
//! 24-31  index_offset   u64
//! 32-39  payload_offset u64
//! 40-63  reserved       (zeros)
//! ```

use crate::error::{RwResult, RwStoreError};
use crate::format::{HEADER_LEN, INDEX_RECORD_LEN, MAGIC, SUPPORTED_VERSIONS, VERSION};

/// Parsed representation of the 64-byte rw-store file header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RwsHeader {
    pub version: u32,
    pub meta_len: u32,
    pub index_count: u64,
    pub index_offset: u64,
    pub payload_offset: u64,
}

impl RwsHeader {
    /// Construct a header with offsets derived from the fixed layout rules:
    /// - `index_offset  = HEADER_LEN + meta_len`
    /// - `payload_offset = index_offset + index_count * INDEX_RECORD_LEN`
    pub fn for_layout(meta_len: u32, index_count: u64) -> Self {
        let index_offset = HEADER_LEN as u64 + meta_len as u64;
        let payload_offset = index_offset + index_count * INDEX_RECORD_LEN as u64;
        Self {
            version: VERSION,
            meta_len,
            index_count,
            index_offset,
            payload_offset,
        }
    }

    /// Serialize to a 64-byte array.
    pub fn pack(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..8].copy_from_slice(MAGIC); // 0-7
        out[8..12].copy_from_slice(&self.version.to_le_bytes()); // 8-11
        out[12..16].copy_from_slice(&self.meta_len.to_le_bytes()); // 12-15
        out[16..24].copy_from_slice(&self.index_count.to_le_bytes()); // 16-23
        out[24..32].copy_from_slice(&self.index_offset.to_le_bytes()); // 24-31
        out[32..40].copy_from_slice(&self.payload_offset.to_le_bytes()); // 32-39
        // 40-63 already zeroed
        out
    }

    /// Parse and validate a header from `bytes`.
    ///
    /// Checks:
    /// - buffer length >= 64
    /// - magic == `b"RWSTORE1"`
    /// - version in `SUPPORTED_VERSIONS`
    /// - `index_offset == HEADER_LEN + meta_len`
    /// - `payload_offset == index_offset + index_count * INDEX_RECORD_LEN`
    pub fn parse(bytes: &[u8]) -> RwResult<Self> {
        if bytes.len() < HEADER_LEN {
            return Err(RwStoreError::Format(format!(
                "header requires {} bytes, got {}",
                HEADER_LEN,
                bytes.len()
            )));
        }

        if &bytes[0..8] != MAGIC.as_slice() {
            return Err(RwStoreError::Format(format!(
                "bad magic: expected {:?}, got {:?}",
                MAGIC,
                &bytes[0..8]
            )));
        }

        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if !SUPPORTED_VERSIONS.contains(&version) {
            return Err(RwStoreError::UnsupportedVersion {
                found: version,
                supported: SUPPORTED_VERSIONS,
            });
        }

        let meta_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let index_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let index_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        let payload_offset = u64::from_le_bytes(bytes[32..40].try_into().unwrap());

        let expected_index_offset = HEADER_LEN as u64 + meta_len as u64;
        if index_offset != expected_index_offset {
            return Err(RwStoreError::Format(format!(
                "inconsistent index_offset: expected {expected_index_offset}, got {index_offset}"
            )));
        }

        // Checked math: a hostile index_count must not be able to wrap the
        // multiply (release) or panic on overflow (debug).
        let expected_payload_offset = index_count
            .checked_mul(INDEX_RECORD_LEN as u64)
            .and_then(|index_bytes| index_offset.checked_add(index_bytes))
            .ok_or_else(|| {
                RwStoreError::Format(format!(
                    "index_count {index_count} overflows the index size computation"
                ))
            })?;
        if payload_offset != expected_payload_offset {
            return Err(RwStoreError::Format(format!(
                "inconsistent payload_offset: expected {expected_payload_offset}, got {payload_offset}"
            )));
        }

        Ok(Self {
            version,
            meta_len,
            index_count,
            index_offset,
            payload_offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trips() {
        let h = RwsHeader::for_layout(256, 42);
        let packed = h.pack();
        assert_eq!(packed.len(), 64);
        let h2 = RwsHeader::parse(&packed).unwrap();
        assert_eq!(h, h2);
    }

    #[test]
    fn header_rejects_overflowing_index_count() {
        // Regression: index_count chosen so count * 64 wraps u64 to 0 made the
        // unchecked math pass validation and later panic on Vec::with_capacity.
        let mut packed = RwsHeader::for_layout(0, 0).pack();
        packed[16..24].copy_from_slice(&(1u64 << 58).to_le_bytes());
        let err = RwsHeader::parse(&packed).unwrap_err();
        match err {
            RwStoreError::Format(msg) => assert!(
                msg.contains("overflow"),
                "expected overflow message, got: {msg}"
            ),
            other => panic!("expected Format error, got {other:?}"),
        }
    }

    #[test]
    fn header_rejects_bad_magic() {
        let mut packed = RwsHeader::for_layout(128, 10).pack();
        packed[0] = b'X'; // corrupt magic
        let err = RwsHeader::parse(&packed).unwrap_err();
        assert!(
            matches!(err, RwStoreError::Format(_)),
            "expected Format error, got {err:?}"
        );
    }

    #[test]
    fn header_rejects_unsupported_version() {
        let mut packed = RwsHeader::for_layout(128, 10).pack();
        // overwrite version field with 2
        packed[8..12].copy_from_slice(&2u32.to_le_bytes());
        let err = RwsHeader::parse(&packed).unwrap_err();
        match err {
            RwStoreError::UnsupportedVersion { found, supported } => {
                assert_eq!(found, 2);
                assert_eq!(supported, &[1]);
            }
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn header_rejects_short_buffer() {
        let packed = RwsHeader::for_layout(0, 0).pack();
        let err = RwsHeader::parse(&packed[..32]).unwrap_err();
        assert!(
            matches!(err, RwStoreError::Format(_)),
            "expected Format error, got {err:?}"
        );
    }

    #[test]
    fn header_rejects_inconsistent_offsets() {
        // Corrupt index_offset: off by one
        let mut packed = RwsHeader::for_layout(128, 10).pack();
        let bad_index_offset = (HEADER_LEN as u64 + 128 + 1).to_le_bytes();
        packed[24..32].copy_from_slice(&bad_index_offset);
        let err = RwsHeader::parse(&packed).unwrap_err();
        assert!(
            matches!(err, RwStoreError::Format(_)),
            "expected Format error for bad index_offset, got {err:?}"
        );

        // Corrupt payload_offset: off by one
        let mut packed2 = RwsHeader::for_layout(128, 10).pack();
        let h = RwsHeader::parse(&packed2).unwrap();
        let bad_payload = (h.payload_offset + 1).to_le_bytes();
        packed2[32..40].copy_from_slice(&bad_payload);
        let err2 = RwsHeader::parse(&packed2).unwrap_err();
        assert!(
            matches!(err2, RwStoreError::Format(_)),
            "expected Format error for bad payload_offset, got {err2:?}"
        );
    }
}
