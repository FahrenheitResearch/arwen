//! Error type for HDF5 parsing.

use thiserror::Error;

/// Result type of this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors from HDF5 parsing and dataset reads.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// A structure ended before its declared length.
    #[error("truncated {what} at offset {offset}: need {needed} bytes, have {available}")]
    Truncated {
        /// Structure being read.
        what: &'static str,
        /// Byte offset of the structure (file offset, or offset inside a
        /// message body).
        offset: usize,
        /// Bytes the structure needs.
        needed: usize,
        /// Bytes actually available.
        available: usize,
    },
    /// Structurally invalid content.
    #[error("invalid message at offset {offset}: {reason}")]
    Invalid {
        /// Byte offset of the problem (0 when not meaningful).
        offset: usize,
        /// Human-readable description.
        reason: String,
    },
    /// A stored checksum does not match the bytes it covers.
    #[error(
        "{structure} checksum mismatch at offset {offset} (stored {stored:#010x}, computed {computed:#010x})"
    )]
    Checksum {
        /// Structure the checksum protects.
        structure: &'static str,
        /// File offset of the structure.
        offset: usize,
        /// Checksum stored in the file.
        stored: u32,
        /// Checksum computed over the bytes.
        computed: u32,
    },
    /// The file declares more data than a documented resource limit allows
    /// (see the crate-level `# Limits` section).
    #[error("decode limit exceeded: {0}")]
    LimitExceeded(String),
    /// A dataset is compressed with a filter this crate does not implement
    /// (szip, LZF, n-bit, scale-offset, or a third-party filter).
    #[error("HDF5 filter {id} ({name}) is not supported")]
    UnsupportedFilter {
        /// Registered HDF5 filter identifier.
        id: u16,
        /// Filter name (registered name, or the name stored in the file).
        name: String,
    },
    /// A well-formed HDF5 feature this crate does not implement (for
    /// example shared object header messages, virtual datasets, external
    /// raw data files).
    #[error("unsupported HDF5 feature: {0}")]
    Unsupported(String),
    /// No object exists at the requested path.
    #[error("HDF5 object '{0}' not found")]
    NotFound(String),
}

pub(crate) fn invalid(offset: usize, reason: impl Into<String>) -> Error {
    Error::Invalid {
        offset,
        reason: reason.into(),
    }
}

pub(crate) fn truncated(offset: usize, needed: usize, available: usize) -> Error {
    Error::Truncated {
        what: "HDF5 structure",
        offset,
        needed,
        available,
    }
}

pub(crate) fn limit(reason: impl Into<String>) -> Error {
    Error::LimitExceeded(reason.into())
}

pub(crate) fn unsupported(feature: impl Into<String>) -> Error {
    Error::Unsupported(feature.into())
}
