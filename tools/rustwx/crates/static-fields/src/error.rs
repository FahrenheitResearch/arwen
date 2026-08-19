//! One error type across the crate; the C seam renders it to the
//! thread-local last-error string.

use std::fmt;

/// Every refusal this crate can make.  The `message` strings carry the
/// same diagnostic content as the Python exceptions they port
/// (`FileNotFoundError` naming missing tiles, `ValueError` naming shape
/// mismatches) because the Python callers surface them verbatim.
#[derive(Debug)]
pub enum StaticError {
    /// Input geometry/config refused (Python `ValueError`).
    Invalid(String),
    /// A required source artifact is absent (Python `FileNotFoundError`).
    Missing(String),
    /// An I/O failure underneath a read/write.
    Io(std::io::Error),
    /// The functionality is declared in the skeleton but its lane has
    /// not landed it yet.  Present so the seam loads and probes before
    /// the port is complete; every remaining variant of this at
    /// integration is a release blocker.
    NotImplemented(&'static str),
}

impl fmt::Display for StaticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StaticError::Invalid(msg) => write!(f, "{msg}"),
            StaticError::Missing(msg) => write!(f, "{msg}"),
            StaticError::Io(err) => write!(f, "I/O failure: {err}"),
            StaticError::NotImplemented(what) => write!(
                f,
                "static-fields: {what} is declared in the port skeleton \
                 but not yet implemented by its lane"
            ),
        }
    }
}

impl std::error::Error for StaticError {}

impl From<std::io::Error> for StaticError {
    fn from(err: std::io::Error) -> Self {
        StaticError::Io(err)
    }
}

pub type Result<T> = std::result::Result<T, StaticError>;
