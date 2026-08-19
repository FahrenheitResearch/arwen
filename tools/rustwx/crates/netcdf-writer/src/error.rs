//! Errors, spelled so the message names the breakage rather than the rule.

use std::fmt;

/// Everything this crate can refuse, and why.
#[derive(Debug)]
pub enum NcWriteError {
    /// The schema cannot be expressed in the classic format at all
    /// (bad name, duplicate, two record dimensions, a type the target
    /// container has no code for).
    Schema(String),
    /// The schema is representable but does not fit the chosen container
    /// (a CDF-1 data section past the 32-bit offset field).
    Capacity(String),
    /// The caller used the writer wrongly: wrong element count, wrong
    /// data type, a variable written twice, `finish` over a hole.
    Usage(String),
    /// The filesystem said no.
    Io(std::io::Error),
}

impl fmt::Display for NcWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NcWriteError::Schema(msg) => write!(f, "netcdf-writer schema: {msg}"),
            NcWriteError::Capacity(msg) => write!(f, "netcdf-writer capacity: {msg}"),
            NcWriteError::Usage(msg) => write!(f, "netcdf-writer usage: {msg}"),
            NcWriteError::Io(err) => write!(f, "netcdf-writer io: {err}"),
        }
    }
}

impl std::error::Error for NcWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NcWriteError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for NcWriteError {
    fn from(err: std::io::Error) -> Self {
        NcWriteError::Io(err)
    }
}

/// The crate's result alias.
pub type Result<T> = std::result::Result<T, NcWriteError>;
