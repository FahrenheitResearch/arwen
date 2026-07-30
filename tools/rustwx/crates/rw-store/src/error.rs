//! Error type for the rw-store crate.

/// Result alias used throughout rw-store.
pub type RwResult<T> = Result<T, RwStoreError>;

/// Errors produced while reading or writing rw-store files.
#[derive(Debug, thiserror::Error)]
pub enum RwStoreError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("malformed store file: {0}")]
    Format(String),
    #[error("unsupported store version {found} (supported: {supported:?})")]
    UnsupportedVersion {
        found: u32,
        supported: &'static [u32],
    },
    #[error("invalid store metadata: {0}")]
    Meta(String),
    #[error("unknown variable '{0}'")]
    UnknownVariable(String),
    #[error("invalid chunk: {0}")]
    Chunk(String),
    #[error("grid mismatch: {0}")]
    Grid(String),
    #[error("run directory locked: {0}")]
    Locked(String),
}
