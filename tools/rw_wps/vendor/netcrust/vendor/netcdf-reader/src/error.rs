/// Errors produced by the NetCDF reader.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid NetCDF magic bytes")]
    InvalidMagic,

    #[error("unsupported NetCDF format version {0}")]
    UnsupportedVersion(u8),

    #[error("variable not found: {0}")]
    VariableNotFound(String),

    #[error("dimension not found: {0}")]
    DimensionNotFound(String),

    #[error("attribute not found: {0}")]
    AttributeNotFound(String),

    #[error("group not found: {0}")]
    GroupNotFound(String),

    #[error("type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    #[error("invalid data: {0}")]
    InvalidData(String),

    #[error("HDF5 error: {0}")]
    #[cfg(feature = "netcdf4")]
    Hdf5(#[from] hdf5_reader::error::Error),

    #[error("NetCDF-4 support not enabled (enable 'netcdf4' feature)")]
    Nc4NotEnabled,
}

pub type Result<T> = std::result::Result<T, Error>;
