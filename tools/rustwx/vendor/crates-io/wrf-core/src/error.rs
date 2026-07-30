use std::fmt;

/// All errors produced by wrf-core.
#[derive(Debug)]
pub enum WrfError {
    /// NetCDF I/O or library error.
    NetCdf(String),
    /// Requested WRF variable not found in file.
    VarNotFound(String),
    /// Global attribute not found.
    AttrNotFound(String),
    /// Attribute has unexpected type.
    AttrType(String),
    /// Dimension mismatch.
    DimMismatch(String),
    /// Unknown variable name (not in registry).
    UnknownVar(String),
    /// Unit conversion error.
    UnitConversion(String),
    /// Invalid parameter.
    InvalidParam(String),
    /// General computation error.
    Compute(String),
}

impl fmt::Display for WrfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NetCdf(s) => write!(f, "NetCDF error: {s}"),
            Self::VarNotFound(s) => write!(f, "variable not found in WRF file: {s}"),
            Self::AttrNotFound(s) => write!(f, "global attribute not found: {s}"),
            Self::AttrType(s) => write!(f, "unexpected attribute type: {s}"),
            Self::DimMismatch(s) => write!(f, "dimension mismatch: {s}"),
            Self::UnknownVar(s) => write!(f, "unknown variable: {s}"),
            Self::UnitConversion(s) => write!(f, "unit conversion error: {s}"),
            Self::InvalidParam(s) => write!(f, "invalid parameter: {s}"),
            Self::Compute(s) => write!(f, "computation error: {s}"),
        }
    }
}

impl std::error::Error for WrfError {}

#[cfg(feature = "netcdf-backend")]
impl From<netcdf::Error> for WrfError {
    fn from(e: netcdf::Error) -> Self {
        Self::NetCdf(e.to_string())
    }
}

pub type WrfResult<T> = Result<T, WrfError>;
