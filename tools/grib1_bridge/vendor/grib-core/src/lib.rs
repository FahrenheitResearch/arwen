pub mod grib1;
pub mod grib2;

/// Error types for GRIB file operations.
#[derive(Debug)]
pub enum GribError {
    /// I/O error reading a file.
    Io(std::io::Error),
    /// Error parsing a GRIB message structure.
    Parse(String),
    /// Error unpacking data values.
    Unpack(String),
    /// Unsupported template number.
    UnsupportedTemplate { template: u16, detail: String },
}

impl std::fmt::Display for GribError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GribError::Io(e) => write!(f, "I/O error: {}", e),
            GribError::Parse(msg) => write!(f, "Parse error: {}", msg),
            GribError::Unpack(msg) => write!(f, "Unpack error: {}", msg),
            GribError::UnsupportedTemplate { template, detail } => {
                write!(f, "Unsupported {} template: {}", detail, template)
            }
        }
    }
}

impl std::error::Error for GribError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GribError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GribError {
    fn from(e: std::io::Error) -> Self {
        GribError::Io(e)
    }
}

/// Convenience type alias for Results using GribError.
pub type Result<T> = std::result::Result<T, GribError>;
