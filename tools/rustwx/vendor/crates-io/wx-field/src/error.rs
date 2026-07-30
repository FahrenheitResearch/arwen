/// Error types for wx-field.

use std::fmt;

/// Errors that can occur in wx-field operations.
#[derive(Debug, Clone)]
pub enum WxFieldError {
    /// Invalid grid dimensions (e.g., zero nx/ny, data length mismatch).
    InvalidDimensions(String),

    /// Invalid projection parameters.
    InvalidProjection(String),

    /// Missing or invalid metadata.
    InvalidMetadata(String),

    /// Data out of expected range.
    OutOfRange(String),
}

impl fmt::Display for WxFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WxFieldError::InvalidDimensions(msg) => write!(f, "invalid dimensions: {}", msg),
            WxFieldError::InvalidProjection(msg) => write!(f, "invalid projection: {}", msg),
            WxFieldError::InvalidMetadata(msg) => write!(f, "invalid metadata: {}", msg),
            WxFieldError::OutOfRange(msg) => write!(f, "out of range: {}", msg),
        }
    }
}

impl std::error::Error for WxFieldError {}

/// Result type for wx-field operations.
pub type Result<T> = std::result::Result<T, WxFieldError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = WxFieldError::InvalidDimensions("nx must be > 0".to_string());
        assert_eq!(format!("{}", err), "invalid dimensions: nx must be > 0");
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WxFieldError>();
    }
}
