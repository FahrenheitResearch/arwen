use thiserror::Error;

#[derive(Error, Debug)]
pub enum RustmetError {
    #[error("GRIB2 parse error: {0}")]
    Parse(String),

    #[error("GRIB2 unpack error: {0}")]
    Unpack(String),

    #[error("Unsupported template {template}: {detail}")]
    UnsupportedTemplate { template: u16, detail: String },

    #[error("HTTP error: {0}")]
    Http(String),

    #[error("HTTP status {code}: {url}")]
    HttpStatus { code: u16, url: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("No data: {0}")]
    NoData(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

pub type Result<T> = std::result::Result<T, RustmetError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_parse_error() {
        let err = RustmetError::Parse("unexpected end of file".to_string());
        assert_eq!(format!("{}", err), "GRIB2 parse error: unexpected end of file");
    }

    #[test]
    fn test_display_unpack_error() {
        let err = RustmetError::Unpack("bad data".to_string());
        assert_eq!(format!("{}", err), "GRIB2 unpack error: bad data");
    }

    #[test]
    fn test_display_unsupported_template() {
        let err = RustmetError::UnsupportedTemplate {
            template: 99,
            detail: "grid definition".to_string(),
        };
        assert_eq!(format!("{}", err), "Unsupported template 99: grid definition");
    }

    #[test]
    fn test_display_http_error() {
        let err = RustmetError::Http("connection refused".to_string());
        assert_eq!(format!("{}", err), "HTTP error: connection refused");
    }

    #[test]
    fn test_display_http_status() {
        let err = RustmetError::HttpStatus {
            code: 404,
            url: "https://example.com/data.grib2".to_string(),
        };
        assert_eq!(
            format!("{}", err),
            "HTTP status 404: https://example.com/data.grib2"
        );
    }

    #[test]
    fn test_display_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = RustmetError::Io(io_err);
        assert_eq!(format!("{}", err), "IO error: file not found");
    }

    #[test]
    fn test_display_model_not_found() {
        let err = RustmetError::ModelNotFound("HRRR".to_string());
        assert_eq!(format!("{}", err), "Model not found: HRRR");
    }

    #[test]
    fn test_display_no_data() {
        let err = RustmetError::NoData("no temperature field".to_string());
        assert_eq!(format!("{}", err), "No data: no temperature field");
    }

    #[test]
    fn test_display_invalid_argument() {
        let err = RustmetError::InvalidArgument("hour must be 0-23".to_string());
        assert_eq!(format!("{}", err), "Invalid argument: hour must be 0-23");
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let err: RustmetError = io_err.into();
        match err {
            RustmetError::Io(_) => {} // expected
            other => panic!("Expected Io variant, got: {:?}", other),
        }
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        // RustmetError contains std::io::Error which is Send+Sync
        // This test verifies the error type can be used across threads
        assert_send_sync::<RustmetError>();
    }
}
