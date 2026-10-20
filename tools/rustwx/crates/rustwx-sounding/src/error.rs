use thiserror::Error;

#[derive(Debug, Error)]
pub enum SoundingBridgeError {
    #[error("field `{field}` needs at least {expected_at_least} values, got {actual}")]
    InvalidLength {
        field: &'static str,
        expected_at_least: usize,
        actual: usize,
    },
    #[error("field `{field}` length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("field `{field}` contains invalid data: {reason}")]
    InvalidValue { field: &'static str, reason: String },
    #[error(transparent)]
    SharprsProfile(#[from] sharprs::profile::ProfileError),
    #[error(transparent)]
    Image(#[from] image::ImageError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("ECAPE bridge requires external values: {0}")]
    EcapeUnavailable(String),
    #[error("invalid external ECAPE summary: {0}")]
    InvalidEcapeSummary(String),
}
