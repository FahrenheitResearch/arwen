//! Two outcomes besides success, and the process exit code each one maps to.
//!
//! A REFUSAL is a request this exporter will not carry out because the
//! result would be wrong in a way nobody would notice until a training job
//! read it.  Its message is one sentence that names that breakage, and the
//! process exits 2.  A FAILURE is everything else (a file that cannot be
//! written, a reader error) and exits 1.

use std::fmt;

#[derive(Debug)]
pub enum ExportError {
    Refused(String),
    Failed(String),
}

impl ExportError {
    pub fn exit_code(&self) -> u8 {
        match self {
            ExportError::Refused(_) => 2,
            ExportError::Failed(_) => 1,
        }
    }

    pub fn message(&self) -> &str {
        match self {
            ExportError::Refused(m) | ExportError::Failed(m) => m,
        }
    }

    pub fn is_refusal(&self) -> bool {
        matches!(self, ExportError::Refused(_))
    }
}

impl fmt::Display for ExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for ExportError {}

pub type Result<T> = std::result::Result<T, ExportError>;

/// A refusal: `message` names the breakage it prevents.
pub fn refuse(message: impl Into<String>) -> ExportError {
    ExportError::Refused(message.into())
}

/// A failure with context.
pub fn fail(message: impl Into<String>) -> ExportError {
    ExportError::Failed(message.into())
}

impl From<std::io::Error> for ExportError {
    fn from(error: std::io::Error) -> Self {
        ExportError::Failed(error.to_string())
    }
}

/// Attach a context sentence to an I/O or reader error.
pub trait Context<T> {
    fn context(self, what: impl FnOnce() -> String) -> Result<T>;
}

impl<T, E: fmt::Display> Context<T> for std::result::Result<T, E> {
    fn context(self, what: impl FnOnce() -> String) -> Result<T> {
        self.map_err(|error| ExportError::Failed(format!("{}: {error}", what())))
    }
}
