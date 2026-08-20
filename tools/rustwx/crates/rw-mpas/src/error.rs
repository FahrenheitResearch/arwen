//! One refusal type. Every check in this crate that can say no says it here,
//! with the numbers that made it say no, so a caller never has to guess.

use std::fmt;

#[derive(Debug)]
pub enum MpasError {
    /// A mesh, history, window or emitted-frame refusal.
    Refusal(String),
    Io(std::io::Error),
    NetCdf(String),
    Store(rw_store::error::RwStoreError),
}

pub type MpasResult<T> = Result<T, MpasError>;

impl fmt::Display for MpasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MpasError::Refusal(m) => write!(f, "{m}"),
            MpasError::Io(e) => write!(f, "io: {e}"),
            MpasError::NetCdf(m) => write!(f, "netcdf read: {m}"),
            MpasError::Store(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for MpasError {}

impl From<std::io::Error> for MpasError {
    fn from(e: std::io::Error) -> Self {
        MpasError::Io(e)
    }
}

impl From<rw_store::error::RwStoreError> for MpasError {
    fn from(e: rw_store::error::RwStoreError) -> Self {
        MpasError::Store(e)
    }
}

impl From<netcrust::Error> for MpasError {
    fn from(e: netcrust::Error) -> Self {
        MpasError::NetCdf(e.to_string())
    }
}
