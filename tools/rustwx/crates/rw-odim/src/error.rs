//! Errors this decoder raises.
//!
//! The house convention (see `rw-glm`'s `RwlError` and `rustwx-io`'s
//! `IoError`) is that `Io` is reserved for genuine I/O and every structural
//! or semantic problem becomes a variant carrying the ODIM path and the
//! failing attribute name, so a refusal names the thing that failed rather
//! than merely the file.
//!
//! Every variant here is a *refusal*. This decoder has no "best effort" mode:
//! a polar volume whose geometry or sentinels cannot be read is not decoded
//! into plausible numbers, because a plausible-looking radial velocity field
//! is exactly the input that would be assimilated without anyone noticing.

use std::path::PathBuf;

/// The result type every public entry point in this crate returns.
pub type Result<T> = std::result::Result<T, OdimError>;

/// Why an ODIM_H5 polar volume could not be read.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum OdimError {
    /// The file could not be opened or read at all.
    #[error("i/o error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// The bytes are not a readable HDF5 container, or a chunk inside it did
    /// not decode. A corrupt compressed block lands here.
    #[error("{path} did not decode as HDF5: {detail}")]
    Hdf5 { path: PathBuf, detail: String },

    /// A required ODIM group is absent.
    #[error("missing ODIM group {path}")]
    MissingGroup { path: String },

    /// A required ODIM attribute is absent from a group that must carry it.
    #[error("missing ODIM attribute {group}@{name}")]
    MissingAttribute { group: String, name: String },

    /// An attribute is present but not readable as the type ODIM specifies.
    #[error("ODIM attribute {group}@{name} is {found}, which is not readable as {expected}")]
    AttributeType {
        group: String,
        name: String,
        found: String,
        expected: String,
    },

    /// The file is structurally HDF5 and structurally ODIM, but says something
    /// that cannot be true. Geometry that disagrees with the payload shape,
    /// a non-finite calibration, an elevation outside the sphere.
    #[error("{context}: {detail}")]
    Format { context: String, detail: String },

    /// `/what/object` names a product this decoder does not claim to read.
    #[error("{path} declares /what/object {object:?}, which this decoder does not read: {detail}")]
    UnsupportedObject {
        path: PathBuf,
        object: String,
        detail: String,
    },

    /// The caller asked for a sweep or quantity the volume does not contain.
    #[error("{what} not found in {path}; this volume has {available}")]
    NotPresent {
        path: PathBuf,
        what: String,
        available: String,
    },
}

impl OdimError {
    /// Shorthand for the catch-all structural refusal.
    pub(crate) fn format(context: impl Into<String>, detail: impl Into<String>) -> Self {
        OdimError::Format {
            context: context.into(),
            detail: detail.into(),
        }
    }
}
