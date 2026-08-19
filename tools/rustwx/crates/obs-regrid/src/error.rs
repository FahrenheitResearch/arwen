//! The error taxonomy, seeded from `rustwx-regrid::error` in Drew's
//! consolidated Rust (`crates/rustwx-regrid/src/error.rs`): the same
//! four failure classes, because they are the same four things that can
//! be wrong with a remap request.  `UnsupportedGeometry` is dropped --
//! this engine takes lat/lon arrays directly rather than a geometry
//! trait object, so there is no geometry that can decline to expose a
//! centre.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegridError {
    /// The grids themselves are wrong: ragged, empty, not 2-D.
    InvalidGrid(String),
    /// The request is wrong: an unknown method, a bound that is not a
    /// distance.
    InvalidOptions(String),
    /// A field does not match the plan it is being applied with.
    ShapeMismatch(String),
}

impl fmt::Display for RegridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegridError::InvalidGrid(message)
            | RegridError::InvalidOptions(message)
            | RegridError::ShapeMismatch(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for RegridError {}
