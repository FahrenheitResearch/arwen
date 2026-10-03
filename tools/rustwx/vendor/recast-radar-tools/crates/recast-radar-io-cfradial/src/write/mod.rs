//! CfRadial writers.
//!
//! - [`write_cfradial1`]: CfRadial 1.4 in classic netCDF (64-bit offset,
//!   CDF-2), from any [`recast_radar_core::model::Volume`].
//! - [`write_cfradial2`]: CfRadial 2 / WMO FM301 in netCDF-4, serialising
//!   the FM301 view ([`recast_radar_core::fm301`]).
//! - [`netcdf3`]: the classic netCDF encoder underneath.
//!
//! Design notes: `docs/design/writers.md`.

mod cfradial1;
mod cfradial2;
pub mod netcdf3;

use thiserror::Error;

pub use cfradial1::{Cfradial1Options, RangeLayout, time_order, write_cfradial1};
pub use cfradial2::{Cfradial2Options, write_cfradial2};

/// Errors from the CfRadial writers.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CfWriteError {
    /// Something the format cannot represent (see each writer).
    #[error("CfRadial cannot represent {0}")]
    Unrepresentable(String),
    /// A structure the encoder refused (an invalid name, a size mismatch).
    #[error("invalid: {0}")]
    Invalid(String),
    /// A structure larger than the format can store.
    #[error("too large: {0}")]
    TooLarge(String),
    /// The HDF5 / netCDF-4 writer refused a structure.
    #[error(transparent)]
    Hdf5(#[from] recast_radar_hdf5::write::WriteError),
    /// The FM301 view refused the volume.
    #[error(transparent)]
    View(#[from] recast_radar_core::fm301::ViewError),
}
