//! ODIM_H5 decoding over the pure-Rust HDF5 reader [`recast_radar_hdf5`]
//! (re-exported as [`hdf5`]).
//!
//! - [`odim`]: polar `PVOL`/`SCAN` objects into the FM301
//!   [`recast_radar_core::model::Volume`] ([`odim::read_odim_h5_volume`]).
//! - [`odim_cartesian`]: Cartesian `IMAGE`/`MAX` products into a gridded
//!   [`odim_cartesian::OdimCartesianGrid`].
//! - [`write`](mod@write): any volume of PPI sweeps as an ODIM_H5 polar volume
//!   ([`write_odim_h5_volume`]) over [`recast_radar_hdf5::write`].
//!
//! Any HDF5 layout reads: superblocks v0-v3, old- and new-style groups,
//! dense attribute storage, every chunk index (see [`recast_radar_hdf5`]).
//! Every attribute reaches the model whatever its datatype (see [`odim`]):
//! one ODIM does not use (compound, reference, sequence, a datatype the
//! HDF5 reader keeps as raw bytes) is kept verbatim instead of failing the
//! file. [`odim::read_odim_hdf5_volume`] decodes an HDF5 file already
//! opened (the format router opens a file once to tell ODIM from
//! netCDF-4).
//!
//! # Limits
//!
//! The HDF5 reader bounds every structure a file header can inflate (its
//! crate documentation lists each limit: group depth 16, 16,384 objects,
//! 256 MiB per dataset, 16 MiB per attribute, ...). Reading a dataset holds
//! its raw bytes and converted elements at the same time (at most 512 MiB);
//! datasets are read one at a time.
//!
//! The polar decoder accepts at most `MAX_SWEEPS_PER_VOLUME` (1,024)
//! `datasetN` groups and `MAX_GATES_PER_RADIAL` (16,384) bins per ray, and
//! charges each sweep's radial table and moment grids (quality planes
//! included) to a `DecodeBudget` of `MAX_DECODED_VOLUME_BYTES` (1 GiB) before
//! allocating them (constants in [`recast_radar_core::bounded_read`]), and
//! the attributes it keeps verbatim, by the memory they hold, to the same
//! budget. Nested groups are read 8 deep below a `how` or unknown group.
//! The Cartesian decoder caps its physical-value grid at the same 1 GiB.
//!
//! Every limit violation is an [`OdimError::LimitExceeded`] error.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

mod h5;
pub mod odim;
pub mod odim_cartesian;
mod tables;
pub mod write;

pub use recast_radar_hdf5 as hdf5;

use thiserror::Error;

pub use odim::{
    looks_like_hdf5_bytes, read_odim_h5_volume, read_odim_hdf5_volume,
    recover_copied_whatgroup_velocity_nodata,
};
pub use odim_cartesian::{OdimCartesianGrid, decode_odim_h5_cartesian_max};
pub use write::{OdimWriteError, OdimWriteOptions, write_odim_h5_volume};

/// Result type for ODIM_H5 and HDF5 decoding.
pub type Result<T> = std::result::Result<T, OdimError>;

/// Errors from ODIM_H5 and HDF5 decoding.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OdimError {
    /// An HDF5 structure ended before its declared length.
    #[error("truncated {what} at offset {offset}: need {needed} bytes, have {available}")]
    Truncated {
        /// Structure being read.
        what: &'static str,
        /// Byte offset of the structure.
        offset: usize,
        /// Bytes the structure needs.
        needed: usize,
        /// Bytes actually available.
        available: usize,
    },
    /// Structurally invalid or unsupported HDF5/ODIM content.
    #[error("invalid message at offset {offset}: {reason}")]
    InvalidMessage {
        /// Byte offset of the problem (0 when not meaningful).
        offset: usize,
        /// Human-readable description.
        reason: String,
    },
    /// The file declares more data than a documented resource limit allows
    /// (see the crate-level `# Limits` section).
    #[error("decode limit exceeded: {0}")]
    LimitExceeded(String),
}
