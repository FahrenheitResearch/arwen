//! CfRadial 1.x and CfRadial 2 decoding, from classic netCDF and netCDF-4,
//! and writing.
//!
//! - [`write`](mod@write): [`write_cfradial1`] (CfRadial 1.4, classic netCDF CDF-2) and
//!   [`write_cfradial2`] (CfRadial 2 / FM301, netCDF-4) from any volume
//!   (design notes: `docs/design/writers.md`).
//! - [`cfradial`]: CfRadial 1.x volumes into the FM301
//!   [`recast_radar_core::model::Volume`] ([`read_cfradial1_volume`]), from
//!   a classic file or a netCDF-4 file alike.
//! - [`cfradial2`]: CfRadial 2 / FM301 group-layout volumes
//!   ([`read_cfradial2_volume`]) as Radx and xradar write them.
//! - [`read_cfradial_volume`] takes either container and either layout;
//!   [`cfradial_layout`] tells the two netCDF-4 layouts apart by content
//!   (what the format router uses on an opened HDF5 file).
//! - [`netcdf3`]: the minimal read-only classic netCDF (CDF-1/CDF-2) parser;
//!   [`netcdf`]: one view over a classic file or a netCDF-4 group
//!   ([`recast_radar_hdf5::netcdf4`] underneath).
//!
//! Field storage stays as the file stores it: `byte`, `short` and `int`
//! (and netCDF-4 `ubyte`/`ushort`) with the file's CF packing; netCDF-4
//! `uint`/`int64`/`uint64` fields keep their codes as `f64` because the
//! model has no wider integer storage.
//!
//! # Limits
//!
//! The classic netCDF reader accepts at most 1,024 dimensions, 4,096
//! variables, 4,096 attributes per list, rank 32, 64 KiB names, 16 MiB
//! attribute values, dimension lengths and record counts of 104,857,600, and
//! 256 MiB per variable array (one record slab, and a whole record variable).
//! A record variable's full byte range is checked against the file before
//! its array is reserved.
//!
//! The CfRadial decoder accepts at most `MAX_GATES_PER_RADIAL` (16,384)
//! gates and `MAX_SWEEPS_PER_VOLUME` (1,024) sweeps, and charges every
//! numeric coordinate array it widens to f64, the ray tables and every
//! field's sweep rows (in the file's storage width) to a `DecodeBudget` of
//! `MAX_DECODED_VOLUME_BYTES` (1 GiB) before allocating (constants in
//! [`recast_radar_core::bounded_read`]), and the attributes it keeps verbatim
//! (by the memory they hold) to the same budget; the netCDF reader's own
//! 256 MiB per-variable cap bounds each full field array it reads. Groups
//! kept verbatim are read 16 deep. Sweeps must not share
//! rays: overlapping `sweep_start_ray_index`/`sweep_end_ray_index` ranges are
//! a [`CfRadialError::InvalidMessage`] error, so each ray's gates are copied
//! into at most one sweep and a moment's grids never outgrow its field. A
//! sweep whose ray index is not a non-negative integer (a fill value, for
//! example) is skipped.
//!
//! netCDF-4 files go through the HDF5 reader's limits (see
//! [`recast_radar_hdf5`]: 256 MiB per dataset, 16,384 objects, ...) and the
//! same `DecodeBudget`; a CfRadial 2 file may hold at most
//! `MAX_SWEEPS_PER_VOLUME` sweep groups and at most 4,096 calibration
//! entries.
//!
//! Every limit violation is a [`CfRadialError::LimitExceeded`] error. An
//! optional variable that is missing or malformed is ignored, but one that
//! exceeds a limit fails the decode.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

pub mod cfradial;
pub mod cfradial2;
pub mod netcdf;
pub mod netcdf3;
pub mod write;

use thiserror::Error;

pub use cfradial::read_cfradial1_volume;
pub use cfradial2::{is_cfradial2, read_cfradial2_volume};
pub use netcdf3::looks_like_netcdf3_bytes;
pub use recast_radar_hdf5::netcdf4::NcFile as Netcdf4File;
pub use write::{
    CfWriteError, Cfradial1Options, Cfradial2Options, RangeLayout, write_cfradial1, write_cfradial2,
};

use recast_radar_core::bounded_read::DecodeBudget;
use recast_radar_core::model::Volume;

/// The CfRadial layout of a netCDF-4 file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CfRadialLayout {
    /// CfRadial 1.x: `time` and `range` dimensions in the root group.
    CfRadial1,
    /// CfRadial 2 / FM301: one group per sweep.
    CfRadial2,
}

/// Which CfRadial layout a netCDF-4 file uses, by content: CfRadial 2 when
/// the root has a `sweep_group_name` variable or `sweep_<n>` groups,
/// CfRadial 1 when the root group defines `time` and `range` dimensions and
/// has an `azimuth` variable; `None` otherwise (ODIM_H5 and other HDF5
/// files).
pub fn cfradial_layout(file: &Netcdf4File<'_>) -> Option<CfRadialLayout> {
    if cfradial2::is_cfradial2(file) {
        return Some(CfRadialLayout::CfRadial2);
    }
    let root = file.root();
    let has_dim = |name: &str| {
        root.dims
            .iter()
            .any(|id| file.dim(*id).is_some_and(|dim| dim.name == name))
    };
    (has_dim("time") && has_dim("range") && root.variable("azimuth").is_some())
        .then_some(CfRadialLayout::CfRadial1)
}

/// Decode a netCDF-4 CfRadial file (either layout, by [`cfradial_layout`]).
pub fn read_cfradial_netcdf4(file: &Netcdf4File<'_>) -> Result<Volume> {
    match cfradial_layout(file) {
        Some(CfRadialLayout::CfRadial2) => cfradial2::decode(file, DecodeBudget::volume()),
        Some(CfRadialLayout::CfRadial1) => cfradial::decode_netcdf4(file, DecodeBudget::volume()),
        None => Err(CfRadialError::InvalidMessage {
            offset: 0,
            reason: "netCDF-4 file is neither CfRadial 1 (root time/range dimensions) nor \
                     CfRadial 2 (sweep groups)"
                .to_owned(),
        }),
    }
}

/// Decode CfRadial from bytes in any container: classic netCDF (CfRadial
/// 1), or netCDF-4 with either layout.
pub fn read_cfradial_volume(bytes: &[u8]) -> Result<Volume> {
    if recast_radar_hdf5::looks_like_hdf5_bytes(bytes) {
        return read_cfradial_netcdf4(&Netcdf4File::open(bytes)?);
    }
    read_cfradial1_volume(bytes)
}

/// Result type for CfRadial and netCDF decoding.
pub type Result<T> = std::result::Result<T, CfRadialError>;

/// Errors from CfRadial and classic netCDF decoding.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CfRadialError {
    /// A netCDF structure ended before its declared length.
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
    /// Structurally invalid or unsupported netCDF/CfRadial content.
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
    /// The rays of one sweep state different gate geometries
    /// (`range(time, range)` rows, or `ray_start_range` /
    /// `ray_gate_spacing`): the model has one range coordinate per sweep.
    #[error("CfRadial sweep {sweep}: its rays have different gate geometries")]
    PerRayGeometry {
        /// Index of the sweep in the file.
        sweep: usize,
    },
}

impl From<recast_radar_hdf5::Error> for CfRadialError {
    fn from(err: recast_radar_hdf5::Error) -> Self {
        match err {
            recast_radar_hdf5::Error::Truncated {
                what,
                offset,
                needed,
                available,
            } => Self::Truncated {
                what,
                offset,
                needed,
                available,
            },
            recast_radar_hdf5::Error::Invalid { offset, reason } => {
                Self::InvalidMessage { offset, reason }
            }
            recast_radar_hdf5::Error::LimitExceeded(reason) => Self::LimitExceeded(reason),
            other => Self::InvalidMessage {
                offset: 0,
                reason: other.to_string(),
            },
        }
    }
}
