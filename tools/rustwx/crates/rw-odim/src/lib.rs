//! EUMETNET OPERA ODIM_H5 polar-volume decoding.
//!
//! ODIM_H5 is the EUMETNET OPERA data information model: an HDF5 container
//! whose groups carry a radar's polar geometry as attributes and its moments
//! as `[ray, bin]` rectangles. It is what the MeteoGate OGC-EDR feed serves
//! per site, and -- unlike the 2-D LAEA composite on the same collection,
//! which is reflectivity only -- it is the product that carries **radial
//! velocity**. That is the whole reason this crate exists: `VRAD` is what
//! makes a European radar assimilable, and nothing upstream could read it.
//!
//! ```no_run
//! use rw_odim::{read_volume, censor};
//!
//! let volume = read_volume("deboo@20260812T2335@0.5@VRADH.h5")?;
//! for sweep in &volume.sweeps {
//!     let Some(vrad) = sweep.moment("VRADH") else { continue };
//!     println!(
//!         "{:.2} deg: {} gates measured, {} clear-air, Nyquist {:?} m/s",
//!         sweep.elevation_deg,
//!         vrad.census.measured,
//!         vrad.census.undetect,
//!         sweep.nyquist.interval_ms,
//!     );
//!     assert_eq!(vrad.censor.len(), vrad.values.len());
//!     assert!(vrad.value(0, 0).is_nan() || vrad.code(0, 0) == censor::MEASURED);
//! }
//! # Ok::<(), rw_odim::OdimError>(())
//! ```
//!
//! # The two things this crate is careful about
//!
//! **Sentinels are classified, never collapsed.** ODIM reserves two raw
//! values per moment, `undetect` and `nodata`, and they are opposites: one is
//! the radar reporting that it looked and found nothing -- a real observation
//! of clear air -- and the other is the radar reporting that it did not look.
//! Mapping both to one missing value throws away every correct negative in
//! the file. Here both become `NaN` in
//! [`Moment::values`](volume::Moment::values), and the *reason* is carried
//! beside them in [`Moment::censor`](volume::Moment::censor), one [`censor`]
//! code per gate. See that module for the code space, which is shared with
//! `wx_radar::level2::censor` so the two radar families can be read with one
//! table.
//!
//! **The Nyquist interval travels with the sweep.** Radial velocity is folded,
//! and a folded velocity field is not an observation until it is paired with
//! the interval it folded at. [`Nyquist`](volume::Nyquist) carries the value
//! *and* its provenance, and refuses to guess: a dual-PRF sweep with no
//! declared `NI` reports [`NyquistSource::Unavailable`](volume::NyquistSource)
//! rather than a single-PRF estimate that would be a factor of three small.
//!
//! # Scope
//!
//! Polar objects only: `/what/object` of `PVOL` or `SCAN`. Composites (`COMP`)
//! have a projected Cartesian geometry and are read by
//! `rustwx_io::extract_eumetnet_opera_dbzh_from_odim_h5`; vertical profiles
//! and RHIs are neither.

pub mod assemble;
pub mod attrs;
pub mod censor;
pub mod decode;
pub mod error;
pub mod pack;
pub mod quantity;
pub mod volume;

pub use assemble::{AssembleReport, MemberFile, VolumeGroup, assemble, odim_files_in, survey_dir};
pub use censor::Census;
pub use decode::{DecodeOptions, read_volume, read_volume_with};
pub use error::{OdimError, Result};
pub use quantity::{QuantityInfo, QuantityKind, describe, is_radial_velocity, is_reflectivity};
pub use volume::{
    AzimuthSource, Calibration, Moment, Nyquist, NyquistSource, PolarVolume, Site, Source, Sweep,
    SystemNotes,
};
