//! The wrfout import lane and the panel/overlay seam, as a library.
//!
//! `main.rs` (the `rw_wrfbatch` binary) keeps its own `#[path]` module
//! declarations and is not built from this target -- adding a library here
//! is purely additive, and the renderer binary's bytes are unaffected by
//! anything in this file.  What the library exists for is the two SIBLING
//! binaries in this crate:
//!
//! * `rw_ensbatch` -- ensemble products (mean, spread, NMEP, PMM,
//!   paintball) from N member wrfouts at one valid time;
//! * `rw_obsgrid` -- the `gpuwm-obs.radar-grid` observation grid (v1 and
//!   the windowed v2), read natively rather than disguised as a forecast.
//!
//! Both need the same three things `rw_wrfbatch` already has: the hardened
//! wrfout import ([`wrf_process`]), a way to put an arbitrary 2-D plane
//! through the production render path ([`panel`]), and a way to put a
//! place known in degrees onto that panel ([`annotate`]).

pub mod annotate;
pub mod panel;

#[path = "grib_import.rs"]
pub mod grib_import;
#[path = "local_import.rs"]
pub mod local_import;
#[path = "postproc_severe.rs"]
pub mod postproc_severe;
#[path = "wrf_process.rs"]
pub mod wrf_process;
#[path = "wrf_volumes.rs"]
pub mod wrf_volumes;

pub mod obs_grid;
pub mod scales;
