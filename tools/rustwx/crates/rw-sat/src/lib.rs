//! Continuous GOES ABI satellite ingest for the rusty-weather platform.
//!
//! The science/IO core (`goes`, `geostationary`, `netcdf`, `abi`, the pure
//! half of `composite`) is the production satellite stack extracted from
//! the old rustwx codebase, unchanged; `palette` ports its render scales as
//! plain anchor data. Around it: `s3` (anonymous paginated/incremental
//! listing of the NOAA open-data buckets), `store` (decoded fields ->
//! rw-store frame files, see its module docs for the on-disk convention),
//! `window` (rolling max-age/max-bytes eviction), `follow` (the live
//! polling engine with typed events and cancellation, mirroring
//! rw-ingest), and `export` (palette PNG quick-looks).

pub mod abi;
pub mod cloud;
pub mod composite;
pub mod cwp;
pub mod events;
pub mod export;
pub mod follow;
pub mod geostationary;
pub mod goes;
pub mod netcdf;
pub mod palette;
pub mod s3;
pub mod store;
pub mod window;

pub use cloud::{
    CloudProduct, CloudProductField, DqfReport, DqfRule, read_cloud_product_field,
    read_cloud_product_field_window,
};
pub use cwp::{CloudPhase, CwpCounts, cloud_water_path_g_m2, cloud_water_path_plane};
pub use events::{NEVER_CANCEL, SatError, SatEvent, print_event};
pub use follow::{FollowConfig, FollowSummary, follow};
pub use store::{StoredFrame, WrittenFrame, read_frame, write_band_frame};
pub use window::{EvictionReport, WindowConfig, enforce_window};
