pub mod compute;
pub mod diag;
pub mod error;
pub mod extract;
pub mod file;
pub mod grid;
pub mod met;
pub mod multi;
pub mod projection;
pub mod units;
pub mod variables;

#[cfg(feature = "pure-rust-reader")]
pub mod classic_netcdf_reader;
#[cfg(feature = "pure-rust-reader")]
pub mod hdf5_reader;
#[cfg(feature = "pure-rust-reader")]
pub mod pure_reader;

pub use compute::{
    getvar, getvar_all_times, ComputeOpts, StormMotion, StormMotionMethod, VarOutput,
};
pub use error::{WrfError, WrfResult};
pub use file::WrfFile;
pub use projection::WrfProjection;
pub use units::WrfUnits;
