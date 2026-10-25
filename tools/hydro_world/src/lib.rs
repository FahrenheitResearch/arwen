//! HYDRO-WORLD fabric contract (`arwen.hydro-world.fabric.v1`).
//!
//! The contract is the exact NextGen-shaped GeoPackage the engine's
//! `rw_hydrofabric prepare` consumes (ngen v2.0.1 table shape, ngen parser pin
//! be11a180, t-route pin 12a8eae0) plus the `model_attributes` and
//! `fabric_metadata` tables every HYDRO-WORLD builder adds. Any builder (the
//! CONUS NHDPlus recipe, the global HydroSHEDS recipe) writes these structs
//! through `gpkg::write`, so the engine cannot tell the builders apart.
//! See docs/dev/HYDRO-WORLD-CONTRACT.md for every column, unit and the ID scheme.
pub mod build;
pub mod check;
pub mod compare;
pub mod contract;
pub mod derive;
pub mod flow;
pub mod geo;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod gpkg;
pub mod grid;
pub mod polygonize;
pub mod parent_grid;
pub mod render;
pub mod shp;
pub mod source;
mod sqlite;
pub mod tiff;
pub mod vector;
pub mod native;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;
