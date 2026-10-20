//! HYDRO-WORLD global parameter derivation (lane hydro-world-params).
//!
//! Per catchment, from open global inputs (SoilGrids 2.0, ESA WorldCover 2021,
//! Copernicus DEM GLO-90), the CFE, Noah-OWP-Modular and LGAR parameters, as
//! `model_attributes` rows of the HYDRO-WORLD fabric contract
//! (`arwen.hydro-world.fabric.v1`, crate `hydro-world`). Every source and recipe
//! is a row of `data/params.v1.json`; each formula cites its publication where
//! it is implemented. All stages run across all cores (Rayon).
pub mod catch;
pub mod grid;
pub mod proj;
pub mod run;
pub mod soil;
pub mod table;
pub mod terrain;
pub mod tiff;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;
