//! wx-math — Meteorological computations and grid math.
//!
//! Pure-math meteorological functions covering thermodynamics, grid dynamics,
//! stability indices, and interpolation/regridding. No I/O or rendering dependencies.

pub mod thermo;
pub mod dynamics;
pub mod gridmath;
pub mod composite;
pub mod regrid;
pub mod interpolate;
