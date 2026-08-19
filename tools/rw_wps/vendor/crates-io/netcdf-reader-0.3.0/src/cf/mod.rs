//! CF Conventions support for NetCDF files.
//!
//! This module provides helpers for interpreting NetCDF data according to
//! the [CF (Climate and Forecast) Conventions](https://cfconventions.org/):
//!
//! - **Axis identification** (`axes`): Determines coordinate variable roles
//!   (T, X, Y, Z) from `axis`, `standard_name`, `units`, and `positive` attributes.
//! - **CRS extraction** (`crs`): Parses `grid_mapping` attributes to extract
//!   projection parameters and identify EPSG codes.
//! - **Time decoding** (`time`): Parses CF time units strings and converts
//!   numeric values to `chrono::DateTime` objects with calendar support.
//! - **Bounds variables** (`bounds`): Resolves cell boundary variables from
//!   the `bounds` attribute on coordinate variables.

pub mod axes;
pub mod bounds;
pub mod crs;
pub mod time;
