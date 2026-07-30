//! GRIB Edition 1 parser and unpacker.
//!
//! This module provides complete support for reading GRIB1 files, including:
//! - Full section parsing (Indicator, PDS, GDS, BMS, BDS, End)
//! - IBM 32-bit floating point to IEEE 754 conversion
//! - Simple packing data unpacking
//! - Grid coordinate generation for lat/lon and Lambert conformal projections
//! - WMO standard parameter table 2 lookups

pub mod grid;
pub mod parser;
pub mod tables;
pub mod unpack;

pub use grid::{grid_coordinates, LatLon};
pub use parser::{
    BinaryDataSection, BitMapSection, Grib1File, Grib1Message, GridDescriptionSection, GridType,
    IndicatorSection, LevelType, ProductDefinitionSection,
};
pub use tables::{level_description, parameter_abbrev, parameter_name, parameter_units};
pub use unpack::{ibm_to_ieee, unpack_bds};
