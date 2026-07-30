//! wx-core — Pure Rust GRIB2 processor and weather model data library.
//!
//! Provides GRIB2 parsing, operational model downloads (HRRR, GFS, NAM, RAP),
//! meteorological computations, and map projections. No rendering dependencies.
//!
//! # Quick Start
//!
//! ```no_run
//! # #[cfg(feature = "network")]
//! # {
//! use wx_core::{grib2, download, models};
//!
//! // Create HTTP client and download HRRR 2m temperature
//! let client = download::DownloadClient::new().unwrap();
//! let url = models::hrrr::HrrrConfig::aws_url("20240115", 12, "sfc", 0);
//! let idx_url = format!("{}.idx", url);
//!
//! // Parse the .idx file to find byte ranges
//! let idx_text = client.get_text(&idx_url).unwrap();
//! let entries = download::parse_idx(&idx_text);
//! let matches = download::find_entries(&entries, "TMP:2 m above ground");
//! let ranges = download::byte_ranges(&entries, &matches);
//!
//! // Download just the needed bytes and parse GRIB2
//! let data = client.get_ranges(&url, &ranges).unwrap();
//! let grib = grib2::Grib2File::from_bytes(&data).unwrap();
//!
//! // Unpack the data values
//! let values = grib2::unpack_message(&grib.messages[0]).unwrap();
//! // values is now a Vec<f64> of temperatures in Kelvin
//! # }
//! ```

pub mod error;
pub mod grib2;
#[cfg(feature = "network")]
pub mod download;
pub mod models;
pub mod products;
pub mod projection;
pub mod metfuncs;
pub mod dynamics;
pub mod gridmath;
pub mod composite;
pub mod regrid;
pub mod render;

pub use error::{RustmetError, Result};
