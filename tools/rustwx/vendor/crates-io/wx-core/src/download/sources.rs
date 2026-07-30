//! Data source definitions for weather models.
//!
//! Each model has an ordered list of data sources (cloud providers, NOMADS, etc.)
//! that can serve its GRIB2 files. Sources are tried in priority order during
//! fallback downloads, matching Herbie's multi-source approach.

use crate::models;

/// A data source for downloading model GRIB2 files.
///
/// Each source knows how to build GRIB and IDX URLs for a given
/// model run and forecast hour.
pub struct DataSource {
    /// Human-readable source name (e.g., "aws", "google", "nomads").
    pub name: &'static str,
    /// Function that builds the GRIB2 file URL.
    pub url_fn: fn(date: &str, hour: u32, product: &str, fhour: u32) -> String,
    /// Function that builds the .idx index URL, if available.
    pub idx_fn: Option<fn(date: &str, hour: u32, product: &str, fhour: u32) -> String>,
    /// Whether this source has .idx files available.
    pub idx_available: bool,
    /// Priority (lower = try first).
    pub priority: u8,
    /// Maximum age of data in hours, if the source only keeps recent data.
    /// `None` means full archive is available.
    pub max_age_hours: Option<u32>,
}

impl DataSource {
    /// Build the GRIB2 URL for this source.
    pub fn grib_url(&self, date: &str, hour: u32, product: &str, fhour: u32) -> String {
        (self.url_fn)(date, hour, product, fhour)
    }

    /// Build the .idx URL for this source, if .idx files are available.
    pub fn idx_url(&self, date: &str, hour: u32, product: &str, fhour: u32) -> Option<String> {
        if self.idx_available {
            if let Some(idx_fn) = self.idx_fn {
                Some(idx_fn(date, hour, product, fhour))
            } else {
                // Default: append .idx to the GRIB URL
                Some(format!("{}.idx", self.grib_url(date, hour, product, fhour)))
            }
        } else {
            None
        }
    }
}

// ──────────────────────────────────────────────────────────
// HRRR sources
// ──────────────────────────────────────────────────────────

fn hrrr_aws_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
    models::HrrrConfig::aws_url(date, hour, product, fhour)
}

fn hrrr_google_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
    let product_code = match product {
        "sfc" | "surface" | "wrfsfc" => "wrfsfc",
        "prs" | "pressure" | "wrfprs" => "wrfprs",
        "nat" | "native" | "wrfnat" => "wrfnat",
        "subh" | "subhourly" | "wrfsubh" => "wrfsubh",
        _ => "wrfsfc",
    };
    format!(
        "https://storage.googleapis.com/high-resolution-rapid-refresh/hrrr.{}/conus/hrrr.t{:02}z.{}f{:02}.grib2",
        date, hour, product_code, fhour
    )
}

fn hrrr_nomads_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
    models::HrrrConfig::nomads_url(date, hour, product, fhour)
}

fn hrrr_azure_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
    let product_code = match product {
        "sfc" | "surface" | "wrfsfc" => "wrfsfc",
        "prs" | "pressure" | "wrfprs" => "wrfprs",
        "nat" | "native" | "wrfnat" => "wrfnat",
        "subh" | "subhourly" | "wrfsubh" => "wrfsubh",
        _ => "wrfsfc",
    };
    format!(
        "https://noaahrrr.blob.core.windows.net/hrrr/hrrr.{}/conus/hrrr.t{:02}z.{}f{:02}.grib2",
        date, hour, product_code, fhour
    )
}

fn hrrr_sources() -> Vec<DataSource> {
    vec![
        DataSource {
            name: "nomads",
            url_fn: hrrr_nomads_url,
            idx_fn: None,
            idx_available: true,
            priority: 1,
            max_age_hours: Some(48),
        },
        DataSource {
            name: "aws",
            url_fn: hrrr_aws_url,
            idx_fn: None,
            idx_available: true,
            priority: 2,
            max_age_hours: None,
        },
        DataSource {
            name: "google",
            url_fn: hrrr_google_url,
            idx_fn: None,
            idx_available: true,
            priority: 3,
            max_age_hours: None,
        },
        DataSource {
            name: "azure",
            url_fn: hrrr_azure_url,
            idx_fn: None,
            idx_available: false,
            priority: 4,
            max_age_hours: None,
        },
    ]
}

// ──────────────────────────────────────────────────────────
// GFS sources
// ──────────────────────────────────────────────────────────

fn gfs_aws_url(date: &str, hour: u32, _product: &str, fhour: u32) -> String {
    models::GfsConfig::aws_url(date, hour, fhour)
}

fn gfs_google_url(date: &str, hour: u32, _product: &str, fhour: u32) -> String {
    format!(
        "https://storage.googleapis.com/global-forecast-system/gfs.{}/{:02}/atmos/gfs.t{:02}z.pgrb2.0p25.f{:03}",
        date, hour, hour, fhour
    )
}

fn gfs_nomads_url(date: &str, hour: u32, _product: &str, fhour: u32) -> String {
    models::GfsConfig::nomads_url(date, hour, fhour)
}

fn gfs_ncei_url(date: &str, hour: u32, _product: &str, fhour: u32) -> String {
    // NCEI historical archive
    let year = &date[..4];
    let month = &date[4..6];
    let day = &date[6..8];
    format!(
        "https://www.ncei.noaa.gov/data/global-forecast-system/access/grid-004-0.5-degree/analysis/{}{}/{}{}{}/gfs_4_{}{}{}_{}00_{:03}.grb2",
        year, month, year, month, day, year, month, day,
        format_args!("{:02}", hour), fhour
    )
}

fn gfs_sources() -> Vec<DataSource> {
    vec![
        DataSource {
            name: "nomads",
            url_fn: gfs_nomads_url,
            idx_fn: None,
            idx_available: true,
            priority: 1,
            max_age_hours: Some(48),
        },
        DataSource {
            name: "aws",
            url_fn: gfs_aws_url,
            idx_fn: None,
            idx_available: true,
            priority: 2,
            max_age_hours: None,
        },
        DataSource {
            name: "google",
            url_fn: gfs_google_url,
            idx_fn: None,
            idx_available: true,
            priority: 3,
            max_age_hours: None,
        },
        DataSource {
            name: "ncei",
            url_fn: gfs_ncei_url,
            idx_fn: None,
            idx_available: false,
            priority: 4,
            max_age_hours: None,
        },
    ]
}

// ──────────────────────────────────────────────────────────
// NAM sources
// ──────────────────────────────────────────────────────────

fn nam_nomads_url(date: &str, hour: u32, _product: &str, fhour: u32) -> String {
    models::NamConfig::nomads_url(date, hour, fhour)
}

fn nam_aws_url(date: &str, hour: u32, _product: &str, fhour: u32) -> String {
    models::NamConfig::aws_url(date, hour, fhour)
}

fn nam_sources() -> Vec<DataSource> {
    vec![
        DataSource {
            name: "nomads",
            url_fn: nam_nomads_url,
            idx_fn: None,
            idx_available: true,
            priority: 1,
            max_age_hours: Some(48),
        },
        DataSource {
            name: "aws",
            url_fn: nam_aws_url,
            idx_fn: None,
            idx_available: true,
            priority: 2,
            max_age_hours: None,
        },
    ]
}

// ──────────────────────────────────────────────────────────
// RAP sources
// ──────────────────────────────────────────────────────────

fn rap_aws_url(date: &str, hour: u32, _product: &str, fhour: u32) -> String {
    models::RapConfig::aws_url(date, hour, fhour)
}

fn rap_nomads_url(date: &str, hour: u32, _product: &str, fhour: u32) -> String {
    models::RapConfig::nomads_url(date, hour, fhour)
}

fn rap_sources() -> Vec<DataSource> {
    vec![
        DataSource {
            name: "nomads",
            url_fn: rap_nomads_url,
            idx_fn: None,
            idx_available: true,
            priority: 1,
            max_age_hours: Some(48),
        },
        DataSource {
            name: "aws",
            url_fn: rap_aws_url,
            idx_fn: None,
            idx_available: true,
            priority: 2,
            max_age_hours: None,
        },
    ]
}

// ──────────────────────────────────────────────────────────
// ECMWF source (single source)
// ──────────────────────────────────────────────────────────

fn ecmwf_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
    models::EcmwfConfig::open_data_url(date, hour, product, fhour)
}

fn ecmwf_sources() -> Vec<DataSource> {
    vec![DataSource {
        name: "ecmwf",
        url_fn: ecmwf_url,
        idx_fn: None,
        idx_available: true,
        priority: 1,
        max_age_hours: None,
    }]
}

// ──────────────────────────────────────────────────────────
// NBM source
// ──────────────────────────────────────────────────────────

fn nbm_aws_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
    models::NbmConfig::aws_url(date, hour, product, fhour)
}

fn nbm_sources() -> Vec<DataSource> {
    vec![DataSource {
        name: "aws",
        url_fn: nbm_aws_url,
        idx_fn: None,
        idx_available: true,
        priority: 1,
        max_age_hours: None,
    }]
}

// ──────────────────────────────────────────────────────────
// RRFS source
// ──────────────────────────────────────────────────────────

fn rrfs_nomads_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
    models::RrfsConfig::nomads_url(date, hour, product, fhour)
}

fn rrfs_sources() -> Vec<DataSource> {
    vec![DataSource {
        name: "nomads",
        url_fn: rrfs_nomads_url,
        idx_fn: None,
        idx_available: true,
        priority: 1,
        max_age_hours: Some(48),
    }]
}

// ──────────────────────────────────────────────────────────
// RTMA source
// ──────────────────────────────────────────────────────────

fn rtma_aws_url(date: &str, hour: u32, product: &str, _fhour: u32) -> String {
    models::RtmaConfig::aws_url(date, hour, product)
}

fn rtma_sources() -> Vec<DataSource> {
    vec![DataSource {
        name: "aws",
        url_fn: rtma_aws_url,
        idx_fn: None,
        idx_available: true,
        priority: 1,
        max_age_hours: None,
    }]
}

// ──────────────────────────────────────────────────────────
// HREF source
// ──────────────────────────────────────────────────────────

fn href_aws_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
    models::HrefConfig::aws_url(date, hour, product, fhour)
}

fn href_sources() -> Vec<DataSource> {
    vec![DataSource {
        name: "aws",
        url_fn: href_aws_url,
        idx_fn: None,
        idx_available: true,
        priority: 1,
        max_age_hours: None,
    }]
}

// ──────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────

/// Get the ordered list of data sources for a model.
///
/// Sources are returned sorted by priority (lowest number first).
/// Returns an empty vec for unknown models.
pub fn model_sources(model: &str) -> Vec<DataSource> {
    let mut sources = match model.to_lowercase().as_str() {
        "hrrr" => hrrr_sources(),
        "gfs" => gfs_sources(),
        "nam" => nam_sources(),
        "rap" => rap_sources(),
        "ecmwf" | "ifs" => ecmwf_sources(),
        "nbm" | "blend" => nbm_sources(),
        "rrfs" => rrfs_sources(),
        "rtma" => rtma_sources(),
        "href" => href_sources(),
        _ => vec![],
    };
    sources.sort_by_key(|s| s.priority);
    sources
}

/// Get sources for a model, filtered to only a specific source name.
///
/// If `source_name` is `None`, returns all sources in priority order.
/// If specified, returns only the matching source (or empty if not found).
pub fn model_sources_filtered(model: &str, source_name: Option<&str>) -> Vec<DataSource> {
    let sources = model_sources(model);
    match source_name {
        None => sources,
        Some(name) => {
            let lower = name.to_lowercase();
            sources.into_iter().filter(|s| s.name == lower).collect()
        }
    }
}

/// List all known source names for a model.
pub fn source_names(model: &str) -> Vec<&'static str> {
    model_sources(model).into_iter().map(|s| s.name).collect()
}
