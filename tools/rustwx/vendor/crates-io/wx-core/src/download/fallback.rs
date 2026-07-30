//! Multi-source fallback download logic.
//!
//! Tries downloading from multiple data sources in priority order,
//! falling back to the next source if one fails. This mirrors Herbie's
//! behavior of trying AWS, then Google Cloud, then NOMADS, etc.

use super::client::DownloadClient;
use super::idx::{self, IdxEntry};
use super::sources::{self, DataSource};

/// Result of a successful fallback download.
pub struct FetchResult {
    /// The downloaded GRIB2 data bytes.
    pub data: Vec<u8>,
    /// Name of the source that succeeded (e.g., "aws", "google").
    pub source_name: String,
    /// The GRIB URL that was actually fetched.
    pub url: String,
}

/// Try downloading GRIB2 data from multiple sources in priority order.
///
/// For each source:
/// 1. If the source has .idx files and `vars` is specified, fetch the .idx,
///    compute byte ranges for the requested variables, and download only
///    those ranges.
/// 2. If .idx is not available or `vars` is `None`, download the full file.
/// 3. If the download fails, log a warning and try the next source.
///
/// Returns the data and the name of the source that succeeded.
///
/// # Arguments
///
/// * `client` - HTTP download client
/// * `model` - Model name (e.g., "hrrr", "gfs")
/// * `date` - Date string in YYYYMMDD format
/// * `hour` - Model initialization hour
/// * `product` - Product type (e.g., "sfc", "prs")
/// * `fhour` - Forecast hour
/// * `vars` - Optional variable patterns for partial download
/// * `forced_source` - If `Some`, only try this specific source
pub fn fetch_with_fallback(
    client: &DownloadClient,
    model: &str,
    date: &str,
    hour: u32,
    product: &str,
    fhour: u32,
    vars: Option<&[&str]>,
    forced_source: Option<&str>,
) -> crate::error::Result<FetchResult> {
    let sources = sources::model_sources_filtered(model, forced_source);

    if sources.is_empty() {
        if let Some(src) = forced_source {
            return Err(crate::RustmetError::InvalidArgument(
                format!(
                    "Source '{}' not available for model '{}'. Available: {:?}",
                    src,
                    model,
                    sources::source_names(model)
                ),
            ));
        }
        return Err(crate::RustmetError::ModelNotFound(
            format!(
                "No data sources configured for model '{}'. Supported: hrrr, gfs, nam, rap, ecmwf, nbm, rrfs, rtma, href",
                model
            ),
        ));
    }

    let mut errors: Vec<String> = Vec::new();

    for source in &sources {
        match try_source(client, source, date, hour, product, fhour, vars) {
            Ok(result) => return Ok(result),
            Err(e) => {
                let msg = format!("[{}] {}", source.name, e);
                eprintln!("  Source '{}' failed: {}", source.name, e);
                errors.push(msg);
            }
        }
    }

    // All sources failed
    Err(crate::RustmetError::Http(format!(
        "All sources failed for {model} {date}/{hour:02}z f{fhour:03}:\n  {}",
        errors.join("\n  ")
    )))
}

/// Try downloading from a single source, with .idx-based partial download
/// when variables are specified.
fn try_source(
    client: &DownloadClient,
    source: &DataSource,
    date: &str,
    hour: u32,
    product: &str,
    fhour: u32,
    vars: Option<&[&str]>,
) -> crate::error::Result<FetchResult> {
    let grib_url = source.grib_url(date, hour, product, fhour);

    // If we have variable patterns and the source has .idx files,
    // try a partial (byte-range) download.
    if let Some(var_patterns) = vars {
        if let Some(idx_url) = source.idx_url(date, hour, product, fhour) {
            match try_idx_download(client, &idx_url, &grib_url, var_patterns) {
                Ok(data) => {
                    return Ok(FetchResult {
                        data,
                        source_name: source.name.to_string(),
                        url: grib_url,
                    });
                }
                Err(e) => {
                    // .idx approach failed -- fall through to full download
                    eprintln!(
                        "  Source '{}': .idx partial download failed ({}), trying full file...",
                        source.name, e
                    );
                }
            }
        }
    }

    // Full file download (either no vars requested, no .idx, or .idx failed)
    let data = client.get_bytes(&grib_url)?;
    Ok(FetchResult {
        data,
        source_name: source.name.to_string(),
        url: grib_url,
    })
}

/// Try downloading specific variables using the .idx file for byte-range selection.
fn try_idx_download(
    client: &DownloadClient,
    idx_url: &str,
    grib_url: &str,
    var_patterns: &[&str],
) -> crate::error::Result<Vec<u8>> {
    let idx_text = client.get_text(idx_url)?;
    let idx_entries = idx::parse_idx(&idx_text);

    if idx_entries.is_empty() {
        return Err(crate::RustmetError::NoData(
            "Empty or unparseable .idx file".to_string(),
        ));
    }

    let mut selected: Vec<&IdxEntry> = Vec::new();
    for pat in var_patterns {
        for entry in idx::find_entries(&idx_entries, pat) {
            if !selected.iter().any(|e| e.byte_offset == entry.byte_offset) {
                selected.push(entry);
            }
        }
    }

    if selected.is_empty() {
        return Err(crate::RustmetError::NoData(format!(
            "No matching variables for patterns: {:?}",
            var_patterns,
        )));
    }

    let ranges = idx::byte_ranges(&idx_entries, &selected);
    client.get_ranges(grib_url, &ranges)
}

/// Probe which sources are currently available for a model/date/hour.
///
/// Sends a HEAD request to each source's GRIB URL for fhour=0 and
/// reports whether the file exists. Useful for diagnostics and for
/// choosing the best source before a batch download.
///
/// Returns a list of (source_name, is_available) tuples.
pub fn probe_sources(
    client: &DownloadClient,
    model: &str,
    date: &str,
    hour: u32,
    product: &str,
) -> Vec<(String, bool)> {
    let sources = sources::model_sources(model);
    sources
        .iter()
        .map(|src| {
            let url = src.grib_url(date, hour, product, 0);
            let available = client.head_ok(&url);
            (src.name.to_string(), available)
        })
        .collect()
}
