//! The machine-readable fetch record `rw_fetch` prints on stdout.
//!
//! Python owns the durable contracts -- the `gpuwm-fetch-manifest-v1`
//! manifest, the resume identity guard, the quarantine rule, the
//! record-count bars.  This record is the raw material those need: what
//! was fetched, from where, by which transport, with the exact byte
//! ranges and index identity behind it.  It deliberately mirrors the
//! `rw-wps.hrrr-native-range-download.v1` receipt's per-range detail so
//! the Python side can author that receipt without a second probe.

use serde::Serialize;

/// Schema of the document on stdout.
pub const FETCH_RECORD_SCHEMA: &str = "gpuwm-rw-fetch-record-v1";

/// Exact-ABI marker for `gpuwm.native_wrf_distribution`.
///
/// The distribution check greps the built binary for this byte string,
/// so a stale `rw_fetch` that still answers `--help` but no longer
/// emits one of these keys fails the check instead of failing later,
/// after a distribution has been built and installed.
pub const FETCH_RECORD_ABI: &str = "gpuwm-rw-fetch-record-v1\tmode\tmode_reason\tsource\
\tgrib_url\tidx_url\tidx_sha256\tidx_record_count\tselected_record_count\tranges\tsha256";

#[derive(Debug, Clone, Serialize)]
pub struct RangeRecord {
    pub start: u64,
    pub end: u64,
    pub bytes: u64,
    pub first_index_row: u32,
    pub last_index_row: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProbeRecord {
    pub object_bytes: Option<u64>,
    pub idx_declared: bool,
    pub idx_fetched: bool,
    pub idx_error: Option<String>,
    pub idx_record_count: usize,
    pub idx_last_offset: Option<u64>,
    pub idx_last_message_bytes: Option<u64>,
    /// `null` when coverage could not be proven either way.
    pub idx_covers_object: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileRecord {
    pub forecast_hour: u16,
    pub name: String,
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub source: String,
    pub grib_url: String,
    pub idx_url: Option<String>,
    pub mode: String,
    pub mode_reason: String,
    pub probe: ProbeRecord,
    pub idx_name: Option<String>,
    pub idx_sha256: Option<String>,
    pub idx_bytes: Option<u64>,
    pub idx_record_count: Option<usize>,
    pub selected_record_count: Option<usize>,
    pub ranges: Vec<RangeRecord>,
    pub etag: Option<String>,
    pub last_modified: Option<String>,
    pub wall_seconds: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CycleRecord {
    pub date: String,
    pub hour: u8,
}

#[derive(Debug, Clone, Serialize)]
pub struct FetchRecord {
    pub schema: &'static str,
    pub tool: &'static str,
    pub tool_version: &'static str,
    pub model: String,
    pub product: String,
    pub cycle: CycleRecord,
    pub requested_mode: String,
    pub requested_source: Option<String>,
    pub var_pattern_count: usize,
    pub out_dir: String,
    pub cache_dir: Option<String>,
    pub files: Vec<FileRecord>,
    pub payload_bytes: u64,
    pub wall_seconds: f64,
}

/// The `probe` subcommand's document: facts, no bytes moved.
#[derive(Debug, Clone, Serialize)]
pub struct ProbeReport {
    pub schema: &'static str,
    pub tool: &'static str,
    pub model: String,
    pub product: String,
    pub cycle: CycleRecord,
    pub requested_mode: String,
    pub var_pattern_count: usize,
    pub hours: Vec<ProbeHour>,
}

pub const PROBE_REPORT_SCHEMA: &str = "gpuwm-rw-fetch-probe-v1";

#[derive(Debug, Clone, Serialize)]
pub struct ProbeHour {
    pub forecast_hour: u16,
    pub source: Option<String>,
    pub grib_url: Option<String>,
    pub idx_url: Option<String>,
    pub probe: Option<ProbeRecord>,
    /// `"full-file"`, `"idx-subset"`, or `null` when the object is not
    /// fetchable at all.
    pub mode: Option<String>,
    pub mode_reason: String,
}

/// The `latest` subcommand's document.
pub const LATEST_REPORT_SCHEMA: &str = "gpuwm-rw-fetch-latest-v1";

#[derive(Debug, Clone, Serialize)]
pub struct LatestReport {
    pub schema: &'static str,
    pub tool: &'static str,
    pub model: String,
    pub product: String,
    pub through_forecast_hour: u16,
    pub cycle: Option<CycleRecord>,
    pub source: Option<String>,
    pub probed: Vec<String>,
}
