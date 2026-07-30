use crate::{FetchRequest, FetchResult, IoError};
use rustwx_core::{
    CanonicalField, FieldProduct, FieldSelector, GridProjection, GridShape, LatLonGrid,
    SelectedField2D, SourceId, VerticalSelector,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

const FETCH_METADATA_SCHEMA_VERSION: u32 = 2;
const GRID_PAYLOAD_SCHEMA_VERSION: u32 = 2;
const FIELD_PAYLOAD_SCHEMA_VERSION: u32 = 4;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedFetchMetadata {
    pub request: rustwx_core::ModelRunRequest,
    pub source_override: Option<rustwx_core::SourceId>,
    pub variable_patterns: Vec<String>,
    pub resolved_source: rustwx_core::SourceId,
    pub resolved_url: String,
    pub resolved_family: String,
    pub bytes_len: usize,
    pub bytes_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedFetchResult {
    pub result: FetchResult,
    pub cache_hit: bool,
    pub bytes_path: PathBuf,
    pub metadata_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CachedFieldResult {
    pub field: SelectedField2D,
    pub cache_hit: bool,
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct CachedGridPayload {
    shape: GridShape,
    lat_deg: Vec<f32>,
    lon_deg: Vec<f32>,
    projection: Option<GridProjection>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct CachedFieldPayload {
    selector: CachedFieldSelector,
    units: String,
    values: Vec<f32>,
    grid_key: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct CachedFieldSelector {
    field: CanonicalField,
    vertical: VerticalSelector,
    product: FieldProduct,
}

impl CachedFieldSelector {
    fn from_selector(selector: FieldSelector) -> Self {
        Self {
            field: selector.field,
            vertical: selector.vertical,
            product: selector.product,
        }
    }

    fn as_selector(self) -> FieldSelector {
        FieldSelector {
            field: self.field,
            vertical: self.vertical,
            product: self.product,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct VersionedJsonPayload<T> {
    schema_version: u32,
    payload: T,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct VersionedBinaryPayload<T> {
    schema_version: u32,
    payload: T,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct LegacyCachedFetchMetadata {
    request: rustwx_core::ModelRunRequest,
    source_override: Option<rustwx_core::SourceId>,
    variable_patterns: Vec<String>,
    resolved_source: rustwx_core::SourceId,
    resolved_url: String,
    bytes_len: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct LegacyCachedGridPayload {
    shape: GridShape,
    lat_deg: Vec<f32>,
    lon_deg: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct LegacySelectedField2D {
    selector: LegacyFieldSelector,
    units: String,
    grid: LatLonGrid,
    values: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct LegacyFieldSelector {
    field: CanonicalField,
    vertical: VerticalSelector,
}

impl LegacyFieldSelector {
    fn as_selector(self) -> FieldSelector {
        FieldSelector {
            field: self.field,
            vertical: self.vertical,
            product: FieldProduct::Default,
        }
    }
}

pub fn artifact_cache_dir(cache_root: &Path, fetch: &FetchRequest) -> PathBuf {
    artifact_cache_dir_with_forecast_component(
        cache_root,
        fetch,
        &format!("f{:03}", fetch.request.forecast_hour),
    )
}

fn fetch_artifact_cache_dir(cache_root: &Path, fetch: &FetchRequest) -> PathBuf {
    let forecast_component = if fetch_uses_shared_forecast_cache(fetch) {
        "all_forecast_hours".to_string()
    } else {
        format!("f{:03}", fetch.request.forecast_hour)
    };
    artifact_cache_dir_with_forecast_component(cache_root, fetch, &forecast_component)
}

fn artifact_cache_dir_with_forecast_component(
    cache_root: &Path,
    fetch: &FetchRequest,
    forecast_component: &str,
) -> PathBuf {
    let product = sanitize_component(&fetch.request.product);
    let source = sanitize_component(
        fetch
            .source_override
            .map(|source| source.as_str())
            .unwrap_or("auto"),
    );
    let variable_slug = variable_patterns_slug(&fetch.variable_patterns);
    cache_root
        .join(sanitize_component(fetch.request.model.as_str()))
        .join(&fetch.request.cycle.date_yyyymmdd)
        .join(format!("{:02}z", fetch.request.cycle.hour_utc))
        .join(forecast_component)
        .join(product)
        .join(source)
        .join(variable_slug)
}

pub fn fetch_cache_paths(cache_root: &Path, fetch: &FetchRequest) -> (PathBuf, PathBuf) {
    let root = fetch_artifact_cache_dir(cache_root, fetch);
    (root.join("fetch.grib2"), root.join("fetch_meta.json"))
}

pub fn raw_fetch_cache_paths(
    cache_root: &Path,
    source: SourceId,
    resolved_url: &str,
) -> (PathBuf, PathBuf) {
    let url_hash = sha256_hex(resolved_url.as_bytes());
    let root = cache_root
        .join("_raw_fetch")
        .join(sanitize_component(source.as_str()))
        .join(&url_hash[..16]);
    (root.join("fetch.grib2"), root.join("fetch_meta.json"))
}

pub fn field_cache_path(
    cache_root: &Path,
    fetch: &FetchRequest,
    selector: FieldSelector,
) -> PathBuf {
    artifact_cache_dir(cache_root, fetch)
        .join("fields")
        .join(format!("{}.bin", sanitize_component(&selector.key())))
}

fn grid_cache_path(cache_root: &Path, fetch: &FetchRequest, grid_key: &str) -> PathBuf {
    artifact_cache_dir(cache_root, fetch)
        .join("fields")
        .join("grids")
        .join(format!("{grid_key}.bin"))
}

pub fn load_cached_fetch(
    cache_root: &Path,
    fetch: &FetchRequest,
) -> Result<Option<CachedFetchResult>, IoError> {
    let (bytes_path, metadata_path) = fetch_cache_paths(cache_root, fetch);
    if !bytes_path.exists() || !metadata_path.exists() {
        if bytes_path.exists() || metadata_path.exists() {
            quarantine_cache_paths(&[&bytes_path, &metadata_path], "incomplete_fetch_cache");
        }
        return Ok(None);
    }
    let bytes = match fs::read(&bytes_path) {
        Ok(bytes) => bytes,
        Err(_) => {
            quarantine_cache_paths(&[&bytes_path, &metadata_path], "fetch_bytes_read_error");
            return Ok(None);
        }
    };
    let metadata_bytes = match fs::read(&metadata_path) {
        Ok(bytes) => bytes,
        Err(_) => {
            quarantine_cache_paths(&[&bytes_path, &metadata_path], "fetch_metadata_read_error");
            return Ok(None);
        }
    };
    let Some(metadata) =
        load_cached_fetch_metadata(&metadata_bytes, &bytes, &fetch.request.product)
    else {
        quarantine_cache_paths(
            &[&bytes_path, &metadata_path],
            "fetch_metadata_decode_error",
        );
        return Ok(None);
    };
    if metadata.bytes_len != bytes.len()
        || !cached_fetch_request_matches(&metadata.request, &fetch.request, fetch)
        || metadata.source_override != fetch.source_override
        || metadata.variable_patterns != fetch.variable_patterns
        || metadata.resolved_family != fetch.request.product
        || metadata.bytes_sha256 != sha256_hex(&bytes)
    {
        quarantine_cache_paths(&[&bytes_path, &metadata_path], "fetch_metadata_mismatch");
        return Ok(None);
    }
    Ok(Some(CachedFetchResult {
        result: FetchResult {
            source: metadata.resolved_source,
            url: metadata.resolved_url,
            bytes,
        },
        cache_hit: true,
        bytes_path,
        metadata_path,
    }))
}

pub fn load_cached_raw_fetch(
    cache_root: &Path,
    source: SourceId,
    resolved_url: &str,
) -> Result<Option<CachedFetchResult>, IoError> {
    let (bytes_path, metadata_path) = raw_fetch_cache_paths(cache_root, source, resolved_url);
    if !bytes_path.exists() || !metadata_path.exists() {
        if bytes_path.exists() || metadata_path.exists() {
            quarantine_cache_paths(&[&bytes_path, &metadata_path], "incomplete_raw_fetch_cache");
        }
        return Ok(None);
    }
    let bytes = match fs::read(&bytes_path) {
        Ok(bytes) => bytes,
        Err(_) => {
            quarantine_cache_paths(&[&bytes_path, &metadata_path], "raw_fetch_bytes_read_error");
            return Ok(None);
        }
    };
    let metadata_bytes = match fs::read(&metadata_path) {
        Ok(bytes) => bytes,
        Err(_) => {
            quarantine_cache_paths(
                &[&bytes_path, &metadata_path],
                "raw_fetch_metadata_read_error",
            );
            return Ok(None);
        }
    };
    let Some(metadata) = load_cached_fetch_metadata(&metadata_bytes, &bytes, "<raw-fetch-cache>")
    else {
        quarantine_cache_paths(
            &[&bytes_path, &metadata_path],
            "raw_fetch_metadata_decode_error",
        );
        return Ok(None);
    };
    if metadata.bytes_len != bytes.len()
        || metadata.resolved_source != source
        || metadata.resolved_url != resolved_url
        || metadata.bytes_sha256 != sha256_hex(&bytes)
    {
        quarantine_cache_paths(
            &[&bytes_path, &metadata_path],
            "raw_fetch_metadata_mismatch",
        );
        return Ok(None);
    }
    Ok(Some(CachedFetchResult {
        result: FetchResult {
            source: metadata.resolved_source,
            url: metadata.resolved_url,
            bytes,
        },
        cache_hit: true,
        bytes_path,
        metadata_path,
    }))
}

fn cached_fetch_request_matches(
    cached: &rustwx_core::ModelRunRequest,
    requested: &rustwx_core::ModelRunRequest,
    fetch: &FetchRequest,
) -> bool {
    if fetch_uses_shared_forecast_cache(fetch) {
        return cached.model == requested.model
            && cached.cycle == requested.cycle
            && cached.product == requested.product;
    }
    cached == requested
}

fn fetch_uses_shared_forecast_cache(fetch: &FetchRequest) -> bool {
    fetch.request.model == rustwx_core::ModelId::Sref
        && fetch.request.product.starts_with("ensprod/pgrb212/")
        && fetch.request.product.ends_with("_3hrly")
}

pub fn store_cached_fetch(
    cache_root: &Path,
    fetch: &FetchRequest,
    result: &FetchResult,
) -> Result<CachedFetchResult, IoError> {
    let (bytes_path, metadata_path) = fetch_cache_paths(cache_root, fetch);
    if let Some(parent) = bytes_path.parent() {
        fs::create_dir_all(parent).map_err(cache_error)?;
    }
    atomic_write_bytes(&bytes_path, &result.bytes)?;
    let metadata = CachedFetchMetadata {
        request: fetch.request.clone(),
        source_override: fetch.source_override,
        variable_patterns: fetch.variable_patterns.clone(),
        resolved_source: result.source,
        resolved_url: result.url.clone(),
        resolved_family: fetch.request.product.clone(),
        bytes_len: result.bytes.len(),
        bytes_sha256: sha256_hex(&result.bytes),
    };
    let metadata_bytes = serde_json::to_vec_pretty(&VersionedJsonPayload {
        schema_version: FETCH_METADATA_SCHEMA_VERSION,
        payload: metadata,
    })
    .map_err(|err| IoError::Cache(err.to_string()))?;
    atomic_write_bytes(&metadata_path, &metadata_bytes)?;

    Ok(CachedFetchResult {
        result: result.clone(),
        cache_hit: false,
        bytes_path,
        metadata_path,
    })
}

pub fn store_cached_raw_fetch(
    cache_root: &Path,
    fetch: &FetchRequest,
    result: &FetchResult,
) -> Result<CachedFetchResult, IoError> {
    let (bytes_path, metadata_path) = raw_fetch_cache_paths(cache_root, result.source, &result.url);
    if let Some(parent) = bytes_path.parent() {
        fs::create_dir_all(parent).map_err(cache_error)?;
    }
    atomic_write_bytes(&bytes_path, &result.bytes)?;
    let metadata = CachedFetchMetadata {
        request: fetch.request.clone(),
        source_override: fetch.source_override,
        variable_patterns: Vec::new(),
        resolved_source: result.source,
        resolved_url: result.url.clone(),
        resolved_family: fetch.request.product.clone(),
        bytes_len: result.bytes.len(),
        bytes_sha256: sha256_hex(&result.bytes),
    };
    let metadata_bytes = serde_json::to_vec_pretty(&VersionedJsonPayload {
        schema_version: FETCH_METADATA_SCHEMA_VERSION,
        payload: metadata,
    })
    .map_err(|err| IoError::Cache(err.to_string()))?;
    atomic_write_bytes(&metadata_path, &metadata_bytes)?;

    Ok(CachedFetchResult {
        result: result.clone(),
        cache_hit: false,
        bytes_path,
        metadata_path,
    })
}

pub fn load_cached_selected_field(
    cache_root: &Path,
    fetch: &FetchRequest,
    selector: FieldSelector,
) -> Result<Option<CachedFieldResult>, IoError> {
    let path = field_cache_path(cache_root, fetch, selector);
    if !path.exists() {
        return Ok(None);
    }
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(_) => {
            quarantine_cache_paths(&[&path], "selected_field_read_error");
            return Ok(None);
        }
    };
    let Some(field) =
        load_cached_selected_field_payload(cache_root, fetch, selector, &path, &bytes)?
    else {
        return Ok(None);
    };
    Ok(Some(CachedFieldResult {
        field,
        cache_hit: true,
        path,
    }))
}

pub fn store_cached_selected_field(
    cache_root: &Path,
    fetch: &FetchRequest,
    field: &SelectedField2D,
) -> Result<CachedFieldResult, IoError> {
    let path = field_cache_path(cache_root, fetch, field.selector);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(cache_error)?;
    }
    let grid_key = grid_cache_key(&field.grid);
    let grid_path = grid_cache_path(cache_root, fetch, &grid_key);
    if !grid_path.exists() {
        if let Some(parent) = grid_path.parent() {
            fs::create_dir_all(parent).map_err(cache_error)?;
        }
        let grid_payload = CachedGridPayload {
            shape: field.grid.shape,
            lat_deg: field.grid.lat_deg.clone(),
            lon_deg: field.grid.lon_deg.clone(),
            projection: field.projection.clone(),
        };
        let grid_bytes = serialize_binary_payload(GRID_PAYLOAD_SCHEMA_VERSION, &grid_payload)?;
        atomic_write_bytes(&grid_path, &grid_bytes)?;
    }

    let field_payload = CachedFieldPayload {
        selector: CachedFieldSelector::from_selector(field.selector),
        units: field.units.clone(),
        values: field.values.clone(),
        grid_key,
    };
    let field_bytes = serialize_binary_payload(FIELD_PAYLOAD_SCHEMA_VERSION, &field_payload)?;
    atomic_write_bytes(&path, &field_bytes)?;
    Ok(CachedFieldResult {
        field: field.clone(),
        cache_hit: false,
        path,
    })
}

fn variable_patterns_slug(patterns: &[String]) -> String {
    if patterns.is_empty() {
        return "full".to_string();
    }
    let joined = patterns.join("__");
    let sanitized = sanitize_component(&joined);
    if sanitized.len() <= 120 {
        sanitized
    } else {
        format!("{}__{}vars", &sanitized[..120], patterns.len())
    }
}

fn sanitize_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut last_was_sep = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('_');
            last_was_sep = true;
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        "default".to_string()
    } else {
        trimmed.to_string()
    }
}

fn load_cached_selected_field_payload(
    cache_root: &Path,
    fetch: &FetchRequest,
    expected_selector: FieldSelector,
    field_path: &Path,
    bytes: &[u8],
) -> Result<Option<SelectedField2D>, IoError> {
    if let Some(payload) =
        load_binary_payload::<CachedFieldPayload>(bytes, FIELD_PAYLOAD_SCHEMA_VERSION)
    {
        let payload_selector = payload.selector.as_selector();
        if payload_selector != expected_selector {
            quarantine_cache_paths(&[field_path], "selected_field_selector_mismatch");
            return Ok(None);
        }
        let grid_path = grid_cache_path(cache_root, fetch, &payload.grid_key);
        let grid_bytes = match fs::read(&grid_path) {
            Ok(bytes) => bytes,
            Err(_) => {
                quarantine_cache_paths(&[field_path, &grid_path], "selected_field_grid_read_error");
                return Ok(None);
            }
        };
        let grid_payload = if let Some(payload) =
            load_binary_payload::<CachedGridPayload>(&grid_bytes, GRID_PAYLOAD_SCHEMA_VERSION)
        {
            payload
        } else if let Some(payload) = load_binary_payload::<LegacyCachedGridPayload>(&grid_bytes, 1)
        {
            CachedGridPayload {
                shape: payload.shape,
                lat_deg: payload.lat_deg,
                lon_deg: payload.lon_deg,
                projection: None,
            }
        } else {
            quarantine_cache_paths(
                &[field_path, &grid_path],
                "selected_field_grid_decode_error",
            );
            return Ok(None);
        };
        let grid = match LatLonGrid::new(
            grid_payload.shape,
            grid_payload.lat_deg,
            grid_payload.lon_deg,
        ) {
            Ok(grid) => grid,
            Err(_) => {
                quarantine_cache_paths(&[field_path, &grid_path], "selected_field_grid_invalid");
                return Ok(None);
            }
        };
        let field =
            match SelectedField2D::new(payload_selector, payload.units, grid, payload.values) {
                Ok(field) => {
                    if let Some(projection) = grid_payload.projection {
                        field.with_projection(projection)
                    } else {
                        field
                    }
                }
                Err(_) => {
                    quarantine_cache_paths(&[field_path, &grid_path], "selected_field_invalid");
                    return Ok(None);
                }
            };
        if !selected_field_grid_is_canonical(&field) {
            quarantine_cache_paths(
                &[field_path, &grid_path],
                "selected_field_grid_noncanonical",
            );
            return Ok(None);
        }
        return Ok(Some(field));
    }

    if let Ok(field) = bincode::deserialize::<LegacySelectedField2D>(bytes) {
        let legacy_selector = field.selector.as_selector();
        let field =
            match SelectedField2D::new(legacy_selector, field.units, field.grid, field.values) {
                Ok(field) => field,
                Err(_) => {
                    quarantine_cache_paths(&[field_path], "legacy_selected_field_invalid");
                    return Ok(None);
                }
            };
        if field.selector != expected_selector {
            quarantine_cache_paths(&[field_path], "legacy_selected_field_selector_mismatch");
            return Ok(None);
        }
        if !selected_field_grid_is_canonical(&field) {
            quarantine_cache_paths(&[field_path], "legacy_selected_field_grid_noncanonical");
            return Ok(None);
        }
        return Ok(Some(field));
    }

    quarantine_cache_paths(&[field_path], "selected_field_decode_error");
    Ok(None)
}

fn load_cached_fetch_metadata(
    bytes: &[u8],
    fetched_bytes: &[u8],
    expected_family: &str,
) -> Option<CachedFetchMetadata> {
    if let Ok(wrapper) = serde_json::from_slice::<VersionedJsonPayload<CachedFetchMetadata>>(bytes)
    {
        if wrapper.schema_version == FETCH_METADATA_SCHEMA_VERSION {
            return Some(wrapper.payload);
        }
    }
    if let Ok(wrapper) =
        serde_json::from_slice::<VersionedJsonPayload<LegacyCachedFetchMetadata>>(bytes)
    {
        return Some(CachedFetchMetadata {
            request: wrapper.payload.request,
            source_override: wrapper.payload.source_override,
            variable_patterns: wrapper.payload.variable_patterns,
            resolved_source: wrapper.payload.resolved_source,
            resolved_url: wrapper.payload.resolved_url,
            resolved_family: expected_family.to_string(),
            bytes_len: wrapper.payload.bytes_len,
            bytes_sha256: sha256_hex(fetched_bytes),
        });
    }
    serde_json::from_slice::<LegacyCachedFetchMetadata>(bytes)
        .ok()
        .map(|legacy| CachedFetchMetadata {
            request: legacy.request,
            source_override: legacy.source_override,
            variable_patterns: legacy.variable_patterns,
            resolved_source: legacy.resolved_source,
            resolved_url: legacy.resolved_url,
            resolved_family: expected_family.to_string(),
            bytes_len: legacy.bytes_len,
            bytes_sha256: sha256_hex(fetched_bytes),
        })
}

fn selected_field_grid_is_canonical(field: &SelectedField2D) -> bool {
    let nx = field.grid.shape.nx;
    let ny = field.grid.shape.ny;
    if nx == 0 || ny == 0 {
        return false;
    }
    if field.grid.lat_deg.len() != nx * ny || field.grid.lon_deg.len() != nx * ny {
        return false;
    }

    for row in 0..ny {
        let start = row * nx;
        let end = start + nx;
        let lat_row = &field.grid.lat_deg[start..end];
        let lon_row = &field.grid.lon_deg[start..end];

        if lat_row.iter().any(|value| !value.is_finite()) {
            return false;
        }
        if lon_row
            .iter()
            .any(|value| !value.is_finite() || *value < -180.0 || *value > 180.0)
        {
            return false;
        }
        if lon_row.windows(2).any(|pair| pair[1] < pair[0]) {
            return false;
        }
    }
    true
}

fn serialize_binary_payload<T: Serialize>(
    schema_version: u32,
    payload: &T,
) -> Result<Vec<u8>, IoError> {
    bincode::serialize(&VersionedBinaryPayload {
        schema_version,
        payload,
    })
    .map_err(|err| IoError::Cache(err.to_string()))
}

fn load_binary_payload<T: for<'de> Deserialize<'de>>(
    bytes: &[u8],
    expected_schema_version: u32,
) -> Option<T> {
    if let Ok(wrapper) = bincode::deserialize::<VersionedBinaryPayload<T>>(bytes) {
        if wrapper.schema_version == expected_schema_version {
            return Some(wrapper.payload);
        }
    }
    None
}

fn grid_cache_key(grid: &LatLonGrid) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    hash = fnv1a_mix(hash, grid.shape.nx as u64);
    hash = fnv1a_mix(hash, grid.shape.ny as u64);
    for value in &grid.lat_deg {
        hash = fnv1a_mix(hash, value.to_bits() as u64);
    }
    for value in &grid.lon_deg {
        hash = fnv1a_mix(hash, value.to_bits() as u64);
    }
    format!("{hash:016x}")
}

fn fnv1a_mix(hash: u64, value: u64) -> u64 {
    let mut out = hash;
    for byte in value.to_le_bytes() {
        out ^= u64::from(byte);
        out = out.wrapping_mul(0x100000001b3);
    }
    out
}

fn cache_error(err: std::io::Error) -> IoError {
    IoError::Cache(err.to_string())
}

fn atomic_write_bytes(path: &Path, bytes: &[u8]) -> Result<(), IoError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(cache_error)?;
    }
    let tmp_path = temp_path_for(path);
    let write_result = (|| {
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp_path)
            .map_err(cache_error)?;
        file.write_all(bytes).map_err(cache_error)?;
        file.sync_all().map_err(cache_error)?;
        Ok::<(), IoError>(())
    })();
    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }
    fs::rename(&tmp_path, path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        cache_error(err)
    })
}

fn quarantine_cache_paths(paths: &[&Path], reason: &str) {
    for path in paths {
        quarantine_cache_path(path, reason);
    }
}

/// Move one suspect cache file aside.  Never deletes it.
///
/// A failed rename used to fall through to `fs::remove_file`, which turns
/// quarantine into deletion at exactly the moment quarantine matters: a full
/// disk, a directory that cannot be created, a file another process holds
/// open.  The evidence of what actually arrived is then gone, and gone
/// precisely in the cases hardest to reproduce.
///
/// Leaving the file where it is instead is safe here: every current-schema
/// entry is re-verified against its recorded request identity, length and
/// SHA-256 at load, so an entry that could not be moved aside is refused on
/// every subsequent use rather than served.  It costs a re-verification per
/// use, which is the correct price for not destroying evidence.
fn quarantine_cache_path(path: &Path, reason: &str) {
    if !path.exists() {
        return;
    }
    let quarantine_path = quarantine_path_for(path, reason);
    if let Some(parent) = quarantine_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Err(err) = fs::rename(path, &quarantine_path) {
        eprintln!(
            "rustwx-io cache: could not move {} aside ({}); leaving it in \
             place -- it stays refused by the use-time verification and is \
             never deleted",
            path.display(),
            err
        );
    }
}

fn temp_path_for(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("cache");
    path.with_file_name(format!(
        ".{file_name}.tmp-{}-{}",
        process::id(),
        unique_suffix()
    ))
}

fn quarantine_path_for(path: &Path, reason: &str) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("cache");
    path.with_file_name(format!(
        "{file_name}.corrupt-{reason}-{}-{}",
        process::id(),
        unique_suffix()
    ))
}

fn unique_suffix() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn sha256_hex(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut message = bytes.to_vec();
    let bit_len = (message.len() as u64) * 8;
    message.push(0x80);
    while (message.len() % 64) != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    let mut h0 = 0x6a09e667u32;
    let mut h1 = 0xbb67ae85u32;
    let mut h2 = 0x3c6ef372u32;
    let mut h3 = 0xa54ff53au32;
    let mut h4 = 0x510e527fu32;
    let mut h5 = 0x9b05688cu32;
    let mut h6 = 0x1f83d9abu32;
    let mut h7 = 0x5be0cd19u32;

    for chunk in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).take(16).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h0;
        let mut b = h1;
        let mut c = h2;
        let mut d = h3;
        let mut e = h4;
        let mut f = h5;
        let mut g = h6;
        let mut h = h7;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
        h5 = h5.wrapping_add(f);
        h6 = h6.wrapping_add(g);
        h7 = h7.wrapping_add(h);
    }

    format!("{h0:08x}{h1:08x}{h2:08x}{h3:08x}{h4:08x}{h5:08x}{h6:08x}{h7:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustwx_core::{
        CanonicalField, CycleSpec, FieldSelector, GridShape, LatLonGrid, ModelId, ModelRunRequest,
        SourceId, VerticalSelector,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    fn sample_fetch_request() -> FetchRequest {
        FetchRequest {
            request: ModelRunRequest::new(
                ModelId::Hrrr,
                CycleSpec::new("20260414", 23).unwrap(),
                0,
                "prs",
            )
            .unwrap(),
            source_override: Some(SourceId::Aws),
            variable_patterns: vec!["TMP:500 mb".to_string(), "UGRD:500 mb".to_string()],
        }
    }

    fn sample_field() -> SelectedField2D {
        let grid = LatLonGrid::new(
            GridShape::new(2, 2).unwrap(),
            vec![40.0, 40.0, 39.0, 39.0],
            vec![-100.0, -99.0, -100.0, -99.0],
        )
        .unwrap();
        SelectedField2D::new(
            FieldSelector::new(
                CanonicalField::Temperature,
                VerticalSelector::IsobaricHpa(500),
            ),
            "K",
            grid,
            vec![255.0, 256.0, 257.0, 258.0],
        )
        .unwrap()
    }

    fn temp_cache_root() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "rustwx_io_cache_test_{}_{unique}",
            std::process::id()
        ))
    }

    #[test]
    fn cached_fetch_and_field_round_trip() {
        let cache_root = temp_cache_root();
        let fetch = sample_fetch_request();
        let result = FetchResult {
            source: SourceId::Aws,
            url: "https://example.test/sample.grib2".to_string(),
            bytes: vec![1, 2, 3, 4, 5],
        };
        let stored = store_cached_fetch(&cache_root, &fetch, &result).unwrap();
        assert!(!stored.cache_hit);
        let loaded = load_cached_fetch(&cache_root, &fetch).unwrap().unwrap();
        assert!(loaded.cache_hit);
        assert_eq!(loaded.result, result);

        let field = sample_field();
        let stored_field = store_cached_selected_field(&cache_root, &fetch, &field).unwrap();
        assert!(!stored_field.cache_hit);
        let grid_path = grid_cache_path(&cache_root, &fetch, &grid_cache_key(&field.grid));
        assert!(grid_path.exists());
        let loaded_field = load_cached_selected_field(&cache_root, &fetch, field.selector)
            .unwrap()
            .unwrap();
        assert!(loaded_field.cache_hit);
        assert_eq!(loaded_field.field, field);

        fs::remove_dir_all(cache_root).ok();
    }

    #[test]
    fn field_cache_reuses_shared_grid_payload() {
        let cache_root = temp_cache_root();
        let fetch = sample_fetch_request();
        let first = sample_field();
        let second = SelectedField2D::new(
            FieldSelector::new(CanonicalField::UWind, VerticalSelector::IsobaricHpa(500)),
            "m/s",
            first.grid.clone(),
            vec![10.0, 11.0, 12.0, 13.0],
        )
        .unwrap();

        store_cached_selected_field(&cache_root, &fetch, &first).unwrap();
        store_cached_selected_field(&cache_root, &fetch, &second).unwrap();

        let grids_dir = artifact_cache_dir(&cache_root, &fetch)
            .join("fields")
            .join("grids");
        let grid_files = fs::read_dir(&grids_dir)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(grid_files.len(), 1);

        fs::remove_dir_all(cache_root).ok();
    }

    #[test]
    fn sref_multilead_fetch_cache_reuses_bytes_across_forecast_hours() {
        let cache_root = temp_cache_root();
        let mut first = sample_fetch_request();
        first.request = ModelRunRequest::new(
            ModelId::Sref,
            CycleSpec::new("20260507", 3).unwrap(),
            3,
            "ensprod/pgrb212/prob_3hrly",
        )
        .unwrap();
        first.source_override = Some(SourceId::Nomads);
        first.variable_patterns = Vec::new();
        let mut second = first.clone();
        second.request.forecast_hour = 24;

        let (first_bytes_path, _) = fetch_cache_paths(&cache_root, &first);
        let (second_bytes_path, _) = fetch_cache_paths(&cache_root, &second);
        assert_eq!(first_bytes_path, second_bytes_path);

        let first_field_path = field_cache_path(&cache_root, &first, sample_field().selector);
        let second_field_path = field_cache_path(&cache_root, &second, sample_field().selector);
        assert_ne!(first_field_path, second_field_path);

        let result = FetchResult {
            source: SourceId::Nomads,
            url: "https://example.test/sref.t03z.pgrb212.prob_3hrly.grib2".to_string(),
            bytes: vec![42, 43, 44],
        };
        store_cached_fetch(&cache_root, &first, &result).unwrap();
        let loaded = load_cached_fetch(&cache_root, &second).unwrap().unwrap();
        assert!(loaded.cache_hit);
        assert_eq!(loaded.result, result);

        fs::remove_dir_all(cache_root).ok();
    }

    #[test]
    fn raw_fetch_cache_reuses_full_file_bytes_by_url() {
        let cache_root = temp_cache_root();
        let first = sample_fetch_request();
        let mut second = first.clone();
        second.request = ModelRunRequest::new(
            ModelId::Hrrr,
            CycleSpec::new("20260414", 23).unwrap(),
            0,
            "sfc",
        )
        .unwrap();
        second.variable_patterns = Vec::new();
        let result = FetchResult {
            source: SourceId::Aws,
            url: "https://example.test/hrrr.t23z.wrfsfcf00.grib2".to_string(),
            bytes: vec![5, 4, 3, 2, 1],
        };

        let stored = store_cached_raw_fetch(&cache_root, &first, &result).unwrap();
        assert!(!stored.cache_hit);
        let loaded = load_cached_raw_fetch(&cache_root, SourceId::Aws, &result.url)
            .unwrap()
            .unwrap();
        assert!(loaded.cache_hit);
        assert_eq!(loaded.result, result);
        let (raw_bytes_path, _) = raw_fetch_cache_paths(&cache_root, SourceId::Aws, &result.url);
        assert_eq!(loaded.bytes_path, raw_bytes_path);
        let (product_bytes_path, _) = fetch_cache_paths(&cache_root, &second);
        assert_ne!(loaded.bytes_path, product_bytes_path);

        fs::remove_dir_all(cache_root).ok();
    }

    #[test]
    fn load_cached_selected_field_reads_legacy_embedded_field_payload() {
        let cache_root = temp_cache_root();
        let fetch = sample_fetch_request();
        let field = sample_field();
        let path = field_cache_path(&cache_root, &fetch, field.selector);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let legacy_bytes = bincode::serialize(&field).unwrap();
        fs::write(&path, legacy_bytes).unwrap();

        let loaded = load_cached_selected_field(&cache_root, &fetch, field.selector)
            .unwrap()
            .unwrap();
        assert_eq!(loaded.field, field);

        fs::remove_dir_all(cache_root).ok();
    }

    #[test]
    fn corrupt_fetch_metadata_is_quarantined_and_treated_as_cache_miss() {
        let cache_root = temp_cache_root();
        let fetch = sample_fetch_request();
        let (bytes_path, metadata_path) = fetch_cache_paths(&cache_root, &fetch);
        fs::create_dir_all(bytes_path.parent().unwrap()).unwrap();
        fs::write(&bytes_path, [1_u8, 2, 3, 4]).unwrap();
        fs::write(&metadata_path, b"{not-json").unwrap();

        let loaded = load_cached_fetch(&cache_root, &fetch).unwrap();
        assert!(loaded.is_none());
        assert!(!bytes_path.exists());
        assert!(!metadata_path.exists());
        let quarantined = fs::read_dir(bytes_path.parent().unwrap())
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(
            quarantined
                .iter()
                .any(|entry| entry.file_name().to_string_lossy().contains("corrupt"))
        );

        fs::remove_dir_all(cache_root).ok();
    }

    #[test]
    fn store_cached_fetch_writes_versioned_metadata() {
        let cache_root = temp_cache_root();
        let fetch = sample_fetch_request();
        let result = FetchResult {
            source: SourceId::Aws,
            url: "https://example.test/sample.grib2".to_string(),
            bytes: vec![9, 8, 7, 6],
        };

        store_cached_fetch(&cache_root, &fetch, &result).unwrap();
        let (_, metadata_path) = fetch_cache_paths(&cache_root, &fetch);
        let wrapper: VersionedJsonPayload<CachedFetchMetadata> =
            serde_json::from_slice(&fs::read(&metadata_path).unwrap()).unwrap();
        assert_eq!(wrapper.schema_version, FETCH_METADATA_SCHEMA_VERSION);
        assert_eq!(wrapper.payload.resolved_source, SourceId::Aws);
        assert_eq!(wrapper.payload.resolved_family, "prs");
        assert_eq!(wrapper.payload.bytes_sha256, sha256_hex(&result.bytes));

        fs::remove_dir_all(cache_root).ok();
    }

    #[test]
    fn fetch_metadata_digest_mismatch_is_quarantined_and_treated_as_cache_miss() {
        let cache_root = temp_cache_root();
        let fetch = sample_fetch_request();
        let result = FetchResult {
            source: SourceId::Aws,
            url: "https://example.test/sample.grib2".to_string(),
            bytes: vec![9, 8, 7, 6],
        };

        store_cached_fetch(&cache_root, &fetch, &result).unwrap();
        let (_, metadata_path) = fetch_cache_paths(&cache_root, &fetch);
        let mut wrapper: VersionedJsonPayload<CachedFetchMetadata> =
            serde_json::from_slice(&fs::read(&metadata_path).unwrap()).unwrap();
        wrapper.payload.bytes_sha256 = "deadbeef".into();
        fs::write(&metadata_path, serde_json::to_vec_pretty(&wrapper).unwrap()).unwrap();

        let loaded = load_cached_fetch(&cache_root, &fetch).unwrap();
        assert!(loaded.is_none());
        let parent = metadata_path.parent().unwrap();
        let quarantined = fs::read_dir(parent)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(
            quarantined
                .iter()
                .any(|entry| entry.file_name().to_string_lossy().contains("corrupt"))
        );

        fs::remove_dir_all(cache_root).ok();
    }

    #[test]
    fn corrupt_field_cache_is_quarantined_and_treated_as_cache_miss() {
        let cache_root = temp_cache_root();
        let fetch = sample_fetch_request();
        let field = sample_field();
        let path = field_cache_path(&cache_root, &fetch, field.selector);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"definitely-not-bincode").unwrap();

        let loaded = load_cached_selected_field(&cache_root, &fetch, field.selector).unwrap();
        assert!(loaded.is_none());
        assert!(!path.exists());
        let quarantined = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(
            quarantined
                .iter()
                .any(|entry| entry.file_name().to_string_lossy().contains("corrupt"))
        );

        fs::remove_dir_all(cache_root).ok();
    }

    #[test]
    fn legacy_noncanonical_field_cache_is_quarantined_and_treated_as_cache_miss() {
        let cache_root = temp_cache_root();
        let fetch = sample_fetch_request();
        let mut field = sample_field();
        field.grid.lon_deg = vec![260.0, 261.0, 260.0, 261.0];
        let path = field_cache_path(&cache_root, &fetch, field.selector);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bincode::serialize(&field).unwrap()).unwrap();

        let loaded = load_cached_selected_field(&cache_root, &fetch, field.selector).unwrap();
        assert!(loaded.is_none());
        assert!(!path.exists());
        let quarantined = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(
            quarantined
                .iter()
                .any(|entry| entry.file_name().to_string_lossy().contains("corrupt"))
        );

        fs::remove_dir_all(cache_root).ok();
    }
}
