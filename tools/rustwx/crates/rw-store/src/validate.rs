//! Validation library for rw-store hour files and run directories.
//!
//! Two depth levels: [`ValidateDepth::Structural`] checks the binary layout,
//! metadata, index geometry, and payload bounds without reading any compressed
//! bytes; [`ValidateDepth::Deep`] additionally decompresses every non-empty
//! chunk and verifies the decoded content against the index statistics.
//!
//! [`validate_hour_file`] returns `Ok(report)` for any file that opens —
//! format problems land in `report.errors` so a CLI can print them all rather
//! than stopping at the first. `Err(_)` is reserved for I/O failures.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use rustwx_core::{MAX_GRID_CELLS, MAX_VOLUME_ELEMENTS};

use crate::codec::MISSING_Q;
use crate::error::{RwResult, RwStoreError};
use crate::format::{
    CODEC_2D, CODEC_3D, COL_X, COL_Y, FLAG_CONSTANT, FLAG_EMPTY, FLAG_HAS_MISSING, HEADER_LEN,
    INDEX_RECORD_LEN, KIND_COLUMN3D, KIND_TILE2D, RwsHourMeta, TILE_X, TILE_Y,
};
use crate::grid::GridFile;
use crate::header::RwsHeader;
use crate::index::ChunkRecord;
use crate::run::{RwsRunManifest, validate_store_component};

/// Metadata is JSON, not gridded payload. Normal hour metadata is far smaller;
/// this leaves room for large variable catalogs while preventing a hostile
/// header from asking serde to scan a multi-GiB sparse region.
const MAX_HOUR_META_BYTES: u64 = 16 * 1024 * 1024;
/// Keep validation within the same index envelope as `HourReader`.
const MAX_INDEX_RECORDS: u64 = 8_388_608;

/// Read-only validation backing. Mapping keeps validation memory independent
/// of the hour-file payload size, so large legitimate stores remain supported
/// and sparse/trailing files cannot force a file-sized heap allocation.
enum ValidationBytes {
    Empty,
    Mapped(Mmap),
}

impl ValidationBytes {
    fn open(path: &Path) -> RwResult<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len == 0 {
            return Ok(Self::Empty);
        }
        usize::try_from(file_len).map_err(|_| {
            RwStoreError::Format(format!(
                "file {} is {file_len} bytes and cannot be addressed on this platform",
                path.display()
            ))
        })?;
        // SAFETY: the mapping is read-only and all access below is through
        // checked slices. The map is created from the already-open handle, so
        // replacing `path` cannot redirect later validation reads.
        let mapped = unsafe { Mmap::map(&file) }.map_err(|err| {
            RwStoreError::Io(std::io::Error::new(
                err.kind(),
                format!("map {} for validation: {err}", path.display()),
            ))
        })?;
        Ok(Self::Mapped(mapped))
    }

    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Empty => &[],
            Self::Mapped(mapped) => &mapped[..],
        }
    }
}

/// How deeply to validate a store file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidateDepth {
    /// Header, meta, index geometry, payload bounds — no decompression.
    Structural,
    /// Structural + decompress every chunk, verify raw_len, stats, flags.
    Deep,
}

/// Aggregate result of validating one or more store files.
#[derive(Debug, Default)]
pub struct ValidationReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub stats: ValidationStats,
}

/// Counts collected during validation.
#[derive(Debug, Default)]
pub struct ValidationStats {
    /// Number of variables found in the hour-file meta block.
    pub variables: usize,
    /// Total index records seen, including EMPTY and CONSTANT chunks.
    pub chunks: u64,
    /// Sum of compressed payload lengths (bytes) across all non-empty records.
    pub payload_bytes: u64,
}

impl ValidationReport {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    fn error(&mut self, msg: impl Into<String>) {
        self.errors.push(msg.into());
    }

    fn warn(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }

    fn merge_prefixed(&mut self, prefix: &str, other: ValidationReport) {
        for e in other.errors {
            self.errors.push(format!("{prefix}: {e}"));
        }
        for w in other.warnings {
            self.warnings.push(format!("{prefix}: {w}"));
        }
        self.stats.variables += other.stats.variables;
        self.stats.chunks += other.stats.chunks;
        self.stats.payload_bytes += other.stats.payload_bytes;
    }
}

/// Validate one `.rws` hour file at the requested depth.
///
/// Returns `Err` only for I/O failure opening the file. All format problems
/// are reported in `report.errors`.
pub fn validate_hour_file(path: &Path, depth: ValidateDepth) -> RwResult<ValidationReport> {
    let data = ValidationBytes::open(path)?;
    let mut report = ValidationReport::default();
    check_hour_file(data.as_slice(), depth, &mut report);
    Ok(report)
}

/// Validate all hour files referenced by the run manifest in `run_dir`, plus
/// the directory structure itself (run.json, grid.rwg, stray files).
pub fn validate_run_dir(run_dir: &Path, depth: ValidateDepth) -> RwResult<ValidationReport> {
    let mut report = ValidationReport::default();

    let run_dir = match std::fs::canonicalize(run_dir) {
        Ok(path) => path,
        Err(err) => {
            report.error(format!(
                "run dir: cannot resolve {}: {err}",
                run_dir.display()
            ));
            return Ok(report);
        }
    };

    // 1. run.json
    let manifest_path = run_dir.join("run.json");
    let manifest_path = match canonical_contained_path(&run_dir, &manifest_path, "run.json") {
        Ok(path) => path,
        Err(err) => {
            report.error(err);
            return Ok(report);
        }
    };
    let manifest = match RwsRunManifest::load_bounded(&manifest_path) {
        Ok(manifest) => manifest,
        Err(err) => {
            report.error(format!("run.json: {err}"));
            return Ok(report);
        }
    };
    if let Err(err) = manifest.validate_contents() {
        // Keep the parsed manifest so the validator can still report grid,
        // hour, and stray-file problems. Every manifest-derived path is
        // independently checked below before it is touched.
        report.error(format!("run.json: {err}"));
    }

    // 2. grid.rwg
    let grid_path = run_dir.join("grid.rwg");
    let grid_file = match canonical_contained_path(&run_dir, &grid_path, "grid.rwg") {
        Ok(path) => match GridFile::open(&path) {
            Ok(grid) => Some(grid),
            Err(err) => {
                report.error(format!("grid.rwg: {err}"));
                None
            }
        },
        Err(err) => {
            report.error(err);
            None
        }
    };
    if let Some(ref grid) = grid_file {
        if grid.hash != manifest.grid_hash {
            report.error(format!(
                "grid.rwg: sha256 {} does not match manifest grid_hash {}",
                grid.hash, manifest.grid_hash
            ));
        }
        if grid.nx != manifest.nx || grid.ny != manifest.ny {
            report.error(format!(
                "grid.rwg: dimensions {}x{} do not match manifest nx={} ny={}",
                grid.nx, grid.ny, manifest.nx, manifest.ny
            ));
        }
    }

    // 3. Validate each registered hour file.
    let mut referenced_files: HashSet<String> = HashSet::new();
    for (hour, entry) in &manifest.hours {
        referenced_files.insert(entry.file.clone());

        // Security: reject any manifest-supplied filename that is not a plain
        // component (no absolute paths, no `..` traversals, no `.` components).
        // An absolute path passed to `run_dir.join()` silently replaces the base
        // directory entirely, allowing path traversal outside the run dir.
        if validate_store_component("hour filename", &entry.file).is_err() {
            report.error(format!(
                "run.json: hour {hour} file '{}' must be a relative path without '..', root, or drive components",
                entry.file
            ));
            continue;
        }

        let hour_path = match canonical_contained_path(
            &run_dir,
            &run_dir.join(&entry.file),
            &format!("hour {hour} file '{}'", entry.file),
        ) {
            Ok(path) => path,
            Err(err) => {
                report.error(err);
                continue;
            }
        };
        let hour_data = match ValidationBytes::open(&hour_path) {
            Ok(b) => b,
            Err(err) => {
                report.error(format!(
                    "hour {hour}: I/O error reading '{}': {err}",
                    entry.file
                ));
                continue;
            }
        };
        let mut hour_report = ValidationReport::default();
        let hour_bytes = hour_data.as_slice();
        check_hour_file(hour_bytes, depth, &mut hour_report);
        report.merge_prefixed(&entry.file, hour_report);

        // Cross-check hour meta against manifest.
        if let Ok(header) = RwsHeader::parse(hour_bytes) {
            if u64::from(header.meta_len) <= MAX_HOUR_META_BYTES {
                let meta_end = HEADER_LEN.checked_add(header.meta_len as usize);
                if let Some(meta_end) = meta_end.filter(|end| *end <= hour_bytes.len()) {
                    if let Ok(meta) =
                        serde_json::from_slice::<RwsHourMeta>(&hour_bytes[HEADER_LEN..meta_end])
                    {
                        if let Err(error) = manifest.validate_hour_meta(*hour, &meta) {
                            report.error(format!("{}: {error}", entry.file));
                        }
                        // Check that all manifest-listed vars are present in the hour file.
                        let hour_var_names: HashSet<&str> =
                            meta.variables.iter().map(|v| v.name.as_str()).collect();
                        for var_name in &entry.variables {
                            if !hour_var_names.contains(var_name.as_str()) {
                                report.error(format!(
                                "{}: manifest lists variable '{}' but it is not in the hour file",
                                entry.file, var_name
                            ));
                            }
                        }
                        // Hour file has more variables than the manifest entry lists -> stale manifest.
                        let manifest_var_set: HashSet<&str> =
                            entry.variables.iter().map(|v| v.as_str()).collect();
                        for var in &meta.variables {
                            if !manifest_var_set.contains(var.name.as_str()) {
                                report.warn(format!(
                                "{}: hour file variable '{}' is not listed in the manifest entry (stale manifest?)",
                                entry.file, var.name
                            ));
                            }
                        }
                    }
                }
            }
        }
    }

    // 4. Stray .rws files not referenced by the manifest.
    let entries = match std::fs::read_dir(&run_dir) {
        Ok(e) => e,
        Err(err) => {
            report.error(format!("run dir read: {err}"));
            return Ok(report);
        }
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        // Ignore lock and temp files.
        if name == ".rw-lock" || name.starts_with('.') && name.contains(".tmp-") {
            continue;
        }
        if name.ends_with(".rws") && !referenced_files.contains(&name) {
            report.warn(format!("stray file '{}' not referenced by run.json", name));
        }
    }

    Ok(report)
}

fn canonical_contained_path(run_dir: &Path, path: &Path, label: &str) -> Result<PathBuf, String> {
    let canonical = std::fs::canonicalize(path)
        .map_err(|err| format!("{label}: cannot resolve {}: {err}", path.display()))?;
    if !canonical.starts_with(run_dir) {
        return Err(format!(
            "{label}: {} resolves outside run directory {}",
            path.display(),
            run_dir.display()
        ));
    }
    Ok(canonical)
}

// ---------------------------------------------------------------------------
// Validation bounds constants
// ---------------------------------------------------------------------------

/// Hard ceiling for a single chunk's uncompressed size in deep validation.
/// Legitimate chunks are <= 256 KiB (2D 256x256 f32) or ~a few MiB (3D
/// columns); anything larger is hostile or corrupt, not worth decoding.
const MAX_DEEP_CHUNK_RAW_LEN: u32 = 64 * 1024 * 1024;
const ISSUE_RANGE: u8 = 1 << 0;
const ISSUE_SPAN: u8 = 1 << 1;
const ISSUE_OVERFLOW: u8 = 1 << 2;
const ISSUE_RAW_LEN: u8 = 1 << 3;

fn try_record_buffer(index_count: usize) -> Result<Vec<ChunkRecord>, String> {
    let mut records = Vec::new();
    records
        .try_reserve_exact(index_count)
        .map_err(|err| format!("cannot allocate index for {index_count} records: {err}"))?;
    Ok(records)
}

fn try_issue_buffer(index_count: usize) -> Result<Vec<u8>, String> {
    let mut issues = Vec::new();
    issues.try_reserve_exact(index_count).map_err(|err| {
        format!("cannot allocate validation flags for {index_count} records: {err}")
    })?;
    issues.resize(index_count, 0);
    Ok(issues)
}

// ---------------------------------------------------------------------------
// Core hour-file checker (operates on in-memory bytes)
// ---------------------------------------------------------------------------

fn check_hour_file(data: &[u8], depth: ValidateDepth, report: &mut ValidationReport) {
    let file_len = data.len() as u64;

    // Check 1: header parses.
    let header = match RwsHeader::parse(data) {
        Ok(h) => h,
        Err(err) => {
            report.error(format!("header: {err}"));
            return;
        }
    };
    if header.index_count > MAX_INDEX_RECORDS {
        report.error(format!(
            "index_count {} exceeds supported limit {MAX_INDEX_RECORDS}",
            header.index_count
        ));
        return;
    }

    // Check 2: meta region.
    if u64::from(header.meta_len) > MAX_HOUR_META_BYTES {
        report.error(format!(
            "meta region is {} bytes; limit is {MAX_HOUR_META_BYTES} bytes",
            header.meta_len
        ));
        return;
    }
    let meta_start = HEADER_LEN;
    let meta_len = match usize::try_from(header.meta_len) {
        Ok(len) => len,
        Err(_) => {
            report.error(format!(
                "meta length {} does not fit this platform",
                header.meta_len
            ));
            return;
        }
    };
    let meta_end = match meta_start.checked_add(meta_len) {
        Some(end) => end,
        None => {
            report.error(format!(
                "meta region length {} overflows this platform",
                header.meta_len
            ));
            return;
        }
    };
    if data.len() < meta_end {
        report.error(format!(
            "meta region [{meta_start},{meta_end}) out of bounds (file_len {file_len})"
        ));
        return;
    }
    let meta_bytes = &data[meta_start..meta_end];
    let meta_str = match std::str::from_utf8(meta_bytes) {
        Ok(s) => s,
        Err(err) => {
            report.error(format!("meta region: invalid UTF-8: {err}"));
            return;
        }
    };
    let meta: RwsHourMeta = match serde_json::from_str(meta_str) {
        Ok(m) => m,
        Err(err) => {
            report.error(format!("meta JSON parse: {err}"));
            return;
        }
    };
    if let Err(error) = meta.validate_time_schema() {
        report.error(error);
    }
    if meta.nx == 0 || meta.ny == 0 {
        report.error(format!(
            "meta: degenerate grid {}x{} (nx and ny must be nonzero)",
            meta.nx, meta.ny
        ));
    }

    // Check 3: variable metadata consistency.
    let mut var_ids_seen: HashSet<u16> = HashSet::new();
    let mut var_names_seen: HashSet<String> = HashSet::new();
    for var in &meta.variables {
        if !var_ids_seen.insert(var.id) {
            report.error(format!("meta: duplicate variable id {}", var.id));
        }
        if !var_names_seen.insert(var.name.clone()) {
            report.error(format!("meta: duplicate variable name '{}'", var.name));
        }
        match var.kind.as_str() {
            "surface2d" => {
                if var.codec != CODEC_2D {
                    report.error(format!(
                        "variable '{}': surface2d codec must be '{}', got '{}'",
                        var.name, CODEC_2D, var.codec
                    ));
                }
                if !var.levels_hpa.is_empty() {
                    report.error(format!(
                        "variable '{}': surface2d must have empty levels_hpa, got {} levels",
                        var.name,
                        var.levels_hpa.len()
                    ));
                }
            }
            "pressure3d" => {
                if var.codec != CODEC_3D {
                    report.error(format!(
                        "variable '{}': pressure3d codec must be '{}', got '{}'",
                        var.name, CODEC_3D, var.codec
                    ));
                }
                if var.levels_hpa.is_empty() {
                    report.error(format!(
                        "variable '{}': pressure3d must have non-empty levels_hpa",
                        var.name
                    ));
                }
            }
            other => {
                report.error(format!(
                    "variable '{}': unknown kind '{}' (expected 'surface2d' or 'pressure3d')",
                    var.name, other
                ));
            }
        }
    }

    // Warn if the chunking values in meta differ from the format constants.
    let chunking = &meta.chunking;
    if chunking.tile_y != TILE_Y || chunking.tile_x != TILE_X {
        report.warn(format!(
            "meta chunking tile_y={} tile_x={} differs from format constants {TILE_Y}/{TILE_X}",
            chunking.tile_y, chunking.tile_x
        ));
    }
    if chunking.col_y != COL_Y || chunking.col_x != COL_X {
        report.warn(format!(
            "meta chunking col_y={} col_x={} differs from format constants {COL_Y}/{COL_X}",
            chunking.col_y, chunking.col_x
        ));
    }
    // Use meta chunking values for tile-geometry checks as the spec says.
    let tile_y = chunking.tile_y;
    let tile_x = chunking.tile_x;
    let col_y = chunking.col_y;
    let col_x = chunking.col_x;

    // Issue 1: Guard against zero chunking values. Any zero causes a
    // division-by-zero panic in checks 6, 8, and 10. Detect all bad fields,
    // push errors, and skip every geometry-dependent check.
    let mut chunking_bad = false;
    for (field, value) in &[
        ("tile_y", tile_y),
        ("tile_x", tile_x),
        ("col_y", col_y),
        ("col_x", col_x),
    ] {
        if *value == 0 {
            report.error(format!(
                "meta: chunking.{field} is zero — chunking values must be nonzero"
            ));
            chunking_bad = true;
        }
    }

    // Check 4: index region in bounds and every record parses.
    let index_region_end = header.payload_offset;
    if file_len < index_region_end {
        report.error(format!(
            "index region ends at {index_region_end} but file_len is {file_len}"
        ));
        return;
    }

    let index_count = match usize::try_from(header.index_count) {
        Ok(count) => count,
        Err(_) => {
            report.error(format!(
                "index_count {} does not fit this platform",
                header.index_count
            ));
            return;
        }
    };
    let var_id_map: HashMap<u16, &crate::format::RwsVariableMeta> =
        meta.variables.iter().map(|v| (v.id, v)).collect();

    // A byte-per-record issue mask is bounded and fallible. Four HashSets of
    // hostile record indices could otherwise multiply memory use and panic in
    // their infallible growth paths.
    let mut record_issues = match try_issue_buffer(index_count) {
        Ok(issues) => issues,
        Err(err) => {
            report.error(err);
            return;
        }
    };

    let mut records = match try_record_buffer(index_count) {
        Ok(records) => records,
        Err(err) => {
            report.error(err);
            return;
        }
    };
    for i in 0..index_count {
        let start = header.index_offset as usize + i * INDEX_RECORD_LEN;
        // Already checked file_len >= index_region_end which covers the index.
        let rec = match ChunkRecord::unpack(&data[start..start + INDEX_RECORD_LEN]) {
            Ok(r) => r,
            Err(err) => {
                report.error(format!("index record {i}: {err}"));
                continue;
            }
        };

        // var_id must exist in meta.
        let var_meta = match var_id_map.get(&rec.var_id) {
            Some(v) => v,
            None => {
                report.error(format!(
                    "index record {i}: var_id {} not found in meta",
                    rec.var_id
                ));
                records.push(rec);
                continue;
            }
        };

        // kind must match variable's kind.
        let expected_kind = match var_meta.kind.as_str() {
            "surface2d" => KIND_TILE2D,
            "pressure3d" => KIND_COLUMN3D,
            _ => {
                records.push(rec);
                continue; // already reported in variable check
            }
        };
        if rec.kind != expected_kind {
            report.error(format!(
                "index record {i} (var '{}'): kind {} != expected {} for kind '{}'",
                var_meta.name, rec.kind, expected_kind, var_meta.kind
            ));
        }

        // flags must be a subset of known flags.
        let valid_flags = FLAG_EMPTY | FLAG_CONSTANT | FLAG_HAS_MISSING;
        if rec.flags & !valid_flags != 0 {
            report.error(format!(
                "index record {i} (var '{}'): flags 0x{:02x} has unknown bits",
                var_meta.name, rec.flags
            ));
        }

        // Reserved bytes [48..64] should be zero.
        let rec_start = header.index_offset as usize + i * INDEX_RECORD_LEN;
        let reserved = &data[rec_start + 48..rec_start + 64];
        if reserved.iter().any(|&b| b != 0) {
            report.warn(format!(
                "index record {i} (var '{}'): reserved bytes [48..64] are non-zero",
                var_meta.name
            ));
        }

        records.push(rec);
    }

    // Check 5: index strictly ascending by sort_key.
    let mut index_sort_bad = false;
    for i in 1..records.len() {
        if records[i - 1].sort_key() >= records[i].sort_key() {
            index_sort_bad = true;
            report.error(format!(
                "index sort order violated at records {}..{}: {:?} !< {:?}",
                i - 1,
                i,
                records[i - 1].sort_key(),
                records[i].sort_key()
            ));
        }
    }

    // Build tile-count expectations per variable.
    let (nx, ny) = (meta.nx, meta.ny);

    // Sanity upper-bounds on meta dimensions and chunking values.
    // Hostile files with nx=ny=18_000_000_000_000_000_000 (fits usize on x64)
    // pass the nx>0 check but make div_ceil produce huge tile counts whose
    // product overflows usize. We bound them here so every downstream multiply
    // is safe. geometry_bad is a superset of chunking_bad.
    const MAX_NX_NY: usize = MAX_GRID_CELLS;
    const MAX_NX_NY_PRODUCT: usize = MAX_GRID_CELLS;
    const MAX_CHUNK_DIM: usize = 65_536;
    const MAX_LEVELS: usize = 4_096;
    let mut geometry_bad = chunking_bad;
    if nx > MAX_NX_NY {
        report.error(format!(
            "meta: nx={nx} exceeds sanity bound {MAX_NX_NY} — file may be hostile"
        ));
        geometry_bad = true;
    }
    if ny > MAX_NX_NY {
        report.error(format!(
            "meta: ny={ny} exceeds sanity bound {MAX_NX_NY} — file may be hostile"
        ));
        geometry_bad = true;
    }
    // Only check the product if both nx and ny are within bounds (geometry_bad
    // not yet set above); otherwise the multiply itself could overflow.
    if !geometry_bad && nx.saturating_mul(ny) > MAX_NX_NY_PRODUCT {
        report.error(format!(
            "meta: nx*ny={} exceeds sanity bound {MAX_NX_NY_PRODUCT} — file may be hostile",
            nx * ny
        ));
        geometry_bad = true;
    }
    for (field, value) in &[
        ("tile_y", tile_y),
        ("tile_x", tile_x),
        ("col_y", col_y),
        ("col_x", col_x),
    ] {
        if *value > MAX_CHUNK_DIM {
            report.error(format!(
                "meta: chunking.{field}={value} exceeds sanity bound {MAX_CHUNK_DIM} — file may be hostile"
            ));
            geometry_bad = true;
        }
    }
    for var in &meta.variables {
        if var.levels_hpa.len() > MAX_LEVELS {
            report.error(format!(
                "variable '{}': levels_hpa.len()={} exceeds sanity bound {MAX_LEVELS}",
                var.name,
                var.levels_hpa.len()
            ));
            geometry_bad = true;
        }
        if var.kind == "pressure3d" {
            match nx
                .checked_mul(ny)
                .and_then(|cells| cells.checked_mul(var.levels_hpa.len()))
            {
                Some(elements) if elements <= MAX_VOLUME_ELEMENTS => {}
                Some(elements) => {
                    report.error(format!(
                        "variable '{}': nx*ny*levels={elements} exceeds supported volume ceiling {MAX_VOLUME_ELEMENTS}",
                        var.name
                    ));
                    geometry_bad = true;
                }
                None => {
                    report.error(format!(
                        "variable '{}': nx*ny*levels overflows usize",
                        var.name
                    ));
                    geometry_bad = true;
                }
            }
        }
    }

    // Check 6: tile coords in range. Skipped entirely if geometry is bad (Issue
    // 1/sanity-bounds guard) since div_ceil would produce huge or hostile
    // values. Records with out-of-range coords are tracked in range_bad_records
    // so that Check 8 and the deep phase can skip them safely (Issue 2 guard).
    if !geometry_bad {
        for (i, rec) in records.iter().enumerate() {
            if let Some(var_meta) = var_id_map.get(&rec.var_id) {
                match var_meta.kind.as_str() {
                    "surface2d" => {
                        let max_ty = ny.div_ceil(tile_y) as u32;
                        let max_tx = nx.div_ceil(tile_x) as u32;
                        if rec.tile_y >= max_ty {
                            report.error(format!(
                                "index record {i} (var '{}'): tile_y {} >= max {} for ny={ny} tile_y={tile_y}",
                                var_meta.name, rec.tile_y, max_ty
                            ));
                            record_issues[i] |= ISSUE_RANGE;
                        }
                        if rec.tile_x >= max_tx {
                            report.error(format!(
                                "index record {i} (var '{}'): tile_x {} >= max {} for nx={nx} tile_x={tile_x}",
                                var_meta.name, rec.tile_x, max_tx
                            ));
                            record_issues[i] |= ISSUE_RANGE;
                        }
                    }
                    "pressure3d" => {
                        let max_ty = ny.div_ceil(col_y) as u32;
                        let max_tx = nx.div_ceil(col_x) as u32;
                        if rec.tile_y >= max_ty {
                            report.error(format!(
                                "index record {i} (var '{}'): tile_y {} >= max {} for ny={ny} col_y={col_y}",
                                var_meta.name, rec.tile_y, max_ty
                            ));
                            record_issues[i] |= ISSUE_RANGE;
                        }
                        if rec.tile_x >= max_tx {
                            report.error(format!(
                                "index record {i} (var '{}'): tile_x {} >= max {} for nx={nx} col_x={col_x}",
                                var_meta.name, rec.tile_x, max_tx
                            ));
                            record_issues[i] |= ISSUE_RANGE;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Check 7: payload spans within [payload_offset, file_len], no overlaps,
    // and len == 0 iff EMPTY or (CONSTANT without HAS_MISSING).
    // Records that fail span validation are tracked in span_bad_records so the
    // deep phase can safely skip them (Issue 3 gate).
    for (i, rec) in records.iter().enumerate() {
        let is_empty = rec.flags & FLAG_EMPTY != 0;
        let is_constant_no_missing =
            rec.flags & FLAG_CONSTANT != 0 && rec.flags & FLAG_HAS_MISSING == 0;
        let expect_zero_len = is_empty || is_constant_no_missing;

        if expect_zero_len && rec.len != 0 {
            report.error(format!(
                "index record {i} (var_id {}): EMPTY/CONSTANT-without-missing chunk has len {} != 0",
                rec.var_id, rec.len
            ));
            record_issues[i] |= ISSUE_SPAN;
        } else if !expect_zero_len && rec.len == 0 {
            report.error(format!(
                "index record {i} (var_id {}): non-empty chunk has len == 0",
                rec.var_id
            ));
            record_issues[i] |= ISSUE_SPAN;
        }

        if rec.len > 0 {
            // offset must be >= payload_offset and payload must not exceed file_len.
            if rec.offset < header.payload_offset {
                report.error(format!(
                    "index record {i} (var_id {}): offset {} < payload_offset {}",
                    rec.var_id, rec.offset, header.payload_offset
                ));
                record_issues[i] |= ISSUE_SPAN;
            }
            let end = match rec.offset.checked_add(rec.len as u64) {
                Some(e) => e,
                None => {
                    report.error(format!(
                        "index record {i} (var_id {}): offset+len overflows u64",
                        rec.var_id
                    ));
                    record_issues[i] |= ISSUE_SPAN | ISSUE_OVERFLOW;
                    continue;
                }
            };
            if end > file_len {
                report.error(format!(
                    "index record {i} (var_id {}): payload [{},{}] exceeds file_len {file_len}",
                    rec.var_id, rec.offset, end
                ));
                record_issues[i] |= ISSUE_SPAN;
            }
            report.stats.payload_bytes += rec.len as u64;
        }
    }

    // Check overlaps and build payload_records for Check 9. Records that
    // overflowed offset+len (overflow_records) are excluded — their end offset
    // is undefined and would panic or produce nonsense. Records that merely
    // exceed file_len are kept so Check 9 can still diagnose truncation.
    let mut payload_records: Vec<&ChunkRecord> = Vec::new();
    if let Err(err) = payload_records.try_reserve_exact(records.len()) {
        report.error(format!(
            "cannot allocate payload-span index for {} records: {err}",
            records.len()
        ));
        return;
    }
    payload_records.extend(
        records
            .iter()
            .enumerate()
            .filter(|(idx, r)| r.len > 0 && record_issues[*idx] & ISSUE_OVERFLOW == 0)
            .map(|(_, r)| r),
    );
    payload_records.sort_by_key(|r| r.offset);
    for pair in payload_records.windows(2) {
        // overflow_records are excluded so offset+len cannot wrap here.
        let end_a = pair[0].offset + pair[0].len as u64;
        if end_a > pair[1].offset {
            report.error(format!(
                "payload overlap: record ending at {end_a} overlaps record starting at {}",
                pair[1].offset
            ));
        }
    }

    // Check 8: expected raw_len per record. Skipped if geometry is bad (sanity
    // bounds or zero chunking — Issue 1) or if the record has out-of-range tile
    // coords (Issue 2) — both cases would cause subtract-with-overflow or
    // divide-by-zero panics.
    // Records that fail raw_len check are tracked so deep phase can skip them.
    if !geometry_bad {
        for (i, rec) in records.iter().enumerate() {
            // Skip records whose tile coords are out of range — their y0/x0
            // values would exceed ny/nx making (ny - y0) wrap around.
            if record_issues[i] & ISSUE_RANGE != 0 {
                continue;
            }

            let Some(var_meta) = var_id_map.get(&rec.var_id) else {
                continue;
            };
            let is_empty = rec.flags & FLAG_EMPTY != 0;
            let is_constant_no_missing =
                rec.flags & FLAG_CONSTANT != 0 && rec.flags & FLAG_HAS_MISSING == 0;

            if is_empty || is_constant_no_missing {
                if rec.raw_len != 0 {
                    report.error(format!(
                        "index record {i} (var '{}'): EMPTY/CONSTANT-without-missing chunk has raw_len {} != 0",
                        var_meta.name, rec.raw_len
                    ));
                }
                continue;
            }

            // Compute expected_raw_len with checked arithmetic throughout.
            // Even after the geometry_bad / range_bad guards, the .min(tile_y)
            // clamp result is bounded by the tile dimension (≤ MAX_CHUNK_DIM),
            // but we use checked_mul as belt-and-braces to make the invariant
            // explicit. On overflow we record the record as raw_len_bad so the
            // deep phase can skip it.
            let expected_raw_len: u32 = match var_meta.kind.as_str() {
                "surface2d" => {
                    // Belt-and-braces: use checked_mul so a hostile (but
                    // range-valid) tile coord cannot wrap in release mode.
                    let y0 = match (rec.tile_y as usize).checked_mul(tile_y) {
                        Some(v) => v,
                        None => {
                            report.error(format!(
                                "index record {i} (var '{}'): tile_y * tile_y overflows usize",
                                var_meta.name
                            ));
                            record_issues[i] |= ISSUE_RAW_LEN;
                            continue;
                        }
                    };
                    let x0 = match (rec.tile_x as usize).checked_mul(tile_x) {
                        Some(v) => v,
                        None => {
                            report.error(format!(
                                "index record {i} (var '{}'): tile_x * tile_x overflows usize",
                                var_meta.name
                            ));
                            record_issues[i] |= ISSUE_RAW_LEN;
                            continue;
                        }
                    };
                    let th = (ny - y0).min(tile_y);
                    let tw = (nx - x0).min(tile_x);
                    // Issue C: belt-and-braces checked_mul for th*tw*4.
                    match th.checked_mul(tw).and_then(|v| v.checked_mul(4)) {
                        Some(v) if v <= u32::MAX as usize => v as u32,
                        _ => {
                            report.error(format!(
                                "index record {i} (var '{}'): 4*th*tw overflows u32 (th={th}, tw={tw})",
                                var_meta.name
                            ));
                            record_issues[i] |= ISSUE_RAW_LEN;
                            continue;
                        }
                    }
                }
                "pressure3d" => {
                    let y0 = match (rec.tile_y as usize).checked_mul(col_y) {
                        Some(v) => v,
                        None => {
                            report.error(format!(
                                "index record {i} (var '{}'): tile_y * col_y overflows usize",
                                var_meta.name
                            ));
                            record_issues[i] |= ISSUE_RAW_LEN;
                            continue;
                        }
                    };
                    let x0 = match (rec.tile_x as usize).checked_mul(col_x) {
                        Some(v) => v,
                        None => {
                            report.error(format!(
                                "index record {i} (var '{}'): tile_x * col_x overflows usize",
                                var_meta.name
                            ));
                            record_issues[i] |= ISSUE_RAW_LEN;
                            continue;
                        }
                    };
                    let ch = (ny - y0).min(col_y);
                    let cw = (nx - x0).min(col_x);
                    let levels = var_meta.levels_hpa.len();
                    // Issue B: belt-and-braces checked_mul for ch*cw*levels*2.
                    match ch
                        .checked_mul(cw)
                        .and_then(|v| v.checked_mul(levels))
                        .and_then(|v| v.checked_mul(2))
                    {
                        Some(v) if v <= u32::MAX as usize => v as u32,
                        _ => {
                            report.error(format!(
                                "index record {i} (var '{}'): 2*ch*cw*levels overflows u32 (ch={ch}, cw={cw}, levels={levels})",
                                var_meta.name
                            ));
                            record_issues[i] |= ISSUE_RAW_LEN;
                            continue;
                        }
                    }
                }
                _ => continue,
            };

            if rec.raw_len != expected_raw_len {
                report.error(format!(
                    "index record {i} (var '{}'): raw_len {} != expected {expected_raw_len}",
                    var_meta.name, rec.raw_len
                ));
                record_issues[i] |= ISSUE_RAW_LEN;
            }
        }
    }

    // Check 9: file length == max over records of (offset+len), or payload_offset when none.
    let expected_file_len: u64 = payload_records
        .iter()
        .map(|r| r.offset + r.len as u64)
        .max()
        .unwrap_or(header.payload_offset);
    if file_len != expected_file_len {
        if file_len > expected_file_len {
            report.error(format!(
                "trailing bytes: file_len {file_len} > expected {expected_file_len} (trailing {} bytes)",
                file_len - expected_file_len
            ));
        } else {
            report.error(format!(
                "file truncated: file_len {file_len} < expected {expected_file_len}"
            ));
        }
    }

    // Check 10: per-variable chunk-set completeness. Skipped if geometry is
    // bad (sanity bounds or zero chunking — Issue 1/sanity guard) since
    // div_ceil would produce hostile tile counts and the multiply below would
    // panic or produce wrong results.
    if !geometry_bad && !index_sort_bad {
        for var_meta in &meta.variables {
            let (tiles_y, tiles_x, expected_kind) = match var_meta.kind.as_str() {
                "surface2d" => (ny.div_ceil(tile_y), nx.div_ceil(tile_x), KIND_TILE2D),
                "pressure3d" => (ny.div_ceil(col_y), nx.div_ceil(col_x), KIND_COLUMN3D),
                _ => continue,
            };
            // Issue A: checked_mul so hostile-but-bounded tile counts cannot
            // overflow in release mode. The geometry_bad guard above ensures
            // tiles_y and tiles_x are sane, but we keep this as belt-and-braces.
            let expected_count = match tiles_y.checked_mul(tiles_x) {
                Some(v) => v,
                None => {
                    report.error(format!(
                        "variable '{}': tiles_y*tiles_x overflows usize (tiles_y={tiles_y}, tiles_x={tiles_x})",
                        var_meta.name
                    ));
                    continue;
                }
            };
            // A valid index is strictly sorted, so expected-kind coordinates
            // are unique and ordered. Count/merge them directly instead of
            // allocating a potentially multi-million-entry HashSet.
            let total_for_var = records
                .iter()
                .enumerate()
                .filter(|(idx, r)| {
                    r.var_id == var_meta.id
                        && r.kind == expected_kind
                        && record_issues[*idx] & ISSUE_RANGE == 0
                })
                .count();
            if total_for_var != expected_count {
                let mut present = records
                    .iter()
                    .enumerate()
                    .filter(|(idx, r)| {
                        r.var_id == var_meta.id
                            && r.kind == expected_kind
                            && record_issues[*idx] & ISSUE_RANGE == 0
                    })
                    .map(|(_, r)| (r.tile_y, r.tile_x))
                    .peekable();
                let mut missing = Vec::new();
                'rows: for ty in 0..tiles_y as u32 {
                    for tx in 0..tiles_x as u32 {
                        let expected = (ty, tx);
                        while present.peek().is_some_and(|coord| *coord < expected) {
                            present.next();
                        }
                        if present.peek().is_some_and(|coord| *coord == expected) {
                            present.next();
                        } else {
                            missing.push(expected);
                            if missing.len() == 5 {
                                break 'rows;
                            }
                        }
                    }
                }
                let extra_msg = if expected_count.saturating_sub(total_for_var) > missing.len() {
                    " (too many missing to list)".to_string()
                } else {
                    String::new()
                };
                report.error(format!(
                    "variable '{}': expected {expected_count} chunks, found {}; missing tiles: {:?}{extra_msg}",
                    var_meta.name,
                    total_for_var,
                    missing
                ));
            }
        }
    }

    // Update stats.
    report.stats.variables = meta.variables.len();
    report.stats.chunks = records.len() as u64;

    // Structural done. Early-exit if not Deep.
    if depth == ValidateDepth::Structural {
        return;
    }

    // =======================================================================
    // Deep checks: decompress each non-empty chunk and verify content.
    // Structurally-invalid records (bad payload spans, out-of-range tile
    // coords, or raw_len mismatch) never reach this phase — they are gated
    // out to avoid panics or unbounded allocation on hostile input.
    // =======================================================================
    for (i, rec) in records.iter().enumerate() {
        // Absolute-backstop cap check: runs before all other gates so it fires
        // even when Check 8 also flagged the record (raw_len_bad_records). A
        // crafted geometry can make the geometry-expected raw_len exceed the cap
        // while still satisfying the Check 8 equality; this guard stops that
        // allocation regardless of how raw_len was reached.
        let is_empty_for_cap = rec.flags & FLAG_EMPTY != 0;
        let is_constant_no_missing_for_cap =
            rec.flags & FLAG_CONSTANT != 0 && rec.flags & FLAG_HAS_MISSING == 0;
        if !is_empty_for_cap
            && !is_constant_no_missing_for_cap
            && rec.raw_len > MAX_DEEP_CHUNK_RAW_LEN
        {
            if let Some(vm) = var_id_map.get(&rec.var_id) {
                report.error(format!(
                    "index record {i} (var '{}' chunk ({},{})): raw_len {} exceeds deep-validation cap {}",
                    vm.name, rec.tile_y, rec.tile_x, rec.raw_len, MAX_DEEP_CHUNK_RAW_LEN
                ));
            } else {
                report.error(format!(
                    "index record {i} (var_id {} chunk ({},{})): raw_len {} exceeds deep-validation cap {}",
                    rec.var_id, rec.tile_y, rec.tile_x, rec.raw_len, MAX_DEEP_CHUNK_RAW_LEN
                ));
            }
            continue;
        }

        // Gate: skip records that failed structural payload-span validation,
        // had out-of-range tile coords, or failed the raw_len geometry check
        // (Issue D guard: those records' raw_len is untrustworthy, and we must
        // not use it as a decompression-buffer allocation size).
        if record_issues[i] & (ISSUE_SPAN | ISSUE_RANGE | ISSUE_RAW_LEN) != 0 {
            continue;
        }

        let Some(var_meta) = var_id_map.get(&rec.var_id) else {
            continue;
        };
        let is_empty = rec.flags & FLAG_EMPTY != 0;
        let is_constant_no_missing =
            rec.flags & FLAG_CONSTANT != 0 && rec.flags & FLAG_HAS_MISSING == 0;

        // Check 14: EMPTY records.
        if is_empty {
            if rec.valid_count != 0 {
                report.error(format!(
                    "index record {i} (var '{}'): EMPTY chunk has valid_count {} != 0",
                    var_meta.name, rec.valid_count
                ));
            }
            if rec.min.is_finite() {
                report.error(format!(
                    "index record {i} (var '{}'): EMPTY chunk min should be NaN, got {}",
                    var_meta.name, rec.min
                ));
            }
            if rec.max.is_finite() {
                report.error(format!(
                    "index record {i} (var '{}'): EMPTY chunk max should be NaN, got {}",
                    var_meta.name, rec.max
                ));
            }
            continue;
        }

        // CONSTANT without missing: no payload.
        if is_constant_no_missing {
            continue;
        }

        // Check 11: decompress. Use checked_add for the end offset (Issue 3
        // belt-and-braces: even though span_bad_records already gates overflow
        // cases, be explicit here to avoid any latent panic in release mode).
        let end = match rec.offset.checked_add(rec.len as u64) {
            Some(e) => e,
            None => {
                report.error(format!(
                    "index record {i} (var '{}'): offset+len overflows u64 in deep phase",
                    var_meta.name
                ));
                continue;
            }
        };
        if end > file_len {
            continue; // already reported in structural
        }
        let compressed = &data[rec.offset as usize..end as usize];
        // Issue D: use rec.raw_len as the allocation cap for zstd::bulk::decompress.
        // Records that failed Check 8 (raw_len != expected_geometry) are excluded by
        // the raw_len_bad_records gate above, so raw_len here equals the geometry-
        // derived expected_raw_len which is bounded by the sanity checks. A crafted
        // zstd frame without an embedded content-size header will request allocation
        // of the cap argument; using the validated geometry-derived raw_len keeps
        // this bounded and prevents memory-DoS from a multi-GiB fake raw_len.
        let raw = match zstd::bulk::decompress(compressed, rec.raw_len as usize) {
            Ok(r) => r,
            Err(err) => {
                report.error(format!(
                    "index record {i} (var '{}' chunk ({},{})): zstd decompress failed: {err}",
                    var_meta.name, rec.tile_y, rec.tile_x
                ));
                continue;
            }
        };
        if raw.len() != rec.raw_len as usize {
            report.error(format!(
                "index record {i} (var '{}' chunk ({},{})): decompressed {} bytes, raw_len says {}",
                var_meta.name,
                rec.tile_y,
                rec.tile_x,
                raw.len(),
                rec.raw_len
            ));
            continue;
        }

        match var_meta.kind.as_str() {
            "surface2d" => check_tile2d_deep(i, rec, var_meta, &raw, report),
            "pressure3d" => check_column3d_deep(i, rec, var_meta, &raw, report),
            _ => {}
        }
    }
}

fn check_tile2d_deep(
    i: usize,
    rec: &ChunkRecord,
    var_meta: &crate::format::RwsVariableMeta,
    raw: &[u8],
    report: &mut ValidationReport,
) {
    // Check 12: TILE2D: f32 count == th*tw; finite-count == valid_count;
    // finite min/max == record min/max; HAS_MISSING iff any non-finite.
    if raw.len() % 4 != 0 {
        report.error(format!(
            "index record {i} (var '{}' tile ({},{})): raw payload length {} not divisible by 4",
            var_meta.name,
            rec.tile_y,
            rec.tile_x,
            raw.len()
        ));
        return;
    }
    let values: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    let mut finite_min = f32::INFINITY;
    let mut finite_max = f32::NEG_INFINITY;
    let mut finite_count: u32 = 0;
    let mut has_non_finite = false;
    for &v in &values {
        if v.is_finite() {
            finite_min = finite_min.min(v);
            finite_max = finite_max.max(v);
            finite_count += 1;
        } else {
            has_non_finite = true;
        }
    }

    if finite_count != rec.valid_count {
        report.error(format!(
            "index record {i} (var '{}' tile ({},{})): decoded finite_count {finite_count} != valid_count {}",
            var_meta.name, rec.tile_y, rec.tile_x, rec.valid_count
        ));
    }

    if finite_count > 0 {
        if rec.min.to_bits() != finite_min.to_bits() {
            report.error(format!(
                "index record {i} (var '{}' tile ({},{})): decoded min {} != record min {}",
                var_meta.name, rec.tile_y, rec.tile_x, finite_min, rec.min
            ));
        }
        if rec.max.to_bits() != finite_max.to_bits() {
            report.error(format!(
                "index record {i} (var '{}' tile ({},{})): decoded max {} != record max {}",
                var_meta.name, rec.tile_y, rec.tile_x, finite_max, rec.max
            ));
        }
    } else {
        // valid_count == 0: min/max should be NaN
        if rec.min.is_finite() {
            report.error(format!(
                "index record {i} (var '{}' tile ({},{})): valid_count==0 but min is {}",
                var_meta.name, rec.tile_y, rec.tile_x, rec.min
            ));
        }
        if rec.max.is_finite() {
            report.error(format!(
                "index record {i} (var '{}' tile ({},{})): valid_count==0 but max is {}",
                var_meta.name, rec.tile_y, rec.tile_x, rec.max
            ));
        }
    }

    let has_missing_flag = rec.flags & FLAG_HAS_MISSING != 0;
    if has_non_finite != has_missing_flag {
        report.error(format!(
            "index record {i} (var '{}' tile ({},{})): HAS_MISSING flag {} but has_non_finite {}",
            var_meta.name, rec.tile_y, rec.tile_x, has_missing_flag, has_non_finite
        ));
    }
}

fn check_column3d_deep(
    i: usize,
    rec: &ChunkRecord,
    var_meta: &crate::format::RwsVariableMeta,
    raw: &[u8],
    report: &mut ValidationReport,
) {
    // Check 13: COLUMN3D: i16 count; count of non-MISSING_Q == valid_count;
    // HAS_MISSING iff any MISSING_Q; CONSTANT => scale==0 and all non-missing q==0.
    if raw.len() % 2 != 0 {
        report.error(format!(
            "index record {i} (var '{}' chunk ({},{})): raw payload length {} not divisible by 2",
            var_meta.name,
            rec.tile_y,
            rec.tile_x,
            raw.len()
        ));
        return;
    }
    let quants: Vec<i16> = raw
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes(b.try_into().unwrap()))
        .collect();

    let valid_count: u32 = quants.iter().filter(|&&q| q != MISSING_Q).count() as u32;
    let has_missing = quants.contains(&MISSING_Q);

    if valid_count != rec.valid_count {
        report.error(format!(
            "index record {i} (var '{}' chunk ({},{})): decoded valid_count {valid_count} != record valid_count {}",
            var_meta.name, rec.tile_y, rec.tile_x, rec.valid_count
        ));
    }

    let has_missing_flag = rec.flags & FLAG_HAS_MISSING != 0;
    if has_missing != has_missing_flag {
        report.error(format!(
            "index record {i} (var '{}' chunk ({},{})): HAS_MISSING flag {} but has_missing_q {}",
            var_meta.name, rec.tile_y, rec.tile_x, has_missing_flag, has_missing
        ));
    }

    let is_constant = rec.flags & FLAG_CONSTANT != 0;
    if is_constant {
        if rec.scale != 0.0 {
            report.error(format!(
                "index record {i} (var '{}' chunk ({},{})): CONSTANT chunk has scale {} != 0",
                var_meta.name, rec.tile_y, rec.tile_x, rec.scale
            ));
        }
        for &q in quants.iter().filter(|&&q| q != MISSING_Q) {
            if q != 0 {
                report.error(format!(
                    "index record {i} (var '{}' chunk ({},{})): CONSTANT chunk has non-missing q={q} != 0",
                    var_meta.name, rec.tile_y, rec.tile_x
                ));
                break; // report once per chunk
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::HourWriter;
    use rustwx_core::{GridShape, LatLonGrid};
    use std::fs;
    use std::path::PathBuf;

    const NX: usize = 40;
    const NY: usize = 30;

    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("rw-store-validate-{}-{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Build a small valid hour file with one 2D var (including a NaN hole)
    /// and one 3D var with 3 levels.
    fn write_valid_hour(path: &std::path::Path) {
        let mut values_2d: Vec<f32> = (0..NY)
            .flat_map(|y| (0..NX).map(move |x| 0.1 * x as f32 + 0.2 * y as f32))
            .collect();
        // A NaN hole in the middle.
        for y in 5..10 {
            for x in 5..10 {
                values_2d[y * NX + x] = f32::NAN;
            }
        }

        let levels: [u16; 3] = [1000, 500, 200];
        let planes: Vec<Vec<f32>> = levels
            .iter()
            .map(|&level| {
                (0..NY)
                    .flat_map(|_y| {
                        (0..NX).map(move |x| 100.0 + 0.5 * x as f32 + 0.01 * level as f32)
                    })
                    .collect()
            })
            .collect();
        let plane_refs: Vec<&[f32]> = planes.iter().map(|p| p.as_slice()).collect();

        let mut writer = HourWriter::new(
            "hrrr",
            "2026-06-10T00:00:00Z",
            3,
            NX,
            NY,
            "test-grid-hash",
            "validate-test",
        );
        writer
            .add_surface2d(
                "temp_2m",
                "K",
                serde_json::json!({"test": true}),
                &values_2d,
            )
            .unwrap();
        writer
            .add_pressure3d(
                "wind_iso",
                "m/s",
                serde_json::json!({"test3d": true}),
                &levels,
                &plane_refs,
            )
            .unwrap();
        writer.finish(path).unwrap();
    }

    /// Build a regular lat/lon grid for use in run-dir tests.
    fn small_grid() -> LatLonGrid {
        let mut lat = Vec::with_capacity(NX * NY);
        let mut lon = Vec::with_capacity(NX * NY);
        for y in 0..NY {
            for x in 0..NX {
                lat.push((35.0 + 0.1 * y as f64) as f32);
                lon.push((-100.0 + 0.1 * x as f64) as f32);
            }
        }
        LatLonGrid::new(GridShape::new(NX, NY).unwrap(), lat, lon).unwrap()
    }

    // -----------------------------------------------------------------------
    // Happy-path tests
    // -----------------------------------------------------------------------

    #[test]
    fn valid_hour_structural_ok() {
        let dir = test_dir("valid-structural");
        let path = dir.join("f003.rws");
        write_valid_hour(&path);
        let report = validate_hour_file(&path, ValidateDepth::Structural).unwrap();
        assert!(
            report.is_ok(),
            "valid file must pass structural: {:?}",
            report.errors
        );
        assert_eq!(report.stats.variables, 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn valid_hour_deep_ok_with_two_variables() {
        let dir = test_dir("valid-deep");
        let path = dir.join("f003.rws");
        write_valid_hour(&path);
        let report = validate_hour_file(&path, ValidateDepth::Deep).unwrap();
        assert!(
            report.is_ok(),
            "valid file must pass deep: {:?}",
            report.errors
        );
        assert_eq!(report.stats.variables, 2, "expected 2 variables");
        assert!(report.stats.chunks > 0, "expected at least one chunk");
        assert!(
            report.stats.payload_bytes > 0,
            "expected non-zero payload bytes"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------------
    // Corruption tests (each from a fresh copy of valid bytes)
    // -----------------------------------------------------------------------

    fn load_valid_bytes(dir: &std::path::Path) -> Vec<u8> {
        let path = dir.join("base.rws");
        write_valid_hour(&path);
        fs::read(&path).unwrap()
    }

    fn write_corrupt(dir: &std::path::Path, name: &str, bytes: &[u8]) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn truncated_file_reports_error() {
        let dir = test_dir("truncate");
        let mut bytes = load_valid_bytes(&dir);
        let orig_len = bytes.len();
        bytes.truncate(orig_len - 10);
        let path = write_corrupt(&dir, "truncated.rws", &bytes);
        let report = validate_hour_file(&path, ValidateDepth::Structural).unwrap();
        assert!(
            !report.is_ok(),
            "truncated file must fail; errors: {:?}",
            report.errors
        );
        let joined = report.errors.join(" ");
        assert!(
            joined.contains("trun") || joined.contains("bound") || joined.contains("length"),
            "error should mention truncation/bounds/length: {joined}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn swapped_index_records_reports_sort_error() {
        let dir = test_dir("swap-index");
        let mut bytes = load_valid_bytes(&dir);
        let header = RwsHeader::parse(&bytes).unwrap();
        let start = header.index_offset as usize;
        // Swap the first two 64-byte index records.
        let (a, b) = (
            bytes[start..start + 64].to_vec(),
            bytes[start + 64..start + 128].to_vec(),
        );
        bytes[start..start + 64].copy_from_slice(&b);
        bytes[start + 64..start + 128].copy_from_slice(&a);
        let path = write_corrupt(&dir, "swapped.rws", &bytes);
        let report = validate_hour_file(&path, ValidateDepth::Structural).unwrap();
        assert!(
            !report.is_ok(),
            "swapped records must fail; errors: {:?}",
            report.errors
        );
        let joined = report.errors.join(" ");
        assert!(
            joined.contains("sort") || joined.contains("order"),
            "error should mention sort/order: {joined}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_payload_deep_only() {
        let dir = test_dir("corrupt-payload");
        let bytes = load_valid_bytes(&dir);
        let header = RwsHeader::parse(&bytes).unwrap();

        // Find first dense chunk record.
        let record_count = header.index_count as usize;
        let mut dense_rec: Option<ChunkRecord> = None;
        for i in 0..record_count {
            let start = header.index_offset as usize + i * 64;
            let rec = ChunkRecord::unpack(&bytes[start..start + 64]).unwrap();
            if rec.len > 0 {
                dense_rec = Some(rec);
                break;
            }
        }
        let dense_rec = dense_rec.expect("must have at least one dense chunk");

        let mut corrupt_bytes = bytes.clone();
        let off = dense_rec.offset as usize;
        // Overwrite 4 mid-payload bytes with 0xFF.
        for b in &mut corrupt_bytes[off..off + 4] {
            *b = 0xFF;
        }
        let path = write_corrupt(&dir, "corrupt.rws", &corrupt_bytes);

        // Structural must pass (no decompression).
        let structural = validate_hour_file(&path, ValidateDepth::Structural).unwrap();
        assert!(
            structural.is_ok(),
            "structural must pass corrupt payload: {:?}",
            structural.errors
        );

        // Deep must report decompress error.
        let deep = validate_hour_file(&path, ValidateDepth::Deep).unwrap();
        assert!(
            !deep.is_ok(),
            "deep must fail corrupt payload; errors: {:?}",
            deep.errors
        );
        let joined = deep.errors.join(" ");
        assert!(
            joined.contains("decompress") || joined.contains("zstd"),
            "error should mention decompress/zstd: {joined}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn wrong_raw_len_reports_error() {
        let dir = test_dir("raw-len");
        let mut bytes = load_valid_bytes(&dir);
        let header = RwsHeader::parse(&bytes).unwrap();

        // Find first dense chunk and bump its raw_len by 2.
        for i in 0..header.index_count as usize {
            let start = header.index_offset as usize + i * 64;
            let rec = ChunkRecord::unpack(&bytes[start..start + 64]).unwrap();
            if rec.len > 0 {
                let current = u32::from_le_bytes(bytes[start + 24..start + 28].try_into().unwrap());
                bytes[start + 24..start + 28].copy_from_slice(&(current + 2).to_le_bytes());
                break;
            }
        }
        let path = write_corrupt(&dir, "rawlen.rws", &bytes);
        let report = validate_hour_file(&path, ValidateDepth::Structural).unwrap();
        assert!(
            !report.is_ok(),
            "wrong raw_len must fail; errors: {:?}",
            report.errors
        );
        assert!(
            report.errors.iter().any(|e| e.contains("raw_len")),
            "error should mention raw_len: {:?}",
            report.errors
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn appended_junk_reports_trailing_error() {
        let dir = test_dir("trailing");
        let mut bytes = load_valid_bytes(&dir);
        bytes.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        let path = write_corrupt(&dir, "trailing.rws", &bytes);
        let report = validate_hour_file(&path, ValidateDepth::Structural).unwrap();
        assert!(
            !report.is_ok(),
            "trailing bytes must fail; errors: {:?}",
            report.errors
        );
        let joined = report.errors.join(" ");
        assert!(
            joined.contains("trailing"),
            "error should mention trailing: {joined}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn bad_version_reports_error_not_panic() {
        let dir = test_dir("bad-version");
        let mut bytes = load_valid_bytes(&dir);
        // Set version field (bytes 8..12) to 9.
        bytes[8..12].copy_from_slice(&9u32.to_le_bytes());
        let path = write_corrupt(&dir, "badversion.rws", &bytes);
        // Must return Ok(report_with_error), not Err or panic.
        let report = validate_hour_file(&path, ValidateDepth::Structural).unwrap();
        assert!(
            !report.is_ok(),
            "bad version must produce errors: {:?}",
            report.errors
        );
        let _ = fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------------
    // Hostile-input / panic-guard tests
    // -----------------------------------------------------------------------

    /// A file whose meta JSON has chunking.tile_y=0 must return Ok(report) with
    /// errors mentioning "chunking", never panic with divide-by-zero.
    #[test]
    fn zero_chunking_reports_error_not_panic() {
        let dir = test_dir("zero-chunking");
        let bytes = load_valid_bytes(&dir);
        let header = RwsHeader::parse(&bytes).unwrap();

        // Locate the meta region and replace "tile_y":256 with "tile_y":  0
        // (same byte length, value becomes 0; leading spaces are valid JSON).
        let meta_start = HEADER_LEN;
        let meta_end = meta_start + header.meta_len as usize;
        let meta_str = std::str::from_utf8(&bytes[meta_start..meta_end]).unwrap();
        assert!(
            meta_str.contains("\"tile_y\":256"),
            "test assumption: meta must contain tile_y:256; got: {meta_str}"
        );
        // "tile_y":256  (12 chars) -> "tile_y":  0  (12 chars, valid JSON, value=0)
        let patched_meta = meta_str.replace("\"tile_y\":256", "\"tile_y\":  0");
        assert_eq!(
            patched_meta.len(),
            meta_str.len(),
            "patched meta must have same byte length to keep header offsets valid"
        );

        let mut corrupt = bytes.clone();
        corrupt[meta_start..meta_end].copy_from_slice(patched_meta.as_bytes());
        let path = write_corrupt(&dir, "zero-chunking.rws", &corrupt);

        let structural = validate_hour_file(&path, ValidateDepth::Structural).unwrap();
        assert!(
            !structural.is_ok(),
            "zero chunking must produce errors (structural): {:?}",
            structural.errors
        );
        let joined_s = structural.errors.join(" ");
        assert!(
            joined_s.contains("chunking"),
            "structural error must mention 'chunking': {joined_s}"
        );

        let deep = validate_hour_file(&path, ValidateDepth::Deep).unwrap();
        assert!(
            !deep.is_ok(),
            "zero chunking must produce errors (deep): {:?}",
            deep.errors
        );
        let joined_d = deep.errors.join(" ");
        assert!(
            joined_d.contains("chunking"),
            "deep error must mention 'chunking': {joined_d}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// A file with a record whose tile_y is out of range (9999) must return
    /// Ok(report) with errors mentioning the out-of-range tile, never panic
    /// with subtract-with-overflow.
    #[test]
    fn out_of_range_tile_reports_error_not_panic() {
        let dir = test_dir("oor-tile");
        let bytes = load_valid_bytes(&dir);
        let header = RwsHeader::parse(&bytes).unwrap();

        // Patch the first index record's tile_y field (bytes 4..8 within the
        // 64-byte record) to 9999.
        let rec_start = header.index_offset as usize;
        let mut corrupt = bytes.clone();
        corrupt[rec_start + 4..rec_start + 8].copy_from_slice(&9999u32.to_le_bytes());
        let path = write_corrupt(&dir, "oor-tile.rws", &corrupt);

        let structural = validate_hour_file(&path, ValidateDepth::Structural).unwrap();
        assert!(
            !structural.is_ok(),
            "out-of-range tile must produce errors (structural): {:?}",
            structural.errors
        );
        let joined_s = structural.errors.join(" ");
        assert!(
            joined_s.contains("tile_y") || joined_s.contains("range") || joined_s.contains("tile"),
            "structural error must mention tile range: {joined_s}"
        );

        let deep = validate_hour_file(&path, ValidateDepth::Deep).unwrap();
        assert!(
            !deep.is_ok(),
            "out-of-range tile must produce errors (deep): {:?}",
            deep.errors
        );
        let joined_d = deep.errors.join(" ");
        assert!(
            joined_d.contains("tile_y") || joined_d.contains("range") || joined_d.contains("tile"),
            "deep error must mention tile range: {joined_d}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// A file with a record whose offset is u64::MAX-1 must return Ok(report)
    /// with errors, never panic with integer overflow in the deep phase.
    #[test]
    fn huge_offset_record_no_panic_deep() {
        let dir = test_dir("huge-offset");
        let bytes = load_valid_bytes(&dir);
        let header = RwsHeader::parse(&bytes).unwrap();

        // Find the first dense chunk record and set its offset to u64::MAX-1.
        // The index record offset field is bytes 12..20 within the 64-byte record.
        let record_count = header.index_count as usize;
        let mut target_rec_start: Option<usize> = None;
        for i in 0..record_count {
            let start = header.index_offset as usize + i * INDEX_RECORD_LEN;
            let rec = ChunkRecord::unpack(&bytes[start..start + INDEX_RECORD_LEN]).unwrap();
            if rec.len > 0 {
                target_rec_start = Some(start);
                break;
            }
        }
        let rec_start = target_rec_start.expect("must have at least one dense chunk");
        let mut corrupt = bytes.clone();
        corrupt[rec_start + 12..rec_start + 20].copy_from_slice(&(u64::MAX - 1).to_le_bytes());
        let path = write_corrupt(&dir, "huge-offset.rws", &corrupt);

        let structural = validate_hour_file(&path, ValidateDepth::Structural).unwrap();
        assert!(
            !structural.is_ok(),
            "huge offset must produce errors (structural): {:?}",
            structural.errors
        );

        let deep = validate_hour_file(&path, ValidateDepth::Deep).unwrap();
        assert!(
            !deep.is_ok(),
            "huge offset must produce errors (deep): {:?}",
            deep.errors
        );

        let _ = fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------------
    // validate_run_dir tests
    // -----------------------------------------------------------------------

    #[test]
    fn valid_run_dir_deep_ok() {
        use rustwx_core::{FieldSelector, GridProjection, SelectedField2D};

        let dir = test_dir("rundir-ok");
        let store_root = dir.join("store");
        let grid = small_grid();

        let values: Vec<f32> = (0..NY)
            .flat_map(|y| (0..NX).map(move |x| 280.0 + 0.05 * x as f32 - 0.02 * y as f32))
            .collect();
        let selector = FieldSelector::height_agl(rustwx_core::CanonicalField::Temperature, 2);
        let field = SelectedField2D::new(selector, "K", grid, values)
            .unwrap()
            .with_projection(GridProjection::Geographic);

        crate::ingest::write_hour_from_fields(
            &store_root,
            "hrrr",
            "20260610_00z",
            3,
            &[("temp_2m", &field)],
            &[],
            "validate-test",
            1_000_000,
        )
        .unwrap();

        let run_dir_path = store_root.join("hrrr").join("20260610_00z");
        let report = validate_run_dir(&run_dir_path, ValidateDepth::Deep).unwrap();
        assert!(
            report.is_ok(),
            "valid run dir must pass deep: {:?}",
            report.errors
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_hour_file_reports_error() {
        use rustwx_core::{FieldSelector, GridProjection, SelectedField2D};

        let dir = test_dir("rundir-missing-hour");
        let store_root = dir.join("store");
        let grid = small_grid();

        let values: Vec<f32> = vec![280.0f32; NX * NY];
        let selector = FieldSelector::height_agl(rustwx_core::CanonicalField::Temperature, 2);
        let field = SelectedField2D::new(selector, "K", grid, values)
            .unwrap()
            .with_projection(GridProjection::Geographic);

        crate::ingest::write_hour_from_fields(
            &store_root,
            "hrrr",
            "20260610_00z",
            3,
            &[("temp_2m", &field)],
            &[],
            "validate-test",
            1_000_000,
        )
        .unwrap();

        let run_dir_path = store_root.join("hrrr").join("20260610_00z");
        // Delete the hour file.
        fs::remove_file(run_dir_path.join("f003.rws")).unwrap();
        let report = validate_run_dir(&run_dir_path, ValidateDepth::Structural).unwrap();
        assert!(
            !report.is_ok(),
            "missing hour file must produce an error: {:?}",
            report.errors
        );
        assert!(
            report.errors.iter().any(|e| e.contains("f003.rws")),
            "error must mention the missing file: {:?}",
            report.errors
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn stray_rws_file_reports_warning() {
        use rustwx_core::{FieldSelector, GridProjection, SelectedField2D};

        let dir = test_dir("rundir-stray");
        let store_root = dir.join("store");
        let grid = small_grid();

        let values: Vec<f32> = vec![280.0f32; NX * NY];
        let selector = FieldSelector::height_agl(rustwx_core::CanonicalField::Temperature, 2);
        let field = SelectedField2D::new(selector, "K", grid, values)
            .unwrap()
            .with_projection(GridProjection::Geographic);

        crate::ingest::write_hour_from_fields(
            &store_root,
            "hrrr",
            "20260610_00z",
            3,
            &[("temp_2m", &field)],
            &[],
            "validate-test",
            1_000_000,
        )
        .unwrap();

        let run_dir_path = store_root.join("hrrr").join("20260610_00z");
        // Add a stray .rws file not in the manifest.
        fs::write(run_dir_path.join("f099.rws"), b"garbage").unwrap();
        let report = validate_run_dir(&run_dir_path, ValidateDepth::Structural).unwrap();
        assert!(
            report.warnings.iter().any(|w| w.contains("f099.rws")),
            "stray file must produce a warning: {:?}",
            report.warnings
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn oversized_run_manifest_is_reported_without_unbounded_read() {
        let dir = test_dir("rundir-oversized-manifest");
        let run_dir_path = dir.join("store").join("hrrr").join("20260610_00z");
        fs::create_dir_all(&run_dir_path).unwrap();
        let file = fs::File::create(run_dir_path.join("run.json")).unwrap();
        file.set_len(crate::run::MAX_RUN_MANIFEST_BYTES + 1)
            .unwrap();

        let report = validate_run_dir(&run_dir_path, ValidateDepth::Structural).unwrap();
        assert!(!report.is_ok());
        assert!(
            report.errors.iter().any(|error| error.contains("limit")),
            "oversized manifest error must name the bound: {:?}",
            report.errors
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn oversized_meta_region_is_rejected_from_sparse_file() {
        let dir = test_dir("oversized-meta-region");
        let path = dir.join("f000.rws");
        let header = RwsHeader::for_layout((MAX_HOUR_META_BYTES + 1) as u32, 0);
        fs::write(&path, header.pack()).unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(header.payload_offset)
            .unwrap();

        let report = validate_hour_file(&path, ValidateDepth::Structural).unwrap();
        assert!(
            report.errors.iter().any(|error| error.contains("limit")),
            "oversized metadata must report the bound: {:?}",
            report.errors
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sparse_trailing_region_is_validated_without_file_sized_allocation() {
        let dir = test_dir("sparse-trailing-region");
        let path = dir.join("f003.rws");
        write_valid_hour(&path);
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(64 * 1024 * 1024)
            .unwrap();

        let report = validate_hour_file(&path, ValidateDepth::Structural).unwrap();
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("trailing bytes")),
            "sparse tail must be diagnosed without reading it: {:?}",
            report.errors
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn impossible_index_capacity_fails_fallibly() {
        let error = try_record_buffer(usize::MAX).unwrap_err();
        assert!(
            error.contains("cannot allocate index"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn excessive_index_count_is_rejected_from_header_only() {
        let header = RwsHeader::for_layout(0, MAX_INDEX_RECORDS + 1).pack();
        let mut report = ValidationReport::default();
        check_hour_file(&header, ValidateDepth::Structural, &mut report);
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("index_count")),
            "unexpected errors: {:?}",
            report.errors
        );
    }

    // -----------------------------------------------------------------------
    // Path-traversal rejection test
    // -----------------------------------------------------------------------

    /// A run.json with two hostile `file` values — one absolute path and one
    /// `..`-relative path — must be rejected before any filesystem join occurs.
    ///
    /// Assertions:
    ///  - `validate_run_dir` returns `Ok` (no I/O error)
    ///  - `report.is_ok()` is false (at least one error)
    ///  - every error for the hostile hours contains "not a plain filename"
    ///  - no error or warning text contains content from the absolute-path target
    ///    (guards against reading outside the run dir and leaking file content
    ///    via error messages such as the bad-magic bytes echo)
    #[test]
    fn run_dir_rejects_traversal_file_entries() {
        use crate::run::{RwsHourEntry, RwsRunManifest};
        use rustwx_core::{FieldSelector, GridProjection, SelectedField2D};

        // Build a valid run dir so we have a real grid.rwg and baseline run.json.
        let dir = test_dir("rundir-traversal");
        let store_root = dir.join("store");
        let grid = small_grid();

        let values: Vec<f32> = vec![280.0f32; NX * NY];
        let selector = FieldSelector::height_agl(rustwx_core::CanonicalField::Temperature, 2);
        let field = SelectedField2D::new(selector, "K", grid, values)
            .unwrap()
            .with_projection(GridProjection::Geographic);

        crate::ingest::write_hour_from_fields(
            &store_root,
            "hrrr",
            "20260610_00z",
            3,
            &[("temp_2m", &field)],
            &[],
            "validate-test",
            1_000_000,
        )
        .unwrap();

        let run_dir_path = store_root.join("hrrr").join("20260610_00z");

        // Load the real manifest so we can borrow its schema/grid_hash/etc.
        let manifest_path = run_dir_path.join("run.json");
        let mut manifest: RwsRunManifest =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();

        // Pick a file that exists outside the run dir as the absolute-path target.
        // Using grid.rwg itself (a real file in the run dir's parent hierarchy)
        // is the clearest choice — we know it exists and has real bytes.
        let outside_file = run_dir_path.join("grid.rwg");
        let absolute_target = outside_file.to_string_lossy().into_owned();

        // Inject two hostile hour entries.
        manifest.hours.insert(
            6,
            RwsHourEntry {
                file: absolute_target.clone(),
                lead_seconds: None,
                valid_unix: None,
                written_unix: 0,
                encode_ms: 0,
                variables: vec!["temp_2m".to_string()],
            },
        );
        manifest.hours.insert(
            9,
            RwsHourEntry {
                file: r"..\evil.rws".to_string(),
                lead_seconds: None,
                valid_unix: None,
                written_unix: 0,
                encode_ms: 0,
                variables: vec!["temp_2m".to_string()],
            },
        );

        // Overwrite run.json with the tampered manifest.
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let report = validate_run_dir(&run_dir_path, ValidateDepth::Structural).unwrap();

        // Must have errors (not ok).
        assert!(
            !report.is_ok(),
            "traversal entries must produce errors; errors: {:?}",
            report.errors
        );

        // Both hostile hours must produce path-rejection errors.
        let errors_for_h6 = report
            .errors
            .iter()
            .any(|e| e.contains("hour 6") && e.contains("must be a relative path"));
        let errors_for_h9 = report
            .errors
            .iter()
            .any(|e| e.contains("hour 9") && e.contains("must be a relative path"));
        assert!(
            errors_for_h6,
            "hour 6 (absolute path) must produce a path-rejection error; errors: {:?}",
            report.errors
        );
        assert!(
            errors_for_h9,
            "hour 9 (.. traversal) must produce a path-rejection error; errors: {:?}",
            report.errors
        );

        // Key: no error or warning text must contain content bytes from the
        // absolute-path target. We read the first 8 bytes of grid.rwg to know
        // what a content-leak would look like (the bad-magic error path echoes
        // raw bytes). Then assert none of the error/warning strings contain it.
        let target_bytes = fs::read(&outside_file).unwrap();
        if target_bytes.len() >= 8 {
            let snippet = format!("{:?}", &target_bytes[..8]);
            let all_messages: Vec<&str> = report
                .errors
                .iter()
                .chain(report.warnings.iter())
                .map(|s| s.as_str())
                .collect();
            for msg in &all_messages {
                assert!(
                    !msg.contains(&snippet),
                    "error/warning must not contain content bytes from outside target: {msg}"
                );
            }
        }

        let _ = fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------------
    // Hostile geometry tests: huge nx/ny/chunking must not panic
    // -----------------------------------------------------------------------

    /// Build a minimal rws file from raw meta JSON + empty index + no payload.
    /// The header is constructed via RwsHeader::for_layout so offsets are
    /// consistent. Zero index records keeps the file small.
    fn build_file_with_meta(meta_json: &str) -> Vec<u8> {
        let meta_bytes = meta_json.as_bytes();
        let meta_len = meta_bytes.len() as u32;
        let header = RwsHeader::for_layout(meta_len, 0);
        let mut out = Vec::with_capacity(crate::format::HEADER_LEN + meta_bytes.len());
        out.extend_from_slice(&header.pack());
        out.extend_from_slice(meta_bytes);
        // No index records; payload immediately follows (payload_offset == file_len).
        out
    }

    #[test]
    fn pressure_volume_over_supported_ceiling_is_reported() {
        let meta_json = r#"{
            "schema": "rw-store.hour.v1",
            "model": "hrrr",
            "run": "20260610_00z",
            "forecast_hour": 3,
            "nx": 5000,
            "ny": 5000,
            "grid_hash": "aaaa",
            "variables": [{
                "id": 0,
                "name": "temperature",
                "units": "K",
                "kind": "pressure3d",
                "codec": "zstd1_affine_i16",
                "levels_hpa": [1000, 900, 800, 700, 600, 500],
                "selector": {}
            }],
            "chunking": {"tile_y": 256, "tile_x": 256, "col_y": 16, "col_x": 16},
            "writer": {"name": "test", "version": "0", "build": "0"}
        }"#;
        let bytes = build_file_with_meta(meta_json);
        let dir = test_dir("volume-ceiling");
        let path = write_corrupt(&dir, "volume-ceiling.rws", &bytes);

        let report = validate_hour_file(&path, ValidateDepth::Structural).unwrap();
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("volume ceiling")),
            "unexpected errors: {:?}",
            report.errors
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// nx = ny = 18_000_000_000_000_000_000 (fits u64; passes the nx>0 guard but
    /// exceeds the sanity bound). Should produce Ok(report) with errors mentioning
    /// nx or ny or "bound", at both depths.  Must not panic.
    #[test]
    fn huge_grid_dims_report_error_not_panic() {
        // A single surface2d variable so the variable-validation path runs.
        let meta_json = r#"{
            "schema": "rw-store.hour.v1",
            "model": "hrrr",
            "run": "2026-06-10T00:00:00Z",
            "forecast_hour": 3,
            "nx": 18000000000000000000,
            "ny": 18000000000000000000,
            "grid_hash": "aaaa",
            "variables": [
                {
                    "id": 0,
                    "name": "temp_2m",
                    "units": "K",
                    "kind": "surface2d",
                    "codec": "zstd1_f32",
                    "levels_hpa": [],
                    "selector": {}
                }
            ],
            "chunking": {"tile_y": 256, "tile_x": 256, "col_y": 16, "col_x": 16},
            "writer": {"name": "test", "version": "0", "build": "0"}
        }"#;
        let bytes = build_file_with_meta(meta_json);

        for depth in &[ValidateDepth::Structural, ValidateDepth::Deep] {
            let report = {
                let dir = test_dir("huge-dims");
                let path = write_corrupt(&dir, "huge-dims.rws", &bytes);
                let r = validate_hour_file(&path, *depth).unwrap();
                let _ = fs::remove_dir_all(&dir);
                r
            };
            assert!(
                !report.is_ok(),
                "huge nx/ny must produce errors ({depth:?}): {:?}",
                report.errors
            );
            let joined = report.errors.join(" ");
            assert!(
                joined.contains("nx")
                    || joined.contains("ny")
                    || joined.contains("bound")
                    || joined.contains("sanity"),
                "error must mention nx/ny/bound ({depth:?}): {joined}"
            );
        }
    }

    /// Hostile chunking values: col_y = col_x = 2_000_000_000. With range-valid
    /// tile coords (tile_y=tile_x=0) the ch/cw computation is bounded by the
    /// clamped .min(col_y) but the *multiply* 2*ch*cw*levels overflowed u32
    /// before the fix. Must produce Ok(report) with errors, not panic.
    ///
    /// Also covers the tile variant (tile_y=tile_x=3_000_000_000, surface2d)
    /// where 4*th*tw overflowed.
    #[test]
    fn huge_chunking_reports_error_not_panic() {
        // Variant B: pressure3d with hostile col_y/col_x = 2_000_000_000.
        let meta_json_b = r#"{
            "schema": "rw-store.hour.v1",
            "model": "hrrr",
            "run": "2026-06-10T00:00:00Z",
            "forecast_hour": 3,
            "nx": 100,
            "ny": 100,
            "grid_hash": "aaaa",
            "variables": [
                {
                    "id": 0,
                    "name": "wind_iso",
                    "units": "m/s",
                    "kind": "pressure3d",
                    "codec": "zstd1_affine_i16",
                    "levels_hpa": [1000, 500, 200],
                    "selector": {}
                }
            ],
            "chunking": {"tile_y": 256, "tile_x": 256, "col_y": 2000000000, "col_x": 2000000000},
            "writer": {"name": "test", "version": "0", "build": "0"}
        }"#;

        // Variant C: surface2d with hostile tile_y/tile_x = 3_000_000_000.
        let meta_json_c = r#"{
            "schema": "rw-store.hour.v1",
            "model": "hrrr",
            "run": "2026-06-10T00:00:00Z",
            "forecast_hour": 3,
            "nx": 100,
            "ny": 100,
            "grid_hash": "aaaa",
            "variables": [
                {
                    "id": 0,
                    "name": "temp_2m",
                    "units": "K",
                    "kind": "surface2d",
                    "codec": "zstd1_f32",
                    "levels_hpa": [],
                    "selector": {}
                }
            ],
            "chunking": {"tile_y": 3000000000, "tile_x": 3000000000, "col_y": 16, "col_x": 16},
            "writer": {"name": "test", "version": "0", "build": "0"}
        }"#;

        for (label, meta_json) in &[("variant_B", meta_json_b), ("variant_C", meta_json_c)] {
            let bytes = build_file_with_meta(meta_json);
            for depth in &[ValidateDepth::Structural, ValidateDepth::Deep] {
                let report = {
                    let dir = test_dir("huge-chunking");
                    let path = write_corrupt(&dir, "huge-chunking.rws", &bytes);
                    let r = validate_hour_file(&path, *depth).unwrap();
                    let _ = fs::remove_dir_all(&dir);
                    r
                };
                assert!(
                    !report.is_ok(),
                    "{label}/{depth:?}: huge chunking must produce errors: {:?}",
                    report.errors
                );
                let joined = report.errors.join(" ");
                assert!(
                    joined.contains("col_y")
                        || joined.contains("col_x")
                        || joined.contains("tile_y")
                        || joined.contains("tile_x")
                        || joined.contains("bound")
                        || joined.contains("sanity"),
                    "{label}/{depth:?}: error must mention chunking field or bound: {joined}"
                );
            }
        }
    }

    /// Patch a real dense chunk's raw_len to u32::MAX. Check 8 must catch the
    /// mismatch (raw_len != expected_geometry). The deep phase must skip the
    /// record (because it is in raw_len_bad_records) and complete quickly
    /// without allocating ~4 GiB.  The test passing without OOM/timeout is the
    /// main assertion.
    #[test]
    fn oversized_raw_len_does_not_allocate() {
        let dir = test_dir("oversized-raw-len");
        let mut bytes = load_valid_bytes(&dir);
        let header = RwsHeader::parse(&bytes).unwrap();

        // Find first dense chunk record (len > 0, not EMPTY/CONSTANT).
        let mut patched = false;
        for i in 0..header.index_count as usize {
            let start = header.index_offset as usize + i * INDEX_RECORD_LEN;
            let rec = ChunkRecord::unpack(&bytes[start..start + INDEX_RECORD_LEN]).unwrap();
            let is_empty = rec.flags & FLAG_EMPTY != 0;
            let is_constant_no_missing =
                rec.flags & FLAG_CONSTANT != 0 && rec.flags & FLAG_HAS_MISSING == 0;
            if rec.len > 0 && !is_empty && !is_constant_no_missing {
                // raw_len is at bytes 24..28 within the 64-byte index record.
                bytes[start + 24..start + 28].copy_from_slice(&u32::MAX.to_le_bytes());
                patched = true;
                break;
            }
        }
        assert!(patched, "test requires at least one dense non-empty chunk");

        let path = write_corrupt(&dir, "oversized-raw-len.rws", &bytes);

        // Structural must report the raw_len mismatch.
        let structural = validate_hour_file(&path, ValidateDepth::Structural).unwrap();
        assert!(
            !structural.is_ok(),
            "oversized raw_len must fail structural: {:?}",
            structural.errors
        );
        assert!(
            structural.errors.iter().any(|e| e.contains("raw_len")),
            "structural error must mention raw_len: {:?}",
            structural.errors
        );

        // Deep must also complete (not OOM/hang) and must NOT produce a
        // "decompress failed" error for this record — it was skipped because it
        // is in raw_len_bad_records.  It may produce other errors for the same
        // record (e.g. the raw_len mismatch is already in structural).
        let deep = validate_hour_file(&path, ValidateDepth::Deep).unwrap();
        assert!(
            !deep.is_ok(),
            "oversized raw_len must fail deep: {:?}",
            deep.errors
        );
        // Confirm the record was skipped in deep (no decompress-error message).
        let joined = deep.errors.join(" ");
        assert!(
            !joined.contains("zstd decompress failed"),
            "deep must skip the oversized-raw_len record, not attempt decompression: {joined}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// Patch a dense chunk's raw_len to (MAX_DEEP_CHUNK_RAW_LEN + 1) — a value
    /// that Check 8 will also flag as mismatching geometry, but crucially the
    /// deep phase must emit the cap-exceeded error and never attempt to
    /// decompress (no "zstd decompress failed" in the error list).
    ///
    /// This directly tests the absolute-backstop gate added after Check 8's
    /// raw_len_bad guard: even if a crafted file somehow passed Check 8, the
    /// cap prevents a multi-GiB allocation.
    #[test]
    fn deep_caps_chunk_decode_size() {
        let dir = test_dir("deep-cap");
        let mut bytes = load_valid_bytes(&dir);
        let header = RwsHeader::parse(&bytes).unwrap();

        // Find the first dense, non-EMPTY, non-CONSTANT chunk.
        let mut patched_idx: Option<usize> = None;
        for i in 0..header.index_count as usize {
            let start = header.index_offset as usize + i * INDEX_RECORD_LEN;
            let rec = ChunkRecord::unpack(&bytes[start..start + INDEX_RECORD_LEN]).unwrap();
            let is_empty = rec.flags & FLAG_EMPTY != 0;
            let is_constant_no_missing =
                rec.flags & FLAG_CONSTANT != 0 && rec.flags & FLAG_HAS_MISSING == 0;
            if rec.len > 0 && !is_empty && !is_constant_no_missing {
                // Patch raw_len (bytes 24..28 in the 64-byte record) to cap + 1.
                let over_cap = MAX_DEEP_CHUNK_RAW_LEN + 1;
                bytes[start + 24..start + 28].copy_from_slice(&over_cap.to_le_bytes());
                patched_idx = Some(i);
                break;
            }
        }
        assert!(
            patched_idx.is_some(),
            "test requires at least one dense non-empty chunk"
        );

        let path = write_corrupt(&dir, "deep-cap.rws", &bytes);

        // Structural must flag the raw_len mismatch (Check 8).
        let structural = validate_hour_file(&path, ValidateDepth::Structural).unwrap();
        assert!(
            !structural.is_ok(),
            "patched raw_len must fail structural: {:?}",
            structural.errors
        );
        assert!(
            structural.errors.iter().any(|e| e.contains("raw_len")),
            "structural error must mention raw_len: {:?}",
            structural.errors
        );

        // Deep must emit the cap error and must NOT attempt decompression.
        let deep = validate_hour_file(&path, ValidateDepth::Deep).unwrap();
        assert!(
            !deep.is_ok(),
            "patched raw_len must fail deep: {:?}",
            deep.errors
        );
        let joined = deep.errors.join(" ");
        assert!(
            joined.contains("exceeds deep-validation cap"),
            "deep error must mention cap: {joined}"
        );
        assert!(
            !joined.contains("zstd decompress failed"),
            "deep must not attempt decompression for over-cap chunk: {joined}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------------
    // Negative tests for deep stat checks (Checks 12/13/14)
    // -----------------------------------------------------------------------

    /// Build a valid hour with NO NaN values in the 2D field (all finite).
    /// Used by tests that need a TILE2D record without HAS_MISSING set.
    fn write_valid_hour_no_nan(path: &std::path::Path) {
        let values_2d: Vec<f32> = (0..NY)
            .flat_map(|y| (0..NX).map(move |x| 0.1 * x as f32 + 0.2 * y as f32 + 273.15))
            .collect();

        let levels: [u16; 2] = [1000, 500];
        let planes: Vec<Vec<f32>> = levels
            .iter()
            .map(|&level| {
                (0..NY)
                    .flat_map(|_y| {
                        (0..NX).map(move |x| 100.0 + 0.5 * x as f32 + 0.01 * level as f32)
                    })
                    .collect()
            })
            .collect();
        let plane_refs: Vec<&[f32]> = planes.iter().map(|p| p.as_slice()).collect();

        let mut writer = HourWriter::new(
            "hrrr",
            "2026-06-10T00:00:00Z",
            3,
            NX,
            NY,
            "test-grid-hash",
            "validate-test",
        );
        writer
            .add_surface2d(
                "temp_2m",
                "K",
                serde_json::json!({"test": true}),
                &values_2d,
            )
            .unwrap();
        writer
            .add_pressure3d(
                "wind_iso",
                "m/s",
                serde_json::json!({"test3d": true}),
                &levels,
                &plane_refs,
            )
            .unwrap();
        writer.finish(path).unwrap();
    }

    /// Helper: find the first TILE2D index record (kind == KIND_TILE2D) with
    /// len > 0 (dense, not EMPTY or CONSTANT-without-missing). Returns the
    /// byte offset of the 64-byte record within the file.
    fn find_dense_tile2d_record_offset(bytes: &[u8]) -> usize {
        let header = RwsHeader::parse(bytes).unwrap();
        for i in 0..header.index_count as usize {
            let start = header.index_offset as usize + i * INDEX_RECORD_LEN;
            let rec = ChunkRecord::unpack(&bytes[start..start + INDEX_RECORD_LEN]).unwrap();
            let is_empty = rec.flags & FLAG_EMPTY != 0;
            let is_constant_no_missing =
                rec.flags & FLAG_CONSTANT != 0 && rec.flags & FLAG_HAS_MISSING == 0;
            if rec.kind == KIND_TILE2D && rec.len > 0 && !is_empty && !is_constant_no_missing {
                return start;
            }
        }
        panic!("no dense TILE2D record found in test file");
    }

    /// Patch a dense TILE2D record's valid_count (bytes [44..48]) to value+1.
    /// Structural must not flag this (valid_count is not structurally checked).
    /// Deep must report an error mentioning "valid_count".
    #[test]
    fn deep_catches_wrong_valid_count() {
        let dir = test_dir("deep-valid-count");
        let base_path = dir.join("base.rws");
        write_valid_hour(&base_path);
        let mut bytes = fs::read(&base_path).unwrap();

        let rec_start = find_dense_tile2d_record_offset(&bytes);
        // valid_count is at bytes [44..48] within the 64-byte record.
        let current = u32::from_le_bytes(bytes[rec_start + 44..rec_start + 48].try_into().unwrap());
        let patched = current.wrapping_add(1);
        bytes[rec_start + 44..rec_start + 48].copy_from_slice(&patched.to_le_bytes());

        let path = write_corrupt(&dir, "wrong-valid-count.rws", &bytes);

        // Structural: valid_count is not checked structurally — must pass.
        let structural = validate_hour_file(&path, ValidateDepth::Structural).unwrap();
        assert!(
            structural.is_ok(),
            "wrong valid_count must not affect structural: {:?}",
            structural.errors
        );

        // Deep: must catch the mismatch.
        let deep = validate_hour_file(&path, ValidateDepth::Deep).unwrap();
        assert!(
            !deep.is_ok(),
            "wrong valid_count must fail deep: {:?}",
            deep.errors
        );
        assert!(
            deep.errors.iter().any(|e| e.contains("valid_count")),
            "deep error must mention valid_count: {:?}",
            deep.errors
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// Patch a dense TILE2D record's min (bytes [36..40]) to a different finite value.
    /// Structural must pass. Deep must report an error mentioning "min".
    #[test]
    fn deep_catches_wrong_min_max() {
        let dir = test_dir("deep-min-max");
        let base_path = dir.join("base.rws");
        write_valid_hour(&base_path);
        let mut bytes = fs::read(&base_path).unwrap();

        let rec_start = find_dense_tile2d_record_offset(&bytes);
        // min is at bytes [36..40] within the 64-byte record.
        let current_min =
            f32::from_le_bytes(bytes[rec_start + 36..rec_start + 40].try_into().unwrap());
        // Replace with a clearly different finite value.
        let patched_min = current_min + 9999.0;
        bytes[rec_start + 36..rec_start + 40].copy_from_slice(&patched_min.to_le_bytes());

        let path = write_corrupt(&dir, "wrong-min.rws", &bytes);

        // Structural: min is not checked structurally — must pass.
        let structural = validate_hour_file(&path, ValidateDepth::Structural).unwrap();
        assert!(
            structural.is_ok(),
            "wrong min must not affect structural: {:?}",
            structural.errors
        );

        // Deep: must catch the mismatch.
        let deep = validate_hour_file(&path, ValidateDepth::Deep).unwrap();
        assert!(!deep.is_ok(), "wrong min must fail deep: {:?}", deep.errors);
        assert!(
            deep.errors
                .iter()
                .any(|e| e.contains("min") || e.contains("max")),
            "deep error must mention min or max: {:?}",
            deep.errors
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// Patch a dense TILE2D record (no NaN values) to set the HAS_MISSING flag.
    /// Structural must pass (len > 0, not EMPTY, not CONSTANT-without-missing,
    /// so the len==0-iff rules are unaffected). Deep must error that HAS_MISSING
    /// is set but no non-finite values were decoded.
    #[test]
    fn deep_catches_wrong_missing_flag() {
        let dir = test_dir("deep-missing-flag");
        let base_path = dir.join("base.rws");
        // Use the no-NaN variant so the target TILE2D record has HAS_MISSING=0.
        write_valid_hour_no_nan(&base_path);
        let mut bytes = fs::read(&base_path).unwrap();

        let rec_start = find_dense_tile2d_record_offset(&bytes);
        // Verify the record truly has no HAS_MISSING flag before patching.
        assert_eq!(
            bytes[rec_start + 3] & FLAG_HAS_MISSING,
            0,
            "pre-condition: no-NaN file must not have HAS_MISSING set"
        );
        // Set HAS_MISSING bit in the flags byte (offset [3]).
        bytes[rec_start + 3] |= FLAG_HAS_MISSING;

        let path = write_corrupt(&dir, "wrong-missing-flag.rws", &bytes);

        // Structural: setting HAS_MISSING on a dense non-CONSTANT record keeps
        // len > 0 legal (expect_zero_len = is_empty || is_constant_no_missing,
        // neither of which is true here). Must pass.
        let structural = validate_hour_file(&path, ValidateDepth::Structural).unwrap();
        assert!(
            structural.is_ok(),
            "setting HAS_MISSING on dense record must not affect structural: {:?}",
            structural.errors
        );

        // Deep: must catch that HAS_MISSING is set but decoded data has no NaN.
        let deep = validate_hour_file(&path, ValidateDepth::Deep).unwrap();
        assert!(
            !deep.is_ok(),
            "wrong HAS_MISSING flag must fail deep: {:?}",
            deep.errors
        );
        assert!(
            deep.errors.iter().any(|e| e.contains("HAS_MISSING")
                || e.contains("has_missing")
                || e.contains("has_non_finite")),
            "deep error must mention HAS_MISSING or related field: {:?}",
            deep.errors
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
