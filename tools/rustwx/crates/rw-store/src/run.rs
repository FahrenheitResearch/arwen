//! Run-level manifest (`run.json`): which hour files exist for a model run,
//! keyed by whole forecast hour in v1 or ordinal storage slot in exact-time
//! v2, plus the grid identity they were written against.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};

use crate::atomic::atomic_write_bytes;
use crate::error::{RwResult, RwStoreError};
use crate::format::{RwsExactTime, RwsHourMeta, RwsWriterInfo, SCHEMA_HOUR, SCHEMA_HOUR_V2};

/// Schema identifier embedded in run manifests.
pub const SCHEMA_RUN: &str = "rw-store.run.v1";
/// Exact-time manifest schema. Map keys are ordinal storage slots, not an
/// assertion that the corresponding lead is a whole number of hours.
pub const SCHEMA_RUN_V2: &str = "rw-store.run.v2";

/// Maximum accepted `run.json` size. Manifests are metadata, not payloads;
/// bounding the read prevents a corrupt or hostile store from forcing an
/// unbounded allocation before JSON parsing begins.
pub const MAX_RUN_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;

/// The grid reader applies the same ceiling to each decompressed f32
/// coordinate array. Reject impossible manifest geometry before a consumer
/// uses it to size work or compares it with an hour/grid file.
const MAX_GRID_COORD_RAW_BYTES: u64 = 1 << 31;

/// Require a store identity or persisted child filename to be exactly one
/// relative path component. The explicit backslash check keeps serialized
/// manifests safe when moved between Unix and Windows, where path separator
/// rules differ.
pub fn validate_store_component(label: &str, value: &str) -> RwResult<()> {
    let mut components = Path::new(value).components();
    let first_is_exact_normal = matches!(
        components.next(),
        Some(Component::Normal(component)) if component == OsStr::new(value)
    );
    if !first_is_exact_normal
        || components.next().is_some()
        || value.contains('\\')
        || value.contains('\0')
        || value.contains(':')
        || value
            .chars()
            .any(|ch| matches!(ch, '<' | '>' | '"' | '|' | '?' | '*'))
        || value.ends_with('.')
        || value.ends_with(' ')
        || value.chars().any(|ch| ch.is_control())
        || is_windows_device_name(value)
    {
        return Err(RwStoreError::Meta(format!(
            "{label} '{value}' must be exactly one normal path component"
        )));
    }
    Ok(())
}

fn is_windows_device_name(value: &str) -> bool {
    let stem = value
        .split('.')
        .next()
        .unwrap_or_default()
        .trim_end_matches(|ch| ch == ' ' || ch == '.')
        .to_ascii_uppercase();
    matches!(
        stem.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$" | "CLOCK$"
    ) || stem
        .strip_prefix("COM")
        .or_else(|| stem.strip_prefix("LPT"))
        .is_some_and(|suffix| suffix.len() == 1 && matches!(suffix.as_bytes()[0], b'1'..=b'9'))
}

/// One registered v1 forecast hour or v2 ordinal storage slot: the hour file,
/// optional exact physical timing, and write provenance.
/// `written_unix` is supplied by the caller (the library never reads the
/// clock), so tests and replays stay deterministic.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RwsHourEntry {
    pub file: String,
    /// Present together only in exact-time v2; absent in v1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lead_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_unix: Option<i64>,
    pub written_unix: u64,
    pub encode_ms: u64,
    pub variables: Vec<String>,
}

impl RwsHourEntry {
    pub fn exact_time(&self) -> Option<RwsExactTime> {
        Some(RwsExactTime {
            lead_seconds: self.lead_seconds?,
            valid_unix: self.valid_unix?,
        })
    }
}

/// Run manifest: identity of the run (model, run, grid) and its ordered map.
/// Keys are whole forecast hours in v1 and ordinal slots in v2. A BTreeMap
/// keeps JSON stable and gives v2 timing validation an unambiguous order.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RwsRunManifest {
    pub schema: String,
    pub model: String,
    pub run: String,
    pub grid_hash: String,
    pub nx: usize,
    pub ny: usize,
    pub hours: BTreeMap<u16, RwsHourEntry>,
    pub writer: RwsWriterInfo,
}

impl RwsRunManifest {
    /// Read and parse one persisted manifest without trusting its contents.
    /// The read is capped even if the file grows after its metadata is
    /// inspected. Diagnostic tools may use this to retain a parsed value and
    /// report several problems; normal consumers should use [`Self::load`].
    pub fn load_bounded(path: &Path) -> RwResult<Self> {
        let initial_meta = path.metadata()?;
        if !initial_meta.is_file() {
            return Err(RwStoreError::Meta(format!(
                "run manifest {} is not a regular file",
                path.display()
            )));
        }
        let file = File::open(path)?;
        let open_meta = file.metadata()?;
        if !open_meta.is_file() {
            return Err(RwStoreError::Meta(format!(
                "run manifest {} changed to a non-file before it was opened",
                path.display()
            )));
        }
        let file_len = open_meta.len();
        if file_len > MAX_RUN_MANIFEST_BYTES {
            return Err(RwStoreError::Meta(format!(
                "run manifest {} is {file_len} bytes; limit is {MAX_RUN_MANIFEST_BYTES} bytes",
                path.display()
            )));
        }

        let mut bytes = Vec::with_capacity(file_len as usize);
        file.take(MAX_RUN_MANIFEST_BYTES + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_RUN_MANIFEST_BYTES {
            return Err(RwStoreError::Meta(format!(
                "run manifest {} grew beyond the {MAX_RUN_MANIFEST_BYTES}-byte limit while reading",
                path.display()
            )));
        }

        let manifest: Self = serde_json::from_slice(&bytes)
            .map_err(|err| RwStoreError::Meta(format!("run manifest JSON: {err}")))?;
        Ok(manifest)
    }

    /// Read, parse, and fully validate one persisted manifest.
    pub fn load(path: &Path) -> RwResult<Self> {
        let manifest = Self::load_bounded(path)?;
        manifest.validate_contents()?;
        Ok(manifest)
    }

    /// Load a manifest and require its persisted model/run identity to match
    /// the directory identity selected by the caller. Expected components are
    /// checked before the file is opened, so `..` cannot redirect the read.
    pub fn load_for_run(path: &Path, model: &str, run: &str) -> RwResult<Self> {
        validate_store_component("model", model)?;
        validate_store_component("run", run)?;
        let manifest = Self::load(path)?;
        manifest.validate_identity(model, run)?;
        Ok(manifest)
    }

    /// Validate the manifest's model/run identity against its containing
    /// directory (or another trusted identity source).
    pub fn validate_identity(&self, model: &str, run: &str) -> RwResult<()> {
        validate_store_component("model", model)?;
        validate_store_component("run", run)?;
        if self.model != model || self.run != run {
            return Err(RwStoreError::Meta(format!(
                "manifest identity is model='{}' run='{}'; expected model='{model}' run='{run}'",
                self.model, self.run
            )));
        }
        Ok(())
    }

    /// Validate the grid identity and geometry recorded by another trusted
    /// store object (`grid.rwg` or an opened hour file).
    pub fn validate_grid(&self, grid_hash: &str, nx: usize, ny: usize) -> RwResult<()> {
        if self.grid_hash != grid_hash || self.nx != nx || self.ny != ny {
            return Err(RwStoreError::Meta(format!(
                "manifest grid is {}x{} hash='{}'; expected {nx}x{ny} hash='{grid_hash}'",
                self.nx, self.ny, self.grid_hash
            )));
        }
        Ok(())
    }

    /// Whether map keys are ordinal storage slots with exact physical timing.
    pub fn is_exact_time_axis(&self) -> bool {
        self.schema == SCHEMA_RUN_V2
    }

    /// Exact times in ordinal storage-slot order. A manifest returned by
    /// [`Self::load`] is guaranteed to yield one item for every v2 entry and
    /// no items for v1.
    pub fn exact_times(&self) -> impl Iterator<Item = (u16, RwsExactTime)> + '_ {
        self.hours
            .iter()
            .filter_map(|(&slot, entry)| entry.exact_time().map(|time| (slot, time)))
    }

    /// Validate one registered hour file's metadata against this manifest,
    /// including the ordinal slot and exact-time pair. This is the shared seam
    /// for store browsers, import recovery, and diagnostic validation.
    pub fn validate_hour_meta(
        &self,
        storage_slot: u16,
        meta: &RwsHourMeta,
    ) -> RwResult<&RwsHourEntry> {
        let entry = self.hours.get(&storage_slot).ok_or_else(|| {
            RwStoreError::Meta(format!(
                "storage slot {storage_slot} is absent from the run manifest"
            ))
        })?;
        let expected_hour_schema = match self.schema.as_str() {
            SCHEMA_RUN => SCHEMA_HOUR,
            SCHEMA_RUN_V2 => SCHEMA_HOUR_V2,
            other => {
                return Err(RwStoreError::Meta(format!(
                    "unexpected manifest schema '{other}'"
                )));
            }
        };
        if meta.schema != expected_hour_schema {
            return Err(RwStoreError::Meta(format!(
                "storage slot {storage_slot} hour schema '{}' does not match manifest schema '{}' (expected '{expected_hour_schema}')",
                meta.schema, self.schema
            )));
        }
        let meta_time = meta.validate_time_schema().map_err(RwStoreError::Meta)?;
        let entry_time = match (entry.lead_seconds, entry.valid_unix) {
            (Some(lead_seconds), Some(valid_unix)) => Some(RwsExactTime {
                lead_seconds,
                valid_unix,
            }),
            (None, None) => None,
            _ => {
                return Err(RwStoreError::Meta(format!(
                    "storage slot {storage_slot} manifest entry must contain both lead_seconds and valid_unix or neither"
                )));
            }
        };
        if self.is_exact_time_axis() != entry_time.is_some() {
            return Err(RwStoreError::Meta(format!(
                "storage slot {storage_slot} timing does not satisfy manifest schema '{}'",
                self.schema
            )));
        }
        if meta.forecast_hour != storage_slot {
            return Err(RwStoreError::Meta(format!(
                "storage slot {storage_slot} contains hour metadata slot {}",
                meta.forecast_hour
            )));
        }
        if meta.model != self.model || meta.run != self.run {
            return Err(RwStoreError::Meta(format!(
                "storage slot {storage_slot} hour identity model='{}' run='{}' does not match manifest model='{}' run='{}'",
                meta.model, meta.run, self.model, self.run
            )));
        }
        if meta.grid_hash != self.grid_hash || meta.nx != self.nx || meta.ny != self.ny {
            return Err(RwStoreError::Meta(format!(
                "storage slot {storage_slot} hour grid {}x{} hash='{}' does not match manifest {}x{} hash='{}'",
                meta.nx, meta.ny, meta.grid_hash, self.nx, self.ny, self.grid_hash
            )));
        }
        if meta_time != entry_time {
            return Err(RwStoreError::Meta(format!(
                "storage slot {storage_slot} hour exact time {:?} does not match manifest {:?}",
                meta_time, entry_time
            )));
        }
        Ok(entry)
    }

    /// Validate a parsed manifest's schema, safe identities, geometry, and
    /// registered hour filenames. Primarily public for diagnostic tooling;
    /// regular readers get this automatically through [`Self::load`].
    pub fn validate_contents(&self) -> RwResult<()> {
        if self.schema != SCHEMA_RUN && self.schema != SCHEMA_RUN_V2 {
            return Err(RwStoreError::Meta(format!(
                "unexpected schema '{}' (expected '{SCHEMA_RUN}' or '{SCHEMA_RUN_V2}')",
                self.schema
            )));
        }
        validate_store_component("manifest model", &self.model)?;
        validate_store_component("manifest run", &self.run)?;
        if self.grid_hash.trim().is_empty() {
            return Err(RwStoreError::Meta(
                "manifest grid_hash must not be empty".to_string(),
            ));
        }
        if self.nx == 0 || self.ny == 0 {
            return Err(RwStoreError::Meta(format!(
                "degenerate manifest grid {}x{} (nx and ny must be nonzero)",
                self.nx, self.ny
            )));
        }
        let raw_bytes = self
            .nx
            .checked_mul(self.ny)
            .and_then(|cells| u64::try_from(cells).ok())
            .and_then(|cells| cells.checked_mul(4))
            .filter(|&bytes| bytes <= MAX_GRID_COORD_RAW_BYTES)
            .ok_or_else(|| {
                RwStoreError::Meta(format!(
                    "manifest grid {}x{} exceeds the supported coordinate array size",
                    self.nx, self.ny
                ))
            })?;
        debug_assert!(raw_bytes > 0);

        let mut files = BTreeSet::new();
        let exact_axis = self.schema == SCHEMA_RUN_V2;
        let mut previous: Option<(u64, i64)> = None;
        let mut origin_unix = None;
        for (&slot, entry) in &self.hours {
            validate_store_component(&format!("manifest slot {slot} file"), &entry.file)?;
            if !files.insert(entry.file.as_str()) {
                return Err(RwStoreError::Meta(format!(
                    "manifest slot {slot} reuses file '{}' already registered to another slot",
                    entry.file
                )));
            }
            let exact_time = match (entry.lead_seconds, entry.valid_unix) {
                (Some(lead_seconds), Some(valid_unix)) => Some(RwsExactTime {
                    lead_seconds,
                    valid_unix,
                }),
                (None, None) => None,
                _ => {
                    return Err(RwStoreError::Meta(format!(
                        "manifest slot {slot} must contain both lead_seconds and valid_unix or neither"
                    )));
                }
            };
            if !exact_axis {
                if exact_time.is_some() {
                    return Err(RwStoreError::Meta(format!(
                        "v1 manifest slot {slot} must not contain exact-time metadata"
                    )));
                }
                continue;
            }
            let exact_time = exact_time.ok_or_else(|| {
                RwStoreError::Meta(format!(
                    "v2 manifest slot {slot} requires lead_seconds and valid_unix"
                ))
            })?;
            let expected_file = format!("f{slot:03}.rws");
            if entry.file != expected_file {
                return Err(RwStoreError::Meta(format!(
                    "v2 manifest slot {slot} file '{}' must use ordinal filename '{expected_file}'",
                    entry.file
                )));
            }
            let this_origin = exact_time.origin_unix().ok_or_else(|| {
                RwStoreError::Meta(format!(
                    "v2 manifest slot {slot} exact time cannot represent valid_unix - lead_seconds"
                ))
            })?;
            if let Some(expected_origin) = origin_unix {
                if this_origin != expected_origin {
                    return Err(RwStoreError::Meta(format!(
                        "v2 manifest slot {slot} implies origin {this_origin}, expected {expected_origin}"
                    )));
                }
            } else {
                origin_unix = Some(this_origin);
            }
            if let Some((previous_lead, previous_valid)) = previous {
                if exact_time.lead_seconds <= previous_lead
                    || exact_time.valid_unix <= previous_valid
                {
                    return Err(RwStoreError::Meta(format!(
                        "v2 manifest exact times must increase strictly by storage slot; slot {slot} has lead_seconds={} valid_unix={} after lead_seconds={previous_lead} valid_unix={previous_valid}",
                        exact_time.lead_seconds, exact_time.valid_unix
                    )));
                }
            }
            previous = Some((exact_time.lead_seconds, exact_time.valid_unix));
        }
        Ok(())
    }

    /// Load the manifest at `path`, or create a fresh empty one if the file
    /// does not exist. An existing manifest must match `model`, `run`, grid
    /// identity, and geometry exactly ([`RwStoreError::Meta`] otherwise).
    pub fn load_or_new(
        path: &Path,
        model: &str,
        run: &str,
        grid_hash: &str,
        nx: usize,
        ny: usize,
        writer: RwsWriterInfo,
    ) -> RwResult<Self> {
        Self::load_or_new_schema(path, model, run, grid_hash, nx, ny, writer, SCHEMA_RUN)
    }

    /// Exact-time counterpart to [`Self::load_or_new`]. Existing v1 runs are
    /// rejected rather than silently mixing whole-hour keys with ordinal
    /// storage slots.
    pub fn load_or_new_exact(
        path: &Path,
        model: &str,
        run: &str,
        grid_hash: &str,
        nx: usize,
        ny: usize,
        writer: RwsWriterInfo,
    ) -> RwResult<Self> {
        Self::load_or_new_schema(path, model, run, grid_hash, nx, ny, writer, SCHEMA_RUN_V2)
    }

    fn load_or_new_schema(
        path: &Path,
        model: &str,
        run: &str,
        grid_hash: &str,
        nx: usize,
        ny: usize,
        writer: RwsWriterInfo,
        schema: &str,
    ) -> RwResult<Self> {
        if path.exists() {
            let manifest = Self::load_for_run(path, model, run)?;
            manifest.validate_grid(grid_hash, nx, ny)?;
            if manifest.schema != schema {
                return Err(RwStoreError::Meta(format!(
                    "existing manifest schema '{}' cannot be opened by writer for '{schema}'",
                    manifest.schema
                )));
            }
            return Ok(manifest);
        }
        let manifest = Self {
            schema: schema.to_string(),
            model: model.to_string(),
            run: run.to_string(),
            grid_hash: grid_hash.to_string(),
            nx,
            ny,
            hours: BTreeMap::new(),
            writer,
        };
        manifest.validate_contents()?;
        Ok(manifest)
    }

    /// Insert or overwrite the entry for `hour`.
    pub fn register_hour(&mut self, hour: u16, entry: RwsHourEntry) {
        self.hours.insert(hour, entry);
    }

    /// Atomically write the manifest as pretty JSON.
    pub fn save(&self, path: &Path) -> RwResult<()> {
        self.validate_contents()?;
        let mut bytes =
            serde_json::to_vec_pretty(self).map_err(|err| RwStoreError::Meta(err.to_string()))?;
        bytes.push(b'\n');
        if bytes.len() as u64 > MAX_RUN_MANIFEST_BYTES {
            return Err(RwStoreError::Meta(format!(
                "serialized run manifest is {} bytes; limit is {MAX_RUN_MANIFEST_BYTES} bytes",
                bytes.len()
            )));
        }
        atomic_write_bytes(path, &bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RwStoreError;
    use crate::format::{RwsChunking, RwsHourMeta, RwsWriterInfo, SCHEMA_HOUR_V2};
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;

    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("rw-store-run-{}-{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn writer_info() -> RwsWriterInfo {
        RwsWriterInfo {
            name: "rw-store".to_string(),
            version: "0.1.0".to_string(),
            build: "test-build".to_string(),
        }
    }

    fn entry(file: &str, written_unix: u64, encode_ms: u64, variables: &[&str]) -> RwsHourEntry {
        RwsHourEntry {
            file: file.to_string(),
            lead_seconds: None,
            valid_unix: None,
            written_unix,
            encode_ms,
            variables: variables.iter().map(|v| v.to_string()).collect(),
        }
    }

    fn exact_entry(slot: u16, lead_seconds: u64, valid_unix: i64) -> RwsHourEntry {
        RwsHourEntry {
            file: format!("f{slot:03}.rws"),
            lead_seconds: Some(lead_seconds),
            valid_unix: Some(valid_unix),
            written_unix: 1_770_000_000,
            encode_ms: 10,
            variables: vec!["temp_2m".to_string()],
        }
    }

    fn exact_manifest() -> RwsRunManifest {
        RwsRunManifest {
            schema: SCHEMA_RUN_V2.to_string(),
            model: "wrf".to_string(),
            run: "research".to_string(),
            grid_hash: "gridhash-test".to_string(),
            nx: 2,
            ny: 2,
            hours: BTreeMap::from([
                (0, exact_entry(0, 0, 1_700_000_000)),
                (1, exact_entry(1, 1_800, 1_700_001_800)),
            ]),
            writer: writer_info(),
        }
    }

    #[test]
    fn run_manifest_round_trips_and_registers_hours() {
        let dir = test_dir("round-trip");
        let path = dir.join("run.json");

        let mut manifest = RwsRunManifest::load_or_new(
            &path,
            "hrrr",
            "20260609_12z",
            "gridhash-test",
            600,
            500,
            writer_info(),
        )
        .unwrap();
        assert_eq!(manifest.schema, "rw-store.run.v1");
        assert_eq!(manifest.model, "hrrr");
        assert_eq!(manifest.run, "20260609_12z");
        assert_eq!(manifest.grid_hash, "gridhash-test");
        assert_eq!((manifest.nx, manifest.ny), (600, 500));
        assert!(manifest.hours.is_empty(), "new manifest starts empty");

        manifest.register_hour(
            0,
            entry("f000.rws", 1_770_000_000, 850, &["temp_2m", "dewpoint_2m"]),
        );
        manifest.register_hour(6, entry("f006.rws", 1_770_000_600, 912, &["temp_2m"]));
        manifest.save(&path).unwrap();

        let loaded = RwsRunManifest::load_or_new(
            &path,
            "hrrr",
            "20260609_12z",
            "gridhash-test",
            600,
            500,
            writer_info(),
        )
        .unwrap();
        assert_eq!(loaded, manifest, "save -> load must round-trip exactly");
        assert_eq!(loaded.hours.len(), 2);
        assert_eq!(loaded.hours[&0].file, "f000.rws");
        assert_eq!(loaded.hours[&0].written_unix, 1_770_000_000);
        assert_eq!(loaded.hours[&0].variables, vec!["temp_2m", "dewpoint_2m"]);
        assert_eq!(loaded.hours[&6].encode_ms, 912);
        let v1_json = serde_json::to_value(&loaded).unwrap();
        assert!(v1_json["hours"]["0"].get("lead_seconds").is_none());
        assert!(v1_json["hours"]["0"].get("valid_unix").is_none());

        // Re-registering an hour overwrites in place; the map does not grow.
        let mut manifest = loaded;
        manifest.register_hour(0, entry("f000-v2.rws", 1_770_001_000, 700, &["temp_2m"]));
        assert_eq!(manifest.hours.len(), 2, "overwrite must not add an entry");
        assert_eq!(manifest.hours[&0].file, "f000-v2.rws");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn exact_manifest_round_trips_and_exposes_ordered_times() {
        let dir = test_dir("exact-round-trip");
        let path = dir.join("run.json");
        let manifest = exact_manifest();
        manifest.save(&path).unwrap();

        let loaded = RwsRunManifest::load(&path).unwrap();
        assert!(loaded.is_exact_time_axis());
        assert_eq!(loaded, manifest);
        assert_eq!(
            loaded.exact_times().collect::<Vec<_>>(),
            vec![
                (
                    0,
                    RwsExactTime {
                        lead_seconds: 0,
                        valid_unix: 1_700_000_000,
                    },
                ),
                (
                    1,
                    RwsExactTime {
                        lead_seconds: 1_800,
                        valid_unix: 1_700_001_800,
                    },
                ),
            ]
        );
        let err = RwsRunManifest::load_or_new(
            &path,
            "wrf",
            "research",
            "gridhash-test",
            2,
            2,
            writer_info(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("cannot be opened"), "{err}");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn schema_time_contract_rejects_missing_mixed_and_inconsistent_axes() {
        let mut v1 = RwsRunManifest {
            schema: SCHEMA_RUN.to_string(),
            model: "wrf".to_string(),
            run: "research".to_string(),
            grid_hash: "gridhash-test".to_string(),
            nx: 2,
            ny: 2,
            hours: BTreeMap::from([(0, entry("f000.rws", 0, 0, &["temp_2m"]))]),
            writer: writer_info(),
        };
        v1.hours.get_mut(&0).unwrap().lead_seconds = Some(0);
        v1.hours.get_mut(&0).unwrap().valid_unix = Some(1_700_000_000);
        assert!(
            v1.validate_contents()
                .unwrap_err()
                .to_string()
                .contains("v1")
        );

        let mut missing = exact_manifest();
        missing.hours.get_mut(&1).unwrap().valid_unix = None;
        assert!(
            missing
                .validate_contents()
                .unwrap_err()
                .to_string()
                .contains("both")
        );

        let mut absent = exact_manifest();
        absent.hours.get_mut(&1).unwrap().lead_seconds = None;
        absent.hours.get_mut(&1).unwrap().valid_unix = None;
        assert!(
            absent
                .validate_contents()
                .unwrap_err()
                .to_string()
                .contains("requires")
        );

        let mut wrong_file = exact_manifest();
        wrong_file.hours.get_mut(&1).unwrap().file = "f900.rws".to_string();
        assert!(
            wrong_file
                .validate_contents()
                .unwrap_err()
                .to_string()
                .contains("ordinal filename")
        );

        let mut non_increasing = exact_manifest();
        non_increasing
            .hours
            .insert(1, exact_entry(1, 0, 1_700_000_000));
        assert!(
            non_increasing
                .validate_contents()
                .unwrap_err()
                .to_string()
                .contains("increase strictly")
        );

        let mut inconsistent_origin = exact_manifest();
        inconsistent_origin.hours.get_mut(&1).unwrap().valid_unix = Some(1_700_001_801);
        assert!(
            inconsistent_origin
                .validate_contents()
                .unwrap_err()
                .to_string()
                .contains("implies origin")
        );

        let mut unrepresentable = exact_manifest();
        unrepresentable.hours = BTreeMap::from([(0, exact_entry(0, u64::MAX, 0))]);
        assert!(
            unrepresentable
                .validate_contents()
                .unwrap_err()
                .to_string()
                .contains("cannot represent")
        );
    }

    #[test]
    fn manifest_hour_cross_check_requires_slot_and_exact_time_equality() {
        let manifest = exact_manifest();
        let mut meta = RwsHourMeta {
            schema: SCHEMA_HOUR_V2.to_string(),
            model: manifest.model.clone(),
            run: manifest.run.clone(),
            forecast_hour: 1,
            lead_seconds: Some(1_800),
            valid_unix: Some(1_700_001_800),
            nx: manifest.nx,
            ny: manifest.ny,
            grid_hash: manifest.grid_hash.clone(),
            variables: Vec::new(),
            chunking: RwsChunking {
                tile_y: 256,
                tile_x: 256,
                col_y: 16,
                col_x: 16,
            },
            writer: writer_info(),
        };
        assert_eq!(
            manifest.validate_hour_meta(1, &meta).unwrap().exact_time(),
            meta.exact_time()
        );

        meta.forecast_hour = 0;
        assert!(
            manifest
                .validate_hour_meta(1, &meta)
                .unwrap_err()
                .to_string()
                .contains("metadata slot")
        );
        meta.forecast_hour = 1;
        meta.valid_unix = Some(1_700_001_801);
        assert!(
            manifest
                .validate_hour_meta(1, &meta)
                .unwrap_err()
                .to_string()
                .contains("does not match")
        );
    }

    #[test]
    fn load_or_new_rejects_mismatched_existing_manifest() {
        let dir = test_dir("mismatch");
        let path = dir.join("run.json");
        let manifest = RwsRunManifest::load_or_new(
            &path,
            "hrrr",
            "20260609_12z",
            "gridhash-a",
            600,
            500,
            writer_info(),
        )
        .unwrap();
        manifest.save(&path).unwrap();

        let err = RwsRunManifest::load_or_new(
            &path,
            "hrrr",
            "20260609_12z",
            "gridhash-b",
            600,
            500,
            writer_info(),
        )
        .unwrap_err();
        assert!(
            matches!(err, RwStoreError::Meta(_)),
            "expected Meta error for grid_hash mismatch, got {err:?}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn bounded_loader_rejects_oversized_manifest_before_json_parse() {
        let dir = test_dir("oversized");
        let path = dir.join("run.json");
        let file = fs::File::create(&path).unwrap();
        file.set_len(MAX_RUN_MANIFEST_BYTES + 1).unwrap();

        let err = RwsRunManifest::load(&path).unwrap_err();
        assert!(matches!(err, RwStoreError::Meta(_)));
        assert!(err.to_string().contains("limit"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn loader_rejects_directory_identity_mismatch_and_unsafe_components() {
        let dir = test_dir("identity");
        let path = dir.join("run.json");
        let mut manifest = RwsRunManifest::load_or_new(
            &path,
            "hrrr",
            "20260609_12z",
            "gridhash-test",
            600,
            500,
            writer_info(),
        )
        .unwrap();
        manifest.register_hour(0, entry("f000.rws", 1_770_000_000, 850, &["temp_2m"]));
        manifest.save(&path).unwrap();

        let err = RwsRunManifest::load_for_run(&path, "hrrr", "20260609_00z").unwrap_err();
        assert!(err.to_string().contains("manifest identity"), "{err}");

        // Expected identities are checked before any attempted read.
        let err =
            RwsRunManifest::load_for_run(&dir.join("missing.json"), "../hrrr", "20260609_12z")
                .unwrap_err();
        assert!(matches!(err, RwStoreError::Meta(_)));

        for (unsafe_model, unsafe_run) in [("../hrrr", "20260609_12z"), ("hrrr", "../20260609_12z")]
        {
            let mut tampered = manifest.clone();
            tampered.model = unsafe_model.to_string();
            tampered.run = unsafe_run.to_string();
            fs::write(&path, serde_json::to_vec(&tampered).unwrap()).unwrap();
            let err = RwsRunManifest::load(&path).unwrap_err();
            assert!(
                matches!(err, RwStoreError::Meta(_)),
                "unsafe manifest identity must be rejected: {err:?}"
            );
        }

        for unsafe_file in [
            "",
            ".",
            "..",
            "../f000.rws",
            "nested/f000.rws",
            "nested\\f000.rws",
            "f000.rws:stream",
            "f000?.rws",
            "f000*.rws",
            "f000<old>.rws",
            "f000|old.rws",
            "f000\"old.rws",
            "f000.rws.",
            "NUL",
            "con.txt",
            "CONIN$",
            "conout$.log",
            "CLOCK$",
            "COM1",
        ] {
            manifest.hours.get_mut(&0).unwrap().file = unsafe_file.to_string();
            fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
            let err = RwsRunManifest::load_for_run(&path, "hrrr", "20260609_12z").unwrap_err();
            assert!(
                matches!(err, RwStoreError::Meta(_)),
                "'{unsafe_file}' must be rejected: {err:?}"
            );
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_or_new_rejects_existing_geometry_mismatch() {
        let dir = test_dir("geometry-mismatch");
        let path = dir.join("run.json");
        let manifest = RwsRunManifest::load_or_new(
            &path,
            "hrrr",
            "20260609_12z",
            "gridhash-test",
            600,
            500,
            writer_info(),
        )
        .unwrap();
        manifest.save(&path).unwrap();

        let err = RwsRunManifest::load_or_new(
            &path,
            "hrrr",
            "20260609_12z",
            "gridhash-test",
            601,
            500,
            writer_info(),
        )
        .unwrap_err();
        assert!(matches!(err, RwStoreError::Meta(_)));
        assert!(err.to_string().contains("600x500"), "{err}");

        let mut degenerate = manifest;
        degenerate.nx = 0;
        fs::write(&path, serde_json::to_vec(&degenerate).unwrap()).unwrap();
        let err = RwsRunManifest::load(&path).unwrap_err();
        assert!(
            err.to_string().contains("degenerate manifest grid"),
            "{err}"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
