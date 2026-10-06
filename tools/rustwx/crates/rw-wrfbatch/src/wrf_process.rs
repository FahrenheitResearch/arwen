//! Full WRF processing, updated from BowEcho v0.30.5's hardened model path.
//! Computes the model's 2D diagnostics (CAPE/severe/etc.) and isobaric
//! sounding volumes through `wrf-core::getvar`, then writes each WRF time as
//! one forecast-hour slot. Heavier than `local_import`, but produces the full
//! model field set.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{SystemTime, UNIX_EPOCH};

use rustwx_core::{
    CanonicalField, FieldSelector, GridProjection, GridShape, LatLonGrid, SelectedField2D,
};
use rw_store::{
    DerivedFieldInput, RwsExactTime, WrittenHour, write_hour_from_grid_with_derived,
    write_hour_from_grid_with_derived_exact,
};
use serde::{Deserialize, Serialize};

use crate::wrf_volumes::{
    IsoVolume, SurfaceFallback, build_iso_volumes, preflight_iso_volume_shape,
};
use wrf_core::variables::{VARS, VarDim};
use wrf_core::{ComputeOpts, VarOutput, WrfFile, getvar};

#[derive(Debug)]
pub struct WrfProcessTask {
    pub label: String,
    pub rx: Receiver<WrfProcessMessage>,
}

/// Stable destination for one independently published `wrfout` from a live
/// simulation. Every frame from the same case carries the same case digest;
/// exact physical time selects a deterministic store slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveWrfTarget {
    pub case_sha256: String,
    pub storage_slot: u16,
    pub exact_time: RwsExactTime,
}

impl LiveWrfTarget {
    fn run_name(&self, options: &WrfProcessOptions) -> Result<String, String> {
        if self.case_sha256.len() != 64
            || !self
                .case_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("live simulation case identity must be a full SHA-256".to_string());
        }
        let origin = self.exact_time.origin_unix().ok_or_else(|| {
            "live simulation exact time cannot represent its initialization".to_string()
        })?;
        let stamp = chrono::DateTime::from_timestamp(origin, 0)
            .map(|time| time.format("%Y%m%d%H%M%S").to_string())
            .ok_or_else(|| "live simulation initialization is out of range".to_string())?;
        Ok(format!(
            "simulation_{stamp}_{}_{}_{}",
            &self.case_sha256[..16],
            processing_profile_suffix(options),
            crate::local_import::IMPORT_SCIENCE_SCHEMA_VERSION
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WrfProcessOptions {
    #[serde(default = "default_true")]
    pub core_fields: bool,
    #[serde(default = "default_true")]
    pub diagnostics: bool,
    #[serde(default)]
    pub heavy_ecape: bool,
    #[serde(default = "default_true")]
    pub raw_extras: bool,
    /// Ingest every stored `(Time, south_north, west_east)` plane the file
    /// carries, including ones no catalog in this tree names.
    ///
    /// Default ON. It is what makes `--products var:wrf_<name>` able to
    /// reach a variable a user added to their own WRF Registry: without it
    /// the science route writes only its two fixed catalogs and the
    /// renderer refuses with `stored 2-D variable "wrf_<name>" does not
    /// exist`. Turning it OFF is a SPEED choice, not a correctness one —
    /// enumerating a file's variables needs the NetCDF metadata index that
    /// the wrf-core fast path otherwise never builds.
    #[serde(default = "default_true")]
    pub stored_planes: bool,
    #[serde(default)]
    pub only: Vec<String>,
    #[serde(default)]
    pub skip: Vec<String>,
    /// Explicit live-view profile: chart planes only, no sounding volumes.
    #[serde(default)]
    pub viewer_2d: bool,
    #[serde(default)]
    pub chart_selectors: Vec<FieldSelector>,
    /// Restrict diagnostic grids to the named renderer catalog. Core planes,
    /// raw input planes and sounding volumes retain their existing paths.
    #[serde(default)]
    pub named_products_only: bool,
}

impl Default for WrfProcessOptions {
    fn default() -> Self {
        Self {
            core_fields: true,
            diagnostics: true,
            heavy_ecape: false,
            raw_extras: true,
            stored_planes: true,
            only: Vec::new(),
            skip: Vec::new(),
            viewer_2d: false,
            chart_selectors: Vec::new(),
            named_products_only: false,
        }
    }
}

impl WrfProcessOptions {
    fn needs_diagnostic_grid(&self, name: &str, store_name: &str) -> bool {
        !self.named_products_only
            || DIAGNOSTIC_CHART_PLANES
                .iter()
                .any(|(diagnostic, _, _)| *diagnostic == name)
            || rustwx_products::derived::store_derived_recipe_slugs().contains(&store_name)
    }

    pub fn normalized(mut self) -> Self {
        self.only = normalize_filter_tokens(self.only);
        self.skip = normalize_filter_tokens(self.skip);
        self.chart_selectors.sort_by_key(|selector| selector.key());
        self.chart_selectors.dedup();
        self
    }

    /// The store field names the current selection WOULD write, for the
    /// import UI's "what will be processed" preview. Mirrors the decisions in
    /// [`read_wrf_products`] using the same [`Self::should_process`] predicate
    /// and the same field catalogs, so the preview tracks the real output.
    /// This is a static plan (it never opens a file); a field a given `wrfout`
    /// happens not to carry is simply skipped at process time with a note.
    pub fn planned_store_fields(&self) -> Vec<String> {
        let mut names = Vec::new();
        for (wrf_name, store_name) in CORE_FIELD_CATALOG {
            if self.should_process(wrf_name, Some(store_name), WrfProductGroup::Core) {
                names.push((*store_name).to_string());
            }
        }
        // Isobaric sounding volumes ride along with the core group (they are
        // gated on `core_fields` in `read_wrf_products`).
        if self.viewer_2d {
            names.extend(self.chart_selectors.iter().map(|selector| selector.key()));
        } else if self.core_fields {
            for iso in ISO_VOLUME_NAMES {
                names.push((*iso).to_string());
            }
        }
        for def in VARS {
            if def.dim != VarDim::TwoD || excluded_from_full_twod_pass(def.name) {
                continue;
            }
            let store_name = derived_name(def.name, None);
            let group = if is_heavy_wrf_diagnostic(&store_name) || is_heavy_wrf_diagnostic(def.name)
            {
                WrfProductGroup::Heavy
            } else {
                WrfProductGroup::Diagnostic
            };
            if self.needs_diagnostic_grid(def.name, &store_name)
                && self.should_process(def.name, Some(&store_name), group) {
                names.push(store_name);
            }
        }
        for raw in RAW_EXTRA_CATALOG {
            let store_name = derived_name(raw, None);
            if self.should_process(raw, Some(&store_name), WrfProductGroup::Raw) {
                names.push(store_name);
            }
        }
        names.extend(crate::wrf_column_planes::planned_store_fields(self));
        names.extend(
            self.planned_store_selectors()
                .into_iter()
                .map(store_name_for_selector),
        );
        names.sort();
        names.dedup();
        names
    }

    /// The canonical SELECTORS the current selection would write: the
    /// question a product recipe asks.
    ///
    /// [`Self::planned_store_fields`] answers in store NAMES, and a
    /// recipe's requirements are selectors (`temperature_2m_agl`, not
    /// `temperature_2m`), so holding one against the other was a
    /// comparison of two vocabularies.  Measured on a real child it called
    /// `2m_temperature` and `500mb_height_winds` undrawable on a run that
    /// then drew 143 pictures of them, and that reading was retired.  This
    /// answers in the recipe's own vocabulary, from the same tables the
    /// writer reads: [`core_field_selector`] for the core planes, the
    /// isobaric recipe ladder the volumes publish,
    /// [`crate::wrf_column_planes::COLUMN_PLANE_CATALOG`], and the three
    /// layer cloud planes.  A selector a given wrfout cannot fill is
    /// skipped at process time with a note, exactly like a planned field.
    pub fn planned_store_selectors(&self) -> Vec<FieldSelector> {
        let mut selectors = Vec::new();
        for (wrf_name, store_name) in CORE_FIELD_CATALOG {
            if self.should_process(wrf_name, Some(store_name), WrfProductGroup::Core) {
                selectors.push(core_field_selector(store_name));
            }
        }
        let mut isobaric = |field: CanonicalField, level: u16| {
            let selector = FieldSelector::isobaric(field, level);
            let key = selector.key();
            if self.should_process(&key, Some(&key), WrfProductGroup::Core) {
                selectors.push(selector);
            }
        };
        if self.viewer_2d {
            for selector in &self.chart_selectors {
                if let rustwx_core::VerticalSelector::IsobaricHpa(level) = selector.vertical {
                    if ISOBARIC_RECIPE_LEVELS_HPA.contains(&level) {
                        isobaric(selector.field, level);
                    }
                }
            }
        } else if self.core_fields {
            for field in ISOBARIC_RECIPE_FIELDS {
                for level in ISOBARIC_RECIPE_LEVELS_HPA {
                    isobaric(field, level);
                }
            }
        }
        selectors.extend(crate::wrf_column_planes::planned_store_selectors(self));
        for (diagnostic, _, field) in DIAGNOSTIC_CHART_PLANES {
            let source = derived_name(diagnostic, None);
            if self.needs_diagnostic_grid(diagnostic, &source)
                && self.should_process(diagnostic, Some(&source), WrfProductGroup::Diagnostic)
            {
                let selector = FieldSelector::entire_atmosphere(field);
                let key = selector.key();
                if self.should_process(&key, Some(&key), WrfProductGroup::Diagnostic) {
                    selectors.push(selector);
                }
            }
        }
        selectors.sort_by_key(|selector| selector.key());
        selectors.dedup();
        selectors
    }

    pub(crate) fn should_process(
        &self,
        wrf_name: &str,
        store_name: Option<&str>,
        group: WrfProductGroup,
    ) -> bool {
        match group {
            WrfProductGroup::Core if !self.core_fields => return false,
            WrfProductGroup::Diagnostic if !self.diagnostics => return false,
            WrfProductGroup::Heavy if !self.heavy_ecape => return false,
            WrfProductGroup::Raw if !self.raw_extras => return false,
            _ => {}
        }

        let keys = product_filter_keys(wrf_name, store_name);
        if !self.only.is_empty()
            && !self
                .only
                .iter()
                .any(|token| keys.iter().any(|key| filter_token_matches(key, token)))
        {
            return false;
        }
        !self
            .skip
            .iter()
            .any(|token| keys.iter().any(|key| filter_token_matches(key, token)))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WrfProductGroup {
    Core,
    Diagnostic,
    Heavy,
    Raw,
}

/// Core 2D surface fields the heavy path writes, as `(WRF filter key, store
/// field name)`. Single source of truth for the `Core` group, shared by the
/// processor and the UI's planned-field preview so the two never drift.
/// `U10`/`V10` remain the public filter keys for compatibility, but their
/// canonical earth-relative values are split from one `uvmet10` diagnostic;
/// raw WRF U10/V10 are grid-relative and must never be published under the
/// canonical component names. `PSFC`/`apcp` are pushed by dedicated blocks but
/// still belong to the `Core` group (they check `should_process(.., Core)`).
const CORE_FIELD_CATALOG: &[(&str, &str)] = &[
    ("terrain", "orography"),
    ("t2", "temperature_2m"),
    ("dp2m", "dewpoint_2m"),
    ("rh2m", "relative_humidity_2m"),
    ("U10", "u_10m"),
    ("V10", "v_10m"),
    ("wspd10", "wind_speed_10m"),
    ("slp", "mslp"),
    ("PSFC", "surface_pressure"),
    ("pw", "pwat"),
    (COMPOSITE_REFLECTIVITY_FILTER, COMPOSITE_REFLECTIVITY_STORE),
    (REFLECTIVITY_1KM_FILTER, REFLECTIVITY_1KM_STORE),
    ("UP_HELI_MAX", "updraft_helicity_2to5km"),
    ("apcp", "apcp"),
];

/// The canonical selector each [`CORE_FIELD_CATALOG`] row is written under.
///
/// ONE declaration for the writer and the plan: [`read_wrf_products`]
/// writes every core plane under the selector this returns, and
/// [`WrfProcessOptions::planned_store_selectors`] plans the same one, so a
/// plan review and the store it predicts cannot name two different
/// selectors for one plane.  Every catalog row has an arm; the
/// `every_core_row_has_one_selector` test holds the two tables together.
pub(crate) fn core_field_selector(store_name: &str) -> FieldSelector {
    match store_name {
        "orography" => FieldSelector::surface(CanonicalField::GeopotentialHeight),
        "temperature_2m" => FieldSelector::height_agl(CanonicalField::Temperature, 2),
        "dewpoint_2m" => FieldSelector::height_agl(CanonicalField::Dewpoint, 2),
        "relative_humidity_2m" => FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2),
        "u_10m" => FieldSelector::height_agl(CanonicalField::UWind, 10),
        "v_10m" => FieldSelector::height_agl(CanonicalField::VWind, 10),
        "wind_speed_10m" => FieldSelector::height_agl(CanonicalField::WindSpeed, 10),
        "mslp" => FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
        "surface_pressure" => FieldSelector::surface(CanonicalField::Pressure),
        "pwat" => FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater),
        COMPOSITE_REFLECTIVITY_STORE => composite_reflectivity_selector(),
        REFLECTIVITY_1KM_STORE => reflectivity_1km_selector(),
        "updraft_helicity_2to5km" => {
            FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000)
        }
        "apcp" => FieldSelector::surface(CanonicalField::TotalPrecipitation),
        other => panic!("core store field {other:?} has no row in core_field_selector"),
    }
}

/// Chart planes use selector keys unless a shared writer table assigns a
/// public store name to that selector.
fn store_name_for_selector(selector: FieldSelector) -> String {
    CORE_FIELD_CATALOG
        .iter()
        .find(|(_, name)| core_field_selector(name) == selector)
        .map(|(_, name)| *name)
        .or_else(|| {
            crate::wrf_column_planes::COLUMN_PLANE_CATALOG
                .iter()
                .find(|plane| plane.selector() == selector)
                .map(|plane| plane.store_name)
        })
        .map(str::to_string)
        .unwrap_or_else(|| selector.key())
}

/// The interpolated volumes whose recipe levels are published as
/// isobaric selector planes, and the canonical field each one is.  Read
/// by the writer ([`push_isobaric_recipe_planes`]) and the plan alike.
const ISOBARIC_RECIPE_VOLUMES: [(&str, CanonicalField); 7] = [
    ("temperature_iso", CanonicalField::Temperature),
    ("dewpoint_iso", CanonicalField::Dewpoint),
    ("u_iso", CanonicalField::UWind),
    ("v_iso", CanonicalField::VWind),
    ("height_iso", CanonicalField::GeopotentialHeight),
    ("rh_chart_levels", CanonicalField::RelativeHumidity),
    ("avo_chart_levels", CanonicalField::AbsoluteVorticity),
];

/// The chart-level selector whose store name is `key`, among the fields
/// and levels [`push_isobaric_recipe_planes`] publishes; `None` for any
/// other name. A `var:` request names a plane by its store name, and the
/// 2-D viewer builds isobaric planes only for the selectors it is handed.
pub(crate) fn isobaric_recipe_selector_for_key(key: &str) -> Option<FieldSelector> {
    ISOBARIC_RECIPE_FIELDS
        .iter()
        .flat_map(|field| {
            ISOBARIC_RECIPE_LEVELS_HPA
                .iter()
                .map(|level| FieldSelector::isobaric(*field, *level))
        })
        .find(|selector| selector.key() == key)
}

/// The canonical fields [`ISOBARIC_RECIPE_VOLUMES`] publishes.
const ISOBARIC_RECIPE_FIELDS: [CanonicalField; 7] = [
    ISOBARIC_RECIPE_VOLUMES[0].1,
    ISOBARIC_RECIPE_VOLUMES[1].1,
    ISOBARIC_RECIPE_VOLUMES[2].1,
    ISOBARIC_RECIPE_VOLUMES[3].1,
    ISOBARIC_RECIPE_VOLUMES[4].1,
    ISOBARIC_RECIPE_VOLUMES[5].1,
    ISOBARIC_RECIPE_VOLUMES[6].1,
];

/// Diagnostic outputs republished under canonical entire-atmosphere
/// selectors: (source diagnostic, split store plane, canonical field).
/// The named import, field plan and writer share these dependency rows.
/// No source produces total cloud cover, so no row publishes that field.
const DIAGNOSTIC_CHART_PLANES: [(&str, &str, CanonicalField); 3] = [
    (
        "cloudfrac",
        "wrf_cloudfrac_low",
        CanonicalField::LowCloudCover,
    ),
    (
        "cloudfrac",
        "wrf_cloudfrac_mid",
        CanonicalField::MiddleCloudCover,
    ),
    (
        "cloudfrac",
        "wrf_cloudfrac_high",
        CanonicalField::HighCloudCover,
    ),
];

/// Isobaric sounding volumes written alongside the `Core` group (skew-T
/// columns). 3D `pressure3d` store variables, not 2D fields.
const ISO_VOLUME_NAMES: &[&str] = &[
    "temperature_iso",
    "dewpoint_iso",
    "u_iso",
    "v_iso",
    "height_iso",
];

/// Raw WRF model outputs pulled verbatim (no `getvar` diagnostic) for the
/// `Raw` extras group. Single source of truth shared by the processor loop and
/// the planned-field preview.
///
/// Public so `tests/engine_precipitation_catalog.rs` can hold it to the
/// engine's own precipitation inventory
/// (gpuwm/physics_consumer_export_v1.json).
pub const RAW_EXTRA_CATALOG: &[&str] = &[
    "PBLH",
    "HFX",
    "LH",
    "SWDOWN",
    "GLW",
    "OLR",
    "TSK",
    "SST",
    "SNOWNC",
    "GRAUPELNC",
    // Hail accumulator. The engine writes HAILNC for every run
    // (gpuwm/io/wrf_output_schema.py PRECIPITATION_OUTPUT_FIELDS) and two
    // shipped schemes fill it -- Milbrandt-Yau mp=9 and NSSL-2 mp=18 --
    // but it had no catalog row, so it was absent from the planned-field
    // preview and reached a panel only through the stored-plane fallback
    // on the generic ramp (audit R-053). `engine_precipitation_catalog.rs`
    // holds this list to the engine's own inventory.
    //
    // Its unit comes from the file (the raw-extras loop in
    // `read_wrf_products` reads the units attribute), not from wrf-core's
    // raw-name fallback table: WRF's Registry declares HAILNC in mm
    // (Registry.EM_COMMON:1592) and every wrfout this renderer reads --
    // stock WRF's and ArWen's alike -- carries that attribute, which is
    // what the viewer's QPF palette arm matches on. The fallback table
    // lives under `vendor/crates-io`, which VENDOR.md forbids editing
    // (each crate carries a `.cargo-checksum.json` cargo validates at
    // build time), so the row that would have been added there is
    // deliberately not.
    "HAILNC",
    "WSPD10MAX",
    "UP_HELI_MAX",
    // Column extremes of vertical velocity. The same family as the two
    // above -- WRF Registry column/period maxima carried verbatim -- and
    // the only way a SURFACE snapshot's updraft reaches a panel at all,
    // since such a file has no profile to take a maximum over. gpuwm's
    // tile-streamed lane publishes its own instantaneous column extremes
    // under these names and states the window divergence in a global
    // attribute of its own.
    "W_UP_MAX",
    "W_DN_MAX",
];

const PROFILE_FNV64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const PROFILE_FNV64_PRIME: u64 = 0x0000_0100_0000_01b3;
/// WRF-specific science marker. Keep this in the processing profile and the
/// writer provenance so a reflectivity-method change cannot silently replace
/// an older imported run. v5: raw extras carry the file's own units, so an
/// import that stored HAILNC or UP_HELI_MAX with no unit is not reused.
const WRF_PROCESS_SCIENCE_MARKER: &str = "wrf_science_v5";
const COMPOSITE_REFLECTIVITY_FILTER: &str = "maxdbz";
const COMPOSITE_REFLECTIVITY_STORE: &str = "composite_reflectivity";
const REFLECTIVITY_1KM_FILTER: &str = "reflectivity_1km";
const REFLECTIVITY_1KM_STORE: &str = "reflectivity_1km";
const MIN_VALID_REFLECTIVITY_DBZ: f64 = -100.0;
const MAX_VALID_REFLECTIVITY_DBZ: f64 = 120.0;

#[derive(Debug)]
pub enum WrfProcessMessage {
    Progress(String),
    Done(Result<WrfProcessSummary, String>),
}

#[derive(Debug, Clone)]
pub struct WrfProcessSummary {
    pub store_root: PathBuf,
    pub model: String,
    pub run: String,
    pub files_seen: usize,
    pub hours_written: usize,
    pub variables: Vec<String>,
    pub notes: Vec<String>,
    /// Which input file each stored slot was read from, `(slot, file)`.
    /// What lets a per-frame event name the FILE it is about: a series
    /// import has many inputs and a slot is not a filename.
    pub frame_sources: Vec<(u16, PathBuf)>,
}

pub(crate) struct WrfHourFields {
    grid: LatLonGrid,
    projection: Option<GridProjection>,
    canonical: Vec<(String, SelectedField2D)>,
    derived: Vec<OwnedDerivedField>,
    volumes: Vec<IsoVolume>,
    pub(crate) notes: Vec<String>,
}

struct OwnedDerivedField {
    name: String,
    units: String,
    values: Vec<f32>,
}

fn volume_omission_note(retained_2d_products: usize, error: &str) -> String {
    let suffix = if retained_2d_products == 1 { "" } else { "s" };
    format!(
        "WRF 3-D pressure-volume products omitted; retained {retained_2d_products} independently available 2-D product{suffix}: {error}"
    )
}

pub fn spawn_process_paths(
    paths: Vec<PathBuf>,
    store_root: PathBuf,
    options: WrfProcessOptions,
) -> WrfProcessTask {
    spawn_process_paths_inner(paths, store_root, options, None)
}

pub fn spawn_process_live_path(
    path: PathBuf,
    store_root: PathBuf,
    options: WrfProcessOptions,
    target: LiveWrfTarget,
) -> WrfProcessTask {
    spawn_process_paths_inner(vec![path], store_root, options, Some(target))
}

fn spawn_process_paths_inner(
    paths: Vec<PathBuf>,
    store_root: PathBuf,
    options: WrfProcessOptions,
    live_target: Option<LiveWrfTarget>,
) -> WrfProcessTask {
    let options = options.normalized();
    let label = if paths.len() == 1 {
        format!("Process WRF {}", display_name(&paths[0]))
    } else {
        format!("Process {} WRF files", paths.len())
    };
    let (tx, rx) = channel();
    let worker_tx = tx.clone();
    let spawn_result = std::thread::Builder::new()
        .name("rw-ui-wrf-process".to_string())
        .spawn({
            let label = label.clone();
            move || {
                let result = isolate_panics("WRF processing worker", || {
                    lower_import_thread_priority();
                    process_paths_with_target(
                        &paths,
                        &store_root,
                        &options,
                        &worker_tx,
                        live_target.as_ref(),
                    )
                    .map_err(|err| {
                        if err.trim().is_empty() {
                            format!("{label} failed")
                        } else {
                            err
                        }
                    })
                });
                let _ = worker_tx.send(WrfProcessMessage::Done(result));
            }
        });
    if let Err(err) = spawn_result {
        let _ = tx.send(WrfProcessMessage::Done(Err(format!(
            "could not start WRF processing worker: {err}"
        ))));
    }
    WrfProcessTask { label, rx }
}

/// Large-grid imports grind for minutes with heavy allocation churn; run the
/// worker below normal priority so the desktop stays responsive. The shared
/// throttle helper is Windows-specific and a no-op on other platforms.
pub(crate) fn lower_import_thread_priority() {
    rw_ingest::throttle::set_current_thread_background_priority();
}

pub fn is_supported_wrf_file(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    name.starts_with("wrfout")
        || matches!(
            path.extension()
                .and_then(|value| value.to_str())
                .map(|value| value.to_ascii_lowercase())
                .as_deref(),
            Some("wrf" | "nc" | "nc4" | "cdf")
        )
}

pub fn wrf_files_in_folder(folder: &Path) -> Vec<PathBuf> {
    const MAX_DEPTH: usize = 8;
    const MAX_FILES: usize = 10_000;

    let mut paths = Vec::new();
    let mut stack = vec![(folder.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && is_supported_wrf_file(&path) {
                paths.push(path);
                if paths.len() >= MAX_FILES {
                    paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
                    return paths;
                }
            } else if depth < MAX_DEPTH && path.is_dir() {
                stack.push((path, depth + 1));
            }
        }
    }
    paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    paths
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WrfSourceKind {
    Raw,
    Postprocessed,
}

#[derive(Debug, Clone)]
struct WrfSourcePlan {
    path: PathBuf,
    kind: WrfSourceKind,
    records: Vec<crate::local_import::PlannedSourceTime>,
    /// Where the file's grid sits in its parent (raw wrfouts only): a
    /// moving nest's frames record different places ([`crate::nest_move`]).
    placement: Option<crate::nest_move::NestPlacement>,
    /// The file's grid spacing (`DX`, metres), which scales how closely two
    /// places' coordinates must agree on shared ground.
    dx_m: Option<f64>,
}

/// A series whose frames sit at more than one place of one moving nest: the
/// place every frame is stored on (the LAST frame's, the one the pictures
/// are drawn for), its grid, and each source's move onto it.
struct NestMoves {
    target: crate::nest_move::NestPlacement,
    target_path: PathBuf,
    target_grid: LatLonGrid,
    tolerance_deg: f64,
    /// Per plan: the move onto the target, `None` for a source already there.
    shifts: Vec<Option<crate::nest_move::NestShift>>,
}

/// Plan how a series that spans a nest's moves lands on one grid, or `None`
/// when every source sits at one place (the store's ordinary case, left
/// byte for byte as it was).  Sources that do not all record a placement of
/// one nest are also left alone: the store then refuses their grids by name,
/// as it always has.
fn plan_nest_moves(plans: &[WrfSourcePlan]) -> Result<Option<NestMoves>, String> {
    use crate::nest_move::{NestShift, coordinate_tolerance_deg};

    if plans.len() < 2 || plans.iter().any(|plan| plan.kind != WrfSourceKind::Raw) {
        return Ok(None);
    }
    let Some(placements) = plans
        .iter()
        .map(|plan| plan.placement)
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    if placements.iter().all(|placement| *placement == placements[0]) {
        return Ok(None);
    }
    let last_valid = |plan: &WrfSourcePlan| plan.records.iter().map(|r| r.valid_unix).max();
    let target_index = (0..plans.len())
        .max_by_key(|&index| last_valid(&plans[index]))
        .ok_or("a series with no source cannot move")?;
    let target = placements[target_index];
    let Some(shifts) = placements
        .iter()
        .map(|placement| {
            NestShift::between(*placement, target).map(|shift| (!shift.is_null()).then_some(shift))
        })
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let target_plan = &plans[target_index];
    let record = target_plan
        .records
        .last()
        .ok_or_else(|| format!("{} plans no time", target_plan.path.display()))?;
    let file = isolate_panics("open WRF file", || {
        WrfFile::open(&target_plan.path).map_err(|err| err.to_string())
    })
    .map_err(|err| format!("Open WRF {} failed: {err}", target_plan.path.display()))?;
    let shape = GridShape::new(file.nx, file.ny).map_err(|err| err.to_string())?;
    let lat = file
        .xlat(record.time_index)
        .map_err(|err| format!("Read XLAT from {} failed: {err}", target_plan.path.display()))?;
    let lon = file
        .xlong(record.time_index)
        .map_err(|err| format!("Read XLONG from {} failed: {err}", target_plan.path.display()))?;
    let target_grid = LatLonGrid::new(
        shape,
        lat.iter().map(|value| *value as f32).collect(),
        lon.iter().map(|value| *value as f32).collect(),
    )
    .map_err(|err| err.to_string())?;
    Ok(Some(NestMoves {
        target,
        target_path: target_plan.path.clone(),
        target_grid,
        tolerance_deg: coordinate_tolerance_deg(target_plan.dx_m.unwrap_or(f64::NAN)),
        shifts,
    }))
}

/// Move one hour's fields from a place the nest has since left onto the
/// series' target place: every plane shifts by whole cells and holds NaN
/// on ground the earlier place did not cover.  Checked against the hour's
/// own coordinates first ([`crate::nest_move::verify_shift`]).
///
/// A place the nest has travelled a whole width away from shares no cell
/// with the target: its recorded move leaves nothing in common and no
/// whole-cell move lands its coordinates there.  Its frame is stored as
/// missing everywhere, which is what it holds on the target's ground, and
/// the series goes on; refusing it refused every product of every later
/// place of the run (a storm-following nest leaves its first footprint
/// within a day).  A frame whose recorded move DOES leave shared cells but
/// whose coordinates disagree is still refused by name.
fn move_hour_fields(
    fields: &mut WrfHourFields,
    moves: &NestMoves,
    shift: crate::nest_move::NestShift,
    path: &Path,
    placement: Option<crate::nest_move::NestPlacement>,
    label: &str,
) -> Result<(), String> {
    use crate::nest_move::{locate_shift, shift_plane, verify_shift};

    let (nx, ny) = (moves.target_grid.shape.nx, moves.target_grid.shape.ny);
    let (shift, shared, read_from) =
        match verify_shift(&fields.grid, &moves.target_grid, shift, moves.tolerance_deg) {
            Ok(shared) => (shift, shared, "its recorded place"),
            Err(err) => match locate_shift(&fields.grid, &moves.target_grid, moves.tolerance_deg)
            {
                Some((located, shared)) => (
                    located,
                    shared,
                    "its coordinates (its recorded place is in a parent that moved too)",
                ),
                None if !shift.shares_cell(nx, ny) => (
                    shift,
                    0,
                    "its recorded place, a whole nest width from the series' place",
                ),
                None => {
                    return Err(format!(
                        "WRF {} ({label}) cannot be moved onto {}'s place: {err}, and no \
                         whole-cell move lands its coordinates there; storing it there would \
                         put the values of different ground on this place's cells",
                        path.display(),
                        moves.target_path.display()
                    ));
                }
            },
        };
    for (_, field) in fields.canonical.iter_mut() {
        field.values = shift_plane(&field.values, nx, ny, shift);
        field.grid = moves.target_grid.clone();
    }
    for field in fields.derived.iter_mut() {
        field.values = shift_plane(&field.values, nx, ny, shift);
    }
    for volume in fields.volumes.iter_mut() {
        for (_, plane) in volume.levels.iter_mut() {
            *plane = shift_plane(plane, nx, ny, shift);
        }
    }
    fields.grid = moves.target_grid.clone();
    let from = placement
        .map(|p| format!("({}, {})", p.i_parent_start, p.j_parent_start))
        .unwrap_or_else(|| "?".to_string());
    fields.notes.push(format!(
        "{} ({label}): a moving nest's earlier place {from} in its parent, moved onto the \
         place ({}, {}) of the series' last frame by ({}, {}) nest cells read from {read_from}; \
         {} of {} cells are shared ground and the rest are missing for this frame",
        display_name(path),
        moves.target.i_parent_start,
        moves.target.j_parent_start,
        shift.dx,
        shift.dy,
        shared,
        nx * ny
    ));
    Ok(())
}

/// A model label the metadata row cannot take is said once and dropped:
/// the plots keep the store identity rather than the import failing over
/// a name.
fn unusable_model_label(
    tx: &Sender<WrfProcessMessage>,
    path: &Path,
    reason: &str,
) -> Option<String> {
    let _ = tx.send(WrfProcessMessage::Progress(format!(
        "{}: {reason}; the plots keep the store identity",
        display_name(path)
    )));
    None
}

fn preflight_wrf_sources(
    files: &[PathBuf],
    source_identity: &str,
    processing_profile: &str,
    tx: &Sender<WrfProcessMessage>,
) -> Result<(Vec<WrfSourcePlan>, String), String> {
    let mut expected_shape = None::<(usize, usize)>;
    let mut sources = Vec::with_capacity(files.len());
    let mut kinds = Vec::with_capacity(files.len());
    let mut places = Vec::with_capacity(files.len());
    let mut model_labels = Vec::<Option<String>>::with_capacity(files.len());
    for path in files {
        let _ = tx.send(WrfProcessMessage::Progress(format!(
            "Preflighting WRF {}",
            display_name(path)
        )));
        let raw = isolate_panics("preflight WRF file", || {
            WrfFile::open(path).map_err(|err| err.to_string())
        });
        let (kind, source_times, shape) = match raw {
            Ok(file) => {
                // Every source of one invocation belongs to one run (the
                // timeline planner refuses sources whose reference times
                // disagree), so what that run was initialized from is
                // recorded once, here, where the file is already open.  Both
                // reader arms must answer this: a wrfout that falls back to
                // netcrust carries the same provenance, and a lead disclosed
                // under one reader and hidden under the other would make the
                // label depend on the file's encoding rather than on the run.
                rustwx_products::shared_context::set_initial_condition_disclosure(
                    crate::local_import::initial_condition_disclosure(&file),
                );
                model_labels.push(
                    crate::local_import::model_label(&file)
                        .unwrap_or_else(|reason| unusable_model_label(tx, path, &reason)),
                );
                let times = crate::local_import::wrf_source_times(&file, path)?;
                places.push((
                    crate::nest_move::NestPlacement::of(&file),
                    file.global_attr_f64("DX").ok(),
                ));
                (WrfSourceKind::Raw, times, (file.nx, file.ny))
            }
            Err(_) => {
                let nc = netcrust::open(path).map_err(|err| {
                    format!("Open post-processed WRF {} failed: {err}", path.display())
                })?;
                rustwx_products::shared_context::set_initial_condition_disclosure(
                    crate::local_import::netcdf_initial_condition_disclosure(&nc),
                );
                model_labels.push(
                    crate::local_import::netcdf_model_label(&nc)
                        .unwrap_or_else(|reason| unusable_model_label(tx, path, &reason)),
                );
                let times = crate::local_import::netcdf_source_times(&nc, path)
                    .map_err(|err| format!("Read times from {} failed: {err}", path.display()))?;
                let shape = crate::local_import::netcdf_grid_shape(&nc, path).map_err(|err| {
                    format!("Read grid shape from {} failed: {err}", path.display())
                })?;
                places.push((None, None));
                (WrfSourceKind::Postprocessed, times, shape)
            }
        };
        crate::local_import::merge_preflight_grid_shape(&mut expected_shape, shape, path)?;
        sources.push((path.clone(), source_times));
        kinds.push(kind);
    }
    // The model the files name (a mesh or global model's frames import as
    // `wrf` like any wrfout) is the run's, when every file names the same
    // one; recorded once, beside the disclosure.
    rustwx_products::shared_context::set_model_label(
        rustwx_products::shared_context::agreed_model_label(&model_labels),
    );
    let timeline = crate::local_import::ForecastHourTimeline::plan_all(&sources)?;
    let mut plans = Vec::with_capacity(sources.len());
    for (index, (((path, _), kind), (placement, dx_m))) in
        sources.into_iter().zip(kinds).zip(places).enumerate()
    {
        let records = timeline
            .records_for_source(index)
            .ok_or_else(|| {
                format!(
                    "internal error: WRF forecast timeline omitted source {}",
                    path.display()
                )
            })?
            .to_vec();
        plans.push(WrfSourcePlan {
            path,
            kind,
            records,
            placement,
            dx_m,
        });
    }
    let run = timeline.run_name(source_identity, processing_profile);
    Ok((plans, run))
}

fn process_paths(
    paths: &[PathBuf],
    store_root: &Path,
    options: &WrfProcessOptions,
    tx: &Sender<WrfProcessMessage>,
) -> Result<WrfProcessSummary, String> {
    process_paths_with_target(paths, store_root, options, tx, None)
}

fn process_paths_with_target(
    paths: &[PathBuf],
    store_root: &Path,
    options: &WrfProcessOptions,
    tx: &Sender<WrfProcessMessage>,
    live_target: Option<&LiveWrfTarget>,
) -> Result<WrfProcessSummary, String> {
    if paths.is_empty() {
        return Err("No WRF files selected".to_string());
    }
    // `spawn_process_paths` normalizes at the public boundary, but keeping the
    // worker itself canonical makes direct/internal callers obey the same
    // filtering semantics and guarantees the profile key describes the plan
    // actually executed.
    let normalized_options = options.clone().normalized();
    let options = &normalized_options;

    let mut files = paths
        .iter()
        .filter(|path| is_supported_wrf_file(path))
        .cloned()
        .collect::<Vec<_>>();
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    if files.is_empty() {
        return Err("No supported WRF files selected".to_string());
    }

    let source_snapshot = crate::local_import::capture_source_set_identity(&files)?;
    let source_identity = &source_snapshot.identity;
    // A store hour is replaced as a unit. Keep every full-processing plan in
    // its own run so a later core-only/custom import cannot erase fields from
    // an earlier default or heavy import of the exact same source files.
    let processing_profile = processing_profile_suffix(options);
    let model = "wrf".to_string();
    let (mut plans, discovered_run) =
        preflight_wrf_sources(&files, source_identity, &processing_profile, tx)?;
    let run = if let Some(target) = live_target {
        if plans.len() != 1 || plans[0].records.len() != 1 {
            return Err(
                "a live wrfout publication must contain exactly one source time".to_string(),
            );
        }
        let record = &mut plans[0].records[0];
        let source_lead = record
            .exact_time
            .map(|time| time.lead_seconds)
            .unwrap_or_else(|| u64::from(record.storage_slot) * 3_600);
        if record.valid_unix != target.exact_time.valid_unix
            || source_lead != target.exact_time.lead_seconds
        {
            return Err(format!(
                "live wrfout physical time does not match its supervised case target: source lead={} valid={}, target lead={} valid={}",
                source_lead,
                record.valid_unix,
                target.exact_time.lead_seconds,
                target.exact_time.valid_unix
            ));
        }
        record.storage_slot = target.storage_slot;
        record.exact_time = Some(target.exact_time);
        target.run_name(options)?
    } else {
        discovered_run
    };
    // The run name keys every byte of the sources and the processing plan, so
    // a complete run already published under it, by this same binary, is the
    // import this call would write.  It is answered from the store.
    if live_target.is_none() {
        if let Some(record) = published_import_record(store_root, &model, &run, &plans) {
            let _ = tx.send(WrfProcessMessage::Progress(format!(
                "{REUSED_IMPORT_PROGRESS} {model}/{run}: this store already holds these \
                 sources imported under this plan by this build"
            )));
            return Ok(WrfProcessSummary {
                store_root: store_root.to_path_buf(),
                model,
                run,
                files_seen: files.len(),
                hours_written: record.hours_written,
                variables: record.variables,
                notes: record.notes,
                frame_sources: plan_frame_sources(&plans),
            });
        }
    }
    // Whole selected-file imports publish one complete run directory. A live
    // simulation instead appends one independently validated hour to a stable
    // case run; rw-store's per-run lock, atomic hour writer, grid identity,
    // exact-time remap guard, and manifest-last publication own that seam.
    let publisher = if live_target.is_some() {
        None
    } else {
        Some(crate::local_import::RunStagingPublisher::new(
            store_root, &model, &run,
        )?)
    };
    let staging_store_root = publisher
        .as_ref()
        .map(|publisher| publisher.staging_store_root().to_path_buf())
        .unwrap_or_else(|| store_root.to_path_buf());
    let mut written = Vec::<WrittenHour>::new();
    let mut all_vars = Vec::<String>::new();
    let mut all_notes = Vec::<String>::new();
    // A series that spans a moving nest's moves is stored on its last
    // frame's place ([`crate::nest_move`]); `None` for every other series.
    let moves = if live_target.is_none() {
        plan_nest_moves(&plans)?
    } else {
        None
    };

    for (plan_index, plan) in plans.iter().enumerate() {
        let shift = moves
            .as_ref()
            .and_then(|moves| moves.shifts.get(plan_index).copied().flatten());
        let path = &plan.path;
        let _ = tx.send(WrfProcessMessage::Progress(format!(
            "Opening WRF {}",
            display_name(path)
        )));
        // Probe the raw wrf-core reader FIRST: raw wrfouts are the common
        // wrench-flow case and the probe fails fast on post-processed files
        // (they carry no raw T). The previous order ran netcrust::open's
        // eager NetCDF-4 metadata indexing (~57 s on a 2 GB compressed
        // wrfout, docs/wrf-import-large-grids.md) on EVERY raw file just to
        // conclude "not post-processed". The two detectors cannot overlap:
        // raw files have T (wrf-core opens), post-processed have TK/Z/P and
        // no PB (netcrust path claims them).
        if plan.kind == WrfSourceKind::Postprocessed {
            let nc = netcrust::open(path).map_err(|err| {
                format!("Open post-processed WRF {} failed: {err}", path.display())
            })?;
            let compute_postproc_severe =
                crate::postproc_severe::APPROX_SEVERE_SLUGS
                    .iter()
                    .any(|slug| {
                        let slug = *slug;
                        options.should_process(slug, Some(slug), WrfProductGroup::Diagnostic)
                    });
            for record in &plan.records {
                let storage_slot = record.storage_slot;
                // Post-processed climate wrfout (CONUS-I/II, GDEX: derived
                // TK/Z/P, no raw T/PB) — wrf-core can't open these, so route
                // them through the netcrust-based reader before reporting the
                // raw open error below.
                match crate::local_import::try_postprocessed_wrf_shared(
                    &nc,
                    path,
                    record.time_index,
                    compute_postproc_severe,
                    &mut |message: String| {
                        let _ = tx.send(WrfProcessMessage::Progress(message));
                    },
                ) {
                    Ok(Some((canonical, severe, volumes, raw_2d, hour_notes))) => {
                        all_notes.extend(hour_notes);
                        let _ = tx.send(WrfProcessMessage::Progress(format!(
                            "Reading post-processed WRF {} time {} ({}) -> {}",
                            display_name(path),
                            record.time_index,
                            record.label,
                            record.display_key()
                        )));
                        let Some((_, grid_field)) = canonical.first() else {
                            return Err(format!(
                                "Post-processed WRF {} did not provide a grid-bearing field",
                                path.display()
                            ));
                        };
                        let refs = canonical
                            .iter()
                            .filter(|(name, _)| {
                                options.should_process(name, Some(name), WrfProductGroup::Core)
                            })
                            .map(|(name, field)| (name.as_str(), field))
                            .collect::<Vec<_>>();
                        // Approximate post-processed severe/thermo suite. Its
                        // `approx_*` namespace deliberately cannot masquerade as
                        // the raw-wrfout getvar diagnostics.
                        let mut derived_refs = severe
                            .iter()
                            .filter(|field| {
                                options.should_process(
                                    field.name,
                                    Some(field.name),
                                    WrfProductGroup::Diagnostic,
                                )
                            })
                            .map(|field| DerivedFieldInput {
                                name: field.name,
                                units: field.units,
                                values: field.values.as_slice(),
                            })
                            .collect::<Vec<_>>();
                        // Raw `wrf_*` planes from the 2-D wrf2d route (empty on
                        // the 3-D route) — the wrench flow imports pure surface
                        // archives the same way the light import does.
                        derived_refs.extend(
                            raw_2d
                                .iter()
                                .filter(|field| {
                                    options.should_process(
                                        field.name.as_str(),
                                        Some(field.name.as_str()),
                                        WrfProductGroup::Raw,
                                    )
                                })
                                .map(|field| DerivedFieldInput {
                                    name: field.name.as_str(),
                                    units: field.units.as_str(),
                                    values: field.values.as_slice(),
                                }),
                        );
                        let volume_inputs = if options.core_fields && !options.viewer_2d {
                            volumes.iter().map(IsoVolume::as_input).collect::<Vec<_>>()
                        } else {
                            Vec::new()
                        };
                        if refs.is_empty() && derived_refs.is_empty() && volume_inputs.is_empty() {
                            return Err(format!(
                                "Post-processed WRF {} produced no fields for the selected processing options",
                                path.display()
                            ));
                        }
                        let result = match record.exact_time {
                            Some(exact_time) => write_hour_from_grid_with_derived_exact(
                                &staging_store_root,
                                &model,
                                &run,
                                storage_slot,
                                exact_time,
                                &grid_field.grid,
                                grid_field.projection.as_ref(),
                                &refs,
                                &derived_refs,
                                &volume_inputs,
                                writer_build(),
                                now_unix(),
                            ),
                            None => write_hour_from_grid_with_derived(
                                &staging_store_root,
                                &model,
                                &run,
                                storage_slot,
                                &grid_field.grid,
                                grid_field.projection.as_ref(),
                                &refs,
                                &derived_refs,
                                &volume_inputs,
                                writer_build(),
                                now_unix(),
                            ),
                        }
                        .map_err(|err| {
                            format!("Write WRF {} failed: {err}", record.display_key())
                        })?;
                        all_vars.extend(result.vars.iter().cloned());
                        written.push(result);
                        continue;
                    }
                    Ok(None) => {
                        return Err(format!(
                            "Open WRF {} failed and the file is not a supported post-processed WRF archive",
                            path.display()
                        ));
                    }
                    Err(err) => {
                        return Err(format!("Process WRF {} failed: {err}", path.display()));
                    }
                }
            }
            continue;
        }

        let file = isolate_panics("open WRF file", || {
            WrfFile::open(path).map_err(|err| err.to_string())
        })
        .map_err(|err| format!("Open WRF {} failed after preflight: {err}", path.display()))?;
        // The wrf-core reader answers by NAME; it cannot LIST what a file
        // carries, and listing is the whole question a user-added plane
        // asks. netcrust's index answers it. Built once per file, only
        // when the stored-plane pass is on or a raw extra is selected, and
        // only for enumeration and units -- every plane's values still come
        // off the wrf-core fast path through `PlaneSource`. It is the
        // metadata indexing the post-processed probe was reordered to avoid
        // paying on raw files, so turning off both of those is the way back
        // to that cost profile.
        //
        // A raw extra needs the index for its UNITS: wrf-core's raw-name
        // fallback table has no row for HAILNC, UP_HELI_MAX, WSPD10MAX,
        // W_UP_MAX or W_DN_MAX and stored them with no unit at all, so
        // hail was drawn as a unitless -1..1 field instead of in inches on
        // the precipitation palette, although the file declares `mm`.
        let raw_extras_selected = RAW_EXTRA_CATALOG.iter().any(|raw| {
            options.should_process(raw, Some(&derived_name(raw, None)), WrfProductGroup::Raw)
        });
        let stored_plane_index = if options.stored_planes || raw_extras_selected {
            let _ = tx.send(WrfProcessMessage::Progress(format!(
                "Indexing {} for {}",
                display_name(path),
                if options.stored_planes {
                    "stored 2-D planes"
                } else {
                    "raw field units"
                }
            )));
            match netcrust::open(path) {
                Ok(index) => Some(index),
                Err(err) => {
                    all_notes.push(if options.stored_planes {
                        format!(
                            "{}: stored 2-D planes unavailable: {err}; a variable \
                             this file carries but no product catalog names will \
                             not render through var:<name>, and raw fields keep \
                             the reader's own units",
                            display_name(path)
                        )
                    } else {
                        format!(
                            "{}: file units unavailable: {err}; raw fields keep \
                             the reader's own units",
                            display_name(path)
                        )
                    });
                    None
                }
            }
        } else {
            None
        };
        for record in &plan.records {
            let timeidx = record.time_index;
            let storage_slot = record.storage_slot;
            let _ = tx.send(WrfProcessMessage::Progress(format!(
                "Computing WRF {} time {} ({}) -> {}",
                display_name(path),
                timeidx,
                record.label,
                record.display_key()
            )));
            let mut progress = |message: String| {
                let _ = tx.send(WrfProcessMessage::Progress(message));
            };
            let mut fields = read_wrf_products(
                &file,
                path,
                timeidx,
                options,
                stored_plane_index.as_ref(),
                &mut progress,
            )?;
            if let (Some(moves), Some(shift)) = (moves.as_ref(), shift) {
                move_hour_fields(
                    &mut fields,
                    moves,
                    shift,
                    path,
                    plan.placement,
                    &record.label,
                )?;
            }
            if fields.canonical.is_empty() && fields.derived.is_empty() && fields.volumes.is_empty()
            {
                return Err(format!(
                    "WRF {} time {} produced no fields for the selected processing options",
                    path.display(),
                    timeidx
                ));
            }

            let refs = fields
                .canonical
                .iter()
                .map(|(name, field)| (name.as_str(), field))
                .collect::<Vec<_>>();
            let derived_refs = fields
                .derived
                .iter()
                .map(|field| DerivedFieldInput {
                    name: field.name.as_str(),
                    units: field.units.as_str(),
                    values: field.values.as_slice(),
                })
                .collect::<Vec<_>>();
            let volume_inputs = fields
                .volumes
                .iter()
                .map(IsoVolume::as_input)
                .collect::<Vec<_>>();
            let result = match record.exact_time {
                Some(exact_time) => write_hour_from_grid_with_derived_exact(
                    &staging_store_root,
                    &model,
                    &run,
                    storage_slot,
                    exact_time,
                    &fields.grid,
                    fields.projection.as_ref(),
                    &refs,
                    &derived_refs,
                    &volume_inputs,
                    writer_build(),
                    now_unix(),
                ),
                None => write_hour_from_grid_with_derived(
                    &staging_store_root,
                    &model,
                    &run,
                    storage_slot,
                    &fields.grid,
                    fields.projection.as_ref(),
                    &refs,
                    &derived_refs,
                    &volume_inputs,
                    writer_build(),
                    now_unix(),
                ),
            }
            .map_err(|err| format!("Write WRF {} failed: {err}", record.display_key()))?;
            all_vars.extend(result.vars.iter().cloned());
            all_notes.extend(fields.notes);
            written.push(result);
        }
    }

    all_vars.sort();
    all_vars.dedup();
    all_notes.sort();
    all_notes.dedup();
    crate::local_import::verify_source_set_unchanged(&source_snapshot)?;
    if let Some(publisher) = publisher {
        // Inside the staged run, so it is published with the run it
        // describes or not at all.
        write_import_record(
            &publisher.staging_store_root().join(&model).join(&run),
            written.len(),
            &all_vars,
            &all_notes,
        )?;
        let _ = tx.send(WrfProcessMessage::Progress(format!(
            "Publishing complete WRF run {model}/{run}"
        )));
        publisher.publish()?;
    } else {
        let _ = tx.send(WrfProcessMessage::Progress(format!(
            "Published live WRF timestep into {model}/{run}"
        )));
    }
    let frame_sources = plan_frame_sources(&plans);
    Ok(WrfProcessSummary {
        store_root: store_root.to_path_buf(),
        model,
        run,
        files_seen: files.len(),
        hours_written: written.len(),
        variables: all_vars,
        notes: all_notes,
        frame_sources,
    })
}

/// Which input file each stored slot was read from, from the source plans.
fn plan_frame_sources(plans: &[WrfSourcePlan]) -> Vec<(u16, PathBuf)> {
    plans
        .iter()
        .flat_map(|plan| {
            plan.records
                .iter()
                .map(|record| (record.storage_slot, plan.path.clone()))
        })
        .collect()
}

/// The progress word a reused import is reported with.
pub const REUSED_IMPORT_PROGRESS: &str = "Reusing the imported WRF run";

/// The file a complete import keeps inside its published run directory.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): `gpuwm render` asks the catalog
/// what a set of frames can draw (`--list-products`) and then renders them,
/// two launches into one store, and each launch imported every frame.  Nine
/// long windows over four 750 m frames cost 219 CPU-s through the door
/// against 11 CPU-s in the renderer alone (about 20 to 25 min per 36 h run),
/// and the listing was about 26 of the 34.7 CPU-s of every live frame.  The
/// run name already keys every byte of the sources and the processing plan;
/// this record adds what the name cannot key, the binary that executed the
/// plan and what that import reported, so the second launch answers from
/// the store and still reports the same notes.
const IMPORT_RECORD_FILE: &str = "import-record.json";
const IMPORT_RECORD_SCHEMA: &str = "rw-wrfbatch-import-record-v1";
/// A record is a few notes and variable names; anything larger is not one.
const IMPORT_RECORD_MAX_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Serialize, Deserialize)]
struct ImportRecord {
    schema: String,
    /// The executable that wrote the run.  The run name keys the sources and
    /// the plan, not the code that executed the plan, so a store kept across
    /// an upgrade is imported again by the new build instead of being
    /// answered with the old build's fields.
    writer: String,
    hours_written: usize,
    variables: Vec<String>,
    notes: Vec<String>,
}

/// This executable's identity: its build line and the SHA-256 of its bytes,
/// read once per process.  `None` when the executable cannot be read, which
/// writes no record and so reuses nothing.
fn import_writer_identity() -> Option<&'static str> {
    use sha2::{Digest, Sha256};
    static IDENTITY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    IDENTITY
        .get_or_init(|| {
            use std::io::Read;
            let exe = std::env::current_exe().ok()?;
            let mut file = std::fs::File::open(exe).ok()?;
            let mut hasher = Sha256::new();
            let mut buffer = vec![0u8; 1 << 20];
            loop {
                let read = file.read(&mut buffer).ok()?;
                if read == 0 {
                    break;
                }
                hasher.update(&buffer[..read]);
            }
            let digest = hasher.finalize();
            let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
            Some(format!("{} {hex}", writer_build()))
        })
        .as_deref()
}

fn write_import_record(
    staged_run_dir: &Path,
    hours_written: usize,
    variables: &[String],
    notes: &[String],
) -> Result<(), String> {
    let Some(writer) = import_writer_identity() else {
        return Ok(());
    };
    let record = ImportRecord {
        schema: IMPORT_RECORD_SCHEMA.to_string(),
        writer: writer.to_string(),
        hours_written,
        variables: variables.to_vec(),
        notes: notes.to_vec(),
    };
    let path = staged_run_dir.join(IMPORT_RECORD_FILE);
    let bytes = serde_json::to_vec(&record)
        .map_err(|err| format!("encode import record {}: {err}", path.display()))?;
    std::fs::write(&path, bytes)
        .map_err(|err| format!("write import record {}: {err}", path.display()))
}

/// The record of a complete run this store already holds under `run`,
/// written by this executable and covering exactly the slots `plans` store,
/// or `None`, and the sources are imported.  Every doubt is a `None`: an
/// import is always correct, and a reuse is only a saving.
fn published_import_record(
    store_root: &Path,
    model: &str,
    run: &str,
    plans: &[WrfSourcePlan],
) -> Option<ImportRecord> {
    let writer = import_writer_identity()?;
    let run_dir = store_root.join(model).join(run);
    let real_dir = std::fs::symlink_metadata(&run_dir).ok()?;
    if real_dir.file_type().is_symlink() || !real_dir.is_dir() {
        return None;
    }
    let record_path = run_dir.join(IMPORT_RECORD_FILE);
    let metadata = std::fs::symlink_metadata(&record_path).ok()?;
    if !metadata.is_file() || metadata.len() > IMPORT_RECORD_MAX_BYTES {
        return None;
    }
    let record: ImportRecord = serde_json::from_slice(&std::fs::read(&record_path).ok()?).ok()?;
    if record.schema != IMPORT_RECORD_SCHEMA || record.writer != writer {
        return None;
    }
    let manifest =
        rw_store::run::RwsRunManifest::load_for_run(&run_dir.join("run.json"), model, run).ok()?;
    let planned: std::collections::BTreeSet<u16> = plans
        .iter()
        .flat_map(|plan| plan.records.iter().map(|record| record.storage_slot))
        .collect();
    let stored: std::collections::BTreeSet<u16> = manifest.hours.keys().copied().collect();
    if planned != stored || !run_dir.join("grid.rwg").is_file() {
        return None;
    }
    // The checks a publish makes of its staged run, so a store kept across
    // renders (the desktop importer's) imports a damaged grid or hour file
    // again instead of answering from it: the grid, and every hour's slot,
    // identity, grid and time against the manifest.
    let grid = rw_store::grid::GridFile::open(&run_dir.join("grid.rwg")).ok()?;
    manifest.validate_grid(&grid.hash, grid.nx, grid.ny).ok()?;
    for (&slot, entry) in &manifest.hours {
        let hour_path = run_dir.join(&entry.file);
        if !hour_path.is_file() {
            return None;
        }
        let reader = rw_store::reader::HourReader::open(&hour_path).ok()?;
        manifest.validate_hour_meta(slot, reader.meta()).ok()?;
    }
    Some(record)
}

fn read_wrf_products(
    file: &WrfFile,
    path: &Path,
    timeidx: usize,
    options: &WrfProcessOptions,
    stored_plane_index: Option<&netcrust::File>,
    progress: &mut impl FnMut(String),
) -> Result<WrfHourFields, String> {
    // Validate hostile/corrupt dimensions before xlat/xlong can allocate
    // coordinate planes. GridShape owns the shared desktop cell ceiling.
    let shape = GridShape::new(file.nx, file.ny).map_err(|err| err.to_string())?;
    let lat = file
        .xlat(timeidx)
        .map_err(|err| format!("Read XLAT from {} failed: {err}", path.display()))?;
    let lon = file
        .xlong(timeidx)
        .map_err(|err| format!("Read XLONG from {} failed: {err}", path.display()))?;
    if lat.len() != shape.len() || lon.len() != shape.len() {
        return Err(format!(
            "WRF {} grid mismatch: expected {} cells, got lat {} lon {}",
            path.display(),
            shape.len(),
            lat.len(),
            lon.len()
        ));
    }
    let grid = LatLonGrid::new(
        shape,
        lat.iter().map(|value| *value as f32).collect(),
        lon.iter().map(|value| *value as f32).collect(),
    )
    .map_err(|err| err.to_string())?;
    let projection = wrf_projection(file);

    let mut fields = WrfHourFields {
        grid: grid.clone(),
        projection: projection.clone(),
        canonical: Vec::new(),
        derived: Vec::new(),
        volumes: Vec::new(),
        notes: Vec::new(),
    };

    macro_rules! push_core {
        ($wrf:expr, $store:expr, $selector:expr, $units:expr) => {
            if options.should_process($wrf, Some($store), WrfProductGroup::Core) {
                push_canonical(
                    &mut fields,
                    file,
                    timeidx,
                    &grid,
                    projection.clone(),
                    $wrf,
                    $store,
                    $selector,
                    $units,
                );
            }
        };
    }

    push_core!(
        "terrain",
        "orography",
        core_field_selector("orography"),
        None
    );
    push_core!(
        "t2",
        "temperature_2m",
        core_field_selector("temperature_2m"),
        Some("K")
    );
    push_core!(
        "dp2m",
        "dewpoint_2m",
        core_field_selector("dewpoint_2m"),
        Some("K")
    );
    push_core!(
        "rh2m",
        "relative_humidity_2m",
        core_field_selector("relative_humidity_2m"),
        Some("%")
    );
    // WRF's raw U10/V10 components are grid-relative. Ask wrf-core for
    // `uvmet10` once, then split its [u_earth, v_earth] planes so the canonical
    // store names and sounding wind barbs are genuinely earth-relative. Keep
    // U10/V10 as the option-filter keys so saved `only`/`skip` profiles retain
    // their existing meaning.
    let want_u10 = options.should_process("U10", Some("u_10m"), WrfProductGroup::Core);
    let want_v10 = options.should_process("V10", Some("v_10m"), WrfProductGroup::Core);
    if want_u10 || want_v10 {
        match compute_var(file, "uvmet10", timeidx, Some("m/s"))
            .and_then(|output| split_uvmet10(output, shape.len()))
        {
            Ok((u_earth, v_earth, units)) => {
                if want_u10 {
                    push_canonical_values(
                        &mut fields,
                        &grid,
                        projection.clone(),
                        "u_10m",
                        core_field_selector("u_10m"),
                        &units,
                        u_earth,
                    );
                }
                if want_v10 {
                    push_canonical_values(
                        &mut fields,
                        &grid,
                        projection.clone(),
                        "v_10m",
                        core_field_selector("v_10m"),
                        &units,
                        v_earth,
                    );
                }
            }
            Err(err) => {
                if want_u10 {
                    fields
                        .notes
                        .push(format!("u_10m unavailable: earth-rotated uvmet10: {err}"));
                }
                if want_v10 {
                    fields
                        .notes
                        .push(format!("v_10m unavailable: earth-rotated uvmet10: {err}"));
                }
            }
        }
    }
    push_core!(
        "wspd10",
        "wind_speed_10m",
        core_field_selector("wind_speed_10m"),
        Some("m/s")
    );
    push_core!(
        "slp",
        "mslp",
        core_field_selector("mslp"),
        Some("Pa")
    );
    // Surface pressure (Pa) — required by the skew-T column builder. WRF PSFC
    // is a raw field (no `getvar` diagnostic), so push it explicitly with a
    // forced "Pa" unit rather than through `push_core!`.
    if options.should_process("PSFC", Some("surface_pressure"), WrfProductGroup::Core) {
        match compute_var(file, "PSFC", timeidx, Some("Pa")) {
            Ok(output) => match single_plane(output, shape.len()) {
                Ok((values, _units)) => push_canonical_values(
                    &mut fields,
                    &grid,
                    projection.clone(),
                    "surface_pressure",
                    core_field_selector("surface_pressure"),
                    "Pa",
                    values,
                ),
                Err(err) => fields.notes.push(format!("PSFC skipped: {err}")),
            },
            Err(err) => fields.notes.push(format!("PSFC unavailable: {err}")),
        }
    }
    push_core!(
        "pw",
        "pwat",
        core_field_selector("pwat"),
        None
    );
    push_reflectivity_products(
        &mut fields,
        file,
        timeidx,
        &grid,
        projection.clone(),
        options,
        progress,
    );
    push_core!(
        "UP_HELI_MAX",
        "updraft_helicity_2to5km",
        core_field_selector("updraft_helicity_2to5km"),
        Some("m2/s2")
    );
    crate::wrf_column_planes::push_column_planes(
        &mut fields,
        file,
        timeidx,
        &grid,
        projection.clone(),
        options,
        progress,
    );

    if options.should_process("apcp", Some("apcp"), WrfProductGroup::Core) {
        if let Some(values) = total_precip(file, timeidx, shape.len()) {
            push_canonical_values(
                &mut fields,
                &grid,
                projection.clone(),
                "apcp",
                core_field_selector("apcp"),
                "kg/m^2",
                values,
            );
        }
    }

    let total_twod = VARS
        .iter()
        .filter(|def| def.dim == VarDim::TwoD && !excluded_from_full_twod_pass(def.name))
        .count();
    let mut diagnostic_index = 0usize;
    for def in VARS {
        if def.dim != VarDim::TwoD || excluded_from_full_twod_pass(def.name) {
            continue;
        }
        let store_name = derived_name(def.name, None);
        let group = if is_heavy_wrf_diagnostic(&store_name) || is_heavy_wrf_diagnostic(def.name) {
            WrfProductGroup::Heavy
        } else {
            WrfProductGroup::Diagnostic
        };
        if !options.needs_diagnostic_grid(def.name, &store_name)
            || !options.should_process(def.name, Some(&store_name), group) {
            continue;
        }
        diagnostic_index += 1;
        progress(format!(
            "Computing WRF diagnostic {diagnostic_index}/{total_twod}: {}",
            def.name
        ));
        match compute_var(file, def.name, timeidx, None) {
            Ok(output) => push_derived_output(&mut fields, def.name, output, shape.len()),
            Err(err) => fields
                .notes
                .push(format!("{} unavailable: {err}", def.name)),
        }
    }

    for raw in RAW_EXTRA_CATALOG {
        let store_name = derived_name(raw, None);
        if !options.should_process(raw, Some(&store_name), WrfProductGroup::Raw) {
            continue;
        }
        if let Ok(mut output) = compute_var(file, raw, timeidx, None) {
            // A raw extra is the file's own variable, read without any
            // conversion, so the file's units attribute is what its values
            // are in. wrf-core's fallback table is only a guess at WRF's
            // conventions and is kept only where the file names no unit.
            if let Some(units) = stored_plane_index
                .and_then(|index| crate::local_import::variable_units(index, raw))
                .filter(|units| !units.trim().is_empty())
            {
                output.units = units;
            }
            push_derived_output(&mut fields, raw, output, shape.len());
        }
    }

    push_stored_planes(
        &mut fields,
        file,
        stored_plane_index,
        timeidx,
        shape,
        options,
        progress,
    );

    // The cloud-cover chart recipes resolve canonical entire-atmosphere
    // selectors.  wrf-core's `cloudfrac` diagnostic (already computed and
    // split into the wrf_cloudfrac_* browse planes above, in %) is the
    // same low/mid/high layer-maximum cloud fraction; publish the three
    // planes under their canonical selectors so the recipes see them.
    // WRF carries no total-cloud field, so `cloud_cover` (total) stays
    // accurately unstored.
    for (_, derived_plane, canonical_field) in DIAGNOSTIC_CHART_PLANES {
        let Some(source) = fields
            .derived
            .iter()
            .find(|field| field.name == derived_plane)
        else {
            continue;
        };
        if source.units != "%" {
            fields.notes.push(format!(
                "{derived_plane} not published as a canonical cloud-cover \
                 plane: units '{}' are not '%'",
                source.units
            ));
            continue;
        }
        let selector = FieldSelector::entire_atmosphere(canonical_field);
        let store_name = selector.key();
        let values = source.values.clone();
        if options.should_process(&store_name, Some(&store_name), WrfProductGroup::Diagnostic) {
            push_canonical_values(
                &mut fields,
                &grid,
                projection.clone(),
                &store_name,
                selector,
                "%",
                values,
            );
        }
    }

    // Isobaric sounding volumes (temperature_iso/dewpoint_iso/u_iso/v_iso/
    // height_iso) so imported WRF runs produce skew-T soundings like the
    // downloaded models. Failure here never fails the hour — the 2D fields
    // still write; only the sounding is unavailable.
    //
    // NOTE: wrf-core's per-timestep intermediate cache must stay WARM into
    // this block. The volume build re-getvars pressure/temp/td/height/uvmet;
    // with the cache populated by the diagnostics above those are cheap
    // copies (measured peak 8.85 GB on the 800x800x79 Enderlin grid).
    // Clearing the cache first was measured to more than DOUBLE the peak
    // (18.3 GB): every read then recomputes its whole dependency chain
    // (staggered reads, destaggering, theta->T, …) with multi-hundred-MB
    // transients stacking on the re-growing cache. `build_iso_volumes` itself
    // clears the cache right after its LAST getvar (the hour's last), so the
    // interpolation loop and the store write below run without the ~5 GB of
    // dead intermediates.
    if options.viewer_2d {
        match isolate_panics("selected isobaric chart planes", || {
            crate::wrf_chart_planes::build_chart_planes(
                file,
                timeidx,
                shape.len(),
                &options.chart_selectors,
                progress,
            )
        }) {
            Ok((planes, height_note)) => {
                push_isobaric_recipe_planes(
                    &mut fields,
                    &grid,
                    projection.clone(),
                    planes.iter(),
                    options,
                );
                // The heights took the named mass-level fallback: the
                // run's notes say which heights its charts are.
                fields.notes.extend(height_note);
            }
            Err(error) => fields.notes.push(format!(
                "Selected isobaric chart planes unavailable: {error}"
            )),
        }
    } else if options.core_fields {
        // Isolated for the same reason as `compute_var`: the volume builder's
        // `getvar` reads must degrade to a note, not kill the hour.
        // Preflight outside the isolated builder as well as inside it so this
        // process layer explicitly chooses the 2-D-only degradation before the
        // first volume getvar. The builder repeats the check to protect every
        // other caller.
        let volumes_result = match preflight_iso_volume_shape(file.nz, shape.len()) {
            Ok(_) => isolate_panics("isobaric volumes", || {
                build_iso_volumes(file, timeidx, shape.len(), progress)
            }),
            Err(err) => Err(err),
        };
        match volumes_result {
            Ok((volumes, chart_volumes, surface, height_note)) => {
                fields.notes.extend(height_note);
                // The production isobaric chart recipes (500mb heights,
                // 700/850mb temperature/dewpoint, upper-level winds, RH,
                // absolute vorticity, ...) resolve per-level canonical
                // selector planes.  The sounding volumes' recipe levels are
                // published as selector planes (same interpolated bytes
                // serving both the skew-T columns and the chart lane), and
                // the chart-only rh/avo level planes ride alongside without
                // ever being persisted as store volumes.
                push_isobaric_recipe_planes(
                    &mut fields,
                    &grid,
                    projection.clone(),
                    volumes.iter().chain(chart_volumes.iter()),
                    options,
                );
                fields.volumes = volumes;
                // Split wrf3d files (CONUS404 / GDEX CONUS-II) omit PSFC (and
                // sometimes T2/Td2/winds). Preserve their lowest-model-level
                // substitutes under explicit `approx_*` names so they remain
                // usable for a sounding without pretending to be true 2 m/10 m
                // observations or diagnostics.
                fill_missing_surface(&mut fields, &grid, projection.clone(), surface, options);
            }
            Err(err) => {
                let retained_2d_products = fields.canonical.len() + fields.derived.len();
                let note = volume_omission_note(retained_2d_products, &err);
                progress(note.clone());
                fields.notes.push(note);
            }
        }
    }

    // Release wrf-core's per-timestep intermediate cache before the caller
    // writes the hour to the store. `getvar` memoizes every 3-D f64
    // intermediate (pressure, theta, temperature, geopotential, heights,
    // QVAPOR, destaggered winds, …) inside `WrfFile` and only evicts when the
    // *timestep changes* (`prepare_cache_for_time`); its `clear_cache` is
    // never invoked upstream despite its doc comment. On a 250 m Enderlin
    // grid (800x800x79 ≈ 50.5 M cells, ~400 MB per cached field) that cache
    // holds ~5 GB. Dropping it here — after the hour's last `getvar`, before
    // `write_hour_from_fields_with_derived` — releases that memory for the
    // write phase and beyond at zero recompute cost (measured: working set
    // fell 8.8 GB -> 1.3 GB at this point instead of riding the write).
    // Usually a no-op now that `build_iso_volumes` clears after its last
    // getvar, but still essential when `core_fields` is off (no volume
    // build) or the volume build failed partway. catch_unwind: if a caught
    // diagnostic panic above poisoned the cache mutex, clearing would
    // re-panic; a stuck cache must not fail the hour.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| file.clear_cache()));

    // wrf-core has no ml/mu parcel ECAPE diagnostic (`wrf_product_slug`
    // leaves those variants unmapped) and no ECAPE / derived-CAPE ratio at
    // all, so with `--heavy` on those grids are computed here through the
    // SAME shared recipe lane the GRIB heavy ingest calls: this hour's own
    // surface planes and `*_iso` volumes are assembled into the products-side
    // input pair and handed to `compute_store_heavy_grids`. Runs after the
    // wrf-core cache is released, so the f64 input volumes do not stack on
    // top of it.
    if options.heavy_ecape {
        push_shared_heavy_recipe_grids(&mut fields, options, progress);
    }

    Ok(fields)
}

/// Add every heavy (ECAPE-class) recipe grid the shared lane realizes and
/// this hour does not already carry under that slug, so wrf-core's own
/// authoritative diagnostics keep their names and only the gaps are filled.
/// A missing input degrades to a note: the 2-D fields and the soundings
/// still write.
fn push_shared_heavy_recipe_grids(
    fields: &mut WrfHourFields,
    options: &WrfProcessOptions,
    progress: &mut impl FnMut(String),
) {
    if fields.volumes.is_empty() {
        fields.notes.push(
            "shared heavy (ECAPE) recipe grids unavailable: this hour has no isobaric \
             sounding volumes for the parcel columns"
                .to_string(),
        );
        return;
    }
    progress(
        "computing the shared heavy (ECAPE) recipe grids wrf-core carries no diagnostic for"
            .to_string(),
    );
    let grid = fields.grid.clone();
    let projection = fields.projection.clone();
    let outcome = {
        let canonical = &fields.canonical;
        let volumes = &fields.volumes;
        isolate_panics("shared heavy recipe grids", || {
            crate::local_import::compute_wrf_heavy_store_grids(
                &grid,
                projection.clone(),
                canonical,
                volumes,
            )
        })
    };
    let (heavy, basis) = match outcome {
        Ok(value) => value,
        Err(err) => {
            fields
                .notes
                .push(format!("shared heavy (ECAPE) recipe grids unavailable: {err}"));
            return;
        }
    };
    if let Some(note) = basis.note() {
        fields.notes.push(note);
    }
    let mut added = Vec::new();
    for grid_values in heavy.grids {
        let slug = grid_values.slug;
        if !options.should_process(slug, Some(slug), WrfProductGroup::Heavy) {
            continue;
        }
        let already = fields
            .canonical
            .iter()
            .any(|(name, _)| name == slug)
            || fields.derived.iter().any(|field| field.name == slug);
        if already {
            continue;
        }
        fields.derived.push(OwnedDerivedField {
            name: slug.to_string(),
            units: grid_values.units,
            values: grid_values
                .values
                .into_iter()
                .map(|value| value as f32)
                .collect(),
        });
        added.push(slug);
    }
    if !added.is_empty() {
        fields.notes.push(format!(
            "shared heavy (ECAPE) recipe grids added from {} isobaric levels: {}",
            basis.levels,
            added.join(", ")
        ));
    }
    for skip in heavy.skipped {
        fields.notes.push(format!(
            "heavy recipe '{}' skipped: {}",
            skip.slug, skip.reason
        ));
    }
    if heavy.ecape_failure_count > 0 {
        fields.notes.push(format!(
            "shared heavy (ECAPE) recipe grids: {} column(s) whose parcel ascent failed carry NaN",
            heavy.ecape_failure_count
        ));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReflectivityVolumeSource {
    NativeRefl10cm,
    GenericDbzFallback,
}

/// Keep source selection separate from decoding so the fail-closed rule is
/// explicit and directly testable: a present-but-broken native field is an
/// error, never permission to substitute a different reflectivity formula.
fn select_reflectivity_volume<T>(
    native_present: bool,
    read_native: impl FnOnce() -> Result<T, String>,
    compute_fallback: impl FnOnce() -> Result<T, String>,
) -> Result<(T, ReflectivityVolumeSource), String> {
    if native_present {
        read_native().map(|volume| (volume, ReflectivityVolumeSource::NativeRefl10cm))
    } else {
        compute_fallback().map(|volume| (volume, ReflectivityVolumeSource::GenericDbzFallback))
    }
}

fn checked_volume_cells(nz: usize, ny: usize, nx: usize) -> Result<usize, String> {
    checked_horizontal_cells(ny, nx)?
        .checked_mul(nz)
        .ok_or_else(|| {
            format!("3-D dimensions [{nz}, {ny}, {nx}] overflow the platform address space")
        })
}

fn validate_native_reflectivity_volume(
    values: Vec<f64>,
    actual_shape: &[usize],
    nz: usize,
    ny: usize,
    nx: usize,
) -> Result<Vec<f64>, String> {
    let expected = checked_volume_cells(nz, ny, nx)?;
    if actual_shape != [nz, ny, nx] {
        return Err(format!(
            "native REFL_10CM shape/order mismatch: expected [{nz},{ny},{nx}], got {actual_shape:?}"
        ));
    }
    if values.len() != expected {
        return Err(format!(
            "native REFL_10CM shape/length mismatch: expected [{nz},{ny},{nx}] ({expected} values), got {} values",
            values.len()
        ));
    }
    Ok(values)
}

fn take_three_dimensional_output(
    name: &str,
    output: VarOutput,
    nz: usize,
    ny: usize,
    nx: usize,
) -> Result<Vec<f64>, String> {
    let expected = checked_volume_cells(nz, ny, nx)?;
    let [actual_nz, actual_ny, actual_nx] = output.shape.as_slice() else {
        return Err(format!(
            "{name} expected shape [{nz},{ny},{nx}], got {:?}",
            output.shape
        ));
    };
    if (*actual_nz, *actual_ny, *actual_nx) != (nz, ny, nx) || output.data.len() != expected {
        return Err(format!(
            "{name} expected shape [{nz},{ny},{nx}] with {expected} values, got {:?} with {} values",
            output.shape,
            output.data.len()
        ));
    }
    Ok(output.data)
}

fn read_reflectivity_volume(
    file: &WrfFile,
    timeidx: usize,
) -> Result<(Vec<f64>, ReflectivityVolumeSource), String> {
    let native_present = file.has_var("REFL_10CM");
    select_reflectivity_volume(
        native_present,
        || {
            isolate_panics("native REFL_10CM", || {
                let shape = file
                    .var_shape_no_time("REFL_10CM")
                    .map_err(|err| err.to_string())?;
                let values = file
                    .read_var("REFL_10CM", timeidx)
                    .map_err(|err| err.to_string())?;
                validate_native_reflectivity_volume(values, &shape, file.nz, file.ny, file.nx)
            })
        },
        || {
            compute_var(file, "dbz", timeidx, Some("dBZ")).and_then(|output| {
                take_three_dimensional_output("generic dbz", output, file.nz, file.ny, file.nx)
            })
        },
    )
}

fn valid_reflectivity_dbz(value: f64) -> bool {
    // WRF and third-party NetCDF writers use several large-magnitude fill
    // conventions (-999, -8888, -9999, +9999). A physical range rejects
    // those sentinels before either a vertical maximum or linear-Z
    // interpolation can turn them into a false echo.
    value.is_finite() && (MIN_VALID_REFLECTIVITY_DBZ..=MAX_VALID_REFLECTIVITY_DBZ).contains(&value)
}

fn composite_reflectivity_from_dbz(
    dbz_3d: &[f64],
    nz: usize,
    cells: usize,
) -> Result<Vec<f64>, String> {
    let expected = nz.checked_mul(cells).ok_or_else(|| {
        "reflectivity volume length overflows the platform address space".to_string()
    })?;
    if dbz_3d.len() != expected {
        return Err(format!(
            "reflectivity volume requires {expected} values for {nz} levels and {cells} columns, got {}",
            dbz_3d.len()
        ));
    }

    let mut composite = vec![f64::NAN; cells];
    for k in 0..nz {
        let offset = k * cells;
        for ij in 0..cells {
            let value = dbz_3d[offset + ij];
            if valid_reflectivity_dbz(value)
                && (!composite[ij].is_finite() || value > composite[ij])
            {
                composite[ij] = value;
            }
        }
    }
    Ok(composite)
}

fn interpolate_dbz_in_linear_z(lower_dbz: f64, upper_dbz: f64, fraction: f64) -> f64 {
    let lower_z = 10.0_f64.powf(lower_dbz / 10.0);
    let upper_z = 10.0_f64.powf(upper_dbz / 10.0);
    let interpolated_z = lower_z + fraction * (upper_z - lower_z);
    if interpolated_z.is_finite() && interpolated_z > 0.0 {
        10.0 * interpolated_z.log10()
    } else {
        f64::NAN
    }
}

/// Interpolate each model column to a geometric AGL height. Reflectivity is
/// logarithmic, so interpolation occurs in linear Z and converts back to dBZ.
/// Targets outside the valid model-height bracket are deliberately missing;
/// no vertical extrapolation is scientifically implied.
fn reflectivity_at_height_agl(
    dbz_3d: &[f64],
    height_agl_3d: &[f64],
    nz: usize,
    cells: usize,
    target_m: f64,
) -> Result<Vec<f64>, String> {
    let expected = nz
        .checked_mul(cells)
        .ok_or_else(|| "reflectivity interpolation volume length overflow".to_string())?;
    if dbz_3d.len() != expected || height_agl_3d.len() != expected {
        return Err(format!(
            "reflectivity interpolation requires {expected} values per volume, got reflectivity {} and height {}",
            dbz_3d.len(),
            height_agl_3d.len()
        ));
    }
    if !target_m.is_finite() {
        return Err("reflectivity interpolation target must be finite".to_string());
    }

    let mut result = vec![f64::NAN; cells];
    for ij in 0..cells {
        // An exact model-level hit must not be perturbed by interpolation.
        for k in 0..nz {
            let index = k * cells + ij;
            if height_agl_3d[index] == target_m && valid_reflectivity_dbz(dbz_3d[index]) {
                result[ij] = dbz_3d[index];
                break;
            }
        }
        if result[ij].is_finite() {
            continue;
        }

        for k in 0..nz.saturating_sub(1) {
            let lower = k * cells + ij;
            let upper = (k + 1) * cells + ij;
            let h0 = height_agl_3d[lower];
            let h1 = height_agl_3d[upper];
            let dbz0 = dbz_3d[lower];
            let dbz1 = dbz_3d[upper];
            if !h0.is_finite()
                || !h1.is_finite()
                || !valid_reflectivity_dbz(dbz0)
                || !valid_reflectivity_dbz(dbz1)
            {
                continue;
            }
            let bracketed = (h0 < target_m && target_m < h1) || (h1 < target_m && target_m < h0);
            if !bracketed {
                continue;
            }
            let fraction = (target_m - h0) / (h1 - h0);
            result[ij] = interpolate_dbz_in_linear_z(dbz0, dbz1, fraction);
            break;
        }
    }
    Ok(result)
}

fn composite_reflectivity_selector() -> FieldSelector {
    FieldSelector::entire_atmosphere(CanonicalField::CompositeReflectivity)
}

fn reflectivity_1km_selector() -> FieldSelector {
    FieldSelector::height_agl(CanonicalField::RadarReflectivity, 1000)
}

#[allow(clippy::too_many_arguments)]
fn push_reflectivity_products(
    fields: &mut WrfHourFields,
    file: &WrfFile,
    timeidx: usize,
    grid: &LatLonGrid,
    projection: Option<GridProjection>,
    options: &WrfProcessOptions,
    progress: &mut impl FnMut(String),
) {
    let want_composite = options.should_process(
        COMPOSITE_REFLECTIVITY_FILTER,
        Some(COMPOSITE_REFLECTIVITY_STORE),
        WrfProductGroup::Core,
    );
    let want_1km = options.should_process(
        REFLECTIVITY_1KM_FILTER,
        Some(REFLECTIVITY_1KM_STORE),
        WrfProductGroup::Core,
    );
    if !want_composite && !want_1km {
        return;
    }

    let native_present = file.has_var("REFL_10CM");
    progress(if native_present {
        "Reading native WRF REFL_10CM reflectivity volume".to_string()
    } else {
        "REFL_10CM absent; computing generic hydrometeor dbz fallback".to_string()
    });
    let (dbz_3d, source) = match read_reflectivity_volume(file, timeidx) {
        Ok(result) => result,
        Err(err) => {
            let detail = if native_present {
                format!(
                    "native REFL_10CM failed; generic dbz fallback was not used because the native variable is present: {err}"
                )
            } else {
                format!("generic dbz fallback failed after native REFL_10CM was absent: {err}")
            };
            if want_composite {
                fields.notes.push(format!(
                    "{COMPOSITE_REFLECTIVITY_STORE} unavailable: {detail}"
                ));
            }
            if want_1km {
                fields
                    .notes
                    .push(format!("{REFLECTIVITY_1KM_STORE} unavailable: {detail}"));
            }
            return;
        }
    };
    if source == ReflectivityVolumeSource::GenericDbzFallback {
        fields.notes.push(
            "REFL_10CM absent; reflectivity products use the generic hydrometeor dbz fallback"
                .to_string(),
        );
    }

    if want_composite {
        match composite_reflectivity_from_dbz(&dbz_3d, file.nz, grid.shape.len()) {
            Ok(values) => push_canonical_values(
                fields,
                grid,
                projection.clone(),
                COMPOSITE_REFLECTIVITY_STORE,
                composite_reflectivity_selector(),
                "dBZ",
                clean_values(&values),
            ),
            Err(err) => fields
                .notes
                .push(format!("{COMPOSITE_REFLECTIVITY_STORE} unavailable: {err}")),
        }
    }

    if want_1km {
        let height_result =
            compute_var(file, "height_agl", timeidx, Some("m")).and_then(|output| {
                take_three_dimensional_output("height_agl", output, file.nz, file.ny, file.nx)
            });
        match height_result.and_then(|height_agl| {
            reflectivity_at_height_agl(&dbz_3d, &height_agl, file.nz, grid.shape.len(), 1_000.0)
        }) {
            Ok(values) => push_canonical_values(
                fields,
                grid,
                projection,
                REFLECTIVITY_1KM_STORE,
                reflectivity_1km_selector(),
                "dBZ",
                clean_values(&values),
            ),
            Err(err) => fields
                .notes
                .push(format!("{REFLECTIVITY_1KM_STORE} unavailable: {err}")),
        }
    }
}

/// Every stored `(Time, south_north, west_east)` plane the file carries
/// that this import has not already produced under some other name.
///
/// The concrete breakage this prevents: a variable a user added to their
/// own WRF Registry — `MSLP_ANOM`, a bespoke tracer, an in-house
/// diagnostic — is in the wrfout but in neither of this route's two fixed
/// catalogs, so it never reached the store and the generic
/// `--products var:wrf_mslp_anom` answered
/// `stored 2-D variable "wrf_mslp_anom" does not exist`. The remedy is
/// metadata-level on purpose: the plane is carried through under the same
/// `wrf_<sanitized>` name the light import already gives it, so nothing
/// about a new user variable needs new product code.
///
/// Two rules keep it additive. A name this import already produced is
/// skipped, so the canonical, unit-checked, earth-relative fields win over
/// the raw plane behind them (`T2` stays `temperature_2m` AND keeps its
/// existing `wrf_t2` browse plane from the diagnostic registry). And the
/// planes answer the same `--only`/`--skip` token grammar under the Raw
/// group as the rest of the import.
fn push_stored_planes(
    fields: &mut WrfHourFields,
    file: &WrfFile,
    index: Option<&netcrust::File>,
    timeidx: usize,
    shape: GridShape,
    options: &WrfProcessOptions,
    progress: &mut impl FnMut(String),
) {
    if !options.stored_planes {
        return;
    }
    let Some(index) = index else {
        // Only reachable when the caller could not build the metadata
        // index; it already reported why, and the science suite above is
        // unaffected, so this is a note-free no-op rather than a refusal.
        return;
    };
    let mut produced = fields
        .canonical
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<std::collections::HashSet<_>>();
    produced.extend(fields.derived.iter().map(|field| field.name.clone()));

    let source = crate::local_import::PlaneSource::new(index, Some(file), timeidx);
    let planes = crate::local_import::read_raw_wrf_mass_grid_fields_where(
        &source,
        shape.nx,
        shape.ny,
        progress,
        &mut |wrf_name, store_name| {
            !produced.contains(store_name)
                && options.should_process(wrf_name, Some(store_name), WrfProductGroup::Raw)
        },
    );
    match planes {
        Ok(planes) => {
            if !planes.is_empty() {
                progress(format!(
                    "carried {} stored 2-D plane(s) no product catalog names: {}",
                    planes.len(),
                    planes
                        .iter()
                        .map(|field| field.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            for plane in planes {
                if plane.values.len() != shape.len() {
                    fields.notes.push(format!(
                        "{} skipped: carries {} values; expected {}",
                        plane.name,
                        plane.values.len(),
                        shape.len()
                    ));
                    continue;
                }
                fields.derived.push(OwnedDerivedField {
                    name: plane.name,
                    units: plane.units,
                    values: plane.values,
                });
            }
        }
        // A failed sweep costs the user's own planes, not the hour: the
        // science suite above is already built and independently valid.
        Err(err) => fields.notes.push(format!(
            "stored 2-D planes unavailable: {err}; a variable this file \
             carries but no product catalog names will not render through \
             var:<name>"
        )),
    }
}

/// `maxdbz` is intentionally omitted here: the dedicated reflectivity path
/// derives the canonical composite from the same preferred 3-D volume used by
/// the 1-km product. Running the registry's generic `maxdbz` would both repeat
/// expensive work and publish a contradictory `wrf_maxdbz` field.
fn excluded_from_full_twod_pass(name: &str) -> bool {
    matches!(
        name,
        "lat" | "lon" | "cape2d" | "cape2d_wrfpython" | "maxdbz"
    )
}

/// Add lowest-model-level [`SurfaceFallback`] values for skew-T use when the
/// file did not provide the real surface product. These are approximations,
/// not true PSFC/T2/Td2/U10/V10 values, so they stay under explicit
/// `approx_*` store names and surface selectors; an exact counterpart always
/// wins.
fn fill_missing_surface(
    fields: &mut WrfHourFields,
    grid: &LatLonGrid,
    projection: Option<GridProjection>,
    surface: SurfaceFallback,
    options: &WrfProcessOptions,
) {
    let entries: [(&str, &str, &str, FieldSelector, &str, Vec<f32>); 5] = [
        (
            "PSFC",
            "surface_pressure",
            "approx_surface_pressure",
            FieldSelector::surface(CanonicalField::Pressure),
            "Pa",
            surface.surface_pressure_pa,
        ),
        (
            "t2",
            "temperature_2m",
            "approx_temperature_2m",
            FieldSelector::surface(CanonicalField::Temperature),
            "K",
            surface.temperature_2m_k,
        ),
        (
            "dp2m",
            "dewpoint_2m",
            "approx_dewpoint_2m",
            FieldSelector::surface(CanonicalField::Dewpoint),
            "K",
            surface.dewpoint_2m_k,
        ),
        (
            "U10",
            "u_10m",
            "approx_u_10m",
            FieldSelector::surface(CanonicalField::UWind),
            "m/s",
            surface.u_10m,
        ),
        (
            "V10",
            "v_10m",
            "approx_v_10m",
            FieldSelector::surface(CanonicalField::VWind),
            "m/s",
            surface.v_10m,
        ),
    ];
    for (source_name, exact_name, approx_name, selector, units, values) in entries {
        if !options.should_process(source_name, Some(approx_name), WrfProductGroup::Core) {
            continue;
        }
        let exact_exists = fields
            .canonical
            .iter()
            .any(|(existing, _)| existing == exact_name);
        if !exact_exists {
            let previous_len = fields.canonical.len();
            push_canonical_values(
                fields,
                grid,
                projection.clone(),
                approx_name,
                selector,
                units,
                values,
            );
            if fields.canonical.len() > previous_len {
                fields.notes.push(format!(
                    "{approx_name} synthesized from the lowest WRF model level because {exact_name} was unavailable"
                ));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_canonical(
    fields: &mut WrfHourFields,
    file: &WrfFile,
    timeidx: usize,
    grid: &LatLonGrid,
    projection: Option<GridProjection>,
    wrf_name: &str,
    store_name: &str,
    selector: FieldSelector,
    units: Option<&str>,
) {
    match compute_var(file, wrf_name, timeidx, units) {
        Ok(output) => match single_plane(output, grid.shape.len()) {
            Ok((values, actual_units)) => push_canonical_values(
                fields,
                grid,
                projection,
                store_name,
                selector,
                &actual_units,
                values,
            ),
            Err(err) => fields.notes.push(format!("{wrf_name} skipped: {err}")),
        },
        Err(err) => fields.notes.push(format!("{wrf_name} unavailable: {err}")),
    }
}

/// The isobaric levels the production direct chart recipes plot. A subset
/// of the canonical 37-level ladder the volumes interpolate, so publishing
/// them is a copy of existing certified planes, never a second
/// interpolation.
const ISOBARIC_RECIPE_LEVELS_HPA: [u16; 6] = [200, 250, 300, 500, 700, 850];

/// Publish the chart-recipe levels of the interpolated `*_iso` volumes as
/// canonical selector planes (`geopotential_height_500hpa`, ...).  Values
/// are byte-identical to the corresponding volume level; cells outside a
/// column's model pressure range stay NaN exactly as the sounding path
/// leaves them.  Store names are the selector keys, and each plane honors
/// the same only/skip filters as every other product.
fn push_isobaric_recipe_planes<'a>(
    fields: &mut WrfHourFields,
    grid: &LatLonGrid,
    projection: Option<GridProjection>,
    volumes: impl Iterator<Item = &'a crate::wrf_volumes::IsoVolume>,
    options: &WrfProcessOptions,
) {
    let canonical_field = |volume_name: &str| -> Option<CanonicalField> {
        ISOBARIC_RECIPE_VOLUMES
            .iter()
            .find(|(name, _)| *name == volume_name)
            .map(|(_, field)| *field)
    };
    for volume in volumes {
        let Some(field) = canonical_field(&volume.name) else {
            continue;
        };
        for (level_hpa, plane) in &volume.levels {
            if !ISOBARIC_RECIPE_LEVELS_HPA.contains(level_hpa) {
                continue;
            }
            let selector = FieldSelector::isobaric(field, *level_hpa);
            let store_name = selector.key();
            if !options.should_process(&store_name, Some(&store_name), WrfProductGroup::Core) {
                continue;
            }
            push_canonical_values(
                fields,
                grid,
                projection.clone(),
                &store_name,
                selector,
                &volume.units,
                plane.clone(),
            );
        }
    }
}

pub(crate) fn push_canonical_values(
    fields: &mut WrfHourFields,
    grid: &LatLonGrid,
    projection: Option<GridProjection>,
    store_name: &str,
    selector: FieldSelector,
    units: &str,
    values: Vec<f32>,
) {
    match SelectedField2D::new(selector, units, grid.clone(), values) {
        Ok(field) => {
            let field = if let Some(projection) = projection {
                field.with_projection(projection)
            } else {
                field
            };
            fields.canonical.push((store_name.to_string(), field));
        }
        Err(err) => fields
            .notes
            .push(format!("{store_name} skipped: invalid field: {err}")),
    }
}

fn push_derived_output(
    fields: &mut WrfHourFields,
    wrf_name: &str,
    output: VarOutput,
    cells: usize,
) {
    let units = output.units.clone();
    match output.shape.as_slice() {
        [ny, nx] => {
            let plane_cells = match checked_horizontal_cells(*ny, *nx) {
                Ok(value) => value,
                Err(err) => {
                    fields.notes.push(format!("{wrf_name} skipped: {err}"));
                    return;
                }
            };
            if plane_cells != cells || output.data.len() != cells {
                fields.notes.push(format!(
                    "{wrf_name} skipped: shape {:?} describes {plane_cells} cells and carries {} values; expected {cells}",
                    output.shape,
                    output.data.len()
                ));
                return;
            }
            fields.derived.push(OwnedDerivedField {
                name: derived_name(wrf_name, None),
                units,
                values: clean_values(&output.data),
            });
        }
        [count, ny, nx] => {
            let plane_cells = match checked_horizontal_cells(*ny, *nx) {
                Ok(value) => value,
                Err(err) => {
                    fields.notes.push(format!("{wrf_name} skipped: {err}"));
                    return;
                }
            };
            if plane_cells != cells {
                fields.notes.push(format!(
                    "{wrf_name} skipped: shape {:?} describes {plane_cells} cells per plane; expected {cells}",
                    output.shape
                ));
                return;
            }
            let expected_values = match count.checked_mul(cells) {
                Some(value) => value,
                None => {
                    fields.notes.push(format!(
                        "{wrf_name} skipped: component count {count} times {cells} cells overflows the platform address space"
                    ));
                    return;
                }
            };
            if output.data.len() != expected_values {
                fields.notes.push(format!(
                    "{wrf_name} skipped: shape {:?} requires {expected_values} values, got {}",
                    output.shape,
                    output.data.len()
                ));
                return;
            }
            for index in 0..*count {
                let Some(start) = index.checked_mul(cells) else {
                    fields.notes.push(format!(
                        "{wrf_name} skipped split {index}: component offset overflow"
                    ));
                    return;
                };
                let Some(end) = start.checked_add(cells) else {
                    fields.notes.push(format!(
                        "{wrf_name} skipped split {index}: component end offset overflow"
                    ));
                    return;
                };
                let Some(values) = output.data.get(start..end) else {
                    fields.notes.push(format!(
                        "{wrf_name} skipped split {index}: offsets {start}..{end} exceeded data length {}",
                        output.data.len()
                    ));
                    return;
                };
                fields.derived.push(OwnedDerivedField {
                    name: derived_name(wrf_name, Some(index)),
                    units: units.clone(),
                    values: clean_values(values),
                });
            }
        }
        other => fields.notes.push(format!(
            "{wrf_name} skipped: unsupported shape {:?} for 2D store",
            other
        )),
    }
}

fn single_plane(output: VarOutput, cells: usize) -> Result<(Vec<f32>, String), String> {
    let (ny, nx) = match output.shape.as_slice() {
        [ny, nx] | [1, ny, nx] => (*ny, *nx),
        other => return Err(format!("expected [ny,nx], got {other:?}")),
    };
    let plane_cells = checked_horizontal_cells(ny, nx)?;
    if plane_cells != cells {
        return Err(format!(
            "shape {:?} describes {plane_cells} cells, expected {cells}",
            output.shape
        ));
    }
    if output.data.len() != cells {
        return Err(format!(
            "shape {:?} requires {cells} values, got {}",
            output.shape,
            output.data.len()
        ));
    }
    Ok((clean_values(&output.data), output.units))
}

/// Split wrf-core's earth-rotated `uvmet10` result into canonical 2-D planes.
/// The diagnostic contract is `[2, ny, nx]`; validating both the advertised
/// shape and backing length keeps malformed output from being mistaken for a
/// pair of surface components.
fn split_uvmet10(output: VarOutput, cells: usize) -> Result<(Vec<f32>, Vec<f32>, String), String> {
    let [components, ny, nx] = output.shape.as_slice() else {
        return Err(format!(
            "expected uvmet10 shape [2,ny,nx], got {:?}",
            output.shape
        ));
    };
    if *components != 2 {
        return Err(format!(
            "expected two uvmet10 components, got shape {:?}",
            output.shape
        ));
    }
    let plane_cells = checked_horizontal_cells(*ny, *nx)?;
    if plane_cells != cells {
        return Err(format!(
            "uvmet10 shape {:?} describes {plane_cells} cells per component, expected {cells}",
            output.shape
        ));
    }
    let expected_values = (*components).checked_mul(cells).ok_or_else(|| {
        "uvmet10 component length overflows the platform address space".to_string()
    })?;
    if output.data.len() != expected_values {
        return Err(format!(
            "uvmet10 shape {:?} requires {expected_values} values, got {}",
            output.shape,
            output.data.len()
        ));
    }
    let (u, v) = output.data.split_at(cells);
    Ok((clean_values(u), clean_values(v), output.units))
}

fn checked_horizontal_cells(ny: usize, nx: usize) -> Result<usize, String> {
    ny.checked_mul(nx).ok_or_else(|| {
        format!("horizontal dimensions [{ny}, {nx}] overflow the platform address space")
    })
}

pub fn compute_var(
    file: &WrfFile,
    name: &str,
    timeidx: usize,
    units: Option<&str>,
) -> Result<VarOutput, String> {
    let opts = ComputeOpts {
        units: units.map(str::to_string),
        ..ComputeOpts::default()
    };
    // Isolate each diagnostic: a panic inside wrf-core (or a crate it calls
    // into, e.g. ecape-rs) on a pathological grid/profile must cost that ONE
    // field — recorded as a note — not the whole multi-minute import. Without
    // this, the unwind kills the rw-ui-wrf-process worker and the entire
    // import dies with "WRF worker stopped unexpectedly".
    isolate_panics(name, || {
        getvar(file, name, Some(timeidx), &opts).map_err(|err| err.to_string())
    })
}

/// Run `f`, converting a panic into an `Err` naming `what`, so one failing
/// field computation degrades to a per-field note instead of unwinding the
/// import worker thread (shared with `local_import`'s volume build, which
/// needs the same guarantee). Inputs are shared references plus `WrfFile`'s
/// internal mutex (which poisons — and is then handled — rather than being
/// observed broken) and progress closures that only append/send messages,
/// so `AssertUnwindSafe` is sound here.
pub(crate) fn isolate_panics<T>(
    what: &str,
    f: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .map(|msg| (*msg).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown panic".to_string());
        Err(format!("panicked computing {what}: {message}"))
    })
}

fn total_precip(file: &WrfFile, timeidx: usize, cells: usize) -> Option<Vec<f32>> {
    let mut total = vec![0.0f32; cells];
    let mut found = false;
    for name in ["RAINC", "RAINNC", "RAINSH"] {
        let Ok(output) = compute_var(file, name, timeidx, None) else {
            continue;
        };
        let Ok((values, _)) = single_plane(output, cells) else {
            continue;
        };
        for (accum, value) in total.iter_mut().zip(values) {
            if value.is_finite() {
                *accum += value;
            } else {
                *accum = f32::NAN;
            }
        }
        found = true;
    }
    found.then_some(total)
}

fn clean_values(values: &[f64]) -> Vec<f32> {
    values
        .iter()
        .map(|value| {
            if !value.is_finite() || value.abs() >= 1.0e30 || *value <= -9998.0 {
                f32::NAN
            } else {
                *value as f32
            }
        })
        .collect()
}

fn derived_name(wrf_name: &str, split_index: Option<usize>) -> String {
    let base = match (wrf_name.to_ascii_lowercase().as_str(), split_index) {
        ("uvmet10", Some(0)) => "uvmet10_u".to_string(),
        ("uvmet10", Some(1)) => "uvmet10_v".to_string(),
        ("cloudfrac", Some(0)) => "cloudfrac_low".to_string(),
        ("cloudfrac", Some(1)) => "cloudfrac_mid".to_string(),
        ("cloudfrac", Some(2)) => "cloudfrac_high".to_string(),
        ("bunkers_rm", Some(0)) => "bunkers_rm_u".to_string(),
        ("bunkers_rm", Some(1)) => "bunkers_rm_v".to_string(),
        ("bunkers_lm", Some(0)) => "bunkers_lm_u".to_string(),
        ("bunkers_lm", Some(1)) => "bunkers_lm_v".to_string(),
        ("effective_inflow", Some(0)) => "effective_inflow_base".to_string(),
        ("effective_inflow", Some(1)) => "effective_inflow_top".to_string(),
        (_, Some(index)) => format!("{wrf_name}_{}", index + 1),
        (_, None) => wrf_name.to_string(),
    };
    let base = slug(&base);
    wrf_product_slug(&base)
        .map(str::to_string)
        .unwrap_or_else(|| format!("wrf_{base}"))
}

/// Canonical store names for authoritative raw-wrfout diagnostics. The
/// post-processed approximation tests strip their mandatory `approx_` prefix
/// and verify the remaining diagnostic family against this map.
pub(crate) fn wrf_product_slug(base: &str) -> Option<&'static str> {
    match base {
        "sbcape" => Some("sbcape"),
        "sbcin" => Some("sbcin"),
        "mlcape" => Some("mlcape"),
        "mlcin" => Some("mlcin"),
        "mucape" => Some("mucape"),
        "mucin" => Some("mucin"),
        "dcape" => Some("dcape"),
        "sbecape" => Some("sbecape"),
        "mlecape" => Some("mlecape"),
        "muecape" => Some("muecape"),
        "sbncape" => Some("sbncape"),
        "sbecin" => Some("sbecin"),
        "mlecin" => Some("mlecin"),
        // wrf-core's heavy ECAPE stack wraps the SAME ecape-rs solver the
        // GRIB heavy lane uses, with parcel_type defaulting to "sb"
        // (diag/ecape.rs::resolve_ecape_opts), so its plain `ecape`,
        // `ncape`, and `ecape_cin` outputs ARE the surface-based recipe
        // quantities and store under those recipe slugs.  wrf-core exposes
        // no ml/mu variant to map, so those grids are not missing: with
        // `--heavy` on they come from the SHARED recipe lane instead
        // (`push_shared_heavy_recipe_grids`), which solves all three
        // parcels from this hour's own planes and volumes.
        "ecape" => Some("sbecape"),
        "ncape" => Some("sbncape"),
        "ecape_cin" => Some("sbecin"),
        "ecape_scp" => Some("ecape_scp"),
        // compute_ecape_ehi is SB ECAPE x 0-1 km SRH / 160000 (depth_m
        // defaults to 1000), the recipe catalog's ecape_ehi_0_1km --
        // DerivedRecipe::parse declares the same equivalence.
        "ecape_ehi" => Some("ecape_ehi_0_1km"),
        "ecape_ehi_0_1km" => Some("ecape_ehi_0_1km"),
        "ecape_ehi_0_3km" => Some("ecape_ehi_0_3km"),
        "ecape_stp" => Some("ecape_stp"),
        "lcl" => Some("lcl"),
        "lfc" => Some("lfc"),
        "el" => Some("el"),
        "ecape_lfc" => Some("ecape_lfc"),
        "ecape_el" => Some("ecape_el"),
        "srh1" => Some("srh_0_1km"),
        "srh3" => Some("srh_0_3km"),
        "srh_0_1km" => Some("srh_0_1km"),
        "srh_0_3km" => Some("srh_0_3km"),
        "shear_0_1km" => Some("bulk_shear_0_1km"),
        "shear_0_6km" => Some("bulk_shear_0_6km"),
        "bulk_shear_0_1km" => Some("bulk_shear_0_1km"),
        "bulk_shear_0_6km" => Some("bulk_shear_0_6km"),
        "stp" => Some("stp"),
        "stp_fixed" => Some("stp_fixed"),
        "stp_effective" => Some("stp_effective"),
        "scp" => Some("scp"),
        // compute_ehi is SBCAPE x 0-1 km SRH / 160000 (depth_m defaults
        // to 1000) -- exactly the catalog's ehi_0_1km "sb proxy" recipe.
        "ehi" => Some("ehi_0_1km"),
        "ehi_0_1km" => Some("ehi_0_1km"),
        // NOT mapped: wrf-core's lapse_rate_700_500 / lapse_rate_0_3km
        // default to PLAIN-temperature lapse rates, while the recipe
        // catalog's grids are VIRTUAL-temperature lapse rates
        // (rustwx-calc::compute_lapse_rate_700_500).  They stay under
        // wrf_* browse names until the import passes use_virtual=true
        // and the kernel equivalence is verified.
        "tehi" => Some("tehi"),
        "tts" => Some("tts"),
        "vtp_mod" => Some("vtp_mod"),
        "uhel" => Some("uhel"),
        _ => None,
    }
}

pub(crate) fn wrf_projection(file: &WrfFile) -> Option<GridProjection> {
    let map_proj = file.global_attr_i32("MAP_PROJ").ok()?;
    match map_proj {
        1 => {
            let truelat1 = file.global_attr_f64("TRUELAT1").ok()?;
            let truelat2 = crate::local_import::normalize_lambert_truelat2(
                truelat1,
                file.global_attr_f64("TRUELAT2").ok(),
            );
            let stand_lon = file
                .global_attr_f64("STAND_LON")
                .ok()
                .or_else(|| file.global_attr_f64("CEN_LON").ok())?;
            Some(GridProjection::LambertConformal {
                standard_parallel_1_deg: truelat1,
                standard_parallel_2_deg: truelat2,
                central_meridian_deg: stand_lon,
            })
        }
        2 => {
            let truelat1 = file.global_attr_f64("TRUELAT1").ok()?;
            let stand_lon = file
                .global_attr_f64("STAND_LON")
                .ok()
                .or_else(|| file.global_attr_f64("CEN_LON").ok())?;
            Some(GridProjection::PolarStereographic {
                true_latitude_deg: truelat1,
                central_meridian_deg: stand_lon,
                // wrf-python chooses the pole from TRUELAT1. CEN_LAT can have
                // the opposite sign for a nested domain and is not authoritative.
                south_pole_on_projection_plane: crate::local_import::wrf_polar_uses_south_pole(
                    truelat1,
                ),
            })
        }
        3 => Some(GridProjection::Mercator {
            latitude_of_true_scale_deg: file.global_attr_f64("TRUELAT1").unwrap_or(0.0),
            central_meridian_deg: crate::local_import::wrf_mercator_central_longitude(
                file.global_attr_f64("STAND_LON").ok(),
            ),
        }),
        6 if crate::local_import::wrf_latlon_is_unrotated(
            file.global_attr_f64("POLE_LAT").ok(),
            file.global_attr_f64("POLE_LON").ok(),
        ) =>
        {
            Some(GridProjection::Geographic)
        }
        // GridProjection has no rotated-pole representation. The caller still
        // supplies the exact curvilinear XLAT/XLONG grid, so None is accurate.
        6 => None,
        _ => None,
    }
}

/// Extract a sortable `YYYYMMDDHHMMSS` stamp from a wrfout-style filename
/// (`wrfout_d03_2025-06-21_01_30_00` / `..._01:30:00`). Shared with
/// `wrf_radar`'s multi-file loop ordering, which must sort frames by model
/// time rather than raw filename.
pub(crate) fn parse_wrf_timestamp(name: &str) -> Option<String> {
    for token in name.split(['.', '/', '\\']) {
        let bytes = token.as_bytes();
        if bytes.len() < 19 {
            continue;
        }
        for start in 0..=bytes.len().saturating_sub(19) {
            let slice = &token[start..start + 19];
            let chars = slice.as_bytes();
            let timestampish = chars[4] == b'-'
                && chars[7] == b'-'
                && chars[10] == b'_'
                && matches!(chars[13], b':' | b'_')
                && matches!(chars[16], b':' | b'_')
                && chars.iter().enumerate().all(|(index, byte)| {
                    matches!(index, 4 | 7 | 10 | 13 | 16) || byte.is_ascii_digit()
                });
            if timestampish {
                return Some(
                    slice
                        .replace('-', "")
                        .replace([':', '_'], "")
                        .chars()
                        .take(14)
                        .collect(),
                );
            }
        }
    }
    None
}

fn slug(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut last_was_underscore = false;
    for ch in value.chars() {
        let lower = ch.to_ascii_lowercase();
        if lower.is_ascii_alphanumeric() {
            output.push(lower);
            last_was_underscore = false;
        } else if !last_was_underscore {
            output.push('_');
            last_was_underscore = true;
        }
    }
    output.trim_matches('_').to_string()
}

fn default_true() -> bool {
    true
}

fn normalize_filter_tokens(tokens: Vec<String>) -> Vec<String> {
    let mut normalized = tokens
        .into_iter()
        .flat_map(|token| {
            token
                .split([',', ';', '\n', '\r', '\t', ' '])
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .map(|token| slug(&token))
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    normalized.sort();
    normalized.dedup();
    normalized
}

/// Stable store-key suffix for a full WRF processing plan. Hash the normalized
/// option contract (rather than caller order or formatting) so semantically
/// equivalent filter lists share a run while any option capable of changing
/// the realized field set gets a distinct run. FNV-1a is explicit here because
/// the standard library's hash implementation is not a persistent-format API.
fn processing_profile_suffix(options: &WrfProcessOptions) -> String {
    let normalized = options.clone().normalized();
    let mut hash = profile_hash_update(PROFILE_FNV64_OFFSET, b"rw-wrf-profile-v2\0");
    hash = profile_hash_update(hash, WRF_PROCESS_SCIENCE_MARKER.as_bytes());
    hash = profile_hash_update(
        hash,
        &[
            u8::from(normalized.core_fields),
            u8::from(normalized.diagnostics),
            u8::from(normalized.heavy_ecape),
            u8::from(normalized.raw_extras),
            u8::from(normalized.stored_planes),
        ],
    );
    for (label, tokens) in [
        (b"only".as_slice(), normalized.only.as_slice()),
        (b"skip".as_slice(), normalized.skip.as_slice()),
    ] {
        hash = profile_hash_update(hash, label);
        hash = profile_hash_update(hash, &(tokens.len() as u64).to_le_bytes());
        for token in tokens {
            hash = profile_hash_update(hash, &(token.len() as u64).to_le_bytes());
            hash = profile_hash_update(hash, token.as_bytes());
        }
    }
    if normalized.viewer_2d {
        hash = profile_hash_update(hash, b"viewer-2d-v1\0");
        for selector in &normalized.chart_selectors {
            hash = profile_hash_update(hash, selector.key().as_bytes());
            hash = profile_hash_update(hash, b"\0");
        }
        format!("viewer2d_{WRF_PROCESS_SCIENCE_MARKER}_{hash:016x}")
    } else if normalized.named_products_only {
        hash = profile_hash_update(hash, b"named-products-v1\0");
        format!("named_{WRF_PROCESS_SCIENCE_MARKER}_{hash:016x}")
    } else {
        format!("full_{WRF_PROCESS_SCIENCE_MARKER}_{hash:016x}")
    }
}

fn profile_hash_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PROFILE_FNV64_PRIME);
    }
    hash
}

fn product_filter_keys(wrf_name: &str, store_name: Option<&str>) -> Vec<String> {
    let mut keys = Vec::new();
    push_filter_key(&mut keys, wrf_name);
    if let Some(store_name) = store_name {
        push_filter_key(&mut keys, store_name);
    }
    if let Some(mapped) = wrf_product_slug(&slug(wrf_name)) {
        push_filter_key(&mut keys, mapped);
    }
    if let Some(store_name) = store_name.and_then(|name| name.strip_prefix("wrf_")) {
        push_filter_key(&mut keys, store_name);
    }
    keys
}

fn push_filter_key(keys: &mut Vec<String>, value: &str) {
    let key = slug(value);
    if !key.is_empty() && !keys.contains(&key) {
        keys.push(key);
    }
}

fn filter_token_matches(key: &str, token: &str) -> bool {
    key == token || key.contains(token)
}

fn is_heavy_wrf_diagnostic(name: &str) -> bool {
    let key = slug(name);
    // "ncape" is a full ecape-rs solve (normalized CAPE) but does not
    // literally contain "ecape", so without the explicit match it leaks
    // into the default pass at ~10 s/file on an 800x800x79 grid; in the
    // Heavy group it rides the ecape stack cache for milliseconds.
    key.contains("ecape")
        || matches!(
            key.as_str(),
            "ncape"
                | "sbecape"
                | "mlecape"
                | "muecape"
                | "sbncape"
                | "sbecin"
                | "mlecin"
                | "muecin"
                | "ecape_scp"
                | "ecape_ehi"
                | "ecape_ehi_0_1km"
                | "ecape_ehi_0_3km"
                | "ecape_stp"
                | "ecape_lfc"
                | "ecape_el"
        )
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| path.display().to_string())
}

fn writer_build() -> &'static str {
    concat!(
        env!("CARGO_PKG_NAME"),
        " ",
        env!("CARGO_PKG_VERSION"),
        " wrf_science_v5"
    )
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_run_identity_is_case_stable_and_profile_scoped() {
        let first = LiveWrfTarget {
            case_sha256: "a".repeat(64),
            storage_slot: 0,
            exact_time: RwsExactTime::new(0, 1_700_000_000),
        };
        let later = LiveWrfTarget {
            case_sha256: first.case_sha256.clone(),
            storage_slot: 3,
            exact_time: RwsExactTime::new(900, 1_700_000_900),
        };
        let options = WrfProcessOptions::default();
        assert_eq!(first.run_name(&options), later.run_name(&options));
        let mut core_only = options.clone();
        core_only.diagnostics = false;
        assert_ne!(first.run_name(&options), first.run_name(&core_only));
        let mut different_case = first.clone();
        different_case.case_sha256 = "b".repeat(64);
        assert_ne!(first.run_name(&options), different_case.run_name(&options));
    }

    #[test]
    fn volume_omission_note_reports_only_the_actual_retained_2d_count() {
        let none = volume_omission_note(0, "working set too large");
        assert!(none.contains("retained 0 independently available 2-D products"));
        assert!(!none.contains("preserved"));

        let one = volume_omission_note(1, "bad native shape");
        assert!(one.contains("retained 1 independently available 2-D product:"));

        let several = volume_omission_note(7, "allocation failed");
        assert!(several.contains("retained 7 independently available 2-D products:"));
    }

    #[test]
    fn parses_wrf_timestamp_with_colons_or_underscores() {
        assert_eq!(
            parse_wrf_timestamp("wrfout_d02_1974-04-03_09:00:00"),
            Some("19740403090000".to_string())
        );
        assert_eq!(
            parse_wrf_timestamp("wrfout_d02_1974-04-03_09_00_00"),
            Some("19740403090000".to_string())
        );
    }

    #[test]
    fn derived_split_names_are_stable() {
        assert_eq!(derived_name("uvmet10", Some(0)), "wrf_uvmet10_u");
        assert_eq!(derived_name("cloudfrac", Some(2)), "wrf_cloudfrac_high");
        assert_eq!(derived_name("sbcape", None), "sbcape");
        assert_eq!(derived_name("srh1", None), "srh_0_1km");
        assert_eq!(derived_name("shear_0_6km", None), "bulk_shear_0_6km");
    }

    #[test]
    fn uvmet10_split_requires_two_complete_earth_relative_planes() {
        let output = VarOutput {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 1, 2],
            units: "m/s".to_string(),
            description: "earth-relative 10 m wind".to_string(),
        };
        let (u, v, units) = split_uvmet10(output, 2).expect("valid uvmet10 split");
        assert_eq!(u, vec![1.0, 2.0]);
        assert_eq!(v, vec![3.0, 4.0]);
        assert_eq!(units, "m/s");

        let malformed = VarOutput {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![1, 2, 2],
            units: "m/s".to_string(),
            description: "malformed wind".to_string(),
        };
        assert!(
            split_uvmet10(malformed, 4)
                .expect_err("one component must fail")
                .contains("two uvmet10 components")
        );

        let truncated = VarOutput {
            data: vec![1.0, 2.0, 3.0],
            shape: vec![2, 1, 2],
            units: "m/s".to_string(),
            description: "truncated wind".to_string(),
        };
        assert!(
            split_uvmet10(truncated, 2)
                .expect_err("a truncated second component must fail")
                .contains("requires 4 values")
        );
    }

    #[test]
    fn native_reflectivity_is_authoritative_and_fallback_is_absence_only() {
        let fallback_calls = std::cell::Cell::new(0_u32);
        let error = select_reflectivity_volume::<Vec<f64>>(
            true,
            || Err("native shape mismatch".to_string()),
            || {
                fallback_calls.set(fallback_calls.get() + 1);
                Ok(vec![99.0])
            },
        )
        .expect_err("a present native read error must fail closed");
        assert!(error.contains("native shape mismatch"));
        assert_eq!(
            fallback_calls.get(),
            0,
            "generic dbz must not replace a present-but-invalid native field"
        );

        let native_calls = std::cell::Cell::new(0_u32);
        let (values, source) = select_reflectivity_volume(
            false,
            || {
                native_calls.set(native_calls.get() + 1);
                Ok(vec![-1.0])
            },
            || Ok(vec![12.0]),
        )
        .expect("an absent native field permits the generic fallback");
        assert_eq!(native_calls.get(), 0);
        assert_eq!(values, vec![12.0]);
        assert_eq!(source, ReflectivityVolumeSource::GenericDbzFallback);
    }

    #[test]
    fn native_reflectivity_rejects_same_length_permuted_shape() {
        assert_eq!(
            validate_native_reflectivity_volume(vec![1.0; 24], &[2, 3, 4], 2, 3, 4)
                .expect("canonical [nz,ny,nx] shape"),
            vec![1.0; 24]
        );
        let error = validate_native_reflectivity_volume(vec![1.0; 24], &[3, 2, 4], 2, 3, 4)
            .expect_err("permuted native dimensions must fail closed");
        assert!(error.contains("shape/order mismatch"));
    }

    #[test]
    fn one_km_reflectivity_interpolates_linear_z_and_handles_bounds() {
        // Three columns: exact 1-km model level, a 0/20 dBZ midpoint, and a
        // profile whose lowest model level is already above 1 km.
        let dbz = vec![7.0, 0.0, 5.0, 9.0, 20.0, 10.0];
        let height_agl = vec![1_000.0, 0.0, 1_500.0, 2_000.0, 2_000.0, 2_500.0];
        let values =
            reflectivity_at_height_agl(&dbz, &height_agl, 2, 3, 1_000.0).expect("valid columns");
        assert_eq!(values[0], 7.0, "an exact level must remain exact");
        assert!(
            (values[1] - 17.032_913_781_187).abs() < 1.0e-12,
            "0/20 dBZ halfway in linear Z should be 17.032913781187 dBZ, got {}",
            values[1]
        );
        assert!(
            values[2].is_nan(),
            "a target outside the model-height range must not extrapolate"
        );
    }

    #[test]
    fn composite_reflectivity_is_the_vertical_max_of_the_shared_volume() {
        let dbz = vec![0.0, 20.0, 15.0, f64::NAN, 10.0, 25.0];
        let composite =
            composite_reflectivity_from_dbz(&dbz, 3, 2).expect("complete reflectivity volume");
        assert_eq!(composite, vec![15.0, 25.0]);
        assert!(
            composite_reflectivity_from_dbz(&dbz[..5], 3, 2)
                .expect_err("a truncated volume must fail")
                .contains("requires 6 values")
        );
    }

    #[test]
    fn reflectivity_fill_sentinels_are_not_published_or_interpolated() {
        let sentinels = [-999.0, -8888.0, -9999.0, 9999.0];
        for sentinel in sentinels {
            assert!(!valid_reflectivity_dbz(sentinel));
            let interpolated =
                reflectivity_at_height_agl(&[sentinel, 20.0], &[0.0, 2_000.0], 2, 1, 1_000.0)
                    .expect("shape-valid sentinel column");
            assert!(
                interpolated[0].is_nan(),
                "sentinel {sentinel} must not synthesize a 1-km echo"
            );
        }

        let composite =
            composite_reflectivity_from_dbz(&[-999.0, 10.0, -8888.0, 20.0, 9999.0, 15.0], 3, 2)
                .expect("shape-valid sentinel volume");
        assert!(composite[0].is_nan());
        assert_eq!(composite[1], 20.0);
        assert!(valid_reflectivity_dbz(MIN_VALID_REFLECTIVITY_DBZ));
        assert!(valid_reflectivity_dbz(MAX_VALID_REFLECTIVITY_DBZ));
    }

    #[test]
    fn reflectivity_selection_and_field_plan_are_product_specific() {
        let default_plan = WrfProcessOptions::default().planned_store_fields();
        assert!(
            default_plan
                .iter()
                .any(|name| name == COMPOSITE_REFLECTIVITY_STORE)
        );
        assert!(
            default_plan
                .iter()
                .any(|name| name == REFLECTIVITY_1KM_STORE)
        );
        assert!(
            !default_plan.iter().any(|name| name == "wrf_maxdbz"),
            "the full VARS pass must not plan a conflicting generic maxdbz"
        );

        let only_composite = WrfProcessOptions {
            only: vec![COMPOSITE_REFLECTIVITY_FILTER.to_string()],
            ..WrfProcessOptions::default()
        }
        .normalized()
        .planned_store_fields();
        assert!(
            only_composite
                .iter()
                .any(|name| name == COMPOSITE_REFLECTIVITY_STORE)
        );
        assert!(
            !only_composite
                .iter()
                .any(|name| name == REFLECTIVITY_1KM_STORE)
        );

        let only_one_km = WrfProcessOptions {
            only: vec![REFLECTIVITY_1KM_STORE.to_string()],
            ..WrfProcessOptions::default()
        }
        .normalized()
        .planned_store_fields();
        assert!(
            !only_one_km
                .iter()
                .any(|name| name == COMPOSITE_REFLECTIVITY_STORE)
        );
        assert!(
            only_one_km
                .iter()
                .any(|name| name == REFLECTIVITY_1KM_STORE)
        );

        let reflectivity_family = WrfProcessOptions {
            only: vec!["reflectivity".to_string()],
            ..WrfProcessOptions::default()
        }
        .normalized()
        .planned_store_fields();
        assert!(
            reflectivity_family
                .iter()
                .any(|name| name == COMPOSITE_REFLECTIVITY_STORE)
        );
        assert!(
            reflectivity_family
                .iter()
                .any(|name| name == REFLECTIVITY_1KM_STORE)
        );

        assert_eq!(
            composite_reflectivity_selector(),
            FieldSelector::entire_atmosphere(CanonicalField::CompositeReflectivity)
        );
        assert_eq!(
            reflectivity_1km_selector(),
            FieldSelector::height_agl(CanonicalField::RadarReflectivity, 1000)
        );
    }

    #[test]
    fn wrf_science_marker_scopes_profile_and_writer_provenance() {
        let profile = processing_profile_suffix(&WrfProcessOptions::default());
        assert!(profile.contains(WRF_PROCESS_SCIENCE_MARKER));
        assert!(writer_build().contains(WRF_PROCESS_SCIENCE_MARKER));
    }

    #[test]
    fn lowest_model_level_fallbacks_are_explicit_surface_approximations() {
        let shape = GridShape::new(1, 1).expect("test shape");
        let grid = LatLonGrid::new(shape, vec![35.0], vec![-97.0]).expect("test grid");
        let mut fields = WrfHourFields {
            grid: grid.clone(),
            projection: None,
            canonical: Vec::new(),
            derived: Vec::new(),
            volumes: Vec::new(),
            notes: Vec::new(),
        };
        push_canonical_values(
            &mut fields,
            &grid,
            None,
            "temperature_2m",
            FieldSelector::height_agl(CanonicalField::Temperature, 2),
            "K",
            vec![290.0],
        );

        fill_missing_surface(
            &mut fields,
            &grid,
            None,
            SurfaceFallback {
                surface_pressure_pa: vec![95_000.0],
                temperature_2m_k: vec![285.0],
                dewpoint_2m_k: vec![280.0],
                u_10m: vec![5.0],
                v_10m: vec![-2.0],
            },
            &WrfProcessOptions::default(),
        );

        for exact_name in ["surface_pressure", "dewpoint_2m", "u_10m", "v_10m"] {
            assert!(
                !fields.canonical.iter().any(|(name, _)| name == exact_name),
                "fallback must not masquerade as {exact_name}"
            );
        }
        assert!(
            !fields
                .canonical
                .iter()
                .any(|(name, _)| name == "approx_temperature_2m"),
            "an exact 2 m temperature must suppress its approximation"
        );

        for (name, expected_selector) in [
            (
                "approx_surface_pressure",
                FieldSelector::surface(CanonicalField::Pressure),
            ),
            (
                "approx_dewpoint_2m",
                FieldSelector::surface(CanonicalField::Dewpoint),
            ),
            (
                "approx_u_10m",
                FieldSelector::surface(CanonicalField::UWind),
            ),
            (
                "approx_v_10m",
                FieldSelector::surface(CanonicalField::VWind),
            ),
        ] {
            let field = fields
                .canonical
                .iter()
                .find_map(|(stored_name, field)| (stored_name == name).then_some(field))
                .unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(field.selector, expected_selector);
        }
    }

    #[test]
    fn optional_real_fixture_processes_wrf_products() {
        let Some(path) = std::env::var_os("RW_WRF_PROCESS_FIXTURE") else {
            return;
        };
        let store =
            std::env::temp_dir().join(format!("rw-wrf-process-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&store);
        let (tx, _rx) = channel();
        let summary = process_paths(
            &[PathBuf::from(path)],
            &store,
            &WrfProcessOptions {
                heavy_ecape: true,
                ..WrfProcessOptions::default()
            },
            &tx,
        )
        .expect("real WRF fixture should process");
        assert!(summary.hours_written >= 1);
        assert!(
            summary
                .variables
                .iter()
                .any(|name| name == "temperature_2m")
        );
        assert!(summary.variables.iter().any(|name| name == "sbcape"));
        assert!(summary.variables.iter().any(|name| name == "wrf_wspd10"));
        let _ = std::fs::remove_dir_all(&store);
    }

    /// End-to-end guard for the sounding fix: a real WRF file must land the
    /// `*_iso` isobaric volumes (as `pressure3d`) plus `surface_pressure`, and
    /// an interior column pull must carry real mid-tropospheric data. Gated on
    /// `RW_WRF_PROCESS_FIXTURE` (a `wrfout_*` path) like the sibling test.
    #[test]
    fn real_fixture_writes_isobaric_sounding_volumes() {
        let Some(path) = std::env::var_os("RW_WRF_PROCESS_FIXTURE") else {
            return;
        };
        let store =
            std::env::temp_dir().join(format!("rw-wrf-sounding-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&store);
        let (tx, _rx) = channel();
        // Sounding-focused: skip the heavy 2D diagnostics (CAPE/severe) so this
        // stays fast even on a ~2M-cell CONUS grid; core fields + volumes +
        // surface fallback are what matter here.
        let summary = process_paths(
            &[PathBuf::from(path)],
            &store,
            &WrfProcessOptions {
                diagnostics: false,
                raw_extras: false,
                heavy_ecape: false,
                ..WrfProcessOptions::default()
            },
            &tx,
        )
        .expect("real WRF fixture should process");

        let hour_path = store
            .join(&summary.model)
            .join(&summary.run)
            .join("f000.rws");
        let reader = rw_store::reader::HourReader::open(&hour_path).expect("open hour file");

        for name in [
            "temperature_iso",
            "dewpoint_iso",
            "u_iso",
            "v_iso",
            "height_iso",
        ] {
            let var = reader
                .variable(name)
                .unwrap_or_else(|| panic!("{name} missing from store"));
            assert_eq!(var.kind, "pressure3d", "{name} should be a 3D volume");
            assert!(!var.levels_hpa.is_empty(), "{name} has no isobaric levels");
        }
        assert!(
            reader.variable("surface_pressure").is_some(),
            "surface_pressure missing — the skew-T column builder needs it"
        );

        // An interior column pull carries real, physical data: temperatures
        // in a sane Kelvin band and geopotential height increasing as pressure
        // decreases (store levels are descending, so 1000 hPa is index 0).
        // This catches the unit/ordering regressions a finite check misses.
        let levels = reader
            .variable("temperature_iso")
            .expect("temperature_iso")
            .levels_hpa
            .clone();
        let temps = reader
            .read_profile_3d("temperature_iso", 5.0, 5.0)
            .expect("read temperature_iso profile");
        let heights = reader
            .read_profile_3d("height_iso", 5.0, 5.0)
            .expect("read height_iso profile");
        assert_eq!(temps.len(), levels.len());
        assert_eq!(heights.len(), levels.len());

        let finite_temps = temps.iter().filter(|value| value.is_finite()).count();
        assert!(
            finite_temps >= 5,
            "expected several finite isobaric temperatures, got {finite_temps} of {}",
            temps.len()
        );
        for (level, temp) in levels.iter().zip(&temps) {
            if temp.is_finite() {
                assert!(
                    (180.0..=330.0).contains(temp),
                    "{level} hPa temperature {temp} K is non-physical (unit bug?)"
                );
            }
        }
        let mut last_height = f32::NEG_INFINITY;
        for height in &heights {
            if height.is_finite() {
                assert!(
                    *height > last_height,
                    "height must increase as pressure decreases, got {height} after {last_height}"
                );
                last_height = *height;
            }
        }

        let _ = std::fs::remove_dir_all(&store);
    }

    /// Instrumented harness for the large-grid full-diagnostics "crash"
    /// (FABLE_BACKLOG #9): runs the DEFAULT full-diagnostics import — the
    /// exact path the "Process WRF" dock button drives, including the same
    /// `spawn_process_paths` worker-thread configuration — on the wrfout named
    /// by `RW_WRF_CRASH_FIXTURE`, forwarding every Progress message to stderr
    /// with a timestamp so any abort pinpoints WHICH diagnostic died. Run with
    /// `--nocapture` and stderr captured to a file. Env-gated like the sibling
    /// fixtures; heavy — release builds only on large grids.
    ///
    /// Findings from the 2026-07-06 investigation (Enderlin 250 m,
    /// 800x800x79): the import COMPLETES in optimized builds (~275 s,
    /// 117 variables) — the reported `0xffffffff` abort was an external
    /// kill (only `process::exit(-1)`-style termination yields that code on
    /// this toolchain; a Rust abort/alloc-failure is 0xC0000409, a stack
    /// overflow 0xC00000FD, a panic 101), i.e. a tool-timeout kill of a
    /// 20-40x-slower debug run — not an in-process bug. See
    /// docs/wrf-import-large-grids.md.
    #[test]
    fn optional_real_fixture_default_import_instrumented() {
        let Some(fixture) = std::env::var_os("RW_WRF_CRASH_FIXTURE") else {
            return;
        };
        let store = std::env::temp_dir().join(format!("rw-wrf-crash-repro-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&store);
        let start = std::time::Instant::now();
        let task = spawn_process_paths(
            vec![PathBuf::from(fixture)],
            store.clone(),
            WrfProcessOptions::default(),
        );
        loop {
            match task.rx.recv() {
                Ok(WrfProcessMessage::Progress(line)) => {
                    eprintln!("[{:9.2?}] {line}", start.elapsed());
                }
                Ok(WrfProcessMessage::Done(result)) => {
                    let summary = result.expect("default full-diagnostics import should succeed");
                    eprintln!(
                        "[{:9.2?}] DONE: {} hour(s), {} variables: {}",
                        start.elapsed(),
                        summary.hours_written,
                        summary.variables.len(),
                        summary.variables.join(", ")
                    );
                    for note in &summary.notes {
                        eprintln!("[note] {note}");
                    }
                    assert!(summary.hours_written >= 1);
                    assert!(summary.variables.iter().any(|name| name == "sbcape"));
                    break;
                }
                // Disconnected without Done == the worker thread panicked
                // (a process-fatal abort never reaches this arm).
                Err(err) => panic!(
                    "[{:9.2?}] worker died without Done (panic in worker): {err}",
                    start.elapsed()
                ),
            }
        }
        let _ = std::fs::remove_dir_all(&store);
    }

    #[test]
    fn isolate_panics_converts_panic_to_error_and_passes_results_through() {
        assert_eq!(
            isolate_panics("field_ok", || Ok::<_, String>(7)),
            Ok(7),
            "successful computations must pass through untouched"
        );
        assert_eq!(
            isolate_panics("field_err", || Err::<(), _>("no such var".to_string())),
            Err("no such var".to_string()),
            "ordinary errors must pass through untouched"
        );
        let caught = isolate_panics::<()>("sbcape", || panic!("index out of bounds: 42"));
        assert_eq!(
            caught,
            Err("panicked computing sbcape: index out of bounds: 42".to_string()),
            "a panicking diagnostic must degrade to a named per-field error"
        );
        let caught_string =
            isolate_panics::<()>("srh3", || std::panic::panic_any("boom".to_string()));
        assert_eq!(
            caught_string,
            Err("panicked computing srh3: boom".to_string()),
            "String payloads must be extracted too"
        );
    }

    #[test]
    fn wrf_options_filter_heavy_and_names() {
        let default_options = WrfProcessOptions::default().normalized();
        assert!(!default_options.should_process(
            "sbecape",
            Some("sbecape"),
            WrfProductGroup::Heavy
        ));
        assert!(default_options.should_process(
            "srh1",
            Some("srh_0_1km"),
            WrfProductGroup::Diagnostic
        ));

        let filtered = WrfProcessOptions {
            only: vec!["srh".to_string()],
            skip: vec!["srh_0_3km".to_string()],
            ..WrfProcessOptions::default()
        }
        .normalized();
        assert!(filtered.should_process("srh1", Some("srh_0_1km"), WrfProductGroup::Diagnostic));
        assert!(!filtered.should_process("srh3", Some("srh_0_3km"), WrfProductGroup::Diagnostic));
        assert!(!filtered.should_process("t2", Some("temperature_2m"), WrfProductGroup::Core));
    }

    #[test]
    fn ncape_is_classified_with_the_heavy_ecape_group() {
        // ncape is a full ecape-rs solve; misclassified as Diagnostic it cost
        // ~10 s per 800x800x79 file in the default pass (perf audit
        // 2026-07-09). In the Heavy group it rides the ecape stack cache.
        assert!(is_heavy_wrf_diagnostic("ncape"));
        assert!(is_heavy_wrf_diagnostic("sbncape"));
        assert!(!is_heavy_wrf_diagnostic("sbcape"));
        assert!(!is_heavy_wrf_diagnostic("stp"));
    }

    /// Real-data proof for the UI field selector: the SAME wrfout processed
    /// with a narrowed selection (core fields only — no diagnostics, raw, or
    /// heavy eCAPE) must write ONLY the selected fields into the store hour — a
    /// strict, strictly-smaller subset of the full default set. This exercises
    /// the exact path the "WRF full diagnostics…" import drives. Gated on
    /// `RW_WRF_PROCESS_FIXTURE` (a `wrfout_*` path) like the sibling fixtures.
    #[test]
    fn real_fixture_selection_narrows_written_fields() {
        let Some(fixture) = std::env::var_os("RW_WRF_PROCESS_FIXTURE") else {
            return;
        };
        let path = PathBuf::from(fixture);

        // Process `path` once under `options`, returning the store hour's
        // authoritative on-disk variable-name set (sorted, deduped).
        let written_fields = |options: WrfProcessOptions, tag: &str| -> Vec<String> {
            let store =
                std::env::temp_dir().join(format!("rw-wrf-select-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&store);
            let (tx, _rx) = channel();
            let summary = process_paths(std::slice::from_ref(&path), &store, &options, &tx)
                .unwrap_or_else(|err| panic!("process ({tag}) failed: {err}"));
            let hour_path = store
                .join(&summary.model)
                .join(&summary.run)
                .join("f000.rws");
            let reader = rw_store::reader::HourReader::open(&hour_path).expect("open hour file");
            let mut names: Vec<String> = reader
                .meta()
                .variables
                .iter()
                .map(|var| var.name.clone())
                .collect();
            names.sort();
            names.dedup();
            let _ = std::fs::remove_dir_all(&store);
            names
        };

        let narrowed = written_fields(
            WrfProcessOptions {
                diagnostics: false,
                raw_extras: false,
                heavy_ecape: false,
                ..WrfProcessOptions::default()
            }
            .normalized(),
            "narrow",
        );
        let default = written_fields(WrfProcessOptions::default().normalized(), "default");

        eprintln!(
            "NARROWED core-only ({} fields): {}",
            narrowed.len(),
            narrowed.join(", ")
        );
        eprintln!(
            "DEFAULT full set ({} fields): {}",
            default.len(),
            default.join(", ")
        );

        // Narrowed keeps the core surface fields + isobaric sounding volumes…
        assert!(
            narrowed.iter().any(|name| name == "temperature_2m"),
            "narrowed selection must still write the core surface fields"
        );
        assert!(
            narrowed.iter().any(|name| name == "temperature_iso"),
            "narrowed selection must still write the sounding volumes"
        );
        // …but drops the severe diagnostics and raw extras the default writes.
        assert!(
            !narrowed.iter().any(|name| name == "sbcape"),
            "narrowed (diagnostics off) must NOT write CAPE and friends"
        );
        assert!(
            default.iter().any(|name| name == "sbcape"),
            "default set must include the severe diagnostics"
        );
        // Strict subset, strictly smaller: the selection genuinely narrowed the
        // written store hour rather than falling back to the full default set.
        assert!(
            narrowed.iter().all(|name| default.contains(name)),
            "narrowed field set must be a subset of the default field set"
        );
        assert!(
            narrowed.len() < default.len(),
            "narrowed selection must write fewer fields ({}) than the default ({})",
            narrowed.len(),
            default.len()
        );
    }

    #[test]
    fn every_direct_product_supported_by_full_import_is_planned_in_named_mode() {
        let full = WrfProcessOptions::default();
        let selectors = full.planned_store_selectors();
        let named = WrfProcessOptions {
            named_products_only: true,
            ..full
        };
        let fields = named.planned_store_fields();
        let mut covered = 0;
        for slug in rustwx_products::direct::store_direct_recipe_slugs() {
            let requirements = rustwx_models::plot_recipe_store_requirements(&slug).unwrap();
            // The full import defines which direct charts this source format
            // can supply; unsupported fields must not become new promises.
            if !requirements.iter().all(|row| {
                row.selector
                    .is_some_and(|selector| selectors.contains(&selector))
            }) {
                continue;
            }
            covered += 1;
            for requirement in requirements {
                let selector = requirement.selector.unwrap();
                let store_name = CORE_FIELD_CATALOG
                    .iter()
                    .find(|(_, name)| core_field_selector(name) == selector)
                    .map(|(_, name)| *name)
                    .or_else(|| {
                        crate::wrf_column_planes::COLUMN_PLANE_CATALOG
                            .iter()
                            .find(|plane| plane.selector() == selector)
                            .map(|plane| plane.store_name)
                    })
                    .map(str::to_string)
                    .unwrap_or_else(|| selector.key());
                assert!(fields.contains(&store_name), "{slug} needs {store_name}");
            }
        }
        assert!(covered > 20, "the full import must cover a direct gallery");
    }

    #[test]
    fn planned_store_fields_track_the_group_selection() {
        // Default: core + diagnostics + raw (no heavy eCAPE).
        let default_plan = WrfProcessOptions::default()
            .normalized()
            .planned_store_fields();
        assert!(default_plan.iter().any(|name| name == "temperature_2m"));
        assert!(default_plan.iter().any(|name| name == "temperature_iso"));
        assert!(default_plan.iter().any(|name| name == "sbcape"));
        // Heavy eCAPE (any entrainment-CAPE field) is off by default.
        assert!(!default_plan.iter().any(|name| name.contains("ecape")));

        // Core-only, heavy off: drops every diagnostic and raw field but keeps
        // the core surface fields and the isobaric sounding volumes.
        let core_only = WrfProcessOptions {
            diagnostics: false,
            raw_extras: false,
            heavy_ecape: false,
            ..WrfProcessOptions::default()
        }
        .normalized()
        .planned_store_fields();
        assert!(core_only.iter().any(|name| name == "temperature_2m"));
        assert!(core_only.iter().any(|name| name == "height_iso"));
        assert!(!core_only.iter().any(|name| name == "sbcape"));

        // Enabling heavy eCAPE adds the entrainment-CAPE family (e.g.
        // ecape_scp / ecape_ehi from wrf-core's VARS).
        let heavy = WrfProcessOptions {
            heavy_ecape: true,
            ..WrfProcessOptions::default()
        }
        .normalized()
        .planned_store_fields();
        assert!(heavy.iter().any(|name| name.contains("ecape")));
        assert!(heavy.len() > default_plan.len());

        // An only-list narrows the plan to the matching fields (plus the iso
        // volumes, which ride with the core group toggle, not the name filter).
        let only_cape = WrfProcessOptions {
            only: vec!["sbcape".to_string()],
            ..WrfProcessOptions::default()
        }
        .normalized()
        .planned_store_fields();
        assert!(only_cape.iter().any(|name| name == "sbcape"));
        assert!(!only_cape.iter().any(|name| name == "temperature_2m"));

        // Legacy U10/V10 filter keys still select the corresponding canonical
        // earth-relative component even though both are computed via uvmet10.
        let only_u10 = WrfProcessOptions {
            only: vec!["U10".to_string()],
            ..WrfProcessOptions::default()
        }
        .normalized()
        .planned_store_fields();
        assert!(only_u10.iter().any(|name| name == "u_10m"));
        assert!(!only_u10.iter().any(|name| name == "v_10m"));
    }

    #[test]
    fn processing_profile_is_order_independent_and_separates_field_plans() {
        let reordered_a = WrfProcessOptions {
            only: vec!["SBCAPE".to_string(), "temperature_2m".to_string()],
            skip: vec!["raw".to_string(), "ecape".to_string()],
            ..WrfProcessOptions::default()
        };
        let reordered_b = WrfProcessOptions {
            only: vec![" temperature_2m, sbcape ".to_string()],
            skip: vec!["ECAPE;raw;raw".to_string()],
            ..WrfProcessOptions::default()
        };
        assert_eq!(
            processing_profile_suffix(&reordered_a),
            processing_profile_suffix(&reordered_b),
            "equivalent normalized filters must reuse the same run"
        );

        let core_only = WrfProcessOptions {
            diagnostics: false,
            raw_extras: false,
            ..WrfProcessOptions::default()
        };
        assert_ne!(
            processing_profile_suffix(&WrfProcessOptions::default()),
            processing_profile_suffix(&core_only),
            "a different realized field plan must not replace the default run"
        );
        assert!(
            processing_profile_suffix(&core_only).starts_with("full_"),
            "full-processing profiles use a recognizable persistent suffix"
        );
    }

    #[test]
    fn malformed_horizontal_dimension_product_returns_an_error() {
        let error = checked_horizontal_cells(usize::MAX, 2)
            .expect_err("oversized file dimensions must fail closed");
        assert!(error.contains("overflow"));
        assert_eq!(checked_horizontal_cells(3, 4), Ok(12));
    }
}
