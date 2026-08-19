//! `rw_ensbatch` -- ensemble products from N member wrfouts, one valid
//! time, one panel.
//!
//! `rw_wrfbatch`'s contract is one run's frames into one store, and its
//! positional list means "these files are the same run" -- so N members at
//! one valid time cannot be expressed in it at all, and `gpuwm.rustwx`
//! already drives it one file per invocation with its own store root for
//! exactly that reason.  This binary is the shape that DOES fit: every
//! member imported into its own store, one 2-D plane read from each, the
//! reduction done in Rust (`rustwx-ensemble`), and the result put through
//! the same production render path a deterministic panel takes.
//!
//! A MEMBER is a time series, not a file: `frames_per_outfile = 1` is
//! WRF's default, so a member normally arrives as N single-frame wrfouts
//! and all N enter that member's store together.  `--frames` then indexes
//! real valid times.
//!
//! What the reductions are, and what each one is for, is documented once,
//! in `rustwx-ensemble`, because that is where they can be tested.
//!
//! ```text
//! rw_ensbatch --store-root DIR --out-dir DIR
//!             (--manifest FILE.json | --member N=WRFOUT_OR_MEMBER_DIR ...)
//!             [--field refl|uh|precip|t2|wspd10]
//!             [--products mean,spread,prob,pmm,paintball]
//!             [--threshold V] [--neighborhood-km V] [--frames all|N]
//!             [--nan-policy mask|refuse] [--pmm-tie-rule flat-index|average]
//!             [--accept-status LIST] [--width N] [--height N]
//!             [--source-label TEXT] [--overlays FILE] [--annotate FILE]
//! rw_ensbatch --list-fields | --help | --abi
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rustwx_core::{CanonicalField, FieldSelector};
use rustwx_ensemble::{
    MemberStack, NanPolicy, PmmTieRule, ensemble_mean, ensemble_spread, exceedance_probability,
    member_color, missingness_report, pmm_tie_report, probability_matched_mean,
};
use rustwx_render::{ContourLayer, LegendControls, LegendMode, LevelDensity, RenderDensity};
use rusty_weather::render_all::StoreFieldSource;
use rw_wrfbatch::annotate::{MapOverlays, PanelAnnotations, parse_color};
use rw_wrfbatch::panel::{PanelRequest, layout_path, render_panel, safe_component};
use rw_wrfbatch::scales;
use rw_wrfbatch::wrf_process::{WrfProcessMessage, WrfProcessOptions, spawn_process_paths};

/// The `--abi` contract line, in the shape the two sibling binaries use:
/// the vocabulary the PYTHON half parses, not a version number.
const ABI_MARKER: &str = "gpuwm-rw-ensbatch-products-v1\tmean\tspread\tprob\tpmm\tpaintball\t\
gpuwm-rw-ensbatch-events-v1\tRENDERED\tSKIPPED\tFAILED\t\
gpuwm-rw-ensbatch-vocabulary-v1\tMEMBERS\tCOVERAGE\tTIES";

/// The uncalibrated-ensemble stamp.  These products have never been
/// verified against observations, and a panel that does not say so gets
/// screenshotted into a briefing that assumes it was.
const EXPERIMENTAL_STAMP: &str = "EXPERIMENTAL -- uncalibrated ensemble";

/// Member statuses averaged without an explicit opt-in.
///
/// Both spellings are real: `DONE` is what `gpuwm.ensemble.manifest`
/// records and `complete` is the older spelling manifests in the field
/// still carry.  Deliberately short -- `RUNNING` and `FAILED` are exactly
/// the members that must not be averaged.
const DEFAULT_ACCEPT_STATUS: [&str; 2] = ["DONE", "complete"];

struct FieldSpec {
    name: &'static str,
    title: &'static str,
    units: &'static str,
    unit_slug: &'static str,
    default_threshold: f64,
}

fn field_specs() -> Vec<FieldSpec> {
    vec![
        FieldSpec {
            name: "refl",
            title: "composite reflectivity",
            units: "dBZ",
            unit_slug: "dbz",
            default_threshold: 40.0,
        },
        FieldSpec {
            name: "uh",
            title: "updraft helicity",
            units: "m2 s-2",
            unit_slug: "m2s2",
            default_threshold: 75.0,
        },
        FieldSpec {
            name: "precip",
            title: "accumulated precipitation",
            units: "mm",
            unit_slug: "mm",
            default_threshold: 25.0,
        },
        FieldSpec {
            name: "t2",
            title: "2 m temperature",
            units: "K",
            unit_slug: "k",
            default_threshold: 303.0,
        },
        FieldSpec {
            name: "wspd10",
            title: "10 m wind speed",
            units: "m s-1",
            unit_slug: "ms",
            default_threshold: 25.0,
        },
    ]
}

/// The canonical store selector each field reads.
///
/// Selectors, never stored variable NAMES: the name is an import-lane
/// spelling and the selector is the physical identity, which is what makes
/// the ensemble mean of reflectivity wear the same NWS ladder the
/// deterministic panel wears.
fn field_selector(field: &str) -> Option<FieldSelector> {
    match field {
        "refl" => Some(FieldSelector::entire_atmosphere(
            CanonicalField::CompositeReflectivity,
        )),
        "uh" => Some(FieldSelector::height_layer_agl(
            CanonicalField::UpdraftHelicity,
            2000,
            5000,
        )),
        "precip" => Some(FieldSelector::surface(CanonicalField::TotalPrecipitation)),
        "t2" => Some(FieldSelector::height_agl(CanonicalField::Temperature, 2)),
        "wspd10" => Some(FieldSelector::height_agl(CanonicalField::WindSpeed, 10)),
        _ => None,
    }
}

struct Args {
    store_root: PathBuf,
    out_dir: PathBuf,
    /// Member number -> the wrfout frames that ARE that member, in
    /// model-time order.  A member is a time series, not a file.
    members: Vec<(u32, Vec<PathBuf>)>,
    field: String,
    products: Vec<String>,
    threshold: Option<f64>,
    neighborhood_km: f64,
    frames: Option<usize>,
    nan_policy: NanPolicy,
    tie_rule: PmmTieRule,
    width: u32,
    height: u32,
    source_label: String,
    overlays: Option<MapOverlays>,
    annotations: Option<PanelAnnotations>,
}

fn usage() -> &'static str {
    "usage: rw_ensbatch --store-root DIR --out-dir DIR \
(--manifest FILE.json | --member N=WRFOUT_OR_MEMBER_DIR ...) [--field refl|uh|precip|t2|wspd10] \
[--products mean,spread,prob,pmm,paintball] [--threshold V] [--neighborhood-km V] \
[--frames all|N] [--nan-policy mask|refuse] [--pmm-tie-rule flat-index|average] \
[--accept-status LIST] [--width N] [--height N] [--source-label TEXT] \
[--overlays FILE.json] [--annotate FILE.json]\n       \
rw_ensbatch --list-fields | --help | --abi"
}

enum Invocation {
    Run(Box<Args>),
    ListFields,
    Abi,
    Help,
}

fn parse_args() -> Result<Invocation, String> {
    let mut store_root = None;
    let mut out_dir = None;
    let mut manifest: Option<PathBuf> = None;
    let mut explicit: Vec<(u32, Vec<PathBuf>)> = Vec::new();
    let mut field = "refl".to_string();
    let mut products = "mean,spread,prob,pmm,paintball".to_string();
    let mut threshold: Option<f64> = None;
    let mut neighborhood_km = 0.0f64;
    let mut frames: Option<usize> = None;
    let mut nan_policy = NanPolicy::Mask;
    let mut tie_rule = PmmTieRule::FlatIndex;
    let mut width = 1_200u32;
    let mut height = 900u32;
    let mut source_label = "ArWen".to_string();
    let mut accept_status: Vec<String> =
        DEFAULT_ACCEPT_STATUS.iter().map(|s| s.to_string()).collect();
    let mut overlays_path: Option<PathBuf> = None;
    let mut annotate_path: Option<PathBuf> = None;

    let mut raw = std::env::args().skip(1);
    while let Some(arg) = raw.next() {
        let mut value = || -> Result<String, String> {
            raw.next()
                .ok_or_else(|| format!("{arg} requires a value"))
        };
        match arg.as_str() {
            "--store-root" => store_root = Some(PathBuf::from(value()?)),
            "--out-dir" => out_dir = Some(PathBuf::from(value()?)),
            "--manifest" => manifest = Some(PathBuf::from(value()?)),
            "--member" => {
                let spec = value()?;
                let (number, path) = spec
                    .split_once('=')
                    .ok_or_else(|| format!("--member wants NUMBER=PATH, got {spec:?}"))?;
                let number: u32 = number
                    .trim()
                    .parse()
                    .map_err(|err| format!("--member {spec:?}: {err}"))?;
                // One file, or the member DIRECTORY holding its frames --
                // the manifest route and this one describe the same thing,
                // so they accept the same thing.
                let path = PathBuf::from(path);
                let frames = if path.is_dir() {
                    member_wrfout_series(&path)
                        .map_err(|problem| format!("--member {number}: {problem}"))?
                } else {
                    vec![path]
                };
                explicit.push((number, frames));
            }
            "--field" => field = value()?,
            "--products" => products = value()?,
            "--threshold" => {
                threshold = Some(
                    value()?
                        .parse()
                        .map_err(|err| format!("invalid --threshold: {err}"))?,
                )
            }
            "--neighborhood-km" => {
                neighborhood_km = value()?
                    .parse()
                    .map_err(|err| format!("invalid --neighborhood-km: {err}"))?
            }
            "--frames" => {
                let raw_value = value()?;
                if !raw_value.eq_ignore_ascii_case("all") {
                    frames = Some(
                        raw_value
                            .parse()
                            .map_err(|err| format!("invalid --frames: {err}"))?,
                    );
                }
            }
            "--nan-policy" => {
                nan_policy = NanPolicy::parse(&value()?).map_err(|err| err.to_string())?
            }
            "--pmm-tie-rule" => {
                tie_rule = PmmTieRule::parse(&value()?).map_err(|err| err.to_string())?
            }
            "--accept-status" => {
                accept_status = value()?
                    .split(',')
                    .map(|token| token.trim().to_string())
                    .filter(|token| !token.is_empty())
                    .collect();
            }
            "--width" => {
                width = value()?
                    .parse()
                    .map_err(|err| format!("invalid --width: {err}"))?
            }
            "--height" => {
                height = value()?
                    .parse()
                    .map_err(|err| format!("invalid --height: {err}"))?
            }
            "--source-label" => source_label = value()?,
            "--overlays" => overlays_path = Some(PathBuf::from(value()?)),
            "--annotate" => annotate_path = Some(PathBuf::from(value()?)),
            "--list-fields" => return Ok(Invocation::ListFields),
            "--abi" => return Ok(Invocation::Abi),
            "--help" | "-h" => return Ok(Invocation::Help),
            other => return Err(format!("unknown option {other}")),
        }
    }

    let mut members = explicit;
    if let Some(path) = &manifest {
        members.extend(load_manifest(path, &accept_status)?);
    }
    members.sort_by_key(|(number, _)| *number);
    members.dedup_by_key(|(number, _)| *number);
    if members.is_empty() {
        return Err("no members: pass --manifest FILE.json or one or more \
                    --member NUMBER=WRFOUT (a single frame) or \
                    NUMBER=MEMBER_DIR (its whole series)"
            .to_string());
    }
    if field_selector(&field).is_none() {
        return Err(format!(
            "unknown --field {field:?}; --list-fields prints the vocabulary"
        ));
    }
    let products: Vec<String> = products
        .split(',')
        .map(|token| token.trim().to_ascii_lowercase())
        .filter(|token| !token.is_empty())
        .collect();
    for product in &products {
        if !matches!(
            product.as_str(),
            "mean" | "spread" | "prob" | "pmm" | "paintball"
        ) {
            return Err(format!(
                "unknown product {product:?}; choose from mean, spread, prob, pmm, paintball"
            ));
        }
    }

    Ok(Invocation::Run(Box::new(Args {
        store_root: store_root.ok_or("--store-root is required")?,
        out_dir: out_dir.ok_or("--out-dir is required")?,
        members,
        field,
        products,
        threshold,
        neighborhood_km,
        frames,
        nan_policy,
        tie_rule,
        width,
        height,
        source_label,
        overlays: overlays_path
            .as_deref()
            .map(MapOverlays::load)
            .transpose()?,
        annotations: annotate_path
            .as_deref()
            .map(PanelAnnotations::load)
            .transpose()?,
    })))
}

/// `ensemble-manifest.json` (`gpuwm-ensemble-manifest.v1`) -> the roster.
///
/// The roster comes from the manifest and is never guessed from a
/// directory listing, because a member that failed to run leaves no
/// directory and would silently shrink the ensemble.  A member whose
/// status is not accepted is REPORTED with its status, so the operator
/// learns the right `--accept-status` from the refusal rather than from
/// documentation.
fn load_manifest(
    path: &Path,
    accept_status: &[String],
) -> Result<Vec<(u32, Vec<PathBuf>)>, String> {
    let bytes =
        std::fs::read(path).map_err(|err| format!("read manifest {}: {err}", path.display()))?;
    let document: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|err| format!("parse manifest {}: {err}", path.display()))?;
    let root = path.parent().unwrap_or(Path::new("."));
    let records = document
        .get("members")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            format!(
                "{}: 'members' must be a list of member records",
                path.display()
            )
        })?;
    let mut members = Vec::new();
    let mut refused: Vec<String> = Vec::new();
    let mut statuses_seen: BTreeMap<String, usize> = BTreeMap::new();
    for (index, record) in records.iter().enumerate() {
        let number = ["member", "id", "index", "number"]
            .iter()
            .find_map(|key| record.get(*key).and_then(serde_json::Value::as_u64))
            .map(|value| value as u32);
        let Some(number) = number else {
            refused.push(format!(
                "members[{index}]: no member number (member/id/index/number)"
            ));
            continue;
        };
        let status = record
            .get("status")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        *statuses_seen.entry(status.to_string()).or_default() += 1;
        if !accept_status.iter().any(|accepted| accepted == status) {
            refused.push(format!("member {number}: status {status:?}"));
            continue;
        }
        let directory = ["member_dir", "dir", "directory", "path"]
            .iter()
            .find_map(|key| record.get(*key).and_then(serde_json::Value::as_str))
            .map(|value| root.join(value))
            .unwrap_or_else(|| root.join(format!("member_{number:03}")));
        match member_wrfout_series(&directory) {
            Ok(wrfouts) => members.push((number, wrfouts)),
            Err(problem) => refused.push(format!("member {number}: {problem}")),
        }
    }
    if members.is_empty() {
        let found = statuses_seen
            .iter()
            .map(|(status, count)| format!("{status:?} x{count}"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "{}: no member survived the roster gate. Accepting {:?}; the \
             manifest carries {}. Problems: {}",
            path.display(),
            accept_status,
            if found.is_empty() {
                "no statuses".to_string()
            } else {
                found
            },
            refused.join("; ")
        ));
    }
    for problem in &refused {
        eprintln!("MEMBER_SKIPPED\t{problem}");
    }
    Ok(members)
}

/// One member's wrfout time series, in model-time order, or `Err` naming
/// the ambiguity.
///
/// A member directory holding two NESTS is not a member with two files,
/// it is two forecasts, and picking one of them by filename order would
/// average d01 for some members and d02 for others with nothing on the
/// transcript saying so.  The domain is therefore required to be
/// unambiguous, and the refusal names both domains it found.
///
/// Several files of ONE domain are a TIME SERIES, and all of them are the
/// member: `frames_per_outfile = 1` is WRF's default and what every
/// member this tree has prepped actually wrote, so a member's frames
/// normally arrive one per file.  Returning only the newest -- which is
/// what this did -- imported exactly the final valid time into the
/// member's store, so `--frames 0` silently rendered the last hour under
/// the name of the first and `--frames 1` refused with "its store has 1
/// frame(s)".  rw-store merges files written into ONE run, which is
/// precisely what a member's own frames are, so the whole series goes in
/// together and `--frames` indexes real valid times.
///
/// Ordered by the model timestamp in the filename rather than by raw
/// name, the same rule `local_import::parse_wrf_timestamp` exists to
/// serve; a file whose name carries no parsable stamp sorts after the
/// stamped ones, by name, rather than being dropped.
fn member_wrfout_series(directory: &Path) -> Result<Vec<PathBuf>, String> {
    let mut found: Vec<PathBuf> = Vec::new();
    let entries = std::fs::read_dir(directory)
        .map_err(|err| format!("read {}: {err}", directory.display()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("wrfout"))
                .unwrap_or(false)
        {
            found.push(path);
        }
    }
    let domains: std::collections::BTreeSet<String> = found
        .iter()
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?;
            let rest = name.strip_prefix("wrfout_d")?;
            let digits: String = rest.chars().take(2).collect();
            (digits.len() == 2 && digits.chars().all(|c| c.is_ascii_digit()))
                .then(|| format!("d{digits}"))
        })
        .collect();
    if domains.len() > 1 {
        return Err(format!(
            "{} holds more than one domain ({}); an ensemble product of \
             mixed nests is not an ensemble product. Point --member at the \
             file you mean",
            directory.display(),
            domains.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    if found.is_empty() {
        return Err(format!("no wrfout under {}", directory.display()));
    }
    found.sort_by_cached_key(|path| {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string();
        match wrf_model_timestamp(&name) {
            Some(stamp) => (0u8, stamp, name),
            None => (1u8, String::new(), name),
        }
    });
    Ok(found)
}

/// The sortable `YYYYMMDDHHMMSS` stamp inside a wrfout filename.
///
/// The same scan `local_import::parse_wrf_timestamp` runs, kept here
/// because that one is `pub(crate)` to the library target and this is a
/// binary: both accept the `_` and `:` time separators WRF and the
/// exact-time lane emit.
fn wrf_model_timestamp(name: &str) -> Option<String> {
    let bytes = name.as_bytes();
    if bytes.len() < 19 {
        return None;
    }
    for start in 0..=bytes.len().saturating_sub(19) {
        let slice = &name[start..start + 19];
        let chars = slice.as_bytes();
        let shaped = chars[4] == b'-'
            && chars[7] == b'-'
            && chars[10] == b'_'
            && matches!(chars[13], b':' | b'_')
            && matches!(chars[16], b':' | b'_')
            && chars
                .iter()
                .enumerate()
                .filter(|(index, _)| !matches!(index, 4 | 7 | 10 | 13 | 16))
                .all(|(_, byte)| byte.is_ascii_digit());
        if shaped {
            return Some(
                slice
                    .chars()
                    .filter(|character| character.is_ascii_digit())
                    .collect(),
            );
        }
    }
    None
}

/// One member's imported store plus the plane the requested field resolves
/// to in it.
struct MemberPlane {
    number: u32,
    values: Vec<f64>,
    ny: usize,
    nx: usize,
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(Invocation::Abi) => {
            println!("{ABI_MARKER}");
            ExitCode::SUCCESS
        }
        Ok(Invocation::Help) => {
            println!("{}", usage());
            ExitCode::SUCCESS
        }
        Ok(Invocation::ListFields) => {
            for spec in field_specs() {
                println!(
                    "FIELD\t{}\t{}\t{}\tdefault_threshold={}",
                    spec.name, spec.title, spec.units, spec.default_threshold
                );
            }
            println!("PRODUCTS\tmean\tspread\tprob\tpmm\tpaintball");
            ExitCode::SUCCESS
        }
        Ok(Invocation::Run(args)) => match run(*args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("{message}");
                ExitCode::FAILURE
            }
        },
        Err(message) => {
            eprintln!("{message}");
            eprintln!("{}", usage());
            ExitCode::from(2)
        }
    }
}

fn run(args: Args) -> Result<(), String> {
    let specs = field_specs();
    let spec = specs
        .iter()
        .find(|spec| spec.name == args.field)
        .ok_or_else(|| format!("unknown --field {}", args.field))?;
    let selector = field_selector(&args.field).expect("validated at parse");
    let threshold = args.threshold.unwrap_or(spec.default_threshold);

    // --- import every member into its OWN store -------------------------
    //
    // One store per member, never a shared one: rw-store merges files
    // written into one run, and merging two members would average a member
    // with itself and call it an ensemble.
    let mut planes: Vec<MemberPlane> = Vec::new();
    let mut geometry: Option<(Vec<f32>, Vec<f32>, Option<rustwx_core::GridProjection>)> = None;
    let mut style: Option<rustwx_products::viewer::StoreVariableStyle> = None;
    let mut member_stores: Vec<(u32, PathBuf, String, String, u16)> = Vec::new();

    for (number, wrfouts) in &args.members {
        let store_root = args.store_root.join(format!("member_{number:03}"));
        std::fs::create_dir_all(&store_root)
            .map_err(|err| format!("create {}: {err}", store_root.display()))?;
        // The member's WHOLE series in one import: rw-store merges files
        // written into one run, and a member's own frames are exactly
        // that.  One file per call would leave the store holding whichever
        // frame went last.
        let task = spawn_process_paths(
            wrfouts.clone(),
            store_root.clone(),
            WrfProcessOptions::default(),
        );
        let import = loop {
            match task
                .rx
                .recv()
                .map_err(|err| format!("member {number}: importer exited: {err}"))?
            {
                WrfProcessMessage::Progress(_) => {}
                WrfProcessMessage::Done(result) => break result?,
            }
        };
        let manifest_path = store_root
            .join(&import.model)
            .join(&import.run)
            .join("run.json");
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&manifest_path)
                .map_err(|err| format!("read {}: {err}", manifest_path.display()))?,
        )
        .map_err(|err| format!("parse {}: {err}", manifest_path.display()))?;
        let mut slots: Vec<u16> = manifest
            .get("hours")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| format!("{} has no hours object", manifest_path.display()))?
            .keys()
            .filter_map(|key| key.parse::<u16>().ok())
            .collect();
        slots.sort_unstable();
        let index = args.frames.unwrap_or(0);
        let slot = slots.get(index).copied().ok_or_else(|| {
            format!(
                "member {number}: --frames {index} out of range; its store has {} frame(s)",
                slots.len()
            )
        })?;
        member_stores.push((
            *number,
            store_root,
            import.model.clone(),
            import.run.clone(),
            slot,
        ));
    }

    for (number, store_root, model, run_slug, slot) in &member_stores {
        let source = StoreFieldSource::open(store_root, model, run_slug, *slot)
            .map_err(|err| format!("member {number}: open store: {err}"))?;
        let Some(variable) = source.resolve(&selector).map(str::to_string) else {
            return Err(format!(
                "member {number}: no stored variable carries {}; the field \
                 '{}' is not in this member's wrfout",
                selector.key(),
                args.field
            ));
        };
        let field = source
            .fetch(&selector)
            .map_err(|err| format!("member {number}: read {variable}: {err}"))?;
        let (ny, nx) = (field.grid.shape.ny, field.grid.shape.nx);
        if geometry.is_none() {
            geometry = Some((
                field.grid.lat_deg.clone(),
                field.grid.lon_deg.clone(),
                source.projection().cloned(),
            ));
            let meta = source
                .surface_variable(&variable)
                .ok_or_else(|| format!("member {number}: {variable} vanished from the store"))?
                .clone();
            let model_id: rustwx_core::ModelId = model
                .parse()
                .map_err(|err| format!("store model slug {model:?}: {err}"))?;
            style = rustwx_products::viewer::operational_style_for_store_variable(
                &variable,
                &meta.selector,
                &meta.units,
                model_id,
            )
            .or_else(|| {
                rustwx_products::viewer::curated_style_for_store_variable(
                    &variable,
                    &meta.selector,
                    &meta.units,
                    model_id,
                )
            });
        }
        let expected = geometry.as_ref().map(|(lat, _, _)| lat.len()).unwrap_or(0);
        if ny * nx != expected {
            return Err(format!(
                "member {number}: grid is {ny}x{nx} ({} point(s)); the first \
                 member's grid has {expected}. An ensemble of two grids is \
                 not an ensemble",
                ny * nx,
                ));
        }
        planes.push(MemberPlane {
            number: *number,
            values: field.values.iter().map(|value| f64::from(*value)).collect(),
            ny,
            nx,
        });
    }

    let (lat_deg, lon_deg, projection) =
        geometry.ok_or("no member produced a grid; nothing to render")?;
    let ny = planes[0].ny;
    let nx = planes[0].nx;
    let stack = MemberStack::new(
        ny,
        nx,
        planes
            .iter()
            .map(|plane| (plane.number, plane.values.clone()))
            .collect(),
    )
    .map_err(|err| err.to_string())?;

    let coverage = missingness_report(&stack, args.nan_policy).map_err(|err| err.to_string())?;
    println!(
        "MEMBERS n={} numbers={:?} grid={ny}x{nx}",
        stack.len(),
        stack.numbers()
    );
    println!(
        "COVERAGE nonfinite={} min_finite_members={} fully_missing_points={} coverage={:.4}",
        coverage.nonfinite_values,
        coverage.min_finite_members,
        coverage.fully_missing_points,
        coverage.coverage
    );

    // Neighbourhood radius in CELLS, from the request in kilometres and
    // the grid's own spacing.  A radius given in cells would mean a
    // different distance on every nest.
    //
    // The radius also reaches the FILENAME (`radius_token`): the CLI takes
    // a list of radii and drives this binary once per radius, so a slug
    // that omitted it made `--neighborhood-km 0,9,27` write one file three
    // times and report three panels.
    let dx_km = grid_spacing_km(&lat_deg, &lon_deg, ny, nx);
    let radius_cells = if args.neighborhood_km > 0.0 && dx_km > 0.0 {
        args.neighborhood_km / dx_km
    } else {
        0.0
    };

    let domain_token = format!("ens{:02}", stack.len());
    let valid_day = "ensemble".to_string();
    let mut rendered = 0usize;
    let mut failed = 0usize;

    for product in &args.products {
        let outcome = render_product(
            product,
            &args,
            spec,
            &stack,
            &coverage,
            threshold,
            radius_cells,
            &lat_deg,
            &lon_deg,
            projection.as_ref(),
            style.as_ref(),
            &domain_token,
            &valid_day,
        );
        match outcome {
            Ok(Some(path)) => {
                rendered += 1;
                println!("RENDERED {product} {}", path.display());
            }
            Ok(None) => println!("SKIPPED {product} not requested for this field"),
            Err(message) => {
                failed += 1;
                eprintln!("FAILED {product} {message}");
            }
        }
    }
    println!("FINISHED rendered={rendered} failed={failed}");
    if rendered == 0 || failed > 0 {
        return Err(format!(
            "ensemble batch incomplete: rendered={rendered} failed={failed}"
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_product(
    product: &str,
    args: &Args,
    spec: &FieldSpec,
    stack: &MemberStack,
    coverage: &rustwx_ensemble::MissingnessReport,
    threshold: f64,
    radius_cells: f64,
    lat_deg: &[f32],
    lon_deg: &[f32],
    projection: Option<&rustwx_core::GridProjection>,
    style: Option<&rustwx_products::viewer::StoreVariableStyle>,
    domain_token: &str,
    valid_day: &str,
) -> Result<Option<PathBuf>, String> {
    let members = stack.len();
    let legend = LegendControls {
        density: LevelDensity::default(),
        mode: LegendMode::Stepped,
    };
    let field_ladder = || match style {
        Some(style) => (style.scale.clone(), style.display_units.clone()),
        // No operational ladder resolved: say so rather than inventing
        // one.  A generic ramp under a reflectivity title would be read
        // as the NWS table by anyone who has seen the deterministic panel.
        None => (
            scales::spread_scale(threshold * 2.0),
            spec.units.to_string(),
        ),
    };

    // The subtitle row has three slots and the renderer centres the middle
    // one on the PANEL, not in the gap between the other two -- so a long
    // centre string is drawn over both of them.  Everything long therefore
    // rides in the LEFT slot (which has the row's width budget and its own
    // ellipsis) and the centre stays empty.  The right slot is the
    // provenance stamp and is never competed with.
    let (values, title, units, scale, subtitle_left, contours, slug, colorbar) = match product {
        "mean" => {
            let mean = ensemble_mean(stack, args.nan_policy).map_err(|e| e.to_string())?;
            let (scale, units) = field_ladder();
            (
                to_f32(&mean),
                format!("Ensemble mean {} ({members} members)", spec.title),
                units,
                scale,
                coverage_line(coverage, args, members),
                Vec::new(),
                format!("ens_mean_{}", spec.name),
                true,
            )
        }
        "spread" => {
            let spread = ensemble_spread(stack, 1, args.nan_policy).map_err(|e| e.to_string())?;
            let upper = finite_max(&spread).unwrap_or(1.0);
            (
                to_f32(&spread),
                format!(
                    "Ensemble spread (1 sigma, ddof=1) {} ({members} members)",
                    spec.title
                ),
                spec.units.to_string(),
                scales::spread_scale(upper),
                coverage_line(coverage, args, members),
                Vec::new(),
                format!("ens_spread_{}", spec.name),
                true,
            )
        }
        "prob" => {
            let probability =
                exceedance_probability(stack, threshold, radius_cells, args.nan_policy)
                    .map_err(|e| e.to_string())?;
            let neighborhood = if radius_cells > 0.0 {
                format!(" within {:.0} km", args.neighborhood_km)
            } else {
                String::new()
            };
            (
                to_f32(&probability),
                format!(
                    "P({} > {threshold} {}){neighborhood} ({members} members)",
                    spec.title, spec.units
                ),
                "fraction".to_string(),
                scales::probability_scale(),
                coverage_line(coverage, args, members),
                Vec::new(),
                format!(
                    "ens_prob_{}_p{}{}{}",
                    spec.name,
                    format!("{threshold}").replace(['.', '-'], "_"),
                    spec.unit_slug,
                    radius_token(radius_cells, args.neighborhood_km)
                ),
                true,
            )
        }
        "pmm" => {
            let pmm = probability_matched_mean(stack, args.nan_policy, args.tie_rule)
                .map_err(|e| e.to_string())?;
            let ties = pmm_tie_report(stack, args.nan_policy).map_err(|e| e.to_string())?;
            println!(
                "TIES tied_points={} largest_group={} tied_fraction={:.4} rule={}",
                ties.tied_points,
                ties.largest_tie_group,
                ties.tied_fraction,
                args.tie_rule.as_str()
            );
            let (scale, units) = field_ladder();
            let note = format!(
                "PMM tie rule {}; {:.1}% of the mean field is plateau (largest group {})",
                args.tie_rule.as_str(),
                100.0 * ties.tied_fraction,
                ties.largest_tie_group
            );
            (
                to_f32(&pmm),
                format!(
                    "Probability-matched mean {} ({members} members)",
                    spec.title
                ),
                units,
                scale,
                match coverage.caption() {
                    Some(caption) => format!("{note} | {caption}"),
                    None => note,
                },
                Vec::new(),
                format!("ens_pmm_{}", spec.name),
                true,
            )
        }
        "paintball" => {
            // Not a plane: N contour layers, one per member, one level,
            // colour keyed to the member NUMBER.  A blank fill underneath
            // so the map, basemap and chrome are the panel's own.
            let mut contours = Vec::with_capacity(members);
            for (number, grid) in stack.members() {
                contours.push(ContourLayer {
                    data: grid
                        .iter()
                        .map(|value| if value.is_finite() { *value as f32 } else { f32::NAN })
                        .collect(),
                    levels: vec![threshold],
                    color: parse_color(member_color(number)),
                    width: 2,
                    labels: false,
                    show_extrema: false,
                    pattern: Default::default(),
                    major_every: None,
                    major_width: None,
                });
            }
            let roster = stack
                .numbers()
                .iter()
                .map(|number| format!("{number}={}", member_color(*number)))
                .collect::<Vec<_>>()
                .join(" ");
            println!("PAINTBALL_LEGEND\t{roster}");
            (
                vec![f32::NAN; stack.points()],
                format!(
                    "Paintball {} > {threshold} {} ({members} members)",
                    spec.title, spec.units
                ),
                spec.units.to_string(),
                scales::spread_scale(1.0),
                // The roster goes on stdout (PAINTBALL_LEGEND), not into
                // the subtitle: 30 member/colour pairs do not fit a
                // subtitle slot and a truncated legend is a wrong legend.
                format!("{members} members | contour at {threshold} {} | member colours on stdout", spec.units),
                contours,
                format!("ens_paintball_{}", spec.name),
                // A paintball fill is all-NaN by construction; a colourbar
                // beside it invites reading a value off an empty scale.
                false,
            )
        }
        other => return Err(format!("unknown product {other}")),
    };

    let subtitle_right = format!("source: {} | {EXPERIMENTAL_STAMP}", args.source_label);
    let out_path = layout_path(
        &args.out_dir,
        domain_token,
        &slug,
        valid_day,
        &format!("arwen_{}_{}", safe_component(&slug, "product"), members),
    );
    let path = render_panel(PanelRequest {
        lat_deg,
        lon_deg,
        projection,
        ny: stack.ny(),
        nx: stack.nx(),
        values,
        product_slug: slug.clone(),
        title,
        display_units: units,
        scale,
        cbar_tick_step: None,
        legend,
        render_density: RenderDensity::default(),
        subtitle_left,
        subtitle_center: None,
        subtitle_right,
        width: args.width,
        height: args.height,
        contours,
        colorbar,
        overlays: args.overlays.as_ref(),
        annotations: args.annotations.as_ref(),
        out_path,
    })?;
    Ok(Some(path))
}

/// The left subtitle: the coverage stamp when anything was masked, and the
/// plain roster line when nothing was.
///
/// The stamp is not optional decoration.  A masked reduction that does not
/// publish its denominator is the thing the propagate-NaN policy was right
/// to refuse; publishing it is what makes masking honest, and the panel is
/// where a reader will see it.
fn coverage_line(
    coverage: &rustwx_ensemble::MissingnessReport,
    args: &Args,
    members: usize,
) -> String {
    coverage.caption().unwrap_or_else(|| {
        format!(
            "{members} members, all finite | nan-policy {}",
            args.nan_policy.as_str()
        )
    })
}

fn to_f32(values: &[f64]) -> Vec<f32> {
    values.iter().map(|value| *value as f32).collect()
}

fn finite_max(values: &[f64]) -> Option<f64> {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(None, |best: Option<f64>, value| {
            Some(match best {
                None => value,
                Some(best) => best.max(value),
            })
        })
}

/// The filename token for a neighbourhood radius: `9` -> `_r9km`.
///
/// Empty when no neighbourhood was applied, which is the same statement
/// `gpuwm.da.enprod.radius_slug` makes and for the same reason: an
/// `r0km` on a point probability claims a neighbourhood of zero radius
/// was applied, which is true and useless.  The spelling is that
/// function's, so the two engines' filenames read as one grammar.
///
/// `radius_cells` rather than the kilometres decides, because that is
/// what the reduction actually used: a radius smaller than a cell is a
/// point probability whatever the request said.
fn radius_token(radius_cells: f64, neighborhood_km: f64) -> String {
    if radius_cells <= 0.0 {
        return String::new();
    }
    // `{}` on f64 is Rust's shortest round-trip form, so 9.0 prints as
    // `9` and 7.5 as `7.5`; the two substitutions are the Python
    // function's, so `.` cannot be misread as an extension and a leading
    // `-` cannot be misread as the next token.
    let number = format!("{neighborhood_km}").replace('-', "m").replace('.', "p");
    format!("_r{number}km")
}

/// Mean cell spacing in kilometres, from the grid's own coordinates.
///
/// Measured rather than read from a global attribute because the store
/// does not retain one (rw-store v1 keeps no grid spacing), and a
/// neighbourhood radius stated in kilometres has to become cells somehow.
/// A degenerate grid returns 0, which turns the radius off rather than
/// dividing by it.
fn grid_spacing_km(lat_deg: &[f32], lon_deg: &[f32], ny: usize, nx: usize) -> f64 {
    if ny < 2 || nx < 2 || lat_deg.len() != ny * nx {
        return 0.0;
    }
    let a = (f64::from(lat_deg[0]), f64::from(lon_deg[0]));
    let b = (f64::from(lat_deg[1]), f64::from(lon_deg[1]));
    let mean_lat = ((a.0 + b.0) / 2.0).to_radians();
    let dy_km = (b.0 - a.0) * 111.195;
    let dx_km = (b.1 - a.1) * 111.195 * mean_lat.cos();
    let spacing = (dx_km * dx_km + dy_km * dy_km).sqrt();
    if spacing.is_finite() && spacing > 0.0 {
        spacing
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "rw-ensbatch-test-{tag}-{}-{:?}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            ));
            std::fs::create_dir_all(&path).expect("scratch dir");
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_member(root: &Path, member_dir: &str, names: &[&str]) {
        let dir = root.join(member_dir);
        std::fs::create_dir_all(&dir).expect("member dir");
        for name in names {
            std::fs::write(dir.join(name), b"not a real netcdf").expect("wrfout stub");
        }
    }

    fn write_manifest(root: &Path, records: &str) -> PathBuf {
        let path = root.join("ensemble-manifest.json");
        std::fs::write(
            &path,
            format!("{{\"schema\":\"gpuwm-ensemble-manifest.v1\",\"members\":[{records}]}}"),
        )
        .expect("manifest");
        path
    }

    fn accepted() -> Vec<String> {
        DEFAULT_ACCEPT_STATUS.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_member_written_one_frame_per_file_ingests_every_frame() {
        // The defect: the roster kept the LAST wrfout in a member
        // directory, so a run written with frames_per_outfile=1 -- WRF's
        // default, and what every gefs/aigefs member produced -- imported
        // exactly its final valid time.  `--frames 0` then rendered hour
        // N while calling itself frame 0, and `--frames 1` refused with
        // "its store has 1 frame(s)".
        let scratch = Scratch::new("series");
        let root = scratch.0.as_path();
        write_member(
            root,
            "member_000",
            &[
                "wrfout_d01_2026-08-17_02_00_00",
                "wrfout_d01_2026-08-17_00_00_00",
                "wrfout_d01_2026-08-17_01_00_00",
            ],
        );
        let manifest = write_manifest(root, "{\"member\":0,\"status\":\"DONE\"}");

        let members = load_manifest(&manifest, &accepted()).expect("roster");

        assert_eq!(members.len(), 1);
        let (number, files) = &members[0];
        assert_eq!(*number, 0);
        let names: Vec<String> = files
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            vec![
                "wrfout_d01_2026-08-17_00_00_00".to_string(),
                "wrfout_d01_2026-08-17_01_00_00".to_string(),
                "wrfout_d01_2026-08-17_02_00_00".to_string(),
            ],
            "a member's frames enter the store in model-time order, all of them"
        );
    }

    #[test]
    fn a_member_holding_two_nests_is_still_refused_by_name() {
        // The refusal the one-file rule was protecting stays: averaging
        // d01 for one member against d02 for another is not an ensemble.
        let scratch = Scratch::new("nests");
        let root = scratch.0.as_path();
        write_member(
            root,
            "member_000",
            &[
                "wrfout_d01_2026-08-17_00_00_00",
                "wrfout_d02_2026-08-17_00_00_00",
            ],
        );
        let manifest = write_manifest(root, "{\"member\":0,\"status\":\"DONE\"}");

        let err = load_manifest(&manifest, &accepted()).expect_err("mixed nests refuse");

        assert!(err.contains("d01"), "{err}");
        assert!(err.contains("d02"), "{err}");
    }

    #[test]
    fn an_unaccepted_status_is_reported_with_its_status() {
        let scratch = Scratch::new("status");
        let root = scratch.0.as_path();
        write_member(root, "member_000", &["wrfout_d01_2026-08-17_00_00_00"]);
        let manifest = write_manifest(root, "{\"member\":0,\"status\":\"RUNNING\"}");

        let err = load_manifest(&manifest, &accepted()).expect_err("RUNNING is not averaged");

        assert!(err.contains("RUNNING"), "{err}");
    }

    #[test]
    fn an_explicit_member_may_name_a_directory_of_frames() {
        // The manifest route and the explicit route describe the same
        // thing, so they accept the same thing: one file, or the member
        // directory that holds its frames.
        let scratch = Scratch::new("explicit");
        let root = scratch.0.as_path();
        write_member(
            root,
            "member_007",
            &[
                "wrfout_d01_2026-08-17_00_00_00",
                "wrfout_d01_2026-08-17_01_00_00",
            ],
        );

        let files = member_wrfout_series(&root.join("member_007")).expect("series");

        assert_eq!(files.len(), 2);
    }
}
