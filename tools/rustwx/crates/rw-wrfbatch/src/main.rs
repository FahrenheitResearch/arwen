//! Headless raw-WRF import plus production Rusty Weather batch rendering.
//!
//! This intentionally reuses the UI crate's hardened WRF processing modules;
//! it only replaces the egui orchestration with a bounded command-line job.
//!
//! gpuwm adaptations over the source workspace's rw_wrf_batch bin:
//! every stored frame renders by default (`--frames all`), a single frame
//! is selectable by stored slot (`--frames N`), and the work limits are
//! sized to the request instead of the GUI's one-hour click ceilings --
//! which is what lets exact-time (sub-hourly) imports render every frame.
//!
//! It also stamps the run's own identity onto every output.  The upstream
//! filename carried model, init cycle, forecast hour, and the generic
//! `native_grid` slug -- which is identical for every nest of one run, so
//! rendering two domains at one lead into one directory silently
//! overwrote one forecast with the other.  The inputs' `GRID_ID` and `DX`
//! become a `d02-3km` domain token in the slug position, and the same
//! spacing appears as `Δx 3 km` in the plot subtitle.  Provenance is the
//! local model (`--source-label`, default `ArWen`) rather than the GDEX
//! source inherited from the `wrf` store-model identity.

#[path = "grib_import.rs"]
mod grib_import;
#[path = "local_import.rs"]
mod local_import;
#[path = "postproc_severe.rs"]
mod postproc_severe;
#[path = "wrf_process.rs"]
mod wrf_process;
#[path = "wrf_volumes.rs"]
mod wrf_volumes;

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from, embedded so the gpuwm release cut can prove a
/// staged bridge matches the commit being released by reading bytes
/// alone (`tools/build_bridge_bundle.py pin --source-rev`).  `build.rs`
/// injects the value; `main` references the constant so the linker
/// cannot discard it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

use rustwx_products::shared_context::TitleProvenance;
use rusty_weather::batch_render::{
    BatchHourScope, BatchRenderDomain, BatchRenderEvent, BatchRenderLimits, BatchRenderRequest,
    inspect_renderable_products, run_batch_render,
};
use wrf_process::{WrfProcessMessage, WrfProcessOptions, spawn_process_paths};

/// Provenance label for a locally-imported run.  The store model slug is
/// `wrf`, whose only registered fetch source is GDEX -- a label this lane
/// never fetched from.  `--source-label` renames it for stock-WRF files,
/// which ArWen did not produce and must not claim.
const DEFAULT_SOURCE_LABEL: &str = "ArWen";

#[derive(Debug)]
struct Args {
    store_root: PathBuf,
    out_dir: PathBuf,
    products: String,
    frames: Option<usize>,
    width: u32,
    height: u32,
    heavy: bool,
    list_products: bool,
    source_label: String,
    inputs: Vec<PathBuf>,
}

fn usage() -> &'static str {
    "usage: rw_wrfbatch --store-root DIR --out-dir DIR [--products all|SLUGS] \
[--frames all|N] [--width N] [--height N] [--heavy] [--source-label TEXT] \
[--list-products] wrfout..."
}

/// What went wrong, and therefore what the user should be shown.
///
/// The three arms exist because the old single `String` channel made every
/// failure look identical: the message, then the usage line, then exit 1 --
/// so "unknown product 'x'" and "the store has no hours object" were
/// reported the same way, and the usage line was the LAST stderr line, which
/// is exactly the line `gpuwm/rustwx.py` surfaces as the reason.
///
/// `Help` also replaces the previous `error == usage()` string-equality
/// test.  That test decided the exit code of `--help` by comparing the
/// error text to the usage text, so any future error that happened to equal
/// the usage string would have exited 0.  The observable contract is
/// unchanged -- `--help` still prints exactly `usage()` and still exits 0,
/// which is what `gpuwm.rustwx.probe_renderer` requires to declare the rust
/// engine usable at all.
enum CliError {
    /// `--help`/`-h`: print the usage line, succeed.
    Help,
    /// The command line is wrong.  Message, then the usage line, exit 2 --
    /// the code `gpuwm render`'s matplotlib engine uses for the same class
    /// of mistake (`gpuwm/render.py` raises on an unknown product and the
    /// CLI turns that into 2).
    Usage(String),
    /// The command line was fine and the work failed.  Message only: the
    /// usage line says nothing about a store that will not open, and
    /// printing it buries the sentence that does.
    Failed(String),
}

impl From<String> for CliError {
    fn from(message: String) -> Self {
        CliError::Usage(message)
    }
}

impl From<&str> for CliError {
    fn from(message: &str) -> Self {
        CliError::Usage(message.to_string())
    }
}

/// A parsed command line.  `--list-products` with no inputs asks a question
/// about this build, not about a store, so it does not need `--store-root`,
/// `--out-dir` or a wrfout to answer.
enum Invocation {
    Batch(Box<Args>),
    /// Print the product vocabulary and exit.
    Catalog,
}

/// The static product vocabulary, for `--list-products` with no inputs.
///
/// The store-aware listing (which of these the imported frames can actually
/// render, and why not) still needs a store and is unchanged; this answers
/// the other question -- "what may I put in --products?" -- which is the one
/// a user has after an unknown-product refusal.
fn print_product_catalog() -> Result<(), CliError> {
    let slugs = rusty_weather::render_all::known_product_slugs();
    println!("group keywords: all, direct, derived, heavy, windowed");
    for slug in &slugs {
        // Deliberately NOT the `PRODUCT\t...` record the store-aware listing
        // emits: that one is parsed by gpuwm/rustwx.py as five tab-separated
        // fields, and a same-prefixed two-field line is a trap.
        println!("  {slug}");
    }
    // Not "total=": the store-aware listing already owns that word for its
    // own count of catalog ROWS (which includes rows no --products spelling
    // selects).  This is the size of the --products vocabulary.
    println!("selectable_slugs={}", slugs.len());
    println!(
        "pass --store-root DIR --out-dir DIR wrfout... with --list-products \
         for per-frame availability"
    );
    Ok(())
}

fn parse_args() -> Result<Invocation, CliError> {
    let mut store_root = None;
    let mut out_dir = None;
    let mut products = "all".to_string();
    let mut frames = None;
    let mut width = 1_200u32;
    let mut height = 900u32;
    let mut heavy = false;
    let mut list_products = false;
    let mut source_label = DEFAULT_SOURCE_LABEL.to_string();
    let mut inputs = Vec::new();
    let mut raw = std::env::args().skip(1);

    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "--store-root" => {
                store_root = Some(PathBuf::from(
                    raw.next().ok_or("--store-root requires a directory")?,
                ));
            }
            "--out-dir" => {
                out_dir = Some(PathBuf::from(
                    raw.next().ok_or("--out-dir requires a directory")?,
                ));
            }
            "--products" => {
                products = raw.next().ok_or("--products requires a value")?;
            }
            "--frames" => {
                let value = raw.next().ok_or("--frames requires 'all' or an index")?;
                if !value.eq_ignore_ascii_case("all") {
                    frames = Some(
                        value
                            .parse::<usize>()
                            .map_err(|err| format!("invalid --frames: {err}"))?,
                    );
                }
            }
            "--width" => {
                width = raw
                    .next()
                    .ok_or("--width requires a value")?
                    .parse()
                    .map_err(|err| format!("invalid --width: {err}"))?;
            }
            "--height" => {
                height = raw
                    .next()
                    .ok_or("--height requires a value")?
                    .parse()
                    .map_err(|err| format!("invalid --height: {err}"))?;
            }
            "--source-label" => {
                let value = raw.next().ok_or("--source-label requires a value")?;
                if value.trim().is_empty() {
                    return Err(CliError::Usage(
                        "--source-label must not be blank".to_string(),
                    ));
                }
                source_label = value.trim().to_string();
            }
            "--heavy" => heavy = true,
            "--list-products" => list_products = true,
            "--help" | "-h" => return Err(CliError::Help),
            _ if arg.starts_with('-') => {
                return Err(CliError::Usage(format!("unknown option {arg}")));
            }
            _ => inputs.push(PathBuf::from(arg)),
        }
    }

    // Asking what this build can render is answerable without a store, a
    // destination or an input; demanding all three first is why the field
    // report saw --list-products answer with the usage line.
    if list_products && inputs.is_empty() && store_root.is_none() && out_dir.is_none() {
        return Ok(Invocation::Catalog);
    }

    if inputs.is_empty() {
        return Err(CliError::Usage(
            "at least one wrfout input is required".to_string(),
        ));
    }
    Ok(Invocation::Batch(Box::new(Args {
        store_root: store_root.ok_or("--store-root is required")?,
        out_dir: out_dir.ok_or("--out-dir is required")?,
        products,
        frames,
        width,
        height,
        heavy,
        list_products,
        source_label,
        inputs,
    })))
}

/// Refuse a command line this build cannot serve, before any file is opened.
///
/// Both checks used to happen deep in the work: `--products` was validated
/// only after a full wrfout import, so a typo cost an import and then
/// reported `No supported WRF files selected` -- the wrong sentence
/// entirely.  A path that does not exist, or that is not a WRF file, was
/// reported without ever naming the path.
fn validate_request(args: &Args) -> Result<(), CliError> {
    rusty_weather::render_all::partition_products(&args.products)
        .map_err(|err| CliError::Usage(err.to_string()))?;

    // Per-path, in the order given, so the first sentence names the first
    // problem.  The wording matches `gpuwm/render.py`'s matplotlib engine
    // ("{path}: unreadable wrfout ({exc})") so `--engine auto` cannot change
    // what a script reads.
    let mut unreadable = Vec::new();
    for path in &args.inputs {
        if !path.exists() {
            unreadable.push(format!("{}: unreadable wrfout (no such file)", path.display()));
        } else if !path.is_file() {
            unreadable.push(format!(
                "{}: unreadable wrfout (not a regular file)",
                path.display()
            ));
        } else if !wrf_process::is_supported_wrf_file(path) {
            unreadable.push(format!(
                "{}: unreadable wrfout (not a raw WRF or post-processed NetCDF \
                 file this build recognises)",
                path.display()
            ));
        }
    }
    // Existing semantics: a mixed set renders the files that work.  Only a
    // set with nothing usable in it is a refusal -- but it now says which
    // paths, and why each one, instead of "No supported WRF files selected".
    if unreadable.len() == args.inputs.len() {
        return Err(CliError::Failed(unreadable.join("\n")));
    }
    Ok(())
}

/// The run's own grid identity, read from the inputs' WRF global
/// attributes: the nest number (`GRID_ID`, else a `wrfout_dNN` filename)
/// and the horizontal spacing (`DX`, metres).
#[derive(Debug, Default, PartialEq)]
struct GridIdentity {
    domain: Option<String>,
    spacing_m: Option<f64>,
}

/// One identity for the whole invocation, or nothing.
///
/// Several inputs import into ONE store and render as one run, so a
/// per-file token would be a lie on the shared output. Inputs that
/// disagree therefore yield no token at all: the generic `native_grid`
/// slug is honest about a mixed run in a way that `d02` would not be.
fn grid_identity(inputs: &[PathBuf]) -> GridIdentity {
    let mut identity: Option<GridIdentity> = None;
    for path in inputs {
        let found = file_grid_identity(path);
        match &identity {
            None => identity = Some(found),
            Some(first) if *first == found => {}
            Some(_) => return GridIdentity::default(),
        }
    }
    identity.unwrap_or_default()
}

fn file_grid_identity(path: &std::path::Path) -> GridIdentity {
    let file = wrf_core::WrfFile::open(path).ok();
    let domain = file
        .as_ref()
        .and_then(|file| file.global_attr_i32("GRID_ID").ok())
        .and_then(domain_token_from_id)
        .or_else(|| domain_token_from_filename(path));
    let spacing_m = file
        .as_ref()
        .and_then(|file| file.global_attr_f64("DX").ok())
        .filter(|value| value.is_finite() && *value > 0.0);
    GridIdentity { domain, spacing_m }
}

fn domain_token_from_id(grid_id: i32) -> Option<String> {
    (1..=99).contains(&grid_id).then(|| format!("d{grid_id:02}"))
}

/// `wrfout_d02_1974-04-03_18:00:00` -> `d02`, for files whose `GRID_ID`
/// is absent (idealized and hand-built wrfouts).
fn domain_token_from_filename(path: &std::path::Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let rest = name.strip_prefix("wrfout_d")?;
    let digits: String = rest.chars().take(2).collect();
    if digits.len() != 2 || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(format!("d{digits}"))
}

/// `3km`, `1.5km`, `333m` -- the resolution half of the output token.
/// Sub-kilometre nests read as integer metres because `0.333km` is a
/// worse label for a 333 m nest than `333m` is.
fn resolution_token(spacing_m: f64) -> Option<String> {
    let (value, unit) = spacing_parts(spacing_m)?;
    Some(format!("{value}{unit}"))
}

/// `Δx 3 km`, `Δx 333 m` -- the same number, spelled for the subtitle.
fn spacing_subtitle(spacing_m: f64) -> Option<String> {
    let (value, unit) = spacing_parts(spacing_m)?;
    Some(format!("\u{0394}x {value} {unit}"))
}

fn spacing_parts(spacing_m: f64) -> Option<(String, &'static str)> {
    if !spacing_m.is_finite() || spacing_m <= 0.0 {
        return None;
    }
    let metres = spacing_m.round();
    if metres >= 1_000.0 {
        Some((trimmed_number(spacing_m / 1_000.0), "km"))
    } else {
        Some((format!("{metres:.0}"), "m"))
    }
}

/// Three decimals at most, with the trailing zeros dropped: `3`, `1.5`,
/// `1.333`.
fn trimmed_number(value: f64) -> String {
    let text = format!("{value:.3}");
    let trimmed = text.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

/// The slug that replaces `native_grid` in every output filename:
/// `d02-3km`, or `d02` when the file declares no usable `DX`.
fn native_domain_slug(identity: &GridIdentity) -> Option<String> {
    let domain = identity.domain.as_ref()?;
    match identity.spacing_m.and_then(resolution_token) {
        Some(resolution) => Some(format!("{domain}-{resolution}")),
        None => Some(domain.clone()),
    }
}

/// `d02 750 m` -- the same identity as the filename slug, spelled for a
/// human reading the plot's headline.  The spacing is spelled the way the
/// subtitle spells it, so one grid is not two different numbers on one
/// image.
///
/// This is what the headline's parenthetical is FOR.  Without it the
/// renderer fell back to the store model's registered source catalog and
/// stamped that catalog's dataset id -- a dataset this lane never read,
/// naming an archive these numbers did not come from, on every frame.
fn domain_title_label(identity: &GridIdentity) -> Option<String> {
    let domain = identity.domain.as_ref()?;
    match identity.spacing_m.and_then(spacing_parts) {
        Some((value, unit)) => Some(format!("{domain} {value} {unit}")),
        None => Some(domain.clone()),
    }
}

fn run(args: Args) -> Result<(), String> {
    let options = WrfProcessOptions {
        heavy_ecape: args.heavy,
        ..WrfProcessOptions::default()
    };
    // Read before the import consumes the paths: the domain token and the
    // subtitle spacing come from the inputs' own global attributes, never
    // from the store (rw-store v1 retains no grid-spacing metadata).
    let identity = grid_identity(&args.inputs);
    let domain_slug = native_domain_slug(&identity);
    let spacing = identity.spacing_m.and_then(spacing_subtitle);
    let title_provenance = TitleProvenance::LocalImport {
        grid_label: domain_title_label(&identity),
    };
    let task = spawn_process_paths(args.inputs, args.store_root.clone(), options);
    let import = loop {
        match task
            .rx
            .recv()
            .map_err(|err| format!("WRF processor exited without a result: {err}"))?
        {
            WrfProcessMessage::Progress(message) => println!("PROCESS {message}"),
            WrfProcessMessage::Done(result) => break result?,
        }
    };
    println!(
        "IMPORTED model={} run={} files={} hours={} variables={} notes={}",
        import.model,
        import.run,
        import.files_seen,
        import.hours_written,
        import.variables.len(),
        import.notes.len()
    );

    let run_manifest = args
        .store_root
        .join(&import.model)
        .join(&import.run)
        .join("run.json");
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&run_manifest)
            .map_err(|err| format!("read {}: {err}", run_manifest.display()))?,
    )
    .map_err(|err| format!("parse {}: {err}", run_manifest.display()))?;
    let stored_slots: Vec<u16> = {
        let mut slots: Vec<u16> = manifest
            .get("hours")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| format!("{} has no hours object", run_manifest.display()))?
            .keys()
            .filter_map(|key| key.parse::<u16>().ok())
            .collect();
        slots.sort_unstable();
        slots
    };
    let first_slot = stored_slots
        .first()
        .copied()
        .ok_or_else(|| format!("{} has no stored forecast slots", run_manifest.display()))?;
    if args.list_products {
        return list_products(
            &args.store_root,
            &import.model,
            &import.run,
            &stored_slots,
            args.heavy,
        );
    }

    // --frames N is an ordinal index into the ascending stored slots (the
    // Nth stored frame), matching `gpuwm render --timeidx`: within one
    // imported file the plan orders slots by valid time, so the index is
    // the frame index whether the axis is whole-hour (slot = forecast
    // hour) or exact-time (slot = ordinal).
    let hour_scope = match args.frames {
        None => BatchHourScope::AllStored,
        Some(index) => {
            let slot = stored_slots.get(index).copied().ok_or_else(|| {
                format!(
                    "--frames {index} out of range; the store has {} frame(s)",
                    stored_slots.len()
                )
            })?;
            BatchHourScope::Current(slot)
        }
    };
    let catalog =
        inspect_renderable_products(&args.store_root, &import.model, &import.run, first_slot)?;
    let product_spec = if args.products.eq_ignore_ascii_case("all") {
        catalog
            .products
            .iter()
            .map(|product| product.slug.as_str())
            .collect::<Vec<_>>()
            .join(",")
    } else {
        args.products
    };
    println!(
        "CATALOG products={} stored_hours={:?}",
        catalog.products.len(),
        catalog.stored_hours
    );

    // Size the limits to the request: this is a command-line job whose
    // work is exactly frames x products, not a GUI guarding against an
    // accidental unbounded click.
    let selected_frames = match hour_scope {
        BatchHourScope::AllStored => stored_slots.len().max(1),
        BatchHourScope::Current(_) => 1,
    };
    let per_frame_products = product_spec
        .split(',')
        .filter(|slug| !slug.trim().is_empty())
        .count();
    let mut limits = BatchRenderLimits::default();
    limits.max_hours = limits.max_hours.max(selected_frames);
    limits.max_products_per_hour = limits.max_products_per_hour.max(per_frame_products);
    limits.max_work_items = limits.max_work_items.max(
        selected_frames
            .saturating_mul(per_frame_products)
            .saturating_add(per_frame_products),
    );
    limits.max_output_width = args.width.max(limits.max_output_width);
    limits.max_output_height = args.height.max(limits.max_output_height);
    limits.max_output_pixels = u64::from(args.width) * u64::from(args.height);
    let request = BatchRenderRequest {
        store_root: args.store_root,
        model_slug: import.model,
        run_slug: import.run,
        hours: hour_scope,
        product_spec,
        out_dir: args.out_dir,
        domain: BatchRenderDomain::NativeGrid,
        native_domain_slug: domain_slug,
        subtitle_spacing: spacing,
        source_label: Some(args.source_label),
        title_provenance,
        date_yyyymmdd: None,
        cycle_utc: None,
        source: None,
        output_width: args.width,
        output_height: args.height,
        limits,
    };
    let cancel = AtomicBool::new(false);
    let summary = run_batch_render(request, &cancel, |event| match event {
        BatchRenderEvent::Started {
            planned_items,
            output_dir,
            ..
        } => println!(
            "RENDER planned={planned_items} out={}",
            output_dir.display()
        ),
        BatchRenderEvent::ItemRendered {
            slug, output_path, ..
        } => println!("RENDERED {slug} {}", output_path.display()),
        BatchRenderEvent::ItemSkipped { slug, reason, .. } => {
            println!("SKIPPED {slug} {reason}")
        }
        BatchRenderEvent::ItemFailed { slug, error, .. } => {
            eprintln!("FAILED {slug} {error}")
        }
        BatchRenderEvent::Finished(summary) => println!(
            "FINISHED rendered={} skipped={} failed={} elapsed_ms={}",
            summary.rendered, summary.skipped, summary.failed, summary.elapsed_ms
        ),
        _ => {}
    })?;
    if summary.rendered == 0 || summary.failed > 0 {
        return Err(format!(
            "batch render incomplete: rendered={} skipped={} failed={}",
            summary.rendered, summary.skipped, summary.failed
        ));
    }
    Ok(())
}

/// One PRODUCT row per catalog entry, with per-store availability:
/// `PRODUCT\t<slug>\t<kind>\t<status>\t<detail>`, then one CATALOG
/// summary line.  Statuses: `renderable` (proven against the imported
/// store), `missing-fields` (direct recipe whose required fields are
/// not all stored), `blocked` (windowed compute reported an honest
/// per-product blocker), and `excluded` (recipe cannot be realized by
/// this lane, reason given).  Every decision routes through stored
/// FIELD availability; there is no model-identity gate and no `gated`
/// status -- a product that cannot render always names the fields it
/// is missing (gpuwm architectural rule; the listing test rejects any
/// identity-gated row).
fn list_products(
    store_root: &std::path::Path,
    model_slug: &str,
    run_slug: &str,
    stored_slots: &[u16],
    heavy_imported: bool,
) -> Result<(), String> {
    use rusty_weather::render_all::StoreFieldSource;
    use rusty_weather::render_all::windowed_store;

    let first_slot = stored_slots
        .first()
        .copied()
        .ok_or("catalog listing needs at least one stored frame")?;
    let store = StoreFieldSource::open(store_root, model_slug, run_slug, first_slot)
        .map_err(|err| err.to_string())?;
    let stored_derived: std::collections::HashSet<&str> = store
        .derived_slugs()
        .iter()
        .map(String::as_str)
        .collect();

    let mut rows: Vec<(String, &str, &str, String)> = Vec::new();

    for spec in rustwx_products::spec::direct_product_specs() {
        if rustwx_products::direct::direct_recipe_requires_explicit_opt_in(&spec.slug) {
            continue;
        }
        match rustwx_models::plot_recipe_store_requirements(&spec.slug) {
            Err(err) => rows.push((
                spec.slug,
                "direct",
                "excluded",
                format!("catalog spec has no plot recipe: {err}"),
            )),
            Ok(requirements) => {
                let missing: Vec<String> = requirements
                    .iter()
                    .filter_map(|requirement| match requirement.selector {
                        Some(selector) => store
                            .resolve(&selector)
                            .is_none()
                            .then(|| selector.key()),
                        None => Some(format!(
                            "{} (no canonical store selector exists for this field)",
                            requirement.field_key
                        )),
                    })
                    .collect();
                if missing.is_empty() {
                    // The title, then the canonical selector the FILL
                    // resolves to.  A title alone cannot distinguish a
                    // chart of a quantity from a chart with that
                    // quantity drawn on top of something else -- "MSLP
                    // / 10m Winds" reads like a wind product and fills
                    // with pressure.  Naming the filled field makes the
                    // listing answer "what am I looking at?" without
                    // reading the catalog source.
                    let detail = match requirements
                        .first()
                        .and_then(|requirement| requirement.selector)
                    {
                        Some(selector) => format!("{} [fill: {}]", spec.title, selector.key()),
                        None => spec.title.clone(),
                    };
                    rows.push((spec.slug, "direct", "renderable", detail));
                } else {
                    rows.push((
                        spec.slug,
                        "direct",
                        "missing-fields",
                        format!("not stored: {}", missing.join(", ")),
                    ));
                }
            }
        }
    }

    for entry in rustwx_products::derived::supported_derived_recipe_inventory() {
        let kind = if entry.heavy { "heavy" } else { "derived" };
        if stored_derived.contains(entry.slug) {
            rows.push((entry.slug.to_string(), kind, "renderable", entry.title.to_string()));
        } else if entry.heavy {
            let reason = if heavy_imported {
                "the wrfout lane's heavy (ECAPE) diagnostics do not produce \
                 this recipe's grid (ml/mu parcels and the CAPE-ratio pairs \
                 need import-side plumbing wrf-core does not expose yet)"
            } else {
                "heavy grid not computed at import; re-run with --heavy \
                 to compute the ECAPE family"
            };
            rows.push((entry.slug.to_string(), kind, "excluded", reason.to_string()));
        } else {
            rows.push((
                entry.slug.to_string(),
                kind,
                "excluded",
                "not realized by the wrfout import lane (no matching \
                 wrf-core diagnostic is stored under this recipe slug)"
                    .to_string(),
            ));
        }
    }
    for entry in rustwx_products::derived::blocked_derived_recipe_inventory() {
        rows.push((
            entry.slug.to_string(),
            "derived",
            "excluded",
            entry.reason.to_string(),
        ));
    }

    let windowed_ready = windowed_store::windowed_axis_ready(store_root, model_slug, run_slug)
        .map_err(|err| err.to_string())?;
    let windowed_slugs: Vec<String> =
        rustwx_products::windowed::HrrrWindowedProduct::supported_products()
            .iter()
            .map(|product| product.slug().to_string())
            .collect();
    if !windowed_ready {
        let reason = if stored_slots.len() <= 1 {
            "windowed accumulations need more than one stored whole-hour frame"
        } else {
            "exact-time ordinal axis; fixed-hour windows are undefined on it"
        };
        for slug in &windowed_slugs {
            rows.push((slug.clone(), "windowed", "excluded", reason.to_string()));
        }
    } else {
        match windowed_store::compute_windowed_products(
            store_root,
            model_slug,
            run_slug,
            stored_slots,
            &windowed_slugs,
        ) {
            Ok(outcome) => {
                let blocked: std::collections::HashMap<String, String> =
                    outcome.blockers.into_iter().collect();
                for grid in &outcome.grids {
                    rows.push((
                        grid.slug.clone(),
                        "windowed",
                        "renderable",
                        grid.strategy.clone(),
                    ));
                }
                for slug in &windowed_slugs {
                    if let Some(reason) = blocked.get(slug) {
                        rows.push((slug.clone(), "windowed", "blocked", reason.clone()));
                    }
                }
            }
            Err(err) => {
                let reason = format!("windowed compute unavailable: {err}");
                for slug in &windowed_slugs {
                    rows.push((slug.clone(), "windowed", "excluded", reason.clone()));
                }
            }
        }
    }

    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
    for (slug, kind, status, detail) in &rows {
        *counts.entry(status).or_default() += 1;
        println!("PRODUCT\t{slug}\t{kind}\t{status}\t{detail}");
    }
    let summary = counts
        .iter()
        .map(|(status, count)| format!("{status}={count}"))
        .collect::<Vec<_>>()
        .join(" ");
    println!("CATALOG total={} {summary}", rows.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An `Args` that is valid apart from what a test deliberately breaks.
    fn args_for(products: &str, inputs: Vec<PathBuf>) -> Args {
        Args {
            store_root: PathBuf::from("store"),
            out_dir: PathBuf::from("out"),
            products: products.to_string(),
            frames: None,
            width: 1_200,
            height: 900,
            heavy: false,
            list_products: false,
            source_label: DEFAULT_SOURCE_LABEL.to_string(),
            inputs,
        }
    }

    fn refusal(result: Result<(), CliError>) -> (u8, String) {
        match result {
            Ok(()) => panic!("expected a refusal"),
            Err(CliError::Help) => panic!("expected a refusal, got --help"),
            Err(CliError::Usage(message)) => (EXIT_USAGE, message),
            Err(CliError::Failed(message)) => (1, message),
        }
    }

    #[test]
    fn the_usage_line_keeps_the_token_the_python_probe_requires() {
        // gpuwm.rustwx.probe_renderer declares the whole rust engine
        // unusable -- silently dropping `--engine auto` to matplotlib --
        // unless `--help` exits 0 with this literal in its transcript.
        assert!(usage().starts_with("usage: rw_wrfbatch"));
    }

    #[test]
    fn an_unknown_product_names_the_token_and_the_choices() {
        let (code, message) = refusal(validate_request(&args_for(
            "definitely_not_a_product",
            vec![PathBuf::from("wrfout_d02_x.nc")],
        )));
        assert_eq!(code, EXIT_USAGE, "a bad command line is a usage failure");
        assert!(
            message.contains("definitely_not_a_product"),
            "the refusal must name the token: {message}"
        );
        for choice in ["'all'", "'direct'", "'derived'", "'heavy'", "'windowed'"] {
            assert!(
                message.contains(choice),
                "the refusal must name the choices, missing {choice}: {message}"
            );
        }
        assert!(
            message.contains("--list-products"),
            "the refusal must route to the full vocabulary: {message}"
        );
    }

    #[test]
    fn a_product_typo_is_caught_before_any_input_is_opened() {
        // The regression this pins: --products used to be validated only
        // after a full wrfout import, so a typo reported "No supported WRF
        // files selected" -- about a file, not about the typo.
        let (_code, message) = refusal(validate_request(&args_for(
            "not_a_product",
            vec![PathBuf::from("no/such/path/wrfout_d01_x.nc")],
        )));
        assert!(
            message.contains("not_a_product"),
            "the product refusal must win over the missing input: {message}"
        );
    }

    #[test]
    fn a_missing_input_is_named_in_the_matplotlib_wording() {
        let (code, message) = refusal(validate_request(&args_for(
            "all",
            vec![PathBuf::from("no/such/path/wrfout_d01_x.nc")],
        )));
        assert_eq!(code, 1, "a real run that cannot proceed is not a usage error");
        assert!(message.contains("wrfout_d01_x.nc"), "{message}");
        assert!(message.contains("unreadable wrfout"), "{message}");
        assert!(message.contains("no such file"), "{message}");
    }

    #[test]
    fn an_input_that_is_not_a_wrf_file_is_named() {
        let dir = std::env::temp_dir().join("rw_wrfbatch_not_a_wrf_file");
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let path = dir.join("plain.txt");
        std::fs::write(&path, b"not a wrfout").expect("scratch file");

        let (code, message) = refusal(validate_request(&args_for("all", vec![path.clone()])));
        assert_eq!(code, 1);
        assert!(
            message.contains("plain.txt"),
            "'No supported WRF files selected' never named the file: {message}"
        );
        assert!(message.contains("unreadable wrfout"), "{message}");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn one_usable_input_among_several_still_runs() {
        // Existing semantics preserved: a mixed set is not a refusal.  Only
        // a set with nothing usable in it is.
        let dir = std::env::temp_dir().join("rw_wrfbatch_mixed_inputs");
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let good = dir.join("wrfout_d02_1974-04-03_18:00:00".replace(':', "-"));
        std::fs::write(&good, b"placeholder").expect("scratch file");
        let bad = dir.join("plain.txt");
        std::fs::write(&bad, b"not a wrfout").expect("scratch file");

        if wrf_process::is_supported_wrf_file(&good) {
            assert!(
                validate_request(&args_for("all", vec![good.clone(), bad.clone()])).is_ok(),
                "a set with one usable input must not be refused"
            );
        }

        let _ = std::fs::remove_file(&good);
        let _ = std::fs::remove_file(&bad);
    }

    #[test]
    fn the_product_vocabulary_is_non_empty_and_deduplicated() {
        let slugs = rusty_weather::render_all::known_product_slugs();
        assert!(!slugs.is_empty(), "no product is selectable at all");
        let mut unique = slugs.clone();
        unique.dedup();
        assert_eq!(unique.len(), slugs.len(), "the vocabulary repeats a slug");
    }

    #[test]
    fn resolution_tokens_are_trimmed_km_or_integer_metres() {
        assert_eq!(resolution_token(3_000.0).as_deref(), Some("3km"));
        assert_eq!(resolution_token(12_000.0).as_deref(), Some("12km"));
        assert_eq!(resolution_token(1_000.0).as_deref(), Some("1km"));
        // A 3:1 nest of a 12 km parent, twice over.
        assert_eq!(resolution_token(1_333.3333).as_deref(), Some("1.333km"));
        assert_eq!(resolution_token(1_500.0).as_deref(), Some("1.5km"));
        // Sub-kilometre reads as metres, not as 0.111 km.
        assert_eq!(resolution_token(111.1111).as_deref(), Some("111m"));
        assert_eq!(resolution_token(500.0).as_deref(), Some("500m"));
        assert_eq!(resolution_token(250.0).as_deref(), Some("250m"));
        // The boundary rounds before it chooses a unit.
        assert_eq!(resolution_token(999.6).as_deref(), Some("1km"));
        assert_eq!(resolution_token(999.4).as_deref(), Some("999m"));
        // Nothing usable is never guessed at.
        assert_eq!(resolution_token(0.0), None);
        assert_eq!(resolution_token(-3_000.0), None);
        assert_eq!(resolution_token(f64::NAN), None);
    }

    #[test]
    fn spacing_subtitles_spell_the_same_number() {
        assert_eq!(spacing_subtitle(3_000.0).as_deref(), Some("\u{0394}x 3 km"));
        assert_eq!(
            spacing_subtitle(1_333.3333).as_deref(),
            Some("\u{0394}x 1.333 km")
        );
        assert_eq!(spacing_subtitle(111.1111).as_deref(), Some("\u{0394}x 111 m"));
        assert_eq!(spacing_subtitle(f64::INFINITY), None);
    }

    #[test]
    fn domain_tokens_come_from_grid_id_then_the_filename() {
        assert_eq!(domain_token_from_id(2).as_deref(), Some("d02"));
        assert_eq!(domain_token_from_id(11).as_deref(), Some("d11"));
        assert_eq!(domain_token_from_id(0), None);
        assert_eq!(domain_token_from_id(-1), None);

        let named = PathBuf::from("/runs/wrfout_d03_1974-04-03_18:00:00");
        assert_eq!(domain_token_from_filename(&named).as_deref(), Some("d03"));
        let anonymous = PathBuf::from("/runs/model_output.nc");
        assert_eq!(domain_token_from_filename(&anonymous), None);
        let truncated = PathBuf::from("/runs/wrfout_d");
        assert_eq!(domain_token_from_filename(&truncated), None);
    }

    #[test]
    fn the_slug_pairs_domain_with_resolution_and_degrades_honestly() {
        let full = GridIdentity {
            domain: Some("d02".to_string()),
            spacing_m: Some(3_000.0),
        };
        assert_eq!(native_domain_slug(&full).as_deref(), Some("d02-3km"));

        // DX absent: the domain still separates the nests, which is the
        // whole point of the token.
        let no_spacing = GridIdentity {
            domain: Some("d02".to_string()),
            spacing_m: None,
        };
        assert_eq!(native_domain_slug(&no_spacing).as_deref(), Some("d02"));

        // Domain absent: no token at all -- `native_grid` is honest about
        // an unidentified grid in a way that `d01` would not be.
        let no_domain = GridIdentity {
            domain: None,
            spacing_m: Some(3_000.0),
        };
        assert_eq!(native_domain_slug(&no_domain), None);
    }

    #[test]
    fn disagreeing_inputs_yield_no_token() {
        // Two nests imported into ONE store render as one run, so no
        // single token can be true of the output.  Filenames alone are
        // enough to reach the disagreement branch.
        let mixed = [
            PathBuf::from("wrfout_d02_1974-04-03_18:00:00"),
            PathBuf::from("wrfout_d03_1974-04-03_18:00:00"),
        ];
        assert_eq!(grid_identity(&mixed), GridIdentity::default());
        assert_eq!(native_domain_slug(&grid_identity(&mixed)), None);

        let agreeing = [
            PathBuf::from("wrfout_d02_1974-04-03_18:00:00"),
            PathBuf::from("wrfout_d02_1974-04-03_19:00:00"),
        ];
        assert_eq!(
            grid_identity(&agreeing).domain.as_deref(),
            Some("d02"),
            "one domain across several files keeps its token"
        );
    }
}

/// The command line is wrong.  Matches `gpuwm render`'s matplotlib engine,
/// so the two engines agree on what a bad `--products` costs.
const EXIT_USAGE: u8 = 2;

fn dispatch() -> Result<(), CliError> {
    match parse_args()? {
        Invocation::Catalog => print_product_catalog(),
        Invocation::Batch(args) => {
            validate_request(&args)?;
            run(*args).map_err(CliError::Failed)
        }
    }
}

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    match dispatch() {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::Help) => {
            // stderr and exit 0, exactly as before: gpuwm.rustwx.probe_renderer
            // reads stdout+stderr together and requires both.
            eprintln!("{}", usage());
            ExitCode::SUCCESS
        }
        Err(CliError::Usage(message)) => {
            // Usage line FIRST, then the sentence that names the problem --
            // argparse's order, which is also the order matplotlib's engine
            // prints in.  It used to be the other way round, and
            // gpuwm/rustwx.py surfaces the LAST non-empty stderr line as the
            // cause, so every argument mistake reached the user as the usage
            // line with the actionable sentence scrolled off above it.
            eprintln!("{}", usage());
            eprintln!("{message}");
            ExitCode::from(EXIT_USAGE)
        }
        Err(CliError::Failed(message)) => {
            // No usage line.  gpuwm/rustwx.py surfaces the LAST non-empty
            // stderr line as the reason, so appending usage() here is what
            // made `gpuwm render` report a usage string as the cause of a
            // store failure.
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
