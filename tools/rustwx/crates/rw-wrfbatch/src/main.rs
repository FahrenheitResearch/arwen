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
#[path = "mesh.rs"]
mod mesh;
#[path = "section.rs"]
mod section;

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

/// The exact `--abi` line `gpuwm.rustwx.RENDERER_ABI_MARKER` pins.
///
/// The same handshake `rw_fetch` and `rw_nexrad` -- this workspace's two
/// other bundled binaries -- already answer, applied to the third.  It
/// exists because "does the binary start" is not a contract check: two
/// rw_wrfbatch builds with different md5s both printed the usage line
/// and both were reported `verified`, so a stale renderer substituted a
/// foreign engine into every rust render gate and drew the plots
/// afterwards (task #106).
///
/// What is listed is the vocabulary the PYTHON half parses, not a
/// version number -- a rebuild bumps a version whether or not anything
/// changed, and a marker that moves on every rebuild proves nothing:
///
/// * the five tab-separated `PRODUCT` fields and the `CATALOG` tally
///   line `gpuwm.rustwx.list_products` reads;
/// * the `RENDERED` / `SKIPPED` / `FAILED` event words
///   `gpuwm.rustwx.run_renderer` reads;
/// * the generic `var:` family and the `selectable_slugs` count of the
///   store-independent catalog -- the lane whose ABSENCE from a stale
///   build is what #106 was reported as (catalog 153 against this
///   tree's 168, zero generic rows) -- the `xsec:` family, the
///   vertical-section lane cut from the wrfout files directly, and the
///   `mesh:`/`meshdiff:` families, the polygon-mesh lane cut from an MPAS
///   history frame and its grid file with no regrid in between.
///
/// Changing any of those is changing this contract, so the literal
/// changes with it and every binary predating the change fails the
/// handshake instead of quietly answering the old grammar.
const ABI_MARKER: &str = "gpuwm-rw-wrfbatch-catalog-v1\tPRODUCT\tslug\tkind\tstatus\tdetail\tCATALOG\t\
gpuwm-rw-wrfbatch-events-v1\tRENDERED\tSKIPPED\tFAILED\t\
gpuwm-rw-wrfbatch-vocabulary-v1\tgeneric\tvar:\txsec:\tmesh:\tmeshdiff:\tselectable_slugs";

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
    /// `--streamlines` / `--barbs`: the wind layer this invocation asks
    /// for.  `None` leaves `RUSTWX_WIND_STREAMLINES`, and then the
    /// automatic per-grid choice, in charge.
    streamlines: Option<bool>,
    source_label: String,
    /// `--overlays FILE.json`: map overlays in geographic degrees.
    overlays: Option<rustwx_products::geographic_overlays::MapOverlays>,
    /// `--annotate FILE.json`: title/subtitle overrides.
    annotations: Option<rustwx_products::geographic_overlays::PanelAnnotations>,
    /// `--theme NAME|FILE.json`: the render theme, resolved before any
    /// import so a typo is refused before a file is opened.  `None` is
    /// the renderer's own look, byte-identical to every earlier build.
    theme: Option<rustwx_render::RenderTheme>,
    /// `--section lat,lon,lat,lon | FILE.json`: the line every `xsec:`
    /// product is cut along; required when one is requested.
    section: Option<section::SectionLine>,
    /// `--section-across KM`: a second frame per section product,
    /// perpendicular to the line through the fill's maximum column.
    section_across_km: Option<f64>,
    /// `--isotherms L,L,...[@H]`: the isotherms drawn on every section.
    isotherms: section::Isotherms,
    /// `--section-top-km N`: the ceiling the fitted height range may not
    /// pass.
    section_top_km: f64,
    /// `--section-size WxH`: the size a SECTION is drawn at.  Absent, a
    /// section is landscape 2:1 at the map's width, because a section
    /// handed the map's own size came out portrait -- the shape a vertical
    /// cut is least readable in.
    section_size: Option<(u32, u32)>,
    /// `--section-reference-km N`: the altitude the fitted height range
    /// keeps at least two kilometres of air above.  Absent, the line's own
    /// highest terrain.
    section_reference_km: Option<f64>,
    /// `--mesh-grid FILE.nc`: the MPAS grid file whose `verticesOnCell`
    /// gives every `mesh:` product its polygons.  Required when one is
    /// requested; a history frame carries no cell boundaries.
    mesh_grid: Option<PathBuf>,
    /// `--mesh-reference DIR|FILE`: the other leg, for `meshdiff:`.
    mesh_reference: Option<PathBuf>,
    /// `--mesh-labels A,B`: the two legs' names in a difference headline.
    mesh_labels: (String, String),
    /// `--mesh-bounds W,E,S,N`: the frame a mesh product is drawn in.
    mesh_bounds: Option<(f64, f64, f64, f64)>,
    /// `--footer-*`: the caption fields the theme's footer strip fills.
    /// Absent every one of them, no strip is drawn even under a theme that
    /// names one, so an existing render is unchanged.
    footer: rustwx_render::FooterFields,
    inputs: Vec<PathBuf>,
}

fn usage() -> &'static str {
    "usage: rw_wrfbatch --store-root DIR --out-dir DIR [--products all|SLUGS] \
[--frames all|N] [--width N] [--height N] [--heavy] [--streamlines|--barbs] \
[--source-label TEXT] [--theme NAME|FILE.json] [--section lat,lon,lat,lon|FILE.json] \
[--section-across KM] [--isotherms L,L,...[@H]] [--section-top-km N] \
[--section-size WxH] [--section-reference-km N] \
[--mesh-grid FILE.nc] [--mesh-reference DIR|FILE] [--mesh-labels A,B] [--mesh-bounds W,E,S,N] \
[--footer-title TEXT] [--footer-valid TEXT] [--footer-mesh TEXT] [--footer-leg TEXT] \
[--footer-note TEXT] [--list-products] wrfout...\n       \
rw_wrfbatch --help | --abi"
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
    /// Print [`ABI_MARKER`] and exit: the stale-build handshake.
    Abi,
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
    // The generic family's vocabulary belongs to the STORE, not this
    // build, so no slug list can be printed here; the store-aware listing
    // names each `var:` row it can serve.
    println!("generic products: var:<stored 2-D variable name>");
    println!(
        "mesh products: mesh:<history variable>[:colmax|:colmin|:level=K][~log] and \
meshdiff:<...> (need --mesh-grid FILE.nc; meshdiff also --mesh-reference)"
    );
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
    let mut streamlines: Option<bool> = None;
    let mut source_label = DEFAULT_SOURCE_LABEL.to_string();
    let mut overlays_path: Option<PathBuf> = None;
    let mut annotate_path: Option<PathBuf> = None;
    let mut theme_spec: Option<String> = None;
    let mut section_spec: Option<String> = None;
    let mut section_across_km: Option<f64> = None;
    let mut isotherms_spec: Option<String> = None;
    let mut section_top_km = 14.0f64;
    let mut section_size: Option<(u32, u32)> = None;
    let mut section_reference_km: Option<f64> = None;
    let mut mesh_grid: Option<PathBuf> = None;
    let mut mesh_reference: Option<PathBuf> = None;
    let mut mesh_labels = ("TREATMENT".to_string(), "CONTROL".to_string());
    let mut mesh_bounds: Option<(f64, f64, f64, f64)> = None;
    let mut footer = rustwx_render::FooterFields::default();
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
            // Absent, these two run no code at all: every product this
            // build already draws is byte-identical without them, which
            // `tools/rustwx_render_regression_gate.py` is the gate for.
            "--overlays" => {
                overlays_path = Some(PathBuf::from(
                    raw.next().ok_or("--overlays requires a JSON file")?,
                ));
            }
            "--annotate" => {
                annotate_path = Some(PathBuf::from(
                    raw.next().ok_or("--annotate requires a JSON file")?,
                ));
            }
            // The render theme: a built-in name (`default`, `light`,
            // `dark`) or a JSON file.  The flag outranks RUSTWX_THEME, and
            // absent both the renderer draws exactly as it always has.
            "--theme" => {
                let value = raw.next().ok_or("--theme requires a name or a JSON file")?;
                if value.trim().is_empty() {
                    return Err(CliError::Usage("--theme must not be blank".to_string()));
                }
                theme_spec = Some(value);
            }
            // The section line and its dressing, for the `xsec:` family.
            "--section" => {
                section_spec =
                    Some(raw.next().ok_or("--section requires lat,lon,lat,lon or a JSON file")?);
            }
            "--section-across" => {
                let value = raw.next().ok_or("--section-across requires a length in km")?;
                let km: f64 = value
                    .parse()
                    .ok()
                    .filter(|km: &f64| km.is_finite() && *km >= 2.0)
                    .ok_or_else(|| {
                        format!("--section-across '{value}' is not a length of at least 2 km")
                    })?;
                section_across_km = Some(km);
            }
            "--isotherms" => {
                isotherms_spec =
                    Some(raw.next().ok_or("--isotherms requires levels in C (or none)")?);
            }
            "--section-size" => {
                let value = raw.next().ok_or("--section-size requires WxH in pixels")?;
                let (w, h) = value
                    .split_once(['x', 'X'])
                    .ok_or_else(|| format!("--section-size '{value}' is not WxH"))?;
                let parse = |text: &str, name: &str| -> Result<u32, String> {
                    text.trim()
                        .parse::<u32>()
                        .ok()
                        .filter(|value| (200..=12_000).contains(value))
                        .ok_or_else(|| {
                            format!("--section-size {name} '{text}' is not 200-12000 pixels")
                        })
                };
                section_size = Some((parse(w, "width")?, parse(h, "height")?));
            }
            "--section-reference-km" => {
                let value = raw
                    .next()
                    .ok_or("--section-reference-km requires a height in km")?;
                section_reference_km = Some(
                    value
                        .parse::<f64>()
                        .ok()
                        .filter(|km| (0.0..=40.0).contains(km))
                        .ok_or_else(|| {
                            format!("--section-reference-km '{value}' is not within 0-40 km")
                        })?,
                );
            }
            "--section-top-km" => {
                let value = raw.next().ok_or("--section-top-km requires a height in km")?;
                section_top_km = value
                    .parse()
                    .ok()
                    .filter(|km: &f64| km.is_finite() && (1.0..=40.0).contains(km))
                    .ok_or_else(|| format!("--section-top-km '{value}' is not within 1-40 km"))?;
            }
            // The polygon-mesh family's three flags.  The grid file is
            // separate from the history frames because one mesh serves a
            // whole run: reading its connectivity once per frame is work
            // with no answer attached to it.
            "--mesh-grid" => {
                mesh_grid = Some(PathBuf::from(
                    raw.next().ok_or("--mesh-grid requires an MPAS grid file")?,
                ));
            }
            "--mesh-reference" => {
                mesh_reference = Some(PathBuf::from(
                    raw.next()
                        .ok_or("--mesh-reference requires the other leg's frame directory or file")?,
                ));
            }
            "--mesh-labels" => {
                let value = raw.next().ok_or("--mesh-labels requires A,B")?;
                let mut parts = value.splitn(2, ',').map(str::trim);
                let a = parts.next().unwrap_or("").to_string();
                let b = parts.next().unwrap_or("").to_string();
                if a.is_empty() || b.is_empty() {
                    return Err(CliError::Usage(format!(
                        "--mesh-labels '{value}' is not two comma-separated leg names"
                    )));
                }
                mesh_labels = (a, b);
            }
            "--mesh-bounds" => {
                let value = raw.next().ok_or("--mesh-bounds requires W,E,S,N")?;
                let parts: Vec<f64> = value
                    .split(',')
                    .map(str::trim)
                    .filter_map(|part| part.parse::<f64>().ok())
                    .collect();
                if parts.len() != 4 || parts.iter().any(|value| !value.is_finite()) {
                    return Err(CliError::Usage(format!(
                        "--mesh-bounds '{value}' is not four finite degrees W,E,S,N"
                    )));
                }
                if parts[2] >= parts[3] {
                    return Err(CliError::Usage(format!(
                        "--mesh-bounds '{value}': south {} is not below north {}",
                        parts[2], parts[3]
                    )));
                }
                mesh_bounds = Some((parts[0], parts[1], parts[2], parts[3]));
            }
            // The footer strip's caption fields.  The strip itself is the
            // theme's; these fill it.  Setting none of them draws none.
            "--footer-title" => {
                footer.product_title = Some(raw.next().ok_or("--footer-title requires text")?);
            }
            "--footer-valid" => {
                footer.valid_time = Some(raw.next().ok_or("--footer-valid requires text")?);
            }
            "--footer-mesh" => {
                footer.mesh_or_grid = Some(raw.next().ok_or("--footer-mesh requires text")?);
            }
            "--footer-leg" => {
                footer.leg = Some(raw.next().ok_or("--footer-leg requires text")?);
            }
            "--footer-note" => {
                footer.note = Some(raw.next().ok_or("--footer-note requires text")?);
            }
            "--heavy" => heavy = true,
            // The wind layer, at the front door.  Drawing streamlines was
            // reachable only through RUSTWX_WIND_STREAMLINES, a name in no
            // help text and no document, so the capability was engine-only.
            // The variable still works; these two outrank it, because a
            // flag the environment can silently overrule is a lie.
            "--streamlines" => streamlines = Some(true),
            "--barbs" => streamlines = Some(false),
            "--list-products" => list_products = true,
            "--help" | "-h" => return Err(CliError::Help),
            // Answered before anything else is validated, and on stdout:
            // the caller is asking what contract this BINARY speaks, not
            // asking it to do work, so no other argument is required and
            // none can suppress the answer.
            "--abi" => return Ok(Invocation::Abi),
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
    let theme = theme_spec
        .or_else(|| {
            std::env::var(rustwx_render::THEME_ENV)
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .map(|spec| rustwx_render::RenderTheme::resolve(&spec))
        .transpose()
        .map_err(CliError::Usage)?;
    let section = section_spec
        .as_deref()
        .map(section::SectionLine::parse)
        .transpose()
        .map_err(CliError::Usage)?;
    let isotherms = match isotherms_spec {
        Some(text) => section::Isotherms::parse(&text).map_err(CliError::Usage)?,
        None => section::Isotherms::default(),
    };
    Ok(Invocation::Batch(Box::new(Args {
        store_root: store_root.ok_or("--store-root is required")?,
        out_dir: out_dir.ok_or("--out-dir is required")?,
        products,
        frames,
        width,
        height,
        heavy,
        list_products,
        streamlines,
        source_label,
        overlays: overlays_path
            .as_deref()
            .map(rustwx_products::geographic_overlays::MapOverlays::load)
            .transpose()
            .map_err(CliError::Usage)?,
        annotations: annotate_path
            .as_deref()
            .map(rustwx_products::geographic_overlays::PanelAnnotations::load)
            .transpose()
            .map_err(CliError::Usage)?,
        theme,
        section,
        section_across_km,
        isotherms,
        section_top_km,
        section_size,
        section_reference_km,
        mesh_grid,
        mesh_reference,
        mesh_labels,
        mesh_bounds,
        footer,
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
    // The two store-free families are split off before the store's own
    // vocabulary is consulted.  `partition_products` knows the STORE's
    // slugs, so it reads `mesh:qi:colmax` and `xsec:QICE` as typos and
    // refuses a command line that is correct -- which is what a
    // mesh-only or section-only invocation is.
    let (non_mesh, mesh_products) =
        mesh::split_product_spec(&args.products).map_err(CliError::Usage)?;
    let (store_products, section_products) =
        section::split_product_spec(&non_mesh).map_err(CliError::Usage)?;
    if !store_products.trim().is_empty() || (mesh_products.is_empty() && section_products.is_empty())
    {
        rusty_weather::render_all::partition_products(&store_products)
            .map_err(|err| CliError::Usage(err.to_string()))?;
    }
    if !mesh_products.is_empty() {
        // A mesh: input is an MPAS history frame, which is deliberately NOT
        // a wrfout and would fail the readability check below by design.
        // The mesh reader refuses a file that is not one, by name, with the
        // dimension it looked for.
        if args.mesh_grid.is_none() {
            return Err(CliError::Usage(
                "mesh: products need --mesh-grid FILE.nc: a history frame carries cell CENTRES                  and no cell boundaries, so there are no polygons to draw without it."
                    .to_string(),
            ));
        }
        for path in &args.inputs {
            if !path.is_file() {
                return Err(CliError::Failed(format!(
                    "{}: unreadable history frame (no such file)",
                    path.display()
                )));
            }
        }
        return Ok(());
    }

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

/// One import note as its own greppable stderr record.  Tab-separated so
/// the note text (which contains spaces and colons) stays one field.
fn import_note_line(note: &str) -> String {
    format!("IMPORT_NOTE\t{note}")
}

fn run(args: Args) -> Result<(), String> {
    // Recorded before any product lane builds a wind layer, and cleared to
    // `None` when neither flag was given so RUSTWX_WIND_STREAMLINES and the
    // automatic per-grid choice keep their existing meaning.
    rustwx_products::shared_context::set_wind_streamline_request(args.streamlines);
    // The theme is installed before the first presentation is built and
    // before the first glyph is drawn (its fonts load with it).  Absent, the
    // renderer's own look is installed by name so a later RUSTWX_THEME read
    // cannot restyle half a run.
    let theme_name = match args.theme {
        Some(theme) => {
            let name = theme.name.clone();
            rustwx_render::install_theme(theme)?;
            name
        }
        None => {
            rustwx_render::install_theme(rustwx_render::RenderTheme::default_theme())?;
            "default".to_string()
        }
    };
    println!("THEME {theme_name}");
    // The section crate draws its own text; a theme with fonts hands it the
    // same bytes so one theme names the type on every surface.
    {
        let theme = rustwx_render::active_theme();
        if theme.font_regular.is_some() || theme.font_bold.is_some() {
            let read = |path: &Option<PathBuf>| path.as_ref().and_then(|p| std::fs::read(p).ok());
            rustwx_cross_section::install_cross_section_fonts(
                read(&theme.font_regular),
                read(&theme.font_bold),
            );
        }
    }
    // The caption fields for the theme's footer strip, if the theme names
    // one and the caller filled any.  The grid row defaults to the run's own
    // domain and spacing, read from the inputs' global attributes -- the
    // same identity the subtitle already carries, so one plot never states
    // two different grids.
    //
    // Installed HERE, before the store-free families are dispatched: a
    // section-only or mesh-only invocation returns without ever reaching
    // the import, so a strip installed after it was a strip those panels
    // could never carry.
    if args.footer != rustwx_render::FooterFields::default() {
        let mut footer = args.footer.clone();
        if footer.mesh_or_grid.is_none() {
            footer.mesh_or_grid = domain_title_label(&grid_identity(&args.inputs));
        }
        rustwx_render::set_footer_fields(footer);
    }
    // Two families never touch the store.  `mesh:` reads an MPAS history
    // frame and its grid file; `xsec:` cuts the wrfout files directly.
    // Both are split off here, and when they are all that was asked for the
    // import is skipped entirely.
    let (non_mesh_products, mesh_products) = mesh::split_product_spec(&args.products)?;
    let (store_products, section_products) = section::split_product_spec(&non_mesh_products)?;
    if !mesh_products.is_empty() && !(store_products.is_empty() && section_products.is_empty()) {
        // The INPUTS differ, not just the drawing.  A mesh: product's input
        // is an MPAS history frame; every other family's is a wrfout.  One
        // invocation cannot be handed both lists, and importing a history
        // frame as a wrfout fails deep inside the importer with a message
        // about a missing Times variable -- which says nothing about the
        // real mistake.
        return Err(format!(
            "mesh: products read an MPAS history frame plus --mesh-grid; {} \
             read wrfout files. Ask for them in separate invocations.",
            if section_products.is_empty() {
                "the store products"
            } else {
                "the store and xsec: products"
            }
        ));
    }
    if !mesh_products.is_empty() {
        let grid_path = args.mesh_grid.as_deref().ok_or_else(|| {
            "mesh: products need the mesh: --mesh-grid FILE.nc. A history frame carries cell \
             CENTRES and no cell boundaries, so there are no polygons to draw without it."
                .to_string()
        })?;
        let read_started = std::time::Instant::now();
        let geometry = mesh::read_mesh_geometry(grid_path)?;
        println!(
            "MESH grid={} cells={} spacing_km={:.3}-{:.3} description={} read_ms={}",
            grid_path.display(),
            geometry.n_cells,
            geometry.spacing_km.0,
            geometry.spacing_km.1,
            geometry.description(),
            read_started.elapsed().as_millis()
        );
        let config = mesh::MeshRenderConfig {
            inputs: &args.inputs,
            out_dir: &args.out_dir,
            grid: &geometry,
            reference: args.mesh_reference.as_deref(),
            labels: args.mesh_labels.clone(),
            width: args.width,
            height: args.height,
            source_label: args.source_label.clone(),
            theme: rustwx_render::active_theme(),
            frame: args.frames,
            bounds: args.mesh_bounds,
            footer: args.footer.clone(),
        };
        let mesh_started = std::time::Instant::now();
        let (rendered, failed) =
            mesh::render_mesh_products(&mesh_products, &config, |outcome| match outcome.result {
                Ok(path) => {
                    // The timing rides its OWN line.  `RENDERED <slug>
                    // <path>` is a two-field record every reader of this
                    // stream splits on the first space, and a third field
                    // would land inside the path they read.
                    println!(
                        "MESH_RENDER {} cells={} render_ms={}",
                        outcome.slug, outcome.cells, outcome.render_ms
                    );
                    println!("RENDERED {} {}", outcome.slug, path.display());
                }
                Err(err) => eprintln!("FAILED {} {err}", outcome.slug),
            })?;
        println!(
            "FINISHED rendered={rendered} skipped=0 failed={failed} elapsed_ms={}",
            mesh_started.elapsed().as_millis()
        );
        if rendered == 0 || failed > 0 {
            return Err(format!(
                "mesh render incomplete: rendered={rendered} failed={failed}"
            ));
        }
        return Ok(());
    }
    if !section_products.is_empty() && args.section.is_none() {
        return Err(
            "xsec: products need a line: --section lat,lon,lat,lon or --section FILE.json"
                .to_string(),
        );
    }
    let section_inputs = args.inputs.clone();
    let section_out_dir = args.out_dir.clone();
    let section_source_label = args.source_label.clone();
    let section_domain_slug = native_domain_slug(&grid_identity(&args.inputs));
    let (section_width, section_height) =
        section_dimensions(args.section_size, args.width, args.height);
    let section_args = SectionArgs {
        line: args.section.clone(),
        across_km: args.section_across_km,
        isotherms: args.isotherms.clone(),
        frame: args.frames,
        width: section_width,
        height: section_height,
        top_km: args.section_top_km,
        reference_km: args.section_reference_km,
    };
    let section_started = std::time::Instant::now();
    if store_products.is_empty() && !section_products.is_empty() && !args.list_products {
        let (rendered, failed) = render_section_products(
            &section_products,
            &section_args,
            &section_inputs,
            &section_out_dir,
            section_domain_slug,
            &section_source_label,
        )?;
        println!(
            "FINISHED rendered={rendered} skipped=0 failed={failed} elapsed_ms={}",
            section_started.elapsed().as_millis()
        );
        if rendered == 0 || failed > 0 {
            return Err(format!(
                "section render incomplete: rendered={rendered} failed={failed}"
            ));
        }
        return Ok(());
    }
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
    // Every note verbatim, not just the count.  A `notes=7` tally hid real
    // uvmet/uvmet10/interpolation failures -- named products degraded or
    // vanished with nothing on the transcript saying why.
    for note in &import.notes {
        eprintln!("{}", import_note_line(note));
    }

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
    let product_spec = if store_products.eq_ignore_ascii_case("all") {
        catalog
            .products
            .iter()
            .map(|product| product.slug.as_str())
            .collect::<Vec<_>>()
            .join(",")
    } else {
        store_products
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
    // The manifest destination survives `args.out_dir` moving into the
    // request below.
    let georef_out_dir = args.out_dir.clone();
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
        geographic_overlays: args.overlays,
        panel_annotations: args.annotations,
        output_width: args.width,
        output_height: args.height,
        limits,
    };
    let cancel = AtomicBool::new(false);
    // Every RENDERED panel's georeference (or the lane's reason it has
    // none), collected off the event stream so the run can publish its
    // geographic transforms beside the PNGs.  The pinned stdout grammar
    // is untouched: the collection rides the same event the RENDERED
    // line already prints.
    let mut panel_georefs: Vec<RenderedPanelGeoref> = Vec::new();
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
            slug,
            output_path,
            georeference,
            georeference_absent_reason,
            ..
        } => {
            println!("RENDERED {slug} {}", output_path.display());
            panel_georefs.push((output_path, georeference, georeference_absent_reason));
        }
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
    // Default-on, no flag: a bare run must stop showing the defect (a
    // consumer holding a PNG with no way to put a lat/lon on a pixel).
    // Written before the failure check below so a partially-failed run
    // still georeferences every panel it DID produce.
    write_georef_manifest(&georef_out_dir, &panel_georefs)?;
    // Sections requested beside store products ride the same run, after
    // the store lane, with their own tally folded into the verdict.
    let (section_rendered, section_failed) = if section_products.is_empty() {
        (0, 0)
    } else {
        let counts = render_section_products(
            &section_products,
            &section_args,
            &section_inputs,
            &section_out_dir,
            section_domain_slug,
            &section_source_label,
        )?;
        println!(
            "SECTIONS rendered={} failed={} elapsed_ms={}",
            counts.0,
            counts.1,
            section_started.elapsed().as_millis()
        );
        counts
    };
    if summary.rendered + section_rendered == 0 || summary.failed + section_failed > 0 {
        return Err(format!(
            "batch render incomplete: rendered={} skipped={} failed={}",
            summary.rendered + section_rendered,
            summary.skipped,
            summary.failed + section_failed
        ));
    }
    Ok(())
}

/// What the section lane needs from the invocation, captured before the
/// store lane consumes the rest of `Args`.
struct SectionArgs {
    line: Option<section::SectionLine>,
    across_km: Option<f64>,
    isotherms: section::Isotherms,
    frame: Option<usize>,
    width: u32,
    height: u32,
    top_km: f64,
    reference_km: Option<f64>,
}

/// The size a SECTION is drawn at: what the caller asked for, or landscape
/// 2:1 at the map's width.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law, CLAUDE.md): the pair tool handed
/// sections the MAP's size and a 1800x1464 near-square map produced a
/// 1800x1464 near-square section -- a vertical cut, which is 100 km wide
/// and 6 km tall, drawn in a portrait-ish frame.
pub fn section_dimensions(
    explicit: Option<(u32, u32)>,
    map_width: u32,
    map_height: u32,
) -> (u32, u32) {
    if let Some((width, height)) = explicit {
        return (width, height);
    }
    let width = map_width.max(map_height).max(640);
    (width, (width / 2).max(320))
}

/// The `xsec:` lane: sections cut from the wrfout files directly and
/// reported through the RENDERED / FAILED grammar the Python side reads.
/// Returns `(rendered, failed)`.
fn render_section_products(
    products: &[section::SectionProduct],
    args: &SectionArgs,
    inputs: &[PathBuf],
    out_dir: &std::path::Path,
    domain_slug: Option<String>,
    source_label: &str,
) -> Result<(usize, usize), String> {
    let line = args
        .line
        .clone()
        .ok_or_else(|| "xsec: products need --section".to_string())?;
    println!(
        "SECTIONS products={} line={:.4},{:.4}->{:.4},{:.4} length_km={:.1}",
        products.len(),
        line.start.lat_deg,
        line.start.lon_deg,
        line.end.lat_deg,
        line.end.lon_deg,
        line.length_km()
    );
    let config = section::SectionRenderConfig {
        inputs,
        out_dir,
        line,
        across_km: args.across_km,
        isotherms: args.isotherms.clone(),
        frame: args.frame,
        width: args.width,
        height: args.height,
        top_km: args.top_km,
        reference_km: args.reference_km,
        domain_slug,
        source_label: source_label.to_string(),
        theme: rustwx_render::active_theme(),
    };
    section::render_sections(products, &config, |outcome| match outcome.result {
        Ok(path) => println!("RENDERED {} {}", outcome.slug, path.display()),
        Err(err) => eprintln!("FAILED {} {err}", outcome.slug),
    })
}

/// The schema string every `render-georef.json` declares.
const GEOREF_MANIFEST_SCHEMA: &str = "rustwx.render-georef/v1";

/// One rendered panel's georeference report off the event stream: the
/// output path, the transform when the lane published one, and the lane's
/// reason when it did not.
type RenderedPanelGeoref = (
    PathBuf,
    Option<rustwx_render::PanelGeoReference>,
    Option<String>,
);

/// `<out_dir>/render-georef.json`: what each PNG of one render run maps to
/// on the Earth.
///
/// This file is the fix for a measured defect: `rw_wrfbatch` published no
/// geographic transform anywhere, so a consumer holding a rendered PNG
/// recovered one by registration -- and got 4.175 px/deg longitude against
/// 4.38 px/deg latitude on a global panel, an aspect no flat map has,
/// because the panel was Robinson and no linear fit describes Robinson.
/// Different products in ONE batch also draw in different projections, so
/// the transforms are per-panel, keyed by the PNG's path relative to the
/// output directory (forward slashes).  Every panel that got no transform
/// is listed in `without_georeference` with the reason -- a silent
/// omission is exactly the defect being fixed.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct GeorefManifest {
    schema: String,
    generated_utc: String,
    panels: std::collections::BTreeMap<String, rustwx_render::PanelGeoReference>,
    without_georeference: Vec<GeorefAbsence>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct GeorefAbsence {
    path: String,
    reason: String,
}

/// The manifest key for one output: relative to the run's output
/// directory, forward slashes, so the file stays meaningful when the run
/// directory moves between machines.
fn georef_manifest_key(out_dir: &std::path::Path, output_path: &std::path::Path) -> String {
    output_path
        .strip_prefix(out_dir)
        .unwrap_or(output_path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn build_georef_manifest(
    out_dir: &std::path::Path,
    generated_utc: String,
    panels: &[RenderedPanelGeoref],
) -> GeorefManifest {
    let mut published = std::collections::BTreeMap::new();
    let mut without_georeference = Vec::new();
    for (output_path, georeference, absent_reason) in panels {
        let key = georef_manifest_key(out_dir, output_path);
        match georeference {
            Some(georeference) => {
                published.insert(key, georeference.clone());
            }
            None => without_georeference.push(GeorefAbsence {
                path: key,
                // The lane's own reason travels with the event; a lane
                // that supplied neither still gets a non-empty entry
                // naming the gap, never a silent omission.
                reason: absent_reason.clone().unwrap_or_else(|| {
                    "the render lane threaded neither a georeference nor an absence \
                     reason through its rendered-product record"
                        .to_string()
                }),
            }),
        }
    }
    GeorefManifest {
        schema: GEOREF_MANIFEST_SCHEMA.to_string(),
        generated_utc,
        panels: published,
        without_georeference,
    }
}

/// Write `<out_dir>/render-georef.json` and print the `GEOREF` line.
/// Default-on: an opt-in flag would leave a bare run showing the defect.
fn write_georef_manifest(
    out_dir: &std::path::Path,
    panels: &[RenderedPanelGeoref],
) -> Result<(), String> {
    let generated_utc = chrono::Utc::now()
        .format("%Y-%m-%dT%H:%M:%SZ")
        .to_string();
    let manifest = build_georef_manifest(out_dir, generated_utc, panels);
    let manifest_path = out_dir.join("render-georef.json");
    let json = serde_json::to_string_pretty(&manifest)
        .map_err(|err| format!("serialize {}: {err}", manifest_path.display()))?;
    std::fs::write(&manifest_path, json)
        .map_err(|err| format!("write {}: {err}", manifest_path.display()))?;
    // A NEW stdout line type after FINISHED.  Safe against the pinned
    // grammar: gpuwm.rustwx matches known prefixes and ignores the rest,
    // and no existing line changed.
    println!(
        "GEOREF {} panels={} without={}",
        manifest_path.display(),
        manifest.panels.len(),
        manifest.without_georeference.len()
    );
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
///
/// WHAT is renderable comes from `inspect_renderable_products` -- the
/// same shared catalog the render pass consumes -- so this listing can
/// never disagree with what a render would attempt (it used to rebuild
/// its own product list and silently omitted whole families, which is
/// how generic `var:` rows stayed invisible).  This function only adds
/// the WHY for rows the catalog does not carry.
fn list_products(
    store_root: &std::path::Path,
    model_slug: &str,
    run_slug: &str,
    stored_slots: &[u16],
    heavy_imported: bool,
) -> Result<(), String> {
    use rusty_weather::batch_render::BatchProductKind;
    use rusty_weather::render_all::StoreFieldSource;
    use rusty_weather::render_all::windowed_store;

    let first_slot = stored_slots
        .first()
        .copied()
        .ok_or("catalog listing needs at least one stored frame")?;
    let catalog = inspect_renderable_products(store_root, model_slug, run_slug, first_slot)?;
    let renderable_slugs: std::collections::HashSet<&str> = catalog
        .products
        .iter()
        .map(|product| product.slug.as_str())
        .collect();
    // Diagnostics only: the catalog decides WHAT renders; the store is
    // still consulted to name the fields a non-renderable recipe misses.
    let store = StoreFieldSource::open(store_root, model_slug, run_slug, first_slot)
        .map_err(|err| err.to_string())?;

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
                if renderable_slugs.contains(spec.slug.as_str()) {
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
                } else if !missing.is_empty() {
                    rows.push((
                        spec.slug,
                        "direct",
                        "missing-fields",
                        format!("not stored: {}", missing.join(", ")),
                    ));
                } else {
                    rows.push((
                        spec.slug,
                        "direct",
                        "excluded",
                        "not offered by the shared render catalog for this store".to_string(),
                    ));
                }
            }
        }
    }

    for entry in rustwx_products::derived::supported_derived_recipe_inventory() {
        let kind = if entry.heavy { "heavy" } else { "derived" };
        if renderable_slugs.contains(entry.slug) {
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

    // Generic rows come straight off the shared catalog: every stored 2-D
    // variable that no named product already renders, as `var:<name>`.
    // The catalog's dedup exclusions were logged on stderr when it built.
    for product in &catalog.products {
        if product.kind != BatchProductKind::Generic {
            continue;
        }
        let field = product
            .source_fields
            .first()
            .map(String::as_str)
            .unwrap_or("");
        rows.push((
            product.slug.clone(),
            "generic",
            "renderable",
            format!(
                "stored 2-D variable '{field}' [{}]",
                product.units.as_deref().unwrap_or("unknown units")
            ),
        ));
    }

    let windowed_ready = catalog
        .products
        .iter()
        .any(|product| product.kind == BatchProductKind::Windowed);
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

    /// WHAT BREAKAGE THIS PREVENTS (gate law, CLAUDE.md): the pair tool
    /// handed a section the MAP's size, so a 1800x1464 near-square map
    /// produced a 1800x1464 near-square section -- a cut 100 km wide and
    /// 6 km tall drawn in a portrait-ish frame.
    #[test]
    fn a_section_is_landscape_by_default_whatever_size_the_map_is() {
        for (map_w, map_h) in [(1800u32, 1464u32), (1200, 1600), (2400, 1200), (960, 720)] {
            let (w, h) = section_dimensions(None, map_w, map_h);
            assert_eq!(w, h * 2, "{map_w}x{map_h} gave {w}x{h}");
            assert!(w >= map_w, "{map_w}x{map_h} lost width");
        }
        // The two shapes the rejected sheets came out at.
        assert_eq!(section_dimensions(None, 1800, 1464), (1800, 900));
        assert_eq!(section_dimensions(None, 2400, 1200), (2400, 1200));
        // A caller that names a size gets exactly it.
        assert_eq!(section_dimensions(Some((1600, 1200)), 2400, 1200), (1600, 1200));
    }

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
            streamlines: None,
            source_label: DEFAULT_SOURCE_LABEL.to_string(),
            overlays: None,
            annotations: None,
            theme: None,
            section: None,
            section_across_km: None,
            isotherms: section::Isotherms::default(),
            section_top_km: 14.0,
            section_size: None,
            section_reference_km: None,
            mesh_grid: None,
            mesh_reference: None,
            mesh_labels: ("TREATMENT".to_string(), "CONTROL".to_string()),
            mesh_bounds: None,
            footer: rustwx_render::FooterFields::default(),
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

    /// A Lambert georeference serialized into the manifest and read back
    /// must place a point on the SAME pixel.  This is the wire-format half
    /// of the fix: a transform that survives rendering but not JSON would
    /// still leave the consumer registering coastlines.
    #[test]
    fn georef_manifest_round_trips_a_projected_point_unchanged() {
        let projection = rustwx_render::ResolvedProjection::LambertConformal {
            standard_parallel_1_deg: 33.0,
            standard_parallel_2_deg: 45.0,
            central_meridian_deg: -96.0,
            reference_latitude_deg: 39.0,
        };
        let (x, y) = projection.project(35.0, -97.5);
        let georeference = rustwx_render::PanelGeoReference::new(
            1_600,
            900,
            rustwx_render::PlotRect {
                x: 22,
                y: 51,
                width: 1_430,
                height: 810,
            },
            projection,
            rustwx_render::ProjectedExtent {
                x_min: x - 500_000.0,
                x_max: x + 500_000.0,
                y_min: y - 300_000.0,
                y_max: y + 300_000.0,
            },
            (-103.0, -92.0, 31.0, 39.0),
        );
        let before = georeference
            .lonlat_to_pixel(35.0, -97.5)
            .expect("the point is inside the frame");

        let out_dir = PathBuf::from("C:\\proof\\out");
        let manifest = build_georef_manifest(
            &out_dir,
            "2026-08-26T00:00:00Z".to_string(),
            &[(
                out_dir.join("d02-3km").join("refl").join("frame.png"),
                Some(georeference),
                None,
            )],
        );
        let json = serde_json::to_string_pretty(&manifest).expect("manifest serializes");
        let parsed: GeorefManifest = serde_json::from_str(&json).expect("manifest parses back");

        assert_eq!(parsed.schema, "rustwx.render-georef/v1");
        assert!(parsed.without_georeference.is_empty());
        // Path relative to out_dir, forward slashes: the key contract.
        let restored = parsed
            .panels
            .get("d02-3km/refl/frame.png")
            .expect("the panel is keyed by its out_dir-relative forward-slash path");
        let after = restored
            .lonlat_to_pixel(35.0, -97.5)
            .expect("the restored transform still places the point");
        assert!(
            (before.0 - after.0).abs() < 1.0e-9 && (before.1 - after.1).abs() < 1.0e-9,
            "JSON round trip moved the point: {before:?} -> {after:?}"
        );
    }

    /// A panel with no transform lands in `without_georeference` with a
    /// non-empty reason -- the lane's own when it gave one, a fallback
    /// naming the gap when it did not.  A silent omission is exactly the
    /// defect the manifest exists to fix.
    #[test]
    fn a_panel_without_a_georeference_is_listed_with_its_reason() {
        let out_dir = PathBuf::from("C:\\proof\\out");
        let manifest = build_georef_manifest(
            &out_dir,
            "2026-08-26T00:00:00Z".to_string(),
            &[
                (
                    out_dir.join("cloud_levels.png"),
                    None,
                    Some("composite panel: one PNG composes multiple member maps".to_string()),
                ),
                (out_dir.join("unthreaded.png"), None, None),
            ],
        );
        assert!(manifest.panels.is_empty());
        assert_eq!(manifest.without_georeference.len(), 2);
        let by_path: std::collections::HashMap<&str, &str> = manifest
            .without_georeference
            .iter()
            .map(|absence| (absence.path.as_str(), absence.reason.as_str()))
            .collect();
        assert_eq!(
            by_path["cloud_levels.png"],
            "composite panel: one PNG composes multiple member maps",
            "the lane's own reason must survive verbatim"
        );
        let fallback = by_path["unthreaded.png"];
        assert!(
            !fallback.is_empty() && fallback.contains("threaded neither"),
            "a lane that said nothing still gets a reason naming the gap: {fallback}"
        );
    }

    #[test]
    fn import_failures_surface_as_import_note_lines_not_a_count() {
        // The regression this pins: the transcript said `notes=1` and
        // nothing else, so a real import failure (wrf_process pushes notes
        // like "PSFC skipped: ...", "uvmet unavailable: ...") was invisible
        // unless the caller went digging.  Every note must become one
        // verbatim, tab-separated IMPORT_NOTE record.
        let notes = [
            "PSFC skipped: missing variable".to_string(),
            "uvmet unavailable: rotation constants absent".to_string(),
        ];
        let lines: Vec<String> = notes.iter().map(|note| import_note_line(note)).collect();
        assert_eq!(lines.len(), notes.len(), "one record per note, no tally");
        for (line, note) in lines.iter().zip(&notes) {
            assert_eq!(line, &format!("IMPORT_NOTE\t{note}"));
            let (tag, body) = line.split_once('\t').expect("tab-separated record");
            assert_eq!(tag, "IMPORT_NOTE");
            assert_eq!(body, note, "the note text must survive verbatim");
        }
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
        Invocation::Abi => {
            println!("{ABI_MARKER}");
            Ok(())
        }
        Invocation::Catalog => print_product_catalog(),
        Invocation::Batch(args) => {
            validate_request(&args)?;
            run(*args).map_err(CliError::Failed)
        }
    }
}

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    if let Some(result) = rw_wrfbatch::process_request::try_cli(&std::env::args().skip(1).collect::<Vec<_>>()) {
        return match result { Ok(()) => ExitCode::SUCCESS, Err(message) => { eprintln!("{message}"); ExitCode::FAILURE } };
    }
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
