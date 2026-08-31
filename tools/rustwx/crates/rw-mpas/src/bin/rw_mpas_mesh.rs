//! `rw_mpas_mesh` -- make an MPAS mesh, in Rust, from a resolution request.
//!
//! Until this existed, "arbitrary resolution" meant arbitrary among the meshes
//! somebody else had published, and a 10 GiB card fell onto a uniform 120 km
//! grid because the next published mesh up did not fit. This binary takes the
//! request -- a background spacing, any number of refinement regions, and
//! either a cell count or a device budget -- and produces the grid file.
//!
//! The resolution request is DATA. `--spec` reads a JSON document whose regions
//! are rows; adding a place to refine is a row, not a code path.
//!
//! WHAT IT DELIVERS, stated so nobody reads it as more: a GRID file. Running
//! the mesh also needs a matching STATIC file (terrain, land use, soil,
//! `nVertLevels = 55`, `nSoilLevels = 4`, an FP32-bit-exact `nominalMinDc`),
//! which is built against a terrain archive and is not produced here.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rw_mpas::mesh::density::MeshSpec;
use rw_mpas::mesh::emit::{self, Provenance};
use rw_mpas::mesh::footprint;
use rw_mpas::staticfile::coordframe::CoordinateRepresentation;
use rw_mpas::mesh::{GenerateRequest, Limits, LloydOptions, generate};

/// The literal a bridge contract handshakes on. It spells the argument vector
/// out, so a stale binary on a user's disk fails the static check before it can
/// accept input it will mis-parse.
///
/// `--card` is IN the marker on purpose. A build predating it accepts
/// `--vram-gib 16` on its own and answers 79,717 cells from one card's baked
/// fixed term, which is the wrong number on any other part -- 133,144 on the
/// 70 SM card people actually own. That is exactly the stale-binary failure
/// the marker exists to catch.
pub const ABI_MARKER: &str = "rw_mpas_mesh --out GRID.nc [--spec SPEC.json | --background-km KM | --from-centres GRID.nc] \
[--cells N | --card KEY [--vram-gib X]] [--fit-spacing yes|no] [--sweeps N] [--tolerance X] [--omega X] \
[--receipt JSON] [--triangulation rebuild|incremental] [--clobber] [--dry-run] [--list-cards]";

/// Progress tokens this binary prints, one per stage, tab separated.
pub const PROGRESS_TOKENS: &str = "SIZED\tSEEDED\tRELAXED\tDERIVED\tVALIDATED\tWROTE\tFINISHED";

/// Progress tokens the `--from-centres` route prints instead of the sizing and
/// relaxation ones, which it does not run.
pub const REBUILD_PROGRESS_TOKENS: &str =
    "READ\tTRIANGULATED\tDERIVED\tVALIDATED\tWROTE\tFINISHED";

fn usage() -> String {
    format!(
        "usage: {ABI_MARKER}\n\n\
         THE REQUEST\n\
         --spec           JSON resolution spec: a background spacing and a list of refinement\n\
        \x20                 regions. Regions are DATA -- adding one is a row, not a code path.\n\
        \x20                 {{\"background_km\": 120.0, \"regions\": [\n\
        \x20                   {{\"shape\": {{\"kind\": \"cap\", \"center_deg\": [39.0, -98.0],\n\
        \x20                                \"radius_km\": 1200}},\n\
        \x20                    \"spacing_km\": 20.0, \"transition_km\": 900.0}}]}}\n\
        \x20                 shapes: cap | lat_lon_box | polygon; ramp: transition_km | transition_cells\n\
         --background-km  a uniform mesh at one spacing, when no spec file is wanted\n\
         --from-centres   take the cell centres of an existing MPAS grid file and rebuild every\n\
        \x20                 derived field from them. No sizing, no relaxation: the centres are\n\
        \x20                 the request. Reads xCell/yCell/zCell, meshDensity and nominalMinDc\n\
        \x20                 by name, so an unseen mesh is a file, not a code path.\n\n\
         THE SIZE\n\
         --cells          exact cell count. No memory model is consulted, so no card is needed.\n\
         --card           which PART the mesh has to run on. Sizing from memory needs this,\n\
        \x20                 because the footprint model's fixed term is a property of the card:\n\
        \x20                 CUDA sizes the per-context local-memory backing store as\n\
        \x20                 (widest kernel frame - 1024 B) x SMs x maxThreadsPerSM, so identical\n\
        \x20                 code pays {:.0} MiB fixed on a 170 SM part and {:.0} MiB on a 70 SM one.\n\
        \x20                 MEASURED PARTS ONLY -- {} -- both at float32, nVertLevels 55, full\n\
        \x20                 physics. A part with no measured row is REFUSED by name; it is not\n\
        \x20                 given another part's number.\n\
        \x20                 {}\n\
         --vram-gib       device budget in GiB, instead of the named card's own memory (for a\n\
        \x20                 card shared with something else). Needs --card: a budget says how\n\
        \x20                 much memory, never which part.\n\
         --list-cards     print the footprint model per card, with anchors and provenance\n\
         --fit-spacing    yes rescales every spacing by ONE factor, keeping the ratios between\n\
        \x20                 them, until the mesh fits the count. Default no: a request that does\n\
        \x20                 not fit is refused rather than silently coarsened.\n\
        \x20                 With neither --cells nor --card the count comes from the spacings, and\n\
        \x20                 no footprint is reported -- there is no card-independent one.\n\n\
         THE QUALITY\n\
         --tolerance      stop when the MEAN of delta/h falls below this (default 1e-3). delta is\n\
        \x20                 how far a generator sits from its own density-weighted Voronoi\n\
        \x20                 centroid and h is the local spacing. Reaching --sweeps without\n\
        \x20                 meeting it is a refusal, not a quiet emit.\n\
         --sweeps         relaxation budget (default 300)\n\
         --omega          over-relaxation factor (default 1.4; 1.0 is plain Lloyd)\n\
         --triangulation  how the Delaunay is kept between relaxation sweeps.\n\
        \x20                 rebuild (DEFAULT, class A) rebuilds it from scratch every sweep.\n\
        \x20                 That is what every registered mesh was generated with and the\n\
        \x20                 only setting that reproduces one byte for byte.\n\
        \x20                 incremental (class B) keeps the facets and repairs them by Lawson\n\
        \x20                 flips. SAME triangulation, but each cell keeps the ring ROTATION\n\
        \x20                 it had where a rebuild re-rolls it for about 27% of cells, so the\n\
        \x20                 FILE differs. Use it for a mesh that has never existed; never to\n\
        \x20                 regenerate one with a registered SHA-256.\n\n\
         THE REGIONAL CULL\n\
         --cull-parent    cut a limited-area mesh out of an existing global grid or static\n\
        \x20                 file instead of generating one. Byte-matches the native\n\
        \x20                 MPAS-Limited-Area v2.2 cull for the same region: bdyMask rings\n\
        \x20                 0..7, parent-subset ordering, contiguous reindex, 0 sentinels\n\
        \x20                 on outermost-ring connectivity, and the METIS graph file.\n\
         --region         the piece to keep, as a Shape row -- the SAME rows a spec's\n\
        \x20                 refinement regions use, so a new region is data, never code:\n\
        \x20                 {{\"kind\": \"polygon\", \"vertices_deg\": [[50,-129],[50,-65],...]}}\n\
        \x20                 {{\"kind\": \"cap\", \"center_deg\": [39,-98], \"radius_km\": 1200}}\n\
         --graph          also write the METIS graph.info for the regional mesh\n\n\
         OUTPUT\n\
         --out            grid file to write\n\
         --receipt        write the measured receipt as JSON here as well as to stdout\n\
         --clobber        replace an existing --out\n\
         --dry-run        size and cost the request, print the receipt, write nothing\n\n\
         This produces a GRID file. Running the mesh also needs a matching STATIC file\n\
         (terrain, land use, soil, nVertLevels=55, nSoilLevels=4, FP32-bit-exact nominalMinDc),\n\
         which rw_mpas_static builds. `gpuwm mesh` runs both and writes the pair.",
        footprint::card("rtx-5090")
            .ok()
            .and_then(|c| c.fixed_mib())
            .unwrap_or(f64::NAN),
        footprint::card("rtx-5070-ti")
            .ok()
            .and_then(|c| c.fixed_mib())
            .unwrap_or(f64::NAN),
        footprint::measured_cards()
            .map(|c| c.key)
            .collect::<Vec<_>>()
            .join(" | "),
        budget_examples(&[10.0, 16.0, 32.0]),
    )
}

/// One line per budget saying what each MEASURED part holds there.
///
/// Written as a table over the cards rather than as a sentence about one,
/// because the whole point is that the same budget is a different cell count
/// on different parts.
fn budget_examples(gib: &[f64]) -> String {
    let mut lines = Vec::new();
    for &g in gib {
        let mut parts = Vec::new();
        for c in footprint::measured_cards() {
            let cells = c
                .cells_that_fit(g * 1024.0)
                .map(|n| format!("{n}"))
                .unwrap_or_else(|_| "no mesh".to_string());
            parts.push(format!("{} {}", c.key, cells));
        }
        lines.push(format!("{g:.0} GiB holds {}", parts.join(", ")));
    }
    lines.join("; ")
}

struct Args {
    map: std::collections::BTreeMap<String, String>,
    flags: std::collections::BTreeSet<String>,
}

impl Args {
    fn parse(argv: Vec<String>) -> Result<Args, String> {
        const BARE: [&str; 3] = ["clobber", "dry-run", "list-cards"];
        let mut map = std::collections::BTreeMap::new();
        let mut flags = std::collections::BTreeSet::new();
        let mut it = argv.into_iter().peekable();
        while let Some(token) = it.next() {
            if !token.starts_with("--") {
                return Err(format!("unexpected argument \"{token}\"\n\n{}", usage()));
            }
            let key = token.trim_start_matches("--").to_string();
            if BARE.contains(&key.as_str()) {
                flags.insert(key);
                continue;
            }
            let value = it
                .next()
                .ok_or_else(|| format!("--{key} needs a value\n\n{}", usage()))?;
            map.insert(key, value);
        }
        Ok(Args { map, flags })
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.map.get(key).map(String::as_str)
    }

    fn flag(&self, key: &str) -> bool {
        self.flags.contains(key)
    }

    fn number<T: std::str::FromStr>(&self, key: &str) -> Result<Option<T>, String> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v
                .parse::<T>()
                .map(Some)
                .map_err(|_| format!("--{key} is not a number: \"{v}\"")),
        }
    }
}

fn yes_no(key: &str, value: &str) -> Result<bool, String> {
    match value {
        "yes" | "true" | "YES" | "T" => Ok(true),
        "no" | "false" | "NO" | "F" => Ok(false),
        other => Err(format!("--{key} takes yes or no, not \"{other}\"")),
    }
}

fn run() -> Result<String, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv.iter().any(|a| a == "--help" || a == "-h") {
        return Err(usage());
    }
    if argv.iter().any(|a| a == "--version") {
        return Ok(concat!("rw_mpas_mesh ", env!("CARGO_PKG_VERSION")).to_string());
    }
    if argv.iter().any(|a| a == "--abi") {
        return Ok(format!(
            "{ABI_MARKER}\n{PROGRESS_TOKENS}\n{REBUILD_PROGRESS_TOKENS}"
        ));
    }
    if argv.iter().any(|a| a == "--list-cards") {
        return Ok(footprint::list_cards());
    }
    let args = Args::parse(argv)?;

    // --- which part is this being sized for? --------------------------------
    //
    // Resolved BEFORE the resolution request, because a card that has never
    // been measured has to refuse before any work is done, not after the
    // relaxation.
    let card = match args.get("card") {
        Some(key) => {
            let c = footprint::card(key)?;
            if c.measured.is_none() {
                return Err(c.unmeasured_refusal());
            }
            Some(c)
        }
        None => None,
    };

    // --- rebuilding from centres somebody else chose ------------------------
    if let Some(source) = args.get("from-centres") {
        if args.get("spec").is_some() || args.get("background-km").is_some() {
            return Err(
                "--from-centres and a resolution request both name the mesh; give one. A spec that quietly lost to a file of centres would write a mesh at a resolution nobody asked for"
                    .to_string(),
            );
        }
        for named in ["cells", "card", "vram-gib", "fit-spacing", "sweeps", "tolerance", "omega"] {
            if args.get(named).is_some() {
                return Err(format!(
                    "--{named} sizes or relaxes a mesh this route never generates: --from-centres takes the cell centres as given and rebuilds the derived fields around them. Accepting the flag and ignoring it would report a cell count or a convergence the run never had"
                ));
            }
        }
        return rebuild_from_centres(&args, source);
    }

    // --- culling a region out of a global parent ----------------------------
    //
    // The regional production path: a Shape row (the same cap / lat_lon_box /
    // polygon rows a resolution spec's regions use) cuts a limited-area mesh
    // out of an existing global grid or static file. Match target and
    // conventions live in `rw_mpas::mesh::cull`.
    if let Some(parent) = args.get("cull-parent") {
        return cull_region(&args, parent);
    }
    if args.get("region").is_some() {
        return Err(
            "--region names the piece to cut but no --cull-parent names the global \
             file to cut it from; a region without a parent has no cells"
                .to_string(),
        );
    }

    // --- the request --------------------------------------------------------
    let (spec, spec_json) = match (args.get("spec"), args.get("background-km")) {
        (Some(path), None) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("cannot read the resolution spec {path}: {e}"))?;
            let spec = MeshSpec::from_json(&text).map_err(|e| e.to_string())?;
            (spec, text)
        }
        (None, Some(km)) => {
            let km: f64 = km
                .parse()
                .map_err(|_| format!("--background-km is not a number: \"{km}\""))?;
            let spec = MeshSpec::uniform(km);
            let json = serde_json::to_string(&spec).map_err(|e| e.to_string())?;
            (spec, json)
        }
        (Some(_), Some(_)) => {
            return Err(
                "--spec and --background-km both name the resolution; give one. A spec file that quietly lost to a flag would produce a mesh nobody asked for"
                    .to_string(),
            );
        }
        (None, None) => {
            return Err(format!(
                "no resolution was given: pass --spec for a variable mesh or --background-km for a uniform one\n\n{}",
                usage()
            ));
        }
    };

    // --- the budget, and the part it is a budget ON -------------------------
    //
    // A budget with no card is refused. It says HOW MUCH memory and never
    // WHICH part, and the fixed term is a property of the part: the same
    // 16 GiB reads 79,717 cells against one measured card's fixed term and
    // 133,144 against the other's. Answering with either one silently is the
    // defect this refusal exists for.
    let budget_mib = match args.number::<f64>("vram-gib")? {
        Some(g) if g > 0.0 => {
            if card.is_none() {
                return Err(footprint::budget_without_card_refusal(g * 1024.0));
            }
            Some(g * 1024.0)
        }
        Some(g) => return Err(format!("--vram-gib is {g}; a device budget has to be positive")),
        // A card named with no explicit budget sizes against the card's own
        // memory -- what CUDA can address, never what nvidia-smi prints.
        None => card
            .filter(|_| args.get("cells").is_none())
            .map(|c| c.sizing_budget_mib()),
    };
    let fit_spacing = match args.get("fit-spacing") {
        Some(v) => yes_no("fit-spacing", v)?,
        None => false,
    };

    let mut lloyd = LloydOptions::default();
    if let Some(v) = args.number::<usize>("sweeps")? {
        lloyd.max_sweeps = v;
    }
    if let Some(v) = args.number::<f64>("tolerance")? {
        lloyd.tolerance = v;
    }
    if let Some(v) = args.number::<f64>("omega")? {
        if !(v > 0.0 && v < 2.0) {
            return Err(format!(
                "--omega is {v}; over-relaxation outside (0, 2) does not converge for any Lloyd iteration, so the run would burn its whole budget and refuse"
            ));
        }
        lloyd.omega = v;
    }
    // THE CLASS SWITCH. Default `rebuild`; see `hull::TriangulationMode` for
    // why a faster default would silently lapse every registered mesh digest.
    if let Some(v) = args.get("triangulation") {
        lloyd.triangulation = rw_mpas::mesh::hull::TriangulationMode::parse(v)?;
    }

    let request = GenerateRequest {
        spec: spec.clone(),
        target_cells: args.number::<usize>("cells")?,
        budget_mib,
        card,
        fit_spacing,
        lloyd,
        // Same source as the grid attribute this route stamps below, so the
        // gate and the file cannot disagree about how precisely this mesh is
        // stored.
        limits: Limits::for_storage(CoordinateRepresentation::for_generated_mesh()),
        ..Default::default()
    };

    // --- dry run: size and cost, write nothing ------------------------------
    if args.flag("dry-run") {
        // THE DRY RUN SIZES THE SPEC THE REAL RUN WILL BUILD, not the one that
        // was typed. `generate` snaps every region onto the background's
        // power-of-two ladder before it seeds anything, and the snap is finer
        // (up to 4x the cells in the refined region), so sizing the unsnapped
        // request would understate the cost of the very run this dry run
        // exists to price -- the "sizing said fine and generation refused
        // after 700 seconds" fault, arrived at from the other side.
        // The snap goes into the RECORD and not onto stdout. A dry run prints
        // the JSON record ALONE -- `gpuwm.mpas_mesh` and `hexcore.swath` both
        // parse the whole of stdout as JSON, and a progress line above it is
        // a `JSONDecodeError` at line 1 column 1 rather than a message anybody
        // reads.
        let (spec, dry_snap) = rw_mpas::mesh::ladder_snap::snap_to_ladder(&spec);
        let predicted = spec.predicted_cells(request.sizing_samples);
        let target = match (request.target_cells, budget_mib) {
            (Some(n), _) => n,
            // `card` is Some here: a budget with no card was already refused
            // above, and it stays a refusal on the dry run too -- a dry run
            // that answers where the real run refuses is a dry run nobody
            // can size against.
            (None, Some(mib)) => card
                .ok_or_else(|| footprint::budget_without_card_refusal(mib))?
                .cells_that_fit(mib)?,
            (None, None) => predicted.round() as usize,
        };
        let (fitted, scale) = if fit_spacing {
            spec.fitted_to(target.max(12), request.sizing_samples)
                .map_err(|e| e.to_string())?
        } else {
            (spec.clone(), 1.0)
        };
        // The gradient of the SCALED spec, which is the one that would be
        // built. Read once and reported whole: the number, what it cost, and
        // whether it is a measurement at all.
        let gradient = fitted.steepest_gradient_reading(50_000);
        let plan = serde_json::json!({
            "engine": concat!("rw-mpas ", env!("CARGO_PKG_VERSION"), " (rust)"),
            "dry_run": true,
            "spec": fitted,
            "spec_scale_applied": scale,
            // What the request had to move to reach a rung the ladder builds.
            // The `spec` above is the SNAPPED one, so a reader comparing it
            // with what they typed has this beside it to say why.
            "ladder_snap": dry_snap,
            "target_cells": target,
            "predicted_cells": fitted.predicted_cells(request.sizing_samples),
            // The card is printed BESIDE the footprint, always. A footprint
            // with no card next to it is a number nobody can check, which is
            // how one part's fixed term came to be quoted as "the measured
            // footprint model" for every part.
            "card": card.map(|c| c.key),
            "footprint_mib": card.and_then(|c| c.footprint_mib(target).ok()),
            "footprint_model": card.map(|c| serde_json::json!({
                "card": c.key,
                "display_name": c.display_name,
                "streaming_multiprocessors": c.streaming_multiprocessors,
                "max_threads_per_sm": c.max_threads_per_sm,
                "compute_capability": c.compute_capability,
                "fixed_mib": c.fixed_mib(),
                "fixed_mib_derived_local_store": c.local_store_mib(),
                "fixed_mib_measured_residue": c.measured.as_ref().map(|m| m.residue_mib),
                "widest_kernel_frame_bytes": footprint::WIDEST_KERNEL_FRAME_BYTES,
                // The frame, and every residue derived against it, belong to
                // ONE build. Reported so a receipt can be checked against the
                // checkout it describes rather than against whichever tree
                // the reader is standing in.
                "measured_against_arwen_commit": footprint::MEASURED_AGAINST_ARWEN_COMMIT,
                "frame_cut_commit": footprint::FRAME_CUT_COMMIT,
                "residue_measured_against": c.measured.as_ref().map(|m| m.residue_measured_against),
                "bytes_per_cell": c.bytes_per_cell(),
                "measured_on": c.measured.as_ref().map(|m| m.measured_on),
                "anchors": c.measured.as_ref().map(|m| m.anchors),
                "provenance": c.measured.as_ref().map(|m| m.provenance),
            })),
            "footprint_not_sized_because": if card.is_none() {
                Some("no --card was given, so no measured footprint model applies. The fixed term is a property of the part and there is no card-independent answer")
            } else {
                None
            },
            "device_budget_mib": budget_mib,
            "grid_file_bytes_estimate": emit::published_schema_bytes(target),
            "steepest_requested_gradient_percent_per_cell": gradient.per_cell * 100.0,
            // What the reading cost and whether it is a reading at all. A
            // consumer that gates on the gradient must refuse a receipt whose
            // coverage is missing or not "complete": missing means an engine
            // that measured on a global lattice, which cannot see a transition
            // narrower than its own point spacing.
            "gradient_probe_points": gradient.probe_points,
            "gradient_probe_coverage": gradient.coverage.as_receipt_word(),
            "published_reference_gradient_percent_per_cell": 1.53,
            // What each region's request will ACTUALLY deliver. The ramp is
            // centred on the region boundary, so a region narrower than a few
            // ramp widths never reaches its own nominal spacing; reporting it
            // here is what lets a caller refuse BEFORE the relaxation is
            // spent instead of discovering it in the finished file.
            "region_attainment": fitted.region_attainment(200_000),
            "deliverable_boundary": "grid file only; running this mesh also needs a matching static file",
        });
        return serde_json::to_string_pretty(&plan).map_err(|e| e.to_string());
    }

    let out: PathBuf = args
        .get("out")
        .ok_or_else(|| format!("--out was not given, and it has no default\n\n{}", usage()))?
        .into();

    // --- generate -----------------------------------------------------------
    let generated = generate(&request, |line| println!("{line}")).map_err(|e| e.to_string())?;

    let provenance = Provenance {
        spec_json: serde_json::to_string(&generated.spec).unwrap_or_else(|_| spec_json.clone()),
        request: format!(
            "{} cells, {:.3} km finest, {:.3} km background",
            generated.receipt.delivered_cells,
            generated.receipt.finest_requested_km,
            generated.receipt.background_requested_km
        ),
        // The stamped document, not the receipt: a duration inside the bytes
        // makes two identical runs write two digests, and the port registry
        // pins a grid by byte count and SHA-256.
        receipt_json: rw_mpas::mesh::provenance_json(&generated.receipt)
            .map_err(|e| e.to_string())?,
        // A GENERATED mesh has no native MPAS-A counterpart -- native MPAS-A
        // cannot produce this point set -- so no dycore byte-identity anchor
        // binds how precisely its static stores it, and the coordinate quantum
        // stops being what a fine mesh runs into.  See
        // `rw_mpas::staticfile::coordframe`.  A published grid, and a cull of
        // one, carry no such attribute and stay binary32.
        static_coordinates: Some(CoordinateRepresentation::for_generated_mesh()),
    };
    let written = rw_mpas::mesh::profile::timed(
        &rw_mpas::mesh::profile::EMIT,
        generated.mesh.n_cells as u64,
        || emit::write_grid(&generated.mesh, &out, &provenance, args.flag("clobber")),
    )
    .map_err(|e| e.to_string())?;
    println!(
        "WROTE\t{}\t{}\t{}\t{}",
        written.path.display(),
        written.bytes,
        written.file_id,
        written.sha256
    );

    let full = serde_json::json!({
        "receipt": generated.receipt,
        "emitted": written,
    });
    let json = serde_json::to_string_pretty(&full).map_err(|e| e.to_string())?;
    if let Some(path) = args.get("receipt") {
        std::fs::write(path, &json).map_err(|e| format!("cannot write {path}: {e}"))?;
    }
    println!("FINISHED");
    Ok(json)
}

use rw_mpas::mesh::gridread::read_grid_generators;

/// The `--cull-parent` route: cut a limited-area mesh from a global parent.
///
/// `--region` is a Shape row -- `{"kind": "polygon", "vertices_deg": [...]}`,
/// `cap` or `lat_lon_box` -- the same rows a resolution spec's regions carry.
/// A new region is a JSON row, never a code path.
fn cull_region(args: &Args, parent: &str) -> Result<String, String> {
    for named in [
        "spec",
        "background-km",
        "from-centres",
        "cells",
        "card",
        "vram-gib",
        "fit-spacing",
        "sweeps",
        "tolerance",
        "omega",
    ] {
        if args.get(named).is_some() {
            return Err(format!(
                "--{named} sizes or relaxes a mesh this route never generates: \
                 --cull-parent subsets an existing global file. Accepting the flag \
                 and ignoring it would report a request the cull never honoured"
            ));
        }
    }
    let region = args.get("region").ok_or_else(|| {
        format!(
            "--cull-parent needs --region SHAPE.json naming the piece to keep: a \
             cap, lat_lon_box or polygon row, e.g. \
             {{\"kind\": \"polygon\", \"vertices_deg\": [[50,-129],[50,-65],[20,-65],[20,-129]]}}\n\n{}",
            usage()
        )
    })?;
    let out: PathBuf = args
        .get("out")
        .ok_or_else(|| format!("--out was not given, and it has no default\n\n{}", usage()))?
        .into();
    let text = std::fs::read_to_string(region)
        .map_err(|e| format!("cannot read the region row {region}: {e}"))?;
    let shape: rw_mpas::mesh::Shape = serde_json::from_str(&text).map_err(|e| {
        format!(
            "the region row {region} is not a Shape: {e}. A row is \
             {{\"kind\": \"cap\"|\"lat_lon_box\"|\"polygon\", ...}}"
        )
    })?;
    if out.exists() {
        if args.flag("clobber") {
            std::fs::remove_file(&out).map_err(|e| format!("cannot replace {}: {e}", out.display()))?;
        } else {
            return Err(format!(
                "{} already exists; pass --clobber to replace it",
                out.display()
            ));
        }
    }
    let graph: Option<PathBuf> = args.get("graph").map(PathBuf::from);
    if let Some(g) = &graph {
        if g.exists() {
            if args.flag("clobber") {
                std::fs::remove_file(g).map_err(|e| format!("cannot replace {}: {e}", g.display()))?;
            } else {
                return Err(format!(
                    "{} already exists; pass --clobber to replace it",
                    g.display()
                ));
            }
        }
    }

    let receipt = rw_mpas::mesh::cull::cull_file(
        Path::new(parent),
        &shape,
        &out,
        graph.as_deref(),
    )
    .map_err(|e| e.to_string())?;

    println!(
        "CULLED\t{}\t{}\t{}",
        receipt.region_cells, receipt.region_edges, receipt.region_vertices
    );
    println!(
        "RINGS\t{}",
        receipt
            .mark
            .ring_cell_counts
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "WROTE\t{}\t{}\t{}",
        receipt.output_file, receipt.output_bytes, receipt.output_sha256
    );
    if let (Some(f), Some(s)) = (&receipt.graph_file, &receipt.graph_sha256) {
        println!("GRAPH\t{f}\t{s}");
    }

    let json = serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?;
    if let Some(path) = args.get("receipt") {
        std::fs::write(path, &json).map_err(|e| format!("cannot write {path}: {e}"))?;
    }
    println!("FINISHED");
    Ok(json)
}

/// `--from-centres`: the centres are the request.
fn rebuild_from_centres(args: &Args, source: &str) -> Result<String, String> {
    let source_path = PathBuf::from(source);
    let bytes = std::fs::metadata(&source_path)
        .map_err(|e| format!("cannot read the source mesh {source}: {e}"))?
        .len();
    let source_sha256 = {
        use sha2::Digest;
        let mut hasher = sha2::Sha256::new();
        let mut file = std::fs::File::open(&source_path)
            .map_err(|e| format!("cannot open the source mesh {source}: {e}"))?;
        std::io::copy(&mut file, &mut hasher).map_err(|e| e.to_string())?;
        format!("{:x}", hasher.finalize())
    };

    // Inherited, not minted: whether this point set has a native counterpart
    // is a property of where it came from.  An unknown tag on the source is a
    // refusal here rather than a silent binary32 default.
    let source_static_coordinates = {
        let f = netcrust::File::open(&source_path)
            .map_err(|e| format!("cannot open the source mesh {source}: {e}"))?;
        match f
            .attribute(emit::STATIC_COORDINATES_ATTR)
            .and_then(|a| a.as_string().map(|t| t.to_string()))
        {
            None => None,
            Some(tag) => Some(CoordinateRepresentation::from_tag(&tag).ok_or_else(|| {
                format!(
                    "the source mesh {source} declares {} = {tag:?}, which this build does not                      know. Carrying it forward on a guess would build the rebuilt mesh's static                      at a coordinate quantum the source never asked for, and every storage                      tolerance downstream is derived from that quantum",
                    emit::STATIC_COORDINATES_ATTR
                )
            })?),
        }
    };

    let src = read_grid_generators(&source_path)?;
    let sphere_radius = src.sphere_radius;
    let nominal_min_dc = src.nominal_min_dc;
    let cell_xyz = src.points;
    let mesh_density = src.mesh_density;

    println!(
        "READ\t{}\t{}\t{}\t{}\t{:.9}\t{:.3e}",
        source_path.display(),
        bytes,
        cell_xyz.len(),
        source_sha256,
        nominal_min_dc,
        src.max_radius_departure
    );

    if args.flag("dry-run") {
        let plan = serde_json::json!({
            "engine": concat!("rw-mpas ", env!("CARGO_PKG_VERSION"), " (rust)"),
            "dry_run": true,
            "route": "from-centres",
            "source": source_path.display().to_string(),
            "source_bytes": bytes,
            "source_sha256": source_sha256,
            "source_sphere_radius": sphere_radius,
            "max_radius_departure": src.max_radius_departure,
            "cells": cell_xyz.len(),
            "nominal_min_dc_radians": nominal_min_dc,
            "grid_file_bytes_estimate": emit::published_schema_bytes(cell_xyz.len()),
            "deliverable_boundary": "grid file only; running this mesh also needs a matching static file",
        });
        return serde_json::to_string_pretty(&plan).map_err(|e| e.to_string());
    }

    let out: PathBuf = args
        .get("out")
        .ok_or_else(|| format!("--out was not given, and it has no default\n\n{}", usage()))?
        .into();

    let rebuilt = rw_mpas::mesh::rebuild(
        cell_xyz,
        mesh_density,
        nominal_min_dc,
        // The rebuild is judged by the representation it INHERITED: a
        // published point set keeps the 200 m binary32 floor, a generated one
        // keeps the floor its own storage earns.
        Limits::for_storage(source_static_coordinates.unwrap_or_default()),
        |line| println!("{line}"),
    )
    .map_err(|e| e.to_string())?;

    // The source is named by its CONTENT, never by where it sat.  The
    // registry pins this file by sha256, so a path in the stamped bytes
    // makes one mesh into two different files depending on the
    // directory it was rebuilt from -- measured: the same source copied
    // to two names produced 5d2782cf... and 955bac3b... at an identical
    // 2,741,008 bytes.  source_sha256 already identifies the content,
    // which is the only thing a reader can act on; the path goes to the
    // side receipt, the same split the durations make.
    let provenance = Provenance {
        spec_json: serde_json::json!({
            "route": "from-centres",
            "source_sha256": source_sha256,
            "source_sphere_radius": sphere_radius,
        })
        .to_string(),
        request: format!(
            "{} cell centres rebuilt from a grid with sha256 {}",
            rebuilt.mesh.n_cells, source_sha256
        ),
        // Same rule on the rebuild route: identity into the file, duration
        // into the side receipt only.
        receipt_json: rw_mpas::mesh::provenance_json(&rebuilt.receipt)
            .map_err(|e| e.to_string())?,
        // The rebuild INHERITS its source's declaration rather than minting
        // one: `--from-centres` takes an existing point set as the request, so
        // whether that mesh has a native counterpart is a property of where
        // the centres came from and not of this run.  A published source
        // declares nothing and the rebuild declares nothing.
        static_coordinates: source_static_coordinates,
    };
    let written = emit::write_grid(&rebuilt.mesh, &out, &provenance, args.flag("clobber"))
        .map_err(|e| e.to_string())?;
    println!(
        "WROTE\t{}\t{}\t{}\t{}",
        written.path.display(),
        written.bytes,
        written.file_id,
        written.sha256
    );

    let full = serde_json::json!({
        "receipt": rebuilt.receipt,
        "source": {
            "path": source_path.display().to_string(),
            "bytes": bytes,
            "sha256": source_sha256,
            "sphere_radius": sphere_radius,
        },
        "emitted": written,
    });
    let json = serde_json::to_string_pretty(&full).map_err(|e| e.to_string())?;
    if let Some(path) = args.get("receipt") {
        std::fs::write(path, &json).map_err(|e| format!("cannot write {path}: {e}"))?;
    }
    println!("FINISHED");
    Ok(json)
}

/// The commit these bytes were built from, embedded as a plain literal so
/// the release cut can read it out of the file without executing it
/// (`tools/build_bridge_bundle.py pin --source-rev`).  `build.rs` injects
/// the value; `main` references the constant so the linker cannot discard
/// it.  Without the stamp a bundle carrying this binary cannot be pinned,
/// which is the concrete breakage: `gpuwm fetch-bridges` would stage a
/// mesh generator nobody can trace to a commit.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let started = std::time::Instant::now();
    let outcome = run();
    // Stage attribution, off unless GPUWM_MESH_PROFILE is set, and on stderr
    // so it can never contaminate the receipt JSON on stdout. Printed on the
    // refusal path too: a run that refuses after four minutes of relaxation
    // is exactly the one whose profile a caller wants.
    if rw_mpas::mesh::profile::on() {
        eprintln!(
            "{}",
            rw_mpas::mesh::profile::report(started.elapsed().as_secs_f64())
        );
    }
    match outcome {
        Ok(text) => {
            println!("{text}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("rw_mpas_mesh: {message}");
            ExitCode::FAILURE
        }
    }
}
