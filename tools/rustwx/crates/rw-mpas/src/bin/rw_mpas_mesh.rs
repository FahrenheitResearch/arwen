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

use std::path::PathBuf;
use std::process::ExitCode;

use rw_mpas::mesh::density::MeshSpec;
use rw_mpas::mesh::emit::{self, Provenance};
use rw_mpas::mesh::footprint;
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
[--receipt JSON] [--clobber] [--dry-run] [--list-cards]";

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
         --omega          over-relaxation factor (default 1.4; 1.0 is plain Lloyd)\n\n\
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

    let request = GenerateRequest {
        spec: spec.clone(),
        target_cells: args.number::<usize>("cells")?,
        budget_mib,
        card,
        fit_spacing,
        lloyd,
        limits: Limits::default(),
        ..Default::default()
    };

    // --- dry run: size and cost, write nothing ------------------------------
    if args.flag("dry-run") {
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
        let plan = serde_json::json!({
            "engine": concat!("rw-mpas ", env!("CARGO_PKG_VERSION"), " (rust)"),
            "dry_run": true,
            "spec": fitted,
            "spec_scale_applied": scale,
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
            "steepest_requested_gradient_percent_per_cell":
                fitted.steepest_gradient_per_cell(50_000) * 100.0,
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
    };
    let written = emit::write_grid(&generated.mesh, &out, &provenance, args.flag("clobber"))
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

/// Generators read off a grid file, with the numbers that let a caller check
/// the read rather than trust it.
struct Generators {
    points: Vec<rw_mpas::mesh::V3>,
    mesh_density: Vec<f64>,
    nominal_min_dc: f64,
    sphere_radius: f64,
    /// The largest `|r / sphere_radius - 1|` over every centre.
    max_radius_departure: f64,
}

/// How far the stored radii may scatter before the points are not one sphere.
///
/// The published meshes measure 6.7e-16, three double ULP. 1e-9 of a unit sphere
/// is 6.4 mm on Earth, and nothing that writes a grid file misses by that much
/// unless its points are on different spheres.
const RADIUS_SCATTER_LIMIT: f64 = 1e-9;

/// Read the cell centres, `meshDensity` and `nominalMinDc` out of a grid file.
///
/// Every name here is the MPAS spelling, so a mesh this crate has never seen is
/// a file rather than a code path.
fn read_grid_generators(path: &std::path::Path) -> Result<Generators, String> {
    let file = netcrust::File::open(path)
        .map_err(|e| format!("{} is not a netCDF file this reader can open: {e}", path.display()))?;

    // UNITS. An MPAS grid file carries sphere_radius, and on both published
    // meshes that radius is 1.0: xCell/yCell/zCell are unit vectors and
    // areaCell, dcEdge, dvEdge and nominalMinDc are unit-sphere quantities
    // despite their m and m^2 units attributes. A reader that took those for
    // metres prints spacings of 0.0 km.
    let sphere_radius = match file.attribute("sphere_radius").and_then(|a| a.as_f64()) {
        Some(r) if r > 0.0 && r.is_finite() => r,
        Some(r) => {
            return Err(format!(
                "{} declares sphere_radius = {r}; every length in the file is divided by that radius to reach the unit sphere this crate derives on, and a non-positive radius would put every generator at infinity or reflect the mesh through the origin",
                path.display()
            ));
        }
        None => {
            return Err(format!(
                "{} carries no sphere_radius global attribute, so nothing tells a reader whether xCell is a unit vector or a length in metres. Guessing wrong scales every derived dcEdge and areaCell by 6.4e6 or its inverse, and the mesh would validate clean at the wrong size",
                path.display()
            ));
        }
    };

    let read = |name: &str| -> Result<Vec<f64>, String> {
        file.read_f64(name).map_err(|e| {
            format!(
                "{} has no readable {name}: {e}",
                path.display()
            )
        })
    };
    let x = read("xCell")?;
    let y = read("yCell")?;
    let z = read("zCell")?;
    if x.len() != y.len() || y.len() != z.len() {
        return Err(format!(
            "{} carries {} xCell, {} yCell and {} zCell values; three components of one point list have to be the same length or the centres pair up component by component into positions no cell ever had",
            path.display(),
            x.len(),
            y.len(),
            z.len()
        ));
    }
    let n_cells = x.len();

    let mut points = Vec::with_capacity(n_cells);
    let mut max_radius_departure = 0.0f64;
    for i in 0..n_cells {
        let p = [x[i] / sphere_radius, y[i] / sphere_radius, z[i] / sphere_radius];
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if !(r > 0.0) || !r.is_finite() {
            return Err(format!(
                "{}: cell centre {i} is {:?}, which has no direction; a point at the origin has no place on the sphere and the hull's orientation predicate is meaningless for every facet touching it",
                path.display(),
                [x[i], y[i], z[i]]
            ));
        }
        max_radius_departure = max_radius_departure.max((r - 1.0).abs());
        points.push([p[0] / r, p[1] / r, p[2] / r]);
    }
    if max_radius_departure > RADIUS_SCATTER_LIMIT {
        return Err(format!(
            "{}: cell centres depart from sphere_radius = {sphere_radius} by up to {max_radius_departure:.3e} relative, past the {RADIUS_SCATTER_LIMIT:.0e} this reader allows. A mesh built from a mixture of radii is not a spherical Voronoi tessellation of its own generators: the circumcentres sit off the surface and every kite area is taken on a different sphere from the cell it belongs to",
            path.display()
        ));
    }

    let mesh_density = match read("meshDensity") {
        Ok(v) if v.len() == n_cells => v,
        Ok(v) => {
            return Err(format!(
                "{} carries {} meshDensity values for {n_cells} cells; MPAS scales its horizontal mixing length by meshDensity, so a mismatched table applies one cell's diffusion to another",
                path.display(),
                v.len()
            ));
        }
        Err(_) => {
            return Err(format!(
                "{} has no meshDensity variable. It records what resolution function produced these centres and MPAS scales horizontal mixing by it; inventing 1.0 everywhere would silently claim a uniform mesh and give a refined region the background diffusion length",
                path.display()
            ));
        }
    };

    let nominal_min_dc = match read("nominalMinDc") {
        Ok(v) if v.len() == 1 => v[0] / sphere_radius,
        Ok(v) => {
            return Err(format!(
                "{} carries {} nominalMinDc values; it is a single scalar stamp and a reader cannot tell which of several the matching static file was built against",
                path.display(),
                v.len()
            ));
        }
        Err(_) => {
            return Err(format!(
                "{} has no nominalMinDc variable. The mesh registry matches a grid file to its static file on an FP32-bit-exact nominalMinDc, so a made-up stamp produces a grid file no static file can ever be paired with",
                path.display()
            ));
        }
    };

    Ok(Generators {
        points,
        mesh_density,
        nominal_min_dc,
        sphere_radius,
        max_radius_departure,
    })
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
        Limits::default(),
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
    match run() {
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
