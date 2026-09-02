//! `rw_mpas_static` -- make the STATIC file a generated grid needs to run.
//!
//! A grid file is not runnable. It carries a mesh on a unit sphere with no
//! terrain, no land use and no soil, and every consumer that tries to run one
//! asks for a matching static and refuses without it. Until this existed, a
//! generated grid was a file nobody could use.
//!
//! It also GRADES itself: `--compare` puts a static beside a reference and
//! reports every field's agreement, so the builder is measured against a known
//! answer rather than trusted.
//!
//! ONE BUILDER, ONE DOOR. This used to be one of two static writers in the
//! crate: this one owned the schema the mesh registry pins, the other owned
//! host-memory admission and a byte-identical parallel tile decode, and one
//! edited `use` line was enough to swap which of them a user got. There is now
//! a single writer -- [`rw_mpas::static_builder`] -- and it is the streaming,
//! admission-gated one, carrying the computations this door's writer owned.
//! `rw_mpas::staticfile::schema::gate_bounded_writer_matches_pin` is what
//! holds that true.

use std::path::PathBuf;
use std::process::ExitCode;

use rw_mpas::static_builder::{
    build_static_reporting, StaticBuildConfig, StaticGeogPaths, DEFAULT_VALID_TIME,
    NOMINAL_DX_CROSS_CHECK_RELATIVE,
};
use rw_mpas::staticfile;

/// The literal a bridge contract handshakes on.
pub const ABI_MARKER: &str = "rw_mpas_static --grid GRID.nc --out STATIC.nc [--geog DIR] \
[--nominal-dx-m M] [--valid-time YYYY-MM-DD_hh:mm:ss] [--receipt JSON] [--clobber] \
| --compare REFERENCE.nc CANDIDATE.nc [--report JSON]";

/// Progress tokens this binary prints, one per stage, tab separated.
pub const PROGRESS_TOKENS: &str =
    "GRID\tNOMINALDX\tVECTORS\tOPERATORS\tGEOGGRID\tGWDBAND\tWROTE\tFINISHED";

pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

/// The geography ladder, spelled from the code that walks it.
///
/// Printed rather than described so the help and the refusal cannot drift from
/// the search: the previous help named two of the candidates and the door
/// refused archives sitting in the third.
fn geog_ladder_text(separator: &str) -> String {
    let candidates = staticfile::geog::geog_root_candidates();
    if candidates.is_empty() {
        return "(no candidate resolves: neither $GPUWM_WPS_GEOG, \
                $GPUWM_CASE_DATA_ROOT nor a home directory is set)"
            .to_string();
    }
    candidates
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(separator)
}

/// The dataset directory each field is read from, from the table itself.
fn geog_table_text(separator: &str) -> String {
    rw_mpas::static_builder::GEOG_DATASET_TABLE
        .iter()
        .map(|(slot, dataset)| format!("{slot:<17}{dataset}"))
        .collect::<Vec<_>>()
        .join(separator)
}

/// The directories `gpuwm fetch-geog` stages under a container, from the
/// table itself, so the help cannot describe a layout the builder does not
/// read.
fn geog_container_text() -> String {
    let mut containers: Vec<&str> = Vec::new();
    for (_, container) in rw_mpas::static_builder::GEOG_CONTAINER_TABLE.iter() {
        if !containers.contains(container) {
            containers.push(container);
        }
    }
    containers
        .iter()
        .map(|container| {
            let children = rw_mpas::static_builder::GEOG_CONTAINER_TABLE
                .iter()
                .filter(|(_, c)| c == container)
                .map(|(child, _)| *child)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{container}/ carries {children}")
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn usage() -> String {
    format!(
        "usage: {ABI_MARKER}\n\n\
         WHAT A STATIC IS\n\
        \x20  The same mesh as the grid, carried onto the earth radius and given geography.\n\
        \x20  Terrain, land use, soil class and composition, deep-soil temperature,\n\
        \x20  green-ness and albedo climatologies, sub-grid orography statistics for the\n\
        \x20  gravity-wave drag, the operator tables, and an FP32-bit-exact nominalMinDc.\n\n\
         THE INPUT\n\
         --grid           the grid file to build against. Its topology is copied VERBATIM, so\n\
        \x20                 the pair passes a consumer's bit-identity cross-examination.\n\
         --geog           WPS_GEOG root. Defaults to the ladder `gpuwm fetch-geog` stages\n\
        \x20                 into, best first:\n\
        \x20                   {}\n\
         --nominal-dx-m   the nominal spacing to DECLARE, in metres. Defaults to the grid's own\n\
        \x20                 implied value. Refused when it disagrees with the grid by more than\n\
        \x20                 {:.1}%, because that means grid and static are not the same mesh.\n\
         --valid-time     the xtime stamp. Default {}.\n\n\
         THE OUTPUT\n\
         --out            static file to write\n\
         --receipt        write the build receipt as JSON here\n\
         --clobber        replace an existing --out\n\n\
         GRADING\n\
         --compare A B    read two statics and report every field's agreement. Prints one line\n\
        \x20                 per field: name, count, exactly-equal, worst absolute, worst relative.\n\
         --report         write the comparison as JSON here\n\n\
         HOST MEMORY\n\
        \x20  Geography is streamed tile-plane by tile-plane and the sub-grid orography is\n\
        \x20  streamed in latitude bands, so peak host memory is bounded by the mesh plus one\n\
        \x20  band and never by the size of the archive. RW_MPAS_HOST_MEMORY_LIMIT_GIB (or\n\
        \x20  _BYTES) sets the budget the admission gate plans against; RW_MPAS_TILE_WORKERS\n\
        \x20  caps how many tiles decode at once. The static is byte-identical at every\n\
        \x20  setting of both.\n\n\
         WHERE EACH FIELD COMES FROM\n\
        \x20  {}\n\
        \x20  Each directory is read at <root>/<directory>, or at the container\n\
        \x20  `gpuwm fetch-geog` stages it under: {}.\n\n\
         WRITTEN AS A BITWISE +0 PLACEHOLDER, NOT AS VALUES\n\
        \x20  {}\n\
        \x20    the local frames and the RBF reconstruction weights belong to the vertical\n\
        \x20    grid the init stage builds; a consumer overlays the exact arrays over these\n\
        \x20    slots. The SLOT is not optional -- a reader that finds the variable absent\n\
        \x20    stops before it allocates any device memory.\n\n\
         NO LONGER ABSENT\n\
        \x20  cell_gradient_coef_x, cell_gradient_coef_y, defc_a, defc_b, soilcomp,\n\
        \x20  soilcl1, soilcl2, soilcl3, soilcl4 are all WRITTEN, with real values. They\n\
        \x20  were absent while two writers split the schema between them; they are named\n\
        \x20  here so a reader who learnt the old absent list does not go looking for a gap\n\
        \x20  that is not there.\n",
        geog_ladder_text("\n\x20                   "),
        NOMINAL_DX_CROSS_CHECK_RELATIVE * 100.0,
        DEFAULT_VALID_TIME,
        geog_table_text("\n\x20  "),
        geog_container_text(),
        staticfile::ZERO_PLACEHOLDER_FIELDS.join(", "),
    )
}

fn value(argv: &[String], flag: &str) -> Option<String> {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .cloned()
}

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let argv: Vec<String> = std::env::args().skip(1).collect();

    if argv.iter().any(|a| a == "--version" || a == "-V") {
        println!("rw_mpas_static {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    if argv.iter().any(|a| a == "--abi") {
        println!("{ABI_MARKER}\n{PROGRESS_TOKENS}");
        return ExitCode::SUCCESS;
    }
    if argv.is_empty() || argv.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", usage());
        return ExitCode::SUCCESS;
    }

    if let Some(i) = argv.iter().position(|a| a == "--compare") {
        let (Some(a), Some(b)) = (argv.get(i + 1), argv.get(i + 2)) else {
            eprintln!("--compare needs two files: REFERENCE.nc CANDIDATE.nc");
            return ExitCode::from(2);
        };
        return match staticfile::compare::compare(&PathBuf::from(a), &PathBuf::from(b)) {
            Ok(c) => {
                println!(
                    "DIMENSIONS\t{}",
                    if c.dimensions_agree { "AGREE" } else { "DIFFER" }
                );
                if !c.dimensions_agree {
                    println!("  reference: {:?}", c.reference_dimensions);
                    println!("  candidate: {:?}", c.candidate_dimensions);
                }
                println!(
                    "FIELD\tN\tEXACT\tMAX_ABS\tMAX_REL\tREF_RANGE\tCAND_RANGE"
                );
                for f in &c.fields {
                    println!(
                        "{}\t{}\t{}\t{:.6e}\t{:.6e}\t[{:.6},{:.6}]\t[{:.6},{:.6}]",
                        f.name,
                        f.n,
                        f.exact,
                        f.max_abs,
                        f.max_rel,
                        f.reference_min,
                        f.reference_max,
                        f.candidate_min,
                        f.candidate_max
                    );
                }
                if !c.only_in_reference.is_empty() {
                    println!("ONLY_IN_REFERENCE\t{}", c.only_in_reference.join(","));
                }
                if !c.only_in_candidate.is_empty() {
                    println!("ONLY_IN_CANDIDATE\t{}", c.only_in_candidate.join(","));
                }
                if !c.shape_mismatch.is_empty() {
                    println!("NOT_GRADED\t{}", c.shape_mismatch.join(","));
                }
                if let Some(p) = value(&argv, "--report") {
                    match serde_json::to_string_pretty(&c)
                        .map_err(|e| e.to_string())
                        .and_then(|s| std::fs::write(&p, s).map_err(|e| e.to_string()))
                    {
                        Ok(()) => println!("REPORT\t{p}"),
                        Err(e) => {
                            eprintln!("cannot write {p}: {e}");
                            return ExitCode::FAILURE;
                        }
                    }
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::FAILURE
            }
        };
    }

    let Some(grid) = value(&argv, "--grid") else {
        eprintln!("--grid is required. {}", ABI_MARKER);
        return ExitCode::from(2);
    };
    let Some(out) = value(&argv, "--out") else {
        eprintln!("--out is required. {}", ABI_MARKER);
        return ExitCode::from(2);
    };
    let geog = match value(&argv, "--geog").map(PathBuf::from).or_else(staticfile::geog::default_geog_root) {
        Some(p) => p,
        None => {
            eprintln!(
                "no geography root. A static carries terrain, land use and soil, so it cannot be \
                 built without one. Pass --geog DIR, run `gpuwm fetch-geog`, or put the archive \
                 at one of these -- every one was looked at and none is a directory:\n  {}",
                geog_ladder_text("\n  ")
            );
            return ExitCode::from(2);
        }
    };
    let nominal_dx_m = match value(&argv, "--nominal-dx-m").map(|s| s.parse::<f64>()) {
        Some(Ok(v)) => Some(v),
        Some(Err(e)) => {
            eprintln!("--nominal-dx-m: {e}");
            return ExitCode::from(2);
        }
        None => None,
    };

    let cfg = StaticBuildConfig {
        grid_path: PathBuf::from(&grid),
        out_path: PathBuf::from(&out),
        geog: StaticGeogPaths::under(&geog),
        supersample: 1,
        supersample_landuse: 1,
        supersample_30s: 1,
        host_memory_limit_bytes: None,
        tile_workers: None,
        receipt_path: None,
        provenance: format!("rw_mpas_static {}", env!("CARGO_PKG_VERSION")),
        valid_time: value(&argv, "--valid-time")
            .unwrap_or_else(|| DEFAULT_VALID_TIME.to_string()),
        nominal_dx_m,
        clobber: argv.iter().any(|a| a == "--clobber"),
        // The GRID says which representation its static must carry; there is
        // no flag, because the choice is a property of the mesh (does it have
        // a native MPAS-A counterpart?) and not of the command line.  A grid
        // that declares nothing gets binary32 -- every published mesh, and
        // every cull of one.
        coordinates: match rw_mpas::staticfile::coordframe::for_static_from_grid(&PathBuf::from(
            &grid,
        )) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::from(2);
            }
        },
    };

    let started = std::time::Instant::now();
    let receipt = match build_static_reporting(&cfg, &|line| println!("{line}")) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "WROTE\t{}\t{}\t{}\t{}",
        receipt.output_path,
        receipt.output_bytes,
        receipt.variables.len(),
        receipt.sha256
    );
    if let Some(p) = value(&argv, "--receipt") {
        match serde_json::to_vec_pretty(&receipt)
            .map_err(|e| e.to_string())
            .and_then(|json| std::fs::write(&p, json).map_err(|e| e.to_string()))
        {
            Ok(()) => println!("RECEIPT\t{p}"),
            Err(e) => {
                eprintln!("cannot write {p}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    println!("FINISHED\t{:.2}", started.elapsed().as_secs_f64());
    ExitCode::SUCCESS
}
