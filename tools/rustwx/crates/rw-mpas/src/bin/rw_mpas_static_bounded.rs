//! `rw_mpas_static_bounded` -- the same static builder, reached with explicit
//! geography paths and the host-memory knobs exposed.
//!
//! NOT a second writer. `rw_mpas_static` and this binary run the identical
//! `rw_mpas::static_builder` and produce the identical file; they differ only
//! in how the twelve geography directories are named. The door resolves them
//! from one `--geog` root through `StaticGeogPaths::under`, which is the
//! arbitrary-archive path; this one names each directory, which is what a
//! bring-your-own-source build and the writer's own oracle need.
//!
//! It also carries the tuning surface the door deliberately does not: the
//! supersample factors, the tile-worker cap and the host-memory limit. All
//! three change how long a build takes and none of them changes a number in
//! the file.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use rw_mpas::static_builder::{
    build_static_reporting, StaticBuildConfig, StaticGeogPaths, DEFAULT_VALID_TIME,
};

pub const ABI_MARKER: &str = "rw_mpas_static_bounded --grid GRID.nc --out STATIC.nc \
--terrain DIR --landuse DIR --soilcat DIR --greenfrac DIR --albedo DIR \
--snow-albedo DIR --soil-temperature DIR --soilcomp DIR --soilcl1 DIR \
--soilcl2 DIR --soilcl3 DIR --soilcl4 DIR [--supersample N] \
[--landuse-supersample N] [--supersample-30s N] [--tile-workers N] \
[--host-memory-limit-gib G] [--receipt JSON]";

fn usage() -> String {
    format!(
        "usage: {ABI_MARKER}\n\n\
         Every geography path is explicit: this binary does not guess a WPS_GEOG \
         product or silently substitute a lower-resolution dataset.\n\
         Host-memory admission can also be supplied through \
         RW_MPAS_HOST_MEMORY_LIMIT_GIB or RW_MPAS_HOST_MEMORY_LIMIT_BYTES.\n\
         --tile-workers (or RW_MPAS_TILE_WORKERS) caps how many geography tiles \
         are processed at once. The default is the host CPU count, bounded by \
         what host-memory admission grants. The static is byte-identical at \
         every setting."
    )
}

fn parse(argv: impl Iterator<Item = String>) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::new();
    let mut it = argv.peekable();
    while let Some(token) = it.next() {
        if token == "--help" || token == "-h" {
            return Err(usage());
        }
        if !token.starts_with("--") {
            return Err(format!("unexpected argument {token:?}\n\n{}", usage()));
        }
        let key = token.trim_start_matches("--").to_string();
        // A bare switch takes no value; consuming the next token for it would
        // silently swallow the flag that follows.
        if key == "clobber" {
            out.insert(key, "true".to_string());
            continue;
        }
        let value = it
            .next()
            .ok_or_else(|| format!("--{key} requires a value\n\n{}", usage()))?;
        if out.insert(key.clone(), value).is_some() {
            return Err(format!("--{key} was supplied more than once"));
        }
    }
    Ok(out)
}

fn need<'a>(m: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str, String> {
    m.get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("--{key} is required\n\n{}", usage()))
}

fn path(m: &BTreeMap<String, String>, key: &str) -> Result<PathBuf, String> {
    Ok(PathBuf::from(need(m, key)?))
}

fn usize_opt(m: &BTreeMap<String, String>, key: &str, default: usize) -> Result<usize, String> {
    match m.get(key) {
        None => Ok(default),
        Some(v) => v
            .parse::<usize>()
            .map_err(|_| format!("--{key} must be an integer")),
    }
}

fn run() -> Result<(), String> {
    let args = parse(std::env::args().skip(1))?;
    if args.is_empty() {
        return Err(usage());
    }
    let limit = match args.get("host-memory-limit-gib") {
        None => None,
        Some(v) => {
            let gib = v
                .parse::<f64>()
                .map_err(|_| "--host-memory-limit-gib must be numeric".to_string())?;
            if !gib.is_finite() || gib <= 0.0 {
                return Err("--host-memory-limit-gib must be finite and positive".to_string());
            }
            Some((gib * 1024.0_f64.powi(3)) as u64)
        }
    };

    let cfg = StaticBuildConfig {
        grid_path: path(&args, "grid")?,
        out_path: path(&args, "out")?,
        geog: StaticGeogPaths {
            terrain: path(&args, "terrain")?,
            landuse: path(&args, "landuse")?,
            soilcat: path(&args, "soilcat")?,
            greenfrac: path(&args, "greenfrac")?,
            albedo: path(&args, "albedo")?,
            snow_albedo: path(&args, "snow-albedo")?,
            soil_temperature: path(&args, "soil-temperature")?,
            soilcomp: path(&args, "soilcomp")?,
            soilcl1: path(&args, "soilcl1")?,
            soilcl2: path(&args, "soilcl2")?,
            soilcl3: path(&args, "soilcl3")?,
            soilcl4: path(&args, "soilcl4")?,
        },
        supersample: usize_opt(&args, "supersample", 1)?,
        supersample_landuse: usize_opt(&args, "landuse-supersample", 1)?,
        supersample_30s: usize_opt(&args, "supersample-30s", 1)?,
        host_memory_limit_bytes: limit,
        tile_workers: match args.get("tile-workers") {
            None => None,
            Some(v) => {
                let n = v
                    .parse::<usize>()
                    .map_err(|_| "--tile-workers must be an integer".to_string())?;
                if n == 0 {
                    return Err("--tile-workers must be at least 1".to_string());
                }
                Some(n)
            }
        },
        receipt_path: args.get("receipt").map(PathBuf::from),
        valid_time: args
            .get("valid-time")
            .cloned()
            .unwrap_or_else(|| DEFAULT_VALID_TIME.to_string()),
        nominal_dx_m: match args.get("nominal-dx-m") {
            None => None,
            Some(v) => Some(
                v.parse::<f64>()
                    .map_err(|_| "--nominal-dx-m must be numeric".to_string())?,
            ),
        },
        clobber: matches!(args.get("clobber").map(String::as_str), Some("true") | Some("1") | Some("yes")),
        provenance: format!(
            "rw_mpas_static_bounded builder; executable={}",
            std::env::current_exe()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "<unknown>".to_string())
        ),
    };

    let receipt =
        build_static_reporting(&cfg, &|line| eprintln!("{line}")).map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt)
            .map_err(|e| format!("receipt serialization failed: {e}"))?
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rw_mpas_static_bounded: {e}");
            ExitCode::FAILURE
        }
    }
}
