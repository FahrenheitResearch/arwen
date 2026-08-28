//! `rw_mpas_lbc` — produce the MPAS v8.4.1 lateral-boundary-condition stream
//! from WPS intermediate files, in Rust, with no Fortran in the chain.
//!
//! One invocation produces the whole boundary series: per `--interval`, one
//! `lbc.<time>.nc` carrying full-mesh fields, exactly as native
//! `init_atmosphere_model` case 9 writes them.  The producer never sees a
//! source model's name: it consumes the source-agnostic intermediates the
//! arbitrary ingest emits for every registered source, and cadence comes from
//! the caller's `--interval` rows.
//!
//! As with `rw_mpas_init`, every switch that selects physics is required and
//! closed: no default `--extrap-airtemp`, no default `--use-spechumd`, no
//! default `--oned-underflow`.
//!
//! `rw_mpas_lbc compare --ours A.nc --native B.nc` grades a produced file
//! against a native one, per variable, and reports measured ULP rather than
//! adjectives.
//!
//! ## Where the source comes in
//! `--source ROW` names a row of the driving-source registry, and the row
//! decides everything about where the numbers come from: the reader, the
//! horizontal sampler, and whether the boundary state is rebuilt from a first
//! guess or carried across from a parent run's own output.  There is no switch
//! here that names a model and none that names a route — a cascade level
//! driving the next one enters exactly the way a newly registered external
//! model does, as a row.  `rw_mpas_lbc list-sources` prints the registry.
//!
//! `--source` defaults to the incumbent intermediate row, so an invocation
//! written before the registry existed produces the same bytes it always did.

use std::path::PathBuf;
use std::process::ExitCode;

use rw_mpas::init::hinterp::Underflow;
use rw_mpas::init::vinterp::Extrap;
use rw_mpas::lbc::parent::{build_from_parent, ParentConfig};
use rw_mpas::lbc::source::{Registry, StateKind};
use rw_mpas::lbc::{build_lbc, compare, BoundaryInterval, LbcConfig};

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: see `rw_mpas_init` for why the
/// stamp exists and how the release cut reads it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

/// The literal the Python bridge contract handshakes on.
pub const ABI_MARKER: &str = "rw_mpas_lbc --grid INIT.nc --out-dir DIR \
--start-time YYYY-MM-DD_HH:MM:SS --stop-time YYYY-MM-DD_HH:MM:SS \
--interval YYYY-MM-DD_HH:MM:SS=MET_PATH [--interval ...] --nfglevels N \
--fg-interval-seconds N --extrap-airtemp MODE --use-spechumd yes|no \
--theta-adv-order N --coef-3rd-order X --oned-underflow preserve|reproduce-ifx-ftz \
[--config-attrs JSON] [--receipt JSON]\n\
rw_mpas_lbc --source ROW --grid CHILD-INIT.nc --parent-grid PARENT-INIT.nc --out-dir DIR \
--start-time YYYY-MM-DD_HH:MM:SS --stop-time YYYY-MM-DD_HH:MM:SS \
--interval YYYY-MM-DD_HH:MM:SS=PARENT_FRAME [--interval ...] --fg-interval-seconds N \
[--source-registry JSON] [--coincidence-snap yes|no] [--config-attrs JSON] [--receipt JSON]\n\
rw_mpas_lbc compare --ours LBC.nc --native LBC.nc [--receipt JSON]\n\
rw_mpas_lbc list-sources [--source-registry JSON]";

fn usage() -> String {
    format!(
        "usage: {ABI_MARKER}\n\n\
         --source             a driving-source registry row.  The row says how the source's\n\
        \x20                    files are read, how they are sampled, and whether the boundary\n\
        \x20                    state is rebuilt from a first guess or carried across from a\n\
        \x20                    parent run's own output.  Defaults to the incumbent\n\
        \x20                    self-describing intermediate.  `list-sources` prints the rows\n\
         --source-registry    a registry file merged over the built-in rows by name.  Adding a\n\
        \x20                    driving source is adding a row here, never a switch below\n\
         --grid               the child's initial-conditions file: mesh geometry, vertical\n\
        \x20                    metrics and parentage all come from it, exactly as native case 9\n\
        \x20                    reads its input stream\n\
         --parent-grid        the parent's mesh and layer heights, when its state frames do not\n\
        \x20                    carry them.  A forecast history stream normally does not\n\
         --interval           one boundary time and the file carrying it; repeat per time, in\n\
        \x20                    order.  No source model is named here, ever: cadence and\n\
        \x20                    availability are the caller's registry data\n\
         --coincidence-snap   yes | no (default yes).  A target point landing on a source point\n\
        \x20                    takes that point's value exactly.  Turning it off measures the\n\
        \x20                    transfer operator's own accuracy at a coincident point\n\
         --extrap-airtemp     constant | linear | lapse-rate  (config_extrap_airtemp)\n\
         --use-spechumd       yes | no                        (config_use_spechumd)\n\
         --oned-underflow     preserve | reproduce-ifx-ftz; a property of how the reference\n\
        \x20                    was compiled, see rw_mpas_init --help\n\
         --config-attrs       JSON object overriding header config attributes that are\n\
        \x20                    namelist metadata (met prefix, geography table names, ...);\n\
        \x20                    producer switches cannot be overridden\n\n\
         The first-guess switches have no defaults, and a row that carries a\n\
         prognostic state does not ask for them: each changes the numbers in a\n\
         file that opens cleanly and reads plausibly either way."
    )
}

struct Args {
    map: std::collections::BTreeMap<String, String>,
    intervals: Vec<(String, PathBuf)>,
}

impl Args {
    fn parse(argv: Vec<String>) -> Result<Args, String> {
        let mut map = std::collections::BTreeMap::new();
        let mut intervals = Vec::new();
        let mut it = argv.into_iter();
        while let Some(token) = it.next() {
            if !token.starts_with("--") {
                return Err(format!("unexpected argument \"{token}\"\n\n{}", usage()));
            }
            let key = token.trim_start_matches("--").to_string();
            let value = it
                .next()
                .ok_or_else(|| format!("--{key} needs a value\n\n{}", usage()))?;
            if key == "interval" {
                let Some((time, path)) = value.split_once('=') else {
                    return Err(format!(
                        "--interval takes YYYY-MM-DD_HH:MM:SS=MET_PATH, not \"{value}\""
                    ));
                };
                intervals.push((time.to_string(), PathBuf::from(path)));
            } else {
                map.insert(key, value);
            }
        }
        Ok(Args { map, intervals })
    }

    fn need(&self, key: &str) -> Result<&str, String> {
        self.map
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| format!("--{key} was not given, and it has no default\n\n{}", usage()))
    }

    fn path(&self, key: &str) -> Result<PathBuf, String> {
        Ok(PathBuf::from(self.need(key)?))
    }

    fn number<T: std::str::FromStr>(&self, key: &str) -> Result<T, String> {
        self.need(key)?
            .parse::<T>()
            .map_err(|_| format!("--{key} is not a number"))
    }
}

fn yes_no(key: &str, value: &str) -> Result<bool, String> {
    match value {
        "yes" | "true" | "YES" | "T" => Ok(true),
        "no" | "false" | "NO" | "F" => Ok(false),
        other => Err(format!("--{key} takes yes or no, not \"{other}\"")),
    }
}

fn run_compare(argv: Vec<String>) -> Result<String, String> {
    let args = Args::parse(argv)?;
    let report = compare::compare_lbc(&args.path("ours")?, &args.path("native")?)
        .map_err(|e| e.to_string())?;
    let json = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    if let Some(path) = args.map.get("receipt") {
        std::fs::write(path, &json).map_err(|e| format!("cannot write {path}: {e}"))?;
    }
    Ok(json)
}

fn load_registry(args: &Args) -> Result<Registry, String> {
    match args.map.get("source-registry") {
        None => Registry::built_in().map_err(|e| e.to_string()),
        Some(p) => Registry::with_file(std::path::Path::new(p)).map_err(|e| e.to_string()),
    }
}

fn run_list_sources(argv: Vec<String>) -> Result<String, String> {
    let args = Args::parse(argv)?;
    let registry = load_registry(&args)?;
    serde_json::to_string_pretty(&registry).map_err(|e| e.to_string())
}

fn run_from_parent(args: &Args, row_name: &str) -> Result<String, String> {
    let cfg = ParentConfig {
        grid_path: args.path("grid")?,
        parent_grid: args.map.get("parent-grid").map(PathBuf::from),
        out_dir: args.path("out-dir")?,
        start_time: args.need("start-time")?.to_string(),
        stop_time: args.need("stop-time")?.to_string(),
        intervals: args
            .intervals
            .iter()
            .map(|(t, p)| BoundaryInterval {
                valid_time: t.clone(),
                met_path: p.clone(),
            })
            .collect(),
        fg_interval_seconds: args.number("fg-interval-seconds")?,
        source_row: row_name.to_string(),
        registry_path: args.map.get("source-registry").map(PathBuf::from),
        attr_overrides: attr_overrides(args)?,
        provenance: format!(
            "rw_mpas_lbc from {}",
            std::env::current_exe()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "an unnamed tree".to_string())
        ),
        without_snap: !yes_no(
            "coincidence-snap",
            args.map
                .get("coincidence-snap")
                .map(String::as_str)
                .unwrap_or("yes"),
        )?,
    };
    let receipt = build_from_parent(&cfg).map_err(|e| e.to_string())?;
    let json = serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?;
    if let Some(path) = args.map.get("receipt") {
        std::fs::write(path, &json).map_err(|e| format!("cannot write {path}: {e}"))?;
    }
    Ok(json)
}

fn attr_overrides(
    args: &Args,
) -> Result<std::collections::BTreeMap<String, serde_json::Value>, String> {
    match args.map.get("config-attrs") {
        None => Ok(Default::default()),
        Some(text) => {
            let value: serde_json::Value = serde_json::from_str(text)
                .map_err(|e| format!("--config-attrs is not JSON: {e}"))?;
            let serde_json::Value::Object(map) = value else {
                return Err("--config-attrs must be a JSON object of name: value".to_string());
            };
            Ok(map.into_iter().collect())
        }
    }
}

fn run_produce(argv: Vec<String>) -> Result<String, String> {
    let args = Args::parse(argv)?;
    if args.intervals.is_empty() {
        return Err(format!(
            "no --interval was given; there is nothing to produce\n\n{}",
            usage()
        ));
    }

    // The row decides the route.  Nothing below this line names a model, and
    // nothing names a route either.
    let registry = load_registry(&args)?;
    let row_name = args
        .map
        .get("source")
        .cloned()
        .unwrap_or_else(|| rw_mpas::lbc::source::DEFAULT_ROW.to_string());
    let row = registry.row(&row_name).map_err(|e| e.to_string())?;
    if row.state == StateKind::Prognostic {
        return run_from_parent(&args, &row_name);
    }

    let attr_overrides = attr_overrides(&args)?;

    let cfg = LbcConfig {
        grid_path: args.path("grid")?,
        out_dir: args.path("out-dir")?,
        start_time: args.need("start-time")?.to_string(),
        stop_time: args.need("stop-time")?.to_string(),
        intervals: args
            .intervals
            .iter()
            .map(|(t, p)| BoundaryInterval {
                valid_time: t.clone(),
                met_path: p.clone(),
            })
            .collect(),
        n_fg_levels: args.number("nfglevels")?,
        extrap_airtemp: Extrap::parse(args.need("extrap-airtemp")?)?,
        use_spechumd: yes_no("use-spechumd", args.need("use-spechumd")?)?,
        theta_adv_order: args.number("theta-adv-order")?,
        coef_3rd_order: args.number("coef-3rd-order")?,
        fg_interval_seconds: args.number("fg-interval-seconds")?,
        oned_underflow: Underflow::parse(args.need("oned-underflow")?)?,
        attr_overrides,
        provenance: format!(
            "rw_mpas_lbc from {}",
            std::env::current_exe()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "an unnamed tree".to_string())
        ),
    };

    let receipt = build_lbc(&cfg).map_err(|e| e.to_string())?;
    let json = serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?;
    if let Some(path) = args.map.get("receipt") {
        std::fs::write(path, &json).map_err(|e| format!("cannot write {path}: {e}"))?;
    }
    Ok(json)
}

fn run() -> Result<String, String> {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv.iter().any(|a| a == "--help" || a == "-h") {
        return Err(usage());
    }
    // The handshake its four crate siblings answer, and for the same
    // reason: this binary is in the release bundle, so a copy of it built
    // before an argument-vector change still launches, still prints a
    // usage line, and would still be blessed as staged-and-fine.  `--abi`
    // is what makes "speaks this release's contract" a question a caller
    // can ask without running a boundary series first.  Answered here,
    // before Args::parse, because that parser reads every `--flag` as
    // taking a value and would refuse `--abi` for want of one.
    if argv.iter().any(|a| a == "--abi") {
        return Ok(ABI_MARKER.to_string());
    }
    if argv.iter().any(|a| a == "--version") {
        return Ok(format!("rw_mpas_lbc {}", env!("CARGO_PKG_VERSION")));
    }
    if argv[0] == "compare" {
        argv.remove(0);
        return run_compare(argv);
    }
    if argv[0] == "list-sources" {
        argv.remove(0);
        return run_list_sources(argv);
    }
    run_produce(argv)
}

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    match run() {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("rw_mpas_lbc: {message}");
            ExitCode::FAILURE
        }
    }
}
