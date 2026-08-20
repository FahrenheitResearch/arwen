//! `rw_mpas_init` — build an MPAS initial-conditions file from a WPS
//! intermediate file, in Rust, with no Fortran in the chain.
//!
//! Every switch that selects physics is required and closed.  There is no
//! default `--extrap-airtemp`, no default `--use-spechumd`, and no default
//! `--virtual-factor`: each of them changes the numbers in a file that will
//! look structurally perfect either way, and this project has already shipped
//! one silent default that disabled three repairs at once.

use std::path::PathBuf;
use std::process::ExitCode;

use rw_mpas::init::dynamics::VirtualFactor;
use rw_mpas::init::hinterp::Underflow;
use rw_mpas::init::soil::DeepMoisture;
use rw_mpas::init::vinterp::Extrap;
use rw_mpas::init::{build_init, InitConfig};

/// The literal the Python bridge contract handshakes on.  It spells the
/// argument vector out, so a stale binary on a user's disk fails the static
/// check before it can accept input it will mis-parse.
pub const ABI_MARKER: &str = "rw_mpas_init --met MET --static STATIC.nc --capsule CAPSULE.nc \
--reference REFERENCE.nc --out INIT.nc --start-time YYYY-MM-DD_HH:MM:SS --nfglevels N \
--nfgsoillevels N --extrap-airtemp MODE --use-spechumd yes|no --theta-adv-order N \
--coef-3rd-order X --virtual-factor reproduce-fortran|consistent \
--deep-soil-moisture reproduce-fortran|corrected --landuse-table NAME \
--frac-seaice yes|no --tsk-seaice-threshold K \n--oned-underflow preserve|reproduce-ifx-ftz [--receipt JSON]";

fn usage() -> String {
    format!(
        "usage: {ABI_MARKER}\n\n\
         --extrap-airtemp     constant | linear | lapse-rate  (config_extrap_airtemp)\n\
         --use-spechumd       yes | no                        (config_use_spechumd)\n\
         --virtual-factor     reproduce-fortran keeps the (rvord-1)/1.61 mix the Fortran has;\n\
        \x20                    consistent uses rvord-1 throughout so the theta round trip closes\n\
         --deep-soil-moisture reproduce-fortran keeps sm_input(nFGSoilLevels), which is the\n\
        \x20                    second-deepest first-guess level; corrected uses the deepest\n\n\
         --oned-underflow     preserve keeps the subnormal 1e-20*1e-20 inside oned's guards,
\n                             which is what the source says and what ifx -O0, ifx -no-ftz and
\n                             gfortran -O3 all produce; reproduce-ifx-ftz flushes it, which is
\n                             what the reference build (mpiifx -O3, -ftz on by default) does

\n         None of these has a default.  Each changes the numbers in a file that\n\
         opens cleanly and reads plausibly either way."
    )
}

struct Args {
    map: std::collections::BTreeMap<String, String>,
}

impl Args {
    fn parse(argv: Vec<String>) -> Result<Args, String> {
        let mut map = std::collections::BTreeMap::new();
        let mut it = argv.into_iter();
        while let Some(token) = it.next() {
            if !token.starts_with("--") {
                return Err(format!("unexpected argument \"{token}\"\n\n{}", usage()));
            }
            let key = token.trim_start_matches("--").to_string();
            let value = it
                .next()
                .ok_or_else(|| format!("--{key} needs a value\n\n{}", usage()))?;
            map.insert(key, value);
        }
        Ok(Args { map })
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

fn run() -> Result<String, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv.iter().any(|a| a == "--help" || a == "-h") {
        return Err(usage());
    }
    let args = Args::parse(argv)?;

    let cfg = InitConfig {
        met_path: args.path("met")?,
        static_path: args.path("static")?,
        capsule_path: args.path("capsule")?,
        reference_path: args.path("reference")?,
        out_path: args.path("out")?,
        start_time: args.need("start-time")?.to_string(),
        n_fg_levels: args.number("nfglevels")?,
        n_fg_soil_levels: args.number("nfgsoillevels")?,
        extrap_airtemp: Extrap::parse(args.need("extrap-airtemp")?)?,
        use_spechumd: yes_no("use-spechumd", args.need("use-spechumd")?)?,
        theta_adv_order: args.number("theta-adv-order")?,
        coef_3rd_order: args.number("coef-3rd-order")?,
        virtual_factor: match args.need("virtual-factor")? {
            "reproduce-fortran" => VirtualFactor::ReproduceFortran,
            "consistent" => VirtualFactor::Consistent,
            other => {
                return Err(format!(
                    "--virtual-factor takes reproduce-fortran or consistent, not \"{other}\""
                ))
            }
        },
        deep_moisture: match args.need("deep-soil-moisture")? {
            "reproduce-fortran" => DeepMoisture::ReproduceFortran,
            "corrected" => DeepMoisture::Corrected,
            other => {
                return Err(format!(
                    "--deep-soil-moisture takes reproduce-fortran or corrected, not \"{other}\""
                ))
            }
        },
        landuse_table: args.need("landuse-table")?.to_string(),
        frac_seaice: yes_no("frac-seaice", args.need("frac-seaice")?)?,
        tsk_seaice_threshold: args.number("tsk-seaice-threshold")?,
        oned_underflow: Underflow::parse(args.need("oned-underflow")?)?,
        provenance: format!(
            "rw_mpas_init from {}",
            std::env::current_exe()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "an unnamed tree".to_string())
        ),
    };

    let receipt = build_init(&cfg).map_err(|e| e.to_string())?;
    let json = serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?;
    if let Some(path) = args.map.get("receipt") {
        std::fs::write(path, &json).map_err(|e| format!("cannot write {path}: {e}"))?;
    }
    Ok(json)
}

fn main() -> ExitCode {
    match run() {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("rw_mpas_init: {message}");
            ExitCode::FAILURE
        }
    }
}
