//! `rw_grib2export --request REQUEST.json`: carry out one GRIB2 export.
//!
//! Progress is JSON lines on stdout (`{"event": "file", ...}`,
//! `{"event": "frame", ...}`), ending with `{"event": "done", ...}` on
//! success or `{"event": "error", "kind": "refused" | "failed", "message":
//! ...}`, with the same sentence on stderr.
//!
//! Exit codes: 0 done, 2 refused (the message names the breakage the
//! refusal prevents), 1 failed.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use rw_grib2export::request::Request;
use rw_grib2export::{export, ABI};

const USAGE: &str = "\
usage: rw_grib2export --request REQUEST.json
       rw_grib2export --list | --abi | --help | --version

Reads WRF-format WOOF history files and writes wrfsfc_dNN_<valid>.grib2 and
wrfprs_dNN_<valid>.grib2 (WMO FM 92 GRIB2) plus manifest.json, computing
every product with the clean-room WOOF post-processor (catalog woof-post/v1)
on a GPU when one has room, else on the CPU.  The request (schema
grib2-export.request/v1) is written by `export-grib2`, `go --grib2` and
`render --grib2-out`; this binary reads no environment.";

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`, embedded so the release cut
/// can prove a staged binary matches the commit being released.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str = concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn emit(value: serde_json::Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{value}");
    let _ = out.flush();
}

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut request_path: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--abi" => {
                println!("{ABI}");
                return ExitCode::SUCCESS;
            }
            "--list" => {
                println!("{}", export::list());
                return ExitCode::SUCCESS;
            }
            "--help" | "-h" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "--version" => {
                println!("rw_grib2export {} ({GPUWM_BRIDGE_SOURCE_REV_STAMP})", env!("CARGO_PKG_VERSION"));
                return ExitCode::SUCCESS;
            }
            "--request" => {
                i += 1;
                match args.get(i) {
                    Some(p) => request_path = Some(PathBuf::from(p)),
                    None => {
                        eprintln!("{USAGE}");
                        return ExitCode::from(2);
                    }
                }
            }
            other => {
                eprintln!("rw_grib2export: unknown argument {other}\n{USAGE}");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }
    let Some(path) = request_path else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let outcome = Request::read(&path).and_then(|req| {
        let pool = match req.threads {
            Some(n) => rayon::ThreadPoolBuilder::new().num_threads(n).build().ok(),
            None => None,
        };
        let mut sink = |v: serde_json::Value| emit(v);
        match pool {
            Some(p) => p.install(|| export::run(&req, &mut sink)),
            None => export::run(&req, &mut sink),
        }
    });
    match outcome {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            let kind = if e.is_refusal() { "refused" } else { "failed" };
            emit(serde_json::json!({"event": "error", "kind": kind, "message": e.message()}));
            eprintln!("rw_grib2export: {}", e.message());
            ExitCode::from(e.exit_code())
        }
    }
}
