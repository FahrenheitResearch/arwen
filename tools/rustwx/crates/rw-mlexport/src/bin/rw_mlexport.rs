//! `rw_mlexport --request REQUEST.json`: carry out one ML export request.
//!
//! Progress is JSON lines on stdout (`{"event": "frame", "frame": 3, "of":
//! 49, ...}`), so every front door relays it without parsing text.  The last
//! line is `{"event": "done", ...}` on success, or `{"event": "refused" |
//! "failed", "message": ...}`, with the same sentence on stderr.
//!
//! Exit codes: 0 done, 2 refused (the message names the breakage the
//! refusal prevents), 1 failed.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use rw_mlexport::export;
use rw_mlexport::request::Request;

const USAGE: &str = "\
usage: rw_mlexport --request REQUEST.json
       rw_mlexport --abi | --help | --version

Reads wrfout-shaped history files and writes one Zarr (format 2) dataset per
domain, as the request (schema ml-export.request/v1) describes.  The request
is written by `ml-export` in the engine's command line; this binary reads no
environment.";

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from, embedded so the release cut can prove a staged
/// binary matches the commit being released by reading bytes alone.
/// `build.rs` injects the value; `main` references the constant so the
/// linker cannot discard it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

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
                println!("{}", rw_mlexport::ABI);
                return ExitCode::SUCCESS;
            }
            "--help" | "-h" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "--version" => {
                println!("rw_mlexport {} ({GPUWM_BRIDGE_SOURCE_REV_STAMP})", env!("CARGO_PKG_VERSION"));
                return ExitCode::SUCCESS;
            }
            "--request" if i + 1 < args.len() => {
                request_path = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            other => {
                eprintln!("rw_mlexport: unknown argument '{other}'\n{USAGE}");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }
    let Some(path) = request_path else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let request: Request = match std::fs::read_to_string(&path)
        .map_err(|e| e.to_string())
        .and_then(|text| serde_json::from_str(&text).map_err(|e| e.to_string()))
    {
        Ok(request) => request,
        Err(error) => {
            let message = format!(
                "the request {} is not a readable ml-export.request/v1 document ({error}), so nothing was exported",
                path.display()
            );
            emit(serde_json::json!({"event": "refused", "message": message}));
            eprintln!("rw_mlexport: {message}");
            return ExitCode::from(2);
        }
    };
    let clock = Instant::now();
    let mut progress = |value: serde_json::Value| emit(value);
    match export::execute(request, &mut progress) {
        Ok(outcome) => {
            emit(serde_json::json!({
                "event": "done",
                "domains": outcome.domains,
                "frames": outcome.frames,
                "bytes": outcome.bytes,
                "zip": outcome.zip.as_ref().map(|(p, b)| serde_json::json!({"path": p, "bytes": b})),
                "seconds": (clock.elapsed().as_secs_f64() * 1000.0).round() / 1000.0,
            }));
            ExitCode::SUCCESS
        }
        Err(error) => {
            let event = if error.is_refusal() { "refused" } else { "failed" };
            emit(serde_json::json!({"event": event, "message": error.message()}));
            eprintln!("rw_mlexport: {}", error.message());
            ExitCode::from(error.exit_code())
        }
    }
}
