//! Serial frame worker. Fixed presentation settings and one CUDA context survive
//! requests; imported frames and render jobs leave scope before the next request.
//! Native stdout/stderr records keep their CLI grammar. WORKER JSON records on
//! stdout delimit them. The controller must drain BOTH pipes continuously.
use serde::Deserialize;
use serde_json::json;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

const SCHEMA: &str = "gpuwm-rw-wrfbatch-worker-v1";
pub(super) const ABI_MARKER: &str = "gpuwm-rw-wrfbatch-worker-v1\t--worker\tWORKER\tready\tstarted\tresult\trejected\tstopped\tensemble-products";
const MAX_LINE: usize = 65_536;

enum Template {
    Batch(Box<super::Args>),
    Ensemble(rw_wrfbatch::ensemble_products::RenderOptions),
}

enum Work {
    Batch(Box<super::Args>),
    Ensemble(PathBuf, PathBuf),
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    EnsembleProducts {
        id: String,
        input: PathBuf,
        out_dir: PathBuf,
    },
    Render {
        id: String,
        store_root: PathBuf,
        out_dir: PathBuf,
        #[serde(default)]
        inputs: Vec<PathBuf>,
        from_import: Option<PathBuf>,
        save_import: Option<PathBuf>,
    },
    Shutdown {
        id: String,
    },
}

fn emit(value: serde_json::Value) -> Result<(), String> {
    let mut out = std::io::stdout().lock();
    // Finish all preceding native records before this boundary.
    writeln!(out, "WORKER\t{value}")
        .and_then(|_| out.flush())
        .map_err(|e| e.to_string())
}

fn memory() -> Result<(u64, u64), String> {
    let text = std::fs::read_to_string("/proc/self/status").map_err(|e| e.to_string())?;
    let bytes = |name: &str| {
        text.lines()
            .find_map(|line| {
                line.strip_prefix(name)?
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()?
                    .checked_mul(1024)
            })
            .ok_or_else(|| format!("{name} absent from /proc/self/status"))
    };
    Ok((bytes("VmRSS:")?, bytes("VmHWM:")?))
}

/// read_until alone can allocate an unbounded line before rejecting it.
fn read_request(input: &mut impl BufRead) -> Result<Option<Vec<u8>>, String> {
    let mut line = Vec::new();
    loop {
        let available = input.fill_buf().map_err(|e| e.to_string())?;
        if available.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                Err("unterminated worker request".into())
            };
        }
        let take = available
            .iter()
            .position(|b| *b == b'\n')
            .map(|n| n + 1)
            .unwrap_or(available.len());
        if line.len() + take > MAX_LINE {
            return Err("worker request exceeds 65536 bytes".into());
        }
        let complete = available[take - 1] == b'\n';
        line.extend_from_slice(&available[..take]);
        input.consume(take);
        if complete {
            return Ok(Some(line));
        }
    }
}

fn loop_requests(template: &Template, host_bytes: u64) -> Result<(), String> {
    emit(
        json!({"schema": SCHEMA, "event": "ready", "pid": std::process::id(),
        "max_inflight": 1, "max_request_bytes": MAX_LINE, "max_inputs": 64,
        "request_mode": match template { Template::Batch(_) => "render", Template::Ensemble(_) => "ensemble_products" },
        "render_threads": std::env::var("RUSTWX_RENDER_THREADS").ok(),
        "host_budget_bytes": host_bytes, "host_check_interval_ms": 100,
        "gpu_payload_budget_bytes": 268435456,
        "gpu_total_budget": "controller must admit and monitor context plus payload; GPU candidate held"}),
    )?;
    let mut input = std::io::stdin().lock();
    while let Some(line) = read_request(&mut input)? {
        let request: Request = match serde_json::from_slice(&line) {
            Ok(request) => request,
            Err(error) => {
                emit(
                    json!({"schema": SCHEMA, "event": "rejected", "id": null, "error": error.to_string()}),
                )?;
                continue;
            }
        };
        let (id, work) = match request {
            Request::Shutdown { id } => {
                emit(json!({"schema": SCHEMA, "event": "stopped", "id": id}))?;
                return Ok(());
            }
            Request::EnsembleProducts { id, input, out_dir } => {
                if !matches!(template, Template::Ensemble(_)) || id.is_empty() || id.len() > 256
                    || input.as_os_str().is_empty() || out_dir.as_os_str().is_empty() {
                    emit(json!({"schema": SCHEMA, "event": "rejected", "id": id,
                        "error": "ensemble_products needs ensemble fixed mode, id (1..256 bytes), input and out_dir"}))?;
                    continue;
                }
                (id, Work::Ensemble(input, out_dir))
            }
            Request::Render {
                id,
                store_root,
                out_dir,
                inputs,
                from_import,
                save_import,
            } => {
                let Template::Batch(template) = template else {
                    emit(json!({"schema": SCHEMA, "event": "rejected", "id": id,
                        "error": "render requests require batch fixed mode"}))?;
                    continue;
                };
                if id.is_empty()
                    || id.len() > 256
                    || inputs.len() > 64
                    || (inputs.is_empty() && from_import.is_none())
                    || store_root.as_os_str().is_empty()
                    || out_dir.as_os_str().is_empty()
                {
                    emit(json!({"schema": SCHEMA, "event": "rejected", "id": id,
                        "error": "render needs id (1..256 bytes), store_root, out_dir and 1..64 inputs or from_import"}))?;
                    continue;
                }
                let mut args = template.clone();
                args.store_root = store_root;
                args.out_dir = out_dir;
                args.inputs = inputs;
                args.from_import = from_import;
                args.save_import = save_import;
                (id, Work::Batch(args))
            }
        };
        let started = std::time::Instant::now();
        emit(json!({"schema": SCHEMA, "event": "started", "id": id}))?;
        let result = match work {
            Work::Ensemble(input, out) => {
                let Template::Ensemble(options) = template else { unreachable!() };
                options.render_at(&input, &out)
            }
            Work::Batch(work) => super::validate_request(&work)
            .map_err(|e| match e {
                super::CliError::Usage(s) | super::CliError::Failed(s) => s,
                super::CliError::Help => "help is not a render request".into(),
            })
            .and_then(|_| super::run(*work)),
        };
        // Read VmHWM too: a short allocation may occur between watchdog samples.
        let (rss, peak) = memory()?;
        let verdict = match result {
            Ok(()) if peak <= host_bytes => Ok(()),
            Ok(()) => Err(format!(
                "worker host peak {peak} exceeds budget {host_bytes}"
            )),
            Err(error) => Err(error),
        };
        emit(json!({"schema": SCHEMA, "event": "result", "id": id,
            "exit_code": if verdict.is_ok() {0} else {1}, "error": verdict.as_ref().err(),
            "elapsed_ms": started.elapsed().as_secs_f64()*1000.0,
            "rss_bytes": rss, "peak_rss_bytes": peak}))?;
        // A partial failure can leave process-wide state. Retire rather than
        // admitting another frame into a state the caller cannot account for.
        verdict?;
    }
    emit(json!({"schema": SCHEMA, "event": "stopped", "id": null, "reason": "eof"}))
}

fn run() -> Result<(), String> {
    #[cfg(not(target_os = "linux"))]
    return Err("worker RSS admission currently requires Linux /proc".into());
    let mut raw = std::env::args().skip(2);
    if raw.next().as_deref() != Some("--worker-host-mib") {
        return Err("--worker requires --worker-host-mib N before fixed render options".into());
    }
    let mib = raw
        .next()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|n| *n > 0)
        .ok_or("--worker-host-mib needs a positive integer")?;
    let host_bytes = mib.checked_mul(1048576).ok_or("host budget overflow")?;
    let mut fixed: Vec<String> = raw.collect();
    if fixed.iter().any(|s| {
        [
            "--store-root",
            "--out-dir",
            "--from-import",
            "--save-import",
            "--inputs-json",
        ]
        .contains(&s.as_str())
    }) {
        return Err("input/store/output paths belong in each worker request".into());
    }
    let sentinel = "__worker_request_path__";
    // Called before any pool/render worker exists. The controller may choose
    // fewer threads; a resident renderer cannot silently take the whole host.
    for name in [
        "RAYON_NUM_THREADS",
        "RUSTWX_RENDER_THREADS",
        "RUSTWX_DERIVED_RENDER_THREADS",
        "OMP_NUM_THREADS",
    ] {
        let n = std::env::var(name)
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4)
            .clamp(1, 4);
        // This entrypoint runs before argument/theme parsing, CUDA, Rayon, or
        // any background threads. No concurrent environment readers exist.
        unsafe {
            std::env::set_var(name, n.to_string());
        }
    }
    // Ensemble mode shares the ordinary CLI parser and render entrypoint. Only
    // the input/output paths vary between requests; no history import occurs.
    let template = if fixed.first().map(String::as_str) == Some("--ensemble-products") {
        fixed.insert(1, sentinel.to_string());
        fixed.extend(["--out-dir", sentinel].map(str::to_string));
        Template::Ensemble(rw_wrfbatch::ensemble_products::RenderOptions::parse(&fixed)?)
    } else {
        fixed.extend(["--store-root", sentinel, "--out-dir", sentinel, sentinel].map(str::to_string));
        match super::parse_args_from(fixed) {
        Ok(super::Invocation::Batch(args))
            if args.inputs == [PathBuf::from(sentinel)]
                && args.from_import.is_none()
                && args.save_import.is_none()
                && !args.list_products
                && args.difference.is_none() =>
        {
            Template::Batch(args)
        }
        _ => return Err(
            "worker fixed options must select a batch; inputs/import paths belong in each request"
                .into(),
        ),
        }
    };
    if memory()?.1 > host_bytes {
        return Err("worker starts above its host budget".into());
    }
    let done = Arc::new(AtomicBool::new(false));
    let flag = done.clone();
    let watch = std::thread::spawn(move || {
        while !flag.load(Ordering::Relaxed) {
            match memory() {
                Ok((_, peak)) if peak <= host_bytes => (),
                value => {
                    eprintln!(
                        "WORKER_RESOURCE_STOP host_budget_bytes={host_bytes} observed={value:?}"
                    );
                    std::process::exit(75);
                }
            }
            std::thread::park_timeout(std::time::Duration::from_millis(100));
        }
    });
    let result = loop_requests(&template, host_bytes);
    done.store(true, Ordering::Relaxed);
    watch.thread().unpark();
    let _ = watch.join();
    result
}

pub(super) fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("WORKER_FAILED\t{error}");
            ExitCode::FAILURE
        }
    }
}
