//! `rw_fetch` -- ArWen's Rust fetch backbone.
//!
//! A thin, fail-closed CLI over the download stack already vendored at
//! `tools/rustwx/vendor/wx-core`: 16 MiB parallel whole-file range
//! GETs, `.idx` range coalescing, the cross-process NOMADS rate
//! governor, and the two-tier disk cache.  Model URL algebra comes from
//! `rustwx-models`, so every model in that registry -- HRRR, GFS, GDAS
//! and twenty more -- is addressable through one surface with
//! priority-ordered source fallback.
//!
//! It moves bytes and reports facts.  It does **not** own durability:
//! the `gpuwm-fetch-manifest-v1` manifest, the resume identity guard,
//! the quarantine rule and the record-count bars all stay in
//! `gpuwm/fetch.py`, on top of the record this prints to stdout.
//!
//! ```text
//! rw_fetch fetch  --model hrrr --date 20260728 --cycle 12 --hours 0-12 \
//!                 --product wrfnat --out DIR [--mode auto] [--source aws]
//! rw_fetch probe  --model hrrr --date 20260728 --cycle 12 --hours 0-1 --product wrfnat
//! rw_fetch latest --model hrrr --product wrfnat --through 12
//! ```

mod net;
mod plan;
mod record;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::str::FromStr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use wx_core::download::TransferProgress;

use rustwx_core::{CycleSpec, ModelId, ModelRunRequest, ResolvedUrl, SourceId};

use net::{grib_framed, hex_digest, publish, select_contains, select_exact, Fetcher};
use plan::{decide, Decision, Mode, ModeRequest};
use record::{
    CycleRecord, DedupRecord, FetchRecord, FileRecord, LatestReport, ProbeHour, ProbeRecord,
    ProbeReport, FETCH_RECORD_ABI, FETCH_RECORD_SCHEMA, LATEST_REPORT_SCHEMA,
    PROBE_REPORT_SCHEMA,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from, embedded so the gpuwm release cut can prove a
/// staged bridge matches the commit being released by reading bytes
/// alone (`tools/build_bridge_bundle.py pin --source-rev`).  `build.rs`
/// injects the value; `main` references the constant so the linker
/// cannot discard it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

const USAGE: &str = "\
usage: rw_fetch <fetch|probe|latest> [OPTIONS]
       rw_fetch --version | --help | --abi

  fetch    download a model-run window and print a fetch record
  probe    report the transport decision for each hour, moving no payload
  latest   report the newest cycle serving every hour through --through

common options
  --model NAME            hrrr, gfs, gdas, rap, nam, ... (rustwx model id)
  --date YYYYMMDD         cycle date
  --cycle HH              cycle hour, 0-23
  --hours SPEC            0-12, 0-12:3, or 0,3,6 (combinable with commas)
  --through N             latest: highest forecast hour that must be present
  --product TOKEN         wrfnat, wrfprs, pgrb2.0p25, ...  (model default if omitted)
  --source NAME           nomads|aws|google|azure|ecmwf|ncei|gdex (default: priority order)

fetch options
  --mode MODE             auto (default) | full-file | idx-subset
  --var-pattern PAT       exact VAR:LEVEL selector, repeatable; the variable
                          half may alternate (CLMR|CLWMR:1 hybrid level)
  --var-pattern-file F    one selector per line ('#' comments, blanks skipped)
  --var-pattern-contains PAT
                          wx-core substring-level selector, repeatable
  --exclude-forecast-contains SUBSTR
                          drop index rows whose raw line contains SUBSTR
                          (e.g. 'acc fcst'), repeatable
  --out DIR               destination directory (created if absent)
  --cache-dir DIR         wx-core disk cache root
  --keep-idx              write the .idx beside each object (default on for
                          idx-subset transfers)

The transport decision is probe-based and carries no time constants: if
the object is present and its .idx is absent, malformed, or provably
shorter than the object, the whole file is taken.  --mode overrides.

While an object moves, fetch prints `rw_fetch-progress fHHH RECEIVED TOTAL`
on stderr about once a second (TOTAL is - until the size is known).

environment
  RUSTWX_DOWNLOAD_STREAMS N       chunk streams per object (default 16)
  RUSTWX_DOWNLOAD_STALL_SECONDS S a connection that delivers under 64 KiB
                                  in S seconds (default 30) is dropped and
                                  its chunk resumed from the byte it reached
";

/// Exit status of a command line this binary could not act on.
const EXIT_USAGE: u8 = 2;
/// Exit status of a payload transfer that failed after the download
/// client's own retries.  A network fault, so the caller may ask again.
const EXIT_TRANSFER: u8 = 3;
/// Exit status of every other refusal.
const EXIT_REFUSED: u8 = 1;

/// Why a subcommand produced no document.
///
/// Three kinds, because the caller acts on each differently: a usage
/// error is a defect in whoever built the command line, a transfer
/// failure is the network and worth another attempt, and a refusal is
/// a fact about the request that another attempt would only repeat.
/// Every failure used to exit 2 and print the whole usage text after
/// its reason, so a dropped connection 38 minutes into a download read
/// as a malformed command line, and the reason itself was lost (see
/// `failure_text`).
#[derive(Debug, PartialEq)]
enum Failure {
    Usage(String),
    Transfer(String),
    Refused(String),
}

impl Failure {
    fn message(&self) -> &str {
        match self {
            Failure::Usage(text) | Failure::Transfer(text) | Failure::Refused(text) => text,
        }
    }

    fn exit_status(&self) -> u8 {
        match self {
            Failure::Usage(_) => EXIT_USAGE,
            Failure::Transfer(_) => EXIT_TRANSFER,
            Failure::Refused(_) => EXIT_REFUSED,
        }
    }
}

impl From<net::TransferFault> for Failure {
    /// Only a transfer the network ended is `Transfer` (exit 3, worth
    /// another attempt).  A 4xx, a 200 to a range request, a foreign
    /// span or a byte count that does not add up is the origin's answer
    /// and stays a refusal (exit 1): retrying repeats it, and a
    /// "the network cut it off" remedy would send the reader after the
    /// wrong fault.
    fn from(fault: net::TransferFault) -> Self {
        match fault {
            net::TransferFault::Network(text) => Failure::Transfer(text),
            net::TransferFault::Refused(text) => Failure::Refused(text),
        }
    }
}

impl From<String> for Failure {
    fn from(text: String) -> Self {
        Failure::Refused(text)
    }
}

/// The stderr a failure prints: its reason on a line of its own.
///
/// It starts with a newline on purpose.  wx-core draws transfer
/// progress as `\r  Downloading chunks N/M...` with no line end, so a
/// reason printed straight after it was glued onto that indented
/// progress line.  A reader that splits on carriage returns and skips
/// indented lines (gpuwm's does, to skip the usage text) then never saw
/// the reason at all and reported the first usage heading instead:
/// `rw_fetch fetch: common options`.  No usage text follows any more;
/// a usage error points at `--help`, which prints it.
fn failure_text(failure: &Failure) -> String {
    let mut text = format!("\nrw_fetch: {}\n", failure.message());
    if matches!(failure, Failure::Usage(_)) {
        text.push_str("rw_fetch --help lists every option\n");
    }
    text
}

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(document) => {
            println!("{document}");
            ExitCode::SUCCESS
        }
        Err(failure) => {
            eprint!("{}", failure_text(&failure));
            if args.is_empty() {
                // The one failure that is answered with the usage text:
                // the packager's no-argument identity probe demands its
                // `usage: rw_fetch` marker.
                eprintln!("{USAGE}");
            }
            ExitCode::from(failure.exit_status())
        }
    }
}

fn run(args: &[String]) -> Result<String, Failure> {
    if args.is_empty() {
        return Err(Failure::Usage("no subcommand given".to_string()));
    }
    let options = || Options::parse(&args[1..]).map_err(Failure::Usage);
    match args[0].as_str() {
        "--version" | "-V" => Ok(format!("rw_fetch {VERSION}")),
        "--help" | "-h" => Ok(USAGE.to_string()),
        // The exact-ABI marker `gpuwm.native_wrf_distribution` greps the
        // binary for.  Printing it here is what guarantees the literal
        // is in the built image.
        "--abi" => Ok(FETCH_RECORD_ABI.to_string()),
        "fetch" => command_fetch(&options()?),
        "probe" => Ok(command_probe(&options()?)?),
        "latest" => Ok(command_latest(&options()?)?),
        other => Err(Failure::Usage(format!("unknown subcommand {other:?}"))),
    }
}

// ──────────────────────────────────────────────────────────
// Options
// ──────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct Options {
    model: Option<String>,
    date: Option<String>,
    cycle: Option<u8>,
    hours: Vec<u16>,
    through: Option<u16>,
    product: Option<String>,
    source: Option<String>,
    mode: ModeRequest,
    exact_patterns: Vec<String>,
    contains_patterns: Vec<String>,
    exclusions: Vec<String>,
    out: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    keep_idx: bool,
}

impl Default for ModeRequest {
    fn default() -> Self {
        Self::Auto
    }
}

/// Consume the value that follows `args[*index]`, advancing the cursor.
fn value_of(args: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut options = Options::default();
        let mut index = 0usize;
        while index < args.len() {
            let flag = args[index].clone();
            let flag = flag.as_str();
            macro_rules! value {
                () => {
                    value_of(args, &mut index, flag)?
                };
            }
            match flag {
                "--model" => options.model = Some(value!()),
                "--date" => options.date = Some(value!()),
                "--cycle" => {
                    let raw = value!();
                    options.cycle = Some(
                        raw.parse()
                            .map_err(|_| format!("--cycle {raw:?} is not an hour 0-23"))?,
                    );
                }
                "--hours" => options.hours = parse_hours(&value!())?,
                "--through" => {
                    let raw = value!();
                    options.through = Some(
                        raw.parse()
                            .map_err(|_| format!("--through {raw:?} is not a forecast hour"))?,
                    );
                }
                "--product" => options.product = Some(value!()),
                "--source" => options.source = Some(value!()),
                "--mode" => options.mode = ModeRequest::parse(&value!())?,
                "--var-pattern" => options.exact_patterns.push(value!()),
                "--var-pattern-contains" => options.contains_patterns.push(value!()),
                "--exclude-forecast-contains" => options.exclusions.push(value!()),
                "--var-pattern-file" => {
                    let path = value!();
                    let text = std::fs::read_to_string(&path)
                        .map_err(|error| format!("could not read {path}: {error}"))?;
                    for line in text.lines() {
                        let line = line.trim();
                        if line.is_empty() || line.starts_with('#') {
                            continue;
                        }
                        options.exact_patterns.push(line.to_string());
                    }
                }
                "--out" => options.out = Some(PathBuf::from(value!())),
                "--cache-dir" => options.cache_dir = Some(PathBuf::from(value!())),
                "--keep-idx" => options.keep_idx = true,
                other => return Err(format!("unknown option {other:?}")),
            }
            index += 1;
        }
        Ok(options)
    }

    fn model(&self) -> Result<ModelId, String> {
        let raw = self
            .model
            .as_deref()
            .ok_or_else(|| "--model is required".to_string())?;
        ModelId::from_str(raw).map_err(|error| format!("--model {raw:?}: {error}"))
    }

    fn cycle_spec(&self) -> Result<CycleSpec, String> {
        let date = self
            .date
            .as_deref()
            .ok_or_else(|| "--date YYYYMMDD is required".to_string())?;
        let hour = self
            .cycle
            .ok_or_else(|| "--cycle HH is required".to_string())?;
        CycleSpec::new(date, hour).map_err(|error| format!("{error}"))
    }

    fn source(&self) -> Result<Option<SourceId>, String> {
        match self.source.as_deref() {
            None => Ok(None),
            Some(raw) => SourceId::from_str(raw)
                .map(Some)
                .map_err(|error| format!("--source {raw:?}: {error}")),
        }
    }

    fn patterns(&self) -> usize {
        self.exact_patterns.len() + self.contains_patterns.len()
    }
}

/// `0-12`, `0-12:3`, `0,3,6`, or any comma-joined mixture.
fn parse_hours(spec: &str) -> Result<Vec<u16>, String> {
    let mut hours: Vec<u16> = Vec::new();
    for token in spec.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let (range, step) = match token.split_once(':') {
            Some((range, step)) => (
                range,
                step.parse::<u16>()
                    .map_err(|_| format!("--hours step {step:?} is not a number"))?,
            ),
            None => (token, 1u16),
        };
        if step == 0 {
            return Err("--hours step must be positive".to_string());
        }
        match range.split_once('-') {
            Some((first, last)) => {
                let first: u16 = first
                    .trim()
                    .parse()
                    .map_err(|_| format!("--hours {token:?} has a non-numeric start"))?;
                let last: u16 = last
                    .trim()
                    .parse()
                    .map_err(|_| format!("--hours {token:?} has a non-numeric end"))?;
                if last < first {
                    return Err(format!("--hours {token:?} runs backwards"));
                }
                let mut hour = first;
                while hour <= last {
                    hours.push(hour);
                    hour += step;
                }
            }
            None => hours.push(
                range
                    .trim()
                    .parse()
                    .map_err(|_| format!("--hours {token:?} is not a forecast hour"))?,
            ),
        }
    }
    if hours.is_empty() {
        return Err("--hours selected no forecast hours".to_string());
    }
    hours.sort_unstable();
    hours.dedup();
    Ok(hours)
}

// ──────────────────────────────────────────────────────────
// URL resolution
// ──────────────────────────────────────────────────────────

fn product_for(model: ModelId, options: &Options) -> Result<String, String> {
    let product = options
        .product
        .clone()
        .unwrap_or_else(|| rustwx_models::model_summary(model).default_product.to_string());
    check_product_token(model, &product)?;
    Ok(product)
}

/// `(spelling accepted here, token `rustwx-models` understands, filename
/// fragment the built URL must contain)`.
///
/// Two columns and not one, because the registry's own vocabulary is
/// the short form: `build_hrrr_url` matches `"nat"`, never `"wrfnat"`,
/// and ArWen has said `wrfnat` since long before there was a registry.
///
/// The table exists at all because `build_hrrr_url` ends in
/// `_ => "wrfsfc"`: an unrecognised product token does **not** fail
/// there, it silently becomes the surface file.  A caller who asked for
/// native levels and got a 2-D object back would find out at decode
/// time, three bars later, with a confusing inventory error.  Every
/// other model ArWen drives (GFS, GDAS) already returns
/// `UnsupportedProduct`, so this guard is deliberately HRRR-shaped
/// rather than generic.
const HRRR_PRODUCT_TOKENS: &[(&str, &str, &str)] = &[
    ("sfc", "sfc", "wrfsfc"),
    ("surface", "sfc", "wrfsfc"),
    ("wrfsfc", "sfc", "wrfsfc"),
    ("prs", "prs", "wrfprs"),
    ("pressure", "prs", "wrfprs"),
    ("wrfprs", "prs", "wrfprs"),
    ("nat", "nat", "wrfnat"),
    ("native", "nat", "wrfnat"),
    ("wrfnat", "nat", "wrfnat"),
    ("subh", "subh", "wrfsubh"),
    ("subhourly", "subh", "wrfsubh"),
    ("wrfsubh", "subh", "wrfsubh"),
];

fn hrrr_product(product: &str) -> Option<(&'static str, &'static str)> {
    let lowered = product.to_ascii_lowercase();
    HRRR_PRODUCT_TOKENS
        .iter()
        .find(|(spelling, _, _)| *spelling == lowered)
        .map(|(_, token, fragment)| (*token, *fragment))
}

/// Spell a product the way `rustwx-models` expects it.
fn normalize_product(model: ModelId, product: &str) -> String {
    match model {
        ModelId::Hrrr | ModelId::HrrrAk => hrrr_product(product)
            .map(|(token, _)| token.to_string())
            .unwrap_or_else(|| product.to_string()),
        _ => product.to_string(),
    }
}

/// The filename fragment the built URL must contain for this product.
fn product_url_fragment(model: ModelId, product: &str) -> Option<&'static str> {
    match model {
        ModelId::Hrrr | ModelId::HrrrAk => {
            hrrr_product(product).map(|(_, fragment)| fragment)
        }
        _ => None,
    }
}

fn check_product_token(model: ModelId, product: &str) -> Result<(), String> {
    if !matches!(model, ModelId::Hrrr | ModelId::HrrrAk) {
        return Ok(());
    }
    if hrrr_product(product).is_some() {
        return Ok(());
    }
    let mut accepted: Vec<&str> = HRRR_PRODUCT_TOKENS
        .iter()
        .map(|(spelling, _, _)| *spelling)
        .collect();
    accepted.sort_unstable();
    accepted.dedup();
    Err(format!(
        "--product {product:?} is not an HRRR product token; the URL builder would \
         silently fall back to the surface file.  Accepted: {}",
        accepted.join(", ")
    ))
}

/// Candidate (source, grib, idx) triples for one hour, best first.
fn candidates(
    model: ModelId,
    cycle: &CycleSpec,
    hour: u16,
    product: &str,
    only: Option<SourceId>,
) -> Result<Vec<ResolvedUrl>, String> {
    let normalized = normalize_product(model, product);
    let request = ModelRunRequest::new(model, cycle.clone(), hour, normalized.clone())
        .map_err(|error| format!("{error}"))?;
    let resolved = rustwx_models::resolve_urls(&request).map_err(|error| format!("{error}"))?;
    let filtered: Vec<ResolvedUrl> = match only {
        Some(wanted) => resolved
            .into_iter()
            .filter(|item| item.source == wanted)
            .collect(),
        None => resolved,
    };
    if filtered.is_empty() {
        return Err(format!(
            "no source serves {model} {product} f{hour:03} (check --source)"
        ));
    }
    // Belt and braces for the HRRR fallback described on
    // HRRR_PRODUCT_TOKENS: the built URL must actually name the product
    // that was asked for.
    if let Some(fragment) = product_url_fragment(model, product) {
        for candidate in &filtered {
            if !candidate.grib_url.contains(fragment) {
                return Err(format!(
                    "the {} URL for product {product:?} does not name {fragment:?}: {}",
                    candidate.source, candidate.grib_url
                ));
            }
        }
    }
    Ok(filtered)
}

fn object_name(grib_url: &str) -> String {
    grib_url
        .split('?')
        .next()
        .unwrap_or(grib_url)
        .rsplit('/')
        .next()
        .unwrap_or("object.grib2")
        .to_string()
}

/// Why no source served hour `hour`, typed by whose doing it was.
///
/// A source whose existence probe got no answer is the network's doing,
/// so the whole failure is a transfer (exit 3) that the caller may ask
/// again, even beside a source that answered "not here": the unreachable
/// one may well have the object.  It used to read as "no source served
/// this object" (exit 1), which a caller rightly never retries.
fn unserved(hour: u16, refusals: &[String], unreachable: &[String]) -> Failure {
    if unreachable.is_empty() {
        return Failure::Refused(format!(
            "f{hour:03}: no source served this object -- {}",
            refusals.join("; ")
        ));
    }
    let reasons: Vec<&str> = unreachable
        .iter()
        .chain(refusals.iter())
        .map(String::as_str)
        .collect();
    Failure::Transfer(format!(
        "f{hour:03}: no source could be reached for this object -- {}",
        reasons.join("; ")
    ))
}

fn probe_record(facts: &plan::ProbeFacts) -> ProbeRecord {
    ProbeRecord {
        object_bytes: facts.object_bytes,
        idx_declared: facts.idx_declared,
        idx_fetched: facts.idx_fetched,
        idx_error: facts.idx_error.clone(),
        idx_record_count: facts.idx_rows,
        idx_last_offset: facts.idx_last_offset,
        idx_last_message_bytes: facts.idx_last_message_bytes,
        idx_covers_object: facts.idx_covers_object,
    }
}

// ──────────────────────────────────────────────────────────
// fetch
// ──────────────────────────────────────────────────────────

fn command_fetch(options: &Options) -> Result<String, Failure> {
    let model = options.model()?;
    let cycle = options.cycle_spec()?;
    let product = product_for(model, options)?;
    let only = options.source()?;
    let out = options
        .out
        .clone()
        .ok_or_else(|| "--out DIR is required".to_string())?;
    if options.hours.is_empty() {
        return Err("--hours is required".to_string().into());
    }
    std::fs::create_dir_all(&out)
        .map_err(|error| format!("could not create {}: {error}", out.display()))?;

    let fetcher = Fetcher::new(options.cache_dir.as_deref())?;
    let reporter = ProgressReporter::start(fetcher.progress(), Box::new(|line| eprint!("{line}")));
    let started = Instant::now();
    let mut files: Vec<FileRecord> = Vec::with_capacity(options.hours.len());

    for hour in &options.hours {
        let hour = *hour;
        let hour_started = Instant::now();
        let mut refusals: Vec<String> = Vec::new();
        let mut unreachable: Vec<String> = Vec::new();
        let mut landed = false;

        for candidate in candidates(model, &cycle, hour, &product, only)? {
            let consult_idx =
                options.mode != ModeRequest::FullFile && options.patterns() > 0;
            let (facts, payload) = fetcher.probe_object(
                &candidate.grib_url,
                candidate.idx_url.as_deref(),
                consult_idx,
            );
            let (mode, reason) = match decide(options.mode, &facts, options.patterns()) {
                Decision::Take(mode, reason) => (mode, reason),
                Decision::Refuse(reason) => {
                    let said = format!("{}: {reason}", candidate.source);
                    if facts.object_unreachable.is_some() {
                        unreachable.push(said);
                    } else {
                        refusals.push(said);
                    }
                    continue;
                }
            };

            let name = object_name(&candidate.grib_url);
            let destination = out.join(&name);
            // The probes above moved a few bytes of their own; the count
            // a reader sees is this object's payload.
            reporter.object(hour);
            let (bytes, ranges, selected, idx_sidecar) = match mode {
                Mode::FullFile => (
                    fetcher
                        .get_full_file(&candidate.grib_url)
                        .map_err(Failure::from)?,
                    Vec::new(),
                    None,
                    None,
                ),
                Mode::IdxSubset => {
                    let payload = payload.as_ref().ok_or_else(|| {
                        "internal error: an idx-subset transfer without an index".to_string()
                    })?;
                    let selection = if options.exact_patterns.is_empty() {
                        select_contains(&payload.text, &options.contains_patterns)?
                    } else {
                        let mut chosen = select_exact(
                            &payload.rows,
                            &options.exact_patterns,
                            &options.exclusions,
                        )?;
                        if !options.contains_patterns.is_empty() {
                            chosen.extend(select_contains(
                                &payload.text,
                                &options.contains_patterns,
                            )?);
                            chosen.sort_unstable();
                            chosen.dedup();
                        }
                        chosen
                    };
                    let (bytes, ranges) = fetcher
                        .get_idx_subset(&candidate.grib_url, payload, &selection)
                        .map_err(Failure::from)?;
                    let idx_name = format!("{name}.idx");
                    write_idx(&out, &idx_name, &payload.text)?;
                    (bytes, ranges, Some(selection.len()), Some(idx_name))
                }
            };

            if !grib_framed(&bytes) {
                return Err(format!(
                    "{name}: the assembled payload is not a complete GRIB2 stream \
                     ({} bytes, mode {})",
                    bytes.len(),
                    mode.as_str()
                )
                .into());
            }
            publish(&destination, &bytes)?;

            if options.keep_idx && idx_sidecar.is_none() {
                if let Some(payload) = payload.as_ref() {
                    write_idx(&out, &format!("{name}.idx"), &payload.text)?;
                }
            }

            files.push(FileRecord {
                forecast_hour: hour,
                name: name.clone(),
                path: destination.display().to_string(),
                bytes: bytes.len() as u64,
                sha256: hex_digest(&bytes),
                source: candidate.source.to_string(),
                grib_url: candidate.grib_url.clone(),
                idx_url: candidate.idx_url.clone(),
                mode: mode.as_str().to_string(),
                mode_reason: reason,
                probe: probe_record(&facts),
                idx_name: idx_sidecar,
                idx_sha256: payload.as_ref().map(|p| p.sha256.clone()),
                idx_bytes: payload.as_ref().map(|p| p.bytes),
                idx_record_count: payload.as_ref().map(|p| p.rows.len()),
                selected_record_count: selected,
                ranges,
                // wx-core's client exposes no response headers, and
                // reaching around it to read one would put unpaced
                // traffic on NOMADS.  The payload and index digests are
                // a stronger identity than an ETag anyway, and Python
                // authors its receipt from those.
                etag: None,
                last_modified: None,
                wall_seconds: hour_started.elapsed().as_secs_f64(),
            });
            landed = true;
            break;
        }

        if !landed {
            return Err(unserved(hour, &refusals, &unreachable));
        }
    }

    reporter.finish();
    let payload_bytes = files.iter().map(|file| file.bytes).sum();
    let document = FetchRecord {
        schema: FETCH_RECORD_SCHEMA,
        tool: "rw_fetch",
        tool_version: VERSION,
        model: model.to_string(),
        product,
        cycle: CycleRecord {
            date: cycle.date_yyyymmdd.clone(),
            hour: cycle.hour_utc,
        },
        requested_mode: options.mode.as_str().to_string(),
        requested_source: options.source.clone(),
        var_pattern_count: options.patterns(),
        out_dir: out.display().to_string(),
        cache_dir: options
            .cache_dir
            .as_ref()
            .map(|dir| dir.display().to_string()),
        files,
        payload_bytes,
        // Read after every transfer, so it covers the whole run: one
        // object reaches the cache under two key shapes and the second
        // costs a reference, and a receipt that could not say so left a
        // multiple of the payload on disk unaccounted for.
        dedup: fetcher.cache_dedup().map(|dedup| DedupRecord {
            cache_bytes_written: dedup.bytes_written,
            cache_bytes_deduplicated: dedup.bytes_deduplicated,
            reference_entries: dedup.reference_entries,
        }),
        wall_seconds: started.elapsed().as_secs_f64(),
    };
    Ok(serde_json::to_string_pretty(&document).map_err(|error| format!("{error}"))?)
}

/// The `rw_fetch-progress` lines `fetch` prints while an object moves.
///
/// A whole object is assembled in memory and published in one rename,
/// so nothing a caller can watch on disk grows while it moves, and a 750
/// MB file used to read as 0 B until the moment it landed.  These lines
/// carry the body bytes wx-core has counted, from every stream, about
/// once a second and only when the count has changed; each starts on a
/// fresh line so wx-core's carriage-return chunk counter cannot swallow
/// it.  They never contain `rw_fetch: `, which is how a failure's reason
/// is found.
struct ProgressReporter {
    progress: Arc<TransferProgress>,
    hour: Arc<AtomicU32>,
    stop: Option<Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

/// Seconds between progress lines.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(1);

/// `hour` before the first object is named.
const NO_OBJECT: u32 = u32::MAX;

fn progress_line(hour: u32, received: u64, total: Option<u64>) -> String {
    let total = total.map_or_else(|| "-".to_string(), |total| total.to_string());
    format!("\nrw_fetch-progress f{hour:03} {received} {total}\n")
}

impl ProgressReporter {
    /// `say` receives each line; `fetch` hands it stderr.
    fn start(progress: Arc<TransferProgress>, mut say: Box<dyn FnMut(&str) + Send>) -> Self {
        let hour = Arc::new(AtomicU32::new(NO_OBJECT));
        let (stop, stopped) = mpsc::channel::<()>();
        let (watched, current) = (progress.clone(), hour.clone());
        let thread = std::thread::spawn(move || {
            let mut said: Option<(u32, u64)> = None;
            loop {
                let last = match stopped.recv_timeout(PROGRESS_INTERVAL) {
                    Err(RecvTimeoutError::Timeout) => false,
                    _ => true,
                };
                let hour = current.load(Ordering::Relaxed);
                let received = watched.received();
                if hour != NO_OBJECT && said != Some((hour, received)) {
                    say(&progress_line(hour, received, watched.total()));
                    said = Some((hour, received));
                }
                if last {
                    return;
                }
            }
        });
        Self {
            progress,
            hour,
            stop: Some(stop),
            thread: Some(thread),
        }
    }

    /// The next object's payload starts now.
    fn object(&self, hour: u16) {
        self.progress.reset();
        self.hour.store(u32::from(hour), Ordering::Relaxed);
    }

    /// Say the last count and stop.
    fn finish(mut self) {
        self.stop_thread();
    }

    fn stop_thread(&mut self) {
        drop(self.stop.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for ProgressReporter {
    fn drop(&mut self) {
        self.stop_thread();
    }
}

fn write_idx(out: &Path, name: &str, text: &str) -> Result<(), String> {
    let path = out.join(name);
    if path.exists() {
        // Byte-identical republication is fine; anything else is not.
        let existing = std::fs::read_to_string(&path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        if existing == text {
            return Ok(());
        }
        return Err(format!(
            "refusing to replace {} with a different index",
            path.display()
        ));
    }
    publish(&path, text.as_bytes())
}

// ──────────────────────────────────────────────────────────
// probe
// ──────────────────────────────────────────────────────────

fn command_probe(options: &Options) -> Result<String, String> {
    let model = options.model()?;
    let cycle = options.cycle_spec()?;
    let product = product_for(model, options)?;
    let only = options.source()?;
    if options.hours.is_empty() {
        return Err("--hours is required".to_string());
    }
    let fetcher = Fetcher::new(options.cache_dir.as_deref())?;
    let mut hours = Vec::with_capacity(options.hours.len());

    for hour in &options.hours {
        let hour = *hour;
        let mut reported: Option<ProbeHour> = None;
        let mut refusals: Vec<String> = Vec::new();
        for candidate in candidates(model, &cycle, hour, &product, only)? {
            let consult_idx = options.mode != ModeRequest::FullFile;
            let (facts, _) = fetcher.probe_object(
                &candidate.grib_url,
                candidate.idx_url.as_deref(),
                consult_idx,
            );
            // `probe` reports what a fetch *would* do, so a bare probe
            // with no selectors still answers the index question rather
            // than short-circuiting to "nothing to subset".
            let patterns = options.patterns().max(1);
            match decide(options.mode, &facts, patterns) {
                Decision::Take(mode, reason) => {
                    reported = Some(ProbeHour {
                        forecast_hour: hour,
                        source: Some(candidate.source.to_string()),
                        grib_url: Some(candidate.grib_url.clone()),
                        idx_url: candidate.idx_url.clone(),
                        probe: Some(probe_record(&facts)),
                        mode: Some(mode.as_str().to_string()),
                        mode_reason: reason,
                    });
                    break;
                }
                Decision::Refuse(reason) => refusals.push(format!("{}: {reason}", candidate.source)),
            }
        }
        hours.push(reported.unwrap_or(ProbeHour {
            forecast_hour: hour,
            source: None,
            grib_url: None,
            idx_url: None,
            probe: None,
            mode: None,
            mode_reason: refusals.join("; "),
        }));
    }

    let document = ProbeReport {
        schema: PROBE_REPORT_SCHEMA,
        tool: "rw_fetch",
        model: model.to_string(),
        product,
        cycle: CycleRecord {
            date: cycle.date_yyyymmdd.clone(),
            hour: cycle.hour_utc,
        },
        requested_mode: options.mode.as_str().to_string(),
        var_pattern_count: options.patterns(),
        hours,
    };
    serde_json::to_string_pretty(&document).map_err(|error| format!("{error}"))
}

// ──────────────────────────────────────────────────────────
// latest
// ──────────────────────────────────────────────────────────

fn command_latest(options: &Options) -> Result<String, String> {
    let model = options.model()?;
    let product = product_for(model, options)?;
    let only = options.source()?;
    let through = options
        .through
        .ok_or_else(|| "--through N is required for `latest`".to_string())?;
    let fetcher = Fetcher::new(options.cache_dir.as_deref())?;

    // Walk backwards from the caller's --date/--cycle when given, else
    // from now, over this model's own legal cycle hours.
    let cycles = candidate_cycles(model, options)?;
    let mut probed: Vec<String> = Vec::new();
    for spec in &cycles {
        let label = format!("{}T{:02}Z", spec.date_yyyymmdd, spec.hour_utc);
        probed.push(label.clone());
        let mut source: Option<SourceId> = None;
        let mut complete = true;
        for hour in [0u16, through] {
            let mut hour_ok = false;
            for candidate in candidates(model, spec, hour, &product, only)? {
                if fetcher
                    .probe_object(&candidate.grib_url, candidate.idx_url.as_deref(), false)
                    .0
                    .object_present
                {
                    source.get_or_insert(candidate.source);
                    hour_ok = true;
                    break;
                }
            }
            if !hour_ok {
                complete = false;
                break;
            }
        }
        if complete {
            let document = LatestReport {
                schema: LATEST_REPORT_SCHEMA,
                tool: "rw_fetch",
                model: model.to_string(),
                product: product.clone(),
                through_forecast_hour: through,
                cycle: Some(CycleRecord {
                    date: spec.date_yyyymmdd.clone(),
                    hour: spec.hour_utc,
                }),
                source: source.map(|id| id.to_string()),
                probed,
            };
            return serde_json::to_string_pretty(&document).map_err(|error| format!("{error}"));
        }
    }
    Err(format!(
        "no {model:?} cycle serving f000 through f{through:03} was found among {} probed \
         cycles ({}); pass an explicit --date/--cycle",
        probed.len(),
        probed.join(", ")
    ))
}

/// Cycle candidates, newest first, over a 48-hour lookback.
///
/// Anchored on `--date`/`--cycle` when both are given, otherwise on
/// now, then snapped down onto the model's own cycle grid.  All the
/// arithmetic happens in whole epoch hours so no local timezone or
/// calendar edge can move a cycle.
fn candidate_cycles(model: ModelId, options: &Options) -> Result<Vec<CycleSpec>, String> {
    let step = i64::from(cycle_step_hours(model));
    let anchor_epoch_hours = match (options.date.as_deref(), options.cycle) {
        (Some(date), Some(hour)) => {
            let day = chrono::NaiveDate::parse_from_str(date, "%Y%m%d")
                .map_err(|error| format!("--date {date:?}: {error}"))?;
            let moment = day
                .and_hms_opt(u32::from(hour), 0, 0)
                .ok_or_else(|| format!("--cycle {hour} is not an hour"))?;
            moment.and_utc().timestamp() / 3600
        }
        _ => chrono::Utc::now().timestamp() / 3600,
    };
    let snapped = anchor_epoch_hours - anchor_epoch_hours.rem_euclid(step);

    let span = 48 / step + 1;
    let mut cycles = Vec::with_capacity(span as usize);
    for back in 0..span {
        let moment = chrono::DateTime::from_timestamp((snapped - back * step) * 3600, 0)
            .ok_or_else(|| "cycle arithmetic overflowed".to_string())?;
        cycles.push(
            CycleSpec::new(
                moment.format("%Y%m%d").to_string(),
                moment
                    .format("%H")
                    .to_string()
                    .parse::<u8>()
                    .map_err(|error| format!("cycle hour is not a number: {error}"))?,
            )
            .map_err(|error| format!("{error}"))?,
        );
    }
    Ok(cycles)
}

fn cycle_step_hours(model: ModelId) -> u8 {
    match model {
        ModelId::Gfs | ModelId::Gefs | ModelId::Aigfs | ModelId::Aigefs | ModelId::Hgefs => 6,
        ModelId::Gdas => 6,
        ModelId::EcmwfOpenData | ModelId::Aifs => 6,
        ModelId::Nam => 6,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::RangeRecord;
    use std::collections::BTreeMap;

    #[test]
    fn hours_parse_ranges_steps_and_lists() {
        assert_eq!(parse_hours("0-4").unwrap(), vec![0, 1, 2, 3, 4]);
        assert_eq!(parse_hours("0-12:3").unwrap(), vec![0, 3, 6, 9, 12]);
        assert_eq!(parse_hours("0,6,3").unwrap(), vec![0, 3, 6]);
        assert_eq!(parse_hours("0-2,6").unwrap(), vec![0, 1, 2, 6]);
    }

    #[test]
    fn hours_refuse_nonsense() {
        assert!(parse_hours("4-0").unwrap_err().contains("backwards"));
        assert!(parse_hours("x").unwrap_err().contains("forecast hour"));
        assert!(parse_hours("0-4:0").unwrap_err().contains("positive"));
        assert!(parse_hours(",").unwrap_err().contains("no forecast hours"));
    }

    #[test]
    fn hrrr_product_tokens_map_arwens_spelling_onto_the_registrys() {
        // The registry matches "nat"; ArWen says "wrfnat"; the URL must
        // end up naming wrfnat either way.
        for spelling in ["nat", "native", "wrfnat"] {
            assert_eq!(normalize_product(ModelId::Hrrr, spelling), "nat");
            assert_eq!(
                product_url_fragment(ModelId::Hrrr, spelling),
                Some("wrfnat")
            );
        }
        assert_eq!(normalize_product(ModelId::Hrrr, "wrfprs"), "prs");
        // A model with a fail-closed builder of its own is left alone.
        assert_eq!(
            normalize_product(ModelId::Gdas, "pgrb2.0p25"),
            "pgrb2.0p25"
        );
        assert_eq!(product_url_fragment(ModelId::Gdas, "pgrb2.0p25"), None);
    }

    #[test]
    fn an_unrecognised_hrrr_product_is_refused_not_downgraded() {
        let error = check_product_token(ModelId::Hrrr, "wrfnative").unwrap_err();
        assert!(error.contains("silently fall back"), "{error}");
        assert!(error.contains("wrfnat"), "{error}");
        assert!(check_product_token(ModelId::Gfs, "anything").is_ok());
    }

    #[test]
    fn object_names_come_from_the_url_tail() {
        assert_eq!(
            object_name("https://example/hrrr.20260728/conus/hrrr.t12z.wrfnatf00.grib2"),
            "hrrr.t12z.wrfnatf00.grib2"
        );
        assert_eq!(object_name("https://example/a/b.grib2?x=1"), "b.grib2");
    }

    #[test]
    fn the_abi_marker_names_every_key_python_consumes() {
        let keys: BTreeMap<&str, ()> = FETCH_RECORD_ABI
            .split('\t')
            .skip(1)
            .map(|key| (key, ()))
            .collect();
        for required in [
            "mode",
            "mode_reason",
            "source",
            "grib_url",
            "idx_url",
            "idx_sha256",
            "idx_record_count",
            "selected_record_count",
            "ranges",
            "sha256",
        ] {
            assert!(keys.contains_key(required), "ABI marker lost {required}");
        }
        assert!(FETCH_RECORD_ABI.starts_with(FETCH_RECORD_SCHEMA));
    }

    #[test]
    fn mode_request_round_trips_its_spelling() {
        for spelling in ["auto", "full-file", "idx-subset"] {
            assert_eq!(ModeRequest::parse(spelling).unwrap().as_str(), spelling);
        }
        assert!(ModeRequest::parse("fastest").is_err());
    }

    #[test]
    fn a_failure_reason_starts_on_its_own_line_after_carriage_return_progress() {
        // What wx-core leaves on stderr mid-transfer, then the reason.
        let transfer = Failure::Transfer("HTTP error: failed to read https://x: timeout".into());
        let stderr = format!("\r  Downloading chunks 26/27...{}", failure_text(&transfer));
        let reasons: Vec<&str> = stderr
            .split(|c| c == '\r' || c == '\n')
            .filter(|line| line.starts_with("rw_fetch: "))
            .collect();
        assert_eq!(reasons, ["rw_fetch: HTTP error: failed to read https://x: timeout"]);
        assert!(!stderr.contains("common options"), "no usage text after a runtime failure");
        assert_eq!(transfer.exit_status(), EXIT_TRANSFER);
    }

    /// A probe that could not reach the origin is the network's fault:
    /// the fetch exits 3 so the caller asks again.  It used to say "no
    /// source served this object" and exit 1, and a whole HRRR fetch
    /// ended on one dropped link without another attempt.
    #[test]
    fn an_origin_that_cannot_be_reached_is_a_transfer_failure_not_an_absent_object() {
        let closed = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
            listener.local_addr().expect("address")
        };
        let (facts, _payload) = Fetcher::for_test().probe_object(
            &format!("http://{closed}/hrrr.t06z.wrfprsf18.grib2"),
            None,
            false,
        );
        assert!(!facts.object_present);
        assert!(facts.object_unreachable.is_some(), "{facts:?}");
        let Decision::Refuse(reason) = decide(ModeRequest::FullFile, &facts, 0) else {
            panic!("an unreachable object cannot be taken");
        };
        assert!(reason.contains("could not be reached"), "{reason}");
        let failure = unserved(18, &[], &[format!("aws: {reason}")]);
        assert_eq!(failure.exit_status(), EXIT_TRANSFER);
        assert!(failure.message().starts_with("f018: no source could be reached"));

        // Beside a source that answered "not here", still a transfer.
        let mixed = unserved(
            18,
            &["nomads: the GRIB object is not present at this source".to_string()],
            &[format!("aws: {reason}")],
        );
        assert_eq!(mixed.exit_status(), EXIT_TRANSFER);
        assert!(mixed.message().contains("not present"), "{}", mixed.message());
    }

    /// An origin that answers 404 has said the object is not there: that
    /// stays a refusal (exit 1), and another attempt would only repeat it.
    #[test]
    fn an_origin_that_answers_404_is_still_an_absent_object() {
        let missing =
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec();
        let origin = local_origin(vec![missing]);
        let (facts, _payload) = Fetcher::for_test().probe_object(
            &format!("{origin}/hrrr.t06z.wrfprsf18.grib2"),
            None,
            false,
        );
        assert!(!facts.object_present);
        assert_eq!(facts.object_unreachable, None);
        let failure = unserved(
            18,
            &["aws: the GRIB object is not present at this source".to_string()],
            &[],
        );
        assert_eq!(failure.exit_status(), EXIT_REFUSED);
        assert!(failure.message().starts_with("f018: no source served this object"));
    }

    /// One reply per connection, in order, from a local origin.
    fn local_origin(replies: Vec<Vec<u8>>) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("address");
        std::thread::spawn(move || {
            for reply in replies {
                let (mut stream, _) = listener.accept().expect("accept");
                let mut request = [0u8; 4096];
                let _ = stream.read(&mut request);
                let _ = stream.write_all(&reply);
                let _ = stream.flush();
            }
        });
        format!("http://{address}")
    }

    /// An origin that answers a chunk's range request with the whole
    /// object (200, not 206) ignored the Range header on purpose.  That
    /// is a refusal (exit 1), not a transfer the network cut off
    /// (exit 3): exit 3 makes gpuwm download the whole object again up
    /// to its retry budget and tell the reader to wait for a steadier
    /// connection, which would never help.
    #[test]
    fn an_origin_that_answers_a_range_request_with_200_exits_1_not_3() {
        // 20 MB: past one 16 MiB chunk, so the object is fetched in ranges.
        let probe = b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-0/20000000\r\n\
                      Content-Length: 1\r\nConnection: close\r\n\r\nG"
            .to_vec();
        let whole = b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\nGRIB"
            .to_vec();
        let origin = local_origin(vec![probe, whole.clone(), whole]);
        let fault = Fetcher::for_test()
            .get_full_file(&format!("{origin}/hrrr.t06z.wrfnatf00.grib2"))
            .expect_err("a 200 to a range request is refused");
        assert!(matches!(fault, net::TransferFault::Refused(_)), "{fault:?}");
        let failure = Failure::from(fault);
        assert_eq!(failure.exit_status(), EXIT_REFUSED);
        assert!(failure.message().contains("not 206"), "{}", failure.message());
    }

    #[test]
    fn a_transfer_the_network_ended_exits_3_and_a_refused_one_exits_1() {
        let network = Failure::from(net::TransferFault::Network("timeout".into()));
        assert_eq!(network.exit_status(), EXIT_TRANSFER);
        let refused = Failure::from(net::TransferFault::Refused("HTTP status 403".into()));
        assert_eq!(refused.exit_status(), EXIT_REFUSED);
    }

    /// A progress line stands on its own line, carries the object's
    /// hour, the bytes so far and the size, and can never be mistaken
    /// for a failure's reason.
    #[test]
    fn a_progress_line_is_its_own_line_and_never_a_failure_reason() {
        let line = progress_line(7, 1_048_576, Some(750_000_000));
        assert_eq!(line, "\nrw_fetch-progress f007 1048576 750000000\n");
        assert!(!line.contains("rw_fetch: "));
        assert_eq!(progress_line(12, 0, None), "\nrw_fetch-progress f012 0 -\n");
    }

    /// The reporter says nothing before an object is named, then the
    /// count the client kept, and one last line when it stops.
    #[test]
    fn the_reporter_relays_the_clients_byte_count() {
        let said = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let sink = said.clone();
        let progress = TransferProgress::new();
        let reporter = ProgressReporter::start(
            progress.clone(),
            Box::new(move |line| sink.lock().unwrap().push(line.to_string())),
        );
        progress.add(99); // a probe's bytes, before the payload
        reporter.object(3);
        progress.set_total(1000);
        progress.add(400);
        reporter.finish();
        let said = said.lock().unwrap().clone();
        assert_eq!(said.last().map(String::as_str), Some("\nrw_fetch-progress f003 400 1000\n"));
        assert!(said.iter().all(|line| line.contains(" f003 ")), "{said:?}");
    }

    /// A whole object fetched through the Fetcher is counted byte for
    /// byte, and its size is known before its chunks arrive.
    #[test]
    fn a_full_file_fetch_counts_every_payload_byte() {
        let object = 17 * 1024 * 1024u64;
        let body = move |first: u64, last: u64| -> Vec<u8> {
            let mut reply = format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {first}-{last}/{object}\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                last - first + 1
            )
            .into_bytes();
            reply.extend((first..=last).map(|index| (index % 251) as u8));
            reply
        };
        let chunk = 16 * 1024 * 1024u64;
        // The probe, then the two chunks in the order two streams ask.
        let origin = {
            use std::io::{Read, Write};
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
            let address = listener.local_addr().expect("address");
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    let Ok(mut stream) = stream else { continue };
                    let mut request = Vec::new();
                    let mut byte = [0u8; 1];
                    while !request.ends_with(b"\r\n\r\n") {
                        if stream.read(&mut byte).unwrap_or(0) == 0 {
                            break;
                        }
                        request.push(byte[0]);
                    }
                    let text = String::from_utf8_lossy(&request).to_string();
                    let reply = if text.contains("bytes=0-0") {
                        body(0, 0)
                    } else if text.contains(&format!("bytes={chunk}-")) {
                        body(chunk, object - 1)
                    } else {
                        body(0, chunk - 1)
                    };
                    std::thread::spawn(move || {
                        let _ = stream.write_all(&reply);
                    });
                }
            });
            format!("http://{address}")
        };
        let fetcher = Fetcher::for_test();
        let bytes = fetcher
            .get_full_file(&format!("{origin}/hrrr.t06z.wrfnatf00.grib2"))
            .expect("whole object");
        assert_eq!(bytes.len() as u64, object);
        assert_eq!(fetcher.progress().total(), Some(object));
        assert_eq!(fetcher.progress().received(), object);
        assert_eq!(fetcher.streams(), 2);
    }

    #[test]
    fn only_a_usage_error_exits_2_and_it_points_at_help_instead_of_dumping_usage() {
        let usage = run(&["fetch".to_string(), "--bogus".to_string()]).unwrap_err();
        assert_eq!(usage.exit_status(), EXIT_USAGE);
        let text = failure_text(&usage);
        assert!(text.contains("unknown option \"--bogus\""));
        assert!(text.contains("--help"));
        assert!(!text.contains("common options"));
        let refused = run(&["fetch".to_string()]).unwrap_err();
        assert_eq!(refused, Failure::Refused("--model is required".to_string()));
        assert_eq!(refused.exit_status(), EXIT_REFUSED);
    }

    #[test]
    fn options_reject_an_unknown_flag_rather_than_ignoring_it() {
        let args = ["--model".to_string(), "hrrr".to_string(), "--fast".to_string()];
        assert!(Options::parse(&args).unwrap_err().contains("--fast"));
    }

    #[test]
    fn a_range_record_is_inclusive_on_both_ends() {
        let record = RangeRecord {
            start: 10,
            end: 19,
            bytes: 10,
            first_index_row: 1,
            last_index_row: 2,
        };
        assert_eq!(record.end - record.start + 1, record.bytes);
    }

    /// The resolved wx-core must be one that actually paces NOMADS.
    ///
    /// Two different wx-core crates were vendored under the same `0.3.9`
    /// version string: the working-tree copy with the cross-process governor,
    /// and the published crates.io copy with no governor at all. Only a path
    /// dependency kept the right one in the graph. That is a trap that
    /// resolves silently, and the symptom -- an ArWen fetch hammering a shared
    /// public service until the user's IP is blocked -- would surface a long
    /// way from its cause.
    ///
    /// This gate does not check which copy was resolved. It asks the resolved
    /// crate what it is configured to do, then makes it DO it: two paced calls
    /// for a NOMADS URL, with the state file and the gap pointed somewhere
    /// harmless, must be spaced by the gap and must leave the shared state
    /// behind. A governorless wx-core fails to link, and a wx-core whose
    /// governor stopped working fails here. No network request is made.
    #[test]
    fn the_resolved_wx_core_actually_paces_nomads() {
        use std::time::{Duration, Instant};

        let scratch = std::env::temp_dir().join(format!(
            "rw_fetch_governor_probe_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|value| value.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&scratch).expect("scratch");
        let state = scratch.join("nomads.state");
        // SAFETY: single-threaded test process, and both variables are read
        // by wx-core on every call rather than cached at startup.
        unsafe {
            std::env::set_var("RUSTWX_NOMADS_RATE_STATE", &state);
            std::env::set_var("RUSTWX_NOMADS_MIN_INTERVAL_MS", "400");
        }

        let capability = wx_core::download::nomads_governor();
        assert_eq!(
            capability.state_path, state,
            "the governor is not reading the state file it says it reads"
        );
        assert_eq!(capability.min_request_gap, Duration::from_millis(400));
        assert!(capability.cooldown >= Duration::from_secs(60));

        let url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=x";
        wx_core::download::pace_nomads_request(url);
        let started = Instant::now();
        wx_core::download::pace_nomads_request(url);
        let spacing = started.elapsed();
        assert!(
            spacing >= Duration::from_millis(350),
            "a second NOMADS request was paced by only {spacing:?}; the              resolved wx-core is not governing this host"
        );
        assert!(
            std::fs::read_to_string(&state)
                .unwrap_or_default()
                .contains("last_request_ms="),
            "the governor recorded no node-wide state"
        );

        // A non-NOMADS host is never paced, so the governor cannot be
        // "working" merely by sleeping on everything.
        let started = Instant::now();
        wx_core::download::pace_nomads_request(
            "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260730/12/atmos/x",
        );
        assert!(started.elapsed() < Duration::from_millis(200));

        unsafe {
            std::env::remove_var("RUSTWX_NOMADS_RATE_STATE");
            std::env::remove_var("RUSTWX_NOMADS_MIN_INTERVAL_MS");
        }
        let _ = std::fs::remove_dir_all(&scratch);
    }
}
