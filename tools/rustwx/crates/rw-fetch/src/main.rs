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
use std::time::Instant;

use rustwx_core::{CycleSpec, ModelId, ModelRunRequest, ResolvedUrl, SourceId};

use net::{grib_framed, hex_digest, publish, select_contains, select_exact, Fetcher};
use plan::{decide, Decision, Mode, ModeRequest};
use record::{
    CycleRecord, FetchRecord, FileRecord, LatestReport, ProbeHour, ProbeRecord, ProbeReport,
    FETCH_RECORD_ABI, FETCH_RECORD_SCHEMA, LATEST_REPORT_SCHEMA, PROBE_REPORT_SCHEMA,
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
";

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(document) => {
            println!("{document}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("rw_fetch: {error}");
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn run(args: &[String]) -> Result<String, String> {
    if args.is_empty() {
        return Err("no subcommand given".to_string());
    }
    match args[0].as_str() {
        "--version" | "-V" => Ok(format!("rw_fetch {VERSION}")),
        "--help" | "-h" => Ok(USAGE.to_string()),
        // The exact-ABI marker `gpuwm.native_wrf_distribution` greps the
        // binary for.  Printing it here is what guarantees the literal
        // is in the built image.
        "--abi" => Ok(FETCH_RECORD_ABI.to_string()),
        "fetch" => command_fetch(&Options::parse(&args[1..])?),
        "probe" => command_probe(&Options::parse(&args[1..])?),
        "latest" => command_latest(&Options::parse(&args[1..])?),
        other => Err(format!("unknown subcommand {other:?}")),
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

fn command_fetch(options: &Options) -> Result<String, String> {
    let model = options.model()?;
    let cycle = options.cycle_spec()?;
    let product = product_for(model, options)?;
    let only = options.source()?;
    let out = options
        .out
        .clone()
        .ok_or_else(|| "--out DIR is required".to_string())?;
    if options.hours.is_empty() {
        return Err("--hours is required".to_string());
    }
    std::fs::create_dir_all(&out)
        .map_err(|error| format!("could not create {}: {error}", out.display()))?;

    let fetcher = Fetcher::new(options.cache_dir.as_deref())?;
    let started = Instant::now();
    let mut files: Vec<FileRecord> = Vec::with_capacity(options.hours.len());

    for hour in &options.hours {
        let hour = *hour;
        let hour_started = Instant::now();
        let mut refusals: Vec<String> = Vec::new();
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
                    refusals.push(format!("{}: {reason}", candidate.source));
                    continue;
                }
            };

            let name = object_name(&candidate.grib_url);
            let destination = out.join(&name);
            let (bytes, ranges, selected, idx_sidecar) = match mode {
                Mode::FullFile => (fetcher.get_full_file(&candidate.grib_url)?, Vec::new(), None, None),
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
                    let (bytes, ranges) =
                        fetcher.get_idx_subset(&candidate.grib_url, payload, &selection)?;
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
                ));
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
            return Err(format!(
                "f{hour:03}: no source served this object -- {}",
                refusals.join("; ")
            ));
        }
    }

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
        wall_seconds: started.elapsed().as_secs_f64(),
    };
    serde_json::to_string_pretty(&document).map_err(|error| format!("{error}"))
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
