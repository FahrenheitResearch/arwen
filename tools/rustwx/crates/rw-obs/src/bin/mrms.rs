//! `rw_mrms` — ArWen's MRMS front door.
//!
//! Lists, fetches and decodes archived MRMS gridded products from the
//! anonymous `noaa-mrms-pds` bucket, and writes the observed field as a
//! `gpuwm-obs.obs-grid.v1` pack the scorer reads with `numpy.frombuffer`.
//!
//! Two facts about the archive shape this whole bin:
//!
//! * **Filenames carry off-cadence seconds.** The nominal cadence is two
//!   minutes, but the stamps are `...-210037`, `...-210240`, `...-210438`:
//!   the second is whatever the product finished at. A URL cannot be
//!   constructed for a valid time; the day prefix must be listed and the
//!   nearest stamp chosen. That is why `nearest` exists as its own
//!   subcommand and why `decode` never fetches.
//! * **Two sentinels, and they mean opposite things.** Measured over the
//!   whole 24.5 M-cell CONUS composite, `-999` covers 38.4 % of cells and
//!   `-99` covers 52.8 %, with real data starting at -31.5 dBZ. `-999` is
//!   *no radar coverage* — genuinely unobserved, and it belongs under the
//!   validity mask. `-99` is *no echo* — an observation that the column was
//!   below detection, which is the single most common true observation on
//!   any day and the one that makes a fractions-skill score mean anything.
//!   Masking it would delete half the CONUS grid, and with it every correct
//!   negative the score is built on. So `-99` maps to a floor value and
//!   stays valid, and the counts of both go in the record.
//!
//! ```text
//! rw_mrms list    --start 2021-12-10T21:00:00Z --end 2021-12-10T21:10:00Z
//! rw_mrms nearest --valid-time 2021-12-10T21:00:00Z [--window-seconds 240]
//! rw_mrms fetch   --start ... --end ... --out DIR
//! rw_mrms decode  --file F.grib2.gz --out F.obspack [--bbox W,S,E,N]
//! rw_mrms grid    --file F.grib2.gz --out grid.obspack [--bbox W,S,E,N]
//! rw_mrms verify  --pack F.obspack
//! ```

use std::error::Error;
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

use rw_nexrad::s3::{build_agent, list_s3, object_url, parse_time, ListRequest, S3Object};
use rw_obs::pack::{
    decode_pack, payload_digest, validate_arrays, write_pack, ArrayEntry, PayloadBuilder,
    GEO_SCHEMA, GRID_SCHEMA,
};
use rw_obs::seam::{
    seam_bounds, seam_time, wrap_longitude, Provenance, QUANTITY_COMPOSITE_REFLECTIVITY, UNITS_DBZ,
};
use rw_obs::{err, gunzip_if_wrapped, hex_sha256};
use rustwx_core::{CanonicalField, FieldSelector};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from, embedded so the gpuwm release cut can prove a
/// staged bridge matches the commit being released by reading bytes
/// alone (`tools/build_bridge_bundle.py pin --source-rev`).  `build.rs`
/// injects the value; `main` references the constant so the linker
/// cannot discard it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

/// The AWS Open Data bucket. Probed 2026-08-03: anonymous `ListObjectsV2`
/// and anonymous GET both answered 200 for keys from 2020 through 2026.
const DEFAULT_BUCKET: &str = "noaa-mrms-pds";
/// The bucket's top-level domain directory. `ALASKA`, `CARIB`, `GUAM`,
/// `HAWAII` and `CONUS_5KM` are the others; the battery is CONUS-only by
/// Stage-IV's construction, but the region is data, not a gate.
const DEFAULT_REGION: &str = "CONUS";
/// The QC'd column-max composite: the standard verification target for
/// convection-permitting reflectivity.
const DEFAULT_PRODUCT: &str = "MergedReflectivityQCComposite_00.50";
/// The altitude-above-MSL selector this product family is filed under.
/// Verified against archived bytes from five separate years: every message
/// carried level type 102 at 500 m, discipline 209 / category 10 / number 0.
const PRODUCT_ALTITUDE_M: u16 = 500;

/// Any value at or below this is *no radar coverage*.
const NO_COVERAGE_CEILING_DBZ: f64 = -900.0;
/// Any value at or below this (and above the coverage ceiling) is *no echo*.
const NO_ECHO_CEILING_DBZ: f64 = -90.0;
/// What a no-echo cell is worth as a number. Below every registered FSS
/// threshold and below the -31.5 dBZ floor real data was measured at, while
/// staying inside the seam's [-40, 100] bound.
const DEFAULT_NO_ECHO_DBZ: f64 = -35.0;
/// What a masked cell is filled with. The scorer never reads under a false
/// mask, but its contract requires the number to be finite anyway — a NaN a
/// mask should have covered is the bug that refusal exists to catch.
const MASKED_FILL_DBZ: f64 = -35.0;

const LIST_SCHEMA: &str = "gpuwm-obs.mrms-list.v1";
const NEAREST_SCHEMA: &str = "gpuwm-obs.mrms-nearest.v1";
const FETCH_SCHEMA: &str = "gpuwm-obs.mrms-fetch.v1";
const DECODE_SCHEMA: &str = "gpuwm-obs.mrms-decode.v1";
const GRIDCMD_SCHEMA: &str = "gpuwm-obs.mrms-grid.v1";
const VERIFY_SCHEMA: &str = "gpuwm-obs.mrms-verify.v1";

/// The exact `--abi` line the Python bridge pins, so a bin that drifted out
/// from under a pinned wrapper is caught at probe time rather than at parse
/// time three receipts later.
const ABI_MARKER: &str = "gpuwm-obs.mrms-fetch.v1\tproduct\twindow\tbucket\tfiles\tbytes\t\
sha256\tgpuwm-obs.obs-grid.v1\tcomposite_reflectivity\tdBZ";

const USAGE: &str = "\
usage: rw_mrms <list|nearest|fetch|decode|grid|verify> [OPTIONS]
       rw_mrms --version | --help | --abi

  list     report the MRMS objects a window resolves to, moving no payload
  nearest  report the one object nearest a valid time, refusing a distant match
  fetch    download objects and print a fetch record with a sha256 per file
  decode   turn one archived object into a `gpuwm-obs.obs-grid.v1` pack
  grid     write the pack's latitude/longitude once, as a `gpuwm-obs.obs-geo.v1`
  verify   re-read a pack and re-prove its header, index and payload digest

archive options
  --bucket NAME           default: noaa-mrms-pds (anonymous list+get, probed
                          2026-08-03 for keys from 2020 through 2026)
  --region NAME           bucket top-level directory. Default: CONUS
  --product NAME          default: MergedReflectivityQCComposite_00.50
  --start TIME            window start, 2021-12-10T21:00:00Z or 20211210T210000
  --end TIME              window end (inclusive)
  --valid-time TIME       nearest: the instant to match
  --window-seconds N      nearest: refuse a match further than this from
                          --valid-time. Default 240 (the +/- 4 min the
                          battery registers around each hourly valid time)
  --limit N               keep at most the first N objects
  --cache DIR             object cache root (default: <out>/.cache, else
                          ./.rw-obs-cache)
  --no-cache              always re-download, never read the cache
  --out PATH              fetch: a directory. decode/grid: the pack file

decode options
  --file FILE             one archived object on disk (gzipped or plain)
  --bbox W,S,E,N          keep only cells inside this lon/lat box. Strongly
                          advised: the full CONUS grid is 7000x3500, which is
                          196 MB of float64 values per frame
  --no-echo-dbz VALUE     what a `-99` no-echo cell is worth. Default -35

verify options
  --pack FILE             the pack to re-prove (--file and --out accepted too)
";

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(output) => {
            print!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("rw_mrms: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<String, Box<dyn Error>> {
    let Some(first) = args.first() else {
        return Ok(USAGE.to_string());
    };
    match first.as_str() {
        "--help" | "-h" | "help" => return Ok(USAGE.to_string()),
        "--version" | "-V" => return Ok(format!("rw_mrms {VERSION}\n")),
        "--abi" => return Ok(format!("{ABI_MARKER}\n")),
        _ => {}
    }
    let options = Options::parse(&args[1..])?;
    match first.as_str() {
        "list" => cmd_list(&options),
        "nearest" => cmd_nearest(&options),
        "fetch" => cmd_fetch(&options),
        "decode" => cmd_decode(&options),
        "grid" => cmd_grid(&options),
        "verify" => cmd_verify(&options),
        other => Err(err(format!("unknown subcommand {other:?}\n\n{USAGE}"))),
    }
}

// ---------------------------------------------------------------- options

#[derive(Debug, Default)]
struct Options {
    bucket: Option<String>,
    region: Option<String>,
    product: Option<String>,
    start: Option<String>,
    end: Option<String>,
    valid_time: Option<String>,
    window_seconds: Option<i64>,
    limit: Option<usize>,
    cache: Option<PathBuf>,
    no_cache: bool,
    out: Option<PathBuf>,
    file: Option<PathBuf>,
    pack: Option<PathBuf>,
    bbox: Option<BoundingBox>,
    no_echo_dbz: Option<f64>,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, Box<dyn Error>> {
        let mut options = Options::default();
        let mut index = 0;
        while index < args.len() {
            let flag = args[index].as_str();
            let mut value = || -> Result<String, Box<dyn Error>> {
                index += 1;
                args.get(index)
                    .cloned()
                    .ok_or_else(|| err(format!("{flag} needs a value")))
            };
            match flag {
                "--bucket" => options.bucket = Some(value()?),
                "--region" => options.region = Some(value()?),
                "--product" => options.product = Some(value()?),
                "--start" => options.start = Some(value()?),
                "--end" => options.end = Some(value()?),
                "--valid-time" => options.valid_time = Some(value()?),
                "--window-seconds" => {
                    let raw = value()?;
                    let seconds: i64 = raw
                        .parse()
                        .map_err(|_| err(format!("--window-seconds expects a count, got {raw:?}")))?;
                    if seconds <= 0 {
                        return Err(err("--window-seconds must be positive"));
                    }
                    options.window_seconds = Some(seconds);
                }
                "--limit" => {
                    let raw = value()?;
                    options.limit = Some(
                        raw.parse()
                            .map_err(|_| err(format!("--limit expects a count, got {raw:?}")))?,
                    );
                }
                "--cache" => options.cache = Some(PathBuf::from(value()?)),
                "--no-cache" => options.no_cache = true,
                "--out" => options.out = Some(PathBuf::from(value()?)),
                "--file" => options.file = Some(PathBuf::from(value()?)),
                "--pack" => options.pack = Some(PathBuf::from(value()?)),
                "--bbox" => options.bbox = Some(BoundingBox::parse(&value()?)?),
                "--no-echo-dbz" => {
                    let raw = value()?;
                    let dbz: f64 = raw
                        .parse()
                        .map_err(|_| err(format!("--no-echo-dbz expects a number, got {raw:?}")))?;
                    let (low, high) = seam_bounds(QUANTITY_COMPOSITE_REFLECTIVITY).unwrap();
                    if !dbz.is_finite() || dbz < low || dbz > high {
                        return Err(err(format!(
                            "--no-echo-dbz {raw:?} is outside the seam's [{low}, {high}] dBZ bound"
                        )));
                    }
                    options.no_echo_dbz = Some(dbz);
                }
                other => return Err(err(format!("unknown option {other:?}\n\n{USAGE}"))),
            }
            index += 1;
        }
        Ok(options)
    }

    fn bucket(&self) -> &str {
        self.bucket.as_deref().unwrap_or(DEFAULT_BUCKET)
    }

    fn region(&self) -> Result<&str, Box<dyn Error>> {
        let region = self.region.as_deref().unwrap_or(DEFAULT_REGION);
        check_prefix_token(region, "--region")?;
        Ok(region)
    }

    fn product(&self) -> Result<&str, Box<dyn Error>> {
        let product = self.product.as_deref().unwrap_or(DEFAULT_PRODUCT);
        check_prefix_token(product, "--product")?;
        Ok(product)
    }

    fn window(&self) -> Result<(DateTime<Utc>, DateTime<Utc>), Box<dyn Error>> {
        let start = parse_time(
            self.start
                .as_deref()
                .ok_or_else(|| err("--start is required"))?,
        )?;
        let end = parse_time(self.end.as_deref().ok_or_else(|| err("--end is required"))?)?;
        if end < start {
            return Err(err(format!(
                "--end {} precedes --start {}",
                seam_time(end),
                seam_time(start)
            )));
        }
        Ok((start, end))
    }

    fn cache_dir(&self) -> PathBuf {
        if let Some(cache) = &self.cache {
            return cache.clone();
        }
        match &self.out {
            Some(out) if out.is_dir() => out.join(".cache"),
            _ => PathBuf::from(".rw-obs-cache"),
        }
    }

    fn no_echo_dbz(&self) -> f64 {
        self.no_echo_dbz.unwrap_or(DEFAULT_NO_ECHO_DBZ)
    }
}

/// A prefix token is *data* — nothing branches on a particular product — but
/// it is concatenated into an S3 prefix and, on the cache path, into a local
/// directory name. A token carrying a separator or a parent reference would
/// list somewhere else or write outside the cache, so it fails closed.
fn check_prefix_token(token: &str, flag: &str) -> Result<(), Box<dyn Error>> {
    if token.is_empty() {
        return Err(err(format!("{flag} may not be empty")));
    }
    if !token
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'.' || b == b'-')
    {
        return Err(err(format!(
            "{flag} {token:?} may carry only letters, digits, '_', '.' and '-'; it is \
             concatenated into an archive prefix and a cache path"
        )));
    }
    if token == "." || token == ".." {
        return Err(err(format!("{flag} {token:?} is not a name")));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
struct BoundingBox {
    west: f64,
    south: f64,
    east: f64,
    north: f64,
}

impl BoundingBox {
    fn parse(raw: &str) -> Result<Self, Box<dyn Error>> {
        let parts: Vec<&str> = raw.split(',').map(str::trim).collect();
        if parts.len() != 4 {
            return Err(err(format!("--bbox expects W,S,E,N, got {raw:?}")));
        }
        let mut values = [0.0f64; 4];
        for (slot, text) in values.iter_mut().zip(parts) {
            *slot = text
                .parse()
                .map_err(|_| err(format!("--bbox component {text:?} is not a number")))?;
            if !slot.is_finite() {
                return Err(err("--bbox components must be finite"));
            }
        }
        let box_ = Self {
            west: values[0],
            south: values[1],
            east: values[2],
            north: values[3],
        };
        if box_.south >= box_.north {
            return Err(err(format!(
                "--bbox south {} is not below north {}",
                box_.south, box_.north
            )));
        }
        if box_.west >= box_.east {
            return Err(err(format!(
                "--bbox west {} is not west of east {}; a box that crosses the antimeridian \
                 is not supported, and silently wrapping one would score the wrong hemisphere",
                box_.west, box_.east
            )));
        }
        if box_.south < -90.0 || box_.north > 90.0 {
            return Err(err("--bbox latitudes must lie inside [-90, 90]"));
        }
        if box_.west < -180.0 || box_.east >= 180.0 {
            return Err(err("--bbox longitudes must lie inside [-180, 180)"));
        }
        Ok(box_)
    }
}

// ------------------------------------------------------------ the archive

/// One listed object that parsed as a product frame.
#[derive(Debug, Clone)]
struct Frame {
    object: S3Object,
    valid_time: DateTime<Utc>,
    gzipped: bool,
}

/// Read the valid time out of `MRMS_<product>_<YYYYMMDD>-<HHMMSS>.grib2[.gz]`.
///
/// The stamp is the file's own, seconds and all. Rounding it to the nominal
/// two-minute cadence would make two frames collide and silently drop one.
fn parse_frame_name(name: &str, product: &str) -> Option<(DateTime<Utc>, bool)> {
    let prefix = format!("MRMS_{product}_");
    let rest = name.strip_prefix(&prefix)?;
    let (stamp, tail) = rest.split_once('.')?;
    let gzipped = match tail {
        "grib2" => false,
        "grib2.gz" => true,
        _ => return None,
    };
    let (day, hms) = stamp.split_once('-')?;
    if day.len() != 8 || hms.len() != 6 {
        return None;
    }
    let naive = NaiveDateTime::parse_from_str(&format!("{day}{hms}"), "%Y%m%d%H%M%S").ok()?;
    Some((Utc.from_utc_datetime(&naive), gzipped))
}

fn day_prefix(region: &str, product: &str, day: NaiveDate) -> String {
    format!(
        "{region}/{product}/{:04}{:02}{:02}/",
        day.year(),
        day.month(),
        day.day()
    )
}

/// Every UTC day the window touches, inclusive of both ends.
fn window_days(start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<NaiveDate> {
    let mut days = Vec::new();
    let mut day = start.date_naive();
    let last = end.date_naive();
    while day <= last {
        days.push(day);
        match day.succ_opt() {
            Some(next) => day = next,
            None => break,
        }
    }
    days
}

fn list_frames(
    agent: &ureq::Agent,
    bucket: &str,
    region: &str,
    product: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> Result<Vec<Frame>, Box<dyn Error>> {
    let mut frames = Vec::new();
    for day in window_days(start, end) {
        let prefix = day_prefix(region, product, day);
        // `list_s3` replaced `list_s3_objects` when the live-chunk route
        // needed a listing's roll-up prefixes and the bucket's own clock.
        // This call wants neither, and passed `start_after: None`, so the
        // undelimited request is the exact equivalent of the old one.
        let objects = list_s3(agent, ListRequest::new(bucket, &prefix))?.objects;
        for object in objects {
            let name = object.key.rsplit('/').next().unwrap_or(&object.key);
            let Some((valid_time, gzipped)) = parse_frame_name(name, product) else {
                continue;
            };
            if valid_time >= start && valid_time <= end {
                frames.push(Frame {
                    object,
                    valid_time,
                    gzipped,
                });
            }
        }
    }
    frames.sort_by_key(|frame| (frame.valid_time, frame.object.key.clone()));
    Ok(frames)
}

/// Localize an S3 key under the cache root, one segment at a time, refusing
/// any segment that is not an ordinary name.
///
/// Same rule the radar client applies to its own cache; the namespace
/// differs (`mrms` rather than `nexrad`) so two instruments sharing one
/// `--cache` cannot collide.
fn cached_object_path(cache_dir: &Path, bucket: &str, key: &str) -> Result<PathBuf, Box<dyn Error>> {
    let root = cache_dir.join("mrms").join(bucket);
    let mut path = root.clone();
    for segment in key.split('/') {
        path.push(segment);
    }
    let under_root = path.strip_prefix(&root).is_ok_and(|tail| {
        tail.components()
            .all(|component| matches!(component, Component::Normal(_)))
    });
    if !under_root {
        return Err(err(format!(
            "archive key {key:?} does not name a file under the cache directory for \
             s3://{bucket}/ ({}); a key is localized one segment at a time, so a key whose \
             segments walk back out would put bucket bytes somewhere this client never \
             offered to manage",
            root.display()
        )));
    }
    Ok(path)
}

struct Downloaded {
    path: PathBuf,
    sha256: String,
    bytes: u64,
    cache_hit: bool,
    fetched_at: String,
}

fn download(
    agent: &ureq::Agent,
    bucket: &str,
    cache_dir: &Path,
    frame: &Frame,
    use_cache: bool,
) -> Result<Downloaded, Box<dyn Error>> {
    let target = cached_object_path(cache_dir, bucket, &frame.object.key)?;
    let fetched_at = seam_time(Utc::now());
    if use_cache && target.is_file() {
        let metadata = target.metadata()?;
        if metadata.len() == frame.object.size_bytes {
            let bytes = std::fs::read(&target)?;
            return Ok(Downloaded {
                sha256: hex_sha256(&bytes),
                bytes: bytes.len() as u64,
                path: target,
                cache_hit: true,
                fetched_at,
            });
        }
    }
    let url = object_url(bucket, &frame.object.key);
    let bytes = rw_obs::net::get_bytes(agent, &url, "MRMS object")?;
    if frame.object.size_bytes > 0 && bytes.len() as u64 != frame.object.size_bytes {
        return Err(err(format!(
            "downloaded byte count mismatch for {}: the listing says {}, {} arrived",
            frame.object.key,
            frame.object.size_bytes,
            bytes.len()
        )));
    }
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = target.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&temporary, &bytes)?;
    if target.exists() {
        std::fs::remove_file(&target)?;
    }
    std::fs::rename(&temporary, &target).inspect_err(|_| {
        let _ = std::fs::remove_file(&temporary);
    })?;
    Ok(Downloaded {
        sha256: hex_sha256(&bytes),
        bytes: bytes.len() as u64,
        path: target,
        cache_hit: false,
        fetched_at,
    })
}

// ------------------------------------------------------------- the fields

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GridSpec {
    kind: String,
    nx: usize,
    ny: usize,
    /// Index range kept out of the source grid, so a subset states what it
    /// is a subset of rather than pretending to be a whole grid.
    source_nx: usize,
    source_ny: usize,
    i_start: usize,
    j_start: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SentinelReport {
    no_coverage_ceiling_dbz: f64,
    no_echo_ceiling_dbz: f64,
    no_echo_value_dbz: f64,
    masked_fill_dbz: f64,
    no_coverage_cells: usize,
    no_echo_cells: usize,
    echo_cells: usize,
    masked_cells: usize,
    observed_fraction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GridPackMeta {
    schema: String,
    status: String,
    quantity: String,
    units: String,
    valid_time: String,
    provenance: Provenance,
    grid: GridSpec,
    sentinels: SentinelReport,
    value_min_dbz: f64,
    value_max_dbz: f64,
    arrays: std::collections::BTreeMap<String, ArrayEntry>,
    payload_bytes: usize,
    content_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeoPackMeta {
    schema: String,
    status: String,
    source_product: String,
    grid: GridSpec,
    arrays: std::collections::BTreeMap<String, ArrayEntry>,
    payload_bytes: usize,
    content_sha256: String,
}

/// The decoded field, already subset and already in seam conventions.
struct Field {
    grid: GridSpec,
    latitude: Vec<f64>,
    longitude: Vec<f64>,
    values: Vec<f64>,
    valid: Vec<bool>,
    sentinels: SentinelReport,
    value_min: f64,
    value_max: f64,
}

/// Decode one archived object into a seam-ready field.
///
/// The GRIB2 work is entirely `rustwx-io`'s: its MRMS selector matches
/// discipline 209 / category 10 / number 0 at level type 102, 500 m, which
/// is what every archived message in this product family was measured to
/// carry. Nothing here re-implements a decoder; it converts conventions.
fn load_field(
    raw: &[u8],
    bbox: Option<BoundingBox>,
    no_echo_dbz: f64,
) -> Result<Field, Box<dyn Error>> {
    let (stream, _was_gzipped) = gunzip_if_wrapped(raw, "MRMS archive object")?;
    let selector = FieldSelector::altitude_msl(
        CanonicalField::CompositeReflectivity,
        PRODUCT_ALTITUDE_M,
    );
    let field = rustwx_io::extract_field_from_bytes(&stream, selector)
        .map_err(|e| err(format!("MRMS decode failed: {e}")))?;
    if field.units != UNITS_DBZ {
        return Err(err(format!(
            "MRMS decode returned units {:?}, the seam pins {UNITS_DBZ:?}",
            field.units
        )));
    }
    let source_nx = field.grid.shape.nx;
    let source_ny = field.grid.shape.ny;
    let cells = source_nx
        .checked_mul(source_ny)
        .ok_or_else(|| err("MRMS grid shape overflows"))?;
    if field.values.len() != cells
        || field.grid.lat_deg.len() != cells
        || field.grid.lon_deg.len() != cells
    {
        return Err(err(format!(
            "MRMS decode returned {} values and {}/{} coordinates for a {source_nx}x{source_ny} \
             grid",
            field.values.len(),
            field.grid.lat_deg.len(),
            field.grid.lon_deg.len()
        )));
    }

    // The grid is a regular latitude/longitude template, so a lon/lat box is
    // a rectangle in index space and can be found from one row and one
    // column rather than by testing every cell.
    let (i_start, i_end, j_start, j_end) = match bbox {
        None => (0, source_nx, 0, source_ny),
        Some(box_) => {
            let mut i_lo = usize::MAX;
            let mut i_hi = 0usize;
            for i in 0..source_nx {
                let lon = wrap_longitude(f64::from(field.grid.lon_deg[i]));
                if lon >= box_.west && lon <= box_.east {
                    i_lo = i_lo.min(i);
                    i_hi = i_hi.max(i);
                }
            }
            let mut j_lo = usize::MAX;
            let mut j_hi = 0usize;
            for j in 0..source_ny {
                let lat = f64::from(field.grid.lat_deg[j * source_nx]);
                if lat >= box_.south && lat <= box_.north {
                    j_lo = j_lo.min(j);
                    j_hi = j_hi.max(j);
                }
            }
            if i_lo == usize::MAX || j_lo == usize::MAX {
                return Err(err(format!(
                    "--bbox {},{},{},{} selects no cell of the {source_nx}x{source_ny} MRMS grid",
                    box_.west, box_.south, box_.east, box_.north
                )));
            }
            (i_lo, i_hi + 1, j_lo, j_hi + 1)
        }
    };
    let nx = i_end - i_start;
    let ny = j_end - j_start;

    let mut latitude = Vec::with_capacity(nx * ny);
    let mut longitude = Vec::with_capacity(nx * ny);
    let mut values = Vec::with_capacity(nx * ny);
    let mut valid = Vec::with_capacity(nx * ny);
    let mut no_coverage_cells = 0usize;
    let mut no_echo_cells = 0usize;
    let mut echo_cells = 0usize;
    let mut value_min = f64::INFINITY;
    let mut value_max = f64::NEG_INFINITY;

    for j in j_start..j_end {
        for i in i_start..i_end {
            let index = j * source_nx + i;
            latitude.push(f64::from(field.grid.lat_deg[index]));
            longitude.push(wrap_longitude(f64::from(field.grid.lon_deg[index])));
            let raw_value = f64::from(field.values[index]);
            if !raw_value.is_finite() || raw_value <= NO_COVERAGE_CEILING_DBZ {
                // Either the decoder masked it (bitmap -> NaN) or the
                // product states no radar coverage. Both are unobserved.
                no_coverage_cells += 1;
                values.push(MASKED_FILL_DBZ);
                valid.push(false);
            } else if raw_value <= NO_ECHO_CEILING_DBZ {
                no_echo_cells += 1;
                values.push(no_echo_dbz);
                valid.push(true);
                value_min = value_min.min(no_echo_dbz);
                value_max = value_max.max(no_echo_dbz);
            } else {
                echo_cells += 1;
                values.push(raw_value);
                valid.push(true);
                value_min = value_min.min(raw_value);
                value_max = value_max.max(raw_value);
            }
        }
    }

    let observed = no_echo_cells + echo_cells;
    if observed == 0 {
        return Err(err(
            "every cell of the selected MRMS region is unobserved; a field with no observation \
             in it cannot be scored, and scoring it as all-missing would silently contribute \
             nothing while looking like a frame",
        ));
    }
    // The seam refuses a field whose valid cells leave the quantity's bound,
    // and it is cheaper to say which archive object did it here.
    let (low, high) = seam_bounds(QUANTITY_COMPOSITE_REFLECTIVITY).unwrap();
    if value_min < low || value_max > high {
        return Err(err(format!(
            "decoded reflectivity spans [{value_min:.3}, {value_max:.3}] dBZ, outside the \
             seam bound [{low}, {high}]"
        )));
    }

    Ok(Field {
        grid: GridSpec {
            kind: "regular_latlon".to_string(),
            nx,
            ny,
            source_nx,
            source_ny,
            i_start,
            j_start,
        },
        latitude,
        longitude,
        values,
        valid,
        sentinels: SentinelReport {
            no_coverage_ceiling_dbz: NO_COVERAGE_CEILING_DBZ,
            no_echo_ceiling_dbz: NO_ECHO_CEILING_DBZ,
            no_echo_value_dbz: no_echo_dbz,
            masked_fill_dbz: MASKED_FILL_DBZ,
            no_coverage_cells,
            no_echo_cells,
            echo_cells,
            masked_cells: no_coverage_cells,
            observed_fraction: observed as f64 / (nx * ny) as f64,
        },
        value_min,
        value_max,
    })
}

// ---------------------------------------------------------------- records

#[derive(Serialize)]
struct FrameRecord {
    key: String,
    filename: String,
    valid_time: String,
    gzipped: bool,
    size_bytes: u64,
    last_modified: String,
}

impl From<&Frame> for FrameRecord {
    fn from(frame: &Frame) -> Self {
        Self {
            key: frame.object.key.clone(),
            filename: frame.object.key.rsplit('/').next().unwrap_or("").to_string(),
            valid_time: seam_time(frame.valid_time),
            gzipped: frame.gzipped,
            size_bytes: frame.object.size_bytes,
            last_modified: frame.object.last_modified.clone(),
        }
    }
}

#[derive(Serialize)]
struct WindowRecord {
    bucket: String,
    region: String,
    product: String,
    start: String,
    end: String,
    day_prefixes: Vec<String>,
}

fn window_record(options: &Options, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<WindowRecord, Box<dyn Error>> {
    let region = options.region()?;
    let product = options.product()?;
    Ok(WindowRecord {
        bucket: options.bucket().to_string(),
        region: region.to_string(),
        product: product.to_string(),
        start: seam_time(start),
        end: seam_time(end),
        day_prefixes: window_days(start, end)
            .into_iter()
            .map(|day| day_prefix(region, product, day))
            .collect(),
    })
}

fn cmd_list(options: &Options) -> Result<String, Box<dyn Error>> {
    let (start, end) = options.window()?;
    let agent = build_agent();
    let frames = list_frames(
        &agent,
        options.bucket(),
        options.region()?,
        options.product()?,
        start,
        end,
    )?;
    let matched = frames.len();
    let kept: Vec<&Frame> = match options.limit {
        Some(limit) => frames.iter().take(limit).collect(),
        None => frames.iter().collect(),
    };

    #[derive(Serialize)]
    struct ListRecord {
        schema: &'static str,
        status: &'static str,
        window: WindowRecord,
        matched_frames: usize,
        frames: Vec<FrameRecord>,
        total_bytes: u64,
    }

    let record = ListRecord {
        schema: LIST_SCHEMA,
        status: if kept.is_empty() { "EMPTY" } else { "READY" },
        window: window_record(options, start, end)?,
        matched_frames: matched,
        total_bytes: kept.iter().map(|f| f.object.size_bytes).sum(),
        frames: kept.into_iter().map(FrameRecord::from).collect(),
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

fn cmd_nearest(options: &Options) -> Result<String, Box<dyn Error>> {
    let target = parse_time(
        options
            .valid_time
            .as_deref()
            .ok_or_else(|| err("--valid-time is required"))?,
    )?;
    let window = options.window_seconds.unwrap_or(240);
    let span = chrono::Duration::seconds(window);
    let agent = build_agent();
    let frames = list_frames(
        &agent,
        options.bucket(),
        options.region()?,
        options.product()?,
        target - span,
        target + span,
    )?;
    let best = frames
        .iter()
        .min_by_key(|frame| (frame.valid_time - target).num_seconds().abs())
        .ok_or_else(|| {
            err(format!(
                "no MRMS frame within {window} s of {}; refusing rather than reaching further, \
                 because a distant frame scored as coincident is a wrong number that looks \
                 like a right one",
                seam_time(target)
            ))
        })?;

    #[derive(Serialize)]
    struct NearestRecord {
        schema: &'static str,
        status: &'static str,
        window: WindowRecord,
        requested_valid_time: String,
        window_seconds: i64,
        offset_seconds: i64,
        candidates: usize,
        frame: FrameRecord,
    }

    let record = NearestRecord {
        schema: NEAREST_SCHEMA,
        status: "READY",
        window: window_record(options, target - span, target + span)?,
        requested_valid_time: seam_time(target),
        window_seconds: window,
        offset_seconds: (best.valid_time - target).num_seconds(),
        candidates: frames.len(),
        frame: FrameRecord::from(best),
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

fn cmd_fetch(options: &Options) -> Result<String, Box<dyn Error>> {
    let (start, end) = options.window()?;
    let bucket = options.bucket().to_string();
    let agent = build_agent();
    let frames = list_frames(&agent, &bucket, options.region()?, options.product()?, start, end)?;
    let matched = frames.len();
    let kept: Vec<Frame> = match options.limit {
        Some(limit) => frames.into_iter().take(limit).collect(),
        None => frames,
    };
    if kept.is_empty() {
        return Err(err(format!(
            "no MRMS frames for {} .. {} under s3://{bucket}/{}",
            seam_time(start),
            seam_time(end),
            day_prefix(options.region()?, options.product()?, start.date_naive())
        )));
    }
    let cache_dir = options.cache_dir();

    #[derive(Serialize)]
    struct FileRecord {
        key: String,
        path: String,
        valid_time: String,
        bytes: u64,
        sha256: String,
        fetched_at: String,
        cache_hit: bool,
    }

    #[derive(Serialize)]
    struct FetchRecord {
        schema: &'static str,
        status: &'static str,
        window: WindowRecord,
        matched_frames: usize,
        cache_dir: String,
        files: Vec<FileRecord>,
        total_bytes: u64,
        cache_hits: usize,
    }

    let mut files = Vec::new();
    let mut cache_hits = 0usize;
    for frame in &kept {
        let got = download(&agent, &bucket, &cache_dir, frame, !options.no_cache)?;
        if got.cache_hit {
            cache_hits += 1;
        }
        files.push(FileRecord {
            key: frame.object.key.clone(),
            path: got.path.to_string_lossy().to_string(),
            valid_time: seam_time(frame.valid_time),
            bytes: got.bytes,
            sha256: got.sha256,
            fetched_at: got.fetched_at,
            cache_hit: got.cache_hit,
        });
    }
    let record = FetchRecord {
        schema: FETCH_SCHEMA,
        status: "READY",
        window: window_record(options, start, end)?,
        matched_frames: matched,
        cache_dir: cache_dir.to_string_lossy().to_string(),
        total_bytes: files.iter().map(|f| f.bytes).sum(),
        cache_hits,
        files,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

/// The file to decode, its bytes, and the provenance those bytes earn.
fn source_bytes(options: &Options) -> Result<(PathBuf, Vec<u8>, String), Box<dyn Error>> {
    let path = options
        .file
        .as_deref()
        .ok_or_else(|| err("--file is required (one archived object on disk)"))?;
    let raw = std::fs::read(path).map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
    let sha256 = hex_sha256(&raw);
    Ok((path.to_path_buf(), raw, sha256))
}

fn out_pack_path(options: &Options) -> Result<PathBuf, Box<dyn Error>> {
    let out = options
        .out
        .as_deref()
        .ok_or_else(|| err("--out is required (the pack destination)"))?;
    if out.is_dir() {
        return Err(err(format!(
            "--out {} is a directory; give the pack file path",
            out.display()
        )));
    }
    Ok(out.to_path_buf())
}

fn cmd_decode(options: &Options) -> Result<String, Box<dyn Error>> {
    let (path, raw, sha256) = source_bytes(options)?;
    let out = out_pack_path(options)?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let (valid_time, _) = parse_frame_name(&name, options.product()?).ok_or_else(|| {
        err(format!(
            "cannot read a valid time out of {name:?}; the archive names frames \
             MRMS_<product>_<YYYYMMDD>-<HHMMSS>.grib2[.gz], and a frame decoded under the \
             wrong time would be scored against the wrong hour"
        ))
    })?;

    let field = load_field(&raw, options.bbox, options.no_echo_dbz())?;
    let shape = vec![field.grid.ny, field.grid.nx];
    let mut builder = PayloadBuilder::new();
    builder.push_f64("values", &field.values, shape.clone());
    builder.push_mask("valid", &field.valid, shape);
    let (payload, arrays) = builder.finish();

    let meta = GridPackMeta {
        schema: GRID_SCHEMA.to_string(),
        status: "READY".to_string(),
        quantity: QUANTITY_COMPOSITE_REFLECTIVITY.to_string(),
        units: UNITS_DBZ.to_string(),
        valid_time: seam_time(valid_time),
        provenance: Provenance::new(
            "mrms",
            options.product()?,
            rw_obs::absolute_uri(&path),
            sha256,
            seam_time(Utc::now()),
        ),
        grid: field.grid.clone(),
        sentinels: field.sentinels.clone(),
        value_min_dbz: field.value_min,
        value_max_dbz: field.value_max,
        arrays,
        payload_bytes: payload.len(),
        content_sha256: payload_digest(&payload),
    };
    let bytes = write_pack(&out, &meta, &payload)?;

    #[derive(Serialize)]
    struct DecodeRecord<'a> {
        schema: &'static str,
        status: &'static str,
        pack_path: String,
        pack_schema: &'a str,
        pack_bytes: usize,
        payload_bytes: usize,
        content_sha256: &'a str,
        quantity: &'a str,
        units: &'a str,
        valid_time: &'a str,
        provenance: &'a Provenance,
        grid: &'a GridSpec,
        sentinels: &'a SentinelReport,
        value_min_dbz: f64,
        value_max_dbz: f64,
    }

    let record = DecodeRecord {
        schema: DECODE_SCHEMA,
        status: "READY",
        pack_path: out.to_string_lossy().to_string(),
        pack_schema: &meta.schema,
        pack_bytes: bytes,
        payload_bytes: meta.payload_bytes,
        content_sha256: &meta.content_sha256,
        quantity: &meta.quantity,
        units: &meta.units,
        valid_time: &meta.valid_time,
        provenance: &meta.provenance,
        grid: &meta.grid,
        sentinels: &meta.sentinels,
        value_min_dbz: meta.value_min_dbz,
        value_max_dbz: meta.value_max_dbz,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

fn cmd_grid(options: &Options) -> Result<String, Box<dyn Error>> {
    let (_path, raw, _sha) = source_bytes(options)?;
    let out = out_pack_path(options)?;
    let field = load_field(&raw, options.bbox, options.no_echo_dbz())?;
    let shape = vec![field.grid.ny, field.grid.nx];
    let mut builder = PayloadBuilder::new();
    builder.push_f64("latitude", &field.latitude, shape.clone());
    builder.push_f64("longitude", &field.longitude, shape);
    let (payload, arrays) = builder.finish();
    let meta = GeoPackMeta {
        schema: GEO_SCHEMA.to_string(),
        status: "READY".to_string(),
        source_product: options.product()?.to_string(),
        grid: field.grid.clone(),
        arrays,
        payload_bytes: payload.len(),
        content_sha256: payload_digest(&payload),
    };
    let bytes = write_pack(&out, &meta, &payload)?;

    #[derive(Serialize)]
    struct GridRecord<'a> {
        schema: &'static str,
        status: &'static str,
        pack_path: String,
        pack_schema: &'a str,
        pack_bytes: usize,
        payload_bytes: usize,
        content_sha256: &'a str,
        grid: &'a GridSpec,
    }

    let record = GridRecord {
        schema: GRIDCMD_SCHEMA,
        status: "READY",
        pack_path: out.to_string_lossy().to_string(),
        pack_schema: &meta.schema,
        pack_bytes: bytes,
        payload_bytes: meta.payload_bytes,
        content_sha256: &meta.content_sha256,
        grid: &meta.grid,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

fn cmd_verify(options: &Options) -> Result<String, Box<dyn Error>> {
    let path = options
        .pack
        .as_deref()
        .or(options.file.as_deref())
        .or(options.out.as_deref())
        .ok_or_else(|| err("verify needs the pack path in --pack (or --file/--out)"))?;
    let bytes =
        std::fs::read(path).map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
    // The metadata shape is not known before the schema is read, so the
    // header is decoded into the permissive form first and the schema
    // decides which typed reading has to succeed.
    let (value, payload): (serde_json::Value, Vec<u8>) = decode_pack(&bytes)?;
    let schema = value
        .get("schema")
        .and_then(|s| s.as_str())
        .ok_or_else(|| err("observation pack metadata carries no schema"))?
        .to_string();
    let digest = payload_digest(&payload);
    let declared = value
        .get("content_sha256")
        .and_then(|s| s.as_str())
        .ok_or_else(|| err("observation pack metadata carries no content_sha256"))?;
    if digest != declared {
        return Err(err(format!(
            "observation pack payload digest mismatch: metadata says {declared}, bytes hash \
             to {digest}"
        )));
    }
    let arrays = match schema.as_str() {
        GRID_SCHEMA => {
            let meta: GridPackMeta = serde_json::from_value(value.clone())?;
            meta.arrays
        }
        GEO_SCHEMA => {
            let meta: GeoPackMeta = serde_json::from_value(value.clone())?;
            meta.arrays
        }
        other => {
            return Err(err(format!(
                "pack declares schema {other:?}; this bin writes {GRID_SCHEMA:?} and \
                 {GEO_SCHEMA:?}"
            )))
        }
    };
    validate_arrays(&arrays, payload.len())?;

    #[derive(Serialize)]
    struct VerifyRecord {
        schema: &'static str,
        status: &'static str,
        path: String,
        pack_schema: String,
        bytes: usize,
        payload_bytes: usize,
        content_sha256: String,
        arrays: usize,
    }

    let record = VerifyRecord {
        schema: VERIFY_SCHEMA,
        status: "PASS",
        path: path.to_string_lossy().to_string(),
        pack_schema: schema,
        bytes: bytes.len(),
        payload_bytes: payload.len(),
        content_sha256: digest,
        arrays: arrays.len(),
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_version_and_abi_are_stable_surfaces() {
        assert!(run(&[]).unwrap().contains("usage: rw_mrms"));
        let help = run(&["--help".to_string()]).unwrap();
        for subcommand in ["list", "nearest", "fetch", "decode", "grid", "verify"] {
            assert!(help.contains(subcommand), "usage must document {subcommand}");
        }
        assert!(run(&["--version".to_string()]).unwrap().starts_with("rw_mrms "));
        assert_eq!(run(&["--abi".to_string()]).unwrap(), format!("{ABI_MARKER}\n"));
    }

    #[test]
    fn unknown_subcommands_and_options_fail_closed() {
        assert!(run(&["superob".to_string()]).is_err());
        assert!(Options::parse(&["--nope".to_string()]).is_err());
        assert!(Options::parse(&["--product".to_string()]).is_err());
    }

    #[test]
    fn frame_names_parse_with_their_own_seconds_and_reject_neighbours() {
        let product = DEFAULT_PRODUCT;
        let (when, gz) = parse_frame_name(
            "MRMS_MergedReflectivityQCComposite_00.50_20211210-210037.grib2.gz",
            product,
        )
        .unwrap();
        assert!(gz);
        assert_eq!(seam_time(when), "2021-12-10T21:00:37");
        // Plain (ungzipped) spelling is accepted too.
        let (when, gz) = parse_frame_name(
            "MRMS_MergedReflectivityQCComposite_00.50_20240521-210037.grib2",
            product,
        )
        .unwrap();
        assert!(!gz);
        assert_eq!(seam_time(when), "2024-05-21T21:00:37");
        // A different product's frame is not this product's frame.
        assert!(parse_frame_name(
            "MRMS_MergedReflectivityQCComposite_00.25_20211210-210037.grib2.gz",
            product
        )
        .is_none());
        // Junk, wrong extension, and short stamps are all refused.
        assert!(parse_frame_name("readme.txt", product).is_none());
        assert!(parse_frame_name(
            "MRMS_MergedReflectivityQCComposite_00.50_20211210-2100.grib2.gz",
            product
        )
        .is_none());
        assert!(parse_frame_name(
            "MRMS_MergedReflectivityQCComposite_00.50_20211210-210037.nc",
            product
        )
        .is_none());
    }

    #[test]
    fn day_prefixes_cover_a_window_that_crosses_midnight() {
        let start = parse_time("2021-12-10T23:00:00Z").unwrap();
        let end = parse_time("2021-12-11T01:00:00Z").unwrap();
        let days = window_days(start, end);
        assert_eq!(days.len(), 2);
        let prefixes: Vec<String> = days
            .into_iter()
            .map(|d| day_prefix(DEFAULT_REGION, DEFAULT_PRODUCT, d))
            .collect();
        assert_eq!(
            prefixes,
            vec![
                "CONUS/MergedReflectivityQCComposite_00.50/20211210/".to_string(),
                "CONUS/MergedReflectivityQCComposite_00.50/20211211/".to_string(),
            ]
        );
    }

    #[test]
    fn a_prefix_token_that_would_walk_out_of_the_cache_is_refused() {
        assert!(check_prefix_token("MergedReflectivityQCComposite_00.50", "--product").is_ok());
        assert!(check_prefix_token("CONUS", "--region").is_ok());
        for hostile in ["../etc", "a/b", "..", ".", "", "with space", "C:\\x"] {
            assert!(
                check_prefix_token(hostile, "--product").is_err(),
                "{hostile:?} must be refused"
            );
        }
    }

    #[test]
    fn the_cache_path_refuses_a_key_that_climbs_out() {
        let root = Path::new("cache");
        assert!(cached_object_path(root, "b", "CONUS/x/20211210/f.grib2.gz").is_ok());
        assert!(cached_object_path(root, "b", "../../escape").is_err());
        assert!(cached_object_path(root, "b", "CONUS/../../escape").is_err());
    }

    #[test]
    fn a_backwards_window_is_refused_before_any_request() {
        let options = Options {
            start: Some("2021-12-10T21:30:00Z".to_string()),
            end: Some("2021-12-10T21:00:00Z".to_string()),
            ..Options::default()
        };
        assert!(options.window().unwrap_err().to_string().contains("precedes"));
    }

    #[test]
    fn a_bounding_box_must_be_ordered_wrapped_and_not_cross_the_antimeridian() {
        let box_ = BoundingBox::parse("-100,33,-88,43").unwrap();
        assert_eq!(box_.west, -100.0);
        assert_eq!(box_.north, 43.0);
        for hostile in [
            "-100,33,-88",          // too few
            "-88,33,-100,43",       // east of west
            "-100,43,-88,33",       // north below south
            "170,33,-170,43",       // crosses the antimeridian
            "-100,33,-88,95",       // latitude out of range
            "-190,33,-88,43",       // longitude out of range
            "-100,33,180,43",       // +180 is not a seam longitude
            "a,b,c,d",
        ] {
            assert!(BoundingBox::parse(hostile).is_err(), "{hostile:?} must be refused");
        }
    }

    #[test]
    fn the_no_echo_value_must_stay_inside_the_seam_bound() {
        let parsed = Options::parse(&["--no-echo-dbz".to_string(), "-35".to_string()]).unwrap();
        assert_eq!(parsed.no_echo_dbz(), -35.0);
        assert_eq!(Options::default().no_echo_dbz(), DEFAULT_NO_ECHO_DBZ);
        for hostile in ["-999", "1e9", "nan", "low"] {
            assert!(
                Options::parse(&["--no-echo-dbz".to_string(), hostile.to_string()]).is_err(),
                "{hostile:?} must be refused"
            );
        }
    }

    #[test]
    fn the_two_sentinels_are_ordered_so_no_coverage_never_reads_as_no_echo() {
        // -999 must be caught by the coverage test before the echo test can
        // see it; the ordering is the whole correctness of the classifier.
        assert!(NO_COVERAGE_CEILING_DBZ < NO_ECHO_CEILING_DBZ);
        assert!(-999.0 <= NO_COVERAGE_CEILING_DBZ);
        assert!(-99.0 > NO_COVERAGE_CEILING_DBZ && -99.0 <= NO_ECHO_CEILING_DBZ);
        // The lowest real datum measured in the archive is -31.5 dBZ, which
        // must land on the observed side of both.
        assert!(-31.5 > NO_ECHO_CEILING_DBZ);
        // And every registered FSS threshold is above the no-echo floor, so
        // a no-echo cell can never be counted as an exceedance.
        for threshold in [20.0, 30.0, 40.0, 50.0] {
            assert!(DEFAULT_NO_ECHO_DBZ < threshold);
        }
        let (low, high) = seam_bounds(QUANTITY_COMPOSITE_REFLECTIVITY).unwrap();
        assert!(DEFAULT_NO_ECHO_DBZ >= low && DEFAULT_NO_ECHO_DBZ <= high);
        assert!(MASKED_FILL_DBZ >= low && MASKED_FILL_DBZ <= high);
    }

    #[test]
    fn decode_and_grid_require_both_a_source_and_a_destination() {
        assert!(cmd_decode(&Options::default()).is_err());
        assert!(cmd_grid(&Options::default()).is_err());
        assert!(cmd_verify(&Options::default()).is_err());
        let only_file = Options {
            file: Some(PathBuf::from("nope.grib2.gz")),
            ..Options::default()
        };
        assert!(cmd_decode(&only_file).is_err());
    }

    #[test]
    fn nearest_requires_a_valid_time() {
        assert!(cmd_nearest(&Options::default()).is_err());
    }
}
