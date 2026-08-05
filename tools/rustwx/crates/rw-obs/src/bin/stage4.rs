//! `rw_stage4` — ArWen's Stage-IV precipitation front door.
//!
//! NCEP/EMC Stage IV multi-sensor QPE on the ~4.7625 km HRAP polar
//! stereographic grid, fetched from the Iowa Environmental Mesonet's
//! archive, decoded through the workspace's vendored `grib-core`, and
//! written as the same `gpuwm-obs.obs-grid.v1` pack the reflectivity front
//! door writes.
//!
//! **The edition question, settled against bytes.** The archive names these
//! files `.grib`, which reads as GRIB1 and is how at least one survey
//! classified them. They are GRIB2. Fourteen objects spanning 2021 through
//! 2025, both the hourly and the six-hourly accumulation, every one:
//! edition 2, grid template 3.20 (polar stereographic), product template
//! 4.8, discipline 0 / category 1 / number 8, data representation template
//! **5.3** — complex packing with spatial differencing — with a Section 6
//! bitmap present and `missing_value_management = 0` on all of them.
//!
//! That last number is the one that matters. `grib-core` parses templates
//! 5.2 and 5.3 but seeks past the missing-value-management octets, so a
//! message that used them would decode to plausible-looking nonsense rather
//! than an error. Every archived Stage-IV message measured leaves them zero
//! and carries its missing cells in the bitmap instead, which `grib-core`
//! does honor. `decode` therefore re-proves that on every object it reads
//! and refuses one that does not, rather than trusting the survey.
//!
//! ```text
//! rw_stage4 list   --start 2024-05-21T00:00:00Z --end 2024-05-21T23:00:00Z
//! rw_stage4 fetch  --start ... --end ... [--accumulation 01h] --out DIR
//! rw_stage4 decode --file ST4.2024052121.01h.grib --out F.obspack
//! rw_stage4 grid   --file ST4.2024052121.01h.grib --out grid.obspack
//! rw_stage4 verify --pack F.obspack
//! ```

use std::error::Error;
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

use chrono::{DateTime, Datelike, NaiveDateTime, TimeZone, Timelike, Utc};
use serde::{Deserialize, Serialize};

use grib_core::grib2::{grid_latlon, unpack, Grib2File, Grib2Message};
use rw_nexrad::s3::parse_time;
use rw_obs::net::{agent, get_bytes, get_text};
use rw_obs::pack::{
    decode_pack, payload_digest, validate_arrays, write_pack, ArrayEntry, PayloadBuilder,
    GEO_SCHEMA, GRID_SCHEMA,
};
use rw_obs::seam::{
    seam_bounds, seam_time, wrap_longitude, Provenance, QUANTITY_PRECIPITATION_ACCUMULATION,
    UNITS_MM,
};
use rw_obs::{err, hex_sha256};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The archive root. Probed 2026-08-03: anonymous HTTPS, hourly and
/// six-hourly objects present for every battery-era day tested.
const DEFAULT_ARCHIVE: &str = "https://mesonet.agron.iastate.edu/archive/data";
/// The accumulation windows the archive publishes.
const ACCUMULATIONS: &[(&str, u32)] = &[("01h", 1), ("06h", 6), ("24h", 24)];

/// GRIB2 identity of Stage-IV total precipitation, verified on every sample.
const DISCIPLINE: u8 = 0;
const CATEGORY: u8 = 1;
const PARAMETER: u8 = 8;
/// Grid definition template 3.20 — polar stereographic (the HRAP grid).
const GRID_TEMPLATE: u16 = 20;
/// Data representation template 5.3 — complex packing, spatial differencing.
const EXPECTED_DRT: u16 = 3;

/// Packing noise around zero. A complex-packed accumulation reconstructs to
/// values a hair below zero in cells that held exactly zero; anything inside
/// this is clamped to zero rather than masked, because a dry cell is an
/// observation and masking it would delete the correct negatives a
/// fractions-skill score is built on.
const ZERO_TOLERANCE_MM: f64 = 0.01;
/// What a masked cell is filled with. Never read under a false mask, but the
/// seam requires it to be finite.
const MASKED_FILL_MM: f64 = 0.0;

const LIST_SCHEMA: &str = "gpuwm-obs.stage4-list.v1";
const FETCH_SCHEMA: &str = "gpuwm-obs.stage4-fetch.v1";
const DECODE_SCHEMA: &str = "gpuwm-obs.stage4-decode.v1";
const GRIDCMD_SCHEMA: &str = "gpuwm-obs.stage4-grid.v1";
const VERIFY_SCHEMA: &str = "gpuwm-obs.stage4-verify.v1";

const ABI_MARKER: &str = "gpuwm-obs.stage4-fetch.v1\taccumulation\twindow\tarchive\tfiles\tbytes\t\
sha256\tgpuwm-obs.obs-grid.v1\tprecipitation_accumulation\tmm";

const USAGE: &str = "\
usage: rw_stage4 <list|fetch|decode|grid|verify> [OPTIONS]
       rw_stage4 --version | --help | --abi

  list     report the Stage-IV objects a window resolves to, moving no payload
  fetch    download those objects and print a sha256 per file
  decode   turn one object into a `gpuwm-obs.obs-grid.v1` pack, mm
  grid     write the HRAP latitude/longitude once as a `gpuwm-obs.obs-geo.v1`
  verify   re-read a pack and re-prove its header, index and payload digest

archive options
  --archive URL           default: https://mesonet.agron.iastate.edu/archive/data
                          (anonymous; the NCEP pcpanl production directory
                          keeps only ~2 weeks and cannot serve a hindcast,
                          and NCAR RDA ds507.5 needs a registered account)
  --accumulation TOKEN    01h (default), 06h or 24h
  --start TIME            window start, 2024-05-21T00:00:00Z or 20240521T000000
  --end TIME              window end (inclusive)
  --limit N               keep at most the first N objects
  --cache DIR             object cache root (default ./.rw-obs-cache)
  --no-cache              always re-download, never read the cache
  --out PATH              fetch: a directory. decode/grid: the pack file

decode options
  --file FILE             one archived object on disk
  --allow-missing-value-management
                          decode a message whose data representation template
                          sets missing-value management. OFF by default: the
                          vendored GRIB2 decoder does not parse those octets,
                          so such a message decodes to plausible nonsense
                          rather than to an error, and every archived object
                          measured leaves them zero

verify options
  --pack FILE             the pack to re-prove (--file and --out accepted too)
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(output) => {
            print!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("rw_stage4: {error}");
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
        "--version" | "-V" => return Ok(format!("rw_stage4 {VERSION}\n")),
        "--abi" => return Ok(format!("{ABI_MARKER}\n")),
        _ => {}
    }
    let options = Options::parse(&args[1..])?;
    match first.as_str() {
        "list" => cmd_list(&options),
        "fetch" => cmd_fetch(&options),
        "decode" => cmd_decode(&options),
        "grid" => cmd_grid(&options),
        "verify" => cmd_verify(&options),
        other => Err(err(format!("unknown subcommand {other:?}\n\n{USAGE}"))),
    }
}

#[derive(Debug, Default)]
struct Options {
    archive: Option<String>,
    accumulation: Option<String>,
    start: Option<String>,
    end: Option<String>,
    limit: Option<usize>,
    cache: Option<PathBuf>,
    no_cache: bool,
    out: Option<PathBuf>,
    file: Option<PathBuf>,
    pack: Option<PathBuf>,
    allow_mvm: bool,
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
                "--archive" => options.archive = Some(value()?),
                "--accumulation" => options.accumulation = Some(value()?),
                "--start" => options.start = Some(value()?),
                "--end" => options.end = Some(value()?),
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
                "--allow-missing-value-management" => options.allow_mvm = true,
                other => return Err(err(format!("unknown option {other:?}\n\n{USAGE}"))),
            }
            index += 1;
        }
        Ok(options)
    }

    fn archive(&self) -> &str {
        self.archive
            .as_deref()
            .unwrap_or(DEFAULT_ARCHIVE)
            .trim_end_matches('/')
    }

    fn accumulation(&self) -> Result<(&'static str, u32), Box<dyn Error>> {
        let token = self.accumulation.as_deref().unwrap_or("01h");
        ACCUMULATIONS
            .iter()
            .find(|(name, _)| *name == token)
            .map(|(name, hours)| (*name, *hours))
            .ok_or_else(|| {
                err(format!(
                    "--accumulation {token:?} is not one the archive publishes; expected one of \
                     {}",
                    ACCUMULATIONS
                        .iter()
                        .map(|(n, _)| *n)
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })
    }

    fn window(&self) -> Result<(DateTime<Utc>, DateTime<Utc>), Box<dyn Error>> {
        let start = parse_time(self.start.as_deref().ok_or_else(|| err("--start is required"))?)?;
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
        self.cache
            .clone()
            .unwrap_or_else(|| PathBuf::from(".rw-obs-cache"))
    }
}

/// `ST4.YYYYMMDDHH.<acc>.grib` — the archive's own name for one object.
fn object_name(valid: DateTime<Utc>, accumulation: &str) -> String {
    format!(
        "ST4.{:04}{:02}{:02}{:02}.{accumulation}.grib",
        valid.year(),
        valid.month(),
        valid.day(),
        valid.hour()
    )
}

fn object_url(archive: &str, valid: DateTime<Utc>, accumulation: &str) -> String {
    format!(
        "{archive}/{:04}/{:02}/{:02}/stage4/{}",
        valid.year(),
        valid.month(),
        valid.day(),
        object_name(valid, accumulation)
    )
}

/// Read a valid time and accumulation back out of an object name.
fn parse_object_name(name: &str) -> Option<(DateTime<Utc>, &'static str, u32)> {
    let rest = name.strip_prefix("ST4.")?;
    let mut parts = rest.split('.');
    let stamp = parts.next()?;
    let token = parts.next()?;
    if parts.next()? != "grib" || parts.next().is_some() {
        return None;
    }
    if stamp.len() != 10 {
        return None;
    }
    let naive = NaiveDateTime::parse_from_str(&format!("{stamp}0000"), "%Y%m%d%H%M%S").ok()?;
    let (name, hours) = ACCUMULATIONS.iter().find(|(n, _)| *n == token)?;
    Some((Utc.from_utc_datetime(&naive), *name, *hours))
}

/// Every top-of-hour instant the window covers, at the accumulation's own
/// stride. Six-hourly objects exist only at 00/06/12/18 UTC and twenty-four
/// hourly only at 12 UTC, so a window is walked at the stride the archive
/// actually publishes rather than hourly-and-hope.
fn window_instants(
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    hours: u32,
) -> Vec<DateTime<Utc>> {
    let mut out = Vec::new();
    let step = chrono::Duration::hours(1);
    let mut when = start
        .with_minute(0)
        .and_then(|t| t.with_second(0))
        .and_then(|t| t.with_nanosecond(0))
        .unwrap_or(start);
    while when <= end {
        let keep = match hours {
            1 => true,
            6 => when.hour() % 6 == 0,
            24 => when.hour() == 12,
            _ => true,
        };
        if keep && when >= start {
            out.push(when);
        }
        when += step;
    }
    out
}

fn cached_object_path(cache_dir: &Path, name: &str) -> Result<PathBuf, Box<dyn Error>> {
    let root = cache_dir.join("stage4");
    let path = root.join(name);
    let ok = path.strip_prefix(&root).is_ok_and(|tail| {
        let mut components = tail.components();
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
    });
    if !ok {
        return Err(err(format!(
            "archive object {name:?} does not name a single file under {}",
            root.display()
        )));
    }
    Ok(path)
}

// -------------------------------------------------------------- decoding

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GridSpec {
    kind: String,
    nx: usize,
    ny: usize,
    /// HRAP geometry, as the message states it.
    lat1_deg: f64,
    lon1_deg: f64,
    lov_deg: f64,
    lad_deg: f64,
    dx_m: f64,
    dy_m: f64,
    scan_mode: u8,
    projection_center_flag: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PackingReport {
    data_representation_template: u16,
    bits_per_value: u8,
    /// Present so a receipt records that the check ran, not only that it
    /// passed. `grib-core` does not parse the missing-value-management
    /// octets of templates 5.2/5.3, so a nonzero value here means the
    /// decode cannot be trusted and `decode` refuses by default.
    spatial_diff_order: u8,
    bitmap_present: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MaskReport {
    missing_cells: usize,
    dry_cells: usize,
    wet_cells: usize,
    clamped_negative_cells: usize,
    observed_fraction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GridPackMeta {
    schema: String,
    status: String,
    quantity: String,
    units: String,
    valid_time: String,
    accumulation_hours: u32,
    provenance: Provenance,
    grid: GridSpec,
    packing: PackingReport,
    mask: MaskReport,
    value_min_mm: f64,
    value_max_mm: f64,
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

struct Field {
    grid: GridSpec,
    packing: PackingReport,
    latitude: Vec<f64>,
    longitude: Vec<f64>,
    values: Vec<f64>,
    valid: Vec<bool>,
    mask: MaskReport,
    value_min: f64,
    value_max: f64,
}

/// Pick the one total-precipitation message and prove it is the shape every
/// archived object was measured to have.
fn select_message(file: &Grib2File, allow_mvm: bool) -> Result<&Grib2Message, Box<dyn Error>> {
    let mut matches = file.messages.iter().filter(|message| {
        message.discipline == DISCIPLINE
            && message.product.parameter_category == CATEGORY
            && message.product.parameter_number == PARAMETER
    });
    let message = matches.next().ok_or_else(|| {
        err(format!(
            "no total-precipitation message (discipline {DISCIPLINE}, category {CATEGORY}, \
             number {PARAMETER}) in this object; it carried {} message(s)",
            file.messages.len()
        ))
    })?;
    if matches.next().is_some() {
        return Err(err(
            "this object carries more than one total-precipitation message; which one is the \
             accumulation is then a guess, and a guess that picks the other one is a silently \
             wrong score",
        ));
    }
    if message.grid.template != GRID_TEMPLATE {
        return Err(err(format!(
            "Stage-IV message states grid template 3.{}, expected 3.{GRID_TEMPLATE} (polar \
             stereographic HRAP)",
            message.grid.template
        )));
    }
    let drt = message.data_rep.template;
    if drt != EXPECTED_DRT && !allow_mvm {
        return Err(err(format!(
            "Stage-IV message uses data representation template 5.{drt}; every archived object \
             measured used 5.{EXPECTED_DRT}. Pass \
             --allow-missing-value-management to decode it anyway, having first confirmed the \
             vendored decoder handles 5.{drt}'s missing-value management"
        )));
    }
    Ok(message)
}

fn load_field(raw: &[u8], allow_mvm: bool) -> Result<Field, Box<dyn Error>> {
    let file = Grib2File::from_bytes(raw)
        .map_err(|e| err(format!("Stage-IV object will not parse as GRIB2: {e}")))?;
    let message = select_message(&file, allow_mvm)?;
    let nx = message.grid.nx as usize;
    let ny = message.grid.ny as usize;
    let cells = nx
        .checked_mul(ny)
        .ok_or_else(|| err("Stage-IV grid shape overflows"))?;

    // `unpack_message`, not the row-normalized variant: `grid_latlon`
    // returns coordinates in the message's own declared scan order, and the
    // normalizing variant flips rows when scan-mode bit 6 is set. Pairing
    // the two would silently mirror the field north-for-south on exactly the
    // grids where the flip fires. Ordering is the scorer's business only
    // through the coordinates it is handed, and these two agree by
    // construction.
    let values = unpack::unpack_message(message)
        .map_err(|e| err(format!("Stage-IV message will not unpack: {e}")))?;
    if values.len() != cells {
        return Err(err(format!(
            "Stage-IV message unpacked to {} values for a {nx}x{ny} grid",
            values.len()
        )));
    }
    let (lats, lons) = grid_latlon(&message.grid);
    if lats.len() != cells || lons.len() != cells {
        return Err(err(format!(
            "Stage-IV grid template 3.{} yielded {}/{} coordinates for {cells} cells; the \
             vendored decoder does not place this projection",
            message.grid.template,
            lats.len(),
            lons.len()
        )));
    }

    let (low, high) = seam_bounds(QUANTITY_PRECIPITATION_ACCUMULATION).unwrap();
    let mut out_values = Vec::with_capacity(cells);
    let mut valid = Vec::with_capacity(cells);
    let mut latitude = Vec::with_capacity(cells);
    let mut longitude = Vec::with_capacity(cells);
    let mut missing_cells = 0usize;
    let mut dry_cells = 0usize;
    let mut wet_cells = 0usize;
    let mut clamped = 0usize;
    let mut value_min = f64::INFINITY;
    let mut value_max = f64::NEG_INFINITY;

    for index in 0..cells {
        latitude.push(lats[index]);
        longitude.push(wrap_longitude(lons[index]));
        let value = values[index];
        if !value.is_finite() {
            // The Section-6 bitmap's cells arrive as NaN; that is the whole
            // mechanism by which Stage-IV states "outside the analysis".
            missing_cells += 1;
            out_values.push(MASKED_FILL_MM);
            valid.push(false);
            continue;
        }
        let value = if value < 0.0 && value >= -ZERO_TOLERANCE_MM {
            clamped += 1;
            0.0
        } else {
            value
        };
        if value < low || value > high {
            // Not packing noise: a number this far outside the quantity is a
            // fill convention this decoder does not know, and guessing would
            // put it in a rain total.
            missing_cells += 1;
            out_values.push(MASKED_FILL_MM);
            valid.push(false);
            continue;
        }
        if value > 0.0 {
            wet_cells += 1;
        } else {
            dry_cells += 1;
        }
        out_values.push(value);
        valid.push(true);
        value_min = value_min.min(value);
        value_max = value_max.max(value);
    }

    let observed = dry_cells + wet_cells;
    if observed == 0 {
        return Err(err(
            "every cell of this Stage-IV object is masked; a field with no observation cannot \
             be scored",
        ));
    }

    Ok(Field {
        grid: GridSpec {
            kind: "polar_stereographic".to_string(),
            nx,
            ny,
            lat1_deg: message.grid.lat1,
            lon1_deg: wrap_longitude(message.grid.lon1),
            lov_deg: wrap_longitude(message.grid.lov),
            lad_deg: message.grid.lad,
            dx_m: message.grid.dx,
            dy_m: message.grid.dy,
            scan_mode: message.grid.scan_mode,
            projection_center_flag: message.grid.projection_center_flag,
        },
        packing: PackingReport {
            data_representation_template: message.data_rep.template,
            bits_per_value: message.data_rep.bits_per_value,
            spatial_diff_order: message.data_rep.spatial_diff_order,
            bitmap_present: message.bitmap.is_some(),
        },
        latitude,
        longitude,
        values: out_values,
        valid,
        mask: MaskReport {
            missing_cells,
            dry_cells,
            wet_cells,
            clamped_negative_cells: clamped,
            observed_fraction: observed as f64 / cells as f64,
        },
        value_min,
        value_max,
    })
}

// --------------------------------------------------------------- commands

#[derive(Serialize)]
struct ObjectRecord {
    filename: String,
    url: String,
    valid_time: String,
    accumulation_hours: u32,
}

fn cmd_list(options: &Options) -> Result<String, Box<dyn Error>> {
    let (start, end) = options.window()?;
    let (token, hours) = options.accumulation()?;
    let archive = options.archive().to_string();
    let agent = agent();

    // The archive publishes an ordinary directory index per day, so
    // membership is read rather than assumed: constructing a URL for every
    // hour and calling a 404 "absent" cannot tell a gap in the record from a
    // server having a bad afternoon.
    let mut present: std::collections::BTreeSet<String> = Default::default();
    let mut days_listed = Vec::new();
    let mut day = start.date_naive();
    while day <= end.date_naive() {
        let url = format!(
            "{archive}/{:04}/{:02}/{:02}/stage4/",
            day.year(),
            day.month(),
            day.day()
        );
        let body = get_text(&agent, &url, "Stage-IV day index")?;
        for name in body.split("href=\"").skip(1).filter_map(|s| s.split('"').next()) {
            if parse_object_name(name).is_some() {
                present.insert(name.to_string());
            }
        }
        days_listed.push(url);
        match day.succ_opt() {
            Some(next) => day = next,
            None => break,
        }
    }

    let mut objects: Vec<ObjectRecord> = window_instants(start, end, hours)
        .into_iter()
        .filter(|when| present.contains(&object_name(*when, token)))
        .map(|when| ObjectRecord {
            filename: object_name(when, token),
            url: object_url(&archive, when, token),
            valid_time: seam_time(when),
            accumulation_hours: hours,
        })
        .collect();
    let matched = objects.len();
    if let Some(limit) = options.limit {
        objects.truncate(limit);
    }

    #[derive(Serialize)]
    struct ListRecord {
        schema: &'static str,
        status: &'static str,
        archive: String,
        accumulation: &'static str,
        accumulation_hours: u32,
        start: String,
        end: String,
        day_indexes: Vec<String>,
        expected_objects: usize,
        matched_objects: usize,
        objects: Vec<ObjectRecord>,
    }

    let record = ListRecord {
        schema: LIST_SCHEMA,
        status: if objects.is_empty() { "EMPTY" } else { "READY" },
        archive,
        accumulation: token,
        accumulation_hours: hours,
        start: seam_time(start),
        end: seam_time(end),
        day_indexes: days_listed,
        expected_objects: window_instants(start, end, hours).len(),
        matched_objects: matched,
        objects,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

fn cmd_fetch(options: &Options) -> Result<String, Box<dyn Error>> {
    let (start, end) = options.window()?;
    let (token, hours) = options.accumulation()?;
    let archive = options.archive().to_string();
    let cache_dir = options.cache_dir();
    let agent = agent();
    let mut instants = window_instants(start, end, hours);
    if let Some(limit) = options.limit {
        instants.truncate(limit);
    }
    if instants.is_empty() {
        return Err(err(format!(
            "no {token} Stage-IV instant lies in {} .. {}",
            seam_time(start),
            seam_time(end)
        )));
    }

    #[derive(Serialize)]
    struct FileRecord {
        filename: String,
        url: String,
        path: String,
        valid_time: String,
        accumulation_hours: u32,
        bytes: u64,
        sha256: String,
        fetched_at: String,
        cache_hit: bool,
    }

    let mut files = Vec::new();
    let mut cache_hits = 0usize;
    for when in instants {
        let name = object_name(when, token);
        let url = object_url(&archive, when, token);
        let target = cached_object_path(&cache_dir, &name)?;
        let fetched_at = seam_time(Utc::now());
        let (bytes, hit) = if !options.no_cache && target.is_file() {
            (std::fs::read(&target)?, true)
        } else {
            let bytes = get_bytes(&agent, &url, "Stage-IV object")?;
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
            (bytes, false)
        };
        if hit {
            cache_hits += 1;
        }
        files.push(FileRecord {
            filename: name,
            url,
            path: target.to_string_lossy().to_string(),
            valid_time: seam_time(when),
            accumulation_hours: hours,
            bytes: bytes.len() as u64,
            sha256: hex_sha256(&bytes),
            fetched_at,
            cache_hit: hit,
        });
    }

    #[derive(Serialize)]
    struct FetchRecord {
        schema: &'static str,
        status: &'static str,
        archive: String,
        accumulation: &'static str,
        start: String,
        end: String,
        cache_dir: String,
        total_bytes: u64,
        cache_hits: usize,
        files: Vec<FileRecord>,
    }

    let record = FetchRecord {
        schema: FETCH_SCHEMA,
        status: "READY",
        archive,
        accumulation: token,
        start: seam_time(start),
        end: seam_time(end),
        cache_dir: cache_dir.to_string_lossy().to_string(),
        total_bytes: files.iter().map(|f| f.bytes).sum(),
        cache_hits,
        files,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

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
    let (valid_time, _token, hours) = parse_object_name(&name).ok_or_else(|| {
        err(format!(
            "cannot read a valid time and accumulation out of {name:?}; the archive names \
             objects ST4.YYYYMMDDHH.<01h|06h|24h>.grib, and an accumulation decoded under the \
             wrong window would be compared against the wrong model total"
        ))
    })?;

    let field = load_field(&raw, options.allow_mvm)?;
    let shape = vec![field.grid.ny, field.grid.nx];
    let mut builder = PayloadBuilder::new();
    builder.push_f64("values", &field.values, shape.clone());
    builder.push_mask("valid", &field.valid, shape);
    let (payload, arrays) = builder.finish();

    let meta = GridPackMeta {
        schema: GRID_SCHEMA.to_string(),
        status: "READY".to_string(),
        quantity: QUANTITY_PRECIPITATION_ACCUMULATION.to_string(),
        units: UNITS_MM.to_string(),
        valid_time: seam_time(valid_time),
        accumulation_hours: hours,
        provenance: Provenance::new(
            "stage4",
            format!("ST4-{hours:02}h"),
            rw_obs::absolute_uri(&path),
            sha256,
            seam_time(Utc::now()),
        ),
        grid: field.grid.clone(),
        packing: field.packing.clone(),
        mask: field.mask.clone(),
        value_min_mm: field.value_min,
        value_max_mm: field.value_max,
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
        accumulation_hours: u32,
        provenance: &'a Provenance,
        grid: &'a GridSpec,
        packing: &'a PackingReport,
        mask: &'a MaskReport,
        value_min_mm: f64,
        value_max_mm: f64,
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
        accumulation_hours: meta.accumulation_hours,
        provenance: &meta.provenance,
        grid: &meta.grid,
        packing: &meta.packing,
        mask: &meta.mask,
        value_min_mm: meta.value_min_mm,
        value_max_mm: meta.value_max_mm,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

fn cmd_grid(options: &Options) -> Result<String, Box<dyn Error>> {
    let (_path, raw, _sha) = source_bytes(options)?;
    let out = out_pack_path(options)?;
    let field = load_field(&raw, options.allow_mvm)?;
    let shape = vec![field.grid.ny, field.grid.nx];
    let mut builder = PayloadBuilder::new();
    builder.push_f64("latitude", &field.latitude, shape.clone());
    builder.push_f64("longitude", &field.longitude, shape);
    let (payload, arrays) = builder.finish();
    let meta = GeoPackMeta {
        schema: GEO_SCHEMA.to_string(),
        status: "READY".to_string(),
        source_product: "stage4-hrap".to_string(),
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
        GRID_SCHEMA => serde_json::from_value::<GridPackMeta>(value.clone())?.arrays,
        GEO_SCHEMA => serde_json::from_value::<GeoPackMeta>(value.clone())?.arrays,
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
        assert!(run(&[]).unwrap().contains("usage: rw_stage4"));
        let help = run(&["--help".to_string()]).unwrap();
        for subcommand in ["list", "fetch", "decode", "grid", "verify"] {
            assert!(help.contains(subcommand), "usage must document {subcommand}");
        }
        assert!(run(&["--version".to_string()]).unwrap().starts_with("rw_stage4 "));
        assert_eq!(run(&["--abi".to_string()]).unwrap(), format!("{ABI_MARKER}\n"));
    }

    #[test]
    fn the_help_records_why_this_archive_and_not_the_two_obvious_ones() {
        for needle in ["pcpanl", "ds507.5", "registered account", "2 weeks"] {
            assert!(USAGE.contains(needle), "--help does not mention {needle:?}");
        }
    }

    #[test]
    fn unknown_subcommands_and_options_fail_closed() {
        assert!(run(&["accumulate".to_string()]).is_err());
        assert!(Options::parse(&["--nope".to_string()]).is_err());
        assert!(Options::parse(&["--accumulation".to_string()]).is_err());
    }

    #[test]
    fn object_names_round_trip_through_their_own_spelling() {
        let when = parse_time("2024-05-21T21:00:00Z").unwrap();
        assert_eq!(object_name(when, "01h"), "ST4.2024052121.01h.grib");
        let (back, token, hours) = parse_object_name("ST4.2024052121.01h.grib").unwrap();
        assert_eq!(back, when);
        assert_eq!(token, "01h");
        assert_eq!(hours, 1);
        let (_, _, hours) = parse_object_name("ST4.2021121018.06h.grib").unwrap();
        assert_eq!(hours, 6);
        let (_, _, hours) = parse_object_name("ST4.2024052112.24h.grib").unwrap();
        assert_eq!(hours, 24);
    }

    #[test]
    fn names_that_are_not_this_products_are_refused() {
        for hostile in [
            "ST4.2024052121.03h.grib",       // an accumulation the archive does not publish
            "ST4.20240521.01h.grib",         // no hour
            "ST4.2024052121.01h.grib2",      // a different extension
            "ST4.2024052121.01h.grib.gz",    // an extra component
            "conus_20240521_24h.grb2",       // the daily source tar's member
            "index.html",
        ] {
            assert!(parse_object_name(hostile).is_none(), "{hostile:?} must be refused");
        }
    }

    #[test]
    fn the_url_is_built_from_the_archives_own_layout() {
        let when = parse_time("2024-05-21T21:00:00Z").unwrap();
        assert_eq!(
            object_url(DEFAULT_ARCHIVE, when, "01h"),
            "https://mesonet.agron.iastate.edu/archive/data/2024/05/21/stage4/\
             ST4.2024052121.01h.grib"
        );
    }

    #[test]
    fn a_window_is_walked_at_the_stride_the_archive_publishes() {
        let start = parse_time("2024-05-21T00:00:00Z").unwrap();
        let end = parse_time("2024-05-21T23:00:00Z").unwrap();
        assert_eq!(window_instants(start, end, 1).len(), 24);
        let six = window_instants(start, end, 6);
        assert_eq!(six.len(), 4);
        assert_eq!(six.iter().map(|t| t.hour()).collect::<Vec<_>>(), vec![0, 6, 12, 18]);
        let day = window_instants(start, end, 24);
        assert_eq!(day.len(), 1);
        assert_eq!(day[0].hour(), 12);
    }

    #[test]
    fn accumulation_tokens_outside_the_archives_set_are_refused() {
        assert_eq!(Options::default().accumulation().unwrap(), ("01h", 1));
        for token in ["03h", "1h", "hourly", ""] {
            let options = Options {
                accumulation: Some(token.to_string()),
                ..Options::default()
            };
            assert!(options.accumulation().is_err(), "{token:?} must be refused");
        }
    }

    #[test]
    fn a_backwards_window_is_refused_before_any_request() {
        let options = Options {
            start: Some("2024-05-21T21:00:00Z".to_string()),
            end: Some("2024-05-21T20:00:00Z".to_string()),
            ..Options::default()
        };
        assert!(options.window().unwrap_err().to_string().contains("precedes"));
    }

    #[test]
    fn the_cache_path_refuses_an_object_name_that_climbs_out() {
        let root = Path::new("cache");
        assert!(cached_object_path(root, "ST4.2024052121.01h.grib").is_ok());
        assert!(cached_object_path(root, "../escape").is_err());
        assert!(cached_object_path(root, "a/b").is_err());
    }

    #[test]
    fn the_zero_tolerance_is_narrower_than_the_smallest_scored_threshold() {
        // The registered precipitation thresholds are 2.5, 10 and 25 mm; a
        // clamp that could reach any of them would be inventing dry cells.
        assert!(ZERO_TOLERANCE_MM < 2.5);
        let (low, high) = seam_bounds(QUANTITY_PRECIPITATION_ACCUMULATION).unwrap();
        assert!(MASKED_FILL_MM >= low && MASKED_FILL_MM <= high);
    }

    #[test]
    fn decode_and_verify_require_their_arguments() {
        assert!(cmd_decode(&Options::default()).is_err());
        assert!(cmd_grid(&Options::default()).is_err());
        assert!(cmd_verify(&Options::default()).is_err());
    }
}
