//! `rw_ndbc` -- the ocean-platform front door.
//!
//! NDBC's public realtime feed: `https://www.ndbc.noaa.gov/data/realtime2/
//! <STATION>.txt` holds the last 45 days of a platform's standard
//! meteorological record (moored buoys, C-MAN coastal stations, ships of
//! opportunity are not in it), and `data/stations/station_table.txt` is the
//! platform table with positions.  `fetch` downloads the table and every
//! platform file the index names (or `--stations`), with a SHA-256 and the
//! server's `Last-Modified` per file; `table` decodes them into the neutral
//! observation table (`gpuwm-obs.table.v1`).
//!
//! The record format (`#YY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES
//! ATMP WTMP DEWP VIS PTDY TIDE`, `MM` for missing, the second header line
//! the units: degT, m/s, hPa, degC): `PRES` is sea-level pressure at a
//! platform whose barometer is within a few metres of the sea, so it is
//! written as a `surface_pressure_pa` row anchored at 0 m and labelled
//! `sea_level_pressure` in the table's measurement column (a reduced
//! pressure, never a station pressure); `ATMP` and `DEWP` are the air
//! temperature and dewpoint at the sensor (about 4 m, `platform_temperature`
//! and `platform_dewpoint`);
//! `WDIR`/`WSPD` are the anemometer's, at 5 m on a 3 m discus buoy and 10
//! m on a C-MAN tower, so a buoy wind is reduced to 10 m by the neutral log
//! law over water (`z0` 1e-4 m, factor ln(10/z0)/ln(5/z0) = 1.064) and a
//! fixed platform's is used as reported; the reduction is stated in the
//! record and the buoy wind carries a 3.0 m/s error for it.  Platform type
//! comes from the station table's `TTYPE` column; a type this reader does
//! not know is treated as a buoy and counted.
//!
//! One report per platform per hour is kept: the report nearest the top of
//! the hour within `--match-minutes` (default 10), so a buoy reporting
//! every ten minutes offers the analysis one row set per hour and the
//! six-minute stations do not weigh six times a buoy.
//!
//! ```text
//! rw_ndbc fetch --out DIR [--stations LIST] [--limit N] [--request-pause-ms N]
//! rw_ndbc table --dir DIR --start TIME --end TIME --out FILE.csv [--fetch-record FILE]
//! rw_ndbc verify --file FILE.csv --record FILE.json
//! ```

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use chrono::{DateTime, Duration, NaiveDate, TimeZone, Timelike, Utc};
use serde::{Deserialize, Serialize};

use rw_nexrad::s3::{parse_http_date, parse_time};
use rw_obs::net::agent;
use rw_obs::seam::seam_time;
use rw_obs::table::{
    wind_components, RowProvenance, TableRow, TableWriter, ERROR_DEWPOINT_BUOY_K,
    ERROR_SURFACE_PRESSURE_PA, ERROR_TEMPERATURE_SURFACE_K, ERROR_WIND_BUOY_M_S,
    ERROR_WIND_SURFACE_M_S, GROSS_DEWPOINT_K, GROSS_SURFACE_PRESSURE_PA, GROSS_TEMPERATURE_K,
    GROSS_WIND_M_S, MEAS_ANEMOMETER_WIND_10M, MEAS_ANEMOMETER_WIND_5M_TO_10M,
    MEAS_PLATFORM_DEWPOINT, MEAS_PLATFORM_TEMPERATURE, MEAS_SEA_LEVEL_PRESSURE, TABLE_SCHEMA,
    VAR_DEWPOINT, VAR_SURFACE_PRESSURE, VAR_TEMPERATURE, VAR_WIND_U, VAR_WIND_V,
};
use rw_obs::{err, hex_sha256};

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

const DEFAULT_ARCHIVE: &str = "https://www.ndbc.noaa.gov/data";
const DEFAULT_REQUEST_PAUSE_MS: u64 = 100;
const DEFAULT_MATCH_MINUTES: i64 = 10;
const SOURCE: &str = "ndbc";
/// Anemometer heights by platform class, metres.
const BUOY_ANEMOMETER_M: f64 = 5.0;
const FIXED_ANEMOMETER_M: f64 = 10.0;
/// Aerodynamic roughness of the sea for the neutral log-law reduction.
const SEA_ROUGHNESS_M: f64 = 1.0e-4;

const FETCH_SCHEMA: &str = "gpuwm-obs.ndbc-fetch.v1";
const TABLE_RECORD_SCHEMA: &str = "gpuwm-obs.ndbc-table.v1";
const VERIFY_SCHEMA: &str = "gpuwm-obs.ndbc-verify.v1";
const ABI_MARKER: &str = "gpuwm-obs.ndbc-fetch.v1\tgpuwm-obs.ndbc-table.v1\tgpuwm-obs.table.v2\t\
surface_pressure_pa\ttemperature_k\tdewpoint_k\twind_u_m_s\twind_v_m_s\tanemometer_reduction";

const USAGE: &str = "\
usage: rw_ndbc <fetch|table|verify> [OPTIONS]
       rw_ndbc --version | --help | --abi

  fetch   download the station table and the realtime2 platform files
  table   decode platform files in a window into a `gpuwm-obs.table.v1` CSV
  verify  re-hash a table against the record written beside it

fetch options
  --archive URL          default https://www.ndbc.noaa.gov/data
  --out DIR              where station_table.txt and realtime2/ land
  --stations LIST        comma-separated platform ids (default: every file the
                         realtime2 index names)
  --limit N              fetch at most N platform files
  --request-pause-ms N   pause between files (default 100)

table options
  --dir DIR              the directory `fetch` wrote
  --start TIME --end TIME
                         the window of report times kept (inclusive)
  --match-minutes N      a report is that hour's when within N minutes of the
                         top of the hour (default 10)
  --out FILE.csv         the table; a .json record is written beside it
  --fetch-record FILE    the `fetch` record, for per-file latency

verify options
  --file FILE.csv --record FILE.json
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
            eprintln!("rw_ndbc: {error}");
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
        "--version" | "-V" => return Ok(format!("rw_ndbc {VERSION}\n")),
        "--abi" => return Ok(format!("{ABI_MARKER}\n")),
        _ => {}
    }
    let options = Options::parse(&args[1..])?;
    match first.as_str() {
        "fetch" => cmd_fetch(&options),
        "table" => cmd_table(&options),
        "verify" => cmd_verify(&options),
        other => Err(err(format!("unknown subcommand {other:?}\n\n{USAGE}"))),
    }
}

#[derive(Debug, Default)]
struct Options {
    archive: Option<String>,
    out: Option<PathBuf>,
    stations: Option<Vec<String>>,
    limit: Option<usize>,
    request_pause_ms: Option<u64>,
    dir: Option<PathBuf>,
    start: Option<String>,
    end: Option<String>,
    match_minutes: Option<i64>,
    fetch_record: Option<PathBuf>,
    file: Option<PathBuf>,
    record: Option<PathBuf>,
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
                "--out" => options.out = Some(PathBuf::from(value()?)),
                "--stations" => {
                    let raw = value()?;
                    let list: Vec<String> = raw
                        .split(',')
                        .map(|s| s.trim().to_ascii_uppercase())
                        .filter(|s| !s.is_empty())
                        .collect();
                    if list.is_empty() || list.iter().any(|s| !s.bytes().all(|b| b.is_ascii_alphanumeric())) {
                        return Err(err("--stations expects alphanumeric platform ids"));
                    }
                    options.stations = Some(list);
                }
                "--limit" => {
                    let raw = value()?;
                    options.limit = Some(raw.parse().map_err(|_| err(format!("--limit expects a count, got {raw:?}")))?);
                }
                "--request-pause-ms" => {
                    let raw = value()?;
                    options.request_pause_ms = Some(raw.parse().map_err(|_| {
                        err(format!("--request-pause-ms expects milliseconds, got {raw:?}"))
                    })?);
                }
                "--dir" => options.dir = Some(PathBuf::from(value()?)),
                "--start" => options.start = Some(value()?),
                "--end" => options.end = Some(value()?),
                "--match-minutes" => {
                    let raw = value()?;
                    let m: i64 = raw.parse().map_err(|_| err(format!("--match-minutes expects a count, got {raw:?}")))?;
                    if !(1..=30).contains(&m) {
                        return Err(err("--match-minutes must lie in [1, 30]"));
                    }
                    options.match_minutes = Some(m);
                }
                "--fetch-record" => options.fetch_record = Some(PathBuf::from(value()?)),
                "--file" => options.file = Some(PathBuf::from(value()?)),
                "--record" => options.record = Some(PathBuf::from(value()?)),
                other => return Err(err(format!("unknown option {other:?}\n\n{USAGE}"))),
            }
            index += 1;
        }
        Ok(options)
    }

    fn archive(&self) -> &str {
        self.archive.as_deref().unwrap_or(DEFAULT_ARCHIVE).trim_end_matches('/')
    }

    fn window(&self) -> Result<(DateTime<Utc>, DateTime<Utc>), Box<dyn Error>> {
        let start = parse_time(self.start.as_deref().ok_or_else(|| err("--start is required"))?)?;
        let end = parse_time(self.end.as_deref().ok_or_else(|| err("--end is required"))?)?;
        if end < start {
            return Err(err(format!("--end {} precedes --start {}", seam_time(end), seam_time(start))));
        }
        Ok((start, end))
    }
}

// ----------------------------------------------------------------- fetch

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FetchedFile {
    name: String,
    url: String,
    path: String,
    bytes: usize,
    sha256: String,
    last_modified: Option<String>,
    fetched_at: String,
    wall_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FetchRecord {
    schema: String,
    status: String,
    archive: String,
    out_dir: String,
    station_table: FetchedFile,
    files: Vec<FetchedFile>,
    /// Platform files the feed answered 404 for.
    #[serde(default)]
    missing: Vec<String>,
    total_bytes: usize,
    wall_s: f64,
}

fn get_with_last_modified(
    client: &ureq::Agent,
    url: &str,
) -> Result<(Vec<u8>, Option<DateTime<Utc>>), Box<dyn Error>> {
    let mut response = client.get(url).call().map_err(|e| err(format!("GET {url} failed: {e}")))?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(err(format!("GET {url} answered HTTP {status}")));
    }
    let last_modified = response
        .headers()
        .get("last-modified")
        .and_then(|v| v.to_str().ok())
        .and_then(parse_http_date);
    let bytes = response
        .body_mut()
        .with_config()
        .limit(rw_obs::net::MAX_RESPONSE_BYTES)
        .read_to_vec()
        .map_err(|e| err(format!("reading {url} failed: {e}")))?;
    if bytes.is_empty() {
        return Err(err(format!("{url} answered with an empty body")));
    }
    Ok((bytes, last_modified))
}

fn fetch_one(client: &ureq::Agent, url: &str, path: &Path) -> Result<FetchedFile, Box<dyn Error>> {
    let t = std::time::Instant::now();
    let (bytes, last_modified) = get_with_last_modified(client, url)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| err(format!("cannot create {}: {e}", parent.display())))?;
    }
    std::fs::write(path, &bytes).map_err(|e| err(format!("cannot write {}: {e}", path.display())))?;
    Ok(FetchedFile {
        name: path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string(),
        url: url.to_string(),
        path: rw_obs::absolute_uri(path),
        bytes: bytes.len(),
        sha256: hex_sha256(&bytes),
        last_modified: last_modified.map(seam_time),
        fetched_at: seam_time(Utc::now()),
        wall_s: t.elapsed().as_secs_f64(),
    })
}

/// The `.txt` files a realtime2 directory index names.
fn parse_index(html: &str) -> Vec<String> {
    let mut out = std::collections::BTreeSet::new();
    let mut rest = html;
    while let Some(at) = rest.find("href=\"") {
        rest = &rest[at + 6..];
        let Some(end) = rest.find('"') else { break };
        let name = &rest[..end];
        rest = &rest[end..];
        if name.ends_with(".txt") && !name.contains('/') && name.len() > 4 {
            let stem = &name[..name.len() - 4];
            if stem.bytes().all(|b| b.is_ascii_alphanumeric()) {
                out.insert(name.to_string());
            }
        }
    }
    out.into_iter().collect()
}

fn cmd_fetch(options: &Options) -> Result<String, Box<dyn Error>> {
    let out_dir = options.out.as_deref().ok_or_else(|| err("--out DIR is required"))?;
    std::fs::create_dir_all(out_dir.join("realtime2"))
        .map_err(|e| err(format!("cannot create {}: {e}", out_dir.display())))?;
    let archive = options.archive().to_string();
    let client = agent();
    let started = std::time::Instant::now();
    let station_table = fetch_one(
        &client,
        &format!("{archive}/stations/station_table.txt"),
        &out_dir.join("station_table.txt"),
    )?;
    let names: Vec<String> = match &options.stations {
        Some(list) => list.iter().map(|s| format!("{s}.txt")).collect(),
        None => {
            let index = rw_obs::net::get_text(&client, &format!("{archive}/realtime2/"), "NDBC realtime2 index")?;
            let names = parse_index(&index);
            if names.is_empty() {
                return Err(err("the realtime2 index names no platform file; the layout changed"));
            }
            names
        }
    };
    let names: Vec<String> = match options.limit {
        Some(n) => names.into_iter().take(n).collect(),
        None => names,
    };
    let pause = std::time::Duration::from_millis(options.request_pause_ms.unwrap_or(DEFAULT_REQUEST_PAUSE_MS));
    let mut files = Vec::with_capacity(names.len());
    let mut missing: Vec<String> = Vec::new();
    for (n, name) in names.iter().enumerate() {
        if n > 0 && !pause.is_zero() {
            std::thread::sleep(pause);
        }
        match fetch_one(&client, &format!("{archive}/realtime2/{name}"), &out_dir.join("realtime2").join(name)) {
            Ok(file) => files.push(file),
            // A platform in the station table with no realtime2 file (adrift,
            // retired, or out of the 45-day window) answers 404; it is a
            // named absence in the record, not a failed run.
            Err(e) if e.to_string().contains("404") => missing.push(name.clone()),
            Err(e) => return Err(e),
        }
    }
    let record = FetchRecord {
        schema: FETCH_SCHEMA.to_string(),
        status: "READY".to_string(),
        archive,
        out_dir: rw_obs::absolute_uri(out_dir),
        total_bytes: station_table.bytes + files.iter().map(|f| f.bytes).sum::<usize>(),
        station_table,
        files,
        missing,
        wall_s: started.elapsed().as_secs_f64(),
    };
    let text = format!("{}\n", serde_json::to_string_pretty(&record)?);
    std::fs::write(out_dir.join("fetch.json"), &text).map_err(|e| err(format!("cannot write the fetch record: {e}")))?;
    Ok(text)
}

// ---------------------------------------------------------------- decode

#[derive(Debug, Clone, PartialEq)]
struct Platform {
    id: String,
    kind: String,
    latitude_deg: f64,
    longitude_deg: f64,
    fixed: bool,
}

/// `12.000 N 23.000 W (12&#176;0'0" N 23&#176;0'0" W)` -> (lat, lon).
fn parse_location(text: &str) -> Option<(f64, f64)> {
    let head = text.split('(').next()?.trim();
    let parts: Vec<&str> = head.split_whitespace().collect();
    if parts.len() < 4 {
        return None;
    }
    let lat: f64 = parts[0].parse().ok()?;
    let lon: f64 = parts[2].parse().ok()?;
    let lat = match parts[1] {
        "N" => lat,
        "S" => -lat,
        _ => return None,
    };
    let lon = match parts[3] {
        "E" => lon,
        "W" => -lon,
        _ => return None,
    };
    (lat.abs() <= 90.0 && lon.abs() <= 180.0).then_some((lat, lon))
}

fn is_fixed_platform(kind: &str) -> bool {
    let k = kind.to_ascii_lowercase();
    k.contains("c-man") || k.contains("weather station") || k.contains("tower") || k.contains("platform") || k.contains("oil")
}

/// `station_table.txt`: `# STATION_ID | OWNER | TTYPE | HULL | NAME | PAYLOAD |
/// LOCATION | TIMEZONE | FORECAST | NOTE`, pipe-separated.
fn parse_station_table(text: &str) -> BTreeMap<String, Platform> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('|').map(str::trim).collect();
        if cols.len() < 7 {
            continue;
        }
        let Some((lat, lon)) = parse_location(cols[6]) else { continue };
        let id = cols[0].to_ascii_uppercase();
        out.insert(
            id.clone(),
            Platform {
                id,
                kind: cols[2].to_string(),
                latitude_deg: lat,
                longitude_deg: lon,
                fixed: is_fixed_platform(cols[2]),
            },
        );
    }
    out
}

#[derive(Debug, Clone)]
struct Report {
    time: DateTime<Utc>,
    wdir: Option<f64>,
    wspd: Option<f64>,
    pres_hpa: Option<f64>,
    atmp_c: Option<f64>,
    dewp_c: Option<f64>,
}

fn field(cols: &[&str], i: usize) -> Option<f64> {
    let v = cols.get(i)?;
    if *v == "MM" {
        return None;
    }
    v.parse::<f64>().ok().filter(|x| x.is_finite())
}

/// One realtime2 standard-met file.  Columns are resolved by the header
/// line's names, not by position, so a file with an extra column still
/// reads.
fn parse_realtime(text: &str, counters: &mut Counters) -> Vec<Report> {
    let mut lines = text.lines();
    let Some(header) = lines.next() else { return Vec::new() };
    let names: Vec<&str> = header.trim_start_matches('#').split_whitespace().collect();
    let at = |n: &str| names.iter().position(|x| *x == n);
    let (Some(iy), Some(im), Some(id), Some(ih), Some(imin)) = (at("YY"), at("MM"), at("DD"), at("hh"), at("mm")) else {
        counters.files_without_standard_header += 1;
        return Vec::new();
    };
    let (iwd, iws, ip, it, itd) = (at("WDIR"), at("WSPD"), at("PRES"), at("ATMP"), at("DEWP"));
    let mut out = Vec::new();
    for line in lines {
        if line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        counters.reports_read += 1;
        let (Some(y), Some(m), Some(d), Some(h), Some(mi)) = (
            cols.get(iy).and_then(|v| v.parse::<i32>().ok()),
            cols.get(im).and_then(|v| v.parse::<u32>().ok()),
            cols.get(id).and_then(|v| v.parse::<u32>().ok()),
            cols.get(ih).and_then(|v| v.parse::<u32>().ok()),
            cols.get(imin).and_then(|v| v.parse::<u32>().ok()),
        ) else {
            counters.reports_malformed += 1;
            continue;
        };
        let Some(time) = NaiveDate::from_ymd_opt(y, m, d).and_then(|dt| dt.and_hms_opt(h, mi, 0)) else {
            counters.reports_malformed += 1;
            continue;
        };
        out.push(Report {
            time: Utc.from_utc_datetime(&time),
            wdir: iwd.and_then(|i| field(&cols, i)),
            wspd: iws.and_then(|i| field(&cols, i)),
            pres_hpa: ip.and_then(|i| field(&cols, i)),
            atmp_c: it.and_then(|i| field(&cols, i)),
            dewp_c: itd.and_then(|i| field(&cols, i)),
        });
    }
    out
}

/// Neutral log-law factor from `from_m` to 10 m over the sea.
fn anemometer_factor(from_m: f64) -> f64 {
    (10.0 / SEA_ROUGHNESS_M).ln() / (from_m / SEA_ROUGHNESS_M).ln()
}

#[derive(Debug, Default, Serialize, Clone)]
struct Counters {
    files_read: usize,
    files_without_standard_header: usize,
    platforms_in_table: usize,
    platforms_with_files: usize,
    platforms_not_in_table: usize,
    platforms_fixed: usize,
    platforms_buoy: usize,
    reports_read: usize,
    reports_malformed: usize,
    reports_in_window: usize,
    reports_not_nearest_hour: usize,
    reports_kept: usize,
    values_pressure_out_of_range: usize,
    values_temperature_out_of_range: usize,
    values_dewpoint_out_of_range: usize,
    values_dewpoint_above_temperature: usize,
    values_wind_direction_out_of_range: usize,
    values_wind_speed_out_of_range: usize,
    values_underivable: usize,
    rows_by_variable: BTreeMap<String, usize>,
    reports_by_hour: BTreeMap<String, usize>,
}

fn rows_for(
    platform: &Platform,
    report: &Report,
    provenance: &RowProvenance,
    counters: &mut Counters,
    writer: &mut TableWriter,
) {
    let base = TableRow {
        source: SOURCE.to_string(),
        station_id: platform.id.clone(),
        latitude_deg: platform.latitude_deg,
        longitude_deg: platform.longitude_deg,
        elevation_m: 0.0,
        level_pa: None,
        valid_time: report.time,
        variable: String::new(),
        value: 0.0,
        error: 0.0,
        provenance: provenance.clone(),
    };
    match report.pres_hpa {
        Some(p) => {
            let pa = p * 100.0;
            if GROSS_SURFACE_PRESSURE_PA.0 <= pa && pa <= GROSS_SURFACE_PRESSURE_PA.1 {
                let mut r = base.clone();
                r.variable = VAR_SURFACE_PRESSURE.to_string();
                r.value = pa;
                r.error = ERROR_SURFACE_PRESSURE_PA;
                r.provenance = provenance.measuring(MEAS_SEA_LEVEL_PRESSURE);
                writer.push(r, &mut counters.rows_by_variable);
            } else {
                counters.values_pressure_out_of_range += 1;
            }
        }
        None => counters.values_underivable += 1,
    }
    let t_k = report.atmp_c.map(|t| t + 273.15);
    match t_k {
        Some(t) if GROSS_TEMPERATURE_K.0 <= t && t <= GROSS_TEMPERATURE_K.1 => {
            let mut r = base.clone();
            r.variable = VAR_TEMPERATURE.to_string();
            r.value = t;
            r.error = ERROR_TEMPERATURE_SURFACE_K;
            r.provenance = provenance.measuring(MEAS_PLATFORM_TEMPERATURE);
            writer.push(r, &mut counters.rows_by_variable);
        }
        Some(_) => counters.values_temperature_out_of_range += 1,
        None => counters.values_underivable += 1,
    }
    match report.dewp_c.map(|t| t + 273.15) {
        Some(td) if !(GROSS_DEWPOINT_K.0 <= td && td <= GROSS_DEWPOINT_K.1) => {
            counters.values_dewpoint_out_of_range += 1
        }
        Some(td) => {
            if matches!(t_k, Some(t) if td > t + 0.05) {
                counters.values_dewpoint_above_temperature += 1;
            } else {
                let mut r = base.clone();
                r.variable = VAR_DEWPOINT.to_string();
                r.value = td;
                r.error = ERROR_DEWPOINT_BUOY_K;
                r.provenance = provenance.measuring(MEAS_PLATFORM_DEWPOINT);
                writer.push(r, &mut counters.rows_by_variable);
            }
        }
        None => counters.values_underivable += 1,
    }
    match (report.wdir, report.wspd) {
        (Some(dir), Some(spd)) => {
            if !(0.0..=360.0).contains(&dir) {
                counters.values_wind_direction_out_of_range += 1;
            } else if !(GROSS_WIND_M_S.0..=GROSS_WIND_M_S.1).contains(&spd) {
                counters.values_wind_speed_out_of_range += 1;
            } else {
                let (speed, error, measurement) = if platform.fixed {
                    (spd, ERROR_WIND_SURFACE_M_S, MEAS_ANEMOMETER_WIND_10M)
                } else {
                    (spd * anemometer_factor(BUOY_ANEMOMETER_M), ERROR_WIND_BUOY_M_S, MEAS_ANEMOMETER_WIND_5M_TO_10M)
                };
                let (u, v) = wind_components(dir, speed);
                let mut ru = base.clone();
                ru.variable = VAR_WIND_U.to_string();
                ru.value = u;
                ru.error = error;
                ru.provenance = provenance.measuring(measurement);
                writer.push(ru, &mut counters.rows_by_variable);
                let mut rv = base.clone();
                rv.variable = VAR_WIND_V.to_string();
                rv.value = v;
                rv.error = error;
                rv.provenance = provenance.measuring(measurement);
                writer.push(rv, &mut counters.rows_by_variable);
            }
        }
        _ => counters.values_underivable += 1,
    }
}

/// The report nearest each top of the hour within `match_minutes`, one per
/// hour, in the window.
fn nearest_hourly(reports: &[Report], start: DateTime<Utc>, end: DateTime<Utc>, match_minutes: i64, counters: &mut Counters) -> Vec<Report> {
    let mut best: BTreeMap<i64, (i64, Report)> = BTreeMap::new();
    for report in reports {
        if report.time < start - Duration::minutes(match_minutes) || report.time > end + Duration::minutes(match_minutes) {
            continue;
        }
        counters.reports_in_window += 1;
        let secs = report.time.timestamp();
        let hour = ((secs as f64) / 3600.0).round() as i64;
        let distance = (secs - hour * 3600).abs();
        if distance > match_minutes * 60 {
            counters.reports_not_nearest_hour += 1;
            continue;
        }
        let top = Utc.timestamp_opt(hour * 3600, 0).single();
        let Some(top) = top else { continue };
        if top < start || top > end {
            continue;
        }
        match best.get(&hour) {
            Some((held, _)) if *held <= distance => counters.reports_not_nearest_hour += 1,
            _ => {
                if let Some((_, displaced)) = best.insert(hour, (distance, report.clone())) {
                    let _ = displaced;
                    counters.reports_not_nearest_hour += 1;
                }
            }
        }
    }
    best.into_values().map(|(_, r)| r).collect()
}

#[derive(Serialize)]
struct SourceFile {
    path: String,
    bytes: usize,
    sha256: String,
    platform: String,
    reports_kept: usize,
    /// The file's `Last-Modified` from the fetch record minus its latest
    /// report time: how far behind real time the feed publishes.
    behind_latest_report_s: Option<i64>,
}

#[derive(Serialize)]
struct TableRecord {
    schema: &'static str,
    status: &'static str,
    source: &'static str,
    table_schema: &'static str,
    path: String,
    sha256: String,
    rows: usize,
    bytes: usize,
    window: WindowRecord,
    match_minutes: i64,
    anemometer_reduction: AnemometerRecord,
    errors: BTreeMap<&'static str, f64>,
    counters: Counters,
    files: Vec<SourceFile>,
    latency_behind_real_time_s: Option<i64>,
    latency_basis: &'static str,
}

#[derive(Serialize)]
struct WindowRecord {
    start: String,
    end: String,
}

#[derive(Serialize)]
struct AnemometerRecord {
    buoy_height_m: f64,
    fixed_height_m: f64,
    sea_roughness_m: f64,
    buoy_factor_to_10m: f64,
    rule: &'static str,
}

fn cmd_table(options: &Options) -> Result<String, Box<dyn Error>> {
    let dir = options.dir.as_deref().ok_or_else(|| err("--dir DIR is required (the directory `fetch` wrote)"))?;
    let out = options.out.as_deref().ok_or_else(|| err("--out FILE.csv is required"))?;
    if out.is_dir() {
        return Err(err(format!("--out {} is a directory; give the CSV path", out.display())));
    }
    let (start, end) = options.window()?;
    let match_minutes = options.match_minutes.unwrap_or(DEFAULT_MATCH_MINUTES);
    let table_text = std::fs::read_to_string(dir.join("station_table.txt"))
        .map_err(|e| err(format!("cannot read {}: {e}; `rw_ndbc fetch` writes it", dir.join("station_table.txt").display())))?;
    let platforms = parse_station_table(&table_text);
    if platforms.is_empty() {
        return Err(err("station_table.txt names no platform with a position; the layout changed"));
    }
    let (last_modified, fetched_at): (BTreeMap<String, DateTime<Utc>>, BTreeMap<String, DateTime<Utc>>) =
        match &options.fetch_record {
            Some(path) => {
                let text = std::fs::read_to_string(path).map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
                let record: FetchRecord = serde_json::from_str(&text).map_err(|e| err(format!("{} is not a fetch record: {e}", path.display())))?;
                if record.schema != FETCH_SCHEMA {
                    return Err(err(format!("{} declares schema {:?}, expected {FETCH_SCHEMA:?}", path.display(), record.schema)));
                }
                let modified = record
                    .files
                    .iter()
                    .filter_map(|f| f.last_modified.as_deref().and_then(|t| parse_time(&format!("{t}Z")).ok()).map(|t| (f.name.clone(), t)))
                    .collect();
                let received = record
                    .files
                    .iter()
                    .filter_map(|f| parse_time(&format!("{}Z", f.fetched_at)).ok().map(|t| (f.name.clone(), t)))
                    .collect();
                (modified, received)
            }
            None => (BTreeMap::new(), BTreeMap::new()),
        };
    let realtime = dir.join("realtime2");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&realtime)
        .map_err(|e| err(format!("cannot list {}: {e}", realtime.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().and_then(|x| x.to_str()) == Some("txt"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(err(format!("{} holds no platform file", realtime.display())));
    }
    let mut counters = Counters {
        platforms_in_table: platforms.len(),
        ..Default::default()
    };
    let mut writer = TableWriter::new();
    let mut files = Vec::new();
    for path in &paths {
        let bytes = std::fs::read(path).map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        let id = name.trim_end_matches(".txt").to_ascii_uppercase();
        counters.files_read += 1;
        let Some(platform) = platforms.get(&id) else {
            counters.platforms_not_in_table += 1;
            continue;
        };
        let text = String::from_utf8_lossy(&bytes);
        let sha = hex_sha256(&bytes);
        let provenance = RowProvenance::of_source(&sha, last_modified.get(&name).copied(), fetched_at.get(&name).copied());
        let reports = parse_realtime(&text, &mut counters);
        let latest = reports.iter().map(|r| r.time).max();
        let kept = nearest_hourly(&reports, start, end, match_minutes, &mut counters);
        if kept.is_empty() {
            continue;
        }
        counters.platforms_with_files += 1;
        if platform.fixed {
            counters.platforms_fixed += 1;
        } else {
            counters.platforms_buoy += 1;
        }
        for report in &kept {
            counters.reports_kept += 1;
            *counters.reports_by_hour.entry(seam_time(report.time.with_minute(0).and_then(|t| t.with_second(0)).unwrap_or(report.time))).or_insert(0) += 1;
            rows_for(platform, report, &provenance, &mut counters, &mut writer);
        }
        files.push(SourceFile {
            path: rw_obs::absolute_uri(path),
            bytes: bytes.len(),
            sha256: sha,
            platform: id.clone(),
            reports_kept: kept.len(),
            behind_latest_report_s: match (latest, last_modified.get(&name)) {
                (Some(l), Some(m)) => Some((*m - l).num_seconds()),
                _ => None,
            },
        });
    }
    let (rows, csv_sha, csv_bytes) = writer.write(out)?;
    let mut errors = BTreeMap::new();
    errors.insert(VAR_SURFACE_PRESSURE, ERROR_SURFACE_PRESSURE_PA);
    errors.insert(VAR_TEMPERATURE, ERROR_TEMPERATURE_SURFACE_K);
    errors.insert(VAR_DEWPOINT, ERROR_DEWPOINT_BUOY_K);
    errors.insert("wind_buoy_m_s", ERROR_WIND_BUOY_M_S);
    errors.insert("wind_fixed_m_s", ERROR_WIND_SURFACE_M_S);
    let record = TableRecord {
        schema: TABLE_RECORD_SCHEMA,
        status: if rows > 0 { "READY" } else { "EMPTY" },
        source: SOURCE,
        table_schema: TABLE_SCHEMA,
        path: rw_obs::absolute_uri(out),
        sha256: csv_sha,
        rows,
        bytes: csv_bytes,
        window: WindowRecord { start: seam_time(start), end: seam_time(end) },
        match_minutes,
        anemometer_reduction: AnemometerRecord {
            buoy_height_m: BUOY_ANEMOMETER_M,
            fixed_height_m: FIXED_ANEMOMETER_M,
            sea_roughness_m: SEA_ROUGHNESS_M,
            buoy_factor_to_10m: anemometer_factor(BUOY_ANEMOMETER_M),
            rule: "buoy winds scaled to 10 m by the neutral log law over water; fixed platforms (C-MAN, \
                   weather stations, towers) used as reported",
        },
        errors,
        latency_behind_real_time_s: files.iter().filter_map(|f| f.behind_latest_report_s).max(),
        latency_basis: "file Last-Modified (fetch record) minus the latest report time in the file",
        counters,
        files,
    };
    let text = format!("{}\n", serde_json::to_string_pretty(&record)?);
    std::fs::write(out.with_extension("json"), &text).map_err(|e| err(format!("cannot write the table record: {e}")))?;
    Ok(text)
}

fn cmd_verify(options: &Options) -> Result<String, Box<dyn Error>> {
    let file = options.file.as_deref().ok_or_else(|| err("--file is required"))?;
    let record_path = options.record.as_deref().ok_or_else(|| err("--record is required"))?;
    let record: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(record_path).map_err(|e| err(format!("cannot read {}: {e}", record_path.display())))?,
    )
    .map_err(|e| err(format!("{} is not JSON: {e}", record_path.display())))?;
    if record.get("schema").and_then(|s| s.as_str()) != Some(TABLE_RECORD_SCHEMA) {
        return Err(err(format!("{} does not declare {TABLE_RECORD_SCHEMA}", record_path.display())));
    }
    let bytes = std::fs::read(file).map_err(|e| err(format!("cannot read {}: {e}", file.display())))?;
    let actual = hex_sha256(&bytes);
    let stated = record.get("sha256").and_then(|s| s.as_str()).unwrap_or("").to_string();
    #[derive(Serialize)]
    struct Record {
        schema: &'static str,
        status: &'static str,
        path: String,
        stated_sha256: String,
        actual_sha256: String,
    }
    let ok = actual == stated;
    let text = format!(
        "{}\n",
        serde_json::to_string_pretty(&Record {
            schema: VERIFY_SCHEMA,
            status: if ok { "VERIFIED" } else { "MISMATCH" },
            path: rw_obs::absolute_uri(file),
            stated_sha256: stated,
            actual_sha256: actual,
        })?
    );
    if ok { Ok(text) } else { Err(err(format!("table digest mismatch:\n{text}"))) }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLE: &str = "# STATION_ID | OWNER | TTYPE | HULL | NAME | PAYLOAD | LOCATION | TIMEZONE | FORECAST | NOTE\n\
#\n\
41001|NDBC|Buoy|3-meter foam buoy|EAST HATTERAS - 150 NM East of Cape HATTERAS|SCOOP payload|34.502 N 72.522 W (34&#176;30'6\" N 72&#176;31'19\" W)|E| |\n\
0y2w3|CG|Weather Station||Sturgeon Bay CG Station, WI||44.794 N 87.313 W (44&#176;47'39\" N 87&#176;18'48\" W)|C| |\n";

    const FEED: &str = "#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE\n\
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi  hPa    ft\n\
2026 09 01 00 50 310  3.0  5.0   2.5     8   6.7 211 1008.8  26.5  27.3  23.3   MM   MM    MM\n\
2026 09 01 00 00 300  4.0  4.0    MM    MM    MM  MM 1008.6  26.4  27.3  23.2   MM   MM    MM\n\
2026 09 01 00 10 300  MM   4.0    MM    MM    MM  MM 1008.6  26.4  27.3  23.2   MM   MM    MM\n\
2026 08 31 23 50 290  5.0  6.0    MM    MM    MM  MM 1008.4  26.3  27.2  27.9   MM   MM    MM\n";

    #[test]
    fn station_table_positions_decode() {
        let platforms = parse_station_table(TABLE);
        assert_eq!(platforms.len(), 2);
        let b = &platforms["41001"];
        assert!((b.latitude_deg - 34.502).abs() < 1e-9 && (b.longitude_deg + 72.522).abs() < 1e-9);
        assert!(!b.fixed);
        assert!(platforms["0Y2W3"].fixed);
    }

    #[test]
    fn hourly_selection_keeps_the_report_nearest_the_top_of_the_hour() {
        let mut c = Counters::default();
        let reports = parse_realtime(FEED, &mut c);
        assert_eq!(reports.len(), 4);
        let start = parse_time("2026-09-01T00:00:00Z").unwrap();
        let end = parse_time("2026-09-01T01:00:00Z").unwrap();
        let kept = nearest_hourly(&reports, start, end, 10, &mut c);
        // 00:00 wins the 00Z hour over 00:10 and 23:50; 00:50 rounds to 01Z (10 min away, kept).
        assert_eq!(kept.len(), 2, "{c:?}");
        assert_eq!(kept[0].time.format("%H:%M").to_string(), "00:00");
        assert_eq!(kept[1].time.format("%H:%M").to_string(), "00:50");
        let platforms = parse_station_table(TABLE);
        let mut w = TableWriter::new();
        rows_for(&platforms["41001"], &kept[0], &RowProvenance::default(), &mut c, &mut w);
        let rows = w.rows();
        assert_eq!(rows.len(), 5);
        let p = rows.iter().find(|r| r.variable == VAR_SURFACE_PRESSURE).unwrap();
        assert_eq!(p.provenance.measurement, MEAS_SEA_LEVEL_PRESSURE);
        let u = rows.iter().find(|r| r.variable == VAR_WIND_U).unwrap();
        assert_eq!(u.provenance.measurement, MEAS_ANEMOMETER_WIND_5M_TO_10M);
        // 300 deg at 4 m/s scaled by 1.064: u = -4*1.064*sin(300) = +3.686
        assert!((u.value - 4.0 * anemometer_factor(5.0) * (300.0f64.to_radians().sin()) * -1.0).abs() < 1e-9);
        assert_eq!(u.error, ERROR_WIND_BUOY_M_S);
        let p = rows.iter().find(|r| r.variable == VAR_SURFACE_PRESSURE).unwrap();
        assert!((p.value - 100_860.0).abs() < 1e-6);
    }

    #[test]
    fn dewpoint_above_temperature_is_counted_not_written() {
        let mut c = Counters::default();
        let reports = parse_realtime(FEED, &mut c);
        let platforms = parse_station_table(TABLE);
        let mut w = TableWriter::new();
        rows_for(&platforms["41001"], &reports[3], &RowProvenance::default(), &mut c, &mut w); // 23:50: DEWP 27.9 > ATMP 26.3
        assert_eq!(c.values_dewpoint_above_temperature, 1);
        assert!(w.rows().iter().all(|r| r.variable != VAR_DEWPOINT));
    }

    #[test]
    fn the_buoy_factor_is_the_log_law_from_five_to_ten_metres() {
        assert!((anemometer_factor(5.0) - 1.0640).abs() < 1e-3);
        assert!((anemometer_factor(10.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn abi_marker_names_the_contracts_it_pins() {
        assert!(ABI_MARKER.contains(FETCH_SCHEMA));
        assert!(ABI_MARKER.contains(TABLE_RECORD_SCHEMA));
        assert!(ABI_MARKER.contains(TABLE_SCHEMA));
    }
}
