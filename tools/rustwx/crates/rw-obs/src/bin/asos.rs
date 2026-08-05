//! `rw_asos` — ArWen's surface-observation front door.
//!
//! ASOS/AWOS routine and special METARs from the Iowa Environmental Mesonet
//! archive, frozen into a hash-pinned station table and an hourly-matched
//! report set in seam units.
//!
//! This is the headless retelling of the query pattern that already works in
//! Drew's Rust meteorology stack, with three deliberate departures, each of
//! which the battery needs:
//!
//! 1. **The query is bounded.** The GUI's fetch sends no `station=` and no
//!    `network=`, so it pulls every station on Earth for the window and
//!    filters locally. A battery that does that seven times over 24-hour
//!    windows is asking the archive for orders of magnitude more than it
//!    scores. Here the station list comes from a frozen table and goes into
//!    the request.
//! 2. **The station table is frozen and hashed, not read off each fetch.**
//!    Coordinates that arrive with the observations can change between the
//!    fetch that registered a case and the fetch that scores it, and the
//!    battery pins its station set at registration.
//! 3. **`mslp` and `p01i` are requested.** The GUI asks for
//!    `tmpf,dwpf,drct,sknt,gust,alti`; the spec reports MSLP as a
//!    diagnostic, and hourly precipitation is worth carrying while the
//!    request is being made anyway.
//!
//! Everything else is kept: the `asos.py` endpoint, `tz=Etc/UTC`,
//! `format=onlycomma`, `missing=empty`, `report_type=3&report_type=4`, the
//! unpadded date components, and resolving CSV columns by name.
//!
//! ```text
//! rw_asos stations --networks IA_ASOS,IL_ASOS --out stations.json
//! rw_asos fetch    --stations stations.json --start ... --end ... --out obs.csv
//! rw_asos decode   --obs obs.csv --stations stations.json --start ... --end ... --out obs.json
//! rw_asos verify   --file obs.json
//! ```

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use chrono::{DateTime, Datelike, NaiveDateTime, TimeZone, Timelike, Utc};
use serde::{Deserialize, Serialize};

use rw_nexrad::s3::parse_time;
use rw_obs::net::{agent, get_text, query_encode};
use rw_obs::seam::{seam_time, wrap_longitude, Provenance};
use rw_obs::{err, hex_sha256};

const VERSION: &str = env!("CARGO_PKG_VERSION");

const DEFAULT_ARCHIVE: &str = "https://mesonet.agron.iastate.edu";
/// The columns requested, in the order the request states them.
const DATA_COLUMNS: &[&str] = &[
    "tmpf", "dwpf", "drct", "sknt", "gust", "alti", "mslp", "p01i",
];

/// The registered gross-error screen, in the units the seam pins.
/// Transcribed from the battery spec: T2 in [-40, 55] C, dewpoint at or
/// below temperature, wind speed in [0, 75] m/s.
const TEMPERATURE_MIN_K: f64 = 233.15; // -40 C
const TEMPERATURE_MAX_K: f64 = 328.15; // +55 C
const WIND_MIN_MS: f64 = 0.0;
const WIND_MAX_MS: f64 = 75.0;
/// A station whose screen fires on more than this share of its reports is
/// dropped entirely rather than partly trusted.
const DEFAULT_MAX_SCREEN_FAILURE_RATE: f64 = 0.05;
/// Nearest report within this many seconds of a valid time is that hour's.
const DEFAULT_MATCH_SECONDS: i64 = 600;
/// A station reporting fewer than this share of the scored hours is dropped.
const DEFAULT_MIN_REPORT_RATE: f64 = 0.80;
/// How many stations one window request may name.
///
/// Measured against the archive 2026-08-04 while pulling the battery's own
/// case boxes: a frozen table of 642 stations answered 200, and 698 answered
/// **HTTP 414 URI Too Long**. A 1440 x 1200 km box over the dense Midwest or
/// Southeast freezes 575-802 stations, so the bounded query has to be split.
/// 400 sits well inside the measured ceiling and costs two or three requests
/// for a battery case rather than one.
const DEFAULT_STATIONS_PER_REQUEST: usize = 400;
/// How long to wait between the chunk requests of one window.
///
/// Also measured 2026-08-04: pulling seven case boxes back to back, three
/// chunks each, earned an **HTTP 429** on the twenty-first request. The
/// archive is free and is asking to be paced, so the pace is a default
/// rather than something an operator has to remember.
const DEFAULT_REQUEST_PAUSE_MS: u64 = 2000;

const STATIONS_SCHEMA: &str = "gpuwm-obs.asos-stations.v1";
const FETCH_SCHEMA: &str = "gpuwm-obs.asos-fetch.v1";
const SURFACE_SCHEMA: &str = "gpuwm-obs.asos-surface.v1";
const VERIFY_SCHEMA: &str = "gpuwm-obs.asos-verify.v1";

const ABI_MARKER: &str = "gpuwm-obs.asos-surface.v1\tstations\treports\tprovenance\t\
temperature_2m\tdewpoint_2m\twind_speed_10m\tmslp\tK\tm s-1\tPa";

const USAGE: &str = "\
usage: rw_asos <stations|fetch|decode|verify> [OPTIONS]
       rw_asos --version | --help | --abi

  stations  freeze a station table from the archive's own network metadata
  fetch     download one bounded CSV window and print its sha256
  decode    screen, convert to seam units, match to valid times, write a
            `gpuwm-obs.asos-surface.v1` record
  verify    re-hash a decoded record's source against the digest it carries

archive options
  --archive URL          default: https://mesonet.agron.iastate.edu
  --networks LIST        comma-separated IEM networks, e.g. IA_ASOS,IL_ASOS
  --bbox W,S,E,N         stations: keep only sites inside this lon/lat box
  --stations FILE        a frozen station table from `stations`
  --start TIME           window start
  --end TIME             window end (inclusive)
  --out PATH             the destination file

fetch options
  --stations-per-request N
                         how many stations one window request may name.
                         Default 400. Measured 2026-08-04: 642 stations
                         answered 200 and 698 answered HTTP 414, and a
                         battery-shaped box freezes 575-802, so the window
                         is fetched in chunks and the CSV bodies joined
  --request-pause-ms N   wait this long between chunk requests. Default 2000.
                         Seven case boxes pulled back to back earned an
                         HTTP 429 on the twenty-first request; the archive is
                         free and asks to be paced

decode options
  --obs FILE             the CSV from `fetch`
  --step-hours N         valid-time stride. Default 1
  --match-seconds N      nearest report within this of a valid time. Default 600
  --min-report-rate F    drop a station reporting fewer than this share of
                         scored hours. Default 0.80
  --max-screen-rate F    drop a station whose gross-error screen fires on more
                         than this share of its reports. Default 0.05

verify options
  --file FILE            the decoded record to re-prove
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(output) => {
            print!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("rw_asos: {error}");
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
        "--version" | "-V" => return Ok(format!("rw_asos {VERSION}\n")),
        "--abi" => return Ok(format!("{ABI_MARKER}\n")),
        _ => {}
    }
    let options = Options::parse(&args[1..])?;
    match first.as_str() {
        "stations" => cmd_stations(&options),
        "fetch" => cmd_fetch(&options),
        "decode" => cmd_decode(&options),
        "verify" => cmd_verify(&options),
        other => Err(err(format!("unknown subcommand {other:?}\n\n{USAGE}"))),
    }
}

#[derive(Debug, Default)]
struct Options {
    archive: Option<String>,
    networks: Option<String>,
    bbox: Option<[f64; 4]>,
    stations: Option<PathBuf>,
    obs: Option<PathBuf>,
    file: Option<PathBuf>,
    start: Option<String>,
    end: Option<String>,
    out: Option<PathBuf>,
    step_hours: Option<u32>,
    match_seconds: Option<i64>,
    min_report_rate: Option<f64>,
    max_screen_rate: Option<f64>,
    stations_per_request: Option<usize>,
    request_pause_ms: Option<u64>,
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
            let rate = |raw: String, flag: &str| -> Result<f64, Box<dyn Error>> {
                let parsed: f64 = raw
                    .parse()
                    .map_err(|_| err(format!("{flag} expects a fraction, got {raw:?}")))?;
                if !parsed.is_finite() || !(0.0..=1.0).contains(&parsed) {
                    return Err(err(format!("{flag} must lie in [0, 1], got {raw:?}")));
                }
                Ok(parsed)
            };
            match flag {
                "--archive" => options.archive = Some(value()?),
                "--networks" => options.networks = Some(value()?),
                "--bbox" => {
                    let raw = value()?;
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
                    if values[0] >= values[2] || values[1] >= values[3] {
                        return Err(err("--bbox must be W<E and S<N"));
                    }
                    options.bbox = Some(values);
                }
                "--stations" => options.stations = Some(PathBuf::from(value()?)),
                "--obs" => options.obs = Some(PathBuf::from(value()?)),
                "--file" => options.file = Some(PathBuf::from(value()?)),
                "--start" => options.start = Some(value()?),
                "--end" => options.end = Some(value()?),
                "--out" => options.out = Some(PathBuf::from(value()?)),
                "--step-hours" => {
                    let raw = value()?;
                    let hours: u32 = raw
                        .parse()
                        .map_err(|_| err(format!("--step-hours expects a count, got {raw:?}")))?;
                    if hours == 0 || hours > 24 {
                        return Err(err("--step-hours must lie in [1, 24]"));
                    }
                    options.step_hours = Some(hours);
                }
                "--match-seconds" => {
                    let raw = value()?;
                    let seconds: i64 = raw
                        .parse()
                        .map_err(|_| err(format!("--match-seconds expects a count, got {raw:?}")))?;
                    if seconds <= 0 {
                        return Err(err("--match-seconds must be positive"));
                    }
                    options.match_seconds = Some(seconds);
                }
                "--min-report-rate" => {
                    let raw = value()?;
                    options.min_report_rate = Some(rate(raw, "--min-report-rate")?)
                }
                "--max-screen-rate" => {
                    let raw = value()?;
                    options.max_screen_rate = Some(rate(raw, "--max-screen-rate")?)
                }
                "--stations-per-request" => {
                    let raw = value()?;
                    let count: usize = raw.parse().map_err(|_| {
                        err(format!("--stations-per-request expects a count, got {raw:?}"))
                    })?;
                    if count == 0 {
                        return Err(err("--stations-per-request must be positive"));
                    }
                    options.stations_per_request = Some(count);
                }
                "--request-pause-ms" => {
                    let raw = value()?;
                    options.request_pause_ms = Some(raw.parse().map_err(|_| {
                        err(format!("--request-pause-ms expects milliseconds, got {raw:?}"))
                    })?);
                }
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

    fn stations_per_request(&self) -> usize {
        self.stations_per_request
            .unwrap_or(DEFAULT_STATIONS_PER_REQUEST)
    }

    fn request_pause_ms(&self) -> u64 {
        self.request_pause_ms.unwrap_or(DEFAULT_REQUEST_PAUSE_MS)
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

    fn out(&self) -> Result<&Path, Box<dyn Error>> {
        let out = self.out.as_deref().ok_or_else(|| err("--out is required"))?;
        if out.is_dir() {
            return Err(err(format!(
                "--out {} is a directory; give the file path",
                out.display()
            )));
        }
        Ok(out)
    }
}

// -------------------------------------------------------------- stations

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Station {
    station_id: String,
    name: String,
    latitude: f64,
    longitude: f64,
    elevation_m: f64,
    network: String,
    state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StationTable {
    schema: String,
    status: String,
    archive: String,
    networks: Vec<String>,
    frozen_at: String,
    /// Digest over the station rows, so a table that changed between the
    /// registration that froze it and the scoring pass that read it is
    /// caught rather than silently scored.
    content_sha256: String,
    stations: Vec<Station>,
}

/// The digest a station table pins: the rows, canonically spelled, and
/// nothing about when or from where they were fetched.
///
/// Hashing the whole document would make the digest change every time the
/// table was re-frozen from the same archive state, which would make it
/// useless as an identity for the *station set*.
fn station_rows_digest(stations: &[Station]) -> String {
    let mut text = String::new();
    for station in stations {
        text.push_str(&format!(
            "{}\t{:.6}\t{:.6}\t{:.3}\n",
            station.station_id, station.latitude, station.longitude, station.elevation_m
        ));
    }
    hex_sha256(text.as_bytes())
}

fn cmd_stations(options: &Options) -> Result<String, Box<dyn Error>> {
    let networks_raw = options
        .networks
        .as_deref()
        .ok_or_else(|| err("--networks is required, e.g. --networks IA_ASOS,IL_ASOS"))?;
    let mut networks: Vec<String> = Vec::new();
    for token in networks_raw.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        if !token
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            return Err(err(format!(
                "network {token:?} may carry only letters, digits and '_'; it is concatenated \
                 into an archive URL"
            )));
        }
        let upper = token.to_ascii_uppercase();
        if !networks.contains(&upper) {
            networks.push(upper);
        }
    }
    if networks.is_empty() {
        return Err(err("--networks named no network"));
    }
    let out = options.out()?;
    let archive = options.archive().to_string();
    let agent = agent();

    let mut stations: Vec<Station> = Vec::new();
    for network in &networks {
        let url = format!("{archive}/geojson/network/{network}.geojson");
        let body = get_text(&agent, &url, "IEM network metadata")?;
        let document: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| err(format!("{network} metadata is not JSON: {e}")))?;
        let features = document
            .get("features")
            .and_then(|f| f.as_array())
            .ok_or_else(|| err(format!("{network} metadata carries no feature list")))?;
        for feature in features {
            let properties = feature.get("properties").unwrap_or(&serde_json::Value::Null);
            let coordinates = feature
                .get("geometry")
                .and_then(|g| g.get("coordinates"))
                .and_then(|c| c.as_array());
            let (Some(coordinates), Some(id)) = (
                coordinates,
                properties
                    .get("sid")
                    .and_then(|s| s.as_str())
                    .or_else(|| feature.get("id").and_then(|s| s.as_str())),
            ) else {
                continue;
            };
            if coordinates.len() < 2 {
                continue;
            }
            let (Some(lon), Some(lat)) = (coordinates[0].as_f64(), coordinates[1].as_f64()) else {
                continue;
            };
            let elevation = properties
                .get("elevation")
                .and_then(|e| e.as_f64())
                .unwrap_or(f64::NAN);
            if !lat.is_finite() || !lon.is_finite() || !elevation.is_finite() {
                // A station the battery cannot place, or cannot compare
                // against model terrain, is not a station the battery can
                // screen. Dropping it here is honest; carrying a NaN
                // elevation into the seam is not.
                continue;
            }
            if !(-90.0..=90.0).contains(&lat) {
                continue;
            }
            let lon = wrap_longitude(lon);
            if let Some([west, south, east, north]) = options.bbox {
                if lon < west || lon > east || lat < south || lat > north {
                    continue;
                }
            }
            let station = Station {
                station_id: id.to_ascii_uppercase(),
                name: properties
                    .get("sname")
                    .and_then(|s| s.as_str())
                    .unwrap_or("")
                    .to_string(),
                latitude: lat,
                longitude: lon,
                elevation_m: elevation,
                network: network.clone(),
                state: properties
                    .get("state")
                    .and_then(|s| s.as_str())
                    .unwrap_or("")
                    .to_string(),
            };
            if !stations.iter().any(|s| s.station_id == station.station_id) {
                stations.push(station);
            }
        }
    }
    if stations.is_empty() {
        return Err(err(
            "no station survived; a frozen table with no stations would score every arm on \
             nothing and report a clean zero",
        ));
    }
    stations.sort_by(|a, b| a.station_id.cmp(&b.station_id));

    let table = StationTable {
        schema: STATIONS_SCHEMA.to_string(),
        status: "READY".to_string(),
        archive,
        networks,
        frozen_at: seam_time(Utc::now()),
        content_sha256: station_rows_digest(&stations),
        stations,
    };
    let text = serde_json::to_string_pretty(&table)?;
    std::fs::write(out, format!("{text}\n"))
        .map_err(|e| err(format!("cannot write {}: {e}", out.display())))?;

    #[derive(Serialize)]
    struct Record<'a> {
        schema: &'static str,
        status: &'static str,
        path: String,
        networks: &'a [String],
        stations: usize,
        content_sha256: &'a str,
    }
    Ok(format!(
        "{}\n",
        serde_json::to_string_pretty(&Record {
            schema: STATIONS_SCHEMA,
            status: "READY",
            path: out.to_string_lossy().to_string(),
            networks: &table.networks,
            stations: table.stations.len(),
            content_sha256: &table.content_sha256,
        })?
    ))
}

fn read_station_table(path: &Path) -> Result<StationTable, Box<dyn Error>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| err(format!("cannot read station table {}: {e}", path.display())))?;
    let table: StationTable = serde_json::from_str(&text)
        .map_err(|e| err(format!("{} is not a station table: {e}", path.display())))?;
    if table.schema != STATIONS_SCHEMA {
        return Err(err(format!(
            "{} declares schema {:?}, expected {STATIONS_SCHEMA:?}",
            path.display(),
            table.schema
        )));
    }
    let digest = station_rows_digest(&table.stations);
    if digest != table.content_sha256 {
        return Err(err(format!(
            "station table {} has been edited since it was frozen: it states {}, its rows hash \
             to {digest}",
            path.display(),
            table.content_sha256
        )));
    }
    Ok(table)
}

// ----------------------------------------------------------------- fetch

/// The archive's historical range parameters: UTC, unpadded, exactly the
/// spelling the working implementation pins.
fn range_params(start: DateTime<Utc>, end: DateTime<Utc>) -> String {
    format!(
        "year1={}&month1={}&day1={}&hour1={}&minute1={}\
         &year2={}&month2={}&day2={}&hour2={}&minute2={}",
        start.year(),
        start.month(),
        start.day(),
        start.hour(),
        start.minute(),
        end.year(),
        end.month(),
        end.day(),
        end.hour(),
        end.minute()
    )
}

fn fetch_url(archive: &str, stations: &[Station], start: DateTime<Utc>, end: DateTime<Utc>) -> String {
    let mut url = format!("{archive}/cgi-bin/request/asos.py?");
    for station in stations {
        url.push_str("station=");
        url.push_str(&query_encode(&station.station_id));
        url.push('&');
    }
    for column in DATA_COLUMNS {
        url.push_str("data=");
        url.push_str(column);
        url.push('&');
    }
    url.push_str(&range_params(start, end));
    url.push_str(
        "&tz=Etc%2FUTC&format=onlycomma&latlon=yes&elev=yes&missing=empty&trace=T\
         &report_type=3&report_type=4",
    );
    url
}

fn cmd_fetch(options: &Options) -> Result<String, Box<dyn Error>> {
    let table = read_station_table(
        options
            .stations
            .as_deref()
            .ok_or_else(|| err("--stations is required (a table from `rw_asos stations`)"))?,
    )?;
    let (start, end) = options.window()?;
    let out = options.out()?;
    let chunk = options.stations_per_request();
    let fetched_at = seam_time(Utc::now());
    let client = agent();

    // The bounded query is bounded by a station list, and a battery-shaped
    // box freezes enough stations to make that list longer than the endpoint
    // will accept. So the window is fetched in station chunks and the CSV
    // bodies are concatenated; the request stays bounded, and the assembled
    // file is what the digest and the decoder see.
    let pause = std::time::Duration::from_millis(options.request_pause_ms());
    let mut urls: Vec<String> = Vec::new();
    let mut body = String::new();
    let mut rows = 0usize;
    for group in table.stations.chunks(chunk.max(1)) {
        if !urls.is_empty() && !pause.is_zero() {
            std::thread::sleep(pause);
        }
        let url = fetch_url(options.archive(), group, start, end);
        let text = get_text(&client, &url, "IEM ASOS window")?;
        let mut lines = text.lines().filter(|line| !line.trim().is_empty());
        let header = lines.next().unwrap_or("");
        if !header.starts_with("station,") {
            return Err(err(format!(
                "the archive did not answer with an ASOS CSV: its first line is {header:?}"
            )));
        }
        if body.is_empty() {
            body.push_str(header);
            body.push('\n');
        } else if body.lines().next() != Some(header) {
            return Err(err(format!(
                "the archive answered chunk {} with a different CSV header ({header:?}); \
                 concatenating columns that do not line up would shift every field",
                urls.len() + 1
            )));
        }
        for line in lines {
            body.push_str(line);
            body.push('\n');
            rows += 1;
        }
        urls.push(url);
    }
    if urls.is_empty() {
        return Err(err(
            "the frozen station table is empty; there is no bounded window to request",
        ));
    }
    std::fs::write(out, &body)
        .map_err(|e| err(format!("cannot write {}: {e}", out.display())))?;
    let sha256 = hex_sha256(body.as_bytes());

    #[derive(Serialize)]
    struct Record {
        schema: &'static str,
        status: &'static str,
        path: String,
        url: String,
        request_urls: Vec<String>,
        requests: usize,
        stations_per_request: usize,
        stations_requested: usize,
        start: String,
        end: String,
        rows: usize,
        bytes: usize,
        sha256: String,
        fetched_at: String,
    }
    Ok(format!(
        "{}\n",
        serde_json::to_string_pretty(&Record {
            schema: FETCH_SCHEMA,
            status: "READY",
            path: out.to_string_lossy().to_string(),
            url: urls[0].clone(),
            requests: urls.len(),
            request_urls: urls,
            stations_per_request: chunk,
            stations_requested: table.stations.len(),
            start: seam_time(start),
            end: seam_time(end),
            rows,
            bytes: body.len(),
            sha256,
            fetched_at,
        })?
    ))
}

// ---------------------------------------------------------------- decode

/// One parsed CSV row, still in the archive's own units.
#[derive(Debug, Clone)]
struct RawReport {
    station_id: String,
    valid: DateTime<Utc>,
    tmpf: Option<f64>,
    dwpf: Option<f64>,
    sknt: Option<f64>,
    mslp_hpa: Option<f64>,
}

fn split_csv(line: &str) -> Vec<String> {
    // The archive quotes fields that contain commas; a naive split would
    // shift every column after one.
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    for ch in line.chars() {
        match ch {
            '"' => quoted = !quoted,
            ',' if !quoted => fields.push(std::mem::take(&mut current)),
            _ => current.push(ch),
        }
    }
    fields.push(current);
    fields.into_iter().map(|f| f.trim().to_string()).collect()
}

fn parse_csv(text: &str) -> Result<Vec<RawReport>, Box<dyn Error>> {
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines.next().ok_or_else(|| err("the ASOS CSV is empty"))?;
    if !header.starts_with("station,") {
        return Err(err(format!(
            "the ASOS CSV header does not begin with 'station,': {header:?}"
        )));
    }
    let columns = split_csv(header);
    let index_of = |name: &str| columns.iter().position(|c| c == name);
    let station_at = index_of("station").ok_or_else(|| err("the ASOS CSV has no station column"))?;
    let valid_at = index_of("valid").ok_or_else(|| err("the ASOS CSV has no valid column"))?;
    let tmpf_at = index_of("tmpf");
    let dwpf_at = index_of("dwpf");
    let sknt_at = index_of("sknt");
    let mslp_at = index_of("mslp");

    let mut reports = Vec::new();
    for line in lines {
        let fields = split_csv(line);
        let get = |at: Option<usize>| -> Option<f64> {
            let value = fields.get(at?)?;
            if value.is_empty() {
                return None;
            }
            // `M` (missing) and `T` (trace) are not numbers and are not
            // errors; `missing=empty` means they should not appear, and
            // parsing them to None rather than failing keeps one archive
            // quirk from voiding a whole window.
            value.parse::<f64>().ok().filter(|v| v.is_finite())
        };
        let (Some(station), Some(valid)) = (fields.get(station_at), fields.get(valid_at)) else {
            continue;
        };
        if station.is_empty() {
            continue;
        }
        // `tz=Etc/UTC` is in the request, so the naive stamp is UTC. That is
        // the only reason reading it as UTC is sound, and it is why the
        // parameter is not optional in `fetch_url`.
        let parsed = NaiveDateTime::parse_from_str(valid, "%Y-%m-%d %H:%M")
            .or_else(|_| NaiveDateTime::parse_from_str(valid, "%Y-%m-%d %H:%M:%S"));
        let Ok(naive) = parsed else { continue };
        reports.push(RawReport {
            station_id: station.to_ascii_uppercase(),
            valid: Utc.from_utc_datetime(&naive),
            tmpf: get(tmpf_at),
            dwpf: get(dwpf_at),
            sknt: get(sknt_at),
            mslp_hpa: get(mslp_at),
        });
    }
    Ok(reports)
}

fn fahrenheit_to_kelvin(value: f64) -> f64 {
    (value - 32.0) * 5.0 / 9.0 + 273.15
}

fn knots_to_ms(value: f64) -> f64 {
    value * 0.5144444444444445
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SeamReport {
    station_id: String,
    valid_time: String,
    values: BTreeMap<String, f64>,
    flags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScreenReport {
    temperature_min_k: f64,
    temperature_max_k: f64,
    wind_min_ms: f64,
    wind_max_ms: f64,
    dewpoint_above_temperature_drops: usize,
    range_drops: usize,
    stations_dropped_by_screen: Vec<String>,
    stations_dropped_by_completeness: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SurfaceRecord {
    schema: String,
    status: String,
    provenance: Provenance,
    station_table_sha256: String,
    valid_times: Vec<String>,
    match_seconds: i64,
    min_report_rate: f64,
    max_screen_rate: f64,
    screen: ScreenReport,
    stations: Vec<Station>,
    reports: Vec<SeamReport>,
}

fn cmd_decode(options: &Options) -> Result<String, Box<dyn Error>> {
    let table = read_station_table(
        options
            .stations
            .as_deref()
            .ok_or_else(|| err("--stations is required"))?,
    )?;
    let obs_path = options
        .obs
        .as_deref()
        .ok_or_else(|| err("--obs is required (the CSV from `rw_asos fetch`)"))?;
    let out = options.out()?;
    let (start, end) = options.window()?;
    let step = options.step_hours.unwrap_or(1);
    let match_seconds = options.match_seconds.unwrap_or(DEFAULT_MATCH_SECONDS);
    let min_report_rate = options.min_report_rate.unwrap_or(DEFAULT_MIN_REPORT_RATE);
    let max_screen_rate = options.max_screen_rate.unwrap_or(DEFAULT_MAX_SCREEN_FAILURE_RATE);

    let text = std::fs::read_to_string(obs_path)
        .map_err(|e| err(format!("cannot read {}: {e}", obs_path.display())))?;
    let source_sha = hex_sha256(text.as_bytes());
    let raw = parse_csv(&text)?;

    let mut valid_times = Vec::new();
    let mut when = start;
    while when <= end {
        valid_times.push(when);
        when += chrono::Duration::hours(i64::from(step));
    }
    if valid_times.is_empty() {
        return Err(err("the window contains no valid time at this stride"));
    }

    let known: std::collections::BTreeSet<String> = table
        .stations
        .iter()
        .map(|s| s.station_id.clone())
        .collect();

    // Group by station, screening as we go.
    let mut by_station: BTreeMap<String, Vec<RawReport>> = BTreeMap::new();
    let mut screened: BTreeMap<String, (usize, usize)> = BTreeMap::new(); // (fired, seen)
    let mut dewpoint_drops = 0usize;
    let mut range_drops = 0usize;
    for report in raw {
        if !known.contains(&report.station_id) {
            // The archive answers with what it has; a station outside the
            // frozen table is not part of this case's set and is not scored
            // into it by accident.
            continue;
        }
        let entry = screened.entry(report.station_id.clone()).or_insert((0, 0));
        entry.1 += 1;
        let mut fired = false;
        if let Some(tmpf) = report.tmpf {
            let kelvin = fahrenheit_to_kelvin(tmpf);
            if !(TEMPERATURE_MIN_K..=TEMPERATURE_MAX_K).contains(&kelvin) {
                fired = true;
                range_drops += 1;
            }
        }
        if let (Some(tmpf), Some(dwpf)) = (report.tmpf, report.dwpf) {
            if dwpf > tmpf {
                fired = true;
                dewpoint_drops += 1;
            }
        }
        if let Some(sknt) = report.sknt {
            let ms = knots_to_ms(sknt);
            if !(WIND_MIN_MS..=WIND_MAX_MS).contains(&ms) {
                fired = true;
                range_drops += 1;
            }
        }
        if fired {
            entry.0 += 1;
            continue;
        }
        by_station.entry(report.station_id.clone()).or_default().push(report);
    }

    let mut dropped_by_screen = Vec::new();
    for (station, (fired, seen)) in &screened {
        if *seen > 0 && (*fired as f64 / *seen as f64) > max_screen_rate {
            dropped_by_screen.push(station.clone());
            by_station.remove(station);
        }
    }

    // Match each station to each valid time.
    let mut reports = Vec::new();
    let mut matched_hours: BTreeMap<String, usize> = BTreeMap::new();
    for (station, mut candidates) in by_station {
        candidates.sort_by_key(|r| r.valid);
        for target in &valid_times {
            let best = candidates
                .iter()
                .min_by_key(|r| (r.valid - *target).num_seconds().abs());
            let Some(best) = best else { continue };
            if (best.valid - *target).num_seconds().abs() > match_seconds {
                continue;
            }
            let mut values = BTreeMap::new();
            if let Some(tmpf) = best.tmpf {
                values.insert("temperature_2m".to_string(), fahrenheit_to_kelvin(tmpf));
            }
            if let Some(dwpf) = best.dwpf {
                values.insert("dewpoint_2m".to_string(), fahrenheit_to_kelvin(dwpf));
            }
            if let Some(sknt) = best.sknt {
                values.insert("wind_speed_10m".to_string(), knots_to_ms(sknt));
            }
            if let Some(hpa) = best.mslp_hpa {
                values.insert("mslp".to_string(), hpa * 100.0);
            }
            if values.is_empty() {
                // A report that carries no scored variable is not a report
                // for this purpose; emitting it would inflate the
                // completeness rate with empty rows.
                continue;
            }
            *matched_hours.entry(station.clone()).or_insert(0) += 1;
            reports.push(SeamReport {
                station_id: station.clone(),
                valid_time: seam_time(*target),
                values,
                flags: Vec::new(),
            });
        }
    }

    // Completeness: a station reporting too few of the scored hours is
    // dropped for every arm, symmetrically, so the pairing stays exact.
    let mut dropped_by_completeness = Vec::new();
    let required = (min_report_rate * valid_times.len() as f64).ceil() as usize;
    let mut keep: std::collections::BTreeSet<String> = Default::default();
    for station in &table.stations {
        let matched = matched_hours.get(&station.station_id).copied().unwrap_or(0);
        if matched >= required && matched > 0 {
            keep.insert(station.station_id.clone());
        } else if matched_hours.contains_key(&station.station_id) {
            dropped_by_completeness.push(station.station_id.clone());
        }
    }
    reports.retain(|r| keep.contains(&r.station_id));
    let stations: Vec<Station> = table
        .stations
        .iter()
        .filter(|s| keep.contains(&s.station_id))
        .cloned()
        .collect();
    if stations.is_empty() {
        // The overwhelmingly common cause is a decode window reaching past
        // the window the CSV was fetched for: every station then matches
        // only part of the hours and none clears the completeness bar. Say
        // so with the numbers rather than making the caller guess.
        let best = matched_hours.values().max().copied().unwrap_or(0);
        return Err(err(format!(
            "no station cleared the screens for {} .. {}: {} valid times were requested, the \
             best-covered station matched {best} of them, and {required} are required at \
             --min-report-rate {min_report_rate}. {} station(s) were dropped for completeness \
             and {} by the gross-error screen. If the CSV was fetched for a shorter window \
             than this one, refetch it before lowering the bar",
            seam_time(start),
            seam_time(end),
            valid_times.len(),
            dropped_by_completeness.len(),
            dropped_by_screen.len(),
        )));
    }
    reports.sort_by(|a, b| {
        (a.station_id.as_str(), a.valid_time.as_str())
            .cmp(&(b.station_id.as_str(), b.valid_time.as_str()))
    });

    let record = SurfaceRecord {
        schema: SURFACE_SCHEMA.to_string(),
        status: "READY".to_string(),
        provenance: Provenance::new(
            "asos",
            "iem-asos-metar",
            rw_obs::absolute_uri(obs_path),
            source_sha,
            seam_time(Utc::now()),
        ),
        station_table_sha256: table.content_sha256.clone(),
        valid_times: valid_times.iter().map(|t| seam_time(*t)).collect(),
        match_seconds,
        min_report_rate,
        max_screen_rate,
        screen: ScreenReport {
            temperature_min_k: TEMPERATURE_MIN_K,
            temperature_max_k: TEMPERATURE_MAX_K,
            wind_min_ms: WIND_MIN_MS,
            wind_max_ms: WIND_MAX_MS,
            dewpoint_above_temperature_drops: dewpoint_drops,
            range_drops,
            stations_dropped_by_screen: dropped_by_screen,
            stations_dropped_by_completeness: dropped_by_completeness,
        },
        stations,
        reports,
    };
    let text = serde_json::to_string_pretty(&record)?;
    std::fs::write(out, format!("{text}\n"))
        .map_err(|e| err(format!("cannot write {}: {e}", out.display())))?;

    #[derive(Serialize)]
    struct Summary<'a> {
        schema: &'static str,
        status: &'static str,
        path: String,
        valid_times: usize,
        stations: usize,
        reports: usize,
        provenance: &'a Provenance,
        station_table_sha256: &'a str,
        screen: &'a ScreenReport,
    }
    Ok(format!(
        "{}\n",
        serde_json::to_string_pretty(&Summary {
            schema: SURFACE_SCHEMA,
            status: "READY",
            path: out.to_string_lossy().to_string(),
            valid_times: record.valid_times.len(),
            stations: record.stations.len(),
            reports: record.reports.len(),
            provenance: &record.provenance,
            station_table_sha256: &record.station_table_sha256,
            screen: &record.screen,
        })?
    ))
}

fn cmd_verify(options: &Options) -> Result<String, Box<dyn Error>> {
    let path = options
        .file
        .as_deref()
        .or(options.out.as_deref())
        .ok_or_else(|| err("verify needs the record path in --file"))?;
    let text = std::fs::read_to_string(path)
        .map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
    let record: SurfaceRecord = serde_json::from_str(&text)
        .map_err(|e| err(format!("{} is not a surface record: {e}", path.display())))?;
    if record.schema != SURFACE_SCHEMA {
        return Err(err(format!(
            "{} declares schema {:?}, expected {SURFACE_SCHEMA:?}",
            path.display(),
            record.schema
        )));
    }
    let source = Path::new(&record.provenance.uri);
    let bytes = std::fs::read(source).map_err(|e| {
        err(format!(
            "cannot re-read the source this record was built from ({}): {e}",
            source.display()
        ))
    })?;
    let digest = hex_sha256(&bytes);
    if digest != record.provenance.sha256 {
        return Err(err(format!(
            "the source {} has changed since it was decoded: the record states {}, the bytes \
             hash to {digest}",
            source.display(),
            record.provenance.sha256
        )));
    }
    let known: std::collections::BTreeSet<&str> = record
        .stations
        .iter()
        .map(|s| s.station_id.as_str())
        .collect();
    if let Some(orphan) = record
        .reports
        .iter()
        .find(|r| !known.contains(r.station_id.as_str()))
    {
        return Err(err(format!(
            "the record carries a report for {}, which is not in its own station set",
            orphan.station_id
        )));
    }

    #[derive(Serialize)]
    struct Record {
        schema: &'static str,
        status: &'static str,
        path: String,
        source: String,
        source_sha256: String,
        stations: usize,
        reports: usize,
        valid_times: usize,
    }
    Ok(format!(
        "{}\n",
        serde_json::to_string_pretty(&Record {
            schema: VERIFY_SCHEMA,
            status: "PASS",
            path: path.to_string_lossy().to_string(),
            source: source.to_string_lossy().to_string(),
            source_sha256: digest,
            stations: record.stations.len(),
            reports: record.reports.len(),
            valid_times: record.valid_times.len(),
        })?
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_version_and_abi_are_stable_surfaces() {
        assert!(run(&[]).unwrap().contains("usage: rw_asos"));
        let help = run(&["--help".to_string()]).unwrap();
        for subcommand in ["stations", "fetch", "decode", "verify"] {
            assert!(help.contains(subcommand), "usage must document {subcommand}");
        }
        assert!(run(&["--version".to_string()]).unwrap().starts_with("rw_asos "));
        assert_eq!(run(&["--abi".to_string()]).unwrap(), format!("{ABI_MARKER}\n"));
    }

    #[test]
    fn unknown_subcommands_and_options_fail_closed() {
        assert!(run(&["sound".to_string()]).is_err());
        assert!(Options::parse(&["--nope".to_string()]).is_err());
        assert!(Options::parse(&["--min-report-rate".to_string(), "2".to_string()]).is_err());
        assert!(Options::parse(&["--step-hours".to_string(), "0".to_string()]).is_err());
    }

    #[test]
    fn the_range_parameters_are_unpadded_utc_exactly_as_the_archive_wants() {
        let start = parse_time("2024-05-21T20:30:00Z").unwrap();
        let end = parse_time("2024-05-21T22:30:00Z").unwrap();
        assert_eq!(
            range_params(start, end),
            "year1=2024&month1=5&day1=21&hour1=20&minute1=30\
             &year2=2024&month2=5&day2=21&hour2=22&minute2=30"
        );
    }

    #[test]
    fn the_query_is_bounded_by_station_and_carries_the_pinned_parameters() {
        let stations = vec![
            Station {
                station_id: "DSM".to_string(),
                name: String::new(),
                latitude: 41.53,
                longitude: -93.65,
                elevation_m: 294.0,
                network: "IA_ASOS".to_string(),
                state: "IA".to_string(),
            },
            Station {
                station_id: "DMX".to_string(),
                name: String::new(),
                latitude: 41.73,
                longitude: -93.72,
                elevation_m: 306.0,
                network: "IA_ASOS".to_string(),
                state: "IA".to_string(),
            },
        ];
        let url = fetch_url(
            DEFAULT_ARCHIVE,
            &stations,
            parse_time("2024-05-21T20:00:00Z").unwrap(),
            parse_time("2024-05-21T22:00:00Z").unwrap(),
        );
        assert!(url.contains("station=DSM"), "{url}");
        assert!(url.contains("station=DMX"), "{url}");
        for pinned in [
            "tz=Etc%2FUTC",
            "format=onlycomma",
            "missing=empty",
            "latlon=yes",
            "elev=yes",
            "report_type=3",
            "report_type=4",
            "data=tmpf",
            "data=mslp",
            "data=p01i",
        ] {
            assert!(url.contains(pinned), "the request must carry {pinned}: {url}");
        }
    }

    #[test]
    fn a_battery_shaped_station_list_is_split_into_requests_the_endpoint_accepts() {
        // Measured against the archive 2026-08-04: 642 stations answered 200
        // and 698 answered HTTP 414, so a table this size must go out as
        // several bounded requests rather than one long one.
        let stations: Vec<Station> = (0..802)
            .map(|index| Station {
                station_id: format!("S{index:03}"),
                name: String::new(),
                latitude: 41.0,
                longitude: -94.0,
                elevation_m: 300.0,
                network: "IA_ASOS".to_string(),
                state: "IA".to_string(),
            })
            .collect();
        let chunk = DEFAULT_STATIONS_PER_REQUEST;
        let groups: Vec<&[Station]> = stations.chunks(chunk).collect();
        assert_eq!(groups.len(), 3, "802 stations at {chunk} per request");
        let start = parse_time("2024-05-21T12:00:00Z").unwrap();
        let end = parse_time("2024-05-22T12:00:00Z").unwrap();
        let mut named = 0usize;
        for group in &groups {
            assert!(group.len() <= chunk);
            let url = fetch_url(DEFAULT_ARCHIVE, group, start, end);
            // Every chunk carries the whole pinned parameter set, not just
            // the first: a chunk fetched under different columns would
            // concatenate into a CSV whose rows do not line up.
            for pinned in ["tz=Etc%2FUTC", "format=onlycomma", "data=tmpf", "report_type=4"] {
                assert!(url.contains(pinned), "chunk request must carry {pinned}");
            }
            for station in group.iter() {
                assert!(url.contains(&format!("station={}", station.station_id)));
                named += 1;
            }
        }
        assert_eq!(named, stations.len(), "every frozen station is requested once");

        // The chunk size is an option, and a zero would loop forever rather
        // than fetch nothing, so it is refused at parse time.
        assert_eq!(Options::default().stations_per_request(), chunk);
        assert_eq!(
            Options::parse(&["--stations-per-request".to_string(), "120".to_string()])
                .unwrap()
                .stations_per_request(),
            120
        );
        for hostile in ["0", "-1", "many"] {
            assert!(
                Options::parse(&["--stations-per-request".to_string(), hostile.to_string()])
                    .is_err(),
                "{hostile:?} must be refused"
            );
        }
    }

    #[test]
    fn units_convert_to_the_ones_the_seam_pins() {
        // Freezing and boiling, and a knot that is exactly a knot.
        assert!((fahrenheit_to_kelvin(32.0) - 273.15).abs() < 1e-9);
        assert!((fahrenheit_to_kelvin(212.0) - 373.15).abs() < 1e-9);
        assert!((fahrenheit_to_kelvin(80.0) - 299.8166666666667).abs() < 1e-9);
        assert!((knots_to_ms(1.0) - 0.5144444444444445).abs() < 1e-12);
        assert!((knots_to_ms(0.0)).abs() < 1e-12);
        // The screen's bounds are the spec's, expressed in seam units.
        assert!((TEMPERATURE_MIN_K - fahrenheit_to_kelvin(-40.0)).abs() < 1e-9);
        assert!((TEMPERATURE_MAX_K - 328.15).abs() < 1e-9);
    }

    #[test]
    fn the_csv_splitter_survives_a_quoted_field_holding_commas() {
        assert_eq!(split_csv("a,b,c"), vec!["a", "b", "c"]);
        assert_eq!(
            split_csv("DSM,2024-05-21 20:54,\"METAR KDSM 21Z, AUTO\",80.00"),
            vec!["DSM", "2024-05-21 20:54", "METAR KDSM 21Z, AUTO", "80.00"]
        );
        assert_eq!(split_csv("a,,c"), vec!["a", "", "c"]);
    }

    #[test]
    fn columns_are_resolved_by_name_so_a_reordered_csv_still_reads() {
        let normal = "station,valid,tmpf,dwpf,sknt,mslp\n\
                      DSM,2024-05-21 20:54,80.00,68.00,17.00,993.70\n";
        let reordered = "station,mslp,valid,sknt,dwpf,tmpf\n\
                         DSM,993.70,2024-05-21 20:54,17.00,68.00,80.00\n";
        for text in [normal, reordered] {
            let rows = parse_csv(text).unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].station_id, "DSM");
            assert_eq!(rows[0].tmpf, Some(80.0));
            assert_eq!(rows[0].dwpf, Some(68.0));
            assert_eq!(rows[0].sknt, Some(17.0));
            assert_eq!(rows[0].mslp_hpa, Some(993.70));
            assert_eq!(seam_time(rows[0].valid), "2024-05-21T20:54:00");
        }
    }

    #[test]
    fn an_empty_field_is_missing_and_a_trace_token_is_not_a_number() {
        let text = "station,valid,tmpf,dwpf,sknt,mslp\n\
                    DSM,2024-05-21 21:11,80.00,67.00,13.00,\n\
                    DSM,2024-05-21 21:33,76.00,70.00,20.00,T\n";
        let rows = parse_csv(text).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].mslp_hpa, None, "an empty field is an absent value");
        assert_eq!(rows[1].mslp_hpa, None, "a trace token is not a pressure");
    }

    #[test]
    fn a_csv_that_is_not_one_is_refused_rather_than_read_as_zero_rows() {
        assert!(parse_csv("").is_err());
        assert!(parse_csv("<html>service unavailable</html>\n").is_err());
        assert!(parse_csv("valid,station\n2024-05-21 20:54,DSM\n").is_err());
    }

    #[test]
    fn a_station_table_digest_covers_the_rows_and_catches_an_edit() {
        let mut stations = vec![Station {
            station_id: "DSM".to_string(),
            name: "DES MOINES".to_string(),
            latitude: 41.5339,
            longitude: -93.6531,
            elevation_m: 294.0,
            network: "IA_ASOS".to_string(),
            state: "IA".to_string(),
        }];
        let before = station_rows_digest(&stations);
        // A cosmetic change does not move the digest...
        stations[0].name = "DES MOINES INTL".to_string();
        assert_eq!(station_rows_digest(&stations), before);
        // ...but moving a station does.
        stations[0].latitude = 41.6;
        assert_ne!(station_rows_digest(&stations), before);
    }

    #[test]
    fn every_command_requires_its_arguments() {
        assert!(cmd_stations(&Options::default()).is_err());
        assert!(cmd_fetch(&Options::default()).is_err());
        assert!(cmd_decode(&Options::default()).is_err());
        assert!(cmd_verify(&Options::default()).is_err());
    }

    #[test]
    fn a_network_name_that_would_reshape_the_url_is_refused() {
        for hostile in ["IA_ASOS/../x", "IA ASOS", "IA-ASOS", "../etc"] {
            let options = Options {
                networks: Some(hostile.to_string()),
                out: Some(PathBuf::from("x.json")),
                ..Options::default()
            };
            assert!(cmd_stations(&options).is_err(), "{hostile:?} must be refused");
        }
    }
}
