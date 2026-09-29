//! `rw_igra2` -- the radiosonde front door.
//!
//! The Integrated Global Radiosonde Archive, version 2, as NCEI serves it:
//! one zip per station under `access/data-y2d/` (the year-to-date files,
//! refreshed daily) or `access/data-por/` (the period of record), each
//! holding one fixed-width text file of every sounding the station
//! reported.  This bin lists that directory, downloads the files with a
//! SHA-256 and the server's `Last-Modified` per file, and decodes them into
//! the neutral observation table (`gpuwm-obs.table.v1`, the layout
//! `gpuwm.arwen_global.obs_table` reads without conversion): one row per
//! station, sounding, level and variable, in the analysis's own vocabulary
//! and units.
//!
//! What a row is.  A level with a pressure (LVLTYP1 1 or 2) becomes aloft
//! rows anchored at `level_pa`; the surface level (LVLTYP2 = 1) becomes a
//! `surface_pressure_pa` row anchored at its geopotential height, which is
//! the station elevation.  Temperature and dewpoint (temperature minus the
//! archive's dewpoint depression) are Kelvin, wind is `u`/`v` from the
//! archive's direction and tenths of m/s.  Every level carries ITS OWN
//! time: the header's release time (RELTIME, HHMM on the nominal date,
//! the previous day when it reads later than the nominal hour by more than
//! twelve hours) plus the level's elapsed time since release (ETIME,
//! MMMSS), so a 500 hPa level launched at 23:03 for the 00Z nominal is
//! compared with the model at 23:25 and not at midnight; a level without
//! an elapsed time, or a sounding without a release time, keeps the
//! nominal hour and is counted (`levels_at_nominal_time`).  The nominal
//! hour rides in the `nominal_time` column of every row.  The archive
//! carries no per-level position, so every level sits at the station.
//!
//! A level whose geopotential height is missing keeps its row (the aloft
//! operators read `level_pa`) and carries the ICAO standard-atmosphere
//! altitude of its pressure as `elevation_m`; the count of such rows is in
//! the record so a reader who needs measured heights can see how many it
//! does not have.
//!
//! Every screen is a count, never a repair: the archive's -9999 and -8888
//! sentinels leave a variable underivable; a pressure outside 1 to 1100
//! hPa, a temperature outside 170 to 340 K, a dewpoint above its
//! temperature, a wind direction outside 0 to 360 or a speed above 150 m/s
//! drops that value by name; a repeated pressure inside one sounding keeps
//! its first appearance.
//!
//! ```text
//! rw_igra2 list  [--archive URL] [--collection y2d|por] [--stations LIST]
//! rw_igra2 fetch --out DIR [--stations LIST] [--limit N] [--request-pause-ms N]
//! rw_igra2 table --zips DIR --start TIME --end TIME --out FILE.csv
//!                [--mandatory-only] [--fetch-record FILE]
//! rw_igra2 verify --file FILE.csv --record FILE.json
//! ```

use std::collections::BTreeMap;
use std::error::Error;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use chrono::{DateTime, Duration, NaiveDate, TimeZone, Utc};
use serde::{Deserialize, Serialize};

use rw_nexrad::s3::{parse_http_date, parse_time};
use rw_obs::net::agent;
use rw_obs::seam::seam_time;
use rw_obs::table::{
    isa_altitude_m, wind_components, RowProvenance, TableRow, TableWriter,
    ERROR_DEWPOINT_ALOFT_K, ERROR_SURFACE_PRESSURE_PA, ERROR_TEMPERATURE_ALOFT_K,
    ERROR_WIND_ALOFT_M_S, GROSS_TEMPERATURE_K, GROSS_WIND_M_S, MEAS_SONDE_LEVEL,
    MEAS_STATION_PRESSURE, TABLE_SCHEMA, VAR_DEWPOINT, VAR_SURFACE_PRESSURE, VAR_TEMPERATURE,
    VAR_WIND_U, VAR_WIND_V,
};
use rw_obs::{err, hex_sha256};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from, embedded so the gpuwm release cut can prove a
/// staged bridge matches the commit being released by reading bytes
/// alone.  `build.rs` injects the value; `main` references the constant so
/// the linker cannot discard it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

/// The station list beside the archive (id, position, ELEVATION in metres
/// per station): fetched with the files and read by `table` as the surface
/// level's anchor when the archive gives that level no geopotential height.
/// Seventy-two percent of the case's surface levels (1,221 of 1,690)
/// arrived without one.
const STATION_LIST_URL: &str =
    "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/doc/igra2-station-list.txt";
const STATION_LIST_FILE: &str = "igra2-station-list.txt";
const DEFAULT_ARCHIVE: &str =
    "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access";
/// Measured 2026-09-06 against the archive: 807 year-to-date files answered
/// eight parallel requests without a refusal; this bin fetches serially and
/// pauses between files because the archive is free and shared.
const DEFAULT_REQUEST_PAUSE_MS: u64 = 200;
const SOURCE: &str = "igra2";

const LIST_SCHEMA: &str = "gpuwm-obs.igra2-list.v1";
const FETCH_SCHEMA: &str = "gpuwm-obs.igra2-fetch.v1";
const TABLE_RECORD_SCHEMA: &str = "gpuwm-obs.igra2-table.v1";
const VERIFY_SCHEMA: &str = "gpuwm-obs.igra2-verify.v1";

/// The exact `--abi` line the Python front door pins: the fetch record,
/// the table record and the neutral table the rows land in.
const ABI_MARKER: &str = "gpuwm-obs.igra2-fetch.v1\tgpuwm-obs.igra2-table.v1\t\
gpuwm-obs.table.v2\tsurface_pressure_pa\ttemperature_k\tdewpoint_k\twind_u_m_s\twind_v_m_s\tlevel_times";

/// The mandatory levels, hPa, the `--mandatory-only` selection keeps.
const MANDATORY_HPA: &[f64] = &[
    1000.0, 925.0, 850.0, 700.0, 500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0, 70.0, 50.0,
    30.0, 20.0, 10.0,
];

const USAGE: &str = "\
usage: rw_igra2 <list|fetch|table|verify> [OPTIONS]
       rw_igra2 --version | --help | --abi

  list    read the archive directory and report the station files it serves
  fetch   download station files, and the station list beside them, with a
          sha256 and Last-Modified per file
  table   decode station files in a window into a `gpuwm-obs.table.v2` CSV
  verify  re-hash a table against the record written beside it

archive options (list, fetch)
  --archive URL          default: https://www.ncei.noaa.gov/data/
                         integrated-global-radiosonde-archive/access
  --collection NAME      y2d (year to date, default) or por (period of record)
  --stations LIST        comma-separated IGRA2 station ids to keep (default all)
  --limit N              fetch at most the first N files
  --request-pause-ms N   pause between files (default 200)
  --out DIR|FILE         fetch: the directory files land in; table: the CSV

table options
  --zips DIR             the directory `fetch` wrote (zip or unzipped txt)
  --start TIME           first nominal hour kept, e.g. 2026-08-31T12:00:00Z
  --end TIME             last nominal hour kept (inclusive)
  --mandatory-only       keep the sixteen mandatory levels only
  --fetch-record FILE    the `fetch` record, so the table record can state
                         how far behind real time each file arrived
  --station-list FILE    the IGRA2 station list (igra2-station-list.txt; `fetch`
                         writes it beside the files and `table` looks there by
                         default): the surface level's anchor when the archive
                         gives that level no geopotential height. Without an
                         anchor a surface level's rows are dropped and counted,
                         never stamped with the ISA height of their pressure

verify options
  --file FILE.csv        the table
  --record FILE.json     the record written beside it
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
            eprintln!("rw_igra2: {error}");
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
        "--version" | "-V" => return Ok(format!("rw_igra2 {VERSION}\n")),
        "--abi" => return Ok(format!("{ABI_MARKER}\n")),
        _ => {}
    }
    let options = Options::parse(&args[1..])?;
    match first.as_str() {
        "list" => cmd_list(&options),
        "fetch" => cmd_fetch(&options),
        "table" => cmd_table(&options),
        "verify" => cmd_verify(&options),
        other => Err(err(format!("unknown subcommand {other:?}\n\n{USAGE}"))),
    }
}

#[derive(Debug, Default)]
struct Options {
    archive: Option<String>,
    collection: Option<String>,
    stations: Option<Vec<String>>,
    limit: Option<usize>,
    request_pause_ms: Option<u64>,
    out: Option<PathBuf>,
    zips: Option<PathBuf>,
    start: Option<String>,
    end: Option<String>,
    mandatory_only: bool,
    fetch_record: Option<PathBuf>,
    station_list: Option<PathBuf>,
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
                "--collection" => options.collection = Some(value()?),
                "--stations" => {
                    let raw = value()?;
                    let list: Vec<String> = raw
                        .split(',')
                        .map(|s| s.trim().to_ascii_uppercase())
                        .filter(|s| !s.is_empty())
                        .collect();
                    if list.is_empty() {
                        return Err(err("--stations named no station"));
                    }
                    for id in &list {
                        if !id.bytes().all(|b| b.is_ascii_alphanumeric()) {
                            return Err(err(format!(
                                "station {id:?} may carry only letters and digits; it is \
                                 concatenated into an archive URL"
                            )));
                        }
                    }
                    options.stations = Some(list);
                }
                "--limit" => {
                    let raw = value()?;
                    let n: usize = raw
                        .parse()
                        .map_err(|_| err(format!("--limit expects a count, got {raw:?}")))?;
                    if n == 0 {
                        return Err(err("--limit must be positive"));
                    }
                    options.limit = Some(n);
                }
                "--request-pause-ms" => {
                    let raw = value()?;
                    options.request_pause_ms = Some(raw.parse().map_err(|_| {
                        err(format!("--request-pause-ms expects milliseconds, got {raw:?}"))
                    })?);
                }
                "--out" => options.out = Some(PathBuf::from(value()?)),
                "--zips" => options.zips = Some(PathBuf::from(value()?)),
                "--start" => options.start = Some(value()?),
                "--end" => options.end = Some(value()?),
                "--mandatory-only" => options.mandatory_only = true,
                "--fetch-record" => options.fetch_record = Some(PathBuf::from(value()?)),
                "--station-list" => options.station_list = Some(PathBuf::from(value()?)),
                "--file" => options.file = Some(PathBuf::from(value()?)),
                "--record" => options.record = Some(PathBuf::from(value()?)),
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

    fn collection(&self) -> Result<&str, Box<dyn Error>> {
        match self.collection.as_deref().unwrap_or("y2d") {
            "y2d" => Ok("data-y2d"),
            "por" => Ok("data-por"),
            other => Err(err(format!(
                "--collection {other:?} is not one of y2d, por"
            ))),
        }
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
}

// ------------------------------------------------------------------ list

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Listed {
    station_id: String,
    file: String,
    url: String,
}

fn station_of(file: &str) -> Option<&str> {
    // AEM00041217-data-beg2026.txt.zip, AEM00041217-data.txt.zip
    let stem = file.split('-').next()?;
    (stem.len() == 11 && stem.bytes().all(|b| b.is_ascii_alphanumeric())).then_some(stem)
}

/// The station files an NCEI directory index names, from its `href`
/// attributes; the index is HTML, so the anchors are all that is read.
fn parse_index(html: &str, base: &str) -> Vec<Listed> {
    let mut out = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    let mut rest = html;
    while let Some(at) = rest.find("href=\"") {
        rest = &rest[at + 6..];
        let Some(end) = rest.find('"') else { break };
        let name = &rest[..end];
        rest = &rest[end..];
        if !(name.ends_with(".txt.zip") || name.ends_with(".txt")) || name.contains('/') {
            continue;
        }
        let Some(station) = station_of(name) else { continue };
        if !seen.insert(name.to_string()) {
            continue;
        }
        out.push(Listed {
            station_id: station.to_string(),
            file: name.to_string(),
            url: format!("{base}/{name}"),
        });
    }
    out.sort_by(|a, b| a.file.cmp(&b.file));
    out
}

fn list_archive(options: &Options) -> Result<(String, Vec<Listed>), Box<dyn Error>> {
    let base = format!("{}/{}", options.archive(), options.collection()?);
    let client = agent();
    let body = rw_obs::net::get_text(&client, &format!("{base}/"), "IGRA2 directory index")?;
    let mut files = parse_index(&body, &base);
    if let Some(wanted) = &options.stations {
        files.retain(|f| wanted.iter().any(|w| w == &f.station_id));
        let missing: Vec<&String> = wanted
            .iter()
            .filter(|w| !files.iter().any(|f| &f.station_id == *w))
            .collect();
        if !missing.is_empty() {
            return Err(err(format!(
                "the archive index names no file for stations {missing:?}; an IGRA2 id is \
                 eleven characters (e.g. USM00072365) and the year-to-date collection holds \
                 only stations that reported this year"
            )));
        }
    }
    if files.is_empty() {
        return Err(err(format!(
            "the directory index at {base}/ names no station file; the archive layout changed \
             or the collection is empty"
        )));
    }
    Ok((base, files))
}

fn cmd_list(options: &Options) -> Result<String, Box<dyn Error>> {
    let (base, files) = list_archive(options)?;
    #[derive(Serialize)]
    struct Record {
        schema: &'static str,
        status: &'static str,
        archive: String,
        files: usize,
        stations: Vec<Listed>,
    }
    Ok(format!(
        "{}\n",
        serde_json::to_string_pretty(&Record {
            schema: LIST_SCHEMA,
            status: "READY",
            archive: base,
            files: files.len(),
            stations: files,
        })?
    ))
}

// ----------------------------------------------------------------- fetch

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FetchedFile {
    station_id: String,
    file: String,
    url: String,
    path: String,
    bytes: usize,
    sha256: String,
    /// The server's `Last-Modified` for the object (seam time), when it
    /// sent one: the archive's own statement of when the file was last
    /// rebuilt, which is what a latency reading is measured against.
    last_modified: Option<String>,
    fetched_at: String,
    /// Wall seconds of the GET.
    wall_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FetchRecord {
    schema: String,
    status: String,
    archive: String,
    out_dir: String,
    files: Vec<FetchedFile>,
    total_bytes: usize,
    wall_s: f64,
    /// The station list downloaded beside the files; absent in a record
    /// written before the surface anchor came from it (`table` then takes
    /// `--station-list`, or drops the height-less surface levels by count).
    #[serde(default)]
    station_list: Option<StationListFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StationListFile {
    url: String,
    path: String,
    bytes: usize,
    sha256: String,
    last_modified: Option<String>,
    fetched_at: String,
    stations: usize,
    stations_without_elevation: usize,
}

/// The IGRA2 station list (fixed width: ID 1-11, LAT 13-20, LON 22-30,
/// ELEV 32-37 in metres; -998.8 is the archive's missing elevation and
/// -999.9 its unknown one): station id to elevation, and the count of
/// stations the list gives no elevation.
fn parse_station_list(text: &str) -> (BTreeMap<String, f64>, usize) {
    let mut out = BTreeMap::new();
    let mut without = 0usize;
    for line in text.lines() {
        let (Some(id), Some(elev)) = (line.get(0..11), line.get(31..37)) else { continue };
        let id = id.trim();
        if id.len() != 11 || !id.bytes().all(|b| b.is_ascii_alphanumeric()) {
            continue;
        }
        match elev.trim().parse::<f64>() {
            Ok(e) if e > -998.0 && e < 9000.0 => {
                out.insert(id.to_string(), e);
            }
            _ => without += 1,
        }
    }
    (out, without)
}

fn get_with_last_modified(
    client: &ureq::Agent,
    url: &str,
) -> Result<(Vec<u8>, Option<DateTime<Utc>>), Box<dyn Error>> {
    let mut response = client
        .get(url)
        .call()
        .map_err(|e| err(format!("GET {url} failed: {e}")))?;
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

fn cmd_fetch(options: &Options) -> Result<String, Box<dyn Error>> {
    let out_dir = options
        .out
        .as_deref()
        .ok_or_else(|| err("--out DIR is required"))?;
    std::fs::create_dir_all(out_dir)
        .map_err(|e| err(format!("cannot create {}: {e}", out_dir.display())))?;
    let (base, mut files) = list_archive(options)?;
    if let Some(limit) = options.limit {
        files.truncate(limit);
    }
    let pause = std::time::Duration::from_millis(
        options.request_pause_ms.unwrap_or(DEFAULT_REQUEST_PAUSE_MS),
    );
    let client = agent();
    let started = std::time::Instant::now();
    let mut fetched = Vec::with_capacity(files.len());
    for (n, listed) in files.iter().enumerate() {
        if n > 0 && !pause.is_zero() {
            std::thread::sleep(pause);
        }
        let t = std::time::Instant::now();
        let (bytes, last_modified) = get_with_last_modified(&client, &listed.url)?;
        let path = out_dir.join(&listed.file);
        std::fs::write(&path, &bytes)
            .map_err(|e| err(format!("cannot write {}: {e}", path.display())))?;
        fetched.push(FetchedFile {
            station_id: listed.station_id.clone(),
            file: listed.file.clone(),
            url: listed.url.clone(),
            path: rw_obs::absolute_uri(&path),
            bytes: bytes.len(),
            sha256: hex_sha256(&bytes),
            last_modified: last_modified.map(seam_time),
            fetched_at: seam_time(Utc::now()),
            wall_s: t.elapsed().as_secs_f64(),
        });
    }
    // The station list, beside the files: the surface level's anchor when
    // the archive gives that level no height.  A fetch without it would
    // hand `table` a case whose height-less surface levels can only be
    // dropped, so its absence is an error that names the URL.
    let (list_bytes, list_modified) = get_with_last_modified(&client, STATION_LIST_URL)
        .map_err(|e| err(format!("the station list did not arrive ({e}); the surface levels need it")))?;
    let (elevations, without_elevation) = parse_station_list(&String::from_utf8_lossy(&list_bytes));
    if elevations.is_empty() {
        return Err(err(format!(
            "{STATION_LIST_URL} parsed to no station with an elevation; the layout this reader \
             knows is ID 1-11, LAT 13-20, LON 22-30, ELEV 32-37"
        )));
    }
    let list_path = out_dir.join(STATION_LIST_FILE);
    std::fs::write(&list_path, &list_bytes)
        .map_err(|e| err(format!("cannot write {}: {e}", list_path.display())))?;
    let station_list = Some(StationListFile {
        url: STATION_LIST_URL.to_string(),
        path: rw_obs::absolute_uri(&list_path),
        bytes: list_bytes.len(),
        sha256: hex_sha256(&list_bytes),
        last_modified: list_modified.map(seam_time),
        fetched_at: seam_time(Utc::now()),
        stations: elevations.len(),
        stations_without_elevation: without_elevation,
    });
    let record = FetchRecord {
        schema: FETCH_SCHEMA.to_string(),
        status: "READY".to_string(),
        archive: base,
        out_dir: rw_obs::absolute_uri(out_dir),
        total_bytes: fetched.iter().map(|f| f.bytes).sum::<usize>() + list_bytes.len(),
        files: fetched,
        wall_s: started.elapsed().as_secs_f64(),
        station_list,
    };
    let text = format!("{}\n", serde_json::to_string_pretty(&record)?);
    std::fs::write(out_dir.join("fetch.json"), &text)
        .map_err(|e| err(format!("cannot write the fetch record: {e}")))?;
    Ok(text)
}

// ------------------------------------------------------------------- zip

/// The members of a zip archive, by local file header, inflated.
///
/// A zip is walked from its first local header: signature `PK\x03\x04`,
/// then the fixed fields (method at 8, sizes at 18 and 22, name and extra
/// lengths at 26 and 28), then the name, the extra field and the data.  The
/// central directory at the end is not consulted: the station files NCEI
/// writes carry the sizes in the local header (no data descriptor), and a
/// reader that needs the directory is a reader that can be lied to by two
/// places at once.  A member the local header sizes at zero with the data
/// descriptor flag set is refused by name rather than guessed at.  Only
/// `stored` (0) and `deflate` (8) are read, which is what every archive
/// file measured uses.
fn zip_members(bytes: &[u8], subject: &str) -> Result<Vec<(String, Vec<u8>)>, Box<dyn Error>> {
    let mut members = Vec::new();
    let mut at = 0usize;
    let u16_at = |i: usize| -> usize { u16::from_le_bytes([bytes[i], bytes[i + 1]]) as usize };
    let u32_at =
        |i: usize| -> usize { u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize };
    while at + 30 <= bytes.len() && &bytes[at..at + 4] == b"PK\x03\x04" {
        let flags = u16_at(at + 6);
        let method = u16_at(at + 8);
        let compressed = u32_at(at + 18);
        let uncompressed = u32_at(at + 22);
        let name_len = u16_at(at + 26);
        let extra_len = u16_at(at + 28);
        let name_start = at + 30;
        let data_start = name_start + name_len + extra_len;
        if data_start > bytes.len() {
            return Err(err(format!("{subject}: a zip local header runs past the end of the file")));
        }
        let name = String::from_utf8_lossy(&bytes[name_start..name_start + name_len]).to_string();
        if flags & 0x08 != 0 && compressed == 0 {
            return Err(err(format!(
                "{subject}: member {name:?} defers its sizes to a data descriptor; this reader \
                 takes sizes from the local header only"
            )));
        }
        if data_start + compressed > bytes.len() {
            return Err(err(format!(
                "{subject}: member {name:?} declares {compressed} compressed bytes and the file \
                 holds fewer; the archive object is truncated"
            )));
        }
        let data = &bytes[data_start..data_start + compressed];
        let payload = match method {
            0 => data.to_vec(),
            8 => {
                let mut out = Vec::with_capacity(uncompressed);
                flate2::read::DeflateDecoder::new(data)
                    .take(rw_obs::MAX_EXPANDED_BYTES + 1)
                    .read_to_end(&mut out)
                    .map_err(|e| err(format!("{subject}: member {name:?} will not inflate: {e}")))?;
                if out.len() as u64 > rw_obs::MAX_EXPANDED_BYTES {
                    return Err(err(format!(
                        "{subject}: member {name:?} expands past the {}-byte ceiling",
                        rw_obs::MAX_EXPANDED_BYTES
                    )));
                }
                if out.len() != uncompressed {
                    return Err(err(format!(
                        "{subject}: member {name:?} declares {uncompressed} bytes and {} came out; \
                         a member cut short can inflate to a prefix without erroring",
                        out.len()
                    )));
                }
                out
            }
            other => {
                return Err(err(format!(
                    "{subject}: member {name:?} uses compression method {other}; only stored and \
                     deflate are read"
                )))
            }
        };
        if !name.ends_with('/') {
            members.push((name, payload));
        }
        at = data_start + compressed;
    }
    if members.is_empty() {
        return Err(err(format!("{subject}: no zip member found (not a zip, or empty)")));
    }
    Ok(members)
}

// ---------------------------------------------------------------- decode

/// The archive's missing sentinels.  Both leave the value underivable.
fn igra_int(text: &str) -> Option<i64> {
    let t = text.trim();
    if t.is_empty() {
        return None;
    }
    let v: i64 = t.parse().ok()?;
    (v != -9999 && v != -8888).then_some(v)
}

/// ETIME, the level's elapsed time since release as MMMSS (minutes may
/// run past 99), in seconds; the sentinels leave it unknown.
fn igra_etime_s(text: &str) -> Option<i64> {
    let v = igra_int(text)?;
    if v < 0 {
        return None;
    }
    Some((v / 100) * 60 + v % 100)
}

/// The release instant from the header's RELTIME (HHMM) and the nominal
/// hour: the release clock is read on the nominal date and moved a day
/// when it lands more than twelve hours from the nominal (a 23:03 release
/// for a 00Z nominal is the evening before).
fn release_instant(release_hhmm: &str, nominal: DateTime<Utc>) -> Option<DateTime<Utc>> {
    if release_hhmm.len() != 4 || release_hhmm == "9999" {
        return None;
    }
    let h: u32 = release_hhmm[0..2].parse().ok()?;
    let m: u32 = release_hhmm[2..4].parse().ok()?;
    if h > 23 || m > 59 {
        return None;
    }
    let candidate = Utc.from_utc_datetime(&nominal.date_naive().and_hms_opt(h, m, 0)?);
    Some(if candidate - nominal > Duration::hours(12) {
        candidate - Duration::days(1)
    } else if nominal - candidate > Duration::hours(12) {
        candidate + Duration::days(1)
    } else {
        candidate
    })
}

#[derive(Debug, Clone)]
struct Level {
    level_type1: u8,
    level_type2: u8,
    /// Seconds since release (ETIME), when the archive has it.
    etime_s: Option<i64>,
    pressure_pa: Option<i64>,
    gph_m: Option<i64>,
    temp_tenths_c: Option<i64>,
    dpdp_tenths_c: Option<i64>,
    wdir_deg: Option<i64>,
    wspd_tenths_m_s: Option<i64>,
}

#[derive(Debug, Clone)]
struct Sounding {
    station_id: String,
    nominal: DateTime<Utc>,
    release_hhmm: Option<String>,
    /// The release instant, when RELTIME is known (see `release_instant`).
    release: Option<DateTime<Utc>>,
    latitude_deg: f64,
    longitude_deg: f64,
    levels: Vec<Level>,
}

/// One IGRA2 v2 station file: every sounding whose nominal hour lies in
/// `[start, end]`.  Header (1-based columns): `#` 1, ID 2-12, YEAR 14-17,
/// MONTH 19-20, DAY 22-23, HOUR 25-26, RELTIME 28-31, NUMLEV 33-36, LAT
/// 56-62, LON 64-71 (1e-4 degrees).  Level: LVLTYP1 1, LVLTYP2 2, ETIME
/// 4-8, PRESS 10-15 (Pa), GPH 17-21 (m), TEMP 23-27 (0.1 C), RH 29-33,
/// DPDP 35-39 (0.1 C), WDIR 41-45, WSPD 47-51 (0.1 m/s).
fn parse_station_file(
    text: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    counters: &mut Counters,
) -> Vec<Sounding> {
    let mut out: Vec<Sounding> = Vec::new();
    let mut current: Option<Sounding> = None;
    let mut keep = false;
    for line in text.lines() {
        if let Some(header) = line.strip_prefix('#') {
            counters.soundings_in_files += 1;
            keep = false;
            current = None;
            if header.len() < 70 {
                counters.headers_malformed += 1;
                continue;
            }
            let station = header[0..11].trim().to_string();
            let field = |a: usize, b: usize| header.get(a..b).map(str::trim).unwrap_or("");
            let (Ok(year), Ok(month), Ok(day), Ok(hour)) = (
                field(12, 16).parse::<i32>(),
                field(17, 19).parse::<u32>(),
                field(20, 22).parse::<u32>(),
                field(23, 25).parse::<u32>(),
            ) else {
                counters.headers_malformed += 1;
                continue;
            };
            if hour == 99 {
                counters.soundings_without_nominal_hour += 1;
                continue;
            }
            let Some(date) = NaiveDate::from_ymd_opt(year, month, day) else {
                counters.headers_malformed += 1;
                continue;
            };
            let Some(nominal) = date.and_hms_opt(hour, 0, 0) else {
                counters.headers_malformed += 1;
                continue;
            };
            let nominal = Utc.from_utc_datetime(&nominal);
            if nominal < start || nominal > end {
                continue;
            }
            let (Some(lat), Some(lon)) = (igra_int(field(54, 61)), igra_int(field(62, 70))) else {
                counters.soundings_without_position += 1;
                continue;
            };
            let release = field(26, 30);
            keep = true;
            current = Some(Sounding {
                station_id: station,
                nominal,
                release_hhmm: (release != "9999" && !release.is_empty())
                    .then(|| release.to_string()),
                release: release_instant(release, nominal),
                latitude_deg: lat as f64 / 1.0e4,
                longitude_deg: lon as f64 / 1.0e4,
                levels: Vec::new(),
            });
            continue;
        }
        if !keep {
            continue;
        }
        let Some(sounding) = current.as_mut() else { continue };
        if line.len() < 51 {
            counters.levels_malformed += 1;
            continue;
        }
        let field = |a: usize, b: usize| line.get(a..b).map(str::trim).unwrap_or("");
        let level_type1 = field(0, 1).parse::<u8>().unwrap_or(9);
        let level_type2 = field(1, 2).parse::<u8>().unwrap_or(9);
        sounding.levels.push(Level {
            level_type1,
            level_type2,
            etime_s: igra_etime_s(field(3, 8)),
            pressure_pa: igra_int(field(9, 15)),
            gph_m: igra_int(field(16, 21)),
            temp_tenths_c: igra_int(field(22, 27)),
            dpdp_tenths_c: igra_int(field(34, 39)),
            wdir_deg: igra_int(field(40, 45)),
            wspd_tenths_m_s: igra_int(field(46, 51)),
        });
    }
    // The last sounding of the file, if kept.
    if let Some(s) = current.take() {
        if keep {
            out.push(s);
        }
    }
    // Soundings are pushed when their successor's header arrives; collect
    // the kept ones in file order.  (The loop above only holds `current`;
    // rebuild the list by a second pass over kept headers is avoided by
    // pushing on header change.)
    out
}

/// `parse_station_file` keeps one sounding at a time; this wrapper pushes
/// each kept sounding when its successor's header (or the end) arrives.
fn parse_station_file_all(
    text: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    counters: &mut Counters,
) -> Vec<Sounding> {
    // Split the file into sounding blocks at header lines, then parse each
    // block on its own so every kept block yields exactly one sounding.
    let mut blocks: Vec<&str> = Vec::new();
    let mut block_start: Option<usize> = None;
    let mut offset = 0usize;
    for line in text.split_inclusive('\n') {
        if line.starts_with('#') {
            if let Some(s) = block_start {
                blocks.push(&text[s..offset]);
            }
            block_start = Some(offset);
        }
        offset += line.len();
    }
    if let Some(s) = block_start {
        blocks.push(&text[s..]);
    }
    let mut out = Vec::new();
    for block in blocks {
        out.extend(parse_station_file(block, start, end, counters));
    }
    out
}

#[derive(Debug, Default, Serialize, Clone)]
struct Counters {
    files_read: usize,
    soundings_in_files: usize,
    soundings_in_window: usize,
    soundings_without_nominal_hour: usize,
    soundings_without_position: usize,
    headers_malformed: usize,
    levels_read: usize,
    levels_malformed: usize,
    levels_without_pressure: usize,
    levels_dropped_not_mandatory: usize,
    levels_repeated_pressure: usize,
    levels_pressure_out_of_range: usize,
    /// Aloft levels without a geopotential height, stamped with the ISA
    /// height of their pressure (`level_pa` is their anchor, so the stamp
    /// is a label, never the anchor).
    levels_without_height_isa_used: usize,
    /// Surface levels by the anchor their rows carry: the archive's own
    /// geopotential height, the station list's elevation, or none, in
    /// which case the level's rows are dropped rather than anchored at
    /// the ISA height of a surface pressure (tens to hundreds of metres
    /// from the station).
    surface_levels_anchored_by_gph: usize,
    surface_levels_anchored_by_station_list: usize,
    surface_levels_dropped_without_anchor: usize,
    /// Stations the list gave an elevation, when a list was read.
    station_list_stations: usize,
    /// Levels whose rows carry release + elapsed time, and levels that
    /// had to keep the nominal hour (no release time or no elapsed time).
    levels_with_elapsed_time: usize,
    levels_at_nominal_time: usize,
    values_temperature_out_of_range: usize,
    values_dewpoint_above_temperature: usize,
    values_wind_direction_out_of_range: usize,
    values_wind_speed_out_of_range: usize,
    values_underivable: usize,
    surface_rows: usize,
    rows_by_variable: BTreeMap<String, usize>,
    soundings_by_nominal: BTreeMap<String, usize>,
    releases_before_nominal: usize,
    release_lead_minutes_max: i64,
}

fn rows_for(
    sounding: &Sounding,
    mandatory_only: bool,
    provenance: &RowProvenance,
    counters: &mut Counters,
    writer: &mut TableWriter,
    station_elevations: &BTreeMap<String, f64>,
) {
    let mut seen_pressure = std::collections::BTreeSet::new();
    let level_provenance = provenance.measuring(MEAS_SONDE_LEVEL).nominal(sounding.nominal);
    let surface_provenance = provenance.measuring(MEAS_STATION_PRESSURE).nominal(sounding.nominal);
    for level in &sounding.levels {
        counters.levels_read += 1;
        if level.level_type1 == 3 {
            counters.levels_without_pressure += 1;
            continue;
        }
        let Some(p_pa) = level.pressure_pa else {
            counters.levels_without_pressure += 1;
            continue;
        };
        if !(100..=110_000).contains(&p_pa) {
            counters.levels_pressure_out_of_range += 1;
            continue;
        }
        if !seen_pressure.insert(p_pa) {
            counters.levels_repeated_pressure += 1;
            continue;
        }
        let is_surface = level.level_type2 == 1;
        if mandatory_only && !is_surface {
            let hpa = p_pa as f64 / 100.0;
            if !MANDATORY_HPA.iter().any(|m| (m - hpa).abs() < 1.0e-6) {
                counters.levels_dropped_not_mandatory += 1;
                continue;
            }
        }
        // A surface level is anchored at its height (its rows carry no
        // pressure, so `elevation_m` IS the anchor): the archive's
        // geopotential height, else the station list's elevation, else the
        // level is dropped by count.  An aloft level is anchored by its
        // pressure and a missing height is stamped ISA as a label only.
        let elevation_m = if is_surface {
            match level.gph_m {
                Some(z) => {
                    counters.surface_levels_anchored_by_gph += 1;
                    z as f64
                }
                None => match station_elevations.get(&sounding.station_id) {
                    Some(e) => {
                        counters.surface_levels_anchored_by_station_list += 1;
                        *e
                    }
                    None => {
                        counters.surface_levels_dropped_without_anchor += 1;
                        continue;
                    }
                },
            }
        } else {
            match level.gph_m {
                Some(z) => z as f64,
                None => {
                    counters.levels_without_height_isa_used += 1;
                    isa_altitude_m(p_pa as f64)
                }
            }
        };
        // The level's own time: release plus elapsed when both are known.
        let valid = match (sounding.release, level.etime_s) {
            (Some(release), Some(elapsed)) => {
                counters.levels_with_elapsed_time += 1;
                release + Duration::seconds(elapsed)
            }
            _ => {
                counters.levels_at_nominal_time += 1;
                sounding.nominal
            }
        };
        let base = TableRow {
            source: SOURCE.to_string(),
            station_id: sounding.station_id.clone(),
            latitude_deg: sounding.latitude_deg,
            longitude_deg: sounding.longitude_deg,
            elevation_m,
            level_pa: if is_surface { None } else { Some(p_pa as f64) },
            valid_time: valid,
            variable: String::new(),
            value: 0.0,
            error: 0.0,
            provenance: level_provenance.clone(),
        };
        if is_surface {
            // The surface level carries the station pressure at the station
            // height (the anchor found above); it is the sounding's surface
            // report.
            let mut row = base.clone();
            row.variable = VAR_SURFACE_PRESSURE.to_string();
            row.value = p_pa as f64;
            row.error = ERROR_SURFACE_PRESSURE_PA;
            row.provenance = surface_provenance.clone();
            writer.push(row, counters_rows(counters));
            counters.surface_rows += 1;
        }
        let temperature_k = level.temp_tenths_c.map(|t| t as f64 / 10.0 + 273.15);
        match temperature_k {
            Some(t) if GROSS_TEMPERATURE_K.0 <= t && t <= GROSS_TEMPERATURE_K.1 => {
                let mut row = base.clone();
                row.variable = VAR_TEMPERATURE.to_string();
                row.value = t;
                row.error = ERROR_TEMPERATURE_ALOFT_K;
                writer.push(row, counters_rows(counters));
            }
            Some(_) => counters.values_temperature_out_of_range += 1,
            None => counters.values_underivable += 1,
        }
        match (temperature_k, level.dpdp_tenths_c) {
            (Some(t), Some(dd)) if GROSS_TEMPERATURE_K.0 <= t && t <= GROSS_TEMPERATURE_K.1 => {
                if dd < 0 {
                    counters.values_dewpoint_above_temperature += 1;
                } else {
                    let mut row = base.clone();
                    row.variable = VAR_DEWPOINT.to_string();
                    row.value = t - dd as f64 / 10.0;
                    row.error = ERROR_DEWPOINT_ALOFT_K;
                    writer.push(row, counters_rows(counters));
                }
            }
            _ => counters.values_underivable += 1,
        }
        match (level.wdir_deg, level.wspd_tenths_m_s) {
            (Some(dir), Some(spd)) => {
                let speed = spd as f64 / 10.0;
                if !(0..=360).contains(&dir) {
                    counters.values_wind_direction_out_of_range += 1;
                } else if !(GROSS_WIND_M_S.0..=GROSS_WIND_M_S.1).contains(&speed) {
                    counters.values_wind_speed_out_of_range += 1;
                } else {
                    let (u, v) = wind_components(dir as f64, speed);
                    let mut ru = base.clone();
                    ru.variable = VAR_WIND_U.to_string();
                    ru.value = u;
                    ru.error = ERROR_WIND_ALOFT_M_S;
                    writer.push(ru, counters_rows(counters));
                    let mut rv = base.clone();
                    rv.variable = VAR_WIND_V.to_string();
                    rv.value = v;
                    rv.error = ERROR_WIND_ALOFT_M_S;
                    writer.push(rv, counters_rows(counters));
                }
            }
            _ => counters.values_underivable += 1,
        }
    }
}

fn counters_rows(counters: &mut Counters) -> &mut BTreeMap<String, usize> {
    &mut counters.rows_by_variable
}

fn read_source_text(path: &Path) -> Result<(String, String, usize), Box<dyn Error>> {
    let bytes = std::fs::read(path).map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
    let sha = hex_sha256(&bytes);
    let subject = path.display().to_string();
    let text = if bytes.starts_with(b"PK\x03\x04") {
        let members = zip_members(&bytes, &subject)?;
        let mut joined = String::new();
        for (_, payload) in members {
            joined.push_str(&String::from_utf8_lossy(&payload));
            if !joined.ends_with('\n') {
                joined.push('\n');
            }
        }
        joined
    } else {
        String::from_utf8_lossy(&bytes).to_string()
    };
    Ok((text, sha, bytes.len()))
}

#[derive(Serialize)]
struct SourceFile {
    path: String,
    bytes: usize,
    sha256: String,
    soundings_in_window: usize,
    /// Seconds the archive file was published after the latest nominal
    /// hour it holds inside the window (its `Last-Modified` from the fetch
    /// record minus that hour), when the fetch record names the file.
    behind_latest_sounding_s: Option<i64>,
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
    mandatory_only: bool,
    /// The station list read for the surface anchors, when one was.
    station_list: Option<String>,
    errors: BTreeMap<&'static str, f64>,
    counters: Counters,
    files: Vec<SourceFile>,
    /// The largest `behind_latest_sounding_s` over the files, when known:
    /// how far behind real time the archive's own timestamps say this
    /// collection arrives.
    latency_behind_real_time_s: Option<i64>,
    latency_basis: &'static str,
}

#[derive(Serialize)]
struct WindowRecord {
    start: String,
    end: String,
}

fn cmd_table(options: &Options) -> Result<String, Box<dyn Error>> {
    let dir = options
        .zips
        .as_deref()
        .ok_or_else(|| err("--zips DIR is required (the directory `fetch` wrote)"))?;
    let out = options
        .out
        .as_deref()
        .ok_or_else(|| err("--out FILE.csv is required"))?;
    if out.is_dir() {
        return Err(err(format!("--out {} is a directory; give the CSV path", out.display())));
    }
    let (start, end) = options.window()?;
    let (last_modified, fetched_at): (BTreeMap<String, DateTime<Utc>>, BTreeMap<String, DateTime<Utc>>) =
        match &options.fetch_record {
            Some(path) => {
                let text = std::fs::read_to_string(path)
                    .map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
                let record: FetchRecord = serde_json::from_str(&text)
                    .map_err(|e| err(format!("{} is not a fetch record: {e}", path.display())))?;
                if record.schema != FETCH_SCHEMA {
                    return Err(err(format!(
                        "{} declares schema {:?}, expected {FETCH_SCHEMA:?}",
                        path.display(),
                        record.schema
                    )));
                }
                let modified = record
                    .files
                    .iter()
                    .filter_map(|f| {
                        f.last_modified
                            .as_deref()
                            .and_then(|t| parse_time(&format!("{t}Z")).ok())
                            .map(|t| (f.file.clone(), t))
                    })
                    .collect();
                let received = record
                    .files
                    .iter()
                    .filter_map(|f| {
                        parse_time(&format!("{}Z", f.fetched_at)).ok().map(|t| (f.file.clone(), t))
                    })
                    .collect();
                (modified, received)
            }
            None => (BTreeMap::new(), BTreeMap::new()),
        };
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| err(format!("cannot list {}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            p.is_file()
                && name != STATION_LIST_FILE
                && (name.ends_with(".txt.zip") || name.ends_with(".txt"))
        })
        .collect();
    paths.sort();
    let station_list_path = options
        .station_list
        .clone()
        .or_else(|| dir.join(STATION_LIST_FILE).is_file().then(|| dir.join(STATION_LIST_FILE)));
    let (station_elevations, station_list_used) = match &station_list_path {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| err(format!("cannot read the station list {}: {e}", path.display())))?;
            let (map, _) = parse_station_list(&text);
            if map.is_empty() {
                return Err(err(format!(
                    "{} holds no station with an elevation; the layout this reader knows is \
                     ID 1-11, LAT 13-20, LON 22-30, ELEV 32-37",
                    path.display()
                )));
            }
            (map, Some(rw_obs::absolute_uri(path)))
        }
        None => {
            eprintln!(
                "rw_igra2: no station list ({STATION_LIST_FILE} beside the files or --station-list); \
                 surface levels without a geopotential height are dropped and counted"
            );
            (BTreeMap::new(), None)
        }
    };
    if paths.is_empty() {
        return Err(err(format!(
            "{} holds no IGRA2 station file (*.txt.zip or *.txt)",
            dir.display()
        )));
    }
    let mut counters = Counters::default();
    counters.station_list_stations = station_elevations.len();
    let mut writer = TableWriter::new();
    let mut files = Vec::with_capacity(paths.len());
    for path in &paths {
        let (text, sha, bytes) = read_source_text(path)?;
        counters.files_read += 1;
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        let provenance = RowProvenance::of_source(
            &sha,
            last_modified.get(&name).copied(),
            fetched_at.get(&name).copied(),
        );
        let mut soundings = parse_station_file_all(&text, start, end, &mut counters);
        soundings.sort_by(|a, b| a.nominal.cmp(&b.nominal));
        let mut latest: Option<DateTime<Utc>> = None;
        for sounding in &soundings {
            counters.soundings_in_window += 1;
            *counters
                .soundings_by_nominal
                .entry(seam_time(sounding.nominal))
                .or_insert(0) += 1;
            if let Some(release) = &sounding.release_hhmm {
                if release.len() == 4 {
                    if let (Ok(h), Ok(m)) = (release[0..2].parse::<i64>(), release[2..4].parse::<i64>()) {
                        // A release is before the nominal hour when its clock
                        // reads earlier in the day than the nominal (or late
                        // the previous day for a 00Z nominal).
                        let nominal_min = sounding.nominal.format("%H").to_string().parse::<i64>().unwrap_or(0) * 60;
                        let mut lead = nominal_min - (h * 60 + m);
                        if lead < -12 * 60 {
                            lead += 24 * 60;
                        }
                        if lead > 0 {
                            counters.releases_before_nominal += 1;
                            counters.release_lead_minutes_max = counters.release_lead_minutes_max.max(lead);
                        }
                    }
                }
            }
            latest = Some(latest.map_or(sounding.nominal, |l| l.max(sounding.nominal)));
            rows_for(sounding, options.mandatory_only, &provenance, &mut counters, &mut writer, &station_elevations);
        }
        let behind = match (latest, last_modified.get(&name)) {
            (Some(l), Some(m)) => Some((*m - l).num_seconds()),
            _ => None,
        };
        files.push(SourceFile {
            path: rw_obs::absolute_uri(path),
            bytes,
            sha256: sha,
            soundings_in_window: soundings.len(),
            behind_latest_sounding_s: behind,
        });
    }
    let (rows, csv_sha, csv_bytes) = writer.write(out)?;
    let mut errors = BTreeMap::new();
    errors.insert(VAR_SURFACE_PRESSURE, ERROR_SURFACE_PRESSURE_PA);
    errors.insert(VAR_TEMPERATURE, ERROR_TEMPERATURE_ALOFT_K);
    errors.insert(VAR_DEWPOINT, ERROR_DEWPOINT_ALOFT_K);
    errors.insert(VAR_WIND_U, ERROR_WIND_ALOFT_M_S);
    errors.insert(VAR_WIND_V, ERROR_WIND_ALOFT_M_S);
    let record = TableRecord {
        schema: TABLE_RECORD_SCHEMA,
        status: if rows > 0 { "READY" } else { "EMPTY" },
        source: SOURCE,
        table_schema: TABLE_SCHEMA,
        path: rw_obs::absolute_uri(out),
        sha256: csv_sha,
        rows,
        bytes: csv_bytes,
        window: WindowRecord {
            start: seam_time(start),
            end: seam_time(end),
        },
        mandatory_only: options.mandatory_only,
        station_list: station_list_used,
        errors,
        latency_behind_real_time_s: files.iter().filter_map(|f| f.behind_latest_sounding_s).max(),
        latency_basis: "archive file Last-Modified minus the latest nominal hour it holds in the window",
        counters,
        files,
    };
    let text = format!("{}\n", serde_json::to_string_pretty(&record)?);
    std::fs::write(out.with_extension("json"), &text)
        .map_err(|e| err(format!("cannot write the table record: {e}")))?;
    Ok(text)
}

// ---------------------------------------------------------------- verify

fn cmd_verify(options: &Options) -> Result<String, Box<dyn Error>> {
    let file = options.file.as_deref().ok_or_else(|| err("--file is required"))?;
    let record_path = options
        .record
        .as_deref()
        .ok_or_else(|| err("--record is required"))?;
    let record: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(record_path)
            .map_err(|e| err(format!("cannot read {}: {e}", record_path.display())))?,
    )
    .map_err(|e| err(format!("{} is not JSON: {e}", record_path.display())))?;
    if record.get("schema").and_then(|s| s.as_str()) != Some(TABLE_RECORD_SCHEMA) {
        return Err(err(format!(
            "{} does not declare {TABLE_RECORD_SCHEMA}",
            record_path.display()
        )));
    }
    let bytes = std::fs::read(file).map_err(|e| err(format!("cannot read {}: {e}", file.display())))?;
    let actual = hex_sha256(&bytes);
    let stated = record.get("sha256").and_then(|s| s.as_str()).unwrap_or("");
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
            stated_sha256: stated.to_string(),
            actual_sha256: actual,
        })?
    );
    if ok {
        Ok(text)
    } else {
        Err(err(format!("table digest mismatch:\n{text}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn header(id: &str, y: i32, m: u32, d: u32, h: u32, reltime: &str, numlev: usize, lat: i64, lon: i64) -> String {
        // #ID(11) YEAR(4) MONTH(2) DAY(2) HOUR(2) RELTIME(4) NUMLEV(4) P_SRC(8) NP_SRC(8) LAT(7) LON(8)
        format!(
            "#{:<11} {:04} {:02} {:02} {:02} {:>4} {:>4} {:<8} {:<8} {:>7} {:>8}",
            id, y, m, d, h, reltime, numlev, "ncdc-gts", "ncdc-gts", lat, lon
        )
    }

    fn level(t1: u8, t2: u8, press: i64, gph: i64, temp: i64, rh: i64, dpdp: i64, wdir: i64, wspd: i64) -> String {
        // LVLTYP1 LVLTYP2 ETIME(5) PRESS(6) PFLAG GPH(5) ZFLAG TEMP(5) TFLAG RH(5) DPDP(5) WDIR(5) WSPD(5)
        format!(
            "{}{} {:>5} {:>6}{}{:>5}{}{:>5}{}{:>5} {:>5} {:>5} {:>5}",
            t1, t2, -9999, press, "A", gph, "B", temp, "B", rh, dpdp, wdir, wspd
        )
    }

    fn sample() -> String {
        let mut s = String::new();
        s.push_str(&header("USM00072365", 2026, 9, 1, 0, "2303", 4, 350400, -1066200));
        s.push('\n');
        s.push_str(&level(2, 1, 83800, 1620, 250, 300, 80, 180, 30));
        s.push('\n');
        s.push_str(&level(1, 0, 50000, 5870, -100, 200, 250, 270, 150));
        s.push('\n');
        s.push_str(&level(1, 0, 50000, 5870, -100, 200, 250, 270, 150)); // repeat
        s.push('\n');
        s.push_str(&level(2, 0, 30000, -9999, -400, -9999, -9999, 250, 300)); // no gph, no dewpoint
        s.push('\n');
        s.push_str(&header("USM00072365", 2026, 9, 1, 12, "1100", 1, 350400, -1066200));
        s.push('\n');
        s.push_str(&level(1, 0, 70000, 3000, 0, 500, 50, 0, 0));
        s.push('\n');
        s
    }

    #[test]
    fn header_and_levels_decode_to_neutral_rows() {
        let start = parse_time("2026-09-01T00:00:00Z").unwrap();
        let end = parse_time("2026-09-01T00:00:00Z").unwrap();
        let mut c = Counters::default();
        let soundings = parse_station_file_all(&sample(), start, end, &mut c);
        assert_eq!(soundings.len(), 1, "{c:?}");
        assert_eq!(c.soundings_in_files, 2);
        let s = &soundings[0];
        assert_eq!(s.station_id, "USM00072365");
        assert!((s.latitude_deg - 35.04).abs() < 1e-9);
        assert!((s.longitude_deg + 106.62).abs() < 1e-9);
        assert_eq!(s.levels.len(), 4);
        let mut w = TableWriter::new();
        rows_for(s, false, &RowProvenance::default(), &mut c, &mut w, &BTreeMap::new());
        let rows = w.rows();
        // surface: ps + T + Td + u + v = 5; 500: T, Td, u, v = 4; repeat dropped; 300: T, u, v = 3
        assert_eq!(rows.len(), 12, "{c:?}");
        assert_eq!(c.surface_levels_anchored_by_gph, 1);
        assert_eq!(c.surface_levels_dropped_without_anchor, 0);
        // Every level of the sample carries ETIME -9999, so every row keeps
        // the nominal hour and says so.
        assert_eq!(c.levels_at_nominal_time, 3);
        assert_eq!(c.levels_with_elapsed_time, 0);
        assert!(rows.iter().all(|r| r.valid_time == s.nominal));
        assert!(rows.iter().all(|r| r.provenance.nominal_time == Some(s.nominal)));
        let ps = rows.iter().find(|r| r.variable == VAR_SURFACE_PRESSURE).unwrap();
        assert_eq!(ps.provenance.measurement, MEAS_STATION_PRESSURE);
        assert!(rows.iter().filter(|r| r.variable != VAR_SURFACE_PRESSURE).all(|r| r.provenance.measurement == MEAS_SONDE_LEVEL));
        assert_eq!(c.levels_repeated_pressure, 1);
        assert_eq!(c.levels_without_height_isa_used, 1);
        let ps = rows.iter().find(|r| r.variable == VAR_SURFACE_PRESSURE).unwrap();
        assert_eq!(ps.value, 83800.0);
        assert_eq!(ps.level_pa, None);
        assert_eq!(ps.elevation_m, 1620.0);
        let t500 = rows
            .iter()
            .find(|r| r.variable == VAR_TEMPERATURE && r.level_pa == Some(50000.0))
            .unwrap();
        assert!((t500.value - 263.15).abs() < 1e-9);
        let td500 = rows
            .iter()
            .find(|r| r.variable == VAR_DEWPOINT && r.level_pa == Some(50000.0))
            .unwrap();
        assert!((td500.value - 238.15).abs() < 1e-9);
        let u500 = rows
            .iter()
            .find(|r| r.variable == VAR_WIND_U && r.level_pa == Some(50000.0))
            .unwrap();
        // 270 deg at 15 m/s: from the west, u = +15, v = 0
        assert!((u500.value - 15.0).abs() < 1e-9, "{}", u500.value);
        let z300 = rows
            .iter()
            .find(|r| r.variable == VAR_TEMPERATURE && r.level_pa == Some(30000.0))
            .unwrap();
        assert!((z300.elevation_m - 9164.0).abs() < 5.0, "{}", z300.elevation_m);
    }

    #[test]
    fn mandatory_only_drops_significant_levels_and_keeps_the_surface() {
        let start = parse_time("2026-09-01T00:00:00Z").unwrap();
        let end = parse_time("2026-09-01T12:00:00Z").unwrap();
        let mut c = Counters::default();
        let soundings = parse_station_file_all(&sample(), start, end, &mut c);
        assert_eq!(soundings.len(), 2);
        let mut w = TableWriter::new();
        for s in &soundings {
            rows_for(s, true, &RowProvenance::default(), &mut c, &mut w, &BTreeMap::new());
        }
        // 00Z: surface 5 + 500 4 (300 is mandatory too: 3) = 12; 12Z: 700 (T, Td, u, v) = 4
        assert_eq!(w.rows().len(), 16, "{c:?}");
        assert_eq!(c.levels_dropped_not_mandatory, 0);
    }

    #[test]
    fn levels_carry_release_plus_elapsed_time_and_the_nominal_hour() {
        // Released 23:03 the evening before the 00Z nominal; the 500 hPa
        // level 22 min 15 s after release, the surface at release.
        let mut s = String::new();
        s.push_str(&header("USM00072365", 2026, 9, 1, 0, "2303", 2, 350400, -1066200));
        s.push('\n');
        s.push_str(&format!(
            "{}{} {:>5} {:>6}{}{:>5}{}{:>5}{}{:>5} {:>5} {:>5} {:>5}",
            2, 1, 0, 83800, "A", 1620, "B", 250, "B", 300, 80, 180, 30
        ));
        s.push('\n');
        s.push_str(&format!(
            "{}{} {:>5} {:>6}{}{:>5}{}{:>5}{}{:>5} {:>5} {:>5} {:>5}",
            1, 0, 2215, 50000, "A", 5870, "B", -100, "B", 200, 250, 270, 150
        ));
        s.push('\n');
        let start = parse_time("2026-09-01T00:00:00Z").unwrap();
        let mut c = Counters::default();
        let soundings = parse_station_file_all(&s, start, start, &mut c);
        assert_eq!(soundings.len(), 1);
        let sounding = &soundings[0];
        assert_eq!(sounding.release, Some(parse_time("2026-08-31T23:03:00Z").unwrap()));
        assert_eq!(sounding.levels[1].etime_s, Some(22 * 60 + 15));
        let mut w = TableWriter::new();
        let provenance = RowProvenance::of_source(
            "16d059c6d0e21396028900d70a35210c171fca1c68409c31765eaeea46b63121",
            Some(parse_time("2026-09-02T21:36:42Z").unwrap()),
            Some(parse_time("2026-09-06T01:01:18Z").unwrap()),
        );
        rows_for(sounding, false, &provenance, &mut c, &mut w, &BTreeMap::new());
        let rows = w.rows();
        assert_eq!(c.levels_with_elapsed_time, 2);
        assert_eq!(c.levels_at_nominal_time, 0);
        let ps = rows.iter().find(|r| r.variable == VAR_SURFACE_PRESSURE).unwrap();
        assert_eq!(ps.valid_time, parse_time("2026-08-31T23:03:00Z").unwrap());
        let t500 = rows.iter().find(|r| r.variable == VAR_TEMPERATURE && r.level_pa == Some(50000.0)).unwrap();
        assert_eq!(t500.valid_time, parse_time("2026-08-31T23:25:15Z").unwrap());
        assert_eq!(t500.provenance.nominal_time, Some(start));
        assert_eq!(t500.provenance.revision, "16d059c6d0e2");
        assert!(t500.csv_line().ends_with(
            ",sonde_level,2026-09-01T00:00:00Z,2026-09-02T21:36:42Z,2026-09-06T01:01:18Z,16d059c6d0e2"
        ), "{}", t500.csv_line());
        // A release clock later than the nominal by less than twelve hours
        // is the same day; 9999 is unknown.
        assert_eq!(
            release_instant("0115", start),
            Some(parse_time("2026-09-01T01:15:00Z").unwrap())
        );
        assert_eq!(release_instant("9999", start), None);
        assert_eq!(igra_etime_s(" 1234"), Some(12 * 60 + 34));
        assert_eq!(igra_etime_s("-9999"), None);
        assert_eq!(igra_etime_s("-8888"), None);
    }

    #[test]
    fn a_surface_level_without_height_takes_the_station_list_anchor_or_is_dropped() {
        // Dar-El-Beida (25 m in the station list) reporting a surface level
        // the archive gives no geopotential height, at 1014 hPa: the ISA
        // height of that pressure is -6 m, thirty-one metres off.
        let mut s = String::new();
        s.push_str(&header("AGM00060390", 2026, 9, 1, 0, "2300", 2, 366899, 32166));
        s.push('\n');
        s.push_str(&level(2, 1, 101400, -9999, 255, 800, 35, 90, 40));
        s.push('\n');
        s.push_str(&level(1, 0, 50000, 5870, -100, 200, 250, 270, 150));
        s.push('\n');
        let start = parse_time("2026-09-01T00:00:00Z").unwrap();
        let mut c = Counters::default();
        let soundings = parse_station_file_all(&s, start, start, &mut c);
        assert_eq!(soundings.len(), 1);
        // No station list: the surface level's rows are dropped and counted
        // (never anchored at the ISA height), the 500 hPa level stays.
        let mut w = TableWriter::new();
        rows_for(&soundings[0], false, &RowProvenance::default(), &mut c, &mut w, &BTreeMap::new());
        assert_eq!(c.surface_levels_dropped_without_anchor, 1, "{c:?}");
        assert_eq!(c.surface_rows, 0);
        assert_eq!(w.rows().len(), 4, "{:?}", w.rows());
        assert!(w.rows().iter().all(|r| r.level_pa == Some(50000.0)));
        // The station list: the archive's fixed columns, a missing elevation counted.
        let (list, without) = parse_station_list(
            "AGM00060390  36.6899    3.2166   25.0    DAR-EL-BEIDA                   1948 2026  71380\n\
             XXM00000001   0.0000    0.0000 -998.8    NOWHERE                        1900 1901      1\n\
             USM00072365  35.0378 -106.6219 1619.0 NM ALBUQUERQUE/INT.; NM.          1931 2026  85912\n",
        );
        assert_eq!(list.get("AGM00060390"), Some(&25.0));
        assert_eq!(list.get("USM00072365"), Some(&1619.0));
        assert_eq!(list.len(), 2);
        assert_eq!(without, 1);
        // With it: every surface row anchored at 25 m, the station pressure row written.
        let mut c = Counters::default();
        let mut w = TableWriter::new();
        rows_for(&soundings[0], false, &RowProvenance::default(), &mut c, &mut w, &list);
        assert_eq!(c.surface_levels_anchored_by_station_list, 1, "{c:?}");
        assert_eq!(c.surface_levels_dropped_without_anchor, 0);
        assert_eq!(c.surface_rows, 1);
        assert_eq!(w.rows().len(), 9, "{:?}", w.rows());
        let ps = w.rows().iter().find(|r| r.variable == VAR_SURFACE_PRESSURE).unwrap();
        assert_eq!((ps.value, ps.elevation_m, ps.level_pa), (101400.0, 25.0, None));
        assert_eq!(ps.provenance.measurement, MEAS_STATION_PRESSURE);
        assert!(w.rows().iter().filter(|r| r.level_pa.is_none()).all(|r| r.elevation_m == 25.0));
        assert_eq!(w.rows().iter().filter(|r| r.level_pa.is_none()).count(), 5);
        assert!(w.rows().iter().all(|r| r.elevation_m != isa_altitude_m(101400.0)));
    }

    #[test]
    fn a_zip_member_inflates_to_the_declared_length() {
        let payload = sample();
        let mut deflated = Vec::new();
        {
            let mut enc = flate2::write::DeflateEncoder::new(&mut deflated, flate2::Compression::default());
            enc.write_all(payload.as_bytes()).unwrap();
            enc.finish().unwrap();
        }
        let name = b"USM00072365-data-beg2026.txt";
        let mut zip = Vec::new();
        zip.extend_from_slice(b"PK\x03\x04");
        zip.extend_from_slice(&[20, 0, 0, 0, 8, 0]); // version, flags, method deflate
        zip.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]); // time, date, crc
        zip.extend_from_slice(&(deflated.len() as u32).to_le_bytes());
        zip.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        zip.extend_from_slice(&(name.len() as u16).to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(name);
        zip.extend_from_slice(&deflated);
        let members = zip_members(&zip, "test").unwrap();
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].0, "USM00072365-data-beg2026.txt");
        assert_eq!(members[0].1, payload.as_bytes());
        // Truncated inside the deflate stream: refused, not returned as a prefix.
        let mut cut = zip.clone();
        cut.truncate(zip.len() - 20);
        assert!(zip_members(&cut, "test").is_err());
    }

    #[test]
    fn the_index_parser_reads_station_files_only() {
        let html = r#"<a href="../">..</a> <a href="AEM00041217-data-beg2026.txt.zip">x</a>
        <a href="igra2-station-list.txt">list</a> <a href="USM00072365-data-beg2026.txt.zip">y</a>"#;
        let listed = parse_index(html, "https://x/data-y2d");
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].station_id, "AEM00041217");
        assert_eq!(listed[1].url, "https://x/data-y2d/USM00072365-data-beg2026.txt.zip");
    }

    #[test]
    fn abi_marker_names_the_contracts_it_pins() {
        assert!(ABI_MARKER.contains(FETCH_SCHEMA));
        assert!(ABI_MARKER.contains(TABLE_RECORD_SCHEMA));
        assert!(ABI_MARKER.contains(TABLE_SCHEMA));
    }
}
