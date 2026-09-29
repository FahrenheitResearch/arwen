//! `rw_amv` -- the satellite atmospheric-motion-vector front door.
//!
//! GOES ABI Level 2 Derived Motion Winds from the NOAA open-data buckets
//! (`noaa-goes16` .. `noaa-goes19`, anonymous S3): the full-disk product
//! `ABI-L2-DMWF` is hourly, six granules per hour (bands 2, 7, 8, 9, 10
//! and 14 on the operational pair GOES-18 West and GOES-19 East as of
//! 2026-09; GOES-16 publishes none that day).  `list` and `fetch` walk the
//! hour prefixes and download through rw-sat's content-addressed cache;
//! `table` decodes granules through `rw_sat::dmw` (DQF-gated, counted)
//! and writes the neutral observation table (`gpuwm-obs.table.v1`): one
//! `wind_u_m_s` and one `wind_v_m_s` row per kept vector, anchored at the
//! vector's assigned pressure, valid at the vector's own image mid-point,
//! station id `<satellite>-C<band>`, source `goes-dmw`, measurement
//! `amv_assigned_pressure`; the row's `published_time` is the granule's
//! creation stamp from its filename (`c...`), its `received_time` the
//! fetch record's `fetched_at` for that granule, its `revision` the
//! granule's digest.
//!
//! Quality control, each a count: the DQF gate (0 is good; the product's
//! flags 1 bad track, 2 bad height assignment and 3 invalid are counted by
//! value), a local zenith angle above `--max-zenith-deg` (default 68, the
//! limb where the height assignment degrades), a pressure outside 100 to
//! 1050 hPa, a speed above 150 m/s.  Thinning, default on: one vector per
//! `--thin-deg` box (default 0.5 degree, about the T255 Gaussian cell) per
//! `--thin-layer-hpa` layer (default 50 hPa) per satellite per hour; the
//! kept vector is the one with the smallest local zenith angle in the box
//! (the best viewing geometry), ties by file order.  Bands are pooled
//! before thinning, so a box keeps one vector however many bands saw it.
//! Observation error by layer: 3.0 m/s below 700 hPa, 4.0 between, 5.0
//! above 400 hPa (the height assignment dominates aloft).
//!
//! Latency: a granule's `LastModified` on the bucket minus the scan end in
//! its filename is how far behind real time the product arrives; the fetch
//! record carries it per file and the table record's
//! `latency_behind_real_time_s` is the largest over the files it read.
//!
//! ```text
//! rw_amv list  --satellite G19 --start TIME --end TIME [--bands 14,8]
//! rw_amv fetch --satellite G19 --start TIME --end TIME --cache DIR [--bands LIST]
//! rw_amv table --files a.nc b.nc ... --out FILE.csv [--fetch-record FILE ...]
//!              [--thin-deg D] [--thin-layer-hpa L] [--no-thin] [--max-zenith-deg A]
//! rw_amv verify --file FILE.csv --record FILE.json
//! ```

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use chrono::{DateTime, Duration, Timelike, Utc};
use serde::{Deserialize, Serialize};

use rw_nexrad::s3::{parse_s3_timestamp, parse_time};
use rw_obs::seam::seam_time;
use rw_obs::table::{
    revision_of, wind_components, RowProvenance, TableRow, TableWriter, ERROR_AMV_HIGH_M_S,
    ERROR_AMV_LOW_M_S, ERROR_AMV_MID_M_S, GROSS_WIND_M_S, MEAS_AMV_ASSIGNED_PRESSURE,
    TABLE_SCHEMA, VAR_WIND_U, VAR_WIND_V,
};
use rw_obs::{err, hex_sha256};
use rw_sat::dmw::{read_dmw_granule, DmwCounts};
use rw_sat::goes::{parse_goes_abi_filename, GoesSatellite};
use rw_sat::s3::{
    bucket_for_satellite, build_agent, download_object, goes_hour_prefix, list_s3_objects,
    S3Object,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

const SOURCE: &str = "goes-dmw";
const DEFAULT_PRODUCT: &str = "ABI-L2-DMWF";
const DEFAULT_MAX_ZENITH_DEG: f64 = 68.0;
const DEFAULT_THIN_DEG: f64 = 0.5;
const DEFAULT_THIN_LAYER_HPA: f64 = 50.0;
const PRESSURE_MIN_HPA: f64 = 100.0;
const PRESSURE_MAX_HPA: f64 = 1050.0;

const LIST_SCHEMA: &str = "gpuwm-obs.amv-list.v1";
const FETCH_SCHEMA: &str = "gpuwm-obs.amv-fetch.v1";
const TABLE_RECORD_SCHEMA: &str = "gpuwm-obs.amv-table.v1";
const VERIFY_SCHEMA: &str = "gpuwm-obs.amv-verify.v1";

const ABI_MARKER: &str = "gpuwm-obs.amv-fetch.v1\tgpuwm-obs.amv-table.v1\tgpuwm-obs.table.v2\t\
wind_u_m_s\twind_v_m_s\tdqf\tzenith\tthin";

const USAGE: &str = "\
usage: rw_amv <list|fetch|table|verify> [OPTIONS]
       rw_amv --version | --help | --abi

  list    report the DMW granules a (satellite, window) resolves to, moving nothing
  fetch   download them into the content-addressed cache with a sha256 and the
          bucket's LastModified per file
  table   decode granules into a `gpuwm-obs.table.v1` CSV of u and v rows
  verify  re-hash a table against the record written beside it

acquisition options (list, fetch)
  --satellite ID        G16 / G17 / G18 / G19; selects noaa-goes{NN}
  --product NAME        default ABI-L2-DMWF (full disk, hourly); DMWC is the CONUS
                        sector at five minutes
  --bands LIST          comma-separated band numbers to keep (default all)
  --start TIME          window start, e.g. 2026-09-01T18:00:00Z
  --end TIME            window end (inclusive; whole hours are listed)
  --cache DIR           object cache root; layout <cache>/satellite/<bucket>/<key>
  --no-cache            always re-download

table options
  --files LIST          granule paths, comma-separated (or repeat --files)
  --out FILE.csv        the table; a .json record is written beside it
  --fetch-record FILE   a `fetch` record, for per-file latency and receipt
                        times; repeat it for several satellites
  --thin-deg D          box size in degrees (default 0.5)
  --thin-layer-hpa L    layer thickness in hPa (default 50)
  --no-thin             keep every vector
  --max-zenith-deg A    drop vectors seen past this local zenith angle (default 68)

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
            eprintln!("rw_amv: {error}");
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
        "--version" | "-V" => return Ok(format!("rw_amv {VERSION}\n")),
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
    satellite: Option<String>,
    product: Option<String>,
    bands: Option<Vec<u8>>,
    start: Option<String>,
    end: Option<String>,
    cache: Option<PathBuf>,
    no_cache: bool,
    files: Vec<PathBuf>,
    out: Option<PathBuf>,
    fetch_record: Vec<PathBuf>,
    thin_deg: Option<f64>,
    thin_layer_hpa: Option<f64>,
    no_thin: bool,
    max_zenith_deg: Option<f64>,
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
            let positive = |raw: String, flag: &str| -> Result<f64, Box<dyn Error>> {
                let v: f64 = raw
                    .parse()
                    .map_err(|_| err(format!("{flag} expects a number, got {raw:?}")))?;
                if !v.is_finite() || v <= 0.0 {
                    return Err(err(format!("{flag} must be positive, got {raw:?}")));
                }
                Ok(v)
            };
            match flag {
                "--satellite" => options.satellite = Some(value()?),
                "--product" => options.product = Some(value()?),
                "--bands" => {
                    let raw = value()?;
                    let mut bands = Vec::new();
                    for token in raw.split(',').map(str::trim).filter(|t| !t.is_empty()) {
                        let b: u8 = token
                            .parse()
                            .map_err(|_| err(format!("--bands token {token:?} is not a band")))?;
                        if !(1..=16).contains(&b) {
                            return Err(err(format!("band {b} is outside ABI's 1..16")));
                        }
                        bands.push(b);
                    }
                    if bands.is_empty() {
                        return Err(err("--bands named no band"));
                    }
                    options.bands = Some(bands);
                }
                "--start" => options.start = Some(value()?),
                "--end" => options.end = Some(value()?),
                "--cache" => options.cache = Some(PathBuf::from(value()?)),
                "--no-cache" => options.no_cache = true,
                "--files" => {
                    let raw = value()?;
                    options.files.extend(
                        raw.split(',').map(str::trim).filter(|t| !t.is_empty()).map(PathBuf::from),
                    );
                }
                "--out" => options.out = Some(PathBuf::from(value()?)),
                "--fetch-record" => options.fetch_record.push(PathBuf::from(value()?)),
                "--thin-deg" => options.thin_deg = Some(positive(value()?, "--thin-deg")?),
                "--thin-layer-hpa" => {
                    options.thin_layer_hpa = Some(positive(value()?, "--thin-layer-hpa")?)
                }
                "--no-thin" => options.no_thin = true,
                "--max-zenith-deg" => {
                    options.max_zenith_deg = Some(positive(value()?, "--max-zenith-deg")?)
                }
                "--file" => options.file = Some(PathBuf::from(value()?)),
                "--record" => options.record = Some(PathBuf::from(value()?)),
                other => return Err(err(format!("unknown option {other:?}\n\n{USAGE}"))),
            }
            index += 1;
        }
        Ok(options)
    }

    fn satellite(&self) -> Result<GoesSatellite, Box<dyn Error>> {
        let raw = self
            .satellite
            .as_deref()
            .ok_or_else(|| err("--satellite is required (G16 / G17 / G18 / G19)"))?;
        Ok(GoesSatellite::parse(raw))
    }

    fn product(&self) -> String {
        self.product
            .as_deref()
            .unwrap_or(DEFAULT_PRODUCT)
            .trim()
            .to_ascii_uppercase()
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

    fn cache(&self) -> PathBuf {
        self.cache
            .clone()
            .unwrap_or_else(|| PathBuf::from(".rw-amv-cache"))
    }
}

// ------------------------------------------------------------------ list

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Granule {
    key: String,
    bucket: String,
    satellite: String,
    band: u8,
    scan_start: String,
    scan_end: String,
    /// The bucket's own statement of when the object landed.
    last_modified: Option<String>,
    /// LastModified minus the scan end, seconds: the product's latency.
    behind_scan_end_s: Option<i64>,
    bytes: u64,
}

fn list_granules(options: &Options) -> Result<(String, Vec<Granule>), Box<dyn Error>> {
    let satellite = options.satellite()?;
    let bucket = bucket_for_satellite(satellite.as_str())?;
    let product = options.product();
    let (start, end) = options.window()?;
    let agent = build_agent();
    let mut hour = start
        .with_minute(0)
        .and_then(|t| t.with_second(0))
        .and_then(|t| t.with_nanosecond(0))
        .unwrap_or(start);
    let mut granules = Vec::new();
    while hour <= end {
        let prefix = goes_hour_prefix(&product, hour);
        let objects: Vec<S3Object> = list_s3_objects(&agent, &bucket, &prefix, None)?;
        for object in objects {
            let Ok(name) = parse_goes_abi_filename(&object.key) else { continue };
            if name.satellite != satellite {
                continue;
            }
            let Some(band) = name.channel else { continue };
            if let Some(bands) = &options.bands {
                if !bands.contains(&band) {
                    continue;
                }
            }
            if name.start_time_utc < start || name.start_time_utc > end {
                continue;
            }
            let last_modified = parse_s3_timestamp(&object.last_modified);
            granules.push(Granule {
                key: object.key.clone(),
                bucket: bucket.clone(),
                satellite: satellite.as_str().to_string(),
                band,
                scan_start: seam_time(name.start_time_utc),
                scan_end: seam_time(name.end_time_utc),
                last_modified: last_modified.map(seam_time),
                behind_scan_end_s: last_modified.map(|m| (m - name.end_time_utc).num_seconds()),
                bytes: object.size_bytes,
            });
        }
        hour += Duration::hours(1);
    }
    granules.sort_by(|a, b| a.key.cmp(&b.key));
    Ok((bucket, granules))
}

fn cmd_list(options: &Options) -> Result<String, Box<dyn Error>> {
    let (bucket, granules) = list_granules(options)?;
    #[derive(Serialize)]
    struct Record {
        schema: &'static str,
        status: &'static str,
        bucket: String,
        product: String,
        granules: usize,
        total_bytes: u64,
        latency_behind_real_time_s: Option<i64>,
        latency_basis: &'static str,
        files: Vec<Granule>,
    }
    Ok(format!(
        "{}\n",
        serde_json::to_string_pretty(&Record {
            schema: LIST_SCHEMA,
            status: if granules.is_empty() { "EMPTY" } else { "READY" },
            bucket,
            product: options.product(),
            granules: granules.len(),
            total_bytes: granules.iter().map(|g| g.bytes).sum(),
            latency_behind_real_time_s: granules.iter().filter_map(|g| g.behind_scan_end_s).max(),
            latency_basis: "bucket LastModified minus the scan end in the filename",
            files: granules,
        })?
    ))
}

// ----------------------------------------------------------------- fetch

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FetchedGranule {
    #[serde(flatten)]
    granule: Granule,
    path: String,
    sha256: String,
    cache_hit: bool,
    fetched_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FetchRecord {
    schema: String,
    status: String,
    bucket: String,
    product: String,
    cache_dir: String,
    files: Vec<FetchedGranule>,
    total_bytes: u64,
    cache_hits: usize,
    latency_behind_real_time_s: Option<i64>,
    latency_basis: String,
    wall_s: f64,
}

fn cmd_fetch(options: &Options) -> Result<String, Box<dyn Error>> {
    let (bucket, granules) = list_granules(options)?;
    if granules.is_empty() {
        return Err(err(format!(
            "no {} granules for {} in the window; GOES-16 publishes none after its 2025 \
             replacement as GOES-East, so name --satellite G19 (East) or G18 (West)",
            options.product(),
            options.satellite()?.as_str()
        )));
    }
    let cache = options.cache();
    let agent = build_agent();
    let started = std::time::Instant::now();
    let mut files = Vec::with_capacity(granules.len());
    let mut hits = 0usize;
    for granule in granules {
        let object = S3Object {
            key: granule.key.clone(),
            size_bytes: granule.bytes,
            last_modified: granule.last_modified.clone().unwrap_or_default(),
        };
        let got = download_object(&agent, &bucket, &cache, &object, !options.no_cache)?;
        if got.cache_hit {
            hits += 1;
        }
        let bytes = std::fs::read(&got.path)
            .map_err(|e| err(format!("cannot re-read {}: {e}", got.path.display())))?;
        files.push(FetchedGranule {
            granule,
            path: rw_obs::absolute_uri(&got.path),
            sha256: hex_sha256(&bytes),
            cache_hit: got.cache_hit,
            fetched_at: seam_time(Utc::now()),
        });
    }
    let record = FetchRecord {
        schema: FETCH_SCHEMA.to_string(),
        status: "READY".to_string(),
        bucket,
        product: options.product(),
        cache_dir: rw_obs::absolute_uri(&cache),
        total_bytes: files.iter().map(|f| f.granule.bytes).sum(),
        cache_hits: hits,
        latency_behind_real_time_s: files.iter().filter_map(|f| f.granule.behind_scan_end_s).max(),
        latency_basis: "bucket LastModified minus the scan end in the filename".to_string(),
        files,
        wall_s: started.elapsed().as_secs_f64(),
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

// ----------------------------------------------------------------- table

fn amv_error(pressure_hpa: f64) -> f64 {
    if pressure_hpa > 700.0 {
        ERROR_AMV_LOW_M_S
    } else if pressure_hpa > 400.0 {
        ERROR_AMV_MID_M_S
    } else {
        ERROR_AMV_HIGH_M_S
    }
}

#[derive(Debug, Default, Serialize, Clone)]
struct Counters {
    files_read: usize,
    files_empty: usize,
    vectors_in_files: usize,
    dqf_good: usize,
    dqf_by_value: BTreeMap<String, usize>,
    dqf_fill: usize,
    fill_position: usize,
    fill_pressure: usize,
    fill_wind: usize,
    fill_time: usize,
    zenith_rejected: usize,
    pressure_out_of_range: usize,
    speed_out_of_range: usize,
    direction_out_of_range: usize,
    candidates: usize,
    thinned_away: usize,
    vectors_kept: usize,
    rows_by_variable: BTreeMap<String, usize>,
    vectors_by_layer: BTreeMap<&'static str, usize>,
    vectors_by_satellite_band: BTreeMap<String, usize>,
}

impl Counters {
    fn add_decode(&mut self, c: &DmwCounts) {
        if c.empty_granule {
            self.files_empty += 1;
        }
        self.vectors_in_files += c.vectors_in_file;
        self.dqf_good += c.dqf_good;
        for (k, v) in &c.dqf_by_value {
            *self.dqf_by_value.entry(k.to_string()).or_insert(0) += v;
        }
        self.dqf_fill += c.dqf_fill;
        self.fill_position += c.position_fill;
        self.fill_pressure += c.pressure_fill;
        self.fill_wind += c.wind_fill;
        self.fill_time += c.time_fill;
    }
}

#[derive(Debug, Clone)]
struct Candidate {
    satellite: String,
    band: u8,
    lat: f64,
    lon: f64,
    pressure_hpa: f64,
    speed: f64,
    direction: f64,
    zenith: f64,
    time: DateTime<Utc>,
    order: usize,
    /// The granule's creation stamp, receipt instant and digest.
    published: DateTime<Utc>,
    received: Option<DateTime<Utc>>,
    revision: String,
}

fn layer_name(pressure_hpa: f64) -> &'static str {
    if pressure_hpa > 700.0 {
        "low_below_700hPa"
    } else if pressure_hpa > 400.0 {
        "mid_700_to_400hPa"
    } else {
        "high_above_400hPa"
    }
}

/// One vector per (satellite, hour, lat box, lon box, pressure layer): the
/// smallest local zenith angle wins, ties by file order.
fn thin(candidates: Vec<Candidate>, thin_deg: f64, layer_hpa: f64) -> (Vec<Candidate>, usize) {
    let mut best: BTreeMap<(String, i64, i64, i64, i64), Candidate> = BTreeMap::new();
    let total = candidates.len();
    for c in candidates {
        let key = (
            c.satellite.clone(),
            c.time.timestamp().div_euclid(3600),
            ((c.lat + 90.0) / thin_deg).floor() as i64,
            ((c.lon + 180.0).rem_euclid(360.0) / thin_deg).floor() as i64,
            (c.pressure_hpa / layer_hpa).floor() as i64,
        );
        match best.get(&key) {
            Some(held)
                if (held.zenith, held.order) <= (c.zenith, c.order) => {}
            _ => {
                best.insert(key, c);
            }
        }
    }
    let mut kept: Vec<Candidate> = best.into_values().collect();
    kept.sort_by_key(|c| c.order);
    let dropped = total - kept.len();
    (kept, dropped)
}

#[derive(Serialize)]
struct SourceFile {
    path: String,
    bytes: usize,
    sha256: String,
    satellite: String,
    band: u8,
    scan_start: String,
    scan_end: String,
    vectors_in_file: usize,
    vectors_good: usize,
    behind_scan_end_s: Option<i64>,
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
    thinning: ThinningRecord,
    max_zenith_deg: f64,
    errors_by_layer_m_s: BTreeMap<&'static str, f64>,
    counters: Counters,
    files: Vec<SourceFile>,
    latency_behind_real_time_s: Option<i64>,
    latency_basis: &'static str,
}

#[derive(Serialize)]
struct ThinningRecord {
    enabled: bool,
    box_deg: f64,
    layer_hpa: f64,
    rule: &'static str,
}

fn cmd_table(options: &Options) -> Result<String, Box<dyn Error>> {
    if options.files.is_empty() {
        return Err(err("--files is required (one or more DMW granules)"));
    }
    let out = options
        .out
        .as_deref()
        .ok_or_else(|| err("--out FILE.csv is required"))?;
    if out.is_dir() {
        return Err(err(format!("--out {} is a directory; give the CSV path", out.display())));
    }
    let max_zenith = options.max_zenith_deg.unwrap_or(DEFAULT_MAX_ZENITH_DEG);
    let thin_deg = options.thin_deg.unwrap_or(DEFAULT_THIN_DEG);
    let layer_hpa = options.thin_layer_hpa.unwrap_or(DEFAULT_THIN_LAYER_HPA);
    let mut latencies: BTreeMap<String, i64> = BTreeMap::new();
    let mut fetched_at: BTreeMap<String, DateTime<Utc>> = BTreeMap::new();
    for path in &options.fetch_record {
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
        let name_of = |key: &str| -> Option<String> {
            Path::new(key).file_name().and_then(|n| n.to_str()).map(str::to_string)
        };
        for f in &record.files {
            let Some(name) = name_of(&f.granule.key) else { continue };
            if let Some(behind) = f.granule.behind_scan_end_s {
                latencies.insert(name.clone(), behind);
            }
            if let Ok(t) = parse_time(&format!("{}Z", f.fetched_at)) {
                fetched_at.insert(name, t);
            }
        }
    }

    let mut counters = Counters::default();
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut files = Vec::with_capacity(options.files.len());
    let mut order = 0usize;
    for path in &options.files {
        let bytes = std::fs::read(path).map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
        let sha = hex_sha256(&bytes);
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        let granule = read_dmw_granule(path)?;
        counters.files_read += 1;
        counters.add_decode(&granule.counts);
        let satellite = granule.filename.satellite.as_str().to_string();
        let received = fetched_at.get(&name).copied();
        let revision = revision_of(&sha);
        for v in &granule.vectors {
            if v.local_zenith_angle_deg.is_finite() && v.local_zenith_angle_deg > max_zenith {
                counters.zenith_rejected += 1;
                continue;
            }
            if !(PRESSURE_MIN_HPA..=PRESSURE_MAX_HPA).contains(&v.pressure_hpa) {
                counters.pressure_out_of_range += 1;
                continue;
            }
            if !(GROSS_WIND_M_S.0..=GROSS_WIND_M_S.1).contains(&v.speed_m_s) {
                counters.speed_out_of_range += 1;
                continue;
            }
            if !(0.0..=360.0).contains(&v.direction_deg) {
                counters.direction_out_of_range += 1;
                continue;
            }
            candidates.push(Candidate {
                satellite: satellite.clone(),
                band: granule.band,
                lat: v.latitude_deg,
                lon: v.longitude_deg,
                pressure_hpa: v.pressure_hpa,
                speed: v.speed_m_s,
                direction: v.direction_deg,
                zenith: if v.local_zenith_angle_deg.is_finite() { v.local_zenith_angle_deg } else { 90.0 },
                time: v.time,
                order,
                published: granule.filename.created_time_utc,
                received,
                revision: revision.clone(),
            });
            order += 1;
        }
        files.push(SourceFile {
            path: rw_obs::absolute_uri(path),
            bytes: bytes.len(),
            sha256: sha,
            satellite,
            band: granule.band,
            scan_start: seam_time(granule.filename.start_time_utc),
            scan_end: seam_time(granule.filename.end_time_utc),
            vectors_in_file: granule.counts.vectors_in_file,
            vectors_good: granule.counts.kept,
            behind_scan_end_s: latencies.get(&name).copied(),
        });
    }
    counters.candidates = candidates.len();
    let (kept, dropped) = if options.no_thin {
        (candidates, 0)
    } else {
        thin(candidates, thin_deg, layer_hpa)
    };
    counters.thinned_away = dropped;
    counters.vectors_kept = kept.len();

    let mut writer = TableWriter::new();
    for c in &kept {
        *counters.vectors_by_layer.entry(layer_name(c.pressure_hpa)).or_insert(0) += 1;
        *counters
            .vectors_by_satellite_band
            .entry(format!("{}-C{:02}", c.satellite, c.band))
            .or_insert(0) += 1;
        let (u, v) = wind_components(c.direction, c.speed);
        let error = amv_error(c.pressure_hpa);
        let base = TableRow {
            source: SOURCE.to_string(),
            station_id: format!("{}-C{:02}", c.satellite, c.band),
            latitude_deg: c.lat,
            longitude_deg: c.lon,
            elevation_m: rw_obs::table::isa_altitude_m(c.pressure_hpa * 100.0),
            level_pa: Some(c.pressure_hpa * 100.0),
            valid_time: c.time,
            variable: VAR_WIND_U.to_string(),
            value: u,
            error,
            provenance: RowProvenance {
                measurement: MEAS_AMV_ASSIGNED_PRESSURE,
                nominal_time: None,
                published_time: Some(c.published),
                received_time: c.received,
                revision: c.revision.clone(),
            },
        };
        let mut rv = base.clone();
        rv.variable = VAR_WIND_V.to_string();
        rv.value = v;
        writer.push(base, &mut counters.rows_by_variable);
        writer.push(rv, &mut counters.rows_by_variable);
    }
    let (rows, csv_sha, csv_bytes) = writer.write(out)?;
    let mut errors = BTreeMap::new();
    errors.insert("low_below_700hPa", ERROR_AMV_LOW_M_S);
    errors.insert("mid_700_to_400hPa", ERROR_AMV_MID_M_S);
    errors.insert("high_above_400hPa", ERROR_AMV_HIGH_M_S);
    let record = TableRecord {
        schema: TABLE_RECORD_SCHEMA,
        status: if rows > 0 { "READY" } else { "EMPTY" },
        source: SOURCE,
        table_schema: TABLE_SCHEMA,
        path: rw_obs::absolute_uri(out),
        sha256: csv_sha,
        rows,
        bytes: csv_bytes,
        thinning: ThinningRecord {
            enabled: !options.no_thin,
            box_deg: thin_deg,
            layer_hpa,
            rule: "one vector per satellite, hour, lat/lon box and pressure layer; smallest local \
                   zenith angle wins, ties by file order; bands pooled",
        },
        max_zenith_deg: max_zenith,
        errors_by_layer_m_s: errors,
        latency_behind_real_time_s: files.iter().filter_map(|f| f.behind_scan_end_s).max(),
        latency_basis: "bucket LastModified minus the scan end in the filename (from --fetch-record)",
        counters,
        files,
    };
    let text = format!("{}\n", serde_json::to_string_pretty(&record)?);
    std::fs::write(out.with_extension("json"), &text)
        .map_err(|e| err(format!("cannot write the table record: {e}")))?;
    Ok(text)
}

fn cmd_verify(options: &Options) -> Result<String, Box<dyn Error>> {
    let file = options.file.as_deref().ok_or_else(|| err("--file is required"))?;
    let record_path = options.record.as_deref().ok_or_else(|| err("--record is required"))?;
    let record: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(record_path)
            .map_err(|e| err(format!("cannot read {}: {e}", record_path.display())))?,
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
    use chrono::TimeZone;

    fn cand(lat: f64, lon: f64, p: f64, zen: f64, order: usize) -> Candidate {
        Candidate {
            satellite: "G19".into(),
            band: 14,
            lat,
            lon,
            pressure_hpa: p,
            speed: 10.0,
            direction: 270.0,
            zenith: zen,
            time: Utc.with_ymd_and_hms(2026, 9, 1, 18, 5, 0).unwrap(),
            order,
            published: Utc.with_ymd_and_hms(2026, 9, 1, 18, 24, 41).unwrap(),
            received: None,
            revision: "0123456789ab".into(),
        }
    }

    #[test]
    fn thinning_keeps_the_best_viewed_vector_per_box_and_layer() {
        let c = vec![
            cand(10.1, -80.1, 300.0, 40.0, 0),
            cand(10.2, -80.2, 310.0, 30.0, 1), // same box+layer, better zenith
            cand(10.2, -80.2, 360.0, 20.0, 2), // next layer
            cand(10.7, -80.2, 300.0, 50.0, 3), // next lat box
        ];
        let (kept, dropped) = thin(c, 0.5, 50.0);
        assert_eq!(dropped, 1);
        assert_eq!(kept.iter().map(|k| k.order).collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn errors_follow_the_layers() {
        assert_eq!(amv_error(850.0), ERROR_AMV_LOW_M_S);
        assert_eq!(amv_error(500.0), ERROR_AMV_MID_M_S);
        assert_eq!(amv_error(250.0), ERROR_AMV_HIGH_M_S);
    }

    #[test]
    fn abi_marker_names_the_contracts_it_pins() {
        assert!(ABI_MARKER.contains(FETCH_SCHEMA));
        assert!(ABI_MARKER.contains(TABLE_RECORD_SCHEMA));
        assert!(ABI_MARKER.contains(TABLE_SCHEMA));
    }
}
