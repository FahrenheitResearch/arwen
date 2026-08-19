//! `rw_opera` — ArWen's European radar-composite front door.
//!
//! Lists, fetches and decodes EUMETNET OPERA composite reflectivity from the
//! MeteoGate OGC-EDR API, and writes the observed field as a
//! `gpuwm-obs.obs-grid.v1` pack — the same container, quantity id and units
//! `rw_mrms` writes, so every consumer downstream of the seam reads a
//! European frame and a North-American one with one code path.
//!
//! The endpoint, the URL shape and the ODIM download are `rustwx-io`'s: this
//! bin calls `eumetnet_opera_dbzh_coverage_url`,
//! `fetch_eumetnet_opera_dbzh_coverage` and `fetch_eumetnet_opera_odim_h5`
//! rather than re-deriving any of them, and inherits that module's
//! host-allowlist check on the download URL.
//!
//! Three facts about this archive shape the rest of the bin. All three were
//! measured against a live frame (`OPERA@20260812T1930@0@DBZH.h5`,
//! 3800x4400, 16 720 000 cells) rather than assumed:
//!
//! * **Two sentinels, and they mean opposite things.** `/dataset1/data1/what`
//!   declares `nodata = -9999000` and `undetect = -8888000`. On that frame
//!   `nodata` covered 49.7 % of cells and `undetect` 46.2 %, with 4.1 %
//!   carrying a measurement. `nodata` is *no radar coverage* — genuinely
//!   unobserved, and it belongs under the validity mask. `undetect` is *no
//!   echo* — the network looked and found nothing, which is an observation,
//!   and on any given frame it is the single most common true one. Collapsing
//!   the two would delete every correct negative a skill score is built on.
//!   This is the same trap `rw_mrms` documents for `-999` versus `-99`, in a
//!   different archive's spelling.
//! * **The grid is ellipsoidal LAEA, and it says so.** `/where/projdef`
//!   carries `+ellps=WGS84`. Inverting it on a sphere displaces the northern
//!   corners by ~0.2 deg of longitude — about 9 km at 67 N, nine cells on a
//!   1 km grid. The file states its own four corner coordinates, so the
//!   geometry has an oracle inside it; this bin computes the corners it
//!   derived and refuses the frame when they disagree with the declared ones
//!   (see [`CORNER_TOLERANCE_DEG`]). A georeference nobody checked is the
//!   kind of wrong number that looks like a right one for a whole campaign.
//! * **Part of the composite is extrapolated, not observed.** `/how/comment`
//!   on the measured frame states that certain DBZH volumes are obtained by
//!   Lucas-Kanade advection at Meteo France. That is a statement about what
//!   the numbers are, so it travels into the pack's provenance instead of
//!   being dropped at decode.
//!
//! ```text
//! rw_opera coverage --start 2026-08-12T19:00:00Z --end 2026-08-12T19:30:00Z
//! rw_opera nearest  --valid-time 2026-08-12T19:30:00Z [--window-seconds 300]
//! rw_opera fetch    --start ... --end ... --out DIR
//! rw_opera decode   --file F.h5 --out F.obspack [--bbox W,S,E,N]
//! rw_opera grid     --file F.h5 --out grid.obspack [--bbox W,S,E,N]
//! rw_opera verify   --pack F.obspack
//! ```

use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

use hdf5_reader::{Datatype, Hdf5File};
use rw_nexrad::s3::parse_time;
use rw_obs::pack::{
    decode_pack, payload_digest, validate_arrays, write_pack, ArrayEntry, PayloadBuilder,
    GEO_SCHEMA, GRID_SCHEMA,
};
use rw_obs::seam::{
    seam_bounds, seam_time, wrap_longitude, Provenance, QUANTITY_COMPOSITE_REFLECTIVITY, UNITS_DBZ,
};
use rw_obs::{err, hex_sha256};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from, embedded so the gpuwm release cut can prove a
/// staged bridge matches the commit being released by reading bytes
/// alone (`tools/build_bridge_bundle.py pin --source-rev`).  `build.rs`
/// injects the value; `main` references the constant so the linker
/// cannot discard it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

/// The source label a decoded frame carries into the seam's provenance.
/// Generic: it names the network, not a case, a country or an event.
const SOURCE_LABEL: &str = "opera";

/// The product label. `MAX` is the ODIM product code the measured frame
/// declared in `/dataset1/what/product`; the composite is a column maximum,
/// which is what makes it the European counterpart of the MRMS composite
/// this pipeline already scores against.
const DEFAULT_PRODUCT: &str = "opera-comp-dbzh-max";

/// What a no-echo cell is worth as a number. Below every registered FSS
/// threshold and below the -32.0 dBZ floor real data was measured at on a
/// live frame, while staying inside the seam's [-40, 100] bound. Same value
/// `rw_mrms` uses, so a European frame and a North-American one put their
/// correct negatives at the same number.
const DEFAULT_NO_ECHO_DBZ: f64 = -35.0;

/// What a masked cell is filled with. The scorer never reads under a false
/// mask, but its contract requires the number to be finite anyway.
const MASKED_FILL_DBZ: f64 = -35.0;

/// How far a corner this bin derived may sit from the corner the file
/// declares, in degrees, before the frame is refused.
///
/// Measured agreement on a live frame was exact to the printed precision at
/// all four corners (< 1e-9 deg). The spherical inversion this check exists
/// to catch misses by 2.2e-1 deg. A 1e-4 deg ceiling — about 11 m — is three
/// orders of magnitude clear of the noise and three orders inside the error,
/// so it is a real screen rather than a formality.
const CORNER_TOLERANCE_DEG: f64 = 1.0e-4;

/// The projection the composite is published on. Any other `+proj` is
/// refused rather than approximated.
const REQUIRED_PROJECTION: &str = "+proj=laea";

const COVERAGE_SCHEMA: &str = "gpuwm-obs.opera-coverage.v1";
const NEAREST_SCHEMA: &str = "gpuwm-obs.opera-nearest.v1";
const FETCH_SCHEMA: &str = "gpuwm-obs.opera-fetch.v1";
const DECODE_SCHEMA: &str = "gpuwm-obs.opera-decode.v1";
const GRIDCMD_SCHEMA: &str = "gpuwm-obs.opera-grid.v1";
const VERIFY_SCHEMA: &str = "gpuwm-obs.opera-verify.v1";

/// The exact `--abi` line the Python bridge pins.
///
/// It names the record contract, not a version: a rebuilt-but-unchanged bin
/// still matches, and a bin whose records changed shape does not. The
/// sentinel words are in it deliberately — a build that stopped separating
/// no-coverage from no-echo would be a different instrument wearing the same
/// name, and the bridge must not accept it silently.
const ABI_MARKER: &str = "gpuwm-obs.opera-fetch.v1\tcollection\twindow\tlinks\tbytes\t\
sha256\tgpuwm-obs.obs-grid.v1\tcomposite_reflectivity\tdBZ\tno_coverage\tno_echo\tcorner_check";

const USAGE: &str = "\
usage: rw_opera <coverage|nearest|fetch|decode|grid|verify> [OPTIONS]
       rw_opera --version | --help | --abi

  coverage  report the ODIM frames a window resolves to, moving no payload
  nearest   report the one frame nearest a valid time, refusing a distant match
  fetch     download frames and print a fetch record with a sha256 per file
  decode    turn one ODIM frame into a `gpuwm-obs.obs-grid.v1` pack
  grid      write the pack's latitude/longitude once, as a `gpuwm-obs.obs-geo.v1`
  verify    re-read a pack and re-prove its header, index and payload digest

window options
  --start TIME            window start, 2026-08-12T19:00:00Z or 20260812T190000
  --end TIME              window end (inclusive)
  --valid-time TIME       nearest: the instant to match
  --window-seconds N      nearest: refuse a match further than this from
                          --valid-time. Default 300 (the composite cadence
                          measured live was one frame every 5 minutes)
  --limit N               keep at most the first N frames
  --out PATH              fetch: a directory. decode/grid: the pack file

decode options
  --file FILE             one ODIM HDF5 frame on disk
  --bbox W,S,E,N          keep only cells inside this lon/lat box. Strongly
                          advised: the published grid is 3800x4400, which is
                          134 MB of float64 values per frame and 267 MB of
                          coordinates
  --no-echo-dbz VALUE     what an `undetect` no-echo cell is worth. Default -35

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
            eprintln!("rw_opera: {error}");
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
        "--version" | "-V" => return Ok(format!("rw_opera {VERSION}\n")),
        "--abi" => return Ok(format!("{ABI_MARKER}\n")),
        _ => {}
    }
    let options = Options::parse(&args[1..])?;
    match first.as_str() {
        "coverage" => cmd_coverage(&options),
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
    start: Option<String>,
    end: Option<String>,
    valid_time: Option<String>,
    window_seconds: Option<i64>,
    limit: Option<usize>,
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

    fn no_echo_dbz(&self) -> f64 {
        self.no_echo_dbz.unwrap_or(DEFAULT_NO_ECHO_DBZ)
    }
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

    fn contains(&self, lat: f64, lon: f64) -> bool {
        lat >= self.south && lat <= self.north && lon >= self.west && lon <= self.east
    }
}

// ------------------------------------------------------------ the archive

/// One ODIM frame the coverage listing offered.
#[derive(Debug, Clone)]
struct Frame {
    href: String,
    filename: String,
    valid_time: DateTime<Utc>,
    length: Option<u64>,
}

/// Read the nominal valid time out of `OPERA@YYYYMMDDTHHMM@0@DBZH.h5`.
///
/// The published cadence is five minutes and the stamp is on the minute, but
/// nothing here rounds: the stamp is the frame's own claim about itself, and
/// the decoder cross-checks it against the frame's `/what/date` and
/// `/what/time` so a mislabelled download cannot be packed under a name it
/// disagrees with.
fn parse_frame_name(name: &str) -> Option<DateTime<Utc>> {
    let mut fields = name.strip_suffix(".h5")?.split('@');
    if fields.next()? != "OPERA" {
        return None;
    }
    let stamp = fields.next()?;
    let naive = NaiveDateTime::parse_from_str(stamp, "%Y%m%dT%H%M").ok()?;
    Some(Utc.from_utc_datetime(&naive))
}

fn frame_filename(href: &str) -> &str {
    href.rsplit('/').next().unwrap_or(href)
}

/// Spell a window the way the OGC-EDR endpoint reads it.
fn datetime_range(start: DateTime<Utc>, end: DateTime<Utc>) -> String {
    format!(
        "{}Z/{}Z",
        start.format("%Y-%m-%dT%H:%M:%S"),
        end.format("%Y-%m-%dT%H:%M:%S")
    )
}

/// Every ODIM frame the collection offers inside a window, in time order.
///
/// The endpoint answers a range, so the window is asked once rather than
/// walked. Links whose filename does not parse as a frame stamp are skipped
/// rather than guessed at: the collection also carries documentation and
/// notification links, and a link this bin cannot date is a link it cannot
/// place in a window.
fn list_frames(
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> Result<(Vec<Frame>, String), Box<dyn Error>> {
    let range = datetime_range(start, end);
    let coverage = rustwx_io::fetch_eumetnet_opera_dbzh_coverage(&range)
        .map_err(|e| err(format!("OPERA coverage request failed: {e}")))?;
    let mut frames = Vec::new();
    for link in &coverage.download_links {
        let filename = frame_filename(&link.href).to_string();
        let Some(valid_time) = parse_frame_name(&filename) else {
            continue;
        };
        if valid_time >= start && valid_time <= end {
            frames.push(Frame {
                href: link.href.clone(),
                filename,
                valid_time,
                length: link.length,
            });
        }
    }
    frames.sort_by(|a, b| {
        a.valid_time
            .cmp(&b.valid_time)
            .then_with(|| a.href.cmp(&b.href))
    });
    Ok((frames, range))
}

#[derive(Serialize)]
struct WindowRecord {
    collection: &'static str,
    datetime_range: String,
    start: String,
    end: String,
}

fn window_record(start: DateTime<Utc>, end: DateTime<Utc>, range: String) -> WindowRecord {
    WindowRecord {
        collection: "eu-eumetnet-weather-radar/observations/0-20010-0-OPERA",
        datetime_range: range,
        start: seam_time(start),
        end: seam_time(end),
    }
}

#[derive(Serialize)]
struct FrameRecord {
    href: String,
    filename: String,
    valid_time: String,
    length_bytes: Option<u64>,
}

impl From<&Frame> for FrameRecord {
    fn from(frame: &Frame) -> Self {
        Self {
            href: frame.href.clone(),
            filename: frame.filename.clone(),
            valid_time: seam_time(frame.valid_time),
            length_bytes: frame.length,
        }
    }
}

fn cmd_coverage(options: &Options) -> Result<String, Box<dyn Error>> {
    let (start, end) = options.window()?;
    let (frames, range) = list_frames(start, end)?;
    let matched = frames.len();
    let kept: Vec<Frame> = match options.limit {
        Some(limit) => frames.into_iter().take(limit).collect(),
        None => frames,
    };

    #[derive(Serialize)]
    struct CoverageRecord {
        schema: &'static str,
        status: &'static str,
        window: WindowRecord,
        matched_frames: usize,
        total_bytes: u64,
        frames: Vec<FrameRecord>,
    }

    let record = CoverageRecord {
        schema: COVERAGE_SCHEMA,
        status: if kept.is_empty() { "EMPTY" } else { "READY" },
        window: window_record(start, end, range),
        matched_frames: matched,
        total_bytes: kept.iter().filter_map(|f| f.length).sum(),
        frames: kept.iter().map(FrameRecord::from).collect(),
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
    let window = options.window_seconds.unwrap_or(300);
    let span = chrono::Duration::seconds(window);
    let (frames, range) = list_frames(target - span, target + span)?;
    let best = frames
        .iter()
        .min_by_key(|frame| (frame.valid_time - target).num_seconds().abs())
        .ok_or_else(|| {
            err(format!(
                "no OPERA composite within {window} s of {}; refusing rather than reaching \
                 further, because a distant frame scored as coincident is a wrong number that \
                 looks like a right one",
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
        window: window_record(target - span, target + span, range),
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
    let out = options
        .out
        .as_deref()
        .ok_or_else(|| err("--out is required (a directory for the downloaded frames)"))?;
    std::fs::create_dir_all(out)
        .map_err(|e| err(format!("cannot create {}: {e}", out.display())))?;
    let (frames, range) = list_frames(start, end)?;
    let matched = frames.len();
    let kept: Vec<Frame> = match options.limit {
        Some(limit) => frames.into_iter().take(limit).collect(),
        None => frames,
    };
    if kept.is_empty() {
        return Err(err(format!(
            "the OPERA collection offered no composite for {} .. {}",
            seam_time(start),
            seam_time(end)
        )));
    }

    #[derive(Serialize)]
    struct FileRecord {
        href: String,
        path: String,
        filename: String,
        valid_time: String,
        bytes: u64,
        sha256: String,
        fetched_at: String,
    }

    #[derive(Serialize)]
    struct FetchRecord {
        schema: &'static str,
        status: &'static str,
        window: WindowRecord,
        matched_frames: usize,
        out_dir: String,
        files: Vec<FileRecord>,
        total_bytes: u64,
    }

    let mut files = Vec::new();
    for frame in &kept {
        // The URL host allowlist lives in `rustwx-io`; calling through it
        // rather than around it is what keeps a coverage document that named
        // some other host from being downloaded here.
        let bytes = rustwx_io::fetch_eumetnet_opera_odim_h5(&frame.href)
            .map_err(|e| err(format!("cannot download {}: {e}", frame.href)))?;
        let target = safe_output_path(out, &frame.filename)?;
        std::fs::write(&target, &bytes)
            .map_err(|e| err(format!("cannot write {}: {e}", target.display())))?;
        files.push(FileRecord {
            href: frame.href.clone(),
            path: rw_obs::absolute_uri(&target),
            filename: frame.filename.clone(),
            valid_time: seam_time(frame.valid_time),
            bytes: bytes.len() as u64,
            sha256: hex_sha256(&bytes),
            fetched_at: seam_time(Utc::now()),
        });
    }
    let record = FetchRecord {
        schema: FETCH_SCHEMA,
        status: "READY",
        window: window_record(start, end, range),
        matched_frames: matched,
        out_dir: out.to_string_lossy().to_string(),
        total_bytes: files.iter().map(|f| f.bytes).sum(),
        files,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

/// `dir/name`, refusing any `name` that is not one ordinary file name.
///
/// The name comes from a remote document. A name carrying a separator or a
/// parent reference would put downloaded bytes somewhere this bin never
/// offered to manage, so it fails closed rather than being sanitized into
/// something that looks close enough.
fn safe_output_path(dir: &Path, name: &str) -> Result<PathBuf, Box<dyn Error>> {
    let ordinary = !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains('/')
        && !name.contains('\\')
        && !name.contains(':');
    if !ordinary {
        return Err(err(format!(
            "the coverage document offered a frame named {name:?}, which is not one ordinary \
             file name; refusing rather than writing bytes outside {}",
            dir.display()
        )));
    }
    Ok(dir.join(name))
}

// ------------------------------------------------------------- the fields

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GridSpec {
    kind: String,
    projdef: String,
    nx: usize,
    ny: usize,
    /// Index range kept out of the source grid, so a subset states what it
    /// is a subset of rather than pretending to be a whole grid.
    source_nx: usize,
    source_ny: usize,
    i_start: usize,
    j_start: usize,
    xscale_m: f64,
    yscale_m: f64,
}

/// What the frame said its corners are, what this bin derived, and how far
/// apart they came out.
///
/// Recorded rather than merely checked: a reader three months later should
/// be able to see that the georeference was proved and by how much margin,
/// without re-running anything.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CornerCheck {
    tolerance_deg: f64,
    max_offset_deg: f64,
    declared: Vec<[f64; 2]>,
    derived: Vec<[f64; 2]>,
    corners: Vec<String>,
    inversion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SentinelReport {
    nodata_raw: Option<f64>,
    undetect_raw: Option<f64>,
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
    corner_check: CornerCheck,
    sentinels: SentinelReport,
    /// What the frame says about how it was produced. The measured frame's
    /// `/how/comment` declared that part of the field is advection-
    /// extrapolated rather than observed, which is a statement about the
    /// numbers and therefore belongs beside them.
    production: ProductionNotes,
    value_min_dbz: f64,
    value_max_dbz: f64,
    arrays: std::collections::BTreeMap<String, ArrayEntry>,
    payload_bytes: usize,
    content_sha256: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ProductionNotes {
    object: Option<String>,
    product: Option<String>,
    prodname: Option<String>,
    source: Option<String>,
    odim_version: Option<String>,
    creator_name: Option<String>,
    institution: Option<String>,
    license: Option<String>,
    comment: Option<String>,
    /// How many contributing radars the frame's `/how/nodes` listed. A count
    /// rather than the list: the list is a kilobyte of station ids, and the
    /// number is the part a reader compares between frames.
    contributing_nodes: Option<usize>,
    /// Window the composite accumulated over, from `/dataset1/what`.
    interval_start: Option<String>,
    interval_end: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeoPackMeta {
    schema: String,
    status: String,
    source_product: String,
    grid: GridSpec,
    corner_check: CornerCheck,
    arrays: std::collections::BTreeMap<String, ArrayEntry>,
    payload_bytes: usize,
    content_sha256: String,
}

/// The decoded field, already subset and already in seam conventions.
struct Field {
    grid: GridSpec,
    corner_check: CornerCheck,
    latitude: Vec<f64>,
    longitude: Vec<f64>,
    values: Vec<f64>,
    valid: Vec<bool>,
    sentinels: SentinelReport,
    production: ProductionNotes,
    valid_time: DateTime<Utc>,
    value_min: f64,
    value_max: f64,
}

/// The `/where` block, which is the frame's own statement of its geometry.
#[derive(Debug, Clone)]
struct OdimGridMeta {
    projdef: String,
    xsize: usize,
    ysize: usize,
    xscale_m: f64,
    yscale_m: f64,
    /// LL, UL, UR, LR as `(lat, lon)`, in that order.
    corners: [(f64, f64); 4],
}

const CORNER_NAMES: [&str; 4] = ["LL", "UL", "UR", "LR"];

// -------------------------------------------------------- the projection

/// Inverse Lambert azimuthal equal-area on an ellipsoid (Snyder, *Map
/// Projections — A Working Manual*, eqs. 24-30 .. 24-33 with the authalic
/// latitude series 3-18).
///
/// The composite declares `+ellps=WGS84`, so this is the inversion the
/// declaration asks for. It is written out here rather than borrowed because
/// the spherical form is the one available upstream, and the difference
/// between them is nine cells at the top of this particular grid — see the
/// module docstring, and [`CORNER_TOLERANCE_DEG`] for the screen that keeps
/// the distinction honest instead of merely asserted.
struct Laea {
    lat0: f64,
    lon0: f64,
    false_easting: f64,
    false_northing: f64,
    /// First eccentricity squared. The only ellipsoid constant the inverse
    /// still needs once `rq`, `beta0` and `d` are precomputed.
    e2: f64,
    rq: f64,
    beta0: f64,
    d: f64,
}

/// WGS84, the ellipsoid the composite's `projdef` names.
const WGS84_A: f64 = 6_378_137.0;
const WGS84_F: f64 = 1.0 / 298.257_223_563;

impl Laea {
    fn new(lat0: f64, lon0: f64, false_easting: f64, false_northing: f64) -> Self {
        let e2 = WGS84_F * (2.0 - WGS84_F);
        let e = e2.sqrt();
        let q = |phi: f64| -> f64 {
            let s = phi.sin();
            (1.0 - e2) * (s / (1.0 - e2 * s * s) - (1.0 / (2.0 * e)) * ((1.0 - e * s) / (1.0 + e * s)).ln())
        };
        let qp = q(std::f64::consts::FRAC_PI_2);
        let rq = WGS84_A * (qp / 2.0).sqrt();
        let beta0 = (q(lat0) / qp).asin();
        let d = WGS84_A * lat0.cos() / (1.0 - e2 * lat0.sin().powi(2)).sqrt()
            / (rq * beta0.cos());
        Self {
            lat0,
            lon0,
            false_easting,
            false_northing,
            e2,
            rq,
            beta0,
            d,
        }
    }

    /// Authalic latitude to geodetic latitude, Snyder 3-18.
    fn geodetic_from_authalic(&self, beta: f64) -> f64 {
        let e2 = self.e2;
        let e4 = e2 * e2;
        let e6 = e4 * e2;
        beta + (e2 / 3.0 + 31.0 * e4 / 180.0 + 517.0 * e6 / 5040.0) * (2.0 * beta).sin()
            + (23.0 * e4 / 360.0 + 251.0 * e6 / 3780.0) * (4.0 * beta).sin()
            + (761.0 * e6 / 45360.0) * (6.0 * beta).sin()
    }

    /// Projected metres to `(lat_deg, lon_deg)`.
    fn inverse(&self, x: f64, y: f64) -> (f64, f64) {
        let dx = x - self.false_easting;
        let dy = y - self.false_northing;
        let rho = ((dx / self.d).powi(2) + (self.d * dy).powi(2)).sqrt();
        if rho <= f64::EPSILON {
            return (
                self.lat0.to_degrees(),
                normalize_longitude(self.lon0.to_degrees()),
            );
        }
        let ce = 2.0 * (rho / (2.0 * self.rq)).clamp(-1.0, 1.0).asin();
        let sin_ce = ce.sin();
        let cos_ce = ce.cos();
        let beta = (cos_ce * self.beta0.sin() + self.d * dy * sin_ce * self.beta0.cos() / rho)
            .clamp(-1.0, 1.0)
            .asin();
        let lambda = self.lon0
            + (dx * sin_ce).atan2(
                self.d * rho * self.beta0.cos() * cos_ce
                    - self.d * self.d * dy * self.beta0.sin() * sin_ce,
            );
        (
            self.geodetic_from_authalic(beta).to_degrees(),
            normalize_longitude(lambda.to_degrees()),
        )
    }
}

fn normalize_longitude(lon_deg: f64) -> f64 {
    wrap_longitude(lon_deg)
}

fn projdef_value(projdef: &str, key: &str) -> Option<f64> {
    projdef
        .split_whitespace()
        .find_map(|part| part.strip_prefix(key)?.parse::<f64>().ok())
}

/// Build the projection the frame declares, and prove it against the frame's
/// own corner coordinates before any cell is placed with it.
///
/// The four corners in `/where` are the grid's outer edges — index `(0,0)`,
/// `(0,ny)`, `(nx,ny)`, `(nx,0)` in projected space — not cell centres, and
/// the check is written against the edges for that reason.
fn projection_for(meta: &OdimGridMeta) -> Result<(Laea, CornerCheck), Box<dyn Error>> {
    if !meta
        .projdef
        .split_whitespace()
        .any(|part| part == REQUIRED_PROJECTION)
    {
        return Err(err(format!(
            "the composite declares projection {:?}; this front door places cells with the \
             {REQUIRED_PROJECTION} inversion and refuses to approximate any other",
            meta.projdef
        )));
    }
    let need = |key: &str| -> Result<f64, Box<dyn Error>> {
        projdef_value(&meta.projdef, key)
            .ok_or_else(|| err(format!("missing {key} in projdef {:?}", meta.projdef)))
    };
    let projection = Laea::new(
        need("+lat_0=")?.to_radians(),
        need("+lon_0=")?.to_radians(),
        need("+x_0=")?,
        need("+y_0=")?,
    );

    let nx = meta.xsize as f64;
    let ny = meta.ysize as f64;
    // LL, UL, UR, LR in the projected frame the grid is indexed on: x grows
    // east from the western edge, y grows south from the northern edge.
    let edges = [
        (0.0, -ny * meta.yscale_m),
        (0.0, 0.0),
        (nx * meta.xscale_m, 0.0),
        (nx * meta.xscale_m, -ny * meta.yscale_m),
    ];
    let mut derived = Vec::with_capacity(4);
    let mut declared = Vec::with_capacity(4);
    let mut worst = 0.0f64;
    for (index, (x, y)) in edges.iter().enumerate() {
        let (lat, lon) = projection.inverse(*x, *y);
        let (want_lat, want_lon) = meta.corners[index];
        worst = worst.max((lat - want_lat).abs()).max((lon - want_lon).abs());
        derived.push([lat, lon]);
        declared.push([want_lat, want_lon]);
    }
    if !worst.is_finite() || worst > CORNER_TOLERANCE_DEG {
        return Err(err(format!(
            "the grid this front door derived misses the corners the frame declares by up to \
             {worst:.6} deg, past the {CORNER_TOLERANCE_DEG} deg ceiling. The frame states its \
             own four corner coordinates, so this is the georeference failing its own oracle; \
             refusing, because every observation in the file would otherwise be assimilated at \
             the wrong place"
        )));
    }
    Ok((
        projection,
        CornerCheck {
            tolerance_deg: CORNER_TOLERANCE_DEG,
            max_offset_deg: worst,
            declared,
            derived,
            corners: CORNER_NAMES.iter().map(|s| s.to_string()).collect(),
            inversion: "lambert-azimuthal-equal-area/ellipsoidal/WGS84".to_string(),
        },
    ))
}

// ------------------------------------------------------------ ODIM reading

fn group_attr_f64(group: &hdf5_reader::group::Group, name: &str) -> Option<f64> {
    group.attribute(name).ok()?.read_as_f64().ok()
}

fn group_attr_string(group: &hdf5_reader::group::Group, name: &str) -> Option<String> {
    group.attribute(name).ok()?.read_string().ok()
}

fn require_f64(group: &hdf5_reader::group::Group, path: &str, name: &str) -> Result<f64, Box<dyn Error>> {
    group_attr_f64(group, name)
        .ok_or_else(|| err(format!("{path} carries no numeric attribute {name:?}")))
}

fn require_string(
    group: &hdf5_reader::group::Group,
    path: &str,
    name: &str,
) -> Result<String, Box<dyn Error>> {
    group_attr_string(group, name)
        .ok_or_else(|| err(format!("{path} carries no string attribute {name:?}")))
}

fn require_usize(group: &hdf5_reader::group::Group, path: &str, name: &str) -> Result<usize, Box<dyn Error>> {
    let value = require_f64(group, path, name)?;
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return Err(err(format!(
            "{path} attribute {name:?} is {value}, which is not a grid size"
        )));
    }
    Ok(value as usize)
}

fn open_group<'a>(file: &'a Hdf5File, path: &str) -> Result<hdf5_reader::group::Group, Box<dyn Error>> {
    file.group(path)
        .map_err(|e| err(format!("the frame has no {path} group: {e}")))
}

/// Every value in a dataset, widened to `f64`.
///
/// The dtype table mirrors the one `rustwx-io` applies to the same archive,
/// so a frame either module can read is read the same way by both. The
/// measured frame published `f64`; the integer forms are here because ODIM
/// permits them and a frame that switched would otherwise fail as "unknown"
/// rather than decode.
fn dataset_values_f64(dataset: &hdf5_reader::Dataset) -> Result<Vec<f64>, Box<dyn Error>> {
    macro_rules! read {
        ($t:ty) => {
            dataset
                .read_array::<$t>()
                .map_err(|e| err(format!("cannot read the composite dataset: {e}")))?
                .iter()
                .map(|&value| value as f64)
                .collect()
        };
    }
    Ok(match dataset.dtype() {
        Datatype::FloatingPoint { size: 4, .. } => read!(f32),
        Datatype::FloatingPoint { size: 8, .. } => read!(f64),
        Datatype::FixedPoint { size: 1, signed: true, .. } => read!(i8),
        Datatype::FixedPoint { size: 1, signed: false, .. } => read!(u8),
        Datatype::FixedPoint { size: 2, signed: true, .. } => read!(i16),
        Datatype::FixedPoint { size: 2, signed: false, .. } => read!(u16),
        Datatype::FixedPoint { size: 4, signed: true, .. } => read!(i32),
        Datatype::FixedPoint { size: 4, signed: false, .. } => read!(u32),
        other => {
            return Err(err(format!(
                "the composite dataset carries datatype {other:?}, which this front door does \
                 not widen to f64"
            )))
        }
    })
}

/// `/what/date` + `/what/time` as one instant.
fn odim_valid_time(what: &hdf5_reader::group::Group) -> Result<DateTime<Utc>, Box<dyn Error>> {
    let date = require_string(what, "/what", "date")?;
    let time = require_string(what, "/what", "time")?;
    let naive = NaiveDateTime::parse_from_str(&format!("{date}{time}"), "%Y%m%d%H%M%S")
        .map_err(|e| err(format!("/what date {date:?} time {time:?} is not an instant: {e}")))?;
    Ok(Utc.from_utc_datetime(&naive))
}

/// Which of the two sentinels — if either — a raw value is.
///
/// ODIM states the sentinels as raw storage values, so the comparison is
/// against the raw value and not against the calibrated one. The half-unit
/// window is what an integer-stored frame needs and what a float-stored one
/// tolerates; the measured frame declared `-9999000` and `-8888000`, which
/// are nine million apart and in no danger from it.
fn is_sentinel(raw: f64, sentinel: Option<f64>) -> bool {
    sentinel.is_some_and(|value| (raw - value).abs() < 0.5)
}

/// Decode one ODIM composite into a seam-ready field.
fn load_field(
    raw: &[u8],
    bbox: Option<BoundingBox>,
    no_echo_dbz: f64,
) -> Result<Field, Box<dyn Error>> {
    let file = Hdf5File::from_bytes(raw)
        .map_err(|e| err(format!("the frame is not readable HDF5: {e}")))?;

    let root_what = open_group(&file, "/what")?;
    let object = require_string(&root_what, "/what", "object")?;
    if object != "COMP" {
        return Err(err(format!(
            "the frame declares /what/object {object:?}; this front door writes a composite \
             grid and a polar volume is a different product with a different geometry"
        )));
    }
    let valid_time = odim_valid_time(&root_what)?;

    let where_group = open_group(&file, "/where")?;
    let meta = OdimGridMeta {
        projdef: require_string(&where_group, "/where", "projdef")?,
        xsize: require_usize(&where_group, "/where", "xsize")?,
        ysize: require_usize(&where_group, "/where", "ysize")?,
        xscale_m: require_f64(&where_group, "/where", "xscale")?,
        yscale_m: require_f64(&where_group, "/where", "yscale")?,
        corners: [
            (
                require_f64(&where_group, "/where", "LL_lat")?,
                require_f64(&where_group, "/where", "LL_lon")?,
            ),
            (
                require_f64(&where_group, "/where", "UL_lat")?,
                require_f64(&where_group, "/where", "UL_lon")?,
            ),
            (
                require_f64(&where_group, "/where", "UR_lat")?,
                require_f64(&where_group, "/where", "UR_lon")?,
            ),
            (
                require_f64(&where_group, "/where", "LR_lat")?,
                require_f64(&where_group, "/where", "LR_lon")?,
            ),
        ],
    };
    if !(meta.xscale_m.is_finite() && meta.xscale_m > 0.0 && meta.yscale_m.is_finite() && meta.yscale_m > 0.0)
    {
        return Err(err(format!(
            "the frame declares cell sizes {} x {} m, which are not lengths",
            meta.xscale_m, meta.yscale_m
        )));
    }
    let (projection, corner_check) = projection_for(&meta)?;

    let data_what = open_group(&file, "/dataset1/data1/what")?;
    let quantity = require_string(&data_what, "/dataset1/data1/what", "quantity")?;
    if quantity != "DBZH" {
        return Err(err(format!(
            "the frame carries quantity {quantity:?}; this front door writes composite \
             reflectivity and refuses to publish another moment under that name"
        )));
    }
    let gain = group_attr_f64(&data_what, "gain").unwrap_or(1.0);
    let offset = group_attr_f64(&data_what, "offset").unwrap_or(0.0);
    let nodata = group_attr_f64(&data_what, "nodata");
    let undetect = group_attr_f64(&data_what, "undetect");
    if undetect.is_none() {
        return Err(err(
            "the frame declares no `undetect` sentinel, so a cell the network looked at and \
             found empty cannot be told from a cell no radar covers. Every correct negative a \
             skill score is built on lives in that distinction, and guessing which one an \
             unmarked cell is would silently invent observations",
        ));
    }

    let dataset = file
        .dataset("/dataset1/data1/data")
        .map_err(|e| err(format!("the frame has no /dataset1/data1/data: {e}")))?;
    let shape = dataset.shape();
    let &[ny_u64, nx_u64] = shape else {
        return Err(err(format!(
            "the composite dataset must be 2D, got shape {shape:?}"
        )));
    };
    let source_nx = usize::try_from(nx_u64).map_err(|_| err("composite x size exceeds usize"))?;
    let source_ny = usize::try_from(ny_u64).map_err(|_| err("composite y size exceeds usize"))?;
    if source_nx != meta.xsize || source_ny != meta.ysize {
        return Err(err(format!(
            "the composite dataset is {source_nx}x{source_ny} but /where declares {}x{}",
            meta.xsize, meta.ysize
        )));
    }
    let raw_values = dataset_values_f64(&dataset)?;
    let cells = source_nx
        .checked_mul(source_ny)
        .ok_or_else(|| err("composite grid shape overflows"))?;
    if raw_values.len() != cells {
        return Err(err(format!(
            "the composite dataset holds {} values for a {source_nx}x{source_ny} grid",
            raw_values.len()
        )));
    }

    // Row 0 is the northern row: the corner check above proved it by
    // matching UL at projected (0, 0).
    let cell_lat_lon = |i: usize, j: usize| -> (f64, f64) {
        let x = ((i as f64) + 0.5) * meta.xscale_m;
        let y = -(((j as f64) + 0.5) * meta.yscale_m);
        projection.inverse(x, y)
    };

    // Latitude and longitude both vary along both axes on an azimuthal
    // projection, so the index rectangle a lon/lat box selects has to be
    // found by testing cells rather than by scanning one row and one column
    // the way a regular lat/lon grid allows.
    let (i_start, i_end, j_start, j_end) = match bbox {
        None => (0, source_nx, 0, source_ny),
        Some(box_) => {
            let mut i_lo = usize::MAX;
            let mut i_hi = 0usize;
            let mut j_lo = usize::MAX;
            let mut j_hi = 0usize;
            for j in 0..source_ny {
                for i in 0..source_nx {
                    let (lat, lon) = cell_lat_lon(i, j);
                    if box_.contains(lat, lon) {
                        i_lo = i_lo.min(i);
                        i_hi = i_hi.max(i);
                        j_lo = j_lo.min(j);
                        j_hi = j_hi.max(j);
                    }
                }
            }
            if i_lo == usize::MAX || j_lo == usize::MAX {
                return Err(err(format!(
                    "--bbox {},{},{},{} selects no cell of the {source_nx}x{source_ny} composite",
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
            let (lat, lon) = cell_lat_lon(i, j);
            latitude.push(lat);
            longitude.push(lon);
            let stored = raw_values[j * source_nx + i];
            if !stored.is_finite() || is_sentinel(stored, nodata) {
                no_coverage_cells += 1;
                values.push(MASKED_FILL_DBZ);
                valid.push(false);
            } else if is_sentinel(stored, undetect) {
                no_echo_cells += 1;
                values.push(no_echo_dbz);
                valid.push(true);
                value_min = value_min.min(no_echo_dbz);
                value_max = value_max.max(no_echo_dbz);
            } else {
                let dbz = stored * gain + offset;
                if !dbz.is_finite() {
                    return Err(err(format!(
                        "a cell calibrated to {dbz} with gain {gain} and offset {offset}; a \
                         non-finite reflectivity is not an observation and is not a mask"
                    )));
                }
                echo_cells += 1;
                values.push(dbz);
                valid.push(true);
                value_min = value_min.min(dbz);
                value_max = value_max.max(dbz);
            }
        }
    }

    let observed = no_echo_cells + echo_cells;
    if observed == 0 {
        return Err(err(
            "every cell of the selected region is unobserved; a field with no observation in it \
             cannot be scored, and scoring it as all-missing would silently contribute nothing \
             while looking like a frame",
        ));
    }
    let (low, high) = seam_bounds(QUANTITY_COMPOSITE_REFLECTIVITY).unwrap();
    if value_min < low || value_max > high {
        return Err(err(format!(
            "decoded reflectivity spans [{value_min:.3}, {value_max:.3}] dBZ, outside the seam \
             bound [{low}, {high}]"
        )));
    }

    let how = file.group("/how").ok();
    let dataset_what = file.group("/dataset1/what").ok();
    let production = ProductionNotes {
        object: Some(object),
        product: dataset_what
            .as_ref()
            .and_then(|g| group_attr_string(g, "product")),
        prodname: dataset_what
            .as_ref()
            .and_then(|g| group_attr_string(g, "prodname")),
        source: group_attr_string(&root_what, "source"),
        odim_version: group_attr_string(&root_what, "version"),
        creator_name: how.as_ref().and_then(|g| group_attr_string(g, "creator_name")),
        institution: how.as_ref().and_then(|g| group_attr_string(g, "institution")),
        license: how.as_ref().and_then(|g| group_attr_string(g, "license")),
        comment: how.as_ref().and_then(|g| group_attr_string(g, "comment")),
        contributing_nodes: how
            .as_ref()
            .and_then(|g| group_attr_string(g, "nodes"))
            .map(|list| list.split(',').filter(|s| !s.trim().is_empty()).count()),
        interval_start: dataset_what.as_ref().and_then(|g| {
            let date = group_attr_string(g, "startdate")?;
            let time = group_attr_string(g, "starttime")?;
            NaiveDateTime::parse_from_str(&format!("{date}{time}"), "%Y%m%d%H%M%S")
                .ok()
                .map(|n| seam_time(Utc.from_utc_datetime(&n)))
        }),
        interval_end: dataset_what.as_ref().and_then(|g| {
            let date = group_attr_string(g, "enddate")?;
            let time = group_attr_string(g, "endtime")?;
            NaiveDateTime::parse_from_str(&format!("{date}{time}"), "%Y%m%d%H%M%S")
                .ok()
                .map(|n| seam_time(Utc.from_utc_datetime(&n)))
        }),
    };

    Ok(Field {
        grid: GridSpec {
            kind: "projected_laea".to_string(),
            projdef: meta.projdef.clone(),
            nx,
            ny,
            source_nx,
            source_ny,
            i_start,
            j_start,
            xscale_m: meta.xscale_m,
            yscale_m: meta.yscale_m,
        },
        corner_check,
        latitude,
        longitude,
        values,
        valid,
        sentinels: SentinelReport {
            nodata_raw: nodata,
            undetect_raw: undetect,
            no_echo_value_dbz: no_echo_dbz,
            masked_fill_dbz: MASKED_FILL_DBZ,
            no_coverage_cells,
            no_echo_cells,
            echo_cells,
            masked_cells: no_coverage_cells,
            observed_fraction: observed as f64 / (nx * ny) as f64,
        },
        production,
        valid_time,
        value_min,
        value_max,
    })
}

// ------------------------------------------------------------- the packs

fn source_bytes(options: &Options) -> Result<(PathBuf, Vec<u8>, String), Box<dyn Error>> {
    let path = options
        .file
        .as_deref()
        .ok_or_else(|| err("--file is required (one ODIM frame on disk)"))?;
    let raw =
        std::fs::read(path).map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
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
    let field = load_field(&raw, options.bbox, options.no_echo_dbz())?;

    // The filename carries a stamp and so does the frame. They must agree:
    // a frame packed under a name it disagrees with would be scored against
    // the wrong hour, and neither statement is more authoritative than the
    // other when they differ.
    if let Some(named) = path.file_name().and_then(|n| n.to_str()).and_then(parse_frame_name) {
        if named != field.valid_time {
            return Err(err(format!(
                "the file is named for {} but its /what block declares {}; refusing rather than \
                 choosing one of two disagreeing statements about when this composite is valid",
                seam_time(named),
                seam_time(field.valid_time)
            )));
        }
    }

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
        valid_time: seam_time(field.valid_time),
        provenance: Provenance::new(
            SOURCE_LABEL,
            DEFAULT_PRODUCT,
            rw_obs::absolute_uri(&path),
            sha256,
            seam_time(Utc::now()),
        ),
        grid: field.grid.clone(),
        corner_check: field.corner_check.clone(),
        sentinels: field.sentinels.clone(),
        production: field.production.clone(),
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
        corner_check: &'a CornerCheck,
        sentinels: &'a SentinelReport,
        production: &'a ProductionNotes,
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
        corner_check: &meta.corner_check,
        sentinels: &meta.sentinels,
        production: &meta.production,
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
        source_product: DEFAULT_PRODUCT.to_string(),
        grid: field.grid.clone(),
        corner_check: field.corner_check.clone(),
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
        corner_check: &'a CornerCheck,
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
        corner_check: &meta.corner_check,
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
            "observation pack payload digest mismatch: metadata says {declared}, bytes hash to \
             {digest}"
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
                "pack declares schema {other:?}; this bin writes {GRID_SCHEMA:?} and {GEO_SCHEMA:?}"
            )))
        }
    };
    validate_arrays(&arrays, payload.len())?;

    #[derive(Serialize)]
    struct VerifyRecord {
        schema: &'static str,
        status: &'static str,
        pack_path: String,
        pack_schema: String,
        pack_bytes: usize,
        payload_bytes: usize,
        content_sha256: String,
        arrays: usize,
    }

    let record = VerifyRecord {
        schema: VERIFY_SCHEMA,
        status: "READY",
        pack_path: path.to_string_lossy().to_string(),
        pack_schema: schema,
        pack_bytes: bytes.len(),
        payload_bytes: payload.len(),
        content_sha256: digest,
        arrays: arrays.len(),
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `/where` block a live frame published on 2026-08-12. Every number
    /// is transcribed from `OPERA@20260812T1930@0@DBZH.h5`, so the tests
    /// below are assertions about the real archive rather than about a shape
    /// invented to make them pass.
    fn measured_meta() -> OdimGridMeta {
        OdimGridMeta {
            projdef: "+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 \
                      +units=m +ellps=WGS84"
                .to_string(),
            xsize: 3800,
            ysize: 4400,
            xscale_m: 1000.0,
            yscale_m: 1000.0,
            corners: [
                (31.7462153182675, -10.4345768386404),
                (67.0228327624372, -39.5357864125034),
                (67.6210371071631, 57.8119647501499),
                (31.987650276733, 29.421038635578),
            ],
        }
    }

    #[test]
    fn the_ellipsoidal_inversion_reproduces_the_corners_the_frame_declares() {
        let (_projection, check) = projection_for(&measured_meta()).expect("live geometry");
        assert!(
            check.max_offset_deg < 1.0e-6,
            "derived corners missed by {} deg: {:?} vs {:?}",
            check.max_offset_deg,
            check.derived,
            check.declared
        );
    }

    /// The screen this front door exists to keep honest. A sphere is what a
    /// reader reaches for when a projdef is skimmed rather than read, and on
    /// this grid it displaces the northern corners by about nine cells.
    #[test]
    fn a_spherical_inversion_of_the_same_grid_misses_by_far_more_than_the_ceiling() {
        let meta = measured_meta();
        const AUTHALIC_RADIUS_M: f64 = 6_371_007.181;
        let lat0 = 55.0f64.to_radians();
        let lon0 = 10.0f64.to_radians();
        let (x0, y0) = (1_950_000.0f64, -2_100_000.0f64);
        let spherical = |x: f64, y: f64| -> (f64, f64) {
            let dx = x - x0;
            let dy = y - y0;
            let rho = dx.hypot(dy);
            let c = 2.0 * (rho / (2.0 * AUTHALIC_RADIUS_M)).clamp(-1.0, 1.0).asin();
            let lat = (c.cos() * lat0.sin() + dy * c.sin() * lat0.cos() / rho).asin();
            let lon = lon0 + (dx * c.sin()).atan2(rho * lat0.cos() * c.cos() - dy * lat0.sin() * c.sin());
            (lat.to_degrees(), normalize_longitude(lon.to_degrees()))
        };
        // The upper-left corner, which is where the two inversions part
        // company hardest.
        let (lat, lon) = spherical(0.0, 0.0);
        let (want_lat, want_lon) = meta.corners[1];
        let worst = (lat - want_lat).abs().max((lon - want_lon).abs());
        assert!(
            worst > 100.0 * CORNER_TOLERANCE_DEG,
            "a spherical inversion missed by only {worst} deg, so this screen would not \
             distinguish it from the ellipsoidal one it exists to separate"
        );
    }

    #[test]
    fn a_projection_that_is_not_the_published_one_is_refused() {
        let mut meta = measured_meta();
        meta.projdef = "+proj=stere +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84".to_string();
        let outcome = projection_for(&meta);
        let message = outcome.err().expect("a foreign projection must refuse").to_string();
        assert!(message.contains("+proj=laea"), "{message}");
    }

    #[test]
    fn corners_that_disagree_with_the_grid_refuse_rather_than_being_averaged() {
        let mut meta = measured_meta();
        // Nudge one declared corner by a tenth of a degree — about a quarter
        // of the error a spherical inversion makes, and still far past the
        // ceiling.
        meta.corners[0].0 += 0.1;
        let message = projection_for(&meta)
            .err()
            .expect("a disagreeing corner must refuse")
            .to_string();
        assert!(message.contains("its own oracle"), "{message}");
    }

    #[test]
    fn frame_names_carry_the_stamp_the_window_is_filtered_on() {
        let parsed = parse_frame_name("OPERA@20260812T1930@0@DBZH.h5").expect("a live filename");
        assert_eq!(seam_time(parsed), "2026-08-12T19:30:00");
        assert!(parse_frame_name("OPERA@notatime@0@DBZH.h5").is_none());
        assert!(parse_frame_name("index.html").is_none());
        // The collection also serves documentation links; none of them may
        // be mistaken for a frame.
        assert!(parse_frame_name("docs").is_none());
    }

    #[test]
    fn the_two_sentinels_are_told_apart_by_the_values_the_archive_declares() {
        let nodata = Some(-9_999_000.0);
        let undetect = Some(-8_888_000.0);
        assert!(is_sentinel(-9_999_000.0, nodata));
        assert!(!is_sentinel(-9_999_000.0, undetect));
        assert!(is_sentinel(-8_888_000.0, undetect));
        assert!(!is_sentinel(-8_888_000.0, nodata));
        // A real measurement is neither.
        assert!(!is_sentinel(-32.0, nodata));
        assert!(!is_sentinel(-32.0, undetect));
        assert!(!is_sentinel(70.5, nodata));
    }

    #[test]
    fn the_edr_window_is_spelled_the_way_the_endpoint_reads_it() {
        let start = Utc.with_ymd_and_hms(2026, 8, 12, 19, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2026, 8, 12, 19, 30, 0).unwrap();
        assert_eq!(
            datetime_range(start, end),
            "2026-08-12T19:00:00Z/2026-08-12T19:30:00Z"
        );
    }

    #[test]
    fn a_frame_name_that_is_not_one_ordinary_name_never_becomes_a_path() {
        let dir = Path::new("frames");
        assert!(safe_output_path(dir, "OPERA@20260812T1930@0@DBZH.h5").is_ok());
        for hostile in ["../escape.h5", "a/b.h5", "", ".", "..", r"C:\x.h5"] {
            assert!(
                safe_output_path(dir, hostile).is_err(),
                "{hostile:?} was accepted as a file name"
            );
        }
    }

    #[test]
    fn the_abi_marker_names_the_distinctions_the_bridge_depends_on() {
        for token in [
            "gpuwm-obs.opera-fetch.v1",
            "gpuwm-obs.obs-grid.v1",
            "composite_reflectivity",
            "no_coverage",
            "no_echo",
            "corner_check",
        ] {
            assert!(ABI_MARKER.contains(token), "the ABI marker lost {token:?}");
        }
    }
}
