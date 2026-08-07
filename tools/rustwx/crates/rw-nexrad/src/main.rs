//! `rw_nexrad` — ArWen's NEXRAD Level-II front door.
//!
//! A fail-closed CLI with two jobs and no opinions beyond them:
//!
//! * **acquire** — list and download archived WSR-88D volumes for a
//!   `(site, time window)` from the anonymous `noaa-nexrad-level2` S3
//!   bucket, atomically, through a content-addressed cache;
//! * **decode** — validate one volume's Archive-II framing, run it through
//!   the vendored `wx-radar` parser, and transcribe the surviving sweeps
//!   into a flat `<f4` pack that the Python superob stage reads with
//!   `numpy.frombuffer`.
//!
//! It moves bytes and reports facts.  It does **not** superob, grid,
//! dealias, or quality-control: those are the Python stage's, and every
//! number here is the RDA's own.
//!
//! There are two acquisition routes and one decode path.  `list`/`fetch`
//! read the archive, which only publishes a volume once the volume has
//! ended.  `live-list`/`live-fetch` read the real-time chunk feed, which
//! publishes the same bytes as they are collected and can therefore hand
//! over a scan that is still in progress; [`live`] is where that route and
//! its refusals live.  Both produce an ordinary Archive-II file on disk, so
//! `decode` neither knows nor cares which one served it.
//!
//! ```text
//! rw_nexrad list       --site KTLX --start 2023-05-20T20:00:00Z --end 2023-05-20T20:30:00Z
//! rw_nexrad fetch      --site KTLX --start ... --end ... --out DIR [--limit N]
//! rw_nexrad live-list  --site KTLX
//! rw_nexrad live-fetch --site KTLX --out DIR [--allow-partial]
//! rw_nexrad decode     --volume FILE --out FILE.rdrpack [--moments REF,VEL]
//! rw_nexrad sites      [--site KTLX]
//! ```

use rw_nexrad::{decode, live, pack, s3};

use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;

use serde::Serialize;

use decode::DecodeRequest;
use pack::{decode_pack, parse_moment_filter, write_pack};
use s3::{
    build_agent, download_volume, iso8601, list_volumes_observed, normalize_bucket, normalize_site,
    parse_time, publish_volume, VolumeObject, ARCHIVE_OF_RECORD_BUCKET, DEFAULT_BUCKET,
    LIVE_DEFAULT_BUCKET,
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

pub const LIST_SCHEMA: &str = "gpuwm-obs.nexrad-list.v1";
pub const FETCH_SCHEMA: &str = "gpuwm-obs.nexrad-fetch.v1";
pub const DECODE_SCHEMA: &str = "gpuwm-obs.nexrad-decode.v1";
pub const SITES_SCHEMA: &str = "gpuwm-obs.nexrad-sites.v1";
pub const VERIFY_SCHEMA: &str = "gpuwm-obs.nexrad-verify.v1";
pub const LIVE_LIST_SCHEMA: &str = "gpuwm-obs.nexrad-live-list.v1";
pub const LIVE_FETCH_SCHEMA: &str = "gpuwm-obs.nexrad-live-fetch.v1";

/// The name a receipt calls each acquisition route, so "which feed served
/// this observation" is a value rather than an inference from a bucket name.
pub const ARCHIVE_FEED: &str = "archive-volumes";
pub const LIVE_FEED: &str = "live-chunks";

/// The exact `--abi` line the Python bridge pins, so a bin that drifted out
/// from under a pinned wrapper is caught at probe time rather than at parse
/// time three receipts later.
///
/// The live half is appended rather than folded in: a wrapper that needs
/// the real-time route needs a bin that has it, and the marker is the one
/// place that difference is stated before anything is run.
const ABI_MARKER: &str = "gpuwm-obs.nexrad-fetch.v1\tsite\twindow\tbucket\tvolumes\tbytes\t\
cache_hits\tsha256\tgpuwm-obs.radar-sweeps.v1\tgpuwm-obs.nexrad-live-fetch.v1\tfeed\t\
volume_id\tchunks\tcomplete\tlag_seconds";

const USAGE: &str = "\
usage: rw_nexrad <list|fetch|live-list|live-fetch|decode|verify|sites> [OPTIONS]
       rw_nexrad --version | --help | --abi

  list        report the Level-II volumes a (site, window) resolves to, moving no payload
  fetch       download those volumes and print a fetch record
  live-list   report what the real-time chunk feed holds for a site right now
  live-fetch  assemble the newest real-time chunks into an Archive-II volume
  decode      validate one volume and write a `gpuwm-obs.radar-sweeps.v1` pack
  verify      re-read a pack and re-prove its header, schema and payload digest
  sites       print the vendored NEXRAD site table (id, name, lat, lon, alt)

archive acquisition options (list, fetch)
  --site ID               four-character radar id, e.g. KTLX (data, not a gate)
  --bucket NAME           S3 bucket. Default: unidata-nexrad-level2, a mirror
                          of the same key space. It is the default on a
                          capability check, not on authority: probed
                          2026-07-30, noaa-nexrad-level2 -- nominally the
                          archive of record, and the complete one -- answered
                          HTTP 403 to BOTH anonymous ListObjectsV2 and
                          anonymous GET, while the mirror answered 200/206 for
                          keys from 2011 through 2026. Pass
                          --bucket noaa-nexrad-level2 to use the archive of
                          record once you have credentials, or if it starts
                          granting anonymous access again; a window the mirror
                          cannot cover is the other reason to reach for it.
                          Keys before ~2016 are gzipped in both and are read
                          transparently.
  --start TIME            window start, 2023-05-20T20:00:00Z or 20230520T200000
  --end TIME              window end (inclusive)
  --limit N               keep at most the first N volumes in the window
  --cache DIR             volume cache root (default: <out>/.cache, else ./.rw-nexrad-cache)
  --no-cache              always re-download, never read the cache
  --out DIR               fetch: publish the volumes here as well as the cache

real-time options (live-list, live-fetch)
  --site ID               four-character radar id (data, not a gate)
  --bucket NAME           chunk bucket. Live default: unidata-nexrad-level2-chunks,
                          a different KEY SPACE from the archive bucket, not
                          another mirror of it -- one object per LDM chunk under
                          {SITE}/{VOLUME_ID}/{YYYYMMDD}-{HHMMSS}-{NNN}-{S|I|E} --
                          which is why the live subcommands default to it and
                          list/fetch never do
  --volumes N             report the newest N volumes (default 1), walking back
                          through the retained volume ids
  --volume-id N           name the volume directory instead of discovering it
  --allow-partial         accept a scan that is still in progress. Without it a
                          volume with no end-of-volume chunk is refused, because
                          a partial volume is a real product and not a default
                          one. A partial is published as
                          {SITE}{YYYYMMDD}_{HHMMSS}_{FMT}_P{NNN} -- a name the
                          archive key parser refuses -- so it can never be
                          mistaken for a finished volume
  --min-chunks N          refuse an assembly shorter than N chunks (default 2).
                          The floor is 2 because chunk 1 carries only the volume
                          header and metadata messages: assembled alone it holds
                          no Message-31 radial and `decode` refuses it
  --out DIR               live-fetch: publish the assembled volume here
  --cache DIR             chunk + assembly cache root
  --no-cache              always re-download every chunk

decode options
  --volume FILE           one Level-II volume on disk
  --out FILE              pack destination (required)
  --moments LIST          comma-separated: REF,VEL (default), SW, ZDR, RHO, PHI, KDP
  --max-range-km KM       drop gates beyond this slant range (default 300)
  --max-elevation-deg DEG drop sweeps above this elevation (default 20)
  --site-latlon LAT,LON,ALT_M
                          place a site the vendored table does not know
  --censor-flags          emit a |u1 censor plane beside every moment plane
                          and declare the pack gpuwm-obs.radar-sweeps.v2. The
                          plane says WHY each NaN gate is not a number: 0
                          measured, 1 below threshold (the radar looked and
                          detected nothing -- a clear-air observation), 2
                          range folded (second-trip ambiguity, never usable
                          as clear air), 3 not collected (a radial that
                          carried no such moment). Without it the pack is
                          byte-identical to what this tool has always
                          written, which is why it is a flag and not the
                          default.

verify options
  --volume FILE           the pack to re-prove (--out is accepted too)

sites options
  --site ID               print just this site (exit 1 if unknown)
";

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(output) => {
            print!("{output}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("rw_nexrad: {err}");
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
        "--version" | "-V" => return Ok(format!("rw_nexrad {VERSION}\n")),
        "--abi" => return Ok(format!("{ABI_MARKER}\n")),
        _ => {}
    }
    let options = Options::parse(&args[1..])?;
    match first.as_str() {
        "list" => cmd_list(&options),
        "fetch" => cmd_fetch(&options),
        "live-list" => cmd_live(&options, false),
        "live-fetch" => cmd_live(&options, true),
        "decode" => cmd_decode(&options),
        "verify" => cmd_verify(&options),
        "sites" => cmd_sites(&options),
        other => Err(s3::boxed_error(format!(
            "unknown subcommand {other:?}\n\n{USAGE}"
        ))),
    }
}

#[derive(Debug, Default)]
struct Options {
    site: Option<String>,
    bucket: Option<String>,
    start: Option<String>,
    end: Option<String>,
    limit: Option<usize>,
    cache: Option<PathBuf>,
    no_cache: bool,
    out: Option<PathBuf>,
    volume: Option<PathBuf>,
    moments: Option<String>,
    max_range_km: Option<f64>,
    max_elevation_deg: Option<f64>,
    site_latlon: Option<(f64, f64, f64)>,
    volumes: Option<usize>,
    volume_id: Option<u32>,
    allow_partial: bool,
    min_chunks: Option<usize>,
    censor_flags: bool,
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
                    .ok_or_else(|| s3::boxed_error(format!("{flag} needs a value")))
            };
            match flag {
                "--site" => options.site = Some(value()?),
                "--bucket" => options.bucket = Some(value()?),
                "--start" => options.start = Some(value()?),
                "--end" => options.end = Some(value()?),
                "--limit" => {
                    let raw = value()?;
                    options.limit = Some(raw.parse::<usize>().map_err(|_| {
                        s3::boxed_error(format!("--limit expects a count, got {raw:?}"))
                    })?)
                }
                "--cache" => options.cache = Some(PathBuf::from(value()?)),
                "--no-cache" => options.no_cache = true,
                "--censor-flags" => options.censor_flags = true,
                "--out" => options.out = Some(PathBuf::from(value()?)),
                "--volume" => options.volume = Some(PathBuf::from(value()?)),
                "--moments" => options.moments = Some(value()?),
                "--max-range-km" => {
                    let raw = value()?;
                    options.max_range_km = Some(parse_positive(&raw, "--max-range-km")?)
                }
                "--max-elevation-deg" => {
                    let raw = value()?;
                    options.max_elevation_deg = Some(parse_positive(&raw, "--max-elevation-deg")?)
                }
                "--site-latlon" => {
                    let raw = value()?;
                    options.site_latlon = Some(parse_site_latlon(&raw)?)
                }
                "--volumes" => {
                    let raw = value()?;
                    let count = raw.parse::<usize>().map_err(|_| {
                        s3::boxed_error(format!("--volumes expects a count, got {raw:?}"))
                    })?;
                    if count == 0 {
                        return Err(s3::boxed_error("--volumes must be at least 1"));
                    }
                    options.volumes = Some(count)
                }
                "--volume-id" => {
                    let raw = value()?;
                    options.volume_id = Some(raw.parse::<u32>().map_err(|_| {
                        s3::boxed_error(format!(
                            "--volume-id expects a real-time volume directory number, got {raw:?}"
                        ))
                    })?)
                }
                "--allow-partial" => options.allow_partial = true,
                "--min-chunks" => {
                    let raw = value()?;
                    let count = raw.parse::<usize>().map_err(|_| {
                        s3::boxed_error(format!("--min-chunks expects a count, got {raw:?}"))
                    })?;
                    if count == 0 {
                        return Err(s3::boxed_error(
                            "--min-chunks must be at least 1; a zero-chunk assembly is an \
                             empty file, not a volume",
                        ));
                    }
                    options.min_chunks = Some(count)
                }
                other => {
                    return Err(s3::boxed_error(format!(
                        "unknown option {other:?}\n\n{USAGE}"
                    )))
                }
            }
            index += 1;
        }
        Ok(options)
    }

    fn bucket(&self) -> Result<String, Box<dyn Error>> {
        normalize_bucket(self.bucket.as_deref().unwrap_or(DEFAULT_BUCKET))
    }

    /// The real-time route's bucket.  Its default is the chunk feed, not
    /// the archive: they are different key spaces, and defaulting the live
    /// subcommands to the archive bucket would list a prefix that does not
    /// exist and report an empty sky.
    fn live_bucket(&self) -> Result<String, Box<dyn Error>> {
        normalize_bucket(self.bucket.as_deref().unwrap_or(LIVE_DEFAULT_BUCKET))
    }

    fn site_id(&self) -> Result<String, Box<dyn Error>> {
        normalize_site(
            self.site
                .as_deref()
                .ok_or_else(|| s3::boxed_error("--site is required"))?,
        )
    }

    fn window(&self) -> Result<(String, chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>), Box<dyn Error>> {
        let site = normalize_site(
            self.site
                .as_deref()
                .ok_or_else(|| s3::boxed_error("--site is required"))?,
        )?;
        let start = parse_time(
            self.start
                .as_deref()
                .ok_or_else(|| s3::boxed_error("--start is required"))?,
        )?;
        let end = parse_time(
            self.end
                .as_deref()
                .ok_or_else(|| s3::boxed_error("--end is required"))?,
        )?;
        if end < start {
            return Err(s3::boxed_error(format!(
                "--end {} precedes --start {}",
                iso8601(end),
                iso8601(start)
            )));
        }
        Ok((site, start, end))
    }

    fn cache_dir(&self) -> PathBuf {
        if let Some(cache) = &self.cache {
            return cache.clone();
        }
        match &self.out {
            Some(out) => out.join(".cache"),
            None => PathBuf::from(".rw-nexrad-cache"),
        }
    }
}

fn parse_positive(raw: &str, flag: &str) -> Result<f64, Box<dyn Error>> {
    let value: f64 = raw
        .parse()
        .map_err(|_| s3::boxed_error(format!("{flag} expects a number, got {raw:?}")))?;
    if !value.is_finite() || value <= 0.0 {
        return Err(s3::boxed_error(format!(
            "{flag} must be finite and positive, got {raw:?}"
        )));
    }
    Ok(value)
}

fn parse_site_latlon(raw: &str) -> Result<(f64, f64, f64), Box<dyn Error>> {
    let parts: Vec<&str> = raw.split(',').map(|part| part.trim()).collect();
    if parts.len() != 3 {
        return Err(s3::boxed_error(format!(
            "--site-latlon expects LAT,LON,ALT_M, got {raw:?}"
        )));
    }
    let mut values = [0.0f64; 3];
    for (slot, text) in values.iter_mut().zip(parts) {
        *slot = text
            .parse()
            .map_err(|_| s3::boxed_error(format!("--site-latlon component {text:?} is not a number")))?;
        if !slot.is_finite() {
            return Err(s3::boxed_error("--site-latlon components must be finite"));
        }
    }
    Ok((values[0], values[1], values[2]))
}

#[derive(Serialize)]
struct VolumeRecord {
    key: String,
    filename: String,
    valid_time: String,
    format: String,
    /// True for the pre-2016 `..._V06.gz` keys.  Recorded because
    /// `size_bytes` then describes the compressed object while the decoded
    /// stream is several times larger, and a reader comparing the two
    /// should see why rather than infer it from the key.
    gzipped: bool,
    size_bytes: u64,
    last_modified: String,
}

impl From<&VolumeObject> for VolumeRecord {
    fn from(volume: &VolumeObject) -> Self {
        Self {
            key: volume.object.key.clone(),
            filename: s3::object_filename(&volume.object.key).to_string(),
            valid_time: iso8601(volume.valid_time),
            format: volume.format.clone(),
            gzipped: volume.gzipped,
            size_bytes: volume.object.size_bytes,
            last_modified: volume.object.last_modified.clone(),
        }
    }
}

#[derive(Serialize)]
struct WindowRecord {
    site: String,
    start: String,
    end: String,
    bucket: String,
    day_prefixes: Vec<String>,
}

fn window_record(
    bucket: &str,
    site: &str,
    start: chrono::DateTime<chrono::Utc>,
    end: chrono::DateTime<chrono::Utc>,
) -> WindowRecord {
    WindowRecord {
        site: site.to_string(),
        start: iso8601(start),
        end: iso8601(end),
        bucket: bucket.to_string(),
        day_prefixes: s3::window_days(start, end)
            .into_iter()
            .map(|day| s3::day_prefix(site, day))
            .collect(),
    }
}

fn selected(volumes: Vec<VolumeObject>, limit: Option<usize>) -> (Vec<VolumeObject>, usize) {
    let matched = volumes.len();
    match limit {
        Some(limit) if limit < matched => (volumes.into_iter().take(limit).collect(), matched),
        _ => (volumes, matched),
    }
}

fn cmd_list(options: &Options) -> Result<String, Box<dyn Error>> {
    let (site, start, end) = options.window()?;
    let bucket = options.bucket()?;
    let agent = build_agent();
    let (volumes, observed_at) = list_volumes_observed(&agent, &bucket, &site, start, end)?;
    let (kept, matched) = selected(volumes, options.limit);

    #[derive(Serialize)]
    struct ListRecord {
        schema: &'static str,
        status: &'static str,
        feed: &'static str,
        window: WindowRecord,
        /// The bucket's clock when it answered, so the age of the newest
        /// volume is a measurement rather than a difference between two
        /// clocks nobody compared.
        observed_at: Option<String>,
        matched_volumes: usize,
        volumes: Vec<VolumeRecord>,
        total_bytes: u64,
    }

    let record = ListRecord {
        schema: LIST_SCHEMA,
        status: if kept.is_empty() { "EMPTY" } else { "READY" },
        feed: ARCHIVE_FEED,
        window: window_record(&bucket, &site, start, end),
        observed_at: observed_at.map(iso8601),
        matched_volumes: matched,
        total_bytes: kept.iter().map(|v| v.object.size_bytes).sum(),
        volumes: kept.iter().map(VolumeRecord::from).collect(),
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

fn cmd_fetch(options: &Options) -> Result<String, Box<dyn Error>> {
    let (site, start, end) = options.window()?;
    let bucket = options.bucket()?;
    let cache_dir = options.cache_dir();
    let agent = build_agent();
    let (volumes, observed_at) = list_volumes_observed(&agent, &bucket, &site, start, end)?;
    let (kept, matched) = selected(volumes, options.limit);
    if kept.is_empty() {
        // An empty window on the mirror can mean the mirror's coverage
        // rather than an empty sky, and the archive of record is the place
        // to check -- with credentials, since it refused anonymous access
        // on 2026-07-30.
        let hint = if bucket == DEFAULT_BUCKET {
            format!(
                " (this is the mirror; s3://{ARCHIVE_OF_RECORD_BUCKET}/ is the archive of \
                 record and may hold the window, but refused anonymous access when last \
                 probed -- see --help)"
            )
        } else {
            String::new()
        };
        return Err(s3::boxed_error(format!(
            "no Level-II volumes for {site} in {} .. {} under s3://{bucket}/{hint}",
            iso8601(start),
            iso8601(end)
        )));
    }

    #[derive(Serialize)]
    struct FileRecord {
        key: String,
        site: String,
        path: String,
        cache_path: String,
        valid_time: String,
        format: String,
        bytes: u64,
        sha256: String,
        cache_hit: bool,
    }

    #[derive(Serialize)]
    struct FetchRecord {
        schema: &'static str,
        status: &'static str,
        feed: &'static str,
        window: WindowRecord,
        observed_at: Option<String>,
        matched_volumes: usize,
        cache_dir: String,
        out_dir: Option<String>,
        files: Vec<FileRecord>,
        total_bytes: u64,
        cache_hits: usize,
    }

    let mut files = Vec::new();
    let mut cache_hits = 0usize;
    for volume in &kept {
        let downloaded = download_volume(&agent, &bucket, &cache_dir, volume, !options.no_cache)?;
        if downloaded.cache_hit {
            cache_hits += 1;
        }
        let published = match &options.out {
            Some(out) => publish_volume(&downloaded.path, out, &volume.object.key)?,
            None => downloaded.path.clone(),
        };
        files.push(FileRecord {
            key: downloaded.volume.object.key.clone(),
            site: downloaded.volume.site.clone(),
            valid_time: iso8601(downloaded.volume.valid_time),
            format: downloaded.volume.format.clone(),
            bytes: downloaded.volume.object.size_bytes,
            path: published.to_string_lossy().to_string(),
            cache_path: downloaded.path.to_string_lossy().to_string(),
            sha256: downloaded.sha256,
            cache_hit: downloaded.cache_hit,
        });
    }

    let record = FetchRecord {
        schema: FETCH_SCHEMA,
        status: "READY",
        feed: ARCHIVE_FEED,
        window: window_record(&bucket, &site, start, end),
        observed_at: observed_at.map(iso8601),
        matched_volumes: matched,
        cache_dir: cache_dir.to_string_lossy().to_string(),
        out_dir: options.out.as_ref().map(|p| p.to_string_lossy().to_string()),
        total_bytes: files.iter().map(|f| f.bytes).sum(),
        cache_hits,
        files,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

/// The shortest assembly this client will publish, in chunks.
///
/// Two, and the floor is measured rather than chosen: chunk 1 is the
/// `-S` chunk, which carries the volume header and the metadata messages
/// and no radials at all.  Assembled alone it is a well-formed Archive-II
/// file that `decode` refuses with "carried no Message-31 radial" -- a
/// correct refusal, three requests too late.
const DEFAULT_MIN_CHUNKS: usize = 2;

/// How much further back than asked the walk looks when the newest volumes
/// are not admissible, so "the newest complete volume" does not cost a
/// request per retained id.
const LOOKBACK_SLACK: usize = 3;

/// Why this volume is not one `live-fetch` will assemble, or `None`.
fn live_refusal(
    volume: &live::LiveVolume,
    allow_partial: bool,
    min_chunks: usize,
) -> Option<String> {
    if !volume.complete && !allow_partial {
        return Some(format!(
            "the scan is still in progress ({} of an unknown total chunks, no \
             end-of-volume chunk); pass --allow-partial to assimilate a volume \
             that is still being collected",
            volume.chunks.len()
        ));
    }
    if volume.chunks.len() < min_chunks {
        return Some(format!(
            "only {} chunk(s) have been published, below the --min-chunks {} \
             floor",
            volume.chunks.len(),
            min_chunks
        ));
    }
    None
}

#[derive(Serialize)]
struct LiveVolumeRecord {
    volume_id: u32,
    volume_time: String,
    chunks: usize,
    listed_chunks: usize,
    complete: bool,
    partial: bool,
    /// Why the validated prefix stops short of what the listing held, when
    /// it does.  `null` is "it does not".
    truncation: Option<String>,
    bytes: u64,
    first_chunk_key: Option<String>,
    newest_chunk_key: Option<String>,
    newest_chunk_last_modified: Option<String>,
    /// Seconds from the newest chunk landing in the bucket to the instant
    /// the bucket answered this listing, on the bucket's clock.  `null`
    /// when the `Date` header could not be read: an unmeasured lag is
    /// stated as unmeasured rather than filled in from the local clock.
    lag_seconds: Option<f64>,
    keys: Vec<String>,
    admissible: bool,
    refusal: Option<String>,
    // -- fetch only --
    #[serde(skip_serializing_if = "Option::is_none")]
    filename: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chunk_cache_hits: Option<usize>,
}

fn live_volume_record(
    volume: &live::LiveVolume,
    observed_at: Option<chrono::DateTime<chrono::Utc>>,
    refusal: Option<String>,
) -> LiveVolumeRecord {
    LiveVolumeRecord {
        volume_id: volume.volume_id,
        volume_time: iso8601(volume.volume_time),
        chunks: volume.chunks.len(),
        listed_chunks: volume.listed_chunks,
        complete: volume.complete,
        partial: !volume.complete,
        truncation: volume.truncation.clone(),
        bytes: volume.total_bytes(),
        first_chunk_key: volume.chunks.first().map(|c| c.object.key.clone()),
        newest_chunk_key: volume.chunks.last().map(|c| c.object.key.clone()),
        newest_chunk_last_modified: volume.newest_last_modified().map(iso8601),
        lag_seconds: volume.lag_seconds(observed_at),
        keys: volume.keys(),
        admissible: refusal.is_none(),
        refusal,
        filename: None,
        path: None,
        cache_path: None,
        sha256: None,
        chunk_cache_hits: None,
    }
}

/// `live-list` and `live-fetch`: the same discovery, the same policy, and
/// one of them moves bytes.
fn cmd_live(options: &Options, fetch: bool) -> Result<String, Box<dyn Error>> {
    let site = options.site_id()?;
    let bucket = options.live_bucket()?;
    let wanted = options.volumes.unwrap_or(1);
    let min_chunks = options.min_chunks.unwrap_or(DEFAULT_MIN_CHUNKS);
    let agent = build_agent();

    let discovery = live::discover(&agent, &bucket, &site, options.volume_id)?;
    let observed_at = discovery.observed_at;
    let local_now = chrono::Utc::now();
    let clock_skew_seconds = observed_at
        .map(|server| (local_now - server).num_milliseconds() as f64 / 1000.0);

    let newest_admissible =
        live_refusal(&discovery.volume, options.allow_partial, min_chunks).is_none();
    let mut pool = vec![discovery.volume.clone()];
    // The fast path -- the newest volume is the one that was asked for and
    // it qualifies -- costs no extra listing at all, which is the point of
    // a real-time route.
    let lookback = if options.volume_id.is_some() {
        0
    } else if wanted == 1 && newest_admissible {
        0
    } else {
        wanted - 1 + LOOKBACK_SLACK
    };
    if lookback > 0 {
        pool.extend(live::preceding_volumes(
            &agent,
            &bucket,
            &site,
            &discovery.ids,
            &discovery.volume,
            lookback,
        )?);
    }

    let mut records: Vec<LiveVolumeRecord> = Vec::new();
    let mut chosen: Vec<live::LiveVolume> = Vec::new();
    for volume in &pool {
        let refusal = live_refusal(volume, options.allow_partial, min_chunks);
        if refusal.is_none() && chosen.len() < wanted {
            chosen.push(volume.clone());
        }
        records.push(live_volume_record(volume, observed_at, refusal));
    }

    #[derive(Serialize)]
    struct LiveRecord {
        schema: &'static str,
        status: &'static str,
        feed: &'static str,
        site: String,
        bucket: String,
        /// The bucket's own clock when it answered discovery.  Every lag in
        /// this record is measured against it.
        observed_at: Option<String>,
        /// This host's clock minus the bucket's, so a reader can tell a
        /// stale feed from a wrong wall clock.
        clock_skew_seconds: Option<f64>,
        retained_volume_ids: usize,
        volume_id_runs: Vec<[u32; 2]>,
        probed_volume_ids: Vec<u32>,
        requested_volumes: usize,
        allow_partial: bool,
        min_chunks: usize,
        matched_volumes: usize,
        admissible_volumes: usize,
        volumes: Vec<LiveVolumeRecord>,
        total_bytes: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_dir: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        out_dir: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        chunk_cache_hits: Option<usize>,
    }

    let mut cache_dir = None;
    let mut out_dir = None;
    let mut chunk_cache_hits = None;
    if fetch {
        if chosen.is_empty() {
            let why = records
                .iter()
                .filter_map(|record| {
                    record
                        .refusal
                        .as_ref()
                        .map(|reason| format!("volume {} ({}): {reason}", record.volume_id, record.volume_time))
                })
                .collect::<Vec<_>>()
                .join("; ");
            return Err(s3::boxed_error(format!(
                "the real-time feed has no volume for {site} that this run will \
                 assemble. {why}"
            )));
        }
        let root = options.cache_dir();
        let mut hits = 0usize;
        for volume in &chosen {
            let assembled = live::assemble(
                &agent,
                &bucket,
                &root,
                options.out.as_deref(),
                volume,
                !options.no_cache,
            )?;
            hits += assembled.chunk_cache_hits;
            let slot = records
                .iter_mut()
                .find(|record| {
                    record.volume_id == volume.volume_id
                        && record.volume_time == iso8601(volume.volume_time)
                })
                .ok_or_else(|| {
                    s3::boxed_error("an assembled volume left no record to fill in")
                })?;
            slot.filename = Some(assembled.filename);
            slot.path = Some(assembled.path.to_string_lossy().to_string());
            slot.cache_path = Some(assembled.cache_path.to_string_lossy().to_string());
            slot.sha256 = Some(assembled.sha256);
            slot.chunk_cache_hits = Some(assembled.chunk_cache_hits);
            slot.bytes = assembled.bytes;
        }
        cache_dir = Some(root.to_string_lossy().to_string());
        out_dir = options.out.as_ref().map(|p| p.to_string_lossy().to_string());
        chunk_cache_hits = Some(hits);
    }

    let record = LiveRecord {
        schema: if fetch { LIVE_FETCH_SCHEMA } else { LIVE_LIST_SCHEMA },
        status: if chosen.is_empty() { "EMPTY" } else { "READY" },
        feed: LIVE_FEED,
        site,
        bucket,
        observed_at: observed_at.map(iso8601),
        clock_skew_seconds,
        retained_volume_ids: discovery.ids.len(),
        volume_id_runs: discovery
            .id_runs
            .iter()
            .map(|&(first, last)| [first, last])
            .collect(),
        probed_volume_ids: discovery.probed_ids,
        requested_volumes: wanted,
        allow_partial: options.allow_partial,
        min_chunks,
        matched_volumes: records.len(),
        admissible_volumes: chosen.len(),
        total_bytes: if fetch {
            records
                .iter()
                .filter(|record| record.sha256.is_some())
                .map(|record| record.bytes)
                .sum()
        } else {
            records.iter().map(|record| record.bytes).sum()
        },
        volumes: records,
        cache_dir,
        out_dir,
        chunk_cache_hits,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

fn cmd_decode(options: &Options) -> Result<String, Box<dyn Error>> {
    let volume_path = options
        .volume
        .as_deref()
        .ok_or_else(|| s3::boxed_error("--volume is required"))?;
    let out = options
        .out
        .as_deref()
        .ok_or_else(|| s3::boxed_error("--out is required (the pack destination)"))?;
    if out.is_dir() {
        return Err(s3::boxed_error(format!(
            "--out {} is a directory; give the pack file path",
            out.display()
        )));
    }
    let moments = parse_moment_filter(options.moments.as_deref().unwrap_or("REF,VEL"))?;
    let raw = std::fs::read(volume_path).map_err(|err| {
        s3::boxed_error(format!("cannot read {}: {err}", volume_path.display()))
    })?;
    // Expands the pre-2016 `.gz` archive transparently; a plain volume
    // passes through untouched.  Strict parsing then runs over whichever
    // stream came out, unchanged.
    let (stream, framing) = pack::read_volume(&raw)?;
    let request = DecodeRequest {
        volume_path,
        raw: &stream,
        source: &raw,
        framing,
        moments,
        max_range_km: options.max_range_km.unwrap_or(300.0),
        max_elevation_deg: options.max_elevation_deg.unwrap_or(20.0),
        site_override: options.site_latlon,
        censor_flags: options.censor_flags,
    };
    let (meta, payload) = decode::build_pack(&request)?;
    let pack_bytes = write_pack(out, &meta, &payload)?;

    #[derive(Serialize)]
    struct DecodeRecord<'a> {
        schema: &'static str,
        status: &'static str,
        pack: PackRecord<'a>,
        volume: &'a pack::VolumeEntry,
        site: &'a pack::SiteEntry,
        params: &'a pack::DecodeParams,
        sweeps: usize,
        radials: usize,
        gates: usize,
        dropped_sweeps: usize,
        dropped_moments: usize,
        trimmed_gates: usize,
        moments_present: Vec<String>,
        sweeps_without_nyquist: usize,
        incomplete_sweeps: usize,
    }

    #[derive(Serialize)]
    struct PackRecord<'a> {
        path: String,
        schema: &'a str,
        bytes: usize,
        payload_bytes: usize,
        content_sha256: &'a str,
    }

    let mut moments_present: Vec<String> = Vec::new();
    let mut radials = 0usize;
    let mut gates = 0usize;
    for sweep in &meta.sweeps {
        radials += sweep.radial_count;
        for moment in &sweep.moments {
            gates += sweep.radial_count * moment.gate_count;
            if !moments_present.contains(&moment.product) {
                moments_present.push(moment.product.clone());
            }
        }
    }
    moments_present.sort();

    let record = DecodeRecord {
        schema: DECODE_SCHEMA,
        status: "READY",
        pack: PackRecord {
            path: out.to_string_lossy().to_string(),
            schema: &meta.schema,
            bytes: pack_bytes,
            payload_bytes: meta.payload_bytes,
            content_sha256: &meta.content_sha256,
        },
        volume: &meta.volume,
        site: &meta.site,
        params: &meta.params,
        sweeps: meta.sweeps.len(),
        radials,
        gates,
        dropped_sweeps: meta.dropped_sweeps,
        dropped_moments: meta.dropped_moments,
        trimmed_gates: meta.trimmed_gates,
        moments_present,
        sweeps_without_nyquist: meta
            .sweeps
            .iter()
            .filter(|s| s.nyquist_velocity_ms.is_none())
            .count(),
        incomplete_sweeps: meta.sweeps.iter().filter(|s| !s.complete).count(),
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

fn cmd_verify(options: &Options) -> Result<String, Box<dyn Error>> {
    let path = options
        .volume
        .as_deref()
        .or(options.out.as_deref())
        .ok_or_else(|| s3::boxed_error("verify needs the pack path in --volume (or --out)"))?;
    let bytes = std::fs::read(path)
        .map_err(|err| s3::boxed_error(format!("cannot read {}: {err}", path.display())))?;
    let (meta, payload) = decode_pack(&bytes)?;
    // The header digest already matched; also prove every declared array
    // lies inside the payload it indexes.
    for (key, entry) in &meta.arrays {
        let end = entry.offset.saturating_add(entry.bytes);
        if end > payload.len() {
            return Err(s3::boxed_error(format!(
                "pack array {key} spans bytes {}..{end} of a {}-byte payload",
                entry.offset,
                payload.len()
            )));
        }
        let elements: usize = entry.shape.iter().product();
        if elements * 4 != entry.bytes {
            return Err(s3::boxed_error(format!(
                "pack array {key} declares shape {:?} ({elements} elements) but {} bytes",
                entry.shape, entry.bytes
            )));
        }
    }

    #[derive(Serialize)]
    struct VerifyRecord<'a> {
        schema: &'static str,
        status: &'static str,
        path: String,
        pack_schema: &'a str,
        bytes: usize,
        payload_bytes: usize,
        content_sha256: &'a str,
        arrays: usize,
        sweeps: usize,
        site: &'a pack::SiteEntry,
        volume: &'a pack::VolumeEntry,
    }

    let record = VerifyRecord {
        schema: VERIFY_SCHEMA,
        status: "PASS",
        path: path.to_string_lossy().to_string(),
        pack_schema: &meta.schema,
        bytes: bytes.len(),
        payload_bytes: payload.len(),
        content_sha256: &meta.content_sha256,
        arrays: meta.arrays.len(),
        sweeps: meta.sweeps.len(),
        site: &meta.site,
        volume: &meta.volume,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

fn cmd_sites(options: &Options) -> Result<String, Box<dyn Error>> {
    #[derive(Serialize)]
    struct SiteRow {
        id: String,
        name: String,
        lat_deg: f64,
        lon_deg: f64,
        alt_m: f64,
        /// False when `alt_m` is the table's unset placeholder rather than
        /// a measurement.  Reporting 0.0 m for 130 of 141 sites without
        /// saying so is how a reader concludes the antenna is at sea level.
        elevation_known: bool,
    }

    #[derive(Serialize)]
    struct SitesRecord {
        schema: &'static str,
        status: &'static str,
        source: &'static str,
        count: usize,
        /// How many rows carry the unset elevation placeholder.  A superob
        /// cannot use those without an explicit --site-latlon.
        sites_without_elevation: usize,
        sites: Vec<SiteRow>,
    }

    let rows: Vec<SiteRow> = match &options.site {
        Some(site) => {
            let site = normalize_site(site)?;
            let found = wx_radar::sites::find_site(&site).ok_or_else(|| {
                s3::boxed_error(format!(
                    "site {site:?} is not in the vendored operational NEXRAD table"
                ))
            })?;
            vec![SiteRow {
                id: found.id,
                name: found.name,
                lat_deg: found.lat,
                lon_deg: found.lon,
                alt_m: found.elevation,
                elevation_known: found.elevation != decode::SITE_ELEVATION_UNSET_M,
            }]
        }
        None => wx_radar::sites::SITES
            .iter()
            .map(|(id, name, lat, lon, alt)| SiteRow {
                id: (*id).to_string(),
                name: (*name).to_string(),
                lat_deg: *lat,
                lon_deg: *lon,
                alt_m: *alt,
                elevation_known: *alt != decode::SITE_ELEVATION_UNSET_M,
            })
            .collect(),
    };

    let record = SitesRecord {
        schema: SITES_SCHEMA,
        status: "READY",
        source: "wx-radar-site-table",
        count: rows.len(),
        sites_without_elevation: rows.iter().filter(|row| !row.elevation_known).count(),
        sites: rows,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_version_and_abi_are_stable_surfaces() {
        assert!(run(&[]).unwrap().contains("usage: rw_nexrad"));
        let help = run(&["--help".to_string()]).unwrap();
        for subcommand in [
            "list", "fetch", "live-list", "live-fetch", "decode", "verify", "sites",
        ] {
            assert!(help.contains(subcommand), "usage must document {subcommand}");
        }
        assert!(run(&["--version".to_string()])
            .unwrap()
            .starts_with("rw_nexrad "));
        assert_eq!(run(&["--abi".to_string()]).unwrap(), format!("{ABI_MARKER}\n"));
    }

    fn flags(args: &[&str]) -> Options {
        Options::parse(&args.iter().map(|a| a.to_string()).collect::<Vec<_>>()).expect("parses")
    }

    fn live_volume(chunks: usize, complete: bool) -> live::LiveVolume {
        let kind = |sequence: usize| {
            if sequence == 1 {
                live::ChunkKind::Start
            } else if complete && sequence == chunks {
                live::ChunkKind::End
            } else {
                live::ChunkKind::Intermediate
            }
        };
        live::LiveVolume {
            site: "TEST".to_string(),
            volume_id: 7,
            volume_time: chrono::Utc::now(),
            chunks: (1..=chunks)
                .map(|sequence| live::LiveChunk {
                    object: s3::S3Object {
                        key: format!("TEST/7/20260805-073454-{sequence:03}-x"),
                        size_bytes: 1000,
                        last_modified: "2026-08-05T07:35:00.000Z".to_string(),
                    },
                    sequence: sequence as u32,
                    kind: kind(sequence),
                })
                .collect(),
            complete,
            listed_chunks: chunks,
            truncation: None,
        }
    }

    #[test]
    fn the_help_states_both_defaults_archive_first() {
        // `tests/test_obs_nexrad.py` reads the FIRST "Default:" out of
        // --help and binds the Python constant to it, so the archive
        // default must stay the first one stated.
        let help = run(&["--help".to_string()]).unwrap();
        let archive = help.find("Default: unidata-nexrad-level2").expect("stated");
        let live = help
            .find("Live default: unidata-nexrad-level2-chunks")
            .expect("stated");
        assert!(archive < live, "the archive default must be stated first");
    }

    #[test]
    fn the_live_route_defaults_to_the_chunk_bucket_and_the_archive_route_does_not() {
        let bare = flags(&[]);
        assert_eq!(bare.bucket().unwrap(), DEFAULT_BUCKET);
        assert_eq!(bare.live_bucket().unwrap(), LIVE_DEFAULT_BUCKET);
        assert_ne!(DEFAULT_BUCKET, LIVE_DEFAULT_BUCKET);
        // Named, both routes honour the name: the chunk feed is a key
        // space, not a hard-coded endpoint.
        let named = flags(&["--bucket", "some-other-mirror"]);
        assert_eq!(named.bucket().unwrap(), "some-other-mirror");
        assert_eq!(named.live_bucket().unwrap(), "some-other-mirror");
    }

    #[test]
    fn the_live_options_parse_and_refuse_meaningless_counts() {
        let parsed = flags(&[
            "--volumes", "3", "--volume-id", "571", "--allow-partial", "--min-chunks", "7",
        ]);
        assert_eq!(parsed.volumes, Some(3));
        assert_eq!(parsed.volume_id, Some(571));
        assert!(parsed.allow_partial);
        assert_eq!(parsed.min_chunks, Some(7));
        assert!(!flags(&[]).allow_partial, "partial is opt-in, never a default");
        for bad in [
            vec!["--volumes", "0"],
            vec!["--volumes", "many"],
            vec!["--min-chunks", "0"],
            vec!["--volume-id", "-1"],
        ] {
            let args: Vec<String> = bad.iter().map(|a| a.to_string()).collect();
            assert!(Options::parse(&args).is_err(), "{bad:?} must fail closed");
        }
    }

    #[test]
    fn a_scan_still_in_progress_is_refused_unless_the_caller_asked_for_one() {
        let mid_scan = live_volume(9, false);
        let refusal = live_refusal(&mid_scan, false, DEFAULT_MIN_CHUNKS)
            .expect("a partial is not served by default");
        assert!(refusal.contains("--allow-partial"), "{refusal}");
        assert!(
            live_refusal(&mid_scan, true, DEFAULT_MIN_CHUNKS).is_none(),
            "asked for, a partial is admissible"
        );
    }

    #[test]
    fn an_assembly_below_the_chunk_floor_is_refused_even_when_partials_are_allowed() {
        // One chunk is the volume header alone: a well-formed Archive-II
        // file carrying no radial, which `decode` refuses three requests
        // later.  The floor turns that into a refusal before the fetch.
        let header_only = live_volume(1, false);
        let refusal = live_refusal(&header_only, true, DEFAULT_MIN_CHUNKS).expect("refused");
        assert!(refusal.contains("--min-chunks"), "{refusal}");
        assert!(live_refusal(&live_volume(2, false), true, DEFAULT_MIN_CHUNKS).is_none());
    }

    #[test]
    fn a_finished_volume_needs_no_permission_to_be_served() {
        assert!(live_refusal(&live_volume(4, true), false, DEFAULT_MIN_CHUNKS).is_none());
    }

    #[test]
    fn unknown_subcommands_and_options_fail_closed() {
        assert!(run(&["superob".to_string()]).is_err());
        assert!(Options::parse(&["--nope".to_string()]).is_err());
        assert!(Options::parse(&["--site".to_string()]).is_err());
    }

    #[test]
    fn window_validation_rejects_a_backwards_window_and_a_bad_site() {
        let options = Options {
            site: Some("KTLX".to_string()),
            start: Some("2023-05-20T20:30:00Z".to_string()),
            end: Some("2023-05-20T20:00:00Z".to_string()),
            ..Options::default()
        };
        let err = options.window().unwrap_err().to_string();
        assert!(err.contains("precedes"), "{err}");

        let options = Options {
            site: Some("K".to_string()),
            start: Some("2023-05-20T20:00:00Z".to_string()),
            end: Some("2023-05-20T20:30:00Z".to_string()),
            ..Options::default()
        };
        assert!(options.window().is_err());
    }

    #[test]
    fn numeric_options_reject_junk_and_nonpositive_values() {
        assert!(parse_positive("300", "--max-range-km").is_ok());
        assert!(parse_positive("0", "--max-range-km").is_err());
        assert!(parse_positive("-5", "--max-range-km").is_err());
        assert!(parse_positive("nan", "--max-range-km").is_err());
        assert!(parse_positive("far", "--max-range-km").is_err());
    }

    #[test]
    fn site_latlon_parses_a_triple_and_refuses_anything_else() {
        assert_eq!(
            parse_site_latlon("35.33,-97.28,370").unwrap(),
            (35.33, -97.28, 370.0)
        );
        assert!(parse_site_latlon("35.33,-97.28").is_err());
        assert!(parse_site_latlon("35.33,-97.28,high").is_err());
    }

    #[test]
    fn limit_truncates_but_still_reports_what_matched() {
        let volumes: Vec<VolumeObject> = (0..5)
            .map(|index| VolumeObject {
                object: s3::S3Object {
                    key: format!("2023/05/20/KTLX/KTLX20230520_2003{index:02}_V06"),
                    size_bytes: 100,
                    last_modified: String::new(),
                },
                site: "KTLX".to_string(),
                valid_time: chrono::Utc::now(),
                format: "V06".to_string(),
                gzipped: false,
            })
            .collect();
        let (kept, matched) = selected(volumes.clone(), Some(2));
        assert_eq!(kept.len(), 2);
        assert_eq!(matched, 5);
        let (kept, matched) = selected(volumes, None);
        assert_eq!(kept.len(), 5);
        assert_eq!(matched, 5);
    }

    #[test]
    fn bucket_defaults_to_the_one_that_answers_and_keeps_the_other_selectable() {
        // The default is the bucket that served anonymous list and get when
        // probed on 2026-07-30, not the one that is nominally
        // authoritative.  The archive of record stays one flag away, so a
        // caller with credentials -- or a future in which it grants
        // anonymous access again -- needs no code change.
        assert_eq!(Options::default().bucket().unwrap(), DEFAULT_BUCKET);
        assert_ne!(DEFAULT_BUCKET, ARCHIVE_OF_RECORD_BUCKET);
        for selected in [ARCHIVE_OF_RECORD_BUCKET, DEFAULT_BUCKET] {
            let options = Options {
                bucket: Some(selected.to_string()),
                ..Options::default()
            };
            assert_eq!(options.bucket().unwrap(), selected);
        }
        let options = Options {
            bucket: Some("no".to_string()),
            ..Options::default()
        };
        assert!(options.bucket().is_err());
    }

    #[test]
    fn the_help_records_which_bucket_is_which_and_why() {
        // A default chosen on a capability check is only useful if the
        // capability is written down where the person hitting the 403 will
        // look.  Both bucket names, the verdict, and the way back.
        for needle in [
            "unidata-nexrad-level2",
            "noaa-nexrad-level2",
            "capability check",
            "archive of record",
            "403",
            "--bucket noaa-nexrad-level2",
            "gzipped",
        ] {
            assert!(USAGE.contains(needle), "--help does not mention {needle:?}");
        }
    }

    #[test]
    fn cache_dir_defaults_next_to_the_output_then_to_the_cwd() {
        let options = Options {
            out: Some(PathBuf::from("volumes")),
            ..Options::default()
        };
        assert_eq!(options.cache_dir(), PathBuf::from("volumes").join(".cache"));
        assert_eq!(Options::default().cache_dir(), PathBuf::from(".rw-nexrad-cache"));
        let options = Options {
            cache: Some(PathBuf::from("explicit")),
            out: Some(PathBuf::from("volumes")),
            ..Options::default()
        };
        assert_eq!(options.cache_dir(), PathBuf::from("explicit"));
    }

    #[test]
    fn sites_command_dumps_the_table_and_refuses_unknown_ids() {
        let json = cmd_sites(&Options::default()).unwrap();
        assert!(json.contains(SITES_SCHEMA));
        assert!(json.contains("KTLX"));
        let one = cmd_sites(&Options {
            site: Some("ktlx".to_string()),
            ..Options::default()
        })
        .unwrap();
        assert!(one.contains("\"count\": 1"));
        assert!(cmd_sites(&Options {
            site: Some("ZZZZ".to_string()),
            ..Options::default()
        })
        .is_err());
    }

    #[test]
    fn decode_refuses_a_truncated_volume_without_writing_a_pack() {
        let dir = std::env::temp_dir().join(format!("rw-nexrad-decode-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let volume = dir.join("KTLX20230520_200356_V06");
        let mut raw = Vec::from(&b"AR2V0006."[..]);
        raw.resize(24, 0);
        raw.extend_from_slice(&4096i32.to_be_bytes());
        raw.extend_from_slice(b"BZh9only-a-handful-of-bytes");
        std::fs::write(&volume, &raw).unwrap();
        let out = dir.join("truncated.rdrpack");

        let options = Options {
            volume: Some(volume),
            out: Some(out.clone()),
            ..Options::default()
        };
        let err = cmd_decode(&options).unwrap_err().to_string();
        assert!(err.contains("truncated Level-II volume"), "{err}");
        assert!(!out.exists(), "a refused volume must leave no pack behind");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn decode_requires_both_a_volume_and_a_destination() {
        assert!(cmd_decode(&Options::default()).is_err());
        assert!(cmd_decode(&Options {
            volume: Some(PathBuf::from("x")),
            ..Options::default()
        })
        .is_err());
    }
}
