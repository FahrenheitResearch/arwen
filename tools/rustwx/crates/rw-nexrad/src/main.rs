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
//! ```text
//! rw_nexrad list   --site KTLX --start 2023-05-20T20:00:00Z --end 2023-05-20T20:30:00Z
//! rw_nexrad fetch  --site KTLX --start ... --end ... --out DIR [--limit N]
//! rw_nexrad decode --volume FILE --out FILE.rdrpack [--moments REF,VEL]
//! rw_nexrad sites  [--site KTLX]
//! ```

mod decode;
mod pack;
mod s3;

use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;

use serde::Serialize;

use decode::DecodeRequest;
use pack::{decode_pack, parse_moment_filter, write_pack};
use s3::{
    build_agent, download_volume, iso8601, list_volumes, normalize_bucket, normalize_site,
    parse_time, publish_volume, VolumeObject, ARCHIVE_OF_RECORD_BUCKET, DEFAULT_BUCKET,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub const LIST_SCHEMA: &str = "gpuwm-obs.nexrad-list.v1";
pub const FETCH_SCHEMA: &str = "gpuwm-obs.nexrad-fetch.v1";
pub const DECODE_SCHEMA: &str = "gpuwm-obs.nexrad-decode.v1";
pub const SITES_SCHEMA: &str = "gpuwm-obs.nexrad-sites.v1";
pub const VERIFY_SCHEMA: &str = "gpuwm-obs.nexrad-verify.v1";

/// The exact `--abi` line the Python bridge pins, so a bin that drifted out
/// from under a pinned wrapper is caught at probe time rather than at parse
/// time three receipts later.
const ABI_MARKER: &str = "gpuwm-obs.nexrad-fetch.v1\tsite\twindow\tbucket\tvolumes\tbytes\t\
cache_hits\tsha256\tgpuwm-obs.radar-sweeps.v1";

const USAGE: &str = "\
usage: rw_nexrad <list|fetch|decode|verify|sites> [OPTIONS]
       rw_nexrad --version | --help | --abi

  list     report the Level-II volumes a (site, window) resolves to, moving no payload
  fetch    download those volumes and print a fetch record
  decode   validate one volume and write a `gpuwm-obs.radar-sweeps.v1` pack
  verify   re-read a pack and re-prove its header, schema and payload digest
  sites    print the vendored NEXRAD site table (id, name, lat, lon, alt)

acquisition options
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

decode options
  --volume FILE           one Level-II volume on disk
  --out FILE              pack destination (required)
  --moments LIST          comma-separated: REF,VEL (default), SW, ZDR, RHO, PHI, KDP
  --max-range-km KM       drop gates beyond this slant range (default 300)
  --max-elevation-deg DEG drop sweeps above this elevation (default 20)
  --site-latlon LAT,LON,ALT_M
                          place a site the vendored table does not know

verify options
  --volume FILE           the pack to re-prove (--out is accepted too)

sites options
  --site ID               print just this site (exit 1 if unknown)
";

fn main() -> ExitCode {
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
    let volumes = list_volumes(&agent, &bucket, &site, start, end)?;
    let (kept, matched) = selected(volumes, options.limit);

    #[derive(Serialize)]
    struct ListRecord {
        schema: &'static str,
        status: &'static str,
        window: WindowRecord,
        matched_volumes: usize,
        volumes: Vec<VolumeRecord>,
        total_bytes: u64,
    }

    let record = ListRecord {
        schema: LIST_SCHEMA,
        status: if kept.is_empty() { "EMPTY" } else { "READY" },
        window: window_record(&bucket, &site, start, end),
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
    let volumes = list_volumes(&agent, &bucket, &site, start, end)?;
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
        window: WindowRecord,
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
        window: window_record(&bucket, &site, start, end),
        matched_volumes: matched,
        cache_dir: cache_dir.to_string_lossy().to_string(),
        out_dir: options.out.as_ref().map(|p| p.to_string_lossy().to_string()),
        total_bytes: files.iter().map(|f| f.bytes).sum(),
        cache_hits,
        files,
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
        for subcommand in ["list", "fetch", "decode", "verify", "sites"] {
            assert!(help.contains(subcommand), "usage must document {subcommand}");
        }
        assert!(run(&["--version".to_string()])
            .unwrap()
            .starts_with("rw_nexrad "));
        assert_eq!(run(&["--abi".to_string()]).unwrap(), format!("{ABI_MARKER}\n"));
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
