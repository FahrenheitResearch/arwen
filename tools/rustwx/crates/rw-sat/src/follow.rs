//! The follow engine: poll the live GOES bucket per band, fetch new objects
//! as soon as they land, ingest them into the rolling store, and evict old
//! frames — with typed [`SatEvent`] progress, a cancel flag, jittered poll
//! intervals, and exponential backoff on HTTP failures (never on "nothing
//! new yet": an empty diff is the normal idle case).
//!
//! Scheduling per the live-bucket survey: keys are diffed with
//! `start-after={last seen key}` under the band-specific hour prefix; near
//! the top of each UTC hour the previous hour's prefix is polled too so
//! stragglers and local clock skew cannot drop frames. Frame timestamps
//! come from the key's `s` (scan start) time, never the local clock.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use chrono::{DateTime, Timelike, Utc};

use crate::events::{SatError, SatEvent, other};
use crate::goes::{GoesSatellite, parse_goes_abi_filename};
use crate::s3::{
    DownloadedObject, S3Object, Sector, abi_filename_product_matches_request, band_hour_prefix,
    bucket_for_satellite, build_agent, cached_object_path, download_object, list_s3_objects,
    object_filename, prune_object_cache,
};
use crate::store::{WrittenFrame, downsample_field, frame_time, write_band_frame};
use crate::window::{WindowConfig, enforce_window};
use crate::abi::read_goes_abi_field;
use rw_store::run::RwsRunManifest;

/// Minutes after the top of the hour during which the previous hour's
/// prefix keeps being polled (stragglers + clock skew).
const HOUR_ROLLOVER_GRACE_MINUTES: u32 = 5;
/// Backoff cap after consecutive poll errors.
const MAX_BACKOFF_SECS: u64 = 300;
/// Ingest attempts before a repeatedly failing object is skipped for good
/// (with poll backoff in between, this spreads the retries over minutes).
const MAX_INGEST_ATTEMPTS: u32 = 5;
/// Cancel-flag check granularity while sleeping.
const SLEEP_SLICE_MS: u64 = 100;

#[derive(Debug, Clone)]
pub struct FollowConfig {
    /// Satellite name (`goes19`, `g18`, ...).
    pub satellite: String,
    pub sector: Sector,
    /// ABI bands to follow (1..=16).
    pub bands: Vec<u8>,
    /// ABI scan mode token in filenames (6 = nominal since 2019).
    pub mode: u8,
    pub store_root: PathBuf,
    pub cache_dir: PathBuf,
    /// Base poll interval; `None` uses the sector default.
    pub poll_interval: Option<Duration>,
    /// +/- jitter fraction applied to every sleep (default 0.2).
    pub jitter_frac: f64,
    /// Per-band stride decimation before storing (1 = native).
    pub downsample: usize,
    pub window: WindowConfig,
    /// Stop after this many poll cycles (`None` = run until cancelled).
    pub max_polls: Option<u32>,
    /// Stop once this many frames have been ingested.
    pub max_frames: Option<u32>,
    pub use_cache: bool,
}

impl FollowConfig {
    pub fn new(satellite: &str, sector: Sector, bands: Vec<u8>) -> Self {
        Self {
            satellite: satellite.to_string(),
            sector,
            bands,
            mode: 6,
            store_root: PathBuf::from("store"),
            cache_dir: PathBuf::from("cache"),
            poll_interval: None,
            jitter_frac: 0.2,
            downsample: 1,
            window: WindowConfig::default(),
            max_polls: None,
            max_frames: None,
            use_cache: true,
        }
    }

    fn base_interval(&self) -> Duration {
        self.poll_interval
            .unwrap_or_else(|| Duration::from_secs(self.sector.default_poll_secs()))
    }
}

/// What a follow session did.
#[derive(Debug, Default)]
pub struct FollowSummary {
    pub polls: u32,
    pub frames: Vec<WrittenFrame>,
    pub downloaded_keys: Vec<String>,
    pub evicted_frames: usize,
    pub evicted_bytes: u64,
}

/// The hour prefixes one poll must cover: the current scan hour, preceded
/// by the previous hour during the first [`HOUR_ROLLOVER_GRACE_MINUTES`]
/// of each hour. Pure for testing.
pub fn poll_prefixes(
    abi_product: &str,
    satellite: &GoesSatellite,
    mode: u8,
    band: u8,
    now: DateTime<Utc>,
) -> Vec<String> {
    let mut prefixes = Vec::with_capacity(2);
    if now.minute() < HOUR_ROLLOVER_GRACE_MINUTES {
        let previous = now - chrono::Duration::hours(1);
        prefixes.push(band_hour_prefix(abi_product, satellite, mode, band, previous));
    }
    prefixes.push(band_hour_prefix(abi_product, satellite, mode, band, now));
    prefixes
}

/// The sleep before the next poll: `base` +/- `jitter_frac` (with
/// `unit_sample` in `[0, 1]` mapping to `[-1, +1]`), doubled per
/// consecutive error and capped at [`MAX_BACKOFF_SECS`]. Pure for testing.
pub fn poll_delay(
    base: Duration,
    jitter_frac: f64,
    unit_sample: f64,
    consecutive_errors: u32,
) -> Duration {
    let jitter = jitter_frac.clamp(0.0, 1.0) * (unit_sample.clamp(0.0, 1.0) * 2.0 - 1.0);
    let jittered = base.as_secs_f64() * (1.0 + jitter);
    let backoff = jittered * f64::from(2u32.saturating_pow(consecutive_errors.min(16)));
    Duration::from_secs_f64(backoff.min(MAX_BACKOFF_SECS as f64).max(0.05))
}

/// Bounded dedup of already-ingested scans, keyed by (band, scan-start
/// MINUTE). Minute granularity matches the store's `tHHMM.rws` frame
/// slots, so entries primed from run manifests (which only know HHMM)
/// dedup live keys (whose `s` token carries seconds + tenths) exactly.
/// Re-listed keys (page overlap, prefix rollover, a session restart) are
/// dropped here even if `start-after` state was lost.
#[derive(Debug, Default)]
pub struct SeenScans {
    seen: BTreeSet<(u8, DateTime<Utc>)>,
}

impl SeenScans {
    /// Record (band, start minute). Returns `false` when already seen.
    pub fn insert(&mut self, band: u8, start: DateTime<Utc>) -> bool {
        self.seen.insert((band, scan_minute(start)))
    }

    pub fn contains(&self, band: u8, start: DateTime<Utc>) -> bool {
        self.seen.contains(&(band, scan_minute(start)))
    }

    /// Drop entries older than `cutoff` (call with `now - window`).
    pub fn prune_older_than(&mut self, cutoff: DateTime<Utc>) {
        self.seen.retain(|&(_, start)| start >= cutoff);
    }

    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

/// Truncate a scan start to its minute (the store's HHMM granularity).
fn scan_minute(start: DateTime<Utc>) -> DateTime<Utc> {
    start
        .with_second(0)
        .and_then(|time| time.with_nanosecond(0))
        .unwrap_or(start)
}

/// Dedup state rebuilt from the run manifests already in the store, so a
/// restarted session never re-downloads or re-ingests frames the rolling
/// window already holds (the whole current-hour prefix is re-listed after
/// a restart because `start-after` state is session-local).
pub fn primed_seen_scans(
    store_root: &std::path::Path,
    model: &str,
    sector_slug: &str,
    bands: &[u8],
) -> SeenScans {
    let mut seen = SeenScans::default();
    let Ok(entries) = std::fs::read_dir(store_root.join(model)) else {
        return seen;
    };
    for entry in entries.flatten() {
        let run_name = entry.file_name().to_string_lossy().to_string();
        // Run dirs are `<sector>_c<band>_<YYYYMMDD>[_<k>]`.
        let Some(band) = bands.iter().copied().find(|band| {
            run_name.starts_with(&format!("{sector_slug}_c{band:02}_"))
        }) else {
            continue;
        };
        let Ok(bytes) = std::fs::read(entry.path().join("run.json")) else {
            continue;
        };
        let Ok(manifest) = serde_json::from_slice::<RwsRunManifest>(&bytes) else {
            continue;
        };
        for &hhmm in manifest.hours.keys() {
            if let Some(time) = frame_time(&run_name, hhmm) {
                seen.insert(band, time);
            }
        }
    }
    seen
}

/// Cheap deterministic xorshift in `[0, 1)` for poll jitter.
#[derive(Debug)]
pub struct JitterRng(u64);

impl JitterRng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Download one object and ingest it as a store frame. Shared by the
/// follow loop and the one-shot `latest` CLI flow.
#[allow(clippy::too_many_arguments)]
pub fn fetch_and_ingest(
    agent: &ureq::Agent,
    bucket: &str,
    cache_dir: &std::path::Path,
    store_root: &std::path::Path,
    object: &S3Object,
    downsample: usize,
    use_cache: bool,
    written_unix: u64,
    sink: &mut dyn FnMut(SatEvent),
) -> Result<(DownloadedObject, WrittenFrame), SatError> {
    sink(SatEvent::DownloadStarted {
        key: object.key.clone(),
        bytes: object.size_bytes,
    });
    let started = Instant::now();
    let download =
        download_object(agent, bucket, cache_dir, object, use_cache).map_err(to_send_sync)?;
    sink(SatEvent::DownloadDone {
        key: object.key.clone(),
        bytes: object.size_bytes,
        ms: started.elapsed().as_millis(),
        cache_hit: download.cache_hit,
    });

    let field = read_goes_abi_field(&download.path, "CMI").map_err(to_send_sync)?;
    let field = downsample_field(field, downsample);
    let frame = write_band_frame(store_root, &field, written_unix).map_err(to_send_sync)?;
    sink(SatEvent::FrameWritten {
        model: frame.model.clone(),
        run: frame.run.clone(),
        hhmm: frame.hhmm,
        scan_time_utc: frame.scan_time_utc,
        path: frame.path.clone(),
        bytes: frame.bytes,
        encode_ms: frame.encode_ms,
    });
    Ok((download, frame))
}

/// Process one prefix's freshly listed objects in key order, advancing the
/// `start-after` watermark in `last_key` only through objects that were
/// skipped on purpose (not ours, stale, already seen) or successfully
/// ingested. On a retryable ingest failure the watermark stays put and the
/// rest of the prefix is left alone, so the next poll re-lists the failed
/// object and retries it — a transient S3 503 or truncated read no longer
/// leaves a permanent gap in the loop. `attempts` caps the retries per key
/// ([`MAX_INGEST_ATTEMPTS`]) so one poisoned object cannot stall its
/// prefix forever. `seen` is only marked on success.
///
/// Returns the warning messages to emit; every entry also means "this
/// poll failed" for backoff purposes.
#[allow(clippy::too_many_arguments)]
fn process_listed_objects(
    prefix: &str,
    objects: &[S3Object],
    band: u8,
    abi_product: &str,
    stale_cutoff: Option<DateTime<Utc>>,
    seen: &mut SeenScans,
    attempts: &mut HashMap<String, u32>,
    last_key: &mut HashMap<String, String>,
    cancel: &AtomicBool,
    ingest: &mut dyn FnMut(&S3Object) -> Result<(), SatError>,
) -> Result<Vec<String>, SatError> {
    let mut warnings = Vec::new();
    for object in objects {
        check_cancel(cancel)?;
        let scan_start = parse_goes_abi_filename(object_filename(&object.key))
            .ok()
            .filter(|parsed| {
                object.key.ends_with(".nc")
                    && abi_filename_product_matches_request(&parsed.product, abi_product)
                    && parsed.channel == Some(band)
                    // Never ingest a scan the rolling window would evict
                    // on the spot: a (re)start re-lists the whole hour
                    // prefix, which can hold frames older than the window
                    // — pure download/encode/evict churn otherwise.
                    && stale_cutoff.is_none_or(|cutoff| parsed.start_time_utc >= cutoff)
            })
            .map(|parsed| parsed.start_time_utc);
        let Some(scan_start) = scan_start else {
            last_key.insert(prefix.to_string(), object.key.clone());
            continue;
        };
        if seen.contains(band, scan_start) {
            last_key.insert(prefix.to_string(), object.key.clone());
            continue;
        }
        match ingest(object) {
            Ok(()) => {
                seen.insert(band, scan_start);
                attempts.remove(&object.key);
                last_key.insert(prefix.to_string(), object.key.clone());
            }
            Err(SatError::Cancelled) => return Err(SatError::Cancelled),
            Err(err) => {
                let tried = {
                    let entry = attempts.entry(object.key.clone()).or_insert(0);
                    *entry += 1;
                    *entry
                };
                if tried >= MAX_INGEST_ATTEMPTS {
                    warnings.push(format!(
                        "ingest {}: {err} (attempt {tried}/{MAX_INGEST_ATTEMPTS}, giving up on this object)",
                        object.key
                    ));
                    attempts.remove(&object.key);
                    last_key.insert(prefix.to_string(), object.key.clone());
                } else {
                    warnings.push(format!(
                        "ingest {}: {err} (attempt {tried}/{MAX_INGEST_ATTEMPTS}, will retry after re-listing)",
                        object.key
                    ));
                    // Hold the watermark before this object: the next
                    // poll re-lists it and everything after it.
                    break;
                }
            }
        }
    }
    Ok(warnings)
}

/// Run a follow session. Returns when `max_polls`/`max_frames` is reached;
/// observing the cancel flag at any boundary returns
/// [`SatError::Cancelled`].
pub fn follow(
    config: &FollowConfig,
    sink: &mut dyn FnMut(SatEvent),
    cancel: &AtomicBool,
) -> Result<FollowSummary, SatError> {
    if config.bands.is_empty() {
        return Err(other("follow requires at least one band"));
    }
    for &band in &config.bands {
        if !(1..=16).contains(&band) {
            return Err(other(format!("ABI band out of range: {band}")));
        }
    }
    let bucket = bucket_for_satellite(&config.satellite).map_err(to_send_sync)?;
    let satellite = GoesSatellite::parse(&config.satellite);
    let abi_product = config.sector.abi_product();
    let agent = build_agent();
    let base_interval = config.base_interval();
    let mut rng = JitterRng::new(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(1),
    );

    let mut summary = FollowSummary::default();
    // Dedup survives restarts: every frame the rolling window already
    // holds (per the run manifests) is pre-marked seen, so re-listing the
    // whole current-hour prefix on session start fetches nothing twice.
    let model = satellite.as_str().to_ascii_lowercase();
    let mut seen = primed_seen_scans(
        &config.store_root,
        &model,
        config.sector.slug(),
        &config.bands,
    );
    if !seen.is_empty() {
        sink(SatEvent::Info {
            message: format!(
                "dedup primed from store manifests: {} frame(s) already ingested",
                seen.len()
            ),
        });
    }
    // start-after state per (band, hour prefix).
    let mut last_key: HashMap<String, String> = HashMap::new();
    // Failed-ingest retry counts per S3 key (bounded: pruned with `seen`).
    let mut ingest_attempts: HashMap<String, u32> = HashMap::new();
    let mut consecutive_errors: u32 = 0;

    loop {
        check_cancel(cancel)?;
        let mut poll_failed = false;
        for &band in &config.bands {
            check_cancel(cancel)?;
            let now = Utc::now();
            let prefixes = poll_prefixes(abi_product, &satellite, config.mode, band, now);
            sink(SatEvent::PollStarted {
                band,
                prefixes: prefixes.clone(),
            });
            let poll_started = Instant::now();
            let mut ingested_this_poll = 0usize;
            for prefix in &prefixes {
                let start_after = last_key.get(prefix).map(String::as_str);
                let objects = match list_s3_objects(&agent, &bucket, prefix, start_after) {
                    Ok(objects) => objects,
                    Err(err) => {
                        poll_failed = true;
                        sink(SatEvent::Warning {
                            message: format!("list {prefix}: {err}"),
                        });
                        continue;
                    }
                };
                let stale_cutoff = config.window.max_age_minutes.map(|minutes| {
                    Utc::now() - chrono::Duration::minutes(i64::from(minutes))
                });
                let mut ingest = |object: &S3Object| -> Result<(), SatError> {
                    let written_unix = Utc::now().timestamp().max(0) as u64;
                    let result = fetch_and_ingest(
                        &agent,
                        &bucket,
                        &config.cache_dir,
                        &config.store_root,
                        object,
                        config.downsample,
                        config.use_cache,
                        written_unix,
                        &mut *sink,
                    );
                    // The store frame is the artifact of record; the raw
                    // cached object never helps this session again
                    // (SeenScans dedups) nor a restart (manifest-primed
                    // dedup skips the fetch entirely). Dropping it on
                    // success keeps the cache footprint bounded; dropping
                    // it on failure makes the retry refetch fresh bytes
                    // instead of replaying a size-matched corrupt cache
                    // hit.
                    if !matches!(result, Err(SatError::Cancelled)) {
                        let _ = std::fs::remove_file(cached_object_path(
                            &config.cache_dir,
                            &bucket,
                            &object.key,
                        ));
                    }
                    let (_download, frame) = result?;
                    summary.downloaded_keys.push(object.key.clone());
                    summary.frames.push(frame.clone());
                    ingested_this_poll += 1;
                    // Scope eviction to this followed band's runs.
                    let run_prefix = format!("{}_c{band:02}", config.sector.slug());
                    match enforce_window(
                        &config.store_root,
                        &frame.model,
                        &run_prefix,
                        Utc::now(),
                        &config.window,
                    ) {
                        Ok(report) if report.removed_frames > 0 => {
                            summary.evicted_frames += report.removed_frames;
                            summary.evicted_bytes += report.removed_bytes;
                            sink(SatEvent::Evicted {
                                model: frame.model.clone(),
                                frames: report.removed_frames,
                                bytes: report.removed_bytes,
                            });
                        }
                        Ok(_) => {}
                        Err(err) => sink(SatEvent::Warning {
                            message: format!("window eviction: {err}"),
                        }),
                    }
                    Ok(())
                };
                let warnings = process_listed_objects(
                    prefix,
                    &objects,
                    band,
                    abi_product,
                    stale_cutoff,
                    &mut seen,
                    &mut ingest_attempts,
                    &mut last_key,
                    cancel,
                    &mut ingest,
                )?;
                for message in warnings {
                    poll_failed = true;
                    sink(SatEvent::Warning { message });
                }
            }
            sink(SatEvent::PollDone {
                band,
                new_keys: ingested_this_poll,
                ms: poll_started.elapsed().as_millis(),
            });
        }
        // Keep the per-prefix bookkeeping bounded: drop start-after
        // watermarks and retry counters for hour prefixes that rotated out
        // of the poll set.
        let now = Utc::now();
        let active: Vec<String> = config
            .bands
            .iter()
            .flat_map(|&band| poll_prefixes(abi_product, &satellite, config.mode, band, now))
            .collect();
        last_key.retain(|prefix, _| active.contains(prefix));
        ingest_attempts.retain(|key, _| active.iter().any(|prefix| key.starts_with(prefix.as_str())));
        // Dedup memory stays bounded: anything older than a day is gone
        // from the hour prefixes we poll anyway.
        seen.prune_older_than(Utc::now() - chrono::Duration::days(1));
        // The raw-object cache obeys the rolling window too: each ingest
        // already deletes its own cached object (above); this sweep catches
        // leftovers — interrupted sessions, repeatedly failing objects —
        // so a 24/7 follow keeps a bounded disk footprint even though
        // `enforce_window` only knows the store. Without a max-age the
        // sweep uses the same one-day horizon as the dedup set.
        let cache_cutoff = Utc::now()
            - chrono::Duration::minutes(i64::from(
                config.window.max_age_minutes.unwrap_or(24 * 60),
            ));
        let pruned = prune_object_cache(&config.cache_dir, &bucket, cache_cutoff);
        if pruned.removed_files > 0 {
            sink(SatEvent::Info {
                message: format!(
                    "cache pruned: {} object(s), {} bytes",
                    pruned.removed_files, pruned.removed_bytes
                ),
            });
        }

        consecutive_errors = if poll_failed {
            consecutive_errors.saturating_add(1)
        } else {
            0
        };
        summary.polls += 1;
        if config.max_polls.is_some_and(|max| summary.polls >= max) {
            return Ok(summary);
        }
        if config
            .max_frames
            .is_some_and(|max| summary.frames.len() as u32 >= max)
        {
            return Ok(summary);
        }

        let delay = poll_delay(
            base_interval,
            config.jitter_frac,
            rng.next_unit(),
            consecutive_errors,
        );
        sink(SatEvent::Sleeping {
            ms: delay.as_millis() as u64,
        });
        sleep_cancellable(delay, cancel)?;
    }
}

fn check_cancel(cancel: &AtomicBool) -> Result<(), SatError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(SatError::Cancelled);
    }
    Ok(())
}

fn sleep_cancellable(total: Duration, cancel: &AtomicBool) -> Result<(), SatError> {
    let deadline = Instant::now() + total;
    while Instant::now() < deadline {
        check_cancel(cancel)?;
        let remaining = deadline.saturating_duration_since(Instant::now());
        std::thread::sleep(remaining.min(Duration::from_millis(SLEEP_SLICE_MS)));
    }
    check_cancel(cancel)
}

fn to_send_sync(err: Box<dyn std::error::Error>) -> SatError {
    other(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn poll_prefixes_cover_hour_rollover_grace() {
        let satellite = GoesSatellite::G19;
        let mid_hour = Utc.with_ymd_and_hms(2026, 6, 10, 18, 30, 0).unwrap();
        let prefixes = poll_prefixes("ABI-L2-CMIPC", &satellite, 6, 13, mid_hour);
        assert_eq!(
            prefixes,
            vec!["ABI-L2-CMIPC/2026/161/18/OR_ABI-L2-CMIPC-M6C13_G19_"]
        );

        let just_rolled = Utc.with_ymd_and_hms(2026, 6, 10, 19, 2, 0).unwrap();
        let prefixes = poll_prefixes("ABI-L2-CMIPC", &satellite, 6, 13, just_rolled);
        assert_eq!(
            prefixes,
            vec![
                "ABI-L2-CMIPC/2026/161/18/OR_ABI-L2-CMIPC-M6C13_G19_",
                "ABI-L2-CMIPC/2026/161/19/OR_ABI-L2-CMIPC-M6C13_G19_",
            ]
        );

        // Day (and year-prefix) rollover comes for free from chrono.
        let new_day = Utc.with_ymd_and_hms(2026, 6, 11, 0, 0, 0).unwrap();
        let prefixes = poll_prefixes("ABI-L2-CMIPC", &satellite, 6, 13, new_day);
        assert_eq!(prefixes[0], "ABI-L2-CMIPC/2026/161/23/OR_ABI-L2-CMIPC-M6C13_G19_");
        assert_eq!(prefixes[1], "ABI-L2-CMIPC/2026/162/00/OR_ABI-L2-CMIPC-M6C13_G19_");
    }

    #[test]
    fn poll_delay_jitters_and_backs_off() {
        let base = Duration::from_secs(30);
        // unit_sample 0.5 -> no jitter.
        assert_eq!(poll_delay(base, 0.2, 0.5, 0), Duration::from_secs(30));
        // Extremes stay within +/- 20%.
        let low = poll_delay(base, 0.2, 0.0, 0);
        let high = poll_delay(base, 0.2, 1.0, 0);
        assert_eq!(low, Duration::from_secs(24));
        assert_eq!(high, Duration::from_secs(36));
        // Errors double the delay...
        assert_eq!(poll_delay(base, 0.0, 0.5, 1), Duration::from_secs(60));
        assert_eq!(poll_delay(base, 0.0, 0.5, 2), Duration::from_secs(120));
        // ... capped at 5 minutes.
        assert_eq!(poll_delay(base, 0.0, 0.5, 10), Duration::from_secs(300));
    }

    #[test]
    fn seen_scans_dedup_and_prune() {
        let mut seen = SeenScans::default();
        let t0 = Utc.with_ymd_and_hms(2026, 6, 10, 18, 51, 18).unwrap();
        assert!(seen.insert(13, t0));
        assert!(!seen.insert(13, t0), "second insert is a duplicate");
        assert!(seen.insert(2, t0), "same time, different band is distinct");
        assert!(seen.insert(13, t0 + chrono::Duration::minutes(5)));
        assert_eq!(seen.len(), 3);
        seen.prune_older_than(t0 + chrono::Duration::minutes(1));
        assert_eq!(seen.len(), 1, "old entries pruned");
        assert!(seen.contains(13, t0 + chrono::Duration::minutes(5)));
    }

    #[test]
    fn seen_scans_key_on_the_scan_minute() {
        let mut seen = SeenScans::default();
        let listed = Utc.with_ymd_and_hms(2026, 6, 10, 18, 51, 18).unwrap()
            + chrono::Duration::milliseconds(100);
        let manifest_slot = Utc.with_ymd_and_hms(2026, 6, 10, 18, 51, 0).unwrap();
        assert!(seen.insert(13, manifest_slot), "primed from a manifest");
        assert!(
            seen.contains(13, listed),
            "live key with seconds dedups against the primed minute"
        );
        assert!(!seen.insert(13, listed));
    }

    #[test]
    fn priming_reads_run_manifests_per_band() {
        use crate::store::test_support::{scan_start, synthetic_field};
        use crate::store::write_band_frame;

        let dir = std::env::temp_dir().join(format!(
            "rw-sat-follow-prime-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        write_band_frame(&dir, &synthetic_field(12, 10, scan_start(18, 51), 13, 0.0), 1).unwrap();
        write_band_frame(&dir, &synthetic_field(12, 10, scan_start(18, 56), 13, 0.0), 2).unwrap();
        write_band_frame(&dir, &synthetic_field(12, 10, scan_start(18, 51), 8, 0.0), 3).unwrap();

        let seen = primed_seen_scans(&dir, "g19", "conus", &[13]);
        assert_eq!(seen.len(), 2, "only the followed band primes");
        // The live listing carries seconds; the primed minute still hits.
        assert!(seen.contains(13, scan_start(18, 51)));
        assert!(seen.contains(13, scan_start(18, 56)));
        assert!(!seen.contains(8, scan_start(18, 51)));

        let both = primed_seen_scans(&dir, "g19", "conus", &[8, 13]);
        assert_eq!(both.len(), 3);

        let missing = primed_seen_scans(&dir, "g18", "conus", &[13]);
        assert!(missing.is_empty(), "absent model dir primes nothing");
        let _ = std::fs::remove_dir_all(&dir);
    }

    const TEST_PREFIX: &str = "ABI-L2-CMIPC/2026/161/18/OR_ABI-L2-CMIPC-M6C13_G19_";

    fn listed_object(key: impl Into<String>) -> S3Object {
        S3Object {
            key: key.into(),
            size_bytes: 1,
            last_modified: String::new(),
        }
    }

    /// A C13 CONUS key under [`TEST_PREFIX`] starting at 18:`minute`.
    fn c13_key(minute: u32) -> String {
        format!(
            "{TEST_PREFIX}s202616118{minute:02}176_e202616118{minute:02}549_c202616118{minute:02}590.nc"
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_process(
        objects: &[S3Object],
        stale_cutoff: Option<DateTime<Utc>>,
        seen: &mut SeenScans,
        attempts: &mut HashMap<String, u32>,
        last_key: &mut HashMap<String, String>,
        ingest: &mut dyn FnMut(&S3Object) -> Result<(), SatError>,
    ) -> Vec<String> {
        let cancel = AtomicBool::new(false);
        process_listed_objects(
            TEST_PREFIX,
            objects,
            13,
            "ABI-L2-CMIPC",
            stale_cutoff,
            seen,
            attempts,
            last_key,
            &cancel,
            ingest,
        )
        .unwrap()
    }

    #[test]
    fn failed_ingest_holds_the_watermark_and_is_retried() {
        let objects = vec![listed_object(c13_key(51)), listed_object(c13_key(56))];
        let mut seen = SeenScans::default();
        let mut attempts = HashMap::new();
        let mut last_key = HashMap::new();

        // First poll: the 18:51 ingest fails transiently.
        let mut fail_first = |object: &S3Object| -> Result<(), SatError> {
            if object.key == objects[0].key {
                Err(other("503 slow down"))
            } else {
                Ok(())
            }
        };
        let warnings = run_process(
            &objects,
            None,
            &mut seen,
            &mut attempts,
            &mut last_key,
            &mut fail_first,
        );
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("will retry"), "{}", warnings[0]);
        assert!(
            !last_key.contains_key(TEST_PREFIX),
            "watermark held before the failed key so the next poll re-lists it"
        );
        assert!(seen.is_empty(), "failures are never marked seen");
        assert_eq!(attempts.get(objects[0].key.as_str()), Some(&1));

        // Next poll re-lists both keys (held watermark) and succeeds.
        let mut ok = |_object: &S3Object| Ok(());
        let warnings = run_process(
            &objects,
            None,
            &mut seen,
            &mut attempts,
            &mut last_key,
            &mut ok,
        );
        assert!(warnings.is_empty());
        assert_eq!(seen.len(), 2, "both scans ingested after the retry");
        assert_eq!(
            last_key.get(TEST_PREFIX),
            Some(&objects[1].key),
            "watermark advanced through the last success"
        );
        assert!(attempts.is_empty(), "retry counter cleared on success");
    }

    #[test]
    fn poisoned_object_is_dropped_after_the_attempt_cap() {
        let objects = vec![listed_object(c13_key(51)), listed_object(c13_key(56))];
        let mut seen = SeenScans::default();
        let mut attempts = HashMap::new();
        let mut last_key = HashMap::new();
        let mut ingest = |object: &S3Object| -> Result<(), SatError> {
            if object.key == objects[0].key {
                Err(other("truncated NetCDF"))
            } else {
                Ok(())
            }
        };

        for attempt in 1..MAX_INGEST_ATTEMPTS {
            let warnings = run_process(
                &objects,
                None,
                &mut seen,
                &mut attempts,
                &mut last_key,
                &mut ingest,
            );
            assert!(warnings[0].contains("will retry"), "{}", warnings[0]);
            assert!(!last_key.contains_key(TEST_PREFIX));
            assert_eq!(attempts.get(objects[0].key.as_str()), Some(&attempt));
            assert!(seen.is_empty(), "the good key stays blocked behind the bad one");
        }

        // Final attempt: give up on the bad object, unblock the prefix.
        let warnings = run_process(
            &objects,
            None,
            &mut seen,
            &mut attempts,
            &mut last_key,
            &mut ingest,
        );
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("giving up"), "{}", warnings[0]);
        assert!(attempts.is_empty(), "no counter leak after giving up");
        assert_eq!(seen.len(), 1, "the 18:56 frame finally ingested");
        assert!(seen.contains(13, Utc.with_ymd_and_hms(2026, 6, 10, 18, 56, 0).unwrap()));
        assert_eq!(
            last_key.get(TEST_PREFIX),
            Some(&objects[1].key),
            "watermark moved past both keys"
        );
    }

    #[test]
    fn skipped_objects_advance_the_watermark_without_ingest() {
        let stale_cutoff = Utc.with_ymd_and_hms(2026, 6, 10, 18, 45, 0).unwrap();
        let already_seen = c13_key(51);
        let objects = vec![
            // Sidecar / non-NetCDF object.
            listed_object(format!("{TEST_PREFIX}manifest.json")),
            // Wrong band under a sibling prefix page.
            listed_object(
                "ABI-L2-CMIPC/2026/161/18/OR_ABI-L2-CMIPC-M6C08_G19_s20261611846176_e20261611848549_c20261611849020.nc",
            ),
            // Older than the rolling window: churn if ingested.
            listed_object(c13_key(41)),
            // Already in the store (restart priming or an earlier poll).
            listed_object(already_seen.clone()),
        ];
        let mut seen = SeenScans::default();
        seen.insert(13, Utc.with_ymd_and_hms(2026, 6, 10, 18, 51, 0).unwrap());
        let mut attempts = HashMap::new();
        let mut last_key = HashMap::new();
        let mut ingest_calls = 0usize;
        let mut ingest = |_object: &S3Object| -> Result<(), SatError> {
            ingest_calls += 1;
            Ok(())
        };

        let warnings = run_process(
            &objects,
            Some(stale_cutoff),
            &mut seen,
            &mut attempts,
            &mut last_key,
            &mut ingest,
        );
        assert!(warnings.is_empty());
        assert_eq!(ingest_calls, 0, "every object was skipped on purpose");
        assert_eq!(
            last_key.get(TEST_PREFIX),
            Some(&already_seen),
            "skips advance the watermark so they are never re-listed"
        );
    }

    #[test]
    fn cancel_mid_listing_propagates() {
        let cancel = AtomicBool::new(true);
        let objects = vec![listed_object(c13_key(51))];
        let mut seen = SeenScans::default();
        let mut attempts = HashMap::new();
        let mut last_key = HashMap::new();
        let mut ingest = |_object: &S3Object| -> Result<(), SatError> { Ok(()) };
        let err = process_listed_objects(
            TEST_PREFIX,
            &objects,
            13,
            "ABI-L2-CMIPC",
            None,
            &mut seen,
            &mut attempts,
            &mut last_key,
            &cancel,
            &mut ingest,
        )
        .unwrap_err();
        assert!(err.is_cancelled());
        assert!(last_key.is_empty(), "nothing consumed after cancel");
    }

    #[test]
    fn jitter_rng_is_deterministic_and_in_range() {
        let mut a = JitterRng::new(42);
        let mut b = JitterRng::new(42);
        for _ in 0..100 {
            let sample = a.next_unit();
            assert_eq!(sample, b.next_unit());
            assert!((0.0..1.0).contains(&sample), "sample {sample}");
        }
    }

    #[test]
    fn cancel_flag_stops_sleep_and_follow() {
        let cancel = AtomicBool::new(true);
        let err = sleep_cancellable(Duration::from_secs(5), &cancel).unwrap_err();
        assert!(err.is_cancelled());

        let config = FollowConfig::new("goes19", Sector::Conus, vec![13]);
        let mut events = Vec::new();
        let result = follow(&config, &mut |event| events.push(event), &cancel);
        assert!(result.is_err_and(|err| err.is_cancelled()));
    }

    #[test]
    fn follow_rejects_empty_or_invalid_bands() {
        let cancel = AtomicBool::new(false);
        let mut sink = |_event: SatEvent| {};
        let empty = FollowConfig::new("goes19", Sector::Conus, vec![]);
        assert!(follow(&empty, &mut sink, &cancel).is_err());
        let bad = FollowConfig::new("goes19", Sector::Conus, vec![17]);
        assert!(follow(&bad, &mut sink, &cancel).is_err());
    }
}
