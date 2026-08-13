use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rayon::prelude::*;
use ureq::http::header::{CONTENT_RANGE, LOCATION};

use super::cache::DiskCache;

/// HTTP client for downloading GRIB2 data with byte-range support.
///
/// Uses ureq (blocking HTTP) with rustls + rustcrypto for TLS.
/// Supports configurable timeouts, retry with exponential backoff,
/// parallel chunk downloads, and optional disk caching.
pub struct DownloadClient {
    agent: ureq::Agent,
    #[allow(dead_code)]
    timeout: Duration,
    max_retries: u32,
    cache: Option<DiskCache>,
}

/// Maximum body size for full file downloads.
///
/// Full HRRR/RRFS family files can exceed the older subset-oriented 500 MB cap,
/// especially `wrfnat`. Keep the cap comfortably above current operational
/// artifacts while still guarding against obviously runaway downloads.
const MAX_BODY_SIZE: u64 = 8 * 1024 * 1024 * 1024;

/// Chunk size for whole-file parallel range downloads.
const FULL_FILE_RANGE_CHUNK_BYTES: u64 = 16 * 1024 * 1024;

/// Default timeout per request.
///
/// Full-family GRIB downloads routinely take longer than the old 30 s subset
/// budget, especially from NOMADS. Use a longer default so whole-file ingest is
/// viable without custom client wiring.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);

/// Default maximum number of retries.
const DEFAULT_MAX_RETRIES: u32 = 3;

/// Maximum redirects we will follow manually.
///
/// NOMADS file URLs should generally be direct. We disable ureq's built-in
/// redirect handling so malformed upstream 3xx responses do not bubble up as
/// opaque protocol errors such as "missing a location header", then follow
/// only well-formed redirects ourselves.
const MAX_REDIRECTS: u32 = 10;

/// Backoff durations for each retry attempt.
const BACKOFF_DURATIONS: [Duration; 3] = [
    Duration::from_millis(500),
    Duration::from_millis(1000),
    Duration::from_millis(2000),
];

/// Longer backoff for the Akamai "Over Rate Limit" behavior seen on NOMADS.
const NOMADS_RATE_LIMIT_BACKOFF_DURATIONS: [Duration; 3] = [
    Duration::from_secs(5),
    Duration::from_secs(10),
    Duration::from_secs(20),
];

/// Default spacing between NOMADS requests across all RustWX processes on this node.
const NOMADS_DEFAULT_MIN_REQUEST_GAP: Duration = Duration::from_millis(2500);

/// If NOMADS returns its Akamai over-rate-limit page, pause all RustWX NOMADS
/// requests on this node long enough for the block to cool off.
const NOMADS_DEFAULT_COOLDOWN: Duration = Duration::from_secs(15 * 60);

const NOMADS_LOCK_STALE_AFTER: Duration = Duration::from_secs(120);

/// Configuration for creating a DownloadClient.
pub struct DownloadConfig {
    /// Timeout per HTTP request.
    pub timeout: Duration,
    /// Maximum number of retry attempts.
    pub max_retries: u32,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            timeout: default_timeout(),
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }
}

fn default_timeout() -> Duration {
    std::env::var("RUSTWX_DOWNLOAD_TIMEOUT_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_TIMEOUT)
}

/// Check whether an error from ureq should be retried.
///
/// Retries on: connection/transport errors, 429 (rate limit),
/// 500, 502, 503, 504 (server errors).
/// Does NOT retry on: 400, 404, or other 4xx client errors.
fn is_retryable(err: &ureq::Error) -> bool {
    match err {
        ureq::Error::StatusCode(code) => {
            let c = *code;
            c == 429 || c == 500 || c == 502 || c == 503 || c == 504
        }
        // Timeout, DNS, connection reset, etc. — all retryable.
        _ => true,
    }
}

fn is_nomads_url(url: &str) -> bool {
    url.contains("nomads.ncep.noaa.gov")
}

fn is_probable_nomads_rate_limit(url: &str, err: &ureq::Error) -> bool {
    is_nomads_url(url) && err.to_string().contains("missing a location header")
}

fn is_redirect_status(status: ureq::http::StatusCode) -> bool {
    status.is_redirection()
}

fn resolve_redirect_url(current_url: &str, location: &str) -> crate::error::Result<String> {
    if location.starts_with("http://") || location.starts_with("https://") {
        return Ok(location.to_string());
    }

    let current_uri: ureq::http::Uri = current_url.parse().map_err(|err| {
        crate::RustmetError::Http(format!(
            "failed to parse redirect source URL {}: {}",
            current_url, err
        ))
    })?;

    let scheme = current_uri.scheme_str().ok_or_else(|| {
        crate::RustmetError::Http(format!(
            "redirect source URL {} is missing a scheme",
            current_url
        ))
    })?;
    let authority = current_uri.authority().ok_or_else(|| {
        crate::RustmetError::Http(format!(
            "redirect source URL {} is missing an authority",
            current_url
        ))
    })?;

    if location.starts_with('/') {
        return Ok(format!("{}://{}{}", scheme, authority, location));
    }

    let path = current_uri.path();
    let directory = path.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("");
    let joined = if directory.is_empty() {
        format!("/{}", location)
    } else {
        format!("{}/{}", directory, location)
    };
    Ok(format!("{}://{}{}", scheme, authority, joined))
}

fn env_duration_ms(name: &str, fallback: Duration) -> Duration {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|millis| *millis > 0)
        .map(Duration::from_millis)
        .unwrap_or(fallback)
}

fn nomads_state_path() -> PathBuf {
    std::env::var("RUSTWX_NOMADS_RATE_STATE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("rustwx_nomads_rate_limit.state"))
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

/// Read the shared pacing state: `(last_request_ms, cooldown_until_ms, sound)`.
///
/// `sound` is false when the file EXISTS but carries no usable
/// `last_request_ms`. That case used to be indistinguishable from "nobody has
/// fetched yet": both mapped to zero timestamps, which puts `last_request +
/// gap` in 1970 and waves the request straight through. A governor that
/// protects a shared public service must not treat corruption as permission to
/// send, so an unusable state is reported as "a request just happened" and the
/// caller waits a full gap. A genuinely absent file is still a real zero.
fn read_nomads_state(path: &Path) -> (u128, u128, bool) {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return (0, 0, true),
        Err(_) => return (now_millis(), 0, false),
    };
    let mut last_request_ms: Option<u128> = None;
    let mut cooldown_until_ms = 0;
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let Ok(parsed) = value.trim().parse::<u128>() else {
            continue;
        };
        match key.trim() {
            "last_request_ms" => last_request_ms = Some(parsed),
            "cooldown_until_ms" => cooldown_until_ms = parsed,
            _ => {}
        }
    }
    match last_request_ms {
        Some(value) => (value, cooldown_until_ms, true),
        None => (now_millis(), cooldown_until_ms, false),
    }
}

/// Publish the shared pacing state; false when the bytes did not land.
///
/// The result used to be discarded. A process that held the sentinel, failed
/// to record its request and proceeded left the next process seeing an older
/// `last_request_ms` and free to send immediately -- the failure opened the
/// gate instead of closing it. Callers now absorb the gap locally instead.
fn write_nomads_state(path: &Path, last_request_ms: u128, cooldown_until_ms: u128) -> bool {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let tmp = path.with_extension("tmp");
    let body = format!(
        "last_request_ms={}\ncooldown_until_ms={}\n",
        last_request_ms, cooldown_until_ms
    );
    fs::write(&tmp, body).is_ok() && fs::rename(tmp, path).is_ok()
}

struct NomadsRateLock {
    path: PathBuf,
}

impl Drop for NomadsRateLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn nomads_lock_is_stale(lock_path: &Path) -> bool {
    if fs::metadata(lock_path)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|elapsed| elapsed > NOMADS_LOCK_STALE_AFTER)
    {
        return true;
    }

    #[cfg(unix)]
    {
        if let Ok(text) = fs::read_to_string(lock_path) {
            if let Some(pid) = text.split_whitespace().next() {
                if pid.parse::<u32>().is_ok() && !Path::new("/proc").join(pid).exists() {
                    return true;
                }
            }
        }
    }

    false
}

fn acquire_nomads_rate_lock(state_path: &Path) -> Option<NomadsRateLock> {
    let lock_path = state_path.with_extension("lock");
    if let Some(parent) = lock_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    loop {
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
        {
            Ok(mut file) => {
                let _ = writeln!(file, "{} {}", std::process::id(), now_millis());
                return Some(NomadsRateLock { path: lock_path });
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                if nomads_lock_is_stale(&lock_path) {
                    let _ = fs::remove_file(&lock_path);
                    continue;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => return None,
        }
    }
}

fn log_nomads_event(url: &str, kind: &str, status: &str, elapsed_ms: Option<u128>) {
    let Ok(path) = std::env::var("RUSTWX_NOMADS_REQUEST_LOG") else {
        return;
    };
    let escaped_url = url.replace('\\', "\\\\").replace('"', "\\\"");
    let elapsed = elapsed_ms
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string());
    let line = format!(
        "{{\"ts_ms\":{},\"pid\":{},\"kind\":\"{}\",\"status\":\"{}\",\"elapsed_ms\":{},\"url\":\"{}\"}}\n",
        now_millis(),
        std::process::id(),
        kind,
        status.replace('"', "'"),
        elapsed,
        escaped_url
    );
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(line.as_bytes());
    }
}

fn mark_nomads_rate_limited(url: &str, reason: &str) {
    if !is_nomads_url(url) {
        return;
    }
    let cooldown = env_duration_ms("RUSTWX_NOMADS_COOLDOWN_MS", NOMADS_DEFAULT_COOLDOWN);
    let state_path = nomads_state_path();
    let _lock = acquire_nomads_rate_lock(&state_path);
    let (last_request_ms, existing_cooldown_until_ms, _sound) = read_nomads_state(&state_path);
    let now = now_millis();
    if existing_cooldown_until_ms > now {
        log_nomads_event(url, "cooldown_existing", reason, None);
        return;
    }
    let cooldown_until_ms = now.saturating_add(cooldown.as_millis());
    write_nomads_state(&state_path, last_request_ms, cooldown_until_ms);
    log_nomads_event(url, "cooldown", reason, None);
}

fn pace_request(url: &str) {
    if !is_nomads_url(url) {
        return;
    }

    let min_gap = env_duration_ms(
        "RUSTWX_NOMADS_MIN_INTERVAL_MS",
        NOMADS_DEFAULT_MIN_REQUEST_GAP,
    );
    let state_path = nomads_state_path();
    // An unusable state buys exactly one full gap, then this process
    // republishes a sound one. Charging it every round would spin forever
    // against a file nobody can parse: refusing to send is not the same as
    // refusing to finish.
    let mut corruption_paid = false;
    loop {
        let Some(_lock) = acquire_nomads_rate_lock(&state_path) else {
            std::thread::sleep(min_gap);
            continue;
        };

        let (mut last_request_ms, cooldown_until_ms, sound) = read_nomads_state(&state_path);
        if !sound {
            if !corruption_paid {
                drop(_lock);
                std::thread::sleep(min_gap);
                corruption_paid = true;
                continue;
            }
            last_request_ms = 0; // the gap has been served in full
        }
        let now = now_millis();
        let sleep_until =
            cooldown_until_ms.max(last_request_ms.saturating_add(min_gap.as_millis()));
        if sleep_until > now {
            drop(_lock);
            std::thread::sleep(Duration::from_millis(
                (sleep_until - now).min(u64::MAX as u128) as u64,
            ));
            continue;
        }
        if !write_nomads_state(&state_path, now, cooldown_until_ms) {
            // The record did not land, so the next process will not see this
            // request. Absorb the gap here instead of letting it send now.
            drop(_lock);
            std::thread::sleep(min_gap);
        }
        return;
    }
}

/// What this build's cross-process NOMADS governor is configured to do.
///
/// Everything here is read from the same places `pace_request` reads, so a
/// consumer that prints or asserts on it is describing the governor that will
/// actually run -- not a constant that happens to sit beside it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NomadsGovernor {
    /// The node-wide state file this process paces against.
    pub state_path: PathBuf,
    /// Minimum spacing between NOMADS requests across all processes.
    pub min_request_gap: Duration,
    /// How long an over-rate-limit answer pauses the whole node.
    pub cooldown: Duration,
    /// When a held sentinel is treated as abandoned.
    pub lock_stale_after: Duration,
}

/// The NOMADS governor this build carries.
///
/// A **capability probe**, and the reason it exists is worth stating. Two
/// different wx-core crates were vendored under the same `0.3.9` version
/// string: this one, and the published crates.io copy, which has no governor
/// at all -- no pacing, no cooldown, no state file. Only a path dependency
/// kept the right one in the graph, and "we must have got the right one
/// because of where it lives" is an assumption, not a check.
///
/// So consumers ask the resolved crate what it can do. A build wired to a
/// governorless wx-core does not link (this symbol is absent there), and a
/// build wired to this one can be made to *demonstrate* the pacing via
/// [`pace_nomads_request`] rather than infer it.
pub fn nomads_governor() -> NomadsGovernor {
    NomadsGovernor {
        state_path: nomads_state_path(),
        min_request_gap: env_duration_ms(
            "RUSTWX_NOMADS_MIN_INTERVAL_MS",
            NOMADS_DEFAULT_MIN_REQUEST_GAP,
        ),
        cooldown: env_duration_ms("RUSTWX_NOMADS_COOLDOWN_MS", NOMADS_DEFAULT_COOLDOWN),
        lock_stale_after: NOMADS_LOCK_STALE_AFTER,
    }
}

/// Block until this process may send `url`, exactly as the HTTP path does.
///
/// Public so a consumer can PROVE the governor runs -- issue two paced calls
/// for a NOMADS URL and observe the spacing and the state file -- instead of
/// trusting that the crate it linked is the one with the governor in it. A
/// no-op for every host but NOMADS, and it makes no network request.
pub fn pace_nomads_request(url: &str) {
    pace_request(url);
}

/// Build a ureq agent with TLS configured via rustls-rustcrypto.
fn build_agent(config: &DownloadConfig) -> ureq::Agent {
    // Install the rustcrypto provider as the process-wide default.
    rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider()).ok();

    let crypto = Arc::new(rustls_rustcrypto::provider());

    ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::Rustls)
                .root_certs(ureq::tls::RootCerts::WebPki)
                .unversioned_rustls_crypto_provider(crypto)
                .build(),
        )
        .max_redirects(0)
        .timeout_global(Some(config.timeout))
        .build()
        .new_agent()
}

impl DownloadClient {
    fn perform_get(
        &self,
        url: &str,
        range_header: Option<&str>,
    ) -> Result<ureq::http::Response<ureq::Body>, ureq::Error> {
        let mut request = self.agent.get(url);
        if let Some(range_header) = range_header {
            request = request.header("Range", range_header);
        }
        let started = now_millis();
        let result = request.call();
        if is_nomads_url(url) {
            let elapsed = now_millis().saturating_sub(started);
            match &result {
                Ok(response) => log_nomads_event(
                    url,
                    if range_header.is_some() {
                        "get_range"
                    } else {
                        "get"
                    },
                    response.status().as_str(),
                    Some(elapsed),
                ),
                Err(err) => log_nomads_event(
                    url,
                    if range_header.is_some() {
                        "get_range"
                    } else {
                        "get"
                    },
                    &format!("error:{err}"),
                    Some(elapsed),
                ),
            }
        }
        result
    }

    fn get_response_following_redirects(
        &self,
        url: &str,
        range_header: Option<&str>,
    ) -> crate::error::Result<ureq::http::Response<ureq::Body>> {
        let mut current_url = url.to_string();
        let mut malformed_redirect_retries = 0u32;

        for redirect_count in 0..=MAX_REDIRECTS {
            let request_url = current_url.clone();
            let response = self.with_retry(&request_url, || {
                self.perform_get(&request_url, range_header)
            })?;
            let status = response.status();

            if is_redirect_status(status) {
                if redirect_count == MAX_REDIRECTS {
                    return Err(crate::RustmetError::Http(format!(
                        "too many redirects while requesting {}",
                        url
                    )));
                }

                let location = response
                    .headers()
                    .get(LOCATION)
                    .and_then(|value| value.to_str().ok());

                let Some(location) = location else {
                    if is_nomads_url(&request_url) && malformed_redirect_retries < self.max_retries
                    {
                        malformed_redirect_retries += 1;
                        mark_nomads_rate_limited(&request_url, "redirect_missing_location");
                        eprintln!(
                            "  NOMADS cooldown {}/{} for {} (probable over-rate-limit redirect {})",
                            malformed_redirect_retries, self.max_retries, request_url, status
                        );
                        continue;
                    }

                    return Err(crate::RustmetError::Http(format!(
                        "redirect response missing Location header for {} (status {})",
                        request_url, status
                    )));
                };

                current_url = resolve_redirect_url(&request_url, location)?;
                continue;
            }

            return Ok(response);
        }

        Err(crate::RustmetError::Http(format!(
            "too many redirects while requesting {}",
            url
        )))
    }

    fn probe_nomads_range_ok(&self, url: &str) -> bool {
        for attempt in 0..=1u32 {
            let mut current_url = url.to_string();
            let mut retry = false;

            for _ in 0..=MAX_REDIRECTS {
                pace_request(&current_url);
                match self.perform_get(&current_url, Some("bytes=0-0")) {
                    Ok(response) => {
                        let status = response.status();
                        if is_redirect_status(status) {
                            let Some(location) = response
                                .headers()
                                .get(LOCATION)
                                .and_then(|value| value.to_str().ok())
                            else {
                                if is_nomads_url(&current_url) {
                                    mark_nomads_rate_limited(
                                        &current_url,
                                        "range_probe_redirect_missing_location",
                                    );
                                }
                                retry = attempt == 0;
                                break;
                            };

                            let Ok(next_url) = resolve_redirect_url(&current_url, location) else {
                                retry = attempt == 0;
                                break;
                            };
                            current_url = next_url;
                            continue;
                        }

                        return true;
                    }
                    Err(ureq::Error::StatusCode(code)) if code == 404 || code == 403 => {
                        return false;
                    }
                    Err(err) => {
                        if is_probable_nomads_rate_limit(&current_url, &err) {
                            mark_nomads_rate_limited(&current_url, "range_probe_rate_limit_error");
                        }
                        retry = attempt == 0 && is_retryable(&err);
                        break;
                    }
                }
            }

            if retry {
                std::thread::sleep(Duration::from_millis(500));
                continue;
            }
            return false;
        }

        false
    }

    /// Create a new download client with TLS configured via rustls-rustcrypto.
    ///
    /// Uses ureq's built-in TlsConfig with the rustcrypto provider and
    /// webpki root certificates (Mozilla's CA bundle). No caching.
    pub fn new() -> crate::error::Result<Self> {
        Self::new_with_config(DownloadConfig::default())
    }

    /// Create a new download client with custom timeout and retry settings.
    /// No caching.
    pub fn new_with_config(config: DownloadConfig) -> crate::error::Result<Self> {
        let agent = build_agent(&config);
        Ok(Self {
            agent,
            timeout: config.timeout,
            max_retries: config.max_retries,
            cache: None,
        })
    }

    /// Create a new download client with disk caching enabled.
    ///
    /// If `cache_dir` is `Some`, files are cached there. If `None`, the
    /// platform default is used (`~/.cache/metrust/` on Linux/macOS,
    /// `%LOCALAPPDATA%/metrust/cache/` on Windows).
    pub fn new_with_cache(cache_dir: Option<&str>) -> crate::error::Result<Self> {
        let config = DownloadConfig::default();
        let agent = build_agent(&config);
        let cache = match cache_dir {
            Some(dir) => DiskCache::with_dir(std::path::PathBuf::from(dir)),
            None => DiskCache::new(),
        };
        Ok(Self {
            agent,
            timeout: config.timeout,
            max_retries: config.max_retries,
            cache: Some(cache),
        })
    }

    /// Attach a `DiskCache` to this client. Replaces any existing cache.
    pub fn set_cache(&mut self, cache: DiskCache) {
        self.cache = Some(cache);
    }

    /// Return a reference to the underlying HTTP agent.
    ///
    /// Used by the streaming download module to make requests with
    /// manual body reading.
    pub fn agent(&self) -> &ureq::Agent {
        &self.agent
    }

    /// Return a reference to the cache, if one is attached.
    pub fn cache(&self) -> Option<&DiskCache> {
        self.cache.as_ref()
    }

    /// Execute a request-producing closure with retry and exponential backoff.
    ///
    /// `attempt_fn` is called on each attempt and must produce the final result
    /// or a ureq::Error. This avoids needing to name the ureq Response type.
    fn with_retry<T, F>(&self, url: &str, attempt_fn: F) -> crate::error::Result<T>
    where
        F: Fn() -> Result<T, ureq::Error>,
    {
        let mut last_err = String::new();

        for attempt in 0..=self.max_retries {
            pace_request(url);
            match attempt_fn() {
                Ok(val) => return Ok(val),
                Err(e) => {
                    let probable_nomads_rate_limit = is_probable_nomads_rate_limit(url, &e);
                    if probable_nomads_rate_limit {
                        mark_nomads_rate_limited(url, "retry_rate_limit_error");
                    }
                    last_err = if probable_nomads_rate_limit {
                        format!("probable NOMADS rate-limit response for {}: {}", url, e)
                    } else {
                        format!("{}", e)
                    };

                    if attempt < self.max_retries && is_retryable(&e) {
                        let backoff = if probable_nomads_rate_limit {
                            NOMADS_RATE_LIMIT_BACKOFF_DURATIONS
                                .get(attempt as usize)
                                .copied()
                                .unwrap_or(
                                    NOMADS_RATE_LIMIT_BACKOFF_DURATIONS
                                        [NOMADS_RATE_LIMIT_BACKOFF_DURATIONS.len() - 1],
                                )
                        } else {
                            BACKOFF_DURATIONS
                                .get(attempt as usize)
                                .copied()
                                .unwrap_or(BACKOFF_DURATIONS[BACKOFF_DURATIONS.len() - 1])
                        };
                        eprintln!(
                            "  Retry {}/{} for {} after {:?} ({})",
                            attempt + 1,
                            self.max_retries,
                            url,
                            backoff,
                            e
                        );
                        std::thread::sleep(backoff);
                    } else {
                        break;
                    }
                }
            }
        }

        Err(crate::RustmetError::Http(format!(
            "HTTP request failed for {}: {}",
            url, last_err
        )))
    }

    /// Send a HEAD request and return true if the server responds with 200 OK.
    ///
    /// Does NOT retry on 404 — only retries on transient/server errors.
    /// Useful for probing whether a remote file exists (e.g., .idx files).
    pub fn head_ok(&self, url: &str) -> bool {
        if is_nomads_url(url) {
            return self.probe_nomads_range_ok(url);
        }

        // Single attempt with one retry on transient errors.
        for attempt in 0..=1u32 {
            match self.agent.head(url).call() {
                Ok(_) => return true,
                Err(ureq::Error::StatusCode(code)) if code == 404 || code == 403 => {
                    return false;
                }
                Err(e) => {
                    if attempt == 0 && is_retryable(&e) {
                        std::thread::sleep(std::time::Duration::from_millis(300));
                        continue;
                    }
                    return false;
                }
            }
        }
        false
    }

    /// Download a full URL and return the response body as bytes.
    ///
    /// If caching is enabled, checks cache first and stores the result after
    /// a successful download. Cache failures are silently ignored.
    pub fn get_bytes(&self, url: &str) -> crate::error::Result<Vec<u8>> {
        let key = DiskCache::cache_key(url, None);

        // Try cache first
        if let Some(cache) = &self.cache {
            if let Some(data) = cache.get(&key) {
                return Ok(data);
            }
        }

        let mut response = self.get_response_following_redirects(url, None)?;
        let data = response
            .body_mut()
            .with_config()
            .limit(MAX_BODY_SIZE)
            .read_to_vec()
            .map_err(|err| crate::RustmetError::Http(format!("failed to read {}: {}", url, err)))?;

        // Store in cache (errors silently ignored)
        if let Some(cache) = &self.cache {
            cache.put(&key, &data);
        }

        Ok(data)
    }

    /// Download a full URL via byte ranges and return the concatenated bytes.
    ///
    /// This does not require an external `.idx` file. It first probes range
    /// support with `Range: bytes=0-0`; if the origin does not respond with a
    /// usable `Content-Range`, it falls back to the normal full-body download.
    pub fn get_bytes_parallel_whole(&self, url: &str) -> crate::error::Result<Vec<u8>> {
        let key = DiskCache::cache_key(url, None);

        if let Some(cache) = &self.cache {
            if let Some(data) = cache.get(&key) {
                return Ok(data);
            }
        }

        let total_len = match self.probe_range_total_length(url) {
            Ok(Some(total_len)) if total_len > 0 => total_len,
            _ => return self.get_bytes(url),
        };
        let ranges = full_file_ranges(total_len, FULL_FILE_RANGE_CHUNK_BYTES);
        if ranges.len() <= 1 {
            return self.get_bytes(url);
        }

        let data = self.get_ranges(url, &ranges)?;
        if data.len() as u64 != total_len {
            return Err(crate::RustmetError::Http(format!(
                "parallel whole-file download for {} returned {} bytes, expected {}",
                url,
                data.len(),
                total_len
            )));
        }

        // `get_ranges` has already stored these bytes under the URL+ranges
        // key. Storing them again under the URL-only key is what a later
        // whole-file request looks itself up by, and it costs a reference
        // rather than a second payload: the cache recognises the content.
        if let Some(cache) = &self.cache {
            cache.put(&key, &data);
        }

        Ok(data)
    }

    fn probe_range_total_length(&self, url: &str) -> crate::error::Result<Option<u64>> {
        let response = self.get_response_following_redirects(url, Some("bytes=0-0"))?;
        if response.status().as_u16() != 206 {
            return Ok(None);
        }
        Ok(response
            .headers()
            .get(CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .and_then(parse_content_range_total))
    }

    /// Download a URL and return the response body as a string (for .idx files).
    ///
    /// Text responses (like .idx) are NOT cached because they are small and
    /// may change between model runs.
    pub fn get_text(&self, url: &str) -> crate::error::Result<String> {
        let mut response = self.get_response_following_redirects(url, None)?;
        let text = response
            .body_mut()
            .read_to_string()
            .map_err(|err| crate::RustmetError::Http(format!("failed to read {}: {}", url, err)))?;
        Ok(text)
    }

    /// Download a specific byte range from a URL.
    ///
    /// If caching is enabled, the result is keyed by URL + byte range.
    /// Cache failures are silently ignored.
    pub fn get_range(&self, url: &str, start: u64, end: u64) -> crate::error::Result<Vec<u8>> {
        let key = DiskCache::cache_key(url, Some((start, end)));

        // Try cache first
        if let Some(cache) = &self.cache {
            if let Some(data) = cache.get(&key) {
                return Ok(data);
            }
        }

        let range_header = if end == u64::MAX {
            format!("bytes={}-", start)
        } else {
            format!("bytes={}-{}", start, end)
        };

        let mut response = self.get_response_following_redirects(url, Some(&range_header))?;
        // Validate the ANSWER before adopting it, not after concatenating it.
        // A range GET whose reply is a 200 (the origin ignored the header and
        // sent the whole object), or a 206 for some other span, or a body of
        // the wrong length, used to be accepted, cached, and concatenated into
        // the caller's subset -- the outer GRIB bars then rejected the
        // assembled file with no indication which chunk was wrong, and the bad
        // bytes stayed in the cache. This is the same three-clause check the
        // Python range transport has always applied.
        let status = response.status().as_u16();
        if status != 206 {
            return Err(crate::RustmetError::Http(format!(
                "range request for {} ({}) returned HTTP {}, not 206; the \
                 origin did not serve the requested span",
                url, range_header, status
            )));
        }
        let content_range = response
            .headers()
            .get(CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let expected_prefix = if end == u64::MAX {
            format!("bytes {}-", start)
        } else {
            format!("bytes {}-{}/", start, end)
        };
        if !content_range.starts_with(&expected_prefix) {
            return Err(crate::RustmetError::Http(format!(
                "range request for {} ({}) answered with Content-Range {:?}, \
                 which is not the requested span",
                url, range_header, content_range
            )));
        }
        let data = response
            .body_mut()
            .with_config()
            .limit(MAX_BODY_SIZE)
            .read_to_vec()
            .map_err(|err| crate::RustmetError::Http(format!("failed to read {}: {}", url, err)))?;
        if end != u64::MAX {
            let expected_len = end - start + 1;
            if data.len() as u64 != expected_len {
                return Err(crate::RustmetError::Http(format!(
                    "range request for {} ({}) returned {} bytes, expected {}",
                    url,
                    range_header,
                    data.len(),
                    expected_len
                )));
            }
        }

        // Store in cache (errors silently ignored)
        if let Some(cache) = &self.cache {
            cache.put(&key, &data);
        }

        Ok(data)
    }

    /// Download multiple byte ranges from a URL in parallel and concatenate the results.
    ///
    /// Each range is downloaded as a separate HTTP request with a Range header.
    /// Uses rayon to download chunks concurrently. Progress is printed to stderr.
    ///
    /// If caching is enabled, the combined result is cached under a key derived
    /// from the URL and all ranges. Individual ranges are also cached by
    /// `get_range`, so partial overlaps with future requests benefit from the
    /// cache too.
    ///
    /// The combined store is not a second copy of the object. A whole-file
    /// caller stores the same bytes again under its URL-only key, and those
    /// two entries used to be two complete payloads at two paths -- the store
    /// is content-addressed, so the second entry is a reference to the first.
    /// The two keys are still distinct keys: a range list that does not cover
    /// the object yields different bytes and gets its own payload, which is
    /// why the cache decides this on the content and not on the key shape.
    pub fn get_ranges(&self, url: &str, ranges: &[(u64, u64)]) -> crate::error::Result<Vec<u8>> {
        let total = ranges.len();
        if total == 0 {
            return Ok(Vec::new());
        }

        // Check for the combined result in cache
        let combined_key = DiskCache::cache_key_ranges(url, ranges);
        if let Some(cache) = &self.cache {
            if let Some(data) = cache.get(&combined_key) {
                return Ok(data);
            }
        }

        let completed = AtomicUsize::new(0);

        let results: Vec<crate::error::Result<Vec<u8>>> = if is_nomads_url(url) {
            ranges
                .iter()
                .map(|&(start, end)| {
                    let data = self.get_range(url, start, end)?;
                    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    eprint!("\r  Downloading chunks {}/{}...", done, total);
                    Ok(data)
                })
                .collect()
        } else {
            // Download all chunks in parallel, preserving order.
            // Each chunk is individually cached via get_range.
            ranges
                .par_iter()
                .map(|&(start, end)| {
                    let data = self.get_range(url, start, end)?;
                    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    eprint!("\r  Downloading chunks {}/{}...", done, total);
                    Ok(data)
                })
                .collect()
        };

        // Concatenate results in order, propagating the first error.
        let mut combined = Vec::new();
        for result in results {
            combined.extend_from_slice(&result?);
        }

        eprintln!(
            "\r  Downloaded {} chunks, {} bytes total.    ",
            total,
            combined.len()
        );

        // Cache the combined result (errors silently ignored)
        if let Some(cache) = &self.cache {
            cache.put(&combined_key, &combined);
        }

        Ok(combined)
    }
}

fn parse_content_range_total(value: &str) -> Option<u64> {
    let (_, total) = value.rsplit_once('/')?;
    if total == "*" {
        return None;
    }
    total.parse().ok()
}

fn full_file_ranges(total_len: u64, chunk_size: u64) -> Vec<(u64, u64)> {
    if total_len == 0 || chunk_size == 0 {
        return Vec::new();
    }

    let mut ranges = Vec::new();
    let mut start = 0u64;
    while start < total_len {
        let end = start.saturating_add(chunk_size - 1).min(total_len - 1);
        ranges.push((start, end));
        start = end.saturating_add(1);
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::{
        full_file_ranges, now_millis, parse_content_range_total, read_nomads_state,
        write_nomads_state, DownloadClient, DownloadConfig,
    };
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::thread;
    use std::time::Duration;

    fn spawn_http_server(responses: Vec<Vec<u8>>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("server addr");
        thread::spawn(move || {
            for response in responses {
                let (mut stream, _) = listener.accept().expect("accept connection");
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                stream.write_all(&response).expect("write response");
                stream.flush().expect("flush response");
            }
        });
        format!("http://{}", addr)
    }

    fn test_client() -> DownloadClient {
        DownloadClient::new_with_config(DownloadConfig {
            timeout: Duration::from_secs(5),
            max_retries: 1,
        })
        .expect("client")
    }

    #[test]
    fn get_bytes_follows_relative_redirects() {
        let base = spawn_http_server(vec![
            b"HTTP/1.1 302 Found\r\nLocation: /final\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                .to_vec(),
            b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello".to_vec(),
        ]);
        let client = test_client();
        let body = client
            .get_bytes(&format!("{}/start", base))
            .expect("redirected body");
        assert_eq!(body, b"hello");
    }

    #[test]
    fn get_bytes_surfaces_clear_error_for_redirect_without_location() {
        let base = spawn_http_server(vec![
            b"HTTP/1.1 302 Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
        ]);
        let client = test_client();
        let err = client
            .get_bytes(&format!("{}/broken", base))
            .expect_err("missing location should fail");
        let message = err.to_string();
        assert!(message.contains("redirect response missing Location header"));
        assert!(!message.contains("protocol: missing a location header"));
    }

    #[test]
    fn content_range_total_parses_known_total() {
        assert_eq!(parse_content_range_total("bytes 0-0/12345"), Some(12345));
        assert_eq!(parse_content_range_total("bytes 10-20/*"), None);
        assert_eq!(parse_content_range_total("not a range"), None);
    }

    #[test]
    fn full_file_ranges_cover_file_once_in_order() {
        assert_eq!(full_file_ranges(0, 4), Vec::<(u64, u64)>::new());
        assert_eq!(full_file_ranges(1, 4), vec![(0, 0)]);
        assert_eq!(full_file_ranges(10, 4), vec![(0, 3), (4, 7), (8, 9)]);
    }

    /// A range GET must be answered with the span it asked for.
    #[test]
    fn a_range_reply_that_is_not_the_requested_span_is_refused() {
        // The origin ignores the Range header and sends the whole object.
        let whole = spawn_http_server(vec![
            b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\nhello world"
                .to_vec(),
        ]);
        let message = test_client()
            .get_range(&format!("{}/x", whole), 0, 4)
            .expect_err("a 200 is not the requested span")
            .to_string();
        assert!(message.contains("not 206"), "{}", message);

        // The origin answers 206 for a different span.
        let wrong_span = spawn_http_server(vec![
            b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 6-10/11\r\n\
              Content-Length: 5\r\nConnection: close\r\n\r\nworld"
                .to_vec(),
        ]);
        let message = test_client()
            .get_range(&format!("{}/x", wrong_span), 0, 4)
            .expect_err("a foreign span must be refused")
            .to_string();
        assert!(message.contains("not the requested span"), "{}", message);

        // The span is right but the body is short.
        let short = spawn_http_server(vec![
            b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-4/11\r\n\
              Content-Length: 3\r\nConnection: close\r\n\r\nhel"
                .to_vec(),
        ]);
        let message = test_client()
            .get_range(&format!("{}/x", short), 0, 4)
            .expect_err("a short body must be refused")
            .to_string();
        assert!(message.contains("expected 5"), "{}", message);
    }

    #[test]
    fn a_well_formed_range_reply_is_accepted() {
        let base = spawn_http_server(vec![
            b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-4/11\r\n\
              Content-Length: 5\r\nConnection: close\r\n\r\nhello"
                .to_vec(),
        ]);
        assert_eq!(
            test_client()
                .get_range(&format!("{}/x", base), 0, 4)
                .expect("well-formed range"),
            b"hello"
        );
    }

    fn state_scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rustwx_state_{}", name));
        let _ = fs::create_dir_all(&dir);
        dir.join("rustwx_nomads_rate_limit.state")
    }

    /// An absent state is a real zero: nobody has fetched yet.
    #[test]
    fn absent_state_reads_as_a_genuine_zero() {
        let path = state_scratch("absent");
        let _ = fs::remove_file(&path);
        assert_eq!(read_nomads_state(&path), (0, 0, true));
    }

    /// The lie: corruption parsed as zero is permission to send.
    #[test]
    fn corrupt_state_reads_as_just_now_not_as_zero() {
        for body in [
            "",
            "garbage without an equals sign\n",
            "last_request_ms=not-a-number\n",
            "cooldown_until_ms=5\n",
        ] {
            let path = state_scratch("corrupt");
            fs::write(&path, body).unwrap();
            let (last, _cooldown, sound) = read_nomads_state(&path);
            assert!(!sound, "body {:?} should be reported unsound", body);
            assert!(
                last > 0 && last >= now_millis().saturating_sub(60_000),
                "body {:?} must read as a recent request, got {}",
                body,
                last
            );
        }
        let path = state_scratch("corrupt");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_sound_state_round_trips_through_both_halves() {
        let path = state_scratch("roundtrip");
        assert!(write_nomads_state(&path, 111, 222));
        assert_eq!(fs::read_to_string(&path).unwrap(), "last_request_ms=111\ncooldown_until_ms=222\n");
        assert_eq!(read_nomads_state(&path), (111, 222, true));
        let _ = fs::remove_file(&path);
    }

    /// A write that cannot land must be reported, not swallowed: the caller
    /// absorbs the gap locally instead of letting the next process send.
    #[test]
    fn a_state_write_that_cannot_land_reports_failure() {
        let dir = std::env::temp_dir().join("rustwx_state_unwritable");
        let _ = fs::create_dir_all(&dir);
        // A directory where the state file should be: neither write nor
        // rename can succeed onto it.
        let path = dir.join("occupied.state");
        let _ = fs::remove_file(&path);
        let _ = fs::create_dir_all(&path);
        assert!(!write_nomads_state(&path, 1, 2));
        let _ = fs::remove_dir_all(&dir);
    }
}
