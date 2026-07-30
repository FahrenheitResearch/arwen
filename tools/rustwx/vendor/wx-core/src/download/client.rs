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

fn read_nomads_state(path: &Path) -> (u128, u128) {
    let Ok(text) = fs::read_to_string(path) else {
        return (0, 0);
    };
    let mut last_request_ms = 0;
    let mut cooldown_until_ms = 0;
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let Ok(parsed) = value.trim().parse::<u128>() else {
            continue;
        };
        match key.trim() {
            "last_request_ms" => last_request_ms = parsed,
            "cooldown_until_ms" => cooldown_until_ms = parsed,
            _ => {}
        }
    }
    (last_request_ms, cooldown_until_ms)
}

fn write_nomads_state(path: &Path, last_request_ms: u128, cooldown_until_ms: u128) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let tmp = path.with_extension("tmp");
    let body = format!(
        "last_request_ms={}\ncooldown_until_ms={}\n",
        last_request_ms, cooldown_until_ms
    );
    if fs::write(&tmp, body).is_ok() {
        let _ = fs::rename(tmp, path);
    }
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
    let (last_request_ms, existing_cooldown_until_ms) = read_nomads_state(&state_path);
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
    loop {
        let Some(_lock) = acquire_nomads_rate_lock(&state_path) else {
            std::thread::sleep(min_gap);
            continue;
        };

        let (last_request_ms, cooldown_until_ms) = read_nomads_state(&state_path);
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
        write_nomads_state(&state_path, now, cooldown_until_ms);
        return;
    }
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

    /// Download multiple byte ranges from a URL in parallel and concatenate the results.
    ///
    /// Each range is downloaded as a separate HTTP request with a Range header.
    /// Uses rayon to download chunks concurrently. Progress is printed to stderr.
    ///
    /// If caching is enabled, the combined result is cached under a key derived
    /// from the URL and all ranges. Individual ranges are also cached by
    /// `get_range`, so partial overlaps with future requests benefit from the
    /// cache too.
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
    use super::{full_file_ranges, parse_content_range_total, DownloadClient, DownloadConfig};
    use std::io::{Read, Write};
    use std::net::TcpListener;
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
}
