use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rayon::prelude::*;

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

/// Maximum body size for full file downloads (500 MB).
const MAX_BODY_SIZE: u64 = 500 * 1024 * 1024;

/// Default timeout per request.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Default maximum number of retries.
const DEFAULT_MAX_RETRIES: u32 = 3;

/// Backoff durations for each retry attempt.
const BACKOFF_DURATIONS: [Duration; 3] = [
    Duration::from_millis(500),
    Duration::from_millis(1000),
    Duration::from_millis(2000),
];

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
            timeout: DEFAULT_TIMEOUT,
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }
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
        .timeout_global(Some(config.timeout))
        .build()
        .new_agent()
}

impl DownloadClient {
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
            match attempt_fn() {
                Ok(val) => return Ok(val),
                Err(e) => {
                    last_err = format!("{}", e);

                    if attempt < self.max_retries && is_retryable(&e) {
                        let backoff = BACKOFF_DURATIONS
                            .get(attempt as usize)
                            .copied()
                            .unwrap_or(BACKOFF_DURATIONS[BACKOFF_DURATIONS.len() - 1]);
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

        Err(crate::RustmetError::Http(
            format!("HTTP request failed for {}: {}", url, last_err)
        ))
    }

    /// Send a HEAD request and return true if the server responds with 200 OK.
    ///
    /// Does NOT retry on 404 — only retries on transient/server errors.
    /// Useful for probing whether a remote file exists (e.g., .idx files).
    pub fn head_ok(&self, url: &str) -> bool {
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

        let data = self.with_retry(url, || {
            let mut response = self.agent.get(url).call()?;
            let data = response
                .body_mut()
                .with_config()
                .limit(MAX_BODY_SIZE)
                .read_to_vec()?;
            Ok(data)
        })?;

        // Store in cache (errors silently ignored)
        if let Some(cache) = &self.cache {
            cache.put(&key, &data);
        }

        Ok(data)
    }

    /// Download a URL and return the response body as a string (for .idx files).
    ///
    /// Text responses (like .idx) are NOT cached because they are small and
    /// may change between model runs.
    pub fn get_text(&self, url: &str) -> crate::error::Result<String> {
        self.with_retry(url, || {
            let mut response = self.agent.get(url).call()?;
            let text = response
                .body_mut()
                .read_to_string()?;
            Ok(text)
        })
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

        let data = self.with_retry(url, || {
            let mut response = self
                .agent
                .get(url)
                .header("Range", &range_header)
                .call()?;
            let data = response
                .body_mut()
                .with_config()
                .limit(MAX_BODY_SIZE)
                .read_to_vec()?;
            Ok(data)
        })?;

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

        // Download all chunks in parallel, preserving order.
        // Each chunk is individually cached via get_range.
        let results: Vec<crate::error::Result<Vec<u8>>> = ranges
            .par_iter()
            .map(|&(start, end)| {
                let data = self.get_range(url, start, end)?;
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                eprint!("\r  Downloading chunks {}/{}...", done, total);
                Ok(data)
            })
            .collect();

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
