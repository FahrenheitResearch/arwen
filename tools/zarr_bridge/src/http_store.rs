//! The HTTP store the reader fetches a public Zarr archive through.
//!
//! `zarrs_http::HTTPStore` builds `reqwest::blocking::Client::new()`, whose
//! whole-request timeout is 30 seconds.  One ERA5 chunk on Google's public
//! copy is one hour of one field on 37 levels, about 80 MB compressed, so on
//! any link slower than about 2.7 MB/s every chunk is cut off mid-body and
//! the fetch fails with "error decoding response body" (measured on a node
//! whose link fell to 1.3 MB/s: 60 s for one chunk).  This store is the same
//! reads with a timeout sized for the chunk, not for a page, and a retry for
//! a transfer the network drops.  The reads follow zarrs_http 0.3.1
//! (MIT OR Apache-2.0), which stays the dependency its notices name.

use std::str::FromStr;
use std::time::Duration;

use reqwest::{
    header::{HeaderValue, CONTENT_LENGTH, RANGE},
    StatusCode, Url,
};
use zarrs::storage::{
    byte_range::ByteRangeIterator, MaybeBytes, MaybeBytesIterator, ReadableStorageTraits, StorageError, StoreKey,
};

/// One request may take this long: an 80 MB chunk at 150 kB/s.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Tries per request; a dropped transfer is tried again after a short wait.
const TRIES: u32 = 3;

#[derive(Debug)]
pub struct HttpStore {
    base_url: Url,
    client: reqwest::blocking::Client,
}

fn other(err: impl ToString) -> StorageError {
    StorageError::Other(err.to_string())
}

impl HttpStore {
    pub fn new(base_url: &str) -> Result<Self, String> {
        let base_url = Url::from_str(base_url).map_err(|_| format!("base URL {base_url} is not valid"))?;
        let client = reqwest::blocking::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT)
            .build()
            .map_err(|e| e.to_string())?;
        Ok(Self { base_url, client })
    }

    fn url(&self, key: &StoreKey) -> Result<Url, StorageError> {
        let mut url = self.base_url.as_str().to_string();
        if !key.as_str().is_empty() {
            url += &("/".to_string() + key.as_str().strip_prefix('/').unwrap_or(key.as_str()));
        }
        Url::parse(&url).map_err(other)
    }

    /// Send a request and read its whole body, trying again when the transfer fails.
    fn fetch(
        &self,
        build: impl Fn() -> reqwest::blocking::RequestBuilder,
    ) -> Result<(StatusCode, Option<bytes::Bytes>, Option<u64>), StorageError> {
        let mut last = String::new();
        for attempt in 0..TRIES {
            if attempt > 0 {
                std::thread::sleep(Duration::from_secs(2 * u64::from(attempt)));
            }
            let response = match build().send() {
                Ok(r) => r,
                Err(e) => {
                    last = e.to_string();
                    continue;
                }
            };
            let status = response.status();
            let length = response
                .headers()
                .get(CONTENT_LENGTH)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| u64::from_str(s).ok());
            if status != StatusCode::OK && status != StatusCode::PARTIAL_CONTENT {
                return Ok((status, None, length));
            }
            match response.bytes() {
                Ok(body) => return Ok((status, Some(body), length)),
                Err(e) => last = e.to_string(),
            }
        }
        Err(other(format!("http transfer failed after {TRIES} tries: {last}")))
    }
}

impl ReadableStorageTraits for HttpStore {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        let url = self.url(key)?;
        match self.fetch(|| self.client.get(url.clone()))? {
            (StatusCode::OK, body, _) => Ok(body),
            (StatusCode::NOT_FOUND, _, _) => Ok(None),
            (status, _, _) => Err(other(format!("http unexpected status code: {status}"))),
        }
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let url = self.url(key)?;
        let Some(size) = self.size_key(key)? else {
            return Ok(None);
        };
        // One single-part range request per range: every server answers those.
        let mut parts = Vec::new();
        for range in byte_ranges {
            let (start, end) = (range.start(size), range.end(size));
            let header = HeaderValue::from_str(&format!("bytes={}-{}", start, end - 1)).map_err(other)?;
            match self.fetch(|| self.client.get(url.clone()).header(RANGE, header.clone()))? {
                (StatusCode::PARTIAL_CONTENT, Some(body), _) if body.len() as u64 == end - start => parts.push(Ok(body)),
                (StatusCode::OK, Some(body), _) => {
                    let (s, e) = (usize::try_from(start).map_err(other)?, usize::try_from(end).map_err(other)?);
                    parts.push(Ok(body.slice(s..e)));
                }
                (status, _, _) => {
                    return Err(other(format!("the http server responded with status {status} for a byte range")));
                }
            }
        }
        Ok(Some(Box::new(parts.into_iter())))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        let url = self.url(key)?;
        match self.fetch(|| self.client.head(url.clone()))? {
            (StatusCode::OK, _, Some(length)) => Ok(Some(length)),
            (StatusCode::OK, _, None) => Err(other("content length response is invalid")),
            (StatusCode::NOT_FOUND, _, _) => Ok(None),
            (status, _, _) => Err(other(format!("http size_key has status code {status}"))),
        }
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}
