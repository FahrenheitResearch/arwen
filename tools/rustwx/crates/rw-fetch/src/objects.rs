//! The unsigned public-object seam. Model transport and its pacing stay unchanged.
//! NDJSON is flushed per listed or verified object, never after a whole batch.
use std::collections::{HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use reqwest::blocking::{Client, Response};
use reqwest::{StatusCode, Url};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const CHUNK: u64 = 16 * 1024 * 1024;
const BUCKETS: &[&str] = &["unidata-nexrad-level2", "noaa-nexrad-level2", "noaa-mrms-pds",
    "noaa-hrrr-bdp-pds", "noaa-rap-pds", "noaa-gfs-bdp-pds"];

#[derive(Clone, Deserialize)]
struct Source { bucket: String, prefix: String }

#[derive(Clone, Deserialize)]
struct Object {
    bucket: String, key: String, bytes: u64, etag: Option<String>, destination: PathBuf,
    #[serde(flatten)] metadata: serde_json::Map<String, Value>,
}

#[derive(Deserialize)]
struct Request {
    #[serde(default)] sources: Vec<Source>,
    #[serde(default)] objects: Vec<Object>,
    #[serde(default)] origin: Option<String>,
    #[serde(default)] timeout_seconds: Option<f64>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Listing {
    #[serde(default)] contents: Vec<Entry>,
    #[serde(default)] is_truncated: bool,
    next_continuation_token: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Entry { key: String, size: u64, #[serde(rename = "ETag")] etag: String, last_modified: String }

fn emit(value: Value) -> Result<(), String> {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    serde_json::to_writer(&mut out, &value).map_err(|e| e.to_string())?;
    out.write_all(b"\n").and_then(|_| out.flush()).map_err(|e| e.to_string())
}

fn url(bucket: &str, key: &str, origin: Option<&str>) -> Result<Url, String> {
    if !BUCKETS.contains(&bucket) { return Err("not an allowed anonymous public bucket".into()); }
    let base = match origin {
        Some(value) => {
            let parsed = Url::parse(value).map_err(|e| e.to_string())?;
            if parsed.scheme() != "http" || parsed.host_str() != Some("127.0.0.1")
                || !parsed.username().is_empty() || parsed.password().is_some()
                || parsed.query().is_some() || parsed.fragment().is_some() {
                return Err("test origin must be an unsigned loopback HTTP origin".into());
            }
            value.to_string()
        }
        None => format!("https://{bucket}.s3.amazonaws.com/"),
    };
    let mut target = Url::parse(&base).map_err(|e| e.to_string())?;
    if !key.is_empty() {
        let mut path = target.path_segments_mut().map_err(|_| "object URL has no path")?;
        path.clear();
        for part in key.split('/') { path.push(part); }
    }
    Ok(target)
}

fn list(client: &Client, source: &Source, origin: Option<&str>) -> Result<(), String> {
    let mut token = None::<String>;
    let mut seen = HashSet::new();
    let mut count = 0;
    loop {
        let mut target = url(&source.bucket, "", origin)?;
        {
            let mut query = target.query_pairs_mut();
            query.append_pair("list-type", "2").append_pair("max-keys", "1000")
                .append_pair("prefix", &source.prefix);
            if let Some(value) = &token { query.append_pair("continuation-token", value); }
        }
        let response = client.get(target).send().map_err(|e| e.to_string())?
            .error_for_status().map_err(|e| e.to_string())?;
        // A page is bounded metadata. Raw data bodies never enter XML/Python.
        let mut text = String::new();
        response.take(16 * 1024 * 1024 + 1).read_to_string(&mut text).map_err(|e| e.to_string())?;
        if text.len() > 16 * 1024 * 1024 { return Err("public listing page exceeds metadata limit".into()); }
        let page: Listing = quick_xml::de::from_str(&text).map_err(|e| e.to_string())?;
        for entry in page.contents {
            if !entry.key.starts_with(&source.prefix) { return Err("listing object escaped requested prefix".into()); }
            emit(json!({"phase":"object", "source":source.prefix, "bucket":source.bucket,
                "key":entry.key, "bytes":entry.size, "etag":entry.etag, "last_modified":entry.last_modified}))?;
            count += 1;
        }
        if !page.is_truncated { break; }
        let next = page.next_continuation_token.filter(|v| !v.is_empty())
            .ok_or("truncated listing has no continuation token")?;
        if !seen.insert(next.clone()) { return Err("listing repeated its continuation token".into()); }
        token = Some(next);
    }
    emit(json!({"phase":"source", "bucket":source.bucket, "prefix":source.prefix,
        "objects":count, "status":"listed"}))
}

// One shared allowance includes every object and every range, including retries.
struct Streams { free: Mutex<usize>, ready: Condvar }
struct Permit<'a>(&'a Streams);
impl Streams {
    fn take(&self) -> Permit<'_> {
        let mut free = self.free.lock().unwrap();
        while *free == 0 { free = self.ready.wait(free).unwrap(); }
        *free -= 1;
        Permit(self)
    }
}
impl Drop for Permit<'_> {
    fn drop(&mut self) { *self.0.free.lock().unwrap() += 1; self.0.ready.notify_one(); }
}

fn checked(response: &Response, object: &Object, start: u64, stop: u64, ranged: bool) -> Result<(), String> {
    let expected = if ranged { StatusCode::PARTIAL_CONTENT } else { StatusCode::OK };
    if response.status() == StatusCode::PRECONDITION_FAILED {
        return Err("object ETag differs from frozen inventory; replaced object refused".into());
    }
    if response.status() != expected { return Err(format!("object HTTP {} prevents a verified transfer", response.status())); }
    if let Some(etag) = &object.etag {
        if response.headers().get("etag").and_then(|v| v.to_str().ok()) != Some(etag.as_str()) {
            return Err("object ETag differs from frozen inventory; replaced object refused".into());
        }
    }
    if ranged {
        let span = format!("bytes {start}-{stop}/{}", object.bytes);
        if response.headers().get("content-range").and_then(|v| v.to_str().ok()) != Some(span.as_str()) {
            return Err("object Content-Range differs from frozen inventory; wrong span refused".into());
        }
    }
    if let Some(length) = response.content_length() {
        if length != stop - start + 1 { return Err("object size differs from frozen inventory; truncated or grown object refused".into()); }
    }
    Ok(())
}

fn part(client: &Client, object: &Object, target: &Url, start: u64, stop: u64,
        ranged: bool, streams: &Streams) -> Result<Vec<u8>, String> {
    let _permit = streams.take();
    let length = stop - start + 1;
    let mut last = String::new();
    for attempt in 0..3 {
        let mut request = client.get(target.clone()).header("Accept-Encoding", "identity");
        if let Some(etag) = &object.etag { request = request.header("If-Match", etag); }
        if ranged { request = request.header("Range", format!("bytes={start}-{stop}")); }
        match request.send() {
            Ok(mut response) => {
                if response.status().is_server_error() || response.status() == StatusCode::TOO_MANY_REQUESTS {
                    last = format!("public transfer HTTP {}", response.status());
                } else {
                    checked(&response, object, start, stop, ranged)?;
                    let mut bytes = Vec::with_capacity(length as usize);
                    match (&mut response).take(length + 1).read_to_end(&mut bytes) {
                        Ok(_) => {
                            if bytes.len() as u64 > length { return Err("object grew beyond frozen inventory".into()); }
                            if bytes.len() as u64 != length { return Err("object size differs from frozen inventory; truncated object refused".into()); }
                            return Ok(bytes);
                        }
                        Err(error) => last = error.to_string(),
                    }
                }
            }
            Err(error) => last = error.to_string(),
        }
        if attempt < 2 { std::thread::sleep(Duration::from_millis(250 * (1 << attempt))); }
    }
    Err(format!("public transfer failed after retries: {last}"))
}

fn download(client: &Client, object: &Object, origin: Option<&str>, streams: &Streams,
            limit: usize) -> Result<Value, String> {
    let target = url(&object.bucket, &object.key, origin)?;
    let path = &object.destination;
    if path.exists() || path.symlink_metadata().is_ok() { return Err("fresh object destination required".into()); }
    if let Some(parent) = path.parent() { fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
    let mut temporary = path.as_os_str().to_os_string(); temporary.push(".partial");
    let temporary = PathBuf::from(temporary);
    let mut output = OpenOptions::new().write(true).create_new(true).open(&temporary).map_err(|e| e.to_string())?;
    let ranged = object.bytes > CHUNK;
    let chunk = if ranged { CHUNK } else { object.bytes };
    let total = object.bytes.div_ceil(chunk);
    let width = limit.min(4) as u64;
    let mut digest = Sha256::new();
    let mut received = 0;
    // Bounded waves keep reordering memory <= four chunks per active object.
    for first in (0..total).step_by(width as usize) {
        let bodies: Result<Vec<Vec<u8>>, String> = std::thread::scope(|scope| {
            let jobs: Vec<_> = (first..(first + width).min(total)).map(|index| {
                let start = index * chunk;
                let stop = (start + chunk).min(object.bytes) - 1;
                let target = &target;
                scope.spawn(move || part(client, object, target, start, stop, ranged, streams))
            }).collect();
            jobs.into_iter().map(|job| job.join().map_err(|_| "object range worker panicked".to_string())?).collect()
        });
        for body in bodies? {
            // Hash the exact ordered bytes being written. No second file pass.
            output.write_all(&body).map_err(|e| e.to_string())?;
            digest.update(&body);
            received += body.len() as u64;
        }
    }
    if received != object.bytes { return Err("object size differs from frozen inventory".into()); }
    output.flush().map_err(|e| e.to_string())?;
    drop(output);
    // Hard-link publication is no-clobber even if another process creates path.
    fs::hard_link(&temporary, path).map_err(|e| format!("could not publish verified object: {e}"))?;
    fs::remove_file(&temporary).map_err(|e| e.to_string())?;
    let mut row = object.metadata.clone();
    row.extend(json!({"bucket":object.bucket, "key":object.key, "bytes":object.bytes,
        "etag":object.etag, "path":path, "sha256":format!("{:x}",digest.finalize()),
        "fetched_at_utc":chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()})
        .as_object().unwrap().clone());
    Ok(json!({"phase":"complete", "object":row, "range_gets":total}))
}

pub fn run(args: &[String]) -> Result<(), String> {
    if args.len() != 3 || args[1] != "--request" { return Err("public objects require --request JSON".into()); }
    let request: Request = serde_json::from_reader(File::open(&args[2]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let inventory = args[0] == "inventory";
    if !inventory {
        let mut destinations = HashSet::new();
        for object in &request.objects {
            url(&object.bucket, &object.key, request.origin.as_deref())?;
            if object.bytes == 0 || object.key.is_empty() || object.key.split('/').any(|p| p.is_empty() || p == "." || p == "..")
                || object.key.contains(['\\', ':', '\r', '\n', '\0']) || !destinations.insert(object.destination.clone()) {
                return Err("invalid frozen public object identity, size or duplicate destination".into());
            }
        }
    }
    let timeout = request.timeout_seconds.unwrap_or(60.0);
    if !timeout.is_finite() || timeout <= 0.0 || timeout > 600.0 {
        return Err("public request timeout must be positive and at most 600 seconds".into());
    }
    let client = Client::builder().timeout(Duration::from_secs_f64(timeout))
        // Feature unification in a full workspace build must never decode
        // an object's Content-Encoding before frozen size/hash verification.
        .no_gzip().no_brotli().no_deflate().no_zstd()
        .redirect(reqwest::redirect::Policy::none()).build().map_err(|e| e.to_string())?;
    let count = if inventory { request.sources.len() } else { request.objects.len() };
    let limit = count.min(32);
    if limit == 0 { return emit(json!({"phase":"end", "objects":0, "streams":0})); }
    let jobs = Arc::new(Mutex::new(VecDeque::from_iter(0..count)));
    let streams = Streams { free: Mutex::new(limit), ready: Condvar::new() };
    let errors = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..limit {
            let (client, request, jobs, streams, errors) = (&client, &request, &jobs, &streams, &errors);
            scope.spawn(move || loop {
                let Some(index) = jobs.lock().unwrap().pop_front() else { break; };
                let result = if inventory {
                    list(client, &request.sources[index], request.origin.as_deref())
                } else {
                    download(client, &request.objects[index], request.origin.as_deref(), streams, limit).and_then(emit)
                };
                if let Err(error) = result {
                    if inventory {
                        let source = &request.sources[index];
                        let _ = emit(json!({"phase":"source", "bucket":source.bucket,
                            "prefix":source.prefix, "status":"unavailable", "error":error}));
                    } else {
                        let _ = emit(json!({"phase":"error", "key":request.objects[index].key, "error":error}));
                        errors.lock().unwrap().push(error);
                    }
                }
            });
        }
    });
    if let Some(error) = errors.into_inner().unwrap().into_iter().next() { return Err(error); }
    emit(if inventory { json!({"phase":"end", "sources":count, "streams":limit}) }
         else { json!({"phase":"end", "objects":count, "streams":limit}) })
}
