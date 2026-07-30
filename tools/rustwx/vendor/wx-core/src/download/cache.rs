use std::path::PathBuf;

/// File-based download cache for GRIB2 data.
///
/// Stores downloaded data keyed by URL (and optional byte range) using a hash
/// of the key as the filename. Files are organized into subdirectories by the
/// first 2 hex characters of the hash to avoid huge flat directories.
///
/// All cache operations are fail-safe: if a read or write fails, the caller
/// gets `None` or the error is silently ignored, so caching never blocks
/// a network fetch.
pub struct DiskCache {
    dir: PathBuf,
}

impl DiskCache {
    /// Create a new cache using the platform-specific default directory.
    ///
    /// - Linux/macOS: `~/.cache/metrust/`
    /// - Windows: `%LOCALAPPDATA%/metrust/cache/`
    pub fn new() -> Self {
        let dir = default_cache_dir();
        std::fs::create_dir_all(&dir).ok();
        Self { dir }
    }

    /// Create a cache with a custom directory.
    pub fn with_dir(dir: PathBuf) -> Self {
        std::fs::create_dir_all(&dir).ok();
        Self { dir }
    }

    /// Build a cache key from a URL and an optional byte range.
    ///
    /// For full-file downloads, pass `None`. For range requests, pass the
    /// `(start, end)` pair so different ranges of the same URL get distinct
    /// cache entries.
    pub fn cache_key(url: &str, range: Option<(u64, u64)>) -> String {
        match range {
            Some((start, end)) => format!("{}|{}-{}", url, start, end),
            None => url.to_string(),
        }
    }

    /// Build a cache key for a multi-range request.
    ///
    /// Hashes the URL together with all ranges so the combined result is
    /// stored as a single cache entry.
    pub fn cache_key_ranges(url: &str, ranges: &[(u64, u64)]) -> String {
        let mut key = url.to_string();
        key.push_str("|ranges:");
        for (i, (start, end)) in ranges.iter().enumerate() {
            if i > 0 {
                key.push(',');
            }
            key.push_str(&format!("{}-{}", start, end));
        }
        key
    }

    /// Return the filesystem path where a given key would be stored.
    fn cache_path(&self, key: &str) -> PathBuf {
        let hash = hash_key(key);
        let prefix = &hash[..2];
        self.dir.join(prefix).join(format!("{}.grib2", hash))
    }

    /// The sidecar describing what the payload beside it is supposed to be.
    fn meta_path(&self, key: &str) -> PathBuf {
        self.cache_path(key).with_extension("meta")
    }

    /// Check if a key is cached and return the cached bytes.
    ///
    /// Returns `None` if the entry does not exist, cannot be read, or does not
    /// match the sidecar written beside it.
    ///
    /// Verification at USE, not only at write, is the point. `put` used to
    /// write the canonical file directly with `std::fs::write`, so a killed
    /// process left a prefix at the canonical name and `get` handed that prefix
    /// back as though it were the object -- with no length, no key and no
    /// checksum consulted. Worse, a persistently bad entry was never displaced,
    /// so every retry read the same poison and the outer validation failed
    /// forever. Now the sidecar records the exact key and byte count and a
    /// checksum of the payload, all three are re-checked on every hit, and an
    /// entry that fails is moved aside (never deleted) so the next request
    /// refetches instead of re-reading the same corruption.
    ///
    /// The checksum is FNV-1a-64 over the payload: an integrity check against
    /// truncation, partial writes and bit rot, which is what a local cache
    /// entry is exposed to. It is not a cryptographic seal and is not offered
    /// as one -- upstream authenticity is the caller's manifest digest, and the
    /// GRIB envelope/record-count bars above this layer remain the completeness
    /// gate.
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        let path = self.cache_path(key);
        let data = std::fs::read(&path).ok()?;
        match std::fs::read_to_string(self.meta_path(key)) {
            Ok(text) if meta_matches(&text, key, &data) => Some(data),
            Ok(_) => {
                self.quarantine(key, "content does not match its sidecar");
                None
            }
            Err(_) => {
                // A payload with no sidecar is an entry from an interrupted
                // write or from before this schema: unverifiable either way.
                self.quarantine(key, "no sidecar to verify it against");
                None
            }
        }
    }

    /// Store bytes in the cache under the given key.
    ///
    /// Creates the subdirectory if needed. Errors are printed to stderr
    /// but never propagated.
    ///
    /// The payload lands by atomic rename from a name unique to this process
    /// and this call, so the canonical path never holds a partial write and two
    /// writers never share a staging file. The sidecar is published after it,
    /// so a kill in between leaves an unverifiable entry that `get` refuses
    /// rather than a verified-looking prefix.
    pub fn put(&self, key: &str, data: &[u8]) {
        let path = self.cache_path(key);
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("metrust cache: failed to create dir {:?}: {}", parent, e);
                return;
            }
        }
        if let Err(e) = atomic_write(&path, data) {
            eprintln!("metrust cache: failed to write {:?}: {}", path, e);
            return;
        }
        let meta = format!(
            "key={}\nbytes={}\nfnv1a64={:016x}\n",
            key.replace('\n', " "),
            data.len(),
            hash_bytes(data)
        );
        if let Err(e) = atomic_write(&self.meta_path(key), meta.as_bytes()) {
            eprintln!("metrust cache: failed to write sidecar for {:?}: {}", path, e);
        }
    }

    /// Move a suspect entry aside so the next request refetches it.
    ///
    /// Renamed, never removed: the bytes that failed are the evidence of what
    /// arrived, and a cache that deletes on suspicion cannot be diagnosed.
    fn quarantine(&self, key: &str, reason: &str) {
        let path = self.cache_path(key);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        let mut aside = path.with_extension(format!("rejected-{}", stamp));
        let mut counter = 0u32;
        while aside.exists() {
            counter += 1;
            aside = path.with_extension(format!("rejected-{}-{}", stamp, counter));
        }
        if std::fs::rename(&path, &aside).is_ok() {
            let aside_meta = PathBuf::from(format!("{}.meta", aside.display()));
            let _ = std::fs::rename(self.meta_path(key), aside_meta);
            eprintln!(
                "metrust cache: set {:?} aside as {:?} ({})",
                path, aside, reason
            );
        }
    }

    /// Check if a key is cached without reading the data.
    ///
    /// A payload without its sidecar is not a hit: nothing can vouch for it.
    pub fn has(&self, key: &str) -> bool {
        self.cache_path(key).exists() && self.meta_path(key).exists()
    }

    /// Alias for `has` — backward compatibility with the old Cache API.
    pub fn contains(&self, key: &str) -> bool {
        self.has(key)
    }

    /// Remove all cached files and subdirectories.
    ///
    /// Recreates the empty cache directory after clearing.
    pub fn clear(&self) {
        if self.dir.exists() {
            std::fs::remove_dir_all(&self.dir).ok();
        }
        std::fs::create_dir_all(&self.dir).ok();
    }

    /// Total size of cached data in bytes.
    ///
    /// Walks the cache directory tree and sums file sizes. Returns 0 if the
    /// directory cannot be read.
    pub fn size(&self) -> u64 {
        dir_size(&self.dir)
    }

    /// Remove a cached entry by key.
    pub fn remove(&self, key: &str) {
        let path = self.cache_path(key);
        std::fs::remove_file(&path).ok();
    }

    /// Return the cache directory path.
    pub fn dir(&self) -> &PathBuf {
        &self.dir
    }
}

/// Write `data` at `path` by atomic rename from a per-call staging name.
fn atomic_write(path: &PathBuf, data: &[u8]) -> std::io::Result<()> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    let tmp = path.with_extension(format!("tmp-{}-{}", std::process::id(), stamp));
    if let Err(err) = std::fs::write(&tmp, data) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }
    if let Err(err) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }
    Ok(())
}

/// Whether a sidecar describes exactly the payload beside it.
fn meta_matches(text: &str, key: &str, data: &[u8]) -> bool {
    let mut recorded_key: Option<&str> = None;
    let mut recorded_bytes: Option<u64> = None;
    let mut recorded_hash: Option<&str> = None;
    for line in text.lines() {
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        match name.trim() {
            "key" => recorded_key = Some(value),
            "bytes" => recorded_bytes = value.trim().parse::<u64>().ok(),
            "fnv1a64" => recorded_hash = Some(value.trim()),
            _ => {}
        }
    }
    let (Some(recorded_key), Some(recorded_bytes), Some(recorded_hash)) =
        (recorded_key, recorded_bytes, recorded_hash)
    else {
        return false;
    };
    recorded_key == key.replace('\n', " ")
        && recorded_bytes == data.len() as u64
        && recorded_hash == format!("{:016x}", hash_bytes(data))
}

/// Recursively compute the total size of all files in a directory.
fn dir_size(path: &PathBuf) -> u64 {
    let mut total: u64 = 0;
    let entries = match std::fs::read_dir(path) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    for entry in entries.flatten() {
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.is_dir() {
            total += dir_size(&entry.path().to_path_buf());
        } else {
            total += meta.len();
        }
    }
    total
}

/// Platform-specific default cache directory.
fn default_cache_dir() -> PathBuf {
    // Windows: %LOCALAPPDATA%/metrust/cache/
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(local).join("metrust").join("cache");
    }
    // XDG_CACHE_HOME or ~/.cache on Unix
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("metrust");
    }
    if let Some(home) = home_dir() {
        return home.join(".cache").join("metrust");
    }
    // Last resort
    PathBuf::from(".metrust").join("cache")
}

/// Get the user's home directory.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// FNV-1a 64-bit hash, rendered as 16 hex characters.
///
/// Chosen for speed and good distribution. Not cryptographic, but
/// collisions are vanishingly unlikely for URL-shaped keys.
fn hash_key(s: &str) -> String {
    format!("{:016x}", hash_bytes(s.as_bytes()))
}

/// FNV-1a 64-bit over raw bytes.
///
/// Used both to name entries and to checksum their payloads. As a payload
/// checksum it detects truncation, partial writes and bit rot; it is not a
/// cryptographic digest and no caller treats it as one.
fn hash_bytes(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h = FNV_OFFSET;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

// ──────────────────────────────────────────────────────────
// Backward-compatible alias
// ──────────────────────────────────────────────────────────

/// Alias for `DiskCache` — kept for backward compatibility.
pub type Cache = DiskCache;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_deterministic() {
        let h1 = hash_key("https://example.com/data.grib2");
        let h2 = hash_key("https://example.com/data.grib2");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 16);
    }

    #[test]
    fn test_hash_different_urls() {
        let h1 = hash_key("https://example.com/a.grib2");
        let h2 = hash_key("https://example.com/b.grib2");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_cache_key_no_range() {
        let key = DiskCache::cache_key("https://x.com/file", None);
        assert_eq!(key, "https://x.com/file");
    }

    #[test]
    fn test_cache_key_with_range() {
        let key = DiskCache::cache_key("https://x.com/file", Some((100, 200)));
        assert_eq!(key, "https://x.com/file|100-200");
    }

    #[test]
    fn test_cache_key_ranges() {
        let key = DiskCache::cache_key_ranges("https://x.com/f", &[(0, 100), (200, 300)]);
        assert_eq!(key, "https://x.com/f|ranges:0-100,200-300");
    }

    #[test]
    fn test_roundtrip() {
        let tmp = std::env::temp_dir().join("metrust_test_cache");
        let cache = DiskCache::with_dir(tmp.clone());
        let key = DiskCache::cache_key("https://example.com/test", None);

        cache.put(&key, b"hello world");
        assert!(cache.has(&key));
        assert_eq!(cache.get(&key), Some(b"hello world".to_vec()));
        assert!(cache.size() > 0);

        cache.clear();
        assert!(!cache.has(&key));
        assert_eq!(cache.size(), 0);

        // Clean up
        std::fs::remove_dir_all(&tmp).ok();
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("metrust_cache_{}", name));
        std::fs::remove_dir_all(&dir).ok();
        dir
    }

    /// A truncated entry was handed back as though it were the object.
    #[test]
    fn test_a_truncated_entry_is_a_miss_and_is_set_aside() {
        let dir = scratch("truncated");
        let cache = DiskCache::with_dir(dir.clone());
        let key = DiskCache::cache_key("https://example.com/a.grib2", None);
        cache.put(&key, b"a complete payload");
        assert_eq!(cache.get(&key), Some(b"a complete payload".to_vec()));

        let path = cache.cache_path(&key);
        std::fs::write(&path, b"a comp").unwrap();
        assert_eq!(cache.get(&key), None, "a prefix must not be a hit");

        // Set aside, not deleted, and no longer on the hit path.
        assert!(!path.exists());
        assert!(!cache.has(&key));
        let aside: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .filter(|name| name.contains("rejected-"))
            .collect();
        assert_eq!(aside.len(), 2, "payload and sidecar both kept: {:?}", aside);

        // The next put repopulates cleanly.
        cache.put(&key, b"a complete payload");
        assert_eq!(cache.get(&key), Some(b"a complete payload".to_vec()));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A same-length mutation is still caught by the payload checksum.
    #[test]
    fn test_a_same_length_mutation_is_a_miss() {
        let dir = scratch("mutated");
        let cache = DiskCache::with_dir(dir.clone());
        let key = DiskCache::cache_key("https://example.com/b.grib2", None);
        cache.put(&key, b"original bytes");
        std::fs::write(cache.cache_path(&key), b"mutated!_bytes").unwrap();
        assert_eq!(cache.get(&key), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A payload written by an older schema (no sidecar) is unverifiable.
    #[test]
    fn test_a_sidecarless_payload_is_never_returned() {
        let dir = scratch("legacy");
        let cache = DiskCache::with_dir(dir.clone());
        let key = DiskCache::cache_key("https://example.com/c.grib2", None);
        let path = cache.cache_path(&key);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"legacy bytes of unknown provenance").unwrap();
        assert!(!cache.has(&key));
        assert_eq!(cache.get(&key), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Two keys that differ only by range must not verify each other.
    #[test]
    fn test_the_sidecar_binds_the_key_not_just_the_bytes() {
        let dir = scratch("keybound");
        let cache = DiskCache::with_dir(dir.clone());
        let first = DiskCache::cache_key("https://example.com/d.grib2", Some((0, 9)));
        let second = DiskCache::cache_key("https://example.com/d.grib2", Some((10, 19)));
        cache.put(&first, b"0123456789");
        // Move the first entry's payload under the second key's name, leaving
        // the second key's sidecar absent: it must not be adopted.
        let target = cache.cache_path(&second);
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::copy(cache.cache_path(&first), &target).unwrap();
        std::fs::copy(cache.meta_path(&first), cache.meta_path(&second)).unwrap();
        assert_eq!(cache.get(&second), None, "a foreign key's sidecar must not vouch");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_put_leaves_no_staging_file_behind() {
        let dir = scratch("staging");
        let cache = DiskCache::with_dir(dir.clone());
        let key = DiskCache::cache_key("https://example.com/e.grib2", None);
        cache.put(&key, b"payload");
        let names: Vec<_> = std::fs::read_dir(cache.cache_path(&key).parent().unwrap())
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(names.len(), 2, "payload + sidecar only: {:?}", names);
        assert!(!names.iter().any(|name| name.contains(".tmp-")));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_cache_path_has_prefix_subdir() {
        let cache = DiskCache::with_dir(PathBuf::from("/tmp/test"));
        let path = cache.cache_path("some key");
        let hash = hash_key("some key");
        let prefix = &hash[..2];
        assert!(path.to_string_lossy().contains(prefix));
        assert!(path.to_string_lossy().ends_with(".grib2"));
    }
}
