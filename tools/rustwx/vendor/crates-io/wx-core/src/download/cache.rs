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

    /// Check if a key is cached and return the cached bytes.
    ///
    /// Returns `None` if the entry does not exist or cannot be read.
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        let path = self.cache_path(key);
        std::fs::read(&path).ok()
    }

    /// Store bytes in the cache under the given key.
    ///
    /// Creates the subdirectory if needed. Errors are printed to stderr
    /// but never propagated.
    pub fn put(&self, key: &str, data: &[u8]) {
        let path = self.cache_path(key);
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("metrust cache: failed to create dir {:?}: {}", parent, e);
                return;
            }
        }
        if let Err(e) = std::fs::write(&path, data) {
            eprintln!("metrust cache: failed to write {:?}: {}", path, e);
        }
    }

    /// Check if a key is cached without reading the data.
    pub fn has(&self, key: &str) -> bool {
        self.cache_path(key).exists()
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
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h = FNV_OFFSET;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    format!("{:016x}", h)
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
