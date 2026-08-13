use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

/// Subdirectory holding exactly one copy of each distinct payload.
///
/// Key subdirectories are named by two hex characters, so a word can never
/// collide with one.
const CONTENT_DIR: &str = "content";

/// The hard-link probe has not run yet.
const LINK_UNPROBED: u8 = 0;
/// The cache directory's filesystem hard-links.
const LINK_SUPPORTED: u8 = 1;
/// It does not, so entries name their payload instead of linking to it.
const LINK_UNSUPPORTED: u8 = 2;

/// What a cache had to write versus what it already held.
///
/// `bytes_written + bytes_deduplicated` is the payload every `put` presented;
/// only the first half is disk. `reference_entries` counts the key entries
/// published against content that was already stored.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheDedup {
    pub bytes_written: u64,
    pub bytes_deduplicated: u64,
    pub reference_entries: u64,
}

/// File-based download cache for GRIB2 data.
///
/// Stores downloaded data keyed by URL (and optional byte range) using a hash
/// of the key as the filename. Files are organized into subdirectories by the
/// first 2 hex characters of the hash to avoid huge flat directories.
///
/// The payload itself lives once, under `content/`, named by its own FNV-1a-64
/// and length; every key entry is a hard link to that copy, or -- where the
/// filesystem cannot hard-link -- a pointer file naming it. One full-file
/// fetch reaches this cache under two different key shapes (URL-only from
/// `get_bytes_parallel_whole`, URL+ranges from `get_ranges`), and those two
/// stores used to be two complete copies of the same object. Dedup is decided
/// on the CONTENT, never on the assumption that two key shapes mean the same
/// thing: a range list that does not cover the whole object legitimately
/// yields different bytes, and it gets its own copy.
///
/// All cache operations are fail-safe: if a read or write fails, the caller
/// gets `None` or the error is silently ignored, so caching never blocks
/// a network fetch.
pub struct DiskCache {
    dir: PathBuf,
    /// Payload bytes this cache actually laid down.
    bytes_written: AtomicU64,
    /// Payload bytes a `put` did not have to write, because the content store
    /// already held them.
    bytes_deduplicated: AtomicU64,
    /// Key entries published against already-stored content.
    reference_entries: AtomicU64,
    /// Result of the hard-link probe: one of the `LINK_*` constants.
    link_support: AtomicU8,
}

impl DiskCache {
    /// Create a new cache using the platform-specific default directory.
    ///
    /// - Linux/macOS: `~/.cache/metrust/`
    /// - Windows: `%LOCALAPPDATA%/metrust/cache/`
    pub fn new() -> Self {
        Self::with_dir(default_cache_dir())
    }

    /// Create a cache with a custom directory.
    pub fn with_dir(dir: PathBuf) -> Self {
        std::fs::create_dir_all(&dir).ok();
        Self {
            dir,
            bytes_written: AtomicU64::new(0),
            bytes_deduplicated: AtomicU64::new(0),
            reference_entries: AtomicU64::new(0),
            link_support: AtomicU8::new(LINK_UNPROBED),
        }
    }

    /// What this cache wrote versus what it already had, since it was created.
    ///
    /// Reported so a fetch receipt can carry the cost of its own cache rather
    /// than leaving a multiple of the payload on disk unaccounted for.
    pub fn dedup(&self) -> CacheDedup {
        CacheDedup {
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            bytes_deduplicated: self.bytes_deduplicated.load(Ordering::Relaxed),
            reference_entries: self.reference_entries.load(Ordering::Relaxed),
        }
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

    /// The pointer an entry uses when the filesystem cannot hard-link.
    ///
    /// A separate file rather than a line in the sidecar: `meta_matches` is
    /// the integrity check and it stays about the bytes, and an entry keeps
    /// the same two-file shape in both forms.
    fn ref_path(&self, key: &str) -> PathBuf {
        self.cache_path(key).with_extension("ref")
    }

    /// Where a given content id's single copy lives.
    fn blob_path(&self, content: &str) -> PathBuf {
        self.dir
            .join(CONTENT_DIR)
            .join(&content[..2])
            .join(format!("{}.blob", content))
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
    ///
    /// A reference entry is held to exactly the same standard: the bytes it
    /// resolves to are checked against ITS OWN sidecar, so a reference that
    /// dangles or resolves to the wrong content is a miss and is set aside,
    /// never a wrong-bytes hit.
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        let data = match self.read_entry(key) {
            Entry::Payload(data) => data,
            Entry::Broken(reason) => {
                self.quarantine(key, reason);
                return None;
            }
            Entry::Absent => return None,
        };
        match std::fs::read_to_string(self.meta_path(key)) {
            Ok(text) if meta_matches(&text, key, &data) => Some(data),
            Ok(text) => {
                self.quarantine(key, "content does not match its sidecar");
                self.reap_canonical(&text);
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

    /// The bytes an entry stands for, however it stands for them.
    ///
    /// The payload path is consulted first, so a hard-linked entry and an
    /// entry written before the content store existed both read directly.
    /// Only when there is no payload at all does the pointer form apply.
    fn read_entry(&self, key: &str) -> Entry {
        if let Ok(data) = std::fs::read(self.cache_path(key)) {
            return Entry::Payload(data);
        }
        let Ok(text) = std::fs::read_to_string(self.ref_path(key)) else {
            return Entry::Absent;
        };
        match self.reference_target(&text) {
            Some(target) => match std::fs::read(target) {
                Ok(data) => Entry::Payload(data),
                Err(_) => Entry::Broken("its reference names a payload that is gone"),
            },
            None => Entry::Broken("its reference does not name a payload"),
        }
    }

    /// Resolve a pointer file's `canonical=` line against the cache root.
    ///
    /// Relative on purpose, so a cache directory stays movable, and rejected
    /// unless every component is an ordinary name: a pointer is a file on
    /// disk like any other, and a cache must not be talked into reading
    /// outside itself by one.
    fn reference_target(&self, text: &str) -> Option<PathBuf> {
        let relative = text
            .lines()
            .filter_map(|line| line.split_once('='))
            .find(|(name, _)| name.trim() == "canonical")
            .map(|(_, value)| value.trim())?;
        if relative.is_empty() {
            return None;
        }
        let mut target = self.dir.clone();
        for part in relative.split('/') {
            if part.is_empty() || part == "." || part == ".." {
                return None;
            }
            target.push(part);
        }
        Some(target)
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
    ///
    /// Content the store already holds is not written twice. The decision is
    /// the content id -- the same FNV-1a-64 and length the sidecar already
    /// carries -- so it holds for any two keys that happen to name the same
    /// bytes and for no two keys that do not.
    pub fn put(&self, key: &str, data: &[u8]) {
        let content = content_id(data);
        let blob = self.blob_path(&content);
        let reused = blob_holds(&blob, data.len());
        if !reused {
            // A same-length blob is taken at its name: re-reading it on every
            // put would cost a full read per store to catch what `get`
            // already catches at use, where it also refetches.
            if let Some(parent) = blob.parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    eprintln!("metrust cache: failed to create dir {:?}: {}", parent, e);
                    return;
                }
            }
            if let Err(e) = atomic_write(&blob, data) {
                eprintln!("metrust cache: failed to write {:?}: {}", blob, e);
                return;
            }
            self.bytes_written
                .fetch_add(data.len() as u64, Ordering::Relaxed);
        }

        let path = self.cache_path(key);
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("metrust cache: failed to create dir {:?}: {}", parent, e);
                return;
            }
        }
        if !self.link_entry(key, &blob) {
            return;
        }
        // Counted where the entry is actually published, so a store that
        // failed halfway is not reported as a saving.
        if reused {
            self.bytes_deduplicated
                .fetch_add(data.len() as u64, Ordering::Relaxed);
            self.reference_entries.fetch_add(1, Ordering::Relaxed);
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

    /// Publish this key's entry against the canonical payload.
    ///
    /// A hard link is preferred because it costs a directory entry and no
    /// data blocks, and because it leaves `get` reading an ordinary file.
    /// Filesystems that cannot hard-link (FAT, some network mounts) are not
    /// hypothetical, so support is probed once rather than assumed, and those
    /// entries fall back to a pointer file naming the canonical payload.
    fn link_entry(&self, key: &str, blob: &Path) -> bool {
        let path = self.cache_path(key);
        let reference = self.ref_path(key);
        if self.hard_links_work() {
            let staging = staging_path(&path);
            match std::fs::hard_link(blob, &staging) {
                Ok(()) => {
                    if let Err(e) = std::fs::rename(&staging, &path) {
                        let _ = std::fs::remove_file(&staging);
                        eprintln!("metrust cache: failed to publish {:?}: {}", path, e);
                        return false;
                    }
                    // A pointer left by an earlier fallback would be a second,
                    // staler answer for this key. It holds no payload bytes,
                    // so removing it destroys no evidence.
                    let _ = std::fs::remove_file(&reference);
                    return true;
                }
                Err(e) => {
                    let _ = std::fs::remove_file(&staging);
                    // A link that fails where the probe said links work is
                    // this call's problem, not the filesystem's; re-probe
                    // rather than condemning every later entry to a pointer.
                    if self.probe_hard_links() {
                        eprintln!("metrust cache: could not link {:?}: {}", path, e);
                    } else {
                        eprintln!(
                            "metrust cache: {:?} cannot hard-link ({}); entries will \
                             name the canonical payload instead",
                            self.dir, e
                        );
                    }
                }
            }
        }
        // A stale payload here would outrank the pointer in `read_entry` and
        // answer for a key it no longer describes: set it aside, which is
        // this cache's rule for suspect bytes -- never deleted.
        if path.exists() {
            self.set_aside(&path, "a payload displaced by a reference entry");
        }
        let pointer = format!("canonical={}\n", self.relative_to_root(blob));
        if let Err(e) = atomic_write(&reference, pointer.as_bytes()) {
            eprintln!("metrust cache: failed to write reference {:?}: {}", reference, e);
            return false;
        }
        true
    }

    /// Whether this cache directory's filesystem hard-links, probed once.
    fn hard_links_work(&self) -> bool {
        match self.link_support.load(Ordering::Relaxed) {
            LINK_SUPPORTED => true,
            LINK_UNSUPPORTED => false,
            _ => self.probe_hard_links(),
        }
    }

    /// Prove a hard link works here, with a real link, and remember the answer.
    fn probe_hard_links(&self) -> bool {
        let works = hard_links_supported(&self.dir);
        self.link_support.store(
            if works { LINK_SUPPORTED } else { LINK_UNSUPPORTED },
            Ordering::Relaxed,
        );
        works
    }

    /// Store entries in the portable pointer form, as a link-less filesystem
    /// would. The fallback is otherwise unreachable on a developer machine
    /// whose disk hard-links happily, and it is the form that has to be
    /// verified just as strictly.
    #[cfg(test)]
    fn without_hard_links(self) -> Self {
        self.link_support.store(LINK_UNSUPPORTED, Ordering::Relaxed);
        self
    }

    /// A path inside the cache, written the way a pointer file records it.
    fn relative_to_root(&self, path: &Path) -> String {
        path.strip_prefix(&self.dir)
            .unwrap_or(path)
            .components()
            .map(|part| part.as_os_str().to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join("/")
    }

    /// Move a suspect entry aside so the next request refetches it.
    ///
    /// Renamed, never removed: the bytes that failed are the evidence of what
    /// arrived, and a cache that deletes on suspicion cannot be diagnosed.
    ///
    /// All three of an entry's files travel together under one aside name. A
    /// reference entry has no payload of its own, and leaving its pointer or
    /// its sidecar behind would leave half an entry claiming to be a whole
    /// one.
    fn quarantine(&self, key: &str, reason: &str) {
        let path = self.cache_path(key);
        let stamp = now_nanos();
        let mut aside = path.with_extension(format!("rejected-{}", stamp));
        let mut counter = 0u32;
        while aside.exists() {
            counter += 1;
            aside = path.with_extension(format!("rejected-{}-{}", stamp, counter));
        }
        let mut moved = std::fs::rename(&path, &aside).is_ok();
        let aside_ref = PathBuf::from(format!("{}.ref", aside.display()));
        moved |= std::fs::rename(self.ref_path(key), aside_ref).is_ok();
        let aside_meta = PathBuf::from(format!("{}.meta", aside.display()));
        moved |= std::fs::rename(self.meta_path(key), aside_meta).is_ok();
        if moved {
            eprintln!(
                "metrust cache: set {:?} aside as {:?} ({})",
                path, aside, reason
            );
        }
    }

    /// Set aside the canonical payload a failed sidecar describes, when that
    /// payload no longer hashes to its own name.
    ///
    /// A reference is only as good as the content it names. Without this, the
    /// refused key would be refetched, `put` would find the poisoned blob
    /// still claiming that content id at that length, and link the fresh key
    /// straight back onto it -- the never-displaced-poison failure this cache
    /// already fixed once, reintroduced one level down. The blob is read
    /// before it is condemned, so a corrupted sidecar over sound content
    /// costs a refetch and not the content.
    fn reap_canonical(&self, text: &str) {
        let Some(content) = declared_content_id(text) else {
            return;
        };
        let blob = self.blob_path(&content);
        let Ok(bytes) = std::fs::read(&blob) else {
            return;
        };
        if content_id(&bytes) == content {
            return;
        }
        self.set_aside(&blob, "the canonical payload no longer matches its content id");
    }

    /// Rename one file aside, under the same never-deleted rule as `quarantine`.
    fn set_aside(&self, path: &Path, reason: &str) {
        let stamp = now_nanos();
        let mut aside = path.with_extension(format!("rejected-{}", stamp));
        let mut counter = 0u32;
        while aside.exists() {
            counter += 1;
            aside = path.with_extension(format!("rejected-{}-{}", stamp, counter));
        }
        if std::fs::rename(path, &aside).is_ok() {
            eprintln!(
                "metrust cache: set {:?} aside as {:?} ({})",
                path, aside, reason
            );
        }
    }

    /// Check if a key is cached without reading the data.
    ///
    /// A payload without its sidecar is not a hit: nothing can vouch for it.
    /// Neither is a reference whose canonical payload is gone -- the pointer
    /// on its own vouches for nothing.
    pub fn has(&self, key: &str) -> bool {
        if !self.meta_path(key).exists() {
            return false;
        }
        if self.cache_path(key).exists() {
            return true;
        }
        std::fs::read_to_string(self.ref_path(key))
            .ok()
            .and_then(|text| self.reference_target(&text))
            .map(|target| target.exists())
            .unwrap_or(false)
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
    ///
    /// A hard-linked entry reports the full length of the payload it shares,
    /// so summing every name would report one object as many times as it has
    /// keys -- exactly the inflation this cache exists to stop reporting. A
    /// key payload whose sidecar names content the store already holds is
    /// therefore counted at the content store and not again at the key.
    pub fn size(&self) -> u64 {
        self.walk_size(&self.dir)
    }

    fn walk_size(&self, path: &Path) -> u64 {
        let mut total: u64 = 0;
        let entries = match std::fs::read_dir(path) {
            Ok(entries) => entries,
            Err(_) => return 0,
        };
        for entry in entries.flatten() {
            let meta = match entry.metadata() {
                Ok(meta) => meta,
                Err(_) => continue,
            };
            if meta.is_dir() {
                total += self.walk_size(&entry.path());
            } else if !self.shares_stored_content(&entry.path()) {
                total += meta.len();
            }
        }
        total
    }

    /// Is this file a key entry's name for a payload the content store holds?
    fn shares_stored_content(&self, path: &Path) -> bool {
        if path.extension().and_then(|value| value.to_str()) != Some("grib2") {
            return false;
        }
        let Ok(text) = std::fs::read_to_string(path.with_extension("meta")) else {
            return false;
        };
        declared_content_id(&text)
            .map(|content| self.blob_path(&content).exists())
            .unwrap_or(false)
    }

    /// Remove a cached entry by key.
    ///
    /// All of the entry goes: the payload name, the pointer form of it, and
    /// the sidecar. Removing only the payload left the sidecar behind to
    /// describe a file that no longer existed, and with references in play
    /// that orphan would be an entry claiming content nothing can produce.
    ///
    /// The canonical payload stays. Other keys may still name it, and the
    /// filesystem is the only thing that knows how many do; `clear` is the
    /// reclaim path.
    pub fn remove(&self, key: &str) {
        std::fs::remove_file(self.cache_path(key)).ok();
        std::fs::remove_file(self.ref_path(key)).ok();
        std::fs::remove_file(self.meta_path(key)).ok();
    }

    /// Return the cache directory path.
    pub fn dir(&self) -> &PathBuf {
        &self.dir
    }
}

/// What one key entry resolves to, or why it does not.
enum Entry {
    Payload(Vec<u8>),
    /// The entry exists and cannot produce its bytes: a miss AND a fault.
    Broken(&'static str),
    /// Nothing is stored under this key: an ordinary miss.
    Absent,
}

/// Nanoseconds since the epoch, or 0 if the clock refuses.
fn now_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0)
}

/// A staging name unique to this process, this call, and this instant.
///
/// The sequence number is what makes it unique between two threads storing
/// the SAME content: they share a destination, and a shared staging file
/// would interleave two writes into one name.
fn staging_path(path: &Path) -> PathBuf {
    use std::sync::atomic::AtomicU64;
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let seq = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    path.with_extension(format!("tmp-{}-{}-{}", std::process::id(), now_nanos(), seq))
}

/// Write `data` at `path` by atomic rename from a per-call staging name.
fn atomic_write(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let tmp = staging_path(path);
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

/// The name one payload's single copy is stored under.
///
/// Its own FNV-1a-64 and its length, which is exactly what the sidecar
/// already records -- so two entries are recognised as the same object by
/// what they contain, never by what their keys look like.
fn content_id(data: &[u8]) -> String {
    format!("{:016x}-{}", hash_bytes(data), data.len())
}

/// The content id a sidecar declares, if it declares a well-formed one.
///
/// A sidecar is a file on disk and may be anything; only 16 hex digits and a
/// length are allowed to become a path component.
fn declared_content_id(text: &str) -> Option<String> {
    let mut recorded_bytes: Option<u64> = None;
    let mut recorded_hash: Option<&str> = None;
    for line in text.lines() {
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        match name.trim() {
            "bytes" => recorded_bytes = value.trim().parse::<u64>().ok(),
            "fnv1a64" => recorded_hash = Some(value.trim()),
            _ => {}
        }
    }
    let (bytes, hash) = (recorded_bytes?, recorded_hash?);
    if hash.len() != 16 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    Some(format!("{}-{}", hash, bytes))
}

/// Does the content store already hold this payload?
///
/// The name asserts the hash, so the length is the whole check here. Content
/// that lies about either is caught at use, where `get` refuses it, sets the
/// blob aside and lets the next request refetch.
fn blob_holds(blob: &Path, len: usize) -> bool {
    std::fs::metadata(blob)
        .map(|meta| meta.is_file() && meta.len() == len as u64)
        .unwrap_or(false)
}

/// Prove a hard link works in `dir` by making one and removing it.
///
/// Probed rather than assumed, and probed where the entries will live: hard
/// links are an NTFS/POSIX facility that FAT volumes and some network mounts
/// do not offer, and a cache that assumed them would store nothing at all on
/// those. Both probe files are removed on every path.
fn hard_links_supported(dir: &Path) -> bool {
    let source = dir.join(format!(".link-probe-{}-{}", std::process::id(), now_nanos()));
    let link = PathBuf::from(format!("{}.link", source.display()));
    if std::fs::write(&source, b"metrust cache hard-link probe\n").is_err() {
        return false;
    }
    let works = std::fs::hard_link(&source, &link).is_ok();
    let _ = std::fs::remove_file(&link);
    let _ = std::fs::remove_file(&source);
    works
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

    /// Every file the cache holds, deepest first, as (name, bytes).
    fn every_file(dir: &Path) -> Vec<(String, u64)> {
        let mut found = Vec::new();
        let Ok(entries) = std::fs::read_dir(dir) else {
            return found;
        };
        for entry in entries.flatten() {
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                found.extend(every_file(&entry.path()));
            } else {
                found.push((entry.file_name().to_string_lossy().to_string(), meta.len()));
            }
        }
        found
    }

    /// The distinct payloads on disk: the content store's own files.
    fn stored_payloads(dir: &Path) -> Vec<(String, u64)> {
        let mut found = every_file(&dir.join(CONTENT_DIR));
        found.retain(|(name, _)| name.ends_with(".blob"));
        found
    }

    /// A 64 KiB body, so the sidecars are noise beside the payload.
    fn body(seed: u8) -> Vec<u8> {
        (0..65536u32)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    /// One full-file fetch stores the object under a URL-only key AND a
    /// URL+ranges key. Those were two complete copies.
    #[test]
    fn test_one_object_under_two_keys_keeps_one_payload() {
        let dir = scratch("twokeys");
        let cache = DiskCache::with_dir(dir.clone());
        let url = "https://example.com/whole.grib2";
        let whole = DiskCache::cache_key(url, None);
        let ranges = DiskCache::cache_key_ranges(url, &[(0, 32767), (32768, 65535)]);
        let data = body(1);

        cache.put(&ranges, &data);
        cache.put(&whole, &data);

        // Both keys still answer, with the same bytes as ever.
        assert_eq!(cache.get(&ranges).as_deref(), Some(data.as_slice()));
        assert_eq!(cache.get(&whole).as_deref(), Some(data.as_slice()));
        assert!(cache.has(&ranges) && cache.has(&whole));

        let payloads = stored_payloads(&dir);
        assert_eq!(payloads.len(), 1, "one object, one payload: {:?}", payloads);
        assert_eq!(payloads[0].1, data.len() as u64);

        // And the cache says so: one copy written, one deduplicated.
        let dedup = cache.dedup();
        assert_eq!(dedup.bytes_written, data.len() as u64);
        assert_eq!(dedup.bytes_deduplicated, data.len() as u64);
        assert_eq!(dedup.reference_entries, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The bytes, not the entry count: two keys must cost about one payload.
    #[test]
    fn test_two_keys_for_one_object_cost_one_payload_of_disk() {
        let dir = scratch("onepayload");
        let cache = DiskCache::with_dir(dir.clone());
        let url = "https://example.com/measured.grib2";
        let data = body(2);
        cache.put(&DiskCache::cache_key_ranges(url, &[(0, 65535)]), &data);
        cache.put(&DiskCache::cache_key(url, None), &data);

        let payload = data.len() as u64;
        // The content store holds the object exactly once. Key entries are
        // hard links to it and consume directory entries, not data blocks,
        // so `size` counts the shared payload where it lives.
        let stored: u64 = stored_payloads(&dir).iter().map(|(_, len)| len).sum();
        assert_eq!(stored, payload, "one copy of the object on disk");
        assert!(
            cache.size() < payload + payload / 5,
            "cache reports {} for a {} byte object",
            cache.size(),
            payload
        );
        // Non-payload files -- sidecars and pointers -- stay small change.
        let overhead: u64 = every_file(&dir)
            .iter()
            .filter(|(name, _)| !name.ends_with(".blob") && !name.ends_with(".grib2"))
            .map(|(_, len)| len)
            .sum();
        assert!(overhead < payload / 10, "sidecar overhead {}", overhead);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The negative control: dedup is keyed on content, so different content
    /// under two keys is still two payloads.
    #[test]
    fn test_two_different_objects_still_cost_two_payloads() {
        let dir = scratch("distinct");
        let cache = DiskCache::with_dir(dir.clone());
        let url = "https://example.com/partial.grib2";
        // A range list that does NOT cover the object: the same URL, the same
        // two key shapes, genuinely different bytes.
        let subset = DiskCache::cache_key_ranges(url, &[(0, 32767)]);
        let whole = DiskCache::cache_key(url, None);
        let head = body(3)[..32768].to_vec();
        let all = body(3);

        cache.put(&subset, &head);
        cache.put(&whole, &all);

        assert_eq!(cache.get(&subset), Some(head.clone()));
        assert_eq!(cache.get(&whole), Some(all.clone()));
        let payloads = stored_payloads(&dir);
        assert_eq!(payloads.len(), 2, "two contents, two payloads: {:?}", payloads);
        let stored: u64 = payloads.iter().map(|(_, len)| len).sum();
        assert_eq!(stored, (head.len() + all.len()) as u64);
        assert_eq!(cache.dedup().reference_entries, 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The portable form, exercised: pointer entries round-trip.
    #[test]
    fn test_reference_entries_round_trip_without_hard_links() {
        let dir = scratch("pointer");
        let cache = DiskCache::with_dir(dir.clone()).without_hard_links();
        let url = "https://example.com/pointer.grib2";
        let whole = DiskCache::cache_key(url, None);
        let ranges = DiskCache::cache_key_ranges(url, &[(0, 65535)]);
        let data = body(4);
        cache.put(&ranges, &data);
        cache.put(&whole, &data);

        assert_eq!(cache.get(&whole), Some(data.clone()));
        assert_eq!(cache.get(&ranges), Some(data.clone()));
        assert_eq!(stored_payloads(&dir).len(), 1);
        // Two files per entry either way: the pointer stands where the
        // payload would.
        let names: Vec<_> = std::fs::read_dir(cache.cache_path(&whole).parent().unwrap())
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(names.len(), 2, "pointer + sidecar only: {:?}", names);
        assert!(names.iter().any(|name| name.ends_with(".ref")));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A reference that has lost its payload is a miss, and is set aside.
    #[test]
    fn test_a_dangling_reference_is_a_miss_and_is_set_aside() {
        let dir = scratch("dangling");
        let cache = DiskCache::with_dir(dir.clone()).without_hard_links();
        let key = DiskCache::cache_key("https://example.com/gone.grib2", None);
        let data = body(5);
        cache.put(&key, &data);
        assert_eq!(cache.get(&key), Some(data.clone()));

        // The canonical payload disappears under the reference.
        let blob = cache.blob_path(&content_id(&data));
        std::fs::remove_file(&blob).unwrap();

        assert!(!cache.has(&key), "a reference to nothing cannot vouch");
        assert_eq!(cache.get(&key), None, "a dangling reference must not hit");
        let aside: Vec<_> = std::fs::read_dir(cache.cache_path(&key).parent().unwrap())
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(aside.len(), 2, "pointer and sidecar both kept: {:?}", aside);
        assert!(aside.iter().all(|name| name.contains("rejected-")));

        // The next put repopulates cleanly.
        cache.put(&key, &data);
        assert_eq!(cache.get(&key), Some(data));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A reference pointed at the wrong content is a miss, never a hit.
    #[test]
    fn test_a_reference_to_foreign_content_is_a_miss() {
        let dir = scratch("crossed");
        let cache = DiskCache::with_dir(dir.clone()).without_hard_links();
        let mine = DiskCache::cache_key("https://example.com/mine.grib2", None);
        let theirs = DiskCache::cache_key("https://example.com/theirs.grib2", None);
        let (my_data, their_data) = (body(6), body(7));
        cache.put(&mine, &my_data);
        cache.put(&theirs, &their_data);

        // Repoint one entry's reference at the other object's payload.
        std::fs::write(
            cache.ref_path(&mine),
            format!(
                "canonical={}\n",
                cache.relative_to_root(&cache.blob_path(&content_id(&their_data)))
            ),
        )
        .unwrap();

        assert_eq!(cache.get(&mine), None, "foreign content must not be served");
        // The other key is untouched, and its payload was sound, so nothing
        // took it away.
        assert_eq!(cache.get(&theirs), Some(their_data));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A poisoned canonical payload does not survive the entry that found it.
    #[test]
    fn test_a_corrupt_canonical_payload_is_set_aside_too() {
        let dir = scratch("poisoned");
        let cache = DiskCache::with_dir(dir.clone());
        let key = DiskCache::cache_key("https://example.com/poison.grib2", None);
        let data = body(8);
        cache.put(&key, &data);

        let blob = cache.blob_path(&content_id(&data));
        let mut mutated = data.clone();
        mutated[0] ^= 0xff;
        std::fs::write(&blob, &mutated).unwrap();

        assert_eq!(cache.get(&key), None);
        assert!(!blob.exists(), "the poisoned copy must not stay canonical");
        // Nothing deleted: it is aside, in the content store, with its bytes.
        let kept: u64 = every_file(&dir.join(CONTENT_DIR))
            .iter()
            .filter(|(name, _)| name.contains("rejected-"))
            .map(|(_, len)| len)
            .sum();
        assert_eq!(kept, data.len() as u64);

        // And the refetch is not linked back onto the poison.
        cache.put(&key, &data);
        assert_eq!(cache.get(&key), Some(data));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `remove` used to delete the payload and leave the sidecar describing it.
    #[test]
    fn test_remove_leaves_no_orphaned_sidecar() {
        let dir = scratch("removed");
        let cache = DiskCache::with_dir(dir.clone());
        let key = DiskCache::cache_key("https://example.com/f.grib2", None);
        cache.put(&key, b"payload");
        cache.remove(&key);

        let names: Vec<_> = std::fs::read_dir(cache.cache_path(&key).parent().unwrap())
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .collect();
        assert!(names.is_empty(), "no orphan left behind: {:?}", names);
        assert!(!cache.has(&key));
        assert_eq!(cache.get(&key), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The same, for an entry stored in the portable pointer form.
    #[test]
    fn test_remove_takes_the_reference_form_too() {
        let dir = scratch("removedref");
        let cache = DiskCache::with_dir(dir.clone()).without_hard_links();
        let key = DiskCache::cache_key("https://example.com/g.grib2", None);
        cache.put(&key, b"payload");
        cache.remove(&key);

        let names: Vec<_> = std::fs::read_dir(cache.cache_path(&key).parent().unwrap())
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .collect();
        assert!(names.is_empty(), "no orphan left behind: {:?}", names);
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
