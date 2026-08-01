use rustwx_core::{ModelRunRequest, SourceId};
use rustwx_io::{CachedFetchResult, FetchRequest};
use serde::Serialize;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub const RUN_PUBLICATION_SCHEMA_VERSION: u32 = 4;

/// Allocate a 16-hex-digit attempt id that mixes the current process id,
/// a monotonic counter, and the current wall-clock ns. Collisions across
/// concurrent invocations of different binaries would require them to
/// pick the same pid, the same counter value, and the same time —
/// extremely unlikely in practice.
pub fn new_attempt_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let pid = std::process::id() as u64;
    let now_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_nanos() as u64)
        .unwrap_or(0);
    let counter = COUNTER.fetch_add(1, Ordering::SeqCst);
    // FNV-1a 64-bit mix of (pid, now_ns, counter) → 16 hex digits.
    let mut hash: u64 = 0xcbf29ce484222325;
    for value in [pid, now_ns, counter] {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    format!("{hash:016x}")
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct ArtifactContentIdentity {
    pub bytes_len: usize,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct PublishedFetchIdentity {
    pub fetch_key: String,
    pub planned_family: String,
    /// Logical planned families (as requested by recipe fetch plans)
    /// that share this canonical fetch. Typically a single-element list
    /// equal to `planned_family`, but for HRRR the same wrfsfc fetch
    /// services both "nat"-planned native-family recipes and
    /// "sfc"-planned surface recipes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub planned_family_aliases: Vec<String>,
    pub request: ModelRunRequest,
    pub source_override: Option<SourceId>,
    pub resolved_source: SourceId,
    pub resolved_url: String,
    pub resolved_family: String,
    pub bytes_len: usize,
    pub bytes_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunPublicationState {
    Planned,
    Running,
    Complete,
    Partial,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactPublicationState {
    Planned,
    Running,
    Complete,
    Failed,
    Blocked,
    CacheHit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct PublishedArtifactRecord {
    pub artifact_key: String,
    pub relative_path: PathBuf,
    pub state: ArtifactPublicationState,
    pub detail: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_identity: Option<ArtifactContentIdentity>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_fetch_keys: Vec<String>,
}

impl PublishedArtifactRecord {
    pub fn planned<K: Into<String>, P: Into<PathBuf>>(artifact_key: K, relative_path: P) -> Self {
        Self {
            artifact_key: artifact_key.into(),
            relative_path: relative_path.into(),
            state: ArtifactPublicationState::Planned,
            detail: None,
            content_identity: None,
            input_fetch_keys: Vec::new(),
        }
    }

    pub fn with_state(mut self, state: ArtifactPublicationState) -> Self {
        self.state = state;
        self
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    pub fn with_content_identity(mut self, identity: ArtifactContentIdentity) -> Self {
        self.content_identity = Some(identity);
        self
    }

    pub fn with_input_fetch_keys(mut self, keys: Vec<String>) -> Self {
        self.input_fetch_keys = keys;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct RunPublicationManifest {
    pub schema_version: u32,
    pub run_kind: String,
    pub run_label: String,
    pub output_root: PathBuf,
    /// Unique 16-hex-digit identifier per invocation. Lets reruns coexist
    /// in the filesystem without silently clobbering each other when paired
    /// with [`publish_run_manifest_with_attempt`], and lets downstream
    /// systems cite "this specific attempt" rather than "the latest one".
    #[serde(default = "default_attempt_id")]
    pub attempt_id: String,
    /// Wall-clock start of this attempt. Distinct from [`started_unix_ms`]
    /// for older schema readers — schema v3 had only a single start time
    /// and no attempt identity, so `attempt_started_unix_ms` defaults to
    /// matching `started_unix_ms` during upgrade.
    #[serde(default)]
    pub attempt_started_unix_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub date_yyyymmdd: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cycle_utc: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forecast_hour: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain_slug: Option<String>,
    pub state: RunPublicationState,
    pub started_unix_ms: u128,
    pub finished_unix_ms: Option<u128>,
    pub detail: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_fetches: Vec<PublishedFetchIdentity>,
    pub artifacts: Vec<PublishedArtifactRecord>,
}

fn default_attempt_id() -> String {
    new_attempt_id()
}

impl RunPublicationManifest {
    pub fn new(
        run_kind: impl Into<String>,
        run_label: impl Into<String>,
        output_root: impl Into<PathBuf>,
    ) -> Self {
        let now = unix_time_ms();
        Self {
            schema_version: RUN_PUBLICATION_SCHEMA_VERSION,
            run_kind: run_kind.into(),
            run_label: run_label.into(),
            output_root: output_root.into(),
            attempt_id: new_attempt_id(),
            attempt_started_unix_ms: now,
            model: None,
            date_yyyymmdd: None,
            cycle_utc: None,
            forecast_hour: None,
            source: None,
            domain_slug: None,
            state: RunPublicationState::Planned,
            started_unix_ms: now,
            finished_unix_ms: None,
            detail: None,
            input_fetches: Vec::new(),
            artifacts: Vec::new(),
        }
    }

    /// Choose a run state based on the current artifact state mix:
    ///
    /// - all `Complete` or `CacheHit` → [`RunPublicationState::Complete`]
    /// - any `Failed`                 → [`RunPublicationState::Failed`]
    /// - any `Blocked` (but no failures) → [`RunPublicationState::Partial`]
    /// - otherwise                    → leaves state unchanged
    ///
    /// Keeps the single source of truth on artifact state so each runner
    /// doesn't have to hand-roll its own completeness calculation.
    pub fn finalize_from_artifact_states(&mut self) {
        let mut has_failure = false;
        let mut has_blocked = false;
        let mut has_planned_or_running = false;
        for artifact in &self.artifacts {
            match artifact.state {
                ArtifactPublicationState::Complete | ArtifactPublicationState::CacheHit => {}
                ArtifactPublicationState::Failed => has_failure = true,
                ArtifactPublicationState::Blocked => has_blocked = true,
                ArtifactPublicationState::Planned | ArtifactPublicationState::Running => {
                    has_planned_or_running = true;
                }
            }
        }
        if has_failure {
            let failed_keys = self
                .artifacts
                .iter()
                .filter(|artifact| artifact.state == ArtifactPublicationState::Failed)
                .map(|artifact| artifact.artifact_key.clone())
                .collect::<Vec<_>>();
            self.mark_failed(format!("failed artifact(s): {}", failed_keys.join(", ")));
        } else if has_blocked {
            let blocked = self
                .artifacts
                .iter()
                .filter(|artifact| artifact.state == ArtifactPublicationState::Blocked)
                .count();
            self.mark_partial(format!("{blocked} artifact(s) blocked"));
        } else if !has_planned_or_running {
            self.mark_complete();
        }
    }

    pub fn with_input_fetches(mut self, input_fetches: Vec<PublishedFetchIdentity>) -> Self {
        self.input_fetches = input_fetches;
        self
    }

    pub fn with_run_metadata(
        mut self,
        model: impl Into<String>,
        date_yyyymmdd: impl Into<String>,
        cycle_utc: u8,
        forecast_hour: u16,
        source: impl Into<String>,
        domain_slug: impl Into<String>,
    ) -> Self {
        self.model = Some(model.into());
        self.date_yyyymmdd = Some(date_yyyymmdd.into());
        self.cycle_utc = Some(cycle_utc);
        self.forecast_hour = Some(forecast_hour);
        self.source = Some(source.into());
        self.domain_slug = Some(domain_slug.into());
        self
    }

    pub fn with_artifacts(mut self, artifacts: Vec<PublishedArtifactRecord>) -> Self {
        self.artifacts = artifacts;
        self
    }

    pub fn push_input_fetch(&mut self, input_fetch: PublishedFetchIdentity) {
        self.input_fetches.push(input_fetch);
    }

    pub fn push_artifact(&mut self, artifact: PublishedArtifactRecord) {
        self.artifacts.push(artifact);
    }

    pub fn mark_running(&mut self) {
        self.state = RunPublicationState::Running;
        self.finished_unix_ms = None;
        self.detail = None;
    }

    pub fn mark_complete(&mut self) {
        self.state = RunPublicationState::Complete;
        self.finished_unix_ms = Some(unix_time_ms());
        self.detail = None;
    }

    pub fn mark_partial(&mut self, detail: impl Into<String>) {
        self.state = RunPublicationState::Partial;
        self.finished_unix_ms = Some(unix_time_ms());
        self.detail = Some(detail.into());
    }

    pub fn mark_failed(&mut self, detail: impl Into<String>) {
        self.state = RunPublicationState::Failed;
        self.finished_unix_ms = Some(unix_time_ms());
        self.detail = Some(detail.into());
    }

    pub fn update_artifact_state(
        &mut self,
        artifact_key: &str,
        state: ArtifactPublicationState,
        detail: Option<String>,
    ) -> bool {
        if let Some(artifact) = self
            .artifacts
            .iter_mut()
            .find(|artifact| artifact.artifact_key == artifact_key)
        {
            artifact.state = state;
            artifact.detail = detail;
            return true;
        }
        false
    }

    pub fn update_artifact_identity(
        &mut self,
        artifact_key: &str,
        identity: ArtifactContentIdentity,
    ) -> bool {
        if let Some(artifact) = self
            .artifacts
            .iter_mut()
            .find(|artifact| artifact.artifact_key == artifact_key)
        {
            artifact.content_identity = Some(identity);
            return true;
        }
        false
    }

    pub fn update_artifact_input_fetch_keys(
        &mut self,
        artifact_key: &str,
        input_fetch_keys: Vec<String>,
    ) -> bool {
        if let Some(artifact) = self
            .artifacts
            .iter_mut()
            .find(|artifact| artifact.artifact_key == artifact_key)
        {
            artifact.input_fetch_keys = input_fetch_keys;
            return true;
        }
        false
    }
}

pub fn default_run_manifest_path(output_root: &Path, run_slug: &str) -> PathBuf {
    output_root.join(format!("{run_slug}_run_manifest.json"))
}

/// Where the immutable attempt-specific sibling manifest lives.
///
/// The canonical path returned by [`default_run_manifest_path`] is
/// overwritten on every rerun and therefore always reflects the latest
/// attempt. This attempt path — `<slug>_run_manifest.<attempt_id>.json`
/// — is written exactly once per attempt, so audits can quote a
/// specific run that will not silently change later. Paired with the
/// `attempt_id` stored inside the manifest body they form the
/// immutable-attempt contract.
pub fn attempt_run_manifest_path(output_root: &Path, run_slug: &str, attempt_id: &str) -> PathBuf {
    output_root.join(format!("{run_slug}_run_manifest.{attempt_id}.json"))
}

pub fn publish_run_manifest(
    path: &Path,
    manifest: &RunPublicationManifest,
) -> Result<(), Box<dyn std::error::Error>> {
    atomic_write_json(path, manifest)
}

/// Publish the canonical (overwritable) run manifest and, alongside it,
/// an attempt-stamped sibling. Returns `(canonical_path, attempt_path)`.
///
/// The canonical path is the value passed via `canonical_path`; the
/// attempt path is derived from `run_slug` + `manifest.attempt_id`. A
/// run that only calls [`publish_run_manifest`] still gets a readable
/// manifest, but reruns clobber it in place — this helper is the
/// supported way to keep per-attempt immutable records.
pub fn publish_run_manifest_with_attempt(
    canonical_path: &Path,
    output_root: &Path,
    run_slug: &str,
    manifest: &RunPublicationManifest,
) -> Result<(PathBuf, PathBuf), Box<dyn std::error::Error>> {
    let attempt_path = attempt_run_manifest_path(output_root, run_slug, &manifest.attempt_id);
    atomic_write_json(canonical_path, manifest)?;
    atomic_write_json(&attempt_path, manifest)?;
    Ok((canonical_path.to_path_buf(), attempt_path))
}

/// Publish a skeleton failure manifest so any operational runner that
/// errors before completing its work still leaves an auditable record
/// on disk. Sets `state = failed` with the supplied `detail`, and
/// writes both the canonical and the attempt-stamped sibling paths.
///
/// Returns the `(canonical_path, attempt_path)` pair so the caller can
/// surface them in stderr or a post-mortem log. Intended to be invoked
/// from a runner's top-level error branch:
///
/// ```no_run
/// # use rustwx_products::publication::{publish_failure_manifest, RunPublicationManifest};
/// # use std::path::Path;
/// # fn work() -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
/// # let out_dir = Path::new("/tmp/rustwx");
/// # let slug = "rustwx_demo";
/// if let Err(err) = work() {
///     let _ = publish_failure_manifest(
///         "demo_batch",
///         slug,
///         out_dir,
///         slug,
///         err.to_string(),
///     );
///     return Err(err);
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn publish_failure_manifest(
    run_kind: &str,
    run_label: &str,
    output_root: &Path,
    run_slug: &str,
    detail: impl Into<String>,
) -> Result<(PathBuf, PathBuf), Box<dyn std::error::Error>> {
    fs::create_dir_all(output_root)?;
    let mut manifest = RunPublicationManifest::new(run_kind, run_label, output_root);
    manifest.mark_failed(detail);
    let canonical = default_run_manifest_path(output_root, run_slug);
    publish_run_manifest_with_attempt(&canonical, output_root, run_slug, &manifest)
}

/// Build the canonical run-slug the operational CLI bins use for their
/// manifest file names. Centralized so failure-path publication can
/// produce a slug that matches the success path byte-for-byte given the
/// same inputs.
pub fn canonical_run_slug(
    model_slug: &str,
    date_yyyymmdd: &str,
    cycle_hint: Option<u8>,
    forecast_hour: u16,
    region_slug: &str,
    kind_suffix: &str,
) -> String {
    let cycle_label = cycle_hint
        .map(|hour| format!("{hour:02}"))
        .unwrap_or_else(|| "XX".to_string());
    format!(
        "rustwx_{model_slug}_{date_yyyymmdd}_{cycle_label}z_f{forecast_hour:03}_{region_slug}_{kind_suffix}"
    )
}

/// One-call finalize-and-publish wiring for runners.
///
/// Derives run state from artifact state via
/// [`RunPublicationManifest::finalize_from_artifact_states`],
/// and publishes both the canonical and attempt-stamped manifest files
/// via [`publish_run_manifest_with_attempt`]. Returns the two published
/// paths. Every operational CLI bin in the workspace funnels through
/// this helper so the publication contract is the single source of
/// truth regardless of which binary ran.
pub fn finalize_and_publish_run_manifest(
    manifest: &mut RunPublicationManifest,
    output_root: &Path,
    run_slug: &str,
) -> Result<(PathBuf, PathBuf), Box<dyn std::error::Error>> {
    manifest.finalize_from_artifact_states();
    let canonical = default_run_manifest_path(output_root, run_slug);
    publish_run_manifest_with_attempt(&canonical, output_root, run_slug, manifest)
}

pub fn atomic_write_json<T: Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = serde_json::to_vec_pretty(value)?;
    atomic_write_bytes(path, &bytes)
}

/// Move `tmp` onto `dst` in one step, replacing whatever `dst` held.
///
/// One call is enough on both families this workspace runs on. POSIX
/// `rename(2)` is specified to replace an existing destination atomically;
/// Rust's Windows `fs::rename` calls `MoveFileExW` with
/// `MOVEFILE_REPLACE_EXISTING`, which does the same. The "rename does not
/// overwrite on Windows" folklore is about `MoveFileW` without that flag.
///
/// `atomic_replace_is_a_single_step_on_this_platform` asserts that against
/// the real filesystem rather than trusting the paragraph above, because
/// the whole no-loss-window claim rests on it and it is a platform fact.
///
/// A failed replace leaves both paths as they were: `dst` keeps its
/// previous content and `tmp` is still there for the caller to clean up.
/// That is the contract any substitute passed to
/// [`atomic_write_bytes_with_replacer`] must honour.
pub fn replace_file(tmp: &Path, dst: &Path) -> Result<(), std::io::Error> {
    fs::rename(tmp, dst)
}

pub fn atomic_write_bytes(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    atomic_write_bytes_with_replacer(path, bytes, replace_file)
}

/// [`atomic_write_bytes`] with the final replace step supplied by the
/// caller, so the crash window can be *tested* rather than reasoned about.
///
/// Production has exactly one implementation and it is [`replace_file`].
pub fn atomic_write_bytes_with_replacer<R>(
    path: &Path,
    bytes: &[u8],
    replace: R,
) -> Result<(), Box<dyn std::error::Error>>
where
    R: FnOnce(&Path, &Path) -> Result<(), std::io::Error>,
{
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = temp_path_for(path);
    let write_result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp_path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }
    // One step, and nothing before it touches `path`. There is no moment at
    // which the destination is absent: it holds the old bytes until the
    // replace returns and the new bytes afterwards.
    if let Err(err) = replace(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(Box::new(err));
    }
    Ok(())
}

pub fn artifact_identity_from_bytes(bytes: &[u8]) -> ArtifactContentIdentity {
    ArtifactContentIdentity {
        bytes_len: bytes.len(),
        sha256: sha256_hex(bytes),
    }
}

pub fn artifact_identity_from_path(
    path: &Path,
) -> Result<ArtifactContentIdentity, Box<dyn std::error::Error>> {
    Ok(artifact_identity_from_bytes(&fs::read(path)?))
}

pub fn fetch_key(planned_family: &str, request: &ModelRunRequest) -> String {
    format!(
        "{}:{}:{:02}z:f{:03}:{}->{}",
        request.model.as_str(),
        request.cycle.date_yyyymmdd,
        request.cycle.hour_utc,
        request.forecast_hour,
        planned_family,
        request.product
    )
}

pub fn fetch_identity_from_cached_result(
    planned_family: &str,
    fetch: &FetchRequest,
    fetched: &CachedFetchResult,
) -> PublishedFetchIdentity {
    fetch_identity_from_cached_result_with_aliases(planned_family, Vec::new(), fetch, fetched)
}

pub fn fetch_identity_from_cached_result_with_aliases(
    planned_family: &str,
    mut planned_family_aliases: Vec<String>,
    fetch: &FetchRequest,
    fetched: &CachedFetchResult,
) -> PublishedFetchIdentity {
    // Preserve only the logical aliases that actually differ from the
    // canonical family — otherwise the aliases list is redundant and we
    // keep the JSON skipped via `skip_serializing_if`.
    planned_family_aliases.retain(|alias| alias != planned_family);
    planned_family_aliases.sort();
    planned_family_aliases.dedup();
    let (bytes_len, bytes_sha256) = fetch_payload_identity(fetched);
    PublishedFetchIdentity {
        fetch_key: fetch_key(planned_family, &fetch.request),
        planned_family: planned_family.to_string(),
        planned_family_aliases,
        request: fetch.request.clone(),
        source_override: fetch.source_override,
        resolved_source: fetched.result.source,
        resolved_url: fetched.result.url.clone(),
        resolved_family: fetch.request.product.clone(),
        bytes_len,
        bytes_sha256,
    }
}

fn fetch_payload_identity(fetched: &CachedFetchResult) -> (usize, String) {
    if !fetched.result.bytes.is_empty() {
        return (
            fetched.result.bytes.len(),
            sha256_hex(&fetched.result.bytes),
        );
    }
    if fetched.bytes_path.is_file() {
        let len = fs::metadata(&fetched.bytes_path)
            .ok()
            .and_then(|metadata| usize::try_from(metadata.len()).ok())
            .unwrap_or(0);
        return (len, "path_backed_not_hashed".to_string());
    }
    (0, sha256_hex(&[]))
}

fn temp_path_for(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    path.with_file_name(format!(
        ".{file_name}.tmp-{}-{}",
        process::id(),
        unix_time_ms()
    ))
}

fn unix_time_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut message = bytes.to_vec();
    let bit_len = (message.len() as u64) * 8;
    message.push(0x80);
    while (message.len() % 64) != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    let mut h0 = 0x6a09e667u32;
    let mut h1 = 0xbb67ae85u32;
    let mut h2 = 0x3c6ef372u32;
    let mut h3 = 0xa54ff53au32;
    let mut h4 = 0x510e527fu32;
    let mut h5 = 0x9b05688cu32;
    let mut h6 = 0x1f83d9abu32;
    let mut h7 = 0x5be0cd19u32;

    for chunk in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).take(16).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h0;
        let mut b = h1;
        let mut c = h2;
        let mut d = h3;
        let mut e = h4;
        let mut f = h5;
        let mut g = h6;
        let mut h = h7;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
        h5 = h5.wrapping_add(f);
        h6 = h6.wrapping_add(g);
        h7 = h7.wrapping_add(h);
    }

    format!("{h0:08x}{h1:08x}{h2:08x}{h3:08x}{h4:08x}{h5:08x}{h6:08x}{h7:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
    struct JsonFixture {
        name: String,
        value: u16,
    }

    #[test]
    fn atomic_write_json_publishes_readable_file() {
        let root =
            std::env::temp_dir().join(format!("rustwx_products_publication_{}", process::id()));
        let path = root.join("fixture.json");
        let fixture = JsonFixture {
            name: "demo".into(),
            value: 7,
        };

        atomic_write_json(&path, &fixture).unwrap();
        let loaded: JsonFixture = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(loaded, fixture);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn atomic_write_json_replaces_existing_file_contents() {
        let root = std::env::temp_dir().join(format!(
            "rustwx_products_publication_replace_{}",
            process::id()
        ));
        let path = root.join("fixture.json");
        let first = JsonFixture {
            name: "first".into(),
            value: 1,
        };
        let second = JsonFixture {
            name: "second".into(),
            value: 2,
        };

        atomic_write_json(&path, &first).unwrap();
        atomic_write_json(&path, &second).unwrap();
        let loaded: JsonFixture = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(loaded, second);

        let _ = fs::remove_dir_all(root);
    }

    /// The platform fact the whole guarantee rests on, asserted against the
    /// real filesystem: a rename onto an existing file succeeds and the
    /// destination ends up holding exactly the source's bytes.
    #[test]
    fn atomic_replace_is_a_single_step_on_this_platform() {
        let root = std::env::temp_dir().join(format!(
            "rustwx_products_publication_platform_{}",
            process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let dst = root.join("manifest.json");
        let tmp = root.join("incoming");
        fs::write(&dst, b"previous good manifest").unwrap();
        fs::write(&tmp, b"the new one").unwrap();

        replace_file(&tmp, &dst).unwrap();

        assert_eq!(fs::read(&dst).unwrap(), b"the new one");
        assert!(!tmp.exists(), "the source name must be gone after a replace");
        let _ = fs::remove_dir_all(root);
    }

    /// The retired code was `remove_file(dst)` then `rename(tmp, dst)`. A
    /// crash between them left no manifest at all -- and every operational
    /// bin in this workspace publishes its run manifest through here.
    ///
    /// The stub honours `replace_file`'s contract in the failing direction:
    /// it moves nothing, which is what a failed rename/MoveFileEx does.
    #[test]
    fn a_failure_at_the_replace_leaves_the_previous_manifest_intact() {
        let root = std::env::temp_dir().join(format!(
            "rustwx_products_publication_crash_{}",
            process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("manifest.json");
        fs::write(&path, b"previous good manifest").unwrap();

        let err = atomic_write_bytes_with_replacer(&path, b"the new one", |tmp, dst| {
            // The instant the retired remove/rename gap sat at: the payload
            // is written and fsynced, the destination is untouched, and the
            // process dies here.
            assert!(tmp.is_file(), "the staged payload must exist");
            assert!(
                dst.is_file(),
                "the destination must still exist when the replace is entered"
            );
            Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "injected crash at the replace",
            ))
        })
        .unwrap_err();
        assert!(err.to_string().contains("injected crash"), "{err}");

        assert_eq!(
            fs::read(&path).unwrap(),
            b"previous good manifest",
            "a failed replace must leave the previous manifest byte-identical"
        );
        let leftovers: Vec<String> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp"))
            .collect();
        assert_eq!(leftovers, Vec::<String>::new());
        let _ = fs::remove_dir_all(root);
    }

    /// The positive half, and the one that would have caught the retired
    /// code: an observer around the real replace records what the
    /// destination held going in and coming out. Both are present.
    #[test]
    fn the_manifest_is_never_absent_across_a_successful_write() {
        let root = std::env::temp_dir().join(format!(
            "rustwx_products_publication_present_{}",
            process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("manifest.json");
        fs::write(&path, b"previous good manifest").unwrap();
        let mut seen: Vec<Option<Vec<u8>>> = Vec::new();

        atomic_write_bytes_with_replacer(&path, b"the new one", |tmp, dst| {
            seen.push(fs::read(dst).ok());
            let outcome = replace_file(tmp, dst);
            seen.push(fs::read(dst).ok());
            outcome
        })
        .unwrap();

        assert_eq!(
            seen,
            vec![
                Some(b"previous good manifest".to_vec()),
                Some(b"the new one".to_vec()),
            ],
            "the destination must read as the old manifest before the replace and \
             the new one after it, and never as absent"
        );
        assert_eq!(fs::read(&path).unwrap(), b"the new one");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn manifest_tracks_artifact_and_run_states() {
        let mut manifest = RunPublicationManifest::new(
            "hrrr_direct_batch",
            "hrrr_20260414_23z_f006_conus",
            PathBuf::from("proof/demo"),
        )
        .with_input_fetches(vec![PublishedFetchIdentity {
            fetch_key: "prs".into(),
            planned_family: "prs".into(),
            planned_family_aliases: Vec::new(),
            request: ModelRunRequest::new(
                rustwx_core::ModelId::Hrrr,
                rustwx_core::CycleSpec::new("20260414", 23).unwrap(),
                6,
                "prs",
            )
            .unwrap(),
            source_override: Some(SourceId::Nomads),
            resolved_source: SourceId::Nomads,
            resolved_url: "https://example.test/hrrr.t23z.wrfprsf06.grib2".into(),
            resolved_family: "prs".into(),
            bytes_len: 1234,
            bytes_sha256: sha256_hex(b"fetch-bytes"),
        }])
        .with_artifacts(vec![
            PublishedArtifactRecord::planned("sbcape", "sbcape.png")
                .with_input_fetch_keys(vec!["prs".into()]),
            PublishedArtifactRecord::planned("mlcape", "mlcape.png"),
        ]);

        manifest.mark_running();
        assert_eq!(manifest.state, RunPublicationState::Running);
        assert!(manifest.update_artifact_state("sbcape", ArtifactPublicationState::Complete, None));
        assert!(
            manifest.update_artifact_identity("sbcape", artifact_identity_from_bytes(b"png-bytes"))
        );
        assert!(manifest.update_artifact_state(
            "mlcape",
            ArtifactPublicationState::Blocked,
            Some("blocked in test".into())
        ));
        manifest.mark_partial("one artifact blocked");

        assert_eq!(manifest.state, RunPublicationState::Partial);
        assert!(manifest.finished_unix_ms.is_some());
        assert_eq!(
            manifest
                .artifacts
                .iter()
                .find(|artifact| artifact.artifact_key == "sbcape")
                .unwrap()
                .state,
            ArtifactPublicationState::Complete
        );
        assert_eq!(manifest.input_fetches.len(), 1);
        assert_eq!(
            manifest
                .artifacts
                .iter()
                .find(|artifact| artifact.artifact_key == "sbcape")
                .unwrap()
                .content_identity
                .as_ref()
                .unwrap()
                .sha256,
            sha256_hex(b"png-bytes")
        );
        assert_eq!(
            manifest
                .artifacts
                .iter()
                .find(|artifact| artifact.artifact_key == "mlcape")
                .unwrap()
                .state,
            ArtifactPublicationState::Blocked
        );
    }

    #[test]
    fn finalize_from_artifact_states_picks_complete_partial_or_failed() {
        let mut complete = RunPublicationManifest::new("x", "y", PathBuf::from("z"))
            .with_artifacts(vec![
                PublishedArtifactRecord::planned("a", "a.png")
                    .with_state(ArtifactPublicationState::Complete),
                PublishedArtifactRecord::planned("b", "b.png")
                    .with_state(ArtifactPublicationState::CacheHit),
            ]);
        complete.finalize_from_artifact_states();
        assert_eq!(complete.state, RunPublicationState::Complete);

        let mut partial =
            RunPublicationManifest::new("x", "y", PathBuf::from("z")).with_artifacts(vec![
                PublishedArtifactRecord::planned("a", "a.png")
                    .with_state(ArtifactPublicationState::Complete),
                PublishedArtifactRecord::planned("b", "b.png")
                    .with_state(ArtifactPublicationState::Blocked)
                    .with_detail("missing input"),
            ]);
        partial.finalize_from_artifact_states();
        assert_eq!(partial.state, RunPublicationState::Partial);
        assert_eq!(partial.detail.as_deref(), Some("1 artifact(s) blocked"));

        let mut failed =
            RunPublicationManifest::new("x", "y", PathBuf::from("z")).with_artifacts(vec![
                PublishedArtifactRecord::planned("a", "a.png")
                    .with_state(ArtifactPublicationState::Failed)
                    .with_detail("boom"),
                PublishedArtifactRecord::planned("b", "b.png")
                    .with_state(ArtifactPublicationState::Blocked),
            ]);
        failed.finalize_from_artifact_states();
        assert_eq!(failed.state, RunPublicationState::Failed);
        assert!(failed.detail.as_deref().unwrap().contains("a"));
    }

    #[test]
    fn publish_with_attempt_writes_both_canonical_and_attempt_paths() {
        let root = std::env::temp_dir().join(format!("rustwx_pub_attempt_{}", process::id()));
        fs::create_dir_all(&root).unwrap();
        let slug = "demo_run";
        let canonical = default_run_manifest_path(&root, slug);
        let manifest = RunPublicationManifest::new("demo_batch", slug, root.clone());

        let (canonical_path, attempt_path) =
            publish_run_manifest_with_attempt(&canonical, &root, slug, &manifest).unwrap();
        assert_eq!(canonical_path, canonical);
        assert!(canonical.exists());
        assert!(attempt_path.exists());
        assert!(
            attempt_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap()
                .contains(&manifest.attempt_id)
        );

        // Rerunning with a fresh attempt id leaves the old attempt file
        // untouched; the canonical file is overwritten in place.
        let second = RunPublicationManifest::new("demo_batch", slug, root.clone());
        assert_ne!(second.attempt_id, manifest.attempt_id);
        let (_, second_attempt) =
            publish_run_manifest_with_attempt(&canonical, &root, slug, &second).unwrap();
        assert!(attempt_path.exists());
        assert!(second_attempt.exists());
        assert_ne!(attempt_path, second_attempt);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn attempt_id_is_stable_within_a_single_manifest() {
        let manifest = RunPublicationManifest::new("demo", "label", PathBuf::from("/tmp"));
        let serialized = serde_json::to_string(&manifest).unwrap();
        let round_tripped: RunPublicationManifest = serde_json::from_str(&serialized).unwrap();
        assert_eq!(manifest.attempt_id, round_tripped.attempt_id);
        assert_eq!(round_tripped.schema_version, RUN_PUBLICATION_SCHEMA_VERSION);
    }

    #[test]
    fn publish_failure_manifest_writes_canonical_and_attempt_paths() {
        let root = std::env::temp_dir().join(format!("rustwx_pub_failure_{}", process::id()));
        let _ = fs::remove_dir_all(&root);
        let slug = "rustwx_demo_failure";
        let (canonical, attempt) =
            publish_failure_manifest("demo_batch", slug, &root, slug, "simulated upstream outage")
                .unwrap();
        assert!(canonical.exists());
        assert!(attempt.exists());
        let parsed: RunPublicationManifest =
            serde_json::from_slice(&fs::read(&canonical).unwrap()).unwrap();
        assert_eq!(parsed.state, RunPublicationState::Failed);
        assert_eq!(parsed.detail.as_deref(), Some("simulated upstream outage"));
        assert!(
            attempt
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap()
                .contains(&parsed.attempt_id)
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn canonical_run_slug_matches_success_path_layout() {
        assert_eq!(
            canonical_run_slug("hrrr", "20260414", Some(23), 6, "midwest", "direct"),
            "rustwx_hrrr_20260414_23z_f006_midwest_direct"
        );
        assert_eq!(
            canonical_run_slug("gfs", "20260414", None, 12, "conus", "derived"),
            "rustwx_gfs_20260414_XXz_f012_conus_derived"
        );
    }

    #[test]
    fn artifact_identity_helper_hashes_bytes_and_files() {
        let root = std::env::temp_dir().join(format!(
            "rustwx_products_publication_hash_{}",
            process::id()
        ));
        let path = root.join("artifact.bin");
        fs::create_dir_all(&root).unwrap();

        atomic_write_bytes(&path, b"abc").unwrap();
        let from_bytes = artifact_identity_from_bytes(b"abc");
        let from_path = artifact_identity_from_path(&path).unwrap();
        assert_eq!(from_bytes, from_path);
        assert_eq!(
            from_bytes.sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        let _ = fs::remove_dir_all(root);
    }
}
