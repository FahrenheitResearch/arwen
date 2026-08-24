//! Host-memory accounting and admission for `rw_mpas_static`.
//!
//! This module intentionally measures *host* memory.  CUDA free-memory
//! counters are not accepted as a substitute.  Linux uses procfs and cgroup
//! v1/v2 data when available; other platforms still support an explicit limit
//! and report unavailable process counters rather than inventing numbers.

#[cfg(target_os = "linux")]
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{MpasError, MpasResult};

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct HostMemorySnapshot {
    pub unix_ms: u128,
    pub rss_bytes: Option<u64>,
    pub private_bytes: Option<u64>,
    pub committed_bytes: Option<u64>,
    pub cgroup_current_bytes: Option<u64>,
    pub cgroup_limit_bytes: Option<u64>,
    pub source: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BufferLedger {
    pub name: String,
    pub bytes: u64,
    pub lifetime: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MemoryEvent {
    pub phase: String,
    pub dataset: Option<String>,
    pub event: String,
    pub snapshot: HostMemorySnapshot,
    pub source_bytes: Option<u64>,
    pub destination_bytes: Option<u64>,
    pub workspace_bytes: Option<u64>,
    pub buffers: Vec<BufferLedger>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct MemoryReceipt {
    pub schema: String,
    pub configured_limit_bytes: Option<u64>,
    pub predicted_peak_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub peak_private_bytes: Option<u64>,
    pub peak_committed_bytes: Option<u64>,
    pub events: Vec<MemoryEvent>,
}

impl MemoryReceipt {
    pub fn new(configured_limit_bytes: Option<u64>, predicted_peak_bytes: Option<u64>) -> Self {
        Self {
            schema: "rw-mpas-static.host-memory/v1".to_string(),
            configured_limit_bytes,
            predicted_peak_bytes,
            ..Self::default()
        }
    }

    pub fn event(
        &mut self,
        phase: impl Into<String>,
        dataset: Option<&Path>,
        event: impl Into<String>,
        source_bytes: Option<u64>,
        destination_bytes: Option<u64>,
        workspace_bytes: Option<u64>,
        buffers: Vec<BufferLedger>,
    ) {
        let snapshot = capture_host_memory();
        self.peak_rss_bytes = max_opt(self.peak_rss_bytes, snapshot.rss_bytes);
        self.peak_private_bytes = max_opt(self.peak_private_bytes, snapshot.private_bytes);
        self.peak_committed_bytes = max_opt(self.peak_committed_bytes, snapshot.committed_bytes);
        self.events.push(MemoryEvent {
            phase: phase.into(),
            dataset: dataset.map(|p| p.display().to_string()),
            event: event.into(),
            snapshot,
            source_bytes,
            destination_bytes,
            workspace_bytes,
            buffers,
        });
    }
}

fn max_opt(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[cfg(target_os = "linux")]
fn parse_kib_line(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let line = line.trim();
        let rest = line.strip_prefix(key)?;
        let mut it = rest.split_whitespace();
        let value = it.next()?.parse::<u64>().ok()?;
        let unit = it.next().unwrap_or("kB");
        match unit {
            "kB" | "KiB" => value.checked_mul(1024),
            "MB" | "MiB" => value.checked_mul(1024 * 1024),
            _ => None,
        }
    })
}

#[cfg(target_os = "linux")]
fn read_u64_file(path: &str) -> Option<u64> {
    let s = fs::read_to_string(path).ok()?;
    let s = s.trim();
    if s == "max" {
        return None;
    }
    s.parse().ok()
}

#[cfg(target_os = "linux")]
fn cgroup_values() -> (Option<u64>, Option<u64>) {
    // cgroup v2
    if Path::new("/sys/fs/cgroup/memory.current").exists() {
        return (
            read_u64_file("/sys/fs/cgroup/memory.current"),
            read_u64_file("/sys/fs/cgroup/memory.max"),
        );
    }
    // cgroup v1
    (
        read_u64_file("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        read_u64_file("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
}

#[cfg(target_os = "linux")]
pub fn capture_host_memory() -> HostMemorySnapshot {
    let mut out = HostMemorySnapshot {
        unix_ms: now_ms(),
        source: "linux:/proc/self/status+/proc/self/smaps_rollup+cgroup".to_string(),
        ..HostMemorySnapshot::default()
    };

    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        out.rss_bytes = parse_kib_line(&status, "VmRSS:");
        // VmSize is the closest procfs "commit-like" process value available
        // without platform-specific allocator hooks.  It is intentionally
        // labelled committed_bytes only as a process virtual commitment proxy.
        out.committed_bytes = parse_kib_line(&status, "VmSize:");
    }
    if let Ok(smaps) = fs::read_to_string("/proc/self/smaps_rollup") {
        let private_clean = parse_kib_line(&smaps, "Private_Clean:").unwrap_or(0);
        let private_dirty = parse_kib_line(&smaps, "Private_Dirty:").unwrap_or(0);
        let private_huge = parse_kib_line(&smaps, "Private_Hugetlb:").unwrap_or(0);
        out.private_bytes = private_clean
            .checked_add(private_dirty)
            .and_then(|v| v.checked_add(private_huge));
        if out.rss_bytes.is_none() {
            out.rss_bytes = parse_kib_line(&smaps, "Rss:");
        }
    }
    let (current, limit) = cgroup_values();
    out.cgroup_current_bytes = current;
    out.cgroup_limit_bytes = limit;
    out
}

#[cfg(not(target_os = "linux"))]
pub fn capture_host_memory() -> HostMemorySnapshot {
    HostMemorySnapshot {
        unix_ms: now_ms(),
        source: format!(
            "{}: process RSS/private counters unavailable in this dependency-free build",
            std::env::consts::OS
        ),
        ..HostMemorySnapshot::default()
    }
}

pub fn resolve_host_limit(explicit: Option<u64>) -> Option<u64> {
    if explicit.is_some() {
        return explicit;
    }
    if let Ok(raw) = std::env::var("RW_MPAS_HOST_MEMORY_LIMIT_BYTES") {
        if let Ok(v) = raw.parse::<u64>() {
            if v > 0 {
                return Some(v);
            }
        }
    }
    if let Ok(raw) = std::env::var("RW_MPAS_HOST_MEMORY_LIMIT_GIB") {
        if let Ok(v) = raw.parse::<f64>() {
            if v.is_finite() && v > 0.0 {
                return Some((v * 1024.0_f64.powi(3)) as u64);
            }
        }
    }
    let snap = capture_host_memory();
    snap.cgroup_limit_bytes
}

/// Bytes of a host limit the builder is allowed to plan against.
///
/// Keep 10% or 512 MiB, whichever is larger, for allocator/runtime/OS
/// overhead not represented by the explicit scientific buffers.
pub fn usable_under_limit(limit: u64) -> u64 {
    limit.saturating_sub(reserve_under_limit(limit))
}

/// The overhead reserve [`usable_under_limit`] withholds.
pub fn reserve_under_limit(limit: u64) -> u64 {
    (limit / 10).max(512 * 1024 * 1024)
}

/// Conservative preflight for the bounded static builder.
///
/// The predicted peak is intentionally the sum of the permanent mesh/operator
/// resident set plus every *concurrent* one-tile/one-plane geography scratch,
/// the tile-map cache budget and one NetCDF encode slab.  It does *not* add
/// the sizes of every geography source.
pub fn enforce_admission(
    phase: &str,
    predicted_peak_bytes: u64,
    explicit_limit: Option<u64>,
) -> MpasResult<Option<u64>> {
    let limit = resolve_host_limit(explicit_limit);
    if let Some(limit) = limit {
        let reserve = reserve_under_limit(limit);
        let usable = usable_under_limit(limit);
        if predicted_peak_bytes > usable {
            return Err(MpasError::Refusal(format!(
                "host memory admission refused in phase '{phase}': predicted peak \
                 {predicted_peak_bytes} bytes exceeds usable {usable} bytes under \
                 configured/cgroup limit {limit} bytes (reserve {reserve} bytes). \
                 Remedy: lower source supersampling/tile size, use a larger host-memory \
                 limit, or run on a host with more RAM; device VRAM does not satisfy \
                 this host-memory requirement."
            )));
        }
    }
    Ok(limit)
}

/// How many tiles may be in flight at once, and what that costs.
///
/// The serial builder held exactly one tile's scratch at a time, so its
/// predicted peak carried one `per_worker_tile_bytes` term.  Running the tile
/// loop on N workers holds N of them at once, and the reusable tile-map cache
/// is a further resident term.  Both are represented here so the admission
/// gate keeps meaning what it says under parallelism.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TileParallelPlan {
    /// Workers the caller asked for (CPU count, env or flag).
    pub requested_workers: usize,
    /// Workers the host-memory budget actually admits.  Never zero.
    pub workers: usize,
    /// Scratch one tile worker holds: encoded plane + decoded plane(s) +
    /// destination map + the map's cache encoding transient.
    pub per_worker_tile_bytes: u64,
    /// Mesh, topology and operator arrays held for the whole build.
    pub resident_bytes: u64,
    /// The largest single non-tile transient (the NetCDF encode slab).
    pub one_shot_bytes: u64,
    /// Allocator/runtime headroom the model does not itemize.
    pub slack_bytes: u64,
    /// Bytes the reusable tile-map cache may hold.  Zero disables it.
    pub tile_map_cache_bytes: u64,
    /// What the caller would have liked the cache to hold.
    pub tile_map_cache_request_bytes: u64,
    pub predicted_peak_bytes: u64,
    pub limit_bytes: Option<u64>,
}

/// Size the tile loop to the admitted host-memory budget.
///
/// Bounding workers by CPU count alone is what would break: on a
/// memory-limited host N cores x one decoded 30-arcsec tile plane each is a
/// multi-gigabyte step the serial model never predicted, so the build would
/// be killed mid-tile by the OS with no refusal naming a cause.  Workers are
/// therefore cut to what the budget admits, and only the leftover funds the
/// tile-map cache.
#[allow(clippy::too_many_arguments)]
pub fn plan_tile_parallelism(
    phase: &str,
    resident_bytes: u64,
    per_worker_tile_bytes: u64,
    one_shot_bytes: u64,
    slack_bytes: u64,
    requested_workers: usize,
    tile_map_cache_request_bytes: u64,
    explicit_limit: Option<u64>,
) -> MpasResult<TileParallelPlan> {
    let requested_workers = requested_workers.max(1);
    let limit = resolve_host_limit(explicit_limit);
    let fixed = resident_bytes
        .saturating_add(one_shot_bytes)
        .saturating_add(slack_bytes);

    let (workers, cache) = match limit {
        None => (requested_workers, tile_map_cache_request_bytes),
        Some(limit) => {
            let usable = usable_under_limit(limit);
            if fixed.saturating_add(per_worker_tile_bytes) > usable {
                return Err(MpasError::Refusal(format!(
                    "host memory admission refused in phase '{phase}': a single tile \
                     worker needs {per_worker_tile_bytes} bytes of source-plane and \
                     destination-map scratch on top of {fixed} bytes of resident mesh, \
                     operator and writer state, which exceeds the usable {usable} bytes \
                     under configured/cgroup limit {limit} bytes. Even a one-worker \
                     build would be killed while decoding a single WPS_GEOG tile plane. \
                     Remedy: lower the supersample factors, use a coarser geography \
                     source, raise the host-memory limit, or run on a host with more \
                     RAM; device VRAM does not satisfy this host-memory requirement."
                )));
            }
            let room = usable - fixed;
            let affordable = if per_worker_tile_bytes == 0 {
                requested_workers
            } else {
                (room / per_worker_tile_bytes) as usize
            };
            let workers = requested_workers.min(affordable).max(1);
            let after_workers =
                room.saturating_sub((workers as u64).saturating_mul(per_worker_tile_bytes));
            (workers, tile_map_cache_request_bytes.min(after_workers))
        }
    };

    let predicted_peak_bytes = fixed
        .saturating_add((workers as u64).saturating_mul(per_worker_tile_bytes))
        .saturating_add(cache);
    // Belt and braces: the plan is only worth having if it passes the same
    // gate every other phase is measured against.
    enforce_admission(phase, predicted_peak_bytes, explicit_limit)?;

    Ok(TileParallelPlan {
        requested_workers,
        workers,
        per_worker_tile_bytes,
        resident_bytes,
        one_shot_bytes,
        slack_bytes,
        tile_map_cache_bytes: cache,
        tile_map_cache_request_bytes,
        predicted_peak_bytes,
        limit_bytes: limit,
    })
}

/// [`checked_vec`] for an element that is neither [`Default`] nor [`Clone`].
///
/// The parallel accumulators are atomics, which are neither, and reserving
/// them through the same refusal path is what keeps a failed reservation
/// naming its phase and buffer instead of panicking in the allocator.
pub fn checked_vec_with<T>(
    len: usize,
    phase: &str,
    name: &str,
    mut make: impl FnMut() -> T,
) -> MpasResult<Vec<T>> {
    let elem = std::mem::size_of::<T>();
    let bytes = len.checked_mul(elem).ok_or_else(|| {
        MpasError::Refusal(format!(
            "host memory size overflow in phase '{phase}' for buffer '{name}': \
             {len} elements x {elem} bytes"
        ))
    })?;
    let mut v = Vec::new();
    v.try_reserve_exact(len).map_err(|e| {
        let snap = capture_host_memory();
        MpasError::Refusal(format!(
            "host memory allocation failed in phase '{phase}' for buffer '{name}': \
             requested {bytes} bytes ({len} elements x {elem}); current RSS={:?}, \
             private={:?}, commit_proxy={:?}, cgroup_current={:?}, cgroup_limit={:?}; \
             allocator error: {e}. Remedy: reduce tile/supersample workspace, lower \
             --tile-workers, or raise the host-memory limit. This is host RAM, not \
             GPU VRAM.",
            snap.rss_bytes,
            snap.private_bytes,
            snap.committed_bytes,
            snap.cgroup_current_bytes,
            snap.cgroup_limit_bytes,
        ))
    })?;
    for _ in 0..len {
        v.push(make());
    }
    Ok(v)
}

/// `a * b` as an element count, refusing instead of wrapping.
///
/// Destination accumulators are sized `n_cells * n_categories` or
/// `n_cells * n_planes`. On a large mesh with a wide categorical source that
/// product is where an overflow would first appear, so it is checked here
/// rather than trusted into [`checked_vec`].
pub fn checked_len(a: usize, b: usize, phase: &str, name: &str) -> MpasResult<usize> {
    a.checked_mul(b).ok_or_else(|| {
        MpasError::Refusal(format!(
            "host memory element-count overflow in phase '{phase}' for buffer \
             '{name}': {a} x {b} does not fit a usize"
        ))
    })
}

/// [`checked_vec`] filled with `value` instead of `T::default()`.
pub fn checked_vec_filled<T: Default + Clone>(
    len: usize,
    value: T,
    phase: &str,
    name: &str,
) -> MpasResult<Vec<T>> {
    let mut v = checked_vec::<T>(len, phase, name)?;
    for slot in v.iter_mut() {
        *slot = value.clone();
    }
    Ok(v)
}

pub fn checked_vec<T: Default + Clone>(
    len: usize,
    phase: &str,
    name: &str,
) -> MpasResult<Vec<T>> {
    let elem = std::mem::size_of::<T>();
    let bytes = len.checked_mul(elem).ok_or_else(|| {
        MpasError::Refusal(format!(
            "host memory size overflow in phase '{phase}' for buffer '{name}': \
             {len} elements x {elem} bytes"
        ))
    })?;
    let mut v = Vec::new();
    v.try_reserve_exact(len).map_err(|e| {
        let snap = capture_host_memory();
        MpasError::Refusal(format!(
            "host memory allocation failed in phase '{phase}' for buffer '{name}': \
             requested {bytes} bytes ({len} elements x {elem}); current RSS={:?}, \
             private={:?}, commit_proxy={:?}, cgroup_current={:?}, cgroup_limit={:?}; \
             allocator error: {e}. Remedy: reduce tile/supersample workspace or raise \
             the host-memory limit. This is host RAM, not GPU VRAM.",
            snap.rss_bytes,
            snap.private_bytes,
            snap.committed_bytes,
            snap.cgroup_current_bytes,
            snap.cgroup_limit_bytes,
        ))
    })?;
    v.resize(len, T::default());
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn an_impossible_reservation_refuses_with_the_buffer_and_the_remedy() {
        // The defect this replaces was a bare allocation panic quoting only a
        // byte count.  The refusal has to carry the phase, the buffer, the
        // request, live counters and a remedy, and it has to say host RAM.
        let err = checked_vec::<u8>(usize::MAX, "geog-read-plane", "encoded tile plane")
            .expect_err("an impossible reservation cannot succeed");
        let text = err.to_string();
        for needle in [
            "host memory allocation failed",
            "phase 'geog-read-plane'",
            "buffer 'encoded tile plane'",
            "RSS=",
            "cgroup_limit=",
            "Remedy:",
            "host RAM, not GPU VRAM",
        ] {
            assert!(text.contains(needle), "missing {needle:?} in: {text}");
        }
    }

    #[test]
    fn an_element_size_overflow_refuses_before_it_reserves() {
        let err = checked_vec::<u64>(usize::MAX, "operators", "deriv_two")
            .expect_err("usize::MAX u64 elements cannot be sized");
        let text = err.to_string();
        assert!(text.contains("host memory size overflow"), "{text}");
        assert!(text.contains("buffer 'deriv_two'"), "{text}");
    }

    #[test]
    fn checked_len_refuses_an_overflowing_product() {
        let err = checked_len(usize::MAX, 2, "categorical", "category counts")
            .expect_err("the product overflows");
        let text = err.to_string();
        assert!(text.contains("element-count overflow"), "{text}");
        assert!(text.contains("buffer 'category counts'"), "{text}");
    }

    #[test]
    fn checked_len_passes_a_product_that_fits() {
        assert_eq!(
            checked_len(2_000, 24, "categorical", "category counts").expect("fits"),
            48_000
        );
    }

    #[test]
    fn a_small_buffer_allocates_and_is_zeroed() {
        let v = checked_vec::<i64>(8, "geog-decode-plane", "decoded tile plane").expect("fits");
        assert_eq!(v, vec![0i64; 8]);
    }

    #[test]
    fn a_filled_buffer_carries_the_value_not_the_default() {
        let v = checked_vec_filled::<f64>(4, f64::NAN, "full-plane", "climatology plane")
            .expect("fits");
        assert_eq!(v.len(), 4);
        assert!(v.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn admission_refuses_over_the_limit_and_names_host_ram() {
        let err = enforce_admission("preflight", 20 * GIB, Some(10 * GIB))
            .expect_err("20 GiB does not fit under a 10 GiB limit");
        let text = err.to_string();
        for needle in [
            "host memory admission refused",
            "phase 'preflight'",
            "Remedy:",
            "device VRAM does not satisfy this host-memory requirement",
        ] {
            assert!(text.contains(needle), "missing {needle:?} in: {text}");
        }
        // The reserve is 10% or 512 MiB, whichever is larger: 1 GiB here.
        assert!(text.contains(&format!("{}", GIB)), "reserve not stated: {text}");
    }

    #[test]
    fn admission_admits_under_the_limit_and_returns_it() {
        let limit = enforce_admission("preflight", 1024, Some(10 * GIB))
            .expect("a kilobyte fits under 10 GiB");
        assert_eq!(limit, Some(10 * GIB));
    }

    #[test]
    fn admission_refuses_at_the_reserve_boundary_and_admits_just_below_it() {
        // A 10 GiB limit reserves 1 GiB, leaving 9 GiB usable.  One byte
        // either side of that edge must fall on opposite sides of the gate.
        let usable = 9 * GIB;
        enforce_admission("preflight", usable, Some(10 * GIB)).expect("exactly usable is admitted");
        enforce_admission("preflight", usable + 1, Some(10 * GIB))
            .expect_err("one byte past usable is refused");
    }

    #[test]
    fn the_small_limit_reserve_floor_is_512_mib() {
        // Below 5 GiB the 10% rule would reserve less than 512 MiB, so the
        // floor takes over: a 1 GiB limit leaves 512 MiB usable, not 900 MiB.
        let half_gib = 512 * 1024 * 1024;
        enforce_admission("preflight", half_gib, Some(GIB)).expect("512 MiB fits");
        enforce_admission("preflight", half_gib + 1, Some(GIB))
            .expect_err("past the 512 MiB floor is refused");
    }

    #[test]
    fn an_explicit_limit_wins_over_every_other_source() {
        // The environment and the cgroup are only consulted when the caller
        // supplied nothing; an explicit limit is never widened by them.
        assert_eq!(resolve_host_limit(Some(7 * GIB)), Some(7 * GIB));
        assert_eq!(resolve_host_limit(Some(1)), Some(1));
    }

    #[test]
    fn a_snapshot_reports_its_source_rather_than_inventing_counters() {
        let snap = capture_host_memory();
        assert!(!snap.source.is_empty());
        assert!(snap.unix_ms > 0);
        if cfg!(target_os = "linux") {
            assert!(snap.source.contains("linux:"), "{}", snap.source);
        } else {
            assert!(snap.rss_bytes.is_none(), "non-Linux must not invent an RSS");
            assert!(
                snap.source.contains("unavailable"),
                "the absence must be stated: {}",
                snap.source
            );
        }
    }

    // -- parallel tile admission -------------------------------------------

    const MIB: u64 = 1024 * 1024;

    fn plan(workers: usize, per_worker: u64, cache: u64, limit: Option<u64>) -> TileParallelPlan {
        plan_tile_parallelism(
            "preflight",
            256 * MIB, // resident
            per_worker,
            64 * MIB, // one-shot writer slab
            GIB,      // slack
            workers,
            cache,
            limit,
        )
        .expect("plan fits")
    }

    #[test]
    fn the_predicted_peak_counts_every_concurrent_tile_not_just_one() {
        // The defect this guards: the serial model carried ONE tile-scratch
        // term.  Running the tile loop on N workers holds N of them, so a
        // model that did not scale the term would admit a build the OS then
        // kills mid-tile.
        let one = plan(1, 512 * MIB, 0, Some(64 * GIB));
        let eight = plan(8, 512 * MIB, 0, Some(64 * GIB));
        assert_eq!(one.workers, 1);
        assert_eq!(eight.workers, 8);
        assert_eq!(
            eight.predicted_peak_bytes - one.predicted_peak_bytes,
            7 * 512 * MIB,
            "the extra seven workers must appear in the peak"
        );
    }

    #[test]
    fn workers_are_cut_to_the_budget_not_to_the_cpu_count() {
        // 8 GiB limit reserves 819 MiB, leaving 7.2 GiB usable.  Fixed cost
        // is 256 MiB + 64 MiB + 1 GiB = 1.31 GiB, so ~5.9 GiB funds workers
        // at 1 GiB each: five, however many cores the host advertises.
        let p = plan(32, GIB, 0, Some(8 * GIB));
        assert!(p.workers < 32, "a 32-core host must not get 32 workers here");
        assert_eq!(p.workers, 5, "workers={} peak={}", p.workers, p.predicted_peak_bytes);
        assert_eq!(p.requested_workers, 32);
        assert!(p.predicted_peak_bytes <= usable_under_limit(8 * GIB));
    }

    #[test]
    fn a_tile_that_cannot_fit_even_once_refuses_and_names_the_breakage() {
        let err = plan_tile_parallelism(
            "preflight",
            256 * MIB,
            32 * GIB, // one tile worker
            64 * MIB,
            GIB,
            8,
            0,
            Some(8 * GIB),
        )
        .expect_err("32 GiB of tile scratch cannot fit under 8 GiB");
        let text = err.to_string();
        for needle in [
            "host memory admission refused",
            "phase 'preflight'",
            "a single tile worker",
            "killed while decoding a single WPS_GEOG tile plane",
            "Remedy:",
            "device VRAM does not satisfy this host-memory requirement",
        ] {
            assert!(text.contains(needle), "missing {needle:?} in: {text}");
        }
    }

    #[test]
    fn the_cache_is_funded_only_from_what_the_workers_leave() {
        // Workers come first: the cache is a speed-up, a worker is throughput.
        let p = plan(4, GIB, 64 * GIB, Some(16 * GIB));
        let usable = usable_under_limit(16 * GIB);
        let fixed = 256 * MIB + 64 * MIB + GIB;
        assert_eq!(p.workers, 4);
        assert_eq!(p.tile_map_cache_bytes, usable - fixed - 4 * GIB);
        assert!(
            p.tile_map_cache_bytes < p.tile_map_cache_request_bytes,
            "a 64 GiB ask cannot be granted under a 16 GiB limit"
        );
        assert_eq!(p.predicted_peak_bytes, usable);
    }

    #[test]
    fn a_modest_cache_ask_is_granted_whole() {
        let p = plan(4, GIB, 256 * MIB, Some(16 * GIB));
        assert_eq!(p.tile_map_cache_bytes, 256 * MIB);
        assert_eq!(
            p.predicted_peak_bytes,
            256 * MIB + 64 * MIB + GIB + 4 * GIB + 256 * MIB
        );
    }

    #[test]
    fn with_no_discoverable_limit_the_plan_is_what_was_asked_for() {
        // resolve_host_limit consults an explicit limit, then the environment,
        // then the cgroup.  With none of those the builder must still run --
        // and must still publish a peak that includes every worker.
        let p = plan_tile_parallelism("preflight", 0, 100, 0, 0, 6, 55, None);
        let p = p.expect("no limit admits");
        assert_eq!(p.workers, 6);
        assert_eq!(p.tile_map_cache_bytes, 55);
        assert_eq!(p.predicted_peak_bytes, 6 * 100 + 55);
        assert_eq!(p.limit_bytes, None);
    }

    #[test]
    fn zero_workers_are_promoted_to_one_rather_than_stalling_the_build() {
        let p = plan_tile_parallelism("preflight", 0, 8, 0, 0, 0, 0, None).expect("admits");
        assert_eq!(p.workers, 1);
        assert_eq!(p.requested_workers, 1);
    }

    #[test]
    fn an_atomic_buffer_reserves_through_the_same_refusal_path() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let v = checked_vec_with(4, "categorical", "category counts", || AtomicU32::new(0))
            .expect("fits");
        assert_eq!(v.len(), 4);
        assert!(v.iter().all(|x| x.load(Ordering::Relaxed) == 0));
        let err = checked_vec_with(usize::MAX, "categorical", "category counts", || {
            AtomicU32::new(0)
        })
        .expect_err("an impossible reservation cannot succeed");
        let text = err.to_string();
        assert!(text.contains("host memory size overflow"), "{text}");
        assert!(text.contains("buffer 'category counts'"), "{text}");
    }

    #[test]
    fn a_receipt_tracks_the_running_peak_across_events() {
        let mut receipt = MemoryReceipt::new(Some(GIB), Some(1024));
        assert_eq!(receipt.schema, "rw-mpas-static.host-memory/v1");
        receipt.event("preflight", None, "start", None, Some(1), None, vec![]);
        receipt.event("write", None, "end", None, Some(2), None, vec![]);
        assert_eq!(receipt.events.len(), 2);
        assert_eq!(receipt.configured_limit_bytes, Some(GIB));
        assert_eq!(receipt.predicted_peak_bytes, Some(1024));
        // The peak is a max over snapshots, so it never decreases.
        if let (Some(a), Some(b)) = (
            receipt.events[0].snapshot.rss_bytes,
            receipt.peak_rss_bytes,
        ) {
            assert!(b >= a);
        }
    }
}
