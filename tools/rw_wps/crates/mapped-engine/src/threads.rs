//! The engine's own thread pool, and the one rule every parallel step
//! in this crate obeys.
//!
//! THE RULE: a parallel step decodes into PRE-ASSIGNED SLOTS and is
//! drained in document order.  Nothing in this crate may let a schedule
//! reach an output byte — not the order of a `Vec`, not which refusal is
//! reported, not the order of a progress line.  Every use here is an
//! `IndexedParallelIterator::collect`, whose element *i* comes from
//! input *i* no matter which thread produced it first, followed by a
//! sequential drain that returns the FIRST error in input order.  That
//! makes determinism a property of the data structure rather than a
//! property of the run, which is the only form of it that survives a
//! loaded box.
//!
//! The pool is the engine's own rather than rayon's global one: a
//! library that installs work on the global pool inherits whatever
//! thread count the host process chose, and this binary is also driven
//! as a subprocess by a Python front door that may already be running
//! several of them at once.
//!
//! THE WIDTH is every core the process may run on (`available_parallelism`
//! honours the affinity mask and a cgroup CPU quota), unless the caller
//! names a count.  It used to be capped at eight, from a sweep of ONE
//! valid time's message decode, which stops scaling past eight because it
//! is memory-bandwidth bound.  That cap left a whole-series compose on
//! about 2.3 cores of work on a 16- or 24-core box: most of a valid
//! time's wall is serial work AFTER its messages are unpacked (assembly,
//! the field digests, the frame write), and the only way to put more
//! cores on that is to have several valid times in flight at once
//! ([`lanes`]), each of which needs pool threads of its own.
//!
//! THE WIDTH IS ALSO PRICED: before a series' first valid time is
//! decoded, the pool is narrowed to the widest one on which that valid
//! time fits in memory ([`admit_width`]), because each thread doing a
//! field's work holds up to one more copy of its field.  The price is
//! made from the inventory pass, which has read every selected record's
//! grid, so the first decode already runs at the priced width.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use rayon::{ThreadPool, ThreadPoolBuilder};
use std::io::Write;

#[path = "../../../../preparation_resources.rs"]
mod resources;

/// Override for the worker count, for measurement and for a caller who
/// is running several engines at once and wants each one narrower.
pub const THREADS_ENV: &str = "GPUWM_MAPPED_ENGINE_THREADS";

/// The single-valid-time sweep the old cap of eight came from, kept
/// because it is still the right number for ONE valid time's message
/// decode (a 32-logical-CPU box, `examples/decode_timing.rs`, no
/// `--output`, warm cache):
///
///   threads      1       2       4       8      16      32
///   hrrr-prs  27.67   18.51   13.24   11.30   10.99   11.20  s
///   gdas      14.97    9.23    6.52    5.65    5.63       -   s
///
/// A whole-series compose no longer runs one valid time at a time, so
/// the pool is not capped at it: [`lanes`] decides how many valid times
/// share the pool, and the pool is as wide as the machine.
pub const SINGLE_TIME_KNEE: usize = 8;

/// The most workers this process may have: the override if it parses to
/// a positive number, else every core this process may run on.  The pool
/// runs on [`width`] of them, which memory may narrow.
pub fn threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        if let Some(declared) = std::env::var(THREADS_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
        {
            return declared.min(resources::available_cpus());
        }
        resources::available_cpus()
    })
}

/// Host memory this process may still fill, in bytes, or `None` when
/// the platform cannot say.
///
/// Linux: `MemAvailable` (which counts reclaimable page cache), lowered
/// to the headroom of every cgroup v2 limit the process runs under --
/// its own group's and each ancestor's -- because a limit is the number
/// the OOM killer applies, not the host's.  Windows: the physical memory
/// `GlobalMemoryStatusEx` reports available.
///
/// Named breakage the walk prevents: only `/sys/fs/cgroup/memory.max`
/// was read, which is the process's own limit only inside a container
/// with a cgroup namespace.  A job started under a limit on a shared
/// host (a systemd scope or service with `MemoryMax`, a batch
/// scheduler's job group) was priced on the whole host's free memory:
/// under a 30 GiB limit on a 123 GiB host with 91 GB free, the 18 h
/// `hrrr-prs` compose is sized for four valid times in flight, which
/// hold 56.7 GiB; priced on the limit, it runs one.
pub fn available_memory() -> Option<u64> {
    available_memory_impl()
}

#[cfg(target_os = "linux")]
fn available_memory_impl() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut available = meminfo.lines().find_map(|line| {
        let rest = line.strip_prefix("MemAvailable:")?;
        let kib = rest.trim().trim_end_matches("kB").trim().parse::<u64>().ok()?;
        Some(kib.saturating_mul(1024))
    })?;
    let own = std::fs::read_to_string("/proc/self/cgroup").ok();
    for group in cgroup_v2_ancestry(own.as_deref()) {
        let read = |name: &str| {
            std::fs::read_to_string(format!("/sys/fs/cgroup{group}/{name}"))
                .ok()
                .and_then(|text| text.trim().parse::<u64>().ok())
        };
        // "max" (no limit) does not parse, and a group without the
        // memory controller has neither file: both leave the host's.
        if let (Some(limit), Some(current)) = (read("memory.max"), read("memory.current")) {
            let inactive_file = std::fs::read_to_string(format!("/sys/fs/cgroup{group}/memory.stat"))
                .ok()
                .and_then(|text| memory_stat(&text, "inactive_file"))
                .unwrap_or(0);
            available = available.min(cgroup_headroom(limit, current, inactive_file));
        }
    }
    Some(available)
}

/// What a cgroup v2 group may still fill: its limit less what it holds
/// that the kernel cannot simply drop.  `memory.current` counts the page
/// cache of every file the group read, and the inactive part of that
/// cache (`inactive_file` in `memory.stat`) is reclaimed before the group
/// is killed, as `MemAvailable` counts the host's reclaimable cache.
///
/// Named breakage: read as `limit - current`, a preparation that fetched
/// or inventoried its objects inside its limit was priced as if their
/// cache were held (the 18 h `hrrr-prs` objects are 7.7 GB), so the pool
/// and the valid times in flight were sized on a fraction of what the
/// group could still fill.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn cgroup_headroom(limit: u64, current: u64, inactive_file: u64) -> u64 {
    limit.saturating_sub(current.saturating_sub(inactive_file))
}

/// One counter of a cgroup v2 `memory.stat` (`key value` per line).
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn memory_stat(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let (name, value) = line.split_once(' ')?;
        (name == key).then(|| value.trim().parse::<u64>().ok()).flatten()
    })
}

/// The cgroup v2 groups a process runs under, from its own to the root
/// of its view, as paths below `/sys/fs/cgroup` ("" is that root).
///
/// `proc_self_cgroup` is the text of `/proc/self/cgroup`; its v2 line is
/// `0::<path>`.  Without one (no file, or cgroup v1 only) only the root
/// of the view is named, which inside a container with a cgroup
/// namespace is the container's own group.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn cgroup_v2_ancestry(proc_self_cgroup: Option<&str>) -> Vec<String> {
    let own = proc_self_cgroup
        .and_then(|text| text.lines().find_map(|line| line.strip_prefix("0::")))
        .map(|path| path.trim().trim_end_matches('/'))
        .filter(|path| path.starts_with('/') && !path.contains(".."))
        .unwrap_or("");
    let mut groups = Vec::new();
    let mut path = own;
    loop {
        groups.push(path.to_owned());
        match path.rfind('/') {
            Some(cut) if !path.is_empty() => path = &path[..cut],
            _ => break,
        }
    }
    if groups.last().is_some_and(|last| !last.is_empty()) {
        groups.push(String::new());
    }
    groups
}

#[cfg(windows)]
fn available_memory_impl() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: the struct is plain data, zeroed, with its length set as the
    // API requires; the call only writes into it.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok == 0 { None } else { Some(status.ullAvailPhys) }
}

#[cfg(not(any(target_os = "linux", windows)))]
fn available_memory_impl() -> Option<u64> {
    None
}

/// The share of [`available_memory`] the valid times in flight may hold
/// between them.  The rest is the parent Python process, the page cache
/// the next reads come from, and anything else on the box.
pub const MEMORY_SHARE: f64 = 0.7;

/// Additional whole-child memory reserved by a concurrently running parent.
/// Unlike available host bytes, this is already a usable budget: no second
/// MEMORY_SHARE discount is applied. Retained primary and donor streams count
/// against the same cap, not a new cap for each stream.
pub const MEMORY_BUDGET_ENV: &str = "GPUWM_MAPPED_ENGINE_MEMORY_BUDGET_BYTES";
pub const MEMORY_BUDGET_SCHEMA: &str = "gpuwm-mapped-host-memory-budget-v1";
static UNPRICED_ACQUISITION: AtomicBool = AtomicBool::new(false);

fn acquisition_budget_check(codec: &str, cap: Option<u64>) -> crate::refusal::Result<()> {
    if cap.is_some() {
        return Err(crate::refusal::host_memory(format!(
            "the {codec} acquisition wrapper has no bounded expansion price; refusing the scoped child before decompression")));
    }
    Ok(())
}

pub fn admit_acquisition_codec(codec: &str) -> crate::refusal::Result<()> {
    // The ordinary acquisition path remains available. Its price cannot
    // qualify a future child reservation until codec expansion is bounded.
    UNPRICED_ACQUISITION.store(true, Ordering::SeqCst);
    acquisition_budget_check(codec, declared_memory_budget()?)
}

#[cfg(target_os = "linux")]
fn process_memory() -> Option<(u64, u64)> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kib = |name: &str| status.lines().find_map(|line| {
        line.strip_prefix(name)?.split_whitespace().next()?.parse::<u64>().ok()
    }).map(|n| n.saturating_mul(1024));
    let resident = kib("VmRSS:")?;
    // Swapped arrays still belong to this child and may become resident at
    // the next access. Do not turn paging into a fresh reservation.
    Some((resident, resident.saturating_add(kib("VmSwap:").unwrap_or(0))))
}

#[cfg(windows)]
fn process_memory() -> Option<(u64, u64)> {
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    let size = counters.cb;
    // SAFETY: the pseudo-handle names this process; counters is writable and
    // its API size is supplied. The call acquires no handle to close.
    if unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, size) } == 0 {
        return None;
    }
    let resident = counters.WorkingSetSize as u64;
    Some((resident, resident.max(counters.PagefileUsage as u64)))
}

#[cfg(not(any(target_os = "linux", windows)))]
fn process_memory() -> Option<(u64, u64)> { None }

fn declared_memory_budget() -> crate::refusal::Result<Option<u64>> {
    match std::env::var(MEMORY_BUDGET_ENV) {
        Err(std::env::VarError::NotPresent) => Ok(None),
        Ok(value) => value.trim().parse::<u64>().map(Some).map_err(|_| {
            crate::refusal::usage(format!("{MEMORY_BUDGET_ENV} must be a nonnegative integer number of usable bytes"))
        }),
        Err(_) => Err(crate::refusal::usage(format!("{MEMORY_BUDGET_ENV} is not Unicode"))),
    }
}

fn usable_budget_on(available: Option<u64>, declared: Option<u64>,
                    held: u64, process_bytes: Option<u64>) -> crate::refusal::Result<Option<u64>> {
    let host = available.map(|n| (n.saturating_add(held) as f64 * MEMORY_SHARE) as u64);
    let child = match declared {
        None => None,
        Some(cap) => {
            let current = process_bytes.ok_or_else(|| crate::refusal::host_memory(
                "the native child cannot measure its own memory to enforce the declared budget"))?;
            // held is this stream's first decoded frame, already included in
            // series_price. Other retained streams and native overhead remain
            // charged. Saturate the debit before subtracting it from the cap.
            Some(cap.saturating_sub(current.saturating_sub(held)))
        }
    };
    Ok(match (host, child) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    })
}

pub fn usable_budget(held: u64) -> crate::refusal::Result<Option<u64>> {
    usable_budget_on(available_memory(), declared_memory_budget()?, held,
                     process_memory().map(|(_, committed)| committed))
}

/// The bytes ONE source time on one worker is held to ([`require_minimum`]).
///
/// With no declared child budget this is the whole reading, available
/// memory plus what this stream already holds: [`MEMORY_SHARE`] is the
/// margin the pool's width and the lanes keep, not a minimum, so a valid
/// time bigger than the share still runs, alone and on one thread, and
/// only a valid time bigger than the memory there is gets refused.  A
/// declared child budget is a reservation the parent holds the rest of the
/// machine against, so there the minimum is held to the usable budget.
///
/// Named breakage: the minimum was checked against the share, so a 7 GiB
/// source time with 9.5 GiB available (share 6.65 GiB) was refused before
/// decode with `host_memory`, where 2.8.4 decoded it on one worker.  The
/// refusal named a 70% planning share, not memory the host did not have.
fn minimum_budget_on(available: Option<u64>, declared: Option<u64>,
                     held: u64, process_bytes: Option<u64>) -> crate::refusal::Result<Option<u64>> {
    match declared {
        Some(_) => usable_budget_on(available, declared, held, process_bytes),
        None => Ok(available.map(|bytes| bytes.saturating_add(held))),
    }
}

fn minimum_budget(held: u64) -> crate::refusal::Result<Option<u64>> {
    minimum_budget_on(available_memory(), declared_memory_budget()?, held,
                      process_memory().map(|(_, committed)| committed))
}

pub fn require_priced_format(format: &str) -> crate::refusal::Result<()> {
    if format != "grib2" {
        let _ = writeln!(std::io::stderr(), "GPUWM_PREP_THREADS {}", serde_json::json!({
            "stage": "mapped_decode_compose", "source_format": format,
            "memory_priced": false, "per_time_bytes": 0,
            "process_rss_bytes": process_memory().map(|(rss, _)| rss)
        }));
        if declared_memory_budget()?.is_some() {
            return Err(crate::refusal::host_memory(format!(
                "a scoped native memory budget cannot price the whole-object {format} decode; refusing before reading its payload")));
        }
    }
    Ok(())
}

fn require_minimum(per_time: u64, fields: &[u64], budget: Option<u64>) -> crate::refusal::Result<()> {
    if let Some(bytes) = budget.filter(|_| per_time > 0) {
        let minimum = series_price(1, 1, per_time, fields);
        if minimum > bytes {
            return Err(crate::refusal::host_memory(format!(
                "one source time on one worker needs {minimum} bytes, but only {bytes} usable bytes remain; refusing before decode")));
        }
    }
    Ok(())
}

/// How many valid times a whole-series writer keeps in flight at once,
/// on the pool as [`admit_width`] left it.
///
/// `per_time_bytes` is what ONE valid time holds from its decode to its
/// write and `field_bytes` its decoded fields, largest first -- both
/// measured from the first valid time, which is decoded before the
/// series is written -- and `held_bytes` what that first valid time
/// holds now, while it waits to be written: it is added back to the
/// memory reading, because it is one of the valid times priced.  The
/// count is the most valid times whose [`series_price`] fits in
/// [`MEMORY_SHARE`] of available memory, never more than the pool has
/// threads or the series has times ([`lanes_on`]).  A platform that
/// cannot report its memory keeps two in flight, the smallest overlap
/// that still hides one time's serial tail behind the next one's decode.
///
/// Named breakage the memory bound prevents: a 3 km CONUS pressure-level
/// valid time is about 7 GB of float64 before it is windowed, so sixteen
/// of them at once is more than a 128 GB box has.
pub fn lanes(per_time_bytes: u64, field_bytes: &[u64], held_bytes: u64, times: usize) -> crate::refusal::Result<usize> {
    release_freed_memory();
    let budget = usable_budget(held_bytes)?;
    require_minimum(per_time_bytes, field_bytes, minimum_budget(held_bytes)?)?;
    // Another stream may now be retained, or available memory may have
    // fallen since the first decode. Narrow before dispatch, not after OOM.
    narrow(width_for_budget(width(), per_time_bytes, field_bytes, budget));
    require_priced_pool()?;
    let fit = lanes_on_budget(width(), per_time_bytes, field_bytes, budget, times);
    let requested = std::env::var(LANES_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0);
    Ok(limit_requested_lanes(fit, requested))
}

fn limit_requested_lanes(fit: usize, requested: Option<usize>) -> usize {
    requested.map_or(fit, |count| count.min(fit)).max(1)
}

/// What `lanes` valid times in flight on a pool `width` threads wide hold
/// at most, in bytes.
///
/// - Each valid time between its decode and its write holds
///   `per_time_bytes` (`DecodeStream::per_time_bytes`).
/// - A pool of `width` threads does at most `width` fields' work at once
///   (a field's stack, unit conversion and transposition, then its
///   window and digest), each holding up to one more copy of its field:
///   the `width` largest of `field_bytes` (largest first).
/// - With two or more in flight, up to half as many prepared frames may
///   wait for an earlier one to be written (`frames::pipeline`), each
///   holding its published arrays.  They are priced at their decoded
///   size, the sum of `field_bytes`, which bounds them: a frame windowed
///   to its domain publishes less.  One valid time in flight is always
///   the one written next, so none waits.
///
/// Measured against this price, the engine's resident peak on the 18 h
/// `hrrr-prs` compose of a 12 km 54x54 domain (19 valid times,
/// `per_time_bytes` 13.1 GiB, 22 fields totalling 5.9 GiB, the largest
/// 0.55 GiB): one valid time on 4 threads 14.4 GiB, on 24 threads
/// 19.5 GiB; two on 24 threads 32.6 GiB; four on 24 threads 56.7 GiB.
pub fn series_price(lanes: usize, width: usize, per_time_bytes: u64, field_bytes: &[u64]) -> u64 {
    let lanes = lanes.max(1) as u64;
    let decoded = field_bytes.iter().copied().fold(0u64, u64::saturating_add);
    let waiting = if lanes >= 2 { lanes / 2 } else { 0 };
    let in_flight = field_bytes.iter().take(width.max(1)).copied().fold(0u64, u64::saturating_add);
    lanes
        .saturating_mul(per_time_bytes)
        .saturating_add(waiting.saturating_mul(decoded))
        .saturating_add(in_flight)
}

/// The widest pool, up to `threads`, on which ONE valid time's
/// [`series_price`] fits in [`MEMORY_SHARE`] of `memory`; never narrower
/// than one thread, so a valid time bigger than the share still runs,
/// alone and on one thread.  A valid time bigger than the memory there is
/// is refused by [`require_minimum`], not here.  An unknown price (0) or
/// an unknown memory keeps every thread.
pub fn width_for(threads: usize, per_time_bytes: u64, field_bytes: &[u64], memory: Option<u64>) -> usize {
    width_for_budget(threads, per_time_bytes, field_bytes,
                     memory.map(|n| (n as f64 * MEMORY_SHARE) as u64))
}

fn width_for_budget(threads: usize, per_time_bytes: u64, field_bytes: &[u64], budget: Option<u64>) -> usize {
    let threads = threads.max(1);
    let Some(share) = budget.filter(|_| per_time_bytes > 0) else { return threads; };
    let mut width = threads;
    while width > 1 && series_price(1, width, per_time_bytes, field_bytes) > share {
        width -= 1;
    }
    width
}

/// [`lanes`] on stated inputs, so the arithmetic is testable: the most
/// valid times, up to the pool's `width` and the series' `times`, whose
/// [`series_price`] on that pool fits in [`MEMORY_SHARE`] of `memory`.
/// An unknown price (0) or memory keeps the rule [`lanes_for`] states.
pub fn lanes_on(
    width: usize,
    per_time_bytes: u64,
    field_bytes: &[u64],
    memory: Option<u64>,
    times: usize,
) -> usize {
    lanes_on_budget(width, per_time_bytes, field_bytes,
                    memory.map(|n| (n as f64 * MEMORY_SHARE) as u64), times)
}

fn lanes_on_budget(width: usize, per_time_bytes: u64, field_bytes: &[u64], budget: Option<u64>, times: usize) -> usize {
    let width = width.max(1);
    let Some(share) = budget.filter(|_| per_time_bytes > 0) else {
        return width.min(if budget.is_some() { width } else { 2 }).min(times).max(1);
    };
    let mut lanes = width.min(times).max(1);
    while lanes > 1 && series_price(lanes, width, per_time_bytes, field_bytes) > share {
        lanes -= 1;
    }
    lanes
}

/// Narrow the pool, before a series' first valid time is decoded, to the
/// widest one on which that valid time fits in memory ([`width_for`] on
/// this process's [`threads`] and [`available_memory`]), and return the
/// width the pool now has.  `per_time_bytes` and `field_bytes` are the
/// first valid time's, priced from the inventory pass before any record
/// is unpacked.  The pool never widens again: a later, smaller price
/// leaves it as it is.
///
/// Named breakage: the 18 h `hrrr-prs` compose ran one valid time on all
/// 24 threads whatever the memory, which holds 19.5 GiB where 13.1 was
/// priced, so the preparation was killed (-9) on a 30 GB host twice and
/// can be on any box with under about 20 GiB free.  On 4 threads the
/// same valid time holds 14.4 GiB in the same wall time.  Narrowing the
/// pool only after the first valid time had been decoded was measured
/// and made it worse: that decode had already run at full width.
pub fn admit_width(per_time_bytes: u64, field_bytes: &[u64], whole_per_time_bytes: u64) -> crate::refusal::Result<usize> {
    release_freed_memory();
    let budget = usable_budget(0)?;
    require_minimum(per_time_bytes, field_bytes, minimum_budget(0)?)?;
    narrow(width_for_budget(threads(), per_time_bytes, field_bytes, budget));
    require_priced_pool()?;
    if let Some(share) = budget.filter(|_| per_time_bytes > 0) {
        let alone = series_price(1, 1, per_time_bytes, field_bytes);
        if alone > share {
            let _ = writeln!(std::io::stderr(), "preparation worker warning: one source time needs {alone} bytes, more than the {share} bytes its planning share of host memory allows; decoding it alone on one worker");
        }
    }
    let actual_pool = pool();
    let actual_width = actual_pool.as_ref().map_or_else(
        rayon::current_num_threads, |pool| pool.current_num_threads());
    if actual_width < threads() {
        let _ = writeln!(std::io::stderr(), "preparation worker warning: native decode requested {} workers, but its host-memory budget permits {actual_width}; using the priced pool", threads());
    }
    if actual_pool.is_none() {
        let _ = writeln!(std::io::stderr(), "preparation worker warning: dedicated decode pool unavailable; using the global Rayon pool with {actual_width} workers");
    }
    let _ = writeln!(std::io::stderr(), "GPUWM_PREP_THREADS {}", serde_json::json!({
        "stage": "mapped_decode_compose", "requested_workers": std::env::var(THREADS_ENV).ok(),
        "available_cpus": resources::available_cpus(), "effective_workers": actual_width,
        "memory_priced": !UNPRICED_ACQUISITION.load(Ordering::SeqCst), "per_time_bytes": per_time_bytes,
        "whole_per_time_bytes": whole_per_time_bytes,
        "process_rss_bytes": process_memory().map(|(rss, _)| rss),
        "usable_budget_bytes": budget,
        "declared_child_budget_bytes": declared_memory_budget()?,
        "pool": if actual_pool.is_some() { "dedicated-rayon" } else { "global-rayon-fallback" }
    }));
    Ok(width())
}

fn require_priced_pool() -> crate::refusal::Result<()> {
    if pool().is_none() && (declared_memory_budget()?.is_some() || rayon::current_num_threads() > width()) {
        return Err(crate::refusal::host_memory(
            "the dedicated native worker pool could not start; an unpriced global-pool fallback would exceed the scoped memory contract"));
    }
    Ok(())
}

/// Hand the heap pages the steps so far have freed back to the system
/// (glibc's `malloc_trim`, which walks every thread's arena), so the
/// reading [`admit_width`] takes and the decode after it start from what
/// the process still holds.
///
/// Named breakage: the inventory pass parses every input object on the
/// full-width pool, and the pages each worker freed stay in its own
/// arena.  The pool the decode then runs on is new, its threads allocate
/// in arenas of their own, so on the 18 h `hrrr-prs` compose narrowed to
/// one thread the engine had reached 17.5 GiB when it was killed, where
/// one valid time on four threads from the start holds 14.4.
fn release_freed_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        unsafe extern "C" {
            fn malloc_trim(pad: usize) -> std::ffi::c_int;
        }
        // SAFETY: malloc_trim takes no pointer and only returns free
        // pages of the allocator's own heaps to the kernel.
        unsafe {
            malloc_trim(0);
        }
    }
}

/// Override for [`lanes`], for measurement: the number of valid times a
/// whole-series writer requests in flight, bounded by memory admission.
pub const LANES_ENV: &str = "GPUWM_MAPPED_ENGINE_LANES";

/// The lane rule without a price: the pool width, lowered so
/// `per_time_bytes` valid times fit in [`MEMORY_SHARE`] of `memory`,
/// never more than the series has; two when the memory is unknown.
pub fn lanes_for(threads: usize, per_time_bytes: u64, memory: Option<u64>, times: usize) -> usize {
    let by_memory = match memory {
        Some(bytes) if per_time_bytes > 0 => {
            ((bytes as f64 * MEMORY_SHARE) / per_time_bytes as f64).floor() as usize
        }
        Some(_) => threads,
        None => 2,
    };
    threads.min(by_memory).min(times).max(1)
}

/// The engine's pool.  Falls back to running the closure on the calling
/// thread if a pool cannot be built: a box that refuses threads must
/// still decode, just serially.
pub fn install<T: Send>(work: impl FnOnce() -> T + Send) -> T {
    match pool() {
        Some(pool) => pool.install(work),
        None => work(),
    }
}

/// The width [`narrow`] lowered the pool to; 0 until it does.
static NARROWED: AtomicUsize = AtomicUsize::new(0);

/// The pool and the width it was built at, built on first use.
static POOL: Mutex<Option<(usize, Arc<ThreadPool>)>> = Mutex::new(None);

/// How many threads the pool runs: [`threads`], unless [`narrow`] has
/// lowered it.
pub fn width() -> usize {
    match NARROWED.load(Ordering::SeqCst) {
        0 => threads(),
        narrowed => narrowed,
    }
}

fn pool() -> Option<Arc<ThreadPool>> {
    let mut slot = POOL.lock().unwrap_or_else(PoisonError::into_inner);
    if slot.is_none() {
        let width = width();
        *slot = ThreadPoolBuilder::new()
            .num_threads(width)
            .thread_name(|index| format!("mapped-engine-{index}"))
            .build()
            .ok()
            .map(|pool| (width, Arc::new(pool)));
    }
    slot.as_ref().map(|(_, pool)| Arc::clone(pool))
}

/// Lower the pool to at most `width` threads, never below one and never
/// wider than it is.  A wider pool already built is dropped (its threads
/// exit once idle) and the next parallel step builds one at the new
/// width.  Called only between parallel steps, from the thread that
/// drives them, so no work is in flight on the pool it drops.
pub fn narrow(width: usize) {
    let width = width.max(1);
    let mut slot = POOL.lock().unwrap_or_else(PoisonError::into_inner);
    if width >= self::width() {
        return;
    }
    NARROWED.store(width, Ordering::SeqCst);
    if slot.as_ref().is_some_and(|(built, _)| *built > width) {
        *slot = None;
    }
}

/// Run `work` on the CALLING thread with a scope whose spawned jobs run
/// on the engine's pool, or `None` when no pool could be built (the
/// caller then does the same work serially).
///
/// The calling thread stays outside the pool on purpose: it is the one
/// that waits on the jobs, writes the frame stream in order and talks to
/// the parent process, and a pool thread that blocked on those would be
/// a worker taken away from the decode.
pub fn in_place_scope<'scope, R>(work: impl FnOnce(&rayon::Scope<'scope>) -> R) -> Option<R> {
    let pool = pool()?;
    Some(pool.in_place_scope(work))
}

/// Drain a parallel step's pre-assigned slots in document order,
/// returning the FIRST refusal exactly where the serial engine would
/// have raised it.
pub fn in_order<T>(slots: Vec<crate::refusal::Result<T>>) -> crate::refusal::Result<Vec<T>> {
    let mut drained = Vec::with_capacity(slots.len());
    for slot in slots {
        drained.push(slot?);
    }
    Ok(drained)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn the_pool_has_at_least_one_worker_and_is_as_wide_as_the_machine() {
        assert!(threads() >= 1);
        if std::env::var(THREADS_ENV).is_err() {
            let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
            assert_eq!(threads(), cores);
        }
    }

    #[test]
    fn lanes_follow_the_cores_until_memory_binds() {
        const GIB: u64 = 1 << 30;
        // Plenty of memory: one valid time per worker, never more than the series.
        assert_eq!(lanes_for(16, GIB, Some(512 * GIB), 49), 16);
        assert_eq!(lanes_for(16, GIB, Some(512 * GIB), 9), 9);
        // 7 GiB per time with 120 GiB available: 0.7 * 120 / 7 = 12.
        assert_eq!(lanes_for(16, 7 * GIB, Some(120 * GIB), 49), 12);
        // A time bigger than the share still runs, alone.
        assert_eq!(lanes_for(16, 100 * GIB, Some(8 * GIB), 49), 1);
        // Memory unknown: the smallest overlap.
        assert_eq!(lanes_for(16, 7 * GIB, None, 49), 2);
        // An explicit single worker is one lane whatever the memory.
        assert_eq!(lanes_for(1, GIB, Some(512 * GIB), 49), 1);
    }

    #[test]
    fn the_pool_is_priced_with_the_fields_its_threads_hold() {
        const MIB: u64 = 1 << 20;
        const GIB: u64 = 1 << 30;
        // Shaped on the 18 h hrrr-prs compose: about 13.1 GiB per valid
        // time, 22 fields of about 6 GiB, the largest about 0.55 GiB.
        let mut fields = vec![560 * MIB; 8];
        fields.extend(vec![120 * MIB; 14]);
        let per_time = 13_440 * MIB;
        // One valid time on every thread is priced with every field's
        // work at once; on one thread, with the largest field's alone.
        assert_eq!(series_price(1, 24, per_time, &fields), 19_600 * MIB);
        assert_eq!(series_price(1, 1, per_time, &fields), 14_000 * MIB);
        // Two in flight: one prepared frame may wait, at its decoded size.
        assert_eq!(series_price(2, 24, per_time, &fields), 39_200 * MIB);
        // An idle 32 GB box (about 29 GiB free) keeps every thread.
        assert_eq!(width_for(24, per_time, &fields, Some(29 * GIB)), 24);
        // 0.7 * 21 GiB = 15,053 MiB: two threads fit (14,560), three not.
        assert_eq!(width_for(24, per_time, &fields, Some(21 * GIB)), 2);
        // 0.7 * 19 GiB = 13,619 MiB, short of even one thread's price:
        // the valid time still runs, on one thread.
        assert_eq!(width_for(24, per_time, &fields, Some(19 * GIB)), 1);
        // Never wider than the machine, never narrower than one thread,
        // and an unknown price or memory keeps every thread.
        assert_eq!(width_for(4, GIB, &[GIB], Some(512 * GIB)), 4);
        assert_eq!(width_for(24, 100 * GIB, &[GIB], Some(8 * GIB)), 1);
        assert_eq!(width_for(24, 0, &[], Some(8 * GIB)), 24);
        assert_eq!(width_for(24, per_time, &fields, None), 24);
    }

    #[test]
    fn lanes_on_a_priced_pool_fit_their_price() {
        const GIB: u64 = 1 << 30;
        let fields = vec![GIB / 2; 12];
        let per_time = 13 * GIB;
        // 0.7 * 91 GiB = 63.7 GiB: three in flight on 24 threads price
        // 3 * 13 + 6 + 6 = 51 GiB, four 4 * 13 + 2 * 6 + 6 = 70 GiB.
        assert_eq!(lanes_on(24, per_time, &fields, Some(91 * GIB), 19), 3);
        // Never more than the pool's threads or the series' times.
        assert_eq!(lanes_on(2, per_time, &fields, Some(512 * GIB), 19), 2);
        assert_eq!(lanes_on(24, per_time, &fields, Some(512 * GIB), 5), 5);
        // A valid time bigger than the share still runs, alone.
        assert_eq!(lanes_on(24, per_time, &fields, Some(8 * GIB), 19), 1);
        // No price or no memory reading: the rule from before the price.
        assert_eq!(lanes_on(16, 0, &[], Some(512 * GIB), 49), 16);
        assert_eq!(lanes_on(16, per_time, &fields, None, 49), 2);
    }

    #[test]
    fn a_narrowed_pool_runs_on_fewer_threads_and_never_widens_again() {
        // The pool may already be built at full width by another test.
        assert_eq!(install(rayon::current_num_threads), width());
        narrow(1);
        assert_eq!(width(), 1);
        assert_eq!(install(rayon::current_num_threads), 1);
        narrow(threads());
        assert_eq!(width(), 1);
        assert_eq!(install(rayon::current_num_threads), 1);
    }

    #[test]
    fn every_cgroup_limit_above_the_process_is_read() {
        let text = "0::/user.slice/user-1000.slice/run-1.scope\n";
        assert_eq!(
            cgroup_v2_ancestry(Some(text)),
            vec!["/user.slice/user-1000.slice/run-1.scope", "/user.slice/user-1000.slice",
                 "/user.slice", ""]
        );
        // A container's namespaced view, cgroup v1 only, or no file: the root alone.
        assert_eq!(cgroup_v2_ancestry(Some("0::/\n")), vec![""]);
        assert_eq!(cgroup_v2_ancestry(Some("12:memory:/docker/x\n")), vec![""]);
        assert_eq!(cgroup_v2_ancestry(None), vec![""]);
    }

    #[test]
    fn a_cgroup_s_inactive_file_cache_counts_as_headroom() {
        const GIB: u64 = 1 << 30;
        let stat = "anon 4294967296\nfile 8589934592\nactive_file 1073741824\n\
                    inactive_file 7516192768\nfile_mapped 0\n";
        assert_eq!(memory_stat(stat, "inactive_file"), Some(7 * GIB));
        assert_eq!(memory_stat(stat, "file"), Some(8 * GIB));
        assert_eq!(memory_stat(stat, "shmem"), None);
        // 30 GiB limit, 12 GiB charged of which 7 GiB is inactive cache.
        assert_eq!(cgroup_headroom(30 * GIB, 12 * GIB, 7 * GIB), 25 * GIB);
        // Never more than the limit, never below zero.
        assert_eq!(cgroup_headroom(30 * GIB, 2 * GIB, 7 * GIB), 30 * GIB);
        assert_eq!(cgroup_headroom(30 * GIB, 40 * GIB, 0), 0);
    }

    #[test]
    fn available_memory_is_reported_where_the_platform_can_say() {
        if cfg!(any(target_os = "linux", windows)) {
            assert!(available_memory().is_some_and(|bytes| bytes > 0));
        }
    }

    #[test]
    fn slots_are_filled_by_position_not_by_finishing_order() {
        // The property every parallel step in this crate leans on: an
        // indexed collect puts input i's answer at index i, however the
        // work was scheduled.  Stated as a test because it is the whole
        // argument for byte identity under threads.
        let inputs: Vec<usize> = (0..512).collect();
        let observed: Vec<usize> = install(|| {
            inputs
                .par_iter()
                .map(|value| {
                    // Deliberately uneven work, so finishing order and
                    // input order cannot coincide by luck.
                    let spin = (511 - *value) % 17;
                    (0..spin).fold(*value, |accumulator, _| accumulator);
                    *value * 2
                })
                .collect()
        });
        assert_eq!(observed, inputs.iter().map(|value| value * 2).collect::<Vec<_>>());
    }

    #[test]
    fn the_first_refusal_in_document_order_is_the_one_reported() {
        let slots: Vec<crate::refusal::Result<usize>> = vec![
            Ok(1),
            Err(crate::refusal::decode_failed("second")),
            Err(crate::refusal::decode_failed("third")),
        ];
        let refusal = in_order(slots).unwrap_err();
        assert_eq!(refusal.message, "second");
    }

    #[test]
    fn a_child_cap_charges_retained_streams_and_is_not_discounted_twice() {
        // Host gives 700 usable bytes, child cap 500 less 120 already held.
        assert_eq!(usable_budget_on(Some(1000), Some(500), 0, Some(120)).unwrap(), Some(380));
        // This stream's 80-byte first frame is already in its series price.
        // Its other 40 resident bytes (including another stream) stay charged.
        assert_eq!(usable_budget_on(Some(1000), Some(500), 80, Some(120)).unwrap(), Some(460));
        assert_eq!(usable_budget_on(Some(100), Some(500), 0, Some(120)).unwrap(), Some(70));
        assert_eq!(usable_budget_on(None, Some(500), 0, Some(120)).unwrap(), Some(380));
        assert_eq!(usable_budget_on(Some(1000), Some(50), 80, Some(160)).unwrap(), Some(0));
        assert_eq!(usable_budget_on(None, None, 0, None).unwrap(), None);
        assert!(usable_budget_on(Some(1000), Some(500), 0, None).is_err());
    }

    #[test]
    fn explicit_time_lanes_cannot_bypass_the_reserved_child_budget() {
        let fit = lanes_on_budget(24, 130, &[60, 20], Some(220), 48);
        assert_eq!(fit, 1);
        assert_eq!(limit_requested_lanes(fit, Some(48)), 1);
        assert_eq!(limit_requested_lanes(5, Some(2)), 2);
    }

    #[test]
    fn an_unaffordable_first_time_is_a_structured_refusal() {
        let failure = require_minimum(130, &[60, 20], Some(189)).unwrap_err();
        assert_eq!(failure.class, crate::refusal::class::HOST_MEMORY);
        assert!(failure.message.contains("190 bytes"));
        assert!(failure.message.contains("189 usable bytes"));
        assert!(failure.message.contains("before decode"));
        assert!(require_minimum(130, &[60, 20], Some(190)).is_ok());
    }

    #[test]
    fn a_source_time_bigger_than_the_share_runs_alone_as_it_did_in_2_8_4() {
        const GIB: u64 = 1 << 30;
        // THE REGRESSION: 7 GiB per source time, 9.5 GiB available, no
        // child budget.  The share is 0.7 * 9.5 = 6.65 GiB, under the
        // price; 2.8.4 narrowed the pool to one thread and decoded it.
        let per_time = 7 * GIB;
        let available = Some(19 * GIB / 2);
        let share = usable_budget_on(available, None, 0, None).unwrap();
        assert_eq!(share, Some((19.0 * GIB as f64 / 2.0 * MEMORY_SHARE) as u64));
        assert!(series_price(1, 1, per_time, &[]) > share.unwrap());
        // The share only narrows: one thread, one lane, never zero.
        assert_eq!(width_for_budget(24, per_time, &[], share), 1);
        assert_eq!(lanes_on_budget(1, per_time, &[], share, 19), 1);
        // The minimum is held to the memory there is, and it fits.
        let minimum = minimum_budget_on(available, None, 0, None).unwrap();
        assert_eq!(minimum, available);
        assert!(require_minimum(per_time, &[], minimum).is_ok());
        // The frame this stream already holds is part of that memory.
        assert_eq!(minimum_budget_on(Some(GIB), None, 8 * GIB, None).unwrap(), Some(9 * GIB));
        // An unknown reading refuses nothing.
        assert_eq!(minimum_budget_on(None, None, 0, None).unwrap(), None);
        assert!(require_minimum(per_time, &[], None).is_ok());
    }

    #[test]
    fn the_minimum_still_refuses_what_the_host_or_a_child_budget_cannot_hold() {
        const GIB: u64 = 1 << 30;
        // No child budget: a source time bigger than ALL available memory
        // is the refusal that names a real out-of-memory.
        let per_time = 7 * GIB;
        let minimum = minimum_budget_on(Some(6 * GIB), None, 0, None).unwrap();
        let refusal = require_minimum(per_time, &[], minimum).unwrap_err();
        assert_eq!(refusal.class, crate::refusal::class::HOST_MEMORY);
        assert!(refusal.message.contains("before decode"));
        // A declared child budget keeps the strict check: the usable
        // budget (the share of the host, or the cap less what the child
        // already holds), not the whole reading.
        let strict = minimum_budget_on(Some(19 * GIB / 2), Some(64 * GIB), 0, Some(GIB)).unwrap();
        assert_eq!(strict, usable_budget_on(Some(19 * GIB / 2), Some(64 * GIB), 0, Some(GIB)).unwrap());
        assert!(require_minimum(per_time, &[], strict).is_err());
        let capped = minimum_budget_on(Some(512 * GIB), Some(6 * GIB), 0, Some(GIB)).unwrap();
        assert_eq!(capped, Some(5 * GIB));
        assert!(require_minimum(per_time, &[], capped).is_err());
        // A child that cannot measure itself still cannot enforce a cap.
        assert!(minimum_budget_on(Some(512 * GIB), Some(6 * GIB), 0, None).is_err());
    }

    #[test]
    fn current_process_memory_is_measured_on_supported_hosts() {
        if cfg!(any(target_os = "linux", windows)) {
            let (rss, owned) = process_memory().expect("own process memory");
            assert!(rss > 0);
            assert!(owned >= rss);
        }
    }

    #[test]
    fn an_unpriced_codec_cannot_expand_inside_a_reserved_child() {
        assert!(acquisition_budget_check("bz2", None).is_ok());
        let error = acquisition_budget_check("bz2", Some(1 << 30)).unwrap_err();
        assert_eq!(error.class, crate::refusal::class::HOST_MEMORY);
        assert!(error.message.contains("before decompression"));
    }
}
