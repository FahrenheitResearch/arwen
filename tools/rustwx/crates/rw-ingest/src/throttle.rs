//! Polite-by-default scheduling for the compute-heavy binaries.
//!
//! The ingest/render bins saturate every core through rayon (the heavy
//! ECAPE stage in particular), which makes an interactive desktop unusable
//! while a run is in flight. This module gives each binary the same two
//! knobs, applied once at the very top of `main`:
//!
//! - drop the process to BELOW_NORMAL priority (Windows; no-op elsewhere)
//! - cap the global rayon pool at `cores - 2` so the desktop keeps two
//!   cores for the foreground
//!
//! `--full-throttle` restores the old behavior (normal priority, every
//! core) for dedicated nodes; `--threads N` pins the pool size explicitly
//! and wins over both defaults.
//!
//! Interactive hosts (the egui shell) must NOT use the process-wide knobs —
//! dropping the whole process would deprioritize the render thread too.
//! They get two additive APIs instead:
//! [`set_current_thread_background_priority`] (per-THREAD drop) and
//! [`build_background_pool`] (a dedicated, non-global rayon pool whose
//! threads all run below normal while the process stays at normal
//! priority — run the CPU half inside `pool.install(..)` so every nested
//! `par_iter` rides the capped pool).

/// Drop the CURRENT PROCESS to BELOW_NORMAL_PRIORITY_CLASS so foreground
/// apps preempt the ingest/render compute. Best-effort: a failure (which
/// SetPriorityClass essentially never reports for the own-process
/// pseudo-handle) just leaves the inherited priority in place.
#[cfg(windows)]
pub fn set_background_priority() {
    use windows_sys::Win32::System::Threading::{
        BELOW_NORMAL_PRIORITY_CLASS, GetCurrentProcess, SetPriorityClass,
    };
    // SAFETY: GetCurrentProcess returns the process pseudo-handle (never
    // fails, never needs closing); SetPriorityClass on it only adjusts the
    // calling process's own scheduling class.
    unsafe {
        let _ = SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
    }
}

/// No-op off Windows: there is no portable std priority API, and the dev
/// box this politeness exists for is Windows. The rayon thread cap in
/// `init_rayon_threads` is the cross-platform half of polite mode.
#[cfg(not(windows))]
pub fn set_background_priority() {}

/// Build the GLOBAL rayon pool with the policy thread count: an explicit
/// `--threads N` wins; else `--full-throttle` uses every core; else polite
/// default `max(1, cores - 2)`. Returns the count the pool runs with.
///
/// MUST be called before anything touches rayon (the global pool is built
/// lazily on first use and cannot be resized) — call it at the very top of
/// `main`. If the pool was already built (e.g. another test in the same
/// process got there first), the build error is ignored and the existing
/// pool's count is reported instead.
pub fn init_rayon_threads(requested: Option<usize>, full_throttle: bool) -> usize {
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let threads = match requested {
        Some(n) => n.max(1),
        None if full_throttle => cores,
        None => cores.saturating_sub(2).max(1),
    };
    if rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .is_err()
    {
        return rayon::current_num_threads();
    }
    threads
}

/// Drop only the CALLING THREAD to below-normal priority (Windows; no-op
/// elsewhere). One cheap syscall — interactive hosts call this at the top
/// of their fetch thread and from [`build_background_pool`]'s start
/// handler, keeping the process (and its render thread) at normal priority.
#[cfg(windows)]
pub fn set_current_thread_background_priority() {
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL,
    };
    // SAFETY: GetCurrentThread returns the thread pseudo-handle (never
    // fails, never needs closing); SetThreadPriority on it only adjusts the
    // calling thread's own priority.
    unsafe {
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
}

/// No-op off Windows (same rationale as [`set_background_priority`]).
#[cfg(not(windows))]
pub fn set_current_thread_background_priority() {}

/// The polite thread count: `requested` if given, else `max(1, cores - 2)`.
pub fn polite_thread_count(requested: Option<usize>) -> usize {
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    match requested {
        Some(n) => n.max(1),
        None => cores.saturating_sub(2).max(1),
    }
}

/// Build a DEDICATED (non-global) rayon pool for background compute in an
/// interactive process: capped at the polite thread count, every pool
/// thread dropped to below-normal priority at spawn. Run work inside
/// `pool.install(...)` — nested `par_iter`s then ride this pool instead of
/// the global one, so the UI thread keeps two cores and normal priority
/// even at 100% compute load.
pub fn build_background_pool(requested: Option<usize>) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(polite_thread_count(requested))
        .thread_name(|index| format!("rw-ingest-{index}"))
        .start_handler(|_| set_current_thread_background_priority())
        .build()
        .expect("build the dedicated rw-ingest rayon pool")
}

/// One-call wiring for a binary's `main`: apply the priority drop (unless
/// `--full-throttle`), build the global rayon pool, and print the one-line
/// mode banner. Call FIRST in `main`, before any rayon use.
pub fn apply(requested: Option<usize>, full_throttle: bool) {
    if !full_throttle {
        set_background_priority();
    }
    let threads = init_rayon_threads(requested, full_throttle);
    if full_throttle {
        println!("throttle: full");
    } else {
        println!(
            "throttle: below-normal priority, {threads} threads \
             (use --full-throttle on dedicated nodes)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polite_thread_count_honors_request_and_floors_at_one() {
        assert_eq!(polite_thread_count(Some(7)), 7);
        assert_eq!(polite_thread_count(Some(0)), 1);
        assert!(polite_thread_count(None) >= 1);
    }

    /// The dedicated pool is non-global and actually executes installed
    /// work (nested parallelism rides it via install()).
    #[test]
    fn build_background_pool_runs_installed_work() {
        use rayon::prelude::*;
        let pool = build_background_pool(Some(2));
        assert_eq!(pool.current_num_threads(), 2);
        let sum: u64 = pool.install(|| (0..1000u64).into_par_iter().sum());
        assert_eq!(sum, 499_500);
        let on_pool_thread = pool.install(|| {
            std::thread::current()
                .name()
                .is_some_and(|name| name.starts_with("rw-ingest-"))
        });
        assert!(on_pool_thread, "install() must run on a named pool thread");
    }
}
