//! Polite-by-default scheduling for background compute.
//!
//! [`set_current_thread_background_priority`] drops only the calling thread.
//! [`build_background_pool`] supplies a dedicated, capped rayon pool whose
//! threads run below normal while the process stays at normal priority.
//! Run CPU work inside `pool.install(..)` so nested `par_iter`s ride that pool.

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

/// No-op off Windows: there is no portable std priority API.
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
