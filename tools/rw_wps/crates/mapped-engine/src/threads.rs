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
//! THE CAP is measured, not guessed: the sweep it comes from is under
//! [`THREAD_CAP`].

use std::sync::OnceLock;

use rayon::{ThreadPool, ThreadPoolBuilder};

/// Override for the worker count, for measurement and for a caller who
/// is running several engines at once and wants each one narrower.
pub const THREADS_ENV: &str = "GPUWM_MAPPED_ENGINE_THREADS";

/// The measured cap.
///
/// The sweep, on a 32-logical-CPU box, whole decode minus the frameset
/// write (`examples/decode_timing.rs`, no `--output`, warm cache):
///
///   threads      1       2       4       8      16      32
///   hrrr-prs  27.67   18.51   13.24   11.30   10.99   11.20  s
///   gdas      14.97    9.23    6.52    5.65    5.63       -   s
///
/// Four workers take three quarters of the available speedup and eight
/// take essentially all of it; sixteen buys 2.7% on one source and 0.4%
/// on the other, and thirty-two is slower than sixteen.  Past the knee
/// the work is memory-bandwidth bound -- every worker is streaming a
/// freshly unpacked field out to RAM -- so more workers buy nothing and
/// cost peak resident memory, which on a 0.25-degree analysis is already
/// the binding constraint.
pub const THREAD_CAP: usize = 8;

/// Worker count for this process: the override if it parses to a
/// positive number, else the machine's parallelism capped at
/// [`THREAD_CAP`].
pub fn threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        if let Some(declared) = std::env::var(THREADS_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
        {
            return declared;
        }
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
            .min(THREAD_CAP)
            .max(1)
    })
}

/// The engine's pool.  Falls back to running the closure on the calling
/// thread if a pool cannot be built: a box that refuses threads must
/// still decode, just serially.
pub fn install<T: Send>(work: impl FnOnce() -> T + Send) -> T {
    static POOL: OnceLock<Option<ThreadPool>> = OnceLock::new();
    let pool = POOL.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(threads())
            .thread_name(|index| format!("mapped-engine-{index}"))
            .build()
            .ok()
    });
    match pool {
        Some(pool) => pool.install(work),
        None => work(),
    }
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
    fn the_pool_has_at_least_one_worker_and_respects_the_cap() {
        assert!(threads() >= 1);
        if std::env::var(THREADS_ENV).is_err() {
            assert!(threads() <= THREAD_CAP);
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
}
