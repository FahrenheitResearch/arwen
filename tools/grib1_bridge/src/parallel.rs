//! Persistent, explicitly sized Rayon pool for independent host kernels.
//! A task owns a disjoint output range and retains the serial arithmetic.

use rayon::prelude::*;
use std::io::Write;
use std::sync::OnceLock;

#[path = "../../preparation_resources.rs"]
pub mod resources;

fn pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| rayon::ThreadPoolBuilder::new()
        .num_threads(resources::available_workers())
        .thread_name(|index| format!("preprocess-{index}"))
        .build().map_err(|error| {
            let _ = writeln!(std::io::stderr(),
                "preparation worker warning: native Rayon pool could not start ({error}); running host kernels with one worker");
        }).ok()).as_ref()
}

pub fn run_ranges(length: usize, workers: usize, body: impl Fn(usize, usize) + Sync) {
    if length == 0 { return; }
    let workers = workers.max(1).min(resources::available_workers()).min(length);
    if workers == 1 { body(0, length); return; }
    let ranges = crate::worker_ranges(length, workers);
    match pool() {
        Some(pool) => pool.install(|| ranges.par_iter().for_each(|&(start, stop)| body(start, stop))),
        None => for (start, stop) in ranges { body(start, stop); },
    }
}

/// Report the actual capacity of the native pool, independent of inherited
/// global Rayon and BLAS settings. Zero asks for all available workers.
#[no_mangle]
pub extern "C" fn gpuwm_preprocess_cpu_parallelism(requested: usize) -> usize {
    let width = pool().map_or(1, rayon::ThreadPool::current_num_threads)
        .min(resources::available_workers());
    if requested == 0 { width } else { requested.min(width).max(1) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[test]
    fn every_element_has_one_owner_at_every_width() {
        for workers in [1, 8, 32, resources::available_cpus()] {
            let output: Vec<_> = (0..257).map(|_| AtomicUsize::new(0)).collect();
            run_ranges(output.len(), workers, |start, stop| {
                for cell in &output[start..stop] { cell.fetch_add(1, Ordering::Relaxed); }
            });
            assert!(output.iter().all(|v| v.load(Ordering::Relaxed) == 1));
        }
    }
}
