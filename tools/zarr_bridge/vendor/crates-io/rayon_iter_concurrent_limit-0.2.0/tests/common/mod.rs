use std::sync::atomic::{AtomicUsize, Ordering};

pub fn incr_active_operations(threads_active: &AtomicUsize) {
    threads_active.fetch_add(1, Ordering::SeqCst);
}

pub fn calc_active_operations(
    threads_active: &AtomicUsize,
    threads_active_max: &AtomicUsize,
) -> usize {
    let n_active = threads_active.fetch_sub(1, Ordering::SeqCst);
    threads_active_max.fetch_max(n_active, Ordering::SeqCst);
    threads_active_max.load(Ordering::SeqCst)
}
