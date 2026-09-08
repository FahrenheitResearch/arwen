mod common;

use std::sync::atomic::AtomicUsize;

use common::{calc_active_operations, incr_active_operations};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;

const DUR: core::time::Duration = core::time::Duration::from_millis(10);

#[test]
fn readme() {
    let threads_active = AtomicUsize::new(0);
    let threads_active_max = AtomicUsize::new(0);
    let threads_active_map = AtomicUsize::new(0);
    let threads_active_map_max = AtomicUsize::new(0);

    let concurrent_limit = 2;
    const N: usize = 1000;
    let output = iter_concurrent_limit!(concurrent_limit, (0..100), map, |i: usize| -> usize {
        let alloc = vec![i; N]; // max of 2 concurrent allocations
        incr_active_operations(&threads_active);
        std::thread::sleep(DUR);
        calc_active_operations(&threads_active, &threads_active_max);
        alloc.into_par_iter().sum::<usize>() // runs on all threads
    })
    .map(|alloc_sum| -> usize {
        incr_active_operations(&threads_active_map);
        std::thread::sleep(DUR);
        calc_active_operations(&threads_active_map, &threads_active_map_max);
        alloc_sum / N // max of 2 concurrent executions
    })
    .collect::<Vec<usize>>();
    assert_eq!(output, (0..100).into_iter().collect::<Vec<usize>>());
    assert_eq!(threads_active_max.into_inner(), concurrent_limit);
    assert_eq!(threads_active_map_max.into_inner(), concurrent_limit);
}
