//! Limit the concurrency of an individual rayon parallel iterator method with a convenient macro.
//!
//! # Example
//! This example demonstrates applying a concurrency-limited `map` to an iterator with the [`iter_concurrent_limit`] macro.
//! `map` is one of many supported methods of the macro.
//! ```rust
//! use rayon::iter::{IntoParallelIterator, ParallelIterator};
//! use rayon_iter_concurrent_limit::iter_concurrent_limit;
//! const N: usize = 1000;
//! let op = |i: usize| -> usize {
//!     let alloc = vec![i; N]; // max 2 concurrent allocations in this example
//!     alloc.into_par_iter().sum::<usize>() // runs on all threads
//! };
//! let sum_iter = iter_concurrent_limit!(2, (0..100), map, op);
//! let output = sum_iter
//!     .map(|alloc_sum| -> usize {
//!         alloc_sum / N // runs on all threads
//!     })
//!     .collect::<Vec<usize>>();
//! assert_eq!(output, (0..100).into_iter().collect::<Vec<usize>>());
//! ```
//! The equivalent `sum_iter` expression using [`iter_subdivide`] is:
//! ```rust
//! # use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
//! # use rayon_iter_concurrent_limit::iter_subdivide;
//! # const N: usize = 1000;
//! # let op = |i: usize| -> usize {
//! #     let alloc = vec![i; N]; // max 2 concurrent allocations in this example
//! #     alloc.into_par_iter().sum::<usize>() // runs on all threads
//! # };
//! let sum_iter = iter_subdivide(2, (0..100).into_par_iter())
//!     .flat_map_iter(|chunk| chunk.into_iter().map(op));
//! # let output = sum_iter
//! #     .map(|alloc_sum| -> usize {
//! #         alloc_sum / N // runs on all threads
//! #     })
//! #     .collect::<Vec<usize>>();
//! # assert_eq!(output, (0..100).into_iter().collect::<Vec<usize>>());
//! ```
//! The equivalent expression without using functionality in this crate is:
//! ```rust
//! # use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
//! # const N: usize = 1000;
//! # let op = |i: usize| -> usize {
//! #     let alloc = vec![i; N]; // max 2 concurrent allocations in this example
//! #     alloc.into_par_iter().sum::<usize>() // runs on all threads
//! # };
//! let sum_iter = (0..100)
//!     .into_par_iter()
//!     .chunks((100 + 2 - 1) / 2)
//!     .flat_map_iter(|chunk| chunk.into_iter().map(op));
//! # let output = sum_iter
//! #     .map(|alloc_sum| -> usize {
//! #         alloc_sum / N // runs on all threads
//! #     })
//! #     .collect::<Vec<usize>>();
//! # assert_eq!(output, (0..100).into_iter().collect::<Vec<usize>>());
//! ```
//!
//! # Motivation
//! Consider this example:
//! ```rust
//! use rayon::iter::{IntoParallelIterator, ParallelIterator};
//! let op = |_: usize| {
//!     // operation involving a large allocation
//! };
//! (0..100).into_par_iter().for_each(op);
//! ```
//! In this case, it may be necessary to limit the number of concurrent executions of `op` due to memory constraints.
//! The number of threads could be limited with [`rayon::ThreadPool::install`](rayon::ThreadPool::install) like so:
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use rayon::iter::{IntoParallelIterator, ParallelIterator};
//! # let op = |_: usize| {};
//! let thread_pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
//! thread_pool.install(|| {
//!     (0..100).into_par_iter().for_each(op);
//! });
//! # Ok(())
//! # }
//! ```
//!
//! However, this has some limitations and footguns:
//! - Any parallel operations within `op` will use the same thread-limited thread pool, unless [`install`](rayon::ThreadPool::install) is called internally with a different thread pool.
//! - If [`install`](rayon::ThreadPool::install) is called internally, `op` can yield and multiple instances of `op` may run concurrently on a thread. This is detailed [here](https://docs.rs/rayon/1.8.1/rayon/struct.ThreadPool.html#warning-execution-order) in the [`install`](rayon::ThreadPool::install) documentation.
//! - An iterator must be consumed in the [`install`](rayon::ThreadPool::install) scope of a [`ThreadPool`](rayon::ThreadPool), otherwise it will not use that thread pool.
//!
//! # Solution
//! This crate provides [`iter_concurrent_limit`], a macro that enables many [`rayon::iter::ParallelIterator`] methods to execute their operands with limited concurrency.
//!
//! The [Examples](crate::iter_concurrent_limit#examples) section of [`iter_concurrent_limit`] has usage examples for each method supported by the macro.
//!
//! ### Implementation
//! The macro limits concurrency by calling [`IndexedParallelIterator::chunks`] on the parallel iterator (using the [`iter_subdivide`] method) to reduce the number of work items for [`rayon`].
//! Internally, the [`iter_subdivide`] method calculates the chunk size as `iterator.len().ceiling_div(concurrent_limit)`.
//! The function passed to the macro is called sequentially on the items in each chunk, but in parallel over the chunks.
//! The output of the function is flattened for methods with an iterator output, like `map` and `filter`.
//!
//! ### Limitations
//! - Iterators passed to [`iter_concurrent_limit`] or [`iter_subdivide`] must implement [`std::iter::IntoIterator`] and [`rayon::iter::IntoParallelIterator`], and the created parallel iterator must implement [`rayon::iter::IndexedParallelIterator`].
//! - Only a subset of relevant [`ParallelIterator`](rayon::iter::ParallelIterator)/[`IndexedParallelIterator`] methods are currently supported by the [`iter_concurrent_limit`] macro.
// TODO: - Methods which rely on thread-local initialisation (e.g. [`rayon::iter::ParallelIterator::map_init`]) will not function identically when run though [`iter_concurrent_limit`].
//! - If the operator/predicate passed to [`iter_concurrent_limit`] is a closure, its signature might have to be made explicit

#![warn(unused_variables)]
#![warn(dead_code)]
#![deny(missing_docs)]

use rayon::iter::{Chunks, IndexedParallelIterator};

/// Subdivide a [`rayon::iter::IndexedParallelIterator`] into `num_chunks` chunks.
///
/// This returns the output of the [`IndexedParallelIterator::chunks`] function with a chunk size calculated according to:
/// ```rust
/// # use rayon::iter::IntoParallelIterator;
/// # use rayon::iter::IndexedParallelIterator;
/// # let num_chunks: usize = 1;
/// # let iterator = (0..1).into_par_iter();
/// (iterator.len() + num_chunks - 1) / num_chunks
/// # ;
/// ```
/// If `num_chunks` is zero, then there will be one chunk per iterator item.
///
/// If `num_chunks` does not evenly divide the iterator length, the last chunk will be smaller than the rest.
///
/// This method is used internally by the [`iter_concurrent_limit`] macro.
pub fn iter_subdivide<I: IndexedParallelIterator>(num_chunks: usize, iterator: I) -> Chunks<I> {
    if num_chunks == 0 {
        iterator.chunks(1)
    } else {
        let chunk_size = std::cmp::max((iterator.len() + num_chunks - 1) / num_chunks, 1);
        iterator.chunks(chunk_size)
    }
}

// TODO: Support more methods
/// Apply a method on a [`rayon::iter::IndexedParallelIterator`] with a limit on the number of concurrent executions of the function passed to the method.
///
/// Concurrent executions are limited by chunking the iterator to reduce the number of work items.
/// The [crate root documentation](crate) explains the motivation for this approach, provides further details on the underlying implementation of the macro, and details its limitations.
///
/// # Arguments
/// The macro arguments are `(concurrent_limit, iterator, method, function)`:
/// - `concurrent_limit` is a [`usize`] specifying the maximum concurrent executions of `function`.
///   - A `concurrent_limit` of zero means no concurrent limit. Some methods will skip internal chunking in this case.
/// - `iterator` implements [`std::iter::IntoIterator`] and [`rayon::iter::IntoParallelIterator`]
///   - The parallel iterator must implement [`rayon::iter::IndexedParallelIterator`].
/// - `method` is the name of a supported iterator method:
///   - Only methods which call a supplied function are supported.
///   - Operations without a function (e.g. min, max) will not allocate and there is little benefit in setting a concurrent limit for such methods.
///   - Not every iterator method matching this criteria is currently supported.
/// - `function` is a function compatible with `method`, such as an operation, predicate, etc.
///   - The function is called *sequentially* on the items in each chunk, but in *parallel* over the chunks, with the number of concurrent executions upper bounded by the `concurrent_limit`.
///   - Parallel rayon methods executed in the function will implicitly utilise the global thread pool unless an alternative thread pool has been installed (see [`rayon::ThreadPool`]).
///
/// # Examples
/// ### for_each
/// ```rust
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// let op = |i: usize| {
///     let alloc = vec![i; 1000]; // limited concurrency
///     alloc.into_par_iter().for_each(|_j| {}); // runs on all threads
/// };
/// iter_concurrent_limit!(2, (0..10), for_each, op);
/// ```
///
/// ### try_for_each
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// let op = |i: usize| {
///     let alloc = vec![i; 1000]; // limited concurrency
///     alloc.into_par_iter().for_each(|_j| {}); // runs on all threads
///     Ok::<(), std::io::Error>(())
/// };
/// iter_concurrent_limit!(2, (0..10), try_for_each, op)?;
/// # Ok(())
/// # }
/// ```
///
/// ### map
/// ```rust
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// let op = |i: usize| {
///     let alloc = vec![i; 1000]; // limited concurrency
///     alloc.into_par_iter().sum::<usize>() // runs on all threads
/// };
/// let sum =
///     iter_concurrent_limit!(2, (0..100), map, op)
///     .sum::<usize>();
/// assert_eq!(sum, (0..100).into_iter().map(op).sum::<usize>());
/// ```
///
/// ### filter
/// ```rust
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// let op = |i: &usize| -> bool {
///     // ... do work with limited concurrency
///     i % 2 == 0
/// };
/// let even =
///     iter_concurrent_limit!(2, (0..100), filter, op)
///     .collect::<Vec<usize>>();
/// assert_eq!(even, (0..100).into_iter().filter(op).collect::<Vec<usize>>());
/// ```
///
/// ### filter_map
/// ```rust
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// let op = |i: usize| -> Option<usize> {
///     // ... do work with limited concurrency
///     if i % 2 == 0 { Some(i * 2) } else { None }
/// };
/// let even_doubled =
///     iter_concurrent_limit!(2, (0..100), filter_map, op)
///     .collect::<Vec<usize>>();
/// assert_eq!(even_doubled, (0..100).into_iter().filter_map(op).collect::<Vec<usize>>());
/// ```
///
/// ### any
/// ```rust
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// let op = |i: usize| -> bool {
///     // ... do work with limited concurrency
///     i == 50
/// };
/// let any_eq_50 = iter_concurrent_limit!(2, (0..100), any, op);
/// assert_eq!(any_eq_50, (0..100).into_iter().any(op));
/// ```
///
/// ### all
/// ```rust
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # use rayon_iter_concurrent_limit::iter_concurrent_limit;
/// let op = |i: usize| -> bool {
///     // ... do work with limited concurrency
///     i == 50
/// };
/// let all_eq_50 = iter_concurrent_limit!(2, (0..100), all, op);
/// assert_eq!(all_eq_50, (0..100).into_iter().all(op));
/// ```
///
#[macro_export]
macro_rules! iter_concurrent_limit {
    ( $concurrent_limit:expr, $iterator:expr, for_each, $op:expr ) => {{
        let concurrent_limit = $concurrent_limit;
        let op = $op;
        if concurrent_limit == 0 {
            $iterator.into_par_iter().for_each(op)
        } else if concurrent_limit == 1 {
            $iterator.into_iter().for_each(op)
        } else {
            let chunks = $crate::iter_subdivide(concurrent_limit, $iterator.into_par_iter());
            chunks.for_each(|chunk| chunk.into_iter().for_each(op))
        }
    }};
    // TODO: for_each_with?
    // TODO: for_each_init?
    ( $concurrent_limit:expr, $iterator:expr, try_for_each, $op:expr ) => {{
        let concurrent_limit = $concurrent_limit;
        let op = $op;
        if concurrent_limit == 0 {
            $iterator.into_par_iter().try_for_each(op)
        } else if concurrent_limit == 1 {
            $iterator.into_iter().try_for_each(op)
        } else {
            let chunks = $crate::iter_subdivide(concurrent_limit, $iterator.into_par_iter());
            chunks.try_for_each(|chunk| chunk.into_iter().try_for_each(op))
        }
    }};
    // TODO: try_for_each_with?
    // TODO: try_for_each_init?
    ( $concurrent_limit:expr, $iterator:expr, map, $map_op:expr ) => {{
        let chunks = $crate::iter_subdivide($concurrent_limit, $iterator.into_par_iter());
        chunks.flat_map_iter(|chunk| chunk.into_iter().map($map_op))
    }};
    // TODO: map_with?
    // TODO: map_init?
    // IGNORE: inspect
    ( $concurrent_limit:expr, $iterator:expr, update, $update_op:expr ) => {{
        let chunks = $crate::iter_subdivide($concurrent_limit, $iterator.into_par_iter());
        chunks.flat_map_iter(|chunk| {
            chunk.into_iter().map(|mut item| {
                $update_op(&mut item);
                item
            })
        })
    }};
    ( $concurrent_limit:expr, $iterator:expr, filter, $filter_op:expr ) => {{
        let chunks = $crate::iter_subdivide($concurrent_limit, $iterator.into_par_iter());
        chunks.flat_map_iter(|chunk| chunk.into_iter().filter($filter_op))
    }};
    ( $concurrent_limit:expr, $iterator:expr, filter_map, $filter_op:expr ) => {{
        let chunks = $crate::iter_subdivide($concurrent_limit, $iterator.into_par_iter());
        chunks.flat_map_iter(|chunk| chunk.into_iter().filter_map($filter_op))
    }};
    // TODO: flat_map?
    // ( $concurrent_limit:expr, $iterator:expr, flat_map, $map_op:expr ) => {{
    //     let chunks = $crate::iter_subdivide($concurrent_limit, $iterator.into_par_iter());
    //     chunks.flat_map_iter(|chunk| chunk.into_iter().map($map_op))
    // }};
    // TODO: flat_map_iter?
    // TODO: reduce?
    // TODO: reduce_with?
    // TODO: try_reduce?
    // TODO: try_reduce_with?
    // TODO: fold?
    // TODO: fold_with?
    // TODO: try_fold?
    // TODO: try_fold_with?
    // ( $concurrent_limit:expr, $iterator:expr, max_by_key, $f:expr ) => {{
    //     let chunks = $crate::iter_subdivide($concurrent_limit, $iterator.into_par_iter());
    //     chunks
    //         .flat_map(|chunk| chunk.into_iter().max_by_key($f))
    //         .max_by_key($f)
    // }};
    // ( $concurrent_limit:expr, $iterator:expr, min_by_key, $f:expr ) => {{
    //     let chunks = iter_subdivide($concurrent_limit, $iterator.into_par_iter());
    //     chunks
    //         .flat_map(|chunk| chunk.into_iter().min_by_key($f))
    //         .min_by_key($f)
    // }};
    // TODO: find_any?
    // TODO: find_first?
    // TODO: find_last?
    // TODO: find_map_any?
    // TODO: find_map_first?
    // TODO: find_map_last?
    ( $concurrent_limit:expr, $iterator:expr, any, $predicate:expr ) => {{
        let concurrent_limit = $concurrent_limit;
        let predicate = $predicate;
        if concurrent_limit == 0 {
            $iterator.into_par_iter().any(predicate)
        } else if concurrent_limit == 1 {
            $iterator.into_iter().any(predicate)
        } else {
            let chunks = $crate::iter_subdivide(concurrent_limit, $iterator.into_par_iter());
            chunks.any(|chunk| chunk.into_iter().any(predicate))
        }
    }};
    ( $concurrent_limit:expr, $iterator:expr, all, $predicate:expr ) => {{
        let concurrent_limit = $concurrent_limit;
        let predicate = $predicate;
        if concurrent_limit == 0 {
            $iterator.into_par_iter().all(predicate)
        } else if concurrent_limit == 1 {
            $iterator.into_iter().all(predicate)
        } else {
            let chunks = $crate::iter_subdivide(concurrent_limit, $iterator.into_par_iter());
            chunks.all(|chunk| chunk.into_iter().all(predicate))
        }
    }};
    // TODO: partition?
    // TODO: partition_map?
    // TODO: take_any_while?
    // TODO: skip_any_while?
    // TODO: IndexedParallelIterator zip, zip_eq, fold_chunks, fold_chunks_with, cmp, partial_cmp, position_any, position_first, position_last, positions?
    ( $concurrent_limit:expr, $iterator:expr, $method:ident, $predicate:expr ) => {{
        std::compile_error!("This macro does not support the requested method");
    }};
}
