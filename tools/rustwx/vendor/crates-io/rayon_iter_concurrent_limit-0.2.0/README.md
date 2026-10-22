# rayon_iter_concurrent_limit

[![Latest Version](https://img.shields.io/crates/v/rayon_iter_concurrent_limit.svg)](https://crates.io/crates/rayon_iter_concurrent_limit)
[![Documentation](https://docs.rs/rayon_iter_concurrent_limit/badge.svg)](https://docs.rs/rayon_iter_concurrent_limit)
![msrv](https://img.shields.io/crates/msrv/rayon_iter_concurrent_limit)
[![build](https://github.com/LDeakin/rayon_iter_concurrent_limit/actions/workflows/ci.yml/badge.svg)](https://github.com/LDeakin/rayon_iter_concurrent_limit/actions/workflows/ci.yml)

Limit the concurrency of an individual rayon parallel iterator method with a convenient macro.

- [API documentation (`docs.rs`)](https://docs.rs/rayon_iter_concurrent_limit/latest/rayon_iter_concurrent_limit/)
- [Changelog (`CHANGELOG.md`)](./CHANGELOG.md)

The documentation outlines the motivation, implementation, and limitations of this crate.

## Example
```rust
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;

const N: usize = 1000;
let output = iter_concurrent_limit!(2, (0..100), map, |i: usize| -> usize {
    let alloc = vec![i; N];              // max of 2 concurrent allocations
    alloc.into_par_iter().sum::<usize>() // runs on all threads
})
.map(|alloc_sum| -> usize {
    alloc_sum / N                        // max of 2 concurrent executions
})
.collect::<Vec<usize>>();
assert_eq!(output, (0..100).into_iter().collect::<Vec<usize>>());
```

## Licence
rayon_iter_concurrent_limit is licensed under either of
 - the Apache License, Version 2.0 [LICENSE-APACHE](./LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
 - the MIT license [LICENSE-MIT](./LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
