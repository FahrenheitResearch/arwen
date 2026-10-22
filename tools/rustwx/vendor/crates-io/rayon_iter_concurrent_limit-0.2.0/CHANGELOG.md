# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2024-02-29

### Changed
 - **Breaking**: The iterator argument in the `iter_concurrent_limit!` macro now expects an expression implementing `rayon::iter::IntoParallelIterator`
   - Example: use `iter_concurrent_limit!(n, (0..100), method, op)` instead of `iter_concurrent_limit!(n, (0..100).into_par_iter(), method, op)`
   - The `into_par_iter()` iterator must implement `rayon::iter::IndexedParallelIterator`
   - Some iterator methods require the iterator argument to also implement `std::iter::IntoIterator` for fast paths
 - Add fast paths in `iter_concurrent_limit!` if `concurrent_limit` is 1 for methods: `for_each`, `try_for_each`, `any`, `all`
   - These paths do not start additional `rayon` work items

## [0.1.0] - 2024-02-19

### Fixed
 - Remove incorrect panics docs for `iter_subdivide`

## [0.1.0-alpha4] - 2024-02-18

### Changed
 - Minor documentation improvements
 - Add fast paths avoiding chunking for various methods in `iter_concurrent_limit` if `concurrent_limit` is zero

### Fixed
 - Fixed `iter_subdivide` if supplied with an empty iterator
 - Handle `num_chunks` of zero in `iter_subdivide`

## [0.1.0-alpha3] - 2024-02-05

### Changed
 - **Breaking**: rename `chunks_concurrent_limit` to `iter_subdivide`
 - Various documentation improvements

## [0.1.0-alpha2] - 2024-02-04

### Added
 - Add `chunks_concurrent_limit_map` tests
 - Add doc example with equivalent code to macro

### Changed
 - **Breaking**: Swapped the argument order of `iterator` and `concurrent_limit` in `chunks_concurrent_limit`

## [0.1.0-alpha] - 2024-02-04

### Added
 - Initial alpha release for review

[unreleased]: https://github.com/LDeakin/rayon_iter_concurrent_limit/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/LDeakin/rayon_iter_concurrent_limit/releases/tag/v0.2.0
[0.1.0]: https://github.com/LDeakin/rayon_iter_concurrent_limit/releases/tag/v0.1.0
[0.1.0-alpha4]: https://github.com/LDeakin/rayon_iter_concurrent_limit/releases/tag/v0.1.0-alpha4
[0.1.0-alpha3]: https://github.com/LDeakin/rayon_iter_concurrent_limit/releases/tag/v0.1.0-alpha3
[0.1.0-alpha2]: https://github.com/LDeakin/rayon_iter_concurrent_limit/releases/tag/v0.1.0-alpha2
[0.1.0-alpha]: https://github.com/LDeakin/rayon_iter_concurrent_limit/releases/tag/v0.1.0-alpha
