# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2025-04-23

### Fixed
 - make corner-case slicing behavior compatible with std::slice ([#4](https://github.com/LDeakin/unsafe_cell_slice/pull/4) by [@gzz2000])

## [0.2.1] - 2024-11-01

### Added
 - Add `UnsafeCellSlice::get_mut()`

### Changed
 - Support more index types in `UnsafeCellSlice::index_mut()`

## [0.2.0] - 2024-10-03

### Added
 - Add `UnsafeCellSlice::index_mut()`
 - Add `UnsafeCellSlice::len()` and `UnsafeCellSlice::is_empty()`

### Changed
 - Change rust edition to 2021
 - Bump MSRV to 1.63
 - Remove `Copy` bound on `T` in `UnsafeCellSlice`

### Removed
 - **Breaking**: Remove `UnsafeCellSlice::as_mut_slice()` (potentially unsound)

## [0.1.0] - 2024-09-01

### Added
 - Initial release

[unreleased]: https://github.com/LDeakin/unsafe_cell_slice/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/LDeakin/unsafe_cell_slice/releases/tag/v0.2.2
[0.2.1]: https://github.com/LDeakin/unsafe_cell_slice/releases/tag/v0.2.1
[0.2.0]: https://github.com/LDeakin/unsafe_cell_slice/releases/tag/v0.2.0
[0.1.0]: https://github.com/LDeakin/unsafe_cell_slice/releases/tag/v0.1.0

[@gzz2000]: https://github.com/gzz2000
