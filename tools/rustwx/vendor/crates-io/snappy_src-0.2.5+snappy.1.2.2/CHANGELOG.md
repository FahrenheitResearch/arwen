# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.5+snappy.1.2.2] - 2025-03-27

### Changed
- Bump `snappy` to 1.2.2

## [0.2.4+snappy.1.2.1] - 2025-01-12

### Changed
 - Bump `bindgen` to 0.71
 - Explicitly set the MSRV to 1.63 (the effective MSRV of 0.2.3)

## [0.2.3+snappy.1.2.1] - 2024-06-22

### Fixed
 - Link C++ and ensure static snappy is included only ([#2](https://github.com/LDeakin/rust_snappy_src/pull/2) by [@mulimoen])
 - Make sure cc/Build::std is available ([#3](https://github.com/LDeakin/rust_snappy_src/pull/3) by [@mulimoen])

## [0.2.2+snappy.1.2.1] - 2024-06-22

### Changed
 - CI: Test Rust toolchain `stable-msvc` and `stable-gnu` on windows

### Fixed
 - Fix incorrect `HAVE_SYS_UIO_H` on `stable-gnu` Rust toolchain on windows

## [0.2.1+snappy.1.2.1] - 2024-06-22

### Fixed
 - Remove out-of-date `Cargo.toml` advice from docs

## [0.2.0+snappy.1.2.1] - 2024-06-22

### Added
 - Add snappy version to crate version and remove from docs

### Changed
 - Cleanup `build.rs`
 - Show C++ warnings in build
 - Add note about generating bindings in docs

### Fixed
 - Add `links=snappy` to `Cargo.toml`
 - Add `cargo:root` and `cargo:include`
   - Dependent crates get `DEP_SNAPPY_ROOT` and `DEP_SNAPPY_INCLUDE` in their build environment

## [0.1.1] - 2024-06-22

### Added
 - Add a `CHANGELOG.md`

### Changed
 - Changed `repository` and `description` in `Cargo.toml`

## [0.1.0] - 2024-06-22

### Added
- Initial release
- Snappy: **1.2.1** (2024/05/22)

[unreleased]: https://github.com/LDeakin/rust_snappy_src/compare/v0.2.5+snappy.1.2.2...HEAD
[0.2.5+snappy.1.2.2]: https://github.com/LDeakin/rust_snappy_src/releases/tag/v0.2.5+snappy.1.2.2
[0.2.4+snappy.1.2.1]: https://github.com/LDeakin/rust_snappy_src/releases/tag/v0.2.4+snappy.1.2.1
[0.2.3+snappy.1.2.1]: https://github.com/LDeakin/rust_snappy_src/releases/tag/v0.2.3+snappy.1.2.1
[0.2.2+snappy.1.2.1]: https://github.com/LDeakin/rust_snappy_src/releases/tag/v0.2.2+snappy.1.2.1
[0.2.1+snappy.1.2.1]: https://github.com/LDeakin/rust_snappy_src/releases/tag/v0.2.1+snappy.1.2.1
[0.2.0+snappy.1.2.1]: https://github.com/LDeakin/rust_snappy_src/releases/tag/v0.2.0+snappy.1.2.1
[0.1.1]: https://github.com/LDeakin/rust_snappy_src/releases/tag/v0.1.1
[0.1.0]: https://github.com/LDeakin/rust_snappy_src/releases/tag/v0.1.0

[@mulimoen]: https://github.com/mulimoen
