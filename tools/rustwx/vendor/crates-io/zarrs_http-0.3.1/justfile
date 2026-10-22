TOOLCHAIN := "nightly"
export RUST_BACKTRACE := "0"

# Display the available recipes
help:
    @just --list --unsorted

# Build (cargo check)
build:
    cargo +{{TOOLCHAIN}} check

# Test
test:
    cargo +{{TOOLCHAIN}} test
    cargo +{{TOOLCHAIN}} test --examples

# Format with rustfmt
fmt:
    cargo +{{TOOLCHAIN}} fmt

# Lint with clippy
clippy:
    cargo +{{TOOLCHAIN}} clippy -- -D warnings

# Generate documentation
doc:
    RUSTDOCFLAGS="-D warnings --cfg docsrs" cargo +{{TOOLCHAIN}} doc -Z unstable-options -Z rustdoc-scrape-examples --no-deps

# Build/test/clippy/doc/check formatting - recommended before a PR
check: build test clippy doc
    cargo +{{TOOLCHAIN}} fmt --all -- --check

# Run clippy with extra lints
_clippy_extra:
    cargo +{{TOOLCHAIN}} clippy -- -D warnings -W clippy::nursery -A clippy::significant-drop-tightening -A clippy::significant-drop-in-scrutinee

_coverage_install:
    cargo install cargo-llvm-cov --locked

_coverage_report:
    cargo +{{TOOLCHAIN}} llvm-cov --doctests --html

_coverage_file:
    cargo +{{TOOLCHAIN}} llvm-cov --doctests --lcov --output-path lcov.info
