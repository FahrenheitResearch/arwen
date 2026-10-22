#!/bin/bash
set -eu
out=${1:?usage: build.sh OUTPUT_DIRECTORY}
mkdir -p "$out"
root=$(cd "$(dirname "$0")" && pwd)
"${RUSTC:-rustc}" --edition=2021 -O --crate-type cdylib "$root/io.rs" -o "$out/libwaves_io.so"
"${RUSTC:-rustc}" --edition=2021 -O "$root/io.rs" -o "$out/rw_waves_io"
"${RUSTC:-rustc}" --edition=2021 --test "$root/io.rs" -o "$out/waves_io_tests"
"$out/waves_io_tests"
