#!/usr/bin/env bash
# WRFINPUT RW_NETCDF RUSTC PYTHON BUILD_DIR OUTPUT_DIR
# Real-state NetCDF decoding and cropping stay in Rust.
set -euo pipefail
input=$(realpath "$1")
reader=$(realpath "$2")
rust_compiler=$3
python_bin=$4
build_dir=$(realpath -m "$5")
output_dir=$(realpath -m "$6")
tool_dir=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$build_dir" "$output_dir"
mapfile -t variables < <(cut -f1 "$tool_dir/state-fields.tsv")
"$reader" dump --raw "$input" "$build_dir/dump" "${variables[@]}"
"$rust_compiler" --edition=2021 "$tool_dir/crop_state.rs" -O -o "$build_dir/crop_state"
"$build_dir/crop_state" "$tool_dir/state-fields.tsv" "$build_dir/dump" "$build_dir/crop" 70 70 12 10
"$python_bin" "$tool_dir/pack_state.py" "$build_dir/crop" "$output_dir/state-real.npz"
sha256sum "$input" "$reader" "$build_dir/crop_state" > "$build_dir/state-source-sha256.txt"
