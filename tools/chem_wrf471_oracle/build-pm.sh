#!/usr/bin/env bash
# Prepare an owned source overlay without changing the pinned reference tree.
# Usage: bash build-pm.sh PINNED_ROOT OWNED_BUILD_DIR
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned=$(realpath "$1")
build=$(realpath -m "$2")
mkdir -p "$build/source/share" "$build/source/tools" "$build/source/oracle"
# Whole WRF chem files remain read-only. Only the generated Registry substitute
# and the two pinned public reference files live in this owned overlay.
if [[ ! -e "$build/source/chem" ]]; then ln -s "$pinned/chem" "$build/source/chem"; fi
cp "$script_dir/reference-pm/module_model_constants.F" "$build/source/share/"
cp "$script_dir/reference-pm/gen_scalar_indices.c" "$build/source/tools/"
cp "$script_dir/reference-pm/registry-pm.F90" "$build/source/oracle/"
(cd "$build/source"; sha256sum -c "$script_dir/SOURCES-pm.sha256")
bash "$script_dir/build.sh" "$build/source" "$build/oracle" run_sum_pm
