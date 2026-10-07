#!/usr/bin/env bash
# Stage the shared harness in separate list directories, without editing it.
# build.sh globs sources-*.list and extract-*.list (GSL lists are gsl-*.list
# and never match); the staging keeps each build directory to its own sources.
# Usage: bash build_fire_oracles.sh SOURCE_ROOT OWNED_BUILD_ROOT
set -euo pipefail
source_root=$(realpath "$1")
build_root=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$build_root/harness-wrf" "$build_root/harness-gsl"
for stage in harness-wrf harness-gsl; do
    cp "$script_dir/oracle_io.F90" "$script_dir/libmvec_positive_control.F90" \
       "$script_dir/SOURCES.sha256" "$script_dir/SOURCES-smoke.sha256" "$build_root/$stage/"
done
cp "$script_dir/build.sh" "$script_dir/stub_wrf.F90" "$script_dir/wetdep_prelude.inc" \
   "$script_dir/extract-smoke.list" "$script_dir/run_wetdep_ls.F90" "$build_root/harness-wrf/"
cp "$script_dir/build_gsl_smoke.sh" "$script_dir/gsl-sources-fire.list" "$script_dir/gsl-extract-fire.list" \
   "$script_dir/gsl_run_add_emiss_burn.F90" "$script_dir/gsl_run_smoke_prep.F90" \
   "$script_dir/gsl_run_fire_rules.F90" "$build_root/harness-gsl/"
bash "$build_root/harness-wrf/build.sh" "$source_root" "$build_root/wrf" run_wetdep_ls
bash "$build_root/harness-gsl/build_gsl_smoke.sh" "$source_root" "$build_root/gsl4" 4
bash "$build_root/harness-gsl/build_gsl_smoke.sh" "$source_root" "$build_root/gsl8" 8
