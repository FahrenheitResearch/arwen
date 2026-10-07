#!/usr/bin/env bash
# Build the real plume solvers in disjoint original/observed WRF and GSL trees.
# Usage: build_plume_oracles.sh WRF_SOURCE_ROOT BUILD_ROOT [PYTHON]
# Numerical statements in observed sources are unchanged; every emitted word
# is compared with the original solver before diagnostics are published.
set -euo pipefail
source_root=$(realpath "$1")
build_root=$(realpath -m "$2")
python=${3:-python3}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "$build_root/" in "$source_root/"*)
    echo 'build root must not overlap pinned sources; building there would overwrite the read-only oracle reference' >&2; exit 2;;
esac
mkdir -p "$build_root/wrf-harness" "$build_root/gsl-harness" "$build_root/wrf4"
for arm in wrf gsl; do
    harness="$build_root/$arm-harness"
    for file in build.sh build_gsl_smoke.sh oracle_io.F90 libmvec_positive_control.F90 \
                stub_wrf.F90 stub_smoke.F90 SOURCES.sha256 SOURCES-smoke.sha256 \
                plume_trace.F90 build_plume_observed.py plume_column_prelude.inc \
                plume_poison_wrf.F90 plume_poison_gsl.F90 \
                run_plumerise_wrfchem.F90 gsl_run_plumerise_frp.F90; do
        cp "$script_dir/$file" "$harness/$file"
    done
done
cp "$script_dir/sources-smoke.list" "$script_dir/extract-smoke.list" "$script_dir/wetdep_prelude.inc" "$build_root/wrf-harness/"
cp "$script_dir/gsl-sources-plume.list" "$build_root/gsl-harness/"
bash "$build_root/wrf-harness/build.sh" "$source_root" "$build_root/wrf4" run_plumerise_wrfchem
"$python" "$script_dir/build_plume_observed.py" wrf "$source_root" "$build_root/wrf4" "$build_root/wrf-harness"
for kind in 4 8; do
    bash "$build_root/gsl-harness/build_gsl_smoke.sh" "$source_root" "$build_root/gsl$kind" "$kind" gsl_run_plumerise_frp
    "$python" "$script_dir/build_plume_observed.py" gsl "$source_root" "$build_root/gsl$kind" "$build_root/gsl-harness"
done
# Copy only the subset's extraction list into its own GSL harness. WRF's
# build.sh intentionally reads every sources-*.list, so sharing the lane's
# live directory would compile the colliding GSL modules into its build.
mkdir -p "$build_root/subset-harness"
cp "$build_root/gsl-harness/"* "$build_root/subset-harness/"
cp "$script_dir/gsl-extract-plume.list" "$script_dir/gsl_run_ebu_distribute.F90" "$build_root/subset-harness/"
for kind in 4 8; do
    bash "$build_root/subset-harness/build_gsl_smoke.sh" "$source_root" "$build_root/subset$kind" "$kind" gsl_run_ebu_distribute
done
