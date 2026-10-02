#!/usr/bin/env bash
# WRF_ROOT BUILD_DIR. Generated configuration interfaces are copied, never edited.
set -euo pipefail
wrf_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
tool_dir=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$build_dir"
constants_sha=$(sha256sum "$wrf_root/share/module_model_constants.F" | cut -d' ' -f1)
if [[ "$constants_sha" != 5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062 ]]; then
    echo 'WRF constants differ from the pinned v4.7.1 source' >&2
    exit 3
fi
python3 "$tool_dir/prep_build.py" "$wrf_root/dyn_em/module_big_step_utilities_em.F" "$build_dir"
cp "$wrf_root/frame/module_configure.mod" "$build_dir/"
cd "$build_dir"
flags=(-O0 -cpp -Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4
       -DDWORDSIZE=8 -DLWORDSIZE=4 -ffree-form -ffree-line-length-none -ffp-contract=off -fcheck=bounds)
gfortran "${flags[@]}" -c "$wrf_root/share/module_model_constants.F"
gfortran "${flags[@]}" -c "$wrf_root/frame/module_state_description.F"
gfortran "${flags[@]}" -c prep_exact.F90 run_prep.F90
gfortran -o run_prep module_model_constants.o module_state_description.o prep_exact.o run_prep.o
gfortran --version > prep-compiler.txt
printf '%s\n' "${flags[*]}" > prep-flags.txt
sha256sum "$wrf_root/dyn_em/module_big_step_utilities_em.F" "$wrf_root/share/module_model_constants.F" \
    "$wrf_root/frame/module_state_description.F" "$wrf_root/frame/module_configure.mod" \
    prep_exact.F90 run_prep.F90 run_prep > prep-source-sha256.txt
nm prep_exact.o > prep-symbols.txt
