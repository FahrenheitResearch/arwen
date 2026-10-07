#!/usr/bin/env bash
# Native WRF v4.7.1 SFIRE routines. WRF sources remain byte-unmodified.
set -euo pipefail
if [[ $# != 2 ]]; then
    echo 'usage: build.sh WRF_SOURCE_ROOT BUILD_DIR' >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$build_dir" == "$source_root" || "$build_dir" == "$source_root/"* ]]; then
    echo 'BUILD_DIR inside WRF source would change the read-only reference' >&2
    exit 3
fi
(cd "$source_root" && sha256sum -c "$script_dir/SOURCES.sha256")
mkdir -p "$build_dir"
cd "$build_dir"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
flags=(-O0 -cpp -ffp-contract=off -fcheck=all -fbacktrace -ffree-form -ffree-line-length-none -fallow-argument-mismatch)
if [[ -n "${SFIRE_ORACLE_EXTRA_FLAGS:-}" ]]; then
    read -r -a extra_flags <<< "$SFIRE_ORACLE_EXTRA_FLAGS"
    flags+=("${extra_flags[@]}")
fi
defines=(-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4)
objects=(stub_wrf.o module_model_constants.o module_wrf_error.o module_fr_fire_util.o module_fr_fire_phys.o module_fr_fire_core.o module_fr_fire_model.o module_fr_fire_atm.o oracle_io.o)
gfortran -c "${flags[@]}" "$script_dir/stub_wrf.F90"
for rel in share/module_model_constants.F frame/module_wrf_error.F phys/module_fr_fire_util.F phys/module_fr_fire_phys.F phys/module_fr_fire_core.F phys/module_fr_fire_model.F phys/module_fr_fire_atm.F; do
    source_file="$source_root/$rel"
    if [[ "$rel" == phys/module_fr_fire_atm.F && -n "${SFIRE_CORRECTED_ATM_SOURCE:-}" ]]; then
        source_file=$(realpath "$SFIRE_CORRECTED_ATM_SOURCE")
    fi
    gfortran -c "${flags[@]}" "${defines[@]}" -I "$build_dir" -I "$source_root/phys" -o "$(basename "${rel%.F}").o" "$source_file"
done
gfortran -c "${flags[@]}" "$script_dir/oracle_io.F90"
for obj in "${objects[@]}"; do
    nm -u "$obj" > "undefined-${obj%.o}.txt"
    if grep -q '_ZGV' "undefined-${obj%.o}.txt"; then
        echo "$obj uses vector libm; scalar binary32 reference invalid" >&2
        exit 4
    fi
done
mkdir -p fixtures
cp "$script_dir/namelist.fire" namelist.fire
shopt -s nullglob
drivers=("$script_dir"/run_*.F90)
[[ ${#drivers[@]} -gt 0 ]] || { echo 'No executable oracle drivers' >&2; exit 5; }
for driver in "${drivers[@]}"; do
    name=$(basename "$driver" .F90)
    if [[ "$name" == run_corrected_tg && -z "${SFIRE_CORRECTED_ATM_SOURCE:-}" ]]; then
        continue
    fi
    gfortran -c "${flags[@]}" -I "$build_dir" "$driver"
    nm -u "$name.o" > "undefined-$name.txt"
    if grep -q '_ZGV' "undefined-$name.txt" && [[ "${SFIRE_ORACLE_EXTRA_FLAGS:-}" != *-O2* ]]; then
        echo "$name.o uses vector libm; scalar binary32 reference invalid" >&2
        exit 4
    fi
    gfortran -o "$name" "${objects[@]}" "$name.o"
    nice -n 10 "./$name" "$build_dir/fixtures" > "$name-stdout.txt"
done
gfortran --version | head -1 > compiler.txt
ldd --version | head -1 > libc.txt
printf '%s\n' "${flags[*]} ${defines[*]}" > compiler-flags.txt
cp "$script_dir/SOURCES.sha256" source-sha256sums.txt
(cd "$script_dir" && sha256sum *.F90 build.sh namelist.fire) > oracle-sha256sums.txt
if [[ -n "${SFIRE_CORRECTED_ATM_SOURCE:-}" ]]; then
    digest=$(sha256sum "$SFIRE_CORRECTED_ATM_SOURCE" | cut -d' ' -f1)
    printf '%s  module_fr_fire_atm_corrected.F\n' "$digest" > corrected-atm-sha256.txt
fi
find fixtures -type f -print0 | sort -z | xargs -0 sha256sum > fixture-sha256sums.txt
echo "Native SFIRE oracle fixtures: $build_dir/fixtures"
