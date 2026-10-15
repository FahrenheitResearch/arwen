#!/usr/bin/env bash
# Evidence arms for section 5 (split_arm.py), never graded against WOOF:
#   only5t   section 5's transfer with WRF's own sub-step count
#   fine8, fine32  section 5 as the defined arm has it, 8x and 32x WOOF's count
# Run after build.sh and run_oracle.sh; reuses build.sh's stock objects.
# usage: measure_split.sh WRF_SOURCE_ROOT BUILD_DIR FIXTURE_DIR OUT_ROOT
set -euo pipefail
source_root=$(realpath "$1"); build_dir=$(realpath "$2"); fixture=$3; out=$4
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
wdm6_source="${source_root}/phys/module_mp_wdm6.F"
echo "78b7c148f7c29b55670f137636d632aba040bf338d8214e1a5dd191dacf67d70  ${wdm6_source}" \
    | sha256sum -c --quiet
fflags=(-O0 -cpp -ffree-form -ffree-line-length-none -DRWORDSIZE=4)
for pair in only5t:transfer fine8:fine8 fine32:fine32; do
    arm=${pair%%:*}; mode=${pair#*:}
    mkdir -p "${build_dir}/${arm}"
    python3 "${script_dir}/split_arm.py" "${wdm6_source}" \
        "${build_dir}/split_module_mp_wdm6_${arm}.F" "${mode}"
    cd "${build_dir}/${arm}"
    cp ../stock/*.mod ../stock/stub_wrf_error.o ../stock/module_model_constants.o \
       ../stock/module_mp_radar.o ../stock/libmassv.o .
    gfortran -c "${fflags[@]}" -I . -o module_mp_wdm6.o \
        "${build_dir}/split_module_mp_wdm6_${arm}.F"
    gfortran -c "${fflags[@]}" -I . "${script_dir}/run_wdm6.F90"
    gfortran -o "../run_wdm6_${arm}" stub_wrf_error.o module_model_constants.o \
        module_mp_radar.o libmassv.o module_mp_wdm6.o run_wdm6.o
    if objdump -d "../run_wdm6_${arm}" | grep -Eq 'vfn?m(add|sub)'; then
        echo "the ${arm} oracle contains FMA instructions" >&2; exit 4
    fi
    cd - > /dev/null
    mkdir -p "${out}/${arm}"
    for f in "${fixture}"/*.in; do
        n=$(basename "$f" .in)
        "${build_dir}/run_wdm6_${arm}" "$f" "${out}/${arm}/${n}.out" \
            "${out}/${arm}/constants-${n}.txt"
    done
    echo "${arm} arm written to ${out}/${arm}"
done
