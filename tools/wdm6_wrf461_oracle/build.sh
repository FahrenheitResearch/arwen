#!/usr/bin/env bash
# Build the WDM6 (mp_physics=16) WRF v4.6.1 column oracle:
#
#   run_wdm6_stock    byte-unmodified phys/module_mp_wdm6.F
#   run_wdm6_defined  the same file with exactly the three documented
#                     defined behaviours of docs/wdm6_oracle_known_deltas.md
#                     (sections 2, 5, 6) substituted by defined_arm.py
#   run_wdm6_only{2,5,6}  one substitution each, to attribute the stock
#                     comparison's differences to the divergence behind them
#
# usage: build.sh WRF_SOURCE_ROOT BUILD_DIR
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
wdm6_sha256="78b7c148f7c29b55670f137636d632aba040bf338d8214e1a5dd191dacf67d70"
constants_source="${source_root}/share/module_model_constants.F"
radar_source="${source_root}/phys/module_mp_radar.F"
wdm6_source="${source_root}/phys/module_mp_wdm6.F"
massv_source="${source_root}/frame/libmassv.F"
for source in "${constants_source}" "${radar_source}" "${wdm6_source}" \
              "${massv_source}"; do
    test -f "${source}"
done

# The reference is the pinned source, not merely a file with that name.
if [[ -d "${source_root}/.git" ]]; then
    head_commit=$(git -C "${source_root}" rev-parse HEAD)
    if [[ "${head_commit}" != "${pinned_commit}" ]]; then
        echo "WRF tree is at ${head_commit}, not pinned ${pinned_commit}" >&2
        exit 3
    fi
    if ! git -C "${source_root}" diff --quiet HEAD -- \
            share/module_model_constants.F phys/module_mp_radar.F \
            phys/module_mp_wdm6.F frame/libmassv.F; then
        echo "a compiled WRF WDM6 dependency differs from pinned HEAD" >&2
        exit 3
    fi
fi
echo "${wdm6_sha256}  ${wdm6_source}" | sha256sum -c --quiet

arms=(stock defined only2 only5 only6)
mkdir -p "${build_dir}"/{stock,defined,only2,only5,only6}
cd "${build_dir}"
python3 "${script_dir}/defined_arm.py" "${wdm6_source}" \
    "${build_dir}/module_mp_wdm6_defined.F"
for s in 2 5 6; do
    python3 "${script_dir}/defined_arm.py" "${wdm6_source}" \
        "${build_dir}/module_mp_wdm6_only${s}.F" "${s}"
done

fflags=(-O0 -cpp -ffree-form -ffree-line-length-none -DRWORDSIZE=4)
for arm in "${arms[@]}"; do
    cd "${build_dir}/${arm}"
    if [[ ${arm} == stock ]]; then src="${wdm6_source}";
    else src="${build_dir}/module_mp_wdm6_${arm}.F"; fi
    gfortran -c "${fflags[@]}" "${script_dir}/stub_wrf_error.F90"
    gfortran -c "${fflags[@]}" "${constants_source}"
    gfortran -c "${fflags[@]}" -I . "${radar_source}"
    # vsrec/vssqrt (the VREC/VSQRT macros at module_mp_wdm6.F:1-7) are
    # frame/libmassv.F's portable loops, as in a stock WRF build.
    gfortran -c -O0 -cpp -ffixed-form "${massv_source}"
    gfortran -c "${fflags[@]}" -I . -o module_mp_wdm6.o "${src}"
    gfortran -c "${fflags[@]}" -I . "${script_dir}/run_wdm6.F90"
    gfortran -o "../run_wdm6_${arm}" stub_wrf_error.o module_model_constants.o \
        module_mp_radar.o libmassv.o module_mp_wdm6.o run_wdm6.o
    # The -O0 object linked into the oracle must stay on scalar libm.
    nm -u module_mp_wdm6.o | sed 's/^ *//' | sort > undefined-O0.txt
    if grep -q '_ZGV' undefined-O0.txt; then
        echo "-O0 WDM6 object (${arm}) pulled in a libmvec vector symbol" >&2
        exit 4
    fi
    # No fused multiply-add anywhere in the linked reference.
    if objdump -d "../run_wdm6_${arm}" | grep -Eq 'vfn?m(add|sub)'; then
        echo "the ${arm} oracle contains FMA instructions" >&2
        exit 4
    fi
done
cd "${build_dir}"

# Evidence-only WRF-flags object: never linked into the reference.
gfortran -c -O2 -ftree-vectorize -funroll-loops -cpp -ffree-form \
    -ffree-line-length-none -DRWORDSIZE=4 -I stock \
    -o module_mp_wdm6_O2vec.o "${wdm6_source}"
nm -u module_mp_wdm6_O2vec.o | sed 's/^ *//' | sort > undefined-O2vec.txt
gfortran -c -O2 -ftree-vectorize -funroll-loops \
    -o libmvec_positive_control.o "${script_dir}/libmvec_positive_control.F90"
nm -u libmvec_positive_control.o | sed 's/^ *//' | sort > undefined-control.txt
if ! grep -q '_ZGV' undefined-control.txt; then
    echo "libmvec positive control produced no _ZGV symbol" >&2
    exit 5
fi
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 WDM6 object (linked reference):"
    grep -E 'expf|powf|logf|log10|sqrtf|atanf|_ZGV' stock/undefined-O0.txt || true
    echo "# -O2 -ftree-vectorize -funroll-loops (WRF physics flags):"
    grep -E 'expf|powf|logf|log10|sqrtf|atanf|_ZGV' undefined-O2vec.txt || true
    echo "# positive control (same EXP, vectorisable loop, WRF flags):"
    grep -E 'expf|_ZGV' undefined-control.txt || true
} > libmvec-report.txt
cat libmvec-report.txt

sha256sum "${constants_source}" "${radar_source}" "${wdm6_source}" \
    "${massv_source}" "${build_dir}"/module_mp_wdm6_*.F \
    "${script_dir}"/*.F90 "${script_dir}/defined_arm.py" \
    > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "${pinned_commit}" > wrf-commit.txt
echo "WDM6 oracle built in ${build_dir}"
