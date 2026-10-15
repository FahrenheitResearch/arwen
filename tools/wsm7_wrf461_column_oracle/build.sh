#!/usr/bin/env bash
# Build the WSM7 (mp_physics = 24) WRF v4.6.1 column oracle and run it over
# the column sets make_columns.py writes.
#
#   bash build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# FILE SET.  WRF's own phys/module_mp_wsm7.F (the scheme and the wrapper
# module_microphysics_driver.F calls), phys/module_mp_radar.F (its 10 cm
# reflectivity machinery), share/module_model_constants.F (every physical
# constant the driver passes) and frame/libmassv.F (the portable vsrec/vssqrt
# loops the VREC/VSQRT macros name in a stock WRF build), plus a service-only
# module_wrf_error stub (wrf_debug discards its message; no arithmetic).
#
# ARMS (defined_arm.py):
#   run_wsm7_stock    the byte-unmodified module_mp_wsm7.F
#   run_wsm7_defined  the two declared divergences D1 (remap search) and D2
#                     (hail in REFL_10CM) substituted -- what WOOF's kernel is
#                     graded against word for word
#   run_wsm7_d1       D1 alone, to attribute the stock comparison
#   run_wsm7_probe    stock arithmetic plus a per-column fall-through counter;
#                     its outputs must equal the stock arm's word for word
#
# -O0 and libmvec.  WRF compiles phys/ at -O2 -ftree-vectorize, where the
# vectoriser may swap scalar expf/logf/powf for glibc's vector forms.  The
# reference is built at -O0; this script FAILS if any _ZGV* symbol appears in
# an -O0 scheme object or an FMA instruction in a linked reference, and a
# positive control proves the libmvec grep can see the symbol here.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
wsm7_sha256="d3ae349fc5e122660a3b1bfafef6ae82dae17c1bd79581688190d41daadac807"
radar_sha256="aa99da858be41efa579966680708d230123a7417560af0eb2e24f4c94e253688"
constants_source="${source_root}/share/module_model_constants.F"
radar_source="${source_root}/phys/module_mp_radar.F"
wsm7_source="${source_root}/phys/module_mp_wsm7.F"
massv_source="${source_root}/frame/libmassv.F"
for source in "${constants_source}" "${radar_source}" "${wsm7_source}" \
              "${massv_source}"; do
    test -f "${source}"
done

# The reference is the pinned source, not merely a file with that name.
if [[ -e "${source_root}/.git" ]]; then
    head_commit=$(git -C "${source_root}" rev-parse HEAD)
    if [[ "${head_commit}" != "${pinned_commit}" ]]; then
        echo "WRF tree is at ${head_commit}, not pinned ${pinned_commit}" >&2
        exit 3
    fi
    if ! git -C "${source_root}" diff --quiet HEAD -- \
            share/module_model_constants.F phys/module_mp_radar.F \
            phys/module_mp_wsm7.F frame/libmassv.F; then
        echo "a compiled WRF WSM7 dependency differs from pinned HEAD" >&2
        exit 3
    fi
fi
echo "${wsm7_sha256}  ${wsm7_source}" | sha256sum -c --quiet
echo "${radar_sha256}  ${radar_source}" | sha256sum -c --quiet

arms=(stock defined d1 probe)
mkdir -p "${build_dir}"
cd "${build_dir}"
for arm in defined d1 probe; do
    python3 "${script_dir}/defined_arm.py" "${wsm7_source}" \
        "${build_dir}/module_mp_wsm7_${arm}.F" "${arm}"
done

fflags=(-O0 -cpp -ffree-form -ffree-line-length-none -DRWORDSIZE=4)
for arm in "${arms[@]}"; do
    mkdir -p "${build_dir}/${arm}"
    cd "${build_dir}/${arm}"
    if [[ ${arm} == stock ]]; then src="${wsm7_source}";
    else src="${build_dir}/module_mp_wsm7_${arm}.F"; fi
    probe_flag=()
    [[ ${arm} == probe ]] && probe_flag=(-DWSM7_PROBE)
    gfortran -c "${fflags[@]}" "${script_dir}/stub_wrf_error.F90"
    gfortran -c "${fflags[@]}" "${constants_source}"
    gfortran -c "${fflags[@]}" -I . "${radar_source}"
    gfortran -c -O0 -cpp -ffixed-form "${massv_source}"
    gfortran -c "${fflags[@]}" -I . -o module_mp_wsm7.o "${src}"
    gfortran -c "${fflags[@]}" "${probe_flag[@]}" -I . \
        "${script_dir}/run_wsm7_columns.F90"
    gfortran -o "../run_wsm7_${arm}" stub_wrf_error.o module_model_constants.o \
        module_mp_radar.o libmassv.o module_mp_wsm7.o run_wsm7_columns.o
    nm -u module_mp_wsm7.o module_mp_radar.o | sed 's/^ *//' | grep -v ':$' \
        | sort -u > undefined-O0.txt
    if grep -q '_ZGV' undefined-O0.txt; then
        echo "-O0 WSM7 object (${arm}) pulled in a libmvec vector symbol" >&2
        exit 4
    fi
    if objdump -d "../run_wsm7_${arm}" | grep -Eq 'vfn?m(add|sub)'; then
        echo "the ${arm} oracle contains FMA instructions" >&2
        exit 4
    fi
done
cd "${build_dir}"

# init / radar_init words (the stock objects; wsm7init is untouched by D1/D2)
gfortran -c "${fflags[@]}" -I stock -o dump_wsm7_init.o \
    "${script_dir}/dump_wsm7_init.F90"
gfortran -o dump_wsm7_init stock/stub_wrf_error.o stock/module_model_constants.o \
    stock/module_mp_radar.o stock/libmassv.o stock/module_mp_wsm7.o dump_wsm7_init.o
./dump_wsm7_init > wsm7-init-words.txt

# Evidence-only WRF-flags object: never linked into the reference.
gfortran -c -O2 -ftree-vectorize -funroll-loops -cpp -ffree-form \
    -ffree-line-length-none -DRWORDSIZE=4 -I stock \
    -o module_mp_wsm7_O2vec.o "${wsm7_source}"
nm -u module_mp_wsm7_O2vec.o | sed 's/^ *//' | sort > undefined-O2vec.txt
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
    echo "# -O0 (THE REFERENCE) libm/libmvec symbols, scheme + radar objects:"
    grep -E 'exp|log|pow|sqrt|atan|hypot|_ZGV' stock/undefined-O0.txt || true
    echo "# -O2 -ftree-vectorize -funroll-loops (WRF's phys/ flags):"
    grep -E 'exp|log|pow|sqrt|atan|hypot|_ZGV' undefined-O2vec.txt || true
    echo "# positive control (same EXP, vectorisable loop, WRF flags):"
    grep -E 'expf|_ZGV' undefined-control.txt || true
} > libmvec-report.txt
cat libmvec-report.txt

# --- run -------------------------------------------------------------------------
python3 "${script_dir}/make_columns.py" "${build_dir}"
# (set, dt, nsteps): dt 30 is one minor loop; 60 one loop at half dtcldcr;
# 150 one loop with dtcld > dtcldcr; 300 nint(2.5) = 3 loops of 100 s.
runs=("columns 30 12" "columns 60 8" "columns 150 3" "columns 300 4" "columns72 30 4")
for r in "${runs[@]}"; do
    set -- $r
    tag="$1-dt$2-n$3"
    for arm in stock defined d1; do
        "./run_wsm7_${arm}" "$1.bin" "wrf-${arm}-${tag}.bin" "$2" "$3"
    done
    ./run_wsm7_probe "$1.bin" "wrf-probe-${tag}.bin" "$2" "$3" "probe-${tag}.txt"
    if ! cmp -s "wrf-probe-${tag}.bin" "wrf-stock-${tag}.bin"; then
        echo "the probe arm's outputs differ from the stock arm's (${tag})" >&2
        exit 6
    fi
done
sha256sum "${constants_source}" "${radar_source}" "${wsm7_source}" \
    "${massv_source}" "${build_dir}"/module_mp_wsm7_*.F \
    "${script_dir}"/*.F90 "${script_dir}/defined_arm.py" \
    "${script_dir}/make_columns.py" columns.bin columns72.bin \
    wrf-*.bin > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "${pinned_commit}" > wrf-commit.txt
echo "WSM7 oracle built and run in ${build_dir}"
