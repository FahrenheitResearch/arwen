#!/usr/bin/env bash
# Build the WSM5 (mp_physics = 4) column oracle from the WRF v4.6.1 build
# trees themselves and run it over the column sets make_columns.py writes.
#
#   bash build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD_ROOT]
#
# WRF_BUILD_ROOT (default /work/wrf/build) holds the full serial builds of
# the pinned source: <root>/scalar/WRF-4.6.1 (-O2 -fno-fast-math
# -ffp-contract=off -fno-tree-vectorize: the 0 ULP referee) and
# <root>/strict/WRF-4.6.1 (the same without -fno-tree-vectorize: WRF's own
# vectoriser on).  No WSM5 source is recompiled for the stock arms: the
# driver links each tree's OWN phys/module_mp_wsm5.o and module_mp_radar.o,
# with the rest of that tree's libwrflib.a, so the words are the words the
# tree's wrf.exe computes.
#
# ARMS.  scalar and strict: the trees' objects.  stock, remap, psevp and
# defined: the scalar tree's preprocessed module_mp_wsm5.f90 with the edits
# patch_arms.py names, compiled with the scalar tree's exact compile line
# (read from its compile.log).  The stock arm differs from the tree's object
# only by integer probe counters; its words must equal the scalar arm's, the
# control that the recompile reproduces the build.  defined is what WOOF's
# kernel computes (both declared fixes).
#
# The WRF Fortran I/O flag -fconvert=big-endian is dropped from the DRIVER's
# compile line only (it sets the byte order of the driver's own unformatted
# files, nothing else); the driver's one float operation (t = th*pii) keeps
# -ffp-contract=off.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_BUILD_ROOT]" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
trees_root=$(realpath "${3:-/work/wrf/build}")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
wsm5_files=(phys/module_mp_wsm5.F phys/module_mp_radar.F share/module_model_constants.F
            frame/libmassv.F)

head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- "${wsm5_files[@]}"; then
    echo "a WSM5 oracle source differs from the pinned commit" >&2
    exit 3
fi
for tree in scalar strict; do
    t="${trees_root}/${tree}/WRF-4.6.1"
    for f in "${wsm5_files[@]}"; do
        cmp -s "${source_root}/${f}" "${t}/${f}" || {
            echo "${tree} tree's ${f} is not the pinned source" >&2; exit 3; }
    done
    for o in phys/module_mp_wsm5.o phys/module_mp_radar.o main/libwrflib.a; do
        [[ -f "${t}/${o}" ]] || { echo "${tree} tree lacks ${o}" >&2; exit 3; }
    done
done

mkdir -p "${build_dir}"
cd "${build_dir}"

# The scalar tree's own compile line for module_mp_wsm5, flags and include
# path, read from its compile.log.
scalar="${trees_root}/scalar/WRF-4.6.1"
line=$(grep -E 'gfortran -o module_mp_wsm5\.o -c ' "${scalar}/compile.log" | head -1)
[[ -n "${line}" ]] || { echo "no module_mp_wsm5 compile line in the scalar log" >&2; exit 4; }
tree_flags=$(sed -E 's/.*gfortran -o module_mp_wsm5\.o -c (.*) module_mp_wsm5\.f90.*/\1/' <<< "${line}")
echo "scalar tree compile flags: ${tree_flags}" | tee compile-flags.txt
grep -q -- '-fno-tree-vectorize' <<< "${tree_flags}" || {
    echo "the scalar tree's line lacks -fno-tree-vectorize" >&2; exit 4; }
read -r -a arm_flags <<< "$(sed "s|-I\.\./|-I${scalar}/phys/../|g" <<< "${tree_flags}")"
# The driver takes the arithmetic flags only: its include path is built per
# arm below, so the arm's module_mp_wsm5.mod is the one it reads.
read -r -a drv_flags <<< "$(sed -E 's/-fconvert=big-endian//; s/-frecord-marker=4//; s/-I[^ ]+//g' <<< "${tree_flags}")"

tree_inc() {   # include path of one tree, for the driver
    local t=$1
    echo "-I${t}/main -I${t}/frame -I${t}/share -I${t}/phys -I${t}/external/esmf_time_f90 -I${t}/inc"
}
# The libraries wrf.exe links after libwrflib.a, as its link line in the
# same log lists them (wrf_debug, which module_mp_radar calls on its error
# paths, brings the WRF frame with it).
tree_libs() {
    local t=$1
    echo "-Wl,--start-group ${t}/main/libwrflib.a" \
         "${t}/external/fftpack/fftpack5/libfftpack.a" \
         "${t}/external/io_grib1/libio_grib1.a" \
         "${t}/external/io_grib_share/libio_grib_share.a" \
         "${t}/external/io_int/libwrfio_int.a" \
         "-L${t}/external/esmf_time_f90 -lesmf_time" \
         "${t}/frame/module_internal_header_util.o ${t}/frame/pack_utils.o" \
         "-L${t}/external/io_netcdf -lwrfio_nf -Wl,--end-group" \
         "-L${NETCDF:-/work/wrf/netcdf}/lib -lnetcdff -lnetcdf -lm"
}

link() {   # $1 exe, $2 tree, $3 module dir for module_mp_wsm5.mod, $4 wsm5 object, rest: cpp defines
    local exe=$1 t=$2 moddir=$3 obj=$4; shift 4
    local d; d=$(dirname "${exe}")
    mkdir -p "${d}"
    # shellcheck disable=SC2046
    gfortran -cpp "$@" "${drv_flags[@]}" -I"${moddir}" $(tree_inc "${t}") -J "${d}" \
        -c -o "${d}/$(basename "${exe}").o" "${src}"
    # shellcheck disable=SC2046
    gfortran -o "${exe}" "${d}/$(basename "${exe}").o" "${obj}" \
        "${t}/phys/module_mp_radar.o" $(tree_libs "${t}")
}

# --- the stock arms: each tree's own objects ----------------------------------
for tree in scalar strict; do
    t="${trees_root}/${tree}/WRF-4.6.1"
    src="${script_dir}/run_wsm5_columns.F90"
    link "arm-${tree}/run_wsm5_columns" "${t}" "${t}/phys" "${t}/phys/module_mp_wsm5.o"
    src="${script_dir}/remap_probe.F90"
    link "arm-${tree}/remap_probe" "${t}" "${t}/phys" "${t}/phys/module_mp_wsm5.o"
    nm -u "${t}/phys/module_mp_wsm5.o" "${t}/phys/module_mp_radar.o" | sed 's/^ *//' \
        | sort -u > "undefined-${tree}.txt"
done

# --- the defined arms: the scalar tree's source, edited, compiled its way -----
python3 "${script_dir}/patch_arms.py" "${scalar}/phys/module_mp_wsm5.f90" arms-src
for arm in stock remap psevp defined; do
    mkdir -p "arm-${arm}"
    ( cd "arm-${arm}" && gfortran -o module_mp_wsm5.o -c "${arm_flags[@]}" \
        -I"${scalar}/phys" -J . "../arms-src/module_mp_wsm5_${arm}.f90" )
    src="${script_dir}/run_wsm5_columns.F90"
    link "arm-${arm}/run_wsm5_columns" "${scalar}" "${PWD}/arm-${arm}" \
        "arm-${arm}/module_mp_wsm5.o" -DWSM5_PROBE
    src="${script_dir}/remap_probe.F90"
    link "arm-${arm}/remap_probe" "${scalar}" "${PWD}/arm-${arm}" "arm-${arm}/module_mp_wsm5.o"
done

# --- the init-constant dump (scalar tree's objects) ----------------------------
src="${script_dir}/dump_wsm5_init.F90"
link "dump/dump_wsm5_init" "${scalar}" "${scalar}/phys" "${scalar}/phys/module_mp_wsm5.o"
./dump/dump_wsm5_init > wsm5-init-f32.txt
cat wsm5-init-f32.txt | head -40

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# scalar tree (THE REFERENCE) libm/libmvec symbols of module_mp_wsm5.o + module_mp_radar.o:"
    grep -E 'exp|log|pow|sqrt|cabs|clog|csqrt|_ZGV' undefined-scalar.txt || true
    echo "# strict tree (WRF's vectoriser on):"
    grep -E 'exp|log|pow|sqrt|cabs|clog|csqrt|_ZGV' undefined-strict.txt || true
} > libmvec-report.txt
cat libmvec-report.txt
if grep -q '_ZGV' undefined-scalar.txt; then
    echo "the scalar referee pulled a vector libm symbol" >&2
    exit 5
fi

# --- run --------------------------------------------------------------------------
python3 "${script_dir}/make_columns.py" "${build_dir}"
# main set (45 levels): dt 30 s one minor loop; 150 s one loop longer than
# dtcldcr; 300 s nint(2.5) = 3 loops; 600 s 5 loops; 1800 s 15 loops.
main_runs=("30 12" "150 3" "300 4" "600 2" "1800 1")
# lev2 set (2 levels): the remap's last interface at every CFL scale.
lev2_runs=("0.001 3" "1 3" "20 4" "300 2")
for arm in scalar strict stock remap psevp defined; do
    for r in "${main_runs[@]}"; do
        set -- $r
        "./arm-${arm}/run_wsm5_columns" main.bin "wrf-${arm}-main-dt$1-n$2.bin" "$1" "$2"
    done
    for r in "${lev2_runs[@]}"; do
        set -- $r
        "./arm-${arm}/run_wsm5_columns" lev2.bin "wrf-${arm}-lev2-dt$1-n$2.bin" "$1" "$2"
    done
    "./arm-${arm}/remap_probe" "remap-probe-${arm}.bin"
done

sha256sum "${source_root}"/phys/module_mp_wsm5.F "${source_root}"/phys/module_mp_radar.F \
    "${source_root}"/share/module_model_constants.F "${source_root}"/frame/libmassv.F \
    "${trees_root}"/scalar/WRF-4.6.1/phys/module_mp_wsm5.o \
    "${trees_root}"/scalar/WRF-4.6.1/phys/module_mp_radar.o \
    "${trees_root}"/strict/WRF-4.6.1/phys/module_mp_wsm5.o \
    "${trees_root}"/strict/WRF-4.6.1/phys/module_mp_radar.o \
    arms-src/*.f90 "${script_dir}"/*.F90 "${script_dir}"/*.py \
    main.bin lev2.bin wrf-*.bin remap-probe-*.bin > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "oracle built and run in ${build_dir}"
