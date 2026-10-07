#!/usr/bin/env bash
# Build and run the NOAA GSL smoke oracles (ccpp-physics physics/smoke_dust/,
# Apache-2.0) in their own build directory.
#
#   bash tools/chem_wrf471_oracle/build_gsl_smoke.sh WRF_SOURCE_ROOT BUILD_DIR KIND [gsl_run_NAME ...]
#
# WHY NOT build.sh.  build.sh links every object it builds into every driver
# from ONE build directory, and GSL's smoke code defines modules WRF-Chem also
# defines, with different contents: module_zero_plumegen_coms,
# module_add_emiss_burn and module_wetdep_ls exist in both chem/ and
# gsl-ccpp-physics/physics/smoke_dust/.  Two .mod files of one name in one
# directory is a silent last-writer-wins, so the GSL oracles build here, in
# BUILD_DIR, and never share it with build.sh.  Their drivers are named
# gsl_run_<name>.F90 -- NOT run_<name>.F90 -- because build.sh compiles every
# run_*.F90 in this directory against WRF-Chem's objects.
#
# KIND is the `machine` module's kind_phys: 4 or 8.  GSL declares every real
# `real(kind_phys)`; RRFS builds it at 8.  The REFERENCE the float32 port is
# graded against is KIND=4: HRRR-Smoke, the model this arm reproduces, ran
# inside WRF at -r4, and the port is float32.  KIND=8 exists only so a lane can
# report how far RRFS's own float64 answer is from the float32 one; it is never
# a gate.  gfortran reads an unsuffixed literal (0.0006, 1.e+7) as REAL(4) in
# both builds, exactly as it does inside RRFS.
#
# Everything else follows build.sh's rules:
#   * every compiled or extracted file must match its SHA-256 in SOURCES.sha256
#     or a SOURCES-*.sha256 in this directory (the pins are shared);
#   * gsl-sources-*.list: whole GSL files, compiled in listed order, paths
#     relative to WRF_SOURCE_ROOT (the lists are named gsl-*, never
#     sources-*/extract-*: build.sh globs those two prefixes and would compile
#     GSL's modules into WRF-Chem's directory); a path named by two lists compiles once, at
#     its first position (lanes add lists, never edit each other's);
#   * gsl-extract-*.list: byte-for-byte LINE RANGES for code that cannot be
#     compiled whole (rrfs_smoke_wrapper.F90 needs mpi_f08 and the whole CCPP):
#         <include_name> <rel_path> <first_line> <last_line>
#     copies lines first..last of <rel_path>, untouched, into
#     BUILD_DIR/<include_name>, for a driver to INCLUDE inside a routine whose
#     declarations the driver writes;
#   * -O0 -cpp -DRWORDSIZE=4, no fast math, the libmvec _ZGV* guard on every
#     -O0 object and the -Ofast positive control proving the guard can fire;
#   * oracle_io.F90 is the fixture writer; each driver gets
#     BUILD_DIR/fixtures/<name> as its one argument.
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: build_gsl_smoke.sh WRF_SOURCE_ROOT BUILD_DIR KIND [gsl_run_NAME ...]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
kind="$3"
shift 3
only=("$@")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ "${kind}" != "4" && "${kind}" != "8" ]]; then
    echo "KIND must be 4 (the float32 reference) or 8 (RRFS's float64 build)" >&2
    exit 2
fi

shopt -s nullglob

declare -A seen=()
sources=()
for list in "${script_dir}"/gsl-sources-*.list; do
    while IFS= read -r rel; do
        rel="${rel%%#*}"; rel="$(echo "${rel}" | xargs)"
        [[ -z "${rel}" ]] && continue
        if [[ -z "${seen[${rel}]:-}" ]]; then
            seen["${rel}"]=1
            sources+=("${rel}")
        fi
    done < "${list}"
done

extracts=()
for list in "${script_dir}"/gsl-extract-*.list; do
    while IFS= read -r line; do
        line="${line%%#*}"; line="$(echo "${line}" | xargs)"
        [[ -n "${line}" ]] && extracts+=("${line}")
    done < "${list}"
done

# --- byte identity to the pinned sources ------------------------------------
pins=("${script_dir}"/SOURCES.sha256 "${script_dir}"/SOURCES-*.sha256)
declare -A pinned=()
for pin in "${pins[@]}"; do
    while read -r sum rel; do
        [[ -z "${sum}" || "${sum}" == \#* ]] && continue
        rel="${rel#\*}"
        pinned["${rel}"]="${sum}"
    done < "${pin}"
done
check() {
    local rel="$1"
    if [[ -z "${pinned[${rel}]:-}" ]]; then
        echo "${rel} is compiled but pinned in no SOURCES*.sha256" >&2
        exit 3
    fi
    local have
    have=$(sha256sum "${source_root}/${rel}" | cut -d' ' -f1)
    if [[ "${have}" != "${pinned[${rel}]}" ]]; then
        echo "${rel}: sha256 ${have}, pinned ${pinned[${rel}]}" >&2
        exit 3
    fi
}
for rel in "${sources[@]}"; do
    check "${rel}"
done
for line in "${extracts[@]}"; do
    read -r _name rel _first _last <<< "${line}"
    check "${rel}"
done

mkdir -p "${build_dir}"
cd "${build_dir}"

free="-ffree-form -ffree-line-length-none"
fc="gfortran -c -O0 -cpp -DRWORDSIZE=4 ${free} -I ${build_dir}"

libmvec_guard() {
    local base="$1"
    nm -u "${base}.o" | sed 's/^ *//' | sort > "undefined-O0-${base}.txt"
    if grep -q '_ZGV' "undefined-O0-${base}.txt"; then
        echo "-O0 object ${base}.o pulled in a libmvec vector symbol:" >&2
        grep '_ZGV' "undefined-O0-${base}.txt" >&2
        exit 4
    fi
}

# The one non-GSL module: ccpp's `machine` reduced to the kind it exports.
cat > machine.F90 <<EOF
! GENERATED by tools/chem_wrf471_oracle/build_gsl_smoke.sh (KIND=${kind}).
! ccpp-physics machine.F exports kind_phys; nothing else is read by the
! smoke_dust files this harness compiles.
module machine
  implicit none
  integer, parameter :: kind_phys = ${kind}
  integer, parameter :: kind_dbl_prec = 8
  integer, parameter :: kind_sngl_prec = 4
end module machine
EOF
${fc} machine.F90
objects=(machine.o)

for rel in "${sources[@]}"; do
    base=$(basename "${rel}")
    base="${base%.*}"
    ${fc} -o "${base}.o" "${source_root}/${rel}"
    objects+=("${base}.o")
    libmvec_guard "${base}"
done

for line in "${extracts[@]}"; do
    read -r name rel first last <<< "${line}"
    total=$(wc -l < "${source_root}/${rel}")
    if (( first < 1 || last < first || last > total )); then
        echo "${rel}: line range ${first}-${last} is outside the file (${total} lines)" >&2
        exit 7
    fi
    sed -n "${first},${last}p" "${source_root}/${rel}" > "${name}"
done

gfortran -c -O0 ${free} -I "${build_dir}" "${script_dir}/oracle_io.F90"
objects+=(oracle_io.o)

gfortran -c -Ofast -ftree-vectorize \
    -o libmvec_positive_control.o "${script_dir}/libmvec_positive_control.F90"
nm -u libmvec_positive_control.o | sed 's/^ *//' | sort > undefined-control.txt
if ! grep -q '_ZGV' undefined-control.txt; then
    echo "libmvec positive control produced no _ZGV symbol; the -O0 guard is" \
         "vacuous on this toolchain" >&2
    exit 5
fi

drivers=()
if [[ ${#only[@]} -gt 0 ]]; then
    for name in "${only[@]}"; do
        drivers+=("${script_dir}/${name%.F90}.F90")
    done
else
    drivers=("${script_dir}"/gsl_run_*.F90)
fi
mkdir -p fixtures
for driver in "${drivers[@]}"; do
    name=$(basename "${driver}" .F90)
    gfortran -c -O0 -cpp -DRWORDSIZE=4 ${free} -I "${build_dir}" "${driver}"
    libmvec_guard "${name}"
    gfortran -o "${name}" "${objects[@]}" "${name}.o"
    out="fixtures/${name#gsl_run_}"
    rm -rf "${out}"
    mkdir -p "${out}"
    "./${name}" "${out}" > "${name}-stdout.txt"
done

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# kind_phys: ${kind}"
    echo "# -O0 libm symbols per object (THE REFERENCE):"
    for f in undefined-O0-*.txt; do
        echo "## ${f#undefined-O0-}"
        grep -E 'expf|powf|logf|log10f|atanf|sqrtf|cbrtf|erff|exp$|pow$|log$|_ZGV' "${f}" || true
    done
    echo "# positive control (plain expf loop at -Ofast):"
    grep -E 'expf|_ZGV' undefined-control.txt || true
} > libmvec-report.txt

gfortran --version | head -1 > compiler.txt
{
    for pin in "${pins[@]}"; do cat "${pin}"; done
    sha256sum "${script_dir}/build_gsl_smoke.sh" "${script_dir}"/gsl_run_*.F90 \
        "${script_dir}"/gsl-sources-*.list "${script_dir}"/gsl-extract-*.list 2>/dev/null || true
    sha256sum ./machine.F90
} > build-inputs-sha256.txt
