#!/usr/bin/env bash
# Build and run the chem WRF v4.7.1 column oracles.
#
#   bash tools/chem_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR [run_NAME ...]
#
# WRF_SOURCE_ROOT is a directory laid out like the program's pinned reference
# tree: WRF v4.7.1 at commit
# f52c197ed39d12e087d02c50f412d90d418f6186 (chem/, dyn_em/, phys/, share/,
# Registry/, ...), MYNN-EDMF under phys/MYNN-EDMF at 90f36c25, and NOAA GSL's
# public ccpp-physics under gsl-ccpp-physics/.  A full clone of the tag works
# too.  The tree is NOT checked by commit: every file this script compiles or
# extracts from must match its SHA-256 in SOURCES.sha256 (a byte copy of
# wrf-src/SHA256SUMS, manifest sha256 9bf0661e...) or in a lane's
# SOURCES-<lane>.sha256, and a mismatch stops the build before anything is
# compiled.
#
# WHY THERE IS NO CORE SET.  Almost every chem/ module USEs the
# Registry-generated module_state_description and module_configure, which do
# not exist outside a configured WRF build.  So a driver takes WRF's code in
# one of two ways, and a lane adds either by adding files, never by editing
# this script:
#
#   sources-<lane>.list   whole files compiled in listed order (paths
#                         relative to WRF_SOURCE_ROOT; for the few modules
#                         with no generated dependency, e.g.
#                         share/module_model_constants.F).
#   extract-<lane>.list   one line per wrapper module:
#                             <module> <prelude|-> <rel_path> <routine>[,<routine>...]
#                         Each routine (SUBROUTINE or FUNCTION, matched
#                         case-insensitively from its header line to its END
#                         line) is copied BYTE FOR BYTE from <rel_path> into
#                         <module>.F90 after the text of <prelude> (a file in
#                         this directory holding the USE lines and the
#                         Registry parameters the routine reads, e.g. the
#                         p_<species> indices).  The prelude is the only
#                         non-WRF text; every WRF statement compiled is the
#                         pinned source's own.
#
# WHAT IT RUNS.  Every run_*.F90 here (or only the ones named after BUILD_DIR),
# each with BUILD_DIR/fixtures/<name> as its one argument (oracle_io's root).
# Copy the fixture directory to tests/data/oracles/chem/<lane>/ to publish it.
#
# Flags are WRF's -r4 build at -O0: -DRWORDSIZE=4 and the EM defines, no fast
# math.  libmvec: WRF compiles chem/ at -O2 -ftree-vectorize, where gfortran
# can swap scalar expf/powf/logf for glibc's 4-ULP vector forms.  The reference
# is -O0 and the script FAILS if any _ZGV* symbol appears in an -O0 object; a
# positive control proves the grep can see one on this toolchain at all.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [run_NAME ...]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
shift 2
only=("$@")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

shopt -s nullglob

# A path two lanes both need (share/module_model_constants.F: the PM sums
# and the plume model) compiles once, at its first position: a second copy
# of one module's object would be a duplicate-symbol link error.
declare -A listed=()
extra=()
for list in "${script_dir}"/sources-*.list; do
    while IFS= read -r rel; do
        rel="${rel%%#*}"; rel="$(echo "${rel}" | xargs)"
        if [[ -n "${rel}" && -z "${listed[${rel}]:-}" ]]; then
            listed["${rel}"]=1
            extra+=("${rel}")
        fi
    done < "${list}"
done

extracts=()
for list in "${script_dir}"/extract-*.list; do
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
for rel in "${extra[@]}"; do
    check "${rel}"
done
for line in "${extracts[@]}"; do
    read -r _module _prelude rel _routines <<< "${line}"
    check "${rel}"
done

mkdir -p "${build_dir}"
cd "${build_dir}"

defines="-DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4"
defines="${defines} -DDWORDSIZE=8 -DLWORDSIZE=4 -DWRF_CHEM=1"
free="-ffree-form -ffree-line-length-none"
fc="gfortran -c -O0 -cpp ${defines} ${free} -fallow-argument-mismatch -I ${build_dir}"

objects=()
gfortran -c -O0 -cpp ${free} -fallow-argument-mismatch "${script_dir}/stub_wrf.F90"
objects+=(stub_wrf.o)
# A lane's service stub (stub_<lane>.F90): the Registry-generated modules a
# byte-for-byte extracted routine USEs inside its own body -- WRF-Chem's
# plumerise_driver opens with USE module_configure and USE
# module_state_description (chem/module_plumerise1.F:33-37), which no prelude
# can supply because the USE is the routine's own text.  Compiled before any
# extract, in name order, like the lists.
for stub in "${script_dir}"/stub_*.F90; do
    base=$(basename "${stub}" .F90)
    [[ "${base}" == "stub_wrf" ]] && continue
    gfortran -c -O0 -cpp ${free} -fallow-argument-mismatch -o "${base}.o" "${stub}"
    objects+=("${base}.o")
done

libmvec_guard() {
    local base="$1"
    nm -u "${base}.o" | sed 's/^ *//' | sort > "undefined-O0-${base}.txt"
    if grep -q '_ZGV' "undefined-O0-${base}.txt"; then
        echo "-O0 object ${base}.o pulled in a libmvec vector symbol:" >&2
        grep '_ZGV' "undefined-O0-${base}.txt" >&2
        exit 4
    fi
}

for rel in "${extra[@]}"; do
    base=$(basename "${rel}")
    base="${base%.*}"
    ${fc} -o "${base}.o" "${source_root}/${rel}"
    objects+=("${base}.o")
    libmvec_guard "${base}"
done

# Wrapper modules: prelude text, then each routine byte for byte.
extract_routine() {
    local path="$1" name="$2"
    awk -v want="$(echo "${name}" | tr 'A-Z' 'a-z')" '
        # WRF chem/ files carry CRLF line ends: norm() drops the CR before
        # every comparison, and the copied text keeps it (gfortran reads it).
        function norm(s) { s = tolower(s); gsub(/\r/, "", s); gsub(/^[ \t]+/, "", s); return s }
        {
            l = norm($0)
            if (!inside) {
                if (l ~ /^(recursive[ \t]+)?(pure[ \t]+)?(elemental[ \t]+)?((real|integer|logical|double precision)[^ \t]*[ \t]+)?(subroutine|function)[ \t]+/) {
                    t = l
                    sub(/^.*(subroutine|function)[ \t]+/, "", t)
                    sub(/[ \t(].*$/, "", t)
                    if (t == want) { inside = 1 }
                }
            }
            if (inside) {
                print $0
                if (l ~ /^end[ \t]*(subroutine|function)/) {
                    t = l
                    sub(/^end[ \t]*(subroutine|function)[ \t]*/, "", t)
                    sub(/[ \t!].*$/, "", t)
                    if (t == want || t == "") { found = 1; exit }
                }
            }
        }
        END { if (!found) exit 7 }' "${path}"
}
for line in "${extracts[@]}"; do
    read -r module prelude rel routines <<< "${line}"
    out="${module}.F90"
    {
        echo "! GENERATED by tools/chem_wrf471_oracle/build.sh from ${rel}."
        echo "! Everything after 'contains' is the pinned source's own text."
        echo "module ${module}"
        if [[ "${prelude}" != "-" ]]; then
            cat "${script_dir}/${prelude}"
        fi
        echo "contains"
    } > "${out}"
    IFS=',' read -r -a names <<< "${routines}"
    for name in "${names[@]}"; do
        if ! extract_routine "${source_root}/${rel}" "${name}" >> "${out}"; then
            echo "${rel}: routine ${name} not found (or never ended)" >&2
            exit 7
        fi
    done
    echo "end module ${module}" >> "${out}"
    ${fc} -o "${module}.o" "${out}"
    objects+=("${module}.o")
    libmvec_guard "${module}"
done

gfortran -c -O0 ${free} -I "${build_dir}" "${script_dir}/oracle_io.F90"
objects+=(oracle_io.o)

# Positive control: the _ZGV grep must be able to fire on this toolchain.
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
    drivers=("${script_dir}"/run_*.F90)
fi
mkdir -p fixtures
for driver in "${drivers[@]}"; do
    name=$(basename "${driver}" .F90)
    gfortran -c -O0 -cpp ${defines} ${free} -I "${build_dir}" "${driver}"
    libmvec_guard "${name}"
    gfortran -o "${name}" "${objects[@]}" "${name}.o"
    rm -rf "fixtures/${name#run_}"
    mkdir -p "fixtures/${name#run_}"
    "./${name}" "fixtures/${name#run_}" > "${name}-stdout.txt"
done

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 libm symbols per object (THE REFERENCE):"
    for f in undefined-O0-*.txt; do
        echo "## ${f#undefined-O0-}"
        grep -E 'expf|powf|logf|log10f|atanf|sqrtf|cbrtf|erff|_ZGV' "${f}" || true
    done
    echo "# positive control (plain expf loop at -Ofast):"
    grep -E 'expf|_ZGV' undefined-control.txt || true
} > libmvec-report.txt

gfortran --version | head -1 > compiler.txt
{
    for pin in "${pins[@]}"; do cat "${pin}"; done
    # Paths relative to the harness directory and the build directory, so a
    # published receipt names no machine.
    (cd "${script_dir}" && sha256sum ./*.F90 ./*.inc ./build.sh ./*.list \
        2>/dev/null | sed 's#  \./#  tools/chem_wrf471_oracle/#') || true
    sha256sum ./*.F90 | sed 's#  \./#  build/#'
} > oracle-sha256sums.txt
echo "chem oracle built in ${build_dir}; fixtures in ${build_dir}/fixtures"
