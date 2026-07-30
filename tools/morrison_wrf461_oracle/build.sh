#!/usr/bin/env bash
# Build and run a standalone Morrison oracle from byte-unmodified WRF v4.6.1.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
kind_source="${source_root}/phys/ccpp_kind_types.F"
constants_source="${source_root}/share/module_model_constants.F"
radar_source="${source_root}/phys/module_mp_radar.F"
morrison_source="${source_root}/phys/module_mp_morr_two_moment.F"

for source in "${kind_source}" "${constants_source}" "${radar_source}" \
              "${morrison_source}"; do
    test -f "${source}"
done

# The reference is the pinned source, not merely a file with the expected
# basename.  Refuse both the wrong revision and local edits to any compiled
# WRF source.
head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- \
        phys/ccpp_kind_types.F share/module_model_constants.F \
        phys/module_mp_radar.F phys/module_mp_morr_two_moment.F; then
    echo "a compiled WRF Morrison dependency differs from pinned HEAD" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

fflags=(-O0 -cpp -ffree-form -ffree-line-length-none)

# kind_phys is not named by Morrison: the module declares default REAL.
# Compile WRF's kind module anyway and have the runner assert that its own
# kind_phys equals kind(1.0), under WRF's -DRWORDSIZE=4 configuration.
gfortran -c "${fflags[@]}" -DRWORDSIZE=4 "${kind_source}"
gfortran -c "${fflags[@]}" "${script_dir}/stub_wrf_error.F90"
gfortran -c "${fflags[@]}" "${constants_source}"
gfortran -c "${fflags[@]}" -I "${build_dir}" "${radar_source}"
gfortran -c "${fflags[@]}" -DWRF_CHEM=0 -fallow-argument-mismatch \
    -I "${build_dir}" -o module_mp_morr_two_moment_O0.o "${morrison_source}"
gfortran -c "${fflags[@]}" -I "${build_dir}" \
    "${script_dir}/run_morrison.F90"
gfortran -o run_morrison \
    stub_wrf_error.o ccpp_kind_types.o module_model_constants.o \
    module_mp_radar.o module_mp_morr_two_moment_O0.o run_morrison.o

# The -O0 object linked into the oracle must stay on scalar libm.
nm -u module_mp_morr_two_moment_O0.o | sed 's/^ *//' | sort \
    > undefined-O0.txt
if grep -q '_ZGV' undefined-O0.txt; then
    echo "-O0 Morrison object pulled in a libmvec vector symbol:" >&2
    grep '_ZGV' undefined-O0.txt >&2
    exit 4
fi

# Evidence-only WRF-flags object: never linked into the reference executable.
gfortran -c -O2 -ftree-vectorize -funroll-loops -cpp \
    -ffree-form -ffree-line-length-none -DWRF_CHEM=0 \
    -fallow-argument-mismatch -I "${build_dir}" \
    -o module_mp_morr_two_moment_O2vec.o "${morrison_source}"
nm -u module_mp_morr_two_moment_O2vec.o | sed 's/^ *//' | sort \
    > undefined-O2vec.txt

# Positive control at the same optimization/vectorization flags.  This is the
# same EXP intrinsic in a loop the vectorizer can take.
gfortran -c -O2 -ftree-vectorize -funroll-loops \
    -o libmvec_positive_control.o \
    "${script_dir}/libmvec_positive_control.F90"
nm -u libmvec_positive_control.o | sed 's/^ *//' | sort \
    > undefined-control.txt
if ! grep -q '_ZGV' undefined-control.txt; then
    echo "libmvec positive control produced no _ZGV symbol" >&2
    cat undefined-control.txt >&2
    exit 5
fi

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 Morrison object (linked reference):"
    grep -E 'expf|powf|logf|sqrtf|_ZGV' undefined-O0.txt || true
    echo "# -O2 -ftree-vectorize -funroll-loops (WRF physics flags):"
    grep -E 'expf|powf|logf|sqrtf|_ZGV' undefined-O2vec.txt || true
    echo "# positive control (same EXP, vectorisable loop, WRF flags):"
    grep -E 'expf|_ZGV' undefined-control.txt || true
} > libmvec-report.txt
cat libmvec-report.txt

./run_morrison morrison-levels.csv morrison-surface.csv

sha256sum "${kind_source}" "${constants_source}" "${radar_source}" \
    "${morrison_source}" "${script_dir}/stub_wrf_error.F90" \
    "${script_dir}/libmvec_positive_control.F90" \
    "${script_dir}/run_morrison.F90" \
    morrison-levels.csv morrison-surface.csv libmvec-report.txt \
    > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "${pinned_commit}" > wrf-commit.txt
echo "Morrison oracle built in ${build_dir}"
