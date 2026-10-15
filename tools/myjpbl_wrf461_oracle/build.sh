#!/usr/bin/env bash
# Build and run the MYJ PBL column oracle against the byte-unmodified WRF
# v4.6.1 phys/module_bl_myjpbl.F.
#
#   bash tools/myjpbl_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR [OUT.npz]
#
# WRF_SOURCE_ROOT must be a wrf-model/WRF checkout at the v4.6.1 commit with
# share/module_model_constants.F and phys/module_bl_myjpbl.F unmodified;
# the script refuses otherwise, so the fixture cannot come from an edited
# tree.  module_bl_myjpbl.F USEs only MODULE_MODEL_CONSTANTS and CALLs only
# its own contained routines, so no stub framework is needed: `nm -u` on the
# object is the receipt (libm + libgfortran only).
#
# Reference build: gfortran -O0, no vectoriser.  WRF's own configure.wrf
# compiles phys/ at -O2 -ftree-vectorize, where gfortran may call glibc's
# vector math (_ZGV*) with its 4-ULP contract.  The script FAILS if any _ZGV
# symbol appears in the -O0 object, and a positive control proves the grep can
# see one on this toolchain at all.  It also records which symbols WRF's own
# -O2 -ftree-vectorize build of this module would pull in.
#
# Output: BUILD_DIR/out_<case>.{bin,txt} per case, and OUT.npz (default
# BUILD_DIR/myjpbl-wrf461.npz), the fixture tests/test_myjpbl_wrf461_parity.py
# reads.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [OUT.npz]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out_npz=$(realpath -m "${3:-${build_dir}/myjpbl-wrf461.npz}")
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
const_rel="share/module_model_constants.F"
myj_rel="phys/module_bl_myjpbl.F"

head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- "${const_rel}" "${myj_rel}"; then
    echo "${const_rel} or ${myj_rel} differs from the pinned commit" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

defines="-DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
free="-ffree-form -ffree-line-length-none"

# --- the reference build: -O0, no vectoriser --------------------------------
gfortran -c -O0 -cpp ${defines} ${free} -o module_model_constants.o \
    "${source_root}/${const_rel}"
gfortran -c -O0 -cpp ${defines} ${free} -I "${build_dir}" \
    -o module_bl_myjpbl_O0.o "${source_root}/${myj_rel}"
gfortran -c -O0 ${free} -I "${build_dir}" -o run_myjpbl.o \
    "${script_dir}/run_myjpbl.F90"
gfortran -o run_myjpbl module_model_constants.o module_bl_myjpbl_O0.o run_myjpbl.o

# --- receipts ---------------------------------------------------------------
nm -u module_bl_myjpbl_O0.o | sed 's/^ *//' | sort > undefined-O0.txt
if grep -q '_ZGV' undefined-O0.txt; then
    echo "-O0 object pulled in a libmvec vector symbol:" >&2
    grep '_ZGV' undefined-O0.txt >&2
    exit 4
fi
if grep -Ev '^U (_gfortran_|expf$|powf$|sqrtf$|malloc$|free$|memset$|memcpy$)' \
        undefined-O0.txt | grep -q .; then
    echo "module_bl_myjpbl_O0.o calls something beyond expf/powf/sqrtf," \
         "malloc/free and libgfortran; review undefined-O0.txt" >&2
    cat undefined-O0.txt >&2
    exit 4
fi
gfortran -c -O2 -ftree-vectorize -funroll-loops -cpp ${defines} ${free} \
    -I "${build_dir}" -o module_bl_myjpbl_O2vec.o "${source_root}/${myj_rel}"
nm -u module_bl_myjpbl_O2vec.o | sed 's/^ *//' | sort > undefined-O2vec.txt
gfortran -c -Ofast -ftree-vectorize -o libmvec_positive_control.o \
    "${script_dir}/libmvec_positive_control.F90"
nm -u libmvec_positive_control.o | sed 's/^ *//' | sort > undefined-control.txt
if ! grep -q '_ZGV' undefined-control.txt; then
    echo "libmvec positive control produced no _ZGV symbol; the guard is" \
         "vacuous on this toolchain" >&2
    exit 5
fi
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 libm symbols (THE REFERENCE):"
    grep -E 'expf|powf|logf|sqrtf|_ZGV' undefined-O0.txt || true
    echo "# -O2 -ftree-vectorize -funroll-loops (WRF's own phys/ setting):"
    grep -E 'expf|powf|logf|sqrtf|_ZGV' undefined-O2vec.txt || true
    echo "# positive control (expf loop at -Ofast), proves the grep works:"
    grep -E 'expf|_ZGV' undefined-control.txt || true
} > libmvec-report.txt
cat libmvec-report.txt

# --- inputs and run ---------------------------------------------------------
python3 "${script_dir}/make_columns.py" "${build_dir}"
for case in $(cat cases.txt); do
    ./run_myjpbl "case_${case}.bin" "out_${case}.bin" "out_${case}.txt"
done
python3 "${script_dir}/pack_fixture.py" "${build_dir}" "${out_npz}"

# --- receipt: WRF's own -O2 phys/ flags write the same words ----------------
# (x86-64 baseline has no FMA, and nothing in MYJPBL is vectorised into a
# libmvec call, so the -O0 reference should be the shipped arithmetic.)
gfortran -o run_myjpbl_O2 module_model_constants.o module_bl_myjpbl_O2vec.o \
    run_myjpbl.o
for case in $(cat cases.txt); do
    ./run_myjpbl_O2 "case_${case}.bin" "o2_${case}.bin" "o2_${case}.txt" > /dev/null
    if ! cmp -s "out_${case}.bin" "o2_${case}.bin"; then
        echo "WRF -O2 -ftree-vectorize -funroll-loops differs from -O0 on ${case}" >&2
        exit 6
    fi
done
echo "receipt: -O2 -ftree-vectorize -funroll-loops outputs byte-identical to -O0" \
    | tee o2-equality.txt

# --- receipt: which lines of module_bl_myjpbl.F the fixture executes --------
mkdir -p coverage
( cd coverage
  gfortran -c -O0 --coverage -cpp ${defines} ${free} -I "${build_dir}" \
      -o module_bl_myjpbl.o "${source_root}/${myj_rel}"
  gfortran --coverage -o run_myjpbl_cov ../module_model_constants.o \
      module_bl_myjpbl.o ../run_myjpbl.o
  for case in $(cat ../cases.txt); do
      ./run_myjpbl_cov "../case_${case}.bin" "cov_${case}.bin" "cov_${case}.txt" > /dev/null
      cmp -s "../out_${case}.bin" "cov_${case}.bin" \
          || { echo "coverage build changed the words on ${case}" >&2; exit 7; }
  done
  gcov -o . module_bl_myjpbl.o > gcov-summary.txt 2>&1 || true
)
python3 - "${build_dir}/coverage" > coverage-myjpbl.txt <<'PY'
import pathlib, re, sys
d = pathlib.Path(sys.argv[1])
g = next(d.glob("module_bl_myjpbl.F.gcov"))
never, ran = [], 0
for line in g.read_text().splitlines():
    m = re.match(r"\s*([^:]+):\s*(\d+):(.*)", line)
    if not m:
        continue
    count, num, text = m.group(1).strip(), int(m.group(2)), m.group(3)
    if count == "#####":
        never.append((num, text.strip()))
    elif count not in ("-",):
        ran += 1
print(f"executable lines executed: {ran}; never executed: {len(never)}")
for num, text in never:
    print(f"  {num:5d}: {text}")
PY
cat coverage-myjpbl.txt

sha256sum "${source_root}/${const_rel}" "${source_root}/${myj_rel}" \
    "${script_dir}/run_myjpbl.F90" "${script_dir}/make_columns.py" \
    "${script_dir}/pack_fixture.py" case_*.bin out_*.bin "${out_npz}" \
    > oracle-sha256sums.txt
gfortran --version | sed -n 1p > compiler.txt
ldd --version 2>&1 | sed -n 1p >> compiler.txt
echo "oracle built in ${build_dir}; fixture ${out_npz}"
