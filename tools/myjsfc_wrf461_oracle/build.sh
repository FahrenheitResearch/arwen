#!/usr/bin/env bash
# Build and run the Eta similarity surface-layer column oracle against the
# byte-unmodified WRF v4.6.1 phys/module_sf_myjsfc.F (sf_sfclay_physics=2).
#
#   bash tools/myjsfc_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR [OUT.npz]
#
# WRF_SOURCE_ROOT must be a wrf-model/WRF checkout at the v4.6.1 commit with
# share/module_model_constants.F and phys/module_sf_myjsfc.F unmodified; the
# script refuses otherwise.  module_sf_myjsfc.F USEs MODULE_MODEL_CONSTANTS
# and, without DM_PARALLEL, nothing else, so no stub framework is needed.
#
# Reference build: gfortran -O0.  WRF's own phys/ flags are
# -O2 -ftree-vectorize -funroll-loops; that build is run too and must write
# the same bytes (o2-equality.txt), and neither object may call libmvec.
# Output: BUILD_DIR/out_<set>.bin and OUT.npz (default
# BUILD_DIR/myjsfc-wrf461.npz), the fixture tests/test_myjsfc_wrf461_parity.py
# and compare_myjsfc.py read; coverage-myjsfc.txt lists the executable lines
# of module_sf_myjsfc.F the fixture never reached.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [OUT.npz]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out_npz=$(realpath -m "${3:-${build_dir}/myjsfc-wrf461.npz}")
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
const_rel="share/module_model_constants.F"
sfc_rel="phys/module_sf_myjsfc.F"

head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- "${const_rel}" "${sfc_rel}"; then
    echo "${const_rel} or ${sfc_rel} differs from the pinned commit" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

defines="-DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
free="-ffree-form -ffree-line-length-none"
wrfflags="-O2 -ftree-vectorize -funroll-loops"

# --- the reference build: -O0 ------------------------------------------------
gfortran -c -O0 -cpp ${defines} ${free} -o module_model_constants.o \
    "${source_root}/${const_rel}"
gfortran -c -O0 -cpp ${defines} ${free} -I "${build_dir}" \
    -o module_sf_myjsfc_O0.o "${source_root}/${sfc_rel}"
gfortran -c -O0 ${free} -I "${build_dir}" -o run_myjsfc.o \
    "${script_dir}/run_myjsfc.F90"
gfortran -o run_myjsfc module_model_constants.o module_sf_myjsfc_O0.o run_myjsfc.o

# --- WRF's own phys/ flags ----------------------------------------------------
gfortran -c ${wrfflags} -cpp ${defines} ${free} -I "${build_dir}" \
    -o module_sf_myjsfc_O2vec.o "${source_root}/${sfc_rel}"
gfortran -o run_myjsfc_O2 module_model_constants.o module_sf_myjsfc_O2vec.o \
    run_myjsfc.o

# --- receipts: what the objects call ----------------------------------------
nm -u module_sf_myjsfc_O0.o | sed 's/^ *//' | sort > undefined-O0.txt
nm -u module_sf_myjsfc_O2vec.o | sed 's/^ *//' | sort > undefined-O2vec.txt
for f in undefined-O0.txt undefined-O2vec.txt; do
    if grep -q '_ZGV' "${f}"; then
        echo "${f}: a libmvec vector symbol is called" >&2
        grep '_ZGV' "${f}" >&2
        exit 4
    fi
done
objdump -d module_sf_myjsfc_O0.o module_sf_myjsfc_O2vec.o \
    | grep -cE 'vfmadd|vfmsub|vfnmadd|vfnmsub' > fma-count.txt || true
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# cpu:      $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
    echo "# WRF:      ${head_commit} (clean: ${sfc_rel}, ${const_rel})"
    echo "# -O0 undefined symbols (THE REFERENCE):"
    sed 's/^/#   /' undefined-O0.txt
    echo "# ${wrfflags} undefined symbols:"
    sed 's/^/#   /' undefined-O2vec.txt
    echo "# FMA instructions in either object: $(cat fma-count.txt)"
} > compiler.txt
cat compiler.txt

# --- inputs and runs ----------------------------------------------------------
python3 "${script_dir}/make_cases.py" "${build_dir}"
for set in $(cat cases.txt); do
    ./run_myjsfc "case_${set}.bin" "out_${set}.bin"
    ./run_myjsfc_O2 "case_${set}.bin" "o2_${set}.bin"
    if ! cmp -s "out_${set}.bin" "o2_${set}.bin"; then
        echo "WRF ${wrfflags} differs from -O0 on ${set}" >&2
        exit 6
    fi
done
echo "receipt: ${wrfflags} outputs byte-identical to -O0 on every set" \
    | tee o2-equality.txt
python3 "${script_dir}/pack_fixture.py" "${build_dir}" "${out_npz}"
sha256sum case_*.bin out_*.bin "${out_npz}" > oracle-sha256sums.txt

# --- receipt: which lines of module_sf_myjsfc.F the fixture executes --------
mkdir -p coverage
( cd coverage
  gfortran -c -O0 --coverage -cpp ${defines} ${free} -I "${build_dir}" \
      -o module_sf_myjsfc.o "${source_root}/${sfc_rel}"
  gfortran --coverage -o run_myjsfc_cov ../module_model_constants.o \
      module_sf_myjsfc.o ../run_myjsfc.o
  for set in $(cat ../cases.txt); do
      ./run_myjsfc_cov "../case_${set}.bin" "cov_${set}.bin"
      cmp -s "../out_${set}.bin" "cov_${set}.bin" \
          || { echo "coverage build changed the words on ${set}" >&2; exit 7; }
  done
  gcov -o . module_sf_myjsfc.o > gcov-summary.txt 2>&1 || true
)
python3 - "${build_dir}/coverage" > coverage-myjsfc.txt <<'PY'
import pathlib, sys
d = pathlib.Path(sys.argv[1])
g = next(d.glob("module_sf_myjsfc.F.gcov"))
never, ran = [], 0
for line in g.read_text().splitlines():
    count, lineno, text = (part.strip() for part in line.split(":", 2))
    if count == "-" or lineno == "0":
        continue
    if count.startswith("#####"):
        never.append(f"{lineno}: {text}")
    else:
        ran += 1
print(f"executable lines reached: {ran}; never reached: {len(never)}")
for row in never:
    print("  never", row)
PY
cat coverage-myjsfc.txt
