#!/usr/bin/env bash
# Build and run the Grell-Freitas REGIME column oracle against the
# byte-unmodified WRF v4.6.1 phys/module_cu_gf_*.F.
#
#   bash build_regime.sh WRF_SOURCE_ROOT BUILD_DIR [PYTHON]
#
# The columns come from gf_regime_columns.py (realistic soundings across
# regimes, a clos_choice 1..16 sweep and named edge cases); the WRF side is
# run_gf_columns.F90, one real GFDRV call per column.  Same pin, same -O0
# reference build and same libmvec guard as build.sh.
#
# Variants, all from the SAME WRF objects:
#   stock     the system libm, as WRF itself links.  gamma() is the C
#             library tgammaf, which is not correctly rounded -- the
#             documented deliberate divergence (docs/gf_gamma_known_delta.md).
#   crgamma   identical, except tgammaf resolves to gf_crgamma_tgammaf.c,
#             the correctly rounded gamma WOOF ships.  This is the bitwise
#             target for the kernel's SHIPPED path (fzu computed).
#   nan       crgamma objects rebuilt with -finit-real=snan/-finit-integer:
#             any read of an uninitialised local changes the answer.
#   bounds    stock objects rebuilt with -fcheck=bounds: an out-of-bounds
#             read in WRF aborts and is localised to its column.
# Each runs in two vertical tile conventions: kte_mode 0 (kte = nz, what
# WRF's solver passes, KTE=min(k_end,kde-1), and the kernel's nz -- the
# oracle of record) and kte_mode 1 (kte = kde = nz + 1 with NaN inputs on the
# extra level, a probe that any read above ktf would show).
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: build_regime.sh WRF_SOURCE_ROOT BUILD_DIR [PYTHON]" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
python=${3:-python3}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)

declare -A pinned=(
  [module_cu_gf_wrfdrv]="0cead99c913b6a74f3a492aea1e1c7ef6d34f95402574336022d1a6c716a4382"
  [module_cu_gf_deep]="798ad7271963757b88be36ce01e24222f7af045dc0692908d6e7c270d6b4ea11"
  [module_cu_gf_sh]="0dd1ed7c620b39e13f7ef77dc4ff1a269fa1f8d746492d360e203df0215b529a"
  [module_gfs_physcons]="2b8ef99663ba62748b57c7002f5e5b8c9338f2a825c1d32c113bd4e3bff2eb52"
  [module_gfs_machine]="300ee4f75663e89d34c8a9b8733cc07a5dd48c3d973da019d31cb76524babe25"
)
for name in "${!pinned[@]}"; do
    src="${source_root}/phys/${name}.F"
    actual=$(sha256sum "${src}" | awk '{print $1}')
    if [[ "${actual}" != "${pinned[$name]}" ]]; then
        echo "phys/${name}.F is ${actual}, not the pinned ${pinned[$name]}" >&2
        exit 3
    fi
done

mkdir -p "${build_dir}"
cd "${build_dir}"

# --- the columns ------------------------------------------------------------
"${python}" "${script_dir}/gf_regime_columns.py" "${build_dir}"

# --- WRF objects, one directory per flag set --------------------------------
mods="module_gfs_machine module_gfs_physcons module_cu_gf_deep module_cu_gf_sh module_cu_gf_wrfdrv"
build_objs() {   # dir, extra flags...
    local dir=$1; shift
    mkdir -p "${dir}"
    for name in ${mods}; do
        gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none "$@" \
            -J "${dir}" -o "${dir}/${name}.o" "${source_root}/phys/${name}.F"
    done
    gfortran -c -O0 -ffree-form -ffree-line-length-none "$@" -I "${dir}" \
        -J "${dir}" -o "${dir}/run_gf_columns.o" "${script_dir}/run_gf_columns.F90"
}
objs() { local d=$1; for n in ${mods}; do printf '%s ' "${d}/${n}.o"; done; printf '%s' "${d}/run_gf_columns.o"; }

build_objs ref
build_objs nanobj -finit-real=snan -finit-integer=-999999
build_objs bndobj -fcheck=bounds

gcc -O2 -c -o gf_crgamma_tgammaf.o "${script_dir}/gf_crgamma_tgammaf.c"

# shellcheck disable=SC2046
gfortran -o run_stock $(objs ref)
# shellcheck disable=SC2046
gfortran -o run_crgamma $(objs ref) gf_crgamma_tgammaf.o -lquadmath
# shellcheck disable=SC2046
gfortran -o run_nan $(objs nanobj) gf_crgamma_tgammaf.o -lquadmath
# shellcheck disable=SC2046
gfortran -o run_bounds $(objs bndobj)

# --- receipts on the binding and the build -----------------------------------
: > undefined-O0.txt
for name in module_cu_gf_deep module_cu_gf_sh module_cu_gf_wrfdrv; do
    nm -u "ref/${name}.o" | sed 's/^ *//' | sed "s|^|${name} |" >> undefined-O0.txt
done
if grep -q '_ZGV' undefined-O0.txt; then
    echo "-O0 object pulled in a libmvec vector symbol" >&2
    exit 4
fi
grep -q 'module_cu_gf_deep U tgammaf' undefined-O0.txt || {
    echo "module_cu_gf_deep.o does not call tgammaf; the interposition is moot" >&2
    exit 5; }
nm run_crgamma | grep -E ' T tgammaf$' > binding.txt || {
    echo "run_crgamma does not define tgammaf itself" >&2; exit 6; }
nm run_stock | grep -E ' U tgammaf' >> binding.txt || {
    echo "run_stock does not take tgammaf from the shared libm" >&2; exit 6; }
cat binding.txt

gcc -O2 -o gf_crgamma_check "${script_dir}/gf_crgamma_check.c" \
    gf_crgamma_tgammaf.o -lquadmath
./gf_crgamma_check "${repo_root}/tests/data/oracles/gf/gf-crgamma-tgammaf.csv" \
    2>/dev/null | tee crgamma-check.txt

# --- run ----------------------------------------------------------------------
for v in stock crgamma nan; do
    for m in 0 1; do
        ./run_${v} gf-regime-inputs.bin "gf-regime-wrf-${v}-k${m}.bin" ${m} \
            2> "run-${v}-k${m}.err"
    done
done

# bounds: run whole, and if WRF aborts, localise the abort to columns
: > bounds.txt
ncol=$("${python}" -c "import json;print(json.load(open('gf-regime-manifest.json'))['ncol'])")
for m in 0 1; do
    if ./run_bounds gf-regime-inputs.bin "gf-regime-wrf-bounds-k${m}.bin" ${m} \
            > /dev/null 2> "run-bounds-k${m}.err"; then
        echo "kte_mode ${m}: no out-of-bounds access on any of ${ncol} columns" >> bounds.txt
    else
        echo "kte_mode ${m}: run aborted; localising" >> bounds.txt
        for ((c = 1; c <= ncol; c++)); do
            if ! ./run_bounds gf-regime-inputs.bin /dev/null ${m} ${c} \
                    > /dev/null 2> "bounds-col.err"; then
                echo "  column $((c - 1)) (0-based): $(grep -m1 -E 'Fortran runtime error|At line' bounds-col.err | tr -d '\n')" >> bounds.txt
            fi
        done
        rm -f bounds-col.err
    fi
done
cat bounds.txt

# --- consistency between the variants --------------------------------------
same() { cmp -s -i 16 "$1" "$2" && echo same || echo DIFFERENT; }
{
    echo "kte_mode 0 vs 1, stock (expected: HBOT's REAL(KTE) initial value only): $(same gf-regime-wrf-stock-k0.bin gf-regime-wrf-stock-k1.bin)"
    echo "kte_mode 0 vs 1, crgamma (expected: HBOT's REAL(KTE) initial value only): $(same gf-regime-wrf-crgamma-k0.bin gf-regime-wrf-crgamma-k1.bin)"
    echo "uninitialised-read probe (snan init vs default), kte_mode 0: $(same gf-regime-wrf-crgamma-k0.bin gf-regime-wrf-nan-k0.bin)"
    echo "uninitialised-read probe (snan init vs default), kte_mode 1: $(same gf-regime-wrf-crgamma-k1.bin gf-regime-wrf-nan-k1.bin)"
    echo "stock vs crgamma (the gamma divergence must show): $(same gf-regime-wrf-stock-k0.bin gf-regime-wrf-crgamma-k0.bin)"
} | tee regime-consistency.txt
# The mode-1 probe must differ from mode 0 ONLY in HBOT's initial value
# (REAL(KTE): nz versus nz + 1) and only on columns GFDRV left untouched.
"${python}" - "${script_dir}" <<'PYEOF' | tee -a regime-consistency.txt
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
import gf_regime_columns as g
bad = 0
for v in ("stock", "crgamma"):
    a = g.read_outputs(Path(f"gf-regime-wrf-{v}-k0.bin"))
    b = g.read_outputs(Path(f"gf-regime-wrf-{v}-k1.bin"))
    nz = a["nz"]
    lev_same = np.array_equal(a["lev"].view(np.uint32), b["lev"].view(np.uint32))
    isc_same = np.array_equal(a["isc"], b["isc"])
    ih = g.OUT_SCA.index("hbot")
    keep = [j for j in range(len(g.OUT_SCA)) if j != ih]
    sca_same = np.array_equal(a["sca"][:, keep].view(np.uint32),
                              b["sca"][:, keep].view(np.uint32))
    ha, hb = a["sca"][:, ih], b["sca"][:, ih]
    moved = ha != hb
    only_init = bool(np.all((ha[moved] == nz) & (hb[moved] == nz + 1)))
    ok = lev_same and isc_same and sca_same and only_init
    bad += not ok
    print(f"{v}: kte_mode 1 differs from 0 only in HBOT's initial value on "
          f"{int(moved.sum())} untouched columns: {ok}")
sys.exit(1 if bad else 0)
PYEOF

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# gcc:      $(gcc --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
} > regime-compiler.txt

sha256sum \
    "${source_root}/phys/module_cu_gf_wrfdrv.F" \
    "${source_root}/phys/module_cu_gf_deep.F" \
    "${source_root}/phys/module_cu_gf_sh.F" \
    "${source_root}/phys/module_gfs_physcons.F" \
    "${source_root}/phys/module_gfs_machine.F" \
    "${script_dir}/gf_regime_columns.py" \
    "${script_dir}/run_gf_columns.F90" \
    "${script_dir}/gf_crgamma_tgammaf.c" \
    "${script_dir}/gf_crgamma_check.c" \
    gf-regime-inputs.bin gf-regime-manifest.json \
    gf-regime-wrf-stock-k0.bin gf-regime-wrf-stock-k1.bin \
    gf-regime-wrf-crgamma-k0.bin gf-regime-wrf-crgamma-k1.bin \
  | sed -e "s#${source_root}#<wrf-4.6.1-checkout>#" \
        -e "s#${repo_root}#<oracle-workspace>#" \
  > regime-sha256sums.txt
echo "regime oracle built in ${build_dir}"
bash "${script_dir}/build_corrected.sh" "${source_root}" "${build_dir}" "${python}"
