#!/usr/bin/env bash
# MYNN wrapper column oracle (see wrapper_columns.py).
#
#   build_columns.sh WRF_SOURCE_ROOT BUILD_DIR [FLAVOUR]
#
# Compiles the unmodified WRF v4.6.1 sources -- ccpp_kind_types.F,
# share/module_model_constants.F, module_bl_mynn_common.F, module_bl_mynn.F
# and module_bl_mynn_wrapper.F -- with no stub, links run_wrapper_columns,
# writes the column inputs, runs both mixing lengths for 12 steps and packs
# the fixtures.
#
# FLAVOUR "oracle" (default): -O0 -ffp-contract=off, the same arithmetic as
#   every other MYNN oracle here: scalar glibc libm calls, no contraction.
#   This is the 0 ULP reference.
# FLAVOUR "stock": WRF's own GNU flags from arch/configure.defaults
#   (-O2 -ftree-vectorize -funroll-loops), so the measured gap to a stock
#   wrf.exe build (vectorised loops may call libmvec) is on record too.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: build_columns.sh WRF_SOURCE_ROOT BUILD_DIR [oracle|stock]" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
flavour=${3:-oracle}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)

case "${flavour}" in
    oracle) flags=(-O0 -ffp-contract=off) ;;
    stock) flags=(-O2 -ftree-vectorize -funroll-loops) ;;
    *) echo "unknown flavour ${flavour}" >&2; exit 2 ;;
esac
common=(-cpp -ffree-form -ffree-line-length-none
        -DEM_CORE=1 -DWRF_CHEM=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8
        -DLWORDSIZE=4 "${flags[@]}")

mkdir -p "${build_dir}"
cd "${build_dir}"
gfortran -c "${common[@]}" "${source_root}/phys/ccpp_kind_types.F"
gfortran -c "${common[@]}" "${source_root}/share/module_model_constants.F"
gfortran -c "${common[@]}" "${source_root}/phys/module_bl_mynn_common.F"
gfortran -c "${common[@]}" "${source_root}/phys/module_bl_mynn.F"
gfortran -c "${common[@]}" "${source_root}/phys/module_bl_mynn_wrapper.F"
# The harness itself: stream I/O is raw words; no big-endian conversion.
gfortran -c -O0 -ffp-contract=off -ffree-form -ffree-line-length-none \
    "${script_dir}/run_wrapper_columns.F90"
gfortran -o run_wrapper_columns ccpp_kind_types.o module_model_constants.o \
    module_bl_mynn_common.o module_bl_mynn.o module_bl_mynn_wrapper.o \
    run_wrapper_columns.o

python3 "${script_dir}/wrapper_columns.py" gen columns-in.bin --nstep 12
variants=(base mixscalars spp)
for variant in "${variants[@]}"; do
    for ml in 1 2; do
        ./run_wrapper_columns columns-in.bin \
            "records-${variant}-ml${ml}.bin" "${ml}" "${variant}"
    done
done

{
    gfortran --version | sed -n 1p
    ldd --version 2>&1 | sed -n 1p
    echo "flavour ${flavour}: ${flags[*]}"
    for f in phys/ccpp_kind_types.F share/module_model_constants.F \
             phys/module_bl_mynn_common.F phys/module_bl_mynn.F \
             phys/module_bl_mynn_wrapper.F; do
        printf '%s  %s (LF)  %s (CRLF)\n' "$f" \
            "$(sha256sum < "${source_root}/$f" | cut -d' ' -f1)" \
            "$(sed 's/$/\r/' "${source_root}/$f" | sha256sum | cut -d' ' -f1)"
    done
    sha256sum "${script_dir}/run_wrapper_columns.F90" \
        "${script_dir}/wrapper_columns.py" columns-in.bin records-*.bin
} > provenance.txt
python3 - "${flavour}" <<'EOF'
import json, sys
lines = open("provenance.txt").read().splitlines()
json.dump({"flavour": sys.argv[1], "provenance": lines},
          open("provenance.json", "w"), indent=1)
EOF
for variant in "${variants[@]}"; do
    for ml in 1 2; do
        python3 "${script_dir}/wrapper_columns.py" pack \
            "records-${variant}-ml${ml}.bin" \
            "wrapper-columns-${flavour}-${variant}-ml${ml}.npz" \
            --mixlength "${ml}" --variant "${variant}" \
            --provenance provenance.json
    done
done
cat provenance.txt
