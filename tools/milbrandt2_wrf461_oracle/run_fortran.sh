#!/usr/bin/env bash
# CPU half of the mp=9 oracle: build, generate the four input sets, run WRF.
#
#   run_fortran.sh WRF_SOURCE_ROOT WORKDIR [WRF_BUILT_TREE]
#
# Writes WORKDIR/in_<set>.bin and WORKDIR/out_<set>_O2.bin for the sets
# dt20, dt60 (the named-regime and edge columns), stress50 and stress80
# (randomised columns at 50 and 80 levels).  With WRF_BUILT_TREE the runner
# linked against that build's own object must return the same bytes.
set -euo pipefail
src_root=$1
work=$(realpath -m "$2")
built_tree=${3:-}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "${work}"
"${here}/build.sh" "${src_root}" "${work}/build" ${built_tree:+"${built_tree}"}
cd "${work}"
python "${here}/gen_columns.py" --out in_dt20.bin --dt 20 --nsteps 10
python "${here}/gen_columns.py" --out in_dt60.bin --dt 60 --nsteps 6
python "${here}/gen_columns.py" --set stress --nz 50 --dt 10 --nsteps 5 --out in_stress50.bin
python "${here}/gen_columns.py" --set stress --nz 80 --dt 30 --nsteps 8 --out in_stress80.bin
for set_name in dt20 dt60 stress50 stress80; do
    ./build/run_my2_O2 "in_${set_name}.bin" "out_${set_name}_O2.bin"
    ./build/run_my2_O0 "in_${set_name}.bin" "out_${set_name}_O0.bin"
    if [[ -x build/run_my2_wrfobj ]]; then
        ./build/run_my2_wrfobj "in_${set_name}.bin" "out_${set_name}_wrfobj.bin"
        cmp "out_${set_name}_O2.bin" "out_${set_name}_wrfobj.bin"
    fi
    if cmp -s "out_${set_name}_O2.bin" "out_${set_name}_O0.bin"; then
        echo "${set_name}: -O0 build agrees with WRF's -O2 build"
    else
        echo "${set_name}: -O0 build differs from WRF's -O2 build (expected:" \
             "-O0 calls libgcc __powisf2 for integer powers and powf for **2.)"
    fi
done
sha256sum in_*.bin out_*_O2.bin > fortran-sha256sums.txt
cat fortran-sha256sums.txt
