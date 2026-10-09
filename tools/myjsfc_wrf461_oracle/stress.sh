#!/usr/bin/env bash
# Large random-regime measurement beside the committed fixture.
#
#   bash tools/myjsfc_wrf461_oracle/stress.sh BUILD_DIR STRESS_DIR [NCOL]
#
# BUILD_DIR is a finished build.sh directory (its run_myjsfc is reused).
# Writes STRESS_DIR/myjsfc-stress.npz (two sets of NCOL columns, default
# 4096, 50 levels, 4 calls each; regimes drawn at random), which
# compare_myjsfc.py --fixture grades like the committed fixture.
set -euo pipefail
build_dir=$(realpath "$1")
stress_dir=$(realpath -m "$2")
ncol=${3:-4096}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "${stress_dir}"
cd "${stress_dir}"
python3 "${script_dir}/make_cases.py" "${stress_dir}" --stress "${ncol}"
for set in $(cat cases.txt); do
    "${build_dir}/run_myjsfc" "case_${set}.bin" "out_${set}.bin"
    "${build_dir}/run_myjsfc_O2" "case_${set}.bin" "o2_${set}.bin"
    cmp -s "out_${set}.bin" "o2_${set}.bin" \
        || { echo "WRF -O2 differs from -O0 on ${set}" >&2; exit 6; }
done
python3 "${script_dir}/pack_fixture.py" "${stress_dir}" "${stress_dir}/myjsfc-stress.npz"
echo "wrote ${stress_dir}/myjsfc-stress.npz"
