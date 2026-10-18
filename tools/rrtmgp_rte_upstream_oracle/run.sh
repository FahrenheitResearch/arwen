#!/usr/bin/env bash
# One oracle pass: columns (once), the two Fortran references (once each),
# WOOF under default arithmetic and under the strict build, and the
# comparisons.
#
#   run.sh TREE BUILD_DIR FIXED_BUILD_DIR WORK_DIR
#
# TREE is the gpuwm checkout to import, BUILD_DIR holds the stock
# rte_column_oracle and FIXED_BUILD_DIR the one built with --boundary-fixes
# (build.sh).  WORK_DIR receives cols.npz, the Fortran stream files, the
# WOOF outputs, compare-{default,strict}{,-fixed}.json and refdiff.json.
# The GPU steps must already be inside a card grant (see README.md).
set -euo pipefail
tree=$(realpath "$1")
build=$(realpath "$2")
fixed=$(realpath "$3")
work=$(realpath -m "$4")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$work"
export PYTHONPATH=$tree:$tree/tests
data=$(python -c "from gpuwm import data_assets; print(data_assets.rrtmgp_data_dir())")
if [[ ! -f "$work/cols.npz" ]]; then
    python "$here/oracle.py" make --npz "$work/cols.npz" \
        --fortran-in "$work/cols.bin"
fi
if [[ ! -f "$work/ref.bin" ]]; then
    "$build/rte_column_oracle" "$data" "$work/cols.bin" "$work/ref.bin"
fi
if [[ ! -f "$work/ref-fixed.bin" ]]; then
    "$fixed/rte_column_oracle" "$data" "$work/cols.bin" "$work/ref-fixed.bin"
fi
for mode in default strict; do
    if [[ $mode == strict ]]; then export GPUWM_WRF_EXACT=1; else unset GPUWM_WRF_EXACT; fi
    python "$here/oracle.py" woof --npz "$work/cols.npz" \
        --out "$work/woof-$mode.npz"
    unset GPUWM_WRF_EXACT
    for ref in ref ref-fixed; do
        tag=${ref#ref}
        echo "== $mode against $ref"
        python "$here/oracle.py" compare --npz "$work/cols.npz" \
            --ref "$work/$ref.bin" --woof "$work/woof-$mode.npz" \
            --json "$work/compare-$mode$tag.json"
    done
done
echo "== stock upstream against upstream with the boundary fixes"
python "$here/oracle.py" refdiff --npz "$work/cols.npz" \
    --ref "$work/ref.bin" --other "$work/ref-fixed.bin" \
    --json "$work/refdiff.json"
