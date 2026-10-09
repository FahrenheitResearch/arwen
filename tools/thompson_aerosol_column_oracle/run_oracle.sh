#!/usr/bin/env bash
# The mp=28 column oracle end to end on one Linux box with a GPU.
#
# usage: run_oracle.sh TREE WRF_BUILD_DIR RUN_DIR [DT ...]
#
# TREE is the gpuwm checkout to import (PYTHONPATH must already name it, as
# /work/pverify/base/env.sh sets it).  WRF_BUILD_DIR is what
# tools/thompson_real_column_parity/build_wrf.sh wrote.  RUN_DIR receives
# columns.npz (built once), gpu-{strict,default}-dtN.npz and
# summary-{strict,default}-dtN.json.  The GPU steps are the only GPU work;
# run this script through the box's GPU mutex.  Default DT list: 20 5.
set -euo pipefail
tree=$(realpath "$1"); wrf=$(realpath "$2"); run=$(realpath -m "$3"); shift 3
dts=("$@"); [ ${#dts[@]} -gt 0 ] || dts=(20 5)
here=$tree/tools/thompson_aerosol_column_oracle
mkdir -p "$run"; cd "$run"
[ -f columns.npz ] || python "$here/make_columns.py" "$wrf" columns.npz
for dt in "${dts[@]}"; do
  GPUWM_WRF_EXACT=1 python "$here/gpu_run.py" columns.npz "gpu-strict-dt$dt.npz" --dt "$dt"
  env -u GPUWM_WRF_EXACT python "$here/gpu_run.py" columns.npz "gpu-default-dt$dt.npz" --dt "$dt"
done
for dt in "${dts[@]}"; do
  for mode in strict default; do
    python "$here/compare.py" columns.npz "gpu-$mode-dt$dt.npz" "$wrf" "summary-$mode-dt$dt.json"
  done
done
