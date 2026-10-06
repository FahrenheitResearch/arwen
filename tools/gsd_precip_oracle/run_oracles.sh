#!/usr/bin/env bash
# CPU half of the oracle proof: build NOAA's code, write the cases, run them.
# Usage: bash run_oracles.sh <work-dir> <python>      (no GPU is touched)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=${1:?work directory}
py=${2:?python with numpy}
bash "$here/build.sh" "$work"
cd "$work"
"$py" "$here/make_cases.py" "$work" > cases.log
./oracle_a case_a.bin out_a.bin
: > runs.txt
# name input mode i_lightpcp iclean_hydro_withRef iclean_hydro_withRef_allcol threshold
while read -r name input mode light clean allcol threshold; do
  ./oracle_b "$input" "out_b_$name.bin" "$mode" "$light" "$clean" "$allcol" "$threshold"
  echo "$name $input $mode $light $clean $allcol $threshold" >> runs.txt
done <<'RUNS'
trim_build case_b.bin trim-build 1 1 1 5.0
trim_build_noctp case_b_noctp.bin trim-build 1 1 1 5.0
trim_build_options_off case_b.bin trim-build 0 1 0 5.0
clear case_b.bin clear 1 1 1 5.0
retrieve_all case_b.bin all 1 1 1 5.0
RUNS
ls -la out_a.bin out_b_*.bin
