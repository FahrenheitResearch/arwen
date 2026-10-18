#!/usr/bin/env bash
# The New Tiedtke column-oracle matrix: every run, both arithmetic modes,
# against all three WRF builds.
#
#   run_matrix.sh STAGE
#
#   STAGE = cases   build the cases from wrfout files (CPU)
#           wrf     run the three WRF builds over every run (CPU)
#           woof    run WOOF over every run, default then strict (GPU;
#                   call it through the GPU mutex)
#           compare compare every run (CPU)
#
# Environment: T (the tree to run), OUT (where cases and runs go),
# WRF_SRC (a WRF v4.6.1 source tree), and the wrfout paths below.
set -euo pipefail
: "${T:?set T to the source tree}"; : "${OUT:?set OUT}"; : "${WRF_SRC:?set WRF_SRC}"
H="$T/tools/ntiedtke_column_oracle"
B="$OUT/build"
DATA="$OUT/data"; RUNS="$OUT/runs"
mkdir -p "$DATA" "$RUNS"

# The runs: name case dx stepcu itimestep bl_lanes replicate
# itimestep here is what WRF is handed; WOOF derives its own from the
# model time it is given (elapsed 0 is the first step, 3600 s is not).
RUNS_TABLE=$(cat <<'EOF'
ncar00z-dx12000      ncar00z 12000 1 2 1 1
ncar00z-dx1500       ncar00z 1500  1 2 1 1
ncar00z-dx4500       ncar00z 4500  1 2 1 1
ncar00z-dx15000      ncar00z 15000 1 2 1 1
ncar00z-dx27000      ncar00z 27000 1 2 1 1
ncar13z-dx12000      ncar13z 12000 1 2 1 1
tor20z-dx1000        tor20z  1000  1 2 1 1
bos20z-dx2250        bos20z  2250  1 2 1 1
mix-dx12000          mix     12000 1 2 1 1
mix-dx12000-fused    mix     12000 1 2 1 9
mix-dx12000-chunked  mix     12000 1 2 1 47
mix-dx12000-stepcu2  mix     12000 2 2 1 1
mix-dx12000-first    mix     12000 1 1 1 1
mix-dx12000-nopbl    mix     12000 1 2 0 1
stable-dx12000       stable  12000 1 2 1 1
quiet-dx12000        quiet   12000 1 2 1 1
diff18-dx12000       diff18  12000 1 2 1 1
diff18-dx12000-first diff18  12000 1 1 1 1
EOF
)

stage=${1:?usage: run_matrix.sh cases|wrf|woof|compare}
case "$stage" in
cases)
  bash "$H/build.sh" "$WRF_SRC" "$B" > "$OUT/build.log"
  python "$H/make_columns.py" "$NCAR00Z" "$DATA/ncar00z" &
  python "$H/make_columns.py" "$NCAR13Z" "$DATA/ncar13z" --coeffs "$NCAR00Z" &
  python "$H/make_columns.py" "$TOR20Z" "$DATA/tor20z" &
  python "$H/make_columns.py" "$BOS20Z" "$DATA/bos20z" &
  wait
  python "$H/make_subset.py" "$DATA/ncar00z" "$DATA/mix" --kind mix
  python "$H/make_subset.py" "$DATA/ncar13z" "$DATA/stable" --kind stable --n 2048
  python "$H/make_subset.py" "$DATA/mix" "$DATA/diff18" --kind cols --cols "$DIFF18_COLS"
  ;;
wrf)
  while read -r name c dx stepcu it bl rep; do
    [[ -z "$name" ]] && continue
    for flavour in ref zfix stock; do
      exe=run_nt_oracle
      [[ $flavour == zfix ]] && exe=run_nt_oracle_zcbase_real
      [[ $flavour == stock ]] && exe=run_nt_oracle_stock
      d="$RUNS/$name/wrf-$flavour"; mkdir -p "$d"
      echo "$B/$exe $DATA/$c $d $dx $stepcu $it $bl"
    done
  done <<< "$(grep -v '^quiet' <<< "$RUNS_TABLE")" | xargs -P 24 -I{} bash -c '{} > /dev/null'
  # The quiet group is cut from WRF's own answer on the stable subset, so
  # it is made after that run.
  python "$H/make_subset.py" "$DATA/stable" "$DATA/quiet" --kind quiet       --n 512 --wrf "$RUNS/stable-dx12000/wrf-zfix"
  for flavour in ref zfix stock; do
    exe=run_nt_oracle
    [[ $flavour == zfix ]] && exe=run_nt_oracle_zcbase_real
    [[ $flavour == stock ]] && exe=run_nt_oracle_stock
    mkdir -p "$RUNS/quiet-dx12000/wrf-$flavour"
    "$B/$exe" "$DATA/quiet" "$RUNS/quiet-dx12000/wrf-$flavour" 12000 1 2 1 > /dev/null
  done
  ;;
woof)
  while read -r name c dx stepcu it bl rep; do
    [[ -z "$name" ]] && continue
    elapsed=3600; [[ $it == 1 ]] && elapsed=0
    extra=(); [[ $bl == 0 ]] && extra=(--no-bl-lanes)
    for mode in default strict; do
      d="$RUNS/$name/woof-$mode"
      if [[ $mode == strict ]]; then export GPUWM_WRF_EXACT=1; else unset GPUWM_WRF_EXACT; fi
      python "$H/woof_side.py" "$DATA/$c" "$d" --dx "$dx" --replicate "$rep" \
          --stepcu "$stepcu" --elapsed "$elapsed" "${extra[@]}" > /dev/null
    done
  done <<< "$RUNS_TABLE"
  unset GPUWM_WRF_EXACT
  ;;
compare)
  while read -r name c dx stepcu it bl rep; do
    [[ -z "$name" ]] && continue
    for mode in default strict; do
      for flavour in ref zfix stock; do
        python "$H/compare.py" "$DATA/$c" "$RUNS/$name/wrf-$flavour" \
            "$RUNS/$name/woof-$mode" --label "$name $mode vs $flavour" \
            --json "$RUNS/$name/cmp-$mode-$flavour.json" || true
      done
    done
  done <<< "$RUNS_TABLE"
  python "$H/summarize.py" "$RUNS" > "$OUT/summary.md"
  cat "$OUT/summary.md"
  ;;
*) echo "unknown stage $stage" >&2; exit 2;;
esac
