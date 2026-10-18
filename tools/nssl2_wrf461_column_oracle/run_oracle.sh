#!/usr/bin/env bash
# One oracle pass on the box:
#   run_oracle.sh OUTDIR DT ITIMESTEP exact|default
# Needs: env.sh sourced with T naming the tree under test, ORACLE_BUILD naming
# the build.sh output, OUTDIR/columns.npz from make_columns.py.  The GPU half
# must already be inside a gpu-mutex grant.
#
# 1. WOOF (GPU) runs the production mp_physics=18 path and writes the
#    microphysics-time inputs it prepared as the Fortran input stream.
# 2. WRF runs on those bytes:
#      fixed   the reference: WRF's preprocessed source with the four
#              documented defects corrected (wrf_bugfix.py), scalar libm
#      scalar  stock WRF source, scalar libm (no libmvec)
#      stock   the stock wrf.exe object (libmvec powf), per column and slab
#      dump    the fixed build plus stage and GS-variable dumps
# 3. compare_bitwise.py grades WOOF against the reference (the verdict), and
#    reports WOOF against stock WRF, what the four defects change in stock
#    WRF (fixed vs scalar), and stock vs scalar libm; compare_stages.py and
#    run_gpu_isolated.py attribute any miss to a stage.
set -euo pipefail
out=$(realpath "$1"); dt=$2; its=$3; arith=${4:-exact}
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
tag=dt${dt}_it${its}_${arith}
if [ "$arith" = exact ]; then export GPUWM_WRF_EXACT=1; else unset GPUWM_WRF_EXACT; fi
python "$here/run_gpu_columns.py" "$out/columns.npz" "$out/$tag" --dt "$dt" \
  --itimestep "$its" --stages
unset GPUWM_WRF_EXACT || true

# nssl_2mom_init opens ./namelist.input for an optional &nssl_mp_params group;
# a WRF run directory has no such group, which an empty file reproduces.
mkdir -p "$out/wrfcwd"; : > "$out/wrfcwd/namelist.input"
fin="$out/$tag.fin.bin"
( cd "$out/wrfcwd"
  "$ORACLE_BUILD/nssl2_column_oracle_fixed" "$fin" "$out/$tag.wrf_fixed.bin" "$dt" "$its" column
  "$ORACLE_BUILD/nssl2_column_oracle_scalar" "$fin" "$out/$tag.wrf_scalar.bin" "$dt" "$its" column
  "$ORACLE_BUILD/nssl2_column_oracle_stock" "$fin" "$out/$tag.wrf_stock.bin" "$dt" "$its" column
  "$ORACLE_BUILD/nssl2_column_oracle_stock" "$fin" "$out/$tag.wrf_stock_slab.bin" "$dt" "$its" slab
  N2M_STAGE_DUMP="$out/$tag.wrf_stages.bin" N2M_GS_DUMP="$out/$tag.wrf_gs.bin" \
    "$ORACLE_BUILD/nssl2_column_oracle_dump" "$fin" "$out/$tag.wrf_dump_final.bin" "$dt" "$its" column
) > "$out/$tag.wrf.log"
cmp -s "$out/$tag.wrf_stock.bin" "$out/$tag.wrf_stock_slab.bin" \
  && echo "WRF stock per-column == WRF stock slab (byte-identical)" \
  || echo "WRF stock per-column != WRF stock slab"
cmp -s "$out/$tag.wrf_fixed.bin" "$out/$tag.wrf_dump_final.bin" \
  && echo "WRF dump build == WRF fixed reference (byte-identical; dumps did not perturb)" \
  || echo "WARNING WRF dump build != WRF fixed reference: stage dumps are not trustworthy"
cd "$here"
echo "== WOOF vs WRF fixed reference (verdict)"
python compare_bitwise.py "$out/$tag.wrf_fixed.bin" "$out/$tag.gpu.npz" \
  "$out/columns.npz" "$out/$tag.report.json" || true
echo "== stage attribution (fixed reference)"
python compare_stages.py "$out/$tag.wrf_stages.bin" "$out/$tag.gpu_stages.npz" \
  "$out/columns.npz" "$out/$tag.stages.json" || true
echo "== stage-isolated WOOF runs from WRF intermediate states"
if [ "$arith" = exact ]; then export GPUWM_WRF_EXACT=1; fi
python run_gpu_isolated.py "$out/$tag" "$out/columns.npz" --dt "$dt" \
  --final "$out/$tag.wrf_fixed.bin" || true
unset GPUWM_WRF_EXACT || true
echo "== WOOF vs stock WRF (scalar libm)"
python compare_bitwise.py "$out/$tag.wrf_scalar.bin" "$out/$tag.gpu.npz" \
  "$out/columns.npz" "$out/$tag.report-stockwrf.json" > "$out/$tag.stockwrf.txt" || true
head -1 "$out/$tag.stockwrf.txt"
echo "== effect of the four WRF defects (stock scalar vs fixed)"
python compare_bitwise.py "$out/$tag.wrf_scalar.bin" "$out/$tag.wrf_fixed.bin" \
  "$out/columns.npz" "$out/$tag.defects.json" --b-is-fortran \
  > "$out/$tag.defects.txt" || true
head -1 "$out/$tag.defects.txt"
echo "== stock wrf.exe object (libmvec powf) vs scalar libm"
python compare_bitwise.py "$out/$tag.wrf_stock.bin" "$out/$tag.wrf_scalar.bin" \
  "$out/columns.npz" "$out/$tag.stock-vs-scalar.json" --b-is-fortran \
  > "$out/$tag.stock-vs-scalar.txt" || true
head -1 "$out/$tag.stock-vs-scalar.txt"
