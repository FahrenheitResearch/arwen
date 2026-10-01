#!/usr/bin/env bash
# Run the BEP+BEM oracle (run_bep_bem, built by this directory's build.sh or
# any build of run_bep_bem.F90 against the pinned WRF v4.7.1 sources) over the
# five fixture variants.  usage: run_bep_bem.sh BUILD_DIR WRF_TABLE_DIR OUT_DIR
#
# run_bep_bem takes OUTDIR USE_WUDAPT_LCZ NSTEPS and reads URBPARM.TBL or
# URBPARM_LCZ.TBL from its working directory, as WRF does, so each variant
# runs in its own directory with its own table.  The gr1pv and gr2 tables are
# the pinned URBPARM.TBL with the sed edits below: WRF ships the green-roof,
# photovoltaic, irrigation and air-conditioning-schedule arms switched off,
# and an arm no fixture reaches is an arm no gate checks.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "usage: run_bep_bem.sh BUILD_DIR WRF_TABLE_DIR OUT_DIR" >&2
    exit 2
fi
bld=$(realpath "$1"); tables=$(realpath "$2"); out=$(realpath -m "$3")
mkdir -p "$out"
run() { # name use_wudapt_lcz table nsteps
    local d="$out/$1"
    local tbl=URBPARM.TBL
    [[ $2 == 1 ]] && tbl=URBPARM_LCZ.TBL
    mkdir -p "$d/run" "$d/dump"
    cp "$3" "$d/run/$tbl"
    (cd "$d/run" && "$bld/run_bep_bem" "$d/dump" "$2" "$4" > "$d/stdout.txt" 2>&1) || {
        echo "run_bep_bem $1 failed" >&2; tail -20 "$d/stdout.txt" >&2; exit 1; }
}
sed -e 's/^GR_FLAG:0/GR_FLAG:1/' -e 's/^GR_TYPE: 2/GR_TYPE: 1/' \
    -e 's/^GR_FRAC_ROOF:0,0,0/GR_FRAC_ROOF:0.5,0.3,0.6/' \
    -e 's/^PV_FRAC_ROOF: 0,0,0/PV_FRAC_ROOF: 0.3,0.2,0.4/' \
    -e 's/^IRHO:.*/IRHO:0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1/' \
    -e 's/^TIME_ON: .*/TIME_ON: 8., 0., 6./' -e 's/^TIME_OFF: .*/TIME_OFF: 18., 24., 20./' \
    -e 's/^SW_COND: .*/SW_COND: 1, 0, 1/' \
    "$tables/URBPARM.TBL" > "$out/URBPARM.gr1pv.TBL"
sed -e 's/^GR_FLAG:0/GR_FLAG:1/' \
    -e 's/^GR_FRAC_ROOF:0,0,0/GR_FRAC_ROOF:0.4,0.8,0.2/' \
    -e 's/^IRHO:.*/IRHO:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1/' \
    "$tables/URBPARM.TBL" > "$out/URBPARM.gr2.TBL"
# a sed that matched nothing would silently hand the stock table back
for v in gr1pv gr2; do
    if cmp -s "$tables/URBPARM.TBL" "$out/URBPARM.$v.TBL"; then
        echo "URBPARM.$v.TBL is the stock table: a sed edit matched nothing" >&2
        exit 3
    fi
done
run stock 0 "$tables/URBPARM.TBL" 4
run lcz 1 "$tables/URBPARM_LCZ.TBL" 4
run gr1pv 0 "$out/URBPARM.gr1pv.TBL" 4
run gr2 0 "$out/URBPARM.gr2.TBL" 4
run long 0 "$tables/URBPARM.TBL" 30
sha256sum "$out"/URBPARM.*.TBL "$out"/*/dump/data.bin
