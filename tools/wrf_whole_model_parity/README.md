# Whole-model comparison against stock WRF, with WRF's own build spread as the yardstick

The per-scheme column oracles (`tools/*_wrf461_oracle`) grade one scheme's
kernel word for word.  This tool asks the other question: started from the
same real.exe pair and the same namelist, how far is a short WOOF forecast
from a stock WRF 4.6.1 forecast, in default and strict arithmetic, and is
that distance larger than the distance between two builds of WRF itself?

## What is compared

- Referee: WRF v4.6.1 built scalar
  (`-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize`, serial).
- Yardstick: the same source built with vectorisation
  (`-O2 -fno-fast-math -ffp-contract=off`, glibc libmvec), against the referee.
- Candidates: WOOF in default arithmetic and under `GPUWM_WRF_EXACT=1`,
  through the WRF-file door (`gpuwm run --wrfinput`).
- Arms (`arms.py`, one table row each): the wizard's default 12 km root
  (Morrison, YSU, Noah, Kain-Fritsch every 5 minutes, radiation as WRF's own
  RRTMG), Kain-Fritsch every step, WSM6, Kessler, Milbrandt-Yau, WDM6, MYJ
  with the Eta surface layer, and a cumulus-off control.  Dudhia shortwave
  and RRTM longwave run in every arm but the first.
- History every step; every floating field both sides write is compared at
  every frame (`compare.py`): differing words, max ULP, max |diff|, RMS over
  the whole array and over the interior (5-cell specified rim excluded).

The rule that turns those numbers into a verdict is in `summarize.py`'s
docstring and was fixed before any WOOF run was scored.

## Run it

```bash
ROOT=/work/parity; CASE=/path/to/real.exe-pair        # wrfinput_d01, wrfbdy_d01, namelist.input
WRF=/path/to/WRF-4.6.1-scalar; WRFV=/path/to/WRF-4.6.1-vectorised
python tools/wrf_whole_model_parity/arms.py $CASE $ROOT --steps 15
python tools/wrf_whole_model_parity/prep_inputs.py $ROOT $CASE $WRF     # per-arm copies both sides can read
for arm in $(ls $ROOT/cases); do
  tools/wrf_whole_model_parity/run_wrf_arm.sh $ROOT $CASE $WRF  $arm scalar &
  tools/wrf_whole_model_parity/run_wrf_arm.sh $ROOT $CASE $WRFV $arm vector &
done; wait
# one card per call; native bridges and the engine's Python environment set by the caller
tools/wrf_whole_model_parity/run_woof_arms.sh $ROOT $CASE $PWD tip default $(ls $ROOT/cases)
GPUWM_WRF_EXACT=1 tools/wrf_whole_model_parity/run_woof_arms.sh $ROOT $CASE $PWD tip strict $(ls $ROOT/cases)
for arm in $(ls $ROOT/cases); do
  python tools/wrf_whole_model_parity/compare.py $ROOT/wrf/$arm/scalar $ROOT/wrf/$arm/vector $ROOT/out/$arm.wrfvec-vs-wrfscalar.json
  for mode in default strict; do
    python tools/wrf_whole_model_parity/compare.py $ROOT/wrf/$arm/scalar $ROOT/woof/tip/$mode/$arm/run/wrfout \
        $ROOT/out/$arm.woof-tip-$mode-vs-wrfscalar.json
  done
done
python tools/wrf_whole_model_parity/summarize.py $ROOT/out $ROOT/out/summary.json
```

## Why the inputs are copied per arm

A real.exe pair is written for one suite.  WRF runs another suite from it;
WOOF's door refuses a header naming another `MP_PHYSICS`, a moisture species
the active scheme does not hold, and a missing one.  `prep_inputs.py` writes,
per arm, a copy that says what WRF itself made of the file (see its
docstring), and WRF is rerun on the copy, so both sides read the same bytes.

## Recorded result

`receipts/` holds the summary of the run of record and its provenance.  This
is release evidence, rerun at a cut on a box with the two WRF builds; it is
not a battery leg, because the battery has no WRF build.  The battery's
standing checks for these schemes are the column oracles.
