# Noah mosaic column oracle against WRF v4.6.1

This directory builds the v4.7.1 mosaic harness (`../noah_mosaic_wrf471_oracle/`)
unchanged from a pristine WRF v4.6.1 tree (tag v4.6.1, commit
d66e442fccc04111067e29274c9f9eaccc3cef28) under its own source pins, then
grades the GPU kernel against it.

## What differs between v4.6.1 and v4.7.1 here

Of the compiled sources, `module_sf_noahdrv.F`, `module_sf_noahlsm.F` and
`module_sf_noahlsm_glacial_only.F` are byte-identical. `module_sf_urban.F`
differs in three executable lines (1035, 1287, 1289): v4.6.1 subtracts the
kinematic vapour flux from rain without the RHOO/1000. conversion to a water
depth in the IMP_SCHEME=2 impervious water-retention update. That is a unit
error in v4.6.1, fixed in v4.7.1; the GPU kernel carries the v4.7.1 form and
does not copy the v4.6.1 one. `module_sf_bep_bem.F` (BEP, which lsm_mosaic never
calls) changes a loop bound.

Measured (receipts/):

* `raw-461crb2-vs-471crb2.json`: the two WRF builds write identical words in
  every family except the four whose urban switches are forced to
  IMP_SCHEME=2 (`ucm`, `ucm_lcz`, `ucm_wrfinit`, `ucm_lcz_wrfinit`).
  The shipped-table families (`ucm_table`, `ucm_lcz_table`, IMP_SCHEME 1)
  are identical.
* `raw-461unitfix-vs-471crb2.json`: `unit_fix_control.sh` recompiles a copy
  of v4.6.1 `module_sf_urban.F` with only those three lines in v4.7.1 form
  (`unit-fix-control.diff`); every UCM family then equals v4.7.1 word for
  word. The v4.6.1/v4.7.1 difference is exactly that fix.

## Libm on the oracle host

gfortran 13.3 on glibc 2.39 calls the host's atanf/log10f, which are not
correctly rounded. `NOAH_MOSAIC_CR_LIBM=1` links `cr_libm_control.c`
(correctly rounded atanf/log10f) so WRF's words stop depending on the host C
library. `raw-461stock2-vs-461crb2.json` measures the difference the host
routines make; `gpu-replay-o461stock2-*.json` grades the GPU against the
stock-libm build as well.

## The subnormal-CHS probe column

The `ftz` family's column 25 (zero-based 24) sets CHS, CHS2 and CQS2 to the
smallest subnormal. RR = EMISSI*T24*6.48E-8/(SFCPRS*CH)+1 overflows to
infinity in WRF and on the GPU alike; PC and ETP are NaN from there on.
`nan_trace.sh` (a diagnostic copy with PRINT statements, never an oracle)
traced WRF through it (`wrf461-chs-probe-trace.txt`). The words that differ
there cannot be reproduced and are not a GPU defect:

* NaN bit patterns: x86 writes 0x7fc00000/0xffc00000, CUDA arithmetic
  always returns 0x7fffffff.
* Soil moisture: WRF's SSTEP leaves SMC NaN through
  `MAX(MIN(STOT,SMCMAX),0.02)`. gfortran's NaN result for MAX/MIN depends on
  how the call is written (`gfortran_minmax_nan_probe.f90`,
  `gfortran-minmax-nan.txt`: the same MIN alone returns SMCMAX), and the
  Fortran standard leaves it processor dependent.
* QFX: NOPAC assigns ETA only when ETP > 0 or ETP <= 0. A NaN ETP takes
  neither branch, so WRF returns an undefined ETA (146.85 in one tile, NaN
  in the others). That is a WRF defect and is not copied.

## Rerun (W1, CPU then GPU through the mutex)

```sh
W461=/path/to/WRF-4.6.1 ; W471=/path/to/WRF-4.7.1
NOAH_MOSAIC_CR_LIBM=1 bash tools/noah_mosaic_wrf461_oracle/build.sh $W461 B461
NOAH_MOSAIC_CR_LIBM=1 bash tools/noah_mosaic_wrf471_oracle/build.sh $W471 B471
bash tools/noah_mosaic_wrf461_oracle/build.sh $W461 B461STOCK
bash tools/noah_mosaic_wrf461_oracle/unit_fix_control.sh $W461 B461 UNITFIX
python tools/noah_mosaic_wrf461_oracle/compare_raw.py B471 B461 out.json \
    tools/noah_mosaic_wrf471_oracle/receipts/fixture-sha256sums.txt
GPUWM_WRF_EXACT=1 python tools/noah_mosaic_wrf461_oracle/replay_gpu.py B461/fixtures strict.json
python tools/noah_mosaic_wrf461_oracle/replay_gpu.py B461/fixtures default.json
```

`replay_gpu.py` holds nothing out and classifies every differing word.
