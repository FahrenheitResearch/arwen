# RUC soil-ingest oracle, WRF v4.6.1

The first WRF number for `real.exe`'s RUC side. RUC LSM was admitted as a
scheme -- a 600-step hour, dispatch proved, restart bit-identical -- with no
way to initialise it from real data, because `gpuwm/ingest/soil_contract.py`
targets Noah's four-layer geometry and refuses RUC by name. A prior lane
declined to invent a RUC remap on the grounds that producing a target with no
oracle behind it would be fabrication. This is that oracle.

## What it builds

```bash
bash tools/ruc_soil_ingest_wrf461_oracle/build.sh <WRF_SOURCE_ROOT> <BUILD_DIR>
```

`WRF_SOURCE_ROOT` must be at `d66e442fccc04111067e29274c9f9eaccc3cef28` with
`share/module_soil_pre.F` clean; `build.sh` refuses otherwise.

The artefact is `gpuwm/data/ruc/oracle/soil_ingest.csv`, 140 rows: seven
source configurations across twenty columns. `libmvec-report.txt` and
`oracle-sha256sums.txt` are the receipts.

## Why the preprocessing is WRF's own, not `gfortran -cpp`

`share/module_soil_pre.F` calls `wrf_error_fatal`, but a full WRF build's
object references `wrf_error_fatal3_`. The rewrite is `tools/standard.exe`
(`arch/standard.sed:8`), a step in WRF's `.F.o` rule at
`configure.wrf:365-370`. Running that rule rather than hand-patching the
source is what keeps the source byte-unmodified, and `build.sh` proves the
pipeline is WRF's by diffing its generated `module_soil_pre.f90` against the
one WRF's own build left in `share/`. They are identical.

## What is stubbed

Exactly the seven WRF symbols `nm -u share/module_soil_pre.o` reports on a
full build: `module_date_time`'s `current_date` / `start_date` (two
`CHARACTER`s, read only by a diagnostic write the soil-depth path never
reaches), `wrf_message`, `wrf_debug`, `wrf_error_fatal3`, `nl_get_mminlu`,
`nl_get_aggregate_lu`.

`frame/module_state_description.F` is **not** stubbed. It is compiled from the
pinned tree, because `process_soil_real` branches on `RUCLSMSCHEME` /
`LSMSCHEME` / `SLABSCHEME` and a stub would be free to get one wrong.

## libmvec

WRF compiles `share/` at `-O2 -ftree-vectorize`, and on gfortran 13.3.0 that
setting **does** pull `_ZGVbN4v_expf` into `module_soil_pre.o` -- from
`adjust_soil_temp_new`'s `exp`. This is the one module in the project where
the swap is observed on WRF's own settings rather than argued about. It cannot
move these numbers: `init_soil_3_real` contains no transcendental at all, only
`+ - * /` and `MAX`. The oracle is still built at `-O0`, `build.sh` still
fails on a `_ZGV*` symbol in the reference object, and it still builds the
positive control, because a guard that has never fired proves nothing.

## Measuring against it

```bash
python tools/ruc_soil_ingest_wrf461_oracle/validate_ruc_soil_ingest_oracle.py
```

`gpuwm.ingest.ruc_soil` is bit-identical on 118 of the 140 rows and refuses
the other 22 by design; `gpuwm/data/ruc/PROVENANCE.md` records which and why,
and `tests/test_ruc_soil_ingest.py` binds both halves. The parity gate passed
on its first run, so that file leads with three negative controls -- layer-
bottom instead of layer-midpoint sampling (536,458 ULP), a "corrected" `dzs`
partition (78,081 ULP), and float64-then-round-once (1-3 ULP) -- because a
gate nobody has seen fail measures nothing.

## Does it work on real data? Yes -- and stock `real.exe` does not

`exercise_on_metgrid.py` assembles a met_em file's soil the way WRF's own
reader does and runs the remap over every column:

```bash
python tools/ruc_soil_ingest_wrf461_oracle/exercise_on_metgrid.py MET_EM.nc
```

On this project's production case (`WRF_1974_MP55_reference_bundle/met_em`,
1974-04-03_12:00:00), all four domains -- **861,001 columns**, the full
four-domain width -- produce nine distinct, physical levels. The sample depths
WRF's reader derives come out as the integer centimetre midpoints 3, 17, 64,
194, which is the convention the oracle proved.

The same files put through **stock `real.exe` at `sf_surface_physics = 3,
num_soil_layers = 9` abort**:

```
 Assume non-RUC LSM input
 from Noah to RUC - compute Noah bucket
 error in the grid%tsk
 i,j=          15           1
 grid%landmask=   0.00000000
 grid%tsk, grid%sst, grid%tmn=   0.00000000  0.00000000  0.00000000
FATAL CALLED FROM FILE:  <stdin>  LINE:  3142
grid%tsk unreasonable
```

The identical met_em at `sf_surface_physics = 2` writes `wrfinput_d01` without
complaint. The single line of difference is `init_soil_3_real:2138`,
`tsk(i,j) = sst(i,j)` on every water column with no validity test;
`init_soil_2_real:1732-1754`, the Noah arm, writes `tslb`/`smois`/`sh2o` over
water and never touches `tsk`. 1,494 of d01's 14,397 water cells carry
`SST = 0` -- inland water with no 1974 SST analysis behind it -- and RUC dies
on the first of them. On d02 and d03 it is 41% and 49% of water.

So the WRF-produced-`wrfinput` route to a RUC oracle **does not exist for this
case**, which is why the harness route was taken. `gpuwm.ingest.ruc_soil`
refuses a nonphysical water SST at the seam with that measurement in the
message, rather than propagating a 0 K ocean; repairing it from the skin
temperature -- which is what `gpuwm/ingest/soil.py:275-277` already does for
Noah -- is what makes the case run.
