# Noah mosaic WRF v4.7.1 column oracle

This harness calls WRF's public `lsm_mosaic_init` and `lsm_mosaic` entry points.
Every compiled or copied upstream file must match `SOURCES.sha256` before
compilation. Sources are compiled byte-unmodified at `-O0`, using the EM core
and four-byte REAL/INTEGER defines. Only CAL_MON_DAY is extracted, byte for byte,
into the service module expected by the Noah driver. Service stubs supply no
physics. Real urban modules are linked so later UCM coverage can use this build.

Regenerate on a Linux host with gfortran and binutils:

```sh
bash tools/noah_mosaic_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
```

The build checks all `-O0` objects for vector libm symbols. An `-Ofast` positive
control must produce a vector symbol or the build fails. The fixture writer
emits little-endian float32/int32 stream files and a Fortran-order manifest.
`compiler.txt`, `libmvec-report.txt` and `oracle-sha256sums.txt` record the build.

`run_mosaic_init.F90` covers 18 cells at two horizontal rows for each of the
16 NLCAT/mosaic-count/sea-ice-mode combinations. It dumps complete NLCAT arrays,
all supplied grid inputs and all initialised tile arrays, including urban state.
Coverage includes stable ties, one-hot fractions, zero totals, tiny totals,
awkward fractions, five-category mixtures, water rotation and elimination,
water-mask inconsistencies, sea-ice dominance, lake category and LCZ entries.

`run_mosaic.F90` produces these families:

- `base`: 48 columns, three tiles, four sequential steps, four switch combinations.
  The physical variety comes from the existing 42-column Noah driver oracle.
  Additional columns include glacial tiles in two loop positions, urban/barren
  tiles, excluded water tiles, open water, sea ice and an ordinary land column.
- `cats`: one and five tiles, four steps each.
- `lcz`: NLCAT=61, eleven LCZ dominant tiles, urban physics disabled, four steps.
- `twins`: three separate one-tile calls using each initial tile of the first
  base variant. They copy WRF's actual init layout, retain the selected tile,
  set the fraction one-hot, and start cell accumulators at zero. The defect
  control grades only non-glacial land columns with rdlai2d false.
- `ftz`: one three-tile step with subnormal and signed-zero probes in snow,
  rain, humidity, canopy water, exchange coefficient and vegetation fraction.
  The GPU test pins 46 column/field pairs from the SNOW and CHS probes,
  including an exact SHA-256 of every held-out GPU output word array.

Inactive dummy arguments are initialised to zero. RC_mosaic is deliberately
not dumped: WRF's glacial arm leaves its source scalar undefined or stale (D3).
It and RS/XLAIDYN are outside the port state contract.

The D1 control needs both input and output tile soil state. WRF's erroneous
NS*mosaic_i reduction executes inside its reverse tile loop. An index into a
lower tile observes that tile before it updates. Reducing final tile outputs
alone does not reproduce the byte-unmodified WRF oracle.

Every step also emits increment_<field> arrays from independent one-tile WRF
calls with the PRE-step tile state, identical forcing, unit area, and zero cell
accumulators. RDLAI2D calls account for a preceding glacial tile's LAI=0.01 carry.
The CPU control reproduces the original unweighted WRF accumulators from these
increments; the GPU gate grades the specified corrected weighted sums.

The CUDA port is in gpuwm/core/kernels/noah_mosaic.cu with the public launcher
in gpuwm/core/noah_mosaic.py.  The gates are pytest suites:

```sh
python -m pytest tests/test_noah_mosaic_init_wrf471.py     tests/test_noah_mosaic_oracle_controls.py          # CPU
python -m pytest tests/test_noah_mosaic_wrf471_parity.py   # GPU
```

The GPU replay checks all 72 output fields at 0 ULP outside the exact FTZ
pairs, word equality (signed zero included), unchanged IVGTYP/ISLTYP and
category arrays, and untouched reslin/ebal sentinels.  It was made to fail
twice before it was committed (receipts/mutation-gpu.txt,
receipts/mutation-snow.txt): restoring WRF's NS*mosaic_i grid-soil index, and
restoring lsm's SWDOWN>10 condition in mosaic snow damping.

The committed corpus is packed one file per case.  To regenerate it:

```sh
bash tools/noah_mosaic_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
python tools/noah_mosaic_wrf471_oracle/pack_fixtures.py     BUILD_DIR/fixtures/mosaic tests/data/oracles/noah_mosaic
python tools/noah_mosaic_wrf471_oracle/pack_fixtures.py     BUILD_DIR/fixtures tests/data/oracles/noah_mosaic   # mosaic_init
```

pack_fixtures.py refuses any .bin whose SHA-256 is not the one recorded in
receipts/fixture-sha256sums.txt, so a regenerated corpus that moved a word must
update that list in the same change.

The scalar init transcription is retained as _lsm_mosaic_init_reference. The
public init is vectorized using stable descending sort, masked rotations and
ordered float32 normalisation. NaN fractions are refused because the sort's
ordering would differ from WRF's strict comparisons. Oracle and randomized tests
compare every word, including signed zero. real_exe_landusef includes the
pre-arm XICE removal over LANDMASK>0.5 at module_soil_pre.F:149-157.

The float operators are explicit. SRT's INTEGER CVFRZ-J power uses libgcc's
square-and-multiply order; real exponents use gfk_pow. The shared glibc exp/log/pow
header is prepended only to this new module. LOG10 uses the glibc 2.43 CORE-MATH
implementation because the existing header's glibc 2.39 routines do not supply it.
Folded arithmetic constants were read from gfortran 15.2 using folded_words.F90;
word and expression receipts are under receipts/.

`run_mosaic_ucm.F90` adds `ucm` and `ucm_lcz`: 48 columns, three tiles,
four carried steps each. Urban tiles occupy all three tile positions;
fractions are 0.1, 0.5, 0.95, 0.99 and 1.0. There are dominant-urban cells,
urban-secondary cells, non-urban controls, calm wind, day, night and rain.
The LCZ family includes categories 51 and 54 sharing one grid state and an
ISURBAN tile mapped to UTYPE 5. It calls urban_param_init and urban_var_init
before lsm_mosaic_init, then gives tile and shared grid state distinct words.

Both families use CH_SCHEME=1, TS_SCHEME=1, AHOPTION=1, ALHOPTION=1,
IMP_SCHEME=2, IRI_SCHEME=0 and the table's boundary switches and FGR.
GROPTION is 0 for `ucm`, 1 for `ucm_lcz`. Every packed UCM table column,
global and switch is dumped by WRF and used directly by the GPU replay.
OASIS remains 1.0; the host continues refusing irrigation and oasis arms.
The green-roof night forcing stays on positive EPGR, avoiding WRF's existing
undefined ETR read under dew. There are no held-out pairs in these families.

The surface-driver category predicate and assignments are captured verbatim
from module_surface_driver.F:3004-3016 in the committed `.inc`. The build
hash-checks that source and compares the include to a fresh extraction.
It never changes upstream WRF. Independent one-tile increments preserve
the preceding urban tile's grid CHS floor before computing the D2 control.

```sh
python tools/noah_mosaic_wrf471_oracle/pack_ucm_run.py \
    BUILD_DIR/fixtures/mosaic_ucm tests/data/oracles/noah_mosaic
python -m pytest tests/test_noah_mosaic_ucm_wrf471_parity.py
```

The replay measures 143 arrays per family, 114,432 words total, at zero ULP
and raw-word equality, including every tile state, shared grid state and
override. D1/D2 retain the existing corrected mosaic reduction contract;
all other comparisons use raw byte-unmodified WRF output. The FRC < 0.99
mutation fails the first step. The old mosaic gate still has its exact FTZ
pairs and hashes. The composed sm_89 frame is 400 bytes, plain mosaic 240.

`run_mosaic_ucm.F90` families 3 and 4, `ucm_wrfinit` and `ucm_lcz_wrfinit`,
repeat `ucm` and `ucm_lcz` with one difference: FRC_URB2D stays exactly as
`urban_var_init` leaves it, the table fraction where the cell's dominant
category is urban and 0 in every urban-secondary cell. That is WRF's own
rule for where the canopy runs under mosaic, and gpuwm's default
(`mosaic_urban_canopy = "dominant"`). The CPU test
`tests/test_noah_mosaic_urban_canopy_rule.py` requires gpuwm's own
urban initialization to produce those fractions word for word, and the GPU
replay `test_wrfs_dominant_urban_rule_every_output_word` runs the tile loop
on gpuwm's fractions against every WRF output word. The rebuild that added
them reproduced every recorded word of the other families
(receipts/ucm-wrfinit-rebuild.txt).

```sh
python tools/noah_mosaic_wrf471_oracle/pack_ucm_run.py \
    BUILD_DIR/fixtures/mosaic_ucm tests/data/oracles/noah_mosaic \
    ucm_wrfinit ucm_lcz_wrfinit
```
