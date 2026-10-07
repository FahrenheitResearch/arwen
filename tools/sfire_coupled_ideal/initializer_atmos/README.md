# Native atmospheric initialization controls

The source extractor pins WRF v4.7.1 `module_initialize_fire.F` and
`module_init_utilities.F`. It keeps the atmospheric initializer block,
`get_sounding`, `read_sounding`, and `interp_0` verbatim inside a small allocated
domain. The original and corrected surface controls use the same wrapper.
The correction inserts the original three temperature expressions before each
column's TSK assignment. Original WRF instead reuses the last column's values.
No other scientific expression is transformed.

`run.F90` covers dry and moist sounding integration, increasing and decreasing
interpolation coordinates with both endpoint extrapolations, terrain-dependent
hydrostatics, staggered winds, positive and negative thermal bubbles, and both
surface initialization branches. The native input vapor is g/kg; initialized
vapor is kg/kg. The header surface vapor is read but unused by native WRF.

The native compiler is GNU Fortran 15.2.0. Module compilation and linking use
`-O0 -ffp-contract=off -ffree-line-length-none -fcheck=all`. The receipts record
the source blocks, wrapper, executable and linked constant/error/I/O objects.
The fixture collector decodes the native stream layout only.

Reproduction uses the linked objects from `tools/sfire_wrf471_oracle`:

```sh
python tools/sfire_coupled_ideal/initializer_atmos/extract.py BUILD
gfortran -O0 -ffp-contract=off -ffree-line-length-none -fcheck=all \
  -I ORACLE_BUILD -J BUILD -c BUILD/native_atmos.F90 -o BUILD/native_atmos.o
gfortran -O0 -ffp-contract=off -ffree-line-length-none -fcheck=all \
  -I ORACLE_BUILD -I BUILD tools/sfire_coupled_ideal/initializer_atmos/run.F90 \
  BUILD/native_atmos.o ORACLE_BUILD/stub_wrf.o \
  ORACLE_BUILD/module_model_constants.o ORACLE_BUILD/module_wrf_error.o \
  ORACLE_BUILD/oracle_io.o -o BUILD/run
cp tools/sfire_coupled_ideal/initializer_atmos/reference/input_sounding BUILD/input_sounding
cd BUILD
./run OUTPUT
```

The checked-in fixtures retain original and corrected native results separately.
Both RTX 4090 and RTX 5090 pass 14 GPU controls, 162 field checks and 71,174 words
with zero differing words and zero ULP. Ten main controls check 71,014 words;
four supplemental sounding controls check 160 words, including nonzero first
height and a two-level profile. The GPU options are `--fmad=false` and
`--ftz=false`; the composed compiler source is recorded in each receipt. TSK/TMN
are graded against corrected WRF; all other initialized atmospheric fields are
unchanged from original WRF. T_DRY retains physical dry potential temperature;
T also exposes WRF's final moist potential temperature conversion for grading.

The public command and checkpoint continuity have separate receipts. These
routine controls alone do not claim forecast skill or a full coupled forecast.
