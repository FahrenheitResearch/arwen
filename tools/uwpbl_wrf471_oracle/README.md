# UW moist-turbulence PBL oracle (WRF v4.7.1, bl_pbl_physics = 9)

Column oracle for the port of WRF's CAMUWPBL scheme
(`phys/module_bl_camuwpbl_driver.F` and the CAM modules it calls: eddy_diff,
diffusion_solver, wv_saturation and their support modules).  The reference
is the byte-unmodified WRF source compiled by gfortran on the oracle host; the port
(`gpuwm/core/uwpbl.py`, `gpuwm/core/kernels/uwpbl*.cu*`) and the CPU
reference (`gpuwm/verify/uwpbl_ref/`) are graded against its fixtures bit
for bit.

## Files

| file | what it is |
|---|---|
| `SOURCES.sha256` | sha256 of every WRF v4.7.1 file used or cited (tag v4.7.1, commit f52c197ed39d12e087d02c50f412d90d418f6186) |
| `fetch_sources.sh DEST` | fetches those files from github.com/wrf-model/WRF at the tag and refuses any that differ from its pin |
| `make_cases.py OUTDIR` | writes the column cases (six regime families on three vertical grids) |
| `stub_wrf.F90` | the four framework names the CAM files USE that carry no physics (see its header) |
| `uwio.F90` | fixture writer (`<stem>.bin` + `<stem>.manifest`) |
| `run_camuwpbl.F90` | the full-driver oracle: CAM_INIT, camuwpblinit, then camuwpbl for N consecutive steps, every step's inputs and outputs recorded |
| `make_spy.py`, `uwspy.F90` | the SPY build: read-only stage recorders around trbintd, caleddy, exacol, zisocl and the compute_vdiff calls |
| `build.sh WRF_SRC BUILD CASES` | builds the four variants below, runs every case file, checks, and collects `BUILD/fixtures` |

## Build variants

* **pristine**: the reference.  gfortran -O0, WRF's preprocessor defines for
  an EM RWORDSIZE=4 build.  Its fixtures are the fixtures of record.
* **o2**: WRF's own gfortran optimisation (-O2 -ftree-vectorize
  -funroll-loops).  Evidence only; the script reports whether it is
  byte-identical to the reference.  On gfortran 15.2 it is NOT on two of the
  three grids: at -O2 the vectoriser routes loops in eddy_diff through
  libmvec (`_ZGVbN2v_cos`, `_ZGVbN2vv_pow`), which are different functions
  from glibc's scalar cos/pow, and folds `x**2._r8` to `x*x`.  The -O0
  build calls the scalar glibc functions everywhere, which is the one
  behaviour a port can reproduce independently of the vectoriser's choices.
* **snan**: the reference with every local real initialised to a signalling
  NaN.  Required byte-identical: no uninitialised local reaches an output on
  the cases run.
* **spy**: the reference with make_spy.py's recorders.  Required
  byte-identical, which is what makes its stage records usable.

## Measured Fortran semantics on this toolchain (gfortran 15.2.0 -O0)

* `MAX(a,b)` returns `a` when `a > b` or `a` is NaN, else `b`:
  max(+0,-0) = -0, max(-0,+0) = +0.  `MIN` is the mirror.
* `x**2._r8` (real exponent) calls glibc `pow`; it is not folded at -O0.
  `x**2` (integer) is `x*x`; other integer exponents call libgcc
  `__powidf2`.
* A literal without `_r8` is single precision: `1.e-6` is the float32
  0x358637BD widened, `3.141592` is 0x40490FD8 widened.

## Running

    bash fetch_sources.sh ~/wrf471
    python3 make_cases.py ~/uw-cases
    bash build.sh ~/wrf471 ~/uw-build ~/uw-cases
