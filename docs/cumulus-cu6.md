# Original cu6 cumulus

This implementation follows WRF v4.6.1 `phys/module_cu_tiedtke.F`, through
`tiecnv`, `cumastr_new` and all their original routines. Select `cu_physics=6`
and `cudt_minutes=0` with a moist state. The WRF namelist importer preserves
selector 6 and writes the every-step cadence explicitly. cu6 is the cumulus
scheme of WRF's `physics_suite = 'CONUS'`, the suite NCAR's v4.4 CONUS
benchmarks run.

## Column parity

The GPU kernel reproduces the original Fortran driver word for word. The
oracle in `tools/tiedtke_wrf461_oracle` compiles the unmodified
`module_cu_tiedtke.F` (SHA256 `3514aaaa...`) with gfortran at `-O0
-ffp-contract=off` and runs it against Ubuntu glibc 2.43, the libm every WRF
oracle in this tree links. Over 72 synthetic columns on 4, 12, 50 and 72
levels and three forcing configurations (864 column evaluations), all six
tendencies, RAINCV, PRATEC and CU_ACT_FLAG match at 0 ULP:

- on the GPU (RTX 5090), with `GPUWM_WRF_EXACT=1` and without it;
- with the columns launched as rows of 12 and of 8 in one call, against one
  Fortran call per row, which is how the forecast adapter walks the domain;
- for the CUDA source compiled for the CPU, and for the
  `-fcheck=all -finit-real=snan` Fortran build.

### The one named residual: a pre-2.41 libm

WRF's only tangent call (F:2939, organized detrainment) is bound by gfortran
to the process libm's `tanf`. WOOF transcribes glibc 2.43's `tanf`, which is
CORE-MATH's correctly rounded one. glibc 2.40 and older carry a `tanf` that is
not correctly rounded. Over every float32 from +0 to fl(pi/2), a superset of
the arguments F:2937 can produce (1,070,141,404 values), WOOF's tangent equals
glibc 2.43's on all of them and differs from Ubuntu 24.04's glibc 2.39 by
exactly 1 ULP on 1,079,656 (0.10 per cent). A WRF built and run on such a host
differs from WOOF through that word alone. On the oracle fixture this moves
RTHCUTEN by at most 6.6e-7 K/s, RUCUTEN by 5.1e-7 m/s2, RQVCUTEN by 3.1e-11
kg/kg/s and RAINCV by 2.6e-6 mm per step; CU_ACT_FLAG never changes. Intel's
compiler uses its own libm and is a different referee again.

The first cu6 lane built its oracle on glibc 2.39 and recorded this libm
difference as a port defect. The receipts now record the libc version and the
libm hash, and `glibc243.sh` runs the oracle under 2.43 on an older host.

## NCAR's CONUS 12 km benchmark

NCAR's v4.4 CONUS 12 km benchmark (425 x 300 x 50 points, 72 s step, 12 h
from 2019-11-26 12Z) runs with `physics_suite = 'CONUS'`, so its cumulus is
cu6. Through `gpuwm run --wrfinput`, with the suite written out as its
explicit schemes, the 12 h run completes on one RTX 5090 at NCAR's 72 s.
Against NCAR's own gnu output at +12 h, with the earlier cu16 substitution
in brackets:

| Field | WOOF cu6 | (WOOF cu16) | NCAR ifort |
| --- | ---: | ---: | ---: |
| Domain-mean RAINC, mm (NCAR 0.081) | 0.080 | (0.435) | 0.082 |
| RAINC RMSE, mm | 0.14 | (2.93) | 0.07 |
| RAINC pattern correlation | 0.94 | (0.20) | 0.99 |
| RAINNC RMSE, mm | 0.36 | (0.78) | 0.12 |
| T2 RMSE, K | 0.30 | (0.64) | 0.06 |
| 10 m wind U/V RMSE, m/s | 0.27/0.30 | (0.60/0.65) | 0.09/0.09 |
| QVAPOR RMSE, g/kg | 0.063 | (0.189) | 0.020 |

These are distances from WRF's own trajectory on one case, not skill against
observations. The ifort column is NCAR's own build-to-build spread. A model
step takes 0.101 s with cu6 against 0.057 s with cu16 (medians over the 600
steps), so cu6 is the slower scheme here.

## Model adapter

The adapter returns the six tendency fields and one rain increment per model
step. It keeps the original vertical reversal, mixing-ratio conversion,
separate advective and PBL moisture forcing, precipitation outputs and
optional tracer writes. WRF calls the driver once per j row, and its first
ascent returns early only when no column in that row is active (F:960). Each
launch walks as many complete rows as fit a 2 GiB arena (NCAR's 12 km CONUS
grid is one launch). The first launch runs every column of a row down one
guessed path, continuation or the early return; the second recomputes, from
pristine inputs and in place, only the inactive columns whose row decided the
other path. Active columns never return early, so either path is theirs, and
the guess moves cost, never a word. Rows are never combined for that
predicate, and no host read decides it: the out-of-bounds refusal below is the
one read per step. On an RTX 5090 a cu6 call on that grid takes 5.8 ms
(captured steps 20, 60 and 100; the first adapter took 50.9 ms).

WRF's `cuflx` can write flux arrays at vertical index 0 when its cloud top is
at level 3. The adapter reports that condition and refuses the column before
its tendencies enter the model. Those undefined Fortran outputs are outside
the bitwise comparison. The checked Fortran oracle reproduces the bounds error.

The cu16 option `ntiedtke_tiedtke_closure=true` remains available. It emulates
two closure choices inside New Tiedtke. It does not run the original cu6 scheme
and cannot serve as a cu6 bitwise comparison.
