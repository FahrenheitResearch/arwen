# Original cu6 oracle

This oracle builds WRF v4.6.1's original `phys/module_cu_tiedtke.F` and
`share/module_model_constants.F`. It calls the original driver without changing
its arithmetic or replacing a physics routine. `oracle.F90` supplies a C binding
and makes synthetic input columns in Fortran. Every field reaches the candidate
as the same float32 words. No NetCDF or GRIB processing is involved.

The source SHA256 is
`3514aaaaf6084edc0f6a6b2097e34607562f0b6f12934cc3c675a9e651566999`.
The arithmetic target is gfortran REAL32, `-O0 -fno-fast-math
-ffp-contract=off`, with `AARCH64_X86_CORRECTNESS_FIX` undefined, running
against Ubuntu glibc 2.43 (libm.so.6 sha256 `1c488db6...`), the libm every
other WRF oracle in this tree links. The alternative double-tangent
preprocessor arm is not the target of this port.

The libm matters because gfortran binds the one `tan()` at F:2939 to the
process libm's `tanf`. glibc 2.41 and newer ship CORE-MATH's correctly
rounded `tanf`, which `glibc_trig_flt32.cuh` transcribes. glibc 2.40 and
older do not: `tan_domain_check.py` sweeps every float32 from +0 to fl(pi/2)
(1,070,141,404 arguments, a superset of what F:2937 can produce) and finds 0
differences against 2.43 and 1,079,656 one-ULP differences against Ubuntu
24.04's 2.39. That single word reaches every tendency through organized
detrainment. The first cu6 lane built its oracle on 2.39 and recorded that
libm difference as a port defect. Every receipt now records the libc version
and the SHA256 of the libm mapped into the process.

On a host older than 2.43, `glibc243.sh SYSROOT CMD...` fetches the pinned
libc6 2.43 package once into a lane-owned SYSROOT and runs CMD under its
loader. On an authorized CPU host, from the engine root:

```sh
sh tools/tiedtke_wrf461_oracle/build.sh "$OUT/oracle"
G="tools/tiedtke_wrf461_oracle/glibc243.sh $OUT/sysroot-2.43"
$G python tools/tiedtke_wrf461_oracle/check.py --host --build "$OUT/oracle" --receipt cpu.json
$G python tools/tiedtke_wrf461_oracle/tan_domain_check.py "$OUT/oracle" tan-domain.json
python tools/tiedtke_wrf461_oracle/thermo_check.py "$OUT/oracle" thermo.json
python tools/tiedtke_wrf461_oracle/guard_check.py "$OUT/oracle" guard.json
```

Under a GPU mutex grant, once with `GPUWM_WRF_EXACT=1` and once without:

```sh
$G python tools/tiedtke_wrf461_oracle/check.py --gpu --build "$OUT/oracle" --receipt gpu.json
```

The GPU check also launches the 72 columns as rows of 12 and of 8 columns in
one call and compares them with one original driver call per row, which is
how the forecast adapter walks many j rows per launch.

`--host` runs the unchanged CUDA transcription with CPU intrinsic shims. That
checks the transcription, not GPU execution. `--native-tan-control` substitutes
the CPU target's tangent call only in a diagnostic build. It is not a forecast
backend or a fix. The production GPU source calls WOOF's own routines.

Each comparison evaluates 72 columns, twelve in each of six input families:
convective, shallow, stable, tropical ocean, high terrain and edge conditions.
The four vertical grids have 4, 12, 50 and 72 levels. Three configurations test
normal forcing, first-step forcing suppression, STEPCU=3 and inactive optional
tracer outputs. The full matrix is 864 column evaluations. All six tendency
fields, both precipitation fields and CU_ACT_FLAG are compared word for word.
Nonfinite outputs fail even when both implementations produce identical words.
Every configuration includes a planted 1 ULP change.

`plant_check.py BUILD RECEIPT.json` also compiles a diagnostic copy of the
actual column driver with one `nextafterf` step planted in the first column's
bottom-level heat tendency. It uses the CPU native-tangent control to isolate
that change. The comparison must find exactly one differing word at 1 ULP and
no difference in any other output field. It does not edit the production source.

The checked oracle also runs with `-fcheck=all -finit-real=snan`. Separate batch
and single-column runs record the original driver's horizontal coupling. These
are diagnostics: WRF's row-wide early return can make a single-column result
differ from that column in a populated row. The candidate preserves that
predicate, rather than treating column isolation as a requirement.

The controlled `cuflx` probe reproduces an original index-zero write:

```sh
python tools/tiedtke_wrf461_oracle/guard_check.py /workspace/sol/pool-tiedtke-cu6-port/oracle --reference
```

It exits with `Index '0' of dimension 2 of array 'pdmfup' below lower bound of 1`
at source line 2289. The candidate checks array extents and reports the unsafe
column before the model can use its tendencies. This probe drives the actual
Fortran and transcribed `cuflx` routines with cloud top 3; it is not a forecast.

To regenerate the source on a SHA-pinned local download:

```sh
python tools/tiedtke_wrf461_oracle/transcribe.py module_cu_tiedtke.F gpuwm/core/kernels/tiedtke_routines.cuh
```

Every generated statement carries the original source line. Floating operations
are explicit rounded operations. Integer powers use multiplication chains;
real powers other than 2.0, -1.0 and 1.0 use the math power routine. The original
cu6 source has only integer squares. Unsupported Fortran syntax or a different
source hash fails the generator.

No script here downloads or runs a forecast automatically.
