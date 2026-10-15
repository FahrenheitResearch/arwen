# WRF 4.6.1 Lin oracle and CUDA transcription

Build the reference on a sprint CPU box:

    bash tools/lin_wrf461_oracle/build.sh "$WRF_SOURCE_ROOT" /work/port-mp2-purdue-lin/oracle-build

The compiled WRF files must match commit d66e442fccc04111067e29274c9f9eaccc3cef28.
The reference is scalar -O0. The driver covers 72 columns at 40 levels in eight
regimes, including signed zeros and subnormal species. The weather arm is
F_QG=true and F_QNDROP=false. No prognostic droplet or chemistry branch is claimed.
The diagnostic flag also records WRF's native REFL_10CM after the call.

To regenerate the reviewed numerical header, compile the same unmodified
module with gfortran 13 -O0 -fdump-tree-original in the oracle build directory.
Then run `transcribe.py DUMP module_mp_lin.F gpuwm/core/kernels/lin_wrf.cuh`.
The development generator requires pycparser 3.0. It checks the source SHA-256,
refuses unknown lowered expressions, and preserves compiler-rounded constants,
operation order, integer multiply chains and real exponent calls. Strict
operations use explicit IEEE round-to-nearest intrinsics. The common strict
loader supplies no-FMA, no-FTZ and precise division and square root.

The generated header is runtime source. End users need neither gfortran nor
pycparser. WOOF's shared float32 math routines provide EXP and real powers.

Run the mechanical GPU gate through the card queue:

    GPUWM_WRF_EXACT=1 python -m pytest -q tests/test_lin_wrf461_parity.py

Run once without GPUWM_WRF_EXACT and set GPUWM_LIN_PARITY_RECEIPT to record
per-field default arithmetic differences. Default mode has no bitwise claim.
The planted-bit control changes the rain intercept by one float32 ULP and
requires the same output comparison to detect it.
