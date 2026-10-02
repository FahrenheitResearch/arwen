Compiled WRF v4.7.1 bigstep oracles

Reference commit: f52c197ed39d12e087d02c50f412d90d418f6186. The builders
pin source and constants, extract unchanged routine bodies, keep the actual
argument lists and compile bounds-checked Fortran. Configuration and scalar
indices use WRF's generated modules. Only unused logging services are stubbed
in the coupling harness. The full RK harness links actual native WRF children;
its receipts record that O0/O2 compiler mixture and library hashes.

The fixture is a 12 by 10 patch of a real WRF initialization with all 49 mass
levels. Native U/V/W staggering, hybrid coefficients and stored float32 words
are preserved. Rust decodes NetCDF and crops arrays. Python packages the
result, orchestrates the reference and drives the engine's normal launchers.

Set WRF to a pinned source tree, WRF_BUILD to its matching gfortran build,
BUILD to a private scratch directory and ORACLE to tests/data/wrf471_bigstep.
The fixtures are test data in a source checkout, not package data: under
gpuwm/data/bigstep/oracle they helped push the 2.8.2 pure wheel over the
100,000,000-byte per-file limit, and no runtime code reads them. The harness
modules (gpuwm/verify/bigstep_*_oracle.py) refuse by name in an install that
has no tests/data. The move changed no fixture content: prep-receipt.json and
momentum-device-sm89-nvrtc12.9.86.json were CRLF and are LF under the
line-ending gate that covers tests/data (a CR-only change; each parses to
the same JSON and no manifest or receipt pins either file's digest), and
every other file is byte-identical.
Run CPU work under nice. Compile and generate in this order:

```
nice -n 10 bash tools/bigstep_wrf471_oracle/build_state.sh \
  "$WRFINPUT" "$RW_NETCDF" "$RUSTC" "$PYTHON" "$BUILD/state" "$ORACLE"
nice -n 10 "$PYTHON" tools/bigstep_wrf471_oracle/coupling_build.py "$WRF" "$WRF_BUILD" "$BUILD/coupling"
nice -n 10 "$PYTHON" tools/bigstep_wrf471_oracle/coupling_fixture.py "$ORACLE" "$BUILD/coupling"
nice -n 10 "$PYTHON" tools/bigstep_wrf471_oracle/momentum_build.py --help
nice -n 10 "$PYTHON" tools/bigstep_wrf471_oracle/rk_build.py --help
nice -n 10 "$PYTHON" tools/bigstep_wrf471_oracle/rk_full_build.py --help
nice -n 10 bash tools/bigstep_wrf471_oracle/prep_build.sh "$WRF_BUILD" "$BUILD/prep"
nice -n 10 "$PYTHON" tools/bigstep_wrf471_oracle/prep_generate.py \
  "$ORACLE/state-real.npz" "$BUILD/prep/run_prep" "$ORACLE"
```

The momentum and RK builder CLIs name their source, generated-module and
output arguments explicitly. Their corresponding fixture tools save every
native output, including untouched memory. The committed provenance files,
momentum.md in the report packet and rk_README.md record the comparison scope
and input transformations.

Every GPU invocation needs the machine's OWNER protocol, or the shared
benchmark's full GPU flock. run_node_gpu.sh checks live exclusive claims
under the OWNER flock before appending a short shared claim. GPU scripts and
probes may not bypass that check. The standalone measurement tools are not
authorization to take a card.

Run the four test files:

```
python -m pytest -q tests/test_bigstep_coupling_wrf471_parity.py \
  tests/test_bigstep_momentum_wrf471_parity.py \
  tests/test_bigstep_prep_wrf471_parity.py tests/test_bigstep_rk_wrf471_parity.py
```

CPU provenance gates also run with GPUWM_NO_LOCAL_GPU=1 and -m 'not gpu'.
The four files are on both CPU and GPU battery lists. All GPU nodes are
explicitly marked. Comparisons use the shared FP32 ULP metric, separate raw
bit identity, exact measured pins and complete output SHA256 hashes. No
allclose tolerance accepts a changed result. Causal arithmetic controls are
separate from the unchanged compiled Fortran references and never replace
them. Bounds describe the retained cases, not arbitrary future inputs or
whole-forecast equivalence.
