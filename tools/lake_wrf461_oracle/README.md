# WRF CLM lake column oracle

The authority is the byte-unmodified WRF v4.6.1
`phys/module_sf_lake.F`, compiled with `EM_CORE=1`. It contains the complete
CLM lake model, including its iterative surface fluxes, ten water layers,
ten sediment layers, phase change and up to five snow layers. The native
WRF `lakeini` and `Lake` drivers are called directly. `stub_wrf.F90` replaces
only diagnostics; a WRF fatal condition still terminates the oracle.

`upstream/module_model_constants.F` is also unchanged. The source and fixture
SHA-256 values are in `gpuwm/data/lake/oracle/manifest.json`. The WRF public
domain notice is retained in `NOTICE.txt` and in the installed package's
`licenses/LICENSE-WRF-public-domain.txt`.

`translate.py` is a closed statement translator. It rejects unknown syntax
and a changed source digest. It preserves Fortran array lower bounds,
default REAL constants, double-precision lake internals, the FP32 WRF driver
boundary and operation order. Generated statements name their original
WRF line. All divisions use a helper that selects `__fdiv_rn` or
`__ddiv_rn`. The WRF driver's single-precision power uses the engine's
existing glibc-compatible helper. CUDA compilation disables FMA contraction.

Binary64 `exp`, `log`, `pow`, `sin`, `atan` and `log10` use CUDA device libm;
the native Fortran oracle uses host libm. This is a declared arithmetic
distinction: universal bitwise identity of those library functions is not
established. The stored cold-start values and every persisted state and
surface output in the supplied 300-step column histories match exactly.

The C++ column arrays hold one column. WRF's unused, unassociated
biogeochemistry pointer declarations are omitted. Diagnostic printing is
replaced by a source-line error code, and a failed column is not committed.
Those are implementation differences, not alternate lake physics. No
frozen-temperature, prescribed-SST or reduced-layer model replaces WRF.

`abs_control.F90` records the native Fortran ABS words for signed zeros,
the smallest positive and negative subnormals, finite values, infinities,
and signed quiet and signaling NaNs. Its C++ companion calls the production
helper. The CUDA control uses the same production helper and compiler flags.
Floating ABS clears the sign bit directly on CUDA so FTZ cannot erase the
subnormal controls or change a NaN payload.

From the repository root, with a Linux Fortran/C++ compiler and NumPy:

```sh
python tools/lake_wrf461_oracle/translate.py
python tools/lake_wrf461_oracle/generate_harness.py
bash tools/lake_wrf461_oracle/build.sh "$PWD/lake-build"
python tools/lake_wrf461_oracle/compare.py "$PWD/lake-build" --steps 300
```

The build puts temporary files under its own build directory. `compare.py`
compares every stored state and output word of native WRF and native C++.
Its compressed fixtures contain the original inputs, cold start and every
one of 300 steps for twelve independent columns. They cover warm/cold
water, every snow-layer count, shallow/deep lakes, soil category 14's WRF
remapping, ice fractions on both sides of 0.5, low wind and zero latitude.
Separate initialization controls select `use_lakedepth=0` with default
depths 50, 0 and -1 m. WRF's nonpositive defaults use its reference geometry,
whose depth is 1 m in this pinned source.

With a GPU available, use the actual runtime loader and its production
source composition, including the common preamble and declared headers:

```sh
PYTHONPATH=. python tools/lake_wrf461_oracle/compare.py "$PWD/lake-build" \
  --cuda gpuwm/core/kernels --steps 300
PYTHONPATH=. python tools/lake_wrf461_oracle/compare.py "$PWD/lake-build" \
  --cuda gpuwm/core/kernels \
  --multi-reference gpuwm/data/lake/oracle/columns.npz --devices 2
pytest -q tests/test_lake_contract.py tests/test_lake_gpu.py
```

The multi-device check assigns disjoint columns to the requested devices,
launches every device before reading results and compares all steps to the
same native Fortran words. It records the exact assembled source digest,
compile options and CUDA frame attributes. This checks column arithmetic
and decomposition, not forecast skill or full-domain throughput.
