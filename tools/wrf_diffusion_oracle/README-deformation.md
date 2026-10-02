# Compiled WRF v4.7.1 deformation and mixing coefficients

`deformation_build.py` compiles unchanged bodies from the pinned WRF diffusion,
boundary and physics-preparation modules. The constants module is compiled from
WRF itself. The C ABI adapters retain WRF's `(i,k,j)` layout, vertical and lateral
staggering, dimensions and halo conventions. The build uses gfortran at `-O0`,
with bounds checks and contraction disabled.

```bash
python tools/wrf_diffusion_oracle/deformation_build.py \
  "$WRF_SOURCE/dyn_em/module_diffusion_em.F" \
  "$WRF_SOURCE/share/module_model_constants.F" "$BUILD" \
  --bc "$WRF_SOURCE/share/module_bc.F" \
  --big-step "$WRF_SOURCE/dyn_em/module_big_step_utilities_em.F"

python tools/wrf_diffusion_oracle/deformation_make_cases.py \
  "$WRFINPUT" "$RUST_NETCDF_READER" "$BUILD/oracle.so" "$FIXTURES" \
  --evolved "$WRFOUT"

python tools/wrf_diffusion_oracle/deformation_compare.py \
  "$FIXTURES" "$FIXTURES/deformation-gpu-receipt.json"
python -m pytest tests/test_deformation_wrf471_parity.py -q
```

The 14 fixtures contain seven initialization-state crops and seven operator
probes using actual evolved C-grid winds on initialization-state thermodynamics.
The latter are labeled explicitly; they are not complete evolved model states.
Both groups cover open and periodic boundaries, steep terrain, map factors from
0.25 to 4, zero and near-zero flow, and both hemispheres. The coefficient oracle
calls `calculate_km_kh` directly for km_opt 2, 3 and 4. Both isotropy settings are
tested for km_opt 2 and 3. All arrays are compared by their float32 words.

WRF's interior physics interpolation arrays FZM/FZP are absent from wrfinput.
The physics-preparation adapter receives FNM/FNP for those interior slots. The
diffusion routines tested here consume only the independently extrapolated
surface and top pressure/temperature values, which do not read FZM/FZP.

The production deformation launcher returns D11, D22 and D12. The diagnostic
`deformation_probe.cu` exposes the same inline production D13/D23 functions and
the D33/divergence expressions, so the complete tensor output can be measured
without introducing a second tensor implementation.

Acceptance pins include every GPU array's SHA256 and exact measured difference
statistics against WRF. A changed word requires review. The diagnostic flags
`--no-fma --reference-metrics --reference-order` restore WRF's rounding stages
only to attribute a difference; they never build the production acceptance
receipt. That replay reproduces all seven deformation outputs word for word.

```bash
python tools/wrf_diffusion_oracle/deformation_compare.py "$FIXTURES" \
  "$FIXTURES/deformation-reference-order-receipt.json" \
  --no-fma --reference-metrics --reference-order
python tools/wrf_diffusion_oracle/deformation_seal.py .
```

The portable build receipt preserves source/body hashes, compiler flags and the
library hash. `deformation-sha256sums.txt` seals the fixtures and producing tools.
GPU execution requires the host's OWNER reservation protocol.
