# Compiled WRF v4.7.1 momentum oracle

The builder byte-extracts `horizontal_pressure_gradient`, `coriolis`, and
`curvature` from WRF v4.7.1. Their numerical bodies and complete argument lists
are unchanged. It compiles WRF's actual `module_model_constants.F`, with WRF's
word-size and ARW preprocessor defines. The measured fixtures use the generated
WRF `module_configure.mod` type. Every configuration field read by the routines
is checked against the builder's field inventory and initialized by the driver.

The reference uses gfortran 15 at `-O0 -ffp-contract=off
-fno-tree-vectorize -fcheck=all`. The build receipt records complete source
hashes, exact extracted-body hashes, compiler flags, and the configuration module
hash. The source must be the pinned tag; the builder rejects changed bytes.

```bash
nice -n 10 python tools/bigstep_wrf471_oracle/momentum_build.py \
  "$WRF_SOURCE/dyn_em/module_big_step_utilities_em.F" \
  "$WRF_SOURCE/share/module_model_constants.F" "$BUILD" \
  --config-module "$WRF_BUILD/frame/module_configure.mod"
PYTHONPATH=. nice -n 10 python tools/bigstep_wrf471_oracle/momentum_fixture.py \
  tests/data/wrf471_bigstep/state-real.npz "$BUILD" \
  tests/data/wrf471_bigstep
```

The shared real state is a Rust-decoded 12 by 10 horizontal crop of all 49
native mass levels from a real WRF initialization. Primitive words are copied
without conversion. Eight cases retain those fields or apply a stated change:
the original state, the southern hemisphere, 0.3 m/s interior vertical motion,
3.3 km added terrain relief, map factors between 0.35 and approximately 2.85,
periodic boundaries, near-zero perturbations, and zero perturbations and winds.
Horizontal spacing is the input file's 3000 m.

The driver allocates WRF arrays `(i,k,j)` with two horizontal halo rows on each
side. Domain indices are `ids=jds=kds=1`, `ide=nx+1`, `jde=ny+1`, `kde=nz+1`.
Native output arrays are `(nz,ny,nx+1)`, `(nz,ny+1,nx)`, and
`(nz+1,ny,nx)`. Every word of all three arrays is measured, including excluded
open boundary faces and the unchanged bottom and top w tendencies. All memory
outside those native output arrays must remain byte-identical to its initial
value. Bounds checking is active in both the routines and driver.

The production launch paths are `dycore._launch_slow_pgf` and
`dycore.launch_coriolis_curvature`. The fused rotation/curvature launch is
isolated by zeroing uncoupled u/v for the Coriolis control or f/e for the
curvature control. The full real state also faces the two consecutive WRF calls
as `combined`, so isolation does not replace the realistic combined comparison.

The pressure launch stores total pressure. The oracle's main comparison uses
exactly the launch's float32 total-minus-base perturbation pressure. Each fixture
also retains the original WRF perturbation pressure, records the representation
loss, and runs an auxiliary `original_pressure` Fortran comparison. The diagnostic
FP32 operation-tree control identifies reassociation; it is never used in place
of the compiled WRF reference.

GPU execution must be under the host's GPU ownership protocol. The measurement
script writes output NPZ files, a per-field table, diagnostic results, and a
device/source receipt. The packaged table is an observation asserted for exact
equality, including differing-word counts. No tolerance is used.

```bash
python -m pytest -q tests/test_bigstep_momentum_wrf471_parity.py
```

Coverage is nonhydrostatic, isotropic map-factor ratios, and the normal
Lambert curvature branch. The declared `map_proj=6`/polar tangent branch is
outside this launch's input contract, as recorded in `coriolis_map.cu`.

The opt-in WRF-exact branches (merged 1efb5a415) moved `dycore` and
`coriolis_map` in `momentum-device-receipt.json` and
`momentum-device-sm89-nvrtc12.9.86.json` (`e318baa1` and `79813843` to `0bf6ab99`
and `27e16185`) with no capture, because the default compile did not move: from
the capture tree (a4c70a270) to the merged head every entry compiles to identical
PTX for compute_89, compute_90 and compute_120 under the loader's and the
RawModule options, with NVRTC 13.4 and 12.9
(`tools/kernel_ptx_identity/receipts/oracle-bigstep-2.8.2-nvrtc{13.4,12.9}.json`).
`rk-ulp-table.json` keeps its snapshot of the tree it was measured in.

The bandwidth implementation changes default PTX and is re-frozen on actual recorded-word reproduction. All 96 momentum outputs and 602,816 binary32 words per card reproduce their complete recorded hashes and WRF discrepancy metrics on the RTX 5090 and RTX 4090. The 4090 replay uses the recorded NVRTC 12.9.86 library. The current dycore and coriolis_map assembled source pins are `433ab4ba` and `505e5ce3`. Readings: `receipts/rtx5090-bw-momentum-word-reproduction.json` and `receipts/rtx4090-nvrtc12.9.86-bw-momentum-word-reproduction.json`. Original measurement and reference words are retained; the source-pin move is supported by these device executions.
