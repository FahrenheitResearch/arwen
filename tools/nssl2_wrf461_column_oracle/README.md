# NSSL option 18: WRF v4.6.1 / WOOF column oracle

Grades WOOF's production `mp_physics=18` step against WRF v4.6.1's own
`nssl_2mom_driver`, bit for bit, over a deterministic set of 96 columns
(12 regimes: convective cores, precipitation shafts, stratiform melting
layers, stable nights with fog, winter snow, shallow ocean cumulus, high
terrain, anvil cirrus, clear air, heavy warm rain, giant hail, and a
thresholds set of trace amounts with mass-without-number and
number-without-mass levels).

Result (2026-10-07, strict build `GPUWM_WRF_EXACT=1`): 0 ULP on all
101,664 output values for dt 12 s, 2.5 s and 30 s at itimestep 2 and for
dt 12 s and 2.5 s at itimestep 1, against the reference below.

## What runs

| file | role |
|---|---|
| `make_columns.py` | builds the 96 columns (seeded; `columns.npz`) |
| `run_gpu_columns.py` | WOOF: a real `DomainState` + `PhysicsDriver`, `gpuwm.core.microphysics.apply` (the forecast loop's call); writes WOOF's outputs and the microphysics-time inputs it prepared (`*.fin.bin`), which both engines then share |
| `column_oracle.F90` | WRF: calls `nssl_2mom_init` / `nssl_2mom_driver` exactly as `module_physics_init.F:4646` and `module_microphysics_driver.F:1952` do for the resolved option-18 defaults, with staggered `w` (`kme = nz+1`), `diagflag`, `ke_diag = nz`, `has_req* = 1`; `column` mode calls the driver once per column, `slab` mode once for all |
| `wrf_bugfix.py` | the reference: corrects four documented WRF defects in the preprocessed source (see below) |
| `build.sh` | links the WRF drivers (stock, scalar, fixed reference, and the fixed dump build) from the WRF serial build's preprocessed `module_mp_nssl_2mom.f90` (pinned source, commit `d66e442f`) |
| `instrument.py`, `instrument_gs.py`, `gs_dump.F90`, `gs_names.py` | dump build only: WRF's slab after sedimentation, after GS and after NUCOND, and GS internal variables by name; `run_oracle.sh` proves its final output is byte-identical to the reference build |
| `compare_bitwise.py` | per-field bit mismatches, max ULP, max abs diff, per regime |
| `compare_stages.py`, `run_gpu_isolated.py`, `big_diffs.py` | attribution: WOOF stages restarted from WRF's own intermediate slab, and the cells whose difference is too large to be rounding |
| `gs_rates_probe.py`, `gs_rates_compare.py` | GS variable-by-variable attribution (WOOF's no-op `NSSL2_GS_DUMP_RATES` hook against the WRF dump) |
| `probes/` | reads WRF's own initialised module values back (`probe.sh` with `scalars.f90`, `arrays.f90`, `consts.f90`, `tables.f90`; `scalars.out`, `arrays.out` are the readbacks the kernels cite), WRF's Lanczos gamma constants (`gamma_consts.py`), the log10 word check (`log10_check.py`), a CPU NVRTC compile check and the device init-table readback |
| `make_fixture.py` | packs the 96 columns (WRF reference side only) into `tests/data/oracles/nssl2_column/` for `tests/test_nssl2_column_oracle.py` |

## The WRF reference build

The WRF serial build's object for this module (`-O2 -ftree-vectorize
-funroll-loops ...`) contains one libmvec call, `_ZGVbN4vv_powf`: GCC
vectorizes `fvent = fschm**(1./3.) * fakvisc**(-0.5)` (`nssl_2mom_gs`,
`module_mp_nssl_2mom.f90:18219`), so in wrf.exe that `powf` comes from the
SIMD library for whole groups of four gather points and from scalar `powf`
for the remainder.  That makes a cell's value depend on its position in the
gather list.  The scalar build is therefore the same preprocessed source
with the same flags plus `-fno-tree-vectorize` (every libm call scalar);
`run_oracle.sh` reports the stock-versus-scalar difference (50 values of
101,664, at most 6 ULP, at dt 12 s).  The stock object per column and over
the whole slab are byte-identical.

WOOF does not copy WRF bugs.  The reference (`fixed`) is the scalar build
with four defects corrected by `wrf_bugfix.py`; WOOF carries the same
corrections default-on (`NSSL2_WRF_BUGFIX_*` in `nssl2_fused_gs.cu`).
`run_oracle.sh` grades WOOF against the fixed reference (the verdict), and
separately against stock WRF; the stock-versus-fixed difference is exactly
the effect of the four defects (468 values at dt 12 s, itimestep 2).

The four defects, as `wrf_bugfix.py` lists them:

| | line | defect | correction |
|---|---|---|---|
| 1 | `:16522` | `raindn(lh) = Min(Max(rimc3, rimdn(lh)), rimc4)` clamps the cloud rime density into the rain rime density it has just computed | clamp `raindn` |
| 2 | `:16806` | the rain-ice mass kernel uses `xdia(lh,3)`, the graupel diameter, beside `dab1lh(li,lr)` | the rain diameter `xdia(lr,3)` |
| 3 | `:16877` | `ELSE ! single-moment rain` closes the outer iacr/eri/temperature test instead of the `ipconc >= 3` test, so the single-moment rate runs for two-moment rain when the outer test fails | pair the `ELSE` with `ipconc >= 3` |
| 4 | `:17761` | the rain-freezing over-depletion limiter multiplies by `qrfrz*dtp/qx` (> 1) | its reciprocal |

At dt 12 s they change 468 of WRF's 101,664 outputs (271 on the first
step), up to 1.3 million ULP.

## Running it (box W1 recipe)

```
build.sh WRF_SRC_ROOT WRF_SERIAL_BUILD_ROOT BUILD_DIR       # CPU, ~1 min
python make_columns.py RUN/columns                           # CPU
# inside a gpu-mutex grant:
ORACLE_BUILD=BUILD_DIR run_oracle.sh RUN 12.0 2 exact        # GPUWM_WRF_EXACT=1
ORACLE_BUILD=BUILD_DIR run_oracle.sh RUN 12.0 2 default      # default arithmetic
```

`exact` sets `GPUWM_WRF_EXACT=1` for the WOOF half only (the strict
options in `gpuwm/wrf_exact.py`); `default` leaves WOOF's normal compile.
