# Kain-Fritsch (cu_physics = 1) column oracle, WRF v4.6.1

Grades WOOF's Kain-Fritsch GPU path word for word against the byte-unmodified
WRF v4.6.1 Fortran, `phys/module_cu_kfeta.F` (tag `v4.6.1`, commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`, SHA-256 of the LF file
`e6376c2d85c45470f49d545b25d513b5ec111bf36b87beebc740bf42825c6e5f`;
`build.sh` refuses any other bytes).

## What is compared

Each column is one call of `KF_eta_CPS` on a 1 x 1 tile, exactly as
`module_cumulus_driver.F:1015-1050` calls it, with the
`share/module_model_constants.F` constants: the W0AVG running-mean update,
`KF_eta_PARA`, and the tendency mapping with the F_QR fold and the
F_QI/F_QS/warm_rain branches. Every output word is compared:

RTHCUTEN, RQVCUTEN, RQCCUTEN, RQRCUTEN, RQICUTEN (with F_QI), RQSCUTEN (with
F_QS), the advanced W0AVG, RAINCV, PRATEC, NCA (where WRF sets it), CUTOP,
CUBOT, SHALL, CLDFRA_DP_KF, CLDFRA_SH_KF, QC_KF, QI_KF.

WOOF's side is its real GPU path: `gpuwm.core.kf.advance_w0avg` and
`gpuwm.core.kf.launch_kf` on the same float32 input words.

## Files

| file | role |
|---|---|
| `build.sh` | pins the source, builds three Fortran oracles (below), runs them on column files |
| `stub_wrf_error.F90` | the only module KF uses; `wrf_error_fatal` records instead of stopping |
| `run_kf_oracle.F90` | the column driver (stream binary in, stream binary out) |
| `kf_columns.py` | column generator (`structured`, `sweep`, `real74`) and the binary readers |
| `compare_woof.py` | runs WOOF's GPU path and grades it (JSON report) |
| `compare_baseline.py` | grades the pre-port kernel (run against a tree that still has it) |
| `e45_probe.F90` | measures the word gfortran folds for `EXP(-4.5)` in PROF5 |

The three Fortran builds of the one unmodified source:

- `O0`: gfortran `-O0`, scalar glibc `expf/logf/powf`. **The reference.**
- `stock`: WRF's own GNU flags for `phys/` (`-O2 -ftree-vectorize -funroll-loops`).
  It links glibc's libmvec (`_ZGVbN4v_expf`, `_ZGVbN4v_logf`, `_ZGVbN4vv_powf`), so
  vectorised loops round differently: on the structured set it differs from `O0`
  in 74 of 150 columns. WRF as built is therefore not one bit pattern; `O0` is.
- `snan`: `-O0 -finit-real=snan -finit-integer=-2147483647`. Byte-identical to
  `O0` on every set, so no output depends on memory the Fortran never wrote.

## Rerun (box W1 or any Linux box with gfortran 13.3 / glibc 2.39)

```bash
# 1. WRF source (pristine; only module_cu_kfeta.F is read)
git clone --branch v4.6.1 https://github.com/wrf-model/WRF.git WRF-4.6.1
# 2. columns
python tools/kf_wrf461_oracle/kf_columns.py structured cols-structured.bin
python tools/kf_wrf461_oracle/kf_columns.py sweep cols-sweep.bin --blocks 64 --per-block 64
python tools/kf_wrf461_oracle/kf_columns.py real74 cols-real74.bin --npz tests/data/kf_real74_12z_columns.npz
# 3. Fortran (CPU only, about a minute)
bash tools/kf_wrf461_oracle/build.sh WRF-4.6.1 oracle cols-structured.bin cols-sweep.bin cols-real74.bin
# 4. WOOF on one GPU; strict build, then default arithmetic
GPUWM_WRF_EXACT=1 python tools/kf_wrf461_oracle/compare_woof.py cols-sweep.bin oracle/cols-sweep.O0.out sweep-strict.json
python tools/kf_wrf461_oracle/compare_woof.py cols-sweep.bin oracle/cols-sweep.O0.out sweep-default.json
```

The structured and real74 sets (inputs and `O0` outputs) are committed under
`tests/data/oracles/kf` and graded by `tests/test_kf_wrf461_oracle.py`.

## Result (W1, RTX PRO 6000 Blackwell, sm_120, NVRTC 12.9, 2026-10-07)

| set | columns | deep / shallow / none | strict build | default build |
|---|---:|---|---|---|
| structured | 150 | 55 / 35 / 60 | 0 of 71,968 words differ | 0 of 71,968 |
| sweep (seed 20261007) | 4,096 | 1,270 / 793 / 2,033 | 0 of 1,848,256 | 0 of 1,848,256 |
| real74 (ERA5 1974-04-03 12Z) | 80 | 40 / 30 / 10 | 0 of 38,112 | 0 of 38,112 |

The pre-port kernel, graded by `compare_baseline.py` on the same columns,
matched WRF on none of the 1,793 convecting columns it could run (it had no
F_QR fold, so `no_qr` columns were not run): max 2.0e9 ULP in RTHCUTEN.
