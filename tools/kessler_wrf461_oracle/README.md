# Kessler (mp_physics=1) WRF v4.6.1 column oracle

`run_kessler.F90` calls SUBROUTINE kessler of the pinned, byte-unmodified
`phys/module_mp_kessler.F` (WRF v4.6.1, commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`) with WRF's own
`module_model_constants` parameters, exactly as
`module_microphysics_driver.F:983-996` passes them. It does no Kessler
arithmetic of its own.

## Build and run (Linux, gfortran, CPU only)

```bash
tools/kessler_wrf461_oracle/build.sh "$WRF_TREE" "$BUILD_DIR"
```

- `make_columns.py` writes the inputs: 104 columns x 50 levels in 13 regimes,
  five dt cases (5, 18, 60, 120, 300 s; 25 calls in all).
- The reference of record is compiled `-O0 -ffp-contract=off`: scalar
  `expf`/`powf`, no FMA. build.sh fails if the object pulls a libmvec `_ZGV`
  symbol or contains an FMA instruction.
- An evidence-only build with WRF's stock physics flags
  (`-O2 -ftree-vectorize -funroll-loops`) is run beside it. It vectorises the
  loops onto libmvec and does not reproduce the -O0 words; `libmvec-report.txt`
  records that.

## Grade the GPU kernel

```bash
GPUWM_WRF_EXACT=1 python tools/kessler_wrf461_oracle/compare_gpu.py "$BUILD_DIR"   # strict
python tools/kessler_wrf461_oracle/compare_gpu.py "$BUILD_DIR"                      # default
```

It runs the production launcher (`gpuwm.core.microphysics.launch_kessler`)
free running and per-call replay, and prints per field the count of words
that are not bit-identical, the max FP32 ULP and the max absolute difference.
`--write-fixture DIR` writes the `.npz` pair the pytest gate
`tests/test_kessler_wrf461_parity.py` reads from `tests/data/oracles/kessler/`.
