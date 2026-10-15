# Kain-Fritsch WRF v4.6.1 column-oracle fixture

Graded by `tests/test_kf_wrf461_oracle.py`. Produced by `tools/kf_wrf461_oracle`
(see its README for the full recipe) on box W1, 2026-10-07: Ubuntu 24.04,
gfortran 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1), glibc 2.39.

| file | what it is | SHA-256 |
|---|---|---|
| `columns-structured.bin` | 150 input columns (`kf_columns.py structured`, seed 20261007), the exact float32 words both sides read | `076b6488b0ecb1dc515d970f0f07e68203433a06368d453d93dc92a19bed313d` |
| `columns-structured.json` | per-column regime, feedback branch, nz, model top, dx, dt, cudt, STEPCU, adaptive flag | `9c0349fdbf0c6711af87b80c9f10be7280c935b27528100ee0bf580b867b0492` |
| `oracle-structured-O0.out` | every output of `KF_eta_CPS` from the byte-unmodified `phys/module_cu_kfeta.F` at `-O0` | `9746cc490192b3e381c09d2fbc86ca72b553cb64d4305d62acae4dfdc13e5fbf` |
| `columns-real74.bin` | 80 columns from real model input (`kf_columns.py real74 --npz tests/data/kf_real74_12z_columns.npz`): the ERA5-initialised 1974-04-03 12Z warm-sector and stable northern d01 columns, 49 levels, crossed with the five feedback branches, four W0AVG priors and two time steps | `83d7cf4e54fcbb4e9161f4cc3aa0fec29f4b42665ddeea6c82d5cddf741bd00e` |
| `columns-real74.json` | the same per-column record | `f6d4e3dbf9584e09501386549303614922fd3e54e142c8e15b76b5a16a2404c2` |
| `oracle-real74-O0.out` | `KF_eta_CPS` on them, `-O0` | `28315ac761f8b48899e05ad48ab95b9ca90f84a917accfb1025c0db7c50991a3` |

WRF source: `phys/module_cu_kfeta.F` of tag `v4.6.1` (commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`), SHA-256 (LF)
`e6376c2d85c45470f49d545b25d513b5ec111bf36b87beebc740bf42825c6e5f`.

Receipts from the same run:

- The SNaN-initialised build (`-finit-real=snan -finit-integer=-2147483647`)
  wrote a byte-identical file (same SHA-256), so no output depends on memory the
  Fortran never wrote.
- WRF's own stock flags (`-O2 -ftree-vectorize -funroll-loops`, which pull in
  glibc's libmvec `_ZGVbN4v_expf/_logf/_ZGVbN4vv_powf`) wrote a different file
  (`468b2578...4a4b`): 74 of the 150 columns differ from the `-O0` reference.
  The `-O0` build, scalar `expf/logf/powf` only, is the reference.
- No column reached `wrf_error_fatal`, and no column wrote to unit 98 (the
  table out-of-bounds warning and the moisture-budget report).

Outcomes: structured 55 deep, 35 shallow, 60 no convection; real74 40 deep
(the warm-sector column, every branch), 30 shallow and 10 no convection (the
stable column).  For real74 the SNaN build is again byte-identical and the
stock build differs (`e9f53f0f...29c6`).
