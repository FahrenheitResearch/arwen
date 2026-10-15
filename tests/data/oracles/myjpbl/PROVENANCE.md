# myjpbl-wrf461.npz

Produced by `bash tools/myjpbl_wrf461_oracle/build.sh /work/pverify/wrf-build/src/WRF-4.6.1 BUILD`
on W1 (AWS us-west-2, Xeon Platinum 8559C), 2026-10-07.

- WRF: wrf-model/WRF v4.6.1, commit `d66e442fccc04111067e29274c9f9eaccc3cef28`,
  `phys/module_bl_myjpbl.F` and `share/module_model_constants.F` unmodified
  (SHA-256 of the LF checkout in `receipts/oracle-sha256sums.txt`).
- Compiler and C library: GNU Fortran 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1),
  glibc 2.39 (Ubuntu 24.04). Reference objects at `-O0`.
- Fixture SHA-256: `6d799d3536495ab7c23b19c9a6e5134a6019cb243da7eaa9f2ea6cc179268f73`.
  A second build on the same box reproduced it byte for byte.

Layout: one array per record, keyed `<case>/<name>`. Level fields are C-order
`(nz+1, ncol)` (Fortran `(ncol, nz+1)`), surface fields `(ncol,)`, `kpbl` int32.
`<case>/meta` is `[ncol, nz, nstep, flqi]`, `<case>/meta_dt` the call's DT,
`<case>/const_a` and `const_b` the module's folded PARAMETER words.
`s<N>_in_*` is everything call N was given, `s<N>_out_*` everything it returned.

Receipts (`receipts/`):

- `libmvec-report.txt`, `undefined-O0.txt`, `undefined-O2vec.txt`: the reference
  object calls scalar `expf` and `powf` only; WRF's own -O2 build of the module
  calls the same two (no vector math).
- `o2-equality.txt`: WRF's -O2 -ftree-vectorize -funroll-loops build writes
  byte-identical outputs on every case.
- `coverage-myjpbl.txt`: gcov's never-executed lines. In MYJPBL's call tree only
  PRODQ2's `ELOQN <= EPS1` arm (1145-1146) is unreached.
- `libm-sweep-strict.json`, `libm-sweep-default.json`: the kernel's EXP and
  `**CAPA` words against the host C library over all 2**32 inputs (0 mismatches
  under the strict build; under default arithmetic only subnormal bases differ
  for the power, which MYJ never passes).
