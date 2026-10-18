# Column fixture boundary and native runtime

The column fixture uses `prepare_microphysics_exner`, the production WRF
physics Exner boundary. The earlier fixture used `cupy.power(p/P0, RCP)`.
Regenerate with `gpu_column_oracle.py`, then `make_column_fixture.py`.
The packer now retains `in_pii` and `in_dz`; the GPU test checks those words
before checking outputs. No source columns, batches or output fields were removed.
The boundary change replaced 3,451 recorded output words across the four batches.

The reference maths library is glibc 2.39. The glibc 2.43 host has a
differently rounded `log10f` implementation. On input `0x1.f33a3cp-13f`, glibc 2.39
returns `0xc067e4aa`; 2.43 returns `0xc067e4a9`. The device transcription
`tf_log10f` implements the former. The native driver checks that word before
running, so rebuilding on a changed library fails with its exact reason.
Selecting the old symbol version on glibc 2.43 does not restore the old
implementation: the compatibility wrapper delegates to the new core.
The [glibc 2.43 release announcement](https://sourceware.org/pipermail/libc-announce/2026/000052.html)
also records the compatibility-symbol change.

The regenerated fixture used the source/table hashes in `PROVENANCE-node2.txt`,
GNU Fortran 15.2.0, `-O2 -fno-tree-vectorize`, and an isolated glibc 2.39 /
libgfortran 14 runtime. Ubuntu runtime package receipts:

- `libc6_2.39-0ubuntu8.9_amd64.deb`, SHA256
  `ff5557d99b51f761c4b7c92368b9cc45565eda17df9bf9eb4b134d09825008be`
- `libgfortran5_14.2.0-4ubuntu2~24.04.1_amd64.deb`, SHA256
  `2b05896c477275ad3473bb63a5ac5a5b8a7e4a6ad6ed7cb23c442ccd1abc28f5`

Both come from `https://archive.ubuntu.com/ubuntu/pool/main/g/`, under `glibc/`
and `gcc-14/` respectively. Extract with `dpkg-deb -x`. Link the driver with
`-L<RUNTIME>/usr/lib/x86_64-linux-gnu -Wl,-rpath-link,<RUNTIME>/usr/lib/x86_64-linux-gnu
-Wl,-l:libgfortran.so.5 -Wl,-l:libm.so.6`; run it with that runtime's
`ld-linux-x86-64.so.2 --library-path <RUNTIME>/usr/lib/x86_64-linux-gnu`.
Use a `run_columns_fork` wrapper in the oracle build directory if the generator
runs from a newer Python host. This changes the reference runtime only.

Generator batches, in fixture order:
`--batch all:20:2 --batch real:5:2 --batch all:150:2 --batch all:20:1`.
With `GPUWM_WRF_EXACT=1 GPUWM_NUMERICS=strict`, all four comparisons reported
zero mismatched words and zero ULP. Running the same rebuilt reference under
glibc 2.43 instead gave 51, 9, 48 and 51 differing output words. The native
fork source and its table bytes were unchanged.
