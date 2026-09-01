# P3 lookup-table oracle

Reproduces the two digests that `gpuwm/core/p3_tables.py` pins as
`FORTRAN_ITAB_SHA256` and `FORTRAN_ITABCOLL_SHA256`, so
`tests/test_p3_table_refusals.py` rests on a repeatable measurement
rather than on a number somebody asserted once.

It runs WRF's own `p3_init` and dumps the module arrays it fills.

## What it needs

- `gfortran` (measured with 15.2.0 on Linux; no GPU, no CUDA, no MPI).
- WRF's `phys/module_mp_p3.F`, supplied by you from your own WRF
  checkout. **It is deliberately NOT vendored here.** WRF is NCAR/UCAR
  public domain and is attributed in `NOTICE`, so vendoring would be
  permitted; gpuwm cites it by line number instead, the same way every
  other transcription in this tree does.
  The script refuses any file whose SHA-256 is not
  `716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6`
  (7,391 lines, `version_p3 = '4.5.2'`, byte-identical in WRF v4.6.1,
  v4.7.1 and v4.8.0).

Do not substitute the P3 scheme's own upstream distribution. It carries
no licence, so gpuwm cannot use it as a source for anything.

## What it does

`run.sh` copies the authority, appends `dump_append.f90` before
`END MODULE microphy_p3` and adds one `public ::` line -- and nothing
else, so the read code under measurement is upstream's bytes; it prints
the resulting diff so you can see that for yourself. It then builds
`drv.f90` at both -O0 and -O2 and runs

    p3_init(lookup_file_dir, nCat=1, trplMomI=.false., model='WRF')

writing `itab`, `itabcoll`, `vn_table`, `vm_table`, `revap_table`,
`mu_r_table`, `itabcolli1` and `itabcolli2` as raw little-endian REAL(4)
in Fortran storage order.

`compare.py` reads those back with `order="F"`, compares against
`gpuwm.core.p3_tables` with `tobytes()`, and reports absolute, relative
and ULP error for anything that differs.

## What it measured, 2026-08-28

- `itab` (5,4,50,14) and `itabcoll` (5,4,50,30,2) parsed by
  `load_lookup_table_1` are BYTE-IDENTICAL to `p3_init`'s arrays, at -O0
  and -O2, and whether `p3_init` read the packaged CRLF table or WRF's
  own LF `run/` copy.
- `generate_rain_tables` is NOT byte-identical: `vn_table` 1,150/3,000
  entries differ (max 5 ULP), `vm_table` 940 (max 4 ULP), `revap_table`
  110 (max 21 ULP, max relative 2.2e-06). The cause is host libm, not
  transcription -- `lamr` is byte-identical, and over the loop's own
  10,000 arguments numpy's float32 `exp` differs from glibc `expf` on
  5,824 of them.
- At `nCat=1`, table 2 is never opened: `p3_init` returns `stat=0` with
  no `p3_lookupTable_2.dat-v5.3` present anywhere. The same executable
  with `nCat=2` hard-fails at `module_mp_p3.F:479` trying to open it.

## Usage

    ./run.sh /path/to/phys/module_mp_p3.F /path/to/table/dir /path/to/outdir
    python compare.py /path/to/outdir
