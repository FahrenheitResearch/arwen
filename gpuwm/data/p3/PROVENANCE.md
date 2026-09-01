# P3 lookup table provenance

`tables/p3_lookupTable_1.dat-v5.4_2momI` is WRF's own
`run/p3_lookupTable_1.dat-v5.4_2momI` **with CRLF line endings**.  P3
v4.5.2 intends table-1 version 5.4_2momI (phys/module_mp_p3.F:138), and
`phys/module_mp_p3.F` is byte-identical across WRF v4.6.1, v4.7.1 and
v4.8.0 (sha-256
`716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6`), so
this table is current for the newest WRF as well as for the frozen v4.6.1
reference bundle this copy came from.

WRF is NCAR/UCAR public domain and is attributed in `NOTICE`.

## The CRLF divergence, measured 2026-08-28

This file is NOT byte-identical to WRF's, and the earlier wording here
that said it was is corrected:

| | bytes | sha-256 | line endings |
| --- | ---: | --- | --- |
| WRF v4.6.1 `run/` | 1,606,038 | `be1ab6fb03481e376e47c6c79d808af5d8ab069f2b242931e9c54801bad4ae84` | LF |
| packaged here | 1,637,040 | `5cb47f2ad15b726abeacac21b5877c737178526051661473a465315348867c2f` | CRLF |

The difference is exactly 31,002 bytes for exactly 31,002 records: one CR
per line, added when the file was copied through a Windows tool.  Both
files carry the same 167,002 tokens.

**It changes no number.**  WRF's own `p3_init` was compiled with gfortran
15.2.0 and run against each file in turn; the `itab` and `itabcoll`
module arrays it produced were byte-identical, because a Fortran
list-directed READ treats the CR as trailing blank.  Our parser produces
those same bytes from either file too (the digests are pinned as
`FORTRAN_ITAB_SHA256` / `FORTRAN_ITABCOLL_SHA256` in
`gpuwm/core/p3_tables.py`, and `tests/test_p3_table_refusals.py` holds
them).

**It does have one consequence.**  `GPUWM_P3_TABLE_ROOT` invites an
operator to point at a mirror, and the most obvious mirror on any machine
with WRF on it is WRF's own `run/` directory -- which this pin refuses.
The refusal names the CRLF fact when it recognises the upstream digest,
so the operator is not left hunting for corruption.  Repinning to the
upstream bytes instead is a RESTART-IDENTITY decision, not a loader one:
the digest is written into every mp=50 checkpoint's setup identity
(`gpuwm/io/restart.py::_p3_setup_identity`), so moving it would make
every existing mp=50 checkpoint refuse to resume.

## What is not packaged

The 3-moment table (`p3_lookupTable_1.dat-v5.4_3momI`) and the
multi-category interaction table (`p3_lookupTable_2.dat-v5.3`) are NOT
packaged: they belong to the unported mp_physics=53 / 52 options, which
gpuwm refuses by name.

Table 2's absence is not a gap.  Measured against the compiled authority
on 2026-08-28: at `nCat=1`, `p3_init` completes and returns `stat=0` with
no table-2 file present anywhere, because the read is inside
`IF_NCAT: if (nCat>1)` (module_mp_p3.F:475).  At `nCat=2` the same
executable, same directory, hard-fails at :479 with
`Cannot open file '.../p3_lookupTable_2.dat-v5.3'`.
