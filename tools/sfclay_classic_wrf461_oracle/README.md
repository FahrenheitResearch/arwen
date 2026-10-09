# Classic MM5 surface layer oracle, WRF v4.6.1 (sf_sfclay_physics = 91)

A column oracle for `phys/module_sf_sfclay.F` (SFCLAYINIT + SFCLAY/SFCLAY1D),
graded word for word against WOOF's `kernels/sfclay.cu` option 91
(`kernels/sfclay_classic.cuh`).

## Files

| file | what it does |
| --- | --- |
| `make_columns.py` | writes the input columns (float32 words). Imports nothing from the engine. |
| `run_sfclay_classic.F90` | the Fortran driver: SFCLAYINIT, then SFCLAY with every optional argument an EM build passes, three chained calls per group |
| `build.sh` | pins the WRF tree, builds the driver at -O0 (the oracle), at WRF's stock flags, and against a real wrf.exe's object, and runs all three |
| `validate_sfclay_classic_oracle.py` | runs the real WOOF kernel on the same words and prints per-field ULP, free-running and replayed |

The committed fixture is `tests/data/oracles/sfclay_classic/`; the gate is
`tests/test_sfclay_classic_wrf461_parity.py`.

## Columns

151 columns in 11 option groups, three chained calls each (step k+1 starts
from step k's ZNT, UST, USTM, MOL, HFX, QFX, QSFC, ZOL), 34 output fields:
15,402 words per comparison.

* regimes 1 to 4 over land, regimes 1, 2, 3, 4 over water, and the
  Br-to-zero clamp (previous MOL < 0 on a stable column);
* convective day (Beljaars VCONV from the previous HFX/QFX), calm days and
  nights (WSPD floor 0.1, VCONV only), z/L clamped at -9.9999 (table index
  999/1000), UST < 0.01 (z/L = Br * GZ1OZ0), UST = 0;
* stable nights (Br >= 0.2, incoming ZOL kept), the Launiainen z/L > 0.5
  branch;
* forest roughness 2 m with a 10 m first level (PSIH clamp, PSIT floor 2),
  a 4 m and a 60 m first level;
* snow, sea ice (QSFC recomputed), a polar-night inversion, high terrain
  (62 kPa), desert (MAVAIL 0), tropical (QV 0.02), QV = 0, THX == THGB
  (FLHC zero branch);
* ocean: trades, calm, warm air over cold water, hurricane wind
  (Charnock cap 2.85e-3), tiny roughness, lakes (no salinity factor);
* 56 seeded random columns (land and water, day and night, 65 to 103 kPa);
* option groups: dx 3, 5 (VSGD exactly 0) and 12 km; ISFTCFLX 1 and 2;
  IZ0TLND 1 and 2; ISFFLX 0; ISFTCFLX 1 with IZ0TLND 1.

## Running it

On a box with the pinned WRF source and gfortran (W1: Ubuntu 24.04,
gfortran 13.3.0, glibc 2.39):

```
bash build.sh /work/pverify/wrf-build/src/WRF-4.6.1 OUT /work/pverify/wrf-build/dmpar/WRF-4.6.1
python validate_sfclay_classic_oracle.py OUT/columns.txt OUT/wrf-ck-corrected.txt
GPUWM_WRF_EXACT=1 python validate_sfclay_classic_oracle.py OUT/columns.txt OUT/wrf-ck-corrected.txt
```

`build.sh` refuses a WRF tree that is not commit
`d66e442fccc04111067e29274c9f9eaccc3cef28` with `module_sf_sfclay.F` clean,
and fails if the -O0 object pulls in any libmvec symbol.

## The reference is the -O0 object

At WRF's own flags (`-O2 -ftree-vectorize -funroll-loops`) gfortran 13.3
vectorises SFCLAYINIT's table loop and SFCLAY1D's simple loops (THGB, THCON)
through libmvec: the object imports `_ZGVbN4v_atanf`, `_ZGVbN4v_logf` and
`_ZGVbN4vv_powf`. The object inside the W1 dmpar wrf.exe does the same and
gives the same words as the stock-flag recompile. Against the -O0 object:

* PSIMTB differs in 340 of 1001 entries (max 32 ULP), PSIHTB in 254 (max 3);
* the outputs differ by up to 235 ULP (RMOL), 183 ULP (BR), no regime change.

libmvec's SSE routines carry a 4 ULP contract, and the vectoriser gives them
only to the columns in a full vector body; the loop remainder calls scalar
powf. So a stock wrf.exe gives a column different bits by its position in
the tile. That is the build, not the scheme; the -O0 object is the scheme's
arithmetic.

## What the first run found, and what changed

The shipped kernel (lane base 91de806e2) against the -O0 oracle:

| build | max ULP | worst fields | branch flips |
| --- | --- | --- | --- |
| strict (`GPUWM_WRF_EXACT=1`) | 295 | RMOL 295, ZOL 262, PSIM/PSIH/HFX 225, MOL 205 | none |
| default | 9.3e8 | a THX == THGB column took regime 2 where WRF takes 3 (BR 5.1e-7 for 0) | 3 words of REGIME |

Causes, each fixed in `sfclay_classic.cuh`:

1. THX was `T * (P0/P)**RCP`; WRF writes `T * (P1000mb*0.001/(P/1000.))**ROVCP` (:430-434).
2. RHOX was `PSFC/(R*TV)`; WRF writes `(PSFC/1000.)*1000./(R*SCR4)` (:396, :475).
3. WSPD was `hypotf(U, V)`; WRF writes `SQRT(U*U+V*V)` (:502).
4. z/L from the previous MOL was `(K*G)/THX*...`; WRF writes `K*(G/THX)*...` (:588, :645, :659).
5. PSIHTB used `sqrtf`; WRF writes the real power `(1-16*ZOLN)**0.5`, which is powf (:965).
6. ISFTCFLX=2 used `powf(RESTAR, 0.25)`; WRF writes `SQRT(SQRT(RESTAR))` (:761-762).
7. IZ0TLND=1 used `(-0.40*ZL)/0.07`; WRF writes `-0.40*(ZL/0.07)` (:786).
8. ALOG, EXP, ATAN and REAL**REAL were CUDA's logf/expf/atanf/powf, which are
   different functions; they are now WOOF's own float32 routines (logf, expf,
   powf from `glibc_flt32.cuh`, and an atanf that matches the MYNN unit's).
   A correctly rounded atanf would not do: it differs on 114 of the 1001
   table arguments.
9. The unit compiled with multiply-add contraction in the default build; it
   is now in `_NO_FMAD_MODULES`. The default module now also uses direct
   NVRTC with `--ftz=false`, preserving tiny humidity and negative MOL.

The ensemble batch compile of member-shared sfclay (and YSU) now takes the
unit's own options too (`gpuwm/ensemble/batch_physics.py`); before, a shared
xland/lakemask batch compiled with contraction and so could not equal the
single-member kernel once the unit dropped it.

## Documented, not copied: module_sf_sfclay.F:767

In the ISFTCFLX=2 (Garratt) block WRF writes
`PSIQ10=GZ10OZ0(I)-PSIH(I)+GZ0OZQ`, the 10 m log contrast with the
lowest-level stability term. Every other 10 m quantity in the file uses
PSIH10 (`:717` default path, `:749` ISFTCFLX=1), and the 2 m lines of the same
block use PSIH2. It reaches only CK on water columns with ISFTCFLX=2. WOOF
uses PSIH10. `build.sh` applies `wrf-ck-correction.patch` to a private copy
of the referee. The test grades all 15,402 words against that corrected
reference, including every CK word, with no exclusion.

## Outside the oracle

* ZNT <= 0, ZNT small enough that ZA/ZNT overflows, NaN ZNT: WRF is
  undefined (ALOG(Inf), U10 = Inf/Inf); WOOF's defined rescue is unchanged
  and tested in `tests/test_sfclay.py`.
* ISFFLX = 0: WRF skips the writes of LH, CHS, CHS2, CQS2 and ZNT, so they
  keep their state values; over a WRF run with the namelist constant
  ISFFLX = 0 that is their initial zero. WOOF writes zero. The fixture starts
  them at zero, as WRF's state does.
* XLAND is 1 or 2 in every WRF input; the oracle does not probe 1.5.
