# P3 (mp_physics = 50) column oracle against WRF 4.6.1

WOOF's P3 device kernels against WRF's own Fortran, compared word for word:
every one of the 22 output fields, every column, every step.

## What runs on each side

**WRF.** `phys/module_mp_p3.F` from WRF v4.6.1 (tag `v4.6.1`, commit
`d66e442f`), sha256 `716950a3...f6a3c6`, compiled unmodified. The driver is
`evidence/p3-fortran-oracle-20260828/harness/p3_ref_driver.f90`, which calls
`p3_init` and then `mp_p3_wrapper_wrf` with the mp_physics = 50 call shape
(one ice category, two-moment ice, specified droplet number). The lookup
table is WRF's `run/p3_lookupTable_1.dat-v5.4_2momI`; `build.sh` checks that
it is the same text as WOOF's packaged copy (WOOF's has CRLF line endings).

Three Fortran builds (`build.sh`):

| arm | flags and C library | role |
|---|---|---|
| `CR` | `-O0 -ffp-contract=off`, with `crlibm_shim.c` linked ahead of libm so `tgammaf` and `log10f` are correctly rounded | **the oracle for the claim** |
| `A` | `-O0 -ffp-contract=off`, this box's glibc 2.39 | reported |
| `B` | WRF's own GNU flags, `-O2 -ftree-vectorize -funroll-loops`, glibc 2.39 | reported |

Why `CR` is the oracle: the Fortran calls six C-library functions (`expf`,
`logf`, `powf`, `exp`, `log10f`, `tgammaf`). glibc 2.39 rounds `tgammaf`
incorrectly on 39.4 % of [0.25, 36] (worst 6 ULP,
`docs/gf_gamma_known_delta.md`) and `log10f` incorrectly on part of its
domain. So "what WRF computes" depends on the C library it was
linked against. `CR` is WRF on a C library that rounds correctly, and it is
byte-identical, on all thirteen p3-fortref fixtures, to the same driver
built with gfortran 15.2 on glibc 2.43 (the committed dumps in
`evidence/p3-fortran-oracle-20260828/dumps/A/`). Matching glibc 2.39's
incorrectly rounded answers would mean copying a C library's error, which
this project does not do (see also `docs/gf_gamma_known_delta.md`).

**WOOF.** `run_woof.py` calls `gpuwm.core.p3_device.run_p3_device`, the
function the forecast step calls, with the wrapper's own locals and the
tables `device_tables()` uploads in a run, for every arm in `ARM_KERNELS`
(`unfused`, `fused`, `levels`, `sedlevels`; the forecast uses `sedlevels`).
Run once with `GPUWM_WRF_EXACT=1` (strict build) and once without (the
shipped default).

## Fixtures (`make_columns.py`)

Analytic soundings on a stretched 50-level grid, built without calling any
P3 or WOOF code. 78 columns:

| regime | columns |
|---|---|
| deep convection | 12 |
| stratiform with a melting layer | 8 |
| stable night, surface inversion, fog and drizzle | 8 |
| snow, ice supersaturation, mixed phase, freezing drizzle | 10 |
| ocean, marine stratocumulus under a capping inversion | 8 |
| high terrain, surface pressure 58 to 76 kPa | 8 |
| anvil, cirrus and homogeneous freezing near 205 K | 6 |
| hail, rime fraction near 1 at 900 kg/m3 | 4 |
| edge cases | 14 |

Edge cases: no condensate; values straddling `qsmall`; rain with zero number;
ice with zero number; ice number at 1e9 /kg; rime mass above total ice and
rime mass with zero volume; small negative values; temperatures at 273.15,
269.15, 258.15 and 233.15 K; 150 % supersaturation; 5 m layers; drops at
10 /kg and at 1e9 /kg; tiny cloud water at the droplet-size clamp; a 180 K
top; melting graupel.

| fixture | columns | dt | steps |
|---|---|---|---|
| `R01-regimes-1step` | 64 regime columns | 20 s | 1 |
| `R30-regimes-30step` | 64 | 20 s | 30 |
| `R60-regimes-dt60` | 64 | 60 s | 10 |
| `E05-edges-5step` | 14 edge columns | 20 s | 5 |
| `S30-subset-30step` | 10 of R30 (committed) | 20 s | 30 |
| `X01-coldstart` | 6, with `th_old = qv_old = 0` | 20 s | 2 |
| `X02-overflow-column` | 1 real-forecast column, 49 levels | 60 s | 2 |

`X01` and `X02` are outside the claim; each documents one divergence that
is a WRF defect, which WOOF does not copy:

* `X01`: on WRF's first call `th_old` and `qv_old` are zero. WRF guards
  the saturation call with `max(t_old, 1.)` (module_mp_p3.F:2328-2330),
  but the saturation mixing ratio at 1 K is zero in float32, so WRF still
  evaluates `0/0` for `sup`/`supi`. WOOF floors `qvs`/`qvi` at `1e-20`
  (gpuwm/core/p3.py).
* `X02`: the one column taken from a real forecast
  (`tests/data/p3_cold_cloud_overflow_column.npz`, the cell that took a
  6 h forecast non-finite). WRF's immersion-freezing product
  `cons6*gamma(7.+mu_c)*tmp1*dum**2` (:2994-2996) overflows float32 on its
  partial product although the answer is an ordinary float, and the
  conservation limiter turns the `+Inf` into NaN. WOOF recomputes only that
  product when it is not finite (gpuwm/core/kernels/p3.cu, "OVERFLOW
  RESCUE"), so every finite value is still WRF's.

The thirteen p3-fortref fixtures (F01-F12, F13) run through the same
oracle as a second set.

## Rerun

On a Linux box with gfortran and a card, from a tree at this commit
(`$T`), with WRF 4.6.1's source at `$WRF`:

```
O=<work dir>
python $T/tools/p3_wrf461_oracle/make_columns.py $O/fixtures
bash   $T/tools/p3_wrf461_oracle/build.sh $WRF $O
bash   $T/tools/p3_wrf461_oracle/run_fortran.sh $O $O/fixtures A B CR
bash   $T/tools/p3_wrf461_oracle/rain_table_dump.sh $WRF $O/rain
T=$T bash $T/tools/p3_wrf461_oracle/run_gpu.sh $O <tag>   # through the GPU mutex
python $T/tools/p3_wrf461_oracle/compare.py $O/fixtures $O/out/CR $O/woof-<tag>/strict/fixtures/sedlevels
```

`compare.py` prints, per fixture, the number of differing words, the
largest ULP distance, the first step that differs and the columns, then
every differing field with its worst element. `SUMMARY` covers the
fixtures inside the claim.

For the legacy set, copy `evidence/p3-fortran-oracle-20260828/fixtures/*`
and `evidence/p3-cuda-verify-20260829/fixtures2/F13*` into one folder,
merge the two `index.json` case lists, and run the same three steps.

The device test `tests/test_p3_wrf461_oracle_gpu.py` replays three of the
fixtures (`R01`, `E05`, `S30`, with their `CR` dumps in
`tests/data/oracles/p3/`) and the thirteen legacy ones on every GPU run.
`tests/test_p3_wrf461_rain_tables.py` holds the start-up rain tables to
WRF's own on any CPU.

## Results (2026-10-07, box W1, RTX PRO 6000 sm_120, tree 051eda99)

**0 ULP against arm CR**: 2,872,468 words (the five claim fixtures plus
the thirteen older ones), all 22 fields, every step, all four device arms,
strict build and default build alike.  Before the two fixes the same
comparison had 162,330 differing words (up to 2,053,916,876 ULP).

Against arm A (same objects, glibc 2.39's `tgammaf`/`log10f`): 230,604 of
2,559,870 claim words differ, step-1 maximum 4,516 ULP (R01, `ni`,
4.1e-4 relative), growing to threshold flips over 30 steps; the older
set differs in 34,656 of 312,598.  Against arm B (WRF's -O2 flags):
295,401 and 34,924.  That gap is the C library's rounding: A and CR link
the same object files.

Defects fixed in WOOF: the rain tables' NumPy libm (now WOOF's own
`expf`/`powf` and a correctly rounded `log10f`;
tests/test_p3_wrf461_rain_tables.py) and the subnormal flush CuPy's
RawModule imposed (now `--ftz=false` through the direct NVRTC route;
tests/test_p3_cuda_gpu.py).  WRF defects documented, not copied: X01
(first-call 0/0) and X02 (immersion-freezing overflow to NaN, 20 NaN
words in WRF).  Open: the CPU reference arm (`p3_backend = "reference"`)
is not bit-identical to WRF (run_reference_arm.py: 5,138 of 54,720 R01
words in one step).
