# Grell-Freitas gamma: a DELIBERATE divergence from WRF

Read this before quoting any GF parity number, and before treating a
non-bitwise `fzu`, `xmb`, `raincv` or `pratec` against a WRF capture as a
port defect. It is not one. It is this.

**Status:** landed in ArWen 2.6.6, in the shipped default of `cu_physics = 3`.
The physics registry's `cumulus_options["grell-freitas"]` warnings cite this
file, as do `gpuwm/core/kernels/gf.cu`, `gpuwm/core/kernels/glibc_flt32.cuh`,
`gpuwm/core/gf.py`, `gpuwm/verify/gf_deep_ref.py`,
`tests/test_gf_gamma_correctly_rounded.py`, `tests/test_gf_deep_cuda.py`,
`tests/test_gf_shallow_cuda.py`, `tests/test_gf_gfdrv_cuda.py`,
`tools/gf_wrf461_oracle/gf_host_parity.py`, `docs/public/PHYSICS.md` and
`docs/manual/03-physics.md`.

**One paragraph.** WRF's `get_zu_zd_pdf_fim` normalises the beta-function
mass-flux shape with `fzu = gamma(alpha+beta)/(gamma(alpha)*gamma(beta))`, and
gfortran binds the F2008 `gamma()` intrinsic to glibc's `tgammaf`. glibc's
`tgammaf` is **not correctly rounded**: measured against a 113-bit oracle over
all 59,768,833 float32 arguments of `[0.25, 36]`, it returns the wrong float32
on **23,575,230 of them (39.4440 %)**, worst **6 ULP**; `tgammaf(4.0f)`
returns 6.00000048 rather than 6. ArWen used to reproduce those wrong answers
bit for bit, by transcribing glibc's own source. That source is
**LGPL**-2.1-or-later and had to go. What replaced it is **correctly rounded**
on every one of those 59,768,833 arguments, so ArWen and WRF now disagree
about `fzu` — deliberately, with ArWen on the correct side.

---

## 0. What actually changed

| | through 2.6.5 | 2.6.6 onward |
| --- | --- | --- |
| `gfk_tgamma` | transcription of glibc `e_gammaf_r.c` + `gamma_productf.c` | ArWen's own gamma, `glibc_flt32.cuh` |
| licence of that code | LGPL-2.1-or-later, FSF copyright | Apache-2.0, original work |
| answer | glibc 2.39's word, right or wrong | the correctly rounded word |
| `fzu` vs WRF | bitwise | differs; bounded and measured below |
| CUDA vs ArWen's own CPU reference | differed on 23,575,230 of 59,768,833 arguments | **identical on all 59,768,833** |
| `gfk_lgamma_pos`, `gfk_expm1`, `gfk_exp2` | transcribed, called only by the gamma block | deleted |

**Why the old code could not stay.** `gamma_productf.c` was created from
nothing by glibc commit `d8cd06db62d9` (2013). It has no FDLIBM, SunPro,
Cygnus or Arm ancestor anywhere, so there is no permissive upstream to point
at: it is glibc-authored, FSF-copyright, LGPL-2.1-or-later expression, and an
Apache-2.0 distribution cannot carry a transcription of it. No NOTICE entry
cures that; only deletion does.

**What went with it.** `gfk_lgamma_pos`, `gfk_expm1` and `gfk_exp2` were
transcriptions of `e_lgammaf_r.c`, `s_expm1f.c` and `e_exp2f.c` whose only
caller anywhere in the tree was the gamma block (glibc's `gammaf` reaches
Gamma through `exp(lgamma)`, rescales with `exp2f`, corrects with `expm1f`).
ArWen's gamma evaluates no logarithm, no exponential and no `exp2` at all, so
all three became dead to the physics and are deleted, along with the three
`gf-libm-{lgammaf,expm1f,exp2f}.csv` sweep fixtures that graded them.
`gfk_lgamma_pos` was the **only** transcription of `e_lgammaf_r.c` anywhere in
this repository, so that file is now retired from the distribution outright.

**What the replacement is.** Gamma on `[1,2)` as a degree-9 polynomial on each
of 16 equal segments, coefficients generated in 113-bit arithmetic from
Stirling's asymptotic series (DLMF 5.11.1) with the standard Bernoulli numbers
`B2..B24`; every other positive argument reduced by the functional equation
`Gamma(z+1) = z Gamma(z)`; negatives by the reflection formula with a
closed-form Taylor `sin(pi r)`. Classical mathematics, no implementation of
any kind consulted. It carries no table of glibc's behaviour and no fitted
correction — which is the point, and is also the structural evidence it is not
derived from the reference: a table fitted to glibc would have inherited
glibc's error, ten orders of magnitude larger than this code's ~5.6e-17
agreement with true Gamma.

---

## 1. The routine itself — MEASURED, exhaustively

`glibc_flt32.cuh` compiled unmodified as host C++ through an IEEE-754 shim
(`__fadd_rn` -> `+`, `__dmul_rn` -> `*`, ...), `gcc 13.3.0`,
`-ffp-contract=off -fno-unsafe-math-optimizations`, x86-64 SSE2 so
`FLT_EVAL_METHOD == 0`. Oracle: libquadmath's 113-bit `tgammaq`, rounded once
to float32. Every float32 in `[0.25, 36]` = `0x3E800000 .. 0x42100000`, the
interval that covers every argument this scheme can reach:

```
n                                = 59,768,833
gfk_tgamma NOT correctly rounded =          0     (worst 0 ULP)
glibc 2.39 NOT correctly rounded = 23,575,230     (39.4440 %, worst 6 ULP,
                                                   at 0x40359285 = 2.83706784)
gfk_tgamma != glibc 2.39         = 23,575,230     (39.4440 %)

glibc's error, by size:  0 ULP 36,193,603 | 1 ULP 20,092,232 | 2 ULP 3,165,956
                         3 ULP    301,820 | 4 ULP     14,950 | 5 ULP     269
                         6 ULP          3
```

The oracle is decisive rather than assumed: no argument in the interval lies
within 2^-100 relative of a float32 rounding boundary, a 1000x margin over
`tgammaq`'s own ~2^-110 error, so the result is rigorous, not probabilistic.

Committed fixture, `gpuwm/data/gf/oracle/gf-crgamma-tgammaf.csv`, 65,638
arguments:

```
shipped gfk_tgamma vs the fixture      0 / 65,638 words differ
glibc 2.39 vs the fixture         25,713 / 65,638 words differ (39.17 %),
                                  worst 4 ULP
```

---

## 2. The divergence at the point that reaches the physics

`fzu = gamma(a+b)/(gamma(a)*gamma(b))`, one float32 rounding per operation,
exactly as `gf.cu` spells it.

**Committed fixture** `gpuwm/data/gf/oracle/gf-crgamma-fzu.csv`, 126
`(alpha, beta)` pairs:

```
shipped kernel vs the fixture                    0 / 126 rows differ
fixture vs fzu composed from glibc's tgammaf    89 / 126 rows (70.6 %), worst 5 ULP
fixture vs WRF's OWN CAPTURED fzu               68 / 100 shared rows (68.0 %), worst 5 ULP
```

On the 26 `(alpha, beta)` pairs the committed 216-column WRF capture actually
reaches, **21 differ, worst 4 ULP** — inside the 4-ULP budget
`tests/test_gf_deep_parity.py::test_fzu_is_the_one_measured_divergence` has
carried for the CPU reference since the port landed.

**Over the whole reachable set** — every float32 `tunning` of `[0.2, 0.9]`
(drafts 0 and 2) and `[0.2, 0.8]` (draft 1), 53,687,093 cases enumerated
completely, not sampled. `tunning` is clamped by
`GMAX(K_P2, GMIN(K_P9|K_P8, ...))` in `gf.cu`, so that grid IS the reachable
set rather than a sample of it, and `alpha in [1.075, 27.9999924]`,
`alpha+beta in [2.375, 31.9999924]` — every argument inside the `[0.25, 36]`
interval section 1 proves, none outside:

| ULP moved | share | cumulative |
| ---: | ---: | ---: |
| 0 | 31.829 % | 31.829 % |
| 1 | 32.150 % | 63.979 % |
| 2 | 20.623 % | 84.602 % |
| 3 | 10.326 % | 94.928 % |
| 4 | 3.462 % | **98.390 %** |
| 5-8 | 1.606 % | 99.996 % |
| 9-12 | 0.004 % | 100 % |

`fzu` changes on **68.1707 %** of the reachable set (36,598,858 of
53,687,093); **98.390 %** of it moves by 4 ULP or less; the worst is
**12 ULP** (draft 1, beta = 2.5, `tunning = 0x3F07BC23`), worst relative
**8.3852e-7** (`tunning = 0x3F2AF59A`). Per draft: UP beta=1.3 changes
42.7168 % worst 9 ULP, SH2 beta=2.5 changes 89.2897 % worst 12, DN beta=4.0
changes 74.4254 % worst 11. The committed 216-column fixture spans only the
first 4 ULP, which is why the gates read 4 and this bound reads 12.

---

## 3. What it costs the forecast

**The amplification is the scheme's, not the port's.** `cup_forcing_ens_3d`
builds every stability closure as `-xff/xk` with `xk = (xaa0 - aa1)/mbdt`, a
difference of two cloud work functions that agree to several digits. A
last-bit change in the mass-flux shape walks through `zu` into the vertical
integral `aa1` (450 ULP) and the cancellation turns that into per cent.

**MEASURED, on real columns, by tests that predate this change**
(`tests/test_gf_deep_parity.py`):

* `test_a_one_ulp_massflux_shape_perturbation_moves_xmb_by_seven_percent` —
  the CPU reference running the correctly rounded gamma against the WRF
  v4.6.1 capture: worst `|dxmb|/xmb` in **(0.05, 0.10)** over the 60 converged
  columns, i.e. about **7.3 per cent**, median **1.9 per cent**;
* `test_the_amplification_is_reproducible_from_one_ulp` — the same band from
  perturbing the oracle's own `fzu` by exactly one ULP, with no gamma involved
  at all. Column sensitivity `A = (dxmb/xmb)/(dfzu/fzu)`: `A_max ~ 1.22e6`,
  `A_median ~ 3.17e5`;
* `test_a_one_ulp_...` also asserts that with `fzu` PINNED, `xmb`, `pre` and
  `aa1` return to **max_ulp 0** — which is what attributes the whole residual
  to gamma and nothing else.

**So the forecast impact of this change is already measured on real data:
worst ~7 per cent of `xmb`, median 1.9 per cent, over the 60 converged columns
of the committed WRF v4.6.1 capture.**

### 3.1 The same thing at the boundary WRF actually publishes — MEASURED

`xmb` is an internal. This is the measurement at the fields GFDRV hands the
model, made by running `tools/gf_wrf461_oracle/gf_host_harness.cpp` (the
shipped kernel compiled as host C++, no GPU) twice over all 216 committed
columns: once with `fzu` pinned from the WRF capture, once with `fzu`
computed, which is the shipped forecast path.

```
columns with at least one pinned fzu    192 of 216
columns whose output moves               90 of 216
columns that move WITHOUT a pinned fzu    0        <- exact, not a heuristic
integer index fields that move            0        <- no branch flips at all
    (ktop, kbcon, ktop_deep, k22_shallow, kbcon_shallow, ktop_shallow)
```

| published field | columns moved | worst rel | median of movers |
| --- | ---: | ---: | ---: |
| `RAINCV`, `PRATEC` | 84 of 84 nonzero | **7.273 %** | 1.613 % |
| `pret` (deep precip) | 60 of 60 | 7.273 % | 1.932 % |
| `xmb_shallow`, `prets` | 30 of 30 | 0.839 % | 0.366 % |
| `RTHCUTEN` | — | 7.221 % * | — |
| `RQVCUTEN` | — | 7.218 % * | — |
| `RQCCUTEN` | — | 7.219 % * | — |
| `dudt_phy`, `dvdt_phy` | — | 7.218 % * | — |

\* per-level tendencies: the relative figure is taken over words whose
magnitude is above 1 per cent of that field's own maximum. Taken over ALL
nonzero words the ratio reaches 130 per cent, which is a small-denominator
artefact at a level whose tendency is near zero, not a 130 per cent forecast
change — the honest scale-free statement is that the largest absolute change
in any tendency field is **2.12 per cent of that field's own maximum**.

Three things in that table are worth stating plainly. The precipitation
number, **7.273 per cent**, agrees with the `xmb` amplification measured
independently by the CPU suite, which is a cross-check on both. **No integer
index moves on any of the 216 columns**, so the divergence displaces the
answer and does not flip a branch — a branch flip would be a much larger
claim than this file makes. And **no column with an all-zero captured `fzu`
moves at all**: 24 of the 216 columns never ran a PDF, get identical inputs
in both runs, and produce identical words. That last one is the exactness
check the gates assert.

**The tail, with its limits stated.** Products of independently measured
quantities, **corner bounds, not measured joint values** — the joint
distribution of (column sensitivity x the `fzu` error that column happens to
draw) is not measured and cannot be without running the column model:

| point in the reachable distribution | ULP | relative | x A_median | x A_max |
| --- | ---: | ---: | ---: | ---: |
| median | 1 | 6.0e-8 | 1.9 % | 7.3 % |
| mean | 1.76 | 1.06e-7 | 3.3 % | 12.9 % |
| 95th pct | 4 | 2.4e-7 | 7.6 % | 29 % |
| maximum | 12 | 8.39e-7 | 27 % | outside the linear regime |

The response is a perturbation only while `A * delta << 1`. Past roughly
20-30 per cent the ensemble closure can change which member controls `xmb`, so
the bottom-right cell describes a different draw rather than a displaced one.
**Honest statement: typical impact is a few per cent, the committed real-data
maximum is ~7 per cent, and on the 1.6 per cent of the reachable set beyond
that capture's 4-ULP span an individual convecting column can move by tens of
per cent. No hard ceiling is claimed for the worst corner.**

---

## 4. Which side is right — the narrow claim, because it is the supported one

Both composites graded against the exact `Gamma(a+b)/(Gamma(a)Gamma(b))`
computed in 113 bits and rounded once to float32, over the same 53,687,093
reachable cases (unit lic-02):

```
distance from the TRUE fzu            ArWen              glibc
  0 ULP                       29,943,177 (55.774 %)   15,394,696 (28.675 %)
  1 ULP                       22,565,059 (42.031 %)   21,790,676 (40.588 %)
  2 ULP                        1,169,684 ( 2.179 %)   10,760,564 (20.043 %)
  3 ULP                            9,172 ( 0.017 %)    4,128,772 ( 7.690 %)
  4 ULP                                1              1,212,671 ( 2.259 %)
  5-11 ULP                             0                399,714 ( 0.744 %)

strictly closer to the truth:  ArWen 49.946 %   glibc 9.591 %   equal 40.463 %
mean |relative error|:         ArWen 3.83e-8   glibc 9.67e-8  (glibc 2.53x worse)
```

**The supported claim:** *ArWen computes the value the Grell-Freitas equations
define. WRF computes a value up to 6 ULP away in gamma and up to 11 ULP away
in `fzu`, because gfortran binds `gamma()` to a `tgammaf` that is not
correctly rounded on 39.44 per cent of its domain. Where the two differ,
ArWen is closer to the defined value on 49.95 per cent of the reachable set
and further on 9.59.*

**The claim that is NOT supported, and must not be made:** "ArWen forecasts
better here." Two reasons, both from this repository's own records. First,
**neither answer is determined**: even with a mathematically perfect gamma,
rounding three gammas to float32 and doing one float32 multiply and one
float32 divide costs up to **4 ULP** of `fzu` on its own — 29 per cent of
`xmb` at `A_max`. WRF's `fzu` is up to 11 ULP out, ~80 per cent at the same
amplification. **No float32 build of this scheme, WRF's included, determines
`xmb` to better than tens of per cent.** Second, Grell-Freitas is
`"maturity": "implemented-unverified"` with `"scientific_evidence": "none"`
in `tools/build_registry.py`, whose own warning says no gpuwm/WRF forecast
trajectory comparison exists for this scheme. There is no instrument in this
project that could support a skill claim about GF, before or after this
change.

**Precedent.** This is the same ruling already applied twice in this tree:
the GF shallow `k22` MAXLOC section-offset correction (shipped default
corrected, WRF-faithful behind `k22_wrf_faithful`), and
`docs/wdm6_oracle_known_deltas.md`'s standing rule — *"float64 is the
defensible arithmetic, and ArWen's standing rule is not to be bit-exact to a
rounding artifact."* glibc's 6-ULP gamma is a rounding artifact.

**And this divergence is not new to ArWen; it shrank by one.** ArWen's float32
CPU authority `gpuwm/verify/gf_deep_ref.py::_tgammaf` has been
`(float)tgamma((double)x)` — correctly rounded — since the port landed, gated
at 4 ULP with the 7.3 per cent consequence recorded. MEASURED: `gfk_tgamma`
now returns the same word as that model on **all 59,768,833** arguments of
`[0.25, 36]`, where before the CUDA and CPU paths differed on 23,575,230.
This change makes ArWen internally consistent; it does not introduce a new
kind of disagreement.

---

## 5. HOW TO GET WRF'S ANSWER BACK — the paired-comparison procedure

**`fzu_override`, a per-column runtime input. No build flag exists and none
is wanted.**

`gfd_get_zu_zd_pdf` takes an `fzu_override`; `<= 0` means "compute it",
anything positive is used as-is. All three entry points expose it as a
per-column `scin` slot:

| kernel | slots |
| --- | --- |
| `gf_deep_stage` | `INS_fzu_up`, `INS_fzu_dn` |
| `gf_shallow_stage` | `SINS_fzu_sh` |
| `gf_gfdrv_stage` | `DINS_fzu_up`, `DINS_fzu_dn`, `DINS_fzu_sh` (added 2.6.6) |

The CPU reference has used the same override since the port landed
(`tests/test_gf_driver_parity.py`). To compare a run against a WRF capture
column by column, feed the capture's own `up_fzu`/`dn_fzu`/`sh_fzu` words
through those slots; the whole chain is then bitwise WRF's again and every
other transcribed line is graded at max_ulp 0.
`tools/gf_wrf461_oracle/gf_field_lists.py::drv_scalar_inputs(fixture, True)`
builds the pinned input array and `captured_fzu(fixture)` returns the words.

What the override does **not** do is give glibc's `fzu` in a free-running
forecast with no capture to draw from. **That use case is not offered on
purpose.** Reproducing glibc's bits without a capture would mean shipping a
table of glibc's measured deviation over the domain — 22.5 MB as it was
actually built, with an information-theoretic floor of 11.47 MB whose entire
content is a measurement of glibc's error with the mathematics subtracted out.
That is **1,601x more glibc-specific information than the 7,511 bytes of LGPL
source it would replace**, and it is the artefact this change exists to
remove. A `#ifdef` is not a licence boundary. It was measured, priced and
rejected (unit lic-02 section 6); do not rebuild it.

A free-running GPU-vs-WRF forecast comparison is not bitwise regardless of
gamma, and no such comparison exists for this scheme at all.

---

## 6. What grades gamma now, and how a reviewer runs it

The WRF oracle can no longer grade this routine, so it is graded against
**correct mathematics** instead — a strictly stronger check, and one any
reviewer can regenerate with any arbitrary-precision library, where
`gf-libm-tgammaf.csv` could only ever be regenerated on x86-64 glibc 2.39.

```
pytest tests/test_gf_gamma_correctly_rounded.py -v
```

Everything in that file except the two `@pytest.mark.gpu` tests runs with no
GPU, no CuPy and no compiler. It asserts, among other things: that the
reference fixture is itself correct (re-derived from an independent
double-precision gamma, with the rounding proven forced rather than assumed);
that the fixture and glibc's are different objects, so nobody can quietly
regenerate the oracle from glibc; the exact shape of glibc's error; that the
LGPL identifiers are absent from every shipped `.cu`/`.cuh`; and that this
file exists and is cited.

On a machine with a GPU, the device gates are:

```
pytest tests/test_gf_gamma_correctly_rounded.py tests/test_gf_deep_cuda.py \
       tests/test_gf_shallow_cuda.py tests/test_gf_gfdrv_cuda.py -v
```

* `test_device_gfk_tgamma_is_correctly_rounded` — max_ulp 0 against the
  113-bit reference over all 65,638 sweep arguments, plus the negative
  control that CUDA's builtin `tgammaf` is a third function again;
* `test_the_device_gamma_is_correct_and_the_builtin_is_not` — the same
  precondition asserted inside the deep gate, before it grades 191 fields
  that stand on it;
* `test_fzu_bitwise_on_the_pgamma_grid` — the composite against
  `gf-crgamma-fzu.csv`;
* `test_the_fzu_pin_was_honoured` / `test_the_sh_fzu_pin_was_honoured` — the
  override is read, so the pinned bitwise results are not a silent no-op;
* `test_the_unpinned_run_is_the_documented_divergence_not_a_regression` and
  `test_the_unpinned_shallow_fzu_is_the_documented_divergence` — negative
  controls that must FIRE. If either ever stops firing, gamma has gone back
  to reproducing glibc and the licence position has silently changed.

### The whole no-GPU crosscheck, run on this change

`tools/gf_wrf461_oracle/gf_host_parity.py` compiles the shipped kernel as
host C++ and grades it against the byte-frozen WRF v4.6.1 fixtures with no
GPU and no pytest. On this change it reports:

```
level fields bitwise: 83/83
float scalars bitwise: 69/69
int fields exact: 39/39
libm tgammaf vs correctly rounded: 65638 args, 0 word mismatches
shallow fields bitwise: 126/126
driver 8-column residual: max_ulp=34 max_rel=3.799e-06 (bounds: 34, 1e-4)
driver gate failures: 0
```

Nothing in that list moved. The 34-ULP driver residual is GFDRV's own
`module_gfs_physcons` mixed precision, unchanged and unrelated. The
scheme's ~4,000 transcribed lines are still graded bitwise against WRF;
what left the WRF oracle is gamma, and gamma is graded harder than it was.

### The contraction check

The kernel's rounding is only reproducible if nothing fuses, and NVRTC
compiles with `--fmad=true` by default. Every float operation in the
replacement is written through `FADD`/`FMUL`/`DADD`/`DMUL`/... , which pin
the unfused form. VERIFIED by compiling to PTX and counting, compile-only,
no GPU:

```
nvcc -arch=sm_90 -ptx <probe including glibc_flt32.cuh>
grep -c 'fma\.' -> 0        # and 0 at sm_75 and sm_120 as well
```

Zero `fma` instructions in `gfk_tgamma` or in the `fzu` composite at any of
the three architectures. The whole assembled translation unit
(`_preamble() + glibc_flt32.cuh + gf.cu`, 4,639 lines) also compiles clean to
PTX at all three.

To rebuild the reference fixtures from scratch, glibc-independent:

```
gcc -O2 -o /tmp/crg tools/gf_wrf461_oracle/gf_crgamma_dump.c -lm -lquadmath
cut -d, -f1 gpuwm/data/gf/oracle/gf-crgamma-tgammaf.csv | /tmp/crg /dev/stdin
```

---

## 7. What a future campaign still has to do

1. **A trajectory comparison.** Everything above is per-column and per-word.
   Nobody has run gpuwm and WRF forward from the same initial state with
   `cu_physics = 3` and compared forecasts, so the sentence "this changes the
   forecast by X" cannot be completed for any X at the model level. The
   registry says so; this file does not weaken that.
2. **The 1.6 per cent tail.** The committed 216-column capture spans 4 ULP of
   `fzu`. The reachable set goes to 12. A capture that reaches the tail would
   turn the corner bound in section 3 into a measurement.
3. **`gf-libm-tgammaf.csv` is still shipped**, now only as the *evidence of
   the divergence* in `test_glibc_is_the_one_that_is_wrong_and_by_how_much`.
   It is a recording of glibc's output over 65,638 arguments (0.11 per cent
   of the interval) used as a comparison, not as a component of any shipped
   code path. Whoever owns the recorded-sweep question should decide whether
   it stays.
