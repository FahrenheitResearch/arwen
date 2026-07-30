# Adding one more Noah-MP leaf fixture

Written for an agent with no prior context on this lane. Following it end to
end should take well under an hour per leaf and requires no decisions that are
not spelled out here.

A "leaf" is one `private ::` subroutine of `phys/module_sf_noahmplsm.F` that
this harness calls directly, pinning its inputs and outputs as raw IEEE-754
FP32 bit patterns so a CPU/CUDA port can be held to `max_ulp 0`.

---

## 0. What already exists, so you do not rebuild it

| Piece | Path |
|---|---|
| Leaf harness driver | `tools/noahmp_wrf461_oracle/run_leaves.F90` |
| Build + audit + validate | `tools/noahmp_wrf461_oracle/build_leaves.sh` |
| Visibility-patch auditor | `tools/noahmp_wrf461_oracle/check_visibility_patch.py` |
| Fixture validator | `tools/noahmp_wrf461_oracle/validate_leaf_oracle.py` |
| Whole-module inertness proof | `tools/noahmp_wrf461_oracle/build_visibility_crosscheck.sh` |
| Object-code differ | `tools/noahmp_wrf461_oracle/compare_object_code.py` |
| Fixtures | `gpuwm/data/noahmp/oracle/noahmp-leaves*.csv` |
| Harness negative controls | `tests/test_noahmp_oracle_harness.py` |
| Port-side tests | `tests/test_noahmp_oracle.py` |

Eight leaves are pinned through `run_leaves.F90`: `atm`, `esat`, `rosr12`,
`csnow`, `tdfcnd`, `thermoprop`, `wdfcnd1`, `wdfcnd2`.

Six subsystems are pinned through **their own** driver, because their state
is too wide to pass through `run_leaves.F90`'s flat argument table. Follow the
same steps, substituting their driver for `run_leaves.F90`:

| Subsystem | Leaves | Driver / build |
|---|---|---|
| vegprecip | PHENOLOGY, PRECIP_HEAT | `run_vegprecip.F90` / `build_vegprecip.sh` |
| radiation | TWOSTREAM, ALBEDO, SNOW_AGE, SNOWALB_CLASS, GROUNDALB, SURRAD | `run_radiation.F90` / `build_radiation.sh` |
| bareflux | BARE_FLUX | `run_bareflux.F90` / `build_bareflux.sh` |
| vegeflux | VEGE_FLUX, SFCDIF1, RAGRB, STOMATA, ESAT | `run_vegeflux.F90` / `build_vegeflux.sh` |
| snow | SNOWFALL, COMPACT, COMBINE, DIVIDE, COMBO, SNOWH2O, SNOWWATER | `run_snow.F90` / `build_snow.py` |
| soilwater | CANWATER, SOILWATER, INFIL, SRT, SSTEP | `run_soilwater.F90` / `build_soilwater.sh` |

Two things in that set are shared and must not be forked again:

* the visibility lift is `visibility_patch_leaves.py` for all of them but snow
  (see its docstring for why snow keeps its own, and for the two byte-level
  spellings of the lift that exist in this tree);
* the FP32 transcendentals are `gpuwm/core/noahmp_libm.py`, the only module in
  the tree allowed to *define* `logf`/`log10f`/`expf`/`powf`/`atanf`/`sqrtf`.
  `tests/test_noahmp_radiation.py` fails if a second module defines one.

**soilwater is the one that is NOT built at WRF's own FCOPTIM.** At
`-O2 -ftree-vectorize -funroll-loops` gfortran vectorises `SOILWATER`'s
frozen-fraction loop into glibc's libmvec `_ZGVbN4v_expf`, which is a
different function from scalar `expf` and which no port can reproduce; `nm -u`
shows it is the only libmvec reference in the whole compiled module. Its
fixture is `-O0` and the `-O2` divergence is recorded, bounded at 1 ULP on two
columns, in `gpuwm/data/noahmp/oracle/PROVENANCE-soilwater.md`. If you add a
leaf that calls `EXP`/`LOG`/`**` inside a soil- or snow-layer loop, check
`nm -u` on your build before trusting the fixture.

## 1. Prerequisites

```
WSL, gfortran 13.3.0, pinned tree /home/drew/wrf-stock-v461-gate-20260721
  (commit d66e442fccc04111067e29274c9f9eaccc3cef28)
sha256(phys/module_sf_noahmplsm.F)
  = bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282
```

Everything here is CPU-only. Shell scripts must stay **LF** — `.gitattributes`
pins `*.sh text eol=lf`, and a CRLF copy dies under WSL bash with
`set: pipefail: invalid option name`.

## 2. Do you need to touch the visibility patch?  Almost certainly not.

`patches/noahmp-lsm-leaf-visibility.patch` already lifts **50** symbols —
every internal routine the module declares `private`:

> ATM, PHENOLOGY, PRECIP_HEAT, ENERGY, THERMOPROP, CSNOW, TDFCND, RADIATION,
> ALBEDO, SNOW_AGE, SNOWALB_BATS, SNOWALB_CLASS, GROUNDALB, TWOSTREAM, SURRAD,
> VEGE_FLUX, SFCDIF1, SFCDIF2, STOMATA, CANRES, ESAT, RAGRB, BARE_FLUX,
> TSNOSOI, HRT, HSTEP, ROSR12, PHASECHANGE, FRH2O, WATER, CANWATER, SNOWWATER,
> SNOWFALL, COMBINE, DIVIDE, COMBO, COMPACT, SNOWH2O, SOILWATER, ZWTEQ, INFIL,
> SRT, WDFCND1, WDFCND2, SSTEP, GROUNDWATER, SHALLOWWATERTABLE, CARBON,
> CO2FLUX, ERROR

If your leaf is on that list — and every leaf in plan steps 2-7 is — **skip to
section 3**. Add the symbol to the `use` list at the top of `run_leaves.F90`
and nothing else changes.

### 2b. If you genuinely need a 51st symbol

You are re-pinning three SHA-256 constants, which means the hash gate becomes
satisfied by construction and the *structural* audit in
`check_visibility_patch.py` becomes the only thing standing between this
project and an arbitrary edit to a pinned physics source. Treat it as such.

1. Regenerate the patch from the pristine source with **only** the sed the
   decision adjudicated:
   `sed 's/^\( *\)private *::/\1public ::/'`
2. Update, in `check_visibility_patch.py`: `PATCH_SHA256`, `PATCHED_SHA256`,
   and `LIFTED_SYMBOLS` (source order, `module_sf_noahmplsm.F` lines 26-84).
   `PRISTINE_SHA256` must **never** change.
3. Update the same two digests in `build_leaves.sh`
   (`PATCHED_SHA`, `PATCH_SHA`) and in `build_visibility_crosscheck.sh`.
4. Re-run `tests/test_noahmp_oracle_harness.py`. It drives the structural
   checks with the hash gate satisfied, which is exactly your situation.
5. Re-run `build_visibility_crosscheck.sh` (section 7). Object-code
   equivalence must still hold over every function body.

`BIND(C, name=...)` symbol aliasing was **rejected as unsafe** and is not an
option: the module exposes IPA-renamed specialisations such as
`..._MOD_sfcdif1.constprop.0.isra.0`, so an alias can silently bind to a
specialized clone and give a passing oracle for the wrong function.

## 3. Read the WRF routine before writing anything

Open `phys/module_sf_noahmplsm.F` at your routine and write down:

* **Exact line range** — it goes in the comment header and in the README.
* **Every dummy argument**, its `INTENT`, and its array bounds. Snow/soil
  arrays are `-NSNOW+1:NSOIL` with `NSNOW=3, NSOIL=4`; the harness flattens
  them to slot `1..7` in that order.
* **Which arguments are unread under the pinned option identity.** Anything
  behind `IF(OPT_* == n)` for an `n` that is not ours is inert; record the
  line numbers, you will need them in section 5.
* **Whether it reads `parameters`.** If it does, the case must build a
  `noahmp_parameters` handle; if it does not, pass an untouched one and say so
  in the comment (see `atm`).
* **Which outputs are `INTENT(OUT)` but left undefined on some paths** — e.g.
  layers above `ISNOW`. Those slots get `live = .false.` and the port must
  produce exactly `0.0` there.

Pinned option identity (WRF Registry defaults):

```
dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1 opt_inf=1
opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_rsf=1 opt_soil=1
opt_pedo=1 opt_crop=0 opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0
```

## 4. Write `eval_<leaf>` and `dump_<leaf>` in `run_leaves.F90`

Two subroutines, modelled on `esat` (the smallest) or `thermoprop` (the one
with integer topology and dead slots). Keep the leaves in the file's existing
order and add a comment header carrying the WRF line range and every fact from
section 3.

`eval_<leaf>` unpacks the flat slot vectors, calls the WRF routine, and packs
the outputs back:

```fortran
  subroutine eval_esat(x, ix, y)
    real,    intent(in)  :: x(:)
    integer, intent(in)  :: ix(:)
    real,    intent(out) :: y(:)
    real :: esw, esi, desw, desi
    call esat(x(1), esw, esi, desw, desi)
    y(1) = esw;  y(2) = esi;  y(3) = desw;  y(4) = desi
  end subroutine eval_esat
```

`dump_<leaf>` declares the case table and hands it to the generic `run_leaf`:

```fortran
  subroutine dump_esat()
    integer, parameter :: NX = 1, NY = 4, NCASE = 7
    ...
    xn = [character(len=12) :: 'tc']          ! input slot names
    yn = [character(len=12) :: 'esw','esi','desw','desi']
    xc(1, :) = [-50.0, -28.4, -7.15, 0.0, 6.35, 27.8, 50.0]
    ylive = .true.
    call run_leaf('esat', eval_esat, cn, ixn, ixc, xn, xi, xc, yn, yi, ylive)
  end subroutine dump_esat
```

`run_leaf` does the rest: it emits one row per `int`/`in`/`out` slot per case
with the raw bit pattern, then runs the **zero-probe sweep** (re-evaluating
with each input slot zeroed and recording how many live outputs moved) into
the discrimination CSV.

Then register it:

* add the routine to the `use module_sf_noahmplsm, only:` list at the top;
* add `call dump_<leaf>()` to `dump_all()`.

### Choosing cases

Cases are the whole value of the fixture. Rules that the validator enforces or
that reviewers will:

* **Every branch of the routine that is live under the pinned identity must be
  taken by some case**, and branch coverage must be assertable *from the
  inputs*, not from the outputs, so it cannot be satisfied by coincidence.
* **No input slot may be zero in every case** unless it is declared inert — a
  slot that is always zero cannot be discriminated from a port that ignores it.
* **Vary inert arguments too**, with non-zero per-case values, so their
  inertness is measured rather than vacuous.
* Include the domain edges the call sites actually clamp to (`esat` spans
  exactly `TDC`'s `[-50, 50]` clamp), not arbitrary round numbers.

## 5. Declare the leaf in `validate_leaf_oracle.py`

Add an entry to `LEAVES`:

```python
    "esat": {
        "cases": ("clamp_low_minus50", ..., "clamp_high_plus50"),
        "n_int": 0, "n_in": 1, "n_out": 4,
        "inert": {},
    },
```

Any argument you found in section 3 to be unread goes in `inert` **with the
WRF line numbers as its reason string** — that is the executable statement of
what this option identity does not consume:

```python
        "inert": {
            ("prcpsnow", 0): "read only inside IF(OPT_SNF==4) at 1217-1228",
        },
```

The validator then requires: every non-inert input slot moves at least one
live output in at least one case where it was not already zero, and every
inert slot moves **nothing anywhere**. If a slot you expected to matter shows
`noutputs_changed = 0` everywhere, your cases are too weak — fix the cases,
do not add it to `inert`.

Extend `check_branch_coverage` with an assertion for each branch you listed.

## 6. Build, validate, commit the fixture

```bash
cd <worktree-root>
bash tools/noahmp_wrf461_oracle/build_leaves.sh \
     /home/drew/wrf-stock-v461-gate-20260721 /home/drew/nmp-leaves
```

The script fails closed, in this order: pristine hash, patch hash, patch
applies, patched hash, `check_visibility_patch.py`, compile, run, validate.
On success copy the two CSVs into the repo and re-pin them:

```bash
cp /home/drew/nmp-leaves/noahmp-leaves.csv \
   /home/drew/nmp-leaves/noahmp-leaves-discrimination.csv \
   gpuwm/data/noahmp/oracle/
sha256sum gpuwm/data/noahmp/oracle/noahmp-leaves*.csv
```

Update the digests in `PINNED_LEAF_ASSETS` (`tests/test_noahmp_oracle.py`),
the `nvalues` total in `test_leaf_fixture_still_discriminates`, and the tables
in `gpuwm/data/noahmp/oracle/README.md`.

## 7. Re-run the whole-module inertness proof

Only needed if you changed the patch (section 2b), but it is cheap:

```bash
bash tools/noahmp_wrf461_oracle/build_visibility_crosscheck.sh \
     /home/drew/wrf-stock-v461-gate-20260721 /home/drew/nmp-xcheck
```

It compiles the pristine and patched modules and requires **every function
body and all of `.rodata` to be identical**, then proves that comparison can
fail by perturbing one FP32 literal and one operator. It also pins the fact
that the patched module **must not** compile against
`phys/module_sf_noahmpdrv.F` (see section 9).

## 8. The port side, and the trap waiting there

`max_ulp 0` is **not** sufficient evidence that a port is right. A sibling lane
in this project reached `max_ulp 0` on 29 columns and then found that 13 of 14
argument-drop mutants still reproduced its pinned CSV — the fixture could not
tell whether the port read those arguments at all.

So after adding the CPU evaluator to `gpuwm/core/noahmp_leaves.py`, add the
leaf to `gpuwm/core/noahmp_leaf_mutation.py` and run
`test_leaf_fixture_kills_every_argument_mutant`. Every input slot, FP32 and
integer topology alike, must fail to reproduce the fixture when replaced by
zero or by any constant the fixture gives it — unless declared inert or
partially observable with a justification.

Transcendentals: use `gpuwm/core/noahmp_libm.py`. glibc's `logf`, `log10f`,
`expf` and `powf` are **not** correctly rounded, so "evaluate in FP64 and round
once" is a different function, not a more accurate one. On `TDFCND`'s
`LOG10(SATRATIO)` domain `(0.1, 1.0]` that choice disagrees with glibc on
18.47% of FP32 inputs — about one soil column in five would miss `max_ulp 0`.

## 9. Constraints you must not quietly break

* **Never edit the pinned WRF tree.** The harness patches a scratch *copy*;
  `build.sh` and `run_sflx.F90` keep compiling the pristine module so the
  whole-column fixture stays byte-unmodified.
* **The patched module cannot be linked with WRF's Noah-MP driver.** Lifting
  the leaf routine `ALBEDO` to public collides with the dummy argument
  `ALBEDO` at `module_sf_noahmpdrv.F:227`
  (`Error: Name 'albedo' ... is an ambiguous reference`). This is load-bearing,
  not an obstacle: it means the visibility-patched module physically cannot
  reach a real forecast build. `build_visibility_crosscheck.sh` stage 4 fails
  if that stops being true. It is also why the patched-vs-pristine
  *whole-column* fixture diff is impossible, and why object-code equivalence is
  used instead — which is stronger anyway, covering all 85 routines rather than
  the paths four columns happen to take.
* **Never widen a gate to make something pass**, and never add a case-data or
  source name (`real74`, `hrrr`, ...) to anything here.
* **Never fabricate a fixture row.** Every number in the CSVs must come out of
  a gfortran run of the pinned source.
