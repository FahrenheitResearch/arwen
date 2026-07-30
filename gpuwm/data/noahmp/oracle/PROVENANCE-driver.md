# noahmp-driver.csv — provenance

Bitwise fixture for the Noah-MP **driver-side cold start** of WRF v4.6.1:

| routine | source | lines |
|---|---|---|
| `SNOW_INIT` | `phys/module_sf_noahmpdrv.F` | 2340-2440 |
| `NOAHMP_INIT` | `phys/module_sf_noahmpdrv.F` | 1828-2335 |

## Source identity

```
tree      <wrf-4.6.1-checkout>
commit    d66e442fccc04111067e29274c9f9eaccc3cef28
sha256(phys/module_sf_noahmpdrv.F)
          9010a757da994ed8796c63ca97da354eaf60c5c732df4ea9acad5bc62a973890
sha256(phys/module_sf_noahmplsm.F)
          bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282
sha256(gpuwm/data/noahmp/MPTABLE.TBL)
          7fae6a77660c90ad80845565ecfb057093c100de41f35f25a7ffa63f41c19e5d
sha256(gpuwm/data/noahmp/SOILPARM.TBL)
          1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f
sha256(gpuwm/data/noahmp/GENPARM.TBL)
          9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2
sha256(noahmp-driver.csv)
          e7f702e90b6df4c0a81e6eb059080bf0f38a2226e929781c7379550ee2323626
compiler  GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
libc      glibc 2.39
harness   tools/noahmp_wrf461_oracle/run_driver.F90
build     tools/noahmp_wrf461_oracle/build_driver.sh
validate  tools/noahmp_wrf461_oracle/validate_driver_oracle.py
```

## No visibility patch

`phys/module_sf_noahmpdrv.F` contains **no accessibility statement at all** —
neither `private` nor `public` — so Fortran's default accessibility makes every
one of its module procedures public and this harness calls `SNOW_INIT` and
`NOAHMP_INIT` directly against the byte-unmodified source. `build_driver.sh`
stage `[2]` asserts that absence (`grep -c '^ *private'` and `'^ *public'` must
both be 0) rather than assuming it, and stage `[3]` refuses to build if
`module_sf_noahmplsm.F` carries the leaf-visibility lift, because this harness
must link the **pristine** LSM module. (The patched one physically cannot
compile against this driver — `ADDING_A_LEAF.md` section 9.)

This makes the fixture strictly stronger than the leaf fixtures: there is no
patched source anywhere in its build.

## Build flags

```
FCBASE   -w -cpp -ffree-form -ffree-line-length-none
FCOPTIM  -O0                             (the fixture)
```

`module_sf_noahmpdrv.F` additionally gets `-DEM_CORE=0
-fallow-argument-mismatch`, which is what WRF's own build uses for this file
and which compiles out the DM_PARALLEL halo exchange in `GROUNDWATER_INIT` —
a routine `iopt_run=3` never reaches.

## Negative controls — all three reproduce the fixture byte for byte

| optlevel | flags | result |
|---|---|---|
| `nocontract` | `-O0 -ffp-contract=off` | byte-identical |
| `snan` | `-O0 -finit-real=snan -finit-integer=-2147483647 -finit-logical=false` | byte-identical |
| `wrf` | `-O2 -ftree-vectorize -funroll-loops` | byte-identical |

The `snan` control is load-bearing here and it earned its keep. Two locals in
this slice are genuinely left undefined on live paths — `SNOW_INIT`'s `DZSNO`
is zeroed only on the `SNODEP < 0.025` branch (2383), and `NOAHMP_INIT` leaves
`cropcat` unwritten on a vegetated column when `iopt_crop=0` — and the control
proves neither is ever *read*. It also caught a defect in the harness itself:
the first version emitted `BEXP_TABLE`/`SMCMAX_TABLE`/`PSISAT_TABLE`/
`SLA_TABLE` in the `input` stage before `NOAHMP_INIT`'s own `read_mp_*` readers
had populated them, which at `-O0` looked like a plausible column of zeros and
under `snan` was visibly `NaN`. The harness now calls the module's own readers
first.

## libmvec — trap 3, and where it actually lands

At `-O0` the compiled `module_sf_noahmpdrv.o` references **no** glibc libmvec
symbol; `nm -u` shows only scalar `expf`, `logf` and `powf`.

At WRF's own `FCOPTIM` the object does reference two:

```
U _ZGVbN4v_logf
U _ZGVbN4vv_powf
```

`objdump -dr` places all three relocations inside
`__module_sf_noahmpdrv_MOD_pedotransfer_sr2006`, which `module_sf_noahmpdrv.F`
calls only from `IF(opt_pedo == 1)` inside `IF(iopt_soil == 3)` at 977-985 —
a path neither `SNOW_INIT` nor `NOAHMP_INIT` can reach, and which the pinned
identity (`opt_soil=1`) kills outright. `build_driver.sh` stage `[6]` asserts
the ownership rather than the mere count: any libmvec reference owned by
`SNOW_INIT` or `NOAHMP_INIT` fails the build at every optimisation level, and
any libmvec reference at all fails at every level but `wrf`.

That is why, unlike `PROVENANCE-soilwater.md`, this fixture has **no recorded
`-O2` divergence**: the vectoriser never touched the pinned routines.

## The one transcendental

`NOAHMP_INIT` 2095-2096:

```fortran
FK = (( (HLICE/(GRAV*(-PSISAT))) * ((TSLB(I,NS,J)-T0)/TSLB(I,NS,J)) )**(-1/BEXP) )*SMCMAX
```

`-1` is a default `INTEGER` and `BEXP` a `REAL`, so `-1/BEXP` is **REAL**
division and the whole power is `REAL**REAL`, which gfortran lowers to a scalar
`powf` call. glibc's `powf` is not correctly rounded, so evaluating this in
binary64 and rounding once is a *different* function, not a more accurate one.
The port uses `gpuwm/core/noahmp_libm.powf` and the CUDA half uses `r_pow` from
`noahmp_leaves.cu` — the single glibc 2.39 transcription in the tree.

## Coverage

54 cases, 7 790 rows, every case emitting its complete entry state and its
complete exit state.

`SNOW_INIT` — 24 cases (12 columns × two soil stacks, `NSOIL` 4 and 9). All six
branches of the depth ladder (2381-2405), each interval entered on **both**
closed edges (`SNODEP` sits exactly on 0.025, 0.05, 0.10, 0.25 and 0.45 as
binary32), so a port that swaps `<` for `<=` anywhere cannot pass. Two soil
stacks and two `NSOIL` values constrain the 2426-2435 loops.

`NOAHMP_INIT` — 28 cases (14 columns × `FNDSNOWH` true/false). Branch tally
from `validate_driver_oracle.py`:

```
glacier 2   nonglacier 26   frozen_layer 48   unfrozen_layer 48
smcmax_clamp 6   params_positive 24   params_degenerate 2
fk_floor 8   fk_ceiling 8   swe_cap 2   warm_snow_clamp 2
veg_zeroed 12   veg_grown 18   cropcat_unwritten 18
fndsnowh_true 14   fndsnowh_false 14
```

Two of the columns exist only to make a predicate discriminable and are worth
naming:

* `swe_at_cap` enters with `SNOW == 2000.0` exactly and
  `SNOWH == 3.8112843` — a value for which `SNOWH*2000.0/2000.0` is **not** the
  identity in binary32 (`4073EC15` → `4073EC16`). Without it, `SNOW > 2000` and
  `SNOW >= 2000` are indistinguishable.
* `veg_freeze_edge` straddles 2094's threshold with `TSLB` = 273.148 K (below
  `273.149`, frozen) and 273.1495 K (above it but **below** `T0`, unfrozen), on
  a soil where the frozen answer is `FK`, not the `MIN(FK, SMOIS)` clamp.
  Without the second layer, testing against `T0` instead of `273.149` is
  indistinguishable; without the soil choice, `MIN` swallows the difference.

Fourteen arguments are inert under the pinned identity — `XLAT`, `TMN`,
`croptype` and the eleven irrigation carriers — and every case drives all
fourteen non-zero and varying, so the validator's requirement that entry equals
exit is a measurement rather than a tautology.

## Mutation study

`tools/noahmp_wrf461_oracle/mutation_study_driver.py`: **108 of 114 mutants
killed**. The six survivors and their arguments are in that file's
`EXPECTED_SURVIVORS`; three of them are provably equivalent program
transformations (2092's guard under the pinned SOILPARM table, 2183's SAI form,
2187's SLA floor) and two of those are asserted executably in
`tests/test_noahmp_driver.py`. The other three are the `INTENT(OUT)` entry
values that 2411-2413 unconditionally zero.

## What this fixture does NOT cover

* `iopt_run=5` — the groundwater cold start (2299-2331), `STEPWTD`, `areaxy`
  and every OPTIONAL groundwater argument. The port refuses it.
* `iopt_crop=1/2` — the Liu and gecros crop blocks (2201-2260) and
  `gecros_init`. The port refuses it.
* `iopt_irr>=1` — the irrigation cold start (2263-2278). The port refuses it.
* `sf_urban_physics>0` — the `NATURAL_TABLE` `masslai` form at 2185. The port
  implements it (the branch is decided by a runtime argument, not by the
  identity) but no fixture case reaches it.
* `restart=.true.` — the whole body is skipped; the port refuses it.
* `ISLTYP < 1` — 2047-2057 calls `wrf_error_fatal`. The port raises. A fixture
  row is impossible by construction, because the harness would abort.
* The `noahmplsm` driver's own 2D↔1D packing (12-1432), which needs
  `NOAHMP_SFLX` and is not part of this slice.
