# noahmp-energy.csv -- provenance

`ENERGY` is `phys/module_sf_noahmplsm.F` lines **1741-2396** of WRF v4.6.1. It
is not a leaf: it is the composition that owns the surface energy balance. Its
callees are each pinned by their own fixture already, so what this one exists
to pin is the branching, the tile-average bookkeeping and the state ENERGY
derives around those calls.

## Build

```
tree      <wrf-4.6.1-checkout> (WSL)
commit    d66e442fccc04111067e29274c9f9eaccc3cef28
compiler  GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
FCOPTIM   -O0
FCBASE    -w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4
command   bash tools/noahmp_wrf461_oracle/build_energy.sh \
               <wrf-4.6.1-checkout> <work>/nmp-energy
```

| file | sha256 |
|---|---|
| `phys/module_sf_noahmplsm.F` (pristine) | `bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282` |
| `phys/module_sf_noahmpdrv.F` (pristine) | `9010a757da994ed8796c63ca97da354eaf60c5c732df4ea9acad5bc62a973890` |
| `phys/module_sf_gecros.F` (pristine) | `ad2864562e95678a25276df82ef96395cca61c3a1bd0ab48ddfb8402902cf2f6` |
| `module_sf_noahmplsm_public.F` (visibility lift only) | `bfdc0f3632cd30b87208b26a309c533b12d9bc2a39d1a36e9165ecf90d0a12c3` |
| `noahmp_drv_helpers.F90` (generated) | `44f18a9cbbe27c3e734acc0c665493f6abfdbfcf5ca22c570cd540d5e7f9d35d` |
| `MPTABLE.TBL` | `7fae6a77660c90ad80845565ecfb057093c100de41f35f25a7ffa63f41c19e5d` |
| `SOILPARM.TBL` | `1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f` |
| `GENPARM.TBL` | `9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2` |
| `noahmp-energy.csv` | `c99cf60c5e58c07f3292c49d14ed950a6f11dcd66ed511f5e8ed22a4a6bc189c` |

The only change to the physics source is the `private ::` -> `public  ::` lift
performed by `visibility_patch_leaves.py`, which self-checks the pristine
sha256 and proves the diff is nothing else.

`noahmp_drv_helpers.F90` is **generated, not written**:
`extract_drv_helpers.py` copies `TRANSFER_MP_PARAMETERS`
(`module_sf_noahmpdrv.F:1435-1710`) and `SNOW_INIT` (`:2340-2440`) verbatim out
of the pinned driver and wraps them in a module. That indirection exists
because the visibility-patched module physically cannot be compiled against
`module_sf_noahmpdrv.F` -- lifting the leaf `ALBEDO` to public collides with
the driver's dummy argument of the same name at `:227`, which
`build_visibility_crosscheck.sh` stage 4 pins deliberately. The extractor
refuses to run if either lifted routine so much as mentions `ALBEDO`.

## The optimiser does not move this fixture

Built at WRF's own `FCOPTIM = -O2 -ftree-vectorize -funroll-loops` instead of
`-O0`, `run_energy` produces a **byte-identical** CSV: all 2731 rows agree.
`nm -u` on the compiled module shows no `_ZGV*` symbol at either level, so the
libmvec substitution recorded in `PROVENANCE-soilwater.md` did not fire here.
Reproduce with:

```
bash tools/noahmp_wrf461_oracle/build_energy.sh <tree> <work>/nmp-energy-o2 wrf
cmp <work>/nmp-energy/noahmp-energy.csv <work>/nmp-energy-o2/noahmp-energy.csv
```

That equality was **not** true of the first draft of this driver, and the way
it failed is worth keeping. `SNOW_INIT` assigns `ZSNSOXY` only from `ISNOW+1`
up, so on a snow-free column the buried slots come back as stack residue; the
`-O0` and `FCOPTIM` builds disagreed on `ZSNSO(-1)`. The driver now zeroes the
buried slots before ENERGY sees them. ENERGY never reads them --
`THERMOPROP`, `TSNOSOI` and `PHASECHANGE` all start at `ISNOW+1` -- and the
byte-identical output across the two builds is the evidence.

## Option identity

WRF Registry defaults, set through the module's own `noahmp_options`:

```
dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1 opt_inf=1
opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_rsf=1 opt_soil=1
opt_pedo=1 opt_crop=0 opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0
soil_update_steps=1  calculate_soil=.true.
```

What that identity makes dead, and what is therefore **asserted off rather
than ported** (`validate_energy_oracle.py::check_options`,
`gpuwm/core/noahmp_energy.py`):

| dead code | killed by |
|---|---|
| `SFCDIF2` | `opt_sfc=1` |
| `CANRES` / `CALHUM` | `opt_crs=1` |
| `SNOWALB_BATS` | `opt_alb=2` |
| ENERGY's `OPT_STC==2` snow-surface reset (`:2384-2396`) | `opt_stc=1` |
| ENERGY's CLM and SSiB `BTRAN` legs (`:2153-2163`) | `opt_btr=1` |
| ENERGY's Sellers `RSURF` legs (`:2188-2202`) | `opt_rsf=1` |
| gecros / crop chain | `opt_crop=0` |
| irrigation | `opt_irr=0` |

Outside the admitted slice, and asserted so in every case: `ICE == 0` (the
`ICE==1` ground-emissivity leg at `:2145-2147` is glacier) and `IST == 1` (the
`IST==2` lake legs at `:2076-2082` and `:2178-2181`). `noahmp-sflx.csv` does
not admit them either.

`ZPDG >= ZLVL` (`:2109`) is **unreachable**, not merely uncovered: `ZLVL` is
`MAX(ZPD,HVT) + ZREF` and `ZPDG` is either `SNOWH` (which is `<= ZPD`) or
`0.65*HVT` (which is `< HVT`), so the test needs `ZREF <= 0`. The validator
asserts `ZREF > 0` in every case rather than claiming coverage.

## Cases

Nine columns. Cases 1-4 are the four columns of `noahmp-sflx.csv`, argument for
argument: the same MPTABLE row, the same forcing, the same `SNOW_INIT`
topology, and the same `NOAHMP_SFLX` prologue (`ATM`, the `DZSNSO`/`TROOT`
reductions, `PHENOLOGY`, the `DVEG=4` `FVEG` rule, `PRECIP_HEAT`) evaluated by
calling WRF's own routines. Cases 5-9 are not in the whole-column fixture and
exist to reach branches the four realistic columns do not.

| # | case | what only it reaches |
|---|---|---|
| 1 | `veg_warm_day_dry` | vegetated, sunlit, snow-free |
| 2 | `veg_warm_night_rain` | `COSZ = 0`, so the undefined night radiation slots |
| 3 | `snowpack_frozen_soil` | `ISNOW = -2`, `FSNO = 1`, both phases frozen |
| 4 | `bare_thin_snow_melt` | `FVEG = 0` bare-only tile leg; the only melting case (`QMELT`, `PONDING`, `IMELT`) |
| 5 | `veg_calm_desert_dry` | `UR` clamped to 1.0; `SH2O(1) < 0.01` with no snow, so `RSURF = 1.E6`; root zone below wilting, so the `MAX(0.,GX)` clamp |
| 6 | `veg_deep_snow_saturated` | `SNOWH > 0.65*HVT`, so `ZPD` is taken from the snow; `SH2O(1) >= SMCMAX(1)`, so `MIN(1.,.)` in `L_RSURF` and `RSURF = 0`; `SH2O > SMCREF`, so the `MIN(1.,GX)` clamp; `ISNOW = -3` |
| 7 | `veg_subfreezing_canopy` | `frozen_canopy` set, `frozen_ground` clear |
| 8 | `urban_snowfree` | `parameters%URBAN_FLAG`: the `Z0MG`/`ZPDG`/`Z0M`/`ZPD` override and the urban `RSURF = 1.E6` |
| 9 | `veg_single_snow_layer` | `ISNOW = -1`; `frozen_ground` set, `frozen_canopy` clear; the only column that discriminates glibc's `tanhf` (see below) |

Every branch above is asserted **from the entry state**, never from the
outputs, so no coincidence in an answer can satisfy the coverage claim.

### Case 9 carries an odd SNEQV on purpose

`FSNO = TANH( SNOWH / (SCFFAC * FMELT) )` at `:2072` is glibc's `tanhf`, which
is still the 1993 SunPro `expm1f`-based routine and disagrees with
`(float)tanh((double)x)` on 23.8% of the FP32 inputs in [1e-6, 22]. With case
9's snow water equivalent at a round 8.0 mm, **all four** snow columns landed
on arguments where the two agree, and the CPU port reproduced this fixture
bit-for-bit with `math.tanh` substituted for the transcription. `SNEQV` is now
8.75 mm -- a snow density of exactly 250 kg/m3 -- which puts the argument just
below 1.75 (`0x3FDFFFFF`), where they differ. `max_ulp 0` on nine columns was not, by itself,
evidence that the fixture could tell a correct `TANH` from a plausible one.

By contrast `UU**2.0` at `:2057` compiles to `powf` rather than a multiply
(`nm -u` shows the symbol) and that turns out to be unobservable: over the
167,177,618 FP32 values in [1e-3, 1e3], glibc's `powf(x, 2.0f)` is
bit-identical to `x*x` on every one. Recorded as a measured null result so it
is not re-derived.

## Cross-check against the whole-column fixture

Because cases 1-4 stand on the `noahmp-sflx.csv` columns, every quantity ENERGY
writes that also appears as a `NOAHMP_SFLX` output has to agree bit for bit.
It does: **327 outputs across the four columns**, matched on the raw FP32 bit
pattern, with the only exemptions being the state `WATER` rewrites after ENERGY
returns (`SH2O`, `SMC`, `SNICE`, `SNLIQ`, `SNEQV`, `SNEQVO`, `SNOWH`). The
exemption is by name, so a new disagreement anywhere else fails the build.

## Slots WRF leaves undefined (role `undef`)

These are `INTENT(OUT)` arguments ENERGY does not assign on the path a case
takes. What the Fortran run leaves there is stack residue -- this driver
observed a `NaN` in `FSRG` at night -- so the fixture pins the **name** as
undefined and carries `0.0`, which is the defined behaviour the ports are held
to. The rule matches `THERMOPROP`'s dead slots in `noahmp-leaves.csv`.

| slot | why |
|---|---|
| `IMELT(k)`, `HCPCT(k)` for `k <= ISNOW` | `PHASECHANGE` and `THERMOPROP` loop from `ISNOW+1` |
| `SNICEV(k)`, `SNLIQV(k)`, `EPORE(k)` for `k <= ISNOW` | `CSNOW` (`:2547-2565`) loops from `ISNOW+1` |
| `BTRANI(k)` for `k > NROOT` | `:2164-2172` covers `1..NROOT` only |
| `FSRV`, `FSRG`, `BGAP`, `WGAP` when `COSZ <= 0` | `ALBEDO`'s init loop (`:2908-2921`) zeroes twelve arrays but not `FREVD/FREVI/FREGD/FREGI/BGAP/WGAP`, then takes `IF(COSZ <= 0) GOTO 100` past `TWOSTREAM`; `SURRAD` (`:3111-3112`) carries the first four into `FSRV`/`FSRG` |

`BTRANI` above the root zone is a live WRF defect, not just an ENERGY one:
`NOAHMP_SFLX` hands all `NSOIL` elements to `WATER`, which multiplies them into
`ETRANI`. That is the column's problem; nothing here pins the residue.

## The device side, and what it does not cover

`gpuwm/core/kernels/noahmp_energy.cu` reproduces the **ENERGY-owned** outputs of
all nine columns bit for bit on an RTX 5090 (sm_120, CUDA 12.8, cupy 14.1.1):
27 slots x 9 columns = 243 values, `max_ulp 0`, plus 50,020-sample device/host
parity on the new `tanhf`/`expm1f` transcriptions.
`tests/test_noahmp_energy_cuda.py` is the gate.

It does **not** re-run the six subsystems on the device, and the reason is a
cross-lane fact rather than a gap in this lane:

* `noahmp_radiation.cu`, `noahmp_vegeflux.cu` and `noahmp_bareflux.cu` each
  define their own `glibc_logf` / `glibc_expf` / `glibc_powf` /
  `powf_log2_inline` / `powf_exp2_inline` / `f_min` / `f_max`, so no two of
  them can share a translation unit;
* `noahmp_thermal.cu` exposes `TSNOSOI` and `PHASECHANGE` only as
  `extern "C" __global__` entry points, with no reusable `__device__` core;
* every entry point takes its own lane's flat fixture packing rather than a
  physical argument list.

Hoisting a single device libm out of `noahmp_leaves.cu` (which already has one)
and extracting `__device__` cores would let a whole-column ENERGY kernel exist.
That is a worthwhile refactor across four lanes' files and it is not this
lane's to make. Until it happens, the subsystem results the kernel consumes
come from this same fixture, so every number on both sides of the device
comparison still came out of gfortran.

One device failure is worth recording because it was invisible on the CPU. `TS`
was wrong in every vegetated column and nowhere else: `TS = FVEG*TV +
(1-FVEG)*TGB` at `:2298` reads the `TV` that `VEGE_FLUX` wrote, while the
psychrometric branch at `:2211` reads the entry `TV`. The CPU port carries one
mutable variable and gets both for free; a flat slot vector has to carry the
value twice, and it now does.
