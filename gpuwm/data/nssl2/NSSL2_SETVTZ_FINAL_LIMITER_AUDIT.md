# NSSL2 MP18 setvtz and final-limiter source audit

Date: 2026-07-22

WRF authority: `/workspace/WRF-v4.6.1-reference/phys/module_mp_nssl_2mom.F`

CUDA target (read-only during this audit): `/workspace/gpuwm-nssl2-fused-gs-work/gpuwm/core/kernels/nssl2_fused_gs.cu`

Admitted configuration: option 18, `ipconc=5`, density and hail enabled, `mixedphase=.false.`, `imaxdiaopt=3`, `imurain=1`, `imusnow=3`, `imydiagalpha=0`, `isnowdens=1`.

This audit is deliberately limited to the GS gather/initial `setvtz` normalization and the post-GS two-moment limiter/scatter. It does not certify the intervening process-rate families or the later saturation-adjustment implementation. The recovery file was untracked and being edited while this review ran. The final recheck records a content hash so that the CUDA line references below identify one exact snapshot.

## Executive result

The initial normalizer and the final limiter are two different algorithms. Reusing the final `CXMIN`-gated bounds for initial `setvtz` was the main structural error. The review found the following MP18-relevant discrepancies in the live recovery implementation:

1. Existing positive cloud number was initially rewritten during `setvtz`; WRF retains it and clamps only diagnostic size.
2. The exact-zero, small-positive cleanup for rain, snow, graupel, and hail was missing, including its diagnostic-only vapor semantics.
3. Initial active low-positive number moments were incorrectly treated as missing through a `N <= CXMIN` gate.
4. Rain/graupel inactive-number retention and hail/cloud/ice/snow clearing semantics were conflated.
5. Initial graupel/hail `setvtz` incorrectly reused the shape-adjusted final maximum volume. Initial dense maxima are the raw 20 mm and 40 mm volumes; only the final limiter applies the MP18 shape factors.
6. Initial ice used the right `setvtz` mass bounds, but the final limiter initially reused those bounds instead of the much wider setpar ice volume bounds.
7. An MP18-inapplicable `1e-20` ice-number floor from the `ipconc==0` branch was present.
8. Hail fallback density was ambiguous until the admitted configuration was traced: the module declaration is 800 kg m-3, but `nssl_params(9)=900` overrides it before `xdn0` is built. MP18/oracle authority is 900.
9. Entry `fmax` loads erased the distinction between diagnostic nonnegative hydromass/vapor and raw Registry values needed for final scatter.
10. The final ice/snow bound helpers initially changed nonpositive mass to zero. WRF only clears number there.
11. The original negative hydromass residue was initially omitted, and its ordering matters: WRF restores it before the final two-moment limiter.

The final number clamp and the rule that bulk graupel/hail volume is not rebuilt by the final limiter were already structurally correct. Both are documented below because confusing mean particle volume `xv` with prognostic bulk volume `vx` would introduce another porting error.

## Source authority and admitted constants

- WRF initializes the configurable parameters from `nssl_params` at lines 1282-1292. The admitted oracle explicitly passes `[... 500, 900, 100]` at `tools/nssl2_wrf461_oracle/initial_state.F90:20-25`, so graupel, hail, and snow fallback densities are 500, 900, and 100 kg m-3. The raw module declaration `rho_qhl=800` at WRF line 224 is not the effective MP18 value; `xdn0(lhl)=rho_qhl` is built later at lines 1968-1975.
- Category thresholds are assigned at WRF lines 2087-2106: cloud and ice `1e-13`, rain `1e-12`, snow `1e-13`, and graupel/hail `1e-12` kg kg-1. `CXMIN=1e-8` is declared at line 587.
- Geometric volume limits are declared at WRF lines 897-918 and copied to `xvmn/xvmx` at lines 2030-2045.
- Number and volume Registry mixing ratios are converted once to concentration space at WRF lines 2933-2944 and divided by density once at lines 3254-3263. The GPU driver's gather/scatter follows the same convention in `nssl2_driver_support.cu:86-96` and `:1149-1157`.

The category bounds required by this admitted configuration are:

| Category | Initial `setvtz` bound | Post-GS two-moment bound |
|---|---|---|
| Cloud | `pi/6*(4 um)^3` to `pi/6*(120 um)^3`; only reconstruct missing `Nc` | Same volume range; adjust `Nc` only when `Nc > CXMIN` |
| Ice | mass `6.88e-13` to `1e-8` kg | volume `pi/6*(10 um)^3` to `pi/6*(2 mm)^3`, equivalent at 900 kg m-3 to about `4.712391e-13` to `3.7699128e-6` kg |
| Rain | `pi/6*(80 um)^3` to `pi/6*(6 mm)^3/(64/6)` | Same, because `imurain=1`, `alpha=0`, `imaxdiaopt=3` |
| Snow | `pi/6*(0.01 mm)^3` to `pi/6*(10 mm)^3` | Same base maximum, multiplied by `max(1,100/min(100,rho_s))` |
| Graupel | `pi/6*(0.3 mm)^3` to **raw** `pi/6*(20 mm)^3` | Same minimum, but maximum `/ (64/6)` for `alpha_h=0` |
| Hail | `pi/6*(0.3 mm)^3` to **raw** `pi/6*(40 mm)^3` | Same minimum, but maximum `/ (125/24)` for `alpha_hl=1` |

The dense-category split follows directly from initial graupel WRF lines 6808-6818 and hail lines 6854-6865 versus the generic final calculation at lines 23730-23750. It is not an algebraic refactor: using the final shape-adjusted maximum initially changes `Ng/Nh` before every process diagnosis.

## Exact entry/gather contract

WRF snapshots raw vapor independently (`qv0=an(lv)`, `qwvp=0`) at lines 13709-13713. It then loads every diagnostic mass as `qx=max(an,0)` at lines 13766-13770, every prognostic number as `max(an,0)` at lines 13786-13912, and every bulk volume as `max(an,0)` at lines 13917-13927.

For cloud and ice, inactive mass (`q <= qmin`) immediately clears number at WRF lines 13786-13809. For the four precipitating categories, gather uses this exact rule:

```text
q = max(raw_q, 0)
N = max(raw_N, 0)
if q > qmin:
    if N == 0 and q < 3*qmin:
        diagnostic_qv += q
        q = 0
    else:
        N = max(1e-9, N)
```

The exact-zero comparison and strict `< 3*qmin` comparison are source behavior (rain 13829-13837, snow 13842-13849, graupel 13865-13872, hail 13892-13899). They must not be replaced by `N <= CXMIN` or `q <= 3*qmin`.

Inactive behavior is category-specific:

- Rain retains its gathered number through initial `setvtz` (13829-13837, 6708-6713).
- Graupel retains its gathered number (13865-13872, 6834-6837).
- Snow gather retains it, but inactive `setvtz` clears it (6782-6786).
- Hail gather clears it when inactive (13892-13895).
- Cloud and ice clear it at gather (13789-13791, 13807-13809).

The tiny-category addition to `qx(lv)` changes process diagnosis only. It is not a conserved transfer into final Registry vapor: final vapor is still `qv0+qwvp` at WRF line 23651. A correct fused implementation can diagnose all rates from the augmented nonnegative vapor, then reset the aggregation base to raw `qv0` before adding the already-diagnosed vapor tendency.

## Exact initial `setvtz` contract

Cloud (`6425-6437`):

```text
if qc > QC_MIN:
    if Nc > CXMIN:
        clamp diagnostic particle mass/volume only; do not write Nc
    else:
        Nc = max(CXMIN, rho*qc/CLOUD_MMAX)
else:
    Nc = 0  # gather behavior
```

Ice (`6538-6543`): for active ice, clamp `Ni` between `rho*qi/1e-8` and `rho*qi/6.88e-13`. The `max(1e-20,Ni)` at WRF line 6534 belongs to the preceding `ipconc==0` branch, not MP18 `ipconc=5`.

Rain (`6653-6683`): for active rain, calculate mean volume with `max(1e-11,Nr)`, apply the admitted mass-weighted-diameter maximum, and rewrite `Nr` only if the mean is outside the range. There is no `Nr > CXMIN` gate.

Snow (`6727-6786`): for active snow, calculate with `max(1e-9,Ns)`. With admitted `isnowdens=1`, clamp below-minimum volume at fixed 100 kg m-3. Above the raw 10 mm maximum, use the Cox mass relation and diagnose the snow density exactly as WRF lines 6765-6770 do. Clear `Ns` throughout the inactive branch.

Graupel/hail (`6808-6885`): calculate active mean volume with `max(1e-9,N)`, clamp against the **raw initial** dense maximum, and adjust number. Again, there is no `N > CXMIN` gate.

## Density and bulk-volume contract

WRF starts from fixed density values, then examines prognostic volume concentration before `setvtz` at lines 14192-14245. For admitted MP18 only graupel and hail have prognostic volume moments (`lvi=lvs=0`; `lvh/lvhl` are enabled at lines 1668-1678 and mapped at 1798-1803).

For each active dense category:

```text
density = configured_default
if bulk_volume > 0:
    density = clamp(rho*q/bulk_volume, category_density_min, category_density_max)
bulk_volume = rho*q/density
```

Thus active graupel/hail bulk volume is rebuilt once at initial density setup, whether the input volume is positive, zero, or was clipped from a negative value. Inactive bulk volume is left as its nonnegative gathered value. Number-size adjustment in `setvtz` does not rebuild bulk volume.

The process sources subsequently evolve bulk volume independently at WRF lines 22492-22638 and 23076-23103. The final two-moment limiter computes a temporary per-particle `xv`; it never assigns prognostic bulk `vx`. Final volume scatter is only `max(0,vx)` at lines 24261-24276. The GPU must likewise clamp `VG/VH` nonnegative but must not force `VG=rho*qg/rho_g` or `VH=rho*qh/rho_h` after GS.

## Exact mass scatter and final limiter contract

WRF first writes vapor from the raw base plus tendency at line 23651. It then scatters each hydromass at lines 23658-23667:

```text
q_final_work = q_after_processes + min(raw_registry_q, 0)
qx = q_final_work
```

The second assignment is essential: the final two-moment limiter at lines 23717-23760 sees the negative-residue-adjusted mass. Restoring the residue only at the GPU output write is too late.

The admitted two-moment final limiter is:

```text
for each hydrometeor category:
    if q <= 0:
        N = 0
    else if N > CXMIN:
        mean_volume = rho*q/(particle_density*N)
        maximum = admitted_final_maximum(category)
        if mean_volume < minimum or mean_volume > maximum:
            mean_volume = clamp(mean_volume, minimum, maximum)
            N = rho*q/(particle_density*mean_volume)

for every number/CCN moment:
    N = max(N, 0)
for every prognostic bulk-volume moment:
    V = max(V, 0)
```

The limiter never changes hydromass. In particular, `q <= 0` clears number but preserves zero or negative mass. Final number/CCN clamping is explicit at WRF lines 24214-24249. Bulk volume is not made consistent with the final number or mass.

## Minimal fused correction skeleton

```text
snapshot raw_qv and raw_qc..raw_qh
load diagnostic qv/qc..qh = max(raw,0)
load N and dense bulk V = max(raw,0)

apply exact gather cleanup and initial setvtz rules
apply active dense density clamp and initial bulk-V rebuild
diagnose all rates from this one normalized diagnostic state
assemble and limit all rates

set qv aggregation base = raw_qv
advance vapor, hydromass, number, volume, and theta once
perform WRF warm cloud-ice melt

for qc..qh:
    q += min(raw_q,0)       # before final size limiter
apply category-specific final two-moment limiter to N only
clamp all N/CCN and bulk V to >= 0
scatter mass directly; scatter N/CCN/V divided by rho in the driver
```

## Required edge tests

1. Parameterize each category over `q={0,qmin,nextafter(qmin,+inf),nextafter(3*qmin,-inf),3*qmin}` and `N={negative,0,5e-10,1e-9,CXMIN,CXMIN+epsilon}`.
2. Verify cloud with an existing `Nc>CXMIN` and an out-of-range diagnostic size retains `Nc` initially, then may be adjusted by the final limiter.
3. Put graupel/hail mean volume between the final shape-adjusted maximum and the raw initial maximum. Initial `setvtz` must retain the number; the final limiter must adjust it.
4. Exercise ice masses near both initial bounds and both much wider final bounds.
5. Use raw negative vapor and raw negative hydromass. Confirm rate diagnosis uses nonnegative temporaries, final vapor uses the raw base, hydromass restores the original negative residue before final number limiting, and negative final mass is preserved.
6. Start active graupel/hail with positive, zero, and negative bulk volume. Confirm the initial rebuild, independent process evolution, nonnegative final clamp, and absence of a final rebuild.
7. Confirm all hydrometeor number moments and the CCN moment receive the final `max(0,N)` clamp even when no size adjustment fires.

## Final snapshot recheck

The recovery kernel was re-read at SHA-256
`e3dc8aae7736fc032a556db26568253556656bc1bf5c87336592154f3e1807f7`
(file timestamp `2026-07-23 01:33:17 UTC`). The exact CUDA locations in that
snapshot are:

| Contract/finding | CUDA lines | Recheck result |
|---|---:|---|
| Category thresholds and distinct initial/final bounds | 29-68 | Correct; dense raw initial maxima are at 59-66 and adjusted final maxima at 61-68 |
| Final number-only helper behavior | 257-313 | Correct; `q<=0` clears only number, not mass |
| Effective 900 kg m-3 hail fallback | 319-321 | Correct |
| Exact-zero small-positive cleanup | 323-358 | Correct, including strict equality and strict `<3*qmin` |
| Cloud existing/missing number distinction | 360-374 | Correct |
| Initial rain/ice/snow normalization | 375-407 | Correct; the stray MP18 ice `1e-20` floor is absent |
| Initial dense density, volume rebuild, and raw max | 408-445 | Correct |
| Final category limiter, number/CCN clamp, volume clamp | 449-472 | Correct; bulk `VG/VH` is not rebuilt |
| Raw entry snapshots versus nonnegative diagnosis | 2940-2967 | Correct |
| Diagnostic-only vapor cleanup versus raw final base | 3036-3039 | Correct: reset occurs after diagnosis/limiting and before one aggregation |
| Negative hydromass residue ordering | 3055-3063 | Correct: restored before `final_bounds` |
| Direct mass and moment scatter | 3064-3079 | Correct within the fused slab; driver performs the one `/rho` conversion for moments |

No remaining discrepancy was found in this narrowly defined normalization,
density/bulk-volume, final two-moment limiter, or scatter scope in that exact
snapshot. This is not a certification of the intervening rate implementation
or of code added after the recorded hash.
