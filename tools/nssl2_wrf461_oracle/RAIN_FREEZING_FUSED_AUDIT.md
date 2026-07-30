# WRF v4.6.1 NSSL option-18 rain-freezing fused-GS audit

This is a source audit for the production `ipconc=5` NSSL two-moment path. It
is intentionally documentation-only: it does not define a new numerical
scheme and must not be used to justify chaining the existing isolated GPU
process kernels.

## Pinned source

- WRF repository commit: `d66e442fccc04111067e29274c9f9eaccc3cef28`
- `phys/module_mp_nssl_2mom.F` SHA-256:
  `5aaae368289694c929d38365d77d445e4f22291a30a48555df7a21d470b72ae3`
- `nssl_2mom_gs` is at source lines 12230-24315, inclusive.
- Every line anchor below refers to that exact file.

## Exact default scope

The relevant defaults and option-18 initialization are:

| Setting | Default | Source |
|---|---:|---:|
| `ipconc` | `5` | 307, 1333 |
| `warmonly` | `0.0` | 204 |
| `mixedphase`, `imixedphase` | `.false.`, `0` | 506-507 |
| hail / predicted density | on / on | 1415-1455, 1651-1680 |
| `irwfrz` | `1` | 322 |
| `ibiggopt`, `imurain` | `2`, `1` | 379, 556 |
| `iacr`, `iacrsize` | `2`, `5` | 372, 381-386 |
| `ibiggsnow`, `ibiggsmallrain` | `3`, `0` | 380, 606-609 |
| `biggsnowdiam` | `-1.0 m` | 609 |
| `dfrz`, `dhmn` | `0.15 mm`, `0.30 mm` | 477, 894-895 |
| `rhofrz` | `900 kg m-3` | 316 |
| `ifrzg`, `ifiacrg`, `ifrzs` | `1`, `1`, `1` | 317-319 |
| `ffrzs`, `ffrzh`, `f2h` | `0`, `1`, `1` | 320-321, 12349, 13381 |
| `ibfc` | `1` | 368 |
| rain/graupel `qxmin` | `1e-12 kg kg-1` | 2087-2104 |
| `cxmin` | `1e-8 m-3` | 587 |

`ibfc=1` controls cloud-droplet freezing and the later homogeneous-freezing
branch. It does **not** guard Bigg rain freezing. The operative Bigg controls
are `ibiggopt=2`, `imurain=1`, and the gates below.

With hail and density enabled, both graupel and hail number and volume moments
exist. Nevertheless, the default `ifrzg=ifiacrg=1` routes the frozen-rain mass,
number, and volume receivers to graupel, not hail.

## Units and one-state rule

The WRF GS workspace uses:

- `q*`: mass mixing ratio or tendency, `kg kg-1` or `kg kg-1 s-1`;
- `c*`: number concentration or tendency, `m-3` or `m-3 s-1`;
- `v*`: volume fraction or tendency, `m3 m-3` or `m3 m-3 s-1`.

GPU registry number and volume fields are per kilogram of air. The fused
adapter must multiply them by air density on gather and divide by air density
on scatter. All rates below are diagnosed from the same gathered and bounded
state. There is one aggregate update at 23050-23161.

The `0.1*q/dt` and `0.1*N/dt` ceilings are built at 15218-15284, but they are
not universal. In particular, default Bigg option 2 uses an incomplete-gamma
tail fraction times `q/dt` and can approach the full rain donor in one call.

## Literal source order

| Stage | Source lines | Rain-freezing significance |
|---|---:|---|
| Gather, clip, and diagnose state/moments | 13636-15284 | Establishes `q`, `N`, density, `D`, and donor ceilings |
| Snow collection of rain | 16171-16204 | `qsacr` is initialized to zero; the `ipconc>=3` branch contains no executable diagnosis, so it remains zero for option 18 |
| Rain-cloud-ice collision/freezing | 16742-16925 | Diagnoses and partitions `qiacr/ciacr`; constructs `viacrf` |
| Bigg rain freezing | 17575-17918 | Diagnoses and partitions `qrfrz/crfrz`; constructs `vrfrzf` |
| Rain ventilation | 18283-18390 | Builds `rwvent` used by the heat budget |
| Wet-growth heat coefficient | 18521-18532 | Builds `fwet1` used by the heat budget |
| Frozen-vapor shared limiter | 18975-19333 | Independent shared limiter; occurs before the rain heat limiter |
| Rain-freezing heat limiter | 20313-20383 | Jointly scales Bigg, `qiacr`, and nominal `qsacr` routes |
| Rain evaporation and later raw rates | 20387-20823 | These are diagnosed after the heat limiter in the literal source |
| Number aggregates and donor limiters | 20880-21297 | Rain number can independently rescale only `c*` freezing routes |
| Mass aggregates and donor limiters | 21400-21808 | Rain mass can independently rescale only `q*` and associated `v*` routes |
| Dense-ice volume aggregates | 22492-22638 | Uses final mass/volume route rates |
| Latent heat and one state update | 22961-23161 | Uses final named mass rates, then updates once |
| Final scatter and two-moment bounds | 23647-24276 | Uses pre-update diagnosed dense-ice density; volume is not rediagnosed |

The source therefore does not literally diagnose every raw rate and then run
all limiters. A fused implementation may reorganize only after proving that
the unchanged gathered state and every intentionally stale dependency are
preserved.

## Bigg option-2 diagnosis

The block is at 17575-17918. Arrays are zeroed at 17578-17588. The active gate
at 17590-17593 is:

```text
qr > qxmin(r)  AND  Tc < -5  AND  ibiggopt > 0
```

For the default `ibiggopt=2, imurain=1` branch at 17613-17658:

```text
V_B = exp(16.2 + Tc) * 1e-6
D_B = (6/pi * V_B)^(1/3)
```

The calculation proceeds only when `D_B < 8e-3 m` (strict). Define
`D_lambda=xdia(r,1)`, the bounded rain characteristic diameter. Then:

```text
ratio = min(100, D_B / D_lambda)
i      = min(400, int(4*ratio))
ip1    = min(i+1, 400)
delx   = ratio - 0.25*i
```

The default rain shape is `alpha_r=0`; `alp0flag=.false.` and the alpha-table
index and interpolation displacement are both zero. WRF linearly interpolates
the adjacent 0.25-ratio nodes of `ciacrratio` and `qiacrratio`. At alpha zero,
the reusable analytic node values are:

```text
F_N(x) = exp(-x)
F_Q(x) = exp(-x) * (1 + x + x^2/2 + x^3/6)
```

where `x=0.25*i` at a node. The total rates are:

```text
crfrz = interp(F_N) * Nr / dt
qrfrz = interp(F_Q) * qr / dt
```

They are both reset to zero when either strict minimum-transfer test at
17654-17658 is true:

```text
qrfrz*dt < qxmin(g)  OR  crfrz*dt < cxmin
```

For option 18 these thresholds are `1e-12 kg kg-1` and `1e-8 m-3`. Equality
passes.

### Default small-frozen-drop routing

The isolated Bigg oracle disables this route, but production defaults do not.
`ibiggsnow=3` activates 17685-17756. Since `ibiggsmallrain=0`, the first branch
at 17673 is disabled. The active strict split gate is:

```text
D_B < max(biggsnowdiam, max(dfrz, dhmn)) = 0.30 mm
```

When it is false:

```text
qrfrzf = qrfrz;  crfrzf = crfrz
qrfrzs = 0;      crfrzs = 0
```

When it is true, WRF repeats the tail lookup at the `0.30 mm` cutoff:

```text
qrfrzf = tail_mass(D >= 0.30 mm) / dt
crfrzf = tail_number(D >= 0.30 mm) / dt
qrfrzs = qrfrz - qrfrzf
crfrzs = crfrz - crfrzf
```

With `ifrzs=1`, the `s` parts become snow mass and number. With `ifrzg=1`, the
`f` parts become graupel mass and number. The graupel-volume receiver is formed
at 17883-17885:

```text
vrfrzf = rho_air * qrfrzf / 900
```

The source contains an apparent overflow quirk at 17760-17767: if
`qrfrz*dt > qr`, it defines `fac=qrfrz*dt/qr` and multiplies the rates by that
factor rather than its reciprocal. The table tail is nominally at most one, so
the branch is not expected for finite, valid bounded inputs, but exact-oracle
work must not silently invent a different branch result.

Useful exact boundary temperatures implied by the Bigg formula are:

| Diameter boundary | Celsius | Kelvin |
|---|---:|---:|
| `D_B=8 mm` | `-17.5164602373` | `255.6335397627` |
| `D_B=0.30 mm` | `-27.3667032753` | `245.7832967247` |
| `D_B=0.15 mm` | `-29.4461448170` | `243.7038551830` |

The actual branch tests are performed in WRF single precision, so oracle
boundary rows must bracket the branch using representable `real` values rather
than assuming decimal equality survives rounding.

## Rain-cloud-ice freezing input

The `qiacr` family is at 16742-16925. Its default active guard is
`iacr>=1`, `eri>0`, and `T<=270.15 K`. `eri` requires rain and cloud-ice mass
above their thresholds and an ice diameter of at least `10 um` (15568-15579).
Default `iacrsize=5` selects the rain tail above `150 um`.

For `imurain=1`, the selected rain mass/number tails, selected ice number,
relative terminal velocity, collision geometry, and `eri` form total
`qiacr` and `ciacr`. Preserve the official source quirk at line 16806: the
middle mass-collision geometry term uses `xdia(g,3)*xdia(i,3)`, not the rain
diameter.

With `iacr=2`, `ciacrf` initially equals `ciacr`. The default
`ibiggsnow=3` then partitions the totals with:

```text
xvfrz = rho_air*qiacr/(ciacr*900)
frach = 0.5*(1 + tanh(0.2e12*(xvfrz - 1.15*xvbiggsnow)))
xvbiggsnow = xvmn(g)

qiacrs = (1-frach)*qiacr;  ciacrs = (1-frach)*ciacrf_initial
qiacrf = frach*qiacr;      ciacrf = frach*ciacrf_initial
viacrf = rho_air*qiacrf/900
```

`qracif=qraci` is mass of collected cloud ice copied at 20680-20684. It is not
rain-donor mass and is intentionally absent from the rain heat limiter.

## Shared rain-freezing heat limiter

The exact guard is `irwfrz>0 .and. .not.mixedphase` at 20317. Under production
defaults it is active.

The heat input coefficient is built at 18521-18532:

```text
fwet1 = 2*pi * [felv*fwvdf*rho*(qss0-qv) - ftka*Tc]
              / [rho*(felf + fcw*Tc)]
qss0  = 380/pres
```

`rwvent` is the default `imurain=1, iferwisventr=2` rain ventilation at
18349-18378. The limiter itself, preserving statement order, is:

```text
qrztot = qrfrz + qiacr + qsacr

qrzmax = D_lambda * rwvent * Nr * fwet1
qrzmax = max(qrzmax, 0)
qrzmax = min(qrztot, qrzmax)
qrzmax = min(qr/dt, qrzmax)

if Tc < -30:
    qrzmax = qr/dt

if qrztot > qrzmax and qrztot > qxmin(r):
    qrzfac = qrzmax/qrztot
else:
    qrzfac = 1
qrzfac = min(1, qrzfac)
```

The `< -30 C` overwrite is strict and occurs after both `min` operations. It
allows up to all available rain to freeze; it does not promise that several
simultaneous rain-freezing processes whose sum exceeds `qr/dt` remain
unscaled.

Only when `T<=273.15 K` and `qrzfac<1`, lines 20355-20379 multiply exactly:

```text
qrfrz, qrfrzs, qrfrzf,
qiacr, qsacr, qiacrf, qiacrs,
crfrz, crfrzf, crfrzs,
ciacr, ciacrf, ciacrs,
vrfrzf, viacrf
```

by the same factor. Do not additionally scale `qracif`, `csacr`, splinter
rates, or reflectivity moments. `qsacr` is named by the generic source but is
zero for `ipconc=5`.

## Independent downstream rain-donor limiters

The heat factor is not the last rain-donor scaling.

### Number donor

At 21084-21166, with `L=il5`:

```text
pcrwd = L*(-ciacr-crfrz) - chacr - chlacr + crcev - cracr
```

If `-pcrwd*dt > Nr`, WRF sets `pcrwd=-Nr/dt` and scales only the named number
rates `ciacr/ciacrf/ciacrs`, `crfrz/crfrzf/crfrzs`, `chacr`, `chlacr`,
`crcev`, and `cracr`. It does not scale their `q*` or `v*` companions.

After that independent number limiter:

```text
Ns source += crfrzs + ciacrs       # credited after the snow-N donor limiter
Ng source += crfrzf + L*ciacrf
NH direct frozen-rain source = 0   # ifrzg=ifiacrg=1
```

### Mass donor

At 21568-21682:

```text
pqrwi = qracw + qrcnw + max(qrcev,0)
        + (1-L)*(-qhmlr-qsmlr-qhlmlr-qimlr) - qrshr

pqrwd = L*(-qiacr-qrfrz) - qsacr - qhacr - qhlacr
        - qwcnr + min(qrcev,0)
```

If `pqrwd<0` and `-(pqrwd+pqrwi)*dt > qr`, then:

```text
facQ = (-qr + pqrwi*dt)/(pqrwd*dt)
```

WRF scales `qiacr/qiacrf/qiacrs/viacrf`,
`qrfrz/qrfrzs/qrfrzf/vrfrzf`, `qsacr`, and the other exact rates listed at
21621-21636. It does not scale the already aggregated `c*` routes. The
post-scale outgoing sum moves `qsacr` inside the cold gate:

```text
pqrwd = L*(-qiacr-qrfrz-qsacr) - qhacr - qhlacr
        - qwcnr + min(qrcev,0)
```

The conditional vapor patch/re-sum at 21614-21677 must remain source-exact.
Number aggregates were already formed and are not recomputed after this mass
scaling.

## Receiver moments, heat, and update

After the independent donor stages, default rain-freezing receivers are:

```text
snow mass    += qiacrs + qrfrzs
graupel mass += L*(qiacrf + qrfrzf + qracif)
hail mass    += 0

graupel volume += viacrf + vrfrzf
hail volume    += 0
```

There is no predicted snow-volume moment in option 18. Latent heating at
22961-23036 uses the final total mass rates once:

```text
pfrz includes L*(qrfrz + qiacr + qsacr)
```

Do not heat from both a total and its routed parts. The one mass/number/volume
update is at 23050-23161. The final two-moment bound at 23703-23761 uses the
graupel/hail density diagnosed from the entry state, not density rediagnosed
from the newly updated volume. Volume is only clamped nonnegative at
24261-24276.

## Mandatory official-oracle cases

The full-GS oracle must cover these as official WRF calls, not synthetic
expected values copied from a GPU implementation:

1. `qr` below, equal to, and above `1e-12`.
2. `Tc=-5 C` and the next representable value below it.
3. `D_B=8 mm` bracketed on both sides; equality does not enter the branch.
4. `qrfrz*dt` below/equal/above `1e-12` and `crfrz*dt`
   below/equal/above `1e-8`; equality passes.
5. Ratios immediately around 0.25-bin boundaries and the ratio-100 cap.
6. `D_B=0.30 mm` bracketed on both sides, proving default snow routing and
   conservation of total versus `f+s` parts.
7. New graupel and pre-existing graupel with low, in-range, and high implied
   density/mean volume.
8. `qrztot=0`; `0<qrztot<=qxmin(r)`; and nondegenerate combined
   Bigg-plus-`qiacr` states.
9. Negative/zero `fwet1`, heat capacity below/equal/above `qrztot`, and the
   `qr/dt` donor cap.
10. `Tc=-30 C` and the next representable value below it, proving the strict
    cold overwrite.
11. Heat-only scaling, rain-number-only scaling, rain-mass-only scaling, and
    cases in which all three activate in source order.
12. Independent mass/number divergence after the two donor limiters, with
    volume following mass rather than number.
13. `ifrzg=ifiacrg=1` invariants: hail mass, number, and volume receive no
    direct rain-freezing transfer.
14. Latent-heat conservation: final `pfrz` uses total mass rates once.
15. Zero/tiny donors and exact-factor-one paths, ensuring WRF skips rather than
    gratuitously multiplying rates by one.

The oracle evidence should record `qrztot`, `qrzmax` before and after the
`Tc<-30` overwrite, `qrzfac`, all total/partitioned `q*` and `c*` rates, both
volume rates, and the downstream number/mass limiter factors.

## Existing isolated Bigg kernel: reusable and incompatible pieces

Reusable pieces in `nssl2_bigg_rain_freezing` are limited to:

- the default alpha-zero lookup-node formula and 0.25-bin interpolation;
- bounded rain characteristic-diameter construction;
- the strict Bigg temperature and 8-mm gates;
- entry-state graupel density/number preparation; and
- `rho*qrfrzf/900` frozen-drop volume construction.

It is not a production fused-GS implementation because it:

- hard-codes graupel `qxmin=1e-7`, while option 18 sets `1e-12`;
- deliberately sets `ibiggsnow=0`, removing the production default
  `D_B<0.30 mm` snow split;
- excludes simultaneous `qiacr` and the shared `qrzfac` heat budget;
- excludes the later independent rain-number and rain-mass donor limiters;
- directly mutates and bounds state instead of contributing named rates to
  the single aggregate update; and
- assumes all admitted frozen rain goes to graupel.

It can supply arithmetic fragments, not a callable stage and not final-state
expected values for the fused implementation.

## Corrections to the provisional `FUSED_GS_SOURCE_ORDER.md`

The provisional uncommitted map inspected during this audit needs these
corrections or clarifications:

1. The exact subroutine range is 12230-24315, not 12231-24316.
2. It says a clean source copy is beside the map, but no such source file was
   present in that directory at audit time.
3. Its scope lists `ibfc` but omits the operative rain-freezing defaults
   `ibiggopt`, `irwfrz`, `ibiggsnow`, `ibiggsmallrain`, `iacr/iacrsize`,
   `dfrz/dhmn`, and `rhofrz`.
4. It should state that `qsacr=0` for the exact `ipconc=5` path.
5. The claim that all `0.1` per-process ceilings bound each diagnosis is false
   for Bigg option 2.
6. `qracif` is collected cloud-ice mass, not rain-donor mass; its omission from
   the heat rescale list is intentional.
7. The high-level `diagnose_all_named_raw_rates` ordering is not the literal
   source order shown above.
8. It omits the official 17760-17767 Bigg overflow quirk.
