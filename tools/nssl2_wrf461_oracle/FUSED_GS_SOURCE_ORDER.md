# NSSL `nssl_2mom_gs` limiter and aggregate-update map

This is a read-only transcription/map of the clean official WRF
`module_mp_nssl_2mom.F` copied beside this file.  Line anchors below refer to
that copy.  The scope is the production option-18 default used by the GPU port:

```yaml
scope:
  subroutine: nssl_2mom_gs
  source_lines: 12231-24316
  option: 18
  ipconc: 5
  warmonly: 0.0
  hail_on: 1
  density_on: 1
  mixedphase: false
  imixedphase: 0
  graupel_volume_moment: true
  hail_volume_moment: true
  sixth_moments: false
  eqtset: 1
  ibfc: 1
  ihlcnh_after_init: 3
  ifrzg: 1.0
  ifiacrg: 1.0
  ifrzs: 1.0
  ffrzs: 0.0
  f2h: 1.0
  ffrzh: 1.0
  cwfrz2snowfrac: 0.0
  imurain: 1
  imusnow: 3
  imaxdiaopt: 3
  ioldlimiter: 0
```

The defaults above are set at WRF lines 199, 205, 308, 318-322, 369-370,
507-508, 538, 557-567, 598, 661, 1334-1345, 1416-1456, 1651-1680, and
1791-1831.  `ihlcnh=-1` becomes `3` during the option-18 (`ipconc=5`)
initialization.  There are no rain/graupel/hail reflectivity moments at
`ipconc=5`; the only predicted volume moments are graupel and hail.

## Notation and units

- `dt` below is WRF `dtp`; `dtinv` is `dtpinv`.
- `L = il5`, with `L=1` when `T < 273.15 K` and `L=0` otherwise (13744-13747).
- `J = il2` is a state-dependent snow/rain collision routing flag (15380-15381,
  15600-15605).  `il3` remains zero for `ipconc=5` because it is only enabled
  for `ipconc < 3` (15607-15609).
- `P(x)=max(x,0)` and `N(x)=min(x,0)`.
- `q*` process rates and `pq*` aggregate rates are mixing-ratio tendencies
  (`kg kg-1 s-1`).  `c*` and `pc*` are number-concentration tendencies
  (`m-3 s-1`).  `v*` and `pv*` are volume-fraction tendencies (`m3 m-3 s-1`).
- Category order in this map is vapor `v`, cloud water `c`, rain `r`, cloud ice
  `i`, snow `s`, graupel `g` (WRF suffix `h`), hail `H` (WRF suffix `hl`).
- All process rates are diagnosed before any category state is updated.  The
  aggregate update is a single state update at 23050-23162.

## Pre-rate state normalization

The active-cell gather is not an identity operation and is part of the oracle:

1. A cell runs GS only if vapor is supersaturated or at least one hydrometeor
   exceeds its category `qxmin` (13636-13682).
2. Every gathered mixing ratio is first `max(input,0)` (13767-13771).
3. Every gathered number is first `max(input,0)`.  Cloud and cloud-ice number
   are zeroed when mass is below `qxmin`; rain/snow/graupel/hail use the special
   small-mass/zero-number handling at 13787-13913.
4. Graupel and hail volume are `max(input,0)`.  Their implied density is clipped
   to `[xdnmn,xdnmx]`, and volume is rewritten to `rho_air*q/density`; a missing
   volume with appreciable mass is initialized from the default density
   (13916-13934, 14188-14246).
5. The per-process depletion ceilings are set from this one gathered state:
   `qvimxd=max(0,0.70*(qv-qis)/dt)` (or `0.99` when cloud is absent), each
   hydrometeor `q?mxd=0.1*q?/dt`, and each number `c?mxd=0.1*N?/dt`
   (15218-15284).  These ceilings bound individual rate diagnoses; they do not
   replace the shared aggregate limiters below.

## Raw process-rate ledger

The exact aggregate equations in later sections are the authoritative inventory.
This table groups their named terms by physical ledger operation.

| Family | Mass-rate symbols | Number/volume companions | Ledger interpretation |
|---|---|---|---|
| Ice vapor diffusion | `qidpv`, `qsdpv`, `qhdpv`, `qhldpv`; `qisbv`, `qssbv`, `qhsbv`, `qhlsbv`; `qiint` | `cisbv`, `cssbv`, `chsbv`, `chlsbv`; deposition number rates are explicitly zero | vapor to/from cloud ice, snow, graupel, hail; ice initiation also consumes vapor |
| Liquid evaporation/condensation | `qrcev`, `qscev`, `qhcev`, `qhlcev`, `qfcev` | `crcev`, `cscev`, `chcev`, `chlcev` | signed vapor/liquid exchange; `qfcev=0` for option 18 |
| Warm-rain conversion/collection | `qrcnw`, `qracw`, `qwcnr`, `qwshw` | `crcnw`, `cautn`, `cracw`, `cwshw`; `cracr` rain self-collection | cloud/rain conversion and rain collection of cloud; `qwcnr`, `qwshw`, and `cwshw` are initialized to zero in this configuration |
| Cloud freezing/contact freezing | `qwfrz`, `qwfrzc`, `qwfrzp`, `qwctfz`, `qwctfzc`, `qwctfzp`, `qiihr` | corresponding `cw*`; `vh*`/`vhl*` only where dense ice receives liquid | cloud water to primary ice/snow/dense ice |
| Rain freezing and ice-rain interaction | `qrfrz`, `qrfrzs`, `qrfrzf`; `qiacr`, `qiacrs`, `qiacrf`, `qracif` | `crfrz*`, `ciacr*`, `vrfrzf`, `viacrf` | rain donor split among snow/graupel/hail routes |
| Snow collection/conversion | `qsacw`, `qsacr`, `qsaci`, `qscni`, `qscnvi`, `qracs`, `qhcns`, `qscnh` | `csacw`, `csacr`, `csaci`, `cscni`, `cscnvi`, `csacs`, `chcns`, `cscnh`; `vsacw`, `vhcns`, `vscnh` | cloud/rain/ice/snow/graupel transfers; `csacs` is snow aggregation/self-collection |
| Graupel collection/conversion | `qhacw`, `qhacr`, `qhacs`, `qhaci`, `qhcni`, `qhlcnh`, `qhcnhl` | matching `ch*`; `vhacw`, `vhacr`, `vhcni`, `vhlcnh`, `vhlcnhl` | collection plus snow/ice to graupel, graupel to hail, and hail back to graupel |
| Hail collection | `qhlacw`, `qhlacr`, `qhlacs`, `qhlaci` | matching `chl*`; `vhlacw`, `vhlacr` | cloud/rain/snow/ice donors to hail |
| Melt/shedding | `qimlr`, `qsmlr`, `qhmlr`, `qhlmlr`; `qsshr`, `qhshr`, `qhlshr` | matching `c*`; `vhmlr`, `vhlmlr`, `vhshdr`, `vhlshdr`, soak rates | frozen-category loss and liquid-rain production |
| Secondary ice | `qicicnt`, `qicichr`, `qsmul`, `qhmul1`, `qhlmul1`, `qsplinter`, `qsplinter2` | `cicint`, `cicichr`, `csmul`, `chmul1`, `chlmul1`, `csplinter`, `csplinter2` | primary/secondary ice production with donor losses represented in the category sinks |

## Upstream shared limiters applied to raw rates

These run before the number and mass category aggregates.

### Shared frozen-vapor limiter (18975-19315)

`DoSublimationFix` is a compile-time `.true.` parameter.  WRF runs a two-pass
test saturation adjustment on the unchanged gathered state to diagnose
`qsimxdep` (maximum total deposition) and `qsimxsub` (maximum total
sublimation). With the compile-time default `DoSublimationFix=.true.`, a cell
without qualifying frozen mass retains the zero initialization for both
limits. The `qsimxdep=qvimxd` and `qsimxsub=1e20` branch at 19178-19185 is only
the compile-time `.false.` alternative and is unreachable for option 18.

```yaml
frozen_vapor_limiter:
  deposition_sum: qidpv + qsdpv + qhdpv + qhldpv
  condition: deposition_sum > qsimxdep
  factor: qsimxdep / deposition_sum
  rescales: [qidpv, qsdpv, qhdpv, qhldpv]
  sublimation_sum: qisbv + qssbv + qhsbv + qhlsbv
  condition_2: sublimation_sum < -qsimxsub
  factor_2: -qsimxsub / sublimation_sum
  rescales_2: [qisbv, qssbv, qhsbv, qhlsbv]
  wrf_lines: 19278-19315
```

Number sublimation rates are derived *after* this rescaling as
`c?sbv=(N?/q?)*q?sbv`; deposition number tendencies are set to zero
(19321-19333).

### Shared rain-freezing heat-budget limiter (20318-20379)

```text
qrztot = qrfrz + qiacr + qsacr
qrzmax = max(0, D_rain*vent_rain*N_rain*fwet1)
qrzmax = min(qrztot, qrzmax, qr/dt)
if T < 243.15 K: qrzmax = qr/dt
qrzfac = min(1, qrzmax/qrztot) when qrztot > qrzmax and qrztot > qxmin(r)
```

When `T <= 273.15 K` and `qrzfac < 1`, WRF multiplies all of
`qrfrz`, `qrfrzs`, `qrfrzf`, `qiacr`, `qsacr`, `qiacrf`, `qiacrs`,
`crfrz`, `crfrzs`, `crfrzf`, `ciacr`, `ciacrf`, `ciacrs`, `vrfrzf`,
and `viacrf` by the same factor.  `qracif` is notably not in this rescale list.

## Number aggregates and number-donor limiters

Number aggregation occurs first (20881-21355), and its rescaling can therefore
change raw number rates used by later mass/number aggregates.

### Cloud ice number (20908-20970)

With option-18 defaults:

```text
pccii = L*cicint
       + L*(cwfrzc + cwctfzc + cicichr)
       + chmul1 + chlmul1 + csplinter + csplinter2 + csmul

pccid = L*(-cscni - cscnvi - craci - csaci - chaci - chlaci - chcni)
       + L*cisbv - (1-L)*cimlr

pccin = ciint
```

`pccii` is formed before the cloud-number limiter below.  There is no shared
cloud-ice number donor limiter.

### Cloud number and limiter (20975-21082)

```text
pccwi = -cwshw                         # zero in this default path
pccwd = -cautn
       + L*(-ciacw - cwfrz - cwctfzp - cwctfzc)
       - cracw - csacw - chacw - chlacw
```

```yaml
cloud_number_limiter:
  condition: -pccwd*dt > Nc
  factor: -Nc/(pccwd*dt)
  aggregate_after: pccwd = -Nc/dt
  rescales:
    - ciacw
    - cwfrz
    - cwfrzp
    - cwctfzp
    - cwfrzc
    - cwctfzc
    - cwctfz
    - cracw
    - csacw
    - chacw
    - cautn
    - chlacw
  wrf_lines: 21052-21078
```

Exact ordering quirk: after the listed raw rates have already been multiplied
by `factor`, WRF applies

```text
pccii -= (1-factor)*L*(cwfrzc + cwctfzc)*(1-ffrzs)
```

using the **already scaled** `cwfrzc/cwctfzc` (21074).  Do not replace that with
a clean recomputation if byte/oracle fidelity is required.

### Rain number and limiter (21084-21170)

```text
pcrwi = crcnw
       + (1-L)*(-chmlrr/rzxh - chlmlrr/rzxhl - csmlrr - cimlr)
       - crshr

pcrwd = L*(-ciacr - crfrz) - chacr - chlacr + crcev - cracr
```

```yaml
rain_number_limiter:
  condition: -pcrwd*dt > Nr
  factor: -Nr/(pcrwd*dt)
  aggregate_after: pcrwd = -Nr/dt
  note: incoming pcrwi is not credited by this limiter
  rescales:
    - ciacr
    - ciacrf
    - ciacrs
    - crfrz
    - crfrzf
    - crfrzs
    - chacr
    - chlacr
    - crcev
    - cracr
  wrf_lines: 21142-21166
```

The scaled frozen-rain rates feed the subsequently formed snow/graupel/hail
number aggregates.

### Snow number and limiter (21173-21241)

At the moment the limiter runs, the default aggregate is:

```text
pcswi_pre = L*(cscnis + cscnvis) + cscnh
pcswd     = -chacs - chlacs - chcns
            + (1-L)*csmlr + csshr + cssbv - csacs
```

(`cwfrz2snowfrac=0` and `ffrzs=0` remove the other source terms.)

```yaml
snow_number_limiter:
  condition: imixedphase == 0 and Ns + dt*(pcswi_pre + pcswd) < 0
  factor: (-Ns + pcswi_pre*dt)/(pcswd*dt)
  aggregate_after: pcswd = factor*pcswd
  rescales: [chacs, chlacs, chcns, csmlr, csshr, cssbv, csacs]
  wrf_lines: 21211-21227
```

Only **after** that limiter, WRF applies the frozen-small-rain routing:

```text
pccii += (1-ifrzs)*(crfrzs + ciacrs)   # zero with ifrzs=1
pcswi  = pcswi_pre + ifrzs*(crfrzs + ciacrs)
```

Thus `crfrzs+ciacrs` is not credited as incoming snow number when the snow
number donor factor is calculated (21231-21237).

### Graupel number (21243-21260)

```text
pchwi = crfrzf + L*ciacrf + chcnsh + chcnih + chcnhl
pchwd = (1-L)*chmlr + chsbv - L*chlcnh - cscnh
```

There is no shared graupel-number donor limiter.

### Hail number and limiter (21264-21297)

With `ifrzg=ifiacrg=1`, the direct frozen-drop part is zero:

```text
pchli = chlcnhhl*rzxhlh
pchld = (1-L)*chlmlr + chlsbv - chcnhl
```

```yaml
hail_number_limiter:
  condition: imixedphase == 0 and NH + dt*(pchli + pchld) < 0
  factor: (-NH + pchli*dt)/(pchld*dt)
  aggregate_after: pchld = factor*pchld
  rescales: [chlmlr, chlsbv, chcnhl]
  wrf_lines: 21279-21293
```

Ordering quirk: `pchwi` was already formed and includes `+chcnhl`; it is not
recomputed after the hail-number limiter scales `chcnhl`.

## Mass aggregates and mass-donor limiters

### Vapor (21401-21450)

For `warmonly=0`:

```text
pqwvi = -N(qrcev) - N(qhcev) - N(qhlcev) - N(qscev)
        - qhsbv - qhlsbv - qssbv - L*qisbv

pqwvd = -P(qrcev) - P(qhcev) - P(qhlcev) - P(qscev)
        + L*(-qiint - qhdpv - qsdpv - qhldpv) - L*qidpv
```

There is no later aggregate vapor-availability limiter.  The upstream shared
frozen-vapor limiter is the protection for deposition/sublimation rates.

### Cloud mass and limiter (21452-21503)

```text
pqcwi = qwcnr - qwshw                  # both zero in this default path
pqcwd = L*(-qiacw - qwfrz - qwctfz)
        - L*qiihr
        - qracw - qsacw - qrcnw - qhacw - qhlacw
```

```yaml
cloud_mass_limiter:
  condition: pqcwd < 0 and -pqcwd*dt > qc
  factor: -max(0,qc)/(pqcwd*dt)
  aggregate_after: pqcwd = -qc/dt
  note: incoming pqcwi is not credited by this limiter
  rescales:
    - qiacw
    - qwfrzc
    - qwfrz
    - qwctfzc
    - qwctfz
    - qracw
    - qsacw
    - qhacw
    - vhacw
    - qrcnw
    - qwfrzp
    - qhlacw
    - vhlacw
  not_rescaled_despite_being_in_pqcwd: [qiihr]
  wrf_lines: 21475-21500
```

`pqcwd` is pinned to `-qc/dt`; it is not recomputed from the scaled named rates.
Cloud-ice/rain/snow/graupel/hail mass aggregates are formed later and therefore
see the scaled raw rates.

### Cloud-ice mass (21505-21567)

With `ffrzs=0` and `cwfrz2snowfrac=0`:

```text
pqcii = L*qicicnt + L*(qwfrzc + qwctfzc) + L*qicichr
        + qsmul + qhmul1 + qhlmul1 + qsplinter + qsplinter2
        + L*qidpv + L*qiacw

pqcid = L*(-qscni - qscnvi - qraci - qsaci)
        - qhaci - qhlaci + L*qisbv
        + (1-L)*qimlr - qhcni
```

There is no shared cloud-ice mass donor limiter.

### Rain mass and limiter (21569-21685)

Define `qrshr=qsshr+qhshr+qhlshr` (20861-20872).  The initial default aggregate
is:

```text
pqrwi = qracw + qrcnw + P(qrcev)
        + (1-L)*(-qhmlr - qsmlr - qhlmlr - qimlr)
        - qrshr

pqrwd_initial = L*(-qiacr - qrfrz)
                - qsacr - qhacr - qhlacr - qwcnr + N(qrcev)
```

```yaml
rain_mass_limiter:
  condition: pqrwd < 0 and -(pqrwd + pqrwi)*dt > qr
  factor: (-qr + pqrwi*dt)/(pqrwd*dt)
  rescales:
    - qiacr
    - qiacrf
    - qiacrs
    - viacrf
    - qrfrz
    - qrfrzs
    - qrfrzf
    - vrfrzf
    - qsacr
    - qhacr
    - vhacr
    - qrcev
    - qhlacr
    - vhlacr
    - qhcev
    - qhlcev
  not_rescaled_despite_being_in_pqrwd: [qwcnr]
  wrf_lines: 21609-21682
```

After scaling, WRF recomputes the outgoing aggregate as:

```text
pqrwd_after = L*(-qiacr - qrfrz - qsacr)
              - qhacr - qhlacr - qwcnr + N(qrcev)
```

This differs from the pre-limiter expression: `qsacr` moves inside the `L*(...)`
group (21640-21653).

Vapor dependency is exact and conditional:

1. Before changing the raw rates, WRF patches `pqwvi/pqwvd` for the change to
   `qrcev` alone (21615-21620).
2. It then scales `qrcev`, `qhcev`, and `qhlcev`.
3. If the **post-scale** `qrcev != 0`, it fully recomputes `pqwvi/pqwvd` from all
   current vapor rates (21656-21678).
4. If post-scale `qrcev == 0`, no full recomputation occurs.  The `qrcev` patch
   remains correct, but any simultaneous changes to `qhcev/qhlcev` are stale in
   `pqwvi/pqwvd`.

### Snow mass and limiter (21687-21741)

With option-18 routing defaults (`ifrzs=1`, `ffrzs=0`, `il3=0`):

```text
pqswi = L*(qscni + qsaci + qsdpv + qscnvi
           + qiacrs + qrfrzs + J*qsacr)
        + P(qscev) + qsacw + qscnh

pqswd = -(1-J)*qracs - qhacs - qhlacs - qhcns
        + (1-L)*qsmlr + qsshr + qssbv + N(qscev) - qsmul
```

```yaml
snow_mass_limiter:
  condition: imixedphase == 0 and pqswd < 0 and qs + dt*(pqswi + pqswd) < 0
  factor: (-qs + pqswi*dt)/(pqswd*dt)
  aggregate_after: pqswd = factor*pqswd
  rescales:
    - qracs
    - qhacs
    - qhlacs
    - qhcns
    - qsmlr
    - qsshr
    - qssbv
    - qsmul
    - qscev_if_negative
  wrf_lines: 21718-21735
```

No vapor recomputation follows the scaling of `qssbv` or negative `qscev`.
Also, `pqcii` was already formed with `+qsmul`, so it is stale if this limiter
scales `qsmul`.  The subsequent `pqcii += (1-ifrzs)*(qrfrzs+qiacrs)` is zero
for the default `ifrzs=1` (21737-21739).

### Graupel mass (21743-21765)

```text
pqhwi = L*(qrfrzf + qiacrf + qracif)
        + (1-J)*(qracs + qsacr)
        + L*qhdpv + P(qhcev)
        + qhacr + qhacw + qhacs + qhaci
        + qhcns + qhcni + qhcnhl

pqhwd = qhshr + (1-L)*qhmlr + qhsbv + N(qhcev)
        - qhmul1 - qhlcnh - qscnh - qsplinter - qsplinter2
```

There is no shared graupel mass donor limiter.

### Hail mass and limiter (21768-21809)

The frozen-rain split is zero with `ifrzg=ifiacrg=1`:

```text
pqhli = L*qhldpv + P(qhlcev)
        + qhlacr + qhlacw + qhlacs + qhlaci + qhlcnh

pqhld = qhlshr + (1-L)*qhlmlr + qhlsbv + N(qhlcev)
        - qhlmul1 - qhcnhl
```

```yaml
hail_mass_limiter:
  condition: imixedphase == 0 and qH + dt*(pqhli + pqhld) < 0
  factor: (-qH + pqhli*dt)/(pqhld*dt)
  aggregate_after: pqhld = factor*pqhld
  rescales:
    - qhlmlr
    - qhlsbv
    - qhcnhl
    - qhlmul1
    - qhlcev_if_negative
  wrf_lines: 21788-21804
```

There is no vapor recomputation after `qhlsbv/qhlcev` changes.  `pqhwi` was
already formed with `+qhcnhl`; `pqcii` was already formed with `+qhlmul1`; and
`pqrwi` was already formed using `qhlmlr`.  Those aggregates are not recomputed.

## Volume aggregates

For non-mixed-phase option 18, WRF first aliases `vhmlr=qhmlr` and
`vhlmlr=qhlmlr`, while `vhfzh=vhlfzhl=0` (21853-21874).  Volume aggregates are
formed after all mass limiters, so they see the final scaled raw `v*`/`q*`
rates.

Snow volume is inactive.  Graupel volume (22493-22611) is:

```text
pvhwi = rho*[ L*qracif/rhofrz
              + (L*qhdpv/qhdpvdn + (qhacs+qhaci)/qhacidn) ]
        + rho*P(qhcev)/1000
        + vhcns + vhacr + vhacw + vhfzh + vhcni
        + viacrf + vrfrzf

pvhwd = rho*[ ((1-L)*vhmlr + qhsbv + N(qhcev) - qhmul1)/xdn_g ]
        - vhlcnh + vhshdr - vhsoak - vscnh
```

Hail volume (22617-22671) is:

```text
pvhli = rho*[ (L*qhldpv + qhlacs + qhlaci)/500 ]
        + rho*P(qhlcev)/1000
        + vhlcnhl + vhlacr + vhlacw + vhlfzhl

pvhld = rho*(qhlsbv + N(qhlcev) - qhlmul1)/xdn_H
        + rho*(1-L)*vhlmlr/xdn_H
        + vhlshdr - vhlsoak
```

The omitted frozen-drop terms in the hail equations are exactly zero under
`ifrzg=ifiacrg=1`; the graupel equation has already substituted
`ffrzh=ifrzg=ifiacrg=f2h=1`.

There is no shared volume donor limiter.  The only final volume bound is
non-negativity on scatter; implied density is not recomputed after this GS
volume update.

## Latent heating and the one aggregate state update

The latent aggregates are formed from the **current named raw rates** after all
limiters (22962-23031):

```text
pfrz = (1-L)*(qhmlr + qsmlr + qhlmlr)
     + L*(1-imixedphase)*(qsacw + qhacw + qhlacw
                         + qsacr + qhacr + qhlacr
                         + qsshr + qhshr + qhlshr
                         + qrfrz + qiacr)
     + L*(qwfrz + qwctfz + qiihr + qiacw)

pmlt = (1-L)*(qhmlr + qsmlr + qhlmlr)

psub = L*(qsdpv + qhdpv + qhldpv + qidpv + qisbv)
     + qssbv + qhsbv + qhlsbv + L*qiint

pvap = qrcev + qhcev + qscev + qhlcev       # qfcev is identically zero

ptem = (felfcp*pfrz + felscp*psub + felvcp*pvap)/pi0
theta_pert += dt*ptem
```

For default `eqtset=1`, no Exner perturbation update is made.

The single aggregate mass/volume/number update is then (23050-23162):

```text
qv += dt*(pqwvi + pqwvd)
qc += dt*(pqcwi + pqcwd)
qr += dt*(pqrwi + pqrwd)
qi += dt*(pqcii + pqcid)
qs += dt*(pqswi + pqswd)
qg += dt*(pqhwi + pqhwd)
qH += dt*(pqhli + pqhld)

Vg += dt*(pvhwi + pvhwd)
VH += dt*(pvhli + pvhld)

Ni += dt*(pccii + pccid)
Nc += dt*(pccwi + pccwd)
Nr += dt*(pcrwi + pcrwd)
Ns += dt*(pcswi + pcswd)
Ng += dt*(pchwi + pchwd)
NH += dt*(pchli + pchld)
```

This ordering is why chaining independently state-updating process kernels is
not equivalent: all rates and all shared factors above must come from one
gathered state, followed by these aggregate updates once.

## Post-update GS adjustment under exact option-18 defaults

The block is labelled “saturation adjustment” (23174 onward), but most of it is
disabled for `ipconc=5`:

1. Temperature is recomputed once from the aggregate-updated theta at
   23187-23192.
2. If that temperature is above freezing and `qi>0`, all cloud ice is moved to
   cloud water; `Ni` is added to `Nc`; `qi` and `Ni` are zeroed; and theta is
   cooled by `fcc3*qi` (23194-23224).
3. The homogeneous-freezing condition at 23259-23260 requires
   `ipconc<2` or `ibfc in {0,2}`.  With `ipconc=5, ibfc=1`, it does not run.
4. Both iterative cloud/vapor adjustment branches are guarded by
   `ipconc<=1` (23348-23604), so neither runs.
5. Exact quirk: after warm cloud-ice melting changes theta, `temg` is not
   recomputed for `ipconc=5`.  `t0` is scattered from that pre-melt `temg` at
   23620-23623, while the theta state contains the melt cooling.

The GPU API's `primary_ice_target` must therefore preserve the default routing
above: initiated/secondary primary ice goes to cloud ice (`ffrzs=0`), while the
small frozen-rain split goes to snow (`ifrzs=1`).  It must not enable the
otherwise-disabled post-GS homogeneous-freezing branch.

## Final WRF bounds/scatter for option 18

### Mass, vapor, and theta (23648-23669)

- `theta_out = theta0 + theta_pert` with no explicit clamp.
- `qv_out = qv0 + qv_pert` with no final explicit clamp.
- For each active hydrometeor, WRF writes
  `q_out = q_work + min(q_input,0)`.  Since normal gathered inputs are
  nonnegative, this is normally just `q_work`; it is **not** a final
  `max(q_work,0)`.

### Two-moment number/size limiter (23703-23761, 24215-24231)

All option-18 categories take the two-moment branch because no category has a
sixth moment:

```text
if q <= 0:
    N = 0
else if N > cxmin:
    mean_volume = rho_air*q/(particle_density*N)

    if category is cloud or cloud ice, or snow with imusnow=3:
        mean_volume_max = xvmx(category)
    else if imaxdiaopt == 3:             # rain/graupel/hail defaults
        mean_volume_max = xvmx(category) /
          ((4+alpha)^3 / ((3+alpha)*(2+alpha)*(1+alpha)))

    if category is snow:
        mean_volume_max *= max(1, 100/min(100, snow_density))

    if mean_volume is outside [xvmn, mean_volume_max]:
        mean_volume = clamp(mean_volume, xvmn, mean_volume_max)
        N = rho_air*q/(particle_density*mean_volume)

N_out = max(N,0)
```

`particle_density` here is the density diagnosed/clipped during the initial
gather.  It is not re-diagnosed from the newly updated graupel/hail volume before
this number limiter.  If `0 < N <= cxmin`, WRF skips the mean-size correction.

### Volume and auxiliary fields (24235-24277)

- `Vg_out=max(Vg,0)` and `VH_out=max(VH,0)`; there is no final density repair.
- CCN, supersaturation diagnostics, and auxiliary ice nuclei are independently
  clamped nonnegative where present.  They are outside the 16-field fused GS
  workspace contract.

## Dependency/order checklist for a fused implementation

```yaml
required_order:
  - gather_and_bound_one_state
  - diagnose_all_named_raw_rates
  - apply_shared_frozen_vapor_limiter
  - derive_number_vapor_rates_from_scaled_mass_rates
  - apply_shared_rain_freezing_heat_budget_limiter
  - form_number_aggregates_in_WRF_order_and_apply_number_limiters
  - form_vapor_aggregate
  - form_cloud_mass_aggregate_and_apply_cloud_limiter
  - form_cloud_ice_mass_aggregate
  - form_rain_mass_aggregate_and_apply_rain_limiter_and_conditional_vapor_resum
  - form_snow_mass_aggregate_and_apply_snow_limiter
  - form_graupel_mass_aggregate
  - form_hail_mass_aggregate_and_apply_hail_limiter
  - form_volume_aggregates_from_final_named_rates
  - form_latent_heat_aggregates_from_final_named_rates
  - apply_mass_number_volume_theta_aggregates_once
  - apply_option18_post_update_cloud_ice_melt_only
  - scatter_with_option18_two_moment_bounds

must_preserve_stale_WRF_dependencies:
  - cloud_number_limiter_partial_pccii_patch_uses_already_scaled_rates
  - hail_number_limiter_does_not_recompute_pchwi
  - snow_mass_limiter_does_not_recompute_vapor_or_pqcii
  - hail_mass_limiter_does_not_recompute_vapor_pqhwi_pqcii_or_pqrwi
  - rain_mass_full_vapor_resum_runs_only_if_postscale_qrcev_is_nonzero
  - post_melt_temperature_is_not_recomputed_for_ipconc5
```
