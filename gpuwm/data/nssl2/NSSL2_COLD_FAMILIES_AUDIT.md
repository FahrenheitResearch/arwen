# WRF v4.6.1 NSSL-2 option-18 cold-family audit

Scope: default option-18 paths in nssl_2mom_gs that remain after the warm,
primary-ice, frozen-vapor, and rain-freezing audits.  This is a source-order
contract for the GPU fused implementation, not a proposal to simplify WRF.

Reference: /workspace/WRF-v4.6.1-reference/phys/module_mp_nssl_2mom.F,
subroutine lines 12230-24315.  The CRLF reference file SHA-256 is
5aaae368289694c929d38365d77d445e4f22291a30a48555df7a21d470b72ae3.

Audit snapshot of the fused kernel: central base bb32cee, uncommitted
gpuwm/core/kernels/nssl2_fused_gs.cu, 1913 lines, SHA-256
a3a4d4e47e4080d4da0cb3ccde0b7e6994cc4bcf2c1d28865e9c9513a9feda73.

## Effective option-18 defaults

- warmonly=0 (204), ipconc=5 (307), imurain=1 (556), imusnow=3 (557),
  mixedphase=.false. and imixedphase=0 (506-507).
- Double-moment minima from 2089-2104 are qc=1e-13, qr=1e-12,
  qi=1e-13, qs=1e-13, qg=1e-12, and qh=1e-12 kg kg-1.
- Most local donor caps are 0.1 category/dt (15217-15283).  Graupel/hail
  cloud collection qhacw/qhlacw has the explicit 0.5 qc/dt cap.
- hdnmn=170 and hldnmn=500 kg m-3 (226-227).  The predicted dense-volume
  fields are active only when their lvol category is enabled.
- Default collection controls are ehi0=ehi1=0.1, hail ehi0=0.2/ehi1=0,
  esi0=0.1, ehs0=ehs1=0.1, ehsmax=0.5, iglcnvi=1, and iglcnvs=2
  (427-455).
- iscni=4 (392); itype1=0 and itype2=2 (361-363).  cimas0=6.62e-11 kg
  and splinter mass cimas1=6.88e-13 kg (387-390).
- nsplinter=0 and isnwfrac=0 (496-498); ihrn=0 (367).  Therefore the
  active secondary-ice family is Hallett-Mossop, not rain splintering,
  snow fragmentation, or homogeneous rain nucleation.
- irimdenopt=1 with rimc1=300, rimc2=.44, rimc3=170, rimc4=900
  (326-329).
- imltshddmr is effectively forced to 1 for ipconc<=5 at 1908-1912,
  despite the nearby imltshddmr2 declaration.  ihmlt=2, ivhmltsoak=1,
  and iwetsoak=.true. (484-596).
- ihlcnh is initialized to 3 for ipconc=5 (1339-1344);
  hlcnhqmin=1e-4, iusedw=0, dwmin=5 mm, dwmax=15 mm, and
  dg0thresh=0.15 m (537-551).  icvhl2h=0 (554).
- Snow aggregation uses iessopt=1, ess0=.5, ess1=.05,
  esstem1=-15 C, esstem2=-10 C, and iessec0flag=0 (439-448).

Notation below follows WRF names.  Qx and Nx are category mixing ratio and
number, rho is air density, xv_x is per-particle volume, Dx3 is the diagnosed
characteristic diameter, and a rate has units per second.  The collision
relative speed is always

    Vrel = sqrt((Va - Vb)^2 + 0.04 Va Vb).

The generic mass cross-collection kernel is

    pi/4 * efficiency * Na * Qb * Vrel
      * (da0a Da3^2 + dab1 Da3 Db3 + da1b Db3^2),

while its donor-number companion substitutes Nb and the dab0/da0 moment
combination.  Implement the same WRF moment helpers rather than fitting these
expressions numerically.

## Source-ordered process map

### 1. Frozen/frozen collection and cloud riming

Efficiencies are diagnosed at 15359-15891, mass rates at 15931-16735, and
number companions at 16961-17323.

Cloud ice collecting cloud water, qiacw/ciacw (15493-15503):

- Requires qi and qc above their minima, cloud characteristic diameter
  strictly greater than 15 um, and ice characteristic diameter strictly
  greater than 30 um.
- eiw=.5, then eiw is set to zero when T>=273.15 K.
- Keep the source's separate mass and cloud-number donor caps.

Snow collecting cloud water, qsacw/csacw (15589-15597, 16072-16127):

- Under ipconc>=4, the operative expression does not multiply by the earlier
  esw scalar:

      tmp = rvt * aa2 * Ns * Nc
            * (((alpha_c+2)/(alpha_c+1))*xv_c + xv_s)
      qsacw = min(0.1*Qc/dt, tmp*m_c/rho)
      csacw = min(0.1*Nc/dt, tmp)

Snow collecting ice, qsaci/csaci (15656-15674, 16130-16167):

- esiclsn=1 and esi=min(.1, .1*exp(.1*min(Tc,0))); it is zero for
  T>273.15 K.
- Use the same A2/rvt collision structure and independently cap donor ice
  mass and number at 0.1/dt.

Snow collecting rain, qsacr/csacr (16176-16187):

- For the default ipconc>=3 branch the body contains only comments.
  The initialized rates remain exactly zero.  Do not infer a formula from a
  non-default branch.

Graupel collecting cloud, qhacw/chacw/vhacw (15682-15741, 16211-16342):

- ehw is the default iehw=1 polynomial in cloud radius, bounded by .9, and is
  zero below the 2.4 um cloud-diameter threshold.
- Use the Seifert kernel and the explicit 0.5 Qc/dt and 0.5 Nc/dt caps.
- vhacw=rho*qhacw/rimdn_g.  Below freezing,

      rimdn_g = clamp(
          300 * (-0.5e6*Dc1*(0.6*Vg)/(T-273.15))^0.44,
          170, 900)

  and it is 1000 above freezing.

Graupel collecting ice, qhaci/chaci (15778-15783, 16345-16370,
17117-17144):

- ehiclsn=1 and ehi=.1*exp(.1*min(Tc,0)).
- Mass uses dab1; donor number uses dab0.  Each is capped at 0.1 of its
  donor category per dt.

Graupel collecting snow, qhacs/chacs (15752-15776, 16393-16420,
17167-17194):

- The source unexpectedly requires qc>=qcmin in addition to qs and qg.
  Preserve that cloud-water gate.
- ehsclsn ramps from zero through Ds3=40 um to .5 at Ds3=150 um.
- ehs=min(.5, [.1*exp(.1*min(Tc,0))]
  *clamp((rho_g-300)/300,0,1)).
- Mass and donor number use the cross kernels and 0.1/dt donor caps.

Graupel collecting rain, qhacr/chacr/vhacr (15743-15750, 16422-16530):

- ehr=min(1,exp(-40um/Dr3)*exp(-40um/Dg3)).
- At T>273.15 the committed qhacr/chacr are zero, but the raw collision is
  retained as qhacrmlr for melt cooling.
- In the cold volume path WRF computes raindn and then overwrites the divisor
  with rimdn(g) at line 16522.  This stale-source quirk is observable:
  vhacr=rho*qhacr/clamp(rimdn_g,170,900).

Hail collection, conditional on lhl>1:

- Cloud qhlacw/chlacw/vhlacw (15802-15855, 16537-16618) mirrors the
  graupel cloud route, with 0.5/dt caps, hail fall speed clipped by dz/dt,
  and a 500 kg m-3 lower rime-density bound.
- Ice qhlaci/chlaci (15872-15877, 16620-16639, 17242-17269) uses ehli=.2,
  but is zero if T>273.15 or qc<qcmin.  Preserve this unrelated cloud gate.
- Snow qhlacs/chlacs (15864-15870, 16641-16660, 17296-17323) uses
  ehls=min(.5,.1*exp(.1*min(Tc,0))) and has no cloud gate.
- Rain qhlacr/chlacr/vhlacr (15857-15862, 16663-16709) uses efficiency 1.
  Above freezing committed mass/number are zero but raw qhlacrmlr survives.
  In the cold route raindn(h) remains its initialized 900 kg m-3, hence
  vhlacr=rho*qhlacr/900.

WRF diagnoses qhdry/qhwet and the shedding decision before wet-growth
rewrites qhaci/chaci/qhacs/chacs to unweighted raw collisions at
19668-19680 (hail 19745-19758).  Those earlier decisions are intentionally
stale; do not recompute them after the rewrite.

### 2. Snow self-aggregation

At 15611-15654 and 16933-16957:

- ess is zero at and below -15 C; for -15<Tc<-10 it is
  .5*exp(-.5)*(Tc+15)/5; for -10<=Tc<0 it is .5*exp(.05*Tc);
  it is zero at and above 0 C.
- Only snow number changes:

      csacs = rvt*aa2*ess*Ns^2*min(xv_s, 4*pi/3*(.02)^3)

  capped at 0.1 Ns/dt.

### 3. Crystal-to-snow conversion

The active iscni=4 path is 19335-19388:

- Require qi>qimin, qidpv>0, and Di3>=100 um.
- qscni=min(.5,Di3/200um)*qidpv.
- cscni=qscni*rho/max(rho_qs*xvmn_s,mass_i); cscnis=cscni.
- The separate ice-aggregation addition is skipped by iscni=4.
- qscnvi/cscnvi remain exactly zero: the ipconc>=4 route at 18206 is
  guarded by an explicit .false.; the alternative is ipconc<4.

### 4. Snow/graupel/hail conversion

Ice to graupel, iglcnvi=1 (19774-19844):

- Require T<273.0 K and qiacw-qidpv>0.
- Diagnose rime density with the 170-900 clamp; require it >=200 kg m-3.
  Let r=max(.5*(rho_i+rime_density),170).
- qhcni=qiacw-qidpv; chcni=Ni*qhcni/Qi is the ice donor number;
  chcnih=min(chcni,rho*qhcni/(r*xvmn_g)) is the graupel receiver number;
  vhcni=rho*qhcni/r.

Graupel to hail, ihlcnh=3 (19847-20094):

- Preserve the dg0/wtest construction at 19882-19968.  The final gate is
  wtest, qhacw*dt>qxming (cloud collection only), T<271.15 K, and
  qg>1e-4.
- Integrate the gamma-distribution tail above dg0:

      qxd1 = Qg*gaminterp(ratio,alpha_g,moment4,upper)
      qhlcnh = qxd1/dt
      chlcnh = chlcnhhl
              = Ng*gaminterp(ratio,alpha_g,moment1,upper)/dt

- For ipconc=5 retain the route only when qxd1>10*qxmin_h.
- vhlcnh=rho*qhlcnh/rho_g and
  vhlcnhl=rho*qhlcnh/max(500,rho_g).
- Hail receiver number is chlcnhhl*rzxhlh, not the graupel donor number.
  For admitted ipconc=5, all three predicted-reflectivity indices are zero
  (lzr=lzh=lzhl=0), so the initialization at 14034-14044 retains
  rzxhlh=rzhl/rz=0.4375.
  There is no later graupel donor mass or number limiter.
- Hail-to-graupel qhcnhl is disabled by icvhl2h=0.

Snow to graupel, iglcnvs=2 (20184-20301):

- Require qs>qsmin, qsacw>0, T<273.0 K, and qsacw-qsdpv>0.
- Rime density is bounded only above by 900 here; require it >=200.  Let
  r=max(.5*(rho_s+rime_density),170).
- qhcns=qsacw-qsdpv; chcns=Ns*qhcns/Qs is donor number;
  chcnsh=min(chcns,rho*qhcns/(r*xvmn_g)) is receiver number;
  vhcns=rho*qhcns/r.
- The qscnh graupel-to-snow assignments are commented out and remain zero.

### 5. Melting, soaking, and receiver number

Ventilation/melt setup is 18240-18559; operative melt is 18560-19333.
Only T>273.15 K melts.  At and below 273.15 the rates are zero.

At 18534-18544:

    fmlt1 = 2*pi*(Lv*Dv*(qvs0-qv) - k*Tc/rho)/Lf
    fmlt2 = -cw*Tc/Lf

Raw mass rates are

    qsmlr = min(c1sw*fmlt1*Ns*swvent*Ds1, 0)
    qhmlr = min(fmlt1*Ng*hwvent*Dg1
                + fmlt2*(qhacrmlr+qhacwmlr), 0)
    qhlmlr = analogous hail expression.

The final donor caps at 18696-18710 are 70 percent of snow and 95 percent of
graupel/hail per dt.  The odd nested min in the snow source still produces
the 70 percent bound.

Soak is diagnosed before those final caps at 18646-18653 and 18678-18685
and is therefore intentionally stale.  Both graupel and hail code use
v2=-rho*qmlr/rho_max despite the hail comment saying 50 percent.

Donor and rain-receiver number routes are distinct:

    csmlr = Ns/Qs*qsmlr
    csmlrr = csmlr/rzxs
    chmlr = Ng/(Qg+1e-20)*qhmlr
    chlmlr = Nh/(Qh+1e-20)*qhlmlr

With effective imltshddmr=1, for graupel

    A = -rho*qhmlr/min(1000*xvmx_r,rho_g*xv_g)
    B = -rho*qhmlr/(1000*volume_of_3mm_drop)
    I = A*(D0-Dg3)/(D0-Dshed)
        + B*(Dg3-Dshed)/(D0-Dshed)
    chmlrr = -max(A,min(B,I))

and hail is analogous with hard-coded D0=20 mm.  Rain number uses the
receivers -chmlrr/rzxh, -chlmlrr/rzxhl, and -csmlrr, never the donor
number rates.  The exact default receiver factors derived at
14032-14050, 14355-14383, and 15053-15074 are
rzxs=.3, rzxh=1, rzxhl=.4375, and rzxhlh=.4375.  These follow from
imurain=1, imusnow=3, alphar=alphah=0, alphahl=1, and
lzr=lzh=lzhl=0 at 1882-1909 and 14034-14049.  Port their source
expressions and assert these numeric defaults in the oracle; values derived
for a predicted-reflectivity configuration are not valid for ipconc=5.

### 6. Wet growth and shedding

Source order is 19416-19772:

- qhdry and qhldry are formed at 19416-19434.
- Only for 243.15<T<273.15 is the wet-growth heat budget qhwet/qhlwet
  evaluated (19451-19468).  Otherwise wet=dry.
- qhshr=min(0,qhwet-qhdry), and similarly hail.  Below 243.15 it is forced
  to zero.  At T>273.15 the snow/graupel/hail shedding rates are explicitly
  the negatives of their cloud/rain accretion terms (19534-19554).
- Exact boundaries matter: T=243.15 takes wet=dry and sheds zero;
  T=273.15 also takes wet=dry and is not in the T>273.15 branch.
- Wet growth is true only if qshr<0 and T<273.15.
- Collector number rates chshr/chlshr stay zero.  Rain receiver rates are

      chshrr = rho*qhshr/(1000*vshdgs_g)
      chlshrr = rho*qhlshr/(1000*vshdgs_h)
      crshr = chshrr/rzxh + chlshrr/rzxhl

  and crshr is negative, so rain number adds -crshr.
- vshdgs (15331-15349) uses area-weighted diameter
  (3+alpha)*Dchar.  Above 20 mm it chooses a 1.5 mm drop volume divided by
  massfacshr; above 8 mm it chooses a 3 mm drop; otherwise it uses
  min(xvmx_r,6/pi*rho_ice*.001*D^3)/massfacshr.  massfacshr defaults 4.5.
- In wet growth WRF subsequently rewrites frozen-collection rates to their
  raw collision values and resets dense volumes toward max density/soak at
  19624-19758.  Keep the pre-rewrite shedding decision stale.

### 7. Hallett-Mossop secondary ice

The active route is 20445-20639:

- Require qc>qcmin, either graupel or hail above its minimum,
  265.15<=T<=271.15 K, and positive cloud per-particle volume.
- Cloud-droplet fraction:

      ex1 = (1/250)
            * gaminterp((1+alpha_c)*7.23e-15/xv_c,
                        alpha_c,moment1,upper)

- Temperature factor:

      ft = clamp(-.11*Tc^2 - 1.1*Tc - 1.7, 0, 1)

- On a dry collector surface:

      chmul1 = ft*ex1*chacw
      qhmul1 = cimas0*chmul1/rho

  with the analogous hail route.  wetsfc/wetsfchl come from the
  source-ordered shedding decision and are forced true in wet growth.
- Collector number is unchanged.  Splinter mass is removed from graupel or
  hail and added to qi; chmul1/chlmul1 add to Ni.
- nsplinter=0, isnwfrac=0, and ihrn=0 make the other secondary-ice routes
  exactly zero under defaults.

## Shared limiters and intentional stale dependencies

Number source order is 20907-21297:

1. Form the ice-number aggregate.
2. Apply the cloud-number donor limiter to ciacw, cloud freezing, cracw,
   csacw, chacw, chlacw, and companions.  The source correction subtracts
   (1-fraction) times already-scaled receiver fields rather than rebuilding
   every aggregate.
3. Apply the rain-number donor limiter to ciacr, crfrz, chacr, chlacr,
   crcev, and cracr.
4. Apply the snow-number donor limiter.  It scales donor chcns, but the
   later graupel receiver uses the earlier unscaled chcnsh.
5. Graupel number has no shared donor limiter.
6. Form hail number and apply the hail donor limiter.  Graupel number was
   already formed before this can rescale chlcnh, so that recipient is stale.

Keep donor and receiver variables distinct:

- ice to graupel: chcni versus chcnih;
- snow to graupel: chcns versus chcnsh;
- graupel to hail: chlcnh versus chlcnhhl*rzxhlh;
- melt/shedding donor rates versus the rain receiver rates above.

Mass source order is 21400-21809:

- Cloud limiter 21454-21500 scales qiacw, qsacw, qhacw, qhlacw and their
  collection volumes.  qhcni/qhcns and vhcni/vhcns were already diagnosed
  and are not rescaled.
- The ice aggregate is then formed from scaled qiacw but unscaled qhcni.
- Rain limiter 21572-21678 scales qiacr, qrfrz, qsacr, qhacr/vhacr,
  qhlacr/vhlacr, and evaporation.  Number aggregates are already stale.
- Snow limiter 21687-21741 scales qhacs, qhlacs, qhcns, qsmlr, qssbv,
  qsmul, and companions.  Graupel mass later sees scaled qhcns, but vhcns
  remains stale.
- Graupel mass has no shared donor limiter.
- Hail limiter 21768-21809 can scale qhcnhl/qhlmul1 after earlier ice,
  graupel, rain, or vapor aggregates were formed.

These are source-order contracts.  A clean two-pass rescale/reaggregate
implementation is not bitwise or processwise equivalent to WRF.

## Exact dense-volume routes

At 22493-22670, with I=il5, the complete rates are:

    Vg_i =
      rho*[ I*ifiacrg*ffrzh*qracif/rhofrz
            + I*qhdpv/qhdpvdn
            + (qhacs+qhaci)/qhacidn ]
      + rho*max(qhcev,0)/1000
      + f2h*vhcns + vhacr + vhacw + vhfzh + f2h*vhcni
      + (ifiacrg*viacrf + ifrzg*vrfrzf)*ffrzh

    Vg_d =
      rho*[(1-I)*vhmlr + qhsbv + min(qhcev,0) - qhmul1]/rho_g
      - vhlcnh + vhshdr - vhsoak - vscnh

    Vh_i =
      rho*[ I*((1-ifiacrg)*ffrzh*qracif/rhofrz + qhldpv)
            + qhlacs + qhlaci ]/500
      + rho*max(qhlcev,0)/1000
      + vhlcnhl
      + ((1-ifiacrg)*ffrzh*viacrf
         + (1-ifrzg)*ffrzh*vrfrzf)
      + vhlacr + vhlacw + vhlfzhl

    Vh_d =
      rho*(qhlsbv + min(qhlcev,0) - qhlmul1)/rho_h
      + rho*(1-I)*vhlmlr/rho_h
      + vhlshdr - vhlsoak

Under defaults, ifrzg=ifiacrg=ffrzh=f2h=1,
qhdpvdn=qhacidn=170, and the hail collection divisor is the literal 500.
For nonmixed melt, vhmlr=qhmlr and vhlmlr=qhlmlr.

## Required oracle boundary matrix

The reference oracle must isolate at least:

- Every cross-collection family, with independent mass, donor number, receiver
  number, and dense-volume assertions.
- qiacw at cloud diameter 15 um, ice diameter 30 um, and T=273.15;
  qhacw/qhlacw at cloud diameter 2.4 um.
- qhacs with qc immediately below/equal/above 1e-13, rho_g around exactly
  300, and Ds3 at 40 and 150 um.  qhlaci needs its qc gate and T at/above
  273.15.  qsacr must prove exact zero.
- Snow aggregation at Tc=-15,-10,0 C.
- Crystal-to-snow at Di3=100 and 200 um and qidpv zero/positive.
- Ice/snow-to-graupel at qacw-qdpv=0, rime density=200, and T=273.0.
- Graupel-to-hail at qg=1e-4, qhacw*dt=qxming, T=271.15,
  converted mass=10*qxmin_h, and the dg0 tail boundary.
- Melt at T=273.15 and just above; isolated snow/graupel/hail; cloud/rain
  collision cooling; 70/95 percent caps; soak capacity; and all receiver
  factors (.3, 1, .4375, .4375).
- Shedding at T=243.15/just above and 273.15/just above; dry/wet heat-budget
  transition; 8/20 mm diameter boundaries; and the stale wet-growth rewrite.
- Hallett-Mossop at both 265.15 and 271.15 endpoints, its polynomial peak/cap,
  dry versus wet surfaces, and separate graupel/hail collectors.
- Combined shared-limiter cases that expose stale conversions, receiver
  numbers, and volume fields instead of only isolated rates.
- Each dense-volume source in isolation, followed by one combined case.
- Final finite/nonnegative state, water-mass closure within oracle tolerance,
  and dense-volume/density bounds.

## Fused-kernel discrepancies at the audit snapshot

The snapshot already calls diagnose_frozen_collection and includes the unusual
collection gates plus several collection volume routes.  However:

- No source-ordered conversion, melt/soak, shedding/wet-growth, or
  Hallett-Mossop diagnosis is called between diagnose_frozen_vapor and
  assemble_and_limit (kernel lines 1865-1877); only an insertion placeholder
  remains.
- assemble_and_limit already names many future rates.  Leaving their
  zero-initialized placeholders makes the aggregate topology look complete
  while silently omitting the process families above.
- Its dense-volume block at kernel lines 1711-1723 is explicitly only a
  vapor/collection subset.  It omits conversion, positive condensation,
  melt, soaking, shedding, and secondary-ice volume terms and uses a reduced
  qraci density route.
- The source-order limiter skeleton must be rechecked once those rates are
  populated: receiver aliases, pre-limiter conversion volumes, and stale
  aggregate dependencies cannot be reconstructed after a generic limiter.

Acceptance is not "all fields remain finite."  It is agreement with the
official WRF oracle at the isolated boundaries above, then combined-rate
agreement in exact source order, before any full-column or 1974 forecast
verification is considered meaningful.
