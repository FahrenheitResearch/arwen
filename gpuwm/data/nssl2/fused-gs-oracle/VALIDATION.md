# NSSL option-18 fused-GS official-source validation

## Authority and scope

The acceptance fixture is produced by directly calling official NCAR WRF
v4.6.1 `nssl_2mom_gs` at commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`.  The canonical-LF SHA-256 of
`phys/module_mp_nssl_2mom.F` is
`1eb1b138b75ff3b0cfe33c23779f4ec9b72e57a5455a53ef11c9e55ae0f42722`.
Acceptance changes only private/public accessibility; no executable WRF
statement is patched.

This is the full coupled GS process oracle, not a sum of isolated kernels.  It
includes native ordering, shared donor/heat/vapor limiters, final moment and
dense-particle-volume bounds, and late in-GS state updates.  It excludes the
driver sedimentation call and the later, separate `NUCOND` and `QVEXCESS`
driver branches.

## Locked evidence

- `fused-gs.csv`: 240 rows, 50 locked columns, SHA-256
  `fc27cd1c1a9a1ddefcd086551d0a0ea53f731800bf398445de684d53fcf15971`.
- `fused-gs-diagnostics.csv`: 27,410 named long-form rows, SHA-256
  `f2281b6fdfed3daa6cd66b6dfb9e9bf949a8a4a232bdc02488635f6b3dae0d69`.
- Raw diagnostic trace SHA-256 (rebuild artifact, not acceptance input):
  `422294fd0cb9face65819644ca8f9f0a83c907762e5cbc0578cee88ca84783b2`.

Two separately compiled acceptance directories produced byte-identical
fixtures.  A third executable was built from a separate source copy with
diagnostic writes patched in; its canonical state output also compared
byte-for-byte with acceptance before the trace was admitted.  The diagnostic
patch is therefore evidence-only and cannot change the acceptance artifact.

## Coverage

Thirty four-level slabs run twice with identical results.  They cover
dt=0.1/1/10/60/300 s, zero/threshold cleanup, all-active warm/cold/mixed
states, independent cloud/rain/ice/snow donor pressure, shared frozen vapor
deposition and sublimation, wet growth/melting/shedding/Hallett--Mossop,
variable 50--1500 kg/m3 dense-category input moments, and exact 243.15,
265.15, 268.15, 270.15, 271.15, and 273.15 K gates.

The expanded rain-freezing slabs preserve official option-18 defaults,
including `qxmin(r)=qxmin(g)=1e-12`, `ibiggopt=2`, and `ibiggsnow=3`.  They
cover strict -5 C and 8-mm Bigg gates, the strict 0.3-mm snow split, just
above/equal/below -30 C heat-budget override, dt*q=1e-12 and dt*c=1e-8 donor
caps, malformed mass-only/number-only moments, negative through positive
`fwet1`, simultaneous Bigg plus qiacr, qr/dt and shared heat caps, and long-dt
competition.  The diagnostic trace records pre-cold `qrzmax`, final `qrzmax`,
`qrztot`, `qrzfac`, split mass/number/volume rates before and after scaling,
and all shared process/donor limit fields.

The validator measured a rain heat factor range of 0--1 and a diagnosed Bigg
diameter range of `3.03239312e-6`--`5.18665671e-1` m, crossing both locked
diameter gates.  Every canonical and diagnostic numeric field is finite.

## State contract

Masses are kg/kg; number and predicted CCN are #/m3; predicted graupel/hail
volume is m3/m3.  `w_center_m_s` is the native
`0.5*(w(k)+w(min(k+1,nz)))`, so the top cell uses a clamped upper index.
`primary_ice_target_m3` is the complete four-level t7 slab passed into GS.

The fixture intentionally records both native final temperature and final
potential temperature.  WRF's late warm cloud-ice melt changes the state
after its saved temperature diagnostic, so these are independent official
outputs; consumers must not replace one with a recomputation from the other.

## Reproduction

From a fresh official checkout:

```text
tools/nssl2_wrf461_fused_gs_oracle/build.sh \
  /workspace/WRF-v4.6.1-fused-gs-oracle-source-20260722 \
  /workspace/nssl2-fused-gs-oracle-build-v5-final-20260723
```

GNU Fortran 13.3.0 was used for the locked reproduction.  The build refuses a
dirty checkout, an incorrect commit/source hash, or an existing build path;
then rebuilds twice, compares all three state outputs, normalizes diagnostics,
and runs the schema/numerical validator.
