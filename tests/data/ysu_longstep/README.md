# YSU explicit entrainment regression

Real engine inputs, x1.40962, GFS cycle 2026-10-11 12Z, cell 980,
step 2 at 600 s (12:20 UTC). Captured before/after `launch_ysu` without
changing the run. Engine baseline 216bdcd2cd8 (2.9.0 release lineage),
hex 10457899 plus admission-only b0a9d584. Native MPAS v8.4.2 unmodified
`bl_ysu.F90` and hex exact port reproduce all baseline output words on
this column. The vapor step adds 10.41 g/kg at level 19 across a 1.68 g/kg
jump. WSM6 then condenses it and produces a hot single layer.

`out_*` are baseline outputs, not acceptance targets for the corrected
closure. Inputs retain the public launch API names. The full lane evidence
(the native hot-layer replay receipt) stays with the release record and is not shipped.
