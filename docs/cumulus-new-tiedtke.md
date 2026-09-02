# New Tiedtke (`cu_physics = 16`)

New Tiedtke is ArWen's third cumulus scheme beside Kain-Fritsch (`cu_physics = 1`)
and Grell-Freitas (`cu_physics = 3`). It is the WRF v4.6.1 `module_cu_ntiedtke`
scheme, ported stage by stage to CUDA and graded against the byte-frozen Fortran
at every capture boundary. This page is what a user needs; the port record with
its measurements is under `docs/ntiedtke/`.

## Selecting it

* `cu_physics = 16`, per domain, on the tree route (`reachability:
  component-override`, the same posture as Grell-Freitas).
* `cudt_minutes = 0` is enforced: the scheme is called every step, as WRF calls it.
* A PBL scheme is required (`ysu`, `mynn`, `shinhong`, `sase` or `myj`).
  `validate_run_config` refuses `cu_physics = 16` with `bl_pbl_physics = 0` and
  names why: the closure reads the boundary-layer tendencies and the surface
  fluxes, and with no PBL scheme nothing writes them. WRF itself does not
  prohibit the pairing; the refusal is ArWen's structural check.
* No new required configuration. `ntiedtke_tiedtke_closure` (default `False`)
  is the one new knob, described below.

## What is proven, and what is not

Conformance is measured at the stage level and at the assembled-pipeline level,
never on a fragment. `tools/ntiedtke_wrf461_oracle` drives the frozen
`module_cu_ntiedtke.F` at `gfortran -O0` over 18 cases at 6 grid spacings (108
columns), the pre- and post-run conversions are graded field by field at zero
differing words, all 21 CUDA stages reproduce their captured boundaries bitwise,
and the assembled pipeline reproduces the oracle's level output bitwise from the
driver inputs alone. The oracle corpus ships under `gpuwm/data/ntiedtke/oracle/`
and the suite is `tests/test_ntiedtke_*.py`.

The maturity label is `implemented-unverified` with `scientific_evidence: none`.
No scored forecast comparison against observations exists for this scheme yet,
which is the project's bar for `supported`.

Chunking does not imprint: the domain is walked in chunks capped at the
Grell-Freitas tile, and the same 30-minute two-domain forecast at chunk widths
38,870 and 19,424 columns produced nine of nine byte-identical output files, with
a positive control confirming the comparison could see a difference. A card with
a different SM count therefore gives the same numbers.

## The deep closure, and the flag

New Tiedtke's deep first guess carries no thermodynamics. It is one tenth of the
mass-flux cap, and the cap is the pressure thickness of the model layer holding
cloud base divided by `g dt`. Classic Tiedtke (`cu_physics = 6`) uses moisture
convergence there. On a stretched vertical grid a lower cloud base lands in a
thinner layer, so a region with a low cloud base (a hurricane eyewall, where
saturated inflow lifts to condensation almost immediately) receives a smaller
first guess than its surroundings for a reason that is entirely about the grid,
and the penalty grows as the base descends. Measured on a real tropical-cyclone
case, deep columns only, the storm core received about 60 percent of the outer
region's first guess. This is a property of the WRF scheme, reproduced exactly;
it is recorded here because it is invisible in a column oracle and only shows
on a storm.

`ntiedtke_tiedtke_closure = True` runs the New Tiedtke kernels with classic
Tiedtke's deep closure: the fixed 2400 s adjustment time and the
moisture-convergence first guess in place of the scaled time and the geometric
fraction. Those two substitutions are the entire difference between the two
schemes' deep arms. `False` leaves every `cu_physics = 16` result bit-identical
to the port. A 30-minute A/B on the same case showed the path is live (seven of
nine later frames differ, the first is byte-identical) and the sign (more
deepening under the classic closure); the magnitude at that length is below
what one run resolves.

## Restart, output and memory

* The scheme's state rides across a checkpoint; a restart under `cu_physics = 16`
  resumes rather than re-initialises.
* Per-level diabatic heating is now written for every configuration:
  `H_DIABATIC`, `RTHRATLW` and `RTHRATSW`. They were computed every step and
  discarded; writing them changes no forecast value.
* The kernel local-memory ceiling gains an `ntiedtke` row (measured on an RTX
  5070 Ti with NVRTC 13.0.88). The table is fail-closed by module, so without the
  row the scheme could not be priced; no existing module's ceiling moves.
* Nest relocation now asks the cumulus adapter to release its driver reference.
  The driver and the adapter formed a reference cycle that CPython's refcounting
  cannot collect, so a relocating run kept every dropped driver graph resident:
  measured per relocation, Grell-Freitas grew 35 MiB, New Tiedtke 61 MiB,
  Kain-Fritsch 1.3 MiB. The relocation receipt carries
  `cumulus_workspace_bytes`.

## Open

* `rthcuten` is not exposed. It lives on a per-step dataclass rather than in
  state, so exposing it is a lifetime change, not a schema entry. It is the one
  term a per-level theta budget cannot localise without.
* Forecast skill against observations is unmeasured.
