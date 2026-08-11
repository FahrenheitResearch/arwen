# Physics options and maturity

Every physics scheme in ArWen is a transcription of WRF v4.6.1 source
(commit `d66e442f`), and every option carries a machine-readable
maturity label in the physics registry
(`gpuwm/physics_registry_v2.json`) -- the registry, not this page, is
the authority, and `tests/test_registry_reachability.py` keeps the two
from drifting. Labels never substitute for the evidence behind them;
each row below links the strongest measurement that exists for that
option.

## Maturity vocabulary

| label | meaning |
|---|---|
| **model-validated** | A matched multi-hour ArWen-vs-WRF forecast of the reference case has been run with this option and its decay tables are published ([VERIFICATION.md](VERIFICATION.md)). |
| **validation-candidate** | Executable and gated, with a ratified reference comparison, but deliberately not the default; the next candidate for full matched-run validation. |
| **supported** | Production option from the longest-certified slice: WRF-transcribed, standing unit/runtime gates, exercised by the certified reference configurations. |
| **experimental-runtime** | Executable, and carrying a documented runtime restriction or an unratified composition -- a table-bound runtime, or a nest edge between two microphysics schemes. Selecting it warns and does not block. |
| **implemented-unverified** | Runs on the GPU and is column-oracle-measured against unmodified WRF Fortran, but no dedicated ArWen/WRF forecast-trajectory comparison exists for it yet. The registry records its measured ULP distances and open divergences verbatim. |
| **planned / port-in-progress** | Not selectable. The registry publishes the target so the roadmap is machine-readable; nothing can resolve to it. |

Every option on the first five rungs is yours to select, two ways that
end at the same loader: name a preset, or type the switch into your own
`[shared]`/`[[domain]]` block. The last rung is the only one out of
reach -- an option that is not ported yet refuses by name at load, and
never silently substitutes something else for it. The one exception is
`gpuwm import-namelist`, which performs three explicit, reported
substitutions (see the end of this page).

**A rung grades one OPTION, and most of the tree sits on the same
rung.** `implemented-unverified` is not a mark against any particular
scheme. It is carried by **23 of the registry's 40 component options**:
YSU, MYJ, MYNN (PBL and surface layer both), Eta similarity, Shin-Hong,
SASE, Milbrandt-Yau, Morrison, WDM6, P3 one-category, Thompson
aerosol-aware, Noah, Noah-MP, RUC, Grell-Freitas, WRF RRTM longwave with
Dudhia shortwave, the RTE+RRTMGP legacy-aggregate selector, and all five
turbulence closures. It says
exactly one thing -- no matched ArWen-versus-WRF forecast trajectory
has been run with this option yet -- and it says it about the option,
not about how freely that option may be composed with others. Those are
different questions, and the second one is answered by measurement in
the next section.

A *suite* then takes the **strict minimum** rung over the options it
selects (the registry's composition rule C2, "a composed suite is only
as conformant as its weakest member"), which is why a suite carrying
Thompson's matched-run microphysics beside YSU and Noah still reads
`implemented-unverified`. Nothing is graded down for being newly named
or newly composed, and adding a named suite never moves a rung.

## Can you compose your own suite? Yes, and it is measured

**The tables on this page are a CATALOGUE, not a gate.** They list the
suites ArWen ships under a name; they do not list the suites a config
may ask for, and they never have. Reading them as a gate has misled
readers twice -- one outside collaborator and one of this project's own
agents, who both concluded from a table cell that MYNN could not be run
with radiation, because no named MYNN suite carried it at the time.
Neither reader made a careless mistake; the page invited the reading.
So, plainly, and in the order the two routes actually work:

- **The preset route is a curated catalogue.** A shipped suite is a
  name (`--physics-profile`, a wizard menu entry, a route declaration)
  bound to a fixed set of switches that someone chose and evidenced.
  Choosing a name is how you pin a run to a published product, and the
  runner refuses on any switch drift from it.
- **The config route composes freely.** Writing the switches yourself
  in `[shared]` or a `[[domain]]` block asks the loader directly, and
  the loader's answer is governed only by the per-option validation
  rules -- not by whether any name exists for what you wrote. Most
  configurations users run are of this second kind.
- **A missing preset is a missing NAME, never a missing capability.**
  That is the entire content of the MYNN incident: composing MYNN with
  radiation was always accepted, and what did not exist was a row in a
  menu. Three now do (see [Nocturnal validity](#nocturnal-validity)),
  and the composition they name was reachable before they were added.

The composition space is measured rather than argued.
`tools/report_physics_composition_walk.py` writes 9781 physics
combinations into real experiment TOMLs, pushes every one through
`gpuwm.experiment.build_experiment` -- the single front door every
runner reaches a per-domain `RunConfig` through -- and records the
verdict. `docs/public/receipts/physics-composition-walk.json` is that
record, and `tests/test_physics_composition_walk.py` regenerates it on
every release cut and compares it byte for byte. As measured:

- **990 of 9781 combinations are accepted**, against 18 registered
  templates. The presets are a corner of the space, not the space.
- **Every accepted run keeps every switch the file set**, checked
  against the resolved per-domain `RunConfig`. Zero rewrites. An
  admission is never a silent substitution.
- **Every admitted value of every axis reaches an accepted run**, with
  no exceptions left. `ra_lw_physics = 1` (WRF RRTM longwave) was the
  last one; 1.9 ports it, and it now reaches 169 accepted combinations
  as WRF's classic pair with Dudhia shortwave.
- **8791 refusals fall into 19 distinct rules**, every one of which
  names the selector to change, and each of which has a
  demonstrated remedy -- the receipt carries a before/after pair per
  rule showing that doing what the message says reaches an accepted
  run.
- The axis tables themselves are measured, not transcribed: every
  integer in [-2, 99] plus 900 and 901 is offered to the loader on each
  axis, and the admitted set it reports must equal the declared one.

The scope is config admission, which is where this question lives; the
receipt's `scope` block states what it does not cover (execution,
per-source route declarations, multi-domain trees).

### Reachability is a menu axis, not a loader gate

This is the distinction the whole misunderstanding turned on, so it is
defined once here and every table that carries the column links back to
it.

`reachability.state` is a field of the physics registry, and the
registry declares its own meaning
(`authority.reachability_declaration`), quoted in full:

> `components.<component>.options.<option>.reachability` declares how a
> user can select the option: `'template'` through a registered base
> template, `'component-override'` through either a route's full
> `allowed_component_overrides` or its option-scoped
> `allowed_component_options`, `'expert-template'` only through a
> route's `expert_template_ids` with its `expert_acknowledgement_id`,
> and `'unreachable'` not normally reachable -- which must name a
> blocker. implemented and reachable are independent.
> `tests/test_registry_reachability.py` recomputes every state.

In plain words -- and every one of these is a way IN, not a wall:

| registry state | what it means for you |
|---|---|
| `template` | **a preset exists**: pick it by name and every later stage enforces it switch for switch |
| `component-override` | **a preset exists** on the routes that declare the override, and everywhere else you **type it in your config** |
| `expert-template` | a preset exists behind one gate: **add one acknowledgement line** and it runs |
| `unreachable` | no preset names it. Either **type it in your config** -- three of the five below do run that way -- or it is **not ported yet**, and then it refuses by name |

Every one of those four states is a statement about the **named**
routes -- what a menu, a `--physics-profile` choice list or a route
declaration can hand you, all three of which are keyed by name. None of
them is a statement about what `build_experiment` accepts from a config
you wrote yourself. Two consequences, both measured:

**`template` does not mean "only what the template selects".** MYNN is
`template` on both of its halves, and that cell says nothing whatever
about composition. The walk's 64-cell MYNN matrix (`mynn_matrix` in the
receipt) crosses the MYNN PBL + MYNN surface-layer pair against all
four land surfaces and all sixteen longwave/shortwave pairings: **the
four land-surface columns are identical, cell for cell**, which is what
arbitrary composition actually looks like -- the radiation verdict does
not depend on the land-surface model. MYNN reaches every radiation
pairing the loader admits at all (`0/0`, `0/1`, `4/4`, `90/90`), all
seven microphysics schemes and all three cumulus schemes.

**`unreachable` does not mean the loader refuses.** It means no
template and no route selects it, and the registry must publish a
blocker saying why. Whether a hand-written config can still reach it is
a separate, measured fact, and it differs per option -- so the page now
prints both:

| registry `unreachable` option | selectors | what a config naming it actually gets |
|---|---|---|
| land surface `off` | `sf_surface_physics = 0` | **type it in your config**: **255 accepted**. Off the menus by policy (its blocker says so), not by the loader |
| surface layer `off` | `sf_sfclay_physics = 0` | **type it in your config**: **50 accepted**, same reason |
| radiation `analytic-clear-sky` | `ra_lw_physics = ra_sw_physics = 90` | **type it in your config**: **200 accepted**, same reason |
| radiation `wrf-rrtm-dudhia` | `ra_lw_physics = 1` | **not ported yet -- refuses by name** at load: 0 accepted of 1600 tried |
| microphysics `sase` | none declared | **not ported yet**: publishes a porting target and declares no selector, so nothing can resolve to it |

Counts are per-axis accepted totals from the same walk. Read a blocker
before using any of the first three: each states why the option is kept
off the named routes, and that reasoning applies to your config too
even though the loader does not enforce it.

## Microphysics (`mp_physics`)

| option | WRF id | maturity | evidence, in one line |
|---|---|---|---|
| Kessler | 1 | supported | warm-rain certified slice; idealized + runtime gates |
| WSM6 | 6 | supported | certified slice; matched-run anchors exist for this scheme on the reference case (refl corr 0.977 at F2, 0.815 at F5, d03) |
| Thompson | 8 | **model-validated** | full matched 6 h, 4-domain run to 500 m; decay tables in [VERIFICATION.md](VERIFICATION.md); WRF's own coefficient tables packaged and SHA-256-validated at load |
| Milbrandt-Yau 2-moment | 9 | implemented-unverified | **no oracle has been run.** Line-by-line transcription of `phys/module_mp_milbrandt2mom.F`; what is tested is a column smoke through the shipped seams (finite, bounded, water budget closing to 1.3e-4 relative or better on three seeding layouts, each resolving a named family of source/sink terms) plus a mutation control. Graupel and hail are separate prognostic categories and all twelve moments are transported. Reachable only as a per-domain override; refuses the RTE+RRTMGP pairing (see below) |
| Morrison 2-moment | 10 | implemented-unverified | 28-column oracle vs unmodified WRF `MP_MORR_TWO_MOMENT`: theta within 154 ULP, but hydrometeor fields cross branch points and are not bitwise; both rimed-ice identities (graupel/hail) implemented |
| WDM6 double-moment warm rain | 16 | implemented-unverified | **no oracle comparison against the WRF Fortran has been run** — the CUDA kernel and `wdm6init` are transcribed line by line from the byte-frozen `phys/module_mp_wdm6.F` with file:line citations, the float64 coefficient block pins the kernel's baked FP32 literals, and a column smoke through the shipped seams asserts finiteness, WDM6's own bounds, water conservation to the surface flux, and that CCN activation actually moves number from `nn` into `nc`; the oracle campaign is the declared next stage. WDM5 (14) and WDM7 (26) are refused by name. Reachable only as a per-domain override |
| NSSL 2-moment | 18 | **validation-candidate** (default lane) / implemented-unverified (variants) | full CUDA port with fused-process oracles and a ratified 500 m comparison; explicitly not the default. The hail-off and diagnosed-CCN variants below carry column smoke and treatment proofs only, with no oracle comparison |
| Thompson aerosol-aware | 28 | implemented-unverified | 22 WRF column fixtures end to end, 23 quantities each: 17 clear a flat 2e-6 gate, 4 do not, 1 clears only under two named allowances (numbers below); it runs multi-step and stays bounded; the one matched WRF forecast comparison is idealized only — a single-domain doubly periodic warm bubble, [validation/mp28-matched-trajectory.md](validation/mp28-matched-trajectory.md), which publishes a failed declared condition alongside a control showing that condition fails for WRF against its own recompilation — and no real-data or nested forecast has ever been validated against WRF; reachable only as a per-domain override |
| P3 one-category | 50 | implemented-unverified | **no oracle has been run** — column smoke only: a 15-step 3-column integration through the shipped seams stays finite and non-negative, conserves total water against surface precipitation to 1e-4 relative, and holds P3's own invariants (rime mass ≤ ice mass, 50 ≤ rime density ≤ 900); WRF's `p3_lookupTable_1.dat-v5.4_2momI` is packaged verbatim and SHA-256-validated at load, but nothing yet checks that gpuwm's interpolation of it reproduces WRF's; CPU-only, no CUDA mirror; reachable only as a per-domain override |

### P3 one-category (`mp_physics = 50`) — read this before selecting it

P3 predicts particle properties instead of sorting ice into species. There
is **one ice category**, and it carries a predicted rime mass fraction and
rime density that together span the range other schemes split into snow,
graupel and hail. The practical consequences are immediate:

- **An mp=50 run has no `QSNOW` and no `QGRAUP` field at all**, and no
  `GRAUPELNC` accumulator. They are not zero — they do not exist. Output,
  rendering and verification tooling written against the six-species
  inventory will not find them.
- Its ice inventory is instead `QICE` with `QIR` (rime mass), `QIB` (rime
  volume) and `QNICE`, plus `QNRAIN` for rain. Bulk rime density is the
  ratio `QIR/QIB`, bounded to 50–900 kg m⁻³.
- Snowfall still reports: `SNOWNC`/`SNOWNCV` and the solid fraction `SR`
  come from the ice category's own sedimentation flux.

**The rime pair travels with the ice.** `QIR` and `QIB` are advected,
mixed, forced across nest edges and written into checkpoints on exactly
the same footing as `QICE` — WRF declares all four in the `p3_1category`
package (`Registry.EM_COMMON:3038`) and advects the `scalar` array beside
the `moist` one. This is not bookkeeping: rime fraction `QIR/QICE` and
rime density `QIR/QIB` are the two indices that select every ice fall
speed and collection rate from the lookup table, so a rime pair that sat
still while the ice moved would describe ice that is no longer there.
Neither enters the water loading: `QIR` is a *component* of `QICE`, not
extra mass, and `QIB` is a volume.

**What a restart carries, and what it does not.** `QIR`, `QIB` and P3's
two cross-step supersaturation carriers `th_old`/`qv_old` are all
serialized, matching the `r` in WRF's own IO strings
(`Registry.EM_COMMON:555-558` and `:1598-1599`); a resumed mp=50 run
picks up the same rimed ice and the same supersaturation tendency it
stopped with. The checkpoint also binds the lookup table's SHA-256, so a
resume onto different table bytes is refused rather than continued —
P3's process rates *are* that table. The three ice diagnostics
`vmi3d`/`di3d`/`rhopo3d` are deliberately not carried, because `p3_main`
zeroes them on entry and refills them before returning. WRF's `itimestep`
is not carried as a number either, but its one effect is: the scheme reads
only "is this the first step", and the adapter answers that from the model
clock, so a resumed run does **not** replay the first-step saturation
adjustment. `th_old`/`qv_old` are written to the restart stream but
**not** to `wrfout`, again following WRF, whose IO string for them
(`rusd`) carries no `h`.

**Only the 1-category, 2-moment-ice configuration is ported.** WRF v4.6.1
maps P3 across four selectors (`Registry.EM_COMMON:3038-3041`), and the
other three are refused by name with the physics each one adds:

| refused | what it is | why it is refused |
|---|---|---|
| `51` | prognostic droplet number | this port specifies `nc = nccnst/rho`; the CCN-activation path is absent |
| `52` | two ice categories | inter-category collection, the category-merge pass and lookup table 2 are absent |
| `53` | three-moment ice | the prognostic reflectivity moment `zitot` and the 3momI table are absent |

**It is CPU-only.** The float32 authority runs on the host and the adapter
round-trips device state every step. That is correct but slow: mp=50 suits
column smokes and small-domain verification, not production-domain
throughput. A CUDA mirror is future work alongside the oracle campaign.

**What "implemented-unverified" means here** is exactly what the registry
row says: the scheme has never been compared against WRF Fortran. No ULP
measurement, no matched run, no trajectory comparison. Everything asserted
about it so far is physics (conservation, boundedness, sign) or gpuwm's
consistency with itself. The WRF-Fortran oracle campaign is the declared
next stage, as it was for Shin-Hong and Grell-Freitas.

One transcription note worth publishing, because it is visible to anyone
who instruments the first step: WRF's own first P3 call evaluates `0/0`
for the supersaturation ratios. Nothing initialises `th_old`/`qv_old`, and
the `max(t_old,1.)` guard leaves the saturation mixing ratios at exactly
zero. The port keeps the resulting NaN rather than repairing it — every
consumer is a comparison, and no finite value makes them all false the way
a NaN does, so a "fix" would silently change WRF's first-step control flow.
Its containment is tested, not assumed: no prognostic, diagnostic or
surface field goes non-finite.

### Milbrandt-Yau (`mp_physics = 9`) — read this before selecting it

The scheme the tornado and hail literature reaches for: double-moment in
all six hydrometeors, with graupel **and** hail carried as separate
prognostic categories rather than one rimed-ice slot chosen by a switch.
Twelve fields are transported (`qc qr qi qs qg qh` and
`nc nr ni ns ng nh`).

**No oracle has been run.** This is the honest scope line and it is the
whole reason the row says implemented-unverified. The port is a
line-by-line transcription of the byte-frozen WRF v4.6.1
`phys/module_mp_milbrandt2mom.F`, and what has been *measured* is a
column smoke driving the shipped entry points — finiteness,
non-negativity, a physical temperature band, moment-versus-mass
consistency, and a column water budget that closes to the reported
precipitation on three seeding layouts (6.4e-5, 6.1e-5 and 1.3e-4
relative) — plus a mutation control proving the smoke fails when the
scheme is stubbed out.

Read the budget narrowly. Conservation can only see a source/sink term
where that term is non-zero on the column being run, so the three
layouts exist to make three families fire: vapour deposition and
ice-to-snow autoconversion, cloud/rain-frozen riming, and melting. Terms
that are zero on all three columns are not covered, and the number-only
source/sink lines carry no mass and cannot appear in a water budget at
all. `tests/test_milbrandt2.py` names each term it resolves and the
measured margin it resolves it by.

There is no ULP table, no column oracle and no matched forecast
trajectory. Running one is the declared next stage, exactly as it was for
Shin-Hong and Grell-Freitas before their measurements landed.

**One identity, not a family.** WRF exposes no namelist for any
Milbrandt-Yau switch: the driver hard-codes continental CCN, all six
process switches on and WRF's own level ordering, and the scheme body
hard-codes non-spherical snow and Meyers-plus-contact ice nucleation. So
mp=9 has exactly one form in WRF v4.6.1 and ArWen ships that one;
`gpuwm/config.py` refuses by name any request to move one of them,
because the 154-entry constant table is derived under those settings.

**Radiation.** WRF leaves `has_reqc/has_reqi/has_reqs` at 0 for this
scheme and the scheme's own effective-radius block is commented out, so
it hands radiation no radii. The legacy RRTMG port computes its own, the
way WRF does; the RTE+RRTMGP variant is **refused** rather than given an
invented cloud-optics coupling. Use `ra_rrtmg_variant = "rrtmg_legacy"`
or Dudhia shortwave.

**Reflectivity** comes from the scheme itself: WRF binds its `Zet` output
straight to `refl_10cm`, so a history frame carries Milbrandt-Yau's own
dBZ rather than ArWen's generic radar operator.

### NSSL variants (`nssl_hail_on`, `nssl_ccn_on`) — one scheme, four modes

WRF v4.6.1 has exactly one NSSL scheme. The `mp_physics` values 17, 19, 21
and 22 you may remember are compatibility spellings: WRF rewrites them onto
`mp_physics = 18` plus explicit flags before any physics runs, printing a
deprecation caution as it goes. ArWen performs the same rewrite, so a
namelist naming any of them imports without substitution.

Four modes have a ported numerical path:

| selectors | equals | maturity |
|---|---|---|
| everything unset (`-1`) | two moments, hail, predicted CCN, graupel and hail volume | validation-candidate (the shipped default lane) |
| `nssl_ccn_on = 0` | as above but CCN diagnosed, not predicted | implemented-unverified — WRF's deprecated `mp_physics = 17` |
| `nssl_hail_on = 0` | two moments, no hail, predicted CCN, graupel volume | implemented-unverified |
| `nssl_hail_on = 0, nssl_ccn_on = 0` | two moments, no hail, diagnosed CCN, graupel volume | implemented-unverified — WRF's deprecated `mp_physics = 22` |

A category a variant does not carry stays exactly absent. WRF gets that
for free — its Registry never allocates the field — while ArWen allocates
one field set for every mode, so it enforces the absence instead: the
fields the resolved mode excludes are pinned to zero when the domain is
built and again on every microphysics step, ahead of the nested-domain
boundary ring. A hail-off run cannot grow, advect, write or restart a
hail category, and the run receipt lists which fields those are.

Turning predicted CCN off is not simply "drop a field". WRF stops
allocating `qnn`, rebuilds the unactivated CCN from the base concentration
on every step, never stores it back, and raises `renucfrac` from 0 to 1 —
which changes the pool feeding droplet nucleation and arms a
low-temperature updraft limiter. All four behaviours are ported.

**What the variants are and are not backed by.** Every variant path has
been driven through the shipped production seam in two column regimes and
checked for finiteness, for keeping absent categories exactly absent, and
for actually differing from the default lane on identical inputs. None of
them has been compared against WRF Fortran: no WRF run, no matched
trajectory, no ULP measurement. That campaign is the declared next stage.
The default lane is unaffected — it reproduces its committed digests
byte-for-byte with all of this in place.

**Not ported, and refused rather than substituted.** The one-moment family
(`nssl_2moment_on = 0`, the deprecated `mp_physics = 19` and `21`) and the
three-moment extension (`nssl_3moment = 1`) have no ported path and are
refused by name at configuration time.

Two further combinations are refused because WRF itself is undefined
there: `nssl_hail_on = 2` with two moments, and `nssl_density_on = 1` with
hail on. Both selectors reach the scheme as plain logicals
(`module_physics_init.F:4647-4649` passes `flag > 0`), so the module cannot
tell 2 from 1 in either case and behaves as though the larger package were
active — reading, and in one case writing, a field the Registry never
allocated for the setting the user actually chose. The standing rule is to
implement the defined behaviour
and document the divergence rather than reproduce an undefined read, so
ArWen refuses and says which Fortran line it is refusing.

### Thompson aerosol-aware (`mp_physics = 28`) — read this before selecting it

mp=28 is a second, near-complete port of Thompson, not a flag on mp=8.
Cloud droplet number `nc` becomes prognostic (WRF's `Nt_c = 100e6`
constant disappears), two aerosol number tracers `nwfa`/`nifa` and two
surface emission fields `nwfa2d`/`nifa2d` are added, and about a dozen
processes with no mp=8 counterpart run: CCN activation, aerosol-only
droplet evaporation, five cloud-number sinks under a shared balance
limiter, six wet-scavenging rates, DeMott ice nucleation replacing
Cooper, Koop homogeneous haze freezing, and number-weighted cloud
sedimentation. It lives in its own CUDA translation units and its own
Python launchers; `thompson.cu` and `thompson.py` are byte-frozen and
mp=8 is unchanged by construction.

**Exactly what has been measured.** All 22 committed WRF v4.6.1 aerosol
column fixtures, driven end to end through the shipped adapter, compared
against unmodified `phys/module_mp_thompson.F` (gfortran 13.3.0 `-O2`),
one RTX 5090, FP32. Twenty-two, not nineteen: the 19 scenarios the port
spec names (`aero-*`) plus three `wp08-*` columns from the same oracle
build that pin every reachable `nu_c` and both branches of the terminal
phase cleanup. Each fixture is compared on **23 quantities** — 15
prognostic column fields, seven surface accumulations (`RAINNC`,
`RAINNCV`, `SNOWNC`, `SNOWNCV`, `GRAUPELNC`, `GRAUPELNCV`, `SR`) and
`REFL_10CM` against WRF's own `calc_refl10cm` — at **2.0e-6 relative**
and **2.0e-4 dB**. The gate asserts that width, so it cannot be narrowed
back. The registry's residual table is published in the narrower 16-field
shape (the 15 column fields plus `RAINNC`); the gate compares all 23.

| result | fixtures |
|---|---|
| clear a **flat** gate on every one of the 23 quantities — no bounds dict, no excluded level, no per-fixture carve-out | 17 of 22 — `aero-ccn-activate`, `aero-ccn-sweep`, `aero-drop-evap`, `aero-ice-demott-dep`, `aero-ice-demott-idxin`, `aero-ice-koop`, `aero-init-profile`, `aero-nc-accrete`, `aero-nc-auto`, `aero-nc-cap`, `aero-nc-effrad`, `aero-nc-sed`, `aero-scav-frozen`, `aero-scav-rain`, `aero-sfc-emit`, `aero-warm-overlap`, `wp08-melt` (16 of the 19 spec'd `aero-*`, plus one `wp08-*`) |
| clear only under a named allowance | 1 — `aero-reduces-to-classic`, taking the gated count to 18 of 22, and it now rests on ONE allowance rather than two. See the allowance table below |
| miss the gate | 4, listed field by field below |

**The one allowance, on that one fixture, and needed for it.** Nothing
here was ever widened, and two of the three this page used to carry have
been **retired** rather than kept:

| allowance | what it does | was | is |
|---|---|---|---|
| `_NEAR_CANCELLATION_LEVELS` | 0-based level 6 held to 32 ULP of the entry value instead of a relative bound | level skipped outright | **32 ULP; measured 0.585 (`qr`) and 0.159 (`nr`), down from 14.9 and 4.05 at the 1.4.1 merge** |
| ~~`_END_TO_END_BOUNDS`~~ | `nr` held to 1.0e-5 instead of 2.0e-6 | 2.5e-3, then 1.0e-4 for `qr` and `nr` both, then 1.0e-5 for `nr` alone | **RETIRED at the 1.4.1 merge. Level 5's `nr` fell from 5.700e-06 to 4.146e-07, inside the flat 2.0e-6 gate, so the bound bought nothing and the constant is now an empty dict** |
| ~~`_REFL_DB_BOUNDS`~~ | reflectivity held to 1.0e-3 dB instead of 2.0e-4 dB | 1.0e-2 dB, then 1.0e-3 dB | **RETIRED. The residual it existed for fell to 3.242e-05 dB and then to 9.537e-06 dB, inside the flat 2.0e-4 dB gate, so the fixture is held to the flat gate on reflectivity** |

**What the 1.4.1 merge changed, and what it did not.** mp=28 sits on
`integration/release-1.4.1` and inherited the mp=8 lane's two rain
sedimentation reconciliations — `5e4af4e3` and `cb765336` — in the
byte-frozen `thompson.cu` this scheme shares for rain fallout. No mp=28
file changed. Its measurements did: `aero-reduces-to-classic` `nr` at
level 5 5.700e-06 → **4.146e-07**, that fixture's worst ULP over all 23
quantities 27.5 → **4.0**, its worst |dBZ − WRF| 3.242e-05 dB →
**9.537e-06 dB**, and the `qr` levels where mp=28 is bit-exact against WRF
and the frozen mp=8 pipeline is not went from three to **four** (1, 2, 4,
5). Nothing moved the other way, and the four missing fixtures below miss
at the same numbers they did before.

`aero-reduces-to-classic` now measures **`nr` 4.146e-07 at 0-based level
5** — inside the flat gate — and nothing on the fixture is above the flat
gate once level 6 is taken in ULPs. That level 5 is the one level of the
column where the step removes a large fraction of the rain number without
emptying it (49.75%: 3.000000e+05 → 1.507546e+05 per kg); it used to carry
5.700e-06, 27.5 ULP of the entry value, and the bound that covered it is
the one the merge retired. Every level of the column is now inside 4 ULP.
It is **not** inherited from the classic path and is
no longer claimed to be: with WRF's level-wise sedimentation density
restored, mp=28 is now **bit-exact against WRF** at `qr` levels 1, 2 and 4
and at `nr` level 1, where the frozen mp=8 pipeline on the identical entry
column is 7.61e-05, 4.00e-05, 6.27e-05 and 4.63e-05 away; over levels 0-7
mp=28 is at least as near WRF as mp=8 at every level and strictly nearer at
seven of eight. (An earlier revision of this page said the residual *was*
carried by the shared warm-rain path; that claim was true of the tree it
was measured on and is now false, so it is replaced rather than edited.)
The level-6 allowance is a *metric* change, not a looser tolerance: that
level enters with `qr` = 3.1695777e-07 kg/kg and evaporates 99.958% of it
in one 10 s step, so the survivor is the difference of two nearly equal
float32 numbers and a relative gate there measures the rate's error
amplified by 1/(1 − 0.99958) = 2370. mp=28 produces 1.3384e-10 there
against WRF's 1.3426e-10. With that level left *in* on a relative metric
the fixture reads `qr` 3.155e-03 / `nr` 3.155e-03, which is the number the
unexceptioned table reports and the reason the allowance is named.

| fixture | quantities that miss, with the measured maximum relative difference |
|---|---|
| `aero-cold-overlap` | `qc` 1.000e+00, `nc` 1.000e+00, `effc` 8.102e-01 — **one mechanism, three views, and it is a one-ULP disagreement wearing a full-scale number.** At 0-based level 4 WRF ends the step with 1.4551915228366852e-11 kg/kg of cloud water (exactly 2⁻³⁶, exactly 1.000 float32 ULP of the 2.3252160e-04 kg/kg the level entered with) and `nc` = 1.8333361 per kg, while ArWen ends at exactly zero; `effc` then reports 8.102e-01 because with no cloud water ArWen takes the 2.49 µm floor while WRF's remainder gives 1.31176e-05 m. Recorded as a MISS rather than allowanced. Separately and genuinely: `nr` 1.261e-04, `qr` 4.443e-05 at level 6, where the rain number falls 255.407 → 0.0739 per kg (99.97% consumed) and the difference is 0.611 ULP of the entry value (`qr`: 1.789 ULP) |
| `aero-cloud-freeze-nc` | `qc` 4.926e-06 — the fixture's only surviving row, at level 4, where 98.2% of the entry cloud water is frozen away and the survivor differs by exactly 1.000 ULP of the 8.247212e-05 kg/kg entry value. It is the SECOND float32 rounding of `qc` inside one step: WRF rounds once at `module_mp_thompson.F:3975` from a `qcten` carrying the source network and the condensation together, while ArWen applies the source network to `qc` and then applies the condensation to the already-rounded value |
| `wp08-nusweep` | `qr` 4.642e-06 — 2.3x the gate, at level 12, created from exactly zero and reaching 2.242e-11 kg/kg; the absolute difference is 1.04e-16 kg/kg. **This cell is ill-conditioned, measured, and no FP32 implementation can hold it to the gate.** Perturbing the level's entry cloud water by ONE float32 ULP moves the exit `qr` by 128 ULP (up) or 32 ULP (down); perturbing the entry droplet number by one ULP moves it by 256 ULP either way. The measured disagreement is 60 ULP — *smaller* than a single-ULP input change produces — so it is consistent with a sub-ULP difference in an intermediate that FP32 cannot represent. The 2.0e-6 gate at that level is ~26 ULP, i.e. below the cell's own condition number. Level 12 is also the only level of this column where the droplet number **rises** across the step (1.417475e+08 → 1.428941e+08 per kg), the signature of the number-weighted cloud sedimentation feeding it from the level above while autoconversion drains it |
| `wp08-freeze` | `nr` 2.724e-06 — 1.4x the gate, at level 0, created from exactly zero. **Attributed, un-attributed, measured, and then narrowed by the fix it prompted: 29 of its 34 ULP are the rain-presence gate, and what is left of that gate's disagreement is that the fallout kernel cannot see WRF's post-evaporation rewrite of `rr(k)`.** The shipped gate at `thompson.cu:450-452` is WRF's conjunction -- the TAU+1 test on the mixing ratio (`module_mp_thompson.F:3236`) with the R1 floor (`:3252`), then `rr(k) .gt. R1` (`:3616`) -- so it compares a mass concentration, as of `cb765336`. At 0-based level 1 of this column ArWen sees qr = 8.526513e-13 (closed, so it inherits the level-above fall speeds) and WRF sees rr = 1.174815e-12 (open, so it computes a real one 5.3x slower in number). Forcing ArWen's gate open on the shipped kernel moves level 0's `nr` from 34 ULP away from WRF to 5, while a mass change of the same size that does *not* flip the gate leaves the output bit-identical — the control that makes it a measurement (`tests/test_thompson_aerosol_adapter.py::test_the_wp08_freeze_residual_is_the_presence_gates_units_measured`). This page published the attribution as falsified for part of 2026-08-01; that falsification read `qr1d + qrten*DT` at the end of the step and took it for the value WRF's `:3236` tested, but the rain-evaporation block at `:3501` subtracts from `qrten` in between — instrumented WRF records `L_qr = .true.` there, so `:3236` took its true branch and `rr` was never floored to R1. `cb765336` DID reconcile the gate's units -- that is what it is -- and it did not move this residual because this level is in the one class that commit enumerated and left standing: `qr <= R1 < qr*rho` with `L_qr` true, 3 627 level-visits of the 2 h 12 km forecast it measured over. WRF's `L_qr` opens the evaporation block at `:3501`, whose rewrite at `:3568` leaves `rr(k)` = 1.174815e-12 above R1; ArWen's fallout kernel sees only qr = 8.526513e-13, fails the TAU+1 test, and floors rr to R1. NOT FIXED, and the reason is now a named change rather than a frozen file: closing it means carrying `L_qr` itself from the evaporation kernel to the fallout kernel, which nothing in the tree does. (An earlier revision of this row said the gate still compared a mixing ratio and that the kernel file was byte-frozen; both were true when the attribution was written and neither survived `cb765336`, so they are replaced rather than edited.) The residual <=5 ULP left over is not separately attributed |
| `aero-reduces-to-classic` | **clears the gate under the one allowance above** and is listed here only because the flat gate is the yardstick this table uses: level 5 is inside the flat gate now (`nr` 4.146e-07), and what remains is `qr`/`nr` 1.238e-04 at level 6 if that level is measured relatively rather than in ULPs |

**No surface accumulation misses any more, on any fixture.** All seven are
compared separately and all seven are now inside the flat gate on all 22
columns: `RAINNC`, `RAINNCV` and `SR` are **bitwise identical to WRF on
every fixture in the deck** (0.000e+00 relative), and `SNOWNC`/`SNOWNCV`
and `GRAUPELNC`/`GRAUPELNCV` peak at 6.285e-08 and 7.062e-08. Earlier
revisions of this page published `rainnc` / `rainncv` / `sr` residuals on
two fixtures and explained them as *one number seen three ways* —
`RAINNCV` is the fallout sum (`module_mp_thompson.F:1298`), `RAINNC`
accumulates it from zero on a first call (`:1299`), and `SR` divides the
exactly-agreeing frozen part by it (`:1308`). The explanation was correct;
the residual is gone, because the sedimentation density it came from was
found and fixed rather than tolerated.

Every surviving residual now sits where the field is either **created from
zero** inside the step or driven to **near-total consumption** — the
regimes where a fixed relative gate measures the amplified rounding of a
difference rather than the rate that produced it. That is stated, not
used: no bound above is relaxed for it. The whole 22 × 23 table is also
published denominated in float32 ULPs of `max(|entry|, |WRF after|)`, and
the two units disagree about which cell is worst, which is the point:
`aero-cold-overlap`'s two full-scale relative rows are **1.000 and 0.229
ULP**, while its `effc` row is **1.1e+07 ULP** — the signature of a branch
that flipped, not of a rounding. The worst ULP figure anywhere *inside* the
relative gate is 20. Four of the 22 fixtures are bit-exact on every one of
the 23 compared quantities.

**What changed in THIS revision, and why the count moved from 15 of 22 to
17 of 22.** Two fixtures were closed outright and one number grew. WRF
forms the working rain mass and number that sedimentation consumes
**twice**: at `module_mp_thompson.F:3237-3238` from the `:3193` τ+1
density, for every level carrying rain, and again at `:3568`/`:3570` from
the `:3490` post-condensation density — but only inside the `:3501-3502`
gate. ArWen was writing the post-condensation density unconditionally, so
every level got the `:3568` answer including the levels WRF never rewrote.
Restoring the level-wise choice closed `aero-drop-evap` (`rainnc` /
`rainncv` 5.165e-04 → 0.000e+00, `qr` 3.533e-05 → 7.35e-08, `nr`
2.258e-05 → 3.92e-07) and `aero-ice-demott-idxin` (`sr` and `rainnc` /
`rainncv` 1.279e-04 → 0.000e+00, `qr` 2.894e-05 → 6.42e-07), took
`aero-cloud-freeze-nc` from six rows to one (`qr` 2.800e-05 → 8.97e-08,
`nr` 1.797e-05 → 2.59e-07, `rainnc` / `rainncv` / `sr` 1.162e-05 →
0.000e+00), took `aero-reduces-to-classic`'s reflectivity 5.283e-04 →
3.242e-05 dB — which is what **retired** the third allowance — and took
that fixture's `qr` 7.813e-05 → 1.788e-07 and `nr` 4.832e-05 → 5.700e-06,
which is what let its bound go 1.0e-4 → 1.0e-5 with the `qr` entry
deleted. A second change pinned the contraction of WRF's terminal
`q1d(k) = q1d(k) + qten(k)*DT` apply (`:3973-4023`), which the `gfortran
-O2` oracle cannot fuse and nvrtc did; that made `aero-nc-cap`'s `qc`/`nc`
and `aero-ice-demott-idxin`'s surface accumulators bitwise exact. **It also
cost one cell, published as grown rather than absorbed:**
`aero-cold-overlap`'s `qr` at level 6 rose 3.667e-05 → 4.443e-05 (1.477 →
1.789 ULP of the entry value). Reverting that single pin restores the old
number and simultaneously un-does four of the improvements above, two of
them exact, so the pin is kept: it makes the instruction sequence WRF's at
every cell rather than at the cells that flatter the table.

**What changed in the revision before that — and the biggest change was a
correction to ArWen's own measuring stick, not to ArWen's physics.**
`aero-ice-koop` was published by the registry, this
page, `PROVENANCE.md` and the evidence document, for four waves, as *the
largest genuine physics gap in the port*, at `qi` 1.612e-03 / `ni`
1.764e-03 / `effi` 5.093e-05. It now measures 1.534e-07 / 3.396e-07 /
1.886e-07 and clears the flat gate. **No kernel changed. The oracle
harness did.**
`tools/thompson_wrf461_oracle/run_column_aero.F90` built the Exner
function with `rd_over_cp = 287.0/1004.0`, while WRF's own `rcp` is
`r_d/cp` with `r_d = 287.` and `cp = 7.*r_d/2.` = 1004.5
(`share/module_model_constants.F:19,:20,:31`) — i.e. exactly `2/7`, and
4774 float32 ULP away from what the harness used. A fixture records
pressure and temperature; the adapter has to solve for the float32
`theta` that reproduces that temperature bitwise under ArWen's own
`RCP`. Across two Exner constants 4774 ULP apart, that solve does not
survive: **47 of the deck's 528 entry levels admit an exact `theta`
under one constant and none under the other** (first published as 40,
the original environment's libm; re-pinned 2026-08-03 — the correction
note in `mp28-column-evidence.md` §3.4 records both counts and why the
number is the host libm's, not the deck's). At those levels the
adapter perturbed the entry pressure by up to 15 ULP to recover the
recorded temperature, and a perturbed entry pressure drives genuinely
different microphysics — on seven fixtures, `aero-ice-koop` among them.
The port was being measured against a yardstick that was not WRF's, and
correcting the yardstick — regenerating the deck with the kernels
untouched — is the whole of the closure. Homogeneous haze freezing was
never measured to be wrong.
`tests/test_physics_md_aerosol_claims.py::test_the_committed_deck_carries_wrfs_own_exner_constant`
re-derives exactly that from the committed CSVs, which is the half a
reader can check without rebuilding anything: all 528 levels of the
shipped deck invert exactly under WRF's `r_d/cp`, and 47 of them do not
under `287.0/1004.0`. The other half — that regenerating under the
correct constant is what moved the numbers, with nothing else changed —
is the differential rebuild recorded in
`tools/thompson_wrf461_oracle/README-AEROSOL.md`.
Against that, `aero-cold-overlap` acquired the level-4 rows above and is
worse than it was, and the two `wp08-*` residuals had never been published
at all because the registry counted 19 fixtures while the gate drove 22.

The gate is `tests/test_thompson_aerosol_adapter.py::test_g3_end_to_end_against_all_nineteen_oracle_fixtures`
(the name still says "nineteen"; it globs the fixture directory and drives
all twenty-two) and it is **honestly red**. The same numbers are published on the
registry option (`extensions.column_oracle_evidence`), and
`tests/test_physics_registry.py::test_mp28_published_residuals_still_equal_a_live_adapter_measurement`
re-measures them against a live adapter run, so nothing has to trust
this page. Beneath it, the per-kernel column gates and the
device-helper probes against a Fortran probe harness pass, including
bitwise agreement on the effective-radius branches.

**No REAL-DATA or NESTED forecast has ever been validated against WRF, and
none can be yet.** WRF's own `real.exe` is a fatal error on
`wif_input_opt = 0` with `mp_physics = 28`, and ArWen has no aerosol lateral
boundary condition — the depletion front advances at 0.993 of the wind speed,
so a 100 km nest sits at WRF's aerosol floor within 83 minutes. That, not an
absence of running, is what holds the label at `implemented-unverified`.

**A matched IDEALIZED trajectory does now exist**, and it publishes its own
failed gate:
[validation/mp28-matched-trajectory.md](validation/mp28-matched-trajectory.md).
A doubly periodic single-domain warm-bubble forecast, 120 × 120 × 40 at
dx = 2 km for 7200 s, with ArWen initialised *from WRF's own `wrfinput_d01`*
(t = 0 field difference exactly zero in ten fields of thirteen, and one
float32 rounding — 2.5e-09 to 2e-08 — in the three ArWen derives rather than
copies) and run against unmodified WRF v4.6.1
built twice from identical source. Of the four conditions declared before the
runs, three pass and **V3 fails** — but the same condition also fails when WRF
is compared against its own recompilation with one optimization flag changed,
so past t ≈ 2400 s the case is chaotic and V3 was mis-specified. The
measurements that do discriminate are favourable: mp=28's per-step
disagreement with WRF is the disagreement mp=8 already has (1.797e-02 versus
1.800e-02 RMS in `w` after five steps from a mature state), and the domain
aerosol budget matches WRF's to 1.530e-04 over two hours.

mp=28 has also been integrated multi-step against itself:
`tests/test_mp28_forecast_smoke.py` runs 150 steps × 12 s on a specified-BC
convective domain, and a second gate carries the same domain to 600 steps
(7200 s, 2.6 domain ventilation times) with 0 non-finite values, 0 bound
violations and 0 specified-zone ring violations across all 600 microphysics
calls. Finite and bounded is not correct: a scheme with a systematically
wrong activation rate passes every one of those checks for two hours. The
bounds are WRF's, but they are *clamps*, not answers. Full evidence grading, including which claims
rest on ArWen-written Fortran drivers rather than on WRF's own answer, is
in [validation/mp28-column-evidence.md](validation/mp28-column-evidence.md).

#### What the aerosol initial condition is worth

WRF's `thompson_init` fills a *synthetic* CCN/IN profile whenever the
aerosol arrays arrive unset: CCN at `module_mp_thompson.F:493-515`, ice
nuclei at `:531-551` (two independent `MAXVAL` presence tests, `:493` and
`:531`, so a domain can get one fill and not the other), and the surface
emission `nwfa2d` derived from the filled surface value at `:510`. It
decays with height towards `naCCN1` = 50.0e6 aloft from a lowest-level
value of `naCCN1 + naCCN0*exp(-(dz1/1000)*niCCN3)` (`:508`, constants at
`:96-97`) — not the bare `naCCN1 + naCCN0` = 350e6 ceiling, because the
exponent carries the first layer's thickness. On the `aero-init-profile`
fixture (lowest level 250 m, 500 m layers) WRF itself fills 1.478987e+08
at the bottom and 5.000000e+07 at the top, and 1.254902e+06 /
5.000002e+05 for the ice nuclei.

ArWen ports that fill exactly, as `gpuwm.core.microphysics.microphysics_init`,
gates it against WRF's own post-`thompson_init` snapshot
(`tests/test_thompson_aerosol_state_gpu.py::test_init_profile_matches_wrf_fixture_101`),
and **calls it**, once per domain, from
`gpuwm/core/physics.py::initialize_physics` — ArWen's
`module_physics_init.F`, the seam where WRF's `phy_init` calls `mp_init`
(`:1635`) and `mp_init`'s `CASE (THOMPSONAERO)` arm calls `thompson_init`
(`:4522-4538`). The call is presence-gated exactly as WRF's is, so a nest
that inherited its parent's aerosol and a restart about to be overwritten
by a checkpoint are both left alone. A freshly initialised mp=28 domain
therefore begins strictly **above** the terminal apply's clamps
(`MAX(11.1E6, ...)` for `nwfa`, `MAX(naIN1*0.01, ...)` = 5.0e3 for `nifa`,
`module_mp_thompson.F:3979-3982`), not pinned at them — 13.3x and 251x
above them respectively on the fixture numbers just quoted. Two tests hold
that:
`tests/test_mp28_forecast_smoke.py::test_microphysics_init_has_a_production_call_site`
scans for the call site and asserts there is **exactly one** (once per
domain silently becoming once per step would overwrite an advected,
activated and scavenged aerosol field with the synthetic one), and
`test_a_freshly_initialised_mp28_domain_carries_the_profile_not_zero`
asserts the filled state on a real `DomainState` taken through the real
`initialize_physics`.

**Until 2026-08-01 this was the port's largest measured error**: the fill
was implemented, proven and called by nothing, so every mp=28 run
integrated from `nwfa = nifa = 0` under WRF's floors. The measurement that
established the size of that gap is unchanged; what it means is not. It is
now the measured **sensitivity** of an mp=28 forecast to its aerosol
initial condition — two otherwise identical 150-step forecasts of the
convective case in
[validation/mp28-column-evidence.md](validation/mp28-column-evidence.md)
6.1, one taking the production init path and one with the profile removed:

| quantity | with the profile (what a run does today) | with it removed | change |
|---|---|---|---|
| initial mean `nwfa` | 6.653e+07 kg-1 | 0 | -- |
| final interior `nwfa` | 3.006e+07 kg-1 | 1.262e+07 kg-1 | at the floor |
| peak `nc` over the run | 1.598e+08 kg-1 | 2.844e+07 kg-1 | **5.6x fewer droplets** |
| domain-total `RAINNC` | 1.791 mm | 3.119 mm | **+74.1%** |
| peak `RAINNC` | 0.678 mm | 0.930 mm | +37.1% |

Both forecasts are re-run and this table rebuilt by
`tests/test_physics_md_aerosol_claims.py::test_the_published_aerosol_sensitivity_is_a_live_measurement`,
which compares at the precision printed here; the two runs are
bit-identical across repeats on the measurement machine, so nothing here
is a tolerance.

Read that precisely: removing the CCN loading a run starts from raises
domain-total surface rain by **74.1%** over half an hour, and cuts the peak
droplet count by a factor of 5.6. That is not a rounding difference; it is a
different forecast. It is also the magnitude the lateral-boundary deviation
below converges to, because after `L/U` the whole domain **is** the
right-hand column.

**Deviations from WRF you must know about before using it.** Every
bullet in *this* list is also a registry warning, printed whenever a
plan selects the option — `gpuwm/physics_registry_v2.json` is the
authority and this page is its summary. Two further things a reader
needs are in a second, separate list below, marked as such because
neither is an open deviation and neither is a registry warning:

- **No aerosol ingest.** There is no WIF metgrid stream, no GOCART
  climatology reader, and no black-carbon (`nbca`) species, so
  `use_aero_icbc`, `use_rap_aero_icbc`, `wif_input_opt`,
  `num_wif_levels` and `qna_update` are published unimplemented and
  refuse rather than being silently dropped. WRF's own fallback for
  exactly that case — `thompson_init`'s synthetic profile — is ported
  *and installed* (see the section immediately above), so the gap is the
  ingest lane, not the initial condition. `qnbca` is refused rather than
  zero-filled, and `taod5502d`/`taod5503d` (radiation-side aerosol
  optical depth) are not produced by `mp_gt_driver` at all.
- **WRF's own initializer refuses the configuration ArWen runs.**
  `dyn_em/module_initialize_real.F:2734-2736` calls
  `wrf_error_fatal('wif_input_opt=0 but mp_physics=28')`, so `real.exe`
  will not build a `wrfinput` for the no-WIF case at all — not even for
  the synthetic-profile variant WRF falls back to internally. The
  *physics* is WRF's line for line; the admission decision is not. A
  consequence worth stating plainly: an ArWen mp=28 run and a
  WIF-initialised WRF mp=28 run are **not** directly comparable, and a
  comparison between them must not be reported as one.
- **No aerosol lateral boundary condition, and this is the deviation
  that grows with run length.** On a specified (external BC) domain
  ArWen couples only `qv` from boundary snapshots and gives every other
  scalar flow-dependent boundaries with zero inflow, so aerosol-free air
  advects in at the upstream face and monotonically depletes
  `nwfa`/`nifa` for as long as the run continues — with no NaN, no
  negative and no health trip, because WRF's own terminal clamps
  (`nwfa >= 11.1e6`, `nifa >= 5.0e3` per m3) hold the floor. WRF's
  Registry gives `qnwfa`/`qnifa` real `bdy` arrays and forces them from
  the boundary file. **Measured** on a deliberately cloud-free run, so
  every kilogram lost is the boundary policy: with a 20.0 m/s inflow the
  depletion front advances at **19.8638 m/s** (0.99319 of the wind), and
  over 1800 s the domain-interior mean `nwfa` falls to 0.4566 of its
  initial value and `nifa` to 0.3363. At 10 m/s the front runs at
  9.808 m/s, so it is a law: *the upstream `U·t` of your domain is at
  WRF's aerosol floor after time `t`, and the whole domain after `L/U`*
  — 13.9 h for a 1000 km domain in a 20 m/s flow, **83 minutes for a
  100 km nest**. The only interior source is the fixed surface emission
  `nwfa2d` at k = 0, measured at 5540.14 kg-1 s-1, which replaces about
  5% of the lowest level's loading over 1800 s and acts on that level
  only. Separately, the `spec_zone` ring itself ends at *exactly zero*
  aerosol on three of its four faces (west/south/north 1.000, east
  0.125) because WRF's clipped microphysics tile means the terminal
  clamp never runs there — a value WRF itself cannot produce, and what a
  nest boundary or a `wrfout` reader sees. This matches ArWen's existing
  hydrometeor policy and is documented, not fixed.
- **MYNN does not mix the aerosol numbers.** ArWen passes
  `flag_qnc`/`flag_qnwfa`/`flag_qnifa` to MYNN as literal `False`, so
  `nc`/`nwfa`/`nifa` are never vertically mixed by the PBL. WRF mixes
  them at `bl_mynn_mixscalars = 1`
  (`phys/module_bl_mynn.F:4735,:4777,:4957`) or through
  `scalar_pblmix` (`phys/module_pbl_driver.F:2251`). At ArWen's pinned
  MYNN identity `bl_mynn_mixscalars = 0`, and WRF's `check_a_mundo`
  raises `scalar_pblmix` to 1 only when `use_aero_icbc` or
  `use_rap_aero_icbc` is set
  (`share/module_check_a_mundo.F:2477-2495`) — both of which ArWen
  refuses — so WRF's own value here is 0 too and today the two models
  agree. What differs is that ArWen's withholding is *structural*
  rather than a namelist value, and mp=28 is the first configuration
  in which those species carry real values and the withholding is
  physically visible. (Snow is a separate contract and is *not*
  withheld — see the second list below.)
- **Mixed mp=8 ↔ mp=28 nesting is refused by name.** An mp=28 domain
  may only sit under an mp=28 parent. No cross-scheme transition rule
  is registered, so the registry refuses the edge with
  `unsupported-component-transition` and
  `gpuwm/core/microphysics_transition.py` refuses it at runtime with a
  message that says why. WRF's own non-aerosol-aware fallbacks
  (`nc = 100e6/rho`, `nwfa = 11.1e6/rho`, `nifa = 5.0e3/rho`) would
  hand a nested child fabricated aerosol instead of its parent's, and
  no gate here would notice.
- **mp=28 and mp=8 are deliberately not bit-identical thermodynamics.**
  mp=28's `RSLF`/`RSIF` saturation Horner chains are
  contraction-pinned; mp=8's stay FMA-contracted. The two saturation
  vapour pressures therefore differ by one ULP — and
  `module_mp_thompson.F:3401` opens the whole condensation/CCN
  activation block on `ssatw > 1.E-15` (`:185`), so one ULP flips a
  branch. mp=28 matches WRF's own gfortran `-O2` arithmetic; mp=8 stays
  byte-frozen at its model-validated trajectory. Neither is a defect,
  and "make them agree" is the wrong fix in both directions.
- **`CCN_ACTIVATE.BIN` is distributed with ArWen, and a different copy
  is refused.** See the asset note below. It ships, so a clean checkout
  validates mp=28; if it is ever missing, every device gate for the
  scheme — including all 22 column fixtures — skips by name rather than
  passing, and the scheme itself fails closed rather than defaulting.
  A byte-different activation table is rejected, not used: it would be
  a silently different activation scheme.

**Two more couplings a reader needs, neither of which is a registry
warning** — one is a defect this port closed, the other is a coupling
that is correct and easy to assume is missing:

- **MYNN *does* mix mp=28's snow, and this was a real defect while it
  did not.** `FLAG_QS` is true for mp=28
  (`gpuwm/core/mynn_pbl_runtime.py::mynn_flag_qs`, whose
  `MYNN_SNOW_MICROPHYSICS` set is derived from the WRF Registry:
  `Registry.EM_COMMON:3036` declares `moist:qv,qc,qr,qi,qs,qg` for
  `thompsonaero`), so MYNN sees the snow field and its condensation
  `rh_hack` reads it. While 28 was missing from that set —
  `module_bl_mynn.F:734` and `:876` substitute `sqs = 0` and `:1324`
  skips `rqsblten` — the committed `snow_anvil` oracle column had
  `qi_bl` driven from 5.4863e-07 to exactly 0, taking `qc_bl`,
  `cldfra_bl`, `rqvblten`, `rthblten` and `exch_h` with it. Closed, and
  pinned in both directions by
  `tests/test_physics_registry.py::test_the_registry_flag_qs_contract_is_the_one_the_shipped_runtime_applies`.
- **mp=28 feeds the radiation, and it feeds it exactly the way mp=8
  does.** A prognostic `nc` only reaches a forecast's energy budget
  because `effc` reaches the cloud optics. WRF declares mp=28 with the
  same `state:re_cloud,re_ice,re_snow` inventory as mp=8
  (`Registry.EM_COMMON:3036` vs `:3024`) and lists `THOMPSON` and
  `THOMPSONAERO` in one `has_reqc`/`has_reqi`/`has_reqs` disjunction
  (`phys/module_physics_init.F:1005-1006`) — and those same three flags
  are the only gate on the block that *computes* the radii
  (`module_mp_thompson.F:1466`), so in WRF a scheme computes them if and
  only if radiation consumes them. ArWen's RTE+RRTMGP and legacy-RRTMG
  paths therefore both take Thompson's explicit-radius coupling for 28.
  A scheme that fell through to the Kessler default would not merely
  lose its radii: its ice and snow would stop producing cloud fraction,
  and an overcast ice cloud would radiate as clear sky. The one standing
  divergence here is ArWen-wide and not mp=28's — the adapter merges
  snow into a single mass-weighted ice species rather than carrying
  WRF's separate Fu snow species.

**How to select it.** mp=28 is registered with
`reachability.state = "component-override"`: among the NAMED routes it
is offered only as an explicit per-domain microphysics override on the
experiment-per-domain tree route. No template selects it, it is no
template's default, and `gpuwm domain` is unchanged. That is
deliberate — an unverified scheme should be opt-in per domain, not
something a suite hands you. As with every row on this page, that is a
statement about the menus and not about the loader: a config that
writes `mp_physics = 28` itself is accepted and runs it (measured, 14
of the 15 combinations the walk tried). Opting in is the point; you
just have to do it explicitly, by either door.

Notes with teeth:

- Thompson's four table assets (`qr_acr_qg_V4.dat`, `qr_acr_qsV2.dat`,
  `freezeH2O.dat`, `thompson_aux_tables.dat`) are byte-validated
  against pinned SHA-256 values at launch.  The two smallest ship in
  the wheel; `freezeH2O.dat` (243 MiB) and `qr_acr_qg_V4.dat` (71 MiB,
  wheel-only exclusion -- a checkout carries it) are staged by
  `gpuwm fetch-tables` (the install scripts run it automatically)
  under the same pins.  `GPUWM_THOMPSON_TABLE_ROOT` can relocate the
  root but never bypasses the pins.
- **mp=28 needs a fifth table, and ArWen ships it.**
  `CCN_ACTIVATE.BIN` (35,288 bytes, sha256 `f2b8d391...`) is the CCN
  activation lookup.  It is the one Thompson artifact that is *not* a
  `thompson_init` product: `table_ccnAct`
  (`module_mp_thompson.F:5110-5166`) only reads it, and the numbers are
  offline parcel-model output (WRF's own comment at `:5102-5108`), so
  no recompilation of WRF and no ArWen code path regenerates it.  It is
  third-party data WRF redistributes; ArWen redistributes the identical
  bytes — WRF v4.6.1's own `run/CCN_ACTIVATE.BIN` under WRF's
  public-domain dedication, whose notice ships in
  `gpuwm/data/wrf_radiation/LICENSE-WRF.txt` — so a default install already
  has it in the table
  root.  `GPUWM_THOMPSON_CCN_ACTIVATE` (full path) or
  `GPUWM_THOMPSON_TABLE_ROOT` still bind a run to a copy in your own WRF
  tree instead.  Absence is fatal and never defaulted, and the size and
  SHA-256 are checked on every load, wherever it was resolved from.  It
  is deliberately outside the classic table set, so no mp=8 launch
  acquires a dependency on it.
- Thompson is the default template scheme: `gpuwm domain` emits
  `mp_physics = 8`, and the registry's declared default template
  (`thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1`) carries the same
  suite. Its warning states the one caveat verbatim: the matched-run
  decay tables were produced with the exact-port legacy RRTMG engine
  (`ra_rrtmg_variant = "rrtmg_legacy"`) to mirror the CPU reference bit
  for bit, while the default template selects the ratified RTE+RRTMGP
  substitution -- the same registry option 4/4 either way.
- Morrison's registry entry records why it is not bitwise (CUDA vs
  glibc transcendentals, FTZ at subnormal branches, FMA contraction)
  and what closing it would take. It was the default template scheme
  until 2026-07-29 and stays fully selectable at its maturity label,
  which states the verification gap honestly.
- Reflectivity (`REFL_10CM`) is computed on output-due microphysics
  steps with WRF's own 50-bin quadrature; the cold-start frame carries
  no reflectivity (no microphysics call precedes it), matching the
  registered deviation D2 in [PROVENANCE.md](../../PROVENANCE.md).

## Planetary boundary layer (`bl_pbl_physics`)

| option | WRF id | maturity | evidence, in one line |
|---|---|---|---|
| YSU | 1 | implemented-unverified | 24-column oracle vs unmodified `bl_ysu.F90`: theta tendency 1 ULP, exchange coefficients 7 ULP, PBLH 1 ULP; momentum/moisture tendencies 4.2e-8 m/s2 / 3.1e-11 kg/kg/s (near-total cancellations); part of the model-validated reference suite alongside Thompson |
| MYJ (Mellor-Yamada-Janjic 2.5) | 2 | implemented-unverified | float32 CPU authority transcribed line by line from the byte-frozen `module_bl_myjpbl.F`, with the CUDA translation unit agreeing with it on land and water columns inside a stated tolerance; column smokes assert finiteness, the `EPSQ2` TKE floor, non-negative mixing length and exchange coefficients, and vapour conservation in a surface-sealed column, each with a mutation control that stubs the ported routine itself. TKE cold-starts at WRF's `epsq2` = 0.2 (`MYJPBLINIT`), not zero -- the seed decides the first-step PBL depth. **Declared divergence:** interface heights are carried above ground rather than above sea level (WRF seeds `ZINT(KTE+1)=HT`), which cancels exactly in real arithmetic and to within 69 ULP in float32 over 4.4 km terrain, with `KPBL` unchanged; gpuwm's column is the better-conditioned one. **No oracle comparison against the WRF Fortran has been run** -- there is no gfortran replay, no fixture of WRF words and no ULP table -- so nothing here claims bit agreement with WRF; that campaign is the declared next stage. Selectable only as the 2/2 pair with the Eta similarity surface layer |
| MYNN (EDMF) | 5 | implemented-unverified | assembled driver bitwise on the warm step vs unmodified `module_bl_mynn.F`; 300-step coupled forecast gate; composes with every radiation pairing the loader admits and with the MYNN (5), classic MM5 (91) or revised MM5 (1) surface layer -- see the MYNN scope note below |
| Shin-Hong (scale-aware) | 11 | implemented-unverified | float32 CPU authority reproduces every output field of both `ctopo` arms at **max ULP 0** against the byte-frozen `module_bl_shinhong.F`, over 30 cases x 6 grid spacings x 40 levels; the CUDA mirror's heat tendency is bitwise (0 ULP through both tridiagonal solves), PBLH/WSTAR/DELTA 1 ULP, `EXCH_H` 8; and its resolved/subgrid partition was scored across a 3200-100 m ladder against pre-registered Honnert (2011) envelope bands -- every gated rung inside, 100 m LES anchor held ([receipts](receipts/grayzone/)) |
| SASE | none: ArWen-only, `bl_pbl_physics = 900` outside WRF's namespace | implemented-unverified, **permanently** | no WRF v4.6.1 counterpart, so no oracle comparison against WRF Fortran exists or can exist and this ladder cannot rank it; numerics self-checked; physics unvalidated -- 2 of 7 frozen acceptance bars met on a single reference case on a single day ([Selecting an experimental scheme](#selecting-an-experimental-scheme)) |

The YSU registry entry records four open items verbatim, including two
FTZ-class subnormal branch disagreements and the `topo_wind=0` driver
arm difference (up to 182 ULP on 6 of 960 fixture lanes). MYNN's entry
records four of its own, the sharpest being that its CUDA leaves are
not bitwise twins of the CPU references away from their oracle
fixtures, and that `phim`/`phih` are still evaluated on the host one
column at a time (a measured 125 µs per column).

Shin-Hong's entry records four of its own: the `q2xk(kpbl+1)`
out-of-bounds read WRF performs and ArWen deliberately does not; WRF's
own `prfac2 = 0/0` NaN reproduced rather than repaired; the sm_120
subnormal-flush branch, closed at the branch by a double-compare
countermeasure with 72 residual flush lanes pinned as counts; and the
CUDA mirror's near-total-cancellation lanes on the momentum and
moisture tendencies (the worst `du` lane is 3.4966e-06 against
3.4756e-06 m/s2). It is the only option in this table whose behaviour
*across grid spacings* has been measured against a published partition
— see [Receipts: gray zone](receipts/grayzone/) — and that measurement
is deliberately fenced: one idealized dry convective boundary layer,
six seeds, one card, one day. It is agreement with a published
similarity curve, not skill against observations, and it moved no
maturity rung. Scheme 11 remains `implemented-unverified` for the same
reason YSU and MYNN do — no matched ArWen-versus-WRF forecast
trajectory has been run with it.

Earlier revisions of this page said MYNN's registry entry recorded a
snow deviation — that snow mixing ratio was not passed to the
condensation RH adjustment (WRF's `rh_hack`). **That is no longer
true, and it was not true of the registry when it was written.**
`FLAG_QS` is derived from the WRF Registry package inventory and is
true for every snow-carrying microphysics ArWen ships (6, 8, 10, 18,
28), so MYNN receives the snow field; the mp=28 bullet above records
the defect that existed while 28 was missing from that set, and its
closure.

### MYNN scope note: what composes, and what is pinned

MYNN is the scheme this page has been misread about, so its scope is
stated here exhaustively, measured rather than described.

**What composes freely.** `bl_pbl_physics = 5` may be written beside
any radiation pairing the loader admits (`0/0`, `0/1`, `4/4`, `90/90`),
any microphysics scheme, any cumulus scheme, any of the four land
surfaces, and the MYNN (5), classic MM5 (91) or revised MM5 (1) surface
layer -- all accepted, all preserved switch for switch on the resolved
per-domain `RunConfig`. There is no engine lock and there never was.
Three of those compositions now have NAMES as well, so MYNN can be
chosen with radiation from a menu:
`wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1`,
`wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1` and
`wsm6-mynn-mynn-noahmp-rte-rrtmgp-expert-only-v1`.

**What is pinned, and it is a real scope limit.** MYNN's *namelist
option identity* is a single validated combination. Twelve knobs --
`bl_mynn_closure` 2.6, `bl_mynn_cloudpdf` 2, `bl_mynn_mixlength` 1,
`bl_mynn_edmf` 1, `bl_mynn_edmf_mom` 1, `bl_mynn_edmf_tke` 0,
`bl_mynn_mixscalars` 0, `bl_mynn_cloudmix` 1, `bl_mynn_mixqt` 0,
`bl_mynn_output` 0, `bl_mynn_tkeadvect` false, `icloud_bl` 1
(`gpuwm/config.py`, `MYNN_PBL_OPTION_IDENTITY`) -- have exactly one
implemented value each, and any other value is refused before the run
starts rather than three hours into a forecast. This is what you will
see, verbatim:

```
bl_mynn_mixlength=2 is outside the admitted MYNN option identity; gpuwm implements bl_mynn_mixlength=1 only, and no nearby branch is substituted for an unported one.
```

So "MYNN now has radiation" does not mean "MYNN now takes options". The
two facts are independent, and both are pinned by tests: radiation-bearing
MYNN configs load, and a moved `bl_mynn_*` knob still refuses with
radiation on.

**One genuine pairing rule, and it points the other way from the usual
misreading.** The MYNN *surface layer* requires the PBL slot to be MYNN
or off -- WRF v4.6.1's own restriction, not ArWen's -- so
`sf_sfclay_physics = 5` with YSU is refused, naming the WRF source
lines:

```
requested WRF physics suite is not executable in gpuwm yet; no substitutions were applied:
  - WRF v4.6.1 PBL/surface-layer compatibility (sf_sfclay_physics=5, bl_pbl_physics=1): WRF v4.6.1 refuses this pairing
[[explain]]
why these pairings are refused:
  - WRF v4.6.1 PBL/surface-layer compatibility: phys/module_physics_init.F:3213-3219,3699-3701: MYNN surface sets isfc=5, and YSU fatals unless isfc=1
```

The MYNN *PBL* carries no such restriction: 5/91, 5/1 and 5/5 are all
accepted -- **60, 60 and 87 distinct accepted combinations**
respectively, counted from the walk receipt's `accepted_combinations`.
Which field is counted matters, so it is said rather than left to be
inferred: `mynn_slice.accepted` in the same receipt reads **88** for
the 5/5 pairing because that field counts ATTEMPTS, and the walk
re-tries its 5/5 anchor suite in a second tier -- 88 attempts over 87
distinct configurations. Distinct configurations are what a reader
asking "what may I compose?" wants, so that is what the three numbers
above are. Earlier revisions of this page described the 5/5 pairing as
mandatory in both directions. It is not, it was measured not to be, and
`tests/test_physics_md_reachability_claims.py` now pins the asymmetry
from both sides -- the refusal above and the three accepted pairings --
so it cannot invert again.

**Maturity is unchanged by any of this.** MYNN stays
`implemented-unverified`, for the same reason YSU, Shin-Hong, Noah,
Noah-MP, RUC, Morrison, Thompson aerosol-aware and Grell-Freitas do: no
matched ArWen-versus-WRF forecast trajectory has been run with it.
Naming a composition is not evidence, and none was claimed for it.

## Surface layer (`sf_sfclay_physics`)

| option | WRF id | maturity (registry) | reachability (registry) | notes |
|---|---|---|---|---|
| MM5 (classic) | 91 | supported | template | the certified-slice surface layer; pairs with YSU and all three LSMs |
| Eta similarity (MYJ) | 2 | implemented-unverified | component-override | Janjic's viscous sublayer over water and the Zilitinkevich thermal roughness over land, transcribed from the byte-frozen `module_sf_myjsfc.F` including its `MYJSFCINIT` similarity tables; publishes `AKHS`/`AKMS`/`THZ0`/`QZ0`/`UZ0`/`VZ0` and NO `MOL`/`ZOL`/`PSIM`/`PSIH`, which is why it is admitted only as the 2/2 pair with the MYJ PBL. `isftcflx`/`iz0tlnd` are refused: WRF passes them in and never reads them (CZIL is hard-coded to 0.1). No oracle comparison against the WRF Fortran has been run |
| MYNN | 5 | implemented-unverified | template | column solver oracle-matched over land and water (max rel. err 4.3e-7); `isftcflx` 0-3 ported; needs the PBL slot to be MYNN or off, which is WRF v4.6.1's own restriction ([MYNN scope note](#mynn-scope-note-what-composes-and-what-is-pinned)) |
| MM5 (revised) | 1 | supported | component-override | no base template selects it (every verified run used the classic scheme); the prepared-domain-tree route offers it as a surface-layer component override, and a config that writes `sf_sfclay_physics = 1` directly is accepted by the loader and runs it -- measured, 360 accepted combinations in [receipts/physics-composition-walk.json](receipts/physics-composition-walk.json) |

All four run. In plain words: `template` means **a preset exists**, and
`component-override` means **a preset exists** on the routes that
declare it and you **type it in your config** anywhere else -- which is
how the revised MM5 row's 360 accepted combinations were measured.
Maturity and reachability are separate registry axes, quoted verbatim
from the registry: `maturity` is the option's evidence tier and
`reachability.state` is how a NAMED route can offer it. Neither column
constrains what you may write in a config -- see [Reachability is a
menu axis, not a loader
gate](#reachability-is-a-menu-axis-not-a-loader-gate) for the
definition and for the measured difference between the two.

## Land surface (`sf_surface_physics`)

| option | WRF id | maturity | evidence, in one line |
|---|---|---|---|
| Noah (4-layer) | 2 | implemented-unverified | 42-column oracle vs unmodified `module_sf_noahdrv.F`: 7 of 31 outputs bit-identical incl. the whole TSLB profile; TSK within 2 ULP, HFX worst at 375 ULP; part of the model-validated reference suite |
| RUC (9-level) | 3 | implemented-unverified | column family oracle-matched vs unmodified `module_sf_ruclsm.F`; full device residency measured at production width (0.47 s per call at 360,000 columns, snow-free) |
| Noah-MP | 4 | implemented-unverified | `NOAHMP_SFLX` bitwise on all four whole-column fixtures; device slab path max ULP 0 vs the scalar authority at 360,000 columns; expert-route option pinned to the exact WRF Registry default option identity |

Divergences the registry states plainly (read the registry warnings
before relying on any of these over unusual surfaces):

- **Noah:** the glacial land-ice column (`SFLX_GLACIAL`) is not ported
  -- a domain with land-ice points must not use Noah; measured cost of
  ignoring this is listed per field in the registry. Frozen-ground
  infiltration on simultaneously-frozen-and-melting columns differs
  through CUDA vs glibc `expf`/`powf` (water redistributed, not lost).
- **Noah-MP:** glacier columns are refused during post-static
  initialization (not silently skipped); sea ice takes WRF's own skip.
  The WRF six-rate precipitation partition and radiation-cadence COSZEN
  carrier are active.
- **RUC:** uses WRF-ARW's `EM_CORE==1` species partition, lake bypass,
  fractional-sea-ice pre/post blend, and radiation-cadence GSW carrier.
  WRF's own uninitialized-`ilnb` read on thin snow (a real WRF defect:
  the value depends on grid traversal order) is *not* reproduced; ArWen
  passes the defined one-layer answer and documents the divergence.

## Radiation (`ra_lw_physics` / `ra_sw_physics`)

| option | selection | maturity | evidence, in one line |
|---|---|---|---|
| RTE+RRTMGP | 4/4 (default) | supported | the default modern k-distribution path; substitution token recorded in every config it touches |
| legacy RRTMG | 4/4 with `ra_rrtmg_variant = "rrtmg_legacy"` | verification tier* | line transcription of WRF v4.6.1 option 4/4: batched LW/SW engines bit-identical (max ULP 0) to the port oracle over the full fixture decks; McICA generators match every stored WRF Fortran mask; used for all matched-run comparisons |
| Dudhia SW (+ LW off) | 0/1 | supported | certified-slice shortwave |
| WRF RRTM + Dudhia | 1/1 | implemented-unverified | line transcription of WRF v4.6.1 `module_ra_rrtm.F`; column smoke of the shipped seams only, **no oracle comparison against the Fortran** |

\* "verification tier" is this page's own term, not registry
vocabulary. Legacy RRTMG is not a separate registry component option:
it is the same registry option 4/4, selected by the
`ra_rrtmg_variant = "rrtmg_legacy"` token and kept for matched-run
verification against the WRF CPU reference. (The registry's
`rte-rrtmgp-legacy-aggregate` entry is a different thing entirely --
an RTE+RRTMGP route retained for a legacy aggregate selector, maturity
`implemented-unverified` -- not this port.)

The two 4/4 implementations are deliberately firewalled: a restart
written under one refuses to resume under the other, and the RTE+RRTMGP
path is byte-unchanged by the legacy port's existence. The legacy
variant implements exactly one WRF mode combination (`icloud=1`, McICA
maximum-random overlap, CAM ozone climatology, year-formula greenhouse
gases, zero aerosol); everything else fails closed rather than
approximating. Measured cost: legacy RRTMG is about 1.9x RTE+RRTMGP
wall time on the same three-domain stack (34.8 vs 18.7 wall-s per
simulated minute, radt 12/3/1, same card and window). Full dossier:
[rrtmg_legacy_integration.md](../rrtmg_legacy_integration.md).

### WRF RRTM longwave + Dudhia shortwave (1/1)

This is the pair a classic WRF namelist asks for, and until now ArWen had
no answer for it: every longwave-on option was a coupled adapter owning
both streams, so a `ra_lw_physics = 1, ra_sw_physics = 1` namelist had
nowhere to land. It lands now, and it lands as a *composition* rather
than a fifth coupled adapter --
`gpuwm.core.rrtm_lw.RRTMDudhiaRadiation` runs the RRTM longwave adapter
and the existing Dudhia shortwave adapter and merges their results,
mirroring the way `module_radiation_driver.F` dispatches `lwrad_select`
and `swrad_select` independently. The four established pairings are
untouched by it.

RRTM owns the GLW buffer: it computes the surface downward longwave from
its own transfer and writes it, so nothing about the pair depends on
what GLW held on entry.

What is implemented: `O3DATA`, `MM5ATM` with the Cavallo buffer layers
above the model top, `SETCOEF`, the `CMBGB1`--`CMBGB16` reduction of the
packaged `RRTM_DATA` records from 256 g-points to 140, `TAUGB1`--
`TAUGB16`, `GASABS` and `RTRN`, transcribed line by line from the
byte-frozen WRF v4.6.1 `phys/module_ra_rrtm.F`. The supported path is
`ghg_input = 0` (the driver's analytic year-dependent CO2/N2O/CH4),
O3DATA's own ozone climatology, no WRF-Chem, and no CFC/CCl4 cross
sections -- WRF's own MM5ATM sets those amounts to zero. Selecting
`ra_lw_physics = 1` with any shortwave other than Dudhia is refused
rather than resolved to something adjacent.

What is NOT established, in the sentence that matters: **no oracle
comparison against the WRF Fortran has been run.** No ULP measurement,
no matched WRF run, no forecast trajectory. The evidence is a column
smoke of the shipped seams (`tests/test_rrtm_longwave.py`) --
finiteness, physical flux bounds, clear-column cooling, cloud-top
cooling against cloud-base warming, OLR falling under cloud, and RTRN's
flux-divergence identity closing over the whole column -- with three
mutation controls proving those checks fail when the scheme is stubbed
out.

Two of those assertions are anchored to numbers from outside the port,
which is worth saying because the rest are self-consistency and
self-consistency is blind to an error every column shares. RRTM's
band-summed Planck table has to reproduce the Stefan-Boltzmann
blackbody `sigma*T^4/pi` to 5e-5 relative, and a 345 K skin temperature
has to clamp to MM5ATM's 339.99 K ceiling rather than run off the end
of the table. Both were added after a review found the Planck lookup
reading one row -- one kelvin -- too warm, biasing every longwave flux
by 1.3 to 2.5 per cent while all seventeen tests of the day passed. A
wrong-but-smooth absorption coefficient would still get through; that
is what the oracle campaign is for. That is why the label is
`implemented-unverified` and the option is
a per-domain override a user asks for by name, never a default. The
oracle campaign is the declared next stage, as it was for Shin-Hong and
Grell-Freitas.

Cost, in time and in memory: the solver is array-vectorised over column
chunks, not a fused CUDA kernel like the two 4/4 engines, and holds
several `(column_chunk, nlayers, 140)` arrays at once. It is materially
slower per radiation call than either 4/4 adapter. Those chunk arrays
are transients, and `--memory-budget-mib` prices persistent allocations
only, so the budget understates a 1/1 run by roughly the chunk cost:
1.67 GiB peak measured at `column_chunk = 4096` and 53 layers on an
RTX 5090, so about 0.2 GiB at the default 512 and linear in between.
Raising `column_chunk` raises the peak with no warning from the budget.
Choosing this pair on a large domain is a deliberate trade both ways.

## Cumulus (`cu_physics`)

| option | WRF id | maturity | notes |
|---|---|---|---|
| Kain-Fritsch | 1 | supported | outer (>=10 km) domains; packaged lookup table; cudt 5 min in the certified templates |
| Grell-Freitas (scale-aware) | 3 | implemented-unverified | whole GFDRV at the WRF v4.6.1 boundary, CPU and CUDA; no template selects it, so among the named routes it is a per-domain override -- a config writing `cu_physics = 3` is accepted directly; runs on the model step (cudt pinned 0) |
| off | 0 | supported | the convection-permitting nests run with cumulus off |

What is certified for Grell-Freitas, and what is not. The certified
half: the entire driver -- column preparation with its mixed-precision
GFS constants, the deep cloud model, the shallow one, both `neg_check`
calls and the output algebra -- reproduces the byte-frozen WRF v4.6.1
`module_cu_gf_*.F` word for word at the GFDRV boundary over the
committed 216-column oracle (18 soundings x 6 grid spacings x 2
`ishallow` arms) on the 208 columns where GFDRV's own decomposition is
exact, with the 8 remainder bounded to the driver's own
`module_gfs_physcons` mixed precision (max 34 ULP, 3.8e-6 relative, no
branch flips). The CUDA path holds that boundary with the gamma
COMPUTED on the device: its transcribed glibc-2.39 float32
`tgammaf`/`lgammaf`/`expm1f`/`exp2f`/`powf` are bitwise against 130k
live-glibc words, which matters because one ULP of the beta-shape
normalisation moves the deep mass flux by up to 7.3 percent. The
registry entry records three deviations: the shallow `k22` trigger
ships with the section-offset indexing corrected (WRF's MAXLOC
off-by-one lives behind a parity-suite flag; measured on the fixture,
the correction moves 3 rejected cases and zero output words), the
inversion-layer search clamps WRF's out-of-bounds `t_cup(kend+8)` read
(clamp count zero on the fixture, asserted), and the engine seam feeds
the advective/boundary-layer halves of the forcing as zeros with
convective momentum tendencies not yet coupled -- the plumb-list any
label upgrade must clear. The uncertified half is the same sentence
YSU, MYNN and Shin-Hong carry: no matched ArWen-versus-WRF forecast
trajectory has been run with `cu_physics = 3`, and no real-case
verification receipt exists -- which is why the label is
`implemented-unverified` and the scheme is always opt-in, never a
default: a per-domain override on the named routes, or an explicit
`cu_physics = 3` in a config you wrote.

## Map projections (`map_proj`)

Projections are not physics schemes and carry no physics-registry
entry; they are listed here because this is the maturity page and
their maturity differs by projection. The authority for the rows is
the binary64 projection oracle gate
(`tests/test_projection_oracle.py`, pinned per-quantity ULP ceilings
against unmodified WRF v4.6.1 `share/module_llxy.F`) and the
worldwide section of [VERIFICATION.md](VERIFICATION.md).

| option | WPS name | maturity | evidence, in one line |
|---|---|---|---|
| Lambert conformal, northern hemisphere | `lambert` | **model-validated** | the 3 April 1974 four-domain matched-run family runs on it; binary64 `module_llxy` oracle at the pinned ceilings |
| Lambert conformal, southern hemisphere | `lambert` | implemented-unverified | binary64 oracle rows (SH secant + SH tangent cones) at the pinned ceilings; Brisbane GPU smoke integration; no matched WRF run |
| Mercator | `mercator` | implemented-unverified | binary64 oracle rows (tropical, subtropical, antimeridian) at the pinned ceilings; Singapore and Fiji (antimeridian) GPU smoke integrations; no matched WRF run |
| Polar stereographic, either pole | `polar` | implemented-unverified | binary64 oracle rows (NH, SH, pole-anchored) at the pinned ceilings; Fairbanks GPU smoke integration; no matched WRF run |
| Latitude-longitude / rotated grids | -- | not implemented | refused at load, never substituted |

Domains containing or touching a pole and forcing footprints wider
than 180 degrees of longitude are refused for every projection
(genuine pipeline limits; the wizard and the loaders name them).

## The default template suite

`gpuwm domain` emits the reference-configuration physics with the
microphysics slot on the model-validated matched-run scheme: Thompson
(mp8, packaged hash-pinned tables), MM5 surface layer (91), Noah (2),
YSU (1), RTE+RRTMGP (4/4), Kain-Fritsch on the 12 km root only, the
49-level eta ladder, and the certified diffusion/damping/acoustic
settings. Morrison (mp10) remains fully selectable at its maturity
label. The acceptance forecast (6 h in 3.6 min on a 250x200x49 domain;
[FIRST-LIGHT.md](FIRST-LIGHT.md)) ran this template's Morrison variant,
the default at the time of that transcript. The matched-run validation
suite runs Thompson with the legacy RRTMG engine
(`ra_rrtmg_variant = "rrtmg_legacy"`) to mirror the CPU reference
exactly ([VERIFICATION.md](VERIFICATION.md)).

## Selecting an experimental scheme

One option on this page is experimental: SASE, the Scale-Adaptive
Stress-Energetics closure, a unified turbulence/PBL closure selected
with `bl_pbl_physics = 900`. 900 sits outside WRF's namespace on
purpose -- WRF's own `bl_pbl_physics` runs to 99 -- so it can never
collide with a scheme WRF adds later. The registry marks it
`reachability.state = "component-override"`: the prepared-domain-tree
runner admits it as a PBL component override, and the single-config
loader takes it directly. This is the whole of a configuration that
validates:

```toml
[grid]
nx = 128
ny = 128
nz = 64
dx = 1000.0
dy = 1000.0
ztop = 20000.0

[dynamics]
dt = 6.0
km_opt = 0
khdif = 0.0
kvdif = 0.0

[run]
run_seconds = 3600.0
moist = true
bl_pbl_physics = 900
sf_sfclay_physics = 91
bldt = 0.0
```

None of those companions is taste. Each is refused at config load --
refused, not warned about -- and the refusal states its own reason:

| requirement | why the closure has it |
|---|---|
| `km_opt = 0` | SASE computes its own horizontal mixing from the closure's own diffusivities, so a `km_opt` mixing operator would double-count it. 0 is admitted for this scheme and no other |
| `khdif = 0.0`, `kvdif = 0.0` | constant-K diffusion may not silently stack on the SASE mixing |
| `bldt = 0.0` | the closure produces a w tendency rebuilt every step rather than carried across a PBL call interval, so it must run every step |
| `sf_sfclay_physics != 0` | its lower boundary condition is the surface layer's friction velocity, heat and moisture fluxes and gust-corrected wind speed; with the slot off those four fields do not exist |
| `moist = true` | it mixes water vapour, cloud water and cloud ice alongside potential temperature, and forms its stability from the saturated Brunt-Vaisala frequency |
| `nz <= 128` | the implicit vertical solve carries its tridiagonal columns in per-thread local memory at that fixed depth |

On real data the path is the ordinary one, with two edits. Emit an
experiment config for your area as
[FIRST-LIGHT.md](FIRST-LIGHT.md) describes:

```bash
gpuwm domain --point=39,-98 --card 32gb --ladder 12 \
  --source gfs --cycle latest --hours 3 --out configs/myarea.toml
```

then in the `[shared]` table change `bl_pbl_physics` to `900` and
`km_opt` to `0`, and run it with `gpuwm go configs/myarea.toml`. The
emitted file already carries `khdif = 0.0`, `kvdif = 0.0`,
`bldt = 0.0`, `moist = true` and a surface-layer scheme, so those five
requirements are met as written; `nz` is the one to check against the
ceiling above. Everything else -- microphysics, radiation, cumulus,
land surface, the domain, the cycle -- is untouched, so a run against
the unedited file is the control for the one you just made.

**The closure is run-wide, never per-nest.** `bl_pbl_physics` is a
`[shared]` key, so on a domain tree the two edits above select SASE on
every domain of the tree or on none of it; there is no way to run one
nest under SASE and its parent under YSU. A `[[domain]]` table that
carries `bl_pbl_physics` is refused at load as an unknown domain key,
which is the right answer -- a per-domain PBL scheme that parsed and
was then ignored would be a silently-wrong run. The one SASE key that
*is* per-domain is the output-only `sase_flux_diag` in the table
below. (The registry lists `sase` among the domain-tree route's
`allowed_component_options`, which is a statement about what a physics
*plan document* may name; it is not a claim that the runner can vary
the PBL scheme between nests, and it cannot.)

Three optional knobs exist. Each is fail-closed on its non-default
value only -- set one without `bl_pbl_physics = 900` and the run is
refused, because a key naming a seam the run does not have would read
as a setting that took effect.

| knob | default | what it is |
|---|---|---|
| `sase_flux_diag` | `false` | output-only, per domain. True adds four history fields recording the closure's own vertical subgrid fluxes with the venting channel separated from the K_v diffusion channel; the prognostic state is bitwise identical either way |
| `sase_moist_n2` | `true` | physics selector, run-wide. True is the closure as built (saturated N^2 at the stability lengths, the subgrid-energy buoyancy source and the K_v/K_h suppression); false consumes the dry N^2 at all three points. Not per-domain: a nest whose domains ran different closures could not be compared across its own boundary |
| `sase_stable_dissipation` | `false` | physics selector, run-wide. The default is false for a measured reason, not for caution: with it true, the closure's own registered stable-boundary-layer calibration gate exits its observation band, and `tests/test_sase.py::test_jet_decoupling_stable_dissipation_exits_obs_band` pins that RED. What is falsified is the pair of stable-limb coefficients, which enter the stability ratio jointly; neither may be re-registered or tuned alone |

Maturity warns and never blocks. Every front
door says so once, and then continues. `gpuwm go` carries the clause
inline in the banner it prints before it does anything:

```
go: configs/myarea.toml -- source gfs, cycle 2026-08-01T12, 3 h, SASE PBL: EXPERIMENTAL, ArWen-original with no WRF counterpart
```

and both forecast runners print a full sentence to standard error as
the run starts -- `tools/prepared_single_domain_forecast.py`, which is
the stage `gpuwm go` drives:

```
prepared forecast: physics: SASE PBL: experimental, ArWen-original with no WRF counterpart -- the run continues.
```

and `tools/prepared_domain_tree_forecast.py`, which is how a domain
tree is run (`gpuwm go` refuses a multi-domain config and names this
runner instead):

```
prepared tree: physics: SASE PBL: experimental, ArWen-original with no WRF counterpart -- the run continues.
```

The sentence after the prefix is one string with one definition,
`gpuwm.physics_compat.experimental_selection_sentence`, and it is
built from the registry maturity rather than from any scheme name: a
second option registered at this rung is named by these same lines
without an edit to any of them.

Read that sentence literally. The status behind it:

- **It is not a WRF scheme.** There is no WRF v4.6.1 counterpart, so
  no oracle comparison against WRF Fortran exists or can exist for
  it, and it carries no WRF selector number. Every other option in
  this registry is measured as a distance from WRF; this one cannot
  be.
- **Its numerics are self-checked.** 293 tests pass on this head, 217
  CPU (`tests/test_sase.py`, plus one registered xfail) and 76 GPU
  (`tests/test_sase_gpu.py`, measured on an RTX 5090, driver
  580.126.09, CuPy 14.1.1): FP64-authority mirrors of every
  operator, analytic closed forms (box-filter transfer, single-mode
  structure function, pure-shear strain, closed-form energy decay,
  log-layer diffusivity identity, Ekman balance), energy-ledger
  theorems that close to roundoff, device-vs-authority parity at
  declared tolerances, and mutation controls and RED/GREEN
  falsification pairs on every switch. At run time the same posture
  applies: every rate the closure hands the dynamics is checked
  finite before it joins the RK stack, and a rate that is not stops
  the run naming the domain it came from and which of the producer's
  inputs -- friction velocity, surface fluxes, first-layer geometry
  -- were already degenerate when the closure received them.
- **Its physics is unvalidated.** Scored against a single reference
  case on a single day, the closure met 2 of its 7 frozen acceptance
  bars and missed 5 -- including a total absence of the low-cloud
  deck its conditional-venting limb exists to act on: median liquid
  water path 0.0 g/m2 against a reference of 132-235. Nothing
  measured speaks to any other case.
- **It has not completed a forecast on operational data.** Its first
  run on real input -- GFS 2026-08-01 18Z, one 306x244x49 domain at
  12 km, through `gpuwm go` -- reached one forecast hour and then
  failed the full-state health gate with every prognostic class
  violated. Two things were measured in the clean hour before it.
  The prognostic subgrid energy falls to its floor above the boundary
  layer as it should and then **regrows between 11.3 and 13.9 km**, to
  1.62 m2/s2 against its own boundary-layer peak of 2.60, centred on
  the level the gate then failed at; with the `km_opt = 0` this scheme
  requires, no other horizontal operator acts up there. And its
  boundary layer is deeper than the control's throughout: PBLH median
  1967 m vs 1558 m, p90 3962 vs 2793, maximum 8900 vs 4847, 9.5% of
  columns above 4 km vs 1.2%. Treat the co-location of that upper
  maximum with the failure level as a lead and not a proven cause:
  it is one case and one hour, and the same configuration also failed
  the gate under the stock YSU control -- later, at 3 h, and on the
  wind class alone. What separates the two is **not measurable today**:
  `km_opt = 0` is admitted only with this scheme, so the control that
  holds the mixing configuration fixed and varies only the closure
  cannot be expressed, and the closure publishes no diagnostic for the
  horizontal mixing it replaces (its four optional flux fields are all
  vertical).
- **It carries a known open bias.** Equilibrium subgrid TKE sits
  roughly 3-5x below the observed value (sqrt(e) about 2x low), which
  biases the stability lengths, the horizontal diffusivity, the
  buoyancy flux and every TKE product it writes. The bias is
  registered and unfixed: the TKE magnitudes this scheme reports are
  not to be believed.
- **Its horizontal operators wrap periodically.** The test filters,
  strain and horizontal flux stencils wrap in x and y
  unconditionally. On a specified or nested domain the mitigation is
  that subgrid energy is held at its floor across the outer
  `spec_bdy_width` rows every step, and the same width is excluded
  from the closure's solve reductions -- an adjudicated boundary
  treatment, not a halo exchange.

Read the maturity label as "the numerics are self-checked", never as
"the physics is validated". Use SASE to experiment. Do not use it for
a forecast you intend to believe.

## What each data route can prepare

**The physics suite is yours to choose, and the runners execute the one
your config writes.** The GFS/ERA5/20CRv3 prepared single-domain runner
and the multi-domain tree runner execute any suite the engine
implements, exactly as your config writes it; there is no profile
whitelist on those routes. `--physics-profile` is optional there:
naming one asserts your config IS that shipped suite, and the runner
refuses on any switch drift, which is how you keep a run pinned to a
published product. Whether the suite you selected carries
WRF-verification evidence is stated -- one sentence in the wizard
output, the run receipt, and `--explain` -- and never gates: a suite
without evidence prints "supported, not yet WRF-verified" and the run
continues. An expert-template suite (Noah-MP) keeps its registry-owned
acknowledgement on every route: **one line**, `--ack <id>` or
`acknowledgements = ["<id>"]` in the experiment, and it runs. `gpuwm
domain --explain` lists the shipped profiles and what each one actually
runs -- several run full RTE+RRTMGP with Kain-Fritsch, several run
longwave OFF with Dudhia shortwave and no cumulus. Read the names.

One route is keyed differently and it is worth knowing which. HRRR cold
start's evidence contract is keyed by shipped profile, so it prepares
its own default profile when none is named and refuses a name outside
its contract -- an unnamed HRRR config's own suite does not run as
written there. The way through is to name a profile the contract
carries; [Defaults](#defaults) lists what that default runs.

What still refuses, on every route, is a switch value the engine
genuinely does not implement (the refusal names the switch), the
registry's land-surface route blockers (for example GFS+RUC, which
dies at its first surface-temperature call), and -- since 1.7.1 -- an
undeclared asymmetric radiation pairing across a window that includes
local night (next section).

Which NAMED suites each route prepares is the narrower question, and
the answer is route-dependent. State of play in v1.1.1:

| route | what it can prepare |
|---|---|
| ERA5 config door (`[case_data]` -> `gpuwm run`) | the registry-admitted combinations |
| GFS single domain | WSM6, Thompson, Morrison, NSSL2, MYNN; Noah-MP with an expert acknowledgement. RUC is deliberately withdrawn on this route |
| ERA5 single domain | the same normal profiles, plus RUC |
| HRRR single domain | the normal profiles, plus RUC and expert Noah-MP |
| prepared domain trees (GFS, ERA5, HRRR, 20CRv3) | the normal profile family, plus expert Noah-MP; microphysics may be overridden per domain, which is the only way to reach Thompson aerosol-aware (28) |

v1.0.1 restricted the GFS/HRRR door to YSU + MM5 surface layer + Noah,
because the front door unconditionally ran the stock-WRF exporter and
that exporter hard-requires `bl_pbl_physics = 1`, `sf_sfclay_physics =
91`, `sf_surface_physics = 2`. v1.1.0 removed that coupling and made
MYNN, RUC and Noah-MP reachable; v1.1.1 withdrew exactly one
combination, GFS + RUC, because the GFS route supplies none of the
soil/surface fields RUC's initialization needs and the failure landed
mid-forecast rather than at preparation.

**This table is a summary of a machine-readable authority, not the
authority itself.** The shipped registry decides, and it will answer for
your exact configuration:

```
rw-wps --show-physics-registry
```

Read `runner_routes.*.source_template_ids` and `expert_template_ids`.
Where this page and that output disagree, the output is right.

## Nocturnal validity

A suite whose shortwave runs while its longwave is OFF
(`ra_sw_physics > 0`, `ra_lw_physics == 0`) is a **daytime validation
configuration**. Shortwave heats the surface by day; after sunset the
surface radiates to space with no downward longwave to balance it, so
skin temperature craters, the surface saturation humidity collapses
with it, and 2 m dewpoints read far below the airmass. A shipped 48 h
case emitted with `thompson-mp8-ysu-mm5-noah-validation-v1` verified
exactly this failure.

If you are reading this because a run already did that to you, the
user-facing walkthrough -- the symptom, how to tell which version is
actually executing, and how to correct the config -- is
[NOCTURNAL-DEWPOINTS.md](NOCTURNAL-DEWPOINTS.md).

The table below classifies all **16 shipped single-domain profiles**
(`gpuwm.physics_compat.SINGLE_DOMAIN_PHYSICS_PROFILES`, which is the
`--physics-profile` choice list). The registry carries two further
templates that are tree-only and so have no row here --
`thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1`, the declared default
template, and
`20crv3-wsm6-ysu-mm5-noah-kf-rte-rrtmgp-implemented-unverified-v1` --
and both run RTE+RRTMGP on both components, so both are nocturnally
valid. Being absent from this table is never a verdict: a config that
composes its own suite is classified by the same rule, at load, by the
same guard.

| profile | radiation (lw / sw) | nocturnally valid |
|---|---|---|
| `morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1` | RTE+RRTMGP / RTE+RRTMGP | **yes** (the wizard's gfs/era5 default) |
| `nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-validation-candidate-v1` | RTE+RRTMGP / RTE+RRTMGP | **yes** |
| `nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-validation-candidate-v1` | legacy RRTMG / legacy RRTMG | **yes** |
| `thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1` | legacy RRTMG / legacy RRTMG | **yes** (the wizard's hrrr default) |
| `thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1` | legacy RRTMG / legacy RRTMG | **yes** |
| `wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1` | RTE+RRTMGP / RTE+RRTMGP | **yes** |
| `wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1` | RTE+RRTMGP / RTE+RRTMGP | **yes** |
| `wsm6-mynn-mynn-noahmp-rte-rrtmgp-expert-only-v1` | RTE+RRTMGP / RTE+RRTMGP | **yes** |
| `thompson-mp8-ysu-mm5-noah-validation-v1` | OFF / Dudhia | **no** |
| `wsm6-ysu-mm5-noah-no-radiation-v1` | OFF / Dudhia | **no** |
| `kessler-mp1-ysu-mm5-noah-dudhia-v1` | OFF / Dudhia | **no** |
| `wsm6-mynn-mynn-noah-no-radiation-implemented-unverified-v1` | OFF / Dudhia | **no** -- nocturnal sibling: `wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1` |
| `wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1` | OFF / Dudhia | **no** |
| `wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1` | OFF / Dudhia | **no** -- nocturnal sibling: `wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1` |
| `wsm6-ysu-mm5-noahmp-no-radiation-expert-only-v1` | OFF / Dudhia | **no** |
| `wsm6-mynn-mynn-noahmp-no-radiation-expert-only-v1` | OFF / Dudhia | **no** -- nocturnal sibling: `wsm6-mynn-mynn-noahmp-rte-rrtmgp-expert-only-v1` |

Each MYNN sibling pair differs in the radiation block and nothing else
-- exactly five switches move, measured: `ra_lw_physics` 0 -> 4,
`ra_sw_physics` 1 -> 4, `radt` 1.0 -> 12.0, `wrf_rrtmg_compatibility`
`none` -> `wrf-rrtmg-4-4-to-rte-rrtmgp-v2`, and `ra_rrtmg_variant`
gaining `rte-rrtmgp` -- so switching between them isolates
radiation and a paired run measures it. Through 1.8.7 the MYNN family
had no `yes` row at all, which is what made "MYNN" and "nocturnally
invalid" look like the same choice; they were never the same thing, and
now they do not even look alike.

**Running one over a night window is one line.** The pairing stays
fully selectable -- for daytime validation windows as it always was,
and for night windows as a **declared experiment**. Add to
`[experiment]`:

```
acknowledgements = [
  "asymmetric-radiation-nocturnal-window-v1",
  "constant-downward-longwave-v1",
]
```

Two tokens, one edit, because a shortwave-only suite makes two separate
claims: it runs a night with no longwave scheme, and it fabricates the
downward longwave its surface reads. The refusal prints them together
so you never make the second trip.

Every front door enforces this at config load (the guard lives in the
one loader they all share, `gpuwm.experiment.build_experiment`): a real
experiment whose window includes local night at its `[projection]`
reference point refuses to load an *undeclared* asymmetric pairing,
naming the physics, the matched profile, and both remedies. Idealized
experiments (no `[projection]`) have no place or clock and are not
guarded.

**The declaration is yours to make, and only yours.** Through 1.8.7
`gpuwm domain` wrote that line into any config it emitted for an
asymmetric profile over a night window --- so the emitted file was one
no later command could refuse, on the strength of a statement its owner
had never made. Since 1.8.8 the wizard **refuses** that combination and
names the two ways forward: pick a profile with both streams on, or
declare it yourself with `gpuwm domain --ack
asymmetric-radiation-nocturnal-window-v1`, which is what writes the
line. Either way the emitted file also carries a `NOT NOCTURNALLY
VALID` header, and a declared emission says so on stderr as it writes.

## Radiation off with a land surface

A suite with **both** radiation streams off (`ra_lw_physics = 0` and
`ra_sw_physics = 0`) beside any land-surface model runs three ways, and
the loader names all three rather than picking one for you:

1. **Give the run radiation.** Set the pairing you meant --
   `ra_lw_physics = 4` with `ra_sw_physics = 4` is what every
   nocturnally valid shipped profile does. For a config that never
   wrote an `ra_*` line at all this is the likeliest fix, because the
   line is simply absent.
2. **Turn the land surface off too.** `sf_surface_physics = 0` is a
   prescribed skin temperature, which reads no `GLW` at all, so nothing
   is left to be fabricated.
3. **Declare the experiment**, in one edit, in `[experiment]`:

```
acknowledgements = [
  "radiation-off-land-surface-v1",
  "constant-downward-longwave-v1",
]
```

Two tokens, again, because there are two claims: "this run has no
radiation scheme" and "the number the surface integrates is one
somebody typed". A declared run integrates gpuwm's declared constant
**300 W m-2** (`gpuwm.core.physics.DECLARED_CONSTANT_GLW_WM2`) every
step -- a surface-budget experiment, not a forecast, and now on the
record as one. Two shipped 60-second VRAM probes
(`configs/real74_4dom_mynn_norad*.toml`) declare exactly that pair, and
each says in the file why its surface energy budget is not something
anyone reads.

Everything below is why the loader asks, rather than something you have
to do.

> **Compatibility break, 1.8.8: this includes configs that never
> mentioned radiation at all.** `ra_lw_physics` and `ra_sw_physics`
> default to off, so a real case (one with a `[projection]`) that
> selects a land surface and simply *omits* a radiation selector
> resolves to the same both-off suite and is refused the same way. That
> is the larger of the two classes and the one that breaks files which
> loaded before, so the refusal says which one you are in: an omitting
> config is told `NO RADIATION SELECTOR SET AT ALL` and offered the
> missing line first, rather than being quoted two zeros it does not
> contain. If a config of yours stops loading at 1.8.8 and you cannot
> find the `ra_*` lines the older message named, this is why --- the
> physics did not change, the silence did.

Noah, Noah-MP and RUC each read downward longwave every surface step, and
with no radiation scheme attached nothing ever computes it. Through 1.8.7
that number was a seed nobody chose: `initialize_physics` defaulted `glw`
to a plausible clear-sky **300.0 W/m2**, seeded once and read forever,
which is why the 1.7.1 dewpoint collapse looked like weather instead of a
missing scheme. There is no seed any more --- the argument has no default,
so the value a land surface integrates is always one somebody typed, and
the declaration above is where they type it.

WRF's own answer for a longwave-free run is **zero**
(`phys/module_physics_init.F:1168-1170` zeroes `GSW`/`GLW` at cold start,
and `phys/module_radiation_driver.F:1719-1722` re-zeroes them at the top
of every radiation call before the longwave branch that would fill them).
gpuwm deliberately diverges and runs the DECLARED 300 W m-2 instead,
because a number somebody declared is auditable where a zero that looks
like a measurement is not. Neither is a sky; the difference is which one
leaves a record.

An **idealized** case that wants a fixed sky passes one explicitly to
`gpuwm.core.physics.initialize_physics(glw=...)`. That is a declaration
at the call site, it survives the longwave-off passthrough by design, and
it is visible in the source where a default is not.

## Defaults

### Where the downward longwave comes from

The nocturnal guard asks whether a WINDOW is survivable. A second,
separate guard (new after 1.8.7) asks whether the downward longwave
EXISTS.

With `ra_lw_physics = 0` nothing computes GLW. gpuwm's Dudhia adapter
is shortwave-only and returns the GLW array it was handed, unchanged,
on every radiation call, so a land-surface scheme (`sf_surface_physics`
2 / 3 / 4) integrates one frozen number for the entire forecast.
Through 1.8.7 that number was `initialize_physics`' `glw = 300.0`
default, and no production call site ever passed a different one.
Radiative equilibrium at 300 W m-2 is 269.7 K (25.8 F); a Gulf-coast
October night runs near 410 W m-2, or 291.6 K (65.2 F). Even with no
land-surface scheme to integrate it, shortwave-on keeps the radiation
slot active and publishes the fabricated constant as the GLW row of
every wrfout frame, so that pairing is guarded too: the load guard and
the initialize-time refusal cover exactly the same set of selectors
(`gpuwm.physics_compat.downward_longwave_disposition` is the one
classification both read), and a config either refuses at load or
runs -- it cannot pass the door and die mid-preparation.

**What WRF v4.6.1 does with the same selectors** (read out of
`phys/module_radiation_driver.F`):

| selectors | WRF v4.6.1 | gpuwm |
|---|---|---|
| `ra_lw_physics = 0`, `ra_sw_physics > 0` | **fatal.** `radiation_driver` returns early only when both are zero (`:1068`), so this reaches `lwrad_select` (`:1839`), which has no `CASE (0)` and falls to `CASE DEFAULT` (`:2245`): `wrf_error_fatal('The longwave option does not exist: lw_physics = 0')` | refuses at config load unless declared |
| `ra_lw_physics = 0`, `ra_sw_physics = 0`, LSM on | driver returns at `:1068`; GLW keeps its Registry-allocated **0.0 W m-2** and the land surface consumes that | refuses at config load unless declared; when declared, uses 300 W m-2 (a **documented divergence** -- zero downward longwave is not a physical atmosphere) |
| `ra_lw_physics > 0` | the scheme writes GLW every radiation call | identical: the buffer allocated at init is scratch the scheme overwrites |

WRF fatally refuses shortwave-only, and longwave-only as well:
`swrad_select`'s `CASE (0)` at `:2827` calls `wrf_error_fatal` for
every `lw_physics` except Held-Suarez (`:2831-2835`), the one
idealized case it excepts. So gpuwm refusing the shortwave-only
pairing is WRF-conformant, not a divergence.

Declare a constant longwave with

```
acknowledgements = ["constant-downward-longwave-v1"]
```

-- beside whichever of the two situational tokens applies, which is how
both of the sections above print it: one edit, both lines.

in `[experiment]`. It is a **different token** from the nocturnal one
on purpose: the nocturnal token is checked before any physics is
inspected, so a config carrying it never had its GLW source examined --
which is how ten shipped configs, including the two proof descriptors
users copy first, came to integrate a frozen 300 W m-2. A config that
means both claims now makes both.

`gpuwm.core.physics.initialize_physics` enforces the same rule one
layer down: its `glw` argument has **no default**. A caller that wants
a constant types it (`glw=300.0`); a route with a source GLW hands over
the field; a suite with a longwave scheme passes nothing. Anything else
refuses rather than inventing a flux. Every resolved-configuration
report carries a `radiation.downward_longwave` line naming which of the
three it is.

**No door emits an asymmetric pairing as a default.** That includes the
HRRR route, which was the exception through 1.7.1, and the radar-DA
storm nowcast (`tools/da_nowcast.py` and the auto/launcher doors), which
was the exception through 1.8.7 --- it defaulted to
`wsm6-ysu-mm5-noah-no-radiation-v1` on cases that are mostly nocturnal,
and now defaults to the HRRR route's own
`thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1`, because the nowcast's
background is HRRR. That costs more per member than a Dudhia-only call,
so a member count or VRAM plan measured before 1.8.8 has to be
re-measured rather than extrapolated.

The HRRR route's default was `wsm6-ysu-mm5-noah-no-radiation-v1` because
every full-radiation suite the route's physics gate admits is an mp8 suite,
and the HRRR root preparer resolved microphysics lookup tables for
exactly one profile, behind two environment variables. Nothing else
staged tables at all, so the route could not default to full radiation:
full radiation here means mp8, and mp8 was not staged. 1.8 stages them
--- for **every** profile whose microphysics reads them, at
profile-binding time, through the project's packaged table ladder
(`gpuwm/data/thompson/tables`, SHA-256 pinned), before the fetch and
before preprocessing --- and the constraint is gone rather than
relaxed. The resolved table set is recorded in the physics receipt as
`microphysics_table_authority`.

The HRRR default is now `thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1`:
Thompson microphysics with RRTMG longwave **and** shortwave and no
cumulus parameterization. That is deliberately the composition the
operational High-Resolution Rapid Refresh runs (NOAA/GSL; the CCPP
`HRRR_suite` pairs Thompson aerosol-aware microphysics with RRTMG
radiation on a convection-permitting 3 km grid), so the default should
not surprise anyone who has driven WRF from HRRR before. gpuwm diverges
from operations on two components: **YSU** rather than MYNN-EDMF, and
**Noah** rather than the RUC LSM. Both of those shipped profiles run
longwave OFF and are refused by the route's own surface-layer and
land-surface pins anyway; closing that gap is a separate item.

It is **not** the `gfs`/`era5` default. `morrison-mp10-...-kf-rte-rrtmgp-v1`
selects Kain-Fritsch, and the HRRR route pins `cu_physics = 0` because
this source's native 3 km grid already resolves convection --- so that
suite is refused at emission, naming the switch. The refusal is physics,
not a limitation, and it is why HRRR's default differs from the others'.

Two consequences worth stating plainly:

* **Layer ceiling.** The legacy-RRTMG shortwave port is a transcription
  of WRF's and its wrapper caps total layers at 64. HRRR's native
  vertical is 51 levels, so this does not bite on the source's own grid;
  a hand-authored deeper vertical is refused by number, before anything
  is paid for.
* **Card cost.** The full-radiation default costs roughly 1.8 GiB more
  peak envelope than the suite it replaced, and no longer fits the
  minimum 12 km layout on a 12 GiB card. The sizing refusal now names a
  lighter `--physics-profile` alongside a shallower ladder and a bigger
  card, so the cheapest lever is stated rather than left to be guessed.

The eight asymmetric profiles stay fully selectable. Through 1.7.1 the
HRRR route wrote the nocturnal declaration for its own DEFAULT, which is
a declaration by silence and declares nothing; that died with the default
that needed it, and **nothing writes that token on a caller's behalf any
more.** Only two modules in the tree mention it at all --- the guard that
requires it (`gpuwm/physics_compat.py`) and the wizard that emits it when
you pass `--ack` (`gpuwm/domain_wizard.py`).

The route builds its experiment in code rather than from a config file,
but that experiment goes through the same `build_experiment` every config
does (`gpuwm/hrrr_route_inputs.py`), so an operator who picks an
asymmetric suite for a night window on this route meets the same load-time
refusal a config file would --- and supplies the declaration the same way,
deliberately, or picks a profile with both streams on.

## Namelist import substitutions

`gpuwm import-namelist` maps exactly two unimplemented WRF selections
to their nearest implemented counterparts, and reports each one in a
structured substitution report rather than silently rewriting:

| WRF namelist asks for | ArWen substitutes | class |
|---|---|---|
| `mp_physics = 55` (ISHMAEL) | Morrison 2-moment (10) | model-form change, no error bound |
| `ra_lw/sw_physics = 4` (RRTMG) | RTE+RRTMGP (default) or legacy RRTMG (`--rrtmg-variant rrtmg_legacy`) | explicit, token-recorded |

`bl_pbl_physics = 11` (Shin-Hong) was the third row of this table until
the Shin-Hong port: it now imports natively as 11 and runs the scheme
itself (registry maturity "implemented-unverified" -- its CPU authority
is bitwise against WRF v4.6.1, and no matched forecast trajectory
exists yet), so it is no longer a substitution.

Every other unimplemented scheme id is a hard error. Options WRF
accepts but ArWen has not validated -- moving nests, vertical
refinement, adaptive time step, `use_theta_m = 1`, non-SINT nest
interpolation -- are rejected loudly at load; the
complete register with WRF source citations is
[PROVENANCE.md](../../PROVENANCE.md).

`mp_physics = 28` is a **translation, not a substitution**: it imports
as 28 and runs the aerosol-aware scheme, so the table above stays at
exactly two substitutions. The aerosol knobs that surround it
(`use_aero_icbc`, `use_rap_aero_icbc`, `wif_input_opt`,
`num_wif_levels`, `qna_update`, `scalar_pblmix`, `grav_settling`,
`dust_emis`, `wif_fire_emit`, `wif_fire_inj`) are published in the
registry as unimplemented and refuse rather than being silently
dropped -- including where WRF *silently overwrites* them under mp=28.
Precisely, in `share/module_check_a_mundo.F`: `grav_settling` is forced
to 0 unconditionally for every mp=28 domain (`:2459-2474`);
`scalar_pblmix` is forced to 1 **only** when `use_aero_icbc` or
`use_rap_aero_icbc` is set (`:2477-2495`), which ArWen never reaches
because both are refused, and is forced back to 0 on any domain running
MYNN with `bl_mynn_mixscalars = 1` (`:2497-2511`). All three are debug-
or warning-level messages, not errors. ArWen's standing posture is to
refuse where WRF overwrites.

The full knob table around the scheme selectors -- every tweakable
namelist knob, its TOML spelling, default and allowed range, plus the
keys ArWen pins at a single validated value -- is
[CONFIGURATION.md](CONFIGURATION.md). The import report itself is
three-sectioned (translated / fixed-by-ArWen / not-implemented), so a
translated namelist never hides a knob decision.
