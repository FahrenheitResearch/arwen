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
| **implemented-unverified** | Runs on the GPU and is column-oracle-measured against unmodified WRF Fortran, but no dedicated ArWen/WRF forecast-trajectory comparison exists for it yet. The registry records its measured ULP distances and open divergences verbatim. |
| **planned / port-in-progress** | Not selectable. The registry publishes the target so the roadmap is machine-readable; nothing can resolve to it. |

An option that is not implemented is unreachable by construction -- a
config asking for it fails loudly at load, never silently substitutes.
The one exception is `gpuwm import-namelist`, which performs three
explicit, reported substitutions (see the end of this page).

## Microphysics (`mp_physics`)

| option | WRF id | maturity | evidence, in one line |
|---|---|---|---|
| Kessler | 1 | supported | warm-rain certified slice; idealized + runtime gates |
| WSM6 | 6 | supported | certified slice; matched-run anchors exist for this scheme on the reference case (refl corr 0.977 at F2, 0.815 at F5, d03) |
| Thompson | 8 | **model-validated** | full matched 6 h, 4-domain run to 500 m; decay tables in [VERIFICATION.md](VERIFICATION.md); WRF's own coefficient tables packaged and SHA-256-validated at load |
| Morrison 2-moment | 10 | implemented-unverified | 28-column oracle vs unmodified WRF `MP_MORR_TWO_MOMENT`: theta within 154 ULP, but hydrometeor fields cross branch points and are not bitwise; both rimed-ice identities (graupel/hail) implemented |
| NSSL 2-moment | 18 | **validation-candidate** | full CUDA port with fused-process oracles and a ratified 500 m comparison; explicitly not the default |

Notes with teeth:

- Thompson's four table assets (`qr_acr_qg_V4.dat`, `qr_acr_qsV2.dat`,
  `freezeH2O.dat`, `thompson_aux_tables.dat`) are byte-validated
  against pinned SHA-256 values at launch.  The two smallest ship in
  the wheel; `freezeH2O.dat` (243 MiB) and `qr_acr_qg_V4.dat` (71 MiB,
  wheel-only exclusion -- a checkout carries it) are staged by
  `gpuwm fetch-tables` (the install scripts run it automatically)
  under the same pins.  `GPUWM_THOMPSON_TABLE_ROOT` can relocate the
  root but never bypasses the pins.
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
| MYNN (EDMF) | 5 | implemented-unverified | assembled driver bitwise on the warm step vs unmodified `module_bl_mynn.F`; 300-step coupled forecast gate; admitted only as the coupled 5/5 pair with the MYNN surface layer |

The YSU registry entry records four open items verbatim, including two
FTZ-class subnormal branch disagreements and the `topo_wind=0` driver
arm difference (up to 182 ULP on 6 of 960 fixture lanes). MYNN's entry
records a real deviation: snow mixing ratio is not passed to the
condensation RH adjustment (WRF's `rh_hack`), which can flip
`CLDFRA_BL` from overcast to clear where snow is the only frozen
species above the PBL top.

## Surface layer (`sf_sfclay_physics`)

| option | WRF id | maturity (registry) | reachability (registry) | notes |
|---|---|---|---|---|
| MM5 (classic) | 91 | supported | template | the certified-slice surface layer; pairs with YSU and all three LSMs |
| MYNN | 5 | implemented-unverified | template | column solver oracle-matched over land and water (max rel. err 4.3e-7); `isftcflx` 0-3 ported; only selectable as the 5/5 pair |
| MM5 (revised) | 1 | supported | unreachable | implemented in the tree, but no registered template selects it and no route allows the override (every verified run used the classic scheme), so no config can reach it |

Maturity and reachability are separate registry axes and are quoted
verbatim here: `maturity` is the option's evidence tier, while
`reachability.state` says whether any registered template or override
route can actually select it (`template` = a registered template
selects it; `unreachable` = nothing can, and the registry's blocker
text records why).

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
| WRF RRTM + Dudhia | 1/1 | port-in-progress | not selectable; refused at load |

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

## Cumulus (`cu_physics`)

| option | WRF id | maturity | notes |
|---|---|---|---|
| Kain-Fritsch | 1 | supported | outer (>=10 km) domains; packaged lookup table; cudt 5 min in the certified templates |
| off | 0 | supported | the convection-permitting nests run with cumulus off |

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

## Which suites each data route can actually prepare

Selectable is not the same as preparable, and the difference is
route-dependent. State of play in v1.1.1:

| route | what it can prepare |
|---|---|
| ERA5 config door (`[case_data]` -> `gpuwm run`) | the registry-admitted combinations |
| GFS single domain | WSM6, Thompson, Morrison, NSSL2, MYNN; Noah-MP with an expert acknowledgement. RUC is deliberately withdrawn on this route |
| ERA5 single domain | the same normal profiles, plus RUC |
| HRRR single domain | the normal profiles, plus RUC and expert Noah-MP |
| prepared domain trees (GFS, ERA5, HRRR, 20CRv3) | the normal profile family, plus expert Noah-MP |

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

The physics suite is yours to choose. The GFS/ERA5/20CRv3 prepared
single-domain runner and the multi-domain tree runner execute any suite
the engine implements, exactly as your config writes it; there is no
profile whitelist on those routes. `--physics-profile` is optional
there: naming one asserts your config IS that shipped suite, and the
runner refuses on any switch drift, which is how you keep a run pinned
to a published product. The one route this does NOT describe is HRRR
cold start: its evidence contract is keyed by shipped profile, so it
prepares WSM6 when no profile is named and refuses a name outside its
contract -- an unnamed HRRR config's own suite does not run as written
there. Whether the suite you selected carries WRF-verification
evidence is stated -- one sentence in the wizard
output, the run receipt, and `--explain` -- and never gates: a suite
without evidence prints "supported, not yet WRF-verified" and the run
continues. An expert-template suite (Noah-MP) keeps its registry-owned
acknowledgement on every route, delivered as `--ack <id>` or
`acknowledgements = ["<id>"]` in the experiment. `gpuwm domain
--explain` lists the shipped profiles and
what each one actually runs -- several run full RTE+RRTMGP with
Kain-Fritsch, several run longwave OFF with Dudhia shortwave and no
cumulus. Read the names. What still refuses, on every route, is a
switch value the engine genuinely does not implement (the refusal names
the switch) and the registry's land-surface route blockers (for
example GFS+RUC, which dies at its first surface-temperature call).

## Namelist import substitutions

`gpuwm import-namelist` maps exactly three unimplemented WRF selections
to their nearest implemented counterparts, and reports each one in a
structured substitution report rather than silently rewriting:

| WRF namelist asks for | ArWen substitutes | class |
|---|---|---|
| `mp_physics = 55` (ISHMAEL) | Morrison 2-moment (10) | model-form change, no error bound |
| `bl_pbl_physics = 11` (Shin-Hong) | YSU (1) | model-form change, no error bound |
| `ra_lw/sw_physics = 4` (RRTMG) | RTE+RRTMGP (default) or legacy RRTMG (`--rrtmg-variant rrtmg_legacy`) | explicit, token-recorded |

Every other unimplemented scheme id is a hard error. Options WRF
accepts but ArWen has not validated -- moving nests, vertical
refinement, adaptive time step, `use_theta_m = 1`, non-SINT nest
interpolation -- are rejected loudly at load; the
complete register with WRF source citations is
[PROVENANCE.md](../../PROVENANCE.md).

The full knob table around the scheme selectors -- every tweakable
namelist knob, its TOML spelling, default and allowed range, plus the
keys ArWen pins at a single validated value -- is
[CONFIGURATION.md](CONFIGURATION.md). The import report itself is
three-sectioned (translated / fixed-by-ArWen / not-implemented), so a
translated namelist never hides a knob decision.
