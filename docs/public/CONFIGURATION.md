# Configuration knobs (WRF namelist parity)

**Two ways in, both first-class.** Write the experiment TOML yourself --
`[shared]` and `[[domain]]` take the WRF namelist keys verbatim, same
spelling, same meaning, per-domain where WRF is per-domain -- or hand
`gpuwm import-namelist WPS INPUT` a WRF `namelist.wps` +
`namelist.input` pair and it writes one for you. A shipped physics
profile is a third and shorter road to the same loader: a name for
switches you could have typed, never a gate on what you may write
([PHYSICS.md](PHYSICS.md)).

ArWen's configuration surface is the experiment TOML
(`[experiment]` / `[projection]` / `[shared]` / `[[domain]]` /
`[case_data]` / `[perturbation]` / `[tiles]` / `[output]`), and the
import contract is *never silent*: every namelist key lands in exactly
one of three report sections --

- **translated** -- a TOML value came out of it (including the three
  ratified physics substitutions);
- **fixed by ArWen** -- WRF has options there, ArWen implements exactly
  one value; the key is validated against that value (anything else is
  a hard error) and the report records the pin and why;
- **not implemented** -- consumed without a counterpart, each with a
  reason.

This page is the complete knob table. "Default" is the value a key
takes when omitted from the TOML (the frozen `RunConfig` default,
`gpuwm/config.py`); where WRF's Registry default differs, the importer
emits the WRF value explicitly so an imported experiment resolves the
way WRF would. Physics *selection* values and their maturity labels
live in [PHYSICS.md](PHYSICS.md); this page covers the knobs around
them.

## `[output]` -- which variables the wrfout files carry

WRF's `iofields_filename`, as a selection over the inventory the run
already produces. `preset = "full"` (the default) writes everything;
`"minimal"` and `"severe"` are named sets; `history_vars` /
`history_drop` are explicit include and exclude lists, mutually
exclusive. The same table sits inline on a `[[domain]]` as
`output = { ... }` and overrides the tree-wide one for that domain, as
`[tiles]` does. It changes no number the model computes and does not
enter the restart identity, so a trimmed run resumes a full run's
checkpoints. Full page:
[OUTPUT-VARIABLES.md](OUTPUT-VARIABLES.md).

## `[case_data]` -- the inputs a config-driven run declares

`gpuwm run CONFIG.toml` is the config-driven route, and `[case_data]` is
where that config names the data it runs on. Every input path and every
policy is **declared, never implicit**: the loader
(`gpuwm/case_data.py`) refuses an unknown key and refuses a missing
required one, listing all of them at once rather than the first it
reached.

`gpuwm domain` and `gpuwm import-namelist` write this table for you. The
table below is for hand-authoring it, and for reading what the wizard
wrote.

**Required -- the run is refused without them.**

| key | what it names |
|---|---|
| `forcing` | the meteorological input: one path, a list of paths, or a glob. A glob that matches nothing is refused, except under `gpuwm static`, which never opens it. |
| `vtable` | the WPS Vtable selector authority for `forcing`. Also unread by `gpuwm static`. |
| `wps_namelist` | the `namelist.wps` whose geometry the domains are built on. |
| `geog_root` | the WPS_GEOG static geography tree ([DATA.md](DATA.md)). |
| `sfcp_to_sfcp` | real-init surface-pressure policy: a knob, and `false` is a legitimate answer, so it is declared rather than defaulted. |
| `output_title` | the `TITLE` global attribute written into every wrfout this config produces. It is required because it is provenance: a history file that cannot say which experiment wrote it is not reproducible, and there is no sensible default for the name of your own run. |

**Optional.**

| key | what it names |
|---|---|
| `forcing_interval_s` | forcing cadence, when it should not be discovered from the files. |
| `output_domain` | which domain's history the run publishes. |
| `source_orography` | a NetCDF file supplying source orography; without it the forcing catalog's invariant geopotential is used. |
| `source_orography_variable` | the variable to read out of `source_orography`. |
| `co2_vmr` | Positive COâ‚‚ mole fraction consumed by the selected classic RRTM, legacy RRTMG or RRTMGP absorption; e.g. `0.000420` for 420 ppm. Off, analytic and Dudhia-only radiation retain it as inactive input authority. |
| `water_temperature_overlay` | a water-temperature overlay file ([water-temperature-overlay](../water-temperature-overlay.md)). |
| `water_temperature_policy` | how that overlay is applied. |
| `preprocess_backend` | where the root domain's preparation runs: `"cuda"`, `"cpu"` or `"auto"`. Absent is `auto`, which prepares on the CPU when the card reads busy or cannot hold the preparation, so two runs you compare could start from different preparations; naming it pins both. `gpuwm run --preprocess-backend` overrides it. Nests prepare on the card either way. |

Per-domain source orography is declared as `d01`, `d02`, ... keys inside `[case_data.source_orography]`.

`gpuwm run` on a config with **no** `[case_data]` table refuses by name
and lists the required inputs; that refusal and this table are the same
list.

## Tweakable knobs

### `[experiment]`

| TOML key | WRF equivalent | default | allowed | note |
|---|---|---|---|---|
| `diff_opt` | `diff_opt` | 2 | 1, 2, per domain | 1 selects model-coordinate horizontal diffusion with `km_opt = 2` or 4; 2 selects terrain-aware metric stress and scalar diffusion. |
| `mix_full_fields` | `mix_full_fields` | true | bool, per domain | WRF logical retained under coordinate diffusion. That operator mixes theta relative to its initial field for either value. |
| `name` | -- | required | non-empty string | run identity |
| `start_time` | `&time_control start_*` | required | TOML datetime, offset-free | |
| `run_seconds` | `run_days/hours/minutes/seconds` or `end_*` | required | > 0 | |
| `restart_interval_s` | `restart_interval` (minutes in WRF) | required | >= 0; 0 disables | whole multiple of d01 dt |
| `feedback` | `feedback` | 0 | **0 or 1** | 0 is one-way and supported; 1 is the EXPERIMENTAL two-way path (stamped as such in run provenance; one-way consumers refuse a feedback-modified parent). Any other value is refused, and a two-way namelist is never quietly imported one-way |
| `smooth_option` | `smooth_option` | 0 | **0 only** | the parent smoother acts only under two-way feedback |
| `blend_width` | `blend_width` | 5 | >= 0 | terrain blend zone; enters the parent-row clearance rule |
| `spec_bdy_width` | `&bdy_control spec_bdy_width` | 5 | >= spec_zone + relax_zone | |
| `smooth_cg_topo` | `&domains smooth_cg_topo` | false | bool | WRF v4.7.1's d01 boundary terrain blend: the outer `spec_bdy_width + blend_width` rows of domain 1's terrain are blended toward the source model's own terrain, bit for bit as WRF's `blend_terrain` does, once at the first time. Needs the source's terrain (its SOILHGT); refused without it |
| `column_chunk` | -- | 3125 | >= 1 | ArWen-only radiation throughput knob; byte-identical across values |
| `physics_mode` | -- | absent (see note) | `"wrf-faithful"` or `"arwen-patched"` | ArWen-only physics-FIDELITY axis. Present, it becomes the author of every divergence-ledger key and writes the faithful or patched side of each edge onto every domain; an explicit occurrence of one of those keys in `[shared]` or `[[domain]]` is then refused rather than merged, because a key with two authors runs a value neither of them chose. ABSENT it authors nothing, which is what every configuration written before the axis means -- and the reported mode is still `wrf-faithful`, because no registered patch is applied. The register is PROVENANCE.md, "Divergence ledger v1"; the resolved vector lands in the run receipt |
| `patchset` | -- | `"v1"` | a registered patch-set version | Which frozen ledger set the axis resolves. A version is frozen when it is registered, so a receipt naming `v1` keeps meaning the vector it meant; later entries get a new version beside it |
| `patches` | -- | the whole set | array of ledger entry ids, e.g. `["L4"]` | The single-patch ablation arms. Only under `physics_mode = "arwen-patched"` -- a subset of the patches APPLIED is meaningless when none is. An entry the ledger holds back (SASE's entry gate; the dormant class-C rows) is refused with the gate as the reason |

### `[projection]` and WPS geometry

`map_proj` is one of `"lambert"` (Lambert conformal, either
hemisphere), `"mercator"`, or `"polar"` (polar stereographic, either
pole), with `ref_lat`, `ref_lon`, `truelat1`, `truelat2`, `stand_lon`
-- the WPS `&geogrid` set, all required, all inside the experiment
fingerprint (Mercator and polar consume `truelat1`; `truelat2`
mirrors it). All three projections are transcription-gated at
binary64 against the pinned WRF v4.6.1 `share/module_llxy.F` oracle
(`tests/test_projection_oracle.py`), but their maturity differs:
northern-hemisphere Lambert carries the matched-run validation
family, while Mercator, polar stereographic, and southern-hemisphere
Lambert are oracle- and smoke-verified only -- see the worldwide
section of [VERIFICATION.md](VERIFICATION.md) and the projection
maturity rows in [PHYSICS.md](PHYSICS.md). Latitude-longitude
(cylindrical) and rotated grids are refused, as are domains
containing or touching a pole and forcing footprints wider than 180
degrees of longitude. Domain layout (`e_we`/`e_sn`, `parent_id`,
`i/j_parent_start`, `parent_grid_ratio`, `parent_time_step_ratio`)
translates 1:1 with WRF's staggered-to-mass conversion (one fewer
point); child `dx`/`dt` are never hand-typed -- they derive exactly
from the parent chain and hand-typed values are cross-checked, not
copied.

### Vertical grid (`[shared]`)

| TOML key | WRF equivalent | default | allowed | note |
|---|---|---|---|---|
| `e_vert` / `nz` | `e_vert` | required (one of) | nz >= 4 | `nz = e_vert - 1` mass levels |
| `eta_levels` | `eta_levels` | () | 1.0 -> 0.0 strictly decreasing | explicit coordinates are required by config-driven native preparation; the `run --met-em` door generates omitted levels with WRF `auto_levels_opt` 1/2 and records the resolved controls; explicit `eta_levels` bypass generation |
| `p_top` | `p_top_requested` | 0.0 | >= 0, Pa | Registry default 5000 Pa applies on import when omitted |
| `hybrid_opt` | `hybrid_opt` | 0 (legacy) | 0/1 (sigma), 2 (WRF cubic-B) | importer default 2 (Registry) |
| `etac` | `etac` | 0.2 | [0, 1] | the value asked for; preparation lowers it when this run's terrain needs it (below) |
| `ztop` | -- | required | > 0 | ArWen-only scaffold height; real runs derive heights from p_top/eta_levels |

#### The coordinate a run's terrain can order

At `hybrid_opt = 2` the WRF cubic `B(eta)` can only order a column while
its surface pressure stays above a floor set by `etac` and `p_top`; WRF
itself calls a column below that floor fatal
(`dyn_em/nest_init_utils.F:1158-1182`, "tends to be caused by very high
topography", remedy "reduce etac"). At the shipped `etac = 0.2` with
`p_top = 10000` Pa that floor is 46408 Pa, about 6082 m of terrain.

Preparation does not stop there. It surveys every terrain field the run
can touch -- each declared domain's static terrain at its own resolution
(a nest carries higher peaks than its parent), and for a following nest
the whole statics corridor it may traverse -- and sets `etac` to the
largest value that orders the lowest surface pressure it found. `p_top`
is never changed: `etac` is WRF's own named remedy, and it keeps the
model top where you put it.

When the derived value differs from the configured one, preparation says
so in one line naming the column, its height and the value chosen, and
`proof.json` carries a `vertical_coordinate` block with the same numbers
plus every terrain field surveyed. The forecast then runs on the
coordinate the prepared inputs carry, not on the configured one: the
prepared caches hold the coefficient arrays themselves, as WRF's
`wrfinput` holds `C3H`/`C4H`, and the runner adopts the value beside them
after checking that it is at or below the one you configured and that it
orders every prepared column.

Every domain of the run takes that one coordinate, not the root alone: a
nest on the streamed road rebuilds its tile buffer from it, a nest
spawned mid-run is priced on it, and an offline child reads it off the
history's `ETAC` attribute. A domain tree that somehow held two is
refused by name rather than integrated.

Terrain that no positive `etac` orders is still refused, with the
constraint, the column and the remaining remedy (a lower model top).

A preparation with nothing to derive writes the block anyway, at status
`NOT_APPLICABLE`, and its `why` names which of the three reasons applied:
the configuration carries no eta ladder, so no terrain was surveyed;
`hybrid_opt` is 0 or 1, where `B(eta) = eta` leaves no terrain ceiling to
derive; or `hybrid_opt` is 2 over an eta ladder with no segment where
`dB/deta` rises above 1, which the ladder and `etac` set between them,
leaving the surface-pressure floor at zero so no ground can reach it.

### Clock (`[[domain]]`, root)

`time_step` (integer seconds) plus optional `time_step_fract_num` /
`time_step_fract_den` -- WRF's exact rational clock keys. Children
carry no clock: `dt_child = dt_parent / parent_time_step_ratio`
exactly, chained in float32 exactly as WRF chains it.
`history_interval_s` is per-domain and must divide into whole domain
steps. `history_begin_s` / `history_end_s` (per-domain, WRF's
`history_begin_*` / `history_end_*`) window the frames: the first frame is
written at `history_begin_s` after the start, rounded up to the domain's
step as WRF's alarm rings, and none after `history_end_s`; absent, every
interval from the start is written. With the adaptive time step on (`use_adaptive_time_step` in
`[shared]`, `docs/ADAPTIVE-TIMESTEP.md`) each domain's step instead follows
its own measured Courant number between `min_time_step` and
`max_time_step`, landing on every output time.

Steep ground under a strong wind at crest height takes a shorter step.
Before a forecast starts, each domain is read for its steepest slope, its
highest ground and the strongest wind its start state and boundary data
carry over the forecast window from the ground up to that height. Where
the engine's measured stability map (`gpuwm/terrain_clock.py`,
`gpuwm/terrain_clock_map.json`) says the configured step and substep count
do not hold there, the domain runs the smallest whole division of its step
the map holds, with the fewest substeps (the configured count or 6) that
hold it, prints one line saying so and records it under `terrain_clock` in
the run report. A nest's step ratio grows by whatever division its parent
does not already give it, so every cadence stays a whole number of steps.
Under the adaptive clock the held step caps `max_time_step`, and a count
the rule raises becomes `min_time_step_sound`, the fewest substeps that
clock takes whatever step it adapts to. A domain the map holds runs
exactly as configured. The map was measured with fixed steps up to 5 s
per km of grid spacing (15 s at 3 km), and at 3 km up to 40 s: rows of
crests up to 4.5 km and slopes up to 0.4 were tried from 40 s down to 20 s
under 20 to 90 m/s on four and six substeps, and every step that held was
run again for three hours. Where a longer step stopped, the step that held
caps an adaptive `max_time_step` and divides a fixed step above it. A
domain is read against every mapped row as gentle as its ground or
gentler, at every wind up to its own, and a stop seen on any of them
counts: 3 km ground steeper than 0.4, itself measured only to 15 s, is
capped by the stops its gentler rows saw at 20 s, winds above 90 m/s by
those seen at 90 m/s, and a spacing between 2 and 3 km by its 3 km rows.
3 km crests above 4.5 km were measured only to 15 s, so a longer step
seen to stop under the 4.5 km and lower crests, at the same slope and
wind, caps them too: a 5.5 km crest of slope 0.36 under 30 m/s is capped
at 15 s as a 3.9 km one is. Only where nothing read saw a longer step
stop is a longer `max_time_step` left to the adaptive clock's own
vertical and horizontal CFL limits. A cap only ever shortens the step:
one at or above the adaptive clock's own longest step (your
`max_time_step`, or 8 s per km of grid spacing when it is -1, 24 s at
3 km) is not written. Those stops were seen on fixed steps, so the
four-substep rows at 2 and 3 km of crests up to 4.5 km and slopes up to
0.4 (under a 4.5 km crest, to 0.47 at 3 km and 0.48 at 2 km) were also run on the adaptive clock,
for three hours at `max_time_step` from 15 down to 5 s per km, 20 to 60 m/s.
Where every row a domain reads held the longest step tried there, and its
ground is no steeper than the steepest row at its crest, its adaptive clock
is capped at that step (45 s at 3 km, 30 s at 2 km) instead of the fixed-step
stop, unless your `target_cfl`, `target_hcfl` or `max_step_increase_pct` is above what was measured (1.4, 0.98 and 5).
Where a row saw a longer adaptive step stop, the step it held caps the
clock whatever the fixed rows say, and where a row held none the line
says the domain may still stop. A fixed step reads only the fixed-step
rows. On a 3 km CONUS domain (highest ground about 3.9 km, steepest
slope 0.36), the 24 s `max_time_step` the adaptive clock takes by default
stands under a crest-level wind of 20 and 30 m/s, since every row read
held 45 s on the adaptive clock there; at 40 m/s it is capped at 15 s,
because 20 s steps stopped there. At 50 and 60 m/s the cap
depends on the configured `time_step`: with 12 s it is 13.5 s, as before;
with the 15 s the domain wizard writes, the clock now takes at least six
substeps with the step capped at 15 s, where before it kept the default
24 s on six. From 70 m/s the cap is 12 s or less with either, as before.
A fixed 3 km step of 15 s or less (the domain wizard
writes 15 s) reads as before. Over that CONUS ground a fixed 18 s or 20 s
runs on six substeps at 30 m/s, and from 40 m/s is halved to 9 s or 10 s.

Each `[[domain]]` may also carry an offset-free `start_time`. It defaults to
`[experiment].start_time`; d01 must equal that root start. A delayed child is
dormant until its timestamp, then follows the ordinary parent-state nest
initialization path -- it initializes from the analysis valid at its
activation instant, so that instant must be one the declared inputs decode.
The timestamp must be an exact boundary of its parent's step clock and an
exact external-forcing seam. Its first history frame is the domain's own
analysis frame and carries no `REFL_10CM`, exactly as d01's frame at t = 0
does; every frame after it carries the field. Restart headers persist
`STARTED`/`NOT_STARTED` lifecycle state, so resuming before the timestamp does
not initialize the child early.

Delayed activation is a `gpuwm run` capability. The prepared-tree runner
(`gpuwm-prepared-tree-forecast`, which `gpuwm stream` and the wizard's nested
route drive) restores every domain from a prepared cache and has no way to
bring one to life mid-run, so it refuses a delayed `start_time` by name and
tells you which door runs it.

External boundary cadence has no whole-hour rule. It must be a positive,
uniform whole-second interval and an exact integer number of d01 steps because
the Davies boundary clock resets at a top-of-step seam. Five-minute forcing is
therefore valid for a 60-second d01 step; 310-second forcing is not
(`310/60 = 31/6` steps). History cadence is independent.

### Dynamics

Shared across domains unless marked per-domain. Every key below is a
consumed `RunConfig` field -- the knob-parity battery
(`tests/test_namelist_import.py`) proves each one lands on the
consuming kernel/module rather than being decorative -- and every one
is importable from a WRF namelist.

**Which keys a `[[domain]]` table may override.** Exactly these 69,
and no others (`gpuwm/experiment.py`'s `_DOMAIN_RUN_OVERRIDES`):

    cu_physics  cudt_minutes  clos_choice  ishallow  radt  radt_minutes  bldt
    ra_physics  ra_lw_physics  ra_sw_physics  ra_rrtmg_variant
    wrf_rrtmg_compatibility  o3input  use_mp_re  swrad_scat  diff_6th_factor  epssm
    spec_exp  mp_physics  moist  moist_cq  nest_microphysics_transition  km_opt
    bl_pbl_physics  sf_sfclay_physics  c_s  c_k  moist_mix6_off  diff_6th_opt
    mix_isotropic  mix_upper_bound  isfflx  tke_heat_flux  tke_drag_coefficient
    tke_upper_bound  diff_6th_slopeopt  diff_6th_thresh  dampcoef  zdamp  emdiv
    smdiv  khdif  kvdif  diff_opt  mix_full_fields  h_sca_adv_order  moist_adv_opt
    tke_budget  sase_flux_diag  hmix_k_diag  inflow_perturbation
    inflow_perturbation_seed  inflow_perturbation_amplitude_scale
    inflow_perturbation_faces  target_cfl  target_hcfl  max_step_increase_pct
    starting_time_step  starting_time_step_den  max_time_step  max_time_step_den
    min_time_step  min_time_step_den  min_time_step_sound  slope_rad  topo_shading
    mosaic_urban_canopy  topo_wind  gwd_opt

`clos_choice` and `ishallow` configure the Grell-Freitas cumulus scheme
(`cu_physics = 3`): which closure the deep scheme uses (0, the default,
is the mean of all sixteen members; 1 to 16 run one member alone, with
a warning that only 0 has been compared against WRF), and whether the
shallow scheme runs. They are per domain because the scheme they
configure is, and they are validated inert on any domain that does not
select `cu_physics = 3`. On a tree where some domains run Grell-Freitas
and others do not, a value in `[shared]` reaches the Grell-Freitas
domains and leaves the others at 0; a value written into a non-Grell
domain's own table is refused. The HRRR route writes both keys into its
namelists, where WRF reads each once for the whole run, so on that
route every Grell-Freitas domain of a tree takes the same values.

The eleven numerics from `diff_6th_slopeopt` through `tke_budget` are
per domain because WRF declares every one of them `max_domains` and the
split here had drifted from that: `diff_6th_opt` and `diff_6th_factor`
were per domain while `diff_6th_slopeopt` and `diff_6th_thresh` -- the
same filter -- were not, and `epssm` was while `emdiv` and `smdiv` were
not. A tree can now damp or filter the nest that needs it without
moving its parent, which a refinement tree needs: the relaxation sponge
is 40 km wide on a 10 km root and 2.7 km on a 667 m nest at the same
cell count. Geometry (`dx`, `dy`, `ztop`, `grid_id`, `nested`,
`specified`) stays tree-wide because the domain tree authors it, and so
do two of the scheme selectors WRF also scopes `max_domains` --
`sf_surface_physics` and the `bl_mynn_*` block: a tree whose domains ran
different land-surface or MYNN closures cannot be compared across its
own boundary, and two-way feedback already requires one microphysics
tree-wide. The radiation selectors are not among them; the radiation
row below says why.

The nine adaptive-time-step keys from `target_cfl` to
`min_time_step_den` are per domain because each domain runs its own
controller (see `docs/ADAPTIVE-TIMESTEP.md`). Note that a clamp is an
absolute number of seconds while a nest's step is a fraction of the
root's, so a `min_time_step` written once in `[shared]` reaches every
domain unchanged and can sit ABOVE the step it was meant to protect on
an inner nest. Set the clamps per domain on a refinement tree.
`min_time_step_sound` is per domain because the ground that needs it is:
the steep-terrain rules set it on the domains whose substep count they
raise and leave every other domain at WRF's derived count.

`sase_flux_diag` and `hmix_k_diag` are output-only diagnostics, and
they are per domain for the same reason: their cost scales with the
grid, so a tree can carry them on the domain whose mixing or subgrid
flux is being read and leave them off the rest.  The turbulence row
(`km_opt` through `tke_upper_bound`) and the PBL/surface selectors are
per domain because that is what makes a PBL parent able to carry a
PBL-off LES child (see `docs/public/LES.md`).  The four
`inflow_perturbation*` keys seed an LES nest child's inflow turbulence
transition (default off; deterministic, seeded, and gated
byte-identical to a build without the mechanism when off); they are
per domain because the mechanism is per nest edge â€” it perturbs one
child's parent-forced boundary tables â€” and, like per-domain
`isfflx`, they have no WRF namelist spelling, so a config using them
cannot round-trip to a namelist.  `inflow_perturbation = true` needs a
parent that runs a PBL scheme: the perturbation's depth is the parent's
diagnosed PBLH, so a child under a `bl_pbl_physics = 0` parent is
refused at load, by name, with the domain to change.  That is the
mesoscale-to-LES edge and only that edge â€” a PBL-off parent is itself
LES, and its resolved eddies already are the child's inflow turbulence.

Only `gpuwm domain`'s own emission and hand-written TOML reach some of
them, so the list is stated here rather than left to be discovered. A
`[[domain]]` table carrying any other key is **refused** naming the
key, not accepted and not silently dropped: a config that appeared to
ask for it while the `[shared]` value ran on every nest would be a
wrong answer reported as a success. Put them in `[shared]`.
`bl_pbl_physics` takes every scheme per domain, SASE (900) included.

| TOML key | WRF equivalent | default | allowed | note |
|---|---|---|---|---|
| `time_step_sound` | `time_step_sound` | 4 | even, > 0 | WRF 0 = auto imports as 4, recorded. A domain whose terrain is steeper than four substeps were measured stable on (a slope of 0.70 in any direction at `epssm` 0.5, lower at smaller `epssm`) runs 6 and the run says so; a larger value is never lowered (`gpuwm/acoustic_adaptation.py`). Under the adaptive clock the count follows the step, and the 6 is held as `min_time_step_sound` |
| `min_time_step_sound` | -- (ArWen) | 0 | even, >= 0, per domain | under the adaptive clock, the fewest acoustic substeps per step: the count the clock derives from its step (WRF's `time_step_sound = 0` rule, 4 at any step under about 3.3 s at 1 km) is raised to this. 0 keeps WRF's count. The steep-terrain rules set it on each adaptive domain whose count they raise. Nothing reads it under a fixed clock |
| `epssm` | `epssm` | 0.1 | per-domain | acoustic off-centering; scalar namelist assignment changes d01 only (Registry tail keeps 0.1), preserved per-domain |
| `smdiv` | `smdiv` | 0.1 | finite | 3-D divergence damping |
| `emdiv` | `emdiv` | 0.0 (ArWen legacy) | finite | WRF Registry default 0.01 is emitted explicitly on import |
| `km_opt` | `km_opt` | 1 | 1 (constant K), 2 (1.5-order prognostic TKE), 3 (3-D Smagorinsky), 4 (2-D Smagorinsky), 0 (no operator -- with `bl_pbl_physics = 900`, or with `km_opt_zero_acknowledgement`) | `diff_opt=2` form implied; WRF -1 must-set honored: omission refuses. **2 and 3 are the LES closures and carry extra conditions:** `km_opt=2` is admitted only with `bl_pbl_physics=0`; on a nest child it cold-starts its own TKE whatever the parent runs, and under a `km_opt=2` parent the load warns that the tree is not yet verified (see `gpuwm/experiment.py`). `km_opt=3` has no nest restriction. See `docs/public/LES.md` |
| `km_opt_zero_acknowledgement` | (none) | `""` | the exact id `no-horizontal-mixing-operator-v1` | admits `km_opt = 0` with a PBL scheme that produces no horizontal mixing of its own -- i.e. a run with NO horizontal mixing operator, WRF's `diff_opt = 0`. Refused by default because that is what a mis-set switch looks like; the acknowledged path is the single-variable research control that varies the closure while holding the mixing at none. A literal id, not a boolean, so no stray `= true` reaches it. Refused where it would acknowledge nothing (`km_opt != 0`, or SASE, which supplies the producer). Not needed with `bl_pbl_physics = 900` |
| `hmix_k_diag` | (none) | false | bool, per domain | publishes the horizontal eddy viscosities the run's own producer used, under that producer's name: `XKMH`/`XKHH` for `km_opt = 4`, `SASE_KMH`/`SASE_KHH` for the SASE closure. Same units (m2 s-1), same mass grid, so the two are directly comparable. A run with no producer publishes neither pair -- an absent variable cannot be misread as a measured zero. Two extra (nz, ny, nx) planes per frame |
| `c_s` | `c_s` | 0.25 | > 0 | Smagorinsky constant (smag2d kernel); per-domain |
| `c_k` | `c_k` | 0.15 | > 0 | km_opt=2 TKE-closure constant, K = c_k sqrt(e) l; the WRF `em_les` reference namelist sets 0.10; per-domain |
| `mix_isotropic` | `mix_isotropic` | auto | 0, 1, `"auto"` | 0 = anisotropic mixing lengths, 1 = isotropic (dx dy dz)^(1/3); per-domain. Unset (or the string `"auto"`) lets the model choose: isotropic where `mix_upper_bound*(dz_max/dx)^2` exceeds 0.25 (announced at load and by `gpuwm check`), the WRF-default anisotropic form otherwise. A written 0 is honoured everywhere, with a warning naming the instability and the measured ratio in the danger zone |
| `mix_upper_bound` | `mix_upper_bound` | 0.1 | > 0 | non-dimensional cap K <= mix_upper_bound len^2 / dt, applied per direction; per-domain |
| `tke_upper_bound` | `tke_upper_bound` | 1000.0 | > 0 | km_opt=2 TKE ceiling in m2 s-2, `bound_tke` clamp; per-domain |
| `tke_heat_flux` | `tke_heat_flux` | 0.0 | finite | prescribed kinematic surface heat flux, K m s-1; consumed under `isfflx` 0 and 2 with the PBL off; per-domain |
| `tke_drag_coefficient` | `tke_drag_coefficient` | 0.0 | >= 0 | prescribed surface drag coefficient; consumed under `isfflx=0` with the PBL off; per-domain |
| `khdif`, `kvdif` | `khdif`, `kvdif` | 0.0 | >= 0 | km_opt=1 only; refused with open/specified boundaries |
| `diff_6th_opt` | `diff_6th_opt` | 0 | 0, 1, 2 | option 1 refused when moist (PD bypass) |
| `diff_6th_factor` | `diff_6th_factor` | 0.12 | per-domain | |
| `diff_6th_slopeopt` | `diff_6th_slopeopt` | 0 | 0, 1 | terrain-slope taper |
| `diff_6th_thresh` | `diff_6th_thresh` | 0.10 | > 0 | slope threshold, m/m |
| `damp_opt` | `damp_opt` | 0 | 0, 3 | Rayleigh implicit w-damping |
| `zdamp` | `zdamp` | 5000.0 | m | |
| `dampcoef` | `dampcoef` | 0.2 | | |
| `w_damping`, `w_crit_cfl` | `w_damping`, `w_crit_cfl` | 0, 1.0 | 0, 1; > 0 | `w_crit_cfl` is where w-damping measures the excess vertical Courant number from, and with `zadvect_implicit = 1` where it starts (WRF suggests 2.0 there); without it damping starts at 1, so a value above 1 is refused there (it would push `w` along its own direction) |
| `zadvect_implicit` | `zadvect_implicit` | 0 | 0, 1 (a positive WRF value imports as 1) | WRF's implicit-explicit vertical advection on the last RK substep; refused with open boundaries. Matches WRF v4.7.1's routines word for word except two boundary terms of the implicit `w` solve, a declared divergence: WRF builds the lower one from the mass-coupled u/v tendencies, about one column mass too large (a steep-ridge run went NaN in three steps), and leaves the upper one's geopotential change over dt undivided by g. ArWen uses the uncoupled tendencies and divides by g |
| `base_temp` | `base_temp` | 290.0 | K | base state; init-time only (see fixed table for `iso_temp`/lapse) |
| `hypsometric_opt` | `hypsometric_opt` | 1 (ArWen legacy) | 1, 2 | WRF Registry default 2 emitted explicitly on import; WRF declares this key in **`&domains`**, as one scalar for the whole run (`Registry.EM_COMMON:2283`) -- a namelist that puts it in `&dynamics` is one `wrf.exe` cannot read, and the importer refuses it there by name |
| `h_sca_adv_order` | `h_sca_adv_order` | 2 (ArWen legacy) | 2, 5 | **feeds the geopotential equation only**; transported-scalar stencils are fixed 5th/3rd order, so the importer accepts only the Registry default 5 |
| `moist_adv_opt` | `moist_adv_opt` | 1 | 0, 1 in TOML; import pins 1 | PD limiter; `scalar_adv_opt` must match (WRF option 1 on both) |
| `top_lid` | `top_lid` | **true** (ArWen) | bool | WRF Registry default is false (open top); ArWen defaults to the rigid lid after the 2026-07-18 open-top NaN probes -- imports emit the Registry value explicitly, flip back only with a stability receipt |
| `moist_cq` | -- (WRF derives cq from its moist state) | **true** | bool | applies whenever water vapor exists, including passive vapor with microphysics off; dry states bypass it. Explicit `false` is a verification counterfactual |
| `spec_zone`, `relax_zone`, `spec_exp` | `&bdy_control` | 1, 4, 0.0 | | `spec_exp` acts on the root (specified) branch only, exactly as in WRF's `lbc_fcx_gcx`; nonzero on a nested child is refused. `gpuwm downscale --point` sets `relax_zone` to two parent cells and `spec_exp` to 0 |
| `relax_timescale_s` | -- (ArWen) | 0.0 | >= 0 | Davies relaxation time scale in seconds on the first relaxed row: `fcx = ramp / relax_timescale_s`, `gcx = ramp / (5 relax_timescale_s)`. 0 is WRF's recipe (`0.1/dt`, `1/(50 dt)`, a time scale of 10 of the domain's own steps). A nest reads it in WRF's nested operation order. `gpuwm downscale --point` sets it to the time a 20 m/s flow takes to cross one child cell, never shorter than 10 child steps |
| `relax_w` | -- (ArWen) | false | bool | a specified domain relaxes `w` toward its boundary table and takes the table's `w` on the specified rows, as a WRF nest does. False is WRF's root rule: `w` is not relaxed and the specified rows copy the first interior row. Needs a `w` table (an offline child's parent history carries `W`); without one the run stops at its first step saying so. `gpuwm downscale --point` sets it |

### Physics cadences and scheme knobs

| TOML key | WRF equivalent | default | allowed | note |
|---|---|---|---|---|
| `ra_lw_physics` | `ra_lw_physics` | -1 | 0 (off), 1 (WRF RRTM), 4 (RTE+RRTMGP), 90 (analytic proxy) | per-domain. The split spelling of the radiation selection, and the only spelling that can ask for a MIXED pair: `ra_physics = N` means N on both streams, so longwave off under Dudhia shortwave (0/1) and WRF RRTM under Dudhia (1/1) can be written no other way. `-1` means this table does not state it, and the stream takes `ra_physics` instead. Stating one half and leaving the other at `-1` is refused -- `ra_lw_physics and ra_sw_physics must both be explicit or both be -1` -- because a half-stated pair reads as a selection and resolves as the aggregate |
| `ra_sw_physics` | `ra_sw_physics` | -1 | 0 (off), 1 (WRF Dudhia), 4 (RTE+RRTMGP), 90 (analytic proxy) | per-domain, resolved by the same rule. Restating one engine on all three keys (`ra_physics = 4` beside `4`/`4`) is one selection written twice and resolves to that pair. A nonzero `ra_physics` beside a split pair naming a DIFFERENT engine is refused as a contradiction rather than resolved: the run and its receipts would name different radiation and nothing in the configuration says which was meant. Keep the pair and set `ra_physics = 0`, or drop both to `-1` and keep `ra_physics` |
| `radt` / `radt_minutes` | `radt` | 0.0 / 12.0 | minutes; 0 = every step | per-domain; WRF `radt = 0` imports as `radt_minutes = 0.0` |
| `bldt` | `bldt` | 0.0 | minutes; 0 = every step | surface layer + LSM + PBL interval |
| `cudt_minutes` | `cudt` | 5.0 | minutes | consumed where `cu_physics = 1` |
| `icloud` | `icloud` | 1 | 0, 1 (Dudhia); fixed 1 with any RRTMG spectrum | |
| `swrad_scat` | `swrad_scat` | 1.0 | >= 0 | Dudhia scattering |
| `no_mp_heating` | `no_mp_heating` | 0 | 0, 1 | disables microphysics latent heating |
| `mp_tend_lim` | `mp_tend_lim` | 10.0 | > 0, K/s | microphysics theta-tendency clamp |
| `morr_rimed_ice` | `morr_rimed_ice` | 1 (hail) | 0, 1 | Morrison dense ice identity |
| `wsm6_hail_opt` | `hail_opt` | 0 (graupel) | 0, 1 | WSM6 rimed-ice identity |
| `ysu_topdown_pblmix` | `ysu_topdown_pblmix` | 1 | 0, 1 | YSU top-down radiation-driven mixing |
| `nwp_diagnostics` | `nwp_diagnostics` (&time_control) | 0 | 0, 1 | per-step UP_HELI_MAX running max (2-5 km updraft helicity, WRF cal_helicity), reset each history frame, restart-carried, trajectory-inert; the other WRF nwp_output maxima are not carried; wizard configs set 1 |
| `isftcflx` | `isftcflx` | 0 | 0, 1, 2 | MM5 sfclay water-point roughness (Garratt/Donelan) |
| `iz0tlnd` | `iz0tlnd` | 0 | 0, 1, 2 | MM5 sfclay land thermal roughness |
| `usemonalb` | `usemonalb` | false | bool | Noah monthly-climatology albedo |
| `rdlai2d` | `rdlai2d` | false | bool | Noah read-in LAI |
| `opt_thcnd` | `opt_thcnd` | 1 | 1, 2 | Noah soil thermal conductivity (Johansen/McCumber-Pielke) |
| `slope_rad` | `slope_rad` | 0 | 0, 1 | per-domain. WRF v4.7.1's slope-dependent surface shortwave: the land surface receives the flux on the local slope (direct beam by slope and aspect, diffuse part unchanged), and SWNORM is written. Needs a longwave, a shortwave and a land-surface scheme, as in WRF. Refused on moving nests and streamed tiles |
| `topo_shading` | `topo_shading` | 0 | 0, 1 | per-domain, with `slope_rad = 1`. WRF's terrain shadowing: a column in a neighbour's shadow gets the diffuse part only |
| `shadlen` | `shadlen` | 25000.0 | > 0, metres | how far the shadow search looks (`[shared]` only) |
| `num_soil_layers` | `num_soil_layers` | 4 | scheme-defined | ArWen *refuses* a count the scheme does not define where WRF silently overwrites it |
| `nest_microphysics_transition` | -- | `same-scheme-only` | + `mp8-to-mp18-mass-diagnosed-v1`, `mp-edge-mass-diagnosed-v1` | ArWen-only, one-way nest MP edges. Left at the default, a mixed edge between two ported schemes resolves to the closure that pair takes (`mp8-to-mp18-mass-diagnosed-v1` for Thompson over NSSL-2, the matrix id for every other pair) and the coupler receipt records the requested and the effective policy; naming the pair's own id pins it, and naming the other mixed id is refused. An `mp_physics = 28` child entering from another scheme is seeded with WRF's own non-aerosol-aware droplet number and aerosol floors, named in the receipt |

The spelling is not part of the selection anywhere, including where a
configuration is checked against a named physics profile. A profile
pins the split pair; a configuration that reached the same two engines
through `ra_physics` matches it, which is what makes `gpuwm
import-namelist` output runnable under the profile its namelist named
(the importer emits the aggregate for a coupled pair, and for radiation
off).

Both radiation selectors are per domain, and so are the six keys that
travel with them (`ra_physics`, `ra_rrtmg_variant`,
`wrf_rrtmg_compatibility`, `o3input`, `use_mp_re`, `swrad_scat`). Every
domain builds its own radiation driver, and spectrum composition, CAM
ozone parent transport and shared workspace sizing all resolve per
domain, so a parent running RTE+RRTMGP on both streams can carry a child
running the legacy RRTMG longwave against Dudhia shortwave -- that exact
tree is loaded, resolved and round-tripped through the rendered
experiment document in `tests/test_cam_ozone.py`. Requiring the streams
to match tree-wide would refuse a configuration the engine runs. A
domain that states neither key still takes the `[shared]` value, so no
experiment written before the split moves. Which VALUES each stream
implements, and how far each is verified, is the radiation section of
[PHYSICS.md](PHYSICS.md).

Scheme selectors (`mp_physics`, `bl_pbl_physics`, `ra_lw_physics`,
`ra_sw_physics`, `sf_sfclay_physics`, `sf_surface_physics`,
`cu_physics`) and their
allowed values are the subject of [PHYSICS.md](PHYSICS.md). Selectable
in the TOML schema is deliberately wider than runnable: readiness is
owned by `gpuwm/physics_compat.py`, which fails closed with a complete
port receipt, and the importer's runnable sets are narrower still.

### `[perturbation]` -- initial-state theta bubbles (ArWen-only)

Adjusted initial conditions for real-data experiments: one or more
warm bubbles added ONCE to the initial potential temperature after the
base real-data state is final, WRF's `em_quarter_ss` cosine-squared
shape evaluated in geographic coordinates. No WRF namelist can express
this block (stock WRF has warm bubbles only in its idealized
initializers), so `import-namelist` never emits it. Applied per domain
inside each domain's own `initialize_real` -- real-data nest init
re-ingests the source analysis per domain, so a bubble inside a nest's
footprint reaches the nest through the nest's own init, not through
the parent's state. Domains with a delayed start take no fresh bubble
(they initialize from the analysis at activation time). Absent block =
byte-identical prepared state and an unchanged experiment fingerprint.

```toml
[[perturbation.bubbles]]
center_lat = 38.5        # degrees
center_lon = -99.5
center_height_m = 1500.0 # bubble center, metres AGL
radius_km = 10.0         # horizontal radius
depth_m = 1500.0         # vertical HALF-depth
amplitude_k = 2.5        # peak theta perturbation, K (above 10 K warns)
rh_preserve = false      # optional: adjust qv so RH survives the theta change
```

The block is honored on two routes: `gpuwm run` / `gpuwm ingest`
(applied inside each domain's `initialize_real`, geopotential
rebalanced) and the prepared domain-tree forecast runner (`gpuwm sim`
or `gpuwm go` on a tree; applied to the restored sealed states, then
the geopotential is rebalanced at the held pressure as WRF's
`em_quarter_ss` does -- the preparation stays the pure analysis, so a
bubble-on arm and its control can share one preparation).
Preparing a domain tree takes the block on every source: GFS, the
mapped sources (HRRR pressure levels among them), ERA5 and native HRRR
all leave their arrays unperturbed and record the bubbles in the
preparation receipt (`proof.json`, or `receipt.json` for a native HRRR
tree) as an `initial_perturbation` entry with status
`DEFERRED_TO_FORECAST_INITIALIZATION`, the same document whichever
source wrote it. The companion stock-WRF file set cannot carry a
deferred bubble: a tree preparation that asks for it as optional (the
default) records that export as refused, and one that requires it is
refused before any source is decoded.
Refusals (never silent): unknown keys; nonpositive
`radius_km`/`depth_m`/`amplitude_k`; a center outside the
coarse domain; an enabled bubble that touches zero cells on a domain
that contains its center; a bubble that heats a layer above the top of
the radiation's temperature table (355 K under RTE+RRTMGP, which stops
the forecast at step 1 on such a layer); `rh_preserve` building more
than 0.09 kg/kg of water vapour (3 km forecasts that built 0.092 kg/kg
and more went non-finite within six minutes; 0.084 kg/kg ran). Routes
that do not thread the block refuse it by name rather than dropping
it: the single-domain prepared runner applies no bubble, so a
single-domain preparation on any source refuses the block before it
decodes a source file, as do `gpuwm go`, `gpuwm check` and `run-plan`
on a single-domain prepared configuration. An `amplitude_k` above
10 K runs with a warning that names it beside WRF's 3 K idealized
bubble, and the warning is recorded in the
receipt. What was actually written
-- per-domain cells touched, max theta delta, qv adjustment under
`rh_preserve` -- lands in `initial-perturbation.json` in the run
directory (the tree runner's `evidence/`), written before integration
starts.

### `[spectral_numerics]` -- Level-2 regional spectral operators (ArWen-only)

Optional, default off, and off is bitwise inert.  Scale-selective
spectral hyperdiffusion of chosen scalars and divergent-mode wind
damping, fired once per completed slow large step in `off` / `shadow`
(receipts only, state-bitwise inert) / `apply` (opt-in) modes, with
hash-bound step receipts feeding the run capsule.  Full schema, the
boundary and streamed-domain refusals, and the CLI evidence door
(`gpuwm spectral-op`) are documented in
[LEVEL2_SPECTRAL_NUMERICS.md](LEVEL2_SPECTRAL_NUMERICS.md).  A present
table binds the restart identity; an absent table leaves every existing
fingerprint untouched.

### `[ingest]` -- soil-state ingest policy (ArWen-only)

One key, and the only reason to write it is to turn a correctness
remedy OFF for a stock-WRF comparison.

```toml
[ingest]
soil_texture_downscale = false   # default: true
```

A forcing model delivers its soil state on its own mesh -- 0.25 degrees
for GFS and ERA5 -- and stock WRF uses the interpolated result as-is, so
`SMOIS` holds no information below the source spacing and prints the
forcing grid into the 2 m dewpoint as rectangular boxes over land.
ArWen carries soil moisture across the resolution change as Noah's own
degree-of-saturation ratio and reconstitutes it against the target
grid's own 30 arc-second soil texture, and anchors the deep `TSLB`
layers on the sub-source-cell part of `TMN` with WRF's own
linear-in-depth weight. Both are ON by default and apply on every
route, nests included.

`soil_texture_downscale = false` restores the previous, WRF-identical
behaviour byte for byte. Every run records the soil-state source
resolution -- and whether the reconstitution ran -- under
`soil_texture_downscale` in `proof.json`, and preparation prints an
advisory when the model resolves more than five cells across one source
cell. Refusals (never silent): an unknown key in the table, or a
non-boolean value. Full rationale, the WRF references, and the
measurements in `docs/soil-texture-downscaling.md`.

## Identity-pinned option families

These are real WRF namelist keys that ArWen carries as configuration
fields but admits at exactly one value each -- the value the port was
validated at against unmodified WRF Fortran. `validate_run_config`
refuses anything else before a run starts, and the importer records
each supplied key as *fixed by ArWen* (or refuses a non-identity
value). Three Noah-MP keys are the exception, because they reach no
transcribed code at all: `opt_pedo`, `noahmp_output` and
`noahmp_acc_dt` run at any value of their own type, warn once, and are
still recorded as fixed at the pin the run used:

- **MYNN** (`&physics`): `bl_mynn_closure 2.6`, `bl_mynn_cloudpdf 2`,
  `bl_mynn_mixlength 1`, `bl_mynn_edmf 1`, `bl_mynn_edmf_mom 1`,
  `bl_mynn_edmf_tke 0`, `bl_mynn_mixscalars 0`, `bl_mynn_cloudmix 1`,
  `bl_mynn_mixqt 0`, `bl_mynn_output 0`, `bl_mynn_tkeadvect false`,
  `icloud_bl 1` (`MYNN_PBL_OPTION_IDENTITY`, `gpuwm/config.py`).
- **Noah-MP** (`&noah_mp`): `dveg 4`, `opt_crs 1`, `opt_btr 1`,
  `opt_run 3`, `opt_sfc 1`, `opt_frz 1`, `opt_inf 1`, `opt_rad 3`,
  `opt_alb 2`, `opt_snf 1`, `opt_tbot 2`, `opt_stc 1`, `opt_gla 1`,
  `opt_rsf 1`, `opt_soil 1`, `opt_pedo 1`, `opt_crop 0`, `opt_irr 0`,
  `opt_irrm 0`, `opt_infdv 0`, `opt_tdrn 0`, `soiltstep 0`,
  `noahmp_output 1`, `noahmp_acc_dt 0` -- each with its evidence line
  in `NOAHMP_OPTION_IDENTITY_EVIDENCE`.
- **RUC** (`&physics`/`&stoch`): `mosaic_lu 0`, `mosaic_soil 0`,
  `flag_sm_adj 0`, `spp_lsm 0`.
- **NSSL 2-moment parameters** (`&physics`, `mp_physics = 18`): the
  port runs at the WRF v4.6.1 Registry defaults pinned by
  `gpuwm/core/nssl2_contract.py` (`nssl_cccn 0.5e9`, `nssl_alphah 0`,
  `nssl_alphahl 1`, `nssl_cnoh 4e5`, ... `nssl_3moment 0`); tunable
  NSSL parameters are not yet plumbed.
- **Thompson aerosol-aware** (`&physics`/`&domains`,
  `mp_physics = 28`): native `met_em` preparation accepts an analyzed
  `QNWFA`/`QNIFA` pair. The monthly WIF dataset is selected by imported
  `use_aero_icbc .true.`, `wif_input_opt 1`, `num_wif_levels 30`;
  `auto` uses the complete analyzed pair when no source was explicitly
  requested. Receipts identify the source and any ignored analyzed pair.
  Fire emissions and black carbon remain unsupported. The synthetic
  fallback uses `use_aero_icbc .false.`, `use_rap_aero_icbc .false.`,
  `wif_input_opt 0`,
  `num_wif_levels` unused, `qna_update 0`, `wif_fire_emit .false.`,
  `wif_fire_inj` unused, `dust_emis 0`, `grav_settling 0`,
  `scalar_pblmix 0`. WRF *derives* `aer_init_opt` and
  `aer_fire_emit_opt` from the first four of those
  (`Registry.EM_COMMON:2656`, `:2658` are declared `derived`, not
  `namelist`), so they are not ArWen settings and are not exposed. The
  activation table `CCN_ACTIVATE.BIN` is a launch prerequisite ArWen
  ships, byte-validated on every load -- see [PHYSICS.md](PHYSICS.md).

  "No aerosol IC/BC" means no *ingest lane*, not an empty aerosol field:
  a cold-start mp=28 domain is initialised with WRF's own fallback, the
  synthetic CCN/IN profile `thompson_init` fills, installed once per
  domain by `gpuwm/core/physics.py::initialize_physics`. What is missing
  is any way to *supply* an aerosol field of your own, and any aerosol at
  a specified lateral boundary -- which matters, because that boundary
  policy sweeps the initial field out of the domain in `L/U`. Both are
  quantified in [PHYSICS.md](PHYSICS.md).

### The one way to supply your own mp=28 aerosol: the WIF climatology

The paragraph above describes the **default**. There is now exactly one
implemented alternative, and it is the same one WRF has: the GLOBAL
monthly water/ice-friendly aerosol climatology
(`QNWFA_QNIFA_SIGMA_MONTHLY.dat`) that WPS routes through metgrid's
`constants_name`. Two keys in `[run]` select it, and nothing else in a
config changes:

```toml
[run]
aer_init_opt  = 1     # WRF's use_aero_icbc = .true. (real.exe derives it)
wif_input_opt = 1     # the use_wif_input package
```

With both set, `initialize_real` reads the dataset, interpolates it to
your grid and your case's valid date exactly as `real.exe` does â€”
metgrid `four_pt` horizontally, `monthly_interp_to_date` temporally
(integer julian-day weighting between month middles), `vert_interp`
linear in `log(p)` onto the dry eta pressure â€” and writes `QNWFA`,
`QNIFA`, `QNWFA2D` and `QNIFA2D`. The nonzero fields are themselves the
signal WRF's `MAXVAL` presence tests
(`phys/module_mp_thompson.F:493/:531`) read to **skip** the synthetic
profile, so nothing downstream needs a new flag.

Nothing else has to be supplied. The grid latitudes/longitudes and the
valid date are derived from the runner's own geometry and the snapshot's
`valid_time`; every real front door (`gpuwm run`, the prepared
single-domain and two-domain runners, the direct HRRR/GFS/ERA5/mapped
routes, nests) passes them for you. There is no per-source branch: the
climatology does not come from the driving model, so the derivation is
identical for every input source.

**Staging the dataset.** It is a fixed 225,443,520-byte external file
that never changes. ArWen does **not** redistribute it â€” that is over
PyPI's 100 MB per-file cap and GitHub's 100 MiB blob limit, the same
reason `freezeH2O.dat` is externalized â€” so it joins the same command:

```
gpuwm fetch-tables --wif --wif-only --from /path/to/WRF/run
```

stages it into `~/.gpuwm/wif` after verifying exact size and SHA-256
`2f828eabd96a45f3872390f901240ea2259a1e9a629247010f42ce7a31cc46be`;
`gpuwm fetch-tables --wif` alone downloads it from the release asset
base under the same verification. Any WRF tree that has ever run
`mp_physics = 28` from climatology already has the file. Precedence,
highest first: `wif_climatology_path` in the config,
`GPUWM_WIF_CLIMATOLOGY` (full path), `GPUWM_WIF_DATA_ROOT` (directory),
then `~/.gpuwm/wif`. Leave `wif_climatology_path` unset and the staged
copy is found. A path you *name* and that does not exist is an error,
never a silent fall-through to the staged copy.

**If the dataset is absent the run refuses at config load**, by name,
with the acquisition route in the message. It never falls back to the
synthetic profile: that is a different, valid configuration
(`aer_init_opt = 0`) and a silent demotion would leave a run whose
receipt says "monthly climatology" and whose aerosol came from an
analytic curve. `wif_input_opt = 2` stays refused by name â€” it
additionally allocates the black-carbon scalar `qnbca`
(`Registry/registry.new3d_wif:82`), which has no consumer here.

**Importing a WRF namelist that uses it.** The key triple
`&physics use_aero_icbc = .true.` with `&domains wif_input_opt = 1` and
`num_wif_levels = 30` imports; `gpuwm import-namelist` emits the two
`[run]` keys
above and prints which dataset the run will read. Half the triple is
refused, naming which half.

**Demo.** `configs/demos/mp28_wif_climatology.toml` is the two-line
delta against an ordinary 3 km real-data configuration, with the staging
command and the receipt in its header.

**The receipt.** `RealInitResult.aerosol_initialization` carries a
`wif_climatology` entry â€” dataset path, the two month indices and their
weights, the operators used â€” and sets `awaiting_profile_fill` to
`false`.

## Fixed by ArWen (WRF has a knob; ArWen has one implemented value)

Each of these keys has exactly one implemented value, and it is already
what your run gets: in the TOML the key does not exist at all, so there
is nothing for you to set. Import a namelist that names the pinned
value and it passes; name anything else and the importer says which key
and which value, rather than flipping it behind you. The two rows to
read before importing are `use_theta_m` and `mix_full_fields`, whose
pins differ from what WRF assumes for an omitted key.

| WRF key | fixed at | where it is pinned |
|---|---|---|
| `rk_ord` | 3 | RK3 stage table, `gpuwm/core/dycore.py` |
| `h_mom_adv_order` | 5 | WRF flux5 stencil hardcoded, `gpuwm/core/kernels/advection.cu` |
| `v_mom_adv_order`, `v_sca_adv_order` | 3 | WRF flux3 stencil, same kernel |
| `momentum_adv_opt` | 1 | standard (non-PD) momentum advection |
| `non_hydrostatic` | .true. | nonhydrostatic-only |
| `use_theta_m` | 0 | the engine evolves dry theta and has no moist-theta branch; every import door (`import-namelist`, `run --wrfinput` and `run --met-em`) admits a namelist's `use_theta_m = 1` (WRF's omitted default) as a DECLARED DIVERGENCE announced at the terminal and recorded under "Physics substitutions" in the import receipt: the initial and boundary state is recovered exactly (moist wrfbdy THM/QV/MU converted at each forcing time; metgrid TT is physical temperature; native initialization builds dry theta from physical temperature) but the integration is dry theta, so it differs from a `use_theta_m = 1` WRF run |
| `scalar_adv_opt` | 1 | must match `moist_adv_opt` |
| `isfflx` | 1 | surface fluxes on |
| `sf_lake_physics`, `mosaic_lu/soil` | 0 | not implemented |
| `sf_urban_physics` | 0 (default) | 1 single-layer urban canopy, 2 BEP, 3 BEP+BEM; requires Noah or Noah-MP; mosaic admits only option 1 |
| `sf_surface_mosaic`, `mosaic_cat` | 0, 3 | Noah land-use tiles; enabled only at 1, positive tile count; requires LANDUSEF; urban option 1 runs per tile; urban options 2 and 3 are refused as in WRF |
| `mosaic_urban_canopy` | "dominant" | gpuwm key, per domain: where mosaic runs urban option 1. "dominant" is WRF's rule (only cells whose dominant category is urban); "every_tile" also runs the town tiles of mostly rural cells at their own land-use weights, sharing the URBPARM urban fraction of the largest urban tile's type; needs `sf_surface_mosaic = 1` and `sf_urban_physics = 1` |
| `swint_opt` | 0 | no SW interpolation between radt calls |
| `use_mp_re` | 1 | microphysics effective radii reach radiation per WRF's scheme table |
| `o3input` | 2 | CAM climatological ozone (RRTMG spectra) |
| `ghg_input` | 0 | analytic year-formula trace gases (no CAMtr reader) |
| `aer_opt` | 0 | no radiation aerosol input |
| `cldovrlp` / `idcor` | 2 / 0 | McICA maximum-random overlap, constant decorrelation |
| `gwd_opt` | 0 | no gravity-wave drag |
| `shcu_physics` | 0 | no shallow cumulus |
| `cu_rad_feedback` | .false. | KF cloud fraction does not feed radiation |
| `kf_edrates` | 0 | no KF rate diagnostics |
| `sst_update`, `sst_skin`, `tmn_update` | 0 | single-analysis case runs |
| `use_aero_icbc`, `use_rap_aero_icbc` | .false. | synthetic fallback identity; imported `use_aero_icbc .true.` with `wif_input_opt 1` selects the monthly WIF dataset. A generic GOCART reader and the RAP source are unavailable |
| `wif_input_opt` | 0 | synthetic fallback identity; the imported monthly WIF route accepts value 1 with `num_wif_levels = 30`. Value 2 requires unimplemented black carbon. At 0, `num_wif_levels` is inert. **WRF's `real.exe` FATALs `mp_physics = 28` at this value** (`dyn_em/module_initialize_real.F:2734-2736`) while ArWen runs it, taking WRF's own internal fallback â€” the synthetic CCN/IN profile `thompson_init` installs â€” as the aerosol initial condition. So an ArWen mp=28 run and a WIF-initialised WRF mp=28 run are **not** directly comparable; see D9a/D9b in [PROVENANCE.md](../../PROVENANCE.md) |
| `qna_update` | 0 | no auxiliary `wrfqnainp` input stream |
| `wif_fire_emit`, `wif_fire_inj` | .false. / unused | no biomass-burning aerosol emission inventory |
| `dust_emis` | 0 | no non-chem dust source; `nifa2d` stays exactly zero, matching `thompson_init` |
| `grav_settling` | 0 | fog gravitational settling not ported. WRF *silently* forces 0 on every `mp_physics = 28` domain (`share/module_check_a_mundo.F:2459-2474`); ArWen refuses a nonzero value instead |
| `scalar_pblmix` | 0 | no 4-D scalar PBL mixing path. WRF forces 1 under `mp_physics = 28` **only with** `use_aero_icbc`/`use_rap_aero_icbc` (`:2477-2495`), which ArWen refuses, and forces 0 again under MYNN with `bl_mynn_mixscalars = 1` (`:2497-2511`); at ArWen's identity WRF's own value is 0 too |
| `interp_method_type` | 2 | SINT nest interpolation only |
| `input_from_file` | .true. | per-domain real init is the T branch |
| every `&stoch` selector | 0 | no stochastic physics (seed keys drop as inert) |

Init-side constants frozen at the WRF reference behavior (no namelist
counterpart is honored): base-state `iso_temp = 200 K` and
`base_lapse = 50 K` (`gpuwm/ingest/real.py`), `p00 = 1e5 Pa`,
turbulent Prandtl number 1/3, Smagorinsky K cap `10*sqrt(dx*dy)`, and
the whole real.exe vertical-interpolation policy set
(`lagrange_order 2`, `extrap_type 2`, `t_extrap_type 2`,
`zap_close_levels 500 Pa`, `force_sfc_in_vinterp 1`,
`use_levels_below_ground/use_surface .true.`), pinned by the
preprocessing provenance contract. `sfcp_to_sfcp` is the one
real-init policy that is a knob (`[case_data]`, and `false` is
fail-loud unimplemented).

## Sub-grid terrain drag

`topo_wind` accepts 0, 1 and 2; `gwd_opt` accepts 0, 1 and 3. Both default
to 0. They are selected in `[shared]` or on a domain and imported from
WRF namelists. See [Terrain drag](TERRAIN-DRAG.md) for the schemes, required
WPS geography and preparation command.

## Terrain smoothing (`[[domain]] static`)

WPS smooths `HGT_M` with whatever its `GEOGRID.TBL` names; the stock table
runs one `smth-desmth_special` pass, and so does ArWen unless a domain asks
otherwise:

```toml
[[domain]]
grid_id = 2
# ...
static = { smooth_option = "none" }                       # sampled terrain, unsmoothed
# static = { smooth_option = "1-2-1", smooth_passes = 3 }  # or any GEOGRID.TBL smoother
```

`smooth_option` is `smth-desmth_special` (the default), `smth-desmth`,
`1-2-1` or `none`; `smooth_passes` is a whole number of passes, 1 or more,
default 1. Each domain chooses its own. The default keeps ArWen's own
double-precision smoother, within 1.5 mm of WPS on the Alpine domains measured; every
other setting is WPS's single-precision arithmetic, identical bit for bit
to `geogrid.exe` given the same unsmoothed terrain.

`smooth_precision = "wps-float32"` is the option that runs the default
smoother in WPS's single-precision arithmetic as well, so the domain's
terrain is `geogrid.exe`'s bit for bit; `"float64"`, ArWen's own smoother,
stays the default. On its own the key keeps the default smoother:

```toml
static = { smooth_precision = "wps-float32" }   # WPS's default, WPS's arithmetic
```

Only the default smoother has the choice. Every other smoother has WPS's
arithmetic alone, so `"float64"` beside one is refused, as is any precision
beside `none`.

Doors: `gpuwm domain --terrain-smoothing none,1-2-1:3` (one item per
domain, the last repeating), and `gpuwm import-namelist`, which reads HGT_M's
setting from `--geogrid-tbl PATH`, else the `opt_geogrid_tbl_path` in
`namelist.wps`, else `./geogrid/GEOGRID.TBL` beside it, and writes it on
every domain as WPS applies one table to all.

The WPS-exact default through the doors: `--terrain-smoothing-precision
wps-float32` on `gpuwm domain` or `gpuwm import-namelist` sets it on every
domain whose smoother is the default. On import, a `GEOGRID.TBL` whose
HGT_M entry carries `smooth_precision = wps-float32` selects it too:

```
name = HGT_M
        ...
        smooth_option = smth-desmth_special; smooth_passes=1
        smooth_precision = wps-float32
```

That key is ArWen's, not WPS's: `geogrid.exe` logs it as an unrecognized
option and smooths in single precision as it always does, so one table
serves both. The flag, when given, wins over the table.

With nests:

* A nest's outer rows are still its parent's terrain. As in WRF's
  `blend_terrain`, the first `spec_bdy_width` rows (default 5) take the
  parent's interpolated terrain and the next `blend_width` rows (default 5)
  ramp to the nest's own, so an unsmoothed nest shows its own valleys from
  the 11th row in. Keep the valleys that matter at least that far inside.
* The parent keeps its own terrain under the nest. ArWen's two-way feedback
  carries the prognostic fields only (WRF also feeds the nest's terrain
  back), so each domain's setting governs that domain alone.
* `[experiment] smooth_option` is WRF's `&domains smooth_option`, the
  two-way feedback smoother, and has nothing to do with this key.
* WRF's `smooth_cg_topo` (d01's outer rows blended toward the driving
  model's terrain) is not implemented; `import-namelist` refuses it.
* A moving domain (a `follow` table or `[relocation]`, and every domain
  under it) takes `none`, `1-2-1` up to 3 passes, or a smoother-desmoother
  with 1 pass. More passes carry the smoother past the 3 cells sampled
  beyond the domain edge, the edge rows then depend on where the domain
  sits, and the first move would be refused; the configuration is refused
  at load instead.

Unsmoothed terrain is steeper (a 1 km Alpine domain went from a steepest
slope of 0.85 to 1.07), and the terrain clock (Clock, above) reads it before
the forecast starts. Past the slopes its map was measured on, the clock runs
the configured step and warns that the run may stop: under a 44 m/s crest
wind that domain's smoothed terrain was held to 3.5 s steps, while the
unsmoothed terrain ran 8 s steps (and finished its 3 hours). Watch for that
warning, and set `max_time_step` yourself if the run stops.

Roots built from `WPS_GEOG` on the ERA5, GFS, native HRRR and mapped-source (ICON, ECMWF and the rest) routes carry the setting, and so does every nest.
A root loaded from a prebuilt static cache that does not record its smoothing refuses a non-default setting before integrating default terrain.

## Not implemented (refused or dropped with a reason)

Moving nests,
vertical nest refinement, FDDA nudging
(active `grid_fdda`/`grid_sfdda`/`obs_nudge_opt` refuse; inert keys
drop), stochastic physics (SPP/SPPT/SKEBS), `mp_zero_out` (documented
absent -- ArWen relies on PD transport), urban/lake/seaice physics,
auxiliary I/O streams (`auxhist*`/`auxinput*`, `iofields_filename`;
ArWen writes one fixed wrfout frame per file per domain -- fields are
not namelist-selectable), quilt servers, and WRF process/tile
decomposition (`numtiles`, `nproc_x/y` -- dropped; GPU decomposition
is internal). A namelist key outside every table above is a hard
`unmapped key(s)` error: the importer never drops a setting silently.

## Where the values come from

- Schema + invariants: `gpuwm/config.py` (`RunConfig`,
  `validate_run_config`), `gpuwm/experiment.py` (experiment tables).
- Importer: `gpuwm/namelist_import.py`; every decision lands in the
  three-section substitution report.
- Reach-tests: `tests/test_namelist_import.py` (knob-parity battery)
  proves each translated knob lands on the consuming `RunConfig`
  field by driving a distinctive value through the import and
  asserting it reaches the consuming kernel/module -- the tests
  themselves name the per-kernel consumption sites.
