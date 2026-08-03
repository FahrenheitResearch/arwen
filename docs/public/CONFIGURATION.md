# Configuration knobs (WRF namelist parity)

ArWen's configuration surface is the experiment TOML
(`[experiment]` / `[projection]` / `[shared]` / `[[domain]]` /
`[case_data]`), and `gpuwm import-namelist WPS INPUT` translates a WRF
`namelist.wps` + `namelist.input` pair into it. The contract of that
translation is *never silent*: every namelist key lands in exactly one
of three report sections --

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

## Tweakable knobs

### `[experiment]`

| TOML key | WRF equivalent | default | allowed | note |
|---|---|---|---|---|
| `name` | -- | required | non-empty string | run identity |
| `start_time` | `&time_control start_*` | required | TOML datetime, offset-free | |
| `run_seconds` | `run_days/hours/minutes/seconds` or `end_*` | required | > 0 | |
| `restart_interval_s` | `restart_interval` (minutes in WRF) | required | >= 0; 0 disables | whole multiple of d01 dt |
| `feedback` | `feedback` | 0 | **0 or 1** | 0 is one-way and supported; 1 is the EXPERIMENTAL two-way path (stamped as such in run provenance; one-way consumers refuse a feedback-modified parent). Any other value is refused, and a two-way namelist is never quietly imported one-way |
| `smooth_option` | `smooth_option` | 0 | **0 only** | the parent smoother acts only under two-way feedback |
| `blend_width` | `blend_width` | 5 | >= 0 | terrain blend zone; enters the parent-row clearance rule |
| `spec_bdy_width` | `&bdy_control spec_bdy_width` | 5 | >= spec_zone + relax_zone | |
| `column_chunk` | -- | 3125 | >= 1 | ArWen-only radiation throughput knob; byte-identical across values |

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
| `eta_levels` | `eta_levels` | () | 1.0 -> 0.0 strictly decreasing | required for real runs; **automatic level generation (`auto_levels_opt`, `max_dz`, `dzbot`, `dzstretch_s/u`) is not implemented** -- with explicit `eta_levels` those keys are inert in WRF too and import as dropped |
| `p_top` | `p_top_requested` | 0.0 | >= 0, Pa | Registry default 5000 Pa applies on import when omitted |
| `hybrid_opt` | `hybrid_opt` | 0 (legacy) | 0/1 (sigma), 2 (WRF cubic-B) | importer default 2 (Registry) |
| `etac` | `etac` | 0.2 | [0, 1] | |
| `ztop` | -- | required | > 0 | ArWen-only scaffold height; real runs derive heights from p_top/eta_levels |

### Clock (`[[domain]]`, root)

`time_step` (integer seconds) plus optional `time_step_fract_num` /
`time_step_fract_den` -- WRF's exact rational clock keys. Children
carry no clock: `dt_child = dt_parent / parent_time_step_ratio`
exactly, chained in float32 exactly as WRF chains it.
`history_interval_s` is per-domain and must divide into whole domain
steps. The adaptive time step (`use_adaptive_time_step`) is refused.

Each `[[domain]]` may also carry an offset-free `start_time`. It defaults to
`[experiment].start_time`; d01 must equal that root start. A delayed child is
dormant until its timestamp, then follows the ordinary parent-state nest
initialization path. The timestamp must be an exact boundary of its parent's
step clock and an exact external-forcing seam. Restart headers persist
`STARTED`/`NOT_STARTED` lifecycle state, so resuming before the timestamp does
not initialize the child early.

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

**Which keys a `[[domain]]` table may override.** Exactly these 26,
and no others (`gpuwm/experiment.py`'s `_DOMAIN_RUN_OVERRIDES`):

    cu_physics  cudt_minutes  radt  radt_minutes  bldt
    diff_6th_factor  diff_6th_opt  epssm  spec_exp  mp_physics  moist
    moist_cq  nest_microphysics_transition
    km_opt  bl_pbl_physics  sf_sfclay_physics  isfflx  c_s  c_k
    mix_isotropic  mix_upper_bound  tke_heat_flux
    tke_drag_coefficient  tke_upper_bound
    sase_flux_diag  hmix_k_diag

`sase_flux_diag` and `hmix_k_diag` are output-only diagnostics, and
they are per domain for the same reason: their cost scales with the
grid, so a tree can carry them on the domain whose mixing or subgrid
flux is being read and leave them off the rest.  The turbulence row
(`km_opt` through `tke_upper_bound`) and the PBL/surface selectors are
per domain because that is what makes a PBL parent able to carry a
PBL-off LES child (see `docs/public/LES.md`).

Only `gpuwm domain`'s own emission and hand-written TOML reach some of
them, so the list is stated here rather than left to be discovered. A
`[[domain]]` table carrying any other key is **refused** naming the
key, not accepted and not silently dropped: a config that appeared to
ask for it while the `[shared]` value ran on every nest would be a
wrong answer reported as a success. Put them in `[shared]`.  One
per-domain VALUE is also refused by name: `bl_pbl_physics = 900`
(SASE) is selected run-wide in `[shared]`, never per nest.

| TOML key | WRF equivalent | default | allowed | note |
|---|---|---|---|---|
| `time_step_sound` | `time_step_sound` | 4 | even, > 0 | WRF 0 = auto imports as 4, recorded |
| `epssm` | `epssm` | 0.1 | per-domain | acoustic off-centering; scalar namelist assignment changes d01 only (Registry tail keeps 0.1), preserved per-domain |
| `smdiv` | `smdiv` | 0.1 | finite | 3-D divergence damping |
| `emdiv` | `emdiv` | 0.0 (ArWen legacy) | finite | WRF Registry default 0.01 is emitted explicitly on import |
| `km_opt` | `km_opt` | 1 | 1 (constant K), 2 (1.5-order prognostic TKE), 3 (3-D Smagorinsky), 4 (2-D Smagorinsky), 0 (no operator -- with `bl_pbl_physics = 900`, or with `km_opt_zero_acknowledgement`) | `diff_opt=2` form implied; WRF -1 must-set honored: omission refuses. **2 and 3 are the LES closures and carry extra conditions:** `km_opt=2` is admitted only with `bl_pbl_physics=0`, and is refused on a nest child under a TKE-carrying parent (see `gpuwm/experiment.py`). `km_opt=3` has no nest restriction. See `docs/public/LES.md` |
| `km_opt_zero_acknowledgement` | (none) | `""` | the exact id `no-horizontal-mixing-operator-v1` | admits `km_opt = 0` with a PBL scheme that produces no horizontal mixing of its own -- i.e. a run with NO horizontal mixing operator, WRF's `diff_opt = 0`. Refused by default because that is what a mis-set switch looks like; the acknowledged path is the single-variable research control that varies the closure while holding the mixing at none. A literal id, not a boolean, so no stray `= true` reaches it. Refused where it would acknowledge nothing (`km_opt != 0`, or SASE, which supplies the producer). Not needed with `bl_pbl_physics = 900` |
| `hmix_k_diag` | (none) | false | bool, per domain | publishes the horizontal eddy viscosities the run's own producer used, under that producer's name: `XKMH`/`XKHH` for `km_opt = 4`, `SASE_KMH`/`SASE_KHH` for the SASE closure. Same units (m2 s-1), same mass grid, so the two are directly comparable. A run with no producer publishes neither pair -- an absent variable cannot be misread as a measured zero. Two extra (nz, ny, nx) planes per frame |
| `c_s` | `c_s` | 0.25 | > 0 | Smagorinsky constant (smag2d kernel); per-domain |
| `c_k` | `c_k` | 0.15 | > 0 | km_opt=2 TKE-closure constant, K = c_k sqrt(e) l; the WRF `em_les` reference namelist sets 0.10; per-domain |
| `mix_isotropic` | `mix_isotropic` | 0 | 0, 1 | 0 = anisotropic mixing lengths, 1 = isotropic (dx dy dz)^(1/3); per-domain |
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
| `w_damping` | `w_damping` | 0 | 0, 1 | the `w_crit_cfl` threshold itself is fixed at WRF's Registry default 1.0 |
| `base_temp` | `base_temp` | 290.0 | K | base state; init-time only (see fixed table for `iso_temp`/lapse) |
| `hypsometric_opt` | `hypsometric_opt` | 1 (ArWen legacy) | 1, 2 | WRF Registry default 2 emitted explicitly on import |
| `h_sca_adv_order` | `h_sca_adv_order` | 2 (ArWen legacy) | 2, 5 | **feeds the geopotential equation only**; transported-scalar stencils are fixed 5th/3rd order, so the importer accepts only the Registry default 5 |
| `moist_adv_opt` | `moist_adv_opt` | 1 | 0, 1 in TOML; import pins 1 | PD limiter; `scalar_adv_opt` must match (WRF option 1 on both) |
| `top_lid` | `top_lid` | **true** (ArWen) | bool | WRF Registry default is false (open top); ArWen defaults to the rigid lid after the 2026-07-18 open-top NaN probes -- imports emit the Registry value explicitly, flip back only with a stability receipt |
| `moist_cq` | -- (WRF always applies cq) | **false** (ArWen) | bool | imported WRF experiments pin `true` explicitly |
| `spec_zone`, `relax_zone`, `spec_exp` | `&bdy_control` | 1, 4, 0.0 | | `spec_exp` acts on the root (specified) branch only, exactly as in WRF's `lbc_fcx_gcx`; nonzero on a nested child is refused |

### Physics cadences and scheme knobs

| TOML key | WRF equivalent | default | allowed | note |
|---|---|---|---|---|
| `radt` / `radt_minutes` | `radt` | 0.0 / 12.0 | minutes; 0 = every step | per-domain; WRF `radt = 0` imports as `radt_minutes = 0.0` |
| `bldt` | `bldt` | 0.0 | minutes; 0 = every step | surface layer + LSM + PBL interval |
| `cudt_minutes` | `cudt` | 5.0 | minutes | consumed where `cu_physics = 1` |
| `icloud` | `icloud` | 1 | 0, 1 (Dudhia); fixed 1 under 4/4 radiation | |
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
| `num_soil_layers` | `num_soil_layers` | 4 | scheme-defined | ArWen *refuses* a count the scheme does not define where WRF silently overwrites it |
| `nest_microphysics_transition` | -- | `same-scheme-only` | + `mp8-to-mp18-mass-diagnosed-v1`, `mp-edge-mass-diagnosed-v1` | ArWen-only, one-way nest MP edges; the first id preserves the ratified Thompson→NSSL path and the matrix id selects the other ported mixed edges. **`mp_physics = 28` is excluded from both mixed ids**: an mp=28 domain may only nest under an mp=28 parent, and a mixed edge is refused by name rather than closed with WRF's non-aerosol-aware fallback constants |

Scheme selectors (`mp_physics`, `bl_pbl_physics`, `ra_lw/sw_physics`,
`sf_sfclay_physics`, `sf_surface_physics`, `cu_physics`) and their
allowed values are the subject of [PHYSICS.md](PHYSICS.md). Selectable
in the TOML schema is deliberately wider than runnable: readiness is
owned by `gpuwm/physics_compat.py`, which fails closed with a complete
port receipt, and the importer's runnable sets are narrower still.

## Identity-pinned option families

These are real WRF namelist keys that ArWen carries as configuration
fields but admits at exactly one value each -- the value the port was
validated at against unmodified WRF Fortran. `validate_run_config`
refuses anything else before a run starts, and the importer records
each supplied key as *fixed by ArWen* (or refuses a non-identity
value):

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
  `mp_physics = 28`): the port runs at exactly one aerosol-forcing
  identity -- no aerosol IC/BC, no WIF metgrid stream, no fire
  emissions, no black-carbon species. Concretely `use_aero_icbc
  .false.`, `use_rap_aero_icbc .false.`, `wif_input_opt 0`,
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

## Fixed by ArWen (WRF has a knob; ArWen has one implemented value)

The importer validates these when present and refuses any other value.
In the TOML they do not exist at all.

| WRF key | fixed at | where it is pinned |
|---|---|---|
| `rk_ord` | 3 | RK3 stage table, `gpuwm/core/dycore.py` |
| `h_mom_adv_order` | 5 | WRF flux5 stencil hardcoded, `gpuwm/core/kernels/advection.cu` |
| `v_mom_adv_order`, `v_sca_adv_order` | 3 | WRF flux3 stencil, same kernel |
| `momentum_adv_opt` | 1 | standard (non-PD) momentum advection |
| `diff_opt` | 2 | the only mixing form behind `km_opt` |
| `mix_full_fields` | .true. | full-field mixing only (must be explicit: WRF's omitted default is false) |
| `non_hydrostatic` | .true. | nonhydrostatic-only |
| `use_theta_m` | 0 | dry-theta branch only (WRF's omitted default 1 is refused, not silently flipped) |
| `scalar_adv_opt` | 1 | must match `moist_adv_opt` |
| `w_crit_cfl` | 1.0 | `#define` in `gpuwm/core/kernels/openbc.cu` (Registry default) |
| `isfflx` | 1 | surface fluxes on |
| `sf_urban_physics`, `sf_lake_physics`, `sf_surface_mosaic`, `mosaic_lu/soil` | 0 | not implemented |
| `swint_opt` | 0 | no SW interpolation between radt calls |
| `use_mp_re` | 1 | microphysics effective radii reach radiation per WRF's scheme table |
| `o3input` | 2 | CAM climatological ozone (4/4 radiation) |
| `ghg_input` | 0 | analytic year-formula trace gases (no CAMtr reader) |
| `aer_opt` | 0 | no radiation aerosol input |
| `cldovrlp` / `idcor` | 2 / 0 | McICA maximum-random overlap, constant decorrelation |
| `gwd_opt` | 0 | no gravity-wave drag |
| `shcu_physics` | 0 | no shallow cumulus |
| `topo_shading`, `slope_rad` | 0 | no terrain radiation geometry |
| `cu_rad_feedback` | .false. | KF cloud fraction does not feed radiation |
| `kf_edrates` | 0 | no KF rate diagnostics |
| `sst_update`, `sst_skin`, `tmn_update` | 0 | single-analysis case runs |
| `use_aero_icbc`, `use_rap_aero_icbc` | .false. | no GOCART climatological aerosol IC/BC reader (`mp_physics = 28` only) |
| `wif_input_opt` | 0 | no WIF metgrid aerosol stream; `num_wif_levels` is inert with it. **WRF's `real.exe` FATALs `mp_physics = 28` at this value** (`dyn_em/module_initialize_real.F:2734-2736`) while ArWen runs it, taking WRF's own internal fallback — the synthetic CCN/IN profile `thompson_init` installs — as the aerosol initial condition. So an ArWen mp=28 run and a WIF-initialised WRF mp=28 run are **not** directly comparable; see D9a/D9b in [PROVENANCE.md](../../PROVENANCE.md) |
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

## Not implemented (refused or dropped with a reason)

Moving nests, adaptive time step,
vertical nest refinement, automatic eta generation, FDDA nudging
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
