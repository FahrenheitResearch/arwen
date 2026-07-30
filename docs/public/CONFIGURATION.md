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
| `feedback` | `feedback` | 0 | **0 only** | one-way nesting only; a two-way namelist (explicit or by the Registry default 1) is refused, never quietly imported one-way |
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

### Dynamics

Shared across domains unless marked per-domain. Every key below is a
consumed `RunConfig` field -- the knob-parity battery
(`tests/test_namelist_import.py`) proves each one lands on the
consuming kernel/module rather than being decorative -- and every one
is importable from a WRF namelist.

| TOML key | WRF equivalent | default | allowed | note |
|---|---|---|---|---|
| `time_step_sound` | `time_step_sound` | 4 | even, > 0 | WRF 0 = auto imports as 4, recorded |
| `epssm` | `epssm` | 0.1 | per-domain | acoustic off-centering; scalar namelist assignment changes d01 only (Registry tail keeps 0.1), preserved per-domain |
| `smdiv` | `smdiv` | 0.1 | finite | 3-D divergence damping |
| `emdiv` | `emdiv` | 0.0 (ArWen legacy) | finite | WRF Registry default 0.01 is emitted explicitly on import |
| `km_opt` | `km_opt` | 1 | 1 (constant K), 4 (2-D Smagorinsky) | `diff_opt=2` form implied; WRF -1 must-set honored: omission refuses |
| `c_s` | `c_s` | 0.25 | > 0 | Smagorinsky constant (smag2d kernel) |
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
| `nest_microphysics_transition` | -- | `same-scheme-only` | + `mp8-to-mp18-mass-diagnosed-v1` | ArWen-only, one-way nest mp edges |

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

Two-way nesting (`feedback`), moving nests, adaptive time step,
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
