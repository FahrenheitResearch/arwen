from __future__ import annotations
import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from gpuwm.physics_compat import (
    RRTMG_VARIANT_LEGACY,
    RRTMG_VARIANT_RTE_RRTMGP,
    WRF_RRTMG_COMPATIBILITY_TOKENS,
    WRF_RRTMG_LEGACY,
    WRF_RRTMG_SUBSTITUTION_TOKENS,
    require_ready_wrf_physics,
    validate_resolved_physics_vertical_levels,
)
from collections.abc import Mapping as _Mapping
from importlib import import_module as _import_module

# The closure's two cross-layer constants.  Deliberately NOT
# gpuwm.verify.sase_ref: that tree is developer verification and the
# standalone CPU preprocessing distribution omits it.
from gpuwm.core import sase_limits as _sase_limits

DEFAULT_COLUMN_CHUNK = 3125


@dataclass(frozen=True)
class RunConfig:
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    ztop: float
    dt: float
    run_seconds: float
    # WRF's model-clock timestep.  Normally identical to ``dt``; the
    # real74 compatibility integrator sets this to 60 s while advancing
    # eight uniform 7.5 s dynamics steps.  Clock-defined coefficients such
    # as Davies relaxation and sixth-order diffusion use this value.
    clock_dt: float = 0.0
    p_surf: float = 1.0e5
    time_step_sound: int = 4
    epssm: float = 0.1
    smdiv: float = 0.1
    khdif: float = 0.0
    kvdif: float = 0.0
    damp_opt: int = 0
    zdamp: float = 5000.0
    dampcoef: float = 0.2
    output_interval_s: float = 300.0
    case: str = ""
    # --- Phase 2 options (defaults preserve Phase 1 behavior exactly) ---
    hybrid_opt: int = 0          # 0/1 = B(eta)=eta (Phase 1); 2 = WRF cubic-B hybrid
    etac: float = 0.2
    moist: bool = False
    # 0 off, 1 Kessler, 6 WSM6, 8 Thompson, 10 Morrison, 18 NSSL,
    # 28 Thompson aerosol-aware (Registry/Registry.EM_COMMON:3036)
    mp_physics: int = 0
    moist_adv_opt: int = 1       # PD limiter on when moist
    # WRF no_mp_heating (Registry.EM_COMMON:2630, default 0): 1 disables
    # microphysics latent heating entirely -- moist_physics_finish_em then
    # leaves theta untouched and zeroes h_diabatic
    # (module_big_step_utilities_em.F:5770-5782).
    no_mp_heating: int = 0
    # WRF mp_tend_lim (Registry.EM_COMMON:2642, default 10. K/s): clamp on
    # the per-step microphysics theta increment, |mpten| <= mp_tend_lim*dt
    # (module_big_step_utilities_em.F:5706-5707).
    mp_tend_lim: float = 10.0
    diff_6th_opt: int = 0
    diff_6th_factor: float = 0.12
    # WRF diff_6th_slopeopt / diff_6th_thresh (Registry.EM_COMMON:2858-2859,
    # defaults 0 / 0.10; the reference namelist sets slopeopt=1): >= 1
    # tapers every 6th-order face flux by slopedamp = MAX(1 -
    # dzmax/(thresh*9.81*dx), 0) with dzmax the BASE-state (phb) face
    # geopotential jump at the field's own level, msf-scaled per face
    # (sixth_order_diffusion, module_big_step_utilities_em.F:6487-6501 x /
    # :6569-6583 y).  Default 0 keeps the frozen untapered operator
    # bitwise; flat (1-D phb) states are untapered either way.
    diff_6th_slopeopt: int = 0
    diff_6th_thresh: float = 0.10
    km_opt: int = 1
    c_s: float = 0.25
    w_damping: int = 0
    open_x: bool = False         # False = periodic (Phase 1 default)
    open_y: bool = False
    # --- Phase 3 (Task 8): WRF real-data base state and lateral BCs. ---
    base_temp: float = 290.0
    specified: bool = False
    spec_bdy_width: int = 5
    spec_zone: int = 1
    relax_zone: int = 4
    spec_exp: float = 0.0
    # WRF external-mode divergence damping (module_small_step_em.F mudf;
    # namelist emdiv, WRF Registry default 0.01).  gpuwm defaults to 0.0 so
    # the frozen Phase 1 acoustic path is bitwise unchanged; open-boundary
    # cases set the WRF value (em_quarter_ss: 0.01) -- the filter damps the
    # column-integrated (external) mode that open boundaries excite.
    emdiv: float = 0.0
    # WRF h_sca_adv_order for the geopotential equation's horizontal
    # advection (rhs_ph, module_big_step_utilities_em.F:1435; Registry
    # default 5).  gpuwm defaults to 2 so every frozen flat regression
    # stays bitwise on the original 2nd-order path; the real74 production
    # configs set the reference value 5 (the <=6-branch 1/60-weighted
    # centered stencil with WRF's spec-zone boundary narrowing).  Order 5
    # is wired for periodic and specified lateral boundaries only.
    h_sca_adv_order: int = 2
    terrain_opt: int = 0         # 0 = flat; 1 = bell hill
    hill_height: float = 100.0
    hill_halfwidth: float = 10000.0
    # --- Phase 3 (Task 3): map projection.  0 = idealized/none (map factors
    # 1, Coriolis parameters 0 — the bitwise Phase 2 state); nonzero uses
    # the WRF convention: 1 = Lambert conformal, 2 = polar stereographic,
    # 3 = Mercator (grids built by gpuwm.static.projection).  The dycore
    # consumes the projection only through the DomainState fields
    # msft/msfu/msfv/f/e/sina/cosa (set_map_coriolis).
    map_proj: int = 0
    # --- Phase 3 (Task 9): WRF MM5 surface layer.  0 preserves the
    # pre-physics path; 1 is revised SFCLAY and 91 is classic MM5 SFCLAY.
    sf_sfclay_physics: int = 0
    # --- Phase 3 (Task 10): land surface model.  0 = none (Phase 1/2
    # idealized default, bitwise unchanged), 2 = Noah LSM (WRF
    # sf_surface_physics numbering; kernels/noah.cu + gpuwm/core/noah.py).
    # Wired into the step loop by the physics driver (Task 12).
    sf_surface_physics: int = 0
    # --- Phase 3 (Task 11): planetary boundary layer physics.  The physics
    # driver arrives in Task 12; this option selects its PBL registry entry.
    # 0 = none, 1 = YSU, 5 = MYNN, 11 = Shin-Hong, 900 = SASE.
    bl_pbl_physics: int = 0
    # WRF v4.6.1 Registry.EM_COMMON default; consumes radiative heating.
    ysu_topdown_pblmix: int = 1
    # --- Phase 4 (Task 1): non-timesplit radiation and cumulus slots.
    # ra_physics=90 is the Phase-3 analytic clear-sky proxy; 4 selects
    # RTE+RRTMGP.  cu_physics=1 is Kain-Fritsch; cu_physics=3 is
    # Grell-Freitas (scale-aware; the sig=(1-frh)^2 taper is the scheme's
    # own, so per-domain admission carries no dx gate).
    ra_physics: int = 0
    cu_physics: int = 0
    radt_minutes: float = 12.0
    cudt_minutes: float = 5.0
    # Phase-3 compatibility spelling.  A positive radt continues to
    # override radt_minutes; new configurations use radt_minutes.  bldt
    # remains the surface-layer/LSM/PBL interval.  Zero means every step.
    radt: float = 0.0
    bldt: float = 0.0
    # WRF hypsometric_opt (Registry.EM_COMMON default 2): selects the
    # hydrostatic al/p diagnostic form in calc_p_rho_phi
    # (dyn_em/module_big_step_utilities_em.F:1025-1052) and the matching
    # real-init geopotential construction (module_initialize_real.F:3811,
    # :3962, :3997).  1 = the currently validated d(phi)/d(eta) form
    # (gpuwm's frozen behavior, bitwise-inert default); 2 = the reference
    # WRF run's log-pressure form al = dphi/phm/LOG(pfd/pfu) - alb.
    # Idealized WRF cases always force 1 (share/input_wrf.F:1038); gpuwm's
    # idealized initializers likewise stay on the opt-1 recurrences.
    hypsometric_opt: int = 1
    # --- Phase 4 (Task 8): restart-file write cadence in seconds (WRF
    # restart_interval, minutes there).  0 disables writing.  A positive
    # interval must be a whole multiple of dt (validated by the case run
    # schedule); resuming FROM a restart file is the `gpuwm run --restart`
    # flag, not a config key, so a resumed run keeps its config identical
    # to the writer's (the restart header enforces that identity).
    restart_interval_s: float = 0.0
    # --- Phase 5 (Task 1): multi-domain experiment schema, two INERT
    # defaults.  WRF &bdy_control ``nested`` (the bundle namelist.input
    # runs specified=T,F,F,F / nested=F,T,T,T): False on every legacy
    # single-domain path; the experiment loader (gpuwm/experiment.py)
    # sets it True on child domains and validates it mutually exclusive
    # with ``specified``.  ``grid_id`` (WRF &domains grid_id) is 1 on
    # every legacy path; per-domain ids come from the experiment loader.
    nested: bool = False
    grid_id: int = 1
    # WRF v4.6.1 Registry.EM_COMMON:2898 defaults top_lid=.false. (open top,
    # live calc_coef_w/advance_w top row).  gpuwm defaults to the RIGID lid:
    # the 2026-07-18 production probes showed the open-top branch NaNs a
    # real74 2-dom run within 15 sim-min (iso-opentop_only_2dom) while the
    # rigid branch is byte-identical to the verified pre-flag model
    # (rigid-check vs pacc-base).  Open top remains available for the
    # WRF-fidelity stabilization campaign; flip only with a full-leg
    # stability + falsification receipt.
    top_lid: bool = True
    # WRF always applies calc_cq when moist species exist.  gpuwm defaults
    # OFF: the cq path threw CUDA_ERROR_ILLEGAL_ADDRESS at launch in the
    # 2026-07-18 production probe (iso-cq_only_2dom) — argument marshaling
    # under the prebound launcher is unproven on device.  Same receipt bar
    # as top_lid before flipping.  Dry states bypass cq either way.
    moist_cq: bool = False
    # WRF v4.6.1 Registry.EM_COMMON:2663-2666 default: Morrison dense ice
    # is hail (1); 0 retains the explicit graupel branch.  Appended to
    # preserve positional RunConfig compatibility.  The scheme selects
    # AG/BG/RHOG in module_mp_morr_two_moment.F:337-382.
    morr_rimed_ice: int = 1
    # WRF v4.6.1 WSM6 ``hail_opt``.  WSM6 defaults to graupel; setting 1
    # selects the denser/faster hail intercept, density and fall-speed
    # coefficients in module_mp_wsm6.  Kept scheme-specific so a WSM6
    # trajectory cannot be restarted under a different rimed-ice mode.
    wsm6_hail_opt: int = 0
    # WRF-native split radiation selection.  ``-1/-1`` preserves the
    # historical gpuwm ``ra_physics`` aggregate exactly; new configurations
    # set BOTH values explicitly and leave ra_physics=0.  WRF commonly pairs
    # RRTM LW (1) and Dudhia SW (1), while the existing RTE+RRTMGP adapter is
    # retained as the coupled 4/4 option.
    ra_lw_physics: int = -1
    ra_sw_physics: int = -1
    # WRF module_ra_sw controls.  Both defaults are Registry.EM_COMMON's
    # values and are trajectory-bound even when Dudhia is inactive so a
    # future scheme switch cannot inherit an implicit policy.
    icloud: int = 1
    swrad_scat: float = 1.0
    # Appended port-lane schema.  ``none`` means option 4 was selected as
    # gpuwm's native RTE+RRTMGP algorithm.  The named token is emitted only by
    # the WRF namelist importer when WRF RRTMG 4/4 is intentionally mapped to
    # that modern solver.  It is part of restart/config identity, so the
    # substitution cannot disappear between launch and resume.
    wrf_rrtmg_compatibility: str = "none"
    # The soil geometry REQUESTED, in WRF's namelist spelling.  It is not the
    # count anything allocates from: soil_layer_count(cfg) resolves that from
    # sf_surface_physics and refuses a request the selected scheme does not
    # define, so this default is Noah's four only in the sense that WRF's
    # namelist has a default too.  A scheme with more than one geometry (RUC:
    # six or nine) therefore has to state which, rather than inheriting a
    # number that belongs to a different scheme -- which is exactly the bug
    # this stopped being.  Frozen at 4 by tests/data/config_freeze_golden.json.
    num_soil_layers: int = 4
    # Which implementation serves a resolved 4/4 RRTMG radiation request.
    # The default keeps every existing configuration on gpuwm's modern
    # RTE+RRTMGP adapter, byte-identically.  "rrtmg_legacy" selects the
    # exact port of WRF v4.6.1's bundled RRTMG and FAILS CLOSED at physics
    # setup until its LW/SW compute kernels land -- it never silently
    # falls back to RTE+RRTMGP.  Trajectory-bound through config identity.
    ra_rrtmg_variant: str = "rte-rrtmgp"
    # GPUWM-specific one-way nest microphysics transition.  The default keeps
    # same-scheme behavior.  Mixed edges require an exact versioned contract.
    nest_microphysics_transition: str = "same-scheme-only"
    # WRF MM5 surface-layer options, both scalar in Registry.EM_COMMON
    # (dimension column 1, defaults 0).  The surface-layer kernel already
    # implements every branch: isftcflx selects the Garratt (1) or
    # Donelan (2) water-point heat/moisture roughness in
    # gpuwm/core/kernels/sfclay.cu:356/:371/:429, and iz0tlnd selects the
    # Chen-Zhang (1) or fixed-0.1 (2) land thermal roughness at :399-402,
    # with gpuwm/core/sfclay.py enforcing {0,1,2} for each.  Only the
    # configuration path was missing, so these expose an implemented
    # capability rather than adding one.  Appended to preserve positional
    # RunConfig compatibility.
    isftcflx: int = 0
    iz0tlnd: int = 0
    # Noah LSM options whose kernel branches already exist, so these expose
    # an implemented capability rather than adding one.  ``usemonalb`` selects
    # the monthly-climatology albedo instead of the table min/max endpoints
    # (gpuwm/core/kernels/noah.cu:1112 and :1117), ``rdlai2d`` selects the
    # read-in LAI field instead of the table endpoints (:1036, :1111, :1116),
    # and ``opt_thcnd`` selects the Johansen (1) or McCumber-Pielke (2) soil
    # thermal conductivity at :181.  Only the configuration path was missing;
    # each default reproduces the previously pinned launcher value, so frozen
    # trajectories are bitwise unchanged.  Appended to preserve positional
    # RunConfig compatibility.
    usemonalb: bool = False
    rdlai2d: bool = False
    opt_thcnd: int = 1
    # MYNN EDMF PBL option identity.  Each default is the value the ported
    # solver was validated at against the byte-unmodified module_bl_mynn.F,
    # and each is the ONLY value gpuwm honours: the driver raises rather than
    # approximating an unported branch (gpuwm/core/mynn_pbl.py:mynn_bl_driver).
    # They are configuration fields rather than constants because the physics
    # registry publishes them as knobs, RunConfig is what carries a knob into
    # a run, and restart identity binds the whole config -- so a future lane
    # that widens one of these identities changes a default here and the
    # change is visible in every receipt.
    bl_mynn_closure: float = 2.6
    bl_mynn_cloudpdf: int = 2
    bl_mynn_mixlength: int = 1
    bl_mynn_edmf: int = 1
    bl_mynn_edmf_mom: int = 1
    bl_mynn_edmf_tke: int = 0
    bl_mynn_mixscalars: int = 0
    bl_mynn_cloudmix: int = 1
    bl_mynn_mixqt: int = 0
    bl_mynn_output: int = 0
    bl_mynn_tkeadvect: bool = False
    icloud_bl: int = 1
    # Noah-MP option identity.  Same contract as the MYNN block above: each
    # default is the value the ported column was validated at against the
    # byte-unmodified module_sf_noahmplsm.F, and each is the ONLY value gpuwm
    # honours.  Read by gpuwm/core/physics.py:_run_noahmp, which forwards
    # them to gpuwm/core/noahmp_runtime.py so the refusal for an unported
    # branch comes from the routine that would have had to implement it.
    #
    # These are WRF's ``NoahMP_OPTIONS`` namelist names verbatim.  The
    # enumeration below is honest about what "validated" means: every value
    # is the WRF Registry default AND the value the four whole-column oracle
    # fixtures were generated at, and no other value of any of them has been
    # measured -- not approximated, not measured.
    dveg: int = 4
    opt_crs: int = 1
    opt_btr: int = 1
    opt_run: int = 3
    opt_sfc: int = 1
    opt_frz: int = 1
    opt_inf: int = 1
    opt_rad: int = 3
    opt_alb: int = 2
    opt_snf: int = 1
    opt_tbot: int = 2
    opt_stc: int = 1
    opt_gla: int = 1
    opt_rsf: int = 1
    opt_soil: int = 1
    opt_pedo: int = 1
    opt_crop: int = 0
    opt_irr: int = 0
    opt_irrm: int = 0
    opt_infdv: int = 0
    opt_tdrn: int = 0
    soiltstep: float = 0.0
    noahmp_output: int = 1
    noahmp_acc_dt: float = 0.0
    # RUC option identity.  WRF's own namelist names verbatim:
    # mosaic_lu/mosaic_soil/flag_sm_adj from Registry.EM_COMMON:2535-2537 and
    # spp_lsm from Registry/registry.stoch:241.  Each default is the only
    # value gpuwm honours; RUC_OPTION_IDENTITY_EVIDENCE says what pins it and
    # validate_run_config refuses anything else before a run starts.  Read by
    # gpuwm/core/physics.py:_run_ruc, which forwards all four to
    # gpuwm/core/ruc_runtime.py so the refusal for an unported branch comes
    # from the seam that would have had to implement it.
    mosaic_lu: int = 0
    mosaic_soil: int = 0
    flag_sm_adj: int = 0
    spp_lsm: int = 0
    # WRF nwp_diagnostics (&time_control, Registry.EM_COMMON:2210, default
    # 0).  At 1, WRF computes severe-weather running maxima every step
    # (solve_em.F:369 forces diag_flag; module_first_rk_step_part2.F:533
    # calls cal_helicity) and resets them each history interval
    # (module_diag_nwp.F:246-269).  gpuwm implements the UP_HELI_MAX
    # member of that family (gpuwm/core/uh_diag.py, dycore epilogue); the
    # others (WSPD10MAX, W_UP_MAX, W_DN_MAX, W_MEAN, GRPL_MAX, HAIL_MAX*)
    # are not carried and their absence from wrfouts is the honest signal.
    # The diagnostic is trajectory-inert by construction and by test.
    nwp_diagnostics: int = 0
    # Lane-K option exposure.  These fields are appended so positional
    # RunConfig construction keeps its historical order, and every default is
    # the behavior the assembled tree had before the option became settable.
    #
    # isfflx is WRF's surface heat/moisture-flux gate.  The ported MM5 and
    # MYNN surface-layer paths implement 0/1; WRF's prescribed-flux value 2
    # also needs the diffusion-side forcing contract and is not admitted.
    isfflx: int = 1
    # Legacy RRTMG's two implemented ozone sources: wrapper O3DATA (0) and
    # the CAM climatology / parent-routed o3rad field (2).  RTE+RRTMGP keeps
    # its established climatological-ozone identity at 2.
    o3input: int = 2
    # WRF's module_physics_init gate around the microphysics effective-radius
    # scheme table.  Only legacy RRTMG owns that wrapper branch.
    use_mp_re: int = 1
    # RUC's live CASE(RUCLSMSCHEME) sea-ice albedo replacement.  The previous
    # literal was 0.65, so that remains the gpuwm default.
    seaice_albedo_default: float = 0.65
    # Noah LSMINIT snow-albedo source.  True preserves the supplied geogrid
    # SNOALB field; false transcribes WRF's MAXALB(IVGTYP)*0.01 branch.
    rdmaxalb: bool = True
    # --- km_opt=3 (3-D Smagorinsky) LES closure knobs, WRF's own names.
    # Appended last to preserve positional construction (the config-freeze
    # discipline); each default is WRF's Registry default, so every frozen
    # km_opt=1/4 trajectory is unchanged.
    #
    # WRF mix_isotropic (Registry.EM_COMMON:2896, "0=anistropic,
    # 1=isotropic", default 0): selects smag_km's mixing-length branch for
    # km_opt=3 -- separate horizontal sqrt(dx*dy) / vertical dz lengths at
    # 0, the single (dx*dy*dz)^(1/3) length at 1.  The em_les reference
    # namelist sets 1.
    mix_isotropic: int = 0
    # WRF mix_upper_bound (Registry.EM_COMMON:2897, default 0.1): the
    # non-dimensional K cap K <= mix_upper_bound * len^2 / dt applied per
    # direction inside smag_km (module_diffusion_em.F:1890-1927).
    mix_upper_bound: float = 0.1
    # WRF tke_heat_flux / tke_drag_coefficient (Registry.EM_COMMON:
    # 2900-2901, defaults 0., per-domain): the prescribed-flux lower
    # boundary consumed by vertical_diffusion_2's SELECT CASE(isfflx)
    # blocks under diff_opt=2 with the PBL off -- constant kinematic heat
    # flux (K m s-1) for isfflx in {0,2}, constant drag coefficient for
    # isfflx=0 (module_diffusion_em.F:4155-4200, :4286-4305).
    tke_heat_flux: float = 0.0
    tke_drag_coefficient: float = 0.0
    # WRF c_k (Registry.EM_COMMON:2863, default 0.15): the TKE-closure
    # exchange-coefficient constant K = c_k * sqrt(e) * l for km_opt=2.
    # The em_les reference namelist sets 0.10 (README.les: recommended
    # with LES).  Appended last per the config-freeze discipline.
    c_k: float = 0.15
    # WRF tke_upper_bound (Registry.EM_COMMON:2899, default 1000., per
    # domain): the ceiling in ``bound_tke`` (dyn_em/module_em.F:2490-2520),
    # which clamps the prognostic TKE into [0, tke_upper_bound] over the
    # whole mass grid after EVERY rk_update_scalar pass
    # (dyn_em/solve_em.F:2434-2440) -- on periodic domains too, not only
    # at lateral boundaries.
    tke_upper_bound: float = 1000.0
    # Report-only per-step TKE budget accumulation (km_opt=2): writes the
    # term-by-term decomposition into diagnostic scratch and reduces it on
    # device once per step.  Trajectory-inert by construction -- it reads
    # model state and writes only its own slots -- so it is a restart-
    # boundary-adjustable diagnostic toggle exactly like nwp_diagnostics
    # (gpuwm/io/restart.py CONFIG_DIAGNOSTIC_FIELDS).
    tke_budget: int = 0
    # --- Aerosol-aware Thompson (mp_physics=28) aerosol-source selectors ---
    #
    # Both defaults are WRF's own Registry defaults, and that is load-bearing
    # rather than cosmetic: gpuwm/physics_compat.py's
    # _SINGLE_DOMAIN_RUNTIME_SWITCHES rows are compared for EXACT equality by
    # the prepared-forecast runner, so a nonzero default here would silently
    # change every shipped profile.
    #
    # aer_init_opt -- Registry/Registry.EM_COMMON:2656, default 0.  WRF
    # declares it ``derived`` (real.exe sets it from use_aero_icbc /
    # use_rap_aero_icbc), not ``namelist``: 0 = no IC/BC aerosol, 1 = climo,
    # 2 = first guess.  ArWen exposes it as a settable field ONLY so that a
    # request for 1 or 2 is refused by name instead of being unrepresentable
    # and therefore silently ignored.  0 is the only implemented value:
    # 1 and 2 both read WIF fields real.exe interpolated from metgrid, and
    # ArWen has no such ingest (see MP28_AEROSOL_SOURCE_DEVIATION).
    aer_init_opt: int = 0
    # wif_input_opt -- Registry/registry.new3d_wif:17, default 0.
    # 0 = do not process the Water/Ice Friendly aerosol input from metgrid;
    # 1 = use_wif_input; 2 = use_wif_input_bc, which additionally allocates
    # the black-carbon scalar qnbca.  ArWen implements neither: there is no
    # WIF metgrid stream and no nbca species anywhere in the port, so any
    # nonzero value fails closed.
    wif_input_opt: int = 0
    # --- SASE closure knobs (bl_pbl_physics = SASE_PBL_SCHEME only).
    # Appended last, per the config-freeze discipline in
    # tests/test_config_freeze.py.  Every one is FAIL-CLOSED on its
    # NON-DEFAULT value: the default is admitted under every PBL scheme,
    # so no existing configuration moves, and the non-default is refused
    # anywhere it would name a seam that does not exist.
    #
    # Split subgrid-flux diagnostic (output-only, per domain).  True adds
    # four face-registered history fields recording the closure's own
    # vertical subgrid fluxes with the conditional-venting channel
    # separated from the K_v implicit-diffusion channel.  It reads no
    # prognostic and writes none, so the prognostic state is bitwise
    # identical either way; per domain because the cost scales with the
    # grid (four planes per frame).
    sase_flux_diag: bool = False
    # Moist-N2 substitution (physics selector, run-wide).  True is the
    # closure as built: the saturated Brunt-Vaisala N^2_m replaces the dry
    # N^2 at the stability lengths, the subgrid-energy buoyancy source and
    # the K_v/K_h stability suppression.  False consumes the dry N^2 at
    # all three points, isolating the diffusion channel from the venting
    # channel.  Not a per-domain override: a nest whose domains ran
    # different closures could not be compared across its own boundary.
    sase_moist_n2: bool = True
    # Stable-limb dissipation decoupling (physics selector, run-wide).
    # True rides the Deardorff lambda << Delta dissipation coefficient
    # where the stability length binds the dissipation length, instead of
    # the neutral-wall value.  DEFAULT FALSE FOR A MEASURED REASON, not
    # merely for caution: with it True the closure's own registered
    # stable-boundary-layer calibration gate exits its observation band
    # (tests/test_sase.py::
    # test_jet_decoupling_stable_dissipation_exits_obs_band pins that
    # RED).  What is falsified is the PAIR of stable-limb coefficients,
    # which enter the stability ratio jointly; neither may be
    # re-registered or tuned alone.  Do not flip this default without
    # that joint re-registration.
    sase_stable_dissipation: bool = False
    # SASE S3-12 ADDITIVE e^{3/2} DISSIPATION CHANNEL.  True ADDS the
    # second, grid-scale member of Deardorff's length-dependent
    # coefficient (C_ED = 0.51) to whichever base the key above selects,
    # divided by a STATE-INDEPENDENT reference length.  Where the
    # stability length binds, HEAD's subgrid-energy equation is exactly
    # homogeneous in e and has no equilibrium amplitude at all -- below a
    # critical Richardson number of 0.1647 the energy grows exponentially
    # with nothing in the closure to stop it (measured on real data:
    # 1.62 m2/s2 at 13.1 km, still doubling every ~3 min at t+60 min).
    # This channel carries no e in its length, so it gives that limb a
    # finite, attracting fixed point.
    #
    # It is the OPPOSITE MOVE from the key above and composes with it:
    # sase_stable_dissipation LOWERS the coefficient (and is RED on the
    # closure's own stable-boundary-layer gate for exactly that reason),
    # this one ADDS to it, so dissipation is nowhere weaker than the
    # default path's.  Both registered calibration gates hold with it
    # True and the jet gate gains margin; the lake gate's margin
    # narrows from 28% to 9% and that is recorded at the fixture.
    # DEFAULT FALSE because flipping it moves bitwise goldens and five
    # RED legs that pin historical formulations, which is a separate
    # decision with its own evidence (authority module docstring, S3-12
    # section, "DEFAULT FALSE, AND WHY").
    sase_additive_dissipation: bool = False
    # --- Horizontal-mixing diagnostic and the km_opt = 0 acknowledgement.
    #
    # Horizontal eddy-viscosity diagnostic (output-only, per domain).  True
    # adds the horizontal eddy viscosities the run's OWN mixing producer
    # used, under the name of that producer: XKMH/XKHH for the km_opt = 4
    # Smagorinsky operator, SASE_KMH/SASE_KHH for the SASE closure's
    # governed horizontal diffusivity.  Both are m2 s-1 on the mass grid
    # and mean the same thing, so a run that removes one producer and adds
    # the other can be MEASURED on the channel it swapped rather than
    # argued about.  Reads no prognostic and writes none.
    hmix_k_diag: bool = False
    # Expert acknowledgement admitting km_opt = 0 with a PBL scheme that
    # does not itself produce horizontal mixing -- i.e. a run with NO
    # horizontal mixing operator at all.  Must be set to the exact id
    # KM_OPT_ZERO_ACK; anything else (including True) is refused.  See
    # KM_OPT_ZERO_ACK for what it means and why the default refuses.
    km_opt_zero_acknowledgement: str = ""
    # --- LES-nest inflow turbulence seeding (P3, per-domain, TOML-only).
    #
    # CPM-style cell-blocked theta perturbation of the child's rolling
    # nest-boundary VALUE tables on the inflow-side relax-zone rows,
    # refreshed on the FORCE cadence (gpuwm/core/inflow_perturbation.py;
    # every constant pinned in docs/superpowers/receipts/les/
    # INFLOW-GENERATOR-ACCEPTANCE-V2.md, registered before data).  An
    # ArWen-over-WRF extension (PROVENANCE.md D10): stock v4.6.1 ships
    # perturb_bdy (stoch-package boundary-tendency patterns), not a
    # cell-perturbation path, and has no namelist column for these, so a
    # config using them cannot round-trip to a namelist -- exactly the
    # per-domain isfflx situation.  Default OFF; the OFF trajectory is
    # gated byte-identical to a build without the mechanism.
    inflow_perturbation: bool = False
    # RNG seed; part of every draw's Philox key (deterministic,
    # counter-based -- same seed, same bytes, every card).
    inflow_perturbation_seed: int = 0
    # Multiplies the pinned Eckert-number amplitude.  0.0 skips every
    # table write and is the registered zero-amplitude negative control
    # (acceptance gate G2): bitwise-identical to OFF.
    inflow_perturbation_amplitude_scale: float = 1.0
    # "inflow" perturbs the flow_dep_bdy inflow faces (the mechanism);
    # "outflow" perturbs the complementary faces instead -- the
    # registered AC-P3.4 mutation control, selected by configuration.
    inflow_perturbation_faces: str = "inflow"
    # WRF v4.6.1 Registry/Registry.EM_COMMON:2889, verbatim:
    #
    #   rconfig   logical  moist_mix6_off   namelist,dynamics  max_domains
    #       .false. rh  "moist_mix6_off"
    #       "de-activate 6th-order horizontal filter for moisture"  ""
    #
    # WRF's own switch, in WRF's own spelling, with WRF's own default: the
    # importer must round-trip a stock namelist that sets it, and inventing
    # a second name for a knob WRF already has is the defect commit a0ef9d29
    # reverted for isfflx.  It is per domain because the Registry column is
    # max_domains.
    #
    # WHAT IT GATES, EXACTLY: rk_scalar_tend calls sixth_order_diffusion on
    # the moist array under ``(diff_6th_opt .NE. 0) .and. (.not. mix6_off)``
    # (dyn_em/module_em.F:1421, reached from dyn_em/solve_em.F:2230 with
    # config_flags%moist_mix6_off).  It touches the MOIST array only --
    # theta keeps its filter, and TKE has its own tke_mix6_off -- so gpuwm
    # gates exactly the moist rows of the diff6 row set and nothing else.
    #
    # Default .false. is WRF's, so every frozen trajectory is bitwise
    # unchanged and diff_6th_opt = 0 runs make it inert either way.  It is
    # divergence-ledger entry L4 (gpuwm/physics_mode.py, PROVENANCE.md):
    # a configuration-policy patch WRF can also express, which is why it is
    # measured against observations rather than asserted.
    moist_mix6_off: bool = False
    # WRF v4.6.1 Registry defaults for the Grell-family namelist keys
    # (Registry.EM_COMMON:2544,2546).  clos_choice=0 is the 16-member
    # ensemble closure -- the only arm the GF oracle covers, and the only
    # admitted value; ishallow toggles CUP_gf_sh, both arms oracle-covered.
    # Read only where cu_physics = 3, which no frozen configuration
    # selects, so appending them cannot move a frozen trajectory.
    clos_choice: int = 0
    ishallow: int = 0


#: The Noah-MP option identity gpuwm admits, field -> the only accepted
#: value, with what pins it.  ``validate_run_config`` refuses anything else
#: *before* a run starts.
#:
#: Read the second column as the evidence, not as a slogan.  "fixture" means
#: the four whole-column ``noahmp-sflx.csv`` cases were generated with that
#: value and the port is bitwise against them; "dead-proved" means the option
#: kills a code region and the kill is proved in
#: :mod:`gpuwm.core.noahmp_sflx`'s module docstring out of the source rather
#: than assumed; "unmeasured" means no other value has been tried at all.
#: Nothing here claims that a *different* value would work.
NOAHMP_OPTION_IDENTITY_EVIDENCE: dict[str, tuple[object, str]] = {
    "dveg": (4, "fixture; FVEG=SHDMAX at :863-864 and dveg_active=.false. "
                "at :1016 kill CARBON -- dead-proved"),
    "opt_crs": (1, "fixture; opt_crs=2 selects CANRES+CALHUM, which is not "
                   "transcribed -- unmeasured"),
    "opt_btr": (1, "fixture; the Noah btran form only"),
    "opt_run": (3, "fixture; Schaake96.  Kills GROUNDWATER, "
                   "SHALLOWWATERTABLE, ZWTEQ, VIC, XAJ and DVIC -- "
                   "dead-proved"),
    "opt_sfc": (1, "fixture; SFCDIF2 is not transcribed -- unmeasured"),
    "opt_frz": (1, "fixture; NY06 supercooled liquid"),
    "opt_inf": (1, "fixture; NY06 frozen-soil permeability"),
    "opt_rad": (3, "fixture; gap = 1 - FVEG"),
    "opt_alb": (2, "fixture; CLASS.  SNOWALB_BATS is not transcribed -- "
                   "unmeasured"),
    "opt_snf": (1, "fixture; Jordan91. FPICE comes from SFCTMP while the "
                   "driver separately supplies WRF's six precipitation "
                   "rates"),
    "opt_tbot": (2, "fixture; Noah lower boundary at ZBOT"),
    "opt_stc": (1, "fixture; semi-implicit snow/soil temperature"),
    "opt_gla": (1, "declared only.  Every glacier column RAISES "
                   "(gpuwm/core/noahmp_runtime.py), so no glacier physics "
                   "runs at any opt_gla and this value is not evidence of "
                   "one -- it exists so a plan that asks for opt_gla=2 is "
                   "refused rather than silently ignored"),
    "opt_rsf": (1, "fixture; Sakaguchi/Zeng ground resistance"),
    "opt_soil": (1, "fixture; one soil category per column.  The 3-D and "
                    "pedotransfer branches at :737-746 are not "
                    "transcribed -- unmeasured"),
    "opt_pedo": (1, "declared only; opt_soil=1 makes PEDOTRANSFER_SR2006 "
                    "unreachable, so this value has no consumer and is "
                    "pinned to keep it that way"),
    "opt_crop": (0, "dead-proved; kills CARBON_CROP, the crop FVEG "
                    "override and the gecros state vector"),
    "opt_irr": (0, "dead-proved; the whole irrigation region 878-936 and "
                   "1042-1045 leaves every amount at zero"),
    "opt_irrm": (0, "dead-proved with opt_irr=0"),
    "opt_infdv": (0, "dead-proved; the dynamic-VIC infiltration variants "
                     "are reachable only at opt_run=6"),
    "opt_tdrn": (0, "dead-proved; QTLDRN is identically zero out of WATER"),
    "soiltstep": (0.0, "fixture; forces soil_update_steps=1 and "
                       "calculate_soil=.true., which is what makes the ten "
                       "ACC_* carriers per-step rather than trajectory "
                       "state"),
    "noahmp_output": (1, "declared only; gpuwm has no counterpart to WRF's "
                         "module_diag_misc output-accumulator block, so no "
                         "value of this knob changes what gpuwm writes"),
    "noahmp_acc_dt": (0.0, "declared only, for the same reason"),
}

#: The enforced form of the table above: field -> the only accepted value.
NOAHMP_OPTION_IDENTITY: dict[str, object] = {
    name: value for name, (value, _why) in
    NOAHMP_OPTION_IDENTITY_EVIDENCE.items()
}


#: The MYNN PBL option identity gpuwm admits, field -> the only accepted
#: value.  ``validate_run_config`` refuses anything else *before* a run
#: starts, so the driver's own refusal is a second line rather than the
#: first: a user who asks for bl_mynn_mixlength=2 must be told at
#: configuration time, not three hours into a forecast.
MYNN_PBL_OPTION_IDENTITY: dict[str, object] = {
    "bl_mynn_closure": 2.6,
    "bl_mynn_cloudpdf": 2,
    "bl_mynn_mixlength": 1,
    "bl_mynn_edmf": 1,
    "bl_mynn_edmf_mom": 1,
    "bl_mynn_edmf_tke": 0,
    "bl_mynn_mixscalars": 0,
    "bl_mynn_cloudmix": 1,
    "bl_mynn_mixqt": 0,
    "bl_mynn_output": 0,
    "bl_mynn_tkeadvect": False,
    "icloud_bl": 1,
}


#: The RUC option identity gpuwm admits, field -> (only accepted value, what
#: pins it).  Same contract as the Noah-MP block above: ``validate_run_config``
#: refuses anything else BEFORE a run starts, and
#: :func:`gpuwm.core.physics.PhysicsDriver._run_ruc` reads every one off the
#: configuration by name so the registry's citation of that file is true.
#:
#: RUC's namelist surface is small -- three knobs in
#: ``Registry.EM_COMMON:2535-2537`` and one in ``Registry/registry.stoch:241``
#: -- and none of the four has a validated nonzero value.  That makes the
#: interesting restrictions the ones with no namelist field at all, which is
#: why :data:`gpuwm.core.ruc_runtime.RUC_RUNTIME_RESTRICTIONS` exists and is
#: published beside this table rather than instead of it.
RUC_OPTION_IDENTITY_EVIDENCE: dict[str, tuple[object, str]] = {
    "mosaic_lu": (0, "dead-proved; gpuwm.core.ruc.ruc_surface_parameters is "
                     "fail-closed on SOILVEGIN's mosaic arms, and LSMRUC's "
                     "irrigation block (:984-1009) is gated on the same "
                     "mosaic_lu==1, so it is unreachable wherever SOILVEGIN "
                     "is"),
    "mosaic_soil": (0, "dead-proved with mosaic_lu=0; the soilctop/nscat "
                       "mosaic soil arm of SOILVEGIN is the other half of "
                       "the same refusal"),
    "flag_sm_adj": (0, "no consumer; share/module_soil_pre.F:2063 reads this "
                       "inside init_soil_ruc, i.e. in real.exe, to adjust a "
                       "Noah-derived RUC soil state.  gpuwm has no RUC soil "
                       "ingest for it to adjust, so the knob is pinned at 0 "
                       "to keep it that way rather than accepted and "
                       "ignored"),
    "spp_lsm": (0, "not ported; the ARW path needs pattern_spp_lsm and "
                   "field_sf stochastic inputs plus their restart contract"),
}

#: The enforced form of the table above: field -> the only accepted value.
RUC_OPTION_IDENTITY: dict[str, object] = {
    name: value for name, (value, _why) in
    RUC_OPTION_IDENTITY_EVIDENCE.items()
}


# --------------------------------------------------------------------------
# WRF selector schema.
#
# These tables answer only "is this a WRF selector value gpuwm's schema
# knows, with a coherent soil geometry".  Whether the scheme is EXECUTABLE is
# a separate question owned by gpuwm/physics_compat.py, which runs first in
# validate_run_config and emits the complete fail-closed port receipt.
# Keeping the two apart is what makes an eventual admission a one-line change
# in physics_compat rather than a scatter of numeric literals here, and it
# stops an in-port scheme collapsing onto a generic "must be 0 or 2" message
# that says nothing about what is actually missing.
# --------------------------------------------------------------------------

#: WRF sf_sfclay_physics values in gpuwm's schema.
SURFACE_LAYER_SCHEMES = (0, 1, 5, 91)
#: WRF sf_surface_physics values in gpuwm's schema.
LAND_SURFACE_SCHEMES = (0, 2, 3, 4)
#: ``bl_pbl_physics`` value selecting the SASE closure.
#:
#: SASE is an ArWen-only scheme -- a Scale-Adaptive Stress-Energetics
#: closure with no WRF v4.6.1 counterpart -- so it cannot take a WRF
#: number without claiming to be a transcription of something.  WRF's own
#: ``bl_pbl_physics`` namespace runs to 99 (99 = MRF), so a value above it
#: can never collide with a scheme WRF adds later, and it deliberately
#: falls outside :data:`gpuwm.wrf461_compatibility.PBL_OPTIONS` so the
#: WRF PBL/surface-layer compatibility matrix -- which states what *WRF*
#: admits -- is not consulted about a scheme WRF does not have.  The
#: readiness authority is the registry entry and the constraints below.
SASE_PBL_SCHEME = 900

#: Deepest column the SASE vertical solve accepts, read from the closure's
#: own authority module so the device define, the launcher's rejection and
#: this admission check cannot drift apart.  ``sase_ref`` is NumPy-only, so
#: importing it here costs the config layer no CuPy dependency.
SASE_MAX_NZ = _sase_limits.MAX_COLUMN_LEVELS

#: WRF's MYNN surface layer (``sf_sfclay_physics``).  Named because its
#: compatibility is unusually narrow: WRF v4.6.1 admits it with the MYNN
#: PBL or with no PBL and with nothing else, which is the 16-cell matrix
#: at ``phys/module_physics_init.F:3699-3704,3837-3839`` that the registry
#: publishes under ``authority.wrf_v461_compatibility_matrix``.
MYNN_SFCLAY_SCHEME = 5

#: ``bl_pbl_physics`` values in gpuwm's schema.  0/1/5/11 are WRF's --
#: 11 is Shin-Hong (module_bl_shinhong.F, ported at max ULP 0 against the
#: byte-frozen WRF v4.6.1 module; runtime wired by _run_shinhong in
#: gpuwm/core/physics.py) -- and 900 is :data:`SASE_PBL_SCHEME`,
#: ArWen-only (see there).
PBL_SCHEMES = (0, 1, 5, 11, SASE_PBL_SCHEME)
# 0 = none, 1 = Kain-Fritsch, 3 = Grell-Freitas.
CU_SCHEMES = (0, 1, 3)

#: The exact id :attr:`RunConfig.km_opt_zero_acknowledgement` must carry.
#:
#: ONE SENTENCE, the one the refusal prints: ``km_opt = 0`` with a PBL
#: scheme that produces no horizontal mixing of its own leaves the run
#: with NO horizontal mixing operator, so the only thing damping
#: grid-scale horizontal structure is the sixth-order numerical filter,
#: and that is normally refused because it is what a mis-set switch
#: looks like rather than what a forecast wants.
#:
#: WHY IT IS ADMITTED AT ALL.  This is a real configuration, not an
#: impossible one: it is WRF's own ``diff_opt = 0``, and it is the
#: control the SASE attribution needs.  SASE is admitted at ``km_opt =
#: 0`` because its closure supplies the horizontal mixing the operator
#: would otherwise apply.  That admission, on its own, made the
#: single-variable control UNWRITABLE -- every SASE run necessarily
#: changed the PBL scheme AND removed the Smagorinsky operator, so no
#: run could say which of the two a difference came from.  Holding the
#: PBL scheme fixed and removing only the mixing is the missing cell.
#:
#: THIS IS NOT A GATE WIDENED TO PASS A TEST.  The gate's job is to stop
#: an unwitting misconfiguration, and it still does: the default still
#: refuses, the refusal still names the missing producer, and nothing
#: passes by accident.  What changed is that a user who states in
#: writing that the absent operator is the point can now write the run.
#: The id is a literal string rather than a boolean precisely so it
#: cannot be reached by a stray ``= true`` or an inherited default.
KM_OPT_ZERO_ACK = "no-horizontal-mixing-operator-v1"
#: sf_surface_physics -> the soil-layer counts that scheme defines.
#:
#: This is WRF's own table.  ``share/module_check_a_mundo.F`` subroutine
#: ``set_physics_rconfigs`` (:3546-3577) *resolves* ``num_soil_layers`` from
#: ``sf_surface_physics(1)`` -- it overwrites whatever the namelist asked for
#: -- and calls ``wrf_error_fatal`` on a scheme it has no entry for.  gpuwm
#: follows that shape in :func:`soil_layer_count`, with one deliberate
#: difference recorded there.
#:
#: No count is written here.  Each value is READ from the module that owns the
#: scheme, so RUC's nine is stated once, next to the nine level depths and the
#: ``__constant__ real ruc_soil_layer_depth[9]`` the kernel indexes:
#:
#: * Noah (2) and Noah-MP (4) share ``init_soil_depth_2``
#:   (``share/module_soil_pre.F:795`` and ``:807``, which is itself fatal at
#:   any other count), so both read
#:   :data:`gpuwm.core.noah.NUM_SOIL_LAYERS`.
#: * RUC (3) is the only multi-geometry scheme.  ``init_soil_depth_3``
#:   (:1153-1194) tabulates zs for 6 and for 9 and is fatal at 4 or 5;
#:   :data:`gpuwm.core.ruc_contract.WRF_SUPPORTED_NUM_SOIL_LAYERS` is that
#:   pair, in WRF's order.  Which of the two gpuwm has actually validated is
#:   a separate question, answered by :func:`validated_soil_layer_count` and
#:   deliberately NOT encoded as this tuple's order -- the schema states what
#:   the scheme defines, not what gpuwm has finished.
#:
#: DIVERGENCE, deliberate: WRF gives NOLSMSCHEME five layers
#: (``module_check_a_mundo.F:3547-3548``, a slab-scheme inheritance).  gpuwm's
#: ``sf_surface_physics=0`` allocates no soil state at all, so the number is
#: only the length of a wrfout axis no field is written on; it is held at
#: Noah's so the frozen idealized-case wrfout headers (hill2d, igw, straka,
#: wk82, moist_bubble) keep byte identity.  Moving it would change five gated
#: files to describe an axis that stays empty.
class _LandSurfaceSoilLayers(_Mapping):
    """The table above, resolved per scheme on first ask.

    A lazy Mapping rather than a dict for one concrete reason: RUC's contract
    module is forecast-side, and the RW-WPS preprocessing wheel stages
    ``gpuwm/config.py`` but deliberately not ``gpuwm/core/ruc_contract.py``
    (``tools/build_rw_wps_release.py``, which fails the staging if config
    carries an import the wheel cannot resolve).  Reading RUC's geometry at
    config-import time would therefore make gpuwm.config unimportable in that
    distribution, and the alternative -- writing 6 and 9 here instead --
    is the very duplication this table exists to prevent.  Deferring the read
    to the first question *about scheme 3* keeps both properties at once.

    ``_PROVIDERS`` names modules and attributes, never counts.
    """

    #: scheme -> (module, attribute holding every count WRF's own generator
    #: tabulates for that scheme, attribute holding the single count gpuwm has
    #: validated -- the same attribute when the scheme defines one geometry).
    _PROVIDERS = {
        0: ("gpuwm.core.noah", "NUM_SOIL_LAYERS", "NUM_SOIL_LAYERS"),
        2: ("gpuwm.core.noah", "NUM_SOIL_LAYERS", "NUM_SOIL_LAYERS"),
        3: ("gpuwm.core.ruc_contract", "WRF_SUPPORTED_NUM_SOIL_LAYERS",
            "NUM_SOIL_LAYERS"),
        4: ("gpuwm.core.noah", "NUM_SOIL_LAYERS", "NUM_SOIL_LAYERS"),
    }

    def __init__(self):
        self._resolved: dict[int, tuple[tuple[int, ...], int]] = {}

    def _read(self, scheme: int) -> tuple[tuple[int, ...], int]:
        try:
            return self._resolved[scheme]
        except KeyError:
            pass
        module_name, counts_attr, validated_attr = self._PROVIDERS[scheme]
        module = _import_module(module_name)
        counts = getattr(module, counts_attr)
        counts = ((int(counts),) if isinstance(counts, int)
                  else tuple(int(value) for value in counts))
        validated = int(getattr(module, validated_attr))
        if validated not in counts:
            raise AssertionError(
                f"{module_name}.{validated_attr}={validated} is not one of "
                f"{module_name}.{counts_attr}={counts}; the scheme's own "
                "module disagrees with itself about its soil geometry")
        entry = (counts, validated)
        self._resolved[scheme] = entry
        return entry

    def __getitem__(self, scheme: int) -> tuple[int, ...]:
        return self._read(scheme)[0]

    def validated(self, scheme: int) -> int:
        """The one geometry gpuwm has validated for ``scheme``.

        Kept apart from ``__getitem__`` on purpose: the schema states what the
        SCHEME defines (RUC: WRF's 6 and 9, in WRF's order), which is a
        different fact from what gpuwm has finished, and encoding the second
        as the first tuple's order would quietly conflate them.
        """
        return self._read(scheme)[1]

    def __iter__(self):
        return iter(self._PROVIDERS)

    def __len__(self) -> int:
        return len(self._PROVIDERS)


LAND_SURFACE_SOIL_LAYERS = _LandSurfaceSoilLayers()


def validated_soil_layer_count(sf_surface_physics: int) -> int:
    """The soil geometry gpuwm has validated for one land-surface scheme."""
    return LAND_SURFACE_SOIL_LAYERS.validated(int(sf_surface_physics))


#: The wrfout soil axis for a run that selected no land-surface scheme.  Named
#: so :mod:`gpuwm.io.wrfout` can default its ``soil_layers`` argument without
#: carrying a soil-geometry constant of its own.
NO_LAND_SURFACE_SOIL_LAYERS = LAND_SURFACE_SOIL_LAYERS[0][0]
#: Human-readable names used only in validation messages.
_LAND_SURFACE_NAMES = {
    0: "none", 2: "Noah LSM", 3: "RUC LSM", 4: "Noah-MP LSM"}


def soil_layer_count(cfg: RunConfig) -> int:
    """Resolved soil-layer count for the configured land-surface scheme.

    Every soil allocation, VRAM count, output dimension and restart shape
    reads this.  It is a *resolver*, not an accessor: it consults
    :data:`LAND_SURFACE_SOIL_LAYERS` for the scheme actually selected, so a
    ``RunConfig`` built in code rather than loaded through
    :func:`load_config` -- which is how most tools and tests build one, and
    which runs no validation -- cannot quietly hand a nine-layer scheme
    Noah's four.  That mattered concretely: the VRAM preflight sizes TSLB,
    SMOIS, SH2O and SMCREL from this number, and on this hardware an
    understated preflight is a correctness failure, not an estimate.

    DIVERGENCE from WRF, deliberate: ``set_physics_rconfigs`` *overwrites* a
    namelist request that disagrees with the scheme and only logs it at debug
    level, so a namelist asking Noah for nine layers runs on four with no
    error.  gpuwm refuses instead.  Silently reinterpreting a soil geometry
    is how a run produces plausible, wrong soil and no receipt of it.
    """
    scheme = int(cfg.sf_surface_physics)
    defined = LAND_SURFACE_SOIL_LAYERS.get(scheme)
    if defined is None:
        # module_check_a_mundo.F:3573-3576, the same fail-closed default.
        raise ValueError(
            f"sf_surface_physics={scheme} has no associated number of soil "
            "levels; a land-surface scheme must declare its soil geometry "
            "in LAND_SURFACE_SOIL_LAYERS before it can be sized, written "
            "or checkpointed.")
    requested = int(cfg.num_soil_layers)
    if requested in defined:
        return requested
    name = _LAND_SURFACE_NAMES.get(scheme, f"sf_surface_physics={scheme}")
    expected = " or ".join(str(value) for value in defined)
    detail = ("" if len(defined) == 1 else
              f"  {name} defines more than one soil geometry, so it has no "
              "default to inherit: gpuwm's validated geometry is "
              f"{LAND_SURFACE_SOIL_LAYERS.validated(scheme)} and must be "
              "requested explicitly, rather than taken from a RunConfig "
              "default that belongs to another scheme.")
    raise ValueError(
        f"num_soil_layers must be {expected} for {name} "
        f"(sf_surface_physics={scheme}), got {requested}.{detail}")


def radiation_scheme_ids(cfg: RunConfig) -> tuple[int, int]:
    """Resolve effective WRF ``(LW, SW)`` radiation scheme IDs."""
    lw = int(getattr(cfg, "ra_lw_physics", -1))
    sw = int(getattr(cfg, "ra_sw_physics", -1))
    if lw == -1 and sw == -1:
        legacy = int(getattr(cfg, "ra_physics", 0))
        return legacy, legacy
    if (lw == -1) != (sw == -1):
        raise ValueError(
            "ra_lw_physics and ra_sw_physics must both be explicit or both "
            "be -1 (legacy ra_physics compatibility)")
    if int(getattr(cfg, "ra_physics", 0)) != 0:
        raise ValueError(
            "explicit ra_lw_physics/ra_sw_physics require ra_physics=0; "
            "do not mix split and legacy radiation selection")
    return lw, sw


def radiation_enabled(cfg: RunConfig) -> bool:
    """Whether either resolved radiation component is active."""
    return any(radiation_scheme_ids(cfg))


# --------------------------------------------------------------------------
# mp_physics = 28 aerosol source: the one deliberate deviation, published.
# --------------------------------------------------------------------------

#: What ArWen's ONLY implemented mp=28 aerosol source is, and the exact WRF
#: line that refuses the same configuration.
#:
#: This is not a footnote.  ``mp_physics = 28`` with ``wif_input_opt = 0``
#: is a combination WRF's own initializer FATALs
#: (``dyn_em/module_initialize_real.F:2735-2736``:
#: ``ELSE IF (config_flags%mp_physics .EQ. THOMPSONAERO .and.
#: config_flags%wif_input_opt .EQ. 0 ) THEN
#: CALL wrf_error_fatal ('wif_input_opt=0 but mp_physics=28')``), because
#: real.exe expects the water/ice-friendly aerosol fields to have been
#: interpolated from a metgrid WIF stream.  ArWen has no such stream, so it
#: runs the synthetic CCN/IN profile ``thompson_init`` itself fills
#: (``phys/module_mp_thompson.F:482-559``; the CCN branch at :493-515 and
#: the IN branch at :531-551, each taken when ``MAXVAL(nwfa)``/
#: ``MAXVAL(nifa)`` is below ``eps``) -- the same code path WRF uses when the
#: aerosol fields arrive unset.
#:
#: The PHYSICS is therefore WRF's, unmodified.  What differs is the
#: INITIALIZATION: a WRF user cannot reach this state through real.exe, and
#: an ArWen mp=28 run is consequently not directly comparable to a WRF mp=28
#: run initialized from a WIF-bearing met_em.  Anyone comparing the two must
#: know that, which is why this string is carried in the namelist importer's
#: printed receipt (:class:`gpuwm.namelist_import.AppliedDefault`) rather
#: than only in a comment.
MP28_AEROSOL_SOURCE_DEVIATION = (
    "mp_physics=28 takes its aerosol initial state from ArWen's port of "
    "thompson_init, which installs WRF's synthetic CCN/IN profile "
    "(phys/module_mp_thompson.F:482-559) once per domain from "
    "gpuwm/core/physics.py::initialize_physics. It does NOT come from a "
    "metgrid WIF stream: ArWen has no QNWFA/QNIFA ingest lane, and no "
    "aerosol crosses a specified lateral boundary either. WRF's own "
    "real.exe FATALs exactly this configuration "
    "(dyn_em/module_initialize_real.F:2735-2736, 'wif_input_opt=0 but "
    "mp_physics=28'), so the microphysics is WRF's unmodified aerosol-aware "
    "Thompson while the aerosol INITIALIZATION is one WRF's initializer "
    "refuses to produce. Column physics is oracle-measured against "
    "unmodified WRF Fortran; a whole-forecast comparison against a "
    "WIF-initialized WRF run is NOT equivalent and must not be reported as "
    "one."
)

#: The only implemented value of each mp=28 aerosol-source selector, and the
#: named capability a user would need for anything else.  Read by
#: :func:`validate_aerosol_source_options`, by the namelist importer's
#: aerosol-key sweep, and by the mp=28 admission test -- one table, so a
#: refusal message and a namelist refusal cannot drift apart.
MP28_AEROSOL_SOURCE_OPTIONS: dict[str, tuple[int, str, str]] = {
    "aer_init_opt": (
        0,
        "Registry/Registry.EM_COMMON:2656",
        "aer_init_opt=1 (climo) and 2 (first guess) both consume the "
        "water/ice-friendly aerosol arrays real.exe interpolates from a "
        "metgrid WIF stream (dyn_em/module_initialize_real.F:2327-2732 3-D, "
        ":4499-4653 2-D); "
        "ArWen has NO WIF metgrid ingest, so there is nothing for either "
        "branch to read and no oracle fixture covers them",
    ),
    "wif_input_opt": (
        0,
        "Registry/registry.new3d_wif:17",
        "wif_input_opt=1 activates the use_wif_input package (13 monthly "
        "WIF levels per species, Registry/registry.new3d_wif:80) and "
        "wif_input_opt=2 additionally allocates the black-carbon scalar "
        "qnbca (:82); ArWen implements NEITHER -- there is no WIF metgrid "
        "ingest and no nbca species anywhere in the mp=28 port",
    ),
}


def validate_aerosol_source_options(cfg: RunConfig) -> None:
    """Fail closed on any mp=28 aerosol-source selector ArWen cannot honour.

    Both selectors default to WRF's Registry default 0, which is also the
    only implemented value, so this refuses rather than reinterprets.  The
    check is unconditional on ``mp_physics``: under any other scheme the two
    keys are inert in WRF as well, and accepting a nonzero inert value here
    would let a configuration that MEANS something under mp=28 be carried
    silently into an mp=28 restart or nest.
    """
    for name, (only, citation, why) in MP28_AEROSOL_SOURCE_OPTIONS.items():
        value = getattr(cfg, name)
        if value == only:
            continue
        raise NotImplementedError(
            f"{name}={value!r} is not implemented; ArWen honours "
            f"{name}={only!r} only (WRF Registry default, {citation}). "
            f"{why}. "
            + MP28_AEROSOL_SOURCE_DEVIATION
        )

_KNOWN_TABLES = ("grid", "dynamics", "run")

def load_config(path: str | Path) -> RunConfig:
    import io

    from gpuwm.config_authority import read_config_authority

    authority = read_config_authority(path)
    raw = tomllib.load(io.BytesIO(authority.payload))
    unknown_tables = [name for name in raw if name not in _KNOWN_TABLES]
    if unknown_tables:
        raise ValueError(
            f"unknown table(s)/top-level key(s) {unknown_tables} in config "
            f"file {path}; known tables: {list(_KNOWN_TABLES)}."
        )
    known_keys = {f.name for f in fields(RunConfig)}
    merged: dict = {}
    key_table: dict[str, str] = {}
    for table in _KNOWN_TABLES:
        entries = raw.get(table, {})
        unknown_keys = [key for key in entries if key not in known_keys]
        if unknown_keys:
            raise ValueError(
                f"unknown key(s) {unknown_keys} in table [{table}] of config "
                f"file {path}; known keys: {sorted(known_keys)}."
            )
        for key in entries:
            if key in key_table:
                raise ValueError(
                    f"duplicate key {key!r} appears in both "
                    f"[{key_table[key]}] and [{table}] of config file "
                    f"{path}; each key belongs in exactly one table."
                )
            key_table[key] = table
        merged.update(entries)
    return validate_run_config(RunConfig(**merged))


#: SASE's structural requirements: attribute -> (admitted value, why).
#:
#: These are the closure's ACTUAL dependencies, and each one is a thing
#: the code reads or a thing that would be double-counted -- not a record
#: of which suite it happened to be smoke-tested with.  The lane that
#: built SASE admitted exactly one physics combination (one microphysics
#: id, one radiation id, one land-surface id) because a development lane
#: wants its evidence ladder narrow.  Those three are NOT dependencies:
#: the closure never reads a hydrometeor tendency, a radiative flux or a
#: soil layer, and pinning them would refuse combinations that work while
#: claiming a coupling that does not exist.  Re-scoping the gate to the
#: real dependency set is what makes the scheme experimentable; every
#: genuine constraint below is kept, and the ones that are about
#: correctness (double-counted mixing, the missing surface fluxes) refuse
#: rather than warn.
_SASE_REQUIREMENTS: tuple[tuple[str, object, str], ...] = (
    ("km_opt", 0,
     "SASE computes its own horizontal mixing from the closure's own "
     "diffusivities, so a km_opt mixing operator would double-count it"),
    ("khdif", 0.0,
     "constant-K diffusion may not silently stack on the SASE mixing"),
    ("kvdif", 0.0,
     "constant-K diffusion may not silently stack on the SASE mixing"),
    ("bldt", 0.0,
     "SASE produces a w tendency that is rebuilt every step rather than "
     "carried across a PBL call interval, so it must run every step"),
)


def validate_sase_config(cfg: RunConfig) -> None:
    """Admission for the SASE closure and its three knobs.

    Warn-not-block applies to *maturity*, never to coherence: an
    experimental scheme still refuses a configuration it cannot honestly
    execute.  What is refused here is only what the closure genuinely
    needs -- see :data:`_SASE_REQUIREMENTS` for why the development
    lane's wider whitelist is not reproduced.
    """
    sase = cfg.bl_pbl_physics == SASE_PBL_SCHEME
    if sase:
        for name, admitted, why in _SASE_REQUIREMENTS:
            value = getattr(cfg, name)
            if value != admitted:
                raise ValueError(
                    f"bl_pbl_physics={SASE_PBL_SCHEME} (SASE) is not "
                    f"admitted with {name}={value!r}: {why}. Set "
                    f"{name}={admitted!r}.")
        if not cfg.sf_sfclay_physics:
            # Not a taste question: the closure reads u*, the heat and
            # moisture fluxes and the gust-corrected wind speed from the
            # surface layer, and its lower boundary condition is those
            # four fields.  With the slot off they do not exist.
            raise ValueError(
                f"bl_pbl_physics={SASE_PBL_SCHEME} (SASE) requires a "
                "surface-layer scheme (sf_sfclay_physics != 0): the "
                "closure's lower boundary condition is the surface "
                "layer's friction velocity, heat and moisture fluxes and "
                "gust-corrected wind speed. Select sf_sfclay_physics=1 "
                "(revised MM5) or 91 (classic MM5), which are the "
                "surface layers SASE's registry option declares.")
        if cfg.sf_sfclay_physics == MYNN_SFCLAY_SCHEME:
            # The refusal belongs to the MYNN surface layer, not to SASE.
            # WRF v4.6.1 admits sf_sfclay_physics=5 with the MYNN PBL or
            # with no PBL at all and nothing else (the 16-cell matrix at
            # phys/module_physics_init.F:3699-3704,3837-3839, which the
            # registry publishes under
            # authority.wrf_v461_compatibility_matrix and pins in
            # tests/test_physics_registry.py).  SASE is neither, so the
            # pair is outside the only compatibility statement either
            # scheme has.  The registry refused it all along; this is the
            # loader agreeing, which is what
            # tests/test_authority_agreement.py exists to require.
            raise ValueError(
                f"bl_pbl_physics={SASE_PBL_SCHEME} (SASE) is not admitted "
                f"with sf_sfclay_physics={MYNN_SFCLAY_SCHEME} (MYNN "
                "surface layer). WRF v4.6.1 admits that surface layer "
                "only with the MYNN PBL or with no PBL, and SASE is "
                "neither; no evidence covers the pairing. Select "
                "sf_sfclay_physics=1 (revised MM5) or 91 (classic MM5), "
                "which are the surface layers SASE's registry option "
                "declares.")
        if not cfg.moist:
            raise ValueError(
                f"bl_pbl_physics={SASE_PBL_SCHEME} (SASE) requires "
                "moist=true: the closure mixes water vapour, cloud water "
                "and cloud ice alongside potential temperature and forms "
                "its stability from the saturated Brunt-Vaisala "
                "frequency.")
        if cfg.nz > SASE_MAX_NZ:
            # A compile-time bound, not a policy: the vertical solve keeps
            # three FP64 columns of this depth in per-thread local memory.
            raise ValueError(
                f"bl_pbl_physics={SASE_PBL_SCHEME} (SASE) is limited to "
                f"nz <= {SASE_MAX_NZ}, got nz={cfg.nz}: the closure's "
                "implicit vertical solve carries its tridiagonal columns "
                "in per-thread local memory at that fixed depth.")
    # The three knobs are fail-closed on their NON-default value only, so
    # every existing configuration keeps validating unchanged.
    for name, default, what in (
            ("sase_flux_diag", False,
             "the split subgrid-flux diagnostic records the SASE venting "
             "and K_v channels, which no other turbulence path computes"),
            ("sase_moist_n2", True,
             "the moist-N2 substitution it disables exists only inside "
             "the SASE closure, so switching it off elsewhere would "
             "disable nothing"),
            ("sase_stable_dissipation", False,
             "the stable-limb dissipation coefficient it decouples lives "
             "in the SASE analytic decay substep, so setting it "
             "elsewhere would decouple nothing"),
            ("sase_additive_dissipation", False,
             "the additive e^{3/2} dissipation channel it enables lives "
             "in the SASE analytic decay substep, so setting it "
             "elsewhere would add nothing")):
        value = getattr(cfg, name)
        if type(value) is not bool:
            raise ValueError(f"{name} must be boolean, got {value!r}.")
        if value != default and not sase:
            raise ValueError(
                f"{name}={value!r} requires bl_pbl_physics="
                f"{SASE_PBL_SCHEME} (SASE), got "
                f"bl_pbl_physics={cfg.bl_pbl_physics}: {what} -- a key "
                "that names a seam this run does not have would read as "
                "a setting that took effect.")


#: ``name -> (predicate, what the value has to be)`` for the dynamics
#: coefficients.  Each predicate is the range the *scheme* is defined
#: on, not a taste bound: outside it the solver is not being used, it is
#: being misused, and the answer that comes back is confident garbage.
#:
#: These keys reached ``forecast validity PASS`` at exit 0 through both
#: loaders in v1.4.0 -- ``epssm = 5.0`` moved 2 m temperature 2.2 K in
#: two simulated hours, and ``dampcoef = -5.0`` turned the upper damping
#: layer into an energy source.  Every value the wizard emits, every
#: value in configs/, and every WRF Registry default passes.
_DYNAMICS_RANGES: dict[str, tuple] = {
    # Forward off-centring weight of the vertically implicit acoustic
    # solve (module_small_step_em.F): a weight in (0, 1].
    "epssm": (lambda v: 0.0 < v <= 1.0,
              "an acoustic off-centring weight in (0, 1] "
              "(WRF Registry default 0.1)"),
    # 3-D divergence damping coefficient; 0 disables it.
    "smdiv": (lambda v: v >= 0.0,
              "a non-negative divergence-damping coefficient "
              "(WRF Registry default 0.1)"),
    # External-mode divergence damping (mudf); 0 disables it.
    "emdiv": (lambda v: v >= 0.0,
              "a non-negative external-mode damping coefficient "
              "(WRF Registry default 0.01)"),
    # Rayleigh/implicit-w damping strength.  Negative is ANTI-damping:
    # the layer injects the energy it exists to remove.
    "dampcoef": (lambda v: v >= 0.0,
                 "a non-negative damping rate in 1/s "
                 "(WRF Registry default 0.2); a negative coefficient is "
                 "anti-damping, and the layer adds energy instead of "
                 "removing it"),
    # Depth of that layer below the model top; _damp_factors divides by
    # it, so 0 is a division by zero and negative inverts the profile.
    "zdamp": (lambda v: v > 0.0,
              "a positive damping-layer depth in m "
              "(WRF Registry default 5000)"),
    # Smagorinsky constant.
    "c_s": (lambda v: v > 0.0,
            "a positive Smagorinsky constant (WRF default 0.25)"),
    # Model-top height: the damping profile is measured down from it.
    "ztop": (lambda v: v > 0.0,
             "a positive model-top height in m"),
    "p_surf": (lambda v: v > 0.0,
               "a positive reference surface pressure in Pa"),
    "base_temp": (lambda v: v > 0.0,
                  "a positive base-state reference temperature in K"),
    # Exponential lateral-sponge exponent (WRF spec_exp).
    "spec_exp": (lambda v: v >= 0.0,
                 "a non-negative lateral-sponge exponent "
                 "(WRF Registry default 0.0)"),
    "diff_6th_factor": (lambda v: v >= 0.0,
                        "a non-negative 6th-order diffusion factor "
                        "(WRF Registry default 0.12)"),
}

#: Integer dynamics selectors, with the values this tree IMPLEMENTS.
#: A selector that is a real WRF choice but unimplemented here is the
#: worst of the three cases: every damping site tests ``damp_opt == 3``,
#: so ``damp_opt = 2`` used to run with the damping layer switched off
#: while the config said it was on.
_DYNAMICS_CHOICES: dict[str, tuple] = {
    "damp_opt": ((0, 3),
                 "0 (no upper damping) or 3 (the Klemp-Dudhia-Hassiotis "
                 "implicit w-only damper); WRF's damp_opt 1 (diffusive) "
                 "and 2 (Rayleigh relaxation to the base state) are not "
                 "implemented here and would silently run undamped"),
    "w_damping": ((0, 1),
                  "0 (off) or 1 (WRF's per-stage vertical-velocity "
                  "limiter)"),
}


def _validate_dynamics_coefficients(cfg: RunConfig) -> None:
    """Refuse dynamics coefficients outside the range they are defined on.

    The value checks the fleet found missing.  Kept in one table so the
    next coefficient added to :class:`RunConfig` has an obvious place to
    declare what it means, rather than joining the set of keys that
    reach the solver unexamined.
    """

    for name, (ok, wanted) in _DYNAMICS_RANGES.items():
        value = float(getattr(cfg, name))
        if not math.isfinite(value) or not ok(value):
            raise ValueError(
                f"{name} = {value!r} must be {wanted}.")
    for name, (allowed, wanted) in _DYNAMICS_CHOICES.items():
        value = getattr(cfg, name)
        if value not in allowed:
            raise ValueError(f"{name} must be {wanted}, got {value!r}.")
    if not isinstance(cfg.time_step_sound, int) \
            or isinstance(cfg.time_step_sound, bool) \
            or cfg.time_step_sound < 1:
        raise ValueError(
            "time_step_sound must be a positive number of acoustic "
            "substeps per dynamics step (WRF Registry default 4), got "
            f"{cfg.time_step_sound!r}.")
    # The damping layer is measured DOWN from the model top; a layer
    # deeper than the column damps every level, including the ground.
    if cfg.damp_opt == 3 and cfg.zdamp >= cfg.ztop > 0.0:
        raise ValueError(
            f"zdamp = {cfg.zdamp} is the depth of the upper damping "
            f"layer below the model top ztop = {cfg.ztop}, so it has to "
            "be smaller than it; as written the damper covers the whole "
            "column.")


#: PBL schemes that produce horizontal mixing of their OWN.
#:
#: The whole content of the ``km_opt = 0`` question: with ``km_opt = 0``
#: the dycore runs no horizontal mixing operator, so whether the run has
#: horizontal mixing at all depends entirely on whether something else
#: produces it.  Exactly one scheme in this tree does -- SASE, whose
#: governed horizontal diffusivity serves its stress, its subgrid-energy
#: transport and its scalar channel.  YSU, MYNN and PBL-off are all
#: vertical-only, and on any of them ``km_opt = 0`` means no horizontal
#: mixing operator anywhere.
HMIX_PRODUCING_PBL_SCHEMES: tuple[int, ...] = (SASE_PBL_SCHEME,)


def km_opt_zero_producer(cfg: RunConfig) -> str | None:
    """What supplies horizontal mixing at ``km_opt = 0``, or None."""
    if cfg.bl_pbl_physics == SASE_PBL_SCHEME:
        return (f"bl_pbl_physics={SASE_PBL_SCHEME} (SASE), whose closure "
                "supplies the mixing the km_opt operator would otherwise "
                "apply")
    return None


def validate_km_opt(cfg: RunConfig) -> None:
    """Admission for ``km_opt``, shared by the loaders and the dycore.

    ONE function so the loader's refusal and the dycore's refusal cannot
    drift: the dycore's is the fail-closed one (it is what actually
    decides whether a mixing operator runs), and a config that got past
    the loader must get past it too.

    ``km_opt = 0`` is admitted on two distinct grounds, and they are not
    the same kind of thing:

    * A PBL scheme in :data:`HMIX_PRODUCING_PBL_SCHEMES` supplies the
      horizontal mixing itself.  Nothing is missing, so nothing is
      acknowledged -- the operator is redundant, and running it too
      would double-count.
    * :data:`KM_OPT_ZERO_ACK`, written out in full by the user.  Here
      the horizontal mixing operator really is absent and the run really
      does have none, which is the RESEARCH CONTROL the acknowledgement
      exists for and the thing the default refusal exists to prevent
      happening by accident.  See :data:`KM_OPT_ZERO_ACK`.
    """
    ack = cfg.km_opt_zero_acknowledgement
    producer = km_opt_zero_producer(cfg)
    if not isinstance(ack, str):
        raise ValueError(
            "km_opt_zero_acknowledgement must be the exact string "
            f"{KM_OPT_ZERO_ACK!r} or absent, got {ack!r}.")
    if ack and ack != KM_OPT_ZERO_ACK:
        raise ValueError(
            f"km_opt_zero_acknowledgement={ack!r} is not the "
            f"acknowledgement id. Write it exactly: "
            f"km_opt_zero_acknowledgement = {KM_OPT_ZERO_ACK!r}. It is a "
            "literal id and not a boolean so that it cannot be reached "
            "by a stray `= true`.")
    if cfg.km_opt not in (0, 1, 2, 3, 4):
        # Checked BEFORE the acknowledgement's placement, so an
        # unimplemented selector reports itself rather than reporting
        # that the acknowledgement is in the wrong place: the id opens
        # km_opt = 0 and nothing else, and cannot shadow this.
        raise ValueError(
            f"km_opt must be 1 (constant K via khdif/kvdif), 2 (1.5-order "
            f"prognostic TKE), 3 (3-D Smagorinsky), or 4 "
            f"(2-D Smagorinsky), got {cfg.km_opt}. 0 is admitted with "
            f"bl_pbl_physics={SASE_PBL_SCHEME} (SASE), whose closure "
            "supplies the mixing the km_opt operator would otherwise "
            f"apply, or with km_opt_zero_acknowledgement = "
            f"{KM_OPT_ZERO_ACK!r}.")
    if ack and cfg.km_opt != 0:
        raise ValueError(
            f"km_opt_zero_acknowledgement is set with km_opt={cfg.km_opt}, "
            "which runs a horizontal mixing operator; the acknowledgement "
            "admits km_opt = 0 and acknowledges nothing here. A key that "
            "names a seam this run does not have would read as a setting "
            "that took effect.")
    if ack and producer is not None:
        raise ValueError(
            "km_opt_zero_acknowledgement is set with "
            f"{producer.split(',')[0]}, which already supplies the "
            "horizontal mixing; km_opt = 0 is admitted here without any "
            "acknowledgement, and the id would record an absence this "
            "run does not have.")
    if cfg.km_opt == 3 and cfg.bl_pbl_physics != 0:
        # The registry says the same thing from the other side
        # (physics_registry_v2.json#/components/turbulence/options/
        # smagorinsky-3d declares required_settings bl_pbl_physics=0 and
        # requires_components pbl=[off]), and until this refusal existed
        # the two authorities disagreed about 3,360 of the 46,080
        # component combinations tests/test_authority_agreement.py
        # enumerates: the registry refused and validate_run_config said
        # OK.  The registry was right.  What makes km_opt=3 different
        # from km_opt=4 is that it computes a genuinely SEPARATE
        # vertical exchange pair (kmv/khv) rather than reusing the
        # horizontal one, and that pair is applied by
        # vertical_diffusion_2, which is PBL-off-gated exactly as the
        # km_opt=2 branch below describes.  So with a PBL scheme on, the
        # vertical half of the closure the user selected does not run at
        # all and the selection quietly means something narrower than
        # its name.  Refusing is the no-silent-override rule.
        raise ValueError(
            "km_opt=3 (3-D Smagorinsky) is admitted with "
            "bl_pbl_physics=0 only. Its vertical exchange pair "
            "(kmv/khv) is applied by vertical_diffusion_2, which is "
            "PBL-off-gated, so with a PBL scheme on only the horizontal "
            "half of the closure would run and the run would not be the "
            "3-D closure it names. Select bl_pbl_physics=0 for an LES "
            "domain, or km_opt=4 (2-D Smagorinsky), which is the "
            "horizontal-only closure every PBL-on template pins.")
    if cfg.km_opt == 2:
        if cfg.bl_pbl_physics != 0:
            raise ValueError(
                "km_opt=2 (prognostic TKE) is admitted with "
                "bl_pbl_physics=0 only: WRF evolves TKE with the PBL on, "
                "but that combination has no vertical TKE mixing "
                "(vertical_diffusion_2 is PBL-off-gated) and is not "
                "ported; select bl_pbl_physics=0 for an LES domain, or "
                "km_opt=4 (2-D Smagorinsky), which is the horizontal-only "
                "closure every PBL-on template pins. km_opt=3 is NOT an "
                "alternative with a PBL scheme on: it is refused three "
                "lines above for the same PBL-off gate."
            )
        # km_opt=2 on a NEST child is no longer refused here.  This
        # function sees one domain at a time and cannot see the parent,
        # and the parent is what decides whether the question is even
        # live: WRF gives tke no ``i`` (nest-interpolation) and no ``f``
        # (feedback) Registry flag (Registry.EM_COMMON:312), so a child
        # cold-starts its own TKE and never returns it.  Under a parent
        # that carries no TKE at all there is nothing to interpolate and
        # nothing to feed back, and that case has now been run and scored
        # (7 h, 250 m child under a km_opt=4 parent, status PASS; see
        # docs/superpowers/receipts/les/nested-les-km2-2026-08-02.md).
        # The residual case -- a km_opt=2 child under a km_opt=2 parent,
        # where the parent really does hold a field WRF declines to hand
        # down -- is refused in gpuwm.experiment, which is the only place
        # that knows the parent.  A single-domain RunConfig cannot tell
        # the two apart, so refusing here would refuse the measured case
        # along with the unmeasured one.
    if cfg.km_opt in (1, 2, 3, 4):
        return
    if producer is None and ack != KM_OPT_ZERO_ACK:
        raise ValueError(
            f"km_opt = 0 runs NO horizontal mixing operator, and "
            f"bl_pbl_physics={cfg.bl_pbl_physics} produces none of its "
            "own, so this run would damp grid-scale horizontal structure "
            "with nothing but the sixth-order numerical filter. That is "
            "refused by default because it is what a mis-set switch "
            "looks like. It is also a legitimate research control -- it "
            "is WRF's own diff_opt = 0, and it is the only way to vary "
            "the PBL closure while holding the horizontal mixing fixed "
            "at none. To run it deliberately, write the acknowledgement "
            f"out in full: km_opt_zero_acknowledgement = "
            f"{KM_OPT_ZERO_ACK!r}. Otherwise set km_opt = 4 (2-D "
            f"Smagorinsky) or 1 (constant K via khdif/kvdif), or select "
            f"bl_pbl_physics={SASE_PBL_SCHEME} (SASE), which supplies "
            "horizontal mixing from its own closure.")


#: Stability limit of an EXPLICIT horizontal Laplacian on the shortest
#: representable mode.  Forward-in-time diffusion of a 2-grid-interval
#: wave multiplies it by ``1 - 4*K*dt/dx^2`` each step, so ``K*dt/dx^2``
#: above 1/4 flips the sign and above 1/2 grows the wave: a diffusion
#: operator that amplifies what it exists to remove.
EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT = 0.25

#: The second threshold on the same per-step factor, and a different
#: statement: between 1/4 and 1/2 the factor is negative but still inside
#: the unit circle, so the 2-grid-interval mode flips sign each step and
#: shrinks; past 1/2 its magnitude exceeds one and the mode GROWS without
#: bound.  Both are reported, because "inverts and decays" and "amplifies
#: until the run aborts" deserve different sentences.
EXPLICIT_HORIZONTAL_DIFFUSION_GROWTH_LIMIT = 0.5

#: The mixing selectors that build SEPARATE horizontal and vertical
#: exchange coefficients when ``mix_isotropic = 0``: 2 (prognostic TKE)
#: and 3 (3-D Smagorinsky).  1 (constant K) and 4 (2-D Smagorinsky) build
#: one coefficient on the horizontal grid spacing and are not exposed.
_ANISOTROPIC_LENGTH_KM_OPTS = (2, 3)


def selects_anisotropic_w_mixing(*, km_opt: int, mix_isotropic: int,
                                 dx: float, dy: float) -> bool:
    """Is this domain on the exposed path, depth aside?

    The SELECTOR half of the criterion, asked without a layer depth.
    :func:`anisotropic_w_mixing_ratio` folds the two halves together and
    answers ``None`` for both "not on the path" and "on the path but the
    depth could not be resolved", and those are opposite readings: the
    first is genuinely nothing to say, the second is the criterion's own
    subject with the number missing.  Splitting the selector out is what
    lets :func:`anisotropic_w_mixing_advice` tell them apart.
    """

    if int(km_opt) not in _ANISOTROPIC_LENGTH_KM_OPTS:
        return False
    if int(mix_isotropic) != 0:
        return False
    return min(float(dx), float(dy)) > 0.0


def anisotropic_w_mixing_ratio(*, km_opt: int, mix_isotropic: int,
                               mix_upper_bound: float, dx: float, dy: float,
                               dz_max: float) -> float | None:
    """Worst-case ``K*dt/dx^2`` the horizontal w operator can be handed.

    With ``mix_isotropic = 0`` the mixing lengths are per-axis, so the
    VERTICAL exchange coefficient is both built and capped on the layer
    depth -- ``xkm_v <= mix_upper_bound * dz^2 / dt`` -- while the
    horizontal diffusion of ``w`` differences over the horizontal grid
    spacing.  Nothing in that chain compares the coefficient against the
    horizontal spacing, so the largest ratio the operator can be asked to
    integrate is

        mix_upper_bound * (dz_max / min(dx, dy))^2

    which is ``dt``-invariant: the cap carries ``1/dt`` and the ratio
    carries ``dt``, so a shorter step buys nothing.  Where the ratio
    exceeds :data:`EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT` the operator can
    amplify a 2-grid-interval mode in ``w`` instead of damping it.

    Returns ``None`` when the configuration is not on that path at all --
    an isotropic mixing length, or a selector that builds a single
    coefficient on the horizontal spacing -- so a caller can distinguish
    "not applicable" from "applicable and small".

    ``None`` ALSO comes back for a domain that IS on the path but whose
    ``dz_max`` is non-finite, because no number exists to return and
    fabricating one would be worse.  That case is not silence: it is the
    subject of the criterion with the depth missing, and
    :func:`anisotropic_w_mixing_advice` separates it out (via
    :func:`selects_anisotropic_w_mixing`) and says so in words.  A caller
    reading this function's ``None`` alone must not read it as a pass.
    """

    if not selects_anisotropic_w_mixing(
            km_opt=km_opt, mix_isotropic=mix_isotropic, dx=dx, dy=dy):
        return None
    if not math.isfinite(float(dz_max)):
        return None
    horizontal = min(float(dx), float(dy))
    return float(mix_upper_bound) * (float(dz_max) / horizontal) ** 2


#: The phrase every no-number mixing advisory carries, so a reader
#: skimming a door and a test asserting on one are matching the same
#: string.  It is upper-case for the same reason the tier verbs are: the
#: distinction a reader must not miss is that there is no ratio here, and
#: that absence is the finding rather than the absence of one.
UNRESOLVED_ANISOTROPIC_DEPTH_MARK = "NO LAYER DEPTH COULD BE RESOLVED"


def _unresolved_depth_advice(*, where: str, km_opt: int, horizontal: float,
                             ladder: str) -> str:
    """The advisory for an exposed domain whose depths will not resolve.

    Separate wording from the over-the-limit sentence because it is a
    separate statement: that one reports a number against a threshold,
    this one reports that no number can be produced at all.  Both remedies
    are named, and they are not the same pair -- lowering
    ``mix_upper_bound`` cannot help when the quantity it multiplies is
    unknown, so what this one offers instead is making the ladder
    resolvable.
    """

    provenance = f" ({ladder})" if ladder else ""
    return (
        f"{where} runs km_opt = {int(km_opt)} with mix_isotropic = 0 at "
        f"dx = {horizontal:g} m, so its horizontal mixing of w is handed "
        f"a coefficient built on the LAYER DEPTH -- but "
        f"{UNRESOLVED_ANISOTROPIC_DEPTH_MARK} for this grid"
        f"{provenance}, so mix_upper_bound*(dz_max/dx)^2 cannot be "
        f"evaluated and the criterion has no number to report. The "
        f"absence of a ratio is NOT a pass: this domain is on the exposed "
        f"path and nothing here can say whether it is over the limit "
        f"{EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT} or under it. Declare "
        f"eta_levels and p_top explicitly, or bring the model top inside "
        f"the analytic base state's representable range, so the depths "
        f"resolve and the criterion can run; or set mix_isotropic = 1 on "
        f"this domain, which builds one length from (dx*dy*dz)^(1/3) and "
        f"takes it off the per-axis path entirely, making the question "
        f"moot. ADVISORY, not a refusal: the load and the run proceed.")


def anisotropic_w_mixing_advice(*, where: str, km_opt: int,
                                mix_isotropic: int, mix_upper_bound: float,
                                dx: float, dy: float, dz_max: float,
                                ladder: str = "",
                                ) -> tuple[float | None, str | None]:
    """``(ratio, one-sentence advisory)`` for the exposed-mixing check.

    One wording, every door.  :func:`warn_anisotropic_w_mixing` prints
    it at config load; ``gpuwm check`` repeats it in its advisory list
    (``gpuwm.core.preflight.anisotropic_w_mixing_advisories``), because
    the load-time line scrolls past hours before the run that pays for
    it and the preflight report is where a reader is actually looking.

    The advisory is ``None`` when the configuration is not on the
    exposed path, or is on it and under the limit.  The ratio comes back
    either way, so a caller can record the number whether or not there
    was anything to say.

    A NON-FINITE ``dz_max`` ON AN OTHERWISE-EXPOSED DOMAIN GETS ITS OWN
    ADVISORY, and the ratio stays ``None``.  This is the third outcome,
    added 2026-08-09: the domain selected the per-axis mixing length, but
    its layer depths could not be resolved, so the criterion has a
    subject and no number.  Before this branch existed the ``None`` from
    :func:`anisotropic_w_mixing_ratio` collapsed that case back into "not
    applicable" and every door went quiet -- the same silence-reads-as-a-
    pass hole the resolved-ladder work had just closed one layer up.  No
    number is invented: the sentence says the depth is unresolvable, why
    (``ladder`` carries the provenance), and what to declare instead.

    The sentence is TIERED, because the criterion has two thresholds and
    they mean different things: above
    :data:`EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT` the 2-grid-interval mode
    inverts each step and still decays; above
    :data:`EXPLICIT_HORIZONTAL_DIFFUSION_GROWTH_LIMIT` it grows.  A
    reader who sees one number against one limit cannot tell 0.3 from
    4.2, and 4.2 is the tier that aborted a run.

    ``ladder`` is a provenance clause for ``dz_max``, appended verbatim
    when non-empty (``gpuwm.experiment.ExposedMixing.ladder`` supplies
    it).  A depth the caller RESOLVED for a config that wrote no eta
    interfaces must not read the same as one the config declared, and
    the number is otherwise indistinguishable.
    """

    ratio = anisotropic_w_mixing_ratio(
        km_opt=km_opt, mix_isotropic=mix_isotropic,
        mix_upper_bound=mix_upper_bound, dx=dx, dy=dy, dz_max=dz_max)
    if ratio is None and not math.isfinite(float(dz_max)) \
            and selects_anisotropic_w_mixing(
                km_opt=km_opt, mix_isotropic=mix_isotropic, dx=dx, dy=dy):
        return None, _unresolved_depth_advice(
            where=where, km_opt=km_opt,
            horizontal=min(float(dx), float(dy)), ladder=ladder)
    if ratio is None or ratio <= EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT:
        return ratio, None
    horizontal = min(float(dx), float(dy))
    admitted = (EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT
                * (horizontal / float(dz_max)) ** 2)
    if ratio > EXPLICIT_HORIZONTAL_DIFFUSION_GROWTH_LIMIT:
        tier = (
            f"AMPLIFIES a 2dx mode instead of damping it -- past "
            f"{EXPLICIT_HORIZONTAL_DIFFUSION_GROWTH_LIMIT} the per-step "
            f"factor 1 - 4*K*dt/dx^2 has magnitude above one, so "
            f"grid-scale vertical velocity grows out of the field the "
            f"operator exists to smooth until a health bound stops the "
            f"run")
    else:
        tier = (
            f"INVERTS a 2dx mode every step instead of damping it; its "
            f"magnitude still decays below "
            f"{EXPLICIT_HORIZONTAL_DIFFUSION_GROWTH_LIMIT}, which is "
            f"where the mode starts to grow instead")
    provenance = f" ({ladder})" if ladder else ""
    advice = (
        f"{where} runs km_opt = {int(km_opt)} with mix_isotropic = 0 at "
        f"dx = {horizontal:g} m over base-state layers up to "
        f"{float(dz_max):.1f} m deep{provenance}: "
        f"mix_upper_bound*(dz_max/dx)^2 = "
        f"{ratio:.3g} exceeds the explicit horizontal diffusion limit "
        f"{EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT}, so the horizontal "
        f"mixing of w {tier}. Set mix_isotropic = 1 on this domain, or "
        f"lower mix_upper_bound below {admitted:.3g}. ADVISORY, not a "
        f"refusal: {ratio:.3g} is the WORST case the cap admits and a "
        f"flow that never reaches it never sees this, which is why "
        f"trees above the limit have completed -- but above the limit "
        f"nothing guarantees the operator's own stability, and what it "
        f"costs when it does bite is an abort late in a long run rather "
        f"than a number you can inspect")
    return ratio, advice


def warn_anisotropic_w_mixing(*, where: str, km_opt: int, mix_isotropic: int,
                              mix_upper_bound: float, dx: float, dy: float,
                              dz_max: float, ladder: str = "") -> float | None:
    """Advise when the anisotropic mixing length exceeds the explicit limit.

    A warning and not a refusal, RE-RULED 2026-08-09 after the shipped
    configs were cleaned: the ratio is the WORST case the cap admits and
    the criterion overestimates what a flow reaches.  The receipt-backed
    leg of that is the 250 m nested tree, which completed five
    multi-hour runs at 0.702 -- 2.8x the limit -- with no sign of the
    failure mode.  Beside it sits one weaker observation, recorded in
    prose rather than in an instrument: the attempt #2b post-mortem
    notes 1.55 at the failing cell where the criterion read 4.23
    (``configs/les_tornado_100m_mayfield_20211210_attempt3.toml:232``,
    the only place that number is written down).  Refusing at load would
    also make the frozen records under ``configs/`` unloadable, which is
    the same as deleting the ability to reproduce a committed crash.

    What it is not is a taste bound -- above the limit the operator's
    own stability is no longer guaranteed by anything, and the failure
    mode is grid-scale vertical velocity that grows out of a field which
    should be being smoothed.  So it stays advisory, it says the number
    and the tier, and ``tests/test_shipped_configs_mixing_stability.py``
    keeps the shipped set out of the exposed state entirely.

    Returns the computed ratio so a caller can record the number whether
    or not it warned.  ``None`` covers two cases and a caller must not
    conflate them: not applicable, and applicable with an unresolvable
    layer depth -- the second one warns, and the warning is the only
    place that distinction is visible from here.
    """

    ratio, advice = anisotropic_w_mixing_advice(
        where=where, km_opt=km_opt, mix_isotropic=mix_isotropic,
        mix_upper_bound=mix_upper_bound, dx=dx, dy=dy, dz_max=dz_max,
        ladder=ladder)
    if advice is None:
        return ratio
    from gpuwm.explain import warn
    warn(
        advice,
        why="With mix_isotropic = 0 the vertical exchange coefficient is "
            "built and capped on the LAYER DEPTH (xkm_v <= "
            "mix_upper_bound*dz^2/dt) and is then the coefficient the "
            "horizontal diffusion of w differences over the HORIZONTAL "
            "spacing. Nothing compares the two, so the reachable "
            "K*dt/dx^2 is mix_upper_bound*(dz_max/dx)^2 -- independent "
            "of dt, because the cap carries 1/dt and the ratio carries "
            "dt. An explicit Laplacian multiplies a 2-grid-interval mode "
            "by 1 - 4*K*dt/dx^2 per step, so beyond 1/4 the sign flips "
            "and beyond 1/2 the mode grows. mix_isotropic = 1 builds one "
            "length from (dx*dy*dz)^(1/3) and caps the vertical "
            "coefficient against the horizontal one, which is why it is "
            "the usual choice where layers are much deeper than the grid "
            "is wide. This is an advisory: the ratio is the worst case "
            "the cap admits, not a value this flow is required to reach.")
    return ratio


def validate_run_config(cfg: RunConfig) -> RunConfig:
    """The RunConfig invariant battery, shared by BOTH loaders.

    Historically these checks lived inline in :func:`load_config`; the
    Phase-5 experiment path (gpuwm/experiment.py ``build_experiment``)
    constructs per-domain RunConfigs directly and MUST apply the same
    battery (p5t1 review F1) -- same checks, same messages, so a
    hand-typed ``mp_physics = 55`` or odd ``time_step_sound`` in an
    experiment TOML fails exactly as it always has on the legacy path.
    Returns ``cfg`` unchanged on success.
    """
    if cfg.time_step_sound % 2 != 0:
        raise ValueError(
            f"time_step_sound must be even, got {cfg.time_step_sound}: RK3 "
            "stage 2 runs time_step_sound//2 acoustic substeps of dt/"
            "time_step_sound, which only integrates to dt/2 for even values "
            "(WRF requires even time_step_sound)."
        )
    if cfg.nz < 4:
        raise ValueError(
            f"nz must be >= 4 (the vertical stencil width), got {cfg.nz}: "
            "the surface extrapolation coefficients (cf1/cf2/cf3) and the "
            "k=0 pressure-gradient stencil read the three lowest half "
            "levels."
        )
    if cfg.hybrid_opt not in (0, 1, 2):
        raise ValueError(
            f"hybrid_opt must be 0 or 1 (B(eta) = eta, the Phase 1 sigma "
            f"coordinate) or 2 (WRF v4 cubic-B hybrid), got "
            f"{cfg.hybrid_opt}."
        )
    if cfg.map_proj not in (0, 1, 2, 3):
        raise ValueError(
            f"map_proj must be 0 (idealized/none), 1 (Lambert conformal), "
            f"2 (polar stereographic), or 3 (Mercator), got {cfg.map_proj}."
        )
    # ---- WRF selector schema, then port readiness ------------------
    # Order matters.  A value outside gpuwm's schema (sf_surface_physics=7,
    # a RUC request carrying Noah's four layers) is nonsense in WRF terms
    # and is reported as such.  Only a schema-COHERENT request reaches
    # require_ready_wrf_physics, which then emits one complete receipt for
    # every unfinished component instead of failing on the first number.
    if cfg.sf_sfclay_physics not in SURFACE_LAYER_SCHEMES:
        raise ValueError(
            "sf_sfclay_physics must be 0 (off), 1 (revised MM5), 5 (MYNN), "
            f"or 91 (classic MM5), got {cfg.sf_sfclay_physics}."
        )
    if cfg.sf_surface_physics not in LAND_SURFACE_SCHEMES:
        raise ValueError(
            "sf_surface_physics must be 0 (none), 2 (Noah LSM), 3 (RUC "
            "LSM), or 4 (Noah-MP LSM), WRF numbering, got "
            f"{cfg.sf_surface_physics}."
        )
    # One implementation of the soil-geometry rule, not two.  This used to
    # repeat the table lookup inline while soil_layer_count did no lookup at
    # all, which meant a validated config and a code-built one disagreed
    # about the same question.  Calling the resolver is what makes them agree.
    soil_layer_count(cfg)
    if cfg.bl_pbl_physics not in PBL_SCHEMES:
        raise ValueError(
            f"bl_pbl_physics must be 0 (none), 1 (YSU), 5 (MYNN), "
            f"11 (Shin-Hong), or {SASE_PBL_SCHEME} (SASE, experimental), "
            f"got {cfg.bl_pbl_physics}."
        )
    validate_sase_config(cfg)
    require_ready_wrf_physics(
        mp_physics=cfg.mp_physics,
        sf_sfclay_physics=cfg.sf_sfclay_physics,
        bl_pbl_physics=cfg.bl_pbl_physics,
        sf_surface_physics=cfg.sf_surface_physics,
        num_soil_layers=cfg.num_soil_layers,
        # The grid width the readiness authority needs for the one blocker that
        # is about throughput rather than a missing branch.  nx*ny is the whole
        # grid: the land fraction is a property of the ingested landmask, not
        # of the configuration, so a fail-closed rail uses the worst case.
        columns=int(cfg.nx) * int(cfg.ny))
    if cfg.isftcflx not in (0, 1, 2):
        raise ValueError(
            "isftcflx must be 0 (standard MM5 water-point roughness), 1 "
            "(Garratt), or 2 (Donelan), got "
            f"{cfg.isftcflx}."
        )
    if cfg.iz0tlnd not in (0, 1, 2):
        raise ValueError(
            "iz0tlnd must be 0 (standard CZIL), 1 (Chen-Zhang thermal "
            "roughness), or 2 (fixed CZIL 0.1), got "
            f"{cfg.iz0tlnd}."
        )
    if cfg.opt_thcnd not in (1, 2):
        raise ValueError(
            "opt_thcnd must be 1 (Johansen soil thermal conductivity) or 2 "
            "(McCumber-Pielke, clamped for soiltyp 4), got "
            f"{cfg.opt_thcnd}."
        )
    if cfg.isfflx not in (0, 1, 2):
        raise ValueError(
            "isfflx must be 0 (prescribed tke_drag_coefficient/"
            "tke_heat_flux under the PBL-off Smagorinsky path; surface-"
            "scheme fluxes off), 1 (surface-scheme fluxes on), or 2 "
            "(surface-scheme drag/moisture with prescribed tke_heat_flux), "
            f"got {cfg.isfflx}."
        )
    _prescribed_flux_consumer = (
        cfg.km_opt in (2, 3, 4) and cfg.bl_pbl_physics == 0)
    if cfg.isfflx == 0 and cfg.sf_sfclay_physics == 0 \
            and not _prescribed_flux_consumer:
        raise ValueError(
            "isfflx=0 has no consumer when sf_sfclay_physics=0 unless the "
            "PBL-off turbulence path (km_opt=2/3/4, bl_pbl_physics=0) is "
            "active to take the prescribed tke_drag_coefficient/"
            "tke_heat_flux forcing; gpuwm otherwise implements the gate "
            "in its MM5 and MYNN surface-layer paths."
        )
    if cfg.isfflx == 2 and not _prescribed_flux_consumer:
        raise ValueError(
            "isfflx=2 (prescribed tke_heat_flux) is consumed only by "
            "WRF's diff_opt=2 vertical_diffusion_2 path, which gpuwm "
            "runs under km_opt=2/3/4 with bl_pbl_physics=0 "
            "(module_diffusion_em.F:4286-4305); enable that path or "
            "choose isfflx 0/1."
        )
    if cfg.o3input not in (0, 2):
        raise ValueError(
            "o3input must be 0 (legacy-RRTMG wrapper O3DATA) or 2 "
            f"(CAM climatology), got {cfg.o3input}."
        )
    if cfg.use_mp_re not in (0, 1):
        raise ValueError(
            "use_mp_re must be 0 (legacy-RRTMG calculated radii) or 1 "
            f"(use the WRF microphysics scheme table), got {cfg.use_mp_re}."
        )
    if (not math.isfinite(cfg.seaice_albedo_default)
            or not 0.0 <= cfg.seaice_albedo_default <= 1.0):
        raise ValueError(
            "seaice_albedo_default must be finite and in [0, 1], got "
            f"{cfg.seaice_albedo_default}."
        )
    if (cfg.seaice_albedo_default != 0.65
            and cfg.sf_surface_physics != 3):
        raise ValueError(
            "a nondefault seaice_albedo_default is implemented only by "
            "RUC LSM (sf_surface_physics=3), got "
            f"sf_surface_physics={cfg.sf_surface_physics}."
        )
    if type(cfg.rdmaxalb) is not bool:
        raise ValueError(
            f"rdmaxalb must be boolean, got {cfg.rdmaxalb!r}."
        )
    if not cfg.rdmaxalb and cfg.sf_surface_physics != 2:
        raise ValueError(
            "rdmaxalb=false is implemented by Noah LSMINIT and requires "
            "sf_surface_physics=2, got "
            f"sf_surface_physics={cfg.sf_surface_physics}."
        )
    if cfg.sf_surface_physics != 2 and (
            cfg.usemonalb or cfg.rdlai2d or cfg.opt_thcnd != 1):
        raise ValueError(
            "usemonalb/rdlai2d/opt_thcnd are Noah LSM options and require "
            "sf_surface_physics=2; the selected land-surface model is "
            f"sf_surface_physics={cfg.sf_surface_physics}."
        )
    # Surface coupling.  The registry has always refused these two through
    # land_surface/pbl requires_components while this battery accepted them,
    # and gpuwm/core/physics.py initialize_physics then raised at driver
    # construction -- so the configuration was dead either way, but the two
    # authorities disagreed about when, and a launcher that trusts the registry
    # and a launcher that trusts validate_run_config saw different answers.
    # tests/test_authority_agreement.py is what found it and what keeps them
    # together; the initialize_physics checks stay as the guard for a directly
    # constructed driver.
    if cfg.sf_surface_physics and not cfg.sf_sfclay_physics:
        raise ValueError(
            f"sf_surface_physics={cfg.sf_surface_physics} requires a surface "
            "layer (sf_sfclay_physics != 0) for its exchange coefficients: "
            "Noah reads CHS/CHS2/CQS2/QGH/RIB, Noah-MP and RUC read the same "
            "seam, and with sf_sfclay_physics=0 nothing writes them. "
            "Select sf_sfclay_physics=1 (revised MM5) or 91 (classic MM5)."
        )
    if cfg.cu_physics and not cfg.moist:
        raise ValueError(
            f"cu_physics={cfg.cu_physics} requires moist=true: the cumulus "
            "schemes are moist convective schemes and gpuwm/core/physics.py "
            "initialize_physics refuses a cumulus scheme on a dry DomainState "
            "(state.qv is None). The registry says the same thing through the "
            "option's required_settings."
        )
    if cfg.cu_physics == 3:
        if not cfg.bl_pbl_physics:
            raise ValueError(
                "cu_physics=3 (Grell-Freitas) requires a PBL scheme: the "
                "trigger's temperature/moisture excesses and the shallow "
                "arm read KPBL and the surface fluxes the PBL stack "
                "maintains; with bl_pbl_physics=0 the engine has no KPBL "
                "to hand the scheme. Select a PBL scheme, e.g. "
                "bl_pbl_physics=1 (YSU); note GF also requires "
                "cudt_minutes=0, so set both in one edit."
            )
        if cfg.cudt_minutes != 0.0:
            raise ValueError(
                "cu_physics=3 (Grell-Freitas) requires cudt_minutes=0: GF "
                "runs on the model step (WRF's usual GF configuration, "
                "STEPCU=1) and carries no NCA hold; cudt is a Kain-Fritsch "
                "cadence knob."
            )
        if cfg.clos_choice != 0:
            raise ValueError(
                f"clos_choice={cfg.clos_choice} is not admitted: only the "
                "16-member ensemble closure (0, the WRF Registry default) "
                "carries GF oracle coverage; the single-closure arms have "
                "no parity receipt."
            )
        if cfg.ishallow not in (0, 1):
            raise ValueError(
                f"ishallow must be 0 or 1, got {cfg.ishallow}.")
    elif cfg.clos_choice != 0 or cfg.ishallow != 0:
        raise ValueError(
            "clos_choice/ishallow are Grell-family keys read only where "
            "cu_physics=3; set them with the scheme or leave the Registry "
            "defaults (0/0)."
        )
    if cfg.sf_sfclay_physics not in (1, 91) and (
            cfg.isftcflx or cfg.iz0tlnd):
        raise ValueError(
            "isftcflx/iz0tlnd are MM5 surface-layer options and require "
            "sf_sfclay_physics=1 or 91; the selected surface layer is "
            f"sf_sfclay_physics={cfg.sf_sfclay_physics}."
        )
    if cfg.ysu_topdown_pblmix not in (0, 1):
        raise ValueError(
            "ysu_topdown_pblmix must be 0 or 1, got "
            f"{cfg.ysu_topdown_pblmix}."
        )
    for name, admitted in MYNN_PBL_OPTION_IDENTITY.items():
        value = getattr(cfg, name)
        if type(value) is not type(admitted) or value != admitted:
            raise ValueError(
                f"{name}={value!r} is outside the admitted MYNN option "
                f"identity; gpuwm implements {name}={admitted!r} only, and "
                "no nearby branch is substituted for an unported one."
            )
    for name, (admitted, evidence) in \
            NOAHMP_OPTION_IDENTITY_EVIDENCE.items():
        value = getattr(cfg, name)
        if type(value) is not type(admitted) or value != admitted:
            raise ValueError(
                f"{name}={value!r} is outside the admitted Noah-MP option "
                f"identity; gpuwm implements {name}={admitted!r} only "
                f"({evidence}), and no nearby branch is substituted for an "
                "unported one."
            )
    for name, (admitted, evidence) in RUC_OPTION_IDENTITY_EVIDENCE.items():
        value = getattr(cfg, name)
        if type(value) is not type(admitted) or value != admitted:
            raise ValueError(
                f"{name}={value!r} is outside the admitted RUC option "
                f"identity; gpuwm implements {name}={admitted!r} only "
                f"({evidence}), and no nearby branch is substituted for an "
                "unported one."
            )
    if cfg.ra_physics not in (0, 4, 90):
        raise ValueError(
            "ra_physics must be 0 (off), 4 (RTE+RRTMGP), or 90 "
            f"(analytic clear-sky proxy), got {cfg.ra_physics}."
        )
    ra_lw_physics, ra_sw_physics = radiation_scheme_ids(cfg)
    if ra_lw_physics not in (0, 1, 4, 90):
        raise ValueError(
            "ra_lw_physics must be 0 (off), 1 (WRF RRTM), 4 "
            f"(RTE+RRTMGP), or 90 (analytic proxy), got {ra_lw_physics}.")
    if ra_sw_physics not in (0, 1, 4, 90):
        raise ValueError(
            "ra_sw_physics must be 0 (off), 1 (WRF Dudhia), 4 "
            f"(RTE+RRTMGP), or 90 (analytic proxy), got {ra_sw_physics}.")
    if ra_lw_physics == 1:
        # Schema-legal and unexecutable, which is exactly the state this
        # battery is supposed to catch before a run starts.  gpuwm/core/physics
        # .py initialize_physics already raises NotImplementedError for it, and
        # the registry refuses the 1/1 pair as an unimplemented option; this
        # function accepted it, so the three did not agree.  ra_sw_physics=1 on
        # its own is the implemented Dudhia shortwave (the 0/1 pair) and stays
        # accepted.
        raise NotImplementedError(
            "ra_lw_physics=1 (WRF RRTM longwave) is not executable yet: the "
            "16-band/140-g-point coefficient and transfer kernels are still "
            "required, and no approximate longwave scheme is substituted. Use "
            "0 (off), 4 (RTE+RRTMGP, with ra_sw_physics=4) or 90 (the "
            "analytic proxy, with ra_sw_physics=90)."
        )
    if ((ra_lw_physics in (4, 90) or ra_sw_physics in (4, 90))
            and ra_lw_physics != ra_sw_physics):
        raise ValueError(
            f"ra_lw_physics={ra_lw_physics} and ra_sw_physics="
            f"{ra_sw_physics}: RTE+RRTMGP (4) and analytic radiation (90) "
            "are coupled LW/SW adapters and must be selected on both "
            "components. Set ra_lw_physics = ra_sw_physics = 4 "
            "(RTE+RRTMGP) or = 90 (the analytic proxy), or select the "
            "0/0 (radiation off) or 0/1 (Dudhia shortwave, longwave off) "
            "pair.")
    if cfg.icloud not in (0, 1):
        raise ValueError(f"icloud must be 0 or 1, got {cfg.icloud}.")
    if not math.isfinite(cfg.swrad_scat) or cfg.swrad_scat < 0.0:
        raise ValueError(
            "swrad_scat must be finite and non-negative, got "
            f"{cfg.swrad_scat}.")
    if cfg.wrf_rrtmg_compatibility not in (
            "none", *WRF_RRTMG_COMPATIBILITY_TOKENS):
        raise ValueError(
            "wrf_rrtmg_compatibility must be 'none' or one of "
            f"{WRF_RRTMG_COMPATIBILITY_TOKENS}, got "
            f"{cfg.wrf_rrtmg_compatibility!r}.")
    if (cfg.wrf_rrtmg_compatibility in WRF_RRTMG_COMPATIBILITY_TOKENS
            and (ra_lw_physics, ra_sw_physics) != (4, 4)):
        raise ValueError(
            f"wrf_rrtmg_compatibility={cfg.wrf_rrtmg_compatibility!r} "
            "requires the resolved 4/4 pair")
    if cfg.ra_rrtmg_variant not in (
            RRTMG_VARIANT_RTE_RRTMGP, RRTMG_VARIANT_LEGACY):
        raise ValueError(
            f"ra_rrtmg_variant must be '{RRTMG_VARIANT_RTE_RRTMGP}' or "
            f"'{RRTMG_VARIANT_LEGACY}', got {cfg.ra_rrtmg_variant!r}.")
    if (cfg.ra_rrtmg_variant == RRTMG_VARIANT_LEGACY
            and (ra_lw_physics, ra_sw_physics) != (4, 4)):
        raise ValueError(
            f"ra_rrtmg_variant='{RRTMG_VARIANT_LEGACY}' requires the "
            "resolved 4/4 RRTMG pair (ra_physics=4 or "
            "ra_lw_physics=ra_sw_physics=4), got "
            f"{ra_lw_physics}/{ra_sw_physics}")
    if ((cfg.o3input != 2 or cfg.use_mp_re != 1)
            and cfg.ra_rrtmg_variant != RRTMG_VARIANT_LEGACY):
        raise ValueError(
            f"o3input={cfg.o3input} and use_mp_re={cfg.use_mp_re}: "
            "nondefault values are implemented only by "
            f"ra_rrtmg_variant='{RRTMG_VARIANT_LEGACY}'."
        )
    if ((cfg.o3input != 2 or cfg.use_mp_re != 1)
            and (ra_lw_physics, ra_sw_physics) != (4, 4)):
        raise ValueError(
            f"o3input={cfg.o3input} and use_mp_re={cfg.use_mp_re} require "
            "the resolved 4/4 legacy-RRTMG radiation pair, got "
            f"{ra_lw_physics}/{ra_sw_physics}."
        )
    if (cfg.wrf_rrtmg_compatibility in WRF_RRTMG_SUBSTITUTION_TOKENS
            and cfg.ra_rrtmg_variant != RRTMG_VARIANT_RTE_RRTMGP):
        raise ValueError(
            f"wrf_rrtmg_compatibility={cfg.wrf_rrtmg_compatibility!r} "
            "records the RTE+RRTMGP substitution and contradicts "
            f"ra_rrtmg_variant={cfg.ra_rrtmg_variant!r}")
    if (cfg.wrf_rrtmg_compatibility == WRF_RRTMG_LEGACY
            and cfg.ra_rrtmg_variant != RRTMG_VARIANT_LEGACY):
        raise ValueError(
            f"wrf_rrtmg_compatibility='{WRF_RRTMG_LEGACY}' records the "
            "legacy RRTMG mapping and requires "
            f"ra_rrtmg_variant='{RRTMG_VARIANT_LEGACY}', got "
            f"{cfg.ra_rrtmg_variant!r}")
    if (ra_lw_physics, ra_sw_physics) == (4, 4) and cfg.icloud != 1:
        raise ValueError(
            "the 4/4 radiation adapters (RTE+RRTMGP today, legacy RRTMG "
            "when it lands) implement cloud-radiation coupling as always "
            "on; icloud=0 would be a silent WRF semantic change")
    if cfg.cu_physics not in CU_SCHEMES:
        raise ValueError(
            f"cu_physics must be 0 (off), 1 (Kain-Fritsch) or 3 "
            f"(Grell-Freitas), got {cfg.cu_physics}."
        )
    if cfg.hypsometric_opt not in (1, 2):
        raise ValueError(
            "hypsometric_opt must be 1 (d(phi)/d(eta) hydrostatic "
            "inversion, the frozen gpuwm form) or 2 (WRF log-pressure "
            "form, calc_p_rho_phi), got "
            f"{cfg.hypsometric_opt}."
        )
    if cfg.mp_physics not in (0, 1, 6, 8, 10, 18, 28):
        raise ValueError(
            "mp_physics must be 0 (off), 1 (Kessler), 6 (WSM6), 8 "
            "(Thompson), 10 (Morrison two-moment), 18 "
            "(NSSL two-moment), or 28 (Thompson aerosol-aware), got "
            f"{cfg.mp_physics}."
        )
    validate_aerosol_source_options(cfg)
    if cfg.mp_physics != 0 and not cfg.moist:
        raise ValueError(
            f"mp_physics={cfg.mp_physics} requires moist=true; the dry "
            "state does not allocate water or hydrometeor fields."
        )
    if cfg.nest_microphysics_transition not in (
            "same-scheme-only", "mp8-to-mp18-mass-diagnosed-v1",
            "mp-edge-mass-diagnosed-v1"):
        raise ValueError(
            "nest_microphysics_transition must be 'same-scheme-only', "
            "'mp8-to-mp18-mass-diagnosed-v1', or "
            "'mp-edge-mass-diagnosed-v1', got "
            f"{cfg.nest_microphysics_transition!r}."
        )
    if (not cfg.nested
            and cfg.nest_microphysics_transition != "same-scheme-only"):
        raise ValueError(
            "nest_microphysics_transition may only be selected on a nested "
            "child domain"
        )
    if cfg.morr_rimed_ice not in (0, 1):
        raise ValueError(
            "morr_rimed_ice must be 0 (graupel) or 1 (hail, the WRF "
            f"default), got {cfg.morr_rimed_ice}."
        )
    if cfg.wsm6_hail_opt not in (0, 1):
        raise ValueError(
            "wsm6_hail_opt must be 0 (graupel, the WRF WSM6 default) or "
            f"1 (hail), got {cfg.wsm6_hail_opt}."
        )
    if cfg.no_mp_heating not in (0, 1):
        raise ValueError(
            "no_mp_heating must be 0 (microphysics latent heating on, the "
            f"WRF default) or 1 (heating off), got {cfg.no_mp_heating}."
        )
    if cfg.nwp_diagnostics not in (0, 1):
        raise ValueError(
            "nwp_diagnostics must be 0 (off, the WRF default) or 1 "
            "(per-step UP_HELI_MAX running-max diagnostic), got "
            f"{cfg.nwp_diagnostics}."
        )
    if not math.isfinite(cfg.mp_tend_lim) or cfg.mp_tend_lim <= 0.0:
        raise ValueError(
            "mp_tend_lim must be a finite positive heating-rate clamp in "
            f"K/s (WRF Registry default 10.0), got {cfg.mp_tend_lim}."
        )
    for name in ("radt", "bldt", "radt_minutes", "cudt_minutes"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{name} must be a finite non-negative interval in minutes "
                f"(0 means every model step), got {value}."
            )
    if not math.isfinite(cfg.restart_interval_s) or cfg.restart_interval_s < 0.0:
        raise ValueError(
            "restart_interval_s must be a finite non-negative interval in "
            f"seconds (0 disables restart writing), got "
            f"{cfg.restart_interval_s}."
        )
    _validate_dynamics_coefficients(cfg)
    validate_km_opt(cfg)
    for name in ("khdif", "kvdif"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{name} must be a finite non-negative constant-K "
                f"diffusivity, got {value}."
            )
    if cfg.km_opt in (2, 3, 4) and (cfg.khdif > 0.0 or cfg.kvdif > 0.0):
        raise ValueError(
            f"km_opt={cfg.km_opt} selects WRF turbulence mixing; "
            "khdif/kvdif are constant-K controls for km_opt=1 and cannot "
            "also be active."
        )
    if not math.isfinite(cfg.c_k) or cfg.c_k <= 0.0:
        raise ValueError(
            f"c_k must be a finite positive TKE-closure constant, got "
            f"{cfg.c_k}."
        )
    if not math.isfinite(cfg.tke_upper_bound) or cfg.tke_upper_bound <= 0.0:
        raise ValueError(
            f"tke_upper_bound must be a finite positive TKE ceiling in "
            f"m2 s-2 (WRF Registry default 1000.0), got "
            f"{cfg.tke_upper_bound}."
        )
    if cfg.tke_budget not in (0, 1):
        raise ValueError(
            f"tke_budget must be 0 (off) or 1 (per-step term-by-term TKE "
            f"budget accumulation), got {cfg.tke_budget}."
        )
    if cfg.tke_budget == 1 and cfg.km_opt != 2:
        raise ValueError(
            "tke_budget=1 requires km_opt=2: the budget decomposes the "
            f"prognostic TKE equation, and km_opt={cfg.km_opt} carries no "
            "TKE."
        )
    if cfg.mix_isotropic not in (0, 1):
        raise ValueError(
            f"mix_isotropic must be 0 (anisotropic mixing lengths) or 1 "
            f"(isotropic (dx*dy*dz)^(1/3)), got {cfg.mix_isotropic}."
        )
    if not math.isfinite(cfg.mix_upper_bound) or cfg.mix_upper_bound <= 0.0:
        raise ValueError(
            f"mix_upper_bound must be a finite positive non-dimensional K "
            f"cap, got {cfg.mix_upper_bound}."
        )
    if not math.isfinite(cfg.tke_heat_flux):
        raise ValueError(
            f"tke_heat_flux must be a finite kinematic heat flux "
            f"(K m s-1), got {cfg.tke_heat_flux}."
        )
    if not math.isfinite(cfg.tke_drag_coefficient) \
            or cfg.tke_drag_coefficient < 0.0:
        raise ValueError(
            f"tke_drag_coefficient must be a finite non-negative drag "
            f"coefficient, got {cfg.tke_drag_coefficient}."
        )
    if cfg.diff_6th_slopeopt not in (0, 1):
        raise ValueError(
            f"diff_6th_slopeopt must be 0 (untapered) or 1 (WRF terrain-"
            f"slope taper), got {cfg.diff_6th_slopeopt}."
        )
    if not math.isfinite(cfg.diff_6th_thresh) or cfg.diff_6th_thresh <= 0.0:
        raise ValueError(
            f"diff_6th_thresh must be a positive slope (m/m), got "
            f"{cfg.diff_6th_thresh}."
        )
    if cfg.h_sca_adv_order not in (2, 5):
        raise ValueError(
            f"h_sca_adv_order must be 2 (frozen Phase 1/2 stencil) or 5 "
            f"(WRF Registry-default rhs_ph advection), got "
            f"{cfg.h_sca_adv_order}."
        )
    if cfg.h_sca_adv_order == 5 and (cfg.open_x or cfg.open_y):
        raise NotImplementedError(
            "h_sca_adv_order=5 with radiative open boundaries is not "
            "wired: only WRF's periodic and specified rhs_ph loop bounds "
            "are implemented (the open branch additionally needs the "
            "boundary-row ph_old upwind terms)."
        )
    if cfg.spec_zone < 1 or cfg.relax_zone < 2:
        raise ValueError("spec_zone must be >= 1 and relax_zone must be >= 2")
    if cfg.spec_bdy_width < cfg.spec_zone + cfg.relax_zone:
        raise ValueError(
            "spec_bdy_width must cover spec_zone + relax_zone")
    # Unsupported lateral-boundary combinations fail loudly here and again
    # in dycore.step (which also catches nonzero terrain heights and directly
    # constructed RunConfigs): their stencils wrap unconditionally across
    # a physical boundary.
    if (cfg.open_x or cfg.open_y) and cfg.terrain_opt != 0:
        raise NotImplementedError(
            "terrain_opt != 0 with open_x/open_y is not wired: the "
            "kinematic surface boundary condition (set_w_surface / "
            "advance_w_phi) differences the terrain height with "
            "unconditional periodic wraps, coupling the two open "
            "boundaries through the terrain slope."
        )
    if ((cfg.open_x or cfg.open_y or cfg.specified)
            and (cfg.khdif > 0.0 or cfg.kvdif > 0.0)):
        raise NotImplementedError(
            "khdif/kvdif > 0 with open or specified lateral boundaries is "
            "not wired: the constant-K diffusion stencils have no "
            "boundary-aware bounds and would wrap across the domain; use "
            "km_opt=4 and/or diff_6th_opt=2 for boundary dissipation."
        )
    if cfg.diff_6th_opt == 1 and cfg.moist:
        raise ValueError(
            "diff_6th_opt=1 (non-monotonic) with moist=true risks negative "
            "moisture: the unlimited 6th-order fluxes bypass the "
            "positive-definite transport limiter; use the monotonic "
            "diff_6th_opt=2."
        )
    if not isinstance(cfg.moist_mix6_off, bool):
        raise ValueError(
            "moist_mix6_off must be a boolean (WRF v4.6.1 declares it "
            "logical, Registry.EM_COMMON:2889), got "
            f"{cfg.moist_mix6_off!r}."
        )
    validate_resolved_physics_vertical_levels(cfg)
    # LES-nest inflow seeding (P3).  The three companion keys are
    # schema-validated even while the mechanism is off (fail-loud, the
    # D3 convention): a typo'd faces mode or a negative scale is a dead
    # configuration either way, and it should say so before a run.
    if type(cfg.inflow_perturbation) is not bool:
        raise ValueError(
            f"inflow_perturbation must be boolean, got "
            f"{cfg.inflow_perturbation!r}.")
    if cfg.inflow_perturbation and not cfg.nested:
        raise ValueError(
            "inflow_perturbation is a nest-boundary mechanism: it "
            "perturbs the rolling parent-forced boundary tables a "
            "NestCoupler refreshes, so it requires nested=true; this "
            "domain is not a nest child."
        )
    if not isinstance(cfg.inflow_perturbation_seed, int) \
            or type(cfg.inflow_perturbation_seed) is bool \
            or cfg.inflow_perturbation_seed < 0:
        raise ValueError(
            "inflow_perturbation_seed must be a non-negative integer "
            "(it keys every Philox draw), got "
            f"{cfg.inflow_perturbation_seed!r}.")
    if not math.isfinite(cfg.inflow_perturbation_amplitude_scale) \
            or cfg.inflow_perturbation_amplitude_scale < 0.0:
        raise ValueError(
            "inflow_perturbation_amplitude_scale must be finite and "
            ">= 0 (0.0 is the registered zero-amplitude control), got "
            f"{cfg.inflow_perturbation_amplitude_scale}.")
    if cfg.inflow_perturbation_faces not in ("inflow", "outflow"):
        raise ValueError(
            "inflow_perturbation_faces must be 'inflow' (the mechanism) "
            "or 'outflow' (the registered AC-P3.4 mutation control), "
            f"got {cfg.inflow_perturbation_faces!r}.")
    return cfg
