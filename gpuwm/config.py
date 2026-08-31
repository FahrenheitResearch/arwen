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



#: THE SURFACE-RADIATION CARRIER POLICY vocabulary.  Defined in the CONFIG
#: module rather than beside the contract it governs
#: (gpuwm.core.radiation_carriers), because the config-load refusal has to
#: be reachable from the standalone RW-WPS preprocessing wheel, which
#: carries gpuwm.config and does not carry gpuwm.core.  The contract
#: imports these names from here; the arrow never points the other way.
SURFACE_RADIATION_POLICY_REQUIRED = "required"
SURFACE_RADIATION_POLICY_WRF_COMPAT_ZERO = "wrf_compat_zero"
SURFACE_RADIATION_POLICIES = (SURFACE_RADIATION_POLICY_REQUIRED,
                              SURFACE_RADIATION_POLICY_WRF_COMPAT_ZERO)


def validate_surface_radiation_policy(policy: str) -> str:
    """Return the policy, or refuse a value that is not one of the two."""
    if policy not in SURFACE_RADIATION_POLICIES:
        raise ValueError(
            f"surface_radiation_policy={policy!r} is not a policy.  The "
            f"choices are {SURFACE_RADIATION_POLICY_REQUIRED!r} (the "
            "default: every carrier a land-surface scheme reads must have "
            "a producer, checked immediately before the scheme consumes "
            f"it) and {SURFACE_RADIATION_POLICY_WRF_COMPAT_ZERO!r} (the "
            "declared escape: unsourced carriers are consumed at their "
            "allocation fill, which reproduces pre-1.9 behaviour and is an "
            "experimental forcing rather than a valid real-case "
            "configuration).")
    return policy


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
    # ArWen ports the CLIMATOLOGY branch (1) and takes it BY DEFAULT on
    # real data through mp28_aerosol_source='auto'; this field stays at
    # WRF's Registry default because the prepared-forecast runner
    # compares the switch rows for exact equality.  2 (first guess) has
    # no ArWen source and refuses by name.
    aer_init_opt: int = 0
    # wif_input_opt -- Registry/registry.new3d_wif:17, default 0.
    # 0 = do not process the Water/Ice Friendly aerosol input from metgrid;
    # 1 = use_wif_input; 2 = use_wif_input_bc, which additionally allocates
    # the black-carbon scalar qnbca.  ArWen implements neither: there is no
    # 1 is ported as the climatology pair with aer_init_opt=1 and is the
    # default real-data route via mp28_aerosol_source='auto'; 2 wants the
    # nbca species, which does not exist in the port, and fails closed.
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
    # DEFAULT TRUE since the 2026-08-17 real-data confirmation leg
    # (fixed-means-default): the 08-01 operational configuration flown
    # through the shipped front door with this channel on.  The RED legs
    # that pin the historical un-channeled formulation set
    # additive_dissipation=False explicitly, exactly as the S3-6j legs
    # pin apply_drag=False (authority module docstring, S3-12 section).
    sase_additive_dissipation: bool = True
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
    # WRF v4.6.1 NSSL variant selectors
    # (Registry.EM_COMMON:2420-2425).  Since v4.5 every NSSL configuration
    # is mp_physics=18 plus these flags; the deprecated scheme IDs
    # 17/19/21/22 are namelist-import spellings that
    # share/module_check_a_mundo.F:3382-3421 rewrites onto them.  ``-1``
    # takes WRF's consistency-pass defaults (:3433-3455), which resolve to
    # the shipped two-moment + hail + predicted-CCN + graupel/hail-volume
    # lane.  Resolution and the ported-combination gate live in
    # gpuwm/core/nssl2_contract.py; unported combinations refuse loudly at
    # validation instead of substituting a nearby branch.
    #
    # Appended last, as every new field must be, so positional construction
    # of every existing RunConfig is unchanged.
    nssl_2moment_on: int = -1
    nssl_hail_on: int = -1
    nssl_ccn_on: int = -1
    nssl_density_on: int = -1
    nssl_3moment: int = 0

    # ---- WDM6 (mp_physics = 16), appended after the NSSL block at the 1.9
    # assembly; both lanes append last, so every existing RunConfig's
    # positional construction is unchanged either way.
    # so positional construction of every existing RunConfig is unchanged.
    #
    # WDM6's half of WRF's SHARED ``hail_opt`` namelist key.  WRF reaches
    # wdm6init with ``config_flags%hail_opt`` (module_physics_init.F:
    # 4582-4584) -- the same scalar mp_wsm6_init reads at :4487 -- and the
    # branch sets the same five rimed-ice constants.  gpuwm keeps the two
    # SEPARATE because the config field is the restart-identity record of
    # which scheme ran under which rimed ice; the importer writes both from
    # the one namelist key, so no namelist can make them disagree.
    #
    # ``wdm6_ccn_conc`` is WRF ``ccn_conc`` (Registry.EM_COMMON:2664,
    # default 1.0E8 # m-3).  Its ONLY effect in module_mp_wdm6.F is the
    # first-step fill of the CCN array (:220-227): ``wdm62D`` and
    # ``wdm6init`` both take ccn0 ``intent(in)`` and never read it.  gpuwm
    # performs that fill once at state allocation, so this value is part of
    # the initial condition and therefore trajectory-bound.
    #
    # Both are SCHEME-SCOPED: nothing outside the mp=16 path reads either,
    # and ``gpuwm.core.model.restart_identity_payload`` drops them from the
    # identity of every run that does not select WDM6, on the same
    # absent-stays-absent convention the ``perturbation`` block and the
    # per-domain ``spawn`` declaration already use.  That is what keeps
    # every pre-WDM6 experiment's fingerprint byte-identical.
    wdm6_hail_opt: int = 0
    wdm6_ccn_conc: float = 1.0e8

    # APPENDED LAST, and deliberately so: the config-freeze discipline
    # requires new fields at the end so positional construction of every
    # frozen case is unchanged.
    #
    # THE SURFACE-RADIATION CARRIER POLICY (gpuwm/core/radiation_carriers.py).
    # "required" is the default for every run, real-data or idealised: a
    # radiative carrier a land-surface scheme reads must have a producer,
    # checked immediately before the LSM consumes it.  "wrf_compat_zero" is
    # the declared escape -- unsourced carriers are consumed at their
    # allocation fill, reproducing pre-1.9 behaviour.  It is never selected
    # automatically, it labels every carrier it touches in the run receipt
    # and the output metadata, and it is an EXPERIMENTAL FORCING rather
    # than a valid configuration for a real case.  Not a WRF namelist key:
    # WRF has no carrier provenance to declare.
    surface_radiation_policy: str = "required"

    # mp=28 WIF aerosol-climatology dataset (change record: appended with
    # the lane/wif-climatology commit onto this line).  Filesystem path to
    # WRF's global monthly QNWFA/QNIFA dataset in WPS intermediate format
    # (QNWFA_QNIFA_SIGMA_MONTHLY.dat as distributed).  Consumed ONLY when
    # aer_init_opt=1 AND wif_input_opt=1 select the ported climatology
    # ingest (gpuwm/ingest/wif_climatology.py); the empty default runs not
    # one new instruction, and validate_aerosol_source_options refuses a
    # set path whose selectors do not consume it, so it can never ride
    # along silently.  A PATH, not a table row, because the dataset is a
    # user-provided 225 MB artifact, not something this repository ships.
    wif_climatology_path: str = ""
    # Which mp=28 aerosol initial state this run wants (change record:
    # appended with the lane/wif-default commit onto this line).  "auto",
    # the default, resolves WRF's monthly WIF climatology and uses it, and
    # falls back to thompson_init's synthetic profile ONLY when no dataset
    # can be found -- announcing that by name in the run receipt
    # (MP28_AEROSOL_SYNTHETIC_FALLBACK).  "climatology" refuses instead of
    # falling back.  "synthetic" selects the fallback deliberately, which is
    # what an idealized case or a reproduction of a pre-lane/wif-default run
    # wants.  Inert under every other mp_physics.
    mp28_aerosol_source: str = "auto"
    #: mp_physics=50 (P3): which P3 implementation integrates the column.
    #: ``"cuda"`` is the DEFAULT and is the shipping path -- the device
    #: kernels in gpuwm/core/kernels/p3.cu, launched one step at a time
    #: (``"cuda"`` is the unfused reference arm; ``"fused"`` is the
    #: three-launch composition proven byte-identical to it on all twelve
    #: p3-fortref fixtures).  ``"reference"`` selects the CPU float32
    #: transcription in gpuwm/core/p3.py, which exists so every device
    #: result stays checkable and so the scheme runs where there is no
    #: CuPy; it is orders slower and is not a production selection.
    #:
    #: SCHEME-SCOPED (gpuwm.core.model.SCHEME_SCOPED_RUN_FIELDS[50]) with an
    #: off-scheme refusal below, in this same commit, for the reason the
    #: mp=28 pair records: an unscoped RunConfig field moves EVERY
    #: experiment fingerprint and makes every existing checkpoint refuse to
    #: resume, for a field the run never reads.
    p3_backend: str = "cuda"


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
    "opt_gla": (1, "ported; NOAHMP_GLACIER (gpuwm/core/noahmp_glacier.py) "
                   "transcribes the opt_gla=1 phase-change arm and every "
                   "opt_gla=2 arm is dead code it refuses, so a plan that "
                   "asks for opt_gla=2 is refused rather than silently "
                   "ignored"),
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
    # bl_mynn_mixscalars left this pin at the W4 full admission (mf-close2
    # Stage B): it is validated by its own block in validate_run_config --
    # {0,1}, with 1 admitted only under the anchored fixture combo
    # (bl_pbl_physics=5, mp_physics=28, bldt=0).
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
SURFACE_LAYER_SCHEMES = (0, 1, 2, 5, 91)
#: WRF sf_surface_physics values in gpuwm's schema.
LAND_SURFACE_SCHEMES = (0, 2, 3, 4)

#: ``cu_physics`` values whose scheme consumes the dycore's PURE ADVECTIVE
#: theta/qv forcing pair -- WRF's ``RTHFTEN``/``RQVFTEN``.
#:
#: A TABLE, not a branch.  The dycore's export, the state allocation, the
#: VRAM projection and the restart inventory all read this one set, so
#: admitting a further consumer (WRF's G3, GD or NTiedtke, each of which
#: takes the same two arguments from ``module_cumulus_driver.F``) is one
#: entry here and nothing else.
#:
#: Grell-Freitas (3) is the entry today.  Kain-Fritsch (1) is deliberately
#: absent: WRF's ``module_cumulus_driver.F`` KFETASCHEME arm passes no
#: RTHFTEN/RQVFTEN at all, so allocating the pair for a KF run would price
#: two full [nz, ny, nx] arrays nothing reads.
#:
#: THE FOLD TRAP, recorded where the table is: WRF's cumulus driver
#: pre-folds ``RTHRATEN + RTHBLTEN`` into ``RTHFTEN`` at
#: ``module_cumulus_driver.F:867`` for G3SCHEME and NTIEDTKESCHEME ONLY.
#: GFSCHEME is not in that list -- GF sums the advective, radiative and
#: boundary-layer lanes itself.  The dycore therefore exports PURE
#: ADVECTION, and any scheme added to this set that expects the pre-folded
#: form must do that fold in its own adapter rather than moving the
#: export, or it double-counts the heating GF must not see twice.
CUMULUS_ADVECTIVE_FORCING_SCHEMES = frozenset({3})
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

#: WRF's Eta similarity surface layer (``sf_sfclay_physics``) and the MYJ
#: PBL (``bl_pbl_physics``).  They share the number 2 in WRF's two
#: namespaces and they are a PAIR, not two independently selectable
#: schemes: ``phys/module_physics_init.F:3770-3772`` fatals a MYJ PBL
#: whose surface layer did not set ``isfc = 2``, and the Eta surface layer
#: is the only ``sf_sfclay_physics`` value gpuwm ports that does
#: (:3169).  :func:`validate_myj_pairing` is that law.
MYJ_SFCLAY_SCHEME = 2
MYJ_PBL_SCHEME = 2

#: ``bl_pbl_physics`` values in gpuwm's schema.  0/1/2/5/11 are WRF's --
#: 2 is MYJ (module_bl_myjpbl.F) and 11 is Shin-Hong
#: (module_bl_shinhong.F, ported at max ULP 0 against the byte-frozen WRF
#: v4.6.1 module; runtime wired by _run_shinhong in
#: gpuwm/core/physics.py) -- and 900 is :data:`SASE_PBL_SCHEME`,
#: ArWen-only (see there).
PBL_SCHEMES = (0, 1, MYJ_PBL_SCHEME, 5, 11, SASE_PBL_SCHEME)
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
#: scheme, so RUC's shipped nine is stated once, next to the level-depth
#: tables the kernel selects between -- ``__constant__ real
#: ruc_soil_layer_depth[RUC_NZS]``, whose ``#if RUC_NZS == 9`` /
#: ``#elif == 6`` arms are the two ``init_soil_depth_3`` tabulates.  (It read
#: ``[9]``, singular, while the forecast column was pinned to nine; that pin
#: is gone -- see docs/wrf_ruc_runtime_admission.md.)
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
# P3 (mp_physics = 50) and its three unported siblings.
# --------------------------------------------------------------------------

#: WRF v4.6.1 maps P3 across four ``mp_physics`` values
#: (Registry.EM_COMMON:3038-3041).  gpuwm ports exactly ONE of them -- 50,
#: ``p3_1category``, the configuration WRF's own driver calls with
#: ``N_ICECAT=1``, no ``nc_3d`` and no ``qzi1_3d``
#: (module_microphysics_driver.F:1557-1602).  The other three are not
#: "unsupported numbers": they are different physics inside the same
#: Fortran module, each gated by a switch the port does not carry, so each
#: gets its own refusal naming what is missing.  Silently running the
#: 1-category solver for a namelist that asked for two categories or a
#: third moment would be exactly the substitution PHYSICS.md forbids.
_P3_UNPORTED_VARIANTS = {
    51: (
        "mp_physics=51 (P3 with prognostic droplet number, "
        "Registry.EM_COMMON:3039) is not ported. gpuwm ports mp_physics=50, "
        "which runs P3 with SPECIFIED droplet number (nccnst/rho, "
        "module_mp_p3.F:2350) -- mp=51 passes nc_3d, which switches on the "
        "CCN-activation path (:3313-3339) that this port does not carry. "
        "Use mp_physics=50 for P3 one-category."
    ),
    52: (
        "mp_physics=52 (P3 two ice categories, "
        "Registry.EM_COMMON:3040) is not ported. gpuwm ports the "
        "nCat=1 configuration only: the inter-category collection "
        "(module_mp_p3.F:2754-2829), the category-merge pass (:4615-4697) "
        "and the second lookup table p3_lookupTable_2 are structurally "
        "absent from this port, and its state carries one ice category. "
        "Use mp_physics=50 for P3 one-category."
    ),
    53: (
        "mp_physics=53 (P3 one category, three-moment ice, "
        "Registry.EM_COMMON:3041) is not ported. gpuwm ports the "
        "TWO-moment-ice configuration: the prognostic reflectivity moment "
        "zitot, its compute_mu_3moment/G_of_mu closure and the 3momI "
        "lookup table (module_mp_p3.F, log_3momentIce branches) are absent, "
        "and only p3_lookupTable_1.dat-v5.4_2momI is packaged. "
        "Use mp_physics=50 for P3 one-category."
    ),
}

#: The mp_physics schema menu, as one sentence, so every out-of-schema
#: value is refused in the same shape.  Held apart from the raise sites
#: because there are two of them: the generic unknown-value branch and the
#: P3 sibling branch below, which states its own missing physics FIRST and
#: then recites this menu.  A refusal that recites the menu is a VALUE
#: refusal; a refusal that does not is about a combination.  That
#: distinction is load-bearing -- tools/report_physics_composition_walk.py
#: separates the two kinds by exactly this recitation -- so a new mp value
#: is added here once and both branches stay in the same form.
_MP_PHYSICS_SCHEMA_MENU = (
    "mp_physics must be 0 (off), 1 (Kessler), 6 (WSM6), 8 "
    "(Thompson), 9 (Milbrandt-Yau two-moment), 10 (Morrison "
    "two-moment), 16 (WDM6 double-moment warm rain), 18 (NSSL "
    "two-moment), 28 (Thompson aerosol-aware), or 50 (P3 "
    "one-category)"
)

#: Every ``mp_physics`` value the config loader admits, as the importable
#: authority the accepted-implies-builds instrument iterates
#: (tests/test_mp_accepted_builds.py).  This tuple and the menu sentence
#: above are two spellings of ONE schema: ``validate_run_config`` tests
#: membership against this tuple and recites the menu when it refuses, so
#: a new scheme is added in both places in the same edit or the loader
#: and its refusal text disagree in front of a user.
MP_PHYSICS_ACCEPTED = (0, 1, 6, 8, 9, 10, 16, 18, 28, 50)


def unported_p3_variant_refusal(value: int) -> str:
    """A P3 sibling's refusal: its own missing physics, then the menu.

    ``mp_physics`` 51/52/53 are real WRF v4.6.1 packages this port does not
    cover, and each is owed the specific reason.  They are still values the
    schema does not admit, so the message ends the way every other
    out-of-schema mp value's does.
    """

    return (f"{_P3_UNPORTED_VARIANTS[value]} "
            f"{_MP_PHYSICS_SCHEMA_MENU}, got {value}.")


# --------------------------------------------------------------------------
# mp_physics = 28 aerosol source.  This WAS "the one deliberate deviation,
# published"; it is now a resolved input with a named fallback.
#
# RETIREMENT RECORD (lane/wif-default, 2026-08-27).  The constant
# ``MP28_AEROSOL_SOURCE_DEVIATION`` used to live here.  It said ArWen took
# its mp=28 aerosol initial state from thompson_init's synthetic CCN/IN
# profile because "ArWen has no QNWFA/QNIFA ingest lane", and warned that a
# whole-forecast comparison against a WIF-initialized WRF run was NOT
# equivalent.  Both halves of that sentence were true when it was written
# and neither is true now: ``gpuwm/ingest/wif_climatology.py`` ports the
# ingest and is oracle-measured against real.exe, and it is what a real-data
# mp=28 run uses BY DEFAULT.  Fixing a defect retires its guards, so the
# constant is gone rather than softened -- a deviation notice that survives
# the deviation is how a tree accumulates warnings nobody can act on.
#
# What replaces it is two constants, because there are now two states and
# they are not the same claim:
#   * MP28_AEROSOL_SOURCE_DEFAULT   -- what a run that found the data did.
#   * MP28_AEROSOL_SYNTHETIC_FALLBACK -- what a run that did not found
#     instead, printed loudly and by name, because a forecast initialized
#     from a synthetic profile is a scientifically different forecast.
# --------------------------------------------------------------------------

#: What a real-data ``mp_physics = 28`` run uses for its aerosol initial
#: state, published as a named constant rather than a comment for the reason
#: the retired deviation was: a receipt can print a constant, and cannot
#: print a source comment.
#:
#: This is the DEFAULT and it is not a flag.  ``real.exe`` reaches the same
#: state through ``use_aero_icbc = .true.`` -> ``aer_init_opt = 1`` with
#: ``wif_input_opt = 1``, reading the global monthly dataset metgrid routes
#: through ``constants_name``; WRF in fact FATALs the other configuration
#: (``dyn_em/module_initialize_real.F:2735-2736``, ``'wif_input_opt=0 but
#: mp_physics=28'``).  ArWen now agrees with WRF by construction instead of
#: documenting why it did not.
MP28_AEROSOL_SOURCE_DEFAULT = (
    "mp_physics=28 takes its aerosol initial state from WRF's global "
    "monthly water/ice-friendly aerosol climatology "
    "(QNWFA_QNIFA_SIGMA_MONTHLY.dat), ported as "
    "gpuwm/ingest/wif_climatology.py: metgrid's four_pt bilinear horizontal "
    "interpolation (METGRID.TBL:885-1150), real.exe's monthly_interp_to_date "
    "temporal weighting (module_initialize_real.F:8029-8095), and real.exe's "
    "vert_interp onto the dry eta pressure (:2452/:2519), with the surface "
    "emission qnwfa2d=w_wif(:,1,:)*0.000196*(50/z1) (:4530-4547). This is "
    "the same dataset, the same operators and the same stage order real.exe "
    "runs under use_aero_icbc=.true. (aer_init_opt=1, wif_input_opt=1), "
    "matched to it to 1e-5. The dataset is located by "
    "gpuwm.ingest.wif_climatology.resolve_wif_climatology and the run "
    "receipt carries its path and SHA-256."
)

#: What a run got INSTEAD when the dataset could not be found, and why that
#: matters enough to be shouted rather than logged.
#:
#: The fallback is not a degraded version of the default.  It is WRF's own
#: synthetic boundary-layer-following CCN/IN profile
#: (``phys/module_mp_thompson.F:493-551``), a real initial condition that
#: WRF installs whenever the aerosol fields arrive empty -- but a forecast
#: started from it is a different forecast, measurably (see the lane/
#: wif-default before/after), and anyone comparing it to a WIF-initialized
#: WRF run is comparing two different experiments.  So it is REPORTED, in
#: the run receipt, by name, with the reason the dataset was not found.
MP28_AEROSOL_SYNTHETIC_FALLBACK = (
    "FALLBACK IN USE -- this mp_physics=28 run was NOT initialized from "
    "aerosol data. WRF's global monthly QNWFA/QNIFA climatology could not "
    "be located, so the aerosol initial state is thompson_init's SYNTHETIC "
    "CCN/IN profile (phys/module_mp_thompson.F:493-551), the profile WRF "
    "installs when the fields arrive empty. That is a scientifically "
    "different initial condition from the default: it is not measured "
    "aerosol, it is a two-parameter analytic profile keyed on the height of "
    "model level 1, and a whole-forecast comparison against a "
    "WIF-initialized WRF run is NOT equivalent and must not be reported as "
    "one. Set wif_climatology_path, $GPUWM_WIF_CLIMATOLOGY, or "
    "$GPUWM_WIF_CLIMATOLOGY_ROOT to WRF's QNWFA_QNIFA_SIGMA_MONTHLY.dat, or "
    "run from a directory holding it, to take the default path."
)

#: The three values of :attr:`RunConfig.p3_backend`.  "cuda" and "fused"
#: are the same device kernels composed into nine launches and three; they
#: are byte-identical on every p3-fortref fixture (evidence/
#: p3-cuda-20260829).  "reference" is the CPU float32 transcription.
P3_BACKENDS = ("cuda", "fused", "reference")

#: The three values of :attr:`RunConfig.mp28_aerosol_source`.
#:
#: ``auto`` is the default and does the correct thing without being asked.
#: The other two exist because a default that cannot be pinned is a default
#: nobody can reproduce: ``climatology`` refuses rather than degrades, and
#: ``synthetic`` is how an idealized or deliberately data-free run NAMES the
#: fallback instead of arriving at it by accident.
MP28_AEROSOL_SOURCES = ("auto", "climatology", "synthetic")

#: The only implemented value of each mp=28 aerosol-source selector, and the
#: named capability a user would need for anything else.  Read by
#: :func:`validate_aerosol_source_options`, by the namelist importer's
#: aerosol-key sweep, and by the mp=28 admission test -- one table, so a
#: refusal message and a namelist refusal cannot drift apart.
MP28_AEROSOL_SOURCE_OPTIONS: dict[str, tuple[int, str, str]] = {
    "aer_init_opt": (
        0,
        "Registry/Registry.EM_COMMON:2656",
        "aer_init_opt=2 is real.exe's FIRST-GUESS branch: it consumes "
        "water/ice-friendly aerosol arrays interpolated from a metgrid WIF "
        "stream carried by the driving model itself "
        "(dyn_em/module_initialize_real.F:2327-2732 3-D, :4499-4653 2-D), "
        "which no ArWen input source provides; the CLIMATOLOGY branch "
        "(aer_init_opt=1 with wif_input_opt=1) IS ported "
        "(gpuwm/ingest/wif_climatology.py) and is what a real-data mp=28 "
        "run does by default",
    ),
    "wif_input_opt": (
        0,
        "Registry/registry.new3d_wif:17",
        "wif_input_opt=2 additionally allocates the black-carbon scalar "
        "qnbca (Registry/registry.new3d_wif:82) and there is no nbca "
        "species anywhere in the mp=28 port; wif_input_opt=1 activates the "
        "use_wif_input package (monthly WIF levels per species, :80) and IS "
        "ported as the climatology pair with aer_init_opt=1, the default "
        "route for real-data mp=28",
    ),
}


def validate_aerosol_source_options(cfg: RunConfig) -> None:
    """Fail closed on any mp=28 aerosol-source selector ArWen cannot honour.

    THE DEFAULT MOVED (lane/wif-default).  ``(aer_init_opt, wif_input_opt)``
    at WRF's Registry defaults ``(0, 0)`` no longer MEANS "synthetic
    profile"; it means "ArWen chooses", and what ArWen chooses is the
    climatology whenever the dataset resolves.  The two namelist fields keep
    WRF's Registry defaults on purpose -- ``physics_compat``'s
    ``_SINGLE_DOMAIN_RUNTIME_SWITCHES`` rows are compared for EXACT equality
    by the prepared-forecast runner, and a nonzero default here would move
    every shipped profile for a reason that has nothing to do with profiles.
    The ArWen-side selection therefore lives in its own field,
    :attr:`RunConfig.mp28_aerosol_source`, and ``(1, 1)`` remains the
    namelist-level way to demand the same thing.

    What is still refused, and why each: ``aer_init_opt=2`` (real.exe's
    first-guess WIF stream -- no ArWen source carries one), and
    ``wif_input_opt=2`` (allocates qnbca, a species the port does not have).
    Those are unimplemented capabilities, not defaults, so they refuse by
    name rather than being reinterpreted.  The check stays unconditional on
    ``mp_physics``: under any other scheme both keys are inert in WRF too,
    and accepting an inert nonzero here would let a configuration that MEANS
    something under mp=28 ride silently into an mp=28 restart or nest.
    """
    source = str(getattr(cfg, "mp28_aerosol_source", "auto") or "auto")
    if source not in MP28_AEROSOL_SOURCES:
        raise ValueError(
            f"mp28_aerosol_source={source!r} is not one of "
            f"{MP28_AEROSOL_SOURCES}; 'auto' (the default) uses WRF's "
            "monthly WIF aerosol climatology when it resolves and announces "
            "the synthetic fallback by name when it does not, 'climatology' "
            "refuses rather than falling back, and 'synthetic' selects the "
            "fallback deliberately")
    selected = (int(cfg.aer_init_opt), int(cfg.wif_input_opt))
    path = str(getattr(cfg, "wif_climatology_path", "") or "")
    if selected == (1, 1):
        # The namelist spelling of mp28_aerosol_source='climatology' --
        # real.exe's use_aero_icbc=.true. state.  It no longer requires an
        # explicit path, because the resolver has a search order; it does
        # still require that the search SUCCEED, which the resolver enforces
        # with explicit_required=True.  Nothing to check here.
        return
    if selected != (0, 0):
        for name, (only, citation, why) in MP28_AEROSOL_SOURCE_OPTIONS.items():
            value = getattr(cfg, name)
            if value in (only, 1):
                continue
            raise NotImplementedError(
                f"{name}={value!r} is not implemented; ArWen honours "
                f"{name}={only!r} (WRF Registry default, {citation}) and "
                f"{name}=1 as the ported climatology pair. {why}.")
        raise NotImplementedError(
            f"aer_init_opt={cfg.aer_init_opt!r}/wif_input_opt="
            f"{cfg.wif_input_opt!r} is a MIXED aerosol-source selection. "
            "real.exe's climatology route derives both together "
            "(use_aero_icbc=.true. -> aer_init_opt=1, and the use_wif_input "
            "package -> wif_input_opt=1); one without the other names half "
            "an input path, and ArWen refuses rather than guessing which "
            "half was meant. Use aer_init_opt=1 with wif_input_opt=1, or "
            "leave both at 0 and let mp28_aerosol_source decide.")
    if path and source == "synthetic":
        raise ValueError(
            "wif_climatology_path names WRF's monthly WIF aerosol dataset "
            "but mp28_aerosol_source='synthetic' selects thompson_init's "
            "synthetic profile instead, so nothing would read it; refusing "
            "to carry a dataset path no code would open rather than "
            "silently ignoring it")


#: Every switch WRF's Milbrandt-Yau path hard-codes, with the line that
#: does it.  ``mp_milbrandt2mom_driver`` fixes the first seven
#: (phys/module_mp_milbrandt2mom.F:3615-3623) and the scheme body fixes the
#: last two (:1174-1175); there is no namelist that moves any of them, so
#: mp=9 has exactly ONE identity in WRF v4.6.1 and gpuwm ships that one.
#: gpuwm's constant table (gpuwm/core/milbrandt2_constants.py) is derived
#: under these settings -- CCNtype=2 fixes N_c_SM=2e8 and
#: snowSpherical=.false. selects the Brandes m(D) pair -- so a knob that
#: moved one of them would silently invalidate 154 constants.
MILBRANDT2_FIXED_IDENTITY: dict[str, tuple[object, str]] = {
    "ccntype": (2, "module_mp_milbrandt2mom.F:3615 (continental)"),
    "precip_diag": (True, "module_mp_milbrandt2mom.F:3618"),
    "sedi": (True, "module_mp_milbrandt2mom.F:3619"),
    "warmphase": (True, "module_mp_milbrandt2mom.F:3620"),
    "autoconv": (True, "module_mp_milbrandt2mom.F:3621"),
    "icephase": (True, "module_mp_milbrandt2mom.F:3622"),
    "snow": (True, "module_mp_milbrandt2mom.F:3623"),
    "snow_spherical": (False, "module_mp_milbrandt2mom.F:1174"),
    "prim_ice_nucl": (1, "module_mp_milbrandt2mom.F:1175 (Meyers+contact)"),
}


def validate_milbrandt2_options(cfg: RunConfig) -> None:
    """Fail closed on the mp=9 pairings gpuwm cannot honour.

    Two things are checked, and neither is a taste call:

    THE PINNED IDENTITY.  WRF exposes no namelist for any of
    :data:`MILBRANDT2_FIXED_IDENTITY`, so there is nothing to validate on
    :class:`RunConfig` today -- the entry exists so that the day someone
    adds an ``mp9_*`` knob, the pin is already written down beside the
    Fortran line that owns it (the MYNN pattern).  The loop below refuses
    any such attribute that appears and disagrees.

    RTE+RRTMGP CLOUD OPTICS.  ``MILBRANDT2MOM`` is absent from WRF's
    ``use_mp_re`` disjunction (phys/module_physics_init.F:1004-1023), so
    the scheme supplies radiation no effective radii -- its own reff block
    is commented out (module_mp_milbrandt2mom.F:3362/:3364/:3372/:3374).
    gpuwm's RTE+RRTMGP adapter needs a cloud-optics row per selector and
    has none that means "ice-active, scheme supplies no radii": Kessler's
    row would silently radiate an overcast ice cloud as clear sky and
    Morrison's row would derive radii from a gamma distribution that is
    Morrison's, not Milbrandt-Yau's.  Rather than invent one, the pairing
    is refused and the legacy RRTMG port -- which computes its own radii
    exactly as WRF does under has_reqc=0 -- is named as the way through.
    """
    if cfg.mp_physics != 9:
        return
    for name, (only, citation) in MILBRANDT2_FIXED_IDENTITY.items():
        if not hasattr(cfg, name):
            continue
        value = getattr(cfg, name)
        if value == only:
            continue
        raise NotImplementedError(
            f"{name}={value!r} is not implemented for mp_physics=9; WRF "
            f"hard-codes {name}={only!r} ({citation}) and gpuwm's constant "
            "table is derived under that identity, so a different value "
            "would silently invalidate it.")
    if ((cfg.ra_lw_physics, cfg.ra_sw_physics) == (4, 4)
            and cfg.ra_rrtmg_variant != "rrtmg_legacy"):
        raise NotImplementedError(
            "mp_physics=9 with ra_lw_physics=4/ra_sw_physics=4 on the "
            "RTE+RRTMGP variant has no cloud-optics coupling: WRF leaves "
            "has_reqc/has_reqi/has_reqs at 0 for MILBRANDT2MOM "
            "(phys/module_physics_init.F:1004-1023) and the scheme's own "
            "effective-radius block is commented out, so there are no "
            "scheme radii to hand RRTMGP and no row in "
            "gpuwm.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME. Set "
            "ra_rrtmg_variant='rrtmg_legacy' (which computes its own radii "
            "the way WRF does under has_reqc=0), or select "
            "ra_lw_physics=0/ra_sw_physics=1 (Dudhia).")


def validate_p3_radiation(cfg: RunConfig) -> None:
    """Refuse P3 on RTE+RRTMGP, where it has no cloud-optics coupling.

    FOUND AT THE 1.9 GATE, while proving mp=50 can actually step.  The
    admission path accepted mp_physics=50 with ra_lw_physics=4 /
    ra_sw_physics=4 and the run then died at the FIRST radiation call:
    ``gpuwm/core/rrtmgp.py:1961`` resolves
    ``cloud_optics_scheme(50)`` and ``_MP_CLOUD_OPTICS_SCHEME`` has no row
    for 50, so the adapter raised NotImplementedError mid-forecast.  A
    refusal a user meets after the prepare is not a refusal.

    Why the answer is a refusal and not a new row.  P3 is not
    Milbrandt-Yau's case: it IS in WRF's ``use_mp_re`` disjunction, so
    ``has_reqc = has_reqi = 1`` and the scheme does supply cloud and ice
    radii (gpuwm/core/p3.py writes ``state.effc``/``state.effi``).  What
    it does NOT supply is snow radii -- ``phys/module_physics_init.F``'s
    P3 / Jensen-Ishmael override sets ``has_reqs = 0`` at :1027-1033 --
    and P3's single ice category spans rime fraction rather than
    separating snow from graupel at all.  No row in
    ``_MP_CLOUD_OPTICS_SCHEME`` means "ice-active, cloud and ice radii
    supplied, no snow radii": Thompson's and Morrison's rows all assume a
    supplied re_snow and a separate snow species.  Choosing one of them
    would hand RRTMGP a snow radius P3 never computed, which is inventing
    physics, and inventing it silently is what put mp=28 on Kessler's row
    until 2026-08-01.

    So the pairing is refused and both working alternatives are named, on
    the mp=9 precedent directly above: the legacy RRTMG variant computes
    its own radii the way WRF does, and the Dudhia pair needs none.
    Adding a real P3 cloud-optics row is a porting decision with its own
    WRF authority and its own evidence, and it is not this gate's to make.
    """

    if int(cfg.mp_physics) != 50:
        return
    if ((cfg.ra_lw_physics, cfg.ra_sw_physics) == (4, 4)
            and cfg.ra_rrtmg_variant != "rrtmg_legacy"):
        raise NotImplementedError(
            "mp_physics=50 with ra_lw_physics=4/ra_sw_physics=4 on the "
            "RTE+RRTMGP variant has no cloud-optics coupling: WRF sets "
            "has_reqs=0 for P3 (phys/module_physics_init.F:1027-1033) and "
            "P3's single ice category carries no separate snow species, so "
            "there is no snow radius to hand RRTMGP and no row in "
            "gpuwm.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME. Set "
            "ra_rrtmg_variant='rrtmg_legacy' (which computes its own radii "
            "the way WRF does), or select "
            "ra_lw_physics=0/ra_sw_physics=1 (Dudhia).")


#: Tables whose keys merge into :class:`RunConfig` field by field.
_RUN_CONFIG_TABLES = ("grid", "dynamics", "run")

#: Every table a RunConfig TOML may legally carry.
#:
#: ``[tiles]`` is here and NOT in :data:`_RUN_CONFIG_TABLES`, because it is
#: not a RunConfig table: it is an EXECUTION choice whose entire claim is
#: that it changes nothing about the forecast
#: (:func:`gpuwm.core.streaming.identity_payload_entry`), so its keys belong
#: to :class:`gpuwm.core.streaming.StreamingOptions` rather than to the
#: config whose fields a restart identity binds.  Read with
#: :func:`load_streaming_options`.
#:
#: It was absent from both lists until now, which is the defect: the
#: experiment TOML the multi-domain front doors read has accepted ``[tiles]``
#: since 2.2.0, and this schema -- the one ``gpuwm downscale`` hands to the
#: offline child -- refused the block outright as an unknown table.  So the
#: one route whose domains are MOST likely to outgrow the card, a child
#: refined out of an archived parent, was the one route that could not ask
#: to stream.
#: ``[output]`` joins it on exactly the same terms.  It selects which
#: history variables reach the product tape
#: (:mod:`gpuwm.io.history_selection`) and changes no number the model
#: computes, so it is not a RunConfig table either and does not enter a
#: restart identity -- a run that trimmed its history must resume from a
#: checkpoint written by one that did not, and the checkpoint stream is
#: a different file written from model state.
_KNOWN_TABLES = (*_RUN_CONFIG_TABLES, "tiles", "output")


def load_history_selection(path: str | Path):
    """The ``[output]`` block of a RunConfig TOML, or the shared FULL object.

    Separate from :func:`load_config` and not a field on what it returns,
    for the reason :func:`load_streaming_options` gives: ``RunConfig``'s
    fields bind into every restart identity, and which variables the
    HISTORY tape carries must not.  Both readers go through the same
    config authority, so they read the same bytes of the same file.
    """
    import io

    from gpuwm.config_authority import read_config_authority
    from gpuwm.io.history_selection import HistorySelection

    authority = read_config_authority(path)
    raw = tomllib.load(io.BytesIO(authority.payload))
    return HistorySelection.from_mapping(raw.get("output"), source=str(path))


def load_streaming_options(path: str | Path):
    """The ``[tiles]`` block of a RunConfig TOML, or the shared OFF object.

    Separate from :func:`load_config` and not a field on what it returns:
    ``RunConfig``'s fields are bound into every restart identity, and
    ``[tiles]`` is the one surface that must NOT bind -- a checkpoint
    written resident has to resume streamed and back again, which is the
    operation the mode exists for.  Both readers go through the same config
    authority, so they read the same bytes of the same file.
    """
    import io

    from gpuwm.config_authority import read_config_authority
    from gpuwm.core.streaming import StreamingOptions

    authority = read_config_authority(path)
    raw = tomllib.load(io.BytesIO(authority.payload))
    return StreamingOptions.from_mapping(raw.get("tiles"), source=str(path))


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
    # Validated here as well as in load_streaming_options, and discarded:
    # [tiles] is refused key by key by StreamingOptions on the
    # [relocation]/[perturbation] precedent -- a misspelled knob that
    # silently does nothing is how a run gets configured for a mode it is
    # not in -- and a caller that reads only the RunConfig must not be the
    # reason a typo survives admission.
    load_streaming_options(path)
    # Same treatment for [output], and for the same reason: a misspelled
    # history_drop that silently does nothing is how a run comes to write
    # the full inventory under the name of your selection.
    load_history_selection(path)
    known_keys = {f.name for f in fields(RunConfig)}
    merged: dict = {}
    key_table: dict[str, str] = {}
    for table in _RUN_CONFIG_TABLES:
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


def validate_myj_pairing(cfg: RunConfig) -> None:
    """Refuse a MYJ half without its other half, in either direction.

    WRF's own law, transcribed: ``phys/module_physics_init.F:3770-3772``
    is ``CASE (MYJPBLSCHEME) / if(isfc .ne. 2) CALL wrf_error_fatal
    ('module_physics_init: use myjsfc scheme for this pbl option')``, and
    of the surface layers gpuwm ports only ``MYJSFCSCHEME`` sets
    ``isfc = 2`` (:3169).  So the MYJ PBL with anything else is a WRF
    fatal, and this is that fatal.

    THE REVERSE DIRECTION IS ARWEN'S, and it is stated as such.  WRF
    admits the Eta surface layer under several PBLs it also ports (the
    ``isfc .ne. 2`` guards at :3742 and :3756 belong to other schemes);
    gpuwm ports NONE of those, so the only PBL that could consume the Eta
    surface layer's output here is MYJ.  What makes the mismatch a
    refusal rather than a warning is that the Eta layer's published set
    is not the MM5 layers' published set: it produces ``AKHS``/``AKMS``/
    ``THZ0``/``QZ0``/``UZ0``/``VZ0`` and produces NO ``MOL``, ``ZOL``,
    ``PSIM``/``PSIH``, ``REGIME``, ``GZ1OZ0`` or ``WSPD`` (the outputs of
    ``SFCDIF``, module_sf_myjsfc.F:361-1056).  YSU, MYNN, Shin-Hong and
    SASE all read at least one of those, so pairing them with the Eta
    layer would feed a PBL scheme a zero where WRF gives it a similarity
    function -- finite, plausible, and wrong.  Refusing beats that.

    Urban is refused by ABSENCE, deliberately: WRF sends the MYJ PBL
    through ``myjurb`` when ``sf_urban_physics`` is 2 or 3 (BEP/BEM,
    module_physics_init.F:3775-3781, phys/module_bl_myjurb.F).  gpuwm has
    no ``sf_urban_physics`` selector at all, so that arm is unreachable
    rather than silently substituted; the registry option records it.
    """
    pbl = int(cfg.bl_pbl_physics)
    sfclay = int(cfg.sf_sfclay_physics)
    if pbl == MYJ_PBL_SCHEME and sfclay != MYJ_SFCLAY_SCHEME:
        raise ValueError(
            f"bl_pbl_physics={MYJ_PBL_SCHEME} (MYJ) requires "
            f"sf_sfclay_physics={MYJ_SFCLAY_SCHEME} (Eta similarity), got "
            f"sf_sfclay_physics={sfclay}. WRF v4.6.1 fatals this exact "
            "pairing at phys/module_physics_init.F:3770-3772 ('use myjsfc "
            "scheme for this pbl option'): the MYJ PBL's every implicit "
            "solve takes AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 as its lower boundary "
            "and only the Eta surface layer produces them. No substitution "
            "is applied.")
    if pbl == MYJ_PBL_SCHEME and not cfg.moist:
        # WRF's own fatal, not a house rule: the PBL driver guards MYJPBL
        # with PRESENT(qv_curr) .AND. PRESENT(qc_curr) and calls
        # wrf_error_fatal('Lack arguments to call MYJ pbl') otherwise
        # (phys/module_pbl_driver.F:1441-1443, :1500-1513).  The scheme
        # mixes vapour and cloud water as rows 2 and 3 of its own
        # tridiagonal solve and forms its mixing length from a moist
        # buoyancy gradient; a dry state has nothing for those rows.
        raise ValueError(
            f"bl_pbl_physics={MYJ_PBL_SCHEME} (MYJ) requires moist=true: "
            "the scheme mixes water vapour and cloud water as species rows "
            "of its own vertical solve and builds its mixing length from a "
            "moist buoyancy gradient (module_bl_myjpbl.F:501-503, "
            ":865-867). WRF refuses the same configuration at "
            "module_pbl_driver.F:1500-1513.")
    if sfclay == MYJ_SFCLAY_SCHEME and pbl != MYJ_PBL_SCHEME:
        raise ValueError(
            f"sf_sfclay_physics={MYJ_SFCLAY_SCHEME} (Eta similarity) is "
            f"admitted with bl_pbl_physics={MYJ_PBL_SCHEME} (MYJ) only, got "
            f"bl_pbl_physics={pbl}. The Eta surface layer publishes "
            "AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 and publishes no MOL, ZOL, "
            "PSIM/PSIH, REGIME, GZ1OZ0 or WSPD (module_sf_myjsfc.F:"
            "361-1056); every other PBL gpuwm ports reads at least one of "
            "those and would silently receive a zero. Select "
            "sf_sfclay_physics=1 (revised MM5) or 91 (classic MM5) for "
            "those schemes, or bl_pbl_physics=2 for this one.")


def validate_sase_config(cfg: RunConfig) -> None:
    """Admission for the SASE closure and its three knobs.

    Warn-not-block applies to *maturity*, never to coherence: an
    experimental scheme still refuses a configuration it cannot honestly
    execute.  What is refused here is only what the closure genuinely
    needs -- see :data:`_SASE_REQUIREMENTS` for why the development
    lane's wider whitelist is not reproduced.
    """
    validate_myj_pairing(cfg)
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
             "elsewhere would decouple nothing")):
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
    # sase_additive_dissipation left the fail-closed loop when its
    # default flipped (False -> True, 2026-08-16, 1a0e8a7f8): the loop's
    # charter is "every existing configuration keeps validating
    # unchanged", and artifacts written under the old default RECORD
    # ``sase_additive_dissipation = false`` beside non-SASE PBLs -- every
    # 2.4.x restart header and every child TOML 2.4.x downscale rendered.
    # Refusing the old default broke `gpuwm downscale` of a 2.4.1
    # archive on this tree (MEASURED 2026-08-17, masked as a VRAM-fit
    # refusal).  Off-SASE the knob is inert in BOTH positions, so only
    # the type is held; on SASE both positions are legal channels.
    value = cfg.sase_additive_dissipation
    if type(value) is not bool:
        raise ValueError(
            f"sase_additive_dissipation must be boolean, got {value!r}.")


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
                                ladder: str = "", forced: bool = False,
                                ) -> tuple[float | None, str | None]:
    """``(ratio, one-sentence advisory)`` for the exposed-mixing check.

    ``forced`` says the configuration WROTE ``mix_isotropic = 0`` -- as
    opposed to inheriting it -- and appends the override state to the
    over-the-limit sentence.  Since the auto-switch (Drew, 2026-08-16;
    :func:`auto_mix_isotropic_selection`), a domain that reaches this
    advisory over the limit through the experiment loader can ONLY have
    written the value: an unset ``mix_isotropic`` resolves to 1 there
    and never lands here.  The caller supplies the flag because only it
    knows which loader the config came through.

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
        f"mixing of w {tier}. The remedy is mix_isotropic = 1 on this "
        f"domain: it builds ONE length from (dx*dy*dz)^(1/3) and caps "
        f"the vertical coefficient against the horizontal one, which "
        f"takes the domain off this path entirely rather than moving it "
        f"nearer the limit. Lowering mix_upper_bound below "
        f"{admitted:.3g} also satisfies the criterion, but it weakens "
        f"the subgrid mixing everywhere to fix a horizontal-operator "
        f"problem. COST OF THE REMEDY, STATED HERE SO IT IS NOT "
        f"DISCOVERED LATER: mix_isotropic is inside the RunConfig "
        f"fingerprint gpuwm/io/restart.py writes as "
        f"configuration_sha256, so a checkpoint written under "
        f"mix_isotropic = 0 CANNOT be resumed under 1 -- take it at "
        f"t = 0, not part-way through a campaign you intend to restart. "
        f"ADVISORY, not a "
        f"refusal: {ratio:.3g} is the WORST case the cap admits and a "
        f"flow that never reaches it never sees this, which is why "
        f"trees above the limit have completed -- but above the limit "
        f"nothing guarantees the operator's own stability, and what it "
        f"costs when it does bite is an abort late in a long run rather "
        f"than a number you can inspect")
    if forced:
        advice += (
            ". OVERRIDE STATE: mix_isotropic = 0 is this configuration's "
            "EXPLICIT setting, so the automatic isotropic selection "
            "stands aside and the anisotropic form runs, at the ratio "
            "above; delete the key (or write \"auto\") to let the model "
            "select the stable length here")
    return ratio, advice


def warn_anisotropic_w_mixing(*, where: str, km_opt: int, mix_isotropic: int,
                              mix_upper_bound: float, dx: float, dy: float,
                              dz_max: float, ladder: str = "",
                              forced: bool = False) -> float | None:
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

    THE DEFAULT IS THE AUTO-SWITCH (Drew, 2026-08-16; supersedes the
    recipe-only half of the 2.5.0 "where the default moved" ruling).
    "Fixed means default" makes an opt-in remedy for a correctness
    criterion a workaround, and an advisory that a bare config scrolls
    past was exactly that.  A domain that leaves ``mix_isotropic`` unset
    (or writes ``"auto"``) and violates this criterion now RUNS
    ``mix_isotropic = 1`` -- resolved by the experiment loader
    (``gpuwm.experiment.resolve_auto_mix_isotropic``), announced by
    :func:`auto_mix_isotropic_selection` at load and by ``gpuwm check``,
    with the fingerprint consequence named at both restart doors
    (:data:`MIX_ISOTROPIC_RESTART_BREAK_NOTICE`).  The wizard's
    :func:`gpuwm.domain_wizard.gray_zone_advisory` recipe keeps naming
    ``mix_isotropic = 1`` explicitly, as before.

    So this function fires only for a configuration that WROTE
    ``mix_isotropic = 0`` (or one on a route the auto-switch does not
    resolve, e.g. a wrapped legacy RunConfig), and that asymmetry is the
    ruling: an explicit setting is kept, in the danger zone too, and
    what it gets is this warning -- the instability by name, the
    measured ratio, and (``forced``) the override state.  Refusing would
    make the frozen crash records unloadable; flipping a WRITTEN key
    would mutate a physics selector the user chose and orphan its
    checkpoints (``mix_isotropic`` is inside ``configuration_sha256``).

    Returns the computed ratio so a caller can record the number whether
    or not it warned.  ``None`` covers two cases and a caller must not
    conflate them: not applicable, and applicable with an unresolvable
    layer depth -- the second one warns, and the warning is the only
    place that distinction is visible from here.
    """

    ratio, advice = anisotropic_w_mixing_advice(
        where=where, km_opt=km_opt, mix_isotropic=mix_isotropic,
        mix_upper_bound=mix_upper_bound, dx=dx, dy=dy, dz_max=dz_max,
        ladder=ladder, forced=forced)
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


#: The TOML sentinel for "let the model choose the mixing length": the
#: same meaning as leaving ``mix_isotropic`` unset, writable so a config
#: can SAY it is deferring rather than merely omit the key.  Only the
#: experiment loader consumes it; ``RunConfig.mix_isotropic`` itself
#: stays the resolved WRF integer (0/1) everywhere downstream.
MIX_ISOTROPIC_AUTO = "auto"


def auto_mix_isotropic_selection(*, where: str, ratio: float,
                                 ladder: str = "") -> str:
    """The one loud line for an auto-selected isotropic mixing length.

    One wording, every door -- exactly the discipline of
    :func:`anisotropic_w_mixing_advice`: the experiment loader prints it
    (through :func:`gpuwm.explain.warn`) at the shared config load every
    front door runs at model start, and ``gpuwm check`` repeats it in
    its advisory list so the report says what the run WILL do, not that
    something is wrong.  It names what happened and why: the criterion
    value, the limit it exceeds, and that isotropic mixing was selected
    -- plus the escape hatch and the restart consequence, so neither is
    discovered later.
    """

    provenance = f" ({ladder})" if ladder else ""
    return (
        f"{where} leaves mix_isotropic UNSET on a grid that violates the "
        f"anisotropic-mixing criterion -- mix_upper_bound*(dz_max/dx)^2 "
        f"= {ratio:.3g} exceeds the explicit horizontal diffusion limit "
        f"{EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT}{provenance} -- so this "
        f"run SELECTS ISOTROPIC MIXING: mix_isotropic = 1, the single "
        f"(dx*dy*dz)^(1/3) length, runs on this domain. Write "
        f"mix_isotropic = 0 explicitly to keep the anisotropic form "
        f"(the instability advisory then applies); a checkpoint written "
        f"under the old anisotropic default will not bit-continue under "
        f"this selection.")


#: The restart doors' one-line honesty about the changed default: said
#: when a checkpoint integrated under ``mix_isotropic = 0`` meets a run
#: that selects 1, because a bare hash/field mismatch does not tell a
#: reader that the DEFAULT moved under them.  A notice beside the
#: existing refusal, not a refusal of its own -- the mismatch machinery
#: already names the concrete breakage (a different trajectory).
MIX_ISOTROPIC_RESTART_BREAK_NOTICE = (
    "note: the checkpoint was integrated under anisotropic mixing "
    "(mix_isotropic = 0) and this run selects ISOTROPIC mixing "
    "(mix_isotropic = 1). A config that leaves mix_isotropic unset now "
    "auto-selects the isotropic length where mix_upper_bound*"
    "(dz_max/dx)^2 exceeds "
    f"{EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT}, and a trajectory cannot "
    "bit-continue across that change. To resume this checkpoint, write "
    "mix_isotropic = 0 explicitly on the domain (the instability "
    "advisory then applies); otherwise restart from t = 0.")


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
            "sf_sfclay_physics must be 0 (off), 1 (revised MM5), 2 (Eta "
            "similarity, the MYJ pair), 5 (MYNN), or 91 (classic MM5), got "
            f"{cfg.sf_sfclay_physics}."
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
            f"bl_pbl_physics must be 0 (none), 1 (YSU), "
            f"{MYJ_PBL_SCHEME} (MYJ), 5 (MYNN), 11 (Shin-Hong), or "
            f"{SASE_PBL_SCHEME} (SASE, experimental), got "
            f"{cfg.bl_pbl_physics}."
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
    # The carrier policy is validated HERE as well as at the seam that
    # consumes it, so a misspelled policy is refused at config load with
    # the two choices named rather than three minutes into a run.
    #
    # The vocabulary lives in THIS module, not in gpuwm.core, and the
    # import direction is the reason: the standalone RW-WPS preprocessing
    # wheel carries gpuwm.config and does NOT carry gpuwm.core, so a
    # config-load refusal that reached into the core would hand a
    # standalone user an ImportError instead of the sentence written for
    # them.  That is the exact defect that withdrew 1.8.8.
    validate_surface_radiation_policy(cfg.surface_radiation_policy)
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
    # W4 full admission (mf-close2 Stage B): bl_mynn_mixscalars leaves the
    # single-value identity table and is admitted at {0,1}.  The 1 arm is
    # pinned to the combo the anchored oracle fixture family
    # (w4-oracle-fixtures) was generated at and the runtime
    # was wired for: MYNN itself (bl_pbl_physics=5), the one scheme whose
    # state carries the qn family (mp_physics=28: nc/ni/nwfa/nifa), and
    # every-step PBL cadence (bldt=0) -- the qn tendencies are held as
    # plain-attribute extras outside the restart TENDENCY_COMPONENTS
    # manifest, which is restart-exact only when every compute() replaces
    # them before any read (gpuwm/core/physics.py scalar_for).
    if cfg.bl_mynn_mixscalars not in (0, 1) or \
            type(cfg.bl_mynn_mixscalars) is not int:
        raise ValueError(
            f"bl_mynn_mixscalars={cfg.bl_mynn_mixscalars!r} is outside the "
            "admitted MYNN option identity; gpuwm implements 0 (off) and 1 "
            "(the fixture-anchored stock qn mixing) only."
        )
    if cfg.bl_mynn_mixscalars == 1:
        if cfg.bl_pbl_physics != 5:
            raise NotImplementedError(
                "bl_mynn_mixscalars=1 requires bl_pbl_physics=5 (MYNN), "
                f"got bl_pbl_physics={cfg.bl_pbl_physics}. The mixscalars "
                "arms are MYNN's own qn solves (module_bl_mynn.F:4654-4860) "
                "and no other PBL scheme reads the key -- accepting it "
                "would record a mixing option no code performs."
            )
        if cfg.mp_physics != 28:
            raise NotImplementedError(
                "bl_mynn_mixscalars=1 requires mp_physics=28, got "
                f"mp_physics={cfg.mp_physics}. The five mixed species are "
                "the aerosol-aware Thompson qn family (nc/ni/nwfa/nifa; "
                "qnbca has no mp=28 field and rides as an exact zero); no "
                "other ported scheme carries them, so the solve would mix "
                "columns that do not exist."
            )
        if cfg.bldt != 0.0:
            raise NotImplementedError(
                f"bl_mynn_mixscalars=1 requires bldt=0, got {cfg.bldt!r}. "
                "The qn tendencies are held outside the restart "
                "TENDENCY_COMPONENTS manifest and are restart-exact only "
                "when recomputed every step (physics.py scalar_for, the rw "
                "precedent); a positive cadence would make a mid-interval "
                "restart silently drop them."
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
    if ra_lw_physics == 1 and ra_sw_physics != 1:
        # WRF RRTM longwave is implemented (gpuwm.core.rrtm_lw), but only
        # the pairing WRF's own classic namelists use is wired: RRTM
        # longwave with Dudhia shortwave.  RRTM-with-RRTMGP-shortwave
        # would be a scheme combination WRF never runs, and
        # RRTM-with-shortwave-off has no adapter, so both refuse here
        # rather than resolve to something nobody asked for.
        raise ValueError(
            "ra_lw_physics=1 (WRF RRTM longwave) is implemented only as "
            "WRF's classic pair with ra_sw_physics=1 (Dudhia shortwave), "
            f"got ra_sw_physics={ra_sw_physics}.")
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
    if cfg.mp_physics in _P3_UNPORTED_VARIANTS:
        # Refuse the sibling P3 options BY NAME rather than letting them
        # fall into the generic "unknown mp_physics" message: each is a real
        # WRF v4.6.1 package (Registry.EM_COMMON:3039-3041) that gpuwm's
        # mp=50 port deliberately does not cover, and a user who typed one
        # is owed the reason, not a claim that the number is meaningless.
        # The menu follows the reason: these are out-of-schema VALUES, and
        # a value refusal recites the menu here the same way as below.
        raise ValueError(unported_p3_variant_refusal(cfg.mp_physics))
    if cfg.mp_physics not in MP_PHYSICS_ACCEPTED:
        # WDM5 (14) and WDM7 (26) are named rather than left to the generic
        # tail: they are WDM6's siblings in the same WRF family and the
        # obvious next thing a WDM6 user types, so the refusal says which
        # one of the three is ported instead of listing nine integers.  It
        # then recites the menu, the same way the P3 siblings above do,
        # because these too are out-of-schema VALUES.
        if cfg.mp_physics in (14, 26):
            sibling = "WDM5" if cfg.mp_physics == 14 else "WDM7"
            raise ValueError(
                f"mp_physics={cfg.mp_physics} ({sibling}) is not ported. "
                "Of WRF's WDM family only WDM6 (mp_physics=16) exists in "
                f"gpuwm; {sibling} carries a different hydrometeor set "
                "(WDM5 has no graupel, WDM7 adds hail) and cannot be run "
                "by substituting WDM6 for it. "
                f"{_MP_PHYSICS_SCHEMA_MENU}, got {cfg.mp_physics}."
            )
        raise ValueError(f"{_MP_PHYSICS_SCHEMA_MENU}, got {cfg.mp_physics}.")
    validate_milbrandt2_options(cfg)
    validate_p3_radiation(cfg)
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
    _NSSL_SELECTOR_DEFAULTS = {
        "nssl_2moment_on": -1, "nssl_hail_on": -1, "nssl_ccn_on": -1,
        "nssl_density_on": -1, "nssl_3moment": 0,
    }
    if cfg.mp_physics == 18:
        # Resolve and gate the variant mode exactly once at validation, so
        # a rejected combination can never reach state allocation or the
        # scheme.  resolve applies WRF's module_check_a_mundo.F:3423-3465
        # consistency pass; require_ported refuses combinations without a
        # ported numerical path (1-moment, 3-moment, fixed-density, and
        # WRF's undefined inactive-pointer pairings).
        from gpuwm.core.nssl2_contract import (
            require_ported_nssl2_mode,
            resolve_nssl2_mode_for_config,
        )
        try:
            require_ported_nssl2_mode(resolve_nssl2_mode_for_config(cfg))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"NSSL variant selectors: {exc}") from exc
    else:
        for _nssl_key, _nssl_default in _NSSL_SELECTOR_DEFAULTS.items():
            if getattr(cfg, _nssl_key) != _nssl_default:
                raise ValueError(
                    f"{_nssl_key} is an NSSL selector and requires "
                    f"mp_physics=18 (got mp_physics={cfg.mp_physics}); "
                    "WRF zeroes stray NSSL flags under other schemes "
                    "(module_check_a_mundo.F:3423-3430), gpuwm refuses "
                    "them instead of silently dropping them."
                )
    if cfg.wdm6_hail_opt not in (0, 1):
        raise ValueError(
            "wdm6_hail_opt must be 0 (graupel, the WRF WDM6 default) or "
            f"1 (hail), got {cfg.wdm6_hail_opt}."
        )
    if not math.isfinite(cfg.wdm6_ccn_conc) or cfg.wdm6_ccn_conc <= 0.0:
        raise ValueError(
            "wdm6_ccn_conc must be a finite positive CCN number "
            "concentration in # m-3 (WRF Registry default 1.0e8), got "
            f"{cfg.wdm6_ccn_conc}."
        )
    if cfg.mp_physics == 16 and not (1.0e8 <= cfg.wdm6_ccn_conc <= 2.0e10):
        # module_mp_wdm6.F:585 clamps the CCN array to [1e8, 2e10] on every
        # entry to wdm62D, so a value outside the window is not the initial
        # condition the operator asked for -- the very first minor loop
        # discards it.  Refusing beats silently running a different run.
        raise ValueError(
            f"wdm6_ccn_conc={cfg.wdm6_ccn_conc} lies outside WDM6's own "
            "[1e8, 2e10] # m-3 clamp (module_mp_wdm6.F:585), so the scheme "
            "would overwrite it on the first minor time step; choose a "
            "value inside the window."
        )
    if cfg.mp_physics != 16:
        # Off-scheme WDM6 knobs are REFUSED, not dropped, on the NSSL
        # precedent (the nssl_* selectors under mp != 18): nothing outside
        # the mp=16 path reads either field, so a set value is a request
        # gpuwm would silently ignore.  The refusal is also what makes
        # gpuwm.core.model.SCHEME_SCOPED_RUN_FIELDS provably lossless --
        # under any other scheme these two can hold ONLY their defaults, so
        # dropping them from the restart identity discards no information.
        for _wdm6_key, _wdm6_default in (("wdm6_hail_opt", 0),
                                         ("wdm6_ccn_conc", 1.0e8)):
            if getattr(cfg, _wdm6_key) != _wdm6_default:
                raise ValueError(
                    f"{_wdm6_key} is a WDM6 selector and requires "
                    f"mp_physics=16 (got mp_physics={cfg.mp_physics}); "
                    "WRF reads hail_opt/ccn_conc only inside the schemes "
                    "that declare them, and gpuwm refuses a stray value "
                    "instead of silently dropping it."
                )
    if cfg.mp_physics != 28:
        # The mp=28 aerosol-source pair is scheme-scoped
        # (gpuwm.core.model.SCHEME_SCOPED_RUN_FIELDS), and this refusal is
        # what makes that scoping provably lossless: under any other scheme
        # the two can hold ONLY their defaults, so dropping them from the
        # restart identity discards no information.  The concrete breakage:
        # they landed unscoped at the 2.5.8 integration and moved every
        # experiment fingerprint, which would have made every existing
        # checkpoint refuse to resume for fields only mp=28 reads.
        for _mp28_key, _mp28_default in (("mp28_aerosol_source", "auto"),
                                         ("wif_climatology_path", "")):
            if getattr(cfg, _mp28_key) != _mp28_default:
                raise ValueError(
                    f"{_mp28_key} selects the mp=28 aerosol initial state "
                    f"and requires mp_physics=28 (got mp_physics="
                    f"{cfg.mp_physics}); no other scheme reads it, and "
                    "gpuwm refuses a stray value instead of silently "
                    "dropping it."
                )
    if cfg.p3_backend not in P3_BACKENDS:
        raise ValueError(
            f"p3_backend={cfg.p3_backend!r} is not one of "
            f"{sorted(P3_BACKENDS)}: 'cuda' is the shipping device arm "
            "(gpuwm/core/kernels/p3.cu, one kernel per step of the "
            "authority), 'fused' is the three-launch composition proven "
            "byte-identical to it, and 'reference' is the CPU float32 "
            "transcription kept for verification.  A fourth arm has to be "
            "byte-compared against 'cuda' before it can be named here.")
    if cfg.mp_physics != 50 and cfg.p3_backend != "cuda":
        # The P3 backend selector is scheme-scoped
        # (gpuwm.core.model.SCHEME_SCOPED_RUN_FIELDS), and this refusal is
        # what makes that scoping provably lossless: under any other scheme
        # it can hold ONLY its default, so dropping it from the restart
        # identity discards no information.  The concrete breakage it
        # prevents is the one the mp=28 aerosol pair caused for real at the
        # 2.5.8 integration -- an unscoped field moved every experiment
        # fingerprint, so every existing checkpoint would have refused to
        # resume for a knob only P3 reads.
        raise ValueError(
            f"p3_backend={cfg.p3_backend!r} selects the P3 implementation "
            f"and requires mp_physics=50 (got mp_physics={cfg.mp_physics}); "
            "no other scheme reads it, and gpuwm refuses a stray value "
            "instead of silently dropping it.")
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
