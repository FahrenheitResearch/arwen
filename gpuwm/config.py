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
    # 0 off, 1 Kessler, 6 WSM6, 8 Thompson, 10 Morrison, 18 NSSL
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
    bl_pbl_physics: int = 0     # 0 = none, 1 = YSU
    # WRF v4.6.1 Registry.EM_COMMON default; consumes radiative heating.
    ysu_topdown_pblmix: int = 1
    # --- Phase 4 (Task 1): non-timesplit radiation and cumulus slots.
    # ra_physics=90 is the Phase-3 analytic clear-sky proxy; 4 selects
    # RTE+RRTMGP.  cu_physics=1 is reserved for Kain-Fritsch.
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
#: WRF bl_pbl_physics values in gpuwm's schema.
PBL_SCHEMES = (0, 1, 5)
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

_KNOWN_TABLES = ("grid", "dynamics", "run")

def load_config(path: str | Path) -> RunConfig:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
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
            f"bl_pbl_physics must be 0 (none), 1 (YSU), or 5 (MYNN), got "
            f"{cfg.bl_pbl_physics}."
        )
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
            "seam, and with sf_sfclay_physics=0 nothing writes them."
        )
    if cfg.bl_pbl_physics and not cfg.sf_sfclay_physics:
        raise ValueError(
            f"bl_pbl_physics={cfg.bl_pbl_physics} requires a surface layer "
            "(sf_sfclay_physics != 0): the PBL scheme consumes UST, HFX, QFX, "
            "WSPD and RMOL from it, and with sf_sfclay_physics=0 they stay at "
            "their cold-start values for the whole run."
        )
    if cfg.cu_physics and not cfg.moist:
        raise ValueError(
            f"cu_physics={cfg.cu_physics} requires moist=true: Kain-Fritsch "
            "is a moist convective scheme and gpuwm/core/physics.py "
            "initialize_physics refuses a cumulus scheme on a dry DomainState "
            "(state.qv is None). The registry says the same thing through the "
            "option's required_settings."
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
            "RTE+RRTMGP (4) and analytic radiation (90) are coupled "
            "LW/SW adapters and must be selected on both components")
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
    if cfg.cu_physics not in (0, 1):
        raise ValueError(
            f"cu_physics must be 0 (off) or 1 (Kain-Fritsch), got "
            f"{cfg.cu_physics}."
        )
    if cfg.hypsometric_opt not in (1, 2):
        raise ValueError(
            "hypsometric_opt must be 1 (d(phi)/d(eta) hydrostatic "
            "inversion, the frozen gpuwm form) or 2 (WRF log-pressure "
            "form, calc_p_rho_phi), got "
            f"{cfg.hypsometric_opt}."
        )
    if cfg.mp_physics not in (0, 1, 6, 8, 10, 18):
        raise ValueError(
            "mp_physics must be 0 (off), 1 (Kessler), 6 (WSM6), 8 "
            "(Thompson), 10 (Morrison two-moment), or 18 "
            "(NSSL two-moment), got "
            f"{cfg.mp_physics}."
        )
    if cfg.mp_physics != 0 and not cfg.moist:
        raise ValueError(
            f"mp_physics={cfg.mp_physics} requires moist=true; the dry "
            "state does not allocate water or hydrometeor fields."
        )
    if cfg.nest_microphysics_transition not in (
            "same-scheme-only", "mp8-to-mp18-mass-diagnosed-v1"):
        raise ValueError(
            "nest_microphysics_transition must be 'same-scheme-only' or "
            "'mp8-to-mp18-mass-diagnosed-v1', got "
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
    if cfg.km_opt not in (1, 4):
        raise ValueError(
            f"km_opt must be 1 (constant K via khdif/kvdif) or 4 "
            f"(2-D Smagorinsky), got {cfg.km_opt}."
        )
    for name in ("khdif", "kvdif"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{name} must be a finite non-negative constant-K "
                f"diffusivity, got {value}."
            )
    if cfg.km_opt == 4 and (cfg.khdif > 0.0 or cfg.kvdif > 0.0):
        raise ValueError(
            "km_opt=4 selects WRF Smagorinsky mixing; khdif/kvdif are "
            "constant-K controls for km_opt=1 and cannot also be active."
        )
    if cfg.km_opt == 4 and cfg.bl_pbl_physics == 0:
        raise NotImplementedError(
            "km_opt=4 with bl_pbl_physics=0 is not yet supported: WRF "
            "diff_opt=2 also runs vertical_diffusion_2 when the PBL is "
            "off (using xkmv=xkmh for the u/v/w stress terms), and gpuwm "
            "does not yet implement that vertical operator and its surface-"
            "flux policy."
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
    validate_resolved_physics_vertical_levels(cfg)
    return cfg
