"""Rewrite physics_registry_v2.json from the verified knob tables.

Importing this module is side-effect free: nothing is read and nothing is
written until :func:`main` runs.  It used to write the registry at module
scope, so any ``import tools.build_registry`` -- a test collecting the tools
tree, a REPL, an editor's autocomplete -- silently rewrote a tracked file.

``tests/test_build_registry.py`` is the enforcement this file lacked: it runs
the builder into a temporary path and requires the bytes to equal the tracked
registry.  Without that gate the builder and the file it generates drift, and
they did: replaying the pre-fix builder reverted every citation commit 611396b
had just corrected.

Usage
-----
    python tools/build_registry.py [--out <path>]
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import pathlib
import re
import sys

# Repo-relative: this script lives in tools/, so the model root is its parent.
# The knob survey it consumes is committed beside it rather than read from a
# session-temp workflow journal, which would not have survived the session.
MODEL = pathlib.Path(__file__).resolve().parents[1]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

JOURNAL = MODEL / "tools" / "data" / "knob_survey_lanes.jsonl"
REGISTRY_PATH = MODEL / "gpuwm" / "physics_registry_v2.json"

from gpuwm.physics_registry import canonical_json  # noqa: E402
from gpuwm.wrf461_compatibility import (  # noqa: E402
    CUMULUS_OPTIONS,
    LAND_SURFACE_OPTIONS,
    MATRIX_CELL_COUNT,
    MP_OPTIONS,
    PBL_OPTIONS,
    PBL_SURFACE_LAYER_AUTHORITY,
    RADIATION_OPTIONS,
    SURFACE_LAYER_OPTIONS,
    WRF_COMMIT,
    WRF_VERSION,
    compatibility_cell,
    iter_compatibility_matrix,
)
# The prose-stripping rule belongs to the gate, so it is imported rather than
# reimplemented.  A second copy is how the citations drifted: this builder
# stripped only ``#`` comments while tools/check_parameter_claims.py stripped
# docstrings too, so the builder kept proposing citations the gate rejected.
from tools.check_parameter_claims import (  # noqa: E402
    UnparseableCitation,
    _code_without_prose,
)

# ---------------------------------------------------------------- implemented
# type / enum / minimum / default taken from gpuwm's own accepted sets
# (gpuwm/config.py validate_run_config), not from WRF's wider sets.
IMPLEMENTED: dict[str, dict] = {
    # acoustic / small step
    "time_step_sound": {"type": "integer", "minimum": 2, "default": 4},
    "smdiv": {"type": "number", "minimum": 0.0, "default": 0.1},
    "emdiv": {"type": "number", "minimum": 0.0, "default": 0.0},
    # explicit / constant-K mixing
    "khdif": {"type": "number", "minimum": 0.0, "default": 0.0},
    "kvdif": {"type": "number", "minimum": 0.0, "default": 0.0},
    "c_s": {"type": "number", "minimum": 0.0, "default": 0.25},
    "diff_6th_thresh": {"type": "number", "minimum": 0.0, "default": 0.10},
    # upper-level damping
    "damp_opt": {"type": "integer", "enum": [0, 3], "default": 0},
    "zdamp": {"type": "number", "minimum": 0.0, "default": 5000.0},
    "dampcoef": {"type": "number", "minimum": 0.0, "default": 0.2},
    "w_damping": {"type": "integer", "enum": [0, 1], "default": 0},
    # vertical coordinate and base state
    "hybrid_opt": {"type": "integer", "enum": [0, 1, 2], "default": 0},
    "etac": {"type": "number", "minimum": 0.0, "default": 0.2},
    "base_temp": {"type": "number", "minimum": 0.0, "default": 290.0},
    "hypsometric_opt": {"type": "integer", "enum": [1, 2], "default": 1},
    # transport
    "h_sca_adv_order": {"type": "integer", "enum": [2, 5], "default": 2},
    "moist_adv_opt": {"type": "integer", "enum": [0, 1], "default": 1},
    # microphysics heating controls
    "no_mp_heating": {"type": "integer", "enum": [0, 1], "default": 0},
    "mp_tend_lim": {"type": "number", "minimum": 0.0, "default": 10.0},
    # lateral boundaries
    "specified": {"type": "boolean", "default": False},
    "open_x": {"type": "boolean", "default": False},
    "open_y": {"type": "boolean", "default": False},
    "spec_bdy_width": {"type": "integer", "minimum": 1, "default": 5},
    "spec_zone": {"type": "integer", "minimum": 1, "default": 1},
    "relax_zone": {"type": "integer", "minimum": 2, "default": 4},
    # projection
    "map_proj": {"type": "integer", "enum": [0, 1, 2, 3], "default": 0},
    # boundary layer
    "ysu_topdown_pblmix": {"type": "integer", "enum": [0, 1], "default": 1},
    # radiation
    "swrad_scat": {"type": "number", "minimum": 0.0, "default": 1.0},
    "o3input": {
        "type": "integer", "enum": [0, 2], "default": 2,
        "warnings": [
            "o3input=0 is implemented only by ra_rrtmg_variant="
            "'rrtmg_legacy', where the WRF wrapper constructs O3DATA. "
            "RTE+RRTMGP admits only 2."]},
    "use_mp_re": {
        "type": "integer", "enum": [0, 1], "default": 1,
        "warnings": [
            "use_mp_re=0 is implemented only by ra_rrtmg_variant="
            "'rrtmg_legacy'; it disables the WRF microphysics effective-"
            "radius scheme table and makes the wrapper calculate radii."]},
    "ra_rrtmg_variant": {
        "type": "string",
        "enum": ["rte-rrtmgp", "rrtmg_legacy"],
        "default": "rte-rrtmgp"},
    # experiment feedback already executes in the native multi-domain runner.
    "feedback": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "warnings": [
            "DIVERGENCE from WRF's default 1: gpuwm defaults feedback to 0 "
            "to preserve every assembled one-way trajectory. feedback=1 is "
            "experimental and supported only by the native gpuwm run "
            "multi-domain executor; prepared hierarchy artifacts remain "
            "static one-way and refuse it."]},
    # surface layer -- newly ported this pass
    "isfflx": {"type": "integer", "enum": [0, 1], "default": 1},
    "isftcflx": {"type": "integer", "enum": [0, 1, 2], "default": 0},
    "iz0tlnd": {"type": "integer", "enum": [0, 1, 2], "default": 0},
    # Noah LSM -- newly ported: kernel branches already existed at
    # noah.cu:1036/:1111/:1112/:1116/:1117 (usemonalb, rdlai2d) and :181
    # (opt_thcnd); only the configuration path was missing.
    "usemonalb": {"type": "boolean", "default": False},
    "rdlai2d": {"type": "boolean", "default": False},
    "opt_thcnd": {"type": "integer", "enum": [1, 2], "default": 1},
    "rdmaxalb": {
        "type": "boolean", "default": True,
        "warnings": [
            "rdmaxalb=false is implemented only by Noah LSM "
            "(sf_surface_physics=2), whose LSMINIT replaces supplied SNOALB "
            "with VEGPARM MAXALB by vegetation category."]},
    "seaice_albedo_default": {
        "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.65,
        "warnings": [
            "A nondefault seaice_albedo_default is implemented only by RUC "
            "LSM (sf_surface_physics=3), in the live sea-ice ALBBCK "
            "override before LSMRUC."]},
    # experiment-scope knobs (whole tree, never per-domain)
    "p_top": {"type": "number", "minimum": 0.0},
    "blend_width": {"type": "integer", "minimum": 0, "default": 5},
    "co2_vmr": {"type": "number", "minimum": 0.0},
}

# Existing declarations that were looser than gpuwm's real accepted set.
TIGHTEN: dict[str, dict] = {
    "ra_physics": {"type": "integer", "enum": [0, 4, 90], "default": 0},
    # -v2 is the assembly resolution of 2026-07-28 (gpuwm/physics_compat.py
    # WRF_RRTMG_TO_RTE_RRTMGP): the tracked registry was rebound to -v2 in
    # that pass but this table still said -v1, so the builder no longer
    # reproduced the tracked bytes and regenerating would have silently
    # reverted the ratified token.  -v1 stays accepted by the CODE for
    # historical receipts (gpuwm/config.py); the registry enum governs what
    # a NEW plan may set, which is the current token only.
    "wrf_rrtmg_compatibility": {
        "type": "string",
        "enum": [
            "none",
            "wrf-rrtmg-4-4-to-rte-rrtmgp-v2",
            "wrf-rrtmg-4-4-legacy-v1",
        ],
        "default": "none"},
    "km_opt": {"type": "integer", "enum": [1, 4], "default": 1},
    "diff_6th_opt": {"type": "integer", "enum": [0, 1, 2], "default": 0},
    "diff_6th_slopeopt": {"type": "integer", "enum": [0, 1], "default": 0},
    # enum [4, 9], not minimum 1.  ``minimum: 1`` advertised 1, 2, 3, 5, 6,
    # 7, 8 as type-legal, and gpuwm/config.py:376 accepts none of them -- the
    # declaration was looser than the code it declares.  The enum is the set
    # of soil-layer counts this registry's own land-surface options carry: 4
    # for Noah and Noah-MP, 9 for RUC.
    #
    # It is NOT enum [4], which is what the code alone would once have said.
    # Nine has to stay type-legal for the component layer to be the thing that
    # speaks: tests/test_physics_registry.py requires a domain asking for
    # num_soil_layers=9 under NOAH to raise component-required-setting, and a
    # parameter-value rejection would pre-empt that -- the value never reaches
    # ``settings``, so the option's required_settings=4 sees the option's own
    # 4 still in place and never fires.
    #
    # The warning was written when RUC was refused and said three things that
    # are no longer true: that only 4 is accepted, that no physics component
    # reads the knob, and that the nine-layer identity was unported.  RUC is
    # admitted at nine layers and runs a forecast, so the text now describes
    # the resolver that actually decides.
    "num_soil_layers": {
        "type": "integer", "enum": [4, 9], "default": 4,
        "warnings": [
            "The resolved layer count comes from the SCHEME, not from this "
            "knob: gpuwm/config.py soil_layer_count consults "
            "LAND_SURFACE_SOIL_LAYERS for the selected sf_surface_physics, so "
            "Noah and Noah-MP resolve 4 and RUC resolves 9, and every soil "
            "allocation, VRAM count, output dimension and restart shape reads "
            "that resolver. DIVERGENCE from WRF, deliberate: "
            "set_physics_rconfigs OVERWRITES a namelist request that "
            "disagrees with the scheme and only logs it at debug level, so a "
            "namelist asking Noah for nine layers runs on four with no error; "
            "gpuwm refuses instead. Both 4 and 9 are selectable -- 4 with "
            "Noah or Noah-MP, 9 with RUC -- and no other value is. WRF's "
            "six-level RUC grid (share/module_soil_pre.F:init_soil_depth_3) "
            "is not declared here because every RUC oracle fixture in the "
            "tree is nine-level and the CUDA leaves index a __constant__ real "
            "ruc_soil_layer_depth[9]."]},
    "terrain_opt": {"type": "integer", "enum": [0, 1], "default": 0},
    "epssm": {"type": "number", "minimum": 0.0, "maximum": 1.0,
              "default": 0.1},
    "diff_6th_factor": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                        "default": 0.12},
    "moist": {"type": "boolean", "default": False},
    "moist_cq": {"type": "boolean", "default": False},
    "top_lid": {"type": "boolean", "default": True},
    "morr_rimed_ice": {"type": "integer", "enum": [0, 1], "default": 1},
    "wsm6_hail_opt": {"type": "integer", "enum": [0, 1], "default": 0},
    "icloud": {"type": "integer", "enum": [0, 1], "default": 1},
    "radt": {"type": "number", "minimum": 0.0, "default": 0.0},
    "radt_minutes": {"type": "number", "minimum": 0.0, "default": 12.0},
    "bldt": {"type": "number", "minimum": 0.0, "default": 0.0},
    "cudt_minutes": {"type": "number", "minimum": 0.0, "default": 5.0},
}

WRF_TYPE = {"integer": "integer", "real": "number",
            "logical": "boolean", "character": "string"}

# Lane K's final ledger for declarations that remain unavailable.  The class
# is kept beside the blocker so the generated registry and the handoff cannot
# drift into calling a bounded option branch a new subsystem.
UNIMPLEMENTED_LEDGER: dict[str, tuple[str, str]] = {
    "aer_opt": (
        "c",
        "Radiation aerosol optics are absent: no aerosol optical-depth/"
        "single-scattering/asymmetry state, ingest, restart carriers, or "
        "legacy/RTE radiation-kernel binding exists."),
    "aercu_fct": (
        "c",
        "This belongs to the unported multiscale Kain-Fritsch aerosol-aware "
        "cumulus scheme; gpuwm's cu_physics set contains only off and the "
        "ported KF scheme, which has no AERCU tendency state."),
    "aercu_opt": (
        "c",
        "This selects aerosol-aware behavior in the unported multiscale "
        "Kain-Fritsch scheme; gpuwm has no MSKF component or its aerosol "
        "state/activation subsystem."),
    "brcr_ub": (
        "c",
        "Not a WRF v4.6.1 namelist option. BRCR_UB is a YSU scheme-internal "
        "constant compiled into the existing kernel, so there is no legal "
        "WRF user setting to expose."),
    "cu_diag": (
        "c",
        "The cumulus-diagnostics subsystem is absent: gpuwm carries none of "
        "WRF's per-step/per-output convective diagnostic accumulators, "
        "restart state, or wrfout variables."),
    "cu_rad_feedback": (
        "c",
        "KF radiation feedback needs persistent convective cloud fraction, "
        "condensate-path profiles, cadence ownership, restart state, and a "
        "radiation-driver merge; the ported KF contract returns none of "
        "those profiles."),
    "cu_used": (
        "c",
        "Not a WRF v4.6.1 namelist option. CU_USED is derived by WRF from "
        "the selected cumulus scheme and domain state, not legally set by a "
        "user."),
    "fractional_seaice": (
        "b",
        "The component branches exist only partially: land-use initialization "
        "and RUC/MYNN wrappers disagree on thresholds and several ingest paths "
        "hardwire opposite modes. A bounded transcription must make one "
        "option govern ingest, surface-layer deblend/reblend, and LSM routing "
        "with WRF oracle coverage."),
    "icloud_cu": (
        "c",
        "Not a WRF v4.6.1 namelist option. ICLOUD_CU is derived cloud-state "
        "routing inside WRF's cumulus/radiation drivers, not a user knob."),
    "ifsnow": (
        "c",
        "IFSNOW controls snow physics in WRF's slab/thermal-diffusion land "
        "surface schemes; those schemes are not ported, and Noah/RUC/"
        "Noah-MP do not consume this selector."),
    "ishallow": (
        "c",
        "ISHALLOW belongs to the unported Grell 3-D cumulus family, not the "
        "ported Kain-Fritsch component; exposing it would require that new "
        "scheme."),
    "kf_edrates": (
        "c",
        "The KF entrainment/detrainment diagnostic output subsystem is "
        "absent: the ported kernel does not return the rate profiles and "
        "gpuwm has no carriers, restart identity, or wrfout variables for "
        "them."),
    "kfeta_trigger": (
        "b",
        "This is an option branch inside the already ported KF scheme, but "
        "the alternate trigger paths and their moisture-advective-tendency "
        "input are not transcribed into the KF kernel/driver or oracle "
        "fixtures."),
    "naer": (
        "c",
        "This Thompson aerosol-aware droplet mode requires prognostic aerosol/"
        "CCN number state and activation coupling that the ported fixed-"
        "aerosol Thompson component does not allocate or restart."),
    "nssl_2moment_on": (
        "c",
        "Disabling NSSL two-moment changes the scheme identity, prognostic "
        "number fields, kernel contract, restart inventory, and nest "
        "transition semantics; only the NSSL two-moment component is ported."),
    "nssl_3moment": (
        "c",
        "NSSL three-moment needs additional prognostic moments, allocations, "
        "kernel equations, restart/wrfout fields, and nest transitions; the "
        "ported component is two-moment only."),
    "nssl_alphah": (
        "b",
        "The hail gamma-shape option is a bounded branch in the already "
        "ported NSSL scheme, but its value is still compiled into coefficient "
        "setup and is not carried through RunConfig, kernel arguments, or "
        "independent WRF oracle fixtures."),
    "nssl_alphahl": (
        "b",
        "The large-hail gamma-shape option is a bounded NSSL coefficient "
        "branch, but config plumbing, device coefficient regeneration, and "
        "two-value WRF oracle fixtures are absent."),
    "nssl_alphar": (
        "c",
        "Not a WRF v4.6.1 namelist option. NSSL_ALPHAR is a scheme-internal "
        "rain-shape constant and has no legal user setting."),
    "nssl_cccn": (
        "b",
        "The initial CCN concentration is a bounded NSSL setup value, but "
        "gpuwm hardwires QNN cold-start initialization and has no RunConfig/"
        "restart-bound path or two-value WRF initialization oracle."),
    "nssl_ccn_on": (
        "b",
        "The NSSL CCN-number branch is inside the ported scheme and QNN state "
        "exists, but the on/off coefficient and tendency paths are not wired "
        "through setup/kernel arguments or covered by independent WRF "
        "oracles."),
    "nssl_density_on": (
        "c",
        "Variable-density NSSL hydrometeors require density prognostic state, "
        "additional kernel equations, restart/wrfout fields, and transition "
        "semantics; the fixed-density NSSL two-moment identity is the only "
        "ported one."),
    "nssl_ehlw0": (
        "c",
        "Not a WRF v4.6.1 namelist option. NSSL_EHLW0 is a scheme-internal "
        "collection-efficiency constant and has no legal user setting."),
    "nssl_ehw0": (
        "c",
        "Not a WRF v4.6.1 namelist option. NSSL_EHW0 is a scheme-internal "
        "collection-efficiency constant and has no legal user setting."),
    "nssl_hail_on": (
        "c",
        "Changing the NSSL hail-species mode changes prognostic species, "
        "kernel/state binding, restart inventory, radiation radii, and nest "
        "transition semantics; only the hail-enabled NSSL two-moment identity "
        "is ported."),
    "nssl_icdx": (
        "b",
        "The NSSL ice-distribution selector is a bounded coefficient branch "
        "inside the ported scheme, but the alternate initialization/table "
        "path is not transcribed or oracle-tested and has no config plumbing."),
    "nssl_icdxhl": (
        "b",
        "The NSSL large-hail distribution selector is a bounded coefficient "
        "branch, but its alternate table/setup path, device plumbing, and "
        "two-value WRF oracle are absent."),
    "num_land_cat": (
        "c",
        "WRF derives this count from the selected land-use dataset/table. "
        "Supporting another value requires alternate static categories, "
        "parameter tables, field validation, and state dimensions; it is not "
        "an independent runtime tune in gpuwm."),
    "num_soil_cat": (
        "c",
        "WRF derives this count from the selected soil-category dataset/table. "
        "Arbitrary values require alternate static categories, parameter "
        "tables, validation, and allocations rather than a scalar branch."),
    "progn": (
        "c",
        "Thompson prognostic cloud-droplet number requires additional "
        "prognostic number/aerosol state, activation equations, restart and "
        "wrfout fields; the ported Thompson identity uses diagnostic droplets."),
    "scm_force_flux": (
        "c",
        "This belongs to WRF's single-column-model forcing subsystem; gpuwm "
        "has no SCM runner, forcing time series, column boundary contract, or "
        "restart semantics."),
    "seaice_albedo_opt": (
        "c",
        "The option is consumed by WRF's separate Noah sea-ice thermodynamics "
        "driver (including the Mills branch and optional ALBSI field); that "
        "driver and field are absent, so the RUC ALBBCK override is not a "
        "substitute."),
    "seaice_snowdepth_max": (
        "c",
        "This bound is consumed only by the absent Noah sea-ice "
        "thermodynamics driver; gpuwm has no sea-ice snow-depth state or "
        "surface-driver call on which it could act."),
    "seaice_snowdepth_min": (
        "c",
        "This bound is consumed only by the absent Noah sea-ice "
        "thermodynamics driver; gpuwm has no sea-ice snow-depth state or "
        "surface-driver call on which it could act."),
    "seaice_snowdepth_opt": (
        "c",
        "The selector belongs to the absent Noah sea-ice thermodynamics "
        "driver and its optional SNOWSI input; no such state, ingest, restart, "
        "or kernel contract exists."),
    "seaice_thickness_default": (
        "c",
        "WRF consumes this in the absent Noah sea-ice thermodynamics driver "
        "when seaice_thickness_opt=0. The unrelated 3 m cold-start soil "
        "interpolation literal is not this knob and cannot honor it."),
    "seaice_thickness_opt": (
        "c",
        "The selector belongs to the absent Noah sea-ice thermodynamics "
        "driver; option 1 additionally requires ICEDEPTH ingest/state/restart "
        "carriers that gpuwm does not have."),
    "seaice_threshold": (
        "c",
        "SEAICE_THRESHOLD is consumed by WRF's unported slab land-surface "
        "scheme. The ported LSMs use their own XICE_THRESHOLD contracts and "
        "cannot honor this different selector."),
    "sf_surface_mosaic": (
        "c",
        "Surface mosaics require a tile dimension, per-tile land-use/soil "
        "fractions and LSM states, aggregation, restart/wrfout contracts, and "
        "driver loops; gpuwm carries one surface column per grid cell."),
    "shallowcu_forced_ra": (
        "c",
        "Forced shallow-cumulus radiation needs persistent shallow-convective "
        "cloud/condensate profiles and radiation-cadence merge state; neither "
        "the shallow-cumulus component nor those carriers are ported."),
    "shalwater_depth": (
        "c",
        "The shallow-water surface branch needs a bathymetry/depth static "
        "field and its surface coupling; gpuwm ingests neither and has no "
        "runtime branch to consume the scalar."),
    "shalwater_z0": (
        "c",
        "The shallow-water roughness branch needs the missing shallow-water/"
        "bathymetry classification and surface-driver coupling; setting a "
        "scalar alone would be inert."),
    "shcu_physics": (
        "c",
        "This selects independent shallow-cumulus schemes. No shallow-"
        "cumulus component, tendencies, cadence state, restart contract, or "
        "radiation coupling is ported."),
    "slope_rad": (
        "c",
        "Slope-aware radiation needs slope/aspect terrain statics and modified "
        "solar-incidence geometry in the radiation driver; those fields and "
        "branches are absent."),
    "smooth_option": (
        "c",
        "WRF nest smoothing requires parent/child feedback-time smoothing "
        "kernels and halo/ownership semantics. gpuwm has no smoothing "
        "operator; prepared hierarchy artifacts are explicitly static one-way."),
    "sst_update": (
        "c",
        "SST updates require a time-varying lower-boundary input stream, "
        "interpolation/cadence state, restart position, and surface ownership; "
        "gpuwm cases use one analysis-time lower boundary."),
    "surface_input_source": (
        "c",
        "Alternate surface-input sources require source-specific static/"
        "met-field selection and provenance semantics. gpuwm's ingest routes "
        "own those choices explicitly and implement no interchangeable "
        "runtime source selector."),
    "swint_opt": (
        "c",
        "Shortwave interpolation needs carried previous/next radiation fields "
        "and per-step interpolation ownership; WRF's additional FARMS choice "
        "also needs an unported solver. gpuwm recomputes on radiation cadence "
        "and carries neither subsystem."),
    "tice2tsk_if2cold": (
        "b",
        "This is a bounded branch in the existing fractional-sea-ice surface "
        "wrapper, but gpuwm currently transcribes only the false arithmetic. "
        "The true get_local_ice_tsk correction must be ported together with "
        "the coherent fractional_seaice option and WRF oracle fixtures."),
    "tmn_update": (
        "c",
        "Updating deep-soil temperature needs a running/calendar mean "
        "algorithm, lower-boundary history, restart state, and ownership "
        "across ingest and LSM cadence; gpuwm carries a fixed analysis TMN."),
    "topo_shading": (
        "c",
        "Terrain shadowing needs horizon/terrain statics, solar azimuth and "
        "shadow geometry, plus radiation-driver state; none is carried."),
    "topo_wind": (
        "c",
        "Topographic wind correction needs terrain-subgrid static fields and "
        "the associated momentum-adjustment subsystem; gpuwm has neither the "
        "inputs nor a runtime operator."),
    "ua_phys": (
        "c",
        "Noah unified-atmosphere coupling is a separate physics path with "
        "additional state and feedback semantics; gpuwm ports the ordinary "
        "Noah LSM driver only."),
    "use_aero_icbc": (
        "c",
        "This belongs to Thompson aerosol-aware microphysics (WRF option 28) "
        "and requires aerosol initial/boundary-condition species, ingest, "
        "nesting, restart, and activation kernels; that scheme is not ported."),
}

NOISE = re.compile(
    r"^(MISSED BY THE CANDIDATE LIST[^.]*\.|MISFILED IN THE CANDIDATE LIST[^.]*\.|"
    r"Adjudication:\s*|DERIVED, not namelist[^.]*\.)\s*", re.I)
CASE_TOKEN = re.compile(r"real74|hrrr|ohio|oklahoma|may1999|20cr|1974", re.I)

# Rows this pass claims. Every other declarer's rows are left exactly as
# written: with no citation they assert nothing, and their owner turns one on
# by citing the read that makes it true.
OWNED = list(IMPLEMENTED) + list(TIGHTEN) + [
    "spec_exp", "nest_microphysics_transition"]


def clean(reason: str) -> str:
    text = " ".join((reason or "").split())
    text = NOISE.sub("", text)
    text = CASE_TOKEN.sub("the reference configuration", text)
    if len(text) > 220:
        cut = text[:220]
        stop = max(cut.rfind(". "), cut.rfind("; "))
        text = (cut[: stop + 1] if stop > 80 else cut.rstrip() + "...")
    return text.strip()


# ------------------------------------------------------ cited consuming reads
# A knob counts as implemented because a citation proves GPUWM reads it, never
# because someone asserted a flag.  Resolving the citation against the same
# rule the gate applies means the claim and its evidence cannot drift apart,
# and it leaves every other declarer's rows untouched: a row with no citation
# makes no claim, so an owner flips their own knob on by citing the read that
# makes it true.
SEARCH_ROOT = "gpuwm"
#: The registry loader names every knob as a dict key, so it matches all of
#: them and proves nothing.  gpuwm/config.py is deliberately NOT skipped: it
#: declares and validates the knobs, which is the only read some of them have.
SKIP = {"gpuwm/physics_registry.py"}
CONFIG = "gpuwm/config.py"

_files: list[str] | None = None
_sources: dict[str, tuple[str, frozenset[str]] | None] = {}


def _searchable_files() -> list[str]:
    """Repo-relative Python files a citation may name, in a stable order.

    Sorted on the POSIX relative path rather than on ``Path`` objects, whose
    ordering depends on the platform separator and on case folding, so the
    citation this picks does not depend on which machine ran the builder.
    """
    global _files
    if _files is None:
        found = []
        for path in (MODEL / SEARCH_ROOT).rglob("*.py"):
            rel = path.relative_to(MODEL).as_posix()
            if rel in SKIP or "__pycache__" in rel:
                continue
            # A generic knob must cite a generic reader.  Citing a
            # source-specific adapter would say the knob only exists for one
            # data source, which is the specialization the case-token gate
            # exists to prevent.
            if CASE_TOKEN.search(path.name):
                continue
            found.append(rel)
        _files = sorted(found)
    return _files


def _identifiers(tree: ast.AST) -> frozenset[str]:
    """Names the module binds or reads as identifiers, not as text.

    ``cfg.isftcflx``, ``num_soil_layers: int`` and ``radt=radt`` are reads the
    interpreter performs.  ``{"num_soil_layers": 9}`` is a string that happens
    to spell a knob; it can be a read (``getattr(cfg, "moist_cq", True)``) but
    it can equally be an unrelated namelist table, so it ranks lower.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            names.add(node.arg)
    return frozenset(names)


def _source(rel: str) -> tuple[str, frozenset[str]] | None:
    """Prose-stripped code and identifier set, or None if it proves nothing."""
    if rel not in _sources:
        path = MODEL / rel
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            _sources[rel] = None
        else:
            try:
                code = _code_without_prose(text, path.suffix)
                tree = ast.parse(text)
            except (UnparseableCitation, SyntaxError, ValueError):
                # The gate fails an unparseable citation, so the builder must
                # never propose one.
                _sources[rel] = None
            else:
                _sources[rel] = (code, _identifiers(tree))
    return _sources[rel]


def reads_knob(rel: str, name: str) -> bool:
    """Whether ``rel`` satisfies the shipped gate as a citation of ``name``.

    This is tools/check_parameter_claims' own test: the file must parse, and
    the knob must survive prose stripping.  A docstring mention does not
    count.
    """
    if not (MODEL / rel).is_file():
        return False
    source = _source(rel)
    if source is None:
        return False
    return re.search(rf"\b{re.escape(name)}\b", source[0]) is not None


def _rank(rel: str, name: str) -> int:
    """Lower is a better citation for ``name``.

    A runtime component that executes on the knob is the strongest evidence,
    so ``gpuwm/core`` identifier reads come first and other packages next.
    ``gpuwm/config.py`` is where the knob is declared and validated, so it is
    cited only when nothing consumes the value -- which is the whole story for
    ``num_soil_layers``.  A name that appears only inside string literals
    ranks below every identifier read, because a namelist key spells a knob
    without reading it.
    """
    source = _source(rel)
    if source is None:  # reads_knob() already excluded these
        return 5
    if name in source[1]:
        if rel.startswith(SEARCH_ROOT + "/core/"):
            return 0
        return 2 if rel == CONFIG else 1
    return 3 if rel.startswith(SEARCH_ROOT + "/core/") else 4


def consuming_read_candidates(name: str) -> list[str]:
    """Every file that satisfies the gate for ``name``, best citation first."""
    return sorted(
        (rel for rel in _searchable_files() if reads_knob(rel, name)),
        key=lambda rel: (_rank(rel, name), rel))


def find_consuming_read(name: str, current: str | None = None) -> str | None:
    """Return a repo-relative file whose code reads ``name``.

    ``current`` -- the citation already in the registry -- wins whenever it is
    still a file this builder would accept and still reads the knob.  Which
    file *consumes* a knob is a judgement someone made by reading the code,
    and a search cannot re-derive it: several files read the same knob and only
    one of them is the component that acts on it.  So the search proposes
    rather than overrules, and a citation is replaced only once it stops being
    true, which is exactly when the claim it backs has outlived its evidence.
    """
    candidates = consuming_read_candidates(name)
    if current and current in candidates:
        return current
    return candidates[0] if candidates else None


# --------------------------------------------------- verified per-domain data
# One-way four-domain chains transcribed from the verified nested
# configurations; the values are NOT derived from a grid-spacing scaling law.
#
# Each template gets its own list, because the two chains are not the same
# chain.  Both descend 12 km -> 3 km -> 1 km on parent grid ratios 1/4/3, and
# then they diverge: the reference chain closes on parent_grid_ratio 3
# (1000/3 m, published as the nominal 333 m) and the validation-candidate
# chain closes on parent_grid_ratio 2 (500 m).  Those ratios were read from
# the two four-domain experiment configurations, not inferred.
#
# One shared list used to serve both, so index 3 published 333 m for a
# template whose fourth domain is 500 m.  nominal_dx_m is provenance for
# display and never resolves into a setting, so the wrong value mislabelled a
# run rather than mis-integrating it -- which is exactly why no numerical gate
# could catch it, and why the two lists are kept apart rather than shared with
# one entry overridden.
NEST_COLUMNS: dict[str, list[dict]] = {
    "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1": [
        {"nominal_dx_m": 12000.0, "diff_6th_factor": 0.12, "radt": 12.0,
         "epssm": 0.5},
        {"nominal_dx_m": 3000.0, "diff_6th_factor": 0.10, "radt": 3.0,
         "epssm": 0.1},
        {"nominal_dx_m": 1000.0, "diff_6th_factor": 0.08, "radt": 1.0,
         "epssm": 0.1},
        # parent_grid_ratio 3 below the 1 km nest.
        {"nominal_dx_m": 333.0, "diff_6th_factor": 0.06, "radt": 1.0,
         "epssm": 0.1},
    ],
    "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-validation-candidate-v1": [
        {"nominal_dx_m": 12000.0, "diff_6th_factor": 0.12, "radt": 12.0,
         "epssm": 0.5},
        {"nominal_dx_m": 3000.0, "diff_6th_factor": 0.10, "radt": 3.0,
         "epssm": 0.1},
        {"nominal_dx_m": 1000.0, "diff_6th_factor": 0.08, "radt": 1.0,
         "epssm": 0.1},
        # parent_grid_ratio 2 below the 1 km nest, not 3.
        {"nominal_dx_m": 500.0, "diff_6th_factor": 0.06, "radt": 1.0,
         "epssm": 0.1},
    ],
    # The default full-physics suite (`gpuwm domain` emits it): columns
    # transcribed from the Thompson matched-run campaign geometry
    # (configs/real74_thompson_1218z_rrtmg_legacy_4dom.toml), whose d04
    # refines the 1 km nest by 2 exactly as the NSSL-2 ladder does.
    "thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1": [
        {"nominal_dx_m": 12000.0, "diff_6th_factor": 0.12, "radt": 12.0,
         "epssm": 0.5},
        {"nominal_dx_m": 3000.0, "diff_6th_factor": 0.10, "radt": 3.0,
         "epssm": 0.1},
        {"nominal_dx_m": 1000.0, "diff_6th_factor": 0.08, "radt": 1.0,
         "epssm": 0.1},
        {"nominal_dx_m": 500.0, "diff_6th_factor": 0.06, "radt": 1.0,
         "epssm": 0.1},
    ],
}


def _surface_coupling_warnings(registry: dict) -> None:
    """Replace the v1.1 surface-seam restrictions retired in v1.2."""
    land = registry["components"]["land_surface"]["options"]
    noahmp = land["noah-mp"]["warnings"]
    noahmp[4] = (
        "WRF's SIX-RATE precipitation seam is active. The surface driver "
        "carries convective, nonconvective, shallow-convective, snow, "
        "graupel and hail accumulations from their real writers, and "
        "noahmplsm takes module_sf_noahmpdrv.F:776-789's PRESENT(MP_*) "
        "branch including PRCPOTHR.")
    noahmp[5] = (
        "COSZEN is a radiation-driver carrier. Noah-MP consumes the last "
        "radiation call's value unchanged between calls, including radconst's "
        "half-radiation-interval hour-angle offset. A radiation-free run "
        "still binds start time and latitude/longitude and seeds the same "
        "offset value once.")
    noahmp[8] = (
        "When the MYNN 5/5 pairing is selected, WRF v4.6.1 runs the MYNN "
        "surface layer first, then NOAHMP_SFLX overwrites "
        "TSK/HFX/QFX/LH on its active land columns while leaving "
        "UST/CHS/CHS2/CQS2/FLHC/FLQC with MYNN. The surface-driver post-pass "
        "unconditionally replaces MYNN's T2/Q2/TH2 with Noah-MP's "
        "water/urban/vegetated diagnostics before MYNN PBL consumes them. "
        "The source transcription and writer-order tests are in "
        "tools/mynn_surface_pairing_wrf461_oracle and "
        "tests/test_mynn_surface_pairing_ownership.py.")
    noahmp_option = land["noah-mp"]
    noahmp_option["constraints"]["requires_components"]["surface_layer"] = [
        "revised-mm5", "classic-mm5", "mynn"]
    noahmp_option["extensions"]["mynn_surface_ownership"] = {
        "surface_layer_writes_first": [
            "ust", "hfx", "qfx", "chs", "chs2", "cqs2", "flhc", "flqc",
            "t2", "q2", "th2"],
        "lsm_overwrites": ["tsk", "hfx", "qfx", "lh"],
        "lsm_preserves": [
            "ust", "chs", "chs2", "cqs2", "flhc", "flqc"],
        "post_lsm_overwrites": ["t2", "q2", "th2"],
        "wrf_source": (
            "phys/module_surface_driver.F:3127-3181,3324-3370; "
            "phys/module_sf_noahmpdrv.F:1206-1207,1223-1285"),
    }

    ruc = land["ruc-lsm"]["warnings"]
    ruc[3] = (
        "WRF-ARW's EM_CORE==1 surface couplings are active: LSMRUC consumes "
        "RAINNCV/SNOWNCV/GRAUPELNCV, LAKEMASK bypasses the column core, "
        "fractional sea ice is deblended before and reblended after the call, "
        "and GSW is carried from radiation cadence. The historical EM_CORE=0 "
        "oracle replay remains explicit-only; independent source "
        "transcription probes cover the ARW seam.")
    ruc[9] = (
        "Mosaic land-use and soil remain fail-closed at 0, spp_lsm at 0 and "
        "flag_sm_adj at 0. Also pinned rather than configurable: "
        "XICE_THRESHOLD=0.5, isncovr_opt=2, c1sn=0.026, c2sn=21.0, "
        "myj=False and rdlai2d=False. seaice_albedo_default is configurable "
        "over [0,1] and defaults to the former literal 0.65. "
        "FRACTIONAL_SEAICE follows WRF's enabled pre/post coupling.")
    ruc[8] = (
        "Admitted at num_soil_layers=9 with revised MM5, classic MM5 or "
        "MYNN surface. Under the MYNN 5/5 pairing, WRF v4.6.1 runs the surface "
        "layer first; LSMRUC then overwrites TSK/HFX/QFX/LH, preserves "
        "UST/FLHC/FLQC/CHS2/CQS2, and the driver recomputes CHS from FLHC. "
        "SFCDIAGS_RUCLSM unconditionally replaces MYNN's T2/Q2/TH2 before "
        "MYNN PBL consumes the post-LSM fields. This is HRRR's operational "
        "MYNN/MYNN/RUC pairing class. The source transcription and "
        "writer-order tests are in tools/mynn_surface_pairing_wrf461_oracle "
        "and tests/test_mynn_surface_pairing_ownership.py.")
    ruc_option = land["ruc-lsm"]
    ruc_option["constraints"]["requires_components"]["surface_layer"] = [
        "revised-mm5", "classic-mm5", "mynn"]
    ruc_option["extensions"]["mynn_surface_ownership"] = {
        "surface_layer_writes_first": [
            "ust", "hfx", "qfx", "chs", "chs2", "cqs2", "flhc", "flqc",
            "t2", "q2", "th2"],
        "lsm_overwrites": ["tsk", "hfx", "qfx", "lh"],
        "lsm_preserves": ["ust", "flhc", "flqc", "chs2", "cqs2"],
        "post_lsm_overwrites": ["chs", "t2", "q2", "th2"],
        "wrf_source": (
            "phys/module_surface_driver.F:3500-3528,3579-3592; "
            "phys/module_sf_ruclsm.F:219-230,284-303"),
    }

    registry["parameters"]["spp_lsm"]["warnings"][0] = (
        "Only spp_lsm=0 is honoured. The ARW/EM_CORE==1 surface path is "
        "ported, but its stochastic pattern_spp_lsm and field_sf inputs are "
        "not; enabling spp_lsm would require that separate stochastic state "
        "and restart contract. Validation and the RUC runtime both refuse a "
        "nonzero value.")

    route_text = (
        "The Noah-MP glacier refusal and sea-ice skip still apply. Its "
        "six-rate precipitation seam and radiation-cadence COSZEN carrier "
        "now follow WRF v4.6.1.")
    for route in (
            "tools.hrrr_single_domain_benchmark",
            "tools.prepared_domain_tree_forecast",
            "tools.prepared_single_domain_forecast"):
        registry["runner_routes"][route]["expert_warnings"][2] = route_text

    template = registry["templates"][
        "wsm6-ysu-mm5-noahmp-no-radiation-expert-only-v1"]
    template["warnings"][0] = (
        "EXPERT ONLY. The throughput reason this template was originally "
        "gated on is retired: the whole column runs on the device and is "
        "bitwise against the scalar authority. It stays expert-only because "
        "no gpuwm/WRF forecast trajectory comparison exists. Dudhia supplies "
        "the carried COSZEN at radiation cadence; a radiation-free Noah-MP "
        "run instead seeds that carrier once from explicit geometry.")

    templates = registry["templates"]
    mynn_noah = templates[
        "wsm6-mynn-mynn-noah-no-radiation-implemented-unverified-v1"]
    mynn_ruc_id = (
        "wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1")
    mynn_ruc = copy.deepcopy(templates[
        "wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1"])
    mynn_ruc["components"]["pbl"] = "mynn"
    mynn_ruc["components"]["surface_layer"] = "mynn"
    mynn_ruc["label"] = (
        "WSM6 + MYNN PBL + MYNN surface layer + RUC LSM + Dudhia SW "
        "(HRRR pairing)")
    inherited_ruc_warnings = [
        warning for warning in mynn_ruc["warnings"]
        if not warning.startswith("This template differs from ")
    ]
    mynn_ruc["warnings"] = [
        mynn_noah["warnings"][0],
        (
            "This is the HRRR operational pairing class. WRF owns its "
            "write-back sequence explicitly: MYNN surface first, RUC "
            "flux/state write-back second, CHS and SFCDIAGS_RUCLSM last. "
            "It is offered only on routes where the established RUC template "
            "is already reachable; its nine-layer ingest and source "
            "restrictions are unchanged."),
        *inherited_ruc_warnings,
    ]
    templates[mynn_ruc_id] = mynn_ruc

    mynn_noahmp_id = (
        "wsm6-mynn-mynn-noahmp-no-radiation-expert-only-v1")
    mynn_noahmp = copy.deepcopy(template)
    mynn_noahmp["components"]["pbl"] = "mynn"
    mynn_noahmp["components"]["surface_layer"] = "mynn"
    mynn_noahmp["label"] = (
        "WSM6 + MYNN PBL + MYNN surface layer + Noah-MP + Dudhia SW "
        "(expert only)")
    mynn_noahmp["warnings"] = [
        mynn_noah["warnings"][0],
        (
            "EXPERT ONLY. WRF owns the write-back sequence explicitly: MYNN "
            "surface first, Noah-MP flux/state write-back second, and the "
            "Noah-MP category/fraction 2-m diagnostic post-pass last. The "
            "existing Noah-MP glacier, sea-ice and validation-status warnings "
            "still apply."),
        *mynn_noahmp["warnings"],
    ]
    templates[mynn_noahmp_id] = mynn_noahmp

    # Pairing reachability follows each LSM's established source discipline:
    # neither land-surface model becomes a broad component override.
    for route in registry["runner_routes"].values():
        for declared in route.get("source_template_ids", {}).values():
            if (
                "wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1"
                in declared
                and mynn_ruc_id not in declared
            ):
                declared.append(mynn_ruc_id)
        for declared in route.get("expert_template_ids", {}).values():
            if (
                "wsm6-ysu-mm5-noahmp-no-radiation-expert-only-v1"
                in declared
                and mynn_noahmp_id not in declared
            ):
                declared.append(mynn_noahmp_id)


def _unimplemented_specs(known: set[str]) -> dict[str, dict]:
    """Knobs the survey found in WRF that no GPUWM component honors."""
    lanes = []
    with JOURNAL.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("type") == "result":
                lanes.append(record["result"])

    unimplemented: dict[str, dict] = {}
    for lane in lanes:
        for knob in lane["knobs"]:
            name = knob["name"]
            if name in known:
                continue
            if knob.get("gpuwm_implemented"):
                continue
            reason = clean(knob.get("not_implemented_reason", ""))
            if not knob.get("wrf_namelist_knob"):
                reason = ("Not a WRF v4.6.1 namelist option. "
                          + reason).strip()
            if not reason:
                reason = "Not implemented by any GPUWM runtime component."
            spec = {
                "type": (knob.get("proposed_spec") or {}).get("type")
                or WRF_TYPE.get((knob.get("wrf_type") or "").lower(),
                                "integer"),
                "implemented": False,
                "unimplemented_reason": reason}
            prior = unimplemented.get(name)
            if (prior is None
                    or len(prior["unimplemented_reason"]) < len(reason)):
                unimplemented[name] = spec
    return unimplemented


def _wrf_compatibility_authority() -> dict:
    """JSON form of the normalized, fully cited WRF compatibility matrix."""

    def citation(value) -> dict[str, str]:
        return {
            "source": value.anchor,
            "law": value.law,
        }

    def cell(**updates):
        values = {
            "mp_physics": 1,
            "bl_pbl_physics": 0,
            "sf_sfclay_physics": 1,
            "sf_surface_physics": 2,
            "radiation": "off",
            "cu_physics": 0,
        }
        values.update(updates)
        return compatibility_cell(**values)

    counts: dict[str, int] = {}
    for matrix_cell in iter_compatibility_matrix():
        verdict = matrix_cell.verdict.value
        counts[verdict] = counts.get(verdict, 0) + 1

    return {
        "wrf_version": WRF_VERSION,
        "wrf_commit": WRF_COMMIT,
        "implementation": "gpuwm.wrf461_compatibility",
        "cell_count": MATRIX_CELL_COUNT,
        "dimensions": {
            "mp_physics": list(MP_OPTIONS),
            "bl_pbl_physics": list(PBL_OPTIONS),
            "sf_sfclay_physics": list(SURFACE_LAYER_OPTIONS),
            "sf_surface_physics": list(LAND_SURFACE_OPTIONS),
            "radiation": list(RADIATION_OPTIONS),
            "cu_physics": list(CUMULUS_OPTIONS),
        },
        "independent_axis_citations": {
            "mp_physics": {
                str(value): citation(cell(mp_physics=value).citations[0])
                for value in MP_OPTIONS
            },
            "sf_surface_physics": {
                str(value): citation(
                    cell(sf_surface_physics=value).citations[2])
                for value in LAND_SURFACE_OPTIONS
            },
            "radiation": {
                value: citation(cell(radiation=value).citations[3])
                for value in RADIATION_OPTIONS
            },
            "cu_physics": {
                str(value): citation(cell(cu_physics=value).citations[4])
                for value in CUMULUS_OPTIONS
            },
            "soil_layer_reconfiguration": citation(
                cell().citations[5]),
        },
        "pbl_surface_layer_cells": [
            {
                "bl_pbl_physics": pbl,
                "sf_sfclay_physics": surface,
                "verdict": verdict.value,
                "citation": citation(source),
            }
            for (pbl, surface), (verdict, source)
            in sorted(PBL_SURFACE_LAYER_AUTHORITY.items())
        ],
        "precedence": [
            "a fatal PBL/surface-layer cell is fatal for every other axis",
            "otherwise analytic radiation is not expressible in WRF v4.6.1",
            "otherwise sf_surface_physics=0 is legal with WRF silently "
            "setting num_soil_layers=5",
            "all remaining cells are legal",
        ],
        "verdict_counts": dict(sorted(counts.items())),
        "test": (
            "tests/test_wrf461_compatibility.py sweeps all 2,400 cells and "
            "requires every cell to carry all six WRF citations"),
    }


def build(registry: dict) -> dict:
    """Apply this pass's tables to ``registry`` in place and return it."""
    _surface_coupling_warnings(registry)
    registry["authority"][
        "wrf_v461_compatibility_matrix"
    ] = _wrf_compatibility_authority()
    registry["authority"]["reachability_declaration"] = (
        "components.<component>.options.<option>.reachability declares how "
        "a user can select the option: 'template' through a registered base "
        "template, 'component-override' through either a route's full "
        "allowed_component_overrides or its option-scoped "
        "allowed_component_options, 'expert-template' only through a route's "
        "expert_template_ids with its expert_acknowledgement_id, and "
        "'unreachable' not normally reachable -- which must name a blocker. "
        "implemented and reachable are independent. "
        "tests/test_registry_reachability.py recomputes every state.")

    # Named native-HRRR Kessler product used by the end-to-end ratification
    # probe.  Its source route is intentionally HRRR-only: no other source
    # inherits evidence from that run.
    kessler_id = "kessler-mp1-ysu-mm5-noah-dudhia-v1"
    wsm6_id = "wsm6-ysu-mm5-noah-no-radiation-v1"
    kessler = copy.deepcopy(registry["templates"][wsm6_id])
    kessler["components"]["microphysics"] = "kessler-mp1"
    kessler["label"] = (
        "Kessler warm rain + YSU + classic MM5 + Noah + Dudhia SW")
    kessler["maturity"] = "implemented-unverified"
    kessler["warnings"] = [
        "Native-HRRR Kessler admission is bound to the Lane C one-hour "
        "end-to-end probe and its frozen-species discard receipt; it is not "
        "evidence for any non-HRRR source route."
    ]
    registry["templates"][kessler_id] = kessler
    registry["components"]["microphysics"]["options"][
        "kessler-mp1"]["reachability"] = {"state": "template"}
    for route_id in (
            "tools.hrrr_single_domain_benchmark",
            "tools.prepared_domain_tree_forecast"):
        declared = registry["runner_routes"][route_id].setdefault(
            "source_template_ids", {}).setdefault("hrrr", [])
        if kessler_id in declared:
            declared.remove(kessler_id)
        position = declared.index(wsm6_id) + 1 if wsm6_id in declared else 0
        declared.insert(position, kessler_id)

    # WRF v4.6.1's actual PBL/surface-layer law is in
    # phys/module_physics_init.F:3699-3701,3837-3839.  In particular MYNN
    # PBL accepts the revised and classic MM5 surface layers, and the MYNN
    # surface layer is legal with PBL off.  These declarative constraints
    # mirror the same 12-cell table used by runtime admission.
    pbl_options = registry["components"]["pbl"]["options"]
    surface_options = registry["components"]["surface_layer"]["options"]
    pbl_options["ysu"]["constraints"]["requires_components"][
        "surface_layer"
    ] = ["revised-mm5", "classic-mm5"]
    pbl_options["mynn"]["constraints"]["requires_components"][
        "surface_layer"
    ] = ["revised-mm5", "classic-mm5", "mynn"]
    surface_options["mynn"]["constraints"]["requires_components"][
        "pbl"
    ] = ["off", "mynn"]
    surface_options["mynn"]["warnings"] = [
        warning for warning in surface_options["mynn"]["warnings"]
        if not warning.startswith("MYNN is admitted only as the coupled")
        and not warning.startswith(
            "WRF v4.6.1 admits this surface layer with PBL off")
    ]
    surface_options["mynn"]["warnings"].insert(
        0,
        "WRF v4.6.1 admits this surface layer with PBL off or MYNN PBL. "
        "MYNN PBL also admits revised/classic MM5 surface layers; the exact "
        "12-cell authority is phys/module_physics_init.F:3699-3701,"
        "3837-3839 and is published under "
        "authority.wrf_v461_compatibility_matrix.",
    )

    # PBL-off with km_opt=4 follows WRF's diff_opt=2 vertical_diffusion_2
    # path.  The prior registry rail described an absent vertical operator;
    # that operator is now ported, including USTM/HFX/QFX surface fluxes.
    pbl_off = pbl_options["off"]
    pbl_off.setdefault("constraints", {}).setdefault(
        "forbidden_setting_values", {}).pop("km_opt", None)
    if not pbl_off["constraints"]["forbidden_setting_values"]:
        pbl_off["constraints"].pop("forbidden_setting_values")
    if not pbl_off["constraints"]:
        pbl_off.pop("constraints")
    pbl_off.setdefault("extensions", {})["wrf_vertical_diffusion_2"] = {
        "activation": "bl_pbl_physics=0, diff_opt=2, km_opt=4",
        "wrf_call_site": "dyn_em/module_first_rk_step_part2.F:1008-1074",
        "wrf_coefficient_policy": (
            "dyn_em/module_diffusion_em.F:2018-2023 sets xkmv=xkmh and "
            "xkhv=0 for km_opt=4"),
        "ported_operators": [
            "tau13 u momentum", "tau23 v momentum", "tau33 w momentum",
            "USTM lower-boundary momentum stress",
            "HFX lower-boundary heat flux", "QFX lower-boundary vapor flux",
        ],
    }
    pbl_off["reachability"] = {"state": "component-override"}

    # A source-driven active LSM still needs a surface-layer writer in ArWen.
    # This is not a WRF prohibition: it is the named local structural seam
    # that keeps the sfclay=0/LSM>0 cells fail-closed.
    land_options = registry["components"]["land_surface"]["options"]
    for option_id in ("noah", "ruc-lsm", "noah-mp"):
        option = land_options[option_id]
        option.setdefault("constraints", {}).setdefault(
            "requires_components", {})["surface_layer"] = [
                "revised-mm5", "classic-mm5", "mynn"]
        option.setdefault("extensions", {})[
            "arwen_surface_exchange_structural_requirement"
        ] = {
            "reason": (
                "ArWen's active LSM drivers consume UST/CHS/CHS2/CQS2/"
                "FLHC/FLQC exchange fields that have no writer when "
                "sf_sfclay_physics=0"),
            "classification": (
                "ArWen structural constraint; WRF v4.6.1 does not prohibit "
                "sf_sfclay_physics=0 with an active LSM"),
        }

    # The experiment-per-domain front door now exposes harmless WRF-legal
    # PBL-off and radiation-off choices, plus every WRF-legal implemented
    # surface-layer pairing.  This is option-scoped so ArWen's analytic
    # radiation proxy remains outside normal registry reachability.
    tree_route = registry["runner_routes"][
        "tools.prepared_domain_tree_forecast"]
    tree_route["allowed_component_options"] = {
        "pbl": ["off", "ysu", "mynn"],
        "surface_layer": ["revised-mm5", "classic-mm5", "mynn"],
        "radiation": [
            "off", "dudhia-shortwave", "rte-rrtmgp",
            "rte-rrtmgp-legacy-aggregate"],
    }
    surface_options["revised-mm5"]["reachability"] = {
        "state": "component-override"}
    radiation_options = registry["components"]["radiation"]["options"]
    radiation_options["off"]["reachability"] = {
        "state": "component-override"}
    analytic = radiation_options["analytic-clear-sky"]
    analytic["reachability"] = {
        "state": "unreachable",
        "blocker": (
            "The 90/90 analytic clear-sky proxy is ArWen-specific: WRF "
            "v4.6.1 registers no equivalent package "
            "(Registry/Registry.EM_COMMON:3107-3125). It remains available "
            "to unnamed domain trees only through the authority-level "
            "expert-tuple-v1 acknowledgement and is intentionally excluded "
            "from allowed_component_options."),
    }
    analytic["warnings"] = [
        "ARWEN-SPECIFIC, NOT A WRF SCHEME. Analytic 90/90 radiation remains "
        "an expert acknowledgement path because WRF v4.6.1 has no equivalent "
        "Registry package."
    ]

    params = registry["parameters"]
    # Citations are read before the tables overwrite the specs that carry
    # them, so a verified citation survives a spec being tightened.
    prior_citations = {
        name: spec.get("consuming_read")
        for name, spec in params.items() if isinstance(spec, dict)}

    selectors: set[str] = set()
    for component in registry["components"].values():
        selectors |= set(component.get("selector_keys", []))

    known = set(IMPLEMENTED) | set(params) | selectors
    unimplemented = _unimplemented_specs(known)

    # Copied, so the citation pass below cannot write a citation back into
    # this module's tables and leak it into a second build.
    for table in (IMPLEMENTED, TIGHTEN, unimplemented):
        for name, spec in table.items():
            params[name] = copy.deepcopy(spec)

    for name, (_, reason) in UNIMPLEMENTED_LEDGER.items():
        prior = params.get(name)
        if not isinstance(prior, dict) or "type" not in prior:
            raise KeyError(
                f"Lane K ledger row {name!r} has no typed registry parameter")
        params[name] = {
            "type": prior["type"],
            "implemented": False,
            "unimplemented_reason": reason,
        }
    remaining_false = {
        name for name, spec in params.items()
        if isinstance(spec, dict) and spec.get("implemented") is False
    }
    if remaining_false != set(UNIMPLEMENTED_LEDGER):
        raise AssertionError(
            "Lane K ledger no longer covers exactly every implemented=false "
            f"parameter; missing={sorted(remaining_false - set(UNIMPLEMENTED_LEDGER))}, "
            f"retired={sorted(set(UNIMPLEMENTED_LEDGER) - remaining_false)}")

    uncited, replaced = [], []
    for name in OWNED:
        spec = params.get(name)
        if not isinstance(spec, dict):
            continue
        before = prior_citations.get(name)
        citation = find_consuming_read(name, before)
        if citation is None:
            uncited.append(name)
            spec.pop("consuming_read", None)
            continue
        if before and citation != before:
            replaced.append(f"{name}: {before} -> {citation}")
        spec["consuming_read"] = citation
    if uncited:
        print("NO CONSUMING READ FOUND (claim withdrawn):", uncited)
    if replaced:
        print("CITATION NO LONGER RESOLVES (repointed):")
        for line in replaced:
            print("  " + line)

    for template_id, columns in NEST_COLUMNS.items():
        registry["templates"][template_id]["per_domain_overrides"] = [
            dict(column) for column in columns]

    registry["authority"]["parameter_declaration"] = (
        "parameters declare every knob a GPUWM runtime component reads; "
        "implemented=false publishes a knob GPUWM does not yet honor so the "
        "registry doubles as the porting roadmap, and such a knob can never "
        "be set. Component selector_keys are declared on their component and "
        "are deliberately absent from parameters.")
    registry["authority"]["per_domain_override_semantics"] = (
        "templates.per_domain_overrides is indexed by depth below the tree "
        "root and carries values transcribed from verified runs; nominal_dx_m "
        "is provenance for display and is never a resolved setting.")

    # WRF v4.6.1 module_pbl_driver.F:873-878 derives FLAG_QS from Registry
    # F_QS.  module_bl_mynn_wrapper.F:452-475 converts the real snow field to
    # specific units, and module_bl_mynn.F:1104-1106 supplies it to
    # mym_condensation.  Radiation then consumes the previous interval's
    # carried MYNN clouds at module_radiation_driver.F:1403-1429.
    mynn = registry["components"]["pbl"]["options"]["mynn"]
    mynn["extensions"]["supplied_moisture_species"] = {
        "supplied": ["qv", "qc", "qi", "qs"],
        "withheld": ["qnc", "qni", "qnwfa", "qnifa", "qnbca", "o3"],
        "flag_qs_true_microphysics_selectors": [6, 8, 10, 18],
        "flag_qs_false_microphysics_selectors": [0, 1],
        "wrf_flag_source": (
            "phys/module_pbl_driver.F:873-878 derives flag_qs from F_QS; "
            "Registry.EM_COMMON declares qs for mp_physics 6, 8, 10 and 18"),
        "wrf_live_consumer": (
            "phys/module_bl_mynn.F:1104-1106 passes real sqs to "
            "mym_condensation when FLAG_QS is true; mynn_tendencies still "
            "receives kzero at :1240-1242, matching WRF"),
    }
    mynn["extensions"]["radiation_cloud_merge"] = {
        "activation": "bl_pbl_physics=5 and icloud_bl>0",
        "ordering": (
            "radiation precedes PBL and consumes the previous interval's "
            "carried QC_BL/QI_BL/CLDFRA_BL"),
        "wrf_source": "phys/module_radiation_driver.F:1403-1429",
        "implementations": ["dudhia-shortwave", "rte-rrtmgp", "rrtmg-legacy"],
    }
    mynn["warnings"] = [
        warning for warning in mynn["warnings"]
        if not warning.startswith("DEVIATION from WRF, affecting MYNN PBL")
    ]
    template = registry["templates"][
        "wsm6-mynn-mynn-noah-no-radiation-implemented-unverified-v1"]
    template["warnings"] = [
        warning.replace(
            "the CUDA-versus-CPU ULP spread and the withheld snow species",
            "the CUDA-versus-CPU ULP spread; the WRF FLAG_QS snow path and "
            "previous-interval radiation cloud merge are coupled")
        for warning in template["warnings"]
    ]
    registry["authority"]["real_source_moisture_contract"] = (
        "runner_routes.<runner>.requires_moist_real_initialization declares "
        "that every source on the route enters gpuwm.ingest.real and therefore "
        "requires a moist state even when microphysics is off. The plan must "
        "then set moist=true explicitly; a component default cannot make that "
        "source-preparation decision for the user.")
    nssl2_id = (
        "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-"
        "validation-candidate-v1")
    nssl2_legacy_id = (
        "nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-"
        "validation-candidate-v1")
    nssl2_legacy = copy.deepcopy(registry["templates"][nssl2_id])
    nssl2_legacy["label"] = (
        "NSSL-2 + YSU + classic MM5 + Noah + KF + legacy RRTMG")
    nssl2_legacy["maturity"] = "validation-candidate"
    nssl2_legacy["parameters"]["wrf_rrtmg_compatibility"] = (
        "wrf-rrtmg-4-4-legacy-v1")
    nssl2_legacy["parameters"]["ra_rrtmg_variant"] = "rrtmg_legacy"
    nssl2_legacy["warnings"] = [
        "Ratified fixed NSSL-2 plus exact WRF v4.6.1 legacy RRTMG profile; "
        "the NSSL-2 trajectory remains validation-candidate maturity."
    ]
    registry["templates"][nssl2_legacy_id] = nssl2_legacy
    for route in registry["runner_routes"].values():
        for declared in route.get("source_template_ids", {}).values():
            if nssl2_id in declared:
                if nssl2_legacy_id in declared:
                    declared.remove(nssl2_legacy_id)
                declared.insert(declared.index(nssl2_id) + 1, nssl2_legacy_id)

    # Owner-ratified declaration: the GFS runner has always advertised this
    # profile and retains the existing Noah-MP route acknowledgement.
    gfs_route = registry["runner_routes"][
        "tools.prepared_single_domain_forecast"]
    gfs_expert = gfs_route.setdefault(
        "expert_template_ids", {}).setdefault("gfs", [])
    noahmp_id = "wsm6-ysu-mm5-noahmp-no-radiation-expert-only-v1"
    if noahmp_id not in gfs_expert:
        gfs_expert.append(noahmp_id)

    registry["authority"][
        "unnamed_tree_outside_reachability_acknowledgement_id"
    ] = "expert-tuple-v1"
    registry["authority"]["unnamed_tree_reachability_contract"] = (
        "An unnamed domain tree resolves each domain's complete component "
        "tuple against the union of registry-declared experiment-per-domain "
        "template, whole-component override, option-scoped component "
        "override, and expert-template reachability. "
        "Normal tuples proceed unchanged; expert-template tuples require the "
        "route's expert acknowledgement; a tuple outside that union requires "
        "the authority-level acknowledgement id. This is a tuple capability "
        "check and never uses source identity.")
    registry["authority"]["v1_launch_behavior_changed"] = True
    for route_id in (
            "tools.hrrr_single_domain_benchmark",
            "tools.prepared_domain_tree_forecast",
            "tools.prepared_single_domain_forecast"):
        registry["runner_routes"][route_id][
            "requires_moist_real_initialization"] = True
    mp_options = (
        (1, "kessler-mp1"),
        (6, "wsm6-mp6"),
        (8, "thompson-mp8"),
        (10, "morrison-mp10"),
        (18, "nssl2-mp18"),
    )
    cross_options = []
    for parent_mp, parent_option in mp_options:
        for child_mp, child_option in mp_options:
            if parent_mp == child_mp:
                continue
            ratified = (parent_mp, child_mp) == (8, 18)
            rule = {
                "parent_option_id": parent_option,
                "child_option_id": child_option,
                "required_parent_settings": {
                    "moist": True,
                    "moist_cq": True,
                },
                "required_child_settings": {
                    "moist": True,
                    "moist_cq": True,
                    "nest_microphysics_transition": (
                        "mp8-to-mp18-mass-diagnosed-v1"
                        if ratified else "mp-edge-mass-diagnosed-v1"
                    ),
                },
                "status": "ratified" if ratified else "experimental",
            }
            if not ratified:
                rule["maturity"] = "experimental-runtime"
            cross_options.append(rule)
    registry["transitions"]["microphysics-one-way-v1"] = {
        "component_id": "microphysics",
        "cross_options": cross_options,
        "same_option": {
            "allowed": True,
            "required_child_settings": {
                "nest_microphysics_transition": "same-scheme-only",
            },
        },
        "topology_id": "one-way-nested-v1",
    }
    registry["parameters"]["nest_microphysics_transition"]["enum"] = [
        "same-scheme-only",
        "mp8-to-mp18-mass-diagnosed-v1",
        "mp-edge-mass-diagnosed-v1",
    ]
    return registry


def render(registry: dict) -> bytes:
    """The exact bytes the tracked registry file must contain."""
    return (canonical_json(registry) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--registry", type=pathlib.Path, default=REGISTRY_PATH,
        help="the registry to transform; defaults to the tracked file. The "
             "tables below own only part of it, so the rest is carried "
             "through from here unchanged")
    parser.add_argument(
        "--out", type=pathlib.Path, default=REGISTRY_PATH,
        help="where to write the registry; defaults to the tracked file, and "
             "a test points it at a temporary path to compare bytes without "
             "touching the tree")
    args = parser.parse_args(argv)

    registry = build(json.loads(args.registry.read_text(encoding="utf-8")))
    params = registry["parameters"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(render(registry))
    print("parameters:", len(params),
          "| implemented:", sum(1 for s in params.values()
                                if s.get("implemented") is not False),
          "| unimplemented:", sum(1 for s in params.values()
                                  if s.get("implemented") is False),
          "|", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
