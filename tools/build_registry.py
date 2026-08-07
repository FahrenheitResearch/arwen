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
    # km_opt is NOT here any more.  It used to be a parameter with
    # ``enum: [1, 4]``, which was wrong twice over: 2 (1.5-order
    # prognostic TKE) and 3 (3-D Smagorinsky) are executable -- the
    # dycore's fail-closed admission takes 1/2/3/4 (gpuwm/config.py:1345,
    # decided at gpuwm/core/dycore.py:2226), the kernels are transcribed
    # from WRF v4.6.1 with dry-CBL oracle receipts, and
    # configs/les_nest_250m_km3.toml has shipped km_opt=3 on a nest child
    # since 1.5.1 -- and the enum said neither could be named.  Widening
    # it in place would have put km_opt on both channels at once, which
    # this document's own authority note forbids ("Component
    # selector_keys are declared on their component and are deliberately
    # absent from parameters") and tests/test_physics_registry_
    # declarations.py::test_selector_keys_are_not_duplicated_as_parameters
    # enforces.  So the selector moved to where every other scheme
    # selector already lives: components.turbulence.selector_keys, whose
    # four option rows carry km_opt 1/2/3/4 with their own maturity,
    # reachability and evidence.  Per the les-completion spec's
    # registry-honesty item (8.1.5) and AC-P6.4.
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
    # WRF's own &dynamics switch (Registry.EM_COMMON:2889, max_domains,
    # default .false.), consumed by gpuwm.core.dycore.diff6_exempt_slots.
    # Declared here because it is divergence-ledger entry L4: an ArWen
    # DEFAULT candidate whose promotion is decided by the observation
    # battery, not by argument.  The warning is what a plan author needs to
    # know before setting it by hand instead of through the axis.
    "moist_mix6_off": {
        "type": "boolean", "default": False,
        "warnings": [
            "moist_mix6_off = true removes the 6th-order horizontal filter "
            "from the WRF moist array only (dyn_em/module_em.F:1421 under "
            "config_flags%moist_mix6_off); theta keeps its filter and the "
            "number/volume tracers keep theirs, which is WRF's own per-array "
            "scoping, not a gpuwm simplification.",
            "This is divergence-ledger entry L4 and it is UNDECIDED: it is "
            "a candidate ArWen default with no obs-skill receipt yet. "
            "Selecting it through the [experiment] physics_mode axis "
            "(gpuwm/physics_mode.py) records the arm in the run receipt; "
            "setting it by hand here does not, and a hand-set value beside "
            "physics_mode is refused rather than merged."]},
    "top_lid": {"type": "boolean", "default": True},
    "morr_rimed_ice": {"type": "integer", "enum": [0, 1], "default": 1},
    "wsm6_hail_opt": {"type": "integer", "enum": [0, 1], "default": 0},
    "icloud": {"type": "integer", "enum": [0, 1], "default": 1},
    "radt": {"type": "number", "minimum": 0.0, "default": 0.0},
    "radt_minutes": {"type": "number", "minimum": 0.0, "default": 12.0},
    "bldt": {"type": "number", "minimum": 0.0, "default": 0.0},
    "cudt_minutes": {"type": "number", "minimum": 0.0, "default": 5.0},
    # Grell-family keys, WRF v4.6.1 Registry defaults
    # (Registry.EM_COMMON:2544,2546); read only where cu_physics = 3.
    "clos_choice": {
        "type": "integer", "enum": [0], "default": 0,
        "warnings": [
            "Only the 16-member ensemble closure (0, the Registry default) "
            "is admitted: the single-closure arms of cup_forcing_ens_3d "
            "carry no GF oracle coverage."]},
    "ishallow": {"type": "integer", "enum": [0, 1], "default": 0},
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
        "cumulus scheme; none of gpuwm's cumulus options (off, the ported "
        "KF, the ported GF) carries AERCU tendency state."),
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
    "dust_emis": (
        "c",
        "Dust emission outside WRF-Chem feeds the ice-friendly aerosol "
        "surface source. gpuwm allocates nifa2d and leaves it exactly zero, "
        "matching thompson_init, and carries no dust inventory, emission "
        "operator, or surface-flux coupling for it."),
    "grav_settling": (
        "c",
        "Gravitational settling of fog droplets is a PBL-side operator gpuwm "
        "has not ported. WRF SILENTLY forces this to 0 on every mp_physics=28 "
        "domain, at debug verbosity "
        "(share/module_check_a_mundo.F:2459-2474); gpuwm's posture is to "
        "refuse where WRF overwrites, so a nonzero value is an error."),
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
        "NAER seeds WRF's naer-based droplet mode inside classic Thompson "
        "(thompson-mp8), which is a fixed-droplet port. The prognostic "
        "aerosol path lives in components.microphysics.options."
        "thompson-aerosol-mp28, whose nc/nwfa/nifa come from CCN activation "
        "rather than from this scalar, so honouring it here would be a "
        "different scheme."),
    "num_wif_levels": (
        "c",
        "This sizes the WIF (water/ice-friendly aerosol) metgrid input "
        "stream. components.microphysics.options.thompson-aerosol-mp28 runs "
        "thompson_init's synthetic CCN/IN profile because gpuwm has no WIF "
        "ingest, so the count would size an array nothing reads."),
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
        "PROGN switches classic Thompson (thompson-mp8) onto WRF's "
        "chemistry-driven droplet source, which needs a QNDROPSOURCE carrier "
        "gpuwm has no writer for. Prognostic droplet number itself IS ported, "
        "as components.microphysics.options.thompson-aerosol-mp28 "
        "(mp_physics=28), where it is driven by CCN activation instead."),
    "qna_update": (
        "c",
        "Updating aerosol number from a wrfqnainp auxiliary stream needs an "
        "auxiliary input subsystem, an update cadence, restart position and "
        "field ownership; gpuwm writes one fixed wrfout frame per domain and "
        "reads no auxiliary input streams at all."),
    "scalar_pblmix": (
        "c",
        "This mixes WRF's 4-D scalar array -- qnc/qnwfa/qnifa -- with the PBL "
        "scheme (phys/module_pbl_driver.F:2251). gpuwm supplies no scalar "
        "array to any PBL component and withholds those species from MYNN, so "
        "the selector has nothing to act on. See the thompson-aerosol-mp28 "
        "warnings for the resulting divergence."),
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
        "GOCART climatological 3-D aerosol IC/BC for "
        "components.microphysics.options.thompson-aerosol-mp28. The scheme is "
        "ported; this ingest is not -- gpuwm has no GOCART reader and no "
        "aerosol lateral-boundary carrier, so mp_physics=28 runs on "
        "thompson_init's synthetic profile only."),
    "use_rap_aero_icbc": (
        "c",
        "The RAP-sourced variant of the GOCART climatological aerosol IC/BC "
        "above, and blocked on the same absent reader and lateral-boundary "
        "carrier; components.microphysics.options.thompson-aerosol-mp28 runs "
        "thompson_init's synthetic profile only."),
    "wif_fire_emit": (
        "c",
        "Biomass-burning aerosol emissions for Thompson-MP-Aero need a fire "
        "inventory, an emission cadence and the derived aer_fire_emit_opt "
        "state WRF computes from this flag; gpuwm carries none of them."),
    "wif_fire_inj": (
        "c",
        "This selects the vertical injection profile for the biomass-burning "
        "aerosol emissions above. It is a branch inside a subsystem gpuwm "
        "does not have, so there is nothing for it to distribute."),
    "wif_input_opt": (
        "c",
        "This selects WRF's WIF metgrid aerosol input stream. gpuwm has no "
        "WIF ingest and no QNWFA/QNIFA initial/boundary carrier. NOTE the "
        "asymmetry recorded on thompson-aerosol-mp28: WRF's real.exe FATALs "
        "mp_physics=28 with wif_input_opt=0 "
        "(dyn_em/module_initialize_real.F:2734-2736) while gpuwm runs it on "
        "thompson_init's synthetic profile."),
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


# ------------------------------------------ aerosol-aware Thompson (mp=28)
#: G3 -- the whole aerosol column deck driven end to end through the shipped
#: adapter -- measured on this tree, one RTX 5090, FP32, gate 2.0e-6 relative
#: on every one of the 16 compared fields (15 column + rainnc).
#:
#: THE DECK IS TWENTY-TWO COLUMNS, NOT NINETEEN, and the registry now says so.
#: ``_FIXTURES`` in the gate is a glob over
#: ``gpuwm/data/thompson/oracle-aero/*-column.csv``: the nineteen ``aero-*``
#: scenarios MP28_PORT_SPEC.md specifies (ids 101-119) PLUS three ``wp08-*``
#: columns (ids 120-122) that the same ``build_aero.sh`` invocation produced
#: in the same format and that pin every reachable ``nu_c`` and both branches
#: of the terminal phase cleanup.  Waves 1-4 published "nineteen" while the
#: gate drove twenty-two, which understated the deck AND hid two residuals
#: (``wp08-freeze``, ``wp08-nusweep``) in no published class at all.
#:
#: These are RE-MEASURED, not transcribed: ``tests/test_physics_registry.py::
#: test_mp28_published_residuals_still_equal_a_live_adapter_measurement``
#: reruns every fixture through the shipped adapter on the device and
#: rebuilds this partition.  Regenerate the numbers, never round them.
#:
#: WHAT MOVED SINCE THE LAST PUBLISHED SET, all of it re-measured on
#: 2026-08-01 through the shipped adapter.  TWO production changes did it,
#: both in mp=28-owned kernels, and the attribution below is the gate's own
#: (tests/test_thompson_aerosol_adapter.py::_G3_RESIDUALS records the
#: per-change deltas) rather than a summary invented here:
#:
#:   WP-13a, THE SEDIMENTATION DENSITY.  WRF builds the working rain mass and
#:   number sedimentation consumes in two places: at
#:   module_mp_thompson.F:3237-3238 from the :3193 TAU+1 density, for every
#:   level with L_qr, and again at :3568/:3570 from the :3490 POST-
#:   condensation density -- but only inside the :3501-3502 gate
#:   (``ssatw < -eps .and. L_qr .and. .not. prw_vcd > 0``).
#:   gpuwm/core/kernels/thompson_aerosol_sat.cu wrote the post-condensation
#:   density into its ``reference_density`` output unconditionally, so every
#:   level got the :3568 answer including the levels WRF never rewrote; it now
#:   defaults to the :3237 density and overwrites it level by level from its
#:   own transcription of those three gates.
#:
#:   WP-13b, CONTRACTION PINNING OF THE SOURCE-NETWORK APPLY.  WRF's
#:   :3973-4023 terminal apply is ``q1d(k) = q1d(k) + qten(k)*DT`` and the
#:   gfortran -O2 baseline-x86-64 oracle has no FMA instruction, so qten*DT is
#:   rounded to REAL(4) before the add; nvrtc contracted the same expression
#:   in thompson_aerosol_cold.cu and thompson_aerosol_warm.cu and never
#:   rounded it.  Both now round it, as thompson_aerosol_sat.cu's rain-
#:   evaporation apply already did.
#:
#: WHAT THAT DID TO THE PUBLISHED PARTITION:
#:   * ``aero-drop-evap`` LEFT the residual list entirely and is published
#:     CLEAN -- WP-13a alone: rainnc 5.165e-04 -> 0.000e+00 (BIT-EXACT),
#:     qr 3.533e-05 -> 7.346e-08, nr 2.258e-05 -> 3.919e-07.
#:   * ``aero-ice-demott-idxin`` LEFT it too, and needed BOTH changes: WP-13a
#:     took sr and rainnc/rainncv 1.279e-04 -> 3.5e-07 and qr 2.894e-05 ->
#:     1.35e-06, then WP-13b took rainnc/rainncv and sr to BITWISE 0.000e+00
#:     and qr to 6.416e-07.  Its nr now measures 3.243e-07.
#:   * ``aero-cloud-freeze-nc`` lost three of the four rows this table
#:     published -- qr 2.800e-05 -> 8.973e-08, nr 1.797e-05 -> 2.594e-07,
#:     rainnc 1.162e-05 -> 0.000e+00, all WP-13a -- and is published with
#:     ``qc`` 4.926e-06 alone.
#:   * ``aero-reduces-to-classic``'s carve-out lost its ``qr`` FIELD:
#:     7.813e-05 -> 1.788e-07, inside the FLAT 2.0e-6 gate, so the gate's
#:     ``_END_TO_END_BOUNDS`` now names ``nr_per_kg`` only and the bound on it
#:     went 1.0e-04 -> 1.0e-05 (ten times stricter) on a measurement of
#:     5.700e-06.  The registry published {qr, nr_per_kg} against a gate that
#:     applies {nr_per_kg}; that is corrected here.
#:   * ``_REFL_DB_BOUNDS`` was RETIRED, not widened: the reflectivity residual
#:     it covered went 5.283e-04 dB -> 3.242e-05 dB, inside the flat 2.0e-4 dB
#:     gate.  The published allowance list is therefore TWO entries where it
#:     was three.
#:   * ``aero-cold-overlap`` GOT WORSE ON ONE FIELD and is published that way:
#:     qr 3.667e-05 -> 4.443e-05 at 0-based level 6.  That growth is WP-13b's,
#:     bisected to a single line -- the cold network's qr apply -- and it was
#:     KEPT rather than reverted because reverting it also loses all four of
#:     aero-ice-demott-idxin's improvements above, two of which are bitwise.
#:     In ulps of the entry value, the scale this cell is recorded at because
#:     99.5% of the level's rain is consumed in the step, the move is 1.477 ->
#:     1.789 ulp.  The same change IMPROVED this fixture's nr, 1.340e-04 ->
#:     1.261e-04, and its level-4 rows are unchanged.  The wave-5 registry
#:     published 3.667e-05, which UNDERSTATED the port's own error by 21%;
#:     publishing the larger number is the point of re-measuring.
#:
#: WHAT MOVED IN THE WAVE BEFORE THAT, kept because the number it retires is
#: still quoted in the public documents:
#:   * ``aero-ice-koop`` LEFT the residual list.  Its published qi 1.612e-03 /
#:     ni 1.764e-03 / effi 5.093e-05 -- called "the largest genuine gap" by
#:     the registry, PHYSICS.md, PROVENANCE.md D9k and the evidence page --
#:     now measure 1.534e-07 / 3.396e-07 / 1.886e-07 and the fixture is
#:     CLEAN.  IT WAS NOT CLOSED BY A KERNEL.  It was closed by correcting
#:     the ORACLE HARNESS: tools/thompson_wrf461_oracle/run_column_aero.F90
#:     built the Exner function with rd_over_cp = 287.0/1004.0 where WRF's
#:     own rcp is r_d/cp = 287./(7.*287./2.) = 2/7
#:     (share/module_model_constants.F:19,:20,:31), 4774 float32 ulps away,
#:     so the recorded (p, theta) pair could not be inverted exactly on the
#:     gpuwm side and the adapter drove dozens of the deck's 528 entry
#:     levels (47 as re-pinned 2026-08-03; first published as 40, the
#:     original environment's libm) from a perturbed pressure.  The deck was regenerated with WRF's own
#:     constant, with no change to any .cu or .cuh file.  The registry was
#:     10,500x pessimistic about its own worst number, and about a residual
#:     that its own reference harness had manufactured.  See
#:     docs/public/validation/mp28-column-evidence.md section 3.4.
#:   * ``aero-cloud-freeze-nc`` lost its ``effc_m`` row (5.018e-06 ->
#:     1.619e-06, inside the gate) and its ``qc`` fell 1.478e-05 -> 4.926e-06.
#:   * ``aero-ice-demott-idxin`` lost its ``qc`` row (6.031e-06 -> 7.556e-08)
#:     and its ``qr`` fell 3.895e-05 -> 2.894e-05.
#:   * ``aero-cold-overlap`` got WORSE and is published that way: a new
#:     ``qc``/``nc_per_kg`` pair at 1.000e+00 and ``effc_m`` at 8.102e-01.
#:     See the note on that entry -- it is one mechanism at one level, not
#:     three failures, and it is a full-scale relative number on a ONE-ULP
#:     absolute difference.
#:   * ``wp08-freeze`` and ``wp08-nusweep`` were published for the first time.
#:
#: THE UNEXCEPTIONED CLEAN SET.  17 of 22: sixteen of the nineteen spec'd
#: ``aero-*`` fixtures plus ``wp08-melt``.  The 1.4.1 merge did NOT change
#: this set -- ``aero-reduces-to-classic`` still needs level 6 taken in ULPs
#: -- but it did retire the OTHER allowance that fixture rested on, so the
#: gated count of 18 now costs one allowance instead of two.  This tuple is asserted EQUAL to
#: the gate's own ``_G3_UNEXCEPTIONED_CLEAN`` by
#: ``tests/test_physics_md_aerosol_claims.py::
#: test_the_published_clean_counts_are_the_gates_own_counts``, so it cannot be
#: a transcription that drifts.
MP28_G3_CLEAN = (
    "aero-ccn-activate", "aero-ccn-sweep", "aero-drop-evap",
    "aero-ice-demott-dep", "aero-ice-demott-idxin", "aero-ice-koop",
    "aero-init-profile", "aero-nc-accrete", "aero-nc-auto", "aero-nc-cap",
    "aero-nc-effrad", "aero-nc-sed", "aero-scav-frozen", "aero-scav-rain",
    "aero-sfc-emit", "aero-warm-overlap", "wp08-melt",
)

#: Fixtures that do NOT clear 2e-6 on every field, with every field that
#: misses and its measured maximum relative difference.  FOUR of twenty-two.
#:
#: ``aero-cold-overlap``'s 1.000e+00 rows are the honest publication of a
#: sub-ulp disagreement and are recorded rather than allowanced.  MEASURED
#: at 0-based level 4: the level enters with qc = 2.3252160e-04 kg/kg and
#: nc = 9.1306704e+07 per kg; WRF ends the step with qc =
#: 1.4551915228366852e-11 kg/kg -- which is EXACTLY 2**-36, exactly 1.000
#: float32 ulp of the entry value -- and nc = 1.8333361 per kg, while gpuwm
#: ends at exactly 0.0.  module_mp_thompson.F:4007-4009 is
#: ``if (qc1d(k) .le. R1) then qc1d = 0.0 ; nc1d = 0.0`` with R1 = 1.E-12
#: (:183), so WRF takes the ELSE branch and its :4011-4020 nu_c/lamc/xDc size
#: bound hands back nc = 1.833336 per kg where gpuwm takes the THEN branch and
#: zeroes it.  A relative metric therefore reports 1.0 (qc, nc) on
#: an absolute difference of one ulp and 0.229 ulp respectively, and effc
#: reports 8.102e-01 because :5638 does the same thing again: a level with
#: rc <= R1 CYCLEs and keeps RE_QC_BG = 2.49 um
#: (share/module_model_constants.F:62, installed at :5619) while WRF's
#: 1.455e-11 kg/kg remainder gives 1.31176e-05 m.  One branch flip, three
#: views.  The fixture's OTHER residual is separate and real: nr 1.261e-04 /
#: qr 4.443e-05 at level 6, where the rain number falls 255.407 -> 0.0739 per
#: kg (99.97% consumed) and the difference is 0.611 ulp (nr) / 1.789 ulp (qr)
#: of the entry value.
#:
#: ``wp08-freeze`` nr and ``wp08-nusweep`` qr are both fields CREATED FROM
#: EXACTLY ZERO inside the step, at 1.4x and 2.3x the gate.  ONE of the two
#: has a traced mechanism and it is in a file mp=28 may not touch:
#: gpuwm/core/kernels/thompson.cu:438 gates rain sedimentation on
#: ``qr > 1.0e-12``, a MIXING RATIO, where module_mp_thompson.F:3616 tests
#: ``rr(k) > R1``, a MASS CONCENTRATION, so at wp08-freeze level 1 -- qr =
#: 8.5265e-13 kg/kg but rr = 1.1748e-12 kg/m3 -- WRF gives the level a real
#: fall speed and ArWen treats it as rain-free.  That kernel is the frozen,
#: model-validated mp=8 one, so the residual is recorded and filed as an
#: integration request rather than fixed here.
#:
#: THE TWO wp08 CELLS SWAPPED PLACES AT THE 1.4.1 MERGE.
#:
#: ``wp08-nusweep`` qr level 12 -- the cell this comment used to say NO
#: MECHANISM IS CLAIMED for -- is now explained, by CONDITIONING.  Perturbing
#: the level's entry state by exactly one float32 ulp and re-running the
#: adapter moves the exit qr by 128 ulp (entry qc +1), 32 ulp (entry qc -1)
#: or 256 ulp (entry nc, either direction).  The disagreement is 60 ulp,
#: smaller than a single-ulp input change produces, and the 2.0e-06 gate is
#: about 26 ulp there -- below the cell's own condition number.  No FP32
#: implementation can hold it, and |got - want| is 1.04e-16 kg/kg, still the
#: smallest absolute disagreement in this table by nine decades.
#:
#: ``wp08-freeze`` nr level 0 is EXPLAINED and the explanation is measured,
#: not argued.  29 of its 34 ulp are the frozen mp=8 kernel gating rain
#: presence on a mixing ratio (thompson.cu:438, qr > 1.0e-12f) where WRF
#: gates on a mass concentration (:3616, rr .gt. R1).  Forcing ArWen's gate
#: open at the one level where the two disagree moves level 0's nr from 34
#: ulp away from WRF to 5, and a same-sized mass change that does not flip
#: the gate leaves the output bit-identical.
#:
#: It was published as falsified for part of 2026-08-01 and that was wrong:
#: the falsification read qr1d + qrten*DT at the END of the step and took it
#: for the value :3236 tested, while :3501's evaporation block subtracts
#: from qrten in between.  Instrumented WRF records L_qr = .true. there.
#: cb765336 did not move the residual because it reconciled the
#: sedimentation DENSITY, not the gate's UNITS.  NOT FIXED: thompson.cu is
#: byte-frozen and mp=8 shares it, so the correction owes 92 classic
#: fixtures a re-measurement and is not mp=28's to make.
MP28_G3_RESIDUALS: dict[str, dict[str, float]] = {
    "aero-cloud-freeze-nc": {"qc": 4.926e-06},
    "aero-cold-overlap": {
        "qc": 1.000e+00, "nc_per_kg": 1.000e+00, "effc_m": 8.102e-01,
        "qr": 4.443e-05, "nr_per_kg": 1.261e-04},
    "wp08-freeze": {"nr_per_kg": 2.724e-06},
    "wp08-nusweep": {"qr": 4.642e-06},
}

#: The one fixture that clears the gate only through a carved-out bound, and
#: the ONE FIELD that bound still covers.
#:
#: THE FIELD SET IS PART OF THE PUBLICATION, not decoration.  ``carved_out_
#: bound`` is what a reader is told the port bought itself, so publishing
#: {qr, nr_per_kg} against a gate whose ``_END_TO_END_BOUNDS`` names
#: {nr_per_kg} overstates the relaxation by a whole quantity.  It is bound to
#: the gate's own dict by ``tests/test_physics_registry.py::
#: test_mp28_evidence_matches_the_bound_the_adapter_gate_actually_applies``,
#: which compares the SETS and not only the values.
#:
#: THE HISTORY, because two successive tightenings are easy to mistake for a
#: widening.  The bound was 2.5e-03 on {qr, nr_per_kg}, accommodating a
#: 1.915e-03 / 1.922e-03 residual at 0-based level 5.  WP-12a found the cause
#: -- module_mp_thompson.F:3490 overwrites rho(k) inside the condensation
#: loop and :3505-3520's orho/rhof/vsc2/rvs read THAT one, so prv_rev scales
#: with the PRE-condensation density and the adapter was not passing it --
#: and the bound went 2.5e-03 -> 1.0e-04, twenty-five times stricter, on a
#: re-measured 7.813e-05 / 4.832e-05.  WP-13a then restored WRF's level-wise
#: :3237-vs-:3568 sedimentation density, ``qr`` fell to 1.788e-07 and LEFT
#: the dict entirely (it clears the FLAT 2.0e-06 gate), and the surviving
#: ``nr_per_kg`` bound went 1.0e-04 -> 1.0e-05, ten times stricter again, on
#: a measurement of 5.700e-06.  The sequence is 2.5e-03 -> 1.0e-04 -> (qr
#: deleted, nr 1.0e-05).  Nothing was widened at any point.
#:
#: WHAT THE SURVIVING NUMBER IS.  0-based level 5 is the one level of this
#: column where the step removes a large fraction of the rain number without
#: emptying it (49.75%: 3.000000e+05 -> 1.507546e+05 per kg), and 5.700e-06
#: is 27.5 ulps of the entry value; every other unexcluded level is 0-3 ulps.
#: RE-MEASURED on 2026-08-01 through the shipped adapter: 5.7005e-06.
MP28_G3_CARVED_OUT: dict[str, dict[str, float]] = {
    # EMPTY.  RETIRED AT THE 1.4.1 MERGE, not narrowed again.
    #
    # It read {"aero-reduces-to-classic": {"nr_per_kg": 5.700e-06}} and its
    # gate-side bound was _END_TO_END_BOUNDS = 1.0e-5.  Merging
    # integration/release-1.4.1 inherited the mp=8 lane's two rain
    # sedimentation reconciliations -- 5e4af4e3 ("the rain MVD bound belongs
    # to TAU+1, not to sedimentation") and cb765336 ("the rain-presence gate
    # is a mass concentration, floor included") -- into the byte-frozen
    # thompson.cu mp=28 shares for rain fallout.  No mp=28 file changed.
    # RE-MEASURED through the shipped adapter on the merged tree, 0-based
    # level 5: nr_per_kg 4.146e-07, inside the FLAT 2.0e-06 gate by a factor
    # of 4.8.  The bound had nothing left to buy and was deleted.
    #
    # The full sequence, none of it a widening: 2.5e-03 on {qr, nr_per_kg}
    # -> 1.0e-04 -> (qr deleted, nr 1.0e-05) -> GONE.
    #
    # aero-reduces-to-classic is still NOT in MP28_G3_CLEAN: it still needs
    # _NEAR_CANCELLATION_LEVELS, which holds 0-based level 6 to 32 ULP of the
    # entry value rather than to a relative bound, and that is now the port's
    # only remaining departure from the flat gate anywhere in the deck.
}

MP28_OPTION_ID = "thompson-aerosol-mp28"


def _thompson_aerosol_mp28(registry: dict) -> None:
    """Register aerosol-aware Thompson at the maturity its evidence earns.

    Three declarations carry the whole claim and are worth stating together,
    because a reader who takes any one of them alone gets a wrong answer:

    ``implemented: true`` -- gpuwm has the component.  The scheme runs on the
    device through ``gpuwm/core/microphysics_aerosol.py`` and is dispatched by
    ``gpuwm/core/microphysics.py`` on ``mp_physics == 28``.

    ``maturity: implemented-unverified`` -- and no higher.  PHYSICS.md's
    published definition of that label is exactly this option's state: column
    -oracle-measured against unmodified WRF Fortran, with no forecast
    -trajectory comparison.  It may not claim ``validation-candidate`` (no
    ratified reference comparison exists) and certainly not
    ``model-validated`` (no matched multi-hour run, no decay tables).  What it
    also may not do is claim the column evidence is CLEAN, so the measured G3
    residuals are published on the option itself rather than left in a test
    file: four of twenty-two fixtures miss the 2e-6 gate and a fifth clears
    it only under a carved-out bound.

    ``reachability: component-override`` -- computed, not chosen.  The tree
    route already lists ``microphysics`` in ``allowed_component_overrides``,
    so an implemented microphysics option carrying no template is reachable
    exactly one way: as a per-domain experiment override.  Registering no
    template and leaving ``DEFAULT_TEMPLATE_ID`` alone is what keeps it out of
    every default suite; ``tests/test_registry_reachability.py`` recomputes
    the state from the routes and would fail on any other declaration.
    """

    options = registry["components"]["microphysics"]["options"]
    options[MP28_OPTION_ID] = {
        "asset_requirements": [
            {
                # ``kind`` stays ``operator-supplied-table-set`` even though
                # gpuwm now ships the file (2026-08-01): the requirement is
                # satisfied by a table an operator may point anywhere via
                # GPUWM_THOMPSON_CCN_ACTIVATE, and it is deliberately NOT a
                # member of the packaged CLASSIC table set that mp=8 resolves
                # through TABLE_SET_ID.  ``redistributed_by_gpuwm`` is the
                # field that says whether the wheel carries it.
                "id": "wrf-v4.6.1-aerosol-thompson-mp28-v1",
                "kind": "operator-supplied-table-set",
                "assets": [
                    {
                        "filename": "CCN_ACTIVATE.BIN",
                        "bytes": 35288,
                        "sha256": (
                            "f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75"
                            "301e422f147c82a3dbd"),
                    },
                ],
                "redistributed_by_gpuwm": True,
                "source": (
                    "WRF v4.6.1 (git tag v4.6.1, commit "
                    "d66e442fccc04111067e29274c9f9eaccc3cef28), file "
                    "run/CCN_ACTIVATE.BIN"),
                "search_root": "gpuwm/data/thompson/tables",
                "root_environment_override": "GPUWM_THOMPSON_TABLE_ROOT",
                "path_environment_override": "GPUWM_THOMPSON_CCN_ACTIVATE",
                "regenerable": False,
                "note": (
                    "table_ccnAct (phys/module_mp_thompson.F:5110-5166) READS "
                    "this file; it computes nothing. The numbers are offline "
                    "parcel-model output (Feingold & Heymsfield as modified "
                    "by Eidhammer and Kreidenweis, WRF's own comment at "
                    ":5102-5108), so no recompilation of WRF, no re-run of "
                    "thompson_init and no gpuwm code path regenerates it. "
                    "gpuwm redistributes WRF's file verbatim -- it is "
                    "committed under search_root, listed in that directory's "
                    "MANIFEST.sha256 and shipped in the wheel, under WRF's "
                    "public-domain dedication whose notice travels in "
                    "gpuwm/data/wrf_radiation/LICENSE-WRF.txt -- so a default "
                    "install satisfies this requirement and the environment "
                    "overrides exist to bind a run to another copy instead. "
                    "It "
                    "is not in thompson_contract.CLASSIC_TABLE_ASSETS and "
                    "TABLE_SET_ID is unchanged, so no mp_physics=8 launch "
                    "acquires a dependency on it. Size and SHA-256 are pinned "
                    "in gpuwm/core/thompson_aerosol_contract.py and checked "
                    "on every load; absence is fatal and never defaulted, and "
                    "a byte-different table is refused rather than used."),
            },
        ],
        "constraints": {"required_settings": {"moist": True}},
        "extensions": {
            "wrf_package": (
                "Registry/Registry.EM_COMMON:3036 binds mp_physics==28 to the "
                "thompsonaero package: moist qv,qc,qr,qi,qs,qg and scalar "
                "qni,qnr,qnc,qnwfa,qnifa,qnbca"),
            "prognostic_species": {
                "transported": ["qi", "qs", "qg", "nr", "ni", "nc",
                                "nwfa", "nifa"],
                "surface_emission_2d": ["nwfa2d", "nifa2d"],
                "not_ported": ["qnbca", "taod5502d", "taod5503d"],
                "note": (
                    "qnbca (black-carbon aerosol number) is out of scope for "
                    "v1 and refused rather than zero-filled; taod5502d/"
                    "taod5503d are radiation-side aerosol optical-depth "
                    "diagnostics that mp_gt_driver does not produce."),
            },
            "column_oracle_evidence": {
                "authority": (
                    "unmodified WRF v4.6.1 phys/module_mp_thompson.F at "
                    "commit d66e442fccc04111067e29274c9f9eaccc3cef28, "
                    "compiled by gfortran 13.3.0 -O2"),
                # 22 columns, not 19.  ``spec_fixtures`` is the count
                # MP28_PORT_SPEC.md names (the aero-* ids 101-119);
                # ``fixtures`` is what the gate actually drives, which is
                # that set plus three wp08-* columns (ids 120-122) from the
                # same build_aero.sh run.  Publishing only the smaller
                # number is how wp08-freeze and wp08-nusweep sat above the
                # gate in no published class for four waves.
                "fixtures": 22,
                "spec_fixtures": 19,
                "compared_fields": 16,
                # The 16 above are the contract shape the residual table is
                # published in (15 prognostic column fields + rainnc_mm).
                # The GATE compares 23 and asserts that width so it cannot be
                # narrowed back: the 15, six more surface diagnostics
                # (rainncv, snownc, snowncv, graupelnc, graupelncv, sr) plus
                # rainnc, and REFL_10CM against WRF's own calc_refl10cm at a
                # 2.0e-4 dB gate.
                "compared_quantities": 23,
                "compared_quantities_breakdown": {
                    "column_prognostic": 15,
                    "surface_accumulation": 7,
                    "reflectivity_db": 1,
                },
                "gate_relative": 2.0e-6,
                "gate_reflectivity_db": 2.0e-4,
                "clean_fixtures": list(MP28_G3_CLEAN),
                "residual_fixtures": {
                    name: dict(fields)
                    for name, fields in sorted(MP28_G3_RESIDUALS.items())},
                "carved_out_bound": {
                    name: dict(fields)
                    for name, fields in sorted(MP28_G3_CARVED_OUT.items())},
                # The gate's SECOND relaxation, published because an
                # unpublished one is indistinguishable from a hidden one.
                # It is not a widened tolerance: it is a different metric at
                # one level where the relative one is below float32
                # resolution, and it is bound to the gate's own constants by
                # tests/test_physics_registry.py::
                # test_mp28_publishes_the_near_cancellation_relaxation_too.
                "near_cancellation_bound": {
                    "fixtures": {"aero-reduces-to-classic": [6]},
                    "ulps_of_entry_value": 32.0,
                    "measured": {
                        "aero-reduces-to-classic": {
                            "qr_ulp": 0.585, "nr_per_kg_ulp": 0.159}},
                    "why": (
                        "aero-reduces-to-classic level 6 enters with qr = "
                        "3.1695777e-07 kg/kg and evaporates 99.958% of it in "
                        "one 10 s step, so the surviving value is the "
                        "difference of two nearly equal float32 numbers and "
                        "the relative error in the difference is the "
                        "relative error in the rate amplified by 1/(1 - "
                        "0.99958) = 2370 -- which puts a 2e-06 relative gate "
                        "below the float32 resolution of the entry value "
                        "itself. The bound is therefore stated in ULPS OF "
                        "THE ENTRY VALUE. This level used to be skipped "
                        "outright; it is bounded rather than skipped because "
                        "mp=28 now produces 1.3384e-10 there against WRF's "
                        "1.3426e-10, where it used to produce exactly 0."),
                },
                # EVERY DEPARTURE FROM THE FLAT GATE, NAMED.  ONE, on one
                # fixture, and it is needed for that one fixture.
                # Published here because an unpublished allowance is
                # indistinguishable from a hidden one, and because every one
                # that ever moved in this port moved STRICTER.
                #
                # THIS LIST WAS THREE, THEN TWO, AND IS NOW ONE.  The
                # 1.4.1 merge retired ``_END_TO_END_BOUNDS``: it carried
                # aero-reduces-to-classic's nr_per_kg at 2.5e-03, then
                # 1.0e-04, then 1.0e-05, and the inherited mp=8 rain
                # sedimentation reconciliations (5e4af4e3, cb765336) took
                # the residual it covered from 5.700e-06 to 4.146e-07 --
                # inside the flat 2.0e-06 gate -- so the dict buys nothing
                # and is now empty.  Before that, ``_REFL_DB_BOUNDS``
                # was RETIRED, not relaxed: it carried aero-reduces-to-
                # classic at 1.0e-02 dB, then 1.0e-03 dB, and WP-13a's
                # level-wise sedimentation density took the residual it
                # covered to 3.242e-05 dB -- inside the flat 2.0e-4 dB gate
                # -- so the dict buys nothing and is now empty.  The gate's
                # own ``_G3_ALLOWANCES`` is the authority and
                # tests/test_physics_registry.py::
                # test_mp28_evidence_publishes_the_allowances_the_gate_
                # actually_has reads it back.
                "allowances": [
                    {"name": "near_cancellation_bound",
                     "gate_constant": "_NEAR_CANCELLATION_LEVELS",
                     "fixtures": ["aero-reduces-to-classic"],
                     "was": "level 6 skipped outright",
                     "is": "level 6 held to 32 ulps of the entry value",
                     "direction": "strictly more than the skip asserted"},
                ],
                "retired_allowances": [
                    {"name": "carved_out_bound",
                     "gate_constant": "_END_TO_END_BOUNDS",
                     "fixtures": ["aero-reduces-to-classic"],
                     "was": 1.0e-05,
                     "is": None,
                     "direction": "retired at the 1.4.1 merge; the residual "
                                  "it covered is now 4.146e-07, inside the "
                                  "flat 2.0e-06 gate"},
                    {"name": "reflectivity_bound_db",
                     "gate_constant": "_REFL_DB_BOUNDS",
                     "fixtures": ["aero-reduces-to-classic"],
                     "was": 1.0e-03,
                     "is": None,
                     "direction": "retired; the residual it covered is now "
                                  "3.242e-05 dB, inside the flat 2.0e-4 dB "
                                  "gate"},
                ],
                # The two counts, stated separately, because conflating them
                # is how a port claims a clean number it did not earn.
                "clean_unexceptioned": 17,
                "clean_as_gated": 18,
                "clean_counts_note": (
                    "17 of 22 clear a FLAT 2.0e-6 relative / 2.0e-4 dB gate "
                    "on all 23 quantities with no bounds dict, no excluded "
                    "level and no per-fixture carve-out -- 16 of the 19 "
                    "spec'd aero-* fixtures plus wp08-melt. 18 of 22 clear "
                    "it with the ONE allowance above applied; that allowance "
                    "buys exactly one fixture, aero-reduces-to-classic, and "
                    "is required for it. The second allowance this note used "
                    "to name was retired at the 1.4.1 merge and is in "
                    "retired_allowances. clean_fixtures below is the "
                    "UNEXCEPTIONED list."),
                # No longer None.  docs/public/validation/
                # mp28-matched-trajectory.md is a matched IDEALIZED forecast
                # against unmodified WRF v4.6.1, and it publishes its own
                # FAILED gate rather than a summary of the parts that passed.
                "forecast_trajectory_comparison": {
                    "document": ("docs/public/validation/"
                                 "mp28-matched-trajectory.md"),
                    "kind": "idealized single-domain doubly-periodic forecast",
                    "not_nested_not_real_data": (
                        "A matched nested real-data forecast is not possible "
                        "and was not attempted. WRF's real.exe is a fatal "
                        "error on wif_input_opt=0 with mp_physics=28 "
                        "(dyn_em/module_initialize_real.F:2734-2736), and "
                        "gpuwm has no aerosol lateral boundary condition -- "
                        "the measured depletion front advances at 0.993 of "
                        "the wind speed, so a 100 km nest sits at WRF's "
                        "aerosol floor within 83 minutes. Both blockers are "
                        "BOUNDARY blockers and neither exists on a periodic "
                        "domain."),
                    "case": (
                        "WRF's own em_quarter_ss initializer with the "
                        "hodograph removed: a 3 K cos^2 thermal, 10 km "
                        "radius, centred at z = 1500 m, in an unsheared WK82 "
                        "sounding. 120 x 120 x 40, dx = dy = 2000 m, "
                        "ztop = 20000 m, dt = 12 s, time_step_sound = 6, 600 "
                        "steps = 7200 s, periodic_x = periodic_y = .true., "
                        "microphysics the only physics."),
                    "initial_condition": (
                        "gpuwm is initialised FROM WRF's own wrfinput_d01, "
                        "not from a transcription of WRF's initializer, so a "
                        "transcription difference cannot enter the "
                        "trajectory disguised as a microphysics difference. "
                        "MEASURED: the t = 0 normalised RMS field difference "
                        "is exactly 0.0 in ten of the thirteen compared "
                        "fields, in both configurations. The three that are "
                        "not are the three gpuwm derives rather than copies: "
                        "T at 2.510e-09 (WRF stores theta - 300 and gpuwm "
                        "stores an absolute base plus a perturbation, so the "
                        "field is split and recombined once each way), and "
                        "QNWFA at 2.176e-08 / QNIFA at 1.731e-08, which both "
                        "models overwrite with thompson_init's synthetic "
                        "profile -- so what is compared there is CUDA's "
                        "evaluation of WRF's analytic CCN/IN profile against "
                        "gfortran's. One float32 rounding each."),
                    "wrf_reference": (
                        "WRF v4.6.1 release tarball sha256 b8ec11b2..., "
                        "gfortran 13.3.0, glibc 2.39, netCDF 4.9.2, "
                        "configure option 32 (serial) / nesting 0, built "
                        "TWICE from identical source: build A with WRF's own "
                        "default -O2 -ftree-vectorize -funroll-loops and "
                        "build B with -O2 -fno-tree-vectorize. Build B is "
                        "the control: it measures how far WRF's own answer "
                        "moves under one optimization flag."),
                    "design": (
                        "difference-in-differences. The tested quantity is "
                        "the aerosol SIGNATURE M(mp28) - M(mp8) measured "
                        "inside each model, read against two published "
                        "floors: the mp=8 model-pair disagreement (gpuwm is "
                        "wrf-matched-run there) and WRF's own flag "
                        "sensitivity. Six runs; every gpuwm configuration "
                        "run twice and byte-compared, the 5090 having no "
                        "ECC."),
                    "declared_verdict": "HOLD",
                    "declared_verdict_detail": (
                        "V1 sign agreement PASS 43/47 = 91.5% (threshold "
                        "90%); V2 floor-calibrated magnitude PASS 9/9; V3 no "
                        "scheme-level amplification FAIL, 7 of 197 rows over "
                        "3x (V3 as declared spans M1-M4 and M8: 111 scalar "
                        "rows and 86 field-difference rows; all seven "
                        "failures are scalar); V4 "
                        "bounded/finite/no-depletion PASS. The rule "
                        "declared before the runs was that any failure holds "
                        "mp=28 out of 1.5. It is not being rewritten."),
                    "control_result": (
                        "THE SAME MACHINERY APPLIED TO WRF AGAINST ITSELF "
                        "ALSO FAILS V3 -- build B vs build A, identical "
                        "source, one optimization flag, 17 of 195 rows over "
                        "3x, worst ratio 808. A gate unmodified WRF cannot "
                        "pass against its own recompilation is not measuring "
                        "the port; past t ~ 2400 s this case is chaotic and "
                        "the scalar metrics compare two samples rather than "
                        "two answers. V3 as written was mis-specified."),
                    "diagnostic_results": {
                        "aerosol_budget_agreement_2h_relative": 1.530e-04,
                        "aerosol_budget_agreement_2h_relative_nifa": 2.451e-04,
                        "dual_run_byte_identical": True,
                        "dual_run_frames_byte_compared": {
                            "long_run": 13, "short_window": 11},
                        "long_run_median_mp28_over_mp8_disagreement": 0.691,
                        "long_run_median_scalar_rows_only": 0.628,
                        "short_window_v3_pass": True,
                        "short_window_v3_worst_ratio": 2.195,
                        "short_window_w_rms_5_steps_mp08": 1.800e-02,
                        "short_window_w_rms_5_steps_mp28": 1.797e-02,
                        "short_window_v3_second_worst_ratio": 1.501,
                        "t0_field_rms_difference_exact_zero_fields": 10,
                        "t0_field_rms_difference_max": 2.176e-08,
                    },
                    "short_window_note": (
                        "ADDED AFTER the long runs and labelled as such in "
                        "the document: both models restarted from the SAME "
                        "mature WRF state at t = 1800 s and run 50 steps, "
                        "which is the only regime in which a matched "
                        "trajectory can mean what it says. The statistic "
                        "(M8) and the condition (V3) are the pre-declared "
                        "ones; only the window is new."),
                    "what_it_establishes": (
                        "mp=28's per-step disagreement with unmodified WRF "
                        "is the disagreement mp=8 already has -- 1.797e-02 "
                        "vs 1.800e-02 RMS in w after five steps from a "
                        "mature state -- so the aerosol-aware scheme adds no "
                        "error of its own that this test can see; nwfa and "
                        "nifa are the best-agreeing 3-D fields measured; and "
                        "the domain aerosol budget matches WRF's to "
                        "1.530e-04 over two hours with no depletion trend."),
                    "what_it_does_not_establish": (
                        "nothing about a real-data or nested forecast, "
                        "nothing about aerosol crossing a lateral boundary "
                        "(there is none), nothing past t ~ 2400 s where the "
                        "trajectories have decorrelated, nothing about "
                        "radiative feedback, PBL interaction, heterogeneous "
                        "surface emission or sheared convection (all physics "
                        "but microphysics is off), and nothing about "
                        "correctness against observations. One case, one "
                        "resolution, one sounding, one bubble."),
                },
                # Named so the "UNVERIFIED" warning cannot be read as "never
                # integrated": it HAS been, against itself, and the row says
                # which gate did it and what that gate can and cannot see.
                "self_forecast_gate": {
                    "test": (
                        "tests/test_mp28_forecast_smoke.py::"
                        "test_g4_multistep_specified_bc_forecast_is_finite"
                        "_and_bounded"),
                    "steps": 150,
                    "timestep_s": 12.0,
                    "checks": (
                        "every prognostic and every radiation-facing "
                        "effective radius finite each step, WRF's terminal "
                        "apply bounds holding in the microphysics-updated "
                        "interior, spec-zone ring bit-restored"),
                    "does_not_check": (
                        "agreement with WRF. There is no reference "
                        "trajectory to compare against, so a finite, bounded "
                        "and wrong forecast passes it."),
                    "longest_integration": {
                        "test": (
                            "tests/test_mp28_forecast_smoke.py::"
                            "test_g4_a_two_hour_forecast_stays_finite"
                            "_bounded_and_ring_clean"),
                        "steps": 600,
                        "seconds": 7200.0,
                        "measured": (
                            "0 non-finite values, 0 bound violations and 0 "
                            "spec-zone ring violations across all 600 "
                            "microphysics calls; peak |w| 56.10 m/s; "
                            "domain-total RAINNC 1.9055 mm; final "
                            "domain-interior mean nwfa 1.3395e+07 kg^-1 "
                            "against a floor of 1.110e+07 and nifa "
                            "5.9041e+03 against a floor of 5.000e+03 -- i.e. "
                            "entirely inflow air after 2.6 domain "
                            "ventilation times, exactly as the lateral-"
                            "boundary deviation below predicts"),
                    },
                },
                "test": (
                    "tests/test_thompson_aerosol_adapter.py::"
                    "test_g3_end_to_end_against_all_nineteen_oracle_fixtures"),
            },
            # WHAT AN mp=28 RUN ACTUALLY STARTS FROM.  Published as structure
            # rather than prose, and DERIVED rather than transcribed:
            # tests/test_physics_registry.py scans gpuwm/ for callers of
            # microphysics_init and fails if this row disagrees with the scan
            # in EITHER direction.
            #
            # THIS ROW FLIPPED ON 2026-08-01.  For four waves
            # ``production_call_site`` was null and the option carried a "NO
            # AEROSOL INITIALISATION" warning: microphysics_init implemented
            # WRF's fill, was oracle-gated against it, and nothing called it,
            # so every mp=28 forecast integrated from nwfa = nifa = 0 under
            # WRF's floors.  gpuwm/core/physics.py::initialize_physics now
            # calls it once per domain, exactly where WRF's phy_init calls
            # mp_init.  The measured numbers below did NOT change -- they are
            # the same two forecasts -- but what they mean did: they were the
            # measured COST of the missing call and they are now the measured
            # VALUE of the profile, i.e. how much of an mp=28 forecast the
            # CCN/IN loading owns.
            "aerosol_initialisation": {
                "wrf_source": (
                    "phys/module_mp_thompson.F:493-558 fills a synthetic "
                    "CCN/IN profile at domain construction whenever "
                    "MAXVAL(nwfa) / MAXVAL(nifa) come in below eps -- two "
                    "independent tests, at :493 and :530 -- and :509-510 "
                    "derives the surface emission nwfa2d from the filled "
                    "lowest level. Nothing inside mp_gt_driver ever refills "
                    "it."),
                "wrf_call_site": (
                    "phys/module_physics_init.F::microphysics_init, once per "
                    "domain, before the first step"),
                "gpuwm_implementation": (
                    "gpuwm/core/microphysics.py::microphysics_init"),
                "gpuwm_implementation_evidence": (
                    "tests/test_mp28_runnable.py::"
                    "test_microphysics_init_fills_wrfs_synthetic_ccn_profile"
                    "_for_mp28 -- the fill itself is measured against WRF and "
                    "is NOT what is missing"),
                "production_call_site": (
                    "gpuwm/core/physics.py::initialize_physics"),
                "production_call_site_note": (
                    "once per domain, unconditionally, at the seam WRF uses. "
                    "gpuwm reaches it through microphysics_cold_start, which "
                    "returns an empty receipt for every scheme without a "
                    "domain-construction step. Presence-gated exactly as WRF "
                    "is: the fill runs only where the domain-wide MAXVAL "
                    "tests open, so a nest that inherited its parent's "
                    "aerosol and a restart that is about to be overwritten by "
                    "a checkpoint are both left alone. WRF: "
                    "phys/module_physics_init.F:1635 calls mp_init from "
                    "phy_init, phys/module_physics_init.F:4522-4538 is the "
                    "CASE (THOMPSONAERO) arm that calls thompson_init, and "
                    "the presence tests are "
                    "phys/module_mp_thompson.F:490-493 (CCN) and "
                    "phys/module_mp_thompson.F:528-531 (IN)."),
                "shipped_profile": (
                    "WRF's synthetic thompson_init CCN/IN profile, installed "
                    "at domain construction: on the aero-init-profile "
                    "fixture's grid nwfa runs 1.478987e+08 kg^-1 at the "
                    "lowest level to 5.000000e+07 aloft and nifa 1.254902e+06 "
                    "to 5.000002e+05, with nwfa2d derived from the filled "
                    "surface value at :509-510"),
                "shipped_consequence": (
                    "a freshly initialised mp=28 domain starts strictly ABOVE "
                    "the terminal apply's clamps "
                    "(phys/module_mp_thompson.F:3972-4021, nwfa >= 11.1e6 and "
                    "nifa >= 5.0e3 per m3) rather than pinned at them, so CCN "
                    "activation sees a continental aerosol loading that "
                    "decays with height instead of a maritime-clean floor "
                    "everywhere. The installed-state pin below asserts the "
                    "floors of WRF's own profile constants (naCCN1 = 50.0e6 "
                    "at phys/module_mp_thompson.F:96-97, naIN1 = 0.5e6 at "
                    "phys/module_mp_thompson.F:94-95) -- 4.5x and 100x above "
                    "the clamp floors, so a run that started at zero and got "
                    "clamped cannot satisfy it."),
                "operator_workaround": (
                    "none needed: the call is wired. An embedder that builds "
                    "a DomainState WITHOUT going through initialize_physics "
                    "gets no profile, and gpuwm.core.microphysics."
                    "microphysics_init is the entry point to call once per "
                    "domain before the first step; it is presence-gated, so "
                    "calling it twice is a no-op, and it returns an empty "
                    "receipt for every scheme other than 28. Do NOT call it "
                    "per step: nothing in mp_gt_driver refills the profile, "
                    "and a per-step call would overwrite an advected, "
                    "activated and scavenged aerosol field with the synthetic "
                    "one while leaving every bound intact."),
                "measured_forecast_sensitivity": {
                    "test": (
                        "tests/test_mp28_forecast_smoke.py::"
                        "test_the_aerosol_profile_changes_the_forecast"
                        "_measurably"),
                    "steps": 150,
                    "timestep_s": 12.0,
                    "domain": (
                        "28 x 16 x 24 at dx = 2 km, specified BC, warm "
                        "bubble, two runs identical but for the init call"),
                    "initial_mean_nwfa_per_kg_with_profile": 6.6532e7,
                    "initial_mean_nwfa_per_kg_without_profile": 0.0,
                    "final_interior_nwfa_per_kg_with_profile": 3.0059e7,
                    "final_interior_nwfa_per_kg_without_profile": 1.2620e7,
                    "peak_nc_per_kg_with_profile": 1.5980e8,
                    "peak_nc_per_kg_without_profile": 2.8439e7,
                    "droplet_ratio_with_over_without": 5.62,
                    "domain_total_rainnc_mm_with_profile": 1.781185,
                    "domain_total_rainnc_mm_without_profile": 3.102043,
                    "domain_total_rainnc_relative_excess": 0.7416,
                    "peak_rainnc_mm_with_profile": 0.675036,
                    "peak_rainnc_mm_without_profile": 0.925492,
                    "reading": (
                        "the LEFT column is what a run does today; the RIGHT "
                        "column is the counterfactual with the profile "
                        "removed. Removing WRF's CCN/IN loading gives 5.6x "
                        "fewer cloud droplets and 74.2% MORE domain-total "
                        "surface rain over 30 minutes. Until 2026-08-01 the "
                        "right column WAS the shipped behaviour and this was "
                        "published as the port's largest measured error; it "
                        "is now the measured sensitivity of an mp=28 forecast "
                        "to its aerosol initial condition, which is also the "
                        "magnitude the lateral-boundary deviation below "
                        "converges to after L/U."),
                    "note": (
                        "a SNAPSHOT on this tree, not a physics pin: any "
                        "legitimate mp=28 numerics change moves these digits. "
                        "What is published is the sign and the order of "
                        "magnitude, and tests/test_physics_registry.py::"
                        "test_the_published_aerosol_initialisation_cost_is"
                        "_still_the_measured_one re-runs both forecasts to "
                        "check them."),
                },
                "call_site_pin": (
                    "tests/test_mp28_forecast_smoke.py::"
                    "test_microphysics_init_has_a_production_call_site"),
                "installed_state_pin": (
                    "tests/test_mp28_forecast_smoke.py::"
                    "test_a_freshly_initialised_mp28_domain_carries_the"
                    "_profile_not_zero"),
            },
            "activation_bin_edge_policy": (
                "activ_ncloud selects a NEAREST 10 K temperature bin and "
                "truncates idx_d/idx_c/idx_n with INT(), so nc is a STEP "
                "function of state. Within one ulp of a bin edge an FP32 GPU "
                "port and the Fortran reference can select different bins and "
                "differ by tens of percent in nc while every mass field "
                "agrees. Fixture states are chosen away from bin edges and "
                "the behaviour is documented here rather than absorbed into a "
                "loose tolerance."),
        },
        "implemented": True,
        "label": "Thompson aerosol-aware / MP28",
        "maturity": "implemented-unverified",
        # Exactly thompson-mp8's pins.  aer_init_opt and aer_fire_emit_opt are
        # DERIVED in WRF (Registry.EM_COMMON:2656/:2658), not namelist knobs,
        # so they are not gpuwm settings and are not pinned here; wif_input_opt
        # and the rest of the WIF family are published as implemented=false
        # roadmap rows, and an implemented option may not pin one of those.
        "parameters": {"moist": True, "moist_cq": True},
        "reachability": {"state": "component-override"},
        "selectors": {"mp_physics": 28},
        "warnings": [
            "UNVERIFIED against a WRF forecast in the sense that matters "
            "operationally: there is no matched REAL-DATA or NESTED "
            "trajectory and no decay table, and both blockers stand until "
            "an aerosol lateral boundary condition exists. There IS now a "
            "matched IDEALIZED trajectory -- a doubly periodic "
            "single-domain warm-bubble forecast against unmodified WRF "
            "v4.6.1, recorded in extensions.column_oracle_evidence."
            "forecast_trajectory_comparison -- and its pre-declared gate "
            "FAILED on one of its four conditions, V3. Read that entry "
            "before relying on this scheme: it also records that WRF fails "
            "the same condition against its own recompilation, which is why "
            "the failure is published rather than acted on. The single-call "
            "evidence is 22 committed WRF v4.6.1 column fixtures plus "
            "per-kernel and device-helper oracles. mp_physics=28 has also "
            "been integrated multi-step against ITSELF: "
            "tests/test_mp28_forecast_smoke.py::"
            "test_g4_multistep_specified_bc_forecast_is_finite_and_bounded "
            "runs 150 steps x 12 s on a specified-BC convective domain and "
            "checks that every prognostic and every radiation-facing "
            "effective radius stays finite, that WRF's own terminal bounds "
            "hold, and that the spec-zone ring is bit-restored; "
            "test_g4_a_two_hour_forecast_stays_finite_bounded_and_ring_clean "
            "carries the same domain to 600 steps (7200 s) with 0 non-finite "
            "values, 0 bound violations and 0 ring violations. Finite and "
            "bounded is NOT correct: a scheme with a systematically wrong "
            "activation rate passes every one of those checks for two hours. "
            "The bounds are WRF's, but they are clamps, not answers.",
            "THE COLUMN EVIDENCE IS NOT CLEAN, and the numbers are published "
            "rather than summarised. Driven end to end through the shipped "
            "adapter, 22 fixtures x 23 quantities, at a flat 2.0e-6 relative "
            "/ 2.0e-4 dB gate with nothing held out: 17 of 22 clear every "
            "quantity (16 of the 19 spec'd aero-* fixtures, plus wp08-melt). "
            "aero-reduces-to-classic clears only through the port's ONE "
            "surviving named allowance -- 0-based level 6 held to 32 ulps of "
            "its entry value instead of the relative metric, measured 0.585 "
            "(qr) and 0.159 (nr) -- taking the gated count to 18 of 22. The "
            "relative bound that used to sit beside it was RETIRED at the "
            "1.4.1 merge: the mp=8 lane's two rain sedimentation "
            "reconciliations (5e4af4e3, cb765336), inherited in the frozen "
            "kernel mp=28 shares for fallout, took level 5's nr from "
            "5.700e-06 to 4.146e-07, inside the flat gate. FOUR MISS "
            "OUTRIGHT -- aero-cold-overlap qc 1.000e+00 / nc 1.000e+00 / "
            "effc 8.102e-01 (all three are ONE branch flip at 0-based level "
            "4, where WRF ends with 1.4551915228366852e-11 kg/kg of cloud "
            "water -- exactly one float32 ulp of the entry value -- and "
            "gpuwm ends at exactly zero, so the qc1d <= R1 test at "
            "phys/module_mp_thompson.F:4007 sends "
            "the two implementations down opposite arms and a relative "
            "metric reports full scale on a one-ulp difference) plus "
            "nr 1.261e-04 / qr 4.443e-05 at level 6; "
            "aero-cloud-freeze-nc qc 4.926e-06; wp08-freeze nr "
            "2.724e-06 (1.4x the "
            "gate); wp08-nusweep qr 4.642e-06 (2.3x the gate). Every one of "
            "the surviving residuals now sits where the field is either "
            "CREATED FROM ZERO inside the step or driven to near-total "
            "consumption; after the aero-ice-koop withdrawal recorded at the "
            "end of this warning there is no surviving residual in the "
            "rate-disagreement class at all. That is stated, not "
            "used: no bound is relaxed for it. WHAT MOVED SINCE THE LAST "
            "PUBLISHED SET, and it is a REAL PORT FIX this time -- TWO of "
            "them, both in mp=28-owned kernels. WP-13a restored WRF's "
            "LEVEL-WISE sedimentation density: WRF forms the "
            "working rain mass and number twice -- "
            "phys/module_mp_thompson.F:3237-3238 from the :3193 TAU+1 "
            "density at every L_qr level, and :3568/:3570 from the :3490 "
            "post-condensation density but ONLY inside the :3501-3502 gate "
            "-- and gpuwm/core/kernels/thompson_aerosol_sat.cu was exporting "
            "the post-condensation density unconditionally. WP-13b pinned "
            "the source-network apply against contraction: WRF's terminal "
            "apply at phys/module_mp_thompson.F:3973-4023 "
            "is q1d(k) = q1d(k) + qten(k)*DT and the gfortran -O2 oracle has "
            "no FMA, so qten*DT is rounded to REAL(4) before the add, while "
            "nvrtc was fusing it in thompson_aerosol_cold.cu and "
            "thompson_aerosol_warm.cu. Together they "
            "took aero-drop-evap (rainnc 5.165e-04, qr 3.533e-05, nr "
            "2.258e-05; WP-13a alone) and aero-ice-demott-idxin (rainnc "
            "1.279e-04, qr 2.894e-05, nr 4.594e-06; both changes, RAINNC and "
            "sr bitwise 0 after WP-13b) to CLEAN, took "
            "aero-cloud-freeze-nc's qr 2.800e-05 / nr 1.797e-05 / rainnc "
            "1.162e-05 rows to 8.973e-08 / 2.594e-07 / 0.000e+00, deleted "
            "qr from the carve-out (7.813e-05 -> 1.788e-07, inside the flat "
            "gate), tightened the surviving nr bound 1.0e-04 -> 1.0e-05 and "
            "RETIRED the reflectivity carve-out (5.283e-04 dB -> 3.242e-05 "
            "dB). One number went the OTHER way and is published as "
            "measured: aero-cold-overlap qr 3.667e-05 -> 4.443e-05, WP-13b's "
            "cost, bisected to the cold network's qr apply and KEPT because "
            "reverting that one line also loses all four of "
            "aero-ice-demott-idxin's improvements, two of which are bitwise; "
            "in ulps of the entry value the move is 1.477 -> 1.789. "
            "AND WHAT MOVED BEFORE THAT, WHICH WAS NOT "
            "A PHYSICS FIX: aero-ice-koop, published by four waves of this "
            "registry as the port's largest genuine physics gap at qi "
            "1.612e-03 / ni 1.764e-03 / effi 5.093e-05, now measures "
            "1.534e-07 / 3.396e-07 / 1.886e-07 and is CLEAN -- and NO KERNEL "
            "CHANGED. The cause was this port's own oracle harness: "
            "tools/thompson_wrf461_oracle/run_column_aero.F90 built the Exner "
            "function with rd_over_cp = 287.0/1004.0, while WRF's own rcp is "
            "r_d/cp with r_d=287. and cp=7.*r_d/2.=1004.5 (declared in "
            "share/module_model_constants.F at lines 19, 20 and 31) -- "
            "exactly 2/7, and "
            "4774 float32 ulps from what the harness used. 47 of the deck's "
            "528 entry levels (first published as 40; re-pinned 2026-08-03, "
            "owner-ratified -- the count is the host libm's, and "
            "mp28-column-evidence.md section 3.4 carries the correction "
            "note) then had no exact float32 theta, the adapter "
            "perturbed the entry pressure by up to 15 ulps to recover the "
            "recorded temperature, and that perturbed pressure drove "
            "different microphysics on 7 fixtures including this one. The "
            "deck was regenerated with WRF's own constant and the residual "
            "went with it; the same regeneration made aero-cold-overlap "
            "WORSE, which is why it can be trusted. See "
            "extensions.column_oracle_evidence for the full "
            "table, the allowance list and the two counts, and "
            "docs/public/validation/mp28-column-evidence.md section 3.4 for "
            "the re-derivation from the committed fixtures.",
            "NO AEROSOL INGEST -- but the aerosol INITIALISATION is wired, "
            "and that distinction is the whole of this warning. INGEST: "
            "gpuwm has no WIF metgrid stream, no GOCART climatology reader "
            "and no black-carbon (nbca) species, so use_aero_icbc, "
            "use_rap_aero_icbc, wif_input_opt, num_wif_levels and qna_update "
            "are all published implemented=false and refuse. INITIALISATION: "
            "WRF's own fallback for exactly that case is thompson_init's "
            "SYNTHETIC CCN/IN profile (phys/module_mp_thompson.F:493-558), "
            "gpuwm implements it in "
            "gpuwm/core/microphysics.py::microphysics_init, measures it "
            "against WRF, and -- since 2026-08-01 -- CALLS it, once per "
            "domain, from gpuwm/core/physics.py::initialize_physics, at the "
            "seam WRF calls mp_init from phy_init. A freshly initialised "
            "mp=28 domain therefore starts on WRF's decaying continental "
            "profile, strictly above the terminal apply's clamps "
            "(phys/module_mp_thompson.F:3972-4021), not pinned at them. HOW "
            "MUCH THAT IS WORTH, measured over 150 steps x 12 s on a "
            "28 x 16 x 24 2 km specified-BC convective domain against an "
            "otherwise identical run with the profile removed: initial mean "
            "nwfa 6.6532e+07 vs 0.0 kg^-1, peak nc 1.5980e+08 vs 2.8439e+07 "
            "kg^-1 (5.6x fewer droplets without it), domain-total RAINNC "
            "1.781185 vs 3.102043 mm -- the aerosol-free run rains 74.2% "
            "MORE. Read that as the sensitivity of an mp=28 forecast to its "
            "aerosol initial condition; it is also the magnitude the "
            "lateral-boundary deviation below converges to after the domain "
            "has ventilated once. See extensions.aerosol_initialisation.",
            "DIVERGENCE, admission: WRF's own initializer REFUSES the "
            "configuration gpuwm runs. dyn_em/module_initialize_real.F:"
            "2734-2736 calls wrf_error_fatal('wif_input_opt=0 but "
            "mp_physics=28'), so real.exe will not build a wrfinput for the "
            "synthetic-profile case at all. The PHYSICS is WRF's; the "
            "admission decision is not.",
            "DIVERGENCE, lateral boundaries: an external-BC (specified) "
            "domain carries NO aerosol inflow, and this is the deviation "
            "that grows with run length. gpuwm couples only qv from external "
            "boundary snapshots and gives every other scalar flow-dependent "
            "boundaries with zero inflow, so aerosol-free air advects in at "
            "the upstream face and monotonically depletes nwfa/nifa for as "
            "long as the run continues -- with no NaN, no negative and no "
            "health trip, because WRF's own terminal clamps (nwfa >= 11.1e6, "
            "nifa >= 5.0e3 per m3) hold the floor. WRF's Registry gives "
            "qnwfa/qnifa real bdy arrays and forces them from the boundary "
            "file. MEASURED on a deliberately cloud-free 150-step run, so "
            "every kilogram lost is the boundary policy and not "
            "microphysics: with a 20.0 m/s inflow the depletion front "
            "advances at 19.8638 m/s (0.99319 of the wind), and over 1800 s "
            "the domain-interior mean nwfa falls to 0.4566 of its initial "
            "value and nifa to 0.3363. At 10 m/s the front runs at "
            "9.808 m/s, so this is a law and not one number: the upstream "
            "U*t of your domain is at WRF's aerosol floor after time t, and "
            "the whole domain after L/U -- 13.9 hours for a 1000 km domain "
            "in a 20 m/s flow, 83 minutes for a 100 km nest. The only "
            "interior source is the fixed surface emission nwfa2d at k=0, "
            "measured at 5540.14 kg^-1 s^-1, which replaces about 5% of the "
            "lowest level's initial loading over 1800 s and acts on that "
            "level only. SEPARATELY, the spec_zone ring itself ends at "
            "EXACTLY zero aerosol on three of its four faces (west/south/"
            "north 1.000, east 0.125), because WRF's clipped microphysics "
            "tile means the terminal clamp never runs there -- a value WRF "
            "itself cannot produce, and what a nest boundary or a wrfout "
            "reader sees. This matches gpuwm's existing hydrometeor policy "
            "and is documented, not fixed.",
            "DIVERGENCE, PBL: gpuwm passes flag_qnc/flag_qnwfa/flag_qnifa to "
            "MYNN as literal False (gpuwm/core/mynn_pbl.py), so nc/nwfa/nifa "
            "are never vertically mixed by the PBL. WRF mixes them when "
            "bl_mynn_mixscalars=1 (phys/module_bl_mynn.F:4735,:4777,:4957) or "
            "through scalar_pblmix (phys/module_pbl_driver.F:2251). At "
            "gpuwm's pinned MYNN identity bl_mynn_mixscalars=0, and WRF's "
            "check_a_mundo raises scalar_pblmix to 1 ONLY when use_aero_icbc "
            "or use_rap_aero_icbc is set (share/module_check_a_mundo.F:"
            "2477-2495) -- both of which gpuwm refuses -- so WRF's own value "
            "here is 0 too and today the two "
            "models agree -- but gpuwm's withholding is STRUCTURAL rather "
            "than a namelist value, and mp_physics=28 is the first "
            "configuration where those species exist and the withholding is "
            "physically visible.",
            "MIXED NESTING IS REFUSED BY NAME. An mp_physics=28 domain may "
            "only sit under an mp_physics=28 parent. No cross-scheme "
            "transition rule is registered for it, so the registry refuses "
            "any mixed edge with unsupported-component-transition and "
            "gpuwm/core/microphysics_transition.py refuses the same edge at "
            "runtime with a named message. WRF's own non-aerosol-aware "
            "fallbacks (nc=100e6/rho, nwfa=11.1e6/rho, nifa=5.0e3/rho) would "
            "give a nested child fabricated aerosol rather than its parent's, "
            "and no gate in this tree would flag it.",
            "DELIBERATE THERMODYNAMIC DIVERGENCE FROM mp_physics=8, and it is "
            "not a defect on either side. mp=28's RSLF/RSIF saturation Horner "
            "chains are contraction-pinned while mp=8's stay FMA-contracted, "
            "so the two schemes' saturation vapour pressures differ by one "
            "ulp. That matters because module_mp_thompson.F:3401 opens the "
            "whole condensation/CCN-activation block on ssatw > 1.E-15 (:185) "
            "-- one ulp flips a branch. mp=28 matches WRF's own gfortran "
            "-O2 arithmetic; mp=8 stays byte-frozen at its model-validated "
            "trajectory. The two are deliberately not bit-identical.",
            "Reachable only as a per-domain microphysics override on the "
            "experiment-per-domain tree route. No registered template selects "
            "it, it is no template's default, and the shipped default suite "
            "is unchanged.",
            "Launch must byte-validate CCN_ACTIVATE.BIN (35,288 bytes, "
            "sha256 f2b8d391...) plus the four classic Thompson tables. The "
            "activation table IS distributed with gpuwm as of 2026-08-01 -- "
            "WRF v4.6.1's own run/CCN_ACTIVATE.BIN, bit for bit -- so a "
            "default install has it; GPUWM_THOMPSON_CCN_ACTIVATE and "
            "GPUWM_THOMPSON_TABLE_ROOT still redirect a run to another copy. "
            "Absence is fatal, never defaulted, and a byte-different table is "
            "refused: a different parcel-model table would silently be a "
            "different activation scheme. The consequence for EVIDENCE, not "
            "just for launch: every device gate for the scheme, including all "
            "22 column fixtures, skips by name if the file is ever missing. "
            "The skip names the one file rather than swallowing a load "
            "failure, so it can never be mistaken for a pass.",
        ],
    }


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
        # The count is computed, not typed: a typed "2,400" survived two
        # axis widenings (mp=28, bl_pbl=11) as stale prose.
        "test": (
            f"tests/test_wrf461_compatibility.py sweeps all "
            f"{MATRIX_CELL_COUNT:,} cells and "
            "requires every cell to carry all six WRF citations"),
    }


#: Path A of the owner-ratified vocabulary decision (D-1/D-2).  The old
#: spellings claimed validation the labels never carried: agreement with a
#: matched WRF run is agreement with another model, and "validated" is read
#: by everyone as skill against the atmosphere.  The new spellings say what
#: the evidence is.  Applied to maturity VALUES at every surface; template
#: ids keep their historical spelling because an id is a name, not a claim,
#: and renaming one silently breaks every config and route that cites it.
MATURITY_RENAMES = {
    "model-validated": "wrf-matched-run",
    "validation-candidate": "wrf-matched-run-candidate",
}

#: The conformance axis, lowest evidence first.  This tuple is the ONLY
#: ordering of maturity names in the repository: gpuwm/physics_registry.py
#: derives MATURITY_RANK, both warning tiers and the composition ceiling
#: from the block this builds, so a rung cannot be reordered in one place
#: and not the other.
_MATURITY_RUNGS = (
    ("planned", "unimplemented-only",
     "Registered so the roadmap is readable. No GPUWM runtime component "
     "exists, implemented is false, and the option cannot be selected."),
    ("port-in-progress", "unimplemented-only",
     "A port exists in the tree but does not reach the runtime. "
     "implemented is false and the option cannot be selected."),
    ("implemented-unverified", "warn",
     "A GPUWM runtime component executes this option and no matched "
     "ArWen-versus-WRF forecast trajectory has been run with it. "
     "Selecting it warns and does not block."),
    ("experimental-runtime", "warn",
     "Executable, and carrying a documented runtime restriction or an "
     "unratified composition -- a table-bound runtime, or a nest edge "
     "between two microphysics schemes. Selecting it warns and does not "
     "block."),
    ("supported", "nonwarning",
     "Agreement with the WRF reference implementation is settled for this "
     "option by committed oracle parity or an equivalent comparison, and "
     "it carries no runtime restriction. Selecting it does not warn."),
    ("wrf-matched-run-candidate", "warn",
     "Executable and gated, with a ratified reference comparison, and "
     "deliberately not the default: the next candidate for a full matched "
     "run. Selecting it warns and does not block."),
    ("wrf-matched-run", "nonwarning",
     "A matched multi-hour ArWen-versus-WRF forecast of a reference case "
     "has been run with this option and its decay tables are published. "
     "This is agreement with WRF, not skill against observations."),
)

#: The independent-science axis (D-26: options only).  ``none`` is the
#: honest default and the only value this pass assigns: the value set above
#: it is the ratified catalogue's, and an option is promoted off ``none``
#: only by an entry in ``scientific_evidence_catalogue``.  Assigning a
#: category here without that entry would be exactly the unbacked claim the
#: two-axis split exists to prevent.
_SCIENTIFIC_ENUM = (
    ("none",
     "No independent scientific evidence is claimed for this option. Its "
     "evidence is conformance with the WRF reference implementation, which "
     "lives on the other axis."),
    ("idealized-analytic",
     "Compared against a closed-form or analytically constrained solution "
     "of an idealized problem."),
    ("converged-numerical-reference",
     "Compared against a converged high-resolution numerical reference "
     "solution of an idealized problem."),
    ("conservation-gated",
     "Carries a committed conservation residual gate over an idealized "
     "problem."),
    ("obs-evaluated",
     "Evaluated against observations. No option in this registry carries "
     "this value; the rung exists so the absence is visible rather than "
     "unrepresentable."),
)


def _rename_maturities(registry: dict) -> None:
    """Rewrite every maturity VALUE at every surface under Path A."""

    for component in registry.get("components", {}).values():
        for option in component.get("options", {}).values():
            if option.get("maturity") in MATURITY_RENAMES:
                option["maturity"] = MATURITY_RENAMES[option["maturity"]]
    for template in registry.get("templates", {}).values():
        if template.get("maturity") in MATURITY_RENAMES:
            template["maturity"] = MATURITY_RENAMES[template["maturity"]]
    for transition in registry.get("transitions", {}).values():
        for rule in transition.get("cross_options", []):
            if rule.get("maturity") in MATURITY_RENAMES:
                rule["maturity"] = MATURITY_RENAMES[rule["maturity"]]
        same = transition.get("same_option")
        if isinstance(same, dict) and same.get("maturity") in MATURITY_RENAMES:
            same["maturity"] = MATURITY_RENAMES[same["maturity"]]


def _evidence_architecture(registry: dict) -> None:
    """Author the two axes, the ladder and the composition rule."""

    rungs = {
        name: {
            "rank": rank,
            "warning_tier": tier,
            "definition": definition,
        }
        for rank, (name, tier, definition) in enumerate(_MATURITY_RUNGS)
    }
    order = [name for name, _tier, _definition in _MATURITY_RUNGS]

    registry["maturity_ladder"] = {
        "axis": "conformance",
        "order": order,
        "rungs": rungs,
        "meaning": (
            "How far agreement with the WRF reference implementation has "
            "been demonstrated for this component, template or nest edge. "
            "Every rung is a statement about agreement with another model "
            "and none of them is a statement about skill against "
            "observations."),
        "composition_rule": _composition_rule(),
    }
    registry["evidence_axes"] = {
        "conformance_implies_scientific_validation": False,
        "conformance_implies_scientific_validation_meaning": (
            "Agreement with WRF is agreement with a model. No rung of the "
            "conformance ladder implies any value on the scientific axis, "
            "and the two are reported separately everywhere."),
        "maturity": {
            "axis": "conformance",
            "ladder": "maturity_ladder",
            "rungs": rungs,
            "surfaces": [
                "components.<component_id>.options.<option_id>.maturity",
                "templates.<template_id>.maturity",
                "transitions.<transition_id>.cross_options[].maturity",
            ],
            "absent_surfaces": [
                {
                    "surface": (
                        "transitions.<transition_id>.cross_options[] whose "
                        "status is 'ratified'"),
                    "owner_decision_id": "D-26",
                    "contract": (
                        "A ratified nest edge carries no maturity key. The "
                        "absence is the contract, not an omission: the edge "
                        "was ratified as a whole against its per-species "
                        "receipt, so there is no separate conformance rung "
                        "to report for it. An unratified edge carries "
                        "'experimental-runtime' and warns."),
                },
            ],
        },
        "scientific": {
            "axis": "independent-scientific-evidence",
            "default": "none",
            "enum": {
                name: {"definition": definition}
                for name, definition in _SCIENTIFIC_ENUM
            },
            "surfaces": [
                "components.<component_id>.options.<option_id>"
                ".scientific_evidence",
            ],
            "absent_surfaces": [
                {
                    "surface": "templates.<template_id>",
                    "owner_decision_id": "D-26",
                    "contract": (
                        "Templates carry no scientific_evidence. A template "
                        "is a composition of options and inherits no "
                        "independent evidence by being composed; read the "
                        "axis on the options it selects."),
                },
                {
                    "surface": "transitions.<transition_id>.cross_options[]",
                    "owner_decision_id": "D-26",
                    "contract": (
                        "A nest edge carries no scientific_evidence for the "
                        "same reason."),
                },
            ],
            "catalogue_contract": (
                "An option is promoted off 'none' only by an entry in "
                "scientific_evidence_catalogue naming the artifact and "
                "quoting its category basis. Every option in this registry "
                "reads 'none': their evidence is conformance, which lives "
                "on the other axis."),
        },
    }

    for component in registry.get("components", {}).values():
        for option in component.get("options", {}).values():
            option.setdefault("scientific_evidence", "none")

    tier_of = {name: tier for name, tier, _definition in _MATURITY_RUNGS}
    policy = registry["warning_policy"]
    policy["nonwarning_maturities"] = [
        name for name in order if tier_of[name] == "nonwarning"]
    policy["warn_maturities"] = [
        name for name in order if tier_of[name] == "warn"]
    policy["tier_authority"] = (
        "Both lists are computed from maturity_ladder.rungs[].warning_tier "
        "in ladder order. There is no second ordering of maturity names.")


def _composition_rule() -> dict:
    """The owner-ratified two-clause rule (D-16), axis A, as an invariant.

    Enforcement point is the registry document, not the loader.  A blocking
    load-time ERROR on clause C2 would take WSM6 and both NSSL-2 templates
    out of service, so the rule is enforced by
    ``tests/test_physics_registry_composition.py`` over the shipped
    document while the loader enforces only the two-axis membership that
    cannot take a working template out of service (D-22).
    """

    return {
        "id": "template-composition-ceiling-v1",
        "axis": "A-strict-min",
        "owner_decision_id": "D-16",
        "enforcement_point": "registry-document-invariant",
        "severity": "invariant-test",
        "enforcement_ref": "tests/test_physics_registry_composition.py",
        "clauses": {
            "C1": {
                "name": "trajectory pointer",
                "statement": (
                    "A template whose maturity is at or above "
                    "'wrf-matched-run-candidate' carries an evidence_pointer "
                    "that resolves to a matched-run manifest under "
                    "gpuwm/authorities/matched_runs/."),
            },
            "C2": {
                "name": "composition ceiling",
                "statement": (
                    "A template's maturity rank does not exceed the lowest "
                    "maturity rank among the component options it selects. "
                    "A composed suite is only as conformant as its weakest "
                    "member."),
            },
        },
        "discharge": (
            "A clause is discharged for a template either by a resolvable "
            "evidence_pointer -- a whole-suite matched run outranks a "
            "component-wise minimum, because the suite itself was compared "
            "-- or by an entry in composition_exemptions naming the "
            "owner-decision id that granted it. Nothing is discharged "
            "silently."),
        "composition_exemptions": _composition_exemptions(),
    }


def _composition_exemptions() -> dict:
    """Every template the ratified axis flags without a matched-run pointer.

    Each entry names the decision that granted it and states, in the terms
    of the rule, what is missing.  These are not waivers of the finding;
    they are the finding, written down where the checker reads it.
    """

    unverified_land_pbl = (
        "Noah and YSU are 'implemented-unverified': neither has its own "
        "matched ArWen-versus-WRF forecast trajectory, so the strict-min "
        "ceiling for every suite selecting them is 'implemented-unverified'."
    )
    return {
        "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1": {
            "owner_decision_id": "D-16",
            "clause": "C2",
            "basis": (
                "The Morrison reference campaign matched the whole suite "
                "against WRF, and no matched-run manifest for it has been "
                "assembled yet. " + unverified_land_pbl),
        },
        "thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1": {
            "owner_decision_id": "D-16",
            "clause": "C2",
            "basis": (
                "The default template selects the substitution radiation "
                "engine, and the matched run of record was produced with "
                "the exact legacy engine, so its manifest does not cover "
                "this template's tuple. " + unverified_land_pbl),
        },
        "thompson-mp8-ysu-mm5-noah-validation-v1": {
            "owner_decision_id": "D-16",
            "clause": "C2",
            "basis": (
                "A table-bound experimental runtime carried above its "
                "component floor. " + unverified_land_pbl),
        },
        "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-validation-candidate-v1": {
            "owner_decision_id": "D-16",
            "clause": "C1+C2",
            "basis": (
                "NSSL-2 has fused-process oracles and a ratified 500 m "
                "comparison, and no matched-run manifest. "
                + unverified_land_pbl),
        },
        "nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-validation-candidate-v1": {
            "owner_decision_id": "D-16",
            "clause": "C1+C2",
            "basis": (
                "The legacy-RRTMG sibling of the entry above, granted on "
                "the same basis."),
        },
        "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1": {
            "owner_decision_id": "D-16",
            "clause": "C1+C2",
            "basis": (
                "The observation battery's registered composition (lead "
                "ruling, obs-battery integration wave 2026-08-04): "
                "Thompson mp8 is wrf-matched-run and the legacy RRTMG "
                "engine is the certified WRF v4.6.1 port, but no receipt "
                "covers the composed suite yet -- the battery shakedown "
                "case's stock-WRF-paired t0/case receipt is the named "
                "payer. " + unverified_land_pbl),
        },
        "thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1": {
            "owner_decision_id": "D-16",
            "clause": "C1+C2",
            "basis": (
                "The gray-zone sibling of the entry above -- the same "
                "composition with Shin-Hong 2015 in place of YSU -- "
                "granted on the same basis, with the same payer: this "
                "composition's first stock-WRF-paired t0/case receipt. "
                "Shin-Hong's own port is measured bitwise against the "
                "byte-frozen WRF v4.6.1 module on both halves (max ULP 0 "
                "on the float32 CPU authority; 0 ULP on the CUDA heat "
                "tendency), which is conformance evidence and not a "
                "matched forecast trajectory. Noah and Shin-Hong are "
                "'implemented-unverified': neither has its own matched "
                "ArWen-versus-WRF forecast trajectory, so the strict-min "
                "ceiling for this suite is 'implemented-unverified'."),
        },
        "wsm6-ysu-mm5-noah-no-radiation-v1": {
            "owner_decision_id": "D-16",
            "clause": "C2",
            "basis": (
                "The reference no-radiation suite is 'supported' on its own "
                "oracle parity. " + unverified_land_pbl),
        },
    }


def build(registry: dict) -> dict:
    """Apply this pass's tables to ``registry`` in place and return it."""
    _surface_coupling_warnings(registry)
    _thompson_aerosol_mp28(registry)
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
    # phys/module_physics_init.F:3699-3704,3837-3839.  In particular MYNN
    # PBL accepts the revised and classic MM5 surface layers, and the MYNN
    # surface layer is legal with PBL off.  These declarative constraints
    # mirror the same 16-cell table used by runtime admission.
    pbl_options = registry["components"]["pbl"]["options"]
    surface_options = registry["components"]["surface_layer"]["options"]
    pbl_options["ysu"]["constraints"]["requires_components"][
        "surface_layer"
    ] = ["revised-mm5", "classic-mm5"]
    pbl_options["mynn"]["constraints"]["requires_components"][
        "surface_layer"
    ] = ["revised-mm5", "classic-mm5", "mynn"]
    # Shin-Hong (bl_pbl_physics=11) requires isfc=1 exactly as YSU does,
    # through WRF's own SHINHONGSCHEME arm: phys/module_physics_init.F:
    # 3702-3704 fatals unless sf_sfclay_physics initialized isfc=1, which
    # only the revised and classic MM5 surface layers do.
    pbl_options["shinhong"]["constraints"]["requires_components"][
        "surface_layer"
    ] = ["revised-mm5", "classic-mm5"]
    # Shin-Hong is now selected by a registered template
    # (thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1, below), which is
    # the easiest path to it, so its recomputed reachability is
    # "template".  It remains a legal per-domain override on the tree
    # route as well (allowed_component_options below); the state names
    # the easiest path, not the only one.  It stays
    # "implemented-unverified": the template that selects it is a
    # composition candidate, not a matched forecast trajectory for this
    # closure.
    pbl_options["shinhong"]["reachability"] = {"state": "template"}
    # SASE is not in the WRF v4.6.1 table above -- WRF has no such scheme,
    # which is why it carries an out-of-namespace selector.  Its
    # surface-layer constraint is therefore NOT a transcription of WRF's
    # 12-cell matrix but a statement of what the closure reads: any
    # surface-layer scheme that produces a friction velocity and the
    # heat/moisture fluxes serves, and only "off" does not.
    #
    # "mynn" is NOT among them, and the reason is the other option's
    # constraint rather than anything SASE needs.  The MYNN surface layer
    # declares requires_components pbl = [off, mynn], transcribed from
    # WRF's 16-cell matrix (phys/module_physics_init.F:3699-3704,
    # 3837-3839) and pinned by tests/test_physics_registry.py::
    # test_mynn_component_dependencies_are_the_wrf_v461_cells.  SASE is
    # neither of those, so the pairing was refused by the registry while
    # gpuwm.config.validate_sase_config admitted it -- a disagreement of
    # 512 combinations that only became visible once the turbulence
    # component gained its km_opt=0 option and SASE plans could resolve
    # at all.  Listing what the closure would ACCEPT while the other half
    # refuses to run is not an admission, so the intersection is what is
    # declared here, and validate_sase_config now refuses the same pair.
    pbl_options["sase"]["constraints"]["requires_components"][
        "surface_layer"
    ] = ["revised-mm5", "classic-mm5"]
    # Stated in this option's own fourth warning as prose since it was
    # written ("it requires moist=true"), and enforced by
    # validate_sase_config, but never declared machine-readably -- so the
    # registry called 72 dry SASE combinations launchable that the loader
    # then refused.  The closure mixes vapour, cloud water and cloud ice
    # beside theta and forms its stability from the SATURATED
    # Brunt-Vaisala frequency; a dry state has nothing for it to integrate.
    pbl_options["sase"]["constraints"]["required_settings"]["moist"] = True
    pbl_options["sase"]["reachability"] = {"state": "component-override"}
    # Grell-Freitas (cu_physics=3), the first cumulus option admitted
    # since KF and the first scale-aware one: sig = (1-frh)^2 is the
    # scheme's own dx taper, so per-domain admission carries no grid gate.
    # No template selects it -- the shinhong/sase posture: a user asks for
    # it explicitly or does not get it.
    cumulus_options = registry["components"]["cumulus"]["options"]
    cumulus_options["grell-freitas"] = {
        "asset_requirements": [],
        "constraints": {
            "required_settings": {"moist": True},
            # config.py enforces the same law: the trigger's excesses and
            # the shallow arm read KPBL and the PBL-maintained surface
            # fluxes, so a PBL scheme must be active.
            "requires_components": {
                "pbl": ["ysu", "mynn", "shinhong", "sase"]},
        },
        "extensions": {
            "arwen_pbl_structural_requirement": {
                "reason": (
                    "ArWen's GF adapter reads KPBL and the PBL-maintained "
                    "surface fluxes for the trigger's excesses and the "
                    "shallow arm; with bl_pbl_physics=0 nothing writes "
                    "them (WRF reads KPBL=0 there and indexes below the "
                    "column base, which ArWen refuses rather than "
                    "reproduces)"),
                "classification": (
                    "ArWen structural constraint; WRF v4.6.1 does not "
                    "prohibit cu_physics=3 with bl_pbl_physics=0"),
            },
        },
        "implemented": True,
        "label": "Grell-Freitas",
        "maturity": "implemented-unverified",
        "parameters": {
            "cudt_minutes": 0.0, "clos_choice": 0, "ishallow": 0},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"cu_physics": 3},
        "warnings": [
            "Grell-Freitas's distance from WRF v4.6.1 is MEASURED on both "
            "halves of the port and on the whole driver, not a scheme "
            "fragment. tools/gf_wrf461_oracle drives the byte-frozen "
            "module_cu_gf_wrfdrv.F/module_cu_gf_deep.F/module_cu_gf_sh.F "
            "at gfortran -O0 over 18 cases x 6 grid spacings x 2 ishallow "
            "arms (216 columns); the float32 CPU authority "
            "(gpuwm/verify/gf_driver.py) reproduces GFDRV word for word "
            "on the 208 columns where GFDRV's own decomposition is exact, "
            "and the CUDA translation unit (gpuwm/core/kernels/gf.cu) "
            "holds the same boundary with the gamma COMPUTED on the "
            "device: its transcribed glibc-2.39 "
            "tgammaf/lgammaf/expm1f/exp2f/powf are bitwise against 130k "
            "live-glibc words, so the fzu normalisation that moves the "
            "deep mass flux by up to 7.3 per cent per ULP needs no oracle "
            "pin on the GPU (tests/test_gf_deep_cuda.py, "
            "tests/test_gf_shallow_cuda.py, tests/test_gf_gfdrv_cuda.py). "
            "The 8 remaining columns are the driver's own "
            "module_gfs_physcons mixed precision, inherited and bounded "
            "(max 34 ULP, 3.8e-6 relative, no branch flips). That is "
            "conformance evidence, not scientific validation: no "
            "gpuwm/WRF forecast trajectory comparison exists for this "
            "scheme, which is why it is 'implemented-unverified' and not "
            "'supported'.",
            "DELIBERATE DIVERGENCE, owner ruling (no inherited WRF bugs): "
            "WRF's shallow k22 trigger is a MAXLOC over the array section "
            "heo_cup(2:kbmax) whose result module_cu_gf_sh.F uses as an "
            "absolute level index without adding the section offset, "
            "leaving k22 one level below the argmax wherever the argmax "
            "sits above level 2. The SHIPPED kernel uses the corrected "
            "indexing; the WRF-faithful off-by-one lives behind a launch "
            "flag only the parity suites set. Measured over the committed "
            "fixture: k22 moves on 3 of 18 cases (6, 13, 16), all three "
            "rejected under both modes with identical ierr, and ZERO "
            "output words differ at the scheme or driver boundary "
            "(tests/test_gf_shallow_cuda.py, the ledger test).",
            "DELIBERATE DIVERGENCE, WRF is undefined: "
            "get_inversion_layers' first-derivative loop reads "
            "t_cup(kend+8) past the array end whenever kend > ktf-8 "
            "(module_cu_gf_deep.F, both live call sites pass "
            "kend = kstabi). The port clamps kend to ktf-8 -- the oracle "
            "capture clamps identically -- and COUNTS the clamps; the "
            "count is zero on the whole committed fixture and the gates "
            "assert it stays zero.",
            "ENGINE SEAM, recorded deviations of the cu_physics=3 "
            "adapter (gpuwm/core/gf.py) -- the kernel behind it is "
            "bitwise; this is what the engine can hand it today, and it "
            "is the plumb-list for any label upgrade: (1) the advective "
            "and boundary-layer halves of GFDRV's forcing "
            "(RTHFTEN/RQVFTEN, RTHBLTEN/RQVBLTEN) are fed as zeros -- "
            "the dycore does not yet export an advective theta/qv pair "
            "and the PBL stack couples its rates before the driver "
            "retains them -- so the forced state carries radiative "
            "forcing only and, with ishallow=1, the shallow blqe closure "
            "member sees dhdt = 0; (2) GF's convective momentum "
            "tendencies are computed but not yet coupled (CumulusResult "
            "carries no momentum slots); (3) mass-level w is the "
            "KF-precedent average of the staggered field.",
        ],
    }
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
        "16-cell authority is phys/module_physics_init.F:3699-3704,"
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
        # "sase" is listed so the closure is selectable per domain on the
        # tree route.  No template selects it, so its reachability is
        # "component-override": a user asks for it explicitly or does not
        # get it, which is the right posture for an experimental scheme.
        # "shinhong" is listed so the closure is selectable per domain
        # here too.  Since the gray-zone template registered it, its
        # reachability is "template" -- a user can also ask for the whole
        # registered suite -- and this entry keeps the per-domain override
        # legal beside it.
        "pbl": ["off", "ysu", "mynn", "sase", "shinhong"],
        "surface_layer": ["revised-mm5", "classic-mm5", "mynn"],
        "radiation": [
            "off", "dudhia-shortwave", "rte-rrtmgp",
            "rte-rrtmgp-legacy-aggregate"],
    }
    # Grell-Freitas is selectable per domain on the tree route, the same
    # terms as shinhong/sase above; off and kain-fritsch are listed so the
    # per-domain override surface names the whole implemented cumulus set.
    tree_route["allowed_component_options"]["cumulus"] = [
        "off", "kain-fritsch", "grell-freitas"]
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
    # ``flag_qs_*_microphysics_selectors`` must partition EVERY implemented
    # microphysics selector -- tests/test_mynn_pbl.py asserts set equality
    # against the registry's own options -- so a new scheme lands here in the
    # same pass that registers it.  mp_physics=28's thompsonaero package
    # carries qs (Registry.EM_COMMON:3036), so F_QS is true for it.
    mynn["extensions"]["supplied_moisture_species"] = {
        "supplied": ["qv", "qc", "qi", "qs"],
        "withheld": ["qnc", "qni", "qnwfa", "qnifa", "qnbca", "o3"],
        "flag_qs_true_microphysics_selectors": [6, 8, 10, 18, 28],
        "flag_qs_false_microphysics_selectors": [0, 1],
        "wrf_flag_source": (
            "phys/module_pbl_driver.F:873-878 derives flag_qs from F_QS; "
            "Registry.EM_COMMON declares qs for mp_physics 6, 8, 10, 18 and "
            "28"),
        "gpuwm_runtime_source": (
            "gpuwm/core/mynn_pbl_runtime.py::MYNN_SNOW_MICROPHYSICS is the "
            "shipped set this list is checked against, selector by selector, "
            "by tests/test_physics_registry.py::"
            "test_the_registry_flag_qs_contract_is_the_one_the_shipped"
            "_runtime_applies. mp_physics=28 was published here before the "
            "runtime honoured it: gpuwm passed flag_qs=False for 28, so "
            "phys/module_bl_mynn.F:734/:876 substituted sqs = 0 and MYNN "
            "never saw snow under the one Thompson variant whose Registry "
            "package declares it. MEASURED on the committed WRF MYNN driver "
            "oracle's snow_anvil column (max sqs 4.08e-05): withholding snow "
            "drove qi_bl from 5.4863e-07 to exactly 0 and moved qc_bl, "
            "cldfra_bl, rqvblten, rthblten and exch_h with it. MEASURED "
            "again as a forecast -- mp_physics=28 + MYNN + SFCLAY + Noah, "
            "8x6x50 at dx = 3 km, 20 steps of dt = 12 s, two runs identical "
            "but for FLAG_QS, snow 4.0e-05 kg/kg seeded at 0-based levels "
            "20-33: max relative difference qr 2.037e-02, qv 7.226e-03, "
            "qc 4.035e-03, qke 5.593e-02, exch_h 8.380e-01, nc 6.915e-04, "
            "nwfa 1.142e-04, with qs and qi bitwise identical because "
            "neither WRF nor gpuwm applies a snow PBL tendency "
            "(phys/module_bl_mynn.F:1240-1242). The same experiment with the "
            "snow seeded in the WARM boundary layer instead is BITWISE "
            "identical, because MYNN's condensation only takes snow into the "
            "ice branch where the liquid fraction is below 1 -- the flag "
            "matters exactly where snow exists."),
        "wrf_live_consumer": (
            "phys/module_bl_mynn.F:1104-1106 passes real sqs to "
            "mym_condensation when FLAG_QS is true; mynn_tendencies still "
            "receives kzero at :1240-1242, matching WRF"),
        "withheld_aerosol_number_note": (
            "qnc/qnwfa/qnifa stay withheld under mp_physics=28, the first "
            "configuration where they carry real prognostic values. gpuwm "
            "passes flag_qnc/flag_qnwfa/flag_qnifa to MYNN as literal False "
            "(gpuwm/core/mynn_pbl.py), so MYNN never mixes them; WRF mixes "
            "them at bl_mynn_mixscalars=1 (phys/module_bl_mynn.F:4735,:4777,"
            ":4957), which gpuwm's MYNN option identity pins to 0. The "
            "withholding is structural rather than a namelist value; see the "
            "microphysics thompson-aerosol-mp28 warnings."),
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
    nssl2_legacy["maturity"] = "wrf-matched-run-candidate"
    nssl2_legacy["parameters"]["wrf_rrtmg_compatibility"] = (
        "wrf-rrtmg-4-4-legacy-v1")
    nssl2_legacy["parameters"]["ra_rrtmg_variant"] = "rrtmg_legacy"
    nssl2_legacy["warnings"] = [
        "Ratified fixed NSSL-2 plus exact WRF v4.6.1 legacy RRTMG profile; "
        "the NSSL-2 trajectory remains wrf-matched-run-candidate maturity."
    ]
    registry["templates"][nssl2_legacy_id] = nssl2_legacy
    for route in registry["runner_routes"].values():
        for declared in route.get("source_template_ids", {}).values():
            if nssl2_id in declared:
                if nssl2_legacy_id in declared:
                    declared.remove(nssl2_legacy_id)
                declared.insert(declared.index(nssl2_id) + 1, nssl2_legacy_id)

    # The observation battery's registered composition (obs-battery
    # integration wave, lead ruling 2026-08-04): Thompson + YSU + classic
    # MM5 + Noah + cumulus off + the exact WRF v4.6.1 legacy RRTMG.  Built
    # from the Thompson validation template on the nssl2_legacy idiom: the
    # radiation component is the resolved 4/4 pair ("rte-rrtmgp" is the
    # registry's spelling of that pair; the ENGINE is named by
    # ra_rrtmg_variant in parameters, exactly as the NSSL-2 legacy row
    # does).  Parameters and the single per-domain row are TRANSCRIBED
    # from configs/battery/shape_3km_thompson_rrtmg_legacy.toml as
    # registered -- notably radt 12.0 at dx 3000 m, where the KF template
    # family's ladder carries radt 3.0 at 3 km.  That divergence is
    # deliberate: this row names what the battery runs, not the ladder.
    thompson_validation_id = "thompson-mp8-ysu-mm5-noah-validation-v1"
    thompson_legacy_id = "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1"
    thompson_legacy = copy.deepcopy(
        registry["templates"][thompson_validation_id])
    thompson_legacy["components"]["radiation"] = "rte-rrtmgp"
    thompson_legacy["label"] = (
        "Thompson + YSU + classic MM5 + Noah + cumulus off + legacy RRTMG")
    thompson_legacy["maturity"] = "wrf-matched-run-candidate"
    thompson_legacy["parameters"]["diff_6th_factor"] = 0.12
    thompson_legacy["parameters"]["radt"] = 12.0
    thompson_legacy["parameters"]["wrf_rrtmg_compatibility"] = (
        "wrf-rrtmg-4-4-legacy-v1")
    thompson_legacy["parameters"]["ra_rrtmg_variant"] = "rrtmg_legacy"
    thompson_legacy["per_domain_overrides"] = [
        {
            "diff_6th_factor": 0.12,
            "epssm": 0.5,
            "nominal_dx_m": 3000.0,
            "radt": 12.0,
        },
    ]
    thompson_legacy["warnings"] = [
        "Composition candidate: every component is individually verified "
        "(Thompson mp8 wrf-matched-run; the legacy RRTMG engine is the "
        "certified WRF v4.6.1 port) but no receipt covers the composed "
        "suite.  The upgrade payer is named: the composition's first "
        "stock-WRF-paired t0/case receipt (the observation battery's "
        "shakedown case) is what moves this label.",
        "radt 12.0 at dx 3000 m is transcribed from the battery's "
        "registered configuration and deliberately diverges from the KF "
        "template family's per-domain ladder (radt 3.0 at 3 km); it is "
        "not an oversight.",
    ]
    registry["templates"][thompson_legacy_id] = thompson_legacy
    # HRRR-only registration, on the Kessler precedent: the battery runs
    # HRRR, and no other source inherits evidence from that run.  The
    # prepared-single-domain route's per-source lists are the runner's
    # own VERIFICATION-EVIDENCE metadata (its drift check compares them
    # to _SOURCE_PHYSICS_PROFILES), and this composition has no receipt
    # on gfs/era5/20crv3 -- so it is deliberately absent there.
    for route in registry["runner_routes"].values():
        for declared in route.get("source_template_ids", {}).values():
            if thompson_legacy_id in declared:
                declared.remove(thompson_legacy_id)
    for route_id in (
            "tools.hrrr_single_domain_benchmark",
            "tools.prepared_domain_tree_forecast"):
        declared = registry["runner_routes"][route_id][
            "source_template_ids"]["hrrr"]
        declared.insert(
            declared.index(thompson_validation_id) + 1, thompson_legacy_id)

    # The gray-zone sibling of the row above: the SAME composition with
    # Shin-Hong 2015 in place of YSU, which is the single edge the
    # divergence ledger's L3 entry moves (gpuwm/physics_mode.py).  It is
    # registered because a physics-fidelity arm that selects L3 resolves
    # to exactly this suite, and an unregistered suite has no root
    # preparation -- so the ledger entry that already carries the
    # strongest scheme-level evidence in the tree had no run route at
    # all.  Built from the sibling by moving ONE component, so a paired
    # run of the two isolates the closure and nothing else; the surface
    # layer stays classic MM5 because WRF v4.6.1's own SHINHONGSCHEME arm
    # (phys/module_physics_init.F:3702-3704) requires isfc=1 exactly as
    # YSU does.
    shinhong_legacy_id = "thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1"
    shinhong_legacy = copy.deepcopy(thompson_legacy)
    shinhong_legacy["components"]["pbl"] = "shinhong"
    shinhong_legacy["label"] = (
        "Thompson + Shin-Hong + classic MM5 + Noah + cumulus off + "
        "legacy RRTMG")
    shinhong_legacy["maturity"] = "wrf-matched-run-candidate"
    shinhong_legacy["warnings"] = [
        "Composition candidate: every component is individually verified "
        "(Thompson mp8 wrf-matched-run; the legacy RRTMG engine is the "
        "certified WRF v4.6.1 port; Shin-Hong is measured bitwise against "
        "the byte-frozen WRF v4.6.1 module on both halves of the port, "
        "max ULP 0 on the float32 CPU authority and 0 ULP on the CUDA "
        "heat tendency) but no receipt covers the composed suite.  The "
        "upgrade payer is named: this composition's first stock-WRF-"
        "paired t0/case receipt -- the first paired case run of the "
        "gray-zone arm -- is what moves this label.",
        "This template differs from "
        "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1 in exactly ONE "
        "component, the PBL closure, so the pair is a controlled "
        "gray-zone comparison rather than two independent suites.  Every "
        "other parameter, including the per-domain row, is transcribed "
        "from that template.",
        "Shin-Hong carries the component-level warnings of its option "
        "(components.pbl.options.shinhong): the entrainment-flux guard "
        "where WRF reads one element past its array, WRF's own 0/0 NaN "
        "column reproduced rather than repaired, and the sm_120 "
        "subnormal-tendency flush.  Selecting this template selects "
        "those.",
    ]
    registry["templates"][shinhong_legacy_id] = shinhong_legacy
    # HRRR-only, on the same Kessler rule as its sibling: the arm that
    # selects this composition runs the HRRR route, and no other source
    # inherits evidence from that run.
    for route in registry["runner_routes"].values():
        for declared in route.get("source_template_ids", {}).values():
            if shinhong_legacy_id in declared:
                declared.remove(shinhong_legacy_id)
    for route_id in (
            "tools.hrrr_single_domain_benchmark",
            "tools.prepared_domain_tree_forecast"):
        declared = registry["runner_routes"][route_id][
            "source_template_ids"]["hrrr"]
        declared.insert(
            declared.index(thompson_legacy_id) + 1, shinhong_legacy_id)

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
    # Last, so both passes see every surface this builder created above --
    # including the legacy NSSL-2 template and the regenerated nest edges.
    _rename_maturities(registry)
    _evidence_architecture(registry)
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
