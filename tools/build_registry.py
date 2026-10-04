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

from gpuwm.config import (  # noqa: E402
    GF_CLOSURE_MEMBERS,
    MOSAIC_URBAN_CANOPY_DEFAULT,
    MOSAIC_URBAN_CANOPY_RULES,
    MP28_AEROSOL_SOURCES as MP28_AEROSOL_SOURCES,
)
from gpuwm.physics_registry import (  # noqa: E402
    TEMPLATE_ID_ALIASES, canonical_json, component_override_declaration,
)
from gpuwm import physics_compat  # noqa: E402

#: The value ``mp28_aerosol_source`` takes when nobody chooses, which is
#: the one value a non-mp=28 option may carry: the enumeration hands every
#: plan the registry's own default, and refusing that would refuse every
#: plan that never mentioned the key.
MP28_AEROSOL_SOURCE_DEFAULT_VALUE = "auto"
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
# Imported INSIDE _source() below rather than here: tools.check_parameter_claims
# reaches gpuwm.config and, through it, every consumer module whose
# import-time agreement check compares against the registry on disk -- the
# registry this builder is about to replace.  main() sets the rebuild flag
# before any of those imports happen; a module-scope import here would run
# them first and refuse the rebuild that fixes a stale registry.

# ---------------------------------------------------------------- implemented
# type / enum / minimum / default taken from gpuwm's own accepted sets
# (gpuwm/config.py validate_run_config), not from WRF's wider sets.
IMPLEMENTED: dict[str, dict] = {
    "bl_mynn_mixlength": {
        "type": "integer", "enum": [1, 2], "default": 1,
        "component_id": "pbl", "read_when": {"bl_pbl_physics": 5}},
    "scalar_pblmix": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "component_id": "pbl", "read_when": {"bl_pbl_physics": 5}},
    # acoustic / small step
    "time_step_sound": {"type": "integer", "minimum": 2, "default": 4},
    "smdiv": {"type": "number", "minimum": 0.0, "default": 0.1},
    "emdiv": {"type": "number", "minimum": 0.0, "default": 0.0},
    # explicit / constant-K mixing
    "khdif": {"type": "number", "minimum": 0.0, "default": 0.0},
    "kvdif": {"type": "number", "minimum": 0.0, "default": 0.0},
    "c_s": {"type": "number", "minimum": 0.0, "default": 0.25},
    "diff_opt": {"type": "integer", "enum": [1, 2], "default": 2,
                 "description": "1: coordinate-surface diffusion for km_opt=2/4; 2: metric stress/scalar diffusion."},
    "mix_full_fields": {"type": "boolean", "default": True,
                        "description": "WRF logical retained under coordinate diffusion; that operator always mixes theta relative to its initial field."},
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
    # WRF 4.3+ implicit-explicit vertical advection (gpuwm.core.ieva), off
    # by default.  Its warning is the declared divergence of its w solve
    # (A179), which the option's own documentation also records.
    "zadvect_implicit": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "warnings": [
            "For zadvect_implicit_variant='wrf_471', DIVERGENCE from WRF "
            "v4.7.1, declared: the implicit w solve's "
            "two boundary terms take the units of the w terms beside them. "
            "WRF's lower boundary (dyn_em/module_ieva_em.F:1231-1244) builds "
            "the surface w increment from the mass-coupled u/v tendencies, "
            "about one column mass too large, and a steep-ridge run went "
            "NaN in three steps; gpuwm uncouples each tendency by its map "
            "factor and face column mass first. WRF's upper boundary "
            "(:1248-1253) leaves the geopotential change over dt undivided "
            "by g; gpuwm divides it. Every other IEVA routine matches WRF "
            "v4.7.1 word for word, and the w solve matches WRF's routine "
            "with those two corrections (tools/ieva_wrf_oracle)."]},
    "zadvect_implicit_variant": {
        "type": "string", "enum": ["wrf_471", "wrf_legacy"],
        "default": "wrf_471",
        "warnings": [
            "Selects the WRF numerical generation when zadvect_implicit=1. "
            "wrf_471 preserves module_ieva_em. wrf_legacy uses "
            "module_advect_em's current-mass coefficients and one-sided "
            "horizontal Courant allowance. Both uncouple the lower w "
            "boundary's momentum tendencies, a declared unit correction."]},
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
    # WRF v4.7.1 slope-dependent surface shortwave and terrain shadowing
    # (gpuwm.core.topo_radiation, lane 281-namelist-gaps), off by default.
    "slope_rad": {"type": "integer", "enum": [0, 1], "default": 0},
    "topo_shading": {"type": "integer", "enum": [0, 1], "default": 0},
    # WRF v4.7.1 sub-grid terrain drag (gpuwm.core.terrain_drag, lane
    # 282-terrain-drag), off by default: topo_wind under YSU, gwd_opt under
    # every PBL scheme.
    "topo_wind": {"type": "integer", "enum": [0, 1, 2], "default": 0},
    "gwd_opt": {"type": "integer", "enum": [0, 1, 3], "default": 0},
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
            "experimental; the native gpuwm run multi-domain executor and "
            "the prepared-tree executor, the native HRRR route's included, "
            "both run it."]},
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
    "sf_lake_physics": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "per_domain": True, "component_id": "land_surface",
        "warnings": ["Option 1 runs WRF's CLM lake column on LAKEMASK "
                     "cells, with ten water and ten sediment layers and "
                     "up to five snow layers. Default 0 retains prescribed "
                     "water temperatures."]},
    "use_lakedepth": {
        "type": "integer", "enum": [0, 1], "default": 1,
        "per_domain": True, "component_id": "land_surface",
        "read_when": {"sf_lake_physics": 1},
        "warnings": ["WRF default 1 requires input LAKE_DEPTH. Explicit 0 "
                     "uses lakedepth_default for every lake column."]},
    "lakedepth_default": {
        "type": "number", "default": 50.0,
        "per_domain": True, "component_id": "land_surface",
        "read_when": {"sf_lake_physics": 1},
        "warnings": ["Depth in metres used by WRF lakeini when a supplied "
                     "depth is nonpositive or use_lakedepth=0. A nonpositive "
                     "default uses WRF's reference layer geometry depth."]},
    "lake_min_elev": {
        "type": "number", "default": 5.0,
        "per_domain": True, "component_id": "land_surface",
        "read_when": {"sf_lake_physics": 1},
        "warnings": ["Minimum water-cell elevation in metres used by WRF "
                     "when the input does not provide a lake mask."]},
    "mosaic_lu": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "component_id": "land_surface", "read_when": {"sf_surface_physics": 3},
        "warnings": ["RUC weighted vegetation parameters and irrigation use "
                     "LANDUSEF. WRF's default is 0; 1 requires category fractions."]},
    "mosaic_soil": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "component_id": "land_surface", "read_when": {"sf_surface_physics": 3},
        "warnings": ["RUC weighted soil parameters use SOILCTOP. WRF's default "
                     "is 0; 1 requires category fractions."]},
    "ruc_soilprop": {
        "type": "string", "enum": ["wrf_45", "wrf_461"], "default": "wrf_45",
        "component_id": "land_surface", "read_when": {"sf_surface_physics": 3},
        "warnings": ["Selects which WRF lineage's LSMRUC SOILPROP sets soil-water "
                     "diffusivity and hydraulic conductivity. wrf_45 (WRF v4.0-4.5, also "
                     "the operational RAP/HRRR branch) normalises both by the moisture "
                     "above the residual, (theta - qmin)/(theta_sat - qmin), and uses "
                     "mineral conductivity 2.0 at every quartz fraction. wrf_461 (WRF "
                     "v4.6.1) uses total moisture over porosity and 3.0 below 20 percent "
                     "quartz; in dry soil its water diffusivity is 2.5 to 8 times "
                     "larger, measured to raise a 3 km afternoon top soil level from "
                     "0.161 to 0.187 m3/m3 in one hour from the levels below, where the "
                     "operational model's own top level fell to 0.157. The default "
                     "changed from the v4.6.1 form to wrf_45: every RUC configuration "
                     "changes answers."]},
    "sf_surface_mosaic": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "warnings": ["Noah only: WRF v4.7.1 lsm_mosaic tile state. "
                     "Stock-WRF export of mosaic land state and moving-nest "
                     "tile interpolation are not reproduced. Urban option 1 "
                     "runs inside the tile loop, by default only where a "
                     "cell's dominant category is urban (mosaic_urban_canopy); "
                     "mosaic option cannot work with urban options 2 and 3 "
                     "(WRF module_check_a_mundo.F:505-518)."]},
    # read_when: the settings a knob is read under at all
    # (gpuwm.physics_registry.registry_knob_is_read), so a WRF namelist or a
    # TOML that carries a tile count with mosaic off runs what a 2.8.0
    # preparation ran and is not refused as new physics (A153).
    "mosaic_cat": {"type": "integer", "minimum": 1, "default": 3,
                   "component_id": "land_surface",
                   "read_when": {"sf_surface_mosaic": 1}},
    "mosaic_urban_canopy": {
        "type": "string", "enum": list(MOSAIC_URBAN_CANOPY_RULES),
        "default": MOSAIC_URBAN_CANOPY_DEFAULT,
        "read_when": {"sf_surface_mosaic": 1, "sf_urban_physics": 1},
        "warnings": ["Read only with sf_surface_mosaic = 1 and "
                     "sf_urban_physics = 1. dominant is WRF v4.7.1: the urban "
                     "canopy runs only where a cell's dominant category is "
                     "urban. every_tile also runs the urban tiles of a mostly "
                     "rural cell at their own land-use weights, sharing the "
                     "largest urban tile type's URBPARM urban fraction, a "
                     "gpuwm option WRF does not have; the 10 m wind override "
                     "stays on dominant-urban cells."]},
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
    # registry-accuracy item (8.1.5) and AC-P6.4.
    "diff_6th_opt": {"type": "integer", "enum": [0, 1, 2], "default": 0},
    "diff_6th_slopeopt": {"type": "integer", "enum": [0, 1], "default": 0},
    # AN ENUM, not minimum 1.  ``minimum: 1`` advertised 1, 2, 3, 5, 6,
    # 7, 8 as type-legal, and gpuwm/config.py:376 accepts none of them -- the
    # declaration was looser than the code it declares.  The enum is the set
    # of soil-layer counts this registry's own land-surface options carry,
    # read from those options' schemes and assigned near the end of
    # ``build`` (the merge below deep-copies this spec over the parameters
    # dict, so the literal here is only a placeholder).
    #
    # It is NOT the single count each scheme defines.  Every count any
    # scheme defines has to stay type-legal for the COMPONENT layer to be
    # the thing that speaks: tests/test_physics_registry.py requires a
    # domain asking for num_soil_layers=9 under NOAH to raise
    # component-admitted-setting, and a parameter-value rejection would
    # pre-empt that -- the value never reaches ``settings``, so the
    # option's own admitted set never sees it.
    #
    # The warning was written when RUC was refused and said three things that
    # are no longer true: that only 4 is accepted, that no physics component
    # reads the knob, and that the nine-layer identity was unported.  RUC is
    # admitted at nine layers and runs a forecast, so the text now describes
    # the resolver that actually decides.
    #
    # AND, since lane/ruc-column-nzs admitted six: this warning is the text a
    # user reads to decide what they may set, so a stale sentence here is a
    # capability the product denies having.  Two clauses were left behind by
    # that lane's rewrite and contradicted the sentence after them --
    # 'RUC resolves 9' (gpuwm/config.py soil_layer_count returns the
    # requested 6 when RUC is asked for 6; measured) and 'Both 4 and 9 are
    # selectable ... and no other value is' (six is selectable, and the very
    # next sentence says how).  Both are corrected.
    #
    # A THIRD clause survived that pass and was caught in review: 'This ENUM
    # declares 4 and 9 ... the enum is a statement about EVIDENCE, not the
    # whole set of selectable values', sitting in the MIDDLE of the same
    # warning string whose tail had just been rewritten to say six is
    # selectable, and contradicting the enum this module now derives.  One
    # user-visible string is one statement: it is corrected here, and the
    # evidence claim it was carrying is made by the maturity paragraph
    # further down, which is where it can be read without denying what the
    # field admits.
    #
    # THE ENUM NOW CARRIES SIX, and the enum is not where evidence is
    # stated.  Holding it at [4, 9] made the plan-review door NARROWER than
    # the run door: a six-level RUC plan was refused here with "must be one
    # of [4, 9]" -- a refusal naming no breakage and no way out -- while
    # gpuwm/config.py admitted it, the kernel sized itself for it and a
    # forecast completed on it.  That is the second-table drift this audit
    # exists for, pointed the other way.  The enum states which counts are
    # SELECTABLE and is derived below from the land-surface schemes' own
    # tables, so a scheme that gains a geometry gains the enum entry in the
    # same edit; the EVIDENCE claim (six is internal-consistency-only) stays
    # exactly where a user reads it, in this warning and in the runtime
    # receipt, which is what a warning is for.
    "num_soil_layers": {
        # Placeholder only: the enum is DERIVED from the land-surface
        # schemes' own tables and reassigned after the parameter merge
        # below.  Held equal to what that derivation currently produces so
        # a reader of this module is never told a different set than the
        # artifact carries.
        "type": "integer", "enum": [4, 6, 9], "default": 4,
        "warnings": [
            "The resolved layer count comes from the SCHEME, not from this "
            "knob: gpuwm/config.py soil_layer_count consults "
            "LAND_SURFACE_SOIL_LAYERS for the selected sf_surface_physics, so "
            "Noah and Noah-MP resolve 4 and RUC resolves the 6 or 9 it was "
            "asked for, and every soil allocation, VRAM count, output dimension "
            "and restart shape reads that resolver. DIVERGENCE from WRF, "
            "deliberate: set_physics_rconfigs OVERWRITES a namelist request "
            "that disagrees with the scheme and only logs it at debug level, so "
            "a namelist asking Noah for nine layers runs on four with no error; "
            "gpuwm refuses instead. This ENUM declares every count a "
            "registered land-surface scheme defines -- 4 with Noah or "
            "Noah-MP, 6 or 9 with RUC -- and is derived from those schemes' "
            "own tables; it states what is SELECTABLE, and the separate "
            "question of what an oracle has judged is answered by this "
            "warning and by the run receipt, never by the enum. WRF's "
            "six-level RUC grid "
            "(share/module_soil_pre.F:init_soil_depth_3) is one of those "
            "counts, and it COMPILES and RUNS: gpuwm/core/kernels/ruc.cu "
            "sizes every "
            "soil scratch from RUC_NZS and selects its level table with it, "
            "gpuwm.core.ruc/.ruc_gpu resolve the count from the profile, and a "
            "1-hour HRRR-initialised forecast completes at six levels with "
            "soil_layers_stag=6 in its wrfout. What six LACKS is a WRF FORECAST "
            "ORACLE -- every lsmruc/sfctmp/soilmoist/snowtemp fixture in the "
            "tree is nine-level -- so it is a MATURITY statement, not an "
            "admission question: it is selected through the hash-bound "
            "experiment config (sf_surface_physics = 3 with num_soil_layers = "
            "6) rather than a named --physics-profile, warns at runtime, and "
            "carries soil_geometry_evidence=internal-consistency-only in its "
            "run receipt. See docs/wrf_ruc_runtime_admission.md."]},
    "terrain_opt": {"type": "integer", "enum": [0, 1], "default": 0},
    # AUDIT R-060.  These three were roadmap rows that had outlived their
    # gap, and the gate that keeps such a row true was one-directional --
    # it proved an implemented row's citation and FORBADE an unimplemented
    # row from carrying one, so a lane that landed the read could not move
    # the row even if it wanted to.  tools/check_parameter_claims.py now
    # proves the negative as well.
    #
    # smooth_option: gpuwm/core/nest_interp.py smoother() dispatches WRF's
    # sm121 and smdsm through the nest_smooth_i/nest_smooth_j kernels, and
    # gpuwm/core/model.py wires ExperimentConfig.smooth_option into it.
    "smooth_option": {"type": "integer", "enum": [0, 1, 2], "default": 0,
                      "warnings": [
                          "0 is no smoothing, 1 is WRF's sm121 and 2 is "
                          "smdsm. Prepared hierarchy artifacts are the "
                          "same at every value: this smooths the parent "
                          "after feedback while the tree runs and does "
                          "nothing to a prepared child."]},
    # wif_input_opt / aer_init_opt: gpuwm/ingest/wif_climatology.py IS the
    # WIF ingest, gpuwm/ingest/real.py branches on the (aer_init_opt,
    # wif_input_opt) == (1, 1) pair, and gpuwm/config.py's own error text
    # instructs the user to "use aer_init_opt=1 with wif_input_opt=1" --
    # a pair the registry made unspellable, since one row refused it and
    # the other did not exist.
    "wif_input_opt": {"type": "integer", "enum": [0, 1], "default": 0},
    "aer_init_opt": {"type": "integer", "enum": [0, 1, 2], "default": 0,
                     "compatible_previous_enums": [[0, 1]]},
    "use_rap_aero_icbc": {
        "type": "boolean", "default": False,
        "component_id": "microphysics",
        "consuming_read": "gpuwm/ingest/real.py",
        "description": "Analyzed QNWFA/QNIFA initial and lateral values with operational monthly surface emissions; mp_physics=28 only.",
    },
    "epssm": {"type": "number", "minimum": 0.0, "maximum": 1.0,
              "default": 0.1},
    "diff_6th_factor": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                        "default": 0.12},
    "moist": {"type": "boolean", "default": False},
    "moist_cq": {"type": "boolean", "default": True},
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
    # mp=28's aerosol source (gpuwm.config.MP28_AEROSOL_SOURCES).  It is
    # REGISTERED, not just a RunConfig field, because the refusal it
    # answers is now a registry refusal too: audit R-044's
    # consumers.lateral_forcing_dataset row names 'synthetic' as the
    # deliberate way out, and a way out a plan cannot express is not a way
    # out.  'auto' resolves WRF's monthly WIF climatology and announces the
    # synthetic fallback by name; 'climatology' refuses rather than
    # degrading; 'synthetic' selects thompson_init's profile on purpose.
    "mp28_aerosol_source": {
        "type": "string",
        "enum": list(MP28_AEROSOL_SOURCES),
        "compatible_previous_enums": [["auto", "climatology", "synthetic"]],
        "default": "auto",
        "consuming_read": "gpuwm/ingest/real.py",
    },
    # WRF v4.6.1's NSSL variant selectors.  There is one NSSL scheme
    # (mp_physics=18) and these four flags on top of it; the deprecated
    # IDs 17/19/21/22 are spellings share/module_check_a_mundo.F
    # rewrites onto exactly this set.  gpuwm/core/nssl2_contract.py
    # resolves them and refuses the combinations with no ported path.
    "nssl_2moment_on": {
        "type": "integer",
        "enum": [-1, 1],
        "default": -1,
        "note": (
        "WRF's nssl_2moment_on (Registry.EM_COMMON:2423). -1 takes WRF's "
        "consistency-pass default of 1 (module_check_a_mundo.F:3437-3439). 0 "
        "selects the one-moment NSSL family (the deprecated mp_physics=19/21 "
        "spellings), which is NOT ported and is refused at validation rather "
        "than substituted."),
        "evidence": (
        "Column smoke over the four ported variant modes on the shipped "
        "run_nssl2_production_step seam (tools/nssl2_variant_probe.py, "
        "evidence/nssl2-variants/variant-column-smoke.json), in two regimes: "
        "the seeded moist column behind the mp18 digest baseline, and a "
        "riming-updraft regime built to put the graupel-to-hail conversion "
        "in range. Each variant is compared against the DEFAULT mode run on "
        "THE SAME inputs, so a delta is a treatment and not an input "
        "difference. Asserted: every output array finite; the fields a "
        "variant's Registry packages would not allocate stay exactly 0.0; "
        "the CCN field comes back byte-identical to its input when the "
        "variant does not predict it; and every variant moves at least one "
        "field against its control (the hail arms move 29 and 30 fields in "
        "the riming regime and legitimately move none in the moist column, "
        "where the conversion never fires). The shipped mp18 default lane is "
        "unchanged by all of this: all 30 digests in "
        "evidence/nssl2-variants/mp18-digest-baseline.json reproduce "
        "byte-for-byte. NO ORACLE COMPARISON AGAINST WRF FORTRAN EXISTS FOR "
        "ANY VARIANT PATH -- no WRF run, no matched trajectory, no ULP "
        "measurement. The oracle campaign is the declared next stage, as it "
        "was for Shin-Hong and Grell-Freitas."),
    },
    "nssl_ccn_on": {
        "type": "integer",
        "enum": [-1, 0, 1],
        "default": -1,
        "note": (
        "WRF's nssl_ccn_on (Registry.EM_COMMON:2421). -1 resolves to 1 "
        "(module_check_a_mundo.F:3433-3435). 0 is the deprecated "
        "mp_physics=17/22 semantics: qnn is not allocated, the unactivated "
        "CCN is diagnosed from the base concentration every step "
        "(module_mp_nssl_2mom.F:2734), never stored back (:3283), and "
        "renucfrac rises to 1.0 (:2555-2557), which changes the nucleation "
        "pool at :10116 and arms the low-temperature limiter at "
        ":10120-10127."),
        "evidence": (
        "Column smoke over the four ported variant modes on the shipped "
        "run_nssl2_production_step seam (tools/nssl2_variant_probe.py, "
        "evidence/nssl2-variants/variant-column-smoke.json), in two regimes: "
        "the seeded moist column behind the mp18 digest baseline, and a "
        "riming-updraft regime built to put the graupel-to-hail conversion "
        "in range. Each variant is compared against the DEFAULT mode run on "
        "THE SAME inputs, so a delta is a treatment and not an input "
        "difference. Asserted: every output array finite; the fields a "
        "variant's Registry packages would not allocate stay exactly 0.0; "
        "the CCN field comes back byte-identical to its input when the "
        "variant does not predict it; and every variant moves at least one "
        "field against its control (the hail arms move 29 and 30 fields in "
        "the riming regime and legitimately move none in the moist column, "
        "where the conversion never fires). The shipped mp18 default lane is "
        "unchanged by all of this: all 30 digests in "
        "evidence/nssl2-variants/mp18-digest-baseline.json reproduce "
        "byte-for-byte. NO ORACLE COMPARISON AGAINST WRF FORTRAN EXISTS FOR "
        "ANY VARIANT PATH -- no WRF run, no matched trajectory, no ULP "
        "measurement. The oracle campaign is the declared next stage, as it "
        "was for Shin-Hong and Grell-Freitas."),
    },
    "nssl_density_on": {
        "type": "integer",
        "enum": [-1, 1, 2],
        "default": -1,
        "note": (
        "WRF's nssl_density_on (Registry.EM_COMMON:2425). -1 resolves to 2 "
        "with hail on and 1 with hail off "
        "(module_check_a_mundo.F:3449-3455). 0 (fixed graupel/hail density) "
        "is not ported. 1 with hail on is refused: the module sets lvhl>0 at "
        "module_mp_nssl_2mom.F:1674-1679 and then reads and writes the qvolh "
        "field that only the nssl_hailvol package (nssl_density_on=2) "
        "allocates."),
        "evidence": (
        "Column smoke over the four ported variant modes on the shipped "
        "run_nssl2_production_step seam (tools/nssl2_variant_probe.py, "
        "evidence/nssl2-variants/variant-column-smoke.json), in two regimes: "
        "the seeded moist column behind the mp18 digest baseline, and a "
        "riming-updraft regime built to put the graupel-to-hail conversion "
        "in range. Each variant is compared against the DEFAULT mode run on "
        "THE SAME inputs, so a delta is a treatment and not an input "
        "difference. Asserted: every output array finite; the fields a "
        "variant's Registry packages would not allocate stay exactly 0.0; "
        "the CCN field comes back byte-identical to its input when the "
        "variant does not predict it; and every variant moves at least one "
        "field against its control (the hail arms move 29 and 30 fields in "
        "the riming regime and legitimately move none in the moist column, "
        "where the conversion never fires). The shipped mp18 default lane is "
        "unchanged by all of this: all 30 digests in "
        "evidence/nssl2-variants/mp18-digest-baseline.json reproduce "
        "byte-for-byte. NO ORACLE COMPARISON AGAINST WRF FORTRAN EXISTS FOR "
        "ANY VARIANT PATH -- no WRF run, no matched trajectory, no ULP "
        "measurement. The oracle campaign is the declared next stage, as it "
        "was for Shin-Hong and Grell-Freitas."),
    },
    "nssl_hail_on": {
        "type": "integer",
        "enum": [-1, 0, 1],
        "default": -1,
        "note": (
        "WRF's nssl_hail_on (Registry.EM_COMMON:2420). -1 resolves to 1 "
        "under two moments (module_check_a_mundo.F:3441-3447). 0 drops the "
        "hail category: lhl=0 at module_mp_nssl_2mom.F:1445-1447 and the "
        "graupel-hail conversion block at :19860 never runs. 2 (one-moment "
        "hail) is refused: WRF would read the hail-number field the "
        "nssl_hail1m package never allocated."),
        "evidence": (
        "Column smoke over the four ported variant modes on the shipped "
        "run_nssl2_production_step seam (tools/nssl2_variant_probe.py, "
        "evidence/nssl2-variants/variant-column-smoke.json), in two regimes: "
        "the seeded moist column behind the mp18 digest baseline, and a "
        "riming-updraft regime built to put the graupel-to-hail conversion "
        "in range. Each variant is compared against the DEFAULT mode run on "
        "THE SAME inputs, so a delta is a treatment and not an input "
        "difference. Asserted: every output array finite; the fields a "
        "variant's Registry packages would not allocate stay exactly 0.0; "
        "the CCN field comes back byte-identical to its input when the "
        "variant does not predict it; and every variant moves at least one "
        "field against its control (the hail arms move 29 and 30 fields in "
        "the riming regime and legitimately move none in the moist column, "
        "where the conversion never fires). The shipped mp18 default lane is "
        "unchanged by all of this: all 30 digests in "
        "evidence/nssl2-variants/mp18-digest-baseline.json reproduce "
        "byte-for-byte. NO ORACLE COMPARISON AGAINST WRF FORTRAN EXISTS FOR "
        "ANY VARIANT PATH -- no WRF run, no matched trajectory, no ULP "
        "measurement. The oracle campaign is the declared next stage, as it "
        "was for Shin-Hong and Grell-Freitas."),
    },
    "icloud": {"type": "integer", "enum": [0, 1], "default": 1},
    "radt": {"type": "number", "minimum": 0.0, "default": 0.0},
    "radt_minutes": {"type": "number", "minimum": 0.0, "default": 12.0},
    "bldt": {"type": "number", "minimum": 0.0, "default": 0.0},
    "cudt_minutes": {"type": "number", "minimum": 0.0, "default": 5.0},
    # Grell-family keys, WRF v4.6.1 Registry defaults
    # (Registry.EM_COMMON:2544,2546); read only where cu_physics = 3.
    # 0 is the ensemble mean, 1..16 one closure member alone; anything
    # else has no meaning in cup_forcing_ens_3d (gpuwm.config
    # gf_clos_choice_refusal names the breakage).
    "clos_choice": {
        "type": "integer",
        "enum": list(range(GF_CLOSURE_MEMBERS + 1)), "default": 0,
        "warnings": [
            "0 (the Registry default) is the 16-member ensemble mean, "
            "compared bitwise against WRF v4.6.1. 1..16 run one closure "
            "member of cup_forcing_ens_3d alone: WRF's own code path, "
            "implemented but not verified against a WRF run."]},
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
        "stream. The WRF namelist importer accepts the monthly WIF dataset's "
        "30 levels with use_aero_icbc=true and wif_input_opt=1; the native "
        "configuration does not expose an independent level-count override."),
    "nssl_3moment": (
        "c",
        "The NSSL three-moment extension (nssl_3moment=1, which WRF rewrites "
        "to 2 with hail on at module_check_a_mundo.F:3457-3463) adds the "
        "qzr/qzg/qzh reflectivity moments and the ipconc>=6 branch "
        "throughout the scheme. The ported family is the two-moment ipconc=5 "
        "branch only; gpuwm/core/nssl2_contract.py refuses the combination "
        "at validation."),
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
    "nssl_ehlw0": (
        "c",
        "Not a WRF v4.6.1 namelist option. NSSL_EHLW0 is a scheme-internal "
        "collection-efficiency constant and has no legal user setting."),
    "nssl_ehw0": (
        "c",
        "Not a WRF v4.6.1 namelist option. NSSL_EHW0 is a scheme-internal "
        "collection-efficiency constant and has no legal user setting."),
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
    "ua_phys": (
        "c",
        "Noah unified-atmosphere coupling is a separate physics path with "
        "additional state and feedback semantics; gpuwm ports the ordinary "
        "Noah LSM driver only."),
    "use_aero_icbc": (
        "c",
        "For components.microphysics.options.thompson-aerosol-mp28, this "
        "namelist-only key imports through the aer_init_opt=1 / "
        "wif_input_opt=1 pair rather than a RunConfig field of its own. "
        "Monthly WIF and analyzed QNWFA/QNIFA initial and lateral carriers "
        "are implemented. An unrestricted GOCART species reader is not."),
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
                from tools.check_parameter_claims import (
                    UnparseableCitation, _code_without_prose)
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
# (1000/3 m, published as the nominal 333 m) and the wrf-matched-run-candidate
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
    "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-wrf-comparison-candidate-v1": [
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
        "Mosaic land-use and soil accept 0/1, default 0, and read LANDUSEF "
        "and SOILCTOP. spp_lsm accepts 0/1 and needs a member-owned SPP "
        "pattern; flag_sm_adj remains at 0. "
        "Also pinned rather than configurable: "
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
    for key in ("mosaic_lu", "mosaic_soil"):
        ruc_option["constraints"]["required_settings"].pop(key, None)
        ruc_option["constraints"].setdefault("admitted_setting_values", {})[key] = [0, 1]
        ruc_option["constraints"].setdefault("admitted_setting_values_reasons", {})[key] = (
            "WRF RUC SOILVEGIN selects dominant parameters at 0 and weighted "
            "category parameters at 1; other values select neither branch")
    ruc_option["constraints"]["required_settings"].pop("ruc_soilprop", None)
    ruc_option["constraints"]["admitted_setting_values"]["ruc_soilprop"] = ["wrf_45", "wrf_461"]
    ruc_option["constraints"]["admitted_setting_values_reasons"]["ruc_soilprop"] = (
        "wrf_45 is the WRF v4.0-4.5 soil-water diffusivity over the moisture "
        "above the residual and wrf_461 the WRF v4.6.1 form over total "
        "porosity; the two move different water between soil levels, so no "
        "other name can select either")
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
        "spp_lsm=1 consumes a member-owned stochastic pattern in RUC's "
        "historical WRF 3.9.1 hydraulic-conductivity operator. Current WRF "
        "4.6.1 and 4.7.1 retain the arguments but omit this operator. The "
        "ensemble provider owns its spectral restart state; ordinary "
        "physics refuses an enabled consumer without a bound pattern.")
    ruc_option["constraints"].setdefault("required_settings", {}).pop("spp_lsm", None)
    ruc_option["constraints"].setdefault("admitted_setting_values", {})["spp_lsm"] = [0, 1]
    for name, consumer, reader in (
            ("spp_conv", "GF closure", "gpuwm/core/gf.py"),
            ("spp_pbl", "MYNN PBL and surface", "gpuwm/core/physics.py")):
        registry["parameters"][name] = {
            "type": "integer", "enum": [0, 1], "per_domain": False,
            "consuming_read": reader,
            "warnings": [f"{name}=1 requires {consumer} and a member-owned stochastic "
                         "pattern bound by the ensemble timestep provider. The "
                         "provider owns the spectral restart state."]}

    route_text = (
        "The Noah-MP glacier refusal and sea-ice skip still apply. Its "
        "six-rate precipitation seam and radiation-cadence COSZEN carrier "
        "now follow WRF v4.6.1.")
    ceiling_text = (
        "Noah-MP's measured column ceiling is 360,000 columns, the widest "
        "configuration it has been timed at (2026-07-27). A wider grid is "
        "NOT refused: plan review WARNS and names the measured cost and its "
        "linear projection to the requested width, and "
        "GPUWM_NOAHMP_EXPERT_COLUMN_BUDGET records consent to a larger "
        "budget when a receipt wants one. The number is measurement "
        "coverage, not a physical or memory limit.")
    for route in (
            "tools.hrrr_single_domain_benchmark",
            "tools.prepared_domain_tree_forecast",
            "tools.prepared_single_domain_forecast"):
        registry["runner_routes"][route]["expert_warnings"][1] = ceiling_text
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
            "existing Noah-MP glacier, sea-ice and verification-status warnings "
            "still apply."),
        *mynn_noahmp["warnings"],
    ]
    templates[mynn_noahmp_id] = mynn_noahmp

    # The RUC templates reach the experiment-per-domain route on exactly the
    # sources whose initializers run RUC's own nine-level soil ingest -- the
    # single-domain route's era5 list and the benchmark's hrrr list already
    # declare them.  A tree whose every domain carries the nine-layer soil
    # is one land surface throughout (land_surface is not a per-domain
    # override on this route), so nothing is mixed; before this the second
    # user-report tuple (new-tiedtke / RUC / Milbrandt-Yau / YSU) was refused
    # on era5 by route declaration alone.  gfs stays withdrawn (v1.1.1).
    tree_route_ruc = registry["runner_routes"][
        "tools.prepared_domain_tree_forecast"].setdefault(
            "source_template_ids", {})
    for source_id in ("era5", "hrrr"):
        declared = tree_route_ruc.setdefault(source_id, [])
        for template_id in (
                "wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1",
                "wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1"):
            if template_id not in declared:
                declared.append(template_id)

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
#:     docs/public/wrf-comparison/mp28-column-evidence.md section 3.4.
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
#: THE UNEXCEPTIONED CLEAN SET.  18 of 22: sixteen of the nineteen spec'd
#: ``aero-*`` fixtures plus ``wp08-freeze`` and ``wp08-melt``.  ``wp08-freeze``
#: joined on 2026-09-23 when the rain fallout was handed WRF's L_qr (see the
#: note on MP28_G3_RESIDUALS).  The 1.4.1 merge did NOT change
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
    "aero-sfc-emit", "aero-warm-overlap", "wp08-freeze", "wp08-melt",
)

#: Fixtures that do NOT clear 2e-6 on every field, with every field that
#: misses and its measured maximum relative difference.  THREE of
#: twenty-two.
#:
#: ``aero-cold-overlap``'s 1.000e+00 rows are the accurate publication of a
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
#: wrf-matched-run mp=8 one, so the residual is recorded and filed as an
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
#: sedimentation DENSITY, not the gate's UNITS.
#:
#: CLOSED 2026-09-23: the mp=28 rain evaporation writes WRF's L_qr into
#: its reference density (zero where :3236 failed) and the adapter launches
#: the fallout's ``_with_presence`` entry points, which read it; the plain
#: entry points keep the mixing-ratio stand-in, and the classic rain
#: evaporation and the mp=8 adapter carry the same hand-off since
#: 7727fda3c.
#: Level 0 nr went 2.724e-06 (34 ulp) to 4.006e-07 (5 ulp) on a card (RTX
#: 4090 and RTX 5090 alike; 8.012e-08, 1 ulp, on the host build of the
#: kernels) and the fixture left this table.
MP28_G3_RESIDUALS: dict[str, dict[str, float]] = {
    "aero-cloud-freeze-nc": {"qc": 4.926e-06},
    "aero-cold-overlap": {
        "qc": 1.000e+00, "nc_per_kg": 1.000e+00, "effc_m": 8.102e-01,
        "qr": 4.443e-05, "nr_per_kg": 1.261e-04},
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
    -trajectory comparison.  It may not claim ``wrf-matched-run-candidate`` (no
    ratified reference comparison exists) and certainly not
    ``wrf-matched-run`` (no matched multi-hour run, no decay tables).  What it
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
                # ``kind`` is ``packaged-table-set`` (audit R-045).  It said
                # ``operator-supplied-table-set`` from before gpuwm shipped
                # the file, and every reader of the row -- including the
                # refusal text mp=28 printed when the table was missing --
                # inherited that stale word and told the operator to supply
                # a table a default install already has.  The row's own
                # note said the opposite in the same object, and
                # ``redistributed_by_gpuwm`` said it in a field.  ``kind``
                # now agrees with both.
                #
                # It remains deliberately NOT a member of the packaged
                # CLASSIC table set mp=8 resolves through TABLE_SET_ID: a
                # separate set with its own id, its own root and its own
                # environment overrides, so no mp_physics=8 launch acquires
                # a dependency on it.  ``kind`` names how the bytes ARRIVE,
                # which set they belong to is the ``id``.
                "id": "wrf-v4.6.1-aerosol-thompson-mp28-v1",
                "kind": "packaged-table-set",
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
                "search_root": "gpuwm_data/data/thompson/tables",
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
                    "MANIFEST.sha256 and shipped in the gpuwm-data companion "
                    "wheel that `pip install gpuwm` pulls, under WRF's "
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
                "clean_unexceptioned": 18,
                "clean_as_gated": 19,
                "clean_counts_note": (
                    "18 of 22 clear a FLAT 2.0e-6 relative / 2.0e-4 dB gate "
                    "on all 23 quantities with no bounds dict, no excluded "
                    "level and no per-fixture carve-out -- 16 of the 19 "
                    "spec'd aero-* fixtures plus wp08-freeze and wp08-melt. "
                    "19 of 22 clear "
                    "it with the ONE allowance above applied; that allowance "
                    "buys exactly one fixture, aero-reduces-to-classic, and "
                    "is required for it. The second allowance this note used "
                    "to name was retired at the 1.4.1 merge and is in "
                    "retired_allowances. clean_fixtures below is the "
                    "UNEXCEPTIONED list."),
                # No longer None.  docs/public/wrf-comparison/
                # mp28-matched-trajectory.md is a matched IDEALIZED forecast
                # against unmodified WRF v4.6.1, and it publishes its own
                # FAILED gate rather than a summary of the parts that passed.
                "forecast_trajectory_comparison": {
                    "document": ("docs/public/wrf-comparison/"
                                 "mp28-matched-trajectory.md"),
                    "kind": "idealized single-domain doubly-periodic forecast",
                    "not_nested_not_real_data": (
                        "This retained comparison is idealized, single-domain "
                        "and periodic. A matched nested or real-data WRF "
                        "comparison was not attempted. WRF's real.exe refuses "
                        "wif_input_opt=0 with mp_physics=28 "
                        "(dyn_em/module_initialize_real.F:2734-2736), but that "
                        "does not make a comparison on the current WIF route "
                        "impossible. The current route couples nwfa/nifa from "
                        "the WIF climatology and refuses specified domains "
                        "without the required dataset. Matching initial and "
                        "boundary inputs and retaining the comparison results "
                        "remain evidence gaps, not absent runtime capability."),
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
                    "final_interior_nwfa_per_kg_with_profile": 2.1737e7,
                    "final_interior_nwfa_per_kg_without_profile": 4.2877e6,
                    "peak_nc_per_kg_with_profile": 1.5980e8,
                    "peak_nc_per_kg_without_profile": 2.9451e7,
                    "droplet_ratio_with_over_without": 5.43,
                    "domain_total_rainnc_mm_with_profile": 1.957357,
                    "domain_total_rainnc_mm_without_profile": 3.207102,
                    "domain_total_rainnc_relative_excess": 0.6385,
                    "peak_rainnc_mm_with_profile": 0.794230,
                    "peak_rainnc_mm_without_profile": 1.029196,
                    "reading": (
                        "the LEFT column is what a run does today; the RIGHT "
                        "column is the counterfactual with the profile "
                        "removed. Removing WRF's CCN/IN loading gives 5.4x "
                        "fewer cloud droplets and 63.8% MORE domain-total "
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
            "No matched REAL-DATA or NESTED WRF trajectory and no decay table "
            "is recorded here. The WIF lateral-boundary carrier now exists, "
            "so missing matched inputs and a retained comparison are the "
            "evidence gap. There IS a "
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
            "/ 2.0e-4 dB gate with nothing held out: 18 of 22 clear every "
            "quantity (16 of the 19 spec'd aero-* fixtures, plus wp08-freeze "
            "and wp08-melt). "
            "aero-reduces-to-classic clears only through the port's ONE "
            "surviving named allowance -- 0-based level 6 held to 32 ulps of "
            "its entry value instead of the relative metric, measured 0.585 "
            "(qr) and 0.159 (nr) -- taking the gated count to 19 of 22. The "
            "relative bound that used to sit beside it was RETIRED at the "
            "1.4.1 merge: the mp=8 lane's two rain sedimentation "
            "reconciliations (5e4af4e3, cb765336), inherited in the frozen "
            "kernel mp=28 shares for fallout, took level 5's nr from "
            "5.700e-06 to 4.146e-07, inside the flat gate. THREE MISS "
            "OUTRIGHT -- aero-cold-overlap qc 1.000e+00 / nc 1.000e+00 / "
            "effc 8.102e-01 (all three are ONE branch flip at 0-based level "
            "4, where WRF ends with 1.4551915228366852e-11 kg/kg of cloud "
            "water -- exactly one float32 ulp of the entry value -- and "
            "gpuwm ends at exactly zero, so the qc1d <= R1 test at "
            "phys/module_mp_thompson.F:4007 sends "
            "the two implementations down opposite arms and a relative "
            "metric reports full scale on a one-ulp difference) plus "
            "nr 1.261e-04 / qr 4.443e-05 at level 6; "
            "aero-cloud-freeze-nc qc 4.926e-06; wp08-nusweep qr 4.642e-06 "
            "(2.3x the gate). wp08-freeze, which missed at nr 2.724e-06, "
            "left the list on 2026-09-23 when the rain fallout was handed "
            "WRF's L_qr (level 0 now 1 ulp from WRF). Every one of "
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
            "docs/public/wrf-comparison/mp28-column-evidence.md section 3.4 for "
            "the re-derivation from the committed fixtures.",
            "AEROSOL INPUT LIMITS: native met_em preparation accepts a complete "
            "analyzed QNWFA/QNIFA pair and the monthly WIF climatology reader "
            "is available. Auto uses the analyzed pair when present; an explicit "
            "source selector is retained and its receipt names the input used. "
            "The imported use_aero_icbc=true / wif_input_opt=1 / "
            "num_wif_levels=30 route selects the monthly dataset. There is no "
            "black-carbon (nbca) species, generic GOCART reader or qna_update "
            "auxiliary stream. SYNTHETIC INITIALISATION: "
            "WRF's synthetic fallback is thompson_init's "
            "SYNTHETIC CCN/IN profile (phys/module_mp_thompson.F:493-558), "
            "gpuwm implements it in "
            "gpuwm/core/microphysics.py::microphysics_init, measures it "
            "against WRF, and -- since 2026-08-01 -- CALLS it, once per "
            "domain, from gpuwm/core/physics.py::initialize_physics, at the "
            "seam WRF calls mp_init from phy_init. A freshly initialised "
            "mp=28 domain selecting this fallback starts on WRF's decaying continental "
            "profile, strictly above the terminal apply's clamps "
            "(phys/module_mp_thompson.F:3972-4021), not pinned at them. A "
            "HISTORICAL SYNTHETIC-PROFILE SENSITIVITY run measured over "
            "150 steps x 12 s on a "
            "28 x 16 x 24 2 km specified-BC convective domain against an "
            "otherwise identical run with the profile removed: initial mean "
            "nwfa 6.6532e+07 vs 0.0 kg^-1, peak nc 1.5980e+08 vs 2.9451e+07 "
            "kg^-1 (5.4x fewer droplets without it), domain-total RAINNC "
            "1.957357 vs 3.207102 mm -- the aerosol-free run rains 63.8% "
            "MORE. This describes the retained synthetic-profile experiment, "
            "not a current WIF-initialized run or a score against "
            "observations. See extensions.aerosol_initialisation.",
            "SYNTHETIC-PROFILE ADMISSION DIFFERENCE: WRF's initializer "
            "refuses this fallback configuration. dyn_em/module_initialize_real.F:"
            "2734-2736 calls wrf_error_fatal('wif_input_opt=0 but "
            "mp_physics=28'), so real.exe will not build a wrfinput for the "
            "synthetic-profile case at all. The PHYSICS is WRF's; the "
            "admission decision is not.",
            "AEROSOL LATERAL BOUNDARIES: the current WIF-climatology route "
            "couples nwfa/nifa through the specified boundary carrier. A "
            "specified domain without its required aerosol dataset is "
            "refused before step 0. The formerly measured zero-inflow "
            "depletion describes the historical uncoupled route, not the "
            "current WIF route. Runtime coupling is not evidence of a "
            "matched real-data or nested WRF forecast.",
            "MYNN aerosol-number mixing is optional. bl_mynn_mixscalars=0 "
            "is the default. Setting it to 1 mixes nc/nwfa/nifa using the "
            "WRF qn solves, and is admitted only with bl_pbl_physics=5, "
            "mp_physics=28 and bldt=0. This is component code verification, "
            "not validation against observations.",
            "PBL number mixing is selectable for mp_physics=28 with MYNN: "
            "scalar_pblmix=1 applies WRF post-PBL diffusion through exch_h "
            "to nc/ni/nwfa/nifa; bl_mynn_mixscalars=1 instead selects "
            "MYNN scalar plume transport. Both default to 0 and require "
            "bldt=0. WRF disables the former when the latter is active, "
            "so selecting both is refused.",
            "MIXED NESTING uses the registered transition policy. Entry "
            "into mp_physics=28 from a non-aerosol parent requires a declared "
            "aerosol source; same-scheme nesting carries the parent state. "
            "See the transition registry for admitted edges and provenance. "
            "Runtime admission is not a matched WRF trajectory comparison.",
            "DELIBERATE THERMODYNAMIC DIVERGENCE FROM mp_physics=8, and it is "
            "not a defect on either side. mp=28's RSLF/RSIF saturation Horner "
            "chains are contraction-pinned while mp=8's stay FMA-contracted, "
            "so the two schemes' saturation vapour pressures differ by one "
            "ulp. That matters because module_mp_thompson.F:3401 opens the "
            "whole condensation/CCN-activation block on ssatw > 1.E-15 (:185) "
            "-- one ulp flips a branch. mp=28 matches WRF's own gfortran "
            "-O2 arithmetic; mp=8 retains its FMA-contracted saturation "
            "chains. Its kernels changed after the 2026-07-28 matched "
            "run, so that run describes an earlier build. The two "
            "schemes are deliberately not bit-identical.",
            "Reachable through the registered aerosol-aware template and "
            "per-domain microphysics overrides where the route admits "
            "them. Read each route's declared templates and constraints; "
            "reachability is not forecast verification evidence.",
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
#: ids migrate through an explicit alias table, with old inputs accepted
#: and preparation-receipt physics identities kept unchanged.
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
    # 'port-in-progress' lived here while the WRF RRTM longwave row was
    # the one option using it.  That port landed (ra_lw_physics=1 +
    # ra_sw_physics=1, gpuwm/core/rrtm_lw.py) and moved to
    # implemented-unverified, leaving the rung with no occupant.  What
    # forces the deletion is tests/test_evidence_axes.py's AC3 case
    # test_the_rung_set_equals_the_set_in_use ("no decorative rungs"),
    # which requires the rung set to equal the set of maturity values
    # actually in use.  NOT the loader: gpuwm/physics_registry.py's D-22
    # check (_enforce_evidence_axes) enforces order-vs-rungs agreement
    # and value membership, and an unoccupied rung passes it.  Either
    # way the ladder is meant to describe states the registry is in, so
    # a future in-tree-but-unreachable port re-adds it here.
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
     "A historical matched-run label. Read verification_scope and its "
     "note: the current code or exact suite may not be covered, and "
     "some template labels are composition exemptions. "
     "This is code verification against WRF. It is not validation "
     "against observations."),
)

#: The independent-science axis (D-26: options only).  ``none`` is the
#: accurate default and the only value this pass assigns: the value set above
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
        "aliases": dict(MATURITY_RENAMES),
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
        "thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1": {
            "owner_decision_id": "D-16",
            "clause": "C2",
            "basis": (
                "A table-bound experimental runtime carried above its "
                "component floor. " + unverified_land_pbl),
        },
        "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-wrf-comparison-candidate-v1": {
            "owner_decision_id": "D-16",
            "clause": "C1+C2",
            "basis": (
                "NSSL-2 has fused-process oracles and a ratified 500 m "
                "comparison, and no matched-run manifest. "
                + unverified_land_pbl),
        },
        "nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-wrf-comparison-candidate-v1": {
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


MP9_OPTION_ID = "milbrandt2mom-mp9"


def _composition_ceiling(registry: dict, components: dict) -> str:
    """Clause C2 as a function: the highest rung a composition may carry.

    A composed suite is only as conformant as its weakest member, so the
    ceiling is the lowest-ranked maturity among the option rows the
    composition selects.  Nothing here reads a declared TEMPLATE maturity:
    the rung order is :data:`_MATURITY_RUNGS`, the one ordering of maturity
    names in this repository, and the option rows are the ones being built,
    so a template minted at this value states what the tree holds and
    cannot outrank it.

    Called from inside :func:`build`, before :func:`_rename_maturities`
    runs, so the raw option values are mapped through
    :data:`MATURITY_RENAMES` here rather than read as written.
    """

    order = [name for name, _tier, _definition in _MATURITY_RUNGS]
    selected = []
    for component_id, option_id in sorted(components.items()):
        option = registry["components"][component_id]["options"][option_id]
        maturity = option.get("maturity")
        maturity = MATURITY_RENAMES.get(maturity, maturity)
        if maturity not in order:
            raise SystemExit(
                f"component {component_id}.{option_id} carries maturity "
                f"{maturity!r}, which is not a rung of the ladder "
                f"({order}): a composition ceiling cannot be computed from "
                "a value the ladder does not define, and a template minted "
                "from it would carry an unreadable rank")
        selected.append(maturity)
    return min(selected, key=order.index)


def _milbrandt2mom_mp9(registry: dict) -> None:
    """Register Milbrandt-Yau two-moment at the maturity its evidence earns.

    ``implemented: true`` -- the scheme runs on the device through
    ``gpuwm/core/milbrandt2.py`` and ``gpuwm/core/kernels/milbrandt2.cu``
    and is dispatched by ``gpuwm/core/microphysics.py`` on
    ``mp_physics == 9``.

    ``maturity: implemented-unverified`` -- and no higher, for the plainest
    possible reason: NO oracle exists yet.  Unlike mp=28 (column-oracle
    measured, residuals published) or Morrison (ULP table against the real
    Fortran), this option's evidence today is a column smoke through the
    shipped seams plus float64 self-consistency.  It may not claim
    ``wrf-matched-run-candidate`` and it may not imply the column evidence is
    conformance evidence, because there is no comparison to conform to.

    ``reachability: component-override`` -- computed, not chosen.  No
    template selects mp=9 and ``DEFAULT_TEMPLATE_ID`` is untouched, so the
    only way in is a per-domain experiment override on the tree route,
    which already lists ``microphysics`` in
    ``allowed_component_overrides``.
    """
    options = registry["components"]["microphysics"]["options"]
    options[MP9_OPTION_ID] = {
        "asset_requirements": [],
        # The RTE+RRTMGP cloud-optics refusal is NOT written here.  It is
        # attached by _rte_rrtmgp_cloud_optics_constraints below, which
        # derives WHICH schemes need it from
        # gpuwm.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME rather than from a
        # list maintained by hand -- mp=9 was not the only scheme missing
        # a row, and the second one (P3) shipped into 1.9 undetected.
        "constraints": {
            "required_settings": {"moist": True},
        },
        "extensions": {
            "fixed_mode": (
                "two-moment in all six hydrometeors, separate graupel AND "
                "hail, continental CCN (CCNtype=2), Meyers+contact primary "
                "ice nucleation, non-spherical snow"),
            # ``arwen_radiation_constraint`` RETIRED with the refusal it
            # described: the RTE+RRTMGP adapter now carries the scheme's
            # own cloud-optics row (gpuwm.core.rrtmgp ``9: "milbrandt2"``),
            # so there is no pairing to refuse and the consumers block's
            # cloud_optics row, derived from that table, is the record.
        },
        "implemented": True,
        "label": "Milbrandt-Yau two-moment / MP9",
        "maturity": "implemented-unverified",
        "parameters": {
            "moist": True,
            "moist_cq": True,
        },
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"mp_physics": 9},
        "warnings": [
            "NO ORACLE HAS BEEN RUN. This option is a line-by-line "
            "transcription of the byte-frozen WRF v4.6.1 "
            "phys/module_mp_milbrandt2mom.F:841-3485 "
            "(mp_milbrandt2mom_main), :564-836 "
            "(sedi_wrapper_2/sedi_1D/count_columns), :31-433 and :3489-3525 "
            "(the helper functions) and :3559-3703 (the 3-D wrapper) -- and "
            "what has been TESTED is a column smoke "
            "driving the shipped seams (initialize_physics + "
            "microphysics.apply) for finiteness, boundedness and "
            "moment-vs-mass consistency, plus a water budget that closes "
            "to the reported precipitation on three seeding layouts. Read "
            "the budget narrowly: it detects a source/sink term only where "
            "that term is non-zero on the column, so it resolves the "
            "deposition, riming and melting families and NOT the terms "
            "that are zero on all three columns, and the number-only "
            "source/sink lines at "
            "phys/module_mp_milbrandt2mom.F:2721-2728 carry no mass and "
            "cannot appear in a water budget at all. The layouts and the "
            "measured per-term margins are in tests/test_milbrandt2.py. "
            "There is NO "
            "comparison against the WRF Fortran: no ULP table, "
            "no column oracle, no matched forecast trajectory. The oracle "
            "campaign is the declared next stage, as it was for Shin-Hong "
            "and Grell-Freitas before their measurements landed.",
            "PINNED IDENTITY, not a subset. WRF exposes no namelist for any "
            "Milbrandt-Yau switch: mp_milbrandt2mom_driver hard-codes "
            "CCNtype=2 (continental, "
            "phys/module_mp_milbrandt2mom.F:3615), precipDiag/sedi/"
            "warmphase/autoconv/icephase/snow all .true. (:3618-3623) and "
            "nk_BOTTOM=.false. (:3591), and the scheme body hard-codes "
            "snowSpherical=.false. (:1174), primIceNucl=1 (:1175) and "
            "grpl/hail/rainAccr/iceDep ON (:1170-1173). mp=9 therefore has "
            "exactly ONE identity in WRF v4.6.1 and ArWen ships that one; "
            "gpuwm/config.py::validate_milbrandt2_options refuses by name "
            "any request to move one, because the 154-entry constant table "
            "in gpuwm/core/milbrandt2_constants.py is derived under these "
            "settings (CCNtype=2 fixes N_c_SM=2e8, snowSpherical=.false. "
            "selects the Brandes m(D) pair).",
            "DELIBERATE DIVERGENCE, WRF is undefined or unreachable, four "
            "of them, each commented at its site in the ported "
            "translation unit gpuwm/core/kernels/milbrandt2.cu. (1) "
            "phys/module_mp_milbrandt2mom.F:2819-2823 is a LIVE "
            "print-and-STOP on T<173 K or T>323 K -- a device kernel cannot "
            "abort a run and ArWen's step-level health gate owns that "
            "decision, so the value is left as computed. (2) count_columns "
            "at phys/module_mp_milbrandt2mom.F:821 would read QX(i,0) when "
            "ktop_sedi==kbot, which needs the lowest two layers to span "
            "20 km, so the walk is clamped to 'inactive'. (3) "
            "phys/module_mp_milbrandt2mom.F:1733, :1754 and :1771 each zero "
            "iLAMxB1 twice and never zero iLAMxB2 -- every iLAMxB2 read "
            "sits inside the Qx>epsQ branch that also writes it, so the "
            "stale value is unreachable and the defined behaviour (zeroing) "
            "is implemented. (4) RT_snd at "
            "phys/module_mp_milbrandt2mom.F:3224-3262 and RT_peL at "
            ":3321-3327 are computed by the scheme and DISCARDED by "
            "mp_milbrandt2mom_driver, so they are not computed at all.",
            "TRANSCRIBED WRF QUIRKS, deliberately NOT corrected because "
            "they are defined behaviour: the surface precipitation "
            "diagnostic reads N_r, DE and T at the model TOP while reading "
            "QR at the bottom (phys/module_mp_milbrandt2mom.F:3298-3313, "
            "nk vs kbot); the raindrop-breakup efficiency tests iLAMr "
            "rather than Dr and goes negative for large iLAMr (:2951); the "
            "hail size-sorting ratio omits the air-density factor its Dm_x "
            "sibling carries (:698); "
            "and with T>0 C the pristine-ice mass is zeroed at :2042 and "
            "the resulting QMLir is then cancelled by the ice "
            "overdepletion guard, so that mass leaves the system rather "
            "than becoming rain. The scheme's own 4-term-truncated Lanczos "
            "gamma (:160-195, 'do j=1,4') is used throughout -- gamma(3) "
            "comes out 1.9999542 and that is the number every WRF mp=9 run "
            "integrates with.",
            "DECLARED DIVERGENCE, RTE+RRTMGP cloud optics (ArWen's own; WRF "
            "has no RRTMGP and hands RRTMG no Milbrandt-Yau radii, "
            "has_reqc=has_reqi=has_reqs=0 at phys/module_physics_init.F:"
            "1004-1023). The adapter radiates the scheme's OWN effective "
            "radii: the block WRF ships commented out at "
            "phys/module_mp_milbrandt2mom.F:3351-3378 -- r_eff = "
            "M_D(3)/(2 M_D(2)) over MY2005a eqn (2), 0.664639/lambda_c for "
            "the alpha_c=1, mu_c=3 cloud distribution and 1.5/lambda_i for "
            "exponential ice, with the scheme's own iLAMDA_x constants and "
            "iLAMmin2 floor -- evaluated from the transported nc/ni on "
            "every radiation call, and extended to snow as 1.5/lambda_s "
            "over the alpha_s=0 exponential with the Brandes m(D) pair "
            "(cms=0.1597, dms=2.078) that snowSpherical=.false. selects. "
            "Ice and snow merge into RRTMGP's one ice species by number "
            "and clip to the table domain exactly as Morrison's row does. "
            "The legacy RRTMG arm is unchanged: it computes its own radii "
            "as WRF does under has_reqc=0, so the two arms radiate "
            "DIFFERENT cloud radii for mp=9 by design; the verification of "
            "record is obs skill, not agreement between them.",
        ],
    }

WDM6_OPTION_ID = "wdm6-mp16"


def _wdm6_mp16(registry: dict) -> None:
    """Register WDM6 (mp_physics=16) at exactly the maturity it has earned.

    ``implemented: true`` -- the component exists and runs.  The column
    kernel is ``gpuwm/core/kernels/wdm6.cu``, transcribed line by line from
    the byte-frozen WRF v4.6.1 ``phys/module_mp_wdm6.F``; the adapter is
    ``gpuwm/core/wdm6.py``; ``gpuwm/core/microphysics.py`` dispatches on
    ``mp_physics == 16``; the state carries qnn/qnc/qnr; the namelist
    importer maps 16 natively.

    ``maturity: implemented-unverified`` -- and NOT for the reason that
    label carries on ``morrison-mp10`` and ``thompson-aerosol-mp28``, where
    an oracle ran and came back mixed.  Here it means the weaker thing, and
    the warning below says so in its first sentence: NO oracle comparison
    against the WRF Fortran has been run at all.  What exists is a
    shipped-seam column smoke and float64 self-consistency.  The oracle
    campaign is the declared next stage, as it was for Shin-Hong and
    Grell-Freitas before theirs ran.

    ``reachability: component-override`` -- computed from the routes, not
    chosen.  No template registers mp_physics=16 and no runner route lists
    one, so an implemented microphysics option with no template is reachable
    exactly one way: as a per-domain experiment override.
    ``tests/test_registry_reachability.py`` recomputes the state and would
    fail on any other declaration.
    """

    options = registry["components"]["microphysics"]["options"]
    options[WDM6_OPTION_ID] = {
        "asset_requirements": [],
        "constraints": {"required_settings": {"moist": True}},
        "extensions": {
            "wrf_package": (
                "Registry/Registry.EM_COMMON:3031 binds mp_physics==16 to "
                "the wdm6scheme package: moist qv,qc,qr,qi,qs,qg and scalar "
                "qnn,qnc,qnr, plus state re_cloud,re_ice,re_snow"),
            "prognostic_species": {
                "transported": ["qi", "qs", "qg", "nn", "nc", "nr"],
                "not_ported": [],
                "note": (
                    "WDM6 is WSM6's six-class mass set plus three number "
                    "moments: nn is the CCN reservoir, nc the cloud droplet "
                    "number and nr the rain number. Nothing in the WRF "
                    "package is left out."),
            },
            "family_siblings_refused": {
                "wdm5_mp14": (
                    "not ported; WDM5 has no graupel class and gpuwm refuses "
                    "mp_physics=14 by name in gpuwm/config.py rather than "
                    "substituting WDM6"),
                "wdm7_mp26": (
                    "not ported; WDM7 adds a hail class and gpuwm refuses "
                    "mp_physics=26 by name in gpuwm/config.py rather than "
                    "substituting WDM6"),
            },
            "surface_field_dependency": {
                "field": "XLAND",
                "authority": "phys/module_mp_wdm6.F:607-614",
                "note": (
                    "WDM6 is the first gpuwm microphysics scheme whose "
                    "PROCESS RATES read a surface field: the autoconversion "
                    "threshold is qc0 (maritime, xncr0=5e7) where xland==2 "
                    "and qc1 (continental, xncr1=5e8) elsewhere, a factor of "
                    "ten. The adapter reads XLAND from the physics driver "
                    "and REFUSES when none is present rather than assuming "
                    "a mask."),
            },
        },
        "implemented": True,
        "label": "WDM6 double-moment warm rain / MP16",
        "maturity": "implemented-unverified",
        "parameters": {
            "moist": True,
            "moist_cq": True,
            "wdm6_hail_opt": 0,
            "wdm6_ccn_conc": 1.0e8,
        },
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"mp_physics": 16},
        "warnings": [
            "NO ORACLE COMPARISON AGAINST THE WRF FORTRAN HAS BEEN RUN FOR "
            "WDM6. This option's evidence is: (1) the CUDA column kernel and "
            "the wdm6init coefficient block are transcribed line by line "
            "from the byte-frozen WRF v4.6.1 phys/module_mp_wdm6.F in the "
            "1974 reference bundle, with file:line citations at every "
            "process block; (2) gpuwm/core/wdm6_constants.py recomputes "
            "wdm6init in float64 and tests/test_wdm6.py pins the FP32 "
            "literals baked into the kernel against it; (3) a column smoke "
            "through the SHIPPED seams (initialize_physics into "
            "microphysics.apply at mp_physics=16) asserts finiteness, "
            "physical bounds, water-substance conservation to the "
            "sedimentation flux, and that a stubbed scheme fails the test. "
            "What is NOT established: any ULP or relative agreement with "
            "WRF's own module_mp_wdm6.F on any column, and any gpuwm/WRF "
            "forecast-trajectory comparison. Neither number exists, so "
            "neither may be quoted. The oracle campaign that produced "
            "Shin-Hong's max_ulp 0 and Grell-Freitas's 216-column boundary "
            "is the declared next stage for this scheme. Its known deltas "
            "are written down BEFORE it runs, in "
            "docs/wdm6_oracle_known_deltas.md: the float64 _rgmma "
            "coefficient floor, the PLM remap kt clamp, the rain-slope "
            "density inconsistency, and the -35 dBZ floor that looks like a "
            "delta and is not.",
            "DELIBERATE DIVERGENCE, WRF is inconsistent with itself (both "
            "halves are defined behaviour, so this is transcribed rather "
            "than repaired, and recorded here so it is not mistaken for a "
            "port error): wdm62D consumes ncr as a VOLUMETRIC number, "
            "dividing by (q*den) at module_mp_wdm6.F:2251, while "
            "refl10cm_wdm6 forms nr = nr1d*rho against rr = qr1d*rho so the "
            "densities cancel and nr1d enters as a PER-MASS number (:3005). "
            "The two rain slope definitions therefore differ by a factor "
            "rho inside a cube root, about 3 per cent in lamr near the "
            "surface. gpuwm reproduces both as written.",
            "ENGINE SEAM, recorded deviations of the mp_physics=16 adapter: "
            "(1) WRF fills the whole CCN array with the namelist ccn_conc "
            "on the first time step (module_mp_wdm6.F:220-227) and ccn0 "
            "reaches nothing else in the module; gpuwm performs that fill "
            "once at state allocation from cfg.wdm6_ccn_conc, which is "
            "equivalent for a cold start and is the NSSL qnn precedent, but "
            "is NOT equivalent if a future ingest supplies an initial CCN "
            "field -- such an ingest must advance the restart algorithm "
            "identity rather than resume onto this one. (2) WDM6 is not in "
            "the registry's cross-scheme nest transition table, so a domain "
            "hierarchy mixing mp=16 with another scheme is refused rather "
            "than diagnosed across the edge.",
        ],
    }

#: Why a microphysics scheme has no RTE+RRTMGP cloud-optics row, one
#: entry per selector that lacks one.  MEMBERSHIP is derived from
#: ``gpuwm.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME``; only the WRF reason is
#: written here, because the reasons genuinely differ and a generic one
#: would be worse than none.  A selector with no row and no entry here
#: STOPS THE BUILD -- that is the whole point, and it is what mp=50 needed
#: and did not get.
_NO_RTE_RRTMGP_CLOUD_OPTICS_REASON: dict[int, str] = {
    # EMPTY: every implemented scheme has a cloud-optics row.  The table
    # stays because the pass below still needs it the day a scheme is
    # implemented without one, and because tests/test_rrtmgp_coupling.py
    # pins it equal to the runtime's own exclusion set.
    # mp=9's entry RETIRED with the defect it described.  Milbrandt-Yau
    # now HAS a cloud-optics row (``9: "milbrandt2"`` in
    # gpuwm.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME): the scheme's own radii,
    # the block WRF ships commented out (module_mp_milbrandt2mom.F:
    # 3351-3378) evaluated over the transported number moments.  The
    # entry's own words -- Morrison's row "would derive the radii from a
    # gamma distribution that is not this scheme's" -- argued for a row of
    # the scheme's own, not for a refusal, and with ra_rrtmg_variant
    # defaulting to rte-rrtmgp the refusal fired on every bare mp=9 run.
    # mp=50's entry RETIRED 2026-08-31 with the defect it described.  P3
    # now HAS a cloud-optics row (``50: "p3"`` in
    # gpuwm.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME), transcribed from WRF's
    # own coupling: has_reqc=1, has_reqi=1, has_reqs=0, and the wrappers'
    # P3 species remap that hands the single ice category to the snow slot
    # at P3's own ice radius rather than consuming a snow radius that does
    # not exist.  Membership below is DERIVED from that dict, so the
    # selector no longer reaches this table and an entry left here would
    # be a reason for a refusal nothing emits.
}

#: The other way out of this pairing, stated in the REASON rather than in
#: the remedy label.
#:
#: A remedy label is printed as the title of a repair a front end can
#: APPLY, above a button that applies ``remedy_settings`` -- so a label
#: naming a second edit the remedy does not perform tells the reader the
#: button does something it does not.  Measured at
#: gpuwm/companion_physics.py: the offered repair sets
#: ra_rrtmg_variant='rrtmg_legacy' and nothing else, while its title also
#: offered the Dudhia pair.  The alternative is real and belongs in the
#: refusal, so it stays -- one clause earlier.
_RTE_RRTMGP_CLOUD_OPTICS_ALTERNATIVE = (
    "The Dudhia pair ra_lw_physics=0 / ra_sw_physics=1 is the other way "
    "out of this pairing.")

#: The remedy sentence both refusals end on, and the one edit
#: ``_RTE_RRTMGP_CLOUD_OPTICS_REMEDY_SETTINGS`` performs, in words.  One
#: copy, because it is the same two doors and a user who is told different
#: things about the same door stops trusting either.
_RTE_RRTMGP_CLOUD_OPTICS_REMEDY = (
    "Set ra_rrtmg_variant='rrtmg_legacy', which computes its own radii "
    "the way WRF does.")

#: The remedy's MACHINE half: the smallest edit that clears the refusal.
#:
#: A rule that carries only prose makes every front end re-derive the way
#: out from the sentence.  gpuwm/companion_physics.py did exactly that --
#: it matched "mp_physics=9" and "cloud-optics" in the refusal text to
#: decide both the summary it printed and the one repair it offered first
#: -- so a scheme that gains this refusal gets no tailored repair, and a
#: reworded sentence silently loses the one that exists.  Written here
#: beside the prose it belongs to, the rule answers "what do I change"
#: for every scheme this pass covers, including ones added later.
_RTE_RRTMGP_CLOUD_OPTICS_REMEDY_SETTINGS = {"ra_rrtmg_variant": "rrtmg_legacy"}


def registry_setting_names(registry: dict) -> set[str]:
    """Every setting name this registry can talk about.

    The parameter table alone is NOT that set, and assuming it was made a
    guard that could not see its own defect: ``ra_lw_physics`` is a
    component SELECTOR, not a parameter, so a remedy label naming the
    Dudhia pair passed a check written to catch exactly that label.  The
    selector keys each component declares, and the selectors and
    parameters its options carry, are the rest of the vocabulary.
    """

    names = set(registry.get("parameters", {}))
    for component in registry.get("components", {}).values():
        names.update(component.get("selector_keys", []) or [])
        for option in component.get("options", {}).values():
            names.update(option.get("selectors", {}) or {})
            names.update(option.get("parameters", {}) or {})
    return names


def _check_remedy_label_describes_its_edit(
        rule: dict, registry: dict, option_id: str) -> None:
    """A remedy's words and a remedy's edit must be the same remedy.

    ``remedy_label`` is printed as the TITLE of an applicable repair --
    gpuwm/companion_physics.py offers it above a button that applies
    ``remedy_settings``, and the desktop panel renders that title -- so a
    label naming a parameter the edit does not touch offers a way out the
    button does not take.  That is what shipped: the label named both
    ra_rrtmg_variant and the Dudhia pair while the edit set only the
    variant.

    Both directions are checked, over the registry's OWN parameter names,
    so this holds for every rule any later pass writes rather than for the
    one it was found on.
    """

    label = str(rule.get("remedy_label") or "")
    settings = rule.get("remedy_settings") or {}
    if not label or not settings:
        return
    missing = sorted(name for name in settings if name not in label)
    if missing:
        raise RuntimeError(
            f"microphysics/{option_id}: the remedy label does not name "
            f"{', '.join(missing)}, which its remedy_settings edits. A "
            "reader offered this repair cannot tell what it changes.")
    extra = sorted(
        name for name in registry_setting_names(registry)
        if name in label and name not in settings)
    if extra:
        raise RuntimeError(
            f"microphysics/{option_id}: the remedy label names "
            f"{', '.join(extra)}, which its remedy_settings does not set, "
            "so a front end that applies the remedy performs a smaller "
            "edit than the title it printed. State the other way out in "
            "the reason instead.")


def _rte_rrtmgp_cloud_optics_constraints(registry: dict) -> None:
    """Refuse, in the registry, every scheme RTE+RRTMGP cannot couple to.

    THE FAILURE THIS CLOSES.  gpuwm/core/rrtmgp.py resolves a cloud-optics
    row per ``mp_physics`` and RAISES on a selector with no row -- correct,
    and it raises at the FIRST RADIATION CALL, deep inside a forecast.  Two
    schemes reached 1.9 admitted by the loader with no row: mp=9, whose
    refusal existed in gpuwm/config.py but only as prose in the registry
    (tests/test_authority_agreement.py measured 320 combinations the two
    authorities decided differently), and mp=50, which had no refusal
    anywhere and would simply die mid-run.

    IT IS A CONJUNCTION, so it needs ``constraints.refused_when``.  A plain
    ``requires_components`` on radiation is WRONG here and was tried first:
    the three shipped ``*-rrtmg-legacy-*`` templates select the SAME
    ``rte-rrtmgp`` component option and switch adapters with the
    ``ra_rrtmg_variant`` PARAMETER, so excluding the option refuses the
    very configuration the refusal message recommends -- measured, 16
    disagreements in the other direction.

    DERIVED, not transcribed, in all three of its parts: which schemes need
    the rule comes from ``_MP_CLOUD_OPTICS_SCHEME`` itself, the radiation
    clause from the options' resolved selectors, and the variant clause
    from the shipped
    ``ra_rrtmg_variant`` enum minus the legacy value.  A new scheme, a new
    radiation option resolving to the same adapter, or a third adapter each
    force a decision here instead of silently widening the admission.
    """

    from types import SimpleNamespace

    from gpuwm.config import radiation_scheme_ids
    from gpuwm.core.rrtmgp import _MP_CLOUD_OPTICS_SCHEME

    radiation_options = registry["components"]["radiation"]["options"]
    rte_rrtmgp_4_4 = sorted(
        option_id for option_id, option in radiation_options.items()
        if radiation_scheme_ids(SimpleNamespace(
            **(option.get("parameters", {}) | option.get("selectors", {}))
        )) == (4, 4))
    if not rte_rrtmgp_4_4:
        raise RuntimeError(
            "no radiation option projects the RTE+RRTMGP 4/4 selector pair; "
            "the cloud-optics constraints would be vacuous.  If the adapter "
            "moved, re-derive from its new spelling rather than deleting "
            "the constraints.")
    variants = registry["parameters"]["ra_rrtmg_variant"]["enum"]
    rte_rrtmgp_variants = sorted(
        value for value in variants if value != "rrtmg_legacy")
    if not rte_rrtmgp_variants:
        raise RuntimeError(
            "ra_rrtmg_variant declares no non-legacy adapter; the "
            "cloud-optics constraints would be vacuous")
    if _RTE_RRTMGP_CLOUD_OPTICS_REMEDY_SETTINGS["ra_rrtmg_variant"] not in (
            variants):
        raise RuntimeError(
            "the cloud-optics remedy names an ra_rrtmg_variant value this "
            "registry does not declare, so every front end that applies it "
            "would offer a way out the parser refuses.  Re-derive the "
            "remedy from the adapter's new spelling rather than shipping "
            "one nothing accepts.")

    for option_id, option in sorted(
            registry["components"]["microphysics"]["options"].items()):
        if option.get("implemented") is not True:
            continue
        selector = option.get("selectors", {}).get("mp_physics")
        if selector is None:
            continue
        if selector in _MP_CLOUD_OPTICS_SCHEME:
            # DERIVED MEANS DERIVED IN BOTH DIRECTIONS.  This builder is a
            # transform over the TRACKED registry, not a render from
            # scratch: everything these passes do not touch is carried
            # through from the file on disk.  So a scheme that GAINS a
            # cloud-optics row -- which is what retiring the defect looks
            # like -- kept publishing the refusal it was fixed out of,
            # because the pass only ever wrote rows and never removed
            # them.  MEASURED at mp=50: the row ``50: "p3"`` landed, the
            # config-door and namelist-report guards were retired with it,
            # and the registry still told every reader P3 "has no
            # RTE+RRTMGP cloud-optics coupling" through a full rebuild.
            # A refusal the engine no longer makes is a refusal that
            # refuses working configurations, so this pass drops the
            # entries it owns once their reason is gone.
            constraints = option.get("constraints")
            if not constraints:
                continue
            kept = [entry for entry in constraints.get("refused_when", [])
                    if "ra_rrtmg_variant" not in entry.get("settings", {})]
            if kept:
                constraints["refused_when"] = kept
            else:
                constraints.pop("refused_when", None)
                if not constraints:
                    option.pop("constraints", None)
            continue
        reason = _NO_RTE_RRTMGP_CLOUD_OPTICS_REASON.get(selector)
        if reason is None:
            raise RuntimeError(
                f"microphysics/{option_id} (mp_physics={selector}) has no "
                "row in gpuwm.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME, so an "
                "RTE+RRTMGP run of it dies at the first radiation call. "
                "Either add the cloud-optics row with its WRF authority, "
                "or add the scheme's reason to "
                "_NO_RTE_RRTMGP_CLOUD_OPTICS_REASON so the registry can "
                "refuse the pairing up front.")
        rule = {
            "components": {"radiation": rte_rrtmgp_4_4},
            "reason": f"{reason} {_RTE_RRTMGP_CLOUD_OPTICS_ALTERNATIVE}",
            "remedy_label": _RTE_RRTMGP_CLOUD_OPTICS_REMEDY,
            "remedy_settings": dict(
                _RTE_RRTMGP_CLOUD_OPTICS_REMEDY_SETTINGS),
            "settings": {"ra_rrtmg_variant": rte_rrtmgp_variants},
        }
        _check_remedy_label_describes_its_edit(rule, registry, option_id)
        option.setdefault("constraints", {})["refused_when"] = [rule]


# ------------------------------------------------------------ consumer rows
#: Stock adapter classes the checkpoint writer recognises, by component and
#: option id.  These used to live as two literal dicts INSIDE
#: gpuwm/io/restart.py::physics_setup_identity, where a scheme with no row
#: was sent down the custom-callable path with a message about declaring a
#: restart_identity -- true of nothing the tree ships.  The 4/4 radiation
#: options carry one class per ``ra_rrtmg_variant`` because one selector
#: pair names two adapters.
_STOCK_CALLABLE_CLASSES: dict[str, dict[str, object]] = {
    "cumulus": {
        "off": None,
        "kain-fritsch": "gpuwm.core.kf.KainFritsch",
        "grell-freitas": "gpuwm.core.gf.GrellFreitas",
        "new-tiedtke": "gpuwm.core.ntiedtke.NewTiedtke",
    },
    "radiation": {
        "off": None,
        "dudhia-shortwave": "gpuwm.core.dudhia.DudhiaShortwaveRadiation",
        "wrf-rrtm-dudhia": "gpuwm.core.rrtm_lw.RRTMDudhiaRadiation",
        "analytic-clear-sky":
            "gpuwm.core.analytic_radiation.AnalyticClearSkyRadiation",
        "rte-rrtmgp": {
            "rte-rrtmgp": "gpuwm.core.rrtmgp.RRTMGPRadiation",
            "rrtmg_legacy": "gpuwm.core.rrtmg_legacy.RRTMGLegacyRadiation",
        },
    },
}

#: Component-owned first-call level windows, by option id: the receipt
#: label the vertical preflight reports and the NAME of the contract
#: constant in gpuwm/physics_vertical_contract.py that holds the pair.  The
#: value is read off the constant at build time so the registry cannot
#: carry a number the launcher does not enforce.  ``off`` options carry no
#: window (null).  The four rows the preflight's old if/elif chain never
#: had -- milbrandt2mom-mp9, wdm6-mp16, p3-mp50, new-tiedtke -- are why
#: a 100-level WDM6 run passed ``gpuwm check`` and died on its first
#: microphysics call.
_VERTICAL_BOUND_SOURCES: dict[str, dict[str, tuple[str, str]]] = {
    "microphysics": {
        "kessler-mp1": ("Kessler microphysics", "KESSLER_VERTICAL_LEVEL_BOUNDS"),
        "wsm6-mp6": ("WSM6 microphysics", "WSM6_VERTICAL_LEVEL_BOUNDS"),
        "thompson-mp8": ("Thompson microphysics",
                         "THOMPSON_VERTICAL_LEVEL_BOUNDS"),
        "thompson-aerosol-mp28": ("Thompson aerosol-aware microphysics",
                                  "THOMPSON_AEROSOL_VERTICAL_LEVEL_BOUNDS"),
        "morrison-mp10": ("Morrison microphysics",
                          "MORRISON_VERTICAL_LEVEL_BOUNDS"),
        "nssl2-mp18": ("NSSL-2 microphysics", "NSSL2_VERTICAL_LEVEL_BOUNDS"),
        "milbrandt2mom-mp9": ("Milbrandt-Yau microphysics",
                              "MILBRANDT2_VERTICAL_LEVEL_BOUNDS"),
        "wdm6-mp16": ("WDM6 microphysics", "WDM6_VERTICAL_LEVEL_BOUNDS"),
        "p3-mp50": ("P3 microphysics", "P3_VERTICAL_LEVEL_BOUNDS"),
    },
    "pbl": {
        "mynn": ("MYNN PBL", "MYNN_VERTICAL_LEVEL_BOUNDS"),
        "ysu": ("YSU PBL", "YSU_VERTICAL_LEVEL_BOUNDS"),
        "myj": ("MYJ PBL", "MYJ_VERTICAL_LEVEL_BOUNDS"),
        "uw": ("UW moist-turbulence PBL", "UWPBL_VERTICAL_LEVEL_BOUNDS"),
        "shinhong": ("Shin-Hong PBL", "SHINHONG_VERTICAL_LEVEL_BOUNDS"),
        "sase": ("SASE PBL", "SASE_VERTICAL_LEVEL_BOUNDS"),
    },
    "cumulus": {
        "kain-fritsch": ("Kain-Fritsch cumulus", "KF_VERTICAL_LEVEL_BOUNDS"),
        "grell-freitas": ("Grell-Freitas cumulus", "GF_VERTICAL_LEVEL_BOUNDS"),
        "new-tiedtke": ("New Tiedtke cumulus",
                        "NEW_TIEDTKE_VERTICAL_LEVEL_BOUNDS"),
    },
}

#: Prognostic moment structure per microphysics selector, the row
#: gpuwm/da/moments.py builds its SchemeMoments from.  Names are the
#: DomainState attributes each scheme's allocator gives it
#: (gpuwm/core/state.py, mp arm by mp arm), so the analysis state vector is
#: derived from the scheme and never typed at a call site.
#: ``repair_authority`` names the scheme's OWN q>0/N=0 repair where it is
#: ported (Morrison's PSD limiter); ``None`` means the analysis detects and
#: refuses instead of inventing an intercept.  mp=18 resolves its set from
#: WRF's option-18 consistency pass at call time (gpuwm.core.nssl2_contract)
#: and is marked so rather than enumerated twice.
#: Repair authorities are TOKENS the moments module resolves to its own
#: prose (gpuwm.da.moments._REPAIR_AUTHORITIES), because the prose carries a
#: WRF citation the registry's citation checker cannot resolve.
_MORRISON_REPAIR_AUTHORITY = "morrison-psd-limiter"
#: Thompson's own entry moment-consistency block
#: (module_mp_thompson.F:1827-1899), which sets a number moment from the
#: mass under the scheme's assumed distribution and zeroes both moments
#: below its activity gate R1.  ``zero_number_below_threshold`` records
#: that second half: a scheme that does not state it does not get it.
_THOMPSON_REPAIR_AUTHORITY = "thompson-entry-block"
_MOMENT_ROWS: dict[int, dict | None] = {
    0: None,
    1: {"name": "Kessler", "mass_only": ["qv", "qc", "qr"], "pairs": [],
        "unpaired": [], "repair_authority": None, "q_threshold": 1.0e-14},
    6: {"name": "WSM6", "mass_only": ["qv", "qc", "qr", "qi", "qs", "qg"],
        "pairs": [], "unpaired": [], "repair_authority": None,
        "q_threshold": 1.0e-14},
    # Thompson's entry block is its repair authority, and R1
    # (module_mp_thompson.F:183) is the activity gate that block compares
    # every mass against -- two orders of magnitude above the table's own
    # default, which stood here until the scheme's threshold was read.
    8: {"name": "Thompson", "mass_only": ["qv", "qc", "qs", "qg"],
        "pairs": [{"species": "rain", "mass": "qr", "number": "nr"},
                  {"species": "ice", "mass": "qi", "number": "ni"}],
        "unpaired": [], "repair_authority": _THOMPSON_REPAIR_AUTHORITY,
        "q_threshold": 1.0e-12, "zero_number_below_threshold": True},
    # Milbrandt-Yau: every one of the six hydrometeors carries a number
    # moment (module_microphysics_driver.F:1857-1862 binds
    # qnc/qnr/qni/qns/qng/qnh INOUT); gpuwm/core/state.py's mp=9 arm
    # allocates qh, nc, nr, ni, ns, ng, nh.  Its number initialisation is
    # not ported, so no repair authority.
    9: {"name": "Milbrandt-Yau two-moment", "mass_only": ["qv"],
        "pairs": [{"species": "cloud", "mass": "qc", "number": "nc"},
                  {"species": "rain", "mass": "qr", "number": "nr"},
                  {"species": "ice", "mass": "qi", "number": "ni"},
                  {"species": "snow", "mass": "qs", "number": "ns"},
                  {"species": "graupel", "mass": "qg", "number": "ng"},
                  {"species": "hail", "mass": "qh", "number": "nh"}],
        "unpaired": [], "repair_authority": None, "q_threshold": 1.0e-14},
    10: {"name": "Morrison two-moment", "mass_only": ["qv"],
         "pairs": [{"species": "cloud", "mass": "qc", "number": "nc"},
                   {"species": "rain", "mass": "qr", "number": "nr"},
                   {"species": "ice", "mass": "qi", "number": "ni"},
                   {"species": "snow", "mass": "qs", "number": "ns"},
                   {"species": "graupel", "mass": "qg", "number": "ng"}],
         "unpaired": [], "repair_authority": _MORRISON_REPAIR_AUTHORITY,
         "q_threshold": 1.0e-14},
    # WDM6: double-moment warm rain (nc, nr) over WSM6's ice; the CCN
    # reservoir nn is prognostic and has no mass partner
    # (gpuwm/core/wdm6_constants.py WDM6_NUMBER_SPECIES).
    16: {"name": "WDM6 double-moment warm rain",
         "mass_only": ["qv", "qi", "qs", "qg"],
         "pairs": [{"species": "cloud", "mass": "qc", "number": "nc"},
                   {"species": "rain", "mass": "qr", "number": "nr"}],
         "unpaired": ["nn"], "repair_authority": None, "q_threshold": 1.0e-14},
    18: {"resolved_by": "gpuwm.core.nssl2_contract.resolve_nssl2_mode"},
    # Aerosol-aware Thompson: mp=8's pairs plus prognostic droplet number
    # and the two aerosol tracers (gpuwm/core/state.py mp=28 arm).
    28: {"name": "Thompson aerosol-aware", "mass_only": ["qv", "qs", "qg"],
         "pairs": [{"species": "cloud", "mass": "qc", "number": "nc"},
                   {"species": "rain", "mass": "qr", "number": "nr"},
                   {"species": "ice", "mass": "qi", "number": "ni"}],
         "unpaired": ["nwfa", "nifa"],
         "repair_authority": _THOMPSON_REPAIR_AUTHORITY,
         "q_threshold": 1.0e-12, "zero_number_below_threshold": True},
    # P3 one-category: rain and the single ice category are two-moment; the
    # rime pair rides with the ice mass and has no number partner.
    50: {"name": "P3 one-category", "mass_only": ["qv", "qc"],
         "pairs": [{"species": "rain", "mass": "qr", "number": "nr"},
                   {"species": "ice", "mass": "qi", "number": "ni"}],
         "unpaired": ["qir", "qib"], "repair_authority": None,
         "q_threshold": 1.0e-14},
}

#: Offline downscale admission per microphysics selector: whether the
#: offline child lane (gpuwm/offline_child.py) can read a same-scheme parent
#: of this scheme, and where it cannot, the breakage that keeps it out.
#: Cross-scheme admission is a different question, derived there from the
#: nest-edge closures.  Every ``false`` names its defect so the retirement
#: sweep is a grep.
_OFFLINE_CHILD_ROWS: dict[int, dict] = {
    # passiveqv (Registry.EM_COMMON:3014) transports qv and nothing else,
    # which is exactly what the lane's transported-field helper now asks
    # for; the generic wrfout field map already carries QVAPOR.
    0: {"same_scheme": True, "refusal": None},
    # Kessler (Registry.EM_COMMON:3015): the qv/qc/qr prefix the helper
    # builds for every scheme IS its whole transported set, and the online
    # nest lane has carried mp=1 in PORTED_MP_PHYSICS throughout.
    1: {"same_scheme": True, "refusal": None},
    6: {"same_scheme": True, "refusal": None},
    8: {"same_scheme": True, "refusal": None},
    # Milbrandt-Yau: the lane carries a third scheme-qualified wrfout map
    # (_MY2_WRF_TO_STATE) beside the NSSL one, because QHAIL/QNHAIL are
    # declared by both packages and bind to different state fields, and
    # gpuwm's own history writer publishes QNHAIL -> nh for mp=9.
    9: {"same_scheme": True, "refusal": None},
    10: {"same_scheme": True, "refusal": None},
    # WDM6: the lane carries a fourth scheme-qualified wrfout map
    # (_WDM6_WRF_TO_STATE) with the QNCCN -> nn row, so the CCN reservoir
    # is read rather than zero-filled.
    16: {"same_scheme": True, "refusal": None},
    18: {"same_scheme": True, "refusal": None},
    28: {"same_scheme": True, "refusal": None},
    50: {"same_scheme": True, "refusal": None},
}

#: The specified-zone ring guard's snapshot family per microphysics
#: selector: the 3-D state fields it captures around every microphysics
#: call and the (ny, nx) accumulator/diagnostic slots the scheme writes.
#: gpuwm/core/physics_inventory.py prices exactly these slots and
#: gpuwm/core/microphysics.py captures the union of the state fields, so the
#: two cannot drift from each other or from this row.  Built in the pass
#: below because mp=18's set is read off gpuwm.core.nssl2_contract.
_RING_SURFACE_STANDARD = ["mp_rainnc", "mp_rainncv", "mp_snownc",
                          "mp_snowncv", "mp_graupelnc", "mp_graupelncv",
                          "mp_sr"]
_RING_STATE_ROWS: dict[int, list[str]] = {
    1: ["thp", "qv", "qc", "qr"],
    6: ["thp", "qv", "qc", "qr", "qi", "qs", "qg", "effc", "effi", "effs"],
    8: ["thp", "qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni",
        "effc", "effi", "effs"],
    # Milbrandt-Yau: hail mass beside graupel plus all six numbers.
    9: ["thp", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
        "nc", "nr", "ni", "ns", "ng", "nh", "effc", "effi", "effs"],
    10: ["thp", "qv", "qc", "qr", "qi", "qs", "qg",
         "nc", "nr", "ni", "ns", "ng", "effc", "effi", "effs", "effr"],
    # WDM6's three transported moments; nn is in the set because the scheme
    # WRITES it (fully evaporated rain and cloud return their number to the
    # reservoir, module_mp_wdm6.F:1249-1252, :1990-1994).
    16: ["thp", "qv", "qc", "qr", "qi", "qs", "qg", "nn", "nc", "nr",
         "effc", "effi", "effs"],
    # mp=18 is completed in the pass from nssl2_contract.DEFAULT_RESTART_FIELDS.
    18: ["thp"],
    # The exact mp=28 moment set; nwfa2d/nifa2d are INTENT(IN) in WRF and
    # no kernel writes them, so the guard has nothing to restore.
    28: ["thp", "qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni", "nc",
         "nwfa", "nifa", "effc", "effi", "effs"],
    # P3: one ice mass with its rime pair, two numbers, the two
    # previous-step carriers, and the cloud/ice radii pair only.
    50: ["thp", "qv", "qc", "qr", "qi", "qir", "qib", "ni", "nr",
         "th_old", "qv_old", "effc", "effi"],
}
_RING_SURFACE_ROWS: dict[int, list[str]] = {
    1: ["mp_rainnc", "mp_rainncv", "mp_kessler_sr"],
    6: list(_RING_SURFACE_STANDARD),
    8: list(_RING_SURFACE_STANDARD),
    9: _RING_SURFACE_STANDARD + ["mp_hailnc", "mp_hailncv"],
    10: list(_RING_SURFACE_STANDARD),
    16: list(_RING_SURFACE_STANDARD),
    18: _RING_SURFACE_STANDARD + ["mp_hailnc", "mp_hailncv"],
    28: list(_RING_SURFACE_STANDARD),
    # P3's wrapper writes RAINNC/RAINNCV/SNOWNC/SNOWNCV/SR and nothing else
    # (module_mp_p3.F:894-898); no graupel category.
    50: ["mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv", "mp_sr"],
}

#: Which wrfout precipitation accumulator each ring surface slot fills, for
#: the consumer export the Rust renderers read.
_RING_SLOT_ACCUMULATOR = {
    "mp_rainnc": "RAINNC", "mp_snownc": "SNOWNC",
    "mp_graupelnc": "GRAUPELNC", "mp_hailnc": "HAILNC",
}

#: The reflectivity producer set gpuwm/core/refl.py dispatches, the
#: scheme-diagnostic route gpuwm/da/obsop.py takes for NSSL, and the
#: selector ids these two facts leave without any route.
_REFLECTIVITY_SCHEME_DIAGNOSTIC = {
    18: "gpuwm.core.nssl2_diagnostics.diagnose_radardd02_if_due (WRF's "
        "radardd02, a separate pure diagnostic)",
    9: "gpuwm.core.milbrandt2.reflectivity (the scheme's own Zet block, "
       "lifted into gpuwm/core/kernels/milbrandt2_zet.cu because it "
       "updates nothing; that file carries the WRF citation)",
}


def _consumer_rows(registry: dict) -> None:
    """Publish, per implemented option, what every consumer of it reads.

    THE DEFECT THIS CLOSES is the one the 2026-09-10 trigger was an
    instance of: forty-four hand-kept scheme lists restating part of the
    registry's inventory, none of them checked at plan review, each read
    for the first time inside the run.  This pass makes the registry carry
    the rows those lists need -- PULLED from the module that owns the fact
    where one does (checkpoint identities, vertical-bound constants, the
    nest resolver's ported set, the radar operator's floors, the export
    inventory), and OWNED here as data where the consumer will derive its
    table from the registry (moments, offline-child admission, ring guard,
    stock adapter classes).  Every consumer then either derives from these
    rows at import or asserts equality with them at import
    (gpuwm.physics_registry.require_registry_agreement), and
    validate_run_config asks consumer_row_gaps() before step 0.

    Builds with the rebuild flag set so the consumers it imports do not
    refuse against the registry it is about to replace.
    """

    import dataclasses

    from gpuwm import checkpoint_identity as ci
    from gpuwm import physics_vertical_contract as vertical
    from gpuwm.core import microphysics_transition as transition
    from gpuwm.core import refl
    from gpuwm.core import rrtmg_legacy
    from gpuwm.core import rrtmgp
    from gpuwm.core.nssl2_contract import DEFAULT_RESTART_FIELDS
    from gpuwm.da import obsop
    from gpuwm.core.radiation_carriers import CONSUMER_CARRIERS
    from gpuwm.physics_registry import (
        CONSUMER_ROWS_KEY, CONSUMER_ROW_CONTRACT)
    from gpuwm.wrf_physics_inventory import _INVENTORIES

    components = registry["components"]

    def selector(option: dict, key: str):
        return option.get("selectors", {}).get(key)

    def bounds_row(component_id: str, option_id: str):
        source = _VERTICAL_BOUND_SOURCES.get(component_id, {}).get(option_id)
        if source is None:
            return None
        label, constant = source
        minimum, maximum = getattr(vertical, constant)
        return {"label": label, "minimum": minimum, "maximum": maximum,
                "contract_constant": constant}

    # -- microphysics ------------------------------------------------------
    for option_id, option in components["microphysics"]["options"].items():
        if option.get("implemented") is not True:
            option.pop(CONSUMER_ROWS_KEY, None)
            continue
        mp = int(selector(option, "mp_physics"))
        rows: dict[str, object] = {}
        rows["restart_algorithm_identity"] = (
            ci.MICROPHYSICS_ALGORITHM_IDENTITIES[mp])
        rows["vertical_level_bounds"] = bounds_row("microphysics", option_id)
        if mp in transition.PORTED_MP_PHYSICS:
            rows["nest_transition"] = {
                "mixed_edge_ported": True,
                "mass_fields": list(transition._MASS_FIELDS[mp]),
                "moment_fields": list(transition._MOMENT_FIELDS[mp]),
                "refusal": None,
            }
        elif mp in transition.UNVALIDATED_MIXED_EDGE_SELECTORS:
            scheme, _reason = transition._UNVALIDATED_MIXED_EDGE_REASONS[mp]
            rows["nest_transition"] = {
                "mixed_edge_ported": False,
                "mass_fields": None,
                "moment_fields": None,
                # The paragraph itself stays in the resolver (it carries WRF
                # citations the registry's citation checker cannot resolve);
                # the registry names the decision and where the reason is.
                "refusal": (
                    f"MP{mp} ({scheme}) is ported and runs, but it has no "
                    "validated cross-scheme entry closure for its moments ("
                    + ", ".join(transition.UNVALIDATED_MIXED_EDGE_MOMENTS[mp])
                    + "); the reason is recorded at gpuwm.core."
                    f"microphysics_transition._UNVALIDATED_MIXED_EDGE_REASONS[{mp}]"),
            }
        else:
            rows["nest_transition"] = {
                "mixed_edge_ported": False,
                "mass_fields": None,
                "moment_fields": None,
                "refusal": (
                    f"mp_physics={mp} is in neither PORTED_MP_PHYSICS nor "
                    "UNVALIDATED_MIXED_EDGE_SELECTORS "
                    "(gpuwm/core/microphysics_transition.py), so a mixed "
                    "edge touching it falls through to the generic "
                    "'ported selectors are ...' refusal; a same-scheme edge "
                    "resolves before either tuple is consulted"),
            }
        if mp == 0:
            radar = {"clear_air_floor_dbz": None,
                     "clear_air_floor_status": "no-microphysics",
                     "clear_air_floor_reason": None,
                     "reflectivity_route": "none",
                     "reflectivity_route_reason": None}
        else:
            if mp in obsop.CLEAR_AIR_FLOOR_DBZ:
                floor = {"clear_air_floor_dbz": obsop.CLEAR_AIR_FLOOR_DBZ[mp],
                         "clear_air_floor_status": "one-number",
                         "clear_air_floor_reason": None}
            elif mp in obsop.CLEAR_AIR_FLOOR_IS_NOT_ONE_NUMBER:
                floor = {"clear_air_floor_dbz": None,
                         "clear_air_floor_status": "not-one-number",
                         "clear_air_floor_reason": (
                             "the operator reports more than one clear-air "
                             "value; the measured reason is recorded at "
                             "gpuwm.da.obsop.CLEAR_AIR_FLOOR_IS_NOT_ONE_NUMBER"
                             f"[{mp}]")}
            else:
                floor = {"clear_air_floor_dbz": None,
                         "clear_air_floor_status": "unread",
                         "clear_air_floor_reason": (
                             "nobody has read this scheme's clear-air floor "
                             "off its kernel yet")}
            if mp in refl.REFL_10CM_INPUT_SPECIES:
                route = {"reflectivity_route": "operator",
                         "reflectivity_route_reason":
                             "gpuwm.core.refl.compute_refl_10cm"}
            elif mp in _REFLECTIVITY_SCHEME_DIAGNOSTIC:
                route = {"reflectivity_route": "scheme-diagnostic",
                         "reflectivity_route_reason":
                             _REFLECTIVITY_SCHEME_DIAGNOSTIC[mp]}
            elif mp in obsop.NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP:
                route = {"reflectivity_route": "native-not-separable",
                         "reflectivity_route_reason": (
                             "the scheme's Z is fused with its state update, "
                             "so there is no pure H(x); recorded at gpuwm.da."
                             "obsop.NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP"
                             f"[{mp}]")}
            else:
                route = {"reflectivity_route": "unrouted",
                         "reflectivity_route_reason": (
                             "the scheme produces REFL_10CM natively "
                             "(gpuwm.core.refl.SCHEME_NATIVE_REFL_10CM) but "
                             "gpuwm/da/obsop.py names no H(x) route for it. "
                             "No shipped scheme is in this state: "
                             "simulated_reflectivity refuses the class by "
                             "name from this row (audit R-050), so a scheme "
                             "that reaches it is one added ahead of its "
                             "operator row rather than one whose refusal "
                             "reads like a typo")}
            radar = {**floor, **route}
        rows["radar_da"] = radar
        rows["moments"] = _MOMENT_ROWS[mp]
        rows["offline_child"] = dict(_OFFLINE_CHILD_ROWS[mp])
        if mp == 0:
            rows["ring_guard"] = None
        else:
            state_fields = list(_RING_STATE_ROWS[mp])
            if mp == 18:
                # WRF's option-18 defaults resolve to the full two-moment,
                # hail, CCN and volume set; every one is advanced in place.
                state_fields += [name for name in DEFAULT_RESTART_FIELDS
                                 if name not in state_fields]
                state_fields += ["effc", "effi", "effs"]
            rows["ring_guard"] = {"state_fields": state_fields,
                                  "surface_slots": list(_RING_SURFACE_ROWS[mp])}
        if mp in refl.REFL_10CM_INPUT_SPECIES:
            rows["reflectivity_input_species"] = list(
                refl.REFL_10CM_INPUT_SPECIES[mp])
            rows["reflectivity_native_reason"] = None
        else:
            rows["reflectivity_input_species"] = None
            rows["reflectivity_native_reason"] = (
                ("the scheme produces REFL_10CM itself; recorded at "
                 f"gpuwm.core.refl.SCHEME_NATIVE_REFL_10CM[{mp}]")
                if mp else "no microphysics, no reflectivity")
        # Cloud optics: the RTE+RRTMGP coupling name (None where the
        # adapter records a judged refusal) and the legacy adapter's
        # use_mp_re declaration, both PULLED from the adapters.  An
        # implemented scheme in neither RTE+RRTMGP table is a gap the
        # build refuses, because the adapter would refuse it at the first
        # radiation call with "add a row".
        coupling = rrtmgp._MP_CLOUD_OPTICS_SCHEME.get(mp)
        if coupling is None and mp not in rrtmgp._NO_CLOUD_OPTICS_COUPLING:
            raise RuntimeError(
                f"mp_physics={mp} is implemented and gpuwm.core.rrtmgp "
                "neither couples it (_MP_CLOUD_OPTICS_SCHEME) nor records "
                "why not (_NO_CLOUD_OPTICS_COUPLING)")
        rows["cloud_optics"] = {
            "rte_rrtmgp_coupling": coupling,
            "rte_rrtmgp_refusal": (
                None if coupling is not None else
                "the scheme hands radiation no radii and no borrowed row "
                "is right; recorded at gpuwm.core.rrtmgp."
                f"_NO_CLOUD_OPTICS_COUPLING[{mp}], the way through at "
                "gpuwm.core.rrtmgp._CLOUD_OPTICS_REMEDY"),
            "legacy_declares_radii": rrtmg_legacy._MP_DECLARES_RADII[mp],
        }
        inventory = _INVENTORIES.get(mp)
        if inventory is None:
            rows["stock_wrf_export"] = {
                "inventoried": False,
                "reason": (
                    "no evidenced WRF v4.6.1 Registry.EM_COMMON package "
                    "contract is packaged for an unchanged WRF; export-only "
                    "scope, independent of the runtime verdict"),
                "wrfinput_fields": None,
                "runtime_state_not_wrfinput": None,
            }
        else:
            rows["stock_wrf_export"] = {
                "inventoried": True,
                "reason": None,
                "scheme": inventory.scheme,
                "registry_package": inventory.registry_package,
                "wrfinput_fields": [
                    {**dataclasses.asdict(field),
                     "dimensions": list(field.dimensions)}
                    for field in inventory.wrfinput_fields],
                "runtime_state_not_wrfinput": [
                    {**dataclasses.asdict(field),
                     "dimensions": list(field.dimensions)}
                    for field in inventory.runtime_state_not_wrfinput],
            }
        # Null for every scheme whose laterally forced form needs no
        # dataset beyond the boundary file itself, which is all of them but
        # one: the row exists on every option so a consumer reads a
        # DECISION rather than a missing key.
        rows["lateral_forcing_dataset"] = None
        if mp != 28:
            # ``mp28_aerosol_source`` is read by ONE scheme and
            # gpuwm.config refuses it on any other ("no other scheme reads
            # it, and gpuwm refuses a stray value instead of silently
            # dropping it").  Every other option therefore forbids its two
            # non-default values here, so the registry decides that tuple
            # the same way the config validator does instead of calling it
            # launchable.  Derived from the enum, not re-typed.
            option.setdefault("constraints", {}).setdefault(
                "forbidden_setting_values", {})["mp28_aerosol_source"] = [
                    value for value in MP28_AEROSOL_SOURCES
                    if value != MP28_AEROSOL_SOURCE_DEFAULT_VALUE]
            option["constraints"]["forbidden_setting_values"]["use_rap_aero_icbc"] = [True]
            previous_forbidden = copy.deepcopy(option["constraints"]["forbidden_setting_values"])
            previous_forbidden["mp28_aerosol_source"] = [
                value for value in registry["parameters"]["mp28_aerosol_source"]["compatible_previous_enums"][0]
                if value != MP28_AEROSOL_SOURCE_DEFAULT_VALUE]
            previous_forbidden.pop("use_rap_aero_icbc", None)
            option["compatible_previous_forbidden_settings"] = [previous_forbidden]
        if mp == 28:
            # THE ONE SPELLING of the dataset precondition, read by both
            # authorities: the run door refuses an externally forced mp=28
            # domain without the dataset
            # (gpuwm.config.mp28_aerosol_lateral_forcing_precondition), and
            # validate_physics_plan reports the same domain from this row.
            # The sentence and the ladder are imported from the modules
            # that own them rather than retyped, so a registry that says
            # LAUNCHABLE where the run door refuses is not expressible.
            from gpuwm.config import MP28_AEROSOL_LATERAL_FORCING_PRECONDITION
            from gpuwm.ingest import wif_climatology

            rows["lateral_forcing_dataset"] = {
                "id": "wrf-wif-monthly-aerosol-climatology-v1",
                "assets": [{"filename": wif_climatology.WIF_CLIMATOLOGY_FILE}],
                # ONE LADDER, named rather than copied.  This row used to
                # carry its own rung list, and plan review walked it with
                # the generic asset resolver -- which has no
                # working-directory rung (WRF's constants_name rule, which
                # the ingest ladder honours) and reads the single-file
                # override as a DIRECTORY.  Two ladders over one dataset is
                # how plan review calls a run launchable that the ingest
                # then cannot feed.  The registry now names the resolver
                # that owns the search, and
                # gpuwm.physics_registry.resolve_lateral_forcing_dataset
                # asks it, with no argument.  There was a
                # ``path_setting`` here naming ``wif_climatology_path``;
                # it is gone because it was never reachable -- that name
                # is not in the registry's ``parameters`` and no runner
                # route admits it, so the plan that would carry it is
                # refused by ``_parameter_error`` before the row is read.
                # A plan is portable and a filesystem path is not: the
                # operator-named path is a RunConfig field the run door
                # resolves, and the ways out this row can offer are the
                # environment override and the staged root.
                "resolver": ("gpuwm.ingest.wif_climatology"
                             ":resolve_wif_climatology"),
                "deliberate_setting": {
                    "name": "mp28_aerosol_source", "value": "synthetic"},
                "refusal": MP28_AEROSOL_LATERAL_FORCING_PRECONDITION,
            }
        option[CONSUMER_ROWS_KEY] = rows

    # -- cumulus / pbl / surface layer / land surface ----------------------
    identity_tables = {
        "cumulus": ("cu_physics", ci.CUMULUS_ALGORITHM_IDENTITIES),
        "pbl": ("bl_pbl_physics", ci.PBL_ALGORITHM_IDENTITIES),
        "surface_layer": ("sf_sfclay_physics",
                          ci.SURFACE_LAYER_ALGORITHM_IDENTITIES),
        "land_surface": ("sf_surface_physics",
                         ci.LAND_SURFACE_ALGORITHM_IDENTITIES),
    }
    for component_id, (key, table) in identity_tables.items():
        for option_id, option in components[component_id]["options"].items():
            if option.get("implemented") is not True:
                option.pop(CONSUMER_ROWS_KEY, None)
                continue
            value = int(selector(option, key))
            rows = {"restart_algorithm_identity": table[value]}
            if component_id in ("cumulus", "pbl"):
                rows["vertical_level_bounds"] = bounds_row(
                    component_id, option_id)
            if component_id == "cumulus":
                rows["stock_callable_class"] = (
                    _STOCK_CALLABLE_CLASSES["cumulus"][option_id])
            if component_id == "land_surface":
                bundle = ci.LAND_SURFACE_PARAMETER_SOURCES.get(value)
                rows["restart_parameter_bundle"] = (
                    None if bundle is None else
                    {"driver_attribute": bundle[0],
                     "asset_roles": list(bundle[1])})
                # FROM THE SCHEME'S OWN CARRIER CONTRACT, not from the
                # guard's list.  CONSUMER_CARRIERS states what each
                # land-surface scheme reads every surface step and REFUSES
                # an unlisted scheme rather than defaulting it to "reads
                # nothing", so a newly registered GLW-consuming LSM turns
                # this row True on its own -- and gpuwm.physics_compat's
                # radiation-off guard, whose table is held equal to these
                # rows at import, then fails until it carries the scheme
                # too.  Deriving it from that guard's own literal made the
                # row a restatement and could never have caught the drift.
                rows["reads_glw"] = "glw" in CONSUMER_CARRIERS[value]
            option[CONSUMER_ROWS_KEY] = rows

    # -- radiation ---------------------------------------------------------
    for option_id, option in components["radiation"]["options"].items():
        if option.get("implemented") is not True:
            option.pop(CONSUMER_ROWS_KEY, None)
            continue
        # Every radiation option is keyed on the pair its engine runs.
        # One row was keyed on (-1, -1) instead -- the sentinel for "the
        # split pair is not stated here" -- and had to be translated back
        # to 4/4 right here.  It is retired: a selector tuple that is an
        # absence matched every configuration written in the aggregate
        # spelling, whatever engine that spelling named.
        lw = int(selector(option, "ra_lw_physics"))
        sw = int(selector(option, "ra_sw_physics"))
        rows = {
            "restart_algorithm_identity": {
                "longwave": ci.LONGWAVE_ALGORITHM_IDENTITIES[lw],
                "shortwave": ci.SHORTWAVE_ALGORITHM_IDENTITIES[sw],
                "longwave_above_atmosphere_policy":
                    ci.LONGWAVE_ABOVE_ATMOSPHERE_POLICIES[lw],
                "shortwave_above_atmosphere_policy":
                    ci.SHORTWAVE_ABOVE_ATMOSPHERE_POLICIES[sw],
                "resolved_pair": [lw, sw],
            },
            "stock_callable_class": _STOCK_CALLABLE_CLASSES["radiation"][option_id],
        }
        if (lw, sw) == (4, 4):
            rows["restart_algorithm_identity"]["rrtmg_legacy_variant"] = (
                "identity resolved from gpuwm.core.rrtmg_legacy module "
                "constants at checkpoint time; no table row")
        option[CONSUMER_ROWS_KEY] = rows

    # -- turbulence --------------------------------------------------------
    for option_id, option in components["turbulence"]["options"].items():
        if option.get("implemented") is not True:
            option.pop(CONSUMER_ROWS_KEY, None)
            continue
        option[CONSUMER_ROWS_KEY] = {
            "restart_identity_binding": (
                "km_opt and its constants are bound by the checkpoint's "
                "configuration_sha256; there is deliberately no turbulence "
                "identity table (a row would be a sixth turbulence "
                "authority)"),
        }

    # -- completeness: the build refuses a registry that violates the
    #    contract, so plan review's consumer-row gate is silent by
    #    construction and fires only on a hand edit or a new consumer.
    for component_id, contract in CONSUMER_ROW_CONTRACT.items():
        for option_id, option in components[component_id]["options"].items():
            if option.get("implemented") is not True:
                continue
            rows = option.get(CONSUMER_ROWS_KEY, {})
            missing = sorted(set(contract) - set(rows))
            if missing:
                raise RuntimeError(
                    f"components.{component_id}.options.{option_id} is "
                    f"implemented and lacks consumer rows {missing}")

    registry["authority"]["consumer_rows_declaration"] = (
        "components.<component>.options.<option>.consumers publishes, per "
        "implemented option, the row every downstream consumer of that "
        "option reads: restart_algorithm_identity (and for land surface the "
        "restart_parameter_bundle, for cumulus and radiation the "
        "stock_callable_class), vertical_level_bounds, nest_transition, "
        "radar_da, moments, offline_child, ring_guard, "
        "reflectivity_input_species, stock_wrf_export and cloud_optics.  "
        "Generated by "
        "tools/build_registry.py from the module that owns each fact. "
        "Consumers derive their tables from these rows or assert equality "
        "with them at import (gpuwm.physics_registry."
        "require_registry_agreement); gpuwm.config.validate_run_config "
        "asks gpuwm.physics_registry.consumer_row_gaps before step 0, and "
        "validate_physics_plan mirrors it as consumer-row-missing.  A "
        "scheme is rows here, not a code path.")


def render_consumer_export(registry: dict) -> bytes:
    """The JSON the Rust crates hold their catalogs against.

    ``tools/rustwx`` (rw-wrfbatch's raw-plane catalog and QPF palette) and
    ``tools/rw_wps`` (the stock-WRF inventory admission set) each keep a
    hand-written table that restates part of this registry.  Neither can
    import Python, so the inventory they need is exported here as plain
    JSON, generated beside the registry and byte-pinned by the same test,
    and each crate's test reads it.
    """

    from gpuwm.io.wrf_output_schema import PRECIPITATION_OUTPUT_FIELDS
    from gpuwm.physics_registry import CONSUMER_ROWS_KEY

    microphysics = {}
    accumulators: dict[str, list[int]] = {
        name: [] for name in PRECIPITATION_OUTPUT_FIELDS}
    for option_id, option in sorted(
            registry["components"]["microphysics"]["options"].items()):
        if option.get("implemented") is not True:
            continue
        mp = int(option["selectors"]["mp_physics"])
        rows = option[CONSUMER_ROWS_KEY]
        ring = rows.get("ring_guard") or {}
        filled = sorted({
            _RING_SLOT_ACCUMULATOR[slot]
            for slot in ring.get("surface_slots", [])
            if slot in _RING_SLOT_ACCUMULATOR})
        for name in filled:
            accumulators[name].append(mp)
        export = rows["stock_wrf_export"]
        microphysics[str(mp)] = {
            "option_id": option_id,
            "label": option["label"],
            "stock_wrf_export_inventoried": bool(export["inventoried"]),
            "wrfinput_netcdf_names": (
                [field["netcdf_name"] for field in export["wrfinput_fields"]]
                if export["inventoried"] else []),
            "wrfinput_dimensions": (
                sorted({"/".join(field["dimensions"])
                        for field in export["wrfinput_fields"]})
                if export["inventoried"] else []),
            "precipitation_accumulators": filled,
        }
    document = {
        "schema": "gpuwm-physics-consumer-export-v1",
        "generated_by": "tools/build_registry.py",
        "registry_version": registry["registry_version"],
        "microphysics": microphysics,
        "precipitation_output_fields": list(PRECIPITATION_OUTPUT_FIELDS),
        "scheme_bound_precipitation_fields": {
            name: sorted(ids) for name, ids in accumulators.items()},
    }
    return (canonical_json(document) + "\n").encode("utf-8")


CONSUMER_EXPORT_PATH = MODEL / "gpuwm" / "physics_consumer_export_v1.json"


#: The one ladder every asset requirement is resolved down, derived here so
#: the registry document carries it and :func:`gpuwm.physics_registry.
#: resolve_asset_requirement` can walk it with nothing but ``pathlib`` and
#: :mod:`gpuwm.data_assets`.  Before this pass each row spelled its root in
#: its own key -- ``relative_root``, ``packaged_root``, ``search_root`` --
#: with no agreement on whether the spelling was the SOURCE tree's or the
#: INSTALL's, and two of them were stale: the RTE+RRTMGP rows still named
#: ``gpuwm/data/rrtmgp``, a directory that has not existed since 2.5.0 moved
#: those bytes into the gpuwm-data companion.  Nothing caught it because
#: nothing read the field: ``validate_physics_plan`` collected asset
#: requirements and never resolved one, so a wheel install missing an asset
#: heard nothing at plan review and refused at load, naming a path inside
#: site-packages.  It is resolved now, and the answer is REPORTED --
#: ``install_state``, with the missing members, the roots walked and the
#: command that stages them -- rather than made the plan's verdict, because
#: what this machine has staged is not a property of the plan.  The load is
#: still where it refuses, and it now refuses something the reader was
#: already told about.
#:
#: ``data_relative`` is the ``gpuwm/data``-relative path
#: :func:`gpuwm.data_assets.data_path` understands, which is the spelling
#: that survived the companion split: the caller states the path it always
#: stated and never which distribution carries it.  ``staged_root`` mirrors
#: the loader's own second rung (``gpuwm fetch-tables`` stages outside every
#: install, so the packaged root can legitimately be short an asset and the
#: run still work).
def _home_relative(parts) -> str:
    """``~``-relative spelling of a home-anchored root, from its owner.

    The document this builder writes must be byte-identical on every
    machine, so an absolute ``Path.home()`` cannot go in it -- which is
    why the roots below were re-typed here.  Importing the SEGMENTS and
    joining them keeps the one spelling in the module that owns the root
    and still writes a portable string.
    """

    return "~/" + "/".join(parts)


_ASSET_RESOLUTION_LADDERS = {
    # mp=8's classic set: two of its four assets are excluded from the
    # companion wheel by size (gpuwm-data/pyproject.toml) and arrive
    # through `gpuwm fetch-tables`, so the staged root is not a fallback
    # here, it is the normal answer on a fresh install.
    "wrf-v4.6.1-classic-thompson-mp8-gfortran13-v1": {
        "data_relative": "thompson/tables",
        "root_environment_override": "GPUWM_THOMPSON_TABLE_ROOT",
        "staged_root": _home_relative(
            physics_compat.USER_THOMPSON_TABLE_ROOT_PARTS),
    },
    # mp=28's CCN activation table: redistributed whole, no staging rung.
    "wrf-v4.6.1-aerosol-thompson-mp28-v1": {
        "data_relative": "thompson/tables",
        "root_environment_override": "GPUWM_THOMPSON_TABLE_ROOT",
        "path_environment_override": "GPUWM_THOMPSON_CCN_ACTIVATE",
    },
    "wrf-v4.6.1-p3-lookuptable1-2momi-v1": {
        "data_relative": "p3/tables",
        "root_environment_override": "GPUWM_P3_TABLE_ROOT",
    },
    # RTE+RRTMGP's NetCDF set.  ``assets`` is filled from
    # ``gpuwm.core.rrtmgp.RRTMGP_TABLE_FILES`` by ``_asset_resolution``:
    # this row declared a root and NO members, so plan review resolved it
    # vacuously -- ``resolve_asset_requirement`` returned
    # ``no-assets-declared`` and the caller skipped it -- and an install
    # short of a table still refused when radiation loaded.  It is one of
    # the two rows audit R-045 names, so it is also the row that must not
    # be allowed to declare nothing again; the guard below refuses that.
    "gpuwm-rte-rrtmgp-tables-v1": {
        "data_relative": "rrtmgp",
    },
    # Kain-Fritsch's lookup table: one file, in this wheel, no override.
    # It declares a relative_path rather than a root, so the ladder names
    # the directory and the row keeps its filename.
    "gpuwm-kf-lutab-v1": {
        "data_relative": "kf_lutab",
    },
}


def _asset_requirement_members() -> dict:
    """Files a requirement must name, imported from the module that opens them.

    Only for rows that carry no ``assets`` of their own.  A requirement is
    resolved by plan review by CHECKING ITS FILES, so a row that names
    none resolves vacuously; the guard in :func:`_asset_resolution`
    refuses that, and this is where the missing names come from -- the
    module that opens the members, never a second list.
    """

    from gpuwm.core import rrtmgp

    return {
        "gpuwm-rte-rrtmgp-tables-v1": [
            {"filename": name} for name in rrtmgp.RRTMGP_TABLE_FILES],
    }


def _asset_resolution(registry: dict) -> None:
    """Give every asset requirement the one ladder that resolves it."""

    from gpuwm import data_assets

    members = _asset_requirement_members()
    seen = set()
    for component in registry["components"].values():
        options = component.get("options")
        if not isinstance(options, dict):
            continue
        for option in options.values():
            for requirement in option.get("asset_requirements", []) or []:
                ladder = _ASSET_RESOLUTION_LADDERS.get(requirement.get("id"))
                if ladder is None:
                    raise SystemExit(
                        "asset requirement " + repr(requirement.get("id"))
                        + " has no row in _ASSET_RESOLUTION_LADDERS; a "
                        "requirement plan review cannot resolve is a "
                        "requirement that refuses after step 0")
                seen.add(requirement["id"])
                requirement["resolution"] = dict(ladder)
                declared = members.get(requirement["id"])
                if declared is not None and not requirement.get("assets"):
                    requirement["assets"] = copy.deepcopy(declared)
                # A requirement that names no FILE is a requirement plan
                # review cannot check: ``resolve_asset_requirement`` reports
                # ``no-assets-declared`` and an empty ``missing`` list would
                # otherwise read as satisfied, so the install would refuse
                # at load with nothing said earlier.  The reader's half
                # raises ``asset-undeclared`` for it -- a registry defect,
                # the same on every machine, so unlike the install-state
                # codes it does decide ``launchable`` -- and this refuses to
                # emit one at all.  Having a ladder is not enough; the
                # ladder has to be walked LOOKING FOR SOMETHING.
                if not (requirement.get("assets")
                        or requirement.get("relative_path")):
                    raise SystemExit(
                        "asset requirement " + repr(requirement.get("id"))
                        + " declares no assets and no relative_path, so "
                        "plan review would report it satisfied without "
                        "checking a single file; give it its file list "
                        "(from the module that opens them, via "
                        "_asset_requirement_members) or a relative_path")
                # The stale per-row root spellings are replaced by the one
                # the ladder carries, so no reader can pick up a path that
                # has not existed since 2.5.0.
                for stale in ("relative_root", "packaged_root",
                              "search_root"):
                    requirement.pop(stale, None)
                # ``relative_path`` names a FILE and is left alone; only the
                # three ROOT spellings are replaced by the ladder.
                # Where the bytes live on an INSTALL, decided by the one
                # module that owns the companion split rather than by a
                # second list here.  This is the field the two RRTMGP rows
                # got wrong for four releases.
                relative = ladder["data_relative"]
                requirement["installed_root"] = (
                    "gpuwm_data/data/" + relative
                    if data_assets._is_companion(relative)
                    else "gpuwm/data/" + relative)
    unused = sorted(set(_ASSET_RESOLUTION_LADDERS) - seen)
    if unused:
        raise SystemExit(
            "_ASSET_RESOLUTION_LADDERS rows with no asset requirement: "
            + ", ".join(unused))


def _lateral_forcing_remedy_is_reachable(registry: dict) -> None:
    """Every route that can select the option can also answer its refusal.

    A refusal names the way out or it does not stand (gate law).  The
    lateral-forcing dataset refusal offers two: stage the dataset, which
    is an environment question every route can answer, and set a
    per-domain parameter, which a route can only answer if its
    ``allowed_parameter_keys`` names it.  This pass derives that from the
    consumers row rather than leaving it to be typed into a route table --
    exactly the drift class this audit is retiring, and a way out only one
    route accepts would be a second per-route physics table.
    """

    from gpuwm.physics_registry import CONSUMER_ROWS_KEY

    remedies: dict[str, set[str]] = {}
    for component_id, component in registry["components"].items():
        for option in (component.get("options") or {}).values():
            row = (option.get(CONSUMER_ROWS_KEY) or {}).get(
                "lateral_forcing_dataset")
            if not isinstance(row, dict):
                continue
            deliberate = row.get("deliberate_setting") or {}
            name = deliberate.get("name")
            if name:
                remedies.setdefault(component_id, set()).add(str(name))
    if not remedies:
        return
    for route in registry["runner_routes"].values():
        reachable = set(route.get("allowed_component_overrides") or ())
        reachable |= set((route.get("allowed_component_options") or {}))
        wanted: set[str] = set()
        for component_id, names in remedies.items():
            if component_id in reachable:
                wanted |= names
        if not wanted:
            continue
        keys = set(route.get("allowed_parameter_keys") or ())
        route["allowed_parameter_keys"] = sorted(keys | wanted)


def _tke_nest_child(registry: dict) -> None:
    """km_opt=2 on a nest child: what the child does, not a refusal.

    The carried-through row refused it "until a nested prognostic-TKE
    domain has been run", a reason that names no breakage.
    gpuwm.experiment admits it under any parent; this row says what the
    child does and what is measured.
    """
    tke = registry["components"]["turbulence"]["options"]["tke-1.5-order"]
    tke.setdefault("extensions", {}).pop("nest_child_restriction", None)
    tke["extensions"]["nest_child"] = {
        "behaviour": (
            "a km_opt=2 nest child cold-starts its own TKE under any parent "
            "and never returns it, as in WRF v4.6.1, whose Registry gives "
            "tke no nest-interpolation (i) and no feedback (f) flag; tke is "
            "not a nest-forced field in gpuwm/core/nest_fields.py"),
        "measured": (
            "a 250 m km_opt=2 child under a km_opt=4 parent, 7 h, status "
            "PASS. A km_opt=2 child under a km_opt=2 parent loads with a "
            "not-yet-verified warning"),
    }
    tke["warnings"] = [
        ("Runs on a nest child under any parent; the child cold-starts its "
         "own TKE. See extensions.nest_child.")
        if warning.startswith("Refused on a nest child") else warning
        for warning in tke.get("warnings", [])]


def _urban_component(registry: dict) -> None:
    """WRF's urban canopy selector (``sf_urban_physics``), per option.

    Implemented: each model's CUDA column is graded against WRF v4.7.1's own
    routine (tests/test_urban_*_wrf471_parity.py).  Maturity stays at the
    implemented-unverified rung until the observation verification of record
    (ASOS, urban against rural stations) is recorded against it.
    """
    # The restart identities are gpuwm.checkpoint_identity's table, the
    # source the checkpoint writer reads, copied here as every component's
    # are.
    from gpuwm.checkpoint_identity import URBAN_ALGORITHM_IDENTITIES

    options = {}
    for value, option_id, module in (
            (0, "none", None), (1, "slucm", "gpuwm.core.urban_ucm"),
            (2, "bep", "gpuwm.core.urban_bep"),
            (3, "bep-bem", "gpuwm.core.urban_bem")):
        row = {
            "label": option_id, "selectors": {"sf_urban_physics": value},
            "implemented": True,
            "maturity": "supported" if value == 0 else "implemented-unverified",
            "scientific_evidence": "none",
            "parameters": ({"use_wudapt_lcz": 0, "num_urban_hi": 15}
                           if value else {}),
            "asset_requirements": [], "warnings": [], "extensions": {},
        }
        if module:
            row["consumers"] = {
                "restart_algorithm_identity": URBAN_ALGORITHM_IDENTITIES[value],
                "vertical_level_bounds": None}
            # Selected in a run's [shared] table.  WRF runs one urban
            # selector on every domain, so the per-domain override lists
            # exclude it (_PER_DOMAIN_EXCLUSION_REASONS) and no template
            # names it yet.
            row["reachability"] = {
                "state": "unreachable",
                "blocker": (
                    "selected by sf_urban_physics in a run's [shared] "
                    "table; WRF runs one urban selector on every domain, "
                    "so no per-domain route override offers it and no "
                    "registered template names it yet")}
            row["constraints"] = {
                "admitted_setting_values": {"use_wudapt_lcz": [0, 1], "num_urban_hi": [15]},
                "admitted_setting_values_reasons": {
                    "use_wudapt_lcz": "urban_param_init opens only URBPARM.TBL or URBPARM_LCZ.TBL",
                    "num_urban_hi": "HI_URB2D uses a 15-bin stride; another count would read past the array"},
                "requires_components": {"land_surface": ["noah", "noah-mp"]},
                "requires_components_reasons": {
                    "land_surface": "RUC and no-LSM never call an urban model; "
                    "the selector would be silently ignored."},
            }
            if value in (2, 3):
                row["constraints"]["requires_components"]["pbl"] = ["ysu", "myj"]
                row["constraints"]["requires_components_reasons"]["pbl"] = (
                    "BEP drag, heat and TKE sources enter the PBL implicit "
                    "solve; other ported PBL schemes would drop them.")
        else:
            row["consumers"] = {
                "restart_algorithm_identity": URBAN_ALGORITHM_IDENTITIES[0],
                "vertical_level_bounds": None}
        if value == 1:
            row["warnings"] = [
                "DECLARED DIVERGENCE from WRF v4.7.1: under Noah-MP the "
                "UCM's 2 m temperature is blended as the absolute "
                "temperature it is. module_surface_driver.F:3393 divides it "
                "by (1e5/PSFC)**RCP as if it were potential temperature, "
                "but module_sf_urban.F:1686 builds it from TS and TA, both "
                "absolute; the conversion cools every Noah-MP city cell by "
                "FRC x T x (1-(PSFC/1e5)**RCP), about 12 K at 1,900 m. "
                "Bitwise against WRF built with that one line fixed "
                "(tests/test_urban_ucm_noahmp_wrf471_parity.py).",
                "WRF stops the model (module_sf_urban.F:825) on any urban "
                "cell whose ZDC + Z0C + 2 m reaches the first model level; "
                "gpuwm refuses such a configuration when it loads, from the "
                "table's classes and the eta ladder (Local Climate Zones 1 "
                "and 4 need a first level above about 34 and 31 m).",
            ]
        elif value in (2, 3):
            row["warnings"] = [
                "DECLARED DIVERGENCE from WRF v4.7.1 under YSU: the rural "
                "surface drag is applied once. bl_ysu.F90:1313-1314 removes "
                "only the urban fraction of YSU's own drag while the BEP "
                "couple (module_sf_noahdrv.F:1708-1711) already folds the "
                "rural drag into a_u_bep, so WRF drags every column that is "
                "not wholly urban twice, ocean included (1.25 m/s of 10 m "
                "wind over the sea within two hours at 750 m). kernels/"
                "ysu.cu removes the whole of YSU's own drag; graded against "
                "WRF built with that one-line fix "
                "(tests/test_ysu_bep_rural_drag.py). MYJ is unaffected.",
            ]
        options[option_id] = row
    registry["components"]["urban"] = {
        "scope": "per-domain", "selector_keys": ["sf_urban_physics"],
        "options": options,
    }
    for name, default in (("use_wudapt_lcz", 0), ("num_urban_hi", 15)):
        registry["parameters"][name] = {
            "type": "integer", "default": default,
            "component_id": "urban", "consuming_read": "gpuwm/config.py",
        }
    for template in registry["templates"].values():
        template["components"]["urban"] = "none"
#: The UW moist-turbulence PBL option's warnings (bl_pbl_physics=9).  The
#: measured numbers are the parity tests' own (tests/test_uwpbl_*).
_UWPBL_WARNINGS = (
    "The UW PBL's distance from WRF v4.7.1 is MEASURED against the "
    "byte-unmodified CAMUWPBL sources: tools/uwpbl_wrf471_oracle builds "
    "them at gfortran -O0 and records every step of six regime families "
    "(convective day, stable night, stratocumulus, valley cold pool, "
    "mixed-phase, shallow cumulus) on three vertical grids, and the "
    "binary64 CPU reference gpuwm/verify/uwpbl_ref and the CUDA kernel are "
    "graded against those words bit for bit (tests/test_uwpbl_driver_"
    "wrf471_parity.py). That is conformance evidence, not scientific "
    "validation: no gpuwm/WRF forecast trajectory comparison and no "
    "observation score exists for this scheme yet, which is why this "
    "option is 'implemented-unverified'.",
    "THE ONE LICENCE-BOUND DIFFERENCE: WRF's cos and acos are glibc's "
    "binary64 IBM Accurate Mathematical Library (LGPL), which is not "
    "transcribed; the kernel uses correctly rounded cos/acos instead "
    "(gpuwm/core/kernels/glibc_flt64.cuh). They enter only the "
    "trigonometric root of the entrainment cubic, and where glibc does not "
    "round correctly there a column can differ from WRF in its last bits; "
    "the rate is measured and stated in the kernel header. exp, log and "
    "pow are glibc's own (Arm optimized-routines, MIT) and bitwise.",
    "WRF-AS-BUILT: gfortran's -O2 -ftree-vectorize build of these sources "
    "(WRF's configure default) is not byte-identical to its -O0 build on "
    "this toolchain: the vectoriser routes loops in eddy_diff through "
    "libmvec's cos/pow and folds x**2._r8 to x*x. The port follows the "
    "-O0 build, which calls glibc's scalar functions everywhere.",
)


def _rename_template_ids(registry: dict) -> None:
    """Move menu IDs while retaining an explicit map for old inputs."""

    def replace(node):
        if isinstance(node, dict):
            return {TEMPLATE_ID_ALIASES.get(key, key): replace(value)
                    for key, value in node.items()}
        if isinstance(node, list):
            return [replace(value) for value in node]
        if isinstance(node, str):
            return TEMPLATE_ID_ALIASES.get(node, node)
        return node

    normalized = replace({key: value for key, value in registry.items()
                          if key != "template_aliases"})
    registry.clear()
    registry.update(normalized)


def _current_verification_scope(registry: dict) -> None:
    """Keep historical/exempt labels from claiming current exact-suite evidence."""

    mp8 = registry["components"]["microphysics"]["options"]["thompson-mp8"]
    mp8["verification_scope"] = "historical-matched-run"
    mp8["verification_scope_note"] = (
        "The 2026-07-28 WRF v4.6.1 comparison used legacy RRTMG and failed "
        "the t=0 digest on all four domains. Kernels changed in 2.7.4 and "
        "on 2026-09-23; that matched run has not been repeated. Current "
        "component evidence is the documented WRF Fortran column comparisons.")
    for template_id, template in registry["templates"].items():
        if template.get("maturity") == "wrf-matched-run":
            exemption = registry["maturity_ladder"]["composition_rule"][
                "composition_exemptions"].get(template_id)
            if exemption:
                template["verification_scope"] = "composition-exemption"
                template["verification_scope_note"] = (
                    "This label is a composition exemption, not a current "
                    "matched run of this exact suite. " + exemption["basis"])


def build(registry: dict) -> dict:
    """Apply this pass's tables to ``registry`` in place and return it."""
    _rename_template_ids(registry)
    # Registered source tables determine which prepared inputs each runner
    # accepts. Carrying yesterday's registry source list omits a newly
    # mapped source even after its decoder and forecast route are usable.
    from gpuwm.prepared_single_domain_forecast import SUPPORTED_SOURCES as single_sources
    from gpuwm.prepared_domain_tree_forecast import SUPPORTED_SOURCES as tree_sources
    registry["runner_routes"]["tools.prepared_single_domain_forecast"]["source_ids"] = sorted(single_sources)
    registry["runner_routes"]["tools.prepared_domain_tree_forecast"]["source_ids"] = sorted(tree_sources)
    _urban_component(registry)
    _surface_coupling_warnings(registry)
    _thompson_aerosol_mp28(registry)
    _milbrandt2mom_mp9(registry)
    _wdm6_mp16(registry)
    from tools.morrison_wrf461_oracle.patch_registry_morrison import (
        MORRISON_WARNINGS)
    registry["components"]["microphysics"]["options"]["morrison-mp10"][
        "warnings"] = list(MORRISON_WARNINGS)
    # After every microphysics option is registered, because it walks them.
    _rte_rrtmgp_cloud_optics_constraints(registry)
    _tke_nest_child(registry)
    registry["authority"][
        "wrf_v461_compatibility_matrix"
    ] = _wrf_compatibility_authority()
    registry["authority"]["reachability_declaration"] = (
        "components.<component>.options.<option>.reachability declares how "
        "a user can select the option: 'template' through a registered base "
        "template, 'component-override' through either a route's full "
        "allowed_component_overrides or its option-scoped "
        "allowed_component_options, 'expert-template' only through a route's "
        "expert_template_ids, with expert_acknowledgement_id advisory, and "
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
    # WHY, beside WHICH, for the three rows above: the evaluator prints
    # this after the list, so the plan door names the breakage the run
    # door names (gpuwm.config / gpuwm.physics_compat) instead of a bare
    # "requires surface_layer in [...]".
    _fm_fh_reason = (
        "{scheme} binds fm/fh, the full similarity denominators "
        "ln(z/z0)-psi, directly and divides by them (zol = br*fm^2/fh); "
        "only the revised and classic MM5 surface layers publish them, so "
        "any other surface layer leaves them at their allocated zeros and "
        "the PBL runs on finite, plausible, wrong values for the whole "
        "forecast")
    pbl_options["ysu"]["constraints"]["requires_components_reasons"] = {
        "surface_layer": _fm_fh_reason.format(scheme="YSU")}
    pbl_options["shinhong"]["constraints"]["requires_components_reasons"] = {
        "surface_layer": _fm_fh_reason.format(scheme="Shin-Hong")}
    pbl_options["mynn"]["constraints"]["requires_components_reasons"] = {
        "surface_layer": (
            "the MYNN PBL takes its lower boundary from the friction "
            "velocity and the surface heat and moisture fluxes (ust/flt/flq) "
            "that the revised MM5, classic MM5 and MYNN surface layers "
            "publish; the Eta layer publishes MYJ's own exchange set "
            "instead and the off option writes none of them (WRF v4.6.1 "
            "admits the same three at phys/module_physics_init.F:3837-3839)")}
    # These implemented choices no longer have a single-value constraint.
    for name in ("bl_mynn_mixlength", "bl_mynn_mixscalars"):
        pbl_options["mynn"]["constraints"]["required_settings"].pop(name, None)
    pbl_options["mynn"]["parameters"]["scalar_pblmix"] = 0
    # MYNN's remaining closure knobs are pinned to the ported configuration; the
    # required_settings rows say so machine-readably and this says why.
    pbl_options["mynn"]["constraints"]["required_settings_reasons"] = {
        name: (
            "the ported MYNN EDMF closure is the WRF v4.6.1 configuration "
            "these knobs name; the option's parameters set this value and "
            "no other value has a ported code path")
        for name in pbl_options["mynn"]["constraints"]["required_settings"]}
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
    # ysu, mynn and shinhong declare NO moist requirement, and that is
    # a decision rather than a gap.  Audit R-025 proposed
    # required_settings moist=true on all five closures, reading the
    # rule off the state allocation; as a rule over the slot it is
    # false.  A dry state hands every seam the persistent zero moisture
    # planes, the moisture rows solve to exactly zero tendencies and
    # nothing consumes them, and a dry YSU plan is admitted by the
    # loader (tests/test_config.py::
    # test_km_opt4_admits_pbl_off_vertical_diffusion).  A
    # required_settings row is refused at plan review, so a row on
    # those three would refuse runs that work.
    # myj and sase keep theirs, each for a reason that is the row's own
    # text: WRF's PBL driver fatals a MYJ column without qv_curr/qc_curr
    # (phys/module_pbl_driver.F:1441-1443, pinned by
    # tests/test_myj_port.py::
    # test_a_dry_myj_run_is_refused_the_way_wrf_refuses_it), and SASE
    # forms its stability from the saturated Brunt-Vaisala frequency and
    # mixes condensate rows a dry column cannot give it.
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
    ] = ["revised-mm5", "classic-mm5", "mynn"]
    pbl_options["sase"]["constraints"]["requires_components_reasons"] = {
        "surface_layer": (
            "SASE's lower boundary condition is the surface layer's "
            "friction velocity, heat and moisture fluxes and "
            "gust-corrected wind speed (ust/hfx/qfx/wspd); the off option "
            "writes none of them"),
    }
    # "mynn" JOINED THAT LIST.  It was excluded by intersecting two
    # tables rather than by a physical reason: the MYNN surface layer's
    # own row transcribes WRF's isfc matrix, which has no cell for SASE
    # at all -- bl_pbl_physics=900 is outside the transcription's PBL
    # axis, and asking it for a verdict RAISES.  What SASE actually reads
    # is ust/hfx/qfx/wspd, and MYNN_SURFACE_OUTPUTS publishes all four,
    # allocated on sf_sfclay_physics=5 alone, independent of the PBL
    # selector.  The pairing is unmeasured, and unmeasured is maturity:
    # this registry's own policy is that maturity warns and never blocks,
    # so it carries the warning below and no constraint.
    _sase_mynn_warning = (
        "The MYNN surface layer is admitted under SASE on the field "
        "contract alone: it publishes the friction velocity, the heat and "
        "moisture fluxes and the gust-corrected wind speed the closure "
        "reads, and nothing else in the closure is surface-layer "
        "specific. No trajectory evidence covers this pairing -- the "
        "measured SASE runs are all revised/classic MM5 -- so it is "
        "experimentable, not evidenced.")
    # This builder is run against the registry it last wrote, so an append
    # has to be conditional or it doubles the warning on the second run.
    if _sase_mynn_warning not in pbl_options["sase"]["warnings"]:
        pbl_options["sase"]["warnings"].append(_sase_mynn_warning)
    # bldt IS NOT PINNED TO 0.  The required_settings row that carried it
    # was refused at plan review and at run start while
    # validate_sase_config admitted any cadence, and the driver is the
    # tiebreaker: it runs SASE at bldt_seconds, holds and recouples the
    # tendencies across skipped calls, and retains the flux diagnostics
    # between calls "at a positive PBL cadence". tests/test_sase.py and
    # tests/test_sase_cadence*.py exercise 0.1 s and 5.0 s cadences.
    pbl_options["sase"]["constraints"]["required_settings"].pop("bldt", None)
    # Stated in this option's own fourth warning as prose since it was
    # written ("it requires moist=true"), and enforced by
    # validate_sase_config, but never declared machine-readably -- so the
    # registry called 72 dry SASE combinations launchable that the loader
    # then refused.  The closure mixes vapour, cloud water and cloud ice
    # beside theta and forms its stability from the SATURATED
    # Brunt-Vaisala frequency; a dry state has nothing for it to integrate.
    pbl_options["sase"]["constraints"]["required_settings"]["moist"] = True
    pbl_options["sase"]["reachability"] = {"state": "component-override"}
    # ---- MYJ (bl_pbl_physics=2) and its Eta similarity surface layer -----
    # The one PAIR in this registry.  WRF v4.6.1 fatals a MYJ PBL whose
    # surface layer did not set isfc=2 (phys/module_physics_init.F:
    # 3770-3772), and among the surface layers gpuwm ports only MYJSFCSCHEME
    # does (:3169).  Both halves therefore declare the other in
    # requires_components, and gpuwm.config.validate_myj_pairing refuses the
    # mismatch in BOTH directions at load.  No template selects either --
    # the shinhong/sase posture: a user asks for the pair explicitly or does
    # not get it.
    _MYJ_PAIR_WARNING = (
        "MYJ and the Eta similarity surface layer are selected TOGETHER or "
        "not at all. WRF v4.6.1's own law is one-directional -- "
        "phys/module_physics_init.F:3770-3772 fatals bl_pbl_physics=2 "
        "unless the surface layer set isfc=2, which only "
        "sf_sfclay_physics=2 does among the layers gpuwm ports (:3169) -- "
        "and ArWen adds the reverse refusal for a stated reason: the Eta "
        "layer publishes AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 and publishes NO MOL, "
        "ZOL, PSIM/PSIH, REGIME, GZ1OZ0 or WSPD "
        "(phys/module_sf_myjsfc.F:361-1056), while YSU, MYNN, Shin-Hong and "
        "SASE each read at least one of those. Pairing them would hand a "
        "PBL scheme a zero where WRF hands it a similarity function: "
        "finite, plausible and wrong. gpuwm.config.validate_myj_pairing is "
        "that refusal.")
    _MYJ_EVIDENCE_WARNING = (
        "MYJ is implemented-unverified and the evidence string says exactly "
        "what was run: the float32 CPU authority "
        "(gpuwm/verify/myj_ref.py) is a line-by-line transcription of the "
        "byte-frozen phys/module_bl_myjpbl.F and phys/module_sf_myjsfc.F, "
        "and tests/test_myj_port.py drives the SHIPPED seams "
        "(initialize_physics + PhysicsDriver.compute) plus the CUDA "
        "translation units, asserting finiteness, physical bounds, the "
        "conservation the scheme has and CPU-vs-CUDA agreement, with "
        "mutation controls that STUB THE PORTED ROUTINES THEMSELVES and "
        "record which bars each stub turns red (the two bars nothing "
        "breaks -- _vdifq and the similarity-table lookup -- are declared "
        "in the same table rather than left silently green). TKE_MYJ "
        "cold-starts at epsq2=0.2, which is what MYJPBLINIT writes "
        "(phys/module_bl_myjpbl.F:1725, share/module_model_constants.F:92) "
        "and not zero: the seed decides MIXLEN's LPBL scan on step one and "
        "a zero column is a 0/0 in EL0 that WRF never reaches. NO "
        "ORACLE COMPARISON AGAINST THE WRF FORTRAN HAS BEEN RUN: there is "
        "no tools/myj_wrf461_oracle, no gfortran replay and no ULP table, "
        "so nothing here claims bit agreement with WRF. That campaign is "
        "the declared next stage, as it was for Shin-Hong and "
        "Grell-Freitas.")
    _MYJ_QUIRK_WARNING = (
        "WRF quirks transcribed rather than repaired, each cited in the "
        "port. CT (the countergradient correction) is identically zero in "
        "ARW: MYJSFC zeroes it every call "
        "(phys/module_sf_myjsfc.F:206-211) and SFCDIF's countergradient "
        "block is commented out (phys/module_sf_myjsfc.F:816-825, CT=0.), "
        "so MIXLEN's DTH+CT fix (phys/module_bl_myjpbl.F:845-850) and "
        "VDIFH's RKCT term add exactly zero -- both seams are still built "
        "so a future WRF that restores them lands in the right place. CZIL "
        "is hard-coded to 0.1 because the Chen-Zhang block is commented "
        "out (phys/module_sf_myjsfc.F:689-697), which is why "
        "IVGTYP/ISURBAN/IZ0TLND are dead arguments and why gpuwm.config "
        "refuses isftcflx/iz0tlnd with this surface layer. The "
        "Zilitinkevich thermal-roughness fix reads Z0BASE, not the working "
        "ZNT (phys/module_sf_myjsfc.F:733). DIFCOF's inversion block is "
        "commented out (phys/module_bl_myjpbl.F:1275-1322), so its T "
        "argument is dead and the port does not take it.")
    _MYJ_DIVERGENCE_WARNING = (
        "DELIBERATE DIVERGENCE, gpuwm goes its own way (float32 "
        "precision): the interface-height column is GROUND-relative. WRF "
        "seeds it with the terrain height, ZINT(I,KTE+1,J)=HT(I,J) "
        "(phys/module_bl_myjpbl.F:312, phys/module_sf_myjsfc.F:162), so "
        "every height it carries is above sea level; gpuwm seeds zero. "
        "Every consumer in both modules reads only DIFFERENCES of those "
        "heights, so HT cancels EXACTLY in real arithmetic and only "
        "APPROXIMATELY in float32, where differencing numbers offset by "
        "1-3 km drops bits the ground-relative column keeps. gpuwm's "
        "column is the more accurate one, which is why it ships. The "
        "cancellation is MEASURED rather than assumed: over five columns "
        "(four stretched so no dz and no interface height is an exactly "
        "representable float32) crossed with terrain at 1523.7, 2987.3 "
        "and 4411.1 m, land and water, KPBL is identical in every case, "
        "non-tendency fields move at most 69 ULP -- attained on lh over "
        "water, where the move is 1.645e-05 W m-2 -- and at most "
        "5.75e-06 relative, attained on qfx rather than on lh, and "
        "tendency rows move at most 2.05 "
        "quanta of the source field's ULP over dt "
        "(tests/test_myj_port.py::"
        "test_the_dropped_terrain_height_cancels_in_float32). It will be "
        "a real term in the ULP table when the oracle campaign runs.")
    _MYJ_SCOPE_WARNING = (
        "OUT OF SCOPE, refused rather than approximated. WRF sends the MYJ "
        "PBL through myjurb (phys/module_bl_myjurb.F:130-771) when sf_urban_physics "
        "is 2 or 3, the BEP/BEM multi-layer urban canopies "
        "(phys/module_physics_init.F:3775-3781). gpuwm does the same: "
        "PhysicsDriver._run_myj_pbl routes MYJ to gpuwm.core.myjurb under "
        "the bep and bep-bem urban options. "
        "With sf_surface_physics=0 the land branch stops evolving surface "
        "humidity rather than failing: MYJPBL rebuilds QSFC over land from "
        "ELFLX and CHKLOWQ, both of which an LSM writes, and its "
        "IF(QFC1>0.) guard leaves QSFC untouched when they are zero "
        "(phys/module_bl_myjpbl.F:547-557). That is WRF's own behaviour and "
        "it is admitted, not refused -- but a land run without an LSM is a "
        "land run whose surface moisture is frozen at its initial value. "
        "The MYJPBL species stack is WRF's own three or four rows (theta, "
        "vapour, cloud water and cloud ice when the moist set carries it): "
        "the QCS/QCR/QCG rows the source declares are commented out "
        "upstream (:274-285) and are not ported. The kernel holds the "
        "column in per-thread local memory like YSU's, so it carries the "
        "same 128-level ceiling and refuses a deeper column rather than "
        "truncating it.")
    pbl_options["myj"] = {
        "asset_requirements": [],
        "constraints": {
            # NO required_settings.  ``moist: True`` stood here and was
            # retired with the loader's arm: a dry gpuwm state is not an
            # absent state (every PBL seam gets the persistent zero
            # qv/qc planes), YSU, MYNN and Shin-Hong are all admitted dry
            # through those same planes, and MYJ's dry limit is defined
            # rather than 0/0.  If dry admission is ever denied it is a
            # property of the PBL SLOT and belongs as one column on every
            # PBL option row, not on this one.
            "required_settings": {},
            "requires_components_reasons": {
                "surface_layer": (
                    "the MYJ PBL's every implicit solve takes "
                    "AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 as its lower boundary and "
                    "only the Eta similarity surface layer produces them; "
                    "WRF v4.6.1 fatals the same pairing "
                    "(phys/module_physics_init.F:3770-3772)")},
            "requires_components": {
                # The surface layer is the ONLY component MYJ constrains.
                # An earlier draft also listed the land-surface options,
                # and that was a gate this port invented: WRF places no
                # LSM restriction on MYJ, and with sf_surface_physics=0
                # the code path is DEFINED rather than broken -- LH and
                # CHKLOWQ are zero, so QFC1 is zero and MYJPBL's
                # IF(QFC1>0.) guard simply leaves QSFC where it was
                # (module_bl_myjpbl.F:547-557).  Degraded, not impossible,
                # and the consequence belongs in a warning rather than in
                # a refusal; the pairing the port drove end to end is
                # named in the evidence warning instead.
                "surface_layer": ["eta-similarity"],
            },
        },
        "extensions": {
            "arwen_pairing_requirement": {
                "reason": (
                    "the MYJ PBL's every implicit solve takes "
                    "AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 as its lower boundary and "
                    "only the Eta similarity surface layer produces them"),
                "classification": (
                    "WRF v4.6.1 law in the PBL direction "
                    "(phys/module_physics_init.F:3770-3772); ArWen structural "
                    "constraint in the surface-layer direction"),
                "wrf_source": (
                    "phys/module_physics_init.F:3169,3770-3772"),
            },
        },
        "implemented": True,
        "label": "MYJ PBL",
        "maturity": "implemented-unverified",
        "parameters": {},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"bl_pbl_physics": 2},
        "warnings": [_MYJ_EVIDENCE_WARNING, _MYJ_PAIR_WARNING,
                     _MYJ_QUIRK_WARNING, _MYJ_DIVERGENCE_WARNING,
                     _MYJ_SCOPE_WARNING],
    }
    # ---- UW moist turbulence (bl_pbl_physics=9) ---------------------------
    # WRF v4.7.1's CAMUWPBLSCHEME, ported with its CAM modules and graded
    # against tools/uwpbl_wrf471_oracle.  It reads UST/HFX/QFX from its
    # surface layer and nothing else of the surface layer's set, so the
    # fm/fh reason that ties YSU and Shin-Hong to the MM5 layers does not
    # reach it and the MYNN layer is admitted beside it; the Eta layer is
    # refused for its own reason (its PBLH scan reads TKE_MYJ).
    pbl_options["uw"] = {
        "asset_requirements": [],
        "constraints": {
            "required_settings": {},
            "requires_components": {
                "surface_layer": ["revised-mm5", "classic-mm5", "mynn"]},
            "requires_components_reasons": {
                "surface_layer": (
                    "the UW scheme takes its lower boundary from the "
                    "friction velocity and the surface heat and moisture "
                    "fluxes (UST/HFX/QFX), which the revised MM5, classic "
                    "MM5 and MYNN surface layers publish; the Eta layer "
                    "publishes them too but its own PBLH scan reads the "
                    "TKE_MYJ column only the MYJ PBL advances, and the off "
                    "option writes none of them")},
        },
        "extensions": {
            "arwen_pairing_requirement": {
                "reason": (
                    "the Eta surface layer's PBLH scan is MYJSFC's TKE scan "
                    "over TKE_MYJ, which only the MYJ PBL allocates"),
                "classification": (
                    "ArWen structural constraint; WRF v4.7.1 places no "
                    "surface-layer law on CAMUWPBLSCHEME and fatals only "
                    "the BEP/BEM urban pairing"),
                "wrf_source": "phys/module_physics_init.F:3825-3832",
            },
        },
        "implemented": True,
        "label": "UW moist-turbulence PBL",
        "maturity": "implemented-unverified",
        "parameters": {},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"bl_pbl_physics": 9},
        "warnings": list(_UWPBL_WARNINGS),
    }
    surface_options["eta-similarity"] = {
        "asset_requirements": [],
        "constraints": {
            "requires_components": {"pbl": ["myj"]},
            "requires_components_reasons": {
                "pbl": (
                    "the Eta surface layer publishes no fm/fh, the full "
                    "similarity denominators ln(z/z0)-psi that YSU and "
                    "Shin-Hong bind directly and divide by, so under it "
                    "they would divide by an allocated zero; and its PBLH "
                    "scan reads the TKE column that only the MYJ selector "
                    "allocates, so with the PBL off the first surface step "
                    "has no column to scan")},
        },
        "extensions": {
            "arwen_pairing_requirement": {
                "reason": (
                    "the Eta surface layer's published set is not the MM5 "
                    "layers' published set, so a non-MYJ PBL would read "
                    "zeros for MOL/ZOL/PSIM/PSIH/REGIME/GZ1OZ0/WSPD"),
                "classification": (
                    "ArWen structural constraint; WRF v4.6.1 admits "
                    "sf_sfclay_physics=2 with PBL schemes gpuwm does not "
                    "port (phys/module_physics_init.F:3742, phys/module_physics_init.F:3756)"),
                "wrf_source": "phys/module_sf_myjsfc.F:361-1056",
            },
        },
        "implemented": True,
        "label": "Eta similarity surface layer",
        "maturity": "implemented-unverified",
        "parameters": {},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"sf_sfclay_physics": 2},
        "warnings": [_MYJ_EVIDENCE_WARNING, _MYJ_PAIR_WARNING,
                     _MYJ_QUIRK_WARNING, _MYJ_DIVERGENCE_WARNING,
                     _MYJ_SCOPE_WARNING],
    }
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
            # "myj" joined this list with the MYJ port, and it is a
            # STRUCTURAL statement rather than an evidence one -- the
            # extension below says so.  MYJ writes KPBL
            # (module_bl_myjpbl.F:421) and maintains the surface fluxes
            # through its Eta surface layer, which is exactly what the
            # adapter reads; leaving it out would have invented a
            # prohibition WRF does not make and would have put the
            # registry and validate_run_config into disagreement over 816
            # combinations.
            "requires_components": {
                "pbl": ["ysu", "mynn", "shinhong", "sase", "myj", "uw"]},
            "requires_components_reasons": {
                "pbl": (
                    "ArWen's Grell-Freitas adapter reads KPBL and the "
                    "PBL-maintained surface fluxes for the trigger's "
                    "temperature and moisture excesses and for the shallow "
                    "arm; with the PBL off nothing writes them, so the "
                    "scheme has no boundary-layer top to hand to it")},
            "required_settings_reasons": {
                "moist": _CUMULUS_MOIST_REASON},
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
            "holds the same boundary with the fzu normalisation PINNED "
            "from the capture, exactly as the CPU suite pins it: its "
            "transcribed glibc-2.39 logf/expf/powf are bitwise against "
            "the live-glibc sweeps, and gamma is graded instead against "
            "a 113-bit oracle because it is a deliberate divergence "
            "since 2.6.6 -- see the next warning "
            "(tests/test_gf_deep_cuda.py, "
            "tests/test_gf_shallow_cuda.py, tests/test_gf_gfdrv_cuda.py, "
            "tests/test_gf_gamma_correctly_rounded.py). "
            "The 8 remaining columns are the driver's own "
            "module_gfs_physcons mixed precision, inherited and bounded "
            "(max 34 ULP, 3.8e-6 relative, no branch flips). That is "
            "conformance evidence, not scientific validation: no "
            "gpuwm/WRF forecast trajectory comparison exists for this "
            "scheme, which is why it is 'implemented-unverified' and not "
            "'supported'.",
            "DELIBERATE DIVERGENCE, owner ruling (no inherited WRF bugs) "
            "-- GAMMA. Read docs/gf_gamma_known_delta.md before quoting "
            "any GF parity number. gfortran binds WRF's F2008 gamma() to "
            "glibc's tgammaf, which is NOT correctly rounded: MEASURED "
            "against a 113-bit oracle over all 59,768,833 float32 of "
            "[0.25, 36], it is wrong on 23,575,230 (39.4440 per cent), "
            "worst 6 ULP; tgammaf(4.0f) returns 6.00000048, not 6. "
            "Through 2.6.5 ArWen's earlier gamma returned those same "
            "words; it is replaced, and like its replacement it was this "
            "project's own work under the project's licence. "
            "gfk_tgamma is now ArWen's own correctly rounded "
            "gamma (0 of 59,768,833 arguments wrong) and the SHIPPED "
            "forecast path computes it, so fzu is no longer WRF's word: "
            "it changes on 68.1707 per cent of the reachable set, 98.390 "
            "per cent of that within 4 ULP, worst 12. MEASURED forecast "
            "consequence on the committed 216-column WRF v4.6.1 capture: "
            "the deep mass flux xmb moves by up to ~7 per cent, median "
            "1.9 (tests/test_gf_deep_parity.py::"
            "test_a_one_ulp_massflux_shape_perturbation_moves_xmb_by_"
            "seven_percent). ArWen is the closer of the two to the value "
            "the scheme's equations define on 49.95 per cent of the "
            "reachable set and further on 9.59; NEITHER is determined to "
            "better than tens of per cent, so this is NOT a skill claim. "
            "To compare against WRF column by column, pin fzu through the "
            "fzu_override slots the three stage kernels expose "
            "(INS_fzu_up/dn, SINS_fzu_sh, DINS_fzu_up/dn/sh) -- a runtime "
            "input, not a build flag.",
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
            "is the plumb-list for any label upgrade: (1) GFDRV's "
            "boundary-layer forcing (RTHBLTEN/RQVBLTEN) is fed the PBL "
            "slot's own raw dry-theta/qv rates, whichever of YSU, MYJ, "
            "MYNN, Shin-Hong or SASE holds the slot, and its radiative "
            "forcing (RTHRATEN) is fed the driver's held rates; the "
            "ADVECTIVE half (RTHFTEN/RQVFTEN) is fed the integrator's "
            "own exported rates -- the ARW dycore captures pure theta "
            "and qv advection at RK stage 1 of every step, uncoupled to "
            "K s-1 and kg kg-1 s-1 with a one-step lag, and the MPAS "
            "seam's caller supplies its own -- so all four forcing "
            "lanes are live.  PURE advection by construction: WRF's "
            "module_cumulus_driver.F:867 pre-folds RTHRATEN+RTHBLTEN "
            "into RTHFTEN for G3/NTiedtke and NOT for GFSCHEME, which "
            "sums the lanes itself; (2) GF's convective momentum "
            "tendencies are computed but not yet coupled (CumulusResult "
            "carries no momentum slots); (3) mass-level w is the "
            "KF-precedent average of the staggered field.",
            "MEASURED regime behaviour (2026-08-17, 12 km single-domain "
            "real-case twins on the tree route, 150x120x49, 6 h, KF "
            "control differing only in cu_physics/cudt): under strong "
            "synoptic forcing (1974-04-03 12-18Z) GF's domain-mean RAINC "
            "is ~40% of the KF control's -- ordinary inter-scheme "
            "spread; under weak forcing (1999-05-03 12-18Z, Ohio "
            "valley) it is 1-2% of KF's, i.e. the scheme is nearly "
            "silent where KF still rains. A column-level kernel probe "
            "of the weak-forcing state (18,000 real columns through "
            "gf_gfdrv_stage) found the deep trigger rejecting every "
            "column under four forcing arms alike -- the then-shipped "
            "zero-forcing seam, HFX/QFX-reconstructed RTHBLTEN/"
            "RQVBLTEN, and radiative forcing of either sign -- so the "
            "weak-forcing silence is the bitwise scheme's own "
            "trigger/closure response to these inputs, not a numeric "
            "defect at the adapter seam. A user expecting KF-like "
            "convective rain from cu_physics=3 on a weakly forced case "
            "will see almost none; that is the measured shape of the "
            "scheme as fed then, recorded here so it is not "
            "re-diagnosed as breakage. NOTE (2026-08-20): those twins "
            "predate the boundary-layer forcing lanes. Re-measured on a "
            "12 km GEM real case, one forecast hour, GF+YSU, feeding "
            "RTHBLTEN/RQVBLTEN moves domain-total RAINC by 12.8% "
            "(3.3% at 3 km, where the scale-aware closure damps "
            "itself), so the KF-relative ratios above are indicative "
            "and not re-verified against the current seam.",
        ],
    }
    surface_options["mynn"]["constraints"]["requires_components"][
        "pbl"
    ] = ["off", "mynn", "sase", "uw"]
    # WHY, not just WHAT.  "requires pbl in [...]" told a reader which
    # tuples were refused and nothing about what breaks, which is half a
    # refusal under the gate law; the evaluator renders this string after
    # the list.  The reason is ArWen's own field contract, not a WRF
    # citation: WRF's matrix is why the pairing is illegal THERE, and this
    # is what would happen HERE.
    surface_options["mynn"]["constraints"]["requires_components_reasons"] = {
        "pbl": (
            "the MYNN surface layer publishes psim/psih/gz1oz0 and does "
            "NOT publish fm/fh, the full similarity denominators "
            "ln(z/z0)-psi that YSU and Shin-Hong bind directly and divide "
            "by (they reconstruct zol = br*fm^2/fh from them). Those two "
            "would run on the allocated zeros -- finite, plausible and "
            "wrong -- for the whole forecast. Select the revised MM5 (1) "
            "or classic MM5 (91) surface layer for them, or the MYNN PBL, "
            "SASE, the UW PBL or no PBL scheme for this surface layer, all "
            "of which read only fields it publishes"),
    }
    # ---- New Tiedtke (cu_physics=16) -------------------------------------
    # THE PBL REQUIREMENT IS RETIRED, and this row is owned here now so
    # the retirement cannot be undone by a carried-through JSON row.  It
    # was cloned from Grell-Freitas, whose reason does not transfer: GF
    # reads fields["kpbl"] as a ONE-BASED column index and divides by
    # t[kpbl], so with the slot off it reads slot 0 of an uninitialised
    # workspace.  New Tiedtke reads no kpbl at all; its hfx/qfx come from
    # the SURFACE stack, which runs independently of bl_pbl_physics, and
    # its gf_rthblten/gf_rqvblten lanes are allocated zeros -- WRF's own
    # RTHBLTEN=0 fold at module_cumulus_driver.F:879-880.  The
    # "cumastrn:509 zdhpbl" integral the old reason cited runs from the
    # CLOUD-BASE index, not from a PBL index.
    ntiedtke = cumulus_options["new-tiedtke"]
    ntiedtke["constraints"]["requires_components"].pop("pbl", None)
    ntiedtke["extensions"].pop("arwen_pbl_structural_requirement", None)
    _ntiedtke_pbl_off_warning = (
        "New Tiedtke runs with the PBL slot off. It reads no KPBL, its "
        "surface fluxes come from the surface layer and the land-surface "
        "model rather than from the PBL scheme, and its advective-forcing "
        "lanes are the zero planes WRF's own cumulus driver folds in when "
        "no PBL tendency exists. The measured runs all carry a PBL "
        "scheme, so the PBL-off configuration is admitted on the field "
        "contract, not on evidence.")
    if _ntiedtke_pbl_off_warning not in ntiedtke["warnings"]:
        ntiedtke["warnings"].append(_ntiedtke_pbl_off_warning)

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

    # SOIL GEOMETRY, FROM THE SCHEMES' OWN TABLES.  RUC's row was
    # ``required_settings.num_soil_layers = 9`` -- a single value, because
    # that kind can say nothing else -- so plan review refused the
    # six-level RUC column that gpuwm/config.py admits, the kernel sizes
    # itself for and a completed forecast has written a wrfout on.  The
    # multi-valued kind says the set instead, and the set is READ from
    # gpuwm.config.LAND_SURFACE_SOIL_LAYERS, which reads it from each
    # scheme's own module (Noah and Noah-MP: gpuwm.core.noah; RUC:
    # gpuwm.core.ruc_contract.WRF_SUPPORTED_NUM_SOIL_LAYERS, the counts
    # WRF's init_soil_depth_3 tabulates).  A scheme that gains a geometry
    # gains the registry row, the parameter enum and the loader's
    # admission in one edit, which is the arbitrary acceptance test
    # applied to a soil table.  The EVIDENCE difference between RUC's two
    # geometries is not stated here: it is the option's warning and the
    # run receipt's soil_geometry_evidence line, because maturity warns
    # and never blocks.
    from gpuwm.config import LAND_SURFACE_SOIL_LAYERS
    # WHICH OPTIONS GET A GEOMETRY IS DERIVED, NOT LISTED.  A hand-typed
    # option-id -> selector map standing beside the registry being built is
    # the second table this whole edit exists to delete: every option
    # already carries its own ``selectors`` row one attribute away, and a
    # map also fixes the membership, so a newly registered land-surface
    # scheme would silently get no admitted geometry and contribute
    # nothing to the enum.  Reading the rows means registering a scheme is
    # the same edit that gives it both, and a scheme whose soil geometry
    # no module publishes fails the BUILD by name below rather than
    # shipping unconstrained.
    #
    # sf_surface_physics=0 is excluded and is the only value named here.
    # It selects no scheme at all: its LAND_SURFACE_SOIL_LAYERS row is the
    # length of a wrfout axis no field is written on
    # (gpuwm.config.NO_LAND_SURFACE_SOIL_LAYERS and the DIVERGENCE note
    # above it), not a soil column, so constraining a knob against it
    # would refuse counts on a run that allocates no soil state.
    (_soil_selector_key,) = (
        registry["components"]["land_surface"]["selector_keys"])
    _NO_LAND_SURFACE_SELECTOR = 0
    _SOIL_SCHEME_SELECTORS = {
        option_id: int(option["selectors"][_soil_selector_key])
        for option_id, option in land_options.items()
        if int(option["selectors"][_soil_selector_key])
        != _NO_LAND_SURFACE_SELECTOR}
    _soil_geometry_reason = (
        "a soil column exists only where its LEVEL DEPTHS do. WRF's "
        "generators tabulate zs for these counts and no others -- "
        "init_soil_depth_2 (Noah, Noah-MP) is fatal at any count but 4, "
        "and init_soil_depth_3 (RUC) tabulates 6 and 9 and leaves zs "
        "uninitialised otherwise -- and dzs is derived from zs, so a "
        "count with no row has no soil column at all rather than a "
        "coarse one. Select a count this scheme defines, or the scheme "
        "that defines the count you want")
    selectable_soil_counts: set[int] = set()
    for option_id, selector in sorted(_SOIL_SCHEME_SELECTORS.items()):
        try:
            defined = LAND_SURFACE_SOIL_LAYERS[selector]
        except KeyError:
            raise KeyError(
                f"land_surface option {option_id!r} selects "
                f"{_soil_selector_key}={selector}, for which no module "
                "publishes a soil geometry: give the scheme a row in "
                "gpuwm.config._LandSurfaceSoilLayers._PROVIDERS, naming the "
                "module and attributes that hold its counts, before "
                "registering it") from None
        counts = [int(value) for value in defined]
        selectable_soil_counts.update(counts)
        constraints = land_options[option_id].setdefault("constraints", {})
        constraints.setdefault("required_settings", {}).pop(
            "num_soil_layers", None)
        constraints.setdefault(
            "admitted_setting_values", {})["num_soil_layers"] = counts
        constraints.setdefault(
            "admitted_setting_values_reasons",
            {})["num_soil_layers"] = _soil_geometry_reason
    # THE KNOBS THAT REACH NO CODE ARE DECLARED, NOT REQUIRED.  Noah-MP's
    # option identity carried a ``required_settings`` row per knob, so plan
    # review refused a value outside the pin with "Set opt_pedo=1 for it, or
    # select another land_surface option."  Three of those knobs reach no
    # gpuwm code at ANY value -- opt_soil=1 makes the pedotransfer branch
    # unreachable, and gpuwm has no counterpart to WRF's output-accumulator
    # block -- so the refusal named no breakage, and the run door
    # (gpuwm.config.validate_run_config) admits them with one warning.
    # A plan-review door that still refused them would be a second door
    # disagreeing with the first about one configuration, so the rows come
    # out here.  The option's own ``parameters`` block still publishes the
    # pin, which is what a reader needs: the value gpuwm behaves as.
    #
    # WHICH knobs is READ from the config table, not listed here, so a row
    # whose evidence changes moves both doors in one edit.
    from gpuwm.config import NOAHMP_OPTIONS_WITHOUT_CONSUMER
    _noahmp_required = land_options["noah-mp"].setdefault(
        "constraints", {}).setdefault("required_settings", {})
    for _name in sorted(NOAHMP_OPTIONS_WITHOUT_CONSUMER):
        _noahmp_required.pop(_name, None)
    # The enum is assigned after the parameter tables are merged below,
    # which deep-copies this module's spec over whatever stands here.

    for option_id in ("noah", "ruc-lsm", "noah-mp"):
        option = land_options[option_id]
        # "eta-similarity" joined all three lists with the MYJ port, and
        # the reason it is all three is the reason this list exists at all.
        # It is a STRUCTURAL statement -- "who writes the exchange fields
        # this seam reads" -- not an evidence one, exactly as the extension
        # below says, and the Eta layer writes every one of them
        # (UST/CHS/CHS2/CQS2/FLHC/FLQC plus the RIB the driver carries as
        # BR); none of gpuwm's three LSM runtimes reads a MM5-only surface
        # field by name.  Restricting it to Noah, the pairing the port
        # actually exercised, was tried and is WRONG here: it invents a
        # prohibition WRF does not make, it puts this table into
        # disagreement with validate_run_config over 744 combinations (the
        # thing tests/test_authority_agreement.py exists to catch), and it
        # states evidence in the place reserved for structure.  Where the
        # evidence scope belongs is the OPTION's maturity and warnings,
        # which say implemented-unverified and say what was run -- the same
        # posture "mynn" already has in these three lists.
        accepted = ["revised-mm5", "classic-mm5", "mynn", "eta-similarity"]
        option.setdefault("constraints", {}).setdefault(
            "requires_components", {})["surface_layer"] = accepted
        option["constraints"]["requires_components_reasons"] = {
            "surface_layer": (
                "ArWen's active land-surface drivers consume the "
                "UST/CHS/CHS2/CQS2/FLHC/FLQC exchange fields the surface "
                "layer writes; with the surface layer off those fields are "
                "allocated and stay identically zero, and Noah-MP's "
                "write-back then divides by chs2/cqs2 behind a "
                "substitute-1.0 guard, so the run degenerates silently "
                "instead of failing")}
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
        # "myj" and "eta-similarity" are listed TOGETHER and only
        # together: they are the one pair in this registry, and a plan that
        # names one without the other is refused by their own
        # requires_components and again by
        # gpuwm.config.validate_myj_pairing.
        # "uw" is listed so the UW moist-turbulence PBL is selectable per
        # domain here; no template selects it (component-override).
        "pbl": ["off", "ysu", "mynn", "sase", "shinhong", "myj", "uw"],
        "surface_layer": [
            "revised-mm5", "classic-mm5", "mynn", "eta-similarity"],
        # "wrf-rrtm-dudhia" is WRF's classic 1/1 pair.  No template
        # selects it, so like sase and grell-freitas it is a
        # component-override: a user asks for RRTM longwave explicitly.
        # It is the only registered longwave-on pairing that is not the
        # coupled RRTMG adapter, which is the point of listing it here.
        "radiation": [
            "off", "dudhia-shortwave", "wrf-rrtm-dudhia", "rte-rrtmgp"],
    }
    # The per-domain override surface names the WHOLE implemented cumulus
    # set, so it is derived from the options rather than typed: a typed
    # list was not updated when New Tiedtke (cu_physics 16) landed, and
    # the published declaration went on naming three schemes while the
    # engine ran four.  Ordered by cu_physics.  Each option's own
    # required_settings row (New Tiedtke's cudt_minutes 0 and moist)
    # still applies to every plan that names it, and the single-domain
    # route inherits this list below.
    tree_route["allowed_component_options"]["cumulus"] = sorted(
        (option_id for option_id, option in cumulus_options.items()
         if option.get("implemented") is True),
        key=lambda option_id: cumulus_options[option_id]["selectors"][
            "cu_physics"])
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
    # THE RRTMG RECEIPT TOKEN BELONGS TO THE RADIATION OPTION.  The RRTMG
    # templates carry wrf_rrtmg_compatibility in their parameters, and a
    # per-domain radiation override on the tree route kept the template's
    # token beside a 0/0, 0/1, 1/1 or 90/90 pair -- plan review accepted
    # the domain and validate_run_config refused it at run start
    # ("requires the resolved 4/4 pair"), the two-door disagreement
    # tests/test_physics_combination_matrix.py measures.  Every radiation
    # option that is not the RRTMG 4/4 pair resolves the token to 'none'
    # through its own parameters, the way the legacy aggregate already
    # did, so the override resolves to the run the door admits.
    from types import SimpleNamespace
    from gpuwm.config import radiation_scheme_ids
    for option_id, option in radiation_options.items():
        if option.get("implemented") is not True:
            continue
        if radiation_scheme_ids(SimpleNamespace(
                **(option.get("parameters", {})
                   | option.get("selectors", {})))) == (4, 4):
            continue
        option.setdefault("parameters", {})["wrf_rrtmg_compatibility"] = "none"

    # WHY a turbulence closure needs the PBL slot it declares, in the
    # words the run door uses (gpuwm.config validate_run_config), so plan
    # review names the breakage and not only the pairing.
    turbulence_options = registry["components"]["turbulence"]["options"]
    _turbulence_reasons = {
        "closure-supplied": (
            "km_opt=0 runs no horizontal mixing operator, so the closure "
            "must supply the mixing itself: SASE computes its own "
            "horizontal mixing from its diffusivities and is the one PBL "
            "option that does, while every other PBL scheme produces none "
            "and the run would carry no explicit horizontal mixing at all "
            "(gpuwm.config admits that deliberately only through the "
            "km_opt_zero_acknowledgement research control)"),
        "smagorinsky-3d": (
            "its vertical exchange pair (kmv/khv) is applied by "
            "vertical_diffusion_2, which runs only with the PBL off, so "
            "with a PBL scheme on only the horizontal half of the closure "
            "would run and the run would not be the 3-D closure it names; "
            "km_opt=4 (2-D Smagorinsky) is the horizontal-only closure for "
            "a PBL-on domain"),
        "tke-1.5-order": (
            "its vertical TKE self-diffusion and surface TKE forcing are "
            "applied by vertical_diffusion_2, which runs only with the PBL "
            "off because a PBL scheme is already the column's vertical "
            "closure and running both would double-count vertical mixing; "
            "with a PBL on, TKE would be produced and dissipated "
            "column-locally with no vertical redistribution, so the run "
            "would not be the prognostic-TKE closure it names"),
    }
    for option_id, reason in _turbulence_reasons.items():
        constraints = turbulence_options[option_id].setdefault("constraints", {})
        if "pbl" in constraints.get("requires_components", {}):
            constraints["requires_components_reasons"] = {"pbl": reason}
        if "bl_pbl_physics" in constraints.get("required_settings", {}):
            constraints["required_settings_reasons"] = {
                "bl_pbl_physics": reason}
    # Coordinate diffusion has no tke_rhs or metric vertical operator.
    # WRF permits its horizontal TKE coefficients alongside a PBL scheme.
    tke_constraints = turbulence_options["tke-1.5-order"]["constraints"]
    for table, key in (("required_settings", "bl_pbl_physics"),
                       ("required_settings_reasons", "bl_pbl_physics"),
                       ("requires_components", "pbl"),
                       ("requires_components_reasons", "pbl")):
        tke_constraints.get(table, {}).pop(key, None)
    tke_constraints["refused_when"] = [{
        "settings": {"diff_opt": [2]},
        "components": {"pbl": sorted(set(pbl_options) - {"off"})},
        "reason": ("diff_opt=2: " + _turbulence_reasons["tke-1.5-order"]
                   + "; diff_opt=1 instead selects WRF coordinate-surface mixing"),
        "remedy_label": "Set bl_pbl_physics=0 for metric TKE closure.",
        "remedy_settings": {"bl_pbl_physics": 0},
    }]
    # The two diffusion selectors' own refusals (gpuwm.config
    # validate_km_opt), stated on every closure so plan review and the
    # front ends refuse what the run door refuses: tests/
    # test_authority_agreement.py measured both as "registry says
    # LAUNCHABLE, validate_run_config REFUSES".  Each closure's list is
    # assigned whole, after its own rules, so a rebuild is a fixpoint.
    non_ysu_pbl = sorted(
        option_id for option_id, option in pbl_options.items()
        if option.get("selectors", {}).get("bl_pbl_physics") != 1)
    no_pbl = sorted(
        option_id for option_id, option in pbl_options.items()
        if option.get("selectors", {}).get("bl_pbl_physics") == 0)
    for option_id, option in sorted(turbulence_options.items()):
        if option.get("implemented") is not True:
            continue
        km_opt = option["selectors"]["km_opt"]
        rules = [rule for rule in
                 option.get("constraints", {}).get("refused_when", [])
                 if not {"topo_wind", "gwd_opt"}.intersection(
                     rule.get("settings", {}))
                 and ("diff_opt" not in rule.get("settings", {})
                      or "components" in rule)]
        if km_opt not in (2, 4):
            rules.append({
                "settings": {"diff_opt": [1]},
                "reason": (
                    f"diff_opt=1 (WRF coordinate-surface diffusion) is "
                    f"implemented for km_opt=2 and 4, which supply its "
                    f"exchange coefficients; km_opt={km_opt} does not, so "
                    f"the run would carry no coordinate operator"),
                "remedy_label": (
                    "Set diff_opt=2 for the metric operator this closure "
                    "runs."),
                "remedy_settings": {"diff_opt": 2},
            })
        rules.append({
            "settings": {"diff_opt": [2], "mix_full_fields": [False]},
            "reason": (
                "mix_full_fields=false selects WRF's perturbation mixing (the "
                "base-state profile subtracted before mixing), which the "
                "diff_opt=2 metric operator does not implement: the run "
                "would mix full fields while its configuration says "
                "otherwise"),
            "remedy_label": "Set mix_full_fields=true.",
            "remedy_settings": {"mix_full_fields": True},
        })
        # Every plan selects a turbulence option.  Its existing constraint
        # object carries this PBL coupling without creating a new empty
        # physics-identity wrapper for PBL off.  No sources clause: the
        # preflight re-checks this configuration-only admission rule.
        rules.append({
            "settings": {"topo_wind": [1, 2]},
            "components": {"pbl": non_ysu_pbl},
            "reason": (
                "topo_wind acts only through YSU's surface drag "
                "(bl_pbl_physics=1); a non-YSU PBL does not consume its "
                "ctopo coefficients, so the run would name a terrain-wind "
                "correction without applying one"),
            "remedy_label": "Set topo_wind=0.",
            "remedy_settings": {"topo_wind": 0},
        })
        rules.append({
            "settings": {"gwd_opt": [1, 3]},
            "components": {"pbl": no_pbl},
            "reason": (
                "gwd_opt adds its drag inside the PBL driver, which a run "
                "with bl_pbl_physics=0 never calls, and reads the PBL "
                "height and top level an active PBL scheme writes"),
            "remedy_label": "Set gwd_opt=0.",
            "remedy_settings": {"gwd_opt": 0},
        })
        option.setdefault("constraints", {})["refused_when"] = rules
    sase_constraints = pbl_options["sase"]["constraints"]
    sase_constraints["required_settings_reasons"] = {
        "km_opt": (
            "SASE computes its own horizontal mixing from the closure's "
            "own diffusivities, so a km_opt mixing operator would "
            "double-count it"),
        "khdif": "constant-K diffusion may not silently stack on the SASE mixing",
        "kvdif": "constant-K diffusion may not silently stack on the SASE mixing",
        "moist": (
            "the closure mixes water vapour, cloud water and cloud ice "
            "beside theta and forms its stability from the saturated "
            "Brunt-Vaisala frequency; a dry state has nothing for it to "
            "integrate"),
    }
    # Every moist scheme says why it needs a moist state, once per slot.
    for option in registry["components"]["cumulus"]["options"].values():
        constraints = option.get("constraints", {})
        if constraints.get("required_settings", {}).get("moist") is True:
            constraints.setdefault("required_settings_reasons", {})[
                "moist"] = _CUMULUS_MOIST_REASON
    for option in registry["components"]["microphysics"]["options"].values():
        constraints = option.get("constraints", {})
        if constraints.get("required_settings", {}).get("moist") is True:
            constraints.setdefault("required_settings_reasons", {})[
                "moist"] = (
                    "a dry state allocates no water vapour or hydrometeor "
                    "fields, so the scheme would have nothing to integrate")
    # The RRTMG 4/4 options pin icloud=1 and the aggregate selector; the
    # rows say why in the words of the run door.
    _icloud_reason = (
        "clear-sky coupling (icloud=0) is not wired through the RRTMG 4/4 "
        "engines and no recorded oracle case runs at icloud=0, so the run "
        "would not be the configuration its receipt names; the 1/1 pair "
        "(WRF RRTM longwave with Dudhia shortwave) honours icloud=0 end to "
        "end")
    for option_id, ra_physics_reason in (
            ("rte-rrtmgp",
             "the 4/4 pair is spelled by ra_lw_physics/ra_sw_physics on "
             "this option and the aggregate ra_physics selector stays 0, so "
             "the two spellings of one radiation choice cannot disagree"),
            ):
        constraints = radiation_options[option_id].setdefault("constraints", {})
        if constraints.get("required_settings"):
            constraints["required_settings_reasons"] = {
                "icloud": _icloud_reason, "ra_physics": ra_physics_reason}

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

    # SELECTABLE, not evidenced -- see the spec's own comment.  Assigned
    # here because the merge above deep-copies this module's spec table
    # over the parameters dict, so an earlier write would be discarded.
    # The counts come from the land-surface schemes' own modules, gathered
    # where their option rows were written.
    params["num_soil_layers"]["enum"] = sorted(selectable_soil_counts)

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
    # carries qs (Registry.EM_COMMON:3036), so F_QS is true for it, and
    # mp_physics=16's wdm6scheme package carries qs at :3031, so F_QS is
    # true for WDM6 too.
    mynn["extensions"]["supplied_moisture_species"] = {
        "supplied": ["qv", "qc", "qi", "qs"],
        "withheld": ["qnc", "qni", "qnwfa", "qnifa", "qnbca", "o3"],
        "flag_qs_true_microphysics_selectors": [6, 8, 9, 10, 16, 18, 28],
        # mp_physics=50 (P3 one-category) is FALSE, and for the substantive
        # reason rather than because it is new: P3 has a single ice category
        # and its Registry package declares moist:qv,qc,qr,qi with NO qs
        # (Registry.EM_COMMON:3038), so WRF's own F_QS is false and MYNN
        # correctly sees sqs = 0.  This is the one case where withholding
        # snow is not a substitution -- there is no snow field to withhold.
        "flag_qs_false_microphysics_selectors": [0, 1, 50],
        "wrf_flag_source": (
            "phys/module_pbl_driver.F:873-878 derives flag_qs from F_QS; "
            "Registry.EM_COMMON declares qs for mp_physics 6, 8, 9, 10, 16, "
            "18 and 28, and does NOT declare it for 50 (P3 has one ice "
            "category, :3038)"),
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
        "aerosol_number_mixing_note": (
            "MYNN scalar plume transport is selected by bl_mynn_mixscalars=1. "
            "WRF post-PBL local diffusion is selected by scalar_pblmix=1 "
            "(phys/module_pbl_driver.F:2251,2641-2844). Both mix the "
            "mp_physics=28 nc/ni/nwfa/nifa family, default to 0 and require "
            "bldt=0. WRF's check_a_mundo disables scalar_pblmix when "
            "bl_mynn_mixscalars=1, so that joint request is refused."),
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
        "wrf-comparison-candidate-v1")
    nssl2_legacy_id = (
        "nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-"
        "wrf-comparison-candidate-v1")
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
    thompson_validation_id = "thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1"
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
        "column reproduced rather than repaired, and the "
        "subnormal-tendency flush on CuPy's -ftz=true compile route.  "
        "Selecting this template selects those.",
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

    # P3 one-category, on the thompson_legacy idiom: the SAME composition
    # with one selector moved (microphysics thompson-mp8 -> p3-mp50), so a
    # paired run of the two isolates the scheme.  Registered because the
    # native-HRRR doors admit mp_physics=50 now
    # (gpuwm/hrrr_route_inputs.SUPPORTED_MICROPHYSICS derives from the
    # ported set) and a scheme no template selects has no root
    # preparation -- the ship-only-what-users-can-reach rule.  Legacy
    # RRTMG is a REQUIREMENT for this scheme's 4/4 pair, not a taste:
    # p3-mp50's own refused_when blocks the RTE+RRTMGP variant (WRF sets
    # has_reqs=0 for P3, no snow radius exists to hand it), so this
    # template pins ra_rrtmg_variant=rrtmg_legacy exactly as its sibling
    # does and is the one full-radiation P3 composition the registry can
    # admit.  moist_cq is pinned True here because the option row does
    # not carry it (unlike thompson-mp8/nssl2-mp18): P3's four moist
    # species feed the same device cq path, and the shipped runtime
    # switches (gpuwm/physics_compat.py) pin the identical value --
    # tests/test_physics_registry.py holds the two equal.
    p3_legacy_id = "p3-mp50-ysu-mm5-noah-rrtmg-legacy-v1"
    p3_legacy = copy.deepcopy(thompson_legacy)
    p3_legacy["components"]["microphysics"] = "p3-mp50"
    p3_legacy["label"] = (
        "P3 one-category + YSU + classic MM5 + Noah + cumulus off + "
        "legacy RRTMG")
    # The composition ceiling: p3-mp50 itself is implemented-unverified
    # (measured against the unmodified P3 v4.5.2 oracle, no composed
    # forecast receipt), so the template cannot rank above it.
    p3_legacy["maturity"] = "implemented-unverified"
    p3_legacy["parameters"]["moist_cq"] = True
    p3_legacy["warnings"] = [
        "Composition candidate: the P3 port is measured against WRF's own "
        "Fortran oracle (see components.microphysics.options.p3-mp50) and "
        "the legacy RRTMG engine is the certified WRF v4.6.1 port, but no "
        "receipt covers the composed suite.  The upgrade payer is named: "
        "the composition's first stock-WRF-paired t0/case receipt is what "
        "moves this label.",
        "This template differs from thompson-mp8-ysu-mm5-noah-rrtmg-"
        "legacy-v1 in exactly ONE component, the microphysics, so the "
        "pair is a controlled scheme comparison; every other parameter, "
        "including the per-domain row, is transcribed from that template.",
        "P3 has ONE ice category: this suite's history carries QIR/QIB "
        "and no QSNOW/QGRAUP/GRAUPELNC fields, exactly as stock WRF's "
        "mp=50 Registry package does.  REFL_10CM is not stashed for "
        "mp=50 on the native-HRRR runner (its native-reflectivity set "
        "does not include 50), so history frames omit it.",
    ]
    registry["templates"][p3_legacy_id] = p3_legacy
    registry["components"]["microphysics"]["options"][
        "p3-mp50"]["reachability"] = {"state": "template"}
    # HRRR-only, on the Kessler rule: the doors that admit mp=50 are the
    # native-HRRR routes, and no other source inherits evidence from them.
    for route in registry["runner_routes"].values():
        for declared in route.get("source_template_ids", {}).values():
            if p3_legacy_id in declared:
                declared.remove(p3_legacy_id)
    for route_id in (
            "tools.hrrr_single_domain_benchmark",
            "tools.prepared_domain_tree_forecast"):
        declared = registry["runner_routes"][route_id][
            "source_template_ids"]["hrrr"]
        declared.insert(declared.index(nssl2_legacy_id) + 1, p3_legacy_id)

    # The radiation-arm siblings of the rows above.  Minted here, after
    # every base they copy exists and before the route declarations are
    # audited, from the table beside _SUITELESS_TEMPLATES.
    _radiation_arm_siblings(registry)
    # And the microphysics-arm siblings, at the same point for the same
    # reason: their bases exist and are declared on their final routes.
    _microphysics_arm_siblings(registry)

    # Owner-ratified declaration: the GFS runner has always advertised this
    # profile and retains the existing Noah-MP route acknowledgement.
    gfs_route = registry["runner_routes"][
        "tools.prepared_single_domain_forecast"]
    gfs_expert = gfs_route.setdefault(
        "expert_template_ids", {}).setdefault("gfs", [])
    noahmp_id = "wsm6-ysu-mm5-noahmp-no-radiation-expert-only-v1"
    if noahmp_id not in gfs_expert:
        gfs_expert.append(noahmp_id)
    # EVERY SOURCE THIS ROUTE SUPPORTS, not gfs alone.  A route that
    # declares any expert list is exhaustive, so the gfs-only declaration
    # made the three Noah-MP templates undeclared on the other seventeen
    # sources: the launcher offered none of them and a plan naming one was
    # refused as off-route instead of raising the acknowledgement advisory
    # the option exists for.  What gates these templates is the route's
    # expert acknowledgement and Noah-MP's own evidence warnings, and both
    # are source-independent -- the scheme is implemented-unverified on
    # every source, not unverified here and verified there.  Derived from
    # the route's own source_ids so a new source inherits the offer
    # instead of quietly losing it.  "mapped" is excluded because it names
    # no model and declares no per-source list at all.
    # The offer is written ONCE, under the route-wide key
    # (gpuwm.physics_registry.EXPERT_TEMPLATES_ANY_SOURCE), which every
    # reader resolves through expert_template_ids_for_source.  It used to
    # be copied under one key per source id, which put seventeen dataset
    # names into a generic table (tests/test_case_token_leakage.py); the
    # per-source copies a tracked registry still carries are removed here
    # so the route's declaration has one shape.
    from gpuwm.physics_registry import EXPERT_TEMPLATES_ANY_SOURCE

    for source_id in gfs_route.get("source_ids", []):
        if source_id != "gfs":
            gfs_route["expert_template_ids"].pop(source_id, None)
    route_wide = gfs_route["expert_template_ids"].setdefault(
        EXPERT_TEMPLATES_ANY_SOURCE, [])
    for template_id in gfs_expert:
        if template_id not in route_wide:
            route_wide.append(template_id)

    # Every graph setting constraint names what it prevents.  The
    # evaluator refuses a reasonless row outright, so these two are
    # written here rather than carried through from the file: they are the
    # only voice on a question a per-domain config cannot see (which
    # domain is the root), and "requires spec_exp=0.0, got 0.33" told a
    # user nothing about the sponge that would be applied.
    _GRAPH_CONSTRAINT_REASONS = {
        ("non-root", "spec_exp"): (
            "the nested lateral-boundary branch carries no exponential "
            "sponge term (WRF dyn_em/module_bc_em.F:1297-1341 applies "
            "spongeweight on the SPECIFIED branch only), while gpuwm's "
            "nested Davies weights do read spec_exp -- so a nonzero value "
            "on a child applies a sponge the branch it transliterates does "
            "not have. The root of this tree may set spec_exp freely; it "
            "is externally forced and takes the specified branch."),
        ("root", "nest_microphysics_transition"): (
            "this setting names the closure used to translate condensate "
            "ACROSS a parent-to-child edge, and the root has no parent "
            "edge to translate over. Select it on the child domain whose "
            "microphysics differs from its parent, where the transition "
            "matrix owns it."),
    }
    for route in registry["runner_routes"].values():
        for constraint in route.get("graph_setting_constraints", []):
            reason = _GRAPH_CONSTRAINT_REASONS.get(
                (constraint.get("scope"), constraint.get("setting_key")))
            if reason is not None:
                constraint["reason"] = reason

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
        (9, MP9_OPTION_ID),
        (10, "morrison-mp10"),
        (18, "nssl2-mp18"),
        # mp=50 joined when its rime-pair mixed-edge closure was ratified
        # into microphysics_transition.PORTED_MP_PHYSICS; mp=16 and mp=28
        # joined with theirs.
        (50, "p3-mp50"),
        (16, WDM6_OPTION_ID),
        (28, MP28_OPTION_ID),
    )
    cross_options = []
    for parent_mp, parent_option in mp_options:
        for child_mp, child_option in mp_options:
            if parent_mp == child_mp:
                continue
            # The p3<->mp9 skip that stood here retired with its defect.
            # It existed because mp=9 was in neither PORTED_MP_PHYSICS nor
            # the named-exclusion tuple, so every published mp9 edge was an
            # over-claim the runtime refused, and the skip only kept the
            # newest pair out of a table that already carried ten of them
            # (audit R-003).  mp=9 is now a ported mixed edge with its own
            # closure, and the assertion below launches every published row
            # through the resolver, so an over-claim fails the BUILD.
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
    # EVERY published cross edge must resolve through the runtime resolver
    # the run will use.  Two authorities said opposite things about ten
    # mp=9 edges -- the registry admitted them at plan review and
    # gpuwm.core.microphysics_transition refused them at nest construction
    # -- and nothing compared the two (audit R-003).  Resolving them here
    # makes the registry's claim and the runtime's answer one statement,
    # checked when the table is written rather than when a nest is built.
    from gpuwm.core.microphysics_transition import (
        resolve_microphysics_transition)
    from types import SimpleNamespace

    for rule in cross_options:
        parent = next(mp for mp, option in mp_options
                      if option == rule["parent_option_id"])
        child = next(mp for mp, option in mp_options
                     if option == rule["child_option_id"])
        # The namespaces carry exactly the settings the rule REQUIRES, so
        # the check is "a run that satisfies this published row resolves",
        # not "some run resolves".
        parent_cfg = SimpleNamespace(
            mp_physics=parent, **rule["required_parent_settings"])
        child_cfg = SimpleNamespace(
            mp_physics=child, **rule["required_child_settings"])
        try:
            resolve_microphysics_transition(parent_cfg, child_cfg)
        except Exception as error:
            raise RuntimeError(
                f"the registry publishes the microphysics edge mp={parent} "
                f"-> mp={child} and gpuwm.core.microphysics_transition "
                f"refuses it at nest construction: {error}") from None

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
    # After every option, template and route the passes above created, and
    # before the maturity/evidence/consumer-row passes below, which walk
    # whatever this one leaves behind.
    for template in registry["templates"].values():
        template["components"]["urban"] = "none"
    # Retire the disabled CQ guard in old registry seeds after every
    # microphysics option is registered. WRF applies calc_cq to every
    # active moist package, including passive vapor. The off option keeps
    # moist=False; dry states bypass the driver.
    for option_id in ("off", "kessler-mp1", "wsm6-mp6"):
        registry["components"]["microphysics"]["options"][option_id][
            "parameters"]["moist_cq"] = True
    _phase2c_route_declarations(registry)
    _every_served_source_declares_a_template_list(registry)
    _phase2c_soil_geometry(registry)
    _phase2c_suiteless_templates(registry)
    _no_radiation_name_warnings(registry)
    # After every option that owns an asset is registered.
    _asset_resolution(registry)
    # Last, so both passes see every surface this builder created above --
    # including the legacy NSSL-2 template and the regenerated nest edges.
    _rename_maturities(registry)
    _evidence_architecture(registry)
    _current_verification_scope(registry)
    # LAST: every option is registered, every constraint written and
    # every maturity renamed, so the consumer rows see the final
    # option set.
    _consumer_rows(registry)
    # After the consumer rows exist, because it reads one of them, and
    # BEFORE the reachability pass below, which prices a way out through
    # the allowed_parameter_keys this one widens.
    _lateral_forcing_remedy_is_reachable(registry)
    # LAST of all: every template and every route declaration above is
    # final, so the easiest path to each option is the one this computes.
    _phase2c_recompute_reachability(registry)
    registry["template_aliases"] = dict(TEMPLATE_ID_ALIASES)
    return registry


#: Every template whose id says "no-radiation" and whose radiation component
#: is ``dudhia-shortwave``.  Recomputed and cross-checked against the built
#: registry below rather than trusted as a list, so a seventh one cannot be
#: added quietly.
_NO_RADIATION_NAMED_TEMPLATE_IDS = (
    "wsm6-mynn-mynn-noah-no-radiation-implemented-unverified-v1",
    "wsm6-mynn-mynn-noahmp-no-radiation-expert-only-v1",
    "wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1",
    "wsm6-ysu-mm5-noah-no-radiation-v1",
    "wsm6-ysu-mm5-noahmp-no-radiation-expert-only-v1",
    "wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1",
)

#: The sentence every one of them now carries, verbatim.
NO_RADIATION_NAME_WARNING = (
    "THE NAME IS WRONG AND THE ID IS FROZEN. This template's id says "
    "'no-radiation'; its radiation component is 'dudhia-shortwave', which "
    "is ra_lw_physics 0 with ra_sw_physics 1 -- WRF's Dudhia SHORTWAVE "
    "scheme RUNS, and no longwave scheme does. Downward longwave is "
    "therefore never computed: GLW stays at zero (WRF's own value for a "
    "longwave-free run, phys/module_physics_init.F:1168-1170 and "
    "phys/module_radiation_driver.F:1719-1722) for the entire forecast, "
    "and every land-surface model reads it every step. That makes this a "
    "DAYTIME-ONLY suite: a real window containing local night "
    "refuses to load unless [experiment] declares acknowledgements = "
    "[\"asymmetric-radiation-nocturnal-window-v1\"]. Read the label, not "
    "the id -- the label has always named Dudhia SW. The id is not being "
    "corrected because it is quoted by shipped configs, evidence receipts "
    "and user plans; this warning, the label and docs/public/PHYSICS.md "
    "are the authority on what the option runs."
)


def _no_radiation_name_warnings(registry: dict) -> None:
    """Make six lying template ids say out loud what they actually run.

    Found 2026-08-09 by execution.  Six shipped templates are named
    ``...-no-radiation-...`` and every one of them selects Dudhia
    shortwave with longwave off -- not radiation off, which is a
    DIFFERENT registered option (``radiation: off``, the (0, 0) pair).
    The gap is not theoretical: ``configs/real74_4dom_mynn_norad.toml``
    reached for ``ra_physics = 0`` and explained itself with "the MYNN
    5/5 registry template ... carries ra_physics 0", which the template
    does not, and ``docs/public/STREAMING.md`` handed
    ``wsm6-ysu-mm5-noah-no-radiation-v1`` to users in a copy-paste
    streaming plan while that template's ``warnings`` list did not exist
    and its maturity, ``supported``, is one of
    ``warning_policy.nonwarning_maturities`` -- so selecting it produced
    no warning of any kind at any door.

    Maturity is deliberately NOT touched.  The ladder's axis is
    ``conformance`` and every component of this suite is measured; saying
    ``experimental-runtime`` would be a false claim about port fidelity
    to buy a warning through the maturity channel.  ``warnings`` is the
    channel that exists for "informed user review"
    (``warning_policy.meaning``) and it is emitted for every template
    regardless of maturity
    (``gpuwm/physics_registry.py`` template-warning pass), so it is the
    accurate lever and it is the one used here.

    Appended last and appended (not inserted at 0) on purpose: this
    builder composes ``mynn_ruc`` and ``mynn_noahmp`` by copying another
    template's ``warnings[0]``, so a leading insert would silently
    reassign which sentence those two inherit.
    """

    templates = registry["templates"]
    named = tuple(sorted(
        template_id for template_id in templates
        if "no-radiation" in template_id))
    if named != _NO_RADIATION_NAMED_TEMPLATE_IDS:
        raise SystemExit(
            "the set of templates whose id says 'no-radiation' moved: "
            f"expected {_NO_RADIATION_NAMED_TEMPLATE_IDS}, built {named}. "
            "Update _NO_RADIATION_NAMED_TEMPLATE_IDS deliberately -- a new "
            "member needs this warning, and a departed one must not be "
            "silently dropped from it.")
    for template_id in named:
        template = templates[template_id]
        radiation = template.get("components", {}).get("radiation")
        if radiation != "dudhia-shortwave":
            raise SystemExit(
                f"template {template_id!r} is named 'no-radiation' and "
                f"selects radiation {radiation!r}; this warning states "
                "'dudhia-shortwave' and would now be the lie it exists to "
                "correct")
        warnings = template.setdefault("warnings", [])
        if NO_RADIATION_NAME_WARNING not in warnings:
            warnings.append(NO_RADIATION_NAME_WARNING)


#: Audit R-021: the one route whose immutable templates are the point.
_BENCHMARK_OVERRIDE_REFUSAL = (
    "its published wall-clock and skill numbers are only "
    "comparable against the immutable template they were measured on, so a "
    "component override would silently retire the comparison the route "
    "exists to publish. Run the same suite on "
    "tools.prepared_single_domain_forecast, which declares the same "
    "component overrides as the domain-tree route.")

#: Audit R-021, per component.  Every implemented option a per-domain
#: route's lists exclude is keyed here with the breakage that exclusion
#: prevents.  Keyed per component because a refusal is delivered about ONE
#: component: a route-wide essay pasted into every refusal told a user
#: naming land_surface what is wrong with an analytic radiation scheme, and
#: restated the admitted lists the surrounding sentence already names.
_PER_DOMAIN_EXCLUSION_REASONS = {
    **{("urban", option): (
        "WRF runs one urban selector on every domain (it resets them all to "
        "the innermost domain's value, share/module_check_a_mundo.F:"
        "1063-1077); set sf_urban_physics in [shared], not per domain.")
       for option in ("none", "slucm", "bep", "bep-bem")},
    ("radiation", "analytic-clear-sky"): (
        "analytic-clear-sky is not a forecast radiation scheme: it computes "
        "a clear-sky flux from solar geometry alone and carries no cloud, "
        "aerosol or gas optics, so every cloudy column's heating is wrong by "
        "the whole cloud effect"),
}

#: Where each route sends the value instead, per component.  The exclusion
#: above is the same physical fact on both routes; the way out is not,
#: because only one of them declares expert selectors.
_TREE_ROUTE_OVERRIDE_WAY_OUT = {
    "radiation": (
        "; this route declares ra_lw_physics and ra_sw_physics in "
        "allowed_expert_selector_keys, which is where a dycore or idealized "
        "experiment writes 90/90"),
}
_SINGLE_ROUTE_OVERRIDE_WAY_OUT = {
    "radiation": (
        "; this route declares no expert selectors, so write it in the "
        "hash-bound experiment config, or run the experiment on "
        "tools.prepared_domain_tree_forecast, which declares ra_lw_physics "
        "and ra_sw_physics as expert selectors"),
}

#: Audit R-059: what each single-domain route refuses an expert SETTING
#: for.  Two sentences, not one, because the two routes stopped refusing
#: the same things when R-021 widened one of them: a component override is
#: accepted on tools.prepared_single_domain_forecast and refused on the
#: benchmark route, so a shared "invalidates the seal" sentence no longer
#: distinguished what is refused from what is allowed.
_BENCHMARK_ROUTE_EXPERT_REFUSAL = (
    "sealed-proof benchmark route: this runner replays ONE immutable "
    "template and verifies it against "
    "--proof-sha256/--prepared-content-sha256, and it varies NOTHING -- an "
    "expert setting and a component override are refused alike, because "
    "either one retires the comparison the published numbers were measured "
    "against. State the value in the hash-bound experiment config and run "
    "it on tools.prepared_domain_tree_forecast, whose declaration lists "
    "the parameters its loader accepts.")
_SINGLE_DOMAIN_ROUTE_EXPERT_REFUSAL = (
    "single-domain route: it runs ONE domain, so there is no per-domain "
    "override table for a plan-level expert setting to be carried in "
    "(gpuwm/experiment.py _DOMAIN_RUN_OVERRIDES is the tree loader's), and "
    "every value this runner uses comes from the experiment config it "
    "hash-binds before step 0. State the value THERE, where "
    "gpuwm.config.validate_run_config checks it, or run the plan on "
    "tools.prepared_domain_tree_forecast, whose declaration lists the "
    "parameters its loader accepts. A COMPONENT choice is a different "
    "question and this route admits the ones it declares: the seal covers "
    "the prepared input and its preparation proof, not the suite, and the "
    "2026-07-31 owner ruling removed this runner's profile whitelist so it "
    "resolves any engine-valid suite from that same hash-bound config.")


def _constraints(option: dict) -> dict:
    """``option['constraints']`` as a dict, created when it is JSON null."""

    current = option.get("constraints")
    if not isinstance(current, dict):
        current = {}
        option["constraints"] = current
    return current


def _reachability(option: dict) -> dict:
    """``option['reachability']`` as a dict, created when it is missing.

    A caller here writes only the BLOCKER.  The state is derived by
    :func:`_phase2c_recompute_reachability` from the declared template and
    override lists, so that "how do I select this" and "why can I not" are
    never two hand-set opinions about the same option.
    """

    current = option.get("reachability")
    if not isinstance(current, dict):
        current = {}
        option["reachability"] = current
    return current


#: Knobs the domain-tree loader accepts per domain and this route does NOT
#: offer at plan review, each with the concrete refusal that keeps it out
#: (audit R-059).  Every one is a coupling that lives only in
#: ``gpuwm.config.validate_run_config``: the registry can say "this option
#: requires that setting" and "this option is refused when that setting
#: holds", and none of these is scoped to a component -- they are
#: parameter-to-parameter.  Offering them here without the rule would make
#: plan review call launchable a plan the loader refuses, which is the
#: exact drift class this audit exists to close, so the omission is
#: DECLARED with its reason instead.
#:
#: FOLLOW-UP, named: encode each as a component-scoped ``refused_when``
#: rule and delete its row here.  The gate that will notice is
#: tests/test_authority_agreement.py::
#: test_every_reachable_plan_the_registry_calls_launchable_actually_validates.
_COUPLINGS_NOT_YET_REGISTRY_ROWS = {
    "diff_6th_opt": (
        "diff_6th_opt=1 is non-monotonic and gpuwm.config refuses it with "
        "moist=true, because the unlimited 6th-order fluxes bypass the "
        "positive-definite transport limiter and can drive negative "
        "moisture. The coupling is to moist, not to any component, so no "
        "option row can carry it yet. Set it in the experiment config, "
        "where validate_run_config checks the pair."),
    "clos_choice": (
        "clos_choice and ishallow are Grell-family keys, read only where "
        "cu_physics=3; gpuwm.config refuses a nonzero value beside any "
        "other cumulus scheme. Select Grell-Freitas and set them in the "
        "experiment config."),
    "ishallow": (
        "clos_choice and ishallow are Grell-family keys, read only where "
        "cu_physics=3; gpuwm.config refuses a nonzero value beside any "
        "other cumulus scheme. Select Grell-Freitas and set them in the "
        "experiment config."),
    "o3input": (
        "o3input and use_mp_re are honoured only by the legacy RRTMG "
        "engine; the modern spectrum does not implement the nondefault "
        "ozone and effective-radius operations, and gpuwm.config refuses "
        "the pair. Select ra_rrtmg_variant='rrtmg_legacy' in the "
        "experiment config and set them there."),
    "use_mp_re": (
        "use_mp_re is honoured only by the legacy RRTMG engine; see "
        "o3input. Set it in the experiment config beside "
        "ra_rrtmg_variant='rrtmg_legacy'."),
    "ra_rrtmg_variant": (
        "the engine and the compatibility token must agree, and "
        "gpuwm.config refuses a variant that contradicts the resolved "
        "wrf_rrtmg_compatibility. A per-domain override of one without the "
        "other is exactly that contradiction, so the pair is set together "
        "in the experiment config."),
    "wrf_rrtmg_compatibility": (
        "the compatibility token records a 4/4 substitution and "
        "gpuwm.config refuses it unless the domain resolves the 4/4 pair. "
        "Selecting the radiation option that resolves that pair is how a "
        "domain asks for it; the token follows the option rather than "
        "varying against it."),
}


def _phase2c_route_declarations(registry: dict) -> None:
    """Audit phase 2c: route declarations stop refusing what the runners run.

    Section 3 of the 2026-09-10 physics-glue audit found every refusal in
    this area names no breakage, which under the gate law means it does
    not exist: a blanket "fixed-template runner accepts only its
    immutable template_id" on a route whose own runner applies registry
    governance as a warning and runs (R-021); a land-surface axis simply
    absent from the tree route's option lists, with no blocker text
    anywhere, while every other axis carries a written reason for each
    option it lists AND for the one it excludes (R-022); three options
    the desktop offers and the engine accepts, declared unreachable for
    parity reasons that were retired (R-023); a uniform-base-template
    check that compares template LABELS while the same route declares
    per-domain variation legal for nearly every component (R-058); an
    expert-override refusal on all three routes with no stated reason and
    42 registered parameters stranded behind it (R-059); and a relocating
    nest tree with no declared topology, which is a sequence of static
    one-way trees and changes nothing plan review validates (R-056).

    Everything here is ROWS.  A scheme is rows, and so is a route
    capability: adding one is table work in this builder, never a code
    path in the validator.
    """

    components = registry["components"]
    routes = registry["runner_routes"]
    tree_route = routes["tools.prepared_domain_tree_forecast"]
    single_route = routes["tools.prepared_single_domain_forecast"]
    benchmark_route = routes["tools.hrrr_single_domain_benchmark"]
    land_options = components["land_surface"]["options"]
    surface_options = components["surface_layer"]["options"]
    radiation_options = components["radiation"]["options"]
    parameters = registry["parameters"]

    # -- R-023 ---------------------------------------------------------
    # Selecting land_surface "off" resolved no num_soil_layers at all, so
    # whatever count the plan was carrying reached
    # gpuwm.config.soil_layer_count, which refuses a count the selected
    # scheme does not define.  Scheme 0's geometry is Noah's four
    # (gpuwm/config.py LAND_SURFACE_SOIL_LAYERS._PROVIDERS[0] ->
    # gpuwm.core.noah.NUM_SOIL_LAYERS, which is also what
    # NO_LAND_SURFACE_SOIL_LAYERS resolves to), so the option states its
    # geometry the way every other land surface does.
    _constraints(land_options["off"]).setdefault(
        "required_settings", {})["num_soil_layers"] = 4
    land_options["off"]["warnings"] = [
        "No land-surface scheme runs: TSK, the soil column and the surface "
        "fluxes below the surface layer stay at their initial values for the "
        "whole forecast.  Every registered source supplies a soil state, so "
        "this is a deliberate degradation -- an idealized or LES composition "
        "-- rather than a saving."
    ]
    surface_options["off"]["warnings"] = [
        "No surface layer runs: nothing computes a friction velocity or the "
        "heat and moisture fluxes, so a land-surface scheme or a PBL closure "
        "selected beside it is refused by gpuwm/core/physics.py -- that "
        "refusal names its own breakage and stands.  What remains legal is a "
        "dry no-physics column."
    ]

    # -- R-059: two couplings that ARE component-scoped, as rows -------
    # cudt is a Kain-Fritsch cadence knob.  Grell-Freitas and New Tiedtke
    # both run on the model step and carry no NCA hold, so gpuwm.config
    # refuses a nonzero cudt beside either -- and each option already
    # PINS cudt_minutes=0 in its own parameter block, which a per-domain
    # override could silently move.  Stating it as a required setting is
    # what makes plan review refuse the override the loader refuses.
    for option_id in ("grell-freitas", "new-tiedtke"):
        _constraints(components["cumulus"]["options"][option_id]).setdefault(
            "required_settings", {})["cudt_minutes"] = 0.0
    # Every radiation option resolves its spectra through the split
    # ra_lw_physics/ra_sw_physics selectors, and gpuwm.config refuses a
    # plan that also sets the legacy aggregate ra_physics to anything but
    # 0 ("do not mix split and legacy radiation selection").  Each option
    # already carries ra_physics=0 in its parameters; the required setting
    # is what refuses a per-domain override of it.
    for option in components["radiation"]["options"].values():
        if (option.get("parameters") or {}).get("ra_physics") == 0:
            _constraints(option).setdefault(
                "required_settings", {})["ra_physics"] = 0

    # -- R-005: the one per-source refusal, as a row -------------------
    # Microphysics off on a real-source route carried TWO refusals fused
    # into one prose error.  The first is per-source, and it is a row now:
    # native HRRR supplies analyzed QC/QR/QI/QS/QG and MP off cannot
    # faithfully retain it, which is why the run died in
    # gpuwm/ingest/real.py rather than at review.  It is refused before
    # step 0, and a source that gains such an incompatibility is table
    # work.
    #
    # The second is the moist carrier, and it STAYS an error at review.
    # This pass briefly resolved it instead -- plan review defaulted
    # moist=true and warned -- and that was a resolution nothing performs:
    # validate_physics_plan reports, while the RunConfig a runner builds
    # takes moist from the microphysics-off option's own row and from the
    # experiment config, so review would have called launchable a plan
    # gpuwm/ingest/real.py refuses before step 0.  What the attempt got
    # right is that the remedy has to be expressible on the route it is
    # prescribed to: the old text prescribed one two of the three routes
    # could not express.  Each route now declares WHERE its moist value
    # lives, and the refusal reads that declaration out.
    mp_off_constraints = _constraints(components["microphysics"]["options"]["off"])
    # Rebuilt, not appended: this builder transforms the tracked registry in
    # place, so a rule appended on every run would multiply.
    mp_off_constraints["refused_when"] = [
        rule for rule in mp_off_constraints.get("refused_when", [])
        if not isinstance(rule, dict) or "sources" not in rule
    ]
    mp_off_constraints["refused_when"].append({
        "sources": ["hrrr"],
        "reason": (
            "native HRRR supplies analyzed QC/QR/QI/QS/QG, and microphysics "
            "off cannot faithfully retain them: no radiation-only "
            "analyzed-cloud carrier is implemented, so the analyzed "
            "condensate would be dropped at initialization and the forecast "
            "would run clear where the analysis was cloudy. Select a "
            "microphysics scheme, or run microphysics off on a source whose "
            "analysis carries no condensate."),
    })
    registry["authority"]["real_source_moisture_contract"] = (
        "runner_routes.<runner>.requires_moist_real_initialization declares "
        "that every source on the route enters gpuwm.ingest.real and "
        "therefore requires a moist state even when microphysics is off. "
        "Plan review REFUSES such a plan until moist=true is stated, "
        "because nothing between review and the loader rewrites the "
        "microphysics-off option's own moist=false: the refusal names "
        "runner_routes.<runner>.moist_declaration_site, which is where that "
        "route's value lives, so the remedy is expressible on the route it "
        "is prescribed to. An explicit moist=false is refused for the same "
        "reason, naming what a dry column would drop. Per-source "
        "incompatibilities are rows -- "
        "components.<id>.options.<id>.constraints.refused_when entries "
        "carrying a sources clause -- so they are refused at review rather "
        "than inside the preparation.")
    # WHERE each route's moist value lives.  A refusal that prescribes a
    # door the route does not have is the defect this key exists to
    # prevent; it is a declaration, so a new route answers the question by
    # adding a row rather than by teaching the validator about itself.
    for route_id, site in (
            ("tools.prepared_domain_tree_forecast",
             "in the plan's per-domain parameters -- this route declares "
             "moist in allowed_parameter_keys -- or in the hash-bound "
             "experiment config the tree loader reads"),
            ("tools.prepared_single_domain_forecast",
             "in the hash-bound experiment config this route runs from, "
             "whose physics keys carry the same names"),
            ("tools.hrrr_single_domain_benchmark",
             "by selecting one of this route's immutable templates: every "
             "one of them runs a microphysics scheme, so this route has no "
             "microphysics-off composition to carry a moist value"),
    ):
        registry["runner_routes"][route_id]["moist_declaration_site"] = site

    # -- R-022 + R-023: the tree route's option lists -------------------
    # land_surface was the one component axis with NO entry at all: no
    # option list, no blocker text, no physical statement anywhere, while
    # every other axis carries a written reason for each option it lists
    # and for the one it excludes.  RUC is a shipped land surface with a
    # runtime gate, a device-path two-sided gate and three declared ERA5
    # templates; its one named consequence under mp_physics=9 is a
    # fidelity divergence carried as the option's own warning, not a
    # failure.
    #
    # Implemented land options retain their own state and coupling constraints.
    # The former throughput acknowledgement is advisory and cannot exclude one.
    tree_route["allowed_component_options"]["land_surface"] = sorted(
        option_id for option_id, option in land_options.items()
        if option.get("implemented") is True)
    # "off" joins the surface-layer list on the same terms: the engine
    # accepts it, the desktop already offers it, and the combinations that
    # would make it dangerous are refused by gpuwm/core/physics.py with
    # their own named breakage.
    tree_route["allowed_component_options"]["surface_layer"] = [
        "off", "revised-mm5", "classic-mm5", "mynn", "eta-similarity"]
    # analytic-clear-sky is the third member of R-023's trio, and it is
    # the one this block does NOT add to a per-domain option list.  Its
    # STATE is not hand-set either way: _phase2c_recompute_reachability
    # derives it, and the derivation answers component-override for a
    # reason worth stating, because it is the reason the audit found the
    # old "unreachable" label untrue.  The tree route declares
    # ra_lw_physics and ra_sw_physics in allowed_expert_selector_keys, so
    # writing 90/90 through expert_overrides.selectors reaches this
    # option on that route today.  Calling it unreachable while a
    # declared door opens it is exactly the drift this pass exists to
    # end; the degradation is carried by the warning below instead, which
    # is what a user actually reads.
    #
    # The BLOCKER is rewritten rather than kept, so that if the option
    # ever does become unreachable it names a breakage in ArWen's own
    # terms.  The old text -- "WRF v4.6.1 registers no equivalent
    # package" -- is a PARITY reason, and parity stopped being the
    # referee.
    analytic = radiation_options["analytic-clear-sky"]
    _reachability(analytic)["blocker"] = (
        "The 90/90 proxy is not a forecast radiation scheme: it computes "
        "a clear-sky flux from solar geometry alone and carries no cloud, "
        "aerosol or gas optics, so in any cloudy column the radiative "
        "heating profile is wrong by the entire cloud effect -- surface "
        "shortwave too high by day, longwave cooling too strong at night. "
        "It is a dycore and idealized instrument, which is why no template "
        "selects it and no per-domain option list admits it.")
    analytic["warnings"] = [
        "ARWEN-SPECIFIC, NOT A WRF SCHEME, AND NOT A FORECAST PRODUCT. The "
        "90/90 analytic clear-sky proxy carries no cloud, aerosol or gas "
        "optics: every cloudy column's radiative heating is wrong by the "
        "whole cloud effect. WRF v4.6.1 registers no equivalent package "
        "(Registry.EM_COMMON:3107-3125), so a run selecting it also has no "
        "stock-WRF counterpart to be compared against."
    ]

    # -- R-022: the RUC warning sentence the shipped tree contradicts ---
    # gpuwm/ingest/soil_contract.py carries RUC_TARGET_LEVEL_DEPTHS_M (nine
    # levels) with an import-time drift check against
    # gpuwm.ingest.ruc_soil.RUC_LEVEL_DEPTHS_M, plus RUC_REMAP_POLICIES.
    # The sentence claiming a composition that ships a soil_layer_contract
    # "cannot ask for RUC" was true before that landed and is false now.
    # The GFS withdrawal it also records was root-caused rather than
    # re-gated: a shoreline land column carrying the water soil category
    # reached soilvegin, which has no arm for it, and 0./0. went into
    # MAVAIL. Every ArWen door reconciles that column now
    # (gpuwm/ingest/soil.py door_reconciled_soil_category), and
    # gpuwm/prepared_single_domain_forecast.py records that no route
    # blocker remains and none is enforced.
    #
    # Rewritten IN PLACE rather than dropped: _surface_coupling_warnings
    # above addresses this list by index, so removing a member would move
    # every warning after it.
    ruc = land_options["ruc-lsm"]
    ruc["warnings"] = [
        warning.replace(
            "WHAT IS STILL REFUSED is the DECLARATIVE contract: "
            "gpuwm/ingest/soil_contract.py validate_soil_layer_contract "
            "declares exactly one target, Noah's four layers, so a "
            "composition that ships a soil_layer_contract -- which is the "
            "20crv3/mapped path -- cannot ask for RUC.",
            "THE DECLARATIVE CONTRACT REACHES RUC TOO: "
            "gpuwm/ingest/soil_contract.py carries "
            "RUC_TARGET_LEVEL_DEPTHS_M and RUC_REMAP_POLICIES beside "
            "Noah's four layers, with an import-time drift check against "
            "gpuwm.ingest.ruc_soil.RUC_LEVEL_DEPTHS_M, so a composition "
            "that ships a soil_layer_contract can target RUC.",
        ).replace(
            "The template is offered through the direct HRRR and ERA5 "
            "preparations; it remains absent from 20crv3/mapped "
            "preparation, and from GFS preparation. The GFS withdrawal is "
            "a v1.1.1 field finding rather than an ingest-contract one: a "
            "GFS-initialised RUC forecast PREPARES cleanly -- proof PASS, "
            "nine soil layers -- and then dies on its first "
            "surface-temperature call with `mavail must be finite` "
            "(gpuwm/core/ruc.py:_horizontal_float_field) having advanced "
            "no model time, so the GFS route's initialised land and soil "
            "state does not reach ruc_cold_start in a condition RUC can "
            "integrate from. Completing that initialisation is a v1.2 "
            "item.",
            "The GFS withdrawal that once stood here is RETIRED, and was "
            "root-caused rather than re-gated: a shoreline land column "
            "carrying the water soil category (SOILTYP 14 under a land "
            "LU_INDEX) reached soilvegin, which has no arm for it, and "
            "0./0. went into MAVAIL. WRF's own real program reconciles "
            "that column at initialization and so does every ArWen door, "
            "through gpuwm/ingest/soil.py door_reconciled_soil_category "
            "(proven both ways by "
            "tests/test_ruc_shoreline_soil_category.py). No route blocker "
            "remains and none is enforced.",
        )
        for warning in ruc.get("warnings", [])
    ]

    # -- R-021: the single-domain route mirrors the tree route ----------
    # The 2026-07-31 owner ruling removed the profile whitelist from this
    # route's runner: gpuwm/prepared_single_domain_forecast.py records the
    # removal, labels its own per-source lists "REPORTED METADATA, NOT A
    # GATE", applies registry governance as a WARNING and runs.  The route
    # DECLARATION was never widened to match, so plan review refused what
    # the runner runs -- and the trigger configuration integrated 59
    # minutes on this exact route with two components the declaration says
    # cannot vary. Mirroring the tree route retains each option's actual
    # field and shared-setting constraints, evaluated by the same authority.
    single_route["allowed_component_overrides"] = list(
        tree_route["allowed_component_overrides"])
    single_route["allowed_component_options"] = {
        component_id: list(option_ids)
        for component_id, option_ids
        in tree_route["allowed_component_options"].items()
    }
    # The benchmark route stays closed, and now says why.  An empty
    # declaration with no reason is a refusal that names no breakage; this
    # is the one route where immutability IS the product.
    benchmark_route["allowed_component_overrides"] = []
    benchmark_route["allowed_component_options"] = {}
    benchmark_route["component_override_refusal_reason"] = (
        _BENCHMARK_OVERRIDE_REFUSAL)

    # The refusal that actually FIRES on the two per-domain routes had no
    # reason at all: only the benchmark route carried one, so the single
    # option the widened lists deliberately exclude came back as "runner
    # route does not allow this component option to vary per domain" and
    # stopped there.  Every option a route excludes is named here with its
    # way out, and the exclusion set is computed rather than trusted, so a
    # list that moves cannot leave this sentence describing the old one.
    excluded = {
        (component_id, option_id)
        for component_id, component in components.items()
        if component_id not in tree_route["allowed_component_overrides"]
        for option_id, option in component["options"].items()
        if option.get("implemented") is True
        and option_id not in tree_route[
            "allowed_component_options"].get(component_id, [])
    }
    if excluded != set(_PER_DOMAIN_EXCLUSION_REASONS):
        raise SystemExit(
            "the per-domain option lists exclude "
            f"{sorted(excluded)}; _PER_DOMAIN_EXCLUSION_REASONS keys "
            f"{sorted(_PER_DOMAIN_EXCLUSION_REASONS)} and nothing else. "
            "Give each new exclusion its own breakage and way out in that "
            "table, or add the option to the list.")
    # One reason per COMPONENT, because one refusal is about one component.
    for route, route_way_out in ((tree_route, _TREE_ROUTE_OVERRIDE_WAY_OUT),
                                 (single_route, _SINGLE_ROUTE_OVERRIDE_WAY_OUT)):
        reasons: dict[str, str] = {}
        for (component_id, _option), reason in sorted(
                _PER_DOMAIN_EXCLUSION_REASONS.items()):
            clause = reason + route_way_out.get(component_id, "")
            if reasons.setdefault(component_id, clause) != clause:
                raise SystemExit(
                    f"component {component_id!r} excludes two implemented "
                    "options with different reasons; one refusal carries one "
                    "component's reason, so merge them into one clause or "
                    "key the table by option as well.")
        route["component_override_refusal_reasons"] = reasons
        # Specific component reasons take precedence. Other components need
        # this route's general declaration, not an inferred benchmark reason.
        # Replace on every build so obsolete wording cannot survive a rebuild.
        route["component_override_refusal_reason"] = (
            component_override_declaration(route))

    # -- R-059: expert overrides ---------------------------------------
    # allowed_parameter_keys and allowed_expert_selector_keys are DERIVED
    # from the loader that actually accepts them, so a parameter the tree
    # loader gains becomes settable at plan review in the same pass.  Hand
    # lists are how 42 registered parameters carrying a named
    # consuming_read came to be settable through no route at all, while
    # the validator's own message pointed at expert_overrides.settings --
    # a door every route nailed shut.
    from gpuwm.experiment import _DOMAIN_RUN_OVERRIDES

    selector_keys = {
        key
        for component in components.values()
        for key in component.get("selector_keys", ())
    }
    # A knob EVERY option of one component pins in ``required_settings`` is
    # that component's own identity spelling, not a free per-domain value: a
    # plan overriding it either restates the option it already selected or
    # contradicts it, and plan review then refuses a configuration
    # gpuwm.config.validate_run_config admits -- two doors disagreeing about
    # one file.  ``ra_physics`` is the live case, the pre-split spelling of
    # the radiation pair, which every radiation option states as 0: the
    # aggregate spelling of a pair resolves to that pair at the capability
    # door (gpuwm.physics_compat._one_radiation_spelling), so no option is
    # selected by the aggregate key.
    # Derived rather than listed, so a second alias is excluded the day it
    # is registered.
    identity_keys: set[str] = set()
    for component in components.values():
        pinned = None
        for option in (component.get("options") or {}).values():
            required = set(
                (option.get("constraints") or {}).get("required_settings")
                or ())
            pinned = required if pinned is None else (pinned & required)
        identity_keys |= set(pinned or ())
    per_domain_loader_keys = frozenset(_DOMAIN_RUN_OVERRIDES)
    tree_route["allowed_parameter_keys"] = sorted(
        name for name, spec in parameters.items()
        if spec.get("implemented") is not False
        and name in per_domain_loader_keys
        and name not in selector_keys
        and name not in identity_keys
        and name not in _COUPLINGS_NOT_YET_REGISTRY_ROWS)
    # Published on the route, so the declaration a door reads carries the
    # reason a loader-accepted knob is not offered here.  A refusal that
    # names no breakage does not exist; neither does an omission.
    tree_route["deferred_parameter_keys"] = {
        name: reason
        for name, reason in sorted(_COUPLINGS_NOT_YET_REGISTRY_ROWS.items())
        if name in per_domain_loader_keys and name in parameters
    }
    for name in sorted(identity_keys & per_domain_loader_keys & set(parameters)):
        component_id = next(
            cid for cid, component in components.items()
            if all(name in ((option.get("constraints") or {}).get(
                       "required_settings") or ())
                   for option in (component.get("options") or {}).values()))
        tree_route["deferred_parameter_keys"][name] = (
            f"every {component_id} option states {name}, so it spells WHICH "
            f"option is selected rather than a value inside one. Select the "
            f"{component_id} option that carries the value you want; "
            f"overriding {name} per domain could only restate or contradict "
            "the option already chosen.")
    tree_route["allowed_expert_selector_keys"] = sorted(
        selector_keys & per_domain_loader_keys)
    # The two sealed routes keep empty lists, and now name what the
    # emptiness protects: the proof they replay.
    for route, refusal in (
            (single_route, _SINGLE_DOMAIN_ROUTE_EXPERT_REFUSAL),
            (benchmark_route, _BENCHMARK_ROUTE_EXPERT_REFUSAL)):
        route["allowed_parameter_keys"] = []
        route["allowed_expert_selector_keys"] = []
        route["expert_override_refusal_reason"] = refusal
    tree_route["expert_override_refusal_reason"] = (
        "the domain-tree loader accepts the parameters and selectors this "
        "route declares; an expert SETTING outside them would reach "
        "gpuwm.config.validate_run_config with no registry row to validate "
        "it against. Register the knob as a parameter -- table work in "
        "tools/build_registry.py -- rather than passing it through.")

    # -- R-058: per-domain declarations, and a value check --------------
    # The uniform-base-template check compared template LABELS, so two
    # templates whose component maps are byte-identical and differ only in
    # a per-domain key were refused for having different names.  The
    # narrow thing it stood in for is real: sf_surface_physics, its soil
    # geometry and the Noah-MP option block are deliberately outside the
    # loader's per-domain table, so ONE land surface runs for a whole
    # tree.  That is now stated per parameter and checked on resolved
    # VALUES.
    for name, spec in parameters.items():
        spec["per_domain"] = name in per_domain_loader_keys
    for component in components.values():
        selectors = {
            key: key in per_domain_loader_keys
            for key in component.get("selector_keys", ())
        }
        if selectors:
            component["per_domain_selectors"] = selectors
    tree_route.pop("template_policy", None)
    registry["authority"]["per_domain_declaration"] = (
        "parameters.<name>.per_domain and components.<id>."
        "per_domain_selectors declare which knobs the domain-tree loader "
        "accepts inside a [[domain]] table (gpuwm/experiment.py "
        "_DOMAIN_RUN_OVERRIDES, which this registry is generated from). "
        "validate_physics_plan refuses a tree whose domains resolve "
        "DIFFERENT values for a knob declared per_domain false, naming both "
        "values and the loader that cannot express them; it no longer "
        "compares template ids, which are labels.")

    # -- R-056: a relocating tree is a sequence of static one-way trees --
    # Nothing plan review validates changes when a nest moves: the
    # topology stays one-way nested, the transition policy stays the
    # microphysics edge policy, and relocation is a runtime schedule.  The
    # route says so rather than leaving a relocating plan with no declared
    # topology at all.  No second topology id is minted -- a receipt naming
    # one would claim a second closure exists.
    tree_route["relocation"] = {
        "supported": True,
        "topology_id": "one-way-nested-v1",
        "declaration": (
            "A relocating child is a SEQUENCE of static one-way nested "
            "trees: at every instant the tree validated is the one this "
            "topology id names, and relocation moves the child's parent "
            "window between instants. Plan review validates the instant; "
            "the schedule is runtime state (gpuwm/core/nest_spawn.py, and "
            "the follow/target itineraries in gpuwm/companion_domains.py)."),
        "unchanged_by_relocation": [
            "topology_ids",
            "transition_policy_id",
            "graph_setting_constraints",
        ],
    }


def _phase2c_soil_geometry(registry: dict) -> None:
    """Audit R-038: one soil geometry, one authority.

    Two authorities described the same geometry and disagreed.  This
    registry's ``num_soil_layers`` enum was ``[4, 9]`` and is enforced as a
    HARD plan error, while ``gpuwm/config.py``'s ``soil_layer_count``,
    built from ``gpuwm.core.ruc_contract.WRF_SUPPORTED_NUM_SOIL_LAYERS``
    (which is itself tabulated from ``gpuwm.ingest.ruc_soil``), accepts six
    and a six-level forecast completes with ``soil_layers_stag = 6`` in its
    wrfout.  The registry's stated reason for the refusal was "a schema
    enum entry would advertise a validated geometry" -- an EVIDENCE
    POLICY, not a physical or numerical incompatibility, so under the gate
    law it is not a refusal at all.

    The enum is now DERIVED from the module that owns the geometry, so a
    count the soil column gains is table work here rather than a second
    hand-typed opinion.  What six actually lacks -- a WRF FORECAST oracle,
    every lsmruc/sfctmp/soilmoist/snowtemp fixture in the tree being
    nine-level -- is said in the warning a user reads, beside the receipt
    field that carries it, which is what an absence of evidence is worth.

    DELIBERATELY NOT DONE HERE, and handed back as a named follow-up: the
    ``ruc-lsm-6level`` OPTION row the audit note proposes.  Land-surface
    options are keyed by ``sf_surface_physics`` throughout the authority
    lane that landed after the audit was written -- ``implemented_selector_
    values``, ``consumer_rows_by_selector`` and ``_option_for_selectors``
    all index by selector value -- so a second option carrying selector 3
    would silently shadow the first in every consumer table derived from
    those.  Admitting six-level RUC as a named option needs an inventory
    keyed by (selector, geometry), which is a design change rather than
    table work, and it rides on the owner decision the audit files as its
    open question 5.  Six-level RUC keeps the door it has: the hash-bound
    experiment config, which ``gpuwm.config.soil_layer_count`` admits and
    ``gpuwm/core/ruc_contract.py`` sizes.
    """

    from gpuwm.core.ruc_contract import WRF_SUPPORTED_NUM_SOIL_LAYERS

    spec = registry["parameters"]["num_soil_layers"]
    # 4 is Noah/Noah-MP's single geometry (gpuwm/core/noah.NUM_SOIL_LAYERS);
    # the rest is every count RUC's own level table defines.
    spec["enum"] = sorted({4, *(int(count)
                                for count in WRF_SUPPORTED_NUM_SOIL_LAYERS)})
    spec["warnings"] = [
        warning.replace(
            "This ENUM declares 4 and 9 -- 4 with Noah "
            "or Noah-MP, 9 with RUC -- because those are the geometries an "
            "oracle has judged; the enum is a statement about EVIDENCE, not "
            "the whole set of selectable values.",
            "This ENUM declares every geometry a plan may STATE, derived "
            "from gpuwm.core.ruc_contract.WRF_SUPPORTED_NUM_SOIL_LAYERS and "
            "Noah's single count, because a schema that refused a value the "
            "shipped resolver accepts was two authorities disagreeing about "
            "one geometry. Which count each SCHEME resolves is a separate "
            "question, answered by soil_layer_count above; how much evidence "
            "each geometry carries is a third, answered here and in the run "
            "receipt.",
        ).replace(
            "A schema enum entry would advertise a validated geometry; "
            "see docs/wrf_ruc_runtime_admission.md.",
            "Six carries no WRF FORECAST oracle -- every "
            "lsmruc/sfctmp/soilmoist/snowtemp fixture in the tree is "
            "nine-level -- so a run at six carries "
            "soil_geometry_evidence=internal-consistency-only in its "
            "receipt and warns at runtime. What IS measured at six: ZS/DZS "
            "reproduce WRF 4.7.1 real.exe wrfinput_d01 exactly, and the "
            "43-field host/device column comparison agrees at max_ulp 0. "
            "See docs/wrf_ruc_runtime_admission.md.",
        )
        for warning in spec.get("warnings", [])
    ]


#: Radiation-arm siblings: the SAME registered composition with the
#: radiation ENGINE moved and nothing else.  Both 4/4 engines live on ONE
#: registry option (``components.radiation.options.rte-rrtmgp``) and are
#: chosen by a SETTING, ``ra_rrtmg_variant``, so a sibling that swaps
#: engines moves no component at all and the suite-less table below --
#: which is keyed on component moves -- cannot express it.
#:
#: Each row is minted by copying its base and applying the settings
#: written here, so the pair differs in the radiation engine and in
#: nothing else and a paired run isolates it.  The maturity is NOT written
#: here: :func:`_composition_ceiling` derives it from the option rows the
#: composition selects, which is clause C2 of the ratified composition
#: rule.  A row here therefore cannot claim evidence the tree does not
#: hold and needs no entry in ``composition_exemptions``.
_RADIATION_ARM_SIBLINGS = (
    # (new id, base id, settings moved, label, warnings)
    (
        "thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1",
        "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1",
        {
            "ra_rrtmg_variant": "rte-rrtmgp",
            "wrf_rrtmg_compatibility": "wrf-rrtmg-4-4-to-rte-rrtmgp-v2",
        },
        "Thompson + YSU + classic MM5 + Noah + cumulus off + RTE+RRTMGP",
        (
            "Composition candidate: every component is individually "
            "verified (Thompson mp8 is wrf-matched-run; the RTE+RRTMGP "
            "longwave and shortwave engines are the shipped 4/4 radiation "
            "arm) and no receipt covers the composed suite, so this "
            "template sits AT its composition ceiling rather than above "
            "it.  The upgrade payer is named: this composition's first "
            "stock-WRF-paired t0/case receipt is what moves the label.",
            "This template differs from "
            "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1 in exactly ONE "
            "setting, the radiation engine, so the pair is a controlled "
            "engine comparison; every component and every other parameter, "
            "including the per-domain row, is transcribed from that "
            "template.",
            "radt 12.0 at dx 3000 m is transcribed from the base template "
            "and deliberately diverges from the KF template family's "
            "per-domain ladder (radt 3.0 at 3 km); every value in the "
            "per-domain row equals this template's own parameters "
            "(diff_6th_factor 0.12, epssm 0.5, radt 12.0), so the row "
            "changes no resolved setting at any grid spacing.",
        ),
    ),
)


def _radiation_arm_siblings(registry: dict) -> None:
    """Mint every row of :data:`_RADIATION_ARM_SIBLINGS`.

    Idempotent, exactly as the sibling blocks around it are: a minted id is
    removed from every route list before it is inserted, so a second build
    produces the same bytes.

    A sibling is declared on precisely the routes and sources its BASE is
    declared on, immediately after it.  The base's route set is the
    statement of where this composition can run at all, and swapping the
    radiation engine neither widens nor narrows it -- so the sibling
    inherits the set rather than asserting one of its own.
    """

    templates = registry["templates"]
    routes = registry["runner_routes"]
    for template_id, base_id, settings, label, warnings in (
            _RADIATION_ARM_SIBLINGS):
        template = copy.deepcopy(templates[base_id])
        template["parameters"].update(settings)
        template["label"] = label
        template["maturity"] = _composition_ceiling(
            registry, template["components"])
        template["warnings"] = list(warnings)
        templates[template_id] = template
        for route in routes.values():
            for declared in route.get("source_template_ids", {}).values():
                if template_id in declared:
                    declared.remove(template_id)
        for route in routes.values():
            for declared in route.get("source_template_ids", {}).values():
                if base_id in declared:
                    declared.insert(
                        declared.index(base_id) + 1, template_id)


#: Microphysics-arm siblings: a registered composition with ONE component
#: moved, the microphysics, and declared on precisely the routes and
#: sources its base is declared on, immediately after it.
#:
#: The suite-less table below cannot carry these: it registers a suite on
#: EVERY declared source of every route, and a RUC suite runs only where
#: RUC's nine-layer soil ingest does, which is what the base's own route
#: set already states.  Moving the microphysics neither widens nor narrows
#: that set, so the sibling inherits it.
#:
#: The maturity is not written here: :func:`_composition_ceiling` derives
#: it from the option rows the composition selects, and the build refuses
#: a row whose id names a maturity the derivation does not reach.
_MICROPHYSICS_ARM_SIBLINGS = (
    # (new id, base id, microphysics option, maturity the id names,
    #  label, lead warnings)
    (
        "thompson-mp8-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1",
        "wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1",
        "thompson-mp8",
        "implemented-unverified",
        "Thompson + MYNN PBL + MYNN surface layer + RUC LSM + RTE+RRTMGP",
        (
            "Composition candidate: Thompson mp8 is wrf-matched-run and "
            "RTE+RRTMGP is the shipped 4/4 radiation arm, while MYNN and RUC "
            "are implemented-unverified, so this suite sits at the "
            "implemented-unverified ceiling its weakest members set. No "
            "receipt covers the composed suite; its first stock-WRF-paired "
            "t0/case receipt is what moves the label.",
            "This template differs from "
            "wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1 in "
            "exactly ONE component, the microphysics (WSM6 -> Thompson), so "
            "the pair isolates it; every other component and parameter is "
            "transcribed from that template, and it is offered on exactly "
            "the routes and sources that template is.",
            "Both radiation streams run, so this is the nocturnally valid "
            "member of the Thompson + MYNN + RUC pair and the one to choose "
            "for low cloud, fog and stratus, where the night-time longwave "
            "cooling is the process being forecast.",
            "RADIATION IS THE ONLY DIFFERENCE from "
            "thompson-mp8-mynn-mynn-ruc-dudhia-implemented-unverified-v1: "
            "ra_lw_physics 0 -> 4, ra_sw_physics 1 -> 4 and radt 1.0 -> 12.0. "
            "This row is nocturnally valid and that one is not.",
        ),
    ),
    (
        "thompson-mp8-mynn-mynn-ruc-dudhia-implemented-unverified-v1",
        "wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1",
        "thompson-mp8",
        "implemented-unverified",
        "Thompson + MYNN PBL + MYNN surface layer + RUC LSM + Dudhia SW",
        (
            "Composition candidate: Thompson mp8 is wrf-matched-run, while "
            "MYNN and RUC are implemented-unverified, so this suite sits at "
            "the implemented-unverified ceiling its weakest members set. No "
            "receipt covers the composed suite.",
            "This template differs from "
            "wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1 in "
            "exactly ONE component, the microphysics (WSM6 -> Thompson), so "
            "the pair isolates it; every other component and parameter is "
            "transcribed from that template, and it is offered on exactly "
            "the routes and sources that template is.",
            "DAYTIME-ONLY SUITE. The radiation component is "
            "'dudhia-shortwave': ra_lw_physics 0 with ra_sw_physics 1, so "
            "Dudhia shortwave runs and no longwave scheme does, and GLW "
            "stays at zero for the whole forecast. A real window containing "
            "local night refuses to load unless [experiment] declares "
            "acknowledgements = [\"asymmetric-radiation-nocturnal-window-v1\"]. "
            "For fog and stratus choose "
            "thompson-mp8-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1, "
            "which runs both streams.",
        ),
    ),
)


#: Warning sentences that describe a sibling's BASE pairing rather than the
#: minted composition: carried over, they would name the base's partner (a
#: WSM6 row) as this row's only difference.
_BASE_DESCRIBING_WARNING_PREFIXES = (
    "This template differs from ",
    "RADIATION IS THE ONLY DIFFERENCE from ",
)


def _microphysics_arm_siblings(registry: dict) -> None:
    """Mint every row of :data:`_MICROPHYSICS_ARM_SIBLINGS`.

    Idempotent, as :func:`_radiation_arm_siblings` is: a minted id is
    removed from every route list before it is inserted after its base, so
    a second build over this build's own output produces the same bytes.

    The base's warnings are inherited after the row's own, less the two
    kinds of sentence that describe the BASE rather than the composition:
    its "differs from" pairing sentence, and the frozen-name warning a
    'no-radiation' id carries, which would be false on an id that does not
    say 'no-radiation'.
    """

    templates = registry["templates"]
    routes = registry["runner_routes"]
    for (template_id, base_id, microphysics, named_maturity, label,
         warnings) in _MICROPHYSICS_ARM_SIBLINGS:
        base = templates[base_id]
        template = copy.deepcopy(base)
        template["components"]["microphysics"] = microphysics
        template["label"] = label
        maturity = _composition_ceiling(registry, template["components"])
        if MATURITY_RENAMES.get(named_maturity, named_maturity) != maturity:
            raise SystemExit(
                f"template {template_id!r} names maturity {named_maturity!r} "
                f"and its composition ceiling is {maturity!r}; an id that "
                "states a rank its components do not reach is a false claim "
                "every door would repeat")
        template["maturity"] = maturity
        inherited = [
            warning for warning in base.get("warnings", [])
            if not warning.startswith(_BASE_DESCRIBING_WARNING_PREFIXES)
            and warning != NO_RADIATION_NAME_WARNING
        ]
        template["warnings"] = [*warnings, *inherited]
        templates[template_id] = template
        for route in routes.values():
            for declared in route.get("source_template_ids", {}).values():
                if template_id in declared:
                    declared.remove(template_id)
        for route in routes.values():
            for declared in route.get("source_template_ids", {}).values():
                if base_id in declared:
                    declared.insert(
                        declared.index(base_id) + 1, template_id)


#: Audit R-067.  Eleven implemented options had no shipped template at all,
#: so every one of them was a scheme a user could not select from a named
#: suite -- ``implemented: true`` with no front door.  Each row below is
#: built from an existing template by moving the FEWEST components that
#: reach the option, so a paired run against its base isolates the change,
#: and each is registered on every route the composition is valid for.
#:
#: The ids carry no case, site or source token: they name the composition.
_SUITELESS_TEMPLATES = (
    # (new id, base id, component moves, extra parameters, label, warnings)
    (
        "milbrandt2mom-mp9-ysu-mm5-noah-ntiedtke-rrtmg-legacy-v1",
        "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1",
        {"microphysics": "milbrandt2mom-mp9", "cumulus": "new-tiedtke"},
        {},
        "Milbrandt-Yau two-moment + YSU + classic MM5 + Noah + New Tiedtke "
        "+ legacy RRTMG",
        (
            "Composition candidate: every component is individually "
            "implemented and the legacy RRTMG engine is the certified WRF "
            "v4.6.1 port, but no receipt covers the composed suite. This is "
            "the suite that ran 59 minutes and died at its first checkpoint "
            "before the identity row for mp_physics=9 existed; it is "
            "registered so the scheme has a named front door rather than "
            "only an unnamed tuple.",
            "This preset selects the legacy RRTMG engine to retain its "
            "named composition. Milbrandt-Yau also supports the modern "
            "RTE+RRTMGP coupling through its implemented cloud-optics row; "
            "choose that engine in a separate composition when wanted.",
        ),
    ),
    (
        "wdm6-mp16-ysu-mm5-noah-grell-freitas-rte-rrtmgp-v1",
        "thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1",
        {"microphysics": "wdm6-mp16", "cumulus": "grell-freitas"},
        {},
        "WDM6 + YSU + classic MM5 + Noah + Grell-Freitas + RTE+RRTMGP",
        (
            "Composition candidate: WDM6 and Grell-Freitas are each "
            "implemented and individually measured, and RTE+RRTMGP is the "
            "shipped radiation arm, but no receipt covers the composed "
            "suite. Registered so both schemes have a named front door.",
            "Grell-Freitas requires a PBL closure that supplies its "
            "boundary-layer state (its own requires_components row); YSU is "
            "the arm this suite pairs it with.",
        ),
    ),
    (
        "thompson-aerosol-mp28-myj-eta-noah-rte-rrtmgp-v1",
        "thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1",
        {
            "microphysics": "thompson-aerosol-mp28",
            "pbl": "myj",
            "surface_layer": "eta-similarity",
            "cumulus": "off",
        },
        {"cudt_minutes": 0.0},
        "Thompson aerosol-aware + MYJ + Eta similarity + Noah + cumulus off "
        "+ RTE+RRTMGP",
        (
            "Composition candidate: no receipt covers the composed suite. "
            "Registered so the aerosol-aware Thompson scheme and the "
            "MYJ/Eta pair each have a named front door.",
            "MYJ and the Eta similarity surface layer require EACH OTHER "
            "(both requires_components rows, and gpuwm.config."
            "validate_myj_pairing), so they are selected together or not at "
            "all. This template is the one registered suite that satisfies "
            "both halves.",
            "The aerosol-aware scheme reads the packaged CCN activation "
            "table shipped in the gpuwm-data companion wheel; a default "
            "install satisfies it, and a run with no bound aerosol input "
            "uses the packaged fallback constants and says so.",
        ),
    ),
    (
        "wsm6-sase-revised-mm5-noah-closure-supplied-v1",
        "wsm6-ysu-mm5-noah-no-radiation-v1",
        {
            "pbl": "sase",
            "surface_layer": "revised-mm5",
            "turbulence": "closure-supplied",
        },
        # NOT restated here.  SASE's own required_settings row pins the
        # quadruple (bldt, khdif, kvdif, km_opt), the derivation reads
        # that row into every runtime product, and a template that
        # mentions ``bldt`` at ANY value is the inheritance vector
        # tests/test_noahmp_surface_interval.py refuses outright: a
        # surface-call interval is a cost mitigation that must stay
        # opt-in per configuration, and a template carries its parameters
        # into every run built from it with no author ever typing them.
        {},
        "WSM6 + SASE + revised MM5 + Noah + closure-supplied mixing",
        (
            "Composition candidate: no receipt covers the composed suite. "
            "Registered so the SASE closure has a named front door.",
            "SASE supplies its own mixing, so this suite runs with km_opt=0 "
            "and bldt, khdif and kvdif at zero -- the quadruple SASE's own "
            "required_settings row states, which every route resolves and "
            "plan review refuses an override of. The suite inherits that "
            "row rather than restating it. A second mixing operator beside "
            "the closure would double-count its own transport.",
            "SASE is not a WRF v4.6.1 scheme: it carries an "
            "out-of-namespace selector and has no stock-WRF counterpart to "
            "be compared against, so its surface-layer requirement states "
            "what the closure READS rather than transcribing WRF's cell "
            "table.",
        ),
    ),
    (
        "wsm6-pbl-off-mm5-noah-tke-1-5-order-v1",
        "wsm6-ysu-mm5-noah-no-radiation-v1",
        {"pbl": "off", "turbulence": "tke-1.5-order"},
        {},
        "WSM6 + PBL off + classic MM5 + Noah + 1.5-order TKE mixing",
        (
            "Composition candidate: no receipt covers the composed suite. "
            "Registered so the 1.5-order TKE closure has a named front "
            "door.",
            "A three-dimensional mixing operator REPLACES the PBL "
            "parameterization rather than joining it: km_opt=2 requires "
            "bl_pbl_physics=0 (the option's own required_settings row), "
            "which is why this suite selects PBL off. The surface layer "
            "stays on and supplies the fluxes Noah needs.",
            "This is a large-eddy composition. On a grid whose spacing does "
            "not resolve the energy-containing eddies it under-mixes the "
            "boundary layer, because nothing else is parameterizing it.",
        ),
    ),
    (
        "wsm6-pbl-off-mm5-noah-smagorinsky-3d-v1",
        "wsm6-ysu-mm5-noah-no-radiation-v1",
        {"pbl": "off", "turbulence": "smagorinsky-3d"},
        {},
        "WSM6 + PBL off + classic MM5 + Noah + 3D Smagorinsky mixing",
        (
            "Composition candidate: no receipt covers the composed suite. "
            "Registered so the three-dimensional Smagorinsky closure has a "
            "named front door.",
            "km_opt=3 requires bl_pbl_physics=0 (the option's own "
            "required_settings row): the operator replaces the PBL "
            "parameterization rather than joining it. The surface layer "
            "stays on and supplies the fluxes Noah needs.",
            "This is a large-eddy composition, and it is the sibling of the "
            "1.5-order TKE row: the two differ in exactly ONE component, so "
            "a paired run isolates the closure.",
        ),
    ),
    (
        "wsm6-pbl-off-mm5-noah-constant-k-v1",
        "wsm6-ysu-mm5-noah-no-radiation-v1",
        {"pbl": "off", "turbulence": "constant-k"},
        {},
        "WSM6 + PBL off + classic MM5 + Noah + constant-K mixing",
        (
            "Composition candidate: no receipt covers the composed suite. "
            "Registered so the constant-K operator has a named front door, "
            "as the km_opt=1 sibling of the two large-eddy rows above.",
            "khdif and kvdif come from the option's own row and are zero "
            "there, which is no horizontal or vertical mixing at all. A "
            "suite that wants constant-K transport states the two "
            "coefficients in its experiment config; this template does not "
            "invent values an oracle has not judged.",
        ),
    ),
)


#: The ONE runner route id that replays a native comparison rather than
#: building its product from the registry alone.  Named once, here, so the
#: exclusion table below and the gate that checks it read the same string.
_NATIVE_BENCHMARK_ROUTE = "tools.hrrr_single_domain_benchmark"

#: Which suite-less template stays off which route, and the concrete
#: breakage that keeps it off.  A template is registered on every route it
#: is valid for; every departure from "all three" is a row here, and
#: :func:`_phase2c_suiteless_templates` FAILS THE BUILD if a template is
#: absent from a route with no row -- which is the reverse leg the first
#: pass lacked, when six suites were declared on a route whose runner
#: refuses all six.
#:
#: RETIRED, with the fix it waited on: the aerosol-aware Thompson row that
#: kept its suite off BOTH fixed-template routes because
#: source_absent_microphysics had no mp_physics=28 arm (audit R-044).  The
#: arm exists (gpuwm/ingest/microphysics_cold_start.py, nc/nr/ni at exact
#: zero), gpuwm/ingest/real.py seeds nwfa/nifa from the WIF monthly
#: climatology, and a missing dataset is refused by name before the fetch
#: by gpuwm.config.mp28_aerosol_lateral_forcing_precondition, which offers
#: mp28_aerosol_source = 'synthetic' as the way out.  The suite is on the
#: prepared single-domain route now; the native benchmark keeps it off for
#: its own reason, in the composition loop below.
_TEMPLATE_ROUTES_REFUSED: dict[str, dict[str, str]] = {}
#: The composition suites are valid compositions and the prepared
#: single-domain route resolves each of them from the registry alone.  The
#: NATIVE BENCHMARK route cannot: its product is a replay of a native WRF
#: run, gated field for field against a transcribed namelist contract
#: (tools/hrrr_single_domain_benchmark.py _NATIVE_HRRR_NAMELIST_CONTRACTS)
#: and forwarded through a per-switch home map (_PROFILE_SWITCH_HOMES).
#: A composition with no native run behind it has no contract to be
#: replayed against, so declaring it there offered a suite the runner
#: refused with `unsupported native HRRR physics profile`.
for _composition_suite_id, _off_the_benchmark_because in (
        ("milbrandt2mom-mp9-ysu-mm5-noah-ntiedtke-rrtmg-legacy-v1",
         "no native run of this composition exists, so there is no "
         "namelist contract to gate its replay against"),
        ("wdm6-mp16-ysu-mm5-noah-grell-freitas-rte-rrtmgp-v1",
         "no native run of this composition exists, so there is no "
         "namelist contract to gate its replay against"),
        ("thompson-aerosol-mp28-myj-eta-noah-rte-rrtmgp-v1",
         "no native run of this composition exists, so there is no "
         "namelist contract to gate its replay against"),
        ("wsm6-sase-revised-mm5-noah-closure-supplied-v1",
         "SASE carries an out-of-namespace selector and has no stock-WRF "
         "counterpart at all, so a native comparison cannot be stated for "
         "it, let alone measured"),
        ("wsm6-pbl-off-mm5-noah-tke-1-5-order-v1",
         "a large-eddy closure on this route's fixed kilometre-scale "
         "single domain has no native run behind it and no resolved "
         "energy-containing eddies to close over"),
        ("wsm6-pbl-off-mm5-noah-smagorinsky-3d-v1",
         "a large-eddy closure on this route's fixed kilometre-scale "
         "single domain has no native run behind it and no resolved "
         "energy-containing eddies to close over"),
        ("wsm6-pbl-off-mm5-noah-constant-k-v1",
         "the constant-K operator runs with the option's own zero "
         "coefficients, so there is nothing for a published wall-clock "
         "and skill comparison to be measured against"),
):
    _TEMPLATE_ROUTES_REFUSED.setdefault(_composition_suite_id, {})[
        _NATIVE_BENCHMARK_ROUTE] = (
            "the native benchmark route replays ONE immutable template and "
            "gates the operator's native WRF namelist field for field "
            "against a transcribed contract: "
            + _off_the_benchmark_because
            + ". Run this suite on tools.prepared_single_domain_forecast, "
            "which resolves every switch of it from the registry, or state "
            "it per domain on tools.prepared_domain_tree_forecast.")
del _composition_suite_id, _off_the_benchmark_because


#: A source whose template contract is another declared source's, because the
#: two decode the SAME producer on the same field and soil contract through
#: routes that differ only in transport.  One row per such pair, so a route
#: that gains a second way into a producer it already serves inherits that
#: producer's declaration instead of acquiring an empty one.  This is table
#: work by construction: a new producer is a key, never a branch.
_CONTRACT_TWIN_SOURCES = {
    # The native and packaged pressure-level routes into one producer, each
    # way in naming the other: whichever of the pair a route declares, the
    # other inherits, so a route that reaches a producer twice does not
    # offer its suites on one road and nothing on the other.
    "hrrr": "hrrr-prs",
    "hrrr-prs": "hrrr",
    # The NetCDF profile decodes the same producer on the same
    # pressure-level and soil contract as the GRIB one.
    "20crv3-cf": "20crv3",
}


#: A source a route SERVES and names NO suite for, with the reason it names
#: none.  The completion below prices every other undeclared source from the
#: route's own generic declaration, so an empty list is reachable only by
#: saying here that the source names nothing and why.  Table work by
#: construction: such a source is a key, never a branch.
_SOURCES_THAT_NAME_NO_SUITE = {
    # The caller-supplied composition: no packaged profile stands behind it
    # and the physics is whatever the caller wrote, so there is no suite to
    # name and no route-wide expert list to reach it either (which is what
    # gpuwm.physics_registry.expert_template_ids_for_source reads an empty
    # normal list as meaning).
    "mapped": "the composition is the caller's, so no suite is named for it",
}


def _route_generic_template_declaration(
        stated: dict[str, list[str]]) -> list[str]:
    """The suites a route declares for EVERY source it has declared.

    A template every declared source names is one that reads nothing
    source-specific, by construction -- no source's own verification row
    can survive the intersection -- so this is the most conservative basis
    the route itself has on record, and it is what a source with no
    measurement of its own is priced from.

    Computed from TWO OR MORE declared sources only.  With one, the
    intersection is that source's whole list, evidence rows and all, and
    handing it to a second source would publish one source's evidence as
    another's -- the exact failure tests/test_build_registry.py's Kessler
    pin caught in the first, route-wide version of this completion.
    """

    lists = [templates for templates in stated.values() if templates]
    if len(lists) < 2:
        return []
    common = list(lists[0])
    for other in lists[1:]:
        common = [template_id for template_id in common if template_id in other]
    return common


def _every_served_source_declares_a_template_list(registry: dict) -> None:
    """A fixed-template route that SERVES a source says what it offers there.

    Such a route runs ONLY its registered templates, so a source it serves
    and declares nothing for offers nothing at all, and the declaration has
    to say which of the two it means.

    ``source_ids`` and ``source_template_ids`` were allowed to disagree:
    three sources of the single-domain route were served and undeclared,
    so the runner drift check raised ``KeyError`` on the first of them
    rather than comparing anything, and
    ``expert_template_ids_for_source``'s rule that an empty declared list
    offers nothing could not fire for a source that had no list at all.

    An undeclared source gets, in order: the declaration of the source it
    shares a producer and contract with where one is recorded; an empty
    list where the source is recorded as naming no suite at all, with the
    reason; and otherwise the route's OWN generic declaration, the suites
    every source it has declared names.

    What it may NOT get is an empty list for want of a measurement.  That
    was the first shape of this pass, and an empty list here is not
    silence: it is published as "this source reaches no named suite", and
    ``expert_template_ids_for_source`` withholds the route-wide expert
    list from such a source as well.  On the single-domain route it took
    the six composition suites and the three Noah-MP expert suites away
    from aigfs and era5-l137 -- two sources the runner runs through the
    same generic mapped route as the seven siblings that declare those
    suites -- and warned a plan naming one with
    ``template-route-evidence``, which says the template is off-route for
    this source, instead of handing it the acknowledgement advisory.  The
    plan still launched; what was published about it was wrong.  A source with nothing measured on it is priced from the
    most conservative basis the route has on record, and the route says
    which basis that was.
    """

    for route_id, route in registry["runner_routes"].items():
        if route.get("mode") != "fixed-template":
            # An experiment-per-domain route composes from its declaration
            # rather than from a template, so an absent per-source list is a
            # different shape there, not an incomplete one -- and filling it
            # would hand a source evidence-scoped registrations made for
            # another (the Kessler and legacy-RRTMG rows are one source's own).
            continue
        declared = route.setdefault("source_template_ids", {})
        if not declared:
            continue
        # Twins and the generic basis both resolve against what the route
        # declared BEFORE this pass, so the completion does not depend on
        # the order source_ids happens to list a pair in, and a second
        # build over this build's own output changes nothing.
        stated = {source_id: list(templates)
                  for source_id, templates in declared.items() if templates}
        generic = _route_generic_template_declaration(stated)
        for source_id in route.get("source_ids", []):
            if declared.get(source_id):
                continue
            twin = _CONTRACT_TWIN_SOURCES.get(source_id)
            if twin is not None:
                declared[source_id] = list(stated.get(twin, ()))
                continue
            if source_id in _SOURCES_THAT_NAME_NO_SUITE:
                declared[source_id] = []
                continue
            if not generic:
                raise RuntimeError(
                    f"runner route {route_id} serves source {source_id}, "
                    "declares no template list for it, and its declared "
                    "sources share no suite for one to be priced from, so "
                    "completing it would publish a source as reaching no "
                    "named suite at all. Record the source it shares a "
                    "producer with in _CONTRACT_TWIN_SOURCES, or record "
                    "why it names no suite in _SOURCES_THAT_NAME_NO_SUITE.")
            declared[source_id] = list(generic)


def _phase2c_suiteless_templates(registry: dict) -> None:
    """Audit R-067: every implemented option gets a named suite.

    Eleven implemented options carried ``reachability: component-override``
    and no template at all, which is the ship-only-what-users-can-reach
    rule failing quietly: the option was selectable only by hand-writing a
    tuple, and the front doors that offer NAMED suites offered none of
    them.

    Each template below is registered on every route it is physically
    valid for.  None of these compositions reads anything source-specific,
    so what a route CAN do with them is the only question, and every
    departure from "all three routes" carries a row in
    ``_TEMPLATE_ROUTES_REFUSED`` naming the breakage and the way out.  A
    source that declares no suite AT ALL (the ``mapped`` row, which is a
    caller-supplied composition) does not get its first one here: that is
    the source's gap, not the composition's.

    The build FAILS if a minted template ends up off a route with no row,
    or carries a row for a route it was never kept off.  That reverse leg
    is why this function is the place the exclusion is enforced: the first
    pass declared six of these on the native benchmark route, whose runner
    refuses every one of them, and nothing in the build noticed.
    """

    templates = registry["templates"]
    routes = registry["runner_routes"]
    benchmark = routes["tools.hrrr_single_domain_benchmark"]
    tree = routes["tools.prepared_domain_tree_forecast"]
    single = routes["tools.prepared_single_domain_forecast"]
    minted: list[str] = []

    for (template_id, base_id, moves, extra_parameters, label,
         warnings) in _SUITELESS_TEMPLATES:
        template = copy.deepcopy(templates[base_id])
        template["components"].update(moves)
        template["parameters"].update(extra_parameters)
        template["label"] = label
        # The composition ceiling: a suite with no receipt of its own
        # cannot rank above the weakest thing in it, and none of these has
        # a composed receipt at all.
        template["maturity"] = "implemented-unverified"
        template["warnings"] = list(warnings)
        # per_domain_overrides transcribe values from a verified run of
        # the BASE suite; none of these suites has one, so the row is
        # dropped rather than inherited as a claim about this composition.
        template.pop("per_domain_overrides", None)
        templates[template_id] = template
        minted.append(template_id)

    # Idempotent: remove before inserting, exactly as the Kessler, legacy
    # NSSL-2, Shin-Hong and P3 rows above do, so a second build produces
    # the same bytes.
    for route in routes.values():
        for declared in route.get("source_template_ids", {}).values():
            for template_id in minted:
                if template_id in declared:
                    declared.remove(template_id)
    declared_on: dict[str, set[str]] = {
        template_id: set() for template_id in minted}
    for route_id, route in (
            (_NATIVE_BENCHMARK_ROUTE, benchmark),
            ("tools.prepared_domain_tree_forecast", tree),
            ("tools.prepared_single_domain_forecast", single),
    ):
        refused_here = {
            template_id
            for template_id, routes in _TEMPLATE_ROUTES_REFUSED.items()
            if route_id in routes
        }
        for source_id, declared in route["source_template_ids"].items():
            if source_id not in route["source_ids"]:
                continue
            if not declared:
                # A source with no declared suite at all keeps none.
                continue
            for template_id in minted:
                if template_id in refused_here:
                    continue
                declared.append(template_id)
                declared_on[template_id].add(route_id)

    # The reverse leg.  A template off a route with no written reason is a
    # silent narrowing; a reason for a route the template is on is a stale
    # refusal that will outlive what it describes.
    all_routes = {
        _NATIVE_BENCHMARK_ROUTE,
        "tools.prepared_domain_tree_forecast",
        "tools.prepared_single_domain_forecast",
    }
    for template_id in minted:
        refused = set(_TEMPLATE_ROUTES_REFUSED.get(template_id, {}))
        absent = all_routes - declared_on[template_id]
        if absent != refused:
            raise SystemExit(
                f"template {template_id!r} is declared on "
                f"{sorted(declared_on[template_id])} and refused on "
                f"{sorted(refused)}: every route a suite-less template is "
                "kept off needs a row in _TEMPLATE_ROUTES_REFUSED naming "
                "the breakage and the way out, and a row for a route it "
                "IS on must be retired with the reason it recorded "
                f"(unreconciled: {sorted(absent ^ refused)})")

    _publish_refused_template_ids(registry)


def _publish_refused_template_ids(registry: dict) -> None:
    """Put every route refusal where the user meets it: the registry.

    ``_TEMPLATE_ROUTES_REFUSED`` above states, for each template kept off
    a route, the concrete breakage and the way out.  It lived only in this
    builder, so none of it reached a user: plan review saw a template that
    the route simply did not declare, warned that "the resolved runtime
    settings still apply", and returned launchable -- and then the runner
    refused at the door with a bare ``unsupported ... physics profile``
    naming neither the breakage nor an alternative.  A refusal that fires
    after review, in a sentence that offers nothing, is the shape this
    project refuses to ship.

    Published as ``runner_routes.<runner>.refused_template_ids``, a map of
    template id to that sentence, so
    :func:`gpuwm.physics_registry.validate_physics_plan` can refuse at plan
    review with the reason written here.  Rewritten from scratch on every
    build, so a retired row leaves the registry with the reason it recorded
    rather than outliving it.
    """

    for route_id, route in registry["runner_routes"].items():
        refused = {
            template_id: routes[route_id]
            for template_id, routes in sorted(_TEMPLATE_ROUTES_REFUSED.items())
            if route_id in routes
        }
        if refused:
            route["refused_template_ids"] = refused
        else:
            route.pop("refused_template_ids", None)


def _phase2c_recompute_reachability(registry: dict) -> None:
    """Recompute every option's ``reachability`` from templates and routes.

    ``reachability.state`` names the EASIEST path a user has to an option,
    and it was hand-set beside each row that created a path.  With the
    route declarations widened (R-021, R-022, R-023, R-059) and eleven
    options gaining their first template (R-067), hand-setting forty
    states is how one of them ends up wrong and nobody notices.  It is
    computed here from the same two facts a user actually has -- the
    declared template lists and the declared override lists -- and
    ``tests/test_registry_reachability.py`` recomputes it independently
    and fails on any difference, which is the agreement that makes this a
    derivation rather than a second opinion.

    An option that ends up unreachable keeps the blocker it declared: an
    implemented option declared unreachable must name what blocks it, and
    that sentence is written where the option is registered, not here.
    """

    components = registry["components"]
    templates = registry["templates"]
    # Easiest first, the order a user finds them in.
    state_order = ("template", "component-override", "expert-template")
    reached: dict[tuple[str, str], set[str]] = {
        (component_id, option_id): set()
        for component_id, component in components.items()
        for option_id in component["options"]
    }

    for route in registry["runner_routes"].values():
        if route.get("implemented") is not True:
            continue
        normal = route.get("source_template_ids", {}) or {}
        expert = route.get("expert_template_ids", {}) or {}
        declares = bool(normal) or bool(expert)
        per_domain = route.get("mode") == "experiment-per-domain"
        overridable = set(route.get("allowed_component_overrides", []) or [])
        option_overrides = route.get("allowed_component_options", {}) or {}
        expert_selector_keys = set(
            route.get("allowed_expert_selector_keys", []) or [])
        for source_id in route.get("source_ids", []) or []:
            if declares:
                normal_ids = list(normal.get(source_id, []) or [])
                expert_ids = list(expert.get(source_id, []) or [])
            else:
                normal_ids, expert_ids = list(templates), []
            for template_id in normal_ids:
                for component_id, option_id in templates[
                        template_id]["components"].items():
                    reached[(component_id, option_id)].add("template")
            for template_id in expert_ids:
                for component_id, option_id in templates[
                        template_id]["components"].items():
                    reached[(component_id, option_id)].add("expert-template")
            if not (normal_ids or expert_ids):
                # An override still needs a base template to override.
                continue
            for component_id, component in components.items():
                selector_keys = set(component.get("selector_keys", []) or [])
                by_override = per_domain and component_id in overridable
                by_selector = bool(selector_keys) and selector_keys <= (
                    expert_selector_keys)
                admitted = set(option_overrides.get(component_id, []) or [])
                for option_id in component["options"]:
                    if by_override or by_selector or option_id in admitted:
                        reached[(component_id, option_id)].add(
                            "component-override")

    for (component_id, option_id), ways in reached.items():
        option = components[component_id]["options"][option_id]
        declared = option.get("reachability")
        declared = declared if isinstance(declared, dict) else {}
        if option.get("implemented") is not True:
            # Nameable is not reachable: the resolver refuses it from every
            # template, on every route, for every source.
            option["reachability"] = declared
            continue
        state = next((name for name in state_order if name in ways), None)
        if state is None:
            if not declared.get("blocker"):
                raise RuntimeError(
                    f"components.{component_id}.options.{option_id} is "
                    "implemented and no template or route declaration "
                    "reaches it, and it names no blocker; give it a route "
                    "row or write what blocks it")
            option["reachability"] = {
                "state": "unreachable", "blocker": declared["blocker"]}
            continue
        option["reachability"] = {"state": state}


def render(registry: dict) -> bytes:
    """The exact bytes the tracked registry file must contain."""
    return (canonical_json(registry) + "\n").encode("utf-8")


#: Why every cumulus option pins moist=true, in the words of the run door
#: (gpuwm.config validate_run_config): rendered by the plan door after the
#: pinned value so both doors name one breakage.
_CUMULUS_MOIST_REASON = (
    "the cumulus schemes are moist convective schemes; "
    "gpuwm/core/physics.py initialize_physics refuses a cumulus scheme on a "
    "dry DomainState, whose qv is None")


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
    parser.add_argument(
        "--export-out", type=pathlib.Path, default=None,
        help="where to write the consumer export the Rust crates read; "
             "defaults to gpuwm/physics_consumer_export_v1.json beside the "
             "tracked registry, or beside --out when that names another "
             "directory")
    args = parser.parse_args(argv)

    import os
    from gpuwm.physics_registry import REGISTRY_REBUILD_ENV
    # The consumer modules this build pulls facts from assert agreement
    # with the registry ON DISK at import, and the disk copy is the one
    # being replaced; the flag tells them so for this process only.
    os.environ[REGISTRY_REBUILD_ENV] = "1"

    registry = build(json.loads(args.registry.read_text(encoding="utf-8")))
    params = registry["parameters"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(render(registry))
    export_out = args.export_out
    if export_out is None:
        export_out = (CONSUMER_EXPORT_PATH if args.out == REGISTRY_PATH
                      else args.out.parent / CONSUMER_EXPORT_PATH.name)
    export_out.write_bytes(render_consumer_export(registry))
    print("parameters:", len(params),
          "| implemented:", sum(1 for s in params.values()
                                if s.get("implemented") is not False),
          "| unimplemented:", sum(1 for s in params.values()
                                  if s.get("implemented") is False),
          "|", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
