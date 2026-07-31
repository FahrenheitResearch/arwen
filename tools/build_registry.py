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
    "map_proj": {"type": "integer", "enum": [0, 1], "default": 0},
    # boundary layer
    "ysu_topdown_pblmix": {"type": "integer", "enum": [0, 1], "default": 1},
    # radiation
    "swrad_scat": {"type": "number", "minimum": 0.0, "default": 1.0},
    "ra_rrtmg_variant": {
        "type": "string",
        "enum": ["rte-rrtmgp", "rrtmg_legacy"],
        "default": "rte-rrtmgp"},
    # surface layer -- newly ported this pass
    "isftcflx": {"type": "integer", "enum": [0, 1, 2], "default": 0},
    "iz0tlnd": {"type": "integer", "enum": [0, 1, 2], "default": 0},
    # Noah LSM -- newly ported: kernel branches already existed at
    # noah.cu:1036/:1111/:1112/:1116/:1117 (usemonalb, rdlai2d) and :181
    # (opt_thcnd); only the configuration path was missing.
    "usemonalb": {"type": "boolean", "default": False},
    "rdlai2d": {"type": "boolean", "default": False},
    "opt_thcnd": {"type": "integer", "enum": [1, 2], "default": 1},
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
        "MYNN surface-layer pairing is admitted as the coupled MYNN 5/5 "
        "suite. WRF v4.6.1 runs MYNN first, then NOAHMP_SFLX overwrites "
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
        "XICE_THRESHOLD=0.5, seaice_albedo_default=0.65, isncovr_opt=2, "
        "c1sn=0.026, c2sn=21.0, myj=False and rdlai2d=False. "
        "FRACTIONAL_SEAICE follows WRF's enabled pre/post coupling.")
    ruc[8] = (
        "Admitted at num_soil_layers=9 with revised MM5, classic MM5 or the "
        "coupled MYNN 5/5 suite. Under MYNN, WRF v4.6.1 runs the surface "
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


def build(registry: dict) -> dict:
    """Apply this pass's tables to ``registry`` in place and return it."""
    _surface_coupling_warnings(registry)
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
        "template, component-override, and expert-template reachability. "
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
