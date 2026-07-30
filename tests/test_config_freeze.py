"""Phase-5 Task 1 byte-identity freeze pins (architecture section A).

``gpuwm/config.py`` carries the Phase-5 fields plus WRF-default
``top_lid = False`` and attribution-only ``moist_cq = True``;
``load_config()`` and the legacy
``[grid]``/``[dynamics]``/``[run]`` tables are untouched.  These pins
prove the guarantee: every existing legacy TOML under ``configs/`` and
every frozen-case constructed RunConfig re-resolves IDENTICALLY --
``dataclasses.asdict`` compared field-for-field against the golden
snapshot captured from the pre-change code plus explicitly reviewed new
configuration fields (``tests/data/config_freeze_golden.json``).

Freeze discipline: a legacy TOML added to ``configs/`` without a golden
entry FAILS here -- new files are pinned consciously, never implicitly.
Experiment-shaped TOMLs ([experiment]/[[domain]]) are exercised by
tests/test_experiment.py instead.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from gpuwm.config import RunConfig, load_config
from gpuwm.experiment import is_experiment_toml

REPO = Path(__file__).resolve().parents[1]
GOLDEN = json.loads(
    (Path(__file__).parent / "data" / "config_freeze_golden.json")
    .read_text())


def _frozen_constructors():
    from gpuwm.verify.cases import (hill2d, igw, moist_bubble, real74_d01,
                                    straka, wk82)
    return {
        "straka.default_config": straka.default_config,
        "igw.default_config": igw.default_config,
        "hill2d.default_config": hill2d.default_config,
        "moist_bubble.default_config": moist_bubble.default_config,
        "wk82.default_config": wk82.default_config,
        "real74_d01.config": real74_d01.config,
        "real74_d01.phase3_config": real74_d01.phase3_config,
    }


def test_new_fields_are_reviewed_defaults_appended_last():
    """New fields remain appended, preserving positional construction."""
    names = [f.name for f in dataclasses.fields(RunConfig)]
    assert names[-60:] == [
        "nested", "grid_id", "top_lid", "moist_cq", "morr_rimed_ice",
        "wsm6_hail_opt", "ra_lw_physics", "ra_sw_physics", "icloud",
        "swrad_scat", "wrf_rrtmg_compatibility", "num_soil_layers",
        "ra_rrtmg_variant",
        "nest_microphysics_transition", "isftcflx", "iz0tlnd",
        "usemonalb", "rdlai2d", "opt_thcnd",
        "bl_mynn_closure", "bl_mynn_cloudpdf", "bl_mynn_mixlength",
        "bl_mynn_edmf", "bl_mynn_edmf_mom", "bl_mynn_edmf_tke",
        "bl_mynn_mixscalars", "bl_mynn_cloudmix", "bl_mynn_mixqt",
        "bl_mynn_output", "bl_mynn_tkeadvect", "icloud_bl",
        "dveg", "opt_crs", "opt_btr", "opt_run", "opt_sfc", "opt_frz",
        "opt_inf", "opt_rad", "opt_alb", "opt_snf", "opt_tbot", "opt_stc",
        "opt_gla", "opt_rsf", "opt_soil", "opt_pedo", "opt_crop", "opt_irr",
        "opt_irrm", "opt_infdv", "opt_tdrn", "soiltstep", "noahmp_output",
        "noahmp_acc_dt",
        "mosaic_lu", "mosaic_soil", "flag_sm_adj", "spp_lsm",
        "nwp_diagnostics"]
    # Noah option selectors. Each default reproduces the value the launcher
    # previously pinned, so exposing them cannot move a frozen trajectory.
    assert RunConfig.__dataclass_fields__["usemonalb"].default is False
    assert RunConfig.__dataclass_fields__["rdlai2d"].default is False
    assert RunConfig.__dataclass_fields__["opt_thcnd"].default == 1
    assert RunConfig.__dataclass_fields__["nested"].default is False
    assert RunConfig.__dataclass_fields__["grid_id"].default == 1
    # Stability defaults (2026-07-18): rigid lid + no cq until the open-top
    # and cq production defects have stability + falsification receipts
    # (see config.py field comments and the iso-* probe records).
    assert RunConfig.__dataclass_fields__["top_lid"].default is True
    assert RunConfig.__dataclass_fields__["moist_cq"].default is False
    # Morrison's scalar Registry default is hail; explicit 0 retains the
    # graupel branch (Registry.EM_COMMON:2663-2666).
    assert RunConfig.__dataclass_fields__["morr_rimed_ice"].default == 1
    # WSM6 uses a distinct switch and the opposite WRF default: graupel.
    assert RunConfig.__dataclass_fields__["wsm6_hail_opt"].default == 0
    assert RunConfig.__dataclass_fields__[
        "nest_microphysics_transition"].default == "same-scheme-only"
    assert RunConfig.__dataclass_fields__["ra_lw_physics"].default == -1
    assert RunConfig.__dataclass_fields__["ra_sw_physics"].default == -1
    assert RunConfig.__dataclass_fields__["icloud"].default == 1
    assert RunConfig.__dataclass_fields__["swrad_scat"].default == 1.0
    assert RunConfig.__dataclass_fields__["wrf_rrtmg_compatibility"].default \
        == "none"
    assert RunConfig.__dataclass_fields__["num_soil_layers"].default == 4
    # 4/4 stays on the modern RTE+RRTMGP adapter unless the exact
    # legacy-RRTMG port is selected explicitly (which fails closed until
    # its compute kernels land).
    assert RunConfig.__dataclass_fields__["ra_rrtmg_variant"].default \
        == "rte-rrtmgp"
    # nwp_diagnostics: WRF's own Registry default is 0 (off), and at 0 the
    # dycore epilogue never launches the UP_HELI_MAX kernel, so every frozen
    # trajectory is unchanged.  At 1 the diagnostic is trajectory-inert by
    # test (tests/test_uh_lifecycle.py).
    assert RunConfig.__dataclass_fields__["nwp_diagnostics"].default == 0
    # MM5 surface-layer options.  Both WRF Registry defaults are 0, and both
    # are inert at 0: the surface-layer kernel's isftcflx/iz0tlnd branches
    # are skipped entirely, so every frozen trajectory is unchanged.
    assert RunConfig.__dataclass_fields__["isftcflx"].default == 0
    assert RunConfig.__dataclass_fields__["iz0tlnd"].default == 0
    # MYNN PBL option identity.  Every default is the single value the ported
    # solver was validated at, and none of them is read unless
    # bl_pbl_physics=5, which no frozen configuration selects -- so exposing
    # them cannot move a frozen trajectory.  validate_run_config refuses any
    # other value, so these are pins rather than choices.
    from gpuwm.config import MYNN_PBL_OPTION_IDENTITY
    for name, admitted in MYNN_PBL_OPTION_IDENTITY.items():
        default = RunConfig.__dataclass_fields__[name].default
        assert default == admitted and type(default) is type(admitted), name
    # Noah-MP option identity, on exactly the same terms: every default is
    # the single admitted value, validate_run_config refuses any other, and
    # none is read unless sf_surface_physics=4, which no frozen configuration
    # selects -- real74_d01 is Noah (2).
    from gpuwm.config import NOAHMP_OPTION_IDENTITY
    for name, admitted in NOAHMP_OPTION_IDENTITY.items():
        default = RunConfig.__dataclass_fields__[name].default
        assert default == admitted and type(default) is type(admitted), name
    # RUC option identity, on exactly the same terms again.  All four defaults
    # are 0, validate_run_config refuses any other value, and none is read
    # unless sf_surface_physics=3 -- which no frozen configuration selects, so
    # appending them cannot move a frozen trajectory.  Unlike the MYNN and
    # Noah-MP blocks these are also WRF's own Registry defaults
    # (Registry.EM_COMMON:2535-2537 and Registry/registry.stoch:241), so a
    # namelist import of a stock ARW run lands on them without adjustment.
    from gpuwm.config import RUC_OPTION_IDENTITY
    for name, admitted in RUC_OPTION_IDENTITY.items():
        default = RunConfig.__dataclass_fields__[name].default
        assert default == admitted and type(default) is type(admitted), name
    for entry in GOLDEN.values():
        assert entry["sf_surface_physics"] not in (3, 4)
    # the golden snapshot carries exactly the current field set
    for key, entry in GOLDEN.items():
        assert set(entry) == set(names), key


def test_every_existing_legacy_toml_resolves_identically():
    """Re-resolve every legacy TOML in configs/ and compare dataclasses
    (as field dicts) against the pre-change golden snapshot."""
    tomls = sorted((REPO / "configs").glob("*.toml"))
    assert tomls, "configs/ directory is empty?"
    legacy = [p for p in tomls if not is_experiment_toml(p)]
    assert legacy, "no legacy TOMLs found in configs/"
    for path in legacy:
        key = f"configs/{path.name}"
        assert key in GOLDEN, (
            f"{key} has no golden freeze entry: pin new legacy TOMLs in "
            "tests/data/config_freeze_golden.json consciously.")
        cfg = load_config(path)
        assert dataclasses.asdict(cfg) == GOLDEN[key], key
        assert cfg.nested is False and cfg.grid_id == 1, key


def test_frozen_case_constructed_configs_resolve_identically():
    """Every frozen verify-case RunConfig constructor re-resolves to the
    pre-change golden values (frozen cases never migrate to the
    experiment path -- they construct RunConfig directly)."""
    for key, ctor in _frozen_constructors().items():
        cfg = ctor()
        assert key in GOLDEN, key
        assert dataclasses.asdict(cfg) == GOLDEN[key], key
        assert cfg.nested is False and cfg.grid_id == 1, key


def test_golden_inventory_matches_known_frozen_surface():
    """The golden file covers exactly the frozen surface: every known
    constructor plus every legacy TOML (no stale or orphaned pins)."""
    expected = set(_frozen_constructors())
    expected |= {f"configs/{p.name}"
                 for p in (REPO / "configs").glob("*.toml")
                 if not is_experiment_toml(p)}
    assert set(GOLDEN) == expected
