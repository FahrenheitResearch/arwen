"""Named source requests select a generation without changing other defaults."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from gpuwm.config import RunConfig
from gpuwm.experiment import build_experiment
from gpuwm.physics_source_defaults import (
    namelist_physics_defaults, recipe_physics_defaults, with_physics_defaults_text)

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/fixtures/source_requests"


@pytest.mark.parametrize("name", ("hrrr_wrf.nl", "hrrr_wrf.nl.c18c"))
def test_retained_clone_namelist_imports_the_fork(tmp_path, name):
    from gpuwm.namelist_import import import_namelists
    namelist = tmp_path / name
    namelist.write_bytes((FIXTURE / "hrrr_wrf.nl.c18c").read_bytes())
    text, report = import_namelists(FIXTURE / "hrrr_namelist.wps.c18", namelist)
    exp = build_experiment(tomllib.loads(text), source="source-request import")
    assert exp.root.run.sf_sfclay_physics == 5
    assert exp.root.run.mynn_sfclay_variant == "gsl_wrf39"
    assert exp.root.run.bl_mynn_version == "gsd_41"
    assert exp.root.run.ra_rrtmg_variant == "rrtmg_legacy"
    assert any(default.key == "mynn_sfclay_variant"
               and default.value == "gsl_wrf39"
               for default in report.defaults_applied)


def test_unnamed_namelist_keeps_the_global_default(tmp_path):
    from gpuwm.namelist_import import import_namelists
    namelist = tmp_path / "namelist.input"
    namelist.write_bytes((FIXTURE / "hrrr_wrf.nl.c18c").read_bytes())
    text, _ = import_namelists(FIXTURE / "hrrr_namelist.wps.c18", namelist)
    # The staged sea-ice importer change predates this surface generation.
    # This pin was emitted by that committed importer, not this module.
    baseline = json.loads((FIXTURE / "identity-6c2dd1535.json").read_text())
    assert hashlib.sha256(text.encode()).hexdigest() == baseline["namelist"]
    assert "mynn_sfclay_variant" not in tomllib.loads(text)["shared"]
    assert build_experiment(tomllib.loads(text), source="unnamed import").root.run.mynn_sfclay_variant == "wrf_461"


def _recipe(source, *, profile=None):
    from gpuwm.domain_wizard import render_config
    from gpuwm.physics_compat import MYNN_RUC_PROFILE_ID
    return render_config(
        name="source-request", start_time=datetime(2026, 10, 2, 21), hours=1,
        projection={"map_proj": "lambert", "ref_lat": 38.5, "ref_lon": -97.5,
                    "truelat1": 38.5, "truelat2": 38.5, "stand_lon": -97.5},
        dims=[(50, 50)], ratios=(), root_dx_m=3000,
        fetch_hints={"source": source}, case_data=None,
        profile=MYNN_RUC_PROFILE_ID if profile is None else profile)


@pytest.mark.parametrize("source", ("hrrr", "hrrr-prs", "hrrr-native"))
def test_authored_recipe_uses_the_fork(source):
    from gpuwm.physics_compat import THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
    text = _recipe(source, profile=THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID)
    from gpuwm.domain_wizard import experiment_from_text
    exp = experiment_from_text(text, source="authored recipe")
    assert exp.root.run.sf_sfclay_physics == 5
    assert exp.root.run.mynn_sfclay_variant == "gsl_wrf39"


@pytest.mark.parametrize("source", ("hrrr", "hrrr-prs", "hrrr-native"))
def test_bare_source_recipe_executes_the_surface_fork(tmp_path, capsys, source):
    """The source recommendation reaches MYNN without a profile override."""
    from gpuwm.cli import main
    from gpuwm.experiment import load_experiment
    from gpuwm.hrrr_route_inputs import route_input_paths, verify_round_trip

    config = tmp_path / "native.toml"
    assert main([
        "domain", "--point=35.2,-97.4", "--card", "24gb",
        "--ladder", "12-3", "--source", source,
        "--cycle", "2026-10-02T21", "--hours", "1",
        "--out", str(config),
    ]) == 0
    capsys.readouterr()
    exp = load_experiment(config)
    for domain in exp.domains:
        assert domain.run.sf_sfclay_physics == 5
        assert domain.run.bl_pbl_physics == 5
        assert domain.run.mynn_sfclay_variant == "gsl_wrf39"
    if source == "hrrr":
        paths = route_input_paths(config)
        verify_round_trip(exp, paths["wps_namelist"], paths["namelist_input"])
    before = config.read_bytes()
    assert main(["go", str(config), "--dry-run"]) == 0
    capsys.readouterr()
    assert config.read_bytes() == before


def test_native_recipe_builder_uses_the_same_table():
    from dataclasses import replace
    from gpuwm.experiment import VerticalConfig
    from gpuwm.ingest.hrrr_target import HrrrTargetDomain
    from tools.hrrr_single_domain_benchmark import _experiment_tables
    target = replace(HrrrTargetDomain.legacy_500x500(), nx=50, ny=50, nz=12)
    vertical = VerticalConfig(eta_levels=tuple(1 - i / 12 for i in range(13)),
                              p_top=5000., hybrid_opt=2, etac=.2)
    raw, _ = _experiment_tables(vertical, target=target, run_seconds=3600)
    run = build_experiment(raw, source="native recipe").root.run
    assert run.sf_sfclay_physics == 5
    assert run.bl_pbl_physics == 5
    assert run.mynn_sfclay_variant == "gsl_wrf39"
    assert run.bl_mynn_version == "gsd_41"
    assert run.bl_mynn_mixlength == 2
    assert run.bl_mynn_gsd41_unsquared_qtke is False
    assert run.ra_rrtmg_variant == "rrtmg_legacy"
    assert run.mp_physics == 28
    assert (run.aer_init_opt, run.wif_input_opt) == (1, 1)


def test_non_source_requests_keep_the_same_bytes_and_default():
    baseline = json.loads((FIXTURE / "identity-7260e48f4.json").read_text())
    for source in ("gfs", "era5", "rrfs"):
        assert recipe_physics_defaults(source) == {}
        text = _recipe(source)
        assert hashlib.sha256(text.encode()).hexdigest() == baseline["recipes"][source]
        assert "mynn_sfclay_variant" not in tomllib.loads(text)["shared"]
        assert with_physics_defaults_text(text, recipe_physics_defaults(source)) is text
    assert namelist_physics_defaults("namelist.input") == {}
    assert RunConfig.__dataclass_fields__["mynn_sfclay_variant"].default == "wrf_461"


def test_explicit_legacy_selection_is_not_replaced():
    text = ('[shared]\nmynn_sfclay_variant = "wrf_461"\n'
            'terrain_clock = "measured"\n')
    shared = tomllib.loads(with_physics_defaults_text(
        text, recipe_physics_defaults("hrrr")))["shared"]
    assert shared["mynn_sfclay_variant"] == "wrf_461"
    assert shared["terrain_clock"] == "measured"


def test_legacy_surface_selection_keeps_new_source_clock_default():
    text = '[shared]\nmynn_sfclay_variant = "wrf_461"\n'
    updated = with_physics_defaults_text(text, recipe_physics_defaults("hrrr"))
    shared = tomllib.loads(updated)["shared"]
    assert shared["mynn_sfclay_variant"] == "wrf_461"
    assert shared["terrain_clock"] == "pinned"


@pytest.mark.parametrize("name", ("hrrr_native_3km_demo", "hrrr_native_quick_demo",
                                   "hrrr_prs_3km_demo", "hrrr_prs_demo"))
def test_shipped_source_templates_declare_the_fork(name):
    from gpuwm.experiment import load_experiment
    path = ROOT / "configs" / f"{name}.toml"
    raw = tomllib.loads(path.read_text())
    assert raw["shared"]["mynn_sfclay_variant"] == "gsl_wrf39"
    exp = load_experiment(path)
    for domain in exp.domains:
        assert domain.run.sf_sfclay_physics == 5
        assert domain.run.bl_pbl_physics == 5
        assert domain.run.mynn_sfclay_variant == "gsl_wrf39"
    if name.startswith("hrrr_native_"):
        from gpuwm.hrrr_route_inputs import route_input_paths, verify_round_trip
        paths = route_input_paths(ROOT / "configs" / f"{name}.toml")
        verify_round_trip(exp, paths["wps_namelist"], paths["namelist_input"])
