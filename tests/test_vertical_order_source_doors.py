"""Vertical source defaults apply to the root and preserve explicit columns."""
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from gpuwm.experiment import VerticalConfig, build_experiment, load_experiment
from gpuwm.physics_source_defaults import recipe_root_defaults, with_recipe_root_defaults

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/fixtures/source_requests"


def test_explicit_order3_keeps_exact_current_staging_importer_bytes(tmp_path):
    from test_namelist_import import _import_with
    baseline = json.loads((FIXTURE / "identity-e62a72e48-vadv.json").read_text())
    assert baseline["revision"] == "e62a72e480d7dafff7381db4c9d35eca22285a87"
    explicit, _ = _import_with(tmp_path,
        extra_dynamics=" v_sca_adv_order = 3, 3,\n v_mom_adv_order = 3, 3,\n")
    omitted, _ = _import_with(tmp_path)
    assert explicit == omitted
    assert hashlib.sha256(explicit.encode()).hexdigest() == baseline["explicit_order3"]


def test_root_recipe_defaults_preserve_explicit_shared_root_and_nest_choices():
    defaults = {"v_sca_adv_order": 5, "v_mom_adv_order": 5}
    domains = [{"parent_id": 0, "v_mom_adv_order": 3},
               {"parent_id": 1, "v_sca_adv_order": 3, "v_mom_adv_order": 5}]
    shared = {}
    assert with_recipe_root_defaults(shared, domains, defaults) is domains
    assert shared == {}
    assert domains[0]["v_sca_adv_order"] == 5
    assert domains[0]["v_mom_adv_order"] == 3
    assert domains[1] == {"parent_id": 1, "v_sca_adv_order": 3, "v_mom_adv_order": 5}
    shared = {"v_sca_adv_order": 3, "v_mom_adv_order": 3}
    domains = [{"parent_id": 0}, {"parent_id": 1}]
    with_recipe_root_defaults(shared, domains, defaults)
    assert domains == [{"parent_id": 0}, {"parent_id": 1}]


@pytest.mark.parametrize("source", ("gfs", "era5", "rrfs"))
def test_other_source_requests_have_no_root_defaults(source):
    assert recipe_root_defaults(source) == {}
    domains = [{"parent_id": 0}, {"parent_id": 1}]
    assert with_recipe_root_defaults({}, domains, recipe_root_defaults(source)) is domains
    assert domains == [{"parent_id": 0}, {"parent_id": 1}]


@pytest.mark.parametrize("source", ("hrrr", "hrrr-prs", "hrrr-native"))
def test_bare_nested_source_recipe_runs_root5_and_nest3(tmp_path, capsys, source):
    from gpuwm.cli import main
    from gpuwm.hrrr_route_inputs import route_input_paths
    from gpuwm.namelist_import import import_namelists, parse_namelist
    path = tmp_path / "vertical.toml"
    assert main([
        "domain", "--point=35.2,-97.4", "--card", "24gb", "--ladder", "12-3",
        "--source", source, "--cycle", "2026-10-02T21", "--hours", "1",
        "--out", str(path),
    ]) == 0
    capsys.readouterr()
    exp = load_experiment(path)
    assert (exp.root.run.v_sca_adv_order, exp.root.run.v_mom_adv_order) == (5, 5)
    assert (exp.domain(2).run.v_sca_adv_order, exp.domain(2).run.v_mom_adv_order) == (3, 3)
    raw = tomllib.loads(path.read_text())
    assert "v_sca_adv_order" not in raw["shared"]
    assert "v_mom_adv_order" not in raw["shared"]
    before = path.read_bytes()
    assert main(["go", str(path), "--dry-run"]) == 0
    capsys.readouterr()
    assert path.read_bytes() == before
    if source == "hrrr":
        companions = route_input_paths(path)
        parsed = parse_namelist(companions["namelist_input"])
        assert parsed["dynamics"]["v_sca_adv_order"] == [5, 3]
        assert parsed["dynamics"]["v_mom_adv_order"] == [5, 3]
        # Read the exact files the runtime reads, separately from the
        # preparation identity where vertical orders are inert.
        text, _ = import_namelists(companions["wps_namelist"], companions["namelist_input"])
        replay = build_experiment(tomllib.loads(text), source="vertical route replay")
        assert [domain.run.v_sca_adv_order for domain in replay.domains] == [5, 3]
        assert [domain.run.v_mom_adv_order for domain in replay.domains] == [5, 3]


@pytest.mark.parametrize("relative,step", (
    ("hrrr_native_3km_demo.toml", 15), ("hrrr_native_quick_demo.toml", 15),
    ("hrrr_prs_3km_demo.toml", 15), ("hrrr_prs_demo.toml", 72),
    ("hrrr_v4_vertical_order5.toml", 15), ("recipes/hrrr_configuration_clock.toml", 20),
))
def test_shipped_source_templates_state_the_root_orders_without_moving_clocks(relative, step):
    path = ROOT / "configs" / relative
    raw = tomllib.loads(path.read_text())
    assert "v_sca_adv_order" not in raw["shared"]
    assert "v_mom_adv_order" not in raw["shared"]
    assert raw["domain"][0]["v_sca_adv_order"] == 5
    assert raw["domain"][0]["v_mom_adv_order"] == 5
    exp = load_experiment(path)
    assert (exp.root.run.v_sca_adv_order, exp.root.run.v_mom_adv_order) == (5, 5)
    assert exp.dt_exact(1) == Fraction(step)


def test_native_source_builder_authors_order5_only_in_its_root_table():
    from gpuwm.ingest.hrrr_target import HrrrTargetDomain
    from tools.hrrr_single_domain_benchmark import _experiment_tables
    target = replace(HrrrTargetDomain.legacy_500x500(), nx=50, ny=50, nz=12)
    vertical = VerticalConfig(eta_levels=tuple(1 - i / 12 for i in range(13)),
                              p_top=5000., hybrid_opt=2, etac=.2)
    raw, _ = _experiment_tables(vertical, target=target, run_seconds=3600)
    assert "v_sca_adv_order" not in raw["shared"]
    assert "v_mom_adv_order" not in raw["shared"]
    assert raw["domain"][0]["v_sca_adv_order"] == raw["domain"][0]["v_mom_adv_order"] == 5
    run = build_experiment(raw, source="native vertical recipe").root.run
    assert (run.v_sca_adv_order, run.v_mom_adv_order) == (5, 5)
