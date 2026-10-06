"""Source-request clocks stay explicit at the importer and recipe doors."""
from dataclasses import replace
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import tomllib

import pytest

from gpuwm.experiment import VerticalConfig, build_experiment, load_experiment

ROOT = Path(__file__).parents[1]
RECIPE = ROOT / "configs/recipes/hrrr_configuration_clock.toml"


def test_full_native_grid_clock_recipe_resolves_and_the_real_go_planner_reads_it(
        tmp_path, capsys):
    from gpuwm.cli import main
    from gpuwm.companion_domains import candidate_wps_text
    from gpuwm.hrrr_route_inputs import write_hrrr_route_inputs
    from gpuwm.physics_source_defaults import read_physics_selector_comment
    config = tmp_path / RECIPE.name
    config.write_bytes(RECIPE.read_bytes())
    exp = load_experiment(config)
    run = exp.root.run
    assert run.dx == run.dy == 3000
    assert exp.dt_exact(1) == Fraction(20)
    assert run.time_step_sound == 6
    assert run.use_adaptive_time_step is False
    assert run.terrain_clock == "pinned"
    before = config.read_bytes()
    assert main(["go", str(config), "--dry-run"]) == 0
    capsys.readouterr()
    assert config.read_bytes() == before
    raw = tomllib.loads(config.read_text())
    wps = candidate_wps_text(raw, exp, exp, config)
    written = write_hrrr_route_inputs(
        config, exp, wps_text=wps,
        writer=lambda path, text: path.write_text(text, encoding="utf-8"))
    namelist = next(path for path in written
                    if path.name.endswith(".namelist.input")
                    and not path.name.endswith(".stock.namelist.input"))
    assert read_physics_selector_comment(namelist.read_text())["terrain_clock"] == "pinned"


@pytest.mark.parametrize("source", ("hrrr", "hrrr-prs", "hrrr-native"))
def test_bare_source_recipe_pins_its_declared_clock_without_replacing_it(source):
    from gpuwm.domain_wizard import (
        experiment_from_text, render_config, resolved_physics_profile)
    text = render_config(
        name="clock-source-request", start_time=datetime(2026, 10, 2, 21),
        hours=1, projection={"map_proj": "lambert", "ref_lat": 38.5,
            "ref_lon": -97.5, "truelat1": 38.5, "truelat2": 38.5,
            "stand_lon": -97.5}, dims=[(50, 50)], ratios=(),
        root_dx_m=3000, fetch_hints={"source": source}, case_data=None,
        profile=resolved_physics_profile(source, None))
    exp = experiment_from_text(text, source="clock source recipe")
    assert exp.root.run.terrain_clock == "pinned"
    assert exp.root.run.use_adaptive_time_step is False
    # The resized wizard clock remains the clock its own table authors.
    assert exp.root.run.time_step_sound == 4


@pytest.mark.parametrize("name,step", (
    ("hrrr_native_3km_demo", 15), ("hrrr_native_quick_demo", 15),
    ("hrrr_prs_3km_demo", 15), ("hrrr_prs_demo", 72),
))
def test_source_demos_retain_their_configured_clock(name, step):
    exp = load_experiment(ROOT / "configs" / (name + ".toml"))
    assert exp.dt_exact(1) == Fraction(step)
    assert exp.root.run.time_step_sound == 4
    assert exp.root.run.terrain_clock == "pinned"


@pytest.mark.parametrize("step,sound", ((20, 6), (15, 4)))
def test_native_builder_uses_the_operational_bound_only_for_its_20s_3km_recipe(
        step, sound):
    from gpuwm.ingest.hrrr_target import HrrrTargetDomain
    from tools.hrrr_single_domain_benchmark import _experiment_tables
    target = replace(HrrrTargetDomain.legacy_500x500(),
                     nx=50, ny=50, nz=12, dx_m=3000, dy_m=3000,
                     time_step_seconds=step)
    vertical = VerticalConfig(
        eta_levels=tuple(1 - i / 12 for i in range(13)),
        p_top=5000, hybrid_opt=2, etac=.2)
    raw, _ = _experiment_tables(vertical, target=target, run_seconds=3600)
    run = build_experiment(raw, source="native clock recipe").root.run
    assert run.terrain_clock == "pinned"
    assert run.time_step_sound == sound
    assert run.use_adaptive_time_step is False


@pytest.mark.parametrize("value", (None, False, "off"))
def test_route_clock_selector_metadata_rejects_unknown_or_untyped_values(value):
    from gpuwm.physics_source_defaults import with_physics_selector_comment
    with pytest.raises(ValueError, match="terrain_clock"):
        with_physics_selector_comment("&domains\n/\n", {"terrain_clock": value})
