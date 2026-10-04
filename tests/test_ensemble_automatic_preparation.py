"""Native window/authority and owned-process lifecycle checks; no model execution."""
from datetime import timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.automatic_preparation import (
    AutomaticPreparationOwner, materialize_source_controls, preflight_source_head)
from gpuwm.ensemble.automatic_sources import EnsembleSourceContext, NativeSourceTemplate, resolve_ensemble_sources
from gpuwm.ensemble.recipes import SourceTrajectory, build_recipe


def test_native_donor_window_uses_real_config_and_wps_writers(tmp_path):
    import tomllib
    from test_gfs_initial_perturbation import _config
    from gpuwm.experiment import load_experiment
    from gpuwm.toml_document import emit_experiment_toml
    from gpuwm.companion_domains import candidate_wps_text
    from gpuwm.fortran_namelist import parse_namelist
    config = _config(tmp_path, domains=1)
    original = load_experiment(config)
    raw = tomllib.loads(config.read_text())
    start = original.start_time + timedelta(hours=1)
    raw["experiment"].update(start_time=start, run_seconds=43200)
    raw["fetch"] = dict(source="hrrr", cycle=start.strftime("%Y-%m-%dT%H"), hours=12, cadence=1)
    config.write_text(emit_experiment_toml(raw))
    base = load_experiment(config)
    wps = config.with_suffix(".namelist.wps")
    wps.write_text(candidate_wps_text(raw, original, base, config, original_wps=wps))
    original_bytes = config.read_bytes(), wps.read_bytes()
    start = start.replace(tzinfo=timezone.utc)
    recipe = build_recipe(source="hrrr", cycle=start, start=start, end=start+timedelta(hours=12),
        count=2, base_seed=7, kind="recentered", donor=SourceTrajectory("gefs", start-timedelta(hours=1)))
    template = NativeSourceTemplate.capture("mapped_composition_v1",
        ("--experiment-config", str(config), "--wps-namelist", str(wps)))
    arguments = materialize_source_controls(recipe, recipe.members[0].trajectory, template, root=tmp_path/"donor")
    derived_path = Path(arguments[arguments.index("--experiment-config")+1])
    derived = load_experiment(derived_path)
    assert derived.start_time == original.start_time
    assert derived.run_seconds == 15*3600
    assert derived.projection == base.projection
    assert derived.root.run.nx == base.root.run.nx and derived.root.run.ny == base.root.run.ny
    assert derived.root.run.mp_physics == base.root.run.mp_physics
    share = parse_namelist(arguments[arguments.index("--wps-namelist")+1])["share"]
    assert share["interval_seconds"] == [10800]
    assert share["start_date"] == [original.start_time.strftime("%Y-%m-%d_%H:%M:%S")]
    assert (config.read_bytes(), wps.read_bytes()) == original_bytes
    binding = json.loads((tmp_path/"donor/native-window.json").read_bytes())
    assert binding["requested_start"] == recipe.start.isoformat()
    assert binding["acquisition_end"] == (recipe.start+timedelta(hours=14)).isoformat()


def test_automatic_singleton_allocates_no_owner_directory(tmp_path):
    from datetime import datetime
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    ordinary = object()
    context = EnsembleSourceContext(start, start+timedelta(hours=12), ordinary)
    selection = resolve_ensemble_sources(1, context)
    owner = AutomaticPreparationOwner(selection, context, output_root=tmp_path/"owner").start()
    assert owner.inputs is ordinary and owner.session_arguments == {} and not owner.processes
    owner.finish()
    assert not (tmp_path/"owner").exists()


def test_reference_fallback_refuses_before_owner_or_acquisition(tmp_path, monkeypatch):
    from datetime import datetime
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    context = EnsembleSourceContext(start, start+timedelta(hours=12), object())
    monkeypatch.setattr(AutomaticPreparationOwner, "_launch",
        lambda *a, **k: pytest.fail("uncalibrated fallback started acquisition"))
    with pytest.raises(ValueError, match="calibrated against observations"):
        resolve_ensemble_sources(3, context)
    assert not (tmp_path/"owner").exists()


def test_tree_head_is_admitted_by_the_existing_tree_reader(tmp_path, monkeypatch):
    from datetime import datetime
    from gpuwm import stage_cli, prepared_domain_tree_forecast
    config = tmp_path/"experiment.toml"
    config.write_text("tree authority fixture")
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    context = EnsembleSourceContext(start, start+timedelta(hours=12), None,
        preflight_options={"devices": 2})
    specification = SimpleNamespace(prepared_root=tmp_path/"native-tree",
        native_arguments=("--experiment-config", str(config)))
    monkeypatch.setattr(stage_cli, "resolve_head_bundle", lambda *args: {"layout": "tree"})
    calls = []
    expected = object()
    monkeypatch.setattr(prepared_domain_tree_forecast, "preflight_prepared_tree",
        lambda **kwargs: calls.append(kwargs) or expected)
    assert preflight_source_head(context, specification, {"head_sha256": "a"*64}) is expected
    assert calls[0]["prepared_head_sha256"] == "a"*64
    assert calls[0]["devices"] == 2
    assert calls[0]["experiment_config"] == config
