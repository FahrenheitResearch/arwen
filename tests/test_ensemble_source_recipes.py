from datetime import datetime, timedelta, timezone

import pytest

from gpuwm.ensemble.recipes import SourceTrajectory, build_recipe, ensemble_population

START = datetime(2026, 9, 28, 12, tzinfo=timezone.utc)


def plan(**kwargs):
    options = dict(source="gfs", cycle=START, start=START, end=START + timedelta(hours=6), count=4, base_seed=42)
    options.update(kwargs)
    return build_recipe(**options)


def test_automatic_singleton_preserves_base():
    recipe = plan(count=1)
    assert recipe.kind == "control"
    assert recipe.acquisitions() == (recipe.base,)


def test_explicit_native_members_and_streaming_requests():
    recipe = plan(source="gefs", kind="input-ensemble")
    assert [row.trajectory.member for row in recipe.members] == ["c00", "p01", "p02", "p03"]
    assert len(recipe.acquisitions()) == 4
    for row in recipe.members:
        args = row.trajectory.fetch_argv(recipe.start, recipe.end, "inputs")
        assert "--as-posted" in args
        assert args[args.index("--mode") + 1] == "full-file"
        assert "--forecast-start-hour" in args
        assert args[args.index("--member") + 1] == row.trajectory.member


def test_gefs_population_does_not_include_statistics():
    members = ensemble_population("gefs", START)
    assert len(members) == 31
    assert len(ensemble_population("gefs", START, perturbed_only=True)) == 30
    assert {item.member for item in members}.isdisjoint({"geavg", "gespr"})


def test_recentered_population_independent_of_requested_count():
    donor = SourceTrajectory("gefs", START)
    small = plan(source="hrrr", kind="recentered", count=2, donor=donor)
    big = plan(source="hrrr", kind="recentered", count=20, donor=donor)
    assert small.donor_population == big.donor_population
    assert small.members == big.members[:2]
    assert len(big.acquisitions()) == 31


def test_time_lags_keep_valid_times_and_skip_expired_horizons():
    recipe = plan(source="hrrr", kind="time-lagged", count=8)
    assert len({member.trajectory.identity for member in recipe.members}) == 8
    for member in recipe.members:
        first, last = member.trajectory.window(recipe.start, recipe.end)
        assert member.trajectory.cycle + timedelta(hours=first) == recipe.start
        assert member.trajectory.cycle + timedelta(hours=last) == recipe.end
    with pytest.raises(ValueError, match="repeating trajectories"):
        plan(source="hrrr", kind="time-lagged", count=20, max_lag_hours=3)


def test_multimodel_rejects_duplicate_trajectories():
    first, second = SourceTrajectory("hrrr", START), SourceTrajectory("gfs", START)
    with pytest.raises(ValueError, match="repeats a source trajectory"):
        plan(kind="multi-model", count=3, trajectories=(first, second, first))
    recipe = plan(kind="multi-model", count=2, trajectories=(first, second))
    assert recipe.members[0].trajectory.source != recipe.members[1].trajectory.source
    older = SourceTrajectory("hrrr", START - timedelta(hours=1))
    with pytest.raises(ValueError, match="selected multi-model roster"):
        plan(kind="multi-model", count=2, trajectories=(first, older, second))


def test_recentered_unposted_source_knots_acquire_native_brackets():
    recipe = plan(source="hrrr", cycle=START + timedelta(hours=1), start=START + timedelta(hours=1),
                  end=START + timedelta(hours=13), kind="recentered", donor=SourceTrajectory("gefs", START))
    assert recipe.acquisition_window(recipe.members[0].trajectory) == (0, 15)
    assert recipe.start == START + timedelta(hours=1)
    assert recipe.end == START + timedelta(hours=13)


def test_unsupported_default_cannot_imply_calibrated_spread():
    with pytest.raises(ValueError, match="no observation-calibrated"):
        plan(source="hrrr")


def test_trajectory_requires_time_zone_and_complete_horizon():
    with pytest.raises(ValueError, match="UTC offset"):
        SourceTrajectory("gfs", START.replace(tzinfo=None))
    with pytest.raises(ValueError, match="requested boundary"):
        plan(source="hrrr", end=START + timedelta(hours=100))


def test_recipe_identity_changes_for_different_boundary_or_member_roster():
    a = plan(source="gefs")
    b = plan(source="gefs", count=5)
    c = plan(source="gefs", end=START + timedelta(hours=9))
    assert len({a.sha256, b.sha256, c.sha256}) == 3


def test_member_replay_keeps_seed_index_and_complete_donor_population():
    donor = SourceTrajectory("gefs", START)
    recipe = plan(source="hrrr", kind="recentered", count=20, donor=donor)
    replay = recipe.select_members((12,))
    assert replay.members == (recipe.members[12],)
    assert replay.donor_population == recipe.donor_population
    assert recipe.select_members((9, 2)).members == (recipe.members[9], recipe.members[2])
    with pytest.raises(ValueError, match="distinct indices"):
        recipe.select_members((True,))


def test_operational_ensemble_defaults_are_adapter_rows():
    assert plan(source="gfs").members[1].trajectory.source == "gefs"
    population = ensemble_population("ecmwf-ens", START)
    assert len(population) == 51
    assert population[0].source == "ecmwf-open-data"
    assert population[0].member is None
    assert len(ensemble_population("ecmwf-ens", START, perturbed_only=True)) == 50


def test_future_input_model_is_adapter_metadata_only(monkeypatch):
    from dataclasses import replace
    from gpuwm import source_adapters
    from gpuwm.source_cycles import CycleGrid
    ordinary = source_adapters.get_source_adapter
    future = replace(ordinary("gfs"), source_id="future-model", aliases=(),
                     ensemble_source="gefs", cycle_grid=CycleGrid((0, 6, 12, 18)))
    monkeypatch.setattr(source_adapters, "get_source_adapter",
                        lambda name: future if name == "future-model" else ordinary(name))
    result = plan(source="future-model")
    assert result.base.source == "future-model"
    assert [m.trajectory.member for m in result.members] == ["c00", "p01", "p02", "p03"]


def test_new_member_grammar_controls_and_population_order_are_metadata(monkeypatch, tmp_path):
    import json
    from dataclasses import replace
    from gpuwm import fetch_routes, source_adapters, source_authorities
    from gpuwm.source_cycles import CycleGrid
    lookup, route_lookup = source_adapters.get_source_adapter, fetch_routes.route_for
    grammar_lookup = source_authorities.packaged_member_grammar
    document = json.loads(grammar_lookup(lookup("gefs").member_set).read_text())
    document["name"] = "future-ensemble"
    document["declared_member_count"] = 4
    document["cycle_hours"] = [4, 16]
    control = document["classes"]["control"]
    control.update(ordinals=[3], member_id="c03", token="c03")
    perturbed = document["classes"]["perturbed"]
    perturbed.update(ordinals=[2, 0, 1], token="p{ordinal:02d}")
    document["classes"] = {"perturbed": perturbed, "control": control}
    grammar_path = tmp_path / "members.json"
    grammar_path.write_text(json.dumps(document))
    grammar_id = "future-member-grammar"
    future = replace(lookup("gefs"), source_id="future-ensemble", aliases=(),
                     member_set=grammar_id, cycle_grid=CycleGrid((4, 16)))
    route = replace(route_lookup("gefs"), source_id="future-ensemble", cycle_hours=(4, 16),
                    members={"default": "c03", "perturbed_range": [0, 2],
                             "perturbed_name_format": "p%02d"})
    monkeypatch.setattr(source_adapters, "get_source_adapter",
                        lambda name: future if name == future.source_id else lookup(name))
    monkeypatch.setattr(source_authorities, "packaged_member_grammar",
                        lambda name: grammar_path if name == grammar_id else grammar_lookup(name))
    monkeypatch.setattr(fetch_routes, "route_for",
                        lambda name: route if name == future.source_id else route_lookup(name))
    time = START.replace(hour=16)
    all_members = ensemble_population(future.source_id, time)
    donor_members = ensemble_population(future.source_id, time, perturbed_only=True)
    assert [member.member for member in all_members] == ["c03", "p00", "p01", "p02"]
    assert [member.member for member in donor_members] == ["p00", "p01", "p02"]
