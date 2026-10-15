"""Named configuration arms retain their own ordinary source trajectories."""
from datetime import datetime, timedelta, timezone

import pytest

from gpuwm.ensemble.request import EnsembleRequest
from gpuwm.ensemble.recipes import SourceTrajectory, build_recipe
from gpuwm.ensemble.door import request_for_payload
from gpuwm.toml_document import emit_experiment_toml


def test_named_arms_combine_sources_and_cycles_without_a_second_recipe():
    cycle = datetime(2022, 6, 13, 18, tzinfo=timezone.utc)
    rows = ({"source": "hrrr", "cycle": "2022-06-13T18"},
            {"source": "gfs", "cycle": "2022-06-13T12"})
    variants = ({"name": "reference"},
                {"name": "dry", "surface": {"soil_moisture_scale": 0.9}})
    request = EnsembleRequest(members=2, recipe="member-roster",
                              trajectories=rows, member_variants=variants)
    trajectories = tuple(SourceTrajectory(row["source"],
        datetime.fromisoformat(row["cycle"]).replace(tzinfo=timezone.utc)) for row in rows)
    plan = build_recipe(source="hrrr", cycle=cycle, start=cycle,
        end=cycle + timedelta(hours=6), count=2, base_seed=0,
        kind=request.recipe, trajectories=trajectories, member_variants=request.member_variants)
    assert tuple(member.trajectory for member in plan.members) == trajectories
    assert plan.acquisition_window(trajectories[1]) == (6, 12)
    assert plan.member_variants == request.member_variants


def test_extra_roster_trajectory_refuses_ambiguous_arm_ownership():
    with pytest.raises(ValueError, match="ownership ambiguous"):
        EnsembleRequest(members=1, recipe="member-roster",
            member_variants=({"name": "reference"},), trajectories=(
                {"source": "hrrr", "cycle": "2022-06-13T18"},
                {"source": "gfs", "cycle": "2022-06-13T12"}))


def test_explicit_roster_flag_retains_configured_trajectory_ownership():
    raw = {"ensemble": {"members": 2, "recipe": "member-roster",
        "trajectories": [{"source": "hrrr", "cycle": "2022-06-13T18"},
                         {"source": "gfs", "cycle": "2022-06-13T12"}],
        "member_variants": [{"name": "reference"},
                            {"name": "dry", "surface": {"soil_moisture_scale": 0.9}}]}}
    implicit = request_for_payload(emit_experiment_toml(raw).encode())
    explicit = request_for_payload(emit_experiment_toml(raw).encode(), recipe="member-roster")
    assert explicit == implicit
    assert len(explicit.trajectories) == 2


def test_plain_roster_keeps_original_base_trajectory_and_receipt():
    cycle = datetime(2022, 6, 13, 18, tzinfo=timezone.utc)
    variants = ({"name": "reference"},
                {"name": "dry", "surface": {"soil_moisture_scale": 0.9}})
    options = dict(source="hrrr", cycle=cycle, start=cycle,
        end=cycle + timedelta(hours=1), count=2, base_seed=0,
        kind="member-roster", member_variants=variants)
    implicit = build_recipe(**options)
    explicit_empty = build_recipe(**options, trajectories=())
    assert implicit == explicit_empty
    assert all(member.trajectory == implicit.base for member in implicit.members)
