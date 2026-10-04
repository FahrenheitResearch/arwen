from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.recipes import build_recipe, SourceTrajectory
from gpuwm.ensemble.member_preparation import (
    SourceManifestBinding, PreparedMemberInput, PreparedMemberRoster, prepare_member_roster,
)


def _recipe(count=2):
    start = datetime(2024, 5, 25, 18, tzinfo=timezone.utc)
    return build_recipe(source="gfs", cycle=start, start=start,
                        end=start + timedelta(hours=12), count=count, base_seed=42, kind="input-ensemble")


def _bound(tmp_path, trajectory):
    path = tmp_path / (trajectory.identity + ".json")
    doc = dict(source=trajectory.source, cycle=trajectory.cycle.isoformat(), member=trajectory.member)
    path.write_text(json.dumps(doc))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return SourceManifestBinding(trajectory, path, sha, {**doc, "manifest_sha256": sha})


def _prepared(tmp_path, recipe, member):
    return PreparedMemberInput(member.index, member.seed, object(), object(), object(),
        member.trajectory, (_bound(tmp_path, recipe.base if recipe.kind == "recentered" else member.trajectory),),
        (recipe.start, recipe.start + timedelta(hours=6), recipe.end),
        "b" * 64, recipe.sha256)


def test_native_preparer_receives_each_actual_trajectory_and_shared_geometry_once(tmp_path):
    recipe = _recipe()
    geometry, calls = object(), []
    def prepare(**kwargs):
        calls.append(kwargs)
        return _prepared(tmp_path, recipe, kwargs["member"])
    roster = prepare_member_roster(recipe, shared_geometry=geometry, geometry_sha256="b" * 64,
                                  native_preparer=SimpleNamespace(prepare_member=prepare))
    assert [call["member"].trajectory.member for call in calls] == [m.trajectory.member for m in recipe.members]
    assert len({call["member"].trajectory.member for call in calls}) == 2
    assert all(call["shared_geometry"] is geometry for call in calls)
    assert roster.select((1, 0)) == (roster.members[1], roster.members[0])
    assert roster.members[0].initial is not roster.members[1].initial
    assert roster.members[0].boundaries is not roster.members[1].boundaries


def test_single_member_replay_retains_original_index_and_seed(tmp_path):
    recipe = _recipe(4)
    replay = recipe.select_members((3,))
    prepared = _prepared(tmp_path, replay, replay.members[0])
    roster = PreparedMemberRoster(replay, (prepared,), shared_geometry_sha256="b" * 64)
    assert roster.receipt()["member_order"] == [3]
    assert roster.members[0].seed == recipe.members[3].seed
    with pytest.raises(ValueError, match="outside"):
        roster.select((0,))


def test_manifest_tampering_and_same_name_wrong_trajectory_are_rejected(tmp_path):
    recipe = _recipe()
    binding = _bound(tmp_path, recipe.members[0].trajectory)
    binding.path.write_text("{}")
    with pytest.raises(ValueError, match="bound bytes"):
        binding.verify()
    binding = _bound(tmp_path, recipe.members[0].trajectory)
    wrong = replace(binding, trajectory=recipe.members[1].trajectory,
                    verification={**binding.verification, "member": recipe.members[1].trajectory.member})
    with pytest.raises(ValueError, match="manifest bytes"):
        wrong.verify()


def test_boundaries_must_keep_same_member_and_cover_entire_window(tmp_path):
    recipe = _recipe()
    member = _prepared(tmp_path, recipe, recipe.members[0])
    with pytest.raises(ValueError, match="timeline"):
        replace(member, boundary_valid_times=(recipe.start, recipe.start + timedelta(hours=6))).verify(
            recipe, shared_geometry_sha256="b" * 64)
    with pytest.raises(ValueError, match="same source member"):
        replace(member, source_manifests=(_bound(tmp_path, recipe.members[1].trajectory),)).verify(
            recipe, shared_geometry_sha256="b" * 64)
    with pytest.raises(ValueError, match="coordinate"):
        member.verify(recipe, shared_geometry_sha256="c" * 64)


def test_recentered_replay_requires_whole_donor_population_not_selected_members(tmp_path):
    start = datetime(2024, 5, 25, 18, tzinfo=timezone.utc)
    recipe = build_recipe(source="gfs", cycle=start, start=start,
        end=start + timedelta(hours=12), count=2, base_seed=42, kind="recentered",
        donor=SourceTrajectory("gefs", start))
    replay = recipe.select_members((1,))
    member = _prepared(tmp_path, replay, replay.members[0])
    selected_only = tuple(_bound(tmp_path, item.trajectory) for item in replay.members)
    with pytest.raises(ValueError, match="complete fixed donor"):
        replace(member, donor_manifests=selected_only).verify(replay, shared_geometry_sha256="b" * 64)
    all_donors = tuple(_bound(tmp_path, donor) for donor in recipe.donor_population)
    record = replace(member, donor_manifests=all_donors).verify(replay, shared_geometry_sha256="b" * 64)
    assert len(record["donor_manifests"]) == len(recipe.donor_population)
    assert record["member_id"] == 1 and record["seed"] == recipe.members[1].seed
    assert record["initialization_source_sha256"] == recipe.base.identity
    assert record["trajectory_sha256"] == recipe.members[1].trajectory.identity
    with pytest.raises(ValueError, match="same source member"):
        replace(member, donor_manifests=all_donors,
            source_manifests=(_bound(tmp_path, member.trajectory),)).verify(
                replay, shared_geometry_sha256="b" * 64)
