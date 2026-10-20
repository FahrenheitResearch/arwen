"""Public source ladder and the receipt/wrfout label contract (no CUDA)."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from gpuwm.ensemble.wofs_recipe import (
    LadderRefusal, WofsOptions, plan_ladder, provenance_label, source_population,
)
from gpuwm.ensemble.request import EnsembleRequest
from gpuwm.ensemble.recipes import SourceTrajectory, build_recipe

START = datetime(2021, 12, 10, 18, tzinfo=timezone.utc)


def ladder(source, **options):
    return plan_ladder(source=source, cycle=START, start=START,
                       end=START + timedelta(hours=1), options=WofsOptions(**options))


def test_gefs_uses_own_members_before_previous_cycle():
    result = ladder("gefs")
    assert result.counts() == {"source-member": 31, "time-lagged": 5, "perturbation": 0}
    assert [m.source_member for m in result.members[:3]] == ["c00", "p01", "p02"]
    assert result.members[31].cycle == START - timedelta(hours=6)
    assert all(m.wind_amplitude == 0 for m in result.members)


@pytest.mark.parametrize("source,bounds", [
    ("era5", (0, 6)), ("hrrr", (0, 1)), ("gfs", (0, 3)), ("gefs", (0, 3)),
])
def test_short_forecast_keeps_its_end_and_acquires_enclosing_knots(source, bounds):
    end = START + timedelta(minutes=10)
    result = plan_ladder(source=source, cycle=START, start=START, end=end,
                         options=WofsOptions(members=2, provider="arco"))
    assert result.end == end
    member = result.members[0]
    trajectory = SourceTrajectory(member.source, member.cycle, member.source_member)
    assert trajectory.window(START, end, native_bracketing=True) == bounds
    with pytest.raises(ValueError, match="whole-hour"):
        trajectory.window(START, end)
    request = EnsembleRequest.from_mapping(result.ensemble_table())
    recipe = build_recipe(source=source, cycle=START, start=START, end=end,
                          count=2, kind="wofs", base_seed=0,
                          member_variants=request.member_variants)
    assert recipe.end == end
    assert recipe.acquisition_window(recipe.members[0].trajectory) == bounds


def test_hrrr_lags_then_fills_and_preserves_boundaries():
    result = ladder("hrrr", members=5, max_lag_hours=2)
    assert result.counts() == {"source-member": 1, "time-lagged": 2, "perturbation": 2}
    assert [m.lag_hours for m in result.members] == [0, 1, 2, 0, 1]
    table = result.ensemble_table()
    request = EnsembleRequest.from_mapping(table)
    recipe = build_recipe(source="hrrr", cycle=START, start=START,
        end=START + timedelta(hours=1), count=5, base_seed=0, kind="wofs",
        trajectories=tuple(SourceTrajectory(row["source"], datetime.fromisoformat(row["cycle"]).replace(tzinfo=timezone.utc), row.get("member"))
                           for row in request.trajectories), member_variants=request.member_variants)
    assert recipe.acquisition_window(recipe.members[2].trajectory) == (2, 3)
    assert recipe.members[3].trajectory == recipe.members[0].trajectory
    assert request.member_variants[3]["wind"] == 0.5


def test_keyless_era5_has_one_analysis_then_reproducible_fill(monkeypatch):
    monkeypatch.setattr("gpuwm.fetch.cds_credentials_present", lambda: False)
    first = ladder("era5", members=4)
    again = ladder("era5", members=4)
    assert first.describe() == again.describe()
    assert first.counts() == {"source-member": 1, "time-lagged": 0, "perturbation": 3}
    assert len({m.seed for m in first.members}) == 4
    assert {m.physics_arm for m in first.members[1:]} == {"thompson-ysu", "thompson-mynn", "thompson-myj"}
    assert all(0.9 <= m.surface["soil_moisture_scale"] <= 1.1 for m in first.members[1:])
    assert first.members[0].surface is None


def test_era5_cds_population_is_encoded_eda_and_arco_is_selectable(monkeypatch):
    monkeypatch.setattr("gpuwm.fetch.cds_credentials_present", lambda: True)
    assert [m.member for m in source_population("era5", START)] == list(map(str, range(10)))
    result = ladder("era5", members=12)
    assert result.counts() == {"source-member": 10, "time-lagged": 0, "perturbation": 2}
    assert len(source_population("era5", START, provider="arco")) == 1


def test_gfs_stays_gfs_and_knobs_remain_selectable():
    result = ladder("gfs", members=3, max_lag_hours=0, physics="off", surface=False)
    assert {m.source for m in result.members} == {"gfs"}
    assert all(m.physics == {} and m.surface is None for m in result.members)
    assert result.members[-1].wind_amplitude == 0.5
    with pytest.raises(LadderRefusal, match="zero spread"):
        ladder("gfs", members=2, max_lag_hours=0, physics="off", surface=False, wind_amplitude=0)


def test_canonical_label_is_identical_in_request_and_wrfout_metadata(monkeypatch):
    monkeypatch.setattr("gpuwm.fetch.cds_credentials_present", lambda: False)
    result = ladder("era5", members=4)
    request = EnsembleRequest.from_mapping(result.ensemble_table())
    restored = EnsembleRequest.from_mapping(request.receipt())
    for member, row in zip(result.members, restored.provenance):
        assert row["label"] == provenance_label(row) == member.provenance()["label"]
        decoded = json.loads(row["label"])
        assert decoded["schema"] == "gpuwm-member-provenance.v2"
        assert decoded["member"] == member.index
        assert "label" not in decoded
    assert json.loads(restored.provenance[-1]["label"])["perturbation"]["amplitude_m_s"] == 0.5


def test_checkpoint_fill_restores_parent_before_perturbation_and_reuses_only_matching_physics(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gpuwm.ensemble.checkpoint_fill import CheckpointFill
    from gpuwm.io import restart
    events, snapshots = [], {}
    monkeypatch.setattr(restart, "configuration_echo", lambda cfg: vars(cfg))
    def save(path, state, cfg):
        events.append("save")
        snapshots[path] = state.word
        path.write_bytes(b"parent")
    def restore(path, state, cfg):
        events.append("restore")
        state.word = snapshots[path]
        return SimpleNamespace(elapsed_seconds=0.)
    monkeypatch.setattr(restart, "write_restart", save)
    monkeypatch.setattr(restart, "restore_restart", restore)
    owner = CheckpointFill(tmp_path)
    provenance = {"source": "era5", "cycle": "2021-12-10T18", "lag_hours": 0}
    def perturb(*, state, **kwargs):
        events.append("perturb")
        assert state.word == 7
        state.word += 1
    for member, physics in ((0, 8), (1, 8), (2, 10)):
        state = SimpleNamespace(elapsed_seconds=0., word=7)
        cfg = SimpleNamespace(grid_id=1, mp_physics=physics)
        callback = owner.callback(provenance, member_id=member, previous=perturb)
        callback(state=state, cfg=cfg)
        assert state.word == 8
        assert restart.classify_state_attr("_ensemble_checkpoint_fill") == "infra"
    assert events == ["save", "restore", "perturb", "restore", "perturb", "save", "restore", "perturb"]
    assert owner.receipts[1][1]["parent_reused"] is True
    assert owner.receipts[2][1]["parent_checkpoint"] != owner.receipts[0][1]["parent_checkpoint"]


def test_checkpoint_fill_lets_a_moved_nest_keep_the_member_branch(tmp_path, monkeypatch):
    # WOOF 1.1.0 dry cut: a two-domain moving-nest hurricane ensemble stopped every fill member at the
    # first nest move (d02 rebuilt at 900 s) with "perturbation-fill parents must be checkpointed".
    from types import SimpleNamespace
    from gpuwm.ensemble.checkpoint_fill import CheckpointFill
    from gpuwm.io import restart
    events = []
    monkeypatch.setattr(restart, "configuration_echo", lambda cfg: vars(cfg))
    monkeypatch.setattr(restart, "write_restart", lambda path, state, cfg: path.write_bytes(b"parent"))
    def restore(path, state, cfg):
        events.append("restore")
        return SimpleNamespace(elapsed_seconds=0.)
    monkeypatch.setattr(restart, "restore_restart", restore)
    owner = CheckpointFill(tmp_path)
    provenance = {"source": "gfs", "cycle": "2022-09-28T00", "lag_hours": 0}
    callback = owner.callback(provenance, member_id=1, previous=lambda **kw: events.append("perturb"))
    callback(state=SimpleNamespace(elapsed_seconds=0., word=7), cfg=SimpleNamespace(grid_id=1, mp_physics=8))
    moved = SimpleNamespace(elapsed_seconds=900., word=9)
    callback(state=moved, cfg=SimpleNamespace(mp_physics=8))
    assert events == ["restore", "perturb", "perturb"] and moved.word == 9
    assert list(owner.receipts[1]) == [1]
    assert owner.reconstructions[1][0]["model_elapsed_seconds"] == 900. and not owner.reconstructions[1][0]["applied"]
    # A member never branched at t0 still refuses an advanced state.
    other = owner.callback(provenance, member_id=2, previous=None)
    with pytest.raises(ValueError, match="checkpointed at initialization"):
        other(state=SimpleNamespace(elapsed_seconds=900., word=9), cfg=SimpleNamespace(grid_id=1, mp_physics=8))


def test_provenance_rejects_a_receipt_label_that_does_not_match_the_member(monkeypatch):
    monkeypatch.setattr("gpuwm.fetch.cds_credentials_present", lambda: False)
    table = ladder("era5", members=2).ensemble_table()
    table["provenance"][1]["lag_hours"] = 3
    with pytest.raises(ValueError, match="canonical JSON label"):
        EnsembleRequest.from_mapping(table)


@pytest.mark.parametrize("options", [{"members": 0}, {"members": True}, {"wind_amplitude": float("nan")},
    {"wind_amplitude": -1}, {"max_lag_hours": -1}])
def test_invalid_ladder_controls_name_the_bad_input(options):
    with pytest.raises(LadderRefusal):
        WofsOptions(**options)
