"""Exact recent-case clocks, matched member identities and history coverage."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import da_recent_run as runner
from tools.da_cycle_prepared import build_parser
from tools.da_member_leg import rain_history_handler


def case(tmp_path, *, smoke=False):
    for name in ("surface", "grid", "obs"):
        (tmp_path/name).write_bytes(b"fixture metadata")
    count = 1 if smoke else 3
    return {"engine_sha": runner.source_revision(), "source": "rap-native",
        "prepared_root": str(tmp_path/"prepared"), "authority_dir": str(tmp_path/"authority"),
        "physics_profile": "fixture-profile", "proof_sha256": "a"*64,
        "source_manifest_sha256": "b"*64, "prepared_content_sha256": "c"*64,
        "run_seconds": 7200 if smoke else 32400, "history_interval_seconds": 120,
        "model_start_utc": "2026-10-03T21:00:00Z",
        "forecast_fork_utc": "2026-10-03T22:00:00Z" if smoke else "2026-10-04T00:00:00Z",
        "slots": [{"obs": str(tmp_path/"obs"), "grid_wrfout": str(tmp_path/"grid"),
            "leg_seconds": 3600, "analysis_time": stamp} for stamp in
            ["2026-10-03T22:00:00Z", "2026-10-03T23:00:00Z", "2026-10-04T00:00:00Z"][:count]],
        "surface_obs": str(tmp_path/"surface"), "out": str(tmp_path/"runs"),
        "members": 4 if smoke else 32, "smoke_only": smoke,
        "free_forecast_seconds": 120 if smoke else 21600}


@pytest.mark.parametrize("smoke", [False, True])
def test_da_control_commands_parse_real_public_door_and_match_every_clock(tmp_path, smoke):
    document = case(tmp_path, smoke=smoke)
    commands = runner.plan(document, seed=20261004)
    da, control = [build_parser().parse_args(row["argv"][3:]) for row in commands]
    assert da.leg_durations_seconds == control.leg_durations_seconds
    assert da.seed == control.seed == 20261004
    assert da.members == control.members == document["members"]
    assert da.rain_forecast_start_seconds == control.rain_forecast_start_seconds
    assert da.rain_history and control.rain_history
    assert da.forecast_members_per_card == control.forecast_members_per_card == "auto"
    assert da.packed_smoke == control.packed_smoke == smoke
    assert da.length_scale_km == control.length_scale_km == (12 if smoke else 40)
    assert da.thermo_length_scale_km == control.thermo_length_scale_km == (12 if smoke else 40)
    assert da.free_legs == 1
    # The deck states no positivity policy of its own: both arms take the
    # door's default.  It used to name clip, which adds the mass of every
    # undershooting member at every analysis (gpuwm.da.positivity).
    assert "--positivity-policy" not in commands[0]["argv"]
    assert da.positivity_policy == control.positivity_policy == "mean-preserving"
    assert not control.obs and control.surface_obs is None and control.resume_ensemble is None
    assert da.surface_obs == Path(document["surface_obs"])


def test_new_seed_outputs_are_isolated(tmp_path):
    document = case(tmp_path)
    first = runner.plan(document, seed=1)
    second = runner.plan(document, seed=2)
    assert [row["argv"][-1] for row in first] != [row["argv"][-1] for row in second]


@pytest.mark.parametrize("mutation, message", [
    ({"engine_sha": "0"*40}, "source revisions"),
    ({"free_forecast_seconds": 3600}, "six hours"),
    ({"history_interval_seconds": 300}, "120-second"),
    ({"run_seconds": 10800}, "forcing horizon"),
])
def test_recent_plan_refuses_unmatched_or_incomplete_forecast(tmp_path, mutation, message):
    document = case(tmp_path)
    document.update(mutation)
    with pytest.raises(ValueError, match=message):
        runner.plan(document, seed=1)


def test_scheduled_rain_history_uses_live_owner_and_skips_duplicate_endpoints(monkeypatch, tmp_path):
    from tools import da_cycle_prepared
    written = []
    monkeypatch.setattr(da_cycle_prepared, "_write_rain_composite", lambda path, **kw:
        written.append((path, kw)))
    cfg = SimpleNamespace(grid_id=1, run=object())
    state = SimpleNamespace(qv=object(), physics=SimpleNamespace(mp_physics=28, refl_10cm=None))
    node = SimpleNamespace(cfg=cfg, state=state, grid=object(),
        clock=SimpleNamespace(elapsed_seconds=60, tick_den=1))
    context = SimpleNamespace(args=SimpleNamespace(rain_history=True, rain_forecast_start_seconds=60),
        t_end=180, out=tmp_path, inputs=SimpleNamespace(experiment=object()))
    driver = object()
    callback = rain_history_handler(context=context, name=2, drivers={1: driver})
    for seconds in (60, 120, 180):
        state.physics.refl_10cm = object()
        node.clock.elapsed_seconds = seconds
        callback(None, node, seconds)
        assert state.physics.refl_10cm is None
    assert len(written) == 2
    assert written[0][1]["elapsed_seconds"] == 120
    assert written[0][1]["driver"] is driver
    assert written[0][1]["state"] is node.state
    assert "history000000000120t1_2" in str(written[0][0])
    assert written[1][1]["elapsed_seconds"] == 180


def test_default_worker_does_not_add_output_callbacks():
    context = SimpleNamespace(args=SimpleNamespace())
    assert rain_history_handler(context=context, name=0, drivers={}) is None
