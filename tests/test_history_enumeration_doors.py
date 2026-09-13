"""Run consumers find episode frames through the history writer's walker."""

import pytest

from gpuwm import go_cli, speedrun_cli


@pytest.mark.parametrize("door", ["go", "speedrun"])
def test_run_readers_include_episode_frames_and_exclude_sidecars(tmp_path, door):
    run = tmp_path / "run"
    root = run / "run" / "wrfout"
    names = [
        "wrfout_d01_2026-09-13_00_00_00",
        "d02/episode-001/wrfout_d02_2026-09-13_01_00_00",
        "d02/episode-002/wrfout_d02_2026-09-13_02_00_00",
    ]
    excluded = [
        "d02/episode-002/wrfout_d02_2026-09-13_03_00_00.tmp.123",
        "ready/wrfout_d02_2026-09-13_02_00_00.json",
        ".quarantine/wrfout_d02_2026-09-13_04_00_00",
    ]
    for name in names + excluded:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"published-frame-placeholder")
    (root / "wrfout_d03_directory").mkdir()

    if door == "go":
        actual = go_cli.wrfout_frames({"run": run / "run"})
    else:
        actual = speedrun_cli.run_artifacts(run)["wrfout_paths"]
    assert actual == [root / name for name in names]


@pytest.mark.parametrize("door", ["go", "speedrun"])
def test_run_readers_allow_no_history_directory(tmp_path, door):
    if door == "go":
        actual = go_cli.wrfout_frames({"run": tmp_path / "run"})
    else:
        actual = speedrun_cli.run_artifacts(tmp_path)["wrfout_paths"]
    assert actual == []
