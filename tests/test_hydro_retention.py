"""Older starts must release private payloads without unlinking shared data or owners."""
import json
import os
from pathlib import Path

from gpuwm.hydro import retention


def start(root, index, *, managed=True):
    p = root / "provider" / f"{index:016x}"
    p.mkdir(parents=True)
    (p / "door-prepared.json").write_text("{}")
    (p / "door-request.json").write_text(json.dumps({"footprint": {
        "west": 1, "east": 2, "south": 3, "north": 4, "time_origin": str(index)}}))
    (p / "hydro-door.toml").write_text("enabled=true")
    os.utime(p / "hydro-door.toml", ns=(index, index))
    (p / "water.bin").write_bytes(bytes(8192))
    if managed:
        (p / "retention-managed.json").write_text("{}")
    return p


def test_last_three_live_checkpoint_and_prepared_root_survive(tmp_path):
    root = tmp_path / "cache"
    starts = [start(root, i) for i in range(8)]
    static = root / "provider/static-digest"
    static.mkdir()
    (static / "fabric").write_bytes(bytes(16384))
    os.link(static / "fabric", starts[0] / "fabric")
    retention.reference(starts[1] / "hydro-door.toml", root=root)
    owners = [tmp_path / "checkpoint.npz", tmp_path / "prepared"]
    for owner, p in zip(owners, starts[2:4]):
        owner.touch()
        retention.reference(p / "hydro-door.toml", owner=owner, root=root)
    for name in ("tile-cache", "forcing-day-cache"):
        (root / name).mkdir()
    expected = sum(p.stat().st_blocks * 512 for p in starts[0].iterdir()
                   if p.name != "fabric") + sum(p.stat().st_blocks * 512 for p in starts[4].iterdir())
    report = retention.reclaim(root)
    assert set(report["removed"]) == {str(starts[0]), str(starts[4])}
    assert report["reclaimed_bytes"] == expected
    assert (static / "fabric").read_bytes() == bytes(16384)
    assert all(p.exists() for p in starts[1:4] + starts[5:])
    assert (root / "tile-cache").is_dir() and (root / "forcing-day-cache").is_dir()
    for owner in owners:
        owner.unlink()
    assert set(retention.reclaim(root)["removed"]) == {str(starts[2]), str(starts[3])}


def test_unknown_legacy_and_symlinks_cannot_be_deleted(tmp_path):
    starts = [start(tmp_path, i, managed=i != 0) for i in range(6)]
    (starts[1] / "outside").symlink_to(tmp_path / "outside")
    report = retention.reclaim(tmp_path)
    assert report["removed"] == [str(starts[2])]
    assert starts[0].exists() and starts[1].exists()
    (tmp_path / ".references/broken.json").parent.mkdir(exist_ok=True)
    (tmp_path / ".references/broken.json").write_text("broken")
    assert "blocked" in retention.reclaim(tmp_path)


def test_plain_go_forecast_pins_its_prepared_root(tmp_path, monkeypatch):
    from gpuwm import go_cli
    root = tmp_path / "cache"
    starts = [start(root, i) for i in range(5)]
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    monkeypatch.setenv("GPUWM_HYDRO_FABRIC_CACHE", str(root))
    monkeypatch.setenv("GPUWM_NEXTGEN_SETTINGS", str(starts[0] / "hydro-door.toml"))

    class ReachedForecast(Exception):
        pass

    def stop(_plan):
        raise ReachedForecast()

    monkeypatch.setattr(go_cli, "_join_kernel_prewarm", stop)
    import pytest
    with pytest.raises(ReachedForecast):
        go_cli._run_forecast({"prepared": prepared}, {}, explain=False)
    assert retention.reclaim(root)["removed"] == [str(starts[1])]
    assert starts[0].is_dir()
    prepared.rmdir()
    assert retention.reclaim(root)["removed"] == [str(starts[0])]
