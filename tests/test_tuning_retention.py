"""A scored day releases histories only after every map frame is retained."""
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from gpuwm import tuning, tuning_retention as retention


def test_retirement_requires_all_map_frames_before_deleting(tmp_path, monkeypatch):
    packet, runs, collection = (tmp_path / name for name in ("packet", "runs", "collection"))
    day = "2024-05-01"
    member = "setting-001"
    round_ = {"members": [{"id": member}], "domain_inventory": [{"id": "d01"}], "heldout_days": [day]}
    tuning.write_json(packet / "round.json", round_)
    tuning.seal_packet(packet)
    tuning.write_json(collection / "collection-receipt.json", {"days": [day]})
    monkeypatch.setattr(retention, "verify_collected_scores", lambda *args, **kwargs: {})
    # Only the module's own view of the platform: patching os.name itself
    # makes pathlib build PosixPath on a Windows host and fail there.
    monkeypatch.setattr(retention, "os", SimpleNamespace(name="posix"))
    directory = runs / day / member
    tuning.write_json(directory / "job.json", {"source": "test"})
    tuning.write_json(directory / "tuning-run.json", {"status": "succeeded", "job_sha256": tuning.digest(directory / "job.json")})
    frames = directory / "run"
    frames.mkdir()
    early = frames / f"wrfout_d01_{day}_21_00_00.nc"
    early.write_bytes(b"test fixture")
    with pytest.raises(ValueError, match="map frames"):
        retention.retire_day(packet, runs, collection, day)
    assert early.exists()
    later = frames / f"wrfout_d01_{day}_22_00_00.nc"
    later.write_bytes(b"test fixture map")
    result = retention.retire_day(packet, runs, collection, day)
    assert result["status"] == "complete"
    assert not early.exists() and later.exists()
    assert result["deleted"][0]["bytes"] == 12
