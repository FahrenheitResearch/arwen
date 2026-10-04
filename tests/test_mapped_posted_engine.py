"""An old complete-window mapper is refused before posted batch decoding."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import mapped_engine_bridge as bridge


def test_posted_mode_refuses_an_old_native_mapper_before_creating_output(monkeypatch, tmp_path):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "schema": bridge.CAPABILITIES_SCHEMA, "subcommands": {"compose": ["grib2"]}}))
    monkeypatch.setattr(bridge.subprocess, "run", run)
    output = tmp_path / "output"
    with pytest.raises(bridge.EngineUnavailable, match="lead-batch capability"):
        bridge.run_engine("compose", engine=Path("native-engine"), mapping="mapping.json",
                          files=("not-yet-present.grib2",), output=output, lead_batch=True)
    assert calls == [["native-engine", "capabilities"]]
    assert not output.exists()


def test_complete_window_keeps_using_an_ordinary_native_mapper(monkeypatch, tmp_path):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stderr="", stdout="")
    monkeypatch.setattr(bridge.subprocess, "run", run)
    bridge.run_engine("compose", engine=Path("native-engine"), mapping="mapping.json",
                      files=("source.grib2",), output=tmp_path / "output")
    assert len(calls) == 1 and calls[0][1] == "compose"
    assert "--lead-batch" not in calls[0]


def test_current_declared_native_mapper_receives_the_posted_mode(monkeypatch, tmp_path):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "schema": bridge.CAPABILITIES_SCHEMA,
            "features": {"lead_batch": "gpuwm-mapped-lead-batch-v1"}}) if command[1] == "capabilities" else "")
    monkeypatch.setattr(bridge.subprocess, "run", run)
    bridge.run_engine("compose", engine=Path("native-engine"), mapping="mapping.json",
                      files=("source.grib2",), output=tmp_path / "output", lead_batch=True)
    assert calls[0] == ["native-engine", "capabilities"]
    assert calls[1][-1] == "--lead-batch"
