"""Installed capability and input admission must not promise ignored output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from gpuwm import simulated_radar as radar
from gpuwm import simulated_radar_capabilities as capabilities
from gpuwm import rustwx


def parser():
    result = argparse.ArgumentParser()
    radar.register_cli(result.add_subparsers(dest="command"))
    return result


def test_missing_native_artifact_does_not_advertise_runnable_output(monkeypatch):
    monkeypatch.setattr(rustwx, "simulated_radar_binary", lambda: None)
    result = capabilities.describe()
    assert result["native"]["status"] == "missing"
    assert result["supports"]["history_replay"] is False
    assert result["supports"]["native_columns_adapter"] is False
    assert all(route["available"] is False for route in result["supports"]["live_routes"].values())


def test_probe_is_read_only_and_distinguishes_old_canonical_contract(monkeypatch, tmp_path):
    from gpuwm import bridges

    def resolve():
        assert bridges._INSPECTION_ONLY
        return tmp_path / "rw_simradar"

    seen = []
    monkeypatch.setattr(rustwx, "simulated_radar_binary", resolve)
    monkeypatch.setattr(rustwx, "renderer_env", lambda: {"CUDA_VISIBLE_DEVICES": ""})

    def run(argv, **kwargs):
        seen.append(argv[1])
        assert kwargs["timeout"] == 10
        output = rustwx.SIMULATED_RADAR_ABI if argv[1] == "--abi" else "native-atmosphere.columns/v0"
        return SimpleNamespace(returncode=0, stdout=output)

    monkeypatch.setattr(capabilities.subprocess, "run", run)
    result = capabilities.describe()
    assert seen == ["--abi", "--canonical-abi", "--capabilities"]
    assert result["supports"]["history_replay"] is True
    assert result["supports"]["native_columns_adapter"] is False
    assert result["native"]["canonical_columns"]["status"] == "incompatible"
    assert result["supports"]["resource_estimate"] is False
    assert result["supports"]["named_input_refusals"] is False


def test_native_features_are_advertised_only_after_the_installed_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(rustwx, "simulated_radar_binary", lambda: tmp_path / "rw_simradar")
    monkeypatch.setattr(rustwx, "renderer_env", lambda: {})
    response = {
        "schema": "rw-simradar.capabilities/v1", "request_abi": rustwx.SIMULATED_RADAR_ABI,
        "resource_estimate_schema": "simulated-radar.resources/v1", "input_validation": "full-columns/v1"}
    outputs = {"--abi": rustwx.SIMULATED_RADAR_ABI, "--canonical-abi": rustwx.CANONICAL_RADAR_ABI,
               "--capabilities": json.dumps(response)}
    monkeypatch.setattr(capabilities.subprocess, "run", lambda argv, **kw:
                        SimpleNamespace(returncode=0, stdout=outputs[argv[1]]))
    result = capabilities.describe()
    assert result["supports"]["resource_estimate"] is True
    assert result["supports"]["named_input_refusals"] is True


def test_native_resolution_failure_remains_machine_readable(monkeypatch):
    def missing():
        raise FileNotFoundError("explicit binary path does not exist")

    monkeypatch.setattr(rustwx, "simulated_radar_binary", missing)
    result = capabilities.describe()
    assert result["native"]["status"] == "resolution_failed"
    assert result["supports"]["history_replay"] is False


def test_probe_timeout_never_becomes_support(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("rw_simradar", 10)

    monkeypatch.setattr(capabilities.subprocess, "run", timeout)
    assert capabilities._probe(Path("rw_simradar"), "--abi", "contract", {})["available"] is False


def test_companion_module_and_version_are_not_live_route_proof(monkeypatch):
    monkeypatch.setattr(capabilities, "native_capabilities", lambda: {
        "available": True, "canonical_columns": {"available": True}})
    monkeypatch.setattr(capabilities, "_module_present", lambda _: True)
    monkeypatch.setattr(capabilities, "_version", lambda _: "0.3.3")
    result = capabilities.describe()
    for name in ("hex", "global"):
        route = result["supports"]["live_routes"][name]
        assert route["available"] is None
        assert route["status"] == "requires_companion_admission"
        assert not route["version_is_capability_proof"]
        assert len(route["qualified_source_commit"]) == 40
        assert route["requirements"]
    assert result["supports"]["display_only_2d"] is False
    assert result["inputs"]["wrf"]["missing_fields_refusal"] == "radar_input_missing_columns"


def test_native_columns_cli_converts_each_source_without_filename_collision(monkeypatch, tmp_path, capsys):
    sources = []
    for folder in ("first", "second"):
        path = tmp_path / folder / "atmosphere.nc"
        path.parent.mkdir()
        path.write_bytes(b"field transport passed unchanged to native")
        sources.append(path)
    conversions = []
    scans = []

    def convert(path, *, outdir):
        conversions.append((path, outdir))
        return outdir / "wrfout_d01_atmosphere.nc"

    monkeypatch.setattr(rustwx, "canonical_radar_scene", convert)
    monkeypatch.setattr(rustwx, "simulate_radar", lambda paths, **kw: scans.append((paths, kw)) or {})
    args = parser().parse_args(["simulated-radar", *map(str, sources), "--input-kind", "native-columns",
                                "--outdir", str(tmp_path / "output")])
    assert args.func(args) == 0
    assert [source for source, _ in conversions] == sources
    assert conversions[0][1] != conversions[1][1]
    assert scans[0][0] == [folder / "wrfout_d01_atmosphere.nc" for _, folder in conversions]
    assert json.loads(capsys.readouterr().out) == {}


def test_cli_geometry_estimate_needs_no_history_and_generates_no_radar(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(rustwx, "estimate_simulated_radar", lambda paths, **kw: seen.append((paths, kw)) or
                        {"schema": "simulated-radar.resources/v1"})
    args = parser().parse_args(["simulated-radar", "--estimate", "--sites", "KTLX"])
    assert args.func(args) == 0
    assert seen[0][0] == []
    assert seen[0][1]["config"]["sites"] == ["KTLX"]
    assert json.loads(capsys.readouterr().out)["schema"] == "simulated-radar.resources/v1"


def test_estimate_does_not_silently_convert_or_ignore_raw_native_columns(tmp_path):
    source = tmp_path / "source.nc"
    source.write_bytes(b"transport")
    args = parser().parse_args(["simulated-radar", str(source), "--input-kind", "native-columns", "--estimate"])
    with pytest.raises(ValueError, match="full canonical scene headers"):
        args.func(args)


def test_estimate_bridge_passes_paths_only_and_checks_resource_schema(monkeypatch, tmp_path):
    monkeypatch.setattr(rustwx, "simulated_radar_binary", lambda: tmp_path / "rw_simradar")
    monkeypatch.setattr(rustwx, "renderer_env", lambda: {})
    requests = []

    def run(argv, **kwargs):
        if argv[1] == "--abi":
            # The request contract is admitted before any estimate is asked.
            return SimpleNamespace(returncode=0, stdout=rustwx.SIMULATED_RADAR_ABI + "\n")
        assert argv[1] == "--estimate"
        requests.append(json.loads(Path(argv[2]).read_text()))
        return SimpleNamespace(returncode=0, stdout='{"schema":"simulated-radar.resources/v1"}')

    monkeypatch.setattr(rustwx.subprocess, "run", run)
    output = tmp_path / "output"
    result = rustwx.estimate_simulated_radar([], outdir=output, config={"enabled": True})
    assert result["schema"] == "simulated-radar.resources/v1"
    assert requests == [{"schema": radar.REQUEST_SCHEMA, "history_paths": [],
                         "outdir": str(output.resolve()), "config": {"enabled": True}}]
    assert not output.exists()
    monkeypatch.setattr(rustwx.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout='{}'))
    with pytest.raises(rustwx.SimulatedRadarRefusal, match="request contract"):
        rustwx.estimate_simulated_radar([], outdir=output, config={})
    with pytest.raises(RuntimeError, match="resource estimate contract"):
        rustwx.estimate_simulated_radar([], outdir=output, config={},
                                        binary=tmp_path / "rw_simradar")
