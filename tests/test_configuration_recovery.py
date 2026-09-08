"""All final creators retain the same exact, non-admitted recovery draft."""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from gpuwm import case_catalog, configuration_recovery as recovery, research_workspaces


def _create(producer, output):
    if producer == "catalog":
        from datetime import datetime, timezone
        return case_catalog.create_case(
            case_catalog.SCHEMA_PATH.with_name("example.json"), "synthetic-profile-example",
            out=output, tier="lower", vram_gib=32,
            native_overrides={"shared": {"mp_physics": 8}},
            now=datetime(2026, 9, 2, tzinfo=timezone.utc))
    parser = argparse.ArgumentParser()
    research_workspaces.register_cli(parser.add_subparsers(required=True))
    args = parser.parse_args(["research", "create", "scenario-convection.gentle",
        "--point=35.3,-97.5", "--cycle=2026-09-05T18", "--hardware-class=8",
        "--vram-gib=8", f"--out={output}"])
    return research_workspaces.create_workspace(args)


@pytest.mark.parametrize("producer", ["catalog", "research"])
def test_final_creators_retain_overrides_not_their_domain_scaffold(tmp_path, monkeypatch, producer):
    directory = tmp_path / "job" / "configuration-recovery"
    monkeypatch.setenv(recovery.RECOVERY_DIR_ENV, str(directory))
    final = []
    def refuse(text, **kwargs):
        final.append(tomllib.loads(text))
        raise recovery.MemoryAdmissionError("final candidate exceeds budget",
            peak_envelope_bytes=12 * 2**30, budget_bytes=6 * 2**30)
    monkeypatch.setattr(research_workspaces, "_admission", refuse)
    output = tmp_path / "requested.toml"
    with pytest.raises(recovery.MemoryAdmissionError) as failed:
        _create(producer, output)
    assert not output.exists()
    assert not output.with_suffix(".namelist.wps").exists()
    assert not list(tmp_path.glob(".arwen-case-*")) and not list(tmp_path.glob(".arwen-research-*"))
    error = recovery.error_document(failed.value)
    assert error["schema"] == "arwen.configuration-error.v1" and error["kind"] == "memory"
    assert error["created"] is False and "recovery_error" not in error
    receipt = json.loads((directory / "recovery.json").read_text())
    assert error["recovery"] == receipt
    assert receipt["schema"] == "arwen.configuration-recovery.v1"
    assert receipt["status"] == "memory-refused" and not receipt["forecast_started"]
    assert receipt["requested_output"] == str(output)
    assert (directory / "draft.namelist.wps").is_file()
    draft = directory / "draft.toml"
    assert hashlib.sha256(draft.read_bytes()).hexdigest() == receipt["config_sha256"]
    saved = tomllib.loads(draft.read_text())
    for key in ("shared", "domain", "experiment", "projection"):
        assert saved[key] == final[0][key]
    if producer == "catalog":
        assert saved["shared"]["mp_physics"] == 8
    else:
        assert saved["perturbation"] == final[0]["perturbation"]
        assert saved["perturbation"]["bubbles"][0]["amplitude_k"] == 1
    for row in receipt["files"]:
        path = Path(row["path"])
        assert path.parent == directory and path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


@pytest.mark.parametrize("producer", ["catalog", "research"])
def test_non_memory_final_failures_do_not_create_recovery(tmp_path, monkeypatch, producer):
    directory = tmp_path / "configuration-recovery"
    monkeypatch.setenv(recovery.RECOVERY_DIR_ENV, str(directory))
    def refuse(*args, **kwargs):
        raise ValueError("source or scientific configuration refusal")
    monkeypatch.setattr(research_workspaces, "_admission", refuse)
    with pytest.raises(ValueError, match="scientific configuration"):
        _create(producer, tmp_path / "requested.toml")
    assert not directory.exists() and not list(tmp_path.iterdir())


def test_no_recovery_environment_performs_no_file_access(tmp_path, monkeypatch):
    monkeypatch.delenv(recovery.RECOVERY_DIR_ENV, raising=False)
    error = recovery.MemoryAdmissionError("memory refusal")
    recovery.retain_final_candidate(error, text="not parsed", requested_path=tmp_path / "out.toml",
                                     stage=tmp_path / "does-not-exist")
    assert "recovery" not in recovery.error_document(error)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("nested", [False, True])
def test_misconfigured_recovery_cannot_create_the_requested_output(tmp_path, monkeypatch, nested):
    output = tmp_path / "requested.toml"
    directory = output / "recovery" if nested else output
    monkeypatch.setenv(recovery.RECOVERY_DIR_ENV, str(directory))
    error = recovery.MemoryAdmissionError("memory refusal")
    recovery.retain_final_candidate(error, text="not parsed", requested_path=output,
                                     stage=tmp_path / "does-not-exist")
    assert error.recovery is None
    assert "originally requested output" in error.recovery_error
    assert not output.exists()


def test_recovery_rebases_only_owned_paths_and_preserves_existing_directory(tmp_path, monkeypatch):
    from gpuwm.toml_document import emit_experiment_toml
    source = tmp_path / "original"
    source.mkdir()
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "wanted.toml").write_text("wrong scaffold")
    (stage / "wanted.namelist.wps").write_text("&share\n max_dom=1,\n/\n&geogrid\n geog_data_path='../geog',\n/\n")
    (stage / "wanted.Vtable").write_bytes(b"original table")
    raw = {"experiment": {"name": "final"}, "shared": {"mp_physics": 8},
           "case_data": {"forcing": ["forcing/*.grib"], "vtable": "wanted.Vtable",
                         "wps_namelist": "wanted.namelist.wps", "geog_root": "../geog"}}
    directory = tmp_path / "job" / "configuration-recovery"
    monkeypatch.setenv(recovery.RECOVERY_DIR_ENV, str(directory))
    error = recovery.MemoryAdmissionError("memory refusal")
    recovery.retain_final_candidate(error, text=emit_experiment_toml(raw),
                                     requested_path=source / "wanted.toml", stage=stage)
    assert error.recovery is not None and error.recovery_error is None
    saved = tomllib.loads((directory / "draft.toml").read_text())
    assert saved["shared"] == raw["shared"]
    assert saved["case_data"]["forcing"] == [str((source / "forcing/*.grib").resolve())]
    assert saved["case_data"]["geog_root"] == str((source / "../geog").resolve())
    assert saved["case_data"]["vtable"] == str(directory / "draft.Vtable")
    assert saved["case_data"]["wps_namelist"] == str(directory / "draft.namelist.wps")
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    again = recovery.MemoryAdmissionError("memory refusal")
    recovery.retain_final_candidate(again, text=emit_experiment_toml(raw),
                                     requested_path=source / "wanted.toml", stage=stage)
    assert again.recovery is None and "preserves the existing path" in again.recovery_error
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}


def test_common_cli_and_standalone_catalog_emit_the_shared_memory_error(monkeypatch, capsys):
    from gpuwm import cli
    def refuse(*args, **kwargs):
        raise recovery.MemoryAdmissionError("typed memory refusal", budget_bytes=123)
    monkeypatch.setattr(cli, "_dispatch", refuse)
    assert cli.main(["version"]) == 2
    first = json.loads(capsys.readouterr().out)
    monkeypatch.setattr(case_catalog, "_catalog_result", refuse)
    assert case_catalog.main(["list", "--json"]) == 2
    second = json.loads(capsys.readouterr().out)
    assert first == second == {"schema": "arwen.configuration-error.v1", "kind": "memory",
        "error": "typed memory refusal", "created": False, "memory": {"budget_bytes": 123}}
