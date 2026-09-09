"""Installed TUI discovery, literal process arguments, and release membership."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest

from gpuwm import bridge_assets, bridges, tui_cli


def _binary(path: Path, *, current=True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bridges.BRIDGE_ABI_MARKERS[tui_cli.TUI_NAME]
                     if current else b"obsolete terminal")
    return path


def test_tui_ladder_covers_source_and_installed_locations(monkeypatch, tmp_path):
    package = tmp_path / "an installed package" / "gpuwm"
    monkeypatch.setattr(tui_cli, "__file__", str(package / "tui_cli.py"))
    staged = tmp_path / "user tools"
    monkeypatch.setattr(bridges, "packaged_bridge_dir", lambda: package / "libexec" / "bridges")
    monkeypatch.setattr(bridges, "default_bridge_dir", lambda: staged)
    override = tmp_path / "Drew's tools" / "arwen-tui"
    monkeypatch.setenv(tui_cli.TUI_ENV, str(override))
    name = bridges.executable_name(tui_cli.TUI_NAME)
    assert tui_cli.tui_candidates() == (
        override,
        package.parent / "tools" / "arwen-tui" / "target" / "release" / name,
        package.parent / "tools" / "arwen-tui" / "target" / "debug" / name,
        package.parent / "libexec" / "bridges" / name,
        package / "libexec" / "bridges" / name,
        staged / name,
    )


def test_missing_override_never_falls_through_to_another_binary(monkeypatch, tmp_path):
    missing = tmp_path / "missing executable"
    fallback = _binary(tmp_path / "fallback")
    monkeypatch.setenv(tui_cli.TUI_ENV, str(missing))
    monkeypatch.setattr(tui_cli, "tui_candidates", lambda: (missing, fallback))
    with pytest.raises(FileNotFoundError, match="GPUWM_TUI_BIN names a missing file"):
        tui_cli.require_tui()


def test_tui_resolution_keeps_the_native_pin_guard(monkeypatch, tmp_path):
    binary = _binary(tmp_path / "packaged" / "arwen-tui")
    monkeypatch.delenv(tui_cli.TUI_ENV, raising=False)
    monkeypatch.setattr(tui_cli, "tui_candidates", lambda: (binary,))
    accepted = []

    def guard(path):
        accepted.append(path)
        return path

    monkeypatch.setattr(bridges, "accept_resolved", guard)
    assert tui_cli.require_tui() == binary.resolve()
    assert accepted == [binary.resolve()]

    def refuse(path):
        raise bridges.StaleBridgeError("the release pins require different bytes")

    monkeypatch.setattr(bridges, "accept_resolved", refuse)
    with pytest.raises(bridges.StaleBridgeError, match="release pins"):
        tui_cli.require_tui()


def test_obsolete_terminal_contract_refuses_without_fallback(monkeypatch, tmp_path):
    old = _binary(tmp_path / "old", current=False)
    current = _binary(tmp_path / "current")
    monkeypatch.delenv(tui_cli.TUI_ENV, raising=False)
    monkeypatch.setattr(tui_cli, "tui_candidates", lambda: (old, current))
    monkeypatch.setattr(bridges, "accept_resolved", lambda path: path)
    with pytest.raises(FileNotFoundError, match="predates this release's arwen-tui contract"):
        tui_cli.require_tui()


def test_public_tui_arguments_survive_a_real_subprocess(monkeypatch, tmp_path):
    from gpuwm.cli import build_parser

    # The probe records the actual argv reconstructed by the OS, including
    # Windows' command-line quoting. Only the executable is substituted;
    # every argument composed by the production launcher crosses a process.
    probe = tmp_path / "Drew's argv probe.py"
    received = tmp_path / "received arguments.json"
    probe.write_text(
        "import json, pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]), encoding='utf-8')\n"
        "raise SystemExit(7)\n", encoding="utf-8")
    binary = tmp_path / "installed tools" / "Drew's arwen-tui"
    monkeypatch.setattr(tui_cli, "require_tui", lambda: binary)
    run = subprocess.run

    def run_probe(command, **kwargs):
        assert kwargs["check"] is False and kwargs["shell"] is False
        assert kwargs["env"]["PYTHONSAFEPATH"] == "1"
        return run([sys.executable, str(probe), str(received), *command], **kwargs)

    monkeypatch.setattr(tui_cli.subprocess, "run", run_probe)
    values = {
        "--config": "Drew's configs/日本語 case.toml",
        "--output": "literal $(echo untouched) output",
        "--prepared": "prepared inputs/one;two",
        "--geog-root": "geography's folder/space here",
        "--snapshot": "Drew's snapshots/terminal preview.html",
        "--snapshot-width": "80",
        "--snapshot-height": "24",
        "--snapshot-screen": "mode:supercell",
    }
    tokens = [item for pair in values.items() for item in pair]
    args = build_parser().parse_args(["tui", *tokens])
    assert args.func(args) == 7
    actual = json.loads(received.read_text(encoding="utf-8"))
    expected = [str(binary), "--python", sys.executable]
    for flag, value in values.items():
        expected.extend((flag, str(Path(value))))
    assert actual == expected


def test_tui_uses_current_interpreter_despite_ambient_tui_python(monkeypatch):
    monkeypatch.setenv("GPUWM_TUI_PYTHON", "another environment/python")
    monkeypatch.setattr(tui_cli, "require_tui", lambda: Path("terminal"))
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["env"]["PYTHONSAFEPATH"] == "1"
        assert kwargs["env"]["GPUWM_TUI_PYTHON"] == "another environment/python"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tui_cli.subprocess, "run", run)
    assert tui_cli.tui_main(SimpleNamespace()) == 0
    assert commands == [["terminal", "--python", sys.executable]]


@pytest.mark.parametrize("error", [
    FileNotFoundError("GPUWM_TUI_BIN names a missing file"),
    OSError("executable could not start"),
    bridges.StaleBridgeError("release pins require different bytes"),
])
def test_tui_operational_refusals_are_named_without_tracebacks(monkeypatch, capsys, error):
    def fail():
        raise error

    monkeypatch.setattr(tui_cli, "require_tui", fail)
    assert tui_cli.tui_main(SimpleNamespace()) == 2
    assert capsys.readouterr().err == f"gpuwm tui: {error}\n"


def test_tui_is_a_stamped_member_of_both_native_release_platforms():
    from tools import stage_wheel_bridges

    artifact, = (entry for entry in bridge_assets.BUNDLED_ARTIFACTS
                 if entry.name == tui_cli.TUI_NAME)
    assert artifact.kind == "executable" and not artifact.vendored
    assert artifact.env_var == tui_cli.TUI_ENV
    assert artifact.crate == tui_cli.TUI_CRATE_RELATIVE
    for platform, filename in (("win-x86_64", "arwen-tui.exe"),
                               ("linux-x86_64", "arwen-tui")):
        assert bridge_assets.artifact_filename(artifact, platform) == filename
        assert stage_wheel_bridges.source_path(artifact, platform) == (
            stage_wheel_bridges._REPO_ROOT / artifact.crate / "target" / "release" / filename)


def test_terminal_and_python_release_versions_agree():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    terminal = tomllib.loads((root / tui_cli.TUI_CRATE_RELATIVE / "Cargo.toml").read_text(encoding="utf-8"))
    assert terminal["package"]["version"] == project["project"]["version"]


def test_release_packer_refuses_to_omit_the_tui(monkeypatch, tmp_path):
    from tools import build_bridge_bundle

    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        if artifact.name != tui_cli.TUI_NAME:
            (tmp_path / bridge_assets.artifact_filename(artifact, "linux-x86_64")).write_bytes(b"built")
    with pytest.raises(SystemExit, match="arwen-tui is in none of the search directories"):
        build_bridge_bundle.pack("v2.7.0", "linux-x86_64", [tmp_path], tmp_path / "out")
