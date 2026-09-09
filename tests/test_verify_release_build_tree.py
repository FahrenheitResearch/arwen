"""A release build tree is the tagged commit plus the pin step, and nothing else.

THE BREAKAGE THIS PREVENTS: the 2.7.0 candidate's SDK kit was generated from
a tree with 48 dirty files while the engine wheels beside it came from the
tagged commit plus pins, and nothing distinguished the two (XPORT-002,
PKGENG-003, RC2-003).  ``tools/verify_release_build_tree.py`` refuses the
first shape and admits the second, without touching the owner ruling that
keeps the COMMITTED pins document unpinned (tools/verify_source_bridge_pins
.py).  Every case here runs against a real temporary git repository.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from tools import verify_release_build_tree as gate

pytestmark = pytest.mark.skipif(shutil.which("git") is None,
                                reason="the gate asks git")

UNPINNED = {"schema": "gpuwm-bridge-pins-v1", "release": None,
            "platforms": {}, "note": "unpinned"}


def _pinned(release: str = "v2.7.0", platforms=gate.SUPPORTED_PLATFORMS) -> dict:
    return {"schema": "gpuwm-bridge-pins-v1", "release": release,
            "platforms": {name: {"bundle": {"filename": f"{name}.zip",
                                            "bytes": 1, "sha256": "0" * 64}}
                          for name in platforms},
            "note": "pinned"}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A committed tree shaped like the release: pyproject 2.7.0, unpinned pins."""

    root = tmp_path / "engine"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "gpuwm"\nversion = "2.7.0"\n', encoding="utf-8")
    pins = root / gate.PINS_RELATIVE
    pins.parent.mkdir(parents=True)
    pins.write_text(json.dumps(UNPINNED, indent=2) + "\n", encoding="utf-8")
    (root / "gpuwm" / "__init__.py").write_text("", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@localhost",
         "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@localhost",
         "commit", "-q", "-m", "tag shape")
    return root


def _pin_working_copy(repo: Path, document: dict | None = None) -> None:
    (repo / gate.PINS_RELATIVE).write_text(
        json.dumps(document or _pinned(), indent=2) + "\n", encoding="utf-8")


def test_the_release_shape_passes(repo: Path) -> None:
    _pin_working_copy(repo)
    receipt = gate.verify_build_tree(repo)
    assert receipt["release_version"] is True
    assert receipt["committed_pins"] == "unpinned"
    assert receipt["working_pins_release"] == "v2.7.0"
    assert receipt["working_pins_platforms"] == sorted(gate.SUPPORTED_PLATFORMS)
    assert "clean" in receipt["verdict"]


def test_an_unpinned_working_copy_on_a_release_version_refuses(repo: Path) -> None:
    with pytest.raises(gate.ReleaseBuildTreeError, match="release: null"):
        gate.verify_build_tree(repo)


def test_a_dirty_file_beside_the_pins_refuses_and_is_named(repo: Path) -> None:
    _pin_working_copy(repo)
    (repo / "gpuwm" / "__init__.py").write_text("# edited\n", encoding="utf-8")
    with pytest.raises(gate.ReleaseBuildTreeError,
                       match="beyond the pin step") as info:
        gate.verify_build_tree(repo)
    assert "gpuwm/__init__.py" in str(info.value)
    assert "48 dirty files" in str(info.value)


def test_an_untracked_file_under_a_packaged_root_refuses(repo: Path) -> None:
    _pin_working_copy(repo)
    (repo / "gpuwm" / "stray.json").write_text("{}", encoding="utf-8")
    with pytest.raises(gate.ReleaseBuildTreeError, match="untracked, under a packaged root"):
        gate.verify_build_tree(repo)


def test_an_untracked_file_outside_the_packaged_roots_is_ignored(repo: Path) -> None:
    _pin_working_copy(repo)
    (repo / "build.log").write_text("noise\n", encoding="utf-8")
    assert "clean" in gate.verify_build_tree(repo)["verdict"]


def test_committed_pinned_pins_refuse_by_the_owner_ruling(repo: Path) -> None:
    _pin_working_copy(repo)
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@localhost",
         "commit", "-q", "-am", "pins committed by mistake")
    with pytest.raises(gate.ReleaseBuildTreeError,
                       match="HEAD's .* is PINNED.*owner ruling 2026-08-03") as info:
        gate.verify_build_tree(repo)
    assert "verify_source_bridge_pins" in str(info.value)


def test_pins_for_a_different_release_refuse(repo: Path) -> None:
    _pin_working_copy(repo, _pinned(release="v2.6.5"))
    with pytest.raises(gate.ReleaseBuildTreeError, match="v2.6.5.*2.7.0"):
        gate.verify_build_tree(repo)


def test_a_missing_platform_refuses(repo: Path) -> None:
    _pin_working_copy(repo, _pinned(platforms=("win-x86_64",)))
    with pytest.raises(gate.ReleaseBuildTreeError, match="linux-x86_64"):
        gate.verify_build_tree(repo)


@pytest.mark.parametrize("version", ["2.7.0rc1", "2.7.0.dev3", "2.7.0+local"])
def test_a_pre_release_tree_may_stay_unpinned(repo: Path, version: str) -> None:
    receipt = gate.verify_build_tree(repo, version=version)
    assert receipt["release_version"] is False
    assert "unpinned (allowed)" in receipt["verdict"]


def test_the_platform_list_matches_the_engine() -> None:
    from gpuwm import bridge_assets

    assert set(gate.SUPPORTED_PLATFORMS) == set(bridge_assets.SUPPORTED_PLATFORMS)


def test_the_cli_exits_one_and_names_the_refusal(repo: Path, capsys) -> None:
    assert gate.main(["--repo", str(repo)]) == 1
    assert "REFUSED" in capsys.readouterr().err
    _pin_working_copy(repo)
    assert gate.main(["--repo", str(repo)]) == 0
    assert json.loads(capsys.readouterr().out)["release_version"] is True
