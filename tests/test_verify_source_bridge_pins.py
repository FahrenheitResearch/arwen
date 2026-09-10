"""The tagged tree stays unpinned while built distributions gain real pins."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tools import verify_source_bridge_pins as source_pins

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "bridge-pins.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_the_committed_source_tree_is_explicitly_unpinned() -> None:
    payload = source_pins.verify_source_pins()
    assert payload["release"] is None
    assert payload["platforms"] == {}


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"release": "ops-public-candidate-private"}, "release=null"),
        ({"platforms": {"linux-x86_64": {}}}, "platforms={}"),
        ({"schema": "something-else"}, "schema must be"),
        ({"candidate": "private"}, "keys drifted"),
        ({"note": ""}, "note must be non-empty"),
    ),
)
def test_release_or_candidate_state_is_refused(
    tmp_path: Path, mutation: dict[str, object], message: str
) -> None:
    payload: dict[str, object] = {
        "schema": source_pins.PINS_SCHEMA,
        "release": None,
        "platforms": {},
        "note": "generated only during the release build",
    }
    payload.update(mutation)
    with pytest.raises(source_pins.SourceBridgePinsError, match=message):
        source_pins.verify_source_pins(_write(tmp_path, payload))


def test_non_object_or_malformed_json_is_refused(tmp_path: Path) -> None:
    with pytest.raises(source_pins.SourceBridgePinsError, match="JSON object"):
        source_pins.verify_source_pins(_write(tmp_path, []))
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(source_pins.SourceBridgePinsError, match="unreadable"):
        source_pins.verify_source_pins(broken)


def test_prepared_publisher_preserves_native_build_qualification_without_rebuilding():
    from tools import build_linux_release_bridges as linux_build
    import yaml
    document = yaml.load((REPO_ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert linux_build.IMAGE.startswith("quay.io/pypa/manylinux_2_28_x86_64@sha256:")
    assert linux_build.RUST_VERSION == "1.94.0"
    assert "tools/arwen-tui" in linux_build.WORKSPACES
    # Rebuilding a native bundle during retry invalidates every prepared hash.
    for name, job in document["jobs"].items():
        if name != "test":
            scripts = "\n".join(step.get("run", "") for step in job["steps"])
            assert "cargo build" not in scripts
            assert "docker run" not in scripts
            assert "python -m build" not in scripts
    controller = (REPO_ROOT / "tools/promote_prepared_release.py").read_text(encoding="utf-8")
    assert "tools/verify_source_bridge_pins.py" in controller
    assert "tools/build_bridge_bundle.py" in controller
    assert '"pin", "--release"' in controller
    assert "prepared native manifest differs from regenerated exact-byte pins" in controller
    assert "check_embedded_natives" in controller


def test_publish_workflow_has_two_publication_ingresses() -> None:
    import yaml
    text = (REPO_ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    document = yaml.load(text, Loader=yaml.BaseLoader)
    assert document["on"]["release"]["types"] == ["published"]
    inputs = document["on"]["workflow_dispatch"]["inputs"]
    assert inputs["release_tag"]["required"] == "true"
    assert inputs["publication_manifest_sha256"]["required"] == "false"
    for optional in ("stable_release_expected", "immutable_releases_enabled"):
        assert inputs[optional]["default"] == "false"
    assert "${{ inputs.release_tag || github.event.release.tag_name }}" in text
    assert "--ref \"$GITHUB_REF\" --commit \"$GITHUB_SHA\"" in text
    assert "environment: pypi" in text
    controller = (REPO_ROOT / "tools/promote_prepared_release.py").read_text(encoding="utf-8")
    assert 'release["prerelease"] == captured["prerelease"]' in controller
    assert '"draft": False, "prerelease": captured["prerelease"]' in controller
    assert 'captured["immutable_required"]' in controller
    assert 'release.get("immutable") is True' in controller
    assert "exactly one authenticated release must carry the selected tag" in controller
    for name, job in document["jobs"].items():
        if name != "test":
            assert job["if"] == "github.event_name != 'pull_request'"
    publish = document["jobs"]["publish"]
    assert publish["permissions"] == {"id-token": "write"}
    assert "assets" in publish["needs"]
    assert "GH_TOKEN" not in str(publish)
    assert "github.token" not in str(publish)
    assert not any(step.get("uses", "").startswith("actions/checkout@") for step in publish["steps"])
