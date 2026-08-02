"""The tagged tree stays unpinned while built distributions gain real pins."""

from __future__ import annotations

import json
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


def test_publish_workflow_has_one_draft_publication_ingress() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8"
    )
    trigger_block = text.split("\npermissions:", 1)[0]
    assert "workflow_dispatch:" in trigger_block
    assert "release_tag:" in trigger_block
    assert "immutable_releases_enabled:" in trigger_block
    assert "types: [published]" not in trigger_block
    assert "\n  release:\n" not in trigger_block
    assert "python tools/verify_source_bridge_pins.py" in text
    assert "tests/test_publish_workflow_state_machine.py" in text
    assert "public release version must be stable X.Y.Z" in text
    assert "existing draft release" in text
    assert "releases/${RELEASE_ID}" in text
    assert "releases?per_page=100" in text
    assert "--paginate --slurp" in text
    assert "releases/tags/" not in text
    assert 'releases/${RELEASE_ID}\")' not in text
    assert "- name: pin the toolchain\n        shell: bash" in text
    assert text.count("name: release-assets") == 4
    assert 'if [ "$state" = "starter" ]' in text
    assert "releases/assets/$asset_id" in text
    assert 'sha256:${local_sha[$name]}' in text
    assert "dist-upload/" in text
    assert "steps.pypi.outputs.upload_required == 'true'" in text
    assert "PyPI distribution differs from the proven artifact" in text
    assert "prove exact PyPI state before release promotion" in text
    assert "\n  authorize_pypi:\n" in text
    assert "needs: [cut, prepare, authorize_pypi]" in text
    publish_block = text.split("\n  publish:\n", 1)[1].split(
        "\n  # GitHub draft promotion", 1
    )[0]
    assert "id-token: write" in publish_block
    assert "contents:" not in publish_block
    assert "GH_TOKEN" not in publish_block
    assert "GH_REPO" not in publish_block
    assert "github.token" not in publish_block
    assert "releases?per_page" not in publish_block
    assert "release-assets" not in publish_block
    assert "/git/ref/tags/" in text
    assert "/git/tags/" in text
    assert "RELEASE_COMMIT" in text
    # Immutability is opt-in rather than a hard gate: it is a repository
    # setting with no API to set it, and every release through v1.4.0 shipped
    # without it, so refusing the cut over it blocked a working publication.
    # What must not quietly disappear is the enforcement itself, so all three
    # parts are pinned -- the read, the opt-in branch, and the hard failure
    # inside it.  Deleting any one of them turns a confirmed claim into a
    # comment.
    assert "after_immutable=$(jq -r '.immutable' <<<\"$after\")" in text
    assert 'if [ "$IMMUTABLE_RELEASES_CONFIRMED" = "true" ]; then' in text
    assert 'if [ "$after_immutable" != "true" ]; then' in text
    assert (
        "immutability was confirmed at dispatch but GitHub reports the "
        "published release as mutable"
    ) in text
    # ...and the un-opted-in path reports rather than asserting.
    assert "immutability was not opted into for this cut" in text
    assert "\n  release:\n" in text
    assert "needs: [cut, publish]" in text
    assert "is already public at the captured id" in text
    assert "after_by_tag=" in text
