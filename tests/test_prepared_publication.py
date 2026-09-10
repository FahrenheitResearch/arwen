"""Prepared publication keeps identities exact and retries within a real deadline.

All manifests and index responses are in-memory fixtures. No test contacts a
package index, uploads an artifact, or reads a local release packet.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
import zipfile

import pytest


TAG = "v9.8.7"
VERSION = "9.8.7"
COMMIT = "a" * 40
REPOSITORY = "FahrenheitResearch/arwen"
PIN = "b" * 64
OTHER_PIN = "c" * 64


@pytest.fixture(scope="module")
def publication():
    path = Path(__file__).resolve().parents[1] / "tools/promote_prepared_release.py"
    spec = importlib.util.spec_from_file_location("prepared_publication_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(filename: str) -> dict:
    return {
        "filename": filename,
        "bytes": 1000 + len(filename),
        "sha256": hashlib.sha256(filename.encode("utf-8")).hexdigest(),
    }


@pytest.fixture
def manifest():
    return {
        "schema": "arwen.publication-assets.v1",
        "status": "PREPARED_NOT_PUBLISHED",
        "engine_source_revision": COMMIT,
        "desktop_source_revision": "d" * 40,
        "github": {
            "repository": REPOSITORY,
            "target_version": TAG,
            "assets": [
                row(f"gpuwm-bridges-{TAG}-linux-x86_64.zip"),
                row(f"gpuwm-bridges-{TAG}-win-x86_64.zip"),
                row("bridge-bundle-manifest.json"),
                row("START-HERE.md"),
            ],
            "also_attach": ["PUBLICATION-ASSETS.json", "DOWNLOAD-SHA256SUMS.txt"],
        },
        "pypi": {
            "artifacts": [
                row(f"gpuwm-{VERSION}-py3-none-any.whl"),
                row(f"gpuwm-{VERSION}-py3-none-manylinux_2_28_x86_64.whl"),
                row(f"gpuwm-{VERSION}-py3-none-win_amd64.whl"),
                row(f"gpuwm-{VERSION}.tar.gz"),
                row(f"gpuwm_data-{VERSION}-py3-none-any.whl"),
                row(f"gpuwm_data-{VERSION}.tar.gz"),
            ]
        },
    }


@pytest.fixture
def engine_rows(manifest):
    return [item for item in manifest["pypi"]["artifacts"] if item["filename"].startswith("gpuwm-")]


def event_with_pin(pin: str = PIN) -> dict:
    return {"release": {"body": f"Release notes.\n<!-- arwen-publication-sha256: {pin} -->\n"}}


def index_payload(rows, *, project="gpuwm", version=VERSION):
    return {
        "info": {"name": project, "version": version},
        "urls": [
            {"filename": item["filename"], "size": item["bytes"],
             "digests": {"sha256": item["sha256"]}, "yanked": False}
            for item in rows
        ],
    }


def load(publication, document):
    return publication.load_manifest(document, TAG, COMMIT, REPOSITORY)


def test_manifest_pin_accepts_explicit_or_matching_release_marker(publication):
    assert publication.manifest_pin(PIN, {}) == PIN
    assert publication.manifest_pin("", event_with_pin()) == PIN
    assert publication.manifest_pin(PIN, event_with_pin()) == PIN


@pytest.mark.parametrize("explicit,event", [
    ("", {}),
    ("", {"release": {"body": "Unpinned release notes"}}),
    (PIN[:-1], {}),
    ("z" * 64, {}),
    (PIN, event_with_pin(OTHER_PIN)),
])
def test_manifest_pin_refuses_missing_invalid_or_conflicting_identity(publication, explicit, event):
    with pytest.raises(publication.PublicationError):
        publication.manifest_pin(explicit, event)


@pytest.mark.parametrize("second_pin", [PIN, OTHER_PIN])
def test_duplicate_release_markers_are_ambiguous_even_with_explicit_pin(publication, second_pin):
    event = event_with_pin()
    event["release"]["body"] += f"<!-- arwen-publication-sha256: {second_pin} -->"
    with pytest.raises(publication.PublicationError):
        publication.manifest_pin(PIN, event)


def test_a_malformed_second_release_marker_is_not_ignored(publication):
    event = event_with_pin()
    event["release"]["body"] += "<!-- arwen-publication-sha256: not-a-sha -->"
    with pytest.raises(publication.PublicationError):
        publication.manifest_pin(PIN, event)


def test_manifest_accepts_exact_six_distribution_release(publication, manifest):
    assert isinstance(load(publication, manifest), dict)


@pytest.mark.parametrize("path,value", [
    (("schema",), "arwen.publication-assets.v0"),
    (("engine_source_revision",), "e" * 40),
    (("github", "repository"), "OtherOwner/arwen"),
    (("github", "target_version"), "v9.8.8"),
])
def test_manifest_refuses_another_schema_commit_repository_or_tag(publication, manifest, path, value):
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(publication.PublicationError):
        load(publication, manifest)


@pytest.mark.parametrize("missing", [
    f"gpuwm-{VERSION}-py3-none-manylinux_2_28_x86_64.whl",
    f"gpuwm-{VERSION}-py3-none-win_amd64.whl",
    f"gpuwm_data-{VERSION}.tar.gz",
])
def test_manifest_refuses_incomplete_pypi_set(publication, manifest, missing):
    manifest["pypi"]["artifacts"] = [item for item in manifest["pypi"]["artifacts"] if item["filename"] != missing]
    with pytest.raises(publication.PublicationError):
        load(publication, manifest)


def test_manifest_refuses_same_count_but_wrong_distribution_version(publication, manifest):
    manifest["pypi"]["artifacts"][0] = row("gpuwm-9.8.8-py3-none-any.whl")
    with pytest.raises(publication.PublicationError):
        load(publication, manifest)


@pytest.mark.parametrize("missing", [
    f"gpuwm-bridges-{TAG}-linux-x86_64.zip",
    f"gpuwm-bridges-{TAG}-win-x86_64.zip",
    "bridge-bundle-manifest.json",
])
def test_manifest_refuses_missing_native_asset_contract(publication, manifest, missing):
    manifest["github"]["assets"] = [item for item in manifest["github"]["assets"] if item["filename"] != missing]
    with pytest.raises(publication.PublicationError):
        load(publication, manifest)


@pytest.mark.parametrize("filename", ["../escape.zip", "sub/file.zip", "sub\\file.zip", "C:\\escape.zip", "/absolute.zip", "", "bad\nname.zip"])
def test_manifest_refuses_filenames_that_are_not_safe_basenames(publication, manifest, filename):
    manifest["github"]["assets"].append(row(filename))
    with pytest.raises(publication.PublicationError):
        load(publication, manifest)


@pytest.mark.parametrize("section,key", [("github", "assets"), ("pypi", "artifacts")])
def test_manifest_refuses_duplicate_rows(publication, manifest, section, key):
    manifest[section][key].append(deepcopy(manifest[section][key][0]))
    with pytest.raises(publication.PublicationError):
        load(publication, manifest)


@pytest.mark.parametrize("field,value", [("bytes", 0), ("bytes", -1), ("bytes", True), ("bytes", "12"), ("sha256", "g" * 64), ("sha256", "a" * 63)])
def test_manifest_refuses_unusable_artifact_size_or_digest(publication, manifest, field, value):
    manifest["github"]["assets"][0][field] = value
    with pytest.raises(publication.PublicationError):
        load(publication, manifest)


def test_manifest_cannot_add_an_unhashed_attachment(publication, manifest):
    manifest["github"]["also_attach"].append("unproven-binary.exe")
    with pytest.raises(publication.PublicationError):
        load(publication, manifest)


def test_index_reconciliation_resumes_only_missing_exact_files(publication, engine_rows):
    assert publication.reconcile_index(engine_rows, 404, None, "gpuwm", VERSION) == engine_rows
    partial = index_payload(engine_rows[:2])
    assert publication.reconcile_index(engine_rows, 200, partial, "gpuwm", VERSION) == engine_rows[2:]
    assert publication.reconcile_index(engine_rows, 200, index_payload(engine_rows), "gpuwm", VERSION) == []


@pytest.mark.parametrize("change", ["foreign", "duplicate", "size", "digest", "yanked", "project", "version"])
def test_index_reconciliation_refuses_a_conflicting_publication(publication, engine_rows, change):
    payload = index_payload(engine_rows[:1])
    if change == "foreign":
        payload["urls"].append(index_payload([row("foreign-1.0.whl")])["urls"][0])
    elif change == "duplicate":
        payload["urls"].append(deepcopy(payload["urls"][0]))
    elif change == "size":
        payload["urls"][0]["size"] += 1
    elif change == "digest":
        payload["urls"][0]["digests"]["sha256"] = OTHER_PIN
    elif change == "yanked":
        payload["urls"][0]["yanked"] = True
    else:
        payload["info"]["name" if change == "project" else "version"] = "another-release"
    with pytest.raises(publication.PublicationError):
        publication.reconcile_index(engine_rows, 200, payload, "gpuwm", VERSION)


@pytest.mark.parametrize("status,payload", [(200, None), (200, {}), (401, None), (429, None), (500, None)])
def test_index_reconciliation_does_not_treat_errors_as_absence(publication, engine_rows, status, payload):
    with pytest.raises(publication.PublicationError):
        publication.reconcile_index(engine_rows, status, payload, "gpuwm", VERSION)


class VirtualTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0, "retry must make forward progress"
        self.sleeps.append(seconds)
        self.now += seconds


def test_index_wait_retries_404_and_partial_200_until_exact_set(publication, engine_rows):
    timeline = VirtualTime()
    full = index_payload(engine_rows)
    responses = [(404, None), (200, index_payload(engine_rows[:1])), (200, full)]
    calls = []

    def fetch(project, version, request_timeout):
        assert (project, version) == ("gpuwm", VERSION)
        assert 0 < request_timeout <= 20 - timeline.now
        calls.append(request_timeout)
        assert responses, "unexpected retry after a complete index"
        return responses.pop(0)

    result = publication.wait_for_index("gpuwm", VERSION, engine_rows, fetch, timeline.clock, timeline.sleep, timeout=20)
    assert isinstance(result, dict)
    assert len(calls) == 3
    assert len(timeline.sleeps) == 2


def test_index_wait_fails_changed_bytes_without_sleeping_or_retrying(publication, engine_rows):
    timeline = VirtualTime()
    payload = index_payload(engine_rows)
    payload["urls"][0]["digests"]["sha256"] = OTHER_PIN
    calls = []

    def fetch(project, version, request_timeout):
        calls.append(request_timeout)
        assert len(calls) == 1, "a byte mismatch must not be retried"
        return 200, payload

    with pytest.raises(publication.PublicationError):
        publication.wait_for_index("gpuwm", VERSION, engine_rows, fetch, timeline.clock, timeline.sleep, timeout=20)
    assert len(calls) == 1
    assert timeline.sleeps == []


def test_index_wait_deadline_counts_time_spent_fetching(publication, engine_rows):
    timeline = VirtualTime()
    deadline = 5.0
    calls = []

    def fetch(project, version, request_timeout):
        assert 0 < request_timeout <= deadline - timeline.now
        calls.append(request_timeout)
        assert len(calls) <= 3, "network time must count toward the deadline"
        timeline.now += min(3.0, request_timeout)
        return 404, None

    def sleep(seconds):
        assert seconds <= deadline - timeline.now, "backoff must respect remaining time"
        timeline.sleep(seconds)

    with pytest.raises(publication.PublicationError):
        publication.wait_for_index("gpuwm", VERSION, engine_rows, fetch, timeline.clock, sleep, timeout=deadline)
    assert calls
    assert timeline.now <= deadline


def test_index_wait_bounds_a_permanently_partial_200(publication, engine_rows):
    timeline = VirtualTime()
    calls = []

    def fetch(project, version, request_timeout):
        assert 0 < request_timeout <= 4 - timeline.now
        calls.append(request_timeout)
        assert len(calls) <= 8, "partial index must not spin indefinitely"
        return 200, index_payload(engine_rows[:1])

    with pytest.raises(publication.PublicationError):
        publication.wait_for_index("gpuwm", VERSION, engine_rows, fetch, timeline.clock, timeline.sleep, timeout=4)
    assert calls
    assert timeline.now <= 4


def test_index_wait_rejects_a_complete_response_after_its_deadline(publication, engine_rows):
    timeline = VirtualTime()

    def fetch(project, version, request_timeout):
        assert 0 < request_timeout <= 1
        timeline.now = 1.01
        return 200, index_payload(engine_rows)

    with pytest.raises(publication.PublicationError, match="deadline"):
        publication.wait_for_index("gpuwm", VERSION, engine_rows, fetch, timeline.clock, timeline.sleep, timeout=1)
    assert timeline.sleeps == []


def metadata_fixture(tmp_path, *, project="gpuwm", readme="Prepared release documentation.\n", archive_kind="wheel"):
    repo = tmp_path / "repo"
    project_root = repo / "gpuwm-data" if project == "gpuwm-data" else repo
    package = project.replace("-", "_")
    source_dir = project_root / package
    source_dir.mkdir(parents=True)
    source = b'"""Fixture package only."""\n'
    (source_dir / "__init__.py").write_bytes(source)
    (project_root / "README.md").write_text(readme, encoding="utf-8")
    config = f'[project]\nname = "{project}"\nreadme = "README.md"\n'
    if project == "gpuwm-data":
        config += 'dynamic = ["version"]\n[tool.setuptools.dynamic]\nversion = {file = ["gpuwm_data/VERSION"]}\n'
        (source_dir / "VERSION").write_text(VERSION + "\n", encoding="utf-8")
    else:
        config += f'version = "{VERSION}"\n'
    (project_root / "pyproject.toml").write_text(config, encoding="utf-8")
    metadata = (f"Metadata-Version: 2.4\nName: {project}\nVersion: {VERSION}\n"
                f"Description-Content-Type: text/markdown\n\n{readme}")
    if archive_kind == "wheel":
        artifact = tmp_path / f"{package}-{VERSION}-py3-none-any.whl"
        with zipfile.ZipFile(artifact, "w") as archive:
            archive.writestr(f"{package}-{VERSION}.dist-info/METADATA", metadata.encode("utf-8"))
            archive.writestr(f"{package}/__init__.py", source)
    else:
        artifact = tmp_path / f"{package}-{VERSION}.tar.gz"
        with tarfile.open(artifact, "w:gz") as archive:
            for name, payload in [("PKG-INFO", metadata.encode("utf-8")), (f"{package}/__init__.py", source)]:
                member = tarfile.TarInfo(f"{package}-{VERSION}/{name}")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    return repo, artifact


def test_metadata_accepts_companion_dynamic_version_file(publication, tmp_path):
    repo, wheel = metadata_fixture(tmp_path, project="gpuwm-data")
    result = publication.check_metadata(wheel, "gpuwm-data", VERSION, repo)
    assert result["version"] == VERSION


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
def test_metadata_compares_utf8_readme_without_replacement_characters(publication, tmp_path, archive_kind):
    repo, artifact = metadata_fixture(tmp_path, readme="# ArWen\n\nWeather in °F — native maps.\n", archive_kind=archive_kind)
    result = publication.check_metadata(artifact, "gpuwm", VERSION, repo)
    assert result["project"] == "gpuwm"


def capture_fixture(tmp_path, manifest):
    """Self-consistent local capture, with no real archive or remote services."""
    packet = tmp_path / "packet"
    assets = packet / "assets"
    assets.mkdir(parents=True)
    document = deepcopy(manifest)
    for item in document["github"]["assets"] + document["pypi"]["artifacts"]:
        payload = ("fixture: " + item["filename"]).encode("utf-8")
        item.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        (assets / item["filename"]).write_bytes(payload)
    publication_manifest = assets / "PUBLICATION-ASSETS.json"
    publication_manifest.write_text(json.dumps(document), encoding="utf-8")
    checksums = "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
                        for path in sorted(assets.iterdir()))
    (assets / "DOWNLOAD-SHA256SUMS.txt").write_text(checksums, encoding="utf-8")
    captured = {
        "schema": "gpuwm.prepared-release-capture.v1",
        "tag": TAG, "version": VERSION, "commit": COMMIT, "repository": REPOSITORY,
        "release_id": 42, "draft": True, "prerelease": False, "immutable_required": False,
        "manifest_sha256": hashlib.sha256(publication_manifest.read_bytes()).hexdigest(),
        "assets": [
            {"filename": path.name, "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "asset_id": index}
            for index, path in enumerate(sorted(assets.iterdir()), start=1)
        ],
    }
    (packet / "CAPTURE.json").write_text(json.dumps(captured), encoding="utf-8")
    return packet, captured


def test_capture_plan_accepts_the_known_manifest_identity(publication, tmp_path, manifest, monkeypatch):
    packet, captured = capture_fixture(tmp_path, manifest)
    monkeypatch.setenv("PUBLICATION_COMMIT", COMMIT)
    monkeypatch.setenv("PUBLICATION_MANIFEST_SHA256", captured["manifest_sha256"])
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    actual, plan = publication._capture_plan(packet)
    assert actual == captured
    assert plan["commit"] == COMMIT


@pytest.mark.parametrize("variable,value", [
    ("PUBLICATION_COMMIT", "e" * 40),
    ("PUBLICATION_MANIFEST_SHA256", PIN),
])
def test_capture_cannot_rebind_known_workflow_authority(publication, tmp_path, manifest, monkeypatch, variable, value):
    packet, captured = capture_fixture(tmp_path, manifest)
    monkeypatch.setenv("PUBLICATION_COMMIT", COMMIT)
    monkeypatch.setenv("PUBLICATION_MANIFEST_SHA256", captured["manifest_sha256"])
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    monkeypatch.setenv(variable, value)
    with pytest.raises(publication.PublicationError):
        publication._capture_plan(packet)


@pytest.mark.parametrize("operation", ["verify", "smoke"])
def test_existing_proof_or_smoke_output_is_preserved(publication, tmp_path, manifest, monkeypatch, operation):
    packet, captured = capture_fixture(tmp_path, manifest)
    plan = load(publication, manifest)
    repo = tmp_path / "verification-source"
    repo.mkdir()
    output = tmp_path / "existing-proof"
    output.mkdir()
    sentinel = output / "previous-receipt.json"
    original = b'{"status":"existing evidence"}\n'
    sentinel.write_bytes(original)
    monkeypatch.setattr(publication, "_capture_plan", lambda path: (captured, plan))
    monkeypatch.setattr(publication, "proof", lambda path: {"captured": captured, "plan": plan})

    def git_read(command, **kwargs):
        if "rev-parse" in command:
            return COMMIT + "\n"
        if "status" in command:
            return ""
        raise AssertionError(f"unexpected external command: {command}")

    def no_environment_creation(*args, **kwargs):
        raise AssertionError("existing output must be refused before creating an environment")

    monkeypatch.setattr(publication.subprocess, "check_output", git_read)
    monkeypatch.setattr(publication, "checked_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(publication.venv, "EnvBuilder", no_environment_creation)
    args = SimpleNamespace(packet=packet, repo=repo, out=output, proof=tmp_path / "proof", platform="linux-x86_64")
    with pytest.raises(publication.PublicationError):
        getattr(publication, operation)(args)
    assert sentinel.read_bytes() == original


def native_wheel_fixture(tmp_path, *, platform="linux-x86_64", mutation=None):
    suffix = "-manylinux_2_28_x86_64.whl" if platform == "linux-x86_64" else "-win_amd64.whl"
    names = ("arwen-tui", "libnetcdf_writer.so") if platform == "linux-x86_64" else ("arwen-tui.exe", "netcdf_writer.dll")
    rows = []
    members = {}
    prefix = "gpuwm/libexec/bridges/"
    for name, artifact, kind in zip(names, ("arwen-tui", "netcdf_writer"), ("executable", "library")):
        payload = ("fixture native bytes: " + name).encode("utf-8")
        rows.append({"artifact": artifact, "filename": name, "kind": kind, "bytes": len(payload),
                     "sha256": hashlib.sha256(payload).hexdigest()})
        members[prefix + name] = payload
    pins = {"schema": "gpuwm-bridge-pins-v1", "release": TAG, "platforms": {platform: {"binaries": deepcopy(rows)}}}
    bundle = {"schema": "gpuwm-wheel-bridge-bundle-v1", "platform": platform, "artifacts": deepcopy(rows)}
    if mutation == "missing-native":
        del members[prefix + names[1]]
    elif mutation == "changed-native":
        members[prefix + names[1]] += b"changed"
    elif mutation == "extra-native":
        members[prefix + "stray.dll"] = b"unlisted bytes"
    elif mutation == "wrong-platform":
        bundle["platform"] = "win-x86_64"
    elif mutation == "wrong-artifact":
        bundle["artifacts"][1]["artifact"] = "a-different-library"
    elif mutation == "wrong-kind":
        bundle["artifacts"][1]["kind"] = "script"
    elif mutation == "changed-manifest-hash":
        bundle["artifacts"][1]["sha256"] = OTHER_PIN
    elif mutation == "duplicate-manifest-row":
        bundle["artifacts"].append(deepcopy(bundle["artifacts"][0]))
    if mutation != "missing-manifest":
        members[prefix + "BUNDLE.json"] = json.dumps(bundle).encode("utf-8")
    wheel = tmp_path / (f"gpuwm-{VERSION}-py3-none" + suffix)
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return wheel, pins


@pytest.mark.parametrize("platform", ["linux-x86_64", "win-x86_64"])
def test_embedded_native_files_match_the_platform_pins(publication, tmp_path, platform):
    wheel, pins = native_wheel_fixture(tmp_path, platform=platform)
    assert isinstance(publication.check_embedded_natives(wheel, pins, platform), dict)


@pytest.mark.parametrize("mutation", ["missing-native", "changed-native", "extra-native", "missing-manifest", "wrong-platform", "wrong-artifact", "wrong-kind", "changed-manifest-hash", "duplicate-manifest-row"])
def test_embedded_native_gate_rejects_wheel_bundle_disagreement(publication, tmp_path, mutation):
    wheel, pins = native_wheel_fixture(tmp_path, mutation=mutation)
    with pytest.raises(publication.PublicationError):
        publication.check_embedded_natives(wheel, pins, "linux-x86_64")


def test_pure_wheel_must_not_hide_a_native_payload(publication, tmp_path):
    wheel, pins = native_wheel_fixture(tmp_path)
    with pytest.raises(publication.PublicationError):
        publication.check_embedded_natives(wheel, pins, None)


def test_pure_wheel_without_staged_native_files_passes(publication, tmp_path):
    repo, wheel = metadata_fixture(tmp_path)
    assert isinstance(publication.check_embedded_natives(wheel, {"platforms": {}}, None), dict)


def python_inventory_fixture(tmp_path, mutation=None):
    repo, wheel = metadata_fixture(tmp_path)
    with (repo / "pyproject.toml").open("a", encoding="utf-8") as stream:
        stream.write('\n[tool.setuptools.packages.find]\ninclude = ["gpuwm", "gpuwm.*"]\nnamespaces = true\n')
    (repo / "setup.py").write_text(
        'DEVELOPMENT_MODULE_GLOBS: dict[str, tuple[str, ...]] = {"gpuwm": ("dev_*",)}\n'
        'raise AssertionError("inventory must inspect setup.py without executing it")\n',
        encoding="utf-8",
    )
    source = b'"""Required runtime module."""\n'
    development = b'"""Excluded development probe."""\n'
    (repo / "gpuwm/core.py").write_bytes(source)
    (repo / "gpuwm/dev_probe.py").write_bytes(development)
    with zipfile.ZipFile(wheel, "a") as archive:
        if mutation != "omitted-runtime-module":
            archive.writestr("gpuwm/core.py", source)
        if mutation == "included-development-module":
            archive.writestr("gpuwm/dev_probe.py", development)
    return repo, wheel


def test_python_inventory_respects_declared_module_exclusions_without_running_setup(publication, tmp_path):
    repo, wheel = python_inventory_fixture(tmp_path)
    publication.check_python_inventory(wheel, repo, "gpuwm")


@pytest.mark.parametrize("mutation", ["omitted-runtime-module", "included-development-module"])
def test_python_inventory_refuses_an_omission_or_excluded_probe(publication, tmp_path, mutation):
    repo, wheel = python_inventory_fixture(tmp_path, mutation)
    with pytest.raises(publication.PublicationError):
        publication.check_python_inventory(wheel, repo, "gpuwm")
