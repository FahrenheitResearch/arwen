"""The parsed workflow and fake controller endpoints enforce publication order.

No test performs a network request or invokes the real publication entrypoint.
The callable controller is exercised with synthetic proofs and local byte files.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
MANIFEST_SHA = "b" * 64
VERSION = "9.8.7"


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load((ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def publication():
    spec = importlib.util.spec_from_file_location("state_machine_publication", ROOT / "tools/promote_prepared_release.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def needs(job):
    value = job.get("needs", [])
    return {value} if isinstance(value, str) else set(value)


def ancestors(jobs, name):
    found = set()
    pending = list(needs(jobs[name]))
    while pending:
        parent = pending.pop()
        assert parent != name, "publication job graph has a cycle"
        if parent not in found:
            found.add(parent)
            pending.extend(needs(jobs[parent]))
    return found


def scripts(job):
    return "\n".join(step.get("run", "") for step in job["steps"])


def test_public_mutation_waits_for_exact_assets_and_both_platforms(workflow):
    jobs = workflow["jobs"]
    ordered = ["test", "release_authority", "prepare", "qualify", "assets", "publish", "release"]
    for predecessor, successor in zip(ordered, ordered[1:]):
        assert predecessor in ancestors(jobs, successor), (predecessor, successor)
    matrix = jobs["qualify"]["strategy"]["matrix"]["include"]
    assert {row["platform"] for row in matrix} == {"linux-x86_64", "win-x86_64"}
    assert len(matrix) == 2
    assert jobs["qualify"]["strategy"]["fail-fast"] is False


def test_release_events_and_dispatch_share_a_serialized_authority(workflow):
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 parses on as true.
    assert triggers["release"]["types"] == ["published"]
    assert "pull_request" in triggers
    dispatch = triggers["workflow_dispatch"]["inputs"]
    assert dispatch["release_tag"]["required"] is True
    assert "publication_manifest_sha256" in dispatch
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert "github.ref" in workflow["concurrency"]["group"]
    for name, job in workflow["jobs"].items():
        if name != "test":
            assert job["if"] == "github.event_name != 'pull_request'"


def test_oidc_and_github_write_are_in_separate_jobs(workflow):
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    assert jobs["publish"]["permissions"] == {"id-token": "write"}
    assert jobs["publish"]["environment"] == "pypi"
    for name, job in jobs.items():
        permissions = job.get("permissions", workflow["permissions"])
        if name != "publish":
            assert permissions.get("id-token") != "write"
        if permissions.get("contents") == "write":
            assert name in {"release_authority", "assets"}
    publisher = jobs["publish"]
    assert "GH_TOKEN" not in json.dumps(publisher)
    assert not any(step.get("uses", "").startswith("actions/checkout@") for step in publisher["steps"])
    for step in publisher["steps"]:
        if step.get("uses", "").startswith("pypa/gh-action-pypi-publish@"):
            assert not {"password", "user", "skip-existing"} & set(step.get("with", {}))


def test_release_jobs_execute_the_tested_controller_after_hash_verification(workflow):
    jobs = workflow["jobs"]
    assert "controller_sha256" in jobs["test"]["outputs"]
    for name, job in jobs.items():
        if name == "test":
            continue
        steps = job["steps"]
        uses = [i for i, step in enumerate(steps)
                if re.search(r"python\s+(?:tools|controller)/promote_prepared_release\.py\s", step.get("run", ""))]
        checks = [i for i, step in enumerate(steps)
                  if "sha256sum --check --strict" in step.get("run", "")
                  and step.get("env", {}).get("CONTROLLER_SHA256") == "${{ needs.test.outputs.controller_sha256 }}"]
        assert uses and len(checks) == 1, name
        assert checks[0] < min(uses), name
        if name in {"prepare", "qualify"}:
            checkouts = [step for step in steps if step.get("uses", "").startswith("actions/checkout@")]
            assert len(checkouts) == 1
            assert checkouts[0]["with"]["ref"] == "${{ needs.release_authority.outputs.commit }}"
            assert checkouts[0]["with"]["persist-credentials"] is False
        else:
            controllers = [step for step in steps if step.get("with", {}).get("name") == "publication-controller"]
            assert len(controllers) == 1
            assert controllers[0]["uses"].startswith("actions/download-artifact@")
        if name != "release_authority":
            assert job["env"]["PUBLICATION_COMMIT"] == "${{ needs.release_authority.outputs.commit }}"
            assert job["env"]["PUBLICATION_MANIFEST_SHA256"] == "${{ needs.release_authority.outputs.manifest_sha256 }}"


def test_actions_and_artifact_handoffs_are_pinned_to_this_run(workflow):
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            action = step.get("uses")
            if action:
                assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", action), action
            if action and action.startswith("actions/download-artifact@"):
                assert not {"run-id", "repository", "github-token"} & set(step.get("with", {}))
            if action and action.startswith("actions/upload-artifact@"):
                assert step["with"]["if-no-files-found"] == "error"


def test_promotion_uses_prepared_bytes_and_keeps_existing_packaging_checks(workflow):
    jobs = workflow["jobs"]
    for name, job in jobs.items():
        if name != "test":
            assert not re.search(r"python\s+-m\s+build|cargo\s+build|gh\s+release\s+(?:upload|create|delete)|git\s+(?:push|tag)", scripts(job)), name
    test_script = scripts(jobs["test"])
    for name in ("bridge_fetch", "table_fetch", "package_data_coverage", "companion_distribution",
                 "configs_are_packaged", "doctor", "doctor_nexrad", "cli", "tui_cli", "source_adapters",
                 "namelist_compat", "native_wrf_distribution", "prepared_publication",
                 "publish_workflow_state_machine", "publish_dist_shape_agreement", "release_version_declaration",
                 "bridge_bundle_adopt", "verify_source_bridge_pins", "verify_release_artifacts"):
        assert f"tests/test_{name}.py" in test_script


def test_docker_publisher_upload_paths_are_workspace_relative(workflow):
    steps = workflow["jobs"]["publish"]["steps"]
    for phase in ("data", "native", "pure"):
        stage = next(step for step in steps if step.get("id") == phase)
        upload = next(step for step in steps if step.get("if") == f"steps.{phase}.outputs.upload_required == 'true'")
        directory = upload["with"]["packages-dir"].rstrip("/")
        assert directory and not PurePosixPath(directory).is_absolute()
        assert not any(token in directory for token in ("$", "..", "\\", ":")), directory
        assert re.search(r"--out\s+[\"']?" + re.escape(directory) + r"[\"']?(?:\s|$)", stage["run"])


@pytest.fixture
def controller_case(publication, monkeypatch, tmp_path):
    filenames = {
        "gpuwm-data": [f"gpuwm_data-{VERSION}-py3-none-any.whl", f"gpuwm_data-{VERSION}.tar.gz"],
        "gpuwm": [f"gpuwm-{VERSION}-py3-none-any.whl", f"gpuwm-{VERSION}-py3-none-manylinux_2_28_x86_64.whl",
                  f"gpuwm-{VERSION}-py3-none-win_amd64.whl", f"gpuwm-{VERSION}.tar.gz"],
    }
    dists = tmp_path / "dists"
    dists.mkdir()
    pypi = {}
    for project, names in filenames.items():
        pypi[project] = []
        for name in names:
            payload = ("proven synthetic " + name).encode()
            (dists / name).write_bytes(payload)
            pypi[project].append({"filename": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    captured = {"commit": COMMIT, "manifest_sha256": MANIFEST_SHA, "repository": "FahrenheitResearch/arwen",
                "tag": "v" + VERSION, "release_id": 42, "prerelease": False}
    proven = {"captured": captured, "plan": {"version": VERSION, "pypi": pypi}}
    monkeypatch.setattr(publication, "proof", lambda path: deepcopy(proven))
    events = []
    state = {"draft": True, "indexes": {project: (404, None) for project in pypi}}

    def fetch(project, version, timeout=30):
        assert version == VERSION
        events.append(("index", project))
        return deepcopy(state["indexes"][project])

    class FakeGitHub:
        def __init__(self, repository):
            assert repository == captured["repository"]
        def json(self, suffix, *, data=None):
            assert suffix == "/releases/42"
            assert data == {"draft": False, "prerelease": False}
            events.append(("patch", suffix))
            state["draft"] = False
            return {"id": 42}

    def remote(client, selected, *, require_public):
        assert selected == captured
        events.append(("remote", require_public))
        assert not require_public or state["draft"] is False
        return {"id": 42, "draft": state["draft"], "immutable": True}

    monkeypatch.setattr(publication, "fetch_index", fetch)
    monkeypatch.setattr(publication, "GitHub", FakeGitHub)
    monkeypatch.setattr(publication, "verify_remote", remote)
    monkeypatch.setattr(publication, "output_values", lambda values: events.append(("outputs", values)))
    smokes = tmp_path / "smokes"
    for platform in ("linux-x86_64", "win-x86_64"):
        publication.write_json(smokes / platform / "SMOKE.json", {
            "platform": platform, "status": "PASS", "commit": COMMIT, "manifest_sha256": MANIFEST_SHA})
    args = SimpleNamespace(proof=tmp_path / "proof", smokes=smokes, out=tmp_path / "PUBLIC.json")
    return SimpleNamespace(args=args, proven=proven, events=events, state=state, dists=dists)


def payload(project, rows):
    return {"info": {"name": project, "version": VERSION}, "urls": [
        {"filename": row["filename"], "size": row["bytes"], "digests": {"sha256": row["sha256"]}}
        for row in rows]}


def test_draft_promotion_checks_both_indexes_before_the_only_write(publication, controller_case):
    case = controller_case
    publication.promote(case.args)
    assert case.events == [("index", "gpuwm-data"), ("index", "gpuwm"), ("remote", False),
                           ("patch", "/releases/42"), ("remote", True)]
    assert publication.read_json(case.args.out)["public_before_pypi"] is True


def test_an_already_public_retry_makes_no_github_write(publication, controller_case):
    case = controller_case
    case.state["draft"] = False
    publication.promote(case.args)
    assert not any(event[0] == "patch" for event in case.events)
    assert publication.read_json(case.args.out)["status"] == "PASS"


@pytest.mark.parametrize("project", ["gpuwm-data", "gpuwm"])
def test_a_pypi_collision_prevents_github_promotion(publication, controller_case, project):
    case = controller_case
    collision = payload(project, case.proven["plan"]["pypi"][project])
    collision["urls"][0]["digests"]["sha256"] = "f" * 64
    case.state["indexes"][project] = (200, collision)
    with pytest.raises(publication.PublicationError, match="differs"):
        publication.promote(case.args)
    assert not any(event[0] in {"remote", "patch"} for event in case.events)
    assert not case.args.out.exists()


@pytest.mark.parametrize("field,value", [("status", "FAIL"), ("commit", "c" * 40),
                                         ("manifest_sha256", "d" * 64), ("platform", "linux-x86_64")])
def test_invalid_platform_proof_stops_before_any_remote_access(publication, controller_case, field, value):
    case = controller_case
    path = case.args.smokes / "win-x86_64/SMOKE.json"
    row = publication.read_json(path)
    row[field] = value
    publication.write_json(path, row)
    with pytest.raises(publication.PublicationError):
        publication.promote(case.args)
    assert case.events == []
    assert not case.args.out.exists()


def test_stage_rechecks_every_local_file_before_copying_any(publication, controller_case):
    case = controller_case
    bad = case.proven["plan"]["pypi"]["gpuwm"][-1]["filename"]
    (case.dists / bad).write_bytes(b"tampered")
    args = SimpleNamespace(proof=case.args.proof, dists=case.dists, project="gpuwm-data", phase="all", out=case.args.out)
    with pytest.raises(publication.PublicationError, match="artifact changed"):
        publication.stage_missing(args)
    assert not args.out.exists()


def test_partial_index_retry_stages_only_missing_exact_bytes(publication, controller_case):
    case = controller_case
    rows = case.proven["plan"]["pypi"]["gpuwm"]
    native = [row for row in rows if row["filename"].endswith(("manylinux_2_28_x86_64.whl", "win_amd64.whl"))]
    case.state["indexes"]["gpuwm"] = (200, payload("gpuwm", native[:1]))
    args = SimpleNamespace(proof=case.args.proof, dists=case.dists, project="gpuwm", phase="native", out=case.args.out)
    publication.stage_missing(args)
    assert [path.name for path in args.out.iterdir()] == [native[1]["filename"]]
    assert (args.out / native[1]["filename"]).read_bytes() == (case.dists / native[1]["filename"]).read_bytes()
    assert case.events[-1] == ("outputs", {"upload_required": "true", "missing_count": "1"})


def test_complete_index_retry_does_not_upload_again(publication, controller_case):
    case = controller_case
    rows = case.proven["plan"]["pypi"]["gpuwm-data"]
    case.state["indexes"]["gpuwm-data"] = (200, payload("gpuwm-data", rows))
    args = SimpleNamespace(proof=case.args.proof, dists=case.dists, project="gpuwm-data", phase="all", out=case.args.out)
    publication.stage_missing(args)
    assert list(args.out.iterdir()) == []
    assert case.events[-1] == ("outputs", {"upload_required": "false", "missing_count": "0"})
