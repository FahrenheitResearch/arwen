"""Executable contracts for the privileged publication state machine.

The release workflow intentionally keeps its authority checks inline so the
privileged jobs do not execute repository code.  These tests therefore extract
the literal ``run: |`` blocks and execute them with fake GitHub and PyPI
endpoints.  A substring assertion cannot prove retry or refusal behavior.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import stat
import subprocess
import textwrap
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish.yml"
TAG = "v9.8.7"
VERSION = "9.8.7"
RELEASE_ID = 42
COMMIT_SHA = "a" * 40


def _step_script(marker: str) -> str:
    """Extract one named/id'd step's literal Bash body from the workflow."""

    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    markers = {f"- name: {marker}", f"- id: {marker}"}
    for start, line in enumerate(lines):
        if not line.startswith("      - ") or line.strip() not in markers:
            continue
        for run_line in range(start + 1, len(lines)):
            candidate = lines[run_line]
            if candidate.startswith("      - "):
                break
            if candidate == "        run: |":
                body: list[str] = []
                for raw in lines[run_line + 1 :]:
                    if raw and not raw.startswith("          "):
                        break
                    body.append(raw[10:] if raw else "")
                script = "\n".join(body).rstrip() + "\n"
                assert script.strip(), marker
                return script
        raise AssertionError(f"workflow step {marker!r} has no literal run block")
    raise AssertionError(f"workflow step {marker!r} was not found")


def _write_executable(path: Path, source: str) -> None:
    path.write_bytes(textwrap.dedent(source).lstrip().encode("utf-8"))
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _install_fake_endpoints(root: Path) -> None:
    # Idempotent: a test that runs two workflow steps in one workspace
    # (two ingresses against the same fixture endpoints) installs once.
    fake_bin = root / "fake-bin"
    fake_bin.mkdir(exist_ok=True)
    _write_executable(
        fake_bin / "gh",
        r'''
        #!/usr/bin/env python3
        import json
        import os
        from pathlib import Path
        import sys

        args = sys.argv[1:]
        log = Path(os.environ.get("FAKE_GH_LOG", "gh.log"))
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(args) + "\n")

        def option(name):
            try:
                return args[args.index(name) + 1]
            except (ValueError, IndexError):
                return None

        def emit(payload):
            query = option("--jq") or option("-q")
            if query in {".sha", ".object.sha", ".object.type"}:
                value = payload
                for part in query.removeprefix(".").split("."):
                    value = value[part]
                print(value)
            else:
                print(json.dumps(payload, separators=(",", ":")))

        method = option("--method") or option("-X") or "GET"
        endpoint = next(
            (arg for arg in reversed(args) if arg.startswith("/repos/")), ""
        )
        if method == "DELETE" and "/releases/assets/" in endpoint:
            raise SystemExit(0)
        if method == "PATCH" and "/releases/" in endpoint:
            path = os.environ.get("GH_PATCH_RESPONSE")
            if path:
                emit(json.loads(Path(path).read_text(encoding="utf-8")))
            raise SystemExit(0)
        if "releases?per_page=" in endpoint:
            sequence_path = os.environ.get("GH_RELEASE_LIST_SEQUENCE")
            if sequence_path:
                sequence = json.loads(
                    Path(sequence_path).read_text(encoding="utf-8")
                )
                counter = Path(
                    os.environ.get("GH_RELEASE_LIST_COUNTER", "gh-list-counter")
                )
                index = (
                    int(counter.read_text(encoding="ascii"))
                    if counter.exists()
                    else 0
                )
                counter.write_text(str(index + 1), encoding="ascii")
                emit(sequence[min(index, len(sequence) - 1)])
            else:
                sys.stdout.write(
                    Path(os.environ["GH_RELEASE_LIST_JSON"]).read_text(
                        encoding="utf-8"
                    )
                )
            raise SystemExit(0)
        if "/releases/" in endpoint:
            sequence = json.loads(
                Path(os.environ["GH_RELEASE_SEQUENCE"]).read_text(
                    encoding="utf-8"
                )
            )
            counter = Path(os.environ.get("GH_RELEASE_COUNTER", "gh-counter"))
            index = int(counter.read_text(encoding="ascii")) if counter.exists() else 0
            counter.write_text(str(index + 1), encoding="ascii")
            emit(sequence[min(index, len(sequence) - 1)])
            raise SystemExit(0)
        if "/git/ref/tags/" in endpoint:
            emit(
                {
                    "object": {
                        "type": "commit",
                        "sha": os.environ["EXPECTED_COMMIT_SHA"],
                    }
                }
            )
            raise SystemExit(0)
        if "/git/tags/" in endpoint:
            emit(
                {
                    "object": {
                        "type": "commit",
                        "sha": os.environ["EXPECTED_COMMIT_SHA"],
                    }
                }
            )
            raise SystemExit(0)
        if "/commits/" in endpoint:
            emit({"sha": os.environ["EXPECTED_COMMIT_SHA"]})
            raise SystemExit(0)
        print(f"unexpected fake gh invocation: {args!r}", file=sys.stderr)
        raise SystemExit(97)
        ''',
    )
    _write_executable(
        fake_bin / "git",
        r'''
        #!/usr/bin/env python3
        import os
        import sys

        args = sys.argv[1:]
        if args[:1] == ["rev-parse"]:
            print(os.environ["EXPECTED_COMMIT_SHA"])
            raise SystemExit(0)
        print(f"unexpected fake git invocation: {args!r}", file=sys.stderr)
        raise SystemExit(96)
        ''',
    )
    _write_executable(
        fake_bin / "curl",
        r'''
        #!/usr/bin/env python3
        import json
        import os
        from pathlib import Path
        import shutil
        import sys

        args = sys.argv[1:]
        log = Path(os.environ.get("FAKE_CURL_LOG", "curl.log"))
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(args) + "\n")
        url = next((arg for arg in reversed(args) if arg.startswith("http")), "")
        if "uploads.github.com" in url:
            raise SystemExit(0)
        if "api.github.com" in url and "/releases/assets/" in url:
            asset_id = url.rstrip("/").rsplit("/", 1)[-1]
            source = Path(os.environ["FAKE_ASSET_DIR"]) / asset_id
            try:
                output = args[args.index("--output") + 1]
            except (ValueError, IndexError):
                output = args[args.index("-o") + 1]
            shutil.copyfile(source, output)
            raise SystemExit(0)
        if "pypi.org" in url:
            try:
                output = args[args.index("--output") + 1]
            except (ValueError, IndexError):
                output = args[args.index("-o") + 1]
            map_path = os.environ.get("PYPI_RESPONSE_MAP")
            sequence_path = os.environ.get("PYPI_RESPONSE_SEQUENCE")
            if map_path:
                # Keyed by the project the URL names, because the release
                # steps under test read more than one PyPI project and a
                # purely call-ordered fake could not catch a step asking
                # for the wrong one.
                project = url.split("/pypi/", 1)[1].split("/", 1)[0]
                mapping = json.loads(
                    Path(map_path).read_text(encoding="utf-8")
                )
                entry = mapping[project]
                shutil.copyfile(entry["file"], output)
                sys.stdout.write(str(entry.get("status", "200")))
            elif sequence_path:
                sequence = json.loads(
                    Path(sequence_path).read_text(encoding="utf-8")
                )
                counter = Path(
                    os.environ.get("PYPI_RESPONSE_COUNTER", "pypi-counter")
                )
                index = (
                    int(counter.read_text(encoding="ascii"))
                    if counter.exists()
                    else 0
                )
                counter.write_text(str(index + 1), encoding="ascii")
                entry = sequence[min(index, len(sequence) - 1)]
                shutil.copyfile(entry["file"], output)
                sys.stdout.write(str(entry["status"]))
            else:
                shutil.copyfile(os.environ["PYPI_RESPONSE_FILE"], output)
                sys.stdout.write(os.environ.get("PYPI_HTTP_STATUS", "200"))
            raise SystemExit(0)
        print(f"unexpected fake curl invocation: {args!r}", file=sys.stderr)
        raise SystemExit(98)
        ''',
    )


def _base_env(root: Path) -> dict[str, str]:
    (root / "runner-temp").mkdir(exist_ok=True)
    (root / "home").mkdir(exist_ok=True)
    return {
        "DISPATCH_RELEASE_TAG": TAG,
        # The dispatch ingress: a draft this run promotes at the end.  The
        # release-event ingress sets EXPECTED_DRAFT=false and is exercised
        # by its own test.
        "EXPECTED_DRAFT": "true",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "CAPTURED_PRERELEASE": "false",
        "STABLE_RELEASE_EXPECTED": "",
        "CAPTURED_REMOTE_COMMIT": COMMIT_SHA,
        "CAPTURED_RELEASE_ID": str(RELEASE_ID),
        "GH_TOKEN": "fixture-token",
        "GH_REPO": "example/project",
        "GITHUB_REPOSITORY": "example/project",
        "GITHUB_REF_NAME": TAG,
        "GITHUB_REF_TYPE": "tag",
        "GITHUB_REF": f"refs/tags/{TAG}",
        "GITHUB_SHA": COMMIT_SHA,
        "GITHUB_OUTPUT": "github-output.txt",
        "GITHUB_WORKSPACE": ".",
        "RUNNER_TEMP": "runner-temp",
        "HOME": "home",
        "RELEASE_ID": str(RELEASE_ID),
        "RELEASE_TAG": TAG,
        "PYPI_PROJECT": "gpuwm",
        "PYPI_DATA_PROJECT": "gpuwm-data",
        "PYPI_VERSION": VERSION,
        "EXPECTED_COMMIT_SHA": COMMIT_SHA,
        "RELEASE_COMMIT": COMMIT_SHA,
        "RELEASE_COMMIT_SHA": COMMIT_SHA,
        "TAG_COMMIT_SHA": COMMIT_SHA,
        "IMMUTABLE_RELEASES_CONFIRMED": "true",
        "GH_RELEASE_LIST_JSON": "release-list.json",
        "GH_RELEASE_SEQUENCE": "release-sequence.json",
        "GH_RELEASE_COUNTER": "gh-release-counter.txt",
        "FAKE_GH_LOG": "gh.log",
        "FAKE_CURL_LOG": "curl.log",
        "PYPI_RESPONSE_FILE": "pypi-response.json",
        "PYPI_HTTP_STATUS": "200",
    }


def _bash_runtime() -> tuple[str, str, str]:
    """Return (mode, launcher, Python command) for a capable POSIX shell."""

    native: list[tuple[str, str]] = []
    if os.name == "nt":
        native.extend(
            (str(path), "python")
            for path in (
                Path(r"C:\Program Files\Git\bin\bash.exe"),
                Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
            )
            if path.exists()
        )
    else:
        bash = shutil.which("bash")
        if bash:
            native.append((bash, "python3"))

    required = " && ".join(
        (
            "command -v jq >/dev/null",
            "command -v sha256sum >/dev/null",
            "command -v stat >/dev/null",
        )
    )
    for bash, python in native:
        probe = subprocess.run(
            [bash, "-c", f"{required} && {python} -c 'import sys'"],
            text=True,
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            return "native", bash, python

    if os.name == "nt" and (wsl := shutil.which("wsl.exe")):
        probe = subprocess.run(
            [
                wsl,
                "--exec",
                "bash",
                "-c",
                f"{required} && python3 -c 'import sys'",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            return "wsl", wsl, "/usr/bin/python3"

    pytest.skip("publication contracts require Bash, jq, and GNU coreutils")


def _run_bash(
    root: Path, script: str, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    mode, launcher, python = _bash_runtime()
    process_env = os.environ.copy()
    process_env.update(env)
    # Keep the workflow's control flow literal while replacing only its
    # endpoints: the two network commands, the one local Git query, and the
    # interpreter name (a runner's `python` is provided by setup-python and
    # need not exist under the shell running these fixtures).  Explicit paths
    # are reliable through the Windows-to-WSL Bash bridge, where shell-
    # function shims are not.
    fixture_script = (
        script.replace("gh api", f"{python} ./fake-bin/gh api")
        .replace("curl ", f"{python} ./fake-bin/curl ")
        .replace("git rev-parse", f"{python} ./fake-bin/git rev-parse")
        .replace("python -c", f"{python} -c")
    )
    exports = "\n".join(
        f"export {name}={shlex.quote(value)}" for name, value in sorted(env.items())
    )
    runner = root / "fixture-workflow-step.sh"
    runner.write_text(f"{exports}\n{fixture_script}", encoding="utf-8", newline="\n")
    command = (
        [launcher, "--cd", str(root), "--exec", "bash", f"./{runner.name}"]
        if mode == "wsl"
        else [launcher, f"./{runner.name}"]
    )
    return subprocess.run(
        command,
        cwd=root,
        env=process_env,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_sums(path: Path, payloads: dict[str, bytes]) -> None:
    path.write_bytes(
        "".join(f"{_sha256(payload)}  {name}\n" for name, payload in payloads.items()).encode(
            "ascii"
        )
    )


def _release(
    *,
    draft: bool,
    assets: list[dict[str, Any]],
    immutable: bool = False,
    prerelease: bool = False,
) -> dict[str, Any]:
    return {
        "id": RELEASE_ID,
        "tag_name": TAG,
        "draft": draft,
        "prerelease": prerelease,
        "immutable": immutable,
        "assets": assets,
    }


def _asset_record(
    asset_id: int,
    name: str,
    payload: bytes,
    *,
    state: str = "uploaded",
) -> dict[str, Any]:
    return {
        "id": asset_id,
        "name": name,
        "state": state,
        "size": len(payload) if state == "uploaded" else 0,
        "digest": f"sha256:{_sha256(payload)}" if state == "uploaded" else None,
    }


def _setup_assets(root: Path) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    payloads = {
        "gpuwm-linux-x86_64.zip": b"linux bridge bytes\n",
        "gpuwm-win-x86_64.zip": b"windows bridge bytes\n",
        "bridge-bundle-manifest.json": b'{"schema":"fixture"}\n',
    }
    asset_dir = root / "dist-assets"
    proof_dir = root / "release-proof"
    asset_dir.mkdir(exist_ok=True)
    proof_dir.mkdir(exist_ok=True)
    for name, payload in payloads.items():
        (asset_dir / name).write_bytes(payload)
    _write_sums(proof_dir / "release-assets-SHA256SUMS", payloads)
    records = [
        _asset_record(index, name, payload)
        for index, (name, payload) in enumerate(payloads.items(), start=1)
    ]
    return payloads, records


def _setup_dists(root: Path) -> dict[str, bytes]:
    # Both projects of the split: gpuwm==X.Y.Z hard-pins gpuwm-data==X.Y.Z,
    # so the one python-dists artifact carries both pairs.
    payloads = {
        f"gpuwm-{VERSION}-py3-none-any.whl": b"wheel bytes\n",
        f"gpuwm-{VERSION}.tar.gz": b"sdist bytes\n",
        f"gpuwm_data-{VERSION}-py3-none-any.whl": b"data wheel bytes\n",
        f"gpuwm_data-{VERSION}.tar.gz": b"data sdist bytes\n",
    }
    dist_dir = root / "dist"
    proof_dir = root / "release-proof"
    dist_dir.mkdir(exist_ok=True)
    proof_dir.mkdir(exist_ok=True)
    for name, payload in payloads.items():
        (dist_dir / name).write_bytes(payload)
    _write_sums(proof_dir / "python-dists-SHA256SUMS", payloads)
    return payloads


def _project_dists(dists: dict[str, bytes], project: str) -> dict[str, bytes]:
    """The subset of distribution files belonging to one PyPI project."""

    prefix = project.replace("-", "_") + "-"
    return {name: payload for name, payload in dists.items()
            if name.startswith(prefix)}


def _pypi_urls(root: Path) -> list[str]:
    """Every PyPI URL the step under test actually requested, in order."""

    return [
        arg
        for call in _logged_calls(root / "curl.log")
        for arg in call
        if arg.startswith("http") and "pypi.org" in arg
    ]


def _pypi_projects_called(root: Path) -> list[str]:
    return [url.split("/pypi/", 1)[1].split("/", 1)[0]
            for url in _pypi_urls(root)]


def _pypi_map_env(
    root: Path, responses: dict[str, tuple[str, list[dict[str, Any]]]]
) -> dict[str, str]:
    """Env for the URL-keyed fake PyPI: {project: (status, urls)}."""

    mapping: dict[str, dict[str, str]] = {}
    for project, (status, urls) in responses.items():
        file_name = f"pypi-map-{project}.json"
        body: dict[str, Any] = (
            {"urls": urls} if status == "200" else {"message": "Not Found"}
        )
        _write_json(root / file_name, body)
        mapping[project] = {"status": status, "file": file_name}
    _write_json(root / "pypi-response-map.json", mapping)
    return {"PYPI_RESPONSE_MAP": "pypi-response-map.json"}


def _pypi_file(name: str, payload: bytes) -> dict[str, Any]:
    return {
        "filename": name,
        "size": len(payload),
        "digests": {"sha256": _sha256(payload)},
        "yanked": False,
    }


def _outputs(root: Path) -> dict[str, str]:
    output = root / "github-output.txt"
    if not output.exists():
        return {}
    return dict(line.split("=", 1) for line in output.read_text().splitlines())


def _logged_calls(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _capture(tmp_path: Path, pages: list[list[dict[str, Any]]],
             env: dict[str, str] | None = None):
    _install_fake_endpoints(tmp_path)
    _write_json(tmp_path / "release-list.json", pages)
    _write_json(tmp_path / "release-sequence.json", [])
    _write_json(tmp_path / "pypi-response.json", {"urls": []})
    return _run_bash(
        tmp_path, _step_script("capture"), _base_env(tmp_path) | (env or {})
    )


@pytest.mark.parametrize(
    ("pages", "message"),
    (
        ([], "exactly one authenticated release must carry tag"),
        (
            [[_release(draft=True, assets=[]),
              _release(draft=True, assets=[]) | {"id": 43}]],
            "exactly one authenticated release must carry tag",
        ),
        (
            [[_release(draft=False, assets=[])]],
            "must have draft=true for a workflow_dispatch cut",
        ),
    ),
)
def test_paginated_capture_refuses_anything_but_one_release_in_state(
    tmp_path: Path, pages: list[list[dict[str, Any]]], message: str
) -> None:
    """Integrity, and it stays a refusal.

    Two releases on one tag means the cut cannot know which one it is
    publishing, and a release in the wrong state for this ingress means
    the motion is not the one the trigger claimed.
    """

    result = _capture(tmp_path, pages)

    assert result.returncode != 0
    assert message in result.stderr


def test_paginated_capture_resolves_one_draft_across_pages(
    tmp_path: Path,
) -> None:
    result = _capture(
        tmp_path,
        [
            [_release(draft=False, assets=[]) | {"id": 7, "tag_name": "v1.0.0"}],
            [_release(draft=True, assets=[])],
        ],
    )

    assert result.returncode == 0, result.stderr
    assert _outputs(tmp_path) == {
        "tag": TAG,
        "release_id": str(RELEASE_ID),
        "commit": COMMIT_SHA,
        "prerelease": "false",
    }


def test_a_prerelease_warns_and_is_captured_rather_than_refused(
    tmp_path: Path,
) -> None:
    """Procedure, demoted to a warning by owner ruling 2026-08-03.

    "Stable versions are never GitHub prereleases" is a convention the
    operator owns, not a property of these bytes.  What the cut does owe
    is honesty about the state it saw, so the value is captured and every
    later job re-proves it.
    """

    result = _capture(
        tmp_path, [[_release(draft=True, assets=[], prerelease=True)]]
    )

    assert result.returncode == 0, result.stderr
    assert "::warning::release v9.8.7 is marked prerelease" in result.stdout
    assert _outputs(tmp_path)["prerelease"] == "true"


def test_the_release_event_ingress_captures_an_already_public_release(
    tmp_path: Path,
) -> None:
    """Publishing a GitHub release is a cut motion again.

    That trigger fires on a release that is already public, so the state
    this ingress expects is the opposite of a dispatch's -- and a draft
    under this trigger is the mismatch, not the norm.
    """

    published = [[_release(draft=False, assets=[])]]
    event = {"EXPECTED_DRAFT": "false", "GITHUB_EVENT_NAME": "release"}

    result = _capture(tmp_path, published, event)
    assert result.returncode == 0, result.stderr
    assert _outputs(tmp_path)["release_id"] == str(RELEASE_ID)

    (tmp_path / "github-output.txt").unlink()
    refusal = _capture(tmp_path, [[_release(draft=True, assets=[])]], event)
    assert refusal.returncode != 0
    assert "must have draft=false for a release cut" in refusal.stderr


@pytest.mark.parametrize(
    ("version", "stable_expected", "succeeds"),
    (
        ("9.8.7", "", True),
        ("9.8.7", "true", True),
        # The demotion: unticked is the default, and a non-stable version
        # publishes with a warning naming it.  Reverting to the old
        # unconditional refusal fails this case.
        ("9.8.7rc1", "", True),
        # ...and the opt-in still enforces exactly what it used to.  Deleting
        # the refusal inside the branch fails this one.
        ("9.8.7rc1", "true", False),
    ),
)
def test_the_stable_version_requirement_is_opt_in(
    tmp_path: Path, version: str, stable_expected: str, succeeds: bool
) -> None:
    """Whether a public cut must carry a stable X.Y.Z is the operator's
    call, not this workflow's -- owner ruling 2026-08-03, following the
    opt-in idiom release immutability already uses.

    The tag/version equality beside it is untouched and still refuses: a
    bundle named for one while the wheel says the other is a download
    nobody can find.
    """

    _install_fake_endpoints(tmp_path)
    tag = f"v{version}"
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "gpuwm"\nversion = "{version}"\n', encoding="utf-8"
    )
    env = _base_env(tmp_path) | {
        "DISPATCH_RELEASE_TAG": tag,
        "GITHUB_REF_NAME": tag,
        "GITHUB_REF": f"refs/tags/{tag}",
        "STABLE_RELEASE_EXPECTED": stable_expected,
    }

    result = _run_bash(tmp_path, _step_script("resolve"), env)

    assert (result.returncode == 0) is succeeds, result.stderr or result.stdout
    if succeeds:
        assert _outputs(tmp_path) == {
            "tag": tag,
            "release_id": str(RELEASE_ID),
            "version": version,
            "commit": COMMIT_SHA,
        }
        assert (
            f"::warning::publishing non-stable version {version}"
            in result.stdout
        ) is (version != "9.8.7")
    else:
        assert "is not a stable X.Y.Z" in result.stderr


def test_the_cut_still_refuses_a_tag_that_disagrees_with_the_version(
    tmp_path: Path,
) -> None:
    """The demotion above is scoped to the shape of the version string."""

    _install_fake_endpoints(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "gpuwm"\nversion = "9.8.6"\n', encoding="utf-8"
    )

    result = _run_bash(tmp_path, _step_script("resolve"), _base_env(tmp_path))

    assert result.returncode != 0
    assert "does not match pyproject version" in result.stderr


def test_draft_asset_capture_downloads_only_uploaded_assets(tmp_path: Path) -> None:
    """The re-run recovery path starts here: the draft-visibility cell
    captures what a previous run already uploaded, so `prepare` can adopt
    those exact bytes by content contract instead of stranding on the
    byte-identity refusal against a non-reproducible rebuild."""

    _install_fake_endpoints(tmp_path)
    payload = b"first run linux bundle bytes\n"
    release = _release(
        draft=True,
        assets=[
            _asset_record(5, "gpuwm-linux-x86_64.zip", payload),
            _asset_record(6, "gpuwm-win-x86_64.zip", b"", state="starter"),
        ],
    )
    _write_json(tmp_path / "release-sequence.json", [release])
    _write_json(tmp_path / "release-list.json", [[release]])
    _write_json(tmp_path / "pypi-response.json", {"urls": []})
    asset_dir = tmp_path / "fake-assets"
    asset_dir.mkdir()
    (asset_dir / "5").write_bytes(payload)
    env = _base_env(tmp_path) | {"FAKE_ASSET_DIR": "fake-assets"}

    result = _run_bash(
        tmp_path,
        _step_script("capture the assets the draft already carries"),
        env,
    )

    assert result.returncode == 0, result.stderr
    held = tmp_path / "draft-held"
    assert (held / "assets" / "gpuwm-linux-x86_64.zip").read_bytes() == payload
    assert not (held / "assets" / "gpuwm-win-x86_64.zip").exists()
    inventory = json.loads(
        (held / "inventory.json").read_text(encoding="utf-8")
    )
    assert {entry["name"]: entry["state"] for entry in inventory} == {
        "gpuwm-linux-x86_64.zip": "uploaded",
        "gpuwm-win-x86_64.zip": "starter",
    }


def test_draft_asset_capture_refuses_a_download_that_misses_its_digest(
    tmp_path: Path,
) -> None:
    _install_fake_endpoints(tmp_path)
    payload = b"first run linux bundle bytes\n"
    release = _release(
        draft=True,
        assets=[_asset_record(5, "gpuwm-linux-x86_64.zip", payload)],
    )
    _write_json(tmp_path / "release-sequence.json", [release])
    _write_json(tmp_path / "release-list.json", [[release]])
    _write_json(tmp_path / "pypi-response.json", {"urls": []})
    asset_dir = tmp_path / "fake-assets"
    asset_dir.mkdir()
    (asset_dir / "5").write_bytes(b"corrupted transfer bytes\n")
    env = _base_env(tmp_path) | {"FAKE_ASSET_DIR": "fake-assets"}

    result = _run_bash(
        tmp_path,
        _step_script("capture the assets the draft already carries"),
        env,
    )

    assert result.returncode != 0
    assert "does not match its recorded size and digest" in result.stderr


def test_assets_upload_bundles_first_and_the_manifest_last(tmp_path: Path) -> None:
    """An interrupted upload must never leave a manifest on the draft
    that pins bundle bytes the draft does not hold -- that is the one
    partial state a re-run can never adopt."""

    _install_fake_endpoints(tmp_path)
    _, exact = _setup_assets(tmp_path)
    empty = _release(draft=True, assets=[])
    final = _release(draft=True, assets=exact)
    _write_json(tmp_path / "release-sequence.json", [empty, final])
    _write_json(tmp_path / "release-list.json", [[empty]])
    _write_json(tmp_path / "release-list-sequence.json", [[[empty]], [[final]]])
    _write_json(tmp_path / "pypi-response.json", {"urls": []})
    env = _base_env(tmp_path) | {
        "GH_RELEASE_LIST_SEQUENCE": "release-list-sequence.json",
        "GH_RELEASE_LIST_COUNTER": "gh-list-counter.txt",
    }

    result = _run_bash(
        tmp_path,
        _step_script("reconcile immutable assets by captured release id"),
        env,
    )

    assert result.returncode == 0, result.stderr
    uploads = [
        parse_qs(urlsplit(arg).query)["name"][0]
        for call in _logged_calls(tmp_path / "curl.log")
        for arg in call
        if arg.startswith("https://uploads.github.com/")
    ]
    assert uploads == [
        "gpuwm-linux-x86_64.zip",
        "gpuwm-win-x86_64.zip",
        "bridge-bundle-manifest.json",
    ]


def test_prepare_adopts_the_drafts_assets_before_pinning() -> None:
    """The adopted bytes must reach the pins before the wheel is built:
    only then do the assets job's byte-identity reconcile, the PyPI wheel's
    bundle pins, and the uploaded zips all describe the same bytes."""

    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count("name: draft-held-assets") == 2  # captured, consumed
    adopt = text.index("build_bridge_bundle.py adopt")
    pin = text.index("build_bridge_bundle.py pin")
    assert text.index("name: draft-held-assets") < adopt < pin


def test_assets_reuse_exact_delete_only_starter_and_upload_missing(tmp_path: Path) -> None:
    _install_fake_endpoints(tmp_path)
    payloads, exact = _setup_assets(tmp_path)
    initial = _release(
        draft=True,
        assets=[
            exact[0],
            _asset_record(2, "gpuwm-win-x86_64.zip", payloads["gpuwm-win-x86_64.zip"], state="starter"),
        ],
    )
    final = _release(draft=True, assets=exact)
    _write_json(tmp_path / "release-sequence.json", [initial, final])
    _write_json(tmp_path / "release-list.json", [[initial]])
    _write_json(tmp_path / "release-list-sequence.json", [[[initial]], [[final]]])
    _write_json(tmp_path / "pypi-response.json", {"urls": []})
    env = _base_env(tmp_path) | {
        "GH_RELEASE_LIST_SEQUENCE": "release-list-sequence.json",
        "GH_RELEASE_LIST_COUNTER": "gh-list-counter.txt",
    }

    result = _run_bash(
        tmp_path,
        _step_script("reconcile immutable assets by captured release id"),
        env,
    )

    assert result.returncode == 0, result.stderr
    gh_calls = _logged_calls(tmp_path / "gh.log")
    deleted = [call for call in gh_calls if "DELETE" in call]
    assert len(deleted) == 1
    assert any(arg.endswith("/releases/assets/2") for arg in deleted[0])
    curl_calls = _logged_calls(tmp_path / "curl.log")
    uploaded = {
        parse_qs(urlsplit(arg).query)["name"][0]
        for call in curl_calls
        for arg in call
        if arg.startswith("https://uploads.github.com/")
    }
    assert uploaded == {
        "gpuwm-win-x86_64.zip",
        "bridge-bundle-manifest.json",
    }


@pytest.mark.parametrize("mismatch", ("size", "digest"))
def test_assets_refuse_changed_uploaded_bytes(tmp_path: Path, mismatch: str) -> None:
    _install_fake_endpoints(tmp_path)
    _, exact = _setup_assets(tmp_path)
    changed = dict(exact[0])
    if mismatch == "size":
        changed["size"] += 1
    else:
        changed["digest"] = f"sha256:{'0' * 64}"
    release = _release(draft=True, assets=[changed])
    _write_json(tmp_path / "release-sequence.json", [release])
    _write_json(tmp_path / "release-list.json", [[release]])
    _write_json(tmp_path / "pypi-response.json", {"urls": []})

    result = _run_bash(
        tmp_path,
        _step_script("reconcile immutable assets by captured release id"),
        _base_env(tmp_path),
    )

    assert result.returncode != 0
    assert "refusing changed bytes for immutable release asset" in result.stderr
    assert not _logged_calls(tmp_path / "curl.log")


#: The two reconcile steps of the publish job: one per PyPI project, each
#: staging into its own upload directory so neither can push the other's
#: bytes.  gpuwm-data goes first because gpuwm==X.Y.Z pins
#: gpuwm-data==X.Y.Z: a gpuwm that lands without its companion breaks
#: every fresh `pip install gpuwm` until the companion exists.
_RECONCILE_STEPS = (
    pytest.param(
        "reconcile exact PyPI distribution state",
        "gpuwm",
        "dist-upload",
        id="gpuwm",
    ),
    pytest.param(
        "reconcile exact PyPI distribution state for gpuwm-data",
        "gpuwm-data",
        "dist-upload-data",
        id="gpuwm-data",
    ),
)


@pytest.mark.parametrize(("step", "project", "upload_dir"), _RECONCILE_STEPS)
@pytest.mark.parametrize(
    ("remote_count", "http_status", "expected_missing", "upload_required"),
    (
        (0, "404", 2, "true"),
        (1, "200", 1, "true"),
        (2, "200", 0, "false"),
    ),
)
def test_pypi_absent_partial_and_exact_states_choose_only_missing_files(
    tmp_path: Path,
    step: str,
    project: str,
    upload_dir: str,
    remote_count: int,
    http_status: str,
    expected_missing: int,
    upload_required: str,
) -> None:
    """Absent (404), partial, and exact states stage only the missing files.

    The 404 arm is the pending-publisher flow for gpuwm-data: the project
    does not exist on PyPI until this workflow's first upload creates it,
    so "not found" means "everything is missing", never an error.  Each
    step must also touch only its own project's files -- staging the
    sibling's bytes would upload them under the wrong project name.
    """

    _install_fake_endpoints(tmp_path)
    _, asset_records = _setup_assets(tmp_path)
    dists = _setup_dists(tmp_path)
    own = _project_dists(dists, project)
    remote = [_pypi_file(name, payload) for name, payload in own.items()][
        :remote_count
    ]
    release = _release(draft=True, assets=asset_records)
    _write_json(tmp_path / "release-sequence.json", [release])
    _write_json(tmp_path / "release-list.json", [[release]])
    _write_json(tmp_path / "pypi-response.json", {"urls": remote})
    env = _base_env(tmp_path) | {"PYPI_HTTP_STATUS": http_status}

    result = _run_bash(tmp_path, _step_script(step), env)

    assert result.returncode == 0, result.stderr
    assert _outputs(tmp_path) == {
        "upload_required": upload_required,
        "missing_count": str(expected_missing),
    }
    staged = (
        {path.name for path in (tmp_path / upload_dir).iterdir()}
        if (tmp_path / upload_dir).exists()
        else set()
    )
    assert staged == set(list(own)[remote_count:])
    assert _pypi_projects_called(tmp_path) == [project]


@pytest.mark.parametrize(("step", "project", "upload_dir"), _RECONCILE_STEPS)
@pytest.mark.parametrize("mismatch", ("size", "digest", "yanked", "unexpected"))
def test_pypi_mismatch_is_fail_closed(
    tmp_path: Path, step: str, project: str, upload_dir: str, mismatch: str
) -> None:
    _install_fake_endpoints(tmp_path)
    _, asset_records = _setup_assets(tmp_path)
    dists = _setup_dists(tmp_path)
    own = _project_dists(dists, project)
    name, payload = next(iter(own.items()))
    remote = _pypi_file(name, payload)
    if mismatch == "size":
        remote["size"] += 1
    elif mismatch == "digest":
        remote["digests"]["sha256"] = "0" * 64
    elif mismatch == "yanked":
        remote["yanked"] = True
    else:
        prefix = project.replace("-", "_")
        remote["filename"] = f"{prefix}-{VERSION}-cp999-none-any.whl"
    release = _release(draft=True, assets=asset_records)
    _write_json(tmp_path / "release-sequence.json", [release])
    _write_json(tmp_path / "release-list.json", [[release]])
    _write_json(tmp_path / "pypi-response.json", {"urls": [remote]})

    result = _run_bash(tmp_path, _step_script(step), _base_env(tmp_path))

    assert result.returncode != 0
    expected = (
        "unexpected distribution"
        if mismatch == "unexpected"
        else "differs from the proven artifact"
    )
    assert expected in result.stderr


def _pypi_curl_calls(root: Path) -> list[list[str]]:
    return [
        call
        for call in _logged_calls(root / "curl.log")
        if any("pypi.org" in arg for arg in call)
    ]


def _post_upload_proof_env(root: Path, sequence: list[dict[str, str]]) -> dict[str, str]:
    """Environment for the post-upload proof with a scripted index sequence.

    The retry budget is shrunk so an exhausted budget costs seconds, not the
    production five minutes; the assertions under test are status-count and
    exit semantics, which are budget-size independent.
    """

    _write_json(root / "pypi-response-sequence.json", sequence)
    return _base_env(root) | {
        "PYPI_RESPONSE_SEQUENCE": "pypi-response-sequence.json",
        "PYPI_RESPONSE_COUNTER": "pypi-counter.txt",
        "PYPI_INDEX_RETRY_BUDGET_SECONDS": "3",
        "PYPI_INDEX_RETRY_INITIAL_DELAY_SECONDS": "1",
    }


#: The two post-upload exact-state proofs of the publish job.  The
#: gpuwm-data one runs BETWEEN the two uploads: it is the hard gate that
#: makes gpuwm unreachable until PyPI provably lists the companion.
_EXACT_PROOF_STEPS = (
    pytest.param(
        "prove exact PyPI state before release promotion",
        "gpuwm",
        "PyPI exact-state proof",
        id="gpuwm",
    ),
    pytest.param(
        "prove exact gpuwm-data PyPI state before gpuwm publishes",
        "gpuwm-data",
        "gpuwm-data exact-state proof",
        id="gpuwm-data",
    ),
)


@pytest.mark.parametrize(("step", "project", "label"), _EXACT_PROOF_STEPS)
def test_post_upload_proof_survives_index_lag_404_then_200(
    tmp_path: Path, step: str, project: str, label: str
) -> None:
    """A 200 OK upload can precede the JSON index listing by a beat.

    The proof must poll the index read again instead of failing the whole
    publish job, and the exact-state assertions must then run unchanged
    against the eventually-consistent body.
    """

    _install_fake_endpoints(tmp_path)
    dists = _project_dists(_setup_dists(tmp_path), project)
    _write_json(tmp_path / "pypi-lagged.json", {"message": "Not Found"})
    _write_json(
        tmp_path / "pypi-exact.json",
        {"urls": [_pypi_file(name, payload) for name, payload in dists.items()]},
    )
    env = _post_upload_proof_env(
        tmp_path,
        [
            {"status": "404", "file": "pypi-lagged.json"},
            {"status": "200", "file": "pypi-exact.json"},
        ],
    )

    result = _run_bash(tmp_path, _step_script(step), env)

    assert result.returncode == 0, result.stderr
    assert len(_pypi_curl_calls(tmp_path)) == 2
    assert _pypi_projects_called(tmp_path) == [project, project]


@pytest.mark.parametrize(("step", "project", "label"), _EXACT_PROOF_STEPS)
def test_post_upload_proof_permanent_404_still_fails_after_budget(
    tmp_path: Path, step: str, project: str, label: str
) -> None:
    """Retry is bounded: a version the index never lists remains a failure.

    For gpuwm-data this is the ordering guarantee itself: its proof step
    failing hard means the job dies with gpuwm still unpublished, so gpuwm
    can never land on PyPI without the companion its pin requires.
    """

    _install_fake_endpoints(tmp_path)
    _setup_dists(tmp_path)
    _write_json(tmp_path / "pypi-lagged.json", {"message": "Not Found"})
    env = _post_upload_proof_env(
        tmp_path, [{"status": "404", "file": "pypi-lagged.json"}]
    )

    result = _run_bash(tmp_path, _step_script(step), env)

    assert result.returncode != 0
    assert f"{label} returned HTTP 404" in result.stderr
    # It genuinely retried before failing closed.
    assert len(_pypi_curl_calls(tmp_path)) >= 2


@pytest.mark.parametrize(("step", "project", "label"), _EXACT_PROOF_STEPS)
def test_post_upload_proof_non_lag_status_fails_immediately(
    tmp_path: Path, step: str, project: str, label: str
) -> None:
    """Only the 404 lag signature is retried; a 503 keeps single-shot failure."""

    _install_fake_endpoints(tmp_path)
    _setup_dists(tmp_path)
    _write_json(tmp_path / "pypi-error.json", {"message": "Service Unavailable"})
    env = _post_upload_proof_env(
        tmp_path, [{"status": "503", "file": "pypi-error.json"}]
    )

    result = _run_bash(tmp_path, _step_script(step), env)

    assert result.returncode != 0
    assert f"{label} returned HTTP 503" in result.stderr
    assert len(_pypi_curl_calls(tmp_path)) == 1


@pytest.mark.parametrize(("step", "project", "label"), _EXACT_PROOF_STEPS)
def test_post_upload_proof_retry_never_wraps_the_content_assertions(
    tmp_path: Path, step: str, project: str, label: str
) -> None:
    """Retry covers the index read only, never the exact-state checks.

    The first response is HTTP 200 with a wrong digest and the second would
    match exactly, so a retry loop wrongly wrapped around the assertions
    would convert this refusal into a pass.  It must fail on the first body.
    """

    _install_fake_endpoints(tmp_path)
    dists = _project_dists(_setup_dists(tmp_path), project)
    exact = [_pypi_file(name, payload) for name, payload in dists.items()]
    mismatched = [dict(entry) for entry in exact]
    mismatched[0]["digests"] = {"sha256": "0" * 64}
    _write_json(tmp_path / "pypi-mismatch.json", {"urls": mismatched})
    _write_json(tmp_path / "pypi-exact.json", {"urls": exact})
    env = _post_upload_proof_env(
        tmp_path,
        [
            {"status": "200", "file": "pypi-mismatch.json"},
            {"status": "200", "file": "pypi-exact.json"},
        ],
    )

    result = _run_bash(tmp_path, _step_script(step), env)

    assert result.returncode != 0
    assert f"{label} failed for" in result.stderr
    assert len(_pypi_curl_calls(tmp_path)) == 1


def test_upload_action_is_conditioned_on_the_reconciliation_decision() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "if: steps.pypi.outputs.upload_required == 'true'" in text
    assert "packages-dir: dist-upload/" in text
    assert "if: steps.pypi_data.outputs.upload_required == 'true'" in text
    assert "packages-dir: dist-upload-data/" in text


def test_gpuwm_data_publishes_first_and_hard_gates_gpuwm() -> None:
    """The companion uploads and is proven BEFORE gpuwm touches PyPI.

    gpuwm==X.Y.Z hard-pins gpuwm-data==X.Y.Z.  If gpuwm lands first and
    anything then fails, every fresh `pip install gpuwm` is broken until
    the companion exists -- and gpuwm-data is a pending trusted publisher,
    so its very first upload happens inside this ordering.  The reverse
    failure mode (companion live, gpuwm absent) breaks nobody.  Both
    uploads use the same OIDC publisher action, and no step in the job may
    soften a failure with continue-on-error.
    """

    # Sliced from the jobs region: `release:` is also a TRIGGER key in the
    # `on:` block, so anchoring from the top of the file finds that one.
    jobs = WORKFLOW.read_text(encoding="utf-8")
    jobs = jobs[jobs.index("\njobs:"):]
    job = jobs[jobs.index("\n  publish:"): jobs.index("\n  release:")]
    marks = [
        "- name: reconcile exact PyPI distribution state for gpuwm-data",
        "packages-dir: dist-upload-data/",
        "- name: prove exact gpuwm-data PyPI state before gpuwm publishes",
        "- name: reconcile exact PyPI distribution state\n",
        "packages-dir: dist-upload/\n",
        "- name: prove exact PyPI state before release promotion",
    ]
    positions = [job.index(mark) for mark in marks]
    assert positions == sorted(positions), (
        "the publish job no longer uploads and proves gpuwm-data before "
        "gpuwm")
    assert job.count("pypa/gh-action-pypi-publish@") == 2
    assert "PYPI_DATA_PROJECT: gpuwm-data" in job
    assert "continue-on-error" not in job


def _exact_pypi_map_env(root: Path, dists: dict[str, bytes]) -> dict[str, str]:
    """URL-keyed fake PyPI where both projects list their exact dists."""

    return _pypi_map_env(
        root,
        {
            project: (
                "200",
                [
                    _pypi_file(name, payload)
                    for name, payload in _project_dists(dists, project).items()
                ],
            )
            for project in ("gpuwm", "gpuwm-data")
        },
    )


def test_exact_draft_promotes_once_to_immutable_public_release(tmp_path: Path) -> None:
    _install_fake_endpoints(tmp_path)
    _, asset_records = _setup_assets(tmp_path)
    dists = _setup_dists(tmp_path)
    draft = _release(draft=True, assets=asset_records)
    public = _release(draft=False, assets=asset_records, immutable=True)
    _write_json(tmp_path / "release-sequence.json", [draft])
    _write_json(tmp_path / "release-list.json", [[draft]])
    _write_json(tmp_path / "release-list-sequence.json", [[[draft]], [[public]]])
    env = _base_env(tmp_path) | _exact_pypi_map_env(tmp_path, dists) | {
        "GH_RELEASE_LIST_SEQUENCE": "release-list-sequence.json",
        "GH_RELEASE_LIST_COUNTER": "gh-list-counter.txt",
    }

    result = _run_bash(
        tmp_path,
        _step_script("publish the fully-proven draft"),
        env,
    )

    assert result.returncode == 0, result.stderr
    patch_calls = [
        call for call in _logged_calls(tmp_path / "gh.log") if "PATCH" in call
    ]
    assert len(patch_calls) == 1
    assert "draft=false" in patch_calls[0]
    assert "prerelease=false" in patch_calls[0]
    # Both PyPI projects are proven, companion first, before AND after the
    # promotion patch: the split's promise is that a public release always
    # stands on a resolvable pair.
    assert _pypi_projects_called(tmp_path) == [
        "gpuwm-data", "gpuwm", "gpuwm-data", "gpuwm",
    ]


def test_promotion_carries_the_captured_prerelease_state_unchanged(
    tmp_path: Path,
) -> None:
    """A prerelease publishes as a prerelease.

    With the non-prerelease precondition demoted to a warning, forcing
    ``prerelease=false`` at promotion would publish as stable something
    the operator deliberately marked otherwise -- a silent state change,
    which is worse than either refusing or preserving.  Restoring the
    literal ``-F prerelease=false`` fails this test.
    """

    _install_fake_endpoints(tmp_path)
    _, asset_records = _setup_assets(tmp_path)
    dists = _setup_dists(tmp_path)
    draft = _release(draft=True, assets=asset_records, prerelease=True)
    public = _release(
        draft=False, assets=asset_records, immutable=True, prerelease=True
    )
    _write_json(tmp_path / "release-sequence.json", [draft])
    _write_json(tmp_path / "release-list.json", [[draft]])
    _write_json(tmp_path / "release-list-sequence.json", [[[draft]], [[public]]])
    env = _base_env(tmp_path) | _exact_pypi_map_env(tmp_path, dists) | {
        "CAPTURED_PRERELEASE": "true",
        "GH_RELEASE_LIST_SEQUENCE": "release-list-sequence.json",
        "GH_RELEASE_LIST_COUNTER": "gh-list-counter.txt",
    }

    result = _run_bash(
        tmp_path,
        _step_script("publish the fully-proven draft"),
        env,
    )

    assert result.returncode == 0, result.stderr
    patch_calls = [
        call for call in _logged_calls(tmp_path / "gh.log") if "PATCH" in call
    ]
    assert len(patch_calls) == 1
    assert "draft=false" in patch_calls[0]
    assert "prerelease=true" in patch_calls[0]


def test_promotion_refuses_a_prerelease_flag_that_moved_since_capture(
    tmp_path: Path,
) -> None:
    """Demoting the precondition does not demote the re-proof: the state
    still has to be the one release_authority captured."""

    _install_fake_endpoints(tmp_path)
    _, asset_records = _setup_assets(tmp_path)
    dists = _setup_dists(tmp_path)
    draft = _release(draft=True, assets=asset_records, prerelease=True)
    _write_json(tmp_path / "release-sequence.json", [draft])
    _write_json(tmp_path / "release-list.json", [[draft]])

    result = _run_bash(
        tmp_path,
        _step_script("publish the fully-proven draft"),
        _base_env(tmp_path)
        | _exact_pypi_map_env(tmp_path, dists)
        | {"CAPTURED_PRERELEASE": "false"},
    )

    assert result.returncode != 0
    assert "authority changed before promotion" in result.stderr


def test_promotion_refuses_asset_changed_while_draft_was_mutable(tmp_path: Path) -> None:
    _install_fake_endpoints(tmp_path)
    _, asset_records = _setup_assets(tmp_path)
    dists = _setup_dists(tmp_path)
    draft = _release(draft=True, assets=asset_records)
    changed_assets = [dict(asset) for asset in asset_records]
    changed_assets[0]["size"] += 1
    public = _release(draft=False, assets=changed_assets, immutable=True)
    _write_json(tmp_path / "release-sequence.json", [draft])
    _write_json(tmp_path / "release-list.json", [[draft]])
    _write_json(tmp_path / "release-list-sequence.json", [[[draft]], [[public]]])
    env = _base_env(tmp_path) | _exact_pypi_map_env(tmp_path, dists) | {
        "GH_RELEASE_LIST_SEQUENCE": "release-list-sequence.json",
        "GH_RELEASE_LIST_COUNTER": "gh-list-counter.txt",
    }

    result = _run_bash(
        tmp_path,
        _step_script("publish the fully-proven draft"),
        env,
    )

    assert result.returncode != 0
    assert "immutable release asset changed during promotion" in result.stderr


def test_promotion_pypi_proof_survives_index_lag_404_then_200(tmp_path: Path) -> None:
    """The final-promotion index read absorbs the same lag the publish job does.

    Measured live: uploads at T+0, this read 404 at T+59s while the publish
    job's own retry loop had logged identical 404s before succeeding.  A
    fully green cut must not halt at promotion on that propagation window.
    """

    _install_fake_endpoints(tmp_path)
    _, asset_records = _setup_assets(tmp_path)
    dists = _setup_dists(tmp_path)
    draft = _release(draft=True, assets=asset_records)
    public = _release(draft=False, assets=asset_records, immutable=True)
    _write_json(tmp_path / "release-sequence.json", [draft])
    _write_json(tmp_path / "release-list.json", [[draft]])
    _write_json(tmp_path / "release-list-sequence.json", [[[draft]], [[public]]])
    _write_json(tmp_path / "pypi-lagged.json", {"message": "Not Found"})
    _write_json(
        tmp_path / "pypi-data-exact.json",
        {"urls": [
            _pypi_file(name, payload)
            for name, payload in _project_dists(dists, "gpuwm-data").items()
        ]},
    )
    _write_json(
        tmp_path / "pypi-exact.json",
        {"urls": [
            _pypi_file(name, payload)
            for name, payload in _project_dists(dists, "gpuwm").items()
        ]},
    )
    # Call-ordered: the companion read lags once and is retried, then the
    # gpuwm read, then the two post-promotion re-proofs.
    env = _post_upload_proof_env(
        tmp_path,
        [
            {"status": "404", "file": "pypi-lagged.json"},
            {"status": "200", "file": "pypi-data-exact.json"},
            {"status": "200", "file": "pypi-exact.json"},
            {"status": "200", "file": "pypi-data-exact.json"},
            {"status": "200", "file": "pypi-exact.json"},
        ],
    ) | {
        "GH_RELEASE_LIST_SEQUENCE": "release-list-sequence.json",
        "GH_RELEASE_LIST_COUNTER": "gh-list-counter.txt",
    }

    result = _run_bash(
        tmp_path,
        _step_script("publish the fully-proven draft"),
        env,
    )

    assert result.returncode == 0, result.stderr
    # The lagged companion read, its retry, the gpuwm read, and the two
    # post-promotion re-proofs -- companion first each time.
    assert len(_pypi_curl_calls(tmp_path)) == 5
    assert _pypi_projects_called(tmp_path) == [
        "gpuwm-data", "gpuwm-data", "gpuwm", "gpuwm-data", "gpuwm",
    ]
    patch_calls = [
        call for call in _logged_calls(tmp_path / "gh.log") if "PATCH" in call
    ]
    assert len(patch_calls) == 1


def test_promotion_pypi_proof_permanent_404_still_fails_after_budget(
    tmp_path: Path,
) -> None:
    """Retry is bounded: an index that never lists the version still refuses,
    and nothing is promoted."""

    _install_fake_endpoints(tmp_path)
    _, asset_records = _setup_assets(tmp_path)
    _setup_dists(tmp_path)
    draft = _release(draft=True, assets=asset_records)
    _write_json(tmp_path / "release-sequence.json", [draft])
    _write_json(tmp_path / "release-list.json", [[draft]])
    _write_json(tmp_path / "pypi-lagged.json", {"message": "Not Found"})
    env = _post_upload_proof_env(
        tmp_path, [{"status": "404", "file": "pypi-lagged.json"}]
    )

    result = _run_bash(
        tmp_path,
        _step_script("publish the fully-proven draft"),
        env,
    )

    assert result.returncode != 0
    assert "final promotion PyPI artifact set changed" in result.stderr
    # It genuinely retried before failing closed -- and it was the
    # companion's read that gated: gpuwm was never even consulted.
    assert len(_pypi_curl_calls(tmp_path)) >= 2
    assert set(_pypi_projects_called(tmp_path)) == {"gpuwm-data"}
    assert not any("PATCH" in call for call in _logged_calls(tmp_path / "gh.log"))


def test_promotion_refuses_when_the_companion_is_missing_despite_exact_gpuwm(
    tmp_path: Path,
) -> None:
    """gpuwm being byte-exact on PyPI cannot rescue an absent companion.

    This is the split's failure mode in miniature: gpuwm==X.Y.Z live,
    gpuwm-data==X.Y.Z missing, every fresh `pip install gpuwm`
    unresolvable.  Promotion must refuse and leave the draft private.
    """

    _install_fake_endpoints(tmp_path)
    _, asset_records = _setup_assets(tmp_path)
    dists = _setup_dists(tmp_path)
    draft = _release(draft=True, assets=asset_records)
    _write_json(tmp_path / "release-sequence.json", [draft])
    _write_json(tmp_path / "release-list.json", [[draft]])
    env = _base_env(tmp_path) | _pypi_map_env(
        tmp_path,
        {
            "gpuwm-data": ("404", []),
            "gpuwm": ("200", [
                _pypi_file(name, payload)
                for name, payload in _project_dists(dists, "gpuwm").items()
            ]),
        },
    ) | {
        "PYPI_INDEX_RETRY_BUDGET_SECONDS": "3",
        "PYPI_INDEX_RETRY_INITIAL_DELAY_SECONDS": "1",
    }

    result = _run_bash(
        tmp_path,
        _step_script("publish the fully-proven draft"),
        env,
    )

    assert result.returncode != 0
    assert "final promotion PyPI artifact set changed" in result.stderr
    assert not any("PATCH" in call for call in _logged_calls(tmp_path / "gh.log"))


def test_already_public_same_id_retry_succeeds_without_patch(tmp_path: Path) -> None:
    _install_fake_endpoints(tmp_path)
    _, asset_records = _setup_assets(tmp_path)
    dists = _setup_dists(tmp_path)
    release = _release(draft=False, assets=asset_records, immutable=True)
    _write_json(tmp_path / "release-sequence.json", [release])
    _write_json(tmp_path / "release-list.json", [[release]])

    result = _run_bash(
        tmp_path,
        _step_script("publish the fully-proven draft"),
        _base_env(tmp_path) | _exact_pypi_map_env(tmp_path, dists),
    )

    assert result.returncode == 0, result.stderr
    assert "already public at the captured id" in result.stdout
    assert not any("PATCH" in call for call in _logged_calls(tmp_path / "gh.log"))


# --------------------------------------------------------------------------
# The prepare job's gpuwm-data distribution proof.
# --------------------------------------------------------------------------

def _stage_data_dists(
    tmp_path: Path,
    *,
    wheel_name: str | None = None,
    sdist_name: str | None = None,
    size: int = 10_000_001,
) -> None:
    """A companion pair in dist/, extended to `size` bytes each."""

    dist = tmp_path / "dist"
    dist.mkdir(exist_ok=True)
    for name in (
        wheel_name or f"gpuwm_data-{VERSION}-py3-none-any.whl",
        sdist_name or f"gpuwm_data-{VERSION}.tar.gz",
    ):
        path = dist / name
        path.write_bytes(b"\0")
        os.truncate(path, size)


def test_data_dist_proof_accepts_the_exact_cut_pair(tmp_path: Path) -> None:
    """One universal wheel and one sdist, named for the cut, sized sanely."""

    _stage_data_dists(tmp_path)

    result = _run_bash(
        tmp_path,
        _step_script("prove the gpuwm-data distributions"),
        _base_env(tmp_path),
    )

    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        pytest.param(
            {"wheel_name": "gpuwm_data-1.2.3-py3-none-any.whl"},
            "gpuwm-data wheel is not",
            id="wheel-version",
        ),
        pytest.param(
            {"wheel_name":
             f"gpuwm_data-{VERSION}-cp311-abi3-manylinux_2_28_x86_64.whl"},
            "gpuwm-data wheel is not",
            id="platform-wheel",
        ),
        pytest.param(
            {"sdist_name": "gpuwm_data-1.2.3.tar.gz"},
            "gpuwm-data sdist is not",
            id="sdist-version",
        ),
        pytest.param({"size": 4096}, "lost its payload", id="hollow"),
        pytest.param(
            {"size": 100_000_001},
            "exceeds PyPI's per-file cap",
            id="over-cap",
        ),
    ),
)
def test_data_dist_proof_refuses_the_named_defects(
    tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    """Each refusal names its breakage.

    A companion at any version but the cut's -- or wearing a platform
    tag under the pinned universal name -- publishes a gpuwm whose
    `gpuwm-data==X.Y.Z` pin no fresh install can resolve.  A hollow
    dist installs cleanly and then dies at the first radiation or
    microphysics table read.  A dist over PyPI's per-file cap dies at
    upload, at job 9 of 10, with the tag already spent -- the 2.4.0
    burn.
    """

    _stage_data_dists(tmp_path, **kwargs)

    result = _run_bash(
        tmp_path,
        _step_script("prove the gpuwm-data distributions"),
        _base_env(tmp_path),
    )

    assert result.returncode != 0
    assert message in result.stderr


def test_data_dist_proof_refuses_a_missing_half_of_the_pair(
    tmp_path: Path,
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    lone = dist / f"gpuwm_data-{VERSION}-py3-none-any.whl"
    lone.write_bytes(b"\0")
    os.truncate(lone, 10_000_001)

    result = _run_bash(
        tmp_path,
        _step_script("prove the gpuwm-data distributions"),
        _base_env(tmp_path),
    )

    assert result.returncode != 0
    assert (
        "expected exactly one gpuwm-data wheel and one gpuwm-data sdist"
        in result.stderr
    )


def test_prepare_builds_the_companion_into_the_shared_dist() -> None:
    """Both projects' dists ride ONE python-dists artifact.

    The companion is built in the same prepare job, into the same dist/
    directory, before the gpuwm-data proof step runs and before the
    artifact upload -- so every later job's re-proof covers both pairs
    and no second artifact channel exists to drift.
    """

    text = WORKFLOW.read_text(encoding="utf-8")
    prepare = text[text.index("\n  prepare:"): text.index("\n  assets:")]
    root_build = prepare.index("python -m build\n")
    data_build = prepare.index("python -m build --outdir dist gpuwm-data")
    proof = prepare.index("- name: prove the gpuwm-data distributions")
    upload = prepare.index("name: python-dists")
    assert root_build < proof and data_build < proof < upload, (
        "the prepare job no longer builds and proves the companion before "
        "uploading the python-dists artifact")
