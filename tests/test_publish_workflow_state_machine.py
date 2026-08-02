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
    fake_bin = root / "fake-bin"
    fake_bin.mkdir()
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
        if "pypi.org" in url:
            try:
                output = args[args.index("--output") + 1]
            except (ValueError, IndexError):
                output = args[args.index("-o") + 1]
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
    # Keep the workflow's control flow literal while replacing only its two
    # network command endpoints.  Explicit paths are reliable through the
    # Windows-to-WSL Bash bridge, where shell-function shims are not.
    fixture_script = script.replace(
        "gh api", f"{python} ./fake-bin/gh api"
    ).replace("curl ", f"{python} ./fake-bin/curl ")
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
    payloads = {
        f"gpuwm-{VERSION}-py3-none-any.whl": b"wheel bytes\n",
        f"gpuwm-{VERSION}.tar.gz": b"sdist bytes\n",
    }
    dist_dir = root / "dist"
    proof_dir = root / "release-proof"
    dist_dir.mkdir(exist_ok=True)
    proof_dir.mkdir(exist_ok=True)
    for name, payload in payloads.items():
        (dist_dir / name).write_bytes(payload)
    _write_sums(proof_dir / "python-dists-SHA256SUMS", payloads)
    return payloads


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


@pytest.mark.parametrize(
    ("pages", "succeeds"),
    (
        (
            [
                [_release(draft=False, assets=[]) | {"id": 7, "tag_name": "v1.0.0"}],
                [_release(draft=True, assets=[])],
            ],
            True,
        ),
        ([], False),
        ([[_release(draft=False, assets=[])]], False),
        ([[_release(draft=True, assets=[], prerelease=True)]], False),
        (
            [[_release(draft=True, assets=[]), _release(draft=True, assets=[]) | {"id": 43}]],
            False,
        ),
    ),
)
def test_paginated_draft_capture_requires_one_exact_tag_and_id(
    tmp_path: Path, pages: list[list[dict[str, Any]]], succeeds: bool
) -> None:
    _install_fake_endpoints(tmp_path)
    _write_json(tmp_path / "release-list.json", pages)
    _write_json(tmp_path / "release-sequence.json", [])
    _write_json(tmp_path / "pypi-response.json", {"urls": []})

    result = _run_bash(tmp_path, _step_script("capture"), _base_env(tmp_path))

    assert (result.returncode == 0) is succeeds, result.stderr
    if succeeds:
        assert _outputs(tmp_path) == {
            "tag": TAG,
            "release_id": str(RELEASE_ID),
            "commit": COMMIT_SHA,
        }
    else:
        assert "exactly one authenticated non-prerelease draft" in result.stderr


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
    remote_count: int,
    http_status: str,
    expected_missing: int,
    upload_required: str,
) -> None:
    _install_fake_endpoints(tmp_path)
    _, asset_records = _setup_assets(tmp_path)
    dists = _setup_dists(tmp_path)
    remote = [_pypi_file(name, payload) for name, payload in dists.items()][
        :remote_count
    ]
    release = _release(draft=True, assets=asset_records)
    _write_json(tmp_path / "release-sequence.json", [release])
    _write_json(tmp_path / "release-list.json", [[release]])
    _write_json(tmp_path / "pypi-response.json", {"urls": remote})
    env = _base_env(tmp_path) | {"PYPI_HTTP_STATUS": http_status}

    result = _run_bash(
        tmp_path,
        _step_script("reconcile exact PyPI distribution state"),
        env,
    )

    assert result.returncode == 0, result.stderr
    assert _outputs(tmp_path) == {
        "upload_required": upload_required,
        "missing_count": str(expected_missing),
    }
    staged = (
        {path.name for path in (tmp_path / "dist-upload").iterdir()}
        if (tmp_path / "dist-upload").exists()
        else set()
    )
    assert staged == set(list(dists)[remote_count:])


@pytest.mark.parametrize("mismatch", ("size", "digest", "yanked", "unexpected"))
def test_pypi_mismatch_is_fail_closed(tmp_path: Path, mismatch: str) -> None:
    _install_fake_endpoints(tmp_path)
    _, asset_records = _setup_assets(tmp_path)
    dists = _setup_dists(tmp_path)
    name, payload = next(iter(dists.items()))
    remote = _pypi_file(name, payload)
    if mismatch == "size":
        remote["size"] += 1
    elif mismatch == "digest":
        remote["digests"]["sha256"] = "0" * 64
    elif mismatch == "yanked":
        remote["yanked"] = True
    else:
        remote["filename"] = f"gpuwm-{VERSION}-cp999-none-any.whl"
    release = _release(draft=True, assets=asset_records)
    _write_json(tmp_path / "release-sequence.json", [release])
    _write_json(tmp_path / "release-list.json", [[release]])
    _write_json(tmp_path / "pypi-response.json", {"urls": [remote]})

    result = _run_bash(
        tmp_path,
        _step_script("reconcile exact PyPI distribution state"),
        _base_env(tmp_path),
    )

    assert result.returncode != 0
    expected = (
        "unexpected distribution"
        if mismatch == "unexpected"
        else "differs from the proven artifact"
    )
    assert expected in result.stderr


def test_upload_action_is_conditioned_on_the_reconciliation_decision() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "if: steps.pypi.outputs.upload_required == 'true'" in text
    assert "packages-dir: dist-upload/" in text


def test_exact_draft_promotes_once_to_immutable_public_release(tmp_path: Path) -> None:
    _install_fake_endpoints(tmp_path)
    _, asset_records = _setup_assets(tmp_path)
    dists = _setup_dists(tmp_path)
    draft = _release(draft=True, assets=asset_records)
    public = _release(draft=False, assets=asset_records, immutable=True)
    _write_json(tmp_path / "release-sequence.json", [draft])
    _write_json(tmp_path / "release-list.json", [[draft]])
    _write_json(tmp_path / "release-list-sequence.json", [[[draft]], [[public]]])
    _write_json(
        tmp_path / "pypi-response.json",
        {"urls": [_pypi_file(name, payload) for name, payload in dists.items()]},
    )
    env = _base_env(tmp_path) | {
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
    _write_json(
        tmp_path / "pypi-response.json",
        {"urls": [_pypi_file(name, payload) for name, payload in dists.items()]},
    )
    env = _base_env(tmp_path) | {
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


def test_already_public_same_id_retry_succeeds_without_patch(tmp_path: Path) -> None:
    _install_fake_endpoints(tmp_path)
    _, asset_records = _setup_assets(tmp_path)
    dists = _setup_dists(tmp_path)
    release = _release(draft=False, assets=asset_records, immutable=True)
    _write_json(tmp_path / "release-sequence.json", [release])
    _write_json(tmp_path / "release-list.json", [[release]])
    _write_json(
        tmp_path / "pypi-response.json",
        {"urls": [_pypi_file(name, payload) for name, payload in dists.items()]},
    )

    result = _run_bash(
        tmp_path,
        _step_script("publish the fully-proven draft"),
        _base_env(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    assert "already public at the captured id" in result.stdout
    assert not any("PATCH" in call for call in _logged_calls(tmp_path / "gh.log"))
