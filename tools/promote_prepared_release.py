"""Validate and promote an exact prepared release; never rebuild or replace it.

PyPI uploads remain in the OIDC publisher action. This controller supplies the
validated missing-file directories, bounded index checks and GitHub promotion.
It has no delete, tag-creation, version-bump or distribution-build operation.
"""
from __future__ import annotations

import argparse
import ast
import email
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
from typing import Callable
import urllib.error
import urllib.parse
import urllib.request
import venv
import zipfile

MANIFEST = "PUBLICATION-ASSETS.json"
BRIDGE_MANIFEST = "bridge-bundle-manifest.json"
PROJECTS = ("gpuwm-data", "gpuwm")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
PIN_MARKER = re.compile(r"<!--\s*arwen-publication-sha256:\s*([0-9a-f]{64})\s*-->")
PROOF_SCHEMA = "gpuwm.prepared-publication-proof.v1"


class PublicationError(RuntimeError):
    """The requested publication does not describe one proven set of bytes."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON member: {key}")
        result[key] = value
    return result


def json_bytes(payload: bytes):
    try:
        return json.loads(payload.decode("utf-8-sig"), object_pairs_hook=_object)
    except (UnicodeError, ValueError) as error:
        raise PublicationError(f"invalid JSON: {error}") from error


def read_json(path: Path):
    return json_bytes(path.read_bytes())


def write_json(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def manifest_pin(explicit: str, event: dict) -> str:
    body = event.get("release", {}).get("body") or ""
    require(isinstance(body, str), "release body must be text")
    matches = PIN_MARKER.findall(body)
    require(body.count("arwen-publication-sha256:") <= 1, "release body has duplicate publication manifest pins")
    if "arwen-publication-sha256:" in body:
        require(len(matches) == 1, "release body has an invalid publication manifest pin")
    value = explicit.strip() if isinstance(explicit, str) else ""
    if value:
        require(bool(SHA256.fullmatch(value)), "manifest SHA-256 must be 64 lowercase hex characters")
    if matches and value:
        require(matches[0] == value, "dispatch and release-body manifest pins disagree")
    value = value or (matches[0] if matches else "")
    require(bool(SHA256.fullmatch(value)), "provide publication_manifest_sha256 or an arwen-publication-sha256 release-body marker")
    return value


def _rows(value, label: str) -> dict[str, dict]:
    require(isinstance(value, list), f"{label} must be a list")
    result = {}
    for row in value:
        require(isinstance(row, dict), f"{label} contains a non-object")
        name = row.get("filename")
        require(isinstance(name, str) and bool(FILENAME.fullmatch(name)), f"unsafe artifact filename: {name!r}")
        require(name not in result, f"duplicate artifact filename in {label}: {name}")
        require(type(row.get("bytes")) is int and row["bytes"] > 0, f"invalid size for {name}")
        require(isinstance(row.get("sha256"), str) and bool(SHA256.fullmatch(row["sha256"])), f"invalid SHA-256 for {name}")
        result[name] = {"filename": name, "bytes": row["bytes"], "sha256": row["sha256"]}
    return result


def load_manifest(document: dict, tag: str, commit: str, repository: str) -> dict:
    require(isinstance(document, dict), "publication manifest must be an object")
    require(document.get("schema") == "arwen.publication-assets.v1", "unsupported publication manifest schema")
    require(bool(COMMIT.fullmatch(commit)), "release commit must be a full lowercase Git SHA")
    require(document.get("engine_source_revision") == commit, "manifest source revision does not equal the public tag commit")
    require(isinstance(document.get("desktop_source_revision"), str) and bool(COMMIT.fullmatch(document["desktop_source_revision"])), "invalid desktop source revision")
    require(isinstance(tag, str) and bool(re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+[A-Za-z0-9.+-]*", tag)), "invalid release tag")
    require(bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)), "invalid repository name")
    github = document.get("github")
    pypi = document.get("pypi")
    require(isinstance(github, dict) and isinstance(pypi, dict), "manifest needs github and pypi objects")
    require(github.get("repository") == repository and github.get("target_version") == tag, "manifest repository or tag differs from the selected release")
    version = tag[1:]
    github_rows = _rows(github.get("assets"), "github.assets")
    pypi_rows = _rows(pypi.get("artifacts"), "pypi.artifacts")
    expected = {
        f"gpuwm-{version}-py3-none-any.whl",
        f"gpuwm-{version}-py3-none-manylinux_2_28_x86_64.whl",
        f"gpuwm-{version}-py3-none-win_amd64.whl",
        f"gpuwm-{version}.tar.gz",
        f"gpuwm_data-{version}-py3-none-any.whl",
        f"gpuwm_data-{version}.tar.gz",
    }
    require(set(pypi_rows) == expected, "PyPI manifest must declare exactly the three engine wheels, engine sdist and data wheel/sdist")
    for row in pypi_rows.values():
        require(row["bytes"] < 100_000_000, f"distribution exceeds the release size gate: {row['filename']}")
    native = {BRIDGE_MANIFEST, f"gpuwm-bridges-{tag}-linux-x86_64.zip", f"gpuwm-bridges-{tag}-win-x86_64.zip"}
    require(native <= set(github_rows), "GitHub manifest is missing the native bundles or bridge-bundle-manifest.json")
    require(MANIFEST not in github_rows and MANIFEST not in pypi_rows, "the manifest is pinned externally and must not hash itself")
    auxiliary = github.get("also_attach", [])
    require(isinstance(auxiliary, list) and all(isinstance(x, str) for x in auxiliary), "github.also_attach must be a list of names")
    require(len(auxiliary) == len(set(auxiliary)), "duplicate auxiliary asset")
    require(set(auxiliary) <= {MANIFEST, "DOWNLOAD-SHA256SUMS.txt"}, "an auxiliary asset lacks a declared content hash")
    require(not (set(auxiliary) - {MANIFEST}) & set(github_rows), "an asset cannot be both hashed and auxiliary")
    combined = dict(github_rows)
    for name, row in pypi_rows.items():
        require(name not in combined or combined[name] == row, f"GitHub and PyPI disagree on {name}")
        combined[name] = row
    return {
        "tag": tag, "version": version, "commit": commit, "repository": repository,
        "desktop_source_revision": document["desktop_source_revision"],
        "rows": combined, "pypi": {
            project: [row for name, row in sorted(pypi_rows.items())
                      if name.startswith("gpuwm_data-" if project == "gpuwm-data" else "gpuwm-")]
            for project in PROJECTS},
        "expected_names": sorted(set(combined) | set(auxiliary) | {MANIFEST}),
    }


def reconcile_index(rows: list[dict], status: int, payload: dict | None,
                    project: str, version: str) -> list[dict]:
    require(project in PROJECTS, "unknown PyPI project")
    expected = _rows(rows, "expected PyPI files")
    if status == 404:
        return list(rows)
    require(status == 200, f"PyPI {project} {version} returned HTTP {status}")
    require(isinstance(payload, dict), "PyPI returned a non-object")
    info = payload.get("info")
    require(isinstance(info, dict), "PyPI response has no project identity")
    canonical = lambda name: re.sub(r"[-_.]+", "-", name).lower() if isinstance(name, str) else None
    require(canonical(info.get("name")) == project and info.get("version") == version, "PyPI response belongs to another project/version")
    files = payload.get("urls")
    require(isinstance(files, list), "PyPI response has no file list")
    seen = set()
    for row in files:
        require(isinstance(row, dict), "PyPI file record is not an object")
        name = row.get("filename")
        require(isinstance(name, str) and name not in seen, "duplicate or invalid PyPI filename")
        require(name in expected, f"PyPI has an undeclared distribution: {name}")
        seen.add(name)
        wanted = expected[name]
        require(row.get("size") == wanted["bytes"] and row.get("digests", {}).get("sha256") == wanted["sha256"],
                f"PyPI distribution differs from the proven artifact: {name}")
        require(row.get("yanked", False) is False, f"PyPI distribution is yanked: {name}")
    return [row for row in rows if row["filename"] not in seen]


def wait_for_index(project: str, version: str, rows: list[dict],
                   fetch: Callable, clock: Callable = time.monotonic,
                   sleep: Callable = time.sleep, timeout: float = 300,
                   required: list[dict] | None = None) -> dict:
    require(math.isfinite(timeout) and timeout > 0, "index timeout must be finite and positive")
    required_names = {row["filename"] for row in (rows if required is None else required)}
    deadline = clock() + timeout
    attempts = 0
    while True:
        remaining = deadline - clock()
        require(remaining > 0, f"PyPI {project} exact-state deadline expired")
        attempts += 1
        status, payload = fetch(project, version, min(30.0, remaining))
        require(clock() <= deadline, f"PyPI {project} exact-state deadline expired")
        if status in (429, 500, 502, 503, 504):
            missing = rows
        else:
            missing = reconcile_index(rows, status, payload, project, version)
        if not any(row["filename"] in required_names for row in missing):
            return {"project": project, "version": version, "files": len(required_names), "attempts": attempts, "status": "PASS"}
        remaining = deadline - clock()
        require(remaining > 0, f"PyPI {project} exact-state deadline expired")
        sleep(min(float(2 ** min(attempts - 1, 4)), remaining))


class _Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if redirected and urllib.parse.urlsplit(request.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def fetch_index(project: str, version: str, timeout: float = 30) -> tuple[int, dict | None]:
    require(project in PROJECTS, "unknown PyPI project")
    url = f"https://pypi.org/pypi/{project}/{urllib.parse.quote(version, safe='')}/json"
    request = urllib.request.Request(url, headers={"Accept": "application/json", "Cache-Control": "no-cache", "User-Agent": "gpuwm-prepared-publisher"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json_bytes(response.read())
    except urllib.error.HTTPError as error:
        if error.code == 404 or error.code in (429, 500, 502, 503, 504):
            return error.code, None
        raise PublicationError(f"PyPI {project} returned HTTP {error.code}") from error
    except (TimeoutError, urllib.error.URLError) as error:
        raise PublicationError(f"PyPI {project} request did not complete within its request budget: {error}") from error


class GitHub:
    def __init__(self, repository: str):
        require(bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)), "invalid repository")
        self.repository = repository
        self.prefix = f"https://api.github.com/repos/{repository}"
        self.opener = urllib.request.build_opener(_Redirect())

    def open(self, suffix: str, *, data=None, binary=False):
        headers = {"Accept": "application/octet-stream" if binary else "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2026-03-10", "User-Agent": "gpuwm-prepared-publisher"}
        token = os.environ.get("GH_TOKEN")
        if token:
            headers["Authorization"] = "Bearer " + token
        payload = None if data is None else json.dumps(data).encode()
        request = urllib.request.Request(self.prefix + suffix, data=payload, headers=headers,
                                         method="GET" if data is None else "PATCH")
        try:
            return self.opener.open(request, timeout=30)
        except urllib.error.HTTPError as error:
            raise PublicationError(f"GitHub {suffix} returned HTTP {error.code}") from error

    def json(self, suffix: str, *, data=None):
        with self.open(suffix, data=data) as response:
            return json_bytes(response.read())

    def releases(self) -> list[dict]:
        result = []
        for page in range(1, 51):
            rows = self.json(f"/releases?per_page=100&page={page}")
            require(isinstance(rows, list), "GitHub releases response is not a list")
            result.extend(rows)
            if len(rows) < 100:
                return result
        raise PublicationError("release inventory exceeds the bounded pagination limit")

    def commit(self, tag: str) -> str:
        obj = self.json("/git/ref/tags/" + urllib.parse.quote(tag, safe=""))["object"]
        for _ in range(16):
            if obj["type"] != "tag":
                break
            obj = self.json("/git/tags/" + obj["sha"])["object"]
        require(obj.get("type") == "commit" and bool(COMMIT.fullmatch(obj.get("sha", ""))), "tag does not peel to one commit")
        return obj["sha"]

    def download(self, asset: dict, destination: Path, expected_sha: str | None = None) -> str:
        temporary = destination.with_name(destination.name + ".part")
        destination.parent.mkdir(parents=True, exist_ok=True)
        require(not destination.exists() and not temporary.exists(), f"refusing to overwrite captured asset {destination.name}")
        hasher = hashlib.sha256()
        total = 0
        with self.open(f"/releases/assets/{asset['id']}", binary=True) as response, temporary.open("xb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                total += len(chunk)
                require(total <= asset["size"], f"download exceeds declared size: {asset['name']}")
                output.write(chunk)
                hasher.update(chunk)
        actual = hasher.hexdigest()
        require(total == asset["size"], f"download size differs: {asset['name']}")
        if expected_sha:
            require(actual == expected_sha, f"download hash differs: {asset['name']}")
        if asset.get("digest") is not None:
            require(asset["digest"] == "sha256:" + actual, f"GitHub digest differs: {asset['name']}")
        temporary.replace(destination)
        return actual


def asset_inventory(release: dict) -> dict[str, dict]:
    rows = release.get("assets")
    require(isinstance(rows, list), "GitHub release has no asset inventory")
    result = {}
    for row in rows:
        require(isinstance(row, dict), "invalid GitHub asset record")
        name = row.get("name")
        require(isinstance(name, str) and bool(FILENAME.fullmatch(name)) and name not in result, "unsafe or duplicate GitHub asset name")
        require(row.get("state") == "uploaded", f"GitHub asset is incomplete: {name}")
        require(type(row.get("id")) is int and row["id"] > 0, f"invalid GitHub asset id: {name}")
        require(type(row.get("size")) is int and row["size"] > 0, f"invalid GitHub asset size: {name}")
        result[name] = row
    return result


def selected_release(client: GitHub, tag: str, commit: str, release_id: int | None = None) -> dict:
    rows = [row for row in client.releases() if row.get("tag_name") == tag]
    require(len(rows) == 1, "exactly one authenticated release must carry the selected tag")
    release = rows[0]
    require(type(release.get("id")) is int and release["id"] > 0, "invalid release id")
    if release_id is not None:
        require(release["id"] == release_id, "selected release id changed")
    require(type(release.get("draft")) is bool and type(release.get("prerelease")) is bool, "invalid GitHub release state")
    require(client.commit(tag) == commit, "the public tag moved from the selected source commit")
    return release


def capture(args) -> None:
    event = read_json(args.event)
    require(args.ref == "refs/tags/" + args.tag, "publication requires the exact selected tag ref")
    pin = manifest_pin(args.manifest_sha256, event)
    client = GitHub(args.repository)
    release = selected_release(client, args.tag, args.commit)
    if args.event_name == "release":
        require(event.get("action") == "published" and release["draft"] is False, "release ingress requires a published release")
        require(event.get("release", {}).get("id") == release["id"], "release event id differs from the current release")
    else:
        require(args.event_name == "workflow_dispatch", "only release publication or manual dispatch may publish")
    require(manifest_pin(pin, {"release": release}) == pin, "current release manifest pin differs")
    inventory = asset_inventory(release)
    require(MANIFEST in inventory, "release does not contain its prepared publication manifest; no rebuild fallback exists")
    require(not args.out.exists() or not any(args.out.iterdir()), "capture directory must be empty")
    assets = args.out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    client.download(inventory[MANIFEST], assets / MANIFEST, pin)
    plan = load_manifest(read_json(assets / MANIFEST), args.tag, args.commit, args.repository)
    require(set(inventory) == set(plan["expected_names"]), "GitHub assets do not match the complete prepared manifest set")
    if args.stable_release_expected:
        require(bool(re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", plan["version"])), "stable_release_expected requires a stable X.Y.Z version")
    elif not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", plan["version"]):
        print("::warning::publishing non-stable version " + plan["version"])
    if not args.immutable_releases_enabled:
        print("::warning::release immutability not confirmed; continuing under the declared optional policy")
    records = []
    for name, asset in sorted(inventory.items()):
        expected = plan["rows"].get(name)
        if expected:
            require(asset["size"] == expected["bytes"], f"GitHub size differs from manifest: {name}")
        actual = pin if name == MANIFEST else client.download(asset, assets / name, expected["sha256"] if expected else None)
        records.append({"filename": name, "bytes": asset["size"], "sha256": actual, "asset_id": asset["id"]})
    check_download_checksums(assets, plan, pin)
    record = {"schema": "gpuwm.prepared-release-capture.v1", "tag": args.tag, "version": plan["version"],
              "commit": args.commit, "repository": args.repository, "release_id": release["id"],
              "manifest_sha256": pin, "draft": release["draft"], "prerelease": release["prerelease"],
              "immutable_required": args.immutable_releases_enabled, "assets": records}
    verify_remote(client, record, require_public=not release["draft"])
    write_json(args.out / "CAPTURE.json", record)
    output_values({"tag": args.tag, "version": plan["version"], "commit": args.commit,
                   "release_id": str(release["id"]), "manifest_sha256": pin})


def verify_remote(client: GitHub, captured: dict, *, require_public: bool) -> dict:
    release = selected_release(client, captured["tag"], captured["commit"], captured["release_id"])
    require(release["prerelease"] == captured["prerelease"], "captured prerelease state changed")
    if require_public:
        require(release["draft"] is False, "required download assets are still private")
    require(manifest_pin(captured["manifest_sha256"], {"release": release}) == captured["manifest_sha256"], "release body pin changed")
    inventory = asset_inventory(release)
    expected = {row["filename"]: row for row in captured["assets"]}
    require(set(inventory) == set(expected), "GitHub asset set changed after capture")
    for name, row in expected.items():
        actual = inventory[name]
        require(actual["id"] == row["asset_id"] and actual["size"] == row["bytes"], f"GitHub asset identity changed: {name}")
        if actual.get("digest") is not None:
            require(actual["digest"] == "sha256:" + row["sha256"], f"GitHub asset bytes changed: {name}")
    if require_public and captured["immutable_required"]:
        require(release.get("immutable") is True, "immutability was confirmed but GitHub reports the published release as mutable")
    return release


def check_files(directory: Path, rows: list[dict]) -> None:
    for row in rows:
        path = directory / row["filename"]
        require(path.is_file() and not path.is_symlink(), f"missing or linked artifact: {row['filename']}")
        require(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], f"artifact changed: {row['filename']}")


def check_download_checksums(assets: Path, plan: dict, manifest_sha256: str) -> None:
    name = "DOWNLOAD-SHA256SUMS.txt"
    if name not in plan["expected_names"]:
        return
    expected = {filename: row["sha256"] for filename, row in plan["rows"].items() if filename != name}
    expected[MANIFEST] = manifest_sha256
    require(set(expected) == set(plan["expected_names"]) - {name}, "checksum coverage differs from the complete asset inventory")
    observed = {}
    for line in (assets / name).read_text(encoding="utf-8-sig").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *]([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        require(match is not None, "invalid download checksum line")
        checksum, filename = match.groups()
        require(filename not in observed, f"duplicate download checksum: {filename}")
        observed[filename] = checksum
    require(observed == expected, "download checksum list differs from the complete pinned asset inventory")


def empty_output(path: Path, label: str) -> None:
    require(not path.exists() or (path.is_dir() and not any(path.iterdir())), f"{label} directory must be empty")
    path.mkdir(parents=True, exist_ok=True)


def checked_run(command: list[str], *, cwd: Path | None = None, env=None, log: Path | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=1200)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise PublicationError(f"verification failed ({command[1:4]}):\n{result.stdout[-12000:]}")


def _metadata(path: Path) -> tuple[email.message.Message, dict[str, bytes]]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), f"duplicate archive member in {path.name}")
            metas = [n for n in names if n.endswith(".dist-info/METADATA")]
            require(len(metas) == 1, f"expected one METADATA in {path.name}")
            sources = {n: archive.read(n) for n in names if n.endswith(".py")}
            return email.message_from_string(archive.read(metas[0]).decode("utf-8")), sources
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        require(len(names) == len(set(names)), f"duplicate archive member in {path.name}")
        metas = [m for m in members if m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")]
        require(len(metas) == 1, f"expected one root PKG-INFO in {path.name}")
        require(all(m.isfile() or m.isdir() for m in members), f"linked/special sdist member in {path.name}")
        sources = {m.name.split("/", 1)[1]: archive.extractfile(m).read() for m in members if m.isfile() and m.name.endswith(".py")}
        return email.message_from_string(archive.extractfile(metas[0]).read().decode("utf-8")), sources


def expected_python_files(repo: Path, project: str = "gpuwm") -> set[str]:
    from setuptools import find_namespace_packages
    from setuptools.command.egg_info import FileList
    root = repo / "gpuwm-data" if project == "gpuwm-data" else repo
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    finder = config["tool"]["setuptools"]["packages"]["find"]
    require(finder.get("where", ["."]) == ["."], "unsupported package source root")
    filters = {}
    if project == "gpuwm":
        for node in ast.parse((root / "setup.py").read_text(encoding="utf-8")).body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "DEVELOPMENT_MODULE_GLOBS":
                filters = ast.literal_eval(node.value)
    packages = find_namespace_packages(str(root), include=finder.get("include", ["*"]), exclude=finder.get("exclude", []))
    data = config["tool"]["setuptools"].get("package-data", {})
    excluded = config["tool"]["setuptools"].get("exclude-package-data", {})
    result = set()
    for package in packages:
        directory = root.joinpath(*package.split("."))
        for source in directory.glob("*.py"):
            if not any(fnmatch.fnmatchcase(source.stem, pattern) for pattern in filters.get(package, ())):
                result.add(source.relative_to(root).as_posix())
        # Some intentionally shipped Python examples/launchers are package
        # data under non-package directories; they are still install content.
        for pattern in data.get("*", []) + data.get(package, []):
            for source in directory.glob(pattern):
                if source.is_file() and source.suffix == ".py":
                    relative = source.relative_to(directory).as_posix()
                    if not any(fnmatch.fnmatchcase(relative, value) for value in excluded.get("*", []) + excluded.get(package, [])):
                        result.add(source.relative_to(root).as_posix())
    manifest = root / "MANIFEST.in"
    if manifest.is_file():
        selected = FileList()
        selected.files = sorted(str(Path(name)) for name in result)
        selected.allfiles = [str(p.relative_to(root)) for p in root.rglob("*.py")]
        for line in manifest.read_text(encoding="utf-8").splitlines():
            # Only Python inclusion/exclusion can affect this census. The
            # full release verifier separately proves all pinned data bytes.
            content = line.split("#", 1)[0].strip()
            if content and (".py" in content or content.split()[0] in {"graft", "prune"}):
                selected.process_template_line(content)
        roots = tuple(package.replace(".", "/") + "/" for package in packages)
        result = {Path(name).as_posix() for name in selected.files
                  if name.endswith(".py") and Path(name).as_posix().startswith(roots)}
    require(bool(result), "declared wheel Python inventory is empty")
    return result


def check_python_inventory(path: Path, repo: Path, project: str) -> None:
    _, sources = _metadata(path)
    expected = expected_python_files(repo, project)
    require(set(sources) == expected,
            f"wheel Python inventory differs from tagged source: missing={sorted(expected - set(sources))}, extra={sorted(set(sources) - expected)}")


def check_embedded_natives(path: Path, pins: dict, platform: str | None) -> dict:
    prefix = "gpuwm/libexec/bridges/"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "duplicate native wheel archive member")
        held = {name[len(prefix):] for name in names if name.startswith(prefix) and not name.endswith("/")}
        if platform is None:
            require(not held, "pure wheel contains embedded native artifacts")
            return {"platform": None, "artifacts": 0}
        rows = pins["platforms"][platform]["binaries"]
        expected = _rows(rows, "native bundle pins")
        require(held == set(expected) | {"BUNDLE.json"}, "embedded native inventory differs from the proven bundle")
        document = json_bytes(archive.read(prefix + "BUNDLE.json"))
        require(document.get("schema") == "gpuwm-wheel-bridge-bundle-v1" and document.get("platform") == platform, "embedded native manifest has the wrong platform/schema")
        declared = _rows(document.get("artifacts"), "embedded native manifest")
        require(declared == expected, "embedded native manifest differs from proven bundle hashes")
        identities = {row["filename"]: row["artifact"] for row in rows}
        for row in document["artifacts"]:
            require(row.get("artifact") == identities[row["filename"]] and row.get("kind") in ("executable", "library"), "embedded native artifact identity/kind differs")
        for name, row in expected.items():
            payload = archive.read(prefix + name)
            require(len(payload) == row["bytes"] and hashlib.sha256(payload).hexdigest() == row["sha256"], f"embedded native bytes differ from bundle: {name}")
    return {"platform": platform, "artifacts": len(expected)}


def check_metadata(path: Path, project: str, version: str, repo: Path) -> dict:
    from packaging.version import Version
    metadata, sources = _metadata(path)
    require(re.sub(r"[-_.]+", "-", metadata.get("Name", "")).lower() == project, f"wrong distribution name: {path.name}")
    require(metadata.get("Version") == version and Version(version).local is None, f"wrong or non-publishable version: {path.name}")
    project_root = repo / "gpuwm-data" if project == "gpuwm-data" else repo
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    config = pyproject["project"]
    declared = config.get("version")
    if declared is None and "version" in config.get("dynamic", []):
        version_files = pyproject.get("tool", {}).get("setuptools", {}).get("dynamic", {}).get("version", {}).get("file")
        require(isinstance(version_files, list) and len(version_files) == 1, "unsupported dynamic release-version declaration")
        relative = PurePosixPath(version_files[0])
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe dynamic version path")
        declared = (project_root / relative).read_text(encoding="utf-8").strip()
    require(declared == version, "pyproject version differs from release tag")
    readme = config["readme"]
    if isinstance(readme, dict):
        readme = readme["file"]
    normalize = lambda text: text.replace("\r\n", "\n").strip()
    require(normalize(metadata.get_payload()) == normalize((project_root / readme).read_text(encoding="utf-8")), f"distribution README differs from tagged source: {path.name}")
    for name, payload in sources.items():
        parts = PurePosixPath(name)
        require(not parts.is_absolute() and ".." not in parts.parts and "\\" not in name, "unsafe distribution source path")
        source = project_root / name
        require(source.is_file() and source.read_bytes() == payload, f"distribution source differs from public checkout: {name}")
    return {"filename": path.name, "project": project, "version": version,
            "metadata_version": metadata["Metadata-Version"], "python_files": len(sources)}


def _capture_plan(packet: Path) -> tuple[dict, dict]:
    captured = read_json(packet / "CAPTURE.json")
    require(captured.get("schema") == "gpuwm.prepared-release-capture.v1", "unsupported capture schema")
    for variable, value in (("PUBLICATION_COMMIT", captured["commit"]),
                            ("PUBLICATION_MANIFEST_SHA256", captured["manifest_sha256"]),
                            ("GITHUB_REPOSITORY", captured["repository"])):
        expected = os.environ.get(variable)
        if expected:
            require(expected == value, f"capture differs from workflow authority: {variable}")
    check_files(packet / "assets", captured["assets"])
    manifest = packet / "assets" / MANIFEST
    require(digest(manifest) == captured["manifest_sha256"], "captured manifest hash changed")
    plan = load_manifest(read_json(manifest), captured["tag"], captured["commit"], captured["repository"])
    require({row["filename"] for row in captured["assets"]} == set(plan["expected_names"]), "capture inventory differs from manifest")
    check_files(packet / "assets", list(plan["rows"].values()))
    check_download_checksums(packet / "assets", plan, captured["manifest_sha256"])
    return captured, plan


def verify(args) -> None:
    empty_output(args.out, "proof")
    captured, plan = _capture_plan(args.packet)
    repo = args.repo.resolve()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    require(commit == plan["commit"], "verification checkout is not the captured public tag commit")
    require(not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "-uno"], text=True).strip(), "verification source checkout is dirty")
    checked_run([sys.executable, str(repo / "tools/verify_source_bridge_pins.py")], cwd=repo)
    assets = args.packet / "assets"
    shutil.copyfile(assets / MANIFEST, args.out / MANIFEST)
    pins = args.out / "bridge-pins.json"
    generated = args.out / BRIDGE_MANIFEST
    command = [sys.executable, str(repo / "tools/build_bridge_bundle.py"), "pin", "--release", plan["tag"],
               "--source-rev", plan["commit"], "--out", str(pins), "--manifest", str(generated)]
    for platform in ("linux-x86_64", "win-x86_64"):
        command += ["--bundle", str(assets / f"gpuwm-bridges-{plan['tag']}-{platform}.zip")]
    checked_run(command, cwd=repo, log=args.out / "native-pin-verification.log")
    require(generated.read_bytes() == (assets / BRIDGE_MANIFEST).read_bytes(), "prepared native manifest differs from regenerated exact-byte pins")
    metadata_records = []
    for project in PROJECTS:
        for row in plan["pypi"][project]:
            metadata_records.append(check_metadata(assets / row["filename"], project, plan["version"], repo))
            if row["filename"].endswith(".whl"):
                check_python_inventory(assets / row["filename"], repo, project)
                if project == "gpuwm":
                    platform = "win-x86_64" if row["filename"].endswith("-win_amd64.whl") else "linux-x86_64" if row["filename"].endswith("-manylinux_2_28_x86_64.whl") else None
                    check_embedded_natives(assets / row["filename"], read_json(pins), platform)
    files = [assets / row["filename"] for project in PROJECTS for row in plan["pypi"][project]]
    checked_run([sys.executable, "-m", "twine", "check", "--strict", *map(str, files)], log=args.out / "twine-check.log")
    for row in plan["pypi"]["gpuwm"]:
        if row["filename"].endswith(".whl"):
            checked_run([sys.executable, "-m", "tools.verify_release_artifacts", "--dry-run",
                "--wheel", str(assets / row["filename"]), "--sdist", str(assets / f"gpuwm-{plan['version']}.tar.gz"),
                "--pins", str(pins), "--manifest", str(generated), "--bundles", str(assets),
                "--release", plan["tag"], "--source-rev", plan["commit"],
                "--receipt", str(args.out / (row["filename"] + ".proof.json"))], cwd=repo)
    collision_checks = {}
    for project in PROJECTS:
        status, payload = fetch_index(project, plan["version"])
        missing = reconcile_index(plan["pypi"][project], status, payload, project, plan["version"])
        collision_checks[project] = {"http_status": status, "missing": [r["filename"] for r in missing]}
    dists = args.out / "dists"
    dists.mkdir(exist_ok=True)
    for path in files:
        shutil.copyfile(path, dists / path.name)
    write_json(args.out / "PROOF.json", {"schema": PROOF_SCHEMA, "status": "PASS", "captured": captured,
        "plan": plan, "metadata": metadata_records, "prewrite_pypi_checks": collision_checks,
        "native_rebuilt": False, "distributions_rebuilt": False})


def proof(path: Path) -> dict:
    result = read_json(path / "PROOF.json")
    require(result.get("schema") == PROOF_SCHEMA and result.get("status") == "PASS", "prepared proof is not PASS")
    captured = result["captured"]
    manifest = path / MANIFEST
    require(digest(manifest) == captured["manifest_sha256"], "proof manifest hash changed")
    normalized = load_manifest(read_json(manifest), captured["tag"], captured["commit"], captured["repository"])
    require(result["plan"] == normalized, "proof plan disagrees with its pinned manifest")
    for variable, value in (("PUBLICATION_COMMIT", captured["commit"]),
                            ("PUBLICATION_MANIFEST_SHA256", captured["manifest_sha256"])):
        expected = os.environ.get(variable)
        if expected:
            require(expected == value, f"proof differs from captured workflow authority: {variable}")
    return result


def smoke(args) -> None:
    empty_output(args.out, "smoke")
    proven = proof(args.proof)
    captured, plan = _capture_plan(args.packet)
    require(captured == proven["captured"], "platform smoke capture differs from prepared proof")
    assets = args.packet / "assets"
    version = plan["version"]
    suffix = "win_amd64" if args.platform == "win-x86_64" else "manylinux_2_28_x86_64"
    wheel = assets / f"gpuwm-{version}-py3-none-{suffix}.whl"
    runtime = args.out / "venv"
    venv.EnvBuilder(with_pip=True).create(runtime)
    python = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    checked_run([str(python), "-m", "pip", "install", str(wheel) + "[render]", str(assets / f"gpuwm_data-{version}-py3-none-any.whl")], log=args.out / "install.log")
    outside = args.out / "outside"
    outside.mkdir(exist_ok=True)
    home = args.out / "home"
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "GPUWM_", "ARWEN_"))}
    env.update(HOME=str(home), USERPROFILE=str(home), LOCALAPPDATA=str(home / "local"), APPDATA=str(home / "roaming"),
               PYTHONDONTWRITEBYTECODE="1", PYTHONSAFEPATH="1", GPUWM_NO_LOCAL_GPU="1")
    receipt = args.out / "native-proof.json"
    checked_run([str(python), "-I", "-B", str(args.repo / "tools/verify_release_artifacts.py"),
        "--wheel", str(wheel), "--sdist", str(assets / f"gpuwm-{version}.tar.gz"),
        "--pins", str(args.proof / "bridge-pins.json"), "--manifest", str(args.proof / BRIDGE_MANIFEST),
        "--bundles", str(assets), "--release", plan["tag"], "--source-rev", plan["commit"],
        "--repo-root", str(args.repo), "--stage", str(args.out / "native-stage"), "--receipt", str(receipt)], cwd=outside, env=env)
    checked_run([str(python), "-I", "-B", "-m", "pip", "check"], cwd=outside, env=env)
    checked_run([str(python), "-I", "-B", "-m", "gpuwm.cli", "tui", "--snapshot", str(args.out / "terminal.html"),
                 "--snapshot-screen", "home", "--output", str(args.out / "runs")], cwd=outside, env=env)
    write_json(args.out / "SMOKE.json", {"schema": "gpuwm.prepared-platform-smoke.v1", "status": "PASS",
        "platform": args.platform, "commit": plan["commit"], "manifest_sha256": captured["manifest_sha256"],
        "wheel_sha256": digest(wheel), "native_proof_sha256": digest(receipt), "forecast_started": False})


def check_indexes(proven: dict) -> None:
    plan = proven["plan"]
    for project in PROJECTS:
        status, payload = fetch_index(project, plan["version"])
        reconcile_index(plan["pypi"][project], status, payload, project, plan["version"])


def promote(args) -> None:
    proven = proof(args.proof)
    captured = proven["captured"]
    smoke_rows = [read_json(path) for path in args.smokes.rglob("SMOKE.json")]
    require(len(smoke_rows) == 2 and {row.get("platform") for row in smoke_rows} == {"linux-x86_64", "win-x86_64"}, "both exact platform smoke proofs are required")
    for row in smoke_rows:
        require(row.get("status") == "PASS" and row.get("commit") == captured["commit"] and row.get("manifest_sha256") == captured["manifest_sha256"], "platform qualification does not match the prepared release")
    # Both projects are read again before the first public mutation.
    check_indexes(proven)
    client = GitHub(captured["repository"])
    release = verify_remote(client, captured, require_public=False)
    if release["draft"]:
        client.json(f"/releases/{captured['release_id']}", data={"draft": False, "prerelease": captured["prerelease"]})
    current = verify_remote(client, captured, require_public=True)
    write_json(args.out, {"schema": "gpuwm.prepared-public-assets.v1", "status": "PASS", "release_id": current["id"],
        "commit": captured["commit"], "manifest_sha256": captured["manifest_sha256"], "immutable": current.get("immutable", False),
        "public_before_pypi": True})


def stage_missing(args) -> None:
    proven = proof(args.proof)
    plan = proven["plan"]
    rows = plan["pypi"][args.project]
    # Reconcile both projects before each project's upload, not only the first.
    check_indexes(proven)
    check_files(args.dists, [row for project in PROJECTS for row in plan["pypi"][project]])
    status, payload = fetch_index(args.project, plan["version"])
    missing = reconcile_index(rows, status, payload, args.project, plan["version"])
    selected = {row["filename"] for row in phase_rows(rows, args.project, args.phase)}
    missing = [row for row in missing if row["filename"] in selected]
    require(not args.out.exists() or not any(args.out.iterdir()), "upload directory must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    for row in missing:
        shutil.copyfile(args.dists / row["filename"], args.out / row["filename"])
    output_values({"upload_required": "true" if missing else "false", "missing_count": str(len(missing))})


def wait_index(args) -> None:
    proven = proof(args.proof)
    plan = proven["plan"]
    rows = plan["pypi"][args.project]
    result = wait_for_index(args.project, plan["version"], rows, fetch_index, timeout=args.timeout,
                            required=phase_rows(rows, args.project, args.phase))
    write_json(args.out, result)


def phase_rows(rows: list[dict], project: str, phase: str) -> list[dict]:
    require(phase in ("all", "native", "pure"), "unknown upload phase")
    require(phase == "all" or project == "gpuwm", "only engine distributions have native/pure phases")
    if phase == "all":
        return rows
    native = lambda row: row["filename"].endswith(("-manylinux_2_28_x86_64.whl", "-win_amd64.whl"))
    return [row for row in rows if native(row) == (phase == "native")]


def finish(args) -> None:
    proven = proof(args.proof)
    plan = proven["plan"]
    release = verify_remote(GitHub(plan["repository"]), proven["captured"], require_public=True)
    indexes = [wait_for_index(project, plan["version"], plan["pypi"][project], fetch_index, timeout=args.timeout) for project in PROJECTS]
    write_json(args.out, {"schema": "gpuwm.prepared-publication-result.v1", "status": "PASS", "tag": plan["tag"],
        "commit": plan["commit"], "release_id": release["id"], "manifest_sha256": proven["captured"]["manifest_sha256"],
        "github_assets": proven["captured"]["assets"], "pypi": plan["pypi"], "index_proofs": indexes,
        "immutable": release.get("immutable", False), "rebuilt": False})


def output_values(values: dict[str, str]) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    for key, value in values.items():
        require("\n" not in value and "\r" not in value, "invalid workflow output")
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            for key, value in values.items():
                stream.write(f"{key}={value}\n")
    else:
        print(json.dumps(values, sort_keys=True))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("capture")
    command.add_argument("--repository", required=True)
    command.add_argument("--tag", required=True)
    command.add_argument("--ref", required=True)
    command.add_argument("--commit", required=True)
    command.add_argument("--event", type=Path, required=True)
    command.add_argument("--event-name", required=True)
    command.add_argument("--manifest-sha256", default="")
    command.add_argument("--stable-release-expected", action="store_true")
    command.add_argument("--immutable-releases-enabled", action="store_true")
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(function=capture)
    command = commands.add_parser("verify")
    command.add_argument("--packet", type=Path, required=True)
    command.add_argument("--repo", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(function=verify)
    command = commands.add_parser("smoke")
    command.add_argument("--packet", type=Path, required=True)
    command.add_argument("--proof", type=Path, required=True)
    command.add_argument("--repo", type=Path, required=True)
    command.add_argument("--platform", choices=("linux-x86_64", "win-x86_64"), required=True)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(function=smoke)
    command = commands.add_parser("promote")
    command.add_argument("--proof", type=Path, required=True)
    command.add_argument("--smokes", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(function=promote)
    command = commands.add_parser("stage-missing")
    command.add_argument("--proof", type=Path, required=True)
    command.add_argument("--dists", type=Path, required=True)
    command.add_argument("--project", choices=PROJECTS, required=True)
    command.add_argument("--phase", choices=("all", "native", "pure"), default="all")
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(function=stage_missing)
    command = commands.add_parser("wait-index")
    command.add_argument("--proof", type=Path, required=True)
    command.add_argument("--project", choices=PROJECTS, required=True)
    command.add_argument("--phase", choices=("all", "native", "pure"), default="all")
    command.add_argument("--timeout", type=float, default=300)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(function=wait_index)
    command = commands.add_parser("finish")
    command.add_argument("--proof", type=Path, required=True)
    command.add_argument("--timeout", type=float, default=300)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(function=finish)
    args = parser.parse_args(argv)
    try:
        args.function(args)
    except (PublicationError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"prepared publication: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
