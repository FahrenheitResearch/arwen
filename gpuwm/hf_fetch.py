"""Pinned Hugging Face packages, with environment-only credentials.

A target is an Earth-2 table row or hf://[datasets/]<owner>/<repo>[@revision][/path].
Files and revisions are data, never model-specific download implementations.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

API = "https://huggingface.co"
CHUNK_BYTES = 32 * 1024**2
MAX_CONNECTIONS = 30


@dataclass(frozen=True)
class Target:
    repository: str
    revision: str = "main"
    files: tuple[str | dict, ...] = ()
    repo_type: str = "model"
    prefix: str = ""


def _safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(p in ("", ".", "..") for p in value.split("/")) or "\\" in value:
        raise ValueError(f"unsafe HF path: {value!r}")
    return value


def target_for(value: str, accept_licence=()) -> Target:
    if value.startswith("hf://"):
        tail = value[5:]
        kind = "dataset" if tail.startswith("datasets/") else "model"
        if kind == "dataset":
            tail = tail[len("datasets/"):]
        fields = tail.split("/", 2)
        if len(fields) < 2:
            raise ValueError("hf:// needs <owner>/<repo>[@revision][/path]")
        repo, _, revision = fields[1].partition("@")
        return Target(fields[0] + "/" + repo, unquote(revision) or "main", repo_type=kind,
                      prefix=unquote(fields[2]) if len(fields) == 3 else "")
    from gpuwm.earth2_run import model_table
    for name, row in model_table()["models"].items():
        if value.lower() == name or value.lower() in [a.lower() for a in row.get("aliases", [])]:
            # A table row carries its weight licence: restricted rows need the
            # same explicit opt-in here as on every other door, before any I/O
            # (the ambient licence_scope covers internal fetches by such doors).
            from gpuwm.earth2_licensing import require_licence
            require_licence(name, row, accept_licence=accept_licence)
            weights = row.get("weights", {})
            if weights.get("host") != "huggingface":
                raise ValueError(f"{name}: weights.host is not huggingface")
            return Target(weights["repository"], weights.get("revision", "main"),
                          tuple(weights.get("files", ())), weights.get("repo_type", "model"))
    raise ValueError(f"{value!r} is not an Earth-2 table row or hf:// URL")


class FetchError(RuntimeError):
    """A named repository, HTTP or integrity failure."""


class AccessRefused(FetchError):
    """HTTP 401/403: a credential refusal, not an interrupted transfer."""


def _access_message(repository, status, token):
    credential = ("HF_TOKEN is not set in the environment; set it or pick a public model"
                  if not token else "the environment token has no access; grant access or pick a public model")
    return f"{repository} is gated or private (HTTP {status}) and {credential}"


class _Redirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urlsplit(req.full_url).netloc != urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


class _Transport:
    def __init__(self, repository, connections):
        self.repository = repository
        self.token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.token_used = False
        self.limit = threading.BoundedSemaphore(connections)
        self.lock = threading.Lock()
        self.active = self.peak = self.bytes = 0

    @contextmanager
    def open(self, url, *, headers=None, anonymous=False):
        # urllib has no netrc or Hugging Face token-file credential discovery.
        request_headers = {"Accept-Encoding": "identity", "User-Agent": "gpuwm-hf-fetch/1"}
        request_headers.update(headers or {})
        if self.token and not anonymous:
            request_headers["Authorization"] = f"Bearer {self.token}"
            self.token_used = True
        with self.limit:
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                try:
                    response = build_opener(_Redirects()).open(Request(url, headers=request_headers), timeout=90)
                except HTTPError as exc:
                    status = exc.code
                    exc.close()
                    if status in (401, 403):
                        raise AccessRefused(_access_message(self.repository, status, self.token)) from None
                    raise FetchError(f"{self.repository}: HTTP {status}") from None
                with response:
                    yield response
            finally:
                with self.lock:
                    self.active -= 1

    def count(self, size):
        with self.lock:
            self.bytes += size


def _digest(path, git=False):
    sha = hashlib.sha256()
    blob = hashlib.sha1() if git else None
    if blob:
        blob.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024**2):
            sha.update(block)
            if blob:
                blob.update(block)
    return sha.hexdigest(), blob.hexdigest() if blob else None


def _metadata(target, transport):
    kind = "datasets" if target.repo_type == "dataset" else "models"
    url = f"{API}/api/{kind}/{target.repository}/revision/{quote(target.revision, safe='')}?blobs=true"
    # Resolve public revisions anonymously, even when an environment token exists.
    try:
        with transport.open(url, anonymous=True) as response:
            metadata = json.load(response)
    except AccessRefused:
        if not transport.token:
            raise
        with transport.open(url) as response:
            return json.load(response)
    # The API serves metadata for gated repos anonymously (gated="manual"),
    # and the resolve endpoint serves their README/LICENSE too, so without
    # this check a tokenless fetch of a gated repo downloads those files and
    # only then refuses on the first gated file, leaving .part files behind.
    if not transport.token and (metadata.get("gated") or metadata.get("private")):
        raise AccessRefused(_access_message(target.repository, 401, None))
    return metadata


def _download_range(transport, url, path, start, end, total):
    """Resume a bounded range; .part is retained on interrupted transfers."""
    have = path.stat().st_size if path.exists() else 0
    expected = end - start + 1
    if have > expected:
        path.unlink()
        have = 0
    if have == expected:
        return
    for attempt in range(4):
        try:
            with transport.open(url, headers={"Range": f"bytes={start + have}-{end}"}) as response:
                status = response.status
                if status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or tuple(map(int, match.groups())) != (start + have, end, total):
                        raise FetchError(f"{transport.repository}: invalid Content-Range for {path.name}")
                elif status != 200 or start + have != 0 or end + 1 != total:
                    raise FetchError(f"{transport.repository}: server ignored HTTP Range (HTTP {status})")
                with path.open("ab") as output:
                    while block := response.read(min(1024**2, expected - have + 1)):
                        transport.count(len(block))
                        if have + len(block) > expected:
                            raise FetchError(f"{transport.repository}: oversized range for {path.name}")
                        output.write(block)
                        have += len(block)
            if have == expected:
                return
        except (URLError, TimeoutError, OSError) as exc:
            have = path.stat().st_size if path.exists() else 0
            if attempt == 3:
                raise FetchError(f"{transport.repository}: transfer interrupted ({type(exc).__name__}); rerun to resume") from None
        if attempt < 3:
            time.sleep(0.5 * 2**attempt)
    raise FetchError(f"{transport.repository}: short range for {path.name}; rerun to resume")


@contextmanager
def _cache_lock(path):
    # OS locks release on process exit, including interrupted downloads.
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            while True:
                try:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def fetch(target: Target | str, *, cache_root=None, connections=None, progress=print, accept_licence=()) -> dict:
    """Fetch a pinned package. The returned receipt never contains credentials."""
    if isinstance(target, str):
        target = target_for(target, accept_licence=accept_licence)
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", target.repository) or any(p in (".", "..") for p in target.repository.split("/")):
        raise ValueError("HF repository must be <owner>/<repo>")
    if target.repo_type not in ("model", "dataset"):
        raise ValueError("HF repo_type must be model or dataset")
    connections = int(connections if connections is not None else os.environ.get("GPUWM_HF_CONNECTIONS", MAX_CONNECTIONS))
    if not 1 <= connections <= MAX_CONNECTIONS:
        raise ValueError("HF connections must be between 1 and 30")
    root = Path(cache_root or os.environ.get("GPUWM_HF_CACHE", Path.home() / ".cache/gpuwm/hf")).expanduser().resolve()
    repo_root = root / ("datasets" if target.repo_type == "dataset" else "") / target.repository
    root.mkdir(parents=True, exist_ok=True)
    # A host-wide governor is independent of the data cache, so cold caches
    # and distinct dataset roots cannot each open another pool of 30.
    governor = Path(os.environ.get("GPUWM_HF_GOVERNOR",
                    Path.home() / ".cache/gpuwm/hf/.hf-box.lock")).expanduser()
    governor.parent.mkdir(parents=True, exist_ok=True)
    with _cache_lock(governor):
        with _cache_lock(root / ".hf-fetch.lock"):
            return _fetch_locked(target, repo_root, connections, progress)


def _fetch_locked(target, repo_root, connections, progress):
    started = time.perf_counter()
    transport = _Transport(target.repository, connections)
    metadata = _metadata(target, transport)
    sha = metadata["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise FetchError(f"{target.repository}: API did not return a commit SHA")
    if re.fullmatch(r"[0-9a-f]{40}", target.revision) and sha != target.revision:
        raise FetchError(f"{target.repository}: API revision differs from the requested pin")
    package = repo_root / sha
    created = [p for p in (package, *package.parents) if not p.exists()]
    package.mkdir(parents=True, exist_ok=True)
    receipt_path = repo_root / f"{sha}.receipt.json"
    try:
        previous = json.loads(receipt_path.read_text())
    except (FileNotFoundError, ValueError):
        previous = {}
    previous_files = {f["path"]: f for f in previous.get("files", [])}
    available = {f["rfilename"]: f for f in metadata["siblings"]}
    # Inventory records pin file sizes and (for LFS files) SHA256 as well
    # as the revision. Legacy path-only records and URL prefixes still work.
    declared = {item if isinstance(item, str) else item["path"]:
                {} if isinstance(item, str) else item for item in target.files}
    wanted = list(declared) if target.files else [p for p in available if not target.prefix or p == target.prefix or p.startswith(target.prefix.rstrip("/") + "/")]
    if not wanted:
        raise FetchError(f"{target.repository}: no files match {target.prefix!r} at {sha}")
    wanted = list(dict.fromkeys(wanted))
    specs = []
    for name in wanted:
        _safe_path(name)
        if name not in available:
            raise FetchError(f"{target.repository}: {name} does not exist at {sha}")
        entry = available[name]
        record = declared.get(name, {})
        if "size" in record and record["size"] != entry["size"]:
            raise FetchError(f"{target.repository}: table size differs from API for {name}")
        expected = record.get("sha256")
        if expected is not None and not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise FetchError(f"{target.repository}: invalid table SHA256 for {name}")
        lfs = entry.get("lfs") or {}
        oid = lfs.get("sha256") or lfs.get("oid", "").removeprefix("sha256:") or None
        if lfs and (not oid or not re.fullmatch(r"[0-9a-f]{64}", oid)):
            raise FetchError(f"{target.repository}: API supplied no valid LFS SHA256 for {name}")
        if not lfs and not entry.get("blobId"):
            raise FetchError(f"{target.repository}: API supplied no Git blob checksum for {name}")
        if expected and oid and expected != oid:
            raise FetchError(f"{target.repository}: table SHA256 differs from API for {name}")
        specs.append({"path": name, "size": entry["size"], "lfs_sha256": oid,
                      "table_sha256": expected,
                      "git_blob": None if lfs else entry.get("blobId")})
    results = []
    jobs = []
    plans = []
    # Hash files concurrently, including the second-fetch receipt check.
    def cached(spec):
        path = package / spec["path"]
        old = previous_files.get(spec["path"], {})
        if path.is_file() and path.stat().st_size == spec["size"] and old.get("size") == spec["size"]:
            digest, blob = _digest(path, bool(spec["git_blob"]))
            if digest == old.get("sha256") and (not spec["table_sha256"] or digest == spec["table_sha256"]) and (not spec["lfs_sha256"] or digest == spec["lfs_sha256"]) and (not spec["git_blob"] or blob == spec["git_blob"]):
                return dict(spec, sha256=digest)
        return None
    with ThreadPoolExecutor(max_workers=connections) as workers:
        checked = list(workers.map(cached, specs))
        for spec, hit in zip(specs, checked):
            name, size = spec["path"], spec["size"]
            if hit:
                results.append(hit)
                progress(f"hf-fetch {name}: verified cache, 0 bytes downloaded")
                continue
            dest = package / name
            created += [p for p in (dest.parent, *dest.parent.parents) if not p.exists()]
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            count = max(1, math.ceil(size / CHUNK_BYTES))
            if count > 1 and connections == 1:
                raise ValueError("multi-chunk HF files require at least 2 connections")
            parts = []
            url = f"{API}/{'datasets/' if target.repo_type == 'dataset' else ''}{target.repository}/resolve/{sha}/{quote(name, safe='/')}"
            if size == 0:
                if not part.exists():
                    created.append(part)
                part.touch()
            for index in range(count) if size else ():
                chunk = part if count == 1 else dest.with_name(dest.name + f".chunk-{index}.part")
                parts.append(chunk)
                if not chunk.exists():
                    created.append(chunk)
                start = index * CHUNK_BYTES
                jobs.append(workers.submit(_download_range, transport, url, chunk, start,
                                           min(size, start + CHUNK_BYTES) - 1, size))
            plans.append((spec, dest, part, parts))
        try:
            for job in jobs:
                job.result()
        except AccessRefused:
            # A credential refusal is not an interrupted transfer: nothing
            # will resume these partials, so drop what this call created.
            for job in jobs:
                job.cancel()
            wait(jobs)
            for path in created:
                if path.is_file():
                    path.unlink(missing_ok=True)
            for path in sorted((p for p in created if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                try:
                    path.rmdir()
                except OSError:
                    pass
            raise
        def finish(plan):
            spec, dest, part, parts = plan
            if len(parts) > 1:
                with part.open("wb") as output:
                    for chunk in parts:
                        with chunk.open("rb") as stream:
                            while block := stream.read(4 * 1024**2):
                                output.write(block)
            digest, blob = _digest(part, bool(spec["git_blob"]))
            if part.stat().st_size != spec["size"] or (spec["table_sha256"] and digest != spec["table_sha256"]) or (spec["lfs_sha256"] and digest != spec["lfs_sha256"]) or (spec["git_blob"] and blob != spec["git_blob"]):
                # Corrupt partials cannot be resumed into a correct file.
                for path in set(parts + [part]):
                    path.unlink(missing_ok=True)
                raise FetchError(f"{transport.repository}: size or checksum mismatch for {spec['path']}")
            part.replace(dest)
            for chunk in parts if len(parts) > 1 else ():
                chunk.unlink()
            progress(f"hf-fetch {spec['path']}: verified {spec['size']} bytes sha256={digest}")
            return dict(spec, sha256=digest)
        results.extend(workers.map(finish, plans))
    receipt = {"schema": "gpuwm-hf-fetch-v1", "repo": target.repository,
               "repo_type": target.repo_type, "requested_revision": target.revision,
               "revision_sha": sha, "package_root": str(package),
               "files": sorted(results, key=lambda f: f["path"]),
               "bytes": transport.bytes, "seconds": round(time.perf_counter() - started, 3),
               "connections": transport.peak, "connection_limit": connections,
               "token_used": transport.token_used, "receipt": str(receipt_path)}
    temp = receipt_path.with_suffix(".json.part")
    temp.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temp.replace(receipt_path)
    return receipt


def cli(args):
    try:
        result = fetch(args.target, cache_root=args.cache, connections=args.connections,
                       accept_licence=getattr(args, "accept_licence", ()) or ())
    except FetchError as exc:
        # A documented refusal: one sentence at exit 2, never a traceback.
        import sys
        print(f"gpuwm hf-fetch: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def register_cli(subparsers):
    parser = subparsers.add_parser("hf-fetch", help="fetch a table row or hf:// repo into the pinned gpuwm cache")
    parser.add_argument("target", help="Earth-2 model row or hf://[datasets/]<owner>/<repo>[@revision][/path]")
    parser.add_argument("--cache", help="cache root (default GPUWM_HF_CACHE or ~/.cache/gpuwm/hf)")
    parser.add_argument("--connections", type=int, help="parallel HTTP connections, 1..30 (default 30)")
    from gpuwm.earth2_licensing import add_licence_argument
    add_licence_argument(parser)
    parser.set_defaults(func=cli)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    register_cli(sub)
    args = parser.parse_args()
    try:
        raise SystemExit(args.func(args))
    except (FetchError, ValueError) as exc:
        parser.exit(2, f"hf-fetch: {exc}\n")
