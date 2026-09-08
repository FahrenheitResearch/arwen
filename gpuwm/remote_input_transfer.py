"""Selected raw input objects over fixed, bounded SSH streams.

Only hashes, byte lengths and explicitly selected paths enter a plan manifest.
The data travels outside the JSON RPC and is never decoded or recomputed here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import sys
import threading
import time

MAX_BLOB_BYTES = 16 * 1024 ** 3
MAX_BUNDLE_BLOB_BYTES = 64 * 1024 ** 3
MAX_BLOBS = 64
MAX_HEADER = 128 * 1024
SHA = re.compile(r"[0-9a-f]{64}\Z")


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _stamp(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def describe(path, role="forcing"):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Selected {role} input '{path.name}' is not a regular file")
    before = _stamp(path)
    if not 0 < before[2] <= MAX_BLOB_BYTES:
        raise ValueError(f"Selected {role} input '{path.name}' is {before[2]:,} bytes; "
                         "one streamed input must be nonempty and at most 16 GiB")
    digest = _hash(path)
    if _stamp(path) != before:
        raise ValueError(f"Selected {role} input '{path.name}' changed while hashing; review again")
    return {"source_path": str(path.resolve()), "size": before[2], "sha256": digest}


def _identity(value):
    if (not isinstance(value, dict) or set(value) != {"size", "sha256"}
            or type(value["size"]) is not int or not 0 < value["size"] <= MAX_BLOB_BYTES
            or not isinstance(value["sha256"], str) or not SHA.fullmatch(value["sha256"])):
        raise ValueError("Streamed input requires its bounded byte length and SHA-256")
    return value


def object_path(workspace, identity, *, require=False):
    from gpuwm.remote_plan import _private_root
    identity = _identity(identity)
    root = _private_root(workspace, ".arwen-input-cache")
    path = root / identity["sha256"]
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if (path.is_symlink() or not stat.S_ISREG(info.st_mode)
                or (hasattr(os, "getuid") and info.st_uid != os.getuid())
                or info.st_size != identity["size"] or _hash(path) != identity["sha256"]):
            raise ValueError("An existing node input-cache object changed; no input was replaced")
    elif require:
        raise ValueError("A selected raw input has not completed its verified node transfer")
    return path


def status(request, workspace):
    if set(request) != {"schema", "action", "workspace", "inputs"}:
        raise ValueError("Unsupported streamed-input status fields")
    values = request.get("inputs")
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_BLOBS:
        raise ValueError("Input status accepts one to sixty-four selected raw objects")
    seen, total, result = set(), 0, []
    for value in values:
        value = _identity(value)
        if value["sha256"] in seen:
            raise ValueError("Input status contains a duplicate raw-object identity")
        seen.add(value["sha256"])
        total += value["size"]
        if total > MAX_BUNDLE_BLOB_BYTES:
            raise ValueError("Selected raw inputs exceed the 64 GiB staging limit")
        result.append({**value, "ready": object_path(workspace, value).is_file()})
    return {"inputs": result}


def receive(request, workspace, source):
    if (not isinstance(request, dict) or set(request) != {"schema", "action", "workspace", "size", "sha256"}
            or request.get("schema") != "gpuwm.remote.request.v1" or request.get("action") != "put-input"):
        raise ValueError("Invalid fixed input-stream request")
    identity = _identity({key: request[key] for key in ("size", "sha256")})
    path = object_path(workspace, identity)
    reused = path.is_file()
    if not reused and shutil.disk_usage(path.parent).free < identity["size"] + 64 * 1024 ** 2:
        raise ValueError("The selected node lacks disk space for this raw input and its staging reserve")
    temporary = path.with_name("." + path.name + "." + secrets.token_hex(8) + ".part")
    digest, copied = hashlib.sha256(), 0
    try:
        with temporary.open("xb") as target:
            while block := source.read(min(1024 * 1024, identity["size"] + 1 - copied)):
                copied += len(block)
                if copied > identity["size"]:
                    raise ValueError("Raw input stream exceeds its declared byte length")
                digest.update(block)
                if not reused:
                    target.write(block)
            target.flush()
            os.fsync(target.fileno())
        if copied != identity["size"] or digest.hexdigest() != identity["sha256"]:
            raise ValueError("Raw input stream failed its complete byte-length/SHA-256 check")
        if not reused:
            # A concurrent successful upload of the same content is harmless.
            # Link is create-only: no process can replace an admitted object.
            try:
                os.link(temporary, path)
            except FileExistsError:
                object_path(workspace, identity, require=True)
                reused = True
        return {**identity, "reused": reused}
    finally:
        if temporary.exists():
            temporary.unlink()
        if path.is_file():
            path.chmod(0o444)


def receive_main():
    from gpuwm import remote_worker as rw
    try:
        if sys.platform != "linux":
            raise ValueError("Remote raw-input streams require Linux")
        header = sys.stdin.buffer.readline(MAX_HEADER + 1)
        if len(header) > MAX_HEADER or not header.endswith(b"\n"):
            raise ValueError("Raw input stream header exceeds its bounded JSON line")
        request = json.loads(header)
        if not isinstance(request, dict):
            raise ValueError("Raw input stream header needs a JSON object")
        result = receive(request, rw._workspace(request), sys.stdin.buffer)
        reply = {"schema": rw.SCHEMA, "ok": True, "action": "put-input", **result}
    except (OSError, ValueError, KeyError) as error:
        reply = {"schema": rw.SCHEMA, "ok": False, "action": "put-input",
                 "error": {"type": type(error).__name__, "message": str(error)[:4000]}}
    sys.stdout.buffer.write(_encoded(reply) + b"\n")
    sys.stdout.buffer.flush()
    return 0 if reply["ok"] else 2


def upload(command, request, path, *, timeout=600):
    """Stream the selected local file while bounding both SSH result pipes."""
    from gpuwm.remote_cli import SCHEMA
    path = Path(path)
    before = _stamp(path)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    chunks, failure = {"stdout": bytearray(), "stderr": bytearray()}, []

    def send():
        try:
            digest, copied = hashlib.sha256(), 0
            process.stdin.write(_encoded(request) + b"\n")
            with path.open("rb") as source:
                while block := source.read(1024 * 1024):
                    copied += len(block)
                    if copied > request["size"]:
                        raise ValueError("Selected raw input grew during transfer")
                    digest.update(block)
                    process.stdin.write(block)
            if copied != request["size"] or digest.hexdigest() != request["sha256"] or _stamp(path) != before:
                raise ValueError("Selected raw input changed during transfer; review again")
        except (BrokenPipeError, ConnectionResetError):
            pass  # The bounded server result owns its early-refusal diagnostic.
        except Exception as error:
            failure.append(error)
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass

    def read(name, maximum):
        pipe = getattr(process, name)
        try:
            while block := pipe.read(4096):
                chunks[name].extend(block[:max(0, maximum + 1 - len(chunks[name]))])
                if len(chunks[name]) > maximum:
                    failure.append(ValueError("SSH input-transfer response exceeds its bounded limit"))
                    break
        finally:
            pipe.close()

    workers = [threading.Thread(target=send, daemon=True),
               threading.Thread(target=read, args=("stdout", MAX_HEADER), daemon=True),
               threading.Thread(target=read, args=("stderr", 16384), daemon=True)]
    for worker in workers:
        worker.start()
    try:
        deadline = time.monotonic() + timeout
        while process.poll() is None or any(worker.is_alive() for worker in workers):
            if failure:
                raise ValueError(str(failure[0]))
            if time.monotonic() >= deadline:
                raise ValueError("Raw input transfer timed out; verified node objects are retained for retry")
            time.sleep(.02)
        if failure:
            raise ValueError(str(failure[0]))
        lines = bytes(chunks["stdout"]).decode("utf-8").splitlines()
        if len(lines) != 1:
            raise ValueError("SSH input transfer returned no complete bounded result: " + bytes(chunks["stderr"]).decode("utf-8", errors="replace")[:2000])
        reply = json.loads(lines[0])
        if (not isinstance(reply, dict) or reply.get("schema") != SCHEMA
                or reply.get("action") != "put-input" or type(reply.get("ok")) is not bool
                or process.returncode != (0 if reply["ok"] else 2)):
            raise ValueError("SSH input transfer result does not match this fixed operation")
        if not reply["ok"]:
            raise ValueError(reply.get("error", {}).get("message", "Node refused the selected raw input"))
        if reply.get("sha256") != request["sha256"] or reply.get("size") != request["size"]:
            raise ValueError("Node admitted a different raw input identity")
        return reply
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        for worker in workers:
            worker.join(timeout=2)


def transfer_bundle_inputs(bundle, command, input_command):
    from gpuwm.remote_cli import _transport
    blobs = bundle.get("blobs", [])
    if not blobs:
        return 0
    identities = {}
    for blob in blobs:
        identities.setdefault(blob["sha256"], {key: blob[key] for key in ("size", "sha256")})
    request = {"schema": "gpuwm.remote.request.v1", "action": "input-status",
               "workspace": bundle["workspace"], "inputs": list(identities.values())}
    reply = _transport(command, request, timeout=120)
    if not reply["ok"]:
        raise ValueError(reply["error"]["message"])
    received = reply.get("inputs")
    if not isinstance(received, list) or len(received) != len(identities):
        raise ValueError("Node input-cache status did not match the selected raw objects")
    ready = {}
    for item in received:
        identity = identities.get(item.get("sha256")) if isinstance(item, dict) else None
        if identity is None or item.get("size") != identity["size"] or type(item.get("ready")) is not bool or item["sha256"] in ready:
            raise ValueError("Node input-cache status changed a selected raw-object identity")
        ready[item["sha256"]] = item["ready"]
    transferred = 0
    for blob in blobs:
        if ready[blob["sha256"]]:
            continue
        upload(input_command, {"schema": request["schema"], "action": "put-input", "workspace": bundle["workspace"],
                               "size": blob["size"], "sha256": blob["sha256"]}, blob["source_path"])
        transferred += blob["size"]
        ready[blob["sha256"]] = True
    return transferred


def source_blobs(bundle):
    records = {}
    for blob in bundle.get("blobs", []):
        records[blob["source_path"]] = {key: blob[key] for key in ("source_path", "size", "sha256")}
    return [records[key] for key in sorted(records)]


def verify_sources(records):
    if not isinstance(records, list) or len(records) > MAX_BLOBS:
        raise ValueError("Selected source-input manifest exceeds its bounded inventory")
    total, seen = 0, set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"source_path", "size", "sha256"}:
            raise ValueError("Invalid selected source-input identity")
        _identity({key: record[key] for key in ("size", "sha256")})
        path = Path(record["source_path"])
        if not path.is_absolute() or str(path) in seen:
            raise ValueError("Selected source inputs need unique absolute paths")
        seen.add(str(path))
        total += record["size"]
        if total > MAX_BUNDLE_BLOB_BYTES:
            raise ValueError("Selected source inputs exceed the 64 GiB staging limit")
        if describe(path) != record:
            raise ValueError(f"Selected raw input '{path.name}' changed after review")
    return hashlib.sha256(_encoded(records)).hexdigest()


def verify_review_file(path):
    with Path(path).open("rb") as source:
        payload = source.read(2 * 1024 * 1024 + 1)
    if len(payload) > 2 * 1024 * 1024:
        raise ValueError("Completed local review exceeds its bounded metadata limit")
    review = json.loads(payload)
    if review.get("schema") != "arwen.companion-remote-review.v1":
        raise ValueError("Raw-input launch validation needs the completed local node review")
    records = review.get("remote_review", {}).get("source_blobs")
    return verify_sources(records)
