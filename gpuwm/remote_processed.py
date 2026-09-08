"""Queue native WRF-to-store work and transport its immutable byte bundles.

Python reads ownership metadata, starts the existing Rust processor, and copies
files. Meteorological fields and derived quantities belong to rw_wrfbatch.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import subprocess
import sys
import tarfile
import time

from gpuwm import remote_artifacts as ra
from gpuwm.remote_artifact_cache import Lease, _owned_directory

SCHEMA = "arwen.remote-processed-frame.v1"
QUEUE_SCHEMA = "arwen.native-store-queue.v1"
MAX_BUNDLE = 4 * 1024**3
MAX_FILES = 16


def _write(path, value):
    payload = ra._encoded(value)
    temporary = path.with_name("." + path.name + "." + secrets.token_hex(8) + ".part")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _root(workspace):
    return _owned_directory(Path(workspace) / ".arwen-processed")


def _job(workspace, job):
    from gpuwm import remote_worker as rw
    directory = rw._directory(workspace, job)
    record, state = rw._record(directory), rw._status(directory)
    if record.get("action") != "start-plan" or not record.get("snapshot_plan"):
        raise ValueError("Native stores require this job's saved run-plan manifest")
    bound = ra.bound_manifest(record, state)
    commits = []
    if bound is not None:
        root, _path, manifest, _bytes, started, _binding = bound
        path = ra._inside(manifest.get("events_path"), root)
        if path.stat().st_size > ra.MAX_EVENTS:
            raise ValueError("Native output commit stream exceeds its metadata limit")
        previous, scanned = 0, 0
        with path.open("rb") as stream:
            for index in range(ra.MAX_RECORDS + 1):
                line = stream.readline(ra.MAX_LINE + 1)
                if not line:
                    break
                scanned += len(line)
                if index == ra.MAX_RECORDS or len(line) > ra.MAX_LINE or scanned > ra.MAX_EVENTS:
                    raise ValueError("Native output commit stream exceeds its record limit")
                if not line.endswith(b"\n"):
                    break
                event = json.loads(line)
                if (not isinstance(event, dict) or event.get("schema_version") != "gpuwm.run-plan.event.v1"
                        or type(event.get("sequence")) is not int or event["sequence"] <= previous):
                    raise ValueError("Native output commit sequence is invalid")
                previous = event["sequence"]
                if event.get("event") != "output_committed":
                    continue
                emitted = event.get("emitted_unix_ms")
                if type(emitted) is not int or emitted < started:
                    raise ValueError("Native output commit predates its bound run")
                ra._domain(event.get("domain"))
                ra._timestamp(event.get("valid_time"))
                commits.append((event, ra._authority(path, line, sequence=previous)))
    return record, state, bound, commits


def _entry_path(root, job, sequence):
    directory = _owned_directory(root / job / "entries")
    return directory / f"{ra._sequence(sequence):012d}.json"


def _load_entry(root, job, event, authority):
    path = _entry_path(root, job, event["sequence"])
    if not path.exists():
        return None
    value, _ = ra._raw(path, 256 * 1024)
    if value.get("commit_sha256") != authority["sha256"] or value.get("sequence") != event["sequence"]:
        raise ValueError("Native store receipt disagrees with its committed source")
    return value


def _processor_identity():
    from gpuwm.render import require_renderer
    path = require_renderer().resolve(strict=True)
    return {"path": str(path), "stamp": list(ra._stamp(path))}


def ensure(workspace, job, *, priority=None):
    """Register all present/future commits; one process owns this workspace queue."""
    from gpuwm import remote_worker as rw
    directory = rw._directory(workspace, job)
    record = rw._record(directory)
    if record.get("action") != "start-plan":
        return
    root = _root(workspace)
    requests = _owned_directory(root / "requests")
    marker = requests / (job + ".json")
    prior = ra._raw(marker, 4096)[0] if marker.exists() else None
    request = {"schema": QUEUE_SCHEMA, "job_id": job}
    if prior and isinstance(prior.get("priority"), dict):
        request["priority"] = prior["priority"]
    if priority is not None:
        request["priority"] = {"domain": ra._domain(priority["domain"]), "sequence": ra._sequence(priority["sequence"])}
    if request != prior:
        _write(marker, request)
    status_path = root / job / "status.json"
    if status_path.exists():
        status = ra._raw(status_path, 65536)[0]
        if status.get("done") is True:
            if not status.get("failed") or status.get("processor") == _processor_identity():
                return
    with Lease(root / "worker.lock") as lease:
        if lease.file is None:
            return
        environment = dict(os.environ)
        environment.pop(rw.TOKEN_ENV, None)
        environment.update(GPUWM_NO_LOCAL_GPU="1", CUDA_VISIBLE_DEVICES="-1",
                           RAYON_NUM_THREADS="2", OMP_NUM_THREADS="2")
        with (root / "worker.log").open("ab", buffering=0) as log:
            subprocess.Popen([sys.executable, "-I", "-m", "gpuwm.remote_processed",
                              "--workspace", str(workspace)], cwd=str(workspace), env=environment,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                             start_new_session=True, close_fds=True)


def _native_files(result, store_root):
    if not isinstance(result, dict) or result.get("schema") != "arwen.wrf-process-result.v1":
        raise ValueError("Native processor did not publish its supported result schema")
    rows = result.get("files")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_FILES:
        raise ValueError("Native processor returned an invalid portable file list")
    files, total = {}, 0
    for row in rows:
        path = ra._inside(row.get("path"), store_root)
        size = path.stat().st_size
        digest = row.get("sha256")
        if (path.suffix not in (".rws", ".rwg", ".json") or type(row.get("bytes")) is not int
                or row["bytes"] != size or not ra.HEX.fullmatch(str(digest)) or ra._file_sha(path) != digest):
            raise ValueError("Native portable file identity or digest is invalid")
        relative = path.relative_to(store_root).as_posix()
        if relative in files:
            raise ValueError("Native processor returned a duplicate portable file")
        files[relative] = {"path": str(path), "sha256": digest, "bytes": size}
        total += size
    if not 0 < total <= MAX_BUNDLE or not any(name.endswith(".rws") for name in files):
        raise ValueError("Native portable store exceeds its bounded transfer size")
    return files


def _check_identity(result, run_id, event, source_sha256=None):
    identity = result.get("frame", {}).get("identity", {})
    # Time and domain come from the native WRF reader; validate its response
    # against the immutable model commit before offering it to any viewer.
    if (result.get("domain") != f"d{event['domain']:02d}"
            or identity.get("source") != "arwen" or identity.get("model") != f"wrf-d{event['domain']:02d}"
            or identity.get("case_id") != run_id or identity.get("member") is not None
            or type(identity.get("valid_unix")) is not int
            or identity["valid_unix"] * 1000 != ra._timestamp(event["valid_time"])
            or (source_sha256 is not None and identity.get("source_sha256") != source_sha256)):
        raise ValueError("Native processed frame does not match its job, domain and committed UTC")


def _convert(root, record, bound, event, authority):
    from gpuwm.render import require_renderer
    producer_root = bound[0]
    source = ra._inside(event.get("path"), producer_root)
    stamp = ra._stamp(source)
    if not 0 < stamp[2] <= 16 * 1024**3 or event.get("size_bytes", stamp[2]) != stamp[2]:
        raise ValueError("Native committed WRF size is invalid or changed")
    digest = ra._file_sha(source)
    if ra._stamp(source) != stamp:
        raise ValueError("Native committed WRF changed while deriving its store identity")
    directory = _owned_directory(root / record["id"] / "objects" / f"d{event['domain']:02d}" / digest)
    store_root = _owned_directory(directory / "native")
    result_path, request_path = directory / "native-result.json", directory / "native-request.json"
    request = {"schema": "arwen.wrf-process-request.v1", "path": str(source), "source_sha256": digest,
               "case_id": bound[2]["run_id"], "domain": f"d{event['domain']:02d}",
               "valid_utc": ra.datetime.fromtimestamp(ra._timestamp(event["valid_time"]) / 1000,
                   ra.timezone.utc).isoformat().replace("+00:00", "Z"),
               "store_root": str(store_root), "heavy_ecape": False}
    _write(request_path, request)
    with (directory / "native.log").open("ab", buffering=0) as log:
        result = subprocess.run([str(require_renderer()), "--process-request", str(request_path),
                                 "--process-result", str(result_path)], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=log, timeout=3600, check=False)
    if result.returncode != 0:
        raise ValueError(f"Native WRF processing exited {result.returncode}; see {directory / 'native.log'}")
    if ra._stamp(source) != stamp:
        raise ValueError("Native committed source changed during processing")
    result, _ = ra._raw(result_path, 256 * 1024)
    _check_identity(result, bound[2]["run_id"], event, digest)
    files = _native_files(result, store_root)
    portable = {"schema": "arwen.native-store-bundle.v1", "remote_store_root": str(store_root),
                "native_result": result, "files": files}
    archive = directory / "portable.tar"
    temporary = directory / (".portable-" + secrets.token_hex(8) + ".tar")
    try:
        with tarfile.open(temporary, "w") as bundle:
            import io
            metadata = ra._encoded(portable)
            member = tarfile.TarInfo("bundle.json"); member.size = len(metadata); member.mode = 0o600
            bundle.addfile(member, io.BytesIO(metadata))
            for name, item in files.items():
                bundle.add(item["path"], arcname="store/" + name, recursive=False)
        if temporary.stat().st_size > MAX_BUNDLE:
            raise ValueError("Native store archive exceeds the transfer limit")
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    return {"schema": SCHEMA, "state": "ready", "sequence": event["sequence"], "domain": event["domain"],
            "source_sha256": digest, "source_stamp": list(stamp), "source_path": str(source),
            "commit_sha256": authority["sha256"], "frame": result["frame"],
            "archive_path": str(archive), "archive_sha256": ra._file_sha(archive),
            "archive_bytes": archive.stat().st_size, "native_result_path": str(result_path)}


def _work_job(workspace, job):
    from gpuwm import remote_worker as rw
    root = _root(workspace)
    directory = _owned_directory(root / job)
    record, state, bound, commits = _job(workspace, job)
    ready, failed = 0, 0
    processor = _processor_identity()
    status = {"schema": QUEUE_SCHEMA, "job_id": job, "committed": len(commits),
              "ready": 0, "failed": 0, "state": "waiting", "done": False, "processor": processor}
    pending = {event["sequence"]: (event, authority) for event, authority in commits}
    visited = set()
    while pending:
        # A live forecast may commit another time while Rust works. Refresh
        # only its bounded event metadata so that a newly selected time can
        # become the very next conversion even during historical backfill.
        record, state, bound, commits = _job(workspace, job)
        pending.update({event["sequence"]: (event, authority) for event, authority in commits
                        if event["sequence"] not in visited})
        status["committed"] = len(commits)
        sequence = min(pending)
        marker = root / "requests" / (job + ".json")
        if marker.exists():
            priority = ra._raw(marker, 4096)[0].get("priority", {})
            candidate = pending.get(priority.get("sequence"))
            if candidate and candidate[0]["domain"] == priority.get("domain"):
                sequence = priority["sequence"]
        event, authority = pending.pop(sequence)
        visited.add(sequence)
        entry = _load_entry(root, job, event, authority)
        if entry is not None and entry.get("state") == "failed" and entry.get("processor") != processor:
            entry = None
        if entry is None:
            status.update(state="processing", domain=event["domain"], sequence=event["sequence"], ready=ready, failed=failed)
            _write(directory / "status.json", status)
            try:
                entry = _convert(root, record, bound, event, authority)
            except Exception as error:
                entry = {"schema": SCHEMA, "state": "failed", "sequence": event["sequence"],
                         "commit_sha256": authority["sha256"], "processor": processor, "error": str(error)[:2000]}
            _write(_entry_path(root, job, event["sequence"]), entry)
        ready += entry.get("state") == "ready"
        failed += entry.get("state") == "failed"
    done = state["state"] in rw.TERMINAL
    status.update(ready=ready, failed=failed, done=done,
                  state="failed" if failed else "complete" if done else "waiting_for_output")
    status.pop("domain", None); status.pop("sequence", None)
    _write(directory / "status.json", status)
    return done


def worker(workspace):
    from gpuwm import remote_worker as rw
    root = _root(workspace)
    with Lease(root / "worker.lock", timeout=3) as lease:
        if lease.file is None:
            return 0
        if hasattr(os, "nice"):
            os.nice(10)
        while True:
            pending = False
            requests = sorted((root / "requests").glob("*.json"))[:1024]
            for marker in requests:
                if marker.is_symlink():
                    continue
                request, _ = ra._raw(marker, 4096)
                job = request.get("job_id")
                if request.get("schema") != QUEUE_SCHEMA or not isinstance(job, str) or not rw.JOB_ID.fullmatch(job) or marker.name != job + ".json":
                    continue
                try:
                    pending |= not _work_job(workspace, job)
                except Exception as error:
                    directory = _owned_directory(root / job)
                    _write(directory / "status.json", {"schema": QUEUE_SCHEMA, "job_id": job,
                           "state": "failed", "done": True, "error": str(error)[:2000]})
            if not pending:
                return 0
            time.sleep(2)


def index_metadata(workspace, job):
    root = _root(workspace)
    path = root / job / "status.json"
    return ra._raw(path, 65536)[0] if path.exists() else {"schema": QUEUE_SCHEMA, "state": "queued", "job_id": job}


def catalog(request, workspace, *, start=True):
    fields = {"schema", "action", "workspace", "job", "domain", "sequence"}
    if set(request) - fields:
        raise ValueError("Unsupported native store request fields")
    job, domain = request.get("job"), ra._domain(request.get("domain", 1))
    sequence = ra._sequence(request["sequence"]) if request.get("sequence") is not None else None
    record, _state, bound, commits = _job(workspace, job)
    selected = [(event, authority) for event, authority in commits
                if event["domain"] == domain and (sequence is None or event["sequence"] == sequence)]
    if start:
        ensure(workspace, job, priority={"domain": domain, "sequence": selected[-1][0]["sequence"]} if selected else None)
    root = _root(workspace)
    value = {"schema": SCHEMA, "job_id": job, "domain": domain, "sequence": sequence,
             "waiting": True, "processing": index_metadata(workspace, job)}
    if bound is None:
        return value
    _producer, manifest_path, manifest, manifest_bytes, _started, binding = bound
    value.update(run_id=manifest["run_id"], run_manifest=ra._authority(manifest_path, manifest_bytes),
                 remote_output_root=record["outdir"], remote_pid=manifest["pid"])
    if binding is not None:
        value["producer_binding"] = binding
    if not selected:
        return value
    event, authority = selected[-1]
    value.update(sequence=event["sequence"], commit=authority, valid_time=event["valid_time"])
    entry = _load_entry(root, job, event, authority)
    if entry is None:
        return value
    if entry.get("state") == "failed":
        value["processing"] = {**value["processing"], "state": "failed", "error": entry.get("error")}
        return value
    source = ra._inside(entry["source_path"], bound[0])
    if list(ra._stamp(source)) != entry["source_stamp"]:
        raise ValueError("Native WRF changed after the processed store was published")
    _check_identity({"frame": entry["frame"], "domain": f"d{domain:02d}"}, manifest["run_id"], event, entry["source_sha256"])
    archive = ra._inside(entry["archive_path"], root)
    if archive.stat().st_size != entry["archive_bytes"]:
        raise ValueError("Native portable store archive changed after publication")
    value.update(waiting=False, source_sha256=entry["source_sha256"], frame=entry["frame"],
                 archive={"sha256": entry["archive_sha256"], "size_bytes": entry["archive_bytes"], "remote_path": str(archive)})
    return ra._bounded(value)


def stream(request, workspace, output):
    fields = {"schema", "action", "workspace", "job", "domain", "sequence", "expected_archive_sha256", "expected_commit_sha256", "expected_manifest_sha256"}
    if set(request) != fields or request.get("action") != "stream-processed-frame":
        raise ValueError("Invalid native store stream request")
    for key in ("expected_archive_sha256", "expected_commit_sha256", "expected_manifest_sha256"):
        if not ra.HEX.fullmatch(str(request[key])):
            raise ValueError("Invalid native store stream digest")
    value = catalog({key: value for key, value in request.items() if not key.startswith("expected_")}, workspace, start=False)
    if (value["waiting"] or value["archive"]["sha256"] != request["expected_archive_sha256"]
            or value["commit"]["sha256"] != request["expected_commit_sha256"]
            or value["run_manifest"]["sha256"] != request["expected_manifest_sha256"]):
        raise ValueError("Native store authority changed before transfer")
    archive = value["archive"]; path = Path(archive["remote_path"]); before = ra._stamp(path)
    import hashlib
    digest, copied = hashlib.sha256(), 0
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            copied += len(block)
            if copied > archive["size_bytes"]:
                raise ValueError("Native store archive grew during transfer")
            digest.update(block); output.write(block)
    output.flush()
    if copied != archive["size_bytes"] or digest.hexdigest() != archive["sha256"] or ra._stamp(path) != before:
        raise ValueError("Native store archive changed during transfer")


def stream_main():
    from gpuwm import remote_worker as rw
    try:
        if sys.platform != "linux":
            raise ValueError("Remote native store streams require Linux")
        payload = sys.stdin.buffer.read(rw.MAX_BYTES + 1)
        if len(payload) > rw.MAX_BYTES:
            raise ValueError("Native store stream request exceeds its metadata limit")
        request = json.loads(payload)
        if not isinstance(request, dict) or request.get("schema") != "gpuwm.remote.request.v1":
            raise ValueError("Native store stream request is invalid")
        stream(request, rw._workspace(request), sys.stdout.buffer)
        return 0
    except (OSError, ValueError, KeyError) as error:
        print("remote native store: " + str(error)[:4000], file=sys.stderr)
        return 2


def _rebase(value, remote_root, local_root):
    if isinstance(value, dict):
        return {key: _rebase(item, remote_root, local_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_rebase(item, remote_root, local_root) for item in value]
    normalized_root = remote_root.replace("\\", "/").rstrip("/")
    if isinstance(value, str) and value.replace("\\", "/").startswith(normalized_root + "/"):
        relative = PurePosixPath(value.replace("\\", "/")).relative_to(PurePosixPath(normalized_root))
        if ".." in relative.parts:
            raise ValueError("Native metadata path contains traversal")
        return str(local_root.joinpath(*relative.parts))
    return value


def _unpack(archive, destination, expected, final_destination=None):
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        if len(members) > MAX_FILES + 1 or len({m.name for m in members}) != len(members):
            raise ValueError("Native store archive has too many or duplicate files")
        for member in members:
            parts = PurePosixPath(member.name)
            if (not member.isfile() or parts.is_absolute() or ".." in parts.parts or "\\" in member.name
                    or member.size < 0 or member.size > MAX_BUNDLE):
                raise ValueError("Native store archive contains an unsafe member")
        metadata = bundle.extractfile("bundle.json")
        if metadata is None:
            raise ValueError("Native store archive lacks its metadata")
        payload = metadata.read(256 * 1024 + 1)
        if len(payload) > 256 * 1024:
            raise ValueError("Native store metadata exceeds its limit")
        description = json.loads(payload)
        if description.get("schema") != "arwen.native-store-bundle.v1":
            raise ValueError("Unknown native store bundle schema")
        files = description.get("files", {})
        if set(m.name for m in members) != {"bundle.json", *("store/" + name for name in files)}:
            raise ValueError("Native store archive disagrees with its exact file manifest")
        result = description["native_result"]
        _check_identity(result, expected["run_id"], {"domain": expected["domain"], "valid_time": expected["valid_time"]}, expected["source_sha256"])
        if result["frame"] != expected["frame"]:
            raise ValueError("Native bundle frame differs from its selected commit metadata")
        store = _owned_directory(destination / "store")
        final_store = (final_destination or destination) / "store"
        for name, row in files.items():
            member = bundle.getmember("store/" + name)
            if type(row.get("bytes")) is not int or member.size != row["bytes"] or not ra.HEX.fullmatch(str(row.get("sha256"))):
                raise ValueError("Native store member size or SHA is invalid")
            path = store.joinpath(*PurePosixPath(name).parts)
            _owned_directory(path.parent)
            with bundle.extractfile(member) as source, path.open("xb") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
            if ra._file_sha(path) != row["sha256"]:
                raise ValueError("Native store member failed its SHA-256 check")
        local = _rebase(result, description["remote_store_root"], final_store)
        # These files contain only native store paths/field manifests. Rebase
        # paths for portability, retaining every native field/value descriptor.
        for name in files:
            if name.endswith(".json"):
                path = store.joinpath(*PurePosixPath(name).parts)
                value, _ = ra._raw(path, 4 * 1024 * 1024)
                _write(path, _rebase(value, description["remote_store_root"], final_store))
        local["files"] = [{"path": str(final_store / path.relative_to(store)), "sha256": ra._file_sha(path), "bytes": path.stat().st_size}
                          for path in (store.joinpath(*PurePosixPath(name).parts) for name in files)]
        local["remote_source"] = {key: expected[key] for key in ("job_id", "run_id", "domain", "sequence", "source_sha256", "run_manifest", "commit")}
        _write(destination / "native-result.json", local)
        return destination / "native-result.json"


def _local_cache_path(path):
    """Keep the native store's full identity path on Windows as well as Linux."""
    path = Path(path)
    value = str(path)
    if os.name == "nt" and path.is_absolute() and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
        return Path(value)
    return path


def sync(args, command, stream_command):
    from gpuwm.remote_cli import _transport
    request = {"schema": "gpuwm.remote.request.v1", "action": "processed-frame", "workspace": args.workspace,
               "job": args.job, "domain": ra._domain(args.domain)}
    if args.sequence is not None:
        request["sequence"] = ra._sequence(args.sequence)
    reply = _transport(command, request, timeout=120)
    if not reply["ok"]:
        raise ValueError(reply["error"]["message"])
    value = reply.get("processed_frame")
    if (not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("job_id") != args.job
            or value.get("domain") != args.domain or type(value.get("waiting")) is not bool):
        raise ValueError("Node returned native store metadata for a different job/domain")
    if value["waiting"]:
        return {"processed_frame": value, "transferred_bytes": 0}
    archive = value["archive"]
    if not ra.HEX.fullmatch(str(archive.get("sha256"))) or type(archive.get("size_bytes")) is not int or not 0 < archive["size_bytes"] <= MAX_BUNDLE:
        raise ValueError("Node native store archive size or SHA is invalid")
    if args.sequence is not None and value["sequence"] != args.sequence:
        raise ValueError("Node selected a different native commit sequence")
    root = _owned_directory(_local_cache_path(args.cache_root))
    destination = root / archive["sha256"]
    transferred = 0
    with Lease(root / (archive["sha256"] + ".lock"), timeout=180) as lease:
        if lease.file is None:
            raise ValueError("Another transfer still owns this native store cache object")
        result_path = destination / "native-result.json"
        if destination.exists() or destination.is_symlink():
            _owned_directory(destination)
        if result_path.is_symlink():
            raise ValueError("Native store result cache must not be a symlink")
        if not result_path.exists():
            archive_path = root / (archive["sha256"] + ".tar")
            stream_request = {**request, "action": "stream-processed-frame", "sequence": value["sequence"],
                "expected_archive_sha256": archive["sha256"], "expected_commit_sha256": value["commit"]["sha256"],
                "expected_manifest_sha256": value["run_manifest"]["sha256"]}
            ra._download(stream_command, stream_request, archive_path, archive, timeout=600)
            transferred = archive["size_bytes"]
            staging = _owned_directory(root / (".unpack-" + secrets.token_hex(8)))
            try:
                _unpack(archive_path, staging, value, final_destination=destination)
                os.rename(staging, destination)
            finally:
                if staging.exists():
                    if staging.resolve().parent != root or not staging.name.startswith(".unpack-"):
                        raise ValueError("Refusing cleanup outside the owned native store staging directory")
                    shutil.rmtree(staging)
                archive_path.unlink(missing_ok=True)
        local, _ = ra._raw(result_path, 256 * 1024)
        authority = local.get("remote_source", {})
        if any(authority.get(key) != value[key] for key in ("job_id", "run_id", "domain", "sequence", "source_sha256", "run_manifest", "commit")):
            raise ValueError("Local native store cache disagrees with the selected immutable source")
        _check_identity(local, value["run_id"], {"domain": value["domain"], "valid_time": value["valid_time"]}, value["source_sha256"])
        if local["frame"]["identity"] != value["frame"]["identity"]:
            raise ValueError("Local native store identity changed after publication")
        for path in (local["frame"]["hour_path"], local["grid_path"], local["run_json_path"], local["receipt_path"]):
            ra._inside(path, destination)
    value.update(local_result_path=str(result_path), frame=local["frame"], transferred_bytes=transferred)
    return {"processed_frame": value, "transferred_bytes": transferred}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args(argv)
    from gpuwm.remote_worker import _workspace
    return worker(_workspace({"workspace": args.workspace}))


if __name__ == "__main__":
    raise SystemExit(main())
