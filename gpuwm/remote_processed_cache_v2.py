"""Individual native-store member transfer with a leased, bounded local cache."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import secrets

from gpuwm import remote_artifacts as ra, remote_processed as legacy
from gpuwm import remote_processed_v2 as viewer
from gpuwm.remote_artifact_cache import Lease, _owned_directory


def _relative(value):
    if (not isinstance(value, str) or not value or "\\" in value
            or any(ord(character) < 32 for character in value)):
        raise ValueError("Native viewer member needs a safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")) or ":" in value:
        raise ValueError("Native viewer member path contains traversal or an absolute path")
    return path


def _members(value):
    rows = value.get("members")
    if not isinstance(rows, list) or not 1 <= len(rows) <= legacy.MAX_FILES:
        raise ValueError("Node returned an invalid native viewer member inventory")
    names, keys, total = set(), set(), 0
    for row in rows:
        name = _relative(row.get("relative_path")).as_posix()
        key, size = row.get("key"), row.get("bytes")
        if (not isinstance(key, str) or not viewer.SLUG.fullmatch(key) or key in keys or name in names
                or type(size) is not int or not 0 < size <= viewer.MAX_MEMBER_BYTES
                or not ra.HEX.fullmatch(str(row.get("sha256")))
                or row.get("grid_sha256") != value["frame"].get("grid_sha256")):
            raise ValueError("Node returned an invalid member size, digest or geographic identity")
        keys.add(key); names.add(name); total += size
    if total > viewer.MAX_PUBLICATION_BYTES or value.get("bytes") != total:
        raise ValueError("Node native viewer publication exceeds its transfer bound")
    return rows


def _used(directory):
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _prune(root, limit, incoming, protected, *, reader_leases):
    objects, leases = _owned_directory(root / "objects"), _owned_directory(root / "leases")
    entries = []
    for count, path in enumerate(objects.iterdir()):
        if count >= viewer.MAX_ENTRIES or path.is_symlink() or not path.is_dir() or not ra.HEX.fullmatch(path.name):
            raise ValueError("Local viewer cache contains an unexpected object")
        lock = leases / (path.name + ".lock")
        entries.append(((lock if lock.exists() else path).stat().st_mtime_ns, path, _used(path)))
    used = sum(size for _age, _path, size in entries)
    if incoming > limit:
        raise viewer.Backpressure("The selected viewer artifact exceeds this local cache budget; increase it or request fewer products")
    if not reader_leases and used + incoming > limit:
        raise viewer.Backpressure("The viewer must retain native-store reader leases before its local cache can rotate")
    for _age, path, size in sorted(entries):
        if used + incoming <= limit:
            break
        if path.name == protected:
            continue
        with Lease(leases / (path.name + ".lock")) as lease:
            if lease.file is None:
                continue
            try:
                viewer._remove_owned_tree(path, objects)
            except PermissionError:
                continue
            used -= size
    if used + incoming > limit:
        raise viewer.Backpressure("The local viewer cache is retained by active frames; release a frame or increase its budget")
    return used


def _validate_local(result, value, destination, lease_path):
    authority = result.get("remote_source", {})
    if any(authority.get(key) != value[key] for key in ("job_id", "run_id", "domain", "sequence", "source_sha256", "run_manifest", "commit", "publication_sha256")):
        raise ValueError("Local viewer cache belongs to a different committed forecast frame")
    legacy._check_identity(result, value["run_id"], {"domain": value["domain"], "valid_time": value["valid_time"]}, value["source_sha256"])
    if (result["frame"]["identity"] != value["frame"]["identity"]
            or result["frame"].get("grid_sha256") != value["frame"].get("grid_sha256")
            or result["frame"].get("cache_lease_path") != str(lease_path)):
        raise ValueError("Local viewer identity, geometry or reader lease changed")
    for member in result.get("files", []):
        path = ra._inside(member["path"], destination)
        if path.stat().st_size != member["bytes"] or ra._file_sha(path) != member["sha256"]:
            raise ValueError("Local viewer member failed its byte identity check")
    for path in (result["frame"]["hour_path"], result["grid_path"], result["run_json_path"], result["receipt_path"]):
        ra._inside(path, destination)


def _download_members(value, request, stream_command, staging, destination, lease_path):
    store = _owned_directory(staging / "store")
    final_store = destination / "store"
    remote_root = value["native_store_root"]
    transferred = 0
    for row in value["members"]:
        relative = _relative(row["relative_path"])
        path = store.joinpath(*relative.parts)
        _owned_directory(path.parent)
        member_request = {**{key: item for key, item in request.items() if key != "prefetch_sequences"}, "action": "stream-processed-member-v2", "sequence": value["sequence"],
                          "member_key": row["key"], "expected_member_sha256": row["sha256"],
                          "expected_publication_sha256": value["publication_sha256"],
                          "expected_commit_sha256": value["commit"]["sha256"],
                          "expected_manifest_sha256": value["run_manifest"]["sha256"]}
        ra._download(stream_command, member_request, path, {"sha256": row["sha256"], "size_bytes": row["bytes"]}, timeout=600)
        if path.stat().st_size != row["bytes"] or ra._file_sha(path) != row["sha256"]:
            raise ValueError("Downloaded native viewer member failed integrity verification")
        transferred += row["bytes"]
    local = legacy._rebase(value["native_result"], remote_root, final_store)
    local["frame"]["cache_lease_path"] = str(lease_path)
    local.update(cache_lease_path=str(lease_path), object_root=str(destination), cache_root=str(destination.parent.parent))
    for row in value["members"]:
        if not row["relative_path"].endswith(".json"):
            continue
        path = store.joinpath(*_relative(row["relative_path"]).parts)
        metadata, _ = ra._raw(path, viewer.MAX_METADATA_BYTES)
        metadata = legacy._rebase(metadata, remote_root, final_store)
        if metadata.get("schema") == local["frame"].get("schema") and metadata.get("identity") == local["frame"].get("identity"):
            metadata["cache_lease_path"] = str(lease_path)
        legacy._write(path, metadata)
    local["files"] = [{**row, "path": str(final_store.joinpath(*_relative(row["relative_path"]).parts)),
                       "sha256": ra._file_sha(store.joinpath(*_relative(row["relative_path"]).parts)),
                       "bytes": store.joinpath(*_relative(row["relative_path"]).parts).stat().st_size}
                      for row in value["members"]]
    local["members"] = local["files"]
    local["remote_members"] = value["members"]
    local["remote_source"] = {key: value[key] for key in ("job_id", "run_id", "domain", "sequence", "source_sha256", "run_manifest", "commit", "publication_sha256")}
    legacy._write(staging / "native-result.json", local)
    return transferred


def sync(args, command, stream_command):
    from gpuwm.remote_cli import _transport
    products = getattr(args, "products", None)
    products = products.split(",") if isinstance(products, str) else products
    selection = viewer._selection(getattr(args, "profile", viewer.PROFILE), products)
    request = {"schema": "gpuwm.remote.request.v1", "action": "processed-frame-v2", "workspace": args.workspace,
               "job": args.job, "domain": ra._domain(args.domain), "profile": selection["profile"], "products": selection["products"] or viewer.DEFAULT_PRODUCTS}
    if args.sequence is not None:
        request["sequence"] = ra._sequence(args.sequence)
    prefetch = getattr(args, "prefetch_sequences", None)
    if isinstance(prefetch, str):
        prefetch = [int(value) for value in prefetch.split(",") if value]
    if prefetch is not None:
        if not isinstance(prefetch, list) or len(prefetch) > viewer.MAX_PREFETCH:
            raise ValueError("Explicit loop prefetch supports at most eight committed frames")
        request["prefetch_sequences"] = [ra._sequence(value) for value in prefetch]
    expected_run = getattr(args, "expected_run_id", None)
    if expected_run is not None:
        request["expected_run_id"] = expected_run
    reply = _transport(command, request, timeout=120)
    if not reply["ok"]:
        raise ValueError(reply["error"]["message"])
    value = reply.get("processed_frame")
    if (not isinstance(value, dict) or value.get("schema") != viewer.SCHEMA or value.get("job_id") != args.job
            or value.get("domain") != args.domain or type(value.get("waiting")) is not bool
            or value.get("profile") != selection["profile"]
            or expected_run is not None and value.get("run_id") not in (None, expected_run)):
        raise ValueError("Node returned a viewer frame for another job, run, domain or processing profile")
    if args.sequence is not None and value.get("sequence") != args.sequence:
        raise ValueError("Node selected another committed viewer frame sequence")
    if value["waiting"]:
        return {"processed_frame": value, "transferred_bytes": 0}
    if not ra.HEX.fullmatch(str(value.get("publication_sha256"))):
        raise ValueError("Node viewer publication has no immutable SHA-256 identity")
    legacy._check_identity(value["native_result"], value["run_id"], {"domain": value["domain"], "valid_time": value["valid_time"]}, value["source_sha256"])
    if value["native_result"]["frame"] != value["frame"]:
        raise ValueError("Node viewer receipt differs from its selected frame")
    _members(value)
    root = _owned_directory(legacy._local_cache_path(args.cache_root))
    objects, leases = _owned_directory(root / "objects"), _owned_directory(root / "leases")
    key = value["publication_sha256"]
    destination, lease_path = objects / key, leases / (key + ".lock")
    result_path = destination / "native-result.json"
    limit = viewer._cache_bytes(getattr(args, "cache_bytes", None) or viewer.DEFAULT_CACHE_BYTES)
    transferred = 0
    with Lease(root / "writer.lock", timeout=180) as writer:
        if writer.file is None:
            raise ValueError("Another native viewer cache transaction is still active")
        if result_path.exists():
            # Reading immutable existing data is allowed while the GUI retains
            # shared leases. Only creation/recovery/eviction needs exclusivity.
            local, _ = ra._raw(result_path, viewer.MAX_METADATA_BYTES)
            _validate_local(local, value, destination, lease_path)
        else:
            _prune(root, limit, value["bytes"] + viewer.MAX_METADATA_BYTES, key,
                   reader_leases=getattr(args, "reader_leases", False))
            with Lease(lease_path, timeout=5) as lease:
                if lease.file is None:
                    raise viewer.Backpressure("Selected native viewer object is still retained by an active reader")
                staging = _owned_directory(root / (".stage-" + secrets.token_hex(12)))
                try:
                    transferred = _download_members(value, request, stream_command, staging, destination, lease_path)
                    if destination.exists():
                        viewer._remove_owned_tree(destination, objects)
                    os.rename(staging, destination)
                finally:
                    if staging.exists():
                        viewer._remove_owned_tree(staging, root)
            local, _ = ra._raw(result_path, viewer.MAX_METADATA_BYTES)
            _validate_local(local, value, destination, lease_path)
        if lease_path.exists():
            os.utime(lease_path, None)
        cache_used = _used(objects)
    value.update(local_result_path=str(result_path), native_result=local, frame=local["frame"],
                 local_members=local["files"], transferred_bytes=transferred,
                 cache_lease_path=str(lease_path), cache_root=str(root), object_root=str(destination),
                 local_cache_bytes=cache_used, local_cache_limit_bytes=limit)
    return {"processed_frame": value, "transferred_bytes": transferred}
