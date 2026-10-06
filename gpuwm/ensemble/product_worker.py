"""Replay one immutable diagnostic hour in an owned CPU process.

Only the supervising spool publishes its manifest and retires diagnostic spills.
This process calls the existing Rust product path and returns verified metadata.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading


REQUEST_SCHEMA = "gpuwm-ensemble-product-worker.request.v1"
RESULT_SCHEMA = "gpuwm-ensemble-product-worker.v1"


def _owned_path(root, value, *, existing=True):
    if not isinstance(value, str) or not value or ".." in Path(value).parts:
        raise ValueError("product worker path must stay in its owned output root")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = Path(os.path.abspath(path))
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("product worker path escapes its owned output root") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("product worker paths cannot traverse symlinks")
    if existing and not path.is_file():
        raise ValueError("product worker input or artifact is not an owned file")
    return path


def _signature(state):
    return (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns)


def _identity(path, *, expected=None):
    """Hash a stable descriptor and reject replacement during the read."""
    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        signature = _signature(before)
        if _signature(os.fstat(descriptor)) != signature or _signature(path.stat()) != signature:
            raise ValueError("product worker file changed during its verified hash")
        record = {"bytes": before.st_size, "sha256": digest.hexdigest()}
        if expected is not None and any(record[key] != expected.get(key)
                                        for key in ("bytes", "sha256")):
            raise ValueError("product worker input differs from its committed hash")
        return record, signature
    finally:
        os.close(descriptor)


def _terminate_group(signum, unused_frame):
    # The parent starts a new session. Never signal another process's group if
    # a caller violates that ownership boundary.
    signal.signal(signum, signal.SIG_DFL)
    if hasattr(os, "getpgrp") and os.getpgrp() == os.getpid():
        os.killpg(os.getpgrp(), signum)
    os._exit(128 + signum)


def _parent_guard(parent_pid):
    if type(parent_pid) is not int or parent_pid <= 1:
        raise ValueError("product worker needs its exact supervising parent PID")
    signal.signal(signal.SIGTERM, _terminate_group)
    if os.getppid() != parent_pid:
        raise RuntimeError("product worker parent ended before child startup")
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        if library.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot bind product worker parent-death signal")
        # The parent can die between the initial check and prctl.
        if os.getppid() != parent_pid:
            _terminate_group(signal.SIGTERM, None)


def _cpu_policy():
    os.environ.update(CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "2"
    if hasattr(os, "setpriority"):
        os.setpriority(os.PRIO_PROCESS, 0, max(10, os.getpriority(os.PRIO_PROCESS, 0)))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    if pending.is_symlink():
        raise ValueError("product worker result staging cannot be a symlink")
    with pending.open("w", encoding="utf-8") as destination:
        destination.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        destination.flush()
        os.fsync(destination.fileno())
    pending.replace(path)
    if os.name == "posix":
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def execute_request(request):
    """Use the existing spool implementation without touching parent authority."""
    if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unrecognized product worker request")
    root_value = request.get("root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise ValueError("product worker needs an absolute owned output root")
    root = Path(root_value)
    if root.is_symlink():
        raise ValueError("product worker output root cannot be a symlink")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("product worker output root is not a directory")
    domain = request.get("domain")
    if not isinstance(domain, str) or not re.fullmatch(r"d[0-9]{2}(?:-episode-[0-9]{3,})?(?:-grid-[0-9a-f]{12})?", domain):
        raise ValueError("product worker needs its exact domain identity")
    valid = request.get("valid_time")
    if not isinstance(valid, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}[T_][0-9]{2}:[0-9]{2}:[0-9]{2}(?:Z|\+00:00)?", valid):
        raise ValueError("product worker needs its exact native UTC clock")
    order = request.get("member_order")
    if (not isinstance(order, list) or not order or len(set(order)) != len(order)
            or any(type(member) is not int or not 0 <= member < 1 << 64 for member in order)):
        raise ValueError("product worker needs its complete original member roster")
    shape = request.get("shape")
    if not isinstance(shape, list) or len(shape) != 2 or any(type(size) is not int or size < 1 for size in shape):
        raise ValueError("product worker coordinates need the exact positive two-dimensional grid")
    coordinate = request.get("coordinate_file")
    if not isinstance(coordinate, dict):
        raise ValueError("product worker requires its committed coordinate file")
    coordinate_path = _owned_path(root, coordinate.get("path"))
    _, coordinate_identity = _identity(coordinate_path, expected=coordinate)
    frame = deepcopy(request.get("frame"))
    if (not isinstance(frame, dict) or frame.get("status") != "pending"
            or frame.get("valid_time") != valid or frame.get("members_received") != sorted(order)):
        raise ValueError("product worker requires one immutable complete-roster pending hour")
    packs = frame.get("packs")
    if not isinstance(packs, list) or not packs:
        raise ValueError("product worker requires its committed diagnostic spills")
    received = []
    source_paths = {coordinate_path}
    for pack in packs:
        if not isinstance(pack, dict) or not isinstance(pack.get("member_ids"), list):
            raise ValueError("product worker diagnostic spill omitted its member roster")
        path = _owned_path(root, pack.get("path"))
        if path in source_paths:
            raise ValueError("product worker diagnostic sources must be distinct")
        source_paths.add(path)
        _identity(path, expected=pack)
        received.extend(pack["member_ids"])
        pack["path"] = path
    if sorted(received) != sorted(order) or any(type(member) is not int for member in received):
        raise ValueError("product worker spill roster differs from its original ensemble")
    available = request.get("available_bytes")
    if type(available) is not int or available < 0:
        raise ValueError("product worker needs its admitted available-byte envelope")

    # These imports do not initialize a forecast model or a CUDA context.
    import numpy as np
    from gpuwm import rustwx
    from gpuwm.netcdf_bridge import open_dataset
    from gpuwm.ensemble.batch_product_output import NativeDiagnosticSpool
    from gpuwm.ensemble.batch_products import FieldProducts, ThresholdCondition

    requests = tuple(FieldProducts(**dict(row, thresholds=tuple(row.get("thresholds", ()))))
                     for row in request.get("requests", ()))
    if not requests or len({row.field for row in requests}) != len(requests):
        raise ValueError("product worker requires unique original field requests")
    events = request.get("events") or {}
    if not isinstance(events, dict):
        raise ValueError("product worker compound event metadata must be a mapping")
    for name, conditions in events.items():
        if not isinstance(name, str) or not isinstance(conditions, list) or not conditions:
            raise ValueError("product worker compound event needs its original conditions")
        for condition in conditions:
            ThresholdCondition(**condition)
    tile_rows = request.get("tile_rows", 32)
    if type(tile_rows) is not int or tile_rows < 2:
        raise ValueError("product worker needs the original positive replay tile size")
    with open_dataset(coordinate_path) as dataset:
        latitude = np.asarray(dataset.variables["XLAT"][:]).copy()
        longitude = np.asarray(dataset.variables["XLONG"][:]).copy()
    if (_signature(coordinate_path.stat()) != coordinate_identity
            or latitude.shape != tuple(shape) or longitude.shape != tuple(shape)):
        raise ValueError("product worker coordinates changed their committed identity or grid")
    renderer_value = request.get("renderer")
    renderer = Path(renderer_value) if renderer_value else rustwx.find_renderer()
    if renderer is None or not Path(renderer).is_file():
        raise ValueError("product worker requires the installed native ensemble renderer")
    spool = NativeDiagnosticSpool.__new__(NativeDiagnosticSpool)
    spool.root, spool.domain, spool.renderer = root, domain, Path(renderer)
    spool.members, spool.member_order = len(order), tuple(order)
    spool.member_positions = {member: position for position, member in enumerate(order)}
    spool.member_metadata = tuple(request.get("member_metadata", ()))
    spool.requests, spool.events = requests, deepcopy(events)
    spool.latitude, spool.longitude, spool.shape = latitude, longitude, tuple(shape)
    spool.gpu_replay, spool.tile_rows = False, tile_rows
    spool._lock, spool._processing = threading.RLock(), set()
    spool.frames, spool.deleted, spool.member_files = {valid: frame}, [], []
    spool.keep_member_files, spool.unavailable_products = True, ()
    spool.coordinate_file = dict(coordinate, path=coordinate_path.relative_to(root).as_posix())
    spool.manifest_path = root / domain / "ensemble-manifest.json"
    spool._manifest = lambda: None
    row = spool._finish(valid, available_bytes=available,
        render_products=tuple(request.get("render_products", ("mean", "spread", "min", "max", "prob", "paintball", "postage"))),
        retire_diagnostics=False)
    artifacts = []
    for value in (*row.get("products", ()), *row.get("maps", ())):
        path = _owned_path(root, value)
        identity, _ = _identity(path)
        artifacts.append(dict(identity, path=path.relative_to(root).as_posix()))
    if len({record["path"] for record in artifacts}) != len(artifacts):
        raise ValueError("product worker returned duplicate artifact identities")
    # Original spills stay available until the parent verifies these records and
    # commits the complete frame. The parent owns their eventual retirement.
    for pack in packs:
        _identity(pack["path"], expected=pack)
    return {"schema": RESULT_SCHEMA, "domain": domain, "valid_time": valid,
            "row": row, "deletedScratch": deepcopy(spool.deleted),
            "artifactRecords": artifacts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("descriptor", type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.descriptor.is_symlink():
        raise ValueError("product worker request cannot be a symlink")
    request = json.loads(args.descriptor.read_text(encoding="utf-8"))
    _parent_guard(request.get("parent_pid"))
    _cpu_policy()
    root = Path(request["root"]).resolve(strict=True)
    descriptor = _owned_path(root, str(args.descriptor.absolute()))
    result_path = _owned_path(root, str(args.result.absolute()), existing=False)
    sources = [request.get("coordinate_file", {}), *request.get("frame", {}).get("packs", ())]
    if (result_path == descriptor or result_path.suffix != ".json"
            or any(result_path == _owned_path(root, row.get("path")) for row in sources)):
        raise ValueError("product worker result cannot replace its request or diagnostic source")
    result = execute_request(request)
    _atomic_json(result_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
