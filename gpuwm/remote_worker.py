"""Linux-side, detached remote job service over one bounded stdin/stdout RPC.

No listener, password store, scheduler, or arbitrary-command RPC is installed.
The same interpreter selected by the client owns the durable worker and CLI.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
import tomllib
import traceback

SCHEMA = "gpuwm.remote.result.v1"
TOKEN_ENV = "ARWEN_REMOTE_JOB_TOKEN"
MAX_BYTES = 128 * 1024
JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
TERMINAL = {"completed", "failed", "stopped"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path, maximum=MAX_BYTES):
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError(f"{path.name} exceeds its {maximum}-byte limit")
    return payload


def _json(path):
    value = json.loads(_read(path))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _write(path, value):
    temporary = path.with_name(path.name + "." + secrets.token_hex(6) + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _absolute(value, name):
    if (not isinstance(value, str) or not value or not Path(value).is_absolute()
            or any(ord(char) < 32 for char in value)):
        raise ValueError(f"{name} must be an absolute Linux path")
    return Path(value).resolve()


def _workspace(request):
    path = _absolute(request.get("workspace"), "workspace")
    if not path.is_dir():
        raise ValueError("workspace must already exist on the remote Linux node")
    return path


def _store(workspace, *, create=False):
    path = workspace / ".arwen-jobs"
    if create:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if path.is_symlink() or not path.is_dir() or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError(".arwen-jobs must be a private, owned directory (mode 700), without a symlink")
    return path


def _directory(workspace, identifier):
    if not isinstance(identifier, str) or not JOB_ID.fullmatch(identifier):
        raise ValueError("invalid remote job ID")
    directory = _store(workspace) / identifier
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"remote job {identifier} does not exist in this workspace")
    if directory.stat().st_uid != os.getuid():
        raise ValueError("remote job directory is not owned by this account")
    return directory


def runtime():
    import gpuwm
    return {"version": gpuwm.__version__, "python": sys.executable,
            "python_version": sys.version.split()[0], "module_path": str(Path(gpuwm.__file__).resolve()),
            "platform": sys.platform, "prefix": sys.prefix}


def _process(pid):
    """Identity includes boot, start ticks, argv and UID, never just a PID."""
    try:
        root = Path("/proc") / str(int(pid))
        stat = (root / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": fields[19], "pgid": int(fields[2]),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "argv_sha256": _sha((root / "cmdline").read_bytes()),
                "uid": root.stat().st_uid}
    except (OSError, ValueError, IndexError):
        return None


def _has_token(pid, token):
    try:
        return (TOKEN_ENV + "=" + token).encode() in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\x00")
    except OSError:
        return False


def _owned_processes(token, *, owner_pid=None):
    # The worker is a Linux child subreaper: an orphan stays in its tree even
    # if a native child clears its environment or creates another session.
    # RPC callers use the token only unless they supply the verified owner.
    owner_pid = os.getpid() if owner_pid is None else owner_pid
    identities, parents, owned = {}, {}, set()
    for path in Path("/proc").iterdir():
        if path.name.isdigit() and int(path.name) != owner_pid:
            identity = _process(int(path.name))
            if identity and identity["uid"] == os.getuid():
                pid = identity["pid"]
                identities[pid] = identity
                try:
                    stat = (path / "stat").read_text()
                    parents[pid] = int(stat[stat.rfind(")") + 2:].split()[1])
                except (OSError, ValueError, IndexError):
                    continue
                if _has_token(pid, token):
                    owned.add(pid)
    pending = {owner_pid, *owned}
    while pending:
        descendants = {pid for pid, parent in parents.items() if parent in pending} - owned
        owned.update(descendants)
        pending = descendants
    return [identities[pid] for pid in sorted(owned)]


def _signal_owned(identity, token, signum):
    # A pidfd pins the process between identity verification and the signal;
    # a PID reused during that interval can never receive this job's signal.
    try:
        handle = os.pidfd_open(identity["pid"], 0)
    except ProcessLookupError:
        return False
    try:
        if _process(identity["pid"]) != identity or identity not in _owned_processes(token):
            return False
        try:
            signal.pidfd_send_signal(handle, signum, None, 0)
            return True
        except ProcessLookupError:
            return False
    finally:
        os.close(handle)


def _record(directory):
    record = _json(directory / "job.json")
    if (record.get("schema") != "gpuwm.remote.job.v1" or record.get("id") != directory.name
            or not re.fullmatch(r"[0-9a-f]{64}", record.get("token", ""))):
        raise ValueError("remote job ownership record is invalid")
    return record


def _status(directory):
    record = _record(directory)
    job = {key: record[key] for key in ("id", "created_at", "config", "outdir", "runtime", "action")}
    job["parent_job"] = record.get("parent_job")
    job["state"] = "starting"
    if (directory / "started.json").exists():
        started = _json(directory / "started.json")
        if started.get("token") != record["token"]:
            raise ValueError("remote worker ownership token disagrees with job record")
        job["started_at"] = started["started_at"]
        identity = started["identity"]
        actual = _process(identity["pid"])
        job["state"] = ("running" if actual == identity and _has_token(identity["pid"], record["token"])
                        else "ownership_mismatch" if actual is not None else "lost")
    if (directory / "result.json").exists():
        ended = _json(directory / "result.json")
        if ended.get("token") != record["token"]:
            raise ValueError("remote result ownership token disagrees with job record")
        job.update({key: ended[key] for key in ("state", "exit_code", "ended_at")})
        if ended.get("error"):
            job["error"] = ended["error"]
    return job


def _inputs(source, *, preserve_bound=False, extra_wps=None):
    from gpuwm.hrrr_route_inputs import route_input_paths
    from gpuwm.toml_document import emit_experiment_toml
    payload = _read(source)
    original = tomllib.loads(payload.decode("utf-8"))
    if "experiment" not in original or not isinstance(original.get("domain"), list):
        raise ValueError("remote start requires an ArWen experiment TOML with [[domain]] tables")
    from gpuwm.experiment import build_experiment_from_config_tables
    build_experiment_from_config_tables(original, source=str(source), base_dir=source.parent)
    raw = copy.deepcopy(original)
    if "case_data" in raw and not preserve_bound:
        from gpuwm.case_data import resolved_case_data_paths
        raw["case_data"] = resolved_case_data_paths(raw["case_data"], base_dir=source.parent, source=str(source))
    if "static" in raw and not preserve_bound:
        from gpuwm.static.highres_production import parse_static_table
        highres = parse_static_table(raw["static"], source=str(source), base_dir=source.parent)
        if highres is not None:
            raw["static"]["highres"]["cache_root"] = str(highres.cache_root.resolve())
    sources = {str(source): payload}
    snapshots = {}
    wps_hash = None
    for label, companion in route_input_paths(source).items():
        if not companion.exists():
            continue
        content = _read(companion)
        sources[str(companion)] = content
        if label == "wps_namelist":
            from gpuwm.starter_template import _tiles_wps
            wps_hash = _sha(content)
            if not preserve_bound:
                content = _tiles_wps(companion, source.parent / ".remote-snapshot" / companion.name,
                                     text=content.decode("utf-8")).encode("utf-8")
        snapshots[companion.name] = content
    # A declared-input experiment may name a WPS authority or Vtable elsewhere.
    # Those small scientific companions are captured too; only the large data
    # and geography references continue to name the existing remote inputs.
    if "case_data" in raw and not preserve_bound:
        for label, name in (("wps_namelist", "declared-inputs/namelist.wps"),
                            ("vtable", "declared-inputs/Vtable")):
            companion = Path(raw["case_data"][label])
            content = _read(companion)
            sources[str(companion)] = content
            if label == "wps_namelist":
                from gpuwm.starter_template import _tiles_wps
                wps_hash = _sha(content)
                content = _tiles_wps(companion, source.parent / ".remote-snapshot" / name,
                                     text=content.decode("utf-8")).encode("utf-8")
            snapshots[name] = content
            raw["case_data"][label] = name
    if extra_wps is not None:
        content = _read(extra_wps)
        sources[str(extra_wps)] = content
        snapshots["prepared-inputs/namelist.wps"] = content
        wps_hash = _sha(content)
    snapshots[source.name] = payload if preserve_bound else emit_experiment_toml(raw).encode("utf-8")
    hashes = {name: _sha(value) for name, value in sources.items()}
    binding = _sha(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode())
    return original, sources, snapshots, {"config_sha256": _sha(payload), "wps_sha256": wps_hash,
                                           "input_sha256": binding, "inputs": hashes}


def _checkpoint(old, spec):
    from gpuwm.resume import _check_set, discover_checkpoint_sets, resolve_resume_checkpoint, route_note
    from gpuwm.io.restart import read_restart_header
    from gpuwm.supervisor import validate_manifest_checkpoint
    root = Path(old["outdir"])
    # These are the output routes owned by Go/run-plan, under this job's unique
    # output root. Never glob another job or an arbitrary user's directory.
    directories = [root, root / "run", root / "chain" / "run"]
    if spec != "latest":
        checkpoint = _absolute(spec, "--from")
        if not checkpoint.is_relative_to(root):
            raise ValueError("--from must belong to the source job's recorded output directory")
        validate_manifest_checkpoint(checkpoint)
        candidate = next((item for item in discover_checkpoint_sets(checkpoint.parent)
                          if checkpoint in item.members.values()), None)
        if candidate is not None:
            _check_set(candidate, validate_manifest_checkpoint, read_restart_header)
        elif read_restart_header(checkpoint).get("domain_ids"):
            raise ValueError("tree checkpoint has no discoverable complete checkpoint set")
        return checkpoint
    candidates = []
    failures = []
    for directory in directories:
        if not directory.resolve().is_relative_to(root.resolve()):
            raise ValueError("checkpoint directory escapes the source job's output tree")
        try:
            resolved = resolve_resume_checkpoint(directory, "latest", config=old["snapshot_config"])
            if not resolved.checkpoint.resolve().is_relative_to(root.resolve()):
                raise ValueError("checkpoint escapes the source job's output tree")
            candidates.append(resolved)
        except (OSError, ValueError) as exc:
            failures.append(str(exc))
    if candidates:
        return max(candidates, key=lambda item: (item.checkpoint_set.valid_time, item.checkpoint.stat().st_mtime_ns)).checkpoint
    raise ValueError("No manifest-valid checkpoint exists in this job's recorded output tree" + route_note(old["snapshot_config"])
                     + "; " + "; ".join(failures)[:4000])


def _checkpoint_binding(checkpoint):
    from gpuwm.resume import discover_checkpoint_sets
    candidate = next((item for item in discover_checkpoint_sets(checkpoint.parent)
                      if checkpoint in item.members.values()), None)
    paths = [checkpoint] if candidate is None else candidate.members.values()
    hashes = {str(path): _file_sha(path) for path in paths}
    return {"checkpoint_sha256": hashes[str(checkpoint)], "checkpoint_inputs": hashes,
            "checkpoint_set_sha256": _sha(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode())}


def _prepared_binding(prepared, config, wps, outdir):
    from gpuwm import bridges, stage_cli
    with bridges.inspection_only():
        bundle = stage_cli.resolve_bundle(prepared)
        # Same read-only receipt and digest relay as gpuwm sim --print-command.
        # The actual forecast still validates every cache and authority byte.
        stage_cli.sim_command(bundle, experiment_config=config, wps_namelist=wps, outdir=outdir)
    return {"prepared_document": str(bundle["document"]), "prepared_sha256": _file_sha(bundle["document"]),
            "prepared_layout": bundle["layout"], "prepared_validation": "receipt and digest relay; full content preflight at launch"}


def _review(request, workspace):
    action = request["action"]
    old = None
    if action == "resume":
        directory = _directory(workspace, request.get("job"))
        old = _record(directory)
        if _status(directory)["state"] not in TERMINAL:
            raise ValueError("resume requires a completed, failed, or stopped source job")
        source = Path(old["snapshot_config"])
    else:
        source = _absolute(request.get("config"), "config")
    if not source.is_file():
        raise ValueError("config must already exist on the remote node")
    outdir = _absolute(request.get("outdir"), "outdir")
    if outdir.exists() or outdir.is_symlink():
        raise ValueError("outdir already exists; choose a new remote output directory")
    if not outdir.parent.is_dir():
        raise ValueError("outdir's parent must already exist on the remote node")
    if old and (outdir.is_relative_to(Path(old["outdir"])) or outdir == Path(old["outdir"])):
        raise ValueError("resume output must be outside the source job's output tree")
    prepared = request.get("prepared_root") if old is None else old.get("prepared_root")
    wps = request.get("wps_namelist") if old is None else old.get("snapshot_wps_namelist")
    if old and prepared is None and "case_data" not in tomllib.loads(_read(source).decode("utf-8")):
        from gpuwm.stage_cli import BUNDLE_DOCUMENTS
        candidates = [Path(old["outdir"]) / "prepared", Path(old["outdir"]) / "chain" / "hrrr-root-prep",
                      Path(old["outdir"]) / "chain" / "prepared", Path(old["outdir"]) / "chain" / "prep"]
        prepared = next((str(path) for path in candidates if any((path / name).is_file() for name in BUNDLE_DOCUMENTS)), None)
        if prepared is None:
            # Give checkpointless routes their public, specific refusal first.
            _checkpoint(old, request.get("from_checkpoint", "latest"))
            raise ValueError("source job has no recorded reusable prepared bundle; resume cannot rebuild one implicitly")
        candidate = source.with_suffix(".namelist.wps")
        wps = str(candidate) if candidate.is_file() else None
    if prepared is not None:
        prepared = _absolute(prepared, "prepared_root")
    if wps is not None:
        if prepared is None:
            raise ValueError("wps_namelist requires prepared_root")
        wps = _absolute(wps, "wps_namelist")
        if not wps.is_file():
            raise ValueError("wps_namelist must already exist on the remote node")
    raw, sources, snapshots, binding = _inputs(source, preserve_bound=prepared is not None, extra_wps=wps)
    if prepared is not None:
        binding.update(_prepared_binding(prepared, source, wps, outdir))
    for name in ("config", "wps", "input", "prepared"):
        expected = request.get(f"expected_{name}_sha256")
        if expected is not None and expected != binding.get(f"{name}_sha256"):
            raise ValueError(f"{name} inputs changed since review; review again before starting")
    geog = request.get("geog_root") if old is None else old.get("geog_root")
    if geog is not None:
        geog = str(_absolute(geog, "geog_root"))
        if not Path(geog).is_dir():
            raise ValueError("geog_root must already exist on the remote node")
    products = request.get("products") if old is None else old.get("products")
    if products is not None and (not isinstance(products, str) or len(products) > 16384 or "\x00" in products):
        raise ValueError("products must be a catalog selector string of at most 16384 characters")
    argv = [sys.executable, "-I", "-u", "-m", "gpuwm.cli", "go", str(source), "--outdir", str(outdir), "--run-stamp", "off"]
    if geog:
        argv += ["--geog-root", geog]
    if products is not None:
        argv += ["--products", products]
    if prepared is not None:
        argv += ["--prepared-root", str(prepared)]
    if wps is not None:
        argv += ["--wps-namelist", str(wps)]
    checkpoint = None
    if old:
        checkpoint = _checkpoint(old, request.get("from_checkpoint", "latest"))
        binding.update(_checkpoint_binding(checkpoint))
        for name in ("checkpoint", "checkpoint_set"):
            expected = request.get(f"expected_{name}_sha256")
            if expected is not None and expected != binding[f"{name}_sha256"]:
                raise ValueError(f"{name} changed since review; review again before restarting")
        argv += ["--restart", str(checkpoint)]
    return {"argv": argv, "config": str(source), "outdir": str(outdir), "cwd": str(source.parent if old is None else Path(old["cwd"])),
            "geog_root": geog, "products": products, "checkpoint": None if checkpoint is None else str(checkpoint),
            "prepared_root": None if prepared is None else str(prepared), "wps_namelist": None if wps is None else str(wps),
            "parent_job": None if old is None else old["id"], "runtime": runtime(), **binding}, sources, snapshots


def _launch(request, workspace, *, command_factory=None, worker_command=None):
    """Private test seams are callables; no request key can select a command."""
    review, sources, snapshots = _review(request, workspace)
    if request.get("dry_run", False):
        return {"dry_run": True, "review": review}
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(8)
    directory = _store(workspace, create=True) / identifier
    directory.mkdir(mode=0o700)
    inputs = directory / "inputs"
    originals = directory / "original-inputs"
    snapshot = inputs / Path(review["config"]).name
    argv = list(review["argv"])
    argv[6] = str(snapshot)
    if "--wps-namelist" in argv:
        index = argv.index("--wps-namelist") + 1
        argv[index] = str(inputs / "prepared-inputs" / "namelist.wps")
    if command_factory is not None:
        argv = command_factory(review, snapshot)
    record = {"schema": "gpuwm.remote.job.v1", "id": identifier, "token": secrets.token_hex(32),
              "created_at": _now(), "action": request["action"], **review, "argv": argv,
              "snapshot_config": str(snapshot), "snapshot_sha256": _sha(snapshots[snapshot.name]),
              "snapshot_inputs": {name: _sha(payload) for name, payload in snapshots.items()},
              "snapshot_wps_namelist": (str(inputs / "prepared-inputs" / "namelist.wps") if review["wps_namelist"] is not None else None),
              "original_inputs": {name: f"{index:03d}-{Path(name).name}" for index, name in enumerate(sources)}}
    _write(directory / "job.json", record)
    command = ([sys.executable, "-I", "-u", "-m", "gpuwm.remote_worker"]
               if worker_command is None else worker_command)
    env = dict(os.environ, **{TOKEN_ENV: record["token"]})
    try:
        inputs.mkdir(mode=0o700)
        originals.mkdir(mode=0o700)
        for name, payload in snapshots.items():
            target = inputs / name
            target.parent.mkdir(mode=0o700, exist_ok=True)
            target.write_bytes(payload)
        for name, payload in sources.items():
            (originals / record["original_inputs"][name]).write_bytes(payload)
            if _read(Path(name)) != payload:
                raise ValueError("remote inputs changed while capturing the job; review again before starting")
        # Claim the output exclusively before the worker can write a forecast.
        Path(review["outdir"]).mkdir(mode=0o700)
        with (directory / "job.log").open("ab", buffering=0) as output:
            process = subprocess.Popen([*command, "--run", str(directory), "--token", record["token"]],
                                       cwd=review["cwd"], env=env, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=output, start_new_session=True, close_fds=True)
    except (OSError, ValueError) as exc:
        _write(directory / "result.json", {"token": record["token"], "state": "failed", "exit_code": 2,
                                           "ended_at": _now(), "error": str(exc)})
        raise ValueError(f"job {identifier} failed before launch: {exc}") from exc
    deadline = time.monotonic() + 10
    while not (directory / "started.json").exists():
        if process.poll() is not None:
            _write(directory / "result.json", {"token": record["token"], "state": "failed",
                    "exit_code": process.returncode, "ended_at": _now(), "error": "remote worker failed before startup; read job logs"})
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(.02)
    return {"job": _status(directory)}


def _stop(directory):
    job = _status(directory)
    if job["state"] in TERMINAL:
        return {"job": job}
    if job["state"] != "running":
        raise ValueError(f"cannot prove ownership of this {job['state']} job; no signal was sent")
    record = _record(directory)
    _write(directory / "stop.json", {"token": record["token"], "requested_at": _now()})
    deadline = time.monotonic() + 16
    while time.monotonic() < deadline:
        job = _status(directory)
        if job["state"] in TERMINAL:
            # Wrapper may still be returning after its durable result. Exclude
            # only that verified owner; its child tree must actually be gone.
            owner = _json(directory / "started.json")["identity"]
            if not _owned_processes(record["token"], owner_pid=owner["pid"]):
                return {"job": job}
        time.sleep(.05)
    raise ValueError("stop has not confirmed termination of all owned processes; reconnect and inspect status/logs")


def _logs(directory, request):
    cursor, limit = request.get("cursor", 0), request.get("limit", 16384)
    if type(cursor) is not int or cursor < 0 or type(limit) is not int or not 1 <= limit <= MAX_BYTES:
        raise ValueError("logs requires cursor >= 0 and limit between 1 and 131072")
    path = directory / "job.log"
    size = path.stat().st_size if path.exists() else 0
    if cursor > size:
        raise ValueError("log cursor is past the end of this job's log; retry with cursor 0")
    payload = b""
    if size:
        with path.open("rb") as stream:
            stream.seek(cursor)
            payload = stream.read(min(limit, 16384))
    # Keep a UTF-8 character split across chunks for the next read. At a tiny
    # caller limit, replacement is permitted and the byte cursor still advances.
    if len(payload) > 3 and cursor + len(payload) < size:
        for count in range(4):
            candidate = payload if count == 0 else payload[:-count]
            try:
                candidate.decode("utf-8")
                payload = candidate
                break
            except UnicodeDecodeError as error:
                if error.reason != "unexpected end of data":
                    break
    return {"job": _status(directory), "text": payload.decode("utf-8", errors="replace"),
            "cursor": cursor + len(payload), "eof": cursor + len(payload) >= size}


def dispatch(request):
    if sys.platform != "linux":
        raise ValueError("remote job workers currently require Linux with /proc")
    if not isinstance(request, dict) or request.get("schema") != "gpuwm.remote.request.v1":
        raise ValueError("unsupported remote request schema")
    allowed = {"schema", "action", "workspace", "config", "outdir", "geog_root", "prepared_root", "wps_namelist", "products", "job", "cursor", "limit",
               "from_checkpoint", "dry_run", "expected_config_sha256", "expected_wps_sha256", "expected_input_sha256",
               "expected_checkpoint_sha256", "expected_checkpoint_set_sha256", "expected_prepared_sha256"}
    if set(request) - allowed:
        raise ValueError("unsupported remote request fields: " + ", ".join(sorted(set(request) - allowed)))
    workspace = _workspace(request)
    action = request.get("action")
    if action == "probe":
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise ValueError("remote Python needs Linux pidfd process handles for safe job ownership")
        handle = os.pidfd_open(os.getpid(), 0)
        os.close(handle)
        return {"runtime": runtime(), "workspace": str(workspace),
                "capabilities": {"durable_jobs": True, "existing_remote_inputs": True, "resume": "manifest-valid route checkpoints",
                                 "host_key_verification": "OpenSSH strict known_hosts", "process_handles": "Linux pidfd"}}
    if action in ("start", "resume"):
        if type(request.get("dry_run", False)) is not bool:
            raise ValueError("dry_run must be a boolean")
        return _launch(request, workspace)
    if action == "list":
        limit = request.get("limit", 20)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("list limit must be between 1 and 100")
        store = _store(workspace)
        paths = [] if not store.exists() else sorted((path for path in store.iterdir()
                if JOB_ID.fullmatch(path.name) and path.is_dir() and not path.is_symlink()), reverse=True)
        jobs = []
        for path in paths[:limit]:
            try:
                jobs.append(_status(_directory(workspace, path.name)))
            except (OSError, ValueError, KeyError) as exc:
                jobs.append({"id": path.name, "state": "unreadable", "error": str(exc)[:1000]})
        return {"jobs": jobs}
    if action not in ("status", "logs", "stop"):
        raise ValueError("unsupported remote action")
    directory = _directory(workspace, request.get("job"))
    if action == "status":
        return {"job": _status(directory)}
    if action == "logs":
        return _logs(directory, request)
    return _stop(directory)


def run_worker(directory, token):
    directory = Path(directory).resolve(strict=True)
    record = _record(directory)
    if token != record["token"] or os.environ.get(TOKEN_ENV) != token:
        raise ValueError("worker token does not match its durable job record")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise ValueError("remote Python needs Linux pidfd process handles for safe job ownership")
    handle = os.pidfd_open(os.getpid(), 0)
    os.close(handle)
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot own orphaned remote job descendants")
    _write(directory / "started.json", {"token": token, "identity": _process(os.getpid()), "started_at": _now()})
    code, stopped, error = 1, False, None
    child = None
    try:
        for name, digest in record["snapshot_inputs"].items():
            if _file_sha(directory / "inputs" / name) != digest:
                raise ValueError("captured configuration or companion changed before the worker started")
        if record.get("prepared_document") and _file_sha(Path(record["prepared_document"])) != record["prepared_sha256"]:
            raise ValueError("prepared receipt changed before the worker started; review again")
        for path, digest in record.get("checkpoint_inputs", {}).items():
            if _file_sha(Path(path)) != digest:
                raise ValueError("checkpoint set changed before the worker started; review again")
        child = subprocess.Popen(record["argv"], cwd=record["cwd"], stdin=subprocess.DEVNULL,
                                 start_new_session=True, close_fds=True)
        while child.poll() is None:
            marker = directory / "stop.json"
            if marker.exists() and _json(marker).get("token") == token:
                stopped = True
                break
            time.sleep(.1)
        # Also reap surviving descendants after an unsuccessful parent exit.
        # Every signal rechecks the full process identity and inherited token.
        if stopped or _owned_processes(token):
            for signum, seconds in ((signal.SIGINT, 3), (signal.SIGTERM, 3), (signal.SIGKILL, 3)):
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    owned = _owned_processes(token)
                    if not owned:
                        break
                    for identity in owned:
                        _signal_owned(identity, token, signum)
                    child.poll()
                    time.sleep(.1)
                if not _owned_processes(token):
                    break
        code = child.wait(timeout=1)
        if _owned_processes(token):
            raise RuntimeError("owned descendants still exist after termination attempts")
    except BaseException as exc:
        error = str(exc)
        traceback.print_exc()
        # A failed cleanup is explicitly nonterminal; never claim stop success.
        if _owned_processes(token):
            _write(directory / "cleanup-error.json", {"error": error, "at": _now()})
            return 1
    _write(directory / "result.json", {"token": token, "state": "stopped" if stopped else "completed" if code == 0 and error is None else "failed",
           "exit_code": code, "ended_at": _now(), "error": error})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpc", action="store_true")
    parser.add_argument("--run")
    parser.add_argument("--token")
    args = parser.parse_args(argv)
    if args.run:
        return run_worker(args.run, args.token)
    action = "unknown"
    try:
        if not args.rpc:
            raise ValueError("select --rpc or --run")
        payload = sys.stdin.buffer.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            raise ValueError("remote request exceeds 128 KiB")
        request = json.loads(payload)
        if isinstance(request, dict) and isinstance(request.get("action"), str):
            action = request["action"][:128]
        # The protocol owns stdout even when a schema validator prints a note.
        import contextlib
        with contextlib.redirect_stdout(sys.stderr):
            data = dispatch(request)
        reply = {"schema": SCHEMA, "ok": True, "action": action, **data}
    except Exception as exc:
        reply = {"schema": SCHEMA, "ok": False, "action": action,
                 "error": {"type": type(exc).__name__, "message": str(exc)[:8000]}}
    output = json.dumps(reply, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(output.encode("utf-8")) > MAX_BYTES:
        reply = {"schema": SCHEMA, "ok": False, "action": action,
                 "error": {"type": "ValueError", "message": "remote result exceeds 128 KiB; request fewer jobs"}}
        output = json.dumps(reply, separators=(",", ":"))
    if hasattr(sys.stdout, "buffer"):
        sys.stdout.buffer.write((output + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()
    else:
        print(output, flush=True)
    return 0 if reply["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
