"""OpenSSH client for durable ArWen jobs on an existing Linux installation."""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time

SCHEMA = "gpuwm.remote.result.v1"
MAX_REPLY = 128 * 1024
MAX_REQUEST = 128 * 1024
ACTIONS = ("probe", "start", "list", "status", "logs", "stop", "resume", "review-plan", "start-plan", "sync-artifacts", "artifact-index", "sync-processed-frame")


def result(action: str, *, ok: bool = True, **data) -> dict:
    return {"schema": SCHEMA, "ok": ok, "action": action, **data}


def remote_path(value, name: str) -> str:
    if (not isinstance(value, str) or not value or "\x00" in value
            or any(ord(char) < 32 for char in value)
            or not PurePosixPath(value).is_absolute()):
        raise ValueError(f"{name} must be an absolute Linux path")
    return value


def ssh_command(args, *, artifact_stream=False, input_stream=False, processed_stream=False) -> list[str]:
    host = args.host
    if (not isinstance(host, str) or len(host) > 255
            or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:\[\]-]*", host)):
        raise ValueError("--host must be an SSH alias or user@host without shell syntax")
    python = remote_path(args.python, "--python")
    remote_path(args.workspace, "--workspace")
    ssh = shutil.which("ssh")
    if ssh is None:
        raise ValueError("OpenSSH client 'ssh' is unavailable; install it and retry")
    command = [ssh, "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10",
               "-o", "ServerAliveCountMax=2"]
    if args.port is not None:
        if not 1 <= args.port <= 65535:
            raise ValueError("--port must be between 1 and 65535")
        command += ["-p", str(args.port)]
    for flag, value in (("-F", args.ssh_config), ("-i", args.identity)):
        if value is not None:
            path = Path(value).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"{flag} must name an existing local file")
            command += [flag, str(path)]
    # ssh passes its command through the remote shell. Only these fixed words
    # and the quoted interpreter enter that shell; request data travels on stdin.
    command += ["--", host, shlex.join([python, "-I", "-m", "gpuwm.remote_worker",
                                     "--input-stream" if input_stream else "--processed-stream" if processed_stream else "--artifact-stream" if artifact_stream else "--rpc"])]
    return command


def _transport(command: list[str], request: dict, *, timeout: float = 40) -> dict:
    payload = (json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if len(payload) > MAX_REQUEST:
        raise ValueError("remote request exceeds 128 KiB")
    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    chunks: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    overflow = threading.Event()

    def read(name, stream, maximum):
        try:
            while block := stream.read(4096):
                room = maximum + 1 - len(chunks[name])
                chunks[name].extend(block[:max(0, room)])
                if len(chunks[name]) > maximum:
                    overflow.set()
        finally:
            stream.close()

    readers = [threading.Thread(target=read, args=(name, getattr(process, name), maximum),
                                daemon=True)
               for name, maximum in (("stdout", MAX_REPLY), ("stderr", 16384))]
    for reader in readers:
        reader.start()
    def write_request():
        try:
            process.stdin.write(payload)
            process.stdin.flush()
        except (OSError, ValueError):
            pass  # the SSH exit and protocol response own the diagnostic
        finally:
            process.stdin.close()

    writer = threading.Thread(target=write_request, daemon=True)
    writer.start()
    try:
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if overflow.is_set():
                raise ValueError("remote response exceeds its bounded protocol limit")
            if time.monotonic() >= deadline:
                raise ValueError("SSH request timed out; a start may already exist. List jobs before retrying")
            time.sleep(.02)
        for reader in readers:
            reader.join(timeout=2)
        if overflow.is_set() or any(reader.is_alive() for reader in readers):
            raise ValueError("remote response exceeds its bounded protocol limit or did not close")
        text = bytes(chunks["stdout"]).decode("utf-8", errors="strict")
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) != 1:
            detail = bytes(chunks["stderr"]).decode("utf-8", errors="replace").strip()
            raise ValueError("SSH did not return one ArWen response" + (f": {detail[:2000]}" if detail else ""))
        reply = json.loads(lines[0])
        if (not isinstance(reply, dict) or reply.get("schema") != SCHEMA
                or type(reply.get("ok")) is not bool
                or reply.get("action") != request["action"]):
            raise ValueError("SSH returned an incompatible ArWen response")
        if process.returncode != (0 if reply["ok"] else 2):
            raise ValueError(f"SSH exit {process.returncode} disagrees with the ArWen response")
        return reply
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        writer.join(timeout=2)


def remote_main(args) -> int:
    action = args.remote_action
    try:
        command = ssh_command(args)
        request = {"schema": "gpuwm.remote.request.v1", "action": action,
                   "workspace": args.workspace}
        for name in ("config", "outdir", "geog_root", "prepared_root", "wps_namelist", "products", "job", "cursor",
                     "limit", "from_checkpoint", "dry_run", "expected_config_sha256",
                     "expected_wps_sha256", "expected_input_sha256", "expected_checkpoint_sha256",
                     "expected_checkpoint_set_sha256", "expected_prepared_sha256", "bundle_id",
                     "expected_bundle_sha256", "expected_plan_sha256"):
            value = getattr(args, name, None)
            if value is not None:
                request[name] = value
        if action == "sync-artifacts":
            from gpuwm.remote_artifacts import sync
            reply = result(action, **sync(args, command, ssh_command(args, artifact_stream=True)))
        elif action == "sync-processed-frame":
            from gpuwm.remote_processed import sync
            reply = result(action, **sync(args, command, ssh_command(args, processed_stream=True)))
        elif action == "artifact-index":
            from gpuwm.remote_artifacts import index
            reply = result(action, **index(args, command))
        elif action == "review-plan":
            from gpuwm.remote_plan import build_bundle
            from gpuwm.remote_input_transfer import transfer_bundle_inputs, source_blobs, verify_sources
            bundle = build_bundle(args.plan, workspace=args.workspace, outdir=args.outdir,
                                  geog_root=args.geog_root,
                                  expected_plan_sha256=args.expected_plan_sha256,
                                  expected_config_sha256=args.expected_config_sha256)
            if bundle.get("blobs"):
                transfer_bundle_inputs(bundle, command, ssh_command(args, input_stream=True))
                verify_sources(source_blobs(bundle))
            staged = _transport(command, {"schema": request["schema"], "action": "stage-plan",
                                          "workspace": args.workspace, "bundle": bundle})
            if not staged["ok"]:
                reply = result(action, ok=False, error=staged["error"])
            else:
                if staged.get("bundle_id") != bundle["id"] or staged.get("bundle_sha256") != bundle["sha256"]:
                    raise ValueError("node staged a different plan bundle")
                reply = _transport(command, {"schema": request["schema"], "action": action,
                    "workspace": args.workspace, "bundle_id": bundle["id"],
                    "expected_bundle_sha256": bundle["sha256"]}, timeout=120)
                if reply["ok"] and bundle.get("blobs"):
                    verify_sources(source_blobs(bundle))
        else:
            if action == "start-plan" and getattr(args, "source_inputs_file", None):
                from gpuwm.remote_input_transfer import verify_review_file
                request["expected_source_blobs_sha256"] = verify_review_file(args.source_inputs_file)
            reply = (_transport(command, request, timeout=120) if action == "start-plan"
                     else _transport(command, request))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        reply = result(action, ok=False, error={"type": type(error).__name__, "message": str(error)})
    if args.json:
        output = json.dumps(reply, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        if hasattr(sys.stdout, "buffer"):
            sys.stdout.buffer.write(output.encode("utf-8"))
            sys.stdout.buffer.flush()
        else:
            sys.stdout.write(output)
    elif not reply["ok"]:
        print("remote: " + reply["error"]["message"], file=sys.stderr)
    elif action == "logs":
        print(reply.get("text", ""), end="")
    else:
        print(json.dumps(reply, ensure_ascii=False, indent=2))
    return 0 if reply["ok"] else 2


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser("remote", help="control durable ArWen jobs on an existing Linux SSH node")
    actions = parser.add_subparsers(dest="remote_action", required=True)
    for action in ACTIONS:
        command = actions.add_parser(action, help={
            "probe": "show the remote interpreter and installed ArWen identity",
            "start": "review or start a forecast using existing remote inputs",
            "list": "list recorded jobs in this remote workspace",
            "status": "read a job's durable state", "logs": "read a bounded log chunk",
            "stop": "terminate only the recorded job's owned processes",
            "resume": "review or restart a job's checkpoint into new output",
            "review-plan": "stage selected saved-plan inputs and review actual node memory without a forecast",
            "start-plan": "start the exact reviewed staged plan with the durable job owner",
            "sync-artifacts": "retrieve one exact or latest committed raw WRF frame for a domain",
            "sync-processed-frame": "retrieve a native processed store for one committed domain and forecast time",
            "artifact-index": "read a bounded page of native committed forecast times without opening WRF files"}[action])
        command.add_argument("--host", required=True, help="existing SSH alias or user@host")
        command.add_argument("--python", required=True, help="absolute remote Python path with ArWen installed")
        command.add_argument("--workspace", required=True, help="existing absolute remote workspace directory")
        command.add_argument("--port", type=int, help="SSH port (otherwise SSH configuration applies)")
        command.add_argument("--identity", help="existing local SSH identity path; contents are never copied")
        command.add_argument("--ssh-config", help="existing local OpenSSH configuration path")
        command.add_argument("--json", action="store_true", help="one versioned JSON result line; exit 0 or 2")
        if action in ("status", "logs", "stop", "resume", "sync-artifacts", "artifact-index", "sync-processed-frame"):
            command.add_argument("--job", required=True, help="job ID returned by start or list")
        if action in ("sync-artifacts", "artifact-index", "sync-processed-frame"):
            command.add_argument("--domain", type=int, default=1, help="selected committed domain, 1..999")
        if action == "sync-artifacts":
            command.add_argument("--cache-root", required=True, help="owned local cache for this job's raw frame objects")
            command.add_argument("--sequence", type=int, help="exact native output commit sequence; omit for latest")
            command.add_argument("--reader-leases", action="store_true", help="the visual reader retains shared OS leases for every frame clone")
        if action == "sync-processed-frame":
            command.add_argument("--cache-root", required=True, help="owned local directory for immutable native processed stores")
            command.add_argument("--sequence", type=int, help="exact native output commit sequence; omit for latest")
        if action == "artifact-index":
            command.add_argument("--after-sequence", type=int, default=0, help="last native sequence from the previous timeline page")
        if action in ("start", "resume"):
            command.add_argument("--outdir", required=True, help="new absolute remote output directory; existing paths are refused")
            command.add_argument("--dry-run", action="store_true", help="review inputs and command; create and launch nothing")
            for name in ("config", "wps", "input"):
                command.add_argument(f"--expected-{name}-sha256", help="refuse inputs changed since the reviewed SHA-256")
            command.add_argument("--expected-prepared-sha256", help="refuse a prepared receipt changed since review")
        if action == "start":
            command.add_argument("--config", required=True, help="existing absolute remote experiment TOML")
            command.add_argument("--geog-root", help="existing absolute remote geography directory")
            command.add_argument("--prepared-root", help="existing absolute remote prepared bundle; reuse it without fetch or preparation")
            command.add_argument("--wps-namelist", help="with --prepared-root: exact absolute remote WPS authority required by a single-domain bundle")
            command.add_argument("--products", help="render catalog selectors, all, or none")
        if action == "resume":
            command.add_argument("--from", dest="from_checkpoint", default="latest", help="latest valid checkpoint, or its absolute remote path")
            command.add_argument("--expected-checkpoint-sha256", help="refuse a selected checkpoint changed since review")
            command.add_argument("--expected-checkpoint-set-sha256", help="refuse any checkpoint set member changed since review")
        if action == "logs":
            command.add_argument("--cursor", type=int, default=0, help="byte cursor from the previous log result")
            command.add_argument("--limit", type=int, default=16384, help="maximum bytes requested, 1..131072; each response may return less")
        if action == "list":
            command.add_argument("--limit", type=int, default=20, help="newest 1..100 jobs")
        if action == "review-plan":
            command.add_argument("--plan", required=True, help="saved local run-plan JSON to stage")
            command.add_argument("--outdir", required=True, help="new absolute remote output directory")
            command.add_argument("--geog-root", help="existing remote geography directory")
            command.add_argument("--expected-plan-sha256", required=True)
            command.add_argument("--expected-config-sha256", required=True)
        if action == "start-plan":
            command.add_argument("--bundle-id", required=True)
            command.add_argument("--source-inputs-file", help="completed local review whose selected raw inputs must still match")
            for name in ("bundle", "plan", "config", "input"):
                command.add_argument(f"--expected-{name}-sha256", required=True)
        command.set_defaults(func=remote_main)
