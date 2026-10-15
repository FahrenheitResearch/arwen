"""Detached worker owns one launch token across scheduler death."""
from __future__ import annotations

import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

from gpuwm import proc_identity
from gpuwm.cycle.ledger import write_receipt_atomically
from gpuwm.mcp.gpulock import _os_lock
from .retention import receipt_bytes, prune


def main(path):
    root = Path(path).resolve()
    with _os_lock(root / "worker.mutex"):
        if (root / "result.json").exists():
            return 0
        old = json.loads((root / "worker.json").read_text()) if (root / "worker.json").exists() else {}
        child = old.get("child")
        if child and proc_identity.alive(child, child.get("pid")):
            # A worker lost while its door is still alive cannot guess the exit.
            while proc_identity.alive(child, child.get("pid")):
                time.sleep(1)
            write_receipt_atomically(root / "result.json", {"exit_code": 1, "reason": "worker_lost"})
            return 1
        if old:
            # Lost wrapper with an unpublished child identity is ambiguous. Do
            # not replay a launch whose door may already have changed the world.
            write_receipt_atomically(root / "result.json", {"exit_code": 2,
                "reason": "launch_identity_ambiguous", "remedy": "Inspect this launch token before resuming the row."})
            return 2
        intent = json.loads((root / "intent.json").read_text())
        identity = proc_identity.identify(os.getpid())
        write_receipt_atomically(root / "worker.json", {"worker": identity})
        stopping = False
        def request_stop(signum, frame):
            nonlocal stopping
            stopping = True
        signal.signal(signal.SIGTERM, request_stop)
        with (root / "stdout.log").open("ab") as stdout, (root / "stderr.log").open("ab") as stderr:
            try:
                child_process = subprocess.Popen(intent["argv"], cwd=intent.get("cwd"),
                    stdout=stdout, stderr=stderr, env=dict(os.environ, **intent.get("env", {})), start_new_session=True)
                child_identity = proc_identity.identify(child_process.pid)
                write_receipt_atomically(root / "worker.json", {"worker": identity,
                    "child": child_identity})
                exceeded = None
                pruned = []
                while child_process.poll() is None:
                    if stopping:
                        proc_identity.signal_process(child_identity, signal.SIGTERM, tree=True)
                        try:
                            child_process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc_identity.signal_process(child_identity, signal.SIGKILL, tree=True)
                        break
                    budget = intent.get("receipt_budget_bytes")
                    if budget:
                        size = receipt_bytes(root)
                        scope = intent.get("receipt_scope")
                        global_over = False
                        if scope:
                            scope = Path(scope).resolve()
                            if not root.is_relative_to(scope):
                                raise ValueError("receipt_scope_outside_owner: worker must belong to its suite receipt scope")
                            report = prune(scope, intent["retention"])
                            pruned.extend(report["deleted"])
                            global_over = report["cap_unmet"]
                        if size > budget or global_over:
                            exceeded = size
                            proc_identity.signal_process(child_identity, signal.SIGTERM, tree=True)
                            try:
                                child_process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                proc_identity.signal_process(child_identity, signal.SIGKILL, tree=True)
                            break
                    try:
                        child_process.wait(timeout=.25)
                    except subprocess.TimeoutExpired:
                        pass
                code = child_process.wait()
                result = {"exit_code": code, "finished_unix": time.time()}
                if pruned:
                    result["retention_deleted"] = pruned
                if exceeded is not None:
                    result.update(exit_code=2, reason="receipt_cap_exceeded", error_class="receipt_cap_exceeded",
                                  receipt_bytes=exceeded, receipt_budget_bytes=budget,
                                  remedy="Reduce receipt volume before resuming; active or combined suite receipts exceed the cap after completed-file pruning.")
                if code == 75 and intent.get("admission_marker") and not Path(intent["admission_marker"]).exists():
                    result["error_class"] = "admission_wait"
            except Exception as error:
                result = {"exit_code": 1, "error_class": type(error).__name__, "message": str(error)}
        result["receipts"] = []
        receipts = list(dict.fromkeys([*intent.get("receipts", []), *intent.get("score_files", [])]))
        for pattern in intent.get("receipt_globs", []):
            found = list((root / "output").glob(pattern))
            if not found and result["exit_code"] == 0:
                result.update(exit_code=1, error_class="receipt_missing", message=pattern)
            receipts.extend(str(p) for p in found if p.is_file())
        for receipt in receipts:
            target = Path(receipt)
            if target.is_file():
                import hashlib
                digest = hashlib.sha256()
                with target.open("rb") as handle:
                    for block in iter(lambda: handle.read(1048576), b""):
                        digest.update(block)
                result["receipts"].append({"path": str(target), "sha256": digest.hexdigest()})
            elif result["exit_code"] == 0:
                result.update(exit_code=1, error_class="receipt_missing", message=str(target))
        if result["exit_code"] == 0 and intent.get("score_files"):
            try:
                result["score"] = [json.loads(Path(p).read_text()) for p in intent["score_files"]]
            except (OSError, UnicodeError, ValueError) as error:
                result.update(exit_code=1, error_class="score_receipt_invalid", message=str(error))
        kinds = json.loads((Path(__file__).parents[1] / "data/nwpsuite/door-kinds.v1.json").read_text())
        kind = kinds.get(intent.get("door_kind", "command"))
        if kind is None:
            result.update(exit_code=2, error_class="door_kind_unknown")
        else:
            result["status_class"] = kind["exit_classes"].get(str(result["exit_code"]), "failed")
            if kind.get("receipt_schema"):
                try:
                    documents = [json.loads(Path(x["path"]).read_text()) for x in result["receipts"]]
                except (OSError, UnicodeError, ValueError):
                    documents = []
                valid = [d for d in documents if isinstance(d, dict) and d.get("schema") == kind["receipt_schema"]]
                if result["exit_code"] == 0 and (not valid or any(d.get(kind["success_field"]) not in kind["success_values"] for d in valid)):
                    result.update(exit_code=1, status_class="failed", error_class="receipt_invalid")
                for document in valid:
                    mapped = kind.get("receipt_classes", {}).get(document.get(kind["success_field"]))
                    if mapped and result["exit_code"] != 130:
                        result.update(exit_code={"failed": 1, "refused": 2, "source_behind": 75}[mapped], status_class=mapped)
        events = intent.get("events")
        if events and Path(events).exists() and result.get("error_class") != "receipt_cap_exceeded":
            # Preserve the failed event's typed remedy, not console prose.
            with Path(events).open() as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("event") == "failed":
                        result.update({k: event[k] for k in ("error_class", "remedy", "message") if k in event})
        write_receipt_atomically(root / "result.json", result)
        return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
