"""Python orchestration of the Rust unsigned public-object seam.

No HTTP, payload hashing or transfer pool lives here. Object completion is
consumed as NDJSON arrives, allowing receipts to precede the end of a batch.
"""
from functools import wraps
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import uuid

from gpuwm.timeline.ledger import emit


def timed(stage):
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            attrs = {"span_id": uuid.uuid4().hex, "thread": threading.get_ident()}
            emit(stage, "begin", product="da", attrs=attrs)
            status = "failed"
            try:
                result = function(*args, **kwargs)
                status = "failed" if isinstance(result, dict) and result.get("status") == "failed" else "complete"
                return result
            finally:
                emit(stage, "end", product="da", attrs={**attrs, "status": status})
        return measured
    return decorate


def events(command, request):
    from gpuwm.rustwx_fetch import find_fetch_bin, fetch_remedy
    binary = find_fetch_bin()
    if binary is None:
        raise ValueError("Rust public-object fetch is required: " + fetch_remedy())
    origin = os.environ.get("GPUWM_RW_FETCH_TEST_ORIGIN")
    if origin:
        request = {**request, "origin": origin}
    with tempfile.TemporaryDirectory(prefix="da-object-request-") as scratch:
        source = Path(scratch) / "request.json"
        source.write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
        with (Path(scratch) / "stderr.log").open("w+") as errors:
            process = subprocess.Popen([str(binary), command, "--request", str(source)],
                                       stdout=subprocess.PIPE, stderr=errors, text=True)
            fault = None
            try:
                for line in process.stdout:
                    event = json.loads(line)
                    if event.get("phase") == "error":
                        fault = event["error"]
                    else:
                        yield event
            finally:
                process.stdout.close()
                status = process.wait()
            errors.seek(0)
            reason = errors.read().strip()
            if status or fault:
                raise ValueError(f"Rust public-object {command} refused: {fault or reason}")


def download(rows):
    for event in events("objects", {"objects": rows}):
        if event["phase"] == "complete":
            yield event["object"]


def start_truth(inv, root, receipt, max_gib):
    """Detach scoring inputs. Neither completion nor failure delays forecast."""
    from tools.da_recent_observations import save, document_sha
    receipt = Path(receipt)
    request = receipt.with_name(receipt.name + ".request.json")
    rows = [row for row in inv["objects"] if row["bucket"] == "noaa-mrms-pds"]
    save(request, {"objects": rows, "source_inventory_sha256": document_sha(inv)})
    save(receipt, {"schema": "da-rerun.truth-fetch.v1", "status": "pending",
                   "inventory_sha256": document_sha(inv)})
    log = receipt.with_name(receipt.name + ".log")
    try:
        with log.open("a") as output:
            process = subprocess.Popen([os.sys.executable, "-m", "tools.da_object_fetch",
                                        str(request), str(Path(root) / "truth"), str(receipt),
                                        str(max_gib)], stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=output, start_new_session=True, close_fds=True)
    except OSError as error:
        save(receipt, {"schema": "da-rerun.truth-fetch.v1", "status": "failed",
                       "inventory_sha256": document_sha(inv), "error": str(error)})
        return {"receipt": str(receipt), "log": str(log), "pid": None, "error": str(error)}
    emit("fetch.truth", "mark", product="da", attrs={"action": "background-start",
         "pid": process.pid, "receipt": str(receipt), "log": str(log)})
    return {"receipt": str(receipt), "log": str(log), "pid": process.pid}


@timed("fetch.truth")
def truth_worker(request, root, receipt, max_gib):
    from tools.da_recent_observations import fetch, save
    request = json.loads(Path(request).read_text())
    child = Path(receipt).with_name(Path(receipt).name + ".objects.json")
    try:
        ancestor = None
        fingerprint = None
        if child.is_file():
            from tools.da_recent_observations import sha
            ancestor = json.loads(child.read_text())
            fingerprint = sha(child)
        # The worker already has a fetch.truth interval. Do not mislabel its
        # background transfer as the forecast's critical fetch.inputs stage.
        fetch.__wrapped__({"objects": request["objects"]}, root, child, max_gib,
              reuse_receipt=ancestor, reuse_receipt_sha256=fingerprint)
        result = {"status": "complete", "fetch_receipt": json.loads(child.read_text())}
    except Exception as error:
        result = {"status": "failed", "error": str(error)}
        emit("fetch.truth", "mark", product="da", attrs=result)
    save(receipt, {"schema": "da-rerun.truth-fetch.v1",
                   "inventory_sha256": request["source_inventory_sha256"], **result})
    return result


def fetch_binding(fetched):
    """Stable truth-job identity while lazy radar moves from planned to landed."""
    from tools.da_recent_observations import document_sha
    if fetched.get("schema") == "da-rerun.recent-fetch.v2":
        value = {key: fetched.get(key) for key in ("schema", "inventory_sha256", "scope", "raw_root")}
        value["truth_receipt"] = (fetched.get("truth") or {}).get("receipt")
        return document_sha(value)
    return document_sha(fetched)


def launch_truth_for_forecast(inv, fetched, out, center_lon, center_lat):
    """Keep duplicate/stale pack refusals in verification, outside forecast."""
    from tools.da_recent_observations import start_truth, save
    try:
        return start_truth(inv, fetched, out, center_lon, center_lat, wait_claim=False)
    except (OSError, ValueError) as error:
        refusal = {"status": "refused", "error": str(error),
                   "prevents": "duplicate or stale truth pack writers and unverified scoring"}
        save(Path(out).parent / (Path(out).name + "-launch-refusal.json"), refusal)
        emit("prep.obs.truth", "mark", product="da", attrs=refusal)
        return refusal


def truth_receipt(inv, forecast, *, wait=False, timeout_seconds=1800):
    from tools.da_recent_observations import document_sha
    path = Path(forecast["truth"]["receipt"])
    deadline = time.monotonic() + timeout_seconds
    while True:
        result = json.loads(path.read_text())
        if result.get("status") != "pending" or not wait or time.monotonic() >= deadline:
            break
        time.sleep(.25)
    if (result.get("schema") != "da-rerun.truth-fetch.v1"
            or result.get("inventory_sha256") != document_sha(inv)):
        raise ValueError("background truth receipt differs from frozen inventory")
    if result.get("status") != "complete":
        raise ValueError("background truth is not complete; scoring cannot read missing or unverified truth: "
                         + str(result.get("error", result.get("status"))))
    return {**result["fetch_receipt"], "inventory_sha256": document_sha(inv), "scope": "truth"}


if __name__ == "__main__":
    import sys
    truth_worker(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))
