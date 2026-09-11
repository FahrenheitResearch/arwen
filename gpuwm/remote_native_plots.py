"""Incremental native PNGs from verified compact stores, with bounded retrieval."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gpuwm import remote_artifacts as ra, remote_processed as legacy, remote_processed_v2 as viewer
from gpuwm.remote_artifact_cache import Lease, _owned_directory

SCHEMA = "arwen.native-plots.v1"
STATUS_SCHEMA = "arwen.native-plot-progress.v1"
MAX_PANEL_BYTES = 32 * 1024**2
MAX_GALLERY_BYTES = 256 * 1024**2


def _root(workspace, job):
    return _owned_directory(viewer._directory(viewer._root(workspace), job) / "native-plots")


def _receipt(root, sequence):
    return root / f"{ra._sequence(sequence):012d}.json"


def status(workspace, job):
    path = _root(workspace, job) / "status.json"
    return ra._raw(path, 64 * 1024)[0] if path.exists() else {
        "schema": STATUS_SCHEMA, "job_id": job, "state": "waiting_for_output", "ready": 0, "failed": 0}


def _published(root, job, event, authority):
    path = _receipt(root, event["sequence"])
    if not path.exists():
        return None
    value, _ = ra._raw(path, viewer.MAX_METADATA_BYTES)
    if (value.get("schema") != SCHEMA or value.get("job_id") != job
            or value.get("domain") != event["domain"] or value.get("sequence") != event["sequence"]
            or value.get("commit_sha256") != authority["sha256"]):
        raise ValueError("Native plots no longer match the committed forecast frame")
    return value


def _render(root, record, bound, event, authority, entry, spacing):
    from gpuwm.render import require_renderer
    source = ra._inside(entry["source_path"], bound[0])
    if list(ra._stamp(source)) != entry["source_stamp"]:
        raise ValueError("WRF output changed after compact preparation")
    legacy._check_identity(entry["native_result"], bound[2]["run_id"], event, entry["source_sha256"])
    products = [row["slug"] for row in entry["products"] if row["available"]]
    output = root / f"frame-{event['sequence']:012d}"
    request_path, result_path = root / f"request-{event['sequence']:012d}.json", root / f"render-{event['sequence']:012d}.json"
    request = {"schema": "arwen.native-store-render-request.v1",
        "process_result": str(Path(entry["object_root"]) / "result.json"),
        "expected_frame_id": entry["frame"]["id"], "expected_source_sha256": entry["source_sha256"],
        "out_dir": str(output), "products": products, "spacing_m": spacing, "width": 1200, "height": 900}
    legacy._write(request_path, request)
    with (root / f"render-{event['sequence']:012d}.log").open("ab", buffering=0) as log:
        process = subprocess.run([str(require_renderer()), "--render-store-request", str(request_path),
            "--render-store-result", str(result_path)], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            timeout=1800, check=False)
    if process.returncode:
        raise ValueError(f"Native plot rendering exited {process.returncode}; see {log.name}")
    result, _ = ra._raw(result_path, viewer.MAX_METADATA_BYTES)
    if (result.get("schema") != "arwen.native-store-render-result.v1"
            or result.get("frame_id") != entry["frame"]["id"]
            or result.get("identity") != entry["frame"]["identity"]
            or result.get("source_store_sha256") != entry["frame"]["rws_sha256"]
            or result.get("wrf_imported") is not False or result.get("volume_store_created") is not False):
        raise ValueError("Native renderer returned plots for another compact frame")
    panels = result.get("panels")
    if not isinstance(panels, list) or sorted(row.get("slug", "") for row in panels) != sorted(products):
        raise ValueError("Native renderer did not publish every available selected product")
    total = 0
    for panel in panels:
        path = ra._inside(panel["path"], output)
        if (path.suffix != ".png" or type(panel.get("bytes")) is not int
                or not 0 < panel["bytes"] <= MAX_PANEL_BYTES or path.stat().st_size != panel["bytes"]
                or ra._file_sha(path) != panel["sha256"]):
            raise ValueError("Native PNG checksum or size changed before publication")
        total += panel["bytes"]
    if total > MAX_GALLERY_BYTES:
        raise ValueError("Native frame gallery exceeds its transfer bound")
    if list(ra._stamp(source)) != entry["source_stamp"]:
        raise ValueError("WRF output changed during native rendering")
    return {"schema": SCHEMA, "state": "ready", "job_id": record["id"], "run_id": bound[2]["run_id"],
        "domain": event["domain"], "sequence": event["sequence"], "valid_time": event["valid_time"],
        "commit_sha256": authority["sha256"], "frame_id": entry["frame"]["id"],
        "source_sha256": entry["source_sha256"], "source_stamp": entry["source_stamp"],
        "source_path": str(source), "panels": panels, "bytes": total,
        "unavailable": [row for row in entry["products"] if not row["available"]],
        "processing": "existing_compact_store", "published_unix_ms": int(time.time() * 1000)}


def _spacing(record, domain):
    from gpuwm.experiment import load_experiment
    viewer._initialization(record)  # verifies the saved config SHA before its spacing is read
    return load_experiment(record["snapshot_config"]).domain(domain).run.dx


def work_once(workspace, job, *, render=True, _completion=None):
    from gpuwm.remote_worker import TERMINAL
    record, state, bound, commits = legacy._job(
        workspace, job, **({"completion": True} if _completion is not None else {}))
    if _completion is not None:
        # Revalidate before any receipt, render or store request, including the
        # first pass that observes the wrapper's terminal result.
        _completion.validate(record, state, bound, commits)
    root = _root(workspace, job)
    compact_root = viewer._root(workspace)
    selection = viewer._selection()
    counts = {"committed": len(commits), "ready": 0, "failed": 0, "pending": 0, "panels": 0}
    candidate = None
    for event, authority in reversed(commits):
        published = _published(root, job, event, authority)
        if published is not None:
            if published["state"] not in ("ready", "failed"):
                raise ValueError("Invalid native plot publication state")
            counts[published["state"]] += 1
            counts["panels"] += len(published.get("panels", []))
            continue
        counts["pending"] += 1
        if candidate is None:
            entry = viewer._entry(compact_root, job, event, authority, selection)
            if viewer._available(entry):
                candidate = (event, authority, entry)
            elif entry is not None and entry["state"] in ("failed", "backpressure"):
                legacy._write(_receipt(root, event["sequence"]), {"schema": SCHEMA, "state": "failed", "job_id": job,
                    "domain": event["domain"], "sequence": event["sequence"], "commit_sha256": authority["sha256"],
                    "error": "Compact store preparation failed: " + entry.get("error", "unknown cause")})
            elif entry is not None and viewer._entry_state(entry) == "evicted" and render:
                # A long run may rotate the compact cache before plots catch up.
                # Request only this missing compact frame, never a full store.
                viewer.ensure(workspace, job, [{**selection, "domain": event["domain"], "sequence": event["sequence"],
                    "run_id": bound[2]["run_id"], "commit_sha256": authority["sha256"]}])
    done = state["state"] in TERMINAL and counts["pending"] == 0
    summary = {"schema": STATUS_SCHEMA, "job_id": job, "simulation_state": state["state"], **counts,
        "done": done, "state": "complete_with_errors" if done and counts["failed"] else "complete" if done
            else "rendering" if candidate else "waiting_for_compact_stores", "updated_unix_ms": int(time.time() * 1000)}
    if bound:
        summary["run_id"] = bound[2]["run_id"]
    if candidate:
        event, authority, entry = candidate
        summary["active"] = {"domain": event["domain"], "sequence": event["sequence"], "valid_time": event["valid_time"]}
    legacy._write(root / "status.json", summary)
    if candidate and render:
        path = viewer._entry_path(compact_root, job, event["sequence"], selection)
        with Lease(viewer._directory(compact_root, job) / (path.stem + ".lock")) as lease:
            if lease.file is not None:
                if not viewer._available(entry):
                    return summary
                try:
                    spacing = _spacing(record, event["domain"])
                    value = _render(root, record, bound, event, authority, entry, spacing)
                except Exception as error:
                    value = {"schema": SCHEMA, "state": "failed", "job_id": job, "domain": event["domain"],
                        "sequence": event["sequence"], "commit_sha256": authority["sha256"], "error": str(error)[:2000]}
                legacy._write(_receipt(root, event["sequence"]), value)
    return summary


def ensure(workspace, job):
    from gpuwm import remote_worker as rw
    record = rw._record(rw._directory(workspace, job))
    if record.get("action") != "start-plan" or not record.get("snapshot_plan"):
        return
    root = _root(workspace, job)
    previous = root / "status.json"
    if previous.exists() and ra._raw(previous, 64 * 1024)[0].get("done"):
        return
    with Lease(root / "worker.lock") as lease:
        if lease.file is None:
            return
        environment = dict(os.environ)
        environment.pop(rw.TOKEN_ENV, None)
        environment.update(GPUWM_NO_LOCAL_GPU="1", CUDA_VISIBLE_DEVICES="-1", RAYON_NUM_THREADS="2",
            OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
        with (root / "worker.log").open("ab", buffering=0) as log:
            subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()), "--workspace", str(workspace),
                "--job", job], cwd=str(workspace), env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, start_new_session=True, close_fds=True)


def _receipt_state(root, job, state, error, *, done):
    legacy._write(root / "status.json", {"schema": STATUS_SCHEMA, "job_id": job, "state": state,
        "done": done, "error": str(error)[:2000]})


def worker(workspace, job, *, cancel=None):
    root = _root(workspace, job)
    with Lease(root / "worker.lock", timeout=3) as lease:
        if lease.file is None:
            return 0
        if hasattr(os, "nice"):
            os.nice(5)
        completion = ra.CompletionWait(workspace, job, cancel, time)
        while True:
            try:
                completion.check()
                completion.begin()
                value = None
                try:
                    value = work_once(workspace, job, _completion=completion)
                except ra.ProducerCompletionPending as pending:
                    # The runner exited and its wrapper has not settled. No
                    # receipt is written and no renderer runs in that window.
                    completion.pending(pending)
                except ra.ProducerCompletionUnprovable as unprovable:
                    # This job's runner-exit window cannot be proved and no
                    # renderer runs inside it either. A terminal receipt here
                    # would make ensure() refuse to relaunch this gallery for
                    # the life of the job, so the receipt stays non-terminal
                    # and says why the gallery is still pending.
                    completion.unprovable(unprovable)  # Raises if this wait held proof.
                    _receipt_state(root, job, "waiting_for_producer_completion",
                                   unprovable, done=False)
                if value is not None and value["done"]:
                    return 0
                completion.wait(.1 if value is not None and value.get("active") else 2)
            except ra.ProducerCompletionCancelled as cancelled:
                # ensure() refuses to relaunch on any done receipt, so a
                # cancelled gallery must stay non-terminal.
                _receipt_state(root, job, "cancelled", cancelled, done=False)
                return 2
            except Exception as error:
                _receipt_state(root, job, "failed", error, done=True)
                return 2


def catalog(request, workspace):
    allowed = {"schema", "action", "workspace", "job", "domain", "sequence"}
    if set(request) - allowed or request.get("action") != "native-plots":
        raise ValueError("Invalid native plot catalog request")
    domain = ra._domain(request.get("domain", 1))
    sequence = ra._sequence(request["sequence"]) if request.get("sequence") is not None else None
    job = request["job"]
    from gpuwm.remote_preparation_v2 import ensure as prepare_stores
    prepare_stores(workspace, job)
    ensure(workspace, job)
    value = {"schema": SCHEMA, "job_id": job, "domain": domain, "sequence": sequence,
        "waiting": True, "progress": status(workspace, job)}
    try:
        record, _state, bound, commits = legacy._job_completing(workspace, job)
    except ra.ProducerCompletionPending:
        # The interactive door waits with the watcher instead of refusing for
        # the seconds a settling wrapper owns. No authority is published here.
        return {**value, "producer_completing": True}
    if bound is None:
        return value
    _, manifest_path, manifest, manifest_bytes, _started, binding = bound
    value.update(run_id=manifest["run_id"], run_manifest=ra._authority(manifest_path, manifest_bytes),
        remote_output_root=record["outdir"], remote_pid=manifest["pid"])
    if binding is not None:
        value["producer_binding"] = binding
    selected = [(event, authority) for event, authority in commits
        if event["domain"] == domain and (sequence is None or event["sequence"] == sequence)]
    if not selected:
        return value
    event, authority = selected[-1]
    value.update(sequence=event["sequence"], valid_time=event["valid_time"], commit=authority)
    published = _published(_root(workspace, job), job, event, authority)
    if published is None:
        return value
    if published["state"] == "failed":
        value["error"] = published["error"]
        return value
    if published["run_id"] != manifest["run_id"] or list(ra._stamp(ra._inside(published["source_path"], bound[0]))) != published["source_stamp"]:
        raise ValueError("Native plot source changed after publication")
    value.update(waiting=False, frame_id=published["frame_id"], panels=published["panels"],
        unavailable=published["unavailable"], bytes=published["bytes"])
    return ra._bounded(value)


def stream(request, workspace, output):
    fields = {"schema", "action", "workspace", "job", "domain", "sequence", "product",
        "expected_panel_sha256", "expected_commit_sha256", "expected_manifest_sha256"}
    if set(request) != fields or request.get("action") != "stream-native-plot":
        raise ValueError("Invalid native plot stream request")
    if any(not ra.HEX.fullmatch(str(request[k])) for k in fields if k.startswith("expected_")):
        raise ValueError("Invalid native plot stream digest")
    query = {k: request[k] for k in ("schema", "workspace", "job", "domain", "sequence")}
    value = catalog({**query, "action": "native-plots"}, workspace)
    if (value["waiting"] or value["commit"]["sha256"] != request["expected_commit_sha256"]
            or value["run_manifest"]["sha256"] != request["expected_manifest_sha256"]):
        raise ValueError("Native plot authority changed before transfer")
    panel = next((row for row in value["panels"] if row["slug"] == request["product"]), None)
    if panel is None or panel["sha256"] != request["expected_panel_sha256"] or not 0 < panel["bytes"] <= MAX_PANEL_BYTES:
        raise ValueError("Native plot product or checksum does not match the selected frame")
    path = ra._inside(panel["path"], _root(workspace, request["job"]))
    before = ra._stamp(path)
    digest, copied = hashlib.sha256(), 0
    with path.open("rb") as source:
        while block := source.read(min(1024**2, panel["bytes"] + 1 - copied)):
            copied += len(block)
            if copied > panel["bytes"]:
                raise ValueError("Native plot grew during transfer")
            digest.update(block)
            output.write(block)
    output.flush()
    if copied != panel["bytes"] or digest.hexdigest() != panel["sha256"] or ra._stamp(path) != before:
        raise ValueError("Native plot changed during transfer")


def sync(args, command, stream_command):
    from gpuwm.remote_cli import _transport
    query = {"schema": "gpuwm.remote.request.v1", "action": "native-plots", "workspace": args.workspace,
        "job": args.job, "domain": ra._domain(args.domain), "sequence": ra._sequence(args.sequence)}
    reply = _transport(command, query, timeout=120)
    if not reply["ok"]:
        raise ValueError(reply["error"]["message"])
    value = reply.get("native_plots")
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("job_id") != args.job or value.get("domain") != args.domain or value.get("sequence") != args.sequence:
        raise ValueError("Native plot gallery belongs to another selection")
    if value["waiting"]:
        return {"native_plots": value, "transferred_bytes": 0}
    panels = value.get("panels")
    if not isinstance(panels, list) or not 1 <= len(panels) <= 96:
        raise ValueError("Invalid native plot gallery inventory")
    names, total = set(), 0
    for panel in panels:
        if (not viewer.SLUG.fullmatch(str(panel.get("slug"))) or panel["slug"] in names
                or not ra.HEX.fullmatch(str(panel.get("sha256"))) or type(panel.get("bytes")) is not int
                or not 0 < panel["bytes"] <= MAX_PANEL_BYTES):
            raise ValueError("Invalid native plot gallery product, checksum or size")
        names.add(panel["slug"])
        total += panel["bytes"]
    if total > MAX_GALLERY_BYTES or total != value.get("bytes"):
        raise ValueError("Native plot gallery exceeds its transfer bound")
    key = ra._sha(ra._encoded([value["run_id"], value["commit"]["sha256"], panels]))
    directory = _owned_directory(_owned_directory(Path(args.cache_root)) / key)
    transferred = 0
    for panel in panels:
        path = directory / (panel["slug"] + ".png")
        if path.exists():
            if path.is_symlink() or path.stat().st_size != panel["bytes"] or ra._file_sha(path) != panel["sha256"]:
                raise ValueError("A retained native plot changed; choose a fresh gallery cache")
        else:
            request = {**query, "action": "stream-native-plot", "product": panel["slug"],
                "expected_panel_sha256": panel["sha256"], "expected_commit_sha256": value["commit"]["sha256"],
                "expected_manifest_sha256": value["run_manifest"]["sha256"]}
            ra._download(stream_command, request, path, {"size_bytes": panel["bytes"], "sha256": panel["sha256"]})
            transferred += panel["bytes"]
    title = f"ArWen native plots · d{args.domain:02d} · {value['valid_time']}"
    body = ''.join(f'<figure><a href="{row["slug"]}.png"><img loading="lazy" src="{row["slug"]}.png" alt="{html.escape(row["slug"])}"></a><figcaption>{html.escape(row["slug"].replace("_", " "))}</figcaption></figure>' for row in panels)
    gallery = directory / "index.html"
    gallery.write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>' + html.escape(title)
        + '</title><style>body{font:16px system-ui;margin:24px;background:#eef2f6;color:#17252f}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}figure{margin:0;padding:12px;background:white}img{width:100%;height:auto}figcaption{padding:8px 0}h1{font-size:24px}</style><h1>'
        + html.escape(title) + '</h1><p>Native plots from this committed forecast frame. Click a plot to open its original PNG.</p><main>' + body + '</main>', encoding="utf-8")
    value.update(gallery_path=str(gallery), gallery_sha256=ra._file_sha(gallery))
    return {"native_plots": value, "transferred_bytes": transferred}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    from gpuwm.remote_worker import _workspace
    return worker(_workspace({"workspace": args.workspace}), args.job,
                  cancel=ra.cancel_on_shutdown())


if __name__ == "__main__":
    raise SystemExit(main())
