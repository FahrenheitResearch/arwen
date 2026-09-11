"""Prepare compact native map stores as each domain's WRF output commits.

This metadata-only watcher feeds the existing CPU viewer worker. It never
opens WRF data, duplicates full-science stores, or owns the forecast process.
Interactive selections stay ahead of background preparation in the same
bounded queue, with the same publication identity and reader leases.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from gpuwm import remote_artifacts as ra, remote_processed as legacy, remote_processed_v2 as viewer
from gpuwm.remote_artifact_cache import Lease

SCHEMA = "arwen.native-store-preparation.v1"
POLL_SECONDS = 2.0
COMPLETION_SECONDS = ra.COMPLETION_SECONDS


def prepare(workspace, job, *, start=True, _completion=None):
    """Append missing committed frames without displacing interactive work.

An existing receipt, even one evicted by the cache budget or documenting a
native failure, is not automatically retried. This avoids endlessly deriving
old frames as a long forecast exceeds its cache. A user selection can still
request an evicted frame through the normal viewer contract.
    """
    from gpuwm.remote_worker import TERMINAL

    _record, state, bound, commits = legacy._job(
        workspace, job, **({"completion": True} if _completion is not None else {}))
    if _completion is not None:
        # Revalidate before any queue or publication mutation, including the
        # first pass that observes the wrapper's terminal result.
        _completion.validate(_record, state, bound, commits)
    root = viewer._root(workspace)
    directory = viewer._directory(root, job)
    counts = {"committed": len(commits), "ready": 0, "evicted": 0,
              "failed": 0, "backpressure": 0, "queued": 0, "pending": 0}
    selection = viewer._selection()
    with Lease(root / "schedule.lock", timeout=3) as schedule:
        if schedule.file is None:
            raise ValueError("The compact viewer queue is being updated")
        rows = viewer._queue(root, job)
        keys = {(row["sequence"], row["selection_id"]) for row in rows}
        added = []
        # Catch up to the newest output first, across every committed domain.
        # Older commits remain discoverable on later passes if capacity fills.
        for event, authority in reversed(commits):
            entry = viewer._entry(root, job, event, authority, selection)
            entry_state = viewer._entry_state(entry)
            if entry is not None:
                if entry_state not in ("ready", "evicted", "failed", "backpressure"):
                    raise ValueError("Unknown compact viewer publication state")
                counts[entry_state] += 1
                continue
            key = (event["sequence"], selection["selection_id"])
            if key in keys:
                counts["queued"] += 1
            elif len(rows) + len(added) < viewer.MAX_QUEUED:
                added.append({**selection, "domain": event["domain"], "sequence": event["sequence"],
                              "run_id": bound[2]["run_id"], "commit_sha256": authority["sha256"]})
                keys.add(key)
                counts["queued"] += 1
            else:
                counts["pending"] += 1
        if added:
            viewer._save_queue(root, job, rows + added)
        # Launch under the scheduling lease, matching viewer.ensure's exit
        # handoff: a worker cannot exit between our queue write and launch.
        if start and (rows or added):
            viewer._launch_worker(root, workspace)
    done = state["state"] in TERMINAL and not counts["queued"] and not counts["pending"]
    value = {"schema": SCHEMA, "job_id": job, "profile": viewer.PROFILE,
             "selection_id": selection["selection_id"], "simulation_state": state["state"],
             **counts, "added": len(added), "done": done,
             "state": "complete_with_errors" if done and (counts["failed"] or counts["backpressure"])
                      else "complete" if done else "preparing" if counts["queued"] else "waiting_for_output",
             "updated_unix_ms": int(time.time() * 1000)}
    if bound is not None:
        value["run_id"] = bound[2]["run_id"]
    legacy._write(directory / "preparation.json", value)
    return value


def ensure(workspace, job):
    """Launch one detached metadata watcher for a durable forecast job."""
    from gpuwm import remote_worker as rw

    record = rw._record(rw._directory(workspace, job))
    if record.get("action") != "start-plan" or not record.get("snapshot_plan"):
        return
    directory = viewer._directory(viewer._root(workspace), job)
    previous = directory / "preparation.json"
    if previous.exists() and ra._raw(previous, 64 * 1024)[0].get("done"):
        return
    with Lease(directory / "preparation.lock") as lease:
        if lease.file is None:
            return
        environment = dict(os.environ)
        environment.pop(rw.TOKEN_ENV, None)
        environment.update(GPUWM_NO_LOCAL_GPU="1", CUDA_VISIBLE_DEVICES="-1",
                           RAYON_NUM_THREADS="2", OMP_NUM_THREADS="2",
                           OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
        with (directory / "preparation.log").open("ab", buffering=0) as log:
            subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()),
                              "--workspace", str(workspace), "--job", job],
                             cwd=str(workspace), env=environment, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=log, start_new_session=True, close_fds=True)


def _receipt(directory, job, state, error, *, done):
    legacy._write(directory / "preparation.json", {
        "schema": SCHEMA, "job_id": job, "state": state, "done": done,
        "error": str(error)[:2000], "updated_unix_ms": int(time.time() * 1000)})


def worker(workspace, job, *, cancel=None):
    directory = viewer._directory(viewer._root(workspace), job)
    with Lease(directory / "preparation.lock", timeout=3) as lease:
        if lease.file is None:
            return 0
        if hasattr(os, "nice"):
            os.nice(5)
        completion = ra.CompletionWait(workspace, job, cancel, time)
        while True:
            try:
                completion.check()
                completion.begin()
                try:
                    if prepare(workspace, job, _completion=completion)["done"]:
                        return 0
                except ra.ProducerCompletionPending as pending:
                    completion.pending(pending)
                except ra.ProducerCompletionUnprovable as unprovable:
                    # The runner-exit window cannot be proved for this job, so
                    # nothing is prepared inside it. That is not a failure of
                    # the job: a terminal receipt here would stop ensure() from
                    # ever relaunching this preparation, closing the very way
                    # out the refusal names, so the job stays pending until it
                    # reports a terminal state and the ordinary path resumes.
                    completion.unprovable(unprovable)  # Raises if this wait held proof.
                    _receipt(directory, job, "waiting_for_producer_completion",
                             unprovable, done=False)
                completion.wait(POLL_SECONDS)
            except ra.ProducerCompletionCancelled as cancelled:
                # A cancelled wait is not a failed one. ensure() refuses to
                # relaunch on any done receipt, so cancellation stays open.
                _receipt(directory, job, "cancelled", cancelled, done=False)
                return 2
            except Exception as error:
                # Never affect the integrator or loop indefinitely on malformed
                # authority. The exact failure remains visible beside the queue.
                _receipt(directory, job, "failed", error, done=True)
                return 2


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
