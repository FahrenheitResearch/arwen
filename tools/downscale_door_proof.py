"""Drive the offline downscale door end to end through the controller queue.

Test tooling, not a product door.  It starts ``arwen-tui --headless-companion``
against a stand-in visual workspace, then speaks the same request queue the
desktop speaks: ``launch_plan`` for a small parent forecast, ``browse_runs``
to read the parent's downscale receipts, ``launch_downscale`` in plan mode and
in run mode, ``browse_runs``/``open_run``/``close_run`` on the child, and the
three refusals (a node target, an output directory that exists, a parent with
no checkpoint).  Every request, response, status sample and receipt is copied
into the evidence directory so the transcript is the proof.

Usage (on the node that owns the GPU)::

    python tools/downscale_door_proof.py \
        --controller /path/to/arwen-tui --python /venv/bin/python \
        --workspace /path/to/stand-in.sh --root /path/to/root \
        --parent-plan plan.json --parent-config parent.toml \
        --evidence /path/to/evidence --point 35,-97 --child-hours 1

``--parent-job`` skips the parent forecast and reuses a finished job.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from tools.saved_setups_proof import Controller, Transcript, copy_receipts, sha256
else:
    from saved_setups_proof import Controller, Transcript, copy_receipts, sha256


def copy_run_receipts(transcript: Transcript, run_dir: Path, prefix: str) -> None:
    for name in ("run-manifest.json", "events.jsonl", "downscale-plan.json", "child.toml",
                 "report.json", "run-progress.json"):
        transcript.copy(run_dir / name, f"{prefix}.{name}")
    listing = sorted(p.name for p in run_dir.iterdir()) if run_dir.is_dir() else []
    (transcript.directory / f"{prefix}.listing.txt").write_text("\n".join(listing) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--workspace", required=True, help="stand-in visual workspace executable")
    parser.add_argument("--root", required=True)
    parser.add_argument("--parent-plan", required=True)
    parser.add_argument("--parent-config", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--point", default=None, help="LAT,LON child centre; default parent centre from browse_runs")
    parser.add_argument("--child-hours", type=float, default=None)
    parser.add_argument("--parent-job", default=None, help="reuse a finished parent job_id")
    parser.add_argument("--parent-timeout", type=float, default=3600.0)
    parser.add_argument("--child-timeout", type=float, default=3600.0)
    args = parser.parse_args()

    transcript = Transcript(Path(args.evidence))
    transcript.say(f"downscale door proof on {os.uname().nodename}")
    controller = Controller(args, transcript)
    outcome = 1
    try:
        local = {"kind": "local"}
        # 1. The parent forecast, launched the way the desktop launches one.
        if args.parent_job:
            parent_job = args.parent_job
        else:
            plan = Path(args.parent_plan)
            config = Path(args.parent_config)
            # The desktop opens the configuration it is about to launch, so the
            # controller can bind the plan's digests to the file it holds.
            controller.send("open_config", {"config_path": str(config)})
            launched = controller.send("launch_plan", {
                "plan_path": str(plan), "plan_sha256": sha256(plan),
                "config_sha256": sha256(config), "target": local})
            if not launched.get("ok"):
                raise SystemExit("parent launch refused")
            parent_job = launched["job_id"]
            parent = controller.wait_job(parent_job, "parent", args.parent_timeout)
            if parent.get("state") != "completed":
                copy_receipts(transcript, Path(parent_job), "parent")
                raise SystemExit(f"parent forecast ended {parent.get('state')}: {parent.get('error')}")
        copy_receipts(transcript, Path(parent_job), "parent")

        # 2. The parent row carries the receipts a downscale door reads.
        rows = controller.send("browse_runs", {"target": local})["jobs"]
        parent_row = next(row for row in rows if row.get("job_id") == parent_job)
        transcript.record("parent_row", row=parent_row)
        (transcript.directory / "parent-row.json").write_text(json.dumps(parent_row, indent=2, sort_keys=True), encoding="utf-8")
        receipts = parent_row.get("downscale") or {}
        transcript.say(f"parent row: state={parent_row.get('state')} run_dir={parent_row.get('run_dir')} downscale={receipts}")
        assert receipts.get("history_frames", 0) >= 2, receipts
        assert receipts.get("restart_sets", 0) >= 1, receipts
        # The frames folder the controller read from the parent's own stream
        # is what the desktop sends and what the engine reads.
        parent_run_dir = Path(receipts["history_dir"])
        assert parent_run_dir.is_dir() and any(parent_run_dir.glob("wrfout_d01_*")), parent_run_dir
        copy_run_receipts(transcript, Path(parent_row["run_dir"]), "parent-run")
        (transcript.directory / "parent-run.frames.txt").write_text(
            "\n".join(sorted(p.name for p in parent_run_dir.iterdir())) + "\n\n"
            + "\n".join(sorted(p.name for p in Path(receipts["checkpoint_dir"]).iterdir())) + "\n",
            encoding="utf-8")

        if args.point:
            lat, lon = (float(part) for part in args.point.split(","))
        else:
            raise SystemExit("--point is required")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = controller.output / f"downscale-{Path(parent_job).name[-12:]}-{stamp}"
        request = {"target": local, "parent_run_dir": str(parent_run_dir), "parent_domain": None,
                   "parent_restart": None, "point": {"lat": lat, "lon": lon}, "ratio": 3,
                   "auto_vram": True, "hours": args.child_hours,
                   "output_interval_seconds": receipts.get("history_interval_s"),
                   "tiles": "auto", "accept_parent_cadence": True,
                   "out_dir": str(out_dir), "mode": "plan"}

        # 3. Plan mode: the engine prices the child and writes the plan document.
        planned = controller.send("launch_downscale", request)
        if not planned.get("ok"):
            raise SystemExit("plan refused: " + planned.get("message", ""))
        plan_job = controller.wait_job(planned["job_id"], "plan", 900)
        copy_receipts(transcript, Path(planned["job_id"]), "plan")
        transcript.say(f"plan job state={plan_job.get('state')} error={plan_job.get('error')}")
        if plan_job.get("state") != "completed":
            raise SystemExit("plan job failed")
        plan_path = Path(planned["downscale_plan_path"])
        plan_document = json.loads(plan_path.read_text(encoding="utf-8"))
        transcript.copy(plan_path, "downscale-plan.plan-mode.json")
        transcript.copy(Path(planned["child_config_path"]), "child.plan-mode.toml")
        transcript.record("plan_document", plan=plan_document)
        transcript.say("plan: grid=%s memory=%s cadence=%s warnings=%d" % (
            json.dumps(plan_document.get("child_grid")), json.dumps(plan_document.get("memory")),
            json.dumps(plan_document.get("cadence")), len(plan_document.get("warnings") or [])))

        # 4. Run mode: the identical payload starts the child.
        request["mode"] = "run"
        started = controller.send("launch_downscale", request)
        if not started.get("ok"):
            raise SystemExit("run refused: " + started.get("message", ""))
        child_job = started["job_id"]
        child = controller.wait_job(child_job, "child", args.child_timeout, sample_every=10.0)
        copy_receipts(transcript, Path(child_job), "child")
        copy_run_receipts(transcript, out_dir, "child-run")
        transcript.say(f"child job state={child.get('state')} exit={child.get('exit_code')} error={child.get('error')}")
        if child.get("state") != "completed":
            raise SystemExit("child failed")

        # 5. The child is listed and opens like any other forecast.
        rows = controller.send("browse_runs", {"target": local})["jobs"]
        child_row = next(row for row in rows if row.get("job_id") == child_job)
        (transcript.directory / "child-row.json").write_text(json.dumps(child_row, indent=2, sort_keys=True), encoding="utf-8")
        transcript.record("child_row", row=child_row)
        transcript.say(f"child row: action={child_row.get('action')} name={child_row.get('name')!r} "
                       f"parent_run_dir={child_row.get('parent_run_dir')} state={child_row.get('state')}")
        assert child_row.get("action") == "downscale", child_row
        assert child_row.get("parent_run_dir") == str(parent_run_dir), child_row
        opened = controller.send("open_run", {"job_id": child_job, "target": local})
        viewer_handoff = opened.get("handoff") or {}
        transcript.record("viewer_handoff", handoff=viewer_handoff)
        if viewer_handoff.get("status_path"):
            time.sleep(3.0)
            transcript.copy(Path(viewer_handoff["status_path"]), "viewer-status.json")
            try:
                viewer_status = json.loads(Path(viewer_handoff["status_path"]).read_text(encoding="utf-8"))
                progress = (viewer_status.get("job") or {}).get("progress") or {}
                transcript.say(f"viewer: state={viewer_status.get('state')} job_state={(viewer_status.get('job') or {}).get('state')} "
                               f"outputs={progress.get('outputs_written')} domains={len(progress.get('domains') or [])}")
            except (OSError, ValueError):
                pass
        # A saved run is closed through its own read-only viewer's queue, the
        # way the desktop closes one; the controller's queue refuses it.
        if viewer_handoff.get("control_dir") and viewer_handoff.get("session_id"):
            viewer_control = Path(viewer_handoff["control_dir"])
            closing = {"schema": "arwen.companion-request.v1", "session_id": viewer_handoff["session_id"],
                       "id": "proof-viewer-close", "action": "close_run", "job_id": child_job, "target": local}
            temporary = viewer_control / "requests" / "proof-viewer-close.tmp"
            temporary.write_text(json.dumps(closing, indent=2), encoding="utf-8")
            os.replace(temporary, viewer_control / "requests" / "proof-viewer-close.json")
            response_path = viewer_control / "responses" / "proof-viewer-close.json"
            for _ in range(300):
                if response_path.is_file():
                    break
                time.sleep(0.1)
            closed = json.loads(response_path.read_text(encoding="utf-8")) if response_path.is_file() else {"ok": None, "message": "no response"}
            transcript.record("exchange", request=closing, response=closed)
            transcript.say(f"close_run (viewer) -> ok={closed.get('ok')} {closed.get('message', '')}")

        # 6. Refusals through the same queue, verbatim.
        refusals = {}
        remote = dict(request, mode="plan", target={"kind": "ssh", "node_id": "remote-a", "connection_sha256": "a" * 64})
        refusals["ssh_target"] = controller.send("launch_downscale", remote)
        taken = dict(request, mode="plan", out_dir=str(out_dir))
        refusals["existing_out_dir"] = controller.send("launch_downscale", taken)
        # A parent that saved history but never a checkpoint: the same on-disk
        # state a run with restart_interval_s = 0 leaves behind.
        bare = controller.output / "parent-without-checkpoints"
        bare.mkdir(exist_ok=True)
        for frame in sorted(parent_run_dir.glob("wrfout_d01_*"))[:2]:
            shutil.copy2(frame, bare / frame.name)
        unchecked = dict(request, mode="plan", parent_run_dir=str(bare),
                         out_dir=str(controller.output / f"downscale-bare-{stamp}"))
        answer = controller.send("launch_downscale", unchecked)
        refusals["no_checkpoint"] = answer
        if answer.get("ok"):
            bare_job = controller.wait_job(answer["job_id"], "bare-plan", 600)
            copy_receipts(transcript, Path(answer["job_id"]), "bare-plan")
            refusals["no_checkpoint_job"] = {"state": bare_job.get("state"), "error": bare_job.get("error"),
                                             "exit_code": bare_job.get("exit_code")}
            transcript.say(f"bare parent plan job: state={bare_job.get('state')} error={bare_job.get('error')!r}")
        (transcript.directory / "refusals.json").write_text(json.dumps(refusals, indent=2, sort_keys=True, default=str), encoding="utf-8")
        for key, value in refusals.items():
            transcript.say(f"refusal {key}: {json.dumps(value, default=str)[:400]}")
        outcome = 0
    finally:
        controller.stop()
        summary = {"outcome": outcome, "records": len(transcript.records), "root": str(controller.root)}
        (transcript.directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return outcome


if __name__ == "__main__":
    sys.exit(main())
