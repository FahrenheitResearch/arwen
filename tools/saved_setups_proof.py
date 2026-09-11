"""Prove saved forecast setups end to end through the controller queue.

Test tooling, not a product door.  It starts ``arwen-tui --headless-companion``
against a stand-in visual workspace, then does what the desktop does: creates a
small forecast with the domain door, reads it with ``companion-query``, saves
it as a setup with ``companion-setups save``, starts a new forecast from that
setup at an earlier cycle with ``companion-setups start``, launches both
forecasts through ``launch_plan``, waits for both to complete, lists them with
``browse_runs``, and reads both configurations again to show that only the
timed fields differ.  Every designed refusal is then exercised through the
same door and its sentence recorded verbatim.  Every request, response,
receipt and document is copied into the evidence directory so the transcript
is the proof.

Usage (on the node that owns the GPU)::

    python tools/saved_setups_proof.py \
        --controller /path/to/arwen-tui --python /venv/bin/python \
        --workspace /path/to/stand-in.sh --root /path/to/root \
        --evidence /path/to/evidence --geog-root /path/to/WPS_GEOG
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

SETUP_NAME = "Proof setup"
SETUP_SLUG = "proof-setup"
RESTART_NAME = "Proof restart"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Transcript:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.log = (directory / "transcript.log").open("a", encoding="utf-8")
        self.records: list[dict] = []

    def say(self, text: str) -> None:
        line = f"{now()} {text}"
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    def record(self, kind: str, **fields) -> None:
        entry = {"time": now(), "kind": kind, **fields}
        self.records.append(entry)
        with (self.directory / "transcript.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True, default=str) + "\n")

    def copy(self, source: Path, name: str) -> None:
        if source.is_file():
            shutil.copy2(source, self.directory / name)

    def write(self, name: str, document) -> None:
        (self.directory / name).write_text(
            json.dumps(document, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


class Engine:
    """The engine's doors, run the way the desktop runs them: one cwd, argv only."""

    def __init__(self, python: str, cwd: Path, transcript: Transcript) -> None:
        self.python = python
        self.cwd = cwd
        self.transcript = transcript

    def run(self, label: str, args: list[str], timeout: float = 600.0) -> tuple[int, str, str]:
        command = [self.python, "-m", "gpuwm.cli", *args]
        started = time.monotonic()
        completed = subprocess.run(command, cwd=self.cwd, capture_output=True, text=True,
                                   timeout=timeout, check=False)
        elapsed = round(time.monotonic() - started, 2)
        self.transcript.record("door", label=label, argv=args, returncode=completed.returncode,
                               seconds=elapsed, stdout=completed.stdout[-20000:],
                               stderr=completed.stderr[-20000:])
        self.transcript.say(f"{label}: exit {completed.returncode} in {elapsed}s")
        return completed.returncode, completed.stdout, completed.stderr

    def json(self, label: str, args: list[str], timeout: float = 600.0) -> tuple[int, dict]:
        code, stdout, stderr = self.run(label, args, timeout)
        try:
            document = json.loads(stdout)
        except ValueError:
            raise SystemExit(f"{label} printed no JSON document:\n{stdout}\n{stderr}") from None
        return code, document


class Controller:
    def __init__(self, args, transcript: Transcript) -> None:
        self.args = args
        self.transcript = transcript
        self.root = Path(args.root)
        self.output = self.root / "runs"
        self.output.mkdir(parents=True, exist_ok=True)
        self.counter = 0
        self.stderr = (transcript.directory / "controller.stderr.log").open("ab")
        command = [args.controller, "--headless-companion", "--python", args.python,
                   "--companion", args.workspace, "--output", str(self.output)]
        transcript.say("controller: " + " ".join(command))
        self.process = subprocess.Popen(command, cwd=self.root, stdin=subprocess.DEVNULL,
                                        stdout=self.stderr, stderr=self.stderr)
        deadline = time.monotonic() + 60
        self.handoff = None
        candidate = None
        while self.handoff is None:
            for candidate in glob.glob(str(self.output / ".arwen-tui" / "companion-*" / "handoff.json")):
                try:
                    self.handoff = json.loads(Path(candidate).read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
            if self.handoff is None:
                if self.process.poll() is not None or time.monotonic() > deadline:
                    raise SystemExit("controller did not publish a handoff")
                time.sleep(0.2)
        transcript.record("handoff", handoff=self.handoff)
        transcript.copy(Path(candidate), "handoff.json")
        self.control = Path(self.handoff["control_dir"])
        self.status_path = Path(self.handoff["status_path"])
        transcript.say(f"handoff: session {self.handoff['session_id']} control {self.control}")

    def status(self) -> dict:
        for _ in range(20):
            try:
                return json.loads(self.status_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                time.sleep(0.05)
        return {}

    def send(self, action: str, fields: dict, timeout: float = 120.0) -> dict:
        self.counter += 1
        request_id = f"proof-{self.counter:03d}-{action}"
        request = dict(fields)
        request.update({"schema": "arwen.companion-request.v1",
                        "session_id": self.handoff["session_id"],
                        "id": request_id, "action": action})
        temporary = self.control / "requests" / f"{request_id}.tmp"
        temporary.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.control / "requests" / f"{request_id}.json")
        response_path = self.control / "responses" / f"{request_id}.json"
        deadline = time.monotonic() + timeout
        while not response_path.is_file():
            if time.monotonic() > deadline:
                raise SystemExit(f"no response to {request_id} within {timeout}s")
            time.sleep(0.1)
        time.sleep(0.05)
        response = json.loads(response_path.read_text(encoding="utf-8"))
        self.transcript.record("exchange", request=request, response=response)
        self.transcript.say(f"{action} -> ok={response.get('ok')} {response.get('message', '')!s}"
                            + (f" job={response['job_id']}" if response.get("job_id") else ""))
        return response

    def wait_job(self, job_id: str, label: str, timeout: float, sample_every: float = 15.0) -> dict:
        """Follow status.json's job until it is terminal; keep progress samples."""
        deadline = time.monotonic() + timeout
        last_sample = 0.0
        samples = []
        last = {}
        while True:
            status = self.status()
            job = status.get("job") or {}
            if job.get("job_id") == job_id:
                last = job
                state = job.get("state")
                if time.monotonic() - last_sample >= sample_every:
                    last_sample = time.monotonic()
                    progress = job.get("progress") or {}
                    sample = {"time": now(), "state": state, "stage": job.get("stage"),
                              "phase": job.get("phase"),
                              "model_elapsed_seconds": job.get("model_elapsed_seconds"),
                              "model_seconds": progress.get("model_seconds"),
                              "valid_time": job.get("valid_time"),
                              "outputs": progress.get("outputs_written")}
                    samples.append(sample)
                    self.transcript.say(f"{label}: {json.dumps(sample, default=str)}")
                if state in ("completed", "failed", "stopped"):
                    break
            elif last and last.get("state") in ("completed", "failed", "stopped"):
                break
            elif last:
                break
            if time.monotonic() > deadline:
                raise SystemExit(f"{label}: job {job_id} did not finish within {timeout}s")
            time.sleep(1.0)
        self.transcript.record("job_end", label=label, job=last, samples=samples)
        return last

    def launch(self, label: str, config: Path, plan: Path, timeout: float) -> tuple[str, dict]:
        """open_config then launch_plan, the way the desktop launches a forecast."""
        opened = self.send("open_config", {"config_path": str(config)})
        if not opened.get("ok"):
            raise SystemExit(f"{label}: open_config refused: {opened.get('message')}")
        launched = self.send("launch_plan", {
            "plan_path": str(plan), "plan_sha256": sha256(plan),
            "config_sha256": sha256(config), "target": {"kind": "local"}})
        if not launched.get("ok"):
            raise SystemExit(f"{label}: launch refused: {launched.get('message')}")
        job_id = launched["job_id"]
        job = self.wait_job(job_id, label, timeout)
        copy_receipts(self.transcript, Path(job_id), label)
        if job.get("state") != "completed":
            raise SystemExit(f"{label} ended {job.get('state')}: {job.get('error')}")
        return job_id, job

    def stop(self) -> None:
        status = self.status()
        pid = status.get("workspace_pid")
        children = subprocess.run(["pgrep", "-P", str(self.process.pid)], check=False,
                                  capture_output=True, text=True).stdout.split()
        for child in children:
            subprocess.run(["kill", child], check=False)
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.terminate()
        self.transcript.record("controller_exit", returncode=self.process.returncode,
                               workspace_pid=pid)
        self.transcript.say(f"controller exited {self.process.returncode}")
        self.transcript.copy(self.status_path, "status-final.json")


def copy_receipts(transcript: Transcript, job_dir: Path, prefix: str) -> None:
    for name in ("job.json", "process.json", "result.json", "job.log"):
        transcript.copy(job_dir / name, f"{prefix}.{name}")


def write_plan(path: Path, config: Path, output_root: Path, name: str, geog_root: str | None) -> None:
    """The gpuwm.run-plan.v1 document the desktop's review builds for a launch."""
    options = {"render_products": "none"}
    if geog_root:
        options["geog_root"] = geog_root
    plan = {"schema": "gpuwm.run-plan.v1", "name": name, "route": "prepared",
            "config": {"path": str(config)}, "output_root": str(output_root / f"run-{name}"),
            "run_options": options}
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def without_the_clock(experiment: dict) -> dict:
    document = json.loads(json.dumps(experiment))
    for key in ("name", "start_time", "run_seconds", "end_time", "source"):
        document.pop(key, None)
    for row in document.get("domains", []):
        row.pop("start_time", None)
        row.get("run", {}).pop("run_seconds", None)
    return document


def compare(saved: dict, started: dict) -> dict:
    """Which parts of two companion-query documents agree, and which differ."""
    timed = ("cycle", "hours", "out", "forecast_start_hour")
    saved_fetch = {k: v for k, v in saved.get("fetch", {}).items() if k not in timed}
    started_fetch = {k: v for k, v in started.get("fetch", {}).items() if k not in timed}
    return {
        "domains_identical": saved["domains"] == started["domains"],
        "tiles_identical": saved.get("tiles") == started.get("tiles"),
        "experiment_identical_outside_the_clock":
            without_the_clock(saved["experiment"]) == without_the_clock(started["experiment"]),
        "fetch_identical_outside_the_timed_fields": saved_fetch == started_fetch,
        "differences": {
            "experiment.name": [saved["experiment"].get("name"), started["experiment"].get("name")],
            "experiment.start_time": [saved["experiment"].get("start_time"), started["experiment"].get("start_time")],
            "experiment.run_seconds": [saved["experiment"].get("run_seconds"), started["experiment"].get("run_seconds")],
            **{f"fetch.{key}": [saved.get("fetch", {}).get(key), started.get("fetch", {}).get(key)] for key in timed},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--workspace", required=True, help="stand-in visual workspace executable")
    parser.add_argument("--root", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--geog-root", default=None)
    parser.add_argument("--point", default="39.0,-98.0")
    parser.add_argument("--hours", type=int, default=3)
    parser.add_argument("--vram-gib", default="4",
                        help="size the created parent for this much GPU memory, so the proof stays small")
    parser.add_argument("--run-timeout", type=float, default=2400.0)
    parser.add_argument("--render-products", default="t2")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    forecasts = root / "forecasts"
    library = root / "setups"
    plans = root / "plans"
    for folder in (forecasts, library, plans):
        folder.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(Path(args.evidence))
    transcript.say(f"saved setups proof on {os.uname().nodename}; root {root}")
    engine = Engine(args.python, root, transcript)
    controller = Controller(args, transcript)
    outcome = 1
    checks: dict[str, bool] = {}
    try:
        # 1. A small forecast created the way the desktop's New forecast creates one.
        proof = forecasts / "proof.toml"
        domain_args = ["domain", f"--name={SETUP_NAME}", "--source=gfs", "--cycle=latest",
                       "--forecast-start-hour=0", f"--hours={args.hours}", "--root-dx=12",
                       "--tiles=auto", "--history-interval=900", f"--out={proof}",
                       f"--point={args.point}"]
        if args.geog_root:
            domain_args.append(f"--geog-root={args.geog_root}")
        if args.vram_gib:
            domain_args.append(f"--vram-gib={args.vram_gib}")
        code, _out, err = engine.run("domain", domain_args, timeout=900)
        if code != 0 or not proof.is_file():
            raise SystemExit(f"domain door failed: {err[-2000:]}")
        transcript.copy(proof, "proof.toml")
        transcript.copy(proof.with_suffix(".namelist.wps"), "proof.namelist.wps")
        code, saved_doc = engine.json("companion-query proof", ["companion-query", str(proof)])
        if code != 0 or saved_doc.get("schema") != "arwen.companion-configuration.v1":
            raise SystemExit("companion-query refused the created forecast")
        transcript.write("proof.companion-query.json", saved_doc)
        cycle = datetime.strptime(saved_doc["fetch"]["cycle"], "%Y-%m-%dT%H")
        transcript.say(f"created {proof.name}: cycle {saved_doc['fetch']['cycle']}, "
                       f"domains {[(d['grid_id'], d['nx'], d['ny'], d['dx_m']) for d in saved_doc['domains']]}")

        # 2. Save it as a setup.
        code, saved = engine.json("save", ["companion-setups", "save", f"--config={proof}",
                                           f"--library={library}", f"--name={SETUP_NAME}"])
        if code != 0 or not saved.get("created"):
            raise SystemExit(f"save refused: {saved.get('error')}")
        folder = library / SETUP_SLUG
        listing = sorted(path.name for path in folder.iterdir())
        transcript.record("setup_folder", listing=listing, setup=saved["setup"])
        transcript.write("save.result.json", saved)
        for name in listing:
            transcript.copy(folder / name, f"setup.{name}")
        checks["setup_layout"] = {"setup.toml", "setup.json", "setup.namelist.wps"} <= set(listing)
        checks["setup_summary_names_geometry"] = "d01 12 km" in saved["setup"]["summary"]
        checks["setup_sha_matches_file"] = saved["setup"]["setup_sha256"] == sha256(folder / "setup.toml")
        checks["save_changed_only_paths"] = {c["field"] for c in saved["changes"]} <= {"fetch.out"}
        transcript.say(f"saved: {listing}; summary: {saved['setup']['summary']}")

        # 3. Start a new forecast from it at the previous cycle.
        previous = cycle - timedelta(hours=6)
        restart = forecasts / "proof-restart.toml"
        code, started = engine.json("start", [
            "companion-setups", "start", str(folder / "setup.toml"),
            f"--cycle={previous:%Y-%m-%dT%H}", "--forecast-start-hour=0", f"--hours={args.hours}",
            f"--name={RESTART_NAME}", f"--out={restart}"], timeout=900)
        if code != 0 or not started.get("created"):
            raise SystemExit(f"start refused: {started.get('error')}")
        transcript.write("start.result.json", started)
        transcript.copy(restart, "proof-restart.toml")
        transcript.copy(restart.with_suffix(".namelist.wps"), "proof-restart.namelist.wps")
        transcript.copy(restart.with_suffix(".setup-start.json"), "proof-restart.setup-start.json")
        checks["start_domains_identical"] = started["configuration"]["domains"] == saved_doc["domains"]
        checks["start_changed_only_timed_fields"] = {c["field"] for c in started["changes"]} <= {
            "experiment.name", "experiment.start_time", "experiment.run_seconds",
            "fetch.cycle", "fetch.hours", "fetch.forecast_start_hour", "fetch.out"}
        checks["start_echoes_setup_sha"] = started["setup_sha256"] == saved["setup"]["setup_sha256"]
        checks["start_not_a_forecast_launch"] = (started["forecast_started"] is False
                                                 and started["acquisition_started"] is False)
        transcript.say(f"started {restart.name} at {started['timing']}; changes "
                       f"{[c['field'] for c in started['changes']]}")

        # 4. Both forecasts run through the queue, the created one first.
        write_plan(plans / "proof.json", proof, controller.output, "proof", args.geog_root)
        write_plan(plans / "proof-restart.json", restart, controller.output, "proof-restart", args.geog_root)
        transcript.copy(plans / "proof.json", "plan.proof.json")
        transcript.copy(plans / "proof-restart.json", "plan.proof-restart.json")
        proof_job, _ = controller.launch("proof", proof, plans / "proof.json", args.run_timeout)
        restart_job, _ = controller.launch("proof-restart", restart, plans / "proof-restart.json", args.run_timeout)

        # 5. Both are listed, and their configurations differ only in the timed fields.
        rows = controller.send("browse_runs", {"target": {"kind": "local"}})["jobs"]
        listed = {row.get("job_id"): row for row in rows}
        transcript.write("browse-runs.json", rows)
        checks["browse_lists_both"] = proof_job in listed and restart_job in listed
        checks["both_completed"] = all(listed.get(job, {}).get("state") == "completed"
                                       for job in (proof_job, restart_job))
        transcript.say(f"browse_runs: {len(rows)} rows; proof={listed.get(proof_job, {}).get('state')} "
                       f"restart={listed.get(restart_job, {}).get('state')}")
        _code, saved_again = engine.json("companion-query proof (after run)", ["companion-query", str(proof)])
        _code, started_doc = engine.json("companion-query restart", ["companion-query", str(restart)])
        transcript.write("proof.companion-query.after-run.json", saved_again)
        transcript.write("proof-restart.companion-query.json", started_doc)
        comparison = compare(saved_again, started_doc)
        transcript.write("geometry-diff.json", comparison)
        transcript.record("comparison", **comparison)
        checks["geometry_identical_after_both_runs"] = (
            comparison["domains_identical"] and comparison["tiles_identical"]
            and comparison["experiment_identical_outside_the_clock"]
            and comparison["fetch_identical_outside_the_timed_fields"])
        transcript.say(f"comparison: {json.dumps(comparison['differences'], default=str)}")

        # 6. Every designed refusal through the same door, verbatim.
        setup = folder / "setup.toml"
        tampered_library = root / "tampered-library"
        if tampered_library.exists():
            shutil.rmtree(tampered_library)
        shutil.copytree(folder, tampered_library / SETUP_SLUG)
        tampered = tampered_library / SETUP_SLUG / "setup.toml"
        tampered.write_text(tampered.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        unwritten = forecasts / "never-written.toml"
        start_common = [f"--name={RESTART_NAME} again", f"--out={unwritten}"]
        cases = {
            "gfs_05z_cycle": ["companion-setups", "start", str(setup),
                              f"--cycle={previous:%Y-%m-%d}T05", f"--hours={args.hours}", *start_common],
            "hours_past_the_horizon": ["companion-setups", "start", str(setup),
                                       f"--cycle={previous:%Y-%m-%dT%H}", "--hours=400", *start_common],
            "zero_hours": ["companion-setups", "start", str(setup),
                           f"--cycle={previous:%Y-%m-%dT%H}", "--hours=0", *start_common],
            "empty_name": ["companion-setups", "start", str(setup),
                           f"--cycle={previous:%Y-%m-%dT%H}", f"--hours={args.hours}",
                           "--name=   ", f"--out={unwritten}"],
            "malformed_cycle": ["companion-setups", "start", str(setup), "--cycle=yesterday",
                                f"--hours={args.hours}", *start_common],
            "out_exists": ["companion-setups", "start", str(setup),
                           f"--cycle={previous:%Y-%m-%dT%H}", f"--hours={args.hours}",
                           f"--name={RESTART_NAME}", f"--out={restart}"],
            "save_same_name": ["companion-setups", "save", f"--config={proof}",
                               f"--library={library}", f"--name={SETUP_NAME.upper()}"],
            "save_bad_name": ["companion-setups", "save", f"--config={proof}",
                              f"--library={library}", "--name=a/b"],
            "save_without_wps": None,
            "tampered_setup": ["companion-setups", "start", str(tampered),
                               f"--cycle={previous:%Y-%m-%dT%H}", f"--hours={args.hours}", *start_common],
            "unreadable_setup": ["companion-setups", "start", str(root / "gone" / "setup.toml"),
                                 f"--cycle={previous:%Y-%m-%dT%H}", f"--hours={args.hours}", *start_common],
        }
        # A forecast whose WPS companion is missing: a copy of the created one without it.
        orphan = forecasts / "orphan.toml"
        shutil.copy2(proof, orphan)
        cases["save_without_wps"] = ["companion-setups", "save", f"--config={orphan}",
                                     f"--library={library}", "--name=Orphan"]
        refusals = {}
        before_restart = sha256(restart)
        for key, argv in cases.items():
            code, document = engine.json(f"refusal {key}", argv, timeout=300)
            refusals[key] = {"argv": argv, "exit": code, "created": document.get("created"),
                             "error": document.get("error")}
            transcript.say(f"refusal {key}: exit {code}: {document.get('error')!r}")
        transcript.write("refusals.json", refusals)
        checks["every_refusal_exit_1_created_false_with_sentence"] = all(
            r["exit"] == 1 and r["created"] is False and isinstance(r["error"], str) and r["error"]
            for r in refusals.values())
        checks["refusals_wrote_nothing"] = (
            not unwritten.exists() and not unwritten.with_suffix(".namelist.wps").exists()
            and not unwritten.with_suffix(".setup-start.json").exists()
            and sha256(restart) == before_restart
            and sorted(path.name for path in library.iterdir()) == [SETUP_SLUG])
        checks["refusal_sentences_as_designed"] = (
            "GFS cycles run at 00/06/12/18 UTC only" in (refusals["gfs_05z_cycle"]["error"] or "")
            and "384" in (refusals["hours_past_the_horizon"]["error"] or "")
            and refusals["zero_hours"]["error"] == "Choose a positive forecast duration in hours."
            and refusals["empty_name"]["error"] == "Type a forecast name."
            and refusals["out_exists"]["error"] == f"A configuration already exists at {restart}. Choose a new name, or use Open configuration."
            and refusals["save_same_name"]["error"] == f'A saved setup named "{SETUP_NAME}" already exists in this library. Choose another name, or delete that setup first.'
            and refusals["save_bad_name"]["error"] == "Type a name for this setup: 1 to 80 characters, no slashes, colons or quotes."
            and refusals["save_without_wps"]["error"] == "This forecast has no orphan.namelist.wps beside it, so the staged route could not run a forecast started from it. Create the forecast again, then save it."
            and refusals["tampered_setup"]["error"] == f'The saved setup "{SETUP_NAME}" was changed after it was saved: setup.toml no longer matches setup.json. Delete it from Saved setups and save the forecast again.'
            and (refusals["unreadable_setup"]["error"] or "").startswith("The saved setup cannot be read: ")
            and (refusals["unreadable_setup"]["error"] or "").endswith(" Delete it from Saved setups and save the forecast again."))

        # 7. One frame of the started forecast, rendered by the Rust renderer.
        start_time = datetime.fromisoformat(started_doc["experiment"]["start_time"].replace("Z", ""))
        last_valid = start_time + timedelta(hours=args.hours)
        frames = sorted(glob.glob(str(controller.output / "run-proof-restart" / "**" /
                                      f"wrfout_d01_{last_valid:%Y-%m-%d_%H_%M_%S}"), recursive=True))
        if not frames:
            frames = sorted(glob.glob(str(controller.output / "run-proof-restart" / "**" / "wrfout_d01_*"),
                                      recursive=True))
        transcript.record("frames", frames=frames)
        render_dir = transcript.directory / "render"
        if frames:
            code, out, err = engine.run("render", ["render", frames[-1], "--engine=rust",
                                                   f"--products={args.render_products}",
                                                   f"--out={render_dir}"], timeout=900)
            rendered = sorted(str(p) for p in render_dir.rglob("*.png"))
            transcript.record("render", frame=frames[-1], exit=code, pngs=rendered)
            checks["rendered_a_frame_with_the_rust_engine"] = code == 0 and bool(rendered)
            transcript.say(f"render: exit {code}, {len(rendered)} png(s) from {Path(frames[-1]).name}")
        else:
            checks["rendered_a_frame_with_the_rust_engine"] = False
            transcript.say("render: no wrfout frame of the started forecast was found")

        outcome = 0 if all(checks.values()) else 2
    finally:
        controller.stop()
        summary = {"outcome": outcome, "checks": checks, "records": len(transcript.records),
                   "root": str(root)}
        transcript.write("summary.json", summary)
        transcript.say(f"summary: {json.dumps(summary)}")
    return outcome


if __name__ == "__main__":
    sys.exit(main())
