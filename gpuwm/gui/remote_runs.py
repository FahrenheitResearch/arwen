"""Forecasts on other machines, render workers, and the follower that mirrors both.

A forecast started on a machine has a mirror folder here,
``<root>/<run>/``, holding:

- ``gui-machine.json``: which machine, its run folder there, the last
  thing the machine said (alive or not, the job's state), and how much
  of the event log has been copied;
- ``events.jsonl`` and ``run-progress.json``: copied as the machine
  writes them, so the page's event stream and progress work unchanged;
- ``engine.log``: the tail of the engine's output there.

A render request (``gui-render.json``) names the machine that draws.
The follower feeds it every frame the forecast commits (the run's own
``output_committed`` events), copying frames through this computer when
the render machine is another one, and copies each picture back into
``<root>/<run>/render-<machine>/<domain>/<product>/<valid-day>/`` the
moment it is listed, so the Timeline shows it while the forecast runs.

The follower (``gpuwm machines follow RUN``) runs detached like a
forecast does: closing the page does not stop it, and it ends by itself
when the forecast has ended and every picture is here.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import re
from pathlib import Path, PurePosixPath
import time
from typing import Any

from . import runs
from .files import long_path, read_json, write_json
from .machines import (FOLLOW_TICK_S, LOCAL, MACHINE, REMOTE_MANIFEST, RENDER, Machine,
                       MachineError, Registry, extract, relay)

END_STATES = runs.REMOTE_END_STATES
#: A follower gives up after this many consecutive failed ticks (about
#: ten minutes), so a machine switched off does not keep a process here
#: polling it forever; the map then says its updates stopped, and its
#: Resume updates button starts a new one.
MAX_FAILED_TICKS = 120
WAIT_TICK_S = 30.0


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def remote_rundir(machine: Machine, run: str) -> str:
    return f"{machine.workspace.rstrip('/')}/runs/{run}"


def render_job_name(run: str, machine: str) -> str:
    return f"{run}--{machine}"[:64]


# ------------------------------------------------------------------ launching

def launch_forecast(registry: Registry, rundir: Path, machine_name: str, plan: dict[str, Any],
                    region: dict[str, Any], *, wait_min: int = 0, bound_min: int = 120,
                    text_files: dict[str, str] | None = None) -> dict[str, Any]:
    """Send the plan to the machine and start it there; writes the mirror's gui-machine.json.

    ``text_files`` are files the plan's configuration names beside itself (a
    storm-following layout's TOML, Vtable and WPS namelist), written there as
    the text they are.
    """

    machine = registry.get(machine_name)
    there = remote_rundir(machine, rundir.name)
    answer = machine.call("launch", "--python", machine.python, "--run", rundir.name,
                          "--owner-tag", str(machine.row.get("owner_tag") or "gpuwm-gui"),
                          *(["--owner-file", str(machine.row["owner_file"])] if machine.row.get("owner_file") else []),
                          payload={"files": {"plan.json": plan, "region.geojson": region},
                                   **({"text_files": dict(text_files)} if text_files else {}),
                                   "wait_min": wait_min, "bound_min": bound_min,
                                   "env": machine.row.get("env") or {}}, timeout=120)
    document = {"machine": machine.name, "host": machine.row.get("host"), "remote_rundir": answer.get("rundir") or there,
                "launched_utc": _utc(), "events_offset": 0, "alive": True,
                "job": answer.get("job"), "command": answer.get("command")}
    try:
        write_json(rundir / MACHINE, document)
    except OSError as error:
        # The machine has the forecast.  Raised, the start read as failed: the page removed the folder of a
        # forecast running there, and the queue launched it there a second time.
        document["unkept"] = getattr(error, "strerror", None) or str(error)
    return document


def plan_for(machine: Machine, plan: dict[str, Any], run: str) -> dict[str, Any]:
    """The plan with its paths moved to the machine and the machine's own data folders."""

    there = PurePosixPath(remote_rundir(machine, run).replace("~/", "/__HOME__/"))
    out = json.loads(json.dumps(plan))
    config = out.get("config") or {}
    if isinstance(config.get("intent"), dict):
        config["intent"]["polygon"] = str(there / "region.geojson")
    elif config.get("path"):
        # A configuration file (a storm-following layout) is sent beside the plan, with its companion files.
        config["path"] = str(there / re.split(r"[\\/]", str(config["path"]))[-1])
    out["output_root"] = str(there)
    options = out.setdefault("run_options", {})
    for key in ("geog_root", "data_dir"):
        if machine.row.get(key):
            options[key] = machine.row[key]
    text = json.dumps(out).replace("/__HOME__/", "~/")
    return json.loads(text)


def request_render(rundir: Path, machine_name: str, *, products: str | None = None,
                   render_section: str | None = None) -> dict[str, Any]:
    current = read_json(rundir / RENDER, default=None)
    # A draw already going on that machine is the one asked for; one that finished, failed or was stopped is
    # asked for again from no frames, so pressing Draw after a failure tries again instead of showing the failure.
    if isinstance(current, dict) and current.get("machine") == machine_name \
            and current.get("state") not in ("finished", "failed", "stopped"):
        if current.get("render_section") != render_section:
            raise MachineError("This machine is already drawing another section line. Reusing that draw "
                               "would show the old slice instead of the requested one.",
                               "Wait for this draw to finish, then request the new line.")
        return current
    document = {"machine": machine_name, "job": render_job_name(rundir.name, machine_name),
                "products": products, "render_section": render_section,
                "state": "requested", "requested_utc": _utc(),
                "fed": [], "pictures": 0, "bytes_relayed": 0, "bytes_pulled": 0}
    write_json(rundir / RENDER, document)
    return document


def follower_alive(rundir: Path) -> bool:
    return runs.follower_alive(rundir)


def follow_argv(root: Path, run_id: str) -> list[str]:
    from .jobs import engine_argv

    return engine_argv("machines", "follow", "--root", str(root), run_id)


# ------------------------------------------------------------------ the follower

def relay_present(source: Machine, dest: Machine, frames: list[str], inbox: str) -> tuple[int, list[str]]:
    """Copy the frames ``source`` still has; returns (bytes moved, frames it no longer had).

    A copy stops when a listed file is not there, and names every such
    file.  A forecast's frames can go after they were committed (removed
    to free disk, or by hand), and its events list them for ever, so
    without setting them aside every later pass stopped on the same
    frame until the follower gave up, and the mirror stopped with it.
    """

    remaining = list(frames)
    lost: list[str] = []
    while remaining:
        try:
            return relay(source, dest, remaining, inbox), lost
        except MachineError as error:
            names = set(error.missing)
            went = [frame for frame in remaining if PurePosixPath(frame.replace("\\", "/")).name in names]
            if not went:
                raise
            lost.extend(went)
            remaining = [frame for frame in remaining if frame not in went]
    return 0, lost


def gone_words(count: int, source: Machine) -> str:
    where = "this computer" if source.is_local else source.name
    if count == 1:
        return f"1 frame was gone from {where} before it could be drawn."
    return f"{count} frames were gone from {where} before they could be drawn."


def _last_lines(text: Any, count: int = 4) -> str:
    lines = [line for line in str(text or "").splitlines() if line.strip() and not line.startswith("#")]
    # The log's tail can start inside a traceback, below its first line.
    if any(line.startswith("Traceback") or line.lstrip().startswith('File "') for line in lines):
        # A traceback says where the code was; the page says why, which is its last line without the class name.
        reason = [line for line in lines if not line.startswith((" ", "\t", "Traceback"))][-1:]
        return re.sub(r"^[A-Za-z_.]*(?:Error|Exception):\s*", "", reason[0]) if reason else ""
    return " ".join(lines[-count:])[-800:]


class Follower:
    def __init__(self, root: Path, run_id: str, registry: Registry | None = None) -> None:
        self.root = root
        self.rundir = runs.existing_run(root, run_id)
        self.registry = registry or Registry()
        self.failed = 0

    # -------------------------------------------------------------- forecast mirror

    def sync_forecast(self) -> dict[str, Any] | None:
        mirror = read_json(self.rundir / MACHINE, default=None)
        if not isinstance(mirror, dict):
            return None
        if mirror.get("ended"):
            return mirror
        machine = self.registry.get(mirror["machine"])
        offset = int(mirror.get("events_offset") or 0)
        while True:
            snap = machine.call("snapshot", "--run", self.rundir.name, "--offset", str(offset))
            chunk = base64.b64decode(snap.get("events_b64") or "")
            if chunk:
                with (self.rundir / runs.EVENTS).open("ab") as stream:
                    stream.write(chunk)
                offset += len(chunk)
            if not chunk or offset >= int(snap.get("events_size") or 0):
                break
        if snap.get("heartbeat") is not None:
            write_json(self.rundir / runs.HEARTBEAT, snap["heartbeat"])
        if snap.get("manifest") is not None:
            write_json(self.rundir / REMOTE_MANIFEST, snap["manifest"])
        if snap.get("engine_log"):
            (self.rundir / "engine.log").write_text(snap["engine_log"], encoding="utf-8")
        job = snap.get("job") or {}
        ended = not snap.get("alive") and job.get("state") in END_STATES
        mirror.update(events_offset=offset, alive=bool(snap.get("alive")), job=job,
                      checked_utc=_utc(), ended=ended)
        if ended:
            mirror["ended_utc"] = job.get("ended_utc") or _utc()
        write_json(self.rundir / MACHINE, mirror)
        return mirror

    def source_ended(self, mirror: dict[str, Any] | None) -> bool:
        if mirror is not None:
            return bool(mirror.get("ended"))
        return runs.status(self.rundir)["state"] in ("finished", "failed", "stopped", "stale", "imported")

    def committed_frames(self) -> list[str]:
        mirror = read_json(self.rundir / MACHINE, default=None)
        base = str(mirror.get("remote_rundir") or "") if isinstance(mirror, dict) else str(self.rundir)
        frames = []
        for record in runs.event_records(self.rundir / runs.EVENTS) if (self.rundir / runs.EVENTS).is_file() else []:
            if record.get("event") == "output_committed" and record.get("path"):
                path = str(record["path"])
                if not (path.startswith("/") or Path(path).is_absolute()):
                    path = f"{base.rstrip('/')}/{path}" if isinstance(mirror, dict) else str(Path(base) / path)
                frames.append(path)
        seen = set()
        return [f for f in frames if not (f in seen or seen.add(f))]

    # -------------------------------------------------------------- render worker

    def sync_render(self, mirror: dict[str, Any] | None) -> dict[str, Any] | None:
        render = read_json(self.rundir / RENDER, default=None)
        if not isinstance(render, dict) or render.get("state") == "finished":
            return render
        worker = self.registry.get(render["machine"])
        source = self.registry.get(mirror["machine"]) if mirror else self.registry.get(LOCAL)
        job = render["job"]
        frames = self.committed_frames()
        if render.get("state") == "requested" and not frames and not self.source_ended(mirror):
            return render  # nothing to draw yet; the worker starts with the first frame
        if render.get("state") == "requested":
            worker.call("render-start", "--job", job, "--python", worker.python,
                        payload={"run": self.rundir.name, "products": render.get("products"),
                                 "render_section": render.get("render_section"),
                                 "env": worker.row.get("env") or {}, "fresh": True})
            render.update(state="rendering", started_utc=_utc())
            write_json(self.rundir / RENDER, render)
        fed = list(render.get("fed") or [])
        gone = list(render.get("gone") or [])
        new = [frame for frame in frames if frame not in fed and frame not in gone]
        if new:
            began = time.monotonic()
            if source.name == worker.name:
                worker.call("render-feed", "--job", job, payload=new)
            else:
                inbox = f"{worker.workspace.rstrip('/')}/renders/{job}/inbox"
                moved, lost = relay_present(source, worker, new, inbox)
                if lost:
                    gone.extend(lost)
                    render["gone"] = gone
                    new = [frame for frame in new if frame not in lost]
                render["bytes_relayed"] = int(render.get("bytes_relayed") or 0) + moved
                render.setdefault("relays", []).append({"frames": len(new), "bytes": moved,
                                                        "seconds": round(time.monotonic() - began, 1),
                                                        "utc": _utc()})
                if new:
                    worker.call("render-feed", "--job", job,
                                payload=[PurePosixPath(frame.replace("\\", "/")).name for frame in new])
            fed.extend(new)
            render["fed"] = fed
            if new:
                render.setdefault("first_fed_utc", _utc())
            write_json(self.rundir / RENDER, render)
        if self.source_ended(mirror) and not new and not render.get("end_sent"):
            worker.call("render-end", "--job", job)
            render["end_sent"] = True
        listing = worker.call("render-list", "--job", job)
        target = self.rundir / f"render-{worker.name}"
        missing = []
        for relative, size in [*(listing.get("files") or []), *(listing.get("manifests") or [])]:
            local = target.joinpath(*PurePosixPath(relative).parts)
            try:
                if long_path(local).stat().st_size == size:
                    continue
            except OSError:
                pass
            missing.append(relative)
        if missing:
            out_dir = str((listing.get("job") or {}).get("out") or "")
            pulled = relay(worker, self.registry.get(LOCAL), missing, str(target), base=out_dir)
            render["bytes_pulled"] = int(render.get("bytes_pulled") or 0) + pulled
            render.setdefault("first_picture_utc", _utc())
        render["pictures"] = len(listing.get("files") or [])
        render["worker_state"] = (listing.get("job") or {}).get("state")
        render["batches"] = (listing.get("job") or {}).get("batches") or []
        render["checked_utc"] = _utc()
        failed = [batch for batch in render["batches"] if batch.get("exit_code")]
        said = [gone_words(len(gone), source)] if gone else []
        if failed:
            said.append(f"{len(failed)} of {len(render['batches'])} render batches failed. "
                        + _last_lines(listing.get("log")))
        if said:
            render["message"] = " ".join(said)
        if render["worker_state"] == "finished" and not missing:
            render["state"] = "failed" if failed and not render["pictures"] else "finished"
            render["finished_utc"] = _utc()
        elif not listing.get("alive") and render["worker_state"] not in ("finished",):
            render["state"] = "failed"
            render["message"] = "The render worker stopped. " + _last_lines(listing.get("log"))
        write_json(self.rundir / RENDER, render)
        return render

    def tick(self) -> bool:
        """One pass; True while there is more to follow."""

        mirror = self.sync_forecast()
        render = self.sync_render(mirror)
        return runs.follow_needed(mirror, render)

    def run(self) -> int:
        while True:
            try:
                more = self.tick()
                self.failed = 0
            except MachineError as error:
                self.failed += 1
                print(f"{_utc()} {error.message} {error.fix}", flush=True)
                if self.failed >= MAX_FAILED_TICKS:
                    return 3
                more = True
            if not more:
                return 0
            time.sleep(self.pause())

    def pause(self) -> float:
        """Five seconds while there is something to copy; thirty while the run waits for a card."""

        mirror = read_json(self.rundir / MACHINE, default=None)
        waiting = isinstance(mirror, dict) and (mirror.get("job") or {}).get("state") == "waiting-for-card"
        render = read_json(self.rundir / RENDER, default=None)
        drawing = isinstance(render, dict) and render.get("state") == "rendering" and render.get("fed")
        return WAIT_TICK_S if waiting and not drawing else FOLLOW_TICK_S


__all__ = ["Follower", "follow_argv", "follower_alive", "launch_forecast", "plan_for",
           "remote_rundir", "render_job_name", "request_render"]
