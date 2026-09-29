"""The page's JSON endpoints, one class with a mixin per screen.

Reads are GETs; every action is a POST whose reply carries ``argv`` and
``command``: the exact ``gpuwm`` command it ran.  Every POST takes
``"dry_run": true``, which checks the request and returns that command
with nothing run, nothing written and no folder made: what "Show
command" shows before a button is pressed.

Run ids are NAME or FOLDER/NAME under ``--root``; in a URL the id is one
path segment with its slash written ``%2F``.

    GET  /api/session                      version, token, root, runner, host, port
    GET  /api/copy                         the page's words (gui/copy/*.json)
    GET  /api/runs                         one row per run folder under the root, newest first; past the
                                           list's limit "truncated" is true and "total" counts them all
    GET  /api/runs/RUN                     the row, the plan, the command log
    GET  /api/runs/RUN/status              the derived state alone
    GET  /api/runs/RUN/pictures            domains, products, days of the render tree
    GET  /api/runs/RUN/pictures/list       ?domain=&episode=&product=&day= the pictures, oldest first,
                                           with their valid times and the renderer's georeferences
    GET  /api/runs/RUN/map                 where the run sits: projection, domains, requested box
    GET  /api/runs/RUN/files/PATH          one file of the run, served as it is
    GET  /api/runs/RUN/events              Server-Sent Events (served by :mod:`.server`)
    GET  /api/sources                      runnable sources, their route, and their physics menus
    GET  /api/sources/availability         ?time=YYYY-MM-DDTHH&hours=N whether each source has that start, and why not;
                                           answers at once, each row "checking" until its probe is in
                                           (ask again while "pending" is above zero)
    GET  /api/system                       this computer's cards (NVML only)
    GET  /api/library                      the Weather Library's main page: kinds, places, featured, recent changes
    GET  /api/library/event/ID             an event page with every source it cites and the runs of it
    GET  /api/library/kind/ID              a phenomenon page and every event of that kind
    GET  /api/library/place/ID             a place page, its events rarest first
    GET  /api/library/places               every place, grouped by kind
    GET  /api/library/run/RUN              a run as an article: facts citing its own files, events it covers
    GET  /api/library/search               ?q=&type=&region=&decade=&season=&place=&seed=&sort= over the store
    GET  /api/library/changes              new events and runs, newest first
    GET  /api/library/recipe/ID            ?card=GB an event's run recipe for that card size, for New forecast
    POST /api/library/simulate             start an event's best run for one card size
    GET  /api/physics                      every physics family, scheme, suite and preset
    POST /api/physics/check                does this combination run; if not, why and what does
    POST /api/create/fit                   the wizard's fit for a draft (run-plan --resolve)
    POST /api/create/start                 write the run folder and launch run-plan ("queue": true waits for the card,
                                           and for a start from the last day no check has confirmed yet)
    GET  /api/queue                        the queued forecasts in order, the card's state, when each expects to start;
                                           ?machine=&need_gib=&source=&cycle=&hours= also where and when a new one
                                           would start, and whether Start or only Queue it takes that start
    POST /api/queue/RUN/up|down|remove     reorder or remove one queued forecast
    POST /api/runs/RUN/start               launch a ready run's plan.json
    POST /api/runs/RUN/stop                stop a running run (on whichever machine runs it)
    POST /api/runs/RUN/render              draw a run on a machine; pictures stream back
    POST /api/runs/RUN/follow              follow a machine's run or drawing again after its follower ended
    GET  /api/runs/RUN/downscale           what a finished run offers a finer forecast inside it, and why not
    POST /api/runs/RUN/downscale           {"mode": "plan"} reviews a finer forecast (the engine's --dry-run),
                                           {"mode": "run"} starts it: the gpuwm downscale line Desktop sends

    GET  /api/machines                     this computer plus every machine row, with live state
    GET  /api/machines/NAME                one machine, checked now
    GET  /api/providers                    the cloud providers a row may name
    POST /api/machines/add                 add a row (checked on add; key auth only)
    POST /api/machines/NAME/check          check again now
    POST /api/machines/NAME/remove         remove the row
    POST /api/machines/NAME/install        install this computer's gpuwm version there
    POST /api/machines/NAME/start|stop|terminate    a cloud machine (dry_run prints every call)
    GET  /api/assistant ...                the assistant (see gpuwm/gui/assistant/service.py)
    POST /api/assistant/...

The Machines section of docs/dev/GUI-API.md has every field.

Errors say what happened and what to do: ``message`` and ``fix``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import errno
import functools
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import threading
import time
import traceback
from typing import Any, Iterator
from urllib.parse import quote, unquote
import uuid

from . import downscale, frames, runs
from .wiki import RUN_LINK, Wiki, recipe_of, recipe_row
from .machines import LOCAL, MachineError, Registry, check_row
from .files import (COPY_DIR, SERVER_DIR, PathRefused, plain_message, read_json, require_name, safe_file, scrub,
                    utc_text, write_json)
from .jobs import (Refused, Runner, card_taken_words, display, engine_argv, holder_name, log_command, plain_command,
                   plain_log, record_stop, started_text)

PLAN_SCHEMA = "gpuwm.run-plan.v1"
#: The card classes New forecast sizes for, smallest first.  A row here is all a new class needs: the engine takes
#: any size spelled NNgb.
CARDS = ("8gb", "12gb", "16gb", "24gb", "32gb")
CACHE_S = 600.0
#: Engine answers also kept under the runs folder, so a restarted page server answers its first page from the
#: last one while it asks the engine again: the source list and the physics sets, which only a new engine changes.
SAVED_KEYS = ("sources", "profiles")
DX_MIN_KM = 0.5
#: The physics composer's checked choice for a run started from New forecast, beside its plan.
PHYSICS_CHOICE = "gui-physics.json"
SYSTEM_CACHE_S = 60.0
#: A running run's card is walked again at most this often: the renderer
#: adds pictures between events.
CARD_LIVE_S = 10.0


@dataclass
class Reply:
    status: int = 200
    body: Any = None
    file: Path | None = None
    headers: dict[str, str] = field(default_factory=dict)
    run_file: bool = False


class ApiError(Exception):
    def __init__(self, status: int, message: str, fix: str = "", **extra: Any) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.fix = fix
        self.extra = extra
        #: The engine's own refusal text, when there is one, for in-process readers; never sent to the page.
        self.engine_text: str | None = None

    def reply(self) -> Reply:
        # A path taken out of either is in the page server's log, with both, on one line.
        whole = f"{self.message} {self.fix}".rstrip()
        message, fix = scrub(self.message, whole=whole), scrub(self.fix, whole=whole)
        return Reply(self.status, {"ok": False, "message": message, "fix": fix, **self.extra})




def run_url(run_id: str) -> str:
    return quote(run_id, safe="")


def refuse_without_geography(runner: Runner, root: str | Path | None = None) -> None:
    """Refuse a forecast on this computer, before anything is written, when its geography data is not set up.

    Every forecast builds its grids' terrain, land use and soil from that tree.  Without it the start
    buttons used to write the run folder and launch a run that failed at once, and the page then showed
    "fix the plan document" with the folder's path taken out.  ``root`` is the tree a plan names for itself;
    None is the one ``gpuwm fetch-geog`` stages into.
    """

    missing = runner.missing_geography(root)
    if missing is None:
        return
    from gpuwm.geog_assets import GEOG_CONSUMER_WRF, WRF_FETCH_COMMAND, datasets_required_by, size_phrase

    needs = "Every forecast builds its terrain, land use and soil from it"
    what = (f"This computer has no geography data yet. {needs}." if missing["absent"] else
            f"The geography data on this computer is incomplete ({'; '.join(missing['gaps'])}). {needs}.")
    raise ApiError(409, what, f"Set it up once in a terminal with {WRF_FETCH_COMMAND} (it downloads "
                              f"{size_phrase(datasets_required_by(GEOG_CONSUMER_WRF))}), then start the "
                              "forecast again.")


def plan_geography(plan: Any, base: Path) -> Path | None:
    """The geography tree a plan names for itself, as the engine reads it, or None for the default one.

    ``base`` is the plan's folder, which a relative path is read from.
    """

    if not isinstance(plan, dict):
        return None
    options = plan.get("run_options") if isinstance(plan.get("run_options"), dict) else {}
    config = plan.get("config") if isinstance(plan.get("config"), dict) else {}
    intent = config.get("intent") if isinstance(config.get("intent"), dict) else {}
    named = options.get("geog_root") or intent.get("geog_root")
    if not isinstance(named, str) or not named:
        return None
    tree = Path(named)
    return tree if tree.is_absolute() else Path(base) / tree


def run_title(rundir: Path, run_id: str) -> str:
    """The name a run reads under: its event's, when it was started from an event page, else its id."""

    link = read_json(rundir / RUN_LINK, default=None)
    return (link.get("title") if isinstance(link, dict) else None) or run_id


def reserve_run(rundir: Path) -> None:
    """Claim a new run folder in one step, before anything is written into it.

    The name was checked free earlier in the request, but two tabs (or two
    page servers on one forecasts folder) could both pass that check: both
    starts then wrote the same folder's plan and both answered 200, one
    plan replacing the other under a run already started.  Creating the
    folder is the claim, so the second start is refused in words.
    """

    try:
        rundir.parent.mkdir(parents=True, exist_ok=True)
        rundir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ApiError(409, f"A run called {rundir.name} already exists.",
                       "Give this forecast another name.") from error


#: The errors that mean the disk, or this user's share of it, is full.
_NO_SPACE = frozenset(code for code in (errno.ENOSPC, getattr(errno, "EDQUOT", None)) if code is not None)


@contextmanager
def releasing_on_failure(rundir: Path) -> Iterator[None]:
    """What a start writes into the folder it claimed (:func:`reserve_run`) before a launch or the queue has it.

    Whatever stops those writes, nothing ran and nothing is in line, so the folder goes and the same name can be
    started or queued again.  Only a refusal released it before: a disk that filled while the plan was written
    answered with the page server's generic error and kept the folder, and once there was room again the same
    name was refused as a run that already exists.  A write that fails is said in words, with 507 for a full disk.
    """

    try:
        yield
    except Exception as error:
        shutil.rmtree(rundir, ignore_errors=True)
        if not isinstance(error, OSError):
            raise
        print(f"gpuwm gui: {rundir.name} could not be saved, and nothing started: {error!r}", file=sys.stderr,
              flush=True)
        if error.errno in _NO_SPACE:
            raise ApiError(507, "The disk that holds your forecasts is full, so this forecast could not be saved. "
                                "Nothing started.",
                           "Free some space on it and try again with the same name.") from error
        raise ApiError(500, f"This forecast could not be saved ({error.strerror or type(error).__name__}). Nothing "
                            "started.",
                       "Make sure the forecasts folder can be written to, then try again with the same "
                       "name.") from error


@functools.lru_cache(maxsize=None)
def _start_facts(source: str) -> dict[str, Any]:
    """A source's start hours for a six-hour forecast and its usual publication delay: fixed facts of its schedule."""

    from gpuwm.source_availability import availability

    try:
        document = availability(source, 6, now=datetime(2026, 1, 1))
    except (ValueError, KeyError):
        return {"cycle_hours": [], "usual_delay_hours": None}
    return {"cycle_hours": list(document["cycle_hours"]), "usual_delay_hours": document["usual_delay_hours"]}


def _data_of(draft: dict[str, Any]) -> dict[str, Any]:
    """The start a queued forecast's data check asks about, and why it waits when it does (:mod:`.queue`)."""

    return {"source": draft["source"], "cycle": draft["cycle"], "hours": draft["hours"],
            "why": draft.get("waits_for_data")}


def _dry(argv: list[str], **extra: Any) -> Reply:
    return Reply(200, {"ok": True, "dry_run": True, "argv": argv, "command": display(argv),
                       "message": "Nothing ran. This is the command the button runs.", **extra})


# ------------------------------------------------------------------ runs

class RunsMixin:
    root: Path
    runner: Runner

    def runs(self) -> dict[str, Any]:
        """The newest runs under the root, each read on its own.

        A folder that cannot be read lists as ``unreadable`` with a plain
        reason (the full one goes to the server's log) and the rest still
        load: one damaged plan used to answer 500 for every run.  Past
        ``runs.LIST_LIMIT`` the oldest are left off and ``truncated`` says so.
        """

        found = list(runs.iter_runs(self.root))
        rows = []
        for path in found[:runs.LIST_LIMIT]:
            try:
                row = runs.summarize(self.root, path)
                row["card"] = self._card(path, row["status"])
                row["title"] = run_title(path, row["id"])
            except Exception as error:  # noqa: BLE001 - one folder never takes the list down
                print(f"gpuwm gui: could not read the run folder {path}: {error!r}", file=sys.stderr, flush=True)
                row = runs.unreadable(self.root, path)
            row["url"] = run_url(row["id"])
            rows.append(row)
        return {"root": str(self.root), "runs": rows, "total": len(found), "limit": runs.LIST_LIMIT,
                "truncated": len(found) > runs.LIST_LIMIT}

    def _card(self, rundir: Path, info: dict[str, Any]) -> dict[str, Any]:
        """A run's card facts, walked again only when the run has written something new.

        The list refreshes every few seconds and a finished run's render
        tree holds thousands of pictures, so the walk is cached on what
        changes when a picture lands: the event count, the state and the
        folder's newest marker file.
        """

        key = (info.get("last_sequence"), info.get("state"), info.get("outputs_committed"),
               runs.folder_mtime(rundir))
        with self._cache_lock:
            hit = self._cards.get(str(rundir))
        fresh = hit is not None and (info.get("state") != "running" or time.monotonic() - hit[2] < CARD_LIVE_S)
        if hit is not None and hit[0] == key and fresh:
            return hit[1]
        facts = frames.card(rundir)
        intent = runs.plan_intent(read_json(rundir / runs.PLAN, default=None))
        root_dx = runs.number(intent.get("root_dx_km"))
        if not facts["dx_km"] and root_dx and root_dx > 0:
            facts["dx_km"] = [root_dx]
        if not facts["dx_km"]:
            # no pictures and no planned spacing (a downscale has no plan.json): the spacing the run resolved to
            facts["dx_km"] = (info.get("grids") or {}).get("grids_km") or []
        facts["hours"] = None if not info.get("run_seconds") else round(float(info["run_seconds"]) / 3600.0, 2)
        # The vertical level count: what the run resolved to once it has, what its plan asked for before that.
        events = rundir / runs.EVENTS
        levels = list((runs.event_facts(events) if events.is_file() else {}).get("levels") or [])
        planned_nz = runs.number(intent.get("nz"))
        if not levels and planned_nz and planned_nz > 0 and planned_nz.is_integer():
            levels = [int(planned_nz)]
        facts["levels"] = levels
        with self._cache_lock:
            self._cards[str(rundir)] = (key, facts, time.monotonic())
        return facts

    def run_detail(self, run_id: str, rundir: Path) -> dict[str, Any]:
        row = runs.summarize(self.root, rundir)
        row["url"] = run_url(run_id)
        row["card"] = self._card(rundir, row["status"])
        row["title"] = run_title(rundir, row["id"])
        plan = read_json(rundir / runs.PLAN, default=None)
        log = ""
        try:
            # A log another program appended to may not be UTF-8; its readable lines still show.
            log = (rundir / runs.COMMANDS_LOG).read_text(encoding="utf-8", errors="replace")[-8000:]
        except OSError:
            # a downscale the page started cannot log into the folder it claims: its command is in the start record
            record = runs.start_record(rundir)
            if record and record.get("command"):
                log = f"# {record.get('started_utc') or ''}  Downscale of {record.get('parent')}\n{record['command']}\n"
        row["plan"] = plan
        row["commands_log"] = plain_log(log, self.root)
        row["start_argv"] = engine_argv("run-plan", str(rundir / runs.PLAN)) if plan else None
        row["start_command"] = display(row["start_argv"]) if plan else None
        row["reattach"] = {name: str(rundir / name) for name in (runs.MANIFEST, runs.HEARTBEAT, runs.EVENTS)}
        row["remote"] = read_json(rundir / runs.REMOTE, default=None)
        render = read_json(rundir / runs.RENDER, default=None)
        if isinstance(render, dict):
            render = {key: value for key, value in render.items() if key != "fed"}
        row["render_job"] = render
        return row

    def start_ready(self, run_id: str, rundir: Path, payload: dict[str, Any], dry: bool) -> Reply:
        if not (rundir / runs.PLAN).is_file():
            raise ApiError(409, "This run has no plan.json to start.", "Make one on the Create page.")
        remote = read_json(rundir / runs.REMOTE, default=None)
        if isinstance(remote, dict):
            # Its plan.json holds the other machine's paths (output_root and
            # polygon under that machine's workspace): started here it would
            # run on this computer's card and write to folders that are not here.
            machine = remote.get("machine") or "another machine"
            raise ApiError(409, f"This forecast belongs to {machine}, so it cannot start on this computer.",
                           f"Make a new forecast on {machine} from the Create page.")
        info = runs.status(rundir)
        if info["state"] != "ready":
            raise ApiError(409, f"This run is {info['state']}, so it cannot be started again.",
                           "Make a new run on the Create page.")
        # Refused while it is still Ready, so it can be started once the geography is set up.
        refuse_without_geography(self.runner, plan_geography(read_json(rundir / runs.PLAN, default=None), rundir))
        argv = engine_argv("run-plan", str(rundir / runs.PLAN))
        if dry:
            return _dry(argv, run=run_id)
        return self._launch(run_id, rundir, argv)

    def _launch(self, run_id: str, rundir: Path, argv: list[str], need_gib: float | None = None) -> Reply:
        # On a card shared through an OWNER file, the start takes the card there first (see queue.py).
        owner_file = self.queue.owner_file
        return self.queue.launch_holding_card(
            run_id, lambda: self._launch_now(run_id, rundir, argv, owner_file=owner_file), need_gib)

    def _launch_now(self, run_id: str, rundir: Path, argv: list[str], owner_file: str | None = None) -> Reply:
        try:
            # On a shared card the run's own wrapper is told of the OWNER file, and removes its line when it ends.
            job = self.runner.launch(rundir, argv, owner_file=owner_file) if owner_file else self.runner.launch(rundir, argv)
        except Refused as error:
            raise ApiError(409, str(error), "Wait for the running forecast to finish, or stop it.") from error
        # The forecast runs from here on, so what follows is said beside the start and never raised as a failed
        # one: raised, the queue put the running forecast back in line and started it a second time, and the
        # card's line was let go while it ran.
        freed, unsaid = None, []
        if job.get("unkept"):
            print(f"gpuwm gui: {run_id} started, but its job record could not be written: {job['unkept']}",
                  file=sys.stderr, flush=True)
            unsaid.append(f"This page could not write its record of the start ({job['unkept']}), so it shows the "
                          "forecast as running only once the forecast writes its own records.")
        try:
            # Only an accepted start takes the model off the card; the run fetches data before it needs the card.
            freed = self.assistant.make_room() if getattr(self, "assistant", None) is not None else None
        except Exception as error:  # noqa: BLE001 - the forecast is running whatever the assistant does
            unsaid.append(_after_start(run_id, "The assistant's model could not be moved off the card", error))
        try:
            log_command(rundir, display(argv), "Start")
        except OSError as error:
            unsaid.append(_after_start(run_id, "The start could not be added to this run's commands.log", error))
        return Reply(200, {"ok": True, "run": run_id, "url": run_url(run_id), "argv": argv,
                           "command": display(argv), "job": job, "assistant": freed,
                           "message": " ".join(["Started. The forecast keeps running if you close this page.",
                                                *unsaid])})

    def stop(self, run_id: str, rundir: Path, payload: dict[str, Any], dry: bool) -> Reply:
        info = runs.status(rundir)
        if info["state"] != "running":
            raise ApiError(409, f"This run is {info['state']}, not running.", "")
        remote = read_json(rundir / runs.REMOTE, default=None)
        if isinstance(remote, dict):
            return self.stop_remote(run_id, rundir, remote, dry)
        try:
            plan = self.runner.stop(rundir, dry=dry)
        except Refused as error:
            raise ApiError(409, str(error), "") from error
        if dry:
            return Reply(200, {"ok": True, "dry_run": True, "run": run_id, **plan,
                               "message": "Nothing ran. This is what Stop does."})
        record_stop(rundir, plan)
        return Reply(200, {"ok": True, "run": run_id, **plan})

    def _launch_new(self, run_id: str, rundir: Path, argv: list[str], need_gib: float | None = None) -> Reply:
        """Launch a run whose folder this request has just written; a start that is refused removes it again.

        Left behind, the folder of a refused start read as a Ready run nobody
        asked for in My forecasts and on its event page, and it took the name:
        the event page's next press made a "-2" copy beside it and New
        forecast refused the same name as taken.  A folder a job was recorded
        in is never removed, since that run started.
        """

        try:
            return self._launch(run_id, rundir, argv, need_gib=need_gib)
        except ApiError:
            if not (rundir / runs.JOB).exists():
                shutil.rmtree(rundir, ignore_errors=True)
            raise


# ------------------------------------------------------------------ pictures

class PicturesMixin:
    def pictures(self, rundir: Path) -> dict[str, Any]:
        found = frames.index(rundir)
        found["favourites"] = frames.favourites(found["products"])
        return found

    def picture_list(self, rundir: Path, query: dict[str, list[str]]) -> dict[str, Any]:
        def one(key: str) -> str | None:
            values = query.get(key)
            return None if not values else values[0]

        placed: list[dict[str, Any]] = []
        items = frames.pictures(rundir, domain=one("domain"), product=one("product"), day=one("day"),
                                episode=one("episode"), placed=placed)
        return {"pictures": items, "count": len(items), "georefs": placed}

    def run_map(self, rundir: Path) -> dict[str, Any]:
        """Where a run sits on the Earth, from its own records only.

        The grids come from the resolved plan in the event log, and
        ``moves`` from the engine's step log: every place a nest that
        follows a storm moved to, so a frame is drawn where its grid was
        when it was written (:func:`runs.nest_moves`).  Before the plan
        exists (a run that is ready or just started) the requested
        box in region.geojson is all there is, and the map draws that.
        """

        events = rundir / runs.EVENTS
        facts = runs.event_facts(events) if events.is_file() else {}
        grid = facts.get("grid") or {}
        region = read_json(rundir / "region.geojson", default=None)
        ring = None
        if isinstance(region, dict) and region.get("type") == "Polygon":
            coordinates = region.get("coordinates") or []
            ring = coordinates[0] if coordinates else None
        info = runs.status(rundir)
        projection, domains, moves = grid.get("projection"), grid.get("domains") or [], runs.nest_moves(rundir)
        if not domains:
            # a downscale's event log carries no grid: its plan places it in its parent run's grid
            child = runs.downscale_grid(self.root, rundir)
            if child is not None:
                projection, domains, moves = child["projection"], child["domains"], child["moves"] + moves
        return {"projection": projection, "domains": domains,
                "moves": moves, "region": ring, "start_time": info.get("start_time"),
                "run_seconds": info.get("run_seconds"), "state": info.get("state")}


# ------------------------------------------------------------------ create

def _utcnow() -> datetime:
    """The clock the page's opening start is judged by (a test fixes it here)."""

    return datetime.now(timezone.utc)


def default_cycle(now: datetime | None = None) -> str:
    """The newest 6-hourly cycle that is at least six hours old, as YYYY-MM-DDTHH."""

    moment = (now or datetime.now(timezone.utc)) - timedelta(hours=6)
    return moment.replace(hour=(moment.hour // 6) * 6, minute=0, second=0,
                          microsecond=0).strftime("%Y-%m-%dT%H")


def recent_cycles(count: int = 4, now: datetime | None = None) -> list[str]:
    """The newest ``count`` 6-hourly cycles old enough to have data, newest first."""

    first = datetime.strptime(default_cycle(now), "%Y-%m-%dT%H")
    return [(first - timedelta(hours=6 * i)).strftime("%Y-%m-%dT%H") for i in range(count)]


def region_polygon(lat: float, lon: float, width_km: float, height_km: float) -> dict[str, Any]:
    """The rectangle a person asked for, as GeoJSON the wizard reads.

    The wizard projects, sizes and checks containment itself; this only
    spells the requested box, as the desktop's draft door did.
    """

    half_lat = height_km / (2 * 111.32)
    half_lon = width_km / (2 * 111.32 * math.cos(math.radians(lat)))
    if abs(lat) + half_lat >= 89.0 or half_lon >= 179.0:
        raise ApiError(400, "This box reaches a pole or spans the whole globe.", "Make it smaller.")

    def wrap(value: float) -> float:
        return (value + 180.0) % 360.0 - 180.0

    west, east = wrap(lon - half_lon), wrap(lon + half_lon)
    south, north = lat - half_lat, lat + half_lat
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return {"type": "Polygon", "coordinates": [[[round(x, 5), round(y, 5)] for x, y in ring]]}


def _number(payload: dict[str, Any], key: str, low: float, high: float, *,
            default: float | None = None, integral: bool = False) -> float | None:
    value = payload.get(key, default)
    if value is None or value == "":
        if default is None:
            return None
        value = default
    if isinstance(value, bool):
        raise ApiError(400, f"{key} must be a number.", "")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ApiError(400, f"{key} must be a number.", "") from error
    if not math.isfinite(number) or not low <= number <= high:
        raise ApiError(400, f"{key} must be between {low:g} and {high:g}.", "")
    if integral:
        if not number.is_integer():
            raise ApiError(400, f"{key} must be a whole number.", "")
        return int(number)
    return number


_CYCLE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}$")


def ladder_names() -> list[str]:
    """The wizard's nest ladders, from its own table, plus auto."""

    from gpuwm.domain_wizard import LADDER_RATIOS

    return [*LADDER_RATIOS, "auto"]


def _check_render_selection(products: str | None, section: str | None) -> None:
    """Check before creating a draft whose requested pictures cannot be drawn."""
    from gpuwm.go_cli import GoRefusal, admit_render_products
    from gpuwm.rustwx import is_section_line

    # A GUI run can draw on another machine; a file on this computer is
    # not a section file on that machine. Coordinates travel with the draft.
    if section and not is_section_line(section):
        raise ApiError(400, "The section line must be lat,lon,lat,lon. A section file on this computer "
                       "is not available to a renderer on another machine, so it could draw nothing.",
                       "Enter the latitude and longitude of each end of the section.")
    try:
        admit_render_products(products, section=section)
    except GoRefusal as error:
        raise ApiError(400, str(error), "Set a valid section line, or remove the section product.") from error


class CreateMixin:
    root: Path
    runner: Runner

    def draft(self, payload: dict[str, Any], *, need_name: bool, settle: bool = False,
              queue: bool = False) -> dict[str, Any]:
        """A checked draft: the fields a plan's intent is built from.

        ``settle`` (Start and Queue it, and their dry runs, which answer what they would do) waits, at most
        :data:`.availability.SETTLE_S` seconds, for the source's check of a start from the last day when that
        check is still out, and then takes the start by the rule in :mod:`.availability`: Start takes a start
        that is confirmed, or past its source's usual publication delay and not found missing; ``queue`` (Queue
        it) also takes a start from the last day that is not confirmed yet, and the draft's ``waits_for_data``
        says why the queue holds it until Start would take it.  A start Start would not take was refused only
        when its download began.  A fit never waits and never refuses on it.
        """

        name = payload.get("name") or ""
        if need_name:
            require_name(str(name), "run name")
        source = str(payload.get("source") or "").strip()
        if not re.match(r"^[a-z0-9][a-z0-9._-]{0,63}$", source):
            raise ApiError(400, "Pick a data source.", "")
        offered = {row["id"]: row for row in self._offered()["sources"]}
        if source not in offered:
            raise ApiError(400, f"New forecast cannot start from {source}.",
                           "Pick one of the sources the page lists.")
        cycle = str(payload.get("cycle") or "").strip() or default_cycle()
        if not _CYCLE.match(cycle):
            raise ApiError(400, "The start time must look like 2026-09-24T12.", "")
        card = str(payload.get("card") or "").strip().lower()
        if card not in CARDS:
            raise ApiError(400, "Pick the card memory the run is sized for.", f"One of {', '.join(CARDS)}.")
        draft = {
            "name": str(name),
            "source": source,
            "route": offered[source]["route"],
            "cycle": cycle,
            "lat": _number(payload, "lat", -85.0, 85.0),
            "lon": _number(payload, "lon", -180.0, 180.0),
            "width_km": _number(payload, "width_km", 50.0, 8000.0, default=600.0),
            "height_km": _number(payload, "height_km", 50.0, 8000.0, default=600.0),
            "hours": _number(payload, "hours", 1, 384, default=6, integral=True),
            "dx_km": _number(payload, "dx_km", DX_MIN_KM, 50.0),
            # Vertical levels: None is the engine's own default ladder; a number is handed to the wizard's --nz,
            # which resamples that ladder and prices the fit at that count.
            "nz": _number(payload, "nz", 1, 1000, integral=True),
            "ladder": str(payload.get("ladder") or "").strip() or None,
            # How the run steps (`gpuwm domain --clock`); None is the engine's own choice (auto).
            "clock": str(payload.get("clock") or "").strip() or None,
            "start_hour": _number(payload, "start_hour", 0, 384, default=0, integral=True),
            "card": card,
            "profile": str(payload.get("profile") or "").strip() or None,
            "preset": str(payload.get("preset") or "").strip() or None,
            "products": str(payload.get("products") or "").strip() or None,
            "render_section": str(payload.get("render_section") or "").strip() or None,
            # The nests of an event's best run, carried through Customise as the engine's own intent keys.
            "chain": _nest_list(payload.get("chain"), "chain", r"^\d+(,\d+)*$"),
            "buffer_km": _nest_list(payload.get("buffer_km"), "buffer_km", r"^\d+(\.\d+)?(,\d+(\.\d+)?)*$"),
            "era5_provider": _word(payload.get("era5_provider"), "era5_provider"),
            # An event's storm-following nest, carried through Customise: the cyclone setup the event page's
            # button runs for that card (bound below from the storm wiki's own layouts, never from the page),
            # run with this draft's source, start, length and card.
            "following": payload.get("following") is True,
            "cyclone_setup": None,
        }
        if draft["preset"]:
            # A preset is a suite and a grid spacing.  An explicit suite or
            # spacing in the same request wins: the preset fills blanks.
            row = self.preset_row(draft["preset"])
            draft["profile"] = draft["profile"] or row.get("suite")
            if draft["dx_km"] is None and row.get("dx_km") is not None:
                if float(row["dx_km"]) < DX_MIN_KM:
                    raise ApiError(400, f"The {row['id']} preset is for {float(row['dx_km']):g} km grids and the "
                                   f"outer grid here is {DX_MIN_KM:g} km or coarser.",
                                   f"Set a spacing of {DX_MIN_KM:g} km or more to run its suite anyway.")
                draft["dx_km"] = float(row["dx_km"])
        if draft["lat"] is None or draft["lon"] is None:
            raise ApiError(400, "Set the centre of the forecast box.", "Click the map or type a latitude and longitude.")
        if payload.get("products_named") is True and draft["products"] is None:
            raise ApiError(400, "Named pictures or a cross-section is chosen and no picture is named. An empty list "
                           "would draw the standard set while the page still shows the named choice.",
                           "Name at least one picture, such as xsec:wa, or pick The standard set.")
        _check_render_selection(draft["products"], draft["render_section"])
        # A start the source does not have is refused here, in words, rather than well into the run when the
        # download finds nothing: the engine's own archive bounds, the same ones the date picker shows.
        # The same clock the page's own rows and opening start are judged by.
        moment = _utcnow()
        answer = (self.settled_availability(source, cycle, draft["hours"], now=moment) if settle
                  else self.availability_of(source, cycle, draft["hours"], now=moment))
        if answer.get("starts") == "no":
            # A run whose files thin out before its end names the lengths that download; every other no is the date's.
            raise ApiError(422, answer["why"], answer.get("fix")
                           or "Pick another date, or a source the date picker marks as having it.")
        draft["waits_for_data"] = None
        if settle and answer.get("starts") == "queue":
            if not queue:
                # Not confirmed and not past its usual publication delay: a forecast started on it failed when its
                # download began.  Queue it takes it and holds it until a check confirms it.
                opening = answer.get("opening")
                other = (f" Or pick {opening.replace('T', ' ')}:00 UTC, the newest start that can start now."
                         if opening and opening != cycle else "")
                raise ApiError(422, answer["why"],
                               f"Queue it instead: it waits for this start and begins once a check confirms it.{other}")
            draft["waits_for_data"] = answer["why"]
        if draft["ladder"] is not None and draft["chain"] is not None:
            # `gpuwm run-plan --resolve` refuses a plan that holds both, after the page has priced it.
            raise ApiError(400, "A nest ladder and a nest chain cannot both be set: the ladder names its own nests.",
                           "Keep the ladder and clear the chain, or set the ladder to none.")
        draft["event_keys"] = self._event_keys(payload, draft)
        if draft["clock"] is not None and draft["clock"] not in CLOCK_WORDS:
            raise ApiError(400, "The time step must be adaptive or fixed.",
                           "Pick one of the listed ones, or leave it to the engine.")
        if draft["following"]:
            # The cyclone setup sizes its own two grids, levels and physics; any of these beside it would be
            # written into the plan and never run.
            clash = [words for key, words in (("ladder", "a nest ladder"), ("chain", "a nest chain"),
                                              ("nz", "a level count"), ("profile", "a physics set"),
                                              ("clock", "a time step choice"))
                     if draft.get(key) is not None]
            if payload.get("physics_choices"):
                clash.append("picked physics schemes")
            if clash:
                raise ApiError(400, "A storm-following nest sets its own grids, levels, physics and time step, so "
                                    f"{' and '.join(clash)} would not be used.",
                               "Clear it, or pick another grid so the event's following nest is left out.")
            self._bind_following(payload, draft)
        if draft["ladder"] is not None:
            ladders = ladder_names()
            if draft["ladder"] not in ladders:
                raise ApiError(400, "The nest ladder must be one of the listed ones.", f"One of {', '.join(ladders)}.")
            if draft["dx_km"] is not None:
                raise ApiError(400, "A nest ladder and a grid spacing cannot both be set; the ladder fixes the spacing.",
                               "Clear one of them.")
        return draft

    def _event_keys(self, payload: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
        """The rest of an event's best run, for a draft that keeps the event's grids.

        The page names the layout (its event and the card size its
        Customise opened with) while the draft runs the event's own grids;
        the values are read here from the storm wiki's store, never from
        the page.  They are the keys of the intent the event page's button
        runs that New forecast has no control for (:data:`DRAFT_INTENT_KEYS`),
        such as how often each grid writes its history, which keeps the run
        inside the disk the event page promised, and a cyclone's sea-surface
        flux option.  A storm-following layout carries its own through its
        cyclone setup, and a draft that names no event carries none.
        """

        ident = str(payload.get("event") or "").strip()
        if not ident or draft["following"]:
            return {}
        size = _number(payload, "recipe_card_gb", 1, 1024)
        recipe = self.wiki.store.data()["recipes"].get(ident)
        row = recipe_row(recipe, size)
        if row is None:
            # Started without them, the run would write its history at the engine's own intervals, more than the
            # disk the event page promised for this layout, and lose the rest of what the button runs.
            raise ApiError(422, "This event has no best run for that card, so the settings of its best run cannot "
                                "be carried into this forecast.",
                           "Open the event's Customise again, or pick another grid so the event's layout is left out.")
        intent = row.get("intent")
        if not isinstance(intent, dict):
            return {}
        return {key: value for key, value in intent.items() if key not in DRAFT_INTENT_KEYS}

    def _bind_following(self, payload: dict[str, Any], draft: dict[str, Any]) -> None:
        """Bind a storm-following draft to the event's own layout for the card it came from.

        The layout is read from the storm wiki's store by the event and the
        card size its Customise opened with: the page carries only which
        layout it means.  Its cyclone setup is the one the event page's
        button runs (history intervals, surface flux and nest budget
        included); the start runs it with this draft's source, start, length
        and card, so another date or card is set up afresh.  The outer grid
        and box are the event's own: the setup places both grids around the
        storm, so a box or grid of the page's own would be drawn and never run.
        """

        ident = str(payload.get("event") or "").strip()
        if not ident:
            raise ApiError(400, "A storm-following nest needs the event it comes from.",
                           "Open the event's Customise again.")
        size = _number(payload, "recipe_card_gb", 1, 1024)
        row = recipe_row(self.wiki.store.data()["recipes"].get(ident), size)
        recipe = (row or {}).get("recipe") or {}
        args = (row or {}).get("args")
        if not row or row.get("door") != "cyclone-setup" or not recipe.get("following") or not isinstance(args, dict):
            raise ApiError(422, "This event has no storm-following layout for that card.",
                           "Open the event's Customise again, or pick another grid so the following nest is left "
                           "out.")
        for key in ("lat", "lon", "width_km", "height_km", "dx_km"):
            asked, own = draft.get(key), recipe.get(key)
            if asked is None or own is None or abs(float(asked) - float(own)) > 0.011:
                raise ApiError(422, "A storm-following nest keeps the event's own box and grid.",
                               "Put the box back, or pick another grid so the following nest is left out.")
        draft["cyclone_setup"] = dict(args)

    def _compile_following(self, draft: dict[str, Any]) -> None:
        """Run a storm-following draft's cyclone setup once, in a scratch folder, and keep every file it writes.

        The configuration and the companion files it names beside itself
        (the Vtable and the WPS namelist) are held as text on the draft, so
        they can be written into this computer's run folder or sent with the
        plan to a Machines node.  Nothing is written into a run folder here.
        """

        if not draft.get("following") or "_following_files" in draft:
            return
        import tempfile

        drafts = self.root / SERVER_DIR / "drafts"
        drafts.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="following-", dir=drafts) as temporary:
            folder = Path(temporary)
            argv = self.cyclone_argv(draft, folder / FOLLOWING_CONFIG)
            try:
                result = self.runner.query(argv, cwd=folder)
            except Refused as error:
                print(f"gpuwm gui: cyclone-setup refused: {display(argv)}\n{error}", file=sys.stderr, flush=True)
                raise ApiError(422, "The engine could not set up the storm-following nest. The full reason is in "
                                    "the page server's log.",
                               "Pick another start, length or card, or another grid so the following nest is left "
                               "out.") from error
            fitting = (result or {}).get("fitting") if isinstance(result, dict) else None
            if (fitting or {}).get("review_required") or not (folder / FOLLOWING_CONFIG).is_file():
                raise ApiError(422, f"The storm-following nest does not fit the {draft['card'].replace('gb', ' GB')} "
                                    "card at its full size.",
                               "Pick a bigger card, or another grid so the following nest is left out.")
            # Every text input the setup writes beside its configuration; its receipt names this scratch folder
            # and is not an input of the run.
            draft["_following_files"] = {
                path.name: path.read_text(encoding="utf-8") for path in sorted(folder.iterdir())
                if path.is_file() and not path.name.endswith(".cyclone.json")}

    @staticmethod
    def _publish_following(draft: dict[str, Any], rundir: Path) -> list[str]:
        """Write a compiled storm-following draft's files into its run folder; returns their names."""

        names = []
        for file_name, text in (draft.get("_following_files") or {}).items():
            target = rundir / file_name
            tmp = target.with_name(f".{file_name}.{os.getpid()}.tmp")
            with tmp.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
            os.replace(tmp, target)
            names.append(file_name)
        return names

    @staticmethod
    def cyclone_argv(draft: dict[str, Any], out: Path | None) -> list[str]:
        """The cyclone setup for a draft that keeps an event's storm-following nest.

        The event's own options (its advisory position, sea-surface drag and the rest) with this draft's source,
        start, length, start hour and card: the same door the event page's button runs, sized for the card the
        person picked.
        """

        layout = draft["cyclone_setup"] or {}
        options = {key: value for key, value in layout.items()
                   if key not in ("source", "cycle", "hours", "card", "nest_budget_gib", "start_hour")}
        options.update(source=draft["source"], cycle=draft["cycle"], hours=int(draft["hours"]), card=draft["card"])
        if "nest_budget_gib" in layout:
            # A layout that grows its nest grows it to the card it runs on; a fixed-size one keeps its size.
            options["nest_budget_gib"] = int(draft["card"].removesuffix("gb"))
        if draft.get("start_hour"):
            options["start_hour"] = int(draft["start_hour"])
        # --flag=value, not two words: a southern storm's "-29.50,-41.00" read as a flag of its own.
        argv = engine_argv("cyclone-setup", *[f"--{key.replace('_', '-')}={value}" for key, value in options.items()])
        if out is not None:
            argv += ["--out", str(out)]
        return [*argv, "--json"]

    @staticmethod
    def plan_document(draft: dict[str, Any], rundir: Path) -> dict[str, Any]:
        section = draft.get("render_section")
        _check_render_selection(draft["products"], section)
        options = {"render_products": draft["products"]} if draft["products"] else {}
        if section:
            options["render_section"] = section
        if draft.get("following"):
            # The cyclone setup writes the configuration; the plan runs it as it is.
            # A configuration file's run draws nothing unless its plan says what to draw, so the page's
            # standard set is named here as an intent plan gets it by default.
            from gpuwm.first_products import DEFAULT_RENDER_PRODUCTS

            return {"schema": PLAN_SCHEMA, "name": draft["name"] or "draft", "route": "experiment",
                    "config": {"path": str(rundir / FOLLOWING_CONFIG)}, "output_root": str(rundir),
                    "run_options": {**options, "render_products": draft["products"] or DEFAULT_RENDER_PRODUCTS}}
        intent: dict[str, Any] = {
            "polygon": str(rundir / "region.geojson"),
            "source": draft["source"],
            "cycle": draft["cycle"],
            "hours": draft["hours"],
            "card": draft["card"],
        }
        if draft["dx_km"] is not None:
            intent["root_dx_km"] = draft["dx_km"]
        if draft.get("nz") is not None:
            intent["nz"] = draft["nz"]
        if draft.get("ladder"):
            intent["ladder"] = draft["ladder"]
        if draft.get("clock"):
            intent["clock"] = draft["clock"]
        if draft.get("start_hour"):
            intent["forecast_start_hour"] = draft["start_hour"]
        if draft["profile"]:
            intent["physics_profile"] = draft["profile"]
        if draft.get("cumulus"):
            # Set only from the physics check's plan_intent (composed_physics): the root's cumulus is the one the
            # Physics step showed, not the named suite's.
            intent["cumulus"] = draft["cumulus"]
        if draft.get("physics_choices"):
            # Schemes no set of this source matches (composed_physics): `gpuwm domain --physics-choices` writes them
            # into the experiment, and the fit prices them.
            intent["physics_choices"] = dict(draft["physics_choices"])
        for key in ("chain", "buffer_km", "era5_provider"):
            if draft.get(key):
                intent[key] = draft[key]
        for key, value in (draft.get("event_keys") or {}).items():
            # The rest of the event's best run (CreateMixin._event_keys): keys this draft has no control for.
            intent.setdefault(key, value)
        plan: dict[str, Any] = {
            "schema": PLAN_SCHEMA,
            "name": draft["name"] or "draft",
            "route": draft.get("route") or "prepared",
            "config": {"intent": intent},
            "output_root": str(rundir),
        }
        if options:
            plan["run_options"] = options
        return plan

    def _write_plan(self, draft: dict[str, Any], rundir: Path) -> Path:
        rundir.mkdir(parents=True, exist_ok=True)
        write_json(rundir / "region.geojson",
                   region_polygon(draft["lat"], draft["lon"], draft["width_km"], draft["height_km"]))
        write_json(rundir / runs.PLAN, self.plan_document(draft, rundir))
        return rundir / runs.PLAN

    def night_refusal(self, draft: dict[str, Any]) -> None:
        """Refuse a day-only suite over a window with local night, before the run.

        The prepared route meets the engine's nocturnal-radiation door at
        load, after Create has launched the run and the user has walked
        away; the physics check asks the same door with the draft's
        cycle, hours and centre, so the refusal comes back on this click.
        Only a suite the catalog marks day-only is asked about.
        """

        source = draft["source"]
        catalog = self._cached("physics:" + source,
                               engine_argv("physics-catalog", "--json", "--source", source), CACHE_S)
        suite = draft["profile"] or catalog.get("default_suite")
        row = next((r for r in catalog.get("suites") or [] if r.get("id") == suite), None)
        if not row or not row.get("day_only"):
            return
        request = {"source": source, "cycle": draft["cycle"], "hours": draft["hours"],
                   "lat": draft["lat"], "lon": draft["lon"]}
        if draft["profile"]:
            request["suite"] = draft["profile"]
        if draft["dx_km"] is not None:
            request["dx_km"] = draft["dx_km"]
        argv = engine_argv("physics-catalog", "--json", "--check",
                           json.dumps(request, separators=(",", ":"), sort_keys=True))
        try:
            verdict = self.runner.query(argv)
        except Refused:
            # The check could not read the draft; the run's own route
            # answers it with its own words.
            return
        refusal = verdict.get("refusal") or {}
        if not verdict.get("valid") and refusal.get("door") == "nocturnal-radiation":
            raise ApiError(422, str(verdict.get("words") or refusal.get("message")),
                           "This suite runs by day only. Pick a start time and length that stay in daylight "
                           "at this place, or a suite with both radiation streams.",
                           argv=argv, command=display(argv))

    def fit(self, payload: dict[str, Any], dry: bool) -> Reply:
        draft = self.draft(payload, need_name=False)
        if draft["following"]:
            return self._following_fit(draft, dry)
        drafts = self.root / SERVER_DIR / "drafts"
        folder = drafts / uuid.uuid4().hex[:12]
        argv = engine_argv("run-plan", str(folder / runs.PLAN), "--resolve")
        if dry:
            return _dry(argv)
        region_polygon(draft["lat"], draft["lon"], draft["width_km"], draft["height_km"])
        # A composed choice is sized as it will run: the suite it names and the cumulus the check left on or off,
        # or the picked schemes themselves when no set of this source matches them.  A mix the check refuses
        # cannot be started, and is sized as the form's set; the Physics step says why.
        try:
            self.composed_physics(payload, draft)
        except ApiError as error:
            if error.status != 422:
                raise
        self.night_refusal(draft)
        try:
            self._write_plan(draft, folder)
            try:
                resolved = self.runner.query(argv, cwd=folder)
            except Refused as error:
                # A draft too big for the card is refused with a document that carries its figures.
                memory = memory_fit(error.document)
                message, fix = plain_refusal(str(error), draft, memory,
                                             (self.offered_sources() or {}).get(draft["source"], draft["source"]))
                print(f"gpuwm gui: the fit was refused: {display(argv)}\n{error}", file=sys.stderr, flush=True)
                refused = ApiError(422, message, fix, argv=argv, command=display(argv), memory=memory)
                # The engine's own words stay in this process for callers that read its numbers (the
                # assistant's planner); the page gets the plain sentence above.
                refused.engine_text = str(error)
                raise refused from error
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        fit = describe_fit(resolved, draft)
        fit["memory"] = memory_fit(resolved)
        warning = self.assistant.card_warning() if getattr(self, "assistant", None) is not None else None
        if warning:
            fit["warning"] = warning
            fit["words"] = f"{fit['words']} {warning}".strip()
        return Reply(200, {"ok": True, "argv": argv, "command": display(argv), "fit": fit})

    def _following_fit(self, draft: dict[str, Any], dry: bool) -> Reply:
        """A storm-following draft priced by the cyclone setup that will write it, as the start runs it."""

        argv = self.cyclone_argv(draft, None)
        if dry:
            return _dry(argv)
        region_polygon(draft["lat"], draft["lon"], draft["width_km"], draft["height_km"])
        self.night_refusal(draft)
        try:
            result = self.runner.query(argv)
        except Refused as error:
            print(f"gpuwm gui: cyclone-setup refused: {display(argv)}\n{error}", file=sys.stderr, flush=True)
            refused = ApiError(422, "The engine could not set up the storm-following nest for this start. The full "
                                    "reason is in the page server's log.",
                               "Pick another start, length or card, or another grid so the following nest is left "
                               "out.", argv=argv, command=display(argv))
            refused.engine_text = str(error)
            raise refused from error
        fit = describe_following(result, draft)
        if (result.get("fitting") or {}).get("review_required"):
            raise ApiError(422, f"The storm-following nest does not fit the {draft['card'].replace('gb', ' GB')} "
                                "card at its full size.",
                           "Pick a bigger card, or another grid so the following nest is left out.",
                           argv=argv, command=display(argv), memory=fit["memory"])
        warning = self.assistant.card_warning() if getattr(self, "assistant", None) is not None else None
        if warning:
            fit["warning"] = warning
            fit["words"] = f"{fit['words']} {warning}".strip()
        return Reply(200, {"ok": True, "argv": argv, "command": display(argv), "fit": fit})

    def _prepare_following(self, draft: dict[str, Any], rundir: Path) -> list[str]:
        """Write a compiled storm-following draft's configuration and companions into its claimed run folder."""

        argv = self.cyclone_argv(draft, rundir / FOLLOWING_CONFIG)
        self._publish_following(draft, rundir)
        log_command(rundir, display(argv), "Set up the storm-following nest")
        return argv

    def _following_disk(self, rundir: Path, queued: bool) -> float | None:
        """What a storm-following run will write, in GiB, from the engine's own projection of its written plan.

        The cyclone setup sizes the nest and sets how often it writes, so the number is known only once the setup
        has written the configuration.  A start the disk cannot hold is refused here, as the event page's button
        refuses it: such a run stops partway when the disk fills, with nothing usable, and can stop other work on
        that disk.  A queued run keeps the number instead, and the queue holds it until the disk has the room.
        """

        argv = engine_argv("run-plan", str(rundir / runs.PLAN), "--resolve")
        try:
            resolved = self.runner.query(argv, cwd=rundir)
        except Refused as error:
            print(f"gpuwm gui: the storm-following plan was refused: {display(argv)}\n{error}", file=sys.stderr,
                  flush=True)
            shutil.rmtree(rundir, ignore_errors=True)
            raise ApiError(422, "The engine refused the storm-following run's plan. The full reason is in the page "
                                "server's log.",
                           "Pick another start, length or card, or another grid so the following nest is left "
                           "out.") from error
        total = ((resolved or {}).get("disk") or {}).get("total_bytes") if isinstance(resolved, dict) else None
        if not total:
            return None
        need = round(float(total) / 2**30, 1)
        free = disk_free_gib(self.root)
        if not queued and free is not None and need > free:
            shutil.rmtree(rundir, ignore_errors=True)
            raise ApiError(507, f"This run writes about {need:.0f} GiB and the disk that holds your runs has "
                                f"{free:.0f} GiB free, so it would stop partway when the disk fills.",
                           "Free some disk, or pick a shorter length, or another grid so the following nest is "
                           "left out.")
        return need

    def create_start(self, payload: dict[str, Any], dry: bool) -> Reply:
        machine = str(payload.get("machine") or LOCAL)
        queued = payload.pop("queue", False)
        if not isinstance(queued, bool):
            raise ApiError(400, "queue must be true or false.", "")
        need_gib = _number(payload, "need_gib", 0, 4096) if payload.get("need_gib") is not None else None
        payload.pop("need_gib", None)
        if machine != LOCAL:
            if queued:
                return self.queue_remote(machine, payload, need_gib, dry)
            return self.create_start_remote(machine, payload, dry)
        draft = self.draft(payload, need_name=True, settle=True, queue=queued)
        rundir = self.root / draft["name"]
        argv = engine_argv("run-plan", str(rundir / runs.PLAN))
        if rundir.exists():
            raise ApiError(409, f"A run called {draft['name']} already exists.",
                           "Give this forecast another name.")
        refuse_without_geography(self.runner)
        region_polygon(draft["lat"], draft["lon"], draft["width_km"], draft["height_km"])
        physics = self.composed_physics(payload, draft)
        if dry:
            extra = {"prepare": self.cyclone_argv(draft, rundir / FOLLOWING_CONFIG)} if draft["following"] else {}
            return _dry(argv, run=draft["name"], plan=self.plan_document(draft, rundir), physics=physics,
                        queued=queued, waits_for_data=draft["waits_for_data"], **extra)
        self.night_refusal(draft)
        self._compile_following(draft)
        # The folder is claimed before anything is written into it: a second start under the same name is
        # refused here, and nothing this start writes can land in another start's folder.
        reserve_run(rundir)
        # Nothing ran until the launch or the queue has it: the folder this start claimed goes on any failure
        # before that, so the same name can be queued or started again.
        with releasing_on_failure(rundir):
            if draft["following"]:
                self._prepare_following(draft, rundir)
            self._write_plan(draft, rundir)
            if physics is not None:
                write_json(rundir / PHYSICS_CHOICE, physics)
            disk_gib = self._following_disk(rundir, queued) if draft["following"] else None
        if queued:
            marker = {"machine": LOCAL, "need_gib": need_gib, "card": draft["card"], "argv": argv,
                      "data": _data_of(draft)}
            if disk_gib is not None:
                marker["disk_gib"] = disk_gib
            return self._enqueue(draft["name"], rundir, marker)
        return self._launch_new(draft["name"], rundir, argv, need_gib=need_gib)

    def _enqueue(self, run_id: str, rundir: Path, marker: dict[str, Any]) -> Reply:
        """Put a folder this request claimed and wrote in the queue.

        The queue may start it the moment it is in line, before this answers, so its place is the one the queue
        gave it as it joined: read back from the line afterwards, a forecast already started was not there and
        the answer was an error for a forecast that had been queued and started.  Until it is in line nothing
        started, and a failure releases the folder as a start's does.
        """

        argv = marker.pop("argv", None) or []
        with releasing_on_failure(rundir):
            # Written first, so a forecast the queue starts at once logs its queuing before its start.
            log_command(rundir, "# queued: starts by itself when its data is published and the card is free",
                        "Queue")
            place = self.queue.add(run_id, rundir, marker)
        self.queue.note_data(run_id)
        waits = (marker.get("data") or {}).get("why")
        return Reply(200, {"ok": True, "queued": True, "run": run_id, "url": run_url(run_id), "place": place,
                           "argv": argv, "command": display(argv) if argv else "", "waits_for_data": waits,
                           "message": ("Queued. It waits for its start to be published, then starts by itself when "
                                       "the card is free." if waits else
                                       "Queued. It starts by itself when the card is free.")})

    def queue_remote(self, machine: str, payload: dict[str, Any], need_gib: float | None, dry: bool) -> Reply:
        """Queue a forecast for a Machines node: every check a start makes now, then a folder that waits."""

        checked = self.create_start_remote(machine, dict(payload), True, queue=True)
        if dry:
            checked.body["queued"] = True
            return checked
        run_id = str(checked.body["run"])
        rundir = self.root / run_id
        reserve_run(rundir)
        return self._enqueue(run_id, rundir, {"machine": machine, "need_gib": need_gib, "card": payload.get("card"),
                                              "payload": {**payload, "name": run_id}, "title": run_id,
                                              "data": checked.body.get("data")})

    def queue_listing(self, query: dict[str, list[str]]) -> dict[str, Any]:
        machine = ((query.get("machine") or [""])[0] or LOCAL).strip() or LOCAL
        raw = ((query.get("need_gib") or [""])[0] or "").strip()
        try:
            need = float(raw) if raw else None
        except ValueError as error:
            raise ApiError(400, "need_gib must be a number.", "") from error
        data = None
        if (query.get("source") or [""])[0] and (query.get("cycle") or [""])[0]:
            cycle = (query.get("cycle") or [""])[0].strip()
            if not _CYCLE.match(cycle):
                raise ApiError(400, "The start time must look like 2026-09-24T12.", "")
            try:
                hours = float((query.get("hours") or ["6"])[0] or 6)
            except ValueError as error:
                raise ApiError(400, "hours must be a number.", "") from error
            data = {"source": (query.get("source") or [""])[0].strip(), "cycle": cycle, "hours": hours}
        listing = self.queue.listing(machine, need, data=data)
        listing["host"] = socket.gethostname()
        return listing

    def queue_action(self, run_id: str, action: str, dry: bool) -> Reply:
        if action not in ("up", "down", "remove"):
            raise ApiError(404, "No such action.")
        if dry:
            return Reply(200, {"ok": True, "dry_run": True, "run": run_id,
                               "message": f"Nothing changed. This would {action} {run_id} in the queue."})
        try:
            if action == "remove":
                try:
                    kept = self.queue.remove(run_id)
                except OSError as error:
                    print(f"gpuwm gui: {run_id} could not be taken out of the queue: {error!r}", file=sys.stderr,
                          flush=True)
                    raise ApiError(500, f"{run_id} could not be removed ({error.strerror or type(error).__name__}), "
                                        "so it is still queued and starts in its turn.",
                                   "Make sure its folder in the forecasts folder can be changed, then press Remove "
                                   "again.") from error
                if kept is not None:
                    # Said, never passed over: the folder stays in the forecasts folder, listed in My forecasts.
                    return Reply(200, {"ok": True, "run": run_id, "order": self.queue.order(), "kept": True,
                                       "message": scrub(f"Removed {run_id} from the queue, and it will not start. "
                                                        f"Its folder could not be deleted ({kept}): delete the "
                                                        f"{run_id} folder in your forecasts folder yourself.")})
                return Reply(200, {"ok": True, "run": run_id, "order": self.queue.order(),
                                   "message": f"Removed {run_id} from the queue."})
            order = self.queue.move(run_id, -1 if action == "up" else 1)
        except KeyError as error:
            raise ApiError(404, f"{run_id} is not in the queue.", "Reload My forecasts.") from error
        return Reply(200, {"ok": True, "run": run_id, "order": order, "message": "Moved."})

    def composed_physics(self, payload: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any] | None:
        """The physics composer's choice, checked again by the engine, as the plan will run it.

        The page sends the schemes chosen per family (``physics_choices``) and the suite its own check named
        (``profile``).  The engine checks the choices once more at this draft's spacing, source and window: a
        mix that does not run is refused in the check's own words.  One that runs and makes a set this source
        offers lands in the plan as that set, the way a preset does.  Any other mix that runs lands in the plan
        as the choices themselves (``config.intent.physics_choices``), which ``gpuwm domain --physics-choices``
        writes into the experiment the way ``gpuwm physics-catalog --into`` writes a mix.  ``None`` when
        nothing was composed.
        """

        choices = payload.get("physics_choices")
        if choices in (None, {}):
            return None
        if not isinstance(choices, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                    for k, v in choices.items()):
            raise ApiError(400, "physics_choices must map each family to one scheme.", "")
        request = {"choices": choices, "source": draft["source"], "cycle": draft["cycle"], "hours": draft["hours"],
                   "lat": draft["lat"], "lon": draft["lon"]}
        if draft["dx_km"] is not None:
            request["dx_km"] = draft["dx_km"]
        if draft.get("nz") is not None:
            request["nz"] = draft["nz"]
        argv = engine_argv("physics-catalog", "--json", "--check",
                           json.dumps(request, separators=(",", ":"), sort_keys=True))
        try:
            verdict = self.runner.query(argv)
        except Refused as error:
            raise ApiError(400, str(error), "Pick the schemes again on the Physics step.",
                           argv=argv, command=display(argv)) from error
        if not verdict.get("valid"):
            raise ApiError(422, str(verdict.get("words") or "This physics does not run."),
                           "Pick another scheme on the Physics step; it lists the nearest ones that run.",
                           argv=argv, command=display(argv))
        suite = verdict.get("named_suite")
        if draft["profile"] and draft["profile"] != suite:
            raise ApiError(409, f"The physics set on the form ({draft['profile']}) is not the one the chosen schemes "
                                f"make ({suite or 'no named set'}).", "Open the Physics step again so the two agree.")
        offered = [row.get("id") for row in (next((r for r in self._offered()["sources"] if r["id"] == draft["source"]),
                                                   {}) or {}).get("profiles") or []]
        if suite and suite in offered:
            # The plan carries what the check says runs, from the check itself: the suite it names and, when no
            # cumulus scheme was picked, cumulus left to the grid, so a 3 km root runs none, as the step said.
            draft["profile"] = suite
            draft["cumulus"] = (verdict.get("plan_intent") or {}).get("cumulus")
            draft["physics_choices"] = None
        else:
            # No set of this source is these schemes, so the plan carries the schemes: the wizard writes them over
            # the source's default set, with the cumulus the check decided at this spacing.
            suite = None
            draft["profile"] = draft["cumulus"] = None
            draft["physics_choices"] = dict(choices)
        return {"schema": "gpuwm.gui-physics-choice.v1", "suite": suite,
                "label": verdict.get("named_suite_label") if suite else None,
                "base_suite": verdict.get("base_suite"), "cumulus": draft["cumulus"], "choices": choices,
                "resolved": verdict.get("resolved") or {}, "words": verdict.get("words"),
                "cost": (verdict.get("cost") or {}).get("words"), "command": display(argv)}


#: The intent keys a New forecast draft writes itself (:meth:`CreateMixin.plan_document`), each from a control on
#: the page.  An event's best run carries every other key of its intent into a draft that keeps its grids.
#: The time step choices New forecast offers, as `gpuwm domain --clock` spells them (the engine's own "auto" is
#: the draft leaving it out).
CLOCK_WORDS = ("adaptive", "fixed", "auto")

DRAFT_INTENT_KEYS = frozenset({"polygon", "source", "cycle", "hours", "card", "root_dx_km", "nz", "ladder", "clock",
                               "forecast_start_hour", "physics_profile", "cumulus", "physics_choices", "chain",
                               "buffer_km", "era5_provider"})

#: Where a storm-following draft's cyclone setup writes its configuration, inside the run folder.
FOLLOWING_CONFIG = "cyclone.toml"
#: The lock a run folder's follower is started under, beside its ``gui-follow.json``.
FOLLOW_LOCK = "gui-follow.lock"


def _nest_list(value: Any, key: str, pattern: str) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).replace(" ", "")
    if not re.match(pattern, text):
        raise ApiError(400, f"{key} must be a list of numbers separated by commas.", "")
    return text


def _word(value: Any, key: str) -> str | None:
    if value in (None, ""):
        return None
    if not re.match(r"^[a-z0-9_-]{1,32}$", str(value)):
        raise ApiError(400, f"{key} must be a plain word.", "")
    return str(value)


def run_name(title: str, start: str, card_gb: Any, taken: Any) -> str:
    """A run folder name a person can read: the event's title and date, then the card size.

    ``F2 tornado, Salt Lake County, Utah, 11 August 1999`` starting at
    ``1999-08-11T12:00Z`` for a 16 GB card becomes
    ``f2-tornado-salt-lake-county-utah-1999-08-11-16gb``.  A date the
    title already spells is not repeated in words, a name already taken
    gets ``-2``, ``-3`` and so on, and the whole name keeps within the 64
    characters a run name may have.
    """

    words = re.findall(r"[a-z0-9]+", str(title or "").lower())
    months = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
              "november", "december"}
    words = [w for w in words if w not in months and not (w.isdigit() and len(w) in (1, 2, 4))] or ["event"]
    size = f"{int(float(card_gb))}gb" if card_gb not in (None, "") else ""
    tail = "-".join(part for part in (str(start or "")[:10], size) if part)
    room = 64 - len(tail) - 4
    head = words[0][:room]
    for word in words[1:]:
        if len(head) + 1 + len(word) > room:
            break
        head = f"{head}-{word}"
    name = f"{head}-{tail}" if tail else head
    base, n = name, 2
    while taken(name):
        name = f"{base}-{n}"
        n += 1
    return name


_GIB = float(1 << 30)


def memory_fit(document: dict[str, Any] | None) -> dict[str, Any] | None:
    """The card memory the engine priced a draft at, from the ``memory`` record of its answer.

    ``gpuwm run-plan --resolve`` answers a fit with the wizard's record: the binding phase's
    ``peak_envelope_bytes`` beside the ``budget_bytes`` the fit was held to.  It refuses a draft too big for
    its card with a memory refusal document that carries the same two fields, whether or not the source's
    preprocessing is priced.  Every source fills them, including one whose inputs are fetched only when the
    run starts.  A document without them (any other refusal) gives ``None``: an accepted fit shows the grid
    alone and a refusal says what went wrong in words.
    """

    record = document.get("memory") if isinstance(document, dict) else None
    if not isinstance(record, dict):
        return None
    need, budget = record.get("peak_envelope_bytes"), record.get("budget_bytes")
    if not isinstance(need, (int, float)) or not isinstance(budget, (int, float)) or budget <= 0:
        return None
    return {"need_gib": round(need / _GIB, 2), "budget_gib": round(budget / _GIB, 2), "fits": need <= budget}


_OUTSIDE = re.compile(r"outside (.+?) coverage", re.I)
# gpuwm.data_assets ends every refusal over its companion data package with the one line that fixes it
_COMPANION_FIX = re.compile(r"pip install (?:--force-reinstall )?gpuwm-data(?:==[0-9A-Za-z.+!_-]+)?")
#: gpuwm.data_assets' refusal when the data tables are a different release from the program that reads them.
_COMPANION = re.compile(r"mismatched companion: gpuwm (\S+) requires (\S+) (\S+?), found (\S+?)\.(?:\s|$)")
_PIP_LINE = re.compile(r"^\s*(pip install \S+)\s*$", re.M)


def plain_refusal(text: str, draft: dict[str, Any], memory: dict[str, Any] | None,
                  source_name: str) -> tuple[str, str]:
    """The engine's fit refusal as one sentence a person can act on, and what to change.

    The engine's own text names grid indices, file paths and flags; it goes to the server log.  A refusal this does
    not recognise still says what happened in plain words and where the full reason is.
    """

    card = str(draft.get("card") or "").replace("gb", " GB")
    if memory and not memory["fits"]:
        return (f"Too big for the {card} card: this grid needs about {memory['need_gib']:.1f} GiB and "
                f"{memory['budget_gib']:.1f} GiB is usable.",
                "Make the box smaller, pick a coarser grid, or pick a bigger card.")
    if "--nz must be at least" in text:
        return ("The engine needs at least 4 vertical levels.", "Pick one of the level counts the page lists.")
    companion = _COMPANION.search(text)
    if companion:
        program, data, required, found = companion.groups()
        pip = _PIP_LINE.search(text)
        command = pip.group(1) if pip else f"pip install {data}=={required}"
        return (f"The program is gpuwm {program} but its data tables are {data} {found}; the two must be the "
                f"same version ({required}).",
                f"Run {command} in a terminal, then check again.")
    if _OUTSIDE.search(text):
        return (f"The box reaches outside the area {source_name} covers.",
                "Move the box inside that area, or pick a source that covers the whole globe.")
    if "took longer than" in text:
        return ("The engine took too long to answer.", "Check again.")
    if "does not have the key(s)" in text and "'nz'" in text:
        # an engine older than the level choice: it refuses the key rather than running its default under it
        return ("The engine this page runs on is older than the level choice and cannot set the number of levels.",
                "Pick Engine default, or update the engine to this page's version.")
    companion = _COMPANION_FIX.search(text)
    if companion:
        return ("The engine's data package on this computer does not match this version of the engine, so the "
                "engine will not run.", f"Install the matching one, then check again: {companion.group(0)}")
    return ("The engine could not fit this box. The full reason is in the page server's log.",
            "Change the box, the source or the card and check again.")


def describe_fit(resolved: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    """The wizard's answer as the numbers the map draws and one plain sentence."""

    experiment = ((resolved.get("configuration") or {}).get("experiment") or {})
    projection = experiment.get("projection") or {}
    domains = []
    for domain in experiment.get("domains") or []:
        run = domain.get("run") or {}
        nx, ny = int(run.get("nx") or 0), int(run.get("ny") or 0)
        dx = float(run.get("dx") or 0.0) / 1000.0
        domains.append({
            "grid_id": domain.get("grid_id"), "parent_id": domain.get("parent_id"),
            "nx": nx, "ny": ny, "nz": run.get("nz"), "dx_km": dx,
            "width_km": round((nx - 1) * dx, 1), "height_km": round((ny - 1) * dx, 1),
            "i_parent_start": domain.get("i_parent_start"), "j_parent_start": domain.get("j_parent_start"),
            "parent_grid_ratio": domain.get("parent_grid_ratio"),
        })
    physics = ""
    for line in str(resolved.get("generated_config") or "").splitlines():
        if line.startswith("# PHYSICS:"):
            physics = line[len("# PHYSICS:"):].strip()
            break
    words = []
    if domains:
        root = domains[0]
        words.append(f"{len(domains)} domain{'s' if len(domains) != 1 else ''}. The outer one is "
                     f"{root['nx']} by {root['ny']} points at {root['dx_km']:g} km, "
                     f"{root['width_km']:g} by {root['height_km']:g} km, {root['nz']} levels.")
        for child in domains[1:]:
            words.append(f"Domain {child['grid_id']}: {child['nx']} by {child['ny']} points at "
                         f"{child['dx_km']:g} km.")
    seconds = experiment.get("run_seconds")
    if seconds:
        hours = float(seconds) / 3600
        start = str(experiment.get("start_time") or "").replace("T", " ")[:16]
        words.append(f"{hours:g} hour{'' if hours == 1 else 's'} from {start} UTC.")
    warnings = [str(item.get("action") or item.get("message") or item) if isinstance(item, dict) else str(item)
                for item in resolved.get("warnings") or []]
    return {
        "centre": {"lat": projection.get("ref_lat"), "lon": projection.get("ref_lon")},
        # The map draws the fitted grid in this same projection, so its outline is the grid's true shape.
        "projection": {key: projection.get(key) for key in
                       ("map_proj", "ref_lat", "ref_lon", "truelat1", "truelat2", "stand_lon")},
        "requested": {"lat": draft["lat"], "lon": draft["lon"],
                      "width_km": draft["width_km"], "height_km": draft["height_km"]},
        "domains": domains,
        "physics": physics,
        "run_seconds": seconds,
        "start_time": experiment.get("start_time"),
        "words": " ".join(words),
        "warnings": warnings,
    }


def describe_following(result: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    """The cyclone setup's answer as the numbers the page draws and one plain sentence, like describe_fit."""

    domains = []
    for domain in result.get("domains") or []:
        nx, ny = int(domain.get("nx") or 0), int(domain.get("ny") or 0)
        dx = float(domain.get("dx_m") or 0.0) / 1000.0
        domains.append({"grid_id": domain.get("grid_id"), "parent_id": domain.get("parent_id"),
                        "nx": nx, "ny": ny, "nz": domain.get("nz"), "dx_km": dx,
                        "width_km": round((nx - 1) * dx, 1), "height_km": round((ny - 1) * dx, 1),
                        "following": bool(domain.get("following"))})
    words = []
    if domains:
        root = domains[0]
        words.append(f"{len(domains)} domain{'s' if len(domains) != 1 else ''}. The outer one is "
                     f"{root['nx']} by {root['ny']} points at {root['dx_km']:g} km, "
                     f"{root['width_km']:g} by {root['height_km']:g} km, {root['nz']} levels.")
        for child in domains[1:]:
            moves = " It follows the storm." if child["following"] else ""
            words.append(f"Domain {child['grid_id']}: {child['nx']} by {child['ny']} points at "
                         f"{child['dx_km']:g} km.{moves}")
    seconds = float(result.get("hours") or draft["hours"]) * 3600
    start = str(result.get("start_time") or "").replace("T", " ")[:16]
    if start:
        words.append(f"{seconds / 3600:g} hour{'' if seconds == 3600 else 's'} from {start} UTC.")
    memory = result.get("memory") or {}
    need, budget = memory.get("peak_envelope_bytes"), memory.get("budget_bytes")
    priced = None
    if need and budget:
        priced = {"need_gib": round(float(need) / 2**30, 2), "budget_gib": round(float(budget) / 2**30, 2)}
        priced["fits"] = priced["need_gib"] <= priced["budget_gib"]
    point = result.get("point") or [draft["lat"], draft["lon"]]
    return {"centre": {"lat": point[0], "lon": point[1]}, "projection": {},
            "requested": {"lat": draft["lat"], "lon": draft["lon"],
                          "width_km": draft["width_km"], "height_km": draft["height_km"]},
            "domains": domains, "physics": str(result.get("profile") or ""), "run_seconds": seconds,
            "start_time": result.get("start_time"), "words": " ".join(words), "warnings": [],
            "memory": priced, "following": True}


def preferred_order(rows: list[dict[str, Any]]) -> list[str]:
    """The order New forecast suggests sources in, from what the registry says of each.

    Certified against stock WRF first, then sources that cover the whole globe (a regional one fits only boxes inside
    its own grid), then forecasts before reanalyses; the registry's own order breaks ties.  No source is named here.
    """

    ranked = sorted(enumerate(rows), key=lambda item: (not item[1]["certified"], item[1]["regional"],
                                                       item[1]["analysis"], item[0]))
    return [row["id"] for _, row in ranked]


# ------------------------------------------------------------------ the event page's button

class SimulateMixin:
    """Start an event's best run for one card size: the row's own intent, as the recipe designer checked it."""

    root: Path
    runner: Runner
    wiki: Wiki

    def simulate(self, payload: dict[str, Any], dry: bool) -> Reply:
        ident = str(payload.get("event") or "").strip()
        store = self.wiki.store.data()
        event = store["events"].get(ident)
        recipe = store["recipes"].get(ident)
        if event is None or recipe is None:
            raise ApiError(404, "This event has no best run to start.", "Open New forecast to set one up by hand.")
        card_gb = _number(payload, "card_gb", 1, 1024)
        row = recipe_row(recipe, card_gb)
        if row is None:
            raise ApiError(404, f"This event has no layout for a {card_gb:g} GB card.", "Pick another card size.")
        offered = {item["id"]: item for item in self._offered()["sources"]}
        source = str(row.get("source") or "")
        if source not in offered:
            raise ApiError(422, f"This computer's engine cannot start from {source.upper()}.",
                           "Run gpuwm doctor in a terminal to see what is missing.")
        system = self._system_or_none()
        own = (system or {}).get("memory_total_gib")
        # A layout sized for a bigger card than this one runs out of card memory partway into the forecast.
        if own and float(row["card_gb"]) > own + 0.6:
            raise ApiError(422, f"This layout is for a {row['card_gb']} GB card and this computer's card has "
                                f"{own:.1f} GiB, so it would run out of card memory.",
                           "Pick the layout for your card size or a smaller one.")
        refuse_without_geography(self.runner)
        free = disk_free_gib(self.root)
        need = row.get("disk_gib")
        # A run that fills the disk stops partway with nothing usable and can stop other work on that disk.
        if need and free is not None and float(need) > free:
            raise ApiError(507, f"This run writes about {float(need):.0f} GiB and the disk that holds your runs has "
                                f"{free:.0f} GiB free, so it would stop partway when the disk fills.",
                           "Free some disk, or pick a smaller card size: its layout writes less.")
        start = str(row.get("start") or recipe.get("start") or "")
        name = str(payload.get("name") or "").strip() or run_name(
            recipe.get("title") or event.get("title"), start, row["card_gb"], lambda n: (self.root / n).exists())
        require_name(name, "run name")
        rundir = self.root / name
        if rundir.exists():
            raise ApiError(409, f"A run called {name} already exists.", "Give this run another name.")
        # Read as Start reads it: a start from the last day whose check is still out is waited for, so the button
        # does not refuse a start its check is about to confirm.
        answer = self.settled_availability(source, row["intent"]["cycle"] if row.get("intent") else start[:13],
                                           float(row.get("length_h") or 6))
        if answer.get("starts") == "queue":
            raise ApiError(422, answer["why"], "Open Customise and choose Queue it: the forecast waits for this start "
                                               "and begins once a check confirms it.")
        if answer.get("starts", "now") != "now":
            raise ApiError(422, answer["why"], "Pick another card size, or open Customise to choose another start.")
        link = {"event": ident, "title": f"{event.get('title') or ident}, {row['card_gb']} GB layout",
                "card_gb": row["card_gb"], "rung": row.get("rung"), "door": row.get("door"),
                "est_minutes": row.get("est_minutes"), "est_total_minutes": row.get("est_total_minutes"),
                "disk_gib": need}
        route = offered[source]["route"]
        prepare = None
        if row.get("door") == "cyclone-setup":
            args = dict(row.get("args") or {})
            # --flag=value, not two words: a southern storm's "-29.50,-41.00" read as a flag of its own.
            prepare = engine_argv("cyclone-setup", *[f"--{key.replace('_', '-')}={value}"
                                                     for key, value in args.items()],
                                  "--out", str(rundir / "cyclone.toml"), "--json")
            config: dict[str, Any] = {"path": str(rundir / "cyclone.toml")}
            route = "experiment"
        else:
            config = {"intent": {**row["intent"], "polygon": str(rundir / "region.geojson")}}
        plan: dict[str, Any] = {"schema": PLAN_SCHEMA, "name": name, "route": route, "config": config,
                                "output_root": str(rundir)}
        if prepare is not None:
            # A configuration file's run draws nothing unless its plan says what to draw, so the storm-following
            # layout names the standard set.  An intent plan gets that set from the engine by default, as New
            # forecast's plan of the same layout does, so the two plans are the same document.
            from gpuwm.first_products import DEFAULT_RENDER_PRODUCTS

            plan["run_options"] = {"render_products": DEFAULT_RENDER_PRODUCTS}
        argv = engine_argv("run-plan", str(rundir / runs.PLAN))
        if dry:
            return _dry(argv, run=name, plan=plan, prepare=prepare, link=link)
        # A press on a busy card is refused before anything is written: the folder of a start that cannot run read as
        # a Ready run nobody asked for, and the storm set-up below is an engine call of its own.
        holder = self.runner.card_holder()
        if holder:
            raise ApiError(409, card_taken_words(holder),
                           "Wait for the running forecast to finish, or open Customise and choose Queue it.")
        reserve_run(rundir)
        with releasing_on_failure(rundir):
            box = row.get("box") or {}
            if box:
                write_json(rundir / "region.geojson", region_polygon(float(box["lat"]), float(box["lon"]),
                                                                    float(box["width_km"]), float(box["height_km"])))
            write_json(rundir / RUN_LINK, link)
            if prepare is not None:
                log_command(rundir, display(prepare), "Set up the storm-following nest")
                try:
                    self.runner.query(prepare, cwd=rundir)
                except Refused as error:
                    print(f"gpuwm gui: cyclone-setup refused: {display(prepare)}\n{error}", file=sys.stderr,
                          flush=True)
                    raise ApiError(422, "The engine could not set up the storm-following nest. The full reason is "
                                        "in the page server's log.", "Pick another card size.") from error
            write_json(rundir / runs.PLAN, plan)
        # The card can be taken between the check above and this start; a refusal here removes the folder too.
        return self._launch_new(name, rundir, argv)

    def _system_or_none(self) -> dict[str, Any] | None:
        try:
            devices = self.system().get("devices") or []
        except ApiError:
            return None
        return devices[0] if devices else None


def disk_free_gib(path: Path) -> float | None:
    """Free space on the disk that holds ``path`` (or its nearest existing parent), in GiB."""

    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return round(shutil.disk_usage(probe).free / 2**30, 1)
    except OSError:
        return None


# ------------------------------------------------------------------ system

class SystemMixin:
    root: Path
    runner: Runner
    version: str

    def _saved_path(self, key: str) -> Path | None:
        return self.root / SERVER_DIR / "engine" / f"{key}.json" if key in SAVED_KEYS else None

    def _saved(self, key: str, argv: list[str]) -> dict[str, Any] | None:
        """The last answer this engine gave for ``key``, kept under the runs folder, when the same command asked it."""

        path = self._saved_path(key)
        saved = read_json(path) if path is not None else None
        if (not isinstance(saved, dict) or saved.get("argv") != list(argv) or saved.get("version") != self.version
                or not isinstance(saved.get("document"), dict)):
            return None
        return saved["document"]

    def _save(self, key: str, argv: list[str], document: dict[str, Any]) -> None:
        path = self._saved_path(key)
        if path is None:
            return
        try:
            with self._save_lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path, {"argv": list(argv), "version": self.version, "document": document})
        except OSError as error:
            print(f"gpuwm gui: could not keep the engine's answer in {path}: {error}", file=sys.stderr, flush=True)

    def _cached(self, key: str, argv: list[str], ttl: float, *,
                bad_request_fix: str = "") -> dict[str, Any]:
        # An answer past its time is still served at once while one background call fetches the next: a page
        # never waits for the engine to start again for something it has already been told.  The source list and
        # the physics sets are also kept under the runs folder, so after a restart the first page is answered from
        # the last answer while the engine is asked again.  Only a key never answered on this runs folder waits,
        # and then for one engine call per key: a page asking while the start-up warm call is still running waits
        # for that answer instead of starting the engine a second time.
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is None:
                saved = self._saved(key, argv)
                if saved is not None:
                    hit = self._cache[key] = (time.monotonic() - ttl, saved)
            if hit is not None:
                if time.monotonic() - hit[0] >= ttl and key not in self._refreshing:
                    self._refreshing.add(key)
                    threading.Thread(target=self._refresh, args=(key, argv), daemon=True,
                                     name=f"arwen-gui-refresh-{key}").start()
                return hit[1]
            gate = self._key_locks.setdefault(key, threading.Lock())
        with gate:
            with self._cache_lock:
                hit = self._cache.get(key)
                if hit is not None and time.monotonic() - hit[0] < ttl:
                    return hit[1]
            try:
                document = self.runner.query(argv)
            except Refused as error:
                if error.document and error.document.get("error"):
                    # The engine read the request and said what is wrong with it.
                    raise ApiError(400, str(error), bad_request_fix,
                                   argv=argv, command=display(argv)) from error
                print(f"gpuwm gui: the engine did not answer: {display(argv)}\n{error}", file=sys.stderr, flush=True)
                raise ApiError(502, "The engine did not answer. The full reason is in the page server's log.",
                               "Run gpuwm doctor in a terminal to see what is missing.",
                               argv=argv, command=display(argv)) from error
            with self._cache_lock:
                self._cache[key] = (time.monotonic(), document)
            self._save(key, argv, document)
            return document

    def _refresh(self, key: str, argv: list[str]) -> None:
        try:
            document = self.runner.query(argv)
            with self._cache_lock:
                self._cache[key] = (time.monotonic(), document)
            self._save(key, argv, document)
        except Exception as error:  # noqa: BLE001 - the old answer stays; the log says why the new one did not come
            print(f"gpuwm gui: refreshing {display(argv)} failed; the page keeps the last answer.\n{error}",
                  file=sys.stderr, flush=True)
        finally:
            with self._cache_lock:
                self._refreshing.discard(key)

    def warm(self) -> threading.Thread:
        """Ask the engine for the sources and the card in the background, so New forecast opens on answers."""

        def one(ask) -> None:
            try:
                ask()
            except Exception:  # noqa: BLE001 - the page asks again and shows the error itself
                pass

        # Asked side by side: the source list (which the When step's rows and the wiki wait for on a cold start)
        # does not queue behind the card probe.
        helpers = [threading.Thread(target=one, args=(ask,), name=f"arwen-gui-warm-{i}", daemon=True)
                   for i, ask in enumerate((
                       lambda: self._cached("profiles", engine_argv("run-plan", "--physics-profiles"), CACHE_S),
                       self.sources, self.system))]

        def run() -> None:
            for helper in helpers:
                helper.start()
            for helper in helpers:
                helper.join()

        thread = threading.Thread(target=run, name="arwen-gui-warm", daemon=True)
        thread.start()
        return thread

    def sources(self, now: datetime | None = None) -> dict[str, Any]:
        """New forecast's source list and the start a new draft opens on, answered at once.

        The opening start is the newest start Start takes for the source the page picks first, by the rule in
        :mod:`.availability`: the newest one a check confirmed whole, or the newest one past the source's usual
        publication delay that no check found missing, whichever is newer.  ``default_newest_run`` names it once
        a check confirmed it whole and found every start after it not yet whole, and is null until then.  The
        page works the same rule out from its own clock and the rows' answers as they arrive; nothing here waits
        for a server.
        """

        moment = now or _utcnow()
        offered = self._offered(moment)
        preferred = offered["preferred"]
        if not preferred:
            return {**offered, "default_newest_run": None, "default_checking": False}
        newest = self.board.newest(preferred[0], 6, now=moment)
        return {**offered, "default_cycle": newest["opening"] or offered["default_cycle"],
                "default_newest_run": newest["newest_run"], "default_checking": newest["checking"]}

    def _offered(self, now: datetime | None = None) -> dict[str, Any]:
        """The source list with the schedule's opening estimate: offline, for every caller that needs only names."""

        sources_argv = engine_argv("run-plan", "--sources")
        profiles_argv = engine_argv("run-plan", "--physics-profiles")
        registry = self._cached("sources", sources_argv, CACHE_S)
        menu = self._cached("profiles", profiles_argv, CACHE_S)
        menus = {row.get("source_id"): row for row in menu.get("sources") or []}
        facts = {row.get("profile_id"): row for row in menu.get("profiles") or []}
        rows = []
        for row in registry.get("sources") or []:
            plan = row.get("run_plan") or {}
            if not plan.get("intent_supported") or plan.get("requires_source_root"):
                continue
            # Each source runs on the route the engine names for it: the prepared chain, or the config-driven
            # one the reanalysis runs on.  Offering only the prepared chain hid every date before the
            # operational archives begin.
            routes = plan.get("intent_routes") or []
            if not routes:
                continue
            source_id = row.get("source_id")
            own = menus.get(source_id) or {}
            profiles = [{"id": item.get("profile_id"), "default": bool(item.get("is_default")),
                         "summary": (facts.get(item.get("profile_id")) or {}).get("summary") or item.get("profile_id")}
                        for item in own.get("profiles") or [] if item.get("admissible")]
            coverage = row.get("coverage") or {}
            bounds = None
            if all(isinstance(coverage.get(key), (int, float)) for key in ("west", "east", "south", "north")):
                bounds = {key: round(float(coverage[key]), 3) for key in ("west", "east", "south", "north")}
            for item in profiles:
                verified = (facts.get(item["id"]) or {}).get("maturity") or {}
                item["status"] = verified.get("verification_status") or verified.get("registry_maturity")
            maturity = row.get("maturity") or {}
            rows.append({"id": source_id, "name": row.get("display_name") or source_id,
                         "route": "prepared" if "prepared" in routes else routes[0],
                         "horizon_hours": row.get("max_forecast_hour"),
                         "analysis": row.get("max_forecast_hour") == 0,
                         # What the bytes are, from the registry's own column: the horizon cannot say it, since an
                         # analysis cycle publishes short forecasts too.
                         "record_kind": row.get("record_kind") or "forecast",
                         "certified": str(maturity.get("status") or "").startswith("certified"),
                         "regional": bool(row.get("coverage")),
                         "step_hours": max(1, int(round(float(row.get("forcing_interval_seconds") or 3600) / 3600))),
                         "coverage": bounds, "coverage_words": coverage.get("describe"),
                         "status": maturity.get("status"),
                         "default_profile": own.get("default_profile_id"), "profiles": profiles})
        preferred = preferred_order(rows)
        moment = now or _utcnow()
        # Each source's hours and usual publication delay, so a page works out where to open from its own clock
        # before any row's check has answered.
        for row in rows:
            row.update(_start_facts(row["id"]))
        opening = default_cycle(moment)
        if preferred:
            from gpuwm.source_availability import availability

            try:
                # Asks no server: this list also serves pages that only need source names.  sources() puts the
                # checks' answers to it.
                opening = availability(preferred[0], 6, now=moment)["due_start"] or opening
            except (ValueError, KeyError):
                pass
        return {"sources": rows, "default_cycle": opening, "cycles": recent_cycles(now=moment), "cards": list(CARDS),
                "preferred": preferred, "ladders": ladder_names(),
                "commands": [display(sources_argv), display(profiles_argv)]}

    def availability_of(self, source: str, cycle: str, hours: float, now: datetime | None = None) -> dict[str, Any]:
        """One source's answer for one start (:meth:`.availability.Board.row`): what it reads, what takes it
        (``starts``), the reason, its archive bounds and the source's newest start.

        Never waits for a server: the fetch's own probe of a start from the last day, and of the newest start
        offered, runs in the background, and its answer is read once it is in (``checking`` true until then).
        """

        return self._row(source, source, cycle, float(hours), now or _utcnow())

    def _row(self, source: str, name: str, cycle: str, hours: float, moment: datetime) -> dict[str, Any]:
        """One source's row, computed on its own: a check that raises is that row's "could not be checked", never
        the request's error.  The source list, the fit, Start, Queue it and the event start all read rows here.
        """

        try:
            return self.board.row(source, name, cycle, hours, now=moment)
        except Exception:  # noqa: BLE001 - one source's failed answer is its own row's, never the whole request's
            print(f"gpuwm gui: the check of {source} for {cycle}, {hours:g} h failed:\n"
                  f"{traceback.format_exc()}", file=sys.stderr, flush=True)
            # Neither Start nor Queue it takes a start whose check failed: nothing says its download works.
            reason = "This source could not be checked. The full reason is in the page server's log."
            return {"id": source, "name": name, "state": "unknown", "reason": reason, "why": reason,
                    "starts": "no", "earliest": None, "cycle_hours": [], "analysis": False, "checking": False,
                    "basis": "schedule", "checked_age_s": None, "confirmed": None, "missing": [], "due": None,
                    "opening": None, "newest_run": None, "newest_checking": False, "usual_delay_hours": None}

    def settled_availability(self, source: str, cycle: str, hours: float,
                             now: datetime | None = None) -> dict[str, Any]:
        """:meth:`availability_of` as a request that writes this start reads it: Start, Queue it and an event's
        button.  When the start's own check is still out, it waits for that check, at most
        :data:`.availability.SETTLE_S` seconds (:meth:`.availability.Board.settle`), and reads the row again.
        """

        moment = now or _utcnow()
        answer = self.availability_of(source, cycle, hours, now=moment)
        if answer.get("checking") and answer.get("basis") in ("checking", "unchecked"):
            self.board.settle(source, cycle, float(hours), now=moment)
            answer = self.availability_of(source, cycle, hours, now=moment)
        return answer

    def source_availability(self, query: dict[str, list[str]]) -> dict[str, Any]:
        """Every source's row for one start, answered at once; ``pending`` counts the rows still being checked."""

        def one(key: str) -> str:
            return ((query.get(key) or [""])[0] or "").strip()

        cycle = one("time") or default_cycle()
        if not _CYCLE.match(cycle):
            raise ApiError(400, "The time must look like 2026-09-24T12.", "")
        try:
            hours = float(one("hours") or 6)
        except ValueError as error:
            raise ApiError(400, "hours must be a number.", "") from error
        if not math.isfinite(hours) or not 0 < hours <= 384:
            raise ApiError(400, "hours must be between 1 and 384.", "")
        moment = _utcnow()
        offered = self._offered(moment)
        rank = {ident: i for i, ident in enumerate(offered["preferred"])}
        ordered = sorted(offered["sources"], key=lambda row: rank.get(row["id"], 10**6))
        rows = []
        for row in ordered:
            answer = self._row(row["id"], row["name"], cycle, hours, moment)
            # The row already shows the source's name, so its reason leaves the name out: fourteen rows each
            # saying "GFS (global, 0.25 degree) The GFS (global, 0.25 degree) archive ..." read as a wall.
            rows.append({**answer, "id": row["id"], "name": row["name"],
                         "why": answer.get("reason", answer.get("why", ""))})
        best = next((row["id"] for row in rows if row["state"] == "yes" and not row["checking"]), None) \
            or next((row["id"] for row in rows if row["state"] == "yes"), None)
        pending = sum(1 for row in rows if row["checking"] or row.get("newest_checking"))
        return {"time": cycle, "hours": hours, "sources": rows, "best": best, "pending": pending}

    def offered_sources(self) -> dict[str, str] | None:
        """New forecast's sources as id to display name, or None when the engine cannot say (the wiki offers none)."""
        try:
            return {row["id"]: row["name"] for row in self._offered()["sources"]}
        except ApiError:
            return None

    def system(self) -> dict[str, Any]:
        argv = engine_argv("run-plan", "--probe", "--no-readiness")
        document = self._cached("system", argv, SYSTEM_CACHE_S)
        devices = []
        for device in document.get("devices") or []:
            total = device.get("memory_total_bytes")
            devices.append({"name": device.get("name"), "index": device.get("index"),
                            "memory_total_gib": None if total is None else round(total / 2**30, 1),
                            "memory_free_gib": None if device.get("memory_free_bytes") is None
                            else round(device["memory_free_bytes"] / 2**30, 1)})
        card = None
        if devices and devices[0]["memory_total_gib"]:
            fitting = [c for c in CARDS if int(c[:-2]) <= devices[0]["memory_total_gib"] + 0.6]
            card = fitting[-1] if fitting else CARDS[0]
        return {"devices": devices, "card": card, "version": document.get("gpuwm_version"),
                "card_gb": None if card is None else int(card[:-2]),
                # the card itself, which is not the layout size: a 10 GB card runs the 8 GB layout
                "memory_gib": devices[0]["memory_total_gib"] if devices else None,
                "disk_free_gib": disk_free_gib(self.root),
                "error": document.get("device_query_error"), "command": display(argv)}


# ------------------------------------------------------------------ physics

class PhysicsMixin:
    """The physics composer: the engine's catalog and its check, nothing restated.

    Both answers come from ``gpuwm physics-catalog``, so the page, the
    terminal and an assistant read one table and one set of refusals.
    """

    runner: Runner

    def physics(self, query: dict[str, list[str]]) -> dict[str, Any]:
        source = (query.get("source") or [""])[0].strip()
        argv = engine_argv("physics-catalog", "--json")
        if source:
            if not re.match(r"^[a-z0-9][a-z0-9._-]{0,63}$", source):
                raise ApiError(400, "Pick a data source.", "GET /api/sources lists them.")
            argv += ["--source", source]
        document = dict(self._cached("physics:" + source, argv, CACHE_S,
                                     bad_request_fix="GET /api/sources lists the data sources."))
        document["command"] = display(argv)
        return document

    def physics_check(self, payload: dict[str, Any], dry: bool) -> Reply:
        request = {key: payload[key] for key in
                   ("suite", "preset", "choices", "settings", "dx_km", "nz", "source", "card",
                    "cycle", "hours", "lat", "lon")
                   if key in payload and payload[key] not in (None, "")}
        argv = engine_argv("physics-catalog", "--json", "--check",
                           json.dumps(request, separators=(",", ":"), sort_keys=True))
        if dry:
            return _dry(argv)
        try:
            verdict = self.runner.query(argv)
        except Refused as error:
            # A request the check cannot read (an unknown scheme, preset or
            # source) comes back as the engine's own error document.
            status = 400 if error.document and error.document.get("error") else 422
            raise ApiError(status, str(error), "Check the names against GET /api/physics.",
                           argv=argv, command=display(argv)) from error
        if verdict.get("error"):
            raise ApiError(400, str(verdict["error"]), "Check the scheme names against GET /api/physics.",
                           argv=argv, command=display(argv))
        return Reply(200, {"ok": True, "argv": argv, "command": display(argv), "check": verdict})

    def preset_row(self, preset_id: str) -> dict[str, Any]:
        argv = engine_argv("physics-catalog", "--json")
        for row in self._cached("physics:", argv, CACHE_S).get("presets") or []:
            if row.get("id") == preset_id:
                return row
        raise ApiError(400, f"No physics preset called {preset_id}.", "GET /api/physics lists the presets.")
# ------------------------------------------------------------------ machines

def _machine_error(error: MachineError) -> ApiError:
    return ApiError(error.status, error.message, error.fix)


def _after_start(run_id: str, what: str, error: BaseException) -> str:
    """A step after an accepted start that did not happen, as one plain sentence; the whole error goes to the
    server's log."""

    print(f"gpuwm gui: {run_id} started, but: {what}: {error!r}", file=sys.stderr, flush=True)
    return f"{what} ({_reason(error)})."


def _reason(error: BaseException) -> str:
    return getattr(error, "strerror", None) or getattr(error, "message", None) or str(error) or type(error).__name__


def _minutes(payload: dict[str, Any], key: str, default: int, high: int) -> int:
    value = _number(payload, key, 0, high, default=default, integral=True)
    return int(value or 0)


class MachinesMixin:
    root: Path
    runner: Runner
    machines: Registry
    version: str

    def _local_machine(self) -> dict[str, Any]:
        row: dict[str, Any] = {"name": LOCAL, "kind": "local", "state": "idle", "detail": "idle",
                               "reachable": True, "version_here": self.version,
                               "version_there": self.version, "version_matches": True}
        holder = self.runner.card_holder()
        if holder:
            row.update(self._holder_row(holder))
        try:
            system = self.system()
            row.update(cards=system.get("devices"), card=system.get("card"))
        except ApiError as error:
            row.update(cards=[], card=None, card_error=error.message)
        return row

    def _holder_row(self, holder: dict[str, Any]) -> dict[str, Any]:
        """This computer's row while its card is held: which forecast runs, since when, and its page.

        From the card lock's record (:meth:`Runner.card_holder`), in the
        words a remote row uses ("running RUN"); the lock's launch refusal
        is for launches, and names job tools only an MCP client has.
        """

        if holder.get("unreadable"):
            return {"state": "busy", "run": None, "title": None, "url": None,
                    "detail": "the card counts as in use: its lock file cannot be read",
                    "fix": f"If nothing is running on the card, remove {holder.get('lock')}."}
        run_id = runs.run_id_of(self.root, holder.get("folder"))
        title = run_title(runs.run_path(self.root, run_id), run_id) if run_id else None
        when = started_text(holder.get("started_utc"))
        detail = f"running {title or holder_name(holder)}" + (f", started {when}" if when else "")
        if run_id is None and holder.get("folder"):
            detail += " (not one of the forecasts on this page)"
        return {"state": "running", "detail": detail, "run": run_id, "title": title,
                "url": run_url(run_id) if run_id else None, "started_utc": holder.get("started_utc"),
                "job_id": holder.get("job_id")}

    def machines_list(self, fresh: bool = False) -> dict[str, Any]:
        try:
            rows = self.machines.listing(fresh=fresh)
        except MachineError as error:
            raise _machine_error(error) from error
        return {"machines": [self._local_machine(), *rows], "file": str(self.machines.file()),
                "version": self.version,
                # The Install box starts from the folder GPUWM_WHEELHOUSE names, when it names one.
                "wheelhouse": os.environ.get("GPUWM_WHEELHOUSE") or ""}

    def machine_detail(self, name: str, fresh: bool = True) -> dict[str, Any]:
        if name == LOCAL:
            return self._local_machine()
        try:
            probe = self.machines.probe(name, fresh=fresh)
            row = next((r for r in self.machines.rows() if r.get("name") == name), {})
        except MachineError as error:
            raise _machine_error(error) from error
        return {**probe, "row": row}

    @staticmethod
    def providers() -> dict[str, Any]:
        from .cloud import providers as table

        return {"providers": {key: {"label": spec.get("label"), "needs": spec.get("needs"),
                                    "defaults": spec.get("defaults"), "credentials": spec.get("credentials"),
                                    "cli": spec.get("cli")} for key, spec in table().items()}}

    def machine_add(self, payload: dict[str, Any], dry: bool) -> Reply:
        replace = bool(payload.pop("replace", False))
        try:
            row = check_row(payload)
            existing = [r for r in self.machines.rows() if r.get("name") == row["name"]]
        except MachineError as error:
            raise _machine_error(error) from error
        if existing and not replace:
            raise ApiError(409, f"There is already a machine called {row['name']}.",
                           "Pick another name, or send replace: true to change it.")
        if dry:
            return Reply(200, {"ok": True, "dry_run": True, "row": row, "file": str(self.machines.file()),
                               "message": "Nothing was saved. This is the row Add writes."})
        if row["kind"] != "ssh":
            self.machines.put(row)
            return Reply(200, {"ok": True, "row": row, "check": None,
                               "message": f"Added {row['name']}. Start it from the Machines list."})
        from .machines import Machine, check

        try:
            report = check(Machine(row))
        except MachineError as error:
            raise ApiError(error.status, f"Not added. {error.message}", error.fix) from error
        self.machines.put(row)
        return Reply(200, {"ok": True, "row": row, "check": report,
                           "message": f"Added {row['name']}: {report.get('detail')}."})

    def machine_action(self, name: str, action: str, payload: dict[str, Any], dry: bool) -> Reply:
        try:
            if action == "check":
                return Reply(200, {"ok": True, **self.machine_detail(name, fresh=True)})
            if action == "remove":
                if dry:
                    return Reply(200, {"ok": True, "dry_run": True,
                                       "message": f"Nothing ran. This removes {name}'s row."})
                self.machines.remove(name)
                return Reply(200, {"ok": True, "message": f"Removed {name}. Nothing on the machine was touched."})
            if action == "install":
                from .machines import install

                wheelhouse = str(payload.get("wheelhouse") or os.environ.get("GPUWM_WHEELHOUSE") or "")
                if not wheelhouse:
                    raise ApiError(400, "Name the folder that holds the wheels.",
                                   "Type the folder that holds this version's gpuwm and gpuwm_data wheels "
                                   "in the Install box, and say which machine it is on.")
                return Reply(200, install(self.machines, name, wheelhouse=wheelhouse,
                                          from_machine=payload.get("from_machine") or None, dry=dry))
            if action in ("start", "stop", "terminate"):
                from .cloud import execute, plan

                row = next((r for r in self.machines.rows() if r.get("name") == name), None)
                if row is None:
                    raise ApiError(404, f"There is no machine called {name!r}.", "")
                if (row.get("kind") or "ssh") == "ssh":
                    raise ApiError(409, f"{name} is an SSH machine; it has no start or stop here.", "")
                if dry:
                    return Reply(200, {"ok": True, "dry_run": True, **plan(row, action),
                                       "message": "Nothing ran and nothing was spent. These are the calls it makes."})
                return Reply(200, execute(self.machines, name, action))
        except MachineError as error:
            raise _machine_error(error) from error
        raise ApiError(404, "No such action.")

    # -------------------------------------------------------------- forecasts on machines

    def create_start_remote(self, machine_name: str, payload: dict[str, Any], dry: bool,
                            queued: Path | None = None, settle: bool = True, queue: bool = False) -> Reply:
        from .machines import require_ready
        from .remote_runs import launch_forecast, plan_for, request_render

        def ready(name: str, report: dict[str, Any]) -> None:
            # One queued while the machine installs waits in line: the queue's own start asks again, and waits
            # while the install goes on.
            if not (queue and (report.get("not_ready") or {}).get("state") == "installing"):
                require_ready(name, report)

        try:
            machine = self.machines.get(machine_name)
            # Asked now, not from the list's cache: an install started a moment ago is not in the last answer.
            probe = self.machines.probe(machine_name, fresh=True)
            if probe.get("version_matches"):
                ready(machine_name, probe)
        except MachineError as error:
            raise _machine_error(error) from error
        versions = {"version_here": probe.get("version_here"), "version_there": probe.get("version_there")}
        if not probe.get("version_matches"):
            # The plan is written here and read there: a plan written by one
            # gpuwm version and run by another can mean different physics,
            # fields or paths than the ones this page showed.
            there = probe.get("version_there") or "no gpuwm"
            raise ApiError(409, f"{machine_name} has {there}, this computer has gpuwm {probe.get('version_here')}. "
                                "A plan made here would be run by different code there.",
                           f"Install this version on {machine_name} (Install on its row), then start again.")
        if not payload.get("card"):
            payload = {**payload, "card": probe.get("card")}
        draft = self.draft(payload, need_name=True, settle=settle, queue=queue)
        physics = self.composed_physics(payload, draft)
        rundir = self.root / draft["name"]
        # A folder the queue wrote for this forecast is the one it starts in; any other is a name clash.
        if rundir.exists() and (queued is None or rundir.resolve() != Path(queued).resolve()):
            raise ApiError(409, f"A run called {draft['name']} is already in {self.root}.",
                           "Give this forecast another name.")
        render_on = str(payload.get("render_on") or machine_name)
        draw = draft["products"] != "none" and render_on != "none"
        if draw and render_on != machine_name:
            try:
                if not self.machines.get(render_on).is_local:
                    ready(render_on, self.machines.probe(render_on, fresh=True))
            except MachineError as error:
                raise _machine_error(error) from error
        region = region_polygon(draft["lat"], draft["lon"], draft["width_km"], draft["height_km"])
        plan = plan_for(machine, self.plan_document(draft, rundir), draft["name"])
        if draw:
            # The render worker draws every frame as it commits; the run itself draws none.
            plan.setdefault("run_options", {})["render_products"] = "none"
        wait_min = _minutes(payload, "wait_min", 0, 24 * 60)
        bound_min = _minutes(payload, "bound_min", 120, 48 * 60)
        engine = [machine.python, "-m", "gpuwm", "run-plan", plan["output_root"] + "/plan.json"]
        where = f"on {machine_name} ({machine.row.get('host')})"
        if dry:
            prepare = {"prepare": self.cyclone_argv(draft, rundir / FOLLOWING_CONFIG)} if draft["following"] else {}
            return Reply(200, {"ok": True, "dry_run": True, "run": draft["name"], "machine": machine_name,
                               "plan": plan, "argv": engine, "command": " ".join(engine), "where": where,
                               "render_on": render_on if draw else None, "wait_min": wait_min,
                               "bound_min": bound_min, **versions, "data": _data_of(draft),
                               "waits_for_data": draft["waits_for_data"], **prepare,
                               "message": "Nothing ran. This is the command the machine runs."})
        # A storm-following layout is set up here, and its configuration and companion files go to the machine
        # beside the plan, which names them by the machine's own path.
        self._compile_following(draft)
        if queued is None:
            reserve_run(rundir)
        else:
            self._queued_handoff(rundir, queued)
        published = [name for name in ("region.geojson", runs.PLAN) + ((PHYSICS_CHOICE,) if physics else ())]
        try:
            published += self._publish_following(draft, rundir)
            write_json(rundir / "region.geojson", region)
            write_json(rundir / runs.PLAN, plan)
            if physics is not None:
                write_json(rundir / PHYSICS_CHOICE, physics)
            mirror = launch_forecast(self.machines, rundir, machine_name, plan, region,
                                     wait_min=wait_min, bound_min=bound_min,
                                     text_files=draft.get("_following_files") or None)
        except (MachineError, OSError) as error:
            if queued is None:
                shutil.rmtree(rundir, ignore_errors=True)
            else:
                # A queued forecast's folder stays as the queue wrote it: only what this attempt published goes,
                # the queue puts its marker back, in its place in line, and tries again.
                for name in published:
                    (rundir / name).unlink(missing_ok=True)
            if isinstance(error, MachineError):
                raise _machine_error(error) from error
            raise ApiError(500, "The forecast's files could not be written here.", str(error)) from error
        # The machine has the forecast from here on (see _launch_now): what follows is said, never raised.
        unsaid, follower = [], None
        if mirror.get("unkept"):
            print(f"gpuwm gui: {draft['name']} started {where}, but its record here could not be written: "
                  f"{mirror['unkept']}", file=sys.stderr, flush=True)
            unsaid.append(f"This computer could not keep its record of it ({mirror['unkept']}), so this page "
                          f"cannot follow it; it runs in {mirror.get('remote_rundir')} there.")
        try:
            log_command(rundir, " ".join(engine), f"Start {where}")
        except OSError as error:
            unsaid.append(_after_start(draft["name"], "The start could not be added to this run's commands.log",
                                       error))
        try:
            if draw and not mirror.get("unkept"):
                request_render(rundir, render_on, products=draft["products"],
                               render_section=draft.get("render_section"))
        except OSError as error:
            unsaid.append(_after_start(draft["name"], f"The pictures could not be asked of {render_on}; Draw it "
                                                      "there, on the Machines page, asks again", error))
        try:
            # with no record here there is nothing to follow
            follower = None if mirror.get("unkept") else self._ensure_follower(draft["name"], rundir)
        except (ApiError, OSError) as error:
            unsaid.append(_after_start(draft["name"], "This page could not start copying the forecast's progress "
                                                      "here; Draw it there, on the Machines page, starts that "
                                                      "again", error))
        return Reply(200, {"ok": True, "run": draft["name"], "url": run_url(draft["name"]),
                           "machine": machine_name, "argv": engine, "command": " ".join(engine),
                           "remote": mirror, "follower": follower, "render_on": render_on if draw else None,
                           **versions,
                           "message": " ".join([f"Started {where}. It keeps running if you close this page.",
                                                *unsaid])})

    def _queued_handoff(self, rundir: Path, queued: Path) -> None:
        """The queue hands a waiting forecast's own folder to its start, and to nothing else.

        A fresh start claims a new folder (:func:`reserve_run`).  A queued
        one starts in the folder the queue wrote when it was queued, which
        exists by then; it is taken only when the queue has marked that very
        folder as starting, under the queue's lock, so a start that names a
        folder the queue did not hand over is refused rather than writing into
        another forecast's folder.
        """

        from .queue import STARTING

        try:
            same = rundir.resolve() == Path(queued).resolve()
        except OSError:
            same = False
        if not same or not (rundir / STARTING).is_file():
            raise ApiError(409, f"{rundir.name} is not being started from the queue.",
                           "Queue it again, or give this forecast another name.")

    def _ensure_follower(self, run_id: str, rundir: Path) -> dict[str, Any] | None:
        from gpuwm.machine_agent import _file_lock

        from .machines import FOLLOW
        from .remote_runs import follow_argv, follower_alive

        # Two pages (or a Draw and a Resume updates) can ask at once. Two followers would both append the
        # machine's events from the same offset, so every record after it would be in the event log twice.
        with _file_lock(rundir / FOLLOW_LOCK):
            if follower_alive(rundir):
                return read_json(rundir / FOLLOW, default=None)
            try:
                job = self.runner.launch_helper(rundir, follow_argv(self.root, run_id), "gui:machines-follow")
            except Refused as error:
                raise ApiError(409, str(error), "") from error
            write_json(rundir / FOLLOW, job)
            return job

    def resume_follow(self, run_id: str, rundir: Path, dry: bool) -> Reply:
        """Start following a machine's run or render again after its follower ended; launches no forecast."""

        from .remote_runs import follow_argv

        remote = read_json(rundir / runs.REMOTE, default=None)
        render = read_json(rundir / runs.RENDER, default=None)
        if not isinstance(remote, dict) and not isinstance(render, dict):
            raise ApiError(409, "This forecast was not run or drawn on another machine, so it has no updates "
                                "from one to resume.", "")
        if dry:
            return _dry(follow_argv(self.root, run_id), run=run_id)
        follower = self._ensure_follower(run_id, rundir)
        return Reply(200, {"ok": True, "run": run_id, "follower": follower,
                           "message": "Following the machine again. Its latest state and pictures appear here."})

    def render_run(self, run_id: str, rundir: Path, payload: dict[str, Any], dry: bool) -> Reply:
        from .machines import require_ready
        from .remote_runs import follow_argv, request_render

        machine_name = str(payload.get("machine") or "")
        if not machine_name:
            raise ApiError(400, "Pick the machine that draws the pictures.", "")
        try:
            if not self.machines.get(machine_name).is_local:
                require_ready(machine_name, self.machines.probe(machine_name, fresh=True))
        except MachineError as error:
            raise _machine_error(error) from error
        products = str(payload.get("products") or "").strip() or None
        section = str(payload.get("render_section") or "").strip() or None
        _check_render_selection(products, section)
        if dry:
            return _dry(follow_argv(self.root, run_id), run=run_id, machine=machine_name,
                        products=products, render_section=section)
        try:
            document = request_render(rundir, machine_name, products=products, render_section=section)
        except MachineError as error:
            raise _machine_error(error) from error
        follower = self._ensure_follower(run_id, rundir)
        return Reply(200, {"ok": True, "run": run_id, "render": document, "follower": follower,
                           "message": f"Drawing on {machine_name}. Pictures appear here as they are written."})

    def stop_remote(self, run_id: str, rundir: Path, remote: dict[str, Any], dry: bool) -> Reply:
        name = str(remote.get("machine"))
        command = f"machine_agent stop --run {rundir.name} (on {name})"
        if dry:
            return Reply(200, {"ok": True, "dry_run": True, "run": run_id, "method": "interrupt",
                               "command": command, "message": "Nothing ran. This is what Stop does."})
        try:
            answer = self.machines.get(name).call("stop", "--run", rundir.name)
        except MachineError as error:
            raise _machine_error(error) from error
        plan = {"method": answer.get("method"), "command": command, "message": answer.get("message")}
        record_stop(rundir, plan)
        return Reply(200, {"ok": True, "run": run_id, **plan})


# ------------------------------------------------------------------ downscale

class DownscaleMixin:
    """A finer forecast inside a finished one, from that run's page (:mod:`.downscale`).

    Review runs the engine's ``--dry-run`` into a draft folder and shows its plan; Start runs the same line into
    a new run folder.  Both are the ``gpuwm downscale`` line the terminal controller builds for the same answers.
    """

    root: Path
    runner: Runner

    def downscale_facts(self, run_id: str, rundir: Path) -> dict[str, Any]:
        info = runs.status(rundir)
        facts = downscale.parent_facts(self.root, rundir, info)
        row = next((item for item in facts["domains"] if item["id"] == facts["domain"]), None)
        hours = None
        if row is not None and row["first"] != row["last"]:
            span = datetime.fromisoformat(row["last"].replace("Z", "+00:00")) \
                - datetime.fromisoformat(row["first"].replace("Z", "+00:00"))
            hours = round(span.total_seconds() / 3600.0, 3)
        name = None
        if facts["domain"] is not None:
            name = downscale.default_name(rundir.name, facts["domain"] + 1, downscale.DEFAULT_RATIO,
                                          lambda candidate: downscale.name_taken(self.root, candidate))
        return {"run": run_id, "state": info["state"], "eligible": facts["eligible"], "reason": facts["reason"],
                "fix": facts["why"], "domains": facts["domains"], "domain": facts["domain"],
                "ratio": downscale.DEFAULT_RATIO, "ratio_range": list(downscale.RATIO_RANGE), "cards": list(CARDS),
                "products": downscale.DEFAULT_PRODUCTS, "window_hours": hours, "name": name,
                "output_minutes": (downscale.minutes_text(row["interval_s"])
                                   if row is not None and row["interval_s"] else None)}

    def _parent_spacing_km(self, history: Path, domain: int) -> float:
        """The parent grid's spacing from the engine's own read of its frames, when the run's records lack it."""

        argv = engine_argv("downscale-parent", str(history), "--parent-domain", str(domain))
        try:
            answer = self.runner.query(argv)
        except Refused as error:
            raise ApiError(422, "The engine could not read this grid's spacing from its saved frames.",
                           "Place the finer forecast by its centre instead of a box.",
                           argv=argv, command=display(argv)) from error
        try:
            return float(answer["dx_m"]) / 1000.0
        except (KeyError, TypeError, ValueError) as error:
            raise ApiError(422, "The engine could not read this grid's spacing from its saved frames.",
                           "Place the finer forecast by its centre instead of a box.") from error

    def downscale_start(self, run_id: str, rundir: Path, payload: dict[str, Any], dry: bool) -> Reply:
        unknown = sorted(set(payload) - downscale.KEYS)
        if unknown:
            raise ApiError(400, f"Downscale does not take {', '.join(unknown)}.",
                           f"It takes {', '.join(sorted(downscale.KEYS))}.")
        mode = str(payload.get("mode") or "run")
        if mode not in downscale.MODES:
            raise ApiError(400, "mode must be plan (review the finer forecast) or run (start it).", "")
        info = runs.status(rundir)
        facts = downscale.parent_facts(self.root, rundir, info)
        domain = _number(payload, "domain", 1, 99, integral=True)
        domain = facts["domain"] if domain is None else int(domain)
        reason, why = downscale.eligibility(info, facts["history"], facts["domains"], domain)
        if reason:
            raise ApiError(409, reason, why)
        row = next(item for item in facts["domains"] if item["id"] == domain)
        spacing: dict[str, float] = {}

        def parent_km() -> float:
            if "km" not in spacing:
                spacing["km"] = float(row["dx_km"]) if row["dx_km"] else self._parent_spacing_km(facts["history"],
                                                                                               domain)
            return spacing["km"]

        ratio = _number(payload, "ratio", *downscale.RATIO_RANGE, integral=True)
        dx_km = _number(payload, "dx_km", 0.001, 1000.0)
        if dx_km is not None:
            derived = int(downscale.round_half_away(parent_km() / dx_km))
            if derived < downscale.RATIO_RANGE[0]:
                raise ApiError(400, f"A {dx_km:g} km forecast is not finer than this {parent_km():g} km grid by a "
                                    "whole refinement of 2 or more.", "Pick a finer spacing.")
            if ratio is not None and int(ratio) != derived:
                raise ApiError(400, f"A refinement of {int(ratio)} and a spacing of {dx_km:g} km disagree: this "
                                    f"{parent_km():g} km grid refined {derived} times is {parent_km() / derived:g} km.",
                               "Send one of them.")
            ratio = derived
        ratio = downscale.DEFAULT_RATIO if ratio is None else int(ratio)
        box = payload.get("box")
        child_size = None
        if box is not None:
            if payload.get("lat") is not None or payload.get("lon") is not None:
                raise ApiError(400, "Send the finer forecast's centre or its box, not both.", "")
            if not isinstance(box, dict):
                raise ApiError(400, "box must be an object with west, south, east and north.", "")
            edges = {key: _number(box, key, -540.0, 540.0) for key in ("west", "south", "east", "north")}
            if any(value is None for value in edges.values()):
                raise ApiError(400, "box must be an object with west, south, east and north.", "")
            bounds = (edges["south"], edges["west"], edges["north"], edges["east"])
            refused = downscale.box_refusal(*bounds)
            if refused:
                raise ApiError(400, refused, "")
            lat, lon = downscale.box_centre(*bounds)
            child_size = downscale.box_cells(*bounds, ratio, parent_km() * 1000.0)
        else:
            lat = _number(payload, "lat", -90.0, 90.0)
            lon = _number(payload, "lon", -180.0, 180.0)
            if lat is None or lon is None:
                raise ApiError(400, "Place the finer forecast: click its centre on the map or draw its box.", "")
        hours = payload.get("hours")
        hours = None if hours in (None, "") else _number(payload, "hours", 0.01, 384.0)
        minutes = payload.get("output_minutes")
        if minutes not in (None, ""):
            output_seconds = float(_number(payload, "output_minutes", 0.01, 1440.0)) * 60.0
        elif row["interval_s"]:
            # the panel's default: the parent's own cadence, as its minutes field spells it
            output_seconds = float(downscale.minutes_text(row["interval_s"])) * 60.0
        else:
            output_seconds = None
        card = str(payload.get("card") or "").strip().lower() or None
        if card in ("auto", "measure"):
            card = None
        if card is not None and card not in CARDS:
            raise ApiError(400, "Pick the card the finer forecast is sized for, or leave it to measure this "
                                "computer's card.", f"One of {', '.join(CARDS)}.")
        products = str(payload.get("products") or "").strip() or downscale.DEFAULT_PRODUCTS
        if products not in ("all", "none") and not downscale.PRODUCTS_RE.match(products):
            raise ApiError(400, "Pictures must be all, none, or product names separated by commas.", "")
        name = str(payload.get("name") or "").strip()
        if name:
            require_name(name, "run name")
        else:
            name = downscale.default_name(rundir.name, domain + 1, ratio,
                                          lambda candidate: downscale.name_taken(self.root, candidate))
        if downscale.name_taken(self.root, name):
            raise ApiError(409, f"A run called {name} already exists.", "Give the finer forecast another name.")
        folder = self.root / SERVER_DIR / "drafts" / uuid.uuid4().hex[:12] if mode == "plan" else None
        # The boundary ceiling is the cadence this grid's frames actually have, the one the page shows: asked for,
        # not accepted on the person's behalf (see gui/downscale.py).
        settings = dict(history=facts["history"], lat=lat, lon=lon, domain=domain, ratio=ratio,
                        child_size=child_size, boundary_seconds=row["interval_s"], hours=hours,
                        output_seconds=output_seconds, card=card, products=products)
        argv = engine_argv("downscale", *downscale.command_args(
            **settings, out=(folder / name) if folder is not None else self.root / name, plan=mode == "plan"))
        start_argv = engine_argv("downscale", *downscale.command_args(**settings, out=self.root / name, plan=False))
        child = {"name": name, "parent": run_id, "parent_domain": domain, "child_domain": domain + 1,
                 "ratio": ratio, "lat": lat, "lon": lon, "child_size": list(child_size) if child_size else None,
                 "hours": hours, "card": card or "measured", "products": products,
                 "dx_km": (parent_km() / ratio) if (row["dx_km"] or child_size) else None}
        if dry:
            return _dry(argv, run=name, mode=mode, child=child)
        if mode == "plan":
            return self._downscale_review(name, argv, start_argv, folder, child)
        return self._downscale_launch(run_id, rundir, name, argv, child)

    def _downscale_review(self, name: str, argv: list[str], start_argv: list[str], folder: Path,
                          child: dict[str, Any]) -> Reply:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            try:
                self.runner.check(argv, cwd=folder)
            except Refused as error:
                print(f"gpuwm gui: the downscale review was refused: {display(argv)}\n{error}",
                      file=sys.stderr, flush=True)
                said = plain_message(re.sub(r"^gpuwm downscale:\s*", "", runs.last_words(str(error)) or ""))
                raise ApiError(422, said or "The engine could not plan this finer forecast. The full reason is in "
                                            "the page server's log.",
                               "Change the place, the refinement or the card and review again.",
                               argv=argv, command=display(argv)) from error
            plan = downscale.review_plan(folder, name)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        if plan is None:
            raise ApiError(502, "The engine reviewed the finer forecast but wrote no plan.",
                           "Review again. The page server's log has the command.", argv=argv, command=display(argv))
        return Reply(200, {"ok": True, "mode": "plan", "run": name, "argv": argv, "command": display(argv),
                           "review": downscale.plan_summary(plan), "child": child, "start_argv": start_argv,
                           "start_command": display(start_argv),
                           "start_shown": plain_command(start_argv, self.root),
                           "message": "Nothing ran. This is the finer forecast Start makes."})

    def _downscale_launch(self, run_id: str, rundir: Path, name: str, argv: list[str],
                          child: dict[str, Any]) -> Reply:
        # Like a forecast's Start: on a card shared through an OWNER file the start takes the card there first,
        # and the child's own wrapper removes the line when it ends (see queue.py).
        owner_file = self.queue.owner_file
        return self.queue.launch_holding_card(
            name, lambda: self._downscale_launch_now(run_id, rundir, name, argv, child, owner_file=owner_file))

    def _downscale_launch_now(self, run_id: str, rundir: Path, name: str, argv: list[str],
                              child: dict[str, Any], owner_file: str | None = None) -> Reply:
        out = self.root / name
        try:
            # left empty: the engine adopts an empty --out and refuses one that holds anything
            out.mkdir(exist_ok=False)
        except FileExistsError as error:
            raise ApiError(409, f"A run called {name} already exists.", "Give the finer forecast another name.") \
                from error
        try:
            job = (self.runner.launch_detached(argv, cwd=self.root, outdir=out, kind="gui:downscale",
                                               owner_file=owner_file) if owner_file
                   else self.runner.launch_detached(argv, cwd=self.root, outdir=out, kind="gui:downscale"))
        except Refused as error:
            try:
                out.rmdir()
            except OSError:
                pass
            raise ApiError(409, str(error), "Wait for the running forecast to finish, or stop it.") from error
        # The finer forecast runs from here on, as a forecast's Start does (see _launch_now): what follows is said
        # beside the start and never raised as a failed one.  Raised, Start said the downscale had failed while
        # it ran, and the card's line was let go while it held the card.
        freed, unsaid = None, []
        try:
            record = runs.start_record_path(out)
            record.parent.mkdir(parents=True, exist_ok=True)
            write_json(record, {"schema": runs.START_SCHEMA, "run": name, "parent": run_id, "argv": argv,
                                "command": display(argv), "job": job,
                                "started_utc": utc_text(datetime.now(timezone.utc))})
        except OSError as error:
            print(f"gpuwm gui: {name} started, but its start record could not be written: {error!r}",
                  file=sys.stderr, flush=True)
            unsaid.append(f"This page could not write its record of the start ({_reason(error)}), so it lists the "
                          "finer forecast only once the forecast writes its own records.")
        try:
            freed = self.assistant.make_room() if getattr(self, "assistant", None) is not None else None
        except Exception as error:  # noqa: BLE001 - the finer forecast is running whatever the assistant does
            unsaid.append(_after_start(name, "The assistant's model could not be moved off the card", error))
        try:
            # the parent's page is where this was asked for, and the child's folder is the engine's until it writes
            log_command(rundir, display(argv), f"Downscale into {name}")
        except OSError as error:
            unsaid.append(_after_start(name, f"The start could not be added to {run_id}'s commands.log", error))
        return Reply(200, {"ok": True, "run": name, "url": run_url(name), "argv": argv, "command": display(argv),
                           "job": job, "assistant": freed, "child": child,
                           "message": " ".join(["Started. The finer forecast keeps running if you close this page.",
                                                *unsaid])})


# ------------------------------------------------------------------ the class

class WikiMixin:
    """The storm wiki's read-only pages (:mod:`.wiki`)."""

    root: Path
    wiki: Wiki

    def wiki_get(self, parts: list[str], query: dict[str, list[str]]) -> Reply:
        if not parts:
            return Reply(body=self.wiki.main())
        head, rest = parts[0], parts[1:]
        if head == "search" and not rest:
            return Reply(body=self.wiki.search(query))
        if head == "changes" and not rest:
            return Reply(body=self.wiki.changes())
        if head == "places" and not rest:
            return Reply(body=self.wiki.places_index())
        if len(rest) == 1:
            ident = rest[0]
            if head == "event":
                return Reply(body=self.wiki.event_page(ident))
            if head == "kind":
                return Reply(body=self.wiki.phenomenon_page(ident))
            if head == "place":
                return Reply(body=self.wiki.place_page(ident))
            if head == "recipe":
                card = ((query.get("card") or [""])[0] or "").strip()
                try:
                    card_gb = float(card) if card else None
                except ValueError as error:
                    raise ApiError(400, "card must be a number of GB.", "") from error
                return Reply(body=recipe_of(self.wiki.store, ident, self.offered_sources(), card_gb))
            if head == "run":
                return Reply(body=self.wiki.run_page(ident, runs.existing_run(self.root, ident)))
        raise ApiError(404, "No such Weather Library page.", "Go to the Weather Library's main page.")


class Api(RunsMixin, PicturesMixin, CreateMixin, SimulateMixin, SystemMixin, PhysicsMixin, MachinesMixin,
          DownscaleMixin, WikiMixin):
    def __init__(self, root: Path, runner: Runner, *, token: str, port: int, version: str,
                 bind: str, machines: Registry | None = None, owner_file: str | None = None) -> None:
        self.root = root
        self.runner = runner
        self.token = token
        self.port = port
        self.version = version
        self.bind = bind
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._cards: dict[str, tuple[tuple[Any, ...], dict[str, Any], float]] = {}
        self._cache_lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._refreshing: set[str] = set()
        self._save_lock = threading.Lock()
        from .availability import Board

        self.board = Board()
        self.wiki = Wiki(root, offered=self.offered_sources)
        self.machines = machines or Registry()
        from .queue import ForecastQueue

        self.queue = ForecastQueue(self, owner_file=owner_file)
        from .assistant.service import Assistant

        self.assistant = Assistant(self)

    def close(self) -> None:
        """The queue stops starting forecasts and the availability checks stop asking hosts.

        A closed page server's checks otherwise went on sending HEADs (126 of them after server_close).  A running
        forecast is not touched.
        """

        self.queue.stop()
        self.board.close()

    def session(self) -> dict[str, Any]:
        from .auth import TOKEN_HEADER

        return {"version": self.version, "token": self.token, "token_header": TOKEN_HEADER,
                "root": str(self.root), "runner": self.runner.kind, "host": socket.gethostname(),
                "bind": self.bind, "port": self.port,
                # optional and off until the person turns it on; the sidebar and Ctrl K say which
                "assistant": {"enabled": self.assistant.enabled()}}

    @staticmethod
    def copy() -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for path in sorted(COPY_DIR.glob("*.json")):
            try:
                merged[path.stem] = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
        return merged

    # -------------------------------------------------------------- routing

    def handle(self, method: str, raw_path: str, query: dict[str, list[str]], body: bytes,
               content_type: str = "") -> Reply:
        reply = self._dispatch(method, raw_path, query, body, content_type)
        # A reply that carries a command also carries it as the page shows it (no machine path);
        # the exact line stays in "command" for Copy.
        if isinstance(reply.body, dict) and isinstance(reply.body.get("argv"), list) and reply.body.get("command"):
            reply.body["shown"] = plain_command(reply.body["argv"], self.root)
        return reply

    #: API sections whose path differs from the name their handler goes by: the Weather Library's
    #: pages are served under library by the handler named wiki, and wiki, their path before the page
    #: took that name, answers the same.
    SECTIONS = {"library": "wiki"}

    def _dispatch(self, method: str, raw_path: str, query: dict[str, list[str]], body: bytes,
                  content_type: str = "") -> Reply:
        segments = [unquote(part) for part in raw_path.split("/") if part]
        if len(segments) > 1 and segments[0] == "api":
            segments[1] = self.SECTIONS.get(segments[1], segments[1])
        try:
            if segments[:1] != ["api"]:
                raise ApiError(404, "No such page.")
            if method in ("GET", "HEAD"):
                return self._get(segments[1:], query)
            if method == "POST":
                payload = self._json_body(body, content_type)
                dry = payload.pop("dry_run", False)
                if not isinstance(dry, bool):
                    raise ApiError(400, "dry_run must be true or false.", "")
                return self._post(segments[1:], payload, dry)
            raise ApiError(405, f"{method} is not used here.")
        except ApiError as error:
            return error.reply()
        except PathRefused as error:
            return Reply(400, {"ok": False, "message": str(error), "fix": "Use a plain name."})
        except FileNotFoundError as error:
            if error.filename:
                print(f"gpuwm gui: {error}", file=sys.stderr, flush=True)
            return Reply(404, {"ok": False, "message": "That is not there." if error.filename
                               else scrub(str(error) or "Not found."), "fix": ""})

    @staticmethod
    def _json_body(body: bytes, content_type: str) -> dict[str, Any]:
        if body and content_type.split(";")[0].strip().lower() != "application/json":
            raise ApiError(415, "The request body must be JSON.", "Send Content-Type: application/json.")
        if not body:
            return {}
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise ApiError(400, f"The request body is not JSON ({error}).", "") from error
        if not isinstance(payload, dict):
            raise ApiError(400, "The request body must be a JSON object.", "")
        return payload

    def _get(self, parts: list[str], query: dict[str, list[str]]) -> Reply:
        if parts == ["session"]:
            return Reply(body=self.session())
        if parts == ["copy"]:
            return Reply(body=self.copy())
        if parts == ["runs"]:
            return Reply(body=self.runs())
        if parts == ["sources"]:
            return Reply(body=self.sources())
        if parts == ["sources", "availability"]:
            return Reply(body=self.source_availability(query))
        if parts == ["system"]:
            return Reply(body=self.system())
        if parts == ["queue"]:
            return Reply(body=self.queue_listing(query))
        if parts[:1] == ["wiki"]:
            return self.wiki_get(parts[1:], query)
        if parts == ["physics"]:
            return Reply(body=self.physics(query))
        if parts == ["machines"]:
            return Reply(body=self.machines_list(fresh=(query.get("fresh") or [""])[0] in ("1", "true")))
        if parts == ["providers"]:
            return Reply(body=self.providers())
        if len(parts) == 2 and parts[0] == "machines":
            return Reply(body=self.machine_detail(parts[1]))
        if parts[:1] == ["assistant"]:
            return self.assistant.get(parts[1:], query)
        if len(parts) >= 2 and parts[0] == "runs":
            run_id, rest = parts[1], parts[2:]
            rundir = runs.existing_run(self.root, run_id)
            if not rest:
                return Reply(body=self.run_detail(run_id, rundir))
            if rest == ["status"]:
                return Reply(body=runs.status(rundir))
            if rest == ["pictures"]:
                return Reply(body=self.pictures(rundir))
            if rest == ["pictures", "list"]:
                return Reply(body=self.picture_list(rundir, query))
            if rest == ["map"]:
                return Reply(body=self.run_map(rundir))
            if rest == ["downscale"]:
                return Reply(body=self.downscale_facts(run_id, rundir))
            if rest[0] == "files" and len(rest) > 1:
                return Reply(file=safe_file(rundir, "/".join(rest[1:])), run_file=True)
        raise ApiError(404, "No such endpoint.")

    def _post(self, parts: list[str], payload: dict[str, Any], dry: bool) -> Reply:
        if parts == ["create", "fit"]:
            return self.fit(payload, dry)
        if parts == ["create", "start"]:
            return self.create_start(payload, dry)
        if parts == ["wiki", "simulate"]:
            return self.simulate(payload, dry)
        if parts == ["physics", "check"]:
            return self.physics_check(payload, dry)
        if parts[:1] == ["assistant"]:
            return self.assistant.post(parts[1:], payload, dry)
        if len(parts) == 3 and parts[0] == "runs":
            run_id, action = parts[1], parts[2]
            rundir = runs.existing_run(self.root, run_id)
            if action == "start":
                return self.start_ready(run_id, rundir, payload, dry)
            if action == "stop":
                return self.stop(run_id, rundir, payload, dry)
            if action == "render":
                return self.render_run(run_id, rundir, payload, dry)
            if action == "follow":
                return self.resume_follow(run_id, rundir, dry)
            if action == "downscale":
                return self.downscale_start(run_id, rundir, payload, dry)
        if len(parts) == 3 and parts[0] == "queue":
            return self.queue_action(parts[1], parts[2], dry)
        if parts == ["machines", "add"]:
            return self.machine_add(payload, dry)
        if len(parts) == 3 and parts[0] == "machines":
            return self.machine_action(parts[1], parts[2], payload, dry)
        raise ApiError(404, "No such action.")


__all__ = ["Api", "ApiError", "Reply", "default_cycle", "describe_fit", "ladder_names", "memory_fit", "recent_cycles", "region_polygon",
           "run_url"]
