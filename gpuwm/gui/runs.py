"""The run folder is the truth: what a folder on disk says about its run.

A run is a folder under the runs root that holds any of

- ``run-manifest.json``, ``events.jsonl``, ``run-progress.json``: what
  ``gpuwm run-plan`` writes (its docstring's reattach recipe: the
  manifest for the pid and the paths, the heartbeat for the current
  state, the event log from byte zero for the history);
- ``plan.json``: a plan this page wrote and has not started yet;
- ``<domain>/<product>/<valid-day>/*.png``: a render tree copied in, which
  opens as ``imported`` (pictures only);
- nothing yet, for a downscale the page just started: ``gpuwm downscale``
  adopts only an empty folder, so the page's record of that start sits in
  the server's folder (:func:`start_record_path`) until the child writes
  its own manifest and event log.

Nothing here keeps state in memory that a restart would lose: a run
started from a terminal lists exactly like one started from the page,
and a reopened page reads the same answer.

States: ``imported``, ``queued``, ``ready``, ``running``, ``finished``,
``stopped``, ``failed``, ``stale`` (no end event and nobody alive to write
one), and ``unreadable`` for a folder whose records could not be read
(listed with a plain reason so the others still load).

Alive means the very process the run recorded, not whatever holds its
PID today: the manifest and the job record carry the process's creation
time beside the PID (:mod:`gpuwm.proc_identity`), and a record whose
process has another identity, or none recorded, reads as ended.  A PID
reused after a crash or a reboot made a crashed run read Running, and
its Stop signalled an unrelated program.

A started job that ended before the engine wrote an end event (a plan
the engine refused at its start, a run killed outright) reads Failed or
Stopped from the job's own result, with its exit code and the first
plain line the engine printed; before this it fell back to Ready and a
failed start said nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import re
from pathlib import Path
from stat import FILE_ATTRIBUTE_REPARSE_POINT, S_ISLNK
import threading
from typing import Any, Iterator

from gpuwm.render_layout import is_scratch_dir

from .files import SCAN_DEPTH, SERVER_DIR, folder_part, load_json, long_path, plain_message, read_json, utc_text, within

MANIFEST = "run-manifest.json"
EVENTS = "events.jsonl"
HEARTBEAT = "run-progress.json"
PLAN = "plan.json"
JOB = "gui-job.json"
STOP = "gui-stop.json"
COMMANDS_LOG = "commands.log"
#: A forecast on another machine: its mirror here (see :mod:`.remote_runs`).
REMOTE = "gui-machine.json"
REMOTE_MANIFEST = "remote-run-manifest.json"
RENDER = "gui-render.json"
#: The follower that mirrors a machine's forecast and its render worker here (``machines.FOLLOW``).
FOLLOW = "gui-follow.json"
#: A forecast waiting in New forecast's queue for its card (see :mod:`.queue`).
QUEUED = "gui-queued.json"
MARKERS = (MANIFEST, EVENTS, HEARTBEAT, PLAN, REMOTE, QUEUED)

#: A machine job in one of these states has ended (see :mod:`.remote_runs`).
REMOTE_END_STATES = ("finished", "failed", "stopped", "refused")
#: The run event saying the renderer has no map assets, so every picture
#: lacks coastlines, borders and state lines.  The engine's own code
#: (``gpuwm.render.BASEMAP_MISSING_CODE``), spelled here because this
#: module stays off the render stack; a test holds the two equal.
BASEMAP_MISSING = "render_basemap_missing"
#: The run event carrying a warning a library the run drives raised (``gpuwm.runplan``'s warning codes): the
#: action as ``message``, the reason as ``detail``.  The run goes on as configured.
LIBRARY_WARNING = "library_warning"
STATES = ("imported", "queued", "ready", "running", "finished", "stopped", "failed", "stale", "unreadable")
#: My forecasts lists at most this many runs, the newest; a larger root says it was cut.
LIST_LIMIT = 4000
READ_CHUNK = 4 * 1024 * 1024
#: Event logs up to this size are read whole for a summary; larger ones
#: are read as their head (the plan) and their tail (the latest state).
WHOLE_READ = 8 * 1024 * 1024
HEAD_READ = 2 * 1024 * 1024
TAIL_READ = 2 * 1024 * 1024


def process_alive(record: Any, pid: Any = None) -> bool:
    """The recorded process (PID and creation time) is running now; a bare or reused PID is not."""

    from gpuwm import proc_identity

    return proc_identity.alive(record, pid)


def manifest_alive(manifest: Any) -> bool:
    if not isinstance(manifest, dict) or manifest.get("pid") is None:
        return False
    return process_alive(manifest.get("process"), manifest.get("pid"))


def has_frames(path: Path) -> bool:
    """A render tree starts here: ``<domain>[/episode-NNN]/<product>/<valid-day>/*.png``, or flat PNGs."""

    for pattern in ("*/*/*/*.png", "*/episode-*/*/*/*.png", "*.png"):
        for found in path.glob(pattern):
            # the renderer's working scratch is never a render tree (png.render-scratch/rwstore-*/png/*.png)
            if not any(is_scratch_dir(part) for part in found.relative_to(path).parts[:-1]):
                return True
    return False


def is_run_dir(path: Path) -> bool:
    return any((path / marker).is_file() for marker in MARKERS) or start_record_path(path).is_file()


# ------------------------------------------------------------------ a downscale started from the page

#: ``gpuwm downscale --out`` adopts only an EMPTY folder (a child's report and frames describe one run), so the
#: page cannot write its ``gui-job.json`` or ``commands.log`` into the run folder it starts one in.  Its record of
#: the start (the job, the exact command, the parent) sits in the server's folder under the run's name instead,
#: until the child's own manifest and event log take over.
STARTS = "starts"
START_SCHEMA = "gpuwm.gui-downscale-start.v1"


def start_record_path(rundir: Path) -> Path:
    return rundir.parent / SERVER_DIR / STARTS / f"{rundir.name}.json"


def start_record(rundir: Path) -> dict[str, Any] | None:
    record = read_json(start_record_path(rundir), default=None)
    return record if isinstance(record, dict) and record.get("schema") == START_SCHEMA else None


def last_words(text: str) -> str | None:
    """The engine's refusal in what a command wrote to stderr: its last ``gpuwm ...:`` line, or its last line."""

    lines = [line.strip() for line in str(text or "").splitlines()
             if line.strip() and "installed wheel at" not in line]
    said = [line for line in lines if line.startswith("gpuwm ")]
    return (said or lines or [None])[-1]


def start_end(record: dict[str, Any], stopped: bool) -> dict[str, Any]:
    """The end of a page-started downscale that wrote nothing of its own: the engine's words from its job log."""

    job = record.get("job") if isinstance(record.get("job"), dict) else {}
    jobs_dir = Path(str(job.get("jobs_dir") or "")) if job.get("jobs_dir") else None
    result = read_json(jobs_dir / "result.json", default=None) if jobs_dir is not None else None
    said = None
    if jobs_dir is not None:
        try:
            with (jobs_dir / "stderr.log").open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 64 * 1024))
                tail = stream.read().decode("utf-8", "replace")
        except OSError:
            tail = ""
        said = last_words(tail)
    if stopped:
        message = "Stopped before the finer forecast wrote anything."
    else:
        message = plain_message(said) or ("The finer forecast ended before it wrote anything, and its log names "
                                          "no reason.")
    return {"event": "failed", "stage": "start", "message": message, "remedy": None, "interrupted": stopped,
            "exit_code": (result or {}).get("exit_code") if isinstance(result, dict) else None, "dry_run": False}


def iter_runs(root: Path) -> Iterator[Path]:
    """Every run folder under ``root``, NAME or FOLDER/NAME, newest first.

    Every folder is found before any is dropped: a root holding more runs
    than the page lists shows its newest ones (``api`` cuts the list and
    says so), where a scan that stopped part way down the alphabet hid
    the newest runs without a word.  A candidate at either depth must
    resolve inside the root: a link to a folder elsewhere is not a run of
    this root, one level down as much as at the top.  Each run is listed
    once, spelled as the folder it resolves to, so the id the list gives
    is the id every page opens.
    """

    try:
        root_real = root.resolve()
        children = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return iter(())
    found: dict[tuple[str, ...], Path] = {}

    def take(candidate: Path) -> bool:
        if not (is_run_dir(candidate) or has_frames(candidate)):
            return False
        parts = _run_parts(root_real, candidate)
        if parts is None:
            return False
        found.setdefault(parts, root.joinpath(*parts))
        return True

    for child in children:
        # a render's working scratch sits beside its case folder (<case>.render-scratch), never a run of its own
        if is_scratch_dir(child.name) or child.name == SERVER_DIR:
            continue
        # A link to a folder elsewhere is not a run of this root (the same rule that
        # confines every file served); listing it made the whole list answer 500.
        if not within(root, child):
            continue
        try:
            if take(child):
                continue
            inner = sorted(p for p in child.iterdir() if p.is_dir() and not is_scratch_dir(p.name))
        except OSError:
            continue
        for candidate in inner:
            try:
                take(candidate)
            except OSError:
                continue
    return iter(sorted(found.values(), key=_mtime, reverse=True))


def _run_parts(root_real: Path, candidate: Path) -> tuple[str, ...] | None:
    """The id a run folder lists under, as folder names: where it resolves under the root.

    None for a folder that resolves outside the root (a link to a folder
    elsewhere, at either depth; one of those reached the plan reader and
    answered 500 for every run) or deeper than NAME or FOLDER/NAME, which
    no page can open, so listing it only showed a row that failed on click.
    """

    try:
        parts = candidate.resolve().relative_to(root_real).parts
    except (OSError, ValueError):
        return None
    if not 1 <= len(parts) <= SCAN_DEPTH or not all(folder_part(part) for part in parts):
        return None
    return parts


def _mtime(path: Path) -> float:
    best = 0.0
    for name in (HEARTBEAT, EVENTS, MANIFEST, PLAN):
        try:
            best = max(best, (path / name).stat().st_mtime)
        except OSError:
            continue
    if not best:
        try:
            best = path.stat().st_mtime
        except OSError:
            pass
    return best


def folder_mtime(path: Path) -> float:
    """The newest of a run folder's marker files, or the folder's own time."""

    return _mtime(path)


# ------------------------------------------------------------------ events

def _records(blob: bytes) -> list[dict[str, Any]]:
    out = []
    for line in blob.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = load_json(line)
        except (ValueError, RecursionError):
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


class EventTail:
    """New complete records of one growing log, poll by poll."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.offset = 0
        self._buffer = b""
        self._identity: tuple[int, int] | None = None

    def read_new(self) -> list[dict[str, Any]]:
        try:
            status = self.path.stat()
        except OSError:
            return []
        identity = (status.st_dev, status.st_ino)
        if self._identity is not None and (identity != self._identity or status.st_size < self.offset):
            self.offset, self._buffer = 0, b""
        self._identity = identity
        if status.st_size <= self.offset:
            return []
        try:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                data = stream.read(READ_CHUNK)
        except OSError:
            return []
        self.offset += len(data)
        self._buffer += data
        complete, _, self._buffer = self._buffer.rpartition(b"\n")
        return _records(complete)


def event_records(path: Path) -> list[dict[str, Any]]:
    """The records a summary needs: all of a small log, head and tail of a big one."""

    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            if size <= WHOLE_READ:
                return _records(stream.read())
            head = stream.read(HEAD_READ)
            stream.seek(max(0, size - TAIL_READ))
            tail = stream.read()
    except OSError:
        return []
    head = head.rpartition(b"\n")[0]
    tail = tail.partition(b"\n")[2]
    return _records(head) + _records(tail)


def grid_facts(experiment: dict[str, Any]) -> dict[str, Any]:
    """Where a run's grids sit: the projection and each domain's size and
    place in its parent, the numbers the map draws the grids from."""

    domains = []
    for domain in _rows(experiment.get("domains")):
        if not isinstance(domain, dict):
            continue
        run = _object(domain.get("run"))
        dx = _positive(run.get("dx"))
        domains.append({
            "grid_id": _integer(domain.get("grid_id")), "parent_id": _integer(domain.get("parent_id")),
            "nx": _integer(run.get("nx")), "ny": _integer(run.get("ny")), "nz": _integer(run.get("nz")),
            "dx_km": None if dx is None else dx / 1000.0,
            "i_parent_start": _integer(domain.get("i_parent_start")),
            "j_parent_start": _integer(domain.get("j_parent_start")),
            "parent_grid_ratio": _integer(domain.get("parent_grid_ratio")),
        })
    projection = experiment.get("projection")
    return {"projection": projection if isinstance(projection, dict) else None, "domains": domains}


_CACHE: dict[str, tuple[tuple[int, float], dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()


def event_facts(path: Path) -> dict[str, Any]:
    """What the event log says: the plan's length and start, the latest
    progress, the stage, the end, and the pictures published early."""

    try:
        stat = path.stat()
        key = (stat.st_size, stat.st_mtime)
    except OSError:
        return {}
    with _CACHE_LOCK:
        hit = _CACHE.get(str(path))
        if hit is not None and hit[0] == key:
            return hit[1]
    facts: dict[str, Any] = {"outputs_committed": 0, "first_pictures": []}
    warnings: set[str] = set()
    for record in event_records(path):
        tag = record.get("event")
        facts["last_sequence"] = record.get("sequence", facts.get("last_sequence"))
        if tag != "warning" or record.get("code") != KERNEL_COMPILE:
            # "Compiling GPU kernels" holds only while nothing else has been said since.
            facts["compiling"] = None
        if tag == "plan_accepted":
            # A resumed run appends a new attempt to the same stream: its state starts from this attempt, not
            # from the end, progress and grids of the one before it (a resumed run read Finished while it ran).
            facts = {"outputs_committed": 0, "first_pictures": [], "last_sequence": facts.get("last_sequence")}
            warnings = set()
        elif tag == "resolved_plan":
            configuration = record.get("configuration")
            child = (downscale_facts(path.parent)
                     if configuration is None and isinstance(record.get("config_source"), str) else None)
            if child is not None:
                # A downscale names its configuration file instead of carrying it: its grid and levels are in the
                # plan ``gpuwm downscale`` wrote, and it is placed in its parent's grid (downscale_grid), so it
                # has no grid of its own here.
                facts.update(child)
                facts["grid"] = {}
                continue
            experiment = configuration.get("experiment") if isinstance(configuration, dict) else None
            if not isinstance(experiment, dict) or not isinstance(experiment.get("domains", []), list):
                warnings.add(GRID_WARNING)
            experiment = _object(experiment)
            facts["run_seconds"] = _measured(experiment.get("run_seconds"), warnings, positive=True)
            facts["start_time"] = experiment.get("start_time") if isinstance(experiment.get("start_time"), str) else None
            facts["name"] = experiment.get("name")
            domains = _rows(experiment.get("domains"))
            facts["domains"] = len(domains)
            facts["grids"] = _grids(domains)
            facts["grid"] = grid_facts(experiment)
            facts["levels"] = sorted({d["nz"] for d in facts["grid"]["domains"] if d["nz"] is not None and d["nz"] > 0})
        elif tag == "stage_started":
            facts["stage"] = record.get("stage")
            # The step a stage is on, read from the stage's own records (see preparation()).
            phase = record.get("phase")
            facts["stage_phase"] = phase if isinstance(phase, str) else None
            facts["stage_since_ms"] = _integer(record.get("emitted_unix_ms"))
            facts["prep_step"] = None
        elif tag in TRANSFER_TAGS:
            _fetch_fact(facts.setdefault("fetch", {"files": {}, "acquisition": {}}), record)
        elif tag == "warning" and record.get("code") == PREPARATION_PROGRESS:
            _prep_fact(facts, record.get("preparation"), _integer(record.get("emitted_unix_ms")))
        elif tag == "warning" and record.get("code") == KERNEL_COMPILE:
            modules = _integer(record.get("modules_compiled"))
            facts["compiling"] = modules if modules is not None and modules > 0 else 1
        elif tag == "prepare_head_ready":
            facts["head_ready"] = True
        elif tag == "prepare_sealed":
            facts["sealed"] = True
        elif tag == "model_progress":
            facts["model_seconds"] = _measured(record.get("model_seconds"), warnings)
            facts["speed_x"] = _measured(record.get("speed_x"), warnings, positive=True)
            facts["wall_seconds"] = _measured(record.get("wall_seconds"), warnings)
            walls = {}
            for row in _rows(record.get("domains")):
                if not isinstance(row, dict):
                    continue
                domain, seconds = _integer(row.get("domain")), _nonnegative(row.get("step_wall_seconds"))
                if domain is not None and seconds is not None:
                    walls[domain] = seconds
            if walls:
                facts["step_walls"] = walls
        elif tag == "output_committed":
            facts["outputs_committed"] += 1
            facts["last_output_time"] = record.get("valid_time")
        elif tag == "first_products_ready":
            facts["first_pictures"] = [item for item in _rows(record.get("paths")) if isinstance(item, str)]
        elif tag == "warning" and record.get("code") == BASEMAP_MISSING:
            facts["basemap_missing"] = True
        elif tag == "warning" and record.get("code") == LIBRARY_WARNING:
            # The run goes on as configured and the warning stays with this attempt, so the map says it while the
            # run goes and after it ends (a warm bubble above 10 K runs, and says what that does to the start).
            said = {"message": plain_message(record.get("message")), "detail": plain_message(record.get("detail"))}
            if said["message"] and said not in facts.setdefault("library_warnings", []):
                facts["library_warnings"].append(said)
        elif tag == "posting_schedule":
            # A window fetched as its source posts: when the run can start, when the cycle's last hour is expected,
            # and how late an hour may be.
            facts["posting"] = {key: record.get(key) for key in POSTING_SCHEDULE_KEYS}
            facts["posting"]["leads"] = len(_rows(record.get("leads")))
            facts["posting"]["leads_ready"] = 0
        elif tag == "lead_ready" and isinstance(facts.get("posting"), dict):
            facts["posting"]["leads_ready"] += 1
            facts["posting"]["last_ready_lead"] = record.get("lead")
        elif tag in ("source_wait_started", "source_wait_progress", "boundary_wait_started"):
            # The wait the forecast is in, as the run said it: the heartbeat's wait record names its cause and
            # lead, and this adds where the model stands (see wait()).
            facts["wait"] = {"on": "preparation" if tag == "boundary_wait_started" else "source",
                             **{key: record.get(key) for key in WAIT_EVENT_KEYS if key in record}}
        elif tag in ("source_wait_finished", "boundary_wait_finished"):
            facts["wait"] = None
        elif tag == "source_behind":
            # The lead a run stopped on because its source fell behind its budget (exit 75), and where it stopped.
            facts["source_behind"] = {key: record.get(key) for key in (
                "source", "cycle", "lead", "valid_time", "expected_at", "late_at", "late_after_minutes",
                "last_answer", "model_elapsed_seconds", "model_valid_time", "frames_kept")}
        elif tag in ("completed", "failed"):
            facts["end"] = {key: record.get(key) for key in (
                "event", "stage", "message", "remedy", "interrupted", "exit_code", "dry_run")}
            # The page shows no machine path; the engine's whole text stays in the event log (files.plain_message).
            # The folders a refusal names as the place to act stay in its words.
            folders = [str(folder) for folder in _rows(record.get("folders")) if isinstance(folder, str)]
            facts["end"]["message"] = plain_message(facts["end"]["message"], folders)
            facts["end"]["remedy"] = plain_message(facts["end"]["remedy"], folders, paragraph=True)
    if warnings:
        facts["metadata_warnings"] = sorted(warnings)
    with _CACHE_LOCK:
        _CACHE[str(path)] = (key, facts)
    return facts


#: The fields of a ``posting_schedule`` event the run page keeps (``gpuwm.runplan.POSTING_EVENT_FIELDS``).
POSTING_SCHEDULE_KEYS = ("source", "member", "cycle", "as_posted", "shape", "streams", "why", "late_after_minutes",
                         "expected_ready_at", "expected_final_at")
#: The fields of a wait event (``source_wait_*`` or ``boundary_wait_started``) the run page keeps.
WAIT_EVENT_KEYS = ("phase", "source", "cycle", "lead", "valid_time", "expected_at", "late_at", "interval", "reason",
                   "model_elapsed_seconds", "model_valid_time")


def wait(heartbeat: dict[str, Any], facts: dict[str, Any], start: datetime | None,
         now: datetime | None = None) -> dict[str, Any]:
    """A running forecast's wait at a seam: what it waits on, since when, and where the model stands.

    The heartbeat's own wait record (``on``, ``lead``, ``expected_at``,
    ``late_at``, ``since_utc``) is the wait now.  The run's wait event of
    the same cause (and, on a source, the same lead) adds the model time
    reached, the boundary interval, the source's valid time and the
    reason; without one, the model time is the heartbeat's own, and its
    valid time is counted from the run's start, written as the engine
    writes it (``2026-09-24T01:30:00Z``).
    """

    block = dict(heartbeat["wait"])
    said = facts.get("wait") if isinstance(facts.get("wait"), dict) else {}
    if said.get("on") == block.get("on") and (block.get("on") != "source" or said.get("lead") == block.get("lead")):
        for key in WAIT_EVENT_KEYS:
            if block.get(key) is None and said.get(key) is not None:
                block[key] = said[key]
    if block.get("model_elapsed_seconds") is None:
        block["model_elapsed_seconds"] = _nonnegative(heartbeat.get("model_elapsed_seconds"))
    elapsed = _nonnegative(block.get("model_elapsed_seconds"))
    if block.get("model_valid_time") is None and start is not None and elapsed is not None:
        try:
            block["model_valid_time"] = (start + timedelta(seconds=elapsed)).astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
        except OverflowError:
            pass
    since = _parse_time(block.get("since_utc"))
    moment = now or datetime.now(timezone.utc)
    if since is not None:
        since = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
        block["waited_seconds"] = max(0.0, round((moment - since).total_seconds(), 1))
    return block


# ------------------------------------------------------------------ preparation

#: The engine's transfer records (``gpuwm.progress.TRANSFER_EVENTS``), spelled here because this module stays off
#: the fetch stack.
TRANSFER_TAGS = ("fetch_started", "fetch_progress", "fetch_completed")
#: The warning code a preparation step's record rides on (``gpuwm.progress.prep_stage`` and ``prep_progress``, and
#: an out-of-process stage's published progress file, relayed by ``gpuwm.runplan``).
PREPARATION_PROGRESS = "preparation_progress"
#: The warning code of one GPU module compiled for this card (``gpuwm.kernel_compile_notice``).
KERNEL_COMPILE = "kernel_compile_progress"
PREP_STAGE_SCHEMA = "gpuwm.prep-stage.v1"
PREPARE_PROGRESS_SCHEMA = "gpuwm.prepare-progress/v1"
ACQUISITION_SCHEMA = "arwen.acquisition-progress.v1"
#: The acquisition figures a page reads, each a whole count.
ACQUISITION_COUNTS = ("files_total", "files_completed", "transferred_bytes", "bytes_available",
                      "forcing_times_total", "forcing_times_completed")


def _count(value: Any) -> int | None:
    value = _integer(value)
    return value if value is not None and value >= 0 else None


def _word(value: Any) -> str | None:
    return value if isinstance(value, str) and 0 < len(value) <= 200 else None


def _fetch_fact(fetch: dict[str, Any], record: dict[str, Any]) -> None:
    """One transfer record folded into the fetch's files and its whole-request count."""

    supplied = record.get("acquisition")
    if isinstance(supplied, dict) and supplied.get("schema") == ACQUISITION_SCHEMA:
        for key in ACQUISITION_COUNTS:
            value = _count(supplied.get(key))
            if value is not None:
                fetch["acquisition"][key] = value
    name = record.get("file")
    if not isinstance(name, str) or not name:
        return
    item = fetch["files"].setdefault(name, {"bytes": 0, "expected": None, "done": False, "failed": False})
    if record.get("event") == "fetch_started":
        item.update(bytes=0, done=False, failed=False)
    moved, expected = _count(record.get("bytes")), _count(record.get("expected_bytes"))
    if moved is not None:
        item["bytes"] = moved
    if expected:
        item["expected"] = expected
    if record.get("event") == "fetch_completed":
        item["done"] = record.get("failed") is False
        item["failed"] = record.get("failed") is not False


def _prep_fact(facts: dict[str, Any], supplied: Any, emitted_ms: int | None = None) -> None:
    """One preparation step record: the step now open, when it opened, and how far a counted step has got.

    ``since_ms`` is the step's own start, so the page times the step it names rather than the stage around it; a
    count heard without its step's start has none.
    """

    if not isinstance(supplied, dict):
        return
    schema = supplied.get("schema")
    open_step = facts.get("prep_step") or {}
    if schema == PREP_STAGE_SCHEMA:
        key, action = _word(supplied.get("stage")), supplied.get("event")
        index, count = _count(supplied.get("index")), _count(supplied.get("count"))
        step = {"key": key, "label": _word(supplied.get("label")), "index": index, "count": count,
                "grid_id": _count(supplied.get("grid_id")), "since_ms": None}
        if action == "started":
            facts["prep_step"] = {**step, "since_ms": emitted_ms}
        elif action == "progress":
            facts["prep_step"] = {**step, "index": None, "done": index,
                                  "since_ms": open_step.get("since_ms") if open_step.get("key") == key else None}
            if key == "root_boundaries" and index is not None and count:
                facts["boundaries"] = {"done": index, "count": count}
        elif action in ("finished", "failed"):
            if (facts.get("prep_step") or {}).get("key") == key:
                facts["prep_step"] = None
            if key == "root_boundaries" and action == "finished" and count:
                facts["boundaries"] = {"done": count, "count": count}
    elif schema == PREPARE_PROGRESS_SCHEMA:
        # An out-of-process preparation's own progress file: its phase, numbered from zero.
        index, total = _count(supplied.get("phase_index")), _count(supplied.get("phases_total"))
        key = _word(supplied.get("phase"))
        facts["prep_step"] = {"key": key, "label": None, "index": None, "count": None,
                              "phase_number": None if index is None else index + 1, "phases": total,
                              "since_ms": open_step.get("since_ms") if open_step.get("key") == key else emitted_ms}


def _fetch_progress(fetch: dict[str, Any]) -> dict[str, Any] | None:
    """Files and bytes of the download so far, against the totals when the request stated them."""

    files = list(fetch.get("files", {}).values())
    whole = fetch.get("acquisition", {})
    if not files and not whole:
        return None
    moved = sum(item["bytes"] for item in files)
    total_files = whole.get("files_total")
    if total_files is not None and total_files < len(files):
        total_files = None
    declared = [item["expected"] for item in files]
    # A byte total only when every file declared its size and no file is still to be declared.
    complete = bool(files) and all(declared) and (total_files is None or total_files == len(files))
    return {
        "files_done": sum(1 for item in files if item["done"]) if files else whole.get("files_completed"),
        "files_total": total_files,
        "files_failed": sum(1 for item in files if item["failed"]),
        "bytes": max(moved, whole.get("transferred_bytes") or 0) or None,
        "bytes_total": sum(declared) if complete else None,
        "times_done": whole.get("forcing_times_completed"),
        "times_total": whole.get("forcing_times_total"),
    }


#: The heartbeat's word for a preparation phase (``gpuwm.runtime._preparation_progress``).
PREPARING = "preparing:"


def preparation(facts: dict[str, Any], heartbeat: Any, *, now_ms: int | None = None) -> dict[str, Any] | None:
    """Where a running forecast's preparation is, from the records its engine wrote.

    The stage, the step inside it (by its engine key, and the grid for a step
    taken grid by grid), the download's files and bytes, how many boundary
    times are ready (a chained forecast steps beside the rest of its
    preparation), whether GPU kernels are compiling, and how long the stage
    and the step it names have been open.  Before this a run page said
    "preparing" and nothing else from the download to the first model step.  ``None`` once the forecast
    steps with nothing left to prepare and no kernel compiling.
    """

    stage = facts.get("stage")
    chained = bool(facts.get("head_ready")) and not facts.get("sealed")
    # The first model step of a card's first forecast compiles its kernels for up to a minute; the page says so.
    compiling = stage == "forecast" and bool(facts.get("compiling"))
    if stage in (None, "forecast", "finalize") and not chained and not compiling:
        return None
    phase = facts.get("stage_phase")
    status = heartbeat.get("status") if isinstance(heartbeat, dict) else None
    if isinstance(status, str) and status.startswith(PREPARING) and stage in ("prepare", "initialize"):
        phase = status[len(PREPARING):] or phase
    step = dict(facts.get("prep_step") or {}) or None
    step_since = None if step is None else step.pop("since_ms", None)
    if step is not None and step.get("key") == "domain_initialize" and step.get("index") is not None:
        grids = {grid["domain"]: grid["dx_km"] for grid in facts.get("grids") or []}
        # By the grid's id: the index is its place among the grids that start with the run, which a dormant nest
        # ahead of it makes differ from its id.  A record with no grid_id numbered its grids by id.
        grid = step.get("grid_id")
        step["grid_km"] = grids.get(step["index"] if grid is None else grid)
    since = facts.get("stage_since_ms")
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000) if now_ms is None else now_ms
    fetch = _fetch_progress(facts["fetch"]) if stage == "fetch" and facts.get("fetch") else None
    return {
        "stage": stage,
        "phase": _word(phase),
        "step": step,
        "fetch": fetch,
        "compiling": facts.get("compiling"),
        "boundaries": facts.get("boundaries"),
        "chained": chained,
        # Whole seconds, so the line moves while a step that says nothing finer is still going: the stage's time,
        # and the named step's own when its start was heard, which the line puts after that step.
        "stage_seconds": None if since is None or stage == "forecast" else max(0, (now_ms - since) // 1000),
        "step_seconds": None if step_since is None or stage == "forecast" else max(0, (now_ms - step_since) // 1000),
    }


def _grids(domains: list[Any]) -> list[dict[str, Any]]:
    """Each grid of the resolved plan: its id and its spacing in km."""

    out = []
    for domain in domains:
        if not isinstance(domain, dict):
            continue
        dx = _positive(_object(domain.get("run")).get("dx"))
        if dx is None:
            continue
        grid_id = domain.get("grid_id")
        grid_id = len(out) + 1 if grid_id is None else _integer(grid_id)
        if grid_id is None:
            continue
        out.append({"domain": grid_id, "dx_km": round(dx / 1000.0, 3)})
    return out


def km_text(dx_km: float) -> str:
    """``12``, ``3``, ``1``, ``0.75``: a grid spacing without trailing zeros."""

    return f"{dx_km:.3f}".rstrip("0").rstrip(".")


def pace(facts: dict[str, Any]) -> dict[str, Any] | None:
    """Every grid of the run and the one whose steps take the most wall.

    ``pace_km`` is ``None`` until the run has measured its grids (the
    engine sums each grid's own step wall), and on a one-grid run, where
    the one grid is trivially the pace.
    """

    grids = facts.get("grids") or []
    if not grids:
        return None
    walls = facts.get("step_walls") or {}
    slowest = None
    if len(grids) > 1 and walls:
        domain = max(walls, key=walls.get)
        slowest = next((grid for grid in grids if grid["domain"] == domain), None)
    return {
        "grids_km": [grid["dx_km"] for grid in grids],
        "grids_label": " / ".join(km_text(grid["dx_km"]) for grid in grids) + " km",
        "pace_km": None if slowest is None else slowest["dx_km"],
        "pace_label": None if slowest is None else f"{km_text(slowest['dx_km'])} km",
    }


# ------------------------------------------------------------------ state

def _parse_time(text: Any) -> datetime | None:
    if not text:
        return None
    try:
        value = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def number(value: Any) -> float | None:
    """A finite number a run's own file wrote, or None: a string, a list or a true/false is not one."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except OverflowError:  # an integer past any float
        return None
    return value if math.isfinite(value) else None


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _rows(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _nonnegative(value: Any) -> float | None:
    value = number(value)
    return value if value is not None and value >= 0 else None


def _positive(value: Any) -> float | None:
    value = number(value)
    return value if value is not None and value > 0 else None


def _integer(value: Any) -> int | None:
    value = number(value)
    return int(value) if value is not None and value.is_integer() else None


#: What a run's row says when a record it was written with holds a value of the wrong kind.
GRID_WARNING = "Some of this run's grid details could not be read, so they are not shown."
SETTINGS_WARNING = "Some of this run's saved settings could not be read, so they are not shown."
PROGRESS_WARNING = "Some of this run's progress figures could not be read, so they are not shown."


def _measured(value: Any, warnings: set[str], *, positive: bool = False) -> float | None:
    """A measurement from a record: a number, or None and a warning when one was written and is not one."""

    if value is None:
        return None
    found = _positive(value) if positive else _nonnegative(value)
    if found is None:
        warnings.add(PROGRESS_WARNING)
    return found


def plan_intent(plan: Any) -> dict[str, Any]:
    """``plan.json``'s ``config.intent``, or ``{}`` when the file is not shaped that way.

    A run folder is written by whoever wrote it (this page, a terminal, a copy
    from another machine), so every level is checked for its type; one
    damaged plan read ``{"config": "invalid"}`` and took My forecasts down.
    """

    config = plan.get("config") if isinstance(plan, dict) else None
    intent = config.get("intent") if isinstance(config, dict) else None
    return intent if isinstance(intent, dict) else {}


def _plan_facts(rundir: Path) -> dict[str, Any]:
    plan = read_json(rundir / PLAN, default=None)
    intent = plan_intent(plan)
    warnings: set[str] = set()
    facts: dict[str, Any] = {}
    config = plan.get("config") if isinstance(plan, dict) else None
    if (rundir / PLAN).is_file() and (not isinstance(config, dict)
                                      or ("intent" in config and not isinstance(config["intent"], dict))):
        warnings.add(SETTINGS_WARNING)
    if intent:
        hours = _positive(intent.get("hours"))
        if hours is not None and hours * 3600.0 < MAX_SECONDS:
            facts["run_seconds"] = hours * 3600.0
        elif intent.get("hours") is not None:
            warnings.add(SETTINGS_WARNING)
        cycle = _parse_time(intent.get("cycle")) if isinstance(intent.get("cycle"), str) else None
        lead = intent.get("forecast_start_hour")
        lead = 0 if lead is None else _integer(lead)
        if cycle is not None and lead is not None and 0 <= lead <= MAX_LEAD_H:
            facts["start_time"] = (cycle + timedelta(hours=lead)).isoformat()
        elif intent.get("cycle") is not None:
            # A start that cannot be read is left unknown; the cycle alone is not the forecast's start.
            warnings.add(SETTINGS_WARNING)
        for key in ("root_dx_km", "nz"):
            if key in intent and _positive(intent.get(key)) is None:
                warnings.add(SETTINGS_WARNING)
        source = intent.get("source")
        facts["source"] = source if isinstance(source, str) else None
    child = downscale_plan(rundir)
    if child is not None:
        # a downscale's plan holds its length and its start (the parent history it starts from)
        seconds = _positive(_object(child.get("child_grid")).get("run_seconds"))
        if seconds is not None and seconds < MAX_SECONDS:
            facts.setdefault("run_seconds", seconds)
        condition = child.get("initial_condition")
        begun = condition.get("GPUWM_INITIAL_CONDITION_MODEL_START_DATE") if isinstance(condition, dict) else None
        moment = _parse_time(str(begun).replace("_", "T")) if begun else None
        if moment is not None:
            facts.setdefault("start_time", moment.isoformat())
    if warnings:
        facts["metadata_warnings"] = sorted(warnings)
    return facts


#: Longer than any forecast a run folder can hold (a year), and far inside what a date can add.
MAX_SECONDS = 366 * 86400.0
MAX_LEAD_H = 24 * 366


# ------------------------------------------------------------------ downscale

#: ``gpuwm downscale`` writes its plan into the child's folder and beside it (``<child>.downscale-plan.json``).
DOWNSCALE_PLAN = "downscale-plan.json"
DOWNSCALE_SCHEMA = "gpuwm.downscale-plan."


def downscale_plan(rundir: Path) -> dict[str, Any] | None:
    for path in (rundir / DOWNSCALE_PLAN, rundir.with_name(f"{rundir.name}.{DOWNSCALE_PLAN}")):
        document = read_json(path, default=None)
        if isinstance(document, dict) and str(document.get("schema", "")).startswith(DOWNSCALE_SCHEMA):
            return document
    return None


def downscale_facts(rundir: Path) -> dict[str, Any] | None:
    """A downscale's grid as its plan records it: ``grids`` (its id and spacing) and ``levels``, or None when the
    plan is missing or holds a value of the wrong kind."""

    plan = downscale_plan(rundir)
    if plan is None:
        return None
    child = _object(plan.get("child_grid"))
    grid_id, dx, nz = _integer(plan.get("child_grid_id")), _positive(child.get("dx")), _integer(child.get("nz"))
    if grid_id is None or grid_id <= 0 or dx is None or nz is None or nz <= 0:
        return None
    return {"grids": [{"domain": grid_id, "dx_km": round(dx / 1000.0, 3)}], "levels": [nz]}


def _parent_run(root: Path, recorded: str) -> Path | None:
    """The parent run's folder under ``root``, found by the names in the path the plan recorded
    (the plan keeps the path the parent had when the child was made; a copied folder keeps the names)."""

    parts = [part for part in recorded.replace("\\", "/").split("/") if part]
    for k in range(len(parts)):
        for depth in (1, 2):
            names = parts[k:k + depth]
            if len(names) < depth or not all(folder_part(name) for name in names):
                continue
            candidate = root.joinpath(*names)
            if within(root, candidate) and is_run_dir(candidate):
                return candidate
    return None


def downscale_grid(root: Path, rundir: Path) -> dict[str, Any] | None:
    """Where a downscale's grid sits: its parent run's projection and the parent grid's chain (marked
    ``context``, drawn by no one), then the child placed in its parent as the plan records it."""

    plan = downscale_plan(rundir)
    if plan is None:
        return None
    parent_dir = _parent_run(root, str((plan.get("parent") or {}).get("run_dir") or ""))
    events = parent_dir / EVENTS if parent_dir is not None else None
    grid = (event_facts(events).get("grid") or {}) if events is not None and events.is_file() else {}
    by_id = {d.get("grid_id"): d for d in grid.get("domains") or []}
    child = plan.get("child_grid") or {}
    place = plan.get("placement") or {}
    try:
        parent_id = int(plan.get("parent_domain") or (plan.get("parent") or {}).get("domain"))
        child_id = int(plan.get("child_grid_id"))
        child_row = {"grid_id": child_id, "parent_id": parent_id, "nx": int(child["nx"]), "ny": int(child["ny"]),
                     "nz": child.get("nz"), "dx_km": float(child["dx"]) / 1000.0,
                     "i_parent_start": int(place["i_parent_start"]), "j_parent_start": int(place["j_parent_start"]),
                     "parent_grid_ratio": int(place["ratio"])}
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    chain, at = [], by_id.get(parent_id)
    while at is not None and at not in chain:
        chain.append(at)
        at = by_id.get(at.get("parent_id")) if at.get("parent_id") != at.get("grid_id") else None
    if not grid.get("projection") or not chain or child_id in {row.get("grid_id") for row in chain}:
        return None
    ids = {row.get("grid_id") for row in chain}
    # the parent grid stood where it was when the child's start was written; later moves are not the child's
    begun = _plan_facts(rundir).get("start_time")
    start = _parse_time(begun)
    moves = [row for row in nest_moves(parent_dir) if row.get("grid_id") in ids
             and (start is None or (_parse_time(row["valid"]) or start) < start)]
    return {"projection": grid["projection"], "moves": moves,
            "domains": [{**row, "context": True} for row in reversed(chain)] + [child_row]}


def job_of(rundir: Path) -> dict[str, Any] | None:
    """The page's record of the job that runs this folder: ``gui-job.json``, or a page-started downscale's."""

    job = read_json(rundir / JOB, default=None)
    if not isinstance(job, dict):
        job = (start_record(rundir) or {}).get("job")
    return job if isinstance(job, dict) else None


def job_alive(rundir: Path) -> bool:
    job = job_of(rundir)
    if job is None:
        return False
    result = Path(str(job.get("jobs_dir") or "")) / "result.json"
    if job.get("jobs_dir") and result.is_file():
        return False
    return process_alive(job.get("wrapper_process"), job.get("wrapper_pid"))


def follow_needed(remote: Any, render: Any) -> bool:
    """A machine's forecast that has not ended, or a render worker still drawing: the follower has work here.

    The follower (:mod:`.remote_runs`) keeps going exactly while this holds, and the page says its updates
    stopped when it does not and no follower is alive.
    """

    forecast_open = isinstance(remote, dict) and not remote.get("ended")
    render_open = isinstance(render, dict) and render.get("state") not in ("finished", "failed")
    return forecast_open or render_open


def follower_alive(rundir: Path) -> bool:
    """The follower recorded in ``gui-follow.json`` is still that process and has not written its result."""

    document = read_json(rundir / FOLLOW, default=None)
    if not isinstance(document, dict):
        return False
    result = Path(str(document.get("jobs_dir") or "")) / "result.json"
    if document.get("jobs_dir") and result.is_file():
        return False
    return process_alive(document.get("wrapper_process"), document.get("wrapper_pid"))


#: How much of a job's stderr is read, from each end, for the line that says why it ended (the list asks for
#: every run's state every few seconds, so this stays small).
STDERR_READ = 64 * 1024


def _stderr_ends(path: Path) -> tuple[str, str]:
    """The first and the last :data:`STDERR_READ` bytes of a job's stderr, as text."""

    try:
        with open(path, "rb") as stream:
            head = stream.read(STDERR_READ)
            size = stream.seek(0, 2)
            stream.seek(max(0, size - STDERR_READ))
            tail = stream.read(STDERR_READ)
    except OSError:
        return "", ""
    return head.decode("utf-8", "replace"), tail.decode("utf-8", "replace")


def _plain(text: str) -> str | None:
    # "gpuwm run-plan: REFUSING: ..." -- the command's name is not news on the page.
    return plain_message(re.sub(r"^gpuwm[\w -]*:\s+", "", text.strip()))


def _why_it_ended(head: str, tail: str, *, began: bool) -> str | None:
    """Why a job ended, as one plain line: never the version banner, a traceback frame or a machine path.

    A traceback's reason is its last line without the exception's class.
    Without one, a start the engine refused says why on its first line
    after the banner; a run that had begun has no such line (its stderr
    is its log), so nothing is guessed from it.
    """

    from gpuwm.mcp.doors import _is_banner_or_pointer

    lines = [line.rstrip() for line in tail.splitlines() if not _is_banner_or_pointer(line)]
    if any(line.startswith("Traceback") for line in lines):
        after = lines[max(i for i, line in enumerate(lines) if line.startswith("Traceback")):]
        reasons = [line for line in after if not line.startswith((" ", "\t", "Traceback"))]
        if reasons:
            return _plain(re.sub(r"^[A-Za-z_.]*(?:Error|Exception|Exit|Interrupt):\s*", "", reasons[-1]))
        return None
    if began:
        return None
    first = next((line for line in head.splitlines() if not _is_banner_or_pointer(line)), "")
    return _plain(first) if first.strip() else None


def job_end(rundir: Path, stage: Any = None, *, began: bool = False) -> dict[str, Any] | None:
    """The end a finished job implies when the engine wrote none: Failed or Stopped, or None.

    Only a job that ended without success (a nonzero exit, or cancelled)
    counts; a clean exit with no end event is left to the other signs.
    ``began`` says the engine had written its manifest or its first
    event, so the job did not end at its start.
    """

    job = job_of(rundir)
    if not isinstance(job, dict) or not job.get("jobs_dir"):
        return None
    jobs_dir = Path(str(job["jobs_dir"]))
    result = read_json(jobs_dir / "result.json", default=None)
    if not isinstance(result, dict):
        return None
    code = result.get("exit_code")
    cancelled = bool(result.get("cancelled"))
    if not cancelled and code == 0:
        return None
    why = None if cancelled else _why_it_ended(*_stderr_ends(jobs_dir / "stderr.log"), began=began)
    if cancelled:
        message = "Stopped." if began else "Stopped before the forecast wrote its first record."
    elif why:
        message = why
    elif began:
        message = (f"The forecast ended (exit code {code}) without writing its last record."
                   if code is not None else "The forecast ended without writing its last record.")
    else:
        message = (f"The forecast ended at its start (exit code {code}) before it wrote its first record."
                   if code is not None else "The forecast ended before it wrote its first record.")
    return {"event": "failed", "stage": stage or "start", "message": message, "remedy": None,
            "interrupted": cancelled, "exit_code": code, "dry_run": False}


def status(rundir: Path) -> dict[str, Any]:
    """The whole derived state of one run folder."""

    remote = read_json(rundir / REMOTE, default=None)
    remote = remote if isinstance(remote, dict) else None
    manifest = read_json(rundir / (REMOTE_MANIFEST if remote else MANIFEST), default=None)
    heartbeat = read_json(rundir / HEARTBEAT, default=None)
    events = rundir / EVENTS
    facts = event_facts(events) if events.is_file() else {}
    planned = _plan_facts(rundir)
    stop = read_json(rundir / STOP, default=None)
    pid = (manifest or {}).get("pid") if isinstance(manifest, dict) else None
    if remote is not None:
        # The pid is the other machine's: only what the machine said counts.
        alive = bool(remote.get("alive")) and not remote.get("ended")
        job = remote.get("job") if isinstance(remote.get("job"), dict) else {}
        ended_there = remote.get("ended") or job.get("state") in REMOTE_END_STATES
        if not facts.get("end") and ended_there and not alive:
            # The machine's job ended without an end event (stopped while it
            # waited for the card, refused, or killed): the job's own state
            # is the end, so the run never reads as ready to start here.
            stopped = job.get("state") == "stopped" or bool(stop)
            default = ("Stopped before it started on the machine." if stopped
                       else "The run did not start on the machine.")
            facts = {**facts, "end": {"event": "failed", "stage": facts.get("stage") or "start",
                                      "message": job.get("message") or default,
                                      "remedy": None, "interrupted": stopped,
                                      "exit_code": job.get("exit_code"), "dry_run": False}}
    else:
        alive = manifest_alive(manifest) or job_alive(rundir)
        if not facts.get("end") and not alive and not (rundir / QUEUED).is_file():
            record = start_record(rundir)
            if record is not None and manifest is None and not events.is_file():
                # A downscale the page started that ended before it wrote a receipt of its own (a refusal at
                # admission): its job's log is the only record of why.
                facts = {**facts, "end": start_end(record, bool(stop))}
            else:
                ended = job_end(rundir, facts.get("stage"), began=manifest is not None or events.is_file())
                if ended is not None:
                    facts = {**facts, "end": ended}

    end = facts.get("end")
    if end is not None:
        if end.get("event") == "completed":
            state = "finished"
        elif end.get("interrupted") or stop:
            state = "stopped"
        else:
            state = "failed"
    elif alive:
        state = "running"
    elif manifest is not None or events.is_file():
        state = "stopped" if stop else "stale"
    elif remote is not None:
        # A mirror of another machine's run is never "ready" here: its
        # plan.json holds that machine's paths, not this computer's.
        state = "stopped" if stop else "stale"
    elif (rundir / QUEUED).is_file():
        state = "queued"
    elif (rundir / PLAN).is_file():
        state = "ready"
    else:
        state = "imported"

    warnings = set(facts.get("metadata_warnings") or []) | set(planned.get("metadata_warnings") or [])
    run_seconds = _positive(facts.get("run_seconds")) or _positive(planned.get("run_seconds"))
    if run_seconds is not None and run_seconds >= MAX_SECONDS:
        run_seconds = None
        warnings.add(PROGRESS_WARNING)
    model_seconds = _nonnegative(facts.get("model_seconds"))
    if isinstance(heartbeat, dict) and heartbeat.get("model_elapsed_seconds") is not None:
        # A heartbeat whose figure is not a time is passed over; the event log's progress still counts.
        elapsed = _nonnegative(heartbeat.get("model_elapsed_seconds"))
        if elapsed is not None:
            model_seconds = max(model_seconds or 0.0, elapsed)
    if model_seconds is not None and model_seconds >= MAX_SECONDS:
        model_seconds = None
        warnings.add(PROGRESS_WARNING)
    percent = None
    if run_seconds and model_seconds is not None:
        percent = max(0.0, min(100.0, 100.0 * model_seconds / run_seconds))
    if state == "finished" and not (end or {}).get("dry_run"):
        percent = 100.0
    start = _parse_time(facts.get("start_time") or planned.get("start_time"))
    storm_time = None
    if start is not None:
        try:
            storm_time = utc_text(start + timedelta(seconds=model_seconds or 0.0))
        except OverflowError:
            warnings.add(PROGRESS_WARNING)
    speed = _positive(facts.get("speed_x"))
    left = None
    if state == "running" and speed and run_seconds and model_seconds is not None:
        left = max(0.0, (run_seconds - model_seconds) / speed)
    phase = heartbeat.get("status") if isinstance(heartbeat, dict) else None
    if remote is not None and state == "running" and (remote.get("job") or {}).get("state") == "waiting-for-card":
        phase = "waiting for the card: " + str((remote.get("job") or {}).get("waiting_because") or "")
    queued = read_json(rundir / QUEUED, default=None) if state == "queued" else None
    # A machine's run is seen here only through its follower. With the follower gone while the run or its
    # drawing is still open there, the state above is the last one received, not the machine's state now.
    follow_lost = follow_needed(remote, read_json(rundir / RENDER, default=None)) and not follower_alive(rundir)
    return {
        "state": state,
        "queue": None if not isinstance(queued, dict) else {
            key: queued.get(key) for key in ("machine", "queued_utc", "held", "waiting", "need_gib")},
        "stage": facts.get("stage"),
        "phase": phase,
        "pid": pid,
        "alive": bool(alive),
        "run_seconds": run_seconds,
        "model_seconds": model_seconds,
        "percent": None if percent is None else round(percent, 1),
        "start_time": None if start is None else utc_text(start),
        "storm_time": storm_time,
        "seconds_left": None if left is None else round(left),
        "speed_x": speed,
        "outputs_committed": facts.get("outputs_committed", 0),
        # The pictures are drawn with no coastlines, borders or state lines.
        "basemap_missing": bool(facts.get("basemap_missing")),
        # What the libraries this attempt drives warned about, each {message, detail}, in the order they said it.
        "library_warnings": list(facts.get("library_warnings") or []),
        "end": end,
        "stop": stop if isinstance(stop, dict) else None,
        "source": planned.get("source"),
        "updated_utc": (heartbeat or {}).get("updated_at_utc") if isinstance(heartbeat, dict) else None,
        "last_sequence": facts.get("last_sequence"),
        "machine": None if remote is None else remote.get("machine"),
        "machine_checked_utc": None if remote is None else remote.get("checked_utc"),
        "follow_lost": follow_lost,
        "grids": pace(facts),
        "preparation": preparation(facts, heartbeat) if state == "running" else None,
        # A running forecast waiting at a seam: on the source (a lead not posted yet, with the lead, when it was
        # expected and when it counts as late) or on the preparation, from the heartbeat's own wait record, with
        # the model time it waits at (wait()).
        "wait": (wait(heartbeat, facts, start) if state == "running" and isinstance(heartbeat, dict)
                 and str(heartbeat.get("status") or "").startswith("waiting:")
                 and isinstance(heartbeat.get("wait"), dict) else None),
        # A window fetched as its source posts: its schedule and how many of its hours are in.
        "posting": facts.get("posting"),
        # The lead a run stopped on when its source fell behind (exit 75).
        "source_behind": facts.get("source_behind"),
        "metadata_warnings": sorted(warnings),
    }


def summarize(root: Path, rundir: Path) -> dict[str, Any]:
    run_id = rundir.resolve().relative_to(root.resolve()).as_posix()
    info = status(rundir)
    render = read_json(rundir / RENDER, default=None)
    return {"id": run_id, "name": rundir.name, "folder": str(rundir), "status": info,
            "machine": info.get("machine") or "this-computer",
            "render": None if not isinstance(render, dict) else {
                key: render.get(key) for key in ("machine", "state", "pictures", "worker_state", "message")}}


#: What an unreadable run's row says; the server's log keeps the exception.
UNREADABLE = "This folder's run records could not be read. The reason is in the page server's log."


def unreadable(root: Path, rundir: Path) -> dict[str, Any]:
    """The row of a run folder whose records could not be read: its name and a plain reason, nothing more."""

    try:
        run_id = rundir.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        run_id = rundir.name
    return {"id": run_id, "name": rundir.name, "folder": str(rundir), "title": run_id,
            "status": {"state": "unreadable", "message": UNREADABLE},
            "machine": "this-computer", "render": None, "card": {}}


def run_path(root: Path, run_id: str) -> Path:
    from .files import PathRefused, split_run_id

    parts = split_run_id(run_id)
    candidate = root.joinpath(*parts)
    if not within(root, candidate):
        raise PathRefused(f"{run_id!r} leads outside the runs folder.")
    # Each folder the id names must be spelled as it is on disk.  Windows opens "a " and "a." as the
    # folder "a", and " " as the forecasts folder itself, which then answered as a run of its own;
    # an id the list never gave opens nothing.  A link keeps the name it has, so it is not compared.
    for depth in range(1, len(parts) + 1):
        step = root.joinpath(*parts[:depth])
        try:
            if not step.exists() or _is_link(step):
                continue
            spelled = step.resolve().name
        except OSError:
            continue
        if spelled != parts[depth - 1]:
            raise PathRefused(f"{run_id!r} is not a run in this folder (NAME or FOLDER/NAME).")
    return candidate


def _is_link(path: Path) -> bool:
    """A symlink, or on Windows any reparse point (a junction as much as a symlink)."""

    try:
        info = path.lstat()
    except OSError:
        return False
    return S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT)


def run_id_of(root: Path, folder: Any) -> str | None:
    """The page's id of ``folder`` when it is one of the runs under ``root`` (NAME or FOLDER/NAME), else None."""

    if not folder:
        return None
    path = Path(str(folder))
    try:
        run_id = path.resolve().relative_to(root.resolve()).as_posix()
        if run_path(root, run_id).resolve() != path.resolve():
            return None
    except (ValueError, OSError):  # outside the root, or not a run id (PathRefused is a ValueError)
        return None
    return run_id if is_run_dir(path) else None


def existing_run(root: Path, run_id: str) -> Path:
    path = run_path(root, run_id)
    if not path.is_dir() or not (is_run_dir(path) or has_frames(path)):
        raise FileNotFoundError(f"No run called {run_id!r}.")
    return path


__all__ = [
    "COMMANDS_LOG", "EVENTS", "EventTail", "FOLLOW", "HEARTBEAT", "JOB", "LIBRARY_WARNING", "MANIFEST",
    "PLAN", "QUEUED", "REMOTE", "REMOTE_MANIFEST", "RENDER", "STARTS", "START_SCHEMA", "STATES", "STOP",
    "downscale_grid", "downscale_plan", "LIST_LIMIT", "event_facts", "existing_run", "folder_mtime", "follow_needed",
    "follower_alive", "grid_facts", "has_frames", "is_run_dir", "iter_runs", "job_alive", "job_of", "last_words",
    "nest_moves", "number",
    "plan_intent", "run_id_of", "run_path", "start_end", "start_record", "start_record_path", "status", "summarize",
    "unreadable",
]


# ------------------------------------------------------------------ nest moves

#: The engine's step log (``gpuwm.progress_log``), written beside the model's output while the run is alive.
STEP_LOG = "progress.jsonl"
STEP_LOG_SCHEMA = "gpuwm.step-log/"
_MOVES: dict[str, tuple[tuple[int, float], list[dict[str, Any]]]] = {}


def _step_logs(rundir: Path) -> list[Path]:
    # The step log sits a few folders down (chain/<run>/run/progress.jsonl on the run-plan route); a fixed-depth
    # glob lists folders only, so it never walks the render tree's thousands of pictures.
    found: list[Path] = []
    for depth in range(4):
        found.extend(sorted(rundir.glob("/".join(["*"] * depth + [STEP_LOG]))))
    return [path for path in found if path.is_file()]


def _step_time(text: Any) -> str | None:
    # "2026-09-24_18:45:00" -> "2026-09-24T18:45:00Z", the spelling the picture listing uses
    moment = _parse_time(str(text or "").replace("_", "T")) if text else None
    return None if moment is None else moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def nest_moves(rundir: Path) -> list[dict[str, Any]]:
    """Every place a nest moved to while the run went, oldest first.

    From the engine's live step log: ``nest_moved`` (a follower took a new
    place in its parent) and ``containment_moved`` (a parent slid under a
    mover that stayed where it was on the Earth).  Each row is
    ``{"valid", "grid_id", "i_parent_start", "j_parent_start"}``; a
    containment slide adds a second row for the mover, whose place in its
    parent changes by the slide times the parent's ratio (``"shift_by"``,
    the moved parent's id).  A frame written at a row's valid time was
    written before the move: the move shows from the next frame on.
    """

    rows: list[dict[str, Any]] = []
    for path in _step_logs(rundir):
        try:
            stat = path.stat()
        except OSError:
            continue
        key = (stat.st_size, stat.st_mtime)
        hit = _MOVES.get(str(path))
        if hit is not None and hit[0] == key:
            rows.extend(hit[1])
            continue
        found: list[dict[str, Any]] = []
        try:
            blob = long_path(path).read_bytes()
        except OSError:
            blob = b""
        for line in blob.splitlines():
            if b"nest_moved" not in line and b"containment_moved" not in line:
                continue
            try:
                record = load_json(line)
            except (ValueError, RecursionError):
                continue
            if not isinstance(record, dict) or not str(record.get("schema", "")).startswith(STEP_LOG_SCHEMA):
                continue
            tag = record.get("event")
            place = _object(record.get("placement_to"))
            valid = _step_time(record.get("valid_time"))
            try:
                grid_id = int(record.get("domain"))
                i_start = int(place["i_parent_start"])
                j_start = int(place["j_parent_start"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if tag not in ("nest_moved", "containment_moved") or valid is None:
                continue
            found.append({"valid": valid, "grid_id": grid_id, "i_parent_start": i_start, "j_parent_start": j_start})
            shift = record.get("executed_shift_parent_cells")
            if tag == "containment_moved" and record.get("mover") is not None and isinstance(shift, list):
                try:
                    found.append({"valid": valid, "grid_id": int(record["mover"]), "shift_by": grid_id,
                                  "shift": [int(shift[0]), int(shift[1])]})
                except (TypeError, ValueError, IndexError):
                    pass
        _MOVES[str(path)] = (key, found)
        rows.extend(found)
    return sorted(rows, key=lambda row: row["valid"])
