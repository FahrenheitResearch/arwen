"""WRF-grade per-step progress: one line per model step, and events.

Why this module exists
----------------------

A user driving this package from his own script reported the gap in one
sentence: the simulation "only prints every 20 seconds and doesnt show
what time step the model is actually on, just the most recently output
frame".  Both halves were true.  The forecast runners republished
``progress.json`` on their own throttle (every sixtieth step) and
``gpuwm go`` printed a stopwatch heartbeat every twenty seconds naming
whatever that file last said.  So a caller could not answer either of
the two questions a driving script actually asks:

* **where is the run?**  Not "some frame landed a while ago" -- which
  model step, on which domain, at which valid time, and how long that
  step took.
* **is this frame safe to open?**  A size or mtime check races the
  writer.  The history writer's own publication is already atomic --
  fsync, self-validate, ``os.replace`` onto the final name, with the
  in-flight temporary hidden behind a leading dot so a ``wrfout_*``
  glob cannot see it -- but that was a property of the code and not a
  documented signal, so every consumer invented its own poll.  One
  measured consequence in this tree: reaching an 89 s time-to-first-plot
  took an EXTERNAL process polling frames for a completion attribute,
  reimplementing a hook the writer already raises.

WRF answers both, and it answers them the same way it has for twenty
years: one ``Timing for main:`` line per model time step per domain, one
``Timing for Writing`` line for every history and restart file it
publishes.  This module matches that grammar rather than inventing one,
so a script that already reads ``rsl.out.0000`` needs a new path and not
a new parser.

The three streams
-----------------

**Text**, on stdout, in WRF's own sentences::

    Timing for main: time 2026-08-15_00:00:12 on domain   1:     0.06382 elapsed seconds  step 1
    Timing for Writing wrfout_d01_2026-08-15_00:00:00 for domain   1:     0.24310 elapsed seconds
    Timing for Writing restart for domain   1:     1.51200 elapsed seconds
    d02 2026-08-15_00:00:00 gpuwm: domain start
    d02 2026-08-15_01:00:00 gpuwm: domain end, 240 steps
    gpuwm: SUCCESS COMPLETE SIMULATION, 300 steps, 182.4 wall seconds

The step index after ``elapsed seconds`` is the one field WRF does not
print and the reporting script asked for.  It is appended AFTER the WRF
sentence on purpose: a prefix-matching WRF parser is unaffected.

**JSONL**, one object per line, at ``<outdir>/progress.jsonl``.  Every
text line has exactly one record and the record carries the text it
printed under ``text``, so the two streams cannot disagree about what
happened -- they are one emit.  ``sequence`` is dense and monotonic, so
a consumer can detect loss rather than assume none.

**Frame markers**, at ``<outdir>/ready/<frame-name>.json``.  See
:func:`write_frame_marker` for exactly what the marker guarantees.

What this module deliberately does NOT do
-----------------------------------------

It never synchronises a device.  The per-step wall time is a
``perf_counter`` pair the executor takes around the step it was already
taking; a "true" per-step GPU time would need a stream synchronise per
step, which would serialise the pipeline the timing exists to observe.
The number reported is therefore WRF's number -- host wall time across
the step's launches -- and that is the one a progress display wants.

It is also not a replacement for anything.  ``run-progress.json``
(:mod:`gpuwm.supervisor`) stays the durable reattach anchor,
``progress.json`` stays the coarse stage sample, and
``events.jsonl`` (:mod:`gpuwm.runplan`) stays the plan-level event
stream.  This is the per-step layer none of them had.
"""

from __future__ import annotations

import atexit
import dataclasses
import json
import math
import os
import sys
import threading
import time
import weakref
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

#: One JSONL line per event, at this schema.
#:
#: v2 adds one event tag, ``phase``, and one field on ``run_end``
#: (``first_step_excess_seconds``).  v3 adds the six :data:`NEST_EVENTS`
#: tags and nothing else.  Nothing an older version emitted changed
#: shape, so a v3 stream differs from a v2 stream only by carrying more
#: -- but the schema string moves anyway, because a consumer meeting a
#: tag it was never told about must refuse loudly rather than skip it,
#: and refusing on the schema is how it does that.
STEP_LOG_SCHEMA = "gpuwm.step-log/v3"

#: Every schema :func:`read_step_log` will replay.  A published wheel's
#: progress.jsonl outlives the version that wrote it, and this tree must
#: not lose the ability to read the streams it has already shipped.
STEP_LOG_SCHEMAS = ("gpuwm.step-log/v3", "gpuwm.step-log/v2",
                    "gpuwm.step-log/v1")

#: One JSON document per durable output frame.
FRAME_MARKER_SCHEMA = "gpuwm.frame-ready/v1"

#: Where the machine stream lands inside a run's output directory.
STEP_LOG_FILENAME = "progress.jsonl"

#: Where the frame markers land.  A DIRECTORY and not a suffix beside
#: the frames: a marker called ``wrfout_d01_....ready`` would be matched
#: by the same ``wrfout_d01_*`` glob a caller already runs, which is the
#: confusion this whole feature exists to remove.
FRAME_MARKER_DIRNAME = "ready"

#: What the tree does to ITSELF, as opposed to what it computes.
#:
#: The gap these close: a run could relocate a nest across half a state,
#: retire an episode and re-arm the slot, and the per-step stream said
#: nothing at all.  Those decisions lived only in
#: ``relocation_receipts.json`` and ``spawn_receipts.json``, files
#: written for a post-mortem, so anything watching a run LIVE could not
#: draw the tree it was watching -- the nest rectangles could not move,
#: the storm track could not be drawn, and an episode beginning or
#: ending was invisible until the run was over.
#:
#: The spellings are a published contract, not an internal name: they
#: are what a live consumer switches on.  What each one means:
#:
#: ``nest_spawned``      a dormant nest's trigger fired and the child was
#:                       materialized -- the birth AND the activation,
#:                       which on this engine are one leg-boundary act.
#: ``nest_retired``      a live episode stopped participating; the domain
#:                       leaves the next leg's tree.
#: ``nest_rearmed``      a retired slot re-opened for a later episode.
#:                       Nothing exists yet, so it carries no position.
#: ``nest_moved``        a follower executed a relocation, carrying the
#:                       placement it left and the one it took.
#: ``containment_moved`` the mover's PARENT slid to keep the mover
#:                       contained, with the mover earth-fixed under it.
#: ``track_fix``         one tracker fix as the track file records it.
#:
#: A relocation that HELD is deliberately not an event.  A hold is the
#: absence of a move, it happens at every cadence boundary a storm sits
#: still, and the receipts already carry every one of them with the
#: reason; a live map has nothing to redraw for it.
NEST_EVENTS = (
    "nest_spawned", "nest_retired", "nest_rearmed",
    "nest_moved", "containment_moved", "track_fix",
)

#: Every tag this module will ever emit.  A consumer switching on
#: ``event`` can be exhaustive against this tuple.
STEP_LOG_EVENTS = (
    "run_start", "phase", "domain_start", "step", "output_written",
    "restart_written", "domain_end", "run_end", *NEST_EVENTS,
)

#: Step 1 must exceed this multiple of the median of the steps that
#: follow it before its excess is reported as a number of its own.
#:
#: CHOSEN AGAINST A MEASURED DISTRIBUTION, not guessed, and the first
#: number tried was wrong.  Ten was the obvious threshold and it fired
#: on a HEALTHY warm run: three warm runs of the reference case on this
#: box, 2026-08-16, gave first-step ratios of 9.5x, 10.2x and 9.2x with
#: no compilation of any kind -- step 1 legitimately pays first-touch
#: allocations, the first history alarm and the first load of each
#: cached kernel, and that costs about an order of magnitude.  The run
#: this instrument exists to catch was 51.1 s against a 0.13 s steady
#: state: roughly 390x.
#:
#: 25 sits 2.5x above the measured healthy maximum and 15x below the
#: measured pathology, which is as much daylight as either side offers.
FIRST_STEP_EXCESS_FACTOR = 25.0

#: How many steps after the first are sampled for that median.
_FIRST_STEP_WINDOW = 6

#: WRF's own time spelling, and the one in every wrfout filename.
WRF_TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"

#: ``--progress-format``'s accepted values.
PROGRESS_FORMATS = ("text", "jsonl", "off")

#: ``--progress-output`` spelling for "the machine stream goes to stdout".
#: A pipe is the shape a driving script actually wants, and there is no
#: ``/dev/stdout`` on the platform this package's reference card lives on.
STDOUT_SENTINEL = "-"


# ---------------------------------------------------------------------------
# Formatters.  Pure, so the grammar can be tested without a run.
# ---------------------------------------------------------------------------


def format_model_time(value) -> str:
    """One model instant, spelled the way WRF and every wrfout name do.

    Refuses a sub-second instant rather than truncating one.  A log line
    and a frame name must agree, ``wrfout_filename`` refuses the same
    input, and a silently rounded valid time is the kind of small lie a
    driving script cannot detect.
    """

    if isinstance(value, str):
        return value
    if getattr(value, "microsecond", 0):
        raise ValueError(
            f"model time {value!r} is not on a whole second; history "
            "frames and their log lines are both named at whole-second "
            "resolution")
    return value.strftime(WRF_TIME_FORMAT)


def format_step_line(*, domain: int, step: int, valid_time,
                     wall_seconds: float) -> str:
    """One model time step, in WRF's ``Timing for main:`` grammar.

    WRF's own write is ``'Timing for main: time ', time, ' on domain ',
    id, ':  ', seconds, ' elapsed seconds'`` with ``I3`` and ``F10.5``
    (``share/mediation_integrate.F``).  The trailing ``step N`` is ours.
    """

    return (f"Timing for main: time {format_model_time(valid_time)} "
            f"on domain {int(domain):3d}:  {float(wall_seconds):10.5f} "
            f"elapsed seconds  step {int(step)}")


def format_output_line(*, domain: int, path, wall_seconds: float) -> str:
    """One history file published, in WRF's ``Timing for Writing``.

    WHAT THE NUMBER IS, because it is not quite WRF's.  WRF writes
    history synchronously and reports how long the model was blocked;
    this package writes it on a per-domain side thread and the model is
    never blocked, so "how long the write took" is not a number a reader
    can act on.  What is reported instead is the DURABLE-PUBLISH
    LATENCY: seconds from the model reaching this frame's valid time to
    the file being fsynced, validated and renamed into place.  That is
    the number that answers "how far behind the run are my plots", and
    it is stated in the JSONL as ``durable_after_seconds`` so nothing
    depends on reading this docstring.
    """

    return (f"Timing for Writing {Path(path).name} for domain "
            f"{int(domain):3d}:  {float(wall_seconds):10.5f} "
            "elapsed seconds")


def format_restart_line(*, domain: int, path, wall_seconds: float) -> str:
    """One restart published.  WRF writes the word, not the filename."""

    return (f"Timing for Writing restart for domain {int(domain):3d}:  "
            f"{float(wall_seconds):10.5f} elapsed seconds  "
            f"{Path(path).name}")


def format_phase_line(*, name: str, wall_seconds: float) -> str:
    """One pre-sim phase, in the module's own ``elapsed seconds`` shape.

    WRF has no sentence for this -- it has no equivalent of "verify the
    preparation receipt" or "compile the local card's kernels" -- so the
    grammar borrows the two things every line here shares: the number
    formatted ``F10.5``, and the word ``elapsed seconds`` after it.  A
    ``Timing for main:`` parser is unaffected: it prefix-matches, and
    this line starts with ``gpuwm:``.
    """

    return (f"gpuwm: phase {str(name)}:  {float(wall_seconds):10.5f} "
            "elapsed seconds")


def format_domain_start_line(*, domain: int, valid_time) -> str:
    return (f"d{int(domain):02d} {format_model_time(valid_time)} "
            "gpuwm: domain start")


def format_domain_end_line(*, domain: int, valid_time, steps: int) -> str:
    return (f"d{int(domain):02d} {format_model_time(valid_time)} "
            f"gpuwm: domain end, {int(steps)} steps")


def format_nest_line(*, event: str, domain: int, valid_time,
                     detail: str) -> str:
    """One lifecycle or relocation event, in the module's own grammar.

    WRF has no sentence for any of these -- it has no trigger-spawned
    nests, no re-armable slots and no tracker of its own -- so the
    grammar borrows the shape ``domain start``/``domain end`` already
    use: the domain, the valid time, ``gpuwm:``, and then the event.
    A ``Timing for main:`` parser is unaffected, because it
    prefix-matches and this line starts with ``dNN``.

    The tag itself is the first word of the sentence rather than a
    prose paraphrase of it, so ``grep nest_moved`` finds the text lines
    and the JSONL records with one pattern.
    """

    return (f"d{int(domain):02d} {format_model_time(valid_time)} "
            f"gpuwm: {str(event)}, {detail}")


def format_position(lat, lon) -> str:
    """The position clause a nest sentence ends with, or nothing.

    ``at lat 35.1900 lon -96.8100``.  Empty when the position could not
    be derived, which is the truthful rendering: an idealized tree has
    no projection, so its nest events have no place on a map and must
    not pretend to.
    """

    if lat is None or lon is None:
        return ""
    return f" at lat {float(lat):.4f} lon {float(lon):.4f}"


def centre_latlon(grid) -> tuple[float | None, float | None]:
    """The geographic centre of one domain's grid, or ``(None, None)``.

    Asked of the grid in ITS OWN index space, through the same
    :func:`gpuwm.core.storm_track_writer.grid_center_latlon` the track
    file uses, so a nest rectangle drawn from these events and a track
    row drawn from that file cannot disagree about where the domain is.
    The grid is rebuilt on every relocation, so this is where the nest
    is NOW rather than where it was declared.

    TOLERANT BY CONTRACT, not by carelessness.  This is telemetry on the
    integration thread: an idealized tree carries a stand-in for a grid,
    a streamed domain may carry none at all, and a run must not die
    mid-forecast because a log line wanted a latitude.  Anything that
    cannot be converted becomes ``(None, None)`` and the record says so.
    """

    if grid is None:
        return (None, None)
    try:
        from gpuwm.core.storm_track_writer import grid_center_latlon

        lat, lon = grid_center_latlon(grid)
        lat, lon = float(lat), float(lon)
    except Exception:  # noqa: BLE001 - telemetry never fails a run
        return (None, None)
    if not (math.isfinite(lat) and math.isfinite(lon)
            and -90.0 <= lat <= 90.0):
        return (None, None)
    return (round(lat, 6), round(lon, 6))


def _placement_json(value) -> dict[str, int] | None:
    """One placement, in the spelling every relocation receipt uses."""

    if value is None:
        return None
    if isinstance(value, dict):
        return {"i_parent_start": int(value["i_parent_start"]),
                "j_parent_start": int(value["j_parent_start"])}
    i, j = value
    return {"i_parent_start": int(i), "j_parent_start": int(j)}


def _shift_json(value) -> list[int] | None:
    return None if value is None else [int(value[0]), int(value[1])]


def format_run_end_line(*, status: str, steps: int,
                        wall_seconds: float) -> str:
    """The last line.  ``wrf: SUCCESS COMPLETE WRF`` is the model."""

    verdict = ("SUCCESS COMPLETE SIMULATION" if str(status) == "SUCCESS"
               else f"{status} INCOMPLETE SIMULATION")
    return (f"gpuwm: {verdict}, {int(steps)} steps, "
            f"{float(wall_seconds):.1f} wall seconds")


# ---------------------------------------------------------------------------
# The frame marker, and exactly what it promises
# ---------------------------------------------------------------------------


def write_frame_marker(marker_dir, *, domain: int, valid_time, path) -> Path:
    """Publish "this frame is complete", atomically, or publish nothing.

    THE GUARANTEE, stated so a caller can rely on the right half of it:

    * The marker is written only after the frame's own publication has
      completed -- :class:`gpuwm.io.wrfout.WrfoutWriter` closes the
      netCDF handle, ``fsync``s the temporary, re-opens and validates its
      inventory, and only then ``os.replace``s it onto the final name.
      The writer thread raises its landing hook after that replace.
    * The marker itself is published by ``tmp -> fsync -> os.replace``,
      so a reader never observes a half-written marker.

    Therefore: **a marker that exists names a frame that is complete and
    readable.**  The converse does not hold -- a marker can be missing
    for a frame that is fine (the marker write failed, markers were
    switched off, the run pre-dates this feature) -- so a consumer polls
    for markers and treats their absence as "not yet", never as
    "corrupt".

    Returns the marker path, or raises if the frame is not there: a
    marker must never outrun its data.
    """

    from gpuwm.supervisor import atomic_write_json

    frame = Path(path).resolve()
    # stat() BEFORE writing anything.  A marker for a file that is not on
    # disk is worse than no marker at all, because it is the one signal a
    # consumer is being told to trust.
    size = frame.stat().st_size
    marker_dir = Path(marker_dir)
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"{frame.name}.json"
    atomic_write_json(marker, {
        "schema": FRAME_MARKER_SCHEMA,
        "domain": int(domain),
        "valid_time": format_model_time(valid_time),
        "path": str(frame),
        "size_bytes": int(size),
        "published_unix_ms": int(time.time() * 1000),
        # Said in the artifact, not only in this docstring: a consumer
        # reading a marker off a machine that has no gpuwm checkout still
        # learns what the marker is claiming.
        "guarantee": ("the named file was fsynced, self-validated and "
                      "renamed onto its final name before this marker "
                      "was published; the marker itself is published by "
                      "rename, so its presence means the frame is "
                      "complete and readable"),
    })
    return marker


# ---------------------------------------------------------------------------
# Reading the stream back
# ---------------------------------------------------------------------------


def read_step_log(path) -> list[dict[str, Any]]:
    """Replay one step log, refusing a stream that lost a line.

    The sequence is dense by construction, so a gap is a lost or
    reordered record and never a skipped one.  Refused rather than
    silently repaired: a consumer that cannot tell "the run is quiet"
    from "I missed the last hundred steps" is exactly the consumer this
    module was written for.
    """

    records: list[dict[str, Any]] = []
    text = Path(path).read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path} line {number} is not JSON: {error}") from error
        if record.get("schema") not in STEP_LOG_SCHEMAS:
            raise ValueError(
                f"{path} line {number} is not a step-log record at any of "
                f"{list(STEP_LOG_SCHEMAS)}: {record.get('schema')!r}")
        expected = len(records) + 1
        if record.get("sequence") != expected:
            raise ValueError(
                f"{path} line {number} carries sequence "
                f"{record.get('sequence')!r}, expected {expected}; the "
                "stream is append-only and its sequence is dense, so a "
                "gap is a lost or reordered line")
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# The log
# ---------------------------------------------------------------------------


def _close_dangling(ref) -> None:
    """Terminate a log whose owner exited without closing it.

    Registered with :mod:`atexit` and held only through a weak
    reference, so it neither keeps a finished run's log alive nor fires
    for one that closed normally.
    """

    log = ref()
    if log is not None:
        log.close(status="INCOMPLETE",
                  error="the process exited without closing the step log; "
                        "the run did not reach its own end")


class StepLog:
    """One emit, two streams, and a marker per durable frame.

    Every public method is telemetry, and telemetry never fails a run:
    a sink that raises is dropped for that one write and the run
    continues.  The one thing that IS allowed to be missing loudly is a
    frame marker, and it is reported as ``marker: null`` on the
    ``output_written`` record rather than by writing a marker that lies.

    Thread safety is not optional here.  ``output_committed`` is raised
    on the per-domain wrfout writer's own daemon thread while
    ``domain_step`` is being called from the integration loop, so two
    threads genuinely reach :meth:`_emit` at once and an interleaved
    JSONL line is not recoverable by any reader.
    """

    #: True here, False on :class:`_NullStepLog`.  A call site that must
    #: not pay for a disabled log -- the per-STEP observer, the writer's
    #: single landing slot -- reads this instead of testing the type.
    enabled = True

    def __init__(self, *, start_time: datetime, run_seconds: float,
                 text_stream=None, jsonl_path=None, frame_marker_dir=None,
                 every: int = 1):
        self._start_time = start_time
        self._run_seconds = float(run_seconds) if run_seconds else 0.0
        self._text = text_stream
        self._every = max(1, int(every))
        self._marker_dir = (None if frame_marker_dir is None
                            else Path(frame_marker_dir))
        self._lock = threading.Lock()
        self._sequence = 0
        self._closed = False
        self._started_wall = time.perf_counter()
        #: grid_id -> [step_count, model_seconds, emitted_step_count].
        #: Written only from the integration thread (``domain_step``) and
        #: read from it and from ``close`` after the writers have
        #: drained, so it needs no lock; the counters that ARE touched
        #: from a writer thread live behind :meth:`_emit`'s.
        self._domains: dict[int, list] = {}
        #: grid_id -> the perf_counter reading of its last completed
        #: step.  Read by :meth:`output_committed` to turn a landing into
        #: a durable-publish LATENCY, which is the only honest timing an
        #: asynchronous writer can report.
        self._domain_wall: dict[int, float] = {}
        #: grid_id -> the wall of its first few steps, in order.  Kept
        #: for :meth:`close`'s first-step excess, and recorded BEFORE the
        #: ``--progress-every`` thinning so a quiet stream and a chatty
        #: one reach the same verdict.
        self._first_steps: dict[int, list[float]] = {}
        #: What the kernel cache said before physics initialisation, or
        #: ``None`` when it said nothing.  Set by
        #: :meth:`announce_kernel_compile`; read by :meth:`close`, which
        #: is the only place with a measurement to attach the name to.
        self._compile_announcement: dict[str, Any] | None = None
        self._outputs = 0
        self._restarts = 0
        self._stream = None
        if self._marker_dir is not None:
            # Eagerly, at construction.  A consumer is told to poll this
            # directory, and "poll a directory that does not exist yet"
            # is a race dressed up as an instruction: the first frame can
            # land minutes into a run.
            try:
                self._marker_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self._marker_dir = None
        self._owns_stream = False
        if str(jsonl_path) == STDOUT_SENTINEL:
            # A pipe, not a file.  Not closed on the way out either --
            # this process did not open stdout and must not shut it.
            self._stream = sys.stdout
            self.jsonl_path = None
        elif jsonl_path is not None:
            path = Path(jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = path.open("a", encoding="utf-8", newline="\n")
            self._owns_stream = True
            self.jsonl_path = path
        else:
            self.jsonl_path = None
        # A terminal event on EVERY exit path, including the ones that
        # never reach the integration loop.  A forecast that dies in
        # preflight used to leave a driving script with `run_start` and
        # then silence, which is indistinguishable from a run that is
        # still going.  `close` is idempotent, so the ordinary paths
        # unregister this and it never fires.
        self._at_exit = lambda ref=weakref.ref(self): _close_dangling(ref)
        atexit.register(self._at_exit)
        self._emit("run_start", self._run_start_line(), {
            "start_time": format_model_time(start_time),
            "run_seconds": self._run_seconds,
            "pid": os.getpid(),
            "frame_marker_dir": (None if self._marker_dir is None
                                 else str(self._marker_dir.resolve())),
            "progress_every": self._every,
        })

    # -- the per-step hook the executor calls ------------------------

    def domain_step(self, *, grid_id: int, step_count: int,
                    model_seconds: float, step_wall_seconds: float) -> None:
        """One model time step of one domain has completed.

        This is the signature :func:`gpuwm.core.model.execute_experiment`
        calls its ``step_observer`` with, once per domain per step,
        immediately after that domain's STEP op returns.  It is the ONLY
        source of ``step`` records: nothing here is derived from a frame
        count or a period boundary.
        """

        grid_id = int(grid_id)
        step_count = int(step_count)
        model_seconds = float(model_seconds)
        state = self._domains.get(grid_id)
        if state is None:
            state = [step_count, model_seconds, 0]
            self._domains[grid_id] = state
            # The model time of the domain's FIRST completed step, which
            # for a delayed-start nest is its birth boundary and for the
            # root is one step in.  Not back-dated by a timestep: this
            # seam is handed elapsed seconds, not dt, and a subtracted
            # dt guessed from step_count would be wrong for exactly the
            # domain that matters (a nest whose clock starts offset).
            self._emit_domain_start(grid_id, model_seconds)
        state[0] = step_count
        state[1] = model_seconds
        # Recorded on EVERY step, including one that --progress-every
        # thins away: the output latency must not become a function of
        # how chatty the log was asked to be.
        self._domain_wall[grid_id] = time.perf_counter()
        # Same rule, same reason, for the first-step window: whether the
        # cold run's kernel compile is nameable must not depend on the
        # cadence the caller asked to be told about steps at.
        window = self._first_steps.setdefault(grid_id, [])
        if len(window) <= _FIRST_STEP_WINDOW:
            window.append(float(step_wall_seconds))
        if step_count != 1 and step_count % self._every:
            return
        state[2] = step_count
        valid = self._valid(model_seconds)
        self._emit("step", format_step_line(
            domain=grid_id, step=step_count, valid_time=valid,
            wall_seconds=step_wall_seconds), {
                "domain": grid_id,
                "step": step_count,
                "valid_time": valid,
                "model_seconds": model_seconds,
                "step_wall_seconds": float(step_wall_seconds),
                "fraction": (round(model_seconds / self._run_seconds, 6)
                             if self._run_seconds > 0.0 else None),
            })

    #: The name :class:`gpuwm.core.model` binds.  Kept as an alias so a
    #: caller can hand the BOUND METHOD straight to ``step_observer``
    #: without wrapping it in a lambda that would hide this object.
    @property
    def step_observer(self):
        return self.domain_step

    # -- the pre-sim phases -------------------------------------------

    def phase(self, name: str, wall_seconds: float | None,
              **fields: Any) -> None:
        """One named stretch of the road to step 1, and what it cost.

        The per-step stream answered "where is the run?" from step 1
        onwards and said nothing at all about how long getting to step 1
        took -- which on a cold run is most of the wall clock a reader
        spends waiting.  These records close that: ``preflight_verify``,
        ``restore_prepared_cache``, ``initialize_physics`` and
        ``kernel_compile``, each a number the runner already had and
        published only into a receipt written when the run was over.

        ``wall_seconds=None`` emits NOTHING.  A road that did not
        measure a phase (the store-direct road does not initialise
        physics separately, a host that ran no preflight) has no number,
        and a record carrying null would be read as "it was free" --
        the opposite of what an unmeasured phase means.
        """

        if wall_seconds is None:
            return
        self._emit("phase", format_phase_line(
            name=name, wall_seconds=wall_seconds), {
                "name": str(name),
                "wall_seconds": round(float(wall_seconds), 6),
                **fields,
            })

    def announce_kernel_compile(self, *, reason: str,
                                compute_capability: str | None = None,
                                **fields: Any) -> None:
        """Record that the kernel cache said a compile was coming.

        Deliberately NOT an emit.  What the cache predicted is not yet a
        measurement, and the measurement -- how much of step 1's wall the
        compile actually took -- does not exist until several steps have
        run.  :meth:`close` joins the two and emits one ``phase``
        record with both the name and the number; until then the claim
        is held here rather than published unbacked.
        """

        self._compile_announcement = {
            "reason": str(reason),
            "compute_capability": (None if compute_capability is None
                                   else str(compute_capability)),
            **fields,
        }

    def _first_step_excess_seconds(self) -> float | None:
        """How much longer step 1 took than the steps around it, or None.

        Measured on the ROOT domain -- the lowest grid id that stepped --
        because it is the one whose first step every physics module's
        kernels are compiled for, and a nest's first step is a different
        event (its birth boundary).

        ``None`` unless there is real evidence: at least three steps
        after the first to take a median of, and a first step at least
        :data:`FIRST_STEP_EXCESS_FACTOR` times that median.  A median
        rather than a mean because one slow neighbour (a history write
        landing) must not dilute the comparison.
        """

        if not self._first_steps:
            return None
        window = self._first_steps[min(self._first_steps)]
        if len(window) < 4:
            return None
        first, rest = window[0], sorted(window[1:])
        middle = len(rest) // 2
        median = (rest[middle] if len(rest) % 2
                  else 0.5 * (rest[middle - 1] + rest[middle]))
        if median <= 0.0 or first < FIRST_STEP_EXCESS_FACTOR * median:
            return None
        return round(first - median, 6)

    # -- the events ---------------------------------------------------

    def output_committed(self, *, domain: int, valid_time, path,
                         wall_seconds: float | None = None) -> None:
        """One history frame is durable.  Raised from the writer thread.

        Matches :class:`gpuwm.io.wrfout.AsyncDomainWrfoutWriter`'s
        landing hook signature exactly (``domain``/``valid_time``/
        ``path``), so it can be attached to the writer directly.

        ``wall_seconds`` is measured here when the caller does not
        supply it: see :func:`format_output_line` for what the number
        means and why it is a latency rather than a blocking time.
        """

        domain = int(domain)
        now = time.perf_counter()
        if wall_seconds is None:
            wall_seconds = now - self._domain_wall.get(
                int(domain), self._started_wall)
        marker = None
        marker_error = None
        if self._marker_dir is not None:
            try:
                marker = str(write_frame_marker(
                    self._marker_dir, domain=domain, valid_time=valid_time,
                    path=path))
            except (OSError, ValueError) as error:
                marker_error = f"{type(error).__name__}: {error}"
        try:
            size = Path(path).stat().st_size
        except OSError:
            size = None
        # NOT incremented here.  There is one writer thread PER DOMAIN,
        # so `+= 1` outside the lock is a genuine lost-update race on a
        # tree; `_emit` owns the counters and holds the lock.
        self._emit("output_written", format_output_line(
            domain=domain, path=path, wall_seconds=wall_seconds), {
                "domain": domain,
                "valid_time": format_model_time(valid_time),
                "path": str(Path(path).resolve()),
                "size_bytes": size,
                "durable_after_seconds": round(float(wall_seconds), 6),
                "marker": marker,
                "marker_error": marker_error,
            })

    def restart_written(self, *, domain: int, valid_time, path,
                        wall_seconds: float = 0.0) -> None:
        """One restart is durable."""

        domain = int(domain)
        try:
            size = Path(path).stat().st_size
        except OSError:
            size = None
        self._emit("restart_written", format_restart_line(
            domain=domain, path=path, wall_seconds=wall_seconds), {
                "domain": domain,
                "valid_time": format_model_time(valid_time),
                "path": str(Path(path).resolve()),
                "size_bytes": size,
            })

    # -- what the tree does to itself ---------------------------------
    #
    # Every one of these is raised from the INTEGRATION thread -- the
    # relocation runner at PERIOD_BEGIN, the spawn runner at a leg
    # boundary -- so reading ``self._domains`` here is the same thread
    # that writes it and needs no lock of its own; ``_emit`` still takes
    # one, because a wrfout writer thread can be landing a frame at the
    # same instant.

    def domain_step_count(self, domain: int) -> int:
        """How many steps this domain has completed, 0 before its first.

        A newborn reports 0 and that is the true answer: it was born at
        a boundary and has integrated nothing.  Borrowing its parent's
        count would put the event on the wrong row of a step ticker.
        """

        state = self._domains.get(int(domain))
        return 0 if state is None else int(state[0])

    def _nest_event(self, event: str, *, domain, model_seconds, detail: str,
                    grid=None, lat=None, lon=None, **fields: Any) -> None:
        domain = int(domain)
        model_seconds = float(model_seconds)
        valid = self._valid(model_seconds)
        if lat is None and lon is None and grid is not None:
            lat, lon = centre_latlon(grid)
        self._emit(event, format_nest_line(
            event=event, domain=domain, valid_time=valid,
            detail=detail + format_position(lat, lon)), {
                "domain": domain,
                "step": self.domain_step_count(domain),
                "valid_time": valid,
                "model_seconds": model_seconds,
                "lat": lat,
                "lon": lon,
                **fields,
            })

    def nest_spawned(self, *, domain, model_seconds, episode,
                     parent=None, placement=None, grid=None,
                     trigger=None, **fields: Any) -> None:
        """A dormant nest's trigger fired and its child now exists.

        Birth AND activation in one record, because on this engine they
        are one act: :meth:`SpawnRunner.on_leg_boundary` materializes
        the child at the same leg boundary that admits it to the next
        leg's schedule, so there is no interval in which a nest is born
        but not yet running for a second event to mark.
        """

        placement = _placement_json(placement)
        where = ("" if placement is None else
                 f" at i={placement['i_parent_start']} "
                 f"j={placement['j_parent_start']}")
        self._nest_event(
            "nest_spawned", domain=domain, model_seconds=model_seconds,
            detail=f"episode {int(episode)}{where}", grid=grid,
            episode=int(episode),
            parent=(None if parent is None else int(parent)),
            placement=placement,
            trigger=(None if trigger is None else str(trigger)),
            **fields)

    def nest_retired(self, *, domain, model_seconds, episode, reason=None,
                     grid=None, **fields: Any) -> None:
        """One live episode stopped; the domain leaves the next leg."""

        self._nest_event(
            "nest_retired", domain=domain, model_seconds=model_seconds,
            detail=f"episode {int(episode)} ended"
                   + ("" if reason is None else f" ({reason})"),
            grid=grid, episode=int(episode),
            reason=(None if reason is None else str(reason)), **fields)

    def nest_rearmed(self, *, domain, model_seconds, episode,
                     cooldown_seconds=None, **fields: Any) -> None:
        """A retired slot re-opened, for the episode it is armed FOR.

        Carries no position, and that absence is the point: nothing has
        been built yet, and repeating where the previous episode sat
        would draw a rectangle for a nest that does not exist.
        """

        self._nest_event(
            "nest_rearmed", domain=domain, model_seconds=model_seconds,
            detail=f"armed for episode {int(episode)}",
            episode=int(episode),
            cooldown_seconds=(None if cooldown_seconds is None
                              else float(cooldown_seconds)),
            **fields)

    def nest_moved(self, *, domain, model_seconds, placement_from,
                   placement_to, requested_shift=None, executed_shift=None,
                   clamped_by=(), grid=None, lat_from=None, lon_from=None,
                   **fields: Any) -> None:
        """One executed relocation, with the placement it LEFT.

        Old and new both, because a live map draws the origin ghost from
        the one and the moving rectangle from the other, and a consumer
        that had to remember the previous event to know where a nest
        came from would get it wrong across a reconnect.
        """

        placement_from = _placement_json(placement_from)
        placement_to = _placement_json(placement_to)
        self._nest_event(
            "nest_moved", domain=domain, model_seconds=model_seconds,
            detail=(f"i={placement_from['i_parent_start']} "
                    f"j={placement_from['j_parent_start']} -> "
                    f"i={placement_to['i_parent_start']} "
                    f"j={placement_to['j_parent_start']}"),
            grid=grid,
            placement_from=placement_from,
            placement_to=placement_to,
            requested_shift_parent_cells=_shift_json(requested_shift),
            executed_shift_parent_cells=_shift_json(executed_shift),
            clamped_by=[str(name) for name in (clamped_by or ())],
            lat_from=lat_from, lon_from=lon_from, **fields)

    def containment_moved(self, *, domain, model_seconds, mover,
                          placement_from, placement_to,
                          requested_shift=None, executed_shift=None,
                          clamped=False, mover_deviation_cells=None,
                          grid=None, lat_from=None, lon_from=None,
                          **fields: Any) -> None:
        """The mover's PARENT slid to keep the mover contained.

        Two grid ids, and they are not interchangeable: ``domain`` is
        the domain that moved, ``mover`` is the one it moved for and
        which stayed earth-fixed under the slide.
        """

        placement_from = _placement_json(placement_from)
        placement_to = _placement_json(placement_to)
        self._nest_event(
            "containment_moved", domain=domain, model_seconds=model_seconds,
            detail=(f"for d{int(mover):02d}: "
                    f"i={placement_from['i_parent_start']} "
                    f"j={placement_from['j_parent_start']} -> "
                    f"i={placement_to['i_parent_start']} "
                    f"j={placement_to['j_parent_start']}"),
            grid=grid, mover=int(mover),
            placement_from=placement_from,
            placement_to=placement_to,
            requested_shift_parent_cells=_shift_json(requested_shift),
            executed_shift_parent_cells=_shift_json(executed_shift),
            clamped=bool(clamped),
            mover_deviation_cells=_shift_json(mover_deviation_cells),
            lat_from=lat_from, lon_from=lon_from, **fields)

    def track_fix(self, *, domain, model_seconds, lat=None, lon=None,
                  found=True, refined_on=None, **fields: Any) -> None:
        """One tracker fix, as the run's track file records it.

        ``domain`` is the domain being STEERED, not the one the signal
        was found on -- that is ``refined_on`` when a two-stage tracker
        refined the centre on a finer grid.

        A consultation that found nothing is still emitted, with a null
        position and ``found`` false, for the same reason the track file
        writes an all-NaN row: the time axis stays complete and the gap
        is visible instead of being inferred from a jump in the clock.
        """

        self._nest_event(
            "track_fix", domain=domain, model_seconds=model_seconds,
            detail=("fix" if found else "no signal"),
            lat=lat, lon=lon, found=bool(found),
            refined_on=(None if refined_on is None else int(refined_on)),
            **fields)

    def close(self, *, status: str = "SUCCESS", error: str | None = None
              ) -> None:
        """Close every domain, say how the run ended, release the file.

        ``status`` describes the INTEGRATION, not the process.  Both
        runners close here -- once the last frame is durable -- and then
        go on to write a run receipt and a certification capsule, work
        that is outside this stream.  So a consumer takes ``run_end`` as
        "did the model finish" and the process's exit code as "did the
        command succeed"; the page says the same thing out loud.

        ``run_end`` is the last record, and that is the CALLER's
        obligation as much as this method's: both runners close only
        after the per-domain writers have drained (or, on the failure
        path, after the writer context has exited), so no landing can
        arrive behind it.  A landing that somehow did would find the
        stream closed and be dropped rather than appended, which keeps
        the sequence dense either way.
        """

        if self._closed:
            return
        self._closed = True
        atexit.unregister(self._at_exit)
        for grid_id in sorted(self._domains):
            steps, model_seconds, emitted = self._domains[grid_id]
            # The LAST step is always reported, even under --progress-every:
            # a thinned stream whose final line is step 990 of 997 leaves a
            # reader unable to tell a finished run from a stalled one.
            if steps != emitted:
                self._emit("step", format_step_line(
                    domain=grid_id, step=steps,
                    valid_time=self._valid(model_seconds),
                    wall_seconds=0.0), {
                        "domain": grid_id, "step": steps,
                        "valid_time": self._valid(model_seconds),
                        "model_seconds": model_seconds,
                        "step_wall_seconds": None,
                        "fraction": (
                            round(model_seconds / self._run_seconds, 6)
                            if self._run_seconds > 0.0 else None),
                        "final": True,
                    })
            valid = self._valid(model_seconds)
            self._emit("domain_end", format_domain_end_line(
                domain=grid_id, valid_time=valid, steps=steps), {
                    "domain": grid_id, "steps": steps, "valid_time": valid,
                    "model_seconds": model_seconds})
        # BEFORE run_end, so a consumer replaying the stream in order
        # learns what the compile cost before it learns the run is over.
        excess = self._first_step_excess_seconds()
        if excess is not None and self._compile_announcement is not None:
            # Named only because the cache SAID a compile was coming.
            # Any slow first step has an excess; calling one a kernel
            # compile without that evidence would be a guess wearing a
            # receipt's clothes.
            self.phase("kernel_compile", excess,
                       **self._compile_announcement,
                       measured_as="step 1 wall minus the median of the "
                                   "steps that follow it")
        wall = time.perf_counter() - self._started_wall
        total = sum(state[0] for state in self._domains.values())
        self._emit("run_end", format_run_end_line(
            status=status, steps=total, wall_seconds=wall), {
                "status": str(status),
                "steps": total,
                "wall_seconds": round(wall, 6),
                "outputs_written": self._outputs,
                "restarts_written": self._restarts,
                # Null on every healthy run, and that is the useful
                # reading: a number here says the run paid something
                # once, up front, that the steady state does not pay.
                "first_step_excess_seconds": excess,
                "error": error,
            })
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.flush()
                if self._owns_stream:
                    stream.close()
            except (OSError, ValueError):
                pass

    def __enter__(self) -> "StepLog":
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        self.close(status="SUCCESS" if exc_type is None else "FAIL",
                   error=None if exc is None else f"{exc_type.__name__}: {exc}")

    # -- internals ----------------------------------------------------

    def _run_start_line(self) -> str:
        return (f"gpuwm: STARTING SIMULATION at "
                f"{format_model_time(self._start_time)} for "
                f"{self._run_seconds:.0f} model seconds")

    def _valid(self, model_seconds: float) -> str:
        """One model instant as its WRF string.

        Whole seconds only, matching the frame names; a step that lands
        mid-second (a nest on a 5/3 s dt) is reported at its truncated
        second rather than refused, because unlike a FRAME name this
        string never has to round-trip to a filename.
        """

        return (self._start_time
                + timedelta(seconds=int(model_seconds))).strftime(
                    WRF_TIME_FORMAT)

    def _emit_domain_start(self, grid_id: int, model_seconds: float) -> None:
        valid = self._valid(model_seconds)
        self._emit("domain_start", format_domain_start_line(
            domain=grid_id, valid_time=valid), {
                "domain": grid_id, "valid_time": valid,
                "model_seconds": float(model_seconds)})

    def _emit(self, event: str, text: str, fields: dict) -> None:
        """Append one event to both streams, under one lock.

        The text line is carried INSIDE the record.  That is what makes
        "the human stdout and the JSONL agree" a property rather than a
        promise: there is one formatting call, and both sinks receive
        its result.
        """

        with self._lock:
            self._sequence += 1
            if event == "output_written":
                self._outputs += 1
            elif event == "restart_written":
                self._restarts += 1
            record = {
                "schema": STEP_LOG_SCHEMA,
                "sequence": self._sequence,
                "emitted_unix_ms": int(time.time() * 1000),
                "event": event,
                "text": text,
            }
            record.update(fields)
            if self._stream is not None:
                try:
                    self._stream.write(json.dumps(record, default=str) + "\n")
                    self._stream.flush()
                except (OSError, ValueError):
                    pass
            if self._text is not None:
                try:
                    self._text.write(text + "\n")
                    self._text.flush()
                except (OSError, ValueError):
                    pass


class _NullStepLog:
    """``--progress-format off``: the same object, doing nothing.

    A null object rather than ``None`` so no call site needs a guard.
    Every guard is a place the log can be silently forgotten, and this
    feature's whole failure mode was a run that said nothing.
    """

    enabled = False
    jsonl_path = None

    @property
    def step_observer(self):
        return self.domain_step

    def domain_step(self, **_fields) -> None:
        return None

    def phase(self, _name=None, _wall_seconds=None, **_fields) -> None:
        return None

    def announce_kernel_compile(self, **_fields) -> None:
        return None

    def output_committed(self, **_fields) -> None:
        return None

    def restart_written(self, **_fields) -> None:
        return None

    def domain_step_count(self, _domain) -> int:
        return 0

    def nest_spawned(self, **_fields) -> None:
        return None

    def nest_retired(self, **_fields) -> None:
        return None

    def nest_rearmed(self, **_fields) -> None:
        return None

    def nest_moved(self, **_fields) -> None:
        return None

    def containment_moved(self, **_fields) -> None:
        return None

    def track_fix(self, **_fields) -> None:
        return None

    def close(self, **_fields) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info) -> None:
        return None


#: The one inert log every un-logged call site lands on.  Stateless, so
#: one instance serves every caller.
NULL_STEP_LOG = _NullStepLog()


#: Where a run's step log hangs off its model, beside the other live
#: attachments the tree already carries (``_relocation_receipts``,
#: ``_relocation_runner``, ``_scratch_arena``).
_MODEL_STEP_LOG_ATTR = "_step_log"


def publish_step_log(model, step_log) -> None:
    """Give a model the log its mid-run emitters will reach for.

    THE MODEL, not the runners, and that is the whole point of this
    seam.  A relocation runner can be REBUILT mid-run -- a follow target
    that was dormant acquires one the moment it is born
    (``gpuwm.runtime._leg_boundary_pass``) -- so a log wired into a
    runner at construction is a log the run's later runners never get,
    and the nest that moved would be the one nothing recorded.  The
    model outlives every one of them.
    """

    setattr(model, _MODEL_STEP_LOG_ATTR, step_log)


def model_step_log(model):
    """The log this run publishes on, or the inert one.

    Never ``None``, so no emit site needs a guard.  Every guard is a
    place a lifecycle event can be silently forgotten, and a run that
    said nothing is the failure this whole module exists for.
    """

    log = getattr(model, _MODEL_STEP_LOG_ATTR, None)
    return NULL_STEP_LOG if log is None else log


def open_step_log(*, outdir, start_time, run_seconds,
                  progress_format: str = "text",
                  progress_output=None, progress_every: int = 1,
                  frame_markers: bool = True, text_stream=None):
    """Build the log a simulation front door runs with.

    DEFAULT-ON, and that is the point.  ``progress_format="text"``
    prints WRF's per-step sentences on stdout AND writes the machine
    stream to ``<outdir>/progress.jsonl`` AND publishes a frame marker
    per durable output.  A caller has to ask for silence
    (``--progress-format off``); it is never the thing that happens
    because nobody passed a flag.

    ``progress_format``:

    ``text``   stdout gets the sentences, the JSONL file is still written
    ``jsonl``  the JSONL stream only; stdout carries no sentences
    ``off``    an inert log; no file, no markers, no lines

    ``progress_output`` of ``"-"`` sends the JSONL stream to stdout
    instead of to a file, which with ``--progress-format jsonl`` gives a
    caller a pure record pipe and nothing else on the channel.
    """

    fmt = str(progress_format)
    if fmt not in PROGRESS_FORMATS:
        raise ValueError(
            f"--progress-format {fmt!r} is not one of "
            f"{list(PROGRESS_FORMATS)}")
    if fmt == "off":
        return _NullStepLog()
    outdir = Path(outdir)
    if progress_output is None:
        jsonl_path = outdir / STEP_LOG_FILENAME
    elif str(progress_output) == STDOUT_SENTINEL:
        jsonl_path = STDOUT_SENTINEL
    else:
        jsonl_path = Path(progress_output)
    stream = sys.stdout if text_stream is None else text_stream
    if fmt == "text" and jsonl_path == STDOUT_SENTINEL:
        raise ValueError(
            "--progress-output - puts the JSONL records on stdout, which "
            "--progress-format text also writes its sentences to; pick "
            "one channel (--progress-format jsonl for the record pipe, "
            "or --progress-output PATH for a file beside the run)")
    return StepLog(
        start_time=start_time, run_seconds=run_seconds,
        text_stream=None if fmt == "jsonl" else stream,
        jsonl_path=jsonl_path,
        frame_marker_dir=(outdir / FRAME_MARKER_DIRNAME
                          if frame_markers else None),
        every=progress_every)


class LandingFanout:
    """Give one wrfout landing to several consumers.

    :meth:`gpuwm.io.wrfout.PerDomainWrfoutWriters.attach_progress_callback`
    reads ONE ``output_committed`` attribute and assigns it to every
    writer, so a second consumer cannot be added by attaching twice --
    the second attach silently unhooks the first.  This is the object to
    attach when there is more than one.

    A sink that raises is dropped for that frame and never takes the
    writer thread, the run, or its sibling sinks down with it.
    """

    def __init__(self, *sinks):
        self._sinks = tuple(sink for sink in sinks if sink is not None)

    def __bool__(self) -> bool:
        return bool(self._sinks)

    def output_committed(self, **event) -> None:
        for sink in self._sinks:
            try:
                sink(**event)
            except Exception:  # noqa: BLE001 - telemetry never fails a run
                pass


@dataclasses.dataclass(frozen=True)
class ProgressOptions:
    """The four flags, carried from a parser to the integration loop.

    A runner cannot build the log at argument-parsing time: the log
    needs the experiment's ``start_time`` and ``run_seconds``, and those
    are only known once the hash-bound configuration has been loaded and
    validated -- hundreds of lines later, past every refusal.  So the
    ANSWERS travel, and :meth:`open` is called at the one point where
    both halves are in scope.
    """

    progress_format: str = "text"
    progress_output: Path | None = None
    progress_every: int = 1
    frame_markers: bool = True

    @classmethod
    def from_args(cls, args) -> "ProgressOptions":
        """Read the four flags off a parsed namespace.

        Tolerant of a namespace that lacks them, so a caller driving a
        runner's function directly (rather than its command line) does
        not have to synthesize flags to get the default behaviour.
        """

        return cls(
            progress_format=getattr(args, "progress_format", "text"),
            progress_output=getattr(args, "progress_output", None),
            progress_every=getattr(args, "progress_every", 1),
            frame_markers=getattr(args, "frame_markers", True))

    def open(self, *, outdir, start_time, run_seconds, text_stream=None):
        return open_step_log(
            outdir=outdir, start_time=start_time, run_seconds=run_seconds,
            progress_format=self.progress_format,
            progress_output=self.progress_output,
            progress_every=self.progress_every,
            frame_markers=self.frame_markers, text_stream=text_stream)


def _positive_cadence(value: str) -> int:
    """``--progress-every``'s type.  Refuses, rather than clamping.

    ``--progress-every 0`` has no sensible reading, and silently turning
    it into 1 teaches a caller that the flag took a value it did not.
    """

    import argparse as _argparse

    try:
        number = int(value)
    except ValueError:
        raise _argparse.ArgumentTypeError(
            f"{value!r} is not an integer") from None
    if number < 1:
        raise _argparse.ArgumentTypeError(
            f"must be 1 or more (got {number}); 1 is WRF's own cadence, "
            "and `--progress-format off` is how a run says nothing")
    return number


def add_progress_arguments(parser) -> None:
    """Register the four flags on a simulation front door's parser.

    One function so the two runners cannot drift: a door that gains a
    flag here gains it on both, spelled and documented identically.
    """

    group = parser.add_argument_group(
        "per-step progress (WRF-grade; on by default)")
    group.add_argument(
        "--progress-format", choices=PROGRESS_FORMATS, default="text",
        help=("how this run reports its progress.  `text` (the default) "
              "prints one WRF-shaped `Timing for main:` line per model "
              "time step per domain on stdout and ALSO writes the "
              "machine stream to OUTDIR/" + STEP_LOG_FILENAME + "; "
              "`jsonl` writes only that stream, leaving stdout free of "
              "sentences; `off` disables per-step reporting entirely"))
    group.add_argument(
        "--progress-output", default=None, metavar="PATH",
        help=("where the machine stream is written; defaults to "
              "OUTDIR/" + STEP_LOG_FILENAME + ".  Append-only JSONL at "
              + STEP_LOG_SCHEMA + ", one record per printed line, with a "
              "dense `sequence` so a consumer can detect a lost line.  "
              "`-` sends the records to stdout instead of to a file, "
              "which with --progress-format jsonl is a pure record pipe"))
    group.add_argument(
        "--progress-every", type=_positive_cadence, default=1, metavar="N",
        help=("report every Nth model step (default 1, WRF's own "
              "cadence).  The first and last step of every domain are "
              "always reported, and this thins ONLY `step` records -- "
              "output, restart and domain events are never thinned"))
    group.add_argument(
        "--frame-markers", dest="frame_markers", action="store_true",
        default=True,
        help=("publish OUTDIR/" + FRAME_MARKER_DIRNAME + "/<frame>.json "
              "after each history frame is fsynced, self-validated and "
              "renamed into place (the default).  A marker that exists "
              "names a frame that is complete and readable, which is the "
              "signal to poll for instead of racing the writer with a "
              "size check"))
    group.add_argument(
        "--no-frame-markers", dest="frame_markers", action="store_false",
        help="do not publish frame-ready markers")


__all__ = [
    "FIRST_STEP_EXCESS_FACTOR", "FRAME_MARKER_DIRNAME", "FRAME_MARKER_SCHEMA",
    "NEST_EVENTS", "NULL_STEP_LOG", "PROGRESS_FORMATS", "STDOUT_SENTINEL",
    "STEP_LOG_EVENTS", "STEP_LOG_FILENAME", "STEP_LOG_SCHEMA",
    "STEP_LOG_SCHEMAS", "LandingFanout", "ProgressOptions", "StepLog",
    "add_progress_arguments", "centre_latlon", "format_domain_end_line",
    "format_domain_start_line", "format_model_time", "format_nest_line",
    "format_output_line", "format_phase_line", "format_position",
    "format_restart_line", "format_run_end_line", "format_step_line",
    "model_step_log", "open_step_log", "publish_step_log", "read_step_log",
    "write_frame_marker",
]
