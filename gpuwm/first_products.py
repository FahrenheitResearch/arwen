"""Render the first committed frame while the forecast is still running.

Time to first plot is this product's headline number, and for most of a
short-cycle run it is spent on things that have nothing to do with the
picture: the download, the preparation, the model's own spin-up.  By the
time the forecast stage opens, the one frame a reader most wants to see
-- the analysis at t = 0, the state the run was initialised from -- is
already durable on disk.  The history alarm is true at t = 0
(:meth:`gpuwm.core.clock.Clock.history_due`), so that frame is written
before a single step is integrated, and the per-domain writer raises
``output_committed`` the instant it has been fsynced, self-validated and
renamed onto its final name.

Until now that frame waited for the finalize stage, and finalize waited
for the whole forecast.  On a two-hour run that is minutes of a finished
picture sitting on disk unlooked-at.  This module renders it as soon as
it lands, on a worker thread, concurrent with the forecast that produced
it, and the run's event stream carries the wall time from
``plan_accepted`` to the moment the pictures were readable -- the TTFP
number, measured by the engine itself, on every run's receipt.

Four properties make that safe rather than merely fast.

**The renderer never touches the card.**  ``gpuwm render`` is a separate
``python -m gpuwm.cli render`` process which drives the Rust
``rw_wrfbatch`` binary (or matplotlib); it imports cupy as a transitive
dependency of the package and creates no CUDA context -- measured, with
``cuCtxGetCurrent`` returning ``CUDA_ERROR_NOT_INITIALIZED`` after the
render front door is fully imported.  The forecast owns the GPU for its
whole run; this contends for CPU, page cache and disk only.

**The command is the finalize stage's own.**  It is composed by
:func:`gpuwm.go_cli.render_command` out of the same plan dict, and run
with the same working directory and the same ``PYTHONSAFEPATH``
environment, differing only in naming one frame where finalize names all
of them.  Byte-identity between a frame rendered early and the same
frame rendered at finalize is therefore a property of construction
rather than a coincidence -- and it is pinned by a test that renders
both ways and compares the bytes.

**Nothing half-written is ever published.**  The render runs into a
scratch directory underneath the render output, and each picture is
moved onto its final name with :func:`os.replace` only once the
subprocess has exited.  A reader watching the render directory sees a
complete PNG or no PNG, and a finalize render that later writes the same
name cannot collide with a write still in flight.

**Finalize skips only what it can prove is already there.**  The early
render leaves a receipt naming the frame it read and every picture it
wrote, all by sha256.  Finalize drops that frame from its own list only
when the frame still hashes to the recorded digest, every recorded
picture is on disk hashing to its recorded digest, and the product spec
has not changed since.  Anything else -- a moved file, an edited one, a
different ``--products`` -- and the frame is simply rendered again.

Telemetry never fails a run.  Every entry point here swallows its own
exceptions into a ``warning`` event: a forecast that completed must not
be turned into a failure by the picture of its first frame.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gpuwm.render_layout import fs_path, iter_rendered

#: The receipt the early render leaves beside the pictures it published.
FIRST_PRODUCTS_RECEIPT = "first-products.json"

#: Schema id of that receipt.  Read by the finalize stage and by anyone
#: reconstructing what a run published and when.
FIRST_PRODUCTS_SCHEMA = "gpuwm.first-products.v1"

#: ONE definition of the receipt's instant, carried IN the receipt.
#:
#: THE DRIFT THIS CLOSES, measured on both 3080 walks: `gpuwm go` printed
#: "time to first plot 0m 46s (first-products receipt)" while the earliest
#: PNG in the published run tree carried 2m 45s.  Two different quantities
#: were both being called time to first plot -- the instant this render
#: published, and the mtime of whatever picture is in the tree now -- and
#: nothing said which the number was.  A reader who checks the artifact and
#: finds it contradicts the headline stops believing the headline.
FIRST_PLOT_DEFINITION = (
    "published_unix_ms is the wall-clock instant at which every picture "
    "named in 'written' was readable at its final path under the render "
    "directory. A picture found there with a LATER mtime was rewritten "
    "afterwards, and this instant then describes nothing on disk."
)

#: What ``gpuwm render`` draws when nobody passes ``--products``
#: (``gpuwm/render.py``'s own default), spelled once in
#: :mod:`gpuwm.render_layout`.  The early render is asked for the
#: explicit spelling because it is given a command line; the finalize stage
#: leaves the flag off.  Comparing the two literally made every `go` run's
#: receipt look like it had been drawn for a different product set than the
#: stage that could have skipped it, so the skip never happened there.
from gpuwm.render_layout import DEFAULT_RENDER_PRODUCTS  # noqa: E402

#: Slack between a published picture's mtime and the instant stamped for
#: it.  The publish is ``os.replace``, which carries the mtime the RENDERER
#: wrote -- at or before the stamp -- and a coarse-granularity filesystem
#: (FAT rounds to 2 s) can land the recorded mtime either side of it.  This
#: is that granularity and nothing more: a finalize re-render lands minutes
#: later, which is the case this check exists to catch.
_MTIME_SLACK_MS = 2000

#: Where the render runs before its output is published.  A dot-prefixed
#: sibling of the pictures rather than a system temp directory, so the
#: publish below is a rename WITHIN one filesystem and therefore atomic;
#: a cross-volume move is a copy, and a copy can be observed half done.
_SCRATCH_NAME = ".first-products-scratch"

#: How long :meth:`FirstProducts.wait` gives the render before it stops
#: waiting and lets finalize render everything itself.  One frame takes
#: seconds; ten minutes is not a timeout a healthy render approaches, it
#: is the point at which a wedged one must stop holding a finished
#: forecast hostage.
DEFAULT_WAIT_SECONDS = 600.0


def early_render_requested(render_products: Any) -> bool:
    """Whether this run asked for products at all, and so asks early too.

    Default OFF by absence rather than by a second flag.  A plan that
    names no products -- ``render_products`` unset, which is the default
    -- gets exactly the behaviour it had before this module existed, and
    one that spells ``none`` has already said it wants no pictures.  Both
    answers come out of the field that already holds "which products", so
    there is no second switch that can disagree with it.
    """

    if render_products is None:
        return False
    text = str(render_products).strip()
    return bool(text) and text.lower() != "none"


def arm(render_plan: Mapping[str, Any], *,
        report: Callable[[dict], None],
        warn: Callable[..., None],
        runner: Callable[[Sequence[str]],
                         subprocess.CompletedProcess] | None = None,
        slot: threading.Lock | None = None,
        own_group: bool = False,
        report_live: Callable[[dict], None] | None = None
        ) -> FirstProducts | None:
    """The early render for one run, or ``None`` when it asked for none.

    ONE function for the whole decision, because the decision is what
    the doors share: every route learns its render plan somewhere
    different (a run-plan observer, a runner's own argv, a child's
    ``--outdir``), so the construction site cannot be shared, but the
    two steps behind it -- does this run draw at all, and what is the
    trigger armed with -- are the same two steps everywhere.  Each door
    that repeated them was a second place for the answer to
    :func:`early_render_requested` to drift.

    ``render_plan`` is the dict the finalize stage will hand
    :func:`gpuwm.go_cli._render_stage`, so the early render and the late
    one cannot differ in output directory or product spec.  ``report``
    and ``warn`` are the caller's own event stream or, for a process
    that has none, its stdout and stderr.  ``slot`` is the lock this
    render shares with the every-frame render
    (:class:`gpuwm.live_products.LiveProducts`) of the same run, so the
    two never draw at once.  ``own_group`` starts the render in a process
    group of its own (:func:`_run_render`), for a host that ends it
    itself when the run is stopped (:meth:`FirstProducts.halt`).

    ``report_live`` asks for every frame, not the first alone: what comes
    back is a :class:`gpuwm.live_products.LandingRenders`, which IS this
    early render with the every-frame render behind it, reporting each
    later frame there.  A door with no every-frame render of its own (a
    runner with no host, an ensemble member) takes this one; it holds its
    own lock, so ``slot`` is for the early render alone, and
    ``own_group`` reaches both of its renders.
    """

    if not early_render_requested(render_plan.get("render_products")):
        return None
    if report_live is not None:
        from gpuwm.live_products import LandingRenders

        return LandingRenders(render_plan, report=report,
                              report_live=report_live, warn=warn,
                              runner=runner, own_group=own_group)
    return FirstProducts(render_plan, report=report, warn=warn,
                         runner=runner, slot=slot, own_group=own_group)


class FrameHook:
    """A ``progress_callback`` whose only job is the early render.

    The in-process integrator finds ``output_committed`` BY NAME on
    whatever sits in its ``progress_callback`` slot
    (``gpuwm/runtime.py:3831``) and CALLS that same object for every
    step heartbeat (``gpuwm/runtime.py:4155``), so a door with no step
    log of its own still has to hand it something callable to receive
    the landing at all.  This is that object and nothing more: the
    heartbeat is dropped, the landing is forwarded to the trigger
    :func:`arm` returned.

    A door that DOES own a step log already has an object in that slot
    and adds this render as a second consumer of the landing with
    :class:`gpuwm.progress_log.LandingFanout`; the fan-out and this
    both exist because attaching twice silently unhooks the first
    consumer.  Which of the two a door needs is decided by whether it
    has a step log, never by what the early render wants, so they do
    not disagree about anything.

    The forward is guarded like every other landing consumer: the
    picture of a frame must never take down the writer that committed
    it, nor the forecast behind it.
    """

    def __init__(self, trigger: FirstProducts):
        self._trigger = trigger

    def __call__(self, **_heartbeat: Any) -> None:
        """Per-step progress, which this door is not here to report."""

        return None

    def output_committed(self, **event: Any) -> None:
        """One durable frame: hand it to the render, and get out of the way."""

        try:
            self._trigger.frame_committed(**event)
        except Exception:  # noqa: BLE001 - telemetry never fails a run
            pass


def effective_products(render_products: Any) -> str:
    """The product spec a render will actually draw.

    ``None``/empty is not "no products" -- it is ``gpuwm render``'s own
    default -- so the two spellings of the same request compare equal
    instead of looking like two different renders.
    """

    text = "" if render_products is None else str(render_products).strip()
    return text or DEFAULT_RENDER_PRODUCTS


def section_text(render_section: Any) -> str | None:
    """A section line as two records compare it; ``None`` for no line.

    An empty spelling is no line, the same answer ``gpuwm render`` gives
    it, so a receipt written before sections were recorded and a plan
    that names none compare equal.
    """

    text = "" if render_section is None else str(render_section).strip()
    return text or None


def render_without_output_refusal(render_products: Any,
                                  io_mode: Any) -> str | None:
    """Explain an explicit request for pictures without history frames."""
    if early_render_requested(render_products) and io_mode == "none":
        return ("--render-products needs committed history frames, but "
                "--io-mode none writes no frames to render. Use --io-mode "
                "history, or --render-products none to run without pictures.")
    return None


def _sha256_file(path: Path) -> str:
    """The digest of one file, readable at any path length.

    Through ``render_layout.fs_path`` because the receipt's whole job is
    re-finding the pictures the layout placed, and the layout can now
    place them deeper than Windows' ordinary API reaches.  A digest that
    raised there would turn a correctly filed picture into "the picture
    it names is not on disk" and re-render the frame.
    """

    from gpuwm.fetch import sha256_file

    return sha256_file(Path(fs_path(path)))


def finished_pictures(scratch: Path, written: Sequence[Path],
                      returncode: int | None) -> list[Path]:
    """The pictures a render may publish from its scratch.

    All of them when the renderer exited 0.  When it failed partway,
    only the ones its own invocation receipt names
    (:func:`receipted_pictures`): those it reported drawn and ``gpuwm
    render`` filed into the layout.  A renderer that dies partway,
    killed for memory or crashed on one product, can leave the picture
    it was writing under its flat staging name, whole or cut short, and
    nothing reported it.  Measured with the real renderer ended after
    its first picture: the early render published that unreported
    staging file as the frame's only picture.  Such a file goes with the
    scratch; the frame is recorded incomplete and finalize draws it
    again.
    """

    if returncode == 0:
        return list(written)
    names = receipted_pictures(scratch)
    return [path for path in written
            if Path(path).relative_to(scratch).as_posix() in names]


def render_outcome(completed: subprocess.CompletedProcess, *, render_dir: Path,
                   published: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What one frame's render finished as, recorded beside its pictures.

    The renderer draws a frame's products one after another and can fail
    partway: temperature drawn, wind refused, exit 1.  The pictures it did
    draw are good and are kept, but the frame is not done, and a record
    that said so let the end-of-run render skip it and lose wind for good.
    ``complete`` is the renderer's own word (exit 0); ``products`` names
    what it drew, and ``diagnostics`` is its last words when it failed.
    """

    from gpuwm.render_layout import parse_engine_output

    products: list[str] = []
    for entry in published:
        parts = Path(str(entry.get("name") or "")).parts
        if len(parts) >= 4:
            product = parts[-3]
        else:
            parsed = parse_engine_output(parts[-1]) if parts else None
            product = parsed[1] if parsed is not None else "unclassified"
        if product not in products:
            products.append(product)
    code = int(completed.returncode)
    outcome: dict[str, Any] = {"exit_code": code, "complete": code == 0,
                               "products": products}
    if code != 0:
        outcome["diagnostics"] = ((completed.stderr or "")[-2000:]
                                  or (completed.stdout or "")[-2000:])
    return outcome


#: The render subprocess each worker thread is waiting on, keyed by the
#: thread, so a stopped run can end the one it no longer wants
#: (:func:`end_render`).  Only renders spawned by :func:`_run_render`
#: are here; one started in a process group of its own carries
#: :data:`_OWN_GROUP_ATTRIBUTE`.
_RUNNING: dict[int, subprocess.Popen] = {}
_RUNNING_LOCK = threading.Lock()

#: Set on a render process started in a process group of its own.
_OWN_GROUP_ATTRIBUTE = "gpuwm_own_group"

#: How long :func:`end_render` looks for the render of a thread that is
#: alive but has not started its subprocess yet: the halt can land in the
#: few instructions between a worker's last check and its spawn.
_SPAWN_LOOK_SECONDS = 0.5

#: How long :func:`end_render` lets a render answer its interrupt before
#: killing it.  ``gpuwm render`` answers SIGINT by ending the renderer it
#: runs and exiting 130 in well under a second; the desktop's own stop
#: escalates to a kill after 5 s, and a stopped run still has its banner
#: and report to write inside that.
END_RENDER_GRACE_SECONDS = 2.0

#: How long a stop waits for a render thread once its render has been
#: ended (:meth:`FirstProducts.halt`).  The desktop and the terminal kill
#: a run 5 s after asking it to stop, and the stopped run's banner and
#: report are written after this.
HALT_WAIT_SECONDS = 3.0

#: ``gpuwm render``'s exit on an interrupt (the shell's 128 + SIGINT).
_INTERRUPTED_EXIT = 130


def render_was_stopped(returncode: int | None) -> bool:
    """Whether a render ended because it was told to stop.

    Exit 130 is ``gpuwm render`` answering SIGINT; a negative code is a
    render killed by a signal.  Either way the scratch holds whatever the
    renderer had staged at that instant (flat staging names beside
    half-organised folders), which is not a drawn frame.
    """

    return returncode is not None and (returncode < 0
                                       or returncode == _INTERRUPTED_EXIT)


def _own_group_options() -> dict[str, Any]:
    """``Popen`` options that start a process in a process group of its own."""

    if os.name == "nt":
        return {"creationflags": getattr(subprocess,
                                         "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def _run_render(command: Sequence[str], *,
                own_group: bool = False,
                env_overrides: Mapping[str, str] | None = None
                ) -> subprocess.CompletedProcess:
    """Spawn the render exactly as the finalize stage spawns it.

    Same cwd and stage environment, with optional resource limits for
    concurrent live frames. The request and picture settings stay the same
    as finalize's so its pictures can be compared byte for byte.

    What :func:`subprocess.run` does, with the process recorded against
    the calling thread while it runs, so :func:`end_render` can end it.

    ``own_group`` starts the render in a process group of its own, for a
    host that stops it itself.  THE BREAKAGE: the
    desktop and terminal Stop send SIGINT to the run's whole process
    group, so a render in that group died mid-organisation, with its
    pictures still under the renderer's flat staging names, and the run
    then published and counted those files as pictures.  Out of the
    group, the stop reaches the run alone; the run decides that nothing
    more is published and then ends the render (:func:`end_render`).
    """

    from gpuwm.go_cli import _stage_cwd, _stage_env

    environment = _stage_env()
    if env_overrides:
        environment.update(env_overrides)
    process = subprocess.Popen(
        list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, errors="replace", cwd=str(_stage_cwd()),
        env=environment, **(_own_group_options() if own_group else {}))
    setattr(process, _OWN_GROUP_ATTRIBUTE, bool(own_group))
    ident = threading.get_ident()
    with _RUNNING_LOCK:
        _RUNNING[ident] = process
    try:
        stdout, stderr = process.communicate()
    except BaseException:
        _kill(process, own_group=own_group)
        process.wait()
        raise
    finally:
        with _RUNNING_LOCK:
            _RUNNING.pop(ident, None)
    # Relay warnings from this second capture into the forecast log, even
    # on success: finalize may skip frames already published early/live.
    from gpuwm.rustwx import relay_native_warnings

    relay_native_warnings(stderr)
    return subprocess.CompletedProcess(process.args, process.returncode,
                                       stdout, stderr)


def _kill(process: subprocess.Popen, *, own_group: bool) -> None:
    """Kill a render now, with every process it started when it has a group.

    A render in a group of its own takes the renderer binary with it:
    killing only ``gpuwm render`` would leave ``rw_wrfbatch`` drawing
    into a folder nobody reads.  In the caller's group only the render
    itself is signalled, because that group is the caller's too.

    On Windows the render's whole process tree goes, group or not:
    ``taskkill /T`` follows the processes a render started, never its
    group, so it cannot reach the caller.  THE BREAKAGE: killing ``gpuwm
    render`` alone left its renderer drawing with pictures open in the
    folder, and a folder with a file open in it does not move (a forecast
    attempt set aside before the forecast runs again).
    """

    try:
        if own_group and os.name == "posix":
            import signal

            os.killpg(process.pid, signal.SIGKILL)
        elif os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if process.poll() is None:
                process.kill()
        else:
            process.kill()
    except OSError:
        pass


class WorkerEnd:
    """Whether a render worker thread has returned, told by the worker.

    NOT ``Thread.is_alive()``.  On CPython 3.11 and 3.12 an exception a
    signal handler raises while ``Thread.join()`` waits -- the
    ``KeyboardInterrupt`` of a Ctrl-C, the ``ChildStopped`` a SIGTERM
    becomes on ``gpuwm downscale`` -- marks the joined thread stopped
    while it is still running, and ``is_alive()`` answers ``False`` from
    then on.  A stop lands in exactly such a join whenever the run is
    waiting for a render: the finalize stage waiting for the last
    frame's every-frame render, a runner waiting for its early render.
    The stop then read the worker as finished, never ended its render,
    and tidied the folder under it.  Measured on a real 250 m child
    stopped 1 s into its finalize stage: ``gpuwm render`` and
    ``rw_wrfbatch`` kept drawing the last frame after the child had
    exited, and left 65 to 69 staging pictures in the picture folder.

    The worker sets this when its target returns (:meth:`run`), and
    every wait and every liveness question on the stop path asks it
    instead.  ``Event.wait`` interrupted the same way leaves nothing
    wrong behind.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def run(self, target: Callable[..., Any], *args: Any,
            **kwargs: Any) -> None:
        """Run ``target`` here, marking its end however it ends."""

        try:
            target(*args, **kwargs)
        finally:
            self._event.set()

    @property
    def ended(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None) -> bool:
        """Wait for the worker to return; whether it has."""

        return self._event.wait(timeout)


def end_render(thread: threading.Thread | None, *,
               grace: float = END_RENDER_GRACE_SECONDS,
               ended: WorkerEnd | None = None) -> bool:
    """End the render subprocess ``thread`` is waiting on, if there is one.

    For a run that was stopped: the picture being drawn is not wanted,
    and a render left running would outlive the run that asked for it.
    On POSIX the render is sent SIGINT, which ``gpuwm render`` answers
    by ending the renderer process it runs and exiting 130; a render
    that ignores it (a run started with SIGINT ignored passes that on)
    is killed after ``grace`` seconds.  Elsewhere it is killed with every
    process it started (:func:`_kill`).

    A render started in a group of its own (:func:`_run_render`) is
    signalled as a group, so the renderer binary under it is told too,
    and killed as a group.  A thread that is alive but has not started
    its subprocess yet is watched for a moment, because a halt can land
    between a worker's last check and its spawn.  ``ended`` is the
    worker's own word on whether it has returned (:class:`WorkerEnd`);
    without it the thread's ``is_alive()`` is asked, which a stop can
    have made wrong.

    Returns whether a running render was found.  Never raises.
    """

    ident = getattr(thread, "ident", None)
    if ident is None:
        return False

    def returned() -> bool:
        return ended.ended if ended is not None else not thread.is_alive()

    deadline = time.monotonic() + _SPAWN_LOOK_SECONDS
    while True:
        with _RUNNING_LOCK:
            running = _RUNNING.get(ident)
        if (running is not None or returned()
                or time.monotonic() >= deadline):
            break
        time.sleep(0.02)
    if running is None:
        return False
    process = running
    own_group = bool(getattr(process, _OWN_GROUP_ATTRIBUTE, False))
    if process.poll() is not None:
        return False
    try:
        if os.name == "posix":
            import signal

            if own_group:
                os.killpg(process.pid, signal.SIGINT)
            else:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(grace)
            except subprocess.TimeoutExpired:
                _kill(process, own_group=own_group)
        else:
            _kill(process, own_group=own_group)
    except OSError:
        pass
    return True


#: How long :func:`halt_renders_and_wait` waits for a render that outlived
#: its halt to exit once killed, and then for each render thread.
REAP_WAIT_SECONDS = 10.0


def _render_processes(renders) -> list[subprocess.Popen]:
    """The render processes ``renders``' threads are waiting on now."""

    idents = {thread.ident for thread, _ in renders.render_threads()
              if thread.ident is not None}
    with _RUNNING_LOCK:
        return [process for ident, process in _RUNNING.items()
                if ident in idents and process.poll() is None]


def halt_renders_and_wait(renders, *,
                          timeout: float | None = HALT_WAIT_SECONDS,
                          reap_seconds: float = REAP_WAIT_SECONDS) -> bool:
    """Halt ``renders`` and return once none of their processes runs.

    For a run about to move the folder its renders read frames from and
    draw pictures into (a forecast attempt set aside before the forecast
    runs again).  The halt ends the render in flight and waits
    ``timeout`` for it (:meth:`FirstProducts.halt`); a render still
    running after that is killed (:func:`_kill`) and waited for here, and
    so is each render thread.  THE BREAKAGE: a halt waits only a bounded
    time and then returns with the render possibly still drawing, and on
    Windows a folder with a picture or a frame open in it does not move.

    ``renders`` is a :class:`FirstProducts`, a
    :class:`gpuwm.live_products.LiveProducts` or a
    :class:`gpuwm.live_products.LandingRenders`.  Returns whether every
    render process exited and every render thread returned.
    """

    renders.halt(timeout)
    for process in _render_processes(renders):
        if process.poll() is None:
            _kill(process, own_group=bool(
                getattr(process, _OWN_GROUP_ATTRIBUTE, False)))
        try:
            process.wait(reap_seconds)
        except subprocess.TimeoutExpired:
            pass
    threads = renders.render_threads()
    for _, ended in threads:
        ended.wait(reap_seconds)
    return (all(ended.ended for _, ended in threads)
            and not _render_processes(renders))


class FirstProducts:
    """The early render of one frame, dispatched and later collected.

    Constructed with the very dict the finalize stage will hand
    :func:`gpuwm.go_cli._render_stage`, so the two cannot drift apart in
    output directory or product spec.

    ``report`` is called with the finished receipt on the worker thread
    and is what emits ``first_products_ready``; ``warn`` is called with
    ``(code, message, **fields)`` for every way this can decline to
    publish.  Both are supplied by the observer, which owns the event
    stream and the clock this run started on.
    """

    def __init__(self, render_plan: Mapping[str, Any], *,
                 report: Callable[[dict], None],
                 warn: Callable[..., None],
                 runner: Callable[[Sequence[str]],
                                  subprocess.CompletedProcess] | None = None,
                 slot: threading.Lock | None = None,
                 own_group: bool = False):
        self._plan = dict(render_plan)
        self._report = report
        self._warn = warn
        self._runner = _run_render if runner is None else runner
        #: Whether the render is started in a process group of its own,
        #: which only a host that ends it on a stop asks for
        #: (:meth:`halt`).  Passed to the runner as ``own_group``.
        self._own_group = bool(own_group)
        self._lock = threading.Lock()
        #: Held for the whole render: shared with the every-frame render
        #: (:mod:`gpuwm.live_products`) to bound their combined concurrency,
        #: whichever grid's frame is committed first.
        self._slot = threading.Lock() if slot is None else slot
        self._thread: threading.Thread | None = None
        #: Set when the render thread returns; asked instead of the
        #: thread's ``is_alive()`` (:class:`WorkerEnd`).
        self._ended = WorkerEnd()
        self._receipt: dict[str, Any] | None = None
        #: Held while the drawn frame is moved into the folder and its
        #: receipt written, so a halt never lands between a picture and
        #: the receipt that names it.
        self._publish = threading.Lock()
        self._abandoned = False

    # -- what this was armed with -------------------------------------

    @property
    def render_dir(self) -> Path:
        return Path(self._plan["render"])

    @property
    def render_products(self) -> str:
        return str(self._plan.get("render_products") or "")

    @property
    def render_section(self) -> str | None:
        """The line the section products are cut along, or ``None``."""

        return section_text(self._plan.get("render_section"))

    @property
    def receipt(self) -> dict[str, Any] | None:
        """The published receipt, or ``None`` until one exists."""

        return self._receipt

    @property
    def dispatched(self) -> bool:
        return self._thread is not None

    def render_threads(self) -> list[tuple[threading.Thread, WorkerEnd]]:
        """The threads that start this render's processes, with their ends
        (:func:`halt_renders_and_wait`)."""

        thread = self._thread
        return [] if thread is None else [(thread, self._ended)]

    # -- the hook the observer calls ----------------------------------

    def frame_committed(self, *, domain: int, valid_time: Any,
                        path: Any) -> bool:
        """Dispatch the render of one frame; return whether this one won.

        Called from :meth:`gpuwm.runplan.RunObserver.output_committed`,
        which on the prepared routes runs on the per-domain wrfout
        writer's own daemon thread with a one-deep admission queue behind
        it.  So this must return immediately and must never raise: it
        starts a thread and gets out of the way.

        Only the first caller wins.  Later frames are the forecast doing
        its job and are the finalize stage's business, not this one's.
        """

        with self._lock:
            if self._thread is not None:
                return False
            thread = threading.Thread(
                target=self._ended.run, args=(self._guarded_render,),
                name="gpuwm-first-products", daemon=True,
                kwargs={"domain": int(domain), "valid_time": valid_time,
                        "frame": Path(path)})
            self._thread = thread
        thread.start()
        return True

    def wait(self, timeout: float | None = DEFAULT_WAIT_SECONDS
             ) -> dict[str, Any] | None:
        """Join the render and return its receipt, or ``None``.

        The finalize stage calls this before it decides what to render.
        ``None`` means "assume nothing was published": no frame was ever
        dispatched, or the render declined, or it is still going after
        ``timeout``.  In every one of those cases finalize renders the
        whole set exactly as it always did, which is the safe answer.

        A render still going after ``timeout`` is given up on for good:
        it is told to publish nothing and its process is ended
        (:meth:`_abandon`).  THE BREAKAGE: it used to be left running.
        Its late finish then moved its pictures over the ones finalize
        had just drawn into the same folder and wrote a receipt claiming
        them, and its render process outlived the run that started it.
        """

        thread = self._thread
        if thread is None:
            return None
        if not self._ended.wait(timeout):
            self._abandon()
            if self._receipt is not None:
                # A publish that was already moving files when the wait
                # ran out finished first, under the lock the abandon
                # takes, so its pictures and receipt are whole.
                return self._receipt
            self._warn(
                "first_products_timeout",
                "the early render of the first frame was still running "
                f"after {timeout:.0f} s, so it was ended and publishes "
                "nothing; the finalize stage is rendering every frame "
                "itself",
                render_dir=str(self.render_dir))
            return None
        return self._receipt

    def _abandon(self, timeout: float | None = None) -> bool:
        """Give up on this early render alone: nothing more is published and its render is ended.

        :meth:`halt` for this render only.  A subclass's ``halt`` may stop
        more than this render (:class:`gpuwm.live_products.LandingRenders`
        also closes the every-frame queue, whose worker can be the very
        caller waiting on this render), so a wait that runs out calls
        this, never ``self.halt``.
        """

        return FirstProducts.halt(
            self, HALT_WAIT_SECONDS if timeout is None else timeout)

    def halt(self, timeout: float | None = HALT_WAIT_SECONDS) -> bool:
        """Stop now, for a run that was stopped.  ``True`` once nothing runs.

        Nothing is published after this returns: a publish already moving
        files finishes first, under the same lock, so every picture it
        moved has its receipt beside it.  The render in flight is ended
        rather than waited for (:func:`end_render`), because the desktop
        kills a stopped run 5 s after asking it to stop and the run still
        has its banner and report to write.  ``False`` means the render
        thread was still alive after ``timeout``: its render may still be
        drawing, so a caller must not tidy the folder under it.

        Safe to call more than once, and before anything was dispatched.
        """

        with self._publish:
            self._abandoned = True
        thread = self._thread
        if thread is None:
            return True
        if not self._ended.ended:
            end_render(thread, ended=self._ended)
            self._ended.wait(timeout)
        return self._ended.ended

    def _requests_windows(self) -> bool:
        """Whether the request holds any window, from the renderer's listing.

        Asked only for a resumed run's first frame that has saved context
        to import.  When the listing cannot be had the answer is yes, and
        the context goes along as before: time, never a picture lost.
        """

        from gpuwm import live_products

        try:
            return live_products.requests_windows(
                self.render_products, live_products.catalog_windowed_slugs)
        except Exception:  # noqa: BLE001 - import the hour as before
            return True

    # -- the worker ---------------------------------------------------

    def _guarded_render(self, **kwargs: Any) -> None:
        try:
            with self._slot:
                if self._abandoned:
                    # Halted before this render took its turn.
                    return
                self._render(**kwargs)
        except BaseException as error:  # noqa: BLE001 - never fail a run
            self._warn(
                "first_products_failed",
                "the early render of the first committed frame raised "
                f"{type(error).__name__}: {error}; the finalize stage "
                "will render it as usual",
                frame=str(kwargs.get("frame")))

    def _render(self, *, domain: int, valid_time: Any, frame: Path) -> None:
        from gpuwm.go_cli import render_command
        from gpuwm.render import announce_missing_basemap
        from gpuwm.restart_render import history_before_restart, hour_before
        from gpuwm.rustwx import COMMAND_LINE_BUDGET

        started = time.perf_counter()
        render_dir = self.render_dir
        # A picture with no coastlines is still a picture, so the render
        # below succeeds either way; the run has to be told here, once.
        announce_missing_basemap(self._warn, render_dir, stage="as-drawn")
        scratch = render_dir / _SCRATCH_NAME
        # Create-only for the scratch: a leftover from a previous run in
        # the same directory would be published as though this render had
        # written it.
        if scratch.exists():
            # ``descend=True``: what is being removed is a TREE, and the
            # renderer's own store underneath it carries names long
            # enough that an ordinary spelling of this root makes the
            # walk fail at the first deep entry -- measured, with
            # WinError 3 on a store path of about 300 characters.
            shutil.rmtree(fs_path(scratch, descend=True))
        scratch.mkdir(parents=True)
        try:
            # A resumed run's first frame closes an hour its checkpoint
            # opened: the frames of that hour, saved before the checkpoint,
            # go along as context, so its rainfall is drawn now.
            context = hour_before(
                frame, history_before_restart(self._plan.get("restart")))
            # And only for a request that holds a window, the one thing a
            # baseline buys, as the every-frame render and the end-of-run
            # batch already decide it (live_products.requests_windows).
            # THE BREAKAGE: a 12 km GFS run resumed from its hour-1
            # checkpoint and asked for composite reflectivity alone drew
            # its first new hour with the saved hour-1 frame imported
            # beside it (the renderer's receipt listed it as a context
            # input), for a picture that reads nothing from it: the same
            # import the live pass stopped paying at every whole hour of a
            # 3 km CONUS run (77 to 82 s against 14 s).
            if context and not self._requests_windows():
                context = []
            command = render_command(
                {**self._plan, "render": scratch}, [frame], context_frames=context)
            if len(subprocess.list2cmdline(command)) > COMMAND_LINE_BUDGET:
                command = render_command(
                    {**self._plan, "render": scratch}, [frame], context_frames=context,
                    inputs_file=scratch / "restart-render-inputs.json")
            completed = (self._runner(command, own_group=True)
                         if self._own_group else self._runner(command))
            if self._abandoned:
                # Halted while it drew: the halt ended this render, and
                # nothing it left is published.
                return
            if render_was_stopped(completed.returncode):
                # A render told to stop by someone else (a Ctrl-C that
                # reached its process group) left whatever it had staged
                # at that instant: pictures under the renderer's flat
                # staging names beside half-filed folders, and no receipt.
                # That is not a drawn frame, and publishing it is how a
                # stopped child's folder came to hold 155 flat files
                # counted as pictures.
                self._warn(
                    "first_products_stopped",
                    f"drawing the first frame ({frame.name}) was stopped "
                    f"before it finished (render exited "
                    f"{completed.returncode}), so none of it was "
                    "published; the frame is on disk and draws with "
                    "gpuwm render",
                    frame=str(frame))
                return
            written = finished_pictures(scratch, iter_rendered(scratch),
                                        completed.returncode)
            if not written:
                # Not a failure.  The cold-start frame carries no
                # REFL_10CM -- no microphysics call precedes it, a
                # registered deviation -- so a run whose only product is
                # reflectivity legitimately has nothing to draw yet.  The
                # frame is left unclaimed and finalize renders it with
                # the rest, which is where it will be skipped for the
                # same accurate reason.
                self._warn(
                    "first_products_empty",
                    "the first committed frame produced no picture for "
                    f"--products {self.render_products!r} (render exited "
                    f"{completed.returncode}), so nothing was published "
                    "early and the finalize stage is unchanged",
                    frame=str(frame),
                    stdout=(completed.stdout or "")[-2000:],
                    stderr=(completed.stderr or "")[-2000:])
                return
            with self._publish:
                if self._abandoned:
                    # Halted while the render was exiting: a stopped run
                    # publishes nothing after its halt.
                    return
                render_dir.mkdir(parents=True, exist_ok=True)
                published: list[dict[str, Any]] = []
                paths: list[Path] = []
                for source in written:
                    # The RELATIVE path, not the bare name: since 2.5.0
                    # the render writes a tree (domain/product/valid-day,
                    # see gpuwm.render_layout) and flattening it here
                    # would publish the early frame into a different
                    # directory from the one finalize renders the rest
                    # into -- two layouts in one run, from one command.
                    relative = source.relative_to(scratch)
                    target = render_dir / relative
                    # Both sides through render_layout.fs_path: the
                    # scratch is one folder DEEPER than the published
                    # tree, so it is the first place the layout's own
                    # path length runs into Windows' MAX_PATH, and a
                    # publish that failed there would drop a drawn
                    # picture on the floor.
                    spelled = fs_path(target)
                    Path(spelled).parent.mkdir(parents=True, exist_ok=True)
                    # Atomic within the volume: a reader tailing the
                    # render directory sees a whole PNG or no PNG, and a
                    # later finalize write of the same name cannot land
                    # on top of a write still in flight.
                    os.replace(fs_path(source), spelled)
                    # Recorded relative to the render directory, in
                    # posix spelling, so `render_dir / entry["name"]`
                    # re-finds it on any platform when finalize re-checks
                    # the digests.
                    published.append({
                        "name": relative.as_posix(),
                        "sha256": _sha256_file(target),
                        "size_bytes": Path(spelled).stat().st_size})
                    paths.append(target)
                published_unix_ms = int(time.time() * 1000)
                elapsed = time.perf_counter() - started
                # The render CLI also wrote its exact invocation receipt
                # in scratch.  Preserve/rebase it before cleanup so final
                # aggregation retains these early images and their native
                # skip/failure outcomes.
                from gpuwm import render_georef
                from gpuwm.render_receipts import relocate_invocations
                # The frame's map record goes with its pictures, merged
                # into the folder's own rather than left in the scratch it
                # drew in.
                render_georef.fold(render_dir, render_georef.read(
                    scratch / render_georef.GEOREF_FILENAME))
                relocate_invocations(scratch, render_dir, published)
                announced = {
                    "schema": FIRST_PRODUCTS_SCHEMA,
                    # One definition, written down where the number is, so
                    # a consumer never has to guess which quantity it is
                    # holding.
                    "measures": FIRST_PLOT_DEFINITION,
                    # THE TIME-TO-FIRST-PLOT INSTANT, on the wall clock.
                    #
                    # The report hook below carries seconds-from-start,
                    # which is the number a HOST wants because the host
                    # owns the start.  A caller that did not host this
                    # render -- `gpuwm go`, which asks the runner
                    # subprocess to do it -- has only the receipt, and a
                    # duration measured from a start it cannot see is not
                    # a number it can use.  So the receipt carries the
                    # absolute instant and every consumer subtracts its
                    # own launch from it.
                    "published_unix_ms": published_unix_ms,
                    "frame": str(frame),
                    "domain": int(domain),
                    "valid_time": (valid_time.isoformat()
                                   if hasattr(valid_time, "isoformat")
                                   else str(valid_time)),
                    "render_products": self.render_products,
                    # The line the pictures' sections were cut along;
                    # None when the render cut none.
                    "render_section": self.render_section,
                    "command": [str(part) for part in command],
                    "written": published,
                    "render_seconds": round(elapsed, 6),
                    # A renderer that failed partway leaves the frame
                    # incomplete: its pictures are kept, and finalize
                    # draws the frame again (render_outcome).
                    **render_outcome(completed, render_dir=render_dir,
                                     published=published),
                }
                if not announced["complete"]:
                    self._warn(
                        "first_products_incomplete",
                        f"the early render of {frame.name} exited "
                        f"{announced['exit_code']} after drawing "
                        f"{len(published)} picture(s); they are kept, and "
                        "the finalize stage draws the frame again for the "
                        "rest",
                        frame=str(frame), exit_code=announced["exit_code"],
                        products=announced["products"],
                        diagnostics=announced["diagnostics"])
                # The event goes out HERE, before the frame is digested
                # and before the receipt is written.  Both of those are
                # the finalize stage's business and can wait; the event
                # is the TTFP number, and its whole value is marking the
                # instant the pictures became readable.  Hashing a 362 MB
                # history frame first would have put about a second of
                # bookkeeping inside the number.  All of it stays under
                # the publish lock, so a halt never finds a picture here
                # without the receipt that names it.
                self._report({**announced,
                              "paths": [str(path) for path in paths]})
                receipt = {**announced, "frame_sha256": _sha256_file(frame)}
                self._receipt = receipt
                _write_receipt(render_dir / FIRST_PRODUCTS_RECEIPT, receipt)
        finally:
            shutil.rmtree(fs_path(scratch, descend=True), ignore_errors=True)
            # And the renderer's OWN scratch, which it parks beside the
            # directory it was asked to deliver into
            # (`gpuwm.render.scratch_root_for`).  That sibling is
            # `<render>/.first-products-scratch.render-scratch` here --
            # INSIDE the picture tree, because the directory this render
            # delivers into is itself inside it -- so leaving it turned
            # every early render into about 11 MB of working files
            # published beside the pictures.
            from gpuwm.render import SCRATCH_SUFFIX

            shutil.rmtree(
                fs_path(scratch.with_name(scratch.name + SCRATCH_SUFFIX),
                        descend=True),
                ignore_errors=True)


#: The plain file a run that did not finish leaves at the top of its
#: render directory, next to the pictures it drew before it stopped.
DID_NOT_FINISH_BANNER = "DID-NOT-FINISH.txt"

#: The one string the banner, the render summary and the report all
#: carry, so a reader keys on one value rather than on three spellings
#: of the same state.
DID_NOT_FINISH_STATUS = "did-not-finish"

#: How many frame names the banner lists before it starts counting the
#: rest.  A stopped child can have hundreds, and a banner nobody reaches
#: the bottom of says nothing.
_BANNER_FRAMES = 40


def _seconds_text(value: Any) -> str:
    """A model second in a fixed format, with thousands separators.

    NEVER exponential.  ``%g`` switches to a mantissa at a million, and
    a model second passes a million inside a fortnight of forecast, so
    the one sentence this file exists for printed ``1.08e+06`` to a
    reader who had just lost a run.  Three decimals survive a
    sub-second stop and trailing zeros go, so a whole second reads as a
    whole number.
    """

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:,.3f}".rstrip("0").rstrip(".") or "0"


def _count_text(value: Any) -> str:
    """A step count as an integer, with thousands separators.

    A step is a whole number and has no mantissa to print.  ``%g`` gave
    one to every count past a million, which a step of 0.1 s reaches
    inside a day and a half of forecast.
    """

    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return str(value)
    return f"{number:,d}"


def _stop_sentence(stopped: Mapping[str, Any] | None) -> str:
    """Where the forecast got to, or that nobody recorded it.

    A run refused before its first step has no model second to quote,
    and a banner that invented one would be a number nobody measured.
    """

    if not stopped:
        return ("The forecast stopped before it finished.  How far it got "
                "was not recorded.")
    model = stopped.get("model_seconds")
    total = stopped.get("run_seconds")
    step = stopped.get("step")
    steps = stopped.get("total_steps")
    reached = (f"model second {_seconds_text(model)} of "
               f"{_seconds_text(total)}"
               if model is not None and total is not None
               else f"model second {_seconds_text(model)}"
               if model is not None
               else "an unrecorded model second")
    after = (f", after step {_count_text(step)} of {_count_text(steps)}"
             if step is not None and steps is not None
             else f", after step {_count_text(step)}"
             if step is not None else "")
    return f"The forecast stopped at {reached}{after}."


def banner_text(*, why: str, stopped: Mapping[str, Any] | None,
                frames: Sequence[Any], pictures: int | None,
                pictures_error: str | None = None,
                requested: bool = False) -> str:
    """The words the banner says.  One writer, so every route agrees.

    Four facts, in the order a reader needs them: how far the forecast
    got, why it stopped, which frames exist, and that every picture in
    this folder is from before the stop.  The last one is the sentence
    the whole file exists for: a picture drawn early is a picture of a
    healthy forecast, and showing it beside no explanation is how a
    reader concludes the run was fine.

    THREE picture outcomes, not two, the same three the failed-render
    capsule states: a count, an empty folder, and a folder that could
    not be LISTED, which arrives as ``pictures=None`` with the error in
    ``pictures_error``.  A tree nothing could read is not a tree with
    nothing in it, and printing the second over the first is what tells
    a reader who still has their pictures that they have none.

    ``requested`` is a run somebody stopped (a Stop button, a Ctrl-C, a
    ``kill -TERM``), and the first line says so: the same banner over a
    stop and over a blow-up read as the same event to a reader, and only
    one of them is something to look into.
    """

    names = [str(frame) for frame in frames]
    listed = names[:_BANNER_FRAMES]
    dropped = len(names) - len(listed)
    # The WHOLE clause agrees with the count, verb and pronoun included.
    # A number made singular over a sentence that kept its plural reads
    # exactly as carelessly as the parenthesised plural it replaced, on
    # a file whose reader has just lost a forecast.
    source = ("" if not names else " (the frame below)"
              if len(names) == 1 else " (the frames below)")
    if pictures is None:
        # THE TREE COULD NOT BE READ.  Not a count, so no sentence here
        # may claim one: what is in this folder is unknown, and the
        # error that made it unknown is quoted so a reader can clear it
        # and look for themselves.
        detail = str(pictures_error or "").strip()
        held = ("This folder could not be listed"
                + (f" ({detail})" if detail else "")
                + ", so whether any picture had been drawn before the "
                "forecast stopped is not known from here.")
    elif not pictures:
        # STOPS here.  The frame block below is the one that says
        # whether anything was named, and it handles both cases; a
        # clause that promised a list stood over "No frames were
        # written before the stop." whenever the run also committed
        # nothing, and the two sentences contradicted each other in the
        # one case they share.
        held = ("No pictures are in this folder: none had been drawn "
                "before the forecast stopped.")
    elif pictures == 1:
        # "Of a frame written before", not "drawn before": a child that
        # failed on its own finishes drawing the frames it had already
        # written (gpuwm.live_products), so a picture can be drawn a few
        # seconds after the stop.  What it can never show is the state
        # the forecast stopped in.
        one = (" (one of the frames below)" if len(names) > 1 else source)
        held = ("1 picture is in this folder.  It is of a frame written "
                f"before the forecast stopped{one}, and it does not show "
                "the state the forecast stopped in.")
    else:
        held = (f"{pictures} pictures are in this folder.  Every one of them "
                "is of a frame written before the forecast "
                f"stopped{source}, and no picture shows the state the "
                "forecast stopped in.")
    lines = [
        ("THIS FORECAST WAS STOPPED BEFORE IT FINISHED" if requested
         else "THIS FORECAST DID NOT FINISH"),
        "",
        _stop_sentence(stopped),
        "",
        f"Why it stopped: {str(why).strip() or 'not recorded'}",
        "",
        held,
        "",
    ]
    if names:
        lines.append("1 frame was written before the stop:" if len(names) == 1
                     else f"{len(names)} frames were written before the stop:")
        lines.extend(f"  {name}" for name in listed)
        if dropped:
            lines.append(f"  ... and {dropped} more")
    else:
        lines.append("No frames were written before the stop.")
    if names and pictures:
        # ONLY where there are frames.  Over a run that committed none,
        # "The frames are kept too" denies the line directly above it,
        # which is the same contradiction the zero-picture clause used
        # to carry, one paragraph further down.
        lines.extend([
            "",
            ("The frames are kept too, and they can be drawn again at any "
             "time: this folder is what the run had already drawn for "
             "itself, not all it could show."),
        ])
    elif names:
        # FRAMES AND NO PICTURE.  "The frames are kept TOO" and "this
        # folder is what the run had already drawn" are both about
        # pictures that are not here: over a folder holding frames and
        # no picture the paragraph read as though the pictures were in
        # it, which is the whole failure this banner exists to prevent.
        # A count that could not be taken takes this arm as well: it
        # promises nothing about what the folder holds.
        lines.extend([
            "",
            ("The frames are kept, and pictures can be drawn from them "
             "at any time: the frames listed above are what this run "
             "got far enough to write."),
        ])
    lines.append("")
    return "\n".join(lines)


def _spelling_of(root: Path, walk_root: Path, filename) -> str:
    """The path an error names, back in the spelling the caller typed.

    The walk goes through :func:`gpuwm.render_layout.fs_path`, which on
    Windows is an extended-length spelling of the same directory; a
    reader comparing the error against the path they passed must not be
    handed a prefix they never typed.  Same file either way.
    """

    if not filename:
        return str(root)
    named = Path(filename)
    if named == walk_root:
        return str(root)
    try:
        return str(root / named.relative_to(walk_root))
    except ValueError:
        return str(named)


def count_pictures(render_dir) -> tuple[int | None, str | None]:
    """``(pictures on disk, why they could not be counted)``.

    ONE counter behind every sentence this tree writes about a stopped
    run's pictures -- the banner, the render summary, the report's
    ``products`` block and the failed-render capsule, which reads it
    through :meth:`gpuwm.offline_child_run._ChildProgress.pictures_drawn`
    -- so those four documents cannot disagree about one tree.

    WHAT BREAKAGE THIS PREVENTS (gate law).  Counting through
    :func:`gpuwm.render_layout.iter_rendered` cannot tell an empty tree
    from an unreadable one.  That reader answers ``[]`` for a root that
    is not a directory and walks with :meth:`pathlib.Path.rglob`, which
    swallows a refused ``scandir``, so a picture folder behind a
    permission wall, on a mount that dropped, or at a path that is a
    regular file came back as a count of zero and printed as "the early
    render had not published one before the forecast stopped" -- the one
    sentence that sends a reader off to re-draw a whole child whose
    pictures are sitting behind the error.  Walking with ``onerror`` is
    what makes a refusal visible instead of silent.

    THREE outcomes, therefore: a count, an empty folder, and a folder
    that could not be LISTED, which comes back as ``(None, the error)``.
    A directory that is simply not there is still the empty case: a
    render that never created its output directory drew nothing, and
    that is a reading rather than a failure to read.

    Nothing raises.  This is a number for a sentence in a refusal that
    is already being raised.
    """

    root = Path(render_dir)
    walk_root = Path(fs_path(root, descend=True))
    refused: list[OSError] = []
    drawn = 0
    for _directory, subdirectories, names in os.walk(
            walk_root, onerror=refused.append):
        # The early render draws into a dotted scratch directory and
        # moves each picture onto its final name when the subprocess
        # exits, so a file still in there is not a published picture.
        # It is the one difference between this walk and a naive one,
        # and :func:`gpuwm.render_layout.iter_rendered` makes it too.
        subdirectories[:] = [name for name in subdirectories
                             if not name.startswith(".")]
        drawn += sum(1 for name in names if name.endswith(".png"))
    unreadable = [error for error in refused
                  if not isinstance(error, FileNotFoundError)]
    if unreadable:
        error = unreadable[0]
        return None, (f"{_spelling_of(root, walk_root, error.filename)}: "
                      f"{error.strerror or error}")
    return drawn, None


#: The long-path prefixes :func:`gpuwm.render_layout.fs_path` puts on a
#: Windows path; a receipt can record either spelling of one file.
_VERBATIM_UNC = "\\\\?\\UNC\\"
_VERBATIM = "\\\\?\\"


def _plain(text: str) -> str:
    """``text`` without a Windows extended-length prefix."""

    if text.startswith(_VERBATIM_UNC):
        return "\\\\" + text[len(_VERBATIM_UNC):]
    if text.startswith(_VERBATIM):
        return text[len(_VERBATIM):]
    return text


def _named_inside(root: Path, recorded: Any) -> str | None:
    """A receipt's picture path, relative to ``root`` in posix spelling."""

    if not recorded:
        return None
    path = Path(_plain(str(recorded)))
    bases = [Path(_plain(str(root.resolve()))), Path(_plain(str(root)))]
    if not path.is_absolute():
        return path.as_posix()
    for base in bases:
        try:
            return path.resolve().relative_to(base.resolve()).as_posix()
        except (ValueError, OSError):
            continue
    return None


def receipted_pictures(render_dir) -> set[str]:
    """Every picture a render receipt in ``render_dir`` names.

    Relative to the directory, in posix spelling.  Three receipts vouch
    for a picture, and each is written only once the pictures it names
    are whole and in place: ``gpuwm render``'s own invocation receipt in
    ``.render-receipts/`` (written after the render has filed every
    picture into the layout, and moved beside them when the early or the
    every-frame render publishes), :data:`FIRST_PRODUCTS_RECEIPT` and
    ``live-products.json``.  A picture no receipt names is the output of
    a render that did not finish.

    Nothing raises: an unreadable receipt vouches for nothing.
    """

    from gpuwm import render_receipts
    from gpuwm.live_products import read_receipt as read_live_receipt

    root = Path(render_dir)
    names: set[str] = set()
    first = read_receipt(root)
    live = read_live_receipt(root)
    entries = list((first or {}).get("written") or [])
    for frame in (live or {}).get("frames") or []:
        if isinstance(frame, dict):
            entries.extend(frame.get("written") or [])
    for entry in entries:
        if isinstance(entry, dict) and entry.get("name"):
            names.add(Path(str(entry["name"])).as_posix())
    receipts = Path(fs_path(root / ".render-receipts", descend=True))
    try:
        documents = sorted(receipts.glob("*.json"))
    except OSError:
        documents = []
    for document in documents:
        try:
            if document.stat().st_size > render_receipts._MAX_RECEIPT_BYTES:
                continue
            payload = json.loads(document.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (not isinstance(payload, dict) or payload.get("schema")
                != render_receipts.INVOCATION_SCHEMA):
            continue
        for row in payload.get("rendered") or []:
            if not isinstance(row, dict):
                continue
            name = _named_inside(root, row.get("path"))
            if name is not None:
                names.add(name)
    return names


def _render_scratch(root: Path) -> list[Path]:
    """The working folders renders leave inside and beside ``root``.

    The early render's and the every-frame render's scratch, and the
    renderer's own store beside each of them and beside the picture
    folder itself (:func:`gpuwm.render.scratch_root_for`).
    """

    from gpuwm.live_products import _SCRATCH_NAME as LIVE_SCRATCH
    from gpuwm.render_layout import SCRATCH_SUFFIX

    inside = [root / name for name in (_SCRATCH_NAME, LIVE_SCRATCH)]
    return [*inside,
            *(path.with_name(path.name + SCRATCH_SUFFIX) for path in inside),
            root.with_name(root.name + SCRATCH_SUFFIX)]


def discard_stopped_render(render_dir) -> dict[str, Any]:
    """Remove what a stopped render left, and keep every receipted picture.

    THE BREAKAGE: a child stopped while its
    analysis frame was being drawn kept 155 to 171 of the renderer's
    staging files as its pictures: ``rustwx_wrf_..._<product>.png`` and
    ``var_wrf_isltyp_<hash>.png`` lying flat in ``png/``, against the
    render folder ruling, and the banner and report counted them as
    pictures drawn.  The engine draws every picture flat under a name of
    its own and only then files it into ``<domain>/<product>/<day>/``;
    a render stopped in between leaves the flat names behind and writes
    no receipt for them.

    So, on a stop, once every render this run started has ended: every
    PNG in the folder that no render receipt names
    (:func:`receipted_pictures`) is removed, and so is the renderers'
    working scratch.  What stays is the organised tree of pictures a
    render finished and vouched for, and the frames themselves, which
    draw again with ``gpuwm render`` at any time.

    Only a caller that has ENDED every render may call this: a render
    still running would lose the pictures it has drawn and not yet
    receipted.  Never raises; ``{"removed": [...], "errors": [...]}``.
    """

    from gpuwm.render_layout import is_scratch_dir

    root = Path(render_dir)
    removed: list[str] = []
    errors: list[str] = []
    for scratch in _render_scratch(root):
        spelled = fs_path(scratch, descend=True)
        if os.path.isdir(spelled):
            shutil.rmtree(spelled, ignore_errors=True)
            if os.path.isdir(spelled):
                errors.append(f"{scratch}: could not be removed")
    keep_names = receipted_pictures(root)
    walk_root = Path(fs_path(root, descend=True))
    emptied: set[str] = set()
    for directory, subdirectories, names in os.walk(walk_root):
        subdirectories[:] = [name for name in subdirectories
                             if not is_scratch_dir(name)]
        for name in names:
            if not name.lower().endswith(".png"):
                continue
            path = Path(directory) / name
            relative = path.relative_to(walk_root).as_posix()
            if relative in keep_names:
                continue
            try:
                os.remove(path)
            except OSError as error:
                errors.append(f"{relative}: {error.strerror or error}")
                continue
            removed.append(relative)
            emptied.add(directory)
    # A folder the removal emptied goes too, deepest first, so a render
    # stopped half way through filing leaves no empty product folders.
    for directory in sorted(emptied, key=len, reverse=True):
        current = Path(directory)
        while current != walk_root and walk_root in current.parents:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent
    return {"removed": removed, "errors": errors}


def keep(render_dir: Path, *, why: str,
         stopped: Mapping[str, Any] | None = None,
         frames: Sequence[Any] = (), requested: bool = False,
         discard: bool = False) -> dict[str, Any]:
    """Keep what the early render drew, and say on disk that it stopped.

    THE DECISION, recorded where it is enforced: a run that did not
    finish KEEPS its pictures.  The early render draws the analysis
    frame while the forecast is still integrating, and a child that then
    stops -- non-finite, a refusal mid-run, an interrupt -- used to have
    those pictures removed, so the only artifact a reader could look at
    disappeared at the exact moment it became the thing they wanted.
    The frames prove what happened; the pictures are what a reader can
    actually see, and they are cheap to keep.

    What replaces the removal is a SENTENCE, in the two places this
    function writes: a banner at the top of this directory and the
    status in the render summary beside it.  A picture with no verdict
    beside it is the state this function exists to prevent.

    IN THE REPORT TOO, where the run leaves one.  A child that stops
    inside its forecast publishes ``report.json`` whatever stopped it,
    carrying its failure capsule or the sentence the banner carries
    where the stop composed none, and both stop arms record what this
    function did in that document's ``products`` block: ``status``
    ``KEPT``, the count on disk and the banner's path.  This function
    does not write the report -- it returns what it did, and the caller
    that holds the report puts it there -- so the folder, the summary
    and the document cannot disagree about how many pictures there are.

    Best effort, exactly as the removal was.  A banner that cannot be
    written is not worth failing an already-failed run over: the count
    still comes back, and ``banner`` is ``None``.  The summary is
    stamped either way -- two independent writes, because the summary
    is the document the readers key on and a banner's failure says
    nothing about whether the status can be recorded.

    A STOP (``requested``) says so in the banner's first line, and
    ``discard`` first removes what a stopped render left
    (:func:`discard_stopped_render`), so the count below covers only
    pictures a render receipt names.  The caller passes ``discard`` only
    once every render the run started has ended.
    """

    from gpuwm import render_receipts

    root = Path(render_dir)
    discarded = (discard_stopped_render(root) if discard
                 else {"removed": [], "errors": []})
    # THREE outcomes, from the counter both stop routes and the
    # failed-render capsule share (:func:`count_pictures`): a count, an
    # empty folder, and a folder that could not be LISTED.  Counting a
    # refused listing as zero made the banner and the summary tell a
    # reader whose pictures were behind a permission wall or a dropped
    # mount that the run had drawn none, which is the one sentence that
    # sends them off to re-draw a whole child.
    pictures, pictures_error = count_pictures(root)
    # The banner is written through the SAME spelling of this directory
    # the summary is stamped through, so a path the ordinary API refuses
    # for its length loses neither document rather than one of them.
    # ``fs_path`` is a spelling and never a different file, and the path
    # this function REPORTS is the plain one a reader compares against.
    banner: Path | None = root / DID_NOT_FINISH_BANNER
    try:
        os.makedirs(fs_path(root, descend=True), exist_ok=True)
        Path(fs_path(root / DID_NOT_FINISH_BANNER)).write_text(
            banner_text(why=why, stopped=stopped, frames=frames,
                        pictures=pictures, pictures_error=pictures_error,
                        requested=requested),
            encoding="utf-8", newline="\n")
    except OSError:
        banner = None
    # UNCONDITIONAL, because these are two independent best-effort
    # writes rather than one gated on the other.  The summary is what
    # the desktop's native-plots door and every run browser key on, and
    # a banner that could not be written says nothing about whether the
    # status can be recorded; ``stamp_status`` has its own try/except
    # and takes ``banner_path=None``.
    summary = render_receipts.stamp_status(
        root, status=DID_NOT_FINISH_STATUS, pictures_on_disk=pictures,
        pictures_error=pictures_error, banner_path=banner)
    return {"pictures": pictures,
            "pictures_error": pictures_error,
            "render": str(root),
            "banner": None if banner is None else str(banner),
            "status": DID_NOT_FINISH_STATUS,
            "discarded": len(discarded["removed"]),
            "discard_errors": list(discarded["errors"]),
            "summary": summary}


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish the receipt through a rename, like every other receipt."""

    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def read_receipt(render_dir: Path) -> dict[str, Any] | None:
    """The early render's receipt from a render directory, or ``None``."""

    path = Path(render_dir) / FIRST_PRODUCTS_RECEIPT
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != FIRST_PRODUCTS_SCHEMA:
        return None
    return payload


def published_pictures_are_original(receipt: Mapping[str, Any],
                                    render_dir: Path) -> bool:
    """Whether the pictures this receipt names still carry its instant.

    A digest cannot answer this.  The early picture and the finalize one
    are byte-identical by construction -- both stages compose the render
    command from the same plan dict, and a test renders both ways and
    compares the bytes -- so a re-render leaves every recorded sha256
    matching and moves only the mtime.  The mtime is therefore the whole
    of the evidence, and it is exactly what a reader comparing the
    printed number against the tree is looking at.

    ``False`` also for a receipt whose pictures have gone: an instant
    stamped for files that are not there describes nothing either.
    """

    published = receipt.get("published_unix_ms")
    written = receipt.get("written")
    if not isinstance(published, int) or not isinstance(written, list):
        return False
    if not written:
        return False
    for entry in written:
        if not isinstance(entry, dict):
            return False
        picture = Path(render_dir) / str(entry.get("name") or "")
        try:
            stamp = Path(fs_path(picture)).stat().st_mtime
        except OSError:
            return False
        if int(stamp * 1000) > published + _MTIME_SLACK_MS:
            return False
    return True


def _receipt_still_holds(receipt: Mapping[str, Any], *, render_dir: Path,
                         render_products: Any,
                         render_section: Any = None) -> str | None:
    """Why this receipt may not be trusted, or ``None`` when it may.

    Every clause is a digest or an existence check against what is on
    disk right now.  A receipt is a claim about the past; skipping work
    on the strength of one is only sound while the claim is still true.
    """

    declared = effective_products(receipt.get("render_products"))
    wanted = effective_products(render_products)
    if declared != wanted:
        return (f"it was rendered for --products {declared!r} and this "
                f"stage renders {wanted!r}")
    drawn = section_text(receipt.get("render_section"))
    along = section_text(render_section)
    if drawn != along:
        return (f"its sections were cut along {drawn or 'no line'} and "
                f"this stage cuts them along {along or 'no line'}")
    code = receipt.get("exit_code", 0)
    if receipt.get("complete") is False or code not in (0, None):
        return (f"its render exited {code} before every product was "
                "drawn, so the frame is drawn again")
    if receipt.get("complete") is not True:
        # Written before receipts recorded the renderer's exit: such a receipt said "done" for a render that
        # failed partway as well, so it cannot show that every product was drawn.
        return ("it does not record whether its render finished, so the "
                "frame is drawn again")
    frame = Path(str(receipt.get("frame") or ""))
    if not frame.is_file():
        return f"the frame it names ({frame}) is not on disk"
    if _sha256_file(frame) != receipt.get("frame_sha256"):
        return f"the frame it names ({frame}) no longer matches its digest"
    written = receipt.get("written")
    if not isinstance(written, list) or not written:
        return "it names no picture"
    for entry in written:
        if not isinstance(entry, dict):
            return "one of its entries is not a record"
        picture = render_dir / str(entry.get("name") or "")
        if not Path(fs_path(picture)).is_file():
            return f"the picture it names ({picture.name}) is not on disk"
        if _sha256_file(picture) != entry.get("sha256"):
            return (f"the picture it names ({picture.name}) no longer "
                    "matches its digest")
    return None


def published_frames(frames: Sequence[Path], plan: Mapping[str, Any]
                     ) -> tuple[list[Path], list[Path], str | None]:
    """Split a finalize frame list into still-to-render and already-done.

    Returns ``(remaining, already, note)``.  ``note`` is prose for the
    chain's own output: either what was skipped and why it could be, or
    why a receipt that exists was not trusted.  ``already`` is never
    non-empty without a note, so a skipped frame is never a silent one.
    """

    render_dir = Path(plan["render"])
    receipt = read_receipt(render_dir)
    if receipt is None:
        return list(frames), [], None
    stale = _receipt_still_holds(
        receipt, render_dir=render_dir,
        render_products=plan.get("render_products"),
        render_section=plan.get("render_section"))
    if stale is not None:
        return (list(frames), [],
                f"early-render receipt not used: {stale}; every frame is "
                "being rendered")
    claimed = Path(str(receipt["frame"])).resolve()
    remaining = [frame for frame in frames
                 if Path(frame).resolve() != claimed]
    already = [frame for frame in frames
               if Path(frame).resolve() == claimed]
    if not already:
        return (list(frames), [],
                "early-render receipt not used: the frame it names is not "
                "in this stage's list; every frame is being rendered")
    count = len(receipt["written"])
    return (remaining, already,
            f"1 frame already published by the early render "
            f"({count} picture(s), digests verified): {claimed.name}")


__all__ = [
    "DEFAULT_RENDER_PRODUCTS",
    "DEFAULT_WAIT_SECONDS",
    "DID_NOT_FINISH_BANNER",
    "DID_NOT_FINISH_STATUS",
    "FIRST_PLOT_DEFINITION",
    "FIRST_PRODUCTS_RECEIPT",
    "FIRST_PRODUCTS_SCHEMA",
    "FirstProducts",
    "FrameHook",
    "WorkerEnd",
    "arm",
    "banner_text",
    "count_pictures",
    "discard_stopped_render",
    "early_render_requested",
    "effective_products",
    "finished_pictures",
    "halt_renders_and_wait",
    "keep",
    "published_frames",
    "published_pictures_are_original",
    "read_receipt",
    "receipted_pictures",
    "render_outcome",
    "render_was_stopped",
    "render_without_output_refusal",
    "section_text",
]
