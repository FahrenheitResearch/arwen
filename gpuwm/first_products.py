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
#: (``gpuwm/render.py``'s own default).  The early render is asked for the
#: explicit spelling because it is given a command line; the finalize stage
#: leaves the flag off.  Comparing the two literally made every `go` run's
#: receipt look like it had been drawn for a different product set than the
#: stage that could have skipped it, so the skip never happened there.
DEFAULT_RENDER_PRODUCTS = "all"

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


def effective_products(render_products: Any) -> str:
    """The product spec a render will actually draw.

    ``None``/empty is not "no products" -- it is ``gpuwm render``'s own
    default -- so the two spellings of the same request compare equal
    instead of looking like two different renders.
    """

    text = "" if render_products is None else str(render_products).strip()
    return text or DEFAULT_RENDER_PRODUCTS


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


def _run_render(command: Sequence[str]) -> subprocess.CompletedProcess:
    """Spawn the render exactly as the finalize stage spawns it.

    Same cwd and same environment, from the same two helpers, because a
    render that differs from finalize's in any way that could reach the
    output is a render whose bytes cannot be assumed to match it.
    """

    from gpuwm.go_cli import _stage_cwd, _stage_env

    return subprocess.run(
        list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, errors="replace", cwd=str(_stage_cwd()),
        env=_stage_env())


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
                                  subprocess.CompletedProcess] | None = None):
        self._plan = dict(render_plan)
        self._report = report
        self._warn = warn
        self._runner = _run_render if runner is None else runner
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._receipt: dict[str, Any] | None = None

    # -- what this was armed with -------------------------------------

    @property
    def render_dir(self) -> Path:
        return Path(self._plan["render"])

    @property
    def render_products(self) -> str:
        return str(self._plan.get("render_products") or "")

    @property
    def receipt(self) -> dict[str, Any] | None:
        """The published receipt, or ``None`` until one exists."""

        return self._receipt

    @property
    def dispatched(self) -> bool:
        return self._thread is not None

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
                target=self._guarded_render,
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
        """

        thread = self._thread
        if thread is None:
            return None
        thread.join(timeout)
        if thread.is_alive():
            self._warn(
                "first_products_timeout",
                "the early render of the first frame was still running "
                f"after {timeout:.0f} s, so the finalize stage is "
                "rendering every frame itself; the early render will "
                "publish nothing",
                render_dir=str(self.render_dir))
            return None
        return self._receipt

    # -- the worker ---------------------------------------------------

    def _guarded_render(self, **kwargs: Any) -> None:
        try:
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

        started = time.perf_counter()
        render_dir = self.render_dir
        scratch = render_dir / _SCRATCH_NAME
        # Create-only for the scratch: a leftover from a previous run in
        # the same directory would be published as though this render had
        # written it.
        if scratch.exists():
            shutil.rmtree(fs_path(scratch))
        scratch.mkdir(parents=True)
        try:
            command = render_command(
                {**self._plan, "render": scratch}, [frame])
            completed = self._runner(command)
            written = iter_rendered(scratch)
            if not written:
                # Not a failure.  The cold-start frame carries no
                # REFL_10CM -- no microphysics call precedes it, a
                # registered deviation -- so a run whose only product is
                # reflectivity legitimately has nothing to draw yet.  The
                # frame is left unclaimed and finalize renders it with
                # the rest, which is where it will be skipped for the
                # same honest reason.
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
            render_dir.mkdir(parents=True, exist_ok=True)
            published: list[dict[str, str]] = []
            paths: list[Path] = []
            for source in written:
                # The RELATIVE path, not the bare name: since 2.5.0 the
                # render writes a tree (domain/product/valid-day, see
                # gpuwm.render_layout) and flattening it here would
                # publish the early frame into a different directory
                # from the one finalize renders the rest into -- two
                # layouts in one run, from the same command.
                relative = source.relative_to(scratch)
                target = render_dir / relative
                # Both sides through render_layout.fs_path: the scratch
                # is one folder DEEPER than the published tree, so it is
                # the first place the layout's own path length runs into
                # Windows' MAX_PATH, and a publish that failed there
                # would drop a drawn picture on the floor.
                spelled = fs_path(target)
                Path(spelled).parent.mkdir(parents=True, exist_ok=True)
                # Atomic within the volume: a reader tailing the render
                # directory sees a whole PNG or no PNG, and a later
                # finalize write of the same name cannot land on top of
                # a write still in flight.
                os.replace(fs_path(source), spelled)
                # Recorded relative to the render directory, in posix
                # spelling, so `render_dir / entry["name"]` re-finds it
                # on any platform when finalize re-checks the digests.
                published.append({"name": relative.as_posix(),
                                  "sha256": _sha256_file(target)})
                paths.append(target)
            elapsed = time.perf_counter() - started
        finally:
            shutil.rmtree(fs_path(scratch), ignore_errors=True)

        announced = {
            "schema": FIRST_PRODUCTS_SCHEMA,
            # One definition, written down where the number is, so a
            # consumer never has to guess which quantity it is holding.
            "measures": FIRST_PLOT_DEFINITION,
            # THE TIME-TO-FIRST-PLOT INSTANT, on the wall clock.
            #
            # The report hook below carries seconds-from-start, which is
            # the number a HOST wants because the host owns the start.
            # A caller that did not host this render -- `gpuwm go`,
            # which asks the runner subprocess to do it -- has only the
            # receipt, and a duration measured from a start it cannot
            # see is not a number it can use.  So the receipt carries
            # the absolute instant and every consumer subtracts its own
            # launch from it.
            "published_unix_ms": int(time.time() * 1000),
            "frame": str(frame),
            "domain": int(domain),
            "valid_time": (valid_time.isoformat()
                           if hasattr(valid_time, "isoformat")
                           else str(valid_time)),
            "render_products": self.render_products,
            "command": [str(part) for part in command],
            "written": published,
            "render_seconds": round(elapsed, 6),
        }
        # The event goes out HERE, before the frame is digested and
        # before the receipt is written.  Both of those are the finalize
        # stage's business and can wait; the event is the TTFP number,
        # and its whole value is marking the instant the pictures became
        # readable.  Hashing a 362 MB history frame first would have put
        # about a second of bookkeeping inside the number.
        self._report({**announced, "paths": [str(path) for path in paths]})
        receipt = {**announced, "frame_sha256": _sha256_file(frame)}
        self._receipt = receipt
        _write_receipt(render_dir / FIRST_PRODUCTS_RECEIPT, receipt)


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
                         render_products: Any) -> str | None:
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
        render_products=plan.get("render_products"))
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
    "FIRST_PLOT_DEFINITION",
    "FIRST_PRODUCTS_RECEIPT",
    "FIRST_PRODUCTS_SCHEMA",
    "FirstProducts",
    "early_render_requested",
    "effective_products",
    "published_frames",
    "published_pictures_are_original",
    "read_receipt",
]
