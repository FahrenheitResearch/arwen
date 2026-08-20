"""What `gpuwm go` cost, on disk, without anyone asking for it.

THE DEFECT THIS CLOSES, measured 2026-08-16.  `gpuwm go` already times
every stage it runs -- it prints ``ok fetch (11s)`` and hands the same
number to an optional observer -- and a bare `gpuwm go`, which is what
everybody types, had no observer, so every one of those numbers died
with the scrollback.  Answering one sentence's worth of question ("how
long from launch to sim step 1?") took a wrapper's timestamps, terminal
prose, proof.json, report.json and progress.jsonl assembled by hand.
Nothing the product writes answered it.

`gpuwm run-plan` -- the least-used front door -- already wrote exactly
the right artifact: an append-only ``events.jsonl`` at
``gpuwm.run-plan.event.v1`` with ``stage_started``/``stage_finished``
carrying wall seconds.  So this module invents no format.  It attaches
that same :class:`gpuwm.runplan.EventStream`, with that same grammar and
that same reader (:func:`gpuwm.runplan.read_events`), to the front door
people actually use.

Three things are deliberate:

**It is not a host.**  ``hosts_forecast`` is False, so the forecast
stays the subprocess it has always been -- process isolation is what
keeps a CUDA failure inside one stage, and telemetry must not be the
reason a chain gives that up.  ``gpuwm run-plan`` passes its own
observer, which does host, and this one steps aside for it entirely.

**The stage names are `go`'s chain, not run-plan's five.**  ``boot``,
``authority``, ``fetch``, ``manifest``, ``prepare``, ``forecast``,
``render``.  The envelope and the tag set are run-plan's; the stage
vocabulary belongs to the chain being described, and a consumer switches
on ``event`` (closed) and reads ``stage`` (open) exactly as it already
does for the HRRR chain's phases.

**Every number is relayed from an artifact.**  Fetch bandwidth is read
back out of ``fetch-manifest.json``; the forecast's internals are read
back out of ``progress.jsonl``; time to first plot is read back out of
the early render's receipt while the pictures it names still carry that
instant, and off the pictures' own mtimes otherwise.  Nothing here
re-derives a number a stage already published, and nothing here quotes a
number the published tree contradicts -- see :data:`TTFP_DEFINITION`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from gpuwm import LAUNCH_MONOTONIC, LAUNCH_UNIX_MS

#: Where the chain's event stream lands, inside the run root.  The same
#: filename run-plan uses, in the same grammar, on purpose.
CHAIN_EVENTS_FILENAME = "events.jsonl"

#: The stage that covers everything before the first subprocess: the
#: CLI's own boot and imports, the config, the capability gate, the
#: memory gate and the geography gate.  MEASURED: 1.5 s warm, 1.7 s
#: cold, which is small and was nowhere.
BOOT_STAGE = "boot"

#: How ``time_to_first_plot_source`` names where the number came from.
TTFP_FROM_RECEIPT = "first-products receipt"
TTFP_FROM_MTIME = "earliest rendered picture mtime"

#: ONE definition behind both sources above, and the reason the receipt is
#: corroborated against the tree before it is quoted.
#:
#: MEASURED on both 3080 walks: `go` printed "time to first plot 0m 46s
#: (first-products receipt)" while the earliest PNG in the published run
#: tree carried 2m 45s.  Both numbers were real -- the early render did
#: publish at 46 s, and the finalize stage did rewrite those same paths at
#: 2m 45s -- and nothing reconciled them, so the headline contradicted the
#: only artifact a reader can check.
TTFP_DEFINITION = (
    "seconds from launch until the first picture still present in the "
    "render tree became readable"
)


def _forecast_breakdown(run_dir: Path) -> dict[str, Any] | None:
    """The forecast stage's own internals, out of its step log.

    The audit had to open progress.jsonl by hand to learn that 51 of a
    cold run's 88 seconds to step 1 were model step 1 itself.  This is
    that reading, done once, into the chain's own stream, so the whole
    launch-to-step-1 decomposition is answerable from one file.

    ``None`` when there is no readable step log -- a forecast that never
    started, a run with ``--progress-format off``.  Never an exception:
    this is telemetry about a run that has already finished.
    """

    from gpuwm.progress_log import STEP_LOG_FILENAME, read_step_log

    path = Path(run_dir) / STEP_LOG_FILENAME
    try:
        records = read_step_log(path)
    except (OSError, ValueError):
        return None
    if not records:
        return None
    phases: dict[str, float] = {}
    first_step: float | None = None
    steps = 0
    end: Mapping[str, Any] = {}
    for record in records:
        event = record.get("event")
        if event == "phase":
            wall = record.get("wall_seconds")
            if isinstance(wall, (int, float)):
                phases[str(record.get("name"))] = float(wall)
        elif event == "step":
            steps += 1
            if record.get("step") == 1 and first_step is None:
                wall = record.get("step_wall_seconds")
                if isinstance(wall, (int, float)):
                    first_step = float(wall)
        elif event == "run_end":
            end = record
    integration = end.get("wall_seconds")
    return {
        "phases": phases,
        "first_step_seconds": first_step,
        "first_step_excess_seconds": end.get("first_step_excess_seconds"),
        "integration_seconds": (float(integration)
                                if isinstance(integration, (int, float))
                                else None),
        "steps": end.get("steps", steps),
        "status": end.get("status"),
    }


def _earliest_picture_unix_ms(render_dir: Path) -> int | None:
    """When the first picture of this run landed, by its own mtime.

    The fallback half of time-to-first-plot, and the method the audit
    itself used.  Coarser than the engine's own measurement -- a
    filesystem timestamp, not a stopwatch around the publish -- so it is
    labelled as such wherever it is reported rather than quietly mixed
    in with the exact one.
    """

    earliest: float | None = None
    try:
        for path in Path(render_dir).rglob("*.png"):
            try:
                stamp = path.stat().st_mtime
            except OSError:
                continue
            if earliest is None or stamp < earliest:
                earliest = stamp
    except OSError:
        return None
    return None if earliest is None else int(earliest * 1000)


class GoChainEvents:
    """`gpuwm go`'s always-present stage observer.

    Duck-typed against the hooks :func:`gpuwm.go_cli._run_stage` already
    raises (``stage_begin``, ``stage_heartbeat``, ``stage_end``), so the
    chain needed no new call sites to be instrumented -- the notifier
    and the elapsed values were already there and had nowhere to go.

    Telemetry never fails a chain.  ``_run_stage`` swallows anything a
    hook raises, and the two entry points it does not cover
    (:meth:`open` and :meth:`finish`) swallow their own.
    """

    #: `go` runs the forecast IN PROCESS for an observer that is hosting
    #: it -- that is the only way per-step and per-frame events can
    #: reach a host.  This observer is not a host: it wants the
    #: subprocess, because process isolation keeps a CUDA failure inside
    #: one stage, and no telemetry is worth trading that for.
    hosts_forecast = False

    def __init__(self, *, launch_monotonic: float | None = None,
                 launch_unix_ms: int | None = None):
        self._launch = (LAUNCH_MONOTONIC if launch_monotonic is None
                        else float(launch_monotonic))
        self._launch_unix_ms = (LAUNCH_UNIX_MS if launch_unix_ms is None
                                else int(launch_unix_ms))
        self._events = None
        self._stages: list[dict[str, Any]] = []
        self._open: dict[str, Any] | None = None
        self._data_dir: Path | None = None
        self._run_dir: Path | None = None
        self._render_dir: Path | None = None
        self.path: Path | None = None

    # -- lifecycle -----------------------------------------------------

    def open(self, path, *, plan: Mapping[str, Any] | None = None) -> None:
        """Start the stream, and close the boot stage that led to it.

        Called AFTER the gates rather than at the top of the chain, and
        that ordering is the point: a refused memory gate must leave the
        disk exactly as it found it, and opening a stream would create
        the run root for a run that never happened.  The boot stage is
        not lost by waiting -- it is emitted here, from the launch
        anchor, carrying its own unix-ms start.
        """

        from gpuwm.runplan import EventStream

        if plan is not None:
            self._data_dir = plan.get("data")
            self._run_dir = plan.get("run")
            self._render_dir = plan.get("render")
        try:
            # mirror=None: run-plan's stream IS its stdout contract and
            # mirrors every line; `go`'s stdout is a person's terminal,
            # and doubling every stage line as JSON would be the noise
            # the one-line-per-stage design exists to avoid.
            self._events = EventStream(Path(path), mirror=None)
            self.path = Path(path)
        except OSError:
            self._events = None
            return
        boot = time.monotonic() - self._launch
        self._emit("stage_started", stage=BOOT_STAGE,
                   started_unix_ms=self._launch_unix_ms)
        self._finish(BOOT_STAGE, wall_seconds=boot, ok=True, exit_code=0,
                     started_unix_ms=self._launch_unix_ms)

    def close(self) -> None:
        events, self._events = self._events, None
        if events is not None:
            try:
                events.close()
            except OSError:
                pass

    # -- the chain's own hooks ----------------------------------------

    def stage_begin(self, *, label: str, command) -> None:
        self._open = {
            "stage": label,
            "started_monotonic": time.monotonic(),
            "started_unix_ms": int(time.time() * 1000),
        }
        self._emit("stage_started", stage=label,
                   command=[str(part) for part in command],
                   started_unix_ms=self._open["started_unix_ms"])

    def stage_heartbeat(self, *, label: str, elapsed_seconds: float,
                        progress) -> None:
        # Deliberately silent.  A heartbeat is a property of WAITING and
        # its whole audience is the person watching the terminal; every
        # fact it could carry is already in the stage's own published
        # progress file, which is on disk beside this stream.  Emitting
        # one line per 20 s per stage would make the event file mostly
        # heartbeats.
        return None

    def stage_end(self, *, label: str, exit_code: int, ok: bool,
                  elapsed_seconds: float, progress) -> None:
        started_unix_ms = None
        if self._open is not None and self._open["stage"] == label:
            started_unix_ms = self._open["started_unix_ms"]
        self._open = None
        extra: dict[str, Any] = {}
        if label == "fetch" and self._data_dir is not None:
            extra = self._fetch_fields()
        self._finish(label, wall_seconds=float(elapsed_seconds), ok=bool(ok),
                     exit_code=int(exit_code),
                     started_unix_ms=started_unix_ms, **extra)

    def arm_first_products(self, render_plan) -> None:
        # `go` offers every observer the chance to render the first
        # frame as it lands.  Taking it would mean hosting the forecast
        # in process (that is where the per-frame hook is raised), and
        # this observer will not do that -- so the early render is asked
        # for on the runner's own command line instead
        # (`go_cli.forecast_command`), which keeps the subprocess AND
        # gets the picture out early.  Nothing to do here; the hook
        # exists so `go` does not have to ask whether it is there.
        return None

    # -- the numbers only the end of the chain knows -------------------

    def first_products_receipt(self, *, render_dir=None
                               ) -> dict[str, Any] | None:
        """The early render's receipt as a TTFP fact, or ``None``.

        NOT named ``first_products``: ``go_cli._render_stage`` reads
        that exact attribute off its observer to decide whether an early
        render is armed and, if it is, calls ``.wait()`` on it.  A
        method by that name here would be truthy, would be waited on,
        and would take the render stage down with an AttributeError --
        outside the ``_notify`` guard that protects the rest.
        """

        from gpuwm.first_products import read_receipt

        directory = self._render_dir if render_dir is None else Path(render_dir)
        if directory is None:
            return None
        receipt = read_receipt(Path(directory))
        if not isinstance(receipt, dict):
            return None
        published = receipt.get("published_unix_ms")
        if not isinstance(published, int):
            return None
        from gpuwm.first_products import published_pictures_are_original

        return {
            "receipt": receipt,
            "published_unix_ms": published,
            "seconds_from_launch": round(
                (published - self._launch_unix_ms) / 1000.0, 6),
            # Whether the pictures this receipt names still carry its
            # instant, or were rewritten by a later render.  Recorded on
            # the event so the stream never carries a number the tree
            # contradicts without saying so.
            "pictures_still_original": published_pictures_are_original(
                receipt, Path(directory)),
        }

    def time_to_first_plot(self, *, render_dir=None) -> dict[str, Any] | None:
        """:data:`TTFP_DEFINITION`, and which measurement answered it.

        The early render's own receipt is preferred, because it is
        stamped by the process that published the pictures rather than
        read off a filesystem timestamp -- but only while the pictures it
        names still carry that instant.  A later render that rewrote
        those paths makes the receipt's number describe files that are no
        longer there, and quoting it then is how a headline comes to
        contradict the tree it is a headline about.  In that case, and
        when there was no early render at all, the earliest picture's own
        mtime answers, labelled as the coarser measurement it is.
        """

        directory = self._render_dir if render_dir is None else render_dir
        if directory is None:
            return None
        directory = Path(directory)
        early = self.first_products_receipt(render_dir=directory)
        if early is not None and early["pictures_still_original"]:
            return {
                "seconds": early["seconds_from_launch"],
                "source": TTFP_FROM_RECEIPT,
                "published_unix_ms": early["published_unix_ms"],
            }
        published = _earliest_picture_unix_ms(directory)
        if published is None:
            return None
        return {
            "seconds": round((published - self._launch_unix_ms) / 1000.0, 6),
            "source": TTFP_FROM_MTIME,
            "published_unix_ms": published,
        }

    def finish(self, *, status: str, exit_code: int = 0) -> dict[str, Any]:
        """Emit the terminal event: the whole chain, assembled, once.

        This is the record the audit could not find anywhere: every
        stage's wall, the forecast's own internals read back out of its
        step log, time to first plot, and -- stated rather than left for
        the reader to subtract -- how much of the process wall the named
        stages account for.
        """

        wall = time.monotonic() - self._launch
        accounted = sum(float(stage["wall_seconds"]) for stage in self._stages)
        ttfp = None
        try:
            ttfp = self.time_to_first_plot()
        except Exception:  # noqa: BLE001 - telemetry never fails a chain
            ttfp = None
        forecast = None
        if self._run_dir is not None:
            try:
                forecast = _forecast_breakdown(Path(self._run_dir))
            except Exception:  # noqa: BLE001
                forecast = None
        summary = {
            "wall_seconds": round(wall, 6),
            "stages": [dict(stage) for stage in self._stages],
            "accounted_seconds": round(accounted, 6),
            # Named, not left implicit.  The acceptance bar for this
            # instrument is that the stages account for the process, and
            # a reader must be able to check that without re-adding the
            # list themselves.
            "unaccounted_seconds": round(wall - accounted, 6),
            "forecast": forecast,
            "time_to_first_plot_seconds": (None if ttfp is None
                                           else ttfp["seconds"]),
            "time_to_first_plot_source": (None if ttfp is None
                                          else ttfp["source"]),
            # Stated, not implied.  Two quantities were both being called
            # "time to first plot" and the printed line named neither.
            "time_to_first_plot_definition": TTFP_DEFINITION,
            "launch_unix_ms": self._launch_unix_ms,
        }
        if ttfp is not None:
            early = self.first_products_receipt()
            if early is not None:
                receipt = early["receipt"]
                self._emit(
                    "first_products_ready",
                    domain=receipt.get("domain"),
                    valid_time=receipt.get("valid_time"),
                    frame=receipt.get("frame"),
                    paths=[str(entry.get("name"))
                           for entry in receipt.get("written", [])
                           if isinstance(entry, dict)],
                    render_products=receipt.get("render_products"),
                    render_seconds=receipt.get("render_seconds"),
                    published_unix_ms=early["published_unix_ms"],
                    seconds_from_launch=early["seconds_from_launch"],
                    pictures_still_original=early["pictures_still_original"])
        if status == "SUCCESS":
            self._emit("completed", **summary)
        else:
            self._emit("failed", status=str(status), exit_code=int(exit_code),
                       **summary)
        self.close()
        return summary

    # -- internals -----------------------------------------------------

    def _fetch_fields(self) -> dict[str, Any]:
        """Bytes and bandwidth for the fetch stage, from its manifest."""

        from gpuwm.fetch import fetch_throughput

        try:
            throughput = fetch_throughput(Path(self._data_dir))
        except Exception:  # noqa: BLE001 - telemetry never fails a stage
            return {}
        if throughput is None:
            return {}
        # `bytes_per_second` is bandwidth, and is null when this run
        # downloaded nothing.  A re-run against an existing --data-dir
        # only re-hashes what is there; reporting that as bandwidth read
        # 1.09 GB/s on the reference box, which is a true statement
        # about sha256 wearing the name of the network.
        return {
            "bytes": throughput["bytes"],
            "files": throughput["files"],
            "stage_seconds": throughput["seconds"],
            "downloaded_files": throughput["downloaded_files"],
            "downloaded_bytes": throughput["downloaded_bytes"],
            "downloaded_seconds": throughput["downloaded_seconds"],
            "bytes_per_second": throughput["bytes_per_second"],
            "verified_files": throughput["verified_files"],
            "verified_bytes": throughput["verified_bytes"],
            # The pooled transport's own receipt: workers, host caps,
            # wall, the serial model and the effective speedup.
            # `stage_seconds` above is the serial model (per-file sum);
            # the wall the user waited is in here.
            "concurrency": throughput.get("concurrency"),
        }

    def _finish(self, stage: str, *, wall_seconds: float, ok: bool,
                exit_code: int, started_unix_ms: int | None = None,
                **fields: Any) -> None:
        record = {
            "stage": stage,
            "wall_seconds": round(float(wall_seconds), 6),
            "ok": bool(ok),
            "exit_code": int(exit_code),
            **fields,
        }
        self._stages.append(record)
        self._emit("stage_finished", started_unix_ms=started_unix_ms,
                   finished_unix_ms=int(time.time() * 1000), **record)

    def _emit(self, event: str, **fields: Any) -> None:
        if self._events is None:
            return
        try:
            self._events.emit(event, **fields)
        except Exception:  # noqa: BLE001 - telemetry never fails a chain
            pass


def read_chain_events(path) -> list[dict[str, Any]]:
    """Replay one chain event stream.  Run-plan's reader, by name.

    Exported so a consumer does not have to know that `go`'s stream and
    `run-plan`'s stream are the same thing -- and so that if they ever
    stopped being, this import would be the thing that broke.
    """

    from gpuwm.runplan import read_events

    return read_events(path)


def stage_walls(events: list[Mapping[str, Any]]) -> dict[str, float]:
    """``{stage: wall_seconds}`` from a replayed stream, in order."""

    return {str(record["stage"]): float(record["wall_seconds"])
            for record in events
            if record.get("event") == "stage_finished"
            and isinstance(record.get("wall_seconds"), (int, float))}


def summarize(path) -> dict[str, Any] | None:
    """The chain's terminal summary, read back off disk.

    One function so a Rust-reset lane comparing a before and an after
    does not each write their own parser of this file.
    """

    for record in reversed(read_chain_events(path)):
        if record.get("event") in ("completed", "failed"):
            return dict(record)
    return None


def load_baseline(path) -> dict[str, Any]:
    """One pinned baseline document, refusing an undated one.

    A baseline without provenance is a number with no claim attached:
    "prepare was 8.3 s" means nothing without the box, the card, the
    date and whether the caches were warm.  Refused rather than read,
    because the whole use of this file is a lane comparing its own
    measurement against it.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [key for key in ("schema", "provenance", "runs")
               if key not in payload]
    if missing:
        raise ValueError(
            f"{path} is not a stage baseline: it has no {missing}")
    provenance = payload["provenance"]
    required = ("box", "card", "date", "gpuwm_version", "case")
    absent = [key for key in required if not provenance.get(key)]
    if absent:
        raise ValueError(
            f"{path} carries no {absent} in its provenance; a baseline "
            "whose box, card, date, version and case are not stated is a "
            "number no later run can honestly compare itself against")
    return payload


__all__ = [
    "BOOT_STAGE", "CHAIN_EVENTS_FILENAME", "TTFP_FROM_MTIME",
    "TTFP_FROM_RECEIPT", "GoChainEvents", "load_baseline",
    "read_chain_events", "stage_walls", "summarize",
]
