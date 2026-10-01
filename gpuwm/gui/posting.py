"""A forecast run as its source posts: what New forecast shows before Start, and what Start sends.

The schedule is the engine's own readiness answer (``gpuwm run-plan
PLAN.json --readiness``, or ``gpuwm fetch ... --readiness`` for a draft
whose configuration is written only at Start; document
``gpuwm.readiness.v1``): which hours the run needs before it can start and
when each is expected, when the cycle's last hour is expected, whether the
start hours are posted now, and how late an hour may be before the run
stops with its frames kept.  As posted is the default; "wait for the whole
cycle" is the opt-out, sent as run-plan's ``as_posted = false`` run option.

Both are read from the engine itself (:func:`gpuwm.mcp.doors.as_posted_support`).
An engine that does not answer ``--readiness``, or whose run plans do not
take ``as_posted``, starts every forecast on the whole cycle: the page says
so instead of drawing a schedule nobody computed, and a plan never carries
an option its engine would refuse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

READINESS_SCHEMA = "gpuwm.readiness.v1"
#: The readiness answer's exit codes: 0 ready (or a source that cannot be asked), 75 not ready yet, 2 refused.
#: Each prints the document.
READINESS_CODES = (0, 75, 2)
#: The fields of a readiness document the page draws.
KEPT = ("state", "ready", "source", "cycle", "window", "expected_ready_at", "expected_final_at", "retry_after_seconds",
        "refusal", "checked_at", "cycle_basis")
POSTING_KEPT = ("shape", "streams", "why", "late_after_minutes")
NEED_KEPT = ("role", "source", "lead", "expected_at", "late_at", "answer")


def support() -> dict[str, Any]:
    """What this engine offers for a run as its source posts (see :func:`gpuwm.mcp.doors.as_posted_support`)."""

    from gpuwm.mcp.doors import as_posted_support

    return as_posted_support()


def runs_as_posted(route: str) -> bool:
    """Whether a plan on ``route`` runs as its source posts on this engine, and can opt out."""

    return route in (support().get("run_plan_as_posted") or ())


def plan_options(route: str, as_posted: bool) -> dict[str, Any]:
    """The run options a plan on ``route`` carries for the page's choice.

    Nothing for as posted, the engine's default.  ``as_posted: false`` for
    "wait for the whole cycle" where the route takes the option; an engine
    whose route does not take it already waits for the whole cycle, and
    would refuse the key.
    """

    return {"as_posted": False} if not as_posted and runs_as_posted(route) else {}


def unavailable(as_posted: bool) -> dict[str, Any]:
    """The page's schedule on an engine that runs every start on the whole cycle."""

    return {"available": False, "asked_as_posted": as_posted, "as_posted": False,
            "why": "whole_cycle_engine"}


def facts(document: Any, *, as_posted: bool, code: int | None = None) -> dict[str, Any]:
    """The schedule the page draws, out of one readiness document."""

    if not isinstance(document, dict) or document.get("schema") != READINESS_SCHEMA:
        return {"available": True, "asked_as_posted": as_posted, "as_posted": as_posted,
                "error": "The engine's readiness answer was not a gpuwm.readiness.v1 document."}
    out: dict[str, Any] = {key: document.get(key) for key in KEPT}
    posting = document.get("posting") if isinstance(document.get("posting"), dict) else {}
    out["posting"] = {key: posting.get(key) for key in POSTING_KEPT}
    out["start_needs"] = [{key: need.get(key) for key in NEED_KEPT}
                          for need in document.get("start_needs") or () if isinstance(need, dict)]
    out.update(available=True, asked_as_posted=as_posted, as_posted=bool(document.get("as_posted", as_posted)),
               exit_code=code)
    return out


def readiness_argv(engine_argv, *, plan: Path | None = None, draft: dict[str, Any] | None = None,
                   as_posted: bool = True) -> list[str]:
    """The readiness question for a written plan, or for a draft's source window when its plan is written at Start.

    A plan carries its own choice (:func:`plan_options`); a source window
    asked through ``gpuwm fetch`` takes ``--whole-cycle`` for the opt-out.
    """

    if plan is not None:
        return engine_argv("run-plan", str(plan), "--readiness")
    assert draft is not None
    argv = engine_argv("fetch", "--source", draft["source"], "--cycle", draft["cycle"],
                       "--hours", str(int(draft["hours"])))
    if draft.get("start_hour"):
        argv += ["--forecast-start-hour", str(int(draft["start_hour"]))]
    if not as_posted and support().get("fetch_whole_cycle"):
        argv.append("--whole-cycle")
    return [*argv, "--readiness"]


def ask(runner, argv: list[str], *, cwd: Path | None, as_posted: bool) -> dict[str, Any]:
    """Ask the engine and read its answer; a question that could not be answered is said, never raised.

    The schedule is information beside Start: the run's own start probe
    decides when the forecast starts, so a schedule that cannot be read
    (a host not heard, a slow engine) leaves Start as it was.
    """

    try:
        code, document = runner.answer(argv, cwd=cwd, codes=READINESS_CODES)
    except Exception as error:  # noqa: BLE001 - see the docstring: a schedule never stops a start
        return {"available": True, "asked_as_posted": as_posted, "as_posted": as_posted,
                "error": str(error) or type(error).__name__}
    return facts(document, as_posted=as_posted, code=code)


__all__ = ["READINESS_CODES", "READINESS_SCHEMA", "ask", "facts", "plan_options", "readiness_argv", "runs_as_posted",
           "support", "unavailable"]
