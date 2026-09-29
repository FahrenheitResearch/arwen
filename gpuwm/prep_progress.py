"""Render the shared preparation events for human command-line hosts."""

from __future__ import annotations

import json
import math

from gpuwm.progress import PREP_EVENT_PREFIX, PREP_EVENT_SCHEMA

#: The actions a step record carries: a step's start and end, and how far a counted step has got.
STEP_ACTIONS = frozenset({"started", "finished", "failed", "progress"})


def step_record(text):
    """The step record one line of a preparer program's output carries, or None for any other line."""
    if not text.startswith(PREP_EVENT_PREFIX):
        return None
    try:
        record = json.loads(text[len(PREP_EVENT_PREFIX):])
    except ValueError:
        return None
    if (not isinstance(record, dict) or record.get("schema") != PREP_EVENT_SCHEMA
            or not isinstance(record.get("stage"), str) or not isinstance(record.get("label"), str)
            or record.get("event") not in STEP_ACTIONS):
        return None
    return record


class PrepProgress:
    def __init__(self):
        self.active = {}

    @property
    def label(self):
        return next(reversed(self.active.values()), "preparing")

    def line(self, text):
        """The words for one line of a preparer program's output, or None for any other line."""
        record = step_record(text)
        return None if record is None else self.event(record)

    def event(self, event):
        """The words for one step record, whether read off a program's output or heard in this process.

        A step taken in the process that holds the run reaches its host only as
        the ``preparation`` of a ``preparation_progress`` event
        (:func:`gpuwm.progress.prep_stage` with ``stderr=False``), so a host
        says it through here, as it says a program's line.
        """
        if not isinstance(event, dict) or event.get("schema") != PREP_EVENT_SCHEMA:
            return None
        stage, label, action = (event.get(key) for key in ("stage", "label", "event"))
        if not isinstance(stage, str) or not isinstance(label, str):
            return None
        if action not in {"started", "finished", "failed"}:
            return None
        label = " ".join(label.split())
        key = (stage, str(event.get("index", "")))
        if action == "started":
            self.active[key] = label
            backend = event.get("backend")
            suffix = f" ({backend})" if isinstance(backend, str) else ""
            return label + suffix
        self.active.pop(key, None)
        elapsed = event.get("elapsed_seconds")
        suffix = (f" ({elapsed:.1f} s)" if isinstance(elapsed, (int, float))
                  and math.isfinite(elapsed) and elapsed >= 0 else "")
        outcome = event.get("outcome")
        status = {"produced": "written", "not_requested": "not requested",
                  "refused": "not produced (native preparation continues)"}.get(outcome)
        if action == "failed":
            status = "failed"
        return label + ": " + (status or "done") + suffix
