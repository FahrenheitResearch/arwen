"""Render the shared preparation events for human command-line hosts."""

from __future__ import annotations

import json
import math

from gpuwm.progress import PREP_EVENT_PREFIX, PREP_EVENT_SCHEMA


class PrepProgress:
    def __init__(self):
        self.active = {}

    @property
    def label(self):
        return next(reversed(self.active.values()), "preparing")

    def line(self, text):
        if not text.startswith(PREP_EVENT_PREFIX):
            return None
        try:
            event = json.loads(text[len(PREP_EVENT_PREFIX):])
        except ValueError:
            return None
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
