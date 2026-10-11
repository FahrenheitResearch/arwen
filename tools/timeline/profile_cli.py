"""Diagnostic C1 call budget around the real CLI, without reading model fields.

This measurement does not replace paired process walls. It isolates time spent
inside the added ledger methods when shared-box timings cannot resolve it.
Nested method calls are counted once per thread; sums across threads are a
conservative occupied-time budget, including forwarded observer work.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class Budget:
    def __init__(self):
        self.local = threading.local()
        self.lock = threading.Lock()
        self.rows = {}

    def wrap(self, name, function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            depth = getattr(self.local, "depth", 0)
            self.local.depth = depth + 1
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                self.local.depth = depth
                with self.lock:
                    row = self.rows.setdefault(name, {"calls": 0, "inclusive_seconds": 0.0,
                                                       "outermost_seconds": 0.0})
                    row["calls"] += 1
                    row["inclusive_seconds"] += elapsed
                    if depth == 0:
                        row["outermost_seconds"] += elapsed
        return call

    def scope(self, decorator):
        """Charge ledger scope setup/cleanup while excluding the real work."""
        def decorate(function):
            @functools.wraps(function)
            def call(*args, **kwargs):
                work = [0.0]
                @functools.wraps(function)
                def body(*body_args, **body_kwargs):
                    start = time.perf_counter()
                    try:
                        return function(*body_args, **body_kwargs)
                    finally:
                        work[0] += time.perf_counter() - start
                wrapped = decorator(body)
                start = time.perf_counter()
                try:
                    return wrapped(*args, **kwargs)
                finally:
                    elapsed = max(0.0, time.perf_counter() - start - work[0])
                    with self.lock:
                        row = self.rows.setdefault("scope:" + function.__name__,
                            {"calls": 0, "inclusive_seconds": 0.0, "outermost_seconds": 0.0})
                        row["calls"] += 1
                        row["inclusive_seconds"] += elapsed
                        row["outermost_seconds"] += elapsed
            return call
        return decorate

    @property
    def occupied_seconds(self):
        return sum(row["outermost_seconds"] for row in self.rows.values())


def install(budget):
    import gpuwm.timeline.ledger as ledger
    import gpuwm.stage_timing as timing
    for cls in (ledger.Ledger, timing.ForecastLedger, timing.LedgerObserver):
        for name, function in list(vars(cls).items()):
            if (callable(function) and name != "_watch"
                    and (not name.startswith("__") or name == "__init__")):
                setattr(cls, name, budget.wrap(cls.__name__ + "." + name, function))
    for name in ("register_ledger_observer", "current_ledger", "end_admission", "bind_ledger"):
        setattr(timing, name, budget.wrap(name, getattr(timing, name)))
    timing.ledger_main = budget.scope(timing.ledger_main)


def main(argv=None):
    destination = os.environ.get("GPUWM_C1_PROFILE_RECEIPT")
    if not destination:
        raise ValueError("C1 profile receipt path is required; an unrecorded diagnostic cannot prove overhead")
    # Package initialization already exists in b287. Charge all subsequent C1
    # imports and installation conservatively, including this diagnostic setup.
    import gpuwm  # noqa: F401
    budget = Budget()
    start = time.perf_counter()
    install(budget)
    setup = time.perf_counter() - start
    from gpuwm.cli import main as cli_main
    start = time.perf_counter()
    status = None
    try:
        status = cli_main(sys.argv[1:] if argv is None else argv)
        return status
    finally:
        wall = time.perf_counter() - start
        occupied = setup + budget.occupied_seconds
        receipt = {"schema": "c1-real-call-budget-v1", "cli_returncode": status,
                   "setup_seconds": setup, "cli_wall_seconds": wall,
                   "occupied_seconds": occupied, "occupied_percent": 100 * occupied / wall,
                   "method": "outermost C1 methods per thread plus exclusive scope bookkeeping and setup; closing methods may be charged twice conservatively",
                   "excludes": "monitor sleep and real CLI work; paired process walls remain separate",
                   "calls": budget.rows}
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
