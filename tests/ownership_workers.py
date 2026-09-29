"""Process bodies for tests/test_exclusive_ownership.py.

Kept in their own importable module because a spawned child process
imports its target by module name, and a test module's name depends on
how pytest collected it.
"""

from __future__ import annotations

import os
from pathlib import Path


def execute_dry_run(plan_path: str, gate, after_open, results) -> None:
    """Open the run's event stream, then run a dry-run plan into it."""

    import contextlib
    import io

    from gpuwm.runplan import (EVENTS_FILENAME, EventStream, PlanError,
                               execute_plan, load_plan)

    plan = load_plan(Path(plan_path))
    gate.wait()
    try:
        events = EventStream(plan.run_dir / EVENTS_FILENAME, mirror=None)
    except PlanError as error:
        after_open.wait()
        results.append(("refused", os.getpid(), str(error)))
        return
    after_open.wait()
    with events, contextlib.redirect_stdout(io.StringIO()):
        code = execute_plan(plan, events=events)
    results.append(("ran", os.getpid(), code))


def reserve(path: str, gate, after, results) -> None:
    """Reserve one downscale output folder and hold it until both tried."""

    from gpuwm.offline_child import (OfflineChildContractError,
                                     reserve_output_root)

    gate.wait()
    try:
        reserve_output_root(Path(path), flag="--out")
    except OfflineChildContractError as error:
        after.wait()
        results.append(("refused", os.getpid(), str(error)))
        return
    after.wait()
    results.append(("reserved", os.getpid(), ""))
