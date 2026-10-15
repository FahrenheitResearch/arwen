"""Suite verbs use the same real CLI as a terminal caller."""
from __future__ import annotations

from gpuwm.mcp.doors import TREE_ROOT, door_json


def register(server, manager):
    names = []

    def tool(fn):
        names.append(fn.__name__)
        server.tool(name=fn.__name__)(fn)
        return fn

    def query(args):
        document, code = door_json(["suite", *args])
        return {"exit_code": code, "document": document}

    @tool
    def suite_plan(file: str):
        """Plan the complete suite without launching any door."""
        return query(["plan", file, "--json"])

    @tool
    def suite_status(state_dir: str = ".suite-state"):
        """Per-row and per-box health, including a stale scheduler heartbeat."""
        return query(["status", "--state-dir", state_dir, "--json"])

    @tool
    def suite_halt(row: str, state_dir: str = ".suite-state"):
        """Persist an operator halt for one row."""
        return query(["halt", row, "--state-dir", state_dir])

    @tool
    def suite_resume(row: str, state_dir: str = ".suite-state"):
        """Release a row halt and resume its unfinished cycle."""
        return query(["resume", row, "--state-dir", state_dir])

    @tool
    def suite_rehearse(file: str, start: str, days: float = 1, state_dir: str = ".suite-state", bindings: str | None = None):
        """Replay past cycles as a job. The scheduler takes queue cards per door."""
        from gpuwm.mcp.engine_tools import _engine_prefix
        return manager.launch("suite-rehearse", [*_engine_prefix(), "suite", "rehearse", file,
            "--from", start, "--days", str(days), "--state-dir", state_dir,
            *(["--bindings", bindings] if bindings else [])], cwd=TREE_ROOT,
            gpu=False, outputs={"outdir": state_dir})

    @tool
    def suite_run(file: str, state_dir: str = ".suite-state", resume: bool = False, bindings: str | None = None):
        """Start the durable controller job; each GPU row uses its box queue."""
        from gpuwm.mcp.engine_tools import _engine_prefix
        return manager.launch("suite-run", [*_engine_prefix(), "suite", "run", file,
            "--state-dir", state_dir, *(["--resume"] if resume else []),
            *(["--bindings", bindings] if bindings else [])], cwd=TREE_ROOT,
            gpu=False, outputs={"outdir": state_dir})

    @tool
    def suite_events(state_dir: str = ".suite-state", after: int = 0):
        """Follow journal JSONL through the returned job's stdout stream."""
        from gpuwm.mcp.engine_tools import _engine_prefix
        return manager.launch("suite-events", [*_engine_prefix(), "suite", "events",
            "--state-dir", state_dir, "--after", str(after), "--follow"], cwd=TREE_ROOT,
            gpu=False, outputs={"outdir": state_dir})

    @tool
    def suite_apply(file: str, state_dir: str = ".suite-state", bindings: str | None = None):
        """Show and persist the admitted plan difference for the next boundary."""
        return query(["apply", file, "--state-dir", state_dir, *(["--bindings", bindings] if bindings else [])])

    return names
