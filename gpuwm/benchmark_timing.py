"""``gpuwm run --benchmark``: a WRF-comparable timing summary of one run.

A wall-clock benchmark of the WRF-input door used to time the post-run
render and observation verification too (14 to 33 s on NCAR's CONUS 12 km
1 h restart leg, 70 s on its 12 h leg), and the door had no way to leave
them out or to say where the rest of the wall clock went (NV recipe dry
run, D6).  ``--benchmark`` draws no products and prints, after the run,
the phases WRF's own ``rsl.out`` reports: read, init, the first step, the
remaining steps and the final write, plus the process total.

The phases are cut from marks this run already records: the process start
time, two marks the worker takes around the input read, and the
``emitted_unix_ms`` of the step log's records (``progress.jsonl``).
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

#: The worker's marks, written beside the run's outputs.
MARKS_NAME = "benchmark-marks.json"
#: The summary, as printed, in machine form.
SUMMARY_NAME = "benchmark-timing.json"
SCHEMA = "gpuwm.benchmark-timing.v1"


def process_start_unix() -> float:
    """When this process started, in Unix seconds.

    Linux reads the kernel's own start tick, so interpreter start-up and
    imports are inside the total, as they are inside a ``time`` of the
    command.  Elsewhere psutil when present, else now.
    """

    try:
        # The process's age from the kernel's own clocks (start tick and
        # uptime, both since boot), subtracted from now: /proc/stat's btime
        # is whole seconds and would put up to 1 s of error in the total.
        stat = Path("/proc/self/stat").read_text()
        ticks = int(stat.rsplit(")", 1)[1].split()[19])
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        age = uptime - ticks / os.sysconf("SC_CLK_TCK")
        if age >= 0.0:
            return time.time() - age
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    try:
        import psutil

        return float(psutil.Process().create_time())
    except Exception:  # noqa: BLE001 - optional dependency
        return time.time()


class Marks:
    """The worker's marks around the input read, in Unix seconds."""

    def __init__(self):
        self.values: dict = {"worker_start": process_start_unix()}

    def mark(self, name: str) -> None:
        self.values[name] = time.time()

    def write(self, outdir) -> Path:
        path = Path(outdir) / MARKS_NAME
        path.write_text(json.dumps(self.values, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")
        return path


def _records(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize(outdir, *, process_start: float, process_end: float | None = None,
              marks: dict | None = None) -> dict:
    """The phases of one finished run, from its marks and step log.

    Raises ``ValueError`` naming what is missing when the run left no
    step log or no step, because a summary with an invented phase is worse
    than none.
    """

    outdir = Path(outdir)
    if marks is None:
        marks = json.loads((outdir / MARKS_NAME).read_text(encoding="utf-8"))
    log = outdir / "progress.jsonl"
    if not log.is_file():
        raise ValueError(f"{log} is missing; the run kept no step log "
                         "(--progress-format off)")
    records = _records(log)
    root_steps = {}
    for record in records:
        if record.get("event") == "step" and record.get("step_wall_seconds") is not None:
            root_steps.setdefault(int(record["domain"]), []).append(record)
    if not root_steps:
        raise ValueError(f"{log} holds no completed step")
    root = min(root_steps)
    steps = root_steps[root]
    if len(steps) != len({r["step"] for r in steps}):
        raise ValueError(f"{log} repeats a step; a resumed or appended log "
                         "cannot be timed as one run")
    start = next((r for r in records if r.get("event") == "run_start"), {})
    end = next((r for r in reversed(records) if r.get("event") == "run_end"), None)
    if end is None:
        raise ValueError(f"{log} has no run_end; the run did not finish")
    if process_end is None:
        process_end = time.time()
    first, last = steps[0], steps[-1]
    t_first = first["emitted_unix_ms"] / 1000.0
    t_last = last["emitted_unix_ms"] / 1000.0
    t_end = end["emitted_unix_ms"] / 1000.0
    first_wall = float(first["step_wall_seconds"])
    rest = [float(r["step_wall_seconds"]) for r in steps[1:]]
    read_start = float(marks.get("read_start", marks["worker_start"]))
    read_end = float(marks["read_end"])
    phases = {
        "startup": read_start - process_start,
        "read": read_end - read_start,
        "init": (t_first - first_wall) - read_end,
        "first_step": first_wall,
        "steps": t_last - t_first,
        "write": t_end - t_last,
    }
    # After run_end: the run receipt, the certification capsule and the
    # input re-hash, then the worker's exit.  Named so the rows add up to
    # the total.
    phases["finish"] = process_end - t_end
    phases["forecast"] = t_end - read_start
    phases["total"] = process_end - process_start
    restart = start.get("restart") or {}
    domain_end = next((r for r in records if r.get("event") == "domain_end"
                       and int(r.get("domain", -1)) == root), {})
    return {
        "schema": SCHEMA,
        "phases_seconds": {k: round(v, 3) for k, v in phases.items()},
        "steps": len(steps),
        "first_step": int(first["step"]),
        "last_step": int(last["step"]),
        "steps_after_first": len(rest),
        "median_step_seconds": (round(statistics.median(rest), 6) if rest else None),
        "mean_step_seconds": (round(statistics.fmean(rest), 6) if rest else None),
        "max_step_seconds": (round(max(rest), 6) if rest else None),
        "first_valid_time": first.get("valid_time"),
        "last_valid_time": domain_end.get("valid_time", last.get("valid_time")),
        "restart": restart or None,
        "status": end.get("status"),
        "excluded": ["render", "observation verification"],
        "kernel_prewarm": marks.get("kernel_prewarm"),
    }


def format_summary(summary: dict) -> str:
    p = summary["phases_seconds"]
    n = summary["steps_after_first"]
    median = summary["median_step_seconds"]
    lines = [
        f"gpuwm benchmark: {summary['steps']} steps "
        f"({summary['first_step']} to {summary['last_step']}), "
        f"{summary['first_valid_time']} to {summary['last_valid_time']}; "
        "render and verification skipped",
        f"  startup     {p['startup']:9.2f} s   process start, card selection, imports",
        f"  read        {p['read']:9.2f} s   input import (wrfinput or wrfrst, wrfbdy, namelist)",
        f"  init        {p['init']:9.2f} s   device restore, physics init, first frame",
        f"  first step  {p['first_step']:9.2f} s",
        (f"  steps       {p['steps']:9.2f} s   {n} steps after the first, median "
         f"{median * 1000.0:.1f} ms, mean {summary['mean_step_seconds'] * 1000.0:.1f} ms"
         if n else f"  steps       {p['steps']:9.2f} s   no steps after the first"),
        f"  write       {p['write']:9.2f} s   last step to the last history frame on disk",
        f"  finish      {p['finish']:9.2f} s   run receipt, certification capsule, worker exit",
        f"  total       {p['total']:9.2f} s   process start to end "
        f"(read to forecast end {p['forecast']:.2f} s)",
    ]
    return "\n".join(lines)


def report(outdir, *, process_start: float, stream=None) -> dict:
    """Summarize, print and record one finished benchmark run."""

    summary = summarize(outdir, process_start=process_start)
    path = Path(outdir) / SUMMARY_NAME
    path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(format_summary(summary), file=stream or sys.stdout, flush=True)
    return summary


__all__ = ["MARKS_NAME", "Marks", "SCHEMA", "SUMMARY_NAME", "format_summary",
           "process_start_unix", "report", "summarize"]
