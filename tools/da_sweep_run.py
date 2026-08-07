"""Run a staged DA sweep unattended, behind whatever already owns the card.

The sweep this drives turns three estimates into measurements: what a
WoFS-sized ensemble costs at our current grid spacing, what finer spacing
costs at a storm-scale footprint, and what the localization radius costs
when it is held at a fixed physical size while the grid refines.

Three properties matter more than speed here, and each is a deliberate
choice rather than a default:

**It never contends and it never kills.**  Before every arm it asks
``gpuwm run``'s own admission predicate whether a launch would be allowed
(:mod:`gpuwm.supervisor.preflight_exclusive_gpu`, via ``--gate``), and if
the answer is no it waits and asks again.  Polling the same function the
launch will call makes the wait condition and the refusal condition the
same condition by construction, so the queue cannot clear its own gate
and then be refused.  A queue that gives up is a *recorded verdict*, not
a silent exit.

**It captures stderr.**  A sibling queue in this tree lost the entire
diagnosis of a failed run because it piped only stdout, leaving a
zero-byte log beside a nonzero exit code.  Every step here gets both
streams, to separate files, and the tail of stderr is copied into the
status log on failure so the status file alone is enough to say what
broke.

**It is resumable and it is per-arm fail-soft.**  Each arm writes a
``.done`` marker; a rerun skips what already finished.  An arm that fails
is recorded and the queue moves to the next one, because five arms where
three succeeded is a result and five arms where the night aborted on the
first is not.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "gpuwm-da.sweep-queue.v1"


class Status:
    """The durable status file. Appended to, never rewritten."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def say(self, message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)


def gate_clear(gate: Path, log: Path) -> tuple[bool, str]:
    """Would a forecast be admitted right now?  Fail closed."""

    proc = subprocess.run([sys.executable, str(gate)],
                          capture_output=True, text=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now(timezone.utc).isoformat()} "
                     f"exit {proc.returncode} {proc.stdout.strip()}"
                     f"{proc.stderr.strip()}\n")
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def wait_for_predecessors(status: Status, files: list[Path], *,
                          poll_seconds: int, deadline: float) -> bool:
    """Block until every predecessor status file exists.

    Existence is the contract, not content: the file this waits on is
    written by its owner exactly once, at the end, whatever the outcome.
    A predecessor that failed has still finished with the card.
    """

    announced = False
    while True:
        missing = [f for f in files if not f.is_file()]
        if not missing:
            if announced:
                status.say("predecessors complete")
            return True
        if not announced:
            status.say("waiting on predecessor status file(s): "
                       + ", ".join(str(m) for m in missing))
            announced = True
        if time.monotonic() > deadline:
            status.say(f"predecessor(s) never completed: "
                       + ", ".join(str(m) for m in missing))
            return False
        time.sleep(poll_seconds)


def wait_for_card(status: Status, gate: Path, gate_log: Path, *, label: str,
                  poll_seconds: int, deadline: float) -> bool:
    polls = 0
    while True:
        clear, detail = gate_clear(gate, gate_log)
        if clear:
            status.say(f"{label}: card clear after {polls} poll(s)")
            return True
        if time.monotonic() > deadline:
            status.say(f"{label}: card still held at the deadline; "
                       f"nothing was run and nothing was stopped. {detail}")
            return False
        if polls % 10 == 0:
            status.say(f"{label}: card busy - waiting, killing nothing. "
                       f"{detail}")
        polls += 1
        time.sleep(poll_seconds)


def run_step(status: Status, step: dict, *, run_dir: Path, repo: Path,
             env: dict, arm: str, index: int) -> int:
    """One subprocess, both streams captured to their own files."""

    argv = [a.replace("${RUN_DIR}", str(run_dir)).replace("${REPO}", str(repo))
            for a in step["argv"]]
    if argv and argv[0] == "${PYTHON}":
        argv[0] = sys.executable
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    out_path = logs / f"{arm}.{index:02d}.{step['name']}.out"
    err_path = logs / f"{arm}.{index:02d}.{step['name']}.err"
    status.say(f"{arm}: step {step['name']} start")
    started = time.monotonic()
    with out_path.open("w", encoding="utf-8") as out, \
            err_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(argv, cwd=str(repo), env=env,
                              stdout=out, stderr=err, text=True)
    elapsed = time.monotonic() - started
    status.say(f"{arm}: step {step['name']} exit {proc.returncode} "
               f"after {elapsed:.1f} s")
    if proc.returncode != 0:
        # The whole reason this queue exists in this form: say what broke,
        # in the status file, without needing the log tree.
        tail = err_path.read_text(encoding="utf-8", errors="replace")
        tail = tail.strip().splitlines()[-12:]
        for line in tail:
            status.say(f"{arm}:   stderr| {line}")
        if not tail:
            head = out_path.read_text(encoding="utf-8", errors="replace")
            for line in head.strip().splitlines()[-12:]:
                status.say(f"{arm}:   stdout| {line}")
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_sweep_run",
        description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True,
                        help="the GPU admission gate script")
    parser.add_argument("--wait-for", type=Path, action="append", default=[],
                        help="a predecessor's status file; the queue does "
                             "not start until each one exists")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-wait-hours", type=float, default=10.0)
    parser.add_argument("--only", action="append", default=[],
                        help="run only these arm names")
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    status = Status(run_dir / "queue-status.log")
    verdict_path = run_dir / "VERDICT.txt"
    gate_log = run_dir / "logs" / "gate.log"
    gate_log.parent.mkdir(parents=True, exist_ok=True)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    status.say(f"QUEUE START pid {os.getpid()} plan {args.plan}")
    status.say(f"plan {plan.get('schema')} arms "
               f"{[a['name'] for a in plan['arms']]}")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.repo)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("GPUWM_GEOG_ROOT", str(Path.home() / "WPS_GEOG"))

    deadline = time.monotonic() + args.max_wait_hours * 3600.0
    if args.wait_for and not wait_for_predecessors(
            status, list(args.wait_for), poll_seconds=args.poll_seconds,
            deadline=deadline):
        verdict = ("PREDECESSOR_TIMEOUT a predecessor queue never wrote its "
                   "status file; nothing was run and nothing was stopped")
        status.say("VERDICT " + verdict)
        verdict_path.write_text(verdict + "\n", encoding="utf-8")
        return 0

    outcomes = {}
    for arm in plan["arms"]:
        name = arm["name"]
        if args.only and name not in args.only:
            continue
        marker = run_dir / "arms" / f"{name}.done"
        marker.parent.mkdir(parents=True, exist_ok=True)
        if marker.is_file():
            status.say(f"{name}: already done - skipping")
            outcomes[name] = "skipped (already done)"
            continue

        status.say(f"{name}: {arm['what']}")
        # An arm that never touches the card must not wait for it.  Radar
        # fetching, decoding and superobbing are network and CPU; holding
        # the gate's verdict while they run would idle the card for tens
        # of minutes and starve whoever is actually queued behind it.
        # Absent flag means True, so every existing plan behaves as before.
        if arm.get("needs_gpu", True):
            if not wait_for_card(status, args.gate, gate_log, label=name,
                                 poll_seconds=args.poll_seconds,
                                 deadline=deadline):
                outcomes[name] = "not run (card held to the deadline)"
                break
        else:
            status.say(f"{name}: declares no GPU need - not gating on the "
                       "card, and not touching it")

        started = time.monotonic()
        failed = None
        for index, step in enumerate(arm["steps"]):
            code = run_step(status, step, run_dir=run_dir, repo=args.repo,
                            env=env, arm=name, index=index)
            if code != 0:
                failed = f"{step['name']} exit {code}"
                break
        elapsed = time.monotonic() - started
        if failed is None:
            marker.write_text(f"DONE after {elapsed:.1f} s\n",
                              encoding="utf-8")
            outcomes[name] = f"OK in {elapsed:.1f} s"
            status.say(f"{name}: COMPLETE in {elapsed:.1f} s")
        else:
            outcomes[name] = f"FAILED ({failed}) after {elapsed:.1f} s"
            status.say(f"{name}: FAILED ({failed}) - continuing to the "
                       "next arm")

    summary = {"schema": SCHEMA,
               "finished": datetime.now(timezone.utc).isoformat(),
               "outcomes": outcomes,
               "results_dir": str(run_dir / "results")}
    (run_dir / "queue-summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")

    ok = [n for n, o in outcomes.items() if o.startswith("OK")]
    bad = [n for n, o in outcomes.items() if o.startswith("FAILED")]
    verdict = (f"SWEEP_COMPLETE {len(ok)}/{len(outcomes)} arm(s) ran clean"
               + (f"; failed: {', '.join(bad)}" if bad else "")
               + f". Numbers in {run_dir / 'results'}")
    status.say("VERDICT " + verdict)
    verdict_path.write_text(verdict + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
