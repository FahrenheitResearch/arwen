"""A warm per-card worker that runs successive forecast legs in one process.

THE COST THIS REMOVES.  A forecast leg launched as ``python -m
gpuwm.wrfinput_forecast ...`` pays, before its first model step, for two
interpreters (the door and its supervised worker), every import, a CUDA
context, loading each compiled kernel module from the disk cache, the
physics tables, and the first call of every scheme.  Measured on a
300 x 300 x 49 member on an RTX 4090: about 8 s from the command to the
first model step, against 50 ms per step after it.  A chain of short legs
(one per analysis cycle) pays it every time: a 5-minute leg is 20
steps, 1 s of stepping.

WHAT IT DOES.  One process, started on the card it should use (the caller
sets ``CUDA_VISIBLE_DEVICES``, as for ``gpuwm run --no-supervise``), runs
each leg through the WRF-input door's own ``main`` with the leg's own
arguments, in this process.  The files in and out are exactly a fresh
leg's: the same command line, the same outputs, written by the same code.
What is kept between legs is what a fresh process rebuilds identically:
imports, the CUDA context, compiled modules, packaged tables and the
memory pools.  No model state is carried from one leg to the next; each
leg restores its own inputs.

WHAT IT REFUSES.  A leg that fails ends the executor (exit 3) after its
result is recorded: a failure can leave the device in a state the next
leg must not inherit, and the caller relaunches a fresh executor.  A leg
that declares a different physics parameter set from an earlier one is
refused by the engine itself (``gpuwm.physics_params.declare``: one
process runs one set).  The worker-only flags (``--_worker``,
``--_output-owner``) are refused, because they name a supervisor this
process is not.

USE.  ``python -m gpuwm.leg_executor run LEGS.json [--receipt OUT.json]``
runs a list of legs in order; ``python -m gpuwm.leg_executor serve DIR``
runs legs as files appear in ``DIR/inbox`` (written by rename) and answers
in ``DIR/done``, until ``DIR/stop`` exists or ``--idle-exit`` seconds pass
with nothing to do.  A leg is ``{"name": ..., "argv": [...], "log": PATH}``
where ``argv`` is what follows ``python -m gpuwm.wrfinput_forecast`` and
``log`` (optional) receives the leg's stdout and stderr.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

LEGS_SCHEMA = "gpuwm.leg-executor.legs/v1"
RESULT_SCHEMA = "gpuwm.leg-executor.result/v1"
RECEIPT_SCHEMA = "gpuwm.leg-executor.receipt/v1"

#: Flags that belong to the door's supervisor, never to a leg.
_SUPERVISOR_ONLY = ("--_worker", "--_output-owner")


class LegRefused(ValueError):
    """A leg this executor will not run, named with the reason."""


def _check_leg(leg) -> tuple[str, list[str], Path | None]:
    if not isinstance(leg, dict):
        raise LegRefused(f"a leg is a JSON object, not {type(leg).__name__}")
    argv = leg.get("argv")
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        raise LegRefused("a leg's argv is a list of strings (the arguments after "
                         "python -m gpuwm.wrfinput_forecast)")
    for flag in _SUPERVISOR_ONLY:
        if any(a == flag or a.startswith(flag + "=") for a in argv):
            raise LegRefused(f"{flag} names the door's supervised worker; the "
                             "executor runs the leg itself")
    name = str(leg.get("name") or f"leg-{abs(hash(tuple(argv))) % 10**8:08d}")
    log = leg.get("log")
    return name, list(argv), (None if log is None else Path(log))


@contextlib.contextmanager
def _redirected(log: Path | None):
    """The leg's stdout and stderr (Python and C level) into ``log``."""
    if log is None:
        yield
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        os.close(fd)
    # Python-level writes go to the log through their own stream as well,
    # so a replaced sys.stdout (a host process capturing it) is covered too.
    stream = open(log, "a", buffering=1, encoding="utf-8", errors="replace")
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            yield
    finally:
        try:
            stream.flush()
            stream.close()
        finally:
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def run_leg(leg) -> dict:
    """Run one leg in this process; return its result record."""
    from gpuwm import wrfinput_forecast

    name, argv, log = _check_leg(leg)
    started = time.time()
    began = time.perf_counter()
    rc = None
    error = None
    with _redirected(log):
        try:
            rc = int(wrfinput_forecast.main(argv, in_process=True) or 0)
        except SystemExit as stop:  # argparse and explicit exits
            code = stop.code
            rc = code if isinstance(code, int) else (0 if code is None else 2)
        except BaseException as failure:  # noqa: BLE001 - recorded, then the executor stops
            rc = 1
            error = f"{type(failure).__name__}: {failure}"
            traceback.print_exc()
            if isinstance(failure, KeyboardInterrupt):
                raise
    # What a finished leg leaves behind is garbage, not state: collect it
    # here so the next leg's peak is its own.
    gc.collect()
    return {"schema": RESULT_SCHEMA, "name": name, "argv": argv, "rc": rc,
            "error": error, "started_unix": round(started, 3),
            "wall_seconds": round(time.perf_counter() - began, 3),
            "pid": os.getpid()}


def run_legs(legs, *, receipt: Path | None = None) -> int:
    """Run ``legs`` in order; stop at the first failure.  Returns the exit code."""
    results = []
    began = time.perf_counter()
    status = 0
    for leg in legs:
        result = run_leg(leg)
        results.append(result)
        print(f"leg-executor: {result['name']} rc {result['rc']} "
              f"{result['wall_seconds']:.2f} s", flush=True)
        if result["rc"] != 0:
            status = 3
            break
    if receipt is not None:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({
            "schema": RECEIPT_SCHEMA, "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "legs": results, "stopped_at_failure": status != 0,
            "wall_seconds": round(time.perf_counter() - began, 3)}, indent=1) + "\n",
            encoding="utf-8")
    return status


def serve(spool: Path, *, idle_exit: float | None = None, poll: float = 0.2) -> int:
    """Run legs from ``spool/inbox`` as they appear, oldest name first."""
    inbox, running, done = spool / "inbox", spool / "running", spool / "done"
    for folder in (inbox, running, done):
        folder.mkdir(parents=True, exist_ok=True)
    (spool / "ready").write_text(json.dumps({"pid": os.getpid()}) + "\n", encoding="utf-8")
    idle_since = time.monotonic()
    while not (spool / "stop").exists():
        waiting = sorted(p for p in inbox.glob("*.json"))
        if not waiting:
            if idle_exit is not None and time.monotonic() - idle_since > idle_exit:
                return 0
            time.sleep(poll)
            continue
        request = waiting[0]
        claimed = running / request.name
        try:
            os.replace(request, claimed)
        except OSError:
            continue
        refused = False
        try:
            leg = json.loads(claimed.read_text(encoding="utf-8"))
            _check_leg(leg)
        except (ValueError, OSError) as refusal:
            # The request itself is malformed: nothing ran, keep serving.
            refused = True
            result = {"schema": RESULT_SCHEMA, "name": request.stem, "rc": 2,
                      "error": f"{type(refusal).__name__}: {refusal}"}
        if not refused:
            result = run_leg(leg)
        answer = done / request.name
        temporary = answer.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        os.replace(temporary, answer)
        claimed.unlink(missing_ok=True)
        idle_since = time.monotonic()
        if not refused and result.get("rc") != 0:
            # A leg that ran and failed ends this executor (see the module
            # docstring); the caller starts a fresh one.
            return 3
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m gpuwm.leg_executor",
                                     description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("run", help="run the legs listed in a JSON file, in order")
    run.add_argument("legs", type=Path, help=f"{{'schema': '{LEGS_SCHEMA}', 'legs': [...]}}")
    run.add_argument("--receipt", type=Path, default=None)
    srv = sub.add_parser("serve", help="run legs as files appear in DIR/inbox")
    srv.add_argument("spool", type=Path)
    srv.add_argument("--idle-exit", type=float, default=None, metavar="SECONDS")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from gpuwm import kernel_cache_seed, kernel_prebuild
    kernel_cache_seed.seed()
    kernel_prebuild.install_recorder()
    if args.mode == "run":
        document = json.loads(Path(args.legs).read_text(encoding="utf-8"))
        if document.get("schema") != LEGS_SCHEMA:
            print(f"leg-executor: {args.legs} is not a {LEGS_SCHEMA} document",
                  file=sys.stderr)
            return 2
        try:
            for leg in document.get("legs", ()):
                _check_leg(leg)
        except LegRefused as refusal:
            print(f"leg-executor: refused: {refusal}", file=sys.stderr)
            return 2
        return run_legs(document["legs"], receipt=args.receipt)
    return serve(args.spool, idle_exit=args.idle_exit)


if __name__ == "__main__":
    raise SystemExit(main())
