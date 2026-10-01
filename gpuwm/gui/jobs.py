"""Every button is one ``gpuwm`` command: build it, show it, run it.

Two kinds of command:

- a QUERY answers in a few seconds with one JSON document on stdout
  (``gpuwm run-plan --sources``, ``--physics-profiles``, ``PLAN --resolve``)
  and runs in the foreground of one request;
- a RUN (``gpuwm run-plan PLAN.json``) is launched DETACHED through the
  engine's one job manager, :class:`gpuwm.mcp.jobs.JobManager`: its
  receipts live on disk, a second GPU run is refused by a sentence naming
  the running one (the same lock the MCP tools take), and closing the
  page or this server never stops it.

Every command that ran is appended to the run's ``commands.log`` as one
line you can paste into a terminal to run it again.

Stop.  ``gpuwm run-plan`` has no control file to poll, so Stop is the job
manager's stop by process: on Linux and macOS a Ctrl+C (SIGINT) to the
run's process group, which the run answers by writing its ``stopped``
event and exiting 130; on Windows the job manager ends the process tree
(``taskkill /T``), because a detached process there has no console a
Ctrl+C could reach.  The reply says which one happened.

Stop signals only a process this run recorded AND that is still that
process: the job record and the manifest keep each PID with its creation
time (:mod:`gpuwm.proc_identity`), and a PID that now names another
program (reused after a crash or a reboot) is refused, not signalled.
The escalation a minute later checks again, because the run can end and
its PID be handed to someone else inside that minute.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
from typing import Any

from gpuwm import proc_identity

from .files import read_json, utc_text, write_json
from .runs import COMMANDS_LOG, JOB, MANIFEST, STOP, job_alive, job_of, manifest_alive

QUERY_TIMEOUT_S = 180.0
#: A downscale review reads the parent's frames, measures the card and asks the renderer for its products.
REVIEW_TIMEOUT_S = 300.0
#: A run that still lives this long after its Ctrl+C has its tree ended.
ESCALATE_AFTER_S = 60.0


class Refused(Exception):
    """A command the engine or the job manager refused; the message says why.

    ``document`` is the JSON object the command printed on stdout before
    it exited nonzero, when it printed one: a ``--json`` command states
    its refusal there, and dropping it left the page with only the
    exit code.
    """

    def __init__(self, message: str = "", document: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.document = document


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        document = json.loads(text)
    except ValueError:
        return None
    return document if isinstance(document, dict) else None


def engine_argv(*args: str) -> list[str]:
    """``python -P -m gpuwm ...`` with this interpreter: the engine this server runs on.

    ``-P`` keeps the working folder off the child's import path, so a
    gpuwm checkout in the folder the page was started from cannot answer
    in place of the installed one.  A source or editable install is put
    back on the path by :func:`engine_env`.
    """

    return [sys.executable, "-P", "-m", "gpuwm", *[str(arg) for arg in args]]


def engine_env() -> dict[str, str]:
    """Environment additions an :func:`engine_argv` child needs to import the gpuwm serving this page.

    ``PYTHONSAFEPATH`` carries ``-P`` to every Python the child starts in
    turn: a query runs in the server's own folder, and a grandchild started
    as ``python -m ...`` without ``-P`` would put that folder, and any gpuwm
    checkout in it, first on its path (``gpuwm go`` sets it for its stages
    for the same reason).  A source or editable install also names its root
    on ``PYTHONPATH``.
    """

    from gpuwm.runtime_manifest import child_python_env

    return {"PYTHONSAFEPATH": "1", **child_python_env()}


def display(argv: list[str]) -> str:
    """The command line, quoted for this computer's shell, as a terminal can run it again.

    An engine line is the exact one, except that a gpuwm served from a
    source tree is written without ``-P``: its child was handed the tree's
    root on PYTHONPATH, which a pasted line does not carry, while without
    ``-P`` the line run in the tree's folder imports that tree.  A wheel
    install keeps ``-P``: that line runs the same from any folder.
    """

    argv = [str(arg) for arg in argv]
    if len(argv) >= 4 and argv[0] == sys.executable and argv[1:3] == ["-P", "-m"] \
            and argv[3].split(".")[0] == "gpuwm" and "PYTHONPATH" in engine_env():
        argv = [argv[0], *argv[2:]]
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def _plain_arg(arg: str, roots: tuple[Path, ...]) -> str:
    flag, eq, value = arg.partition("=") if arg.startswith("-") and "=" in arg else ("", "", arg)
    path = Path(value)
    if not value or not path.is_absolute():
        return arg
    for root in roots:
        try:
            value = str(path.relative_to(root))
            break
        except ValueError:
            continue
    else:
        try:
            value = str(Path("~") / path.relative_to(Path.home()))
        except ValueError:
            value = path.name or value
    return f"{flag}{eq}{value}"


def plain_command(argv: list[str], root: Path | None = None) -> str:
    """The command as the page shows it: ``gpuwm run-plan NAME/plan.json``.

    The interpreter becomes ``gpuwm`` and a path in the forecasts folder is
    written from that folder, so no machine path is on screen.  The exact
    line (:func:`display`) is what Copy takes and what ``commands.log`` keeps.
    """

    args = [str(arg) for arg in argv]
    if len(args) >= 4 and args[1:4] == ["-P", "-m", "gpuwm"]:
        args = ["gpuwm", *args[4:]]
    elif len(args) >= 3 and args[1:3] == ["-m", "gpuwm"]:
        args = ["gpuwm", *args[3:]]
    roots: tuple[Path, ...] = ()
    if root is not None:
        roots = (Path(root),)
        try:
            resolved = Path(root).resolve()
            if resolved != Path(root):
                roots += (resolved,)
        except OSError:
            pass
    return display([args[0], *[_plain_arg(arg, roots) for arg in args[1:]]]) if args else ""


def plain_log(text: str, root: Path | None = None) -> str:
    """``commands.log`` as the page shows it: each exact line spelled as :func:`plain_command` spells it."""

    lines = []
    for line in str(text or "").splitlines():
        if line.startswith("#") or not line.strip():
            lines.append(line)
            continue
        try:
            argv = shlex.split(line, posix=os.name != "nt")
        except ValueError:
            lines.append(line)
            continue
        argv = [arg[1:-1] if len(arg) >= 2 and arg[0] == arg[-1] == '"' else arg for arg in argv]
        lines.append(plain_command(argv, root))
    return "\n".join(lines)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


#: A job id as the job manager names one (``job-YYYYmmdd-HHMMSS-hex``); a lock naming anything else is not looked up.
_JOB_ID = re.compile(r"job-[A-Za-z0-9-]{1,64}")


def holder_facts(jobs_root: Path, held: dict[str, Any] | None) -> dict[str, Any] | None:
    """What holds the card, from the card lock's record and the holding job's receipt; None when the card is free.

    ``job_id``; ``started_utc``, when the job took the card; ``kind``; and
    ``folder``, the folder the job writes (a forecast's run folder).  A lock
    file that cannot be read holds the card too: ``{"unreadable": True,
    "lock": path}``.
    """

    if not held:
        return None
    if held.get("unreadable"):
        return {"unreadable": True, "lock": held.get("path")}
    job_id = str(held.get("job_id") or "")
    receipt = read_json(Path(jobs_root) / job_id / "receipt.json", default=None) if _JOB_ID.fullmatch(job_id) else None
    receipt = receipt if isinstance(receipt, dict) else {}
    outputs = receipt.get("outputs") if isinstance(receipt.get("outputs"), dict) else {}
    return {"job_id": job_id or None, "started_utc": held.get("created_utc") or receipt.get("created_utc"),
            "kind": receipt.get("kind"), "folder": outputs.get("outdir") or receipt.get("cwd") or None}


def started_text(stamp: Any) -> str | None:
    """A recorded ISO time as the page writes a moment ("2026-09-27 09:30 UTC"), or None."""

    try:
        moment = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return utc_text(moment.astimezone(timezone.utc))


def holder_name(holder: dict[str, Any]) -> str:
    """The holding job's folder name, or its job id when it writes no folder."""

    folder = holder.get("folder")
    return Path(str(folder)).name if folder else f"job {holder.get('job_id') or 'with no name'}"


def card_taken_words(holder: dict[str, Any] | None) -> str:
    """Why a start from the page did not happen, naming what holds the card."""

    if holder is None:
        return "The card is running another job, so this forecast was not started."
    if holder.get("unreadable"):
        return "The card counts as in use because its lock file cannot be read, so this forecast was not started."
    when = started_text(holder.get("started_utc"))
    return (f"The card is running {holder_name(holder)}{', started ' + when if when else ''}, "
            "so this forecast was not started.")


def runtime_install() -> str:
    """How to install what :meth:`Runner.runtime_gap` says is missing, as one clause."""

    from gpuwm.capabilities import GPU_RUNTIME

    return ("Install it with pip install " + " or ".join(f"'gpuwm[{extra}]'" for extra in GPU_RUNTIME.extras)
            + " (gpuwm doctor names the one this computer needs)")


def log_command(rundir: Path, line: str, note: str = "") -> None:
    """Append one replayable line to ``<run>/commands.log``."""

    rundir.mkdir(parents=True, exist_ok=True)
    with (rundir / COMMANDS_LOG).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"# {_utc()}{'  ' + note if note else ''}\n{line}\n")


class Runner:
    """Runs the real engine."""

    kind = "real"

    def __init__(self, jobs_root: Path | None = None) -> None:
        self.jobs_root = jobs_root

    def _manager(self):
        from gpuwm.mcp.jobs import JobManager

        return JobManager(self.jobs_root)

    @staticmethod
    def _complete(argv: list[str], cwd: Path | None, timeout: float) -> tuple[int, str, str]:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            done = subprocess.run(argv, cwd=None if cwd is None else str(cwd), capture_output=True,
                                  timeout=timeout, stdin=subprocess.DEVNULL, creationflags=flags,
                                  env={**os.environ, **engine_env()})
        except subprocess.TimeoutExpired as error:
            raise Refused(f"The command took longer than {int(timeout)} s and was stopped.") from error
        return done.returncode, done.stdout.decode("utf-8", "replace"), done.stderr.decode("utf-8", "replace")

    def check(self, argv: list[str], *, cwd: Path | None = None, timeout: float = REVIEW_TIMEOUT_S) -> str:
        """Run a command that answers in lines and files rather than one JSON document (``gpuwm downscale
        --dry-run`` writes its plan beside ``--out``); returns its stdout, or raises :class:`Refused` carrying
        the last lines it wrote to stderr."""

        code, stdout, stderr = self._complete(argv, cwd, timeout)
        if code != 0:
            lines = [line for line in stderr.splitlines() if line.strip() and "installed wheel at" not in line]
            raise Refused("\n".join(lines[-12:]) or f"The command exited {code}.")
        return stdout

    def missing_geography(self, root: str | Path | None = None) -> dict[str, Any] | None:
        """What this computer's geography tree lacks for a forecast, or None when it is usable.

        Every forecast started here builds its grids' terrain, land use and
        soil from the tree at ``root`` (by default the one ``gpuwm
        fetch-geog`` stages into), so a start without it fails at once.
        The answer is the one ``gpuwm doctor`` and ``gpuwm go`` give:
        ``{"root", "absent" (no tree at all), "gaps" (one short line each)}``.
        """

        from gpuwm.doctor import geography_gaps
        from gpuwm.geog_assets import default_geog_root

        tree = Path(root) if root else default_geog_root()
        gaps = geography_gaps(tree)
        if not gaps:
            return None
        return {"root": str(tree), "absent": not tree.is_dir(), "gaps": [gap.brief or gap.detail for gap in gaps]}

    def query(self, argv: list[str], *, cwd: Path | None = None,
              timeout: float = QUERY_TIMEOUT_S, log: list[str] | None = None) -> dict[str, Any]:
        """Run a query command; its stdout must be one JSON document.

        ``log``, when given, receives the command's stderr, the words
        the command printed for a person beside its JSON.
        """

        code, stdout, stderr = self._complete(argv, cwd, timeout)
        if log is not None:
            log.append(stderr)
        if code != 0:
            document = _json_object(stdout)
            if document is not None and document.get("error"):
                raise Refused(str(document["error"]), document)
            lines = [line for line in stderr.splitlines()
                     if line.strip() and "installed wheel at" not in line]
            raise Refused("\n".join(lines[-12:]) or f"The command exited {code}.")
        try:
            return json.loads(stdout)
        except ValueError as error:
            raise Refused("The command did not answer with JSON.") from error

    def answer(self, argv: list[str], *, cwd: Path | None = None, codes: tuple[int, ...] = (0,),
               timeout: float = QUERY_TIMEOUT_S) -> tuple[int, dict[str, Any]]:
        """Run a query whose exit code is part of its answer; returns ``(code, document)``.

        For a question such as ``--readiness``, which prints its document at
        each of its ``codes`` (0 ready, 75 not yet, 2 refused).  Any other
        exit is refused as :meth:`query` refuses it.
        """

        code, stdout, stderr = self._complete(argv, cwd, timeout)
        document = _json_object(stdout)
        if code in codes and document is not None:
            return code, document
        lines = [line for line in stderr.splitlines() if line.strip() and "installed wheel at" not in line]
        raise Refused("\n".join(lines[-12:]) or f"The command exited {code}.", document)

    def launch(self, rundir: Path, argv: list[str], owner_file: str | None = None) -> dict[str, Any]:
        """Launch one run detached; returns the job document.

        ``owner_file``: the card-sharing OWNER file whose line is handed to
        the run's wrapper, which removes it when the run ends.
        """

        job = self.launch_detached(argv, cwd=rundir, outdir=rundir, kind="gui:run-plan", owner_file=owner_file)
        try:
            write_json(rundir / JOB, job)
        except OSError as error:
            # The run is running: a job record that could not be written is said beside the start (``unkept``),
            # never raised as a failed start, which put a queued forecast back in line to be started again.
            return {**job, "unkept": error.strerror or str(error) or type(error).__name__}
        return job

    def launch_detached(self, argv: list[str], *, cwd: Path, outdir: Path, kind: str,
                        owner_file: str | None = None) -> dict[str, Any]:
        """Launch one card run detached and write nothing into ``outdir``; returns the job document.

        For a command that claims its output folder itself (``gpuwm downscale --out`` adopts only an empty one):
        the caller keeps the returned job document wherever that command's folder allows.  ``owner_file`` is
        handed to the wrapper as in :meth:`launch`.
        """

        from gpuwm.mcp.doors import ArwenRefusal
        from gpuwm.mcp.jobs import CardTaken

        try:
            manager = self._manager()
            launched = manager.launch(kind, argv, cwd=cwd, gpu=True, env_additions=engine_env(),
                                      outputs={"outdir": str(outdir)}, owner_file=owner_file)
        except CardTaken as taken:
            # The job manager's sentence names job tools only an MCP client has; the page says who holds the card.
            raise Refused(card_taken_words(holder_facts(manager.root, taken.holder))) from taken
        except ArwenRefusal as error:
            raise Refused(str(error)) from error
        receipt = read_json(Path(launched["jobs_dir"]) / "receipt.json", default={}) or {}
        return {"job_id": launched["job_id"], "jobs_dir": launched["jobs_dir"],
                "wrapper_pid": receipt.get("wrapper_pid"), "wrapper_process": receipt.get("wrapper_process"),
                "argv": argv, "launched_utc": _utc()}

    def launch_helper(self, rundir: Path, argv: list[str], kind: str) -> dict[str, Any]:
        """Launch a detached helper that holds no card (the machines follower)."""

        from gpuwm.mcp.doors import ArwenRefusal

        try:
            launched = self._manager().launch(kind, argv, cwd=rundir, gpu=False, env_additions=engine_env())
        except ArwenRefusal as error:
            raise Refused(str(error)) from error
        receipt = read_json(Path(launched["jobs_dir"]) / "receipt.json", default={}) or {}
        return {"job_id": launched["job_id"], "jobs_dir": launched["jobs_dir"],
                "wrapper_pid": receipt.get("wrapper_pid"), "wrapper_process": receipt.get("wrapper_process"),
                "argv": argv, "launched_utc": _utc()}

    def runtime_gap(self) -> str | None:
        """Why the engine this runner starts cannot run a forecast on this computer, or None.

        The engine is this server's own Python (:func:`engine_argv`), and ``gpuwm run-plan`` refuses at its first
        step when the GPU runtime is not installed there (:data:`gpuwm.capabilities.GPU_RUNTIME`).  Start and the
        queue ask first, so neither starts a forecast only for it to fail and be listed as failed.  The runtime is
        looked for on the import path itself: the page server refuses to import CuPy
        (:func:`.server.block_gpu_imports`), which an ordinary lookup reads as not installed.  Whether an
        installed runtime then works on the card is ``gpuwm doctor``'s question.
        """

        from importlib.machinery import PathFinder

        from gpuwm.capabilities import GPU_RUNTIME

        try:
            spec = PathFinder.find_spec(GPU_RUNTIME.module, sys.path)
        except (ImportError, ValueError):
            spec = None
        # A folder of that name with no package in it (a namespace) is not the runtime.
        if spec is not None and spec.origin is not None:
            return None
        return "gpuwm's GPU runtime (CuPy) is not installed on this computer, so no forecast can run here."

    def card_holder(self) -> dict[str, Any] | None:
        """What holds this computer's card (:func:`holder_facts`), or None when it is free.

        Built from the card lock's record, never from its launch refusal:
        nothing is being launched when a page asks, and that sentence names
        job tools only an MCP client has.
        """

        try:
            manager = self._manager()
            return holder_facts(manager.root, manager.gpu_lock.held_by())
        except Exception:  # noqa: BLE001 - a jobs folder that cannot be read at all names no holder
            return None

    @staticmethod
    def _targets(rundir: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """The run's job record and the identities of its processes that are still those processes.

        The wrapper comes first (its group holds the engine), then the
        engine the manifest names.  A recorded PID whose process has
        another identity, or no identity recorded, is left out.
        """

        # A page-started downscale keeps its job record beside its folder, not in it (runs.job_of).
        job = job_of(rundir)
        manifest = read_json(rundir / MANIFEST, default=None)
        targets = []
        if job is not None and job_alive(rundir):
            targets.append(job["wrapper_process"])
        if manifest_alive(manifest):
            targets.append(manifest["process"])
        return job, targets

    def stop(self, rundir: Path, *, dry: bool = False) -> dict[str, Any]:
        """Stop the run; returns {method, command, message}."""

        job, targets = self._targets(rundir)
        if not targets:
            raise Refused("No process of this run is alive, so there is nothing to stop.")
        if os.name == "nt":
            target = targets[0]
            plan = {"method": "end-process-tree",
                    "command": f"taskkill /F /T /PID {target['pid']}",
                    "message": "Ended the run's process tree. The run folder keeps everything it wrote."}
            if not dry:
                cancelled = False
                if job is not None and job.get("job_id") and target == job.get("wrapper_process"):
                    try:
                        self._manager().cancel(str(job["job_id"]))
                        cancelled = True
                    except Exception:  # noqa: BLE001 - fall back to the verified pid
                        cancelled = False
                if not cancelled:
                    # The checked process's handle is held across taskkill, so its number cannot be handed on.
                    try:
                        proc_identity.signal_process(target, 15, tree=True)
                    except OSError as error:
                        raise Refused(f"The run's processes could not be ended: {error}") from error
            return plan
        import signal

        group = leader = None
        for target in targets:
            try:
                group = os.getpgid(int(target["pid"]))
            except (ProcessLookupError, PermissionError):
                continue
            if not proc_identity.alive(target):  # ended, or handed on, between the check and now
                group = None
                continue
            # The group's leader (the wrapper, which started the session) as it is now, so the escalation can
            # tell the same group from a new one that a reused PID started.
            leader = proc_identity.identify(group) if proc_identity.running(group) else None
            break
        if group is None:
            raise Refused("No process of this run is alive, so there is nothing to stop.")
        plan = {"method": "interrupt",
                "command": f"kill -INT -{group}",
                "message": "Sent Ctrl+C to the run. It stops at its next step and records that it was stopped."}
        if not dry:
            if leader is not None:
                # Through the leader's own process handle, so a group that ended in the meantime is never a new
                # one that took its number.
                if not proc_identity.signal_process(leader, signal.SIGINT, tree=True):
                    raise Refused("No process of this run is alive, so there is nothing to stop.")
            else:
                # The leader has gone; while any member lives no process can be given the group's number.
                os.killpg(group, signal.SIGINT)
            job_id = (job or {}).get("job_id")
            timer = threading.Timer(ESCALATE_AFTER_S, self._escalate, args=(group, job_id, leader))
            timer.daemon = True
            timer.start()
        return plan

    def _escalate(self, group: int, job_id: str | None, leader: dict[str, Any] | None = None) -> None:
        """End the group the Ctrl+C went to, if it is still that group.

        ``leader`` is the group leader's identity when Stop signalled it
        (None when the leader had already gone).  A leader PID that now
        answers as another process means the run ended and its number
        was handed on, so nothing is signalled: a SIGTERM there hit an
        unrelated program.  While a group lives no process can be given
        its id, so a group whose leader is gone and whose id nobody holds
        is still the run's.
        """

        import signal

        if leader is not None and proc_identity.alive(leader):
            try:
                proc_identity.signal_process(leader, signal.SIGTERM, tree=True)
            except OSError:
                pass
            return
        if proc_identity.running(group):
            return  # the group's number now names a process the Ctrl+C never went to
        try:
            os.killpg(group, 0)
        except (ProcessLookupError, PermissionError):
            return
        try:
            os.killpg(group, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


def record_stop(rundir: Path, plan: dict[str, Any]) -> None:
    write_json(rundir / STOP, {**plan, "at_utc": _utc()})
    log_command(rundir, plan["command"], "Stop")


__all__ = ["ESCALATE_AFTER_S", "Refused", "Runner", "display", "engine_argv",
           "log_command", "record_stop"]
