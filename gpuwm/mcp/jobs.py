"""The async job pattern: launch returns a job id, five tools follow it.

Anything long -- fetch, prep, forecast, render -- runs
as a DETACHED subprocess of the real CLI, so no MCP tool call is ever
held open by a forecast and a server restart loses nothing:

* every job owns a directory under the jobs root, holding a launch
  ``receipt.json`` (argv, cwd, env additions, pids, log paths, declared
  outputs), the child's ``stdout.log``/``stderr.log``, and the
  wrapper's ``started.json``/``result.json``;
* liveness is derived from the receipt's recorded wrapper process (its
  pid and creation time, :mod:`gpuwm.proc_identity`) plus the result
  document, never from server memory, so ``job_status`` answers the
  same after a restart and a pid reused by another program is not the
  job;
* ``job_events`` tails a chosen stream incrementally with a byte
  cursor -- the run's own JSONL (``events.jsonl``/``progress.jsonl``)
  where the door writes one, stdout/stderr always;
* a GPU-touching job takes the one-per-card lock
  (:mod:`gpuwm.mcp.gpulock`) at launch and a second GPU launch is
  refused by sentence, naming the running job id.

Jobs root: ``$GPUWM_MCP_JOBS_DIR``, else ``~/.gpuwm/mcp-jobs``.
Nothing here deletes anything: cancel stops processes and writes a
result document; logs and receipts stay for the reader.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from gpuwm import proc_identity
from gpuwm.mcp.doors import ArwenRefusal, refusal_text
from gpuwm.mcp.gpulock import CardHeld, GpuLock

#: Job event streams a cursor can follow, mapped to how the file is
#: found: fixed job-dir logs, or a name searched for under the job's
#: declared outdir (the doors write events.jsonl / progress.jsonl into
#: run folders whose timestamped names the launcher cannot predict).
STREAMS = ("stdout", "stderr", "events", "progress")

#: Cap on bytes one job_events call returns, so a chatty step log is
#: paged rather than dumped into the model's context in one turn.
MAX_EVENT_BYTES = 64 * 1024


class CardTaken(ArwenRefusal):
    """A GPU launch refused because another job holds the card.

    ``str()`` is the refusal sentence an MCP client reads, with the job
    tools it can use; ``holder`` is the card lock's record the sentence was
    built from (:meth:`GpuLock.held_by`), so a caller whose reader has no
    such tools (the page) says who holds the card in its own words.
    """

    def __init__(self, sentence: str, holder: dict | None) -> None:
        super().__init__(sentence)
        self.holder = holder


def jobs_root() -> Path:
    root = os.environ.get("GPUWM_MCP_JOBS_DIR")
    if root:
        return Path(root)
    return Path.home() / ".gpuwm" / "mcp-jobs"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _publish(path: Path, document: dict) -> None:
    # The temporary file is this writer's own.  A cancel publishes
    # result.json while the wrapper it just signalled publishes its own;
    # with one shared "result.tmp" the first replace took the other's
    # file and the second raised FileNotFoundError, so Stop failed.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(document, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class JobManager:
    """Launch, follow, and stop detached CLI jobs under one root."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else jobs_root()
        self.gpu_lock = GpuLock(self.root)

    # -- launch ----------------------------------------------------------

    def launch(self, kind: str, argv: list[str], *, cwd: str | Path,
               gpu: bool, env_additions: dict[str, str] | None = None,
               outputs: dict[str, str] | None = None,
               owner_file: str | None = None) -> dict:
        """Detach one job; returns the launch document with its job_id.

        ``gpu=True`` takes the card lock first and refuses (verbatim
        sentence, running job named) when it is held -- never a silent
        queue: a launch tool that queues has not launched.  The refusal is
        a :class:`CardTaken` carrying the lock's record.

        ``owner_file`` is the card-sharing OWNER file whose line the
        caller hands to this job's wrapper pid; the wrapper removes that
        line when the job ends (see :mod:`gpuwm.mcp._jobwrap`).
        """

        if not gpu:
            return self._spawn(kind, argv, cwd=cwd, gpu=False, env_additions=env_additions,
                               outputs=outputs, owner_file=owner_file)
        # Check the card, spawn and record the holder as one step under the lock's mutex: checked apart,
        # two starts in the same instant both passed the check and both ran on the card.
        launched: dict = {}

        def start() -> tuple[str, int, dict | None]:
            launched.update(self._spawn(kind, argv, cwd=cwd, gpu=True, env_additions=env_additions,
                                        outputs=outputs, owner_file=owner_file))
            return launched["job_id"], launched["wrapper_pid"], launched["wrapper_process"]

        try:
            self.gpu_lock.claim(start, before=self._reap_gpu_lock_locked)
        except CardHeld as held:
            raise CardTaken(str(held), held.holder) from None
        return launched

    def _spawn(self, kind: str, argv: list[str], *, cwd: str | Path, gpu: bool,
               env_additions: dict[str, str] | None, outputs: dict[str, str] | None,
               owner_file: str | None = None) -> dict:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        job_id = f"job-{stamp}-{secrets.token_hex(3)}"
        jobdir = self.root / job_id
        jobdir.mkdir(parents=True, exist_ok=False)

        receipt = {
            "schema": "gpuwm-mcp-job/1",
            "job_id": job_id,
            "kind": kind,
            "argv": [str(a) for a in argv],
            "cwd": str(cwd),
            "env_additions": dict(env_additions or {}),
            "gpu": bool(gpu),
            "created_utc": _utc_now(),
            "stdout_log": str(jobdir / "stdout.log"),
            "stderr_log": str(jobdir / "stderr.log"),
            "outputs": dict(outputs or {}),
        }
        if owner_file:
            receipt["owner_file"] = str(owner_file)
        _publish(jobdir / "receipt.json", receipt)

        # -P: the wrapper starts in the job's folder, and a gpuwm checkout
        # there must not be the gpuwm that runs it; a source or editable
        # install is named on PYTHONPATH instead.
        from gpuwm.runtime_manifest import child_python_env

        wrapper_argv = [sys.executable, "-P", "-m", "gpuwm.mcp._jobwrap",
                        str(jobdir)]
        popen_kwargs: dict = {
            "stdin": subprocess.DEVNULL,
            "stdout": open(jobdir / "wrapper.log", "ab"),
            "stderr": subprocess.STDOUT,
            "cwd": str(cwd),
            "env": {**os.environ, **child_python_env()},
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:  # pragma: no cover - windows dev box
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(wrapper_argv, **popen_kwargs)
        popen_kwargs["stdout"].close()
        # The wrapper's identity is read now, before the reaper below can collect it, so the receipt names
        # this process and no later holder of its PID.
        identity = proc_identity.identify(proc.pid)
        if os.name != "nt":  # pragma: no cover - windows dev box
            # Reap the wrapper when it exits.  Unreaped, it stays a zombie
            # of this long-lived server, kill(pid, 0) keeps answering for
            # it, and a job whose wrapper was killed read "running" for as
            # long as the server lived (seen through `gpuwm gui` after a
            # stop).
            threading.Thread(target=proc.wait, name=f"reap-{job_id}", daemon=True).start()

        receipt["wrapper_pid"] = proc.pid
        receipt["wrapper_process"] = identity
        _publish(jobdir / "receipt.json", receipt)
        return {"job_id": job_id, "kind": kind, "gpu": bool(gpu),
                "argv": receipt["argv"], "jobs_dir": str(jobdir),
                "outputs": receipt["outputs"], "wrapper_pid": proc.pid,
                "wrapper_process": identity,
                "follow_with": ["job_status", "job_events", "job_result"]}

    # -- state -----------------------------------------------------------

    def _jobdir(self, job_id: str) -> Path:
        jobdir = self.root / job_id
        if not (jobdir / "receipt.json").is_file():
            known = sorted(p.name for p in self.root.glob("job-*")
                           if (p / "receipt.json").is_file())
            listing = ", ".join(known[-8:]) if known else "none yet"
            raise ArwenRefusal(
                f"no job named {job_id} exists under {self.root} (its "
                "receipt.json is absent), so there is nothing to report "
                f"on; recent jobs here: {listing}.")
        return jobdir

    def status(self, job_id: str) -> dict:
        jobdir = self._jobdir(job_id)
        receipt = _read_json(jobdir / "receipt.json") or {}
        started = _read_json(jobdir / "started.json")
        result = _read_json(jobdir / "result.json")
        wrapper_pid = int(receipt.get("wrapper_pid", 0))

        if result is not None:
            state = "cancelled" if result.get("cancelled") else "exited"
        elif proc_identity.alive(receipt.get("wrapper_process"), wrapper_pid):
            # The wrapper this receipt launched, not whatever holds its PID now: a receipt written without an
            # identity, or a PID reused after a crash or a reboot, reads as lost.
            state = "running"
        else:
            # No result document and nobody alive to ever write one:
            # the wrapper was killed outright or the machine went down.
            state = "lost"
        if state in ("exited", "cancelled") and receipt.get("gpu"):
            # The mutex is taken only when this job is the holder: a job list read every ended GPU job's status,
            # and each one locked the card's mutex to find it held by nobody.
            held = self.gpu_lock.holder()
            if isinstance(held, dict) and held.get("job_id") == job_id:
                self.gpu_lock.release(job_id)

        document = {
            "job_id": job_id,
            "state": state,
            "kind": receipt.get("kind"),
            "gpu": receipt.get("gpu", False),
            "created_utc": receipt.get("created_utc"),
            "argv": receipt.get("argv"),
            "outputs": receipt.get("outputs", {}),
            "child_pid": (started or {}).get("child_pid"),
            "exit_code": (result or {}).get("exit_code"),
            "ended_utc": (result or {}).get("ended_utc"),
        }
        if state == "lost":
            document["note"] = (
                "the wrapper process is gone and wrote no result.json "
                "(killed outright, or the machine restarted); the logs "
                "in the job directory are the surviving record.")
        return document

    def _reap_gpu_lock_locked(self) -> None:
        """Release the GPU lock for a holder job that has exited (the caller holds the lock's mutex).

        The wrapper cannot release it (the lock is the server's), so
        release happens lazily at the next launch or status call.
        """

        held = self.gpu_lock.holder()
        if held is None:
            return
        job_id = held.get("job_id", "")
        result = _read_json(self.root / job_id / "result.json")
        if result is not None:
            self.gpu_lock.release_locked(job_id)

    def card_refusal(self) -> str | None:
        """The sentence a launch that needs the card would be refused with now, or None when it would start.

        The same question :meth:`launch` asks, finished jobs reaped first,
        so a caller that checks before writing anything cannot disagree
        with the launch that follows.
        """

        self._reap_gpu_lock()
        return self.gpu_lock.refusal()

    # -- events ----------------------------------------------------------

    def _stream_path(self, jobdir: Path, receipt: dict,
                     stream: str) -> Path | None:
        if stream == "stdout":
            return jobdir / "stdout.log"
        if stream == "stderr":
            return jobdir / "stderr.log"
        name = "events.jsonl" if stream == "events" else "progress.jsonl"
        outdir = receipt.get("outputs", {}).get("outdir")
        roots = [Path(p) for p in (outdir,) if p]
        for root in roots:
            if not root.is_dir():
                continue
            direct = root / name
            if direct.is_file():
                return direct
            found = sorted(root.rglob(name),
                           key=lambda p: p.stat().st_mtime)
            if found:
                return found[-1]
        return None

    def events(self, job_id: str, *, stream: str = "stdout",
               cursor: int = 0, max_bytes: int = MAX_EVENT_BYTES) -> dict:
        """Incremental tail of one stream from a byte cursor.

        Pass the returned ``next_cursor`` back in to read only what is
        new.  ``events``/``progress`` follow the run's own JSONL where
        the door writes one (searched under the job's declared outdir);
        until the run creates it the stream reports absent, not empty.
        """

        if stream not in STREAMS:
            raise ArwenRefusal(
                f"stream {stream!r} is not one of {STREAMS}, so there is "
                "no file to tail.")
        jobdir = self._jobdir(job_id)
        receipt = _read_json(jobdir / "receipt.json") or {}
        path = self._stream_path(jobdir, receipt, stream)
        state = self.status(job_id)["state"]
        if path is None or not path.is_file():
            return {"job_id": job_id, "stream": stream, "state": state,
                    "present": False, "lines": [], "next_cursor": cursor,
                    "eof": state != "running"}
        size = path.stat().st_size
        cursor = max(0, min(int(cursor), size))
        span = max(0, min(int(max_bytes), MAX_EVENT_BYTES))
        with open(path, "rb") as fh:
            fh.seek(cursor)
            chunk = fh.read(span)
        next_cursor = cursor + len(chunk)
        text = chunk.decode("utf-8", errors="replace")
        return {
            "job_id": job_id, "stream": stream, "state": state,
            "present": True, "path": str(path),
            "lines": text.splitlines(),
            "next_cursor": next_cursor,
            "remaining_bytes": size - next_cursor,
            "eof": next_cursor >= size and state != "running",
        }

    # -- result / cancel / list -----------------------------------------

    def result(self, job_id: str) -> dict:
        status = self.status(job_id)
        if status["state"] == "running":
            raise ArwenRefusal(
                f"job {job_id} is still running (child pid "
                f"{status.get('child_pid')}), so it has no result yet; "
                "poll job_status or tail job_events instead of asking "
                "for a result that does not exist.")
        jobdir = self._jobdir(job_id)
        exit_code = status.get("exit_code")
        stderr_text = ""
        try:
            stderr_text = (jobdir / "stderr.log").read_text(
                encoding="utf-8", errors="replace")
        except OSError:
            pass
        stdout_tail = ""
        try:
            raw = (jobdir / "stdout.log").read_bytes()
            stdout_tail = raw[-4096:].decode("utf-8", errors="replace")
        except OSError:
            pass
        document = dict(status)
        document["ok"] = exit_code == 0
        document["stdout_tail"] = stdout_tail
        if exit_code == 2:
            document["refusal"] = refusal_text(stderr_text)
        elif exit_code not in (0, None):
            document["stderr_tail"] = "\n".join(
                stderr_text.splitlines()[-20:])
        return document

    def cancel(self, job_id: str) -> dict:
        status = self.status(job_id)
        if status["state"] != "running":
            raise ArwenRefusal(
                f"job {job_id} is not running (state: {status['state']}), "
                "so there is nothing to cancel; its receipts and logs are "
                "untouched.")
        jobdir = self._jobdir(job_id)
        receipt = _read_json(jobdir / "receipt.json") or {}
        # The wrapper this receipt launched and its tree (Windows) or process
        # group (Linux), through its identity: never whatever holds its PID
        # now.  Only a signal that went through publishes the cancelled
        # result and frees the card.  A kill the system refused left the
        # run going on the card, and a cancelled result with the claim gone
        # let the next GPU job start beside it.
        import signal

        try:
            signalled = proc_identity.signal_process(
                receipt.get("wrapper_process"), signal.SIGTERM, tree=True)
        except OSError as error:
            raise ArwenRefusal(
                f"job {job_id} could not be stopped: {error}. It may still "
                "be running, so it keeps its card claim and no result was "
                "written; try the cancel again.") from error
        if not signalled:
            raise ArwenRefusal(
                f"job {job_id} ended before it could be stopped, so nothing "
                "was signalled; job_status says how it ended.")
        _publish(jobdir / "result.json", {
            "exit_code": None,
            "cancelled": True,
            "ended_utc": _utc_now(),
        })
        if receipt.get("gpu"):
            self.gpu_lock.release(job_id)
        return {"job_id": job_id, "state": "cancelled",
                "note": "the process tree was terminated; logs and "
                        "receipts remain in the job directory."}

    def list(self) -> dict:
        jobs = []
        if self.root.is_dir():
            for jobdir in sorted(self.root.glob("job-*")):
                if not (jobdir / "receipt.json").is_file():
                    continue
                jobs.append(self.status(jobdir.name))
        return {"jobs_root": str(self.root), "count": len(jobs),
                "jobs": jobs}
