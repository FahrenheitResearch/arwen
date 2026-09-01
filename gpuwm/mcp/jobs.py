"""The async job pattern: launch returns a job id, five tools follow it.

Anything long -- fetch, prep, forecast, render -- runs
as a DETACHED subprocess of the real CLI, so no MCP tool call is ever
held open by a forecast and a server restart loses nothing:

* every job owns a directory under the jobs root, holding a launch
  ``receipt.json`` (argv, cwd, env additions, pids, log paths, declared
  outputs), the child's ``stdout.log``/``stderr.log``, and the
  wrapper's ``started.json``/``result.json``;
* liveness is derived from the receipt's recorded pid plus the result
  document, never from server memory, so ``job_status`` answers the
  same after a restart;
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
from datetime import datetime, timezone
from pathlib import Path

from gpuwm.mcp.doors import ArwenRefusal, refusal_text
from gpuwm.mcp.gpulock import GpuLock, _pid_alive

#: Job event streams a cursor can follow, mapped to how the file is
#: found: fixed job-dir logs, or a name searched for under the job's
#: declared outdir (the doors write events.jsonl / progress.jsonl into
#: run folders whose timestamped names the launcher cannot predict).
STREAMS = ("stdout", "stderr", "events", "progress")

#: Cap on bytes one job_events call returns, so a chatty step log is
#: paged rather than dumped into the model's context in one turn.
MAX_EVENT_BYTES = 64 * 1024


def jobs_root() -> Path:
    root = os.environ.get("GPUWM_MCP_JOBS_DIR")
    if root:
        return Path(root)
    return Path.home() / ".gpuwm" / "mcp-jobs"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _publish(path: Path, document: dict) -> None:
    tmp = path.with_suffix(".tmp")
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
               outputs: dict[str, str] | None = None) -> dict:
        """Detach one job; returns the launch document with its job_id.

        ``gpu=True`` takes the card lock first and refuses (verbatim
        sentence, running job named) when it is held -- never a silent
        queue: a launch tool that queues has not launched.
        """

        self._reap_gpu_lock()
        if gpu:
            refusal = self.gpu_lock.refusal()
            if refusal is not None:
                raise ArwenRefusal(refusal)

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
        _publish(jobdir / "receipt.json", receipt)

        wrapper_argv = [sys.executable, "-m", "gpuwm.mcp._jobwrap",
                        str(jobdir)]
        popen_kwargs: dict = {
            "stdin": subprocess.DEVNULL,
            "stdout": open(jobdir / "wrapper.log", "ab"),
            "stderr": subprocess.STDOUT,
            "cwd": str(cwd),
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:  # pragma: no cover - windows dev box
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(wrapper_argv, **popen_kwargs)
        popen_kwargs["stdout"].close()

        receipt["wrapper_pid"] = proc.pid
        _publish(jobdir / "receipt.json", receipt)
        if gpu:
            self.gpu_lock.acquire(job_id, proc.pid)
        return {"job_id": job_id, "kind": kind, "gpu": bool(gpu),
                "argv": receipt["argv"], "jobs_dir": str(jobdir),
                "outputs": receipt["outputs"],
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
        elif _pid_alive(wrapper_pid):
            state = "running"
        else:
            # No result document and nobody alive to ever write one:
            # the wrapper was killed outright or the machine went down.
            state = "lost"
        if state in ("exited", "cancelled") and receipt.get("gpu"):
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

    def _reap_gpu_lock(self) -> None:
        """Release the GPU lock for a holder job that has exited.

        The wrapper cannot release it (the lock is the server's), so
        release happens lazily at the next launch or status call.
        """

        held = self.gpu_lock.holder()
        if held is None:
            return
        job_id = held.get("job_id", "")
        result = _read_json(self.root / job_id / "result.json")
        if result is not None:
            self.gpu_lock.release(job_id)

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
        wrapper_pid = int(receipt.get("wrapper_pid", 0))
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(wrapper_pid)],
                capture_output=True)
        else:  # pragma: no cover - windows dev box
            import signal
            try:
                os.killpg(os.getpgid(wrapper_pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
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
