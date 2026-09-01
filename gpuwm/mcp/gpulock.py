"""One GPU job at a time, arbitrated by a lockfile the server owns.

Two CUDA compute contexts on one card contend for VRAM and can OOM or
slow each other into uselessness -- ``gpuwm run`` itself documents
shared-GPU operation as UNSUPPORTED -- so the MCP server admits ONE
GPU-touching job at a time and REFUSES the second by sentence, naming
the running job id.  Refusing rather than queueing is deliberate v1
policy: a silent queue is a launch tool that lies about having
launched.

The lock is a JSON file in the jobs root.  It is advisory and scoped to
this server's own launches (a person at the terminal is outside it);
one server per machine is the v1 assumption.  Staleness is resolved by
liveness: a lock whose holder process is gone releases itself at the
next acquire, so a crashed job cannot wedge the card forever.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

LOCK_NAME = "gpu.lock"


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness for a pid this user owns, both platforms."""

    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # SOMETHING answers to the pid; treat a process we cannot signal
        # as alive rather than reclaiming a lock out from under it.
        return True
    except OSError:
        return False
    return True


class GpuLock:
    """The jobs root's ``gpu.lock``: acquire, holder, release."""

    def __init__(self, jobs_root: Path):
        self.path = Path(jobs_root) / LOCK_NAME

    def holder(self) -> dict | None:
        """The current lock document, or None (absent or unreadable)."""

        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def refusal(self) -> str | None:
        """The refusal sentence when the card is held, else None.

        A holder whose recorded pid is dead is stale -- the job crashed
        or the machine rebooted -- and is reclaimed here, so a dead job
        can never wedge the card.
        """

        held = self.holder()
        if held is None:
            if self.path.exists():
                # Unreadable lock: refuse rather than run beside an
                # unknown holder.
                return (f"the GPU lockfile {self.path} exists but cannot "
                        "be read, so this launch cannot prove the card is "
                        "free; inspect or remove the file and retry.")
            return None
        pid = int(held.get("wrapper_pid", 0))
        if not _pid_alive(pid):
            try:
                self.path.unlink()
            except OSError:
                pass
            return None
        job_id = held.get("job_id", "<unknown>")
        started = held.get("created_utc", "<unknown time>")
        return (f"the GPU is held by running job {job_id} (pid {pid}, "
                f"started {started}); a second CUDA run on the same card "
                "contends for VRAM and can OOM or corrupt the timing of "
                "both, so this launch is refused -- poll job_status "
                f"{job_id}, or job_cancel it, then relaunch.")

    def acquire(self, job_id: str, wrapper_pid: int) -> None:
        """Record this job as the holder (caller checked refusal first)."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "job_id": job_id,
            "wrapper_pid": wrapper_pid,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(document, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def release(self, job_id: str) -> None:
        """Drop the lock iff this job holds it (never someone else's)."""

        held = self.holder()
        if held is not None and held.get("job_id") == job_id:
            try:
                self.path.unlink()
            except OSError:
                pass
