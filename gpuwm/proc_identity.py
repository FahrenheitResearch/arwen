"""Which process a recorded PID names: the PID plus when that process was created.

A PID alone is not a process.  After a crash or a restart the number a
run folder, a job receipt or the card lock recorded can belong to an
unrelated program, and a liveness check by PID then reads that program
as the run: the page showed a crashed forecast as Running and its Stop
signalled the stranger's process group.  So every PID the page server
and the job manager keep is written beside its identity (the process's
creation time and, on Linux, the boot it belongs to), and a record is
only ever treated as a live process of ours when the process behind the
PID today has that same identity.  A record written without an identity
(before this existed) cannot prove which process it named and reads as
ended.

Deliberately stdlib-only and import-light: the detached job wrapper
imports it.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any


_KERNEL32 = None


def _kernel32():
    """A private kernel32 with its prototypes declared (the shared ``windll`` one is left as others set it)."""

    global _KERNEL32
    if _KERNEL32 is None:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        _KERNEL32 = kernel32
    return _KERNEL32


def _windows_times(pid: int) -> tuple[bool, str | None]:
    """(alive, creation time) of a Windows process, from one handle."""

    import ctypes
    from ctypes import wintypes

    kernel32 = _kernel32()
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False, None
    try:
        code = wintypes.DWORD()
        alive = bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                        ctypes.byref(kernel), ctypes.byref(user)):
            return alive, None
        return alive, str((created.dwHighDateTime << 32) | created.dwLowDateTime)
    finally:
        kernel32.CloseHandle(handle)


def _linux_stat(pid: int) -> tuple[bool, str | None] | None:
    """(alive, start time in clock ticks since boot) from /proc, or None where there is no /proc."""

    if not Path("/proc/self/stat").exists():
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return False, None
    # fields[0] is the state (field 3), fields[19] the start time (field 22).
    try:
        return fields[0] not in ("Z", "X"), fields[19]
    except IndexError:
        return False, None


def _ps_start(pid: int) -> tuple[bool, str | None]:
    try:
        done = subprocess.run(["ps", "-o", "stat=,lstart=", "-p", str(pid)], capture_output=True,
                              text=True, timeout=10, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return False, None
    text = done.stdout.strip()
    if done.returncode != 0 or not text:
        return False, None
    state, _, start = text.partition(" ")
    return not state.startswith("Z"), " ".join(start.split()) or None


def pid_number(value: Any) -> int | None:
    """A recorded PID as a process number, or None for anything a record should never have held.

    ``true``, ``1.5``, ``NaN`` or a number past 32 bits is not a PID: read
    as one it became ``1`` (the system's first process), or Windows cut it
    to 32 bits and it named some other process.
    """

    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if 0 < number <= 0xFFFFFFFF else None


def _probe(pid: int) -> tuple[bool, str | None]:
    if pid <= 0:
        return False, None
    if os.name == "nt":
        return _windows_times(pid)
    found = _linux_stat(pid)
    return found if found is not None else _ps_start(pid)


def boot_id() -> str | None:
    """This boot's id on Linux (a PID's start time counts from boot), else None."""

    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except OSError:
        return None


def identify(pid: Any) -> dict[str, Any] | None:
    """The identity record of the process ``pid`` names now, or None when there is none to name.

    Taken for a child right after it is spawned (and before it is reaped),
    so the record names that child and no later holder of the number.
    """

    pid = pid_number(pid)
    if pid is None:
        return None
    _, start = _probe(pid)
    if start is None:
        return None
    record: dict[str, Any] = {"pid": pid, "start": start}
    boot = boot_id()
    if boot:
        record["boot"] = boot
    return record


def alive(record: Any, pid: Any = None) -> bool:
    """True only when the process ``record`` names is running now: same PID, same creation, same boot.

    ``pid``, when given, must be the record's PID too (a record copied
    beside a different number names neither).  A missing record, a
    record without a creation time, an ended process and a PID that now
    belongs to another process all read False.
    """

    if not isinstance(record, dict) or not record.get("start"):
        return False
    number = pid_number(record.get("pid"))
    if number is None or (pid is not None and pid_number(pid) != number):
        return False
    live, start = _probe(number)
    if not live or start is None or str(start) != str(record["start"]):
        return False
    recorded_boot = record.get("boot")
    if recorded_boot:
        boot = boot_id()
        if boot and boot != recorded_boot:
            return False
    return True


def running(pid: Any) -> bool:
    """Some process holds ``pid`` now (which one is not asked; see :func:`alive` for that)."""

    number = pid_number(pid)
    return number is not None and _probe(number)[0]


def _pidfd(pid: int, record: Any) -> int | None:
    """A Linux process handle bound to the process ``record`` names, or None when that process is not there.

    The identity is checked, the handle opened, and the identity checked
    again through the handle's own view of its PID: a handle opened on a
    number whose process ended in between reports PID -1, so it can never
    stand for a process given the number since.
    """

    if not alive(record, pid):
        return None
    try:
        handle = os.pidfd_open(pid)
    except ProcessLookupError:
        return None
    try:
        info = Path(f"/proc/self/fdinfo/{handle}").read_text()
        held = next((int(line.split()[1]) for line in info.splitlines() if line.startswith("Pid:")), None)
        if held == pid and alive(record, pid):
            kept, handle = handle, None
            return kept
        return None
    finally:
        if handle is not None:
            os.close(handle)


def signal_process(record: Any, sig: int, *, tree: bool = False) -> bool:
    """Send ``sig`` to the process ``record`` names, and to nothing else; False when that process is gone.

    Checking a PID and then signalling it leaves a window in which the
    process can end and its number be handed to another program, which the
    signal then reaches.  Linux signals through a process handle (pidfd)
    opened on the checked process, which cannot stand for another one;
    Windows holds the checked process's handle open while it ends the tree,
    so its number cannot be handed on meanwhile.  ``tree`` also signals
    the process group the process leads (Linux; each member through its own
    handle), or ends its process tree (Windows).  Where neither exists the
    identity is checked just before a plain signal.  Call again with the
    same record for every later escalation.
    """

    if not isinstance(record, dict):
        return False
    pid = pid_number(record.get("pid"))
    if pid is None:
        return False
    if os.name == "nt":
        return _windows_end(pid, record, tree)
    import signal

    if not (hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal") and Path("/proc/self/fdinfo").is_dir()):
        if not alive(record, pid):
            return False
        try:
            if tree and os.getpgid(pid) == pid:
                os.killpg(pid, sig)
            else:
                os.kill(pid, sig)
        except ProcessLookupError:
            return False
        return True
    handle = _pidfd(pid, record)
    if handle is None:
        return False
    members: list[int] = []
    try:
        if tree and os.getpgid(pid) == pid:
            # Every other member of the group the process leads, each through a handle of its own, opened before
            # the leader is checked a last time.
            for entry in Path("/proc").iterdir():
                if not entry.name.isdecimal() or int(entry.name) == pid:
                    continue
                member = int(entry.name)
                try:
                    if os.getpgid(member) != pid:
                        continue
                    own = _pidfd(member, identify(member))
                    if own is not None:
                        if os.getpgid(member) == pid:
                            members.append(own)
                        else:
                            os.close(own)
                except (ProcessLookupError, PermissionError):
                    continue
            if not alive(record, pid):
                return False
        for member in members:
            try:
                signal.pidfd_send_signal(member, sig)
            except ProcessLookupError:
                pass
        signal.pidfd_send_signal(handle, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        for member in members:
            os.close(member)
        os.close(handle)


def _windows_end(pid: int, record: dict[str, Any], tree: bool) -> bool:
    """End the Windows process ``record`` names (and its tree), holding its handle so the PID cannot move."""

    import ctypes
    from ctypes import wintypes

    kernel32 = _kernel32()
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        if not alive(record, pid):
            return False
        command = ["taskkill", "/F", *(["/T"] if tree else []), "/PID", str(pid)]
        done = subprocess.run(command, capture_output=True, timeout=30,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        code = wintypes.DWORD()
        still = bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        if done.returncode and still:
            raise OSError(done.stderr.decode("utf-8", "replace").strip() or f"taskkill could not end process {pid}")
        return True
    finally:
        kernel32.CloseHandle(handle)


def reused(record: Any) -> bool:
    """True when the record's PID answers today but as a different process (so it must never be signalled)."""

    if not isinstance(record, dict):
        return False
    number = pid_number(record.get("pid"))
    if number is None:
        return False
    live, _ = _probe(number)
    return live and not alive(record)


__all__ = ["alive", "boot_id", "identify", "pid_number", "reused", "running", "signal_process"]
