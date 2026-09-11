"""Durable CLI worker for a detached terminal UI job.

The start marker is published only after the launcher owns the process group
or Windows JobObject. Importing and invoking the CLI happens after that seam.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


#: How every gpuwm refusal opens its stderr line: ``gpuwm <command>: ``
#: (the CLI's refusal print boundary) or ``gpuwm <command>: error: ``
#: (argparse).  Warnings open with ``warning:`` and never match.
_REFUSAL_LINE = re.compile(r"^gpuwm(?: [A-Za-z0-9_.-]+)*: ", re.MULTILINE)


class _StderrTail:
    """Tee for the worker's stderr that remembers its last stretch.

    A gpuwm refusal is a sentence printed to stderr at a nonzero exit,
    not an exception: the CLI's refusal boundary catches the error,
    prints ``gpuwm <command>: <sentence>`` and returns 2.  The job log
    keeps the line, but a front door reading ``result.json`` saw only
    ``exit_code: 2`` and had nothing to show the user but "failed".
    This keeps the tail of what the process said so the worker can copy
    the refusal into its own receipt.
    """

    def __init__(self, stream, limit: int = 16384) -> None:
        self._stream = stream
        self._limit = limit
        self._tail = ""

    def write(self, text) -> int:
        count = self._stream.write(text)
        self._tail = (self._tail + str(text))[-self._limit:]
        return count

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def refusal(self) -> str | None:
        """The last refusal sentence this process printed, or ``None``.

        Everything from the last ``gpuwm <command>: `` prefix to the end
        of the tail, so a layered refusal keeps its follow-on lines and
        an advisory ``warning:`` printed earlier is not mistaken for it.
        """

        matches = list(_REFUSAL_LINE.finditer(self._tail))
        if not matches:
            return None
        return self._tail[matches[-1].end():].strip() or None


def _write_result(directory: Path, record: dict) -> None:
    temporary = directory / f"result.{os.getpid()}.tmp"
    with temporary.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(record, output, allow_nan=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, directory / "result.json")


def _join_windows_job(name: str) -> None:
    """Cover Windows venv redirectors before the real interpreter runs CLI."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenJobObjectW.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
    kernel.OpenJobObjectW.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = (wintypes.HANDLE, wintypes.HANDLE,
                                      ctypes.POINTER(wintypes.BOOL))
    kernel.IsProcessInJob.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenJobObjectW(0x0001 | 0x0004, False, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        process = kernel.GetCurrentProcess()
        assigned = wintypes.BOOL()
        if not kernel.IsProcessInJob(process, handle, ctypes.byref(assigned)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not assigned.value and not kernel.AssignProcessToJobObject(handle, process):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--windows-job")
    parser.add_argument("cli_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    cli_args = args.cli_args
    if cli_args[:1] == ["--"]:
        cli_args = cli_args[1:]
    directory = args.job_dir.resolve(strict=True)
    record = {"schema": "gpuwm-tui-result-v1", "pid": os.getpid(),
              "cli_args": cli_args, "started_at": _now(), "exit_code": None}
    code = 1
    stderr = _StderrTail(sys.stderr)
    sys.stderr = stderr
    try:
        if args.windows_job:
            _join_windows_job(args.windows_job)
        with (directory / "process.json").open("x", encoding="utf-8") as process:
            json.dump({"schema": "gpuwm-tui-process-v1", "pid": os.getpid(),
                       "parent_pid": os.getppid(), "started_at": record["started_at"],
                       "cwd": str(Path.cwd()), "cli_args": cli_args,
                       "process_group": os.getpgrp() if os.name != "nt" else None,
                       "windows_job": args.windows_job}, process, indent=2)
            process.write("\n")
            process.flush()
            os.fsync(process.fileno())
        with (directory / "ready").open("x") as ready:
            ready.flush()
            os.fsync(ready.fileno())
        # Bound an abandoned pre-launch handshake. This does not set a limit
        # on CLI preparation or a forecast once the launcher releases it.
        deadline = time.monotonic() + 60.0
        marker = directory / "start"
        while not marker.is_file():
            if time.monotonic() >= deadline:
                raise RuntimeError("TUI launcher did not release the startup handshake")
            time.sleep(0.02)
        if not cli_args:
            raise ValueError("TUI job requires a gpuwm command")
        from gpuwm.cli import main as cli_main
        result = cli_main(cli_args)
        code = 0 if result is None else int(result)
    except SystemExit as error:
        if error.code is None:
            code = 0
        elif isinstance(error.code, int):
            code = error.code
        else:
            print(error.code, file=sys.stderr, flush=True)
            code = 1
    except KeyboardInterrupt:
        print("gpuwm: interrupted; partial output has no completion receipt.",
              file=sys.stderr, flush=True)
        code = 130
    except Exception as error:
        traceback.print_exc()
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        code = 1
    finally:
        if code not in (0, 130) and "error" not in record:
            # A refusal the CLI printed and returned from, recorded the way
            # an uncaught exception already is: the receipt names the
            # reason, and a reader of result.json shows that sentence
            # instead of "failed".
            refusal = stderr.refusal()
            if refusal is not None:
                record["error"] = {"type": "Refusal", "message": refusal}
        record.update(ended_at=_now(), exit_code=code,
                      status=("completed" if code == 0 else
                              "interrupted" if code == 130 else "failed"))
        _write_result(directory, record)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
