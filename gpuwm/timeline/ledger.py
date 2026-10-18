"""Best-effort, append-only C1 emission using one write per JSON line.

GPUWM_TIME_LEDGER wins over a run-local default. IDs come from
GPUWM_RENTAL_ID and GPUWM_RUN_ID; GPUWM_PRODUCT defaults to regional.
Local regular files are the contract. POSIX flock also protects a partial
write from another cooperating writer. Windows opens the file for
append-only access, so the kernel places each record at the end of the
file. There is no per-event fsync.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from datetime import datetime, timezone

if __package__:
    from .taxonomy import VERSION, validate
    from .parallelism import tree_cpu_seconds
else:  # tools/timeline/ledger.sh works before gpuwm is installed
    import importlib.util as _importlib_util
    _spec = _importlib_util.spec_from_file_location(
        "gpuwm_timeline_taxonomy", Path(__file__).with_name("taxonomy.py"))
    _taxonomy = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_taxonomy)
    VERSION, validate = _taxonomy.VERSION, _taxonomy.validate
    _spec = _importlib_util.spec_from_file_location(
        "gpuwm_timeline_parallelism", Path(__file__).with_name("parallelism.py"))
    _parallelism = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_parallelism)
    tree_cpu_seconds = _parallelism.tree_cpu_seconds


def _cpu_attrs(attrs, phase, t_mono, cpu_pid=None):
    """Live begin/end events carry the process tree's CPU seconds (attrs.cpu_s), so every stage's cores are
    CPU / wall (gpuwm/timeline/parallelism.py; 2026-10-16: no single-CPU stage on the critical path).
    Relayed boundaries (an explicit t_mono) were measured by someone else and carry none."""
    if phase not in ("begin", "end") or t_mono is not None or (attrs and "cpu_s" in attrs):
        return attrs
    try:
        cpu = tree_cpu_seconds(cpu_pid)
    except Exception:
        return attrs
    return attrs if cpu is None else {**(attrs or {}), "cpu_s": cpu}

_warned = set()
_warning_lock = threading.Lock()
try:
    _host = socket.gethostname() or "unknown"
except BaseException:
    _host = "unknown"


def _warn(path, error):
    """One diagnostic per process and destination, even with -Werror."""
    try:
        key = (os.getpid(), str(path))
        with _warning_lock:
            if key in _warned:
                return
            _warned.add(key)
        print(f"time ledger: cannot append {path}: {type(error).__name__}: {error}",
              file=sys.stderr)
    except BaseException:
        pass


_windows_api = None


def _windows_append_api():
    global _windows_api
    if _windows_api is None:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel32.CreateFileW
        create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                           wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                           wintypes.HANDLE)
        create.restype = wintypes.HANDLE
        close = kernel32.CloseHandle
        close.argtypes = (wintypes.HANDLE,)
        close.restype = wintypes.BOOL
        _windows_api = ctypes, create, close
    return _windows_api


def _open_append(path):
    """A descriptor whose every write lands whole at the end of the file.

    POSIX O_APPEND moves to the end inside the write itself. The Windows
    CRT only emulates O_APPEND: it seeks to the end and then writes, so two
    writers that seek together overwrite each other. Concurrent observation
    stages then lost begin/end events and left blank and spliced lines in
    timeline.jsonl, which the timeline report refuses as malformed timing
    JSON. A handle holding FILE_APPEND_DATA without FILE_WRITE_DATA has the
    kernel place each write at the end of the file, the way Rust and Go
    open a file for append there. The descriptor is binary, so each record
    keeps its LF terminator instead of the CRT's text-mode CRLF.
    """
    if os.name != "nt":
        return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    import msvcrt
    ctypes, create, close = _windows_append_api()
    FILE_APPEND_DATA, SYNCHRONIZE = 0x0004, 0x00100000
    SHARE_READ_WRITE_DELETE, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL = 0x7, 4, 0x80
    handle = create(str(path), FILE_APPEND_DATA | SYNCHRONIZE, SHARE_READ_WRITE_DELETE,
                    None, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, None)
    if handle is None or handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(handle, 0)
    except BaseException:
        close(handle)
        raise


class Ledger:
    def __init__(self, path=None, *, run_id=None, product=None, rental_id=None, card=None):
        self.path = os.environ.get("GPUWM_TIME_LEDGER") or path
        self.run_id = run_id if run_id is not None else os.environ.get("GPUWM_RUN_ID")
        self.rental_id = rental_id if rental_id is not None else os.environ.get("GPUWM_RENTAL_ID")
        self.product = product or os.environ.get("GPUWM_PRODUCT", "regional")
        self.card = card

    def emit(self, stage, phase="mark", *, attrs=None, t_mono=None, t_unix=None, cpu_pid=None):
        """Return success, never throw. Optional clocks relay measured boundaries."""
        if self.path is None:
            return False
        fd = None
        try:
            attrs = _cpu_attrs(attrs, phase, t_mono, cpu_pid)
            mono = time.monotonic() if t_mono is None else float(t_mono)
            wall = time.time() if t_unix is None else float(t_unix)
            record = validate({
                "v": VERSION,
                "t_utc": datetime.fromtimestamp(wall, timezone.utc).isoformat(
                    timespec="microseconds").replace("+00:00", "Z"),
                "t_mono": mono, "host": _host, "rental_id": self.rental_id,
                "run_id": self.run_id, "product": self.product,
                "stage": stage, "phase": phase, "card": self.card,
                "pid": os.getpid(), "attrs": {} if attrs is None else attrs,
            })
            line = (json.dumps(record, separators=(",", ":"), ensure_ascii=True,
                               allow_nan=False) + "\n").encode("utf-8")
            if len(line) > 65536:
                raise ValueError("event exceeds 64 KiB; put bulk evidence in a receipt")
            path = Path(self.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = _open_append(path)
            if os.name == "posix":
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
            written = os.write(fd, line)
            if written != len(line):
                # Do not retry a partial record after another writer can append.
                raise OSError(f"short append: {written}/{len(line)} bytes")
            return True
        except BaseException as error:
            _warn(self.path, error)
            return False
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except BaseException:
                    pass


def emit(stage, phase="mark", *, path=None, attrs=None, **context):
    """Convenience entry point for boot, workers and engine orchestration."""
    try:
        return Ledger(path, **context).emit(stage, phase, attrs=attrs)
    except BaseException as error:
        _warn(os.environ.get("GPUWM_TIME_LEDGER") or path, error)
        return False


@contextlib.contextmanager
def span(stage, *, ledger=None, path=None, attrs=None):
    """Time orchestration without swallowing an exception from the work."""
    target = ledger if ledger is not None else Ledger(path)
    target.emit(stage, "begin", attrs=attrs)
    try:
        yield
    finally:
        target.emit(stage, "end", attrs=attrs)


def main(argv=None):
    # Invalid CLI inputs are also best effort for shell callers.
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("stage")
        parser.add_argument("phase", choices=("begin", "end", "mark"), nargs="?", default="mark")
        parser.add_argument("--attrs", default="{}")
        parser.add_argument("--path")
        parser.add_argument("--product")
        parser.add_argument("--run-id")
        parser.add_argument("--rental-id")
        parser.add_argument("--card", type=int)
        parser.add_argument("--cpu-pid", type=int,
                            help="whose process tree's CPU seconds a begin/end carries (ledger.sh: the script)")
        args = parser.parse_args(argv)
        Ledger(args.path, product=args.product, run_id=args.run_id,
               rental_id=args.rental_id, card=args.card).emit(
                   args.stage, args.phase, attrs=json.loads(args.attrs), cpu_pid=args.cpu_pid)
    except SystemExit as error:
        if error.code:
            _warn(os.environ.get("GPUWM_TIME_LEDGER"), error)
    except BaseException as error:
        _warn(os.environ.get("GPUWM_TIME_LEDGER"), error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
