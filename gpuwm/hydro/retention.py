"""Reclaim completed per-start hydro state; shared fabric and source caches stay put.

References are registered when settings are loaded, a prepared root is published,
or a checkpoint is written. A live process pins its settings until it exits;
durable references expire only when their owning file/directory disappears.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

KEEP_STARTS = 3
#: A start folder: its request digest, or a retry sibling the door builds beside
#: an incomplete attempt (door._build_location), managed the same way.
START_NAME = r"[0-9a-f]{16}(?:-retry-[0-9]{4})?"


@contextmanager
def cache_lock(root):
    """Serialize reference publication and reclamation across launchers."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".retention.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _start(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return None
    if len(parts) < 2 or not re.fullmatch(START_NAME, parts[1]):
        return None
    return root / parts[0] / parts[1]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def reference(settings_path, *, owner=None, root=None):
    """Pin settings for this process, or for an existing durable owner."""
    from .door import cache_root
    root = Path(root or cache_root()).resolve()
    start = _start(settings_path, root)
    if start is None or not start.is_dir():
        return
    owner = None if owner is None else Path(owner).resolve()
    if owner is not None and not owner.exists():
        return
    key = f"live:{os.getpid()}:{start}" if owner is None else f"{owner}:{start}"
    row = {"start": str(start), "owner": None if owner is None else str(owner),
           "pid": os.getpid() if owner is None else None}
    with cache_lock(root):
        if not start.is_dir():
            return
        _write(root / ".references" / (hashlib.sha256(key.encode()).hexdigest() + ".json"), row)


def pin_current(owner):
    from .runtime import ENV_SETTINGS
    settings = os.environ.get(ENV_SETTINGS)
    if settings:
        reference(settings, owner=owner)


def _alive(pid):
    if os.name == "nt":
        # os.kill(pid, 0) uses TerminateProcess on Windows. Query only.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # Access denied remains pinned.
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True  # Unreadable is protected, never evidence of a dead run.


def _domain(request):
    from .door import _TIME_KEYS, _digest
    return _digest({k: v for k, v in request["footprint"].items() if k not in _TIME_KEYS})


def _allocated_bytes(path, info):
    if os.name != "nt":
        return info.st_blocks * 512
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel.GetCompressedFileSizeW
    query.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    query.restype = wintypes.DWORD
    high = wintypes.DWORD()
    ctypes.set_last_error(0)
    low = query(str(path), ctypes.byref(high))
    if low == 0xffffffff and ctypes.get_last_error():
        raise ctypes.WinError(ctypes.get_last_error())
    return (high.value << 32) | low


def reclaim(root=None, *, keep=KEEP_STARTS, protected=(), dry_run=False, include_legacy=False):
    """Keep newest starts per domain and every registered live/durable reference.

    Unknown/incomplete folders, symlinks and trees containing symlinks are never
    candidates. Only digest folders with a completed door request are removed.
    Allocated bytes account for hard links, including links shared with static-*.
    """
    from .door import cache_root
    if type(keep) is not int or keep < 1:
        raise ValueError("hydro retention must keep at least one start per domain")
    root = Path(root or cache_root()).resolve()
    report = {"schema": "gpuwm-hydro-retention.v1", "root": str(root), "keep": keep,
              "removed": [], "protected": [], "logical_bytes": 0, "reclaimed_bytes": 0,
              "seconds": 0.0, "parallelism": "metadata scan; filesystem unlink"}
    if not root.is_dir():
        return report
    started = time.perf_counter()
    with cache_lock(root):
        pinned = {str(p) for v in protected if (p := _start(v, root)) is not None}
        for path in (root / ".references").glob("*.json"):
            try:
                row = json.loads(path.read_text())
                live = (_alive(row["pid"]) if row.get("pid") is not None
                        else Path(row["owner"]).exists())
                if live:
                    pinned.add(str(_start(row["start"], root)))
                elif not dry_run:
                    path.unlink()
            except (OSError, ValueError, KeyError, TypeError):
                # A broken reference index cannot authorize deletion.
                report["blocked"] = f"unreadable hydro reference {path}"
                return report
        groups = {}
        for provider in root.iterdir():
            if (not provider.is_dir() or provider.is_symlink() or provider.name.startswith(".")
                    or (hasattr(provider, "is_junction") and provider.is_junction())):
                continue
            for start in provider.iterdir():
                if (not re.fullmatch(START_NAME, start.name) or start.is_symlink()
                        or (hasattr(start, "is_junction") and start.is_junction())
                        or not (start / "door-prepared.json").is_file()):
                    continue
                if not include_legacy and not (start / "retention-managed.json").is_file():
                    pinned.add(str(start))
                try:
                    request = json.loads((start / "door-request.json").read_text())
                    group = (provider.name, _domain(request))
                    stamp = (start / "hydro-door.toml").stat().st_mtime_ns
                except (OSError, ValueError, KeyError, TypeError):
                    continue
                groups.setdefault(group, []).append((stamp, start))
        victims = []
        for starts in groups.values():
            starts.sort(reverse=True)
            for _, path in starts[:keep]:
                pinned.add(str(path))
            victims.extend(path for _, path in starts[keep:] if str(path) not in pinned)
        report["protected"] = sorted(pinned)
        links = {}
        safe = []
        for path in victims:
            files = list(path.rglob("*"))
            if any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()) for p in files):
                continue
            safe.append(path)
            for file in files:
                if not file.is_file():
                    continue
                info = file.stat()
                report["logical_bytes"] += info.st_size
                key = (info.st_dev, info.st_ino)
                count, size, total = links.get(key, (0, _allocated_bytes(file, info), info.st_nlink))
                links[key] = count + 1, size, total
        report["reclaimed_bytes"] = sum(size for count, size, total in links.values() if count == total)
        for path in safe:
            if not dry_run:
                shutil.rmtree(path)
            report["removed"].append(str(path))
        report["seconds"] = time.perf_counter() - started
        if not dry_run:
            _write(root / "retention-last.json", report)
    return report
