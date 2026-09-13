"""Native Windows sharing denials at the local analysis status boundary."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import threading

import pytest

from gpuwm.local_da_runtime import _atomic


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows sharing contract")


@contextmanager
def _reader_without_delete_sharing(path):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(path), 0x80000000, 0x3, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    guard = threading.Lock()

    def release():
        nonlocal handle
        with guard:
            if handle is not None:
                if not kernel.CloseHandle(handle):
                    raise ctypes.WinError(ctypes.get_last_error())
                handle = None

    try:
        yield release
    finally:
        release()


def test_local_status_retries_a_real_reader_until_it_releases(tmp_path, monkeypatch):
    path = tmp_path / "execution.json"
    old = b'{"status":"ASSIMILATING","gridpoints_done":4}\n'
    path.write_bytes(old)
    updated = {"status": "ASSIMILATING", "gridpoints_done":8}
    denied = threading.Event()
    cancelled = threading.Event()
    errors = []
    native_replace = os.replace

    def observe_replace(source, destination):
        try:
            return native_replace(source, destination)
        except PermissionError as error:
            if Path(destination) == path:
                errors.append(error.winerror)
                assert path.read_bytes() == old
                denied.set()
            raise

    monkeypatch.setattr(os, "replace", observe_replace)
    with _reader_without_delete_sharing(path) as release:
        def release_during_backoff():
            if denied.wait(2.0) and not cancelled.wait(0.035):
                release()

        reader = threading.Thread(target=release_during_backoff)
        reader.start()
        try:
            _atomic(path, updated)
        finally:
            cancelled.set()
            denied.set()
            reader.join(timeout=2.0)
            assert not reader.is_alive()
    assert errors and all(code in (5, 32) for code in errors)
    assert json.loads(path.read_text(encoding="utf-8")) == updated
    assert list(tmp_path.iterdir()) == [path]


def test_local_status_permanent_denial_keeps_old_json_and_raises(tmp_path):
    path = tmp_path / "execution.json"
    old = b'{"status":"ASSIMILATING","gridpoints_done":4}\n'
    path.write_bytes(old)
    with _reader_without_delete_sharing(path):
        with pytest.raises(PermissionError) as caught:
            _atomic(path, {"status": "COMPLETE", "gridpoints_done":8})
        assert caught.value.winerror in (5, 32)
        assert path.read_bytes() == old
        assert json.loads(path.read_text(encoding="utf-8"))["status"] == "ASSIMILATING"
    assert list(tmp_path.iterdir()) == [path]
