"""A short-lived Windows reader must not abort fetch JSON publication."""
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import threading
import time

import pytest

from gpuwm import fetch_routes
from gpuwm.filesystem_paths import io_path


@contextmanager
def reader_without_delete_share(path):
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                               wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    api.CreateFileW.restype = wintypes.HANDLE
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    handle = api.CreateFileW(str(io_path(path)), 0x80000000, 0x3, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    guard = threading.Lock()

    def release():
        nonlocal handle
        with guard:
            if handle is not None:
                assert api.CloseHandle(handle)
                handle = None
    try:
        yield release
    finally:
        release()


@pytest.mark.skipif(os.name != 'nt', reason='Actual Windows delete-sharing behavior')
@pytest.mark.parametrize('name', ['fetch-manifest.json', 'fetch-recovery-request.json'])
def test_fetch_json_survives_a_transient_actual_reader(tmp_path, monkeypatch, name):
    path = tmp_path / name
    old = b'{"old":true}\n'
    path.write_bytes(old)
    payload = {'status': 'complete', 'sha256': 'a' * 64, 'bytes': 1234567}
    expected = (json.dumps(payload, indent=2, sort_keys=False) + '\n').encode('utf-8')
    denied = threading.Event()
    errors = []
    native = os.replace

    def observe(source, destination):
        try:
            return native(source, destination)
        except PermissionError as error:
            errors.append(error.winerror)
            assert path.read_bytes() == old
            denied.set()
            raise

    with reader_without_delete_share(path) as release:
        def let_reader_finish():
            if denied.wait(2):
                time.sleep(.035)
                release()
        reader = threading.Thread(target=let_reader_finish)
        reader.start()
        try:
            monkeypatch.setattr(os, 'replace', observe)
            fetch_routes._write_json(path, payload)
        finally:
            denied.set()
            reader.join(timeout=3)
            assert not reader.is_alive()
    assert errors and all(code in (5, 32) for code in errors)
    assert path.read_bytes() == expected
    assert json.loads(path.read_text())['sha256'] == 'a' * 64
    assert not path.with_name(name + '.partial').exists()


@pytest.mark.skipif(os.name != 'nt', reason='Actual Windows persistent delete-sharing behavior')
def test_persistent_reader_reports_failure_and_preserves_both_json_revisions(tmp_path):
    path = tmp_path / 'fetch-manifest.json'
    old = b'{"old":true}\n'
    path.write_bytes(old)
    payload = {'status': 'complete', 'sha256': 'b' * 64}
    expected = (json.dumps(payload, indent=2, sort_keys=False) + '\n').encode('utf-8')
    started = time.monotonic()
    with reader_without_delete_share(path):
        with pytest.raises(PermissionError) as failed:
            fetch_routes._write_json(path, payload)
        assert failed.value.winerror in (5, 32)
        assert path.read_bytes() == old
    elapsed = time.monotonic() - started
    assert .45 <= elapsed < 3, elapsed
    assert path.with_name(path.name + '.partial').read_bytes() == expected
