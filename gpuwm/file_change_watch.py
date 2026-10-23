"""Kernel mutation notices and file identity for fresh stream receipts.

Neither reads a payload. Linux watches with inotify. Windows holds a read
oplock, and reads its identity through the same two calls rw_fetch's
Windows receipt makes, because ``os.stat`` cannot name a change there.
"""
import ctypes
import functools
import os
import struct
import sys
import threading


#: Modification, attributes, close-after-write, deletion and movement.
_INOTIFY_MASK = 0x2 | 0x4 | 0x8 | 0x400 | 0x800
_IN_Q_OVERFLOW = 0x4000
_EVENT = struct.Struct('iIII')


class _Inotify:
    """The one inotify instance of this process; every watch is a descriptor on it.

    Breakage it prevents: an instance per watched file.  Linux allows 128
    inotify instances per user (fs.inotify.max_user_instances), shared by
    every process of that user, while a watch descriptor costs one of about
    a million.  A fetch that published a few hundred stream files used them
    all: the watches past the limit failed, so those files' fresh receipts
    were re-read, and every other program of the same user (tail -F, an
    editor, a file watcher) got EMFILE until the facts were evicted.
    Measured on a shared host: 91 of 200 watches opened, then
    inotify_init1 failed with errno 24.
    """

    def __init__(self):
        self.pid = os.getpid()
        self.lock = threading.Lock()
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        self.watchers = {}

    def add(self, watch, path):
        """The watch descriptor for ``path`` with ``watch`` registered on it, or -1."""
        with self.lock:
            if self.fd < 0:
                return -1
            # Events already queued belong to the watches made before this one.
            self._drain()
            wd = self.libc.inotify_add_watch(self.fd, os.fsencode(path), _INOTIFY_MASK)
            if wd < 0:
                return -1
            self.watchers.setdefault(wd, set()).add(watch)
            return wd

    def changed(self, watch):
        with self.lock:
            self._drain()
            return watch.changed

    def remove(self, watch, wd):
        with self.lock:
            group = self.watchers.get(wd)
            if group is None:
                return
            group.discard(watch)
            if not group:
                del self.watchers[wd]
                self.libc.inotify_rm_watch(self.fd, wd)

    def _drain(self):
        while True:
            try:
                data = os.read(self.fd, 65536)
            except BlockingIOError:
                return
            except OSError:
                data = None
            if not data:
                # An unreadable queue cannot vouch for any file.
                for group in self.watchers.values():
                    for watch in group:
                        watch.changed = True
                return
            offset = 0
            while offset + _EVENT.size <= len(data):
                wd, mask, _cookie, length = _EVENT.unpack_from(data, offset)
                offset += _EVENT.size + length
                if wd < 0 or mask & _IN_Q_OVERFLOW:
                    groups = list(self.watchers.values())
                else:
                    groups = [self.watchers.get(wd, ())]
                for group in groups:
                    for watch in group:
                        watch.changed = True


_INOTIFY = None
_INOTIFY_LOCK = threading.Lock()


def _inotify():
    global _INOTIFY
    with _INOTIFY_LOCK:
        if _INOTIFY is None or _INOTIFY.pid != os.getpid():
            # A forked child shares its parent's queue; it opens its own.
            _INOTIFY = _Inotify()
        return _INOTIFY


class FileChangeWatch:
    def __init__(self, path):
        self.wd = -1
        self.changed = False
        self.pid = os.getpid()
        self._oplock = None
        self._inotify = None
        if os.name == 'nt':
            self._oplock = _ReadOplock(path)
            return
        if not sys.platform.startswith('linux'):
            return
        self._inotify = _inotify()
        self.wd = self._inotify.add(self, path)

    @property
    def holds_file(self):
        """True while this watch keeps the file itself open (Windows)."""
        return self._oplock is not None and self._oplock.holding()

    def unchanged(self):
        if self._oplock is not None:
            return self._oplock.intact()
        if self.pid != os.getpid():
            return False
        if self.wd < 0:
            return False
        if self._inotify.changed(self):
            self.close()
            return False
        return True

    def close(self):
        if self._oplock is not None:
            self._oplock.close()
        if self.wd >= 0:
            if self.pid == os.getpid():
                # In a forked child the descriptor is the parent's: never remove it.
                self._inotify.remove(self, self.wd)
            self.wd = -1

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Interpreter shutdown may already have torn ctypes down.


def stat_identity(path):
    """``(device, file id, bytes, mtime_ns, ctime_ns)`` of ``path``.

    ``ctime_ns`` must move on every content or metadata change. On POSIX
    that is ``st_ctime``. On Windows ``os.stat`` reports the CREATION time
    as ``st_ctime``, so a same-size rewrite whose LastWriteTime was put
    back matched the identity taken before it; and its ``st_dev`` is the
    32-bit volume serial up to Python 3.11 but the 64-bit one from 3.12,
    so it cannot be compared with another program's. Windows reads
    GetFileInformationByHandle's volume serial, file index and size and
    FILE_BASIC_INFO's LastWriteTime and ChangeTime from one
    attribute-only handle, as rw_fetch's Windows receipt does.
    """
    if os.name != 'nt':
        status = os.stat(path)
        return (status.st_dev, status.st_ino, status.st_size,
                status.st_mtime_ns, status.st_ctime_ns)
    win = _windows()
    name = os.fsdecode(os.fspath(path))
    handle = win.kernel32.CreateFileW(name, _FILE_READ_ATTRIBUTES, _SHARE_ALL,
                                      None, _OPEN_EXISTING,
                                      _FILE_FLAG_BACKUP_SEMANTICS, None)
    if handle == _INVALID_HANDLE_VALUE:
        error = ctypes.get_last_error()
        raise OSError(None, ctypes.FormatError(error).strip(), name, error)
    try:
        by_handle = win.ByHandle()
        basic = win.Basic()
        if (not win.kernel32.GetFileInformationByHandle(handle, ctypes.byref(by_handle))
                or not win.kernel32.GetFileInformationByHandleEx(
                    handle, _FILE_BASIC_INFO, ctypes.byref(basic), ctypes.sizeof(basic))):
            error = ctypes.get_last_error()
            raise OSError(None, ctypes.FormatError(error).strip(), name, error)
    finally:
        win.kernel32.CloseHandle(handle)
    return (by_handle.volume_serial,
            (by_handle.index_high << 32) | by_handle.index_low,
            (by_handle.size_high << 32) | by_handle.size_low,
            (basic.write - _UNIX_EPOCH_FILETIME) * 100,
            (basic.change - _UNIX_EPOCH_FILETIME) * 100)


_FILE_READ_ATTRIBUTES = 0x80
_SHARE_ALL = 0x1 | 0x2 | 0x4
_OPEN_EXISTING = 3
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_FLAG_OVERLAPPED = 0x40000000
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
_FILE_BASIC_INFO = 0
_UNIX_EPOCH_FILETIME = 116444736000000000
_FSCTL_REQUEST_OPLOCK = 0x00090240
_OPLOCK_LEVEL_CACHE_READ = 0x1
_REQUEST_OPLOCK_INPUT_FLAG_REQUEST = 0x1
_ERROR_IO_PENDING = 997
#: Oplock buffers whose request never reported back after its handle closed.
#: The kernel may still write them, so they are kept rather than freed.
_UNSETTLED = []


@functools.lru_cache(maxsize=1)
def _windows():
    from ctypes import wintypes

    class ByHandle(ctypes.Structure):
        _fields_ = [("attributes", wintypes.DWORD), ("creation", wintypes.FILETIME),
                    ("access", wintypes.FILETIME), ("write", wintypes.FILETIME),
                    ("volume_serial", wintypes.DWORD), ("size_high", wintypes.DWORD),
                    ("size_low", wintypes.DWORD), ("links", wintypes.DWORD),
                    ("index_high", wintypes.DWORD), ("index_low", wintypes.DWORD)]

    class Basic(ctypes.Structure):
        _fields_ = [("creation", ctypes.c_longlong), ("access", ctypes.c_longlong),
                    ("write", ctypes.c_longlong), ("change", ctypes.c_longlong),
                    ("attributes", wintypes.DWORD)]

    class Overlapped(ctypes.Structure):
        _fields_ = [("internal", ctypes.c_size_t), ("internal_high", ctypes.c_size_t),
                    ("offset", wintypes.DWORD), ("offset_high", wintypes.DWORD),
                    ("event", wintypes.HANDLE)]

    class OplockRequest(ctypes.Structure):
        _fields_ = [("version", wintypes.WORD), ("length", wintypes.WORD),
                    ("level", wintypes.DWORD), ("flags", wintypes.DWORD)]

    class OplockReply(ctypes.Structure):
        _fields_ = [("version", wintypes.WORD), ("length", wintypes.WORD),
                    ("original_level", wintypes.DWORD), ("new_level", wintypes.DWORD),
                    ("flags", wintypes.DWORD), ("access", wintypes.DWORD),
                    ("share", wintypes.WORD)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                     ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.HANDLE)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.GetFileInformationByHandle.argtypes = (wintypes.HANDLE, ctypes.c_void_p)
    kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    kernel32.GetFileInformationByHandleEx.argtypes = (wintypes.HANDLE, ctypes.c_int,
                                                      ctypes.c_void_p, wintypes.DWORD)
    kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
    kernel32.CreateIoCompletionPort.argtypes = (wintypes.HANDLE, wintypes.HANDLE,
                                                ctypes.c_size_t, wintypes.DWORD)
    kernel32.CreateIoCompletionPort.restype = wintypes.HANDLE
    kernel32.DeviceIoControl.argtypes = (wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p,
                                         wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                         ctypes.c_void_p, ctypes.c_void_p)
    kernel32.DeviceIoControl.restype = wintypes.BOOL
    kernel32.GetQueuedCompletionStatus.argtypes = (
        wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_void_p), wintypes.DWORD)
    kernel32.GetQueuedCompletionStatus.restype = wintypes.BOOL
    kernel32.CancelIoEx.argtypes = (wintypes.HANDLE, ctypes.c_void_p)
    kernel32.CancelIoEx.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    class Win:
        pass
    win = Win()
    win.kernel32 = kernel32
    win.ByHandle, win.Basic, win.Overlapped = ByHandle, Basic, Overlapped
    win.OplockRequest, win.OplockReply = OplockRequest, OplockReply
    win.DWORD = wintypes.DWORD
    return win


class _ReadOplock:
    """A read-caching oplock: Windows' notice that another handle changed bytes.

    Windows breaks it, before the conflicting call proceeds, when any other
    handle writes, appends, truncates or extends the file, maps it for
    writing or opens it to overwrite, including a handle opened before
    the request. It is refused while a writable mapping exists. Reads,
    stats, renames and deletes leave it; renames and deletes move the
    path or the change time, which the identity holds. The handle asks only
    for attributes, so it takes no part in sharing checks, but an open handle
    refuses a rename of any folder above the file and a MoveFileEx
    replacement of the file itself; ``fetch`` closes these when its run lets
    go of the output folder.

    The break is read from a private completion port. Its packet is in the
    port by the time the conflicting call returns, while
    ``OVERLAPPED.Internal`` is written only when the packet is dequeued,
    so it cannot be the signal.
    """

    def __init__(self, path):
        self._lock = threading.Lock()
        self._handle = None
        self._port = None
        self._buffers = None
        self._pending = False
        self._intact = False
        win = _windows()
        handle = win.kernel32.CreateFileW(os.fsdecode(os.fspath(path)), _FILE_READ_ATTRIBUTES,
                                          _SHARE_ALL, None, _OPEN_EXISTING,
                                          _FILE_FLAG_OVERLAPPED, None)
        if handle == _INVALID_HANDLE_VALUE:
            return
        self._handle = handle
        self._port = win.kernel32.CreateIoCompletionPort(handle, None, 0, 1)
        if not self._port:
            self._port = None
            self._release()
            return
        request = win.OplockRequest(1, ctypes.sizeof(win.OplockRequest),
                                    _OPLOCK_LEVEL_CACHE_READ,
                                    _REQUEST_OPLOCK_INPUT_FLAG_REQUEST)
        reply = win.OplockReply(1, ctypes.sizeof(win.OplockReply))
        overlapped = win.Overlapped()
        self._buffers = (request, reply, overlapped)
        if win.kernel32.DeviceIoControl(handle, _FSCTL_REQUEST_OPLOCK, ctypes.byref(request),
                                        ctypes.sizeof(request), ctypes.byref(reply),
                                        ctypes.sizeof(reply), None, ctypes.byref(overlapped)):
            self._pending = True  # Completed at once: queued, but no oplock held.
            self._release()
            return
        if ctypes.get_last_error() != _ERROR_IO_PENDING:
            self._release()  # Not granted; a synchronous refusal queues nothing.
            return
        self._pending = True
        self._intact = True

    def holding(self):
        with self._lock:
            return self._handle is not None

    def intact(self):
        with self._lock:
            if self._intact and self._take_packet(0):
                self._intact = False
                self._release()
            return self._intact

    def close(self):
        with self._lock:
            self._intact = False
            self._release()

    def _take_packet(self, timeout_ms):
        win = _windows()
        count = win.DWORD()
        key = ctypes.c_size_t()
        overlapped = ctypes.c_void_p()
        win.kernel32.GetQueuedCompletionStatus(self._port, ctypes.byref(count),
                                               ctypes.byref(key), ctypes.byref(overlapped),
                                               timeout_ms)
        if not overlapped.value:
            return False  # The wait timed out: nothing has completed.
        self._pending = False
        return True

    def _release(self):
        kernel32 = _windows().kernel32
        if self._handle is not None:
            if self._pending:
                kernel32.CancelIoEx(self._handle, ctypes.byref(self._buffers[2]))
            kernel32.CloseHandle(self._handle)
            self._handle = None
        if self._port is not None:
            if self._pending:
                self._take_packet(5000)
            if self._pending:
                _UNSETTLED.append(self._buffers)
            kernel32.CloseHandle(self._port)
            self._port = None
