"""Sealed native-dtype RAM transport for retained forecast fields on Linux."""
from __future__ import annotations

from array import array
from contextlib import contextmanager
import json
import math
import mmap
import os
from pathlib import Path
import socket
import struct
import threading

import numpy as np

SCHEMA = "gpuwm.da.vts-ram/v1"
MAX_HEADER = 16 * 1024 * 1024


class RetentionError(RuntimeError):
    pass


def _fcntl():
    try:
        import fcntl
    except ImportError as error:
        raise RetentionError("VTSM packed retention needs Linux sealed anonymous RAM; ordinary state files would duplicate the forecast roster") from error
    return fcntl


#: The Linux memfd sealing ABI (include/uapi/linux/fcntl.h and
#: include/uapi/linux/memfd.h).  Python's ``fcntl`` and ``os`` publish these
#: names only when the interpreter was compiled against headers that define
#: them: the uv-managed CPython 3.13 builds lack ``fcntl.F_SEAL_*`` and
#: ``F_ADD_SEALS``/``F_GET_SEALS`` although the kernel accepts them, and a
#: release venv is that kind of interpreter.  The kernel never renumbers its
#: ABI, so a module that omits a name gets the kernel's number; a module that
#: has the name is read first.
_LINUX_SEALING_ABI = {"F_ADD_SEALS": 1033, "F_GET_SEALS": 1034,
                      "F_SEAL_SEAL": 0x0001, "F_SEAL_SHRINK": 0x0002,
                      "F_SEAL_GROW": 0x0004, "F_SEAL_WRITE": 0x0008}
_LINUX_MEMFD_ABI = {"MFD_CLOEXEC": 0x0001, "MFD_ALLOW_SEALING": 0x0002}


def _sealing():
    """``(fcntl, add, get, seals, memfd_flags)`` from the module or the ABI."""
    fcntl = _fcntl()
    if not hasattr(os, "memfd_create"):
        raise RetentionError("VTSM packed retention needs Linux memfd_create; ordinary state files would duplicate the forecast roster")
    value = {name: getattr(fcntl, name, number) for name, number in _LINUX_SEALING_ABI.items()}
    flags = {name: getattr(os, name, number) for name, number in _LINUX_MEMFD_ABI.items()}
    seals = value["F_SEAL_SEAL"] | value["F_SEAL_SHRINK"] | value["F_SEAL_GROW"] | value["F_SEAL_WRITE"]
    return (fcntl, value["F_ADD_SEALS"], value["F_GET_SEALS"], seals,
            flags["MFD_CLOEXEC"] | flags["MFD_ALLOW_SEALING"])


def _key(value):
    if not isinstance(value, str) or not value or len(value) > 255 or "\0" in value:
        raise RetentionError("retained forecast keys must be bounded nonempty strings")
    return value


def _exact(connection, count):
    data = bytearray()
    while len(data) < count:
        part = connection.recv(count-len(data))
        if not part:
            raise RetentionError("retained forecast IPC ended before its complete message")
        data.extend(part)
    return bytes(data)


def _send(connection, record):
    data = json.dumps(record, separators=(",", ":"), allow_nan=False).encode()
    if not 0 < len(data) <= MAX_HEADER:
        raise RetentionError("retained forecast metadata exceeds its bounded header")
    connection.sendall(struct.pack("!I", len(data))+data)


def _receive(connection):
    count, = struct.unpack("!I", _exact(connection, 4))
    if not 0 < count <= MAX_HEADER:
        raise RetentionError("retained forecast metadata length is invalid")
    result = json.loads(_exact(connection, count))
    if not isinstance(result, dict):
        raise RetentionError("retained forecast metadata must be an object")
    return result


def _send_fd(connection, fd):
    if connection.sendmsg([b"F"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array("i", [fd]))]) != 1:
        raise RetentionError("retained forecast descriptor transfer was incomplete")


def _receive_fd(connection):
    marker, control, flags, _ = connection.recvmsg(1, socket.CMSG_SPACE(4*array("i").itemsize),
        getattr(socket, "MSG_CMSG_CLOEXEC", 0))
    values = []
    for level, kind, data in control:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            item = array("i"); item.frombytes(data[:len(data)//item.itemsize*item.itemsize]); values.extend(item)
    if marker != b"F" or len(values) != 1 or flags & socket.MSG_CTRUNC:
        for fd in values:
            os.close(fd)
        raise RetentionError("retained forecast requires exactly one sealed descriptor")
    os.set_inheritable(values[0], False)
    return values[0]


@contextmanager
def _connection(path):
    stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stream.settimeout(60.)
    try:
        stream.connect(os.fspath(path)); yield stream
    finally:
        stream.close()


def _answer(stream):
    result = _receive(stream)
    if result.get("ok") is not True:
        raise RetentionError(result.get("reason", "retained forecast IPC refused its input"))
    return result


def put(path, key, fields, metadata):
    """Copy actual native bytes once, seal them, transfer their descriptor."""
    _key(key); fcntl, add_seals, _get, seals, memfd_flags = _sealing()
    if not fields:
        raise RetentionError("a retained forecast cannot have no fields")
    arrays = {}; entries = []; offset = 0
    for name, value in fields.items():
        _key(name)
        value = np.ascontiguousarray(value)
        if value.dtype.kind not in "biuf" or value.dtype.hasobject:
            raise RetentionError("retained forecasts require real native numeric dtypes")
        arrays[name] = value
        entries.append({"name": name, "dtype": value.dtype.str, "shape": list(value.shape),
                        "offset": offset, "bytes": value.nbytes})
        offset = (offset+value.nbytes+7)//8*8
    record = {"schema": SCHEMA, "bytes": max(1, offset), "fields": entries, "metadata": metadata}
    fd = os.memfd_create("gpuwm-vts", memfd_flags)
    try:
        os.ftruncate(fd, record["bytes"])
        with mmap.mmap(fd, record["bytes"], access=mmap.ACCESS_WRITE) as storage:
            for item in entries:
                storage.seek(item["offset"]); storage.write(arrays[item["name"]].tobytes(order="C"))
        try:
            fcntl.fcntl(fd, add_seals, seals)
        except OSError as error:
            raise RetentionError("the kernel refused to seal the retained forecast; unsealed RAM could be rewritten or resized after the analysis validated it") from error
        with _connection(path) as stream:
            _send(stream, {"op": "put", "key": key, "record": record}); _send_fd(stream, fd)
            return _answer(stream)["record"]
    finally:
        os.close(fd)


def _validate(fd, record):
    fcntl, _add, get_seals, seals, _flags = _sealing()
    if (record.get("schema") != SCHEMA or type(record.get("bytes")) is not int
            or record["bytes"] < 1 or os.fstat(fd).st_size != record["bytes"]
            or fcntl.fcntl(fd, get_seals) & seals != seals
            or not os.readlink(f"/proc/self/fd/{fd}").startswith("/memfd:gpuwm-vts")):
        raise RetentionError("retained forecast must be sealed anonymous RAM with its declared length")
    if not isinstance(record.get("fields"), list) or not record["fields"]:
        raise RetentionError("retained forecast field inventory is empty")
    names = set(); end = 0
    for item in record["fields"]:
        name = _key(item["name"]); shape = item["shape"]; dtype = np.dtype(item["dtype"])
        if (name in names or not isinstance(shape, list) or len(shape) > 32
                or any(type(n) is not int or n < 0 for n in shape) or dtype.kind not in "biuf"):
            raise RetentionError("retained forecast dtype/shape/member inventory is invalid")
        size = math.prod(shape)*dtype.itemsize; offset = item["offset"]
        if type(offset) is not int or offset % max(1, dtype.alignment) or offset < end or item["bytes"] != size or offset+size > record["bytes"]:
            raise RetentionError("retained forecast field range disagrees with its sealed storage")
        names.add(name); end = offset+size


@contextmanager
def get(path, key):
    with _connection(path) as stream:
        _send(stream, {"op": "get", "key": _key(key)})
        record = _answer(stream)["record"]; fd = _receive_fd(stream)
    try:
        _validate(fd, record)
        # A sealed O_RDWR descriptor must not request a shared MAYWRITE map.
        storage = mmap.mmap(fd, record["bytes"], flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ)
    finally:
        os.close(fd)
    fields = {item["name"]: np.ndarray(tuple(item["shape"]), dtype=np.dtype(item["dtype"]),
        buffer=storage, offset=item["offset"]) for item in record["fields"]}
    try:
        yield fields, record
    finally:
        fields.clear()
        try:
            storage.close()
        except BufferError as error:
            raise RetentionError("a retained forecast view escaped its borrow lease; release the pooled roster before closing it") from error


def drop(path, key):
    with _connection(path) as stream:
        _send(stream, {"op": "drop", "key": _key(key)}); return _answer(stream)["dropped"]


class RetainedRam:
    """One bounded server owned by the cycle, never a weather-file cache."""
    def __init__(self, path, *, max_bytes):
        if max_bytes <= 0:
            raise RetentionError("retained forecast RAM needs a positive admitted byte budget")
        self.path = Path(path); self.max_bytes = int(max_bytes)
        self.records = {}; self.lock = threading.RLock(); self.stop = threading.Event()
        self.listener = None; self.thread = None

    def _handle(self, stream):
        fd = None
        try:
            stream.settimeout(60.); request = _receive(stream); key = _key(request["key"])
            with self.lock:
                if request["op"] == "put":
                    fd = _receive_fd(stream); record = request["record"]; _validate(fd, record)
                    if key in self.records:
                        raise RetentionError("a retained slot already exists; replacing it could mix forecast origins")
                    if sum(value[1]["bytes"] for value in self.records.values())+record["bytes"] > self.max_bytes:
                        raise RetentionError("retained forecast RAM admission would be exceeded")
                    self.records[key] = (fd, record); fd = None
                    _send(stream, {"ok": True, "record": record})
                elif request["op"] == "get":
                    if key not in self.records:
                        raise RetentionError("a required actual forecast slot is missing")
                    stored, record = self.records[key]; fd = os.dup(stored)
                    _send(stream, {"ok": True, "record": record}); _send_fd(stream, fd)
                elif request["op"] == "drop":
                    value = self.records.pop(key, None)
                    if value is not None:
                        os.close(value[0])
                    _send(stream, {"ok": True, "dropped": value is not None})
                else:
                    raise RetentionError("unknown retained forecast IPC operation")
        except Exception as error:
            try:
                _send(stream, {"ok": False, "reason": str(error)})
            except OSError:
                pass
        finally:
            if fd is not None:
                os.close(fd)
            stream.close()

    def _serve(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="vts-ram") as pool:
            while not self.stop.is_set():
                try:
                    stream, _ = self.listener.accept()
                except socket.timeout:
                    continue
                pool.submit(self._handle, stream)

    def start(self):
        _fcntl()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise RetentionError("retained forecast socket exists; another cycle's server must not be replaced")
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(self.path)); os.chmod(self.path, 0o600)
        self.listener.listen(8); self.listener.settimeout(.2)
        self.thread = threading.Thread(target=self._serve, name="vts-ram", daemon=True); self.thread.start()
        return self

    def clear(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        if self.listener is not None:
            self.listener.close(); self.path.unlink(missing_ok=True)
        with self.lock:
            for fd, _ in self.records.values():
                os.close(fd)
            self.records.clear()
