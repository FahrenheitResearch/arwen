"""Record the existing compiled root-ozone provider and its live upload."""
from pathlib import Path
import ctypes as C
import hashlib
import numpy as np

SOURCE_SHA256 = "8594158cb44e65a2bbd25fa861c3ede4ff379f45251a4e45cb03ac2d9ebcd69c"


class OzoneArgs(C.Structure):
    _fields_ = [("function", C.c_void_p), ("values", C.c_void_p),
                ("nvalues", C.c_size_t), ("latitude", C.c_void_p),
                ("nlatitude", C.c_size_t), ("ozone", C.c_void_p),
                ("levels", C.c_size_t), ("months", C.c_size_t),
                ("nm", C.c_size_t), ("np", C.c_size_t),
                ("fact1", C.c_float), ("fact2", C.c_float),
                ("workers", C.c_size_t), ("out", C.c_void_p)]


class OzoneUpload(C.Structure):
    _fields_ = [("base", OzoneArgs), ("upload", C.c_void_p)]


class RecordedOzone:
    def __init__(self, plan, helper, ozone_module, *, cupy=None):
        if hashlib.sha256(Path(ozone_module.__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
            raise RuntimeError("changed original root-ozone provider source")
        if cupy is None:
            import cupy
        self.plan, self.helper, self.ozone, self.cp = plan, helper, ozone_module, cupy
        self.arguments = None
        self.uploaded = False

    def compute(self, julday, julian, xlat, climo, *, out=None):
        if self.arguments is not None:
            raise RuntimeError("recorded ozone provider supports one root call per template")
        # Keep the public validation, buffer reuse and original fallback behavior
        # during capture. A missing compiled route is an explicit capture refusal.
        result = self.ozone.ozn_latitude_time_int(julday, julian, xlat, climo, out=out)
        entries = self.ozone._native_ozone_entries()
        inputs = self.ozone._native_climatology(climo)
        if entries is None or inputs is None:
            raise RuntimeError("recorded root ozone requires the ordinary compiled provider")
        from gpuwm.core import portable_math
        values = self.ozone._native_xlat(entries, np.asarray(xlat, dtype=np.float32))
        latitude, ozone = inputs
        if result.dtype != np.float32 or not result.flags.c_contiguous or result.size != values.size * self.ozone.LEVSIZ:
            raise RuntimeError("unexpected compiled root-ozone output layout")
        nm, nn, fact1, fact2 = self.ozone._ozone_time_weights(julian)
        base = OzoneArgs(C.cast(entries[2], C.c_void_p), values.ctypes.data,
                         values.size, latitude.ctypes.data, latitude.size,
                         ozone.ctypes.data, self.ozone.LEVSIZ, 12, nm, nn,
                         fact1, fact2, portable_math._workers(None), result.ctypes.data)
        self.arguments = OzoneUpload(base, base.out)
        self.output = result
        self.library = portable_math._load()
        self.native = self.plan.native_host(
            self.helper, 'host_ozone_upload', self.arguments,
            self.library, values, latitude, ozone, result)
        return result

    def upload(self, values):
        if self.arguments is None or self.uploaded or values is not self.output:
            raise RuntimeError("unexpected recorded root-ozone upload")
        with self.plan.native_upload() as operation:
            device = self.cp.asarray(values)
        address, size = self.plan.host_buffer(operation[0])
        if size != values.nbytes:
            raise RuntimeError("root-ozone H2D payload has unexpected size")
        self.arguments.upload = address
        self.uploaded = True
        self.plan.resources.extend((self, device))
        return device

    def bind(self, frame, field):
        if not self.uploaded:
            raise RuntimeError("recorded root-ozone upload is missing")
        setattr(frame, field, self.arguments)
        self.plan.bind(self.native, 0, type(frame), field)

    def calendar(self, frame, field, julian):
        nm, nn, fact1, fact2 = self.ozone._ozone_time_weights(julian)
        target = getattr(frame, field).base
        target.nm, target.np, target.fact1, target.fact2 = nm, nn, fact1, fact2
