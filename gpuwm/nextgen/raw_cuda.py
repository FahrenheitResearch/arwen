"""Fresh metadata-only CUDA staging through nextgen_raw.h. Payload stays Rust/CUDA.

SPDX-License-Identifier: Apache-2.0
Only call at setup/export boundaries, never in the resident numerical step.
"""
import ctypes as ct
from pathlib import Path


class _Read(ct.Structure):
    _fields_ = [("path", ct.c_char_p), ("sha256", ct.c_char_p),
                ("destination", ct.c_void_p), ("bytes", ct.c_size_t)]


class _Write(ct.Structure):
    _fields_ = [("path", ct.c_char_p), ("source", ct.c_void_p), ("bytes", ct.c_size_t)]


class RawCuda:
    def __init__(self, library):
        self._lib = ct.CDLL(str(library))
        for name, item in (("read", _Read), ("write", _Write)):
            fn = getattr(self._lib, f"nextgen_raw_{name}_many")
            fn.argtypes = [ct.POINTER(item), ct.c_size_t, ct.c_void_p, ct.c_size_t]
            fn.restype = ct.c_int
        self._lib.nextgen_raw_sha256.argtypes = [ct.c_char_p, ct.c_void_p, ct.c_void_p, ct.c_size_t]
        self._lib.nextgen_raw_sha256.restype = ct.c_int

    def _call(self, name, *args):
        error = ct.create_string_buffer(2048)
        if getattr(self._lib, name)(*args, error, len(error)):
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))

    @staticmethod
    def _buffer(nbytes):
        import cupy as cp
        pin = cp.cuda.alloc_pinned_memory(max(nbytes, 1))
        return pin, ct.addressof(ct.c_char.from_buffer(pin))

    def load_many(self, specs, *, stream=None):
        """specs: name -> {path,sha256,dtype,shape,order}; paths are explicit."""
        import cupy as cp
        stream = stream if stream is not None else cp.cuda.get_current_stream()
        arrays, pins, rows = {}, [], []
        with stream:
            for name, spec in specs.items():
                if spec["dtype"] not in ("f32", "f64", "i32", "i64", "u8"):
                    raise ValueError("unsupported packed raw dtype")
                if spec["order"] not in ("c", "fortran"):
                    raise ValueError("order must be c or fortran")
                dtype = {"f32": "float32", "f64": "float64", "i32": "int32",
                         "i64": "int64", "u8": "uint8"}[spec["dtype"]]
                array = cp.empty(tuple(spec["shape"]), dtype=dtype,
                                 order="C" if spec["order"] == "c" else "F")
                pin, pointer = self._buffer(array.nbytes)
                arrays[name] = array; pins.append(pin)
                rows.append(_Read(str(Path(spec["path"])).encode(), spec["sha256"].encode(), pointer, array.nbytes))
            self._call("nextgen_raw_read_many", (_Read * len(rows))(*rows), len(rows))
            for array, row in zip(arrays.values(), rows):
                cp.cuda.runtime.memcpyAsync(array.data.ptr, row.destination, array.nbytes,
                                           cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
            stream.synchronize()  # pins remain alive until every setup copy completes
        return arrays

    def store_many(self, arrays_and_paths, *, stream=None):
        """Export [(CUDA array,new raw path),...] without conversion or host decode."""
        import cupy as cp
        stream = stream if stream is not None else cp.cuda.get_current_stream()
        pins, rows = [], []
        with stream:
            for array, path in arrays_and_paths:
                if not (array.flags.c_contiguous or array.flags.f_contiguous):
                    raise ValueError("raw export requires packed contiguous CUDA storage")
                pin, pointer = self._buffer(array.nbytes); pins.append(pin)
                rows.append(_Write(str(Path(path)).encode(), pointer, array.nbytes))
                cp.cuda.runtime.memcpyAsync(pointer, array.data.ptr, array.nbytes,
                                           cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
            stream.synchronize()  # offline export boundary, before Rust reads pinned bytes
            self._call("nextgen_raw_write_many", (_Write * len(rows))(*rows), len(rows))

    def sha256(self, path):
        digest = ct.create_string_buffer(65)
        self._call("nextgen_raw_sha256", str(Path(path)).encode(), digest)
        return digest.value.decode("ascii")
