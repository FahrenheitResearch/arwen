"""Offline binding to the shared NextGen Rust host-byte staging C ABI."""
import ctypes as c
import os


class RawCuda:
    def __init__(self, path):
        self.lib = c.CDLL(os.fspath(path))
        self.lib.nextgen_raw_load.argtypes = [c.c_char_p,c.c_char_p,c.c_void_p,c.c_size_t,c.c_void_p,c.c_size_t]
        self.lib.nextgen_raw_store.argtypes = [c.c_char_p,c.c_void_p,c.c_size_t,c.c_void_p,c.c_size_t]
        self.lib.nextgen_raw_sha256.argtypes = [c.c_char_p,c.c_void_p,c.c_void_p,c.c_size_t]

    def _invoke(self, function, *args):
        error = c.create_string_buffer(4096)
        if function(*args,error,len(error)):
            raise RuntimeError(error.value.decode())

    def sha256(self, path):
        result = c.create_string_buffer(65)
        self._invoke(self.lib.nextgen_raw_sha256,os.fsencode(path),result)
        return result.value.decode()

    def load(self, spec, base):
        import cupy as cp
        from pathlib import Path
        a=cp.empty(spec["shape"],dtype={"f32":"float32","f64":"float64","i32":"int32","i64":"int64"}[spec["dtype"]],order="C")
        host=cp.cuda.alloc_pinned_memory(max(1,a.nbytes))
        self._invoke(self.lib.nextgen_raw_load,os.fsencode(Path(base)/spec["path"]),spec["sha256"].encode(),host.ptr,a.nbytes)
        cp.cuda.runtime.memcpy(a.data.ptr,host.ptr,a.nbytes,cp.cuda.runtime.memcpyHostToDevice)
        return a

    def store(self, array, path, *, order="c"):
        import cupy as cp
        if order!="c" or not array.flags.c_contiguous:
            raise ValueError("reservoir fixture exports must be C-contiguous")
        host=cp.cuda.alloc_pinned_memory(max(1,array.nbytes))
        cp.cuda.runtime.memcpy(host.ptr,array.data.ptr,array.nbytes,cp.cuda.runtime.memcpyDeviceToHost)
        self._invoke(self.lib.nextgen_raw_store,os.fsencode(path),host.ptr,array.nbytes)
