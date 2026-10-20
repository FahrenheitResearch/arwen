"""Fixed-capacity daylight transport for a recorded radiation call.

The caller still owns the actual radiation adapter and publication transaction.
Padding is never scientific output: every scatter is bounded by the live count.
"""
import ctypes as C
from contextlib import contextmanager
import hashlib
from pathlib import Path
import numpy as np

SOURCE_SHA256 = "48c49d79d56e94507ec006488f2bc702fdd76b7223c58c2b8b647c048d9daa40"


class Daylight(C.Structure):
    _fields_ = [("coszen", C.c_void_p), ("ncols", C.c_size_t),
                ("capacity", C.c_size_t), ("workers", C.c_size_t),
                ("day", C.c_void_p), ("night", C.c_void_p),
                ("counts", C.c_void_p), ("work", C.c_void_p)]


class DaylightUpload(C.Structure):
    _fields_ = [("base", Daylight), ("upload_day", C.c_void_p),
                ("upload_counts", C.c_void_p)]


def masked_source(source_path):
    """Keep the original SW output body byte-for-byte except its write guard."""
    raw = Path(source_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("changed original radiation output kernel source")
    source = raw.decode()
    prefix = source[source.index("#define RLA_AD"):source.index("// Live-device proof:")]
    start = source.index('extern "C" __global__ void rla_sw_out_grid(')
    end = source.index("// Driver SWDOWN:", start)
    body = source[start:end].replace("rla_sw_out_grid(", "rla_sw_out_grid_counted(", 1)
    original = "float* __restrict__ gsw)"
    assert body.count(original) == 1
    body = body.replace(original, "float* __restrict__ gsw, const unsigned long long* counts, unsigned long long offset)", 1)
    original = "    int j = (int)(i % nc);"
    assert body.count(original) == 1
    body = body.replace(original, original + "\n    if (offset + (unsigned long long)j >= counts[0]) return;", 1)
    return prefix + body + r'''
extern "C" __global__ void day_coszen(
    long long n, unsigned long long offset, const unsigned long long* count,
    const long long* indices, const float* source, float* destination)
{
    long long j=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(j<n) destination[j]=(offset+(unsigned long long)j<count[0])?source[indices[j]]:1.0f;
}
extern "C" __global__ void day_scatter(
    long long n, unsigned long long offset, const unsigned long long* count,
    const long long* indices, const float* source, long long stride, float* destination)
{
    long long j=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(j<n && offset+(unsigned long long)j<count[0])destination[indices[j]]=source[j*stride];
}
'''


class RecordedDaylight:
    """One native CPU producer and two live uploads, no capture-time selection."""
    def __init__(self, plan, helper, deferred, coszen_device, *, chunk,
                 source_path, bit=2, workers=4, cupy=None):
        if cupy is None:
            import cupy
        self.cp, self.plan, self.deferred, self.bit = cupy, plan, deferred, bit
        if not isinstance(chunk, int) or chunk < 1 or not 1 <= workers <= 64 or not 0 <= bit < 64:
            raise ValueError("invalid daylight capacity/worker/predicate configuration")
        if np.dtype(coszen_device.dtype) != np.dtype('float32') or coszen_device.ndim != 1:
            raise ValueError("daylight requires a flat float32 cosine field")
        self.ncols, self.chunk = int(coszen_device.size), chunk
        if self.ncols < 1:
            raise ValueError("daylight requires at least one column")
        self.capacity = ((self.ncols + chunk - 1) // chunk) * chunk
        self.coszen_device = coszen_device
        self.day = np.empty(self.capacity, np.int64)
        self.night = np.empty(self.capacity, np.int64)
        self.counts = (C.c_uint64 * 2)()
        self.day_count = C.c_uint64.from_buffer(self.counts)
        self.work = (C.c_size_t * (2 * workers))()
        with plan.readback() as read:
            self.capture_coszen = cupy.asnumpy(coszen_device)
        address, size = plan.host_buffer(read[0])
        if size != self.ncols * 4:
            raise RuntimeError("cosine readback has unexpected byte size")
        self.live_coszen = np.ctypeslib.as_array((C.c_float * self.ncols).from_address(address))
        base = Daylight(address, self.ncols, self.capacity, workers,
                        self.day.ctypes.data, self.night.ctypes.data,
                        C.addressof(self.counts), C.addressof(self.work))
        self.arguments = DaylightUpload(base, base.day, base.counts)
        self.native = plan.native_host(helper, 'host_daylight_upload', self.arguments,
                                       self.day, self.night, self.counts, self.work, self.live_coszen)
        with plan.native_upload() as day_upload:
            self.day_device = cupy.asarray(self.day)
        with plan.native_upload() as count_upload:
            self.count_device = cupy.asarray(np.ctypeslib.as_array(self.counts))
        self.arguments.upload_day, day_bytes = plan.host_buffer(day_upload[0])
        self.arguments.upload_counts, count_bytes = plan.host_buffer(count_upload[0])
        if day_bytes != self.capacity * 8 or count_bytes != 16:
            raise RuntimeError("daylight upload has unexpected byte size")
        self.readback = read[0]
        # Compile using the original adapter's explicit non-FTZ NVRTC route.
        from gpuwm import nvrtc_ptx_cache
        code = masked_source(source_path)
        ptx, _ = nvrtc_ptx_cache.compile_using_nvrtc(
            code, ('-std=c++17', '--ftz=false'), None, 'arwen_recorded_daylight.cu')
        self.module = cupy.cuda.function.Module()
        self.module.load(ptx.encode() if isinstance(ptx, str) else ptx)
        self.kernels = {name: self.module.get_function(name) for name in
                        ('day_coszen', 'day_scatter', 'rla_sw_out_grid_counted')}
        plan.resources.extend((self, self.module, self.day_device, self.count_device))

    def bind(self, frame, field):
        """Bind the native producer to its retained H2D buffers after capture."""
        setattr(frame, field, self.arguments)
        self.plan.bind(self.native, 0, type(frame), field)

    def indices(self, offset):
        if offset < 0 or offset % self.chunk or offset >= self.capacity:
            raise ValueError("invalid captured daylight chunk offset")
        return self.day_device[offset:offset + self.chunk]

    @contextmanager
    def active(self, offset):
        self.indices(offset)
        self.plan.count_predicate(self.day_count, greater_than=offset, bit=self.bit)
        with self.plan.when(require=1 << self.bit), self.deferred.active_chunk(self.day_count, offset):
            yield

    def coszen(self, offset):
        indices = self.indices(offset)
        output = self.cp.empty(self.chunk, self.cp.float32)
        self.kernels['day_coszen'](((self.chunk + 127) // 128,), (128,),
            (np.int64(self.chunk), np.uint64(offset), self.count_device,
             indices, self.coszen_device, output))
        return output

    def scatter(self, destination, source, offset):
        indices = self.indices(offset)
        if source.dtype != np.float32 or destination.dtype != np.float32 or source.shape != (self.chunk,) or destination.shape != (self.ncols,) or not destination.flags.c_contiguous or source.strides[0] % 4:
            raise ValueError("unsupported daylight scatter layout")
        self.kernels['day_scatter'](((self.chunk + 127) // 128,), (128,),
            (np.int64(self.chunk), np.uint64(offset), self.count_device, indices,
             source, np.int64(source.strides[0] // 4), destination))

    def output(self, grid, block, arguments, offset):
        # The original arguments include the same indices view and arithmetic.
        self.kernels['rla_sw_out_grid_counted'](grid, block,
            (*arguments, self.count_device, np.uint64(offset)))

    def current_night_coszen(self):
        """Read only after successful submission, for original CPU metadata."""
        return self.live_coszen[self.night[:self.counts[1]]]
