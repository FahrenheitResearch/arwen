"""NextGen output orchestration. CUDA retains history; Rust processes bytes.

The history factory is a realization stage, prepared after the routing stage.
It participates in the driver's original inventory before stepping starts.
No withdrawn WRF-Hydro implementation, schema or fixture is reused.
"""
from __future__ import annotations

import ctypes as C
import json
import os
from pathlib import Path
import sys
import time
from types import MappingProxyType
from uuid import uuid4

_DTYPES = {"float32": "f32le", "float64": "f64le", "int8": "i8",
           "uint8": "u8", "int16": "i16le", "uint16": "u16le",
           "int32": "i32le", "uint32": "u32le", "int64": "i64le",
           "uint64": "u64le"}
_LIBRARY = None


class _Response(C.Structure):
    _fields_ = [("status", C.c_int32), ("json", C.c_void_p),
                ("json_len", C.c_size_t), ("payload", C.c_void_p),
                ("payload_len", C.c_size_t)]


def _library():
    global _LIBRARY
    if _LIBRARY is None:
        target = _library_path()
        library = C.CDLL(str(target))
        library.rw_nextgen_output_abi_version.restype = C.c_uint32
        if library.rw_nextgen_output_abi_version() != 1:
            raise RuntimeError("Rust coupled hydro output ABI mismatch")
        library.rw_nextgen_output_call.argtypes = [C.c_void_p, C.c_size_t,
                                                 C.c_void_p, C.c_size_t]
        library.rw_nextgen_output_call.restype = C.POINTER(_Response)
        library.rw_nextgen_output_free.argtypes = [C.POINTER(_Response)]
        library.rw_nextgen_output_free.restype = None
        _LIBRARY = library
    return _LIBRARY


def _library_path():
    from gpuwm import bridges
    filename = ("rw_nextgen_output.dll" if os.name == "nt" else
                "librw_nextgen_output.dylib" if sys.platform == "darwin" else
                "librw_nextgen_output.so")
    env = "GPUWM_NEXTGEN_OUTPUT_LIBRARY"
    def found():
        paths = bridges.artifact_candidates(env, filename, crate_relative="tools/nextgen_output")
        for path in paths:
            if path.is_file():
                return path
            if os.environ.get(env) and path == Path(os.environ[env]):
                raise FileNotFoundError(f"{env} names a missing file: {path}")
        return None
    target = found()
    if target is None and not os.environ.get(env) and bridges.fetch_on_demand("rw_nextgen_output"):
        target = found()
    if target is None:
        raise FileNotFoundError(f"coupled hydro output needs {filename} ({env})")
    return target


def _call(request, payload=None):
    encoded = json.dumps(request, allow_nan=False, separators=(",", ":")).encode()
    source = C.create_string_buffer(encoded)
    holder = None
    if payload is not None and len(payload):
        holder = (C.c_ubyte * len(payload)).from_buffer(payload)
    library = _library()
    pointer = library.rw_nextgen_output_call(source, len(encoded), holder,
                                           0 if payload is None else len(payload))
    if not pointer:
        raise RuntimeError("Rust NextGen output returned no response")
    try:
        response = pointer.contents
        metadata = json.loads(C.string_at(response.json, response.json_len))
        if response.status:
            raise RuntimeError(metadata["error"])
        # Opaque transport only: Python never decodes numerical field words.
        data = bytearray(C.string_at(response.payload, response.payload_len))
        return metadata, data
    finally:
        library.rw_nextgen_output_free(pointer)


_HISTORY = r'''
extern "C" __global__ void capture(
    const REAL* q, const REAL* v, const REAL* d, const REAL* n,
    REAL* hq, REAL* hv, REAL* hd, REAL* hn,
    const unsigned long long* step, unsigned long long* overflow,
    unsigned long long reaches, unsigned long long capacity,
    unsigned long long cadence, int final) {
    const unsigned long long tick = *step + (final == 2 ? 0 : 1);
    if ((tick % cadence) && !final) return;
    const unsigned long long frame = (tick + cadence - 1) / cadence - 1;
    if (frame >= capacity) { if (!threadIdx.x && !blockIdx.x) *overflow=1; return; }
    const unsigned long long r = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= reaches) return;
    const unsigned long long dest = frame * reaches + r;
    hq[dest]=q[r];
    if (v) hv[dest]=v[r];
    if (d) hd[dest]=d[r];
    if (n) hn[dest]=n[r];
}
extern "C" __global__ void tick(unsigned long long* step) { ++*step; }
'''


class History:
    """Prepared GPU capture component; two bound launches per model step."""

    def __init__(self, *, settings, options, source, device_id):
        import cupy as cp
        import numpy as np
        self.device_id = int(device_id)
        self.interval = float(settings["interval_seconds"])
        self.cadence = int(options.get("cadence_steps", 1))
        self.capacity_steps = int(options["capacity_steps"])
        self.routing_clock = bool(options.get('routing_clock'))
        self.final_time_seconds = None
        if self.cadence < 1 or self.capacity_steps < (1 if self.routing_clock else self.cadence):
            raise ValueError("history capacity must contain at least one chosen output interval")
        self.capacity = ((self.capacity_steps + self.cadence - 1) // self.cadence
                         if self.routing_clock else self.capacity_steps // self.cadence)
        fields = source.reach_output()
        self.reaches = int(fields["reach_id"].size)
        flow = fields["discharge_m3_s"]
        if str(flow.dtype) not in ("float32", "float64") or self.reaches < 1:
            raise ValueError("history needs nonempty native-real routing output")
        self.dtype = str(flow.dtype)
        self.options = dict(options)
        arrays = {"step": cp.zeros((), dtype=cp.uint64),
                  "overflow": cp.zeros((), dtype=cp.uint64),
                  "reach_id": fields["reach_id"]}
        keys = (("flow", "discharge_m3_s"), ("velocity", "velocity_m_s"),
                ("depth", "depth_m"), ("nudge", "nudge_m3_s"))
        sources, targets = [], []
        for key, field in keys:
            value = fields.get(field)
            if value is None:
                sources.append(np.uintp(0))
                targets.append(np.uintp(0))
                continue
            if (str(value.dtype) != self.dtype or tuple(value.shape) != (self.reaches,)
                    or not value.flags.c_contiguous or value.device.id != self.device_id):
                raise ValueError(f"output field {field} differs in order, shape, card or precision")
            history = cp.zeros((self.capacity, self.reaches), dtype=value.dtype)
            arrays[key] = history
            sources.append(value)
            targets.append(history)
        ids = arrays["reach_id"]
        if (str(ids.dtype) != "int64" or tuple(ids.shape) != (self.reaches,)
                or not ids.flags.c_contiguous or ids.device.id != self.device_id):
            raise ValueError("output IDs must be contiguous int64 in source order on the same card")
        self.arrays = MappingProxyType(arrays)
        self.restart_identity = {"schema": "arwen.nextgen.gpu-history/v1",
                                 "source": source.restart_identity,
                                 "interval_seconds": self.interval,
                                 "cadence_steps": self.cadence,
                                 "capacity_steps": self.capacity_steps,
                                 "fields": [key for key, _ in keys if key in arrays],
                                 "origin": options["origin"],
                                 "types": options["types"],
                                 "nexus_ids": options.get("nexus_ids", []),
                                 "candidate_ids": options.get("candidate_ids", [])}
        code = _HISTORY.replace("REAL", "double" if self.dtype == "float64" else "float")
        module = cp.RawModule(code=code, options=("--std=c++11",))
        from gpuwm.certify.kernel_manifest import record_module
        record_module("nextgen_history", source=code, options=("--std=c++11",), module=module)
        self._copy = module.get_function("capture")
        self._tick = module.get_function("tick")
        self._grid = ((self.reaches + 255) // 256,)
        self._args = (*sources, *targets, arrays["step"], arrays["overflow"],
                      np.uint64(self.reaches), np.uint64(self.capacity), np.uint64(self.cadence))
        self._tick_args = (arrays["step"],)
        self.parallelism = "GPU history: one thread per reach; capture then scalar cursor launch"
        if self.routing_clock:
            self.restart_identity['routing_clock'] = True
        else:
            # Fixed-cadence histories have no additional host clock. Retain
            # their pre-clock checkpoint envelope and continuation identity.
            self.checkpoint_metadata = self.validate_checkpoint = self.restore_metadata = None

    def advance(self, dt_seconds, *, stream=None, final=False, time_seconds=None, recapture=False):
        import numpy as np
        if dt_seconds != self.interval:
            raise ValueError("history cadence differs from the routing interval")
        self._copy(self._grid, (256,), self._args + (np.int32(2 if recapture else final),), stream=stream)
        if not recapture:
            self._tick((1,), (1,), self._tick_args, stream=stream)
        if final:
            self.final_time_seconds = time_seconds

    def checkpoint_metadata(self):
        return {'final_time_seconds': self.final_time_seconds}

    def validate_checkpoint(self, metadata):
        import math
        time = metadata.get('final_time_seconds')
        if set(metadata) != {'final_time_seconds'} or (time is not None and
                (type(time) not in (float, int) or not math.isfinite(time) or time <= 0)):
            raise ValueError('history continuation needs its final capture clock')

    def restore_metadata(self, metadata):
        self.final_time_seconds = metadata['final_time_seconds']


def prepare_history(*, settings, options, state, physics, device_id, mode, components):
    """Table factory. Bind after the source routing/reservoir stages."""
    effective = dict(settings)
    if mode == "coupled":
        effective["interval_seconds"] = physics.bldt_seconds
    if "interval_seconds" in options:
        effective["interval_seconds"] = options["interval_seconds"]
    return History(settings=effective, options=options,
                   source=components[options["source"]], device_id=device_id)


def prepare_history_memory(*, settings, options, context, components):
    from .coupling_memory import history_memory
    return history_memory(settings=settings, options=options, context=context, components=components)


def _snapshot(arrays, stream):
    import cupy as cp
    roster, offsets, size = [], {}, 0
    for key, array in arrays.items():
        if str(array.dtype) not in _DTYPES:
            raise ValueError(f"resident checkpoint array {key} must be primitive")
        roster.append({"name": key, "dtype": _DTYPES[str(array.dtype)],
                       "shape": list(array.shape)})
        offsets[key] = size
        size += array.nbytes
    memory = cp.cuda.alloc_pinned_memory(max(1, size))
    address = C.addressof((C.c_ubyte * max(1, size)).from_buffer(memory))
    packed = []
    for key, array in arrays.items():
        if array.nbytes:
            # Components may expose a strided view alongside its backing slab.
            # Preserve every declared array in logical C order; CUDA packs views
            # only at this output boundary, without host numerical processing.
            with stream:
                source = array if array.flags.c_contiguous else cp.ascontiguousarray(array)
            packed.append(source)  # Keep temporary GPU storage alive through D2H.
            cp.cuda.runtime.memcpyAsync(address + offsets[key], source.data.ptr,
                                        array.nbytes, cp.cuda.runtime.memcpyDeviceToHost,
                                        stream.ptr)
    stream.synchronize()
    return roster, offsets, memoryview(memory).cast("B")[:size]


def restore_checkpoint(path, driver):
    """Validate the full Rust envelope before restoring existing CUDA carriers."""
    import cupy as cp
    import numpy as np
    metadata, data = _call({"operation": "checkpoint-read", "path": str(path)})
    reverse = {value: key for key, value in _DTYPES.items()}
    incoming, offset = {}, 0
    for array in metadata["arrays"]:
        value = np.ndarray(tuple(array["shape"]), dtype=reverse[array["dtype"]],
                           buffer=data, offset=offset)
        incoming[array["name"]] = value
        offset += value.nbytes
    driver.validate_checkpoint(metadata["metadata"], incoming)
    # Every envelope, digest, identity, field, shape and dtype has now
    # validated. Mutate only existing device allocations on the same stream.
    with cp.cuda.Device(driver.device_id):
        stream = cp.cuda.get_current_stream()
        packed = []
        for key, target in driver.arrays.items():
            value = incoming[key]
            if value.nbytes:
                if target.flags.c_contiguous:
                    cp.cuda.runtime.memcpyAsync(target.data.ptr, value.ctypes.data, value.nbytes,
                                                cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
                else:
                    with stream:
                        source = cp.asarray(value)
                        cp.copyto(target, source)
                    packed.append(source)
        stream.synchronize()
    driver.restore_metadata(metadata["metadata"])
    return metadata


def _select_output_directory(path):
    """Keep previous publications and choose a fresh sibling for another run."""
    if (not path.is_symlink()
            and (not path.exists()
                 or (path.is_dir() and next(path.iterdir(), None) is None))):
        return path
    while True:
        candidate = path.absolute().parent / ("hydro-run-" + uuid4().hex)
        if not candidate.exists() and not candidate.is_symlink():
            return candidate


def _report_output_directory(path):
    print(f"Hydro output: {path} (previous files preserved)", file=sys.stderr)


def _history_steps(history):
    """Read checkpointed capture counters only at the publication boundary."""
    import cupy as cp
    with cp.cuda.Device(history.arrays["step"].device.id):
        steps = int(history.arrays["step"].item())
        overflow = int(history.arrays["overflow"].item())
    if overflow:
        raise ValueError("history buffer exhausted; increase realization capacity_steps before running")
    return steps


class Writer:
    def __init__(self, *, settings, driver, outdir):
        self.requested_outdir = Path(outdir)
        self.outdir = _select_output_directory(self.requested_outdir)
        self.settings = settings
        self.histories = [(name, component) for name, component in driver.components.items()
                          if isinstance(component, History)]
        if not self.histories:
            raise ValueError("realization needs a gpuwm.hydro.output:prepare_history stage for a time series")
        _library()
        if self.outdir != self.requested_outdir:
            _report_output_directory(self.outdir)

    def restore(self, path, driver):
        return restore_checkpoint(path, driver)

    def finish(self, driver, receipt):
        if getattr(driver, 'routing_clock', None):
            driver.finish_clock()
        # A caller may have populated the path since preparation. Recover again
        # before allocating host buffers or publishing any checkpoint/product.
        directory = _select_output_directory(self.outdir)
        if directory != self.outdir:
            self.outdir = directory
            _report_output_directory(self.outdir)
        import cupy as cp
        start = time.perf_counter()
        # A scheduled stage can capture several times per parent macrostep.
        # Validate every history before publishing even the restart envelope.
        completed = []
        for name, history in self.histories:
            steps = _history_steps(history)
            frames = (steps + (history.cadence - 1 if history.final_time_seconds is not None else 0)) // history.cadence
            if frames > history.capacity:
                raise ValueError("history buffer exhausted; increase realization capacity_steps before running")
            completed.append((name, history, frames))
        self.outdir.mkdir(parents=True, exist_ok=True)
        with cp.cuda.Device(driver.device_id):
            stream = cp.cuda.get_current_stream()
            arrays, offsets, payload = _snapshot(driver.arrays, stream)
        snapshot_ms = (time.perf_counter() - start) * 1000
        checkpoint, _ = _call({"operation": "checkpoint-write",
                              "path": str(self.outdir / "restart.nc"),
                              "metadata": driver.checkpoint_metadata(), "arrays": arrays}, payload)
        products = []
        for name, history, frames in completed:
            if frames == 0:
                continue
            base = name + "/"
            options = history.options
            plan = {"output_directory": str(self.outdir / name), "origin": options["origin"],
                    "interval_seconds": int(history.interval),
                    "cadence_steps": history.cadence, "frames": frames,
                    "final_time_seconds": history.final_time_seconds,
                    "capacity": history.capacity, "reaches": history.reaches,
                    "dtype": _DTYPES[history.dtype], "ids_offset": offsets[base + "reach_id"],
                    "flow_offset": offsets[base + "flow"],
                    "velocity_offset": offsets.get(base + "velocity"),
                    "depth_offset": offsets.get(base + "depth"),
                    "nudge_offset": offsets.get(base + "nudge"),
                    "types": options["types"], "nexus_ids": options.get("nexus_ids", []),
                    "candidate_ids": options.get("candidate_ids", []),
                    "provenance": options.get("provenance", {}),
                    "parquet": options.get("parquet"),
                    "renderer": options.get("renderer"),
                    "formats": options.get("formats", ["analysis_csv", "renderer_request"])}
            product, _ = _call({"operation": "series-write", "plan": plan}, payload)
            products.append(product)
        receipt.update({"output_directory": str(self.outdir),
                        "requested_output_directory": str(self.requested_outdir),
                        "output_snapshot_ms": snapshot_ms, "checkpoint": checkpoint,
                        "products": products, "output_total_ms": (time.perf_counter()-start)*1000,
                        "output_scope": "GPU capture launches per step; D2H and Rust I/O only at finish"})
        (self.outdir / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        return receipt


def prepare(*, settings, driver, outdir):
    return Writer(settings=settings, driver=driver, outdir=outdir)
